"""Generate Phase-1 training data for tp01 ("Teleporters").

tp01 (games/tp01/tp01.py) is a 5-level grid walk: ACTION1..4 move the magenta-
badged agent one cell up/down/left/right, a move into a wall or off the board is a
NO-OP, and every level puts a FULL wall barrier between the agent and the yellow
target -- so the only way across is a magenta portal. Portals come in BIDIRECTIONAL
pairs: step onto either tile and you are warped to its partner (and you then STAND
on that partner tile, which only warps you again if you step off and back on).

Reaching the target ends the level.

The hidden part: WHICH portal pairs with which
---------------------------------------------
The frame shows where the portals are, but not how they are wired. With one pair
(levels 1, 2, 4, 5) the wiring is forced -- two portals can only be partners --
but level 3 shows FOUR portals, and its two pairs cross, which no frame reveals.
So this solver is deliberately NOT privileged: it never reads
``game._portal_to_partner``. It keeps its own partial map and fills it in the way a
player does:

  * OBSERVE -- `drive` watches every action it sends to the engine (expert,
    exploration and burst alike): if the agent ends up somewhere other than the
    cell it stepped into, that step was a warp, and the pair is now known (both
    ways, since portals are symmetric).
  * DEDUCE -- portals pair up perfectly, so once all but two have known partners
    those last two must be each other's. That is what makes level 3 fully known
    after a single warp, and every one-pair level fully known before the first
    move.

Expert
------
A knowledge-conditioned exact distance field, recomputed per query (the boards are
at most 10x8, so a field is ~80 cells x 4 directions of plain Python):

  * ``dist_win``  -- moves to step onto the target, using only KNOWN transitions;
  * ``dist_probe`` -- moves to step onto a portal whose partner is still unknown,
    i.e. the cheapest information-gathering action.

``solve_from`` descends ``dist_win`` when a win is already reachable and
``dist_probe`` otherwise, so the plan is either "finish" or "go learn the wiring".
It is optimal given what the agent has seen -- and omniscient-optimal on every
level whose wiring is deducible (all but level 3, where a first probe may cost a
detour, exactly as it would for a player). Ties are broken at random, so repeated
episodes of one seed take different equally-good routes.

Recovery (supports_recovery = True, recovery_mode = "replan")
------------------------------------------------------------
Everything the expert needs is read off the LIVE board -- the agent's cell from its
sprite, the geometry from the level's sprites, the wiring from what has been
observed -- so it re-plans from wherever the exploration prefix or a perturbation
burst leaves the agent, including from the far side of a warp it did not intend.
tp01 is also fully REVERSIBLE (walking undoes walking; a warp is undone by
stepping off the destination portal and back onto it), so no detour can strand the
level: bursts are kept as recovery data rather than rolled back, and RESET
effectively never fires. ``optimal_set_from`` returns the whole co-optimal tie set
as the training target, at the cost of one extra field lookup.

`_burst_recovery_wins` is overridden because the base's version replays ONE plan
and demands a win: here a plan may legitimately end at a probe (an action whose
outcome is unknown by construction), so the override keeps re-planning as the dry
run learns, and discards the knowledge it gained afterwards -- a hypothetical
rollout must not teach the agent a wiring it never actually saw on camera.

Rotation: tp01 is an ``AugmentedGame``, so each level is displayed at a per-(seed,
level) rotation. Planning happens in GAME space and every emitted action is
converted to the SCREEN key the agent must press with
``inverse_remap_action_full``, which the game's own ``screen_action_to_game``
maps back.

Usage (run from the repo root):
    python solvers/generate_tp01_training.py --episodes 1000 \
        --out data/training_multi_level/tp01
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                                  # noqa: E402
from games.tp01.tp01 import Tp01                                   # noqa: E402
from solvers.base_solver import BaseSolver, _as_action              # noqa: E402
from utils.rotation import inverse_remap_action_full               # noqa: E402

# Game-space directions, index-aligned with ``_ACTIONS`` and with the ACTION1..4
# -> (dx, dy) mapping in ``Tp01.step``.
_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_ACTIONS = (GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4)
_ACTION_TO_DIR = {a: d for d, a in enumerate(_ACTIONS)}

#: Edge kinds returned by ``_Geom.succ``.
_MOVE, _WIN, _PROBE = "move", "win", "probe"


class _Geom:
    """One level's STATIC geometry in game space, plus the field machinery.

    Cells are flat ints ``c = y * gw + x``. ``nbr[d][c]`` is the cell one step in
    direction ``d``, or -1 when that step is off-board or into a wall (both are
    no-ops in ``Tp01.step``, and a wasted action is never optimal). Which portal
    leads where is NOT geometry -- it is knowledge, so it is passed in as the
    ``know`` map on every call.
    """

    def __init__(self, gw: int, gh: int, walls, portals, targets) -> None:
        self.gw, self.gh = gw, gh
        self.n = gw * gh
        wall_cells = {y * gw + x for (x, y) in walls}
        self.free = [c for c in range(self.n) if c not in wall_cells]
        self.portals = frozenset(y * gw + x for (x, y) in portals)
        self.targets = frozenset(y * gw + x for (x, y) in targets)
        self.nbr = [[-1] * self.n for _ in range(4)]
        for d, (dx, dy) in enumerate(_DELTAS):
            for y in range(gh):
                for x in range(gw):
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < gw and 0 <= ny < gh):
                        continue
                    c, t = y * gw + x, ny * gw + nx
                    if c in wall_cells or t in wall_cells:
                        continue
                    self.nbr[d][c] = t

    def cell(self, x: int, y: int) -> int:
        return y * self.gw + x

    # ── the engine transition, exactly (see ``Tp01.step``) ───────────────────
    def succ(self, c: int, d: int, know: dict):
        """``(kind, cell)`` for stepping from ``c`` in direction ``d``, or None
        when the step is a no-op (edge / wall).

        ``kind`` is ``_WIN`` when the step lands on the target (directly or via a
        warp), ``_PROBE`` when it steps onto a portal whose partner is not known
        yet -- the outcome is then unknown by construction, so the returned cell
        is meaningless (-1) -- and ``_MOVE`` otherwise. The warp is applied ONCE,
        exactly as the game does, so the destination portal is a resting cell.
        """
        e = self.nbr[d][c]
        if e < 0:
            return None
        if e in self.portals:
            partner = know.get(e)
            if partner is None:
                return _PROBE, -1
            e = partner
        return (_WIN if e in self.targets else _MOVE), e

    # ── knowledge-conditioned distance field ────────────────────────────────
    def field(self, know: dict, goal_kind: str) -> list:
        """``dist[c]`` = optimal number of actions from ``c`` to take an edge of
        kind ``goal_kind`` (``_WIN`` or ``_PROBE``); 0 means unreachable.

        Only ``_MOVE`` edges are traversed on the way: a ``_PROBE`` edge has an
        unknown outcome, so no plan may route THROUGH one, and a ``_WIN`` edge
        ends the level.
        """
        dist = [0] * self.n
        rev: dict[int, list[int]] = {}
        frontier = deque()
        for c in self.free:
            for d in range(4):
                s = self.succ(c, d, know)
                if s is None:
                    continue
                kind, node = s
                if kind == goal_kind and dist[c] == 0:
                    dist[c] = 1
                    frontier.append(c)
                elif kind == _MOVE:
                    rev.setdefault(node, []).append(c)
        while frontier:
            c = frontier.popleft()
            for p in rev.get(c, ()):
                if dist[p] == 0:
                    dist[p] = dist[c] + 1
                    frontier.append(p)
        return dist

    def optimal_dirs(self, dist: list, c: int, goal_kind: str,
                     know: dict) -> list[int]:
        """Every direction that strictly descends ``dist`` from ``c`` (the
        equally-optimal actions), in ACTION1..4 order."""
        here = dist[c]
        if here == 0:
            return []
        out = []
        for d in range(4):
            s = self.succ(c, d, know)
            if s is None:
                continue
            kind, node = s
            if kind == goal_kind:
                if here == 1:
                    out.append(d)
            elif kind == _MOVE and dist[node] == here - 1:
                out.append(d)
        return out


class Tp01Solver(BaseSolver):
    game_id = "tp01"
    # solve_from reads the live board (agent cell, geometry, observed wiring) and
    # re-plans from any reachable state; tp01 is reversible, so no exploratory
    # detour can strand a level and RESET is only a theoretical fallback.
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Single-entry per-level cache. ``_key`` includes the SEED, so the wiring
        # learned in one episode is never carried into the next one (that would be
        # privilege by the back door); a RESET inside a level keeps it, because the
        # board is unchanged and a player would remember it too.
        self._key = None
        self._geom: _Geom | None = None
        self._know: dict[int, int] = {}
        self._cur = (0, 0)                       # (seed, level_idx) of the last read

    def make_game(self, seed: int):
        # Tp01 is an AugmentedGame: the seed fixes each level's display rotation,
        # so a recorded episode replays byte-for-byte.
        return Tp01(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    # ── live board ──────────────────────────────────────────────────────────
    def _read(self, game, level_idx: int, seed: int):
        """``(geom, agent_cell)`` for the LIVE board. Everything here is what the
        frame shows: wall / portal / target / agent positions. The portal WIRING is
        not read -- see `_observe`."""
        level = game.current_level
        gw, gh = level.grid_size
        walls = tuple(sorted((s.x, s.y) for s in level.get_sprites_by_tag("wall")))
        portals = tuple(sorted((s.x, s.y)
                               for s in level.get_sprites_by_tag("portal")))
        targets = tuple(sorted((s.x, s.y)
                               for s in level.get_sprites_by_tag("target")))
        key = (seed, level_idx, gw, gh, walls, portals, targets)
        if key != self._key:
            self._key = key
            self._geom = _Geom(gw, gh, walls, portals, targets)
            self._know = {}
            self._deduce()
        self._cur = (seed, level_idx)
        player = level.get_sprites_by_tag("player")[0]
        return self._geom, self._geom.cell(player.x, player.y)

    # ── learning the wiring ─────────────────────────────────────────────────
    def _deduce(self) -> None:
        """Portals pair up perfectly, so when exactly two are still unpaired they
        must be each other's partner. This is inference from the VISIBLE portal
        set, not a peek at the level data: it settles every one-pair level up
        front, and level 3's second pair once its first has been seen."""
        if self._geom is None:
            return
        unpaired = [p for p in sorted(self._geom.portals) if p not in self._know]
        if len(unpaired) == 2:
            a, b = unpaired
            self._know[a] = b
            self._know[b] = a

    def _observe(self, before: tuple[int, int], direction: int | None,
                 after: tuple[int, int]) -> None:
        """Learn from one performed action.

        Stepping onto a portal ALWAYS warps (a portal is in-bounds and not
        collidable, so the step can never be refused), so wherever the agent ends
        up after entering one is that portal's partner -- learned both ways, since
        the pairs are symmetric. Note this must not be gated on the agent having
        MOVED: partners can sit next to each other, and warping from one straight
        onto the other looks like a no-op while still being a real observation.
        """
        geom = self._geom
        if geom is None or direction is None:
            return
        dx, dy = _DELTAS[direction]
        ex, ey = before[0] + dx, before[1] + dy
        if not (0 <= ex < geom.gw and 0 <= ey < geom.gh):
            return                                # off-board: the step was refused
        entered = geom.cell(ex, ey)
        if entered not in geom.portals:
            return                                # plain walk (or into a wall)
        landed = geom.cell(*after)
        self._know[entered] = landed
        self._know[landed] = entered
        self._deduce()

    def drive(self, game, action):
        """Perform one action and watch the outcome. This is the single choke point
        every action goes through -- expert, exploration prefix and burst alike --
        so the wiring is learned from exactly the frames the corpus records."""
        level = game.current_level
        player = level.get_sprites_by_tag("player")[0]
        before = (player.x, player.y)
        direction = None
        # Only learn while the cached geometry IS this level's (``solve_from``
        # always runs first in ``record_level``, so it normally is); a stale map
        # must never absorb another level's warp.
        if self._key is None or self._key[1] != game._current_level_index:
            return super().drive(game, action)
        if not action.is_click and action.action_id in (1, 2, 3, 4):
            game_action = game.screen_action_to_game(_ACTIONS[action.action_id - 1])
            direction = _ACTION_TO_DIR.get(game_action)
        res = super().drive(game, action)
        self._observe(before, direction, (player.x, player.y))
        return res

    # ── planning ────────────────────────────────────────────────────────────
    def _plan_mode(self, geom: _Geom, agent: int):
        """``(goal_kind, dist)`` for the objective in force at ``agent``: finish if
        a win is reachable with the wiring known so far, else go learn some wiring.
        ``(None, None)`` when neither is reachable (the level is unwinnable from
        here -- the base then records a RESET)."""
        for goal_kind in (_WIN, _PROBE):
            dist = geom.field(self._know, goal_kind)
            if dist[agent]:
                return goal_kind, dist
        return None, None

    def _screen(self, game, dirs) -> list:
        """Game-space direction indices -> the SCREEN keys to press, through this
        level's live rotation."""
        k = game.rotation_k
        return [_ACTIONS[d] for d in dirs]

    def solve_from(self, game, level_idx: int, seed: int):
        """An optimal plan from the game's LIVE state, given what has been observed:
        a full route to the target when one is known, otherwise the shortest route
        to the cheapest un-probed portal (whose outcome the base learns about via
        `drive`, after which the next call re-plans). ``[]`` only if the board is
        unwinnable from here."""
        geom, agent = self._read(game, level_idx, seed)
        goal_kind, dist = self._plan_mode(geom, agent)
        if goal_kind is None:
            return []
        plan: list[int] = []
        c = agent
        while True:
            dirs = geom.optimal_dirs(dist, c, goal_kind, self._know)
            if not dirs:                          # the field is exact: unreachable
                return []
            d = self.rng.choice(dirs)
            plan.append(d)
            kind, node = geom.succ(c, d, self._know)
            if kind == goal_kind:                 # the win, or the probing action
                break
            c = node
        return self._screen(game, plan)

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next action at the LIVE state -- the moves that
        strictly descend the field in force. One field lookup, so the full tie set
        is recorded as the training target essentially for free."""
        geom, agent = self._read(game, level_idx, seed)
        goal_kind, dist = self._plan_mode(geom, agent)
        if goal_kind is None:
            return None
        return self._screen(game, geom.optimal_dirs(dist, agent, goal_kind,
                                                    self._know))

    # ── burst verification ──────────────────────────────────────────────────
    def _burst_recovery_wins(self, game, plan: list) -> bool:
        """Would the agent's own policy actually WIN from the live (post-burst)
        state? The base's version replays one plan and requires a win, but a tp01
        plan may end at a PROBE, whose outcome is unknown by construction -- so
        replay to the end of the plan and keep re-planning as this dry run learns,
        which is exactly the loop `record_level` would run for real.

        The rollout is thrown away: the game is restored (as in the base) AND so is
        the observed wiring, because a hypothetical future must not teach the agent
        a pairing it never saw on camera."""
        snap = self._game_snapshot(game)
        known = dict(self._know)
        seed, level_idx = self._cur
        try:
            pending = list(plan)
            for _ in range(self.step_guard):
                if not pending:
                    pending = [_as_action(a)
                               for a in self.solve_from(game, level_idx, seed)]
                    if not pending:
                        return False              # stranded: unwinnable from here
                res = self.drive(game, pending.pop(0))
                if res.solved:
                    return True
                if res.dead:
                    return False
            return False
        except Exception:                          # noqa: BLE001 -- a crash is no win
            return False
        finally:
            self._game_restore(game, snap)
            self._know = known


if __name__ == "__main__":
    sys.exit(Tp01Solver.main())
