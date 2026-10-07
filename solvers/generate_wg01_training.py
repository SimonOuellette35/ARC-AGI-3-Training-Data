"""Generate Phase-1 training data for the wg01 "Wind Gust" game.

wg01 (games/wg01/wg01.py): a 10x10 walk-to-the-goal game with a periodic EAST
wind. ACTION1..4 step the player up/down/left/right; the level is won when the
player ends an action on the green goal cell. What makes it more than a trivial
walk is the gust, applied by ``Wg01.step`` in this exact order:

1. the player's move resolves (blocked by the board edge and by wall sprites, in
   which case the action is consumed and the player stays put);
2. the gust counter ticks; every ``K``-th action it fires and pushes the player
   ONE cell east, unless that cell is a wall or off the board;
3. only THEN is the win tested.

Rule 3 is the whole puzzle. Stepping onto the goal is not enough: if the gust
fires on that same action and the cell east of the goal is free, the player is
blown off the goal before the test runs and nothing happens. So the player has to
arrive at the goal on an action whose gust does not fire (or whose push is walled
off) -- the phase of the gust counter is a resource to be managed, and a move
that changes nothing (bumping the board edge or a wall) is a free "wait" that
advances that phase. Levels 4 and 5 (``K = 2``) cannot be solved by walking
straight at the goal; the route has to burn a step to fix the parity.

Expert
------
The state is exactly ``(player cell, gust counter)`` -- 10*10*K <= ~400 states
per level -- so ``_Model`` enumerates it, builds the exact successor table for
the four moves (mirroring rules 1-3 above, including "the gust can push you ONTO
the goal", which is a legitimate and sometimes optimal win), and runs ONE reverse
BFS to get the exact distance-to-win field. `solve_from` is then a lookup plus a
greedy descent and `optimal_set_from` is the exact co-optimal tie set, so
re-planning from an arbitrary state is free.

The win is a TRANSITION, not a state (as in tk01): the level ends the instant an
action leaves the player on the goal, so "on the goal" is not a state the field
can contain. The field therefore stores ``dist == 1`` for every state with a
winning MOVE and BFSes backwards from those, rather than from a won state.

Partial observability: the counter is NOT privileged
---------------------------------------------------
Everything the model needs but the gust is frame-visible: the grid, the player
(colour 9), the goal (14) and the walls (3). ``K`` and the live counter are NOT
-- ``K`` only appears as HUD ticks and the counter appears nowhere at all -- so
this expert never reads ``game._k`` / ``game._ctr`` / the level's ``wind_k``.
`_probe_gust` learns both by black-box probing a CLONE of the live game, exactly
as tt02's ``_learn_patrol`` learns its patrol cycle: it parks the clone's player
in column 0 of a free row and repeatedly presses LEFT, which can never move the
player east, so an observed x == 1 means "the gust just fired". The gap between
two fires is ``K`` and the wait until the first fire names the current phase.

Probing costs a deepcopy, so it is not done per step: the phase is probed once
per level, then TRACKED (one increment per action, on the game object itself so
that a burst snapshot/restore rolls it back with everything else), and every
`drive` CHECKS the tracked phase against reality -- the model predicts the exact
cell an action lands on, so a mismatch means the phase is wrong and re-probes.
The tracked value is nothing an agent could not count for itself; the probe is
what keeps it honest and self-healing.

Recovery (supports_recovery = True, recovery_mode = "replan")
------------------------------------------------------------
`solve_from` reads the LIVE game -- player cell, walls, goal, and the gust phase
-- and answers from the precomputed field, so it re-plans optimally from ANY
reachable state: after the exploration prefix, mid-burst, after the wind has
blown the player somewhere the plan never expected. wg01 has no death and no
irreversible move, and the field covers every ``(cell, phase)``, so recovery is
always possible and costs one array lookup; RESET is never needed in practice.

wg01 is static (five authored levels, no seed), so cross-episode variety comes
from the exploration prefix plus stochastic-optimal play over the co-optimal tie
sets -- which are large here, since any interleaving of the moves along a
staircase route is co-optimal and so is the choice of WHICH free wait to burn.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_wg01_training.py --episodes 1000 \
        --out data/training_multi_level/wg01
"""

from __future__ import annotations

import copy
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction                  # noqa: E402
from games.wg01.wg01 import Wg01                               # noqa: E402
from solvers.base_solver import BaseSolver, DriveResult        # noqa: E402
from utils.explore import Action                               # noqa: E402

#: (action id, dx, dy) in the order ``Wg01.step`` tests them.
_MOVES = ((1, 0, -1), (2, 0, 1), (3, -1, 0), (4, 1, 0))

#: Cap on the gust probe. It needs 2*K + 1 actions to see two fires; a game whose
#: gust is not on a short cycle is not the game this expert models, so the probe
#: gives up (and the episode fails) rather than guess.
_MAX_PROBE = 64

#: Where the tracked gust phase lives. On the GAME object, not the solver, so a
#: burst snapshot/restore (`BaseSolver._game_snapshot` deep-copies the whole game)
#: rolls the phase back in lock-step with the state it describes.
_PHASE_ATTR = "_wg01_gust_phase"


class _Model:
    """One wg01 level's exact state graph + distance-to-win field.

    State is ``(cell, phase)`` packed as ``cell * k + phase``, where ``phase`` is
    the number of actions already taken inside the current gust window (the
    game's ``_ctr``, in ``[0, k)``). ``succ[a][s]`` is the successor of ``s``
    under move ``a`` (``-1`` when that move wins, or when ``s`` is not a state the
    player can occupy), ``win[a][s]`` says the move wins outright, and ``dist[s]``
    is the exact number of actions from ``s`` to a win (``-1`` = unreachable).
    """

    def __init__(self, gw: int, gh: int, walls: frozenset,
                 goal: tuple, k: int) -> None:
        self.gw, self.gh = gw, gh
        self.walls, self.goal, self.k = walls, goal, k
        self.N = gw * gh * k
        self._build()

    # ── packing ──────────────────────────────────────────────────────────────
    def sid(self, x: int, y: int, phase: int) -> int:
        return (y * self.gw + x) * self.k + phase

    def _free(self, x: int, y: int) -> bool:
        """``not Wg01._blocked``: on the board and not a wall. The goal sprite is
        non-collidable, so it is a free cell for both moves and the gust."""
        return 0 <= x < self.gw and 0 <= y < self.gh and (x, y) not in self.walls

    # ── the state graph ──────────────────────────────────────────────────────
    def _build(self) -> None:
        k = self.k
        self.succ = [[-1] * self.N for _ in range(4)]
        self.win = [[False] * self.N for _ in range(4)]
        for y in range(self.gh):
            for x in range(self.gw):
                # Walls can't be stood on, and the goal is never a resting state
                # (arriving there ends the level).
                if not self._free(x, y) or (x, y) == self.goal:
                    continue
                for phase in range(k):
                    s = self.sid(x, y, phase)
                    for ai, (_aid, dx, dy) in enumerate(_MOVES):
                        nx, ny = x + dx, y + dy
                        if not self._free(nx, ny):      # rule 1: blocked -> stay
                            nx, ny = x, y
                        nphase = phase + 1
                        if nphase >= k:                 # rule 2: the gust fires
                            nphase = 0
                            if self._free(nx + 1, ny):
                                nx += 1
                        if (nx, ny) == self.goal:       # rule 3: win tested LAST
                            self.win[ai][s] = True
                        else:
                            self.succ[ai][s] = self.sid(nx, ny, nphase)
        self.dist = self._field()

    def _field(self) -> list:
        """Exact distance-to-win for every state, by reverse BFS from the states
        that have a winning move (the win is a transition, so those are the
        ``dist == 1`` layer -- there is no ``dist == 0`` state)."""
        dist = [-1] * self.N
        preds: list[list[int]] = [[] for _ in range(self.N)]
        seeds = []
        for s in range(self.N):
            for ai in range(4):
                if self.win[ai][s]:
                    if dist[s] < 0:
                        dist[s] = 1
                        seeds.append(s)
                elif self.succ[ai][s] >= 0:
                    preds[self.succ[ai][s]].append(s)
        q = deque(seeds)
        while q:
            s = q.popleft()
            for p in preds[s]:
                if dist[p] < 0:
                    dist[p] = dist[s] + 1
                    q.append(p)
        return dist

    # ── queries ──────────────────────────────────────────────────────────────
    def step(self, s: int, aid: int):
        """``(successor, wins)`` for action id ``aid`` at ``s``."""
        ai = aid - 1
        return self.succ[ai][s], self.win[ai][s]

    def optimal(self, s: int) -> list:
        """Every action id that strictly decreases the distance to a win. Often
        several: the co-optimal set contains every order of a staircase route and
        every equally-good free "wait", which is what makes episodes of these
        five fixed boards differ."""
        d = self.dist[s]
        if d < 1:
            return []
        out = []
        for ai, (aid, _dx, _dy) in enumerate(_MOVES):
            if self.win[ai][s]:
                if d == 1:
                    out.append(aid)
                continue
            ns = self.succ[ai][s]
            if ns >= 0 and self.dist[ns] == d - 1:
                out.append(aid)
        return out

    def plan(self, s: int) -> list:
        """One optimal action-id plan from ``s`` (greedy descent of the field)."""
        out: list[int] = []
        while self.dist[s] >= 1:
            opts = self.optimal(s)
            if not opts:                     # field is consistent; be defensive
                return []
            aid = opts[0]
            out.append(aid)
            ns, wins = self.step(s, aid)
            if wins:
                return out
            s = ns
        return out


class Wg01Solver(BaseSolver):
    game_id = "wg01"
    #: solve_from answers from a field covering EVERY (cell, phase) of the live
    #: board, so it re-plans optimally from any state the exploration prefix or a
    #: burst can reach. wg01 has no death and no irreversible move, so recovery
    #: never needs a RESET.
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._models: dict = {}
        self._periods: dict = {}      # level geometry -> learned gust period K

    def make_game(self, seed: int):
        # wg01 takes no seed (its five levels are fixed); cross-episode variety
        # comes from stochastic-optimal play + the exploration prefix.
        return Wg01()

    # ── observable reads (all of these are visible in the 64x64 frame) ────────
    @staticmethod
    def _geometry(game):
        """``(gw, gh, walls, goal)`` for the live level, or ``None`` if the board
        is not the one this expert models (no goal, several goals, ...)."""
        level = game.current_level
        if not level.grid_size:
            return None
        gw, gh = (int(v) for v in level.grid_size)
        goals = level.get_sprites_by_tag("goal")
        players = level.get_sprites_by_tag("player")
        if len(goals) != 1 or len(players) != 1:
            return None
        walls = frozenset((int(s.x), int(s.y))
                          for s in level.get_sprites_by_tag("wall"))
        return gw, gh, walls, (int(goals[0].x), int(goals[0].y))

    @staticmethod
    def _player(game) -> tuple:
        p = game.current_level.get_sprites_by_tag("player")[0]
        return int(p.x), int(p.y)

    # ── the gust, learned by black-box probing ───────────────────────────────
    def _probe_gust(self, game):
        """``(K, phase)`` for the LIVE gust, learned without reading the game's
        hidden counter.

        Clones the game, parks the clone's player in column 0 of a row whose two
        leftmost cells are free (and are not the goal, so the clone can never win
        and end the probe early), and repeatedly presses LEFT. LEFT can only ever
        move the player west, and from x == 0 it is blocked outright, so the
        player is at x == 1 after an action if and only if the gust fired on that
        action. The gap between two fires is the period ``K``; the wait until the
        first fire names the phase (a fire on the very first probe action means
        the counter was already at ``K - 1``).

        Only the player's rendered cell is read, and only from a throwaway clone,
        so this leaves the live game untouched. ``None`` when no free park row
        exists or no two fires are seen inside `_MAX_PROBE` actions.
        """
        geom = self._geometry(game)
        if geom is None:
            return None
        gw, gh, walls, goal = geom
        park = next((y for y in range(gh)
                     if all((x, y) not in walls and (x, y) != goal
                            for x in (0, 1))), None)
        if park is None:
            return None
        clone = copy.deepcopy(game)
        # A pending level-advance would make ``is_action_complete`` never fire on
        # the clone (it ANDs in ``not _next_level``), so each probe action would
        # run to the step guard and the gust would be miscounted. The clone is a
        # throwaway, so clear it -- exactly as ``BaseSolver.set_level`` does.
        clone._next_level = False
        clone.current_level.get_sprites_by_tag("player")[0].set_position(0, park)
        fires = []
        for t in range(1, _MAX_PROBE + 1):
            clone._full_reset = False
            clone._set_action(ActionInput(id=GameAction.ACTION3))   # LEFT
            guard = 0
            while not clone.is_action_complete() and guard < 8:
                guard += 1
                clone.step()
            if self._player(clone)[0] == 1:                          # gust fired
                fires.append(t)
                if len(fires) == 2:
                    k = fires[1] - fires[0]
                    return k, (k - fires[0]) % k
        return None

    def _gust(self, game):
        """``(K, phase)`` for the live game: the TRACKED phase where available,
        else a fresh probe (which also caches ``K`` for this level geometry).
        ``None`` when the gust cannot be learned."""
        geom = self._geometry(game)
        if geom is None:
            return None
        key = geom
        phase = getattr(game, _PHASE_ATTR, None)
        k = self._periods.get(key)
        if phase is None or k is None:
            probed = self._probe_gust(game)
            if probed is None:
                return None
            k, phase = probed
            self._periods[key] = k
            setattr(game, _PHASE_ATTR, phase)
        return k, phase % k

    # ── model cache ──────────────────────────────────────────────────────────
    def _model(self, game):
        """The `_Model` for the live level (memoised by geometry + period), or
        ``None`` if the level/gust cannot be modelled."""
        geom = self._geometry(game)
        gust = self._gust(game)
        if geom is None or gust is None:
            return None
        gw, gh, walls, goal = geom
        k, _phase = gust
        key = (gw, gh, walls, goal, k)
        m = self._models.get(key)
        if m is None:
            m = _Model(gw, gh, walls, goal, k)
            self._models[key] = m
        return m

    def _state(self, game):
        """``(model, state_id)`` for the LIVE board, or ``(None, -1)`` when there
        is nothing to plan from (unmodelled level, unknown gust, or a player cell
        the field does not cover)."""
        m = self._model(game)
        if m is None:
            return None, -1
        gust = self._gust(game)
        if gust is None:
            return None, -1
        _k, phase = gust
        px, py = self._player(game)
        if not m._free(px, py) or (px, py) == m.goal:
            return None, -1
        return m, m.sid(px, py, phase)

    # ── phase bookkeeping: one increment per action, verified every action ────
    def set_level(self, game, level_idx: int) -> None:
        # A new level restarts the gust window, but rather than assume that,
        # forget the tracked phase so the next query re-probes it.
        super().set_level(game, level_idx)
        setattr(game, _PHASE_ATTR, None)

    def drive(self, game, action: Action) -> DriveResult:
        """Perform the action, then advance the tracked gust phase -- and CHECK
        it. The model predicts the exact cell the action lands on (for the random
        exploration/burst actions just as much as for the expert's own), so an
        observed cell that disagrees means the tracked phase is wrong: drop it and
        let the next query re-probe. This keeps the cheap tracking honest without
        paying for a clone per step."""
        m, s = self._state(game)
        expect = None
        if m is not None and not action.is_click and 1 <= action.action_id <= 4:
            ns, wins = m.step(s, action.action_id)
            expect = (m.goal, 0) if wins else (
                (ns // m.k % m.gw, ns // m.k // m.gw), ns % m.k)
        res = super().drive(game, action)
        if expect is None or res.solved:
            # A solve advances the engine to the next level (whose set_level
            # clears the phase), so there is nothing to track here.
            setattr(game, _PHASE_ATTR, None)
            return res
        cell, nphase = expect
        setattr(game, _PHASE_ATTR,
                nphase if self._player(game) == cell else None)
        return res

    # ── BaseSolver hooks ─────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """Optimal plan from the game's CURRENT state; ``[]`` when the level
        cannot be won from here (which wg01 never is -- there is no death and no
        irreversible move -- so this is only a guard)."""
        m, s = self._state(game)
        if m is None or m.dist[s] < 1:
            return []
        return m.plan(s)

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """The exact co-optimal tie set at the live state: every move that keeps
        the player on a shortest route to the goal, including the free waits
        (edge/wall bumps) that only exist to fix the gust parity."""
        m, s = self._state(game)
        if m is None:
            return None
        return m.optimal(s) or None


if __name__ == "__main__":
    sys.exit(Wg01Solver.main())
