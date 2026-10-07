"""Generate Phase-1 training data for the tt02 "Collection Patrol Hazards" game.

tt02 (games/tt02/tt02.py): a collection game with MOVING hazards. ACTION1..4 step
the player up/down/left/right; stepping onto a cyan target collects it; the level
is won once every target is gone. Three levels (8x8, 16x16, 24x24). Each red
hazard walks a fixed closed loop, advancing ONE cell after every player action --
including an action that changed nothing (a wall bump or a move off the board), so
bumping a wall is a free "wait" and parity is a controllable resource.

The lethal rules, in the exact order ``Tt02.step`` applies them:

1. the player's move is resolved against the TOPMOST sprite of the destination
   cell (``Level.get_sprite_at``, layer-descending then sprite-list order): a
   target is collected and entered, a non-collidable is entered, a hazard KILLS,
   a wall blocks (the action is consumed, the player stays);
2. every hazard then advances one step along its loop;
3. if a hazard now shares the player's cell, the player dies -- so collecting the
   last target does NOT save you from a hazard stepping onto you afterwards.

Step 1's "topmost sprite" is load-bearing because the authored loops walk THROUGH
walls: level 1's hazard crosses the wall at (4,4) and level 2's left hazard
crosses the wall column at (8,6) and (8,8). Level 1 lists its walls before its
hazard, so there the wall wins and the cell merely blocks; level 2 lists hazards
before walls, so there the hazard wins and the same geometry is lethal. The model
below reproduces that by ranking the candidate sprites, not by assuming a
priority.

Expert
------
The state is exactly ``(player cell, remaining-target mask, patrol phase)``: every
hazard walks a deterministic cycle, so the whole hazard configuration is periodic
with period ``P = lcm(loop lengths)`` and "which cell is lethal when" is a pure
function of the phase. That space is tiny (<= ~41k states per level), so
``_Model`` enumerates it, builds the exact successor table for the four moves
(mirroring the three rules above), and runs ONE reverse BFS from the won states
to get the exact distance-to-win field for the level. `solve_from` is then a
lookup + greedy descent and `optimal_set_from` is the exact co-optimal tie set --
no search per step, and re-planning from an arbitrary state is free.

The patrol schedule is LEARNED, not read from the level's ``patrols`` data (which
no viewer of the frame can see): `_learn_patrol` clones the game, parks the clone's
player off the board so it cannot die or collect anything, and steps that clone --
the environment as a black-box simulator, exactly as the dc22/tu93/wa30 experts do
-- recording the hazard sprite cells (the observable quantity) after each action
until the configuration returns to where it started. That closes the cycle and
gives both ``P`` and the phase table. Reading the live hazard cells then names the
phase, because the ``P`` configurations of a cycle are necessarily distinct.

Recovery (supports_recovery = True, recovery_mode = "replan")
------------------------------------------------------------
``solve_from`` reads the LIVE game -- player cell, remaining target sprites, and
the hazard cells (which name the phase) -- and answers from the precomputed field,
so it re-plans optimally from ANY reachable state: after the exploration prefix,
mid-burst, after an off-plan target was collected, or at any patrol phase. A state
with no finite distance returns ``[]`` (the base then rolls the burst back or takes
a RESET); a burst that walks into a hazard is GAME_OVER, likewise handled by the
base. Because the field covers every ``(cell, mask, phase)``, recovery costs
nothing: the same lookup serves the initial state and any perturbed one.

The field is memoised per level geometry and tt02 is static (no seed), so the
whole run pays for three BFS builds; cross-episode variety comes from the
exploration prefix plus stochastic-optimal play over the (large) co-optimal tie
sets -- waiting against a wall for the right parity is frequently co-optimal, so
episodes differ a lot despite the fixed boards.

Usage (run from the repo root):
    python solvers/generate_tt02_training.py --episodes 1000 \
        --out data/training_multi_level/tt02
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                           # noqa: E402

from arcengine import ActionInput, GameAction                 # noqa: E402
from games.tt02.tt02 import Tt02                              # noqa: E402
from solvers.base_solver import BaseSolver                     # noqa: E402

#: (action id, dx, dy) in the order ``Tt02.step`` tests them.
_MOVES = ((1, 0, -1), (2, 0, 1), (3, -1, 0), (4, 1, 0))

#: Cap on the patrol-cycle probe. The authored loops are 6..10 long; anything
#: past this is a game whose hazards are not on a short cycle, which this expert
#: does not model (it fails the episode rather than guess).
_MAX_PERIOD = 512

_BIG = 1 << 30

# Sprite-kind codes used by the successor table.
_EMPTY, _TARGET, _HAZARD, _WALL = 0, 1, 2, 3


class _Model:
    """One level's exact state graph + distance-to-win field.

    ``rank`` maps a sprite tag to its position in the level's topmost-first sprite
    order, which is what decides a contested destination cell (see the module
    docstring). Per-TAG ranks are exact here because tt02 groups its sprites by
    kind, so no two tags interleave -- `Tt02Solver._tag_ranks` checks that and
    refuses the level otherwise rather than silently mis-modelling a collision.

    State ``(cell, mask, phase)`` is packed into one int as
    ``(phase * ncell + cell) * nmask + mask``; ``succ[a, s]`` is the successor of
    ``s`` under move ``a`` or ``-1`` when that move is lethal (or ``s`` is not a
    state the player can be in), and ``dist[s]`` is the exact number of actions
    from ``s`` to a won level (``-1`` = cannot win from here).
    """

    def __init__(self, gw: int, gh: int, walls: frozenset,
                 targets: tuple, occ: tuple, rank: dict) -> None:
        self.gw, self.gh = gw, gh
        self.ncell = gw * gh
        self.walls = walls
        self.targets = targets                  # index j <-> bit j of the mask
        self.k = len(targets)
        self.nmask = 1 << self.k
        self.occ = occ                          # occ[t] = hazard cells at phase t
        self.P = len(occ)
        self.phase_of = {occ[t]: t for t in range(self.P)}
        self.rank = rank
        self.N = self.P * self.ncell * self.nmask
        self._build()

    # ── packing ──────────────────────────────────────────────────────────────
    def cell(self, x: int, y: int) -> int:
        return y * self.gw + x

    def sid(self, cell: int, mask: int, phase: int) -> int:
        return (phase * self.ncell + cell) * self.nmask + mask

    def mask_of(self, live: set) -> int:
        return sum(1 << j for j, t in enumerate(self.targets) if t in live)

    # ── the state graph ──────────────────────────────────────────────────────
    def _build(self) -> None:
        gw, gh = self.gw, self.gh
        ncell, nmask, P = self.ncell, self.nmask, self.P
        cells = np.arange(ncell, dtype=np.int64)
        xs, ys = cells % gw, cells // gw

        wall_c = np.zeros(ncell, bool)
        for x, y in self.walls:
            wall_c[self.cell(x, y)] = True
        tbit = np.full(ncell, -1, np.int64)
        for j, (x, y) in enumerate(self.targets):
            tbit[self.cell(x, y)] = j
        haz_c = []                              # per phase: which cells are lethal
        for t in range(P):
            b = np.zeros(ncell, bool)
            for x, y in self.occ[t]:
                if 0 <= x < gw and 0 <= y < gh:
                    b[self.cell(x, y)] = True
            haz_c.append(b)

        r_t = self.rank.get("target", _BIG)
        r_h = self.rank.get("hazard", _BIG)
        r_w = self.rank.get("wall", _BIG)

        succ = np.full((4, self.N), -1, np.int32)
        valid = np.zeros(self.N, bool)
        for t in range(P):
            t2 = (t + 1) % P
            h_now, h_next = haz_c[t], haz_c[t2]
            # The player can never stand on a wall, and standing on a hazard is
            # death -- neither is a state to plan from.
            base_valid = ~wall_c & ~h_now
            # Topmost sprite per cell, mask-independent part (walls + hazards).
            kind0 = np.full(ncell, _EMPTY, np.int8)
            best0 = np.full(ncell, _BIG, np.int64)
            for present, r, kv in ((wall_c, r_w, _WALL), (h_now, r_h, _HAZARD)):
                take = present & (r < best0)
                kind0 = np.where(take, kv, kind0).astype(np.int8)
                best0 = np.where(take, r, best0)
            for m in range(nmask):
                kind = kind0
                if self.k and r_t < _BIG:
                    live = np.zeros(ncell, bool)
                    for j, (x, y) in enumerate(self.targets):
                        if m >> j & 1:
                            live[self.cell(x, y)] = True
                    take = live & (r_t < best0)
                    kind = np.where(take, _TARGET, kind0).astype(np.int8)
                s0 = (t * ncell + cells) * nmask + m
                valid[s0] = base_valid
                for ai, (_aid, dx, dy) in enumerate(_MOVES):
                    nx, ny = xs + dx, ys + dy
                    inb = (nx >= 0) & (nx < gw) & (ny >= 0) & (ny < gh)
                    nc = np.where(inb, ny * gw + nx, 0)
                    kd = np.where(inb, kind[nc], _EMPTY)
                    # rule 1: enter an empty cell or a target; a wall blocks
                    mv = inb & ((kd == _EMPTY) | (kd == _TARGET))
                    newc = np.where(mv, nc, cells)
                    coll = mv & (kd == _TARGET)
                    bits = np.where(coll & (tbit[newc] >= 0),
                                    1 << np.maximum(tbit[newc], 0), 0)
                    nm = m & ~bits
                    # rule 1 (hazard) + rule 3 (hazard steps onto the player)
                    dead = (inb & (kd == _HAZARD)) | h_next[newc]
                    succ[ai, s0] = np.where(
                        dead, -1, (t2 * ncell + newc) * nmask + nm)
        succ[:, ~valid] = -1
        self.succ, self.valid = succ, valid
        self.dist = self._field(cells)

    def _field(self, cells: np.ndarray) -> np.ndarray:
        """Exact distance-to-win for EVERY state, by reverse BFS from the won
        states (``mask == 0``). Each layer is a handful of vectorised gathers over
        the whole state space rather than a per-state expansion, so the build is
        milliseconds and every later query is a lookup."""
        dist = np.full(self.N, -1, np.int32)
        phases = np.arange(self.P, dtype=np.int64)
        won = ((phases[:, None] * self.ncell + cells[None, :])
               * self.nmask).ravel()             # mask == 0
        won = won[self.valid[won]]
        dist[won] = 0
        d = 0
        while True:
            upd = np.zeros(self.N, bool)
            for ai in range(4):
                sa = self.succ[ai]
                ok = sa >= 0
                upd |= ok & (dist[np.where(ok, sa, 0)] == d)
            newly = upd & (dist < 0)
            if not newly.any():
                return dist
            dist[newly] = d + 1
            d += 1

    # ── queries ──────────────────────────────────────────────────────────────
    def optimal(self, s: int) -> list:
        """Every action id that strictly decreases the distance to a won level."""
        d = int(self.dist[s])
        if d <= 0:
            return []
        out = []
        for ai, (aid, _dx, _dy) in enumerate(_MOVES):
            ns = int(self.succ[ai, s])
            if ns >= 0 and int(self.dist[ns]) == d - 1:
                out.append(aid)
        return out

    def plan(self, s: int) -> list:
        """One optimal action-id plan from ``s`` (greedy descent of the field)."""
        out = []
        while int(self.dist[s]) > 0:
            best = None
            d = int(self.dist[s])
            for ai, (aid, _dx, _dy) in enumerate(_MOVES):
                ns = int(self.succ[ai, s])
                if ns >= 0 and int(self.dist[ns]) == d - 1:
                    best = (aid, ns)
                    break
            if best is None:                      # field is consistent; unreachable
                return []
            out.append(best[0])
            s = best[1]
        return out


class Tt02Solver(BaseSolver):
    game_id = "tt02"
    #: solve_from answers from a field covering EVERY (cell, mask, phase), so it
    #: re-plans optimally from any live state; the only unrecoverable states are
    #: deaths, which the base handles with burst-undo / RESET.
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._models: dict = {}

    def make_game(self, seed: int):
        # tt02 takes no seed (its three levels are fixed); cross-episode variety
        # comes from stochastic-optimal play + the exploration prefix.
        return Tt02()

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    # ── observable reads ─────────────────────────────────────────────────────
    @staticmethod
    def _hazards(game) -> tuple:
        """The hazard cells, in the level's stable sprite order."""
        return tuple((int(s.x), int(s.y))
                     for s in game.current_level.get_sprites_by_tag("hazard"))

    @staticmethod
    def _tag_ranks(level) -> dict | None:
        """``{tag: rank}`` in ``Level.get_sprite_at``'s topmost-first order.

        A per-tag rank is only a faithful summary of the per-SPRITE order when the
        tags do not interleave (all walls before all hazards, say). tt02's levels
        are authored that way; if a level ever isn't, return ``None`` so the caller
        fails the episode instead of mis-resolving a contested cell."""
        order = sorted(level.get_sprites(), key=lambda s: s.layer, reverse=True)
        spans: dict = {}
        for i, s in enumerate(order):
            for tag in ("player", "target", "wall", "hazard"):
                if tag in s.tags:
                    lo, hi = spans.get(tag, (i, i))
                    spans[tag] = (min(lo, i), max(hi, i))
        for a in spans:
            for b in spans:
                if a != b and spans[a][0] <= spans[b][0] <= spans[a][1]:
                    return None                   # interleaved -> not summarisable
        return {tag: span[0] for tag, span in spans.items()}

    # ── the patrol schedule, learned by black-box probing ────────────────────
    def _learn_patrol(self, game) -> tuple | None:
        """The hazard-configuration cycle, starting at the LIVE configuration.

        Clones the game and parks the clone's player off the board, so every action
        is a no-op for the player (out-of-bounds moves are skipped) while the
        patrols still advance -- the player can neither die nor collect, so the
        clone runs forever and the level never changes under the probe. Steps it
        (without rendering; the parked sprite is outside the camera) until the
        hazard cells return to the starting configuration, which is the period.
        Only the hazard SPRITE CELLS are read -- the same thing the frame shows.
        """
        clone = copy.deepcopy(game)
        clone._player.set_position(-3, -3)
        occ0 = self._hazards(clone)
        occ = [occ0]
        for _ in range(_MAX_PERIOD):
            clone._full_reset = False
            clone._set_action(ActionInput(id=GameAction.ACTION1))
            guard = 0
            while not clone.is_action_complete() and guard < 16:
                guard += 1
                clone.step()
            cfg = self._hazards(clone)
            if cfg == occ0:
                return tuple(occ)
            occ.append(cfg)
        return None

    # ── model cache ──────────────────────────────────────────────────────────
    def _model(self, game, live_targets: frozenset):
        """The `_Model` for the live level, built on first use and memoised by the
        level's geometry. The field is built over the SUPERSET of targets seen so
        far, so its mask axis already contains every sub-mask -- collecting a
        target is just a lookup at a smaller mask, and a RESET that restores the
        targets needs no rebuild."""
        level = game.current_level
        gw, gh = (int(v) for v in level.grid_size)
        walls = frozenset((int(s.x), int(s.y))
                          for s in level.get_sprites_by_tag("wall"))
        key = (gw, gh, walls, len(self._hazards(game)))
        m = self._models.get(key)
        if m is not None and live_targets <= set(m.targets):
            return m
        rank = self._tag_ranks(level)
        if rank is None:
            return None
        occ = m.occ if m is not None else self._learn_patrol(game)
        if occ is None:
            return None
        targets = tuple(sorted(set(live_targets)
                               | (set(m.targets) if m is not None else set())))
        m = _Model(gw, gh, walls, targets, occ, rank)
        self._models[key] = m
        return m

    # ── live state -> state id ───────────────────────────────────────────────
    def _state(self, game):
        """``(model, state_id)`` for the LIVE board, or ``(None, -1)`` when the
        state cannot be planned from (no targets left, an unknown patrol phase,
        or a level this expert does not model)."""
        level = game.current_level
        live = frozenset((int(s.x), int(s.y))
                         for s in level.get_sprites_by_tag("target"))
        if not live:
            return None, -1                       # nothing left to collect
        m = self._model(game, live)
        if m is None:
            return None, -1
        phase = m.phase_of.get(self._hazards(game))
        if phase is None:
            return None, -1                       # off-schedule: not our level
        p = game._player
        px, py = int(p.x), int(p.y)
        if not (0 <= px < m.gw and 0 <= py < m.gh):
            return None, -1
        return m, m.sid(m.cell(px, py), m.mask_of(set(live)), phase)

    # ── BaseSolver hooks ─────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """Optimal plan from the game's CURRENT state; ``[]`` when the level
        cannot be won from here (so the base rolls a burst back or RESETs)."""
        m, s = self._state(game)
        if m is None or int(m.dist[s]) <= 0:
            return []
        return m.plan(s)

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """The exact co-optimal tie set at the live state -- every move that keeps
        the player on a shortest route to a won level. Often several: with the
        patrol phase in the state, waiting against a wall to fix parity is
        regularly as good as advancing, which is what makes episodes of these
        three fixed boards differ."""
        m, s = self._state(game)
        if m is None:
            return None
        return m.optimal(s) or None


if __name__ == "__main__":
    sys.exit(Tt02Solver.main())
