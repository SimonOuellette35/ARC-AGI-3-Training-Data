"""Generate Phase-1 training data for the PuzzleScript game ps:clean_up
("Clean Up" by Alex Yang).

The harness -- the rotation contract, the trajectory recorder and the BaseSolver
plumbing -- lives in `solvers/common/ps_astar.py`, shared with the other
search-the-interpreter ps: generators. This file is only the game-specific part:
its name, the exact distance-to-win field it plans with, and the sprite/flip
notes below.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_clean_up",
      "levels": [
        {
          "level_id": 0,
          "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
          "actions":      [a_0, a_1, ..., a_{T-1}], # length T
        },
        ...
      ]
    }

actions[0] is the RESET that produced obs[0]; actions[i] for i>=1 is the action
that took the agent from obs[i-1] to obs[i]. The recorded index is the *screen*
action (post rotation/flip remap), i.e. the button an agent presses in the
augmented view, so replaying the recorded actions reproduces the recorded frames
exactly.

The game
--------
A sokoban with a twist that inverts the usual goal. There are six colours of
block and six colours of target; a target EATS a block of its own colour the
instant one lands on it (``late [ A 0 ] -> [ 0 ]``), and the win condition is
``No Block`` -- clear the board, not cover the targets. Targets are never
consumed, and a level can ship more targets than blocks, so a block only has to
find SOME target of its colour.

Three rules matter, and only the first is ordinary:

  * **Push, in chains.** ``[> Player | Block] -> [> Player | > Block]`` plus
    ``[> Block | Block] -> [> Block | > Block]``, so the player shoves a whole
    row of blocks at once. Walls (and the board edge) cancel the turn;
    ``require_player_movement`` means any press that fails to move the player is
    a no-op, not a wasted turn.

  * **PULL: every target is a block dispenser.** ``[ < Player | 0 no Block ] ->
    [ < Player | < A 0 ]`` -- when the player steps AWAY from a cell holding a
    target with no block on it, the target spits out a fresh block of its colour
    into the cell the player just left. The target is not "used up" by having
    eaten something; it fires every single time. So walking is not free the way
    it is in a sokoban: a step whose *behind* cell is a bare target manufactures
    work. Block count is not conserved and the board can be made arbitrarily
    dirty. (It is at least immediately undoable: the pulled block lands directly
    behind the player, so reversing the step pushes it straight back onto the
    target that made it, which eats it again.)

  * **Blocking.** The pull needs ``no Block`` on the target cell, so parking a
    MISMATCHED block on a target -- red on orange, say -- switches that target's
    dispenser off. Level 6 ("learn about blocking targets") and level 7 are built
    on it: their only route past a bare target is to plug it first.

ACTION5 is bound to nothing (``noaction``), so the four directions are the whole
action space.

All 8 shipped levels are winnable and all 8 are recorded.

Expert solver
-------------
An EXACT distance-to-win field over the level's reachable states, not a search.
The pull rule is what forces this shape:

  * `PSPushExpert`, the family's macro searcher, is unusable here. It models a
    walk as a side-effect-free BFS between pushes, and in this game a walk that
    passes a bare target fabricates a block. Every macro would be a lie.

  * A primitive A* would work, but it does not have to: the reachable state space
    is TINY. Blocks that can never be eaten again make a state dead (see below),
    and pruning those leaves 61 - 23_362 states per level, ~32k over all eight.
    The whole space is cheaper to enumerate than to search.

So the expert forward-BFSes the real interpreter from the level start, keeping
every ``(player cell, {block colour: cell})`` state and the four labelled edges
out of it, then runs a backward BFS from the winning TRANSITIONS to get an exact
distance-to-win for every state. Two things fall out that a search does not give:

  * Plans are provably SHORTEST, with no heuristic to trust and no weight to tune.
  * The **exact optimal-action set** at every step for free -- every direction
    whose successor is one closer to the win -- rather than one arbitrary
    shortest path presented as the only right answer (`optimal_for`). These
    boards have real ties: the walks between pushes are mostly free-order.

**Dead-state pruning** is what keeps the enumeration small, and it is exact
enough to be safe. Per colour, a reverse BFS over push edges (a block reaches
``X`` from ``X - d`` when ``X - d`` is not a wall and the pusher's cell ``X - 2d``
is not a wall either -- other BLOCKS never block, since a chain push drives
through them) marks the cells from which a block of that colour can still reach a
target of that colour. A block outside its colour's set can never be eaten again,
so the state can never win: the successor is dropped instead of expanded. It is a
relaxation of the real dynamics in the safe direction -- it only ever calls a
state alive that might really be doomed -- so no winning line is ever pruned.

The state key merges PlayerL/PlayerR. The facing is set by
``[left Player] -> [left PlayerL]`` and is pure sprite art: no rule reads it, and
``require_player_movement`` cancels any press that would flip it without moving,
so it cannot vary independently of the position. Merging it halves the space and
changes no plan.

Rendering
---------
Checked against a per-cell frame dump before being trusted, per the usual
palette-collision rule, and one fix was needed.

Targets sit on the collision layer BELOW ``Player, Wall, A..F``, and the block
sprites were fully opaque, so a mismatched block parked on a target rendered
byte-identically to that block on bare background -- the frames hid the exact
piece of state levels 6 and 7 are about, and hid it for as long as the block
stayed there. `data/puzzlescript_games/Clean_Up.txt` now punches the block
sprite's 3x3 interior out to transparent, keeping a centre pip::

    02222        02222
    10002        1...2
    10002   ->   1.0.2
    10002        1...2
    11110        11110

The hole is exactly the eight cells the target ring occupies, so a covered target
shows through in full (verified: red-on-orange-target now differs from
red-on-background), while a block on bare ground still reads as a solid coloured
frame with a pip. Mechanically inert -- PuzzleScript sprites are art only.

Colours survive the 16-entry ARC quantization: wall 5/3, background 1/0, player
13/12/9/5, and the six block/target pairs land on 8 (red), 9 (blue), 11 (yellow),
12 (orange), 14 (green), 15 (purple), each block a 5x5 frame and each target a
3x3 ring, so the two are told apart by shape as well as by colour. GreenBlock's
three source colours all collapse onto 14 (the ARC palette has one green) --
harmless here, it just makes the green block flat.

The player standing on a target still hides it, and that one is left alone: the
player sprite has to stay legible, no level starts the player on a target, and it
is transient by construction -- the target is back in the frame the moment the
player steps off.

Augmentation
------------
Clean Up's engine state after reset is identical for every seed (levels are fixed
ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to), re-sampling the
board's 8-element symmetry group -- 8 levels x 4 rotations = 32 presentations
without the flips, 128 with them. The flips are exact symmetries here: no gravity,
screen-relative input, and every rule is written in all four directions. The
player's L/R facing sprite is not a counterexample -- under a horizontal flip the
adapter remaps screen-left to engine-right, which paints PlayerR, whose mirrored
art faces left again. There is no colour augmentation: block and target are
matched BY COLOUR and by nothing else, so a recolor could only destroy the rule.

The field is therefore seed-independent: built once per level, cached, and
replayed per seed with that seed's remapped screen actions. Seed 0 pays ~55s for
all eight enumerations; every later seed replays them from cache.

Usage (run from the repo root):
    python solvers/generate_clean_up_training.py --episodes 200 \
        --out data/training_multi_level/clean_up

    python solvers/generate_clean_up_training.py --plans   # per-level report
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (  # noqa: E402
    PSAStarSolver, PSExpert, restore, snapshot,
)

GAME_NAME = "Clean_Up"

#: (dr, dc) per engine direction, for the per-colour push-reachability tables.
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Stand-in for "a player is here" in the state key. PlayerL and PlayerR are one
#: state -- see the module docstring. Negative so it can never collide with an
#: object index.
_PLAYER = -1

#: ``target object name -> the block object name it eats``.
_PAIRS = (
    ("redtarget", "redblock"),
    ("purpletarget", "purpleblock"),
    ("bluetarget", "blueblock"),
    ("greentarget", "greenblock"),
    ("yellowtarget", "yellowblock"),
    ("orangetarget", "orangeblock"),
)


class CleanUpExpert(PSExpert):
    """Exact shortest-path planner over the level's whole reachable state space.

    `plan` returns a provably shortest win path and `optimal_set` returns every
    press that is on SOME shortest path from the current state. See the module
    docstring for why this game is enumerated rather than searched."""

    #: ACTION5 is bound to nothing (`noaction`), so it is not worth a branch.
    directions = ["up", "down", "left", "right"]

    #: `_key` is blocks + player only; walls and targets are static per level but
    #: differ between levels, so the memo must not be shared across them.
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        self.wall = g.obj_name_to_idx["wall"]
        self.player_ids = set(self.game._engine._player_indices)
        #: ``block id -> the target ids that eat it``.
        self.eaten_by: dict[int, set[int]] = {}
        for target, block in _PAIRS:
            self.eaten_by.setdefault(g.obj_name_to_idx[block], set()).add(
                g.obj_name_to_idx[target])
        self.block_ids = set(self.eaten_by)
        #: ``level -> (reachable keys, distance to win, optimal presses)``.
        self._fields: dict[int | None, tuple] = {}
        self._level: int | None = None

    # -- state key ------------------------------------------------------------
    def _key(self, eng) -> frozenset:
        """Blocks (with their colour) and the player. Walls and targets never
        change -- no rule creates or destroys either -- so they are redundant
        within a level, which is what ``scope_by_level`` pays for."""
        blocks, players = self.block_ids, self.player_ids
        out = []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                for o in cell:
                    if o in blocks:
                        out.append((r, c, o))
                    elif o in players:
                        out.append((r, c, _PLAYER))
        return frozenset(out)

    # -- dead-state test ------------------------------------------------------
    def _live_cells(self, eng) -> dict[int, set]:
        """``{block id: cells from which a block of that colour can still reach
        a target of that colour}``.

        Reverse BFS over push edges from every matching target, ignoring the
        other blocks (they are not obstacles -- a chain push drives straight
        through them -- only walls are). A block outside its own set can never
        be eaten, so any state holding one is unwinnable."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        wall = self.wall
        blocked = [[wall in grid[r][c] for c in range(w)] for r in range(h)]
        live = {}
        for block, targets in self.eaten_by.items():
            seats = [(r, c) for r in range(h) for c in range(w)
                     if grid[r][c] & targets]
            seen = set(seats)
            queue = deque(seats)
            while queue:
                r, c = queue.popleft()
                for dr, dc in _DELTA:
                    pr, pc = r - dr, c - dc          # where the block comes from
                    sr, sc = pr - dr, pc - dc        # who pushes it from there
                    if not (0 <= pr < h and 0 <= pc < w) or blocked[pr][pc]:
                        continue
                    if not (0 <= sr < h and 0 <= sc < w) or blocked[sr][sc]:
                        continue
                    if (pr, pc) not in seen:
                        seen.add((pr, pc))
                        queue.append((pr, pc))
            live[block] = seen
        return live

    @staticmethod
    def _doomed(key: frozenset, live: dict[int, set]) -> bool:
        return any(o != _PLAYER and (r, c) not in live[o] for r, c, o in key)

    # -- the field ------------------------------------------------------------
    def _build_field(self, eng) -> tuple:
        """Enumerate every state reachable from the engine's CURRENT grid, then
        solve backwards from the wins. Returns ``(seen, dist, opt)``:

          * ``seen``  -- every reachable, non-doomed state key (so a key absent
            from ``dist`` but present here is reachable and hopeless, which is a
            different answer from "this field is about a different position").
          * ``dist``  -- exact presses remaining to the win.
          * ``opt``   -- every press whose successor is one closer.

        Leaves the engine grid exactly as it found it."""
        live = self._live_cells(eng)
        start = snapshot(eng)
        k0 = self._key(eng)
        states = {k0: start}
        edges: dict[frozenset, list] = {}
        queue = deque([k0])
        while queue:
            k = queue.popleft()
            here = states[k]
            out = []
            for direction in self.directions:
                restore(eng, here)
                eng.step(direction)
                if eng.check_win():
                    out.append((direction, None))     # None == "and that wins"
                    continue
                nk = self._key(eng)
                # A press the engine refused (a wall, or a push it cancelled) --
                # `require_player_movement` makes those true no-ops.
                if nk == k or self._doomed(nk, live):
                    continue
                if nk not in states:
                    states[nk] = snapshot(eng)
                    queue.append(nk)
                out.append((direction, nk))
            edges[k] = out
        restore(eng, start)

        # Backward BFS from the winning transitions. A win is an EDGE, not a
        # state (`check_win` reads a per-step flag), so the base layer is every
        # state with a winning press, at distance 1.
        preds: dict[frozenset, list] = {}
        dist: dict[frozenset, int] = {}
        layer = []
        for k, out in edges.items():
            for _d, nk in out:
                if nk is None:
                    if k not in dist:
                        dist[k] = 1
                        layer.append(k)
                else:
                    preds.setdefault(nk, []).append(k)
        while layer:
            nxt = []
            for k in layer:
                for p in preds.get(k, ()):
                    if p not in dist:
                        dist[p] = dist[k] + 1
                        nxt.append(p)
            layer = nxt

        opt = {}
        for k, out in edges.items():
            d = dist.get(k)
            if d is None:
                continue                              # reachable but hopeless
            opt[k] = [direction for direction, nk in out
                      if (d - 1 if nk is None else dist.get(nk, -1)) == d - 1]
        return set(states), dist, opt

    def _field(self, eng) -> tuple:
        """The field covering the engine's current state, building it if needed.

        Normally built once per level from that level's start (``discover_solvable``
        gets there first) and hit from cache forever after. The rebuild is the
        safety net for a caller that arrives at a state outside the cached
        component -- it can only happen if something reached this level by a route
        other than `set_level`, and re-enumerating from here is still exact. That
        one is deliberately NOT cached: the level's entry has to keep the field
        built from its START, which is the one every later seed replays from."""
        level = self._level
        field = self._fields.get(level)
        if field is None:
            field = self._fields[level] = self._build_field(eng)
        elif self._key(eng) not in field[0]:
            field = self._build_field(eng)
        return field

    # -- planning -------------------------------------------------------------
    def plan(self, eng, level: int | None = None):
        self._level = level
        return super().plan(eng, level)

    def _search(self, eng) -> list | None:
        """Walk the field downhill. Ties break on `directions` order, so the plan
        is deterministic; the alternatives are not lost, they are what
        `optimal_set` hands the recorder."""
        if eng.check_win():
            return []
        _seen, _dist, opt = self._field(eng)
        best = opt.get(self._key(eng))
        if best is None:
            return None
        out = []
        while best:
            direction = best[0]
            out.append(direction)
            eng.step(direction)
            if eng.check_win():
                break
            best = opt.get(self._key(eng))
        return out                      # `plan` restores the grid around this

    def optimal_set(self, eng, level: int | None = None) -> list | None:
        """Every press that is on a shortest win path from the CURRENT state."""
        self._level = level
        return self._field(eng)[2].get(self._key(eng))


class CleanUpSolver(PSAStarSolver):
    game_id = "puzzlescript_clean_up"
    game_name = GAME_NAME
    expert_cls = CleanUpExpert

    #: The expert enumerates rather than searches, so neither of the search dials
    #: is read; `weight` stays 1 because the plans really are shortest.
    weight = 1
    #: Plans are 6-40 presses. The ceiling only has to cover a re-plan after an
    #: epsilon detour, and `epsilon` is 0 for this family.
    max_steps = 150

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The exact set of shortest-path presses at this step.

        `record_level` calls this with the engine still AT the state being
        labelled, so the field can be read live. The fall-back to the press about
        to be taken is belt-and-braces -- the field covers every state the plan
        visits -- and it is there so no expert step can ever ship unlabelled (the
        always-emit-optimal-targets rule)."""
        best = expert.optimal_set(expert.game._engine, level)
        if best:
            return best
        return [plan[pi]] if pi < len(plan) else None


def _report() -> int:
    """Per-level report: how big the enumerated space is, how long the shortest
    win is, and how many of its steps carry a genuine tie. The plan length is the
    column that matters -- `PuzzleScriptAdapter` cuts an episode off at 200
    presses, and a level that crept over would only show up as a level quietly
    going missing from the corpus."""
    import time
    solver = CleanUpSolver()
    game = solver.make_game(0)
    expert = CleanUpExpert(game, node_cap=solver.node_cap)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        started = time.time()
        found = expert.plan(game._engine, level)
        seen, dist, opt = expert._fields[level]
        if found is None:
            print(f"level {level}: NO PLAN ({len(seen)} states reachable)")
            continue
        # Re-walk the plan to count the steps with a real choice.
        start = snapshot(game._engine)
        ties = 0
        for direction in found:
            ties += len(expert.optimal_set(game._engine, level) or [1]) > 1
            game._engine.step(direction)
        restore(game._engine, start)
        total += len(found)
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level}: {len(found):3d} presses "
              f"(budget {game._max_steps}, {room}), "
              f"{len(seen):6d} states, {len(dist):6d} winnable, "
              f"{ties:3d} steps with a tie set "
              f"({ties / max(1, len(found)):.0%}), {time.time() - started:.1f}s")
    print(f"total {total} presses over {game.n_levels} levels")
    return 0


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    sys.exit(CleanUpSolver.main())
