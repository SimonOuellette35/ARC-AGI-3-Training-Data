"""Generate Phase-1 training data for the PuzzleScript game ps:bridge
("Bridge", by gamez7).

The harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: the mechanic's model of
what a push costs, the goal heuristic, and the sprite/render fixes the game
needed before any of it was worth taping.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_bridge",
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
Sokoban with a consumable resource. Three rules and one win condition::

    [ > Player | Crate ] -> [ > Player | > Crate ]     # ordinary single push
    [ > Player | Lava  ] -> restart                    # walking into lava kills
    [ > Crate  | Lava  ] -> [ | Dummy ]                # a crate FILLS one lava
    WINCONDITIONS: Some Crate on Pad

The third rule is the whole game. A crate shoved into lava is *destroyed* and
the lava cell it hit becomes a ``Dummy`` -- a walkable, pushable-over bridge
tile. So a crate is not a block to be delivered, it is one of two things: the
one crate that ends on the pad, or a plank spent to open a route. Every level
is therefore a budget problem on top of a Sokoban: with ``n`` crates you may
bridge at most ``n - 1`` lava cells, and the box that reaches the pad has to
travel a route made of the bridges you already paid for.

Three consequences drive everything below:

  * **The player must never step on lava.** ``restart`` reloads the level (the
    adapter does it inside `perform_action`), which mid-plan would silently
    rewind the recording. Lava is in `blocker_names`, so the walk BFS behind
    every macro simply refuses to route through it.
  * **A crate cannot be pushed *through* lava either** -- it dies there. It can
    only cross a cell that some earlier crate already turned into a Dummy.
  * **Bridges are irreversible.** Spend a crate on the wrong cell and the level
    is dead with no visible sign of it, which is exactly what `dead` is for.

Four levels, of which this generator records the two it can solve:

  * **Level 0 "push"** -- a plain Sokoban corridor; the lava is scenery. 14
    moves.
  * **Level 1 "bridge"** -- the teaching level. A 3-cell lava channel is the
    only door between the two halves of the board, there are exactly 4 crates,
    and the crate that wins is the one already sitting on the far side. So
    three crates are planks and the fourth is the delivery: zero slack. 40
    moves.
  * **Level 2 "journey"** and **level 3 "ocean"** are in `skip_levels` -- see
    that attribute for what was tried and why they are out of reach.

Expert solver
-------------
`PSPushExpert`: A* over the REAL interpreter whose successors are ``walk to the
push cell, then push`` MACROS, costed in primitive moves. Three overrides make
it fit this game:

  * **The state key carries the bridges.** `PSPushExpert` keys a state on the
    pieces plus the player's reachable region, which assumes the terrain is
    static per level. Here the terrain IS the puzzle -- every push into lava
    rewrites it -- so `_region_key` and `_key` both take the lava set in as
    well. Without that, two states with identical crates and an identical
    player region but different bridges alias, and the search serves one's plan
    for the other.
  * **A walk is a teleport.** `_apply_macro` normally replays every primitive
    of the walk through the interpreter, and the interpreter step is the entire
    cost of these searches. No rule here fires on a bare player move (each one
    needs a crate or lava in front), and the walk provably ends on neither, so
    the walk's whole effect is "the player is somewhere else": the override
    relocates it in the grid and steps the engine ONCE, for the push. Same
    resulting state, ~3x less wall clock on level 1.
  * **`dead`** prunes the budget failures the heuristic alone would only sink:
    no crates left, no crate that can still reach the pad, or a pad route whose
    remaining lava needs more planks than there are crates.

The heuristic is a backward Dijkstra from the pad over PUSH edges, charging
``lava_fill`` for every lava cell the box still has to cross (i.e. the plank it
will cost), plus the player's own walk to that crate under the same lava
charge. Both halves matter and they measure different things: on level 1 the
winning crate is 2 pushes from the pad and 3 bridges away from the *player*, so
a box-only estimate reports 2 for a 40-move state and the search degenerates to
uniform cost. Tables are memoized per lava configuration, which is what makes a
per-node Dijkstra affordable.

``lava_fill`` is a real dial, not a constant to leave alone: it is what a plank
costs, and it is the only thing telling the search that the far bank is far.
Measured on level 1 (level 0 has no lava on any route and is 0.03s at every
setting)::

    lava_fill    4     6     8    10    14    20
    h at start  22    28    34    40    52    70
    seconds     66    29   9.4   4.2  0.51  0.81

Every setting returns the same 40-move plan, and 40 is provably shortest: at
``lava_fill = 0`` the heuristic degenerates to "walk to a crate, then push it to
the pad, tolls waived", which is a genuine lower bound (the player must reach
the winning crate before it can move, and the crate must then be pushed), and
weight-1 A* under it returns 40 as well. 10 is kept because it is where the
estimate stops being a guess: h at the start of level 1 is exactly the 40 moves
the level costs. The 8x left on the table at 14 buys nothing -- this is a
one-time search at seed 0 that every later seed replays from cache.

Labels
------
Most steps of a Bridge plan are not pushes, they are the player walking to the
next push cell, and a walk's order is free. `BridgeSolver.optimal_for` labels
each walk step with EVERY direction that keeps it on a shortest route to the
same cell, rather than with the one interleaving the BFS happened to emit; a
push is labelled with itself alone, because a sibling push is a different plan.
Verified against an oracle rather than by inspection: for all 54 expert steps,
taking any emitted label and re-solving the resulting state with the admissible
(``lava_fill = 0``) expert gives a completion exactly one move shorter, so every
label is genuinely optimal and no optimal alternative on a walk is missing.

Rendering (fixed in the game file before anything was recorded)
---------------------------------------------------------------
Per [[ps-palette-collisions]] the frame was checked before the mechanic was
trusted, and as shipped Bridge was unplayable from pixels alone. Cells are drawn
at ``cell_px = min(64 // H, 64 // W)``, which is 4 on levels 0-1, 3 on level 2
and **2** on level 3, and the sprite is point-sampled at cell_px positions --
so a sprite's detail does not shrink, it is *decimated*. Three fatal collisions
followed, all of them silent:

  * Crate was blue with transparent holes at exactly the four pixels cell_px=2
    samples, so **every crate on level 3 rendered as bare background**.
  * Dummy was ``DARKBLUE``, and darkblue and blue are both ARC index 9, so a
    crate standing on a bridge was pixel-identical to the bare bridge, on every
    level.
  * Pad was orange, the player's colour, and its dot pattern vanished entirely
    at cell_px=3 (level 2) -- the goal was invisible on the level that has the
    longest walk to it.

The objects were repainted and reshaped so that all eleven composites a cell can
actually hold -- background, wall, lava, dummy, pad, crate, crate/dummy,
crate/pad (the win), player, player/dummy, player/pad -- are pairwise distinct
at cell_px 2, 3 AND 4. Lava is solid red, Dummy solid grey (a filled-in lava
cell reads as stone), Pad a yellow X, Crate blue with holes on the
anti-diagonal and Player orange with holes on the diagonal, the two hole motifs
chosen so each punches through in all three sampling grids and the tile
underneath shows its own colour.

Augmentation
------------
Bridge's engine state after reset is identical for every seed (levels are fixed
ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, to which ``"Bridge"`` was added). The flips
are exact symmetries here: the game is gravity-free, its input is
screen-relative, and all three of its rules are written with the relative ``>``
so not one names a compass direction. With only two recordable levels that is
the difference between 8 presentations and 32.

The expert plan is therefore seed-independent: solved once per level, cached,
and replayed per seed with that seed's remapped screen actions. Seed 0 pays ~4s
for both searches; every later seed replays them from cache.

Validated: 12/12 seeds WIN both levels in 16s total, every recorded action
replays byte-exactly against a fresh adapter at the same seed, and two runs
`diff -rq` identical. Generation itself is the user's to run.

Usage (run from the repo root):
    python solvers/generate_bridge_training.py --episodes 200 \
        --out data/training_multi_level/bridge
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSPushExpert  # noqa: E402

GAME_NAME = "Bridge"

#: (dr, dc) per engine direction.
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))
_STEP = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

_INF = 1 << 30

#: Heuristic charge for a state that can no longer be won -- no crate left that
#: can reach the pad, or not enough crates left to plank the route there. Large
#: enough to sink the node, finite so the search stays complete. `dead` drops
#: most of these before they are ever queued; this is the value the rest see.
_DEAD = 10_000


class BridgeExpert(PSPushExpert):
    """Walk-and-push A* over the real interpreter, for a Sokoban whose crates
    are also the bridge planks. See the module docstring for the mechanic, the
    three overrides and the `lava_fill` measurement."""

    pushable_names = ("crate",)
    #: Lava blocks the PLAYER: stepping on it triggers the game's ``restart``,
    #: which reloads the level under the recorder. It does not block a push --
    #: shoving a crate into lava is the entire point of the game, and it leaves
    #: the player on the crate's old (safe) cell.
    blocker_names = ("wall", "lava")

    #: Primitive moves charged for each lava cell still on a route: what it
    #: costs to fetch another crate and shove it in. See the docstring -- this
    #: is the search's speed/quality dial, and 10 prices level 1 exactly.
    lava_fill = 10

    def setup(self) -> None:
        super().setup()
        g = self.g
        self.crate = g.obj_name_to_idx["crate"]
        self.lava = g.obj_name_to_idx["lava"]
        self.dummy = g.obj_name_to_idx["dummy"]
        self.pad = g.obj_name_to_idx["pad"]
        self.wall = g.obj_name_to_idx["wall"]
        # The terrain is mutable here, so it belongs in the plan memo's key
        # alongside the crates and the player (see the module docstring).
        self.dyn_ids = self.dyn_ids | {self.lava, self.dummy}
        self._table_cache: dict = {}
        self._level: int | None = None

    def plan(self, eng, level: int | None = None):
        """Bind the level before planning, then plan as usual.

        `_tables` is memoized on the lava set alone -- everything else it reads
        (the walls, the pad) is static -- but "static" only holds WITHIN a
        level: two levels are different boards at the same coordinates, so the
        memo has to be scoped by level the same way `plan`'s own plan memo is
        (`scope_by_level`). Recording the level here rather than threading it
        through `heuristic` keeps the per-node path free of it."""
        self._level = level
        return super().plan(eng, level)

    def _region_key(self, eng, region) -> tuple:
        crate, lava = self.crate, self.lava
        return (frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in cell if o == crate or o == lava), region)

    # -- board scan -----------------------------------------------------------
    def _scan(self, eng) -> tuple:
        """One pass over the grid for everything the heuristic needs.

        Returns ``(h, w, walls, islava, crates, player)``. Run per node, so it
        is a single loop rather than the four comprehensions it reads as."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        wall, lava, crate, players = (self.wall, self.lava, self.crate,
                                      self.player_ids)
        walls = [[False] * w for _ in range(h)]
        islava = [[False] * w for _ in range(h)]
        crates: list[tuple[int, int]] = []
        player = None
        for r in range(h):
            row = grid[r]
            wrow, lrow = walls[r], islava[r]
            for c in range(w):
                cell = row[c]
                if not cell:
                    continue
                if wall in cell:
                    wrow[c] = True
                if lava in cell:
                    lrow[c] = True
                if crate in cell:
                    crates.append((r, c))
                if cell & players:
                    player = (r, c)
        return h, w, walls, islava, crates, player

    # -- push-cost table (backward Dijkstra from the pad) ---------------------
    def _tables(self, eng, h, w, walls, islava) -> dict:
        """``{cell: (moves, planks)}`` -- the cheapest way to push a crate from
        ``cell`` onto the pad, and how many lava cells that route still crosses.

        Backward from the pad over push edges: a box reaches ``Y`` from
        ``X = Y - d`` with the player standing at ``Y - 2d``, so both of those
        must be on the board and not walls. Crossing a lava ``Y`` is allowed --
        that is what a bridge is for -- and charged ``lava_fill`` moves and one
        plank; the ``planks`` count is what `dead` tests the crate budget
        against. The cell the box STARTS on is never charged, which is right:
        a box never stands on lava.

        Memoized per (level, lava configuration). The table depends on nothing
        else (no rule in this game moves a wall or a pad), and the lava set only
        changes when a crate is spent, so a search that shuffles crates around
        reuses one table for thousands of nodes. The level has to be in the key
        because the walls are only static WITHIN one -- see `plan`."""
        key = (self._level,
               frozenset((r, c) for r in range(h) for c in range(w) if islava[r][c]))
        cached = self._table_cache.get(key)
        if cached is not None:
            return cached

        grid = eng.grid
        cost: dict[tuple[int, int], tuple[int, int]] = {}
        pq: list = []
        for r in range(h):
            for c in range(w):
                if self.pad in grid[r][c]:
                    cost[(r, c)] = (0, 0)
                    heapq.heappush(pq, (0, 0, (r, c)))
        fill = self.lava_fill
        while pq:
            moves, planks, y = heapq.heappop(pq)
            if cost.get(y, (_INF, _INF)) < (moves, planks):
                continue
            yr, yc = y
            toll, plank = (fill, 1) if islava[yr][yc] else (0, 0)
            for dr, dc in _DELTA:
                xr, xc = yr - dr, yc - dc            # where the box comes from
                sr, sc = yr - 2 * dr, yc - 2 * dc    # where the player stands
                if not (0 <= xr < h and 0 <= xc < w) or walls[xr][xc]:
                    continue
                if not (0 <= sr < h and 0 <= sc < w) or walls[sr][sc]:
                    continue
                nxt = (moves + 1 + toll, planks + plank)
                if nxt < cost.get((xr, xc), (_INF, _INF)):
                    cost[(xr, xc)] = nxt
                    heapq.heappush(pq, (nxt[0], nxt[1], (xr, xc)))
        self._table_cache[key] = cost
        return cost

    # -- player walk cost -----------------------------------------------------
    def _walk_cost(self, h, w, walls, islava, player) -> dict:
        """Dijkstra from the player over the board, charging ``lava_fill`` to
        enter a lava cell (it has to be bridged first) and ignoring crates (a
        relaxation -- the real walk routes around them).

        Not memoizable: the player moves every macro. It is the half of the
        heuristic that knows the player is on the wrong side of the water."""
        fill = self.lava_fill
        dist = {player: 0}
        pq = [(0, player)]
        while pq:
            d, cur = heapq.heappop(pq)
            if dist.get(cur, _INF) < d:
                continue
            r, c = cur
            for dr, dc in _DELTA:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or walls[nr][nc]:
                    continue
                nd = d + 1 + (fill if islava[nr][nc] else 0)
                if nd < dist.get((nr, nc), _INF):
                    dist[(nr, nc)] = nd
                    heapq.heappush(pq, (nd, (nr, nc)))
        return dist

    # -- heuristic ------------------------------------------------------------
    def heuristic(self, eng) -> int:
        h, w, walls, islava, crates, player = self._scan(eng)
        if not crates or player is None:
            return _DEAD
        cost = self._tables(eng, h, w, walls, islava)
        budget = len(crates)
        live = [(cost[c][0], c) for c in crates
                if c in cost and cost[c][1] + 1 <= budget]
        if not live:
            return _DEAD
        walk = self._walk_cost(h, w, walls, islava, player)
        best = _DEAD
        for pushes, (cr, cc) in live:
            # The player pays to reach a cell BESIDE the crate, not the crate's
            # own cell -- that is where a push is made from.
            reach = min(walk.get((cr + dr, cc + dc), _INF) for dr, dc in _DELTA)
            if reach < _INF:
                best = min(best, pushes + reach)
        return best

    def dead(self, eng) -> bool:
        """True once the level cannot be won any more.

        Three ways to get here, and none of them is visible on the board: the
        last crate was spent, every surviving crate has been shoved somewhere
        with no push route to the pad, or the cheapest surviving route still
        needs more planks than there are crates to spend. All three are
        permanent -- nothing in this game creates a crate or removes a bridge
        -- so the successor is dropped rather than queued.

        Deliberately cheaper than `heuristic`: it reuses the memoized push
        table and skips the player's Dijkstra, because it runs on every
        successor and most of what it kills is killed by the plank budget
        alone."""
        h, w, walls, islava, crates, player = self._scan(eng)
        if not crates or player is None:
            return True
        cost = self._tables(eng, h, w, walls, islava)
        budget = len(crates)
        return not any(c in cost and cost[c][1] + 1 <= budget for c in crates)

    # -- macro application ----------------------------------------------------
    def _apply_macro(self, eng, macro) -> int:
        """Run one ``walk + push`` macro, stepping the interpreter ONCE.

        Every rule in this game needs a crate or a lava cell in front of the
        player to fire, and `_analyze`'s walk is built over cells that hold
        neither, so replaying the walk through the interpreter can only ever
        move the player -- one grid edit expresses the whole of it. Only the
        final primitive (the push) is a real step. The win can only be reached
        by that step for the same reason (the win is a crate on the pad, and a
        walk moves no crate), so the returned index is the macro's last."""
        if len(macro) > 1:
            grid = eng.grid
            src = who = None
            for r, row in enumerate(grid):
                for c, cell in enumerate(row):
                    if cell & self.player_ids:
                        src, who = (r, c), cell & self.player_ids
                        break
                if src is not None:
                    break
            if src is None:                       # no player: nothing to walk
                return -1
            pr, pc = src
            for direction in macro[:-1]:
                dr, dc = _STEP[direction]
                pr, pc = pr + dr, pc + dc
            grid[src[0]][src[1]] -= who
            grid[pr][pc] |= who
            # Same invalidation `restore` does -- the engine caches a position
            # index and a per-rule no-op cache off the grid.
            eng._position_index_dirty = True
            eng._rule_noop_cache.clear()
        eng.step(macro[-1])
        return len(macro) - 1 if eng.check_win() else -1


class BridgeSolver(PSAStarSolver):
    game_id = "puzzlescript_bridge"
    game_name = GAME_NAME
    expert_cls = BridgeExpert

    #: ``games/ps:bridge/ps:bridge.py`` subclasses the adapter to raise the
    #: per-level step limit (the shipped 200 is under the plan length of the
    #: later levels). Recording through it means the frames are the ones a live
    #: agent sees.
    game_module_id = "ps:bridge"

    #: Unweighted: both recorded levels fall out in seconds together, and the
    #: 40-move level-1 plan is the same one an over-charged `lava_fill` finds,
    #: so there is nothing to buy by trading optimality away.
    weight = 1
    node_cap = 400_000
    #: Plans are 14 and 40 moves; the ceiling only has to cover a re-plan after
    #: an epsilon detour, and `epsilon` is 0 for this family.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Build the private engine `optimal_for` annotates plans on.

        It cannot borrow the recording adapter: `record_level` asks for a step's
        optimal set AFTER performing that step, so by the first call the live
        engine has already left the state the plan starts from, and re-seating
        the level to get it back would throw the recording away."""
        self._sim = self.make_game(0)
        self._annotations: dict = {}

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The optimal ENGINE-direction SET at plan step ``pi``.

        Most of a Bridge plan is not pushes, it is the player WALKING to the
        next push cell, and a walk's order is free -- see
        `PSPushExpert.annotate_walks`, which is the whole labeller. All this
        adds is the private engine to replay on and a memo, because
        `record_level` asks per step and the answer is per plan."""
        key = (level, tuple(plan))
        ann = self._annotations.get(key)
        if ann is None:
            self._sim.set_level(level)
            ann = self._annotations[key] = expert.annotate_walks(
                self._sim._engine, plan)
        return ann[pi]

    #: Levels 2 ("journey") and 3 ("ocean"). Skipped up front so discovery does
    #: not burn the node cap on them at every startup.
    #:
    #: They are not "hard for the search", they are ZERO-SLACK budget puzzles,
    #: and that is what defeats it. Splitting level 2's board into land
    #: components (non-wall, non-lava) gives four: the player and 5 crates in a
    #: 24-cell pocket, 5 more crates in a 40-cell middle, a 2-cell landing, and
    #: the pad alone on a 20-cell shore with no crate on it at all. The narrowest
    #: lava crossings between them are 4, 3 and 2 cells, so the player alone
    #: needs 9 planks, and the crate that ends on the pad has to be pushed over
    #: every one of them -- 9 spent + 1 delivered = 10, against exactly 10
    #: crates. There is no move to spare anywhere in a ~50-push solution: shove
    #: any crate one cell wrong and the level is lost with nothing on screen to
    #: say so. A macro search has to find a near-unique sequence, and `dead`
    #: only starts paying once the budget is already tight.
    #:
    #: Measured, all on level 2 unless noted, none of them finishing and none of
    #: them near the node cap: A* at ``lava_fill`` 20/30/60 (weight 1 and 2,
    #: caps to 4M nodes) for 20-23 min each; `PSBeamExpert` at width 400 x depth
    #: 80 for 25 min; and level 3 ("ocean" -- 185 lava cells, 14 crates, the
    #: player and 10 crates walled into one corner) under a width 200 x depth
    #: 120 beam for 23 min. Cracking either wants a staged planner that treats
    #: "feed N crates into this corridor" as one move, not a better heuristic
    #: over single pushes.
    skip_levels = frozenset({2, 3})


if __name__ == "__main__":
    sys.exit(BridgeSolver.main())
