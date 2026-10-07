"""Generate Phase-1 training data for the PuzzleScript game
ps:add_man_2_this_time_its_arithmetical ("Add Man 2: This Time It's Arithmetical").

The harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: its name, the
walk-and-push search it uses, and its goal heuristic.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_add_man_2",
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
action (post rotation-remap), i.e. the button an agent presses in the rotated
view, so replaying the recorded actions reproduces the recorded frames exactly.

The game
--------
Adam Gashlin's Add Man 2 is Sokoban with BINARY ARITHMETIC. The Player pushes
white digit tiles -- Zero (a small ring) and One (a vertical bar) -- around a
walled board and must cover every blue target with a digit of the target's own
value (``All ZeroGoal on Zero`` / ``All OneGoal on One``). Only the four
directions do anything; the game declares ``noaction``, so ACTION5 is a no-op
and the search never branches on it.

Pushing one digit into another does not shove it along -- it ADDS:

    0+0 -> 0     0+1 -> 1     1+0 -> 1     1+1 -> 10 (carry!)

so the pushed digit is *consumed* and the stationary one takes the sum. The
carry is the mechanic the title is about: ``1 + 1`` leaves a Two, and a Two
resolves by writing its carry into the cell to its LEFT (``Right [One | Two] ->
[Two | Zero]`` propagates the carry leftward through a run of Ones; ``Right [No
One | Two] -> [One | Zero]`` terminates it, turning the left neighbour into a
One and the Two itself into a Zero). Any Two that cannot resolve -- one against
the left edge -- hits ``[Two] -> Cancel`` and the whole turn is undone, so such
a push is simply a no-op.

Two consequences drive the expert:

  * **Digits can be pushed into walls, the Player cannot walk into them.**
    Walls share a collision layer with the targets, not with the digits ("MESSAGE
    Numbers can go into walls"), so a digit shoved onto a wall cell stays there
    and can never be pushed again -- the Player would have to stand on the wall
    to push it. It is a permanent trap, and the search has to find it that way.
  * **A digit can also be CREATED where no push can reach.** Level 6's target is
    walled off on every side a push could come from; it is won by parking a One on
    the wall cell beside it and pushing a second One into that, so the resulting
    Two's carry writes a One into the pocket. A push-only reachability model calls
    that level unsolvable, which is why the heuristic's distance tables carry a
    carry edge (see `AddMan2Expert`).

Expert solver
-------------
`PSPushExpert`: A* over the REAL engine dynamics whose successors are
``walk to the push cell, then push`` MACROS rather than single key presses. A
push is the only move with any effect here, so the primitive search burns its
whole budget re-deriving walks -- most of level 2's 50-move solution is the
Player walking between pushes. Costs stay in primitive MOVES (the unit the agent
pays) and the emitted plan is a flat list of directions, so the recorder is
unchanged; see `PSPushExpert` for the region-canonicalisation that makes the
dedup bite.

Solvable levels
---------------
13 of the 15. The interpreter reproduces every level; what bounds the set is
search budget, not mechanics. The kept levels are re-discovered once at startup
by `ps_astar.discover_solvable`, and that holds for every seed (engine state
after reset is seed-independent). All 13 searches together cost ~90s at seed 0
(3s or less each bar level 3 at ~23s and level 12 at ~55s); every later seed
replays them from cache.

Levels 9 and 14 are in ``skip_levels``: both need several digits funnelled to the
SAME cell before any of it pays off -- level 9's target is walled in on every
side a push could come from, so its One has to be written by a carry that first
marched one cell left through another One, which takes all four of the board's
Ones delivered through one gap. The heuristic estimates from the nearest single
digit and so gives no gradient for delivering the other three, and the search
had not won either board after 25 minutes. They are skipped up front rather than
retried (and then fruitlessly re-searched) at every startup.

Augmentation
------------
Add Man 2's engine state after reset is identical for every seed (levels are
fixed ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group. The game is gravity-free and its inputs are
screen-relative moves, so a flip is an exact symmetry; the carry rule is
directional but it acts on the BOARD, not on the input, so mirroring it is no
different from what the rotation already does. There is deliberately no RECOLOR
augmentation: the recolor surfaces flatten an object to a single flat colour, and
a digit here is read by SHAPE against a target drawn in the same shape -- Zero is
a ring and One a bar, and flattening would erase exactly that distinction.

The expert plan is therefore seed-independent: solved once per level, cached, and
replayed per seed with that seed's remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_add_man_2_training.py --episodes 200 \
        --out data/training_multi_level/add_man_2
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSPushExpert  # noqa: E402

GAME_NAME = "Add_Man_2__This_Time_It's_Arithmetical"

#: (dr, dc) per engine direction, for the distance tables.
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Heuristic charge for a state that can no longer be won (every digit consumed
#: while a target is still bare). Large enough to sink such a node to the back of
#: the queue, finite so the search stays complete.
_DEAD = 1000


class AddMan2Expert(PSPushExpert):
    """Walk-and-push A* whose heuristic is a per-target REACHABILITY table.

    A plain "count the bare targets" heuristic is worthless here: the boards are
    corridors, and what costs moves is the detour around a wall, not the number
    of targets. So for each target this precomputes ``steps[cell]`` -- the
    minimum number of pieces-moves to bring a digit from ``cell`` onto that
    target, ignoring the other digits (the standard sokoban push-distance
    relaxation) -- by a reverse Dijkstra over two kinds of edge:

      * a PUSH lands a digit on ``X`` from ``X - d`` (cost 1) provided ``X - d``
        is not a wall (a digit on a wall can never be pushed again) and the
        Player's stand cell ``X - 2d`` is on the board and not a wall. Note ``X``
        itself may be a wall: shoving a digit into one is legal, it just strands
        it there.
      * a CARRY writes a digit into ``X`` from a Two at ``X + 1``, so ``X + 1`` is
        a predecessor of ``X`` too. Without this edge level 6 -- whose target is
        reachable ONLY by carry -- looks unreachable and the level is declared
        unsolvable. It costs 2, not 1, because forming that Two takes a SECOND
        One delivered into the same cell, and it is only offered where a Two can
        really appear (some push reaches ``X + 1``, or the carry cascades in from
        further right along the row): an unconditional carry edge walks through
        solid wall and deflates the whole table back to manhattan distance. Level
        9's target needs the cascade specifically -- nothing can be pushed into
        the cell beside it either, so its One is written by a carry that first
        marched one cell left through a One.

    The estimate is then, per bare target, the nearest digit's table entry plus
    one if that digit has the wrong value (it needs at least one arithmetic step
    to become the right one), and finally the Player's walk to the nearest digit.
    Summing over targets double-counts a digit that could serve two, and the
    carry edge undercounts (a carry needs a SECOND One delivered to the same
    cell), so it is not admissible in either direction -- like the rest of this
    family it is a search guide, not an optimality certificate, and the plans it
    returns are near-optimal genuine WIN paths.
    """

    pushable_names = ("zero", "one", "two")
    blocker_names = ("wall",)

    def setup(self) -> None:
        super().setup()
        g = self.g
        self.zero = g.obj_name_to_idx["zero"]
        self.one = g.obj_name_to_idx["one"]
        self.two = g.obj_name_to_idx["two"]
        self.zerogoal = g.obj_name_to_idx["zerogoal"]
        self.onegoal = g.obj_name_to_idx["onegoal"]
        self._table_cache: dict = {}
        self._tables: dict = {}

    # -- per-target distance tables ------------------------------------------
    def plan(self, eng, level: int | None = None):
        """Bind the distance tables for this board, then plan as usual.

        The tables depend only on the walls and targets, which no rule ever
        touches, so they are built ONCE per layout here rather than inside
        `heuristic` -- which runs on every node and must not re-scan the board
        for static facts."""
        self._tables = self._build_tables(eng)
        return super().plan(eng, level)

    def _build_tables(self, eng) -> dict:
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        wall = tuple(tuple(bool(cell & self.blocker_ids) for cell in row)
                     for row in grid)
        targets = tuple(
            (r, c)
            for r, row in enumerate(grid)
            for c, cell in enumerate(row)
            if (self.zerogoal in cell or self.onegoal in cell)
        )
        sig = (wall, targets)
        cached = self._table_cache.get(sig)
        if cached is not None:
            return cached

        # Where a Two can appear -- the source of a carry into the cell on its
        # left. Either some push lands a second digit there, or a Two further
        # right walks in: `Right [One | Two] -> [Two | Zero]` marches the carry
        # leftward through a run of Ones, so a single pushable cell makes every
        # cell to its left in that row a possible Two. Scanned right to left,
        # which is the direction the cascade travels.
        def pushable_into(r: int, c: int) -> bool:
            for dr, dc in _DELTA:
                pr, pc = r - dr, c - dc                     # the digit's cell
                sr, sc = pr - dr, pc - dc                   # the Player's cell
                if (0 <= pr < h and 0 <= pc < w and not wall[pr][pc]
                        and 0 <= sr < h and 0 <= sc < w and not wall[sr][sc]):
                    return True
            return False

        carry_ok = []
        for r in range(h):
            row = [False] * w
            for c in range(w - 1, -1, -1):
                row[c] = pushable_into(r, c) or (c + 1 < w and row[c + 1])
            carry_ok.append(tuple(row))
        carry_ok = tuple(carry_ok)

        tables = {}
        for target in targets:
            # Dijkstra, not BFS: the carry edge costs 2 and the push edge 1.
            steps = {target: 0}
            frontier = [(0, target)]
            while frontier:
                cost, (r, c) = heapq.heappop(frontier)
                if cost > steps[(r, c)]:
                    continue
                preds = []
                for dr, dc in _DELTA:                       # push predecessors
                    pr, pc = r - dr, c - dc                 # the digit's cell
                    sr, sc = pr - dr, pc - dc               # the Player's cell
                    if not (0 <= pr < h and 0 <= pc < w) or wall[pr][pc]:
                        continue
                    if not (0 <= sr < h and 0 <= sc < w) or wall[sr][sc]:
                        continue
                    preds.append((cost + 1, (pr, pc)))
                if c + 1 < w and carry_ok[r][c + 1]:        # carry predecessor
                    preds.append((cost + 2, (r, c + 1)))
                for ncost, cell in preds:
                    if ncost < steps.get(cell, 1 << 30):
                        steps[cell] = ncost
                        heapq.heappush(frontier, (ncost, cell))
            tables[target] = steps
        self._table_cache[sig] = tables
        return tables

    # -- heuristic ------------------------------------------------------------
    def heuristic(self, eng) -> int:
        zero, one, two = self.zero, self.one, self.two
        zerogoal, onegoal = self.zerogoal, self.onegoal
        player = None
        digits: list[tuple[int, int, bool]] = []     # (r, c, is_one)
        bare: list[tuple[int, int, bool]] = []       # (r, c, wants_one)
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & self.player_ids:
                    player = (r, c)
                if zero in cell:
                    digits.append((r, c, False))
                elif one in cell or two in cell:
                    digits.append((r, c, True))
                if zerogoal in cell and zero not in cell:
                    bare.append((r, c, False))
                if onegoal in cell and one not in cell:
                    bare.append((r, c, True))
        if not bare:
            return 0
        if not digits:
            return _DEAD                            # nothing left to cover with
        tables = self._tables
        h = 0
        for gr, gc, wants_one in bare:
            steps = tables[(gr, gc)]
            h += min(steps.get((dr, dc), abs(gr - dr) + abs(gc - dc))
                     + (0 if is_one == wants_one else 1)
                     for dr, dc, is_one in digits)
        if player is not None:
            h += max(0, min(abs(player[0] - dr) + abs(player[1] - dc)
                            for dr, dc, _ in digits) - 1)
        return h


class AddMan2Solver(PSAStarSolver):
    game_id = "puzzlescript_add_man_2"
    game_name = GAME_NAME
    expert_cls = AddMan2Expert

    #: Build the adapter through the game folder, which recolors the target
    #: indicator -- see `PSAStarSolver.make_game`.
    game_module_id = "ps:add_man_2_this_time_its_arithmetical"

    #: The two boards the macro search cannot win inside ``node_cap`` (neither
    #: had won after 25 minutes / 300k nodes, where every kept level lands in
    #: under 90s). Skipped up front so startup discovery does not spend that
    #: budget on them at every run. See "Solvable levels" for why they are hard.
    skip_levels = frozenset({9, 14})

    node_cap = 300_000
    #: Heavily weighted. The family only needs a genuine WIN path, not a shortest
    #: one, and the weight is what keeps the symmetric boards tractable: level 7
    #: (two mirrored target pockets, six interchangeable Ones) does not finish at
    #: weight 8 and lands in ~3s at 20, for plans a few moves longer elsewhere.
    weight = 20
    max_steps = 200


if __name__ == "__main__":
    sys.exit(AddMan2Solver.main())
