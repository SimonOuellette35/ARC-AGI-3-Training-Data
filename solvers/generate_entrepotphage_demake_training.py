"""Generate Phase-1 training data for the PuzzleScript game
ps:entrepotphage_demake ("EntrepotPhage Demake" by Xavier Direz -- a 30-level
warehouse Sokoban dressed in fake 3-D shading).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, a native model of the interpreter, the search over it, and the
optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_entrepotphage_demake",
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
presented view, so replaying the recorded actions reproduces the recorded frames
exactly. Each step also carries the full set of equally-optimal presses.

The game
--------
Textbook Sokoban -- one pushing rule and a positional win:

    [ > Player | Crate ] -> [ > Player | > Crate ]        (the whole mechanic)
    All Target on Crate                                    (the whole goal)

Player, Wall and Crate share a collision layer, so a crate shoved into a wall or
into a second crate cancels the turn and pushes never chain. ACTION5 is bound to
nothing, so the four directions are the entire action space.

Two header flags matter:

  * ``require_player_movement`` -- a press that does not move the player cancels
    the WHOLE turn, so walking into a wall (or into a crate that cannot move) is
    a true no-op, not a turn spent. The adapter agrees: it only charges *moved*
    presses against the per-level step budget.
  * ``run_rules_on_level_start`` -- the decoration rules below fire once before
    the first press, which is why a level's first frame already carries its
    shading.

Everything else in the file is DECORATION, and this is the one thing worth
knowing about the game:

  * ``CrateOffTarget`` / ``CrateOnTarget`` are the same crate in two colours.
    ``[> Crate | Target] -> [> CrateOnTarget | Target]`` recolours a crate the
    turn it is pushed onto a target and its sibling recolours it back on the way
    off. The win condition reads the ``Crate`` or-group, so this is purely how
    the board LOOKS -- brown crate on grey floor, green crate on a green target.
  * ``WallUp`` / ``WallDown`` / ``CrateOnDown`` / ``CrateOffDown`` /
    ``CrateOnDownFlag`` / ``CrateOffDownFlag`` are drop shadows: thin bands
    painted into the cell above or below a wall or a crate to fake a 3-D
    warehouse. Six of the game's thirteen rules delete and re-lay them every
    single turn, over the whole board.
  * ``PlayerEars`` is a two-pixel decal kept in the cell above the player by
    another three rules.
  * ``vide`` ("empty") is the black filler outside the wall ring. It sits on its
    own collision layer, so the engine would happily let the player walk onto it
    -- but in all 30 levels the wall ring seals the board completely and no vide
    cell is ever adjacent to a walkable one (``--selfcheck`` re-verifies this),
    so the model treats it as wall and loses nothing.

Why a native model
------------------
Those decoration rules are what makes this game expensive to SEARCH. Each of the
thirteen rules is matched against the whole board on every turn, so one
``eng.step`` here costs about 2.1 ms against 0.046 ms for Count Mover and 0.040 ms
for Bad Example -- the same one-rule mechanic with none of the shading. Measured
at the level starts: **467 engine steps/s here vs 21757 and 24871 there**, a 47x
tax for redrawing shadows a search never looks at.

A `PSSokobanExpert`, which searches by stepping the real interpreter, therefore
runs ~50x slower on this game than on the mechanically identical ones it was
written for, and these are BIG sokoban boards (up to 13 crates). So this
generator follows [[dotsnake]]'s route instead: read the board off the engine
once, search a native model of it, and let the interpreter only CERTIFY the
result -- `record_level` replays every press through the real adapter and keeps
the episode only if it reaches ``GameState.WIN``, and ``--plans`` replays each
plan on its own. `Board` is a few hundred lines of pure python and expands
macro nodes several thousand times a second on the same boards.

The model is `Board`: walls (Wall or vide), targets, crate cells, player cell.
Nothing else exists in it -- not the shading, not the ears, not which of the two
colours a crate is currently wearing. ``--selfcheck`` is the argument that this
is safe: up to 12000 random presses across all 30 levels (10 rollouts each,
stopped early on an accidental win), comparing the player cell, the crate cells
and the win flag against the interpreter after every single one, with zero
divergence.

The search
----------
A* whose successors are ``walk to the push cell, then push`` MACROS, with ``g``
counted in PRIMITIVE MOVES (the cost the agent actually pays), so plans are
shortest in presses rather than in pushes.

**Dedup is on the exact ``(player cell, crates)`` pair, and that is not a
detail.** The standard sokoban canonicalisation -- key the player by its
REACHABLE REGION, as `PSPushExpert` does -- is sound when cost is counted in
PUSHES but *unsound* when it is counted in moves: two states with the same
crates and the same player region admit the same futures but not at the same
price, and merging them keeps whichever arrived cheaper rather than whichever
stands closer to the next push cell. Measured on this game it cost 11 moves
across 11 levels, and on level 3 it returned a 38-move plan where 37 is
achievable. Exhaustive breadth-first search over primitive moves (`--truth`)
confirms 37, and confirms the exact-dedup search on every level small enough to
enumerate.

The heuristic is the classical one, in units of moves:

  * per-target PUSH-DISTANCE tables, a reverse BFS from each target over push
    edges (a crate reaches ``X`` from ``X-d`` only when ``X-d`` and ``X-2d`` are
    both on the board and not walls), so distance is a crate's DETOUR round the
    walls, not its separation;
  * a min-cost perfect ASSIGNMENT (Hungarian) of uncovered targets to
    off-target crates over those tables -- not the greedy nearest-crate match
    `PSSokobanExpert` uses, which overcounts and is what makes that expert's
    plans "shortest in practice but not certified";
  * plus the player's walk to the nearest crate, minus one (it pushes from
    beside a crate, not from its cell). Those moves are disjoint from the pushes
    the assignment counts, so the sum stays a lower bound.

Both terms are lower bounds, so at ``weight = 1`` and with the goal test taken
on POP rather than on generation, the plans are optimal. The assignment is
memoised on the crate set, which is where most of the search's time went before.
``--truth`` checks that claim against exhaustive breadth-first search over
primitive moves: all 8 boards small enough to enumerate (up to 935k states)
match the A* plan exactly.

Proving a plan shortest is not affordable on every board, though. Eight of the
30 are open rooms with 7 to 12 crates where the admissible search cannot close
inside the node cap, so `EntrepotExpert.weights` escalates -- 1, then 2, 3, 5,
10 -- and takes the first rung that returns. Six of the eight fall to a greedy
rung almost immediately (level 9 has twelve crates and takes 48 expansions at
weight 3, against 2M generated nodes and no answer at weight 1); those plans are
engine-verified wins but not certified shortest, and ``--plans`` names the rung
each level used. The last two do not fall at all -- see
`EntrepotSolver.skip_levels`.

Two sound prunes carry the dense boards:

  * **dead cells** -- a cell absent from every target's push table is a cell no
    crate can ever leave (a corner, a pocket with no room to stand behind it), so
    a push into one is dropped. The tables give this away for free;
  * **freeze deadlock** -- the standard recursive test: a crate is frozen when
    both its axes are blocked, where an axis counts as blocked by a wall or by a
    neighbouring crate that is itself frozen (with the crate under test treated
    as a wall inside the recursion, which is what makes the recursion terminate
    and stay sound). A state with a frozen off-target crate can never be won.
    Only the pushed crate and its four neighbours are tested -- a push can freeze
    those and nothing else.

Optimal-action sets
-------------------
Most of a sokoban solution is the player WALKING to the next push cell, and a
walk's order is free: no rule fires on a bare move onto an empty cell, so every
interleaving of the two axes that stays on a shortest route to the same cell
costs the same and leaves an identical board. `Board.optsets` labels each walk
step with every direction that keeps it on a shortest route to the run's
destination, and each push with itself alone (which crate to shove where is the
puzzle, and a sibling push is a different plan, not a reordering of this one).
No step ever ships unlabelled.

``--ties`` audits the labels the expensive way: it re-solves the board from the
state each of the four presses leads to and compares the totals. Every labelled
press has to be genuinely optimal, and across the checked levels every one is.
The reverse does not hold and is not meant to: an optimal press that is NOT
labelled is a sibling plan (a different crate shoved somewhere else for the same
total), and labelling those would mean enumerating every optimal solution of a
sokoban level rather than the order-free stretches of one of them. Ties are
scarce here -- 144 of 5280 recorded expert steps in a 3-seed run -- because
these boards are corridors and pockets, where a shortest walk is usually the
only walk.

The levels
----------
28 of the 30 are recorded. ``moves`` is the expert plan the interpreter walked
to a WIN, ``w`` the heuristic weight it was found at (1 = shortest), and
``budget`` the level's GAME_OVER limit from `games/ps:entrepotphage_demake`:

    level  size    crates  moves  w  budget      level  size    crates  moves  w  budget
    0       7x10    4       17    1     200      15      9x10    3       52    1     200
    1       9x12    4       62    1     200      16     10x7     5       33    1     200
    2       9x8     4       46    1     200      17     12x9     6       52    1     200
    3       9x7     3       37    1     200      18     12x10    6      180    1     540
    4      11x12    4       30    1     200      19     12x10    5       58    1     200
    5       7x9     4       39    1     200      20     12x11    7      190    3     570
    6      11x7     6       60    1     200      21     10x11    3       93    1     279
    7      10x11    7       65    2     200      22     11x8     7       82    1     246
    8      11x9    11       38    2     200      23      9x12    8       72    3     216
    9      12x10   12       70    3     210      24     11x12   10     SKIPPED
    10      9x8     3       38    1     200      25     12x7     7       43    1     200
    11      8x9     3       48    1     200      26     10x8     6       46    1     200
    12      9x8     4       35    1     200      27     10x8     3       41    1     200
    13     11x11    4       51    1     200      28     12x10    7       68    1     204
    14     11x15    9      114    5     342      29     12x13   11     SKIPPED

1760 moves over the 28. Every level is well-formed -- crates and targets match
and no crate starts on a cell it can never leave -- including the two skipped
ones, which are simply harder than the search.

Augmentation
------------
The engine state after reset is identical for every seed -- the levels are fixed
ASCII maps -- so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to). Sokoban is
gravity-free, its one rule is stated with the relative ``>`` force, its win
condition is positional and its input is screen-relative, so a flip is an exact
symmetry of the mechanic. No sprite is chiral enough to matter: the player's
mirror is still plainly the player and lands on no other object's art, and the
shading bands stay consistent with each other because the whole frame is
transformed as one. There is deliberately no colour augmentation -- crate-on-
target is read as a solid green block against the target's green dot ring, which
a flattening recolor would erase.

The expert plan is therefore seed-independent: solved once per level, cached to
``data/entrepotphage_demake_plans.json``, and replayed per seed with that seed's
remapped screen actions. Cold, filling that cache is ~25 minutes, nearly all of
it the weight-1 searches that fail on the eight hard boards before a greedier
rung wins; warm, a run starts in a second. **Run ``--plans`` once before
sharding with `parallelize_generator`** -- otherwise every shard re-derives the
same plans on its own core, and a search at the node cap holds close to a
gigabyte.

Verified: 28/28 levels WIN, three seeds regenerated byte-identically
(``diff -rq``), and all 5322 recorded actions replay through
``games/ps:entrepotphage_demake``'s adapter to frames identical to the taped
ones, ending in ``GameState.WIN``. Every one of the 5280 expert steps carries an
``optimal`` set. All 16 presentations appear across the first 20 seeds.

The palette
-----------
Nothing needed fixing, which is worth stating because it usually does. Wall
``#4080e0`` -> blue 9, floor ``DARKGREY`` -> 3, crate ``#8d683d`` -> maroon 13,
target ``Green`` -> 14, player ``#ccc133`` -> yellow 11. Target and CrateOnTarget
do collide on green 14, but they are different SHAPES -- a sparse dot ring versus
a solid block -- and it is the composition, not the colour, that has to stay
readable; ``--audit`` checks exactly that, at every cell size the 30 boards
render at (4, 5, 6 and 7 px, all clean). ``vide`` (black 5) matches the letterbox border, which is harmless
and arguably right: it is the dead margin outside the warehouse.

Usage (run from the repo root):
    python solvers/generate_entrepotphage_demake_training.py --episodes 200 \
        --out data/training_multi_level/entrepotphage_demake

    python solvers/generate_entrepotphage_demake_training.py --plans      # level report
    python solvers/generate_entrepotphage_demake_training.py --selfcheck  # model fuzz
    python solvers/generate_entrepotphage_demake_training.py --audit      # rendering audit
    python solvers/generate_entrepotphage_demake_training.py --truth      # optimality check
    python solvers/generate_entrepotphage_demake_training.py --ties       # label check
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (PuzzleScriptAdapter,       # noqa: E402
                                           _render_cell_sprite)
from solvers.common.ps_astar import (Plan, PSAStarSolver,             # noqa: E402
                                     PSExpert)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "EntrepotPhage_Demake"

#: Disk cache of the per-level start plan AND its optimal-action sets. The
#: searches are seed-independent but cost minutes of A* on the dense boards,
#: which every shard of `parallelize_generator` would otherwise repeat on every
#: core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "entrepotphage_demake_plans.json"

DIRS = ("up", "down", "left", "right")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}

#: Stands in for "unreachable" in the assignment matrix, and for the heuristic of
#: a state no assignment can cover. Finite, so the search stays complete; far
#: above any real board distance, so it never competes with one.
INF = 1 << 20


# ---------------------------------------------------------------------------
# Min-cost perfect assignment
# ---------------------------------------------------------------------------

def hungarian(cost: list[list[int]]) -> int:
    """Minimum-cost perfect assignment of an ``n x n`` matrix (JV, O(n^3)).

    Written out rather than imported so the generator has no scipy dependency
    and so its output is bit-reproducible across environments. ``n`` is the
    number of uncovered targets, at most 13 on these boards, and the result is
    memoised per crate-set, so the cubic term never shows up in a profile."""
    n = len(cost)
    if n == 0:
        return 0
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            row = cost[i0 - 1]
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return sum(cost[p[j] - 1][j - 1] for j in range(1, n + 1))


def bits(mask: int) -> list:
    """The cell ids set in a crate mask, ascending."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class Board:
    """A native sokoban board read off the interpreter: walls, targets, crates,
    player. Cells are integers ``r * w + c``; ``nbr[d][cell]`` is the neighbour
    in direction ``d`` or -1 off the board.

    It is both the SIMULATOR (`step`, fuzzed against the interpreter by
    `selfcheck`) and the SEARCH SPACE (`solve`). See the module docstring for
    why the interpreter itself is too slow to search."""

    def __init__(self, h: int, w: int, walls, targets, crates, player) -> None:
        self.h, self.w = h, w
        n = h * w
        self.wall = [False] * n
        for (r, c) in walls:
            self.wall[r * w + c] = True
        self.nbr = {}
        for d, (dr, dc) in _DELTA.items():
            arr = [-1] * n
            for r in range(h):
                for c in range(w):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        arr[r * w + c] = rr * w + cc
            self.nbr[d] = arr
        # Crate SETS are bitmasks over cell ids, not python sets: a search
        # state is (player cell, crate mask) and a million of them are held at
        # once, where a frozenset of 11 cells costs ~700 bytes against ~40 for
        # the int. The same board searched with frozensets peaked at 2.4 GB,
        # which is not a size `parallelize_generator` can run one of per core.
        self.target_cells = sorted(r * w + c for (r, c) in targets)
        self.targets = 0
        for cell in self.target_cells:
            self.targets |= 1 << cell
        self.crates0 = 0
        for (r, c) in crates:
            self.crates0 |= 1 << (r * w + c)
        self.player0 = player[0] * w + player[1]

        # Per-target push-distance tables, and the union of their keys: the
        # cells from which SOME target is reachable at all. Both are static --
        # no rule in this game moves a wall or a target.
        self.tables = {t: self._push_table(t) for t in self.target_cells}
        self.live = 0
        for tab in self.tables.values():
            for cell in tab:
                self.live |= 1 << cell
        self._hcache: dict = {}
        self.nodes = self.expansions = 0

    # -- static tables --------------------------------------------------------
    def _push_table(self, target: int) -> dict:
        """``{cell: pushes needed to get a crate from cell onto target}``.

        Reverse BFS over push edges, ignoring the other crates (the standard
        relaxation). A crate reaches ``cell`` from ``src = cell - d`` only if
        ``src`` is on the board and not a wall AND ``stand = src - d`` is too --
        the player has to have somewhere to push from."""
        steps = {target: 0}
        queue = deque([target])
        wall = self.wall
        while queue:
            cell = queue.popleft()
            cost = steps[cell]
            for d in DIRS:
                back = self.nbr[_OPP[d]]
                src = back[cell]
                if src < 0 or wall[src]:
                    continue
                stand = back[src]
                if stand < 0 or wall[stand]:
                    continue
                if src not in steps:
                    steps[src] = cost + 1
                    queue.append(src)
        return steps

    # -- simulation (what `selfcheck` fuzzes) --------------------------------
    def step(self, player: int, crates: int, d: str):
        """One press. Returns the settled ``(player, crates)``.

        A blocked press returns the state unchanged, which is exact rather than
        approximate: ``require_player_movement`` cancels the whole turn when the
        player does not move, and every rule in the game either fires on the
        push itself or only repaints decoration."""
        nxt = self.nbr[d][player]
        if nxt < 0 or self.wall[nxt]:
            return player, crates
        if crates >> nxt & 1:
            beyond = self.nbr[d][nxt]
            if beyond < 0 or self.wall[beyond] or crates >> beyond & 1:
                return player, crates
            return nxt, crates ^ (1 << nxt) | (1 << beyond)
        return nxt, crates

    def won(self, crates: int) -> bool:
        """``All Target on Crate``: every target cell holds a crate. Which of
        the two crate colours is standing there is a rendering fact."""
        return crates & self.targets == self.targets

    # -- reachability ---------------------------------------------------------
    def reach(self, player: int, crates: int) -> dict:
        """BFS tree over the cells the player may WALK on (no wall, no crate --
        walking into a crate is a push, which is a macro, not a walk), as
        ``{cell: (previous cell, direction) | None}``."""
        parent = {player: None}
        queue = deque([player])
        wall, nbr = self.wall, self.nbr
        while queue:
            cur = queue.popleft()
            for d in DIRS:
                nxt = nbr[d][cur]
                if (nxt >= 0 and nxt not in parent and not wall[nxt]
                        and not crates >> nxt & 1):
                    parent[nxt] = (cur, d)
                    queue.append(nxt)
        return parent

    @staticmethod
    def walk(parent: dict, cell: int) -> list:
        """Materialise the shortest walk to ``cell`` from a `reach` tree."""
        out = []
        while parent[cell] is not None:
            cell, d = parent[cell]
            out.append(d)
        out.reverse()
        return out

    # -- heuristic ------------------------------------------------------------
    def assignment(self, crates: int) -> int:
        """Min-cost matching of uncovered targets to off-target crates over the
        push-distance tables: a lower bound on the pushes still owed.

        Memoised on the crate set -- the player moves far more often than the
        crates do, and this is the expensive half of `heuristic`."""
        got = self._hcache.get(crates)
        if got is None:
            bare = [t for t in self.target_cells if not crates >> t & 1]
            if not bare:
                got = 0
            else:
                loose = bits(crates & ~self.targets)
                cost = [[self.tables[t].get(c, INF) for t in bare]
                        for c in loose]
                got = min(hungarian(cost), INF)
            self._hcache[crates] = got
        return got

    def heuristic(self, crates: int, player: int) -> int:
        """Lower bound on the MOVES left: the pushes the assignment owes, plus
        the walk to the nearest crate (the player pushes from beside one, hence
        the -1). The two counts are disjoint stretches of any solution, so the
        sum stays admissible -- and note the walk term looks at EVERY crate, not
        only the off-target ones: an optimal plan is free to start by shoving a
        crate off a target it is squatting on, and a bound that assumed
        otherwise would overshoot."""
        total = self.assignment(crates)
        if total >= INF or total == 0:
            return total
        w = self.w
        pr, pc = divmod(player, w)
        return total + min(abs(pr - divmod(c, w)[0]) + abs(pc - divmod(c, w)[1])
                           for c in bits(crates)) - 1

    # -- deadlock -------------------------------------------------------------
    def deadlocked(self, pushed: int, crates: int) -> bool:
        """True when some crate can provably never move again while off-target.

        Only the just-pushed crate and its four neighbours can have become
        frozen by the push, so those are the only ones tested."""
        suspects = [pushed]
        for d in DIRS:
            n = self.nbr[d][pushed]
            if n >= 0 and crates >> n & 1:
                suspects.append(n)
        for cell in suspects:
            if not self.targets >> cell & 1 and self._frozen(cell, crates, 0):
                return True
        return False

    def _frozen(self, cell: int, crates: int, ignore: int) -> bool:
        """The standard recursive freeze test: blocked on BOTH axes."""
        ignore = ignore | (1 << cell)
        return (self._axis_blocked(cell, "up", "down", crates, ignore)
                and self._axis_blocked(cell, "left", "right", crates, ignore))

    def _axis_blocked(self, cell: int, d1: str, d2: str, crates: int,
                      ignore: int) -> bool:
        """An axis is blocked by a wall on either side, or by a neighbouring
        crate that is itself frozen. Crates already under test count as walls
        (``ignore``) -- that is what terminates the recursion, and it is sound:
        a cycle of mutually blocking crates is exactly a jam."""
        wall, nbr = self.wall, self.nbr
        n1, n2 = nbr[d1][cell], nbr[d2][cell]
        if n1 < 0 or wall[n1] or n2 < 0 or wall[n2]:
            return True
        for n in (n1, n2):
            if crates >> n & 1 and (ignore >> n & 1
                                    or self._frozen(n, crates, ignore)):
                return True
        return False

    # -- search ---------------------------------------------------------------
    def solve(self, node_cap: int = 2_000_000, weight: int = 1) -> list | None:
        """A* over ``walk + push`` macros, costed in primitive moves. Returns a
        flat list of directions, or None if the node cap runs out first.

        The goal test is taken on POP, not when a winning successor is
        generated: with an admissible heuristic that is the difference between
        "a plan" and "the shortest plan", and it costs one extra pop.

        ``node_cap`` counts GENERATED nodes, not expansions. Expansions are the
        wrong dial for the same reason they are the cheap-sounding one: an
        eleven-crate board generates up to 44 successors per expansion, each of
        which lands in the heap, the closed set and the trail at once, so a cap
        stated in expansions leaves the memory it implies free to vary by a
        factor of forty between levels. Stated in generated nodes it is roughly
        400 bytes each, which is a number that can be budgeted against the cores
        `parallelize_generator` runs on."""
        crates, player = self.crates0, self.player0
        targets = self.targets
        if crates & targets == targets:
            return []
        # Nodes keep a PARENT POINTER plus the moves of their own macro rather
        # than the whole path: a 100-move plan copied into each of a million
        # queue entries is most of the search's memory and a real slice of its
        # time.
        trail: list = [(-1, ())]
        pq = [(weight * self.heuristic(crates, player), 0, 0, player, crates, 0)]
        closed = {(player, crates): 0}
        counter = 0
        self.nodes = self.expansions = 0
        wall, nbr, live = self.wall, self.nbr, self.live
        while pq:
            _f, g, _c, player, crates, ti = heapq.heappop(pq)
            if crates & targets == targets:
                return self._path(trail, ti)
            if closed.get((player, crates), 1 << 30) < g:
                continue
            parent = self.reach(player, crates)
            self.expansions += 1
            for crate in bits(crates):
                for d in DIRS:
                    stand = nbr[_OPP[d]][crate]
                    if stand < 0 or stand not in parent:
                        continue
                    dest = nbr[d][crate]
                    if (dest < 0 or wall[dest] or crates >> dest & 1
                            or not live >> dest & 1):
                        continue
                    ncrates = crates ^ (1 << crate) | (1 << dest)
                    if self.deadlocked(dest, ncrates):
                        continue
                    moves = self.walk(parent, stand)
                    ng = g + len(moves) + 1
                    key = (crate, ncrates)          # the player ends ON the old
                    if closed.get(key, 1 << 30) <= ng:   # crate cell
                        continue
                    if ncrates & targets == targets:
                        hh = 0
                    else:
                        hh = self.heuristic(ncrates, crate)
                        if hh >= INF:
                            continue                # no crate can cover a target
                    closed[key] = ng
                    counter += 1
                    self.nodes = counter
                    trail.append((ti, tuple(moves) + (d,)))
                    heapq.heappush(pq, (ng + weight * hh, ng, counter,
                                        crate, ncrates, len(trail) - 1))
                    if counter >= node_cap:
                        return None
        return None

    def solve_from(self, player: int, crates: int, **kw) -> int | None:
        """Length of the optimal plan from an arbitrary state on this map, or
        None if the cap runs out. Used by `ties` to price an alternative press;
        the static tables and the assignment memo are shared, so this is only
        the search."""
        sub = Board.__new__(Board)
        sub.__dict__.update(self.__dict__)
        sub.player0, sub.crates0 = player, crates
        plan = sub.solve(**kw)
        return None if plan is None else len(plan)

    @staticmethod
    def _path(trail: list, ti: int) -> list:
        out = []
        while ti > 0:
            ti, moves = trail[ti]
            out.append(moves)
        return [d for moves in reversed(out) for d in moves]

    # -- optimal-action sets --------------------------------------------------
    def optsets(self, plan: list) -> list:
        """Per-step optimal-direction SETS for a flat ``plan``.

        A step that moved a crate is a PUSH and is labelled with itself: which
        crate to shove where is the puzzle. A maximal run of walk steps always
        ends on the stand cell of the push that follows it -- that is how a
        macro is built -- and a walk changes nothing on the board, so one BFS
        from that stand cell over the run's (constant) walkable map gives the
        distance field every alternative is read off: any neighbour one step
        closer is an equally optimal press.

        Classification watches the board rather than trusting a macro boundary
        the flattened plan no longer carries."""
        player, crates = self.player0, self.crates0
        steps = []
        for d in plan:
            nplayer, ncrates = self.step(player, crates, d)
            steps.append(("push" if ncrates != crates else
                          "walk" if nplayer != player else "stuck",
                          player, crates, d))
            player, crates = nplayer, ncrates

        out: list = [None] * len(plan)
        i = 0
        while i < len(steps):
            if steps[i][0] != "walk":
                out[i] = [steps[i][3]]
                i += 1
                continue
            run = i
            while i < len(steps) and steps[i][0] == "walk":
                i += 1
            if i >= len(steps):                  # a trailing walk pushes nothing
                for j in range(run, i):          # -- keep the recorded choice
                    out[j] = [steps[j][3]]
                continue
            stand = steps[i][1]
            dist = self._distances(stand, steps[run][2])
            for j in range(run, i):
                here = steps[j][1]
                d0 = dist.get(here)
                alts = [] if d0 is None else [
                    d for d in DIRS
                    if dist.get(self.nbr[d][here], INF) == d0 - 1]
                # The recorded step is on a shortest route by construction, so
                # an empty (or disagreeing) `alts` would mean the reconstruction
                # has drifted -- fall back to labelling what the expert did.
                out[j] = alts if steps[j][3] in alts else [steps[j][3]]
        return out

    def _distances(self, target: int, crates: int) -> dict:
        """BFS distance to ``target`` over walkable cells."""
        dist = {target: 0}
        queue = deque([target])
        wall, nbr = self.wall, self.nbr
        while queue:
            cur = queue.popleft()
            for d in DIRS:
                nxt = nbr[d][cur]
                if (nxt >= 0 and nxt not in dist and not wall[nxt]
                        and not crates >> nxt & 1):
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        return dist


def board_from_engine(eng, game) -> Board:
    """Read the live interpreter grid into a `Board`.

    ``vide`` joins the walls: it is the black filler outside the ring, on its
    own collision layer (so the engine would let the player stand on it), but no
    level ever leaves a gap in the ring for the player to get there -- which
    `selfcheck` re-asserts rather than taking on trust."""
    wall_ids = {i for n in ("wall", "vide") for i in game.resolve_object_name(n)}
    crate_ids = set(game.resolve_object_name("crate"))
    target_ids = set(game.resolve_object_name("target"))
    player_ids = set(eng._player_indices)
    walls, targets, crates, player = set(), set(), set(), None
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if cell & wall_ids:
                walls.add((r, c))
            if cell & target_ids:
                targets.add((r, c))
            if cell & crate_ids:
                crates.add((r, c))
            if cell & player_ids:
                player = (r, c)
    if player is None:
        raise AssertionError("no player on the board")
    return Board(eng.height, eng.width, walls, targets, crates, player)


def read_state(eng, game) -> tuple:
    """``(player cell, crate mask)`` off the live grid, in `Board` coordinates
    -- the pair `selfcheck` compares against the model."""
    crate_ids = set(game.resolve_object_name("crate"))
    player_ids = set(eng._player_indices)
    w = eng.width
    player, crates = None, 0
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if cell & player_ids:
                player = r * w + c
            if cell & crate_ids:
                crates |= 1 << (r * w + c)
    return player, crates


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class EntrepotExpert(PSExpert):
    """`PSExpert` with the whole search replaced: planning happens on `Board`,
    never by stepping the interpreter, so `heuristic` -- the hook a stepping A*
    would call -- is never reached.

    What is inherited is everything around the search: the in-memory memo, the
    on-disk start-plan cache with its staleness check, and the snapshot/restore
    discipline `PSExpert.plan` wraps `_search` in."""

    directions = list(DIRS)          # ACTION is bound to no rule in this game
    plan_cache_path = PLAN_CACHE

    #: Heuristic weights to try, in order, until one returns a plan inside the
    #: node cap. Weight 1 is admissible A*, i.e. the SHORTEST plan, and it is
    #: what 22 of the 28 recorded levels are solved at. The rest are big open
    #: rooms with 7 to 12 crates, where proving a plan shortest means enumerating
    #: the whole reachable configuration space and no cap this side of a lunch
    #: break is enough -- but where a greedier search walks in almost immediately
    #: (level 9 has twelve crates and falls in 48 expansions at weight 3, against
    #: 2M generated nodes and no answer at weight 1). A weighted plan is
    #: still a genuine, engine-verified win, just not certified shortest, which
    #: is the same trade `PSBeamExpert` makes for the same reason. Only the
    #: successful rung's plan is ever used, so the ladder costs nothing on the
    #: levels that solve at 1.
    weights = (1, 2, 3, 5, 10)

    def __init__(self, game, node_cap: int = 2_000_000, weight: int = 1):
        super().__init__(game, node_cap=node_cap, weight=weight)
        self._how: dict = {}         # level -> one line for `describe`
        self._level: int | None = None

    def heuristic(self, eng) -> int:                        # pragma: no cover
        raise AssertionError("EntrepotExpert plans on Board, not on the engine")

    def _search(self, eng):
        board = board_from_engine(eng, self.g)
        started = time.time()
        for weight in self.weights:
            plan = board.solve(node_cap=self.node_cap, weight=weight)
            if plan is not None:
                self._how[self._level] = (
                    f"weight {weight}, {board.nodes} nodes / "
                    f"{board.expansions} expansions, "
                    f"{time.time() - started:.1f}s")
                return Plan(plan, board.optsets(plan))
        self._how[self._level] = f"NO PLAN ({time.time() - started:.1f}s)"
        return None

    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan`, remembering which level `describe` is reporting on."""
        self._level = level
        return super().plan(eng, level)

    def describe(self, level) -> str:
        """How this level's plan was found, for ``--plans``; "cached" when it
        came off disk and nothing ran."""
        return self._how.get(level, "cached")


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class EntrepotSolver(PSAStarSolver):
    game_id = "puzzlescript_entrepotphage_demake"
    game_name = GAME_NAME
    expert_cls = EntrepotExpert

    #: The two boards no rung of the weight ladder cracks: "Encounter" (10
    #: crates, four mirrored pockets round a shared centre) and "Opened" (11
    #: crates, three chambers joined by single doors). Both are well-formed --
    #: crates and targets match and nothing starts dead -- they are simply
    #: harder than the search, at every weight from 1 to 50 and up to 3M nodes.
    #: Skipped up front so discovery does not burn the whole ladder on each of
    #: them at every cold startup; the other 28 levels are the corpus.
    skip_levels = frozenset({24, 29})

    #: GENERATED nodes per weight rung per level, and therefore the memory dial:
    #: about 400 bytes each across the heap, the closed set and the trail, so a
    #: search that runs to the cap peaks near a gigabyte -- which is the size
    #: that matters, because `parallelize_generator` runs one of these per core
    #: whenever the plan cache is cold.
    node_cap = 2_000_000
    #: The ladder's first rung; see `EntrepotExpert.weights`.
    weight = 1
    #: Above the longest plan (188 presses on level 20) with room for the
    #: recovery prefix's re-plan. The per-level GAME_OVER budget is the game
    #: folder's `_STEP_LIMITS`, not this; the RESET that ends the prefix zeroes
    #: that counter, so only the post-reset plan is charged against it.
    max_steps = 300

    #: Record against the GAME FOLDER's adapter, not a bare `PuzzleScriptAdapter`:
    #: `games/ps:entrepotphage_demake` raises the per-level step limit for the
    #: long boards, three of which cannot be finished inside the adapter's
    #: 200-press default at all. That is the game a live agent is handed, so it
    #: is the game the frames have to come from.
    game_module_id = "ps:entrepotphage_demake"

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup
        rather than as a mysteriously slow first seed, and it fills the disk
        cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)

    def optimal_for(self, expert, level: int, plan, pi: int):
        """The optimal press set at this step, off the plan's own optsets.
        Falls back to the press about to be taken so no expert step ever ships
        unlabelled -- `train_policy` v2 supervises ``optimal`` only, so a step
        without one contributes nothing to the loss."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets) and sets[pi]:
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Model self-check (fuzz against the real interpreter)
# ---------------------------------------------------------------------------

def selfcheck(trials: int = 10, steps: int = 40, verbose: bool = True) -> int:
    """Audit the claim the solver rests on: `Board` reproduces the interpreter
    EXACTLY on the state the search reasons about.

    Random play is the fuzz. It walks into walls and into jammed crates (the
    cancelled turns `require_player_movement` produces), shoves crates on and
    off targets in both directions (the colour-swap rules), parks them in
    corners the plans never visit, and occasionally wins by accident. After
    every single press the player cell, the crate cells and the win flag have to
    agree with the interpreter.

    It also re-asserts the one structural claim `board_from_engine` makes: that
    no ``vide`` cell is reachable, so folding it into the walls loses nothing."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    vide_ids = set(game._game.resolve_object_name("vide"))
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = board_from_engine(eng, game._game)

        # No vide cell is adjacent to a cell the player can walk to.
        reach = board.reach(board.player0, board.crates0)
        loose_vide = [cell for cell in reach
                      if eng.grid[cell // board.w][cell % board.w] & vide_ids]
        if loose_vide:
            print(f"  L{level}: vide reachable at {sorted(loose_vide)}")
            bad += 1

        blocked = wins = pushes = 0
        for t in range(trials):
            game.set_level(level)
            player, crates = board.player0, board.crates0
            rng = random.Random(f"entrepotphage:selfcheck:{level}:{t}")
            for _ in range(steps):
                d = DIRS[rng.randrange(4)]
                nplayer, ncrates = board.step(player, crates, d)
                if nplayer == player:
                    blocked += 1
                elif ncrates != crates:
                    pushes += 1
                player, crates = nplayer, ncrates
                eng.step(d)
                if read_state(eng, game._game) != (player, crates):
                    ep, ec = read_state(eng, game._game)
                    print(f"  L{level}: state divergence on {d} -- "
                          f"model player {player} crates {bits(crates)}, "
                          f"engine player {ep} crates {bits(ec)}")
                    bad += 1
                    break
                if eng.check_win() != board.won(crates):
                    print(f"  L{level}: win divergence on {d} -- engine "
                          f"{eng.check_win()} model {board.won(crates)}")
                    bad += 1
                    break
                if eng.check_win():
                    wins += 1
                    break
            if bad:
                break
        if verbose:
            print(f"  L{level}: {'OK' if not bad else 'VIOLATIONS'} "
                  f"({trials} rollouts, {pushes} pushes, {blocked} cancelled "
                  f"turns, {wins} accidental wins)")
        if bad:
            break
    print("selfcheck clean" if not bad else f"SELFCHECK FAILED ({bad})")
    return bad


# ---------------------------------------------------------------------------
# Optimality check (exhaustive search over primitive moves)
# ---------------------------------------------------------------------------

def truth(levels: list[int] | None = None, cap: int = 4_000_000) -> int:
    """Compare the macro A* against exhaustive breadth-first search over
    PRIMITIVE moves on the boards small enough to enumerate.

    This is the check behind the "dedup on the exact (player, crates) pair"
    claim in the module docstring: the plain BFS has no heuristic, no macros and
    no dedup cleverness to be wrong about, so where it terminates it settles the
    question. The region-keyed variant this generator does NOT use fails it on
    level 3 (38 moves against the true 37)."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    todo = levels if levels is not None else list(range(game.n_levels))
    bad = 0
    for level in todo:
        game.set_level(level)
        board = board_from_engine(eng, game._game)
        start = (board.player0, board.crates0)
        seen = {start: 0}
        queue = deque([start])
        best = None
        while queue:
            player, crates = queue.popleft()
            g = seen[(player, crates)]
            if board.won(crates):
                best = g
                break
            if len(seen) > cap:
                break
            for d in DIRS:
                nxt = board.step(player, crates, d)
                if nxt not in seen:
                    seen[nxt] = g + 1
                    queue.append(nxt)
        if best is None:
            print(f"  L{level}: BFS did not terminate under {cap} states")
            continue
        plan = board.solve(node_cap=EntrepotSolver.node_cap)
        if plan is None:
            # A level that only the weighted rungs crack has no optimality claim
            # to check -- it is not expected to match, and does not count.
            print(f"  L{level}: BFS optimum {best}, no weight-1 plan "
                  f"({len(seen)} BFS states)")
            continue
        ok = len(plan) == best
        bad += not ok
        print(f"  L{level}: BFS optimum {best}, A* {len(plan)} "
              f"({'match' if ok else 'MISMATCH'}, {len(seen)} BFS states)")
    print("truth clean" if not bad else f"TRUTH FAILED ({bad} mismatches)")
    return bad


# ---------------------------------------------------------------------------
# Optimal-action-set check
# ---------------------------------------------------------------------------

def ties(levels: list[int] | None = None, cap: int = 400_000) -> int:
    """Audit the LABELS: every direction `Board.optsets` marks optimal at a step
    must actually keep the plan's total cost, checked by re-solving the board
    from the state that direction leads to.

    The rule this enforces is one-way, and deliberately so. A labelled press
    that is NOT optimal is a bug -- it trains the policy toward a longer
    solution -- and the check fails on it. A press that IS optimal but is not
    labelled is not: it is a SIBLING PLAN, a different crate shoved somewhere
    else for the same total, and labelling those would mean enumerating every
    optimal solution of a sokoban level rather than the order-free stretches of
    one of them. The count is printed per level so the gap stays visible.

    Only for the small boards -- it re-runs a full search per (step, direction)
    -- so pass the levels to check."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    todo = levels if levels is not None else [0, 2, 3, 5, 10, 11, 12]
    bad = 0
    for level in todo:
        game.set_level(level)
        board = board_from_engine(eng, game._game)
        plan = board.solve(node_cap=cap)
        if plan is None:
            print(f"  L{level}: no weight-1 plan, skipped")
            continue
        sets = board.optsets(plan)
        player, crates = board.player0, board.crates0
        checked = sibling = 0
        for i, taken in enumerate(plan):
            for alt in DIRS:
                nplayer, ncrates = board.step(player, crates, alt)
                if (nplayer, ncrates) == (player, crates):
                    cost = None                  # a refused press: never optimal
                else:
                    rest = board.solve_from(nplayer, ncrates, node_cap=cap)
                    cost = None if rest is None else 1 + rest
                optimal = cost == len(plan) - i
                checked += 1
                if alt in sets[i] and not optimal:
                    bad += 1
                    print(f"  L{level} step {i} {alt}: LABELLED BUT NOT "
                          f"OPTIMAL ({cost} vs {len(plan) - i})")
                sibling += optimal and alt not in sets[i]
            player, crates = board.step(player, crates, taken)
        print(f"  L{level}: {len(plan)} steps, "
              f"{sum(1 for s in sets if len(s) > 1)} with a tie set, "
              f"{checked} presses checked, {sibling} optimal-but-unlabelled "
              f"(sibling plans)")
    print("ties clean" if not bad else f"TIES FAILED ({bad} mislabelled)")
    return bad


# ---------------------------------------------------------------------------
# Rendering audit
# ---------------------------------------------------------------------------

#: Every cell COMPOSITION the game can present, as the objects stacked in it.
#: The decoration objects are deliberately absent: they are painted by the cell
#: ABOVE or BELOW and are therefore neighbour-determined, so a composition that
#: differs only in shading is the same board fact seen in two contexts, not two
#: board facts that need telling apart.
_COMPOSITIONS = {
    "floor": (),
    "wall": ("wallvisible",),
    "target": ("target",),
    "crate": ("crateofftarget",),
    "crate_on_target": ("target", "crateontarget"),
    "player": ("player",),
    "player_on_target": ("target", "player"),
    "outside": ("vide",),
}


def audit(verbose: bool = True) -> int:
    """Assert every cell composition renders distinctly, at every cell size the
    30 boards actually use.

    Composition, not object: a crate ON a target has to be readable as both, and
    it is -- the crate is a solid green block where the bare target is a sparse
    ring of green dots, even though the two share ARC palette index 14. The
    check is at the sprite resolution the renderer composites at (``cell_px``),
    which is the resolution the distinction has to survive; the later upscale to
    the 64x64 frame is nearest-neighbour with a factor >= 1, so it duplicates
    pixels and can never merge two blocks that differ here."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    player_ids = set(eng._player_indices)

    sizes: dict[int, list[int]] = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault(max(1, min(64 // eng.height, 64 // eng.width)),
                         []).append(level)

    def block(names: tuple, cell_px: int):
        cell_objs, player_objs = [], []
        for name in names:
            obj_idx = idx[name]
            entry = (eng._obj_layers.get(obj_idx, 0), g.objects[name])
            (player_objs if obj_idx in player_ids else cell_objs).append(entry)
        if not cell_objs and not player_objs:
            return np.zeros((cell_px, cell_px), np.int16)
        cell_objs.sort(key=lambda x: x[0])
        player_objs.sort(key=lambda x: x[0])
        return np.asarray(_render_cell_sprite(cell_objs + player_objs,
                                              cell_px, None, 0))

    bad = 0
    for cell_px, levels in sorted(sizes.items()):
        shots = {name: block(objs, cell_px)
                 for name, objs in _COMPOSITIONS.items()}
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        if verbose:
            print(f"cell {cell_px}px (levels "
                  f"{','.join(str(x) for x in levels)}): "
                  f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad else
          f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Level report
# ---------------------------------------------------------------------------

def _plan_report() -> int:
    """Per-level board size, crate count, plan length and tie coverage -- and
    the plan REPLAYED through the interpreter, so the column that says a level
    is solved is a win the real game granted."""
    solver = EntrepotSolver()
    game = solver.make_game(0)
    expert = EntrepotExpert(game, node_cap=EntrepotSolver.node_cap,
                            weight=EntrepotSolver.weight)
    eng = game._engine
    crate_ids = set(game._game.resolve_object_name("crate"))
    total = solved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        crates = sum(1 for row in eng.grid for cell in row if cell & crate_ids)
        head = f"level {level:2d}: {eng.height:2d}x{eng.width:2d} {crates:2d} crates"
        if level in EntrepotSolver.skip_levels:
            print(f"{head}, SKIPPED (see EntrepotSolver.skip_levels)")
            continue
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"{head}, NO PLAN [{expert.describe(level)}]")
            continue
        for d in plan:
            eng.step(d)
        won = eng.check_win()
        sets = getattr(plan, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        solved += 1
        print(f"{head}, {len(plan):3d} moves "
              f"({'WIN' if won else 'DID NOT WIN'}, budget {game._max_steps}), "
              f"{ties:3d} steps with a tie set "
              f"({ties / max(1, len(plan)):.0%}) [{expert.describe(level)}]")
    print(f"{solved} levels solved, {total} moves total")
    return 0


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plan_report())
    if "--selfcheck" in sys.argv:
        sys.exit(selfcheck())
    if "--audit" in sys.argv:
        sys.exit(audit())
    if "--ties" in sys.argv:
        rest = [a for a in sys.argv[2:] if not a.startswith("-")]
        sys.exit(ties([int(x) for x in rest[0].split(",")] if rest else None))
    if "--truth" in sys.argv:
        rest = [a for a in sys.argv[2:] if not a.startswith("-")]
        sys.exit(truth([int(x) for x in rest[0].split(",")] if rest else None))
    sys.exit(EntrepotSolver.main())
