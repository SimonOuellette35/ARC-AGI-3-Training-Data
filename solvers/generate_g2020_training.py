"""Generate Phase-1 training data for the PuzzleScript game ps:g2020
(bregehr's "2020", a New Year's puzzle).

The harness -- the rotation contract, the trajectory recorder and the BaseSolver
plumbing -- lives in `solvers/common/ps_astar.py`, shared with the other ps:
generators. This file is the game-specific part: a native model of the mechanic,
the searches that plan on it, and the optimal-action tie sets the plans ship as
training targets.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_g2020",
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
action (post rotation remap), i.e. the button an agent presses in the augmented
view, so replaying the recorded actions reproduces the recorded frames exactly.

The game
--------
Sokoban with a PATTERN goal. The pushables are two glyph blocks, a ``two`` and a
``zero``, and the push rules are textbook -- one block one cell, never a chain:

    [ >  player   | pushable ] -> [ > player | > pushable ]
    [ >  player   | wall     ] -> cancel
    [ >  pushable | wall     ] -> cancel
    [ >  pushable | pushable ] -> cancel

What makes it not-Sokoban is that there are no target cells. Two ``late`` rules
stamp an invisible ``win`` marker under any four blocks that spell the year:

    late right [two | zero | two | zero] -> [two win | zero win | two win | zero win]
    late down  [two | zero | two | zero] -> [two win | zero win | two win | zero win]

and the win condition is ``all pushable on win``. So the goal is a SHAPE -- 2020
read left-to-right or top-to-bottom -- to be assembled anywhere on the board, and
EVERY block on the board has to end up inside one.

THE MARKERS ARE PERMANENT
-------------------------
Nothing ever removes a ``win``. Once four cells have carried a 2020 they stay
marked for the rest of the level, so the condition is really "every block is
standing on a cell that has EVER been part of a 2020" -- a predicate on the
HISTORY, not on the board. That is why a model state here carries a marker mask
and why `read_board` reads the interpreter's markers rather than deriving them:
a search that only asked "is a 2020 spelled right now" would be answering a
different, stricter question and could miss the shortest plan outright.

Levels 1-4 ship exactly four blocks, so one pattern is both the first marking
and the win. (Level 4 also opens with four markers already stamped:
``run_rules_on_level_start`` fires on a 2020 sealed into a one-cell-wide
channel, and those four blocks can never be pushed again.) Level 5 ships SIX,
three of each glyph, and four cells cannot hold six blocks -- so it needs either
two lines spelled at once (on a 5-column board that means the six-block column
``2 0 2 0 2 0``, whose two overlapping windows mark all six cells in one turn)
or an older pattern's markers reused under the leftover pair.

The plan found is the second, and it is worth stating exactly, because it is
what the whole marker dimension buys: it spells 2020 down one column at press
43, then dismantles half of it to spell a different 2020 down the NEXT column at
press 58, leaving two blocks parked on cells that have been bare for fifteen
presses. The winning frame shows ONE pattern and a stray pair beside it. The
history-free route -- the six-block column -- is reachable as well: an earlier
configuration of this same two-phase search (weight 2 only, no `shorten` pass)
returned it, at 104 presses.

This is a rendering problem as much as a mechanical one, and it is why
``data/puzzlescript_games/2020.txt`` now draws ``win`` instead of leaving it
``transparent`` -- see the note at the bottom of this docstring.

Searching a native model
------------------------
The other ps: generators search the interpreter itself. Here that is too slow:
``eng.step`` runs at **671 steps/s** on these boards -- the four ``walldeco``
rules (``up [wall| no wall] -> [wall wall1 | ]`` and its three siblings) rescan
the whole grid every single turn to repaint a fake bevel that never changes --
and the exact searches below need millions of steps. So planning happens on a
native bitmask model (~100k transitions/s, 150x) and the interpreter only ever
CERTIFIES the result: ``--verify`` replays every plan through it and requires a
win, and ``--fuzz`` plays random boards press-for-press against it comparing the
blocks, the player, the marker set AND the win flag after every press.

A model state is ``(twos, zeros, player, marks)``: three bitmasks over
``row * width + col`` cells plus the player's cell. Marks are part of the state
because they are part of the goal, and they only ever grow.

The searches (two, and which one runs is not a preference)
----------------------------------------------------------
``_Board.bfs`` is an exhaustive breadth-first search. It returns a SHORTEST plan
and -- from a backward sweep over the BFS layers -- the EXACT optimal-action set
at every step of it: mark the winning states in the final layer, walk the layers
backwards keeping every state with an edge into the marked set, and a step's
optimal set is every press landing in the next layer's marked set. That is the
whole labelling story for levels 1-4, which close in 10 / 58908 / 148886 / 5311
states (2 / 21 / 72 / 22 presses).

Level 5 is out of its reach: 32 free cells and six blocks put its layout space
past 20M states before the marker dimension is even counted, and BFS ran out of
memory without finding a win. It falls back to `_Board.staged`, a two-phase
search over ``walk to the push cell, then push`` MACROS:

  * phase 1 picks one of the board's 24 possible 2020 lines and finds the
    shortest way to spell it there (heuristic: greedy sum of push-distances into
    the four cells);
  * phase 2 searches on from the state that left behind, now with those four
    cells marked, for a real win (heuristic: the cheapest second line plus the
    walk that parks the leftover blocks on cells that are, or are about to be,
    marked).

Both phases are tried at several weights, every (line, weight) combination is
run, and the shortest total wins.

The split is there because the direct search cannot steer. Its estimate is a sum
of push-distances, which prices level 5's win at 13 moves from the start when
the shortest route it can even see costs 104 -- flat and wrong over the whole
middle of the plan, so weighted A* wanders and runs out of nodes at every
weight. "Spell 2020 on THAT line" is a subgoal the same push-distance sum
measures honestly, and once four cells are marked the win estimate has a real
gradient again.

What it costs is the optimality proof: phase 1 optimises a subgoal, and a
shortest way to spell the first pattern is frequently one that wedges a block
somewhere phase 2 cannot use it from -- 16 of level 5's 24 lines dead-end
exactly that way, with phase 2 exhausting its whole reachable space without a
win. So the plan is engine-verified and winning, not certified shortest, and
`_Board.shorten` cuts the seam between the two phases back out of it afterwards.
Its steps are labelled by `_Board.annotate`, which is exact for what it claims:
a push is labelled with itself, and a walk step with every direction that keeps
it on a shortest route to the same push cell.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the per-(seed, level) variables are purely presentation: the frame
rotation (rotation_k in {0,1,2,3}) with the matching directional action remap.
The expert plan is therefore seed-independent -- solved once per level, cached on
disk (`G2020Expert.plan_cache_path`), and replayed per seed with that seed's
remapped screen actions.

The four orientations are exact and they are not cosmetic here: the win rules
read ``right`` and ``down`` in ENGINE space, so on screen the pattern to build
runs in whichever pair of directions the rotation sends those to. That is a
consistent per-seed relabelling of the same puzzle, which is what the
augmentation is for.

Rendering note (the game file is edited)
----------------------------------------
``win`` shipped as ``transparent``: the markers -- the state the win condition is
a predicate on -- were invisible. Three concrete costs, none hypothetical:

  * **Level 5's win is otherwise unreadable.** Its frame shows one 2020 and a
    stray pair of blocks off to the side, and the only thing that makes it a win
    is a pattern that was spelled fifteen presses earlier and taken apart again.
  * **Level 4 opens with four markers already stamped**, under the 2020 sealed
    into its left channel. Nothing on screen said so, so the frame did not carry
    the fact that four of the level's eight blocks were already done and could
    never move.
  * **Any play that spells a pattern permanently changes what counts as a win,
    with no visual feedback at all.** The exploration prefix does exactly this,
    and so does any agent shoving blocks around: afterwards a board that shows
    no pattern anywhere can be a win, and the only record of why is in frames
    that have scrolled past.

So ``data/puzzlescript_games/2020.txt`` now paints ``win`` as three YELLOW
corner pips:

    win
    yellow
    0....
    .....
    .....
    .....
    0...0

Corners because they are the pixels a nearest-neighbour downscale can never drop
(cell 0 always samples sprite row/column 0, and the last always samples 4) and
because they are clear of ``zero``'s ring; yellow because ARC index 11 is unused
by every other object here (background 1, walls 3/4/5, player 9, blocks 14 --
see [[ps-palette-collisions]]).

THREE pips, not four, and the bare one is the TOP-RIGHT. The adapter's per-seed
recolor FLATTENS the player surface to a single index, which repaints
``playerleft``'s third colour -- the ``#c9c9c9`` it uses to draw the left-facing
face, and which was the background grey anyway -- the player's own colour. What
survives of the left/right distinction is then exactly one pixel, the top-right
one, and a fourth pip there made ``player + win`` and ``playerleft + win``
render identically. ``--audit`` is what caught it and what keeps it caught: it
asserts every cell composition this game can produce renders to a distinct
frame, at every level's cell size and under all eight seeds' recolors.

Usage (run from the repo root):
    python solvers/generate_g2020_training.py --plans      # solve + warm cache
    python solvers/generate_g2020_training.py --verify     # replay on the engine
    python solvers/generate_g2020_training.py --ties       # re-derive the labels
    python solvers/generate_g2020_training.py --ties 2     # ... one level (0-based)
    python solvers/generate_g2020_training.py --fuzz       # model vs interpreter
    python solvers/generate_g2020_training.py --audit      # render distinctness
    python solvers/generate_g2020_training.py --episodes 200 \
        --out data/training_multi_level/g2020

``--plans`` first is worth the habit before a `parallelize_generator.py` run: it
leaves the plan cache on disk, and shards that find it warm search nothing at
all. Levels 1-4 are under a second each; level 5's macro search is **478s**, and
`parallelize_generator.py` starts every shard at the same moment, so a cold
cache means each of them paying that eight minutes in parallel with the others.
The ``--plans`` output carries a per-level seconds column for exactly this.
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

from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan  # noqa: E402

GAME_NAME = "2020"

#: The four presses, in the engine's own names. ACTION5 has no rule in this
#: game, so it is not a successor -- it would only widen the branching.
PRESSES = ("up", "down", "left", "right")

#: Stand-in for "unreachable / no estimate". Finite so a search stays complete.
_BIG = 1 << 29


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """The static half of a level, plus the mechanic.

    Cells are ``row * width + col`` ints -- the hot loops index dicts with them
    millions of times and tuple keys cost several times as much. A state is
    ``(twos, zeros, player, marks)``: three bitmasks over cells and the player's
    cell.

    Geometry goes through ONE table, `nbr`: ``nbr[press][cell]`` is the cell that
    press leads to, or -1 when it leaves the board or would wrap ``left`` from
    column 0 onto the previous row. Every shipped level is walled all the way
    round so neither can actually happen in play, but `_fuzz` assembles boards
    this file did not write and a wrap is exactly the kind of thing that would
    make the model and the interpreter disagree in one cell, once, deep inside a
    search.
    """

    def __init__(self, h: int, w: int, wall: int):
        self.h, self.w, self.wall = h, w, wall
        n = h * w
        self.nbr = {
            "up": [k - w if k - w >= 0 else -1 for k in range(n)],
            "down": [k + w if k + w < n else -1 for k in range(n)],
            "left": [k - 1 if k % w else -1 for k in range(n)],
            "right": [k + 1 if (k + 1) % w else -1 for k in range(n)],
        }
        self.free = [k for k in range(n) if not (wall >> k) & 1]
        # Every place a 2020 could be spelled: four cells left-to-right or
        # top-to-bottom, all of them off the walls. This is the game's whole
        # goal vocabulary -- 7 to 24 lines on the shipped boards.
        runs = [tuple(r * w + c + i for i in range(4))
                for r in range(h) for c in range(w - 3)]
        runs += [tuple((r + i) * w + c for i in range(4))
                 for r in range(h - 3) for c in range(w)]
        self.runs = [t for t in runs
                     if all(not (wall >> k) & 1 for k in t)]
        # Only a line through a cell that just changed can newly complete, so
        # `step` rescans those and not all of them.
        self.runs_at: dict[int, list] = {}
        for t in self.runs:
            for k in t:
                self.runs_at.setdefault(k, []).append(t)
        self.push = self._push_tables()

    # -- static tables --------------------------------------------------------
    def _push_tables(self) -> dict:
        """``{target: {cell: pushes to get a block from cell onto target}}``.

        Reverse BFS over push edges, ignoring the other blocks (the standard
        sokoban relaxation): a block reaches ``X`` from ``X - d`` only when
        ``X - d`` and the player's stand cell ``X - 2d`` are both on the board
        and not walls. A cell missing from a table is a cell no block can ever
        be pushed to from there, which is also the deadlock test the heuristics
        below get for free."""
        back = {"up": "down", "down": "up", "left": "right", "right": "left"}
        tables = {}
        for target in self.free:
            dist = {target: 0}
            queue = deque([target])
            while queue:
                cur = queue.popleft()
                for press in PRESSES:
                    src = self.nbr[back[press]][cur]      # block comes from here
                    if src < 0 or (self.wall >> src) & 1:
                        continue
                    stand = self.nbr[back[press]][src]    # player stands here
                    if stand < 0 or (self.wall >> stand) & 1:
                        continue
                    if src not in dist:
                        dist[src] = dist[cur] + 1
                        queue.append(src)
            tables[target] = dist
        return tables

    def complete_runs(self, twos: int, zeros: int) -> int:
        """Mask of every cell currently inside a spelled 2020."""
        marks = 0
        for a, b, c, d in self.runs:
            if ((twos >> a) & 1 and (zeros >> b) & 1
                    and (twos >> c) & 1 and (zeros >> d) & 1):
                marks |= (1 << a) | (1 << b) | (1 << c) | (1 << d)
        return marks

    # -- the mechanic ---------------------------------------------------------
    @staticmethod
    def won(state) -> bool:
        twos, zeros, _player, marks = state
        return (twos | zeros) & ~marks == 0

    def step(self, state, press):
        """One press. Returns ``(state, won)``; the state is returned UNCHANGED
        (identical object) for a press the rules cancel.

        The three cancel rules make every refusal a whole-turn no-op: walking
        into a wall, shoving a block into a wall, and shoving a block into
        another block all leave the player where it stood."""
        twos, zeros, player, marks = state
        step = self.nbr[press]
        target = step[player]
        if target < 0:
            return state, (twos | zeros) & ~marks == 0
        bit = 1 << target
        if self.wall & bit:
            return state, (twos | zeros) & ~marks == 0
        blocks = twos | zeros
        if blocks & bit:
            beyond = step[target]
            if beyond < 0:
                return state, (twos | zeros) & ~marks == 0
            bbit = 1 << beyond
            if (self.wall | blocks) & bbit:
                return state, (twos | zeros) & ~marks == 0
            if twos & bit:
                twos ^= bit | bbit
            else:
                zeros ^= bit | bbit
            # The late rules run after the move. Only a line through the cell
            # the block landed on can have newly completed.
            for a, b, c, e in self.runs_at.get(beyond, ()):
                if ((twos >> a) & 1 and (zeros >> b) & 1
                        and (twos >> c) & 1 and (zeros >> e) & 1):
                    marks |= (1 << a) | (1 << b) | (1 << c) | (1 << e)
        return (twos, zeros, target, marks), (twos | zeros) & ~marks == 0

    # -- exhaustive search, with exact optimal-action sets --------------------
    def bfs(self, state, cap: int):
        """Shortest plan from ``state`` plus the EXACT optimal set per step, or
        ``(None, None)`` when the search would exceed ``cap`` distinct states
        (or when the reachable space closes with no win in it).

        The labels come out of the layers for free, which is why this is worth
        running wherever it fits. Mark the winning states in the last layer;
        sweep backwards keeping every state of layer ``k`` with an edge into
        layer ``k+1``'s marked set; then a step's optimal set is every press
        that lands in the next marked set. Both a press that ties and one that
        does not are decided by the same sweep, so the sets are exact rather
        than inferred. Same trick as [[ps-flood-solver]]'s.

        ``cap`` is a MEMORY budget: every state is a 4-tuple of Python ints held
        alive in both the seen-set and its layer, so a million of them is a few
        hundred MB."""
        if self.won(state):
            return [], []
        seen = {state}
        layers = [[state]]
        wins: list = []
        while not wins:
            nxt = []
            for s in layers[-1]:
                for press in PRESSES:
                    t, won = self.step(s, press)
                    if t in seen:
                        continue
                    seen.add(t)
                    nxt.append(t)
                    if won:
                        wins.append(t)
            if not nxt:
                return None, None            # closed without a win
            layers.append(nxt)
            if not wins and len(seen) > cap:
                return None, None

        good = set(wins)
        for k in range(len(layers) - 2, -1, -1):
            keep = set()
            for s in layers[k]:
                for press in PRESSES:
                    if self.step(s, press)[0] in good:
                        keep.add(s)
                        break
            good = keep
            layers[k] = good                 # only the useful states survive
        # `good` is now layer 0's keep-set, which is {state} by construction.

        plan, optsets = [], []
        cur = state
        for k in range(1, len(layers)):
            nxt_good = layers[k] if k < len(layers) - 1 else set(wins)
            best = [press for press in PRESSES
                    if self.step(cur, press)[0] in nxt_good]
            plan.append(best[0])
            optsets.append(best)
            cur = self.step(cur, best[0])[0]
        return plan, optsets

    # -- macros ---------------------------------------------------------------
    def macros(self, state) -> list[list[str]]:
        """Every ``walk to the push cell, then push`` the player can reach right
        now, each a list of primitive presses.

        Branching on macros makes the search depth the number of PUSHES: a bare
        move has no effect on this board but where the player stands, so a
        primitive search spends its whole budget re-deriving walks."""
        twos, zeros, player, _marks = state
        back = {"up": "down", "down": "up", "left": "right", "right": "left"}
        blocked = self.wall | twos | zeros
        parent: dict[int, tuple | None] = {player: None}
        queue = deque([player])
        while queue:
            cur = queue.popleft()
            for name in PRESSES:
                nxt = self.nbr[name][cur]
                if nxt < 0 or (blocked >> nxt) & 1 or nxt in parent:
                    continue
                parent[nxt] = (cur, name)
                queue.append(nxt)

        def walk_to(cell):
            out = []
            while parent[cell] is not None:
                cell, name = parent[cell]
                out.append(name)
            out.reverse()
            return out

        out = []
        blocks = twos | zeros
        for k in self.free:
            if not (blocks >> k) & 1:
                continue
            for name in PRESSES:
                stand, dest = self.nbr[back[name]][k], self.nbr[name][k]
                if stand < 0 or dest < 0:
                    continue
                if (blocked >> dest) & 1 or stand not in parent:
                    continue
                out.append(walk_to(stand) + [name])
        return out

    def _macro_search(self, state, goal, estimate, cap: int, weight: int):
        """Weighted A* over `macros`, costed in primitive MOVES (the unit the
        agent pays). Returns the press list reaching ``goal``, or None.

        Dedup keys the EXACT state, not the player's reachable region: the usual
        sokoban canonicalisation is only exact when ``g`` counts pushes, and
        merging by region mis-costs a move-counted search by the region's
        diameter (see `PSPushExpert`). After a macro the player always stands on
        the pushed block's old cell, so the exact key barely grows the space."""
        if goal(state):
            return []
        pq = [(weight * estimate(state), 0, 0, state, [])]
        best_g = {state: 0}
        counter = nodes = 0
        while pq:
            _f, g, _c, s, path = heapq.heappop(pq)
            for macro in self.macros(s):
                cur = s
                hit = -1
                for i, press in enumerate(macro):
                    cur, _won = self.step(cur, press)
                    if goal(cur):
                        hit = i
                        break
                nodes += 1
                if hit >= 0:
                    return path + macro[:hit + 1]
                if cur is s:
                    continue                 # a push the rules cancelled
                ng = g + len(macro)
                if best_g.get(cur, _BIG) <= ng:
                    continue
                best_g[cur] = ng
                counter += 1
                heapq.heappush(pq, (ng + weight * estimate(cur), ng, counter,
                                    cur, path + macro))
            if nodes >= cap:
                return None
        return None

    # -- heuristics -----------------------------------------------------------
    def fill_cost(self, state, run) -> int:
        """Push-distance to spell 2020 on ``run``, blocks matched greedily.

        Greedy rather than exact because it runs on every generated node; it
        commits each of the four cells, in order, to its nearest unused block of
        the right glyph."""
        twos, zeros, _player, _marks = state
        available = ([k for k in self.free if (twos >> k) & 1],
                     [k for k in self.free if (zeros >> k) & 1])
        total = 0
        for cell, pool in zip(run, (available[0], available[1],
                                    available[0], available[1])):
            table = self.push[cell]
            pick = best = None
            for block in pool:
                d = table.get(block)
                if d is not None and (best is None or d < best):
                    best, pick = d, block
            if pick is None:
                return _BIG
            total += best
            pool.remove(pick)
        return total

    def heuristic(self, state) -> int:
        """Estimated moves to a WIN: the cheapest ``spell one more 2020 and park
        everything else on a marked cell``.

        Zero exactly at a win, because a state where every block already sits on
        a marked cell has some complete line to charge nothing for and no
        leftovers to move. It is a search guide, not a bound: the matching is
        greedy, push-distances ignore the other blocks, and on a board that
        still needs TWO more patterns it prices only one of them."""
        twos, zeros, _player, marks = state
        if (twos | zeros) & ~marks == 0:
            return 0
        tc = [k for k in self.free if (twos >> k) & 1]
        zc = [k for k in self.free if (zeros >> k) & 1]
        mc = [k for k in self.free if (marks >> k) & 1]
        push = self.push
        best = _BIG
        for run in self.runs:
            free_t, free_z = list(tc), list(zc)
            total = 0
            ok = True
            for cell, pool in zip(run, (free_t, free_z, free_t, free_z)):
                table = push[cell]
                pick = near = None
                for block in pool:
                    d = table.get(block)
                    if d is not None and (near is None or d < near):
                        near, pick = d, block
                if pick is None or total + near >= best:
                    ok = False
                    break
                total += near
                pool.remove(pick)
            if not ok:
                continue
            targets = mc + list(run)
            for block in free_t + free_z:
                near = None
                for cell in targets:
                    d = push[cell].get(block)
                    if d is not None and (near is None or d < near):
                        near = d
                if near is None:
                    ok = False
                    break
                total += near
            if ok and total < best:
                best = total
        return best

    # -- the fallback plan ----------------------------------------------------
    def staged(self, state, cap: int, weights=(1, 2)):
        """Plan for a board `bfs` cannot reach: spell one 2020, then win.

        Runs the direct search first (one pattern is the whole answer whenever
        the board carries only four blocks) and then, because that is not
        enough on level 5, every ``(line, weight)`` two-phase combination:
        shortest way to spell 2020 on that line, then shortest way on from
        there to an actual win. The shortest total is returned.

        Both phases are needed and neither is optional. Phase 1 optimises a
        SUBGOAL, and the shortest way to spell a pattern frequently ends with a
        block wedged where phase 2 can no longer use it -- on level 5, 16 of the
        24 lines dead-end that way and phase 2 exhausts its reachable space
        without a win. That is also why several weights are tried: a greedier
        phase 1 reaches the same pattern by a different route, and the route is
        what decides whether phase 2 has anything left to work with."""
        best = None
        for weight in weights:
            plan = self._macro_search(state, self.won, self.heuristic,
                                      cap, weight)
            if plan is not None and (best is None or len(plan) < len(best)):
                best = plan

        for run in self.runs:
            def spelled(s, r=run):
                twos, zeros = s[0], s[1]
                return bool((twos >> r[0]) & 1 and (zeros >> r[1]) & 1
                            and (twos >> r[2]) & 1 and (zeros >> r[3]) & 1)

            for w1 in weights:
                first = self._macro_search(
                    state, spelled, lambda s, r=run: self.fill_cost(s, r),
                    cap, w1)
                if first is None or (best is not None
                                     and len(first) >= len(best)):
                    continue
                mid = state
                for press in first:
                    mid, _won = self.step(mid, press)
                for w2 in weights:
                    rest = self._macro_search(mid, self.won, self.heuristic,
                                              cap, w2)
                    if rest is None:
                        continue
                    if best is None or len(first) + len(rest) < len(best):
                        best = first + rest
                    break              # a greedier w2 cannot be shorter
        return None if best is None else self.shorten(state, best)

    def shorten(self, state, plan) -> list:
        """Delete every contiguous block of presses the plan does not need,
        longest first, keeping only deletions the model still calls a win.

        A two-phase plan stitches two searches together and the seam shows: the
        state phase 1 hands over is chosen for a subgoal, so phase 2 routinely
        opens by undoing part of it, and neither search can see the other's
        moves. This is the standard beam-plan cleanup (see `PSBeamExpert`) and
        it is sound by construction -- a deletion is kept only when replaying
        what is left still reaches a win on the model, and the model is the same
        one `--verify` checks against the interpreter."""
        def wins(seq):
            cur = state
            for press in seq:
                cur, won = self.step(cur, press)
                if won:
                    return True
            return False

        changed = True
        while changed:
            changed = False
            for size in range(len(plan) - 1, 0, -1):
                for at in range(len(plan) - size + 1):
                    cut = plan[:at] + plan[at + size:]
                    if wins(cut):
                        plan = cut
                        changed = True
                        break
                if changed:
                    break
        # A deletion can leave the win happening before the last press.
        cur = state
        for i, press in enumerate(plan):
            cur, won = self.step(cur, press)
            if won:
                return plan[:i + 1]
        return plan

    # -- optimal-action sets for a macro plan ---------------------------------
    def _walk_distances(self, state, target: int) -> dict:
        """BFS distance to ``target`` over the cells the player may WALK on: no
        wall, and no block (walking into a block is a push, not a walk)."""
        blocked = self.wall | state[0] | state[1]
        dist = {target: 0}
        queue = deque([target])
        while queue:
            cur = queue.popleft()
            for press in PRESSES:
                nxt = self.nbr[press][cur]
                if nxt < 0 or (blocked >> nxt) & 1 or nxt in dist:
                    continue
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
        return dist

    def annotate(self, state, plan) -> list[list[str]]:
        """Per-step optimal-direction SETS for a macro ``plan``.

        Most of a macro plan is the player WALKING to the next push cell, and a
        walk's order is FREE: no rule here fires on a bare move, so every
        interleaving of the two axes that stays on a shortest route to the same
        cell costs the same and leaves an identical board. Labelling one
        arbitrary interleaving as the only right answer would train the policy
        to a coin flip it cannot win. Pushes are labelled with themselves --
        which block to shove where IS the puzzle, so a sibling push is a
        different plan rather than a reordering of this one.

        Steps are classified by watching the BOARD, not by trusting a macro
        boundary the flat plan no longer carries: a step that moved a block is a
        push, a step that moved only the player is a walk. A maximal run of
        walks always ends on the stand cell of the push that follows it, and the
        board is constant across that run, so one BFS answers the whole run."""
        steps = []
        cur = state
        for press in plan:
            nxt, _won = self.step(cur, press)
            kind = ("push" if nxt[:2] != cur[:2]
                    else "walk" if nxt[2] != cur[2] else "stuck")
            steps.append((kind, cur, press))
            cur = nxt

        out: list = [None] * len(plan)
        i = 0
        while i < len(steps):
            if steps[i][0] != "walk":
                out[i] = [steps[i][2]]
                i += 1
                continue
            run = i
            while i < len(steps) and steps[i][0] == "walk":
                i += 1
            if i >= len(steps):              # a trailing walk pushes nothing
                for j in range(run, i):
                    out[j] = [steps[j][2]]
                continue
            stand = steps[i][1][2]
            dist = self._walk_distances(steps[run][1], stand)
            for j in range(run, i):
                here = steps[j][1][2]
                d0 = dist.get(here)
                alts = [] if d0 is None else [
                    name for name in PRESSES
                    if self.nbr[name][here] >= 0
                    and dist.get(self.nbr[name][here]) == d0 - 1]
                # The recorded step is on a shortest route by construction, so
                # an empty or disagreeing `alts` means the reconstruction has
                # drifted -- fall back to labelling what the expert did.
                out[j] = alts if steps[j][2] in alts else [steps[j][2]]
        return out


def read_board(eng, g):
    """``(board, state)`` for the interpreter's current grid."""
    idx = g.obj_name_to_idx
    wall_id, two_id, zero_id, win_id = (idx["wall"], idx["two"],
                                        idx["zero"], idx["win"])
    players = set(eng._player_indices)
    h, w = len(eng.grid), len(eng.grid[0])
    wall = twos = zeros = marks = 0
    player = None
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            k = r * w + c
            if wall_id in cell:
                wall |= 1 << k
            if two_id in cell:
                twos |= 1 << k
            if zero_id in cell:
                zeros |= 1 << k
            if win_id in cell:
                marks |= 1 << k
            if cell & players:
                player = k
    board = _Board(h, w, wall)
    # The engine stamps markers under any 2020 already spelled (the prelude has
    # `run_rules_on_level_start`, and level 4 ships one), so this is normally a
    # no-op on a real level. It is not a no-op on a board assembled by `--fuzz`,
    # which never ran the start rules -- and a model whose marker set lagged the
    # interpreter's by one press would report the wrong win.
    marks |= board.complete_runs(twos, zeros)
    return board, (twos, zeros, player, marks)


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class G2020Expert(PSExpert):
    """Plans on `_Board`; the interpreter only ever certifies the result.

    `plan_cache_path` keeps every level's start plan on disk, so a shard of
    `parallelize_generator` does not re-derive the searches (see
    `PSExpert.plan`). That matters here: level 5's fallback search is minutes,
    and every shard starts at the same moment."""

    directions = list(PRESSES)

    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "g2020_plans.json")

    #: Distinct states the exhaustive search may hold. Sized as MEMORY, not
    #: patience: every state is a 4-tuple of Python ints kept alive in both the
    #: seen-set and its layer. Levels 1-4 close in 10 / 58908 / 148886 / 5311;
    #: level 5 is past 20M and is MEANT to fall through this to
    #: `_Board.staged`, so do not raise it hoping for a shortest plan there --
    #: 20M of these states is well past this machine's memory.
    bfs_cap = 1_000_000

    #: Macros generated per phase of the fallback search before it gives up.
    #: Every phase-1 line that cannot close runs the full cap, so this is most
    #: of level 5's 478s -- and lowering it loses lines that DO close, which is
    #: how the shortest total is found.
    macro_cap = 1_500_000

    #: Weights the fallback tries in each phase. See `_Board.staged` for why a
    #: greedier phase 1 is not merely a faster one.
    weights = (1, 2)

    def heuristic(self, eng) -> int:                      # pragma: no cover
        raise NotImplementedError("the model owns the heuristic; see _Board")

    def _search(self, eng) -> list | None:
        """Exhaustive BFS first, because a proof of shortest is what buys the
        exact tie sets; `_Board.staged` only for the board it cannot hold."""
        board, state = read_board(eng, self.g)
        plan, optsets = board.bfs(state, self.bfs_cap)
        if plan is not None:
            return Plan(plan, optsets)
        plan = board.staged(state, self.macro_cap, self.weights)
        if plan is None:
            return None
        return Plan(plan, board.annotate(state, plan))


class G2020Solver(PSAStarSolver):
    game_id = "puzzlescript_g2020"
    game_name = GAME_NAME
    expert_cls = G2020Expert

    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so anything that folder ever does to the game cannot
    #: silently diverge from what is taped here.
    game_module_id = "ps:g2020"

    #: The longest plan is level 5's; `--plans` prints the budget column that
    #: says how it sits against the adapter's stock 200-press per-level cap.
    #: The exploration prefix does not eat into it: it ends in a RESET, and
    #: `PuzzleScriptAdapter._do_reset` zeroes the step counter.
    max_steps = 200


# ---------------------------------------------------------------------------
# Reports and checks
# ---------------------------------------------------------------------------

def _levels():
    solver = G2020Solver()
    game = solver.make_game(0)
    return game, G2020Expert(game)


def _render(board, state) -> str:
    twos, zeros, player, marks = state
    out = []
    for r in range(board.h):
        line = []
        for c in range(board.w):
            k = r * board.w + c
            bit = 1 << k
            ch = ("#" if board.wall & bit else "P" if player == k else
                  "2" if twos & bit else "0" if zeros & bit else ".")
            if marks & bit:
                ch = {".": "+", "P": "p", "2": "T", "0": "O", "#": "#"}[ch]
            line.append(ch)
        out.append("".join(line))
    return "\n".join(out)


def _report() -> int:
    game, expert = _levels()
    eng = game._engine
    total = solved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = read_board(eng, game._game)
        blocks = bin(state[0] | state[1]).count("1")
        clock = time.time()
        found = expert.plan(eng, level)
        elapsed = time.time() - clock
        if found is None:
            print(f"level {level + 1}: NO PLAN", flush=True)
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        solved += 1
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level + 1}: {board.h:2d}x{board.w:2d} "
              f"{blocks} blocks, {len(board.runs):2d} lines   "
              f"{len(found):3d} presses ({room}), {ties:3d} tie steps "
              f"({ties / max(1, len(found)):.0%}), {elapsed:7.1f}s",
              flush=True)
    print(f"total {total} presses over {solved} solved levels "
          f"({game.n_levels} shipped)")
    return 0 if solved == game.n_levels else 1


def _verify() -> int:
    """Replay every plan on the REAL interpreter and require a WIN."""
    game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level + 1}: NO PLAN")
            bad += 1
            continue
        game.set_level(level)
        for press in plan:
            eng.step(press)
        ok = eng.check_win()
        bad += not ok
        board, state = read_board(eng, game._game)
        print(f"level {level + 1}: {len(plan):3d} presses -> "
              f"{'WIN' if ok else 'NOT WON'}\n{_render(board, state)}")
    print("verify clean" if not bad else f"VERIFY FAILED on {bad} levels")
    return 0 if not bad else 1


def _ties(levels=None) -> int:
    """Re-derive every labelled step with an independent search and compare.

    At each step, press everything, re-solve from each successor for its true
    distance to a win, and require that the presses reaching one in the moves
    remaining are exactly the labelled set -- so both an unlabelled press that
    ties and a labelled press that does not are failures. The re-solve is an
    independent exhaustive BFS, not the layered sweep the labels came from.

    Only the levels `_Board.bfs` answered make a tie CLAIM; level 5's fallback
    plan is not proved shortest, so `_UNLABELLED_LEVELS` excuses it."""
    game, expert = _levels()
    eng = game._engine
    bad = tested = 0
    for level in range(game.n_levels):
        if (levels and level not in levels) or level in _UNLABELLED_LEVELS:
            continue
        game.set_level(level)
        board, state = read_board(eng, game._game)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        for i, press in enumerate(plan):
            remaining = len(plan) - i
            measured = []
            for alt in PRESSES:
                nxt, won = board.step(state, alt)
                tested += 1
                if won:
                    if remaining == 1:
                        measured.append(alt)
                    continue
                if nxt is state:
                    continue
                sub, _ = board.bfs(nxt, expert.bfs_cap)
                if sub is not None and 1 + len(sub) == remaining:
                    measured.append(alt)
            if sorted(measured) != sorted(sets[i]):
                bad += 1
                print(f"level {level + 1} step {i:3d}: labelled {sets[i]} "
                      f"but measured {measured}")
            state, _won = board.step(state, press)
        print(f"level {level + 1}: {len(plan):3d} steps checked", flush=True)
    print(f"ties: {tested} alternatives re-solved, {bad} disagreements")
    return 0 if not bad else 1


#: Levels whose plan is NOT proved shortest, so its per-step labels claim only
#: that the walks are order-free (`_Board.annotate`), never that a push ties.
#: Only `--ties` uses this.
_UNLABELLED_LEVELS = frozenset({4})


def _audit(seeds=range(8)) -> int:
    """Assert every cell COMPOSITION renders distinctly, at every cell size and
    under every seed's recolor.

    The marker is the point: ``win`` is what the win condition reads, it sits on
    the topmost collision layer, and it shipped ``transparent``. This is the
    check that the sprite added in `data/puzzlescript_games/2020.txt` actually
    survives every level's downscale -- and that it did not, in surviving,
    swallow the difference between a two and a zero underneath it.

    It runs over seeds because the adapter's per-seed recolor is what decides
    how much of a sprite's palette survives: the player surface is FLATTENED to
    one index, so ``playerleft``'s third colour (``#c9c9c9``, the background
    grey it uses to draw the left-facing face) is repainted the player's own
    colour and stops being visible at all. What is left of the left/right
    distinction is a single pixel, the top-right one -- which is why the marker
    is three pips and not four, and why the bare corner is that one."""
    from adapters.puzzlescript_adapter import _render_frame  # noqa: PLC0415

    bad = 0
    for seed in seeds:
        bad += _audit_seed(seed, _render_frame)
    print("audit clean" if not bad else f"AUDIT FAILED: {bad} problems")
    return 0 if not bad else 1


def _audit_seed(seed: int, _render_frame) -> int:
    game = G2020Solver().make_game(seed)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {
        "bg": (), "wall": ("wall",),
        "two": ("two",), "zero": ("zero",), "player": ("player",),
        "player_left": ("player", "playerleft"),
        "mark": ("win",),
        "two_mark": ("two", "win"), "zero_mark": ("zero", "win"),
        "player_mark": ("player", "win"),
        "player_left_mark": ("player", "playerleft", "win"),
    }
    sizes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level + 1)

    def shoot(h, w, objs):
        # Fill the WHOLE board with the composition and compare whole frames:
        # `_render_frame` upscales any sub-64 render to fill the frame, so
        # cropping one cell out by arithmetic lands in the wrong place and
        # every composition reads as identical (see [[explod-rendering-fixes]]).
        eng.height, eng.width = h, w
        eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                     for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g))

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {name: shoot(h, w, objs) for name, objs in comps.items()}
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"seed {seed}  {h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    return bad


def _fuzz(n_boards: int = 600, n_steps: int = 40, seed: int = 0) -> int:
    """Differential fuzz: random boards played press-for-press against the real
    interpreter, comparing blocks, player, markers AND the win flag every press.

    Random boards alone would barely ever spell a 2020 -- a marker appeared 9
    times in 30k presses of pure noise -- so half of them are seeded with a line
    that is already one push away from complete, and the shipped levels are
    replayed along their own solutions first. The per-event COVERAGE counters
    below exist so a clean run cannot quietly mean "never exercised"."""
    game, expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    rng = random.Random(seed)
    cover = dict.fromkeys(("push", "walk", "refused", "newmark", "win"), 0)
    mismatches = steps = 0

    def truth():
        w = len(eng.grid[0])
        twos = zeros = marks = 0
        player = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                k = r * w + c
                if idx["two"] in cell:
                    twos |= 1 << k
                if idx["zero"] in cell:
                    zeros |= 1 << k
                if idx["win"] in cell:
                    marks |= 1 << k
                if idx["player"] in cell:
                    player = k
        return twos, zeros, player, marks

    def compare(board, state, press):
        nonlocal mismatches, steps
        eng.step(press)
        seen = truth()
        seen_won = eng.check_win()
        nxt, won = board.step(state, press)
        steps += 1
        if nxt is state:
            cover["refused"] += 1
        elif nxt[:2] != state[:2]:
            cover["push"] += 1
        else:
            cover["walk"] += 1
        if nxt[3] != state[3]:
            cover["newmark"] += 1
        if won:
            cover["win"] += 1
        if nxt != seen or won != seen_won:
            mismatches += 1
            print(f"MISMATCH press={press} {eng.height}x{eng.width}\n"
                  f"  model  {nxt} won={won}\n  engine {seen} won={seen_won}")
            return seen
        return nxt

    # Phase 1: the shipped levels, replayed along their own solutions and then
    # walked randomly from the states those reach.
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        game.set_level(level)
        board, state = read_board(eng, g)
        for press in (plan or []):
            state = compare(board, state, press)
        for _ in range(20):
            game.set_level(level)
            board, state = read_board(eng, g)
            for _ in range(60):
                state = compare(board, state, rng.choice(PRESSES))

    # Phase 2: random boards, half of them one push from spelling a 2020.
    for board_i in range(n_boards):
        # A seeded board needs six interior columns (``2 0 2 _ 0 P``), so it
        # fixes the width; the rest are free.
        seeded = board_i % 2 == 0
        h, w = rng.randint(5, 8), 8 if seeded else rng.randint(5, 8)
        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        rng.shuffle(cells)
        take = 0

        def grab(k):
            nonlocal take
            out = set(cells[take:take + k])
            take += k
            return out

        walls = grab(rng.randint(0, max(0, len(cells) // 6)))
        twos = grab(rng.randint(1, 3))
        zeros = grab(rng.randint(1, 3))
        player = grab(1)
        if not player:
            continue
        if seeded:
            # Seed a line one press from spelling 2020: ``2 0 2 _`` with a spare
            # zero and the player lined up behind it, so pressing left shoves
            # the zero home and stamps the markers. Pure noise almost never gets
            # there -- an unseeded phase 2 stamped 9 markers in 30k presses --
            # and the marker rules are the half of this mechanic the shipped
            # levels exercise least.
            r = rng.randrange(1, h - 1)
            line = [(r, 1 + i) for i in range(4)]
            walls -= set(line) | {(r, 5), (r, 6)}
            twos = {line[0], line[2]}
            zeros = {line[1], (r, 5)}
            player = {(r, 6)}

        grid = []
        for r in range(h):
            row = []
            for c in range(w):
                cell = {idx["background"]}
                if r in (0, h - 1) or c in (0, w - 1) or (r, c) in walls:
                    cell.add(idx["wall"])
                elif (r, c) in twos:
                    cell.add(idx["two"])
                elif (r, c) in zeros:
                    cell.add(idx["zero"])
                elif (r, c) in player:
                    cell.add(idx["player"])
                row.append(cell)
            grid.append(row)
        eng.grid, eng.height, eng.width = grid, h, w
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        eng._rule_win = False

        board, state = read_board(eng, g)
        for _ in range(n_steps):
            state = compare(board, state, rng.choice(PRESSES))
        if mismatches > 4:
            return 1

    print(f"fuzz: {steps} transitions, {mismatches} mismatches")
    print("coverage: " + "  ".join(f"{k}={v}" for k, v in cover.items()))
    missing = [k for k, v in cover.items() if not v]
    if missing:
        print(f"NOT EXERCISED: {missing}")
    return 0 if not mismatches and not missing else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--ties" in sys.argv:
        sys.exit(_ties([int(x) for x in sys.argv[sys.argv.index("--ties") + 1:]
                        if x.isdigit()]))
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    sys.exit(G2020Solver.main())
