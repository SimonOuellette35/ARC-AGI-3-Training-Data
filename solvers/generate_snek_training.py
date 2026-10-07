"""Generate Phase-1 training data for the PuzzleScript game ps:snek
("Snek", Aaron Steed).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, the admissible heuristic, a memory-lean A* that returns a distance
FIELD rather than a bare path, and the reports that prove every shipped plan is
winnable and shortest.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_snek",
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
Every expert step carries the full set of equally-optimal presses.

The game
--------
A snake crawls around a walled board. The win condition is ``all target on
obj``, and ``obj`` is ``playerpart or wall or crate`` -- so every blue Target
cell has to be sitting under a piece of snake or under a pushed Crate at the
same moment. Apples lengthen the snake, spikes kill it.

Everything below was MEASURED against the interpreter, not read off the .txt.

**Only TWO of the four arrow keys do anything, and the game is deterministic.**
There is no RNG in the ruleset. ``[ up Player ] -> [ Player ]`` and
``[ down Player ] -> [ Player ]`` strip the force off an up/down press, and with
``require_player_movement`` in the prelude the whole turn is then cancelled: up
and down are EXACT no-ops on every state (measured on all nine level starts, and
implied everywhere else -- nothing else in the ruleset can fire without a
movement). ``noaction`` is set and no rule reads ACTION either. So the search
branches on ``left`` and ``right`` alone, which is the single biggest reason
these boards are tractable at all: the branching factor is TWO.

**A live key is a 3-way priority chain, not a direction.** The "wiggle" rules
fire in the order left, up, right, down, each one turning a blocked mover a
quarter turn:

    left  [ left  player | turn ] -> [ up    player | turn ]
    up    [ up    player | turn ] -> [ down  player | turn ]
    right [ right player | turn ] -> [ down  player | turn ]
    down  [ down  player | turn ] -> [ up    player | turn ]

so pressing LEFT means "go left; if that is blocked go up; if that is blocked
go down", and pressing RIGHT means "go right, else down, else up" -- each key
is ``[d, CW(d), CCW(d)]``. If all three are blocked nothing moves and
``require_player_movement`` cancels the turn, which the search sees as a
non-edge. ``turn`` is ``playerpart or wall``, so the snake's own NECK is always
one of the blockers, and that is exactly what makes two keys enough to steer:
you can never reverse, so "left" and "right" behave like "bear one way" and
"bear the other".

A Crate blocks too, but only when it cannot be pushed: the second half of each
wiggle rule (``[ left player | crate | obj ]``) fires when there is an ``obj``
BEHIND the crate, and otherwise ``[ > Player | Crate ] -> [ > Player | > Crate ]``
shoves it one cell.

**The board EDGE does not wiggle you -- it cancels the turn.** Both wiggle
shapes name cells beyond the player (``| turn`` needs one, ``| crate | obj``
needs two), and a PuzzleScript pattern cannot match off the board, so neither
rule fires when what stops you is the edge itself. The force stays on, the move
fails for want of anywhere to go, and ``require_player_movement`` cancels the
whole turn instead of bearing a quarter turn. It is a real difference in
outcome, not a formality: on the spike-bordered levels (5, 6, 8) the outer ring
is Spike, which is walkable (and lethal), so the true edge is reachable, and a
crate shoved against it becomes a cell that eats presses. ``--keys`` found this
-- the first model of the chain here predicted a quarter turn there and
disagreed with the interpreter on 22 of 21338 live presses.

Two more consequences of "the pattern must fit on the board": an Apple against
the edge cannot be eaten (``[ > Player | Apple | ]`` names a third cell) and
blocks like a wall instead, and a Crate against the edge can never be pushed
again.

**The pink "tongue" is the game's own built-in hint** and it is COSMETIC. The
31 ``(ui)`` rules re-simulate both keys every turn on dummy ``dir`` / ``tmark``
tiles and paint a pink pip in the cell ahead of the head showing where each key
would take you (``t_left_down`` = "left keeps going left, right turns down").
Together with the two ``covered`` rules that is 33 of the game's 63 rules, and
they are 2/3 of the cost of an interpreter step -- see `_lean_rules`.

**Death is a FREEZE, not a restart.** ``late [playerpart spike] -> [blood
spike]`` turns the part that touched the spike into Blood, and ``[blood]
[moving player] -> [blood] [stationary player]`` then pins the player forever.
So a spiked board is a sink with no outgoing edges, not a level restart:
``_rule_restart`` is never set anywhere in this game. `SnekExpert.dead` drops
those successors instead of queueing them.

**Apples are the only way the snake grows.** ``left [ > Player | Apple | ] ->
[ bod Leftmarker | Player | ]`` moves the head ONTO the apple cell and leaves a
new body segment behind, so eating is a head visit to the apple's own cell.

Board symmetry
--------------
Snek gets the rot90 frame augmentation and NO flips (it is in neither
`PuzzleScriptAdapter._HFLIP_GAMES` nor `_FLIP_GAMES`), and that is correct
rather than incidental: the wiggle chain is CHIRAL. "Left, else CW, else CCW"
mirrors to "right, else CCW, else CW", which is not a rule this game has, so a
mirrored Snek is a different game. Rotation is fine -- the chain is stated
relative to the direction of travel, so it turns with the board. Do not add
Snek to the flip sets.

The expert
----------
`SnekExpert` is A* over the real interpreter (`PSExpert`), with four
game-specific pieces: three admissible lower bounds whose MAXIMUM is the
heuristic (the max of admissible bounds is admissible), and a search that
returns a distance field.

**1. An admissible heuristic, from one structural fact.** Every uncovered
Target ``t`` forces the HEAD to visit the closed neighbourhood ``N[t]`` (``t``
and its four neighbours), because at the win ``t`` holds a playerpart or a
crate, and

* a playerpart on ``t`` means the head was on ``t`` -- every body cell is a
  cell the head occupied earlier (the ``bod`` / Marker follow chain, and the
  apple rule, only ever place a segment where the head just was);
* a crate landing on ``t`` was pushed there from ``t - d`` by a head that ends
  the push standing on ``t - d``.

A Wall on a target would also satisfy the win, but nothing in this ruleset ever
creates a wall, so that case cannot arise. The heuristic is therefore a lower
bound on a WALK from the head that visits ``N[t]`` for every uncovered target:
an open TSP path over set-to-set distances, solved exactly by Held-Karp on the
static per-level tables (`_bind`) and read off in a handful of lookups per node.
Distances are BFS distances over the cells that are neither Wall nor Spike --
spikes are permanent no-go for the head, so blocking them keeps the estimate a
lower bound rather than breaking it.

**2. Apples enter the same bound when they are provably forced.** The snake can
only body-cover a set ``S`` of targets if it is long enough to be a simple path
through all of them, i.e. ``length >= 1 + TSPpath(S)``. Crates cover at most one
target each, so at least ``|uncovered| - |crates|`` targets must be body-covered,
which gives a required final length and hence a minimum number of apples that
MUST be eaten. When that count reaches the number of apples left, every apple
cell joins the required-visit set; when it is smaller, the cheapest single apple
does. This is what makes level 5 tractable -- its five targets sit in an X that
no path of fewer than nine cells can cover, the snake starts at five, and there
are exactly four apples, so all four are forced.

**3. Crates enter it the same way when THEY are forced.** The same
length argument run the other way says at least ``|uncovered| - capacity``
crates have to be pushed, where ``capacity`` is the most targets the snake
could ever span (its current length plus every apple left). A crate that is
going to be pushed has to be walked to first, so the head must also visit
``N[c]`` for that many crates. Crates MOVE, so unlike the other two this bound
cannot use the level-start tables: it is a minimum spanning tree (a lower bound
on any spanning path, hence on the walk) over all-pairs distances at the
crates' CURRENT cells, minimised over which crates are chosen. A few hundred
operations against an interpreter step of about a millisecond.

**4. A memory-lean search that returns a distance FIELD, not a path.** The
shared `PSExpert._astar` keeps a full grid snapshot (a list of ``h*w`` python
sets, ~28 KB) inside every heap entry. On this game's deeper levels that is tens
of thousands of live nodes and it will exhaust a 32 GB machine. `_search` here
keeps a 2-bytes-per-cell key instead (~290 B) and rebuilds the grid from a
static template on pop, which is the same work `restore` already did.

Having paid for the search, it then keeps the EDGES it expanded and runs the
shared backward BFS over them, so what comes back is a `StateGraph` -- and every
plan step gets its EXACT optimal set (``dist(succ) == dist - 1``) instead of an
inferred one. That is sound on the pruned ball and not just on a full
enumeration: the search runs until the frontier's ``f`` passes the winning cost
``d*``, and every state on a shortest continuation of a plan state satisfies
``g + h <= d*``, so it is inside the ball. A press that is genuinely optimal
therefore always has an exact distance; one that is not can only be
over-estimated, which still keeps it out of the set.

What it finds
-------------
``--plans``, seed 0. Every one of these is PROVED SHORTEST, in the sense the
search supports: an admissible heuristic, unit weight, and a run continued past
the first win until the frontier's ``f`` cleared it.

    level  board   targets crates apples  presses  A* ball    time
      0     9x9       1      0      0        7         36     0.1s
      1     9x13      2      0      0        8         36     0.1s
      2     9x13      2      1      0       13        182     0.4s
      3     9x13      3      3      0       26       2806     5.9s
      4    11x11      3      0      0       15       1303     2.7s
      5    11x11      5      0      4       61     143715   381.1s
      6    11x11      3      3      0       41     349104   771.9s
      7    11x11      3      1      0       35      22039    49.6s
      8    11x11      8      2      2       52      30784    66.2s

All nine, 258 presses, 20 of them with a genuine tie. Nothing is skipped.

Levels 5 and 6 are the two the extra bounds were built for, and they are the
two the plain visit-the-targets estimate is hopeless on.

* **Level 5** is the apple case. Five targets in an X that no path of fewer
  than nine cells can cover, a snake that starts at five, and exactly four
  apples -- so all four are forced, and the estimate goes from "walk to the
  nearest target" to "tour all four corners, then coil".
* **Level 6** is the crate case. Its three targets sit within a few steps of
  the snake's nose, so the target bound scores about 7 against a real answer of
  41; but no two of those targets are within seven cells of each other, a
  five-long snake cannot span any pair, and there are no apples -- so at least
  two of the three crates provably have to be fetched from the corners they sit
  in, and the crate bound charges for that walk.

Level 6 is also the one that decides `node_cap`: its ball is 349104 states, so
the cap has to be several times that to be a runaway guard rather than a
silent skip.

Why not exhaustive enumeration
------------------------------
The other enumerating ps: generators (ps:roller_boi, ps:slippy_penguin) get away
with `StateGraph.build` because one press is a whole slide and the reachable
space collapses to tens of states. Snek's press is one CELL, so its space does
not collapse. Measured: 1450 reachable states on level 0, 3078 on level 1, 3291
on level 4 -- and past 40000 on every other level. Level 3 is the clearest case.
Its plain breadth-first ball was still growing 1.4x per layer at depth 23
(128235 states, 8 minutes) with the goal at depth 26, i.e. roughly a million
states away from an answer; the heuristic reaches the same 26 presses after
2806.

``--enumerate`` runs the exhaustive version anyway on the three levels where it
fits and checks that it agrees, which is the real test of the heuristic: an
inadmissible estimate would show up there as a plan the exhaustive field says is
too long. All three agree (7, 8, 15). Levels 7 and 8 were checked the same way
before this generator existed, with a plain breadth-first search on the same
interpreter: d* = 35 and d* = 52, which is what ships.

Level indices here and in every report are 0-BASED.

Plans are cached in ``data/snek_plans.json``: they are seed-independent (only
the PRESENTATION is augmented per seed), and the deep levels cost minutes, so
without the file every `parallelize_generator` shard would re-derive all nine.

Rendering: the win was not in the picture
----------------------------------------
``Covered`` is the mark the game paints on a satisfied target, and it is the
only thing in a frame that says how close the level is to its ``all target on
obj`` win. It was invisible on every level, for two independent reasons:
`_render_frame` sorts player objects LAST ("so the player is always visible on
top"), which overrides this game's own collision layers and paints the head over
the dot, and the Crate sprite is opaque across the whole cell. So a covered
target under an up- or right-facing head, or under a crate, was drawn exactly
like an uncovered one -- and every level ENDS with at least one of those, which
means the frame each episode won on was identical to a frame that had not won.

``games/ps:snek/ps:snek.py`` fixes it with two pixels (a Covered dot in each of
the two opposite corners, which no head or crate sprite paints over). ``--audit``
proves it both ways: no two producible cell compositions render alike at any of
the three cell sizes, and every target's covered-ness changes the winning frame.
Both halves MEASURE rather than name the sprites they were written against, so
neither can go stale into a vacuous pass.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the
cached plan replays to a guaranteed WIN. The reset is load-bearing here rather
than decorative -- this game is full of one-way doors. A snake that coils into
its own body has no legal press left, a crate shoved against a wall can never
come back, and a single touched spike freezes the board permanently. None of
those can be re-planned out of, so RESET is the only recovery, exactly as in the
rest of the ps: family.

CLI
---
    --plans      per-level board, plan length, search size and tie coverage
    --keys       prove up/down/ACTION are no-ops and the wiggle chain is [d,CW,CCW]
    --fuzz       prove the lean rule list is a speed-up and nothing else
    --ties       re-derive every optimal-action label by independent re-solve
                 (expensive -- run it a level or two at a time)
    --enumerate  cross-check the plan against an exhaustive `StateGraph`
    --audit      assert every cell composition renders distinctly at every
                 size, and that each level's WIN is visible in its own frame
    --fresh      make any of the above ignore the disk plan cache
    --levels=A,B restrict any of the above to those levels
"""

from __future__ import annotations

import heapq
import random
import sys
import time
from collections import deque
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402

from solvers.common.ps_astar import (                       # noqa: E402
    WIN, PSAStarSolver, PSExpert, StateGraph, restore, snapshot,
)

GAME_NAME = "Snek"

#: The two live presses. Up, down and ACTION are measured no-ops on every state
#: (see the module docstring and ``--keys``); branching on them would double the
#: interpreter steps for edges that are always self-loops. The order is the
#: plan's tie-break, so it is fixed.
_DIRS = ("left", "right")

#: Or-groups whose members are pure PRESENTATION: the pink tongue hint, the
#: dummy tiles the hint is simulated on, and the red dot drawn over a covered
#: target. `_lean_rules` drops every rule that touches one.
_COSMETIC_GROUPS = ("covered", "dir", "tmark", "tongue")

#: Objects that make up the DYNAMIC state. Everything else on the board is
#: either static scenery (background / wall / target / spike -- no rule in this
#: game creates or destroys any of them) or cosmetic.
_DYNAMIC_NAMES = ("player", "bod", "blood", "apple", "crate", "marker")

_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Heuristic charge for a state that provably cannot win. Large enough to sink
#: the node behind everything real, finite so the search stays complete.
_DEAD = 1 << 20


# ---------------------------------------------------------------------------
# Retiring the cosmetic rules (for the SEARCH only)
# ---------------------------------------------------------------------------

def _rule_object_ids(rule, known: set) -> set:
    """Every object index named anywhere in ``rule``'s patterns."""
    out: set = set()
    for side in (rule.patterns_lhs, rule.patterns_rhs):
        for pattern in side:
            for cell in pattern:
                for item in cell:
                    if not isinstance(item, tuple):
                        continue
                    for part in item:
                        if isinstance(part, int):
                            out.add(part)
                        elif isinstance(part, (list, tuple, set, frozenset)):
                            out |= {x for x in part if isinstance(x, int)}
    return out & known


def _lean_rules(parsed) -> list:
    """``parsed.rules`` with every COSMETIC rule dropped (63 -> 30).

    A rule is cosmetic here iff it names one of `_COSMETIC_GROUPS`, and that
    one-line test happens to be exact for this game: the 31 ``(ui)`` tongue
    rules and the two ``covered`` rules are the only ones that mention those
    objects, and NO other rule reads them. That second half is what makes the
    drop sound rather than merely plausible -- since nothing else can see a
    cosmetic object, removing the rules that write them cannot change how any
    surviving rule matches. Their collision layers agree: ``covered``,
    ``tongue``, ``dir`` and the four ``tmark`` layers are theirs alone, so a
    leftover cosmetic tile cannot block a moving snake either.

    Worth doing because they are 2/3 of the work: measured 354 -> 925
    interpreter steps/s on level 3. ``--fuzz`` replays random presses on an
    untouched interpreter and on this one and requires the dynamic state to
    agree at every press.

    Used for SEARCH only. The tongue is on screen and the corpus is made of
    frames, so the recording adapter always runs the full list -- which is why
    this is a context around `SnekExpert._search` rather than the permanent
    swap `solvers/generate_savior_training.py` installs.
    """
    known = set(parsed.obj_name_to_idx.values())
    cosmetic = {i for name in _COSMETIC_GROUPS
                for i in parsed.resolve_object_name(name)}
    return [r for r in parsed.rules
            if not (_rule_object_ids(r, known) & cosmetic)]


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class SnekExpert(PSExpert):
    """A* over the real interpreter, on two presses, with the visit-the-target
    heuristic and a search that returns a distance field.

    See the module docstring for the heuristic's derivation and for why the
    shared `PSExpert._astar` is replaced rather than reused.
    """

    directions = list(_DIRS)

    #: `_key` is the DYNAMIC objects only, which is canonical within a level but
    #: not across them (walls, spikes and targets are static per level yet
    #: differ between levels).
    scope_by_level = True

    #: The searches are minutes on the deep levels and seed-independent, so
    #: every shard would otherwise re-derive all nine.
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "snek_plans.json")

    def setup(self) -> None:
        g = self.g
        self.dyn_ids = {i for name in _DYNAMIC_NAMES
                        for i in g.resolve_object_name(name)}
        self.player_ids = set(g.resolve_object_name("player"))
        self.part_ids = set(g.resolve_object_name("playerpart"))
        self.obj_ids = set(g.resolve_object_name("obj"))
        self.blood_id = g.obj_name_to_idx["blood"]
        self.apple_id = g.obj_name_to_idx["apple"]
        self.crate_id = g.obj_name_to_idx["crate"]
        self.target_id = g.obj_name_to_idx["target"]
        self.wall_id = g.obj_name_to_idx["wall"]
        self.spike_id = g.obj_name_to_idx["spike"]
        self.lean = _lean_rules(g)
        #: id -> bit, for the compact state key.
        self.bit = {o: 1 << i for i, o in enumerate(sorted(self.dyn_ids))}
        self._bound: tuple | None = None      # signature of the bound board

    # -- state key ----------------------------------------------------------
    def _key(self, eng) -> frozenset:
        """Dynamic objects only. Used by `PSExpert.plan`'s disk-cache signature
        and by the plan memo; the SEARCH uses `_encode` instead."""
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    def _encode(self, eng) -> bytes:
        """The same state as `_key`, as two bytes per cell.

        This is the whole memory story of the search: a `snapshot` of an 11x13
        board is 143 python sets (~28 KB) and the shared A* holds one per live
        heap entry, which on level 8 is gigabytes. This is ~290 bytes and the
        grid is rebuilt from `_template` on pop -- the same set-building work
        `restore` was doing anyway."""
        bit = self.bit
        out = bytearray()
        for row in eng.grid:
            for cell in row:
                v = 0
                for o in cell:
                    b = bit.get(o)
                    if b is not None:
                        v |= b
                out += v.to_bytes(2, "little")
        return bytes(out)

    def _decode(self, eng, key: bytes) -> None:
        """Write the state ``key`` back onto ``eng.grid``.

        The static scenery (and any leftover cosmetic tile) comes from
        `_template`, which is sound because no rule in this game creates or
        destroys a Background, Wall, Target or Spike -- ``--fuzz`` and
        ``--enumerate`` both exercise it."""
        inv = self._inv_bits
        grid = []
        i = 0
        for row in self._template:
            new = []
            for base in row:
                v = key[i] | (key[i + 1] << 8)
                i += 2
                new.append(set(base) | {o for b, o in inv if v & b})
            grid.append(new)
        eng.grid = grid
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    # -- per-level static tables --------------------------------------------
    def _bind(self, eng) -> None:
        """Build this board's distance tables. Cheap, and keyed by the static
        layout so re-binding the same level is free.

        Everything here is a function of the WALLS, SPIKES, TARGETS and the
        apples' cells, none of which any rule moves (an apple is eaten, never
        relocated), so the tables are valid for the whole level. Crates are
        deliberately absent -- they DO move, so a crate-aware table would have
        to be rebuilt per node."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        blocked = {self.wall_id, self.spike_id}
        free = [[not (grid[r][c] & blocked) for c in range(w)]
                for r in range(h)]
        targets = [(r, c) for r in range(h) for c in range(w)
                   if self.target_id in grid[r][c]]
        apples = [(r, c) for r in range(h) for c in range(w)
                  if self.apple_id in grid[r][c]]
        sig = (h, w, tuple(map(tuple, free)), tuple(targets), tuple(apples))
        if self._bound == sig:
            return
        self._bound = sig
        self.targets, self.apples = targets, apples

        self._template = [
            [cell - self.dyn_ids for cell in row] for row in grid]
        self._inv_bits = [(b, o) for o, b in self.bit.items()]

        def bfs(sources) -> dict:
            dist, queue = {}, deque()
            for cell in sources:
                r, c = cell
                if 0 <= r < h and 0 <= c < w and free[r][c] and cell not in dist:
                    dist[cell] = 0
                    queue.append(cell)
            while queue:
                r, c = queue.popleft()
                for dr, dc in _DELTA:
                    nxt = (r + dr, c + dc)
                    if (0 <= nxt[0] < h and 0 <= nxt[1] < w
                            and free[nxt[0]][nxt[1]] and nxt not in dist):
                        dist[nxt] = dist[(r, c)] + 1
                        queue.append(nxt)
            return dist

        # A requirement is a set of cells the head must ENTER. For a target that
        # is its closed neighbourhood (it is covered either by the head standing
        # on it or by a crate pushed on from a neighbour); for an apple it is
        # the apple's own cell, because eating moves the head onto it.
        self.req_cells = (
            [[t] + [(t[0] + dr, t[1] + dc) for dr, dc in _DELTA]
             for t in targets]
            + [[a] for a in apples])
        self.n_t, self.n_a = len(targets), len(apples)
        n = self.n_t + self.n_a
        self.reach = [bfs(cells) for cells in self.req_cells]
        # Set-to-set distance: enter requirement j from wherever requirement i
        # was left. Entering and leaving a set at different cells is a
        # relaxation of the real walk, so the bound stays a lower bound.
        sd = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                best = min((self.reach[i][cell] for cell in self.req_cells[j]
                            if cell in self.reach[i]), default=_DEAD)
                sd[i][j] = best
        # Q[mask][i]: the shortest walk that visits every requirement in `mask`
        # and ends at `i`. Independent of where the head is, so it is built once
        # per level; `sd` is symmetric (grid distance is), so the same table
        # answers "starts at i" and the head's own leg is added at query time.
        Q = [[_DEAD] * n for _ in range(1 << n)]
        for i in range(n):
            Q[1 << i][i] = 0
        for mask in range(1, 1 << n):
            row = Q[mask]
            for i in range(n):
                cur = row[i]
                if cur >= _DEAD or not (mask >> i) & 1:
                    continue
                for j in range(n):
                    if (mask >> j) & 1 or sd[i][j] >= _DEAD:
                        continue
                    nxt = mask | (1 << j)
                    if cur + sd[i][j] < Q[nxt][j]:
                        Q[nxt][j] = cur + sd[i][j]
        self.Q = Q
        # Target-CELL distances and their TSP table, for "how long must the
        # snake be to body-cover this set of targets".
        tsp = [[_DEAD] * self.n_t for _ in range(1 << self.n_t)]
        for i in range(self.n_t):
            tsp[1 << i][i] = 0
        # `reach[i]` measures to the NEIGHBOURHOOD of target i; the path through
        # the target cells themselves is what bounds the snake's length, so read
        # cell-to-cell distances off a fresh BFS from each target cell.
        cell_dist = [bfs([t]) for t in targets]
        td = [[cell_dist[i].get(targets[j], _DEAD) for j in range(self.n_t)]
              for i in range(self.n_t)]
        for mask in range(1, 1 << self.n_t):
            row = tsp[mask]
            for i in range(self.n_t):
                cur = row[i]
                if cur >= _DEAD or not (mask >> i) & 1:
                    continue
                for j in range(self.n_t):
                    if (mask >> j) & 1 or td[i][j] >= _DEAD:
                        continue
                    nxt = mask | (1 << j)
                    if cur + td[i][j] < tsp[nxt][j]:
                        tsp[nxt][j] = cur + td[i][j]
        #: `path_cells[mask]` = the fewest cells a simple path covering every
        #: target in `mask` can have, i.e. the shortest snake that could cover
        #: them all at once.
        self.path_cells = [1 + min(row) if mask else 0
                           for mask, row in enumerate(tsp)]
        # All-pairs distances, for the crate bound: a crate MOVES, so the
        # distance from the head to the one it has to go and shove cannot come
        # off a table keyed on where the crates were at level start. Built over
        # EVERY free cell rather than over one requirement's reachable set --
        # a missing entry reads as "unreachable", which would let the estimate
        # claim a winnable state is hopeless and stop being a lower bound.
        self.allD = {(r, c): bfs([(r, c)])
                     for r in range(h) for c in range(w) if free[r][c]}
        self._hmemo: dict = {}
        self._needmemo: dict = {}
        self._bodymemo: dict = {}

    # -- the heuristic ------------------------------------------------------
    def _read(self, eng) -> tuple:
        """``(head, uncovered-target mask, snake length, apple mask, the cells
        of the crates not already parked on a target)`` -- one pass over the
        grid, which is all the heuristic reads.

        The crate CELLS, not just how many there are: they are part of
        `heuristic`'s memo key, and a memo keyed on the count alone would serve
        a state its answer from a state whose crates are somewhere else."""
        head = None
        length = 0
        mask = 0
        amask = 0
        crates = []
        target_id, crate_id = self.target_id, self.crate_id
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & self.player_ids:
                    head = (r, c)
                if cell & self.part_ids:
                    length += 1
                if crate_id in cell and target_id not in cell:
                    crates.append((r, c))
        for i, t in enumerate(self.targets):
            if not (eng.grid[t[0]][t[1]] & self.obj_ids):
                mask |= 1 << i
        for i, a in enumerate(self.apples):
            if self.apple_id in eng.grid[a[0]][a[1]]:
                amask |= 1 << i
        return head, mask, length, amask, tuple(crates)

    def _apples_forced(self, mask: int, crates: int, length: int) -> int:
        """A lower bound on how many apples still have to be eaten.

        Crates cover at most one target each, so at least ``|mask| - crates``
        of the uncovered targets end up under the snake itself; the snake is a
        simple path, so covering a set ``S`` of them at once needs at least
        ``path_cells[S]`` segments. Taking the cheapest such ``S`` of that size
        turns into a floor on the snake's FINAL length, and every apple adds
        exactly one segment."""
        want = bin(mask).count("1") - crates
        if want <= 0:
            return 0
        key = (mask, want)
        need = self._needmemo.get(key)
        if need is None:
            bits = [i for i in range(self.n_t) if (mask >> i) & 1]
            need = min(self.path_cells[sum(1 << i for i in sub)]
                       for sub in combinations(bits, want))
            self._needmemo[key] = need
        return max(0, need - length)

    def _walk(self, head, mask: int) -> int:
        """The shortest walk from ``head`` visiting every requirement in
        ``mask`` -- the head's leg to whichever it does first, plus the
        Held-Karp table for the rest."""
        if not mask:
            return 0
        best = _DEAD
        Q = self.Q
        for i in range(self.n_t + self.n_a):
            if not (mask >> i) & 1:
                continue
            first = self.reach[i].get(head)
            if first is None:
                continue
            total = first + Q[mask][i]
            if total < best:
                best = total
        return best

    def _body_capacity(self, mask: int, length: int) -> int:
        """The most uncovered targets the snake could EVER cover at once.

        A snake is a simple path, so a set ``S`` of targets fits under it only
        if ``path_cells[S] <= length``. ``length`` here is the snake's maximum
        FUTURE length (what it is now, plus one per apple still on the board) --
        using its current length would let the estimate claim more crates are
        needed than really are, and the bound would stop being a bound."""
        key = (mask, length)
        got = self._bodymemo.get(key)
        if got is not None:
            return got
        bits = [i for i in range(self.n_t) if (mask >> i) & 1]
        best = 0
        for size in range(len(bits), 0, -1):
            if any(self.path_cells[sum(1 << i for i in sub)] <= length
                   for sub in combinations(bits, size)):
                best = size
                break
        self._bodymemo[key] = best
        return best

    def _crate_bound(self, crates, head, mask: int, extra: int,
                     need_crates: int) -> int:
        """A second admissible estimate, for the levels whose answer is crates.

        The visit-the-targets bound is blind to the fact that a crate has to be
        FETCHED: on level 6 its three targets sit within a few steps of the
        snake's nose and it scores about 7, while the real answer is dozens of
        presses spent walking out to three crates in three corners and shoving
        each one home.

        So: if ``need_crates`` of the crates provably have to be pushed, the
        head has to reach the neighbourhood of each of them (a push starts with
        the head standing beside the crate), on top of visiting every target.
        That is another walk-through-required-sets bound, and since the crates
        move it is measured against their CURRENT cells and scored with a
        minimum spanning tree rather than the static Held-Karp table -- an MST
        is a lower bound on any spanning path, hence on the walk, and it costs
        a few hundred operations against an interpreter step of about a
        millisecond.

        Which crates is not known, so the estimate is the MINIMUM over every
        choice of ``need_crates`` of them; taking the cheapest keeps it a lower
        bound. Returns 0 when no crate is forced, so `heuristic` can always take
        the max of the two.
        """
        if need_crates <= 0:
            return 0
        if len(crates) < need_crates:
            return _DEAD          # not enough crates left to finish
        req = [i for i in range(self.n_t + self.n_a)
               if ((mask | extra) >> i) & 1]
        # Node 0 is the head; then the requirement sets; then the crates.
        sets = [[head]] + [self.req_cells[i] for i in req]
        for (r, c) in crates:
            sets.append([(r, c)] + [(r + dr, c + dc) for dr, dc in _DELTA])
        n = len(sets)
        allD = self.allD
        dist = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                best = _DEAD
                for a in sets[i]:
                    row = allD.get(a)
                    if row is None:
                        continue
                    for b in sets[j]:
                        d = row.get(b)
                        if d is not None and d < best:
                            best = d
                dist[i][j] = dist[j][i] = best
        base = 1 + len(req)                   # head + the requirement sets
        pool = list(range(base, n))
        best_total = _DEAD
        for chosen in combinations(pool, need_crates):
            keep = list(range(base)) + list(chosen)
            # Prim, seeded at the head.
            inside = {keep[0]}
            edge = {v: dist[keep[0]][v] for v in keep[1:]}
            total = 0
            while edge:
                v = min(edge, key=edge.get)
                w = edge.pop(v)
                if w >= _DEAD:
                    total = _DEAD
                    break
                total += w
                inside.add(v)
                for u in list(edge):
                    if dist[v][u] < edge[u]:
                        edge[u] = dist[v][u]
            if total < best_total:
                best_total = total
        return best_total

    def heuristic(self, eng) -> int:
        head, mask, length, amask, crates = self._read(eng)
        if not mask:
            return 0
        if head is None:
            return _DEAD
        memo_key = (head, mask, length, amask, crates)
        got = self._hmemo.get(memo_key)
        if got is not None:
            return got
        forced = self._apples_forced(mask, len(crates), length)
        left = [i for i in range(self.n_a) if (amask >> i) & 1]
        if forced >= len(left):
            # Every apple still on the board has to be eaten, so each one is a
            # requirement in its own right.
            choices = [sum(1 << (self.n_t + i) for i in left)]
        elif forced >= 1:
            # At least one, but we do not know which -- charge the cheapest,
            # which is a lower bound on charging the right one.
            choices = [1 << (self.n_t + i) for i in left]
        else:
            choices = [0]
        best = min(self._walk(head, mask | extra) for extra in choices)
        # The crate bound is a SECOND lower bound on the same quantity, so the
        # larger of the two is still a lower bound -- and on the crate levels it
        # is the larger by a wide margin.
        need_crates = (bin(mask).count("1")
                       - self._body_capacity(mask, length + len(left)))
        if need_crates > 0:
            crate_h = min(self._crate_bound(crates, head, mask, extra,
                                            need_crates)
                          for extra in choices)
            if crate_h > best:
                best = crate_h
        self._hmemo[memo_key] = best
        return best

    def dead(self, eng) -> bool:
        """A spiked board is frozen forever, so it can never reach a win.

        ``[playerpart spike] -> [blood spike]`` plus ``[blood] [moving player]
        -> [blood] [stationary player]``: once any Blood exists the player
        cannot be given a force again, so every later press is cancelled by
        ``require_player_movement``. Dropping these instead of queueing them
        matters -- on the spike-ringed levels (5, 6, 8) walking into the border
        is a large slice of the branching."""
        blood = self.blood_id
        for row in eng.grid:
            for cell in row:
                if blood in cell:
                    return True
        return False

    # -- search -------------------------------------------------------------
    def _search(self, eng) -> "list | None":
        """`PSExpert._search`, but on the lean rule list.

        The engine grid is untouched by the swap: the cosmetic tiles already on
        the board just sit there (no surviving rule reads them, and none of them
        shares a collision layer with anything that moves) and `_plan_memoized`
        restores the grid afterwards either way."""
        parsed = self.g
        full = parsed.rules
        parsed.rules = self.lean
        try:
            return self._field_search(eng)
        finally:
            parsed.rules = full

    def _field_search(self, eng) -> "list | None":
        """A* that keeps the ball it expanded and returns the plan read off a
        backward distance field over it.

        Two things separate this from `PSExpert._astar`, both explained at
        length in the module docstring: heap entries carry a ~290-byte key
        instead of a ~28 KB grid snapshot, and the search does not stop at the
        first win -- it keeps going until the frontier's ``f`` passes the
        winning cost. That bound is what makes both results exact: it PROVES
        the plan shortest (the first win generated is only shortest if the
        heuristic is strictly positive away from the goal, which this one is
        not -- a head parked beside its last target scores zero), and it
        guarantees the ball contains every shortest continuation of every plan
        state, so the optimal SETS read off the field are exact too.
        """
        self._bind(eng)
        if eng.check_win():
            return []
        start = self._encode(eng)
        succ: dict = {}
        best_g = {start: 0}
        counter = 0
        steps = 0
        dstar: int | None = None
        pq = [(self.heuristic(eng), 0, counter, start)]
        while pq:
            f, g, _c, k = heapq.heappop(pq)
            if dstar is not None and f > dstar:
                break
            if g > best_g.get(k, 1 << 30):
                continue                      # a stale heap entry
            edges: dict = {}
            for direction in self.directions:
                self._decode(eng, k)
                eng.step(direction)
                steps += 1
                if eng.check_win():
                    edges[direction] = WIN
                    if dstar is None or g + 1 < dstar:
                        dstar = g + 1
                    continue
                if self.dead(eng):
                    continue
                nk = self._encode(eng)
                if nk == k:
                    continue                  # a press the engine refused
                edges[direction] = nk
                ng = g + 1
                if ng >= best_g.get(nk, 1 << 30):
                    continue
                best_g[nk] = ng
                counter += 1
                heapq.heappush(
                    pq, (ng + self.heuristic(eng), ng, counter, nk))
                if len(best_g) >= self.node_cap:
                    # Out of budget. Bail rather than return the incumbent: a
                    # win found before the ``f`` bound was reached is not proved
                    # shortest, and its optimal sets would be read off a field
                    # with holes in it. The level is then skipped by
                    # `discover_solvable`, which is the family's contract --
                    # better no data than data labelled with the wrong answer.
                    self._last = (len(succ), steps, None)
                    return None
            succ[k] = edges
        self._last = (len(succ), steps, dstar)
        if dstar is None:
            return None
        graph = StateGraph(succ, StateGraph._distances(succ), start, steps)
        return graph.plan(self.directions)

    #: Cap for the EXHAUSTIVE cross-check (``--enumerate``), deliberately far
    #: below `node_cap`. `StateGraph.build` keeps a full grid snapshot per state
    #: -- about 28 KB on these boards, which is the shape that will happily eat
    #: a 32 GB machine -- whereas `_field_search` keeps ~290 bytes. This is a
    #: memory budget, not a search budget: at 40000 states it peaks near 1 GB.
    enum_cap: int = 40_000

    def graph(self, eng) -> "StateGraph | None":
        """The exhaustive reachable-state graph, for ``--enumerate``. Only
        affordable on the small levels; `_field_search` is what ships."""
        parsed = self.g
        full = parsed.rules
        parsed.rules = self.lean
        try:
            return StateGraph.build(eng, self._key, self.directions,
                                    self.enum_cap)
        finally:
            parsed.rules = full


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class SnekSolver(PSAStarSolver):
    game_id = "puzzlescript_snek"
    game_name = GAME_NAME
    expert_cls = SnekExpert

    #: Record against the GAME FOLDER's adapter, and here that is load-bearing
    #: rather than defensive: ``games/ps:snek/ps:snek.py`` patches the Covered
    #: sprite so a satisfied target is visible under the snake's head and under
    #: a crate. Building a bare `PuzzleScriptAdapter` instead would tape frames
    #: in which the win is invisible, and would not match what `game_envs`
    #: hands a live agent (the ps:count_mover trap).
    game_module_id = "ps:snek"

    #: A ceiling on the A* ball, not a tuning dial: it exists so a level the
    #: heuristic cannot reach fails loudly instead of running the machine out of
    #: memory. The largest shipped search is level 6 at 349104 states, so this
    #: is roughly 3.5x the worst case. See ``--plans``.
    node_cap = 1_200_000

    #: Room for the longest plan (level 5's 61 presses) plus the re-plan after
    #: the exploration prefix. The prefix itself does not compete for this
    #: budget, nor for the adapter's own 200-action per-level cap: the RESET
    #: that ends it goes through ``set_level`` -> ``_do_reset``, which zeroes
    #: ``_action_count``.
    max_steps = 150

    supports_recovery = True
    recovery_mode = "reset"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0, fresh: bool = False):
    """A solver, its adapter and an expert.

    ``fresh`` detaches the disk plan cache for this process, so a report
    RE-DERIVES every plan instead of reading back the one it is meant to be
    checking. Any report that is supposed to prove something about the search
    has to pass it -- otherwise a heuristic change is verified against plans
    the old heuristic wrote."""
    solver = SnekSolver()
    game = solver.make_game(seed)
    expert = SnekExpert(game, node_cap=SnekSolver.node_cap)
    if fresh:
        expert.plan_cache_path = None
        expert._disk = {}
    return solver, game, expert


def _levels(argv) -> "list[int] | None":
    for arg in argv:
        if arg.startswith("--levels="):
            return [int(x) for x in arg.split("=", 1)[1].split(",")]
    return None


def _report(only, fresh: bool = False) -> int:
    """Per-level board, plan length, search size and tie coverage."""
    _solver, game, expert = _new(fresh=fresh)
    eng = game._engine
    bad = 0
    total = tied = 0
    for level in range(game.n_levels):
        if only is not None and level not in only:
            continue
        game.set_level(level)
        expert._bind(eng)
        crates = sum(1 for row in eng.grid for cell in row
                     if expert.crate_id in cell)
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        head = (f"level {level}: {eng.height:2d}x{eng.width:2d}, "
                f"{len(expert.targets)} targets, {crates} crates, "
                f"{len(expert.apples)} apples")
        if found is None:
            print(f"{head} -- NO PLAN within {expert.node_cap} states")
            bad += 1
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        tied += ties
        ball, steps, _d = getattr(expert, "_last", (0, 0, None))
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(found):3d} presses (budget {game._max_steps}, "
              f"{room}), {ties:3d} tied ({ties / max(1, len(found)):3.0%}), "
              f"ball {ball} states / {steps} engine steps, {dt:6.1f}s")
    print(f"total {total} presses, {tied}/{total} steps with a tie "
          f"({tied / max(1, total):.0%})")
    return 1 if bad else 0




def _walk_states(expert, eng, cap: int) -> list:
    """Every state reachable from the engine's current board, breadth-first,
    as compact keys. Stops at ``cap`` and restores the board.

    Uses the expert's own `_encode` / `_decode`, so a bug in either shows up
    here as well as in the search."""
    parsed = expert.g
    full, parsed.rules = parsed.rules, expert.lean
    try:
        expert._bind(eng)
        start = expert._encode(eng)
        seen = [start]
        index = {start}
        queue = deque([start])
        while queue and len(seen) < cap:
            k = queue.popleft()
            for direction in expert.directions:
                expert._decode(eng, k)
                eng.step(direction)
                if eng.check_win():
                    continue
                nk = expert._encode(eng)
                if nk != k and nk not in index:
                    index.add(nk)
                    seen.append(nk)
                    queue.append(nk)
        return seen
    finally:
        parsed.rules = full


def _keys(cap: int = 2500) -> int:
    """Prove the two claims the whole search rests on.

    1. **Up, down and ACTION are exact no-ops.** ``[up Player] -> [Player]``
       strips the force and ``require_player_movement`` then cancels the turn,
       so the board cannot change; ``noaction`` means no rule reads ACTION at
       all. If that failed anywhere, restricting `SnekExpert.directions` to two
       presses would be dropping real edges and the "shortest" plans would be
       shortest only among two-key plans.
    2. **A live key is the chain ``[d, CW(d), CCW(d)]``.** Pressing left goes
       left, else up, else down; right goes right, else down, else up; if all
       three are blocked the turn is cancelled. ``Blocked`` is the board edge, a
       wall, any snake part, or a crate with an ``obj`` immediately behind it.

    Both are checked across the REACHABLE states of every level (breadth-first,
    capped), not on the nine level starts -- a claim about the ruleset has to
    hold everywhere the search will go."""
    _solver, game, expert = _new()
    eng = game._engine
    chain = {"left": ("left", "up", "down"), "right": ("right", "down", "up")}
    delta = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    turn_ids = expert.part_ids | {expert.wall_id}
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        states = _walk_states(expert, eng, cap)
        parsed = game._game
        full, parsed.rules = parsed.rules, expert.lean
        dead_keys = live = 0
        try:
            for key in states:
                expert._decode(eng, key)
                before = expert._encode(eng)
                head = next(((r, c) for r, row in enumerate(eng.grid)
                             for c, cell in enumerate(row)
                             if cell & expert.player_ids), None)
                for direction in ("up", "down", "action"):
                    expert._decode(eng, key)
                    eng.step(direction)
                    if expert._encode(eng) != before:
                        bad += 1
                        print(f"  L{level}: {direction} CHANGED the board")
                    dead_keys += 1
                if head is None:
                    continue
                # Read the board back at THIS state -- the dead-key loop above
                # left the engine on its last (no-op) press.
                expert._decode(eng, key)
                grid_ro = [[set(cell) for cell in row] for row in eng.grid]

                def on(cell) -> bool:
                    return (0 <= cell[0] < eng.height
                            and 0 <= cell[1] < eng.width)

                def wiggles(cell, dr, dc) -> bool:
                    """Does a press in this direction get TURNED a quarter?

                    Only when the wiggle rule can actually match, and both of
                    its shapes need every cell they name to be ON the board:
                    ``[ left player | turn ]`` needs the neighbour, and
                    ``[ left player | crate | obj ]`` needs the cell past the
                    crate too. A crate shoved up against the board EDGE
                    therefore does NOT wiggle you -- the rules simply do not
                    fire, the force stays on, the push fails for want of
                    somewhere to go, and ``require_player_movement`` cancels the
                    whole turn. That is a different outcome from being turned,
                    and it is the one case the chain alone gets wrong."""
                    nxt = (cell[0] + dr, cell[1] + dc)
                    if not on(nxt):
                        return False
                    here = grid_ro[nxt[0]][nxt[1]]
                    if here & turn_ids:
                        return True
                    if expert.crate_id in here:
                        far = (nxt[0] + dr, nxt[1] + dc)
                        return on(far) and bool(
                            grid_ro[far[0]][far[1]] & expert.obj_ids)
                    return False

                def movable(cell, dr, dc) -> bool:
                    """Can the head actually take this step? Spikes are legal
                    (and lethal); a crate needs a free cell behind it; an apple
                    needs one too, because the eat rule names a third cell."""
                    nxt = (cell[0] + dr, cell[1] + dc)
                    if not on(nxt):
                        return False
                    here = grid_ro[nxt[0]][nxt[1]]
                    if (here & turn_ids) or expert.blood_id in here:
                        return False
                    far = (nxt[0] + dr, nxt[1] + dc)
                    if expert.crate_id in here:
                        return on(far) and not (
                            grid_ro[far[0]][far[1]]
                            & (expert.obj_ids | {expert.apple_id,
                                                 expert.blood_id}))
                    if expert.apple_id in here:
                        return on(far)
                    return True

                for press in expert.directions:
                    want = None
                    final = chain[press][-1]
                    for cand in chain[press]:
                        if not wiggles(head, *delta[cand]):
                            final = cand
                            break
                    if movable(head, *delta[final]):
                        want = (head[0] + delta[final][0],
                                head[1] + delta[final][1])
                    expert._decode(eng, key)
                    eng.step(press)
                    # The head is wherever the player sprite ended up -- or, if
                    # it walked onto a Spike, wherever the Blood that replaced
                    # it is. A death is a legal outcome of a legal move, and the
                    # chain still has to have predicted the cell it died on.
                    got = next(((r, c) for r, row in enumerate(eng.grid)
                                for c, cell in enumerate(row)
                                if cell & expert.player_ids), None)
                    if got is None:
                        got = next(((r, c) for r, row in enumerate(eng.grid)
                                    for c, cell in enumerate(row)
                                    if expert.blood_id in cell), None)
                    moved = expert._encode(eng) != before
                    if want is None:
                        if moved:
                            bad += 1
                            print(f"  L{level}: {press} moved from a state the "
                                  "chain calls fully blocked")
                    elif got != want:
                        bad += 1
                        print(f"  L{level}: {press} from {head} went to {got}, "
                              f"the chain says {want}")
                    live += 1
        finally:
            parsed.rules = full
        print(f"level {level}: {len(states):5d} states "
              f"({'capped' if len(states) >= cap else 'complete'}), "
              f"{dead_keys} dead-key presses and {live} live presses checked")
    print("keys clean" if not bad else f"KEYS FAILED: {bad} disagreements")
    return 1 if bad else 0


def _fuzz(trials: int = 8, steps: int = 60) -> int:
    """`_lean_rules` must be a speed-up and nothing else.

    Replay the same random presses on an untouched interpreter and on the lean
    one and require the DYNAMIC state -- everything the search and the win
    condition read -- to agree at every press. The frames are deliberately NOT
    compared: the lean list is exactly the rules that draw the pink tongue and
    the covered-target dot, so the two interpreters are *supposed* to render
    differently. That is also why the swap is scoped to `SnekExpert._search`
    and the recording adapter never sees it.

    ACTION and the two dead arrow keys are in the alphabet on purpose: a rule
    drop that resurrected one of them would be invisible to a two-key replay.
    """
    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    plain = PuzzleScriptAdapter(GAME_NAME, seed=0)
    solver, lean, expert = _new()
    # `_lean_rules` returns a fresh list, but the swap below mutates the parsed
    # game in place; a shared parse would make this compare an interpreter with
    # itself and pass for the wrong reason.
    assert plain._game is not lean._game, "the two adapters share one parse"
    alphabet = ["left", "right", "up", "down", "action"]
    bad = 0
    for level in range(plain.n_levels):
        for trial in range(trials):
            rng = random.Random(1000 * level + trial)
            plain.set_level(level)
            lean.set_level(level)
            expert._bind(lean._engine)
            parsed = lean._game
            full, parsed.rules = parsed.rules, expert.lean
            try:
                if expert._key(plain._engine) != expert._key(lean._engine):
                    bad += 1
                    print(f"  L{level} t{trial}: the seated levels differ")
                    continue
                for step in range(steps):
                    direction = rng.choice(alphabet)
                    plain._engine.step(direction)
                    lean._engine.step(direction)
                    if expert._key(plain._engine) != expert._key(lean._engine):
                        bad += 1
                        print(f"  L{level} t{trial}: diverged at press {step} "
                              f"({direction})")
                        break
                    if plain._engine.check_win() != lean._engine.check_win():
                        bad += 1
                        print(f"  L{level} t{trial}: win disagrees at {step}")
                        break
            finally:
                parsed.rules = full
    print("fuzz clean" if not bad else f"FUZZ FAILED: {bad} divergences")
    return 1 if bad else 0


def _ties(only, fresh: bool = True) -> int:
    """Re-derive every optimal-action label the independent way.

    The shipped sets come from the backward field over the A* ball. This takes
    each candidate press, re-solves from the state it lands in with a FRESH
    search, and calls the press optimal iff ``1 + len(replan)`` equals the plan
    length remaining -- so it checks the field, the ball's completeness and the
    plan-following against a second search rather than against themselves.

    EXPENSIVE, and much more so than `_report`: the shipped sets come out of ONE
    search, whereas this runs two per plan step, and the one that re-solves from
    the press that is NOT optimal has to search all the way to that press's own
    (longer) winning depth -- a strictly bigger ball than the plan's. Budget it
    per level (``--ties --levels=0,1,2``) rather than running the set."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        if only is not None and level not in only:
            continue
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level}: no plan (skipped)")
            continue
        parsed = game._game
        full, parsed.rules = parsed.rules, expert.lean
        checked = 0
        try:
            for pi, (taken, claimed) in enumerate(zip(plan, plan.optsets)):
                remaining = len(plan) - pi
                here = snapshot(eng)
                measured = []
                for direction in expert.directions:
                    restore(eng, here)
                    eng.step(direction)
                    if eng.check_win():
                        cost = 1
                    else:
                        rest = expert._field_search(eng)
                        cost = None if rest is None else 1 + len(rest)
                    if cost == remaining:
                        measured.append(direction)
                restore(eng, here)
                if measured != list(claimed):
                    print(f"  level {level} step {pi}: labelled "
                          f"{list(claimed)} but measured {measured}")
                    bad += 1
                if taken not in claimed:
                    print(f"  level {level} step {pi}: took {taken}, not in "
                          f"its own optimal set {list(claimed)}")
                    bad += 1
                checked += 1
                eng.step(taken)
        finally:
            parsed.rules = full
        print(f"level {level}: {checked} steps verified")
    print("ties clean" if not bad else f"TIES FAILED: {bad} mismatches")
    return 1 if bad else 0


def _enumerate(only, fresh: bool = True) -> int:
    """Cross-check the shipped plan against an EXHAUSTIVE enumeration.

    On the levels small enough to afford it, `StateGraph.build` generates every
    state the interpreter admits and a backward BFS gives each one's exact
    distance to a win. That is an independent answer to both questions the A*
    ball answers -- the shortest plan length, and whether a level is winnable at
    all -- computed with no heuristic, no ordering and no pruning. Where the two
    disagree, the heuristic is wrong.

    Levels past the cap print their size and are skipped; see the module
    docstring for why they are past it."""
    _solver, game, expert = _new(fresh=fresh)
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        if only is not None and level not in only:
            continue
        game.set_level(level)
        expert._bind(eng)
        t0 = time.time()
        graph = expert.graph(eng)
        dt = time.time() - t0
        if graph is None:
            print(f"level {level}: past the {expert.enum_cap}-state cap "
                  f"after {dt:.1f}s -- skipped (this is expected)")
            continue
        exhaustive = graph.plan(expert.directions)
        game.set_level(level)
        found = expert.plan(eng, level)
        live = len(graph.dist)
        got = None if found is None else len(found)
        want = None if exhaustive is None else len(exhaustive)
        verdict = "agree" if got == want else f"DISAGREE (A* {got})"
        if got != want:
            bad += 1
        print(f"level {level}: {len(graph.succ):6d} reachable states "
              f"({live} can still win, {len(graph.succ) - live} dead ends), "
              f"{graph.steps:6d} engine steps, {dt:6.1f}s -- exhaustive "
              f"shortest {want}, {verdict}")
    print("enumerate clean" if not bad
          else f"ENUMERATE FAILED: {bad} disagreements")
    return 1 if bad else 0


#: Cell compositions the game DRAWS identically and that nothing needs to tell
#: apart. `_audit` FAILS if an accepted pair does NOT actually render
#: identically, so an accept cannot go stale and quietly stop auditing a
#: composition an agent can now see.
_ACCEPTED: list = [
    # The four Markers are the snake's follow chain -- each segment carries the
    # direction it will step next -- and they are DELIBERATELY invisible: the
    # ``Marker`` collision layer is declared FIRST, i.e. below Background, so
    # the floor paints straight over them. Nothing is hidden by that. The
    # chain is a function of the body's own geometry (each segment moves toward
    # the one in front of it, and the head is the visible end), so a viewer of
    # the frame can read off everything a marker records.
    frozenset({"body", "body_upmark", "body_downmark",
               "body_leftmark", "body_rightmark"}),
]


def _audit(only=None) -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every cell size the boards use.

    Whole 64x64 frames of uniform boards are compared rather than one cell out
    of a mixed board: `_render_frame` centre-pads a non-square board, so
    indexing a cell by ``cell_px * r`` silently reads the wrong pixels on the
    wide levels (the ps:explod lesson). Two uniform boards render identically
    iff their cells do.

    The compositions are the ones the game actually produces: the four head
    facings, a body segment with each of the four Markers (the Marker layer is
    drawn UNDER the snake, so it must not change what a segment looks like --
    if it did, the marker would be leaking the follow-chain's internal state
    into the picture), a target with and without the ``covered`` dot, and the
    pink tongue pips that are this game's built-in hint. A tongue that rendered
    the same as a bare floor cell would mean the hint is invisible and the
    policy has to re-derive it."""
    import itertools

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, _expert = _new()
    eng, parsed = game._engine, game._game
    idx = parsed.obj_name_to_idx

    # Only compositions the RULES can actually produce. In particular
    # ``Covered`` is placed by ``late [obj target no covered]`` and removed by
    # ``late [no obj target covered]``, so it appears on an occupied target and
    # NEVER on a bare one -- auditing "target + covered" on its own would be
    # auditing a board the game cannot reach.
    comps: dict = {
        "floor": (),
        "wall": ("wall",),
        "spike": ("spike",),
        "target": ("target",),
        "apple": ("apple",),
        "crate": ("crate",),
        "blood": ("blood",),
        "blood_on_spike": ("spike", "blood"),
        "body": ("bod",),
    }
    for facing, name in (("up", "pup"), ("down", "pdown"),
                         ("left", "pleft"), ("right", "pright")):
        comps[f"head_{facing}"] = (name,)
        comps[f"head_{facing}_on_target"] = ("target", name, "covered")
    for mark in ("up", "down", "left", "right"):
        comps[f"body_{mark}mark"] = ("bod", f"{mark}marker")
    comps["body_on_target"] = ("target", "bod", "covered")
    comps["crate_on_target"] = ("target", "crate", "covered")
    for name in ("t_left_up", "t_left_down", "t_left_up_down",
                 "t_right_up", "t_right_down", "t_right_up_down",
                 "t_up_left", "t_up_right", "t_up_right_left",
                 "t_down_left", "t_down_right", "t_down_left_right"):
        comps[name] = (name,)
        # The hint is painted in the cell AHEAD of the head, which is often a
        # target -- the pip has to stay legible over one.
        comps[f"{name}_on_target"] = ("target", name)
    accepted = {frozenset({a, b})
                for group in _ACCEPTED
                for a, b in itertools.combinations(sorted(group), 2)}

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, parsed)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        ok = [p for p in clashes if frozenset(p) in accepted]
        hard = [p for p in clashes if frozenset(p) not in accepted]
        stale = sorted(tuple(sorted(pair)) for pair in accepted
                       if frozenset(pair) not in {frozenset(c) for c in clashes})
        bad += len(hard) + len(stale)
        note = "OK" if not hard else f"IDENTICAL {hard}"
        if ok:
            note += f" (+{len(ok)} accepted: {ok})"
        if stale:
            note += f" -- STALE ACCEPT (these render differently): {stale}"
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} "
              f"compositions -- {note}")
    bad += _audit_wins(only)
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} problems")
    return 1 if bad else 0


def _audit_wins(only=None) -> int:
    """Is each level's WIN legible in the frame that shows it?

    This is the part of the render audit with teeth, and it exists because of
    what the composition sweep above turned up. Snek's win condition is ``all
    target on obj`` and ``Covered`` is the only mark in the picture that says a
    target is satisfied -- but `_render_frame` sorts player objects LAST,
    deliberately, "so the player is always visible on top"
    (`adapters/puzzlescript_adapter.py`), which overrides this game's own
    collision layers and paints the head OVER the dot. Two of the four head
    sprites, ``pup`` and ``pright``, had no transparent pixel anywhere a dot
    landed, so a head facing up or right standing on a target was drawn exactly
    like a head standing on bare floor -- and four of the nine levels END that
    way. The winning frame was identical to a frame that had not won.
    ``games/ps:snek/ps:snek.py`` fixes it in two pixels; see the docstring
    there.

    The check MEASURES rather than naming the facings it was written against,
    so it cannot quietly stop testing: replay each plan to its win, then for
    every target render the board twice, once as it is and once with that one
    target's ``Covered`` mark taken away. If the two frames are equal, then
    nothing in the picture distinguishes "this target is satisfied" from "this
    target is not", and the level fails.
    """
    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, expert = _new()
    eng, parsed = game._engine, game._game
    covered_id = parsed.obj_name_to_idx["covered"]
    bad = 0
    for level in range(game.n_levels):
        if only is not None and level not in only:
            continue
        game.set_level(level)
        expert._bind(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"win frame level {level}: no plan -- skipped")
            continue
        full, parsed.rules = parsed.rules, expert.lean
        try:
            for direction in plan:
                eng.step(direction)
            # The lean list does not paint Covered, so put the marks the real
            # interpreter would have on before rendering.
            for (r, c) in expert.targets:
                if eng.grid[r][c] & expert.obj_ids:
                    eng.grid[r][c].add(covered_id)
            eng._position_index_dirty = True
            won = np.asarray(_render_frame(eng, parsed)).copy()
            covers: dict = {}
            invisible = []
            for (r, c) in expert.targets:
                cell = eng.grid[r][c]
                names = sorted(n for n, i in parsed.obj_name_to_idx.items()
                               if i in cell and n not in ("target", "covered",
                                                          "background"))
                key = "+".join(names) or "BARE"
                covers[key] = covers.get(key, 0) + 1
                if key == "BARE":
                    bad += 1                       # not actually a win
                    continue
                cell.discard(covered_id)
                eng._position_index_dirty = True
                without = np.asarray(_render_frame(eng, parsed))
                cell.add(covered_id)
                eng._position_index_dirty = True
                if np.array_equal(won, without):
                    invisible.append((r, c))
        finally:
            parsed.rules = full
        note = "legible"
        if invisible:
            note = f"COVERED MARK INVISIBLE at {invisible}"
            bad += len(invisible)
        print(f"win frame level {level}: "
              + ", ".join(f"{n}x {k}" for k, n in sorted(covers.items()))
              + f" -- {note}")
    return bad


if __name__ == "__main__":
    _only = _levels(sys.argv)
    _fresh = "--fresh" in sys.argv
    if "--plans" in sys.argv:
        sys.exit(_report(_only, _fresh))
    if "--keys" in sys.argv:
        sys.exit(_keys())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    if "--ties" in sys.argv:
        sys.exit(_ties(_only))
    if "--enumerate" in sys.argv:
        sys.exit(_enumerate(_only))
    if "--audit" in sys.argv:
        sys.exit(_audit(_only))
    sys.exit(SnekSolver.main())
