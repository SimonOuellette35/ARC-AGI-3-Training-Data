"""Generate Phase-1 training data for the PuzzleScript game ps:snakeoban
("Snakeoban", Jack Lance).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a NATIVE model of the
mechanic (fuzz-certified against the interpreter), the A* that plans on it, and
the reports that prove the plans win.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema::

    {
      "game_id": "puzzlescript_snakeoban",
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
exactly. Every expert step carries an optimal-action set (see "Optimal-action
sets" below).

The game
--------
Sokoban played by a SNAKE. Two win conditions at once -- ``All Crate on Target``
and ``No Apple`` -- so a level is over only when every crate sits on a target
AND every apple has been eaten, and eating is what makes the puzzle: the snake
grows by one segment per apple and its own body is the wall it has to plan
around.

Everything below was MEASURED against the interpreter (``--selfcheck`` re-runs
the measurement), not read off the .txt.

* **The snake is a marker trail.** Every cell a segment stands on carries a
  Marker naming the direction that segment leaves in; the head has none (its
  marker is dropped by ``[ left Player ] -> [ left Player Leftmarker ]`` on the
  turn it moves away). The bodies advance one cell per turn along that trail and
  the tail advances behind them, deleting its marker as it goes. So the snake is
  always the last ``L`` cells of the head's trajectory, and `_read_state`
  reconstructs it by walking the trail backwards from the head.

* **A press that does not move the head does nothing at all.** The game declares
  ``require_player_movement``, and the engine implements that by reverting the
  WHOLE turn when the player's position set is unchanged -- so a blocked shove, a
  wall bump and a head-into-body are all exact no-ops, not "wasted turns in which
  the body still slid".

* **ACTION5 is a no-op for the same reason.** No rule reads the action force, so
  an ACTION turn moves nothing and ``require_player_movement`` reverts it.
  `directions` therefore names four presses; branching on a fifth would add a
  quarter more search for edges that are all self-loops.

* **The head cannot follow its own tail.** Player, PlayerBody, PlayerTail, Wall,
  Crate and Apple share one collision layer, and the tail only moves in the LATE
  rules -- i.e. after the movement phase has already refused the head. The cell
  the tail is about to leave is blocked this turn and free the next one.

* **Eating needs a cell BEYOND the apple.** The rule is
  ``left [ > Player | Apple | ] -> [ PlayerBody Leftmarker | Player | SFX0 ]``:
  three cells, so an apple with nothing behind it (the board edge) cannot be
  eaten from that side at all, and since Apple is in the player's collision layer
  the press is then simply refused. Every level here is walled, so the
  third cell always exists -- the model implements the rule as written anyway.

* **Eating does NOT advance the tail.** The rule rewrites the old head cell into
  a body segment in place, so the whole trail behind it stays put and the snake
  is one cell longer. A plain move shifts every segment forward. That is the
  entire growth mechanic.

* **A crate cannot be pushed onto an apple** (same collision layer), so an apple
  lying on a crate's route has to be eaten before the crate can pass. On level 8
  that is the level: three crates, three targets, and an apple parked between
  each pair.

* **Crates chain-push** (``[ > Crate | Crate] -> [> Crate | > Crate]``) and are
  blocked by walls, apples and the snake alike. ``Crate2`` is nothing but
  ``Crate1`` standing on a Target -- two late rules convert back and forth -- so
  the model keeps one crate set and reads "on target" off the board.

* **GhostTarget and the four PlayerUp/Down/Left/Right sprites are cosmetics**:
  they live in their own collision layers, block nothing, and are recreated from
  scratch by the late rules on every turn (the first rule in the file,
  ``[Body] -> []``, wipes the connectors). They are what draws the snake as a
  joined-up body and a target as still-there under it. Markers are cosmetically
  invisible too, but for a different reason: ``Marker`` is the FIRST collision
  layer, i.e. painted UNDER the opaque Background.

* Nothing on the board is ever destroyed except apples, and nothing is ever
  created except snake segments. There is no death, no restart rule and no
  ``again``: every turn settles in one iteration.

The ten levels (0-based, as everywhere in this file) are 3x7 up to 11x9:

    lvl  size    snake  apples  crates/targets   presses
      0   3x7      3       1        -/-              2
      1   5x11     2       4        -/-             15
      2   8x8      2       6        -/-             24
      3   3x13     3       -        3/3              5
      4   7x7      3       1        4/4             25
      5   6x9      3       -        2/2             26
      6   9x9      2      30        -/-             58
      7   9x9      4       4        4/4             48
      8   9x8      2       4        3/3             74
      9  11x9      2      55        1/1        (skipped)

Why a native model, and not the shared engine-blackbox A*
---------------------------------------------------------
The interpreter runs at ~1.4k presses/second here, and the searches below take
millions of presses -- level 8 alone expands 3.3M. The native model runs the
same board at ~300k presses/second, which is the difference between a 20-second
startup and a three-hour one. The model is not trusted on the strength of the
rules section: ``--selfcheck`` drives random walks on the real interpreter and
compares the FULL state (snake chain, crates, apples, and the win flag) after
every single press. It was clean on the first 139k presses across all ten levels
and is re-run by ``--verify``.

The search
----------
Plain A* over the native states with an ADMISSIBLE estimate, so ``weight = 1``
plans are shortest:

    h = apples + pushes + approach

* ``apples`` -- one press per remaining apple, because a press eats at most one.
* ``pushes`` -- per-target push-distance tables (reverse BFS over push edges,
  ignoring apples, crates and the snake), combined by the BOTTLENECK of the best
  assignment of loose crates to bare targets: ``min`` over assignments of ``max``
  over pairs, brute-forced over permutations (never more than four crates).
  Not their SUM -- one press shoves a whole contiguous run of crates, so summing
  charged level 3 thirteen presses for a board that wins in five. A bare target
  no crate can reach at all scores `_DEAD`, which is also the deadlock test: a
  crate shoved into a corner has no outgoing push, so the reverse BFS never
  reaches it.
* ``approach`` -- presses that are neither an eat nor a push. Before the first
  eat there are at least ``dist(head, nearest apple) - 1`` of them, and before
  the first push at least ``dist(head, nearest stand cell)``; the smaller of the
  two is a lower bound on the whole prefix.

`_DEAD` is the only prune, and it is enough: the snake is its own pruning: a
press is legal only when the head has somewhere to go, so the branching factor
collapses as the body grows, and the 30-apple board (level 6) closes in 0.7M
native nodes.

A TIME-EXPANDED reachability prune -- flood from the head, opening a body
segment's cell at the earliest turn the tail could pass it, and demanding every
apple stay reachable -- was tried and REMOVED. It is unsound: a cell blocked at
the head's earliest arrival can still be entered later, by a longer route, so
the flood rules out states that win (it made level 1 unsolvable). Made sound, by
letting arrival wait for the opening turn, it degenerates to plain wall
connectivity and prunes nothing. It also cost more per node than the search it
was meant to save.

Level 9 is in `skip_levels`, and it is the only one. Its 7x9 field holds 55
apples and one crate, so a win is a near-Hamiltonian walk of the whole board
(final snake: 57 segments in 63 cells) that ALSO threads a >=10-push crate
delivery through it, early, before the body fills the space. Each half is
reachable on its own and the two together are not, which is what makes it hard
rather than merely big:

* apples alone -- a width-3000 beam ranked on apples-left plus a
  fragmentation penalty eats all 55 by depth 70, then wanders for another 300
  depths with the crate still sitting where it started. By then the snake is 57
  long and there is no room behind the crate to stand, so the delivery is not
  late, it is impossible.
* crate first -- A* on "all crates on target" alone, with the shipped push
  estimate, exhausts 4M native nodes without delivering. The estimate is blind
  to the reason it cannot: a crate may not be pushed onto an apple, so the route
  has to be EATEN clear first, and plain push distance charges nothing for that.
  Re-weighting the push graph (one press per push, one more per apple on the
  landing cell or the stand cell) fixes the guidance and a beam on it does climb
  -- from route-cost 15 down to 7 -- but has not delivered by depth 200.

So it is a search failure, and a documented one, not a mechanic the model gets
wrong: ``--selfcheck`` fuzzes level 9 exactly like the other nine and it is
clean. Nor is it the adapter's 200-press budget: the floor is 55 eats plus a
10-push delivery plus travel, and the apples-only beam does its half in 70, so a
combined solution has room. It is skipped rather than shipped half-solved.

Optimal-action sets
-------------------
``optsets[i]`` is measured, not inferred. At each plan step the alternatives are
re-solved from the state they land in under an f-BOUND of ``remaining - 1``
presses; a press joins the set only when a win inside that bound is actually
found. Two sound prunes keep it affordable -- a press whose admissible estimate
already exceeds the bound cannot tie, and neither can one the board refuses --
and a per-step node cap (`tie_cap`) stops the measurement from costing more than
the plan did. Hitting that cap is recorded as "not a tie", so the sets are
CONSERVATIVE: everything in one provably ties, and a long plan's early steps may
be labelled with the taken press alone rather than with a tie nobody could
afford to prove. The taken press is always in its own set, so no step is ever
emitted unlabelled.

Ties are genuinely rare here -- 14 of the 277 plan steps, 5% -- which is why
that is an acceptable trade, and the rarity is the mechanic rather than an
accident: unlike a sokoban walk, a snake's walk is never order-free. Two
interleavings of the same route leave the body in different cells, so they are
different states with different futures, and only rarely equally short ones.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an exploration prefix flails
first and ONE RESET restores the level start, from which the plan replays to a
guaranteed WIN. RESET is the honest key here, because the game is genuinely
irreversible in two ways -- an eaten apple never comes back, and there is no
rule that shrinks the snake, so a body that has grown into the wrong shape can
only be undone by starting over.

`SnakeobanSolver._reset_prefix` widens that RESET by one case the shared harness
skips: a prefix that WINS. Level 0 is two presses, and the exploration budget is
at its largest on the episode's first level, so a random flail finishes it
outright often enough to matter -- and the shared prefix, which resets only when
it did NOT win, then leaves a level made entirely of ``phase="explore"`` steps,
i.e. a WIN that `train_policy` masks out of the policy loss completely. See the
method.

Validated
---------
``--selfcheck 100``: 34880 presses over all ten levels against the interpreter,
0 mismatches. ``--plans``: all 9 levels planned, 277 presses, every step
labelled. ``--verify``: every plan replays through the adapter to
``GameState.WIN``, all 335 shipped labels execute on the interpreter, and each
one re-solves to exactly the remaining plan length (57 of those re-solves ran
out of the per-step budget and are reported as such rather than passed).
``--symmetry``: all 16 presentations win all 9 levels, with 0 board differences
over each plan plus a 60-press random walk. ``--audit``: clean at all 8 cell
sizes, 0 marker leaks. End to end, 20 seeds / 180 levels: every level a WIN,
every recorded action replays frame-exact, 5540/5540 expert steps labelled
(1.05 presses per set), no action index above 5, and two runs byte-identical.
Generation itself is the user's to run.

CLI
---
    --plans        per-level size, inventory, plan length and tie coverage
    --selfcheck N  fuzz N random walks per level against the interpreter
    --verify [N]   replay every plan through the ADAPTER and assert WIN, then
                   press every shipped optimal label on the interpreter and
                   re-solve from it (N native nodes per re-solve, default
                   `SnakeobanExpert.tie_cap`; raise it to shrink the
                   out-of-budget count the report prints)
    --symmetry     replay every plan at all rotations/flips the adapter uses
    --audit        assert every reachable cell composition renders distinctly
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402

from arcengine import GameState                             # noqa: E402
from solvers.common.ps_astar import (                       # noqa: E402
    PSAStarSolver, PSExpert, Plan, restore, snapshot,
)
from utils.explore import Action, RESET_ACTION              # noqa: E402

GAME_NAME = "Snakeoban"

#: The four presses, in the order that breaks plan ties -- fixed, so a plan
#: re-derived in another process is byte-identical. ACTION5 is not here: no rule
#: reads the action force, so ``require_player_movement`` reverts the turn.
_DIRS = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Heuristic charge for a board that can no longer be won -- a crate parked where
#: no target is reachable from. Large enough to sink the node, finite so the
#: search stays complete.
_DEAD = 1_000_000


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """The parts of a level no rule ever touches: size, walls and targets.

    Split out from the state because the search hashes the state on every node
    and the scenery is dead weight in that key -- which is also why
    `SnakeobanExpert` scopes its plan memo by level (see `PSExpert.scope_by_level`).
    """

    __slots__ = ("h", "w", "walls", "targets")

    def __init__(self, h: int, w: int, walls: frozenset, targets: frozenset):
        self.h, self.w, self.walls, self.targets = h, w, walls, targets

    @classmethod
    def from_engine(cls, eng, ids: dict) -> "_Board":
        wall, target = ids["wall"], ids["target"]
        cells = [(r, c) for r in range(eng.height) for c in range(eng.width)]
        return cls(eng.height, eng.width,
                   frozenset(p for p in cells if wall in eng.grid[p[0]][p[1]]),
                   frozenset(p for p in cells if target in eng.grid[p[0]][p[1]]))

    def inb(self, p) -> bool:
        return 0 <= p[0] < self.h and 0 <= p[1] < self.w


def _read_state(eng, ids: dict) -> tuple:
    """``(snake, crates, apples)`` from the engine grid, snake head-first.

    The chain is reconstructed from the MARKERS rather than assumed: each
    segment's cell carries the direction that segment leaves in, so the segment
    behind a cell is the one whose marker points AT it. That is the game's own
    bookkeeping, so a level whose author drew the snake in some order the
    rendering does not reveal still reads correctly -- and the three asserts turn
    any drift (a stranded marker, a forked chain, a marker on the head) into a
    failure here rather than into a plan for a board that does not exist.
    """
    head_id, body_id, tail_id = ids["player"], ids["playerbody"], ids["playertail"]
    crate_ids = {ids["crate1"], ids["crate2"]}
    apple_id = ids["apple"]
    markers = {ids["upmarker"]: "up", ids["downmarker"]: "down",
               ids["leftmarker"]: "left", ids["rightmarker"]: "right"}

    head = None
    parts: set = set()
    crates: set = set()
    apples: set = set()
    points_at: dict = {}                      # cell a segment steps into -> segment
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            p = (r, c)
            if head_id in cell:
                head = p
            if cell & {head_id, body_id, tail_id}:
                parts.add(p)
            if cell & crate_ids:
                crates.add(p)
            if apple_id in cell:
                apples.add(p)
            for m, d in markers.items():
                if m in cell:
                    dr, dc = _DELTA[d]
                    nxt = (r + dr, c + dc)
                    assert nxt not in points_at, f"two markers aim at {nxt}"
                    points_at[nxt] = p
    assert head is not None, "no player on the board"
    assert head not in points_at.values(), "the head must not carry a marker"

    snake = [head]
    while True:
        nxt = points_at.get(snake[-1])
        if nxt is None:
            break
        snake.append(nxt)
    assert len(snake) == len(parts), f"snake {snake} != segments {sorted(parts)}"
    return tuple(snake), frozenset(crates), frozenset(apples)


def _step(board: _Board, state: tuple, direction: str):
    """The state after pressing ``direction``, or None if the press is refused.

    None and "no change" are the same thing here (``require_player_movement``
    reverts the whole turn), which is why the search can drop a refused press
    outright instead of dedup'ing it a level later.
    """
    snake, crates, apples = state
    dr, dc = _DELTA[direction]
    head = snake[0]
    ahead = (head[0] + dr, head[1] + dc)
    if not board.inb(ahead) or ahead in board.walls or ahead in snake:
        return None
    if ahead in apples:
        # The eat rule spans three cells, so the apple needs a cell behind it;
        # without one the press is refused (Apple blocks the player).
        if not board.inb((ahead[0] + dr, ahead[1] + dc)):
            return None
        return ((ahead,) + snake, crates, apples - {ahead})
    if ahead in crates:
        run = []
        cur = ahead
        while cur in crates:
            run.append(cur)
            cur = (cur[0] + dr, cur[1] + dc)
        if (not board.inb(cur) or cur in board.walls
                or cur in snake or cur in apples):
            return None
        moved = frozenset((p[0] + dr, p[1] + dc) for p in run)
        return ((ahead,) + snake[:-1], (crates - frozenset(run)) | moved, apples)
    return ((ahead,) + snake[:-1], crates, apples)


def _win(board: _Board, state: tuple) -> bool:
    _snake, crates, apples = state
    return not apples and all(p in board.targets for p in crates)


# ---------------------------------------------------------------------------
# Heuristic + prunes
# ---------------------------------------------------------------------------

def _push_tables(board: _Board) -> dict:
    """``{target: {cell: pushes to get a crate from cell onto target}}``.

    Reverse BFS over push edges, ignoring apples, the other crates and the snake
    -- the standard sokoban relaxation, so the numbers are lower bounds. A cell
    absent from every table is one a crate can never leave; see `_DEAD`.
    """
    tables = {}
    for target in board.targets:
        steps = {target: 0}
        queue = deque([target])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _DELTA.values():
                crate = (r - dr, c - dc)          # where the crate comes from
                stand = (r - 2 * dr, c - 2 * dc)  # where the player has to be
                if not board.inb(crate) or crate in board.walls:
                    continue
                if not board.inb(stand) or stand in board.walls:
                    continue
                if crate not in steps:
                    steps[crate] = steps[(r, c)] + 1
                    queue.append(crate)
        tables[target] = steps
    return tables


def _walk_field(board: _Board, src) -> dict:
    """BFS distances from ``src`` over the non-wall cells, ignoring everything
    that moves. Used only by the approach term, which must stay a lower bound."""
    dist = {src: 0}
    queue = deque([src])
    while queue:
        p = queue.popleft()
        for dr, dc in _DELTA.values():
            n = (p[0] + dr, p[1] + dc)
            if board.inb(n) and n not in board.walls and n not in dist:
                dist[n] = dist[p] + 1
                queue.append(n)
    return dist


class _Heuristic:
    """``apples + pushes + approach`` -- admissible, so ``weight = 1`` A* plans
    are shortest. See the module docstring for the derivation of each term."""

    def __init__(self, board: _Board):
        self.board = board
        self.tables = _push_tables(board)
        self._fields: dict = {}

    def _field(self, src) -> dict:
        field = self._fields.get(src)
        if field is None:
            field = self._fields[src] = _walk_field(self.board, src)
        return field

    def pushes(self, crates) -> int:
        """A lower bound on the number of PUSH presses still owed, or `_DEAD`
        when no assignment of loose crates to bare targets is possible at all.

        It is the BOTTLENECK of the best assignment -- ``min`` over assignments
        of ``max`` over pairs -- and not their sum, because one press moves a
        whole contiguous RUN of crates: ``[ > Crate | Crate] -> [> Crate | >
        Crate]``. Summing would have charged level 3 thirteen presses for a
        board that wins in five (three crates shoved along one corridor
        together), and an over-estimate is not a slower search, it is a plan
        that is no longer certified shortest. Each crate does need its own
        ``d`` presses to move ``d`` cells -- a press advances any one crate by at
        most one -- so the max over the assignment is sound.

        The assignment is over EVERY crate and every target, not over the loose
        crates and the bare ones. A crate already standing on a target is not
        finished: in level 3's corridor all three crates shuffle two cells to
        the right together, each landing on the target its neighbour was
        occupying, and an estimate that froze the covered target charged 3 for a
        board two presses from done.

        Brute-forced over permutations rather than matched greedily: the game
        never has more than four crates, so it is at most 24 tuples, and greedy
        is what costs `PSSokobanExpert` its admissibility.
        """
        board = self.board
        crates = list(crates)
        if not crates:
            return 0
        best = _DEAD
        for pick in itertools.permutations(board.targets, len(crates)):
            worst = 0
            for crate, target in zip(crates, pick):
                d = self.tables[target].get(crate)
                if d is None:
                    worst = _DEAD
                    break
                worst = max(worst, d)
            best = min(best, worst)
            if best == 0:
                break
        return best

    def __call__(self, state: tuple) -> int:
        snake, crates, apples = state
        pushes = self.pushes(crates)
        if pushes >= _DEAD:
            return _DEAD
        if not apples and not pushes:
            return 0
        # The first press that is neither an eat nor a push-into-place is only
        # reached after a walk, and the whole of that walk is such a press. If
        # the first productive press is an eat, at least ``dist - 1`` plain
        # presses precede it; if it is a push, at least ``dist`` do (nothing has
        # been eaten yet, by assumption). The smaller of the two bounds both.
        field = self._field(snake[0])
        approach = 1 << 30
        for apple in apples:                      # reach it, then eat it
            approach = min(approach, field.get(apple, 1 << 30) - 1)
        if pushes:
            for crate in crates:                  # stand behind it, then push it
                for dr, dc in _DELTA.values():
                    stand = (crate[0] - dr, crate[1] - dc)
                    if self.board.inb(stand) and stand not in self.board.walls:
                        approach = min(approach, field.get(stand, 1 << 30))
        return len(apples) + pushes + max(0, approach)


# ---------------------------------------------------------------------------
# A*
# ---------------------------------------------------------------------------

def _astar(board: _Board, start: tuple, heur: _Heuristic, node_cap: int,
           weight: int = 1, bound: int | None = None):
    """``(presses, nodes)`` -- a shortest win from ``start`` at ``weight == 1``.

    ``bound`` caps the total plan length: nodes whose ``g + h`` exceeds it are
    dropped, so an exhausted queue PROVES no win exists within ``bound``. That is
    what makes the optimal-set measurement affordable -- and, because the
    heuristic is admissible, sound. ``presses`` is None when there is no plan
    (within ``bound``, when one is given) or when the node cap bit; the two are
    told apart by the caller through ``nodes``.

    Memory: the queue carries parent pointers, not paths. A 74-press plan on a
    3M-node search would otherwise hold 3M list copies.
    """
    if _win(board, start):
        return [], 0
    parent: dict = {start: None}
    best_g = {start: 0}
    counter = 0
    queue = [(weight * heur(start), 0, counter, start)]
    nodes = 0
    while queue:
        _f, g, _c, state = heapq.heappop(queue)
        if g > best_g.get(state, -1):
            continue                                  # a stale heap entry
        for direction in _DIRS:
            nxt = _step(board, state, direction)
            nodes += 1
            if nxt is None:
                continue
            ng = g + 1
            if _win(board, nxt):
                path = [direction]
                cur = state
                while parent[cur] is not None:
                    cur, took = parent[cur]
                    path.append(took)
                path.reverse()
                return path, nodes
            if best_g.get(nxt, 1 << 30) <= ng:
                continue
            h = heur(nxt)
            if h >= _DEAD:
                continue
            if bound is not None and ng + h > bound:
                continue
            best_g[nxt] = ng
            parent[nxt] = (state, direction)
            counter += 1
            heapq.heappush(queue, (ng + weight * h, ng, counter, nxt))
        if nodes >= node_cap:
            return None, nodes
    return None, nodes


def _optimal_sets(board: _Board, heur: _Heuristic, start: tuple, plan: list,
                  tie_cap: int) -> list:
    """Per-step optimal-press SETS for ``plan``, MEASURED by bounded re-solve.

    At step ``i`` there are ``remaining = len(plan) - i`` presses left, so a
    sibling press ties exactly when a win exists within ``remaining - 1`` presses
    of the state it lands in. `_astar` is asked that question directly, under an
    f-bound of ``remaining - 1``: an exhausted queue is a proof of "no tie", and
    the admissible-estimate test in front of it answers most siblings without a
    search at all.

    ``tie_cap`` bounds each of those re-solves. A capped one is recorded as NOT a
    tie, which makes the sets conservative rather than wrong -- see the module
    docstring. The taken press is in its own set by construction.
    """
    sets = []
    state = start
    for i, taken in enumerate(plan):
        remaining = len(plan) - i
        tied = {taken}
        for direction in _DIRS:
            if direction == taken:
                continue
            nxt = _step(board, state, direction)
            if nxt is None:
                continue
            if _win(board, nxt):
                if remaining == 1:
                    tied.add(direction)
                continue
            if remaining <= 1 or heur(nxt) > remaining - 1:
                continue
            sub, _nodes = _astar(board, nxt, heur, tie_cap,
                                 bound=remaining - 1)
            if sub is not None and len(sub) == remaining - 1:
                tied.add(direction)
        sets.append([d for d in _DIRS if d in tied])
        state = _step(board, state, taken)
    return sets


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

def _ids(game) -> dict:
    """``name -> obj_index`` for the objects the model reads."""
    return dict(game.obj_name_to_idx)


class SnakeobanExpert(PSExpert):
    """Plans on the native `_Board`/`_step` model; `PSExpert` keeps the memo, the
    on-disk plan cache with its staleness check, and the snapshot discipline.

    Only `_search` is overridden. `heuristic` is unreachable -- the A* that runs
    here is the native one, which carries its own `_Heuristic`.
    """

    directions = list(_DIRS)

    #: `_key` is inherited (every non-background cell, markers and connectors
    #: included), which is exact and canonical ACROSS levels -- the walls are in
    #: it -- so no ``scope_by_level`` is needed.

    #: Per-step budget for the optimal-set measurement, in native nodes. Sized so
    #: the measurement never costs appreciably more than the plan it labels.
    tie_cap: int = 120_000

    #: The searches are the whole cost of startup and they are seed-independent,
    #: so without a file every `parallelize_generator` shard would re-derive all
    #: of them. The entry is keyed on the level's START board and carries the
    #: optimal sets with it, so a shard replays labelled steps without
    #: re-deriving anything -- and an edited level is a miss, not a wrong plan.
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "snakeoban_plans.json")

    def setup(self) -> None:
        self._ids = _ids(self.g)
        self._boards: dict = {}

    def heuristic(self, eng) -> int:                          # pragma: no cover
        raise AssertionError("planning is native; this A* is not used")

    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid."""
        key = (eng.height, eng.width)
        board = _Board.from_engine(eng, self._ids)
        cached = self._boards.get(key)
        if cached is not None and (cached.walls, cached.targets) == (
                board.walls, board.targets):
            board = cached
        else:
            self._boards[key] = board
        return board, _read_state(eng, self._ids)

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        heur = _Heuristic(board)
        presses, _nodes = _astar(board, state, heur, self.node_cap, self.weight)
        if presses is None:
            return None
        return Plan(presses, _optimal_sets(board, heur, state, presses,
                                           self.tie_cap))


class SnakeobanSolver(PSAStarSolver):
    game_id = "puzzlescript_snakeoban"
    game_name = GAME_NAME
    expert_cls = SnakeobanExpert

    #: Record against the GAME FOLDER's adapter. ``games/ps:snakeoban`` is a
    #: plain passthrough today; naming it anyway means a sprite patch or a step
    #: cap added there later cannot silently make this generator tape a game
    #: nobody plays (the ps:count_mover trap).
    game_module_id = "ps:snakeoban"

    #: Level 9 -- see "The search" in the module docstring. Skipped up front so
    #: startup does not burn the node cap on it once per process.
    skip_levels = frozenset({9})

    #: Native nodes, not interpreter steps: level 8's shortest plan (74 presses)
    #: costs ~3.3M of them, and the cap is set clear of that so a level is never
    #: silently downgraded to "unsolvable" by a budget.
    node_cap = 8_000_000

    #: Room for the longest plan (74 presses) plus the <=15-press exploration
    #: prefix and its RESET, well inside the adapter's 200-step per-level budget.
    max_steps = 150

    def _reset_prefix(self, *, reset_to_level, record_obs, record_act, **kw):
        """`BaseSolver._reset_prefix`, plus a RESET when the prefix WON.

        The shared prefix deliberately skips its closing RESET on a terminal win
        (``if perturbed and last_terminal != "win"``) -- for a game whose levels
        take a dozen presses that branch is unreachable. Level 0 here is TWO
        presses (walk right onto the apple) and the opening exploration budget is
        at its largest on the episode's first level, so a random flail finishes
        it outright a good fraction of the time. `record_level`'s loop then sees
        an already-won level and breaks, and the level tapes a WIN made entirely
        of ``phase="explore"`` steps -- which `train_policy` masks out of the
        policy loss, so the recording trains on NOTHING. Measured at 2 of 27
        levels over 3 seeds before this override and 0 of 27 after.

        Resetting anyway costs one frame and puts the level back at its start,
        where the expert replays the real 2-press solution, so every level
        carries a labelled demonstration and every episode has the same
        flail / reset / solve arc. (`slippy_penguin` hit this first and for the
        same reason; the fix lives here rather than in `BaseSolver` because for
        most games the branch is genuinely unreachable and changing it would
        re-tape every corpus already on disk.)

        The single-frame assumption is safe: ``record_spans`` is False, so
        `record_level`'s ``reset_to_level`` hands back one ``(H, W)`` frame.
        """
        prev = super()._reset_prefix(reset_to_level=reset_to_level,
                                     record_obs=record_obs,
                                     record_act=record_act, **kw)
        if self._game is None or self._game._state != GameState.WIN:
            return prev
        frame = np.asarray(reset_to_level())
        record_obs(frame)
        record_act(self._encode_step(
            Action(RESET_ACTION), optimal=[Action(RESET_ACTION)],
            phase="reset", changed=not np.array_equal(frame, prev), n_obs=1))
        return frame


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = SnakeobanSolver()
    game = solver.make_game(seed)
    expert = SnakeobanExpert(game, node_cap=SnakeobanSolver.node_cap)
    return solver, game, expert


def _levels(game) -> list:
    return [lvl for lvl in range(game.n_levels)
            if lvl not in SnakeobanSolver.skip_levels]


def _report() -> int:
    """Per-level size, inventory, plan length and tie coverage."""
    _solver, game, expert = _new()
    eng = game._engine
    total = tied = 0
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        snake, crates, apples = state
        head = (f"level {level}: {eng.height:2d}x{eng.width:2d}, snake {len(snake)}, "
                f"{len(apples):2d} apple(s), {len(crates)} crate(s)/"
                f"{len(board.targets)} target(s)")
        if level in SnakeobanSolver.skip_levels:
            print(f"{head} -- SKIPPED (see the module docstring)")
            continue
        t0 = time.time()
        plan = expert.plan(eng, level)
        dt = time.time() - t0
        if plan is None:
            print(f"{head} -- NO PLAN within {expert.node_cap} nodes")
            bad += 1
            continue
        sets = getattr(plan, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        labelled = sum(1 for s in sets if s)
        total += len(plan)
        tied += ties
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(plan):3d} presses (budget {game._max_steps}, {room}), "
              f"{labelled}/{len(plan)} labelled, {ties} with a tie, {dt:6.2f}s")
    print(f"total {total} presses over {len(_levels(game))} levels, "
          f"{tied} step(s) with a tie ({tied / max(1, total):.0%})")
    return 0 if not bad else 1


def _selfcheck(trials: int) -> int:
    """Fuzz the native model against the interpreter.

    ``trials`` random walks per level, 40 presses each, comparing the WHOLE
    state after every press -- the snake chain read back off the markers, the
    crates, the apples and the win flag -- plus the refusal itself: the model
    says None exactly where ``require_player_movement`` reverts the turn, so a
    press the interpreter silently swallowed and one the model thought legal
    would both show up here as a state that stopped matching.
    """
    _solver, game, expert = _new()
    eng = game._engine
    rng = random.Random(20260818)
    bad = presses = 0
    for level in range(game.n_levels):
        for _ in range(trials):
            game.set_level(level)
            board, state = expert.read(eng)
            for _k in range(40):
                direction = rng.choice(_DIRS)
                predicted = _step(board, state, direction)
                eng.step(direction)
                presses += 1
                got = _read_state(eng, expert._ids)
                want = state if predicted is None else predicted
                if got != want:
                    print(f"  level {level}: press {direction} from {state}\n"
                          f"    model {want}\n    engine {got}")
                    bad += 1
                    break
                if eng.check_win() != _win(board, got):
                    print(f"  level {level}: win flag engine="
                          f"{eng.check_win()} model={_win(board, got)}")
                    bad += 1
                    break
                state = got
                if eng.check_win():
                    break
    print(f"selfcheck: {presses} presses across {game.n_levels} levels, "
          f"{bad} mismatch(es)")
    return 0 if not bad else 1


def _verify(cap: int | None = None) -> int:
    """Drive every plan through the real ADAPTER and assert it wins, then press
    every shipped optimal label on the interpreter.

    Three things are checked, and the model is the authority for none of them:

    * **the plan wins.** Replayed as SCREEN presses through
      `PuzzleScriptAdapter`, it has to reach ``GameState.WIN`` -- i.e. the
      rotation contract holds and the native dynamics agree with the interpreter
      over a whole solution, not only over the random walks `_selfcheck` fuzzes.
    * **every label is a press the game accepts.** Each optimal press is executed
      on the INTERPRETER at the state it labels, and the board has to change;
      ``require_player_movement`` reverts a refused press, so a label the engine
      would swallow shows up here as a board that did not move.
    * **every label is shortest.** From the state each label lands in, a fresh
      bounded A* has to find a win in exactly ``remaining - 1`` presses. That is
      re-derived per step from the level's own board rather than read off the
      plan, so a plan that drifted from optimal at step 40 fails at step 40.
      A re-solve that runs out of budget is reported, not counted as a pass.
    """
    from arcengine import ActionInput, GameState                # noqa: PLC0415

    from solvers.common.ps_astar import screen_action           # noqa: PLC0415

    _solver, game, expert = _new()
    eng = game._engine
    cap = cap or expert.tie_cap
    bad = 0
    for level in _levels(game):
        game.set_level(level)
        board, start = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level}: NO PLAN")
            bad += 1
            continue
        heur = _Heuristic(board)

        labels = capped = 0
        state = start
        for i, (taken, best) in enumerate(zip(plan, plan.optsets)):
            remaining = len(plan) - i
            if taken not in best:
                print(f"  level {level} step {i}: took {taken}, not in {best}")
                bad += 1
            here = snapshot(eng)
            for direction in best:
                labels += 1
                restore(eng, here)
                eng.step(direction)                  # the INTERPRETER, not the model
                if eng.grid == here:
                    print(f"  level {level} step {i}: label {direction} is a "
                          f"press the game refuses")
                    bad += 1
                    continue
                if eng.check_win():
                    if remaining != 1:
                        print(f"  level {level} step {i}: label {direction} "
                              f"wins with {remaining - 1} presses still claimed")
                        bad += 1
                    continue
                nxt = _step(board, state, direction)
                sub, nodes = _astar(board, nxt, heur, cap, bound=remaining - 1)
                if sub is None and nodes >= cap:
                    capped += 1
                    continue
                if sub is None or len(sub) != remaining - 1:
                    print(f"  level {level} step {i}: label {direction} needs "
                          f"{None if sub is None else len(sub)} presses, not "
                          f"{remaining - 1}")
                    bad += 1
            restore(eng, here)
            eng.step(taken)
            state = _step(board, state, taken)

        # The adapter replay, at this seed's own presentation.
        game.set_level(level)
        rot, hflip, vflip = game._rotation_k, game._hflip, game._vflip
        for direction in plan:
            game.perform_action(ActionInput(
                id=screen_action(direction, rot, hflip, vflip)))
        ok = game._state == GameState.WIN
        bad += not ok
        note = "" if not capped else f", {capped} re-solve(s) out of budget"
        print(f"level {level}: {len(plan):3d} presses, {labels} label(s) pressed "
              f"on the interpreter{note} -- adapter replay "
              f"{'WIN' if ok else 'FAILED: ' + str(game._state)}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} problem(s)")
    return 0 if not bad else 1


def _symmetry() -> int:
    """Drive the same ENGINE directions at every presentation the adapter can
    draw, and require the board to come out identical every time.

    Two things are being measured, and neither is asserted:

    * **the rotation contract.** `screen_action` has to invert the adapter's own
      forward remap, and getting it backwards does not raise -- it just drives a
      different direction on three of every four orientations. Here every plan is
      replayed at each ``(rotation, hflip, vflip)`` and has to WIN.
    * **the flip augmentation itself.** Snakeoban is in
      `PuzzleScriptAdapter._FLIP_GAMES`, which is a claim that the game is
      mirror-symmetric. The adapter only mirrors the PRESENTATION, so if the
      claim holds the ENGINE grid after a given press must be the same at every
      presentation. That is compared cell by cell after every press, over each
      plan and then over a fixed random walk that leaves the plan's states.
    """
    from arcengine import ActionInput, GameState                # noqa: PLC0415

    from solvers.common.ps_astar import screen_action           # noqa: PLC0415

    _solver, game, expert = _new()
    eng = game._engine
    plans = {}
    for level in _levels(game):
        game.set_level(level)
        plans[level] = expert.plan(eng, level)

    # A fixed press script per level: the plan, then a random walk that carries
    # the board off it (the plan alone only ever visits states the search liked).
    rng = random.Random(20260818)
    scripts = {lvl: list(plan) + [rng.choice(_DIRS) for _ in range(60)]
               for lvl, plan in plans.items()}

    seen: dict = {}
    seed = 0
    while len(seen) < 16 and seed < 2000:
        game._seed = seed
        game.set_level(_levels(game)[0])
        seen.setdefault((game._rotation_k, game._hflip, game._vflip), seed)
        seed += 1

    baseline: dict = {}
    bad = 0
    for key in sorted(seen):
        game._seed = seen[key]
        wins = drift = 0
        for level, script in scripts.items():
            game.set_level(level)
            rot, hflip, vflip = game._rotation_k, game._hflip, game._vflip
            boards = []
            won = False
            for i, direction in enumerate(script):
                game.perform_action(ActionInput(
                    id=screen_action(direction, rot, hflip, vflip)))
                boards.append(snapshot(eng))
                if not won and i == len(plans[level]) - 1:
                    won = game._state == GameState.WIN
            wins += won
            ref = baseline.setdefault(level, boards)
            drift += sum(1 for a, b in zip(ref, boards) if a != b)
        bad += (wins != len(scripts)) + (drift > 0)
        print(f"rot={key[0]} hflip={int(key[1])} vflip={int(key[2])}: "
              f"{wins}/{len(scripts)} levels win, {drift} board(s) differ from "
              f"the unaugmented run")
    print(f"symmetry clean over {len(seen)} presentation(s)" if not bad
          else f"SYMMETRY FAILED: {bad}")
    return 0 if not bad else 1


def _reachable_compositions(expert, game, parsed, eng) -> set:
    """Every cell composition a settled frame of this game can show.

    Derived from the model rather than hand-listed: walk each level's plan and
    every optimal alternative off it, and collect the cell contents the
    interpreter leaves behind. That covers the compositions the agent actually
    sees -- a segment on a target, a crate on a target, every connector the late
    rules paint -- without pretending to enumerate a space this game does not
    have (level 6 alone has more states than any audit could visit).
    """
    inv = {v: k for k, v in parsed.obj_name_to_idx.items()}
    out: set = set()

    def harvest():
        for row in eng.grid:
            for cell in row:
                out.add(tuple(sorted(inv[o] for o in cell)))

    for level in _levels(game):
        game.set_level(level)
        plan = expert.plan(eng, level)
        harvest()
        for taken, best in zip(plan, plan.optsets):
            here = snapshot(eng)
            for direction in best:
                restore(eng, here)
                eng.step(direction)
                harvest()
            restore(eng, here)
            eng.step(taken)
            harvest()
    return out


def _audit() -> int:
    """Assert every reachable cell COMPOSITION renders distinctly, at every cell
    size the boards use.

    Whole 64x64 frames of uniform boards are compared rather than one cell out of
    a mixed board: `_render_frame` centre-pads a non-square board, so indexing a
    cell by ``cell_px * r`` silently reads the wrong pixels on the wide levels
    (the ps:explod lesson). Two uniform boards render identically iff their cells
    do.

    MARKERS ARE MEANT TO BE INVISIBLE and are handled separately. ``Marker`` is
    the FIRST collision layer, i.e. composited under the opaque Background, so
    the trail the game keeps under the snake is bookkeeping the agent never sees
    -- and it must stay that way, or the frame would leak the snake's future.
    That invisibility is PROVED here (each marker added to a board, the frame
    required to be unchanged) and the markers are then stripped, so the rest of
    the report is about what the agent can actually look at instead of about
    2^4 copies of every composition.

    The pairs that matter, and why:

    * ``player`` vs ``playerbody``/``playertail`` -- the sprites differ in ONE
      pixel (the head's dark-blue eye), so a cell size that point-samples past it
      leaves the agent unable to tell which end of the snake moves.
    * ``crate1`` vs ``crate2`` -- a crate off a target vs on one, which is half
      the win condition made visible.
    * ``ghosttarget`` over a segment vs the bare segment -- the game paints
      GhostTarget on any target a snake part is standing on precisely so the
      target does not vanish under the body.
    """
    import numpy as np                                          # noqa: PLC0415

    from adapters.puzzlescript_adapter import _render_frame     # noqa: PLC0415

    _solver, game, expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    markers = ("upmarker", "downmarker", "leftmarker", "rightmarker")

    comps = _reachable_compositions(expert, game, g, eng)
    sizes: dict = {}
    for level in _levels(game):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, comp):
        eng.height, eng.width = h, w
        eng.grid = [[{idx[o] for o in comp} | {idx["background1"]}
                     for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g)).copy()

    bad = 0
    visible = sorted({tuple(o for o in comp if o not in markers)
                      for comp in comps})
    for (h, w), levels in sorted(sizes.items()):
        # The markers-are-invisible proof, on this cell size.
        leaks = [(comp, m) for comp in visible for m in markers
                 if not np.array_equal(shoot(h, w, comp),
                                       shoot(h, w, comp + (m,)))]
        bad += len(leaks)
        shots = {comp: shoot(h, w, comp) for comp in visible}
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(shots)} visible "
              f"composition(s), {len(leaks)} marker leak(s)")
        for a, b in clashes:
            print(f"    IDENTICAL: {'+'.join(a)}  ==  {'+'.join(b)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} problem(s)")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--selfcheck" in sys.argv:
        i = sys.argv.index("--selfcheck")
        n = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 40
        sys.exit(_selfcheck(n))
    if "--verify" in sys.argv:
        i = sys.argv.index("--verify")
        n = (int(sys.argv[i + 1])
             if len(sys.argv) > i + 1 and sys.argv[i + 1].isdigit() else None)
        sys.exit(_verify(n))
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(SnakeobanSolver.main())
