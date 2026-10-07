"""Generate Phase-1 training data for the PuzzleScript game ps:veggie_jam
("Veggie Jam" by Big Tiger).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the `BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
measured mechanic, a native model of it, the EXACT distance field over the whole
reachable space, the optimal-action labeller and the render/symmetry audits.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_veggie_jam",
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
exactly. Every expert step carries the full set of equally-optimal presses.

The game
--------
Three levels, three vegetables, one joke per level ("Costco now has cauliflower
pizza"). Mechanically it is the canonical PuzzleScript sokoban written out three
times, once per veggie::

    [ > Player | Cauliflower ] -> [ > Player | > Cauliflower ]
    [ > Player | Carrot ]      -> [ > Player | > Carrot ]
    [ > Player | Peas ]        -> [ > Player | > Peas ]

with the collision layers ``Background`` / ``Target`` / ``Player Wall
Cauliflower Carrot Peas``, so:

  * **Walls and veggies block.** They share the player's layer. Walking into a
    wall or off the board does nothing; walking into a veggie shoves it one
    cell, and if THAT cell is a wall, the board edge or a second veggie the
    whole press is cancelled -- the player does not step either, because the
    rule's right-hand side moves both bodies or neither.
  * **There is no chain push.** Every rule names ``Player`` on its left, so a
    moving veggie never re-triggers one: two veggies in a row are a wall.
  * **Targets block nothing.** ``Target`` is alone on its own layer, under the
    player's, so the player and a sliding veggie both walk straight over one.
    The player standing on the goal is an ordinary state, and the plans below do
    it.
  * **Four arrow keys, no ACTION.** ``action`` parses (the game does not declare
    ``noaction``) but no rule reads it and there is no ``late`` rule to tick, so
    it cannot change the board. ``--selfcheck`` puts it in the fuzz alphabet and
    asserts that against the interpreter rather than reading it off the rules
    section -- ps:ouroboros is the game where exactly that assumption was false.
  * **No ``restart``, no ``again``, no ``random``.** The rules section is the
    three lines above, so a press is one deterministic board rewrite.

The win direction is the one thing here that is NOT the canonical sokoban's, and
it is worth stating because it flips the predicate. ps:piedra and
ps:sokoban_sanity win on ``All Target on Crate`` -- every target square covered.
This game wins on three clauses in the OTHER direction::

    All Cauliflower on Target
    All Carrot on Target
    All Peas on Target

i.e. every VEGGIE must be standing on a target, and a level that ships none of a
kind satisfies that kind's clause vacuously (which is how levels 0-2, holding
one veggie each, win at all). `_Board.won` is written as the literal reading --
"every veggie cell is a target cell" -- so it stays right on an edited level with
a target to spare, where the covered-targets reading would not be.

The three shipped levels are one veggie and one target each:

  * **L0** (7x9): an open walled room, a Cauliflower two cells below-right of
    the Target. 8 presses.
  * **L1** (7x9): a room split by a stub wall, a Carrot to be walked around and
    driven down-then-left into the Target. 12 presses.
  * **L2** (7x17): the long one. A comb of pillars along the top, and the Target
    is cut into the RIGHT wall at row 3 -- so the Peas have to be walked the
    full width of the board and then shoved right into it from (3,15), which is
    reachable only from above or below because (3,13) is a pillar. 38 presses.

Veggie TYPE is deliberately not part of the state. All three rules are the same
rule and all three win clauses are the same clause, so the veggies are
interchangeable to a planner; each shipped level has exactly one anyway. What
the frames show is of course not interchangeable, and `--audit` checks that.

Expert solver
-------------
A native model of the mechanic above (`_Board`), enumerated to an EXACT distance
field rather than searched.

The state is ``(player cell, sorted veggie cells)`` -- the only things any rule
moves; walls and targets never change and live on the board rather than in the
state. With one veggie per level that is at most (free cells)^2, which comes out
at 1158, 861 and 3346 reachable states (the winning boards included, as terminal
nodes): the whole game is 5365 states and enumerates in milliseconds. So there is
no search here. `_Board.enumerate` walks the space once
from the level start, keeping the successor table, then runs one backward BFS
from the winning states over the predecessor map it inverts to. A plan is a
descent of the field, a tie set is a lookup, and both answer from ANY reachable
state.

That last property is the one that earns everything below. The enumerated set is
CLOSED under every press (each state contributes all four successors), so a
state reached by the exploration prefix, by an epsilon detour or by a RESET is
already in the table -- which makes "is this board still winnable" a dict lookup
in a game where it is a real question. HALF the space is dead (2670 of 5365, and
62% of level 1): a veggie shoved flat against a wall it cannot come off is
permanent, and into a corner it cannot move at all.

Optimal-action targets and recovery
-----------------------------------
`optimal_for` reads the LIVE engine state and returns every press that lowers the
exact distance -- true tie sets, not a reordering heuristic. Where it bites is
the walk between pushes, which is order-free (going three right then two up puts
the player on the same cell in the same number of presses as two up then three
right, and no rule fires on a bare move), so labelling one arbitrary interleaving
as the single right answer would train a coin flip the policy cannot win. The
measured answer is that this game has few ties -- 5 of the 58 expert steps have a
second equally-right press, 1.25 optimal presses/step on L0, 1.00 on L1 and 1.11
on L2 -- because the veggie sits in the walk's way often enough that most
interleavings either shove it somewhere useless or cost a press. ``--plans``
prints it.

``epsilon = 0.12`` is safe here DESPITE the irreversibility, and it is the field
that makes it so. `record_level`'s detour probe steps a random alternative, asks
the expert whether it can still plan a win, and keeps the detour only if it can;
for a bounded-search expert that test is a guess that can decline a live state,
but here it is a lookup in a field covering the whole reachable space, so a
detour provably cannot strand the veggie. The step then records the mistake as
the action taken and the exact recovery as ``optimal``, on top of the
explore-then-RESET prefix that opens every episode.

Shortest, and how that is known
-------------------------------
Three independent derivations agree on all three plan lengths (8, 12, 38) AND on
every one of the 58 tie sets:

  * this file's field (forward BFS over the model + backward BFS over the
    inverted successor table) -- ``--plans``;
  * a Bellman fixpoint re-derived from the successor table alone, sharing no code
    with the backward sweep, plus every shipped label pressed on the real
    interpreter -- ``--verify``;
  * a full enumeration of the INTERPRETER itself -- `StateGraph.build` walking
    every reachable engine state by pressing real buttons, with the native model
    nowhere in the loop -- ``--engine``. The game is small enough (5361 states,
    which is the 5365 above minus the four terminal winning boards the
    interpreter keeps as edge targets rather than nodes; 21444 engine presses;
    ~4 s) that this is exhaustive rather than a spot check, which the bigger ps:
    sokobans cannot afford.

`--selfcheck` is what makes the first two statements about this GAME rather than
about a model of it: a seeded random walk on every level, comparing the model's
board against the engine's grid after EVERY press -- including the presses that
do nothing, which is where a collision-layer mistake hides.

Rendering
---------
Nothing had to change in ``data/puzzlescript_games/Veggie_Jam.txt``, which is
rarer in this corpus than it sounds and was checked rather than assumed. The
composition the win condition is made of -- a veggie standing ON its target --
survives at both cell sizes because all three veggie sprites are drawn with
transparent margins and the Target is a filled red frame: 1204 px of Target show
past a Cauliflower at 7 px and 576 px at 3 px.

The ARC quantization does collapse things, and ``--audit`` is what says the
collapses do not matter. ``DarkBlue`` background and the player's ``blue``
trousers are both index 9, so the player's legs are invisible; ``lightgreen`` and
``green`` are both 14, so the Peas render as one flat colour; ``brown`` and
``orange`` are both 12. What survives is enough: floor 9, wall gray 2, target
red 8 + white 0, cauliflower 0 + 1, carrot 12 + 14, peas 14, player 15 + 7 + 12.
``--audit`` renders every cell composition the game can show as a whole 64x64
frame of a uniform board, at both cell sizes the levels use (7 px for the two
7x9 boards, 3 px for the 7x17 one), and requires them pairwise distinct --
whole frames rather than cell crops, because `_render_frame` upscales and
centre-pads so slicing by ``cell_px`` arithmetic reads the wrong window (the
ps:explod lesson). ``player + veggie`` is absent on purpose: they share a
collision layer, so the engine can never put them in one cell.

Augmentation
------------
The engine state after reset is identical for every seed (the levels are fixed
ASCII maps), so the only per-(seed, level) variables are presentational: the
frame rotation (0..3) and, since ``Veggie_Jam`` is in
`PuzzleScriptAdapter._FLIP_GAMES`, an independent horizontal and vertical flip
with the matching directional remap. 3 levels x 16 presentations = 48, and with
only three levels the flips are most of the corpus's variety rather than a
nicety.

The structural argument for the flips is the unqualified one: three rules, all
written with the relative ``>`` force, none naming an axis; no gravity;
screen-relative input; a win condition that is positional; and one press moves at
most two bodies, landing on ``p + d`` and ``p + 2d``, which can never be the same
square -- so no two moves can contest a cell and the order the interpreter
expands a rule's four directions cannot decide anything. The ART is not invariant
(the Carrot's stalk points up-left, the Peas sit on a diagonal, the Target's
bottom row is inset), so a mirrored board does show art the .txt does not
literally contain; that only matters if the mirrored art of one composition is
another composition's art, and ``--audit``'s second pass turns and mirrors every
composition eight ways and compares it against every other. ``--symmetry``
measures the mechanic the same way: it rebuilds each level's LAYOUT under all 8
transforms, reloads it into the interpreter and replays the level's own plan plus
a 400-press random walk with the presses transformed, requiring the player AND
the veggie to land where the transform says after every one.

Usage (run from the repo root):
    python solvers/generate_veggie_jam_training.py --episodes 200 \
        --out data/training_multi_level/veggie_jam

    python solvers/generate_veggie_jam_training.py --plans      # level report
    python solvers/generate_veggie_jam_training.py --selfcheck  # model vs engine
    python solvers/generate_veggie_jam_training.py --verify     # field + labels
    python solvers/generate_veggie_jam_training.py --engine     # interpreter proof
    python solvers/generate_veggie_jam_training.py --audit      # rendering
    python solvers/generate_veggie_jam_training.py --symmetry   # augmentation
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (                     # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from solvers.common.ps_astar import (                           # noqa: E402
    PSAStarSolver, PSExpert, Plan, StateGraph, restore, snapshot)

GAME_NAME = "Veggie_Jam"

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The presses the search branches on, in the order ties are broken. ``action``
#: is bound to nothing (no rule has it on a left-hand side), so branching on it
#: would double the work for nothing -- but the exploration prefix still presses
#: it, which is why `selfcheck` and `_symmetry` do too.
_DIRS: tuple[str, ...] = ("up", "down", "left", "right")

#: The three veggies, which are one object as far as any rule is concerned.
_VEGGIES = ("cauliflower", "carrot", "peas")


# ---------------------------------------------------------------------------
# The level, natively: one turn, the whole state space, the distance field
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry plus its COMPLETE state space and the exact
    distance-to-win field over it.

    Cells are flat ``r * w + c`` ids and a state is
    ``(player cell, sorted tuple of veggie cells)``. Walls and Targets are static
    -- no rule in this game creates, destroys or moves either -- so they live
    here, and the pair IS the state.

    `enumerate` is what everything else reads: one forward BFS from the level
    start collecting every reachable state and its four successors, then one
    backward BFS from the winning states over the predecessor map it inverts to.
    `plan` is a descent of the field, `optimal` a lookup, and both work from ANY
    reachable state -- which is what lets the recording take real detours and
    label the recovery, and what makes the detour probe's "is this still
    winnable" test exact in a game where a wrong shove is permanent.
    """

    __slots__ = ("h", "w", "walls", "targets", "start", "_next",
                 "states", "index", "succ", "dist")

    def __init__(self, h: int, w: int, walls, targets, start):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.targets = frozenset(targets)
        self.start = start

        # ``_next[d][cell]`` is the cell one step along d, or -1 when that is off
        # the board or a Wall. Both are the same thing to this game: the mover
        # stops, and if the mover is a shoved veggie the player's own step is
        # cancelled with it. Precomputed because it is the inner loop of the
        # enumeration and of the fuzz.
        self._next = {}
        for d, (dr, dc) in _DELTA.items():
            table = [-1] * (h * w)
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w
                            and nr * w + nc not in self.walls):
                        table[r * w + c] = nr * w + nc
            self._next[d] = table

        self.states: list | None = None
        self.index: dict = {}
        self.succ: list = []
        self.dist: list = []

    # -- one turn -------------------------------------------------------------
    def won(self, state) -> bool:
        """``All <veggie> on Target``, three times over: every veggie standing on
        a target square. Note the direction -- this is not the canonical
        sokoban's "every target covered", and on a board with a spare target the
        two differ."""
        return all(v in self.targets for v in state[1])

    def step(self, state, d: str):
        """One press, natively.

        Walk unless a Wall or the board edge is there; if a veggie is in the way,
        shove it, and if IT cannot move -- wall, edge, or a second veggie -- the
        whole press is cancelled and the player does not step either, because the
        rule's right-hand side binds both movements into one match."""
        p, vs = state
        n = self._next[d][p]
        if n < 0:
            return state
        if n in vs:
            m = self._next[d][n]
            if m < 0 or m in vs:
                return state
            moved = list(vs)
            moved[moved.index(n)] = m
            return (n, tuple(sorted(moved)))
        return (n, vs)

    # -- the whole space ------------------------------------------------------
    def enumerate(self) -> tuple[int, int]:
        """Build the reachable state set, its successor table and the exact
        distance-to-win field. Returns ``(states, dead states)``.

        The set is CLOSED under every press (each state contributes all four of
        its successors), so any state the agent can reach by any sequence of
        presses -- plan, exploration prefix, epsilon detour, or a RESET back to
        the start -- is already in it. Winning states are terminal: the adapter
        ends the level there, so they are not expanded.

        The dead count is LARGE and that is the mechanic, not a bug: a veggie
        shoved flat against a wall can only ever slide along it, and into a
        corner it cannot move at all."""
        if self.states is not None:
            return len(self.states), sum(1 for x in self.dist if x < 0)

        states = [self.start]
        index = {self.start: 0}
        succ: list = []
        i = 0
        while i < len(states):
            state = states[i]
            if self.won(state):
                succ.append(None)                    # terminal: the level ends
            else:
                row = []
                for d in _DIRS:
                    nxt = self.step(state, d)
                    j = index.get(nxt)
                    if j is None:
                        j = len(states)
                        index[nxt] = j
                        states.append(nxt)
                    row.append(j)
                succ.append(tuple(row))
            i += 1

        pred: list[list[int]] = [[] for _ in states]
        for src, row in enumerate(succ):
            if row is not None:
                for dst in row:
                    pred[dst].append(src)

        dist = [-1] * len(states)
        queue = deque()
        for j, state in enumerate(states):
            if self.won(state):
                dist[j] = 0
                queue.append(j)
        while queue:
            j = queue.popleft()
            for src in pred[j]:
                if dist[src] < 0:
                    dist[src] = dist[j] + 1
                    queue.append(src)

        self.states, self.index, self.succ, self.dist = states, index, succ, dist
        return len(states), sum(1 for x in dist if x < 0)

    # -- reading the field ----------------------------------------------------
    def distance(self, state) -> int | None:
        """Presses to a win from ``state``, or None when no veggie arrangement
        reachable from here is a win (or the state is not reachable from the
        level start at all, which cannot happen -- see `enumerate`)."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] < 0:
            return None
        return self.dist[j]

    def optimal(self, state) -> list:
        """Every press that keeps the game on a SHORTEST route to the win.

        Exact, straight off the field. A press that is refused (a wall, the board
        edge, an unshovable veggie) maps the state to itself, whose distance is
        unchanged rather than one less, so no-ops are excluded for free -- and so
        is every shove that kills the level, since a dead state has no distance
        at all."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] <= 0:
            return []
        here = self.dist[j]
        return [_DIRS[k] for k, nxt in enumerate(self.succ[j])
                if self.dist[nxt] == here - 1]

    def plan(self, state) -> "Plan | None":
        """A SHORTEST press sequence from ``state`` to a win, carrying the EXACT
        optimal set at every step, or None if this board can no longer be won.

        A descent of the field, taking the first tied direction in ``_DIRS``
        order, so a re-derived plan is byte-identical across processes."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] < 0:
            return None
        presses, optsets = [], []
        cur = state
        while self.dist[j] > 0:
            best = self.optimal(cur)
            presses.append(best[0])
            optsets.append(best)
            cur = self.step(cur, best[0])
            j = self.index[cur]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class VeggieJamExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Board` field.

    The base class keeps the memo, the restore discipline and the level scoping;
    only the strategy underneath changes, the way `PSEnumExpert` replaces it.
    Here the strategy is "look the answer up in a field covering the whole
    reachable space", so `heuristic` is never called and asserts rather than
    returning a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary board -- a recovery prefix, an epsilon detour -- is answered
    exactly, including "this board is now dead", which is what `record_level`
    needs to hear to fall back to a RESET.
    """

    directions = list(_DIRS)

    #: `_key` is the DYNAMIC objects only (player + veggies), which is canonical
    #: within a level but not across them: the walls and the target that complete
    #: the state are static per level yet differ between levels.
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.target_ids = set(g.resolve_object_name("target"))
        self.veggie_ids = set()
        for name in _VEGGIES:
            self.veggie_ids |= set(g.resolve_object_name(name))
        self.player_ids = set(self.game._engine._player_indices)
        self.dyn_ids = self.player_ids | self.veggie_ids
        self._boards: dict[int | None, _Board] = {}
        self._cur: _Board | None = None

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "VeggieJamExpert reads an exact distance field; heuristic is unused")

    # -- reading the engine ---------------------------------------------------
    def read(self, eng):
        """The ``(player cell, sorted veggie cells)`` state of the engine grid."""
        w = eng.width
        player = None
        veggies = []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & self.player_ids:
                    player = r * w + c
                if cell & self.veggie_ids:
                    veggies.append(r * w + c)
        if player is None:                           # pragma: no cover
            raise AssertionError("the board has no Player")
        return (player, tuple(sorted(veggies)))

    def board(self, eng, level: int | None) -> _Board:
        """The level's `_Board`, built (and enumerated) once and cached.

        The Walls and the Targets are static -- no rule touches either -- so
        whichever state happens to build it describes the same level. The START
        recorded in it is the state the engine holds at that moment, which
        `prepare_expert` makes the level's true start; the field is over the
        component reachable from there either way."""
        board = self._boards.get(level)
        if board is not None:
            return board
        h, w = eng.height, eng.width
        walls, targets = set(), set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & self.wall_ids:
                    walls.add(r * w + c)
                if cell & self.target_ids:
                    targets.add(r * w + c)
        board = _Board(h, w, walls, targets, self.read(eng))
        board.enumerate()
        self._boards[level] = board
        return board

    # -- planning -------------------------------------------------------------
    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn))

    def plan(self, eng, level: int | None = None):
        self._cur = self.board(eng, level)
        return super().plan(eng, level)

    def _search(self, eng) -> "Plan | None":
        return self._cur.plan(self.read(eng))

    def optimal_dirs(self, eng, level: int | None) -> list:
        """Every press on a shortest route, from the engine's CURRENT state."""
        return self.board(eng, level).optimal(self.read(eng))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class VeggieJamSolver(PSAStarSolver):
    game_id = "puzzlescript_veggie_jam"
    game_name = GAME_NAME
    expert_cls = VeggieJamExpert

    #: `games/ps:veggie_jam/ps:veggie_jam.py` is a plain passthrough (it
    #: constructs the adapter and nothing else), so routing through it would gain
    #: nothing -- but it IS what a live agent is handed, so if that wrapper ever
    #: grows a patch this must be set to ``"ps:veggie_jam"``.
    game_module_id = ""

    #: Unused: the expert reads an exact field rather than searching under a
    #: heuristic, so there is no weight to trade and no node budget that could
    #: quietly bite. Left at the family defaults so nothing reads a lie off them.
    node_cap = 400_000
    weight = 1

    #: The longest plan is 38 presses (level 2); the rest is room for a re-plan
    #: after the exploration prefix and for the detours. Well under the adapter's
    #: own 200-press per-level budget, which the ``set_level`` that ends the
    #: prefix resets anyway.
    max_steps = 120

    #: Non-zero even though this game is IRREVERSIBLE, which is a departure from
    #: most of this family and is earned by the exact field. `record_level`'s
    #: detour probe steps a random alternative, asks the expert whether it can
    #: still plan a win, and keeps the detour only if it can; for an expert that
    #: is a bounded search that test is a guess, but here it is a dict lookup in
    #: a field covering the whole reachable space, so a detour provably cannot
    #: strand the veggie. Roughly one press in eight is then a random legal
    #: alternative, after which the expert re-plans from wherever it landed: the
    #: taken action is the mistake and ``optimal`` is the exact recovery, which
    #: is the signal a policy needs after its own error. The RESET prefix still
    #: runs in front of it.
    epsilon = 0.12

    def prepare_expert(self, game, expert) -> None:
        """Enumerate every level's state space up front, from its true START.

        Pure front-loading -- the reachable component does not depend on which of
        its states builds it -- but it pays the per-level BFS once, at startup,
        rather than inside the first query that needs it, and it pins each
        `_Board.start` to the level's reset state so `--plans` and `--verify`
        report the shipped answer."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.board(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route to the win, read
        off the LIVE engine state.

        `record_level` asks for this BEFORE executing the press, so the engine is
        still at the state being labelled -- which is what makes the label right
        after an epsilon detour too, where the state is off the plan entirely and
        replaying the plan from the level start would label the wrong states.

        Falls back to the press about to be taken if the field has nothing to say
        -- no expert step may ship unlabelled."""
        best = expert.optimal_dirs(expert.game._engine, level)
        if best:
            return best
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _new():
    """``(solver, game, expert, solvable)`` with every level enumerated."""
    solver = VeggieJamSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _walk(board: _Board, plan) -> list:
    """The states ``plan`` passes through, starting at the level start."""
    out, state = [], board.start
    for d in plan:
        out.append(state)
        state = board.step(state, d)
    return out


def _report() -> int:
    """Print each level's space, its shortest plan and the engine's verdict --
    the quick "is this game still fully solved" check.

    The dead column is expected to be large: this is a sokoban, so a veggie
    shoved against the wrong wall ends the level's chances. What must hold is
    that every level's START is live, which is the ``win=True`` column."""
    _solver, game, expert, solvable = _new()
    eng = game._engine
    bad = dead_total = states_total = 0
    presses_total = ties_total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        n_states, n_dead = board.enumerate()
        states_total += n_states
        dead_total += n_dead
        plan = board.plan(board.start)
        if plan is None:
            bad += 1
            print(f"  L{level}: UNSOLVED")
            continue
        for d in plan:
            eng.step(d)
        won = eng.check_win()                   # read BEFORE anything reloads
        bad += not won
        ties = sum(len(s) for s in plan.optsets)
        presses_total += len(plan)
        ties_total += sum(1 for s in plan.optsets if len(s) > 1)
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"  L{level}: {eng.height:2d}x{eng.width:2d}  "
              f"{len(board.start[1])} veggie / {len(board.targets)} target  "
              f"{len(plan):3d} presses  win={won}  (budget {game._max_steps}, "
              f"{room})  {n_states:4d} states ({n_dead:4d} dead, "
              f"{n_dead / n_states:.0%})  {ties / len(plan):.2f} optimal "
              f"presses/step")
    print(f"  solvable levels: {solvable}")
    print(f"  total: {states_total} states, {dead_total} dead "
          f"({dead_total / states_total:.0%} -- a veggie against the wrong wall "
          f"never comes back)")
    print(f"  {presses_total} presses, {ties_total} of them with a second "
          f"equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def selfcheck(trials: int = 60, steps: int = 80, verbose: bool = True) -> int:
    """Drive random presses through BOTH the interpreter and `_Board.step`,
    comparing the Player AND veggie cells AND the win flag after every one. This
    is the guard that lets the planner trust the native model.

    ``action`` is included in the press alphabet even though the search excludes
    it: the claim that it is a pure no-op is exactly the kind of thing that
    should be measured against the interpreter rather than read off a rules
    section, however short (ps:ouroboros's ACTION is a shrink button and its
    rules section does not mention ``action`` either).

    The walks run from a COLD level start, with no warm-up press, because
    turn-one bookkeeping is precisely the class of mechanic a warm-up hides. They
    also re-assert the closure `_Board.enumerate` relies on: a state a random
    walk from the start can reach is already in the enumerated space, so the
    field can answer for anything the exploration prefix or an epsilon detour
    lands on.

    Returns the number of mismatches."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = VeggieJamExpert(game)
    alphabet = _DIRS + ("action",)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine, level)
        rng = random.Random(f"veggie_jam:selfcheck:{level}")
        bad = presses = 0
        for _ in range(trials):
            game.set_level(level)
            eng = game._engine
            state = expert.read(eng)
            for _ in range(steps):
                d = rng.choice(alphabet)
                want = state if d == "action" else board.step(state, d)
                eng.step(d)
                presses += 1
                got = expert.read(eng)
                if want != got or board.won(want) != eng.check_win():
                    bad += 1
                    break
                state = want
                if state not in board.index:
                    bad += 1
                    break
                if eng.check_win():
                    break
        total += bad
        if verbose:
            print(f"  L{level}: {'OK' if not bad else f'{bad} MISMATCHES'} "
                  f"({presses} presses vs the interpreter)")
    return total


def _verify() -> int:
    """Double-entry check of the distance field and of every tie set it ships.

    `_Board.enumerate` answers everything from ONE pass -- a forward sweep that
    builds the successor table and a reverse sweep over the predecessor map it
    inverts to. A bug in that inversion would produce a self-consistent field, a
    plan that still wins, and tie sets that are quietly wrong; and the training
    labels ARE those tie sets, so "it wins" is not enough of a check. Two passes,
    which fail differently:

      * **Bellman fixpoint.** Re-derive every distance by relaxing
        ``d(s) = 1 + min d(succ)`` to a fixpoint over the successor table alone
        -- no predecessor map, no BFS ordering, nothing shared with the reverse
        sweep -- and require it to agree everywhere, plus ``d == 0`` exactly at
        the winning states. Given a transition function that matches the
        interpreter (which `selfcheck` fuzzes) this is a proof of minimality,
        because the enumerated set is closed under every press.
      * **Executable labels.** Walk each level's plan on the INTERPRETER and, at
        every step, actually press each direction the label calls optimal: the
        board the engine lands on must be the native successor, and its field
        distance must be exactly one less. That takes the labels out of the model
        and puts them through the real thing.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)

        # -- pass 1: Bellman fixpoint over the successor table ----------------
        check = [0 if board.won(s) else -1 for s in board.states]
        changed = True
        while changed:
            changed = False
            for j, row in enumerate(board.succ):
                if row is None:
                    continue
                best = min((check[n] for n in row if check[n] >= 0), default=-1)
                if best >= 0 and (check[j] < 0 or best + 1 < check[j]):
                    check[j] = best + 1
                    changed = True
        if check != board.dist:
            bad += 1
            print(f"  L{level}: the field disagrees with its Bellman fixpoint "
                  f"at {sum(1 for a, b in zip(check, board.dist) if a != b)} "
                  f"states")
        if any((d == 0) != board.won(s)
               for d, s in zip(board.dist, board.states)):
            bad += 1
            print(f"  L{level}: distance 0 is not exactly the winning states")

        # -- pass 2: every shipped label, pressed on the interpreter ----------
        plan = board.plan(board.start)
        state = board.start
        labels = 0
        for i, press in enumerate(plan):
            here = board.distance(state)
            best = board.optimal(state)
            if press not in best:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            before = snapshot(eng)
            for alt in best:
                eng.step(alt)
                landed = expert.read(eng)
                there = board.distance(landed)
                labels += 1
                if landed != board.step(state, alt) or there != here - 1:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but the "
                          f"interpreter does not land one press closer")
                restore(eng, before)
            eng.step(press)
            state = board.step(state, press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        print(f"  L{level}: {len(board.states):4d} states relaxed, "
              f"{len(plan):3d} plan steps, {labels} labels pressed on the "
              f"interpreter -- {'OK' if not bad else 'see above'}")
    print(f"verify: {bad} problems")
    return 1 if bad else 0


def _engine() -> int:
    """Re-derive the WHOLE game from the INTERPRETER and require it to agree with
    the native field, state for state.

    `StateGraph.build` enumerates by pressing real buttons -- the native model is
    nowhere in this loop -- so this is the derivation that closes the loop back to
    the engine. The state key it is given is this file's ``(player, veggies)``
    reading of the grid, which is the only thing shared, and the whole point of
    `selfcheck` is that the reading is not where the risk is: the mechanic is.

    Three things must match, over every reachable state rather than a sample
    (5361 states, ~21k engine presses -- this game is small enough to afford it,
    which the bigger ps: sokobans are not):

      * the state SET (the interpreter keeps winning boards as edge targets
        rather than nodes, so the comparison is against the native non-winning
        states);
      * the distance field, including which states are dead;
      * every tie set.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        graph = StateGraph.build(eng, expert.read, list(_DIRS),
                                 node_cap=200_000)
        if graph is None:                             # pragma: no cover
            print(f"  L{level}: the interpreter enumeration hit the node cap")
            return 1

        native = {s: d for s, d in zip(board.states, board.dist)
                  if not board.won(s)}
        missing = set(native) - set(graph.succ)
        extra = set(graph.succ) - set(native)
        if missing or extra:
            bad += 1
            print(f"  L{level}: state sets differ ({len(missing)} only in the "
                  f"model, {len(extra)} only in the interpreter)")

        d_bad = t_bad = 0
        for s in set(native) & set(graph.succ):
            mine = native[s]
            theirs = graph.dist.get(s, -1)
            if mine != theirs:
                d_bad += 1
            if board.optimal(s) != graph.optimal(s, list(_DIRS)):
                t_bad += 1
        bad += bool(d_bad) + bool(t_bad)
        plan = graph.plan(list(_DIRS))
        mine_plan = board.plan(board.start)
        if (plan is None) != (mine_plan is None) or (
                plan is not None and len(plan) != len(mine_plan)):
            bad += 1
            print(f"  L{level}: plan lengths differ "
                  f"({None if plan is None else len(plan)} vs "
                  f"{None if mine_plan is None else len(mine_plan)})")
        print(f"  L{level}: {len(graph.succ):4d} interpreter states, "
              f"{graph.steps:5d} engine presses, d*="
              f"{None if plan is None else len(plan)}  "
              f"{d_bad} distance and {t_bad} tie-set disagreements")
    print("engine agrees with the native field everywhere" if not bad
          else f"ENGINE CROSS-CHECK FAILED: {bad} problems")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# Presentation: the render audit and the symmetry proof
# ---------------------------------------------------------------------------

def _compositions(game) -> list:
    """Every cell composition the engine can put on screen.

    ``player + <veggie>`` is absent on purpose and is not an omission: they share
    a collision layer, so no rule and no level can ever put them in one cell.
    ``wall + target`` is representable (different layers) but no rule creates a
    Target, so it can only exist if a level ships one -- asserted below rather
    than assumed."""
    idx = game._game.obj_name_to_idx
    eng = game._engine
    for level in range(game.n_levels):
        game.set_level(level)
        for row in eng.grid:
            for cell in row:
                assert not (idx["wall"] in cell and idx["target"] in cell), \
                    "a level ships a Wall on a Target cell"
    comps = [(), ("wall",), ("target",), ("player",), ("player", "target")]
    for v in _VEGGIES:
        comps += [(v,), (v, "target")]
    return comps


def _shoot(game, comps, h: int, w: int) -> dict:
    """A whole 64x64 frame per composition, on a uniform ``h`` x ``w`` board.

    Whole frames rather than cell crops: `_render_frame` upscales the board to
    fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    shots = {}
    for objs in comps:
        eng.height, eng.width = h, w
        eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                     for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        shots[objs] = np.asarray(_render_frame(eng, g)).copy()
    return shots


def _audit() -> int:
    """Assert every cell COMPOSITION the game can show is distinct in the frame,
    at every cell size the three levels render at -- and stays distinct under the
    eight turns and mirrors the augmentation applies.

    The pair this exists for is ``<veggie>+target`` against ``<veggie>``: the win
    condition is "every veggie on a Target", so a veggie that hid the target it
    is standing on would make the winning board indistinguishable from any other.
    All three veggie sprites have transparent margins and the Target is a filled
    red frame, so it survives -- but that is a measurement, not a reading of the
    .txt, and the 7x17 level renders at THREE pixels a cell, where a 5x5 sprite
    is decimated to 3x3 and margins are the first thing to go.

    Pass 2 is what the flip augmentation needs on top of `_symmetry`'s mechanical
    proof: this game's art is chiral (the Carrot's stalk points up-left, the Peas
    sit on a diagonal, the Target's bottom row is inset), so a mirrored board
    shows art the .txt does not literally contain. That is only a problem if the
    mirrored art of one composition is another composition's art."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    comps = _compositions(game)

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    name = lambda t: "+".join(t) if t else "floor"               # noqa: E731
    clashes_total = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = _shoot(game, comps, h, w)
        print(f"{h}x{w} board (level{'s' if len(levels) > 1 else ''} "
              f"{', '.join(str(x) for x in levels)}), "
              f"cell {max(1, min(64 // h, 64 // w))}px")
        for objs in comps:
            painted = int((shots[objs] != shots[()]).sum())
            over = ""
            if len(objs) == 2:
                body = (objs[0],)
                over = (f", {int((shots[objs] != shots[body]).sum()):4d} px of "
                        f"Target survive under the {objs[0].capitalize()}")
            print(f"  {name(objs):20s} {painted:4d} px differ from bare floor"
                  f"{over}")

        # -- pass 1: pairwise distinct as rendered ----------------------------
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        for a, b in clashes:
            print(f"  IDENTICAL: {name(a)} == {name(b)}")
        clashes_total += len(clashes)

        # -- pass 2: still distinct after the 8 presentation transforms -------
        mirrored = 0
        for a, b in itertools.permutations(shots, 2):
            for k, flip in itertools.product(range(4), (False, True)):
                if (k, flip) == (0, False):
                    continue
                img = np.fliplr(shots[a]) if flip else shots[a]
                if np.array_equal(np.rot90(img, k), shots[b]):
                    print(f"  MIRRORED CLASH: {name(a)} turned {k} "
                          f"{'and mirrored ' if flip else ''}== {name(b)}")
                    mirrored += 1
        clashes_total += mirrored
        print(f"  {len(comps)} compositions, "
              f"{len(comps) * (len(comps) - 1) * 7} transformed comparisons: "
              f"{'all distinct' if not (clashes or mirrored) else 'SEE ABOVE'}")
    print("audit clean" if not clashes_total
          else f"AUDIT FAILED: {clashes_total} indistinguishable pairs")
    return 0 if not clashes_total else 1


def _transform(k: int, mirror: bool):
    """``(cell_fn, dims_fn, direction_map)`` for one element of the 8-group:
    mirror left-right first, then turn clockwise ``k`` times.

    The direction map is DERIVED from the same linear part that moves the cells,
    so the two cannot drift apart."""
    def cell(rc, hw):
        r, c = rc
        h, w = hw
        if mirror:
            c = w - 1 - c
        for _ in range(k):
            r, c, h, w = c, h - 1 - r, w, h
        return (r, c)

    def dims(hw):
        h, w = hw
        return (w, h) if k % 2 else (h, w)

    dmap = {}
    for name, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[name] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    #: ACTION carries no direction, so every element of the group fixes it. It is
    #: in the map because `_symmetry`'s random walk presses it -- the button a
    #: live agent has -- and a walk that skipped it would not be measuring the
    #: alphabet the exploration prefix uses.
    dmap["action"] = "action"
    return cell, dims, dmap


def _replay_transformed(eng, expert, layout, presses) -> list[str]:
    """Replay ``presses`` on all seven non-identity turned/mirrored copies of
    ``layout`` and return a note per presentation that diverged.

    BOTH pieces are compared, not only the player: a mirror that moved the veggie
    somewhere else while leaving the player where the transform says would be
    exactly the kind of chirality this is looking for.

    The transformed board is built from the LEVEL LAYOUT and reloaded, so the
    interpreter parses and resolves everything itself rather than being handed a
    grid this file transformed after the fact."""
    hw = (len(layout), len(layout[0]))
    eng.load_level(layout)
    ref = [expert.read(eng)]
    for d in presses:
        eng.step(d)
        ref.append(expert.read(eng))

    notes = []
    for k, mirror in itertools.product(range(4), (False, True)):
        if (k, mirror) == (0, False):
            continue
        cell, dims, dmap = _transform(k, mirror)
        th, tw = dims(hw)
        turned = [[set() for _ in range(tw)] for _ in range(th)]
        for r, row in enumerate(layout):
            for c, objs in enumerate(row):
                tr, tc = cell((r, c), hw)
                turned[tr][tc] = set(objs)
        eng.load_level(turned)
        for i, d in enumerate(presses):
            eng.step(dmap[d])
            player, veggies = ref[i + 1]
            flat = lambda x: (lambda rc: rc[0] * tw + rc[1])(   # noqa: E731
                cell(divmod(x, hw[1]), hw))
            want = (flat(player), tuple(sorted(flat(v) for v in veggies)))
            if expert.read(eng) != want:
                notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                break
    return notes


def _symmetry(walk: int = 400) -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an exact
    symmetry of the mechanic, which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation.

    The structural argument (see the comment beside "Veggie_Jam" in that set) is
    the unqualified one -- three rules, all stated with the relative `>` force,
    no gravity, screen-relative input, a positional win condition, and one press
    moving at most two bodies onto ``p + d`` and ``p + 2d`` so nothing can ever
    contest a cell -- but the check is nearly free, so it is measured anyway,
    twice over:

      * each level's own PLAN, which is the sequence the corpus actually records
        and the only one that reaches the win; but a shortest plan never presses
        into a wall and never shoves a veggie that cannot move, which are the two
        things the interpreter has to decide here;
      * a seeded RANDOM WALK per level, which does little else, and which presses
        the unbound ACTION key too.
    """
    _solver, game, expert, _solvable = _new()
    eng, g = game._engine, game._game
    alphabet = _DIRS + ("action",)
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        layout = g.levels[level]
        rng = random.Random(f"veggie_jam:symmetry:{level}")
        runs = {"plan": list(board.plan(board.start)),
                "walk": [rng.choice(alphabet) for _ in range(walk)]}
        notes = []
        for kind, presses in runs.items():
            notes += [f"{kind}:{n}" for n in
                      _replay_transformed(eng, expert, layout, presses)]
        bad += len(notes)
        print(f"  L{level}: ({len(runs['plan'])} plan + {walk} random) presses "
              f"x 7 presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print("symmetry clean -- the flip augmentation is exact" if not bad
          else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        bad = selfcheck()
        print(f"selfcheck: {bad} mismatches")
        sys.exit(1 if bad else _report())
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--engine" in sys.argv:
        sys.exit(_engine())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    sys.exit(VeggieJamSolver.main())
