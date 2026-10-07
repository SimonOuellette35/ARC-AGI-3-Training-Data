"""Generate Phase-1 training data for the PuzzleScript game ps:piedra
("Piedra" by Enzo).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, the native model of the turn, the exact distance field over the
whole state space, and the optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_piedra",
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
exactly.

The game
--------
Sisyphus, in nine by nine. Five objects, ONE rule -- the classic PuzzleScript
push, ``[ > Player | Crate ] -> [ > Player | > Crate ]`` -- and a single crate
to shove into the single Target, which on every level is a notch cut in the top
wall row. The four level titles ("Empujar una piedra cuesta arriba", "Esfuerzo
en vano", "Labor sin frutos", "Tiempo perdido") are the joke; mechanically
nothing rolls back and there is no gravity, so this is a one-crate sokoban:

  * **Player, Wall and Crate share a collision layer.** A Wall therefore blocks
    both, and a crate whose destination is a wall or the board edge cancels the
    WHOLE move -- the player does not step either.
  * **Target is on its own layer**, so both the Player and the Crate walk onto
    it freely, and the win (``All Target on Crate``) is a snapshot rather than a
    latch. The player standing on the target is a perfectly ordinary state, and
    an early press can put it there.
  * **Four arrow keys, no ACTION.** ``action`` parses (the game does not declare
    ``noaction``) but no rule reads it and there is no ``late`` rule to tick, so
    it cannot change the board -- ``--selfcheck`` puts it in the fuzz alphabet
    and asserts that against the interpreter rather than reading it off the
    rules section (see ps:ouroboros, where exactly that assumption was false).
  * **No ``restart``, no ``again``, no ``random``** -- the whole rules section is
    the one line above, so a press is one deterministic board rewrite.

So the mechanic is a plain one-crate sokoban, which is worth saying plainly:
there is no trick here and a solver that hunts for one is inventing it. What the
four levels escalate is the maze around the crate -- level 1 is an empty room,
and by level 3 the route winds enough that d* runs 7, 19, 35, 33.

Where it is NOT a maze is that it is IRREVERSIBLE: shoving the crate against a
wall it cannot come off is unrecoverable, and 40% of the reachable states are
exactly that (``--plans`` prints the column). That is what the exact field below
is for.

Model + search
--------------
`_Board.step` re-implements one turn natively; ``--selfcheck`` drives 19k random
presses through BOTH the interpreter and the model across all four levels and
compares the player AND crate cells (and the win flag) after every one. That is
the guard that lets the search trust it.

The state is ``(player cell, crate cell)``, so the reachable space is at most the
free area squared -- 1168 to 2499 states per level, 7286 in the whole game. At
that size there is no search: `_Board.enumerate` walks the space once from the
level start, keeps the successor table and runs one backward BFS from the
winning states to get the exact distance-to-win field. A plan is a descent of the
field, a tie set is a lookup, and a state reached by an exploration detour is
answered from the same table as the start state. The whole game is built in
milliseconds, once per process, and shared by every seed.

Optimal-action targets and recovery
-----------------------------------
`optimal_for` reads the LIVE engine state and returns every press that lowers the
exact distance -- true tie sets, not a reordering heuristic. Where it bites is
the walk between pushes, which is completely order-free ("three right then two
up" and "two up then three right" put the player on the same cell in the same
number of presses, and no rule fires on a bare move), so labelling one arbitrary
interleaving as the single right answer would train a coin flip the policy cannot
win. ``--plans`` prints the measured average.

``epsilon = 0.12`` is safe here DESPITE the irreversibility, and it is the field
that makes it so. `record_level`'s detour probe accepts a random alternative only
if the expert can still plan a win from the state it lands in; for most of this
family that test is a bounded search and therefore a guess, but here it is a
dict lookup in an exact field, so a detour can never strand the crate. The step
then records the mistake as the action taken and the recovery as ``optimal`` --
on top of the explore-then-RESET prefix that opens every episode.

Rendering
---------
One colour changed and two sprites ADDED, in the game FILE (see the comment at
the top of data/puzzlescript_games/Piedra.txt). Target shipped as ``#27AE60`` --
the Background's own colour, so the goal cell rendered as bare floor -- and every
object shipped spriteless, so the Crate (drawn above the Target) painted the
winning board exactly like a crate anywhere else. Target is now red, and Player
and Crate are 3x3 bodies with a transparent border, so what they stand on shows
through as a ring. ``--audit`` is the check, and it compares WHOLE FRAMES rather
than cell crops (`_render_frame` upscales the board to fill 64x64 and letterboxes
it, so an arithmetic cell crop reads the wrong window).

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (0..3) and, since ``Piedra``
is in `PuzzleScriptAdapter._FLIP_GAMES`, an independent horizontal and vertical
flip -- 16 presentations of each of the 4 levels, which matters more here than
usual because there are only four. The flips are measured rather than argued:
``--symmetry`` rebuilds every level's LAYOUT under all 8 transforms, reloads it
into the interpreter, and replays the level's own plan plus a 400-press random
walk with the presses transformed; both pieces must land where the transform says
after every one.

Usage (run from the repo root):
    python solvers/generate_piedra_training.py --episodes 200 \
        --out data/training_multi_level/piedra
    python solvers/generate_piedra_training.py --selfcheck
    python solvers/generate_piedra_training.py --plans
    python solvers/generate_piedra_training.py --verify
    python solvers/generate_piedra_training.py --audit
    python solvers/generate_piedra_training.py --symmetry
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
    PSAStarSolver, PSExpert, restore, snapshot)

GAME_NAME = "Piedra"

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The action space the search branches on. ``action`` parses but no rule reads
#: it and there is no `late` rule for it to tick, so it cannot change the board
#: -- `selfcheck` asserts that against the interpreter rather than trusting the
#: (one-line) rules section.
_DIRS = ("up", "down", "left", "right")


# ---------------------------------------------------------------------------
# The level, natively: one turn, the whole state space, the distance field
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry plus its COMPLETE state space and the exact
    distance-to-win field over it.

    Cells are flat ``r * w + c`` ids and a state is the pair
    ``(player cell, crate cell)``: there is one Player and one Crate, the Walls
    and the Target never change, and the one rule creates and destroys nothing,
    so the pair IS the state.

    `enumerate` is what everything else reads: one forward BFS from the level
    start collecting every reachable state and its four successors, then one
    backward BFS from the winning states over that edge set. `plan` is a descent
    down the field, `optimal` a lookup, and both work from ANY reachable state --
    which is what lets the recording take real detours and label the recovery,
    and what makes the detour probe's "is this still winnable" test exact in a
    game where a wrong shove is permanent.
    """

    __slots__ = ("h", "w", "walls", "targets", "start", "_next",
                 "states", "index", "succ", "dist")

    def __init__(self, h: int, w: int, walls, targets, player: int, crate: int):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.targets = frozenset(targets)
        self.start = (player, crate)

        # ``_next[d][cell]`` is the cell one step along d, or -1 when that is off
        # the board or a Wall. Both are the same thing to this game: the mover
        # stops, and if the mover is a shoved crate the player's own step is
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
        """``All Target on Crate`` -- one target, one crate on every level."""
        return state[1] in self.targets

    def step(self, state, d: str):
        """One press, natively.

        Walk unless a Wall or the board edge is there; if the Crate is in the
        way, shove it, and if IT cannot move then the whole press is cancelled --
        the player does not step either, because the interpreter's push rule
        binds both movements into one match."""
        p, c = state
        n = self._next[d][p]
        if n < 0:
            return state
        if n == c:
            m = self._next[d][c]
            if m < 0:
                return state
            return (n, m)
        return (n, c)

    # -- the whole space ------------------------------------------------------
    def enumerate(self) -> tuple[int, int]:
        """Build the reachable state set, its successor table and the exact
        distance-to-win field. Returns ``(states, dead states)``.

        The set is CLOSED under every press (each state contributes all four of
        its successors), so any state the agent can reach by any sequence of
        presses -- plan, exploration prefix, epsilon detour, or a RESET back to
        the start -- is already in it. Winning states are terminal: the adapter
        ends the level there, so they are not expanded.

        Unlike the maze games in this family, the dead count is LARGE and that is
        the mechanic, not a bug: a crate shoved flat against a wall can only ever
        slide along it, and into a corner it cannot move at all.
        """
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
        """Presses to a win from ``state``, or None when the crate can no longer
        reach the target (or the state is not reachable from the level start,
        which cannot happen -- see `enumerate`)."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] < 0:
            return None
        return self.dist[j]

    def plan(self, state) -> list | None:
        """A SHORTEST press sequence from ``state`` to a win, or None.

        A descent of the field, taking the first tied direction in ``_DIRS``
        order, so the plan is a pure function of the board and does not vary with
        the interpreter's hash seed."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] < 0:
            return None
        out = []
        while self.dist[j] > 0:
            row = self.succ[j]
            for k, nxt in enumerate(row):
                if self.dist[nxt] == self.dist[j] - 1:
                    out.append(_DIRS[k])
                    j = nxt
                    break
            else:                                    # pragma: no cover
                raise AssertionError("distance field has no descent")
        return out

    def optimal(self, state) -> list:
        """Every press that keeps the game on a SHORTEST route to the win.

        Exact, straight off the field. A press that is refused (a wall, the board
        edge, an unshovable crate) maps the state to itself, whose distance is
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


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class PiedraExpert(PSExpert):
    """Exact planner over the enumerated state space (see `_Board`).

    `PSExpert` supplies the plan memo, the restore discipline and the level
    scoping; `_search` swaps in the field descent, so the interpreter is only
    ever stepped by the recorder -- which is also what verifies every plan, since
    a level is kept only when the engine reports WIN.
    """

    directions = list(_DIRS)

    #: `_key` is the (player, crate) pair, canonical only WITHIN a level (the
    #: Walls and the Target that complete the state are static per level but
    #: differ between levels).
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.target_ids = set(g.resolve_object_name("target"))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.player_ids = set(self.game._engine._player_indices)
        self._boards: dict[int | None, _Board] = {}
        self._cur: _Board | None = None

    # -- reading the engine ---------------------------------------------------
    def read(self, eng):
        """The ``(player cell, crate cell)`` pair from the engine grid."""
        w = len(eng.grid[0])
        player = crate = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & self.player_ids:
                    player = r * w + c
                if cell & self.crate_ids:
                    crate = r * w + c
        if player is None or crate is None:          # pragma: no cover
            raise AssertionError("the board is missing a Player or a Crate")
        return (player, crate)

    def board(self, eng, level: int | None) -> _Board:
        """The level's `_Board`, built (and enumerated) once and cached.

        The Walls and the Target are static -- the game's one rule only moves a
        Crate -- so whichever state happens to build it describes the same
        level."""
        board = self._boards.get(level)
        if board is not None:
            return board
        h, w = len(eng.grid), len(eng.grid[0])
        walls, targets = set(), set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & self.wall_ids:
                    walls.add(r * w + c)
                if cell & self.target_ids:
                    targets.add(r * w + c)
        # One crate and one target on every shipped level, which is what lets
        # `_Board.won` be a membership test rather than a matching. Asserted
        # rather than assumed, because a level with two of either would make the
        # state key wrong (not slow -- wrong).
        assert len(targets) == 1, f"level {level} ships {len(targets)} Targets"
        player, crate = self.read(eng)
        board = _Board(h, w, walls, targets, player, crate)
        board.enumerate()
        self._boards[level] = board
        return board

    # -- planning -------------------------------------------------------------
    def _key(self, eng):
        return self.read(eng)

    def plan(self, eng, level: int | None = None) -> list | None:
        self._cur = self.board(eng, level)
        return super().plan(eng, level)

    def heuristic(self, eng) -> int:
        """The EXACT remaining press count (the base `_astar` is never used here,
        but the contract is that this is 0 at a win and admissible, and the field
        is both)."""
        d = self._cur.distance(self.read(eng))
        return 0 if d is None else d

    def _search(self, eng) -> list | None:
        return self._cur.plan(self.read(eng))

    def optimal_dirs(self, eng, level: int | None) -> list:
        """Every press on a shortest route, from the engine's CURRENT state."""
        return self.board(eng, level).optimal(self.read(eng))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class PiedraSolver(PSAStarSolver):
    game_id = "puzzlescript_piedra"
    game_name = GAME_NAME
    expert_cls = PiedraExpert

    #: Unused -- `PiedraExpert._search` never calls the base A* -- but left at
    #: the family default so a future subclass that does is not silently starved.
    node_cap = 2_000_000
    weight = 1
    #: The adapter GAME_OVERs at 200 actions per level; the longest plan here is
    #: 35 presses, leaving the rest of the budget to the exploration prefix and
    #: the detours.
    max_steps = 200

    #: Non-zero even though this game is IRREVERSIBLE, which is a departure from
    #: the rest of this family and is earned by the exact field. `record_level`'s
    #: detour probe steps a random alternative, asks the expert whether it can
    #: still plan a win, and keeps the detour only if it can; for a game whose
    #: expert is a bounded search that test is a guess that can decline a live
    #: state, but here it is a dict lookup in a field that covers the whole
    #: reachable space, so a detour provably cannot strand the crate. Roughly one
    #: press in eight is then a random legal alternative, after which the expert
    #: re-plans from wherever it landed: the taken action is the mistake and
    #: ``optimal`` is the recovery, which is the signal a policy needs after its
    #: own error. The RESET prefix still runs in front of it.
    epsilon = 0.12

    def prepare_expert(self, game, expert) -> None:
        """Enumerate every level's state space up front.

        Pure front-loading -- the space does not depend on which state builds it
        -- but it means the per-level BFS is paid once, at startup, rather than
        inside the first query that needs it."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.board(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route to the win, read
        off the LIVE engine state.

        `record_level` asks for this before executing the press, so the engine is
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

def _levels():
    """``(solver, game, expert)`` with every level's space enumerated."""
    solver = PiedraSolver()
    game, expert, _solvable = solver._ensure(0)
    return solver, game, expert


def selfcheck(trials: int = 60, steps: int = 80, verbose: bool = True) -> int:
    """Drive random presses through BOTH the interpreter and `_Board.step`,
    comparing the Player AND Crate cells AND the win flag after every one. This
    is the guard that lets the planner trust the native model.

    ``action`` is included in the press alphabet even though the search excludes
    it: the claim that it is a pure no-op is exactly the kind of thing that
    should be measured against the interpreter rather than read off a rules
    section, however short (ps:ouroboros's ACTION is a shrink button and its
    rules section does not mention ``action`` either).

    The walks run from a COLD level start, with no warm-up press, because
    turn-one bookkeeping is precisely the class of mechanic a warm-up hides.

    Returns the number of mismatches."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = PiedraExpert(game)
    alphabet = _DIRS + ("action",)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine, level)
        rng = random.Random(f"piedra:selfcheck:{level}")
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
                # The closure `_Board.enumerate` relies on, checked rather than
                # argued: a state a random walk from the start can reach is
                # already in the enumerated space, so the field can answer for
                # anything the exploration prefix or an epsilon detour lands on.
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

    The dead column is expected to be large: this is a sokoban, so a crate shoved
    against the wrong wall ends the level's chances. What must hold is that every
    level's START is live, which is the ``win=True`` column."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = dead_total = states_total = 0
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
        ties = sum(len(board.optimal(s)) for s in _walk(board, plan))
        print(f"  L{level}: {eng.height}x{eng.width}  {len(plan):3d} presses  "
              f"win={won}  {n_states:4d} states "
              f"({n_dead:4d} dead, {n_dead / n_states:.0%})  "
              f"{ties / len(plan):.2f} optimal presses/step")
    print(f"  total: {states_total} states, {dead_total} dead "
          f"({dead_total / states_total:.0%} -- a crate against the wrong wall "
          f"never comes back)")
    return 1 if bad else 0


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
    _solver, game, expert = _levels()
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
              f"{len(plan):3d} plan steps, "
              f"{sum(len(board.optimal(s)) for s in _walk(board, plan))} labels "
              f"pressed on the interpreter -- "
              f"{'OK' if not bad else 'see above'}")
    print(f"verify: {bad} problems")
    return 1 if bad else 0


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
    return cell, dims, dmap


def _replay_transformed(eng, expert, layout, presses) -> list[str]:
    """Replay ``presses`` on all seven non-identity turned/mirrored copies of
    ``layout`` and return a note per presentation that diverged.

    Both pieces are compared, not only the player: a mirror that moved the crate
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
            want = tuple(
                (lambda rc: rc[0] * tw + rc[1])(cell(divmod(x, hw[1]), hw))
                for x in ref[i + 1])
            if expert.read(eng) != want:
                notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                break
    return notes


def _symmetry(walk: int = 400) -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an exact
    symmetry of the mechanic, which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation.

    The argument (see the comment beside "Piedra" in that set) is as strong as it
    gets in this family -- one rule stated with the relative `>` force, one mover
    plus the crate it shoves so nothing can ever contest a cell, no gravity, a
    win condition that names no direction, and sprites invariant under the whole
    group -- but the check is nearly free, so it is measured anyway, twice over:

      * each level's own PLAN, which is the sequence the corpus actually records
        and the only one that reaches the win; but a shortest plan never presses
        into a wall and never shoves a crate that cannot move, which are the two
        things the interpreter has to decide here;
      * a seeded RANDOM WALK per level, which does little else.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        layout = g.levels[level]
        rng = random.Random(f"piedra:symmetry:{level}")
        runs = {"plan": board.plan(board.start),
                "walk": [rng.choice(_DIRS) for _ in range(walk)]}
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


def _audit() -> int:
    """Assert every cell COMPOSITION the game can show is distinct in the frame,
    at every cell size the four levels render at.

    The pair this exists for is ``crate+target`` against ``crate``: the win
    condition is ``All Target on Crate``, so a Crate that hides the Target it is
    standing on makes the winning board indistinguishable from any other. That is
    exactly what shipped -- Target was painted the Background's own colour AND
    every object was a bare colour with no sprite, so the Crate covered it twice
    over -- see the header of data/puzzlescript_games/Piedra.txt.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    # Wall and Target are on different collision layers, so `wall+target` is
    # representable -- but the one rule creates nothing, so it can only exist if
    # a level ships it, and this asserts none does. That is what keeps it out of
    # the composition list below. (Player+Crate is NOT representable: they share
    # a layer.)
    for level in range(game.n_levels):
        game.set_level(level)
        for row in eng.grid:
            for cell in row:
                assert not (idx["wall"] in cell and idx["target"] in cell), \
                    "a level ships a Wall on a Target cell"

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    comps = [(), ("wall",), ("target",), ("player",), ("crate",),
             ("player", "target"), ("crate", "target")]
    clashes_total = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for objs in comps:
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[objs] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        clashes_total += len(clashes)
        name = lambda t: "+".join(t) if t else "floor"           # noqa: E731
        print(f"{h}x{w} board (level{'s' if len(levels) > 1 else ''} "
              f"{', '.join(str(x) for x in levels)}), "
              f"cell {max(1, min(64 // h, 64 // w))}px")
        for objs in comps:
            painted = int((shots[objs] != shots[()]).sum())
            over = ""
            if len(objs) == 2:
                body = (objs[0],)
                over = (f", {int((shots[objs] != shots[body]).sum()):4d} "
                        f"px of Target survive under the {objs[0].capitalize()}")
            print(f"  {name(objs):16s} {painted:4d} px differ from bare floor"
                  f"{over}")
        for a, b in clashes:
            print(f"  IDENTICAL: {name(a)} == {name(b)}")
    print("audit clean" if not clashes_total
          else f"AUDIT FAILED: {clashes_total} indistinguishable pairs")
    return 0 if not clashes_total else 1


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        bad = selfcheck()
        print(f"selfcheck: {bad} mismatches")
        sys.exit(1 if bad else _report())
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    sys.exit(PiedraSolver.main())
