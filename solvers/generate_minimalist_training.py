"""Generate Phase-1 training data for the PuzzleScript game ps:minimalist
("Minimalist" by Marcos Perez).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, the native model of the turn, the exact distance field over the
whole state space, and the optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_minimalist",
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
The title is the design document. Four objects, ONE rule, and that rule is the
identity ``[ > Player ] -> [ > Player ]`` -- it rewrites the board to itself, so
it can be deleted without changing anything the interpreter does. The whole game
is therefore the interpreter's movement resolution against a collision layer:

  * **Player and Block share a collision layer**, so a Block is a wall. Nothing
    pushes it (no rule mentions it) and nothing destroys it, so the maze is
    static for the whole level.
  * **Finish is on its own layer**, so the Player walks onto it freely, and the
    win (``all Player on Finish``) is a snapshot rather than a latch.
  * **One Player, four arrow keys, no ACTION.** ``ACTION`` parses (the game does
    not declare ``noaction``) but no rule reads it; ``--selfcheck`` includes it
    in the fuzz alphabet and asserts it is a pure no-op rather than reading that
    off the rules section.

So ps:minimalist is a plain 4-connected maze with one start and one goal, which
is worth saying plainly: there is no hidden mechanic here, and a solver that
looks for one is inventing it. What the levels escalate is the maze. Level 1 is
an open 5x5 with no walls at all; every level after it is essentially a single
corridor, wound tighter each time up to a 9x9 spiral on level 10. So the
interesting part of this corpus is the LENGTH of the route -- d* runs 2, 6, 7,
16, 22, 30, 22, 28, 30, 46 -- and not its branching, which past level 1 is
almost none.

Model + search
--------------
`_Board.step` re-implements one turn natively; ``--selfcheck`` drives 60k random
presses through BOTH the interpreter and the model across all 10 levels and
compares the player cell (and the win flag) after every one. That is the guard
that lets the search trust it.

The state is the Player's cell, so the reachable space IS the free area of the
maze -- 8 to 48 cells per level, 259 in the whole game. At that size there is no
search: `_Board.enumerate` walks the space once from the level start, keeps the
successor table and runs one backward BFS from the winning cells to get the
exact distance-to-win field. A plan is a descent of the field, a tie set is a
lookup, and a state reached by an exploration detour is answered from the same
table as the start state. The whole game is built in milliseconds, once per
process, and shared by every seed.

The enumeration also measures the fact the recording relies on: **not one
reachable state is dead.** Movement in a maze is its own inverse (step back the
way you came), so the reachable set is one connected component and it contains
the Finish on all ten levels -- which is why `epsilon` is non-zero here (see the
class comment).

Optimal-action targets and recovery
-----------------------------------
`optimal_for` reads the LIVE engine state and returns every press that lowers the
exact distance -- true tie sets, not a reordering heuristic. Where it bites is
open floor, which is completely order-free ("three right then two up" and "two up
then three right" are the same route in the same number of presses), so
labelling one arbitrary interleaving as the single right answer would train a
coin flip the policy cannot win. This game has less of that than most: measured
over the ten shortest plans (``--plans`` prints the column), the wall-free level
1 averages 1.50 optimal presses per step and level 8 averages 1.04, and the other
eight are corridors where the route really is forced at 1.00. So the tie sets are
cheap insurance here rather than the main event -- but they are also what makes
the epsilon detours below labelable, since a detour lands off the plan entirely.

Because the field answers from any state, ``epsilon = 0.12`` is safe: roughly one
press in eight is a random legal alternative, and the step records the mistake as
the action taken and the recovery as ``optimal`` -- on top of the
explore-then-RESET prefix that opens every episode.

Rendering
---------
One sprite ADDED, in the game FILE (see the comment at the top of
data/puzzlescript_games/Minimalist.txt): the Player is now a 3x3 body with a
transparent border instead of a solid block. Every object in this game shipped as
a bare colour with no sprite, and the Player is drawn last, so a Player standing
on the Finish painted exactly the same lightblue square as a Player standing on
floor -- i.e. the win condition was not drawn. ``--audit`` is the check, and it
compares WHOLE FRAMES rather than cell crops (`_render_frame` upscales the board
to fill 64x64 and letterboxes it, so an arithmetic cell crop reads the wrong
window).

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (0..3) and, since
``Minimalist`` is in `PuzzleScriptAdapter._FLIP_GAMES`, an independent horizontal
and vertical flip -- 16 presentations of each of the 10 levels. The flips are
measured rather than argued: ``--symmetry`` rebuilds every level's LAYOUT under
all 8 transforms, reloads it into the interpreter, and replays the level's own
plan plus a 400-press random walk with the presses transformed; the player must
land where the transform says after every one.

Usage (run from the repo root):
    python solvers/generate_minimalist_training.py --episodes 200 \
        --out data/training_multi_level/minimalist
    python solvers/generate_minimalist_training.py --selfcheck
    python solvers/generate_minimalist_training.py --plans
    python solvers/generate_minimalist_training.py --verify
    python solvers/generate_minimalist_training.py --audit
    python solvers/generate_minimalist_training.py --symmetry
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

GAME_NAME = "Minimalist"

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The action space the search branches on. ``action`` parses but no rule reads
#: it, so it cannot change the board -- `selfcheck` asserts that against the
#: interpreter rather than trusting the (one-line, identity) rules section.
_DIRS = ("up", "down", "left", "right")


# ---------------------------------------------------------------------------
# The level, natively: one turn, the whole state space, the distance field
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry plus its COMPLETE state space and the exact
    distance-to-win field over it.

    Cells are flat ``r * w + c`` ids and a state is just the Player's cell: there
    is one Player, nothing else on the board ever moves, and no rule creates or
    destroys anything, so the cell IS the state.

    `enumerate` is what everything else reads: one forward BFS from the level
    start collecting every reachable cell and its four successors, then one
    backward BFS from the winning cells over that edge set. `plan` is a descent
    down the field, `optimal` a lookup, and both work from ANY reachable state --
    which is what lets the recording take real detours and label the recovery.
    """

    __slots__ = ("h", "w", "walls", "finish", "start", "_ahead",
                 "states", "index", "succ", "dist")

    def __init__(self, h: int, w: int, walls, finish, start: int):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.finish = frozenset(finish)
        self.start = start

        # ``_ahead[d][cell]`` is the cell one step along d, or the cell itself
        # when that is off the board or a Block -- the interpreter cancels the
        # move, it does not refuse the press. Precomputed because it is the inner
        # loop of the enumeration and of the fuzz.
        self._ahead = {}
        for d, (dr, dc) in _DELTA.items():
            table = list(range(h * w))
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < h and 0 <= nc < w
                            and nr * w + nc not in self.walls):
                        table[r * w + c] = nr * w + nc
            self._ahead[d] = table

        self.states: list | None = None
        self.index: dict = {}
        self.succ: list = []
        self.dist: list = []

    # -- one turn -------------------------------------------------------------
    def won(self, state: int) -> bool:
        """``all Player on Finish``."""
        return state in self.finish

    def step(self, state: int, d: str) -> int:
        """One press, natively: walk unless a Block or the board edge is there."""
        return self._ahead[d][state]

    # -- the whole space ------------------------------------------------------
    def enumerate(self) -> tuple[int, int]:
        """Build the reachable state set, its successor table and the exact
        distance-to-win field. Returns ``(states, dead states)``.

        The set is CLOSED under every press (each state contributes all four of
        its successors), so any state the agent can reach by any sequence of
        presses -- plan, exploration prefix, epsilon detour, or a RESET back to
        the start -- is already in it. Winning states are terminal: the adapter
        ends the level there, so they are not expanded.
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
    def distance(self, state: int) -> int | None:
        """Presses to a win from ``state``, or None if it cannot win (or is not
        reachable from the level start, which cannot happen -- see `enumerate`)."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] < 0:
            return None
        return self.dist[j]

    def plan(self, state: int) -> list | None:
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

    def optimal(self, state: int) -> list:
        """Every press that keeps the game on a SHORTEST route to the win.

        Exact, straight off the field. A press into a wall or off the board maps
        the state to itself, whose distance is unchanged rather than one less, so
        no-ops are excluded for free."""
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

class MinimalistExpert(PSExpert):
    """Exact planner over the enumerated state space (see `_Board`).

    `PSExpert` supplies the plan memo, the restore discipline and the level
    scoping; `_search` swaps in the field descent, so the interpreter is only
    ever stepped by the recorder -- which is also what verifies every plan, since
    a level is kept only when the engine reports WIN.
    """

    directions = list(_DIRS)

    #: `_key` is the Player cell alone, canonical only WITHIN a level (the Blocks
    #: and the Finish that complete the state are static per level but differ
    #: between levels).
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        self.block_ids = set(g.resolve_object_name("block"))
        self.finish_ids = set(g.resolve_object_name("finish"))
        self.player_ids = set(self.game._engine._player_indices)
        self._boards: dict[int | None, _Board] = {}
        self._cur: _Board | None = None

    # -- reading the engine ---------------------------------------------------
    def read(self, eng) -> int:
        """The Player's flat cell id from the engine grid."""
        w = len(eng.grid[0])
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & self.player_ids:
                    return r * w + c
        raise AssertionError("no Player on the board")           # pragma: no cover

    def board(self, eng, level: int | None) -> _Board:
        """The level's `_Board`, built (and enumerated) once and cached.

        Blocks and the Finish are static -- the game's one rule is the identity,
        so nothing but the Player ever changes -- and whichever state happens to
        build it therefore describes the same level."""
        board = self._boards.get(level)
        if board is not None:
            return board
        h, w = len(eng.grid), len(eng.grid[0])
        walls, finish = set(), set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & self.block_ids:
                    walls.add(r * w + c)
                if cell & self.finish_ids:
                    finish.add(r * w + c)
        board = _Board(h, w, walls, finish, self.read(eng))
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

class MinimalistSolver(PSAStarSolver):
    game_id = "puzzlescript_minimalist"
    game_name = GAME_NAME
    expert_cls = MinimalistExpert

    #: Unused -- `MinimalistExpert._search` never calls the base A* -- but left
    #: at the family default so a future subclass that does is not silently
    #: starved.
    node_cap = 2_000_000
    weight = 1
    #: The adapter GAME_OVERs at 200 actions per level; the longest plan here is
    #: 46 presses, leaving the rest of the budget to the exploration prefix and
    #: the detours.
    max_steps = 200

    #: Non-zero, unlike most of this family. The `PSAStarSolver` default is 0
    #: because those games are IRREVERSIBLE -- a detour can strand them, so their
    #: recovery data has to come from the explore-then-RESET prefix alone. This
    #: game has nothing to strand: every move is undone by the opposite one, and
    #: the enumeration MEASURES it -- of the 259 states reachable across the ten
    #: levels, not one is dead (``--plans`` prints the column). So one press in
    #: eight is a random legal alternative, after which the expert re-plans from
    #: wherever it landed: the taken action is the mistake and ``optimal`` is the
    #: recovery, which is the signal a policy needs after its own error. The
    #: RESET prefix still runs in front of it.
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
    solver = MinimalistSolver()
    game, expert, _solvable = solver._ensure(0)
    return solver, game, expert


def selfcheck(trials: int = 60, steps: int = 100, verbose: bool = True) -> int:
    """Drive random presses through BOTH the interpreter and `_Board.step`,
    comparing the Player cell AND the win flag after every one. This is the guard
    that lets the planner trust the native model.

    ``action`` is included in the press alphabet even though the search excludes
    it: the claim that it is a pure no-op is exactly the kind of thing that
    should be measured against the interpreter rather than read off a rules
    section, however short.

    Returns the number of mismatches."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = MinimalistExpert(game)
    alphabet = _DIRS + ("action",)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine, level)
        rng = random.Random(f"minimalist:selfcheck:{level}")
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
    the quick "is this game still fully solved" check."""
    _solver, game, expert = _levels()
    eng = game._engine
    dead_total = states_total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        n_states, n_dead = board.enumerate()
        states_total += n_states
        dead_total += n_dead
        plan = board.plan(board.start)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            continue
        for d in plan:
            eng.step(d)
        won = eng.check_win()                   # read BEFORE anything reloads
        ties = sum(len(board.optimal(s)) for s in _walk(board, plan))
        print(f"  L{level}: {eng.height}x{eng.width}  {len(plan):3d} presses  "
              f"win={won}  {n_states:3d} states ({n_dead} dead)  "
              f"{ties / len(plan):.2f} optimal presses/step")
    print(f"  total: {states_total} states, {dead_total} dead "
          f"-- every reachable state can still win"
          if not dead_total else
          f"  total: {states_total} states, {dead_total} DEAD")
    return 0 if not dead_total else 1


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
        print(f"  L{level}: {len(board.states):3d} states relaxed, "
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
            tr, tc = cell(divmod(ref[i + 1], hw[1]), hw)
            if expert.read(eng) != tr * tw + tc:
                notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                break
    return notes


def _symmetry(walk: int = 400) -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an exact
    symmetry of the mechanic, which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation.

    The argument (see the comment beside "Minimalist" in that set) is as strong
    as it gets in this family -- one identity rule stated with the relative `>`
    force, one mover, no gravity, a direction-free win condition and sprites that
    are invariant under the whole group -- but the check is nearly free, so it is
    measured anyway, twice over:

      * each level's own PLAN, which is the sequence the corpus actually records
        and the only one that reaches the win; but a shortest plan never presses
        into a wall, which is the one thing the interpreter has to decide here;
      * a seeded RANDOM WALK per level, which does little else.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        layout = g.levels[level]
        rng = random.Random(f"minimalist:symmetry:{level}")
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
    at every cell size the ten levels render at.

    The pair this exists for is ``player+finish`` against ``player``: the win
    condition is ``all Player on Finish``, so a Player that hides the Finish it is
    standing on makes the winning board indistinguishable from any other. That is
    exactly what shipped -- every object was a bare colour with no sprite, so the
    Player painted a solid block over everything -- see the header of
    data/puzzlescript_games/Minimalist.txt.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    # Block and Finish are on different collision layers, so `block+finish` is
    # representable -- but the one rule is the identity, so nothing can ever
    # create it, and this asserts no level ships it either, which is what keeps
    # it out of the audit below.
    for level in range(game.n_levels):
        game.set_level(level)
        for row in eng.grid:
            for cell in row:
                assert not (idx["block"] in cell and idx["finish"] in cell), \
                    "a level ships a Block on a Finish cell"

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    comps = [(), ("block",), ("finish",), ("player",), ("player", "finish")]
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
            if objs == ("player", "finish"):
                over = (f", {int((shots[objs] != shots[('player',)]).sum()):4d} "
                        f"px of Finish survive under the Player")
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
    sys.exit(MinimalistSolver.main())
