"""Generate Phase-1 training data for the PuzzleScript game ps:modality
("Modality" by Sean Barrett).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, the native model of the turn, the exact distance-to-win field
over the WHOLE reachable space, and the optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_modality",
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
A sokoban with no walls at all. Every cell is floor, and the whole puzzle is the
floor's COLOUR: each cell is White, Black or Grey, and those three tiles gate
both walking and pushing. Four rules, and the last two are ``CANCEL``:

    [ >  Player Nonblack | Crate Nonblack ] -> [  > Player Nonblack | > Crate Nonblack ]
    [ >  Player Black    | Crate Black    ] -> [  > Player Black    | > Crate Black    ]
    [ > Player White | Black ] -> CANCEL
    [ > Player Black | White ] -> CANCEL

with ``Nonwhite = Black or Grey`` and ``Nonblack = White or Grey``. Measured
against the interpreter (``--selfcheck``), that comes to:

  * **Walking.** White<->Black is the only forbidden step. Grey is the bridge:
    W->G, G->W, B->G, G->B and every same-colour step are all legal. So the board
    is two half-worlds welded together at the grey cells, and a level is usually
    "get to the other colour without stranding the crate on this one".
  * **Pushing.** A crate moves only when the pusher and the crate agree about
    BLACK: ``(player tile is Black) == (crate tile is Black)``. That is subtly
    NOT the walking rule -- Grey is *Nonblack*, so it sides with White. A player
    on Grey can push a crate standing on White but is stopped dead by a crate
    standing on Black, and a player on Black cannot push a crate off a Grey tile.
    The asymmetry is the whole game: Grey is a free bridge for your feet and a
    White tile for your hands.
  * **The crate's DESTINATION is unrestricted.** Nothing tests the tile a crate
    lands on, so a legal push can shove a crate from Black onto White -- and that
    crate is now unpushable by the Black-side player who put it there. Most of
    the dead states in the field are exactly that.
  * **No chain pushes, no walls, no wait.** A crate with another crate behind it
    does not move (and neither does the player), the board edge stops everything,
    and ``noaction`` means the four arrows are the entire action space. A refused
    press is a total no-op: nothing else in the game ticks, so there is no way to
    pass a turn.
  * **The win is a snapshot** (``All Crate on Target``), not a latch.

Model + search
--------------
`_Board.step` re-implements one turn natively; ``--selfcheck`` drives ~130k
random presses through BOTH the interpreter and the model across all levels and
compares the player, the crates AND the win flag after every one. That is the
guard that lets the search trust it. The model is needed rather than merely
convenient: the interpreter runs at 2000-4000 presses/s here, so enumerating the
233k states of the eighteen levels on it would be minutes rather than the 1.7 s
the model takes.

The state is ``(player cell, sorted crate cells)`` -- the crates are
interchangeable (one Crate object, no per-crate rule, and the win condition asks
only that each of them stands on a target), so ordering them would multiply the
space for nothing. Every level's space is small enough to just enumerate (the
biggest is 55k states), so `_Board.enumerate` walks it once from the level start,
keeps the successor table, and runs one backward BFS from the winning states to
get the EXACT distance-to-win field. From there a plan is a descent, a tie set is
a lookup, and a state reached by an exploration detour is answered from the same
table as the start state.

Unlike ps:miner_to_miner_empire, this game is mostly DEAD: 169k of those 233k
reachable states -- 73% -- can no longer win, because a crate shoved onto the
wrong colour, or into an edge, is stranded for good and nothing in the game can
undo it. ``--plans`` prints the count per level. That does not cost the epsilon
detours, because `record_level` only keeps a detour whose successor still has a
plan, and with an exact field that test is a dictionary lookup rather than a
search -- which is what makes a non-zero ``epsilon`` affordable in a game where
two thirds of the random presses available at any state are fatal.

Optimal-action targets and recovery
-----------------------------------
`optimal_for` reads the LIVE engine state and returns every press that lowers the
exact distance -- true tie sets, not a reordering heuristic. Walking to a push
cell is order-free (no rule fires on a bare move, so any interleaving of the two
axes leaves an identical board), so labelling one arbitrary interleaving as the
single right answer would train a coin flip the policy cannot win.

Levels
------
The game ships three boards (3x5, 5x5, 6x6), each with ONE crate and one target,
solved in 6, 33 and 66 presses. Three boards is not a corpus -- 16 presentations
each and nothing in them a policy could not memorise outright -- so fifteen more
are authored after them in data/puzzlescript_games/Modality.txt (4x4 to 7x7, one
to three crates, 10 to 50 presses); see the block comment there for how they were
built and why, including the test that each one is hard because of the MODALITY
rather than because of the sokoban. Every level, shipped or authored, is proved
solvable AND proved shortest by the field in this file, and ``--plans`` replays
each plan through the interpreter and requires a WIN.

Rendering
---------
The OBJECTS block is rewritten (see the sprite contract at the top of
data/puzzlescript_games/Modality.txt). The mechanic is the floor COLOUR under
every piece, so the two bodies had to become hollow and the Target had to move
out of their middles: as shipped, the Player was opaque over the whole of the
Target's ring, so a player standing on the goal hid the goal completely, and a
crate parked on a target left exactly ONE pixel of floor showing -- i.e. the
tile colour that decides whether that crate can ever be pushed again was
unreadable on the frame the game is about. ``--audit`` is the check, and it
compares WHOLE frames rather than cell crops (`_render_frame` upscales the board
to fill 64x64 and letterboxes it, so an arithmetic cell crop reads the wrong
window).

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (0..3) and, since
``Modality`` is in `PuzzleScriptAdapter._FLIP_GAMES`, an independent horizontal
and vertical flip -- 16 presentations of each level. The flips are measured
rather than argued: ``--symmetry`` rebuilds every level's LAYOUT under all 8
transforms, reloads it into the interpreter and replays both the level's own plan
and a seeded random walk with the presses transformed; every piece must land
where the transform says. (The rules here name no absolute direction and there is
no gravity, and every sprite is drawn as a shape invariant under all eight
symmetries of the square, so nothing in the frame can betray the presentation.)

Usage (run from the repo root):
    python solvers/generate_modality_training.py --episodes 200 \
        --out data/training_multi_level/modality
    python solvers/generate_modality_training.py --selfcheck
    python solvers/generate_modality_training.py --plans
    python solvers/generate_modality_training.py --verify
    python solvers/generate_modality_training.py --audit
    python solvers/generate_modality_training.py --symmetry
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

GAME_NAME = "Modality"

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The action space the search branches on. The game declares ``noaction``, so no
#: rule can read the ACTION key -- but the adapter still offers it as a fifth
#: button (``_available_actions`` is ``[1..5]``), so a live agent has it and the
#: exploration prefix presses it. `selfcheck` measures that it is a pure no-op
#: against the interpreter; the search leaves it out because a press that cannot
#: change the board is never a useful alternative.
_DIRS = ("up", "down", "left", "right")

#: Floor modality codes. Only the BLACK/non-black split matters to the rules; the
#: three-way distinction exists because Grey behaves differently for feet (it
#: bridges to either side) than for hands (it counts as Nonblack, i.e. White).
_WHITE, _BLACK, _GREY = 0, 1, 2


# ---------------------------------------------------------------------------
# The level, natively: one turn, the whole state space, the distance field
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry plus its COMPLETE state space and the exact
    distance-to-win field over it.

    Cells are flat ``r * w + c`` ids. A state is ``(player cell, sorted tuple of
    crate cells)`` -- a SET of crates, because they are interchangeable: the game
    ships one Crate object, no rule distinguishes them and the win condition asks
    only that each of them stands on a target.

    `enumerate` is what everything else reads: one forward BFS from the level
    start collecting every reachable state and its four successors, then one
    backward BFS from the winning states over the predecessor map it inverts.
    `plan` is a descent of the field, `optimal` a lookup, and both answer from ANY
    reachable state -- which is what lets the recording take real detours and
    label the recovery.
    """

    __slots__ = ("h", "w", "mod", "targets", "start", "_ahead",
                 "states", "index", "succ", "dist")

    def __init__(self, h: int, w: int, mod, targets, player, crates):
        self.h, self.w = h, w
        self.mod = list(mod)                       # cell -> _WHITE/_BLACK/_GREY
        self.targets = frozenset(targets)
        self.start = (player, tuple(sorted(crates)))

        # ``_ahead[d][cell]`` is the cell one step along d, or -1 off the board.
        # There are no walls in this game -- the board edge is the only geometry.
        self._ahead = {}
        for d, (dr, dc) in _DELTA.items():
            table = [-1] * (h * w)
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        table[r * w + c] = nr * w + nc
            self._ahead[d] = table

        self.states: list | None = None
        self.index: dict = {}
        self.succ: list = []
        self.dist: list = []

    # -- one turn -------------------------------------------------------------
    def won(self, state) -> bool:
        """``All Crate on Target``."""
        return all(p in self.targets for p in state[1])

    def step(self, state, d: str):
        """One press, natively. Returns the new state (``state`` itself when the
        press is refused -- a refused press is a total no-op here).

        The two tests are deliberately different, and that difference IS the
        game (see the module docstring):

          * the CANCEL rules forbid stepping between White and Black, so a legal
            destination is same-colour or Grey on at least one side;
          * a push needs pusher and crate to AGREE ABOUT BLACK, and Grey counts
            as Nonblack there -- so Grey is a bridge for the feet and a White
            tile for the hands.
        """
        p, crates = state
        nxt = self._ahead[d][p]
        if nxt < 0:                                   # off the board
            return state
        mp, mn = self.mod[p], self.mod[nxt]
        if mp != mn and _GREY not in (mp, mn):        # White <-> Black: CANCEL
            return state
        if nxt in crates:
            if (mp == _BLACK) != (mn == _BLACK):      # neither push rule matches
                return state
            beyond = self._ahead[d][nxt]
            # A crate stopped by the edge or by another crate stops the player
            # too: the engine resolves the crate first, and the player then walks
            # into a cell that is still occupied. There are no chain pushes.
            if beyond < 0 or beyond in crates:
                return state
            moved = list(crates)
            moved[moved.index(nxt)] = beyond
            return (nxt, tuple(sorted(moved)))
        return (nxt, crates)

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
    def distance(self, state) -> int | None:
        """Presses to a win from ``state``, or None when it can no longer win.

        Unlike ps:miner_to_miner_empire this really does return None sometimes:
        a crate pushed onto the wrong colour, or into an edge, can be stranded
        for good."""
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

        Exact, straight off the field. A refused press maps the state to itself,
        whose distance is unchanged rather than one less, so no-ops are excluded
        for free."""
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

class ModalityExpert(PSExpert):
    """Exact planner over the enumerated state space (see `_Board`).

    `PSExpert` supplies the plan memo, the restore discipline and the level
    scoping; `_search` swaps in the field descent, so the interpreter is only
    ever stepped by the recorder -- which is also what verifies every plan, since
    a level is kept only when the engine reports WIN.
    """

    directions = list(_DIRS)

    #: `_key` is the pieces alone, canonical only WITHIN a level (the floor
    #: colours and the targets that complete the state are static per level but
    #: differ between levels).
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        self.white_ids = set(g.resolve_object_name("white"))
        self.black_ids = set(g.resolve_object_name("black"))
        self.grey_ids = set(g.resolve_object_name("grey"))
        self.target_ids = set(g.resolve_object_name("target"))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.player_ids = set(self.game._engine._player_indices)
        self._boards: dict[int | None, _Board] = {}
        self._cur: _Board | None = None

    # -- reading the engine ---------------------------------------------------
    def read(self, eng) -> tuple:
        """``(player cell, sorted crate cells)`` from the engine grid.

        Sorted because the crate tuple is a SET key -- `_Board.step` keeps it
        sorted and the whole index is built on that, so a read that returned the
        same crates in another order would miss the field entirely. (Row-major
        iteration already yields ascending ids; the sort states the invariant
        rather than relying on that.)"""
        w = len(eng.grid[0])
        player = -1
        crates = []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & self.player_ids:
                    player = r * w + c
                if cell & self.crate_ids:
                    crates.append(r * w + c)
        return (player, tuple(sorted(crates)))

    def board(self, eng, level: int | None) -> _Board:
        """The level's `_Board`, built (and enumerated) once and cached.

        The floor tiles and the targets are static -- no rule creates, destroys
        or recolours any of them -- so whichever state happens to build it
        describes the same level."""
        board = self._boards.get(level)
        if board is not None:
            return board
        h, w = len(eng.grid), len(eng.grid[0])
        mod = [None] * (h * w)
        targets = set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & self.white_ids:
                    mod[r * w + c] = _WHITE
                elif cell & self.black_ids:
                    mod[r * w + c] = _BLACK
                elif cell & self.grey_ids:
                    mod[r * w + c] = _GREY
                else:                                # pragma: no cover
                    raise AssertionError(f"level {level} cell {r},{c} has no floor")
                if cell & self.target_ids:
                    targets.add(r * w + c)
        player, crates = self.read(eng)
        board = _Board(h, w, mod, targets, player, crates)
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

class ModalitySolver(PSAStarSolver):
    game_id = "puzzlescript_modality"
    game_name = GAME_NAME
    expert_cls = ModalityExpert

    #: Unused -- `ModalityExpert._search` never calls the base A* -- but left at
    #: the family default so a future subclass that does is not silently starved.
    node_cap = 2_000_000
    weight = 1
    #: The adapter GAME_OVERs at 200 actions per level, and `set_level` (which
    #: the RESET prefix ends with) zeroes that counter, so this budget is the
    #: expert's alone. The longest shortest-plan in the game is 66 presses
    #: (level 2); measured over 40 seeds the epsilon detours push the worst
    #: level to 120, so the cap has a comfortable margin and no seed was ever
    #: retried.
    max_steps = 200

    #: Non-zero even though this game DOES have dead states, because the field is
    #: exact: `record_level` probes each candidate detour and keeps it only when
    #: the successor still has a plan, and with an enumerated space that test is a
    #: dict lookup rather than a search. So one press in eight is a random legal,
    #: still-winnable alternative, after which the expert re-plans from wherever
    #: it landed -- the taken action is the mistake and ``optimal`` is the
    #: recovery, which is the signal a policy needs after its own error. The RESET
    #: prefix still runs in front of it.
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
    solver = ModalitySolver()
    game, expert, _solvable = solver._ensure(0)
    return solver, game, expert


def selfcheck(trials: int = 60, steps: int = 120, verbose: bool = True) -> int:
    """Drive random presses through BOTH the interpreter and `_Board.step`,
    comparing the player, the crates AND the win flag after every one. This is
    the guard that lets the search trust the native model.

    Random walks are the right fuzz here because a shortest PLAN never presses
    into an edge, never shoves a crate onto the colour that strands it and rarely
    tries the pushes the two ``CANCEL`` rules refuse -- i.e. it exercises almost
    none of the mechanic (see ps:idols_to_the_burnt_god, where the certification
    pass passed on all 27 levels while the fuzz convicted the first press of
    every one of them). Every walk starts COLD, from the level's own start, with
    no warm-up press, so turn-one bookkeeping cannot hide.

    ``action`` is in the press alphabet even though the search excludes it: the
    claim that the adapter's fifth button is a pure no-op is exactly the kind of
    thing that should be measured against the interpreter rather than read off a
    ``noaction`` header.

    Returns the number of mismatches."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = ModalityExpert(game)
    alphabet = _DIRS + ("action",)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine, level)
        rng = random.Random(f"modality:selfcheck:{level}")
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
                if want != expert.read(eng) or board.won(want) != eng.check_win():
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


def _report() -> int:
    """Print each level's space, its shortest plan and the engine's verdict --
    the quick "is this game still fully solved" check."""
    _solver, game, expert = _levels()
    eng = game._engine
    dead_total = states_total = 0
    unsolved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        n_states, n_dead = board.enumerate()
        states_total += n_states
        dead_total += n_dead
        assert not eng.check_win(), f"level {level} starts already won"
        plan = board.plan(board.start)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            unsolved += 1
            continue
        for d in plan:
            eng.step(d)
        won = eng.check_win()                   # read BEFORE anything reloads
        unsolved += not won
        ties, state = 0, board.start
        for d in plan:
            ties += len(board.optimal(state))
            state = board.step(state, d)
        print(f"  L{level}: {eng.height}x{eng.width}  "
              f"{len(board.start[1])} crate(s)  {len(plan):3d} presses  "
              f"win={won}  {n_states:6d} states ({n_dead:6d} dead)  "
              f"{ties / len(plan):.2f} optimal presses/step")
    print(f"  total: {states_total} states, {dead_total} dead "
          f"(a crate shoved onto the wrong colour is unrecoverable)")
    return 1 if unsolved else 0


def _walk(board: _Board, plan) -> list:
    """The states ``plan`` passes through, starting at the level start."""
    out, state = [], board.start
    for d in plan:
        out.append(state)
        state = board.step(state, d)
    return out


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
        print(f"  L{level}: {len(board.states):6d} states relaxed, "
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

        def move(flat):
            tr, tc = cell(divmod(flat, hw[1]), hw)
            return tr * tw + tc

        for i, d in enumerate(presses):
            eng.step(dmap[d])
            rp, rc = ref[i + 1]
            want = (move(rp), tuple(sorted(move(x) for x in rc)))
            if expert.read(eng) != want:
                notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                break
    return notes


def _symmetry(walk: int = 400) -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an exact
    symmetry of the mechanic, which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation.

    The four rules name no absolute direction -- every one of them is a ``>``
    relative to the press -- so there is no per-direction rule block to hide a
    chirality in the way ps:gobble_rush and ps:lovendpieces did. What is left to
    measure is the interpreter's own resolution of a push, and that is done twice
    over, because a plan alone is a thin sample of it:

      * each level's own PLAN, which is the sequence the corpus actually records
        and the only one that reaches the win; but a shortest plan never presses
        into an edge, never tries a push the CANCEL rules refuse and never
        strands a crate;
      * a seeded RANDOM WALK per level, which does nothing else.

    Every press of both must leave the player and the crates exactly where the
    transform of the reference run put them."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        layout = g.levels[level]
        rng = random.Random(f"modality:symmetry:{level}")
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
    at every cell size the levels render at.

    Two things have to survive, and the shipped art lost both (see the sprite
    contract at the top of data/puzzlescript_games/Modality.txt):

      * **the floor colour under every body.** It is not scenery -- it is the
        whole mechanic. A crate parked on a target showed ONE pixel of floor,
        i.e. the tile colour that decides whether that crate can ever be pushed
        again was effectively unreadable on the frame the game is about.
      * **the target under every body.** The Player was opaque across the whole
        of the Target's ring, so standing on the goal erased it.

    So the audit is over the full product ``floor x {nothing, target} x {nothing,
    player, crate}`` -- 18 compositions -- not over objects one at a time.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    comps = [(f,) + t + b
             for f in ("white", "black", "grey")
             for t in ((), ("target",))
             for b in ((), ("player",), ("crate",))]
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
        print(f"{h}x{w} board (level{'s' if len(levels) > 1 else ''} "
              f"{', '.join(str(x) for x in levels)}), "
              f"cell {max(1, min(64 // h, 64 // w))}px")
        cells = h * w
        for objs in comps:
            # The two things that must never reach zero: how much of the FLOOR
            # still shows through whatever is stacked on it (the tile colour is
            # the mechanic), and how much of the TARGET still shows through the
            # body standing on it (the goal is the win condition).
            floor = int((shots[objs] == shots[(objs[0],)]).sum()) // cells
            note = ""
            if "target" in objs and len(objs) > 2:
                bare = (objs[0],) + objs[2:]
                note = (f", {int((shots[objs] != shots[bare]).sum()) // cells:3d} "
                        f"px/cell of target survive under the "
                        f"{objs[2]}")
            print(f"  {'+'.join(objs):22s} {floor:4d} px/cell of floor still "
                  f"visible{note}")
        for a, b in clashes:
            print(f"  IDENTICAL: {'+'.join(a)} == {'+'.join(b)}")
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
    sys.exit(ModalitySolver.main())
