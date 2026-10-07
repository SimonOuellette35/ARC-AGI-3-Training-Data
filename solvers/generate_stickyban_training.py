"""Generate Phase-1 training data for the PuzzleScript game ps:stickyban
("Stickyban", Connorses / Loneship Games).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, the native model of them, the exhaustive distance field that plans
every level, the reports that prove the plans shortest and the labels exact, and
the render fix the game needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_stickyban",
      "levels": [
        {
          "level_id": 0,
          "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
          "actions":      [a_0, a_1, ..., a_{T-1}], # length T
        },
        ...
      ]
    }

``actions[0]`` is the RESET that produced ``obs[0]``; ``actions[i]`` for i>=1 is
the action that took the agent from ``obs[i-1]`` to ``obs[i]``. The recorded
index is the *screen* action (post rotation/flip remap), i.e. the button an agent
presses in the augmented view, so replaying the recorded actions reproduces the
recorded frames exactly. Every expert step carries the full set of
equally-optimal presses.

Level indices everywhere below are 0-BASED, one less than the game's own
"Level N" messages.


THE GAME: A SOKOBAN WHERE YOU ARE GLUED TO THE CRATES
=====================================================
Seven levels, ``All Target on Crate``: every dark-blue goal square must end with
an orange crate standing on it. One player, four arrow keys, no other verb.
Some levels ship SPARE crates (level 5 has six crates for two targets, level 6
four for one), so the puzzle is which crates, not all of them.

What makes it not a sokoban is one word in two rules -- the missing direction
prefix:

    [ moving Player | Crate ] -> [ moving Player | moving Crate ]
    [ moving Crate  | Crate ] -> [ moving Crate  | moving Crate  ]

A PuzzleScript rule with no direction prefix is expanded to all four, so
``| Crate`` means "a crate on ANY side", not "the crate ahead". So:

* every crate orthogonally adjacent to the player is dragged, whichever way you
  walk -- behind you, beside you, ahead of you;
* the drag propagates crate-to-crate in all four directions, so what actually
  moves is the whole orthogonally-CONNECTED COMPONENT of every crate you are
  touching.

You and the crates you touch are one rigid body. Walking left with a crate above
you and a crate below you takes both of them left (measured; ``--selfcheck``
replays exactly this). A four-crate blob crosses a room at one press per CELL
rather than per crate, and a crate you merely brush in passing comes with you.

THE BRAKE, AND WHEN IT CANCELS THE TURN
---------------------------------------
Four rules decide what a wall does to that body, and their ORDER is the whole
subtlety:

    [> player | wall] -> cancel                            (1)
    [> Crate | Wall] -> [ Crate wallHit | Wall ]            (4)
    [ Crate wallHit | Crate ] -> [ Crate wallHit | Crate wallHit ]   (5)
    [ moving Crate wallHit ] -> [ Crate wallHit ]           (6)
    [> player | wallHit] -> cancel                          (7)

`wallHit` is a transparent marker object on its own collision layer, created and
destroyed inside the turn (two ``late`` rules sweep it) -- it is not state the
board carries between presses.

Read together, one press in direction ``d`` is:

    1. the cell ahead of the player is a Wall  ->  CANCEL, nothing happens;
    2. otherwise the moving body is the player plus every connected component of
       crates that touches the player;
    3. any moving crate with a Wall in direction ``d``  ->  wallHit (4), which
       then floods its whole component (5) and freezes it (6). Rule 5 has no
       direction prefix either, so the freeze is the component, not the row: ONE
       member wedged against a wall costs you all of them;
    4. if the crate directly ahead of the player is one of the frozen ones ->
       CANCEL, nothing happens at all -- not even the other components move;
    5. otherwise the player and the unfrozen components move one cell.

Step 4 is why an ordinary sokoban push against a wall is a clean no-op, and step
5 is the mechanic worth naming:

**A BLOCKED COMPONENT DOES NOT STOP YOU -- IT LETS GO.** Rule 7 only cancels when
the wallHit crate is the one the player is walking INTO. A crate you are dragging
behind or beside you that jams against a wall simply stays there while you walk
on. That is the game's only way to put a crate down, and every level needs it.
Measured on level 2: standing between two crates with the upper one against the
``###`` overhang, ``right`` leaves the upper crate where it is and takes the
lower one with you.

Note also what rule 5 does NOT do: it floods through CRATES, never through the
player, so two components attached to opposite sides of the player are
independent. One can freeze while the other travels.

Everything above was MEASURED, not read off the .txt -- ``--selfcheck`` fuzzes
the model in this file against the real interpreter and prints a coverage
counter per branch, so "the model agrees" is only reported alongside proof that
the branch was reached.

Two more measured facts:

* **ACTION5 is dead.** No rule in the file mentions ``action``, and an action
  turn sets no directional force, so nothing can match. `directions` therefore
  branches on the four moves only. (The exploration prefix still presses it; a
  no-op press is honest recovery data.)
* **Nothing is created, destroyed, or transformed.** Crate count is invariant,
  the player cannot die, there is no ``restart`` and no lose condition, so no
  press can end an episode. The board state is exactly ``(player cell, set of
  crate cells)``.

WHY OFF-GRID NEVER COMES UP
---------------------------
Rules 1 and 4 both name the Wall OBJECT, so the grid EDGE is not a wall to them
-- a press off the edge would be a silent failure to move rather than a cancel,
and the model here does not implement that asymmetry. It does not have to:
`_Board` asserts at build time that the player's free component touches no grid
boundary cell, and all seven levels are fully enclosed (the stray background
cells outside the border on levels 3 and 6 are sealed pockets nothing reaches).
A level that ever broke that assertion would fail loudly rather than plan
against a wrong model.


A NATIVE MODEL, AND WHY
=======================
`eng.step` runs at **~500 steps/s** on this game's biggest board. That is the
[[entrepotphage]] test in `ps_astar`'s family notes and this game fails it: level
2 alone reaches 319888 states, which is ~1.3M interpreter steps to enumerate --
over 40 minutes, against 2.4 seconds for the model in `_Board`. So the
interpreter is demoted to the CERTIFIER: it never plans, and ``--plans`` replays
every plan it produced press by press and takes the win from
``eng.check_win()``.

`_Board` is bitmask-per-cell throughout (crates are one Python int, not a
frozenset) and its `step` is a direct transcription of the five numbered steps
above. `--selfcheck` is what makes that transcription trustworthy: a seeded
random walk on every level, compared against the interpreter press by press,
with a counter per model branch so the report shows that the walk reached the
wall-cancel, the crate-cancel, the multi-component drag and -- the one that is
easy to get wrong -- the PARTIAL freeze where one component lets go and another
travels.

EXHAUSTIVE, NOT HEURISTIC
-------------------------
With a model this cheap the right search is no search: `_Field` enumerates every
state reachable from a level's start and runs a backward BFS from the winning
edges, giving the exact distance-to-win of every state. All seven levels
together enumerate in about **5 seconds**. That buys three things a heuristic
search cannot have:

* plans that are provably SHORTEST -- no weight, no heuristic, no node cap that
  could quietly bite: 11, 14, 29, 14, 37, 41 and 26 presses;
* EXACT optimal-action sets, read straight off the distance field
  (``dist(succ) == dist - 1``) rather than inferred from macro structure;
* the DEAD-END map. Five of the seven levels can be spoiled, two of them badly:
  level 4 reaches 1629 states of which only 538 can still win, and level 5 loses
  65% of its space (30638 live of 88383). Nothing marks a spoiled board -- no
  death, no restart, no visual tell, just a crate parked somewhere it can never
  be dragged out of -- which is exactly what the RESET recovery arc is for.

The exhaustive field is also what ``--ties`` checks the labels against, and it
does it by PROVING the field rather than re-solving: `_certify` verifies the
Bellman conditions over every reachable state, which is a proof that the
distances are the true distances (and so that the plans are shortest and the
tie sets complete). A re-solve would cost a fresh 320k-state enumeration per
press on level 2 and would still only speak for the states one plan visits.

Augmentation
------------
``Stickyban`` is in `PuzzleScriptAdapter._FLIP_GAMES`, so a (seed, level) is
presented at one of 16 orientations rather than 4 -- 112 presentations for the
whole game instead of 28, which matters on a game with seven levels. The
argument is written out beside that entry; ``--symmetry`` is the measurement,
and it is not a formality here. Two of this game's rules carry NO direction
prefix at all, so they expand over all four directions; that makes them MORE
symmetric than a relative ``>``, but ps:gobble_rush's chirality hid inside
exactly that kind of reasoning, so all 16 presentations are driven through the
adapter -- every plan plus a seeded 200-press random walk -- and every frame is
required to be the exact transform of the unaugmented one.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the plan
replays to a guaranteed WIN. Given the dead-end counts above the reset is doing
real work on this game rather than decorating the trajectory.

Rendering
---------
The Target sprite is redrawn in ``data/puzzlescript_games/Stickyban.txt``
(the comment at the top of that file has the detail; ``--audit`` is the check).
As shipped, Target's dark-blue ring occupied precisely the nine pixels the
Player's sprite paints over, so ``player on target`` rendered identically to a
player on bare floor -- and the player crosses its own goal squares constantly,
because a target is ordinary floor to it. Every such frame erased a goal from
the picture. Target now also paints the four CORNERS, which the Player leaves
transparent, and the Crate's hollow middle keeps showing the ring as before.

CLI
---
    --plans      per-level size, plan length, tie coverage; every plan replayed
                 through the REAL interpreter
    --bfs        the exhaustive reachable-state report (winnability + dead ends)
    --ties       prove each distance field (Bellman conditions over every
                 reachable state), re-derive every optimal-action label from it,
                 and cross-check the model against the interpreter along the plan
    --selfcheck  fuzz the native model against the interpreter, with a coverage
                 counter per model branch
    --symmetry   drive all 16 presentations and require exact transforms -- the
                 evidence for the `_FLIP_GAMES` entry
    --audit      assert every cell composition renders distinctly, at every
                 board shape
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from collections import Counter, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                       # noqa: E402
    Plan, PSAStarSolver, PSExpert, restore, snapshot,
)

GAME_NAME = "Stickyban"

#: The four presses, and the plan's tie-break order -- fixed, so a re-derived
#: plan is byte-identical across processes. ACTION5 is not here: no rule in the
#: game reads it (see the module docstring).
_DIRS = ("up", "down", "left", "right")

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry plus the one-press transition.

    A state is ``(player_cell, crate_mask)``: the player as a flat ``r * W + c``
    index and the crates as a bitmask over the same indexing. Bitmasks rather
    than frozensets because the enumeration holds hundreds of thousands of
    states at once and an int is ~40 bytes where a frozenset of six cells is
    ~700 (the ps:entrepotphage lesson in the `ps_astar` family notes).

    Nothing here is created or destroyed: the crate COUNT is invariant, the
    player cannot die and there is no restart, so this pair is the entire board.
    """

    __slots__ = ("height", "width", "walls", "targets", "start", "_nb",
                 "_around", "_cells")

    def __init__(self, eng, ids):
        wall_id, target_id, crate_id, player_id = ids
        self.height, self.width = eng.height, eng.width
        n = self.height * self.width
        self._cells = n
        self.walls = 0
        self.targets = 0
        crates = 0
        player = None
        for r in range(self.height):
            for c in range(self.width):
                cell = eng.grid[r][c]
                bit = 1 << (r * self.width + c)
                if wall_id in cell:
                    self.walls |= bit
                if target_id in cell:
                    self.targets |= bit
                if crate_id in cell:
                    crates |= bit
                if player_id in cell:
                    player = r * self.width + c
        if player is None:
            raise ValueError("Stickyban level has no player")
        self.start = (player, crates)

        # Per-direction "cell one step that way", -1 off the grid, and the
        # orthogonal-neighbour mask used for both the drag closure and the
        # wallHit flood.
        self._nb = {}
        for d, (dr, dc) in _DELTA.items():
            table = [-1] * n
            for r in range(self.height):
                for c in range(self.width):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.height and 0 <= cc < self.width:
                        table[r * self.width + c] = rr * self.width + cc
            self._nb[d] = table
        self._around = [0] * n
        for i in range(n):
            m = 0
            for d in _DIRS:
                j = self._nb[d][i]
                if j >= 0:
                    m |= 1 << j
            self._around[i] = m

        self._assert_enclosed()

    # -- invariants ----------------------------------------------------------
    def _assert_enclosed(self) -> None:
        """The player's free component must not touch the grid boundary.

        Rules 1 and 4 both name the Wall OBJECT, so the grid EDGE brakes
        differently from a wall -- a press off the edge fails to move instead of
        cancelling, and a crate shoved off the edge stops without handing its
        component a wallHit. `step` implements the WALL semantics for both and
        would be wrong at an open edge. Every shipped level is fully enclosed,
        so this never fires; it exists so that a level which is not fails loudly
        here rather than planning against a model of a different game."""
        seen = {self.start[0]}
        queue = deque(seen)
        while queue:
            i = queue.popleft()
            r, c = divmod(i, self.width)
            if r in (0, self.height - 1) or c in (0, self.width - 1):
                raise ValueError(
                    f"Stickyban: free cell ({r}, {c}) touches the grid edge; "
                    "_Board.step models walls, not open boundaries")
            m = self._around[i] & ~self.walls
            while m:
                bit = m & -m
                m ^= bit
                j = bit.bit_length() - 1
                if j not in seen:
                    seen.add(j)
                    queue.append(j)

    # -- mechanics -----------------------------------------------------------
    def _closure(self, seed: int, crates: int) -> int:
        """The crates orthogonally connected to ``seed``, ``seed`` included.

        The same flood serves both un-prefixed rules: the drag
        (``[moving Crate|Crate]``) and the wallHit spread
        (``[Crate wallHit|Crate]``). Neither carries a direction, so both are
        the connected COMPONENT rather than the row."""
        comp = seed
        frontier = seed
        around = self._around
        while frontier:
            reach = 0
            x = frontier
            while x:
                bit = x & -x
                x ^= bit
                reach |= around[bit.bit_length() - 1]
            frontier = reach & crates & ~comp
            comp |= frontier
        return comp

    def step(self, state, direction):
        """One press. Returns the new state, or ``state`` itself when the turn
        cancels (which is also what a press that changes nothing returns, and
        the two are indistinguishable to a player)."""
        player, crates = state
        nb = self._nb[direction]
        ahead = nb[player]
        if ahead < 0 or (self.walls >> ahead) & 1:
            return state                                    # rule 1: cancel

        touching = self._around[player] & crates
        if not touching:
            return (ahead, crates)                          # nothing is glued on

        moving = self._closure(touching, crates)            # rules 2 + 3

        blocked = 0
        x = moving
        while x:
            bit = x & -x
            x ^= bit
            target = nb[bit.bit_length() - 1]
            if target < 0 or (self.walls >> target) & 1:
                blocked |= bit                              # rule 4: wallHit
        frozen = self._closure(blocked, crates) if blocked else 0   # rules 5 + 6

        if (frozen >> ahead) & 1:
            return state                                    # rule 7: cancel

        go = moving & ~frozen
        if not go:
            return (ahead, crates)                          # let go of all of it
        # Vacate first, then land: a component moving one cell always overlaps
        # itself, and a destination can never be a crate that is staying put
        # (that crate would be adjacent, hence in the same component).
        rest = crates & ~go
        x = go
        while x:
            bit = x & -x
            x ^= bit
            rest |= 1 << nb[bit.bit_length() - 1]
        return (ahead, rest)

    def won(self, state) -> bool:
        return state[1] & self.targets == self.targets


# ---------------------------------------------------------------------------
# The exhaustive distance field
# ---------------------------------------------------------------------------

class _Field:
    """Every state reachable from one board, with its exact distance to a win.

    Forward BFS over `_Board.step` for the reachable set, then a backward BFS
    from the winning edges for the distance-to-win. Because the forward pass
    terminates having generated everything the model admits, the field is exact:
    ``plan`` is provably shortest, ``optimal`` is the complete list of shortest
    continuations, and a start absent from ``dist`` is a PROOF of unwinnability
    rather than a search giving up.

    A press that leaves the state unchanged (a cancel, or a walk into a wall) is
    dropped rather than stored as a self-loop: it wastes a move to reach the
    state it started from, so no shortest path contains one and no optimal set
    should ever name one.
    """

    __slots__ = ("board", "succ", "dist", "start", "capped")

    def __init__(self, board: _Board, start=None, node_cap: int = 4_000_000):
        self.board = board
        self.start = board.start if start is None else start
        self.succ: dict = {}
        self.capped = False
        seen = {self.start}
        queue = deque([self.start])
        won = board.won
        step = board.step
        while queue:
            state = queue.popleft()
            edges = {}
            for direction in _DIRS:
                nxt = step(state, direction)
                if nxt == state:
                    continue
                edges[direction] = nxt
                if nxt not in seen:
                    if len(seen) >= node_cap:
                        self.capped = True
                        self.succ = {}
                        self.dist = {}
                        return
                    seen.add(nxt)
                    if not won(nxt):        # a win is a sink, never expanded
                        queue.append(nxt)
            self.succ[state] = edges
        self.dist = self._backward()

    def _backward(self) -> dict:
        """Presses-to-win for every state, by backward BFS from the winning
        edges. States absent from the result cannot win at all."""
        won = self.board.won
        rev: dict = {}
        dist: dict = {}
        frontier = []
        for state, edges in self.succ.items():
            for nxt in edges.values():
                if won(nxt):
                    if state not in dist:
                        dist[state] = 1
                        frontier.append(state)
                else:
                    rev.setdefault(nxt, []).append(state)
        queue = deque(frontier)
        while queue:
            state = queue.popleft()
            for prev in rev.get(state, ()):
                if prev not in dist:
                    dist[prev] = dist[state] + 1
                    queue.append(prev)
        return dist

    def cost(self, state, direction):
        """Presses to win if ``direction`` is pressed at ``state``, or None if
        that press is unavailable or leads nowhere."""
        nxt = self.succ[state].get(direction)
        if nxt is None:
            return None
        if self.board.won(nxt):
            return 1
        rest = self.dist.get(nxt)
        return None if rest is None else rest + 1

    def optimal(self, state) -> list:
        """Every press on a shortest path from ``state``."""
        best = self.dist.get(state)
        if best is None:
            return []
        return [d for d in _DIRS if self.cost(state, d) == best]

    def plan(self) -> "Plan | None":
        """The shortest press sequence from the start, with the exact optimal
        SET at every step. Ties break on `_DIRS` order."""
        if self.start not in self.dist:
            return None
        presses, optsets = [], []
        state = self.start
        while True:
            best = self.optimal(state)
            presses.append(best[0])
            optsets.append(best)
            state = self.succ[state][best[0]]
            if self.board.won(state):
                return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class StickybanExpert(PSExpert):
    """`PSExpert`'s plan memo, disk cache and snapshot discipline around the
    exhaustive `_Field`.

    `_search` is replaced the way ps:dotsnake and ps:escaping_limbo replace it:
    the base class keeps the memo, the restore discipline and the level scoping,
    and only the strategy underneath changes. Here the strategy reads the board
    off the engine ONCE, enumerates it natively, and never steps the interpreter
    again -- see the module docstring for the 500 steps/s that forces it.

    `heuristic` therefore asserts rather than returning a number nothing would
    use, and `_key` is inherited (every non-background cell), which is exact and
    canonical ACROSS levels here because the walls are in the key and no two
    levels share a wall layout -- so no ``scope_by_level``.
    """

    directions = list(_DIRS)

    #: The searches are the whole cost of generation and they are
    #: seed-independent, so without a file on disk every `parallelize_generator`
    #: shard re-derives all seven. Measured: 5 episodes cost 10.3s / 325 MB peak
    #: on a cold cache against 1s/episode warm, and the two runs are
    #: byte-identical. A stored entry carries the start layout it was solved
    #: from, so an edited level is a miss rather than a wrong plan.
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "stickyban_plans.json")

    #: Runaway guard on the enumeration, not a tuning dial: the largest level in
    #: the game reaches 320k states.
    field_cap: int = 4_000_000

    def setup(self) -> None:
        idx = self.g.obj_name_to_idx
        self.ids = (idx["wall"], idx["target"], idx["crate"], idx["player"])

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "StickybanExpert enumerates the reachable space; heuristic is unused")

    def board(self, eng) -> _Board:
        """The native model of the engine's CURRENT grid."""
        return _Board(eng, self.ids)

    def field(self, eng) -> _Field:
        """The exhaustive distance field over the engine's current state."""
        return _Field(self.board(eng), node_cap=self.field_cap)

    def _search(self, eng) -> "Plan | None":
        field = self.field(eng)
        if field.capped:
            return None
        return field.plan()


class StickybanSolver(PSAStarSolver):
    game_id = "puzzlescript_stickyban"
    game_name = GAME_NAME
    expert_cls = StickybanExpert

    #: `games/ps:stickyban/ps:stickyban.py` is a plain passthrough -- it builds
    #: the adapter and nothing else, and the render fix is in the .txt, which
    #: both paths read. Set this if that wrapper ever grows a patch.
    game_module_id = ""

    #: Unused: `StickybanExpert._search` does not call `_astar`. Left at the base
    #: value so the constructor signature keeps working; `field_cap` is the knob
    #: that actually bounds the enumeration.
    node_cap = 400_000

    #: Room for the longest plan (41 presses) plus the RESET exploration prefix
    #: and the re-plan after it, and well under the adapter's own 200-step
    #: per-level budget.
    max_steps = 150

    #: Every level is solved exhaustively; nothing is skipped.
    skip_levels = frozenset()

    def prepare_expert(self, game, expert) -> None:
        """Enumerate every level before `discover_solvable` asks for it -- the
        same work either way, but it fills the disk cache in one pass and makes
        the startup cost visible as startup."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    """The solver, its adapter and an expert -- without planning anything.

    The reports below each want a different slice (``--audit`` needs no plan at
    all, ``--selfcheck`` needs only the model), so planning is left to whoever
    asks for it rather than paid on every entry point."""
    solver = StickybanSolver()
    game = solver.make_game(seed)
    return solver, game, StickybanExpert(game, node_cap=solver.node_cap)


def _ascii(board: _Board, state) -> str:
    """A model state as ASCII, so a failing check can print the board it failed
    on rather than a pair of integers."""
    player, crates = state
    out = []
    for r in range(board.height):
        line = ""
        for c in range(board.width):
            bit = 1 << (r * board.width + c)
            crate = bool(crates & bit)
            target = bool(board.targets & bit)
            if board.walls & bit:
                ch = "#"
            elif r * board.width + c == player:
                ch = "p" if target else "P"
            elif crate:
                ch = "@" if target else "*"
            else:
                ch = "o" if target else "."
            line += ch
        out.append("    " + line)
    return "\n".join(out)


def _state_of(eng, ids):
    """The model state read back out of the INTERPRETER's grid."""
    wall_id, target_id, crate_id, player_id = ids
    player, crates = None, 0
    for r in range(eng.height):
        for c in range(eng.width):
            cell = eng.grid[r][c]
            if player_id in cell:
                player = r * eng.width + c
            if crate_id in cell:
                crates |= 1 << (r * eng.width + c)
    return (player, crates)


def _plans(verbose: bool = True) -> int:
    """Every level's plan, replayed through the REAL interpreter.

    The end-to-end test of the model, the enumeration and the tie labelling at
    once: the plan is derived natively, then stepped through the interpreter
    press by press, and the win is the interpreter's own ``check_win``. A plan
    that the model believes in and the interpreter does not is caught here.

    On a cold `plan_cache_path` this pays for every enumeration (~5s for all
    seven); afterwards it is instant.
    """
    solver, game, expert = _new()
    eng = game._engine
    idx = game._game.obj_name_to_idx
    total = ties = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        crates = sum(1 for row in eng.grid for cell in row if idx["crate"] in cell)
        targets = sum(1 for row in eng.grid for cell in row if idx["target"] in cell)
        t0 = time.time()
        plan = expert.plan(eng, level)
        dt = time.time() - t0
        head = (f"  L{level}: {eng.height:2d}x{eng.width:2d}  "
                f"{crates} crates / {targets} targets")
        if plan is None:
            bad += 1
            if verbose:
                print(f"{head}   UNWINNABLE (see --bfs)")
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tie_steps = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        ties += tie_steps
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        if verbose:
            print(f"{head}  {len(plan):3d} presses  interpreter win={won}  "
                  f"(budget {game._max_steps}, {room})  "
                  f"{tie_steps:3d} steps with a tie set  {dt:6.2f}s")
    if verbose:
        print(f"  {total} presses, {ties} of them with a second equally-right "
              f"answer")
        print("  every plan reaches a WIN in the interpreter" if not bad
              else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _bfs(verbose: bool = True) -> int:
    """The exhaustive reachable-state report -- and therefore the proof that the
    plans are shortest: the enumeration terminates having generated every state
    the model admits, so the backward distance field is exact.

    It is also the DEAD-END report, and this game has plenty. A crate is only
    movable while the player can reach a cell beside it and drag it somewhere,
    so a crate parked in a corner, or welded into a component that no longer
    fits anywhere useful, is gone for good. Nothing on screen says so.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        t0 = time.time()
        field = expert.field(eng)
        dt = time.time() - t0
        if field.capped:
            print(f"  L{level}: node cap hit -- this report is not a proof")
            bad += 1
            continue
        live = len(field.dist)
        reach = len(field.succ)
        wins = sum(1 for e in field.succ.values()
                   if any(field.board.won(n) for n in e.values()))
        found = field.plan()
        verdict = (f"shortest {len(found)} presses" if found is not None
                   else "UNWINNABLE (no winning edge is reachable)")
        bad += found is None
        if verbose:
            print(f"  L{level}: {reach:7d} reachable states "
                  f"({live:7d} can still win, {reach - live:6d} spoiled, "
                  f"{wins:5d} touch a win), {dt:5.2f}s -- {verdict}")
    return 0 if not bad else 1


def _certify(field: _Field) -> "list[str]":
    """Check the Bellman optimality conditions on a whole distance field.

    This is a PROOF that ``field.dist`` is the true distance-to-win function,
    and it is the reason ``--ties`` does not re-solve. The obvious independent
    check -- re-enumerate from every successor of every plan step -- costs a
    fresh 320k-state enumeration per press on level 2 (~5 minutes for that level
    alone) and still only says something about the states one plan happens to
    visit. Two linear passes over the field say something about ALL of them:

      (a) for every ``s`` in ``dist``:  ``dist(s) == min over its edges of
          (1 if the edge wins else 1 + dist(edge))``, that minimum being over
          finite terms only;
      (b) for every reachable ``s`` NOT in ``dist``: no edge of ``s`` wins and no
          edge of ``s`` lands in ``dist``.

    (b) makes the complement of ``dist`` closed and win-free, so every state
    outside ``dist`` provably cannot win. (a) makes ``dist`` a chain that drops
    by exactly one per press, so following any minimiser from ``s`` reaches a win
    in ``dist(s)`` presses, and an induction on the true distance gives the other
    inequality. Together: ``dist`` IS the distance-to-win, hence the plans are
    shortest and ``optimal`` is the complete set of shortest continuations.

    Returns the list of violations found (empty when the field is certified).
    """
    won = field.board.won
    dist = field.dist
    bad: list[str] = []
    for state, edges in field.succ.items():
        costs = []
        for nxt in edges.values():
            if won(nxt):
                costs.append(1)
            elif nxt in dist:
                costs.append(1 + dist[nxt])
        if state in dist:
            if not costs:
                bad.append(f"{state}: in dist but no edge reaches a win")
            elif min(costs) != dist[state]:
                bad.append(f"{state}: dist={dist[state]} but the best edge "
                           f"costs {min(costs)}")
        elif costs:
            bad.append(f"{state}: absent from dist yet an edge costs "
                       f"{min(costs)}")
    return bad


def _ties(verbose: bool = True) -> int:
    """Prove the distance field, then re-derive every optimal-action label from
    it -- and cross-check the model against the interpreter along the way.

    Three things happen per level:

    1. `_certify` verifies the Bellman conditions over the WHOLE field, which
       proves the distances (and therefore that the plan is shortest and the
       labels complete). See that function for why this replaces a re-solve.
    2. Every label on the plan is recomputed from the certified field and
       compared with what the `Plan` shipped.
    3. The plan is walked in the INTERPRETER, and at every step all four
       successors are read back out of its grid and compared with
       `_Board.step`. `--selfcheck` fuzzes the model on random boards; this
       pins it on exactly the boards the recording will visit.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        field = expert.field(eng)
        if field.capped:
            print(f"  L{level}: node cap hit -- nothing to certify")
            bad += 1
            continue
        violations = _certify(field)
        for v in violations[:5]:
            print(f"  L{level}: BELLMAN VIOLATION {v}")
        bad += len(violations)

        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: no plan (skipped)")
            bad += 1
            continue
        board = field.board
        state = field.start
        for pi, (taken, claimed) in enumerate(zip(plan, plan.optsets)):
            here = snapshot(eng)
            engine_state = _state_of(eng, expert.ids)
            if engine_state != state:
                print(f"  L{level} step {pi}: the interpreter is at a different "
                      f"board than the plan\n{_ascii(board, engine_state)}")
                bad += 1
            for direction in expert.directions:
                restore(eng, here)
                eng.step(direction)
                if _state_of(eng, expert.ids) != board.step(state, direction):
                    print(f"  L{level} step {pi} ({direction}): MODEL DISAGREES "
                          f"with the interpreter\n{_ascii(board, state)}")
                    bad += 1
            restore(eng, here)
            measured = field.optimal(state)
            if measured != list(claimed):
                print(f"  L{level} step {pi}: labelled {claimed}, "
                      f"field says {measured}\n{_ascii(board, state)}")
                bad += 1
            if not measured or taken != measured[0]:
                print(f"  L{level} step {pi}: took {taken}, not the tie-break "
                      f"winner {measured}")
                bad += 1
            eng.step(taken)
            state = board.step(state, taken)
        if not board.won(state) or not eng.check_win():
            print(f"  L{level}: the plan does not end in a win "
                  f"(model={board.won(state)}, interpreter={eng.check_win()})")
            bad += 1
        if verbose:
            ties = sum(1 for s in plan.optsets if len(s) > 1)
            print(f"  L{level}: {len(field.succ):7d} states certified, "
                  f"{len(plan):3d} labels re-derived, {ties:2d} with a tie set "
                  f"-- {'ok' if not violations else 'FIELD NOT CERTIFIED'}")
    print("  every distance field is certified and every label reproduces"
          if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _selfcheck(presses: int = 6000, seed: int = 0, verbose: bool = True) -> int:
    """Fuzz `_Board` against the real interpreter, and report BRANCH COVERAGE.

    A seeded random walk on every level, compared press by press: the model's
    successor against the state read back out of the interpreter's grid, and the
    model's win predicate against ``eng.check_win()``. Random rather than
    plan-driven on purpose -- a plan never wedges a crate in a corner, never
    drags a component it did not mean to and never presses into a wall, and
    those are exactly the transitions the model has to be right about.

    The coverage counter is the half that makes the "0 mismatches" line worth
    anything. It counts, per press, which branch of `_Board.step` ran, so the
    report shows that the walk actually reached:

        wall-cancel       rule 1: the cell ahead is a Wall
        free-walk         no crate is touching the player
        crate-cancel      rule 7: the crate ahead froze, so nothing moved
        detach-all        every dragged component froze; the player walked off
        partial-freeze    one component froze and another travelled -- the one
                          branch that is easy to get wrong, and the mechanic the
                          levels are built on
        push-ahead        the crate being entered is one of the movers
        drag-Ncomp        components attached to the player at once
        dragsize-N        crates moving at once (5 = "5 or more")
    """
    _solver, game, expert = _new()
    eng = game._engine
    ids = expert.ids
    rng = random.Random(seed)
    cover = Counter()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board(eng, ids)
        state = board.start
        if state != _state_of(eng, ids):
            print(f"  L{level}: START state disagrees with the interpreter")
            bad += 1
        for n in range(presses):
            direction = rng.choice(_DIRS)
            _classify(board, state, direction, cover)
            eng.step(direction)
            engine_state = _state_of(eng, ids)
            model_state = board.step(state, direction)
            if model_state != engine_state:
                print(f"  L{level} press {n} ({direction}): model disagrees\n"
                      f"    from:\n{_ascii(board, state)}\n"
                      f"    model:\n{_ascii(board, model_state)}\n"
                      f"    interpreter:\n{_ascii(board, engine_state)}")
                bad += 1
                if bad > 3:
                    return 1
            if eng.check_win() != board.won(engine_state):
                print(f"  L{level} press {n}: win predicates disagree")
                bad += 1
            state = engine_state
            if eng.check_win():
                game.set_level(level)
                state = board.start
        if verbose:
            print(f"  L{level}: {presses} presses agree")
    if verbose:
        print("  branch coverage over the whole fuzz:")
        for key in sorted(cover):
            print(f"    {key:20s} {cover[key]:7d}")
        missing = [k for k in ("wall-cancel", "free-walk", "crate-cancel",
                               "detach-all", "partial-freeze", "push-ahead")
                   if not cover[k]]
        if missing:
            print(f"  NOT EXERCISED: {missing} -- the agreement above says "
                  "nothing about these branches")
            bad += len(missing)
    # ACTION5 is claimed dead in the docstring; measure it rather than assert it.
    for level in range(game.n_levels):
        game.set_level(level)
        before = snapshot(eng)
        eng.step("action")
        if snapshot(eng) != before:
            print(f"  L{level}: ACTION5 changed the board -- `directions` is wrong")
            bad += 1
    print("  model matches the interpreter on every press"
          if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _classify(board: _Board, state, direction, cover: Counter) -> None:
    """Which branch of `_Board.step` this press will take. A duplicate of the
    step's control flow on purpose -- instrumenting `step` itself would put a
    counter in the enumeration's inner loop, which runs millions of times."""
    player, crates = state
    nb = board._nb[direction]
    ahead = nb[player]
    if ahead < 0 or (board.walls >> ahead) & 1:
        cover["wall-cancel"] += 1
        return
    touching = board._around[player] & crates
    if not touching:
        cover["free-walk"] += 1
        return
    moving = board._closure(touching, crates)
    seen, comps = 0, 0
    x = touching
    while x:
        bit = x & -x
        x ^= bit
        if bit & seen:
            continue
        seen |= board._closure(bit, crates)
        comps += 1
    cover[f"drag-{comps}comp"] += 1
    cover[f"dragsize-{min(bin(moving).count('1'), 5)}"] += 1
    blocked = 0
    x = moving
    while x:
        bit = x & -x
        x ^= bit
        target = nb[bit.bit_length() - 1]
        if target < 0 or (board.walls >> target) & 1:
            blocked |= bit
    frozen = board._closure(blocked, crates) if blocked else 0
    if (frozen >> ahead) & 1:
        cover["crate-cancel"] += 1
        return
    if frozen and frozen == moving:
        cover["detach-all"] += 1
    elif frozen:
        cover["partial-freeze"] += 1
    if (crates >> ahead) & 1:
        cover["push-ahead"] += 1


def _symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for putting the game in
    `PuzzleScriptAdapter._FLIP_GAMES` (the rotation is mandatory and is checked
    here too). The structural argument is written out beside that entry, and it
    is a good one -- the two un-prefixed rules compute a 4-neighbourhood closure,
    which every element of the dihedral group maps to the corresponding closure,
    and everything that moves in a turn moves by the same vector so no two
    bodies can contest a cell. But ps:gobble_rush's chirality hid inside exactly
    that kind of argument, so it is measured rather than asserted.

    Both the PLANS and a seeded random walk are replayed. The walk is the half
    that matters here: a plan never wedges a crate against a wall, never drags a
    component it did not mean to and never presses the unbound ACTION key, and
    the partial-freeze branch (one component lets go, another travels) is the
    one whose mirror image would be worth doubting.
    """
    import numpy as np

    from arcengine import ActionInput, GameState
    from solvers.base_solver import _ID_TO_GAMEACTION
    from solvers.common.ps_astar import screen_action
    from utils.rotation import inverse_remap_action_full

    solver, game, expert = _new()
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    def drive(g, level, presses):
        g.set_level(level)
        out = [np.asarray(g._current_frame)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame else g._current_frame))
        return out

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    ref, seen, bad = {}, set(), 0
    for seed in range(400):
        if len(seen) == 16 and seed > 40:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"stickyban:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (
                    ("plan", [screen_action(d, *k) for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *k) for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"    seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[key] = frames
                    continue
                if any(not np.array_equal(transform(a, *k), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"    seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
        print("  every presentation is an exact transform of the unaugmented one"
              if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _audit(verbose: bool = True) -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every board shape.

    Whole 64x64 frames of UNIFORM boards are compared rather than one cell out of
    a mixed board: `_render_frame` centre-pads a non-square board, so indexing a
    cell by ``cell_px * r`` silently reads the wrong pixels (the ps:explod
    lesson). Two uniform boards render identically iff their cells do.

    Seven compositions, which is all a board of this game can hold: nothing is
    ever created, destroyed or transformed, and `wallHit` is swept by two
    ``late`` rules before the frame is drawn (it also has a ``transparent``
    sprite, so it could not draw anything anyway).

    ``player_on_target`` is why the Target sprite is redrawn in the .txt: as
    shipped it collided with ``player`` at every board shape, which is exactly
    what this report prints if the sprite is reverted.
    """
    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, _expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    comps = {
        "floor": (),
        "wall": ("wall",),
        "target": ("target",),
        "player": ("player",),
        "player_on_target": ("target", "player"),
        "crate": ("crate",),
        "crate_on_target": ("target", "crate"),
    }

    shapes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        shapes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(shapes.items()):
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        note = "OK" if not clashes else f"IDENTICAL {clashes}"
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(comps)} "
                  f"compositions -- {note}")
    print("  audit clean" if not bad
          else f"  AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--bfs" in sys.argv:
        sys.exit(_bfs())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--selfcheck" in sys.argv:
        sys.exit(_selfcheck())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(StickybanSolver.main())
