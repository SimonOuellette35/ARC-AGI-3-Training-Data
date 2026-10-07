"""Generate Phase-1 training data for the PuzzleScript game
ps:stand_aside_everyone_i_take_large_steps ("Stand aside, everyone! I take
large steps!", ncrecc).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exhaustive distance FIELD that model is
searched with, the proof that every plan is shortest, the proof that the model
IS the interpreter, the proof that the SHIPPED level could not be won, and the
render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_stand_aside_everyone_i_take_large_steps",
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
exactly. Every expert step carries the full set of equally-optimal presses.

The game
--------
A sokoban whose player moves TWO cells per press. That one change is the whole
game: it turns "which crate do I shove where" into "can I even stand where I
need to stand", because a two-cell step conserves the parity of the row and the
column you are on.

Eight rules resolve a press, in file order, and each of them is a whole
mechanic. Writing ``c1 c2 c3`` for the first three cells ahead of the player
and ``Obstacle`` for Wall-or-Crate-or-Rock, they are (all MEASURED against the
interpreter -- ``--verify`` is the proof, not the .txt):

* ``[> Player|Wall]`` -- a wall you are touching CANCELS the press outright.
  Note it names Wall, not Obstacle: a rock or a crate you are touching does not
  cancel anything by itself, which is what makes the next four rules matter.
* ``[> Player|Obstacle|Obstacle]`` -- two obstacles in a row ahead cancel it
  too. This is what stops a crate flat against a wall, and what stops the
  player leaping over two crates.
* ``[> Player||Wall] -> [|Player|Wall]`` -- **the short step.** A wall two
  cells ahead shortens the press to ONE cell. It is the only cheap way to
  change the parity of your position, and it exists only next to walls.
* ``[> Player||Obstacle|Obstacle] -> [|Player|Obstacle|Obstacle]`` -- the same
  shortening when the two cells behind the gap are blocked by anything.
* ``[> Player|Crate||Obstacle] -> [|Player|Crate|Obstacle]`` -- **the
  cut-short push.** A crate you are touching with an obstacle three cells out
  is shoved ONE cell and you follow one cell.
* ``[> Player||Crate|] -> [||Player|Crate]`` -- **the gap push.** A crate TWO
  cells ahead is shoved one cell and you land where it stood. You can push
  something you are not touching.
* ``[> Player|Crate||] -> [||Player|Crate]`` -- **the two-cell push.** A crate
  you are touching is shoved TWO cells, and you take its old cell -- so the
  crate LEAPS over the square you are about to stand on.
* ``[> Player||] -> [||Player]`` -- otherwise, two cells. This is also how a
  ROCK is passed: no rule pushes a Rock, so with a free cell behind it the
  player steps straight OVER it. A crate in the same place would be shoved.

**And it is how a rock DIES.** That last rule is the only one that puts the
player on a cell the rules above have not proved empty, and the only thing
that can still be sitting there is a rock two cells ahead with a free cell in
between (a wall two ahead is the short step, a crate two ahead is the gap
push). Player and Rock share a collision layer, so the player simply replaces
it: an ordinary-looking press eats the rock, silently and permanently. It is
the one object in this game that can be destroyed, which is why the rocks are
part of the model's STATE and not of its scenery -- the bug that first showed
it up was a plan whose rock the interpreter had eaten three presses earlier.

Every rule's RHS strips the movement flag, so the interpreter's own movement
phase never runs and a press is exactly one of the eight outcomes above. The
one case no rule covers is a pattern that runs off the board (only two cells
exist ahead), which cannot arise on a level with a wall border -- `_Board.step`
models it anyway, as the engine's default one-cell move, so an edited level
cannot silently mean something else.

There is no ``restart``, no ``again``, no lose condition and no randomness:
every press is one frame, ACTION5 is read by no rule at all, and no press can
end an episode. A settled board is exactly ``(player cell, crate cells, the
rocks still standing)``.

Win: ``All Target on Crate`` **and** ``All Player on VictoryStand`` -- both, so
the level ends only when the crates are parked AND the player has walked to the
yellow stand afterwards. With zero targets the first half is vacuously true
(measured), which is what lets the tutorial levels teach one movement rule with
no crates on the board at all.

The shipped level could not be won
----------------------------------
ncrecc published one board, and its player start makes it impossible. Run
``--repair`` for the derivation; the short version is:

* the board has **38378** reachable states, and in not one of them are both
  targets covered -- so the crate half of the win alone is unreachable, never
  mind the player also standing on the stand. That is an exhaustive forward
  BFS, and it is a fact about the GAME rather than about this model because
  ``--repair`` also seats those states back onto the real interpreter and
  presses all five actions from each (``--repair full`` does every one of them:
  191890 presses, zero disagreements). The board is not in the .txt any more,
  so the report carries it as a constant and checks it in place;
* the reason is parity. A press moves the player two cells, so its row and
  column parities are conserved. The short step breaks one -- but it needs a
  wall two cells ahead, every wall on that board lies on an even row or an even
  column, and so every short step it offers runs even -> odd. The player starts
  odd/odd at (1,1) and the VictoryStand is even/even at (6,2). The remaining
  way to change parity is to shove a crate, since a push moves the player one
  cell, and the field says every shove reachable from that start strands a
  crate where no later push recovers it;
* ``--repair`` then re-derives the claim the other way, by SEARCH: it tries
  every relocation of the player, of each crate, of each target and of the
  stand, and every one-cell wall edit, and reports which of them can be won.
  Exactly eight repairs exist and all eight move the PLAYER -- to one of the
  eight cells that are even/even. Moving anything else, or editing any single
  wall, leaves the board unwinnable.

So `data/puzzlescript_games/Stand_aside,_everyone!_I_take_large_steps!.txt`
keeps that board as level 7 with the start moved from (1,1) to (8,2) -- the
repair with the longest plan of the eight (13 presses) -- and every wall, both
crates, both targets and the stand exactly where they were drawn. Sixteen
levels were authored around it (see that file's header): 2 to 44 presses, the
first six one rule each.

The native model, and why the interpreter is not searched directly
------------------------------------------------------------------
The seventeen levels have 126206 reachable boards between them, and at ~5200
interpreter steps a second enumerating those through the engine is minutes of
work and hundreds of thousands of grid snapshots; the same enumeration over
`_Board` takes 0.6 s. So the search runs on a native model of the bullets above
-- a state is ``(player cell, sorted crate cells, rock bitmask)`` -- and
``--verify`` is the receipt: it re-seats sampled model states onto the real
engine, presses all five actions, and compares the resulting board, the STATIC
scenery and the win flag against the model's answer. ``--verify full`` checks
every reachable state of every level instead of a sample; it has been run --
631095 presses over all 126187 playable boards, not one disagreement, 157 s.

Because the space is finite and small it is enumerated EXHAUSTIVELY rather than
searched: a forward BFS for the whole reachable component and a backward BFS
from the won boards give the exact distance-to-win of every state. That buys
the three things a heuristic search cannot:

* plans that are provably SHORTEST -- no heuristic, no weight, no node cap that
  could quietly bite;
* exact optimal-action SETS for free (``dist(succ) == dist - 1``) rather than
  inferred ones, so every expert step is labelled with every press that ties;
* a PROOF of which boards can still be won, which is what turns "the search
  gave up" into the dead-end counts ``--bfs`` reports -- and they are large.
  Level 7 is the extreme: 51493 boards reachable, **28** of them still
  winnable. A crate shoved one cell too far is usually the end of the level,
  there is no death, no restart and no visual tell, and that is exactly what
  the RESET recovery arc is for.

Rendering
---------
Two sprites are redrawn in the .txt (its header is the full account; ``--audit``
is the check, and it FAILS on the unpatched file). As shipped, Target was a 3x3
block and VictoryStand a plus, both drawn inside rows/cols 1-3 -- which is
precisely the window the Player sprite is opaque across. So **the win frame was
invisible**: "player on the VictoryStand", the board this game's win condition
is about, rendered pixel-identical to the player standing on bare floor, and so
did the player standing on either target. Both are framed squares now, painted
out to the sprite border where the Player is transparent and solid across the
middle where the Crate's ring is open. Target also moves off ``#336``, which
quantises to ARC 13 -- the Wall's own DARKBROWN.

Optimal-action sets
-------------------
``optsets[i]`` is every press ``d`` with ``dist(succ(state_i, d)) ==
dist(state_i) - 1``, read straight off the exhaustive field, so it is exact
rather than inferred. ``--ties`` re-derives every label independently, by
taking each candidate press and running a FRESH enumeration from the state it
lands in.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the plan
replays to a guaranteed WIN. The RESET earns its place here -- most of the
reachable space of most levels cannot be won, and a crate shoved into a corner
is permanent.

There is no ``plan_cache_path``: all seventeen fields together are well under a
second, which is cheaper than the staleness risk a cached entry would carry. It
is paid once per process and every later seed replays the memoized plans at its
own rotation.

CLI
---
    --plans      every level's plan, replayed through the interpreter
    --bfs        the exhaustive reachable-state report (winnable vs stranded)
    --ties       re-derive every optimal-action label by independent re-solve
    --repair     the proof that ncrecc's board is unwinnable, and the search
                 over every single-cell edit of it; ``--repair full`` presses
                 every one of its states through the interpreter too
    --verify     model vs. interpreter; ``--verify full`` checks every state
    --audit      assert every cell composition renders distinctly
    --symmetry   replay every level at all 16 presentations (the `_FLIP_GAMES`
                 evidence)
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                                # noqa: E402

from arcengine import ActionInput, GameState                      # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION                 # noqa: E402
from solvers.common.ps_astar import (                             # noqa: E402
    PSAStarSolver, PSExpert, Plan, screen_action,
)
from utils.rotation import inverse_remap_action_full              # noqa: E402

GAME_NAME = "Stand_aside,_everyone!_I_take_large_steps!"
GAME_ID = "puzzlescript_stand_aside_everyone_i_take_large_steps"

#: The four presses, in the order that breaks every tie -- fixed, so a plan
#: re-derived in another process is byte-identical.
DIRS = ("up", "down", "left", "right")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: ncrecc's board exactly as published, kept for ``--repair`` (the .txt now
#: holds it with the player start moved -- see the module docstring).
SHIPPED_LEVEL = """\
#########
#p#.....#
#.#.....#
#.#.o.*.#
#.#.....#
#...*...#
#.+.....#
#.....o.#
#...#...#
#...#...#
#########"""


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic, plus the exhaustive
    distance field over it.

    A state is ``(player, crates, rocks)``: a cell index ``r * w + c``, a
    SORTED TUPLE of them (canonical, hashable and cheap at the one-to-three
    crates these levels carry), and a BITMASK of the rocks still standing.

    **The rocks are in the state because a rock can be EATEN.** No rule pushes
    a Rock, so with a free cell in front of it the last rule steps the player
    two cells and LANDS on it -- and Player and Rock share a collision layer,
    so the rock is simply replaced. It is the only object in the game that can
    be destroyed, it is destroyed silently, and the mask only ever loses bits.
    A crate is safe from the same fate only because the two push rules always
    match first (`step` models the off-board case where they cannot anyway).

    Nothing else is dynamic: no rule creates or destroys a Wall, a Target, a
    VictoryStand or the Player, no rule moves a Rock, and the game has no
    hidden bookkeeping objects, no ``again`` and no randomness -- so a settled
    board is exactly this triple. ``--verify`` asserts the static half rather
    than trusting it: it compares the scenery after every press it makes, not
    only the pieces.
    """

    __slots__ = ("h", "w", "wall", "targets", "stand", "edge", "_fields")

    def __init__(self, h: int, w: int, walls, targets, stand):
        self.h, self.w = h, w
        self.wall = [False] * (h * w)
        for i in walls:
            self.wall[i] = True
        self.targets = tuple(sorted(targets))
        self.stand = stand
        # ``edge[i][di]`` is the neighbour of cell ``i`` in direction ``di``, or
        # -1 off the board. Precomputed once: the enumeration reads it a few
        # million times.
        self.edge = []
        for r in range(h):
            for c in range(w):
                row = []
                for d in DIRS:
                    dr, dc = _DELTA[d]
                    nr, nc = r + dr, c + dc
                    row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w
                               else -1)
                self.edge.append(tuple(row))
        #: ``[(seen, dist), ...]`` -- one exhaustive sweep per state it was
        #: asked from. See `field`.
        self._fields: list = []

    # -- the mechanic --------------------------------------------------------
    def step(self, state, di: int):
        """The state after pressing ``DIRS[di]``, or ``state`` itself if the
        press does nothing (a non-move: no shortest path contains one).

        The eight rules of the RULES section, in the order the interpreter
        resolves them, with ``c1 c2 c3`` the cells ahead. A rule whose pattern
        runs off the board cannot match, which is why every branch tests the
        cells it names -- on a bordered level the last branch is unreachable,
        and on an unbordered one it is the engine's own one-cell move.

        The last branch is also where a rock dies: it is the only one that
        puts the player on a cell the rules did not prove empty, and the only
        occupant that can still be there is a Rock (a wall two ahead is the
        short step, a crate two ahead is the gap push). It drops a crate the
        same way for the truncated pattern a borderless level would allow --
        which is what the layer replacement would really do, not a shortcut.
        """
        p, crates, rocks = state
        e = self.edge
        wall = self.wall
        c1 = e[p][di]
        c2 = e[c1][di] if c1 >= 0 else -1
        c3 = e[c2][di] if c2 >= 0 else -1

        def obstacle(i: int) -> bool:
            return wall[i] or (rocks >> i) & 1 or i in crates

        if c1 >= 0 and wall[c1]:                            # touching a wall
            return state
        if c1 >= 0 and c2 >= 0 and obstacle(c1) and obstacle(c2):
            return state                                    # two deep: refused
        if c1 >= 0 and c2 >= 0 and wall[c2]:                # the short step
            return (c1, crates, rocks)
        if c3 >= 0 and obstacle(c2) and obstacle(c3):       # short step again
            return (c1, crates, rocks)
        if c3 >= 0 and c1 in crates and obstacle(c3):       # cut-short push
            return (c1, _moved(crates, c1, c2), rocks)
        if c3 >= 0 and c2 in crates:                        # the gap push
            return (c2, _moved(crates, c2, c3), rocks)
        if c3 >= 0 and c1 in crates:                        # the two-cell push
            return (c2, _moved(crates, c1, c3), rocks)
        if c2 >= 0:                                         # two cells
            return (c2, tuple(c for c in crates if c != c2),
                    rocks & ~(1 << c2))
        if c1 >= 0 and not obstacle(c1):                    # off-board pattern
            return (c1, crates, rocks)
        return state

    def won(self, state) -> bool:
        """``All Target on Crate`` and ``All Player on VictoryStand``.

        The target half is a SUBSET test, so it is vacuously true on a level
        with no targets (measured against the interpreter -- the tutorial
        levels rely on it) and a level with a spare crate still answers."""
        p, crates, _rocks = state
        return p == self.stand and all(t in crates for t in self.targets)

    # -- the exhaustive field ------------------------------------------------
    def field(self, state):
        """``(seen, dist)`` for the component ``state`` lives in: every board
        reachable from it, and the exact presses-to-win of each board that can
        still be won. Boards in ``seen`` but not in ``dist`` are PROVEN dead.

        Sweep 1 is a forward BFS over `step`; a won board is terminal and is
        never expanded (the episode ends there). Sweep 2 is a backward BFS from
        the won boards over the reverse edges sweep 1 recorded. There is no node
        cap and no heuristic: the component is finite (finitely many placements
        of a player and a fixed number of crates) and small enough to hold whole
        -- see ``--bfs``.

        Cached per component: a later query is answered by whichever completed
        sweep already contains the state, and only a state no sweep has seen
        pays for another one. Everything the recorder asks about is reachable
        from the level start by definition, so in practice exactly one sweep is
        run per level."""
        for seen, dist in self._fields:
            if state in seen:
                return seen, dist
        ids = {state: 0}
        states = [state]
        rev: list = [[]]
        queue = deque([0])
        while queue:
            u = queue.popleft()
            here = states[u]
            if self.won(here):
                continue                      # a won board is terminal
            for di in range(4):
                nxt = self.step(here, di)
                if nxt == here:
                    continue
                v = ids.get(nxt)
                if v is None:
                    v = len(states)
                    ids[nxt] = v
                    states.append(nxt)
                    rev.append([])
                    queue.append(v)
                rev[v].append(u)
        depth = [-1] * len(states)
        queue = deque()
        for v, st in enumerate(states):
            if self.won(st):
                depth[v] = 0
                queue.append(v)
        while queue:
            v = queue.popleft()
            for u in rev[v]:
                if depth[u] < 0:
                    depth[u] = depth[v] + 1
                    queue.append(u)
        dist = {states[v]: d for v, d in enumerate(depth) if d >= 0}
        seen = set(ids)
        self._fields.append((seen, dist))
        return seen, dist

    def optimal(self, dist, state):
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``. Ties come out in ``DIRS`` order."""
        rest = dist.get(state)
        if not rest:
            return []
        out = []
        for di in range(4):
            nxt = self.step(state, di)
            if nxt != state and dist.get(nxt, -1) == rest - 1:
                out.append((di, nxt))
        return out

    def solve(self, state) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal
        set at every step, or None if this board can no longer be won."""
        _seen, dist = self.field(state)
        if state not in dist:
            return None
        presses, optsets = [], []
        cur = state
        while not self.won(cur):
            best = self.optimal(dist, cur)
            if not best:                      # unreachable: dist>0 has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


def _moved(crates: tuple, src: int, dst: int) -> tuple:
    """``crates`` with the crate at ``src`` moved to ``dst``, still sorted --
    sorted AFTER the insert, so the tuple is canonical and one board is one
    state however it was reached.

    ``dst`` is always free of obstacles when this is called -- every push
    branch in `step` reaches it only after the rules that would have refused
    the shove have declined -- so no two crates can ever be merged onto one
    cell (``--verify`` would catch it as a lost crate)."""
    return tuple(sorted([c for c in crates if c != src] + [dst]))


def parse_level(text: str) -> "tuple[_Board, tuple]":
    """``(board, state)`` for a level drawn in the .txt's own LEGEND
    characters. Used by ``--repair``, which has to reason about a board that is
    no longer in the file."""
    rows = [r for r in text.strip("\n").split("\n") if r.strip()]
    h, w = len(rows), max(len(r) for r in rows)
    walls, targets, crates = [], [], []
    rocks = 0
    player = stand = None
    for r, line in enumerate(rows):
        for c, ch in enumerate(line.ljust(w, ".")):
            i = r * w + c
            if ch == "#":
                walls.append(i)
            elif ch in "pP":
                player = i
            elif ch == "*":
                crates.append(i)
            elif ch == "%":
                rocks |= 1 << i
            elif ch == "@":
                crates.append(i)
                targets.append(i)
            elif ch in "oO":
                targets.append(i)
            elif ch == "+":
                stand = i
    return (_Board(h, w, walls, targets, stand),
            (player, tuple(sorted(crates)), rocks))


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class LargeStepsExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Board` field.

    The base class keeps the memo, the level scoping and the recording
    contract; only the strategy underneath changes, the way `PSEnumExpert`
    replaces it. Here the strategy is "read the live board, enumerate its
    component, hand back the shortest plan and its exact tie sets", so
    `heuristic` is never called and asserts rather than returning a number
    nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state -- the state a recovery prefix left behind -- is answered
    exactly, and answered with None when that state is one of the many this
    game cannot win from.
    """

    #: ACTION5 is read by no rule (see the module docstring and ``--verify``);
    #: branching on it would add a fifth of the enumeration for self-loops.
    directions = list(DIRS)

    #: `_key` is the dynamic objects only (player, crates and the rocks, which
    #: can be eaten), which is canonical WITHIN a level but not across them --
    #: the walls, targets and stand are static per level and differ between
    #: levels.
    scope_by_level = True

    def setup(self) -> None:
        self.player_ids = set(self.game._engine._player_indices)
        self.crate_ids = set(self.g.resolve_object_name("crate"))
        self.wall_ids = set(self.g.resolve_object_name("wall"))
        self.rock_ids = set(self.g.resolve_object_name("rock"))
        self.target_ids = set(self.g.resolve_object_name("target"))
        self.stand_ids = set(self.g.resolve_object_name("victorystand"))
        self.dyn_ids = self.player_ids | self.crate_ids | self.rock_ids
        #: `_Board`s by STATIC signature (see `read`), not by level index.
        self._boards: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "LargeStepsExpert reads an exact distance field; heuristic is "
            "unused")

    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn))

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng):
        """``(board, state)`` for the engine's current grid.

        Boards are cached by their STATIC signature (dimensions, walls,
        targets, stand -- the rocks are dynamic and live in the state), so the
        sweeps never rebuild the edge table, one board is shared by every state
        of its level, and ``--verify`` can assert that a press left the scenery
        alone simply by checking it got the same object back."""
        h, w = eng.height, eng.width
        walls, targets, crates = [], [], []
        rocks = 0
        player = stand = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.rock_ids:
                    rocks |= 1 << i
                if cell & self.target_ids:
                    targets.append(i)
                if cell & self.stand_ids:
                    stand = i
                if cell & self.player_ids:
                    player = i
                if cell & self.crate_ids:
                    crates.append(i)
        sig = (h, w, tuple(walls), tuple(targets), stand)
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, targets, stand)
        return board, (player, tuple(sorted(crates)), rocks)

    def seat(self, eng, board, state) -> None:
        """Write a model state back onto the engine as a settled board. Used
        only by ``--verify``, which is the whole point of it: a decoder that is
        not exactly `read`'s inverse would silently check a different game, so
        the two are kept next to each other."""
        bg = self.bg_id
        p, crates, rocks = state
        w = board.w
        eng.grid = [[{bg} for _ in range(board.w)] for _ in range(board.h)]
        for i, is_wall in enumerate(board.wall):
            if is_wall:
                eng.grid[i // w][i % w] |= self.wall_ids
        for i in range(board.h * board.w):
            if (rocks >> i) & 1:
                eng.grid[i // w][i % w] |= self.rock_ids
        for i in board.targets:
            eng.grid[i // w][i % w] |= self.target_ids
        if board.stand is not None:
            eng.grid[board.stand // w][board.stand % w] |= self.stand_ids
        for i in crates:
            eng.grid[i // w][i % w] |= self.crate_ids
        eng.grid[p // w][p % w] |= self.player_ids
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if state[0] is None:
            return None                       # no player (cannot happen here)
        if board.stand is None:
            # `All Player on VictoryStand` can never hold, so the level is not
            # winnable at all. Not reachable on the shipped levels; not a crash
            # if one is ever edited.
            return None
        if board.won(state):
            return Plan([], [])
        if len(board.targets) > len(state[1]):
            return None                       # more targets than crates
        return board.solve(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class LargeStepsSolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    expert_cls = LargeStepsExpert

    #: `games/ps:stand_aside_everyone_i_take_large_steps/...py` is a plain
    #: passthrough (it constructs the adapter and nothing else), so there is
    #: nothing to gain by routing through it -- but it IS what a live agent is
    #: handed, so if that wrapper ever grows a patch this must be set to the
    #: folder name.
    game_module_id = ""

    #: Unused: the expert enumerates an exact field rather than searching under
    #: a heuristic, so there is no weight to trade and no node budget that could
    #: quietly bite. Left at the base values so nothing reads a lie off them.
    weight = 1
    node_cap = 400_000

    #: The longest plan is 44 presses; the rest is room for the exploration
    #: prefix and the re-plan after it. Stays well under the adapter's own
    #: 200-step per-level budget, which `set_level` resets before the plan
    #: starts anyway.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Sweep every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time startup visible as startup
        rather than as a mysteriously slow first seed."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new():
    solver = LargeStepsSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _plans() -> int:
    """The plan for every level, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once. A plan that does not win the interpreter, or that wins EARLY, fails
    the report: a shortest plan must win on its last press and on no other."""
    _solver, game, expert, _solvable = _new()
    bad = 0
    total = ties_total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        t0 = time.time()
        plan = expert.plan(eng, level)
        dt = time.time() - t0
        head = f"level {level:>2}: {eng.height}x{eng.width}"
        if plan is None:
            print(f"{head} -- UNWINNABLE (see --bfs)")
            bad += 1
            continue
        early = None
        for i, d in enumerate(plan):
            eng.step(d)
            if eng.check_win() and i < len(plan) - 1:
                early = i
                break
        won = eng.check_win()
        if not won or early is not None:
            bad += 1
        sets = getattr(plan, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        ties_total += ties
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        note = ("WIN" if won else "NOT WON") if early is None else \
               f"won EARLY at press {early}"
        print(f"{head}, {len(plan):>2} presses (budget {game._max_steps}, "
              f"{room}), {ties}/{len(plan)} steps with a tie set, {dt:.2f}s "
              f"-- interpreter says {note}")
        print(f"  {' '.join(plan)}")
    print(f"{total} expert steps, {ties_total} of them with a tie set")
    print("plans clean" if not bad else f"PLANS FAILED: {bad} level(s)")
    return 0 if not bad else 1


def _bfs() -> int:
    """The exhaustive reachable-state report -- and therefore the proof that
    every plan is SHORTEST: each sweep terminates having generated every board
    the model admits, so the backward distance field is exact.

    It is also the DEAD-END report. Nothing here kills you and nothing restarts
    the level, but a crate shoved one cell too far is permanent, so most of this
    game's reachable space is boards that look perfectly ordinary and can never
    be won again."""
    _solver, game, expert, _solvable = _new()
    tot_seen = tot_live = 0
    t_all = time.time()
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(game._engine)
        # A FRESH board, so the printed time is the sweep and not a cache hit on
        # the one `prepare_expert` already ran at startup.
        fresh = _Board(board.h, board.w,
                       [i for i, x in enumerate(board.wall) if x],
                       board.targets, board.stand)
        t0 = time.time()
        seen, dist = fresh.field(state)
        dt = time.time() - t0
        wins = sum(1 for s in seen if fresh.won(s))
        d = dist.get(state)
        verdict = (f"shortest {d} presses" if d is not None
                   else "UNWINNABLE (no won board is reachable)")
        tot_seen += len(seen)
        tot_live += len(dist)
        print(f"level {level:>2}: {len(seen):>6} reachable boards "
              f"({len(dist):>5} can still win, {len(seen) - len(dist):>6} "
              f"stranded, {wins} already won), {dt:.2f}s -- {verdict}")
    print(f"{tot_seen} boards over {game.n_levels} levels, {tot_live} of them "
          f"winnable ({100 * (tot_seen - tot_live) / max(1, tot_seen):.0f}% "
          f"stranded), {time.time() - t_all:.1f}s")
    return 0


def _ties() -> int:
    """Re-derive every optimal-action label the independent way: take each
    candidate press and run a FRESH exhaustive sweep from the board it lands in.
    A press is optimal iff ``1 + dist(successor)`` equals the presses remaining,
    so this checks the distance field and the plan-following against sweeps that
    share no state with the ones that produced the labels."""
    _solver, game, expert, _solvable = _new()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(game._engine)
        plan = expert.plan(game._engine, level)
        if plan is None:
            print(f"level {level}: no plan (skipped)")
            continue
        walls = [i for i, x in enumerate(board.wall) if x]
        for pi, (taken, claimed) in enumerate(zip(plan, plan.optsets)):
            remaining = len(plan) - pi
            measured = []
            for di, d in enumerate(DIRS):
                nxt = board.step(state, di)
                if nxt == state:
                    continue
                if board.won(nxt):
                    cost = 1
                else:
                    fresh = _Board(board.h, board.w, walls,
                                   board.targets, board.stand)
                    _s2, d2 = fresh.field(nxt)
                    if nxt not in d2:
                        continue
                    cost = 1 + d2[nxt]
                if cost == remaining:
                    measured.append(d)
            if measured != list(claimed):
                print(f"  level {level} step {pi}: labelled {list(claimed)} "
                      f"but measured {measured}")
                bad += 1
            if taken not in claimed:
                print(f"  level {level} step {pi}: took {taken}, not in its "
                      f"own optimal set {list(claimed)}")
                bad += 1
            state = board.step(state, DIRS.index(taken))
        print(f"level {level:>2}: {len(plan)} steps verified")
    print("ties clean" if not bad else f"TIES FAILED: {bad} mismatches")
    return 0 if not bad else 1


def _repair(argv=()) -> int:
    """The proof that ncrecc's published board cannot be won, and the search
    over every single-cell edit of it.

    Four things are reported and each is measured rather than argued:

    * the forward sweep of the shipped board, and the count of its reachable
      states in which BOTH targets are covered (zero) -- so the crate half of
      the win alone is unreachable;
    * that sweep checked against the REAL interpreter, because a claim this
      strong may not rest on the model alone: sampled states (``--repair full``
      for all 38378 of them) are seated onto the engine and pressed in all five
      directions, board and win flag compared. The board is no longer in the
      .txt, so it is carried here as `SHIPPED_LEVEL` and checked in place; its
      walls, targets and stand are level 7's, so `read` hands back the very
      same `_Board` the generator plans that level with;
    * the parity account: which cells the player can stand on, split by
      ``(row % 2, column % 2)``, against the parity of the VictoryStand;
    * every single-cell edit -- each of the player, the two crates, the two
      targets and the stand moved to each free cell, plus every interior wall
      added or removed -- solved, and the winnable ones listed. Exactly eight
      exist, all of them relocations of the PLAYER, and the .txt ships the one
      with the longest plan.
    """
    board, start = parse_level(SHIPPED_LEVEL)
    w = board.w

    def rc(i):
        return (i // w, i % w)

    t0 = time.time()
    seen, dist = board.field(start)
    both = sum(1 for s in seen if all(t in s[1] for t in board.targets))
    print(f"ncrecc's board: {len(seen)} reachable states in "
          f"{time.time() - t0:.2f}s, {len(dist)} of them winnable, "
          f"{both} with BOTH targets covered")
    for t in board.targets:
        cov = sum(1 for s in seen if t in s[1])
        print(f"  target {rc(t)}: covered in {cov} reachable states")
    par: dict = {}
    for s in seen:
        r, c = rc(s[0])
        par[(r % 2, c % 2)] = par.get((r % 2, c % 2), 0) + 1
    print(f"  player start {rc(start[0])}, VictoryStand {rc(board.stand)} "
          f"(parity {rc(board.stand)[0] % 2},{rc(board.stand)[1] % 2})")
    print(f"  player parity classes reached: "
          f"{ {k: v for k, v in sorted(par.items())} }")

    engine_bad = _repair_engine_check(board, seen, argv)

    rows = [list(r) for r in SHIPPED_LEVEL.split("\n")]
    h = len(rows)
    free = [(r, c) for r in range(h) for c in range(w) if rows[r][c] == "."]
    elems = {"player": ((1, 1), "p"), "crate A": ((3, 6), "*"),
             "crate B": ((5, 4), "*"), "target A": ((3, 4), "o"),
             "target B": ((7, 6), "o"), "stand": ((6, 2), "+")}
    hits, tried = [], 0
    for name, ((r0, c0), ch) in elems.items():
        for (r, c) in free:
            grid = [x[:] for x in rows]
            grid[r0][c0] = "."
            grid[r][c] = ch
            b, s = parse_level("\n".join("".join(x) for x in grid))
            tried += 1
            plan = b.solve(s)
            if plan is not None:
                hits.append((len(plan), f"{name} {(r0, c0)} -> {(r, c)}"))
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if rows[r][c] not in ".#":
                continue
            grid = [x[:] for x in rows]
            grid[r][c] = "#" if rows[r][c] == "." else "."
            b, s = parse_level("\n".join("".join(x) for x in grid))
            tried += 1
            plan = b.solve(s)
            if plan is not None:
                verb = "add" if rows[r][c] == "." else "remove"
                hits.append((len(plan), f"{verb} wall {(r, c)}"))
    hits.sort()
    print(f"{tried} single-cell edits tried, {len(hits)} of them winnable")
    for d, what in hits:
        print(f"  {d:>3} presses: {what}")
    ok = (both == 0 and start not in dist and hits and not engine_bad
          and all("player" in what for _d, what in hits))
    print("repair report clean -- the shipped board is unwinnable and only "
          "moving the player fixes it" if ok else "REPAIR REPORT FAILED")
    return 0 if ok else 1


def _repair_engine_check(board, seen, argv) -> int:
    """Replay the shipped board's reachable states through the interpreter.

    This is what stops ``--repair``'s headline being a claim about `_Board`
    rather than about the game: the states are seated onto the real engine and
    pressed in all five directions, exactly as `_verify` does for the levels
    that are still in the .txt. Sampled by default; ``--repair full`` presses
    from every one of them."""
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    arg = next((a for a in argv if not a.startswith("--")), None)
    full = arg == "full"
    want = 4000 if arg is None or full else int(arg)
    pool = sorted(k for k in seen if not board.won(k))
    if not full and len(pool) > want:
        pool = random.Random(12345).sample(pool, want)
    eng.height, eng.width = board.h, board.w
    bad = 0
    t0 = time.time()
    checked = 0
    for state in pool:
        for di, d in enumerate(("up", "down", "left", "right", "action")):
            expert.seat(eng, board, state)
            eng.step(d)
            _b, got = expert.read(eng)
            wanted = state if d == "action" else board.step(state, di)
            checked += 1
            if got != wanted or eng.check_win() != board.won(wanted):
                print(f"  {state} {d}: engine {got} win {eng.check_win()}, "
                      f"model {wanted} win {board.won(wanted)}")
                bad += 1
    scope = "EVERY reachable board" if full else f"{len(pool)} sampled boards"
    print(f"  interpreter cross-check: {checked} presses over {scope} "
          f"(of {len(pool)}), {time.time() - t0:.1f}s -- "
          f"{'agrees' if not bad else f'{bad} DISAGREEMENTS'}")
    return bad


def _verify(argv) -> int:
    """The receipt for the native model: seat a model board onto the REAL
    interpreter, press all FIVE actions, and compare against `_Board.step`.

    Four things are asserted per press:

    * the resulting ``(player, crates)`` matches -- ``step`` IS the eight rules;
    * ``check_win()`` matches ``won()`` -- including the vacuous target half on
      the levels that have no targets at all;
    * the STATIC scenery is unchanged, checked by requiring `read` to hand back
      the very same `_Board` object (it caches on the walls, rocks, targets and
      stand), which is what licenses leaving all of it out of the state;
    * ACTION5 leaves the board untouched, which is what lets `directions` drop
      it.

    ``--verify`` samples (default 3000 boards per level, reproducibly, from the
    whole reachable component); ``--verify full`` checks every one of them;
    ``--verify <N>`` sets the sample size."""
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    arg = next((a for a in argv if not a.startswith("--")), None)
    full = arg == "full"
    want = 3000 if arg is None or full else int(arg)

    bad = 0
    checked_all = 0
    t_all = time.time()
    for level in range(game.n_levels):
        game.set_level(level)
        board, start = expert.read(eng)
        seen, _dist = board.field(start)
        # ALREADY-WON boards are excluded because nothing ever presses from
        # one: `record_level` stops on the win, `field` treats a won board as
        # terminal, and the adapter has ended the episode. (Unlike the `again`
        # games there is no mechanical reason here -- a press from a won board
        # would behave normally -- it is simply not part of the game.)
        won = [k for k in seen if board.won(k)]
        pool = sorted(k for k in seen if not board.won(k))
        live = len(pool)
        if not full and len(pool) > want:
            pool = random.Random(12345).sample(pool, want)
        t0 = time.time()
        checked = 0
        for state in pool:
            for di, d in enumerate(("up", "down", "left", "right", "action")):
                expert.seat(eng, board, state)
                eng.step(d)
                got_board, got_state = expert.read(eng)
                want_state = (state if d == "action"
                              else board.step(state, di))
                checked += 1
                if got_state != want_state:
                    print(f"  level {level} {state} {d}: engine {got_state}, "
                          f"model {want_state}")
                    bad += 1
                if eng.check_win() != board.won(want_state):
                    print(f"  level {level} {state} {d}: engine win "
                          f"{eng.check_win()}, model {board.won(want_state)}")
                    bad += 1
                if got_board is not board:
                    print(f"  level {level} {state} {d}: the static scenery "
                          f"changed")
                    bad += 1
        dt = time.time() - t0
        checked_all += checked
        scope = "EVERY playable board" if full else f"{len(pool)} sampled boards"
        print(f"level {level:>2}: {checked:>6} presses over {scope} "
              f"(of {live} playable, {len(won)} already-won skipped, "
              f"{len(seen)} reachable), {dt:.1f}s")
    print(f"{checked_all} presses in {time.time() - t_all:.0f}s")
    print("verify clean -- the model IS the interpreter" if not bad
          else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every cell size these levels render at.

    Whole 64x64 frames of uniform boards are compared rather than one cell out
    of a mixed board: `_render_frame` centre-pads a non-square board, so
    indexing a cell by ``cell_px * r`` silently reads the wrong pixels (the
    ps:explod lesson). Two uniform boards render identically iff their cells do.

    ``player_on_stand`` is the WIN frame and the reason the .txt is patched: as
    shipped it rendered as a bare player, i.e. the board this game's win
    condition is about was invisible, and so was every target the player was
    standing on."""
    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, _expert, _solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    comps = {
        "floor": (),
        "wall": ("wall",),
        "rock": ("rock",),
        "crate": ("crate",),
        "target": ("target",),
        "stand": ("victorystand",),
        "player": ("player",),
        "player_on_target": ("player", "target"),
        "player_on_stand": ("player", "victorystand"),
        "crate_on_target": ("crate", "target"),
        "crate_on_stand": ("crate", "victorystand"),
        "rock_on_target": ("rock", "target"),
        "rock_on_stand": ("rock", "victorystand"),
    }

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
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        note = "OK" if not clashes else f"IDENTICAL {clashes}"
        print(f"{h}x{w} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} "
              f"compositions -- {note}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


def _symmetry(walk_presses: int = 60) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for putting the game in `_FLIP_GAMES` (the rotation is
    mandatory and is checked here too). The structural argument is strong --
    gravity-free, every rule written with the relative ``>`` force so all four
    directional expansions exist, no directional art, a win condition that
    names no direction, screen-relative input, and ONE player, so nothing can
    contest a cell and the rule-order chirality Gobble Rush has to argue around
    cannot arise -- but arguments of exactly that shape hid it, so it is
    measured.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (a crate flat against a wall, a crate dead in a
    corner) and it presses the unbound ACTION key as well."""
    solver, game, expert, _solvable = _new()
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    def drive(g, level, presses):
        g.set_level(level)
        out = [np.asarray(g._current_frame)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame
                                  else g._current_frame))
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
            rng = random.Random(f"large_steps:symmetry:{level}")
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
    print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    print("symmetry clean" if not bad else f"SYMMETRY FAILED: {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--bfs" in sys.argv:
        sys.exit(_bfs())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--repair" in sys.argv:
        sys.exit(_repair(sys.argv[sys.argv.index("--repair") + 1:]))
    if "--verify" in sys.argv:
        sys.exit(_verify(sys.argv[sys.argv.index("--verify") + 1:]))
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    sys.exit(LargeStepsSolver.main())
