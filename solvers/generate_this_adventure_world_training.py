"""Generate Phase-1 training data for the PuzzleScript game
ps:this_adventure_world ("This Adventure World", Orange_Nitro).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, the native model of them, the layered search that plans every level,
the reports that prove those plans shortest and their labels exact, and the two
sprite fixes the game needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_this_adventure_world",
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
index is the *screen* action (post rotation remap), i.e. the button an agent
presses in the augmented view, so replaying the recorded actions reproduces the
recorded frames exactly. Every expert step carries the full set of
equally-optimal presses.

Level indices everywhere below are 0-BASED. The game's own ``message`` screens
sit between levels and are not levels; there are seven.


THE GAME
========
A sokoban with three extra verbs. Win is ``All Crates on Target``, where
``Crates = Crate or StickyCrate``. Five presses matter -- four arrows and
ACTION5.

    1  [> Player | Crate]       -> [> Player | > Crate]        ordinary push
    2  [< Player | StickyCrate] -> [< Player | < StickyCrate]  PULL
    3  [> Player | StickyCrate] -> [> Player | > StickyCrate]  push
    4  [> PlayerNothing | Key]  -> [> PlayerKey | ]            pick the key up
    5  [action PlayerKey | ]    -> [action PlayerNothing | Key] DROP the key
    8  [> PlayerKey | TargetLock] -> [> PlayerNothing | Target] unlock
    9  [> PlayerNothing | Father] -> [ Tombstone | Father ]     death
   10  [> PlayerKey | Father]     -> [ Tombstone | Father ]     death

So the four mechanics are:

* **the sticky crate is pulled as well as pushed.** Rule 2's ``<`` is the
  player moving AWAY from the crate, so a StickyCrate directly BEHIND the
  player comes along on every step. That is not optional and not aimed: you
  drag it whenever you walk off it, which is how levels 5 and 6 are solved --
  their crates sit under an overhang the player cannot get above.
* **the key opens the goal.** Some levels ship no Target at all, only a
  TargetLock; walking into it while carrying the Key converts it to a Target
  (and eats the key) with the player stepping onto the square it just made.
* **the Father kills you.** Walking into him replaces the player with a
  Tombstone. Nothing revives it, nothing marks it as a loss -- the engine has
  no ``playerdead`` object, so `check_game_over` stays False and the level
  simply sits there unwinnable until a RESET.
* **ACTION drops the key, destructively.** See the next section; it is the
  whole character of this adaptation.

Three of the game's objects never appear. No level places a Sword, so
``PlayerSword`` can never exist and rules 6/7/11 are dead; no level places the
Lady, so the ``Win`` command on rule 13 can never fire and the ONLY win is the
crate condition. `_Board` asserts all three at build time rather than assuming
it.


THE DROP DESTROYS WHATEVER IT LANDS ON
======================================
Rule 5's second cell is EMPTY, which in PuzzleScript is a wildcard, not "an
empty cell". So the RHS puts a Key into a neighbouring square whatever is
already there, and an object arriving in an occupied collision layer REPLACES
the occupant. Measured on the interpreter (``--selfcheck`` covers every branch
of it):

* the drop goes UP; if up is off the grid it goes DOWN, then LEFT, then RIGHT
  -- the rule carries no direction, so it is expanded to four copies and the
  first that fits fires, after which the player is a PlayerNothing and the
  other three no longer match;
* a **Wall** in that square is deleted and replaced by the Key. Pressing ACTION
  under a wall punches a hole through it, and stepping into the hole picks the
  key back up, so a shortcut costs exactly one extra press;
* a **Crate or StickyCrate** in that square is deleted and replaced by the Key.
  And ``All Crates on Target`` is vacuously true when no crate is left on the
  board (`_check_single_win_condition` only guards the vacuous case for PLAYER
  classes), so **destroying every crate wins the level**;
* a Target, a TargetLock or the Father is on the other collision layer and is
  untouched -- the key just lands on top of it.

That is not a bug in the adapter: real PuzzleScript expands a directionless
rule the same way and overwrites the same way. It IS the shortest solution to
three of the seven levels, and the plans below are shortest under the game as
adapted, which is the contract the corpus is trained and evaluated against:

    level        0    1    2    3    4    5    6   total
    shortest     7   10    5   18    6    3   24      73    (this generator)
    no ACTION    7   10   10   42   11    3   24     107    (``--designed``)

Levels 2, 3 and 4 are the three with a Key, and all three are won by dropping
it onto the crates instead of by solving the sokoban. Level 3's plan is worth
reading in full: walk to the key, cross the room, press ACTION under the Crate
(destroying it and putting the key back on the floor), step UP onto the key to
pick it up again, walk back and press ACTION under the StickyCrate. Eighteen
presses against forty-two for the intended route.

``--designed`` derives, and drives through the real interpreter, the shortest
win with ACTION5 removed from the alphabet -- the game as its author meant it.
It is a report, not a recording mode: what the corpus contains is the shortest
win, full stop.


THE ONE THING THE MODEL HAS TO GET RIGHT
========================================
**A front-cell rule fires on the FORCE, not on the move.** Rules 4, 8, 9 and 10
ask only that the player be MOVING INTO the square; whether the movement
resolves is settled afterwards, when forces are applied. Both of the
consequences below were found by fuzzing rather than by reading the rules, both
are reachable, and getting either wrong plans against a different game:

* **you can unlock through something.** Pressing towards a crate parked on a
  TargetLock opens the lock and spends the key with the player going nowhere at
  all -- and it opens one through a WALL as happily. Level 2 has a board where
  that press IS the win: the crate is already sitting on the lock, so turning
  the lock into a Target under it satisfies ``All Crates on Target`` on a press
  that moved nothing. The first version of `_Board` applied the unlock only on
  the branch where the player actually walks, and only a random walk long
  enough to shove that crate onto that lock found the difference.
* **you can die through something.** Walking into a crate that is standing on
  the Father kills you, and if that push had somewhere to go the crate still
  travels while you die. So a press that kills the player can COMPLETE THE WIN,
  the crate it shoved landing on the last Target after the player is gone --
  which is why `_Board`'s dead states keep their piece masks instead of
  collapsing to one sink, so `won` can still see them.

What is NOT force-based is the movement itself, and it resolves as a unit: a
press pushes what is ahead and pulls what is behind together, but a refused
push refuses the pull with it, and a press that kills the player does not pull
at all (the Tombstone lands in the square the crate was to be dragged into).

Everything above was MEASURED. ``--selfcheck`` fuzzes `_Board` against the real
interpreter and prints a coverage counter per model branch, so "0 mismatches"
is only reported next to proof that every branch was reached -- and the rare
ones are driven deterministically by a table of hand-built boards rather than
being left to the walk's luck.


A NATIVE MODEL, AND THE SEARCH IT AFFORDS
=========================================
`_Board` is the mechanics as bitmasks: the player as a flat ``r * W + c`` index
and every object class as one Python int over the same indexing. Walls and
torches are in the STATE rather than in the geometry, because the drop deletes
them.

The reachable space is not enumerable. Level 3's full closure blows past
400k states in a second and keeps going -- every subset of walls the drop can
punch out is its own board -- so the exhaustive-field approach that suits a
game with a fixed map is unavailable here. What is available is much cheaper:

`_Field` is a LAYERED forward BFS that stops at the first depth containing a
winning edge, followed by one backward sweep. Because layer ``i`` holds every
state at distance exactly ``i`` from the start, stopping at the first winning
layer ``D`` is a PROOF that ``D`` is the shortest solution -- every shorter
press sequence was generated and none of them won. The backward sweep then
marks, layer by layer, the states from which a win is still ``D - i`` presses
away, and that marking is exactly the optimal-action oracle:

    at a state on layer ``i``, press ``d`` is optimal  iff  it wins outright
    (only possible at ``i = D - 1``) or its successor is MARKED on layer
    ``i + 1``.

A successor that appears on an EARLIER layer is not optimal and the rule above
already excludes it: reaching it costs less than ``i + 1`` presses, so its own
distance-to-win must exceed ``D - i - 1`` or ``D`` would not be the minimum.

The whole game plans in 0.05 s and 16k states, level 3 being all of it:

    level     0    1    2     3    4    5    6
    d*        7   10    5    18    6    3   24
    states  106  179   49 15713   56   23  291

Six of the seven levels ARE small enough to enumerate exhaustively, and
``--space`` does exactly that as an independent check: a full forward closure
plus a backward distance field, Bellman-certified, whose ``d*`` must agree with
the layered search's. It also prints the dead-end map, and this game has one:
312 of level 0's 576 reachable states can no longer be won, and 19080 of level
2's 54945, with nothing on screen to say so -- which is what the RESET recovery
arc is for.

CLI
---
    --plans      per-level plan, replayed press by press through the REAL
                 interpreter, with the step budget checked
    --space      the exhaustive closure + Bellman-certified distance field for
                 the six levels that admit one; agreement with --plans' d*
    --ties       re-derive every optimal-action label by an INDEPENDENT bounded
                 re-search from each candidate successor, and cross-check the
                 model against the interpreter at every state a plan visits
    --selfcheck  fuzz the native model against the interpreter, with a coverage
                 counter per model branch
    --designed   the shortest win with ACTION5 removed -- the game as authored
                 -- derived and driven through the interpreter
    --symmetry   drive all four rotations and require every frame to be the
                 exact transform of the unaugmented one
    --audit      render every cell composition at every board shape and require
                 that the pairs which still collide are provably unreachable
    --record     record real episodes and drive the RECORDED actions back
                 through a fresh adapter: every frame identical, every level a
                 WIN, every expert press labelled
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

GAME_NAME = "This_Adventure_World"

#: The four moves, in the plan's tie-break order.
_MOVES = ("up", "down", "left", "right")

#: Every press the game reads, in the plan's tie-break order. ACTION5 is last so
#: a plan prefers a move when a move ties with it -- and so the drop, which is
#: the one irreversible press in the game, is never taken gratuitously.
_DIRS = _MOVES + ("action",)

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}

#: Object classes no level places and no rule can create. `_Board` asserts they
#: are absent rather than modelling them: without a Sword there is no
#: PlayerSword and no sword drop, and without the Lady the ``Win`` command on
#: rule 13 can never fire, so the crate condition is the only way to win.
_ABSENT = ("sword", "playersword", "lady")


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry plus the one-press transition.

    A state is the 8-tuple

        (player, carrying, key, crates, sticky, walls, torches, locks)

    with ``player`` and ``key`` flat ``r * W + c`` indices (``-1`` for "the
    player is dead" / "no key is lying on the board") and the rest bitmasks over
    the same indexing. ``carrying`` is the PlayerNothing / PlayerKey distinction.

    Walls, torches and locks are STATE, not geometry: the ACTION drop deletes a
    wall or a torch it lands on, and walking into a TargetLock with the key
    turns that lock into a Target. Only the Fathers and the shipped Targets
    never change, and they live on the board object.

    A dead state keeps its piece masks (``player == -1``, everything else as it
    ended up) rather than collapsing to one sink, because a press that kills the
    player can still shove the last crate onto the last Target -- see the module
    docstring. Dead states are absorbing: with no player on the grid no rule can
    match, so `step` returns them unchanged.
    """

    __slots__ = ("height", "width", "fathers", "base_targets", "base_locks",
                 "start", "_nb")

    def __init__(self, eng, idx):
        self.height, self.width = eng.height, eng.width
        w, n = self.width, self.height * self.width
        walls = torches = crates = sticky = locks = targets = fathers = 0
        key = player = -1
        carrying = dead = 0
        for r in range(self.height):
            row = eng.grid[r]
            for c in range(w):
                cell = row[c]
                i, bit = r * w + c, 1 << (r * w + c)
                if idx["wall"] in cell:
                    walls |= bit
                if idx["torchleft"] in cell or idx["torchright"] in cell:
                    torches |= bit
                if idx["crate"] in cell:
                    crates |= bit
                if idx["stickycrate"] in cell:
                    sticky |= bit
                if idx["targetlock"] in cell:
                    locks |= bit
                if idx["target"] in cell:
                    targets |= bit
                if idx["father"] in cell:
                    fathers |= bit
                if idx["key"] in cell:
                    key = i
                if idx["playernothing"] in cell:
                    player, carrying = i, 0
                if idx["playerkey"] in cell:
                    player, carrying = i, 1
                if idx["tombstone"] in cell:
                    dead = 1
                for name in _ABSENT:
                    if idx[name] in cell:
                        raise ValueError(
                            f"{GAME_NAME}: {name} on the board at ({r}, {c}); "
                            "no level places one and no rule creates one, so "
                            "_Board does not model it")
        if dead:
            player, carrying = -1, 0
        if (walls | torches) & fathers:
            # `step` would have to decide what rules 9/10 do when the move is
            # refused by something sharing the Father's own square. Nothing can
            # put them there -- a Wall or a Torch is never created and never
            # moves (the drop only DELETES one), and the Father never moves --
            # so the case is unreachable and `step` is written without it. A
            # level that broke that fails here rather than planning against a
            # model of a different game. ``--selfcheck`` reports the
            # corresponding branch as provably unreachable for the same reason.
            raise ValueError(
                f"{GAME_NAME}: a Wall or Torch shares a cell with the Father; "
                "_Board.step does not model that board")
        self.fathers = fathers
        self.base_targets = targets
        self.base_locks = locks
        self.start = (player, carrying, key, crates, sticky, walls, torches,
                      locks)

        # Per-direction "the cell one step that way", -1 off the grid. The grid
        # EDGE and a Wall block the player identically here, which is what the
        # interpreter does: a press that would leave the board simply fails to
        # move (measured, and covered by --selfcheck's `edge` counter).
        self._nb = {}
        for d, (dr, dc) in _DELTA.items():
            table = [-1] * n
            for r in range(self.height):
                for c in range(w):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.height and 0 <= cc < w:
                        table[r * w + c] = rr * w + cc
            self._nb[d] = table

    # -- the win condition ---------------------------------------------------
    def targets_of(self, locks: int) -> int:
        """The Target squares of a board whose remaining locks are ``locks``.

        A TargetLock only ever becomes a Target (rule 8) and a Target is never
        removed, so the shipped locks that are missing from ``locks`` are
        exactly the ones that have been opened."""
        return self.base_targets | (self.base_locks & ~locks)

    def won(self, state) -> bool:
        """``All Crates on Target`` -- including its vacuous case.

        With no crate left on the board the condition is satisfied, which is
        what the interpreter does (the "that is not a win" guard in
        `_check_single_win_condition` covers PLAYER classes only) and is how
        the drop exploit wins. The player being dead does not enter into it."""
        return (state[3] | state[4]) & ~self.targets_of(state[7]) == 0

    # -- mechanics -----------------------------------------------------------
    def step(self, state, direction, tally=None):
        """One press. Returns the new state, or ``state`` itself when the press
        changes nothing (a walk into a wall, a refused push, ACTION with no key
        -- all indistinguishable to a player).

        ``tally`` is the ``--selfcheck`` branch counter and is None everywhere
        else; it is threaded through here rather than reconstructed by a second
        classifier so the coverage report can never drift from the transition it
        claims to be covering.
        """
        player, carrying, key, crates, sticky, walls, torches, locks = state
        if player < 0:
            self._tick(tally, "dead-absorbing")
            return state

        if direction == "action":
            if not carrying:
                self._tick(tally, "action-no-key")
                return state
            # Rule 5 carries no direction, so it is four rules; the first whose
            # neighbour is on the grid fires and turns the player back into a
            # PlayerNothing, so the rest can no longer match.
            for probe in _MOVES:
                t = self._nb[probe][player]
                if t >= 0:
                    break
            else:                                   # pragma: no cover - 1x1 grid
                self._tick(tally, "drop-nowhere")
                return state
            bit = 1 << t
            self._tick(tally,
                       "drop-destroys" if (crates | sticky | walls | torches) & bit
                       else "drop-onto-floor")
            return (player, 0, t, crates & ~bit, sticky & ~bit,
                    walls & ~bit, torches & ~bit, locks)

        nb = self._nb[direction]
        f = nb[player]
        if f < 0:
            self._tick(tally, "edge")
            return state
        fbit = 1 << f
        behind = self._nb[_OPP[direction]][player]
        pull = behind >= 0 and (sticky >> behind) & 1

        # -- the front-cell rules, which fire on the FORCE ------------------
        # Rules 4, 8, 9 and 10 all ask only that the player be MOVING into the
        # square; whether the movement resolves is settled afterwards, when
        # forces are applied. So all four of them fire through a crate that
        # will not budge, and rule 8 fires through a wall. That is not a
        # detail: it is a trap. Pressing towards a crate parked on a
        # TargetLock spends the key and opens the lock without the player
        # going anywhere -- which is how one of level 2's boards wins -- and
        # pressing towards a crate parked on the Father kills you.
        #
        # Rule 4 precedes rules 8/9/10, so a key lying ON a lock is picked up
        # and spent on that lock in a single press, and a key lying on the
        # Father is picked up and lost with the corpse.
        if key == f and not carrying:
            carrying, key = 1, -1
            self._tick(tally, "pickup")
        if carrying and (locks & fbit):
            carrying = 0
            locks &= ~fbit
            self._tick(tally, "unlock")
        death = bool(self.fathers & fbit)

        # -- the movement, which is resolved as a unit ----------------------
        if (crates | sticky) & fbit:
            t = nb[f]
            tbit = (1 << t) if t >= 0 else 0
            keybit = (1 << key) if key >= 0 else 0
            if t < 0 or (walls | torches | crates | sticky | keybit) & tbit:
                moved = False
                self._tick(tally, "death-push-refused" if death
                           else "push-refused")
            else:
                moved = True
                if crates & fbit:
                    crates = (crates & ~fbit) | tbit
                else:
                    sticky = (sticky & ~fbit) | tbit
                self._tick(tally, "death-with-push" if death else "push")
        elif (walls | torches) & fbit:
            # ``death`` is provably False here: `__init__` rejects a board that
            # puts a Wall or Torch on the Father's own square, and nothing can
            # create one.
            moved = False
            self._tick(tally, "blocked")
        else:
            moved = True
            self._tick(tally, "death" if death else "walk")

        if death:
            # No pull: the Tombstone lands in the square the sticky crate was
            # about to be dragged into, so the drag has nowhere to go.
            return (-1, 0, key, crates, sticky, walls, torches, locks)
        if not moved:
            # NOT necessarily ``state``: a refused push still leaves behind
            # whatever the front-cell rules above did.
            return (player, carrying, key, crates, sticky, walls, torches,
                    locks)
        if pull:
            sticky = (sticky & ~(1 << behind)) | (1 << player)
            self._tick(tally, "pull")
        return (f, carrying, key, crates, sticky, walls, torches, locks)

    @staticmethod
    def _tick(tally, name) -> None:
        if tally is not None:
            tally[name] += 1

    # -- presentation --------------------------------------------------------
    def ascii(self, state) -> str:
        """A model state as ASCII, so a failing check can print the board it
        failed on rather than a tuple of large integers."""
        player, carrying, key, crates, sticky, walls, torches, locks = state
        targets = self.targets_of(locks)
        out = []
        for r in range(self.height):
            line = ""
            for c in range(self.width):
                i = r * self.width + c
                bit = 1 << i
                if walls & bit:
                    ch = "#"
                elif torches & bit:
                    ch = "T"
                elif i == player:
                    ch = "p" if carrying else "P"
                elif i == key:
                    ch = "k"
                elif crates & bit:
                    ch = "C"
                elif sticky & bit:
                    ch = "S"
                elif self.fathers & bit:
                    ch = "F"
                elif locks & bit:
                    ch = "L"
                elif targets & bit:
                    ch = "o"
                else:
                    ch = "."
                line += ch
            out.append("    " + line)
        if player < 0:
            out.append("    (the player is dead)")
        return "\n".join(out)


def _read_state(eng, idx, width):
    """The model state read back out of the INTERPRETER's grid.

    The inverse of `_Board.__init__`'s scan, kept separate because it must also
    read boards the model produced rather than only shipped levels -- in
    particular ones holding a Tombstone."""
    walls = torches = crates = sticky = locks = 0
    key = player = -1
    carrying = dead = 0
    for r in range(eng.height):
        row = eng.grid[r]
        for c in range(width):
            cell = row[c]
            i, bit = r * width + c, 1 << (r * width + c)
            if idx["wall"] in cell:
                walls |= bit
            if idx["torchleft"] in cell or idx["torchright"] in cell:
                torches |= bit
            if idx["crate"] in cell:
                crates |= bit
            if idx["stickycrate"] in cell:
                sticky |= bit
            if idx["targetlock"] in cell:
                locks |= bit
            if idx["key"] in cell:
                key = i
            if idx["playernothing"] in cell:
                player, carrying = i, 0
            if idx["playerkey"] in cell:
                player, carrying = i, 1
            if idx["tombstone"] in cell:
                dead = 1
    if dead:
        player, carrying = -1, 0
    return (player, carrying, key, crates, sticky, walls, torches, locks)


def _seat(eng, idx, board: _Board, state) -> None:
    """Write a model state onto the interpreter's grid.

    Only ``--selfcheck`` uses this: it is what lets the fuzz walk the MODEL and
    ask the interpreter about each state it reaches, instead of being limited to
    the states one plan happens to visit. The Tombstone of a dead state is not
    re-seated -- a dead board has no player, so no press can change it and
    nothing downstream reads it."""
    player, carrying, key, crates, sticky, walls, torches, locks = state
    w = board.width
    grid = [[{idx["background"]} for _ in range(w)] for _ in range(board.height)]

    def spread(mask, name):
        while mask:
            bit = mask & -mask
            mask ^= bit
            i = bit.bit_length() - 1
            grid[i // w][i % w].add(idx[name])

    spread(walls, "wall")
    spread(torches, "torchleft")
    spread(crates, "crate")
    spread(sticky, "stickycrate")
    spread(locks, "targetlock")
    spread(board.targets_of(locks), "target")
    spread(board.fathers, "father")
    if key >= 0:
        grid[key // w][key % w].add(idx["key"])
    if player >= 0:
        grid[player // w][player % w].add(
            idx["playerkey" if carrying else "playernothing"])
    eng.grid = grid
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


# ---------------------------------------------------------------------------
# The layered search
# ---------------------------------------------------------------------------

class _Field:
    """A shortest win from one state, with the exact optimal SET at every step.

    A layered forward BFS over `_Board.step` -- layer ``i`` is every state at
    distance exactly ``i`` from the start -- stopped at the first layer ``D``
    that contains a winning edge, followed by one backward sweep marking the
    states each layer can still win from in the presses it has left. See the
    module docstring for why that is a proof of shortestness and why the
    marking is the complete optimal-action oracle.

    Two things it is deliberately NOT:

    * a distance field over the whole reachable space. This game's space is not
      enumerable (the drop punches holes in walls, so every subset of them is a
      board), and a search that has to close it would never finish on level 3.
      ``--space`` builds the exhaustive field for the six levels that admit one,
      as an independent check on this one.
    * a heuristic search. There is no heuristic and no node budget that could
      quietly bite: ``capped`` says so loudly, and ``depth`` / ``states`` are in
      the ``--plans`` report so a level that grew would be visible.

    A search whose frontier empties with no win found is a PROOF that the start
    cannot be won -- every state it can reach was generated.
    """

    __slots__ = ("board", "start", "layers", "marked", "depth", "states",
                 "capped")

    def __init__(self, board: _Board, start=None,
                 depth_cap: int = 80, node_cap: int = 2_000_000):
        self.board = board
        self.start = board.start if start is None else start
        self.layers: list = []
        self.marked: list = []
        self.depth = None
        self.capped = False
        self.states = 1

        step, won = board.step, board.won
        if won(self.start) or self.start[0] < 0:
            # A start that is already won has no press to label, and a dead one
            # has no press at all. `record_level` handles the former itself.
            self.depth = 0 if won(self.start) else None
            self.layers = [{self.start: {}}]
            return

        seen = {self.start}
        self.layers = [{self.start: None}]
        for _ in range(depth_cap):
            frontier = self.layers[-1]
            nxt: dict = {}
            winning = False
            for state in frontier:
                edges = {}
                for direction in _DIRS:
                    succ = step(state, direction)
                    if succ == state:
                        continue            # a press that changes nothing
                    edges[direction] = succ
                    if won(succ):
                        winning = True
                        continue            # a win is a sink, never expanded
                    if succ not in seen:
                        if len(seen) >= node_cap:
                            self.capped = True
                            return
                        seen.add(succ)
                        if succ[0] >= 0:    # a corpse expands to nothing
                            nxt[succ] = None
                frontier[state] = edges
            self.states = len(seen)
            if winning:
                self.depth = len(self.layers)
                break
            if not nxt:
                return                      # the space closed with no win
            self.layers.append(nxt)
        if self.depth is None:
            self.capped = True              # depth_cap, not a closed space
            return
        self._mark()

    def _mark(self) -> None:
        """Layer by layer from the back: which states still win in the presses
        their layer has left."""
        won = self.board.won
        D = self.depth
        self.marked = [set() for _ in range(D)]
        self.marked[D - 1] = {s for s, e in self.layers[D - 1].items()
                              if any(won(n) for n in e.values())}
        for i in range(D - 2, -1, -1):
            ahead = self.marked[i + 1]
            self.marked[i] = {s for s, e in self.layers[i].items()
                              if any(n in ahead for n in e.values())}

    def optimal(self, state, layer: int) -> list:
        """Every press that is on a shortest win from ``state``, which must be a
        state of layer ``layer``."""
        if self.depth is None:
            return []
        edges = self.layers[layer][state]
        if layer == self.depth - 1:
            return [d for d in _DIRS
                    if d in edges and self.board.won(edges[d])]
        ahead = self.marked[layer + 1]
        return [d for d in _DIRS if d in edges and edges[d] in ahead]

    def plan(self) -> "Plan | None":
        """The shortest press sequence from the start, carrying the exact
        optimal SET at every step. Ties break on `_DIRS` order, so a re-derived
        plan is byte-identical across processes."""
        if self.depth is None:
            return None
        if self.depth == 0:
            return Plan([], [])
        presses, optsets = [], []
        state = self.start
        for layer in range(self.depth):
            best = self.optimal(state, layer)
            presses.append(best[0])
            optsets.append(best)
            state = self.layers[layer][state][best[0]]
        return Plan(presses, optsets)


class _Space:
    """The EXHAUSTIVE closure of one board, with its true distance-to-win field.

    Reports only. Six of the seven levels close in well under a second and this
    is the independent check on `_Field`: a full forward BFS with no depth
    limit, then a backward BFS from the winning edges, whose distance at the
    start must equal `_Field`'s ``depth``. It is also the only way to see the
    DEAD ENDS -- the states from which no win remains -- which this game has in
    quantity and shows on screen not at all.

    ``capped`` is the honest answer for level 3, whose closure is unbounded in
    practice: the drop deletes walls, so every subset of them the player can
    afford is a distinct board.
    """

    __slots__ = ("board", "succ", "dist", "start", "capped")

    def __init__(self, board: _Board, start=None, node_cap: int = 400_000):
        self.board = board
        self.start = board.start if start is None else start
        self.succ: dict = {}
        self.dist: dict = {}
        self.capped = False
        step, won = board.step, board.won
        seen = {self.start}
        queue = deque([self.start])
        while queue:
            state = queue.popleft()
            edges = {}
            for direction in _DIRS:
                succ = step(state, direction)
                if succ == state:
                    continue
                edges[direction] = succ
                if succ not in seen:
                    if len(seen) >= node_cap:
                        self.capped = True
                        self.succ = {}
                        return
                    seen.add(succ)
                    if not won(succ) and succ[0] >= 0:
                        queue.append(succ)
            self.succ[state] = edges
        self.dist = self._backward()

    def _backward(self) -> dict:
        won = self.board.won
        rev: dict = {}
        dist: dict = {}
        frontier = []
        for state, edges in self.succ.items():
            for succ in edges.values():
                if won(succ):
                    if state not in dist:
                        dist[state] = 1
                        frontier.append(state)
                else:
                    rev.setdefault(succ, []).append(state)
        queue = deque(frontier)
        while queue:
            state = queue.popleft()
            for prev in rev.get(state, ()):
                if prev not in dist:
                    dist[prev] = dist[state] + 1
                    queue.append(prev)
        return dist

    def optimal(self, state) -> list:
        best = self.dist.get(state)
        if best is None:
            return []
        out = []
        for d in _DIRS:
            succ = self.succ[state].get(d)
            if succ is None:
                continue
            cost = 1 if self.board.won(succ) else (
                None if succ not in self.dist else 1 + self.dist[succ])
            if cost == best:
                out.append(d)
        return out

    def certify(self) -> list:
        """Check the Bellman optimality conditions over the whole field.

        A proof that ``dist`` is the true distance-to-win function:

          (a) for every ``s`` in ``dist``: ``dist(s)`` equals the minimum over
              its edges of (1 if the edge wins else 1 + dist(edge)), that
              minimum taken over finite terms only;
          (b) for every reachable ``s`` NOT in ``dist``: no edge of ``s`` wins
              and no edge lands in ``dist``.

        (b) makes the complement of ``dist`` closed and win-free, so nothing
        outside ``dist`` can win. (a) makes ``dist`` drop by exactly one per
        press, so following any minimiser reaches a win in ``dist(s)`` presses.
        Together they pin ``dist`` to the true distance."""
        won = self.board.won
        bad = []
        for state, edges in self.succ.items():
            costs = []
            for succ in edges.values():
                if won(succ):
                    costs.append(1)
                elif succ in self.dist:
                    costs.append(1 + self.dist[succ])
            if state in self.dist:
                if not costs:
                    bad.append(f"{state}: in dist but no edge reaches a win")
                elif min(costs) != self.dist[state]:
                    bad.append(f"{state}: dist={self.dist[state]} but the best "
                               f"edge costs {min(costs)}")
            elif costs:
                bad.append(f"{state}: absent from dist yet an edge costs "
                           f"{min(costs)}")
        return bad


def _shortest_no_action(board: _Board, cap: int = 4_000_000):
    """The shortest win using the four MOVES only -- the game as authored.

    ``--designed`` is the whole caller. Same layered argument as `_Field`, over
    a four-press alphabet, and it returns the plan rather than a distance so the
    report can drive it through the interpreter."""
    step, won = board.step, board.won
    parent = {board.start: None}
    frontier = [board.start]
    while frontier:
        nxt = []
        for state in frontier:
            for direction in _MOVES:
                succ = step(state, direction)
                if succ == state or succ in parent:
                    continue
                parent[succ] = (state, direction)
                if won(succ):
                    out = []
                    cur = succ
                    while parent[cur] is not None:
                        cur, d = parent[cur]
                        out.append(d)
                    out.reverse()
                    return out, len(parent)
                if succ[0] >= 0:
                    nxt.append(succ)
            if len(parent) > cap:
                return None, len(parent)
        frontier = nxt
    return None, len(parent)


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class ThisAdventureWorldExpert(PSExpert):
    """`PSExpert`'s plan memo, disk cache and snapshot discipline around
    `_Field`.

    `_search` is replaced the way ps:stickyban and ps:escaping_limbo replace it:
    the base class keeps the memo, the restore discipline and the level scoping,
    and only the strategy underneath changes. Here the strategy reads the board
    off the engine ONCE and searches it natively -- the interpreter is the
    CERTIFIER (``--plans``, ``--ties``, ``--selfcheck``), never the planner.

    The search runs from whatever state the engine is in, not only from a level
    start, so this expert can genuinely re-plan out of a perturbed board. What
    it cannot do is re-plan out of a SPOILED one, and this game has plenty of
    those (see ``--space``): a crate shoved into a corner, a crate destroyed
    onto a square that leaves the rest unreachable, a dead player. That is why
    recovery is `PSAStarSolver`'s RESET arc rather than an epsilon detour.

    `heuristic` therefore asserts rather than returning a number nothing would
    use, and `_key` is inherited (every non-background cell), which is exact and
    canonical ACROSS levels here because the walls are in the key and no two
    levels share a wall layout -- so no ``scope_by_level``.
    """

    directions = list(_DIRS)

    #: Seed-independent, so without a file on disk every `parallelize_generator`
    #: shard would re-derive all seven. They are cheap here (0.05 s for the game)
    #: but the cache costs nothing and keeps the shards uniform with the rest of
    #: the family. A stored entry carries the start layout it was solved from,
    #: so an edited level is a miss rather than a wrong plan.
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "this_adventure_world_plans.json")

    #: Runaway guards on the layered BFS, not tuning dials: the deepest level
    #: needs 24 layers and the widest 15713 states.
    depth_cap: int = 80
    field_cap: int = 2_000_000

    def setup(self) -> None:
        self.idx = self.g.obj_name_to_idx

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "ThisAdventureWorldExpert searches layer by layer; "
            "heuristic is unused")

    def board(self, eng) -> _Board:
        """The native model of the engine's CURRENT grid."""
        return _Board(eng, self.idx)

    def field(self, eng, board: "_Board | None" = None) -> _Field:
        board = self.board(eng) if board is None else board
        return _Field(board, depth_cap=self.depth_cap, node_cap=self.field_cap)

    def _search(self, eng) -> "Plan | None":
        field = self.field(eng)
        if field.capped:
            return None
        return field.plan()


class ThisAdventureWorldSolver(PSAStarSolver):
    game_id = "puzzlescript_this_adventure_world"
    game_name = GAME_NAME
    expert_cls = ThisAdventureWorldExpert

    #: `games/ps:this_adventure_world/ps:this_adventure_world.py` is a plain
    #: passthrough -- it builds the adapter and nothing else, and the two sprite
    #: fixes are in the .txt, which both paths read. Set this if that wrapper
    #: ever grows a patch.
    game_module_id = ""

    #: Unused: `ThisAdventureWorldExpert._search` never calls `_astar`. Left at
    #: the base value so the constructor signature keeps working; ``depth_cap``
    #: and ``field_cap`` are the knobs that actually bound the search.
    node_cap = 400_000

    #: Room for the longest plan (24 presses) plus the RESET exploration prefix
    #: and the re-plan after it, and well under the adapter's own 200-step
    #: per-level budget.
    max_steps = 120

    #: Every level is solved; nothing is skipped.
    skip_levels = frozenset()

    #: Recovery comes from the RESET prefix, not from a detour inside the
    #: replay -- see `ThisAdventureWorldExpert` for why a perturbed board here
    #: is frequently a SPOILED one.
    epsilon = 0.0

    def prepare_expert(self, game, expert) -> None:
        """Search every level before `discover_solvable` asks for it -- the same
        work either way, but it fills the disk cache in one pass and makes the
        startup cost visible as startup."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    """The solver, its adapter and an expert -- without planning anything.

    The reports each want a different slice (``--audit`` needs no plan at all,
    ``--selfcheck`` needs only the model), so planning is left to whoever asks
    for it rather than paid on every entry point."""
    solver = ThisAdventureWorldSolver()
    game = solver.make_game(seed)
    return solver, game, ThisAdventureWorldExpert(game, node_cap=solver.node_cap)


def _press_line(presses) -> str:
    return " ".join("X" if p == "action" else p[0].upper() for p in presses)


def _plans(verbose: bool = True) -> int:
    """Every level's plan, replayed through the REAL interpreter.

    The end-to-end test of the model, the layered search and the tie labelling
    at once: the plan is derived natively, then stepped through the interpreter
    press by press, and the win is the interpreter's own ``check_win``. A plan
    the model believes in and the interpreter does not is caught here.
    """
    solver, game, expert = _new()
    eng = game._engine
    total = ties = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        crates = bin(board.start[3] | board.start[4]).count("1")
        goals = bin(board.base_targets).count("1")
        locks = bin(board.base_locks).count("1")
        t0 = time.time()
        field = expert.field(eng, board)
        plan = field.plan()
        dt = time.time() - t0
        head = (f"  L{level}: {board.height:2d}x{board.width:2d}  "
                f"{crates} crate(s) / {goals} target(s) + {locks} lock(s)")
        if plan is None:
            bad += 1
            print(f"{head}   NO PLAN "
                  f"({'search capped' if field.capped else 'UNWINNABLE'})")
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        tie_steps = sum(1 for s in plan.optsets if len(s) > 1)
        total += len(plan)
        ties += tie_steps
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        if verbose:
            print(f"{head}  {len(plan):3d} presses  interpreter win={won}  "
                  f"(budget {game._max_steps}, {room})  "
                  f"{field.states:6d} states  {tie_steps:2d} tie steps  "
                  f"{dt:5.2f}s")
            print(f"        {_press_line(plan)}")
    if verbose:
        print(f"  {total} presses, {ties} of them with a second equally-right "
              f"answer")
        print("  every plan reaches a WIN in the interpreter" if not bad
              else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _space(verbose: bool = True) -> int:
    """The exhaustive closure + Bellman-certified distance field, for the levels
    that admit one -- an INDEPENDENT check on `_Field`'s answer.

    Nothing here is used for planning. Its job is to say two things the layered
    search cannot: that a completely different derivation of ``d*`` agrees, and
    how much of each level's space is already lost. A crate is only movable
    while the player can reach a cell beside it, a destroyed crate is gone for
    good and a dead player is a dead level -- so most of these boards can be
    spoiled with nothing on screen to say so.

    Level 3 is expected to cap: its drop punches holes in walls, so the closure
    is every subset of them the player can afford. Its ``d*`` is proved by the
    layered search in ``--plans`` instead, which needs no closure at all.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        t0 = time.time()
        space = _Space(board)
        dt = time.time() - t0
        if space.capped:
            layered = expert.field(eng, board)
            print(f"  L{level}: closure capped (not enumerable) -- "
                  f"d*={layered.depth} stands on the layered proof, "
                  f"{layered.states} states")
            continue
        violations = space.certify()
        for v in violations[:5]:
            print(f"  L{level}: BELLMAN VIOLATION {v}")
        bad += len(violations)
        reach = len(space.succ)
        live = len(space.dist)
        here = space.dist.get(space.start)
        layered = expert.field(eng, board)
        agree = here == layered.depth
        bad += not agree
        if verbose:
            print(f"  L{level}: {reach:7d} reachable states "
                  f"({live:7d} can still win, {reach - live:6d} spoiled), "
                  f"{dt:5.2f}s -- field d*={here}, layered d*={layered.depth} "
                  f"{'AGREE' if agree else 'DISAGREE'}"
                  f"{'' if not violations else '  FIELD NOT CERTIFIED'}")
    print("  every enumerable level is certified and agrees with the layered "
          "search" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _ties(verbose: bool = True) -> int:
    """Re-derive every optimal-action label INDEPENDENTLY, and pin the model to
    the interpreter at every state a plan visits.

    Two things happen per level, and neither reuses `_Field`'s marking:

    1. at plan step ``i``, for each of the five presses, a FRESH bounded search
       (`_reaches`) asks whether the successor can still win in the
       ``d* - i - 1`` presses that would be left. That set must be exactly the
       label the plan shipped. It is a different derivation of the same fact --
       forward reachability from one state against a backward sweep over
       layers -- so a bug in the sweep cannot hide in both.
    2. the plan is walked in the INTERPRETER, and at every step all five
       successors are read back out of its grid and compared with
       `_Board.step`. ``--selfcheck`` fuzzes the model on random boards; this
       pins it on exactly the boards the recording will visit.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        field = expert.field(eng, board)
        plan = field.plan()
        if plan is None:
            print(f"  L{level}: no plan (skipped)")
            bad += 1
            continue
        state = field.start
        for pi, (taken, claimed) in enumerate(zip(plan, plan.optsets)):
            here = snapshot(eng)
            engine_state = _read_state(eng, expert.idx, board.width)
            if engine_state != state:
                print(f"  L{level} step {pi}: the interpreter is at a "
                      f"different board than the plan\n"
                      f"{board.ascii(engine_state)}")
                bad += 1
            for direction in _DIRS:
                restore(eng, here)
                eng.step(direction)
                if _read_state(eng, expert.idx, board.width) != board.step(
                        state, direction):
                    print(f"  L{level} step {pi} ({direction}): MODEL "
                          f"DISAGREES with the interpreter\n"
                          f"{board.ascii(state)}")
                    bad += 1
            restore(eng, here)

            budget = len(plan) - pi - 1
            measured = []
            for direction in _DIRS:
                succ = board.step(state, direction)
                if succ == state:
                    continue
                if board.won(succ):
                    measured.append(direction)
                elif budget and _reaches(board, succ, budget):
                    measured.append(direction)
            if measured != list(claimed):
                print(f"  L{level} step {pi}: labelled {claimed}, an "
                      f"independent search says {measured}\n"
                      f"{board.ascii(state)}")
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
            tie_steps = sum(1 for s in plan.optsets if len(s) > 1)
            print(f"  L{level}: {len(plan):3d} labels re-derived "
                  f"({tie_steps:2d} with a tie set), "
                  f"{len(plan) * len(_DIRS)} model/interpreter comparisons "
                  f"-- {'ok' if not bad else 'PROBLEM'}")
    print("  every label reproduces and the model matches the interpreter at "
          "every state the plans visit" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _reaches(board: _Board, state, budget: int) -> bool:
    """Can ``state`` reach a win in at most ``budget`` presses?

    A plain depth-limited BFS with its own ``seen`` set -- deliberately the
    dumbest possible independent oracle, so that ``--ties`` comparing it with
    `_Field`'s backward marking is a real comparison and not the same code run
    twice."""
    if board.won(state):
        return True
    seen = {state}
    frontier = [state]
    for _ in range(budget):
        nxt = []
        for cur in frontier:
            for direction in _DIRS:
                succ = board.step(cur, direction)
                if succ == cur or succ in seen:
                    continue
                if board.won(succ):
                    return True
                seen.add(succ)
                if succ[0] >= 0:
                    nxt.append(succ)
        if not nxt:
            return False
        frontier = nxt
    return False


def _designed(verbose: bool = True) -> int:
    """The shortest win with ACTION5 removed from the alphabet -- the game as
    its author meant it -- derived natively and driven through the interpreter.

    A report, not a recording mode. What it measures is the price of the drop
    exploit: the three levels that ship a Key are all won faster by destroying
    their crates than by solving their sokoban, and this is how much faster. It
    is also a second, ACTION-free proof that every level is genuinely solvable,
    which matters because the recorded plans lean on a mechanic that is
    arguably an accident of the rule's empty cell."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    rows = []
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        t0 = time.time()
        presses, states = _shortest_no_action(board)
        dt = time.time() - t0
        shortest = expert.field(eng, board).depth
        if presses is None:
            print(f"  L{level}: NO ACTION-FREE WIN ({states} states)")
            bad += 1
            continue
        for direction in presses:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        rows.append((level, shortest, len(presses)))
        if verbose:
            print(f"  L{level}: {len(presses):3d} presses without ACTION "
                  f"(shortest overall {shortest:3d}), interpreter win={won}, "
                  f"{states:7d} states, {dt:5.2f}s")
            print(f"        {_press_line(presses)}")
    if verbose:
        exploit = sum(s for _l, s, _d in rows)
        designed = sum(d for _l, _s, d in rows)
        print(f"  {designed} presses as authored against {exploit} with the "
              f"drop -- the exploit saves {designed - exploit}")
    return 1 if bad else 0


#: Hand-built boards, each here to reach one rare transition deterministically.
#: ``(name, rows, extras, carrying)`` where ``rows`` uses the game's own legend
#: characters, ``extras`` adds objects the legend cannot express in one cell
#: (a crate STANDING on a lock, a key lying on the Father), and ``carrying``
#: turns the shipped PlayerNothing into a PlayerKey.
#:
#: Every one of these was measured on the interpreter first and several of them
#: are the reason `_Board.step` is shaped the way it is.
_SCENARIOS = (
    ("crate on a lock, push refused -- the key is spent anyway",
     ("#####", "#PC##", "#####"), ((1, 2, "targetlock"),), True),
    ("crate on a lock, push allowed -- unlock and walk on",
     ("#####", "#PC.#", "#####"), ((1, 2, "targetlock"),), True),
    ("a lock behind a wall -- rule 8 opens it through the wall",
     ("#####", "#P###", "#####"), ((1, 2, "targetlock"),), True),
    ("a lock behind a torch",
     ("#####", "#PE.#", "#####"), ((1, 2, "targetlock"),), True),
    ("the key lying ON a lock -- picked up and spent in one press",
     ("#####", "#PL.#", "#####"), ((1, 2, "key"),), False),
    ("the key lying on the Father -- picked up and lost with the corpse",
     ("#####", "#PF.#", "#####"), ((1, 2, "key"),), False),
    ("a crate standing on the Father, push allowed -- it travels, you die",
     ("######", "#PC.##", "######"), ((1, 2, "father"),), False),
    ("a crate standing on the Father, push refused",
     ("######", "#PC###", "######"), ((1, 2, "father"),), False),
    ("a crate on the Father shoved onto the last Target -- a win on death",
     ("######", "#PCQ.#", "######"), ((1, 2, "father"),), False),
    ("a sticky crate behind you and the Father ahead -- no pull",
     ("######", "#SPF.#", "######"), (), False),
    ("sticky behind, crate ahead, push refused -- the pull is refused too",
     ("######", "#SPC##", "######"), (), False),
    ("sticky behind, crate ahead, push allowed -- both move",
     ("######", "#SPC.#", "######"), (), False),
    ("sticky on both sides",
     ("######", "#SPS.#", "######"), (), False),
    ("two sticky crates in a row -- the chain does not push",
     ("######", "#PSS.#", "######"), (), False),
    ("a crate cannot be pushed onto the key",
     ("######", "#PCK.#", "######"), (), False),
    ("ACTION under a wall punches a hole in it",
     ("#####", "#####", "#.P.#", "#####"), (), True),
    ("ACTION under a crate destroys it -- and the level",
     ("#####", "#.C.#", "#.P.#", "#####"), (), True),
    ("ACTION with up off the grid drops downwards",
     ("#P.#", "#..#", "####"), (), True),
    ("ACTION with no key does nothing",
     ("#####", "#.C.#", "#.P.#", "#####"), (), False),
)


def _scenarios(game, expert, tally, verbose: bool = True) -> int:
    """Drive `_SCENARIOS` through model and interpreter, all five presses each.

    Deterministic coverage of the transitions a random walk reaches only by
    luck. Returns the number of mismatches, and ticks ``tally`` exactly as the
    random walk does so the coverage table below counts both."""
    eng, gm = game._engine, game._game
    idx = expert.idx
    bad = 0
    for name, rows, extras, carrying in _SCENARIOS:
        base = gm._build_level(list(rows))
        for direction in _DIRS:
            eng.load_level(base)
            for (r, c, obj) in extras:
                eng.grid[r][c].add(idx[obj])
            if carrying:
                for row in eng.grid:
                    for cell in row:
                        if idx["playernothing"] in cell:
                            cell.discard(idx["playernothing"])
                            cell.add(idx["playerkey"])
            eng._position_index_dirty = True
            eng._rule_noop_cache.clear()
            board = _Board(eng, idx)
            state = board.start
            eng.step(direction)
            got = _read_state(eng, idx, board.width)
            expected = board.step(state, direction, tally)
            engine_won = eng.check_win()
            if got != expected or engine_won != board.won(expected):
                bad += 1
                print(f"  scenario {name!r} press={direction}: MISMATCH")
                print(f"    from:\n{board.ascii(state)}")
                print(f"    model (win={board.won(expected)}):\n"
                      f"{board.ascii(expected)}")
                print(f"    interpreter (win={engine_won}):\n"
                      f"{board.ascii(got)}")
    if verbose:
        print(f"  {len(_SCENARIOS)} hand-built scenarios x {len(_DIRS)} "
              f"presses: {bad} mismatch(es)")
    return bad


def _selfcheck(presses: int = 20000, seed: int = 0, verbose: bool = True) -> int:
    """Fuzz `_Board` against the real interpreter, and report BRANCH COVERAGE.

    A seeded random walk on every level, compared press by press: the model's
    successor against the state read back out of the interpreter's grid, and the
    model's win predicate against ``eng.check_win()``. Random rather than
    plan-driven on purpose -- a plan never wedges a crate in a corner, never
    walks into the Father and never drops the key onto a wall, and those are
    exactly the transitions the model has to be right about.

    The walk drives the MODEL and re-seats each state onto the interpreter
    (`_seat`), so it is not confined to the states a plan visits; on a win or a
    death it restarts from the level.

    The coverage counter is the half that makes a "0 mismatches" line worth
    anything. It counts which branch of `_Board.step` ran, so the report shows
    that the walk actually reached:

        walk / blocked / edge      the ordinary move and its two refusals
        push / push-refused        the sokoban
        pull                       a StickyCrate dragged out from behind
        pickup / unlock            the key picked up, and spent on a lock
        death                      walking into the Father
        death-with-push            the death that still moves the crate
        death-push-refused         the death behind a crate that cannot budge
        drop-destroys              ACTION deleting a wall, a torch or a crate
        drop-onto-floor            ACTION putting the key down harmlessly
        action-no-key              ACTION with nothing to drop

    ``unlock`` firing on a press the player could not complete is the branch
    this report exists for: it was wrong in the first version of `_Board` and
    only a walk long enough to shove a crate onto level 2's lock found it. The
    rare branches are therefore not left to the walk's luck -- `_SCENARIOS`
    below drives each of them deterministically first, on hand-built boards,
    with all five presses compared at each one.
    """
    _solver, game, expert = _new()
    eng = game._engine
    idx = expert.idx
    tally: Counter = Counter()
    bad = _scenarios(game, expert, tally, verbose)
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        rng = random.Random(f"this_adventure_world:{level}:{seed}")
        state = board.start
        for t in range(presses):
            if state[0] < 0 or board.won(state):
                state = board.start
                continue
            direction = rng.choice(_DIRS)
            _seat(eng, idx, board, state)
            eng.step(direction)
            got = _read_state(eng, idx, board.width)
            expected = board.step(state, direction, tally)
            engine_won = eng.check_win()
            if got != expected or engine_won != board.won(expected):
                bad += 1
                print(f"  L{level} t={t} press={direction}: MISMATCH")
                print(f"    from:\n{board.ascii(state)}")
                print(f"    model (win={board.won(expected)}):\n"
                      f"{board.ascii(expected)}")
                print(f"    interpreter (win={engine_won}):\n"
                      f"{board.ascii(got)}")
                if bad >= 5:
                    return bad
            state = expected
    expected_branches = (
        "walk", "blocked", "edge", "push", "push-refused", "pull", "pickup",
        "unlock", "death", "death-with-push", "death-push-refused",
        "drop-destroys", "drop-onto-floor", "action-no-key")
    missing = [b for b in expected_branches if not tally[b]]
    if verbose:
        print(f"  {presses * game.n_levels} presses compared against the "
              f"interpreter, {bad} mismatch(es)")
        for name in expected_branches:
            print(f"    {name:20s} {tally[name]:8d}"
                  + ("   NEVER REACHED" if not tally[name] else ""))
        extra = sorted(set(tally) - set(expected_branches))
        for name in extra:
            print(f"    {name:20s} {tally[name]:8d}   (undeclared branch)")
    if missing:
        print(f"  COVERAGE GAP: {missing} never ran -- the 0-mismatch line "
              f"says nothing about them")
    return 1 if (bad or missing) else 0


def _symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Drive all four rotations and require every frame to be the exact
    transform of the unaugmented one.

    ``This_Adventure_World`` is in none of the adapter's exemption sets, so it
    takes the default augmentation: `_do_reset` draws ``rotation_k`` from
    ``(seed, level)`` and `perform_action` forward-remaps directional input
    through it. The engine itself never rotates -- only the presentation does --
    so this is a check that the harness's `screen_action` contract holds, which
    is the one thing in the ps: family that fails silently.

    The game is NOT in ``_FLIP_GAMES``, so it gets 4 presentations rather than
    16. The argument for adding it would be sound (nothing here is
    gravity-dependent, and the engine sees the same presses whatever the
    presentation), but that is an edit to the shared adapter, and 4 x 7 = 28
    presentations is the family default.
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
        plan = expert.plan(game._engine, level)
        if plan is None:
            print(f"  L{level}: no plan, cannot check symmetry")
            return 1
        plans[level] = list(plan)

    def drive(g, level, presses):
        g.set_level(level)
        out = [np.asarray(g._current_frame)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame else g._current_frame))
        return out

    ref, seen, bad = {}, set(), 0
    for seed in range(200):
        if len(seen) == 4 and seed > 20:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"this_adventure_world:symmetry:{level}")
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
                if any(not np.array_equal(np.rot90(a, k=k[0]), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"    seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
        print("  every presentation is an exact transform of the unaugmented "
              "one" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


#: Every cell stack the two mutable collision layers can hold. ``PlayerSword``,
#: ``Sword`` and ``Lady`` are excluded because no level places them and no rule
#: can create one (`_Board` asserts it).
_LAYER2 = ("", "target", "targetlock", "father")
_LAYER3 = ("", "playernothing", "playerkey", "wall", "crate", "stickycrate",
           "key", "torchleft", "torchright", "tombstone")


def _audit(verbose: bool = True) -> int:
    """Render every cell composition at every board shape, and require that the
    pairs which still collide are provably UNREACHABLE.

    Whole 64x64 frames of UNIFORM boards are compared rather than one cell out
    of a mixed board: `_render_frame` centre-pads a non-square board, so
    indexing a cell by ``cell_px * r`` silently reads the wrong pixels (the
    ps:explod lesson). Two uniform boards render identically iff their cells do.

    THE TWO SPRITE FIXES THIS GAME NEEDED are in
    ``data/puzzlescript_games/This_Adventure_World.txt`` (with the detail in the
    comment at the top of that file), and reverting either one is exactly what
    this report prints:

    * Crate and StickyCrate shipped with their transparency in the MIDDLE, on
      sprite row/col 2. Level 3 is 9x13, which `_render_frame` draws at
      cell_px 4, and the centred nearest-neighbour sampler picks sprite
      rows/cols {0,1,3,4} at that size -- so row/col 2 is dropped and both
      crates went fully opaque. On the one level where a crate has to be pushed
      onto a Target, "is this crate home?" -- the win condition itself -- was
      invisible. Both sprites now give up their four CORNERS instead, which
      survive at every cell size this game draws (9, 7 and 4 px).
    * TargetLock's black keyhole sat on the pixels the Player paints over, so a
      player standing on a lock and a player standing on an open Target were the
      same picture at every size. It now also paints the four corners, which the
      Player leaves transparent.

    THE THREE FAMILIES THAT STILL COLLIDE, and why none can occur:

    * ``wall`` with anything on the lower layer. A Wall is never created and
      never moves -- the drop only DELETES one -- so a wall can share a cell
      with a Target, a TargetLock or the Father only if a level shipped it that
      way. Checked below, on every level.
    * ``father + tombstone``. A Tombstone appears only in the square the player
      was standing in, and the player can never be standing on the Father: a
      press towards him kills before anything moves. Checked below by asserting
      no level starts a player on a Father, which with the rule above is the
      whole argument.
    * ``crate`` / ``stickycrate`` with ``father``, at cell_px 4 only. Only level
      3 draws at 4 px and level 3 has no Father. Checked below, per shape.

    Any collision outside those three is a failure.
    """
    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, expert = _new()
    eng, g = game._engine, game._game
    idx = expert.idx

    # -- the reachability side of the argument, on the real levels ------------
    bad = 0
    shapes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        shapes.setdefault((board.height, board.width), []).append(level)
        immobile = board.start[5] | board.start[6]        # walls | torches
        layer2 = board.base_targets | board.base_locks | board.fathers
        if immobile & layer2:
            print(f"  L{level}: a Wall or Torch SHIPS on a Target / lock / "
                  f"Father -- the 'wall+X is unreachable' argument fails")
            bad += 1
        if board.start[0] >= 0 and (board.fathers >> board.start[0]) & 1:
            print(f"  L{level}: the player STARTS on the Father -- the "
                  f"'tombstone+father is unreachable' argument fails")
            bad += 1

    comps = {}
    for a in _LAYER2:
        for b in _LAYER3:
            name = "+".join(x for x in (a, b) if x) or "floor"
            comps[name] = tuple(x for x in (a, b) if x)

    for (h, w), levels in sorted(shapes.items()):
        cell_px = min(64 // h, 64 // w)
        has_father = any(_level_has_father(game, expert, lvl) for lvl in levels)
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        excused, fatal = [], []
        for pair in clashes:
            if all("wall" in x.split("+") for x in pair) or any(
                    x == "wall" for x in pair):
                excused.append(pair)
            elif set(pair) == {"tombstone", "father+tombstone"}:
                excused.append(pair)
            elif not has_father and any("father" in x for x in pair):
                excused.append(pair)
            else:
                fatal.append(pair)
        bad += len(fatal)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {cell_px}px, levels "
                  f"{','.join(str(x) for x in levels)}, father="
                  f"{'yes' if has_father else 'no'}): {len(comps)} "
                  f"compositions -- {len(excused)} unreachable collision(s), "
                  f"{len(fatal) or 'no'} fatal")
        for pair in fatal:
            print(f"      INDISTINGUISHABLE AND REACHABLE: {pair}")
    print("  audit clean" if not bad
          else f"  AUDIT FAILED: {bad} problem(s)")
    return 1 if bad else 0


def _level_has_father(game, expert, level: int) -> bool:
    game.set_level(level)
    return bool(expert.board(game._engine).fathers)


def _record(episodes: int = 5, verbose: bool = True) -> int:
    """Record real episodes and REPLAY them -- the end-to-end check on the
    deliverable rather than on the plan.

    For each seed: run `solve_episode` exactly as generation does (exploration
    prefix, RESET, expert replay), then drive the RECORDED screen actions
    through a fresh adapter at the same seed and require every frame to come
    back identical and the level to end in ``GameState.WIN``. That is what says
    the rotation contract held, that the prefix's presses were taped as they
    were driven, and that nothing in the recorder desynced.

    It also audits the labels, whose standard is stated in the corpus rules: an
    ``optimal`` set on every step EXCEPT the opening exploration prefix and the
    RESET that closes it, neither of which `train_policy` supervises.
    """
    import numpy as np

    from arcengine import ActionInput, GameState
    from solvers.base_solver import _ID_TO_GAMEACTION

    solver = ThisAdventureWorldSolver()
    bad = 0
    frames_checked = expert_steps = prefix_steps = 0
    for seed in range(episodes):
        solver.rng = random.Random(seed)
        ok, levels = solver.solve_episode(seed, explore=True)
        if not ok or not levels:
            print(f"  seed {seed}: solve_episode failed ({len(levels)} levels)")
            bad += 1
            continue
        game = solver.make_game(seed)
        for entry in levels:
            level = entry["level_id"]
            game.set_level(level)
            got = [np.asarray(game._current_frame)]
            for act in entry["actions"][1:]:
                fd = game.perform_action(
                    ActionInput(id=_ID_TO_GAMEACTION[act["index"]]))
                got.append(np.asarray(
                    fd.frame[-1] if fd.frame else game._current_frame))
            want = [np.asarray(o) for o in entry["observations"]]
            if len(want) != len(got) or any(
                    not np.array_equal(a, b) for a, b in zip(want, got)):
                print(f"  seed {seed} L{level}: the recorded actions do not "
                      f"reproduce the recorded frames")
                bad += 1
            if game._state != GameState.WIN:
                print(f"  seed {seed} L{level}: replay does not end in a WIN")
                bad += 1
            frames_checked += len(want)

            # The label audit. The prefix runs until the RESET that ends it,
            # and everything after that RESET is an expert press.
            acts = entry["actions"]
            resets = [i for i, a in enumerate(acts) if a["index"] == 0]
            first = (resets[-1] + 1) if len(resets) > 1 else 1
            prefix_steps += first - 1
            for i, act in enumerate(acts[first:], start=first):
                expert_steps += 1
                if not act.get("optimal"):
                    print(f"  seed {seed} L{level} step {i}: an EXPERT press "
                          f"with no optimal set")
                    bad += 1
    if verbose:
        print(f"  {episodes} episodes, {frames_checked} frames replayed, "
              f"{expert_steps} expert presses all labelled, "
              f"{prefix_steps} exploration presses before the RESET")
        print("  every episode replays frame-for-frame to a WIN" if not bad
              else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--space" in sys.argv:
        sys.exit(_space())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--selfcheck" in sys.argv:
        sys.exit(_selfcheck())
    if "--designed" in sys.argv:
        sys.exit(_designed())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--record" in sys.argv:
        sys.exit(_record())
    sys.exit(ThisAdventureWorldSolver.main())
