"""Generate Phase-1 training data for the PuzzleScript game ps:weird_bug
("Weird Bug", Jonah Ostroff).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a NATIVE model of the
mechanic (fuzz-verified against the interpreter), a RELAXED model that abstracts
the web sheet away, the searches those two make possible, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_weird_bug",
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

THE GAME, AND THE JOKE IT IS PLAYING
------------------------------------
"Weird Bug" is a deliberately half-finished game: the author ships it as a
puzzle-game-shaped bug report ("Please fix any bugs you find with the HACK link
at the bottom of the page"). Several of its levels are broken ON PURPOSE, and
the interesting part of writing a solver for it is that "this level cannot be
won" has to be PROVED rather than assumed -- see `--proof`.

You are a magenta bug in a walled maze. You walk, you shove things, and you win
by standing on the Goal. Three mechanics carry the whole game:

* **Switches and doors are one global latch.** Four `late` rules put a `Temp`
  marker on every Door, strip it again if any Switch is UNWEIGHTED, and -- if a
  marker survives -- delete every Switch and every Door on the board. So the
  doors do not open one at a time and they do not close again: the instant
  EVERY switch is simultaneously covered by a Weight (Player, Pebble or Gem,
  *not* Ant), every door in the level is gone permanently. A level with doors
  and no switches opens on its first turn (measured). A level with more
  switches than weights can never open its doors at all, which is the whole of
  level 6's bug.
* **Webs are rope, and they make the crates a RIGID BODY.** X drops or lifts a
  web on the square you are standing on. A web on its own does nothing at all:
  it is on its own collision layer, you walk over it, it blocks nothing. But
  `[> Sticky Web]`, `[> Web Sticky]` and `[Moving Web|Web]` are a grouped rule
  block, so the moment you shove a Pebble/Gem/Ant that is STANDING ON a web,
  the force floods the whole 4-connected sheet of webs and every sticky thing
  standing on any of them moves too -- in your direction, across the board,
  through walls' worth of distance. That is how you move a crate you cannot
  reach, and it is the only way to move anything out of the corners levels 2
  and 3 park them in.
* **Fire ants are static, silent and instantly lethal.** Nothing moves an ant
  on its own -- there is no AI rule in the file. `late [Player|Ant] ->
  [DeadBug|Ant]` simply says that ENDING a turn orthogonally adjacent to one
  kills you. So an ant is a fixed 5-cell exclusion zone, and the only way past
  one is to drag it away on a web sheet.

Everything below was MEASURED against the interpreter, not read off the .txt.

* **`noaction` is declared in the prelude and the interpreter ignores it.** The
  file says `noaction (Disables X key; for testing early levels.)`, which in
  real PuzzleScript would kill the X key -- and with it the web mechanic that
  levels 3 and up are built around, and the message "Press X to create or
  remove webs" that introduces it. This interpreter does not implement the
  prelude flag, so X works and the game is playable as designed. Worth knowing
  because it is a case where the adapter is KINDER than the original.
* **A press into a Wall or a Door is a CANCEL, and a cancelled turn runs no
  late rules.** `[> Blockable|Blocking] -> Cancel` aborts the whole turn, so
  bumping a wall is not a turn: the doors do not latch on it and an ant
  standing next to you does not kill you on it. Measured both ways round. A
  press off the GRID EDGE is not a cancel -- the mover simply stays and the
  late rules do run -- but every playable level is walled, so that branch is
  unreachable here and is in the model only to make it total.
* **The Cancel is global and it is what stops a sheet.** If ANY moving
  Blockable -- the player, a crate, or a bare web square -- has a wall in front
  of it, nothing on the board moves. A four-square sheet with one web against a
  wall is frozen solid until you walk over and lift that web.
* **Pebble and Gem are the same object.** `Pushable`, `Sticky` and `Weight` all
  list both, the "magic gem rules are missing" (the author says so in a comment
  where they should be), and nothing else reads Gem. So the gem level 8
  introduces as the answer to its ants is an ordinary crate with a different
  sprite, which is exactly why level 8 cannot be won.
* **Pushing an ant is suicide and cannot be used.** The push rule moves you
  INTO the square the ant just left, so you end the turn adjacent to it. And to
  push it you had to already be adjacent at the START of the turn, which you
  cannot survive. A pebble shoved into an ant chains through it safely (the
  pebble ends up between you and the ant) -- that, and the web drag, are the
  only ways an ant ever moves.
* **`run_rules_on_level_start` is False**, so the late rules have NOT fired
  when a level is presented: a level that starts with every switch covered
  still shows its doors until the first non-cancelled press.
* ACTION5 is the web key and is in the search on every level. Mouse actions are
  read by nothing.

WHAT IS SOLVED, AND WHAT IS PROVED IMPOSSIBLE
----------------------------------------------
Level indices are 0-BASED and count only playable boards -- the game's four
MESSAGE screens are not levels and the adapter does not present them, so level
0 here is the opening maze.

    L0  21x11 maze                 42 presses, PROVED SHORTEST
    L1  switches and one pebble    19 presses, PROVED SHORTEST
    L2  webs stick to pebbles      25 presses, PROVED SHORTEST
    L3  build your own web         37 presses, verified; lower bound 24
    L4  --                         PROVED UNWINNABLE: the level has NO GOAL
    L5  --                         PROVED UNWINNABLE: the level has NO PLAYER
    L6  --                         PROVED UNWINNABLE: 3 switches, 2 weights
    L7  --                         unsolved, and NOT proved either way
    L8  --                         PROVED UNWINNABLE: the ants seal the goal
    L9  --                         PROVED UNWINNABLE: the ant seals the goal

Four of the five impossible levels are the author's own annotations coming
true. L4 carries the comment `(Missing goal.)` and indeed contains no Goal
object, so `All Player on Goal` is false for as long as a player exists. L5
carries `(Not sure where the player should start.)` and contains no Player at
all -- in real PuzzleScript that WINS the level vacuously, but the adapter
refuses a win with no player on the board (`_check_single_win_condition`, the
"a deleted player is a broken state" guard), so the level is inert: its whole
reachable state space is ONE state. L6 has three switches and only two weights
(you and one pebble), and its goal is walled in behind the door those switches
open. L8 is the level whose message is "Use the magic gem to get past the
ants!" -- and the magic gem rules are the ones the author left out, so the gem
is a crate and the two ants either side of the goal corridor cannot be shifted.

That same guard is why DYING is not a shortcut here. In the shipped ruleset a
corpse deletes the last Player and `All Player on Goal` goes vacuously true, so
every level with an ant could be "won" by walking into it; the adapter blocks
it, so death is simply terminal and the searches treat it as a dead state.

HOW THE LEVELS ARE PROVED IMPOSSIBLE (`--proof`)
-------------------------------------------------
A search that fails proves nothing, and the real state space here is not
enumerable: X toggles a web on the square you are standing on, so the web set
alone is 2^(free cells) and a plain BFS drowns in it (measured: 1.2M states at
depth ~12 on every unsolved level).

So the proof runs on a RELAXATION in which the web set is thrown away. A
relaxed state is just ``(player, weights, ants, doors_open)`` and a relaxed
press may drag ANY chain-closed set of pushables that lies in the same
component of `SAFE(d)` -- the squares a web could occupy and still move one
step in direction `d`. That is an over-approximation of the real game for a
reason that is exactly the mechanic: a real drag needs a connected sheet
through those squares, and the relaxation grants the sheet for free. Every real
press maps to a legal relaxed press and every real X press maps to nothing, so

    relaxed_dist(project(s))  <=  real_dist(s)          for every state s

which makes the relaxed field BOTH a proof of impossibility (no win is
reachable in a space that contains everything the real game can do) AND an
admissible A* heuristic for the real model. The five impossible levels close in
between 1 and 195,271 relaxed states. It also hands the searches a free
deadlock test: a real state whose projection cannot reach a relaxed win is
dead, and is dropped rather than expanded.

Level 7 is the one the relaxation does not settle. It has sixteen pushables --
two pebbles and fourteen ants in two solid rows -- and its relaxed space does
not close inside the cap, so this generator makes NO claim about it. What is
measured is narrower and stated as such: its web-free space is 6048 states,
fully enumerated, and contains no win. Structurally the level asks you to weigh
down a switch at (6,9) whose only two neighbours are ants, along a row where
every square but (6,1) is lethal, so it needs an ant DRAGGED off its square by
a sheet you have to build from four squares away.

THE SEARCH
----------
Plans come from A* over the native model with the relaxed field as `h`.

* **`weight == 1` means PROVED SHORTEST.** The heuristic is admissible, so an
  A* that terminates without hitting its node cap returns an optimal plan --
  which is where L0/L1/L2's 42/19/25 come from, and where their exact
  optimal-action SETS come from too (a press is optimal iff re-solving from the
  state it lands in costs exactly one less).
* **L3 needs a weight.** Its answer is 37 presses against a lower bound of 24,
  and admissible A* does not close that gap in a search this machine can hold:
  measured, it expands 1.54M states in 46s and then runs out of its 6M
  generated-state cap. The ladder escalates to w=1.2, which finds the 37-press
  plan in 1.42M expansions and 43s, and the plan is reported as VERIFIED but
  not shortest. Its steps are labelled with the press the expert took, since no
  exact tie set is available for them -- never `None`.
* **"Proved shortest" is decided by ``len(plan) == the relaxed lower bound``**
  rather than by which rung produced the plan, so a cold process reading a
  cached plan reaches the same verdict as the one that derived it. It is
  conservative in the safe direction (see `proved_shortest`), and here the
  three shortest plans sit exactly on their bounds: 42, 19 and 25.

Plans are cached in ``data/weird_bug_plans.json``: they are seed-independent
(the engine state after reset is the same for every seed, only the presentation
is augmented), and L3's search is the whole cost of a cold start.

RENDERING (`--audit`), AND WHY THE .TXT IS PATCHED
---------------------------------------------------
Four sprites in `data/puzzlescript_games/Weird_Bug.txt` are edited, each with
the reason written into the file beside it. Three of them were collapses that
made the puzzle unreadable:

* **The Goal's sprite was five blank rows** -- `(Finalize graphics below.)` --
  so the win condition was INVISIBLE in every frame of every level. It now
  paints (0,0), (0,2), (1,0), (3,0) and (4,0): (0,0) is a hole in Player,
  Pebble, Gem, Ant *and* Web, so the goal reads through anything that can stand
  on it, at every cell size the levels use.
* **The Switch's ring covered rows/cols 1-3 only**, which is precisely where
  Player, Pebble and Gem are opaque -- so a switch that was already weighed
  down was pixel-identical to bare floor, and "how many switches are still
  uncovered" is the entire mechanic of five levels. It gains four show-through
  pixels, disjoint from the Goal's.
* **The Web's checkerboard was opaque at every even/even pixel**, and those are
  the only nine a `cell_px == 3` cell samples, so a web hid whatever was under
  it completely. Two corners are punched out; it still paints 7 of those 9.
* **All four Background shades collapse to palette index 5**, which is also the
  colour the renderer letterboxes with, so the floor and out-of-bounds were the
  same colour. One shade is lifted to `#333333` (index 4) to give the play area
  a texture the pad does not have.

A fourth edit is a pure geometry fix that only `--audit` could have found: the
Web's row 3 has its parity flipped, because Player and DeadBug are opaque at
exactly the even/even pixels a `cell_px == 4` cell samples, so on the four 4px
boards a web UNDER the player was invisible. The six show-through slots that
size offers are then split three ways -- Goal (0,0)/(1,0), Switch
(0,4)/(1,4), Web (3,0)/(3,4) -- which is what makes all six of
floor/goal/switch and their +web variants readable under any cover.

`--audit` then asserts every cell COMPOSITION a settled frame can show is
distinct, at every board size in the game, rendering whole uniform frames
rather than indexing into a mixed one. It passes at all nine board shapes,
covering cell sizes 3, 4, 5, 6 and 7.

The game takes ROTATION augmentation only (all four turns, verified to vary
with the seed) and no flips: the adapter's `_FLIP_GAMES` is an opt-in list and
this game's Player, DeadBug, Ant and Pebble sprites are left-right symmetric
but not top-bottom, so a vertical mirror has no partner sprite to map onto.

VERIFICATION
------------
``--selfcheck``  the native model against the interpreter, press for press, on
                 the shipped levels AND on randomly seated boards (which is
                 what covers the web/cancel/off-grid branches the levels do not
                 reach).
``--proof``      the relaxed field per level: its size, the lower bound it
                 gives, and the impossibility verdict where it closes with no
                 win reachable.
``--plans``      per-level plan, length, budget headroom and tie coverage.
``--ties``       re-derives every optimal-action label by re-solving from each
                 successor, independently of the field that produced it.
``--bfs``        the exhaustive WEB-FREE state space per level (the narrow
                 claim level 7 gets).
``--audit``      the render check described above.
``--symmetry``   the same plan replayed at every (seed, level) augmentation.
``--recovery``   re-plan from perturbed mid-episode boards.
``--speed``      wall clock per episode.

As measured on this machine:

    --selfcheck   36,000 presses over the shipped levels + 37,500 over 1,500
                  randomly seated boards, 0 mismatches
    --proof       nine levels' relaxed fields, five of them closing with no
                  win reachable; level 7 reported as not claimed
    --plans       4 levels, 123 presses, 8 steps with a tie, 0 unlabelled
    --ties        86 labels re-derived by re-solving, 0 mismatches
    --audit       9 board shapes, 0 indistinguishable compositions
    --symmetry    24 (seed, level) pairs replayed to WIN, 0 failures
    --recovery    160 perturbations: 155 re-planned to a WIN, 1 PROVED lost,
                  4 over the reduced ladder, 0 wrong
    --speed       ~1.2s per episode from a warm plan cache, 4 levels, ~135
                  frames

and end to end, six scratch episodes: byte-identical between two runs, every
action index in 0..5, every observation in the 0..15 palette, and 0 unlabelled
EXPERT steps out of 738 (the 65 unlabelled are the exploration prefix, which
the closing RESET discards).

Corpus generation is the user's to run:

    /home/simon/anaconda3/envs/ARC-AGI-3/bin/python \\
        solvers/generate_weird_bug_training.py --episodes N \\
        --out data/training_multi_level/weird_bug
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                          # noqa: E402
    Plan, PSAStarSolver, PSExpert,
)

GAME_NAME = "Weird_Bug"

#: Seed-independent plan cache. L3's search is the whole cost of a cold start.
PLAN_CACHE = Path(__file__).resolve().parent.parent / "data" / "weird_bug_plans.json"

#: The five presses, in the order the tie sets list them. ACTION5 is the web
#: key, so it is in the branching on every level.
_DIRS = ("up", "down", "left", "right", "action")

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Objects read off the engine grid into `_Static` / a model state.
_OBJECTS = ("wall", "goal", "switch", "door", "player", "pebble", "gem",
            "ant", "web", "deadbug")

#: A* escalation ladder. ``1`` is admissible -- a plan found at weight 1 is
#: PROVED SHORTEST -- and the rest trade that away for a search that closes.
#: Each rung gets its own cap because the point of escalating is that the rung
#: below ran out of ROOM, not out of patience.
#:
#: The cap counts GENERATED states, not expansions, which is the number that
#: has to fit in memory: level 3 reaches its 37-press answer after 1.4M
#: expansions and about 4.7M generated states, so a 3M cap silently reports the
#: level unwinnable. The interning in `astar` is what makes 8M affordable
#: (~2.6 GB peak instead of ~10 GB).
_LADDER = ((1, 6_000_000), (1.2, 8_000_000), (1.5, 8_000_000),
           (2.0, 8_000_000), (3.0, 8_000_000), (5.0, 8_000_000))

#: Runaway guards on the relaxed enumeration. Level 9 -- the largest that
#: closes -- reaches 195,271 states in ~2s; level 7 closes at neither bound,
#: and the point of having both is that it must SAY SO rather than run for an
#: hour. A guard that trips is reported as "nothing is claimed", never as a
#: proof: an enumeration is only evidence when it terminates on its own.
_RELAX_CAP = 1_500_000
_RELAX_SECONDS = 90.0


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Static:
    """A level's immovable geometry: walls, goals, switches and doors.

    Doors and switches are static POSITIONS rather than dynamic objects even
    though the late rules delete them, because they are deleted together and
    exactly once -- the ``doors_open`` flag in the state says whether that has
    happened. Nothing in the ruleset ever creates one.
    """

    __slots__ = ("h", "w", "walls", "goals", "switches", "doors")

    def __init__(self, h, w, walls, goals, switches, doors):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.goals = frozenset(goals)
        self.switches = frozenset(switches)
        self.doors = frozenset(doors)

    def sig(self):
        return (self.h, self.w, tuple(sorted(self.walls)),
                tuple(sorted(self.goals)), tuple(sorted(self.switches)),
                tuple(sorted(self.doors)))


def _at(p, d, h, w):
    """``p + d`` inside the grid, or None when it falls off the edge."""
    r, c = p[0] + d[0], p[1] + d[1]
    return (r, c) if 0 <= r < h and 0 <= c < w else None


def _read(eng, ids):
    """``(static, state)`` for the engine's current grid.

    A model state is ``(player, pebbles, gems, ants, webs, doors_open, dead)``.
    Pebbles and Gems are kept apart even though they are mechanically the same
    object, so the state maps back onto the grid exactly for `--selfcheck`;
    every rule that reads either one reads the union.
    """
    cells = {n: set() for n in _OBJECTS}
    for r in range(eng.height):
        for c in range(eng.width):
            cell = eng.grid[r][c]
            for name in _OBJECTS:
                if ids[name] in cell:
                    cells[name].add((r, c))
    static = _Static(eng.height, eng.width, cells["wall"], cells["goal"],
                     cells["switch"], cells["door"])
    player = next(iter(cells["player"]), None)
    dead = next(iter(cells["deadbug"]), None)
    state = (player, frozenset(cells["pebble"]), frozenset(cells["gem"]),
             frozenset(cells["ant"]), frozenset(cells["web"]), False, dead)
    return static, state


def _late(st, static):
    """The four door rules and the ant rule, in file order.

    The door block is `late [Door] -> [Temp Door]`, then strip the marker for
    every UNCOVERED switch, then (if a marker survived) delete every switch,
    then delete every door -- i.e. one global latch that fires the first time
    every switch carries a Weight, and never un-fires. A level with doors and
    no switches latches on its first turn, which `all()` over an empty set
    gives for free and which was measured.

    The ant rule runs AFTER the door block, so the weight of a player who is
    about to be eaten still counts towards opening the doors.
    """
    player, pebbles, gems, ants, webs, doors_open, dead = st
    if static.doors and not doors_open:
        weight = pebbles | gems
        if all(s in weight or s == player for s in static.switches):
            doors_open = True
    if player is not None:
        for d in _DELTA.values():
            t = _at(player, d, static.h, static.w)
            if t is not None and t in ants:
                dead, player = player, None
                break
    return (player, pebbles, gems, ants, webs, doors_open, dead)


def _step(static, st, press):
    """One press. Returns the settled state, or ``st`` itself when the turn was
    CANCELLED -- and a cancelled turn runs no late rules, which is measured and
    is not a detail: it is why bumping a wall cannot be used as a wait.
    """
    player, pebbles, gems, ants, webs, doors_open, dead = st
    h, w = static.h, static.w

    if press == "action":
        # `[Action Player no Web] -> [Player Web]` / `[Action Player Web] ->
        # [Player]`: X toggles a web on the square you are standing on. The two
        # rules do not undo each other -- the second does not see the web the
        # first just made (measured).
        if player is not None:
            webs = webs ^ {player}
        return _late((player, pebbles, gems, ants, webs, doors_open, dead),
                     static)

    d = _DELTA[press]
    if player is None:
        return _late(st, static)

    # Pushable and Sticky are the same three objects, so one set serves both.
    push = pebbles | gems | ants
    moving_push, moving_web = set(), set()
    q = _at(player, d, h, w)
    if q is not None and q in push:
        moving_push.add(q)                      # [> Player | Pushable]

    # The grouped block, run to a fixpoint the way `+` rules are: a moving
    # sticky drags the web under it, a moving web drags the sticky on it, a
    # moving pushable shoves the pushable ahead of it, and a moving web drags
    # every web it touches. Every force in a turn points the same way, so
    # "moving" needs no direction of its own.
    changed = True
    while changed:
        changed = False
        for p in tuple(moving_push):
            if p in webs and p not in moving_web:
                moving_web.add(p); changed = True
            t = _at(p, d, h, w)
            if t is not None and t in push and t not in moving_push:
                moving_push.add(t); changed = True
        for p in tuple(moving_web):
            if p in push and p not in moving_push:
                moving_push.add(p); changed = True
            for dd in _DELTA.values():
                t = _at(p, dd, h, w)
                if t is not None and t in webs and t not in moving_web:
                    moving_web.add(t); changed = True

    # `[> Blockable|Blocking] -> Cancel`. Blockable is the player, the crates
    # AND the bare webs, so one web square facing a wall freezes the whole
    # board; Blocking is Wall and any door that has not been deleted yet.
    blocking = static.walls if doors_open else (static.walls | static.doors)
    for p in itertools.chain((player,), moving_push, moving_web):
        t = _at(p, d, h, w)
        if t is not None and t in blocking:
            return st

    # Off-grid blocking. NOT a cancel -- the mover simply stays and the late
    # rules run (measured on an unwalled test board). Every playable level is
    # walled so this is unreachable there; it is here to make the model total.
    stuck = set()
    changed = True
    while changed:
        changed = False
        for p in tuple(moving_push):
            if p in stuck:
                continue
            t = _at(p, d, h, w)
            if t is None or t in stuck:
                stuck.add(p); changed = True
        for p in tuple(moving_web):
            if _at(p, d, h, w) is None:
                moving_web.discard(p); changed = True
    moving_push -= stuck
    t = _at(player, d, h, w)
    player_moves = t is not None and t not in stuck

    def shifted(objs, sel):
        return frozenset(_at(p, d, h, w) if p in sel else p for p in objs)

    return _late((_at(player, d, h, w) if player_moves else player,
                  shifted(pebbles, moving_push), shifted(gems, moving_push),
                  shifted(ants, moving_push), shifted(webs, moving_web),
                  doors_open, dead), static)


def _won(static, st):
    """``All Player on Goal``, with the adapter's guard: a board with no player
    left is a broken state, never a win (`_check_single_win_condition`)."""
    return st[0] is not None and st[0] in static.goals


# ---------------------------------------------------------------------------
# The relaxation: the same game with the web sheet granted for free
# ---------------------------------------------------------------------------

class _Relax:
    """The over-approximation the impossibility proofs and the A* heuristic
    both run on.

    A relaxed state is ``(player, weights, ants, doors_open)``: the web set is
    gone and Pebble/Gem are merged (they are the same object). A relaxed press
    still needs the player to shove something -- force only ever originates at
    `[> Player | Pushable]` -- but it may then drag ANY chain-closed set of
    pushables lying in the same component of `SAFE(d)`.

    `SAFE(d)` is the set of squares a web may occupy and still move one step in
    direction `d`: not a wall, and with a non-wall, non-closed-door square
    ahead. A real drag needs a connected sheet of such squares joining the
    things it moves; the relaxation grants the sheet, so it admits everything
    the real game can do and more. Hence a relaxed win is not evidence, but a
    relaxed DEAD END is a proof.
    """

    def __init__(self, static):
        self.static = static
        self.safe, self.comp = {}, {}
        for name, d in _DELTA.items():
            for opened in (False, True):
                cells = self._safe_cells(d, opened)
                self.safe[(name, opened)] = cells
                self.comp[(name, opened)] = self._components(cells)

    def _safe_cells(self, d, opened):
        st = self.static
        block = st.walls if opened else (st.walls | st.doors)
        out = set()
        for r in range(st.h):
            for c in range(st.w):
                if (r, c) in st.walls:
                    continue
                t = _at((r, c), d, st.h, st.w)
                if t is not None and t not in block:
                    out.add((r, c))
        return out

    @staticmethod
    def _components(cells):
        comp, cid = {}, 0
        for start in cells:
            if start in comp:
                continue
            comp[start] = cid
            queue = deque([start])
            while queue:
                p = queue.popleft()
                for d in _DELTA.values():
                    t = (p[0] + d[0], p[1] + d[1])
                    if t in cells and t not in comp:
                        comp[t] = cid
                        queue.append(t)
            cid += 1
        return comp

    def late(self, st):
        player, weights, ants, doors_open = st
        static = self.static
        if static.doors and not doors_open:
            if all(s in weights or s == player for s in static.switches):
                doors_open = True
        if player is not None:
            for d in _DELTA.values():
                t = _at(player, d, static.h, static.w)
                if t is not None and t in ants:
                    player = None
                    break
        return (player, weights, ants, doors_open)

    def succ(self, st):
        """Every relaxed successor. The `action` press is kept as a pure
        late-rule tick: `run_rules_on_level_start` is False, so the very first
        press of a level can latch the doors without moving anything, and
        dropping X entirely would miss that."""
        player, weights, ants, doors_open = st
        static = self.static
        out = [self.late(st)]
        if player is None:
            return out
        block = static.walls if doors_open else (static.walls | static.doors)
        push = weights | ants
        for name, d in _DELTA.items():
            q = _at(player, d, static.h, static.w)
            if q is None:
                out.append(self.late(st))            # off-grid: nothing moves
                continue
            if q in block:
                continue                             # CANCEL: no late rules
            if q not in push:
                out.append(self.late((q, weights, ants, doors_open)))
                continue
            safe = self.safe[(name, doors_open)]
            comp = self.comp[(name, doors_open)]
            if q not in safe:
                continue                             # the shoved one is stuck
            cid = comp[q]
            # Chains along `d`, over the pushables sharing q's component. A
            # chain may only move as a SUFFIX (everything ahead of a mover is
            # shoved too) and only if its tail has somewhere to go.
            chains = []
            for head in push:
                if (head not in safe or comp.get(head) != cid
                        or (head[0] - d[0], head[1] - d[1]) in push):
                    continue
                chain, p = [], head
                while p in push:
                    chain.append(p)
                    p = (p[0] + d[0], p[1] + d[1])
                if all(x in safe and comp.get(x) == cid for x in chain):
                    chains.append(chain)
            mine = next((ch for ch in chains if q in ch), None)
            if mine is None:
                continue
            bases = [mine[i:] for i in range(mine.index(q) + 1)]
            extras = [()]
            for chain in chains:
                if chain is mine:
                    continue
                extras = [e + tuple(s) for e in extras
                          for s in [()] + [chain[i:] for i in range(len(chain))]]
                if len(extras) > 4096:
                    break
            for base in bases:
                for extra in extras:
                    moved = set(base) | set(extra)
                    nw = frozenset(_at(p, d, static.h, static.w) if p in moved
                                   else p for p in weights)
                    na = frozenset(_at(p, d, static.h, static.w) if p in moved
                                   else p for p in ants)
                    if (len(nw) + len(na) != len(weights) + len(ants)
                            or nw & na):
                        continue
                    out.append(self.late((q, nw, na, doors_open)))
        return out


def project(st):
    """A model state seen as a relaxed one."""
    player, pebbles, gems, ants, _webs, doors_open, _dead = st
    return (player, pebbles | gems, ants, doors_open)


def relax_field(relax, start, cap=_RELAX_CAP, seconds=_RELAX_SECONDS):
    """``(dist, n_states)``: distance-to-win for every relaxed state reachable
    from ``start``, or ``(None, n)`` when the enumeration hits either guard.

    ``dist`` missing a state means that state cannot reach a win even with the
    sheet granted for free -- which is the impossibility proof when it is the
    start state, and a sound deadlock test for every other state.
    """
    static = relax.static
    deadline = time.time() + seconds
    edges = {start: None}
    queue = deque([start])
    ticks = 0
    while queue:
        s = queue.popleft()
        ticks += 1
        if ticks % 4096 == 0 and time.time() > deadline:
            return None, len(edges)
        out = relax.succ(s)
        edges[s] = out
        for t in out:
            if t not in edges:
                edges[t] = None
                queue.append(t)
                if len(edges) > cap:
                    return None, len(edges)
    pred = {}
    for s, out in edges.items():
        for t in out or ():
            pred.setdefault(t, []).append(s)
    dist, queue = {}, deque()
    for s in edges:
        if s[0] is not None and s[0] in static.goals:
            dist[s] = 0
            queue.append(s)
    while queue:
        s = queue.popleft()
        for p in pred.get(s, ()):
            if p not in dist:
                dist[p] = dist[s] + 1
                queue.append(p)
    return dist, len(edges)


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class WeirdBugExpert(PSExpert):
    """A* over the native model, `h` = the relaxed distance-to-win.

    The base class's engine-blackbox A* is replaced wholesale (`_search`): an
    interpreter step is ~350us and these searches run to millions of states, so
    they run on the fuzz-verified native model instead and only the plan comes
    back to the engine. Everything else -- the memo, the disk cache, the
    restore discipline, the recording -- is the base's.
    """

    directions = list(_DIRS)
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        self.ids = {n: self.g.obj_name_to_idx[n] for n in _OBJECTS}
        #: static signature -> (relax, dist). Keyed on the geometry rather than
        #: the level index so a re-plan from a mid-level state reuses it.
        self._fields: dict = {}

    # -- model <-> engine ----------------------------------------------------
    def read(self, eng):
        return _read(eng, self.ids)

    def field(self, static):
        """The relaxed field for this geometry, built once and memoized."""
        key = static.sig()
        if key not in self._fields:
            relax = _Relax(static)
            self._fields[key] = (relax, None, 0)
        relax, dist, n = self._fields[key]
        return relax, dist, n

    def ensure_field(self, static, state):
        key = static.sig()
        relax, dist, n = self.field(static)
        if dist is None:
            dist, n = relax_field(relax, project(state))
            self._fields[key] = (relax, dist, n)
        return dist, n

    # `_key` is the base's: every non-background cell, walls and goals
    # included. It is canonical ACROSS levels (so `scope_by_level` stays off
    # and two levels can never serve each other's plans), it is what the disk
    # cache stores its start signature from, and on boards of at most 231 cells
    # the extra static entries cost nothing. A narrower dynamic-only key would
    # have to re-derive "have the doors been deleted, or did this level never
    # have any" -- a distinction the full key makes for free.

    # -- the search ----------------------------------------------------------
    def astar(self, static, dist, start, weight, cap):
        """A* on the native model. ``weight == 1`` is admissible, so a plan it
        returns is SHORTEST; anything above trades that for a search that
        closes. Returns ``(plan, expansions)``."""
        import heapq

        def h(s):
            return dist.get(project(s))

        if _won(static, start):
            return [], 0
        h0 = h(start)
        if h0 is None:
            return None, 0
        openq = [(weight * h0, 0, 0, start)]
        best = {start: 0}
        parent = {start: None}
        expanded = tie = 0
        # The four object sets of a state are what the search stores millions
        # of, and successive states share almost all of them -- the ants never
        # move on most levels, the crates move once every several presses.
        # Interning them means `best` holds a reference where it would
        # otherwise hold a distinct frozenset per state; on level 3 that is the
        # difference between the search fitting in memory and not.
        pool: dict = {}

        def keep(st):
            p, b, g, a, w, o, d = st
            return (p, pool.setdefault(b, b), pool.setdefault(g, g),
                    pool.setdefault(a, a), pool.setdefault(w, w), o, d)

        while openq:
            _f, gs, _t, s = heapq.heappop(openq)
            if gs > best.get(s, 1 << 30):
                continue
            expanded += 1
            for press in _DIRS:
                nxt = _step(static, s, press)
                if nxt == s:
                    continue                    # a no-op press wastes a move
                nxt = keep(nxt)
                ng = gs + 1
                if ng >= best.get(nxt, 1 << 30):
                    continue
                hn = h(nxt)
                if hn is None:
                    continue                    # relaxed-dead => really dead
                best[nxt] = ng
                parent[nxt] = (s, press)
                if _won(static, nxt):
                    out, cur = [], nxt
                    while parent[cur] is not None:
                        prev, pressed = parent[cur]
                        out.append(pressed)
                        cur = prev
                    return list(reversed(out)), expanded
                tie += 1
                heapq.heappush(openq, (ng + weight * hn, ng, tie, nxt))
            if len(best) > cap:
                return None, expanded
        return None, expanded

    def solve(self, static, state, ladder=_LADDER):
        """``(plan, weight, expansions)`` -- climb the ladder until one rung
        returns a plan. ``weight == 1`` means the plan is proved shortest.

        A None plan means one of two very different things and the caller has
        to know which: the relaxed field can say the state is DEAD (proof), or
        every rung can run out of room (no claim). `dead` answers the first
        question on its own, instantly, which is what `--recovery` uses to keep
        the two apart -- and why it can afford a much smaller ladder.
        """
        dist, _n = self.ensure_field(static, state)
        if dist is None or project(state) not in dist:
            return None, None, 0
        total = 0
        for weight, cap in ladder:
            plan, expanded = self.astar(static, dist, state, weight,
                                        min(cap, self.node_cap))
            total += expanded
            if plan is not None:
                return plan, weight, total
        return None, None, total

    def dead(self, eng) -> bool:
        """True when the engine's state cannot reach a win -- PROVED, not
        guessed: it is the relaxed field's verdict, and the relaxation admits
        everything the real game can do. Walking into an ant is the common
        case, a crate shoved into a corner the other one."""
        static, state = self.read(eng)
        dist, _n = self.ensure_field(static, state)
        return dist is not None and project(state) not in dist

    def proved_shortest(self, static, state, plan) -> bool:
        """True when ``plan`` is PROVED to be a shortest win.

        The test is ``len(plan) == relaxed lower bound``, which is sound
        (nothing can beat the relaxation) and, unlike "the ladder returned it
        at weight 1", costs nothing and can be applied to a plan read back off
        the disk cache -- so `--plans` and `--ties` reach the same verdict in a
        cold process as `_search` did in the one that derived it. It is
        conservative in the safe direction: a weight-1 plan LONGER than the
        bound would be optimal and reported as merely verified, which loses tie
        sets and never claims one that is wrong. It does not arise here (42,
        19 and 25 all sit exactly on the bound).
        """
        dist, _n = self.ensure_field(static, state)
        if dist is None:
            return False
        return dist.get(project(state)) == len(plan)

    def _search(self, eng):
        static, state = self.read(eng)
        plan, _weight, _n = self.solve(static, state)
        if plan is None:
            return None
        exact = self.proved_shortest(static, state, plan)
        return Plan(plan, self.optsets(static, state, plan, exact))

    def optsets(self, static, state, plan, exact):
        """Per-step optimal SETS.

        When the plan came from admissible A* the sets are MEASURED: a press is
        optimal exactly when re-solving from the state it lands in costs one
        less than what is left. When it came from a weighted rung there is no
        distance to compare against, so the step is labelled with the press the
        expert took -- a target on every step, never `None` (a step with no
        target contributes nothing to the loss and silently trains nothing).
        """
        dist, _n = self.ensure_field(static, state)
        out, cur = [], state
        for i, taken in enumerate(plan):
            if not exact:
                out.append([taken])
                cur = _step(static, cur, taken)
                continue
            remaining = len(plan) - i
            best = []
            for press in _DIRS:
                nxt = _step(static, cur, press)
                if nxt == cur:
                    continue
                if _won(static, nxt):
                    cost = 1
                else:
                    h = dist.get(project(nxt))
                    if h is None or h > remaining - 1:
                        continue                 # sound prune: admissible h
                    sub, weight, _n = self.solve(static, nxt)
                    cost = None if sub is None or weight != 1 else 1 + len(sub)
                if cost == remaining:
                    best.append(press)
            out.append(best if taken in best else [taken])
            cur = _step(static, cur, taken)
        return out


class WeirdBugSolver(PSAStarSolver):
    game_id = "puzzlescript_weird_bug"
    game_name = GAME_NAME
    expert_cls = WeirdBugExpert

    #: Record against the GAME FOLDER's adapter. ``games/ps:weird_bug`` is a
    #: plain passthrough today; naming it means a sprite patch or a step cap
    #: added there later cannot silently make this generator tape a game nobody
    #: plays (the ps:count_mover trap).
    game_module_id = "ps:weird_bug"

    #: Level 7 is the one level neither solved nor proved impossible, and its
    #: searches would burn the whole ladder at every startup. It is skipped up
    #: front; `--proof` says exactly what is and is not claimed about it.
    skip_levels = frozenset({7})

    #: The ladder's own caps bound each rung; this is the outer guard.
    node_cap = 4_000_000

    #: Room for the longest plan (42) plus the RESET exploration prefix and the
    #: re-plan after it, inside the adapter's 200-press per-level budget.
    max_steps = 90


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0, skip: bool = True):
    solver = WeirdBugSolver()
    game = solver.make_game(seed)
    expert = WeirdBugExpert(game, node_cap=WeirdBugSolver.node_cap)
    return solver, game, expert


def _levels(game, skip=True):
    bad = WeirdBugSolver.skip_levels if skip else frozenset()
    return [lv for lv in range(game.n_levels) if lv not in bad]


def _plans() -> int:
    """Per-level plan, length, budget headroom and tie coverage."""
    _solver, game, expert = _new()
    eng = game._engine
    total = tied = 0
    for level in _levels(game):
        game.set_level(level)
        static, state = expert.read(eng)
        t0 = time.time()
        plan = expert.plan(eng, level)
        dt = time.time() - t0
        head = (f"level {level}: {eng.height:2d}x{eng.width:2d}, "
                f"{len(state[1])} pebble(s), {len(state[2])} gem(s), "
                f"{len(state[3])} ant(s), {len(static.switches)} switch(es), "
                f"{len(static.doors)} door(s), {len(static.goals)} goal(s)")
        if plan is None:
            print(f"{head} -- NO PLAN (see --proof)")
            continue
        sets = getattr(plan, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        tied += ties
        exact = expert.proved_shortest(static, state, plan)
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(plan):3d} presses "
              f"({'PROVED SHORTEST' if exact else 'verified, not shortest'}, "
              f"budget {game._max_steps} {room}), {ties} tie set(s), "
              f"{dt:6.2f}s", flush=True)
        print(f"    {' '.join(plan)}", flush=True)
    print(f"total {total} presses, {tied} step(s) with a tie")
    return 0


def _proof() -> int:
    """The relaxed field per level: its size, the lower bound it gives, and the
    impossibility verdict where it closes with no win reachable.

    A closed enumeration with no reachable win is a PROOF that the level cannot
    be won, because the relaxation admits everything the real game can do (see
    `_Relax`). A closed enumeration WITH a win gives a lower bound on the real
    plan, and nothing more.
    """
    _solver, game, expert = _new(skip=False)
    eng = game._engine
    impossible = winnable = unclaimed = 0
    for level in _levels(game, skip=False):
        game.set_level(level)
        static, state = expert.read(eng)
        t0 = time.time()
        dist, n = expert.ensure_field(static, state)
        dt = time.time() - t0
        if dist is None:
            unclaimed += 1
            print(f"level {level}: relaxed space did not close -- {n:,} states "
                  f"in {dt:5.1f}s, over the {_RELAX_CAP:,}-state / "
                  f"{_RELAX_SECONDS:.0f}s guard. NOTHING is claimed either "
                  f"way; see --bfs for the narrow web-free result.", flush=True)
            continue
        lower = dist.get(project(state))
        if lower is None:
            impossible += 1
            why = _why_impossible(static, state)
            print(f"level {level}: {n:8,} relaxed states, {dt:5.1f}s -- "
                  f"PROVED UNWINNABLE ({why})", flush=True)
        else:
            winnable += 1
            print(f"level {level}: {n:8,} relaxed states, {dt:5.1f}s -- "
                  f"winnable, lower bound {lower} presses", flush=True)
    # Nothing here can "fail": an enumeration that closes is a result either
    # way and one that does not is an honest gap, so the exit status is 0 and
    # the counts are the report.
    print(f"{impossible} level(s) PROVED UNWINNABLE, {winnable} with a lower "
          f"bound, {unclaimed} where the relaxation did not close")
    return 0


def _why_impossible(static, state) -> str:
    """A human-readable GLOSS on an impossibility, read off the board.

    The proof is the closed enumeration the caller has already done -- this
    only says which of the board's features is the obvious culprit, so a reader
    does not have to go and look. Nothing depends on it being right.
    """
    player, pebbles, gems, ants, _webs, _open, _dead = state
    weights = len(pebbles) + len(gems) + (1 if player else 0)
    if not static.goals:
        return "the level contains no Goal"
    if player is None:
        return ("the level contains no Player, and the adapter refuses a "
                "vacuous win")
    if static.doors and len(static.switches) > weights:
        return (f"{len(static.switches)} switches but only {weights} weights, "
                f"so the doors can never open")
    if ants:
        return "the ants cannot be shifted off the only route to the goal"
    return "no win is reachable in the relaxation"


def _bfs() -> int:
    """The exhaustive WEB-FREE state space per level.

    Dropping X from the branching makes every level's space small enough to
    enumerate outright, which is a real (if narrow) claim: a level with no win
    here cannot be won without building a web. It is the only measurement level
    7 gets, and it is reported as exactly that.
    """
    _solver, game, expert = _new(skip=False)
    eng = game._engine
    for level in _levels(game, skip=False):
        game.set_level(level)
        static, start = expert.read(eng)
        t0 = time.time()
        seen = {start}
        queue = deque([start])
        win = None
        depth = {start: 0}
        while queue:
            s = queue.popleft()
            for press in ("up", "down", "left", "right"):
                t = _step(static, s, press)
                if t in seen:
                    continue
                seen.add(t)
                depth[t] = depth[s] + 1
                if _won(static, t) and win is None:
                    win = depth[t]
                queue.append(t)
        verdict = (f"shortest web-free win {win} presses" if win is not None
                   else "NO web-free win exists")
        print(f"level {level}: {len(seen):6,} web-free states, "
              f"{time.time() - t0:5.1f}s -- {verdict}", flush=True)
    return 0


def _selfcheck(walks: int = 60, steps: int = 60, boards: int = 1500) -> int:
    """The native model against the interpreter, press for press.

    Two passes, and the second is the one that matters. Random walks on the
    SHIPPED levels cover what a plan touches; randomly SEATED boards -- random
    walls, crates, gems, ants, webs, switches and doors -- are what reach the
    web-sheet, cancel and off-grid branches the shipped levels never put a
    press through.
    """
    _solver, game, expert = _new(skip=False)
    eng = game._engine
    ids = dict(expert.ids)
    ids["background"] = game._game.obj_name_to_idx["background"]
    rng = random.Random(20260822)
    bad = presses = 0

    def compare(static, st):
        _s2, live = expert.read(eng)
        doors = {(r, c) for r in range(eng.height) for c in range(eng.width)
                 if ids["door"] in eng.grid[r][c]}
        switches = {(r, c) for r in range(eng.height) for c in range(eng.width)
                    if ids["switch"] in eng.grid[r][c]}
        want_d = set() if st[5] else set(static.doors)
        want_s = set() if st[5] else set(static.switches)
        return (live[0] == st[0] and live[1] == st[1] and live[2] == st[2]
                and live[3] == st[3] and live[4] == st[4] and live[6] == st[6]
                and doors == want_d and switches == want_s)

    for level in _levels(game, skip=False):
        for _walk in range(walks):
            game.set_level(level)
            static, st = expert.read(eng)
            for _t in range(steps):
                press = rng.choice(_DIRS)
                eng.step(press)
                st = _step(static, st, press)
                presses += 1
                if not compare(static, st):
                    print(f"  level {level}: model != engine after {press}")
                    bad += 1
                    break
    print(f"shipped levels: {presses} presses, {bad} mismatch(es)")

    seated = 0
    for _trial in range(boards):
        h, w = rng.randint(5, 9), rng.randint(5, 9)
        free = []
        grid = [[{ids['background']} for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if r in (0, h - 1) or c in (0, w - 1) or rng.random() < 0.15:
                    grid[r][c].add(ids["wall"])
                else:
                    free.append((r, c))
        rng.shuffle(free)

        def take(n, free=free):
            return [free.pop() for _ in range(min(n, len(free)))]

        for name, n in (("pebble", rng.randint(0, 4)), ("gem", rng.randint(0, 2)),
                        ("ant", rng.randint(0, 2)), ("door", rng.randint(0, 2)),
                        ("switch", rng.randint(0, 3)), ("goal", rng.randint(0, 1))):
            for p in take(n):
                grid[p[0]][p[1]].add(ids[name])
        if free:
            p = free.pop()
            grid[p[0]][p[1]].add(ids["player"])
        for r in range(h):
            for c in range(w):
                if ids["wall"] not in grid[r][c] and rng.random() < 0.3:
                    grid[r][c].add(ids["web"])
        eng.height, eng.width, eng.grid = h, w, grid
        eng._rule_win = eng._rule_restart = eng._late_cancel = False
        eng._position_index_dirty = True
        eng._force_rigid_group = {}
        eng._rule_noop_cache.clear()
        static, st = expert.read(eng)
        for _t in range(25):
            press = rng.choice(_DIRS)
            eng.step(press)
            st = _step(static, st, press)
            seated += 1
            if not compare(static, st):
                print(f"  seated board {_trial}: model != engine after {press}")
                bad += 1
                break
    print(f"seated boards: {boards} boards, {seated} presses")
    print("selfcheck clean" if not bad
          else f"SELFCHECK FAILED: {bad} mismatch(es)")
    return 0 if not bad else 1


def _ties() -> int:
    """Re-derive every optimal-action label independently: take each candidate
    press, re-solve from the state it lands in, and keep it iff ``1 + len(re-
    solve)`` equals what was left. This checks the labels against a fresh
    search from every state on the path rather than against the field that
    produced them. Weighted levels are skipped and said to be skipped -- there
    is no distance to compare against for them."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in _levels(game):
        game.set_level(level)
        static, state = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level}: no plan (skipped)")
            continue
        if not expert.proved_shortest(static, state, plan):
            print(f"level {level}: not proved shortest -- labelled with the "
                  f"press taken, so there is no exact set to check")
            continue
        cur, checked = state, 0
        for pi, (taken, claimed) in enumerate(zip(plan, plan.optsets)):
            remaining = len(plan) - pi
            measured = []
            for press in _DIRS:
                nxt = _step(static, cur, press)
                if nxt == cur:
                    continue
                if _won(static, nxt):
                    cost = 1
                else:
                    sub, weight, _n = expert.solve(static, nxt)
                    cost = (None if sub is None or weight != 1
                            else 1 + len(sub))
                if cost == remaining:
                    measured.append(press)
            if measured != list(claimed):
                print(f"  level {level} step {pi}: labelled {list(claimed)} "
                      f"but measured {measured}")
                bad += 1
            if taken not in claimed:
                print(f"  level {level} step {pi}: took {taken}, not in its "
                      f"own set {list(claimed)}")
                bad += 1
            cur = _step(static, cur, taken)
            checked += 1
        print(f"level {level}: {checked} step(s) verified")
    print("ties clean" if not bad else f"TIES FAILED: {bad} mismatch(es)")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every board size in the game.

    Whole 64x64 frames of UNIFORM boards are compared rather than one cell out
    of a mixed board: `_render_frame` upscales and centre-pads, so indexing a
    cell by ``cell_px * r`` silently reads the wrong pixels (the ps:explod
    lesson). Two uniform boards render identically iff their cells do.

    The compositions are everything a layer-4 object can be stacked on: bare
    floor, a Goal or a Switch, either of those under a Web, and each of those
    under the Player / a Pebble / a Gem / an Ant / the corpse. That is what the
    four sprite patches in the .txt are there to keep apart -- see the module
    docstring.

    DeadBug is only listed for boards that HAVE an ant, because it can only
    exist there: it shares the Player's sprite and its two body colours both
    collapse to palette 7 where the Player's split 6/7, and those differing
    pixels sit at (1,1)/(1,3)/(3,1)/(3,3) -- rows a ``cell_px == 3`` cell does
    not sample. The only 3px board in the game is level 0, which has no ant.
    """
    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, expert = _new(skip=False)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    floors = {"floor": (), "goal": ("goal",), "switch": ("switch",),
              "web": ("web",), "goal_web": ("goal", "web"),
              "switch_web": ("switch", "web")}
    covers = {"": (), "player": ("player",), "pebble": ("pebble",),
              "gem": ("gem",), "ant": ("ant",), "dead": ("deadbug",)}

    sizes: dict = {}
    ants: dict = {}
    for level in _levels(game, skip=False):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)
        _s, st = expert.read(eng)
        ants[(eng.height, eng.width)] = ants.get((eng.height, eng.width),
                                                 False) or bool(st[3])

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        comps = {}
        for fname, fobjs in floors.items():
            for cname, cobjs in covers.items():
                if cname == "dead" and not ants[(h, w)]:
                    continue
                comps[f"{fname}+{cname}" if cname else fname] = fobjs + cobjs
        comps["wall"] = ("wall",)
        comps["door"] = ("door",)
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        # The ps:stand_iii check: a composition that renders as nothing but the
        # letterbox pad is invisible against the frame's own border.
        pad = [n for n, s in shots.items()
               if len(set(s.flatten().tolist())) == 1]
        bad += len(clashes) + len(pad)
        note = "OK" if not (clashes or pad) else f"IDENTICAL {clashes} PAD {pad}"
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} "
              f"compositions -- {note}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable composition(s)")
    return 0 if not bad else 1


def _symmetry(seeds: int = 6) -> int:
    """Replay each level's plan at several seeds -- i.e. at several board
    rotations/flips/recolourings -- and assert every one reaches WIN.

    This is the rotation contract under test: the plans are derived in ENGINE
    space, and `screen_action` has to invert whatever the adapter forward-remaps
    before the press is driven or recorded. Get it backwards and nothing raises;
    three of every four orientations simply play a different game.
    """
    from arcengine import ActionInput, GameState

    from solvers.common.ps_astar import screen_action

    solver = WeirdBugSolver()
    bad = pairs = 0
    for seed in range(seeds):
        game = solver.make_game(seed)
        expert = WeirdBugExpert(game, node_cap=WeirdBugSolver.node_cap)
        for level in _levels(game):
            game.set_level(level)
            plan = expert.plan(game._engine, level)
            if plan is None:
                continue
            rot, hf, vf = game._rotation_k, game._hflip, game._vflip
            for press in plan:
                game.perform_action(ActionInput(
                    id=screen_action(press, rot, hf, vf,
                                     WeirdBugSolver.remap_actions)))
            pairs += 1
            if game._state != GameState.WIN:
                print(f"  seed {seed} level {level}: {game._state} "
                      f"(rot {rot}, hflip {hf}, vflip {vf})")
                bad += 1
    print(f"symmetry: {pairs} (seed, level) pair(s), {bad} failure(s)")
    return 0 if not bad else 1


#: The ladder `--recovery` re-plans on. Recording never pays this cost -- with
#: `epsilon` at 0 and RESET-mode recovery the only `plan` call is at a level's
#: start, which is a cache hit -- so the report exists to show the expert CAN
#: re-plan off-path, and a rung that would take a minute per perturbation would
#: only make the report unrunnable.
_RECOVERY_LADDER = ((1, 300_000), (1.5, 600_000), (3.0, 600_000))


def _recovery(trials: int = 40) -> int:
    """Re-plan from perturbed mid-episode boards.

    Walk each plan to a random point, take a random press off it, and ask the
    expert for a plan from wherever that left the board.

    A perturbed state falls into exactly one of three buckets, and keeping them
    apart is the point: it re-plans to a WIN, or it is PROVED lost (the relaxed
    field cannot reach a win from it -- walking into an ant, or shoving a crate
    into a corner, and this game really does have unrecoverable states), or the
    reduced ladder ran out of room, which is not a claim about the board. Only
    a plan that does NOT win is a failure.
    """
    _solver, game, expert = _new()
    eng = game._engine
    rng = random.Random(7)
    ok = proved_lost = over = wrong = 0
    for level in _levels(game):
        game.set_level(level)
        static, start = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        dist, _n = expert.ensure_field(static, start)
        for _t in range(trials):
            cur = start
            for press in plan[:rng.randrange(0, len(plan))]:
                cur = _step(static, cur, press)
            cur = _step(static, cur, rng.choice(_DIRS))
            if project(cur) not in dist:
                proved_lost += 1
                continue
            again, _w, _n = expert.solve(static, cur, _RECOVERY_LADDER)
            if again is None:
                over += 1
                continue
            check = cur
            for press in again:
                check = _step(static, check, press)
            if _won(static, check):
                ok += 1
            else:
                wrong += 1
    print(f"recovery: {ok} re-planned to a WIN, {proved_lost} PROVED lost, "
          f"{over} over the reduced ladder, {wrong} WRONG")
    return 0 if not wrong else 1


def _speed(seeds: int = 3) -> int:
    """Wall clock per episode, from a warm plan cache."""
    solver = WeirdBugSolver()
    for seed in range(seeds):
        t0 = time.time()
        ok, levels = solver.solve_episode(seed)
        frames = sum(len(lv["observations"]) for lv in levels)
        print(f"seed {seed}: ok={ok} {len(levels)} level(s), {frames} frames, "
              f"{time.time() - t0:5.1f}s")
    return 0


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--proof" in sys.argv:
        sys.exit(_proof())
    if "--bfs" in sys.argv:
        sys.exit(_bfs())
    if "--selfcheck" in sys.argv:
        sys.exit(_selfcheck())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--recovery" in sys.argv:
        sys.exit(_recovery())
    if "--speed" in sys.argv:
        sys.exit(_speed())
    sys.exit(WeirdBugSolver.main())
