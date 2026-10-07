"""ps:two_level_puzzle -- Weeble's "Two level puzzle": a sokoban played on TWO
elevations at once, where the thing you are standing on decides which of the two
worlds you are in and every crate is a piece of scaffolding.

THE ONE IDEA THE WHOLE GAME IS BUILT ON

    late [ HighPlayer no BlocksLow ] -> [ LowPlayer ] again
    BlocksLow = Crate or HighFloor or StaticIce or Pillar or BurntCrate

There is no "height" variable. There is a LowPlayer object and a HighPlayer
object, and the only thing that keeps the HighPlayer high is that something
solid happens to occupy its square on the LOW collision layer. Walk off the
edge of the high floor and the rule above swaps you for a LowPlayer -- but walk
onto a low square that happens to hold a CRATE and you stay high, standing on
the crate. So the crates in the low pit are the bridge the high world is missing,
and the puzzle is to build that bridge from underneath and then climb back up
and walk across it. `late [ HighCrate no BlocksLow ] -> [ Crate ]` is the same
rule for boxes: shove a high crate off the plateau and it FALLS, and what lands
is a low crate, i.e. one more paving stone.

The only way UP is `UP [ > LowPlayer StairsNorth | HighFloor ]`: stand on a
stairs square, press up, and a HighPlayer appears on the high floor above. The
only way DOWN is to walk off an edge. So a level is a set of low rooms and high
plateaus wired together by one-way falls and a handful of staircases, and
"which room am I in" is a state variable that the player changes by moving.

ICE IS A ONE-SHOT FIRE EXTINGUISHER, AND THE COUNT IS THE PUZZLE

    [ IceCube FirePit ] -> [ MeltingIce FirePit ] again
    [ MeltingIce FirePit ] -> [ ]

Pushed ice SLIDES (`RIGHT [IceCubeE] -> [> IceCubeE] again`) until something
stops it, and the two rules above are what stop it over lava: the tick after the
cube enters a FirePit square it becomes MeltingIce -- which is not MovingIce, so
the sliding rules no longer match and it halts on the spot -- and the tick after
that BOTH objects are deleted. So a cube is spent on the FIRST lava square it
touches and that square becomes plain floor forever. One cube, one lava square.
Level 1 is a player sealed inside a ring of twelve cubes inside a lava field two
squares thick, so getting out costs TWO cubes on the same LINE -- and the
solver's 18-press answer spends four: two punch the hole through row 3 to the
right-hand border, and the other two are spent purely to get the cubes that were
in the way OUT of the way, because the only thing that moves a cube off a square
is shoving it somewhere, and the only place there is to shove it is the lava.

Crates do not extinguish anything -- `late [ Crate FirePit ] -> [ BurntCrate
FirePit ]` -- but a BurntCrate is in BlocksLow, is not Pushable, and never moves
again: permanent scaffolding, bought with a crate you will not get back. Level 2's
plan chars one at (2,9), not on purpose but as the PRICE of a move: shoving that
crate three squares right is how the player gets past it into the right-hand
half of the board, and the third shove has nowhere to put it but the lava.

WHAT THE PRESS ACTUALLY DOES, AND WHY THE SEARCH IS OVER MACROS

Nothing in this game moves unless the player presses, ACTION is bound to no rule
at all, and a press moves the PLAYER at most one square (a slide moves the ice a
long way; the player still steps exactly once). So every press is one of

  * a WALK -- it changes nothing but the player's square and elevation. Walking,
    climbing a staircase and falling off a ledge are all walks;
  * an INTERACTION -- a push (crate, ice, high crate) or a coin pickup;
  * a no-op, or death by lava.

A shortest press sequence is therefore a shortest sequence of INTERACTIONS with
shortest walks between them, which is exactly a path in the macro graph "walk to
the square an interaction is taken from, then take it", costing `walk + 1`. That
substitution is sound in both directions: a walk changes nothing but the player,
so replacing any walk with a shortest walk to the same square leaves a plan of
the same length or shorter. `_Board.macros` builds that graph and every
rung of the ladder below runs on it.

THE SEARCH LADDER, AND WHAT EACH RUNG PROVES

1. EXACT macro A* over states `(world, player square, elevation)`, ordered by
   `_Board.lower_bound` -- which is admissible and consistent, so the first
   winning state POPPED is the shortest and an exhausted queue proves nothing
   shorter exists. Levels 0 and 1 fall to it outright, at 53 and 18 presses,
   both re-derived by an independent primitive-press BFS in `--ties`.
2. REGION A*, the same search with the player's square collapsed to its WALK
   REGION (`min` over the walk closure). Two states whose worlds agree and whose
   players can walk to each other are merged, which divides the state count by
   roughly the size of a room; the cost kept is the exact press cost of the
   representative, so what it returns is a real press sequence -- usually optimal
   and never guaranteed (level 0: 55 against the true 53; level 2: 89 against
   the bottleneck rung's 79).
3. The BOTTLENECK split, which is the rung that solves level 2 and the one worth
   reading. See `_Board.bottleneck`: the level's single ice cube is a consumable
   and spending it is irreversible, so every winning plan has two halves that
   want OPPOSITE things -- the first shoves crates out of the way to clear a lane
   for the cube, the second needs them exactly where they stand because they are
   the scaffolding the high player climbs. One frontier paying both penalties at
   once is why rungs 1 and 2 stall on it -- and so do an exhaustive
   primitive-press BFS abandoned at 8M states, beams up to 8000 wide, and 78000
   randomised greedy rollouts. Phase 1 is a weighted region A* with the support
   penalty switched OFF that STOPS the moment the cube is gone; each such state
   is handed to phase 2, one bounded EXACT search. Candidates are rare -- `lower_bound` has
   already reported every board that wasted the cube DEAD -- so the expensive
   half runs a handful of times.
4. A width-capped BEAM over the macro graph, the only rung that commits rather
   than keeping a frontier. Kept as a backstop; it has never been the rung that
   answers.

Then the ordering that turns a plan into a theorem: whatever rung produced it,
rung 1 is re-run with `bound = cost - 1`. "Exhausted" proves the plan shortest
without re-deriving it, "win" means the bounded search found something strictly
shorter and -- A* on an admissible heuristic popping in cost order -- that one IS
shortest, and "capped" means the budget ran out and nothing is proved. Every
report says which of the three happened, per level.

THE TWO HEURISTICS, AND WHY ONE OF THEM IS A THEOREM

Both are shortest paths for a GHOST player over `_Board._build_edges` -- someone
who walks through crates and ice as if they were not there, but still obeys
everything static (pillars block, the high floor is unreachable from below except
by a staircase, a fall is one-way). Every real press is one of those edges at
cost one, because a press moves the player at most ONE square: a slide carries
the cube a long way and the player still steps exactly once.

`lower_bound` charges nothing and is therefore admissible, and it is sharpened by
the one resource in the game. A lava square is cleared only by a cube, one cube
per square, and cubes are never created -- so with `k` cubes left, the true
distance is at least the ghost distance with at most `k` lava squares open. That
is evaluated exactly for `k <= 1` over the squares a cube could still REACH
(`_Board._openable`, which knows that a cube shoved into level 2's row 2 can
never leave it, because row 1 is high floor and no player can ever stand above or
below it there). Its infinity is an exact deadness test, and it is what makes the
bottleneck rung affordable: nearly every branch that spends the cube spends it
somewhere useless, and those are reported dead rather than searched.

`field` is the same ghost paying `FIRE_PEN` plus the distance from the nearest
cube to cross lava and `SUPPORT_PEN` plus the distance from the nearest crate to
stand on thin air. Those delivery distances are the whole point: without them the
estimate is FLAT across every crate shuffle in the game -- the ghost walks through
crates, so moving one changes nothing -- and a search guided by it has no gradient
at all on the only moves that matter. It is a ranking and never a bound.

OPTIMAL SETS, sound on every step and exact where the budget reaches

  * WALK steps carry the full tie set for free. A walk changes nothing but the
    player, so every shortest route to the square the next interaction is taken
    from reaches it in the same number of presses and leaves the identical world:
    "that press, the rest of a shortest route, then the plan's own tail" is a
    plan of exactly the same length. `_Board.walk_optsets` derives it from a
    reverse BFS on the walk graph. This needs no optimality assumption at all.
  * INTERACTION steps carry the single press the plan takes, unless the level was
    PROVED shortest, in which case `_Board.exact_optsets` re-solves from each of
    the four successors under a bound and keeps every press that still finishes in
    the presses that remain -- a proof in both directions, nothing optimal omitted
    and nothing second-best offered. A re-solve that hits `optset_cap` is reported
    as unknown and simply not offered, so the label is never a guess.

Never `optimal=None`: every recorded expert step has a set.

RECOVERY is the family's RESET prefix (`recovery_mode = "reset"`). Melting a
lava square, charring a crate and shoving a high crate off its plateau are all
irreversible, so a flailed board cannot be re-planned from in general; the
exploration prefix flails freely and ONE RESET restores the state the cached plan
was solved from. `epsilon` stays 0 for the same reason. Plans live in
`data/two_level_puzzle_plans.json`.

ART: three objects had to be redrawn, all of them the same bug. Crate
(`#999944 #88883a #777733`), HighCrate (`#dddd55 #eeee50 #cccc48`) and
BurntCrate (`#222222 #111111 #000000`) each quantise to ONE ARC index across all
three of their hexes, so all three shipped as SOLID 5x5 blocks that hid whatever
they stood on: a crate parked on a staircase was pixel-identical to a crate on
bare floor (and the staircase is the only exit from a low room), a high crate
resting on a low crate was identical to one resting on the high floor (and those
differ by whether pulling the crate out drops it), and a burnt crate covered the
lava it is welded to while being painted in the LowFloor's own two colours. They
are now a hollow box, a 3x3 middle block and a purple ring; `--audit` renders all
28 reachable cell compositions at all three board shapes and asserts pairwise
distinctness, plus the `stand_iii` letterbox check. The header of
`data/puzzlescript_games/Two_level_puzzle.txt` carries the same note.

SYMMETRY: rotation only -- the game is NOT in `_FLIP_GAMES`, deliberately. The
one cue that says which world the player is in is that LowPlayer is drawn on
sprite rows 1-4 and HighPlayer on rows 0-3, i.e. the figure sits LOWER in its
cell when it is on the low floor. A rotation carries that offset to a rotated
offset consistently; a vertical MIRROR swaps it, so a mirrored frame would show
the low player riding high in its cell and vice versa -- the one bit the whole
game is about, inverted. `--symmetry` replays every level's plan and a seeded
random walk at all four rotations and requires exact frame equality with the
transform.

VERIFIED. 3 of 3 levels, 150 presses in 40 macros -- 53, 18 and 79 -- every one
of them replayed through the real interpreter to a WIN, and levels 0 and 1
provably shortest with EXACT tie sets (13 and 3 tie-presses, 0 candidates left
unjudged). Level 2's 79 presses came from the bottleneck split, are a measured
win and are NOT proved shortest -- the bounded exact re-run capped -- so its 4
tie-presses are the walk ties, which claim only that they are as good as what was
recorded. Every report says which is which, per level.

`--selfcheck`: 171600 presses compared to the interpreter object for object at
its default settings -- 180 rollouts from the level starts plus 5000 synthetic
boards -- 0 divergences, covering 6980 pushes, 6119 slides, 5253 quenches, 5393
crates charred, 2757 falls, 2514 deaths and 2569 coins; a 324000-press pass at
twice the boards is also 0. It is what found the collision-layer EVICTION
the .txt does not mention (853 divergences, then 115): a PuzzleScript RHS that
PLACES an object evicts whatever shares that object's layer in the target square,
so a low player walking onto a coin that a crate is standing on DESTROYS the
crate, and a high crate falling onto the square you are standing in destroys YOU.
`--audit`: 28 cell compositions, pairwise distinct at all three board shapes,
none of them the letterbox colour. `--symmetry`: all 4 presentations drawn, every
plan a WIN at every one of them, and every frame of both the plans and a seeded
150-press random walk exactly the transform of the unaugmented one, 0 violations.
`--ties`: 34 presses re-derived by a primitive-press BFS that knows nothing about
macros, 0 disagreements in either direction, 0 candidates left unjudged (37
presses of the proved plans are too deep for an exhaustive BFS and are reported
skipped rather than quietly passed).

Recorded in-process: 3 seeds x 3 levels, 9/9 WIN, 450 expert steps of which 450
carry an optimal set (60 tie-presses, the taken press always inside its own set),
the only unlabelled steps being the 35 RESET-prefix explore presses the closing
RESET discards; indices in 0..5; the longest level recorded is 80 actions against
the adapter's 200-press cap; and two processes at the same rng seed produce a
byte-identical episode. Cold build of the plan cache is ~13 minutes -- level 0's
exact tie sets and level 2's four failed rungs are nearly all of it -- and after
that a 3-seed episode is 3 seconds.

CLI
---
    --selfcheck   the native model fuzzed against the interpreter
    --audit       every cell composition renders distinctly, at every board shape
    --symmetry    every presentation is an exact transform of the unaugmented
    --plans       every level's plan, replayed through the interpreter
    --ties        every optimal-action label re-derived by a primitive BFS
"""

from __future__ import annotations

import heapq
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (        # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from arcengine import ActionInput, GameState        # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION   # noqa: E402
from solvers.common.ps_astar import (              # noqa: E402
    PSAStarSolver, PSExpert, Plan, screen_action)
from utils.rotation import inverse_remap_action_full  # noqa: E402

GAME_NAME = "Two_level_puzzle"
GAME_ID = "ps:two_level_puzzle"

#: Engine directions, in the order the model indexes them. ACTION is not bound
#: to a single rule in this game, so the expert searches four presses, not five.
DIRNAMES = ("up", "down", "left", "right")
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: The two collision layers that matter. Layer 3 holds Crate / IceCube /
#: LowPlayer / MeltingIce / BurntCrate, layer 4 holds Pillar / HighPlayer /
#: HighCrate / Coin. Everything else in the game is static scenery.
LAY_LOW, LAY_HIGH = 3, 4

_INF = 1 << 30

#: `heuristic` weights, in presses. FIRE_PEN is what a ghost player pays to walk
#: over a lava square -- i.e. a guess at the cost of fetching an ice cube and
#: sliding it in -- and SUPPORT_PEN what it pays to stand on a low square with
#: nothing under it, i.e. a guess at the cost of putting a crate there. Both are
#: rankings and neither is a bound; see the module docstring.
FIRE_PEN = 12
SUPPORT_PEN = 10


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level, as a model the searches can step ~50k times a second.

    A state is the ten-tuple

        (player, phigh, crates, hcrates, burnt, ice, fire, coins, melting, moving)

    -- a flat cell index and an elevation flag for the player (``player == -1``
    is a player who walked into the lava), six bitmasks, and the two transient
    objects a slide passes through. ``melting`` and ``moving`` are empty in every
    settled state; they are in the tuple because a `again`-continuation is
    literally a second tick of the same press, and `selfcheck` compares the model
    against the interpreter object for object, so leaving them out would make the
    comparison lie rather than make the state smaller.

    The static scenery -- which squares are HighFloor, which hold a Pillar, which
    hold a staircase -- never changes and lives on the instance. WallFacade is
    not modelled at all: it is in no rule, on a collision layer of its own, and
    exists only to paint the side of a plateau.
    """

    OBJECTS = ("lowfloor", "highfloor", "wallfacade", "highplayer", "lowplayer",
               "crate", "highcrate", "burntcrate", "meltingice", "staticice",
               "icecuben", "icecubee", "icecubes", "icecubew", "firepit",
               "pillar", "icecubeshadow", "coin", "stairsnorth")

    #: engine object name -> the index `tick` gives that sliding direction.
    _MOVING = {"icecuben": 0, "icecubes": 1, "icecubew": 2, "icecubee": 3}

    def __init__(self, eng, ids):
        grid = eng.grid
        self.h, self.w = len(grid), len(grid[0])
        n = self.n = self.h * self.w
        self.nb = [[-1] * 4 for _ in range(n)]
        for r in range(self.h):
            for c in range(self.w):
                i = r * self.w + c
                for k, (dr, dc) in enumerate(_DELTA):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.h and 0 <= cc < self.w:
                        self.nb[i][k] = rr * self.w + cc
        self.ids = ids
        self.state = self.read(eng)
        self.high = self.pillar = self.stairs = 0
        for r in range(self.h):
            for c in range(self.w):
                cell, b = grid[r][c], 1 << (r * self.w + c)
                if ids["highfloor"] in cell:
                    self.high |= b
                if ids["pillar"] in cell:
                    self.pillar |= b
                if ids["stairsnorth"] in cell:
                    self.stairs |= b
        self._hfield: dict = {}     # ranking fields, by what `field` reads
        self._lb: dict = {}         # admissible fields, by (coins, fire, k)
        self._rev: dict | None = None   # the ghost graph's skeleton
        self._locus: dict = {}      # openable lava, by (ice, moving, fire)
        #: What `field` charges a ghost for standing on thin air. An instance
        #: attribute rather than the constant so `bottleneck` can switch it OFF
        #: for its first phase -- where the crates are scaffolding for a job
        #: that has not started yet, and charging for them only repels the
        #: search from the one move it has to make.
        self.support_pen = SUPPORT_PEN
        # The one placement in the game that could move a PILLAR: rule 17 writes
        # a HighPlayer into the square above a staircase and that write evicts
        # whatever shares its collision layer there. Pillars are modelled as
        # static scenery (they are in no rule that moves them and nothing can be
        # pushed into one), which is exact as long as no staircase has a pillar
        # directly above it. Assert it rather than assume it -- `_random_grid`
        # honours the same invariant, and an edited level that broke it would
        # otherwise diverge silently.
        for i in range(n):
            if (self.stairs >> i) & 1:
                j = self.nb[i][0]
                assert j < 0 or not ((self.pillar >> j) & 1), (
                    f"pillar directly above the staircase at cell {i}: "
                    "rule 17 would delete it and the model keeps pillars static")

    # -- reading the interpreter --------------------------------------------
    def read(self, eng):
        """The interpreter's grid, in the model's own terms. `selfcheck` compares
        against this, and it is also how a level's start state is built."""
        ids = self.ids
        player, phigh = -1, False
        crates = hcrates = burnt = ice = fire = coins = melting = 0
        mv = {}
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i = r * self.w + c
                b = 1 << i
                if ids["crate"] in cell:      crates |= b
                if ids["highcrate"] in cell:  hcrates |= b
                if ids["burntcrate"] in cell: burnt |= b
                if ids["staticice"] in cell:  ice |= b
                if ids["firepit"] in cell:    fire |= b
                if ids["coin"] in cell:       coins |= b
                if ids["meltingice"] in cell: melting |= b
                if ids["lowplayer"] in cell:  player, phigh = i, False
                if ids["highplayer"] in cell: player, phigh = i, True
                for name, d in self._MOVING.items():
                    if ids[name] in cell:
                        mv[i] = d
        return (player, phigh, crates, hcrates, burnt, ice, fire, coins,
                melting, tuple(sorted(mv.items())))

    def won(self, st) -> bool:
        return st[7] == 0

    # -- dynamics ------------------------------------------------------------
    def tick(self, st, indir):
        """One pass of the rule list + force resolution + the late rules, i.e.
        exactly one iteration of `PSEngine.step`'s `again` loop.

        Returns ``(state, any_again)``. ``indir`` is the pressed direction on the
        first iteration and None on every continuation -- PuzzleScript consumes
        the input once and re-runs the rules with no key held.

        The rule order below is the .txt's own, and it is load-bearing three
        times: the pillar rule (3) runs BEFORE the ice-slide rules (6-14), so it
        only ever cancels a cube the PLAYER just shoved and never one already in
        flight (the high-floor rule (18) is what stops those); the melt rule (5)
        runs before the slide rules, which is why a cube halts on the first lava
        square it enters instead of sailing over it; and the staircase rule (17)
        runs before the high-floor rule (18), which is the only reason pressing
        up on a staircase is not simply cancelled.
        """
        (player, phigh, crates, hcrates, burnt, ice, fire, coins,
         melting, moving) = st
        nb, high, pillar, stairs = self.nb, self.high, self.pillar, self.stairs
        again = False
        mv = dict(moving)
        pforce = indir
        cforce: dict[int, int] = {}
        iforce: dict[int, int] = {}
        hforce: dict[int, int] = {}

        # 1/2 -- the two push rules.
        if player >= 0 and pforce is not None:
            j = nb[player][pforce]
            if j >= 0:
                b = 1 << j
                if phigh:
                    if hcrates & b:
                        hforce[j] = pforce
                elif crates & b:
                    cforce[j] = pforce
                elif ice & b:
                    iforce[j] = pforce
        # 3 -- Grounded may not step under a Pillar.
        if pforce is not None and player >= 0 and not phigh:
            j = nb[player][pforce]
            if j >= 0 and (pillar >> j) & 1:
                pforce = None
        for forces in (cforce, iforce):
            for cell in list(forces):
                j = nb[cell][forces[cell]]
                if j >= 0 and (pillar >> j) & 1:
                    del forces[cell]
        # 4 -- melting ice takes the lava with it.
        quenched = melting & fire
        if quenched:
            melting &= ~quenched
            fire &= ~quenched
        # 5 -- any ice cube standing in lava starts melting, and stops sliding.
        movmask = 0
        for cell in mv:
            movmask |= 1 << cell
        hit = (ice | movmask) & fire
        if hit:
            again = True
            melting |= hit
            ice &= ~hit
            if player >= 0 and not phigh and (hit >> player) & 1:
                player = -1                     # layer-3 placement evicts it
            for cell in list(mv):
                if (hit >> cell) & 1:
                    del mv[cell]
            for cell in list(iforce):
                if (hit >> cell) & 1:
                    del iforce[cell]
        # 6-9 -- a shoved StaticIce becomes a directed MovingIce.
        for cell, d in list(iforce.items()):
            if (ice >> cell) & 1:
                ice &= ~(1 << cell)
                mv[cell] = d
        # 10-14 -- MovingIce drops a shadow where it stands and keeps its force.
        if mv:
            again = True
            for cell, d in mv.items():
                iforce[cell] = d
        shadow = set(mv)
        # 15/16 -- coin pickup, which is a teleport rather than a move. The RHS
        # PLACES the player in the coin's square, and `_apply_rule_match_forces`
        # evicts whatever else is on the arriving object's collision layer, so a
        # low player walking onto a coin that a crate or an ice cube is sitting
        # on DESTROYS it. (Level 1 can build exactly that: its coin is on the low
        # floor and a shoved cube slides right onto it.)
        if player >= 0 and pforce is not None:
            j = nb[player][pforce]
            if j >= 0 and (coins >> j) & 1 and bool((high >> j) & 1) == phigh:
                coins &= ~(1 << j)
                player = j
                pforce = None
                if phigh:
                    hcrates &= ~(1 << j)
                    hforce.pop(j, None)
                else:
                    crates &= ~(1 << j)
                    ice &= ~(1 << j)
                    burnt &= ~(1 << j)
                    melting &= ~(1 << j)
                    mv.pop(j, None)
                    cforce.pop(j, None)
                    iforce.pop(j, None)
        # 17 -- the staircase, another placement (and another eviction, this one
        # on the high layer).
        if (player >= 0 and not phigh and pforce == 0
                and (stairs >> player) & 1):
            j = nb[player][0]
            if j >= 0 and (high >> j) & 1:
                player, phigh, pforce = j, True, None
                hcrates &= ~(1 << j)
                coins &= ~(1 << j)
                hforce.pop(j, None)
        # 18 -- Grounded may not step onto the high floor (nor off the board).
        if pforce is not None and player >= 0 and not phigh:
            j = nb[player][pforce]
            if j < 0 or (high >> j) & 1:
                pforce = None
        for forces in (cforce, iforce):
            for cell in list(forces):
                j = nb[cell][forces[cell]]
                if j < 0 or (high >> j) & 1:
                    del forces[cell]

        # -- force resolution ------------------------------------------------
        if pforce is not None or cforce or iforce or hforce:
            movmask = 0
            for cell in mv:
                movmask |= 1 << cell
            occ_low = crates | burnt | melting | movmask | ice
            occ_high = pillar | hcrates | coins
            if player >= 0:
                if phigh:
                    occ_high |= 1 << player
                else:
                    occ_low |= 1 << player
            ents = []
            if pforce is not None and player >= 0:
                ents.append(("p", player, LAY_HIGH if phigh else LAY_LOW, pforce))
            for cell, d in cforce.items():
                ents.append((("c", cell), cell, LAY_LOW, d))
            for cell, d in iforce.items():
                ents.append((("i", cell), cell, LAY_LOW, d))
            for cell, d in hforce.items():
                ents.append((("h", cell), cell, LAY_HIGH, d))
            for tag, frm, to in self._resolve(ents, occ_low, occ_high):
                if tag == "p":
                    player = to
                elif tag[0] == "c":
                    crates = (crates & ~(1 << frm)) | (1 << to)
                elif tag[0] == "h":
                    hcrates = (hcrates & ~(1 << frm)) | (1 << to)
                else:
                    mv[to] = mv.pop(frm)

        # -- the late rules, in the .txt's order -----------------------------
        if player >= 0 and not phigh and (fire >> player) & 1:
            player = -1                                   # L1: died in the lava
        for cell in list(mv):                             # L2: never left its
            if cell in shadow:                            #     shadow -> stopped
                del mv[cell]
                ice |= 1 << cell
                if player == cell and not phigh:
                    player = -1                           #     (layer eviction)
        burn = crates & fire                              # L4
        if burn:
            crates &= ~burn
            burnt |= burn
            if player >= 0 and not phigh and (burn >> player) & 1:
                player = -1                               #     (layer eviction)
        blocks_low = crates | high | ice | pillar | burnt
        if player >= 0 and phigh and not ((blocks_low >> player) & 1):
            phigh = False                                 # L5: fell
            again = True
            melting &= ~(1 << player)                     #     (layer eviction)
            mv.pop(player, None)
        drop = hcrates & ~blocks_low                      # L6
        if drop:
            hcrates &= ~drop
            crates |= drop
            melting &= ~drop
            for cell in list(mv):
                if (drop >> cell) & 1:
                    del mv[cell]
            if player >= 0 and not phigh and (drop >> player) & 1:
                player = -1        # a falling crate lands ON the player, and the
                                   # arriving Crate evicts it off the low layer
            again = True
        return ((player, phigh, crates, hcrates, burnt, ice, fire, coins,
                 melting, tuple(sorted(mv.items()))), again)

    def _resolve(self, ents, occ_low, occ_high):
        """`PSEngine._resolve_forces`, for the chains this game can build.

        Kept as the engine's four phases (trace every chain to its end, subsume a
        chain whose head is somebody else's tail, block chains that claim the same
        square on the same layer, move the rest, repeat) rather than special-cased
        to "player plus one crate" -- the general shape is what `selfcheck` is
        comparing against, and it costs nothing when there are two entities.
        """
        moved: dict = {}
        pending = {e[0]: e for e in ents}
        pos = {e[0]: e[1] for e in ents}
        for _ in range(20):
            if not pending:
                break
            at = {(pos[t], pending[t][2]): t for t in pending}
            chains, blocked, seen = [], set(), set()
            deferred = False
            for tag in list(pending):
                if tag in seen:
                    continue
                _t, _frm, lay, d = pending[tag]
                chain, cur, free, mover = [tag], pos[tag], False, False
                while True:
                    j = self.nb[cur][d]
                    if j < 0:
                        break
                    occ = occ_low if lay == LAY_LOW else occ_high
                    if not ((occ >> j) & 1):
                        free = True
                        break
                    other = at.get((j, lay))
                    if other is not None and pending[other][3] == d:
                        chain.append(other)
                        cur = j
                    else:
                        mover = other is not None
                        break
                if free:
                    chains.append((chain, d))
                elif mover:
                    deferred = True
                else:
                    blocked.update(chain)
                seen.update(chain)
            for t in blocked:
                pending.pop(t, None)
            tails = {t for chain, _d in chains for t in chain[1:]}
            claims, conflict = {}, set()
            for ci, (chain, d) in enumerate(chains):
                if chain[0] in tails:
                    continue
                for t in chain:
                    key = (self.nb[pos[t]][d], pending[t][2])
                    if key in claims and claims[key] != ci:
                        conflict.add(ci)
                        conflict.add(claims[key])
                    else:
                        claims[key] = ci
            any_moved = False
            for ci, (chain, d) in enumerate(chains):
                if chain[0] in tails or ci in conflict:
                    continue
                for t in reversed(chain):
                    frm = pos[t]
                    to = self.nb[frm][d]
                    if pending[t][2] == LAY_LOW:
                        occ_low = (occ_low & ~(1 << frm)) | (1 << to)
                    else:
                        occ_high = (occ_high & ~(1 << frm)) | (1 << to)
                    pos[t] = to
                    moved[t] = (moved.get(t, (frm, None))[0], to)
                    pending.pop(t, None)
                    any_moved = True
            for ci in conflict:
                for t in chains[ci][0]:
                    pending.pop(t, None)
            if not any_moved and not deferred:
                break
        return [(t, f, o) for t, (f, o) in moved.items()]

    def step(self, st, k):
        """One press of direction ``k``, run to a settled state.

        The `again` loop stops when no rule asked for a continuation, when the
        iteration changed nothing, or when the win condition is satisfied --
        which is `PSEngine.step`'s own set of exits, in its own order."""
        cur, indir = st, k
        for _ in range(50):
            before = cur
            cur, again = self.tick(cur, indir)
            indir = None
            if not again or cur == before or self.won(cur):
                break
        return cur

    # -- the walk graph ------------------------------------------------------
    def walk_succ(self, st, node):
        """``[(direction, node)]`` for every press at ``node`` that changes
        NOTHING but the player -- walking, climbing a staircase and falling off
        a ledge, but not a push, not a coin (both are interactions) and not a
        step into the lava (that is death, and death is never on a win path).

        Written out geometrically rather than by calling `step` and comparing,
        because the searches ask for it hundreds of thousands of times; the two
        agree by construction on rules 3, 15-18 and L5, and `--ties` re-derives
        whole plans with a primitive-press BFS that never looks at this method.
        """
        (player, phigh, crates, hcrates, burnt, ice, fire, coins,
         melting, moving) = st
        high, pillar, stairs, nb = self.high, self.pillar, self.stairs, self.nb
        movmask = 0
        for cell, _d in moving:
            movmask |= 1 << cell
        occ_low = crates | burnt | ice | melting | movmask
        occ_high = pillar | hcrates
        blocks_low = crates | high | ice | pillar | burnt
        i, hp = node
        out = []
        for k in range(4):
            j = nb[i][k]
            if j < 0:
                continue
            if hp:
                if (occ_high >> j) & 1 or (coins >> j) & 1:
                    continue
                nhp = bool((blocks_low >> j) & 1)
                if not nhp and (fire >> j) & 1:
                    continue                      # falls straight into the lava
                out.append((k, (j, nhp)))
            else:
                if (pillar >> j) & 1:
                    continue
                if (high >> j) & 1:
                    if k == 0 and (stairs >> i) & 1:
                        out.append((k, (j, True)))
                elif ((occ_low >> j) & 1 or (coins >> j) & 1
                      or (fire >> j) & 1):
                    continue
                else:
                    out.append((k, (j, False)))
        return out

    def walk_closure(self, st):
        """``node -> (presses, parent)`` over the walk graph from the player's
        square. ``{}`` when the player is dead."""
        if st[0] < 0:
            return {}
        start = (st[0], st[1])
        dist = {start: (0, None)}
        q = deque([start])
        while q:
            node = q.popleft()
            g = dist[node][0]
            for k, nxt in self.walk_succ(st, node):
                if nxt not in dist:
                    dist[nxt] = (g + 1, (node, k))
                    q.append(nxt)
        return dist

    def walk_route(self, closure, node):
        """The presses of the closure's own shortest walk to ``node``."""
        out = []
        while closure[node][1] is not None:
            prev, k = closure[node][1]
            out.append(k)
            node = prev
        out.reverse()
        return out

    def walk_dist_to(self, st, target):
        """``node -> presses still needed to reach ``target`` by walking``.

        A reverse BFS, because the walk graph is NOT symmetric: falling off a
        ledge is one-way and so is a staircase."""
        rev: dict = {}
        for node in self.walk_closure(st):
            for _k, nxt in self.walk_succ(st, node):
                rev.setdefault(nxt, []).append(node)
        dist = {target: 0}
        q = deque([target])
        while q:
            node = q.popleft()
            for prev in rev.get(node, ()):
                if prev not in dist:
                    dist[prev] = dist[node] + 1
                    q.append(prev)
        return dist

    def macros(self, st, closure=None):
        """``[(presses, (cell, elevation, direction), state)]`` -- walk to the
        square an interaction is taken from, then take it.

        An interaction is a push or a coin pickup, so the candidates are read
        straight off the board rather than probed: only a square holding a
        Pushable (low player), a HighCrate (high player) or a Coin can be one.
        A press that leaves the state untouched (a push into a wall) and one that
        kills the player are dropped -- neither can be on a shortest win path.
        """
        if closure is None:
            closure = self.walk_closure(st)
        crates, hcrates, ice, coins = st[2], st[3], st[5], st[7]
        out = []
        for (i, hp), (g, _p) in closure.items():
            for k in range(4):
                j = self.nb[i][k]
                if j < 0:
                    continue
                b = 1 << j
                if hp:
                    if not ((hcrates & b) or (coins & b)):
                        continue
                elif not ((crates & b) or (ice & b) or (coins & b)):
                    continue
                src = (i, hp) + st[2:]
                ns = self.step(src, k)
                if ns == src or (ns[0] < 0 and not self.won(ns)):
                    continue      # a no-op, or a death that wins nothing
                out.append((g + 1, (i, hp, k), ns))
        return out

    # -- the two heuristics --------------------------------------------------
    def _build_edges(self):
        """The GHOST graph, once per level: the moves of a player who walks
        through crates and ice as if they were not there.

        Every real press is one of these edges. Walking is; a push is (the player
        steps onto the square the crate just left); climbing a staircase is;
        falling off a ledge is; and the coin pickup is the last edge, into the
        coin's own square. Nothing STATIC is relaxed -- pillars still block, the
        high floor is still unreachable from below except by a staircase, and a
        fall is still one-way -- so the ghost's moves are a SUPERSET of the
        player's at one press each, and every distance over this graph undershoots.

        Stored as reverse adjacency ``node -> [(predecessor, kind, square)]``
        with ``kind`` saying what the edge borrows: 0 nothing, 1 a low step onto
        ``square`` (which needs an ice cube if that square is lava), 2 a high
        step onto ``square`` (which needs something solid there to stand on).
        Keeping the skeleton and re-weighting it is what makes a fresh field
        affordable at every node of a search.
        """
        high, pillar, stairs, nb = self.high, self.pillar, self.stairs, self.nb
        rev: dict = {}
        for i in range(self.n):
            for k in range(4):
                j = nb[i][k]
                if j < 0 or (pillar >> j) & 1:
                    continue
                if (high >> j) & 1:
                    if k == 0 and (stairs >> i) & 1:
                        rev.setdefault((j, True), []).append(((i, False), 0, j))
                else:
                    rev.setdefault((j, False), []).append(((i, False), 1, j))
                    rev.setdefault((j, False), []).append(((i, True), 1, j))
                rev.setdefault((j, True), []).append(((i, True), 2, j))
        return rev

    def _field(self, coins, lava_cost, support_cost):
        """Reverse Dijkstra from the coins over `_build_edges`, charging
        ``lava_cost[j]`` for a low step onto ``j`` and ``support_cost[j]`` for a
        high step onto it. ``None`` in either table means the edge does not exist
        at all. One traversal answers for every square at once."""
        if self._rev is None:
            self._rev = self._build_edges()
        dist, pq = {}, []
        c = coins
        while c:
            j = (c & -c).bit_length() - 1
            node = (j, bool((self.high >> j) & 1))
            dist[node] = 0
            pq.append((0, node))
            c &= c - 1
        heapq.heapify(pq)
        while pq:
            d, node = heapq.heappop(pq)
            if d > dist.get(node, _INF):
                continue
            for prev, kind, j in self._rev.get(node, ()):
                extra = 0
                if kind == 1:
                    extra = lava_cost[j]
                elif kind == 2:
                    extra = support_cost[j]
                if extra is None:
                    continue
                nd = d + 1 + extra
                if nd < dist.get(prev, _INF):
                    dist[prev] = nd
                    heapq.heappush(pq, (nd, prev))
        return dist

    def _ice_count(self, st) -> int:
        return bin(st[5]).count("1") + len(st[9])

    def _openable(self, st):
        """The lava squares an ice cube could still, conceivably, reach.

        A cube only ever moves by being SHOVED: the player stands on one side and
        the cube slides until something stops it. So from a square ``i`` and a
        direction ``d`` the shove is possible only when ``i - d`` is a square a
        low player could ever occupy -- and HighFloor and Pillar are static, so
        "could ever" is decidable without knowing where the crates are. The ray
        from ``i`` then either runs into static scenery (and the cube stops
        somewhere along it) or reaches lava, which is where the cube DIES, taking
        that square with it.

        Optimistic on purpose: crates are ignored (they move), and a lava square
        in the way of the player is treated as standable (it might be quenched
        first, at the cost of another cube). So the set can only be too big,
        which is what a bound needs.

        This is what notices that a cube shoved into ROW 2 of level 2 is gone for
        good -- row 1 is high floor, so no player can ever stand above or below it
        in that row, and it can never leave the row it is in. Every branch that
        wastes the level's single cube that way is reported DEAD instead of being
        searched.
        """
        key = (st[5], st[9], st[6])
        got = self._locus.get(key)
        if got is not None:
            return got
        high, pillar, nb = self.high, self.pillar, self.nb
        blocked = high | pillar
        seen = set()
        stack = [c for c in range(self.n) if (st[5] >> c) & 1]
        stack += [c for c, _d in st[9]]
        seen.update(stack)
        open_lava = set()
        while stack:
            i = stack.pop()
            for d in range(4):
                back = nb[i][(0, 1, 3, 2)[d]]      # the square the player needs
                if back < 0 or (blocked >> back) & 1:
                    continue
                j = nb[i][d]
                while j >= 0 and not ((blocked >> j) & 1):
                    if (st[6] >> j) & 1:
                        open_lava.add(j)           # the cube dies here
                        break
                    if j not in seen:
                        seen.add(j)
                        stack.append(j)
                    j = nb[j][d]
        self._locus[key] = open_lava
        return open_lava

    def lower_bound(self, st) -> int:
        """An ADMISSIBLE lower bound on the presses left, and an exact deadness
        test when it is infinite.

        Every press moves the player at most one square -- a slide carries the ice
        a long way and the player still steps exactly once -- and every real move
        is a `_build_edges` edge, so the zero-cost ghost distance already
        undershoots. What sharpens it is the ICE BUDGET: a lava square is cleared
        only by an ice cube, one cube per square, and cubes are never created. So
        with ``k`` cubes left at most ``k`` lava squares will ever open, and

            min over sets S of lava squares, |S| <= k, of (ghost distance with
            exactly S walkable)

        is still a bound. It is evaluated exactly for ``k <= 1``, which is the
        case that does the work: it is what rules out every route needing two
        holes in a two-thick lava field, and -- once the last cube has been spent
        on the wrong square -- what reports the board DEAD instead of letting the
        search wander it. For ``k >= 2`` all lava is treated as already open,
        which is weaker and still a bound; the levels with that many cubes solve
        in under a second anyway.

        Re-crossing is why the budget is expressed as a SET rather than as a
        counter carried along the path: a player who quenches a square may walk
        over it any number of times afterwards, and a counter would charge for
        each crossing and could then over-estimate, which is exactly what a bound
        may not do.
        """
        if self.won(st):
            return 0
        if st[0] < 0:
            return _INF
        k = min(self._ice_count(st), 2)
        openable = sorted(self._openable(st)) if k == 1 else ()
        key = (st[7], st[6] if k < 2 else 0, k, tuple(openable))
        f = self._lb.get(key)
        if f is None:
            free = [0] * self.n
            walls = [0 if not ((st[6] >> j) & 1) else None
                     for j in range(self.n)]
            if k >= 2:
                f = self._field(st[7], free, free)
            else:
                f = self._field(st[7], walls, free)
                for opened in openable:
                    cost = list(walls)
                    cost[opened] = 0
                    one = self._field(st[7], cost, free)
                    for node, d in one.items():
                        if d < f.get(node, _INF):
                            f[node] = d
            self._lb[key] = f
        return f.get((st[0], st[1]), _INF)

    def field(self, st):
        """The RANKING heuristic: the same ghost, now paying for what it borrows
        and paying a DELIVERY charge on top.

        Crossing a lava square costs `FIRE_PEN` plus the distance from the nearest
        ice cube to it, and standing on thin air costs `SUPPORT_PEN` plus the
        distance from the nearest crate or cube. Without those two distances the
        estimate is flat across every crate shuffle in the game -- the ghost walks
        through crates, so moving one changes nothing -- and a search guided by it
        has no gradient at all on the only moves that matter. With them, sliding
        the cube towards the lava square the route needs is visibly downhill.

        A ranking, not a bound: only the weighted and beam tiers use it, and a
        plan either of them finds is proved (or not) afterwards by the bounded
        exact search. Its infinities, though, are the same sound ones
        `lower_bound` reports -- no cube left means no lava square will ever open.
        """
        crates, ice, burnt, fire = st[2], st[5], st[4], st[6]
        key = (fire, crates, ice, burnt, st[7], self.support_pen)
        got = self._hfield.get(key)
        if got is not None:
            return got
        w = self.w
        cubes = [j for j in range(self.n) if (ice >> j) & 1] + \
                [c for c, _d in st[9]]
        solid = crates | ice | burnt | self.high | self.pillar
        movers = [j for j in range(self.n) if ((crates | ice) >> j) & 1]
        lava_cost, support_cost = [0] * self.n, [0] * self.n
        for j in range(self.n):
            if (fire >> j) & 1:
                if not cubes:
                    lava_cost[j] = None
                else:
                    rj, cj = divmod(j, w)
                    lava_cost[j] = FIRE_PEN + min(
                        abs(rj - i // w) + abs(cj - i % w) for i in cubes)
            if not ((solid >> j) & 1):
                if not movers:
                    support_cost[j] = None
                else:
                    rj, cj = divmod(j, w)
                    support_cost[j] = self.support_pen + min(
                        abs(rj - i // w) + abs(cj - i % w) for i in movers)
        got = self._field(st[7], lava_cost, support_cost)
        self._hfield[key] = got
        return got

    def heuristic(self, st) -> int:
        if self.won(st):
            return 0
        if st[0] < 0:
            return _INF
        return self.field(st).get((st[0], st[1]), _INF)

    # -- the searches --------------------------------------------------------
    def region_key(self, st, closure):
        """The world, plus the SMALLEST node the player can walk to. Two states
        that agree on both are interchangeable up to a walk, which is what makes
        the collapse legal for FINDING a plan and illegal for proving one -- the
        walk between the two players costs presses the merge throws away."""
        return st[2:] + (min(i * 2 + int(h) for (i, h) in closure)
                         if closure else -1,)

    def search(self, cap, weight=0, canonical=False, bound=None, start=None,
               verbose=False):
        """Macro A* over the (world, player) graph. Returns ``(cost, chain,
        status)`` with status ``"win"`` / ``"exhausted"`` / ``"capped"``, and
        ``chain`` the ``(state, (cell, elevation, direction))`` macros to the win.

        ``weight == 0`` and ``canonical=False`` is the EXACT rung: the ordering
        heuristic is `lower_bound`, which is admissible and consistent, so the
        first winning state POPPED is the shortest and an exhausted queue proves
        nothing exists at or under ``bound``. Any other setting finds a real plan
        and proves nothing.

        ``bound`` prunes on ``g + lower_bound`` rather than on ``g`` alone, which
        is what makes the hundreds of re-solves in `exact_optsets` affordable.

        The region key and the ranking heuristic are computed at POP time, not
        when a successor is generated: each costs a graph traversal and the great
        majority of generated states are never expanded.
        """
        start = self.state if start is None else start
        t0 = time.time()
        lb0 = self.lower_bound(start)
        if lb0 >= _INF or (bound is not None and lb0 > bound):
            return None, None, "exhausted"
        seen = {start: 0}
        closed: dict = {}
        par: dict = {start: None}
        h0 = self.heuristic(start) if weight else lb0
        if h0 >= _INF:
            return None, None, "exhausted"
        pq = [(weight * h0 if weight else h0, 0, 0, start)]
        cnt = 0
        while pq:
            _f, g, _c, st = heapq.heappop(pq)
            if g > seen.get(st, _INF):
                continue
            closure = self.walk_closure(st)
            if canonical:
                key = self.region_key(st, closure)
                if closed.get(key, _INF) <= g:
                    continue
                closed[key] = g
            if self.won(st):
                if verbose:
                    print(f"      win in {g} presses, {len(seen)} states, "
                          f"{time.time() - t0:.1f}s")
                return g, self._chain(par, st), "win"
            fld = self.field(st) if weight else None
            for cost, act, ns in self.macros(st, closure):
                ng = g + cost
                if seen.get(ns, _INF) <= ng:
                    continue
                lb = self.lower_bound(ns)
                if lb >= _INF and not self.won(ns):
                    continue
                if bound is not None and ng + lb > bound:
                    continue
                if weight:
                    nf = fld if ns[6] == st[6] else self.field(ns)
                    hh = weight * nf.get((ns[0], ns[1]), _INF)
                    if hh >= _INF:
                        continue
                else:
                    hh = lb
                seen[ns] = ng
                par[ns] = (st, act)
                cnt += 1
                heapq.heappush(pq, (ng + hh, ng, cnt, ns))
            if len(seen) > cap:
                if verbose:
                    print(f"      capped at {len(seen)} states, "
                          f"{time.time() - t0:.1f}s")
                return None, None, "capped"
        if verbose:
            print(f"      exhausted at {len(seen)} states, "
                  f"{time.time() - t0:.1f}s")
        return None, None, "exhausted"

    def beam(self, width, weight, max_depth=60, verbose=False):
        """A width-capped layered beam over the macro graph, ordered by
        ``g + weight * heuristic``.

        The A* rungs above stall on this game for a structural reason: the
        estimate is nearly FLAT across crate shuffles (see `field`), so a
        best-first frontier keeps every rearrangement of six crates alive at once
        and the queue grows faster than the depth does. A beam commits instead --
        it keeps the best ``width`` states of each macro layer and throws the rest
        away -- so it reaches depth 30 in seconds and finds a win where a frontier
        search only runs out of memory. What it gives up is any claim of
        shortest; that claim, if it can be had, comes afterwards from `search`
        with the beam's own cost as the bound.

        Returns ``(cost, chain, status)`` in `search`'s own shape so the ladder
        can treat every rung alike.
        """
        start = self.state
        if self.won(start):
            return 0, [], "win"
        t0 = time.time()
        par = {start: None}
        seen = {start: 0}
        layer = [(0, start)]
        for depth in range(max_depth):
            cands = []
            for g, st in layer:
                for cost, act, ns in self.macros(st):
                    ng = g + cost
                    if seen.get(ns, _INF) <= ng:
                        continue
                    seen[ns] = ng
                    par[ns] = (st, act)
                    if self.won(ns):
                        if verbose:
                            print(f"      beam(w={width}, W={weight}) win in "
                                  f"{ng} presses at depth {depth + 1}, "
                                  f"{len(seen)} states, {time.time() - t0:.1f}s")
                        return ng, self._chain(par, ns), "win"
                    if self.lower_bound(ns) >= _INF:
                        continue
                    h = self.heuristic(ns)
                    if h >= _INF:
                        continue
                    cands.append((ng + weight * h, ng, ns))
            if not cands:
                break
            cands.sort(key=lambda x: (x[0], x[1]))
            layer = [(g, st) for _f, g, st in cands[:width]]
        if verbose:
            print(f"      beam(w={width}, W={weight}) found nothing in "
                  f"{max_depth} macros, {len(seen)} states, "
                  f"{time.time() - t0:.1f}s")
        return None, None, "capped"

    def bottleneck(self, cap, tail_cap, weight, verbose=False):
        """Split the search at the moment the last ice cube is spent.

        The cube is the level's one CONSUMABLE and spending it is irreversible,
        so every winning plan on a level whose lava has to be crossed has the
        same two halves: get a cube to the right lava square, then finish. The
        halves want opposite things from a heuristic -- the first half has to
        SHOVE CRATES OUT OF THE WAY to clear a lane for the cube, the second half
        has to LEAVE THEM WHERE THEY STAND because they are the scaffolding the
        high player climbs -- and a single frontier search paid the penalty for
        both at once, which is why every rung above stalls here. `field` is
        therefore evaluated with `support_pen` switched off while phase 1 runs.

        Phase 1 is a weighted region A* that STOPS at the bottleneck: a successor
        with no cube left is not expanded, it is handed to phase 2 -- one bounded
        EXACT search (`search`, so its answer is the shortest completion from that
        state, and an exhausted queue is a real refutation of that candidate). The
        first candidate that finishes wins, and the plan is phase 1's route plus
        phase 2's optimum.

        Candidates are cheap and rare -- most branches waste the cube on a lava
        square that opens nothing, and `lower_bound` has already reported those
        DEAD -- so the expensive half runs a handful of times, not once per state.

        Returns ``(cost, chain, status)`` in `search`'s shape. The cost is real
        and the plan is real; neither is claimed shortest.
        """
        start = self.state
        if self._ice_count(start) == 0 or self.won(start):
            return None, None, "exhausted"
        t0 = time.time()
        saved = self.support_pen
        self.support_pen = 0
        try:
            par = {start: None}
            seen = {start: 0}
            closed: dict = {}
            pq = [(0, 0, 0, start)]
            cnt = tried = 0
            while pq:
                _f, g, _c, st = heapq.heappop(pq)
                if g > seen.get(st, _INF):
                    continue
                closure = self.walk_closure(st)
                key = self.region_key(st, closure)
                if closed.get(key, _INF) <= g:
                    continue
                closed[key] = g
                fld = self.field(st)
                for cost, act, ns in self.macros(st, closure):
                    ng = g + cost
                    if seen.get(ns, _INF) <= ng:
                        continue
                    lb = self.lower_bound(ns)
                    if lb >= _INF and not self.won(ns):
                        continue
                    seen[ns] = ng
                    par[ns] = (st, act)
                    cnt += 1
                    if self._ice_count(ns) == 0:
                        tried += 1
                        tail, tchain, tstatus = self.search(tail_cap, start=ns)
                        if tstatus == "win":
                            if verbose:
                                print(f"      bottleneck(W={weight}) win in "
                                      f"{ng} + {tail} = {ng + tail} presses, "
                                      f"candidate {tried}, {len(seen)} states, "
                                      f"{time.time() - t0:.1f}s")
                            return (ng + tail, self._chain(par, ns) + tchain,
                                    "win")
                        continue        # the cube is gone: phase 1 ends here
                    fresh = fld if ns[6] == st[6] else self.field(ns)
                    h = fresh.get((ns[0], ns[1]), _INF)
                    if h >= _INF:
                        continue
                    heapq.heappush(pq, (ng + weight * h, ng, cnt, ns))
                if len(seen) > cap:
                    if verbose:
                        print(f"      bottleneck(W={weight}) capped at "
                              f"{len(seen)} states, {tried} candidates, "
                              f"{time.time() - t0:.1f}s")
                    return None, None, "capped"
        finally:
            self.support_pen = saved
        if verbose:
            print(f"      bottleneck(W={weight}) exhausted at {len(seen)} "
                  f"states, {tried} candidates, {time.time() - t0:.1f}s")
        return None, None, "exhausted"

    @staticmethod
    def _chain(par, goal):
        out = []
        cur = goal
        while par[cur] is not None:
            st, act = par[cur]
            out.append((st, act))
            cur = st
        out.reverse()
        return out

    def flatten(self, chain):
        """The macro chain as a flat list of presses."""
        presses = []
        for st, (i, hp, k) in chain:
            closure = self.walk_closure(st)
            presses.extend(self.walk_route(closure, (i, hp)))
            presses.append(k)
        return presses

    # -- optimal-action sets -------------------------------------------------
    def walk_optsets(self, chain):
        """The tie set every WALK step carries for free, and ``[taken]`` on the
        interactions.

        A walk changes nothing but the player, so at a square ``d`` presses from
        the one the next interaction is taken from, EVERY press that lands ``d-1``
        away is exactly as good: that press, any shortest route on from there, the
        same interaction and the plan's own tail is a plan of the same length. The
        distances come from a reverse BFS on the walk graph, because falling off a
        ledge and climbing a staircase are one-way and a forward BFS would answer
        the wrong question.

        Sound with no reference to optimality: it never claims a press is optimal,
        only that it is as good as the one recorded.
        """
        out = []
        for st, (ti, th, tk) in chain:
            dist = self.walk_dist_to(st, (ti, th))
            closure = self.walk_closure(st)
            node = (st[0], st[1])
            for k in self.walk_route(closure, (ti, th)):
                succ = dict(self.walk_succ(st, node))
                d = dist[node]
                best = [kk for kk, nxt in succ.items()
                        if dist.get(nxt, _INF) == d - 1]
                out.append(best if k in best else [k])
                node = succ[k]
            out.append([tk])
        return out

    def exact_optsets(self, presses, base, cap, verbose=False):
        """Widen ``base`` into the EXACT tie set at every step, by re-solving.

        At step ``t`` of a plan proved shortest, a press is optimal iff a shortest
        win from the state it lands on takes exactly the ``len(plan) - t - 1``
        presses that remain. `search` with that bound answers it three ways: a win
        at that cost is a tie, an exhausted queue is a refutation, and a capped
        one is neither -- and a capped candidate is simply not offered, so a label
        is never a guess. Returns ``(sets, unknown_candidates)``.

        The bound prunes on ``g + lower_bound``, which is what keeps hundreds of
        re-solves affordable; without it every one of them is a fresh full search.
        """
        st = self.state
        total = len(presses)
        out, unknown = [], 0
        for t, taken in enumerate(presses):
            remaining = total - t - 1
            best = list(base[t])
            for k in range(4):
                if k in best:
                    continue
                ns = self.step(st, k)
                if ns == st:
                    continue
                if self.won(ns):
                    if remaining == 0:
                        best.append(k)
                    continue
                if ns[0] < 0:
                    continue                  # dead, and the coins are still out
                if remaining == 0 or self.lower_bound(ns) > remaining:
                    continue
                cost, _c, status = self.search(cap, bound=remaining, start=ns)
                if status == "win" and cost == remaining:
                    best.append(k)
                elif status == "capped":
                    unknown += 1
            out.append(sorted(best))
            st = self.step(st, taken)
        if verbose:
            print(f"      exact sets: {sum(len(s) - 1 for s in out)} tie-presses,"
                  f" {unknown} candidates left unjudged")
        return out, unknown

    # -- the ladder ----------------------------------------------------------
    def solve(self, exact_cap, region_cap, weights, bottlenecks,
              bottleneck_cap, bottleneck_tail, beams, optset_cap,
              verbose=False):
        """Plan the level, then try to turn the plan into a theorem.

        Rung 1 alone can end it: a win it POPS is the shortest, full stop. When
        it caps, EVERY inexact rung is run and the shortest win among them is
        kept -- which one answers is a property of the level and not of the game,
        and the rungs disagree by real margins (level 2 is 89 presses from the
        region search and 79 from the bottleneck split, on the same board). The
        beams are the exception: they are only reached when nothing else found a
        win at all, because they have never yet been the rung that answers.

        Then, whatever produced the plan, rung 1 is re-run bounded by one press
        less. Returns ``{"presses", "optsets", "proved", "exact_sets", "kind",
        "macros", "unknown"}``, or None when no rung finds a win.
        """
        if self.won(self.state):
            return {"presses": [], "optsets": [], "proved": True,
                    "exact_sets": True, "kind": "vacuous", "macros": 0,
                    "unknown": 0}
        if verbose:
            print("    rung 1: exact macro A*")
        cost, chain, status = self.search(exact_cap, verbose=verbose)
        kind, proved = "exact", status == "win"
        if not proved:
            found = []
            for w in weights:
                if verbose:
                    print(f"    rung 2: region A*, weight {w}")
                c, ch, st = self.search(region_cap, weight=w, canonical=True,
                                        verbose=verbose)
                if st == "win":
                    found.append((c, f"region(w={w})", ch))
            for w in bottlenecks:
                if verbose:
                    print(f"    rung 3: bottleneck split, weight {w}")
                c, ch, st = self.bottleneck(bottleneck_cap, bottleneck_tail, w,
                                            verbose=verbose)
                if st == "win":
                    found.append((c, f"bottleneck(w={w})", ch))
            if not found:
                for width, w in beams:
                    if verbose:
                        print(f"    rung 4: beam, width {width}, weight {w}")
                    c, ch, st = self.beam(width, w, verbose=verbose)
                    if st == "win":
                        found.append((c, f"beam({width},{w})", ch))
                        break
            if not found:
                return None
            # Shortest wins; the name breaks a tie, which is arbitrary but
            # deterministic -- the point is that two processes derive the same
            # plan, not which rung gets the credit.
            found.sort(key=lambda x: (x[0], x[1]))
            cost, kind, chain = found[0]
            if verbose:
                print("    best of the inexact rungs: "
                      + ", ".join(f"{k} {c}" for c, k, _ch in found))
                print(f"    rung 5: exact macro A* bounded by {cost - 1}")
            bcost, bchain, bstatus = self.search(exact_cap, bound=cost - 1,
                                                 verbose=verbose)
            if bstatus == "win":
                cost, chain, kind, proved = bcost, bchain, "exact", True
            elif bstatus == "exhausted":
                proved = True
        presses = self.flatten(chain)
        assert len(presses) == cost, (len(presses), cost)
        sets = self.walk_optsets(chain)
        assert len(sets) == len(presses)
        unknown = 0
        exact_sets = False
        if proved and optset_cap:
            sets, unknown = self.exact_optsets(presses, sets, optset_cap,
                                               verbose=verbose)
            exact_sets = unknown == 0
        return {"presses": presses, "optsets": sets, "proved": proved,
                "exact_sets": exact_sets, "kind": kind, "macros": len(chain),
                "unknown": unknown}


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class TwoLevelExpert(PSExpert):
    """`PSExpert`'s plan memo, restore discipline and disk cache around the
    native searches. `heuristic` is never called: tier 1 orders itself by
    `_Board.lower_bound` and tier 2 by `_Board.field`, both on the native board.
    """

    directions = list(DIRNAMES)          # ACTION is bound to no rule in this game
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "two_level_puzzle_plans.json")

    #: States ONE exact search may hold before it gives the level up as unproved.
    #: Level 0 spends 116k of them to prove its 53 presses shortest, so this is
    #: five times what the levels it CAN close need. Raising it does not close
    #: level 2 -- an exhaustive primitive-press BFS on it was abandoned at 8M
    #: states -- it only makes rungs 1 and 5 spend longer failing there, which is
    #: most of the cold build's wasted time.
    exact_cap: int = 600_000

    #: States ONE region search may hold. The collapse is worth roughly a room's
    #: worth of player squares, so this is a much bigger search than the number
    #: suggests -- and it is the budget level 2 needs: the region rung finds its
    #: 89-press plan 560k states in, and its weight-12 pass a 95-press one at
    #: 650k.
    region_cap: int = 2_000_000

    #: Weights tried on the ranking heuristic, in order, when the exact search
    #: caps. Smallest first: a small weight is closest to the exact search and so
    #: to the shortest plan, and a large one is what gets a plan at all on a level
    #: whose answer is long.
    weights: tuple = (3, 12)

    #: Weights tried on the BOTTLENECK split, the rung that solves level 2.
    #: See `_Board.bottleneck`. Both find the same 79 presses -- 3 on its
    #: seventh candidate in 24 seconds, 1 on its forty-seventh in 52 -- and a
    #: weight of 8 was dropped for costing minutes and improving on neither.
    bottlenecks: tuple = (3, 1)

    #: States one bottleneck phase 1 may hold, and states one of its bounded
    #: exact completions may hold. Level 2 finds its win 70 seconds in, on its
    #: seventh candidate, well inside both.
    bottleneck_cap: int = 1_500_000
    bottleneck_tail: int = 150_000

    #: ``(width, weight)`` beams tried, in order, when nothing above has found a
    #: plan -- the last rung, and the only one that commits rather than keeping a
    #: frontier (see `_Board.beam`). It has never been the rung that answers.
    beams: tuple = ((3000, 1), (3000, 3), (8000, 3))

    #: The budget for ONE bounded re-solve inside `exact_optsets`, of which there
    #: are up to three per press of the plan. Deliberately far smaller than
    #: `exact_cap`: the fallback (`walk_optsets`) is exact for what it claims, so
    #: the trade is a weaker LABEL on a plan that is still proved shortest, never
    #: a weaker plan.
    optset_cap: int = 250_000

    OBJECTS = _Board.OBJECTS

    def setup(self):
        self.ids = {n: self.g.obj_name_to_idx[n] for n in self.OBJECTS}
        #: level -> the provenance of its plan, filled by `plan` from the disk
        #: entry so a cache hit in a later process reports what the search that
        #: found it reported.
        self.stats = {}
        self.verbose = False
        self._last = None

    def heuristic(self, eng):
        raise AssertionError(
            "TwoLevelExpert plans on its native model; heuristic is unused")

    def board(self, eng) -> _Board:
        return _Board(eng, self.ids)

    def _search(self, eng):
        board = self.board(eng)
        got = board.solve(self.exact_cap, self.region_cap, self.weights,
                          self.bottlenecks, self.bottleneck_cap,
                          self.bottleneck_tail, self.beams, self.optset_cap,
                          verbose=self.verbose)
        if got is None:
            self._last = {"kind": "unsolved", "proved": False,
                          "exact_sets": False, "macros": 0, "unknown": 0}
            return None
        self._last = {k: got[k] for k in
                      ("kind", "proved", "exact_sets", "macros", "unknown")}
        return Plan([DIRNAMES[k] for k in got["presses"]],
                    [[DIRNAMES[k] for k in s] for s in got["optsets"]])

    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan`, plus the provenance of the answer.

        The disk cache stores only ``start`` / ``plan`` / ``optsets``, so a later
        process that hits it would otherwise have no idea whether the plan it is
        replaying was proved shortest or merely found. The extra keys ride along
        in the same entry and are written the first time a level is searched."""
        self._last = None
        found = super().plan(eng, level)
        if level is None:
            return found
        entry = self._disk.get(level)
        if entry is None:
            return found
        if "proved" not in entry and self._last is not None:
            entry.update(self._last)
            self._save_disk()
        self.stats[level] = {
            "kind": entry.get("kind", "?"),
            "proved": bool(entry.get("proved")),
            "exact_sets": bool(entry.get("exact_sets")),
            "macros": entry.get("macros", 0),
            "unknown": entry.get("unknown", 0),
        }
        return found


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class TwoLevelSolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    game_module_id = GAME_ID
    expert_cls = TwoLevelExpert
    #: The longest plan is level 2's 79 presses; this bounds the replay only,
    #: well inside the adapter's own 200-press per-level budget (which
    #: `record_level`'s closing ``set_level`` resets after the exploration
    #: prefix).
    max_steps = 160

    #: Print each rung of the ladder as it runs. Off for generation; `--plans`
    #: turns it on BEFORE `_ensure`, because the cold build happens inside the
    #: solvable-level discovery and would otherwise be a silent half hour.
    plan_verbose: bool = False

    def prepare_expert(self, game, expert) -> None:
        expert.verbose = self.plan_verbose


# ---------------------------------------------------------------------------
# --selfcheck: the native model against the real interpreter
# ---------------------------------------------------------------------------

#: The synthetic-board generator's per-layer menus. Layer 0 is the background
#: (exactly one), layer 1 is the staircase / lava layer, layer 2 the purely
#: decorative facade, and layers 3 and 4 hold at most one object each -- which is
#: what a PuzzleScript collision layer means.
_LAYER_MENU = (
    (("lowfloor", 6), ("highfloor", 3)),
    ((None, 6), ("firepit", 2), ("stairsnorth", 1)),
    ((None, 4), ("wallfacade", 1)),
    ((None, 6), ("crate", 3), ("staticice", 3), ("burntcrate", 1)),
    ((None, 8), ("pillar", 1), ("highcrate", 2), ("coin", 2)),
)


def _random_grid(rng, ids, h, w):
    """A random board, legal by collision layer and nothing else.

    Deliberately unconstrained beyond the layers: a high crate hanging over bare
    floor, a burnt crate NOT on lava, ice already sitting in a fire pit and a
    coin under a crate are all states ordinary play never produces, and all of
    them exercise a rule the shipped levels reach rarely or never (`late [
    HighCrate no BlocksLow ]`, `[ IceCube FirePit ]` firing on the very first
    tick, the pickup rules against an occupied square). The model has to agree
    with the interpreter on those too, or a perturbed board would diverge."""
    def pick(menu):
        total = sum(wt for _n, wt in menu)
        x = rng.randrange(total)
        for name, wt in menu:
            x -= wt
            if x < 0:
                return name
        return menu[-1][0]

    grid = [[set() for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            for menu in _LAYER_MENU:
                name = pick(menu)
                if name is not None:
                    grid[r][c].add(ids[name])
    for r in range(h - 1):                       # see `_Board.__init__`: a pillar
        for c in range(w):                       # directly above a staircase is
            if ids["stairsnorth"] in grid[r + 1][c]:   # the one thing that could
                grid[r][c].discard(ids["pillar"])      # move a pillar
    pr, pc = rng.randrange(h), rng.randrange(w)
    cell = grid[pr][pc]
    if rng.random() < 0.5:                       # a LowPlayer, on layer 3
        for name in ("crate", "staticice", "burntcrate"):
            cell.discard(ids[name])
        cell.add(ids["lowplayer"])
    else:                                        # a HighPlayer, on layer 4
        for name in ("pillar", "highcrate", "coin"):
            cell.discard(ids[name])
        cell.add(ids["highplayer"])
    return grid


def selfcheck(trials=60, steps=120, boards=5000, board_steps=30, verbose=True):
    """Fuzz `_Board` against the interpreter, two ways.

    ROLLOUTS from every level start are the honest fuzz: random play pushes
    crates into lava, slides cubes into it, drops the player off ledges and
    strands high crates. SYNTHETIC boards are what reaches the rest -- see
    `_random_grid` -- and they are the only way `late [ HighCrate no BlocksLow ]`
    and a coin under a crate get exercised at all, because no shipped level can
    produce them.

    Every press compares the WHOLE state object for object, including the two
    transient objects (`MeltingIce`, the four `IceCube` directions) that exist
    only inside an `again` continuation, so a model that got the right settled
    board by the wrong route would still be caught.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, parsed = game._engine, game._game
    ids = {n: parsed.obj_name_to_idx[n] for n in _Board.OBJECTS}
    rng = random.Random(f"{GAME_ID}:selfcheck")
    bad = presses = 0
    tally = {"pushes": 0, "slides": 0, "quenches": 0, "burns": 0,
             "falls": 0, "deaths": 0, "coins": 0, "noops": 0}

    def compare(board, st, tag):
        nonlocal bad, presses
        for _ in range(steps if tag == "level" else board_steps):
            k = rng.randrange(4)
            eng.step(DIRNAMES[k])
            nxt = board.step(st, k)
            presses += 1
            if nxt[2:8] != st[2:8]:
                if nxt[6] != st[6]:
                    tally["quenches"] += 1
                if nxt[4] != st[4]:
                    tally["burns"] += 1
                if nxt[7] != st[7]:
                    tally["coins"] += 1
                if nxt[5] != st[5]:
                    tally["slides"] += 1
                if nxt[2] != st[2] or nxt[3] != st[3]:
                    tally["pushes"] += 1
            elif nxt == st:
                tally["noops"] += 1
            if st[1] and not nxt[1]:
                tally["falls"] += 1
            if nxt[0] < 0 <= st[0]:
                tally["deaths"] += 1
            got = board.read(eng)
            if got != nxt:
                bad += 1
                if bad <= 3:
                    print(f"  MISMATCH ({tag}, press {DIRNAMES[k]})")
                    print(f"    engine: {got}")
                    print(f"    model : {nxt}")
                return
            st = nxt

    for level in range(game.n_levels):
        for _ in range(trials):
            game.set_level(level)
            board = _Board(eng, ids)
            compare(board, board.state, "level")

    shapes = []
    for level in range(game.n_levels):
        game.set_level(level)
        shapes.append((eng.height, eng.width))
    for _ in range(boards):
        h, w = shapes[rng.randrange(len(shapes))]
        eng.height, eng.width = h, w
        eng.grid = _random_grid(rng, ids, h, w)
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        board = _Board(eng, ids)
        compare(board, board.state, "synthetic")
    if verbose:
        print(f"  {presses} presses compared object for object "
              f"({game.n_levels * trials} level rollouts + {boards} synthetic "
              f"boards)")
        print("  " + ", ".join(f"{v} {k}" for k, v in tally.items()))
    return bad


# ---------------------------------------------------------------------------
# --audit: every cell composition, pixel-distinct at every board shape
# ---------------------------------------------------------------------------

#: Every stack a square can hold in play, written as the objects in it. The
#: WallFacade variants are in the list although the facade is decoration -- it is
#: on a collision layer of its own and in no rule -- because a stack that renders
#: identically with and without it would mean the facade is invisible, and it is
#: the only thing that draws the SIDE of a plateau.
_AUDIT_CASES = {
    "floor":               ("lowfloor",),
    "floor+facade":        ("lowfloor", "wallfacade"),
    "lava":                ("lowfloor", "firepit"),
    "lava+facade":         ("lowfloor", "firepit", "wallfacade"),
    "stairs":              ("lowfloor", "stairsnorth"),
    "player":              ("lowfloor", "lowplayer"),
    "player+facade":       ("lowfloor", "wallfacade", "lowplayer"),
    "player on stairs":    ("lowfloor", "stairsnorth", "lowplayer"),
    "crate":               ("lowfloor", "crate"),
    "crate+facade":        ("lowfloor", "wallfacade", "crate"),
    "crate on stairs":     ("lowfloor", "stairsnorth", "crate"),
    "burnt crate":         ("lowfloor", "firepit", "burntcrate"),
    "burnt crate+facade":  ("lowfloor", "firepit", "wallfacade", "burntcrate"),
    "ice":                 ("lowfloor", "staticice"),
    "ice+facade":          ("lowfloor", "wallfacade", "staticice"),
    "ice on stairs":       ("lowfloor", "stairsnorth", "staticice"),
    "low coin":            ("lowfloor", "coin"),
    "low coin under ice":  ("lowfloor", "staticice", "coin"),
    "high player on crate": ("lowfloor", "crate", "highplayer"),
    "high player on ice":  ("lowfloor", "staticice", "highplayer"),
    "high player on burnt": ("lowfloor", "firepit", "burntcrate", "highplayer"),
    "high crate on crate": ("lowfloor", "crate", "highcrate"),
    "high crate on ice":   ("lowfloor", "staticice", "highcrate"),
    "high floor":          ("highfloor",),
    "pillar":              ("highfloor", "pillar"),
    "high player":         ("highfloor", "highplayer"),
    "high crate":          ("highfloor", "highcrate"),
    "high coin":           ("highfloor", "coin"),
}


def audit(verbose=True):
    """Assert every cell COMPOSITION renders differently, at every board shape
    the levels use, and that none of them renders as the letterbox colour.

    The board is FILLED with the composition and WHOLE frames are compared,
    rather than one cell being sliced out by ``cell_px`` arithmetic:
    `_render_frame` upscales a sub-64 render to fill the frame and then
    letterboxes it, so the cell grid in the output is not ``cell_px``-aligned.
    Rendering is per-cell independent, so a whole-frame comparison asks the same
    question where the geometry cannot drift.

    The pad check is the `ps:stand_iii` one: a composition that renders as a flat
    5 is invisible against the letterbox, and a pairwise matrix passes that bug
    every time because the other half of the pair is not a composition.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, parsed = game._engine, game._game
    idx = parsed.obj_name_to_idx
    shapes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        shapes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(shapes.items()):
        shots = {}
        for name, objs in _AUDIT_CASES.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx[o] for o in objs} for _ in range(w)]
                        for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, parsed))
        names = list(shots)
        clashes = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
                   if np.array_equal(shots[a], shots[b])]
        flat = [a for a in names if np.all(shots[a] == 5)]
        bad += len(clashes) + len(flat)
        if verbose:
            print(f"  {h}x{w} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, "
                  f"{'all distinct' if not clashes else 'IDENTICAL ' + str(clashes)}"
                  f"{'' if not flat else ', LETTERBOX-COLOURED ' + str(flat)}")
    return bad


# ---------------------------------------------------------------------------
# --symmetry: the presentation augmentation is a true symmetry of the mechanic
# ---------------------------------------------------------------------------

def symmetry(walk_presses=150, verbose=True):
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    The game takes ROTATION only -- it is deliberately not in `_FLIP_GAMES`. The
    one cue that says which of the two worlds the player is in is that LowPlayer
    is drawn on sprite rows 1-4 and HighPlayer on rows 0-3, i.e. the figure sits
    LOWER in its cell when it is on the low floor. A rotation carries that offset
    to a rotated offset consistently and a vertical mirror inverts it, which
    would show a low player riding high in its cell: the one bit the whole game
    is about, drawn backwards.

    Both the PLANS and a seeded random walk are replayed. The walk reaches boards
    a plan never visits -- crates charred in the lava, cubes spent on the wrong
    square, the player dead -- and it presses the unbound ACTION key too, which
    is the check that ACTION really is inert.
    """
    solver = TwoLevelSolver()
    game, expert, solvable = solver._ensure(0)
    plans = {}
    for level in solvable:
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
    for seed in range(200):
        if len(seen) == 4 and seed > 20:
            break
        g = solver.make_game(seed)
        for level in solvable:
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"{GAME_ID}:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            plan = plans[level] or []
            for tag, presses in (("plan", [screen_action(d, *k) for d in plan]),
                                 ("walk", [inverse_remap_action_full(a, *k)
                                           for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and plan and g._state != GameState.WIN:
                    print(f"  seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue
                    ref[key] = frames
                    continue
                if any(not np.array_equal(transform(a, *k), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"  seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


# ---------------------------------------------------------------------------
# --ties: optimal-action labels re-derived by an independent primitive BFS
# ---------------------------------------------------------------------------

def ties_report(levels=None, state_cap=400_000, max_remaining=16, verbose=True):
    """Re-derive every step's optimal SET with a bounded primitive-press BFS and
    require it to equal the label the plan carries.

    Only meaningful on the levels whose plan is PROVED shortest, and only those
    are checked: there the label claims to be every press that still finishes in
    the presses that remain, and this BFS -- which knows nothing about macros,
    walk closures or region keys, and simply presses all four keys
    breadth-first -- answers the same question from the other side. Agreement in
    both directions is the claim: a press the labels omit and the BFS finds is a
    right answer taught as wrong, and one the labels offer and the BFS refutes is
    a second-best press taught as optimal.

    Unproved levels are LISTED and skipped, because their labels deliberately
    claim something weaker (see `_Board.walk_optsets`) and a shortest-press BFS
    is not the question those answer.

    Only the TAIL of each plan is re-derived -- the last ``max_remaining``
    presses. The BFS is exhaustive to the depth that is left, so at step 0 of a
    53-press plan it is the whole level over again (1.7M states) and at step 40
    it is thirteen presses of it. The report says how many presses it managed,
    rather than quietly checking the easy half.
    """
    solver = TwoLevelSolver()
    game, expert, solvable = solver._ensure(0)

    def shortest_within(board, st, limit):
        """Presses to the nearest win from ``st``, or None past ``limit`` /
        the state budget."""
        if board.won(st):
            return 0
        if limit <= 0 or board.lower_bound(st) > limit:
            return None
        seen = {st}
        frontier = [st]
        for d in range(1, limit + 1):
            nxt = []
            for cur in frontier:
                for k in range(4):
                    ns = board.step(cur, k)
                    if ns in seen:
                        continue
                    if board.won(ns):
                        return d
                    if ns[0] < 0:
                        continue
                    if board.lower_bound(ns) > limit - d:
                        continue
                    seen.add(ns)
                    nxt.append(ns)
            if len(seen) > state_cap:
                return "capped"
            frontier = nxt
            if not frontier:
                return None
        return None

    bad = unjudged = checked = skipped = 0
    for level in solvable:
        if levels is not None and level not in levels:
            continue
        game.set_level(level)
        board = expert.board(game._engine)
        plan = expert.plan(game._engine, level)
        info = expert.stats.get(level, {})
        if plan is None:
            print(f"  L{level}: unsolved, skipped")
            continue
        if not info.get("proved"):
            print(f"  L{level}: plan found by {info.get('kind')} and not proved "
                  f"shortest, skipped")
            continue
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        st = board.state
        total = len(plan)
        for t, press in enumerate(plan):
            remaining = total - t - 1
            if remaining > max_remaining:
                skipped += 1
                st = board.step(st, DIRNAMES.index(press))
                continue
            truth = []
            for k in range(4):
                ns = board.step(st, k)
                if ns == st or (ns[0] < 0 and not board.won(ns)):
                    continue
                got = (0 if board.won(ns)
                       else shortest_within(board, ns, remaining))
                if got == "capped":
                    unjudged += 1
                    continue
                if got == remaining:
                    truth.append(DIRNAMES[k])
            label = sorted(sets[t])
            if sorted(truth) != label:
                bad += 1
                if bad <= 6:
                    print(f"  L{level} step {t}: label {label} vs BFS "
                          f"{sorted(truth)}")
            checked += 1
            st = board.step(st, DIRNAMES.index(press))
        if verbose:
            print(f"  L{level}: {total} presses re-derived")
    if verbose:
        print(f"  {checked} presses re-derived by the primitive BFS, "
              f"{bad} disagreements, {skipped} presses skipped as too deep "
              f"(more than {max_remaining} presses from the win), "
              f"{unjudged} candidates left unjudged at the state budget")
    return bad


# ---------------------------------------------------------------------------
# --plans: every level's plan, replayed through the real interpreter
# ---------------------------------------------------------------------------

def plan_report(verbose=True):
    """Every level's plan, replayed through the real interpreter -- the
    end-to-end test of the model, the search ladder and the tie labelling at
    once."""
    solver = TwoLevelSolver()
    solver.plan_verbose = verbose
    game, expert, solvable = solver._ensure(0)
    total = ties = proved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        if verbose:
            print(f"  L{level}: {board.h}x{board.w}, planning...")
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            continue
        for direction in plan:
            eng.step(direction)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        step_ties = sum(len(s) - 1 for s in sets)
        info = expert.stats.get(level, {})
        tag = "shortest" if info.get("proved") else "found   "
        sets_tag = "exact" if info.get("exact_sets") else "walk "
        proved += 1 if info.get("proved") else 0
        total += len(plan)
        ties += step_ties
        print(f"  L{level}: {len(plan):3d} presses in {info.get('macros', 0):3d} "
              f"macros [{info.get('kind', '?'):>12s} / {tag} / {sets_tag} sets]"
              f"  win={eng.check_win()}  {step_ties:3d} tie-presses")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second right answer, "
          f"{proved}/{len(solvable)} levels provably shortest")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--ties" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--ties") + 1:]
                if a.isdigit()]
        violations = ties_report(args or None)
        print(f"ties: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--plans" in sys.argv:
        plan_report()
        sys.exit(0)
    sys.exit(TwoLevelSolver.main())
