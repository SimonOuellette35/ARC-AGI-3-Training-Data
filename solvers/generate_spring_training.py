"""ps:spring -- Beacon Duckworth's "Spring": you are the season, and the way
out of every room is to finish turning it green.

    late [nodoor floor no player][nlit] -> [door][nlit]
    late [nodoor floor no player][ice]  -> [door][ice]

The door is DELETED and rebuilt at the end of every tick, and those two rules
are the only things that rebuild it -- so it stands as long as one blue square
or one square of ice is left anywhere on the board, and the moment neither is,
the doorway is simply open and walking into it wins.  Nineteen of the game's
twenty-one levels are solved here, fifteen of them PROVED SHORTEST off an exact
distance-to-win field; the two that are not are the pair the last section
describes, and they are in ``skip_levels``.

Part of the `solvers/common/ps_astar.py` family: `PSExpert` supplies the plan
memo, the snapshot discipline and the on-disk plan cache, and `_search` is
replaced by a ladder of searches over a NATIVE model of the game.

THE FOUR MECHANICS, and the thing each of them is really doing

* **A FLOWER LIGHTS ITS 3x3 -- AND MELTS THE ICE IN IT.**  Every tick throws
  the whole floor away (``[floor] -> []``) and repaints it: ``late [no block no
  aqua] -> [nlit]`` makes every square that is not wall, water or ice BLUE, and
  then the flower rules turn some of them green -- ``[flow nlit]`` its own
  square, ``[flow |nlit]`` its four sides, and a `vert` marker stamped above and
  below it carries ``[vert|nlit]`` out to the four diagonals.  Four rules, one
  effect: the king-neighbourhood.  Their `ice` twins (``[flow |ice] -> [flow
  |water watermark]``) do the same job to ice, so a flower planted beside ice
  melts it on the tick it is planted, which is why no shipped board ever shows
  ice next to a flower.

* **YOU CAN ONLY PLANT ON BLUE, AND THAT IS THE WHOLE COMBINATORICS.**
  ``[action spring no unlit no flower no aqua] -> [spring flower]`` refuses a
  square that is already green, so two flowers can never be king-adjacent: the
  second would have to go on a square the first has lit.  The condition is
  symmetric, so the ORDER flowers are planted in is free, and the end state of
  every level is an independent DOMINATING set of the squares that must end
  green.  That is what `LevelSolver.layouts` enumerates, and choosing the layout
  before planning a single press is what turns "where do I plant" into "in what
  order do I visit these twelve squares".  (The start square and the doorway are
  kept green by their own rules, so they need no flower -- and cannot take one.)

* **ICE MELTS UNDER YOUR FEET, ONE STEP BEHIND YOU.**  ``late [spring ice] ->
  [spring ice watermark]`` marks the ice you are standing on and ``late [no
  spring ice watermark] -> [water watermark]`` turns it to water the tick you
  step off.  Water shares the player's collision layer, so a walk across ice is
  SELF-AVOIDING and every crossing is a one-way door.  Level 9 is nothing but
  that: its only plantable square is the one blue cell beside the exit, so all
  forty-odd squares of ice have to be walked, and its 48-press plan is a
  Hamiltonian path found and proved by the same A* everything else uses.

* **A BUNNY RUNS UNTIL SOMETHING STOPS IT, THEN TURNS CLOCKWISE.**  The move
  rules are `late` and re-arm through ``late [moved] -> [] again``, so one
  keypress is a whole cascade: the player moves once, then every bunny advances
  a square per tick until none can, and only on that last tick -- gated on the
  `nomoved` marker the cascade strips whenever anything moved -- does the
  rotation block spin a blocked bunny clockwise to its first free side.  Three
  consequences.  A bunny crosses half a board in one press.  It EATS a flower it
  steps on (``[bunnyd no moved | no cre flower] -> [| bunnyd moved eaten]``),
  which un-lights nine squares and slams the door.  And `cre`, the group that
  stops it, contains ice but NOT water -- so a bunny walks into water, drowns,
  and ``[drowned][start] -> [start watermark]`` / ``[start watermark] ->
  restart`` throw the level away two presses later.  Melting ice near a bunny is
  therefore not a step forward, it is a countdown.

Three smaller facts, none of them guessable from the rule listing, all three
found by the differential fuzz:

* **A crate shoved onto a flower DESTROYS it.**  ``[> spring | crat | no cre no
  aqua flower] -> [|spring|crat]`` names the flower on its left and omits it on
  its right, and this interpreter (like PuzzleScript) drops what a rule names
  and does not re-place.
* **Melting the ice under your own feet DELETES you** -- `water` goes into the
  player's collision layer and the player goes out of it.  No shipped level can
  reach it (a flower melts its 3x3 on the tick it is planted, so ice never
  survives beside one), but the fuzz seats boards that can, and the model
  returns "dead" there rather than a board the interpreter would not agree with.
* **Level 5 arrives already won.**  ``right [winter no moved | block] -> [...
  winning]`` fires during `run_rules_on_level_start` while the winter sprite
  sweeps the board, so `check_win` is true on the start frame.  That is what
  `vacuous_start_win` is for; the level tapes one press, and `--verify` presses
  all five and checks each of them wins.

THE MODEL, AND WHY THERE IS ONE

`Board.step` is a native port of the whole turn -- the plant/un-plant pair, the
push, the melt, the bunny cascade and the rotation block, the door rebuild and
the win.  It exists because the interpreter runs at 855 presses/s on level 0 and
**50 presses/s on level 20** (a 9x15 board whose every tick repaints every floor
square and re-runs a nine-rule bunny cascade), which is three orders of
magnitude short of what these searches need.  ``--fuzz`` steps the model and the
interpreter side by side from three families of start state -- the level's own
start with no warm-up press, a walk steered at the rare rules, and a CROWDED
board re-seated with extra crates, flowers and ice so the configurations random
play never builds are the common case -- and requires identical boards and
identical win flags at every press.  53k presses over all twenty levels it
models, 0 mismatches, every branch of `step` exercised.

THE LADDER

Strongest tier first, and `describe` reports which one answered each level, so a
beam plan is never mistaken for a proof.

1. **A-star at weight 1** over the model with the admissible `LevelSolver.h`, which
   PROVES ``d*``.  `h` is one ACTION press per flower still needed (blue squares
   more than a Chebyshev 2 apart cannot share one) plus an MST over the player,
   the doorway, and one square standing for every remaining job -- a spanning
   TREE bound on a walk that has to visit them all in some order and end at the
   door.  The ice is in that terminal set, and putting it there is what took the
   estimate on the ice-ring levels from a third of the true answer to most of it.
2. **The bounded layered sweep**, which turns the proof into the labels.  A
   forward BFS keeping only ``g + h <= d*`` holds every state on every shortest
   path (``g(u) + h(u) <= g(u) + dist(u) = d*``), and a backward BFS over the
   edges it collected gives the exact presses-to-win for all of them -- so the
   optimal-action SET at every step is MEASURED, not inferred.  It is worth
   running whatever tier produced the plan: it is a complete search of that ball,
   so if it closes it has proved the length (level 15's beam-tier 35 turned out
   to be shortest that way).
3. **Layout first, then route.**  When A-star cannot close, enumerate the candidate
   final flower sets and A-star through each with ACTION allowed only on that
   layout's squares.  Every terminal is then an exact square instead of "somewhere
   in this 3x3", which is what makes the MST bite.  Every layout is tried, not
   just the cheapest-looking: level 12's plan runs through the 34th of its 44
   eight-flower layouts and the six cheapest are all unroutable.
4. **Probe, bound, prove** -- a weighted rung for an upper bound, then weight-1
   A-star pruned by it, which either improves it or closes.  The layout tier and the
   probe answer different levels better (level 12 is 40 through a layout and
   unfindable by the probe; level 14 is 35 by the probe and 36 through its best
   routable layout), so both run and the shorter wins.
5. **A greedy beam**, ordered by remaining WORK rather than by `h` -- the bound
   is deliberately loose in the middle of a big level and a beam ordered by it
   chases boards thirty presses from the end.

WHAT WAS INVISIBLE.  Three sprite fixes in
``data/puzzlescript_games/Spring.txt``, all of them found by ``--audit`` and none
of them touching a rule, a level, an object or a collision layer:

* **ice rendered exactly like blue floor** -- `lightblue` and ``#58BBE0`` are the
  same ARC palette index -- so the difference between "needs a flower" and "needs
  melting", which is most of the second half of the game, was not on the screen.
  Ice is now white.
* **a flower under the player was invisible**, and at ``cell_px=4`` so was the
  player: the two biggest levels render at four pixels a square, where the
  sampler drops the sprite's middle row and column, and both sprites carried all
  their detail there.  The player's eyes moved off the centre row and the flower
  became a yellow bloom with petals in the CORNERS -- the four pixels the player
  sprite leaves transparent -- so "am I standing on my own flower", which decides
  whether ACTION plants or uproots, reads at every size.
* the drowned-bunny marker was solid blue at ``cell_px=4``, i.e. indistinguishable
  from the water it is floating in.

``--audit`` is the regression test: it collects every cell composition the game
actually produces by playing all twenty-one levels, and requires them to render
differently at each of the seven cell sizes the levels use.  75 compositions, all
distinct at all seven.

VERIFIED.  19 of 21 levels, 509 presses.  ``--plans``: every plan replayed
through the real interpreter to a WIN; 15 levels proved shortest with measured
optimal-action sets (33 of the 509 steps carry a second right answer), 4 (levels
8, 12, 14, 19) not proved, and those label each step with the expert's own press
-- never `None`.  ``--verify``: 0 inadmissible states over every field it can
rebuild, and an INDEPENDENT exhaustive BFS -- no heuristic, no pruning, no bound
-- re-derives the same ``d*`` on all ten levels whose whole reachable space fits.
``--fuzz``: 0 mismatches.  ``--symmetry``: every level replayed through the
adapter at all four presentations, plans still WIN and both the plans and a
150-press random walk are exactly the transform of the unaugmented frames.  4
seeds x 19 levels recorded: 2084 steps, every one of them replaying frame-exact
through `perform_action` from the recorded SCREEN action, every level a WIN, all
509 expert presses labelled, and two processes at different `PYTHONHASHSEED`
emitting byte-identical episodes.  The longest level is 56 presses against the
adapter's 200-press cap.

The game takes the four ROTATIONS and not the mirrors.  "BUNNIES ALWAYS ROTATE
CLOCKWISE UPON HITTING A WALL" is chiral -- a mirrored presentation would show a
bunny turning the way this game never turns -- so Spring is deliberately not in
`PuzzleScriptAdapter._FLIP_GAMES`, and ``--symmetry`` checks the rotations it
does take.

THE TWO LEVELS THAT ARE SKIPPED (18 and 20) are the same level twice, small and
large: a ring of ice with a bunny running around inside it.  The ring has to melt
for the door to open, melting it makes water, and the bunny walks into water and
restarts the level -- so neither can be started until the bunny is shut away, and
both were built with somewhere to shut it: one square whose four sides are nest,
nest, crate, crate, at ``(4, 5)`` and ``(4, 7)``.  ``--pens`` measures it: level
18's box is 21 presses away and the BFS finds it; level 20's is not reachable at
all while planting is off (crossing its ice ring strands the corridor you came
from unless a flower goes down first, so the no-plant search closes at 16
states).  The flower tour that has to follow either box is beyond every tier
here -- fourteen of level 18's sixteen layouts refused to route in 2M nodes
apiece.  Both levels are in ``skip_levels`` so discovery does not pay for them at
every startup; everything needed to finish them (the model, the pen search, the
layout enumerator) is here and measured.

RECOVERY is the family's RESET prefix (``recovery_mode = "reset"``).  It has to
be: melted ice cannot be re-frozen, a bunny that has eaten a flower cannot be
made to spit it out, and a drowned bunny takes the level with it -- so one RESET
back to the level start is the only sound undo, and it lands on exactly the state
the cached plan was solved from.

CLI: ``--fuzz`` (the model against the interpreter), ``--verify`` (the cutscene,
admissibility, and an independent exhaustive BFS for ``d*``), ``--audit`` (every
composition the game can draw, pixel-distinct at every cell size), ``--symmetry``
(every presentation is an exact transform and every plan still wins),
``--pens`` (where each level's bunny box is and how far), ``--plans`` (every
level's plan, replayed through the interpreter), otherwise the BaseSolver CLI.
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                      # noqa: E402
from adapters.puzzlescript_adapter import (                       # noqa: E402
    PuzzleScriptAdapter, _render_cell_sprite,
)
from solvers.base_solver import _ID_TO_GAMEACTION                 # noqa: E402
from solvers.common.ps_astar import (                             # noqa: E402
    DIRECTIONS, PSAStarSolver, PSExpert, Plan, screen_action,
)
from utils.rotation import inverse_remap_action_full              # noqa: E402

GAME_ID = "ps:spring"
GAME_NAME = "Spring"

#: Engine press names, in the order the model indexes them.  ACTION is index 4
#: and is the plant / un-plant key; a direction pressed into a wall is a WAIT,
#: because every bunny rule fires regardless of what the player did.
PRESSES = ("up", "right", "down", "left", "action")

#: Face indices are the first four presses, in CLOCKWISE order, so a blocked
#: bunny's turn is ``(face + 1) % 4`` -- "BUNNIES ALWAYS ROTATE CLOCKWISE UPON
#: HITTING A WALL".
_DELTA = ((-1, 0), (0, 1), (1, 0), (0, -1))

_BUNNY_FACE = {"bunnyu": 0, "bunnyr": 1, "bunnyd": 2, "bunnyl": 3}

#: The engine gives up after this many ``again`` continuations.  A bunny run is
#: bounded by the board width, so the model never comes near it; exceeding it is
#: a modelling error and raises rather than returning a wrong board.
_MAX_TICKS = 50


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class Board:
    """Everything static about one level, plus `step`.

    A state is ``(player, flowers, ice, water, crates, bunnies)``: four cell
    BITMASKS over ``r * W + c``, one cell index and a tuple of
    ``(cell, face)``.  Nothing else survives a turn -- the floor colours, the
    door object, the ``moved`` / ``nomoved`` / ``vert`` / ``watermark`` markers
    and the whole ``m`` facing-stamp family are recomputed from these six by
    the game's own `late` rules every tick, so the model recomputes them too.
    """

    def __init__(self, eng, ids: dict[str, int]):
        grid = eng.grid
        self.H, self.W = len(grid), len(grid[0])
        self.N = self.H * self.W
        self.full = (1 << self.N) - 1
        blocks = glass = water = ice = crates = flowers = 0
        self.player = self.start = self.door = -1
        bunnies = []
        self.has_winter = False
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                i = r * self.W + c
                for name, idx in ids.items():
                    if idx not in cell:
                        continue
                    if name == "block":
                        blocks |= 1 << i
                    elif name == "glass":
                        glass |= 1 << i
                    elif name == "water":
                        water |= 1 << i
                    elif name == "ice":
                        ice |= 1 << i
                    elif name == "crat":
                        crates |= 1 << i
                    elif name == "flower":
                        flowers |= 1 << i
                    elif name == "spring":
                        self.player = i
                    elif name == "start":
                        self.start = i
                    elif name in ("door", "nodoor"):
                        self.door = i
                    elif name == "winter":
                        self.has_winter = True
                    elif name in _BUNNY_FACE:
                        bunnies.append((i, _BUNNY_FACE[name]))
        #: The winter cutscene ships with its win already granted and no door
        #: object anywhere, so there is nothing for the model to model.
        self.unmodelled = self.has_winter or self.door < 0
        if self.door < 0:
            self.door = self.start
        self.blocks, self.glass = blocks, glass
        self.orig_water = water
        self.nonblock = self.full & ~blocks
        self.start_state = (self.player, flowers, ice, water, crates,
                            tuple(sorted(bunnies)))

        # neighbours and the king 3x3, precomputed
        self.nb = [[-1] * 4 for _ in range(self.N)]
        self.cover = [0] * self.N
        for r in range(self.H):
            for c in range(self.W):
                i = r * self.W + c
                for f, (dr, dc) in enumerate(_DELTA):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.H and 0 <= cc < self.W:
                        self.nb[i][f] = rr * self.W + cc
                m = 0
                for rr in range(max(0, r - 1), min(self.H, r + 2)):
                    for cc in range(max(0, c - 1), min(self.W, c + 2)):
                        m |= 1 << (rr * self.W + cc)
                self.cover[i] = m
        self._lit: dict[int, int] = {}
        self._anchor = (1 << self.door) | (1 << self.start)

        # A cell that can NEVER hold a flower: a wall, a nest, water the level
        # ships, ice (ice is on the floor layer and no rule turns it back into
        # floor, so an ice cell is aqua for the whole episode), and the two
        # cells the `late [nodoor floor] -> [nodoor unlit]` /
        # `late [start floor] -> [start unlit]` pair keeps permanently lit --
        # `[action spring no unlit ...]` refuses to plant on a lit cell, so the
        # doorway and the square you start on are not plantable either.
        self.plantable = (self.nonblock & ~glass & ~water & ~ice & ~self._anchor)
        # Ice a flower can never reach, so the player must walk it: an
        # admissible and CONSISTENT count, because a plant only ever melts ice
        # that had a plantable cell beside it.
        self.walk_ice = 0
        for i in range(self.N):
            if ice >> i & 1 and not (self.cover[i] & self.plantable):
                self.walk_ice |= 1 << i

    # -- derived board facts -------------------------------------------------
    def lit(self, flowers: int) -> int:
        """Cells the `late [flow ...] -> [... unlit]` family paints GREEN: the
        3x3 around every flower, plus the doorway and the start square."""
        got = self._lit.get(flowers)
        if got is None:
            m = self._anchor
            f = flowers
            while f:
                b = f & -f
                m |= self.cover[b.bit_length() - 1]
                f ^= b
            self._lit[flowers] = m = got = m
        return got

    def nlit(self, flowers: int, ice: int, water: int) -> int:
        """Cells still BLUE.  ``late [no block no aqua] -> [nlit]`` paints every
        cell that is neither wall nor water nor ice, and the flower rules then
        turn some of them green; what is left is what keeps the door shut."""
        return self.nonblock & ~ice & ~water & ~self.lit(flowers)

    def door_shut(self, st) -> bool:
        """True when the `door` object is on the board.

        ``late [nodoor floor no player][nlit] -> [door]`` (and its `ice` twin)
        rebuild the door at the end of every tick as long as ONE blue cell or
        ONE ice cell is left anywhere -- and only while the player is not
        standing in the doorway, which is the whole win.
        """
        player, flowers, ice, water, _crates, _bun = st
        if player == self.door:
            return False
        return bool(self.nlit(flowers, ice, water)) or bool(ice)

    # -- the transition ------------------------------------------------------
    def step(self, st, press: int):
        """One keypress and the whole ``again`` cascade it starts.

        Returns ``(state, win)``, or ``None`` when a bunny drowned -- which is
        not a loss condition but a level RESTART three ticks later
        (``[drowned][start] -> [start watermark]``, ``[start watermark] ->
        restart``), so no plan may pass through it.
        """
        player, flowers, ice, water, crates, bunnies = st
        bun = list(bunnies)
        shut = self.door_shut(st)
        blocked_p = self.blocks | self.glass | crates | water
        prev = player

        # ---- MAIN rules, first tick only (an `again` continuation carries no
        # input force, so nothing here can fire again).
        if press == 4:
            pb = 1 << player
            if flowers & pb:
                flowers &= ~pb            # [action spring flower] -> [spring eaten]
            elif not (self.lit(flowers) & pb) and not ((ice | water) & pb):
                flowers |= pb             # [action spring no unlit no flower no aqua]
        else:
            t = self.nb[player][press]
            if t >= 0:
                occupied = any(b[0] == t for b in bun) or (t == self.door and shut)
                if crates >> t & 1:
                    u = self.nb[t][press]
                    # [> spring | crat | no cre no aqua] -- the far cell must be
                    # clear of everything, ice and water included.
                    if (u >= 0 and not ((blocked_p | ice) >> u & 1)
                            and not any(b[0] == u for b in bun)
                            and not (u == self.door and shut)):
                        crates = (crates & ~(1 << t)) | (1 << u)
                        # `[> spring | crat | no cre no aqua flower] ->
                        # [|spring|crat]` names the flower and drops it: a
                        # crate shoved onto a flower EATS it.
                        flowers &= ~(1 << u)
                        player = t
                elif not (blocked_p >> t & 1) and not occupied:
                    player = t

        # ---- the LATE phase, once per `again` tick
        for tick in range(_MAX_TICKS):
            # late [spring ice] -> [... watermark] ; late [no spring ice
            # watermark] -> [water watermark].  The mark is re-stamped every
            # tick under the player, so the only cell it can melt is the ice
            # square the player has just stepped OFF.
            if tick == 0 and prev != player and (ice >> prev & 1):
                ice &= ~(1 << prev)
                water |= 1 << prev

            wmark = water & ~self.orig_water
            # late [bunny watermark] -> [drowned]
            if any(wmark >> b[0] & 1 for b in bun):
                return None

            # late <dir> [bunnyX no moved | no cre (no) flower] -> [| bunnyX moved]
            # cre = spring or block or crat or glass or door or bunny or ice, so
            # a bunny is stopped by ice and walks into water; the flower variant
            # comes first in the file and EATS what it steps on.
            moved = False
            eaten = 0
            occupied = {b[0] for b in bun}
            for want_flower in (True, False):
                for face in (0, 3, 1, 2):          # up, left, right, down
                    for bi, (pos, bf) in enumerate(bun):
                        if bf != face:
                            continue
                        t = self.nb[pos][face]
                        if t < 0 or t in occupied:
                            continue
                        if ((self.blocks | self.glass | crates | ice) >> t & 1
                                or t == player
                                or (t == self.door and shut)):
                            continue
                        if bool(flowers >> t & 1) != want_flower:
                            continue
                        if want_flower:
                            flowers &= ~(1 << t)
                            eaten |= 1 << t
                        occupied.discard(pos)
                        occupied.add(t)
                        bun[bi] = (t, bf)
                        moved = True

            # late right [|nodoor player][eaten] -> [player|door][eaten]:
            # a bunny eating a flower while you stand in the doorway SHOVES you
            # back out of it, one cell to the left, and re-shuts the door.
            if eaten:
                if player == self.door and self.door % self.W:
                    back = self.door - 1
                    if not ((self.blocks | self.glass | crates | water)
                            >> back & 1):
                        player = back
                        occupied.discard(back)
                        bun = [b for b in bun if b[0] != back]
                shut = shut or player != self.door

            # late [flow |ice] -> [flow |water watermark] (and its `vert`
            # diagonal twin): a flower melts every ice square in its 3x3.
            if ice and flowers:
                melt = ice & self.lit(flowers)
                if melt:
                    ice &= ~melt
                    water |= melt
                    if melt >> player & 1:
                        # `water` shares the player's collision layer, so the
                        # rule that melts the ice under your feet DELETES you.
                        # No shipped level can reach it (a flower melts its 3x3
                        # on the tick it is planted, so ice never survives
                        # beside one), but the fuzz seats boards that can.
                        return None

            if not moved:
                # The rotation block is gated on `nomoved`, which `late [nomoved
                # spring][moved] -> [spring]` strips the moment anything moved.
                # So a bunny turns only on the tick its run ends: spin clockwise
                # to the first free side, and if there is none stay put (that is
                # what the fourth, `no upm`-guarded copy of the block buys).
                occupied = {b[0] for b in bun}
                stop = self.blocks | self.glass | crates | ice
                for bi, (pos, bf) in enumerate(bun):
                    # k == 0 first: the block is a chain of
                    # `[bunnyX | cre] -> [bunny<cw X> | cre]`, so a bunny whose
                    # OWN side came free during the tick (the flower melt above
                    # runs before this) does not turn at all.
                    for k in range(4):
                        nf = (bf + k) & 3
                        t = self.nb[pos][nf]
                        if (t >= 0 and not (stop >> t & 1) and t != player
                                and t not in occupied
                                and not (t == self.door and shut)):
                            bun[bi] = (pos, nf)
                            break
                st = (player, flowers, ice, water, crates,
                      tuple(sorted(bun)))
                shut = self.door_shut(st)
                # late [spring nodoor no watermark nomoved] -> [... winning]
                return st, player == self.door
            st = (player, flowers, ice, water, crates, tuple(sorted(bun)))
            shut = self.door_shut(st)
        raise AssertionError("again loop did not settle")

    def read(self, eng, ids: dict[str, int]):
        """The model state the interpreter's grid is showing (fuzz only)."""
        flowers = ice = water = crates = 0
        player = -1
        bun = []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i = r * self.W + c
                for name, idx in ids.items():
                    if idx not in cell:
                        continue
                    if name == "flower":
                        flowers |= 1 << i
                    elif name == "ice":
                        ice |= 1 << i
                    elif name == "water":
                        water |= 1 << i
                    elif name == "crat":
                        crates |= 1 << i
                    elif name == "spring":
                        player = i
                    elif name in _BUNNY_FACE:
                        bun.append((i, _BUNNY_FACE[name]))
        return (player, flowers, ice, water, crates, tuple(sorted(bun)))

    def show(self, st) -> str:
        player, flowers, ice, water, crates, bun = st
        faces = "^>v<"
        out = []
        for r in range(self.H):
            row = []
            for c in range(self.W):
                i = r * self.W + c
                ch = "."
                if self.blocks >> i & 1:
                    ch = "#"
                elif water >> i & 1:
                    ch = "~"
                elif ice >> i & 1:
                    ch = "I"
                elif not (self.nlit(flowers, ice, water) >> i & 1):
                    ch = "+"
                if self.glass >> i & 1:
                    ch = "Q"
                if crates >> i & 1:
                    ch = "7"
                if flowers >> i & 1:
                    ch = "*"
                for pos, bf in bun:
                    if pos == i:
                        ch = faces[bf]
                if i == self.door:
                    ch = "d" if not self.door_shut(st) else "D"
                if i == player:
                    ch = "S"
                row.append(ch)
            out.append("".join(row))
        return "\n".join(out)


_OBJECTS = ("block", "glass", "water", "ice", "crat", "flower", "spring",
            "start", "door", "nodoor", "winter", "bunnyu", "bunnyr", "bunnyd",
            "bunnyl", "drowned", "winning", "background", "unlit", "nlit",
            "watermark")


def _ids(game) -> dict[str, int]:
    return {n: game._game.obj_name_to_idx[n] for n in _OBJECTS}


def _seat(eng, board: Board, ids: dict[str, int], st) -> None:
    """Write a model state back into the interpreter's grid.

    Only the objects a tick can READ have to be seated: the main rules open by
    deleting `vert`, `eaten`, `nomoved` and the `m` facing stamps, and
    ``[floor] -> []`` throws the floor away and the `late` block rebuilds it --
    so the floor colours seated here matter for exactly one thing, the
    ``[action spring no unlit ...]`` plant test, and they are computed the same
    way the game computes them.
    """
    player, flowers, ice, water, crates, bun = st
    bg = ids["background"]
    lit = board.lit(flowers)
    shut = board.door_shut(st)
    faces = {v: k for k, v in _BUNNY_FACE.items()}
    grid = []
    for r in range(board.H):
        row = []
        for c in range(board.W):
            i = r * board.W + c
            cell = {bg}
            if board.blocks >> i & 1:
                cell.add(ids["block"])
            if board.glass >> i & 1:
                cell.add(ids["glass"])
            if crates >> i & 1:
                cell.add(ids["crat"])
            if flowers >> i & 1:
                cell.add(ids["flower"])
            if ice >> i & 1:
                cell.add(ids["ice"])
            elif water >> i & 1:
                cell.add(ids["water"])
            elif not (board.blocks >> i & 1):
                cell.add(ids["unlit"] if lit >> i & 1 else ids["nlit"])
            if water >> i & 1 and not (board.orig_water >> i & 1):
                cell.add(ids["watermark"])
            if i == board.start:
                cell.add(ids["start"])
            if i == board.door:
                cell.add(ids["door"] if shut else ids["nodoor"])
            if i == player:
                cell.add(ids["spring"])
            for pos, bf in bun:
                if pos == i:
                    cell.add(ids[faces[bf]])
            row.append(cell)
        grid.append(row)
    eng.grid = grid
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

def _bits(m: int):
    while m:
        b = m & -m
        yield b.bit_length() - 1
        m ^= b


class LevelSolver:
    """The searches, over one level's model.  See `SpringExpert._search` for
    the order they are tried in and the module docstring for why.

    `astar` (weight 1, the admissible `h`) proves ``d*``; `sweep` turns that
    proof into the exact distance-to-win field and so into MEASURED optimal
    sets; `layouts` + `route` pick the final flower set first and plan the tour
    through it; `pen_search` / `pen_beam` look for the state where a bunny is
    shut away for good; `beam` is the last resort.  Everything above the beam
    shares one precomputation: the relaxed all-pairs walk distances, the
    per-square set of places a flower could light it from, and the walkable
    regions of a board keyed on its water.
    """

    def __init__(self, board: Board, node_cap: int = 600_000):
        self.b = board
        self.node_cap = node_cap
        n, W, H = board.N, board.W, board.H
        # cells within Chebyshev 2 -- the pairs that COULD share one flower, so
        # a set of cells pairwise outside it needs one flower each
        self.near2 = [0] * n
        for r in range(H):
            for c in range(W):
                m = 0
                for rr in range(max(0, r - 2), min(H, r + 3)):
                    for cc in range(max(0, c - 2), min(W, c + 3)):
                        m |= 1 << (rr * W + cc)
                self.near2[r * W + c] = m
        self.door_dist = self._walk_field(board.door)
        # Where a cell's light can come from: the plantable squares of its own
        # 3x3.  For ice no flower can reach that set is empty and the only
        # remedy is the player's own foot, so the cell stands for itself.
        self.reg = [0] * n
        orig_ice = board.start_state[2]
        for i in range(n):
            m = board.cover[i] & board.plantable
            if orig_ice >> i & 1:
                # An ice square is discharged EITHER by a flower in its 3x3 or
                # by the player's own foot, so both count as touching it.  The
                # two kinds never mix: no rule turns ice back into floor and
                # none turns floor into ice, so a cell is one or the other for
                # the whole episode and one table covers both.
                m |= 1 << i
            self.reg[i] = m if m else (1 << i)
        self.regfield = [self._walk_field_multi(self.reg[i]) for i in range(n)]
        self.regdoor = [min((self.door_dist[q] for q in _bits(self.reg[i])
                             if self.door_dist[q] >= 0), default=n)
                        for i in range(n)]
        self.jobdist = [[min((self.regfield[i][q] for q in _bits(self.reg[j])
                              if self.regfield[i][q] >= 0), default=n)
                         for j in range(n)] for i in range(n)]
        self.cdist = [self._walk_field(i) for i in range(n)]
        self._spread_cache: dict[int, int] = {}
        self._comp: dict[int, list[int]] = {}
        self._jobs: dict[tuple, tuple] = {}
        #: Squares whose four sides are all wall, nest or a crate where the
        #: level put it -- the box a bunny can be shut into for good.
        self.pens = [i for i in range(board.N)
                     if not ((board.blocks | board.glass) >> i & 1)
                     and all(j >= 0 and ((board.blocks | board.glass
                                          | board.start_state[4]) >> j & 1)
                             for j in board.nb[i])]
        #: Set once a bunny has been penned: from then on no successor may let
        #: one out again (pushing the crate back off would hand it the water).
        self.pen_guard = False

    # -- static relaxations ---------------------------------------------------
    def _walk_field(self, src: int) -> list[int]:
        """Presses to walk to ``src`` ignoring everything that can move.

        Crates are pushable and ice is walkable, so treating both as floor only
        ever shortens the estimate; the level's own water and its nests and
        walls never move, so they stay.
        """
        b = self.b
        blocked = b.blocks | b.glass | b.orig_water
        dist = [-1] * b.N
        dist[src] = 0
        q = deque([src])
        while q:
            i = q.popleft()
            for j in b.nb[i]:
                if j >= 0 and dist[j] < 0 and not (blocked >> j & 1):
                    dist[j] = dist[i] + 1
                    q.append(j)
        return dist

    def _walk_field_multi(self, sources: int) -> list[int]:
        """`_walk_field` from a SET of squares at once."""
        b = self.b
        blocked = b.blocks | b.glass | b.orig_water
        dist = [-1] * b.N
        q = deque()
        for i in _bits(sources):
            if not (blocked >> i & 1):
                dist[i] = 0
                q.append(i)
        while q:
            i = q.popleft()
            for j in b.nb[i]:
                if j >= 0 and dist[j] < 0 and not (blocked >> j & 1):
                    dist[j] = dist[i] + 1
                    q.append(j)
        return dist

    def _components(self, water: int) -> list[int]:
        """One bitmask per connected walkable region, for the given water.

        Cached on ``water`` alone: it is the only thing in a state that can cut
        the board in two (crates are pushable, ice is walkable and melts into
        water, which is what this is keyed on).
        """
        got = self._comp.get(water)
        if got is not None:
            return got
        b = self.b
        blocked = b.blocks | b.glass | water
        seen = 0
        out = []
        for i in range(b.N):
            if (blocked >> i & 1) or (seen >> i & 1):
                continue
            m = 1 << i
            q = [i]
            seen |= m
            while q:
                k = q.pop()
                for j in b.nb[k]:
                    if j >= 0 and not (blocked >> j & 1) and not (seen >> j & 1):
                        seen |= 1 << j
                        m |= 1 << j
                        q.append(j)
            out.append(m)
        self._comp[water] = out
        return out

    def _region(self, player: int, water: int) -> int:
        for m in self._components(water):
            if m >> player & 1:
                return m
        return 1 << player

    def _spread(self, region: int) -> int:
        """Cells a flower could still light from inside one walkable region."""
        got = self._spread_cache.get(region)
        if got is not None:
            return got
        b = self.b
        m = 0
        for i in _bits(b.plantable & region):
            m |= b.cover[i]
        self._spread_cache[region] = m
        return m

    # -- pruning and cost -----------------------------------------------------
    def dead(self, st) -> bool:
        """States that provably cannot win, dropped un-expanded.

        Three ways this game closes a door behind you, and all three are about
        water: melting the ice under a cell you still need to plant on, melting
        a moat between yourself and the doorway, and melting the last dry route
        to an ice square only a walk can reach.
        """
        b = self.b
        player, flowers, ice, water, crates, bun = st
        if self.pen_guard and not self.penned(st):
            return True                       # a bunny let back out of its box
        region = self._region(player, water)
        if not (region >> b.door & 1):
            return True                       # the doorway is behind the water
        if ice & b.walk_ice & ~region:
            return True                       # ice only a walk can melt, marooned
        need = b.nonblock & ~ice & ~water & ~b.lit(flowers)
        if need & ~self._spread(region):
            return True                       # blue cell, no reachable plant site
        return False

    def h(self, st) -> int:
        """A lower bound on the presses left, in two independent halves.

        ``plants``: every cell still blue has to end up inside some flower's
        3x3, and two cells more than a Chebyshev 2 apart cannot share one, so a
        greedy pairwise-far selection counts flowers that must still be planted
        -- one ACTION press each.

        ``max(walk, travel)``: both are lower bounds on the MOVE presses left.
        `walk_ice` is the ice no flower can ever reach (a static fact, so a
        plant can never discharge it), and each such square costs its own press
        to step onto; `door_dist` is the walk that has to end the level.  The
        two halves never charge the same press, so their sum is admissible.
        """
        b = self.b
        player, flowers, ice, water, _crates, _bun = st
        if player == b.door:
            return 0
        key = (flowers, ice, water)
        got = self._jobs.get(key)
        if got is None:
            blue = b.nonblock & ~ice & ~water & ~b.lit(flowers)
            rem = blue
            plants = 0
            while rem:
                i = (rem & -rem).bit_length() - 1
                plants += 1
                rem &= ~self.near2[i]
            # The MST terminals are the blue cells AND the ice: melting is work
            # too, and on the levels built round an ice ring it is most of it.
            # Leaving the ice out left `h` three times short of `d*` there.
            rem = blue | ice
            jobs = []
            while rem:
                i = (rem & -rem).bit_length() - 1
                jobs.append(i)
                rem &= ~self.near2[i]
            self._jobs[key] = got = (plants, jobs)
        plants, jobs = got
        return plants + self._mst(player, jobs)

    def _mst(self, player: int, jobs: list) -> int:
        """A lower bound on the MOVE presses left: the minimum spanning tree
        over the player, the doorway and one representative of every job.

        The walk that finishes the level touches each job's squares in SOME
        order and ends at the door, so its length is at least the weight of
        that spanning PATH, and a path is a spanning tree -- so at least the
        MST.  Every distance in it is measured on the relaxed board (crates and
        ice count as floor, only walls, nests and the level's own water do
        not), which can only make it smaller.
        """
        big = self.b.N
        far = self.door_dist[player]
        if far < 0:
            far = big
        if not jobs:
            return far
        # node 0 is the player, 1..k the jobs, k+1 the door
        k = len(jobs)
        best = [self.regfield[j][player] for j in jobs]
        best = [b if b >= 0 else big for b in best] + [far]
        used = [False] * (k + 1)
        total = 0
        for _ in range(k + 1):
            pick, pv = -1, 1 << 30
            for i in range(k + 1):
                if not used[i] and best[i] < pv:
                    pick, pv = i, best[i]
            if pick < 0:
                break
            used[pick] = True
            total += pv
            if pick == k:                       # the door joined the tree
                for i in range(k):
                    if not used[i]:
                        d = self.regdoor[jobs[i]]
                        if d < best[i]:
                            best[i] = d
            else:
                row = self.jobdist[jobs[pick]]
                for i in range(k):
                    if not used[i]:
                        d = row[jobs[i]]
                        if 0 <= d < best[i]:
                            best[i] = d
                d = self.regdoor[jobs[pick]]
                if d < best[k]:
                    best[k] = d
        return total

    # -- tier 1: A* ------------------------------------------------------------
    def astar(self, start, weight: int = 1, bound: int | None = None,
              node_cap: int | None = None):  # -> (path, nodes, closed)
        """Shortest (at ``weight == 1``) press sequence to a win, or None.

        ``bound`` prunes every node whose ``g + h`` already reaches a known
        winning length: a bounded run that CLOSES with nothing shorter is the
        optimality proof, and one that improves the bound is the next probe.
        """
        b = self.b
        cap = self.node_cap if node_cap is None else node_cap
        pq = [(weight * self.h(start), 0, 0, start)]
        best = {start: 0}
        parent: dict = {start: None}
        nodes = 0
        counter = 0
        while pq:
            _f, g, _c, st = heapq.heappop(pq)
            if best.get(st, 1 << 30) < g:
                continue
            for press in range(5):
                got = b.step(st, press)
                nodes += 1
                if got is None:
                    continue
                nxt, win = got
                if win:
                    return _trace(parent, st) + [press], nodes, False
                if nxt == st or best.get(nxt, 1 << 30) <= g + 1:
                    continue
                if self.dead(nxt):
                    continue
                hn = self.h(nxt)
                if bound is not None and g + 1 + hn >= bound:
                    continue
                best[nxt] = g + 1
                parent[nxt] = (st, press)
                counter += 1
                heapq.heappush(pq, (g + 1 + weight * hn, g + 1, counter, nxt))
            if nodes >= cap:
                return None, nodes, False
        return None, nodes, True

    # -- tier 2: the exact field on the shortest-path ball ---------------------
    def sweep(self, start, dstar: int, state_cap: int = 1_200_000):
        """The exact distance-to-win field within ``d*`` of ``start``.

        A layered forward BFS keeps a state only while ``g + h <= d*``.  With an
        admissible ``h`` and a BFS ``g`` (which is the true shortest distance to
        that state) every state on every shortest path survives: ``g(u) + h(u)
        <= g(u) + dist(u) = d*``.  A backward BFS over the edges collected on
        the way then gives the exact presses-to-win for all of them, and the
        optimal SET at each step is every press that goes one closer -- measured
        rather than inferred.
        """
        b = self.b
        succ: dict = {}
        gval = {start: 0}
        q = deque([start])
        while q:
            st = q.popleft()
            g = gval[st]
            edges = {}
            for press in range(5):
                got = b.step(st, press)
                if got is None:
                    continue
                nxt, win = got
                if win:
                    edges[press] = None            # a winning press
                    continue
                if nxt == st:
                    continue
                if nxt in gval:
                    if gval[nxt] == g + 1:
                        edges[press] = nxt
                    continue
                if g + 1 + self.h(nxt) > dstar or self.dead(nxt):
                    continue
                gval[nxt] = g + 1
                edges[press] = nxt
                q.append(nxt)
            succ[st] = edges
            if len(succ) > state_cap:
                return None
        dist = _backward(succ)
        got = dist.get(start)
        if got is None or got > dstar:
            return None
        dstar = got                    # the sweep can also IMPROVE the bound
        presses, optsets = [], []
        st = start
        while True:
            best = [p for p in range(5)
                    if _cost(succ, dist, st, p) == dist[st]]
            presses.append(best[0])
            optsets.append(best)
            nxt = succ[st][best[0]]
            if nxt is None:
                break
            st = nxt
        return presses, optsets, len(succ)

    # -- tier 2b: decide the flower LAYOUT first, then route through it -------
    def layouts(self, melt: bool = False, extra: int = 0, forbid: int = 0,
                limit: int = 200, budget: int = 300_000):
        """Candidate FINAL flower sets, cheapest-looking first.

        The end state of every level is the same shape: a set of flowers that
        (a) sit on plantable squares, (b) are pairwise more than a king step
        apart -- two adjacent flowers cannot both exist, because the second
        would have to be planted on a square the first has already turned green
        and `[action spring no unlit ...]` refuses that -- and (c) light every
        square that is not wall, water or ice.  In other words an independent
        DOMINATING set, and because (b) is symmetric the ORDER the flowers go
        in never matters: any independent dominating set is reachable in any
        order.  So the layout can be chosen before a single press is planned,
        which turns the search from "where should I plant" into "in what order
        do I visit these twelve squares", and that is a problem a heuristic can
        actually see the end of.
        """
        b = self.b
        orig_ice = b.start_state[2]
        need = b.nonblock & ~orig_ice & ~b.orig_water & ~b._anchor
        if melt:
            # The other half of the job.  A layout chosen to light the FLOOR
            # leaves every ice square to the player's own feet, and walking ice
            # is not free: the square turns to water behind you, so the route
            # over it is self-avoiding, and on a level built round an ice ring
            # there is no such route.  Ask the flowers to melt it instead.
            need |= orig_ice & ~b.walk_ice
        opts = {}
        for c in _bits(need):
            o = b.cover[c] & b.plantable & ~forbid
            if not o:
                return []                       # a square no flower can light
            opts[c] = o
        lb, rem = 0, need
        while rem:
            i = (rem & -rem).bit_length() - 1
            lb += 1
            rem &= ~self.near2[i]
        out: list[frozenset] = []
        seen: set[frozenset] = set()
        spent = [0]

        def rec(covered: int, chosen: tuple, cap: int) -> None:
            if len(out) >= limit or spent[0] >= budget:
                return
            spent[0] += 1
            rest = need & ~covered
            if not rest:
                key = frozenset(chosen)
                if key not in seen:
                    seen.add(key)
                    out.append(key)
                return
            if len(chosen) >= cap:
                return
            pick, avail = -1, 0
            for c in _bits(rest):
                o = opts[c] & ~covered
                if not o:
                    return                      # this square is now unlightable
                n = bin(o).count("1")
                if pick < 0 or n < bin(avail).count("1"):
                    pick, avail = c, o
                    if n == 1:
                        break
            for site in _bits(avail):
                rec(covered | b.cover[site], chosen + (site,), cap)

        # The greedy lower bound is not the minimum: level 19 needs eight
        # flowers where `lb` says six.  So climb until something is found, and
        # only then optionally one size further -- the variety that matters is
        # WITHIN a size (level 12's shortest plan uses the 34th of its 44
        # eight-flower layouts), not between sizes.
        room = extra
        for cap in range(lb, lb + 7):
            rec(0, (), cap)
            if out:
                if room <= 0:
                    break
                room -= 1
            if len(out) >= limit or spent[0] >= budget:
                break
        return sorted(out, key=self.layout_cost)

    def layout_cost(self, sites: frozenset) -> int:
        """A lower bound on a plan that ends in exactly this layout: one ACTION
        per flower plus the shortest tour that visits every one of them, every
        ice square none of them reaches, and finally the door."""
        b = self.b
        mask = 0
        for s in sites:
            mask |= b.cover[s]
        walk = b.start_state[2] & ~mask
        return len(sites) + self._mst_cells(
            b.start_state[0], list(sites) + list(_bits(walk)))

    def _mst_cells(self, player: int, cells: list) -> int:
        """MST over the player, a list of exact squares and the doorway."""
        big = self.b.N
        far = self.door_dist[player]
        if far < 0:
            far = big
        if not cells:
            return far
        k = len(cells)
        pf = self.cdist[player]
        best = [pf[c] if pf[c] >= 0 else big for c in cells] + [far]
        used = [False] * (k + 1)
        total = 0
        for _ in range(k + 1):
            pick, pv = -1, 1 << 30
            for i in range(k + 1):
                if not used[i] and best[i] < pv:
                    pick, pv = i, best[i]
            if pick < 0:
                break
            used[pick] = True
            total += pv
            if pick == k:
                for i in range(k):
                    d = self.door_dist[cells[i]]
                    if not used[i] and 0 <= d < best[i]:
                        best[i] = d
            else:
                row = self.cdist[cells[pick]]
                for i in range(k):
                    if not used[i] and 0 <= row[cells[i]] < best[i]:
                        best[i] = row[cells[i]]
                d = self.door_dist[cells[pick]]
                if 0 <= d < best[k]:
                    best[k] = d
        return total

    def route(self, start, sites: frozenset, node_cap: int = 400_000,
              bound: "int | None" = None):
        """A* to a win, with ACTION allowed ONLY on the chosen layout's squares.

        The restriction is what makes the bound sharp: every terminal is now an
        exact square rather than "somewhere in this 3x3", so the MST over them
        is a real tour bound instead of the near-zero one the free search gets
        when every job's region overlaps every other's.
        """
        b = self.b
        mask = 0
        for s in sites:
            mask |= b.cover[s]
        site_mask = 0
        for s in sites:
            site_mask |= 1 << s
        walk_mask = b.start_state[2] & ~mask

        def hr(st):
            player, flowers, ice, _water, _crates, _bun = st
            if player == b.door:
                return 0
            todo = site_mask & ~flowers
            cells = list(_bits(todo)) + list(_bits(ice & walk_mask))
            return bin(todo).count("1") + self._mst_cells(player, cells)

        pq = [(hr(start), 0, 0, start)]
        best = {start: 0}
        parent = {start: None}
        nodes = counter = 0
        while pq:
            _f, g, _c, st = heapq.heappop(pq)
            if best.get(st, 1 << 30) < g:
                continue
            player, flowers = st[0], st[1]
            for press in range(5):
                if press == 4 and not ((site_mask & ~flowers) >> player & 1):
                    continue                    # never plant off the layout
                got = b.step(st, press)
                nodes += 1
                if got is None:
                    continue
                nxt, win = got
                if win:
                    return _trace(parent, st) + [press], nodes
                if nxt == st or best.get(nxt, 1 << 30) <= g + 1:
                    continue
                if nxt[1] & ~site_mask or self.dead(nxt):
                    continue
                if bound is not None and g + 1 + hr(nxt) >= bound:
                    continue
                best[nxt] = g + 1
                parent[nxt] = (st, press)
                counter += 1
                heapq.heappush(pq, (g + 1 + hr(nxt), g + 1, counter, nxt))
            if nodes >= node_cap:
                return None, nodes
        return None, nodes

    # -- tier 2c: shut the bunny in a box first --------------------------------
    def penned(self, st) -> bool:
        """True when no bunny can ever move again.

        A bunny is stopped by `cre`, and of the things in that group only the
        crates and the player can move -- so a bunny with wall, nest or crate on
        all four sides is finished, and the last two levels are built around
        exactly one such square: a nest either side and a pushable crate above
        and below.  It is not decoration.  Melting the ice ring those levels are
        wrapped in turns it into water, a bunny walks into water and drowns, and
        a drowned bunny restarts the level three presses later -- so the ring
        cannot come down until the bunny is in the box.
        """
        b = self.b
        crates, bun = st[4], st[5]
        stop = b.blocks | b.glass | crates
        for pos, _face in bun:
            for j in b.nb[pos]:
                if j < 0 or not (stop >> j & 1):
                    return False
        return True

    def pen_search(self, start, node_cap: int = 2_000_000, keep: int = 12):
        """Shortest press sequences that shut every bunny in, by plain BFS.

        Planting is forbidden for the whole of it -- a flower changes nothing
        about where a bunny can walk, and leaving ACTION out of the branching is
        what keeps the search inside a couple of million states.  Melting is
        NOT forbidden: crossing the ice ring is how you get to the crates at
        all, and the water it leaves is exactly what makes the pen urgent.
        """
        b = self.b
        self.pen_states = 0
        if not self.pens or not start[5]:
            return []
        seen = {start: None}
        layer = [start]
        while layer:
            nxt = []
            hits = []
            for st in layer:
                for press in range(4):
                    got = b.step(st, press)
                    if got is None:
                        continue
                    ns, _win = got
                    if ns[1] != start[1] or ns == st or ns in seen:
                        continue
                    if self.dead(ns):
                        continue
                    seen[ns] = (st, press)
                    if self.penned(ns):
                        hits.append(ns)
                        if len(hits) >= keep:
                            break
                    else:
                        nxt.append(ns)
                if len(hits) >= keep:
                    break
            self.pen_states = len(seen)
            if hits:
                return [(_trace(seen, h), h) for h in hits]
            if len(seen) > node_cap:
                return []
            layer = nxt
        return []

    def pen_score(self, st) -> int:
        """Beam order for the pen manoeuvre, in two phases.

        Getting the bunny INTO the box and shutting the box behind it want
        opposite things -- the box has to be open for the first and closed for
        the second -- so a single "distance to a penned board" number pulls both
        ways and the beam oscillates.  Phase one is "walk the bunny to the pen",
        phase two, worth strictly more than any phase-one board, is "stand where
        ONE press shoves a crate back into the gap".  That last term is the
        whole trick: the bunny leaves the box on the very next press, so the
        player has to be lined up before it arrives, and a plain
        distance-to-the-pen score puts it in the doorway instead -- where its
        own body holds the bunny in and no press can ever close the gap.
        """
        b = self.b
        player, _f, _i, _w, crates, bun = st
        pen = self.pens[0]
        pos = bun[0][0]
        if pos != pen:
            d = self.cdist[pen][pos]
            dp = self.cdist[pos][player]
            return 100 + 8 * (d if d >= 0 else b.N) + (dp if dp >= 0 else b.N)
        stop = b.blocks | b.glass | crates
        opens = [j for j in b.nb[pen] if j < 0 or not (stop >> j & 1)]
        ready = 20
        for j in opens:
            if j < 0:
                continue
            jr, jc = divmod(j, b.W)
            for k in b.nb[j]:
                if k < 0 or not (crates >> k & 1):
                    continue
                kr, kc = divmod(k, b.W)
                sr, sc = kr - (jr - kr), kc - (jc - kc)
                if 0 <= sr < b.H and 0 <= sc < b.W:
                    d = self.cdist[sr * b.W + sc][player]
                    if 0 <= d < ready:
                        ready = d
        return 3 * len(opens) + ready

    def pen_beam(self, start, widths=(800, 3000), limit: int = 200):
        """`pen_search` for the levels whose pen is out of BFS range."""
        b = self.b
        if not self.pens or len(start[5]) != 1:
            return []
        for width in widths:
            layer = {start: ()}
            seen = {start}
            for _step in range(limit):
                cand: dict = {}
                for st, path in layer.items():
                    for press in range(5):
                        got = b.step(st, press)
                        if got is None:
                            continue
                        ns, _win = got
                        if ns == st or ns in seen or ns in cand:
                            continue
                        if self.dead(ns):
                            continue
                        if self.penned(ns):
                            return [(list(path + (press,)), ns)]
                        cand[ns] = path + (press,)
                if not cand:
                    break
                keep = sorted(cand, key=self.pen_score)[:width]
                seen.update(keep)
                layer = {k: cand[k] for k in keep}
        return []

    def layout_plan(self, start, deadline: float, node_cap: int = 120_000,
                    forbid: int = 0):
        """The best plan any candidate layout can be routed through.

        Every layout is tried, not just the cheapest-looking ones: the bound
        `layout_cost` ranks by is about the TOUR, and what actually decides a
        layout here is whether its order can be realised at all -- ps:spring
        level 12's shortest plan runs through the 34th-cheapest of its 44
        layouts, and the six cheapest are all unroutable (their flowers melt a
        moat across the board before the last of them can be reached).
        """
        best = None
        for extra in (0, 1):
            for melt in (True, False):
                for lay in self.layouts(melt=melt, extra=extra,
                                        forbid=forbid):
                    if time.time() > deadline:
                        return best
                    got, _n = self.route(
                        start, lay, node_cap=node_cap,
                        bound=None if best is None else len(best))
                    if got is not None and (best is None
                                            or len(got) < len(best)):
                        best = got
            if best is not None:
                break
        return best

    def progress(self, st) -> int:
        """How much of the JOB is left, which is not the same as `h`.

        `h` has to be a lower bound and is therefore loose in the middle of a
        big level -- it happily reads 8 on a board thirty presses from the end,
        because a single flower can light nine squares at once and the bound
        must assume it will.  A beam ordered by it chases those boards.  This
        counts the actual remaining work instead -- blue squares plus unmelted
        ice, with the walk home as the tie-break -- and is used ONLY to order
        the beam, never to prune or to prove anything.
        """
        b = self.b
        player, flowers, ice, water, _crates, _bun = st
        blue = b.nonblock & ~ice & ~water & ~b.lit(flowers)
        d = self.door_dist[player]
        return 6 * (bin(blue).count("1") + bin(ice).count("1")) + \
            (d if d >= 0 else b.N)

    # -- tier 3: a greedy beam -------------------------------------------------
    def beam(self, start, widths=(400, 1500, 6000), limit: int = 300,
             sites: "frozenset | None" = None):
        """Layered best-first over the same model, ordered by the bound.

        Reached for only when the exact pass cannot close a level.  Two details
        matter more than the width.  A state that is GENERATED but does not make
        the cut must NOT be marked seen -- doing that walls the beam off from
        every board it declined once and was what made the widest levels here
        return "no plan" rather than a long one.  And the widths are climbed
        rather than fixed, so an easy level is not charged for a hard one.
        """
        b = self.b
        site_mask = None
        if sites is not None:
            site_mask = 0
            for site in sites:
                site_mask |= 1 << site
        for width in widths:
            layer = {start: ()}
            seen = {start}
            for _step in range(limit):
                cand: dict = {}
                for st, path in layer.items():
                    for press in range(5):
                        if (site_mask is not None and press == 4
                                and not ((site_mask & ~st[1]) >> st[0] & 1)):
                            continue
                        got = b.step(st, press)
                        if got is None:
                            continue
                        nst, win = got
                        if win:
                            return list(path + (press,))
                        if nst == st or nst in seen or nst in cand:
                            continue
                        if site_mask is not None and nst[1] & ~site_mask:
                            continue
                        if self.dead(nst):
                            continue
                        cand[nst] = path + (press,)
                if not cand:
                    break
                keep = sorted(cand, key=self.progress)[:width]
                seen.update(keep)
                layer = {k: cand[k] for k in keep}
        return None


def _trace(parent: dict, st) -> list:
    """The press sequence that reached ``st``, from the A* parent map."""
    out = []
    while parent[st] is not None:
        st, press = parent[st]
        out.append(press)
    out.reverse()
    return out


def _cost(succ, dist, st, press) -> "int | None":
    """Presses to win if ``press`` is taken at ``st``, or None when it is not
    an edge of the shortest-path subgraph."""
    if press not in succ[st]:
        return None
    nxt = succ[st][press]
    if nxt is None:
        return 1
    rest = dist.get(nxt)
    return None if rest is None else rest + 1


def _backward(succ: dict) -> dict:
    """Presses-to-win for every state, by backward BFS from the winning edges."""
    rev: dict = {}
    dist: dict = {}
    frontier = []
    for st, edges in succ.items():
        for nxt in edges.values():
            if nxt is None:
                if st not in dist:
                    dist[st] = 1
                    frontier.append(st)
            else:
                rev.setdefault(nxt, []).append(st)
    q = deque(frontier)
    while q:
        st = q.popleft()
        for prev in rev.get(st, ()):
            if prev not in dist:
                dist[prev] = dist[st] + 1
                q.append(prev)
    return dist



# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class SpringExpert(PSExpert):
    """`PSExpert` with the whole strategy replaced by a search over the model.

    `PSExpert` still supplies the plan memo, the snapshot discipline and the
    on-disk cache; `heuristic` is never called because `_astar` is not the
    search here.  `_search` is the tier ladder, and `describe` reports which
    tier answered each level so a beam plan is never mistaken for a proof.
    """

    directions = list(DIRECTIONS)
    scope_by_level = True
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "spring_plans.json")

    #: Weights the probe climbs when weight 1 will not close a level.  A probe
    #: only has to return SOMETHING: its length becomes the bound the exact
    #: pass is then re-run under, which either improves it or closes.
    ladder = (2, 4, 8, 16, 40)
    #: Node budget for one probe rung, as a fraction of `node_cap`.
    probe_cap = 400_000
    #: Seconds the layout tier may spend sweeping a level's candidate layouts.
    layout_secs = 300.0

    def setup(self) -> None:
        self.ids = _ids(self.game)
        self.stats: dict[int, dict] = {}
        self._level: int | None = None
        self.last: dict | None = None

    def heuristic(self, eng) -> int:                     # pragma: no cover
        raise AssertionError("SpringExpert plans on its native model")

    def plan(self, eng, level=None):
        """Remember which level `describe` is reporting on, then plan as usual.

        The `Board` is built from the grid this is called at, which for a
        RESET-recovery generator is always a level START -- the only state the
        static half of a level (its walls, nests, the water it ships, the ice
        that decides `walk_ice`) can be read from.
        """
        self._level = level
        return super().plan(eng, level)

    def _search(self, eng) -> "Plan | None":
        board = Board(eng, self.ids)
        if board.unmodelled:
            # The winter cutscene: `right [winter no moved | block] -> [...
            # winning]` has already fired during `run_rules_on_level_start`, so
            # the level is over the moment any press is taken and every press
            # is equally right.
            self.stats[self._level] = {"tier": "cutscene", "dstar": 1,
                                       "states": 0, "secs": 0.0}
            return Plan(["action"], [list(DIRECTIONS)])
        ls = LevelSolver(board, node_cap=self.node_cap)
        start = board.start_state
        t0 = time.time()
        found, nodes, _closed = ls.astar(start, weight=1)
        tier = "proved"
        if found is None:
            # Tier 2b: pick the final flower LAYOUT first and route through it.
            found = ls.layout_plan(start, deadline=t0 + self.layout_secs)
            tier = "layout"
            # The layout tier and the weighted probe answer different levels
            # better -- level 12 is 40 presses through a layout and unfindable
            # by the probe, level 14 is 34 by the probe and 36 through its best
            # routable layout -- so run both and keep the shorter, with the
            # first answer as the second one's bound.
            for w in self.ladder:
                got, _n, _c = ls.astar(
                    start, weight=w, node_cap=self.probe_cap,
                    bound=None if found is None else len(found))
                nodes += _n
                if got is not None:
                    found = got
                    tier = "bounded"
                    break
            if found is None:
                # Tier 2c: the last two levels cannot melt their ice ring until
                # the bunny inside it is shut in the nest-and-crate box, so
                # look for that state first and plan the rest from there.
                prefix = ls.pen_search(start) or ls.pen_beam(start)
                for presses, mid in prefix:
                    ls.pen_guard = True
                    forbid = 0
                    for pos, _face in mid[5]:
                        for j in board.nb[pos]:
                            if j >= 0 and (mid[4] >> j & 1):
                                forbid |= 1 << j
                    rest = ls.layout_plan(
                        mid, deadline=time.time() + self.layout_secs,
                        forbid=forbid)
                    if rest is None:
                        rest = ls.beam(mid)
                    ls.pen_guard = False
                    if rest is not None:
                        found = presses + rest
                        tier = "pen"
                        break
            if found is None:
                found = ls.beam(start)
                tier = "beam"
            if found is not None:
                while True:
                    got, _n, closed = ls.astar(start, weight=1,
                                               bound=len(found))
                    nodes += _n
                    if got is None:
                        if closed:
                            tier = "proved"
                        break
                    found = got
                    tier = "bounded"
        if found is None:
            self.stats[self._level] = {"tier": "none", "dstar": None,
                                       "states": 0, "secs": time.time() - t0}
            return None
        states = 0
        optsets = None
        # The sweep is worth trying whatever tier answered.  It is a complete
        # breadth-first search of the ball `g + h <= len(found)`, so if it
        # closes at all it has PROVED the length (and may shorten it) -- a beam
        # plan that turns out to have been shortest gets to say so, with
        # measured tie sets, instead of shipping one arbitrary interleaving.
        swept = ls.sweep(start, len(found))
        if swept is not None:
            found, sets, states = swept
            optsets = [[PRESSES[p] for p in row] for row in sets]
            tier = "proved"
        presses = [PRESSES[p] for p in found]
        if optsets is None:
            # Not proved shortest, so there is nothing to measure a tie
            # against: the expert's own press is the target (never `None` --
            # an unlabelled step trains nothing).
            optsets = [[p] for p in presses]
        self.stats[self._level] = {
            "tier": tier, "dstar": len(presses), "states": states,
            "nodes": nodes, "secs": time.time() - t0,
            "ties": sum(1 for row in optsets if len(row) > 1)}
        return Plan(presses, optsets)

    def describe(self, level: int) -> str:
        got = self.stats.get(level)
        if got is None:
            return "cached"
        return (f"{got['tier']:8s} {got.get('nodes', 0):8d} nodes  "
                f"{got['states']:7d} field states  {got['secs']:6.1f}s")


class SpringSolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    game_module_id = GAME_ID
    expert_cls = SpringExpert
    max_steps = 300
    node_cap = 600_000
    vacuous_start_win = True
    #: The two levels built round an ice ring with a bunny inside it.  See the
    #: module docstring: the pen is found (21 presses on level 18, measured by
    #: `--pens`) but the flower tour that has to follow it is beyond every
    #: search here.  Skipped up front so discovery does not burn the budget on
    #: them at every startup.
    skip_levels = frozenset({18, 20})



# ---------------------------------------------------------------------------
# --fuzz: the model against the interpreter
# ---------------------------------------------------------------------------

def _mechanic(board: Board, st, press: int, nxt) -> str:
    """Which branch of `Board.step` a transition exercised.  The fuzz reports
    coverage per NAME, because a run that never pushed a crate onto a flower
    has not tested the rule that keeps the flower."""
    player, flowers, ice, _water, crates, bun = st
    if nxt is None:
        return "drown"
    npos, nflowers, _ni, _nw, ncrates, nbun = nxt
    if nflowers & ~flowers:
        return "plant"
    lost = flowers & ~nflowers
    if lost:
        return "unplant" if lost == (1 << player) and press == 4 else "eat"
    if press == 4:
        return "wait"
    if ncrates != crates:
        return "push"
    if npos != player:
        return "melt-walk" if ice >> player & 1 else "walk"
    if nbun != bun:
        return "refused"
    return "still"


def fuzz(trials: int = 60, steps: int = 60, verbose: bool = True) -> int:
    """Step the model and the interpreter side by side and require identical
    boards, from three families of start state.

    * **cold** -- the level's own start, no warm-up press, because a turn-one
      bookkeeping bug hides behind any warm-up (see the family notes on
      idols_to_the_burnt_god).
    * **guided** -- the walk is biased toward presses that fire a rule the
      random walk rarely reaches (planting, pushing, stepping on ice).
    * **crowded** -- the board is re-seated with extra crates, flowers and ice
      scattered over it, so the configurations a random walk almost never
      builds (a bunny facing a flower beside a crate, ice hemmed in by water)
      are the common case.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    ids = _ids(game)
    bad = 0
    cover: dict[str, int] = {}
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = Board(eng, ids)
        if board.unmodelled:
            if verbose:
                print(f"  L{level:2d}: winter cutscene, not modelled")
            continue
        mism = 0
        checked = 0
        for trial in range(trials):
            rng = random.Random(f"spring:fuzz:{level}:{trial}")
            mode = ("cold", "guided", "crowded")[trial % 3]
            st = board.start_state
            if mode == "crowded":
                st = _scatter(board, st, rng)
            game.set_level(level)
            eng = game._engine
            _seat(eng, board, ids, st)
            for _ in range(steps):
                if mode == "guided" and rng.random() < 0.5:
                    press = rng.choice((4, 4, 0, 1, 2, 3))
                else:
                    press = rng.randrange(5)
                before = st
                got = board.step(st, press)
                name = _mechanic(board, st, press,
                                 None if got is None else got[0])
                cover[name] = 1 + cover.get(name, 0)
                eng.step(PRESSES[press])
                checked += 1
                if got is None:
                    break                      # the interpreter is about to restart
                st, win = got
                truth = board.read(eng, ids)
                if truth != st:
                    if mism == 0:
                        print(f"  L{level} {mode} trial {trial}: DIVERGED on "
                              f"{PRESSES[press]}")
                        print("   from:\n" + board.show(before))
                        print("   model:\n" + board.show(st))
                        print("   engine:\n" + board.show(truth))
                    mism += 1
                    break
                if win != eng.check_win():
                    print(f"  L{level} {mode} trial {trial}: win {win} vs "
                          f"{eng.check_win()}")
                    mism += 1
                    break
                if win:
                    break
        bad += mism
        if verbose:
            print(f"  L{level:2d}: {checked:6d} presses, "
                  f"{'clean' if not mism else f'{mism} MISMATCHES'}")
    if verbose:
        print("  mechanics covered: " + ", ".join(
            f"{k}={v}" for k, v in sorted(cover.items())))
    return bad


def _scatter(board: Board, st, rng: random.Random):
    """Crowd a level's start with extra crates, flowers and ice."""
    player, flowers, ice, water, crates, bun = st
    taken = (1 << player) | ice | water | crates | flowers
    for pos, _f in bun:
        taken |= 1 << pos
    free = [i for i in range(board.N)
            if (board.plantable >> i & 1) and not (taken >> i & 1)]
    rng.shuffle(free)
    for i in free[:max(1, len(free) // 4)]:
        pick = rng.randrange(3)
        if pick == 0:
            crates |= 1 << i
        elif pick == 1 and not (board.cover[i] & flowers):
            flowers |= 1 << i
        else:
            ice |= 1 << i
    # A flower under a bunny is not a board the game can produce -- the bunny
    # eats what it steps on -- and seating one would fuzz a fiction.
    for pos, _f in bun:
        flowers &= ~(1 << pos)
    return (player, flowers, ice, water, crates, bun)


# ---------------------------------------------------------------------------
# --pens: the measurement behind `skip_levels`
# ---------------------------------------------------------------------------

def pen_report(node_cap: int = 2_000_000) -> None:
    """Which levels have a bunny box, and how far away it is.

    This is the evidence for skipping levels 18 and 20 rather than a guess
    about them: both are an ice ring with a bunny inside and a single square
    (`(4, 5)` and `(4, 7)`) whose four sides are nest, nest, crate, crate, and
    neither can melt a square of that ring until the bunny is shut into it.
    Level 18's box is 21 presses away and this finds it; level 20's is not
    reachable inside the budget, and the flower tour that has to follow either
    of them is beyond every tier here.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    ids = _ids(game)
    for level in range(game.n_levels):
        game.set_level(level)
        board = Board(game._engine, ids)
        if board.unmodelled:
            continue
        ls = LevelSolver(board)
        pens = [(i // board.W, i % board.W) for i in ls.pens]
        bunnies = len(board.start_state[5])
        if not pens or not bunnies:
            print(f"  L{level:2d}: {bunnies} bunnies, no box")
            continue
        t0 = time.time()
        hits = ls.pen_search(board.start_state, node_cap=node_cap)
        got = (f"{len(hits[0][0])} presses away" if hits else
               f"NOT FOUND, {ls.pen_states} states searched")
        print(f"  L{level:2d}: {bunnies} bunnies, box at {pens}, {got} "
              f"({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------------------
# --plans
# ---------------------------------------------------------------------------

def plan_report(levels=None) -> None:
    """Every level's plan, replayed through the real interpreter."""
    solver = SpringSolver()
    game, expert, _ = solver._ensure(0)
    total = ties = won = 0
    for level in range(game.n_levels):
        if levels and level not in levels:
            continue
        if not levels and level in solver.skip_levels:
            continue
        game.set_level(level)
        eng = game._engine
        t0 = time.time()
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED    ({time.time() - t0:.1f}s)")
            continue
        for direction in plan:
            eng.step(direction)
            if eng.check_win():
                break
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tied = sum(1 for row in sets if len(row) > 1)
        total += len(plan)
        ties += tied
        won += bool(eng.check_win())
        print(f"  L{level:2d}: {len(plan):3d} presses  win={eng.check_win()}  "
              f"{sum(len(r) for r in sets) / len(sets):.2f} optimal/step  "
              f"{expert.describe(level)}")
    print(f"  {won} levels won, {total} presses, {ties} steps with a tie")


# ---------------------------------------------------------------------------
# --verify: the three claims the plans rest on
# ---------------------------------------------------------------------------

def verify(bfs_cap: int = 400_000, verbose: bool = True) -> int:
    """Measure, rather than assert, what the plans are standing on.

    1. **The cutscene really is a cutscene.**  Level 5 arrives with `winning`
       already on the board (`right [winter no moved | block]` fires during
       `run_rules_on_level_start`), so the expert emits one press and labels all
       five as equally right.  Every one of them is pressed here and checked.
    2. **`h` is admissible**, at every state of every exact field.  That is the
       whole optimality proof: `d*` is only a bound because no state on a
       shortest path can be priced above its true distance, and the field sweep
       that measures the optimal SETS keeps a state only while `g + h <= d*`.
    3. **`d*` from an independent exhaustive BFS.**  On every level whose whole
       reachable space fits in `bfs_cap` states, a plain breadth-first search
       over the model -- no heuristic, no `dead`, no bound -- settles the
       shortest length, and it has to agree with the length A* proved.  It is
       the check that convicts a wrong prune, which an A* re-run cannot.
    """
    solver = SpringSolver()
    game, expert, _ = solver._ensure(0)
    ids = expert.ids
    bad = 0

    for level in range(game.n_levels):
        if level in solver.skip_levels:
            continue
        game.set_level(level)
        board = Board(game._engine, ids)
        if board.unmodelled:
            for press in DIRECTIONS:
                game.set_level(level)
                game._engine.step(press)
                if not game._engine.check_win():
                    print(f"  L{level:2d}: {press} does not win the cutscene")
                    bad += 1
            if verbose:
                print(f"  L{level:2d}: cutscene, all five presses win")
            continue

        plan = expert.plan(game._engine, level)
        if plan is None:
            if verbose:
                print(f"  L{level:2d}: no plan")
            continue
        ls = LevelSolver(board)
        start = board.start_state

        # 2. admissibility over the ball the labels were measured on.  Re-run
        # the sweep here rather than trusting the tier the plan was cached
        # with: a disk-cached plan carries no field, and the point of this pass
        # is to rebuild the thing the proof rests on.
        adm = "ball larger than the cap"
        tier = "unproved"
        if ls.sweep(start, len(plan)) is not None:
            tier = "proved"
            seen = {start: 0}
            q = deque([start])
            while q:                           # re-walk the same ball
                st = q.popleft()
                for press in range(5):
                    got = board.step(st, press)
                    if got is None:
                        continue
                    nxt, win = got
                    if win or nxt == st or nxt in seen:
                        continue
                    g = seen[st] + 1
                    if g + ls.h(nxt) > len(plan) or ls.dead(nxt):
                        continue
                    seen[nxt] = g
                    q.append(nxt)
            bad_h = sum(1 for st, g in seen.items()
                        if ls.h(st) > len(plan) - g)
            adm = f"{bad_h} inadmissible of {len(seen)}"
            bad += bad_h

        # 3. an exhaustive BFS, where the space fits
        truth = _bfs_dstar(board, bfs_cap)
        note = "space too big" if truth is None else f"BFS d*={truth}"
        if truth is not None and truth != len(plan):
            print(f"  L{level:2d}: plan is {len(plan)}, BFS says {truth}")
            bad += 1
        if verbose:
            print(f"  L{level:2d}: {tier:8s} {len(plan):3d} presses  "
                  f"h: {adm}  {note}")
    return bad


def _bfs_dstar(board: Board, cap: int) -> "int | None":
    """Shortest winning length by exhaustive BFS over the model, or None when
    the reachable space passes ``cap`` states first."""
    seen = {board.start_state}
    layer = [board.start_state]
    depth = 0
    while layer:
        depth += 1
        nxt = []
        for st in layer:
            for press in range(5):
                got = board.step(st, press)
                if got is None:
                    continue
                ns, win = got
                if win:
                    return depth
                if ns == st or ns in seen:
                    continue
                seen.add(ns)
                nxt.append(ns)
        if len(seen) > cap:
            return None
        layer = nxt
    return None


# ---------------------------------------------------------------------------
# --symmetry: the presentation augmentation is exactly a transform
# ---------------------------------------------------------------------------

def symmetry(walk_presses: int = 150, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it draws
    and require the frames to be exactly the transform of the unaugmented ones.

    ps: games are augmented by the adapter, not in the engine: the frame is
    turned and the input is forward-remapped, so what this measures is the
    `screen_action` contract (get the inverse remap backwards and the adapter
    silently executes a different direction on three of every four
    orientations) and that nothing is drawn from outside the grid.

    It is also the argument for leaving Spring OUT of `_FLIP_GAMES` and taking
    only the four rotations.  The bunnies "ALWAYS ROTATE CLOCKWISE UPON HITTING
    A WALL", which is chiral: a mirrored presentation would show a bunny
    turning the way this game never turns, and the whole of level 7 onwards is
    about predicting where a bunny will be.  Rotations are safe for the usual
    reason -- the engine underneath is untouched and clockwise stays clockwise.

    A seeded random WALK is replayed beside each plan; it is what reaches the
    boards a plan never visits (a drowned bunny, a crate wedged on a nest, the
    player sealed in by its own melt-water) and it presses ACTION too.
    """
    solver = SpringSolver()
    game, expert, _ = solver._ensure(0)
    plans = {}
    for level in range(game.n_levels):
        if level in solver.skip_levels:
            continue
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
        for level in range(g.n_levels):
            if plans.get(level) is None:
                continue
            g.set_level(level)
            pres = (g._rotation_k, g._hflip, g._vflip)
            seen.add(pres)
            rng = random.Random(f"spring:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (
                    ("plan", [screen_action(d, *pres) for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *pres)
                              for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"  seed {seed} L{level} {pres}: plan did not win")
                    bad += 1
                slot = (level, tag)
                if slot not in ref:
                    if pres != (0, False, False):
                        continue
                    ref[slot] = frames
                    continue
                if any(not np.array_equal(transform(a, *pres), b)
                       for a, b in zip(ref[slot], frames)):
                    print(f"  seed {seed} L{level} {pres}: {tag} frames are "
                          "not the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


# ---------------------------------------------------------------------------
# --audit: every composition the game can draw is pixel-distinct
# ---------------------------------------------------------------------------

#: Objects the audit is allowed to see through.  Every one of them is declared
#: `transparent` in the .txt and draws nothing: `vert` / `moved` / `nomoved` /
#: `eaten` and the four `m` facing stamps are one-tick bookkeeping, `nodoor` and
#: `start` mark squares whose colour already says everything the player can do
#: with them (both are permanently green, and green is exactly "no flower can go
#: here"), `winning` is the win flag, and `watermark` separates melted water
#: from the water a level ships -- a distinction only a drowning bunny reads,
#: and the one level that ships water has no bunny.
_AUDIT_IGNORE = ("vert", "moved", "nomoved", "eaten", "watermark", "winning",
                 "nodoor", "start", "upm", "downm", "leftm", "rightm")


def audit(trials: int = 30, steps: int = 50, verbose: bool = True) -> int:
    """Collect every cell composition the game actually produces -- by playing
    each level, not by enumerating the object powerset -- and require them to
    render differently at every cell size the levels use.

    Composition, not object.  Almost everything a Spring player has to read is
    a STACK: a blue square with a nest on it still needs a flower beside it, a
    blue square that is ice needs melting instead, and the square you are
    standing on is the one whose colour decides whether ACTION plants anything.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    parsed = game._game
    layers = game._engine._obj_layers
    players = set(game._engine._player_indices)
    names = parsed.obj_idx_to_name

    comps: set[tuple[str, ...]] = set()
    sizes: set[int] = set()
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        sizes.add(max(1, min(64 // len(eng.grid), 64 // len(eng.grid[0]))))
        for trial in range(trials):
            game.set_level(level)
            eng = game._engine
            rng = random.Random(f"spring:audit:{level}:{trial}")
            for _ in range(steps):
                for row in eng.grid:
                    for cell in row:
                        comps.add(tuple(sorted(names[o] for o in cell)))
                if eng.check_win():
                    break
                eng.step(rng.choice(DIRECTIONS))

    def stack(comp):
        body, player = [], []
        for name in comp:
            idx = parsed.obj_name_to_idx[name]
            (player if idx in players else body).append(
                (layers.get(idx, -1), parsed.objects[name]))
        body.sort(key=lambda x: x[0])
        player.sort(key=lambda x: x[0])
        return body + player

    def visible(comp):
        return tuple(n for n in comp if n not in _AUDIT_IGNORE)

    def forgiven(a, b):
        """Two boards that both hold a corpse are both the same board: a
        drowned bunny sets `drowned`, which `[drowned][start] -> [start
        watermark]` and `[start watermark] -> restart` turn into a level
        RESTART two presses later, and the opaque corpse standing on top of the
        square in between hides a floor colour nothing will ever act on."""
        return "drowned" in a and "drowned" in b

    bad = 0
    for px in sorted(sizes):
        blocks = {c: _render_cell_sprite(stack(c), px) for c in comps}
        clashes = [(a, b) for a, b in itertools.combinations(sorted(comps), 2)
                   if bool((blocks[a] == blocks[b]).all())
                   and visible(a) != visible(b) and not forgiven(a, b)]
        for a, b in clashes:
            print(f"  cell_px={px}: {a} and {b} render identically")
        bad += len(clashes)
        if verbose:
            print(f"  cell_px={px}: {len(comps)} compositions, "
                  f"{'all distinct' if not clashes else 'COLLISIONS'}")
    return bad


if __name__ == "__main__":
    if "--verify" in sys.argv:
        violations = verify()
        print(f"verify: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--symmetry" in sys.argv:
        violations = symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--pens" in sys.argv:
        pen_report()
        sys.exit(0)
    if "--plans" in sys.argv:
        plan_report()
        sys.exit(0)
    if "--fuzz" in sys.argv:
        # Optional positional trials / steps, e.g. `--fuzz 12 40` for a quick one.
        _a = [int(x) for x in sys.argv[1:] if x.isdigit()]
        violations = fuzz(*_a)
        print(f"fuzz: {violations} mismatches")
        sys.exit(1 if violations else 0)
    sys.exit(SpringSolver.main())
