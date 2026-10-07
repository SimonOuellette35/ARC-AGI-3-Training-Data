"""ps:stand_iii -- Connorses' "Stand III": the crowd that walks on ONE key,
the two crates from Stand II, and now a floor you can fall through.

    [ >  Player | Crate ]    -> [ >  Player | > Crate ]
    [ >  Player | BigCrate ] -> [    Player | > BigCrate ]
    late [Target1 Player] -> [Target2 Player]
    late [Target2 no Player] -> [Target1]
    late [falls hole] -> [hole] sfx0            (falls = crate or bigcrate)
    late [player hole] -> [lose hole] sfx0
    [lose hole] -> restart
    ==============
    WINCONDITIONS
    ==============
    No Target1

Fifteen levels, all fifteen WON and fourteen of them PROVED SHORTEST -- 14, 11,
20, 7, 7, 21, 18, 11, 25, 34, 21, 5, 13 and 20 presses -- and proved TWICE:
once off an exact distance-to-win field, and once by a heuristic-free
breadth-first press search that shares none of the field's machinery
(`--bfs`).  Part of the
`solvers/common/ps_astar.py` family, on the native-model side of it
(`plan_cache_path` + a `_search` override): the whole mechanic is
re-implemented as integer bitmasks, and the interpreter's job is to REFEREE the
model (`--verify`) and to CERTIFY its answers (`--plans`).

WHAT THIS GAME IS: ps:stand_ii PLUS A FLOOR THAT KILLS

Everything `solvers/generate_stand_ii_training.py` says still holds and is not
repeated here: one arrow key assigns its force to EVERY Player sprite, so the
puzzle is BREAKING FORMATION; the win is one body on each light AT THE SAME
INSTANT (`No Target1`, a light being lit only while somebody stands on it);
`Crate` is an ordinary sokoban crate whose pusher keeps its force; `BigCrate`
is the inverted one whose RHS DROPS the pusher's force, making it the only WAIT
BUTTON one member of the crowd has; neither rule chains; `CrateTop` /
`BigCrateTop` are decals re-derived every turn; and `noaction` is in the
prelude, so ACTION assigns no force (measured, not trusted -- see `verify`).

What Stand III adds is `hole`, and Stand II's model asserted it would never
come -- `StandIIBoard.__init__` RAISES if a level places one, precisely so that
this game could not be planned by the wrong model.  Here holes are everywhere:
eleven of the fifteen levels place one, and most of those use them as the
level's BORDER.

A hole is not a wall.  It is FLOOR that everything may walk onto, and the three
late rules then decide what that meant:

* **A crate or a BigCrate shoved onto a hole is DESTROYED** and the hole stays
  a hole -- it does not fill.  So a hole is the game's crate BIN, and it is the
  only way in either Stand game to make a crate stop being in the way.  The
  pusher's own fate is unchanged by the destruction: it keeps its force behind
  an ordinary Crate and so follows it to the hole's near edge, and it loses its
  force to a BigCrate and so stays where it was.
* **A Player who ends a turn on a hole becomes `lose`**, and the *next* press
  matches `[lose hole] -> restart`.  Death therefore costs TWO presses and puts
  the board back at the level's start state -- `PuzzleScriptAdapter` reloads the
  level when the engine raises `_rule_restart` (see `verify`, which measures
  both halves of that: that the engine raises the flag exactly one press after
  the model marks somebody dead, and that the adapter's reload lands on exactly
  `StandIIIBoard.start`).

DEATH IS NEVER ON A SHORTEST PATH -- EXCEPT AS THE LAST PRESS

The search prunes any press that kills somebody, which is worth stating
carefully because it is a completeness claim, not a convenience.

Every winning press sequence decomposes as: a prefix containing zero or more
restarts, then a suffix containing none.  A restart returns the board to
`start`, so the suffix is itself a winning sequence FROM `start` and is
strictly shorter than the whole.  Induction on the number of restarts: the
shortest winning sequence from `start` contains no restart at all.  And a
sequence contains a death without a restart only when the death is on its very
LAST press -- because the press after a `lose` appears restarts unconditionally.

That last case is real and this game builds on it.  Levels 11 and 13 ship MORE
PEOPLE THAN LIGHTS, and the win is checked after the late rules have run, so a
press that lands the matched people on every light while a spare walks into a
hole is a WIN on that press: `No Target1` holds, and the restart rule never gets
a turn to fire.  (`verify` measures that the engine really does report the win
on the turn the `lose` sprite appears; it is not an argument this file is
allowed to make on its own.)  So `astar`, `field` and `bfs_report` all prune a
successor whose `dead` mask is non-empty UNLESS that successor also WINS.

THE MODEL: FOUR BITMASKS, AND ONE FRONT-FIRST PASS

A state is ``(players, crates, bigcrates, dead)`` -- four ``int`` masks over
cell indices.  Walls and holes never move and a light is a pure function of
whether a live body is on it, so this is exact and canonical.  `dead` is the
`lose` sprites, and it is 0 at every state the SEARCH ever touches (a dead
state is pruned); it exists so that the model can reproduce the interpreter
press for press in `verify` rather than being a planner-only approximation.

`StandIIIBoard.step` is Stand II's, with the late rules appended:

1. if `dead` is non-empty the press RESTARTS -- return `start`;
2. every Player takes the pressed force;
3. a Player facing a Crate lends the force to it (rule 1);
4. a Player facing a BigCrate gives the force away entirely (rule 2);
5. one FRONT-FIRST pass settles the moves;
6. crates and BigCrates now standing on a hole are deleted, and Players now
   standing on a hole move from `players` into `dead`.

Step 5 is exact in one pass for the same reason as in both earlier games: every
force a press assigns points the SAME way, so there is exactly one chain per
queue and the front-most member decides for everybody behind.  Holes take no
part in it -- a hole is floor while the pass runs, and only the late rules of
step 6 make it matter -- which is why `nxt` bakes in walls and NOT holes.

THE HEURISTIC: ps:stand's DIRECTION-DEBT BOUND, WITH THE HOLES BLOCKED

The bound is ps:stand's and ps:stand_ii's.  From the one-key rule, a plan with
`n_down` down-presses gives EVERY person at most `n_down` down-steps, so with
`req_d(p, t)` the fewest `d`-steps on any walk from cell `p` to cell `t`,

    n_d  >=  max over people of req_d(person, its light)      for each d,
    L    >=  SUM over the four directions of those four maxima,

minimised over the matchings of people to lights (`itertools.permutations`, so
a level with more people than lights matches the best subset and leaves the
spares unconstrained -- which is exactly right, a spare owes nothing).  `req_d`
is a 0-1 BFS out of each light in which a step in direction `d` costs 1 and the
other three cost 0, so it is aware of the board it is computed on.

The crates are RELAXED AWAY, as in Stand II: `req_d` treats a crate as floor,
which keeps `h` a function of the PLAYER mask alone and so keeps the memo cheap.

**The holes are NOT relaxed away, and that is this game's one heuristic
change.**  `req_d` treats a hole as a WALL.  It is still admissible, and the
reason is the pruning argument above rather than anything about the relaxation:
the search only ever walks death-free paths, so every person the matching binds
to a light survives to stand on it, and a surviving person's trajectory never
enters a hole cell.  A walk that must not use holes cannot be shorter than one
that may, so blocking them can only RAISE the bound, and it raises it a lot on
the eleven levels whose border is holes -- there, relaxing holes to floor would
let the bound route people straight off the edge of the board.  It is also what
makes `h` correctly INFINITE when a light has no hole-free approach, which the
search reads as "this state can never win" and drops.

Consistency is unchanged by the swap: `req_d` is still a static field on a
fixed graph, and prepending an `e`-step to a walk lowers `req_e` by at most one
and lowers no other direction's debt at all, so `h(s) <= 1 + h(s')` across every
edge.  `--verify` MEASURES both properties over every state of every exact
field rather than resting on this paragraph.

THE FIELD: A* FOR THE BOUND, THEN ONE SWEEP FOR EVERYTHING ELSE

The `solvers/generate_spiders_hollow_training.py` pattern, unchanged from Stand
II.  `astar` proves `d*` (weight 1, the admissible heuristic above).  `field`
then sweeps forward from the level start keeping only states with
``g + h <= d*`` -- with a consistent `h` that is a superset of every state on
every shortest path -- and runs a backward BFS from the winning edges over
exactly those edges.  Out of the exact distance-to-win field come, in one pass,
a plan whose length is checked against `d*` (so: provably shortest) and the
EXACT optimal-action SET at every step.

LEVEL 13 IS THE ONE THAT IS NOT PROVED

Level 13 is the game's own joke -- "Wait a minute, half of these puzzles are
just edited levels from the other games!" -- and it is Stand II's level 9 with
the walls redrawn: 7 people, 7 lights and 11 crates on 10x9, with no holes at
all.  It fails for exactly the reason it failed there.  The heuristic charges
for the WALK and nothing for shoving eleven crates out of the corridors first,
so weight-1 A* is still far from a win after hundreds of thousands of states.
It is planned at the weight in `StandIIIExpert.level_weight`, which finds a
genuine WIN that is not proved shortest, so its steps are labelled with
themselves alone rather than with a measured optimal set.

Declaring the weight PER LEVEL rather than discovering it with a wall-clock
fallback is deliberate and is the same rule Stand II follows: a time limit
would make `plan_cache_path` machine-dependent, and byte-identical plans across
processes is the whole point of that file.

WHAT WAS INVISIBLE (three things, all in the frame and none in the rules)

1. **The winning square, for the third time in this trilogy.**  `Target2` --
   what a light becomes while somebody is standing on it -- shipped
   `Darkgreen lightgreen`, and the ARC palette has exactly ONE green, so both
   of those and the Background's own `GREEN` are index 14.  A person standing
   on a lit light was pixel-identical to a person standing on grass, on a game
   whose win condition is "all of them at once".  Repainted `Yellow Yellow`
   (11), the author's sprite geometry untouched.  Same bug, same fix, as
   `Stand.txt` and `Stand_II.txt`; check it on every Connorses file.

2. **A light under a BigCrate, again.**  Level 11 is 13 cells wide and so
   renders at 4 px per cell, and it opens with a BigCrate standing ON a light
   (the legend's `&`).  The renderer's centred nearest sampling of a 5x5 sprite
   into 4x4 keeps sprite rows/cols 0,1,3,4 and DROPS row/col 2; `Target1`
   paints rows 1-3, of which only 1 and 3 survive, and `BigCrate`'s rows 1 and
   3 were both fully opaque.  One pixel of BigCrate's row 3 is now transparent,
   at a column the 4-px sampling keeps.  Same fix as `Stand_II.txt`.

3. **THE HOLES WERE THE LETTERBOX.**  This one is new, and it is the mechanic
   the game introduces.  `hole` shipped `black #170220` -- and `#170220` is a
   near-black purple, so BOTH of its colours quantize to index 5, leaving the
   sprite a FLAT BLACK BLOCK.  `_render_frame` letterboxes with index 5 too.
   Eleven of the fifteen levels draw their border out of holes, so the frame
   showed a green playfield surrounded by black with NO seam between the black
   that kills you if you step into it and the black that is merely the edge of
   the picture -- two different mechanics (a hole restarts the level, the board
   edge is a no-op) rendered as one colour, on the game that stops between
   levels to say "and don't step in the holes".  The author's intent survives
   the fix: `#170220` is purple, so `hole` is now `black purple` (5 and 15) and
   the author's own lattice geometry paints a purple pit-mouth on black.  Row 0
   of that sprite is solid, so the mark survives the 4-px decimation, which is
   the smallest cell any of these levels uses.  `--audit` is the regression
   test and reported `hole` == a flat 5 block before the change.

All three live in `data/puzzlescript_games/Stand_III.txt` and all three are
sprite-only: no rule, no legend character and no level is touched.

`Stand_III` is in `PuzzleScriptAdapter._FLIP_GAMES`, so it is drawn at all 16
presentations rather than the 4 rotations every ps: game gets.  The argument is
Stand II's -- gravity-free, screen-relative input, a win condition that names no
direction (`No Target1`), no sprite that encodes a facing, and one directional
rule (`late up [crate|no crateTop]`) that paints a read-by-nothing decal
re-derived every turn.  The holes add nothing to that argument: falling is
decided by co-location, not by any direction, so `late [falls hole]` and
`late [player hole]` are as orientation-free as the win condition.
`--symmetry` measures it at all 16 presentations anyway.

RECOVERY is the family's RESET prefix (``recovery_mode = "reset"``).  It has to
be, and this game makes the case louder than either predecessor: a crate shoved
against a wall can never be pulled back, a wall that has eaten one person's step
has changed the crowd's shape forever, and now a crate can be DESTROYED
outright.  A perturbed board cannot be re-planned from, and one RESET lands on
exactly the state the cached field was built from.

VERIFIED.  15/15 levels, 275 presses, every one of the fifteen plans replayed
on the real interpreter to a WIN (`--plans`), and the fourteen weight-1 plans
each their level's proven ``d*``.

* ``--bfs``: a heuristic-free breadth-first press search re-derives 14, 11, 20,
  7, 7, 21, 18, 11, 25, 34, 21, 5 and 13 for levels 0-12 out of 24 / 270 / 1620
  / 31 / 69 / 2532 / 1259 / 29 / 227149 / 1226 / 6718 / 34 / 9807 reachable
  states, and ``--bfs 14`` re-derives 20 out of 12759045.  So every level this
  generator calls shortest is proved shortest TWICE, by two searches sharing no
  machinery.
* ``--verify``: 32799 presses of random and CROWDED play with the model and the
  interpreter stepped side by side, 0 mismatches; 10299 of those presses killed
  somebody, and every one of the 10180 that was pressed on from raised the
  engine's restart flag on exactly the next press; 31 presses WON on the turn a
  spare fell in a hole, which is the case `_live` exists to let through; the
  adapter's restart landed on the level's start state on all 7 levels a
  suicide could be walked on; ACTION pressed 4432 times and inert every time;
  and over the 341042 states of the fourteen exact fields, 0 inadmissible
  states and 0 inconsistent edges.
* ``--audit``: 36 reachable compositions at each of the eleven cell sizes the
  levels use, no two that MEAN different things sharing a picture under any of
  the eight rotations and mirrors, no composition indistinguishable from the
  letterbox, and a fuzz (12 ground/solid pairs seen) confirming nothing outside
  the declared reachable set ever appears.  It FAILED on all three sprite bugs
  before the fixes: `player` == `target2+player` at every size, `bigcrate` ==
  `target1+bigcrate` at the two 4-px sizes, and `hole` a flat letterbox block
  everywhere.
* ``--symmetry``: 15 levels x 16 presentations, every plan a WIN and every
  frame of both the plan and a 200-press random walk exactly the transform of
  the unaugmented one -- restarts included, the walk being left free to fall in
  holes and reload the level mid-replay.
* Generation: 15 levels per seed, ~150 recorded steps each, all 15 WIN, every
  recorded step replays frame-exact through the adapter from the recorded
  SCREEN actions, 276 of ~303 labelled (the rest are the exploration prefixes
  and each level's opening RESET), every index in 0..6, and two processes at
  different hash seeds emit byte-identical episodes -- and, from a cold cache,
  a byte-identical ``data/stand_iii_plans.json``.  The longest plan is 48
  presses against the adapter's 200-press per-level cap.

CLI: ``--verify`` (the model against the interpreter over random AND crowded
play, the hole rules and the two-press death measured explicitly, the adapter's
restart landing on the start state, ACTION's inertness, and the heuristic's
admissibility/consistency against the exact fields), ``--audit`` (every cell
composition the game can draw, pixel-wise distinct-or-synonymous at every cell
size the levels use and under the whole 8-element symmetry group, plus a fuzz
that no composition outside the declared reachable set ever appears),
``--symmetry`` (every presentation is an exact transform of the unaugmented
frames and every plan still wins through the adapter), ``--bfs`` (a
heuristic-free exhaustive re-proof of the levels whose whole space fits),
``--plans`` (every level's plan, replayed through the interpreter), otherwise
the BaseSolver CLI.
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
    PuzzleScriptAdapter, _render_frame,
)
from solvers.base_solver import _ID_TO_GAMEACTION                 # noqa: E402
from solvers.common.ps_astar import (                             # noqa: E402
    PSAStarSolver, PSExpert, Plan, screen_action,
)
from utils.rotation import inverse_remap_action_full              # noqa: E402

GAME_ID = "ps:stand_iii"
GAME_NAME = "Stand_III"

#: The presses the search branches on.  ACTION is deliberately absent: the file
#: declares `noaction` and no rule reads it, so it assigns no force and changes
#: no cell, and a press that cannot change the board can never be on a shortest
#: path.  `verify` measures that rather than trusting the prelude.
DIRS = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Everything on these boards.  Wall1/Wall2 are two paint jobs for one
#: behaviour (neither appears in a rule; both merely occupy the pushables'
#: collision layer) and Target1/Target2 are the unlit and lit faces of one
#: light.  CrateTop/BigCrateTop are decals in a layer of their own and are not
#: read here at all.  `hole` and `lose` are this game's addition and, unlike in
#: ps:stand_ii, both are live.
_OBJECTS = ("wall1", "wall2", "target1", "target2", "player", "crate",
            "bigcrate", "hole", "lose", "cratetop", "bigcratetop")

_INF = float("inf")

#: Sentinel successor for "this press wins".  A `str` can never collide with a
#: board, which is a 4-tuple of `int`.
_WIN = "win"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class StandIIIBoard:
    """One level, and the whole mechanic, as bit twiddling over cell indices.

    A state is ``(players, crates, bigcrates, dead)``: four ``int`` masks in
    which bit ``r * W + c`` is set when that cell holds one.  Nothing else
    varies -- walls and holes never move, a light is a pure function of whether
    a live body is on it, and the crate-top decals are re-derived from the
    crates every turn.

    ``dead`` is the `lose` sprites a hole has just claimed.  It is 0 at every
    state the search touches, because a state with anybody dead is pruned
    (see the module docstring); it is in the tuple so that `step` can be the
    interpreter's equal press for press under `verify`, rather than a
    planner-only approximation that quietly disagrees with the game.
    """

    def __init__(self, eng, ids: dict):
        grid = eng.grid
        self.H, self.W = len(grid), len(grid[0])
        wall_ids = {ids["wall1"], ids["wall2"]}
        light_ids = {ids["target1"], ids["target2"]}
        walls, lights, people, crates, bigs = set(), [], [], [], []
        holes = 0
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                i = r * self.W + c
                if cell & wall_ids:
                    walls.add((r, c))
                if cell & light_ids:
                    lights.append(i)
                if ids["player"] in cell:
                    people.append(i)
                if ids["crate"] in cell:
                    crates.append(i)
                if ids["bigcrate"] in cell:
                    bigs.append(i)
                if ids["hole"] in cell:
                    holes |= 1 << i
                # A board is only ever read at a level's start (or at the start
                # of a `verify` trial), and `[lose hole] -> restart` means a
                # `lose` can survive exactly one press before the level is
                # reloaded -- so one can never be present here.  Check it
                # rather than assume it: a `lose` read as nothing would make
                # `start` a board the game cannot be in.
                if ids["lose"] in cell:
                    raise AssertionError(
                        f"Stand_III board carries a `lose` at {(r, c)}: that "
                        "state restarts on the next press and cannot be a "
                        "planning start")
        self.holes = holes
        # `nxt[d][i]` is the cell a body on `i` moves to when `d` is pressed,
        # or -1 when the board edge or a WALL stops it.  Walls are baked in
        # because they never move.  HOLES ARE NOT: a hole is ordinary floor
        # while the movement pass runs, and only the late rules make standing
        # on one matter.  Crates and people are not baked in either, because
        # they move.
        self.nxt = {}
        for d, (dr, dc) in _DELTA.items():
            arr = [-1] * (self.H * self.W)
            for r in range(self.H):
                for c in range(self.W):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < self.H and 0 <= nc < self.W
                            and (nr, nc) not in walls):
                        arr[r * self.W + c] = nr * self.W + nc
            self.nxt[d] = arr
        self.walls = walls
        self.lights = tuple(lights)
        self.goal = sum(1 << i for i in lights)
        self.start = (sum(1 << i for i in people),
                      sum(1 << i for i in crates),
                      sum(1 << i for i in bigs),
                      0)
        self._h_cache: dict[int, float] = {}
        self._fields = self._dir_fields()

    # -- dynamics ------------------------------------------------------------
    @staticmethod
    def _cells(mask: int) -> list[int]:
        """The occupied cells of ``mask``, in ASCENDING index order."""
        out = []
        while mask:
            low = mask & -mask
            out.append(low.bit_length() - 1)
            mask ^= low
        return out

    def step(self, state: tuple, d: str) -> tuple:
        """The board after pressing ``d``.

        A restart if anybody is already dead; otherwise the two push rules,
        one FRONT-FIRST pass over the solids, and then the three late rules
        that make a hole a hole.

        The pass is exact because every force this press assigns points the
        same way, so there is exactly one chain per queue and the front-most
        member decides for everybody behind.  Ascending row-major order visits
        the cell ahead first for `up` and `left` (its reverse does for `down`
        and `right`), so a cell's blocked-ness is always decided before it is
        read.  Cross-row order is irrelevant: two bodies in different rows can
        never block each other on a horizontal press.

        A solid with NO force is a blocker exactly as a wall is -- an unpushed
        crate, a BigCrate nobody is shoving, and a Player whose force rule 2
        just confiscated are all in that state, which is what makes a BigCrate
        stop the whole queue standing behind its shover.

        Falling happens LAST and to whoever is standing on a hole when the dust
        settles, which is why a crate can be pushed INTO a hole (it is solid
        for the whole of the pass that carries it there, and only then is it
        deleted) and why the person who pushed it does not follow it in.
        """
        players, crates, bigs, dead = state
        if dead:
            # `[lose hole] -> restart`: the press is spent reloading the level.
            # `PuzzleScriptAdapter.perform_action` does the reload when the
            # engine raises `_rule_restart`; `verify` measures that the board
            # it lands on is exactly this one.
            return self.start
        nxt = self.nxt[d]
        occupied = players | crates | bigs
        forced_p, forced_c, forced_b = players, 0, 0
        for cell in self._cells(players):
            ahead = nxt[cell]
            if ahead < 0:
                continue
            if (crates >> ahead) & 1:
                forced_c |= 1 << ahead          # [ > Player | Crate ]
            elif (bigs >> ahead) & 1:
                forced_b |= 1 << ahead          # [ > Player | BigCrate ]
                forced_p &= ~(1 << cell)        # ... and the pusher stays put
        forced = forced_p | forced_c | forced_b

        order = self._cells(occupied)
        if d == "down" or d == "right":
            order.reverse()
        blocked = 0
        for cell in order:
            if not (forced >> cell) & 1:
                blocked |= 1 << cell            # no force: a wall, for today
                continue
            ahead = nxt[cell]
            if ahead < 0 or ((occupied >> ahead) & 1
                             and (blocked >> ahead) & 1):
                blocked |= 1 << cell

        out = []
        for mask in (players, crates, bigs):
            moved = 0
            for cell in self._cells(mask):
                moved |= 1 << (cell if (blocked >> cell) & 1 else nxt[cell])
            out.append(moved)
        people, boxes, boulders = out
        holes = self.holes
        # `late [falls hole] -> [hole]` -- the hole stays a hole, so a crate
        # spent on one is gone for good and the cell is still lethal.
        # `late [player hole] -> [lose hole]` -- one press before the restart.
        return (people & ~holes, boxes & ~holes, boulders & ~holes,
                people & holes)

    def won(self, state: tuple) -> bool:
        """Every light lit AT ONCE -- the engine's `No Target1`.

        Read off the LIVE players only, which is what makes the win-on-the-
        last-press case in the module docstring come out right: a spare who
        falls in a hole on the winning press has already left `players` for
        `dead` by the time this is asked, and lights nothing -- but the people
        who matter are on their lights, so `No Target1` holds and the game ends
        before `[lose hole] -> restart` gets a turn.  `verify` measures that
        the interpreter agrees.
        """
        return state[0] & self.goal == self.goal

    def killed(self, state: tuple) -> bool:
        """True when this state has somebody in a hole, i.e. the next press is
        a restart.  The search drops such a state unless it also WON."""
        return state[3] != 0

    # -- the heuristic -------------------------------------------------------
    def _dir_fields(self) -> dict:
        """``{(light, d): [fewest d-steps on any walk from cell to light]}``.

        A 0-1 BFS backwards out of each light in which a step in direction
        ``d`` costs 1 and the other three cost 0.  Unlike a plain distance this
        is the DEBT a plan owes one direction, which is the quantity the
        one-key rule bounds.

        Crates are floor here and HOLES ARE WALLS -- see the module docstring.
        Relaxing the crates is what keeps `h` a function of the player mask
        alone; blocking the holes is sound because the search only walks
        death-free paths, and it is what stops the bound from routing people
        straight through the lethal border eleven of these levels are made of.
        """
        n = self.H * self.W
        holes = self.holes
        fields = {}
        for d in DIRS:
            # Reverse adjacency: `pred[v]` holds every (u, cost) with u --e--> v.
            pred: list[list[tuple[int, int]]] = [[] for _ in range(n)]
            for e in DIRS:
                cost = 1 if e == d else 0
                for u, v in enumerate(self.nxt[e]):
                    if v >= 0 and not (holes >> u) & 1 and not (holes >> v) & 1:
                        pred[v].append((u, cost))
            for light in self.lights:
                dist = [_INF] * n
                dist[light] = 0
                queue = deque([light])
                while queue:
                    v = queue.popleft()
                    dv = dist[v]
                    for u, cost in pred[v]:
                        nd = dv + cost
                        if nd < dist[u]:
                            dist[u] = nd
                            (queue.appendleft if cost == 0
                             else queue.append)(u)
                fields[(light, d)] = dist
        return fields

    def heuristic(self, state: tuple) -> float:
        """Presses still owed, as a lower bound: the smallest over
        person-to-light matchings of the sum over the four directions of the
        largest debt any matched person owes that direction.

        Levels 11 and 13 ship more people than lights, so the matching picks
        the best SUBSET of the crowd and the spares are charged nothing --
        which is right, a spare owes the win condition nothing at all.

        `_INF` when every matching leaves some light unreachable -- the state
        can never win (death-free) and the search drops it.
        """
        players = state[0]
        got = self._h_cache.get(players)
        if got is not None:
            return got
        people = self._cells(players)
        lights = self.lights
        n = len(lights)
        if len(people) < n:
            self._h_cache[players] = _INF
            return _INF
        # One row per direction: rows[d][person][light] is that person's debt.
        rows = [[[self._fields[(t, d)][p] for t in lights] for p in people]
                for d in DIRS]
        best = _INF
        for pick in itertools.permutations(range(len(people)), n):
            total = 0
            for row in rows:
                worst = 0
                for j in range(n):
                    debt = row[pick[j]][j]
                    if debt > worst:
                        worst = debt
                if worst == _INF:
                    total = _INF
                    break
                total += worst
            if total < best:
                best = total
                if best == 0:
                    break
        self._h_cache[players] = best
        return best

    # -- search --------------------------------------------------------------
    def _live(self, state: tuple, d: str):
        """``(successor, wins)`` for pressing ``d``, or ``None`` when the press
        is not worth taking: it changed nothing, or it killed somebody without
        winning.

        The one place the death pruning of the module docstring is implemented,
        so `astar`, `field` and `bfs_report` cannot drift apart on it.
        """
        nxt = self.step(state, d)
        if nxt == state:
            return None                 # pressed into a wall, as a group
        if self.won(nxt):
            return nxt, True
        if nxt[3]:
            return None                 # somebody fell in, and it did not win
        return nxt, False

    def astar(self, node_cap: int = 1_500_000,
              weight: int = 1) -> "list[str] | None":
        """A press sequence to a win, or None if there is none within the cap.

        At ``weight == 1`` the heuristic is admissible and consistent, so the
        first win popped is optimal -- and unlike a macro search every edge
        here costs exactly one press, so generating a win IS reaching it.  A
        larger weight trades that guarantee for reach; see
        `StandIIIExpert.level_weight` for the one level that needs it.
        """
        start = self.start
        if self.won(start):
            return []
        queue = [(weight * self.heuristic(start), 0, 0, start, ())]
        best_g = {start: 0}
        counter = 0
        while queue:
            _f, g, _c, state, path = heapq.heappop(queue)
            for d in DIRS:
                got = self._live(state, d)
                if got is None:
                    continue
                nxt, wins = got
                if wins:
                    return list(path) + [d]
                ng = g + 1
                if best_g.get(nxt, 1 << 30) <= ng:
                    continue
                hv = self.heuristic(nxt)
                if hv == _INF:
                    continue                    # a light nobody can ever reach
                best_g[nxt] = ng
                counter += 1
                heapq.heappush(queue, (ng + weight * hv, ng, counter, nxt,
                                       path + (d,)))
            if len(best_g) > node_cap:
                return None
        return None

    def field(self, dstar: int) -> tuple:
        """``(succ, dist)`` -- the exact distance-to-win field within ``d*`` of
        the start, over the states a shortest plan can pass through.

        Forward sweep keeping a successor only when ``g + h <= d*``.  With an
        admissible, consistent ``h`` that keeps every state on every shortest
        path: for a state ``u`` on one, ``g(u) + h(u) <= g(u) + dist(u) = d*``,
        and the same holds of each of its prefixes -- so the backward BFS that
        follows measures the true distance-to-win for all of them.
        ``dist[start] == d*`` at the end is the check that it did.
        """
        start = self.start
        gval = {start: 0}
        succ: dict = {}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            g = gval[state]
            edges: dict = {}
            for d in DIRS:
                got = self._live(state, d)
                if got is None:
                    continue
                nxt, wins = got
                if wins:
                    edges[d] = _WIN
                    continue
                if nxt in gval:
                    edges[d] = nxt             # keep the edge for the sweep
                    continue
                if g + 1 + self.heuristic(nxt) > dstar:
                    continue                   # cannot be on a shortest path
                gval[nxt] = g + 1
                edges[d] = nxt
                queue.append(nxt)
            succ[state] = edges
        return succ, _backward(succ)

    def plan(self, node_cap: int = 1_500_000,
             weight: int = 1) -> "Plan | None":
        """The plan and its per-step optimal SETS.

        At ``weight == 1`` the sets are MEASURED off the exact field, and the
        plan's length is checked against `d*` -- it is provably shortest.  At a
        larger weight there is no such guarantee and no field to read sets off,
        so every step is labelled with itself alone: a genuine win, honestly
        labelled, which is what level 13 gets.

        Ties are broken by `DIRS` order, which is what makes a re-derived plan
        byte-identical across processes (and therefore what lets the disk cache
        be trusted between `parallelize_generator` shards).
        """
        found = self.astar(node_cap, weight)
        if found is None:
            return None
        if weight != 1:
            return Plan(found, [[d] for d in found])
        dstar = len(found)
        succ, dist = self.field(dstar)
        if dist.get(self.start) != dstar:
            return None
        presses, optsets = [], []
        state = self.start
        while True:
            best = [d for d in DIRS
                    if _cost(succ, dist, state, d) == dist[state]]
            presses.append(best[0])
            optsets.append(best)
            nxt = succ[state][best[0]]
            if nxt == _WIN:
                return Plan(presses, optsets)
            state = nxt


def _cost(succ: dict, dist: dict, state, d) -> "int | None":
    """Presses to win if ``d`` is pressed at ``state``, or None when that press
    is unavailable (it changed nothing, or it killed somebody) or leads
    nowhere."""
    nxt = succ[state].get(d)
    if nxt is None:
        return None
    if nxt == _WIN:
        return 1
    rest = dist.get(nxt)
    return None if rest is None else rest + 1


def _backward(succ: dict) -> dict:
    """Presses-to-win for every state, by backward BFS from the winning edges.
    States absent from the result cannot win within the swept subgraph."""
    rev: dict = {}
    dist: dict = {}
    frontier = []
    for state, edges in succ.items():
        for nxt in edges.values():
            if nxt == _WIN:
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


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class StandIIIExpert(PSExpert):
    """`PSExpert`'s plan memo, snapshot discipline and disk cache around the
    native search.  `heuristic` is never called on the interpreter -- the model
    owns it -- so it asserts rather than returning a number nothing would use.
    """

    directions = list(DIRS)
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "stand_iii_plans.json")

    #: Levels planned at a weight above 1, i.e. WON but not proved shortest.
    #: Level 13 is the game's self-aware "half of these puzzles are just edited
    #: levels from the other games" board -- Stand II's level 9 with the walls
    #: redrawn: 7 people, 7 lights and 11 crates on 10x9.  It fails for the
    #: same reason it failed there: the heuristic relaxes the crates away and
    #: so charges nothing for shoving eleven of them out of the corridors
    #: before anybody can walk.  Declared per level rather than discovered with
    #: a wall-clock fallback so that two processes derive byte-identical plans
    #: and the disk cache stays portable.
    level_weight: dict[int, int] = {13: 3}

    def setup(self) -> None:
        self.ids = {n: self.g.obj_name_to_idx[n] for n in _OBJECTS}
        self.stats: dict[int, dict] = {}      # level -> what the search cost
        self._last: dict | None = None
        self._level: int | None = None

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "StandIIIExpert plans on its native model; heuristic is unused")

    def board(self, eng) -> StandIIIBoard:
        return StandIIIBoard(eng, self.ids)

    def weight_for(self, level: "int | None") -> int:
        return self.level_weight.get(level, 1)

    def _search(self, eng) -> "Plan | None":
        board = self.board(eng)
        weight = self.weight_for(self._level)
        t0 = time.time()
        found = board.plan(node_cap=self.node_cap, weight=weight)
        self._last = {
            "people": len(StandIIIBoard._cells(board.start[0])),
            "crates": len(StandIIIBoard._cells(board.start[1])),
            "bigcrates": len(StandIIIBoard._cells(board.start[2])),
            "holes": len(StandIIIBoard._cells(board.holes)),
            "lights": len(board.lights),
            "h0": board.heuristic(board.start),
            "weight": weight,
            "states": len(board._h_cache),
            "seconds": time.time() - t0,
        }
        return found

    def plan(self, eng, level=None):
        self._level, self._last = level, None
        got = super().plan(eng, level)
        if level is not None and self._last is not None:
            self.stats[level] = self._last
        return got

    def describe(self, level: int) -> str:
        got = self.stats.get(level)
        if got is None:
            return "cached"
        note = "" if got["weight"] == 1 else f", weight {got['weight']}"
        return (f"{got['states']} player layouts, h0 {got['h0']}, "
                f"{got['seconds']:.1f}s{note}")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class StandIIISolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    game_module_id = GAME_ID
    expert_cls = StandIIIExpert
    #: The exact fields of the bigger levels run to hundreds of thousands of
    #: states; the cap is the runaway guard, not the budget.
    node_cap = 1_500_000
    max_steps = 150


# ---------------------------------------------------------------------------
# --verify: the model against the interpreter, and the heuristic's two claims
# ---------------------------------------------------------------------------

def _engine_state(eng, ids: dict, width: int) -> tuple:
    """``(players, crates, bigcrates, dead)`` read off the interpreter's grid."""
    out = [0, 0, 0, 0]
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            bit = 1 << (r * width + c)
            for k, name in enumerate(("player", "crate", "bigcrate", "lose")):
                if ids[name] in cell:
                    out[k] |= bit
    return (out[0], out[1], out[2], out[3])


def _grid_copy(eng) -> list:
    return [[cell.copy() for cell in row] for row in eng.grid]


def _settle(eng, grid) -> None:
    """Seat ``grid`` on the engine the way loading a level does.

    `PSEngine.load_level` is what `PuzzleScriptAdapter.perform_action` calls to
    serve a restart, and because the file declares `run_rules_on_level_start`
    it does not merely copy the grid in: it runs one whole tick over it.  With
    `noaction` in the prelude that tick assigns no force and moves nothing, so
    all it does is settle the parts of the board that are DERIVED rather than
    placed -- the crate-top decals (`late [CrateTop] -> [ ]` then
    `late up [crate|no crateTop] -> [Crate|Cratetop]`, deleted and repainted
    from scratch), and which face each light is showing.

    `verify` has to go through this rather than assigning `eng.grid`, and the
    reason is worth recording: a raw grid whose decals have not been settled
    is a board no level load can produce, and the first tick of ANY key --
    including the unbound ACTION -- repaints them.  Assigning the grid directly
    therefore makes the inertness check fail on a board the game can never be
    in, blaming the mechanic for the harness."""
    eng.load_level([[cell.copy() for cell in row] for row in grid])


def verify(trials: int = 40, presses: int = 80, verbose: bool = True) -> int:
    """Six measurements, none of which the docstring is allowed to assert.

    1. **The model IS the game.**  Random play on every level, with the model
       and the interpreter stepped side by side and all four masks compared
       after every press, plus the win predicate.  Half the trials run on
       CROWDED boards -- extra Players, Crates, BigCrates and HOLES scattered
       over the free cells before play starts.  That mode is not decoration:
       the interesting parts of `step` are the queue rule (a chain moves iff
       its front can), the two ways a queue stalls (an unpushed crate ahead, or
       a BigCrate eating its shover's force) and the three late rules, and the
       shipped levels place their pieces far too sparsely to hit those often
       under random play.  Scattering extra holes in particular is what makes
       the fuzz push crates into them and walk people into them thousands of
       times rather than a handful.  It is the lesson ps:sokobaiogenesis
       learned the hard way -- a clean fuzz can be blind to exactly the
       configuration a shortest plan manufactures on purpose.

    2. **Death takes exactly two presses, and the engine agrees when.**  Every
       time the model marks somebody dead, the interpreter must be carrying a
       `lose` in the same cells and must NOT yet have raised `_rule_restart`;
       on the following press it must raise it.  That is the whole basis for
       treating `dead` as a one-press state and for `step` returning `start`
       from it.

    3. **A win on the press that kills a spare is really a win.**  Counted
       wherever the fuzz or a plan produces one: the interpreter's
       `check_win()` must be True on a turn whose board carries both a full set
       of lit lights and a fresh `lose`.  This is what licenses the one
       exception to the death pruning in `_live`, so it is measured, and the
       count is reported so a zero is visible rather than silently passing.

    4. **The adapter's restart lands on the start state.**  Driven through
       `PuzzleScriptAdapter.perform_action` (not the raw engine, which only
       raises a flag): walk a person into a hole, press again, and require the
       resulting board to equal `StandIIIBoard.start` exactly.  `step`'s
       ``if dead: return self.start`` is exactly this claim.

    5. **ACTION is inert.**  The file declares `noaction`; this presses it at
       every state of the fuzz and requires the grid not to change, which is
       what justifies leaving it out of `DIRS`.

    6. **The heuristic is ADMISSIBLE and CONSISTENT**: `h(s) <= dist(s)` at
       every state of every exact field, and `h(s) <= 1 + h(s')` across every
       edge of one.  Consistency is the property the field sweep rests on (an
       inconsistent `h` could prune a state that is on a shortest path), so it
       is measured rather than argued -- and here it also stands in for the
       hole-blocking change to `req_d`, which is the one thing this game's
       heuristic does that neither predecessor's did.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    ids = {n: game._game.obj_name_to_idx[n] for n in _OBJECTS}
    rng = random.Random(f"{GAME_NAME}:verify")
    bad = checked = action_checked = action_moved = 0
    deaths = restarts_seen = deaths_pressed = win_with_dead = 0

    for level in range(game.n_levels):
        game.set_level(level)
        plain = StandIIIBoard(eng, ids)
        free = [(r, c) for r in range(plain.H) for c in range(plain.W)
                if (r, c) not in plain.walls]
        for trial in range(trials):
            game.set_level(level)
            crowded = trial >= trials // 2
            if crowded:
                spots = rng.sample(free, min(len(free), rng.randint(3, 14)))
                for r, c in spots:
                    cell = eng.grid[r][c]
                    cell -= {ids["player"], ids["crate"], ids["bigcrate"],
                             ids["hole"], ids["target1"], ids["target2"]}
                    pick = rng.choice(("player", "player", "crate", "crate",
                                       "bigcrate", "hole", "hole"))
                    cell.add(ids[pick])
                eng._position_index_dirty = True
                eng._rule_noop_cache.clear()
                _settle(eng, eng.grid)      # a board a level load could serve
            board = StandIIIBoard(eng, ids)
            opening = _grid_copy(eng)
            state = _engine_state(eng, ids, board.W)
            for i in range(presses):
                if i % 7 == 3 and not state[3]:       # (5) ACTION is inert
                    before = [set(cell) for row in eng.grid for cell in row]
                    eng.step("action")
                    after = [set(cell) for row in eng.grid for cell in row]
                    action_checked += 1
                    if before != after:
                        action_moved += 1
                        bad += 1
                        if bad < 4:
                            print(f"  ACTION changed the grid on L{level}")
                dying = state[3] != 0
                d = rng.choice(DIRS)
                eng.step(d)
                want = board.step(state, d)
                if dying:
                    # (2) the press after a `lose` appears is the restart
                    deaths_pressed += 1
                    if eng._rule_restart:
                        restarts_seen += 1
                    else:
                        bad += 1
                        if bad < 4:
                            print(f"  L{level}: no restart the press after a "
                                  f"death")
                    # The adapter reloads the level here; the raw engine only
                    # raises the flag, so reload the opening board by hand --
                    # through `load_level`, which is the very call the adapter
                    # makes -- and require the model to have said exactly that.
                    _settle(eng, opening)
                    got = _engine_state(eng, ids, board.W)
                    if want != board.start:
                        bad += 1
                        if bad < 4:
                            print(f"  L{level}: model did not restart to start")
                    state = got
                    continue
                got = _engine_state(eng, ids, board.W)
                checked += 1
                win = eng.check_win()
                if got != want or win != board.won(got):
                    bad += 1
                    if bad < 4:
                        where = "crowded" if crowded else "plain"
                        print(f"  MISMATCH L{level} {where} press {d}: "
                              f"model {want} engine {got} "
                              f"win {win}/{board.won(got)}")
                if got[3]:
                    deaths += 1
                    if eng._rule_restart:
                        bad += 1                # the death press restarts too?
                        if bad < 4:
                            print(f"  L{level}: restart on the death press "
                                  f"itself")
                    if win:                     # (3) won as somebody fell in
                        win_with_dead += 1
                if win:
                    break                       # the episode is over
                state = got
    if verbose:
        print(f"  model vs interpreter: {checked} presses, {bad} mismatches")
        print(f"  ACTION pressed {action_checked} times, "
              f"{action_moved or 'none'} of them changed the grid")
        print(f"  {deaths} deaths; of the {deaths_pressed} that were pressed "
              f"on from, {restarts_seen} raised the engine's restart flag "
              f"(the rest ended their trial); {win_with_dead} presses won WITH "
              f"a fresh `lose` on the board")

    # (4) the adapter's restart, end to end.
    restarts = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = StandIIIBoard(eng, ids)
        if not board.holes:
            continue
        walk = _suicide(board)
        if walk is None:
            continue
        game.set_level(level)
        for d in walk:
            game.perform_action(ActionInput(id=_ACTION_ID[d]))
        if _engine_state(eng, ids, board.W)[3] == 0:
            continue                            # nobody died; nothing to check
        game.perform_action(ActionInput(id=_ACTION_ID[walk[-1]]))
        restarts += 1
        if _engine_state(eng, ids, board.W) != board.start:
            bad += 1
            print(f"  L{level}: the adapter's restart did not land on start")
    if verbose:
        print(f"  adapter restart checked on {restarts} levels, all landing "
              f"on the level's start state")

    inadmissible = inconsistent = states = fields = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = StandIIIBoard(eng, ids)
        if StandIIIExpert.level_weight.get(level, 1) != 1:
            continue                  # no exact field to measure against
        plan = board.plan(node_cap=StandIIISolver.node_cap)
        if plan is None:                        # `--plans` reports the level
            continue
        succ, dist = board.field(len(plan))
        states += len(succ)
        fields += 1
        for state, edges in succ.items():
            here = dist.get(state)
            hv = board.heuristic(state)
            if here is not None and hv > here:
                inadmissible += 1
            for nxt in edges.values():
                other = 0 if nxt == _WIN else board.heuristic(nxt)
                if hv > 1 + other:
                    inconsistent += 1
    if verbose:
        print(f"  heuristic over {states} states of {fields} exact fields: "
              f"{inadmissible} inadmissible, {inconsistent} inconsistent edges")
    return bad + inadmissible + inconsistent


#: Engine action ids for the four presses, for the adapter-level restart check.
_ACTION_ID = {"up": 1, "down": 2, "left": 3, "right": 4}


def _suicide(board: StandIIIBoard) -> "list[str] | None":
    """The shortest press sequence that gets SOMEBODY into a hole, or None.

    Used only by `verify`'s adapter-restart check, which needs a board with a
    `lose` on it and does not care who or where.  A plain BFS over the model,
    stopping the moment any state has a non-empty `dead` mask."""
    seen = {board.start}
    queue = deque([(board.start, [])])
    while queue:
        state, path = queue.popleft()
        if len(path) > 24:
            return None
        for d in DIRS:
            nxt = board.step(state, d)
            if nxt[3]:
                return path + [d]
            if nxt in seen or board.won(nxt):
                continue
            seen.add(nxt)
            queue.append((nxt, path + [d]))
    return None


# ---------------------------------------------------------------------------
# --audit: every cell composition the game can draw
# ---------------------------------------------------------------------------

#: The three collision layers a rendered cell is built from.  `hole` shares its
#: layer with the two lights, which is why no level can ever put a light and a
#: hole in one cell -- and why `_reachable` treats the layer as one slot.
_GROUND = ("", "target1", "target2", "hole")
_SOLIDS = ("", "player", "wall1", "wall2", "crate", "bigcrate", "lose")
_DECALS = ("", "cratetop", "bigcratetop")


def _reachable(ground: str, solid: str) -> bool:
    """Can a rendered frame ever show this ground-and-solid pair?

    * The two light rules are each other's inverse and both run before the
      frame is drawn (`late [Target1 Player] -> [Target2 Player]` /
      `late [Target2 no Player] -> [Target1]`), so a light shows its LIT face
      if and only if somebody is standing on it: `target2` never appears alone
      and `target1` never appears under a player.
    * No legend character and no rule ever puts a light under a wall.
    * A hole is EMPTIED by the late rules before the frame is drawn: a crate or
      BigCrate on one is deleted (`late [falls hole] -> [hole]`) and a player
      on one has already become `lose` (`late [player hole] -> [lose hole]`).
      So the only solids a hole can be seen with are nothing and `lose` -- and
      `lose` is seen with nothing else, being creatable only on a hole.
    * Walls and holes are in different layers so the engine would allow the
      pair, but no legend character and no rule makes one.

    `audit` FUZZES all of this as well as using it."""
    if ground == "hole":
        return solid in ("", "lose")
    if solid == "lose":
        return False                    # a `lose` exists only on a hole
    if ground == "target2":
        return solid == "player"
    if ground == "target1":
        return solid not in ("player", "wall1", "wall2")
    return True


def _cases() -> dict:
    """``{name: (objects, meaning)}`` for every composition the game can draw.

    ``meaning`` is what the cell SAYS: which of the four grounds is here
    (floor, an unlit light, a lit light, a hole) and which of the six
    behaviours occupies it.  `Wall1` and `Wall2` collapse to one behaviour --
    neither appears in any rule, both merely occupy the pushables' collision
    layer -- and the crate-top decals say nothing at all, being re-derived from
    the crate below them every turn.  Two compositions may render identically
    only if their meanings match; see `audit`.
    """
    out = {}
    for ground in _GROUND:
        for solid in _SOLIDS:
            if not _reachable(ground, solid):
                continue
            for decal in _DECALS:
                objs = tuple(o for o in (ground, solid, decal) if o)
                name = "+".join(objs) if objs else "floor"
                meaning = (ground,
                           "wall" if solid in ("wall1", "wall2") else solid)
                out[name] = (objs, meaning)
    return out


def audit(verbose: bool = True) -> int:
    """Assert every cell COMPOSITION that means something different renders
    differently, at every cell size the levels use -- and that no rotation or
    mirror of one is another with a different meaning.

    Composition, not object: two of the three bugs this is the regression test
    for were in the STACK.  `player on target2` -- the winning square of the
    game -- was the Background's own green under a body, so it was one picture
    with `player` on grass; and `bigcrate on target1`, which is how level 11
    OPENS, lost the light entirely at the 4 px per cell that level renders at.

    The third is not a stack at all but a single object against the frame:
    `hole` quantized to a flat index-5 block, which is also `_render_frame`'s
    letterbox colour, on a game that draws eleven of its fifteen level borders
    out of holes.  A composition audit cannot see that by itself -- the pad is
    not a composition -- so `_pad_clash` below compares every composition
    against a full field of the letterbox colour, which is the shape the frame
    actually shows around a board.

    MEANING, not identity: `wall1+cratetop` and `wall2+cratetop` are one
    picture and are meant to be.  Wall1 is a solid brown cell and Wall2 is the
    same brown with its bottom row clear, so a crate's bottom decal fills
    exactly the gap that told them apart -- but the two objects are two paint
    jobs for one behaviour, so the frame is not hiding anything.  What would be
    a bug is two compositions that SAY different things sharing a picture.

    The board is FILLED with the composition and whole frames are compared,
    rather than one cell being sliced out by ``cell_px`` arithmetic:
    `_render_frame` upscales a sub-64 render to fill the frame and then
    letterboxes it, so the cell grid in the output is not ``cell_px``-aligned
    and the crop lands in the wrong window.

    The eight-way pass is what the presentation augmentation needs: this game
    takes rotations (every ps: game does) and, per `_FLIP_GAMES`, mirrors too,
    so a transform of one composition that equalled a composition MEANING
    something else would make the corpus say two different things about one
    picture.

    Finally the declared reachable set is FUZZED: random play on every level,
    with every composition that appears on the board checked against
    `_reachable`.  That is what keeps `player on target1` and `hole+crate` out
    of the audit honest -- they are excluded because the late rules make them
    unrenderable, and this measures that rather than believing it."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, parsed = game._engine, game._game
    idx = parsed.obj_name_to_idx
    cases = _cases()
    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots, meaning = {}, {}
        for name, (objs, says) in cases.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, parsed))
            meaning[name] = says
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if meaning[a] != meaning[b]
                   and np.array_equal(shots[a], shots[b])]
        turned = [(a, b, k, hf)
                  for a, b in itertools.permutations(shots, 2)
                  for k in range(4) for hf in (False, True)
                  if meaning[a] != meaning[b]
                  and np.array_equal(_transform(shots[a], k, hf, False),
                                     shots[b])]
        padded = [a for a in shots if _pad_clash(shots[a])]
        synonyms = [(a, b) for a, b in itertools.combinations(shots, 2)
                    if meaning[a] == meaning[b]
                    and np.array_equal(shots[a], shots[b])]
        bad += len(clashes) + len(turned) + len(padded)
        if verbose:
            verdict = ("all distinct" if not clashes
                       else "IDENTICAL " + str(clashes))
            turn_note = ("" if not turned
                         else ", TRANSFORM CLASH " + str(turned))
            pad_note = ("" if not padded
                        else ", INDISTINGUISHABLE FROM THE LETTERBOX "
                             + str(padded))
            syn_note = ("" if not synonyms
                        else f", {len(synonyms)} synonymous pairs "
                             f"({', '.join(a + '==' + b for a, b in synonyms)})")
            print(f"  {h}x{w} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): "
                  f"{len(cases)} compositions, {verdict}"
                  f"{turn_note}{pad_note}{syn_note}")

    # The reachable set, fuzzed rather than believed.
    rng = random.Random(f"{GAME_NAME}:audit")
    unexpected: set = set()
    seen: set = set()
    solids = {idx[o]: o for o in _SOLIDS if o}
    grounds = {idx[o]: o for o in _GROUND if o}
    for level in range(game.n_levels):
        for _ in range(6):
            game.set_level(level)
            for _ in range(60):
                for row in eng.grid:
                    for cell in row:
                        ground = "".join(grounds[o] for o in cell
                                         if o in grounds)
                        solid = "".join(solids[o] for o in cell if o in solids)
                        seen.add((ground, solid))
                        if not _reachable(ground, solid):
                            unexpected.add((ground, solid))
                eng.step(rng.choice(DIRS))
                if eng._rule_restart:
                    game.set_level(level)
    bad += len(unexpected)
    if verbose:
        print(f"  reachability fuzz: {len(seen)} ground/solid pairs seen, "
              f"{len(unexpected) or 'none'} outside the declared set"
              + (f" {sorted(unexpected)}" if unexpected else ""))
    return bad


def _pad_clash(frame, pad: int = 5) -> bool:
    """True when a board filled with this composition renders as nothing but
    `_render_frame`'s letterbox colour -- i.e. the composition is invisible
    against the border the frame draws around every board.

    This is the check the composition matrix cannot make, and it is the one
    that catches this game's third rendering bug: `hole` quantized to a flat
    index-5 block, and 5 is the pad, so the lethal border eleven levels are
    built out of was the same colour as the edge of the picture."""
    return bool((np.asarray(frame) == pad).all())


def _transform(frame, k: int, hflip: bool, vflip: bool):
    if k:
        frame = np.rot90(frame, k=k)
    if hflip:
        frame = np.fliplr(frame)
    if vflip:
        frame = np.flipud(frame)
    return np.ascontiguousarray(frame)


# ---------------------------------------------------------------------------
# --symmetry: the presentation augmentation is a true symmetry of the mechanic
# ---------------------------------------------------------------------------

def symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for putting the game in `_FLIP_GAMES`.  The structural
    argument is Stand II's -- gravity-free, screen-relative input, a win
    condition that names no direction (`No Target1`), no sprite that encodes a
    facing, and one directional rule (`late up [crate|no crateTop]`) whose
    decal is deleted and re-derived every turn, lives in a collision layer of
    its own and is read by nothing, so a flipped presentation simply draws
    crate tops on the flipped side.  Two things are worth measuring rather than
    asserting.  A press moves up to SEVEN bodies at once, which is the shape
    that hid Gobble Rush's rule-order chirality; it is safe because every force
    a press assigns points the SAME way, so two chains can never claim one cell
    and the order `_resolve_forces` traced them in cannot decide anything.  And
    this game's own addition, falling, is decided purely by CO-LOCATION -- both
    `late [falls hole]` and `late [player hole]` name a cell and no direction --
    so it cannot distinguish a presentation either.

    Both the PLANS and a seeded random walk are replayed.  The walk is what
    reaches the boards a plan never visits, and on this game that emphatically
    includes the ones a plan is built to AVOID: crates falling into holes,
    people falling into holes, and the level restarting underneath the replay.
    A restart is exactly the kind of event that would expose an augmentation
    bug if there were one -- the reloaded board has to be the transform of the
    reloaded board -- so it is left in rather than steered around."""
    solver = StandIIISolver()
    game, expert, _ = solver._ensure(0)
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

    ref, seen, bad = {}, set(), 0
    for seed in range(400):
        if len(seen) == 16 and seed > 40:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"{GAME_NAME}:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            plan = plans[level] or []
            for tag, presses in (("plan", [screen_action(d, *k)
                                           for d in plan]),
                                 ("walk", [inverse_remap_action_full(a, *k)
                                           for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"  seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[key] = frames
                    continue
                if any(not np.array_equal(_transform(a, *k), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"  seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


# ---------------------------------------------------------------------------
# --bfs: the independent optimality proof, for the levels whose space fits
# ---------------------------------------------------------------------------

def bfs_report(levels=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
               verbose: bool = True) -> int:
    """Exhaustive breadth-first press search to the depth of the first win, and
    the shortest win it finds compared against the planned one.

    This is the check that the heuristic is not quietly losing a shorter
    answer: the BFS has no notion of a matching, a debt or a bound, it just
    presses all four keys breadth-first and stops expanding once it is as deep
    as the best win so far -- which is exhaustive for the question being asked,
    since nothing beyond that depth can be shorter.  It shares `_live` with the
    field search, and so shares the one thing that is a CHOICE rather than a
    mechanic (dropping a press that kills without winning); everything else
    about it is independent, which is the point.

    The thirteen levels in the default cost 24 / 270 / 1620 / 31 / 69 / 2532 /
    1259 / 29 / 227149 / 1226 / 6718 / 34 / 9807 states between them and all
    thirteen come back PROVED SHORTEST in about four seconds.  Level 14 does
    too -- ``--bfs 14``, measured -- but on its own it is 12759045 states and
    several GB, so it is opt-in rather than part of the default sweep.  (That
    number is Stand II level 10's exactly, which is the game's own joke
    checking out: level 14 IS that board, redrawn.)  Level 13 is the only level
    not proved by anything -- see the module docstring."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    ids = {n: game._game.obj_name_to_idx[n] for n in _OBJECTS}
    bad = 0
    for level in levels:
        game.set_level(level)
        board = StandIIIBoard(eng, ids)
        plan = board.plan(node_cap=StandIIISolver.node_cap)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            bad += 1
            continue
        dist = {board.start: 0}
        queue = deque([board.start])
        best = None
        while queue:
            state = queue.popleft()
            if best is not None and dist[state] >= best:
                continue
            for d in DIRS:
                got = board._live(state, d)
                if got is None:
                    continue
                nxt, wins = got
                if wins:
                    if best is None or dist[state] + 1 < best:
                        best = dist[state] + 1
                    continue
                if nxt not in dist:
                    dist[nxt] = dist[state] + 1
                    queue.append(nxt)
        ok = best == len(plan)
        bad += 0 if ok else 1
        if verbose:
            print(f"  L{level}: {len(dist)} reachable states, shortest win "
                  f"{best} presses, plan {len(plan)} -- "
                  f"{'PROVED SHORTEST' if ok else 'MISMATCH'}")
    return bad


# ---------------------------------------------------------------------------
# --plans
# ---------------------------------------------------------------------------

def plan_report() -> None:
    """Every level's plan, replayed through the real interpreter -- the
    end-to-end test of the model, the search and the tie labelling at once."""
    solver = StandIIISolver()
    game, expert, solvable = solver._ensure(0)
    eng = game._engine
    total = ties = proved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            continue
        for d in plan:
            eng.step(d)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        step_ties = sum(len(s) - 1 for s in sets)
        weight = expert.weight_for(level)
        proved += weight == 1
        total += len(plan)
        ties += step_ties
        print(f"  L{level}: {eng.height:2d}x{eng.width:2d}, "
              f"{len(StandIIIBoard._cells(board.start[0]))} people / "
              f"{len(board.lights)} lights / "
              f"{len(StandIIIBoard._cells(board.start[1]))} crates / "
              f"{len(StandIIIBoard._cells(board.start[2]))} big / "
              f"{len(StandIIIBoard._cells(board.holes)):2d} holes, "
              f"h0 {board.heuristic(board.start):2.0f}, "
              f"{len(plan):3d} presses  win={eng.check_win()}  "
              f"{'shortest' if weight == 1 else 'WON only'}  "
              f"{step_ties:2d} tie-presses  [{expert.describe(level)}]")
    print(f"  solvable levels: {solvable}")
    print(f"  {proved} of {len(solvable)} plans proved shortest")
    print(f"  {total} presses, {ties} of them with a second right answer "
          f"({(total + ties) / max(1, total):.2f} optimal presses per step)")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        violations = verify()
        print(f"verify: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--bfs" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--bfs") + 1:]
                if a.isdigit()]
        violations = bfs_report(tuple(args) if args
                                else (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
        print(f"bfs: {violations} mismatches")
        sys.exit(1 if violations else 0)
    if "--plans" in sys.argv:
        plan_report()
        sys.exit(0)
    sys.exit(StandIIISolver.main())
