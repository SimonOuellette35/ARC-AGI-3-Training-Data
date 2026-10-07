"""Generate Phase-1 training data for the PuzzleScript game ps:spacekoban
("Spacekoban", Connorses [Loneship Games]).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exhaustive enumeration that model is
searched with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_spacekoban",
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
Every expert step carries the full set of equally-optimal presses.

The mechanic: you have no legs, only Newton's third law
-------------------------------------------------------
Spacekoban is a sokoban played in FREE FALL. The arrow keys are not steps. A
press is an attempt to SHOVE OFF whatever is behind you, and if there is nothing
behind you the press does nothing at all:

    [ < player no ladder | no solid no crate ] -> [ player | ]

Read it in the rule's own direction: ``< player`` is the player moving *against*
the scan, so the second cell is the one BEHIND it. Nothing solid, no crate ->
the movement is cancelled and the turn is over. That single rule is the whole
game's economy: every board is a question about which walls you can still reach.

When there IS something behind you, one of three rules fires instead, in this
order (which is the order they are written in, and the order matters):

    up  [ < player | solid ] -> [ player D | solid ] again     (1) push off a wall
    [ up player ladder ]     -> [ player U ladder ]            (2) a ladder is a handhold
    up  [ < player | crate ] -> [ player D | crate U ] again   (3) push off a CRATE

Each gives the player a ``U``/``D``/``L``/``R`` marker -- an invisible object on
its own collision layer that *is* the momentum -- and (3) gives the crate behind
you the OPPOSITE marker. That is the recoil: shoving off a crate throws the
crate the other way, and it is how most of the crates in this game are moved at
all. Because (2) is written before (3), a player standing on a ladder with a
crate behind it takes the ladder branch and the crate does NOT recoil;
`--selfcheck` measures that against the interpreter rather than assuming it.

Momentum then runs to completion inside the one keypress, through an ``again``
loop of four more rules:

    up [ obj U | obj ] -> [ obj U | obj U ]      momentum spreads through a TRAIN
    [ obj U ] -> [ up obj up U ] again           marked objects (and markers) move
    [ > dir | solid ] -> [ | solid ]             a marker about to hit a wall dies
    late [ player dir ladder ] -> [ player ladder ]   a ladder catches the PLAYER
    late [ obj dir beam ] -> [ obj beam ]             a beam catches ANYTHING

So you fly until something stops you, pushing whatever you catch up with, and
the two brakes are a ladder (players only -- a crate sails straight over one)
and a beam (players and crates alike). ``obj = player or crate`` and
``solid = wall or cover``, so nothing else on the board participates.

WIN: ``all crate on target`` and ``all exit on player`` and ``no dir``. No level
in the file contains an ``e``, so the middle clause is vacuously true and the
``exit``/``cover`` rules never fire. The third clause is not decoration -- see
below.

The five measurements this rests on
-----------------------------------
`--selfcheck`'s first pass is these, as executable claims against the
interpreter:

  * nothing behind -> the press does nothing, and the cell behind is NOT erased
    by the cancel rule's empty right-hand side (ladders, beams and targets
    behind you all survive it);
  * a wall behind -> you launch and travel until something stops you;
  * a crate behind -> you launch AND the crate recoils the other way;
  * standing on a ladder -> you launch in any direction, with no recoil even
    when a crate is behind you, and land after ONE cell only if that cell is
    itself a ladder;
  * off the board behind you (level 8's open right edge) -> no rule matches, the
    raw input force survives and the player takes a single plain step.

THE MARKER LAYER, and why ``no dir`` is a real win condition
------------------------------------------------------------
The markers are objects, not attributes. They move on their own collision layer
with their own force, so they can come UNSTUCK from the object carrying them:
when a train packs against a wall, the front marker is deleted by
``[ > dir | solid ]`` and every marker behind it SLIDES ONE FORWARD into the
freed slot, so a train of k objects takes k extra ``again`` iterations to go
quiet. That is cosmetic here (nothing moves during those iterations) but it is
the reason the model simulates the marker layer cell by cell instead of tracking
"the moving set": get it wrong and the iteration count, and with it the 50-tick
``again`` cap, is wrong.

It stops being cosmetic at a board EDGE. ``[ > dir | solid ]`` needs a cell to
look at, so a marker pressed against the outside of the board can never be
deleted -- it sits there forever, ``no dir`` is false forever, and the level is
lost with nothing on screen to say so. Level 8 is the only board with an open
side (its rows 3-5 run off the right edge), so it is the only board where that
can happen. `_Board.step` returns None for such a state and the search drops it.

**The pruning is vacuous, and that is measured, not assumed.** The enumeration
below walks every state of every level -- 207003 of them -- and counts the
successors that come back stranded: it is **zero**, on all twenty boards, level 8
included. So no shortest path was ever cut, and the plans below are shortest
against the full game and not merely against a safe subset of it.

`--fuzz` is the other half of that claim: 55207 presses over 1920 RANDOM boards,
model and interpreter in lockstep, 0 disagreements -- and of the 236 walks that
did strand momentum, **not one was on a board with a closed border**. So the
branch exists, the model gets it right, and only an open side can reach it.

Expert solver
-------------
A NATIVE model (`_Board`): a state is ``(player cell, frozenset of crate
cells)`` over static wall / ladder / beam / target masks, and the interpreter is
never stepped during planning. The reason is measured: this game's interpreter
runs at 260-420 presses/s (a flight is one whole ``again`` iteration per cell
crossed, so a press costs as much as the distance it covers), and the model at
68k-128k, i.e. **260-370x** -- which is what turns level 16's 159370 states from
half an hour into six seconds, and all twenty levels into twelve.

Every level is solved by EXHAUSTIVE ENUMERATION, not by a heuristic search:

  1. forward BFS from the level start over `_Board.step`, to closure -- no node
     cap is doing any work here, the biggest board is 159370 states;
  2. backward BFS from every winning state over the reversed edges, giving
     exact presses-to-win for every state that can win at all;
  3. the plan is a greedy descent of that field, and the optimal SET at a state
     is every press whose successor is one step closer -- exact, not inferred.

Enumeration buys three things a heuristic search cannot give: plans that are
provably shortest with no weight/cap/heuristic that could quietly bite, exact
tie sets (the always-emit-optimal-targets rule), and the dead-successor census
above. States are keyed by a packed, decodable 2-bytes-per-cell key so the
biggest level costs ~10 MB rather than a snapshot per state.

The levels
----------
All twenty parse and all twenty are winnable, at

    1, 1, 4, 8, 5, 20, 12, 11, 6, 14, 9, 9, 1, 1, 21, 2, 21, 6, 29, 35

presses (216 total), every one provably shortest. Level 5 is the first that
needs a ladder as a launch pad; level 6 is the nine-crate lattice; level 16 is
the deep one (a 4x4 block of crates inside a ladder cage) and level 19 the
longest (35 presses). Levels 12, 13 and 15 are the author's beam tutorials and
fall in one or two presses.

ACTION (X) is unbound: no rule in the file reads it, and `--selfcheck` presses it
on every level to show the grid never moves. It is dropped from the search, which
halves the interpreter work the recorder does per node.

The tutorial levels needed the RESET prefix widened
---------------------------------------------------
Five levels finish in one or two presses, and the episode-wide exploration
budget is at its largest on the first level, so the recovery prefix's random
flail simply WINS them -- level 0 every single time. The shared
`BaseSolver._reset_prefix` skips its closing RESET on a terminal win, so those
levels taped a WIN made only of ``phase="explore"`` steps, which `train_policy`
masks out of the policy loss: 52 of 600 levels over 30 seeds trained on nothing.
`SpacekobanSolver._reset_prefix` resets anyway, and it is 0 of 600 after, with
expert steps at exactly 30 x 216.

Four sprites were redrawn
-------------------------
All four are in `data/puzzlescript_games/Spacekoban.txt`, and none touches a
rule, a colour declaration, a level layout or a win condition -- only the 5x5
art. `--audit` renders every cell COMPOSITION the game can show, on a board
filled with it, at every cell size the levels actually use, and requires them
pairwise distinct. Before the redraw it reported **40 collisions**; after, zero.

`_render_cell_sprite` takes CENTRED nearest samples, so a 5x5 sprite is read at
rows/cols **{1,3} at cell_px 2** and **{0,1,3,4} at 4**, and the levels here run
at cell_px 2, 4, 5, 6, 7 and 8 ([[ps-palette-collisions]]). Two consequences,
both of which the shipped art walked into:

  * **A player standing on a beam was pixel-identical to a player standing on
    nothing, at EVERY cell size.** The beam was a hairline diamond drawn on
    rows/cols 1-3 and the player is opaque across exactly that, and the adapter
    draws the player above everything regardless of layer. The beam is one of
    the two brakes in the game -- "am I parked on a beam" is the single most
    load-bearing bit on the board after crate positions -- and it was not in the
    frame. The same diamond is invisible against bare background at cell_px 4
    (rows 1 and 3 of it are painted only on column 2, which that grid does not
    sample), which is levels 12 and 15, i.e. the two boards that are ABOUT
    beams.
  * **On level 8 -- 31 cells wide, so cell_px 2, so four pixels per cell -- the
    target was invisible and a crate on a target was pixel-identical to a crate
    anywhere else.** Target painted nothing on the {1,3} grid and Crate painted
    all four of it. That is the win condition itself, unreadable.

The fix allocates the four surviving pixels by role, which is what makes every
stack legible at the smallest size and therefore at all of them:

    (1,1) -> TARGET   -- player, crate and beam are all transparent here
    (1,3), (3,1) -> BODY  -- the player in white, the crate in pink
    (3,3) -> BEAM     -- target, player and crate are all transparent here

so Target keeps a corner frame but bites out (3,3), Crate and Player bite out
(1,1) and (3,3) (which also lets a target show through a crate parked on it at
every size), and Beam gains a foot at (3,3) plus four mid-edge pixels at
(1,0)/(1,4)/(3,0)/(3,4) that no other sprite paints, so it survives cell_px 4
too. Crate's ring moved from red to pink over a red core: a red ring on a red
target composites to one uniform block at cell_px 2, so "crate on target" needed
a colour of its own as well as a hole ([[sokubunny-solver]] is the same trap).

The game now takes the FLIP augmentation too
--------------------------------------------
``Spacekoban`` was added to `PuzzleScriptAdapter._FLIP_GAMES`, which takes it
from 4 presentations (rotation only) to 16. It qualifies on the usual argument
-- no gravity, screen-relative directional input, a win condition that names no
direction -- with one wrinkle that had to be looked at rather than waved past:
the game DOES own directional objects, the four momentum markers, and a mirror
that showed their art would be showing art the game does not have. It cannot:
their sprites are entirely ``transparent``, and no settled frame contains one at
all (that is the ``no dir`` win clause, plus the zero-stranded-marker census
above). What is left is rule-order chirality, which is measured, not argued:
`--symmetry` drives each of the 20 plans and a seeded 150-press random walk at
all 16 presentations and requires exact frame equality with the transform. 16
presentations drawn, 0 violations.

Reports
-------
    python solvers/generate_spacekoban_training.py --plans      # per-level table
    python solvers/generate_spacekoban_training.py --selfcheck  # model vs engine
    python solvers/generate_spacekoban_training.py --selfcheck --full
    python solvers/generate_spacekoban_training.py --fuzz       # random boards
    python solvers/generate_spacekoban_training.py --audit      # rendering
    python solvers/generate_spacekoban_training.py --symmetry   # augmentation

and generation itself:

    python solvers/generate_spacekoban_training.py --episodes N \
        --out data/training_multi_level/puzzlescript_spacekoban
"""

from __future__ import annotations

import itertools
import random
import struct
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                    # noqa: E402
from adapters.puzzlescript_adapter import (                     # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from solvers.base_solver import _ID_TO_GAMEACTION               # noqa: E402
from solvers.common.ps_astar import (                           # noqa: E402
    Plan, PSAStarSolver, PSExpert, screen_action)
from utils.explore import Action, RESET_ACTION                  # noqa: E402
from utils.rotation import inverse_remap_action_full            # noqa: E402

GAME_NAME = "Spacekoban"

#: Engine directions the search branches on. ACTION is unbound -- no rule in the
#: file mentions it -- so branching on it would double the work for nothing;
#: `--selfcheck` presses it on every level to show the grid never moves.
DIRECTIONS = ["up", "down", "left", "right"]

#: Index into DIRECTIONS -> the index of its opposite.
_OPP = (1, 0, 3, 2)

#: The interpreter's own `again` budget (`PSEngine.step`). The model runs the
#: same loop so that a flight long enough to exhaust it settles the same way in
#: both; the longest flight in the file is level 8's 29-cell corridor.
_MAX_AGAIN = 50


# ---------------------------------------------------------------------------
# Native model of the mechanic
# ---------------------------------------------------------------------------

class _Board:
    """One level's static terrain plus the exact settled dynamics of a press.

    Cells are flat ints ``r * w + c``; ``nbr[d][p]`` is the neighbour of ``p`` in
    direction ``d`` or -1 off the board, which is the only bounds check anything
    below needs.

    A state is ``(player, frozenset(crates))``. `step` returns the settled state,
    or None when momentum is left on the board -- see the module docstring for
    why that is unwinnable and why no level ever produces it.
    """

    __slots__ = ("h", "w", "n", "wall", "ladder", "beam", "target", "nbr")

    def __init__(self, h, w, wall, ladder, beam, target):
        self.h, self.w, self.n = h, w, h * w
        self.wall = wall            # list[bool], indexed by flat cell
        self.ladder = ladder
        self.beam = beam
        self.target = target
        self.nbr = []
        for d, (dr, dc) in enumerate(((-1, 0), (1, 0), (0, -1), (0, 1))):
            row = []
            for p in range(self.n):
                r, c = divmod(p, w)
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self.nbr.append(row)

    # -- dynamics -----------------------------------------------------------
    def step(self, player, crates, d):
        """One press. Returns ``(player, frozenset(crates))`` settled, or None
        when a marker survives (a permanently lost board)."""
        back = self.nbr[_OPP[d]][player]
        crates = frozenset(crates)
        marks: dict[int, int] = {}
        if back < 0:
            # Off the board behind us: no launch rule can match (they all need a
            # cell to look at) and the cancel rule cannot either, so the raw
            # input force survives and the player takes one plain step.
            if not self.ladder[player]:
                t = self.nbr[d][player]
                if t >= 0 and not self.wall[t] and t not in crates:
                    return t, crates
                return player, crates
            marks[player] = d                       # rule 2: ladder handhold
        elif self.wall[back]:
            marks[player] = d                       # rule 1: push off a wall
        elif self.ladder[player]:
            marks[player] = d                       # rule 2, BEFORE the recoil
        elif back in crates:
            marks[player] = d                       # rule 3: push off a crate
            marks[back] = _OPP[d]                   #         ... and it recoils
        else:
            return player, crates                   # nothing behind: cancelled

        for _ in range(_MAX_AGAIN):
            player, crates, marks, changed = self._settle(player, crates, marks)
            if not changed:
                break
        if marks:
            return None
        return player, crates

    def _settle(self, player, crates, marks):
        """One `again` iteration: the four momentum rules, force resolution on
        both layers, then the two `late` brakes."""
        objs = set(crates)
        objs.add(player)
        marks = dict(marks)

        # [ obj U | obj ] -> [ obj U | obj U ]: momentum spreads forward through
        # a train, transitively -- the rule re-matches within its own pass.
        stack = [x for x in marks if x in objs]
        while stack:
            x = stack.pop()
            m = marks.get(x)
            if m is None or x not in objs:
                continue
            nxt = self.nbr[m][x]
            if nxt >= 0 and nxt in objs and marks.get(nxt) != m:
                marks[nxt] = m
                stack.append(nxt)

        # [ obj U ] -> [ up obj up U ]: a marker sharing a cell with an object
        # gives that object (and itself) a force. A marker alone in a cell gets
        # none, which is exactly why an orphaned one is permanent.
        # [ > dir | solid ] -> [ | solid ]: ... unless it is aimed at a wall.
        for x in [x for x in marks if x in objs]:
            nxt = self.nbr[marks[x]][x]
            if nxt >= 0 and self.wall[nxt]:
                del marks[x]
        forces = {x: m for x, m in marks.items() if x in objs}

        moved_o = self._slide(forces, objs, wall_blocks=True)
        moved_m = self._slide(forces, set(marks), wall_blocks=False)

        newp = moved_o.get(player, player)
        crates2 = frozenset(moved_o.get(c, c) for c in crates)
        marks2 = {}
        for x, m in marks.items():
            marks2[moved_m.get(x, x)] = m

        # late [ player dir ladder ] / late [ obj dir beam ]: the two brakes.
        # A train behind can shove the braked object straight off again on the
        # next iteration -- the spread rule runs before these do.
        for x in list(marks2):
            if (x == newp and self.ladder[x]) or \
                    ((x == newp or x in crates2) and self.beam[x]):
                del marks2[x]

        changed = (newp != player or crates2 != crates or marks2 != marks)
        return newp, crates2, marks2, changed

    def _slide(self, forces, occupied, wall_blocks):
        """Simultaneous movement on ONE collision layer -> ``{from: to}``.

        Objects are blocked by walls and by objects; markers are blocked only by
        markers, which is what lets a marker slide forward off an object that is
        itself jammed. A chain moves when its front can; two movers aimed at the
        same cell both stay put.
        """
        can = dict.fromkeys(forces, True)
        dirty = True
        while dirty:
            dirty = False
            for x, m in forces.items():
                if not can[x]:
                    continue
                t = self.nbr[m][x]
                if t < 0 or (wall_blocks and self.wall[t]) or \
                        (t in occupied and not can.get(t, False)):
                    can[x] = False
                    dirty = True
        dest: dict[int, list[int]] = {}
        for x, m in forces.items():
            if can[x]:
                dest.setdefault(self.nbr[m][x], []).append(x)
        return {xs[0]: t for t, xs in dest.items() if len(xs) == 1}

    def won(self, player, crates):
        """``all crate on target``. ``no dir`` holds by construction (a state
        with a marker is never kept) and ``all exit on player`` is vacuous --
        no level in the file contains an exit."""
        target = self.target
        return all(target[c] for c in crates)

    # -- exhaustive enumeration + exact distance field -----------------------
    def solve(self, player, crates):
        """Enumerate the whole reachable component, then read a provably
        shortest plan with exact tie sets off the backward distance field.

        Returns ``(Plan | None, states, dead)``: ``states`` is the size of the
        reachable component and ``dead`` the number of successors dropped for
        leaving momentum on the board (see the module docstring -- it is 0 on
        every level of this game, which is what makes the plan shortest against
        the real game rather than against a pruned one).
        """
        start = _pack(player, crates)
        succ: dict[bytes, dict[int, bytes]] = {}
        seen = {start}
        queue = deque([start])
        dead = 0
        while queue:
            k = queue.popleft()
            p, cr = _unpack(k)
            if self.won(p, cr):
                succ[k] = {}                     # absorbing: the episode ends
                continue
            edges: dict[int, bytes] = {}
            for d in range(4):
                res = self.step(p, cr, d)
                if res is None:
                    dead += 1
                    continue
                nk = _pack(*res)
                if nk == k:
                    continue                     # a press that did nothing
                edges[d] = nk
                if nk not in seen:
                    seen.add(nk)
                    queue.append(nk)
            succ[k] = edges

        # Backward BFS from every winning state over the reversed edges.
        rev: dict[bytes, list[bytes]] = {}
        dist: dict[bytes, int] = {}
        frontier = []
        for k, edges in succ.items():
            if not edges and self.won(*_unpack(k)):
                dist[k] = 0
                frontier.append(k)
            for nk in edges.values():
                rev.setdefault(nk, []).append(k)
        queue = deque(frontier)
        while queue:
            k = queue.popleft()
            for p in rev.get(k, ()):
                if p not in dist:
                    dist[p] = dist[k] + 1
                    queue.append(p)

        if start not in dist:
            return None, len(seen), dead
        presses, optsets = [], []
        k = start
        while dist[k]:
            best = [d for d in range(4)
                    if d in succ[k] and dist.get(succ[k][d], 1 << 30)
                    == dist[k] - 1]
            presses.append(DIRECTIONS[best[0]])
            optsets.append([DIRECTIONS[d] for d in best])
            k = succ[k][best[0]]
        return Plan(presses, optsets), len(seen), dead


def _pack(player, crates):
    """State -> a compact, decodable key: 2 bytes per cell, player first.

    The keys are what the enumeration stores, and the reason level 16's 159370
    states cost ~10 MB instead of a grid snapshot each."""
    return struct.pack(f">{len(crates) + 1}H", player, *sorted(crates))


def _unpack(key):
    vals = struct.unpack(f">{len(key) // 2}H", key)
    return vals[0], frozenset(vals[1:])


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class SpacekobanExpert(PSExpert):
    """`PSExpert`'s plan memo and disk cache around `_Board`'s exact field.

    `_search` is replaced the way the other native-model ps: generators replace
    it: the base class keeps the memo, the snapshot/restore discipline and the
    level scoping, and only the strategy underneath changes. `heuristic` is
    never called and asserts rather than returning a number nothing would use.
    """

    directions = DIRECTIONS

    #: `_key` is inherited -- every non-background cell, which is exact and
    #: canonical ACROSS levels here because the walls are in the key too.
    scope_by_level = False

    #: The enumeration is ~12s for all twenty levels, which every shard of
    #: `parallelize_generator` would otherwise re-pay at startup. Only a level's
    #: START state is ever cached, and only against the layout it was solved
    #: from, so an edited level is a miss rather than a wrong plan.
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "spacekoban_plans.json")

    def setup(self) -> None:
        g = self.g
        self.ids = {n: g.obj_name_to_idx[n] for n in
                    ("player", "wall", "crate", "target", "ladder", "beam")}
        self._boards: dict[tuple, _Board] = {}

    def heuristic(self, eng) -> int:
        raise AssertionError("SpacekobanExpert enumerates the reachable space; "
                             "heuristic is unused")

    def read(self, eng):
        """Engine grid -> ``(board, player, frozenset(crates))``.

        The terrain masks are memoised by their own signature: the boards are
        static within a level, and two levels are different boards even at the
        same coordinates."""
        w = eng.width
        pid, wid = self.ids["player"], self.ids["wall"]
        cid, tid = self.ids["crate"], self.ids["target"]
        lid, bid = self.ids["ladder"], self.ids["beam"]
        n = eng.height * w
        wall = [False] * n
        ladder = [False] * n
        beam = [False] * n
        target = [False] * n
        player = -1
        crates = []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                p = r * w + c
                if wid in cell:
                    wall[p] = True
                if lid in cell:
                    ladder[p] = True
                if bid in cell:
                    beam[p] = True
                if tid in cell:
                    target[p] = True
                if pid in cell:
                    player = p
                if cid in cell:
                    crates.append(p)
        sig = (eng.height, w, tuple(wall), tuple(ladder), tuple(beam),
               tuple(target))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(eng.height, w, wall, ladder,
                                               beam, target)
        return board, player, frozenset(crates)

    def _search(self, eng) -> "Plan | None":
        board, player, crates = self.read(eng)
        plan, _states, _dead = board.solve(player, crates)
        return plan


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class SpacekobanSolver(PSAStarSolver):
    game_id = "puzzlescript_spacekoban"
    game_name = GAME_NAME
    expert_cls = SpacekobanExpert

    #: `games/ps:spacekoban/ps:spacekoban.py` is a plain passthrough (it
    #: constructs the adapter and nothing else), so there is nothing to gain by
    #: routing through it -- but it IS what a live agent is handed, so if that
    #: wrapper ever grows a patch this must be set to ``"ps:spacekoban"``.
    game_module_id = ""

    #: Unused: the expert enumerates an exact field rather than searching under
    #: a heuristic, so there is no weight to trade and no node budget that could
    #: quietly bite. Left at the base values so nothing reads a lie off them.
    weight = 1
    node_cap = 400_000

    #: The longest plan is 35 presses (level 19); the rest is room for a re-plan
    #: after the exploration prefix. Stays well under the adapter's own 200-step
    #: per-level budget, which `set_level` resets before the plan starts anyway.
    max_steps = 120

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it fills the disk cache in one pass and makes the
        startup cost visible as startup."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)

    def _reset_prefix(self, *, reset_to_level, record_obs, record_act, **kw):
        """`BaseSolver._reset_prefix`, plus a RESET when the prefix WON.

        The shared prefix deliberately skips its closing RESET on a terminal win
        (``if perturbed and last_terminal != "win"``), which is unreachable for a
        game whose levels take a dozen presses. This game has five levels that
        finish in one or two -- 0, 1, 12, 13 and 15, the author's tutorials --
        and the opening exploration budget is at its largest on the episode's
        first level, so a random flail finishes level 0 EVERY time. `record_level`
        then sees an already-won level and breaks, and the level tapes a WIN made
        entirely of ``phase="explore"`` steps, which `train_policy` masks out of
        the policy loss: those recordings train the policy on nothing. Measured
        over 30 seeds before this override: 52 of 600 levels had no expert step
        at all, and level 0 was 30 of 30.

        Resetting anyway costs one frame and puts the level back at its start,
        where the expert replays the real solution, so every level carries a full
        labelled demonstration and every episode has the same flail / reset /
        solve arc. It lives here rather than in `BaseSolver` because changing the
        shared branch would re-cut every corpus already recorded with it.

        The single-frame assumption is safe: ``record_spans`` is False, so
        `record_level`'s ``reset_to_level`` hands back one ``(H, W)`` frame.
        """
        prev = super()._reset_prefix(reset_to_level=reset_to_level,
                                     record_obs=record_obs,
                                     record_act=record_act, **kw)
        if self._game is None or self._game._state != GameState.WIN:
            return prev
        frame = np.asarray(reset_to_level())
        record_obs(frame)
        record_act(self._encode_step(
            Action(RESET_ACTION), optimal=[Action(RESET_ACTION)],
            phase="reset", changed=not np.array_equal(frame, prev), n_obs=1))
        return frame


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new():
    solver = SpacekobanSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once."""
    _solver, game, expert, solvable = _new()
    idx = game._game.obj_name_to_idx
    total = ties = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        crates = sum(1 for row in eng.grid for cell in row
                     if idx["crate"] in cell)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            bad += 1
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tie_steps = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        ties += tie_steps
        print(f"  L{level:2d}: {len(plan):3d} presses  {crates} crate(s)  "
              f"{eng.height}x{eng.width}  tie steps {tie_steps:3d}  "
              f"{'WIN' if won else 'NOT A WIN'}")
    print(f"  {game.n_levels} levels, {len(solvable)} solvable, {total} presses, "
          f"{ties} with a tie, {bad} broken")
    return bad


def _selfcheck(full: bool = False, verbose: bool = True) -> int:
    """Five passes, all against the real interpreter.

    1. the five launch measurements, on hand-built boards;
    2. ACTION is unbound on every level;
    3. `_Board.step` == the interpreter, over every state of every level's
       reachable component (capped per level unless ``full``);
    4. every plan replays to a WIN and its length equals the field's d*;
    5. the dead-successor census: 0 stranded states on every board, which is
       what makes the plans shortest against the whole game.
    """
    _solver, game, expert, _solvable = _new()
    g = game._game
    idx = g.obj_name_to_idx
    BG = idx["background"]
    eng = game._engine
    bad = 0

    def load(rows):
        chars = {'.': (), 'p': ("player",), '#': ("wall",), '*': ("crate",),
                 'o': ("target",), '"': ("ladder",), '%': ("beam",),
                 ':': ("player", "ladder"), '@': ("crate", "target"),
                 '!': ("beam", "target"), '$': ("crate", "ladder"),
                 ']': ("player", "target"), 'B': ("player", "beam")}
        eng.load_level([[{BG, *(idx[n] for n in chars[ch])} for ch in row]
                        for row in rows])
        return eng

    def where(name):
        return sorted((r, c) for r, row in enumerate(eng.grid)
                      for c, cell in enumerate(row) if idx[name] in cell)

    def marks():
        dirs = (idx["u"], idx["d"], idx["l"], idx["r"])
        return sorted((r, c) for r, row in enumerate(eng.grid)
                      for c, cell in enumerate(row) if cell & set(dirs))

    # -- 1. the five launch measurements ------------------------------------
    cases = [
        ("nothing behind -> the press does nothing",
         ["#######", "#..p..#", "#######"], "right",
         lambda: where("player") == [(1, 3)] and not marks()),
        ("nothing behind -> the cancel rule erases nothing",
         ["#######", '#."p..#', "#######"], "right",
         lambda: where("player") == [(1, 3)] and where("ladder") == [(1, 2)]),
        ("a wall behind -> fly to the far wall",
         ["#######", "#p....#", "#######"], "right",
         lambda: where("player") == [(1, 5)] and not marks()),
        ("a crate behind -> fly, and the crate recoils",
         ["########", "#.*p...#", "########"], "right",
         lambda: where("player") == [(1, 6)] and where("crate") == [(1, 1)]),
        ("a ladder is a launch pad in any direction",
         ["#######", '#..:..#', "#######"], "right",
         lambda: where("player") == [(1, 5)]),
        ("a ladder catches the player after one cell",
         ["#######", '#..:"'+".#", "#######"], "right",
         lambda: where("player") == [(1, 4)]),
        ("a crate sails straight over a ladder",
         ["########", '#p*..".#', "########"], "left",
         lambda: where("crate") == [(1, 6)] and where("player") == [(1, 1)]),
        ("a ladder launch does NOT recoil the crate behind",
         ["########", "#.*:...#", "########"], "right",
         lambda: where("crate") == [(1, 2)] and where("player") == [(1, 6)]),
        ("a beam catches the player",
         ["########", "#p...%.#", "########"], "right",
         lambda: where("player") == [(1, 5)] and not marks()),
        ("a train shoves the braked crate off the beam again",
         ["##########", "#p..*.%..#", "##########"], "right",
         lambda: where("player") == [(1, 6)] and where("crate") == [(1, 8)]),
        ("a train packs against the far wall",
         ["############", "#p.***.....#", "############"], "right",
         lambda: where("player") == [(1, 7)]
         and where("crate") == [(1, 8), (1, 9), (1, 10)]),
        ("off the board behind -> a single plain step",
         ["##########", "#........p", "##########"], "left",
         lambda: where("player") == [(1, 8)] and not marks()),
        ("off the board ahead -> momentum is STRANDED",
         ["##########", "#p........", "##########"], "right",
         lambda: where("player") == [(1, 9)] and marks() == [(1, 9)]
         and not eng.check_win()),
    ]
    for name, rows, direction, ok in cases:
        load(rows)
        eng.step(direction)
        if not ok():
            print(f"    launch rule DISAGREES: {name}")
            bad += 1
    if verbose:
        print(f"  1. launch rules: {len(cases)} measurements")

    # -- 2. ACTION is unbound ------------------------------------------------
    for level in range(game.n_levels):
        game.set_level(level)
        rng = random.Random(f"spacekoban:action:{level}")
        for _ in range(40):
            before = [[frozenset(cell) for cell in row] for row in eng.grid]
            eng.step("action")
            if [[frozenset(cell) for cell in row] for row in eng.grid] != before:
                print(f"    L{level}: ACTION changed the grid")
                bad += 1
                break
            eng.step(rng.choice(DIRECTIONS))
    if verbose:
        print(f"  2. ACTION unbound on all {game.n_levels} levels")

    # -- 3. the model IS the interpreter, over the reachable space -----------
    cap = None if full else 3000
    checked = states_total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, player, crates = expert.read(eng)
        start = (player, crates)
        seen = {start}
        queue = deque([start])
        while queue:
            p, cr = queue.popleft()
            if board.won(p, cr):
                continue
            for d, name in enumerate(DIRECTIONS):
                _seat(eng, expert, p, cr)
                eng.step(name)
                got = _read_state(eng, expert)
                exp = board.step(p, cr, d)
                checked += 1
                if got != exp:
                    print(f"    L{level}: model != engine from {(p, cr)} "
                          f"{name}: engine {got} model {exp}")
                    bad += 1
                    queue.clear()
                    break
                if exp is None:
                    continue
                if exp not in seen and (cap is None or len(seen) < cap):
                    seen.add(exp)
                    queue.append(exp)
        states_total += len(seen)
        if cap is not None and len(seen) >= cap:
            if verbose:
                print(f"    L{level}: cross-check capped at {cap} states "
                      f"(use --full for the whole component)")
    if verbose:
        print(f"  3. model vs engine: {checked} presses over {states_total} "
              f"states")

    # -- 4. plans replay to a WIN, at exactly d* -----------------------------
    for level in range(game.n_levels):
        game.set_level(level)
        board, player, crates = expert.read(eng)
        plan = expert.plan(eng, level)
        field_plan, _states, _dead = board.solve(player, crates)
        if plan is None or field_plan is None or len(plan) != len(field_plan):
            print(f"    L{level}: plan disagrees with the field")
            bad += 1
            continue
        for direction in plan:
            eng.step(direction)
        if not eng.check_win():
            print(f"    L{level}: the plan does not win in the interpreter")
            bad += 1
    if verbose:
        print(f"  4. all {game.n_levels} plans replay to a WIN at d*")

    # -- 5. the dead-successor census ---------------------------------------
    dead_total = states_seen = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, player, crates = expert.read(eng)
        _plan, states, dead = board.solve(player, crates)
        dead_total += dead
        states_seen += states
        if dead:
            print(f"    L{level}: {dead} stranded successors -- the DEAD prune "
                  f"is NOT vacuous here and the plan is only shortest among "
                  f"the states that keep momentum off the board")
            bad += 1
    if verbose:
        print(f"  5. stranded successors over all {states_seen} states of all "
              f"{game.n_levels} levels: {dead_total}")
    return bad


def _seat(eng, expert, player, crates):
    """Write a model state onto the engine grid (terrain untouched)."""
    pid, cid = expert.ids["player"], expert.ids["crate"]
    dirs = [expert.g.obj_name_to_idx[n] for n in ("u", "d", "l", "r")]
    w = eng.width
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            cell.discard(pid)
            cell.discard(cid)
            for x in dirs:
                cell.discard(x)
            p = r * w + c
            if p == player:
                cell.add(pid)
            elif p in crates:
                cell.add(cid)
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _read_state(eng, expert):
    """Engine grid -> ``(player, crates)``, or None when a marker is left on the
    board (which is what `_Board.step` reports as None).

    Scans for the three mutable object classes only: `SpacekobanExpert.read`
    rebuilds the whole terrain and hashes it, which is the right thing once per
    level and the wrong thing once per press."""
    pid, cid = expert.ids["player"], expert.ids["crate"]
    dirs = {expert.g.obj_name_to_idx[n] for n in ("u", "d", "l", "r")}
    w = eng.width
    player = -1
    crates = []
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if not cell:
                continue
            if cell & dirs:
                return None
            if pid in cell:
                player = r * w + c
            elif cid in cell:
                crates.append(r * w + c)
    return player, frozenset(crates)


def _audit(verbose: bool = True) -> int:
    """Every cell COMPOSITION the game can show, rendered on a board filled with
    it, at every cell size the levels actually use, required pairwise distinct.

    Whole FRAMES are compared, not a cell crop: `_render_frame` upscales the
    board to fill 64x64 and then letterboxes it, so the cell grid in the output
    is not `cell_px`-aligned and a crop compares the wrong window
    ([[ps-palette-collisions]]).
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    g = game._game
    idx = g.obj_name_to_idx
    eng = game._engine
    BG = idx["background"]
    comps = {
        "background": (), "wall": ("wall",), "target": ("target",),
        "ladder": ("ladder",), "beam": ("beam",),
        "beam+target": ("beam", "target"),
        "player": ("player",), "player+target": ("player", "target"),
        "player+ladder": ("player", "ladder"),
        "player+beam": ("player", "beam"),
        "player+beam+target": ("player", "beam", "target"),
        "crate": ("crate",), "crate+target": ("crate", "target"),
        "crate+ladder": ("crate", "ladder"),
        "crate+beam": ("crate", "beam"),
        "crate+beam+target": ("crate", "beam", "target"),
    }
    sizes: dict[int, tuple[int, int]] = {}
    for level in range(game.n_levels):
        game.set_level(level)
        h, w = eng.height, eng.width
        sizes.setdefault(max(1, min(64 // h, 64 // w)), (h, w))

    bad = 0
    for cell_px, (h, w) in sorted(sizes.items()):
        frames = {}
        for name, objs in comps.items():
            eng.load_level([[{BG, *(idx[n] for n in objs)} for _ in range(w)]
                            for _ in range(h)])
            frames[name] = _render_frame(eng, g).copy()
        for a, b in itertools.combinations(comps, 2):
            if np.array_equal(frames[a], frames[b]):
                print(f"    cell_px {cell_px}: {a} == {b}")
                bad += 1
        if verbose:
            print(f"  cell_px {cell_px} ({h}x{w}): {len(comps)} compositions, "
                  f"{len(set(f.tobytes() for f in frames.values()))} distinct")
    return bad


#: `--fuzz`'s board mix: (boards, presses per board, seed, open right edge,
#: height, width). The open-edge shapes are the point of the exercise -- they are
#: the only ones that ever strand momentum, which is the branch no real level
#: exercises and the one `_Board.step`'s None return exists for.
_FUZZ_SHAPES = ((400, 30, 1, False, 6, 7), (300, 30, 2, True, 6, 8),
                (200, 40, 3, False, 9, 12), (200, 40, 4, True, 9, 12),
                (300, 20, 5, False, 4, 5), (200, 30, 6, False, 12, 4),
                (200, 30, 7, False, 7, 7), (120, 50, 8, False, 11, 14))


def _fuzz(verbose: bool = True) -> int:
    """Walk `_Board` and the interpreter in lockstep over RANDOM boards.

    `--selfcheck` proves the model on the twenty shipped levels; this proves it
    on boards the author never drew -- crates jammed in threes, beams under
    crates, ladders in mid-air, and (the point) boards with an OPEN side, which
    is what produces the stranded-momentum states level 8's geometry allows but
    none of its reachable states reach. A walk stops when the model reports one,
    since the state it lands in is outside what the search will ever step from.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    g = game._game
    idx = g.obj_name_to_idx
    eng = game._engine
    BG = idx["background"]
    PL, CR = idx["player"], idx["crate"]
    marker_ids = {idx[n] for n in ("u", "d", "l", "r")}
    chars = {'.': (), 'p': ("player",), '#': ("wall",), '*': ("crate",),
             'o': ("target",), '"': ("ladder",), '%': ("beam",),
             ':': ("player", "ladder"), '@': ("crate", "target"),
             '!': ("beam", "target"), '$': ("crate", "ladder"),
             ']': ("player", "target"), 'x': ("target", "ladder"),
             'B': ("player", "beam")}
    fill = ['.', '#', '*', 'o', '"', '%', '@', '$', '!', 'x']
    weights = [40, 8, 14, 10, 10, 8, 4, 3, 3, 3]
    stand = {'.': 'p', 'o': ']', '"': ':', '%': 'B', 'x': ':'}

    def read():
        player, crates = -1, []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & marker_ids:
                    return None
                if PL in cell:
                    player = r * eng.width + c
                elif CR in cell:
                    crates.append(r * eng.width + c)
        return player, frozenset(crates)

    bad = presses = stranded = 0
    closed_stranded = 0
    for boards, walk, seed, open_edge, h, w in _FUZZ_SHAPES:
        rng = random.Random(f"spacekoban:fuzz:{seed}")
        for _ in range(boards):
            while True:
                rows = [[('#' if (r in (0, h - 1) or c == 0
                                  or (c == w - 1 and not open_edge))
                          else rng.choices(fill, weights=weights)[0])
                         for c in range(w)] for r in range(h)]
                free = [(r, c) for r in range(1, h - 1)
                        for c in range(1, w if open_edge else w - 1)
                        if rows[r][c] in stand]
                if free:
                    break
            pr, pc = rng.choice(free)
            rows[pr][pc] = stand[rows[pr][pc]]
            eng.load_level([[{BG, *(idx[n] for n in chars[ch])} for ch in row]
                            for row in rows])
            board = _Board(
                h, w,
                *[[o in eng.grid[p // w][p % w] for p in range(h * w)]
                  for o in (idx["wall"], idx["ladder"], idx["beam"],
                            idx["target"])])
            state = read()
            for _ in range(walk):
                d = rng.randrange(4)
                eng.step(DIRECTIONS[d])
                got = read()
                exp = board.step(state[0], state[1], d)
                presses += 1
                if got != exp:
                    print(f"    model != engine on {rows} from {state} "
                          f"{DIRECTIONS[d]}: engine {got} model {exp}")
                    bad += 1
                    break
                if exp is None:
                    stranded += 1
                    closed_stranded += not open_edge
                    break
                state = exp
    if verbose:
        print(f"  {presses} presses over {sum(s[0] for s in _FUZZ_SHAPES)} "
              f"random boards: {bad} disagreements, {stranded} stranded "
              f"({closed_stranded} of them on a CLOSED board)")
    if closed_stranded:
        print("    momentum stranded on a board with no open side -- the "
              "module docstring's claim about where this can happen is wrong")
        bad += 1
    return bad


def _symmetry(walk_presses: int = 150, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    The rotation augmentation is mandatory and is checked here; the same check
    is the evidence for whether the game may join `_FLIP_GAMES`. The structural
    argument is that nothing here is axis-sensitive -- there is no gravity, the
    win condition names no direction, and the only directional objects in the
    file (the U/D/L/R momentum markers) have TRANSPARENT sprites, so a mirror
    cannot show art the game does not own. Chirality can still hide in rule
    ORDER, so it is measured: both the PLANS and a seeded random walk are
    replayed, the walk reaching boards a plan never visits (a crate flat against
    a wall, a train jammed in a corridor) and pressing the unbound ACTION key.
    """
    solver, game, expert, _solvable = _new()
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    def drive(gm, level, presses):
        gm.set_level(level)
        out = [np.asarray(gm._current_frame)]
        for act in presses:
            fd = gm.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame else gm._current_frame))
        return out

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    # The presentation is drawn per (seed, level), so collect one representative
    # seed per (level, presentation) first and drive only those -- driving all
    # 400 seeds would be 2.4M presses to re-measure the same 16 transforms.
    reps: dict[int, dict[tuple, int]] = {lvl: {} for lvl in range(game.n_levels)}
    seen = set()
    for seed in range(400):
        gm = solver.make_game(seed)
        for level in range(gm.n_levels):
            gm.set_level(level)
            k = (gm._rotation_k, gm._hflip, gm._vflip)
            seen.add(k)
            reps[level].setdefault(k, seed)

    bad = 0
    for level, byk in reps.items():
        if (0, False, False) not in byk:
            print(f"    L{level}: the unaugmented presentation never came up")
            bad += 1
            continue
        rng = random.Random(f"spacekoban:symmetry:{level}")
        walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                for _ in range(walk_presses)]
        ref = {}
        for k in sorted(byk, key=lambda kk: kk != (0, False, False)):
            gm = solver.make_game(byk[k])
            for tag, presses in (
                    ("plan", [screen_action(d, *k) for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *k) for a in walk])):
                frames = drive(gm, level, presses)
                assert (gm._rotation_k, gm._hflip, gm._vflip) == k
                if tag == "plan" and gm._state != GameState.WIN:
                    print(f"    L{level} {k}: plan did not win")
                    bad += 1
                if tag not in ref:
                    ref[tag] = frames
                    continue
                if any(not np.array_equal(transform(a, *k), b)
                       for a, b in zip(ref[tag], frames)):
                    print(f"    L{level} {k}: {tag} frames are not the "
                          f"transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--selfcheck" in sys.argv:
        violations = _selfcheck(full="--full" in sys.argv)
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--fuzz" in sys.argv:
        violations = _fuzz()
        print(f"fuzz: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(SpacekobanSolver.main())
