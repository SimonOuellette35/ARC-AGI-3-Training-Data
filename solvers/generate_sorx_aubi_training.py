"""Generate Phase-1 training data for the PuzzleScript game ps:sorx_aubi
("Sorx-Aubi", Ali Nikkhah).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a FAITHFUL native
port of the interpreter's force/rigid machinery, the searches that port makes
affordable, the proof that the plans it returns are shortest, and the render
fixes without which the win itself was invisible.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_sorx_aubi",
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
Every expert step carries an optimal-press set.

The game
--------
Reach the little square. ``All Target on Player`` and nothing else -- no crate
has to end up anywhere. What stands between the player and the square is a
four-way crate algebra with one genuinely unusual member.

There are four crate objects and they are two PAIRS, a light one and a dark one
of each colour:

* ``AubiCrate``  (light blue) -- an ordinary sokoban crate: the player pushes
  it, a moving crate pushes it, chains of them push each other.
* ``AubiiCrate`` (dark blue) -- pushable like the above, but RIGID: the rule
  ``rigid [ moving AubiiCrate | AubiiCrate ] -> [ moving | moving ]`` spreads
  the force sideways as well as forwards, so a connected clump of them wants to
  travel as one body.
* ``SorxCrate``  (light red) -- appears on no rule's left-hand side with a
  force, so NOTHING can move it. It is a wall that happens to be a crate.
* ``SorxxCrate`` (dark red) -- likewise immovable, and it is never created or
  destroyed either.

The two dark crates are the FIELDS, and they are what the game is actually
about. Two late rules run after every press:

    late [ AubiCrate | SorxxCrate ] -> [ SorxCrate | SorxxCrate ]
    late [ SorxCrate | AubiiCrate ] -> [ AubiCrate | AubiiCrate ]

so a dark-red crate FREEZES every light-blue crate it touches into an immovable
light-red one, and a dark-blue crate THAWS every light-red crate it touches back
into a pushable light-blue one. They fire in that order and each runs to
fixpoint, so the settled type of a crate after a press is:

    adjacent to an Aubii            -> Aubi   (thaw wins over freeze)
    adjacent to a Sorxx, not an Aubii -> Sorx
    adjacent to neither             -> whatever it already was

-- and that last line is why the crate types have to live in the search state
rather than be derived from the board: a crate out in the open remembers which
field touched it last. SorxxCrate and AubiiCrate never change type and are never
created or destroyed, so their COUNTS are invariants of a level; the Sorx/Aubi
population is not.

Put together: a dark-red crate is a permanent freezer with a one-cell aura, a
dark-blue clump is a rolling thawer you push around, and the walls of light-red
crate that seal most of these levels are opened by shoving a dark-blue clump up
against them and then pushing the thawed crate out of the way. Nine of the
fifteen levels have no Sorx/Sorxx at all and are pure rigid-body sokoban;
"Soldiers", "The Quest" and "Dear Fibonacci" are built entirely out of the
thaw.

Three levels ("The Cries", "The Unpredicted Problem", "Dear Fibonacci") put TWO
players on the board. PuzzleScript gives the input force to every Player object,
so they move in lockstep and the puzzle is DE-SYNCING them -- you have to wedge
one against something while the other keeps going. Two of those three also have
two targets, and ``All Target on Player`` wants both covered at the same
instant.

``BrokenPlayer`` (the ``q`` legend character and the ``[ > NormalPlayer ]
[ BrokenPlayer ] -> [ > NormalPlayer ] [ < BrokenPlayer ]`` mirror rule) appears
in no shipped level, and the gravity block at the top of the rules section is
inside parentheses, i.e. commented out. `_Board.read` asserts the first of those
rather than assuming it.

The native model, and why it is a PORT and not a model
-----------------------------------------------------
Every other ps: generator in this repo writes down what the mechanic DOES and
fuzzes that against the interpreter. That was tried here first and it fails, for
a reason worth recording.

The clean statement of the rigid rule is "a connected clump of AubiiCrates moves
as one body, and if any part of it is blocked the whole clump stays put". The
adapter does not implement that. `_tag_rigid_forces` writes
``_force_rigid_group[key] = group_id`` with a FRESH id per rule match and
last-write-wins, so the "rigid group" of a cell is the PAIR it was matched in
most recently, not the connected component; `_resolve_forces` then cancels
along those pairs, one `_resolve_forces` iteration at a time. The observable
consequence is that a clump SHEARS -- on level 1 ("Frozen"), pressing DOWN on a
twelve-cell clump moves eight of its cells and leaves four behind, and which
four depends on the order the interpreter scanned the board in. The
component-cancelling model disagreed with the interpreter within five presses on
four of the fifteen levels.

Re-implementing "rigid" properly in the adapter was rejected: thirty-seven games
in ``data/puzzlescript_games`` use the keyword and a dozen of them already have
verified solvers, so a semantics change there is a change to their corpora, not
a bug fix here. The corpus has to match the engine the agent will actually play.

So `_Board.step` is a port of the adapter's own pipeline for THIS rule set:

  1. every Player gets the input force, scanned row-major (the insertion ORDER
     of the forces dict is load-bearing -- `_resolve_forces` iterates it and
     mutates it as it goes);
  2. the ``startloop`` block is run to a fixpoint: ``[ > Player | AubiCrate ]``,
     then the ``+`` group ``[ > Player | AubiiCrate ]`` / the rigid propagation
     / ``[ > Crate | AubiiCrate ]``, then ``[ > Crate | AubiCrate ]`` -- each
     rule iterating its four directions with the interpreter's forward scan
     order and its second, REVERSE scan pass, and the rigid rule stamping a
     fresh group id on each matched pair;
  3. `_resolve_forces` verbatim: trace each chain, drop the blocked ones,
     cascade the rigid cancellation, detect conflicts, move what is left, up to
     twenty times;
  4. the two late conversion rules.

Two shortcuts are taken and both are exact rather than approximate. Every force
on the board is the INPUT direction (the player's force is, and both ``>`` rules
and the rigid rule's ``moving`` RHS only ever copy it), so a rule instantiated
in any other direction can never match and is skipped -- that one line is the
difference between a model that diverges on the first press and one that does
not. And the interpreter's ``+`` group spins for a full 200 iterations because
the rigid rule reports a match every time even when it changes nothing; the port
stops as soon as the forces dict is unchanged in CONTENT AND ORDER, which is a
fixpoint of the whole iteration (the untouched keys keep their order and the
touched ones are re-appended in the same sequence), so the group-id PARTITION
the last pass leaves behind -- the only thing the cancellation reads -- is the
same one 200 iterations would.

It is ~5600x faster than the interpreter on the biggest level (152us vs 850ms
per press) and that ratio is the whole reason any of these levels can be
searched at all.

``--selfcheck`` is the evidence: a seeded random walk on every level with the
model's board compared against the engine's grid after EVERY press, including
the presses that do nothing.

Expert solver
-------------
`_Board.solve` is a three-rung ladder, and it reports which rung answered:

  1. **A* with an ADMISSIBLE heuristic** (weight 1) -- ``h`` is a bottleneck
     assignment of players to targets over a free-space BFS distance that
     blocks only walls and SorxxCrates, both of which are permanently immovable.
     Sound because a press moves each player at most one cell and some player
     has to finish on each target. On a board with no AubiiCrate anywhere the
     SorxCrates can never thaw, so they are added to the blocked set -- an
     invariant, since Aubii is never created.
  2. **the exact FIELD, for the tie sets.** With ``d*`` known, a forward sweep
     collects every state with ``g + h <= d*`` -- which is every state on a
     shortest start-to-win path, because ``h`` is admissible -- keeping the
     edges between consecutive layers, and a backward sweep from the won states
     over those edges gives the exact distance-to-win on that DAG. The optimal
     set at a state ``d`` presses from a win is then every press whose successor
     is ``d - 1`` from one, with no inference at all. The plan is re-extracted
     from the field in ``DIRS`` order so it is byte-identical across processes.
  3. **a beam** under an inadmissible cost guide (a Dijkstra that charges 12 to
     cross a SorxCrate, 3 an AubiiCrate, 1 an AubiCrate) for anything the first
     two cannot afford. A beam plan is a real WIN -- the win test is the model's,
     which is the interpreter's -- but not a proved-shortest one, and its steps
     are labelled with the press taken.

Measured: rungs 1 and 2 answer eleven of the fifteen levels, all eleven proved
shortest, every step carrying an exact optimal set. The other four are in
``skip_levels`` and the reason is in `_plans`.

What "too hard" means here, because it is not the usual story
------------------------------------------------------------
Levels 5 ("Soldiers"), 10 ("The Quest"), 12 ("The Unpredicted Problem") and 14
("Dear Fibonacci") are not skipped because the search is slow. A breadth-first
sweep of level 5 enumerated four million states out to depth 60 without
reaching a win, so its shortest answer is at least 61 presses of a
four-branching search -- and the thing those presses have to accomplish is to
walk a dark-blue clump across the board, thaw a specific light-red crate with
it, then push that crate aside. No distance heuristic sees any of that: the
free-space estimate at level 5's start is 8. Weighted A* to 200k expansions and
a 12000-wide beam to depth 300 under the cost guide both fail on all four.

The honest fix is a MACRO layer -- successors that are "push this clump to
there" rather than "press left" -- which is what the sokoban-shaped ps: games in
this repo get from `PSPushExpert`; it does not exist for a mechanic where a push
can shear a body, chain through three crates and retype two more. That is the
work these four levels are waiting on, and the eleven that ship are not blocked
on it. ``skip_levels`` is checked by `_plans`, which re-derives the four with
the full ladder and prints what happened, so the day the ladder grows a rung the
list is one line to shorten.

Recovery
--------
`supports_recovery` with RESET-mode recovery. `_search` reads the LIVE board and
runs the whole ladder from there, so an exploration prefix that leaves the board
anywhere is answered exactly -- including "this board is now dead", which really
happens here (wedge the only dark-blue clump into a corner and the light-red
wall can never be opened) and which the A* rung PROVES rather than guesses: it
returns ``dead`` only when the priority queue empties, i.e. the reachable space
is closed and win-free. ``--selfcheck``'s fourth pass walks random boards on
every shipped level and requires every plan to WIN when replayed through the
interpreter and every refusal to be a board an independent forward BFS also
finds no win from.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (``rotation_k`` in {0,1,2,3}) plus an
independent horizontal and vertical flip with the matching directional action
remap (`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group. 11 recorded levels x 16 presentations = 176.

The flips are here on the GOBBLE RUSH argument and not the usual one, because
this game's dynamics are genuinely NOT symmetric. The rigid rule spreads a force
sideways through a clump, the adapter cancels the resulting forces pairwise
rather than per component, and which cells of a partly-blocked clump move
therefore depends on the order the board was scanned in -- which is
chirality-dependent (a press UP scans rows bottom-to-top, LEFT scans columns
right-to-left). ``--symmetry`` measures it instead of arguing about it: over
6000 (state, press) pairs across all fifteen levels, 55 are order-sensitive and
every single one is already split by a ROTATION alone, none by a mirror only. So
the mirrors add no inconsistency the mandatory rotation does not already add,
and they buy four times the presentations.

Everything else is the ordinary argument: gravity-free (the gravity block in the
.txt is commented out), screen-relative input, a win condition that names no
direction, and every sprite invariant under the whole group after the render
fixes below. The only directional rules in the game are the two ``late`` ones
that stamp AubiCorner seam decals between adjacent AubiiCrates, and that decal is
an exact orbit of the group -- it marks the two corners either side of a seam, so
a turn or a mirror maps seam art to seam art exactly as it maps the seam.
``--audit``'s second pass measures that, and ``--symmetry`` replays every level's
plan AND a seeded random walk (which presses the unbound ACTION key as well) at
all 16 presentations, requiring every frame to be the exact transform of the
unaugmented one.

Rendering
---------
Three sprites/layers had to change and the first of them is the whole win
condition. See the header comment in
``data/puzzlescript_games/Sorx-Aubi.txt`` for the full statement; in short:

* ``Target`` shipped as a DarkBlue 3x3 ring on rows/cols 1-3, and the
  NormalPlayer sprite is opaque on exactly those eight pixels. PLAYER ON TARGET
  was pixel-identical to PLAYER ON FLOOR at every cell size -- the single
  composition ``All Target on Player`` is made of could not be seen. It is now
  Purple on the four corners plus the centre, which are pixels the player leaves
  transparent and which survive both the 5x5 -> 3x3 sampler (rows/cols {0,2,4})
  and the 5x5 -> 4x4 one (rows/cols {0,1,3,4}).
* ``SorxxCrate`` was ``Red``, which is ARC palette index 8 -- the same index as
  ``SorxCrate``'s ``LightRed``. The permanent freezer and the crate it freezes
  were the same colour, telling themselves apart by sprite shape alone. Now
  ``DarkRed`` (13), which nothing else uses.
* ``Target`` moved to the LAST collision layer. Layer ORDER decides only draw
  order -- blocking is per-layer equality and Target is alone on its layer
  either way -- so it is mechanically inert, and it is what keeps the goal
  square visible under a crate parked on it. Inert was MEASURED twice: a
  900-press seeded walk over all fifteen levels leaves the engine's grid hash
  bit-identical across the edit (recorded in the .txt header), and
  ``--selfcheck``'s first pass re-derives every board from a model that has no
  notion of layers at all.

``--audit`` has two passes. The first renders every reachable cell COMPOSITION
as a whole 64x64 frame of a uniform board, at every cell size the fifteen levels
use (3, 4, 5, 7, 8 and 9 px), and requires them pairwise distinct. Whole frames
rather than one cell sliced out of a mixed board: `_render_frame` upscales and
centre-pads, so slicing by ``cell_px`` arithmetic reads the wrong pixels (the
ps:explod lesson). The second turns and mirrors every composition eight ways and
compares it against every other, which is the art half of the ``_FLIP_GAMES``
argument; it runs on SQUARE boards, because a rotation of a non-square board also
moves the letterbox and every comparison would pass for the wrong reason.

One family of collisions is reported and ACCEPTED rather than fixed, and it is
named in the audit output: an ``AubiiCrate`` standing on the target renders the
same with or without its ``AubiCorner`` seam decals, because the only pixels the
Aubii sprite leaves transparent are its four corners and that is exactly where
both the decals and the target's own pixels go. The decals are redundant -- they
are a function of which neighbouring cells hold Aubii, which the neighbours
themselves show -- and the target is not, so the target wins the corner.
``player + crate``, ``crate + crate`` and ``wall + anything`` are absent from
the audit on purpose: those all share one collision layer and the engine can
never put two of them in a cell.

Usage (run from the repo root):
    python solvers/generate_sorx_aubi_training.py --episodes 200 \
        --out data/training_multi_level/puzzlescript_sorx_aubi

    python solvers/generate_sorx_aubi_training.py --plans      # level report
    python solvers/generate_sorx_aubi_training.py --plans --include-skipped
    python solvers/generate_sorx_aubi_training.py --selfcheck  # model + optimality
    python solvers/generate_sorx_aubi_training.py --selfcheck --pass 4
    python solvers/generate_sorx_aubi_training.py --engine     # interpreter proof
    python solvers/generate_sorx_aubi_training.py --audit      # rendering
    python solvers/generate_sorx_aubi_training.py --symmetry   # augmentation
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                          # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION                     # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,   # noqa: E402
                                     screen_action)
from utils.rotation import inverse_remap_action_full                  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Sorx-Aubi"

#: Engine directions, in the order ties are broken. ACTION5 is bound to nothing
#: in this game (no rule has ``action`` on its left-hand side and the late rules
#: are already at their fixpoint), so it is a genuine no-op and branching on it
#: would double the search for nothing. The exploration prefix still presses it,
#: which is why ``--symmetry``'s random walk does too.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Cell codes for the ONE collision layer that holds everything that can block
#: or be blocked (``Player, Wall, SorxxCrate, SorxCrate, AubiiCrate,
#: AubiCrate``). One code per cell is exact precisely because they share a layer.
EMPTY, WALL, SORXX, PLAYER, AUBI, SORX, AUBII = range(7)

#: The ``Crate`` or-group, i.e. what ``[ > Crate | ... ]`` matches.
_CRATE_CODES = frozenset((SORXX, AUBI, SORX, AUBII))

_INF = 1 << 28


class _OverBudget(Exception):
    """Raised inside `_Board._astar_plan` when the tie-set measurement has spent
    its whole budget. The caller still has a shortest plan; only the labelling
    is given up on."""

#: Cost of stepping onto a cell for the inadmissible BEAM guide. A light-blue
#: crate is free to walk through (you push it as you go), a dark-blue clump
#: costs a little (it may be rigid-blocked), and a light-red crate costs a lot
#: (it has to be thawed by a clump first, which is a detour of its own).
_GUIDE_COST = {AUBI: 1, AUBII: 3, SORX: 12}

#: Disk cache of every level's start plan AND its optimal-action sets. Level 11
#: alone is a few seconds of search, so this is here so a
#: `parallelize_generator` fan-out shares one derivation rather than repeating
#: it on every core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "sorx_aubi_plans.json"


# ---------------------------------------------------------------------------
# The native model -- a port of the adapter's own pipeline
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, the ported dynamics, and the searches.

    Cells are flat ``r * w + c`` indices. A STATE is

        (players, aubi crates, sorx crates, aubii crates)

    -- a sorted tuple and three frozensets. Walls and SorxxCrates never move and
    never change type, so they live here rather than in the state; the other
    three sets are all dynamic, and the crate TYPES have to be carried because a
    crate out of reach of both fields keeps whatever type it last acquired.

    Targets are in the state's way of nothing: they are alone on their own
    collision layer, block nothing, and appear only in `won`.
    """

    __slots__ = ("h", "w", "n", "wall", "sorxx", "targets", "static",
                 "edge", "nbrs", "_tables", "_no_aubii")

    def __init__(self, h, w, walls, sorxx, targets):
        self.h, self.w, self.n = h, w, h * w
        self.wall = frozenset(walls)
        self.sorxx = frozenset(sorxx)
        self.targets = frozenset(targets)
        static = bytearray(self.n)
        for i in walls:
            static[i] = WALL
        for i in sorxx:
            static[i] = SORXX
        #: The immovable furniture, pre-painted so `step` starts from a
        #: C-level bytearray copy rather than rebuilding the board.
        self.static = bytes(static)
        self.edge = []
        self.nbrs = []
        for i in range(self.n):
            r, c = divmod(i, w)
            row = []
            for d in DIRS:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self.edge.append(tuple(row))
            self.nbrs.append(tuple(x for x in row if x >= 0))
        #: Free-space distance tables, memoized by the set of extra blocked
        #: cells (see `heuristic`).
        self._tables: dict = {}
        self._no_aubii = None

    # -- the interpreter's scan orders ---------------------------------------
    def _orders(self, cells):
        """The four position orders `_apply_single_rule_forces` can scan in.

        The interpreter takes ``_candidate_positions`` (a set) and SORTS it, so
        the order is a fact about the rule's direction and nothing else:
        forward it is ``(-r, c)`` for UP, ``(r, -c)`` for LEFT and plain
        ascending otherwise; the second, reverse pass is plain ascending for UP
        and LEFT and descending otherwise. Reproducing it is not optional --
        which cells of a sheared clump move is decided here."""
        w = self.w
        asc = sorted(cells)
        return (asc, asc[::-1],
                sorted(cells, key=lambda i: (-(i // w), i % w)),
                sorted(cells, key=lambda i: (i // w, -(i % w))))

    @staticmethod
    def _scan(orders, di, reverse):
        asc, desc, up_order, left_order = orders
        if not reverse:
            if di == 0:                       # up
                return up_order
            if di == 2:                       # left
                return left_order
            return asc
        if di in (0, 2):
            return asc
        return desc

    # -- one press -----------------------------------------------------------
    def step(self, state, di):
        """The state after pressing ``DIRS[di]``, or ``state`` if nothing moves
        (a non-move: no shortest path contains one)."""
        players, aubi, sorx, aubii = state
        if not players:
            return state
        cells = bytearray(self.static)
        for i in players:
            cells[i] = PLAYER
        for i in aubi:
            cells[i] = AUBI
        for i in sorx:
            cells[i] = SORX
        for i in aubii:
            cells[i] = AUBII

        edge = self.edge
        #: cell -> the input direction. Every force in this game IS the input
        #: direction, so the value carries no information -- but the dict's
        #: insertion ORDER does, because `_resolve` walks it and mutates it as
        #: it goes, and a rule that re-asserts a force pops and re-inserts the
        #: key, moving it to the back.
        forces = {}
        for i in sorted(players):
            forces[i] = 1
        rigid: dict = {}
        gid = 0

        order_player = self._orders(players)
        order_aubii = self._orders(aubii)
        order_crate = self._orders(tuple(aubi) + tuple(sorx) + tuple(aubii)
                                   + tuple(self.sorxx))

        def single(kind, dd):
            """One `_apply_single_rule_forces` call: the forward scan, then the
            interpreter's second REVERSE scan if the first found anything.

            ``kind`` 0..4 is A / B1 / B3 / C / the rigid rule (see the module
            docstring). Only the rigid rule uses ``moving`` (matches a force in
            any direction); the other four use ``>``, which binds to the RULE's
            own direction, and since every force on the board is the input
            direction those four can only ever match when the rule direction IS
            the input direction."""
            nonlocal gid
            if kind != 4 and dd != di:
                return False
            if kind == 4:
                orders = order_aubii
            elif kind in (0, 1):
                orders = order_player
            else:
                orders = order_crate
            had = False
            for reverse in (False, True):
                if reverse and not had:
                    break
                for pos in self._scan(orders, dd, reverse):
                    nxt = edge[pos][dd]
                    if nxt < 0:
                        continue
                    here = cells[pos]
                    if kind == 0:                    # [ > Player | AubiCrate ]
                        if (here != PLAYER or pos not in forces
                                or cells[nxt] != AUBI):
                            continue
                    elif kind == 1:                  # [ > Player | AubiiCrate ]
                        if (here != PLAYER or pos not in forces
                                or cells[nxt] != AUBII):
                            continue
                    elif kind == 2:                  # [ > Crate | AubiiCrate ]
                        if (here not in _CRATE_CODES or pos not in forces
                                or cells[nxt] != AUBII):
                            continue
                    elif kind == 3:                  # [ > Crate | AubiCrate ]
                        if (here not in _CRATE_CODES or pos not in forces
                                or cells[nxt] != AUBI):
                            continue
                    else:                     # rigid [ moving Aubii | Aubii ]
                        if (here != AUBII or pos not in forces
                                or cells[nxt] != AUBII):
                            continue
                    # The left cell's force is consumed by the LHS movement
                    # modifier and re-asserted by the RHS, which moves its key
                    # to the back of the dict. The right cell keeps its own
                    # position if it already had a force.
                    del forces[pos]
                    forces[pos] = 1
                    if nxt not in forces:
                        forces[nxt] = 1
                    if kind == 4:
                        gid += 1
                        rigid[pos] = gid
                        rigid[nxt] = gid
                    had = True
            return had

        def run(kind):
            """`_execute_rule`: iterate the rule's four directions until the
            forces stop changing. The grid is never touched by any of these
            rules, so "changed" is exactly "the forces changed"."""
            fired = False
            for _ in range(200):
                before = set(forces)
                had = False
                for dd in range(4):
                    if single(kind, dd):
                        had = True
                if had:
                    fired = True
                if not had or set(forces) == before:
                    break
            return fired

        for _ in range(200):                        # the startloop block
            outer = set(forces)
            run(0)
            previous = None
            for _ in range(200):                    # the '+' group
                fired = run(1) | run(4) | run(2)
                if not fired:
                    break
                current = list(forces)
                if current == previous:
                    # Content AND order are unchanged, so every further
                    # iteration is the identity -- including the rigid rule's
                    # re-stamping, which lands on the same PARTITION with
                    # different ids, and only the partition is ever read.
                    break
                previous = current
            run(3)
            if set(forces) == outer:
                break

        self._resolve(cells, forces, rigid, di)

        players_out, aubi_out, sorx_out, aubii_out = [], [], [], []
        for i in range(self.n):
            code = cells[i]
            if code == PLAYER:
                players_out.append(i)
            elif code == AUBI:
                aubi_out.append(i)
            elif code == SORX:
                sorx_out.append(i)
            elif code == AUBII:
                aubii_out.append(i)
        aubi_new, sorx_new = self._convert(aubi_out, sorx_out, aubii_out)
        return (tuple(players_out), aubi_new, sorx_new, frozenset(aubii_out))

    # -- `_resolve_forces`, ported -------------------------------------------
    def _resolve(self, cells, forces, rigid, di):
        """Move what the forces say can move.

        Trace each chain of same-direction forces to its end; a chain whose
        endpoint is free moves whole, one whose endpoint is a wall, a Sorx, a
        Sorxx or the board edge has every force in it dropped. A dropped force
        that carried a rigid group id takes the rest of its group with it, which
        is what makes a clump refuse to shear -- SOMETIMES, because the id is
        the last PAIR the cell was matched in and not its component. Repeat up
        to twenty times: a cell freed by one pass can be claimed by the next.

        The conflict phase of the interpreter is ported but can never fire here:
        every force is the same direction, so two distinct chains have distinct
        destination cells."""
        edge = self.edge

        def cascade(seed):
            if not rigid:
                return set(seed)
            dead = {rigid[k] for k in seed if k in rigid}
            if not dead:
                return set(seed)
            extra = set(seed)
            for key, group in list(rigid.items()):
                if group in dead and key in forces:
                    extra.add(key)
            for key in extra:
                forces.pop(key, None)
                rigid.pop(key, None)
            return extra

        for _ in range(20):
            moved_any = False
            movable = []
            blocked = set()
            resolved = set()
            for cell in list(forces):
                if cell in resolved:
                    continue
                if not cells[cell]:
                    forces.pop(cell, None)
                    resolved.add(cell)
                    rigid.pop(cell, None)
                    continue
                chain = [cell]
                cur = cell
                free = False
                while True:
                    nxt = edge[cur][di]
                    if nxt < 0:
                        break
                    if not cells[nxt]:
                        free = True
                        break
                    if nxt in forces:
                        chain.append(nxt)
                        cur = nxt
                    else:
                        break
                if free:
                    movable.append(chain)
                else:
                    for x in chain:
                        forces.pop(x, None)
                        resolved.add(x)
                        blocked.add(x)
            if blocked and rigid:
                extra = cascade(blocked)
                if extra - blocked:
                    movable = [ch for ch in movable
                               if not any(x in extra for x in ch)]
            for x in blocked:
                rigid.pop(x, None)

            #: A chain whose head is somebody else's tail is the same push seen
            #: from further back; only the longest is moved.
            non_head = {}
            for i, chain in enumerate(movable):
                for x in chain[1:]:
                    non_head[x] = i
            subsumed = {i for i, chain in enumerate(movable)
                        if chain[0] in non_head}

            for i, chain in enumerate(movable):
                if i in subsumed:
                    continue
                for x in reversed(chain):
                    dest = edge[x][di]
                    cells[dest] = cells[x]
                    cells[x] = EMPTY
                    forces.pop(x, None)
                    rigid.pop(x, None)
                    moved_any = True
            if not moved_any:
                break

    # -- the two late conversion rules ---------------------------------------
    def _convert(self, aubi, sorx, aubii):
        """``late [ AubiCrate | SorxxCrate ] -> [ SorxCrate | ... ]`` then
        ``late [ SorxCrate | AubiiCrate ] -> [ AubiCrate | ... ]``.

        Each runs to a fixpoint and neither creates matches for itself (the
        first makes no Aubi, the second makes no Sorx), so the fixpoint is
        exactly "every Aubi touching a Sorxx becomes Sorx" followed by "every
        Sorx touching an Aubii becomes Aubi" -- which is why a crate touching
        BOTH fields ends up Aubi, and why a crate touching neither is unchanged
        and has to be remembered."""
        aubi = set(aubi)
        sorx = set(sorx)
        aubii = set(aubii)
        sorxx = self.sorxx
        nbrs = self.nbrs
        for cell in list(aubi):
            if any(x in sorxx for x in nbrs[cell]):
                aubi.discard(cell)
                sorx.add(cell)
        for cell in list(sorx):
            if any(x in aubii for x in nbrs[cell]):
                sorx.discard(cell)
                aubi.add(cell)
        return frozenset(aubi), frozenset(sorx)

    def won(self, state) -> bool:
        """``All Target on Player`` -- every target square carries a player.
        Written as the subset test the condition literally states, so a level
        with more players than targets (levels 11 and 14) is answered right."""
        return self.targets.issubset(state[0])

    # -- the admissible heuristic --------------------------------------------
    def _blocked(self, state) -> frozenset:
        """Cells that are provably impassable for the rest of the episode.

        Walls and SorxxCrates always. SorxCrates too when the board holds no
        AubiiCrate at all: the only rule that thaws a Sorx needs an Aubii beside
        it, and no rule ever creates one, so on such a board every Sorx is
        permanent (and the Aubi crates that freeze later only ADD to the set,
        which keeps the estimate a lower bound)."""
        if not state[3]:
            return frozenset(self.wall | self.sorxx | state[2])
        return frozenset(self.wall | self.sorxx)

    def _dist_tables(self, blocked):
        """``[{cell: presses}, ...]`` per target, over the free-space graph."""
        tables = self._tables.get(blocked)
        if tables is not None:
            return tables
        tables = []
        for target in sorted(self.targets):
            dist = {target: 0}
            queue = deque([target])
            while queue:
                cur = queue.popleft()
                for nxt in self.nbrs[cur]:
                    if nxt in blocked or nxt in dist:
                        continue
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
            tables.append(dist)
        if len(self._tables) < 4096:
            self._tables[blocked] = tables
        return tables

    def heuristic(self, state) -> int:
        """A lower bound on the presses left, or ``_INF`` when the board is
        provably lost.

        The win needs some player standing on each target, and a press moves any
        one player at most one cell, so the answer is at least the BOTTLENECK of
        an injective players-to-targets assignment: minimise, over assignments,
        the largest free-space distance any assigned player still has to cover.
        Free-space means walls and permanently-immovable crates only -- every
        other crate can be pushed out of the way, and pushing one costs nothing
        beyond the step that does it, which is what keeps the bound sound.

        The levels have at most two targets and at most two players, so the
        assignment is enumerated rather than matched."""
        players = state[0]
        if not players or len(players) < len(self.targets):
            return _INF
        tables = self._dist_tables(self._blocked(state))
        best = _INF
        for choice in itertools.permutations(players, len(tables)):
            worst = 0
            for player, table in zip(choice, tables):
                dist = table.get(player, _INF)
                if dist >= _INF:
                    worst = _INF
                    break
                if dist > worst:
                    worst = dist
            if worst < best:
                best = worst
        return best

    def guide(self, state) -> int:
        """The BEAM's inadmissible score: the same assignment, but over a
        Dijkstra that charges for what is standing in the way instead of walking
        through it for free. Crossing a light-red crate costs 12 because it has
        to be thawed by a dark-blue clump first; a clump costs 3 because it may
        refuse to move; a light-blue crate costs 1 because pushing it IS the
        step. Summed over the assignment rather than maxed -- both targets have
        to be reached, and the beam needs to feel the second one."""
        players, aubi, sorx, aubii = state
        if not players or len(players) < len(self.targets):
            return _INF
        cost = {}
        for cell in aubi:
            cost[cell] = _GUIDE_COST[AUBI]
        for cell in sorx:
            cost[cell] = _GUIDE_COST[SORX]
        for cell in aubii:
            cost[cell] = _GUIDE_COST[AUBII]
        hard = self.wall | self.sorxx
        tables = []
        for target in sorted(self.targets):
            dist = {target: 0}
            queue = [(0, target)]
            while queue:
                dcur, cur = heapq.heappop(queue)
                if dcur > dist.get(cur, _INF):
                    continue
                for nxt in self.nbrs[cur]:
                    if nxt in hard:
                        continue
                    step = dcur + cost.get(nxt, 1)
                    if step < dist.get(nxt, _INF):
                        dist[nxt] = step
                        heapq.heappush(queue, (step, nxt))
            tables.append(dist)
        best = _INF
        for choice in itertools.permutations(players, len(tables)):
            total = 0
            for player, table in zip(choice, tables):
                value = table.get(player, _INF)
                if value >= _INF:
                    total = _INF
                    break
                total += value
            if total < best:
                best = total
        return best

    # -- rung 1: exact A* ----------------------------------------------------
    def astar(self, state, cap: int, want_path: bool = False,
              spend: list | None = None) -> tuple:
        """``(d_star, status, path)``: the length of a shortest win, one of
        ``"exact"`` / ``"dead"`` / ``"capped"``, and (when ``want_path``) the
        press indices of one shortest plan.

        ``"dead"`` is a PROOF, not a budget: the priority queue empties only
        when every state reachable from here has been expanded and none of them
        wins. That distinction is what `record_level` needs -- a dead board must
        RESET, a capped one has simply not been searched hard enough.

        ``want_path`` keeps a parent map, which is why it is opt-in: the tie-set
        machinery calls this thousands of times just to ask "how far is it from
        here?" and has no use for the route. ``spend`` is a one-element list the
        expansions are added to, so a caller running many searches can hold them
        to one shared budget."""
        if self.won(state):
            return 0, "exact", []
        if self.heuristic(state) >= _INF:
            return None, "dead", None
        if spend is None:
            spend = [0]
        spent_at_entry = spend[0]
        counter = 0
        queue = [(self.heuristic(state), 0, 0, state)]
        best = {state: 0}
        parent: dict = {}
        expanded = 0

        def route(end):
            out = []
            cur = end
            while cur in parent:
                cur, di = parent[cur]
                out.append(di)
            return out[::-1]

        while queue:
            _f, g, _c, cur = heapq.heappop(queue)
            if g > best.get(cur, _INF):
                continue
            for di in range(4):
                nxt = self.step(cur, di)
                expanded += 1
                spend[0] += 1
                if nxt == cur:
                    continue
                ng = g + 1
                if ng >= best.get(nxt, _INF):
                    continue
                best[nxt] = ng
                if want_path:
                    parent[nxt] = (cur, di)
                if self.won(nxt):
                    return ng, "exact", (route(nxt) if want_path else None)
                est = self.heuristic(nxt)
                if est >= _INF:
                    continue
                counter += 1
                heapq.heappush(queue, (ng + est, ng, counter, nxt))
            if expanded >= cap:
                return None, "capped", None
        return None, "dead", None

    # -- rung 2: the exact field, for the tie sets ---------------------------
    def field(self, state, d_star: int, cap: int):
        """``{state: presses-to-win}`` over every state on a shortest
        start-to-win path, or None if that set exceeds ``cap``.

        Sweep 1 walks forward from ``state`` keeping only states with
        ``g + h <= d_star`` and only the edges between consecutive layers.
        Nothing on a shortest path is lost: ``h`` is admissible, so a state
        ``s`` on one has ``g[s] + h[s] <= g[s] + rest[s] = d_star``, and so does
        everything after it. Sweep 2 runs backward from the won states over the
        edges sweep 1 kept, which is why no predecessor function is needed --
        this game's moves are not invertible in closed form (a push can shear a
        rigid clump and retype two crates on the way past).

        States off every shortest path may be reached by sweep 1 and left
        unreached by sweep 2; nothing asks about those."""
        layers = [[state]]
        g_of = {state: 0}
        succ: dict = {}
        seen = 1
        for depth in range(d_star):
            layer = []
            for cur in layers[-1]:
                out = []
                for di in range(4):
                    nxt = self.step(cur, di)
                    if nxt == cur:
                        continue
                    known = g_of.get(nxt)
                    if known is None:
                        if self.heuristic(nxt) + depth + 1 > d_star:
                            continue
                        g_of[nxt] = depth + 1
                        layer.append(nxt)
                        seen += 1
                        if seen > cap:
                            return None
                    elif known != depth + 1:
                        continue
                    out.append((di, nxt))
                succ[cur] = out
            layers.append(layer)
        preds: dict = {}
        for cur, out in succ.items():
            for _di, nxt in out:
                preds.setdefault(nxt, []).append(cur)
        rest = {}
        queue = deque()
        for cur in g_of:
            if self.won(cur):
                rest[cur] = 0
                queue.append(cur)
        while queue:
            cur = queue.popleft()
            for prev in preds.get(cur, ()):
                if prev in rest:
                    continue
                rest[prev] = rest[cur] + 1
                queue.append(prev)
        return rest

    def optimal(self, rest, state):
        """``[(direction index, successor), ...]``: every press on a shortest
        path from ``state``. Ties come out in ``DIRS`` order, which is what
        makes a re-derived plan byte-identical across processes."""
        left = rest.get(state)
        if not left:
            return []
        out = []
        for di in range(4):
            nxt = self.step(state, di)
            if nxt != state and rest.get(nxt, -1) == left - 1:
                out.append((di, nxt))
        return out

    # -- rung 3: the beam ----------------------------------------------------
    def beam(self, state, width: int, depth: int):
        """A win under the cost guide, or None. Not shortest and not claimed to
        be.

        The ``seen`` set is per-LAYER on purpose: a beam that dedupes against
        everything it has ever expanded starves, because a plateau in the guide
        is exactly where it needs to revisit."""
        if self.won(state):
            return []
        frontier = [(self.guide(state), state, [])]
        for _ in range(depth):
            nxt_layer = []
            seen = set()
            for _score, cur, path in frontier:
                for di in range(4):
                    nxt = self.step(cur, di)
                    if nxt == cur or nxt in seen:
                        continue
                    if self.won(nxt):
                        return path + [di]
                    seen.add(nxt)
                    score = self.guide(nxt)
                    if score >= _INF:
                        continue
                    nxt_layer.append((score, nxt, path + [di]))
            if not nxt_layer:
                return None
            nxt_layer.sort(key=lambda item: item[0])
            frontier = nxt_layer[:width]
        return None

    # -- the ladder ----------------------------------------------------------
    def solve(self, state, astar_cap=600_000, field_cap=400_000,
              beam_width=1500, beam_depth=300, allow_beam=True,
              tie_budget=400_000):
        """``(Plan, rung)`` or ``(None, rung)``.

        ``rung`` is ``"field"`` (shortest, exact tie sets), ``"astar"``
        (shortest, tie sets measured one bounded re-search at a time),
        ``"astar-plain"`` (shortest, every step labelled with its own press
        because the measurement exceeded ``tie_budget``), ``"beam"`` (a win, no
        optimality claim), or ``"dead"`` / ``"capped"`` when there is no plan.

        Every rung that returns a plan labels every step, which is the
        always-emit-optimal-targets rule: the worst case is a singleton set
        containing the press the expert actually took, never nothing."""
        d_star, status, path = self.astar(state, astar_cap, want_path=True)
        if status == "dead":
            return None, "dead"
        if status == "exact":
            if d_star == 0:
                return Plan([], []), "field"
            rest = self.field(state, d_star, field_cap)
            if rest is not None and rest.get(state) == d_star:
                presses, optsets = [], []
                cur = state
                for _ in range(d_star):
                    best = self.optimal(rest, cur)
                    if not best:
                        break
                    presses.append(DIRS[best[0][0]])
                    optsets.append([DIRS[di] for di, _ in best])
                    cur = best[0][1]
                if len(presses) == d_star:
                    return Plan(presses, optsets), "field"
            # The field did not fit. The plan is still shortest -- A* found it
            # -- so the only thing left to decide is how well the steps can be
            # labelled, and that is measured under a BUDGET: without one, a
            # 40-press plan asks for 160 fresh A* runs and a recovery re-plan
            # that used to take a second takes half an hour.
            plan = self._astar_plan(state, d_star, astar_cap, tie_budget)
            if plan is not None:
                return plan, "astar"
            presses = [DIRS[di] for di in path]
            return Plan(presses, [[q] for q in presses]), "astar-plain"
        if not allow_beam:
            return None, status if status != "exact" else "capped"
        found = self.beam(state, beam_width, beam_depth)
        if found is None:
            return None, "capped"
        presses = [DIRS[di] for di in found]
        return Plan(presses, [[p] for p in presses]), "beam"

    def _astar_plan(self, state, d_star: int, cap: int, budget: int):
        """The shortest plan plus its tie sets, measured by re-search rather
        than read off a field, or None if the measurement runs past ``budget``
        expansions in total.

        Used when the field does not fit: at every step, each successor is asked
        "can you still finish in what is left?", pruned first by the admissible
        estimate and answered by an A* whose answers are memoized across the
        whole plan. The total budget is what keeps that from degenerating -- a
        long plan on a board whose field does not fit means every one of those
        questions is its own expensive search, and the caller has a perfectly
        good shortest plan to fall back on."""
        memo: dict = {}
        #: Shared across every sub-search, so the budget is expansions in total
        #: rather than per question.
        spent = [0]

        def within(node, limit):
            if self.won(node):
                return 0
            if limit <= 0 or self.heuristic(node) > limit:
                return None
            cached = memo.get(node)
            if cached is not None and (cached[0] is not None
                                       or cached[1] >= limit):
                return cached[0]
            if spent[0] >= budget:
                raise _OverBudget
            got, status, _path = self.astar(node, min(cap, budget - spent[0]),
                                            spend=spent)
            if status == "capped":
                raise _OverBudget
            if status != "exact":
                got = None
            memo[node] = (got, limit)
            return got

        presses, optsets = [], []
        cur = state
        try:
            for i in range(d_star):
                remaining = d_star - i
                best = []
                for di in range(4):
                    nxt = self.step(cur, di)
                    if nxt == cur:
                        continue
                    if within(nxt, remaining - 1) == remaining - 1:
                        best.append(di)
                if not best:
                    return None
                presses.append(DIRS[best[0]])
                optsets.append([DIRS[di] for di in best])
                cur = self.step(cur, best[0])
        except _OverBudget:
            return None
        if not self.won(cur):
            return None
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class SorxAubiExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Board` ladder.

    The base class keeps the memo, the level scoping and the disk cache; only
    the strategy underneath changes. `_search` reads the ENGINE's current grid
    every time, so a re-plan from an arbitrary state -- a recovery prefix, an
    epsilon detour -- is answered from that state and not from a stored path,
    including "this board is now dead"."""

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, which is canonical WITHIN a level but
    #: not across them: walls, SorxxCrates and targets are static per level and
    #: differ between the fifteen.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Budgets for `_Board.solve`. The field is the expensive rung and it is the
    #: one that buys exact tie sets, so it gets the room; level 11 (33 presses,
    #: the longest shipped plan) is the level that uses it.
    astar_cap = 600_000
    field_cap = 400_000
    beam_width = 1500
    beam_depth = 300
    #: Total expansions `_Board._astar_plan` may spend MEASURING tie sets when
    #: the exact field does not fit. Past it the plan still ships -- shortest,
    #: with every step labelled by the press taken.
    tie_budget = 400_000
    #: Let `_Board.solve` fall through to its beam rung. On by default -- a
    #: recovery board the exact rungs cannot afford is still worth a real WIN,
    #: even an unproved one. `_selfcheck`'s recovery pass turns it off, because
    #: there a capped board should be REPORTED as inconclusive rather than
    #: answered: the beam costs a Dijkstra per node and would dominate the run.
    allow_beam = True

    def setup(self) -> None:
        self.player_ids = set(self.game._engine._player_indices)
        self.broken_ids = set(self.g.resolve_object_name("brokenplayer"))
        self.target_ids = set(self.g.resolve_object_name("target"))
        self.wall_ids = set(self.g.resolve_object_name("wall"))
        self.aubi_ids = set(self.g.resolve_object_name("aubicrate"))
        self.sorx_ids = set(self.g.resolve_object_name("sorxcrate"))
        self.sorxx_ids = set(self.g.resolve_object_name("sorxxcrate"))
        self.aubii_ids = set(self.g.resolve_object_name("aubiicrate"))
        #: The `_key` alphabet: players and the three crate classes that can
        #: change. SorxxCrates are static and the four AubiCorner decals are a
        #: FUNCTION of the AubiiCrate positions (the late rules delete them all
        #: and redraw the seams from scratch every press), so leaving both out
        #: is an exact projection rather than an aliasing one -- which is what
        #: lets `--engine` enumerate the interpreter on this key.
        self.dyn_ids = (self.player_ids | self.aubi_ids | self.sorx_ids
                        | self.aubii_ids)
        #: `_Board`s by STATIC signature (see `read`), not by level index.
        self._boards: dict = {}
        #: Which rung of the ladder answered each level, for `_plans`.
        self.rungs: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "SorxAubiExpert plans on a native model; heuristic is unused")

    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn))

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid.

        Boards are cached by their STATIC signature (dimensions, walls,
        SorxxCrates, targets) rather than by level index, so the edge tables and
        the distance tables are shared by every state of a level."""
        h, w = eng.height, eng.width
        walls, sorxx, targets = [], [], []
        players, aubi, sorx, aubii = [], [], [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.sorxx_ids:
                    sorxx.append(i)
                if cell & self.target_ids:
                    targets.append(i)
                if cell & self.aubi_ids:
                    aubi.append(i)
                if cell & self.sorx_ids:
                    sorx.append(i)
                if cell & self.aubii_ids:
                    aubii.append(i)
                if cell & self.player_ids:
                    players.append(i)
                    if cell & self.broken_ids:
                        # The mirror rule is real but no shipped level places a
                        # `q`. A model that quietly treated one as an ordinary
                        # player would walk it the wrong way.
                        raise AssertionError(
                            "BrokenPlayer on the board: the mirror rule "
                            "[ > NormalPlayer ] [ BrokenPlayer ] is not modelled")
        sig = (h, w, tuple(walls), tuple(sorxx), tuple(targets))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, sorxx, targets)
        return board, (tuple(players), frozenset(aubi), frozenset(sorx),
                       frozenset(aubii))

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        plan, rung = board.solve(state, astar_cap=self.astar_cap,
                                 field_cap=self.field_cap,
                                 beam_width=self.beam_width,
                                 beam_depth=self.beam_depth,
                                 allow_beam=self.allow_beam,
                                 tie_budget=self.tie_budget)
        self.rungs[self._key(eng)] = rung
        return plan


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class SorxAubiSolver(PSAStarSolver):
    game_id = "puzzlescript_sorx_aubi"
    game_name = GAME_NAME
    expert_cls = SorxAubiExpert

    #: `games/ps:sorx_aubi/ps:sorx_aubi.py` is a plain passthrough (it
    #: constructs the adapter and nothing else), so there is nothing to gain by
    #: routing through it -- but it IS what a live agent is handed, so if that
    #: wrapper ever grows a patch this must be set to ``"ps:sorx_aubi"``.
    game_module_id = ""

    #: The four the ladder cannot answer. Not a budget problem -- see the module
    #: docstring and `_plans`, which re-derives them with the full ladder and
    #: prints what happened, so this list is one line to shorten the day a macro
    #: rung exists.
    skip_levels = frozenset({5, 10, 12, 14})

    #: Unused: the expert plans on a native model rather than searching the
    #: engine under a heuristic. Left at the base values so nothing reads a lie
    #: off them.
    weight = 1
    node_cap = 400_000

    #: The longest shipped plan is 33 presses (level 11); the rest is room for a
    #: re-plan after the exploration prefix. Stays under the adapter's own
    #: 200-step per-level budget, which `set_level` resets before the plan
    #: starts anyway.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it fills the disk cache in one pass and makes the
        startup cost visible as startup."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

#: The author's own level titles, in shipped order (the `message` lines between
#: them are not levels). Level "7" is missing from his numbering, not from here.
LEVEL_NAMES = ("Aubi", "Frozen", "Sorx", "The Shore", "The Switch", "Soldiers",
               "Shift", "There and Back Again", "The Prison", "Upsilon",
               "The Quest", "The Cries", "The Unpredicted Problem",
               "The Failed Sequel", "Dear Fibonacci")


def _new():
    solver = SorxAubiSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _plans(include_skipped: bool = False) -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the ladder and the tie labelling at
    once.

    With ``--include-skipped`` the four levels in ``skip_levels`` are re-derived
    here too, with the same ladder and the same budgets, so that list stays a
    measurement rather than a claim. It is off by default because each of those
    four spends its full A* budget and then its full beam budget before
    reporting the failure -- about ten minutes for the four of them."""
    _solver, game, expert, solvable = _new()
    eng = game._engine
    total = ties = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        skipped = level in SorxAubiSolver.skip_levels
        if skipped and not include_skipped:
            continue
        if skipped:
            plan, rung = board.solve(state, astar_cap=expert.astar_cap,
                                     field_cap=expert.field_cap,
                                     beam_width=expert.beam_width,
                                     beam_depth=expert.beam_depth)
        else:
            plan = expert.plan(eng, level)
            rung = expert.rungs.get(expert._key(eng), "cached")
        name = LEVEL_NAMES[level] if level < len(LEVEL_NAMES) else "?"
        shape = f"{eng.height:2d}x{eng.width:2d}"
        pieces = (f"{len(state[0])}P {len(state[1])}b {len(state[2])}r "
                  f"{len(board.sorxx)}x {len(state[3])}i")
        if plan is None:
            print(f"  L{level:2d} {name:24s} {shape} {pieces:22s} "
                  f"UNSOLVED ({rung})"
                  + ("   [skip_levels]" if skipped else ""))
            bad += not skipped
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
        print(f"  L{level:2d} {name:24s} {shape} {pieces:22s} "
              f"{len(plan):3d} presses  win={won}  rung={rung:5s}  "
              f"({room})  {tie_steps:3d} tie step(s)")
    print(f"  solvable levels: {len(solvable)}/{game.n_levels} "
          f"(skipping {sorted(SorxAubiSolver.skip_levels)})")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _brute(board, state, limit, cap: int = 400_000):
    """``(presses to a win or None, complete)``, searched fresh.

    Shares no code with `_Board.astar` or `_Board.field` beyond the mechanic
    itself: a plain forward BFS whose first win is shortest by construction.

    ``complete`` says whether a ``None`` is a PROOF -- it is True only when the
    sweep ran out of states rather than out of depth or out of node budget, i.e.
    the reachable component is closed and win-free. `_selfcheck`'s recovery pass
    needs that distinction: a refusal matched against an inconclusive sweep is
    not evidence of anything."""
    if board.won(state):
        return 0, True
    seen = {state}
    queue = deque([(state, 0)])
    truncated = False
    while queue:
        cur, d = queue.popleft()
        if d >= limit:
            truncated = True
            continue
        for di in range(4):
            nxt = board.step(cur, di)
            if nxt == cur or nxt in seen:
                continue
            if board.won(nxt):
                return d + 1, True
            seen.add(nxt)
            queue.append((nxt, d + 1))
        if len(seen) > cap:
            return None, False
    return None, not truncated


def _selfcheck(walk_presses: int = 300, samples: int = 40,
               passes: "tuple[int, ...]" = (1, 2, 3, 4),
               verbose: bool = True) -> int:
    """Four things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's.** A seeded random walk on every
       level -- INCLUDING the four skipped ones, which are the ones with the
       most crates -- comparing `_Board.step`'s board against the engine's grid
       after EVERY press, the presses that do nothing included. This is the pass
       that matters most here: the model is a port of the interpreter's force
       and rigid-cancellation machinery, not a description of the mechanic, and
       the only reason to believe the port is that it is measured.
    2. **The plans are shortest.** A plain forward BFS to the first win, which
       is shortest by construction and shares no code with the field's backward
       half, must return the ladder's plan length on every level it can afford.
    3. **The tie sets are exact.** Every step's optimal set is re-derived by
       brute force -- a fresh depth-bounded BFS from each of the four
       successors -- and must match.
    4. **Recovery is answered from ANY board.** From ``samples`` states reached
       by seeded random presses, the ladder must either return a plan that WINS
       when replayed through the interpreter, or refuse on a board an
       independent forward BFS also finds no win from. A refusal that is merely
       a give-up would be a silent bug: `record_level` reads it as "reset".

    It takes the better part of an hour, and pass 3 is most of it: the tie sets
    are re-derived EXHAUSTIVELY -- every step of every level, a bounded BFS from
    each of the four successors -- rather than sampled, and levels 8, 11 and 13
    each cost tens of millions of model steps at that. Pass 4 is the other
    half: proving a board DEAD means exhausting its reachable space, twice
    over (the expert's A* and the independent BFS), and levels 4 and 6 have
    enough dark-blue crates to make that minutes a sample. None of it is on
    the generation path, and ``--pass N`` runs one of the four on its own.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0

    for level in range(game.n_levels if 1 in passes else 0):
        game.set_level(level)
        board, state = expert.read(eng)
        rng = random.Random(f"sorx_aubi:selfcheck:{level}")
        # Level 13's interpreter step is ~850ms, so it gets a shorter walk --
        # it is also the level with the fewest degrees of freedom per press.
        presses = walk_presses if eng.height * eng.width < 200 else 60
        drift = 0
        for _ in range(presses):
            di = rng.randrange(4)
            eng.step(DIRS[di])
            state = board.step(state, di)
            _b, live = expert.read(eng)
            if live != state:
                drift += 1
                break
        bad += drift
        if verbose:
            print(f"  L{level:2d}: model "
                  f"{'MATCHES' if not drift else 'DIVERGES from'} the "
                  f"interpreter over {presses} random presses")

    for level in range(game.n_levels if 2 in passes else 0):
        if level in SorxAubiSolver.skip_levels:
            continue
        game.set_level(level)
        board, state = expert.read(eng)
        plan = expert.plan(eng, level)
        shortest, _complete = _brute(board, state, len(plan))
        ok = plan is not None and shortest == len(plan)
        bad += not ok
        if verbose:
            print(f"  L{level:2d}: ladder {len(plan) if plan else None} presses "
                  f"vs BFS {shortest} -- {'SHORTEST' if ok else 'NOT SHORTEST'}")

    for level in range(game.n_levels if 3 in passes else 0):
        if level in SorxAubiSolver.skip_levels:
            continue
        game.set_level(level)
        board, state = expert.read(eng)
        plan = expert.plan(eng, level)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        mismatched = 0
        cur = state
        for i, direction in enumerate(plan):
            remaining = len(plan) - i
            truth = []
            for di in range(4):
                nxt = board.step(cur, di)
                if nxt == cur:
                    continue
                if _brute(board, nxt, remaining - 1)[0] == remaining - 1:
                    truth.append(DIRS[di])
            if sorted(truth) != sorted(sets[i]):
                mismatched += 1
            cur = board.step(cur, DIRS.index(direction))
        bad += mismatched
        if verbose:
            print(f"  L{level:2d}: {len(plan)} tie sets "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'}"
                  f" brute force")

    # Scoped-down budgets, and the beam rung switched off: a DEAD board is only
    # proved dead by an A* that exhausts the reachable space, and at the
    # shipping cap that is up to a minute a sample. A smaller cap turns some of
    # those proofs into "inconclusive", which this pass reports rather than
    # counts against the expert -- but only with the beam off, because
    # otherwise every capped board would then be handed to a 400x150 beam that
    # runs a Dijkstra per node and dominates the whole run.
    caps = (expert.astar_cap, expert.field_cap, expert.allow_beam,
            expert.tie_budget)
    expert.astar_cap, expert.field_cap = 120_000, 120_000
    expert.tie_budget = 120_000
    expert.allow_beam = False
    try:
        for level in range(game.n_levels if 4 in passes else 0):
            if level in SorxAubiSolver.skip_levels:
                continue
            rng = random.Random(f"sorx_aubi:recovery:{level}")
            wrong = dead = unclear = won_after = 0
            for _ in range(samples):
                game.set_level(level)
                for _ in range(rng.randrange(1, 20)):
                    eng.step(DIRS[rng.randrange(4)])
                    if eng.check_win():
                        break
                if eng.check_win():
                    continue             # the walk already won; nothing to plan
                board, state = expert.read(eng)
                plan = expert._search(eng)
                rung = expert.rungs.get(expert._key(eng))
                # A generous depth: a prefix of at most 19 presses cannot put a
                # board more than that further from a win than its start was,
                # and the longest shipped plan is 33. The node cap matches the
                # expert's, so both sides of the comparison give up at the same
                # budget and a board neither side can settle is reported as
                # inconclusive instead of being charged to either.
                truth, complete = _brute(board, state, 80, cap=120_000)
                if plan is None:
                    if truth is not None:
                        wrong += 1       # gave up on a board that CAN be won
                    elif complete:
                        dead += 1        # refusal matched by an independent proof
                    else:
                        unclear += 1     # neither side reached an answer
                    continue
                for direction in plan:
                    eng.step(direction)
                if not eng.check_win():
                    wrong += 1           # the plan does not replay to a win
                elif (rung in ("field", "astar", "astar-plain")
                        and truth is not None and truth != len(plan)):
                    wrong += 1           # claimed shortest and is not
                else:
                    won_after += 1
            bad += wrong
            if verbose:
                print(f"  L{level:2d}: {samples} random boards -- {won_after} "
                      f"re-planned to a WIN in the interpreter, {dead} refused "
                      f"and proved dead, {unclear} inconclusive, {wrong} WRONG")
    finally:
        (expert.astar_cap, expert.field_cap, expert.allow_beam,
         expert.tie_budget) = caps
    return bad


def _engine(node_cap: int = 200_000, verbose: bool = True) -> int:
    """The proof that owes nothing to the native model: enumerate the REAL
    interpreter.

    `StateGraph.build` walks every state reachable from the level start by
    pressing actual buttons on the engine and keying the resulting grid, then
    solves the graph exactly. It shares no line with `_Board` -- no ported force
    resolution, no ported rigid cancellation, no modelled crate typing -- so
    agreement on both the plan LENGTH and the per-step optimal SET is an
    independent derivation of everything this file claims.

    It is affordable on a MINORITY of levels here, and that is the honest
    limit: an engine press costs 0.4ms on the levels with no AubiiCrate and
    850ms on level 13, so a level whose reachable space is tens of thousands of
    states is hours of interpreter time. Levels whose enumeration exceeds
    ``node_cap`` are reported as not attempted, not as agreements."""
    from solvers.common.ps_astar import StateGraph                     # noqa: PLC0415

    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        if level in SorxAubiSolver.skip_levels:
            continue
        game.set_level(level)
        plan = expert.plan(eng, level)
        game.set_level(level)
        graph = StateGraph.build(eng, expert._key, list(DIRS), node_cap=node_cap)
        eplan = graph.plan(list(DIRS)) if graph is not None else None
        if eplan is None:
            print(f"  L{level:2d}: not attempted -- the interpreter enumeration "
                  f"exceeds node_cap {node_cap}")
            continue
        same_len = len(eplan) == len(plan)
        esets = [sorted(s) for s in eplan.optsets]
        fsets = [sorted(s) for s in plan.optsets]
        same_sets = esets == fsets
        bad += (not same_len) + (not same_sets)
        if verbose:
            print(f"  L{level:2d}: {len(graph.succ)} engine states, "
                  f"{len(eplan)} presses vs the ladder's {len(plan)} -- "
                  f"{'AGREE' if same_len else 'DISAGREE'}; tie sets "
                  f"{'AGREE' if same_sets else 'DISAGREE'}")
    return bad


#: Every cell stack the engine can produce. ``player + crate``,
#: ``crate + crate`` and ``wall + anything`` are absent because those all share
#: one collision layer. ``sorxxcrate + target`` and ``wall + target`` are drawn
#: anyway although no level can reach them (nothing moves a Sorxx or a Wall and
#: no cell starts with two legend characters), because they cost nothing and an
#: edited level might.
_CORNERS = ("aubicorner1", "aubicorner2", "aubicorner3", "aubicorner4")
_COMPOSITIONS = [
    (),
    ("target",),
    ("wall",), ("wall", "target"),
    ("normalplayer",), ("normalplayer", "target"),
    ("sorxcrate",), ("sorxcrate", "target"),
    ("aubicrate",), ("aubicrate", "target"),
    ("sorxxcrate",), ("sorxxcrate", "target"),
    ("aubiicrate",), ("aubiicrate", "target"),
    ("aubiicrate",) + _CORNERS, ("aubiicrate", "target") + _CORNERS,
    ("aubiicrate", "aubicorner1", "aubicorner2"),
    ("aubiicrate", "target", "aubicorner1", "aubicorner2"),
]

#: Collisions that are structural and accepted -- see the module docstring. A
#: target sitting under an AubiiCrate can only show through the four corners the
#: Aubii sprite leaves transparent, which is exactly where the AubiCorner seam
#: decals go, so the decals are what gets covered. They are redundant (they are
#: a function of which neighbours hold Aubii) and the target is not.
_ACCEPTED_CLASHES = {
    frozenset((("aubiicrate", "target"),
               ("aubiicrate", "target") + _CORNERS)),
    frozenset((("aubiicrate", "target"),
               ("aubiicrate", "target", "aubicorner1", "aubicorner2"))),
    frozenset((("aubiicrate", "target") + _CORNERS,
               ("aubiicrate", "target", "aubicorner1", "aubicorner2"))),
}


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the fifteen levels use.
    Whole 64x64 frames of uniform boards rather than one cell sliced out of a
    mixed board: `_render_frame` upscales and centre-pads, so slicing a cell by
    ``cell_px`` arithmetic reads the wrong pixels (the ps:explod lesson). Two
    uniform boards render identically iff their cells do.

    This is the pass that caught the bug the .txt header documents -- with the
    shipped sprites, ``normalplayer + target`` was identical to
    ``normalplayer`` at all six sizes, i.e. the win condition was unreadable.

    **Pass 2 -- the group.** The game is in `_FLIP_GAMES`, so a board is drawn
    at any of the eight turns and mirrors. That is only safe as long as no
    transform of one composition is another composition's art, which is what
    this checks: all eight transforms of each, against all others. It runs on
    SQUARE boards, because a rotation of a non-square board also moves the
    letterbox and every comparison would pass for the wrong reason."""
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert, _solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    def name(comp):
        return "+".join(comp) or "floor"

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, comp):
        eng.height, eng.width = h, w
        eng.grid = [[{idx[o] for o in comp} | {idx["background"]}
                     for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g)).copy()

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {c: shoot(h, w, c) for c in _COMPOSITIONS}
        clashes = [(a, b) for a, b in itertools.combinations(_COMPOSITIONS, 2)
                   if np.array_equal(shots[a], shots[b])]
        real = [(a, b) for a, b in clashes
                if frozenset((a, b)) not in _ACCEPTED_CLASHES]
        bad += len(real)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, {len(clashes) - len(real)} accepted decal "
                  f"clash(es), {'OK' if not real else 'CLASH'}")
        for a, b in real:
            print(f"      IDENTICAL: {name(a)}  ==  {name(b)}")

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    group = [(k, hf, vf) for k in range(4)
             for hf in (False, True) for vf in (False, True)]
    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {c: shoot(n, n, c) for c in _COMPOSITIONS}
        clashes = [(a, b, t)
                   for a, b in itertools.permutations(_COMPOSITIONS, 2)
                   for t in group
                   if frozenset((a, b)) not in _ACCEPTED_CLASHES
                   and np.array_equal(transform(shots[a], *t), shots[b])]
        invariant = [name(c) for c in _COMPOSITIONS
                     if all(np.array_equal(transform(shots[c], *t), shots[c])
                            for t in group)]
        bad += len(clashes)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(clashes)} composition(s) "
                  f"whose transform is another's art; "
                  f"{len(invariant)}/{len(_COMPOSITIONS)} group-invariant")
        for a, b, t in clashes:
            print(f"      {name(a)} under {t} == {name(b)}")
    return bad


def _symmetry(walk_presses: int = 40, equivariance_presses: int = 400,
              verbose: bool = True) -> int:
    """Two measurements, and only the first of them has to pass.

    **The rotation contract.** Every level's plan and a seeded random walk
    (which presses the unbound ACTION key as well) replayed through the ADAPTER
    at every presentation it draws, requiring every frame to be the exact
    transform of the unaugmented one. This is the mandatory augmentation and a
    failure here is a bug.

    **Whether the game's own dynamics are equivariant** under the square's
    symmetry group, which is the question `PuzzleScriptAdapter._FLIP_GAMES`
    turns on. The model is driven on a TRANSFORMED copy of each level with the
    correspondingly transformed press and compared against the transform of the
    untransformed result. They are NOT equivariant and are not expected to be --
    which cells of a partly-blocked rigid clump move is decided by the
    interpreter's chirality-dependent scan order -- so what this counts is the
    GOBBLE RUSH question instead: of the (state, press) pairs whose outcome any
    transform changes, how many does a mirror change that no rotation does?
    Measured zero out of 55 over 6000 pairs, which is what puts the game in
    ``_FLIP_GAMES``: the mirrors add no inconsistency the mandatory rotation is
    not already adding."""
    solver, game, expert, _solvable = _new()
    plans = {}
    for level in range(game.n_levels):
        if level in SorxAubiSolver.skip_levels:
            continue
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

    # Sixteen presentations (rotation x hflip x vflip). The loop stops the
    # moment it has drawn them all, because level 13 costs 850ms an engine press
    # and a redundant presentation is a minute of interpreter time.
    # The presentation is drawn per (seed, LEVEL), so a level is skipped once it
    # has been driven at a presentation -- without that, every extra seed
    # re-drives all eleven levels for whichever presentations it happens to
    # repeat, and level 13 alone costs 850ms an engine press.
    ref, seen, done, bad = {}, set(), {}, 0
    for seed in range(400):
        if all(len(done.get(lv, ())) == 16 for lv in plans):
            break
        g = solver.make_game(seed)
        for level in sorted(plans):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            shown = done.setdefault(level, set())
            if k in shown or (k != (0, False, False)
                              and (level, "plan") not in ref):
                continue          # already measured, or no reference drawn yet
            shown.add(k)
            rng = random.Random(f"sorx_aubi:symmetry:{level}")
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
                    ref[key] = frames
                    continue
                if any(not np.array_equal(transform(a, *k), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"    seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
        print(f"  every level replayed at "
              f"{min(len(v) for v in done.values())}-"
              f"{max(len(v) for v in done.values())} of them")

    # -- the equivariance measurement (reported, not required) ---------------
    _solver, game2, expert2, _s2 = _new()
    eng = game2._engine
    group = [(k, hf, vf) for k in range(4)
             for hf in (False, True) for vf in (False, True)]
    broken = tested = 0
    pairs = rot_pairs = flip_only_pairs = 0
    for level in range(game2.n_levels):
        game2.set_level(level)
        board, start = expert2.read(eng)
        cache = {t: _transform_board(board, t) for t in group[1:]}
        rng = random.Random(f"sorx_aubi:equivariance:{level}")
        state = start
        for _ in range(equivariance_presses):
            di = rng.randrange(4)
            by_rotation = by_flip = False
            for t in group[1:]:
                tested += 1
                if not _equivariant(board, state, di, t, cache[t]):
                    broken += 1
                    if t[1] or t[2]:
                        by_flip = True
                    else:
                        by_rotation = True
            pairs += by_rotation or by_flip
            rot_pairs += by_rotation
            # The Gobble Rush criterion, exactly: a (state, press) whose
            # outcome a MIRROR changes but no rotation does is inconsistency
            # the mandatory augmentation does not already carry.
            flip_only_pairs += by_flip and not by_rotation
            state = board.step(state, di)
    if verbose:
        print(f"  dynamics equivariance: {broken} of {tested} "
              f"(state, press, transform) triples disagree; "
              f"{pairs} order-sensitive (state, press) pairs, {rot_pairs} of "
              f"them split by a ROTATION and {flip_only_pairs} ONLY by a mirror")
        if broken:
            print("    -> the dynamics are NOT symmetric: which cells of a "
                  "partly-blocked rigid clump move is decided by the "
                  "interpreter's chirality-dependent scan order")
        if flip_only_pairs:
            print(f"    -> {flip_only_pairs} pair(s) are inconsistency the "
                  f"MANDATORY rotation does not already carry, so the mirrors "
                  f"would add it; this game stays out of _FLIP_GAMES")
        elif broken:
            print("    -> every order-sensitive pair is already split by the "
                  "MANDATORY rotation, which is the Gobble Rush argument for "
                  "adding the mirrors anyway")
    return bad


def _transform_board(board, t):
    """``(transformed board, cell mapping, direction mapping)`` for one element
    of the square's symmetry group.

    The cell permutation comes from applying the transform to an INDEX grid --
    the same ``np.rot90`` / ``fliplr`` / ``flipud`` the adapter applies to a
    frame -- and the direction permutation is then READ OFF that permutation
    rather than written down a second time: a neighbour one cell ``d`` away must
    land one cell ``d'`` away in the new grid, and that is what ``d'`` means."""
    k, hflip, vflip = t
    h, w = board.h, board.w
    grid = np.arange(h * w, dtype=np.int64).reshape(h, w)
    if k:
        grid = np.rot90(grid, k=k)
    if hflip:
        grid = np.fliplr(grid)
    if vflip:
        grid = np.flipud(grid)
    th, tw = grid.shape
    mapping = np.empty(h * w, dtype=np.int64)
    for new_cell, old_cell in enumerate(grid.reshape(-1)):
        mapping[old_cell] = new_cell

    dir_map = {}
    for di in range(4):
        for cell in range(h * w):
            nxt = board.edge[cell][di]
            if nxt < 0:
                continue
            r0, c0 = divmod(int(mapping[cell]), tw)
            r1, c1 = divmod(int(mapping[nxt]), tw)
            dir_map[di] = DIRS.index(
                {(-1, 0): "up", (1, 0): "down",
                 (0, -1): "left", (0, 1): "right"}[(r1 - r0, c1 - c0)])
            break

    def conv(cells):
        return [int(mapping[x]) for x in cells]

    tboard = _Board(th, tw, conv(board.wall), conv(board.sorxx),
                    conv(board.targets))
    return tboard, conv, dir_map


def _equivariant(board, state, di, t, prepared=None) -> bool:
    """Does ``transform(step(s, d)) == step(transform(s), transform(d))``?

    Used only by `_symmetry`'s reported measurement of whether this game's
    dynamics are symmetric under the square's group -- the question
    `PuzzleScriptAdapter._FLIP_GAMES` turns on. ``prepared`` is a
    `_transform_board` result reused across calls."""
    tboard, conv, dir_map = prepared or _transform_board(board, t)

    def move(s):
        return (tuple(sorted(conv(s[0]))), frozenset(conv(s[1])),
                frozenset(conv(s[2])), frozenset(conv(s[3])))

    return tboard.step(move(state), dir_map[di]) == move(board.step(state, di))


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans(include_skipped="--include-skipped" in sys.argv))
    if "--selfcheck" in sys.argv:
        chosen = ((int(sys.argv[sys.argv.index("--pass") + 1]),)
                  if "--pass" in sys.argv else (1, 2, 3, 4))
        violations = _selfcheck(passes=chosen)
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--engine" in sys.argv:
        violations = _engine()
        print(f"engine enumeration: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(SorxAubiSolver.main())
