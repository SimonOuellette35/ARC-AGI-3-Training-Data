"""Generate Phase-1 training data for the PuzzleScript game ps:winter
("Winter", part four of a series; the player is Winter itself).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a native model of the
slide, the searches built on it, the optimal-action sets, and the reports that
certify all of it against the interpreter.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_winter",
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

The game
--------
Winter cannot step: one press sends him sliding until something stops him, and
every square he crosses turns from UNLIT (orange) to LIT (pale blue). The door
opens only when there is no unlit square left anywhere on the board, and walking
into it wins. So a level is a COVERING WALK on a slide graph, not a maze.

Everything below was measured against the interpreter, not read off the .txt:

* **One press is a whole SLIDE.** ``[left winter moved] -> [lef winter] again``
  re-arms the direction on the next ``again`` tick, so the run finishes inside a
  single ``eng.step``, one cell per tick, and the branching factor is four.
* **He paints the cell he leaves AND the cell he is on** (``[> winter floor no
  moved | ...] -> [nlit | > winter moved]`` plus ``late [winter it] -> [winter
  nlit]``), so a slide lights its whole corridor. Lighting is permanent; nothing
  in the game ever unlights a square.
* **WATER stops him dead and freezes** (``[winter water uq] -> [winter ice]``
  drops the direction marker as it converts). So a water cell can only be lit by
  ENDING a press on it -- one per press -- and after that it is ordinary ice he
  slides straight over. `_Board` never carries an ice set: a frozen cell is
  exactly a lit water cell, because there is no other way to light one.
* **A TREE is bumped, never pushed.** ``[> player | tree floor] -> [player |
  dead nlit]`` runs after the push rule and overrides it: the tree goes dormant
  IN PLACE, its cell is lit, and the slide stops. Dormant trees still block
  (``dead`` is in the ``flow`` group), so a tree cell is lit exactly when it is
  dead -- again no extra state. One tree per press, and a press that bumps a
  tree cannot also freeze water, which is what makes the heuristic below add up.
* **Crates, mirrors and diagonals ARE pushed** (``[> winter floor no moved |
  cre | no flow no door no block] -> [nlit > winter | > cre |]``), a whole run
  of them at once, and the press stops there: winter ends on the cell the front
  piece left, which lights it.
* **A MIRROR bends the slide.** ``XYmir`` joins sides X and Y -- enter through
  one, leave through the other, lighting the mirror's own cell on the way, and a
  ``startloop`` chains it through a run of mirrors. Hit a mirror on either of
  its other two sides, or bend into a blocked cell, and the bend is cancelled;
  what happens then is the ordinary PUSH.
* **BUNNIES flee.** ``[winter | bunny] -> [winter | > bunny]`` has no direction
  prefix, so it fires in all four and every bunny orthogonally adjacent to
  winter is shoved away, once per tick of his slide, in runs. A bunny shoved into
  open water DROWNS: the water becomes a ``lose`` object, and the next time
  winter moves ``[> winter][lose] -> restart`` throws the level away. The model
  carries the drowned cell (it stops being water without becoming ice) and the
  searches never expand such a state.
* **A press into a wall is a pure no-op** -- no ``moved``, so no ``again``, so
  no state change at all. There is no wait move and no death other than
  drowning a bunny.
* ACTION5 is read by no rule, so `WinterExpert.directions` is the four moves.

The one board shape the game has no rule for: level 9's water ring, once frozen,
is a closed MIRROR ORBIT. The four corner mirrors bend the slide round and round
and nothing ever stops it, so the run eats `PSEngine.step`'s 50-iteration
``again`` budget and returns with a live ``lef``/``righ``/``uq``/``dow`` marker
on the board -- which then hijacks the NEXT press, whatever it is (the
ps:slidyyyyyyy trap, here caused by the level rather than by an animation).
`_Board.step` reports such a press as ``truncated`` and no search ever takes one,
so no recorded episode can contain one; the exploration prefix may, and its
closing RESET clears it.

Level indices here and in every report are 0-BASED.

Why a native model, and not the shared engine-blackbox A*
---------------------------------------------------------
Measured first, per the ps:entrepotphage_demake lesson: the interpreter runs at
220-600 ``eng.step``/s here, and the state carries a LIT MASK over up to 67
squares, so the reachable component is far too big to enumerate the way
ps:icecrates or ps:roller_boi do. `_Board.step` is the same mechanic in ~120
lines of bitmask Python at **1.7 us** -- 200x faster -- and ``--fuzz`` holds it
to the interpreter over 30k random transitions on all nineteen playable levels.

The search ladder
-----------------
Per level, first rung that returns (all deterministic, so a re-derived plan is
byte-identical):

1. **A* with an ADMISSIBLE heuristic** -- the plan it returns is provably
   SHORTEST, and the optimal-action sets are then measured exactly.
2. **A* at weight 2, then 3** -- still a genuine win, no longer proved shortest.
3. **Beam search at width 3k / 30k / 100k**, ranked by the same heuristic and
   pruned by `make_alive`.
4. **Nested Monte-Carlo search** (level 2, uniform rollouts, 100 restarts per
   seed, seeds 4/0/1). Level 12 is the reason it exists: 47 of its 48 squares
   are water, so every press has to stop on a fresh one and the level is a
   Hamiltonian path through a jump graph -- a shape on which every state at a
   given depth has the same heuristic value, the beam has nothing left to rank
   with, and it strands its own tail four squares from the end.

Anything below rung 1 wanders -- it is steered by "light the next thing", which
takes detours a later press undoes -- so every non-optimal plan then goes through
`shorten`, which deletes the longest block of presses the model still wins
without.

The heuristic adds the two things a press can only do ONE of: freeze a water
cell (it has to stop there) and bump a tree (it has to stop there too). That is
already tight on the water-heavy levels -- level 12 is 47 water cells in 48
squares, so its lower bound is 47 -- and it is maxed against a rook bound (a
press lights cells on ONE line, so at most one cell of any pairwise
row-and-column-distinct set) and a corridor-length bound.

`make_alive` is the prune that makes the covering walks tractable. It is an
over-approximate reachability fixpoint: a False is a PROOF the state is dead, a
True proves nothing. It kills a state when an unlit square is no longer on any
slide from any reachable rest cell, when a water cell can no longer be stopped
on, and -- the one that matters most -- when no neighbour of the door is a
reachable rest cell any more. Level 18 dies exactly there: its door sits at the
end of a row whose only two rest cells are its two water squares, so freezing
both before the endgame makes the door unreachable while the board still looks
perfectly winnable.

What it solves
--------------
19 of the 20 levels, 381 presses, 12 of them PROVED SHORTEST (the weight-1 rung);
level 6 is Autumn's and falls to any single press. The lower bound in brackets is
`make_h` at the level start, so a plan that matches it is optimal by
construction:

    lvl  board  squares  presses            lvl  board  squares  presses
      0   4x 5      11    7  (lb  3, opt)    10   5x 7      26   15  (lb  7, opt)
      1   4x 7      17   11  (lb  4, opt)    11   5x 7      26   13  (lb  5, opt)
      2   4x 6      13   11  (lb  4, opt)    12   7x10      48   53  (lb 47)
      3   5x 7      26   14  (lb  5, opt)    13   8x10      65   26  (lb 19)
      4   4x 7      21   13  (lb  4, opt)    14   7x11      60   44  (lb 40)
      5   4x 6      17    9  (lb  2, opt)    15   5x12      51   20  (lb  3, opt)
      6   7x 7       -    1  (Autumn)        16  11x 8      67   34  (lb 22)
      7   7x 7      22   18  (lb  5, opt)    17   7x 7      24   36  (lb  5, opt)
      8   5x 5      14    9  (lb  3, opt)    18  11x 6      50   27  (lb 21)
      9   7x 9      50   21  (lb 20)         19   7x11      62   NOT SOLVED

Level 12 is the only one that needs the NMCS rung, and it is the only one that
costs real time to plan (about half an hour cold), which is why
``data/winter_plans.json`` exists and why ``--plans`` is worth running once
before a generation sweep.

Level 19 is `WinterSolver.skip_levels`, and the reason is there: its door can
only be entered from a square nothing can stop on unless the level's single
bunny is standing in the column, so it is won by herding the bunny rather than by
covering, and no rung here has a reason to want that.

Optimal-action sets
-------------------
For a level solved at weight 1 the sets are EXACT and measured: at plan step
``i`` with ``L - i`` presses left, press ``p`` is optimal iff ``d*(succ) ==
L - i - 1``, decided by a bounded A* that shares nothing with the plan except
the heuristic. For a level that needed a weight or a beam, the shortest distance
is not known, so the set is the singleton the expert actually played -- never
None (the always-emit-optimal-targets rule). ``--verify`` re-derives every set.

The .txt fixes
--------------
Three, all in ``data/puzzlescript_games/Winter.txt``; the first two are about the
trees:

  * **The five tree objects were five COLOURS, chosen at random.** ``[rtree] ->
    [random tree]`` runs at level start and the adapter's ``random`` RHS draws
    from the process-wide ``random`` module, so levels 1, 4, 14 and 18 rendered
    their trees a different colour on every reset -- frames that no seed
    reproduces. All five are now ``lightgreen``, which makes the draw invisible
    and the game deterministic. (They are five copies of one object on purpose;
    ``--audit`` declares the collision.)
  * **``treed``/``treee`` were the FLOOR's colour.** ``orange`` and ``brown``
    both quantize to ARC 12, which is exactly ``it``, the unlit floor -- and the
    tree sprite is transparent only at its four corners, so a live tree standing
    on unlit floor was a uniform block of 12, i.e. invisible. Same repaint fixes
    it: 14 against 12 and 10.
  * **``nodoor`` -- the OPEN door -- was ``transparent``.** The door is replaced
    by ``nodoor`` + ``nlit`` the moment nothing is unlit, so the one square the
    whole game is heading for became indistinguishable from ordinary lit floor
    exactly when it started to matter, and the winning frame was a player
    standing on nothing in particular. It is now a white frame drawn on the
    cell's border, which the Player's four transparent corners let through, so
    both "the door has opened" and "Winter is in it" read.

Deliberately NOT changed: ``ice`` and ``nlit`` are both ARC 10. Ice is lit by
construction (the only way to freeze a cell is to end a press on it, which lights
it) and behaves exactly like lit floor, so the merge hides no state.

Augmentation
------------
This game's engine state after reset is identical for every seed (levels are
fixed ASCII maps), so the only per-(seed, level) variable is the frame rotation
(``rotation_k`` in {0,1,2,3}) with its directional action remap. It is exact
here even though two of the sprite families encode a direction, because both
families are CLOSED under a quarter turn: rotating ``nemir``'s art gives
``semir``'s art, byte for byte, and that is the mirror the rotated mechanic
needs; the same holds for the four bunny facings. ``--audit`` asserts it rather
than trusting the reading. The game is NOT in ``_FLIP_GAMES``: the mirror-merge
rules (``right [> nemir | swmir] -> [|diag]``) are written per direction and a
chirality there would be invisible, so the flips are left unclaimed.

The expert plan is therefore seed-independent: solved once per level, cached in
``data/winter_plans.json``, and replayed per seed with that seed's remapped
screen actions.

Usage (run from the repo root):
    python solvers/generate_winter_training.py \
        --episodes 200 --out data/training_multi_level/winter

    python solvers/generate_winter_training.py --plans
    python solvers/generate_winter_training.py --verify
    python solvers/generate_winter_training.py --fuzz
    python solvers/generate_winter_training.py --audit
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

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan      # noqa: E402

GAME_NAME = "Winter"

#: Engine direction -> (dr, dc). The four moves are the whole action space.
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
#: Fixed iteration order, so a plan (and its tie sets) is reproducible.
_ORDER: tuple[str, ...] = ("up", "down", "left", "right")

#: ``XYmir`` joins sides X and Y: a slide entering through one side leaves
#: through the other. Read off the eight deflection rules with the adapter's own
#: relative-direction table (`PSEngine._get_dir_symbol_map`), then fuzz-checked.
_MIRROR: dict[str, dict[str, str]] = {
    "swmir": {"right": "down", "up": "left"},
    "nwmir": {"down": "left", "right": "up"},
    "nemir": {"left": "up", "down": "right"},
    "semir": {"up": "right", "left": "down"},
}

#: `PSEngine.step`'s ``max_again``. A slide that has not come to rest by then is
#: abandoned mid-run with a live direction marker on the board; see the module
#: docstring and `_Board.step`'s ``truncated``.
_MAX_TICKS = 50

WIN = "WIN"

#: Object classes `_Board` reads off the interpreter. ``tree`` covers the five
#: random-coloured tree objects AND ``dead``, their dormant form, because a tree
#: never moves and its cell blocks either way.
_CLASSES = ("block", "door", "nodoor", "crat", "tree", "dead", "bunny", "water",
            "ice", "nlit", "semir", "swmir", "nwmir", "nemir", "diags",
            "winter", "autumn", "lose")


def _popcount(x: int) -> int:
    return bin(x).count("1")


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """Winter's rule set as a state machine over

        ``(pos, lit, crates, bunnies, mirrors, diags, lose)``

    with every cell set a BITMASK over ``r * w + c`` and ``mirrors`` a sorted
    tuple of ``(cell, kind)``. Walls, the door, the water cells and the tree
    cells are static -- no rule creates or destroys any of them -- so they live
    on the board rather than in the state.

    Two pieces of state that look like they are missing are DERIVED, and that is
    the whole reason the model is small enough to search:

      * **ice** = ``waters & lit`` (minus drowned cells). A water cell can only
        be lit by ending a press on it, which is also the only thing that
        freezes it, so "frozen" and "lit" are the same bit.
      * **dormant trees** = ``trees & lit``. A tree is only ever bumped, which
        lights its cell and kills it in the same rule.
    """

    def __init__(self, h, w, walls, door, trees, waters):
        self.h, self.w = h, w
        self.n = h * w
        self.walls = walls
        self.door = door                     # cell index, or -1 (level 6)
        self.trees = trees
        self.waters = waters
        floor = 0
        for i in range(self.n):
            if not (walls >> i) & 1 and i != door:
                floor |= 1 << i
        #: Every cell the win condition wants lit: everything that is not a wall
        #: and not the door itself carries a floor object (``[winter][no floor
        #: no door no nodoor no block no flow] -> [winter][it]`` seeds one under
        #: water, crates, bunnies, mirrors and trees alike).
        self.floor = floor
        self.nbr = {}
        for d, (dr, dc) in _DELTA.items():
            row = [-1] * self.n
            for i in range(self.n):
                r, c = divmod(i, w)
                r2, c2 = r + dr, c + dc
                if 0 <= r2 < h and 0 <= c2 < w:
                    row[i] = r2 * w + c2
            self.nbr[d] = row
        self.rays = {d: [tuple(self._ray(i, d)) for i in range(self.n)]
                     for d in _ORDER}

    def _ray(self, i, d):
        out, q = [], self.nbr[d][i]
        while q >= 0:
            out.append(q)
            q = self.nbr[d][q]
        return out

    def rc(self, i):
        return divmod(i, self.w)

    # -- dynamics ------------------------------------------------------------
    def step(self, st, d):
        """One press. Returns `WIN`, or ``(state, truncated)``.

        The loop body is one ``again`` tick, in the interpreter's rule order:
        move if the next cell is free (freezing it, and stopping, if it is live
        water); else bend off a mirror; else bump a tree; else push the run of
        pushables ahead; else, at the door, win iff nothing is unlit. Bunnies
        are shoved at the END of every tick, off winter's post-move cell, which
        is where the rule sits in the file.
        """
        pos, lit, crates, bunnies, mirrors, diags, lose = st
        mirr = dict(mirrors)
        waters, trees, walls, door = self.waters, self.trees, self.walls, self.door
        nbr = self.nbr

        def blocked(cell):
            """`cra` / `flow` / `door`: winter may not enter."""
            if cell < 0:
                return True
            b = 1 << cell
            return (bool((walls | crates | bunnies | diags | trees) & b)
                    or cell == door or cell in mirr)

        def cre(cell):
            """crat / mirror / diag / LIVE tree -- the pushable classes."""
            if cell < 0:
                return False
            b = 1 << cell
            return (bool((crates | diags) & b) or cell in mirr
                    or bool(trees & b and not lit & b))

        def cra(cell):
            """Anything on the shared collision layer: blocks a push and a bunny."""
            if cell < 0:
                return True
            b = 1 << cell
            return (bool((walls | crates | bunnies | diags | trees) & b)
                    or cell == door or cell in mirr)

        def live_water(cell):
            b = 1 << cell
            return bool(waters & b) and not (lit | lose) & b

        def push(cell0):
            """``[> winter .. | cre | ..]`` plus its ``[> cre | cre | ..]``
            chain: shove the whole run one cell, or refuse."""
            nonlocal crates, diags, mirr
            chain, q = [], cell0
            while cre(q):
                chain.append(q)
                q = nbr[d][q]
            if not chain or cra(q):
                return False
            kinds = []
            for cell in chain:
                b = 1 << cell
                if crates & b:
                    kinds.append(("crat", cell))
                elif cell in mirr:
                    kinds.append((mirr[cell], cell))
                elif diags & b:
                    kinds.append(("diag", cell))
                else:
                    # A live tree inside a run. No shipped level can build one
                    # (a tree is only ever reached by a bump, which kills it in
                    # place) and the model has no state for a moved tree, so say
                    # so rather than quietly simulate a different game.
                    raise AssertionError(
                        "a tree was pushed; _Board cannot represent that")
            for kind, cell in kinds:
                if kind == "crat":
                    crates &= ~(1 << cell)
                elif kind == "diag":
                    diags &= ~(1 << cell)
                else:
                    del mirr[cell]
            for kind, cell in kinds:
                nc = nbr[d][cell]
                if kind == "crat":
                    crates |= 1 << nc
                elif kind == "diag":
                    diags |= 1 << nc
                else:
                    mirr[nc] = kind
            return True

        def shove_bunnies():
            """Every bunny orthogonally adjacent to winter, and the run behind
            it, moves one cell away. The front one DROWNS facing open water --
            the cell becomes ``lose`` (no longer water, and never lightable) and
            the level is finished; it REFUSES to step onto ice."""
            nonlocal bunnies, lose
            if not bunnies:
                return
            for bdir in _ORDER:
                b0 = nbr[bdir][pos]
                if b0 < 0 or not (bunnies >> b0) & 1:
                    continue
                chain, q = [], b0
                while q >= 0 and (bunnies >> q) & 1:
                    chain.append(q)
                    q = nbr[bdir][q]
                if q >= 0 and live_water(q):
                    bunnies &= ~(1 << chain[-1])
                    lose |= 1 << q
                    chain.pop()
                    if not chain:
                        continue
                    q = nbr[bdir][chain[-1]]
                if q < 0 or cra(q) or ((waters >> q) & 1 and (lit >> q) & 1
                                       and not (lose >> q) & 1):
                    continue
                for cell in chain:
                    bunnies &= ~(1 << cell)
                for cell in chain:
                    bunnies |= 1 << nbr[bdir][cell]

        lit |= 1 << pos
        truncated = True
        for _tick in range(_MAX_TICKS):
            tgt = nbr[d][pos]
            moved = False
            if not blocked(tgt):
                froze = live_water(tgt)
                lit |= (1 << pos) | (1 << tgt)
                pos = tgt
                moved = not froze
            else:
                kind = mirr.get(tgt) if tgt >= 0 else None
                if kind is not None and d in _MIRROR[kind]:
                    path, nd, m = [tgt], _MIRROR[kind][d], tgt
                    while True:
                        nxt = nbr[nd][m]
                        k2 = mirr.get(nxt) if nxt >= 0 else None
                        if k2 is not None and nd in _MIRROR[k2]:
                            path.append(nxt)
                            nd = _MIRROR[k2][nd]
                            m = nxt
                            continue
                        break
                    if not blocked(nxt):
                        froze = live_water(nxt)
                        lit |= 1 << pos
                        for cell in path:
                            lit |= 1 << cell
                        pos = nxt
                        lit |= 1 << pos
                        d = nd
                        moved = not froze
                    elif push(tgt):
                        # The bend is cancelled and the ordinary push rule, which
                        # runs after it, gets its turn on the same mirror.
                        lit |= (1 << pos) | (1 << tgt)
                        pos = tgt
                elif tgt >= 0 and (trees >> tgt) & 1 and not (lit >> tgt) & 1:
                    lit |= 1 << tgt                  # the tree goes dormant
                elif cre(tgt):
                    if push(tgt):
                        lit |= (1 << pos) | (1 << tgt)
                        pos = tgt
                elif tgt == door and not (self.floor & ~lit):
                    return WIN
            shove_bunnies()
            if not moved:
                truncated = False
                break
        return (pos, lit, crates, bunnies, tuple(sorted(mirr.items())), diags,
                lose), truncated

    # -- construction --------------------------------------------------------
    @classmethod
    def read(cls, eng, ids) -> "tuple[_Board, tuple]":
        """``(board, state)`` off the interpreter's current grid."""
        w = eng.width
        walls = trees = waters = crates = bunnies = diags = lit = lose = 0
        door = pos = -1
        mirr = {}
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i, b = r * w + c, 1 << (r * w + c)
                if cell & ids["block"]:
                    walls |= b
                if cell & ids["door"] or cell & ids["nodoor"]:
                    door = i
                if cell & ids["tree"] or cell & ids["dead"]:
                    trees |= b
                if cell & ids["water"] or cell & ids["ice"]:
                    waters |= b
                if cell & ids["crat"]:
                    crates |= b
                if cell & ids["bunny"]:
                    bunnies |= b
                if cell & ids["diags"]:
                    diags |= b
                if cell & ids["nlit"]:
                    lit |= b
                if cell & ids["lose"]:
                    lose |= b
                if cell & ids["winter"]:
                    pos = i
                for nm in _MIRROR:
                    if cell & ids[nm]:
                        mirr[i] = nm
        board = cls(eng.height, w, walls, door, trees, waters)
        state = (pos, lit, crates, bunnies, tuple(sorted(mirr.items())), diags,
                 lose)
        return board, state


# ---------------------------------------------------------------------------
# Heuristic, liveness, searches
# ---------------------------------------------------------------------------

def make_h(bd: _Board):
    """An ADMISSIBLE lower bound on the presses left.

    Three bounds, maxed:

      * **stops**: a press comes to rest on exactly one cell, and both freezing
        a water cell and bumping a tree require the press to end there, so the
        unlit water cells and the live trees each need a press of their own and
        the two counts ADD.
      * **corridor**: the cells a press lights lie on ONE straight run, so at
        least ``unlit / longest-run`` presses are needed. (A mirror bend lights
        an L, so the run length is relaxed to ``h + w`` on a level that has
        mirrors.)
      * **rook**: for the same reason, a set of unlit cells no two of which
        share a row or a column needs one press each -- the maximum such set is
        a bipartite rows-to-columns matching. Skipped on mirror levels.
    """
    longest = 1
    for r in range(bd.h):
        run = 0
        for c in range(bd.w):
            run = run + 1 if (bd.floor >> (r * bd.w + c)) & 1 else 0
            longest = max(longest, run)
    for c in range(bd.w):
        run = 0
        for r in range(bd.h):
            run = run + 1 if (bd.floor >> (r * bd.w + c)) & 1 else 0
            longest = max(longest, run)

    def h(st, mirrored=False):
        lit = st[1]
        unlit = bd.floor & ~lit
        if not unlit:
            return 1                     # the door still has to be walked into
        stops = (_popcount(bd.waters & ~lit & ~st[6])
                 + _popcount(bd.trees & ~lit))
        span = (bd.h + bd.w) if mirrored else longest
        best = max(stops, -(-_popcount(unlit) // span))
        if not mirrored:
            best = max(best, _rook(bd, unlit))
        return best
    return h


def _rook(bd: _Board, unlit: int) -> int:
    """Maximum set of cells no two of which share a row or a column, by
    Kuhn's matching over (row -> column)."""
    adj: dict[int, list[int]] = {}
    u = unlit
    while u:
        b = u & -u
        i = b.bit_length() - 1
        u ^= b
        r, c = divmod(i, bd.w)
        adj.setdefault(r, []).append(c)
    match: dict[int, int] = {}

    def try_(r, seen):
        for c in adj[r]:
            if c in seen:
                continue
            seen.add(c)
            if c not in match or try_(match[c], seen):
                match[c] = r
                return True
        return False

    return sum(try_(r, set()) for r in adj)


def make_alive(bd: _Board, door_check: bool = True):
    """``alive(state) -> bool``: an OVER-approximation, so False PROVES the
    state can never be won and True proves nothing.

    Rest positions are a fixpoint over the current board with two deliberate
    relaxations:

      * a live water cell may be either stopped on OR slid over -- sliding over
        it only becomes possible once it is frozen, which needs a stop first, so
        allowing both covers every future board;
      * crates and bunnies do not block, because they can still be shoved aside.

    Three things then kill a state: an unlit cell that no slide from any
    reachable rest cell passes over (or, for a tree, bumps); a water cell or a
    tree that can no longer be stopped on or bumped; and a door none of whose
    neighbours is a reachable rest cell.

    ``door_check`` is OFF for a level holding bunnies or crates. Those can be
    shoved into a cell and BECOME the obstacle that CREATES a rest position --
    level 19's whole solution is herding its bunny into the column beside the
    door so that a slide along it stops on the door's row -- and the relaxation
    above, which lets a piece move away but never in, cannot see that.

    A level with MIRRORS never reaches this: a bend lights an L, which the
    straight-ray fixpoint would miss, so `beam` and `nmcs` hand those levels a
    constant True instead.
    """
    rays = bd.rays
    door_nbrs = ([x for x in (bd.nbr[d][bd.door] for d in _ORDER)
                  if x >= 0 and (bd.floor >> x) & 1]
                 if (bd.door >= 0 and door_check) else [])
    static_hard = bd.walls | bd.trees | ((1 << bd.door) if bd.door >= 0 else 0)
    trees, floor, waters = bd.trees, bd.floor, bd.waters

    def alive(st):
        pos, lit, _crates, _bunnies, mirrors, diags, lose = st
        live = waters & ~lit & ~lose
        hard = static_hard | diags
        for cell, _k in mirrors:
            hard |= 1 << cell
        rest = passed = 1 << pos
        bump = 0
        stack = [pos]
        while stack:
            p = stack.pop()
            for d in _ORDER:
                prev = -1
                for q in rays[d][p]:
                    if (hard >> q) & 1:
                        if (trees >> q) & 1:
                            bump |= 1 << q
                        break
                    passed |= 1 << q
                    if (live >> q) & 1 and not (rest >> q) & 1:
                        rest |= 1 << q
                        stack.append(q)
                    prev = q
                if prev >= 0 and not (rest >> prev) & 1:
                    rest |= 1 << prev
                    stack.append(prev)
        if floor & ~lit & ~(passed | bump):
            return False
        if live & ~rest:
            return False
        if trees & ~lit & ~bump:
            return False
        if door_nbrs and not any((rest >> x) & 1 for x in door_nbrs):
            return False
        return True
    return alive


def astar(bd, st0, hf, weight=1, cap=300_000, bound=None):
    """Weighted A* over `_Board`. Returns the press list, or None.

    ``weight == 1`` with `make_h` admissible means the result is provably
    shortest. ``bound`` caps ``f``: a search that only has to decide "is there a
    plan of at most B presses" prunes everything above it, which is what
    `optsets_exact` uses.
    """
    mirrored = bool(st0[4])
    openq = [(weight * hf(st0, mirrored), 0, 0, st0)]
    best = {st0: 0}
    parent: dict = {}
    counter = 0
    while openq:
        _f, g, _c, st = heapq.heappop(openq)
        if best.get(st, 1 << 30) < g:
            continue
        for d in _ORDER:
            res = bd.step(st, d)
            if res is WIN:
                path = [d]
                cur = st
                while cur in parent:
                    cur, pd = parent[cur]
                    path.append(pd)
                return list(reversed(path))
            ns, trunc = res
            if trunc or ns[6] or ns == st:
                continue
            ng = g + 1
            if best.get(ns, 1 << 30) <= ng:
                continue
            f = ng + weight * hf(ns, mirrored)
            if bound is not None and f > bound:
                continue
            best[ns] = ng
            parent[ns] = (st, d)
            counter += 1
            heapq.heappush(openq, (f, ng, counter, ns))
            if len(best) > cap:
                return None
    return None


def door_path(bd, st, cap=20_000):
    """Shortest press sequence from a FULLY LIT state to the win, by plain BFS.

    On such a state the board is frozen solid and the search is over winter's
    position alone, so it is small. None means the level is dead: everything is
    lit and the door can never be entered again."""
    seen = {st}
    q = deque([(st, [])])
    while q:
        cur, path = q.popleft()
        for d in _ORDER:
            res = bd.step(cur, d)
            if res is WIN:
                return path + [d]
            ns, trunc = res
            if trunc or ns[6] or ns in seen or len(seen) > cap:
                continue
            seen.add(ns)
            q.append((ns, path + [d]))
    return None


def make_door_reach(bd):
    """``door_reach(state) -> 0 if the door can still be walked into, else 1``,
    computed like `make_alive`'s fixpoint but with the crates and bunnies where
    they ACTUALLY are rather than relaxed away.

    It is a beam RANKING term, never a prune -- the pieces can move, so a
    non-zero is not a proof of anything. It exists for the levels whose door is
    reachable only once a piece has been parked as a STOPPER: level 19's door
    sits beside a column with no wall to stop against, and the level is won by
    herding a bunny into that column so a slide along it comes to rest on the
    door's row. Nothing in the coverage heuristic wants that bunny there, so
    without this term the beam never keeps the states that put it there.

    A plain yes/no would only pay off on the last press of the herd, which is far
    too late to steer a beam, so a state that cannot reach the door is graded by
    how far its nearest movable piece is from a STOPPER SLOT -- a cell that, if
    something stood in it, would let a slide stop on a door neighbour."""
    rays = bd.rays
    door_nbrs = ([x for x in (bd.nbr[d][bd.door] for d in _ORDER)
                  if x >= 0 and (bd.floor >> x) & 1] if bd.door >= 0 else [])
    slots = sorted({bd.nbr[d][x] for x in door_nbrs for d in _ORDER
                    if bd.nbr[d][x] >= 0 and (bd.floor >> bd.nbr[d][x]) & 1})
    static_hard = bd.walls | bd.trees | ((1 << bd.door) if bd.door >= 0 else 0)

    def door_reach(st):
        if not door_nbrs:
            return 0
        pos, lit, crates, bunnies, mirrors, diags, lose = st
        live = bd.waters & ~lit & ~lose
        hard = static_hard | diags | crates | bunnies
        for cell, _k in mirrors:
            hard |= 1 << cell
        rest = 1 << pos
        stack = [pos]
        while stack:
            p = stack.pop()
            for d in _ORDER:
                prev = -1
                for q in rays[d][p]:
                    if (hard >> q) & 1:
                        break
                    if (live >> q) & 1 and not (rest >> q) & 1:
                        rest |= 1 << q
                        stack.append(q)
                    prev = q
                if prev >= 0 and not (rest >> prev) & 1:
                    rest |= 1 << prev
                    stack.append(prev)
        if any((rest >> x) & 1 for x in door_nbrs):
            return 0
        movers = [i for i in range(bd.n) if ((crates | bunnies) >> i) & 1]
        if not movers or not slots:
            return 1
        return 1 + min(abs(i // bd.w - j // bd.w) + abs(i % bd.w - j % bd.w)
                       for i in movers for j in slots)
    return door_reach


def make_score(bd, hf):
    """Beam ranking key: the heuristic (which is nearly tight, so it stands in
    for depth), then whether the door is still walk-in-able with the pieces
    where they are (`make_door_reach`, only on the levels that have pieces),
    then WARNSDORFF over the cells that still need a press to themselves -- how
    many have no passable neighbour left to be approached from, then how many
    have exactly one. Serving the most constrained cells first is what stops a
    covering walk stranding its own tail."""
    nbr = bd.nbr
    reach = make_door_reach(bd)

    def score(st):
        _pos, lit, crates, bunnies, mirrors, diags, lose = st
        need = (bd.waters & ~lit & ~lose) | (bd.trees & ~lit)
        block = (bd.walls | bd.trees | crates | bunnies | diags
                 | (bd.waters & ~lit & ~lose))
        if bd.door >= 0:
            block |= 1 << bd.door
        for cell, _k in mirrors:
            block |= 1 << cell
        d0 = d1 = 0
        u = need
        while u:
            b = u & -u
            i = b.bit_length() - 1
            u ^= b
            k = 0
            for d in _ORDER:
                v = nbr[d][i]
                if v >= 0 and (bd.floor >> v) & 1 and not (block >> v) & 1:
                    k += 1
            if k == 0:
                d0 += 1
            elif k == 1:
                d1 += 1
        far = reach(st) if (crates or bunnies) else 0
        return (hf(st, bool(mirrors)), far, d0, d1, -_popcount(lit))
    return score


def beam(bd, st0, hf, width=3000, max_depth=250):
    """Width-capped breadth-first search ranked by `make_score` and pruned by
    `make_alive`. A candidate that is fully lit is finished with `door_path`
    (or dropped, if that returns None)."""
    sf = make_score(bd, hf)
    alive = (make_alive(bd, door_check=not (st0[2] or st0[3]))
             if not st0[4] else (lambda s: True))
    parents: dict = {st0: None}
    kept = {st0}

    def path_to(st):
        out = []
        while parents[st] is not None:
            st, d = parents[st][0], parents[st][1]
            out.append(d)
        return list(reversed(out))

    if not (bd.floor & ~st0[1]):
        return door_path(bd, st0)
    frontier = [st0]
    for _depth in range(max_depth):
        cand: dict = {}
        for st in frontier:
            for d in _ORDER:
                res = bd.step(st, d)
                if res is WIN:
                    return path_to(st) + [d]
                ns, trunc = res
                if trunc or ns[6] or ns == st or ns in kept or ns in cand:
                    continue
                if not (bd.floor & ~ns[1]):
                    tail = door_path(bd, ns)
                    if tail is None:
                        continue
                    parents[ns] = (st, d)
                    return path_to(ns) + tail
                cand[ns] = (st, d)
        if not cand:
            return None
        best = []
        for s in sorted(cand, key=sf):
            if alive(s):
                best.append(s)
                if len(best) >= width:
                    break
        if not best:
            return None
        for s in best:
            parents[s] = cand[s]
            kept.add(s)
        frontier = best
    return None


def nmcs(bd, st0, seed=0, restarts=150, level=2):
    """Nested Monte-Carlo search, level 2, with restarts. The last rung.

    A beam collapses on the water-only boards -- level 12 is 47 water cells in
    48 squares, so EVERY press has to come to rest on a fresh one and the whole
    level is a Hamiltonian path through a jump graph. Every state at a given
    depth then has the same heuristic value and the beam has nothing to rank
    with, so it commits early and strands its own tail. NMCS is the standard
    answer to exactly that shape: it decides each press by playing the rest of
    the level out from every option and keeping the best line found.

    The rollouts are UNIFORM and pruned by `make_alive`, which is what turns a
    rollout that ends four squares short into one that ends one short. They were
    also tried biased by Warnsdorff (step to the cell with the fewest unlit
    neighbours 70% of the time, the standard trick for a Hamiltonian path) and
    that is MEASURABLY WORSE here: eight biased seeds x 150-400 restarts never
    solved level 12, while the uniform rollouts solve it on seed 4, restart 65.
    The bias serves the board's corners early, and on this board that is exactly
    what strands the two squares of the door's own pocket.

    Deterministic: the RNG is seeded and the restart count is fixed, so the plan
    it returns is a function of ``(seed, restarts)`` alone and a re-derived plan
    cache is byte-identical.
    """
    alive = (make_alive(bd, door_check=not (st0[2] or st0[3]))
             if not st0[4] else (lambda s: True))
    rng = random.Random(seed)
    WINSCORE = 1 << 30

    def moves(st, seen):
        out = []
        for d in _ORDER:
            res = bd.step(st, d)
            if res is WIN:
                return [(d, WIN)]
            ns, trunc = res
            if trunc or ns[6] or ns == st or ns in seen:
                continue
            if (bd.floor & ~ns[1]) and not alive(ns):
                continue
            out.append((d, ns))
        return out

    def finish(ns):
        """The tail that wins from a fully lit state, or None."""
        return None if (bd.floor & ~ns[1]) else door_path(bd, ns)

    def rollout(st, seen):
        seen = set(seen)
        seq = []
        while True:
            ms = moves(st, seen)
            if not ms:
                return _popcount(st[1]), seq
            if ms[0][1] is WIN:
                return WINSCORE, seq + [ms[0][0]]
            d, ns = rng.choice(ms)
            tail = finish(ns)
            if tail is not None:
                return WINSCORE, seq + [d] + tail
            if not (bd.floor & ~ns[1]):
                return _popcount(ns[1]), seq + [d]
            seq.append(d)
            seen.add(ns)
            st = ns

    def nested(st, seen, lvl):
        seen = set(seen)
        played = []
        best_score, best_seq = -1, []
        while True:
            ms = moves(st, seen)
            if not ms:
                return max(best_score, _popcount(st[1])), played
            if ms[0][1] is WIN:
                return WINSCORE, played + [ms[0][0]]
            for d, ns in ms:
                tail = finish(ns)
                if tail is not None:
                    return WINSCORE, played + [d] + tail
                if not (bd.floor & ~ns[1]):
                    continue
                below = set(seen)
                below.add(ns)
                sc, sq = (rollout(ns, below) if lvl <= 1
                          else nested(ns, below, lvl - 1))
                if sc > best_score:
                    best_score, best_seq = sc, [d] + sq
                if sc >= WINSCORE:
                    return WINSCORE, played + best_seq
            if not best_seq:
                return best_score, played
            d = best_seq[0]
            nxt = next((n for dd, n in ms if dd == d), None)
            if nxt is None:
                return best_score, played + best_seq
            played.append(d)
            best_seq = best_seq[1:]
            seen.add(nxt)
            st = nxt

    for _ in range(restarts):
        score, seq = nested(st0, {st0}, level)
        if score >= WINSCORE:
            return seq
    return None


def replays(bd, st0, presses) -> bool:
    """True when ``presses`` wins from ``st0`` on its LAST press and no earlier."""
    st = st0
    for i, d in enumerate(presses):
        res = bd.step(st, d)
        if res is WIN:
            return i == len(presses) - 1
        ns, trunc = res
        if trunc or ns[6]:
            return False
        st = ns
    return False


def shorten(bd, st0, presses):
    """Delete the longest contiguous BLOCK of presses that still wins, and keep
    doing it until none can go.

    A beam or an NMCS plan wins but WANDERS -- it is steered by "light the next
    thing", which happily takes a detour that a later press undoes. Deleting a
    block and replaying is exact here (the model is the game), longest-first is
    what makes one pass worth several, and it costs O(L^3) model steps, i.e.
    under a second on the longest plan in the game."""
    cur = list(presses)
    dropping = True
    while dropping:
        dropping = False
        n = len(cur)
        for length in range(n - 1, 0, -1):
            for i in range(n - length + 1):
                cand = cur[:i] + cur[i + length:]
                if replays(bd, st0, cand):
                    cur, dropping = cand, True
                    break
            if dropping:
                break
    return cur


#: The ladder `WinterExpert._search` walks, first rung that returns.
_WEIGHTS = (1, 2, 3)
_WIDTHS = (3_000, 30_000, 100_000)
#: Seed 4 first because it is the one that solves level 12 -- measured, at
#: restart 65 of 100, in about 25 minutes. The others are there so a level this
#: rung has never been asked about has more than one stream to try.
_NMCS_SEEDS = (4, 0, 1)
_NMCS_RESTARTS = 100


def solve_board(bd, st0, cap=300_000, widths=_WIDTHS, seeds=_NMCS_SEEDS,
                restarts=_NMCS_RESTARTS):
    """``(presses, proved_shortest)`` or ``(None, False)``."""
    hf = make_h(bd)
    for weight in _WEIGHTS:
        found = astar(bd, st0, hf, weight, cap)
        if found is not None:
            return (found, True) if weight == 1 else (shorten(bd, st0, found), False)
    for width in widths:
        found = beam(bd, st0, hf, width)
        if found is not None:
            return shorten(bd, st0, found), False
    for seed in seeds:
        found = nmcs(bd, st0, seed, restarts)
        if found is not None:
            return shorten(bd, st0, found), False
    return None, False


def optsets_exact(bd, st0, presses, cap=300_000):
    """The EXACT per-step optimal sets for a plan proved shortest.

    At step ``i`` there are ``L - i`` presses left, so press ``p`` is optimal
    iff its successor is ``L - i - 1`` from the win -- decided by an ``f``-bounded
    A*, which is cheap because the bound prunes everything a longer route would
    open. A press that cannot move, truncates, or drowns a bunny is never in the
    set."""
    hf = make_h(bd)
    sets = []
    st = st0
    for i, played in enumerate(presses):
        rest = len(presses) - i
        best = []
        for d in _ORDER:
            res = bd.step(st, d)
            if res is WIN:
                if rest == 1:
                    best.append(d)
                continue
            ns, trunc = res
            if trunc or ns[6] or ns == st or rest == 1:
                continue
            if astar(bd, ns, hf, 1, cap, bound=rest - 1) is not None:
                best.append(d)
        if played not in best:              # cannot happen; a loud guard if it does
            raise AssertionError(f"step {i}: the played press {played} is not optimal")
        sets.append(best)
        st = bd.step(st, played)[0]
    return sets


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class WinterExpert(PSExpert):
    """Plans on `_Board` and never steps the interpreter.

    `PSExpert` owns everything around that -- the in-memory memo, the disk plan
    cache with its staleness check, the level scoping and the snapshot/restore
    discipline -- so the only overrides are `_key` and `_search`.
    """

    #: `_key` carries the static scenery too, so it is canonical across levels;
    #: the memo is scoped anyway, which costs nothing and keeps a hand-edited
    #: level from ever serving another one's plan.
    scope_by_level = True
    #: No rule reads ACTION5, so it is not a move.
    directions = ["up", "down", "left", "right"]
    #: The searches are the whole cost of generation and they are
    #: seed-independent; without this every `parallelize_generator` shard
    #: re-derives all twenty.
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "winter_plans.json")

    def setup(self) -> None:
        self.ids = {name: set(self.g.resolve_object_name(name))
                    for name in _CLASSES}

    def _key(self, eng) -> frozenset:
        """The whole board with each object CLASS collapsed to one tag.

        Three collapses matter, and they are why the default `PSExpert._key`
        (every non-background object) will not do here:

          * the five tree objects are one tag, because ``[rtree] -> [random
            tree]`` draws between five identical objects at level start -- key
            on the concrete one and the ``plan_cache_path`` signature differs in
            every fresh process, i.e. the disk cache never hits;
          * ``water``/``ice`` are one tag and ``door``/``nodoor`` are one tag,
            because both of those states are FUNCTIONS of the lit set (a frozen
            cell is a lit water cell; an open door is a board with nothing
            unlit) and keying them separately would only shadow states;
          * the four bunny facings are one tag -- the facing is art and no rule
            reads it.

        The static scenery IS in the key even though nothing moves it, so the
        disk cache's staleness check has something to check: an edited level
        with the same start position would otherwise read as a hit."""
        ids = self.ids
        classes = ((ids["winter"], 0), (ids["nlit"], 1), (ids["crat"], 2),
                   (ids["bunny"], 3), (ids["diags"], 4), (ids["lose"], 5),
                   (ids["semir"], 6), (ids["swmir"], 7), (ids["nwmir"], 8),
                   (ids["nemir"], 9), (ids["block"], 10),
                   (ids["door"] | ids["nodoor"], 11),
                   (ids["tree"] | ids["dead"], 12),
                   (ids["water"] | ids["ice"], 13))
        return frozenset(
            (r, c, tag)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for group, tag in classes
            if cell & group
        )

    def heuristic(self, eng) -> int:
        raise AssertionError("WinterExpert plans natively; heuristic is unused")

    def board(self, eng):
        return _Board.read(eng, self.ids)

    def _search(self, eng) -> "Plan | None":
        # Level 6 has no door: its win is `[autumn no moved three] -> win`, and
        # the interpreter's own level-start rules have already chased Autumn
        # into the one corner where he can move neither right nor down. So the
        # rule fires on the first `again` tick of ANY press -- there is nothing
        # to plan and `_Board`, which models Winter and not Autumn, is not
        # asked to.
        if any(cell & self.ids["autumn"] for row in eng.grid for cell in row):
            return Plan(["right"], [list(_ORDER)])
        bd, st = self.board(eng)
        presses, proved = solve_board(bd, st, self.node_cap)
        if presses is None:
            return None
        sets = (optsets_exact(bd, st, presses, self.node_cap) if proved
                else [[d] for d in presses])
        return Plan(presses, sets)


class WinterSolver(PSAStarSolver):
    game_id = "puzzlescript_winter"
    game_name = GAME_NAME
    expert_cls = WinterExpert

    #: Level 19 is the one the ladder cannot win, and it is skipped up front so
    #: a cold `discover_solvable` does not spend three quarters of an hour
    #: proving it again on every fresh checkout.
    #:
    #: Its door hangs off the side of a column with no wall at either end, so
    #: nothing can come to rest on the door's own row -- unless something is
    #: STANDING in that column, and the only movable thing on the board is the
    #: bunny. So the level is won by herding the bunny into the column and then
    #: sliding up into it, which the covering search has no reason to want:
    #: every rung here lights all 62 squares easily (`--plans` and the NMCS both
    #: reach 62/62 every single run) and then finds the door shut for good.
    #: `make_door_reach` was added to steer the beam at it and is not enough --
    #: parking the bunny early enough to matter costs the coverage its own
    #: mobility. Left in the file, not deleted: it is what would have to be
    #: strengthened, and it is exactly the term that would find the level.
    skip_levels = frozenset({19})

    #: The longest plan is ~50 presses; the rest is room for the exploration
    #: prefix and the re-plan after it. Well under the adapter's 200-step
    #: per-level budget, which would otherwise flip a level to GAME_OVER.
    max_steps = 150
    node_cap = 300_000


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _pick(solver, game) -> list:
    """The levels a report runs over: the 0-based indices given on the command
    line, or every level the solver actually records.

    Naming a level explicitly overrides ``skip_levels``, which is how level 19
    gets looked at at all -- and why the reports take the argument, since one
    cold level can cost minutes."""
    want = [int(x) for x in sys.argv[1:] if x.isdigit()]
    if want:
        return want
    skipped = sorted(solver.skip_levels)
    if skipped:
        print(f"(skipping level{'s' if len(skipped) > 1 else ''} "
              f"{', '.join(str(x) for x in skipped)} -- see "
              f"WinterSolver.skip_levels; name one to run it anyway)")
    return [lvl for lvl in range(game.n_levels) if lvl not in solver.skip_levels]


def _levels():
    """``(solver, game, expert)`` with the adapter parked at level 0."""
    solver = WinterSolver()
    game = solver.make_game(0)
    return solver, game, WinterExpert(game, node_cap=solver.node_cap)


def _dynamic(eng, expert) -> tuple:
    """The pieces, read off the interpreter, phase-free so it compares across
    runs: ``(pos, lit, crates, bunnies, mirrors, diags, lose)``."""
    ids = expert.ids
    w = eng.width
    pos = -1
    lit = crates = bunnies = diags = lose = 0
    mirr = {}
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i, b = r * w + c, 1 << (r * w + c)
            if cell & ids["winter"]:
                pos = i
            if cell & ids["nlit"]:
                lit |= b
            if cell & ids["crat"]:
                crates |= b
            if cell & ids["bunny"]:
                bunnies |= b
            if cell & ids["diags"]:
                diags |= b
            if cell & ids["lose"]:
                lose |= b
            for nm in _MIRROR:
                if cell & ids[nm]:
                    mirr[i] = nm
    return pos, lit, crates, bunnies, tuple(sorted(mirr.items())), diags, lose


def _report() -> int:
    """Per level: the board, the plan, how it was found, its tie coverage -- and
    CERTIFY the plan by replaying it through the interpreter, which must win on
    the last press and on no earlier one."""
    solver, game, expert = _levels()
    total = bad = 0
    for level in _pick(solver, game):
        game.set_level(level)
        eng = game._engine
        t = time.time()
        plan = expert.plan(eng, level)
        took = time.time() - t
        if plan is None:
            bd, st = expert.board(eng)
            print(f"level {level:2d}: {bd.h:2d}x{bd.w:2d} NO PLAN "
                  f"({_popcount(bd.floor)} squares to light) in {took:5.1f}s")
            bad += 1
            continue
        if any(cell & expert.ids["autumn"] for row in eng.grid for cell in row):
            # The plan claims EVERY press wins, and its optimal set says so, so
            # every press is what gets certified.
            wins = []
            for d in _ORDER:
                game.set_level(level)
                eng.step(d)
                wins.append(eng.check_win())
            ok = all(wins)
            bad += not ok
            print(f"level {level:2d}: AUTUMN -- the level-start rules corner him "
                  f"where he can move neither right nor down, so his own win "
                  f"rule fires on the first tick of any press: "
                  f"{sum(wins)}/4 presses win "
                  f"({'CERTIFIED' if ok else 'REJECTED'})")
            continue
        bd, st = expert.board(eng)
        won_at = None
        for i, d in enumerate(plan):
            eng.step(d)
            if eng.check_win():
                won_at = i
                break
        ok = won_at == len(plan) - 1
        bad += not ok
        total += len(plan)
        sets = getattr(plan, "optsets", None) or []
        ties = sum(1 for s in sets if len(s) > 1)
        exact = _proved(bd, st, plan, solver.node_cap)
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {bd.h:2d}x{bd.w:2d} "
              f"{_popcount(bd.floor):3d} squares "
              f"({_popcount(bd.waters):2d} water, {_popcount(bd.trees):2d} trees, "
              f"{_popcount(st[2])} crates, {_popcount(st[3])} bunnies, "
              f"{len(st[4])} mirrors), "
              f"{len(plan):3d} presses (budget {game._max_steps}, {room}), "
              f"lb {make_h(bd)(st, bool(st[4])):3d}"
              f"{' PROVED SHORTEST' if exact else ''}, "
              f"{ties:3d} tie steps, {took:5.1f}s, "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    print(f"total {total} presses")
    return 0 if not bad else 1


def _proved(bd, st, plan, cap=300_000) -> bool:
    """True when the plan is provably SHORTEST -- i.e. the weight-1 rung is the
    one that produced it. Re-run rather than remembered, because a plan served
    from the disk cache carries no note of which rung found it."""
    exact = astar(bd, st, make_h(bd), 1, cap)
    return exact is not None and len(exact) == len(plan)


def _verify(cap: int = 400_000) -> int:
    """Double entry on the plan LENGTHS, the tie SETS and the HEURISTIC itself.

    The plans come off a ladder whose first rung is an A* over `make_h`, and the
    optimal sets come off a bounded A* over the same heuristic. If `make_h` were
    not admissible both would be silently wrong and neither `_report` (which only
    checks that the interpreter wins on the last press) nor `--fuzz` (which only
    exercises `_Board.step`) would see it.

    So each level whose reachable component fits under ``cap`` gets an exhaustive
    forward BFS -- no heuristic anywhere in it -- and three things are compared
    against it: the plan's length is the BFS's distance-to-win from the start,
    every optimal set is exactly the presses whose successor is one closer, and
    ``h(s) <= d*(s)`` at EVERY state of the field, which is admissibility
    measured rather than argued.

    A level whose field does not fit reports what it could check: that the
    singleton labels are the presses the plan takes."""
    solver, game, expert = _levels()
    bad = 0
    for level in _pick(solver, game):
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None or any(cell & expert.ids["autumn"]
                               for row in eng.grid for cell in row):
            print(f"level {level:2d}: skipped (no plan, or Autumn's level)")
            continue
        bd, st0 = expert.board(eng)
        sets = list(getattr(plan, "optsets", None) or [[d] for d in plan])
        t = time.time()
        dist = _bfs_field(bd, st0, len(plan), cap)
        if dist is None:
            notes = [i for i, (opt, d) in enumerate(zip(sets, plan)) if opt != [d]]
            multi = sum(1 for opt in sets if len(opt) > 1)
            bad += len(notes)
            print(f"level {level:2d}: {len(plan):3d} presses, field over "
                  f"{cap} states -- not re-derivable; "
                  f"{multi} measured sets, "
                  f"{'the rest are the presses played' if not notes else notes[:3]}")
            continue
        hf = make_h(bd)
        mirrored = bool(st0[4])
        notes = []
        over = [st for st, d in dist.items() if hf(st, mirrored) > d]
        if over:
            notes.append(f"HEURISTIC INADMISSIBLE at {len(over)} states")
        if dist.get(st0) != len(plan):
            notes.append(f"LENGTH {dist.get(st0)} != {len(plan)}")
        st = st0
        for i, d in enumerate(plan):
            here = dist.get(st)
            want = []
            for alt in _ORDER:
                res = bd.step(st, alt)
                if res is WIN:
                    if here == 1:
                        want.append(alt)
                    continue
                ns, trunc = res
                if trunc or ns[6] or ns == st:
                    continue
                if dist.get(ns, -1) == here - 1:
                    want.append(alt)
            if want != sets[i]:
                notes.append(f"step {i}: {sets[i]} != {want}")
            st = bd.step(st, d)[0]
        bad += len(notes)
        print(f"level {level:2d}: d*={dist.get(st0)} over {len(dist)} states by an "
              f"independent BFS, {sum(len(x) for x in sets):3d} labelled presses "
              f"and the heuristic checked in {time.time() - t:5.1f}s: "
              f"{'AGREES' if not notes else '; '.join(str(x) for x in notes[:3])}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _bfs_field(bd, st0, depth, cap):
    """``{state: presses to the win}`` for every state within ``depth`` of the
    start, or None past ``cap`` states.

    A plain forward BFS with NO heuristic, then a backward sweep over the same
    frontier order until the distances settle. Deliberately the slow way round:
    it is the independent witness `_verify` holds the A* to."""
    order = [st0]
    seen = {st0: 0}
    wins = set()
    i = 0
    while i < len(order):
        st = order[i]
        i += 1
        g = seen[st]
        if g >= depth:
            continue
        for d in _ORDER:
            res = bd.step(st, d)
            if res is WIN:
                wins.add(st)
                continue
            ns, trunc = res
            if trunc or ns[6] or ns == st or ns in seen:
                continue
            if len(seen) >= cap:
                return None
            seen[ns] = g + 1
            order.append(ns)
    dist = {st: 1 for st in wins}
    changed = True
    while changed:
        changed = False
        for st in reversed(order):
            best = dist.get(st)
            for d in _ORDER:
                res = bd.step(st, d)
                if res is WIN:
                    cand = 1
                else:
                    ns, trunc = res
                    if trunc or ns[6] or ns == st or ns not in dist:
                        continue
                    cand = dist[ns] + 1
                if best is None or cand < best:
                    best = cand
            if best is not None and best != dist.get(st):
                dist[st] = best
                changed = True
    return dist


def _fuzz(trials: int = 40, walk: int = 40) -> int:
    """Assert `_Board` reproduces the interpreter EXACTLY.

    Random play from the level start only, but from every level: the mechanics
    that could drift (a mirror bend, a crate run, a bunny drowning, a tree bump)
    all sit within a few presses of any board, and the coverage counters below
    say whether the run actually exercised them -- a fuzz that reports agreement
    having bounced off no mirror has checked nothing about mirrors.

    Level 6 is excluded by construction: its win is Autumn's, and `_Board`
    models Winter."""
    solver, game, expert = _levels()
    rng = random.Random(20260823)
    bad = 0
    tot = {"slide": 0, "freeze": 0, "bump": 0, "push": 0, "bunny": 0,
           "drown": 0, "refused": 0, "truncated": 0}
    for level in _pick(solver, game):
        game.set_level(level)
        eng = game._engine
        if any(cell & expert.ids["autumn"] for row in eng.grid for cell in row):
            print(f"level {level:2d}: skipped (Autumn wins it on any press)")
            continue
        layout = game._game.levels[level]
        bd, st0 = expert.board(eng)
        cnt = {k: 0 for k in tot}
        mism = 0
        for _trial in range(trials):
            eng.load_level(layout)
            st = st0
            for _t in range(walk):
                d = rng.choice(_ORDER)
                eng._rule_restart = False
                eng.step(d)
                res = bd.step(st, d)
                if eng._rule_restart:
                    break                       # a drowned bunny: level thrown away
                if res is not WIN and res[1]:
                    cnt["truncated"] += 1
                    break                       # the abandoned slide; see the docstring
                won = eng.check_win()
                if res is WIN:
                    if not won:
                        mism += 1
                        print(f"  level {level}: model WON after {d}, engine did not")
                    break
                if won:
                    mism += 1
                    print(f"  level {level}: engine WON after {d}, model did not")
                    break
                ns, _ = res
                got = _dynamic(eng, expert)
                want = (ns[0], ns[1] & bd.floor, ns[2], ns[3], ns[4], ns[5], ns[6])
                got = (got[0], got[1] & bd.floor) + got[2:]
                if got != want:
                    mism += 1
                    if mism <= 2:
                        print(f"  level {level} MISMATCH after {d}:\n"
                              f"    engine {got}\n    model  {want}")
                    break
                if ns == st:
                    cnt["refused"] += 1
                elif ns[6] != st[6]:
                    cnt["drown"] += 1
                elif ns[2] != st[2] or ns[4] != st[4] or ns[5] != st[5]:
                    cnt["push"] += 1
                elif (bd.trees & ns[1]) != (bd.trees & st[1]):
                    cnt["bump"] += 1
                elif (bd.waters & ns[1]) != (bd.waters & st[1]):
                    cnt["freeze"] += 1
                else:
                    cnt["slide"] += 1
                if ns[3] != st[3]:
                    cnt["bunny"] += 1
                st = ns
        for k in cnt:
            tot[k] += cnt[k]
        bad += mism
        print(f"level {level:2d}: {cnt['slide']:5d} slides {cnt['freeze']:4d} freezes "
              f"{cnt['bump']:4d} bumps {cnt['push']:4d} pushes "
              f"{cnt['bunny']:4d} bunny shoves {cnt['drown']:3d} drownings "
              f"{cnt['refused']:5d} refused {cnt['truncated']:3d} truncated: "
              f"{'OK' if not mism else str(mism) + ' MISMATCHES'}")
    n = sum(tot.values())
    print(f"{n} transitions: "
          f"{'the model matches the interpreter' if not bad else f'{bad} MISMATCHES'}")
    thin = [k for k in ("freeze", "bump", "push", "bunny") if not tot[k]]
    if thin and not any(x.isdigit() for x in sys.argv[1:]):
        # Only meaningful over the whole game: naming three levels on the command
        # line and getting no crate push back is not a weak fuzz, it is arithmetic.
        print(f"FUZZ TOO WEAK: it never exercised {', '.join(thin)}")
        return 1
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION a level can show is distinct at every cell
    size in use, and that the two DIRECTIONAL sprite families are closed under a
    quarter turn.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window.

    The five trees are declared identical on purpose -- that is the fix for
    ``[rtree] -> [random tree]`` drawing from an unseeded RNG -- and so is
    ice-versus-lit-floor, which is a state the game does not have (a frozen cell
    IS a lit cell).
    """
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {
        "unlit": ("it",), "lit": ("nlit",),
        "water": ("it", "water"), "ice": ("nlit", "ice"),
        "wall": ("block",),
        "door_shut": ("door",), "door_open": ("nlit", "nodoor"),
        "tree": ("it", "treea"), "tree_lit": ("nlit", "treea"),
        "dormant": ("nlit", "dead"),
        "crate": ("it", "crat"), "crate_lit": ("nlit", "crat"),
        "bunny": ("it", "bunnyu"), "bunny_lit": ("nlit", "bunnyu"),
        "winter": ("nlit", "winter"), "winter_on_ice": ("nlit", "ice", "winter"),
        "winter_wins": ("nlit", "nodoor", "winter"),
        "mirror_ne": ("it", "nemir"), "mirror_se": ("it", "semir"),
        "mirror_nw": ("it", "nwmir"), "mirror_sw": ("it", "swmir"),
        "mirror_ne_lit": ("nlit", "nemir"),
        "diag": ("nlit", "diag"), "diago": ("nlit", "diago"),
        "drowned": ("it", "lose"),
    }
    #: Pairs the game cannot tell apart and does not need to.
    accepted = {
        ("lit", "ice"),                     # a frozen cell IS a lit cell
        ("winter", "winter_on_ice"),        # ditto, under the player
    }
    for a, b in itertools.combinations(("treea", "treeb", "treec", "treed",
                                        "treee"), 2):
        accepted.add((a, b))

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
                   if np.array_equal(shots[a], shots[b])
                   and (a, b) not in accepted and (b, a) not in accepted]
        stale = [(a, b) for a, b in accepted
                 if a in shots and b in shots
                 and not np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        # The two squares the whole game is about are unlit and lit floor, and
        # the two that decide a press are water and ice, so their margins are
        # reported rather than only their distinctness.
        floor_px = int((shots["unlit"] != shots["lit"]).sum()) / (h * w)
        water_px = int((shots["water"] != shots["ice"]).sum()) / (h * w)
        tree_px = int((shots["tree"] != shots["unlit"]).sum()) / (h * w)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"unlit-vs-lit {floor_px:4.1f}px/cell, water-vs-ice "
              f"{water_px:4.1f}, tree-on-unlit {tree_px:4.1f}, "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}"
              f"{'' if not stale else ' STALE ACCEPTS ' + str(stale)}")

    # The rotation augmentation turns the whole frame, so a sprite family that
    # encodes a direction must be CLOSED under a quarter turn or a rotated board
    # shows art the game does not own. Both families here are.
    turns = {"nemir": "semir", "semir": "swmir", "swmir": "nwmir",
             "nwmir": "nemir",
             "bunnyu": "bunnyr", "bunnyr": "bunnyd", "bunnyd": "bunnyl",
             "bunnyl": "bunnyu"}
    for src, want in turns.items():
        a = g.objects[src].sprite
        b = g.objects[want].sprite
        turned = [[a[4 - c][r] for c in range(5)] for r in range(5)]
        if turned != b:
            print(f"ROTATION: {src} turned once is not {want}")
            bad += 1
    print("audit clean" if not bad else f"AUDIT FAILED: {bad} problems")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(WinterSolver.main())
