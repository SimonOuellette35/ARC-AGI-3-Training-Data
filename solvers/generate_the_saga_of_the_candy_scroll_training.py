"""Generate Phase-1 training data for the PuzzleScript game
ps:the_saga_of_the_candy_scroll ("The Saga of the Candy Scroll", Jim Palmeri).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a tick-exact native model of it, the exact distance field that model
is solved with where it fits, the beam that finishes the levels it does not,
the five levels this interpreter makes UNWINNABLE and the proof of that, and
the render audit behind the two sprite fixes in the .txt.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_the_saga_of_the_candy_scroll",
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
Thirty levels. You walk with the arrow keys; walls, ninjas and candies all
block you and you can push none of them. X is the whole game: it feeds the
candy you are STANDING NEXT TO into the Candy Scroll, a 4-column well down the
left of the screen, and the scroll plays match-3 with itself. The win is
``No Candy`` and ``No Enemy`` -- every candy on the floor collected AND every
candy in the scroll cleared AND every ninja dead.

The one rule that makes it a puzzle is which COLUMN a candy lands in, and it is
decided by which side of it you are standing on, not by where you walked from:

    left  [ Action Player | Candy ] [ Drop1 no Candy ]   candy on your LEFT  -> column 0
    up    [ Action Player | Candy ] [ Drop2 no Candy ]   candy ABOVE you     -> column 1
    down  [ Action Player | Candy ] [ Drop3 no Candy ]   candy BELOW you     -> column 2
    right [ Action Player | Candy ] [ Drop4 no Candy ]   candy on your RIGHT -> column 3

The four arrows drawn over the scroll (red/blue/green/purple, left to right)
are that legend. Three consequences worth stating because none of them is
obvious from the art:

  * the four rules are SEPARATE, so ONE press picks up EVERY candy adjacent to
    you -- up to four at once, one into each column, and you cannot decline
    any of them. In the dense levels (26, 28, 30) standing in a hole you ate
    is how a plan of 64 candies fits in 29 presses; in the sparse ones it is
    how a plan jams itself;
  * ``[ DropN no Candy ]`` gates each rule on that column's TOP cell being
    free, so a full column silently swallows the press;
  * a candy is dropped at the top row and the fall rule runs AFTER the match
    rules in a tick, so three candies fed to three adjacent columns by a single
    press match AT THE DROP ROW, before any of them falls. `_cascade` is
    written as the interpreter's tick loop for exactly this reason and not as
    "settle, then look for lines".

Ninjas are killed by the scroll, at range, through walls:

    [ ScrollBG ClearMarker CandyR ] [ Player | ... | NinjaR ]
        -> [ ScrollBG ClearMarker CandyR ] [ Player | ... | NinjaR ClearMarker ]

so a red ninja dies when a RED line clears while you share its row or column.
Measured, not read off the rule text (`--selfcheck` pass 3 is the measurement):

  * the ellipsis cells are unconstrained, so distance is irrelevant and WALLS
    DO NOT BLOCK -- "aligned" means aligned, full stop;
  * but only the NEAREST ninja of that colour in each direction dies. This is
    an interpreter property, not a PuzzleScript one: `_apply_multi_group_rule_forces`
    builds the second bracket's match list with `_check_rule_match_forces`,
    which returns the FIRST ellipsis bind per start cell, and there is one
    player. Two same-coloured ninjas on one ray therefore need two separate
    match rounds. It cost two fuzz failures to find and it is what makes
    levels 18-20 as tight as they are;
  * the alignment is tested at the tick the line CLEARS, which can be a dozen
    ticks after the press -- but `PSEngine.step` drains the whole `again`
    cascade inside one `perform_action`, so the player cannot have moved and
    the press square is the only square that matters.

FIVE LEVELS CANNOT BE WON HERE, and the reason is that same drain. Real
PuzzleScript schedules an `again` continuation ``again_interval .1`` seconds
later and keeps taking input in between, so the original game lets you press X
and then RUN to line yourself up with a ninja while the candy is still falling.
This adapter runs the cascade to a fixpoint inside the keypress, which turns
that into "you must already be aligned when you press". Levels 12, 13, 17, 20
and 21 are built on the other reading:

  * level 12 is the tutorial for it. The lone red candy sits in the ninja's
    COLUMN, and the two reds already in the scroll are in columns 1 and 2, so
    the third has to go to column 0 or column 3 -- i.e. you must stand to the
    candy's left or right, i.e. one square off the ninja's column, on purpose;
  * level 13 has three candies (so exactly one match) and two ninjas, and the
    only two squares aligned with both are (2,12) and (5,8), neither of which
    is ever adjacent to a candy.

``--prove`` re-derives this rather than asserting it: the live macro-move space
of each of the five is enumerated WHOLE -- 1, 59, 25,015, 368,618 and 34,059
states, none truncated and none stopped by the cap -- and not one of those
states is a win, so the start is unreachable from every winning board. They are
in `skip_levels` so startup does not pay for that discovery every time.

Expert solver
-------------
A tick-exact native model of the above (`_cascade`, `_Level.press`) and two ways
solving it, picked per level by whether the first one fits:

  * EXACT. The macro-move graph -- a node is a settled board plus the square
    you are standing on after a press, an edge is "walk there and press",
    weighted by the walk -- is a DAG, because every useful press destroys at
    least one floor candy and none is ever created. So the whole reachable
    space is enumerated once and solved by one DP sweep over layers of
    remaining-candy count, with no heuristic and no search. That buys plans
    that are provably SHORTEST, EXACT optimal-action sets, and recovery from
    anywhere as a dictionary lookup. SIXTEEN levels fit inside the state cap
    -- 1-9, 11, 14, 15, 16, 19, 23 and 26 -- the largest being level 16 at
    994,345 live states, 85 seconds and 0.36GB, which is where the cap is set;
  * BEAM otherwise, over the same macro moves, as a portfolio of (width,
    successor-cap, weighting, tie-jitter) settings tried in order. It finishes
    the other SEVEN: 10, 22, 24, 25, 28, 29 and 30. Those plans are wins of a
    stated length and are NOT claimed shortest; the report says which is which.

So 23 of the 30 levels are solved, 1254 presses in all, 226 of them carrying a
second equally-right answer. Two are neither solved nor proved impossible --
18 and 27, whose spaces are past 3.5M states and whose beams dead-end -- and
they are simply dropped from the episode. Nothing is ever taped half-done.

The budget is worth one line: level 25's plan is 197 presses against the
adapter's 200-per-level cap. It fits (the `set_level` that ends the
exploration prefix resets the counter to zero), but it is the one level with
no room, so `--plans` prints every plan's length against that cap rather than
leaving it to be discovered by a GAME_OVER during generation.

What is proved and what is not
------------------------------
  * the MODEL is measured against the interpreter, not argued: `--selfcheck`
    fuzzes it three ways (random play on all 30 levels from randomly refilled
    scrolls; every candy approached from every legal side on every level; and
    presses with 1-4 candies around the player over random scroll fills and
    random ninja layouts, which is the only pass that reaches the multi-drop
    and same-ray-ninja collisions the shipped levels barely touch), comparing
    the WHOLE board after every press. 21,660 presses, no disagreement -- and
    both of the model bugs that ever existed were found by that third pass;
  * a press never leaves the board mid-animation: the longest cascade measured
    is 17 ticks against `PSEngine.step`'s budget of 50, and `--selfcheck`
    asserts the settled-ness after every press rather than assuming it;
  * the sixteen exact levels' plans are shortest against their entire space,
    and their optimal sets are the true tie sets (`dist(successor) == dist-1`),
    not inferred ones;
  * the seven beam levels' plans are wins of a stated length and nothing more.
    Their walk steps still carry sound tie sets -- every first step of a
    shortest route to the press square -- which is a SUBSET of the true optimal
    set, never a superset, so no suboptimal action is ever labelled optimal;
  * end to end: replaying a recorded episode's actions into a fresh adapter at
    the same seed reproduces every frame exactly and wins every level, at seeds
    whose rotations differ -- which is the rotation contract, measured.

Recovery
--------
`supports_recovery = True` with the family's RESET-mode prefix: an episode
opens with ~10 random presses and one RESET back to the level start, which is
the state every cached plan was solved from. The exact levels could support
mid-plan epsilon detours as well (their recovery really is a table lookup);
the beam levels could not, since every detour would cost a fresh beam search,
so `epsilon` stays 0 and the whole game uses one recovery mode.

Rendering
---------
Two sprite fixes in `data/puzzlescript_games/The_Saga_of_the_Candy_Scroll.txt`
(and its lowercase twin), both measured by ``--audit``, which renders every
cell stack a settled board of this game can hold at every board shape the game
uses and asserts they are all distinct:

 1. THE NINJAS WERE ALL THE SAME NINJA. Their colour lived on sprite row 1
    only (``.000.`` under a black hood), and a cell renders by sampling
    rows/cols ``(2i+1)*5 // (2*cell_px)`` -- which at ``cell_px = 3``, the size
    every 18- and 20-wide board gets, is rows {0, 2, 4}. Row 1 is not among
    them, so NinjaR, NinjaG, NinjaB and NinjaI rendered as the same four black
    pixels on levels 22 through 30, on a game whose whole late-act mechanic is
    "blue ninjas need blue candy". --audit found all six pairs colliding at 81
    cells. The colour now sits on the sprite's four CORNERS, which survive
    {0,2,4} and {0,1,3,4} alike, and reads differently from a candy (a filled
    disc, so a PLUS at cell_px 3) at every size.
 2. THE WALLS WERE THE LETTERBOX. Wall shipped as ``Black #111111``, both of
    which are ARC index 5 -- exactly the colour `_render_frame` pads a
    non-square board with, and every board here is padded. Wall is now
    ``DarkGray`` (ARC 3), which nothing else in this game uses. Same bug and
    same fix as ps:the_blob and ps:stand_iii.

No object, collision layer, legend entry, rule, level or win condition is
touched, so the mechanic is bit-for-bit the game it was -- which is what
--selfcheck measures, since it fuzzes the native model against this .txt.

CLI
---
    --audit       render every cell stack at every board shape; all distinct?
    --selfcheck   fuzz the native model against the interpreter (4 passes;
                  add --quick for a third of the presses)
    --prove       enumerate the five unwinnable levels whole
    --plans       every level's plan, replayed through the real interpreter
                  (~10 min the first time, then served from the plan cache)
    --report      all of the above
    (no flag)     generate episodes -- see `BaseSolver.main`

The searches are the whole cost of generation and they are seed-independent
(the engine state after a reset is the same for every seed -- only the
presentation is augmented), so they are cached in
``data/candy_scroll_plans.json``, keyed by the level's start layout so an
edited level is a miss rather than a wrong plan. First run ~10 minutes, every
later seed about a second.
"""

from __future__ import annotations

import random
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame  # noqa: E402
from solvers.common.ps_astar import (  # noqa: E402
    Plan, PSAStarSolver, PSExpert,
)

GAME_NAME = "The_Saga_of_the_Candy_Scroll"

#: Colour index -> the letter the reports print. Order matches the CandyR /
#: CandyG / CandyB / CandyI and NinjaR / ... object suffixes, which is the only
#: place the mapping is defined -- `_read` resolves the object ids from it.
COLOURS = "RGBI"

#: The four add rules, in FILE ORDER: (engine direction, (dr, dc) from the
#: player to the candy, Drop object index). ``Drop{k+1}`` is the object; which
#: scroll column it sits in is read off the grid (`_Board.dropcol`), not
#: assumed -- the levels all put Drop1..Drop4 in columns 0..3, and the assert
#: in `_Board` is what says so.
_ADD = (("left", (0, -1), 0), ("up", (-1, 0), 1),
        ("down", (1, 0), 2), ("right", (0, 1), 3))

_DELTA = {"left": (0, -1), "up": (-1, 0), "down": (1, 0), "right": (0, 1)}
_STEPS = ("up", "down", "left", "right")

#: `PSEngine.step`'s own `again` budget. A press that needed more would come
#: back mid-animation and the model would be describing a board the game never
#: settles into; the longest measured is 17, and `--selfcheck` re-measures it.
MAX_AGAIN = 50


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class _Board:
    """The static half of a level: what blocks the player, and the scroll.

    Read off the live grid rather than off the level text, so it is equally
    valid at a perturbed state (nothing here can change during play: no rule of
    this game creates or destroys a wall, a scroll piece or a Drop)."""

    __slots__ = ("h", "w", "blocked", "srows", "scols", "sh", "scell", "dropcol")

    def __init__(self, eng, game):
        n = game.obj_name_to_idx
        self.h, self.w = eng.height, eng.width
        grid = eng.grid
        wall = {n[k] for k in ("wall", "scrollleft", "scrollright",
                               "scrolltop", "scrollbottom")}
        self.blocked = frozenset((r, c) for r in range(self.h)
                                 for c in range(self.w) if grid[r][c] & wall)
        bg = n["scrollbg"]
        cells = [(r, c) for r in range(self.h) for c in range(self.w)
                 if bg in grid[r][c]]
        rows = sorted({r for r, _ in cells})
        cols = sorted({c for _, c in cells})
        assert len(cols) == 4 and len(cells) == len(rows) * 4, (rows, cols)
        self.srows, self.scols, self.sh = rows, cols, len(rows)
        self.scell = frozenset(cells)
        self.dropcol = []
        for k in range(4):
            spot = [(r, c) for r in range(self.h) for c in range(self.w)
                    if n[f"drop{k + 1}"] in grid[r][c]]
            assert len(spot) == 1 and spot[0][0] == rows[0], spot
            self.dropcol.append(cols.index(spot[0][1]))
        assert sorted(self.dropcol) == [0, 1, 2, 3]
        self.dropcol = tuple(self.dropcol)


def _read(eng, game):
    """``(board, pos, cands, ninjas, stacks)`` off the LIVE engine grid.

    ``stacks[k]`` is scroll column k BOTTOM FIRST. A settled board is always
    bottom-aligned (the fall rule runs to a fixpoint inside every press) and
    holds no animation objects, so four stacks of colours is a complete
    description of the scroll -- which is what makes the state small enough to
    enumerate."""
    n = game.obj_name_to_idx
    board = _Board(eng, game)
    grid = eng.grid
    cand = {n["candy" + s.lower()]: k for k, s in enumerate(COLOURS)}
    ninja = {n["ninja" + s.lower()]: k for k, s in enumerate(COLOURS)}
    pos = None
    cands: dict[tuple[int, int], int] = {}
    ninjas: dict[tuple[int, int], int] = {}
    cols: list[list[tuple[int, int]]] = [[], [], [], []]
    for r in range(board.h):
        for c in range(board.w):
            cell = grid[r][c]
            if n["player"] in cell:
                pos = (r, c)
            for idx, k in cand.items():
                if idx in cell:
                    if (r, c) in board.scell:
                        cols[board.scols.index(c)].append((r, k))
                    else:
                        cands[(r, c)] = k
            for idx, k in ninja.items():
                if idx in cell:
                    ninjas[(r, c)] = k
    stacks = tuple(tuple(k for _, k in sorted(col, reverse=True)) for col in cols)
    return board, pos, cands, ninjas, stacks


def _settled(eng, game) -> bool:
    """False while any Candy0..Candy5 / ClearMarker is still on the grid, i.e.
    while a press's `again` cascade has not finished. Everything below assumes
    settled boards; --selfcheck asserts the engine only ever hands us one."""
    n = game.obj_name_to_idx
    live = [n[f"candy{i}"] for i in range(6)] + [n["clearmarker"]]
    return not any(idx in eng.grid[r][c]
                   for r in range(eng.height) for c in range(eng.width)
                   for idx in live)


# ---------------------------------------------------------------------------
# The scroll: one press, tick for tick
# ---------------------------------------------------------------------------
#
# A scroll cell during a cascade is None, ("C", colour) or ("A", stage 0..5)
# -- the clear animation, which is a Candy for every rule that asks (so it
# blocks the fall and the DropN gate) but matches nothing.

def _marks(grid) -> set:
    """Every cell in a run of >=3 of one colour, horizontally or vertically.

    ``[ ScrollBG CandyR | ScrollBG CandyR | ScrollBG CandyR ]`` carries no
    direction word, so it is expanded over all four -- which makes it "runs of
    three in a row OR a column", and a run of four (the scroll is exactly four
    wide) clears all four, not three of them. Overlapping runs mark their
    union: an L of five same-coloured candies goes in one round."""
    sh = len(grid)
    out = set()

    def col(r, k):
        v = grid[r][k]
        return v[1] if v is not None and v[0] == "C" else None

    for k in range(4):
        run = 1
        for r in range(1, sh):
            here = col(r, k)
            run = run + 1 if here is not None and here == col(r - 1, k) else 1
            if run >= 3:
                out.update((j, k) for j in range(r - run + 1, r + 1))
    for r in range(sh):
        run = 1
        for k in range(1, 4):
            here = col(r, k)
            run = run + 1 if here is not None and here == col(r, k - 1) else 1
            if run >= 3:
                out.update((r, j) for j in range(k - run + 1, k + 1))
    return out


_CASCADE: dict = {}


def _cascade(sh: int, stacks: tuple, drops: tuple):
    """Drain one press's `again` cascade over the SCROLL alone.

    Returns ``(stacks, rounds, ticks)``: the settled columns, the colours
    cleared in each match round IN ORDER, and how many `again` iterations the
    engine spent. Memoized -- the scroll is at most 4x10 cells and the same
    board recurs constantly across a search.

    The tick loop is the interpreter's rule order and the order is the point:
    animation, then match, then fall. Matching BEFORE falling is why a press
    that feeds three columns at once can clear at the drop row; and a marked
    candy becomes Candy0, which is a Candy, so the column above it is frozen
    for the six ticks the animation takes rather than collapsing at once.

    Ninjas are not modelled here (they live on the floor, and nothing they do
    feeds back into the scroll) -- `_press` walks ``rounds`` against the live
    ninja set instead, which is also what keeps this memo pure."""
    memo = (sh, stacks, drops)
    hit = _CASCADE.get(memo)
    if hit is not None:
        return hit
    grid = [[None] * 4 for _ in range(sh)]
    for k in range(4):
        for i, colour in enumerate(stacks[k]):
            grid[sh - 1 - i][k] = ("C", colour)
    for k in range(4):
        if drops[k] is not None:
            grid[0][k] = ("C", drops[k])
    rounds = []
    ticks = 1                                   # the press turn itself
    while ticks < MAX_AGAIN:
        ticks += 1
        changed = False
        for r in range(sh):                     # 1. the clear animation
            for k in range(4):
                v = grid[r][k]
                if v is not None and v[0] == "A":
                    grid[r][k] = None if v[1] == 5 else ("A", v[1] + 1)
                    changed = True
        mk = _marks(grid)                       # 2. mark, then destroy
        if mk:
            changed = True
            rounds.append(frozenset(grid[r][k][1] for r, k in mk))
            for r, k in mk:
                grid[r][k] = ("A", 0)
        while True:                             # 3. fall, to a fixpoint
            moved = False
            for k in range(4):
                for r in range(sh - 2, -1, -1):
                    v = grid[r][k]
                    if v is not None and v[0] == "C" and grid[r + 1][k] is None:
                        grid[r + 1][k] = v
                        grid[r][k] = None
                        moved = True
            if not moved:
                break
            changed = True
        if not changed:
            break
    out = (tuple(tuple(grid[r][k][1] for r in range(sh - 1, -1, -1)
                       if grid[r][k] is not None and grid[r][k][0] == "C")
                 for k in range(4)),
           tuple(rounds), ticks)
    _CASCADE[memo] = out
    return out


# ---------------------------------------------------------------------------
# The planning universe: one level, as bitmasks
# ---------------------------------------------------------------------------

class _Level:
    """Board geometry plus the candy / ninja inventory of ONE planning state.

    Built fresh from whatever the engine currently holds, so it is as valid
    after a perturbation as at a level start. Everything the search does per
    node is a table lookup off this: `adj` is the press geometry (which
    columns a press from a square feeds), `ray` is the kill geometry (which
    ninjas each direction reaches, nearest first).

    A search state is ``(pos, cmask, nmask, stacks)`` with pos an int, cmask /
    nmask bitmasks over `cand_cells` / `ninja_cells`, and stacks the four
    columns. `pack` folds one into a single int for the dictionaries the
    enumeration keeps -- 800k tuple-of-tuples states is most of a gigabyte,
    800k ints is not."""

    def __init__(self, board, cands: dict, ninjas: dict):
        self.board = board
        self.h, self.w = board.h, board.w
        self.sh = board.sh
        self.cand_cells = tuple(sorted(cands))
        self.cand_col = tuple(cands[c] for c in self.cand_cells)
        self.ninja_cells = tuple(sorted(ninjas))
        self.ninja_col = tuple(ninjas[c] for c in self.ninja_cells)
        self.nc, self.nn = len(self.cand_cells), len(self.ninja_cells)
        cand_of = {c: i for i, c in enumerate(self.cand_cells)}
        ninja_of = {c: i for i, c in enumerate(self.ninja_cells)}
        self.cand_at = {self.idx(c): i for c, i in cand_of.items()}
        self.ninja_at = {self.idx(c): i for c, i in ninja_of.items()}
        self.static = frozenset(self.idx(c) for c in board.blocked)

        # press geometry: standing on `p`, which (column, candy) each rule feeds
        self.adj: dict[int, tuple] = {}
        for r in range(self.h):
            for c in range(self.w):
                hits = []
                for _name, (dr, dc), k in _ADD:
                    nb = cand_of.get((r + dr, c + dc))
                    if nb is not None:
                        hits.append((board.dropcol[k], nb))
                if hits:
                    self.adj[self.idx((r, c))] = tuple(hits)

        # kill geometry: the ninjas each of the four rays from `p` reaches,
        # NEAREST FIRST. Walls are absent on purpose -- the ellipsis cells of
        # ``[ Player | ... | NinjaR ]`` are unconstrained, so a beam crosses
        # everything, which --selfcheck measures rather than assumes.
        self.ray: dict[int, tuple] = {}
        if self.nn:
            for r in range(self.h):
                for c in range(self.w):
                    rays = []
                    for dr, dc in ((0, -1), (-1, 0), (1, 0), (0, 1)):
                        seen = []
                        rr, cc = r + dr, c + dc
                        while 0 <= rr < self.h and 0 <= cc < self.w:
                            hit = ninja_of.get((rr, cc))
                            if hit is not None:
                                seen.append(hit)
                            rr += dr
                            cc += dc
                        rays.append(tuple(seen))
                    self.ray[self.idx((r, c))] = tuple(rays)

        self._pos_bits = max(1, (self.h * self.w).bit_length())
        self._stack_shift = self._pos_bits + self.nc + self.nn

    # -- coordinates ------------------------------------------------------
    def idx(self, cell) -> int:
        return cell[0] * self.w + cell[1]

    def cell(self, i: int) -> tuple[int, int]:
        return divmod(i, self.w)

    # -- state packing ----------------------------------------------------
    def pack(self, pos: int, cmask: int, nmask: int, stacks: tuple) -> int:
        code = 0
        for k in range(3, -1, -1):
            v = 0
            for colour in reversed(stacks[k]):
                v = v * 5 + colour + 1
            code = code * (5 ** (self.sh + 1)) + v
        return (((code << self.nn | nmask) << self.nc | cmask)
                << self._pos_bits | pos)

    def unpack(self, key: int):
        pos = key & ((1 << self._pos_bits) - 1)
        rest = key >> self._pos_bits
        cmask = rest & ((1 << self.nc) - 1) if self.nc else 0
        rest >>= self.nc
        nmask = rest & ((1 << self.nn) - 1) if self.nn else 0
        code = rest >> self.nn
        base = 5 ** (self.sh + 1)
        stacks = []
        for _k in range(4):
            v, code = code % base, code // base
            col = []
            while v:                     # low digit first == BOTTOM first,
                col.append(v % 5 - 1)    # which is the order `pack` wrote
                v //= 5
            stacks.append(tuple(col))
        return pos, cmask, nmask, tuple(stacks)

    # -- the two primitives ------------------------------------------------
    def occupied(self, cmask: int, nmask: int) -> set:
        """Squares the player may NOT stand on right now: the static scenery
        plus every candy and ninja still on the floor."""
        blocked = set(self.static)
        for i, cell in enumerate(self.cand_cells):
            if (cmask >> i) & 1:
                blocked.add(self.idx(cell))
        for i, cell in enumerate(self.ninja_cells):
            if (nmask >> i) & 1:
                blocked.add(self.idx(cell))
        return blocked

    def walk(self, pos: int, blocked: set) -> dict:
        """Steps from `pos` to every square reachable without pressing."""
        dist = {pos: 0}
        queue = deque((pos,))
        w, h = self.w, self.h
        while queue:
            p = queue.popleft()
            d = dist[p] + 1
            r, c = divmod(p, w)
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                q = rr * w + cc
                if q in dist or q in blocked:
                    continue
                dist[q] = d
                queue.append(q)
        return dist

    def press(self, pos: int, cmask: int, nmask: int, stacks: tuple):
        """The state one X press leaves behind, or None if it changes nothing.

        Every rule that fires does so here: the four add rules (each gated on
        its own column's top cell), the cascade, and the kills the cascade's
        match rounds pay for."""
        hits = self.adj.get(pos)
        if not hits:
            return None
        drops = [None, None, None, None]
        taken = 0
        for column, cand in hits:
            if not (cmask >> cand) & 1:
                continue                       # already collected
            if drops[column] is not None or len(stacks[column]) >= self.sh:
                continue                       # `[ DropN no Candy ]` says no
            drops[column] = self.cand_col[cand]
            taken |= 1 << cand
        if not taken:
            return None
        out, rounds, ticks = _cascade(self.sh, stacks, tuple(drops))
        nmask2 = nmask
        if rounds and nmask:
            rays = self.ray[pos]
            for cleared in rounds:
                for colour in cleared:
                    for ray in rays:
                        for ninja in ray:
                            if ((nmask2 >> ninja) & 1
                                    and self.ninja_col[ninja] == colour):
                                nmask2 &= ~(1 << ninja)
                                break
        # A dead ninja leaves a Candy0 on the FLOOR, six more `again` ticks
        # that change nothing in the scroll. Charged here so the budget check
        # stays conservative without polluting the cascade memo.
        if nmask2 != nmask:
            ticks += 7
        if ticks >= MAX_AGAIN:
            return None                        # would come back mid-animation
        return (cmask & ~taken, nmask2, out)


def _press_ticks(level: "_Level", pos: int, cmask: int, stacks: tuple) -> int:
    """`again` iterations one press would cost, for the budget check in
    `--selfcheck`. Kept out of `_Level.press`, which is the hot path and has
    no use for the number once the cap is known to be far away."""
    hits = level.adj.get(pos)
    if not hits:
        return 1
    drops = [None, None, None, None]
    for column, cand in hits:
        if not (cmask >> cand) & 1:
            continue
        if drops[column] is not None or len(stacks[column]) >= level.sh:
            continue
        drops[column] = level.cand_col[cand]
    if all(d is None for d in drops):
        return 1
    return _cascade(level.sh, stacks, tuple(drops))[2]


def _stand(level: "_Level", cell, dr: int, dc: int):
    """The square you must stand on for `cell` to be in direction (dr, dc),
    or None when that square is off the board.

    The bounds check is not decoration: the flat index of `(r, c-1)` at c == 0
    is the PREVIOUS ROW's last column, so skipping it would silently let a
    press be planned from the wrong side of the map. (It happens to be a wall
    in every level of this game, which is exactly the kind of luck that stops
    being luck the moment someone edits a level.)"""
    r, c = cell[0] - dr, cell[1] - dc
    if 0 <= r < level.h and 0 <= c < level.w:
        return r * level.w + c
    return None


def _won(cmask: int, nmask: int, stacks: tuple) -> bool:
    return not cmask and not nmask and not any(stacks)


def _alive(level: _Level, cmask: int, nmask: int, stacks: tuple) -> bool:
    """False for states that provably can never win. All three tests are
    sound, which matters: `--prove`'s "this level has no win" rests on them.

      * a colour with one or two candies left in the world can never form a
        line, so it can never leave the scroll -- and if it is still on the
        floor it can never leave the floor either;
      * a ninja whose colour has fewer than three candies left has no match
        round coming;
      * with no floor candies left, nothing on the board can change again."""
    total = [0, 0, 0, 0]
    for i, colour in enumerate(level.cand_col):
        if (cmask >> i) & 1:
            total[colour] += 1
    for col in stacks:
        for colour in col:
            total[colour] += 1
    live = [0, 0, 0, 0]
    for i, colour in enumerate(level.ninja_col):
        if (nmask >> i) & 1:
            live[colour] += 1
    for colour in range(4):
        if 0 < total[colour] < 3:
            return False
        if live[colour] and total[colour] < 3:
            return False
    if not cmask and (any(stacks) or nmask):
        return False
    return True


def _successors(level: _Level, pos: int, cmask: int, nmask: int, stacks: tuple,
                keep: int = 0):
    """``[(walk + 1, press square, next state)]`` -- one entry per square worth
    pressing from.

    The candidates are the four squares around each surviving candy, which is
    complete: a press that collects nothing changes nothing, and walking
    changes nothing but where you stand, so every distinct board a turn can
    produce is produced by one of these.

    ``keep`` trims to the nearest few (favouring presses that collect more or
    kill more), which is what makes the crowded levels searchable at all -- it
    is a beam knob and is left at 0 for every exhaustive use."""
    blocked = level.occupied(cmask, nmask)
    dist = level.walk(pos, blocked)
    out = []
    seen = set()
    for i, cell in enumerate(level.cand_cells):
        if not (cmask >> i) & 1:
            continue
        for _name, (dr, dc), _k in _ADD:
            p = _stand(level, cell, dr, dc)
            step = None if p is None else dist.get(p)
            if step is None or p in seen:
                continue
            seen.add(p)
            nxt = level.press(p, cmask, nmask, stacks)
            if nxt is None:
                continue
            out.append((step + 1, p, nxt))
    if keep and len(out) > keep:
        out.sort(key=lambda o: (o[0]
                                - 3 * bin(cmask ^ o[2][0]).count("1")
                                - 4 * bin(nmask ^ o[2][1]).count("1"), o[1]))
        out = out[:keep]
    return out


# ---------------------------------------------------------------------------
# Exact: the whole macro space, and the distance to a win in it
# ---------------------------------------------------------------------------

INF = float("inf")


class _Space:
    """Every macro state a level can reach, and the exact steps-to-win of each.

    The graph is a DAG layered by how many floor candies are left -- a useful
    press always collects at least one and nothing ever puts one back -- so no
    priority queue is needed: enumerate forward, then sweep the layers from
    fewest candies to most, and each state's cost is the best of its own
    edges, all of which point at a layer already done.

    Costs are real steps (walk + the press), so `dist` is directly the number
    of buttons an agent has left to press, and the tie sets fall out of it."""

    def __init__(self, level: _Level, dist: dict):
        self.level = level
        self.dist = dist

    @classmethod
    def build(cls, level: _Level, pos: int, cmask: int, nmask: int,
              stacks: tuple, cap: int = 900_000, seconds: float = 150.0):
        """The space, or None when it does not fit in ``cap`` states /
        ``seconds``. Both bounds are real: level 8 is ~806k states and about
        three minutes of the two passes, and the levels that do not fit are
        past a million by a wide margin."""
        t0 = time.time()
        start = level.pack(pos, cmask, nmask, stacks)
        seen = {start}
        layers: dict[int, list[int]] = {}
        queue = deque(((pos, cmask, nmask, stacks),))
        layers.setdefault(bin(cmask).count("1"), []).append(start)
        while queue:
            p, cm, nm, st = queue.popleft()
            if _won(cm, nm, st):
                continue
            for _cost, np_, (cm2, nm2, st2) in _successors(level, p, cm, nm, st):
                if not _alive(level, cm2, nm2, st2):
                    continue
                key = level.pack(np_, cm2, nm2, st2)
                if key in seen:
                    continue
                seen.add(key)
                layers.setdefault(bin(cm2).count("1"), []).append(key)
                queue.append((np_, cm2, nm2, st2))
                if len(seen) > cap or time.time() - t0 > seconds:
                    return None
        dist: dict[int, float] = {}
        for depth in sorted(layers):
            for key in layers[depth]:
                p, cm, nm, st = level.unpack(key)
                if _won(cm, nm, st):
                    dist[key] = 0
                    continue
                best = INF
                for cost, np_, (cm2, nm2, st2) in _successors(level, p, cm, nm, st):
                    got = dist.get(level.pack(np_, cm2, nm2, st2))
                    if got is not None and cost + got < best:
                        best = cost + got
                dist[key] = best
        return cls(level, dist)

    # -- the field an agent standing anywhere is judged against -------------
    def field(self, cmask: int, nmask: int, stacks: tuple):
        """``(steps-to-win per square, value of pressing here)`` for one board.

        Pressing is only ever done from a square next to a candy, so seed a
        Dijkstra with ``1 + dist(result)`` at each such square and relax it
        over the walk graph: the answer at a square is exactly "walk to the
        best press, press, finish", which is the true remaining cost from
        standing there. One sweep answers both the plan's next step and its
        whole tie set."""
        level = self.level
        blocked = level.occupied(cmask, nmask)
        press: dict[int, float] = {}
        for i, cell in enumerate(level.cand_cells):
            if not (cmask >> i) & 1:
                continue
            for _name, (dr, dc), _k in _ADD:
                p = _stand(level, cell, dr, dc)
                if p is None or p in blocked or p in press:
                    continue
                nxt = level.press(p, cmask, nmask, stacks)
                if nxt is None:
                    continue
                got = self.dist.get(level.pack(p, *nxt), INF)
                if got < INF:
                    press[p] = 1 + got
        # Relax the seeds over the walk graph (unit edges, integer seeds):
        # a label-correcting sweep, which converges to the true minimum.
        best = dict(press)
        order = deque(sorted(press, key=press.get))
        while order:
            p = order.popleft()
            d = best[p] + 1
            r, c = divmod(p, level.w)
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if not (0 <= rr < level.h and 0 <= cc < level.w):
                    continue
                q = rr * level.w + cc
                if q in blocked or best.get(q, INF) <= d:
                    continue
                best[q] = d
                order.append(q)
        return best, press


# ---------------------------------------------------------------------------
# Beam: the levels the space does not fit
# ---------------------------------------------------------------------------

#: (width, successor cap, candy weight, scroll weight, ninja weight, step
#: weight, tie jitter). Tried in order; the first that wins is kept. The
#: spread is deliberate -- "clear the scroll" and "collect the floor" pull
#: against each other and different levels want different mixes, so a
#: portfolio finishes levels no single setting does.
#:
#: The last two repeat earlier weightings with a JITTER on the beam's sort
#: key, which is not a decoration: level 24's frontier is full of exact ties
#: (a symmetric board of four colours), and which of them the sort happens to
#: keep decides whether the level is solved at all. The jitter is drawn from a
#: `random.Random` seeded with the configuration's own index, so the portfolio
#: is still deterministic run to run.
_BEAM_CFGS = (
    (3000, 0, 1.0, 1.0, 1.0, 0.00, 0.0),
    (4000, 16, 1.0, 1.5, 1.0, 0.00, 0.0),
    (4000, 16, 1.0, 1.0, 3.0, 0.00, 0.0),
    (3000, 24, 1.0, 2.0, 2.0, 0.02, 0.0),
    (2500, 40, 1.0, 0.5, 1.0, 0.00, 0.0),
    (4000, 16, 1.0, 1.5, 1.0, 0.00, 0.01),
    (6000, 12, 1.0, 1.0, 1.0, 0.00, 0.01),
)

#: Total successor expansions one beam configuration may spend. This is the
#: memory bound as much as the time bound -- the visited table is what grows.
_BEAM_NODES = 4_000_000


def _beam(level: _Level, pos: int, cmask: int, nmask: int, stacks: tuple,
          cfg, index: int = 0, nodes: int = _BEAM_NODES):
    """A macro path to a win, or None. Layer-synchronous: expand the whole
    frontier, drop everything that cannot win, keep the best `width`."""
    width, keep, wc, ws, wn, wg, jitter = cfg
    rng = random.Random(index)
    start = level.pack(pos, cmask, nmask, stacks)
    seen = {start: 0}
    parent: dict[int, tuple] = {start: None}
    frontier = [(0, start, (pos, cmask, nmask, stacks))]
    spent = 0
    for _depth in range(400):
        nxt = []
        for g, key, (p, cm, nm, st) in frontier:
            if _won(cm, nm, st):
                path = []
                while parent[key] is not None:
                    key, square = parent[key]
                    path.append(square)
                return list(reversed(path)), g
            for cost, np_, (cm2, nm2, st2) in _successors(level, p, cm, nm, st, keep):
                spent += 1
                if spent > nodes:
                    return None
                if not _alive(level, cm2, nm2, st2):
                    continue
                nkey = level.pack(np_, cm2, nm2, st2)
                ng = g + cost
                if seen.get(nkey, 1 << 30) <= ng:
                    continue
                seen[nkey] = ng
                parent[nkey] = (key, np_)
                nxt.append((ng, nkey, (np_, cm2, nm2, st2)))
        if not nxt:
            return None
        if len(nxt) > width:
            nxt.sort(key=lambda item: (
                wc * bin(item[2][1]).count("1")
                + ws * sum(len(col) for col in item[2][3])
                + wn * bin(item[2][2]).count("1")
                + wg * item[0]
                + (rng.random() * jitter if jitter else 0.0)))
            nxt = nxt[:width]
        frontier = nxt
    return None


# ---------------------------------------------------------------------------
# Macro path -> engine presses, with the tie set for every one of them
# ---------------------------------------------------------------------------

def _walk_route(level: _Level, pos: int, target: int, blocked: set):
    """Steps from `pos` to `target` as ``[(direction, tie set)]``.

    The tie set is every direction whose square is one step closer -- i.e.
    every equally short route, which is what stops the recording from teaching
    one arbitrary interleaving of two axes as the only right answer. It is a
    SUBSET of the true optimal set (there may be a different press square that
    is equally good), never a superset, so nothing suboptimal is ever labelled
    optimal."""
    back = level.walk(target, blocked)
    out = []
    here = pos
    while here != target:
        d0 = back.get(here)
        if d0 is None:
            return None
        tie = []
        step = None
        for name in _STEPS:
            dr, dc = _DELTA[name]
            r, c = divmod(here, level.w)
            rr, cc = r + dr, c + dc
            if not (0 <= rr < level.h and 0 <= cc < level.w):
                continue
            q = rr * level.w + cc
            if q in blocked:
                continue
            if back.get(q, INF) == d0 - 1:
                tie.append(name)
                if step is None:
                    step = (name, q)
        if step is None:
            return None
        out.append((step[0], tie))
        here = step[1]
    return out


def _neighbour(level: _Level, pos: int, name: str):
    dr, dc = _DELTA[name]
    r, c = divmod(pos, level.w)
    rr, cc = r + dr, c + dc
    if 0 <= rr < level.h and 0 <= cc < level.w:
        return rr * level.w + cc
    return None


def _expand_exact(space: _Space, pos: int, cmask: int, nmask: int,
                  stacks: tuple):
    """The shortest press sequence, with the TRUE optimal set for every step.

    No macro path is needed: the field says what every square is worth, so the
    plan is "take any action that drops the count by one" and the tie set IS
    the set of such actions. A blocked move is absent from the field, so it can
    never tie. The field only changes when a press does, so it is recomputed
    per press rather than per step."""
    level = space.level
    presses: list[str] = []
    optsets: list[list[str]] = []
    best, press = space.field(cmask, nmask, stacks)
    for _ in range(4 * level.h * level.w + 400):
        if _won(cmask, nmask, stacks):
            return presses, optsets
        here = best.get(pos, INF)
        if here == INF:
            return None, None
        tie = [name for name in _STEPS
               if best.get(_neighbour(level, pos, name), INF) == here - 1]
        if press.get(pos, INF) == here:
            tie.append("action")
        if not tie:
            return None, None
        take = "action" if "action" in tie else tie[0]
        presses.append(take)
        optsets.append(tie)
        if take == "action":
            nxt = level.press(pos, cmask, nmask, stacks)
            if nxt is None:
                return None, None
            cmask, nmask, stacks = nxt
            best, press = space.field(cmask, nmask, stacks)
        else:
            pos = _neighbour(level, pos, take)
    return None, None


def _expand_beam(level: _Level, pos: int, cmask: int, nmask: int,
                 stacks: tuple, path):
    """A macro path as ``(presses, optsets)``. No field exists here, so a walk
    is labelled with its own route ties and a press with itself."""
    presses: list[str] = []
    optsets: list[list[str]] = []
    for target in path:
        route = _walk_route(level, pos, target, level.occupied(cmask, nmask))
        if route is None:
            return None, None
        for name, tie in route:
            presses.append(name)
            optsets.append(tie)
        presses.append("action")
        optsets.append(["action"])
        nxt = level.press(target, cmask, nmask, stacks)
        if nxt is None:
            return None, None
        cmask, nmask, stacks = nxt
        pos = target
    return presses, optsets


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class CandyScrollExpert(PSExpert):
    """Plans from the LIVE grid: read the board, build the level, solve it.

    The per-level `_Space` (when one fits) is kept for the process's lifetime,
    so the expensive half is paid once no matter how many seeds are recorded;
    `plan_cache_path` carries the answer between processes as well, which is
    what makes a `parallelize_generator` shard cheap."""

    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "candy_scroll_plans.json")

    #: Cap for `_Space.build`, in live states and in seconds. Level 16 is the
    #: largest that fits -- 994,345 states, 85s, 0.36GB -- and it is the reason
    #: the cap is where it is: at 900k it fell to the beam, which does not
    #: solve it at all. The levels that do NOT fit are far past this (18 is
    #: still going at 3.5M) so they bail on the state cap in under half a
    #: minute rather than on the clock.
    space_cap = 1_200_000
    space_seconds = 240.0

    def setup(self) -> None:
        self._spaces: dict[int, _Space | None] = {}
        self._exact_levels: set[int] = set()
        self._cur_level: int | None = None

    # -- reading -----------------------------------------------------------
    def read(self, eng):
        """``(level, pos, cmask, nmask, stacks)`` for the engine's grid."""
        board, pos, cands, ninjas, stacks = _read(eng, self.g)
        level = _Level(board, cands, ninjas)
        return (level, level.idx(pos),
                (1 << level.nc) - 1, (1 << level.nn) - 1, stacks)

    def heuristic(self, eng) -> int:            # never used: `_search` is ours
        return 0

    def plan(self, eng, level: int | None = None):
        self._cur_level = level
        return super().plan(eng, level)

    # -- solving -----------------------------------------------------------
    def _search(self, eng):
        board, pos, cands, ninjas, stacks = _read(eng, self.g)
        if not cands and not ninjas and not any(stacks):
            return Plan([], [])
        lvl = _Level(board, cands, ninjas)
        here = (lvl.idx(pos), (1 << lvl.nc) - 1, (1 << lvl.nn) - 1)
        if not _alive(lvl, here[1], here[2], stacks):
            return None

        space = self._space(lvl, here, stacks)
        if space is not None:
            seat = _reindex(space.level, pos, cands, ninjas)
            if seat is not None:
                presses, optsets = _expand_exact(space, *seat, stacks)
                if presses is None:
                    return None
                return Plan(presses, optsets)

        for index, cfg in enumerate(_BEAM_CFGS):
            found = _beam(lvl, here[0], here[1], here[2], stacks, cfg, index)
            if found is None:
                continue
            path, _cost = found
            presses, optsets = _expand_beam(lvl, here[0], here[1], here[2],
                                            stacks, path)
            if presses is not None:
                return Plan(presses, optsets)
        return None

    def _space(self, lvl: _Level, here, stacks) -> "_Space | None":
        """This level's exact space, built once and kept. None means the level
        did not fit the cap and is a beam level for the rest of the process."""
        key = self._cur_level
        if key is None:
            return None
        if key in self._spaces:
            return self._spaces[key]
        space = _Space.build(lvl, here[0], here[1], here[2], stacks,
                             cap=self.space_cap, seconds=self.space_seconds)
        self._spaces[key] = space
        if space is not None:
            self._exact_levels.add(key)
        return space


def _reindex(level: _Level, pos, cands: dict, ninjas: dict):
    """Express a live board in the bitmask indexing of an already-built
    `_Level`, or None when it holds something that level does not know about
    (which is how a state outside the enumerated space declines it)."""
    cmask = 0
    for cell, colour in cands.items():
        i = level.cand_at.get(level.idx(cell))
        if i is None or level.cand_col[i] != colour:
            return None
        cmask |= 1 << i
    nmask = 0
    for cell, colour in ninjas.items():
        i = level.ninja_at.get(level.idx(cell))
        if i is None or level.ninja_col[i] != colour:
            return None
        nmask |= 1 << i
    return level.idx(pos), cmask, nmask


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class CandyScrollSolver(PSAStarSolver):
    game_id = "puzzlescript_the_saga_of_the_candy_scroll"
    game_name = GAME_NAME
    expert_cls = CandyScrollExpert

    #: `games/ps:the_saga_of_the_candy_scroll/...py` is a plain passthrough --
    #: it builds the adapter and nothing else, and both rendering fixes are in
    #: the .txt, which that path reads too.
    game_module_id = ""

    #: Unused: `CandyScrollExpert._search` never calls `_astar`.
    weight = 1
    node_cap = 400_000

    #: Room for the longest plan (162 presses on level 30) inside the adapter's
    #: own 200-press per-level budget, which the `set_level` that ends the
    #: exploration prefix resets to zero.
    max_steps = 200

    #: Provably unwinnable under this interpreter -- see the module docstring
    #: and ``--prove``. Levels 12, 13, 17, 20 and 21 (0-based here) are built
    #: on being able to move while the scroll is still resolving, which
    #: `PSEngine.step` does not allow. Skipped up front so startup does not
    #: re-enumerate them; ``--prove`` is where the claim is re-measured.
    skip_levels = frozenset({11, 12, 16, 19, 20})

    #: Recovery is the family's RESET prefix. Mid-plan detours would cost a
    #: fresh beam search on the levels that have no exact space, so they stay
    #: off for the whole game rather than for some of it.
    epsilon = 0.0

    supports_recovery = True
    recovery_mode = "reset"


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = CandyScrollSolver()
    game = solver.make_game(seed)
    return solver, game, CandyScrollExpert(game, node_cap=solver.node_cap)


def _ascii(eng, game) -> str:
    n = game.obj_name_to_idx
    glyph = [("wall", "#"), ("player", "P"), ("candyr", "R"), ("candyg", "G"),
             ("candyb", "B"), ("candyi", "I"), ("ninjar", "r"), ("ninjag", "g"),
             ("ninjab", "b"), ("ninjai", "i"), ("scrollleft", "{"),
             ("scrollright", "}"), ("scrolltop", "^"), ("scrollbottom", "v"),
             ("scrollbg", "."), ("background", " ")]
    rows = []
    for r in range(eng.height):
        line = ""
        for c in range(eng.width):
            cell = eng.grid[r][c]
            line += next((ch for name, ch in glyph if n[name] in cell), " ")
        rows.append(line)
    return "\n".join(rows)


# -- audit ------------------------------------------------------------------

def _stacks_of_cells():
    """Every cell stack a SETTLED board of this game can hold, as object-name
    lists.

    Two deliberate absences.  The clear animation: `PSEngine.step` drains a
    press's cascade before `perform_action` returns, so no frame an agent ever
    sees holds a Candy0..Candy5 (which `--selfcheck` measures rather than
    assumes).  And Drop1..Drop4: their sprites are BLANK by design, so a drop
    cell is pixel-identical to the scroll cell under it, and auditing them
    would only re-report that.  No information is lost with them -- the drop
    row is the scroll's top row and the direction-to-column legend is the four
    arrows, which `_audit` checks separately."""
    out = [["background"], ["background", "wall"], ["background", "player"]]
    for suffix in COLOURS:
        out.append(["background", "candy" + suffix.lower()])
        out.append(["background", "ninja" + suffix.lower()])
    for name in ("scrollleft", "scrollright", "scrolltop", "scrollbottom"):
        out.append(["background", name])
    for name in ("arrowl", "arrowu", "arrowd", "arrowr"):
        out.append(["background", name])
    out.append(["background", "scrollbg"])
    for suffix in COLOURS:
        out.append(["background", "scrollbg", "candy" + suffix.lower()])
    return out


def _audit() -> int:
    """Render every cell stack at every board shape; all distinct?

    Whole frames are compared, not cell blocks: `_render_frame` upscales the
    board to fill 64x64, so the board origin is not where the arithmetic says
    it is and indexing a cell would be reading the wrong pixels."""
    _solver, game, _expert = _new()
    eng, psg = game._engine, game._game
    shapes = {}
    for lvl in range(game.n_levels):
        eng.load_level(psg.levels[lvl])
        shapes.setdefault((eng.height, eng.width), lvl)
    print(f"  board shapes: {sorted(shapes)}")
    bad = 0
    stacks = _stacks_of_cells()
    arrows = {"background+" + n for n in ("arrowl", "arrowu", "arrowd", "arrowr")}
    for (h, w), lvl in sorted(shapes.items()):
        eng.load_level(psg.levels[lvl])
        base = [[set(cell) for cell in row] for row in eng.grid]
        spot = (h // 2, w - 2)
        frames = {}
        for stack in stacks:
            eng.grid = [[set(cell) for cell in row] for row in base]
            eng._position_index_dirty = True
            eng.grid[spot[0]][spot[1]] = {psg.obj_name_to_idx[o] for o in stack}
            key = game._present_frame(_render_frame(eng, psg)).tobytes()
            frames.setdefault(key, []).append("+".join(stack))
        eng.grid = base
        cell_px = max(1, min(64 // h, 64 // w))
        clash = [names for names in frames.values() if len(names) > 1]
        bad += len(clash)
        # the direction-to-column legend has to survive too
        legend = sum(1 for names in frames.values() if arrows & set(names))
        note = ("all distinct" if not clash
                else "COLLISIONS: " + str(clash))
        if legend != 4:
            note += "  -- and the four ARROWS are not 4 distinct pictures"
            bad += 1
        print(f"  {h:2d}x{w:-2d} (cell_px {cell_px}): {len(stacks)} stacks, {note}")
    print("  every stack reads differently at every shape" if not bad
          else f"  {bad} COLLIDING GROUP(S)")
    return 1 if bad else 0


# -- selfcheck --------------------------------------------------------------

def _seat(game, board, pos, cands, ninjas, stacks):
    """Write a model state into the engine grid (leaving scenery alone)."""
    eng, psg = game._engine, game._game
    n = psg.obj_name_to_idx
    movable = ([n["candy" + s.lower()] for s in COLOURS]
               + [n["ninja" + s.lower()] for s in COLOURS]
               + [n[f"candy{i}"] for i in range(6)]
               + [n["clearmarker"], n["player"]])
    for r in range(board.h):
        for c in range(board.w):
            for idx in movable:
                eng.grid[r][c].discard(idx)
    eng.grid[pos[0]][pos[1]].add(n["player"])
    for (r, c), k in cands.items():
        eng.grid[r][c].add(n["candy" + COLOURS[k].lower()])
    for (r, c), k in ninjas.items():
        eng.grid[r][c].add(n["ninja" + COLOURS[k].lower()])
    for k in range(4):
        col = board.scols[k]
        for i, colour in enumerate(stacks[k]):
            eng.grid[board.srows[board.sh - 1 - i]][col].add(
                n["candy" + COLOURS[colour].lower()])
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _model_step(board, pos, cands, ninjas, stacks, action):
    """The native model's answer for one engine action, in the engine's own
    ``(pos, cands, ninjas, stacks)`` vocabulary -- the shape --selfcheck can
    compare against `_read`."""
    if action != "action":
        dr, dc = _DELTA[action]
        nxt = (pos[0] + dr, pos[1] + dc)
        if (0 <= nxt[0] < board.h and 0 <= nxt[1] < board.w
                and nxt not in board.blocked and nxt not in cands
                and nxt not in ninjas):
            pos = nxt
        return pos, cands, ninjas, stacks
    level = _Level(board, cands, ninjas)
    got = level.press(level.idx(pos), (1 << level.nc) - 1,
                      (1 << level.nn) - 1, stacks)
    if got is None:
        return pos, cands, ninjas, stacks
    cmask, nmask, out = got
    cands = {cell: level.cand_col[i] for i, cell in enumerate(level.cand_cells)
             if (cmask >> i) & 1}
    ninjas = {cell: level.ninja_col[i] for i, cell in enumerate(level.ninja_cells)
              if (nmask >> i) & 1}
    return pos, cands, ninjas, out


def _selfcheck(quick: bool = False) -> int:
    _solver, game, _expert = _new()
    eng, psg = game._engine, game._game
    rng = random.Random(7)
    bad = 0
    longest = 0

    # pass 0 -- `_Level.pack` / `_Level.unpack` round-trip. The enumeration
    # keeps 800k states as ints and the DP reads them back; a packing that
    # loses the order of a column would silently give every level a wrong
    # distance field, which is exactly what it did once.
    trips = 0
    for lvl in range(game.n_levels):
        eng.load_level(psg.levels[lvl])
        board, pos, cands, ninjas, _stacks = _read(eng, psg)
        level = _Level(board, cands, ninjas)
        for _ in range(200):
            state = (rng.randrange(board.h * board.w),
                     rng.getrandbits(max(1, level.nc)) & ((1 << level.nc) - 1),
                     rng.getrandbits(max(1, level.nn)) & ((1 << level.nn) - 1),
                     tuple(tuple(rng.randrange(4)
                                 for _ in range(rng.randrange(board.sh + 1)))
                           for _ in range(4)))
            trips += 1
            if level.unpack(level.pack(*state)) != state:
                bad += 1
                print(f"    PACK ROUND-TRIP FAILED, level {lvl + 1}: {state}")
                break
    print(f"  pass 0  state packing round-trip: {trips} states, "
          f"{bad} failure(s)")

    # pass 1 -- random play from randomly refilled scrolls, all 30 levels
    trials, steps = (4, 20) if quick else (12, 40)
    total = 0
    for lvl in range(game.n_levels):
        eng.load_level(psg.levels[lvl])
        board, pos0, cands0, ninjas0, stacks0 = _read(eng, psg)
        floor = [(r, c) for r in range(board.h) for c in range(board.w)
                 if (r, c) not in board.blocked and (r, c) not in board.scell]
        for trial in range(trials):
            if trial:
                pos = rng.choice(floor)
                cands = {k: v for k, v in cands0.items()
                         if k != pos and rng.random() < .5}
                ninjas = {k: v for k, v in ninjas0.items()
                          if k != pos and rng.random() < .6}
                stacks = tuple(tuple(rng.randrange(4)
                                     for _ in range(rng.randrange(board.sh + 1)))
                               for _ in range(4))
                _seat(game, board, pos, cands, ninjas, stacks)
                eng.continue_animation()          # settle the injected board
                _b, pos, cands, ninjas, stacks = _read(eng, psg)
            else:
                pos, cands, ninjas, stacks = pos0, cands0, ninjas0, stacks0
            _seat(game, board, pos, cands, ninjas, stacks)
            if not _settled(eng, psg):
                continue
            for _ in range(steps):
                act = rng.choice(["left", "up", "down", "right",
                                  "action", "action"])
                want = _model_step(board, pos, cands, ninjas, stacks, act)
                eng.step(act)
                if not _settled(eng, psg):
                    bad += 1
                    print(f"    UNSETTLED after one press, level {lvl + 1}")
                    break
                _b, pos, cands, ninjas, stacks = _read(eng, psg)
                total += 1
                if want != (pos, cands, ninjas, stacks):
                    bad += 1
                    print(f"    MISMATCH level {lvl + 1} action {act}\n"
                          f"      model {want}\n      engine {(pos, cands, ninjas, stacks)}")
                    break
    print(f"  pass 1  random play, refilled scrolls: {total} presses, "
          f"{bad} mismatch(es)")

    # pass 2 -- every candy, from every legal side, on every level
    was = bad
    tested = 0
    for lvl in range(game.n_levels):
        eng.load_level(psg.levels[lvl])
        board, _pos, cands0, ninjas0, stacks0 = _read(eng, psg)
        for cell in list(cands0):
            for _name, (dr, dc), _k in _ADD:
                stand = (cell[0] - dr, cell[1] - dc)
                if not (0 <= stand[0] < board.h and 0 <= stand[1] < board.w):
                    continue
                if (stand in board.blocked or stand in cands0
                        or stand in ninjas0 or stand in board.scell):
                    continue
                _seat(game, board, stand, cands0, ninjas0, stacks0)
                want = _model_step(board, stand, dict(cands0), dict(ninjas0),
                                   stacks0, "action")
                eng.step("action")
                tested += 1
                if not _settled(eng, psg):
                    bad += 1
                    continue
                got = _read(eng, psg)[1:]
                if want != got:
                    bad += 1
                    if bad - was < 4:
                        print(f"    MISMATCH level {lvl + 1} standing {stand}\n"
                              f"      model {want}\n      engine {got}")
    print(f"  pass 2  every candy from every side: {tested} presses, "
          f"{bad - was} mismatch(es)")

    # pass 3 -- 1-4 candies around the player, random scrolls, random ninjas.
    # The only pass that reaches the multi-drop match at the drop row and the
    # two-ninjas-on-one-ray case; both were model bugs before it existed.
    was = bad
    tested = 0
    rounds = 60 if quick else 220
    for lvl in range(game.n_levels):
        eng.load_level(psg.levels[lvl])
        board, _p, _c, _n, _s = _read(eng, psg)
        floor = [(r, c) for r in range(board.h) for c in range(board.w)
                 if (r, c) not in board.blocked and (r, c) not in board.scell]
        for _ in range(rounds):
            pos = rng.choice(floor)
            cands = {}
            for _name, (dr, dc), _k in _ADD:
                nb = (pos[0] + dr, pos[1] + dc)
                if (0 <= nb[0] < board.h and 0 <= nb[1] < board.w
                        and nb not in board.blocked and nb not in board.scell
                        and rng.random() < .7):
                    cands[nb] = rng.randrange(4) if rng.random() < .45 else 0
            ninjas = {}
            for _ in range(rng.randrange(4)):
                nb = rng.choice(floor)
                if nb != pos and nb not in cands:
                    ninjas[nb] = rng.randrange(4) if rng.random() < .5 else 0
            stacks = tuple(tuple(rng.randrange(4) if rng.random() < .4 else 0
                                 for _ in range(rng.randrange(board.sh + 1)))
                           for _ in range(4))
            _seat(game, board, pos, cands, ninjas, stacks)
            eng.continue_animation()
            _b, pos, cands, ninjas, stacks = _read(eng, psg)
            _seat(game, board, pos, cands, ninjas, stacks)
            if not _settled(eng, psg):
                continue
            level = _Level(board, cands, ninjas)
            longest = max(longest, _press_ticks(
                level, level.idx(pos), (1 << level.nc) - 1, stacks))
            want = _model_step(board, pos, dict(cands), dict(ninjas), stacks,
                               "action")
            eng.step("action")
            tested += 1
            if not _settled(eng, psg):
                bad += 1
                print(f"    UNSETTLED after one press, level {lvl + 1}")
                continue
            got = _read(eng, psg)[1:]
            if want != got:
                bad += 1
                if bad - was < 4:
                    print(f"    MISMATCH level {lvl + 1} at {pos}\n"
                          f"      model {want}\n      engine {got}")
    print(f"  pass 3  1-4 candies around the player: {tested} presses, "
          f"{bad - was} mismatch(es)")
    print(f"  longest cascade seen: {longest} `again` ticks "
          f"(PSEngine.step allows {MAX_AGAIN}, so no press is ever clipped)")
    print(f"  the model agrees with the interpreter on every board it was shown"
          if not bad else f"  {bad} DISAGREEMENT(S)")
    return 1 if bad else 0


# -- prove ------------------------------------------------------------------

def _prove(nodes: int = 1_500_000) -> int:
    """Enumerate the skipped levels' macro spaces WHOLE and report that none
    holds a win. A run that stops on the node cap says so rather than claiming
    a proof."""
    _solver, game, expert = _new()
    eng, psg = game._engine, game._game
    bad = 0
    for lvl in sorted(CandyScrollSolver.skip_levels):
        eng.load_level(psg.levels[lvl])
        print(f"  --- level {lvl + 1}"
              " ('#' wall, 'P' you, RGBI candy, rgbi ninja, '.' scroll) ---")
        for line in _ascii(eng, psg).splitlines():
            print("      " + line)
        level, pos, cmask, nmask, stacks = expert.read(eng)
        t0 = time.time()
        space = _Space.build(level, pos, cmask, nmask, stacks,
                             cap=nodes, seconds=900.0)
        if space is None:
            print(f"  L{lvl + 1:2d}: did NOT fit in {nodes} states -- no claim")
            bad += 1
            continue
        start = level.pack(pos, cmask, nmask, stacks)
        reach = space.dist.get(start, INF)
        wins = sum(1 for v in space.dist.values() if v == 0)
        print(f"  L{lvl + 1:2d}: {len(space.dist):7d} live states enumerated "
              f"whole in {time.time() - t0:5.1f}s, {wins} of them won, "
              f"D(start) = {'unreachable' if reach == INF else reach}")
        if reach != INF:
            print("      ^ THIS LEVEL IS WINNABLE -- take it out of skip_levels")
            bad += 1
    print("  none of the skipped levels holds a win" if not bad
          else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


# -- plans ------------------------------------------------------------------

def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the model, the field and the tie labelling at once."""
    solver, game, expert = _new()
    eng, psg = game._engine, game._game
    total = ties = bad = 0
    solved = []
    for lvl in range(game.n_levels):
        if lvl in solver.skip_levels:
            print(f"  L{lvl + 1:2d}: skipped (proved unwinnable -- see --prove)")
            continue
        game.set_level(lvl)
        level, pos, cmask, nmask, stacks = expert.read(eng)
        t0 = time.time()
        plan = expert.plan(eng, lvl)
        took = time.time() - t0
        if plan is None:
            print(f"  L{lvl + 1:2d}: UNSOLVED  ({took:5.1f}s)")
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tie_steps = sum(1 for s in sets if len(s) > 1)
        exact = lvl in expert._exact_levels
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"  L{lvl + 1:2d}: {level.h:2d}x{level.w:-2d}  "
              f"{level.nc:2d} candy {level.nn:2d} ninja "
              f"{sum(len(c) for c in stacks):2d} in scroll  "
              f"{len(plan):3d} presses  win={won}  "
              f"({'SHORTEST' if exact else 'beam'}, budget {game._max_steps}, "
              f"{room})  {tie_steps:3d} steps with a tie set  {took:5.1f}s")
        total += len(plan)
        ties += tie_steps
        solved.append(lvl)
    print(f"  {len(solved)} levels solved, {total} presses, "
          f"{ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _report() -> int:
    rc = 0
    print("== audit: every cell stack, every board shape")
    rc |= _audit()
    print("== selfcheck: the native model against the interpreter")
    rc |= _selfcheck()
    print("== prove: the skipped levels have no win")
    rc |= _prove()
    print("== plans: every level, replayed through the interpreter")
    rc |= _plans()
    return rc


def main() -> None:
    quick = "--quick" in sys.argv          # a third of the fuzz, for a smoke test
    flags = {"--audit": _audit,
             "--selfcheck": lambda: _selfcheck(quick),
             "--prove": _prove, "--plans": _plans, "--report": _report}
    for flag, fn in flags.items():
        if flag in sys.argv:
            sys.exit(fn())
    CandyScrollSolver.main()


if __name__ == "__main__":
    main()
