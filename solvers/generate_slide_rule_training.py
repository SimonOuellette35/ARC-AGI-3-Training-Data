"""Generate Phase-1 training data for the PuzzleScript game ps:slide_rule
("Slide Rule", Jim Palmeri).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a NATIVE model of the
mechanic, the exact distance-to-win field built on it, the differential fuzz that
certifies the model against the real interpreter, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema::

    {
      "game_id": "puzzlescript_slide_rule",
      "levels": [
        {"level_id": 0,
         "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
         "actions":      [a_0, ..., a_{T-1}]},     # length T
        ...
      ]
    }

``actions[0]`` is the RESET that produced ``obs[0]``; ``actions[i]`` for ``i>=1``
is the action that took the agent from ``obs[i-1]`` to ``obs[i]``. The recorded
index is the *screen* action (post rotation/flip remap), so replaying the
recorded actions reproduces the recorded frames exactly. Every expert step
carries the full set of equally-optimal presses.

The game
--------
You do not step, you SLIDE: one press launches the player and he travels until
something stops him. Scattered about are CRATES tethered to their starting
square by a rope. A crate shows two digits ``cur/max``: ``max`` is how long its
rope can pay out and ``cur`` is how much slack is left. Shoving a crate spends
one unit of slack and lays one link of rope; shoving it back along its own rope
reels a link in. A crate with no slack left can only be reeled back. Reach the
finish.

Everything below was MEASURED against the interpreter, not read off the .txt.

* **A press is a whole slide.** ``left [ > Player no PMarker | no Solid ] ->
  [ Player PLeft | ]`` stamps a direction marker, and the marker rule
  ``left [ Player PLeft | no FixedL ] -> [ > Player > PLeft | ] again`` re-fires
  one cell per ``again`` tick until something ahead is *fixed*.
* **A heavy player slides THROUGH a crate he can move**, pushing it one cell per
  tick and ending directly behind it, so one press can spend a crate's whole
  rope. On level 5 ``right`` from (3,2) drives the 3/3 crate at (3,4) out to
  (3,7) as 0/3 and parks the player at (3,6).
* **Bumping a crate you are standing NEXT to is a single step, not a slide.**
  The adjacent-nudge rules place the player explicitly and stamp no marker, so
  a shove from touching distance moves both bodies exactly one cell.
* **What stops a heavy player is `Fixed`, which is recomputed every tick**:
  walls are fixed in all four directions; a crate is fixed in direction ``d``
  when the cell that way is a `CrateBlock` (wall, crate, ROPE, finish, pillow
  stand or pullback tile), and an EMPTY crate is additionally fixed in every
  direction it has no rope in -- which is what makes a spent crate a wall you
  can only pull back.
* **Ropes block crates but not the player.** Rope is in `CrateBlock` and in no
  other group, so a crate can never be pushed onto any rope (its own included,
  which is why a rope path never crosses itself) while the player slides over
  rope freely. That is the whole of the "Don't cross me" chapter.
* **A pillow makes you LIGHT: you can no longer push, and you stop at anything
  solid instead of anything fixed.** Picking up and dropping happen on ARRIVAL
  at a pillow stand, mid-slide included, and the new weight governs the very
  next tick -- so a heavy player can enter a stand, go light and carry straight
  on past a crate he would otherwise have shoved.
* **A pullback tile reels EVERY crate in by one link**, fires on arrival
  (mid-slide included) and re-arms only after the player has left the tile. The
  ``[ Player Pullback ] [ Crate ] -> [ ] [ Crate PullbackMarker ]`` bracket is
  an or-group, so the adapter's Cartesian product marks every crate rather than
  one.
* **A `PullbackMarker` marks a CELL, not a crate.** Every rule that shoves a
  crate names `Crate` on its LHS and not the marker, so shoving a marked crate
  leaves the marker in the cell it came from -- where it waits, possibly for
  many presses, until some roped crate comes to rest on it. Getting this wrong
  is what the first differential fuzz caught: modelling it per-crate made the
  pullback follow the crate around and rewound shoves the interpreter kept.
* **A crate that moves twice inside one tick has its DIGIT changed only once,
  so the number and the rope come apart.** The digit rules are
  ``[ C11 DecMarker ] -> [ C01 ]``: they need the marker and the crate in the
  same cell, and a crate that has moved on since the marker was stamped has
  left it behind on a bare cell where nothing can spend it. So a marker cashing
  in on a crate the same press has just shoved, or two markers on consecutive
  links, leaves a crate showing (say) 1/3 on a two-link rope, plus a stranded
  marker that will change the next crate to stop there. It is 158 transitions
  of a quarter-million across levels 26 and 28 -- rare enough to be invisible to
  a random fuzz, common enough to be inside the ball a shortest solution
  explores -- which is why `slacks`, `incm` and `decm` are carried as state and
  why `--selfcheck` gives that branch a witness of its own.
* **The finish has two flavours and the level layout picks which.** The
  cosmetic rules turn a `FinishX` that shares a row or column with a `Corner`
  into an arrow, and the win rules are ``[ Player Finish no FinishX ] -> win``
  versus ``[ stationary Player FinishX ] -> win``. Corners sit on the wall ring,
  so a finish set INTO the wall is an arrow you win by sliding into, and a
  finish out in the open is an X you must come to a full stop on -- sliding over
  it does nothing. That is the "Please come to a full stop" chapter.
* **ACTION5 is read by no rule and measured as a no-op** over every state of
  every ball (`--selfcheck`), so `directions` is the four arrows.
* There is no `restart` and no lethal object, but there IS a way to lose: rope
  only reels back the way it was paid out, so a crate shoved into the wrong
  corner can wall a level off for good. `--report`'s ``dead`` column counts the
  ball states from which no win remains -- 128 604 of level 28's 162 327 before
  the tether-marker canonicalisation, and a third to a half of most levels
  after it. That is what `recovery_mode = "reset"` is for.

Rendering
---------
The sprite set needed real work before the corpus could teach this game; see
the note at the top of ``data/puzzlescript_games/Slide_Rule.txt`` and `_audit`.
Levels here are 9-14 cells wide, so ``cell_px = 64 // max(h, w)`` is 4 on eight
of them, and at 4 px the renderer DROPS sprite row and column 2. As shipped,
that made all four rope links render as bare floor and collapsed C01/C11,
C03/C13, C04/C05, C14/C15, C24/C25, C34/C35 and C44/C45/C55 -- i.e. neither the
rope nor the crate's own number was readable on the levels that need them most.
Three further fixes came out of auditing the reachable stacks rather than the
bare objects: the crate body is notched at the rope lanes (it used to hide the
link under itself), the player's head and legs moved to the centre column (they
used to hide a vertical link he was standing on), and the filled-slot colour is
no longer ``#e0e0e0`` -- that grey is one of Background's own five, so it
collapsed to the floor's ARC palette index and a full 5/5 crate rendered as
nothing at all.

Why a native model
------------------
The interpreter runs this game at ~78 presses/s: five rules re-stamp `Fixed` on
every wall and crate every tick, the level-number art is recomputed every tick,
and one press is a whole ``again`` slide. Enumerating even one level's ball on
`eng.step` is minutes. The native `_Board` runs the same mechanic at ~40 000
presses/s, so the interpreter's only job here is to CERTIFY: `--selfcheck`
differential-fuzzes the model against it from cold states, `--moves` pins each
named mechanic to a purpose-built board, and `--verify` replays every shipped
plan through the adapter.

The search
----------
`_Field` is [[ps-rbg-solver]]'s shape: a LAYERED forward BFS over the model that
stops at the end of the first layer producing a win, plus a reverse BFS over the
edges it collected. A state at depth ``g`` with true distance ``h`` has its whole
shortest route inside that ball whenever ``g + h <= d*``, which every state on a
shortest path satisfies by construction, and a state outside can only be
OVER-priced (a truncated subgraph omits edges, it never invents them). So the
plans are provably SHORTEST and the optimal-action sets are EXACT -- read off as
``dist(succ) == dist - 1`` -- without enumerating spaces that run past a million
states.

Results
-------
**31/31 levels, every plan provably SHORTEST** (d* runs 1..28; the whole game
plans in ~17s and 196k ball states, so there is no ``plan_cache_path`` worth the
staleness risk). Every step of every plan carries its exact optimal set.

What the reports say, all measured::

    --report      31/31 solved, ball sizes 1..70875, dead-state counts
    --selfcheck   1400 presses re-stepped on the interpreter from COLD, over
                  13 mechanic branches (the rarest, ``state.desync``, given a
                  witness of its own): MODEL CERTIFIED
    --moves       8 purpose-built boards, one per hard-to-reach mechanic: agree
    --audit       14 board sizes, 4-7 px cells, 0 indistinguishable compositions
    --verify      0 ball states render alike but disagree on their optimal set;
                  692 transitions prove the tether-marker canonicalisation
                  inert ON THE INTERPRETER; every d* and every tie set re-priced
                  by an independent bounded BFS: AGREES; 124 plans replayed
                  through the real adapter over 4 seeds: 124 wins
    --symmetry    636k transitions turned and mirrored: 28 are decided by the
                  interpreter's rule order (all in the double-move region of
                  levels 26 and 28), none of them only by a mirror, and NONE is
                  a press the corpus labels -- which is the evidence behind the
                  game's entry in `PuzzleScriptAdapter._FLIP_GAMES`

Four generated seeds replay frame-exact from the recorded SCREEN actions, cover
all 16 presentations, and label 1308 of 1308 expert steps.

Level indices here and in every report are 0-BASED, i.e. one less than the
number the game itself prints.

Recovery
--------
`recovery_mode = "reset"`: shoving a crate spends rope, and rope can only be
reeled back the way it was paid out, so a perturbed board is not re-plannable in
general. The episode-wide epsilon prefix explores freely and ONE RESET restores
the level start, from which the cached plan replays a guaranteed win.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                    # noqa: E402
from adapters.puzzlescript_adapter import _render_frame         # noqa: E402
from solvers.common.ps_astar import (                          # noqa: E402
    Plan, PSAStarSolver, PSExpert, screen_action,
)

GAME_NAME = "Slide_Rule"

#: Direction ids. The order is the plan's tie-break order, which is what makes a
#: re-derived plan byte-identical across processes.
_DIRS = ("up", "down", "left", "right")
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))
_OPP = (1, 0, 3, 2)

#: Per-direction rope object names, indexed the same way as `_DIRS`. A rope link
#: is stamped at BOTH ends: the cell the crate left gets the link pointing after
#: it, the cell it arrived in gets the link pointing back.
_ROPE = ("ropeu", "roped", "ropel", "roper")

#: `Fixed*` is cleared and recomputed at the top of every tick, so it is on the
#: grid between presses but is a pure function of the board -- the model
#: recomputes it and never carries it.
_DERIVED = ("fixedu", "fixedd", "fixedl", "fixedr")

#: Objects that live and die inside one tick, so a settled grid holding one is a
#: mechanic the model does not know about. (`PUp`/`PLeft`/... can survive when a
#: slide runs off the board edge, which only a finish set into the wall allows,
#: i.e. only on a board that has already been won -- `--selfcheck` checks that
#: rather than assuming it.)
_TRANSIENT = ("pup", "pdown", "pleft", "pright", "pillowmarker")

#: The digit markers. Usually transient -- stamped on a crate and spent by the
#: digit rules at the end of the same tick -- but a crate that moves twice in
#: one tick strands one, and a stranded marker is live state. See `_Board`.
_DIGITMARK = ("incmarker", "decmarker")

#: Every crate object, as ``name -> (cur, max)``.
_CRATES = {f"c{cur}{mx}": (cur, mx)
           for mx in range(1, 6) for cur in range(0, mx + 1)}

# ---------------------------------------------------------------------------
# The level, natively
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry plus the whole mechanic as one `step`.

    Cells are flat ``r * w + c`` ids. A state is

        ``(player, light, paths, slacks, marked, incm, decm, pillows)``

    * ``player`` -- the cell he is standing on;
    * ``light``  -- 1 while he is carrying a pillow (cannot push, stops at
      anything solid);
    * ``paths``  -- one tuple per crate, ``(tether, ..., current cell)``: the
      rope IS the path, with consecutive cells carrying a link each way;
    * ``slacks`` -- each crate's left-hand digit, carried SEPARATELY from the
      rope length rather than derived as ``max - (len(path) - 1)``. They agree
      in almost every state, and the exceptions are the whole reason this field
      exists: a crate that moves twice inside one tick (a pullback marker
      cashing in on a crate the same press has just shoved, or two markers on
      consecutive links) gets its digit changed only ONCE, because the
      ``Dec``/``IncMarker`` the first move stamped is left on a cell the crate
      has already left and no rule can consume it there. The board then really
      does show a crate whose number disagrees with its own rope, and the
      mechanic reads BOTH -- `CEmpty` is the digit, `CrateBlock` is the rope.
      Measured on levels 26 and 28 (72 and 86 transitions of ~250 000);
    * ``marked`` -- a bitmask over CELLS holding a `PullbackMarker`. Not over
      crates: every rule that shoves a crate names `Crate` on its LHS and not
      the marker, so the marker is left behind in the cell the crate came from.
      A marker therefore outlives the crate it was stamped on, and cashes in
      whenever a roped crate next comes to rest on that cell (see the module
      docstring). Markers on a crate's OWN TETHER CELL are dropped: the reel
      rule needs the crate to be standing on rope, a crate stands on its tether
      cell only when its path is the tether alone (no rope under it at all), and
      no other crate can ever reach that cell -- it holds either this crate or
      this crate's rope. So such a marker can never fire, and keeping it would
      split otherwise identical boards (152 540 of level 26's 155 399 ball
      states) on a bit nothing can observe or use. `--verify` measures that
      rather than trusting the argument;
    * ``incm`` / ``decm`` -- bitmasks over cells holding a stranded `IncMarker`
      or `DecMarker`. Normally empty: a marker is stamped on the crate and
      consumed by the digit rules at the end of the same tick. They survive
      exactly in the double-move case above, and they stay live -- a crate that
      later comes to rest on one has its digit changed then;
    * ``pillows``-- a bitmask over the pillow stands that still hold a pillow.

    Everything else on the interpreter's grid is either static (walls, corners,
    header art, sources, stands, pullback tiles, the finish) or derived from
    this tuple (`Fixed*`, `StandMarker`, `PullbackUseMarker`), which
    `--selfcheck` asserts rather than assumes.
    """

    __slots__ = ("h", "w", "wall", "block", "stands", "stand_idx", "pulls",
                 "finish", "strict", "maxes", "sources", "inert_mask",
                 "nxt", "start")

    def __init__(self, h, w, wall, block, stands, pulls, finish, strict,
                 maxes, start):
        self.h, self.w = h, w
        #: ``wall[cell]`` -- Wall (the header row and the corners are Walls too).
        self.wall = wall
        #: ``block[cell]`` -- the STATIC half of `CrateBlock`: wall, finish,
        #: pillow stand or pullback tile. Crates and rope are added per tick.
        self.block = block
        self.stands = frozenset(stands)
        self.stand_idx = {cell: i for i, cell in enumerate(sorted(stands))}
        self.pulls = frozenset(pulls)
        self.finish = finish
        #: True for a `FinishX` -- the player must come to a full STOP on it.
        self.strict = strict
        self.maxes = tuple(maxes)
        self.start = start
        #: Each crate's tether point -- static, and what `_extract` walks the
        #: rope forward from to rebuild a path off the interpreter's grid.
        self.sources = tuple(p[0] for p in start[2])
        #: The complement of the tether cells, as a mask: ``marked & inert_mask``
        #: is the canonical marker set (see the class docstring).
        self.inert_mask = ~sum(1 << c for c in set(self.sources))

        # ``nxt[d][cell]``: one step along d, or -1 off the board. The inner
        # loop of every search runs on this.
        self.nxt = self._neighbours(h, w)

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_engine(cls, eng, ids) -> "_Board":
        """Read a board off the engine's grid -- a level start or any position
        the game can be in.

        Deliberately not restricted to a level start: `Source` is a static
        object marking each crate's tether cell, so a crate's path can be
        recovered by walking its rope, and the expert can therefore plan from
        wherever the recorder happens to be (an epsilon detour, a re-plan after
        the exploration prefix) instead of only from a reset.

        The invariants the model rests on are ASSERTED in `_read_state`, because
        each of them would make the model wrong rather than slow.
        """
        h, w = eng.height, eng.width
        wall = [False] * (h * w)
        block = [False] * (h * w)
        stands, pulls, sources = [], [], []
        finish = None
        strict = False
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                pos = r * w + c
                names = {ids[o] for o in cell if o in ids}
                if "wall" in names:
                    wall[pos] = block[pos] = True
                if "pillowstand" in names:
                    stands.append(pos)
                    block[pos] = True
                if "pullback" in names:
                    pulls.append(pos)
                    block[pos] = True
                if "source" in names:
                    sources.append(pos)
                for name in ("finishx", "finishl", "finishr", "finishu",
                             "finishd"):
                    if name in names:
                        assert finish is None, "more than one finish"
                        finish, strict = pos, name == "finishx"
                        block[pos] = True
        assert finish is not None, "no finish"
        sources.sort()
        stand_idx = {cell: i for i, cell in enumerate(sorted(stands))}
        nxt = cls._neighbours(h, w)
        state, maxes = _read_state(eng, ids, w, nxt, sources, stand_idx)
        return cls(h, w, wall, block, stands, pulls, finish, strict,
                   maxes, state)

    @staticmethod
    def _neighbours(h, w) -> list:
        tables = []
        for dr, dc in _DELTA:
            table = [-1] * (h * w)
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        table[r * w + c] = nr * w + nc
            tables.append(table)
        return tables

    # -- one press ------------------------------------------------------------
    def step(self, state, d: int):
        """One keypress, as ``(new_state, won)``, or None when the press changes
        nothing at all.

        A faithful transcription of the rule list's execution order. The three
        orderings that actually matter, and that a plausible re-ordering gets
        wrong, are: the pillow phase runs BEFORE the movement rules (so the
        weight change governs the tick that leaves the stand); the pullback
        marking runs BEFORE the push rules and the reel-in runs AFTER them (so a
        crate shoved on the tick you land on a pullback tile is shoved straight
        back); and `Fixed` is computed ONCE per tick, so a crate that has just
        been shoved leaves no `Fixed` behind in its new cell and the slide
        carries on into it.
        """
        return self._run(state, d)[:2]

    def resolve(self, state, d: int):
        """`step`, plus the set of mechanic branch names the press exercised.

        Split out from `step` so `--selfcheck`'s coverage counters report what
        actually fired rather than reverse-engineering it from a before/after
        diff -- which cannot tell a shove into a spent crate from a walk into a
        wall, since both are "nothing moved"."""
        new, won, tags = self._run(state, d)
        return new, won, tags

    def _run(self, state, d: int):
        nxt, wall, block = self.nxt, self.wall, self.block
        stands, pulls, maxes = self.stands, self.pulls, self.maxes
        finish, strict = self.finish, self.strict
        player, light, paths, slacks, marked, incm, decm, pillows = state
        paths = [list(p) for p in paths]
        slacks = list(slacks)
        ncr = len(paths)
        tags: set[str] = set()

        pmark = -1                       # the PLeft/PRight/... slide marker
        force = d                        # the input force, tick 0 only
        # StandMarker / PullbackUseMarker sit on the cell the player is on and
        # are cleared the tick after he leaves, so they are derived at rest.
        sm = player if player in stands else -1
        pu = player if player in pulls else -1
        won = False

        for _ in range(50):
            before = (player, light, [tuple(p) for p in paths], tuple(slacks),
                      marked, incm, decm, pillows, pmark, sm, pu)
            again = False

            # -- per-tick tables: crates, rope, and Fixed ---------------------
            crate_at = {}
            for i in range(ncr):
                crate_at[paths[i][-1]] = i
            rope = {}
            for p in paths:
                for j in range(len(p) - 1):
                    a, b = p[j], p[j + 1]
                    e = 0 if b == a - self.w else 1 if b == a + self.w else \
                        2 if b == a - 1 else 3
                    rope[a] = rope.get(a, 0) | (1 << e)
                    rope[b] = rope.get(b, 0) | (1 << _OPP[e])
            fixed = {}
            for i in range(ncr):
                cell = paths[i][-1]
                slack = slacks[i]
                rc = rope.get(cell, 0)
                m = 0
                for e in range(4):
                    if slack == 0 and not (rc >> e) & 1:
                        m |= 1 << e
                        continue
                    t = nxt[e][cell]
                    if t >= 0 and (block[t] or t in crate_at or t in rope):
                        m |= 1 << e
                fixed[cell] = m

            def solid(cell):
                return wall[cell] or cell in crate_at

            # -- pillows ------------------------------------------------------
            if sm != player:
                sm = -1
            if player in stands:
                if sm != player:
                    bit = 1 << self.stand_idx[player]
                    if not light and pillows & bit:
                        light, pillows = 1, pillows & ~bit
                        tags.add("pillow.take")
                    elif light:
                        light, pillows = 0, pillows | bit
                        tags.add("pillow.drop")
                    sm = player

            # -- stamp the slide marker (input force, tick 0) -----------------
            if force >= 0 and pmark < 0:
                t = nxt[force][player]
                if t >= 0 and not solid(t):
                    pmark, force = force, -1
                    tags.add("slide.start")

            # -- pullback tile ------------------------------------------------
            if pu != player:
                pu = -1
            if player in pulls and pu != player:
                for i in range(ncr):
                    marked |= 1 << paths[i][-1]
                marked &= self.inert_mask
                pu = player
                tags.add("pullback.arm")

            # -- shove a crate you are touching (input force, heavy) ----------
            if force >= 0 and not light:
                b = nxt[force][player]
                if b >= 0 and b in crate_at:
                    i = crate_at[b]
                    if not (rope.get(b, 0) >> force) & 1:
                        if not (fixed[b] >> force) & 1:
                            cc = nxt[force][b]
                            if cc >= 0:
                                paths[i].append(cc)
                                decm |= 1 << cc
                                player, force, again = b, -1, True
                                tags.add("nudge.out")
                    else:
                        cc = nxt[force][b]
                        if cc >= 0 and (rope.get(cc, 0) >> _OPP[force]) & 1:
                            paths[i].pop()
                            incm |= 1 << cc
                            player, force, again = b, -1, True
                            tags.add("nudge.back")

            # -- advance the slide --------------------------------------------
            pforce = -1
            if pmark >= 0:
                t = nxt[pmark][player]
                if t >= 0 and (not solid(t) if light
                               else not (fixed.get(t, 0) >> pmark) & 1
                               and not wall[t]):
                    pforce, again = pmark, True

            # -- a sliding heavy player meets a crate -------------------------
            if pmark >= 0 and not light:
                b = nxt[pmark][player]
                if b >= 0 and b in crate_at:
                    i = crate_at[b]
                    if not (rope.get(b, 0) >> pmark) & 1:
                        if not (fixed[b] >> pmark) & 1:
                            cc = nxt[pmark][b]
                            if cc >= 0:
                                paths[i].append(cc)
                                decm |= 1 << cc
                                player, pforce, again = b, -1, True
                                tags.add("slide.push")
                    else:
                        cc = nxt[pmark][b]
                        if cc >= 0 and (rope.get(cc, 0) >> _OPP[pmark]) & 1:
                            paths[i].pop()
                            incm |= 1 << cc
                            player, pforce, again = b, -1, True
                            tags.add("slide.reel")

            # -- the pullback marker cashes in --------------------------------
            # Four separate rules, one per direction, each driven to a fixpoint
            # in the order they are written (`left, right, up, down`) and never
            # re-run as a group -- so a rope with markers on consecutive links
            # reels several of them in one tick while a rope that BENDS back
            # into an already-run direction stops at the bend. See trap 7 in
            # [[ps-engine-render-gotchas]]; the fixpoint is the interpreter's,
            # not a convenience.
            if marked:
                for e in (2, 3, 0, 1):
                    while True:
                        fired = False
                        for i in range(ncr):
                            p = paths[i]
                            if len(p) < 2:
                                continue
                            cell = p[-1]
                            if not (marked >> cell) & 1:
                                continue
                            if nxt[e][cell] != p[-2]:
                                continue
                            p.pop()
                            marked &= ~(1 << cell)
                            incm |= 1 << p[-1]
                            fired = again = True
                            tags.add("pullback.reel")
                        if not fired:
                            break

            # -- the digit rules, dec block then inc block --------------------
            # Each is 15 rules of the shape ``[ C11 DecMarker ] -> [ C01 ]
            # again``, so a marker only spends itself on a crate STANDING on it
            # at this point in the tick. A crate that has already moved on has
            # left its marker behind on a bare cell, where nothing consumes it
            # -- that is the whole of `slacks`/`incm`/`decm` being state.
            for i in range(ncr):
                cell = paths[i][-1]
                if (decm >> cell) & 1 and slacks[i] > 0:
                    slacks[i] -= 1
                    decm &= ~(1 << cell)
                    again = True
                    tags.add("digit.dec")
            for i in range(ncr):
                cell = paths[i][-1]
                if (incm >> cell) & 1 and slacks[i] < maxes[i]:
                    slacks[i] += 1
                    incm &= ~(1 << cell)
                    again = True
                    tags.add("digit.inc")

            # -- drop the slide marker when the way ahead is blocked ----------
            if pmark >= 0:
                t = nxt[pmark][player]
                if light:
                    if t < 0 or wall[t] or any(paths[i][-1] == t
                                               for i in range(ncr)):
                        pmark = -1
                elif t < 0 or (fixed.get(t, 0) >> pmark) & 1 or wall[t]:
                    pmark = -1

            # -- win ----------------------------------------------------------
            if player == finish and (not strict or (pforce < 0 and force < 0)):
                won = True

            # -- resolve the movement force -----------------------------------
            if pforce >= 0:
                t = nxt[pforce][player]
                if t >= 0 and not wall[t] and not any(paths[i][-1] == t
                                                      for i in range(ncr)):
                    player = t

            force = -1                   # the input force is tick 0 only
            after = (player, light, [tuple(p) for p in paths], tuple(slacks),
                     marked, incm, decm, pillows, pmark, sm, pu)
            if won or not again or after == before:
                break

        new = (player, light, tuple(tuple(p) for p in paths), tuple(slacks),
               marked, incm, decm, pillows)
        if new == state and not won:
            return None, False, tags
        return new, won, tags


# ---------------------------------------------------------------------------
# The exact distance-to-win field
# ---------------------------------------------------------------------------

class _Field:
    """Every state a SHORTEST solution can pass through, with its exact distance
    to a win.

    Layered forward BFS over `_Board.step`, stopped at the end of the first
    layer that produces a win (depth ``d* - 1``), then a reverse BFS over the
    edges collected on the way. See the module docstring for why that ball is
    enough for the plans to be provably shortest and the optimal sets exact.
    """

    __slots__ = ("board", "cap", "keys", "succ", "dist", "star", "overflow")

    #: The sentinel successor for a winning press.
    WIN = -1

    def __init__(self, board: _Board, cap: int = 400_000):
        self.board = board
        self.cap = cap
        self.keys: list = []
        self.succ: list = []
        self.dist: dict[int, int] = {}
        self.star: int | None = None
        self.overflow = False

    def build(self) -> "_Field":
        board = self.board
        start = board.start
        index = {start: 0}
        self.keys = [start]
        self.succ = [None]
        layer = [0]
        depth = 0
        while layer and self.star is None:
            nxt: list[int] = []
            for i in layer:
                row = []
                for e in range(4):
                    out, won = board.step(self.keys[i], e)
                    if won:
                        row.append(self.WIN)
                        continue
                    if out is None:
                        row.append(None)          # a press that did nothing
                        continue
                    j = index.get(out)
                    if j is None:
                        j = len(self.keys)
                        index[out] = j
                        self.keys.append(out)
                        self.succ.append(None)
                        nxt.append(j)
                    row.append(j)
                self.succ[i] = row
                if self.WIN in row and self.star is None:
                    self.star = depth + 1         # finish the layer, then stop
            if len(self.keys) > self.cap:                     # pragma: no cover
                self.overflow = True
                return self
            depth += 1
            layer = nxt

        preds: list[list[int]] = [[] for _ in self.keys]
        frontier = []
        for i, row in enumerate(self.succ):
            if row is None:                # a depth-d* state, left unexpanded
                continue
            for j in row:
                if j == self.WIN:
                    if i not in self.dist:
                        self.dist[i] = 1
                        frontier.append(i)
                elif j is not None:
                    preds[j].append(i)
        queue = deque(frontier)
        while queue:
            i = queue.popleft()
            for p in preds[i]:
                if p not in self.dist:
                    self.dist[p] = self.dist[i] + 1
                    queue.append(p)
        return self

    def optimal(self, state: int) -> list[str]:
        """Every press that is equally shortest at ``state``."""
        d = self.dist[state]
        row = self.succ[state]
        if d == 1:
            return [a for a, j in zip(_DIRS, row) if j == self.WIN]
        return [a for a, j in zip(_DIRS, row)
                if j is not None and j != self.WIN
                and self.dist.get(j, 1 << 30) == d - 1]

    def plan(self):
        """``(presses, optsets)`` for the walk downhill from the level start, or
        ``(None, [])`` when no win is reachable."""
        if 0 not in self.dist:
            return None, []
        presses: list[str] = []
        optsets: list[list[str]] = []
        state = 0
        while True:
            best = self.optimal(state)
            optsets.append(best)
            presses.append(best[0])
            if self.dist[state] == 1:
                return presses, optsets
            state = self.succ[state][_DIRS.index(best[0])]


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

def _ids(game) -> dict:
    """``obj_index -> name`` for every object the model reads."""
    return {v: k for k, v in game.obj_name_to_idx.items()}


class SlideRuleExpert(PSExpert):
    """Plans on the native `_Board`, which the interpreter only certifies.

    `PSExpert` keeps the in-memory memo, the on-disk plan cache with its
    staleness check and the snapshot/restore discipline; the only override is
    `_search`. `heuristic` is unreachable -- the field is exact, nothing here
    runs A*.
    """

    directions = list(_DIRS)
    #: No `plan_cache_path`: the whole game's fields build in ~17s, which every
    #: `parallelize_generator` shard can afford once, and the cache is keyed on
    #: the level START board -- which a change to the MODEL does not touch, so a
    #: stale entry would be served silently. Cheaper to re-derive than to risk
    #: that.

    def heuristic(self, eng) -> int:                          # pragma: no cover
        raise AssertionError("the field is exact; no search runs here")

    def board(self, eng) -> _Board:
        return _Board.from_engine(eng, _ids(self.g))

    def field(self, eng) -> _Field:
        return _Field(self.board(eng)).build()

    def _search(self, eng) -> list | None:
        presses, optsets = self.field(eng).plan()
        return None if presses is None else Plan(presses, optsets)


class SlideRuleSolver(PSAStarSolver):
    game_id = "puzzlescript_slide_rule"
    game_name = GAME_NAME
    game_module_id = "ps:slide_rule"
    expert_cls = SlideRuleExpert

    #: Room for the longest plan plus the exploration prefix and the re-plan
    #: after it, well inside the adapter's 200-step per-level budget.
    max_steps = 120

    def prepare_expert(self, game, expert) -> None:
        """Build every level's field before `discover_solvable` asks for it --
        the same work either way, but it makes the one-time cost visible as
        startup and fills the disk cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with the adapter parked at level 0."""
    solver = SlideRuleSolver()
    game = solver.make_game(0)
    return solver, game, SlideRuleExpert(game)


#: Everything a model state owns. Stripped from the level-start grid to leave
#: the static scenery `_seat` paints a state back onto.
_DYNAMIC = (*_TRANSIENT, *_DERIVED, *_DIGITMARK, *_ROPE,
            "playerheavy", "playerlight", "pillow", "standmarker",
            "pullbackusemarker", "pullbackmarker", *_CRATES)


def _static_grid(eng, idx: dict) -> list:
    """The engine's current grid with every dynamic object stripped.

    `Source`, `PillowStand`, `Pullback`, the finish, the walls and the whole
    level-number frieze are kept: no rule creates, moves or destroys any of
    them, which `--selfcheck` re-checks after every press rather than assuming.
    """
    drop = {idx[n] for n in _DYNAMIC if n in idx}
    return [[cell - drop for cell in row] for row in eng.grid]


def _seat(eng, idx: dict, board: _Board, static, state) -> None:
    """Load a model state onto the interpreter's grid.

    `Fixed*` is deliberately NOT painted: the first rule of the fixed block
    clears every marker and the next five re-derive them, so seating them would
    only give a stale copy the chance to disagree.
    """
    player, light, paths, slacks, marked, incm, decm, pillows = state
    w = board.w
    grid = [[set(cell) for cell in row] for row in static]

    def put(pos, name):
        grid[pos // w][pos % w].add(idx[name])

    put(player, "playerlight" if light else "playerheavy")
    if player in board.stands:
        put(player, "standmarker")
    if player in board.pulls:
        put(player, "pullbackusemarker")
    for cell in range(board.h * board.w):
        if (marked >> cell) & 1:
            put(cell, "pullbackmarker")
        if (incm >> cell) & 1:
            put(cell, "incmarker")
        if (decm >> cell) & 1:
            put(cell, "decmarker")
    for i, p in enumerate(paths):
        put(p[-1], f"c{slacks[i]}{board.maxes[i]}")
        for j in range(len(p) - 1):
            a, b = p[j], p[j + 1]
            e = (0 if b == a - w else 1 if b == a + w else
                 2 if b == a - 1 else 3)
            put(a, _ROPE[e])
            put(b, _ROPE[_OPP[e]])
    for cell, i in board.stand_idx.items():
        if (pillows >> i) & 1:
            put(cell, "pillow")
    eng.grid = grid
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _read_state(eng, names: dict, w: int, nxt: list, sources, stand_idx,
                *, won: bool = False):
    """``(state, maxes)`` read off the engine's grid.

    Raises on anything the model does not represent -- a surviving `DecMarker`,
    a crate whose digits disagree with its rope length, a rope that forks or
    belongs to no crate, a pillow off its stand. That is what makes the fuzz a
    real differential test: a mechanic the model is blind to shows up here as an
    exception rather than as two states that happen to agree on the fields the
    model does track. (The digit/rope check is the one that convicts the chained
    reel -- see `_Unmodelled`.)
    """
    player = light = None
    crates: dict[int, tuple[int, int]] = {}
    rope: dict[int, int] = {}
    marked = incm = decm = 0
    pillows = 0
    sm = pu = None
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            pos = r * w + c
            ns = {names[o] for o in cell if o in names}
            if "playerheavy" in ns:
                player, light = pos, 0
            if "playerlight" in ns:
                player, light = pos, 1
            for e, rn in enumerate(_ROPE):
                if rn in ns:
                    rope[pos] = rope.get(pos, 0) | (1 << e)
            cr = ns & set(_CRATES)
            if cr:
                assert len(cr) == 1, f"two crates in one cell at {pos}"
                crates[pos] = _CRATES[cr.pop()]
            if "pullbackmarker" in ns:
                marked |= 1 << pos
            if "incmarker" in ns:
                incm |= 1 << pos
            if "decmarker" in ns:
                decm |= 1 << pos
            if "pillow" in ns:
                assert pos in stand_idx, "a pillow off its stand"
                pillows |= 1 << stand_idx[pos]
            if "standmarker" in ns:
                sm = pos
            if "pullbackusemarker" in ns:
                pu = pos
            if not won:
                for name in _TRANSIENT:
                    assert name not in ns, f"{name} survived the press at {pos}"
    assert player is not None, "the player left the board"
    assert sm in (None, player), "StandMarker away from the player"
    assert pu in (None, player), "PullbackUseMarker away from the player"

    paths, maxes, slacks = [], [], []
    used: set[int] = set()
    inert = 0
    for i, src in enumerate(sources):
        path, seen, cell = [src], {src}, src
        while cell not in crates:
            step = [nxt[e][cell] for e in range(4)
                    if (rope.get(cell, 0) >> e) & 1]
            step = [t for t in step if t not in seen]
            assert len(step) == 1, f"crate {i}'s rope forks or dead-ends"
            cell = step[0]
            seen.add(cell)
            path.append(cell)
        cur, mx = crates[cell]
        assert len(path) - 1 <= mx, f"crate {i}'s rope is longer than its max"
        used |= seen
        inert |= 1 << src
        paths.append(tuple(path))
        maxes.append(mx)
        slacks.append(cur)
    assert len(crates) == len(sources), "a crate appeared or vanished"
    assert set(rope) <= used, "rope that belongs to no crate"
    return (player, light, tuple(paths), tuple(slacks), marked & ~inert,
            incm, decm, pillows), maxes


def _extract(eng, names: dict, board: _Board, *, won: bool = False):
    """`_read_state` against a board already known, with the static scenery
    re-checked -- nothing here is supposed to move, and a rule that moved one
    would otherwise be invisible."""
    stands, pulls = set(), set()
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            ns = {names[o] for o in cell if o in names}
            if "pillowstand" in ns:
                stands.add(r * board.w + c)
            if "pullback" in ns:
                pulls.add(r * board.w + c)
    assert stands == board.stands, "a pillow stand moved"
    assert pulls == board.pulls, "a pullback tile moved"
    state, maxes = _read_state(eng, names, board.w, board.nxt, board.sources,
                               board.stand_idx, won=won)
    assert tuple(maxes) == board.maxes, "a crate changed its maximum"
    player = state[0]
    assert (player in board.stands) == any(
        "standmarker" in {names[o] for o in cell if o in names}
        for row in eng.grid for cell in row), "StandMarker desynced"
    return state


def _report() -> int:
    """Per level: the board's census, the ball size and the proved d*."""
    _solver, game, expert = _levels()
    ids = _ids(game._game)
    total = t0 = 0
    chains = 0
    print(f"{'lvl':>3} {'size':>7} {'crates':>11} {'pill':>4} {'pull':>4} "
          f"{'fin':>4} {'ball':>7} {'d*':>3} {'dead':>6} {'mark':>5} "
          f"{'desync':>6}  plan")
    tstart = time.time()
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board.from_engine(game._engine, ids)
        field = _Field(board).build()
        presses, _sets = field.plan()
        ball = len(field.keys)
        total += ball
        dead = sum(1 for i in range(ball)
                   if field.succ[i] is not None and i not in field.dist)
        # `chain` is the reachability check behind `_Unmodelled`: a state with
        # PullbackMarkers on two consecutive links of one rope. Zero everywhere
        # is the claim the model's one refusal rests on.
        mark = chain = 0
        for _p, _l, paths, slacks, marked, incm, decm, _pi in field.keys:
            if marked:
                mark += 1
            if incm or decm or any(
                    slacks[i] != board.maxes[i] - (len(paths[i]) - 1)
                    for i in range(len(paths))):
                chain += 1
        chains += chain
        crates = "/".join(str(m) for m in board.maxes) or "-"
        print(f"{level:>3} {board.h}x{board.w:<5} {crates:>11} "
              f"{len(board.stands):>4} {len(board.pulls):>4} "
              f"{'X' if board.strict else 'arw':>4} {ball:>7} "
              f"{len(presses) if presses else -1:>3} {dead:>6} {mark:>5} "
              f"{chain:>5}  "
              f"{' '.join(p[0] for p in presses) if presses else 'UNSOLVED'}")
        if presses:
            t0 += 1
    print(f"solved {t0}/{game.n_levels}; {total} ball states; "
          f"{chains} states could chain a pullback reel; "
          f"{time.time() - tstart:.1f}s")
    return 0


#: Model press id -> engine direction. ``-1`` is ACTION5, which no rule reads --
#: `selfcheck` measures that against the interpreter rather than assuming it.
def _dname(e: int) -> str:
    return "action" if e < 0 else _DIRS[e]


def _selfcheck(walks: int = 8, steps: int = 40, samples: int = 90,
               seed: int = 0, pool_cap: int = 20_000) -> int:
    """Differential fuzz: `_Board` against the real interpreter, from COLD.

    Every sampled press re-seats the model state on the engine and steps once,
    so nothing is hidden behind a warm-up -- turn-one bookkeeping (the pillow
    toggle, the pullback arming, a marker left over from a previous press) is
    exactly the class of mechanic a warm run conceals, and it is exactly what an
    exploration prefix walks into.

    The sample is not uniform: the pool is classified by which mechanic BRANCH
    the press exercises and every branch gets at least one witness, because a
    uniform draw over the reachable states misses the rare ones every time (a
    pullback marker cashing in on the tick it is paid out is a handful of pairs
    out of thousands). `_Board.resolve` reports the branch directly rather than
    having it reverse-engineered from a before/after diff, which cannot tell a
    shove into a spent crate from a walk into a wall.
    """
    _solver, game, expert = _levels()
    idx = game._game.obj_name_to_idx
    names = _ids(game._game)
    eng = game._engine
    rng = random.Random(seed)
    cover: dict[str, int] = {}
    tested: dict[str, int] = {}
    bad = checked = 0
    t0 = time.time()
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board.from_engine(eng, names)
        static = _static_grid(eng, idx)

        # The pool is the ball plus random walks. The ball alone would miss
        # nothing a shortest solution needs, but the walks reach states no
        # optimal route visits -- which is where an exploration prefix lives.
        pool = set(_Field(board).build().keys)
        if len(pool) > pool_cap:
            pool = set(rng.sample(sorted(pool), pool_cap))
        pool.add(board.start)
        for _ in range(walks):
            st = board.start
            for _ in range(steps):
                out, won = board.step(st, rng.randrange(4))
                if out is None or won:
                    continue
                st = out
                pool.add(st)

        bytag: dict[tuple, list] = {}
        for st in pool:
            # A state whose digits disagree with its rope, or that carries a
            # stranded digit marker, is its own branch: those are ~215 of a
            # quarter-million ball states and a uniform draw never sees one.
            odd = ("state.desync",) if (
                st[5] or st[6] or any(
                    st[3][i] != board.maxes[i] - (len(st[2][i]) - 1)
                    for i in range(len(st[2])))) else ()
            for e in (0, 1, 2, 3, -1):
                _out, _won, tags = board.resolve(st, e)
                key = (tuple(sorted(tags)) or ("noop",)) + odd
                bytag.setdefault(key, []).append((st, e))
        for key, lst in bytag.items():
            for tag in key:
                cover[tag] = cover.get(tag, 0) + len(lst)
        share = max(1, samples // max(1, len(bytag)))
        chosen = []
        for key in sorted(bytag):
            lst = bytag[key]
            rng.shuffle(lst)
            take = lst[:share]
            chosen.extend((key, st, e) for st, e in take)
        rng.shuffle(chosen)

        notes = []
        for key, st, e in chosen[:samples]:
            for tag in key:
                tested[tag] = tested.get(tag, 0) + 1
            _seat(eng, idx, board, static, st)
            eng._rule_restart = False
            eng.step(_dname(e))
            won_i = eng.check_win()
            assert not eng._rule_restart, "this game has no restart rule"
            got = _extract(eng, names, board, won=won_i)
            out, won_m, _tags = board.resolve(st, e)
            want = st if out is None else out
            checked += 1
            if got != want or won_i != won_m:
                bad += 1
                if len(notes) < 3:
                    notes.append(f"{_dname(e)} at {st}: model {want}/{won_m} "
                                 f"!= engine {got}/{won_i}")
        print(f"level {level:>2}: {len(pool):>5} states, {len(bytag):>3} "
              f"branch signatures, {min(len(chosen), samples):>3} presses "
              f"{'AGREE' if not notes else 'DISAGREE: ' + '; '.join(notes)}")
    print(f"\n{'branch':<16} {'in pool':>9} {'fuzzed':>7}")
    for tag in sorted(set(cover) | set(tested)):
        print(f"{tag:<16} {cover.get(tag, 0):>9} {tested.get(tag, 0):>7}")
    print(f"\n{checked} presses compared against the interpreter in "
          f"{time.time() - t0:.1f}s: "
          f"{'MODEL CERTIFIED' if not bad else f'{bad} DISAGREEMENTS'}")
    return 0 if not bad else 1


#: Objects with an all-transparent sprite: pure bookkeeping, and `_audit`
#: asserts that rather than assuming it before leaving them out of a
#: composition. `Corner` is in here too -- it exists only to orient the finish.
_INVISIBLE = ("corner", *_DERIVED, *_TRANSIENT,
              "standmarker", "pullbackusemarker")



def _compositions(board: _Board, field: _Field, cap: int, rng) -> set:
    """Every cell composition the reachable ball can SHOW, as name tuples.

    Derived from the ball's states rather than from a hand-written product of
    the object list: the product over-counts wildly (a rope cell can carry four
    links, a crate five digits) and, worse, under-counts the stacks that only
    arise mid-mechanic. The winning boards are in `field.keys` too, so the win
    frame -- the one the corpus exists to teach -- is covered.
    """
    w = board.w
    static: dict[int, list] = {}
    for cell in range(board.h * board.w):
        names = []
        if board.wall[cell]:
            names.append("wall")
        if cell == board.finish:
            names.append("finishx" if board.strict else "finishl")
        if cell in board.stands:
            names.append("pillowstand")
        if cell in board.pulls:
            names.append("pullback")
        if cell in board.sources:
            names.append("source")
        static[cell] = names
    comps = {tuple(sorted(v)) for v in static.values()}

    keys = field.keys
    if len(keys) > cap:
        keys = rng.sample(keys, cap)
    for player, light, paths, slacks, marked, _i, _d, pillows in keys:
        touched: dict[int, list] = {}
        for i, p in enumerate(paths):
            touched.setdefault(p[-1], []).append(f"c{slacks[i]}{board.maxes[i]}")
            for j in range(len(p) - 1):
                a, b = p[j], p[j + 1]
                e = (0 if b == a - w else 1 if b == a + w else
                     2 if b == a - 1 else 3)
                touched.setdefault(a, []).append(_ROPE[e])
                touched.setdefault(b, []).append(_ROPE[_OPP[e]])
        for cell, i in board.stand_idx.items():
            if (pillows >> i) & 1:
                touched.setdefault(cell, []).append("pillow")
        m = marked
        while m:
            low = m & -m
            touched.setdefault(low.bit_length() - 1, []).append("pullbackmarker")
            m ^= low
        touched.setdefault(player, []).append(
            "playerlight" if light else "playerheavy")
        for cell, extra in touched.items():
            comps.add(tuple(sorted(static[cell] + extra)))
    return comps


def _audit(cap: int = 4000) -> int:
    """Assert every cell COMPOSITION the reachable space can show is distinct.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64, so the output's cell grid is not ``cell_px``-aligned
    and an arithmetic crop reads the wrong window (see
    [[explod-rendering-fixes]]).

    The size that decides everything here is ``cell_px = 64 // max(h, w) == 4``,
    which eight levels have and which DROPS sprite row and column 2 -- see the
    sprite note at the top of the .txt. Before the sprite pass this report found
    19 collisions per 4px size: all four rope links rendered as bare floor, and
    C01/C11, C03/C13, C04/C05, C14/C15, C24/C25, C34/C35 and C44/C45/C55 were
    pairwise identical, i.e. neither the rope nor the crate's own number could
    be read on the levels that need them most.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    idx, names = g.obj_name_to_idx, _ids(g)
    rng = random.Random(20260817)

    for name in _INVISIBLE:
        obj = next(o for o in g.objects.values() if o.name == name)
        assert all(ci < 0 or obj.colors[ci] < 0
                   for row in (obj.sprite or []) for ci in row
                   if ci >= 0) or not any(
                       ci >= 0 for row in (obj.sprite or []) for ci in row), \
            f"{name} is not invisible after all"

    by_size: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board.from_engine(eng, names)
        field = _Field(board).build()
        entry = by_size.setdefault((eng.height, eng.width), [set(), []])
        entry[0] |= _compositions(board, field, cap, rng)
        entry[1].append(level)

    bad = 0
    for (h, w), (comps, levels) in sorted(by_size.items()):
        shots = {}
        for comp in sorted(comps):
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in comp}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[comp] = np.asarray(_render_frame(eng, g)).copy()
        keys = sorted(shots)
        clashes = [(a, b) for i, a in enumerate(keys) for b in keys[i + 1:]
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:>2}x{w:<2} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps):>3} "
              f"compositions, {'OK' if not clashes else 'IDENTICAL ' + str(clashes[:3])}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


def _dist_upto(board: _Board, state, limit: int) -> "int | None":
    """The exact number of presses from ``state`` to a win, or None when that is
    more than ``limit``. A plain forward BFS: no reverse sweep, no predecessor
    map, no truncation depth, nothing `_Field` touched."""
    if limit <= 0:
        return None
    seen = {state}
    layer = [state]
    for depth in range(1, limit + 1):
        nxt = []
        for st in layer:
            for e in range(4):
                out, won = board.step(st, e)
                if won:
                    return depth
                if out is not None and out not in seen:
                    seen.add(out)
                    nxt.append(out)
        if not nxt:
            return None
        layer = nxt
    return None


def _verify(seeds: int = 4) -> int:
    """Three independent checks of what the corpus actually ships.

    1. **The inert-marker canonicalisation, measured on the INTERPRETER.** The
       model drops a `PullbackMarker` sitting on a crate's own tether cell,
       which is an argument (see `_Board`) and therefore a thing to check: seat
       each sampled board with those markers restored and require the
       interpreter's next board to be identical either way.
    2. **Every plan length and every shipped tie set, re-priced from the other
       end.** `_Field` answers both from one sweep pair, so a bug in the reverse
       sweep (or in the ball's truncation depth) would give a self-consistent
       field, a plan that still wins, and tie sets that are quietly wrong -- and
       the training labels ARE those tie sets. Each candidate press is therefore
       re-priced by `_dist_upto`, a plain bounded forward BFS that shares
       nothing with `_Field` but `_Board.step`.
    3. **Every plan replayed through the real ADAPTER**, per seed, as the
       SCREEN presses the corpus records. That is the end-to-end statement: the
       native model's plan wins the actual game an agent would be handed. Note
       `set_level` re-seeds the presentation per (seed, LEVEL), so the rotation
       is read after it, not once per seed (see [[ps-pegs-solver]]).
    """
    solver, game, expert = _levels()
    eng = game._engine
    idx, names = game._game.obj_name_to_idx, _ids(game._game)
    rng = random.Random(20260817)

    hidden = groups = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board.from_engine(eng, names)
        field = _Field(board).build()
        seen: dict = {}
        for i, st in enumerate(field.keys):
            if i not in field.dist:
                continue
            # Everything a frame shows: the two digit markers are the only
            # objects in a state that render nothing (`_audit` asserts the rest
            # of the invisible list is bookkeeping derived from the player).
            shown = (st[0], st[1], st[2], st[3], st[4], st[7])
            best = tuple(field.optimal(i))
            if shown in seen:
                groups += 1
                hidden += seen[shown] != best
            else:
                seen[shown] = best
    print(f"0. {groups} pairs of ball states render identically: "
          f"{'every one agrees on its optimal set' if not hidden else f'{hidden} DISAGREE'}")

    inert = same = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board.from_engine(eng, names)
        if not board.pulls:
            continue
        static = _static_grid(eng, idx)
        field = _Field(board).build()
        pool = [k for k in field.keys if k[4]] or field.keys
        for st in rng.sample(pool, min(40, len(pool))):
            restored = st[4] | sum(1 << c for c in set(board.sources))
            for e in range(4):
                _seat(eng, idx, board, static, st)
                eng.step(_DIRS[e])
                want = (_extract(eng, names, board, won=eng.check_win()),
                        eng.check_win())
                _seat(eng, idx, board, static,
                      (st[0], st[1], st[2], st[3], restored, *st[5:]))
                eng.step(_DIRS[e])
                got = (_extract(eng, names, board, won=eng.check_win()),
                       eng.check_win())
                inert += 1
                same += got == want
    print(f"1. {inert} transitions stepped with and without the tether-cell "
          f"markers: {'INERT' if same == inert else f'{inert - same} DIFFERED'}")

    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board.from_engine(eng, names)
        field = _Field(board).build()
        presses, optsets = field.plan()
        t = time.time()
        notes = []
        star = len(presses)
        if _dist_upto(board, board.start, star) != star:
            notes.append(f"LENGTH: an independent BFS disagrees with d*={star}")
        state = 0
        for i, direction in enumerate(presses):
            want = []
            for e in range(4):
                out, won = board.step(field.keys[state], e)
                if won:
                    got = 1
                elif out is None:
                    got = None
                else:
                    d = _dist_upto(board, out, star - i - 1)
                    got = None if d is None else d + 1
                if got == star - i:
                    want.append(_DIRS[e])
            if want != optsets[i]:
                notes.append(f"step {i}: {optsets[i]} != {want}")
            state = field.succ[state][_DIRS.index(direction)]
        bad += len(notes)
        print(f"2. level {level:>2}: d*={star:>3}, "
              f"{sum(len(s) for s in optsets):>3} labelled presses re-priced by "
              f"an independent bounded BFS in {time.time() - t:>5.1f}s: "
              f"{'AGREES' if not notes else '; '.join(notes[:2])}")

    played = wins = 0
    for seed in range(seeds):
        g2 = solver.make_game(seed)
        exp2 = SlideRuleExpert(g2)
        for level in range(g2.n_levels):
            g2.set_level(level)
            plan = exp2.plan(g2._engine, level)
            if plan is None:
                continue
            rot_k, hf, vf = g2._rotation_k, g2._hflip, g2._vflip
            for direction in plan:
                g2.perform_action(ActionInput(
                    id=screen_action(direction, rot_k, hf, vf)))
            played += 1
            wins += g2._state == GameState.WIN
    print(f"3. {played} plans replayed through the adapter over {seeds} seeds: "
          f"{wins} wins" + ("" if wins == played else "  <-- FAILURES"))
    return 0 if not (bad or hidden) and wins == played and same == inert else 1


#: Scene alphabet for `_moves`. Each board is built as a level start (crates at
#: full slack), then the state is edited to the configuration under test.
_SCENE = {
    "#": ("wall",), ".": (), "P": ("playerheavy",), "p": ("playerlight",),
    "F": ("finishx",), "S": ("pillowstand",), "W": ("pillowstand", "pillow"),
    "B": ("pullback",),
    **{str(m): (f"c{m}{m}", "source") for m in range(1, 6)},
}

#: ``(name, rows, edits, press, prose)``. ``edits`` mutates the level start into
#: the configuration under test -- ``paths`` pays out a crate's rope along a
#: named route, ``marked`` stamps pullback markers, ``player`` re-parks the
#: player -- because the interesting boards are ones no shipped level reaches
#: from its own start inside a fuzz walk. What `_moves` asserts is that the
#: INTERPRETER and `_Board.step` agree cell for cell; the prose says what the
#: scenario was built to catch.
_SCENES = (
    ("reel along the last link", ["#######",
                                  "#P....#",
                                  "#..3..#",
                                  "#....F#",
                                  "#######"],
     {"paths": {0: "up,left"}}, "right",
     "a rope that turns a corner reels back along the LAST link paid out, not "
     "toward its tether"),
    ("marker chain", ["#######",
                      "#3....#",
                      "#P....#",
                      "#....F#",
                      "#######"],
     {"paths": {0: "right,right"}, "marked": [(1, 2), (1, 3)]}, "down",
     "two markers on consecutive links of a STRAIGHT rope: the reel rule is "
     "driven to a fixpoint per direction, so both cash in on one tick"),
    ("marker round a bend", ["#######",
                             "#3...F#",
                             "#.....#",
                             "#P....#",
                             "#######"],
     {"paths": {0: "right,down"}, "marked": [(1, 2), (2, 2)]}, "right",
     "the same chain but BENT: the second link needs a direction the rule list "
     "has already run, so exactly one of the two reels"),
    ("rope walls a crate in", ["#######",
                              "#.....#",
                              "#.4...#",
                              "#.....#",
                              "#.P..F#",
                              "#######"],
     {"paths": {0: "right,down,left"}}, "up",
     "a crate's own rope is a CrateBlock, so a rope that loops back above it "
     "fixes it upward and the press is a total no-op"),
    ("drop the pillow mid-slide", ["########",
                                   "#.....F#",
                                   "#pS.2..#",
                                   "#......#",
                                   "########"],
     {}, "right",
     "a light player crossing an empty stand drops the pillow, goes heavy on "
     "the next tick and shoves the crate it would otherwise have stopped at"),
    ("take the pillow mid-slide", ["########",
                                   "#.....F#",
                                   "#P.W.2.#",
                                   "#......#",
                                   "########"],
     {}, "right",
     "the mirror image: a heavy player picks a pillow up in passing and then "
     "stops dead at the crate instead of shoving it"),
    ("spent crate is a wall", ["#######",
                              "#.....#",
                              "#.1.P.#",
                              "#....F#",
                              "#######"],
     {"paths": {0: "left"}, "player": (2, 2)}, "left",
     "a crate at zero slack is fixed in every direction it has no rope in, so "
     "pressing into it changes nothing at all"),
    ("shove it back home", ["#######",
                           "#.....#",
                           "#..2.P#",
                           "#....F#",
                           "#######"],
     {"paths": {0: "left"}, "player": (2, 1)}, "right",
     "pressing along the rope from the far side reels the crate in one link "
     "and the player takes the cell it vacated"),
)


def _scene(eng, idx, rows, edits):
    """``(board, state)`` for one `_SCENES` entry, engine grid loaded."""
    h, w = len(rows), len(rows[0])
    eng.height, eng.width = h, w
    eng.grid = [[{idx["background"]} | {idx[o] for o in _SCENE[ch]}
                 for ch in row] for row in rows]
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()
    board = _Board.from_engine(eng, {v: k for k, v in idx.items()})
    player, light, paths, slacks, marked, incm, decm, pillows = board.start
    paths = [list(p) for p in paths]
    slacks = list(slacks)
    for i, route in edits.get("paths", {}).items():
        for name in route.split(","):
            paths[i].append(board.nxt[_DIRS.index(name)][paths[i][-1]])
            slacks[i] -= 1
    for (r, c) in edits.get("marked", ()):
        marked |= 1 << (r * w + c)
    if "player" in edits:
        player = edits["player"][0] * w + edits["player"][1]
    state = (player, light, tuple(tuple(p) for p in paths), tuple(slacks),
             marked & board.inert_mask, incm, decm, pillows)
    return board, state


def _moves() -> int:
    """Pin every named mechanic to a purpose-built board.

    `--selfcheck` fuzzes the shipped levels, and a certification pass cannot
    find a mechanic the shipped boards do not happen to reach (see
    [[idols-to-the-burnt-god-solver]]). These eight boards are the ones that
    decide the model's hardest lines -- the rule-order fixpoint of the pullback
    reel, a rope that turns a corner, and the two directions of the mid-slide
    weight change -- each stepped on the interpreter and on `_Board` and
    compared cell for cell."""
    _solver, game, _expert = _levels()
    eng = game._engine
    idx, names = game._game.obj_name_to_idx, _ids(game._game)
    bad = 0
    for name, rows, edits, press, prose in _SCENES:
        board, state = _scene(eng, idx, rows, edits)
        static = _static_grid(eng, idx)
        _seat(eng, idx, board, static, state)
        eng.step(press)
        won_i = eng.check_win()
        try:
            out, won_m, tags = board.resolve(state, _DIRS.index(press))
        except _Unmodelled as exc:
            # The model refuses this board; the check is then that the
            # interpreter really does produce something unrepresentable, so the
            # refusal is a fact about the game and not a gap in the model.
            try:
                _extract(eng, names, board, won=won_i)
                ok, note = False, "the interpreter's board WAS representable"
            except AssertionError as bad_board:
                ok, note = True, f"engine: {bad_board}"
            bad += not ok
            print(f"{'ok ' if ok else 'BAD'} {name:<26} {press:<5} "
                  f"{'REFUSED: ' + str(exc):<28} {prose}\n"
                  f"      {note}")
            continue
        got = _extract(eng, names, board, won=won_i)
        want = state if out is None else out
        ok = (got, won_i) == (want, won_m)
        bad += not ok
        print(f"{'ok ' if ok else 'BAD'} {name:<26} {press:<5} "
              f"{','.join(sorted(tags)) or 'noop':<28} {prose}")
        if not ok:
            print(f"      model {want}/{won_m}\n      engine {got}/{won_i}")
    print("every named mechanic agrees with the interpreter" if not bad
          else f"MOVES FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _symmetry(cap: int = 40_000) -> int:
    """Measure which transitions are decided by the board's absolute FACING, and
    whether any of them can reach a training label.

    Nothing in the mechanic names an absolute direction -- but the pullback reel
    is four separate rules run in the fixed order ``left, right, up, down``,
    each to its own fixpoint and never re-run as a group, so a rope carrying
    markers on links that BEND reels a different number of them depending on
    which way it bends (see [[ps-engine-render-gotchas]] trap 7). That is a fact
    about the ENGINE, not about the screen, and the mandatory rotation presents
    the same physical situation both ways, so it has to be counted rather than
    argued about.

    Every state of every ball (sampled past ``cap``, but always including every
    state that carries a pullback marker and every state the shipped plan walks
    through) and every press: turn state and press, step, and require the result
    to be the turn of the untouched one. The transform runs on the MODEL --
    the interpreter is ~78 presses/s and this is six figures of transitions --
    which is sound because `--selfcheck` and `--moves` are what tie the model to
    the interpreter.

    The number that decides anything is the last one: how many of the chiral
    presses are on a SHORTEST path, i.e. could ever appear in a recorded
    optimal-action set. `--report`'s ``desync`` column is the same region seen
    from the other side.
    """
    _solver, game, _expert = _levels()
    eng = game._engine
    names = _ids(game._game)
    rng = random.Random(20260817)
    total = chiral = mirror_only = labelled = shipped = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board.from_engine(eng, names)
        field = _Field(board).build()
        presses, sets = field.plan()
        walk, node = [], 0
        for direction in presses or ():
            walk.append(node)
            nxt = field.succ[node][_DIRS.index(direction)]
            if nxt == field.WIN:
                break
            node = nxt
        onplan = {(walk[i], _DIRS.index(d))
                  for i in range(len(walk)) for d in sets[i]}
        keys = set(range(len(field.keys)))
        if len(keys) > cap:
            keys = set(rng.sample(sorted(keys), cap))
        keys |= {i for i, st in enumerate(field.keys) if st[4]}
        keys |= set(walk)
        keys = sorted(keys)

        rot, mir_only = {}, {}
        for k in range(4):
            for mir in (False, True):
                if (k, mir) == (0, False):
                    continue
                turned = _turn_board(board, k, mir)
                for i in keys:
                    st = field.keys[i]
                    tst = _turn_state(board, st, k, mir)
                    for e in range(4):
                        ref, won = board.step(st, e)
                        got, won2 = turned.step(tst, _turn_dir(e, k, mir))
                        want = (None if ref is None
                                else _turn_state(board, ref, k, mir))
                        if (got, won2) != (want, won):
                            (mir_only if mir else rot)[(i, e)] = True
        bad = set(rot) | set(mir_only)
        only_mirror = set(mir_only) - set(rot)
        onpath = {(i, e) for (i, e) in bad
                  if i in field.dist and _DIRS[e] in field.optimal(i)}
        total += len(keys) * 4
        chiral += len(bad)
        mirror_only += len(only_mirror)
        labelled += len(onpath)
        shipped += len(bad & onplan)
        print(f"level {level:>2}: {len(keys) * 4:>7} transitions x 7 "
              f"presentations, {len(bad):>4} chiral "
              f"({len(only_mirror):>3} split only by a mirror), "
              f"{len(onpath):>3} on a shortest path, "
              f"{len(bad & onplan):>3} in a SHIPPED label")
    print(f"\n{total} transitions, {chiral} decided by the rule order, "
          f"{mirror_only} of them only by a mirror, {labelled} of them on some "
          f"shortest path, {shipped} in a label the corpus actually ships")
    print("no press the corpus labels is order-sensitive: the rotation "
          "augmentation presents nothing inconsistent"
          if not shipped else
          f"WARNING: {shipped} shipped labels are order-sensitive")
    return 0 if not shipped else 1


def _turn_cell(cell, h, w, k, mir):
    r, c = divmod(cell, w)
    if mir:
        c = w - 1 - c
    hh, ww = h, w
    for _ in range(k):
        r, c, hh, ww = c, hh - 1 - r, ww, hh
    return r * ww + c


def _turn_dir(e: int, k: int, mir: bool) -> int:
    d = _DIRS[e]
    if mir:
        d = {"left": "right", "right": "left"}.get(d, d)
    for _ in range(k):
        d = {"up": "right", "right": "down", "down": "left", "left": "up"}[d]
    return _DIRS.index(d)


def _turn_board(board: _Board, k: int, mir: bool) -> _Board:
    h, w = board.h, board.w
    nh, nw = (w, h) if k % 2 else (h, w)

    def t(cell):
        return _turn_cell(cell, h, w, k, mir)

    wall = [False] * (nh * nw)
    block = [False] * (nh * nw)
    for cell in range(h * w):
        wall[t(cell)] = board.wall[cell]
        block[t(cell)] = board.block[cell]
    start = _turn_state(board, board.start, k, mir)
    return _Board(nh, nw, wall, block, [t(c) for c in board.stands],
                  [t(c) for c in board.pulls], t(board.finish), board.strict,
                  board.maxes, start)


def _turn_state(board: _Board, state, k: int, mir: bool):
    h, w = board.h, board.w
    player, light, paths, slacks, marked, incm, decm, pillows = state

    def t(cell):
        return _turn_cell(cell, h, w, k, mir)

    def tmask(bits):
        out = 0
        while bits:
            low = bits & -bits
            out |= 1 << t(low.bit_length() - 1)
            bits ^= low
        return out

    m, im, dm = tmask(marked), tmask(incm), tmask(decm)
    # The pillow bitmask is indexed by the stands' SORTED order, which the turn
    # permutes, so it is rebuilt rather than carried.
    turned_idx = {t(c): i for c, i in board.stand_idx.items()}
    order = {cell: i for i, cell in enumerate(sorted(turned_idx))}
    pm = 0
    for cell, i in board.stand_idx.items():
        if (pillows >> i) & 1:
            pm |= 1 << order[t(cell)]
    return (t(player), light, tuple(tuple(t(c) for c in p) for p in paths),
            slacks, m, im, dm, pm)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true",
                    help="per-level ball size, d* and plan")
    ap.add_argument("--selfcheck", action="store_true",
                    help="differential-fuzz the native model vs the interpreter")
    ap.add_argument("--samples", type=int, default=90,
                    help="--selfcheck: interpreter presses per level")
    ap.add_argument("--verify", action="store_true",
                    help="re-price every label and replay every plan")
    ap.add_argument("--audit", action="store_true",
                    help="assert every reachable cell composition is distinct")
    ap.add_argument("--symmetry", action="store_true",
                    help="measure transitions decided by the board's facing")
    ap.add_argument("--moves", action="store_true",
                    help="pin each named mechanic to a purpose-built board")
    args, rest = ap.parse_known_args(argv)
    if args.report:
        return _report()
    if args.selfcheck:
        return _selfcheck(samples=args.samples)
    if args.verify:
        return _verify()
    if args.audit:
        return _audit()
    if args.symmetry:
        return _symmetry()
    if args.moves:
        return _moves()
    return SlideRuleSolver.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
