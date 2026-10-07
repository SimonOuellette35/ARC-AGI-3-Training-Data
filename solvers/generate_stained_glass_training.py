"""Generate Phase-1 training data for the PuzzleScript game ps:stained_glass
("Stained Glass", @krabby.pabby).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a native model of the
mechanic, the exhaustive bounded oracle that plans on it, the relaxation that
decides whether a level can be won at all (which convicted two of the shipped
ones), and the reports that check every one of those claims against the real
interpreter.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_stained_glass",
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
action (post rotation/flip remap), i.e. the button an agent presses in the
augmented view, so replaying the recorded actions reproduces the recorded frames
exactly. Every expert step carries the full set of equally-optimal presses.

The game
--------
A sokoban whose crates are four TYPED panes of glass -- UpGlass, DownGlass,
LeftGlass, RightGlass -- and whose single target is a Frame with four slots.
The win condition is literally "no glass of any kind is left on the board"
(``All FrameUDLR`` carries no ``on`` clause, and `_check_single_win_condition`
returns True unconditionally for that form), and the only thing that ever
deletes a pane is a frame absorbing it. So every pane must end up in the frame,
and the frame holds one pane of each type.

Everything below was MEASURED against the interpreter (``--mechanics`` is the
executable form of this section), not read off the .txt:

* **A pane is absorbed when it CARRIES the matching force and a frame is
  adjacent in ANY of the four directions** -- not only in the direction it is
  travelling. The rules are written ``[ down UpGlass | Frame ] -> [ no UpGlass |
  FrameU ]`` with no rule direction, so PuzzleScript expands them over all four
  and the ``|`` neighbour can be any side. Pushing a pane sideways PAST a frame
  seats it, and that is not a curiosity: two of the four seats in level 1's
  shortest plan are sideways, and a model that only absorbed the pane travelling
  INTO the frame returns 33 presses there instead of 31.
  The matching force is fixed by the four rules: UpGlass needs ``down``,
  DownGlass ``up``, LeftGlass ``right``, RightGlass ``left``.
* **Absorption happens BEFORE movement**, so a pane shoved into a wall is still
  absorbed by a frame beside it, and the player always steps into the cell the
  pane vacated.
* **A frame that already holds that slot does not absorb and is a wall.** The
  eight absorb rules per pane cover exactly the eight frame variants missing
  that slot, so the frame is a clean 4-bit set.
* **There are no chain pushes.** Only ``[ > Player | <pane> ]`` grants a force;
  a pane shoved into another pane simply refuses, and the player with it.
* **A Spinner is an in-place TYPE CHANGER, and a pane can never enter its
  cell.** ``[ > UpGlass | Spinner ] -> [ > ShardGlass | Spinner ]`` turns a pane
  moving into the spinner into a still-moving green shard -- and then, in the
  SAME tick, ``[ right Player | Shardglass ] -> [ right Player | LeftGlass ]``
  fires. The player pushing the pane is by construction adjacent to it, so that
  rule ALWAYS matches, the shard becomes an ordinary pane again, and the
  replacement drops the force (the rule names no or-group, so
  `_apply_rule_match_forces` takes its trivial per-cell path and pops it). Net
  effect of shoving a pane at a spinner: the pane changes type where it stands,
  nothing moves, and **ShardGlass never survives a tick** -- it is never
  rendered in a settled frame at all.
  The type you get is the one the press direction seats: right -> LeftGlass,
  left -> RightGlass, up -> DownGlass, down -> UpGlass. So a spinner reachable
  only from its left can only ever manufacture LeftGlass.
* A hand-placed shard is converted by any press of an adjacent player, in any of
  the four relative positions, and the player still completes its own move when
  its target cell is free.
* The player may stand on a Spinner (its own collision layer); a pane may not
  (it is converted on the way in). That single fact decides two of the four
  levels -- see below.
* ACTION5 exists in the adapter's action list but no rule reads it: an action
  turn sets no directional force, so nothing matches and the board is unchanged.
  `directions` says so and the search never branches on it.
* ``check_win()`` is False on all four level starts, so no ``vacuous_start_win``.

Two of the four shipped levels could not be won, and were replaced
-----------------------------------------------------------------
Levels 2 and 3 as published are impossible, and it is not a search giving up:
``--proof`` still prints the argument for every level, in a RELAXATION that
deletes every other pane and lets the player teleport -- so it can only ever be
too permissive, and a level it rejects is genuinely lost. Both convictions
followed from the spinner fact above.

* **The old level 2** was two rooms joined by one doorway cell with the Spinner
  standing in it. A pane pushed at a spinner converts where it stands, so no
  pane ever crosses one. The left room held an UpGlass and no frame; that pane
  could never reach a cell adjacent to the only frame, so it could never be
  deleted, so ``No UpGlass`` could never hold.
* **The old level 3** had four panes -- two DownGlass, one UpGlass, one
  LeftGlass -- and one frame, so it needed one pane of each of the four types.
  Its Spinner sat at (4,10) with walls above and below, so the only pane cell
  that could ever border it in the direction of travel was (4,9), reached by
  pushing RIGHT -- and pushing right manufactures LeftGlass. The cell on the
  other side, (4,11), lay in a pocket whose only entrance was the spinner
  itself. RightGlass therefore could not exist anywhere in the level, the
  frame's R slot could never be filled, and four panes cannot fit three slots.

Both were rebuilt around the mechanic the engine actually has, at a size that
renders (see "The art"):

* **Level 2** is the spinner introduction, on the same 7x7 room as levels 0 and
  1. Its four panes are D, L, R, D, so the frame's U slot is the one nothing is
  aimed at: a DownGlass has to be walked to the spinner in the corner, shoved
  DOWN into it to come out an UpGlass, and walked back. 32 presses.
* **Level 3** is the compass rose. Four identical DownGlass panes, one frame,
  and a spinner on each of the frame's four diagonals, so which slot a pane
  fills is decided purely by which side you shove it in from -- and because the
  conversion cell is already beside the frame, the press that re-aims a pane and
  the press that seats it are the same key twice. The room is a cross whose arms
  are long enough that getting to the right side is the puzzle. 32 presses.

The relaxation is implemented once (`_Board.slot_sets`) and turned into a
verdict by a bipartite matching of panes onto (frame, slot) pairs. It still runs
BEFORE the search on every level, because it costs microseconds and an
unwinnable board's component runs past eight million states.

Why a native model, and why an exhaustive one
---------------------------------------------
The interpreter runs this game at ~1350 presses/s, and the four levels have
reachable components of 268, 3.25M, 9.58M and 966k states -- enumerating on
`eng.step` is seconds for one and days for the others. The model here is the rules
above over a state PACKED into a single Python int (player cell, one slot per
pane holding ``cell * 4 + type``, four bits per frame), which runs the same
enumeration natively.

It does not enumerate the whole component either. A layered BFS run only as far
as the first winning edge gives ``d*`` and the complete ball of that radius
(d* = 21, 31, 32 and 32, over balls of 260, 758615, 798938 and 739299 states --
level 2's ball is a twelfth of its component). Every shortest path lies inside
that ball, so a second sweep BACKWARD over the same layers --
"a state at depth j is on a shortest path iff one of its successors at depth
j+1 is" -- marks exactly the shortest-path subgraph without ever building a
reverse edge list, and the plan and its per-step optimal SETS are read off it.
All four plans are therefore provably shortest and every label is exact, not
inferred.

The model is the authority for the plan, never for the frames: `record_level`
drives the real adapter, and ``--ties`` re-derives every optimal-action label by
stepping the interpreter itself.

The art
-------
The sprites were redrawn, because the original ones lost the game at the sizes
it is played at. ``_render_cell_sprite`` samples a 5x5 sprite by centred
nearest neighbour, so at ``cell_px = 4`` -- which is what a 13- or 15-wide board
gets -- sprite row 2 and column 2 are simply dropped, and both replaced levels
were that wide. Three things changed, all of them checked by ``--audit``
(every reachable cell composition rendered as a whole uniform board and required
pairwise distinct, at every cell size the levels use) and by ``--symmetry``:

* **The frame is no longer the wall's colour.** It was a darkgray X and Wall is
  darkgray, so the one thing the whole game is about read as scenery. It is now
  a RED window frame -- a ring with a centre pip -- and each filled slot paints
  the middle three pixels of the edge that pane entered from, in that pane's own
  colour. The four markers are disjoint (they avoid the corners), so all sixteen
  variants are one sprite family rather than sixteen hand-drawn ones.
* **The panes are chunky arrows.** Each still points the way it must be pushed,
  but it now fills the cell instead of being a thin wedge whose point was the
  first thing a downscale threw away.
* **The spinner SAYS what it does.** It was a four-colour dither; it is now a
  cross whose arms are painted the colour of the pane you get by shoving one in
  from that side (top blue = UpGlass, bottom pink = DownGlass, left yellow =
  LeftGlass, right orange = RightGlass).

All three keep the mirror-orbit property the flip augmentation rests on: the
four pane shapes are an exact orbit (L/R are each other's hflip, U/D each
other's vflip), each frame marker is symmetric about its own axis and maps onto
its opposite's cells, and a mirror swaps the spinner's arms exactly as it swaps
the outcomes.

And the new levels are 7x7 and 9x9, so they render at 9 and 7 pixels per cell
instead of 4 -- every sprite row and column survives.

Presentation
------------
The game is in `PuzzleScriptAdapter._FLIP_GAMES` as well as taking the mandatory
rotation, so it is drawn at all sixteen (rotation, hflip, vflip) presentations
rather than four. ``--symmetry`` is the measurement behind that: all four plans
plus a seeded 60-press tail that carries the board off them, driven at every one of
the sixteen, engine grid compared cell for cell after every press -- 0 boards
differ and every plan still wins. The art half of the argument (the panes are
arrows that mirror correctly as SHAPES, while their colours do not follow the
mirror -- which the mandatory rotation already does) is written out in that
report's docstring and beside the `_FLIP_GAMES` entry.

Recovery
--------
``recovery_mode = "reset"``. Stranding a pane is silent and permanent -- shove
one into a corner, or seat the wrong type in a slot another pane needed, and
nothing on screen changes colour; the level is simply over. ``--bfs`` measures
it, and level 0 is the outlier: its walls pin every pane against its own slot,
so all 268 of its reachable states can still win and it cannot be lost at all.
The other three are the opposite -- 10692 of level 1's 3250324 states, 24469 of
level 2's 9581905 and 21154 of level 3's 966170 can still win, so 97.8% to 99.7%
of what a player can reach is already over. Nothing on screen says so. The
exploration prefix flails, one RESET returns to the level start, and the cached
plan replays a guaranteed win from there.

CLI
---
    --plans      per-level ball size, plan length and tie coverage
    --bfs        the exhaustive reachable-state report (winnability + dead ends)
    --ties       re-derive every optimal-action label against the interpreter
    --proof      the two unwinnability arguments, with their relaxation
    --model      differential fuzz of the native model against the interpreter
    --mechanics  the measured rules of the game, asserted on crafted boards
    --symmetry   the same engine directions driven at all 16 presentations
    --audit      assert every reachable cell composition renders distinctly
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                       # noqa: E402
    PSAStarSolver, PSExpert, Plan, restore, snapshot,
)

GAME_NAME = "Stained_Glass"

#: The four presses, in the order that breaks plan ties (so a re-derived plan is
#: byte-identical across processes). ACTION5 is a measured no-op -- see
#: ``--mechanics`` -- and branching on it would buy nothing.
_DIRS = ("up", "down", "left", "right")
_UP, _DOWN, _LEFT, _RIGHT = 0, 1, 2, 3
_OPP = (_DOWN, _UP, _RIGHT, _LEFT)

#: Pane types. The value is also the bit this pane occupies in a frame's mask.
_GU, _GD, _GL, _GR = 0, 1, 2, 3
_PANE_NAME = {_GU: "upglass", _GD: "downglass", _GL: "leftglass",
              _GR: "rightglass"}
_NAME_PANE = {v: k for k, v in _PANE_NAME.items()}

#: The force a pane must carry to be absorbed, and the inverse: the pane a press
#: manufactures at a spinner (or out of a shard), which is exactly the pane that
#: same press would seat.
_ABSORB = {_GU: _DOWN, _GD: _UP, _GL: _RIGHT, _GR: _LEFT}
_MADE_BY = {d: t for t, d in _ABSORB.items()}

#: ``frame`` plus the fifteen filled variants, keyed by the slot mask they hold.
_FRAME_MASK = {}
for _suffix in ("", "u", "d", "l", "r", "ud", "ul", "ur", "rl", "rd", "dl",
                "udl", "udr", "url", "drl", "udlr"):
    _m = 0
    for _ch, _bit in (("u", _GU), ("d", _GD), ("l", _GL), ("r", _GR)):
        if _ch in _suffix:
            _m |= 1 << _bit
    _FRAME_MASK["frame" + _suffix] = _m
_MASK_FRAME = {v: k for k, v in _FRAME_MASK.items()}

#: The level START plans, cached between processes. The searches are the whole
#: cost of generation and are seed-independent, so without a file on disk every
#: `parallelize_generator` shard would re-derive them.
PLAN_CACHE = (Path(__file__).resolve().parent.parent
              / "data" / "stained_glass_plans.json")


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """The measured rules of Stained Glass over a state packed into one int.

    Layout, from the low bits up: the player's flat cell index, then one slot
    per pane the level started with holding ``cell * 4 + type`` (or a sentinel
    that sorts last, so the slot list is canonical when sorted), then four bits
    per frame.

    Everything static -- walls, frames, spinners -- lives on the instance, so a
    state is only the four things that move.
    """

    def __init__(self, eng, gm):
        inv = {v: k for k, v in gm.obj_name_to_idx.items()}
        self.h, self.w = eng.height, eng.width
        n = self.h * self.w
        self.blocked = [False] * n          # wall or frame: nothing enters
        self.isspin = [False] * n
        self.fat = [-1] * n                 # cell -> frame index, else -1
        frames: dict[int, int] = {}
        panes: dict[int, int] = {}
        player = -1
        for r in range(self.h):
            for c in range(self.w):
                cell = r * self.w + c
                for o in eng.grid[r][c]:
                    name = inv[o]
                    if name == "background":
                        continue
                    if name == "wall":
                        self.blocked[cell] = True
                    elif name == "player":
                        player = cell
                    elif name == "spinner":
                        self.isspin[cell] = True
                    elif name in _FRAME_MASK:
                        self.blocked[cell] = True
                        frames[cell] = _FRAME_MASK[name]
                    elif name in _NAME_PANE:
                        panes[cell] = _NAME_PANE[name]
                    else:                                    # pragma: no cover
                        raise AssertionError(f"unmodelled object {name!r}")
        assert player >= 0, "no player on the board"
        # One frame per level. With two, "which frame absorbs a pane touching
        # both" would be decided by the interpreter's rule-match order, which a
        # model cannot mirror; every shipped level has exactly one.
        assert len(frames) == 1, (
            f"{len(frames)} frames on this board; with two, which one absorbs "
            f"a pane touching both is the interpreter's rule-match order")
        for cell in range(n):
            assert not (self.isspin[cell] and self.blocked[cell]), \
                "a spinner under a wall/frame would break the push logic"

        # cell -> the four neighbours (or -1 off the board)
        self.nbr = []
        for r in range(self.h):
            for c in range(self.w):
                row = []
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    rr, cc = r + dr, c + dc
                    row.append(rr * self.w + cc
                               if 0 <= rr < self.h and 0 <= cc < self.w else -1)
                self.nbr.append(row)

        self.fcells = tuple(sorted(frames))
        self.npane = len(panes)
        self.cb = max(1, (n - 1).bit_length())               # player cell bits
        self.sb = (n * 4).bit_length()                       # pane slot bits
        self.sent = (1 << self.sb) - 1                       # "absorbed"
        self.cmask = (1 << self.cb) - 1
        self.smask = self.sent
        self.fbase = self.cb + self.sb * self.npane
        #: The slot field, and its value when every pane has been absorbed.
        self.panefield = ((1 << (self.sb * self.npane)) - 1) << self.cb
        self.allgone = sum(self.sent << (self.cb + self.sb * i)
                           for i in range(self.npane))
        for i, f in enumerate(self.fcells):
            self.fat[f] = i
        self.start = self.pack(player, panes,
                               [frames[f] for f in self.fcells])

    # -- packing -------------------------------------------------------------
    def pack(self, player: int, panes: dict, masks) -> int:
        assert len(panes) <= self.npane, "more panes than this packing holds"
        slots = sorted(c * 4 + t for c, t in panes.items())
        slots += [self.sent] * (self.npane - len(slots))
        key = player
        for i, s in enumerate(slots):
            key |= s << (self.cb + i * self.sb)
        for i, m in enumerate(masks):
            key |= m << (self.fbase + 4 * i)
        return key

    def unpack(self, key: int) -> tuple:
        player = key & self.cmask
        panes = {}
        for i in range(self.npane):
            s = (key >> (self.cb + i * self.sb)) & self.smask
            if s != self.sent:
                panes[s >> 2] = s & 3
        masks = [(key >> (self.fbase + 4 * i)) & 15
                 for i in range(len(self.fcells))]
        return player, panes, masks

    def read_engine(self, eng, gm) -> "int | None":
        """The interpreter's CURRENT grid as a key in THIS board's packing, or
        None when it holds something this packing cannot express (a surviving
        shard, or more panes than the level started with). Used by ``--ties`` to
        look a live interpreter state up in the model's own distance field."""
        inv = {v: k for k, v in gm.obj_name_to_idx.items()}
        player = -1
        panes: dict = {}
        masks = [0] * len(self.fcells)
        for r in range(self.h):
            for c in range(self.w):
                for o in eng.grid[r][c]:
                    name = inv[o]
                    if name == "player":
                        player = r * self.w + c
                    elif name in _NAME_PANE:
                        panes[r * self.w + c] = _NAME_PANE[name]
                    elif name in _FRAME_MASK:
                        masks[self.fat[r * self.w + c]] = _FRAME_MASK[name]
                    elif name == "shardglass":
                        return None
        if player < 0 or len(panes) > self.npane:
            return None
        return self.pack(player, panes, masks)

    def won(self, key: int) -> bool:
        """No pane of any type is left -- which is the whole win condition."""
        return key & self.panefield == self.allgone

    # -- the transition ------------------------------------------------------
    def succ(self, key: int) -> list:
        """The four successors of ``key``, in `_DIRS` order; None where the
        press changes nothing at all (a wall, a refused push, a spinner that
        would re-make the type the pane already has)."""
        player, panes, masks = self.unpack(key)
        out = []
        for di in range(4):
            out.append(self._step(player, panes, masks, di))
        return out

    def _step(self, player: int, panes: dict, masks: list, di: int):
        tgt = self.nbr[player][di]
        if tgt < 0 or self.blocked[tgt]:
            return None                       # a wall or a frame: nothing at all
        t = panes.get(tgt)
        if t is None:
            return self.pack(tgt, panes, masks)            # an ordinary walk
        # The push rule fires: the pane now carries the press's force.
        if _ABSORB[t] == di:
            # Absorption is checked before movement and the frame may be on ANY
            # side, so this runs before the pane is asked to move at all.
            for nd in range(4):
                n = self.nbr[tgt][nd]
                if n < 0:
                    continue
                fi = self.fat[n]
                if fi >= 0 and not (masks[fi] >> t & 1):
                    rest = dict(panes)
                    del rest[tgt]
                    nm = list(masks)
                    nm[fi] |= 1 << t
                    return self.pack(tgt, rest, nm)
        nxt = self.nbr[tgt][di]
        if nxt >= 0 and self.isspin[nxt]:
            # Shard, then the player's own press turns it straight back into a
            # pane, stationary. Nothing moves; only the type changes.
            nt = _MADE_BY[di]
            if nt == t:
                return None
            rest = dict(panes)
            rest[tgt] = nt
            return self.pack(player, rest, masks)
        if nxt >= 0 and not self.blocked[nxt] and nxt not in panes:
            rest = dict(panes)
            del rest[tgt]
            rest[nxt] = t
            return self.pack(tgt, rest, masks)
        return None                            # the push is refused, so is the walk

    # -- the exhaustive bounded oracle ---------------------------------------
    def field(self, node_cap: int = 4_000_000):
        """``(dstar, depth, onpath, stats)`` -- the exact shortest-path oracle,
        or ``dstar = None`` when the level cannot be won (or the cap bites).

        A layered BFS is run only as far as the first layer that touches a win,
        which both proves ``d*`` and leaves the complete ball of radius ``d*``
        behind. Every shortest path lives inside that ball, so a backward sweep
        over the same layers -- a state at depth j is on a shortest path iff one
        of its successors at depth j+1 is -- marks the shortest-path subgraph
        exactly, with no reverse edge list. ``depth[k] == j and k in onpath``
        therefore says "k is exactly ``d* - j`` presses from a win", which is
        what makes the per-step optimal SETS measured rather than inferred."""
        start = self.start
        stats = {"states": 0, "dstar": None, "capped": False}
        if self.won(start):
            return 0, {start: 0}, {start}, stats
        depth = {start: 0}
        layers = [[start]]
        dstar = None
        k = 0
        while True:
            nxt = []
            hit = False
            for s in layers[k]:
                for ns in self.succ(s):
                    if ns is None:
                        continue
                    if self.won(ns):
                        hit = True
                        continue
                    if ns not in depth:
                        depth[ns] = k + 1
                        nxt.append(ns)
            if hit:
                dstar = k + 1
                break
            if not nxt:
                break                          # the component is closed and lost
            if len(depth) > node_cap:
                stats["capped"] = True
                stats["states"] = len(depth)
                return None, depth, set(), stats
            layers.append(nxt)
            k += 1
        stats["states"] = len(depth)
        stats["dstar"] = dstar
        if dstar is None:
            return None, depth, set(), stats

        # Backward sweep: which states lie on a shortest path.
        onpath = set()
        for s in layers[dstar - 1]:
            if any(ns is not None and self.won(ns) for ns in self.succ(s)):
                onpath.add(s)
        for j in range(dstar - 2, -1, -1):
            want = j + 1
            for s in layers[j]:
                for ns in self.succ(s):
                    if ns is not None and depth.get(ns) == want and ns in onpath:
                        onpath.add(s)
                        break
        stats["onpath"] = len(onpath)
        return dstar, depth, onpath, stats

    def optimal_at(self, key: int, j: int, dstar: int, depth: dict,
                   onpath: set) -> list:
        """Every press that is on a shortest path from ``key``, which sits at
        depth ``j``. A winning press counts only from the last layer; anything
        else must land on a state one layer deeper that is itself on a shortest
        path."""
        best = []
        for di, ns in enumerate(self.succ(key)):
            if ns is None:
                continue
            if self.won(ns):
                if j == dstar - 1:
                    best.append(di)
            elif depth.get(ns) == j + 1 and ns in onpath:
                best.append(di)
        return best

    def solve(self, node_cap: int = 4_000_000):
        """``(presses, optsets, stats)`` for the shortest win, or ``(None, None,
        stats)``. Ties are broken by `_DIRS` order, which is what makes a
        re-derived plan byte-identical across processes."""
        dstar, depth, onpath, stats = self.field(node_cap)
        if dstar is None:
            return None, None, stats
        presses, optsets = [], []
        key = self.start
        for j in range(dstar):
            best = self.optimal_at(key, j, dstar, depth, onpath)
            assert best, "a state on a shortest path with no optimal press"
            presses.append(_DIRS[best[0]])
            optsets.append([_DIRS[d] for d in best])
            key = self.succ(key)[best[0]]
        return presses, optsets, stats

    # -- the relaxation the unwinnability proofs are built on ----------------
    def slot_sets(self) -> list:
        """For every pane the level starts with, the ``(frame, slot)`` pairs it
        could EVER be absorbed into, computed with every OTHER pane deleted and
        the player free to stand anywhere it is not a wall.

        Both relaxations only ADD moves, so a pane whose set comes back empty --
        or a set of panes with no perfect matching onto distinct slots -- is
        proof that the level cannot be won, not evidence.

        Returns ``[(start_cell, start_type, reached, slots), ...]`` where
        ``reached`` is the (cell, type) pairs the pane can attain."""
        _player, panes, masks = self.unpack(self.start)
        out = []
        for c0, t0 in sorted(panes.items()):
            seen = {(c0, t0)}
            queue = [(c0, t0)]
            slots = set()
            while queue:
                cell, t = queue.pop()
                for di in range(4):
                    behind = self.nbr[cell][_OPP[di]]
                    if behind < 0 or self.blocked[behind]:
                        continue               # nothing could push from there
                    if _ABSORB[t] == di:
                        for nd in range(4):
                            n = self.nbr[cell][nd]
                            fi = self.fat[n] if n >= 0 else -1
                            # A slot the frame already holds can never be
                            # filled again -- no rule clears a bit.
                            if fi >= 0 and not (masks[fi] >> t & 1):
                                slots.add((fi, t))
                    nxt = self.nbr[cell][di]
                    if nxt < 0:
                        continue
                    if self.isspin[nxt]:
                        step = (cell, _MADE_BY[di])
                    elif self.blocked[nxt]:
                        continue
                    else:
                        step = (nxt, t)
                    if step not in seen:
                        seen.add(step)
                        queue.append(step)
            out.append((c0, t0, seen, slots))
        return out

    def matchable(self, sets=None) -> bool:
        """True iff every pane can be assigned a DISTINCT frame slot out of
        `slot_sets`. False is a proof that the level is unwinnable."""
        sets = self.slot_sets() if sets is None else sets
        taken: dict = {}

        def augment(i, banned):
            for slot in sorted(sets[i][3]):
                if slot in banned:
                    continue
                banned.add(slot)
                if slot not in taken or augment(taken[slot], banned):
                    taken[slot] = i
                    return True
            return False

        for i in range(len(sets)):
            if not augment(i, set()):
                return False
        return True


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class StainedGlassExpert(PSExpert):
    """`PSExpert`'s memo, disk cache and snapshot discipline around `_Board`.

    `_search` is replaced the way ps:dotsnake and ps:crocodiles_love_cookies
    replace it -- the base class keeps the plan memo, the on-disk start-plan
    cache with its staleness check and the level scoping, and only the strategy
    underneath changes. Here the strategy is "build the native model off the
    engine grid, prove the level is winnable at all, then enumerate the ball of
    radius d*", so `heuristic` is never called and asserts rather than returning
    a number nothing would use.

    `_key` is inherited (every non-background cell), which is exact and
    canonical ACROSS levels here -- the walls are in it and no two levels share
    a layout -- so no ``scope_by_level`` is needed.
    """

    directions = list(_DIRS)
    plan_cache_path = PLAN_CACHE

    def heuristic(self, eng) -> int:                        # pragma: no cover
        raise AssertionError(
            "StainedGlassExpert enumerates the ball of radius d*; "
            "heuristic is unused")

    def board(self, eng) -> _Board:
        return _Board(eng, self.g)

    def _search(self, eng) -> "Plan | None":
        board = self.board(eng)
        # The cheap proof first: two of the four levels are lost before a press
        # is made, and enumerating a 3M-state component to discover that would
        # cost minutes per startup.
        if not board.matchable():
            return None
        presses, optsets, _stats = board.solve(self.node_cap)
        return None if presses is None else Plan(presses, optsets)


class StainedGlassSolver(PSAStarSolver):
    game_id = "puzzlescript_stained_glass"
    game_name = GAME_NAME
    expert_cls = StainedGlassExpert

    #: A memory guard on the ball, not a tuning dial: the larger winnable level
    #: reaches 758615 states at d* = 31.
    node_cap = 4_000_000

    #: Room for the longest plan (31 presses) plus the RESET exploration prefix
    #: (10 +/- 5) and the re-plan after it. Well under the adapter's own 200-step
    #: per-level budget, which would otherwise flip a level to GAME_OVER
    #: mid-plan.
    max_steps = 120


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = StainedGlassSolver()
    game = solver.make_game(seed)
    return solver, game, StainedGlassExpert(game,
                                            node_cap=StainedGlassSolver.node_cap)


def _report() -> int:
    """Per-level ball size, plan length and tie coverage."""
    _solver, game, expert = _new()
    eng = game._engine
    total = tied = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        t0 = time.time()
        if not board.matchable():
            print(f"level {level}: {eng.height:2d}x{eng.width:2d} -- "
                  f"UNWINNABLE (proved, see --proof)")
            continue
        presses, optsets, stats = board.solve(expert.node_cap)
        dt = time.time() - t0
        if presses is None:
            print(f"level {level}: no plan ({stats})")
            continue
        ties = sum(1 for s in optsets if len(s) > 1)
        total += len(presses)
        tied += ties
        room = "ok" if len(presses) < game._max_steps else "OVER BUDGET"
        print(f"level {level}: {eng.height:2d}x{eng.width:2d}, "
              f"{stats['states']:7d} states in the ball, "
              f"{len(presses):3d} presses (budget {game._max_steps}, {room}), "
              f"{ties:3d} with a tie set "
              f"({ties / max(1, len(presses)):3.0%}), {dt:6.2f}s")
    print(f"total {total} presses, {tied}/{total} steps with a tie "
          f"({tied / max(1, total):.0%})")
    return 0


def _bfs() -> int:
    """The exhaustive reachable-state report, and the DEAD-END report with it.

    The generator only ever builds the ball of radius ``d*``; this walks the
    WHOLE reachable component and then computes, by a reverse BFS over an
    explicit CSR edge table, which of its states can still win. That is the fact
    worth knowing about this game: stranding a pane against a wall, or seating
    the wrong type in a slot a later pane needed, changes nothing on screen and
    ends the level. The RESET recovery arc is what that measurement is for.

    Levels the relaxation has already convicted are skipped -- their components
    run past eight million states and enumerating them would prove nothing that
    ``--proof`` has not already proved."""
    from array import array

    _solver, game, expert = _new()
    eng = game._engine
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        if not board.matchable():
            print(f"level {level}: UNWINNABLE by the relaxation -- the "
                  f"component is not enumerated (see --proof)")
            continue
        t0 = time.time()
        index = {board.start: 0}
        order = [board.start]
        edges = array("i")           # 4 per state: -1 no-op, -2 win, else index
        wins = 0
        i = 0
        while i < len(order):
            for ns in board.succ(order[i]):
                if ns is None:
                    edges.append(-1)
                elif board.won(ns):
                    edges.append(-2)
                    wins += 1
                else:
                    j = index.get(ns)
                    if j is None:
                        j = index[ns] = len(order)
                        order.append(ns)
                    edges.append(j)
            i += 1
        n = len(order)
        # Reverse edges as CSR -- a list-of-lists costs more than the states.
        deg = array("i", bytes(4 * (n + 1)))
        for e in edges:
            if e >= 0:
                deg[e + 1] += 1
        for k in range(n):
            deg[k + 1] += deg[k]
        fill = array("i", deg[:n])
        rev = array("i", bytes(4 * deg[n]))
        for s in range(n):
            for k in range(4):
                e = edges[4 * s + k]
                if e >= 0:
                    rev[fill[e]] = s
                    fill[e] += 1
        live = bytearray(n)
        frontier = [s for s in range(n)
                    if -2 in edges[4 * s:4 * s + 4]]
        for s in frontier:
            live[s] = 1
        while frontier:
            nxt = []
            for s in frontier:
                for q in range(deg[s], deg[s + 1]):
                    pr = rev[q]
                    if not live[pr]:
                        live[pr] = 1
                        nxt.append(pr)
            frontier = nxt
        alive = sum(live)
        print(f"level {level}: {n:8d} reachable states ({alive} can still win, "
              f"{n - alive} stranded = {(n - alive) / n:.1%}, {wins} winning "
              f"presses), {time.time() - t0:6.1f}s -- start "
              f"{'can win' if live[0] else 'CANNOT WIN'}")
    return 0


def _ties() -> int:
    """Re-derive every optimal-action label against the INTERPRETER.

    The plan is replayed on the real engine, and at each step every candidate
    press is taken from that live board and the state it lands in is read back
    into the model's packing and looked up in the shortest-path field. That
    checks four things the model cannot check about itself: that the
    interpreter's successors are the ones the field was built from (a divergence
    shows up as a state the field has never seen), that the labels are the
    COMPLETE set of shortest presses, that the plan's own press is in its own
    label, and -- because the replay runs to the end -- that the plan wins on
    the interpreter rather than only on the model."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        if not board.matchable():
            print(f"level {level}: unwinnable (skipped)")
            continue
        dstar, depth, onpath, _stats = board.field(expert.node_cap)
        if dstar is None:
            print(f"level {level}: no plan (skipped)")
            continue
        presses, optsets, _st = board.solve(expert.node_cap)
        game.set_level(level)
        checked = won = 0
        for pi, (taken, claimed) in enumerate(zip(presses, optsets)):
            here = snapshot(eng)
            live = board.read_engine(eng, expert.g)
            if live is None or depth.get(live) != pi or (
                    pi and live not in onpath):
                print(f"  level {level} step {pi}: the interpreter is in a "
                      f"state the field does not place at depth {pi}")
                bad += 1
            measured = []
            for d in _DIRS:
                eng.step(d)
                if eng.check_win():
                    optimal = (pi == dstar - 1)
                else:
                    nk = board.read_engine(eng, expert.g)
                    optimal = (nk is not None and depth.get(nk) == pi + 1
                               and nk in onpath)
                restore(eng, here)
                if optimal:
                    measured.append(d)
            if measured != list(claimed):
                print(f"  level {level} step {pi}: labelled {list(claimed)} "
                      f"but measured {measured}")
                bad += 1
            if taken not in claimed:
                print(f"  level {level} step {pi}: took {taken}, not in its "
                      f"own optimal set {list(claimed)}")
                bad += 1
            checked += 1
            eng.step(taken)
            if eng.check_win():
                won = 1
                break
        if not won:
            print(f"  level {level}: the plan did not reach a win")
            bad += 1
        print(f"level {level}: {checked} steps verified, plan "
              f"{'WINS' if won else 'FAILED'}")
    print("ties clean" if not bad else f"TIES FAILED: {bad} mismatches")
    return 0 if not bad else 1


def _proof() -> int:
    """The unwinnability arguments, printed with the relaxation they rest on.

    For every pane: how many (cell, type) pairs it could ever attain if every
    other pane were deleted and the player could stand anywhere, and which frame
    slots it could then be absorbed into. Both relaxations only ADD moves, so an
    empty slot set -- or a set of panes with no perfect matching onto DISTINCT
    slots -- is a proof of unwinnability rather than a failure to find a plan.
    Levels that pass the test are then actually solved, so the report never
    leaves "the relaxation allowed it" as the last word."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        sets = board.slot_sets()
        print(f"level {level}: {board.h}x{board.w}, {board.npane} panes, "
              f"{len(board.fcells)} frame, "
              f"{sum(board.isspin)} spinner")
        for c0, t0, reached, slots in sets:
            names = ",".join(sorted(_PANE_NAME[t][0].upper() for _f, t in slots)
                             ) or "NONE"
            print(f"  {_PANE_NAME[t0]:11s} at "
                  f"({c0 // board.w},{c0 % board.w}): "
                  f"{len(reached):4d} (cell,type) pairs reachable, "
                  f"absorbable into slots {{{names}}}")
        if board.matchable(sets):
            presses, _opt, stats = board.solve(expert.node_cap)
            verdict = (f"WINNABLE, shortest {len(presses)} presses"
                       if presses is not None else
                       f"the matching allows it but no plan was found {stats}")
            if presses is None:
                bad += 1
        else:
            verdict = ("PROVED UNWINNABLE: the panes cannot be matched onto "
                       "distinct frame slots even under the relaxation")
        print(f"  -> {verdict}")
    print("proof clean" if not bad else f"PROOF FAILED: {bad} levels")
    return 0 if not bad else 1


def _model() -> int:
    """Differential fuzz of the native model against the interpreter.

    Three modes, because the first two are blind to the case a shortest plan
    manufactures:

      * RANDOM play from the level start;
      * GUIDED -- restarted from every prefix of the level's own solution, which
        is the only cheap way to reach the boards a plan sets up on purpose
        (a pane parked beside the frame, a slot already filled);
      * CROWDED -- boards seeded with extra panes and a pre-filled frame, so the
        configurations that are rare in play (a pane wedged against another, a
        frame that must REFUSE a pane it already holds) become the common case.

    Every press of every mode is compared cell-for-cell: the interpreter's grid
    after ``eng.step(d)`` against the model's successor, seated back onto a
    fresh grid. A mismatch prints the board it happened on."""
    _solver, game, expert = _new()
    eng, gm = game._engine, expert.g
    idx = gm.obj_name_to_idx
    rng = random.Random(20260818)
    bad = 0

    def seat(board, key):
        player, panes, masks = board.unpack(key)
        grid = [[{idx["background"]} for _ in range(board.w)]
                for _ in range(board.h)]
        for cell in range(board.h * board.w):
            r, c = divmod(cell, board.w)
            if board.isspin[cell]:
                grid[r][c].add(idx["spinner"])
            if board.blocked[cell] and cell not in board.fcells:
                grid[r][c].add(idx["wall"])
        for i, f in enumerate(board.fcells):
            grid[f // board.w][f % board.w].add(idx[_MASK_FRAME[masks[i]]])
        for cell, t in panes.items():
            grid[cell // board.w][cell % board.w].add(idx[_PANE_NAME[t]])
        grid[player // board.w][player % board.w].add(idx["player"])
        eng.height, eng.width = board.h, board.w
        eng.grid = grid
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    def read(board):
        """The interpreter's grid, as a model key -- or None if it holds
        something the model has no state for (a surviving shard)."""
        inv = {v: k for k, v in gm.obj_name_to_idx.items()}
        player = -1
        panes = {}
        masks = [0] * len(board.fcells)
        for r in range(board.h):
            for c in range(board.w):
                for o in eng.grid[r][c]:
                    name = inv[o]
                    if name == "player":
                        player = r * board.w + c
                    elif name in _NAME_PANE:
                        panes[r * board.w + c] = _NAME_PANE[name]
                    elif name in _FRAME_MASK:
                        masks[board.fcells.index(r * board.w + c)] = \
                            _FRAME_MASK[name]
                    elif name == "shardglass":
                        return "SHARD SURVIVED"
        return board.pack(player, panes, masks)

    def compare(board, key, tag):
        nonlocal bad
        mine = board.succ(key)
        for di, d in enumerate(_DIRS):
            seat(board, key)
            eng.step(d)
            real = read(board)
            want = key if mine[di] is None else mine[di]
            if real != want:
                bad += 1
                if bad <= 5:
                    print(f"  MISMATCH [{tag}] key={key} press={d}")
                    print(f"    model {board.unpack(want) if isinstance(want, int) else want}")
                    print(f"    real  {board.unpack(real) if isinstance(real, int) else real}")

    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        n = 0

        # 1 -- random play from the start.
        for _ in range(40):
            key = board.start
            for _ in range(40):
                compare(board, key, f"L{level} random")
                n += 1
                live = [s for s in board.succ(key) if s is not None]
                if not live:
                    break
                key = rng.choice(live)
                if board.won(key):
                    break

        # 2 -- guided: restart from every prefix of the level's own solution.
        presses = (board.solve(expert.node_cap)[0]
                   if board.matchable() else None)
        if presses:
            key = board.start
            prefix = [key]
            for d in presses:
                key = board.succ(key)[_DIRS.index(d)]
                prefix.append(key)
            for base in prefix:
                if board.won(base):
                    continue
                for _ in range(6):
                    key = base
                    for _ in range(8):
                        compare(board, key, f"L{level} guided")
                        n += 1
                        live = [s for s in board.succ(key) if s is not None]
                        if not live:
                            break
                        key = rng.choice(live)
                        if board.won(key):
                            break

        # 3 -- crowded: more panes than the level ships and a pre-filled
        # frame, so "a pane wedged against another" and "a frame that must
        # REFUSE a pane it already holds" stop being rare. The board is seated
        # on the engine and the model is rebuilt FROM it, so the extra panes
        # are inside the packing rather than overflowing it.
        free = [c for c in range(board.h * board.w)
                if not board.blocked[c] and not board.isspin[c]]
        fcell = board.fcells[0]
        for _ in range(120):
            cells = rng.sample(free, min(len(free), rng.randint(4, 8)))
            eng.height, eng.width = board.h, board.w
            eng.grid = [[{idx["background"]} for _ in range(board.w)]
                        for _ in range(board.h)]
            for cell in range(board.h * board.w):
                r, c = divmod(cell, board.w)
                if board.isspin[cell]:
                    eng.grid[r][c].add(idx["spinner"])
                if board.blocked[cell] and cell != fcell:
                    eng.grid[r][c].add(idx["wall"])
            eng.grid[fcell // board.w][fcell % board.w].add(
                idx[_MASK_FRAME[rng.randrange(16)]])
            eng.grid[cells[0] // board.w][cells[0] % board.w].add(idx["player"])
            for cell in cells[1:]:
                eng.grid[cell // board.w][cell % board.w].add(
                    idx[_PANE_NAME[rng.randrange(4)]])
            eng._position_index_dirty = True
            eng._rule_noop_cache.clear()
            crowd = _Board(eng, gm)
            key = crowd.start
            for _ in range(6):
                compare(crowd, key, f"L{level} crowded")
                n += 1
                live = [x for x in crowd.succ(key) if x is not None]
                if not live:
                    break
                key = rng.choice(live)
                if crowd.won(key):
                    break
        print(f"level {level}: {n * 4} transitions fuzzed "
              f"(random + guided + crowded)")
    print("model clean" if not bad else f"MODEL FAILED: {bad} mismatches")
    return 0 if not bad else 1


def _mechanics() -> int:
    """The measured rules of the game, asserted on crafted boards.

    Every claim in the module docstring's "The game" section that a reader could
    otherwise only take on trust, executed. The boards are built by hand rather
    than reached by play so each one isolates exactly one rule."""
    _solver, game, expert = _new()
    eng, gm = game._engine, expert.g
    idx = gm.obj_name_to_idx
    chars = {".": (), "#": ("wall",), "P": ("player",), "S": ("spinner",),
             "U": ("upglass",), "D": ("downglass",), "L": ("leftglass",),
             "R": ("rightglass",), "G": ("shardglass",), "F": ("frame",),
             "u": ("frameu",), "d": ("framed",), "l": ("framel",),
             "r": ("framer",), "X": ("frameudlr",), "p": ("spinner", "player")}

    def build(rows):
        eng.height, eng.width = len(rows), len(rows[0])
        eng.grid = [[{idx["background"]} | {idx[o] for o in chars[ch]}
                     for ch in row] for row in rows]
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    def cells(name):
        o = idx[name]
        return sorted((r, c) for r in range(eng.height)
                      for c in range(eng.width) if o in eng.grid[r][c])

    def press(rows, d):
        build(rows)
        eng.step(d)
        out = {"player": cells("player"), "shard": cells("shardglass"),
               "win": bool(eng.check_win())}
        for n in _PANE_NAME.values():
            out[n] = cells(n)
        for n in _FRAME_MASK:
            if cells(n):
                out.setdefault("frames", []).append((n, cells(n)))
        out.setdefault("frames", [])
        return out

    bad = 0

    def check(name, got, **want):
        nonlocal bad
        wrong = {k: (v, got.get(k)) for k, v in want.items() if got.get(k) != v}
        print(f"  {'ok  ' if not wrong else 'FAIL'} {name}")
        for k, (w, x) in wrong.items():
            print(f"         {k}: wanted {w}, measured {x}")
        bad += len(wrong)

    print("the push")
    check("moves one pane one cell and the player follows",
          press(["######", "#....#", "#PU..#", "#....#", "######"], "right"),
          upglass=[(2, 3)], player=[(2, 2)])
    check("does not chain: a pane shoved into a pane refuses, and so does you",
          press(["######", "#....#", "#PUD.#", "#....#", "######"], "right"),
          upglass=[(2, 2)], downglass=[(2, 3)], player=[(2, 1)])

    print("the frame")
    check("absorbs a pane carrying the matching force (UpGlass needs down)",
          press(["#####", "#.P.#", "#.U.#", "#.F.#", "#####"], "down"),
          upglass=[], frames=[("frameu", [(3, 2)])], player=[(2, 2)], win=True)
    check("absorbs from ANY side, not only the way the pane is travelling",
          press(["#####", "#..P#", "#.FU#", "#...#", "#####"], "down"),
          upglass=[], frames=[("frameu", [(2, 2)])], player=[(2, 3)], win=True)
    check("absorbs before movement, so a pane shoved into a wall still seats",
          press(["#####", "#.P.#", "#FU.#", "#####", "#####"], "down"),
          upglass=[], frames=[("frameu", [(2, 1)])], player=[(2, 2)])
    check("ignores a pane whose force does not match (UpGlass pushed up)",
          press(["#####", "#.U.#", "#.F.#", "#.P.#", "#####"], "up"),
          upglass=[(1, 2)], frames=[("frame", [(2, 2)])], player=[(3, 2)])
    check("does not take a slot twice, and is then just a wall",
          press(["#####", "#.P.#", "#.U.#", "#.u.#", "#####"], "down"),
          upglass=[(2, 2)], frames=[("frameu", [(3, 2)])], player=[(1, 2)])
    check("is not reached diagonally",
          press(["#####", "#.P.#", "#.U.#", "#F..#", "#####"], "down"),
          upglass=[(3, 2)], frames=[("frame", [(3, 1)])], player=[(2, 2)])

    print("the spinner")
    for d, rows, want in (
            ("right", ["#####", "#...#", "#PDS#", "#...#", "#####"], "leftglass"),
            ("left", ["#####", "#...#", "#SDP#", "#...#", "#####"], "rightglass"),
            ("down", ["#####", "#.P.#", "#.D.#", "#.S.#", "#####"], "upglass")):
        got = press(rows, d)
        check(f"pushing a pane {d} at it makes a {want} WHERE IT STANDS",
              got, **{want: [(2, 2)]}, shard=[])
    check("is a no-op when the press would re-make the type the pane has",
          press(["#####", "#.S.#", "#.D.#", "#.P.#", "#####"], "up"),
          downglass=[(2, 2)], player=[(3, 2)])
    check("never lets a pane onto its cell",
          press(["#####", "#...#", "#PDS#", "#...#", "#####"], "right"),
          leftglass=[(2, 2)], player=[(2, 1)])
    check("does not stop the PLAYER walking onto it",
          press(["#####", "#...#", "#PS.#", "#...#", "#####"], "right"),
          player=[(2, 2)])

    print("the shard")
    for d, rows, want in (("right", ["#####", "#...#", "#GP.#", "#...#", "#####"],
                           "leftglass"),
                          ("up", ["#####", "#...#", "#GP.#", "#...#", "#####"],
                           "downglass"),
                          ("left", ["#####", "#...#", "#.PG#", "#...#", "#####"],
                           "rightglass")):
        check(f"is turned into a {want} by a press of {d}, from any side",
              press(rows, d), **{want: [(2, 1) if d != "left" else (2, 3)]},
              shard=[])
    check("is converted even when the player itself is blocked",
          press(["#####", "#...#", "#GP##", "#...#", "#####"], "right"),
          leftglass=[(2, 1)], player=[(2, 2)])
    check("does not survive its own tick when a pane is pushed at a spinner",
          press(["######", "#....#", "#PDSU#", "#....#", "######"], "right"),
          shard=[], leftglass=[(2, 2)], upglass=[(2, 4)])

    print("the win condition")
    check("is exactly 'no pane left' -- All FrameUDLR has no `on` clause",
          press(["#####", "#...#", "#.XP#", "#...#", "#####"], "up"), win=True)
    check("is not satisfied while a pane is still on the board",
          press(["#####", "#..P#", "#.XU#", "#...#", "#####"], "up"), win=False)

    print("ACTION5")
    build(["#####", "#...#", "#.P.#", "#...#", "#####"])
    before = [[set(c) for c in row] for row in eng.grid]
    eng.step("action")
    same = before == [[set(c) for c in row] for row in eng.grid]
    print(f"  {'ok  ' if same else 'FAIL'} no rule reads it: the board is "
          f"unchanged")
    bad += int(not same)

    print("the level starts")
    for level in range(game.n_levels):
        game.set_level(level)
        ok = not eng.check_win()
        print(f"  {'ok  ' if ok else 'FAIL'} level {level} does not read as "
              f"already won")
        bad += int(not ok)

    print("mechanics clean" if not bad else f"MECHANICS FAILED: {bad} claims")
    return 0 if not bad else 1


def _symmetry() -> int:
    """Drive the same ENGINE directions at every presentation the adapter could
    draw, and require the board to come out identical every time.

    Two things are being measured, and neither is asserted:

    * **the rotation contract.** `screen_action` has to invert the adapter's own
      forward remap, and getting it backwards does not raise -- it just drives a
      different direction on three of every four orientations. Every plan is
      replayed at each ``(rotation, hflip, vflip)`` and has to WIN.
    * **the flip augmentation.** The eight mirrored presentations are forced on
      by hand here, whether or not the game is currently in
      `PuzzleScriptAdapter._FLIP_GAMES`, so the claim "this game is
      mirror-symmetric" is a measurement rather than an opinion: the adapter
      mirrors only the PRESENTATION, so if the claim holds the ENGINE grid after
      a given press must be identical at all sixteen.

    What the engine comparison canNOT see is the ART, so that argument is
    written out rather than measured: every pane is an arrow pointing the way it
    must be pushed, and the four sprites are an exact orbit of the dihedral
    group AS SHAPES (LeftGlass and RightGlass are each other's hflip, UpGlass
    and DownGlass each other's vflip), so a mirrored board still shows each pane
    pointing where the remapped press sends it. Their COLOURS do not follow the
    mirror -- a mirrored LeftGlass is a yellow right-pointing arrow, which the
    unmirrored game never shows -- but the mandatory ROTATION already permutes
    colour against direction the same way, so the flip adds no ambiguity that
    was not there, and within one episode the mapping is constant and visible.
    """
    from arcengine import ActionInput, GameState                # noqa: PLC0415

    from solvers.common.ps_astar import screen_action           # noqa: PLC0415

    _solver, game, expert = _new()
    eng = game._engine
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        found = expert.plan(eng, level)
        if found is not None:
            plans[level] = found
    levels = sorted(plans)

    # A fixed press script per level: the plan, then a random walk that carries
    # the board off it (the plan alone only visits states the search liked).
    rng = random.Random(20260818)
    scripts = {lvl: list(plan) + [rng.choice(_DIRS) for _ in range(60)]
               for lvl, plan in plans.items()}

    natural = set()
    for seed in range(64):
        game._seed = seed
        for level in levels:
            game.set_level(level)
            natural.add((game._rotation_k, game._hflip, game._vflip))

    baseline: dict = {}
    bad = 0
    for rot in range(4):
        for hflip in (False, True):
            for vflip in (False, True):
                wins = drift = 0
                for level, script in scripts.items():
                    game.set_level(level)
                    # Forced, so the eight mirrored presentations are measured
                    # even while the game is not in _FLIP_GAMES.
                    game._rotation_k, game._hflip, game._vflip = rot, hflip, vflip
                    boards = []
                    won = False
                    for i, direction in enumerate(script):
                        game.perform_action(ActionInput(
                            id=screen_action(direction, rot, hflip, vflip)))
                        boards.append(snapshot(eng))
                        if not won and i == len(plans[level]) - 1:
                            won = game._state == GameState.WIN
                    wins += won
                    ref = baseline.setdefault(level, boards)
                    drift += sum(1 for a, b in zip(ref, boards) if a != b)
                bad += (wins != len(scripts)) + (drift > 0)
                mark = "" if (rot, hflip, vflip) in natural else "  (not drawn today)"
                print(f"rot={rot} hflip={int(hflip)} vflip={int(vflip)}: "
                      f"{wins}/{len(scripts)} levels win, {drift} board(s) "
                      f"differ from the unaugmented run{mark}")
    print(f"the adapter currently draws {len(natural)} of the 16 presentations")
    print("symmetry clean over all 16 presentations" if not bad
          else f"SYMMETRY FAILED: {bad}")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every cell size the boards use.

    Whole 64x64 frames of uniform boards are compared rather than one cell out
    of a mixed board: `_render_frame` centre-pads a non-square board, so
    indexing a cell by ``cell_px * r`` reads the wrong pixels (the ps:explod
    lesson). Two uniform boards render identically iff their cells do.

    Which compositions matter is measured, not guessed: the ball of radius d* is
    scanned for the cell contents it holds. Two things are added to it by hand
    and named rather than assumed:

      * the WINNING board -- it is a state no expansion stores (a winning press
        is an edge), and it is the one frame the agent has to recognise;
      * ShardGlass is deliberately EXCLUDED. It cannot survive the tick that
        creates it (``--mechanics``), the recorder tapes settled frames only,
        and so it is never shown.
    """
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, expert = _new()
    eng, gm = game._engine, expert.g
    idx = gm.obj_name_to_idx
    bg = idx["background"]

    layer1 = ["spinner"]
    layer2 = (["player", "wall", "shardglass"] + list(_PANE_NAME.values())
              + list(_FRAME_MASK))
    comps = {}
    for a in [None] + layer1:
        for b in [None] + layer2:
            objs = tuple(x for x in (a, b) if x)
            comps["+".join(objs) or "floor"] = objs

    sizes: dict = {}
    reachable = {"floor", "wall", "player"}
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        sizes.setdefault((eng.height, eng.width), []).append(level)
        if not board.matchable():
            continue                       # never recorded: see --proof
        keys = {board.start}
        frontier = [board.start]
        while frontier:                    # the ball, re-walked cheaply
            nxt = []
            for k in frontier:
                for ns in board.succ(k):
                    if ns is not None and ns not in keys:
                        keys.add(ns)
                        nxt.append(ns)
            frontier = nxt
            if len(keys) > 2_000_000:
                break
        for k in keys:
            _p, panes, masks = board.unpack(k)
            for t in panes.values():
                reachable.add(_PANE_NAME[t])
            for m in masks:
                reachable.add(_MASK_FRAME[m])
        if board.isspin.count(True):
            reachable.add("spinner")
            reachable.add("spinner+player")
        reachable.add(_MASK_FRAME[15])     # the WIN board, by hand

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{bg} | {idx[o] for o in objs} for _ in range(w)]
                        for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, gm)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        fatal = [(a, b) for a, b in clashes
                 if a in reachable and b in reachable]
        bad += len(fatal)
        note = "OK" if not fatal else f"IDENTICAL {fatal}"
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{len(reachable)} reachable of {len(comps)} compositions -- "
              f"{note}")
        if clashes and not fatal:
            print(f"     {len(clashes)} clashing pairs, none reachable: "
                  f"{clashes}")
    print(f"reachable compositions: {sorted(reachable)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable reachable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--bfs" in sys.argv:
        sys.exit(_bfs())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--proof" in sys.argv:
        sys.exit(_proof())
    if "--model" in sys.argv:
        sys.exit(_model())
    if "--mechanics" in sys.argv:
        sys.exit(_mechanics())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(StainedGlassSolver.main())
