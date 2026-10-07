"""Generate Phase-1 training data for the PuzzleScript game ps:flood
("Flood", Franklin P. Dyer -- the one where you do not race the water to an
exit, you shut it out and then stand still while it rises).

The harness -- the rotation contract, the trajectory recorder, the plan cache
and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared
with the other ps: generators. This file is the game-specific part: a native
model of the flood ruleset, the searches over it, and the rendering fix.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_ps_flood",
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
Every step also carries a set of equally-optimal presses.

The game
--------
Water rises from one tile and spreads one cell per press. You push crates (one
press moves the whole contiguous run in front of you) to wall it off, and you
win the moment the water has nowhere left to go -- ``No Wave3 and Some Player``.

FIVE RULES THAT SHAPE EVERY PLAN, and only the first is visible in the art:

  * THE WIN IS THE WATER STOPPING, NOT YOU ARRIVING. There is no exit. Once the
    board is sealed the rest of the plan is literally waiting: the back half of
    every level here is a run of presses that do nothing while the flood
    finishes filling the cells it can still reach. That also makes the plan
    LENGTH a property of the water, not of the player -- the same seal built
    three presses earlier wins on the same press -- which is why so many steps
    have a large tie set.
  * THE LEVEL STARTS ALREADY "WON". ``No Wave3`` holds before the first press
    (nothing has spread yet), so the win predicate is vacuously true at reset.
    Everything that tests it -- the search, `discover_solvable`, `record_level`
    -- has to ignore it until a press has happened; see ``vacuous_start_win``.
  * RULES RUN BEFORE MOVEMENT. PuzzleScript applies the whole rule list, THEN
    resolves forces. So the water spreads from where you were standing, not
    where you are going: you cannot dodge out of a cell the flood enters this
    press, and being orthogonally adjacent to live water at the start of a turn
    is already fatal. Water and player share a collision layer, so the Wave3
    lands ON the player and deletes it -- silently. There is no GAME_OVER; a
    drowned level simply can never be won.
    The same ordering means a button is read BEFORE you step onto it, so a door
    shuts one press after you arrive, and re-opens one press after you leave.
  * A DOOR IS A MAJORITY OF ONE. ``[Pusher Button][Open] -> [...][Closed]``
    then ``[No Pusher Button][Closed] -> [...][Open]``: the second rule undoes
    the first unless EVERY button on the board is held down by a crate or by
    you. And Closed lands on the wall layer, so a door shutting on a crate --
    or on you -- destroys it. Water flowing over an Open door destroys the door
    instead (``[Wave Open] -> [Wave No Open]``), permanently: a doorway the
    flood reaches first can never be shut.
  * WATER THAT POURED DOWN GOES INERT, AND CAN EVAPORATE. The DOWN spread rule
    turns its own source into Wave4, which never spreads again; a Wave4 walled
    in above and below becomes Wave2, and a Wave2 walled in left and right is
    DELETED. The hole is re-flooded by a neighbour on the same press, which
    costs one more Wave3 and so one more press before the win. It is a
    transient, not a cycle -- the cell below has filled by then, so the
    refilled cell comes back as Wave1 and stays -- but it is worth exactly the
    press it costs, and a model that ignores it gets the plan length wrong.

The expert
----------
A NATIVE model of the ruleset (`_Board`, state = the player's cell plus seven
bitmasks) and two searches over it. The model exists because the interpreter
runs this game at 250-1500 presses/s -- the wave rules rescan the grid five
times a turn -- where the model runs at ~50k, and the exhaustive search below
needs 172k states on one level. ``--fuzz`` plays random boards press-for-press
against the real interpreter and compares every mask and the win flag after
every press. It was accepted on 105k random-board transitions plus 24k on the
shipped levels, all with zero mismatches, and the random boards reach the rare
paths on purpose: one 45k run covered 3,190 Wave4->Wave2 conversions, 841
Wave2 deletions with more than one candidate in the row, 573 doors shutting
(17 of them on a crate or on the player) and 1,057 drownings. Re-run it after
ANY change to the .txt rules or to the adapter's rule code.

`_search` runs, in order:

  * an exhaustive layered BFS, which returns a genuinely SHORTEST plan and, in
    the same sweep, the EXACT set of equally-shortest presses at every step
    (see `_Board.bfs`);
  * a width-capped beam for the one level the BFS cannot hold, ranked by
    `fill_time` and aiming at a state the player can simply WAIT out.

THE ONE HEURISTIC. `fill_time` is a multi-source flood from the live water
front over the cells it can still enter: the presses the flood needs to reach
the last of them if nothing else changes. Once the player is walled off it is
exactly one short of the presses left to WIN -- the last cell arrives as a
Wave3 and the win is the press after it settles -- and before that it is an
over-estimate, since every crate still to be placed shrinks the region. So it
ranks the beam and certifies nothing.

The levels
----------
All nine are Dyer's own boards in shipped order (the ``message`` screens
between them are not levels and the adapter does not count them).

    level  size    crates  plan  states  found by  title
    0      7x7      1        7      125  BFS       Blockage
    1      7x7      2        6      135  BFS       Close the door behind you
    2      13x13    6       25  172,179  BFS       Barricade
    3      7x7      0        8       44  BFS       Button
    4      14x14    5       17  154,343  BFS       Button or Bust
    5      13x13    5       34        -  beam      Busted Pipe
    6      13x13    5       10    1,030  BFS       Hidden Switch
    7      13x13   19        8   10,222  BFS       Simpler than it Seems
    8      15x16    0       45      440  BFS       (untitled: you start sealed in)

160 presses over the nine, every one replayed through the real interpreter to a
WIN by ``--verify``, and all inside the adapter's 200-step per-level budget
with room for the exploration prefix. Eight of the nine plans are certified
shortest by the BFS. The whole cold run is ~10 seconds, which is why the disk
plan cache here is a convenience rather than the necessity it is for the
heavier games in this family.

LEVEL 5 IS THE ONE THE BFS CANNOT HOLD. Five loose crates in an open pocket is
millions of states before the plan is half built, so it is beam-searched
instead. The plan it finds is 34 presses -- and 34 is also exactly how long the
water alone needs to finish once the pocket is sealed, measured by walling the
seal cell off at press zero and pressing nothing. So the plan spends no press
the flood does not spend anyway, and it is shortest in everything but name. It
is still labelled with singleton optimal sets: a beam cannot prove a sibling
press ties.

It is also the level where the search does not play the author's puzzle.
"Busted Pipe" lays out five crates and five buttons and means for you to hold
the door shut with all five at once; the search instead shoves ONE crate up
through the doorway in 14 presses, parks it in the corridor above to plug the
pipe, and then stands in the empty doorway for the remaining 20 while the flood
runs out of board. That plug is also provably the best seal available: walling
the corridor cell at press zero makes the flood finish on press 34 and walling
the doorway itself makes it finish on press 35, so the 14-press detour costs
nothing at all. The all-buttons-at-once mechanic is still covered by levels 3,
4 and 6.

Optimal-action sets
-------------------
For the eight BFS levels they are EXACT and come free with the search. BFS by
layers gives every explored state its distance from the start, and a state
``j`` can only be an optimal successor of ``i`` when ``depth[j] == depth[i]+1``
-- a cross edge to a shallower ``j`` would need ``depth[j] + f(j) >= d*`` with
``depth[j] < depth[i]+1``, i.e. ``f(j) > d* - depth[i] - 1``, which is already
too long. So the whole labelling is one backward sweep over the depth layers of
the forward edges: no reverse adjacency, no re-solve, and the answer is the
true tie set rather than an inferred one.

That matters more here than in most of this family. Because the win press is
fixed by the water, the tail of every plan is a run of presses where *nothing
the player can do matters* -- level 8 is 45 steps of "all five presses are
equally right", and it would be a lie to label any one of them as the answer.
Coverage runs from 17% (level 1, whose six presses really are forced) to 100%
(level 8), and every plan's LAST step is a full five-way tie: the win is
decided by the wave rules, which run before the press moves anybody.

``--ties`` is the independent check: it re-solves from every press at every
step with a fresh BFS and asserts the measured tie set is exactly the labelled
one -- both a press the labels missed and a press they claim wrongly fail it.
It is a real test of the layered argument above, not a spot check: 630
alternatives re-solved over the eight BFS levels, 0 disagreements.

The rendering fix
-----------------
One, in ``data/puzzlescript_games/Flood.txt``; no rule and no level was
touched.

  * A HELD-DOWN BUTTON WAS INVISIBLE. Player, Block and the four Waves were
    solid 5x5 tiles, so a crate parked on a button, the player standing on one,
    and a button the flood has swallowed all rendered exactly like the same
    object on bare floor -- and which buttons are held is the whole state of
    levels 3 to 6. Level 6 is the sharp case: it opens with a crate already on
    its only button, which is the ONLY reason its two doors are shut one press
    later, and nothing in the frame said so. (A drowned button is the other
    one: it can never be held again, so the door it feeds is open for good.
    Waiting floods a button on levels 3 and 4 within 21 presses, which is well
    inside what the exploration prefix does.) All six sprites now carry four
    holes at the four pixels the ``cell_px = 4`` downsample keeps (rows and
    columns {0,1,3,4} of the 5x5 art), which are exactly the four pixels the
    Button sprite paints, so the maroon shows through at every cell size the
    levels use. The hole pattern is symmetric under the whole 8-element
    symmetry group, so no rotated frame shows art this game does not own.

Two identities are DELIBERATELY left alone. Wave1/2/3/4 still share one sprite
-- the distinction is which water spread this press, which is exactly what a
pair of consecutive frames shows -- and Closed still shares the Wall sprite,
because a shut door is a wall and the game means it to read as one.

``--audit`` is the regression test: it renders every cell COMPOSITION the game
can show, at every cell size the levels actually use, and asserts they are
pairwise distinct modulo those two intended identities.

Augmentation
------------
Engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variable is the presentation: frame
rotation (k in {0,1,2,3}) with the matching directional action remap. Flood is
deliberately NOT added to `PuzzleScriptAdapter._FLIP_GAMES`: the DOWN spread
rule is not the mirror of the UP one (only pouring down goes inert), so a
vertically flipped board is not a board of this game.

Usage (run from the repo root):
    python solvers/generate_ps_flood_training.py --episodes 200 \
        --out data/training_multi_level/ps_flood

    python solvers/generate_ps_flood_training.py --plans   # level report
    python solvers/generate_ps_flood_training.py --verify  # replay on the engine
    python solvers/generate_ps_flood_training.py --audit   # rendering audit
    python solvers/generate_ps_flood_training.py --fuzz    # model vs engine
    python solvers/generate_ps_flood_training.py --ties 0 3  # optimal-set check
                                                             # (levels optional)
"""

from __future__ import annotations

import itertools
import random
import sys
from array import array
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame              # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan    # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Flood"

#: Disk cache of the per-level start plan AND its optimal-action sets.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "ps_flood_plans.json"

#: The five presses. No rule in this game reads ACTION5, so it is the pure
#: WAIT -- which matters, because waiting out the flood is the back half of
#: every plan. (Walking into a wall waits too: unlike most of this family,
#: Flood has no movement-cancel, so a turn happens whether you moved or not.)
PRESSES = ["up", "down", "left", "right", "action"]
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """The static half of a level (size, walls, buttons) plus the ruleset.

    A state is ``(player, blocks, closed, opens, w1, w2, w3, w4)``: the
    player's cell index (-1 once the water or a door has destroyed it) and
    seven bitmasks over ``r * w + c``. Walls and buttons never change, so they
    live on the board rather than in the state; everything else does, including
    the doors (Open and Closed are different objects on different layers, and
    a door can also be destroyed outright).

    `step` reproduces one whole keypress in the interpreter's own order: the
    force rules, the button rules, the wave physics, and only THEN force
    resolution. That order is not a detail -- it is why the flood spreads from
    where the player was standing rather than from where the press sends them.
    """

    def __init__(self, h, w, walls, buttons):
        self.h, self.w = h, w
        self.n = h * w
        self.walls = walls
        self.buttons = buttons
        self.full = (1 << self.n) - 1
        col0 = 0
        colw = 0
        for r in range(h):
            col0 |= 1 << (r * w)
            colw |= 1 << (r * w + w - 1)
        # Shifting by 1 would wrap a row into its neighbour, so the two edge
        # columns are masked out of every horizontal shift.
        self.not_col0 = self.full & ~col0
        self.not_colw = self.full & ~colw
        self._fill_cache: dict = {}
        self._reach_cache: dict = {}

    def signature(self):
        return (self.h, self.w, self.walls, self.buttons)

    def nb(self, i, d):
        """Neighbour cell index in direction ``d``, or -1 off the board."""
        r, c = divmod(i, self.w)
        r += d[0]
        c += d[1]
        if 0 <= r < self.h and 0 <= c < self.w:
            return r * self.w + c
        return -1

    # -- dynamics ------------------------------------------------------------
    def step(self, state, press):
        """One keypress: ``(state, won)``."""
        p, blocks, closed, opens, w1, w2, w3, w4 = state
        W = self.w
        FULL = self.full
        d = _DELTA.get(press)

        # --- forces -----------------------------------------------------
        # [> Player|Block] then [> Block|Block] hand the press to the whole
        # contiguous run of crates in front of the player. Nothing is checked
        # here; whether any of it can move is decided by resolution, at the
        # bottom of this method, on the board as the rules leave it.
        chain = []
        if d is not None and p >= 0:
            cur = p
            while True:
                nxt = self.nb(cur, d)
                if nxt < 0 or not (blocks >> nxt) & 1:
                    break
                chain.append(nxt)
                cur = nxt

        # --- buttons and doors -------------------------------------------
        btn = self.buttons
        if btn:
            pushers = blocks | ((1 << p) if p >= 0 else 0)
            if (btn & pushers) and opens:
                # Closed lands on the wall/player/wave/crate layer, so it
                # DESTROYS whatever was standing in the doorway.
                v = opens
                if p >= 0 and (v >> p) & 1:
                    p = -1
                blocks &= ~v
                w1 &= ~v
                w2 &= ~v
                w3 &= ~v
                w4 &= ~v
                closed |= v
                opens = 0
            if closed:
                # Re-read the board: the rule above may have removed a pusher.
                pushers = blocks | ((1 << p) if p >= 0 else 0)
                if btn & ~pushers:
                    opens |= closed
                    closed = 0

        # --- wave physics -------------------------------------------------
        w1 |= w3                       # [Wave3] -> [Wave1]
        w3 = 0
        wp = self.walls | blocks | closed | w1 | w2 | w4
        if w4:
            # VERTICAL [WP|Wave4|WP] -> [WP|Wave2|WP]. Order-free: Wave2 is
            # still a Wave, so no conversion changes what is Waterproof.
            conv = w4 & ((wp << W) & FULL) & (wp >> W)
            if conv:
                w4 &= ~conv
                w2 |= conv
        if w2:
            # HORIZONTAL [WP|Wave2|WP] -> [WP|No Wave|WP]. This one DELETES
            # water, so a run of Wave2s un-qualifies itself as it is consumed
            # and the scan order is observable: `horizontal` expands to
            # (left, right) and the left pass scans columns descending, so the
            # rightmost Wave2 of the topmost row goes first.
            while True:
                wp = self.walls | blocks | closed | w1 | w2 | w4
                cand = (w2 & ((wp & self.not_colw) << 1)
                        & ((wp & self.not_col0) >> 1))
                if not cand:
                    break
                w2 &= ~self._scan_pick(cand)
        wp = self.walls | blocks | closed | w1 | w2 | w4

        # The four spread rules, in the .txt's order (left, right, up, down).
        # Each one's fresh Wave3s are Waterproof for the rules after it, which
        # is what decides whether DOWN still finds a free cell -- and DOWN is
        # the one that turns its own source inert.
        t = ((w1 & self.not_col0) >> 1) & ~wp
        if t:
            w3 |= t
            wp |= t
        t = ((w1 & self.not_colw) << 1) & ~wp
        if t:
            w3 |= t
            wp |= t
        t = (w1 >> W) & ~wp
        if t:
            w3 |= t
            wp |= t
        t = ((w1 << W) & FULL) & ~wp
        if t:
            src = t >> W
            w1 &= ~src
            w4 |= src
            w3 |= t
        if p >= 0 and (w3 >> p) & 1:
            p = -1                     # the flood took the player's cell
        if opens:
            opens &= ~(w1 | w2 | w3 | w4)      # [Wave Open] -> [Wave No Open]

        # --- force resolution ----------------------------------------------
        # A chain moves only if the cell past its end is clear of the whole
        # collision layer, which now includes any water that arrived this
        # press: you cannot walk into the flood, and you cannot push a crate
        # into it either. A chain member destroyed by a door shutting splits
        # the run, and the pieces behind the break resolve on their own -- the
        # interpreter keeps a force per object, not per push.
        if d is not None and (p >= 0 or chain):
            off = d[0] * W + d[1]
            seq = [('p', p) if p >= 0 else None]
            for b in chain:
                seq.append(('b', b) if (blocks >> b) & 1 else None)
            runs: list[list] = []
            cur_run: list = []
            for ent in seq:
                if ent is None:
                    if cur_run:
                        runs.append(cur_run)
                        cur_run = []
                else:
                    cur_run.append(ent)
            if cur_run:
                runs.append(cur_run)
            occ = self.walls | blocks | closed | w1 | w2 | w3 | w4
            if p >= 0:
                occ |= 1 << p
            for run in runs:
                ep = self.nb(run[-1][1], d)
                if ep < 0 or (occ >> ep) & 1:
                    continue
                mask = 0
                for _kind, cell in run:
                    mask |= 1 << cell
                shifted = (mask << off) if off > 0 else (mask >> -off)
                occ = (occ & ~mask) | shifted
                bmask = mask
                if run[0][0] == 'p':
                    bmask &= ~(1 << run[0][1])
                    p += off
                blocks = ((blocks & ~bmask)
                          | ((bmask << off) if off > 0 else (bmask >> -off)))

        return (p, blocks, closed, opens, w1, w2, w3, w4), (w3 == 0 and p >= 0)

    def _scan_pick(self, cand):
        """The bit a row-ascending / column-descending scan reaches first."""
        lo = (cand & -cand).bit_length() - 1
        row = lo // self.w
        inrow = cand & (((1 << self.w) - 1) << (row * self.w))
        return 1 << (inrow.bit_length() - 1)

    # -- the flood, relaxed --------------------------------------------------
    def _flood(self, state):
        """``(reachable cells, presses to fill them)`` for the live front.

        A multi-source BFS from the water over the cells it can still enter,
        with the board frozen as it stands. Once the player is walled off the
        distance is exactly one short of the presses left to win (the last cell
        arrives as a Wave3, and the win is the press after it settles into a
        Wave1 with nowhere to go); before that it is an over-estimate, since
        every crate still to be placed shrinks the region. It ranks the beam
        and certifies nothing -- the +1 is a constant, so it does not affect
        the ranking and is deliberately not added.
        """
        _p, blocks, closed, _opens, w1, w2, w3, w4 = state
        key = (blocks, closed, w1, w2, w3, w4)
        hit = self._fill_cache.get(key)
        if hit is not None:
            return hit
        W = self.w
        FULL = self.full
        water = w1 | w2 | w3 | w4
        free = FULL & ~(self.walls | blocks | closed | water)
        # Wave4/Wave2 are inert -- only Wave1 (and the Wave3 that becomes one
        # next press) still spreads -- but they are still WATER, so they count
        # as reached. The front is what the search runs from.
        cur = w1 | w3
        reach = water
        dist = 0
        while cur:
            nxt = ((((cur & self.not_col0) >> 1) | ((cur & self.not_colw) << 1)
                    | (cur >> W) | ((cur << W) & FULL))
                   & free & ~reach)
            if not nxt:
                break
            reach |= nxt
            cur = nxt
            dist += 1
        out = (reach, dist)
        if len(self._fill_cache) < 400_000:
            self._fill_cache[key] = out
        return out

    def fill_time(self, state):
        return self._flood(state)[1]

    def wait_tail(self, state, limit=250):
        """Presses of pure WAIT that win from ``state``, or None.

        The cheap test first: if the flood can still reach the player's cell,
        standing still cannot end well and the simulation is skipped. Standing
        still is also what makes the answer trustworthy -- the player never
        steps off a button, so the doors cannot re-open under it.
        """
        p = state[0]
        if p < 0 or (self._flood(state)[0] >> p) & 1:
            return None
        s = state
        for n in range(1, limit + 1):
            s, won = self.step(s, "action")
            if won:
                return n
            if s[0] < 0:
                return None
        return None

    # -- the exhaustive search -----------------------------------------------
    def bfs(self, state, cap):
        """Layered breadth-first search over the five presses.

        Returns ``(plan, optsets)`` -- a SHORTEST press sequence to a win and,
        for every step, the exact set of presses that are equally shortest --
        or ``(None, None)`` when the level cannot be won or ``cap`` states are
        exhausted.

        Every press costs 1, so BFS is already the optimal search; what it also
        buys is the labelling. A state ``j`` can only be an optimal successor
        of ``i`` when ``depth[j] == depth[i] + 1``: a cross edge to a shallower
        ``j`` would need ``depth[j] + f(j) >= d*`` with ``depth[j] <
        depth[i] + 1``, i.e. ``f(j) > d* - depth[i] - 1``, which is already too
        long to be on a shortest path. So "can still win in exactly the presses
        remaining" propagates backwards through the depth layers of the forward
        edges alone -- one sweep, no reverse adjacency, no re-solve.
        """
        states = [state]
        index = {state: 0}
        depth = [0]
        # succ[5 * i + k]: the successor of press k, -1 for a WIN, -2 for a
        # press that drowned the player or a state never expanded. A flat
        # array because these searches run to ~200k states and a list per
        # state would be most of the memory.
        succ = array('i', [-2] * 5)
        frontier = [0]
        won_depth = None
        while frontier and won_depth is None:
            nxt_frontier = []
            for i in frontier:
                s = states[i]
                base = 5 * i
                for k, press in enumerate(PRESSES):
                    ns, won = self.step(s, press)
                    if won:
                        succ[base + k] = -1
                        won_depth = depth[i] + 1
                        continue
                    if ns[0] < 0:
                        continue                 # the player drowned
                    j = index.get(ns)
                    if j is None:
                        if len(states) >= cap:
                            return None, None
                        j = len(states)
                        index[ns] = j
                        states.append(ns)
                        depth.append(depth[i] + 1)
                        succ.extend((-2, -2, -2, -2, -2))
                        nxt_frontier.append(j)
                    succ[base + k] = j
            frontier = nxt_frontier
        if won_depth is None:
            return None, None

        # `tight[i]`: a win is still reachable from i in exactly
        # ``won_depth - depth[i]`` presses.
        tight = bytearray(len(states))
        layers: list[list[int]] = [[] for _ in range(won_depth + 1)]
        for i, dep in enumerate(depth):
            layers[dep].append(i)
        for i in layers[won_depth - 1]:
            base = 5 * i
            if any(succ[base + k] == -1 for k in range(5)):
                tight[i] = 1
        for dep in range(won_depth - 2, -1, -1):
            for i in layers[dep]:
                base = 5 * i
                for k in range(5):
                    j = succ[base + k]
                    if j >= 0 and depth[j] == dep + 1 and tight[j]:
                        tight[i] = 1
                        break

        plan, optsets = [], []
        i = 0
        for dep in range(won_depth):
            base = 5 * i
            best = []
            for k, press in enumerate(PRESSES):
                j = succ[base + k]
                if dep == won_depth - 1:
                    if j == -1:
                        best.append((press, j))
                elif j >= 0 and depth[j] == dep + 1 and tight[j]:
                    best.append((press, j))
            plan.append(best[0][0])
            optsets.append([pr for pr, _j in best])
            i = best[0][1]
        return plan, optsets

    # -- the fallback --------------------------------------------------------
    def beam(self, state, width, depth_cap):
        """Width-capped breadth-first beam ranked by `fill_time`.

        For the level the exhaustive BFS cannot hold. It aims at a state the
        player can WAIT out rather than at the win itself: `wait_tail` answers
        "is this sealed, and for how many more presses" in one simulation, so
        a candidate found at depth ``d`` is a complete plan of ``d + tail``.
        The beam keeps going past the first candidate until its own depth
        reaches the best total found -- sealing sooner does NOT win sooner
        (the flood finishes when it finishes), and a later, tighter seal is
        routinely shorter overall: on level 5 the first candidate is 42 presses
        and the best is 34.
        """
        frontier = [(state, [])]
        seen = {state}
        best = None
        for _d in range(depth_cap):
            if best is not None and len(frontier[0][1]) >= len(best):
                break
            kids = []
            for s, path in frontier:
                for press in PRESSES:
                    ns, won = self.step(s, press)
                    cand = None
                    if won:
                        cand = path + [press]
                    elif ns[0] < 0 or ns in seen:
                        continue
                    else:
                        seen.add(ns)
                        tail = self.wait_tail(ns)
                        if tail is not None:
                            cand = path + [press] + ["action"] * tail
                        kids.append((self.fill_time(ns), ns, path + [press]))
                    if cand is not None and (best is None or len(cand) < len(best)):
                        best = cand
            if not kids:
                break                        # the reachable space closed
            kids.sort(key=lambda kid: kid[0])
            frontier = [(ns, path) for _h, ns, path in kids[:width]]
        return best


# ---------------------------------------------------------------------------
# Reading a board out of the interpreter
# ---------------------------------------------------------------------------

_MASK_NAMES = ("wall", "button", "block", "closed", "open",
               "wave1", "wave2", "wave3", "wave4")


def read_board(eng, g):
    """``(_Board, state)`` for the interpreter's current grid."""
    inv = {v: k for k, v in g.obj_name_to_idx.items()}
    w = eng.width
    bits = dict.fromkeys(_MASK_NAMES, 0)
    player = -1
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            for o in cell:
                name = inv[o]
                if name == "player":
                    player = i
                elif name in bits:
                    bits[name] |= 1 << i
    board = _Board(eng.height, w, bits["wall"], bits["button"])
    state = (player, bits["block"], bits["closed"], bits["open"],
             bits["wave1"], bits["wave2"], bits["wave3"], bits["wave4"])
    return board, state


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class FloodExpert(PSExpert):
    """Plans read off the native model; `PSExpert` supplies the in-memory memo,
    the on-disk start-plan cache and the snapshot discipline, so only `_search`
    is overridden."""

    directions = PRESSES
    plan_cache_path = PLAN_CACHE

    #: States the exhaustive BFS may hold before it gives up and the beam runs.
    #: The eight levels it solves need at most 172k, so this is ~2x headroom
    #: and still bounds a failed attempt (level 5) to a few seconds and a few
    #: hundred MB -- which matters, because `parallelize_generator` runs
    #: several shards at once.
    bfs_cap = 400_000
    beam_width = 3_000
    beam_depth = 120

    def setup(self) -> None:
        self._board = None
        self._board_sig = None

    def heuristic(self, eng) -> int:                 # pragma: no cover
        raise NotImplementedError(
            "the model owns the heuristic; see _Board.fill_time")

    def _board_for(self, board):
        """Reuse one `_Board` per level so its flood cache survives across
        every search on that board."""
        sig = board.signature()
        if sig != self._board_sig:
            self._board, self._board_sig = board, sig
        return self._board

    def _search(self, eng) -> list | None:
        board, state = read_board(eng, self.g)
        if state[0] < 0:
            return None                          # the player already drowned
        board = self._board_for(board)
        plan, optsets = board.bfs(state, self.bfs_cap)
        if plan is None:
            plan = board.beam(state, self.beam_width, self.beam_depth)
            if plan is None:
                return None
            optsets = [[p] for p in plan]
        return Plan(plan, optsets)


class FloodSolver(PSAStarSolver):
    game_id = "puzzlescript_ps_flood"
    game_name = GAME_NAME
    expert_cls = FloodExpert

    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so a later step limit or sprite fix there cannot silently
    #: diverge from what is taped here.
    game_module_id = "ps:flood"

    #: ``No Wave3 and Some Player`` holds at reset, before anything has
    #: spread, so the win predicate is vacuously true on the level's start
    #: frame. See `record_level`: without this every level would tape a
    #: zero-press "win" and then be dropped for never reaching GameState.WIN.
    vacuous_start_win = True

    #: The longest plan is level 8's 45 presses; the rest is room for the
    #: exploration prefix and its RESET. Stays inside the adapter's 200-step
    #: per-level budget.
    max_steps = 120


# ---------------------------------------------------------------------------
# Reports and checks
# ---------------------------------------------------------------------------

def _levels(solver=None):
    solver = solver or FloodSolver()
    game = solver.make_game(0)
    return game, FloodExpert(game)


def _report() -> int:
    """Per-level board size, crate count, plan length and tie coverage."""
    game, expert = _levels()
    eng = game._engine
    total = solved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        _board, state = read_board(eng, game._game)
        crates = bin(state[1]).count("1")
        if level in FloodSolver.skip_levels:
            print(f"level {level:2d}: SKIPPED")
            continue
        found = expert.plan(eng, level)
        if found is None:
            print(f"level {level:2d}: NO PLAN")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        solved += 1
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
              f"{crates:2d} crates  {len(found):3d} presses ({room}), "
              f"{ties:3d} steps with a tie set "
              f"({ties / max(1, len(found)):.0%})")
    print(f"total {total} presses over {solved} solved levels "
          f"({game.n_levels} shipped)")
    return 0


def _verify() -> int:
    """Replay every level's plan on the REAL interpreter and require a WIN.

    The plans come off a model; this is the line that says the model and the
    interpreter agree about the boards that actually ship."""
    game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        if level in FloodSolver.skip_levels:
            continue
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level:2d}: NO PLAN")
            bad += 1
            continue
        game.set_level(level)
        for d in plan:
            eng.step(d)
        ok = eng.check_win()
        bad += not ok
        print(f"level {level:2d}: {len(plan):3d} presses -> "
              f"{'WIN' if ok else 'NOT WON'}")
    print("verify clean" if not bad else f"VERIFY FAILED on {bad} levels")
    return 0 if not bad else 1


def _ties(levels=None) -> int:
    """Re-derive every step's tie set with a fresh BFS and compare.

    An independent check of the layered argument `_Board.bfs` labels with: at
    each step of the plan, press everything, BFS from each successor for its
    true distance to a win, and assert that the presses reaching one in the
    moves remaining are exactly the labelled set. Both an unlabelled press
    that ties (the label claims too little) and a labelled press that does not
    (it claims too much) fail.

    Only the BFS levels are checked; level 5's beam labels are singletons by
    construction and make no claim to be complete."""
    game, expert = _levels()
    eng = game._engine
    bad = tested = 0
    for level in range(game.n_levels):
        if levels and level not in levels:
            continue
        game.set_level(level)
        board, state = read_board(eng, game._game)
        board = expert._board_for(board)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        if all(len(s) == 1 for s in sets) and level in _BEAM_LEVELS:
            print(f"level {level:2d}: beam plan, no tie claim")
            continue
        for i, press in enumerate(plan):
            remaining = len(plan) - i
            measured = []
            for alt in PRESSES:
                ns, won = board.step(state, alt)
                tested += 1
                if won:
                    if remaining == 1:
                        measured.append(alt)
                    continue
                if ns[0] < 0:
                    continue
                sub, _o = board.bfs(ns, expert.bfs_cap)
                if sub is not None and 1 + len(sub) == remaining:
                    measured.append(alt)
            if sorted(measured) != sorted(sets[i]):
                bad += 1
                print(f"level {level:2d} step {i:3d}: labelled {sets[i]} "
                      f"but measured {measured}")
            state, _won = board.step(state, press)
        print(f"level {level:2d}: {len(plan):3d} steps checked")
    print(f"ties: {tested} alternatives re-solved, {bad} disagreements")
    return 0 if not bad else 1


#: Levels whose plan comes off the beam rather than the BFS (see the module
#: docstring). Only used to keep `--ties` from asking a beam plan a question it
#: cannot answer.
_BEAM_LEVELS = frozenset({5})


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Composition, not object: a crate ON a button is what makes a door shut and
    has to differ from a crate on the floor, and the player standing on one has
    to differ from the player standing on grass.

    Two identities are EXPECTED and asserted as identities rather than
    flagged: the four Wave objects share one sprite (which water spread this
    press is a two-frame question, not a one-frame one), and Closed shares the
    Wall sprite (a shut door is a wall).
    """
    game = FloodSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {
        "floor": (), "wall": ("wall",), "button": ("button",),
        "open": ("open",),
        "player": ("player",), "player_on_button": ("button", "player"),
        "player_on_open": ("open", "player"),
        "block": ("block",), "block_on_button": ("button", "block"),
        "block_on_open": ("open", "block"),
        "wave": ("wave1",), "wave_on_button": ("button", "wave1"),
    }
    same = {"closed": ("wall", ("closed",))}
    for k in ("wave2", "wave3", "wave4"):
        same[k] = ("wave", (k,))

    sizes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, objs):
        # Fill the WHOLE board with the composition and compare whole frames:
        # `_render_frame` upscales any sub-64 render to fill the frame, so
        # cropping one cell out of a 13x13 board lands in the wrong place and
        # every composition reads as identical. Rendering is per-cell
        # independent, so a whole-frame comparison is the same test done where
        # the geometry cannot drift.
        eng.height, eng.width = h, w
        eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                     for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g))

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {name: shoot(h, w, objs) for name, objs in comps.items()}
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        broken = [name for name, (twin, objs) in same.items()
                  if not np.array_equal(shoot(h, w, objs), shots[twin])]
        bad += len(clashes) + len(broken)
        note = "OK"
        if clashes:
            note = "IDENTICAL " + str(clashes)
        if broken:
            note += "  EXPECTED-IDENTITY BROKEN " + str(broken)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {note}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} problems")
    return 0 if not bad else 1


def _fuzz(n_boards: int = 2000, n_steps: int = 30, seed: int = 0) -> int:
    """Differential fuzz: random boards played press-for-press against the real
    interpreter, comparing every object mask and the win flag after every
    press.

    This is the only thing standing between `_Board` and a silently wrong
    corpus, so it is built to hit what the shipped levels barely do: water in
    all four wave states (including the Wave4/Wave2 evaporation the levels show
    once), doors both open and shut with several buttons in play, doors
    shutting on a crate or on the player, and crates jammed into water.
    """
    game = FloodSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    rng = random.Random(seed)
    mismatches = steps = 0
    for _ in range(n_boards):
        h, w = rng.choice([4, 5, 6, 7]), rng.choice([4, 5, 6, 7])
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)
        n = len(cells)
        take = 0

        def grab(k):
            nonlocal take
            out = set(cells[take:take + k])
            take += k
            return out

        walls = grab(rng.randint(0, n // 4))
        blocks = grab(rng.randint(0, n // 4))
        waves = {cell: rng.choice(["wave1", "wave2", "wave3", "wave4"])
                 for cell in grab(rng.randint(1, max(1, n // 3)))}
        closed = grab(rng.randint(0, 2))
        # Button and Open share a collision layer, so a cell is one or other.
        buttons = grab(rng.randint(0, 3))
        opens = grab(rng.randint(0, 3))
        rest = cells[take:]
        player = rest[0] if rest and rng.random() < 0.95 else None

        grid = []
        for r in range(h):
            row = []
            for c in range(w):
                cell = {idx["background"]}
                if (r, c) in buttons:
                    cell.add(idx["button"])
                elif (r, c) in opens:
                    cell.add(idx["open"])
                if (r, c) in walls:
                    cell.add(idx["wall"])
                elif (r, c) in blocks:
                    cell.add(idx["block"])
                elif (r, c) in closed:
                    cell.add(idx["closed"])
                elif (r, c) in waves:
                    cell.add(idx[waves[(r, c)]])
                elif (r, c) == player:
                    cell.add(idx["player"])
                row.append(cell)
            grid.append(row)
        eng.grid, eng.height, eng.width = grid, h, w
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        eng._rule_win = False

        board, state = read_board(eng, g)
        for _s in range(n_steps):
            press = rng.choice(PRESSES)
            eng.step(press)
            truth_won = eng.check_win()
            _b, truth = read_board(eng, g)
            state, won = board.step(state, press)
            steps += 1
            if state != truth or won != truth_won:
                mismatches += 1
                print(f"MISMATCH press={press} board {h}x{w}\n"
                      f"  walls {sorted(walls)} blocks {sorted(blocks)}\n"
                      f"  waves {waves} closed {sorted(closed)}\n"
                      f"  buttons {sorted(buttons)} opens {sorted(opens)} "
                      f"player {player}\n"
                      f"  model  {state} won={won}\n"
                      f"  engine {truth} won={truth_won}")
                state = truth
                if mismatches > 4:
                    return 1
    print(f"fuzz: {steps} transitions, {mismatches} mismatches")
    return 0 if not mismatches else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--ties" in sys.argv:
        sys.exit(_ties([int(x) for x in sys.argv[sys.argv.index("--ties") + 1:]
                        if x.isdigit()]))
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    sys.exit(FloodSolver.main())
