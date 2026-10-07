"""Generate Phase-1 training data for the PuzzleScript game ps:explod
("Explod", CHz -- the game where the only way through a wall is to blow it up,
and the only way to buy time is to spend a turn).

The harness -- the rotation contract, the trajectory recorder, the plan cache and
the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a native model of the
fuse/explosion ruleset, the searches over it, and the rendering fixes.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_explod",
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
exactly. Every step also carries a set of equally-optimal presses.

The game
--------
Walk to the Goal. Walls are in the way, and the ONLY thing that removes a wall is
an explosion. X lights every UNLIT bomb orthogonally adjacent to you (never one
standing in water); you also push a bomb by walking into it, one bomb at a time,
and a bomb with anything behind it does not budge -- and neither do you.

FOUR RULES THAT SHAPE EVERY PLAN, none of them obvious from the art:

  * TIME ONLY PASSES WHEN YOU MOVE. ``late [ Player MovementCheck ] -> CANCEL``
    undoes any turn the player did not physically move on, so walking into a
    wall is not a wasted turn -- it is NO turn, and fuses do not tick. The one
    exception is X, which clears the flag itself: X is the game's *wait*, and it
    costs exactly one tick. (An `again` continuation is also exempt, which is
    what stops an explosion frame from cancelling itself.)
  * A FUSE IS FOUR TURNS, NOT THREE. Lighting adds one to the fuse and the tick
    at the end of that same turn takes it back off, so lighting an UnlitBomb3
    leaves a 3 and it detonates on the third press after the one that lit it.
  * AN EXPLOSION IS A PLUS, AND IT KILLS. It clears the four orthogonal
    neighbours of walls, turns a live player standing on one into a corpse, and
    lights any UNLIT bomb it touches -- at that bomb's own fuse plus one. So a
    chain does NOT run instantly: it crawls one bomb per two-to-four presses,
    which is what gives you time to walk clear, and what makes a bomb field a
    fuse of its own. A dead player can never move again and can never press X,
    so every later turn cancels: death is a silent, permanent softlock, never a
    GAME_OVER.
  * WATER IS AN ERASER. A bomb standing in water cannot be lit by X, and a lit
    one is doused back to unlit at the top of the late pass. A chain can still
    set a bomb in water to LitBomb4/3, but the douse rule catches it on the way
    down and it never reaches 1. Bombs can be pushed across water freely, and so
    can the player -- water blocks nothing, it just un-arms.

The expert
----------
A NATIVE model of the ruleset (`_Board`) and two searches over it. The model
exists because the interpreter runs this game at ~1.2k steps/s (five rules fire
on every turn and the explosion pass runs the whole rule list twice), where the
model runs at ~400k -- and the searches here are hundreds of thousands of steps
deep. ``--fuzz`` plays random boards move-for-move against the real interpreter
and compares the player, the walls, every bomb's fuse and the win flag after
every press; 120k transitions with zero mismatches is what it was accepted on.
Re-run it after ANY change to the .txt rules or to the adapter's rule code.

`_search` runs, in order:

  * weighted A* over primitive presses, on a ladder of weights, taking the first
    rung that returns -- so the levels that can be solved cheaply keep the
    shortest plan the ladder can prove;
  * a width-capped breadth-first BEAM over the same presses for the levels A*
    cannot reach. A beam plan is engine-verified and winning but not shortest.

THE HEURISTIC AND ITS PRUNE. Both searches rank states by a Dijkstra from the
goal in which stepping onto a wall costs ``1 + WALL_COST`` -- but only for walls
the level can still open. `_Board.blastable` answers that with a closure: the
player owns its whole wall-free component (bombs count as floor, since they can
be pushed), a component holding at least one bomb can deliver a bomb to any
non-water cell in it, so every wall touching such a cell falls; repeat until
nothing new opens. The relaxation only ever ADDS reachability, so a wall it
calls unopenable really is unopenable and a state whose goal sits behind one is
DEAD -- dropped un-expanded. That prune is what makes the searches finish: the
common way to lose this game is to spend the last bomb in the wrong place, and
without it the search happily explores the whole space beyond that mistake.

The levels
----------
All ten are CHz's own boards in shipped order (the four ``message`` screens
between them are not levels and the adapter does not count them).

    level  size    bombs  plan  found by
    0      7x7      3       25  A*
    1      4x6      4       15  A*
    2      5x10    32       17  A*      a solid block: one X sets off the lot
    3      10x5     8       60  A*
    4      5x5      6       24  A*
    5      5x13    15        -  SKIPPED
    6      4x7      8       29  A*      the water tutorial
    7      5x11    12       88  beam
    8      5x5      5       25  A*
    9      9x9      9       67  beam    nine bombs inside a ring of water

350 presses over the nine, every one of them replayed through the real
interpreter to a WIN by ``--verify``, and all inside the adapter's 200-step
per-level budget with room for the exploration prefix.

LEVEL 5 IS THE ONE THAT GOT AWAY. It is a 5x13 with seven columns of wall
between the player and the goal and fifteen bombs laid out as a CHECKERBOARD, so
no two of them touch and nothing ever chains: every wall costs its own bomb,
delivered and lit by hand. Worse, every cell of that field is orthogonally
adjacent to three or four bombs, so the X that lights the one you want lights
the others too and the fifteen bombs buy far fewer than fifteen blasts unless
they are pushed apart first. A* never gets near it, and a beam three times the
width and twice the depth of the one that solves levels 7 and 9 does not find
the sequence either. It is skipped rather than shipped half-solved; the other
nine cover every mechanic it uses.

The searches are seed-independent, so they are written to
``data/explod_plans.json`` (start plan + optimal sets per level) and no shard of
`parallelize_generator` re-derives them. A cold run is ~5 minutes and under 1 GB,
nearly all of it the two beams (the seven A* levels together take 3 seconds) --
**run ``--plans`` once before a parallel run**; delete that file to re-derive.
The re-derivation is exact: dropping the cache and re-solving returns the seven
A* levels byte-identical, plans and optimal sets both.

Optimal-action sets
-------------------
A plan step is labelled with the press taken, plus -- for the WALK stretches --
every other direction that keeps the player on a shortest route to the cell the
walk ends on. A walk is identified from the board, not from a macro boundary: a
step is a walk when the player moved and the board did nothing but AGE (same
walls, same bombs on the same cells, every fuse doused and ticked exactly as it
would have been on any other press). That is order-free because no rule in this
game fires on a bare move onto an empty cell, so every shortest re-route reaches
the same cell after the same number of presses with the same fuses burnt. It
deliberately includes stretches walked while a fuse burns: a bomb goes off on
the press it goes off on whichever way the player walked, and a step an
explosion landed on is not a walk step (it removes a bomb, and usually a wall),
so no re-route can be sent into one.

Every other step -- a push, a light, a wait, the press an explosion lands on --
is labelled with itself alone: which bomb to shove where, and how long to stand
still, IS the puzzle, and one press earlier or later is a different plan, not a
reordering of this one.

Coverage is low on purpose -- 1 to 7 steps a level -- because these boards are
corridors and pockets where the shortest route really is unique, and because a
press one turn early or late is a different plan.

``--ties`` is the check: it re-solves from every press at every step and reports
any press that reaches a win at least as fast as the labelled one. On the five
levels checked it found 17 such presses over 420 alternatives. They are REPORTED
rather than folded into the labels: the labels above are provably sound (a
shortest re-route reaches an identical state after an identical number of
presses), whereas a re-solve is only as sound as the heuristic it ran under, and
this one trades admissibility for guidance. Sound-but-incomplete is the right
side to be wrong on for a policy target.

The rendering fixes
-------------------
Three, all in ``data/puzzlescript_games/Explod.txt``; no rule and no level was
touched.

  * THE FUSE WAS INVISIBLE AT THE SIZE THE BIGGEST LEVELS RENDER AT. Levels run
    to 5x13, i.e. cell_px 4, where `_render_cell_sprite` samples sprite rows and
    columns {0, 1, 3, 4} only. The 1-fuse art was a single pixel at (1, 2) and
    the 2-fuse art one at (0, 3): at cell_px 4 a bomb about to detonate, a bomb
    two turns away and a bomb three turns away all rendered identically, i.e.
    the state the whole game turns on was not in the frame. The fuse is now a
    PIXEL COUNT on row 0 at columns {4}, {3,4}, {1,3,4}, {0,1,3,4} -- every one
    of which survives the downsample, and a count (unlike a position) is also
    invariant under the rotation/flip augmentation. Brown = unlit, yellow = lit.
  * LitBomb4 shipped as a solid black square with no sprite, on the assumption
    that it never survives a turn. It does: a bomb lit on the same turn another
    one detonates keeps its ExplodingCheck, which holds the tick off, so
    LitBomb4 is on screen for a whole keypress. It now has the same art as the
    rest, with a four-pixel fuse.
  * THE WINNING FRAME WAS THE PLAYER'S OWN ART. Goal was a hollow cup and the
    player a thin stick figure, so at cell_px 4 the player standing on the goal
    differed from the player standing on grass by two pixels. Goal is a solid
    tile now, and the player a pink ring with holes exactly at the pixels the
    cell_px 4 grid keeps, so the tile it stands on shows through in eight of
    them. Both the player and the corpse are symmetric under the whole
    8-element symmetry group, so no mirrored frame shows art this game does not
    own.

``--audit`` is the regression test: it renders every cell COMPOSITION the game
can show -- including bomb-on-goal, bomb-on-water and the player in water -- at
every cell size the levels actually use, and asserts they are pairwise distinct.

Augmentation
------------
Engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: frame rotation (k in {0,1,2,3}) plus an independent horizontal
and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to). The flips are
exact symmetries: there is no gravity, input is screen-relative, and every rule
is stated over all four directions at once, so a mirrored board is a legal board
of the same game.

Usage (run from the repo root):
    python solvers/generate_explod_training.py --episodes 200 \
        --out data/training_multi_level/explod

    python solvers/generate_explod_training.py --plans   # level report
    python solvers/generate_explod_training.py --verify  # replay on the engine
    python solvers/generate_explod_training.py --audit   # rendering audit
    python solvers/generate_explod_training.py --fuzz    # model vs engine
    python solvers/generate_explod_training.py --ties 0 4  # optimal-set check
                                                          # (levels optional)
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

from adapters.puzzlescript_adapter import _render_frame               # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan     # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Explod"

#: Disk cache of the per-level start plan AND its optimal-action sets.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "explod_plans.json"

#: The five presses. ACTION5 ("X") lights bombs and is also the only one-press
#: way to let a turn pass.
PRESSES = ["up", "down", "left", "right", "action"]
DIRS = PRESSES[:4]
_DELTA = [(-1, 0), (1, 0), (0, -1), (0, 1)]

#: Bomb fuse states, in the interpreter's own object order.
U3, U2, U1, L4, L3, L2, L1 = range(7)
#: Lighting ADDS ONE to the fuse; the tick at the end of the same turn takes it
#: straight back off, so an UnlitBomb3 lit by X is a 3 when the turn ends.
_LIGHT = {U3: L4, U2: L3, U1: L2}
_DOUSE = {L3: U3, L2: U2, L1: U1}
_TICK = {L4: L3, L3: L2, L2: L1}

NAME_TO_CODE = {"unlitbomb3": U3, "unlitbomb2": U2, "unlitbomb1": U1,
                "litbomb4": L4, "litbomb3": L3, "litbomb2": L2, "litbomb1": L1}
CODE_TO_NAME = {v: k for k, v in NAME_TO_CODE.items()}

#: Extra presses charged for stepping onto a wall in the heuristic's Dijkstra.
#: A lower bound on opening one is 2 (light, then one press for the fuse to run
#: out) and the true cost is fetching a bomb as well, so this trades a little
#: admissibility for a heuristic that actually pulls the search toward the
#: cheapest barrier. Plans are "shortest found", never certified shortest.
WALL_COST = 3

#: Charged instead of a distance when the goal is provably out of reach.
DEAD = 10 ** 6


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """The static half of a level (size, goals, water) plus the ruleset.

    A state is ``(player, alive, walls, bombs, echk)``: the player's cell, the
    live/corpse flag, a frozenset of wall cells (walls are DESTROYED by play, so
    they are state, not scenery), the bombs as a sorted tuple of
    ``(cell, fuse code)``, and the ExplodingCheck flag.

    ``echk`` is only ever True in a state the game is already WON in -- the
    engine breaks its `again` loop the moment the win condition holds, which
    leaves that turn's ExplodingCheck token on the grid, and the token suppresses
    the movement-cancel on the following turn. Nothing in a search ever sees it;
    it is in the state so that ``--fuzz`` can hold the model to the interpreter
    past a win instead of resynchronising there.

    `step` reproduces one whole KEYPRESS: the first turn, plus the `again`
    continuation the explosion rule drives. One turn is the main rule list, then
    force resolution, then the late rule list -- in that order, because the
    movement-cancel test reads a token the main list places and the resolution
    moves the player out from under.
    """

    def __init__(self, h, w, goals, water):
        self.h, self.w = h, w
        self.goals = frozenset(goals)
        self.water = frozenset(water)
        self._blast_cache: dict = {}
        self._field_cache: dict = {}
        self._comp_cache: dict = {}
        # Interning tables for the two big state components. Walls change only
        # when something explodes and bombs only when one is pushed, lit or
        # spent, so a search holding millions of states holds a handful of
        # distinct wall sets and orders of magnitude fewer bomb layouts than
        # states -- but `step` rebuilds both every press, so without this every
        # state keeps its own copy and the beam on the 9x9 board runs the
        # machine out of memory.
        self._iwalls: dict = {}
        self._ibombs: dict = {}

    def inb(self, p):
        return 0 <= p[0] < self.h and 0 <= p[1] < self.w

    def signature(self):
        return (self.h, self.w, self.goals, self.water)

    # -- dynamics ------------------------------------------------------------
    def step(self, state, press):
        """One keypress: ``(state, won)``."""
        player, alive, walls, bombs, echk = state
        start = state
        walls = set(walls)
        bombs = dict(bombs)
        cur = start
        for it in range(50):
            inp = press if it == 0 else None
            before = (player, alive, frozenset(walls), tuple(sorted(bombs.items())))
            echk_in = echk

            # ---- main rules ---------------------------------------------
            # [ Player ] -> [ Player MovementCheck ] arms the cancel; X and an
            # ExplodingCheck anywhere on the board each disarm it.
            mcheck = not echk_in
            moved = False
            if inp == "action":
                if alive:
                    mcheck = False
                    for dr, dc in _DELTA:
                        nb = (player[0] + dr, player[1] + dc)
                        code = bombs.get(nb)
                        if code in _LIGHT and nb not in self.water:
                            bombs[nb] = _LIGHT[code]
            elif inp is not None and alive:
                # [ MOVING DeadPlayer ] -> [ STATIONARY DeadPlayer ]: a corpse
                # never gets a force, which is why death is a permanent freeze.
                dr, dc = _DELTA[PRESSES.index(inp)]
                nb = (player[0] + dr, player[1] + dc)
                if self.inb(nb) and nb not in walls:
                    if nb in bombs:
                        # [ > LivePlayer | Bomb ] -> [ > LivePlayer | > Bomb ].
                        # One bomb only: there is no chain rule, so a second bomb
                        # behind the first jams the whole push, player included.
                        beyond = (nb[0] + dr, nb[1] + dc)
                        if (self.inb(beyond) and beyond not in walls
                                and beyond not in bombs):
                            bombs[beyond] = bombs.pop(nb)
                            player, moved = nb, True
                    else:
                        player, moved = nb, True

            # ---- late rules ---------------------------------------------
            for cell in list(bombs):                       # douse
                if cell in self.water and bombs[cell] in _DOUSE:
                    bombs[cell] = _DOUSE[bombs[cell]]
            echk = False
            lit1 = [c for c, code in bombs.items() if code == L1]
            if lit1:
                # ``late [LitBomb1] [Bomb] -> [... ExplodingCheck]`` marks EVERY
                # bomb, and the tick rules all test for its absence, so an
                # exploding turn does not tick at all -- the tick happens on the
                # `again` pass instead. One keypress, one tick, always.
                echk = True
                for b in lit1:
                    for dr, dc in _DELTA:
                        nb = (b[0] + dr, b[1] + dc)
                        if alive and nb == player:
                            alive = False
                        walls.discard(nb)
                        code = bombs.get(nb)
                        if code in _LIGHT:
                            bombs[nb] = _LIGHT[code]       # chain, at fuse + 1
                for b in lit1:
                    del bombs[b]
            else:
                for cell in list(bombs):
                    if bombs[cell] in _TICK:
                        bombs[cell] = _TICK[bombs[cell]]

            if mcheck and not moved:
                # CANCEL reverts the whole turn. The win is a STATE predicate,
                # so a player already standing on the goal still reads as a win.
                return start, (start[1] and start[0] in self.goals)

            fw = frozenset(walls)
            bt = tuple(sorted(bombs.items()))
            cur = (player, alive, self._iwalls.setdefault(fw, fw),
                   self._ibombs.setdefault(bt, bt), echk)
            won = alive and player in self.goals
            if not lit1 or cur[:4] == before:
                break                                      # no `again` requested
            if won:
                break                                      # engine breaks here too
        return cur, won

    # -- the heuristic and its reachability closure ---------------------------
    def _region(self, walls, start):
        seen = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _DELTA:
                nb = (r + dr, c + dc)
                if self.inb(nb) and nb not in walls and nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        return seen

    def _components(self, walls):
        """``{cell: component id}`` over the wall-free cells. Cached per wall
        set, of which one board has very few (walls only ever come off)."""
        hit = self._comp_cache.get(walls)
        if hit is not None:
            return hit
        lab: dict = {}
        n = 0
        for r in range(self.h):
            for c in range(self.w):
                if (r, c) in walls or (r, c) in lab:
                    continue
                for cell in self._region(walls, (r, c)):
                    lab[cell] = n
                n += 1
        self._comp_cache[walls] = lab
        return lab

    def blastable(self, player, walls, bomb_cells):
        """The walls this state can still open, over-approximated.

        The relaxation: the player owns its whole wall-free component (bombs
        count as floor -- they can be pushed out of the way or blown up), a
        component holding at least one bomb can deliver a bomb to any non-water
        cell of it, and every wall touching such a cell falls. Opening walls
        merges components, so repeat to a fixpoint.

        Every step of that only ADDS reachability, so a wall left out really
        cannot be opened and a goal behind one is genuinely lost. Used both as
        the heuristic's passability test and as the search's death test.

        Cached on the player's COMPONENT id, not its cell: the answer depends on
        where the player stands only through which wall-free component that is,
        and every state on one board shares a handful of components while the
        player visits every cell of them. That one substitution is the
        difference between a cache that hits and a full closure per node."""
        key = (self._components(walls).get(player), walls, bomb_cells)
        hit = self._blast_cache.get(key)
        if hit is not None:
            return hit
        left = set(walls)
        while True:
            reg = self._region(left, player)
            if not (bomb_cells & reg):
                break                       # no bomb in reach: nothing else opens
            blow = {nb for cell in reg if cell not in self.water
                    for nb in ((cell[0] + dr, cell[1] + dc) for dr, dc in _DELTA)
                    if nb in left}
            if not blow:
                break
            left -= blow
        out = frozenset(left)
        if len(self._blast_cache) < 200_000:
            self._blast_cache[key] = out
        return out

    def field(self, player, walls, bomb_cells):
        """``{cell: presses to a goal}``: Dijkstra from the goals, charging
        ``1 + WALL_COST`` to step onto an openable wall and refusing to step
        onto one that can never open."""
        shut = self.blastable(player, walls, bomb_cells)
        key = (walls, shut)
        hit = self._field_cache.get(key)
        if hit is not None:
            return hit
        dist = {}
        pq = []
        for gc in self.goals:
            if gc in shut:
                continue                    # a goal sealed behind its own wall
            dist[gc] = 0
            heapq.heappush(pq, (0, gc))
        while pq:
            d, cell = heapq.heappop(pq)
            if d > dist.get(cell, 1 << 30):
                continue
            for dr, dc in _DELTA:
                nb = (cell[0] + dr, cell[1] + dc)
                if not self.inb(nb) or nb in shut:
                    continue
                nd = d + 1 + (WALL_COST if nb in walls else 0)
                if nd < dist.get(nb, 1 << 30):
                    dist[nb] = nd
                    heapq.heappush(pq, (nd, nb))
        if len(self._field_cache) < 100_000:
            self._field_cache[key] = dist
        return dist

    def estimate(self, state):
        # NB not `h`: `self.h` is the board HEIGHT.
        player, alive, walls, bombs, _echk = state
        if not alive:
            return DEAD
        return self.field(player, walls,
                          frozenset(c for c, _k in bombs)).get(player, DEAD)

    # -- searches -------------------------------------------------------------
    def astar(self, state, cap, weight):
        """Weighted A* over primitive presses. Returns a press list or None.

        States are kept in a flat list with parent pointers rather than a path
        per heap entry: these searches run to millions of nodes and a list per
        node is most of the memory."""
        if state[1] and state[0] in self.goals:
            return []
        states = [state]
        parent = [(-1, None)]
        best = {state: 0}
        pq = [(weight * self.estimate(state), 0, 0)]
        nodes = 0
        while pq:
            _f, g, i = heapq.heappop(pq)
            s = states[i]
            if g > best.get(s, 1 << 30):
                continue
            for p in PRESSES:
                ns, won = self.step(s, p)
                nodes += 1
                if won:
                    out = [p]
                    j = i
                    while parent[j][0] >= 0:
                        out.append(parent[j][1])
                        j = parent[j][0]
                    return out[::-1]
                if ns == s or not ns[1]:
                    continue               # cancelled turn, or a corpse
                ng = g + 1
                if best.get(ns, 1 << 30) <= ng:
                    continue
                hh = self.estimate(ns)
                if hh >= DEAD:
                    continue               # the goal can never be opened again
                best[ns] = ng
                states.append(ns)
                parent.append((i, p))
                heapq.heappush(pq, (ng + weight * hh, ng, len(states) - 1))
            if nodes >= cap:
                return None
        return None

    def beam(self, state, width, depth):
        """Breadth-first beam over the same presses, ``width`` states per depth
        ranked by the heuristic alone.

        For the levels where A* runs out: their plans are 60+ presses and the
        heuristic is FLAT across the middle of one (a dozen presses that fetch a
        bomb and light it without the player getting a step nearer the goal), so
        A* has nothing to steer with and degenerates to uniform-cost search at a
        depth where that is hopeless. A beam spends the same budget on breadth
        at every depth and only uses the heuristic to break ties, which is all a
        flat heuristic is good for. The trade is length: a beam plan wanders."""
        if state[1] and state[0] in self.goals:
            return []

        # Paths as parent pointers, not lists: a width-4000 beam 150 deep would
        # otherwise hold ~3M press lists of growing length.
        trail: list = [(-1, None)]

        def unwind(i, last):
            out = [last]
            while trail[i][0] >= 0:
                out.append(trail[i][1])
                i = trail[i][0]
            return out[::-1]

        frontier = [(state, 0)]
        seen = {state}
        for _d in range(depth):
            kids = []
            for s, i in frontier:
                for p in PRESSES:
                    ns, won = self.step(s, p)
                    if won:
                        return unwind(i, p)
                    if ns == s or not ns[1] or ns in seen:
                        continue
                    seen.add(ns)
                    hh = self.estimate(ns)
                    if hh >= DEAD:
                        continue
                    kids.append((hh, ns, i, p))
            if not kids:
                return None                # the reachable space closed
            kids.sort(key=lambda kid: kid[0])
            frontier = []
            for _h, ns, i, p in kids[:width]:
                trail.append((i, p))
                frontier.append((ns, len(trail) - 1))
        return None

    # -- optimal-action sets --------------------------------------------------
    def annotate(self, state, plan):
        """Per-step optimal-press SETS for ``plan``, starting at ``state``.

        A step is a WALK step when the player moved and the board did nothing
        but AGE: same walls, same bombs on the same cells with every fuse doused
        and ticked exactly as it would have been on any other press. That rules
        out a push, a light, and an explosion (which removes a bomb or a wall),
        which is what makes the run's board constant and its order free -- no
        rule in this game fires on a bare move onto an empty cell, so every
        interleaving that stays on a shortest route to the cell the run ends on
        reaches an identical state after the same number of presses, ticking the
        same fuses on the way. Note this deliberately allows a fuse to be
        BURNING through the run: the bomb goes off on the press it goes off on
        regardless of which way the player walked, and since no explosion
        happens inside the run, no re-route can walk into one.

        Everything else -- a push, a light, a wait, and the press an explosion
        lands on -- is labelled with the press taken alone."""
        steps = []
        s = state
        for p in plan:
            ns, _won = self.step(s, p)
            walk = (ns[2] == s[2] and ns[0] != s[0] and ns[3] == self._age(s[3]))
            steps.append((walk, s, p, ns))
            s = ns

        out = [None] * len(plan)
        i = 0
        while i < len(steps):
            if not steps[i][0]:
                out[i] = [steps[i][2]]
                i += 1
                continue
            run = i
            while i < len(steps) and steps[i][0]:
                i += 1
            # A walk run ends on the cell the next step is taken from -- or, for
            # a trailing run, on the goal itself: the win IS a walk onto it.
            stand = steps[i][1][0] if i < len(steps) else steps[i - 1][3][0]
            dist = self._walk_field(steps[run][1], stand)
            for j in range(run, i):
                here = steps[j][1][0]
                d0 = dist.get(here)
                alts = [] if d0 is None else [
                    DIRS[k] for k, (dr, dc) in enumerate(_DELTA)
                    if dist.get((here[0] + dr, here[1] + dc)) == d0 - 1]
                # The recorded step is on a shortest route by construction, so an
                # empty (or disagreeing) `alts` means the reconstruction drifted
                # -- fall back to labelling what the expert did.
                out[j] = alts if steps[j][2] in alts else [steps[j][2]]
        return out

    def _age(self, bombs):
        """The bombs one press later when the player only walked: douse first
        (the late list douses before it ticks), then tick. Nothing explodes,
        because a plan step that set one off is not a walk step."""
        out = []
        for cell, code in bombs:
            if cell in self.water and code in _DOUSE:
                code = _DOUSE[code]
            out.append((cell, _TICK.get(code, code)))
        return tuple(out)

    def _walk_field(self, state, target):
        """BFS distance to ``target`` over the cells the player may walk on: no
        wall, no bomb (walking into a bomb is a push, not a walk)."""
        walls, bombs = state[2], {c for c, _k in state[3]}
        dist = {target: 0}
        queue = deque([target])
        while queue:
            cur = queue.popleft()
            for dr, dc in _DELTA:
                nb = (cur[0] + dr, cur[1] + dc)
                if (self.inb(nb) and nb not in walls and nb not in bombs
                        and nb not in dist):
                    dist[nb] = dist[cur] + 1
                    queue.append(nb)
        return dist


# ---------------------------------------------------------------------------
# Reading a board out of the interpreter
# ---------------------------------------------------------------------------

def read_board(eng, g):
    """``(_Board, state)`` for the interpreter's current grid."""
    inv = {v: k for k, v in g.obj_name_to_idx.items()}
    walls, goals, water, bombs = set(), set(), set(), {}
    player, alive, echk = None, True, False
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            for o in cell:
                name = inv[o]
                if name == "wall":
                    walls.add((r, c))
                elif name == "goal":
                    goals.add((r, c))
                elif name == "water":
                    water.add((r, c))
                elif name == "liveplayer":
                    player, alive = (r, c), True
                elif name == "deadplayer":
                    player, alive = (r, c), False
                elif name == "explodingcheck":
                    echk = True
                elif name in NAME_TO_CODE:
                    bombs[(r, c)] = NAME_TO_CODE[name]
    board = _Board(eng.height, eng.width, goals, water)
    return board, (player, alive, frozenset(walls),
                   tuple(sorted(bombs.items())), echk)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class ExplodExpert(PSExpert):
    """Plans read off the native model; `PSExpert` supplies the in-memory memo,
    the on-disk start-plan cache and the snapshot discipline, so only `_search`
    is overridden."""

    directions = PRESSES
    plan_cache_path = PLAN_CACHE

    #: Weights tried in order; the first that returns wins, so a level that can
    #: be solved at 1 keeps the shortest plan the ladder can prove.
    weights = (1, 2, 3, 5)
    #: GENERATED nodes per A* rung, not expansions -- a state generates five
    #: successors, so an expansion cap would let peak memory vary with the
    #: branching. Every level A* reaches solves inside a third of this.
    astar_cap = 800_000
    #: Beam fallback (states kept per depth, and how many depths). Levels 7 and
    #: 9 need it; it costs ~2 minutes and ~350 MB each, once, into the cache.
    beam_width = 4_000
    beam_depth = 150

    def setup(self) -> None:
        self._board = None
        self._board_sig = None

    def heuristic(self, eng) -> int:                 # pragma: no cover
        raise NotImplementedError(
            "the model owns the heuristic; see _Board.estimate")

    def _board_for(self, board):
        """Reuse one `_Board` per level so its blastable/field caches survive
        across every search on that board."""
        sig = board.signature()
        if sig != self._board_sig:
            self._board, self._board_sig = board, sig
        return self._board

    def _search(self, eng) -> list | None:
        board, state = read_board(eng, self.g)
        if state[0] is None:
            return None
        board = self._board_for(board)
        if not state[1]:
            return None                              # a corpse never moves again
        found = None
        for weight in self.weights:
            found = board.astar(state, self.astar_cap, weight)
            if found is not None:
                break
        if found is None:
            found = board.beam(state, self.beam_width, self.beam_depth)
        if found is None:
            return None
        return Plan(found, board.annotate(state, found))


class ExplodSolver(PSAStarSolver):
    game_id = "puzzlescript_explod"
    game_name = GAME_NAME
    expert_cls = ExplodExpert

    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so a later step limit or sprite fix there cannot silently
    #: diverge from what is taped here.
    game_module_id = "ps:explod"

    #: Level 5's checkerboard bomb field; see the level table in the module
    #: docstring. Skipped up front so discovery does not burn the whole node
    #: budget and then a full beam on it at every cold startup.
    skip_levels = frozenset({5})

    #: The longest plan is level 7's 88 presses; the rest is room for the
    #: exploration prefix and its RESET. Stays inside the adapter's 200-step
    #: per-level budget.
    max_steps = 150


# ---------------------------------------------------------------------------
# Reports and checks
# ---------------------------------------------------------------------------

def _levels(solver=None):
    solver = solver or ExplodSolver()
    game = solver.make_game(0)
    return game, ExplodExpert(game)


def _report() -> int:
    """Per-level board size, bomb count, plan length and tie coverage."""
    game, expert = _levels()
    eng = game._engine
    total = solved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        _board, state = read_board(eng, game._game)
        if level in ExplodSolver.skip_levels:
            print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
                  f"{len(state[3]):2d} bombs   SKIPPED (no search reaches it)")
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
              f"{len(state[3]):2d} bombs  {len(found):3d} presses ({room}), "
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
        if level in ExplodSolver.skip_levels:
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


def _ties(levels=None, cap: int = 150_000) -> int:
    """Re-solve from every press at every step of every plan and report any
    press that reaches a win at least as fast as the labelled one.

    Sound labels mean: every press we label IS as good as the expert's. A press
    this finds and the labels do not is a SIBLING PLAN -- an equally short route
    through a different state -- not a mislabelled reordering, so it is reported,
    not asserted against.

    The re-solve runs under a small node cap, so it UNDER-reports: an
    alternative whose re-solve times out is counted as no better. That is the
    right side to be wrong on for a check that exists to catch labels claiming
    too much. Pass level numbers to restrict it (a full pass is minutes)."""
    game, expert = _levels()
    eng = game._engine
    extra = tested = 0
    for level in range(game.n_levels):
        if level in ExplodSolver.skip_levels:
            continue
        if levels and level not in levels:
            continue
        game.set_level(level)
        board, state = read_board(eng, game._game)
        board = expert._board_for(board)
        plan = expert.plan(eng, level)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        found = 0
        for i, press in enumerate(plan):
            remaining = len(plan) - i
            for alt in PRESSES:
                if alt in sets[i]:
                    continue
                ns, won = board.step(state, alt)
                tested += 1
                if won:
                    found += 1
                    continue
                if ns == state or not ns[1]:
                    continue
                sub = board.astar(ns, cap, 1)
                if sub is not None and 1 + len(sub) <= remaining:
                    found += 1
            state, _won = board.step(state, press)
        extra += found
        print(f"level {level:2d}: {len(plan):3d} steps, "
              f"{found} unlabelled press(es) that are also optimal")
    print(f"ties: {tested} alternatives re-solved, {extra} sibling plans")
    return 0


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Composition, not object: the player ON the goal is the winning frame and has
    to differ from the player standing on grass, a bomb on water is un-armable
    and has to differ from a bomb on land, and each of the seven fuse states has
    to differ from the other six."""
    game = ExplodSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {
        "floor": (), "wall": ("wall",), "water": ("water",), "goal": ("goal",),
        "player": ("liveplayer",),
        "player_on_goal": ("goal", "liveplayer"),
        "player_in_water": ("water", "liveplayer"),
        "corpse": ("deadplayer",),
    }
    for name in NAME_TO_CODE:
        comps[name] = (name,)
        comps[name + "_on_water"] = ("water", name)
        comps[name + "_on_goal"] = ("goal", name)

    sizes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in comps.items():
            # Fill the WHOLE board with the composition and compare whole
            # frames. Cropping one cell out is what the sibling generators do
            # and it is wrong here: `_render_frame` upscales any sub-64 render
            # to fill the frame, so on a 5x13 board (20x52 px, scaled by 64/52)
            # the arithmetic cell_px*r lands in the wrong cell and every
            # composition reads as identical. Rendering is per-cell
            # independent, so a whole-frame comparison is the same test done
            # where the geometry cannot drift.
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g))
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


def _fuzz(n_boards: int = 3000, n_steps: int = 40, seed: int = 0) -> int:
    """Differential fuzz: random boards played move-for-move against the real
    interpreter, comparing the player, the walls, every fuse and the win flag
    after every press.

    This is the only thing standing between `_Board` and a silently wrong
    corpus, so it covers what the shipped levels do not: bombs already lit at
    every fuse value, bombs standing in water (unlightable, and doused on the
    way down), a player in water, a corpse, and dense clusters that chain."""
    game = ExplodSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    rng = random.Random(seed)
    mismatches = steps = 0
    for _ in range(n_boards):
        h, w = rng.choice([5, 6, 7]), rng.choice([5, 6, 7])
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)
        nwall = rng.randint(0, (h * w) // 4)
        walls = set(cells[:nwall])
        rest = cells[nwall:]
        if len(rest) < 3:
            continue
        nbomb = rng.randint(1, min(12, len(rest) - 2))
        bombs = {c: rng.randrange(7) for c in rest[:nbomb]}
        rest = rest[nbomb:]
        player = rest[0]
        goals = set(rest[1:1 + rng.randint(1, 2)])
        alive = rng.random() < 0.95
        # Water is on its own collision layer, so it is independent of
        # everything else -- bombs, goals and the player can all stand in it.
        water = {c for c in cells if c not in walls and rng.random() < 0.25}

        grid = []
        for r in range(h):
            row = []
            for c in range(w):
                cell = {idx["background"]}
                if (r, c) in water:
                    cell.add(idx["water"])
                if (r, c) in goals:
                    cell.add(idx["goal"])
                if (r, c) in walls:
                    cell.add(idx["wall"])
                elif (r, c) in bombs:
                    cell.add(idx[CODE_TO_NAME[bombs[(r, c)]]])
                elif (r, c) == player:
                    cell.add(idx["liveplayer" if alive else "deadplayer"])
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
                      f"  walls {sorted(walls)} water {sorted(water)} "
                      f"goals {sorted(goals)}\n"
                      f"  model  {state} won={won}\n"
                      f"  engine {truth} won={truth_won}")
                state = truth
                if mismatches > 5:
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
    sys.exit(ExplodSolver.main())
