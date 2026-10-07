"""Generate Phase-1 training data for the PuzzleScript game
ps:where_did_all_this_ice_come_from ("where did all this ice come from?",
thefifthmatt).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a NATIVE model of the
mechanic (fuzz-verified against the interpreter, velocity markers and all), the
one search that model makes affordable, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_where_did_all_this_ice_come_from",
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

THE GAME
--------
A desert mouse has fallen through an interdimensional hole into an ice cavern.
You walk, you shove boulders, and a level is won when EVERY Target square holds
a Boulder and EVERY Exit square holds the player -- whichever of the two exists
(a condition over an object the level does not contain is vacuously true, which
is what makes level 0 and the last level pure "reach the exit" boards and the
other twelve pure sokobans).

One mechanic carries all fourteen levels: **ICE, and the fact that momentum on
it is an OBJECT.** `VU`, `VD`, `VL` and `VR` are four invisible objects
(declared `Transparent`, with no sprite at all) on a collision layer of their
own, and the whole ruleset is a hand-built animation engine over them:

* A Movable (the player or a boulder) that is pressed INTO an ice square does
  not move that turn. `Up [ > Movable No VX | Ice No Movable ] -> [ VU Movable |
  Ice ]` eats the force and drops a marker on it instead; then
  `[ Movable VU ] -> [ Up Movable Up VU ] again` gives the marker AND its
  carrier a fresh force every tick, so the thing slides one square per `again`
  tick until something ends it.
* `[ Movable No Ice VX ] -> [ Movable ]` is what ends it, and it is the FIRST
  rule of every tick: a slider that lands on a square with no Ice loses its
  marker before the mover rule can fire again. So a slide stops on the first
  NON-ICE square it reaches -- not against the first obstacle.
* **Against an obstacle, the marker keeps going and the slider does not.** The
  marker is on its own collision layer, so a wall does not block it: the mover's
  force is cancelled, the marker slides one square INTO the wall, and
  `late [ VX No Movable ] -> []` deletes it there. That is the whole
  stop-at-a-wall behaviour, and it is why a slider jammed against the GRID EDGE
  (where the marker cannot go) keeps its marker forever -- see the model's
  totality note below.
* **When the obstacle is a boulder, the marker lands ON it and the momentum is
  transferred.** The blocked slider stops dead and the boulder it hit inherits
  the marker, then either slides away next tick or (if it is not on ice) has the
  marker stripped by the first rule. Billiards. `Up [ VU Boulder | Boulder Ice
  No VX | Ice ] -> [ Boulder | VU Boulder Ice | Ice ]` is the same transfer
  written out a tick earlier for the boulder-into-boulder case, which is what
  lets a strike travel down a whole ROW of boulders in one press.

Everything below was MEASURED against the interpreter, not read off the .txt.

* **A sliding player cannot push.** `[ > Player | No VX Boulder ]` is the only
  rule that puts a force on a boulder, and it needs the player to be carrying a
  force at the point the rule runs. On the turn you press a key you are; on an
  `again` tick you are not (forces are dropped and the input is not reasserted
  -- see `PuzzleScriptAdapter.step`), because the mover rule that re-arms you
  sits ELEVEN rules later in the file. So a player who slides into a boulder does
  not shove it: it hands over its marker and stops, and the boulder leaves under
  its own momentum. Both halves matter -- the boulder moves, the player does
  not follow it.
* **Pushing is a ONE-SQUARE move even onto ice.** The same ordering: the rule
  that would give the player a marker (`[ > Movable No VX | Ice No Movable ]`)
  requires the square ahead to hold `No Movable`, and while you are pushing, a
  boulder is standing in it. So you step in behind the boulder and stop there,
  on the ice, motionless. To slide you have to press into an EMPTY ice square.
* **`[ > Player | Boulder VX Ice ] -> [ Player | Boulder VX Ice ]` cancels the
  step you take behind a boulder you have just set sliding** -- but only when
  the boulder was standing ON ice. Shove a boulder that is on bare floor into an
  ice square and you do follow it in, one square, because the rule's `Ice` is
  read at the boulder's OLD square. Two different outcomes for the same press,
  and the model reproduces both.
* **`[ > Boulder | Exit ] -> [ Boulder | Exit ]`**: a boulder may never enter an
  Exit. The rule strips the force but not the marker, so a boulder that slides
  into an exit stops beside it and its marker is deleted in the exit square.
* **`noaction` is declared in the prelude and this interpreter ignores it, but
  there is nothing for the X key to do**: no rule in the file requires `action`,
  so ACTION5 leaves the grid untouched. `--selfcheck` presses it as one of five
  and measures that; the searches branch on four.
* **The win is checked after every `again` tick, so a slide that PASSES over the
  goal wins on the way past.** `PuzzleScriptAdapter.step` breaks its `again`
  loop the moment `check_win()` holds, which freezes the board mid-slide with
  the marker still on it. That is the only way a settled frame of this game ever
  shows a live marker, and the searches treat a win as terminal, so it never
  matters -- but the model reproduces it and `--selfcheck` compares it.
* Levels are all 11x11 and all walled, so the frame renders at `cell_px == 5`:
  full sprite resolution, nothing decimated.

THE MODEL
---------
Because no rule creates or destroys Ice, Target, Exit or Wall, the whole state
of a level is ``(player square, boulder set)`` -- and, on a walled board, the
markers are always gone by the time a press settles. `_simulate` is nonetheless
a faithful tick-by-tick simulation carrying the markers explicitly, including
the two branches the shipped levels can never reach (a marker stranded against
the grid edge, and a marker landing on a movable that already carries one), so
that `--selfcheck` can drive it over randomly seated boards without a wall
border and still compare grid-for-grid. `_step` is the search's view of it and
simply drops the markers; `--selfcheck` asserts that is lossless on every
reachable state of every shipped level.

THE SEARCH: ONE BOUNDED BFS, AND WHY NO HEURISTIC IS NEEDED
------------------------------------------------------------
Every level is solved by breadth-first search from its start state, stopped at
the FIRST depth that produces a win. That is a proof of shortestness by
construction, and the reason it is affordable is that the bound cuts the space
long before it closes: the deepest level is 140 presses and the widest search
any of them runs is level 7's 764,871 states, with nine of the fourteen under
53,000. Level 0 is the extreme case -- 11
boulders, whose full reachable space is over 3,000,000 states, of which the
search touches 2,924, because its exit is 18 presses away.

The exact per-step optimal SETS come out of the same search for one more pass
over the same states. Seed the ON-PATH set with the winning states at depth `D`
and sweep the layers backwards: a state at depth `k` is on a shortest win path
iff one of its successors is on-path at depth `k+1`. Then a press taken at plan
step `i` is optimal EXACTLY when it lands on an on-path state at depth `i+1`,
which is measured rather than inferred. `--ties` then re-derives every label two
further ways: a Bellman certification of the whole field against a distance
computed by a different algorithm, and a from-scratch re-solve of each plan's
tail.

A search that runs out of its state cap makes NO claim; a search that closes
with no win at any depth PROVES the level unwinnable. Nothing here reports a
plan it did not verify.

Plans are cached in ``data/where_did_all_this_ice_come_from_plans.json``: they
are seed-independent (the engine state after reset is the same for every seed,
only the presentation is augmented), so without a file on disk every shard
`parallelize_generator` starts would re-derive all fourteen.

RENDERING (`--audit`), AND WHY THE .TXT IS PATCHED
---------------------------------------------------
One sprite in ``data/puzzlescript_games/where_did_all_this_ice_come_from_.txt``
is edited -- Target's -- with the reason written into the file beside it. The
win condition was INVISIBLE: Target shipped as a ring in the middle 3x3, Boulder
is transparent only at its four CORNERS, and Player only at six pixels the ring
does not touch, so `boulder on a target` and `boulder on bare floor` rendered as
the same picture, as did the two over ice and both under the player. `--audit`
found exactly those four identical pairs. Target now paints two opposite corners
and the four edge midpoints as well -- two corners and not four, because the
corners are the only pixels a boulder shows anything through and they have to
carry both "is this a target" and "is this ice"; painting all four made `boulder
on target` and `boulder on target+ice` identical instead, and whether the square
under a boulder is slippery decides every push in the game.

`--audit` then asserts every cell COMPOSITION a frame of this game can show is
distinct, rendering whole uniform 64x64 frames rather than indexing into a mixed
one (`_render_frame` upscales and centre-pads, so indexing a cell by
``cell_px * r`` silently reads the wrong pixels -- the ps:explod lesson). It
passes at the single 11x11 board shape every level uses.

The game takes ROTATION augmentation (all four turns, verified to vary with the
seed) and no flips: `_FLIP_GAMES` is an opt-in list in the adapter and this game
is not on it. Nothing in the mechanic is chiral -- the four sliding rules are a
complete orbit and there is only ever one marker direction alive on the board at
a time -- but the Boulder and Ice sprites are asymmetric art with no mirror
partner, so a flip would buy presentations at the cost of a picture the game
never draws.

VERIFICATION
------------
``--selfcheck``  the native model against the interpreter, press for press, on
                 the shipped levels AND on randomly seated boards (which is what
                 covers the grid-edge, marker-collision and exit-block branches
                 the levels do not reach).
``--plans``      per-level plan, length, states searched, budget headroom and
                 tie coverage.
``--ties``       re-derives every optimal-action label two ways -- a Bellman
                 certification over the whole field, and a from-scratch
                 re-solve of each plan's tail.
``--closure``    the FULL reachable space per level and how much of it is
                 DEAD, for the levels where it closes -- an exhaustive BFS that
                 closes must find the same shortest win.
``--audit``      the render check described above.
``--symmetry``   the same plan replayed at every (seed, level) augmentation.
``--recovery``   re-plan from perturbed mid-episode boards.
``--speed``      wall clock per episode.

As measured on this machine:

    --plans       14 levels, 660 presses, all 14 PROVED SHORTEST, 26 steps with
                  a tie, 182s cold (the whole cost of a cold start; the plans go
                  to disk and every later process reads them)
    --selfcheck   33,600 presses over the fourteen shipped levels + 22,035 over
                  1,200 randomly seated boards, 0 mismatches; then 5,299,968
                  presses sweeping the entire searched space of every level,
                  0 of which left a marker on a settled board
    --ties        660 labels Bellman-certified against a second, independent
                  distance computation (which also re-checks the on-path
                  marking at every one of the 1,324,992 states in the fourteen
                  fields), 112 of them re-solved from scratch, 0 mismatches
    --closure     11 of the 14 levels' FULL spaces close (levels 0, 7 and 13
                  do not), and every one of the 11 reports the same shortest
                  win the bounded search proved -- 33, 41, 34, 44, 62, 25, 140,
                  28, 45, 22 and 77. Between 55% and 99.7% of each closed space
                  is DEAD
    --audit       15 compositions at the single 11x11 board shape, 0
                  indistinguishable, 0 rendered as nothing but the letterbox
    --symmetry    84 (seed, level) pairs over all 4 rotations replayed to WIN,
                  0 failures
    --recovery    112 perturbations: 107 re-planned to a WIN, 5 over the reduced
                  cap, 0 wrong. None of them landed in a dead state, which is
                  luck rather than a property of the game -- `--closure` counts
                  those, and on level 9 they are 98% of the reachable space
    --speed       ~3.5s per episode from a warm plan cache, 14 levels, ~684
                  frames

and end to end, three scratch episodes: byte-identical between two runs, every
action index in 0..5, every observation in the 0..15 palette, all 42 levels
replayed FRAME-EXACT through `perform_action` to `GameState.WIN`, and 0
unlabelled EXPERT steps out of 1,980 (78 of which carry a multi-press optimal
set).

Corpus generation is the user's to run:

    /home/simon/anaconda3/envs/ARC-AGI-3/bin/python \\
        solvers/generate_where_did_all_this_ice_come_from_training.py \\
        --episodes N --out data/training_multi_level/where_did_all_this_ice_come_from
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                          # noqa: E402
    Plan, PSAStarSolver, PSExpert,
)

GAME_NAME = "where_did_all_this_ice_come_from_"

#: Seed-independent plan cache. The fourteen searches are the whole cost of a
#: cold start -- 182s all told, of which level 7 (764,871 states) is 93s and
#: level 12 (174,746) is 38s; nine of the fourteen are under two seconds.
PLAN_CACHE = (Path(__file__).resolve().parent.parent / "data"
              / "where_did_all_this_ice_come_from_plans.json")

#: The four presses the searches branch on, in the order the tie sets list them.
#: ACTION5 is not among them: no rule in the file requires `action`, so it is a
#: measured no-op (`--selfcheck` presses it as one of five).
_DIRS = ("up", "down", "left", "right")

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Objects read off the engine grid into `_Static` / a model state. The four
#: velocity markers are invisible (Transparent sprites) and are gone by the time
#: a press settles on a walled board, but the model carries them because the
#: fuzz drives it on boards that are not walled.
_OBJECTS = ("wall", "ice", "target", "exit", "player", "boulder",
            "vu", "vd", "vl", "vr")
_VMARK = {"vu": "up", "vd": "down", "vl": "left", "vr": "right"}

#: `PuzzleScriptAdapter.step`'s own `again` cap, mirrored so the model stops
#: where the interpreter does. A slide crosses at most 9 squares on an 11x11
#: board, so no shipped press comes near it.
_MAX_AGAIN = 50

#: Runaway guard on the bounded search. The widest level touches 764,871
#: states; a search that trips this reports "nothing is claimed", never a
#: verdict -- an enumeration is only evidence when it terminates on its own.
_BFS_CAP = 3_000_000


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Static:
    """A level's immovable geometry.

    No rule in the file creates or destroys a Wall, an Ice, a Target or an Exit,
    which is what collapses a state to ``(player square, boulder set)``.
    """

    __slots__ = ("h", "w", "walls", "ice", "targets", "exits")

    def __init__(self, h, w, walls, ice, targets, exits):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.ice = frozenset(ice)
        self.targets = frozenset(targets)
        self.exits = frozenset(exits)

    def sig(self):
        return (self.h, self.w, tuple(sorted(self.walls)),
                tuple(sorted(self.ice)), tuple(sorted(self.targets)),
                tuple(sorted(self.exits)))


def _at(p, d, h, w):
    """``p + d`` inside the grid, or None when it falls off the edge."""
    r, c = p[0] + _DELTA[d][0], p[1] + _DELTA[d][1]
    return (r, c) if 0 <= r < h and 0 <= c < w else None


def _read(eng, ids):
    """``(static, player, boulders, markers)`` for the engine's current grid."""
    cells = {n: set() for n in _OBJECTS}
    for r in range(eng.height):
        for c in range(eng.width):
            cell = eng.grid[r][c]
            for name in _OBJECTS:
                if ids[name] in cell:
                    cells[name].add((r, c))
    static = _Static(eng.height, eng.width, cells["wall"], cells["ice"],
                     cells["target"], cells["exit"])
    vx = {p: d for name, d in _VMARK.items() for p in cells[name]}
    return (static, next(iter(cells["player"]), None),
            frozenset(cells["boulder"]), vx)


def _won(static, player, boulders):
    """``All Target on Boulder`` and ``All Exit on Player``.

    Both are vacuous for a level that contains none of that object, which is
    what makes the two exit levels pure races and the other twelve pure
    sokobans. The adapter's "a deleted player is not a win" guard cannot bite
    here: no rule in this game deletes the player.
    """
    return (all(t in boulders for t in static.targets)
            and all(e == player for e in static.exits))


def _resolve(occ, forces, h, w, immovable):
    """Chain-based force resolution on ONE collision layer, mirroring
    `PuzzleScriptAdapter._resolve_forces`.

    ``occ`` maps a square to the token standing on it for this layer;
    ``immovable`` is the set of squares the layer holds that never move (the
    walls, for the Player/Boulder layer). A chain is traced in its own
    direction; it moves only if its far end is free and no other chain claims a
    square it wants. ``forces`` is consumed.

    The layers do not see each other, which is the entire velocity mechanic: a
    marker walks into the wall that just stopped its carrier.
    """
    for _iter in range(20):
        moved_any = False
        resolved = set()
        chains = []
        for pos, d in list(forces.items()):
            if pos in resolved:
                continue
            if pos not in occ:
                forces.pop(pos, None)
                resolved.add(pos)
                continue
            chain = [pos]
            cur = pos
            chain_free = blocker_is_mover = False
            while True:
                nxt = _at(cur, d, h, w)
                if nxt is None:
                    break                       # off the grid: never a mover
                if nxt not in occ and nxt not in immovable:
                    chain_free = True
                    break
                if nxt in occ and forces.get(nxt) == d:
                    chain.append(nxt)           # pushed in the same direction
                    cur = nxt
                else:
                    blocker_is_mover = nxt in forces
                    break
            if chain_free:
                chains.append((chain, d))
            elif blocker_is_mover:
                pass                            # may vacate: retry next pass
            else:
                for cell in chain:
                    forces.pop(cell, None)
                    resolved.add(cell)
        non_head = {ent: i for i, (chain, _d) in enumerate(chains)
                    for ent in chain[1:]}
        subsumed = {i for i, (chain, _d) in enumerate(chains)
                    if chain[0] in non_head}
        claims: dict = {}
        conflicting = set()
        for i, (chain, d) in enumerate(chains):
            if i in subsumed:
                continue
            for cell in chain:
                key = _at(cell, d, h, w)
                other = claims.get(key)
                if other is not None and other != i:
                    conflicting.add(i)
                    conflicting.add(other)
                else:
                    claims[key] = i
        for i, (chain, d) in enumerate(chains):
            if i in subsumed:
                continue
            if i in conflicting:
                for cell in chain:
                    forces.pop(cell, None)
                    resolved.add(cell)
                continue
            for cell in reversed(chain):
                occ[_at(cell, d, h, w)] = occ.pop(cell)
                forces.pop(cell, None)
                resolved.add(cell)
                moved_any = True
        if not moved_any:
            forces.clear()
            break
        if not forces:
            break
    return occ


def _simulate(static, player, boulders, press, vx0=None):
    """One press, tick by tick. Returns ``(player, boulders, markers)``.

    This is the faithful model: the sixteen main rules in file order (eight
    blocks, three of which are a direction apiece), then the force resolution,
    then the one `late` rule, repeated while `again` holds AND
    the grid actually changed (the interpreter's own guard, without which a
    slider jammed against a wall would spin forever). A win breaks the loop
    where the interpreter breaks it, which is why a settled frame can carry a
    marker after a slide that won on its way past the goal.

    ``markers`` is normally empty; it is threaded in and out so `--selfcheck`
    can drive boards without a wall border, where a slider pinned against the
    GRID EDGE keeps its marker (the marker has nowhere to be deleted).
    """
    h, w, ice = static.h, static.w, static.ice
    P = player
    B = set(boulders)
    vx = dict(vx0) if vx0 else {}
    fm: dict = {}
    if press in _DELTA and P is not None:
        fm[P] = press

    def movable(p):
        return p == P or p in B

    for _tick in range(_MAX_AGAIN):
        before = (P, frozenset(B), frozenset(vx.items()))
        fv: dict = {}
        again = False

        # [ > Player | No VX Boulder ] -> [ > Player | > Boulder ]
        # The player is only ever carrying a force here on the turn a key was
        # pressed, so this is the rule that makes a SLIDING player unable to
        # push (see the module docstring).
        if P is not None and P in fm:
            q = _at(P, fm[P], h, w)
            if q is not None and q in B and q not in vx:
                fm[q] = fm[P]

        # [ Movable No Ice VX ] -> [ Movable ]   -- a slide ends on the first
        # square with no ice under it, and this runs FIRST, before the mover.
        for p in list(vx):
            if p not in ice and movable(p):
                del vx[p]

        # Dir [ > Movable No VX | Ice No Movable ] -> [ VDir Movable | Ice ]
        for d in _DIRS:
            for p in [x for x in list(fm) if fm.get(x) == d]:
                if p in vx or not movable(p):
                    continue
                q = _at(p, d, h, w)
                if q is not None and q in ice and not movable(q):
                    vx[p] = d
                    del fm[p]

        # [ > Player | Boulder VX Ice ] -> [ Player | Boulder VX Ice ]
        if P is not None and P in fm:
            q = _at(P, fm[P], h, w)
            if q is not None and q in B and q in vx and q in ice:
                del fm[P]

        # Dir [ VDir Boulder | Boulder Ice No VX | Ice ]
        #  -> [ Boulder | VDir Boulder Ice | Ice ]      -- billiards down a row
        for d in _DIRS:
            changed = True
            while changed:
                changed = False
                for a in [x for x in list(vx) if vx.get(x) == d and x in B]:
                    b = _at(a, d, h, w)
                    if b is None or b not in B or b not in ice or b in vx:
                        continue
                    c = _at(b, d, h, w)
                    if c is None or c not in ice:
                        continue
                    del vx[a]
                    vx[b] = d
                    changed = True

        # [ Movable VDir ] -> [ Dir Movable Dir VDir ] again
        for d in _DIRS:
            for p in [x for x in list(vx) if vx.get(x) == d]:
                if not movable(p):
                    continue
                again = True
                if fm.get(p) is None:
                    fm[p] = d
                elif fm[p] != d:
                    del fm[p]              # conflicting forces cancel
                fv[p] = d

        # [ > Boulder | Exit ] -> [ Boulder | Exit ]
        for p in [x for x in list(fm) if x in B]:
            q = _at(p, fm[p], h, w)
            if q is not None and q in static.exits:
                del fm[p]

        occ = {} if P is None else {P: "P"}
        for b in B:
            occ[b] = "B"
        occ = _resolve(occ, fm, h, w, static.walls)
        P = next((p for p, t in occ.items() if t == "P"), None)
        B = {p for p, t in occ.items() if t == "B"}
        vx = _resolve(dict(vx), fv, h, w, frozenset())

        # late [ VX No Movable ] -> []
        for p in list(vx):
            if not movable(p):
                del vx[p]

        if not again or (P, frozenset(B), frozenset(vx.items())) == before:
            break
        if _won(static, P, B):
            break                       # the interpreter breaks here too
    return P, frozenset(B), vx


def _step(static, state, press):
    """The search's view of a press: ``(player, boulders)`` in, the same out.

    The markers are dropped, which is lossless on every state a walled level can
    reach -- `--selfcheck` asserts exactly that over the whole searched space of
    every shipped level.
    """
    P, B, _vx = _simulate(static, state[0], state[1], press)
    return (P, B)


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

class _Field:
    """A bounded breadth-first search and the ON-PATH marking over it.

    ``depth`` is the length of a shortest win (None when there is none, or when
    the cap was hit); ``closed`` says the search ended on its OWN terms rather
    than on the cap, which is what turns "no win was found" into a PROOF that
    there is none. ``onpath`` is every state that lies on SOME shortest win
    path, which is what makes the per-step optimal sets exact rather than
    inferred.
    """

    __slots__ = ("depth", "closed", "capped", "dist", "onpath", "n_states")

    def __init__(self, depth, closed, capped, dist, onpath, n_states):
        self.depth = depth
        self.closed = closed
        self.capped = capped
        self.dist = dist
        self.onpath = onpath
        self.n_states = n_states

    def solvable(self) -> bool:
        return self.depth is not None


def build_field(static, start, cap=_BFS_CAP) -> _Field:
    """Breadth-first search from ``start``, stopped at the first depth that
    produces a win, then one backward sweep to mark the on-path states.

    Stopping at the first winning depth is what makes this affordable: level 0's
    reachable space is over 3,000,000 states and the search touches 2,924 of
    them, because the answer is 18 presses deep. It is also why the result is a
    PROOF of shortestness -- BFS reaches every state at its true press distance,
    so nothing shallower exists to be missed.
    """
    if _won(static, *start):
        return _Field(0, True, False, {start: 0}, {start}, 1)
    dist = {start: 0}
    layers = [[start]]
    wins: list = []
    capped = False
    depth = 0
    while True:
        nxt = []
        for s in layers[depth]:
            for press in _DIRS:
                t = _step(static, s, press)
                if t == s or t in dist:
                    continue
                dist[t] = depth + 1
                if _won(static, *t):
                    wins.append(t)
                else:
                    nxt.append(t)
            if len(dist) > cap:
                capped = True
                break
        # `capped` is tested FIRST and on purpose: the cap can trip part-way
        # through a layer, and a layer that was not fully scanned cannot carry
        # a complete on-path marking. A win found in that half-layer is dropped
        # rather than shipped with tie sets that might be missing a press. No
        # shipped level comes within a factor of three of the cap.
        if capped or wins or not nxt:
            break
        layers.append(nxt)
        depth += 1
    if capped:
        return _Field(None, False, True, dist, set(), len(dist))
    if not wins:
        # The space closed with no win at any depth: PROVED unwinnable.
        return _Field(None, True, False, dist, set(), len(dist))


    D = depth + 1
    onpath = set(wins)
    for k in range(D - 1, -1, -1):
        for s in layers[k]:
            for press in _DIRS:
                t = _step(static, s, press)
                if t != s and dist.get(t) == k + 1 and t in onpath:
                    onpath.add(s)
                    break
    return _Field(D, True, False, dist, onpath, len(dist))


def field_plan(field: _Field, static, start):
    """``(plan, optsets)`` read straight off the field, or ``(None, None)``.

    ``optsets[i]`` is every press at step ``i`` that lands on an ON-PATH state
    one layer deeper -- i.e. every press that still finishes in the moves the
    recorded one leaves. Measured, not inferred; `--ties` re-derives every one
    of them two further ways.
    """
    if not field.solvable():
        return None, None
    plan, sets, cur = [], [], start
    for k in range(field.depth):
        best = []
        for press in _DIRS:
            t = _step(static, cur, press)
            if t != cur and field.dist.get(t) == k + 1 and t in field.onpath:
                best.append(press)
        if not best:                                   # cannot happen on-path
            return None, None
        plan.append(best[0])
        sets.append(best)
        cur = _step(static, cur, best[0])
    return plan, sets


def dist_to_win(field: _Field, static) -> dict:
    """``{state: presses to a shortest win}`` over the field's own subgraph,
    derived a SECOND way: an explicit predecessor map and a backward BFS from
    the winning states, instead of the forward layer sweep `build_field` uses.

    `--ties` certifies the shipped labels against this. It is the same answer by
    a different algorithm over the same states, and it checks EVERY state in the
    field rather than only the ones a plan walks through -- which is the check a
    from-scratch re-solve cannot afford here, a single re-solve near the start of
    level 7 being most of a 765,000-state search.
    """
    pred: dict = {}
    wins = []
    for s in field.dist:
        if _won(static, *s):
            wins.append(s)
            continue
        for press in _DIRS:
            t = _step(static, s, press)
            if t != s and t in field.dist:
                pred.setdefault(t, []).append(s)
    out = {s: 0 for s in wins}
    queue = deque(wins)
    while queue:
        s = queue.popleft()
        for p in pred.get(s, ()):
            if p not in out:
                out[p] = out[s] + 1
                queue.append(p)
    return out


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class IceExpert(PSExpert):
    """Bounded BFS over the native model.

    The base class's engine-blackbox A* is replaced wholesale (`_search`): an
    interpreter step is ~350us and these searches run to hundreds of thousands
    of states, so they run on the fuzz-verified native model instead and only
    the plan comes back. Everything else -- the memo, the disk cache, the
    restore discipline, the recording -- is the base's.
    """

    directions = list(_DIRS)
    plan_cache_path = PLAN_CACHE

    #: ``node_cap`` and ``weight`` are the base class's A* dials and nothing
    #: here reads them: the search is a breadth-first one with no heuristic to
    #: weight, and its own guard is `_BFS_CAP`. They stay on the constructor
    #: because `PSAStarSolver._ensure` passes them.

    def setup(self) -> None:
        self.ids = {n: self.g.obj_name_to_idx[n] for n in _OBJECTS}
        #: (static signature, start state, cap) -> field, capped at two
        #: entries. A field is hundreds of thousands of states and only the one
        #: being planned from is live, so an unbounded memo would hold every
        #: level's at once for no gain -- the plans themselves are memoized (and
        #: cached to disk) by the base class, which is what repeat calls hit.
        self._fields: dict = {}

    def read(self, eng):
        static, player, boulders, vx = _read(eng, self.ids)
        return static, (player, boulders), vx

    def field(self, static, state, cap=_BFS_CAP) -> _Field:
        key = (static.sig(), state, cap)
        got = self._fields.get(key)
        if got is None:
            got = build_field(static, state, cap)
            while len(self._fields) >= 2:
                self._fields.pop(next(iter(self._fields)))
            self._fields[key] = got
        return got

    # `_key` is the base's: every non-background cell, walls and ice included.
    # It is canonical ACROSS levels (so `scope_by_level` stays off and two
    # levels can never serve each other's plans), it is what the disk cache
    # stores its start signature from, and on 121-cell boards the extra static
    # entries cost nothing.

    def dead(self, eng) -> bool:
        """True when this state can never reach a win -- PROVED, not guessed:
        the search closed over the whole reachable space without one. A capped
        search is NOT a death sentence and returns False."""
        static, state, _vx = self.read(eng)
        field = self.field(static, state)
        return field.closed and not field.solvable()

    def _search(self, eng):
        static, state, _vx = self.read(eng)
        field = self.field(static, state)
        plan, sets = field_plan(field, static, state)
        return None if plan is None else Plan(plan, sets)


class IceSolver(PSAStarSolver):
    game_id = "puzzlescript_where_did_all_this_ice_come_from"
    game_name = GAME_NAME
    expert_cls = IceExpert

    #: Record against the GAME FOLDER's adapter. ``games/ps:where_did_all_this_
    #: ice_come_from`` is a plain passthrough today; naming it means a sprite
    #: patch or a step cap added there later cannot silently make this generator
    #: tape a game nobody plays (the ps:count_mover trap).
    game_module_id = "ps:where_did_all_this_ice_come_from"

    #: Room for the longest plan (140, level 8) plus the RESET exploration
    #: prefix and the re-plan after it, inside the adapter's 200-press
    #: per-level budget.
    max_steps = 160


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = IceSolver()
    game = solver.make_game(seed)
    expert = IceExpert(game, node_cap=IceSolver.node_cap)
    return solver, game, expert


def _levels(game):
    return [lv for lv in range(game.n_levels)
            if lv not in IceSolver.skip_levels]


def _plans() -> int:
    """Per-level plan, length, states searched, budget headroom, tie coverage.

    The field is built here rather than read off `expert.plan`, so the state
    counts are the ones this run measured; the plan the base class would ship
    is then fetched and asserted equal, which is what checks that the disk
    cache (and the `Plan` JSON round-trip) still says the same thing.
    """
    _solver, game, expert = _new()
    eng = game._engine
    total = tied = bad = 0
    t_all = time.time()
    for level in _levels(game):
        game.set_level(level)
        static, state, _vx = expert.read(eng)
        t0 = time.time()
        field = expert.field(static, state)
        plan, sets = field_plan(field, static, state)
        dt = time.time() - t0
        head = (f"level {level:2d}: {len(state[1])} boulder(s), "
                f"{len(static.targets)} target(s), {len(static.exits)} exit(s), "
                f"{len(static.ice)} ice")
        if plan is None:
            why = ("PROVED UNWINNABLE -- the reachable space closed with no win"
                   if field.closed
                   else f"no claim -- capped at {field.n_states:,} states")
            print(f"{head} -- NO PLAN ({why})", flush=True)
            continue
        ties = sum(1 for x in sets if len(x) > 1)
        total += len(plan)
        tied += ties
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        shipped = expert.plan(eng, level)
        if list(shipped or []) != plan or list(shipped.optsets) != sets:
            print(f"  level {level}: the shipped plan differs from this field's")
            bad += 1
        print(f"{head}, {len(plan):3d} presses (PROVED SHORTEST, "
              f"{field.n_states:,} states, budget {game._max_steps} {room}), "
              f"{ties} tie set(s), {dt:6.1f}s", flush=True)
        print(f"    {' '.join(plan)}", flush=True)
    print(f"total {total} presses, {tied} step(s) with a tie, "
          f"{time.time() - t_all:.0f}s")
    return 0 if not bad else 1


def _closure(cap: int = 3_000_000) -> int:
    """The FULL reachable space per level, and how much of it is DEAD.

    Two things the bounded search does not have to touch, reported for contrast
    and as an independent check on it: an exhaustive BFS that closes finds every
    win there is, so the shortest one it reports must equal the bounded search's
    depth. The dead census is the second half -- a backward BFS from the wins
    over a predecessor map, counting the states from which no win is reachable
    at all.

    That census is why `--recovery` has a "PROVED lost" bucket. Most of this
    game is unrecoverable: measured, between 55% (level 4) and 99.7% (level 3)
    of a closed space is states no sequence of presses wins from -- a
    boulder slid into a corner cannot be pulled back out, because nothing in
    this game pulls. A level whose space does not close inside the cap is
    reported as exactly that and nothing is claimed about it; level 0's 11
    boulders are the case, and the cap is where it is because the predecessor
    map is what costs (about 1 GB per million states).

    The successors are generated TWICE -- once to close the space, once to build
    the predecessor map -- rather than kept from the first pass, which is a
    straight trade of a second sweep for holding only one big dict instead of
    two.
    """
    _solver, game, expert = _new()
    eng = game._engine
    for level in _levels(game):
        game.set_level(level)
        static, start, _vx = expert.read(eng)
        t0 = time.time()
        seen = {start}
        queue = deque([start])
        over = False
        while queue:
            s = queue.popleft()
            if _won(static, *s):
                continue
            for press in _DIRS:
                t = _step(static, s, press)
                if t != s and t not in seen:
                    seen.add(t)
                    queue.append(t)
            if len(seen) > cap:
                over = True
                break
        if over:
            print(f"level {level:2d}: over {cap:,} states in "
                  f"{time.time() - t0:6.1f}s -- did not close, nothing claimed",
                  flush=True)
            continue
        pred: dict = {}
        wins = []
        for s in seen:
            if _won(static, *s):
                wins.append(s)
                continue
            for press in _DIRS:
                t = _step(static, s, press)
                if t != s:
                    pred.setdefault(t, []).append(s)
        dtw = {s: 0 for s in wins}
        q2 = deque(wins)
        while q2:
            s = q2.popleft()
            for p in pred.get(s, ()):
                if p not in dtw:
                    dtw[p] = dtw[s] + 1
                    q2.append(p)
        dead = len(seen) - len(dtw)
        win = dtw.get(start)
        print(f"level {level:2d}: {len(seen):9,} states, {dead:9,} dead "
              f"({100 * dead / len(seen):4.1f}%), "
              f"{'shortest win %d presses' % win if win is not None else 'NO win exists'}"
              f", {time.time() - t0:6.1f}s", flush=True)
    return 0


def _selfcheck(walks: int = 40, steps: int = 60, boards: int = 1200) -> int:
    """The native model against the interpreter, press for press.

    Three passes, and the last two are the ones that matter. Random walks on the
    SHIPPED levels cover what a plan touches. Randomly SEATED boards -- random
    walls (with the border present only half the time), ice, targets, exits and
    up to a third of the free squares filled with boulders -- are what reach the
    grid-edge, marker-collision and exit-block branches the walled levels never
    put a press through. And the searched space of every shipped level is swept
    to assert that dropping the markers (`_step`) is lossless there.
    """
    _solver, game, expert = _new()
    eng = game._engine
    ids = dict(expert.ids)
    ids["background"] = game._game.obj_name_to_idx["background"]
    rng = random.Random(20260823)
    presses5 = list(_DIRS) + ["action"]
    bad = presses = 0

    for level in _levels(game):
        for _walk in range(walks):
            game.set_level(level)
            static, (P, B), vx = expert.read(eng)
            for _t in range(steps):
                press = rng.choice(presses5)
                eng.step(press)
                P, B, vx = _simulate(static, P, B, press, vx)
                presses += 1
                _s, live, live_vx = expert.read(eng)
                if live != (P, B) or live_vx != vx:
                    print(f"  level {level}: model != engine after {press}")
                    bad += 1
                    break
                if eng.check_win():
                    break
    print(f"shipped levels: {presses} presses, {bad} mismatch(es)")

    seated = 0
    for trial in range(boards):
        h, w = rng.randint(4, 9), rng.randint(4, 9)
        border = rng.random() < 0.5
        grid = [[{ids["background"]} for _ in range(w)] for _ in range(h)]
        free = []
        for r in range(h):
            for c in range(w):
                if ((border and (r in (0, h - 1) or c in (0, w - 1)))
                        or rng.random() < 0.12):
                    grid[r][c].add(ids["wall"])
                else:
                    free.append((r, c))
        for p in free:
            if rng.random() < 0.55:
                grid[p[0]][p[1]].add(ids["ice"])
            if rng.random() < 0.10:
                grid[p[0]][p[1]].add(ids["target"])
            elif rng.random() < 0.08:
                grid[p[0]][p[1]].add(ids["exit"])
        rng.shuffle(free)
        for p in free[:rng.randint(0, max(1, len(free) // 3))]:
            grid[p[0]][p[1]].add(ids["boulder"])
        rest = [p for p in free if ids["boulder"] not in grid[p[0]][p[1]]]
        if rest:
            p = rng.choice(rest)
            grid[p[0]][p[1]].add(ids["player"])
        eng.height, eng.width, eng.grid = h, w, grid
        eng._rule_win = eng._rule_restart = eng._late_cancel = False
        eng._position_index_dirty = True
        eng._force_rigid_group = {}
        eng._rule_noop_cache.clear()
        static, (P, B), vx = expert.read(eng)
        for _t in range(20):
            press = rng.choice(presses5)
            eng.step(press)
            P, B, vx = _simulate(static, P, B, press, vx)
            seated += 1
            _s, live, live_vx = expert.read(eng)
            if live != (P, B) or live_vx != vx:
                print(f"  seated board {trial}: model != engine after {press}")
                bad += 1
                break
            if eng.check_win():
                break
    print(f"seated boards: {boards} boards, {seated} presses")

    # Markers are STATE on a board with no wall border; on the shipped levels
    # they are not, and the searches depend on that. Sweep every state the
    # bounded search reaches and assert no press leaves one behind (a win is
    # the one exception, and it is terminal).
    stranded = swept = 0
    for level in _levels(game):
        game.set_level(level)
        static, start, _vx = expert.read(eng)
        field = expert.field(static, start)
        for s in field.dist:
            for press in _DIRS:
                P2, B2, vx2 = _simulate(static, s[0], s[1], press)
                swept += 1
                if vx2 and not _won(static, P2, B2):
                    stranded += 1
    print(f"searched space: {swept} presses over {len(_levels(game))} levels, "
          f"{stranded} settled state(s) carrying a marker")
    bad += stranded
    print("selfcheck clean" if not bad
          else f"SELFCHECK FAILED: {bad} mismatch(es)")
    return 0 if not bad else 1


def _ties(tail: int = 8) -> int:
    """Re-derive every optimal-action label two independent ways.

    (1) **Bellman certification over the whole field.** `dist_to_win` recomputes
    the distance to a shortest win with an explicit predecessor map and a
    backward BFS -- a different algorithm over the same states from the forward
    layer sweep that produced the labels. It certifies the plan's length, every
    step's set, AND the on-path marking for every state in the field, not just
    the ones the plan walks through.

    (2) **A from-scratch re-solve of the plan's tail.** For the last ``tail``
    steps of each level a fresh bounded BFS is run from every candidate
    successor and the label kept iff ``1 + its depth`` is what was left. That is
    the fully independent check; it is limited to the tail because a re-solve
    from near the START of level 7 is most of a 765,000-state search, four times
    over, at every one of 83 steps.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = certified = resolved = 0
    for level in _levels(game):
        game.set_level(level)
        static, state, _vx = expert.read(eng)
        field = expert.field(static, state)
        plan, sets = field_plan(field, static, state)
        if plan is None:
            print(f"level {level:2d}: no plan (skipped)")
            continue
        dtw = dist_to_win(field, static)
        if dtw.get(state) != field.depth:
            print(f"  level {level}: Bellman says {dtw.get(state)}, "
                  f"the field says {field.depth}")
            bad += 1
        for s, d in field.dist.items():
            want = (d + dtw[s] == field.depth) if s in dtw else False
            if want != (s in field.onpath):
                print(f"  level {level}: on-path disagrees at {s}")
                bad += 1
                break
        cur = state
        for pi, (taken, claimed) in enumerate(zip(plan, sets)):
            remaining = field.depth - pi
            measured = []
            for press in _DIRS:
                nxt = _step(static, cur, press)
                if nxt != cur and dtw.get(nxt, 1 << 30) == remaining - 1:
                    measured.append(press)
            if measured != list(claimed):
                print(f"  level {level} step {pi}: labelled {list(claimed)} "
                      f"but Bellman measured {measured}")
                bad += 1
            if taken not in claimed:
                print(f"  level {level} step {pi}: took {taken}, not in its "
                      f"own set {list(claimed)}")
                bad += 1
            certified += 1
            cur = _step(static, cur, taken)

        cur = state
        for pi, (taken, claimed) in enumerate(zip(plan, sets)):
            if pi >= field.depth - tail:
                remaining = field.depth - pi
                measured = []
                for press in _DIRS:
                    nxt = _step(static, cur, press)
                    if nxt == cur:
                        continue
                    if _won(static, *nxt):
                        cost = 1
                    else:
                        sub = build_field(static, nxt, cap=400_000)
                        cost = None if not sub.solvable() else 1 + sub.depth
                    if cost == remaining:
                        measured.append(press)
                if measured != list(claimed):
                    print(f"  level {level} step {pi}: labelled "
                          f"{list(claimed)} but a re-solve measured {measured}")
                    bad += 1
                resolved += 1
            cur = _step(static, cur, taken)
        print(f"level {level:2d}: {len(plan)} step(s) certified, "
              f"{min(tail, len(plan))} re-solved", flush=True)
    print(f"{certified} label(s) Bellman-certified, {resolved} re-solved "
          f"from scratch")
    print("ties clean" if not bad else f"TIES FAILED: {bad} mismatch(es)")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION a frame of this game can show is distinct.

    Whole 64x64 frames of UNIFORM boards are compared rather than one cell out
    of a mixed board: `_render_frame` upscales and centre-pads, so indexing a
    cell by ``cell_px * r`` silently reads the wrong pixels (the ps:explod
    lesson). Two uniform boards render identically iff their cells do.

    The compositions are the five things a square can BE -- bare floor, ice, a
    target, a target that is also ice, an exit -- times the three things that
    can stand on one, plus the wall. A boulder in an exit is not among them:
    `[ > Boulder | Exit ]` makes it unreachable.
    """
    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    floors = {"floor": (), "ice": ("ice",), "target": ("target",),
              "target_ice": ("target", "ice"), "exit": ("exit",)}
    covers = {"": (), "player": ("player",), "boulder": ("boulder",)}

    sizes: dict = {}
    for level in _levels(game):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        comps = {}
        for fname, fobjs in floors.items():
            for cname, cobjs in covers.items():
                if fname == "exit" and cname == "boulder":
                    continue
                comps[f"{fname}+{cname}" if cname else fname] = fobjs + cobjs
        comps["wall"] = ("wall",)
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        # The ps:stand_iii check: a composition that renders as nothing but the
        # letterbox pad is invisible against the frame's own border.
        pad = [n for n, s in shots.items()
               if len(set(s.flatten().tolist())) == 1]
        bad += len(clashes) + len(pad)
        note = "OK" if not (clashes or pad) else f"IDENTICAL {clashes} PAD {pad}"
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} "
              f"compositions -- {note}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable composition(s)")
    return 0 if not bad else 1


def _symmetry(seeds: int = 6) -> int:
    """Replay each level's plan at several seeds -- i.e. at several board
    rotations -- and assert every one reaches WIN.

    This is the rotation contract under test: the plans are derived in ENGINE
    space, and `screen_action` has to invert whatever the adapter forward-remaps
    before the press is driven or recorded. Get it backwards and nothing raises;
    three of every four orientations simply play a different game.
    """
    from arcengine import ActionInput, GameState

    from solvers.common.ps_astar import screen_action

    solver = IceSolver()
    bad = pairs = 0
    rots = set()
    for seed in range(seeds):
        game = solver.make_game(seed)
        expert = IceExpert(game, node_cap=IceSolver.node_cap)
        for level in _levels(game):
            game.set_level(level)
            plan = expert.plan(game._engine, level)
            if plan is None:
                continue
            rot, hf, vf = game._rotation_k, game._hflip, game._vflip
            rots.add((rot, hf, vf))
            for press in plan:
                game.perform_action(ActionInput(
                    id=screen_action(press, rot, hf, vf,
                                     IceSolver.remap_actions)))
            pairs += 1
            if game._state != GameState.WIN:
                print(f"  seed {seed} level {level}: {game._state} "
                      f"(rot {rot}, hflip {hf}, vflip {vf})")
                bad += 1
    print(f"symmetry: {pairs} (seed, level) pair(s) over {len(rots)} "
          f"presentation(s), {bad} failure(s)")
    return 0 if not bad else 1


def _recovery(trials: int = 8, cap: int = 100_000) -> int:
    """Re-plan from perturbed mid-episode boards.

    Walk each plan to a random point, take a random press off it, and ask for a
    plan from wherever that left the board. A perturbed state falls into exactly
    one of three buckets and keeping them apart is the point: it re-plans to a
    WIN, or it is PROVED lost (its whole reachable space closed with no win in
    it -- a boulder shot into a corner it can never be pulled out of, which this
    game really does have), or the reduced cap ran out, which is not a claim
    about the board at all. Only a plan that does NOT win is a failure.

    Recording never pays this cost: with `epsilon` at 0 and RESET-mode recovery
    the only `plan` call is at a level's start, which is a disk-cache hit. The
    report exists to show the expert CAN re-plan off-path, and a cap that let
    every trial re-run a 765,000-state search would only make it unrunnable.
    """
    _solver, game, expert = _new()
    eng = game._engine
    rng = random.Random(7)
    ok = proved_lost = over = wrong = 0
    for level in _levels(game):
        game.set_level(level)
        static, start, _vx = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        t0 = time.time()
        buckets = [0, 0, 0, 0]
        for _t in range(trials):
            cur = start
            for press in plan[:rng.randrange(0, len(plan))]:
                cur = _step(static, cur, press)
            cur = _step(static, cur, rng.choice(_DIRS))
            if _won(static, *cur):
                buckets[0] += 1
                continue
            field = build_field(static, cur, cap=cap)
            if not field.solvable():
                buckets[1 if field.closed else 2] += 1
                continue
            again, _sets = field_plan(field, static, cur)
            check = cur
            for press in again:
                check = _step(static, check, press)
            buckets[0 if _won(static, *check) else 3] += 1
        ok += buckets[0]
        proved_lost += buckets[1]
        over += buckets[2]
        wrong += buckets[3]
        print(f"level {level:2d}: {buckets[0]} win, {buckets[1]} proved lost, "
              f"{buckets[2]} over the cap, {buckets[3]} WRONG "
              f"({time.time() - t0:5.1f}s)", flush=True)
    print(f"recovery: {ok} re-planned to a WIN, {proved_lost} PROVED lost, "
          f"{over} over the reduced cap, {wrong} WRONG")
    return 0 if not wrong else 1


def _speed(seeds: int = 3) -> int:
    """Wall clock per episode, from a warm plan cache."""
    solver = IceSolver()
    for seed in range(seeds):
        t0 = time.time()
        ok, levels = solver.solve_episode(seed)
        frames = sum(len(lv["observations"]) for lv in levels)
        print(f"seed {seed}: ok={ok} {len(levels)} level(s), {frames} frames, "
              f"{time.time() - t0:5.1f}s")
    return 0


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--closure" in sys.argv:
        sys.exit(_closure())
    if "--selfcheck" in sys.argv:
        sys.exit(_selfcheck())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--recovery" in sys.argv:
        sys.exit(_recovery())
    if "--speed" in sys.argv:
        sys.exit(_speed())
    sys.exit(IceSolver.main())
