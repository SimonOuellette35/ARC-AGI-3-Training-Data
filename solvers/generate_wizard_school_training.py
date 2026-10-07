"""Generate Phase-1 training data for the PuzzleScript game ps:wizard_school
("Wizard School!", Luke Davies).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the exact distance FIELD that model is solved
with, the proofs that every plan is shortest and that the sixth level cannot be
won at all, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_wizard_school",
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

The game
--------
You are a wizard with no hands. You cannot walk into a box, you cannot push one,
and you cannot pick one up by standing next to it. What you have is one spell,
bound to X, and everything below was MEASURED against the interpreter rather than
read off the .txt.

* **X while empty-handed is a SUMMON, fired down all four rays at once.** The
  four ``[action playeridle | no blocker] -> [ | castdir]`` rules all fire on the
  same press, so a cast leaves the player in every direction that is not blocked
  at point-blank range. Each cast then walks its ray outward one cell per
  ``again`` tick until it meets a wall (it dies) or a box (the box turns into
  ``boxsummoned`` and flies back down the same ray to the square beside you,
  where it becomes ``boxcarried``). A box already touching you is taken by
  ``[action playeridle | boxidle] -> [ | boxcarried]`` in the first tick instead.
  So one press collects **up to four boxes, one per direction: the nearest box in
  each of your row and column halves**, and nothing else on the board moves.
* **The cast cannot pass a box, and only the FIRST box on a ray comes.** The
  propagation rule is ``[castdir | no wall]``, but the cast is on the same
  collision layer as the boxes, so a box in the way stops it dead -- and the
  ``[castdir | boxidle]`` rule that precedes it consumes the cast on that box
  anyway. Boxes further along the same ray never move.
* **Carried boxes ride along with you, and they keep their offset FOREVER.**
  ``rigid [moving player | boxcarried] -> [moving player | moving boxcarried]``
  gives every box touching you the press's force, so the whole cross translates.
  There is no rule anywhere that changes which side of you a box is on, which is
  the whole difficulty of the game: the direction a box will be thrown is decided
  the moment it is summoned.
* **A carried box shoved into a WALL cancels the entire turn.** ``[> boxcarried |
  wall] -> cancel`` -- so a cross with a box above you simply cannot climb into
  the top row, and the press is a total no-op rather than a partial move.
* **A carried box shoved into a BOX does not cancel -- it SHEARS the cross.**
  This is the one genuinely surprising rule and it is not in the .txt at all; it
  falls out of how `PuzzleScriptAdapter` tags rigid groups. ``_tag_rigid_forces``
  writes ``force_rigid_group[key] = gid`` for *both* cells of each match, and the
  player is in all four matches, so the player's tag is overwritten by every
  later direction and survives only from the LAST one (rule directions run in the
  order up, down, left, right). The player is therefore rigidly bound to exactly
  one of its boxes. Block that box and the cascade cancels the player too, while
  the other three boxes move on without you; block any other box and only that
  box stops. `_Board._move` reproduces this force-resolution verbatim, and
  ``--fuzz`` is what proves it does.
* **X while carrying is a THROW, and it is all-or-nothing.** Every box TOUCHING
  you is flung in the direction it was sitting, travels until a wall or a box is
  in front of it, and lands there as ``boxidle``. You cannot throw one box and
  keep another -- ``[action playercarrying | boxcarried]`` fires in all four
  directions on the same press.
* **The player is empty-handed exactly when no ``boxcarried`` exists** anywhere
  on the board -- ``[boxcarried][playeridle] -> [ | playercarrying]`` is the last
  rule of every turn and has no adjacency in it. That is why a SHEARED-off box
  (one the cross left behind, no longer touching you) is a soft trap: it is still
  ``boxcarried``, so you are still "carrying", so X throws nothing and summons
  nothing. Walking back next to it re-attaches it.
* **A box in flight is not a box.** ``boxthrown*`` is outside the ``box``
  synonym, so the win condition ignores it mid-air, and it does not block another
  thrown box either (their rays never cross, so that never comes up).
* **A press with nothing to move is an exact no-op**, there is no death, no
  restart and nothing destructible: the box count is invariant (no rule creates
  or deletes one) and the target set is untouched by every rule in the file.

The win is ``all target on box``: every target square carrying a box, in any
state -- idle, summoned, or still in your hands.

The levels
----------
Six, of which the interpreter strips the four ``message`` screens:

    level    0     1     2     3     4     5
    boxes    1     1     1     4     2     4
    targets  1     1     1     4     2     8
    d*       1     2     7    19    18     -

**Level 5 cannot be won, and that is a proof rather than a search giving up.**
Its map is ``@``-heavy -- four boxes each starting on a target -- plus four bare
targets, so it asks for eight covered targets from four boxes. No rule in the
file creates or destroys a box or a target, so the counts are invariant and the
win condition is unsatisfiable from the start. ``--proof`` states that argument
and then backs it with the whole-space enumeration: all 895,664 reachable states,
not one winning edge. The level is in `WizardSchoolSolver.skip_levels` so nothing
pays for that enumeration at startup.

The model and the field
-----------------------
``eng.step`` runs at ~950 presses/s here: a summon is a whole ``again`` chain,
one tick per cell the cast and then the box travels, and every tick re-scans 35
rules over the grid. Level 3's reachable space is 1.15 MILLION states and 5.76M
edges, so walking it through the interpreter is about 1.7 HOURS where `_Board`
does it in 16 s -- and the interpreter's job here is to CERTIFY the answer rather
than to find it (``--engine`` and ``--selfcheck`` replay every plan through it
press for press, ``--enumerate`` replays entire levels).

`_Board` is the mechanic above as one function; `_Field` enumerates a level's
whole reachable space forward from its start, then labels it by ONE backward
sweep over the recorded edges. Backward is the only direction that gives
presses-to-win, and it has to be done over recorded edges rather than by
inverting the mechanic: a throw is irreversible (the box is gone down its ray)
and so is a shear (the cross that broke up cannot be un-broken), so this game has
no predecessor function to sweep with. The reward for enumerating instead of
searching is what the rest of the family gets from `StateGraph`:

* plans that are provably SHORTEST -- no heuristic, no weight, no node cap;
* EXACT optimal-action sets (``dist(succ) == dist - 1``) rather than inferred
  ones, which matters here because level 3 is four-fold symmetric and its first
  press genuinely ties FOUR ways and its second THREE -- an inferred label would
  have taught one arbitrary quarter of that board as the only right answer;
* an exact answer for ANY state the recorder lands in, so an epsilon detour is
  re-planned rather than reset;
* and a PROOF of unwinnability for level 5.

The two tutorial levels take no exploration prefix
--------------------------------------------------
The prefix is a random walk that stops the moment it terminates the level, and
levels 0 and 1 are one and two presses long, so it simply won them -- level 0 in
18 recorded episodes out of 20, level 1 in 6. A level the prefix wins is taped
entirely as unlabelled ``phase="explore"`` steps: `_reset_prefix` skips its
closing RESET, `record_level`'s win check fires before the expert is ever asked
for a plan, and the level trains nothing. Since those two levels are exactly the
ones that teach the game's only two verbs, `WizardSchoolSolver.explore_level`
turns the prefix off for them; the other three are 7 to 19 presses and keep it.
``--labels`` is the check that says so, and `PSAStarSolver.explore_level` is the
(default-inert) hook it needed.

Optimal-action sets
-------------------
Every recorded expert press carries ``optimal``: every press whose successor is
one step nearer a win, straight out of the field. Ties break in `DIRS` order, so
a re-derived plan is byte-identical across processes.

How much of the game is genuinely order-free, per level's own plan:

    level 2   1 1 1 1 1 1 1                     (nothing ties: 7 forced presses)
    level 3   4 3 2 1 2 2 2 1 1 1 ...           (the symmetric room)
    level 4   1 1 1 1 1 2 2 2 1 1 ... 2

``--selfcheck`` re-derives every one of those sets from the field a second time,
checks the recorded press is in its own set, and then checks each press the set
NAMES really does finish in the same number of moves. ``--labels`` checks the
other half of the contract: that a label reaches every expert step of every
level at all.

Rendering
---------
``Target`` is on its own collision layer UNDER the boxes and both box sprites are
opaque 5x5 squares, so a box on a target used to render pixel-identically to a
box on plain floor -- the win condition was invisible. `games/ps:wizard_school`'s
``make_game`` punches the four edge-midpoint pixels of ``boxidle`` and
``boxcarried`` out to transparent, which is exactly where the target's black
diamond has its tips, and `WizardSchoolSolver.game_module_id` points the
generator at that wrapper so it records the frames a live agent sees.
``--audit`` compares WHOLE frames of a board filled with each of the thirteen
legal cell compositions, on every level, and all thirteen are now distinct.

Reports (none of them writes training data)
-------------------------------------------
``--plans``      per level: d*, the plan, the tie count and the press budget
``--proof``      why level 5 is unwinnable, argued and then enumerated
``--fuzz``       differential model-vs-interpreter over random walks AND
                 synthetic boards (the shear cases random play rarely reaches)
``--enumerate``  the same check over a level's WHOLE enumerated space -- every
                 reachable state, every press. Levels 0, 1, 2 and 4 by default;
                 ``--enumerate 3 5 --sample 40000`` covers the two big levels'
                 deep states without paying ~3 h of interpreter for them
``--selfcheck``  replay every plan through the interpreter, assert WIN, and
                 check every optimal-action label against the field
``--recovery``   re-plan from perturbed mid-episode boards
``--engine``     drive the real adapter (rotation and all) through each plan
``--audit``      whole-frame distinctness of every cell composition
``--symmetry``   the recorded frames are the transform of the unaugmented ones
``--labels``     record episodes IN MEMORY and audit what would actually train:
                 every expert step labelled, every level contributing one, and
                 the same seeds recorded twice coming out identical
``--replay``     replay a recorded episode's OWN action stream (prefix, RESET,
                 detours and all) through a fresh adapter, pixel for pixel

Generate with (see the note in `solvers/common/ps_astar.py` on the disk plan
cache -- delete `data/wizard_school_plans.json` to re-derive):

    python solvers/generate_wizard_school_training.py --episodes 1000 \
        --out data/training_multi_level/wizard_school
"""

from __future__ import annotations

import random
import sys
from array import array
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                          # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION                     # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,   # noqa: E402
                                     screen_action)
from utils.rotation import inverse_remap_action_full                  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_ID = "ps:wizard_school"
GAME_NAME = "Wizard_School!"

#: Engine presses, in the order ties are broken. ACTION is the whole game (it is
#: both the summon and the throw), so unlike the pure-walking ps: games this one
#: cannot drop it from the branching.
DIRS: tuple[str, ...] = ("up", "down", "left", "right", "action")

#: The four MOVE directions, in the order `PuzzleScriptAdapter` expands a
#: direction-less rule over them. That order is load-bearing: it decides which
#: box the player is rigidly bound to, and therefore which way the cross shears.
#: See `_Board._move`.
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

_ACTION = 4          # index of "action" in DIRS

#: Disk cache of every level's start plan AND its optimal-action sets. Level 3's
#: field is 1.15M states and 16 s to sweep, so this is load-bearing for a
#: `parallelize_generator` fan-out: without it every shard re-derives it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "wizard_school_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic.

    Cells are flat ``r * w + c`` indices. A STATE is ``(player, idle mask,
    carried mask)`` -- the three things every rule in this game moves, and the
    only three: walls and targets are untouched by the whole rule list, so they
    live here rather than in the state. The two box sets are bitmasks because the
    sweeps below key over a million states and a Python int is the cheapest exact
    spelling of a subset of squares.

    There is deliberately no state for a box in FLIGHT or mid-SUMMON: neither
    survives a settled turn, EXCEPT on the frame the adapter stops its ``again``
    loop on because the win condition just became true (see `_cast`). Such a
    state is terminal, and `WizardSchoolExpert.read` folds the stranded
    ``boxsummoned`` into the carried mask, where `won` counts it as the box it
    is. If that were wrong the differential fuzz would say so -- it reads the
    engine's grid back through exactly that mapping and compares.
    """

    __slots__ = ("h", "w", "n", "wall", "tmask", "free", "nbr", "ray", "sig")

    def __init__(self, h, w, walls, targets):
        self.h, self.w = h, w
        self.n = h * w
        self.wall = frozenset(walls)
        self.tmask = 0
        for t in targets:
            self.tmask |= 1 << t
        self.free = tuple(i for i in range(self.n) if i not in self.wall)
        #: ``nbr[cell][di]`` is the neighbour cell, or -1 off the board. The wall
        #: test is NOT folded in here (unlike the plain sokobans in this family):
        #: the cancel rule needs "a wall specifically", and off-the-board is a
        #: different answer from a wall to it.
        self.nbr = []
        #: ``ray[cell][di]`` is every cell from ``cell + d`` outward to the edge.
        #: A cast and a thrown box both walk one of these.
        self.ray = []
        for i in range(self.n):
            r, c = divmod(i, w)
            row, rays = [], []
            for d in ("up", "down", "left", "right"):
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
                path = []
                while 0 <= nr < h and 0 <= nc < w:
                    path.append(nr * w + nc)
                    nr, nc = nr + dr, nc + dc
                rays.append(tuple(path))
            self.nbr.append(tuple(row))
            self.ray.append(tuple(rays))
        self.sig = (h, w, tuple(sorted(self.wall)), tuple(sorted(targets)))

    # -- the win condition ---------------------------------------------------
    def won(self, state) -> bool:
        """``all target on box`` -- verbatim. ``box`` is the synonym for idle,
        summoned and carried alike, so a box still in the wizard's hands counts;
        a box in FLIGHT does not, but flight never survives a settled turn."""
        return ((state[1] | state[2]) & self.tmask) == self.tmask

    # -- the mechanic --------------------------------------------------------
    def step(self, state, di: int):
        """``(state after pressing DIRS[di], did that press win)``.

        The win is reported separately from the state because it can happen
        DURING a press: the adapter stops its ``again`` loop the moment the win
        condition holds, and a summoned box sweeping down its ray can cover the
        last target on its way past. See `_cast`."""
        if di == _ACTION:
            if state[2]:
                nxt = self._throw(state)
                return nxt, self.won(nxt)
            nxt, transient = self._cast(state)
            return nxt, transient or self.won(nxt)
        nxt = self._move(state, di)
        return nxt, self.won(nxt)

    def _adjacent_carried(self, state) -> list:
        """``[(cell, direction index), ...]`` for the carried boxes touching the
        player, in the order the adapter expands a direction-less rule. Every
        rule that reads ``boxcarried`` beside a player -- the rigid rule, the
        cancel, the throw -- is expanded in exactly this order, and for the rigid
        rule that order is what decides the shear."""
        p, _idle, car = state
        nbr = self.nbr[p]
        return [(nbr[dj], dj) for dj in range(4)
                if nbr[dj] >= 0 and (car >> nbr[dj]) & 1]

    def _move(self, state, di: int):
        """A direction press: the player and every box touching it translate.

        This is `PuzzleScriptAdapter._resolve_forces` specialised to the one
        collision layer this game moves on, and it is written to MATCH that
        function rather than to be the obvious thing, because the obvious thing
        (rigid = all-or-nothing) is not what the adapter does. See the module
        docstring's shear note; ``--fuzz`` and ``--enumerate`` are what hold the
        two in step.
        """
        p, idle, car = state
        nbr, wall = self.nbr, self.wall
        adj = self._adjacent_carried(state)

        # ``[> boxcarried | wall] -> cancel`` -- the whole turn is reverted, so a
        # cross with a box facing a wall does not shear, it simply does nothing.
        for b, _dj in adj:
            nb = nbr[b][di]
            if nb >= 0 and nb in wall:
                return state

        forces = {p: True}
        gid = {}
        for i, (b, _dj) in enumerate(adj):
            forces[b] = True
            gid[b] = i
            gid[p] = i          # overwritten by every later match: the shear

        pl = p
        for _ in range(20):
            resolved, movable, blocked = set(), [], set()
            for e in list(forces):
                if e in resolved:
                    continue
                chain, cur, free = [e], e, False
                while True:
                    nx = nbr[cur][di]
                    if nx < 0:
                        break
                    if not (nx in wall or (idle >> nx) & 1
                            or (car >> nx) & 1 or nx == pl):
                        free = True
                        break
                    if nx in forces:        # every force here points at ``di``
                        chain.append(nx)
                        cur = nx
                    else:
                        break
                if free:
                    movable.append(chain)
                else:
                    for x in chain:
                        forces.pop(x, None)
                        resolved.add(x)
                        blocked.add(x)

            if blocked:
                dead = {gid[k] for k in blocked if k in gid}
                if dead:
                    extra = set(blocked)
                    for k, gv in list(gid.items()):
                        if gv in dead and k in forces:
                            extra.add(k)
                    for k in extra:
                        forces.pop(k, None)
                        resolved.add(k)
                        gid.pop(k, None)
                    if extra - blocked:
                        movable = [m for m in movable
                                   if not any(x in extra for x in m)]
            for k in blocked:
                gid.pop(k, None)

            non_head = {}
            for i, ch in enumerate(movable):
                for x in ch[1:]:
                    non_head[x] = i
            subsumed = {i for i, ch in enumerate(movable) if ch[0] in non_head}

            claims, conflicting = {}, set()
            for i, ch in enumerate(movable):
                if i in subsumed:
                    continue
                for x in ch:
                    t = nbr[x][di]
                    if t in claims and claims[t] != i:
                        conflicting.add(i)
                        conflicting.add(claims[t])
                    else:
                        claims[t] = i
            if conflicting and gid:
                ckeys = set()
                for i in conflicting:
                    ckeys |= set(movable[i])
                cgroups = {gid[k] for k in ckeys if k in gid}
                if cgroups:
                    ext = set(ckeys)
                    for k, gv in gid.items():
                        if gv in cgroups:
                            ext.add(k)
                    for i, ch in enumerate(movable):
                        if i in subsumed or i in conflicting:
                            continue
                        if any(x in ext for x in ch):
                            conflicting.add(i)

            moved = False
            for i, ch in enumerate(movable):
                if i in subsumed:
                    continue
                if i in conflicting:
                    for x in ch:
                        forces.pop(x, None)
                        resolved.add(x)
                        gid.pop(x, None)
                    continue
                for x in reversed(ch):
                    t = nbr[x][di]
                    if x == pl:
                        pl = t
                    elif (car >> x) & 1:
                        car ^= 1 << x
                        car |= 1 << t
                    elif (idle >> x) & 1:
                        idle ^= 1 << x
                        idle |= 1 << t
                    forces.pop(x, None)
                    gid.pop(x, None)
                    moved = True
            if not moved:
                break
        return (pl, idle, car)

    def _cast(self, state):
        """X with empty hands: summon down all four rays at once.

        Returns ``(state, transient win)``. The transient flag is the one place
        this model has to know about TICKS rather than settled boards: a summoned
        box crosses every square between where it stood and the wizard's side,
        and if one of those crossings covers the last bare target the adapter
        stops its ``again`` loop right there and the level is won -- with the box
        sitting somewhere the settled state would never show it. Level 0 is
        exactly that: its single target is on the box's way home.

        The arrival schedule is measured, not assumed. A cast spawns beside the
        player on tick 1 and rides one cell per tick, so the box at distance ``m``
        is converted on tick ``m - 1``; ``[boxsummoned | ... | player]`` is rule
        18 and the conversion is rule 6, so the box also takes its first step
        home on that same tick. That gives ``pos(t) = max(1, m - max(0, t - m +
        2))``, and the last interesting tick is ``2m - 2``.
        """
        p, idle, _car = state           # ``_car`` is 0: X only casts empty-handed
        car = 0
        travels = []
        for dj in range(4):
            nb = self.nbr[p][dj]
            if nb < 0 or nb in self.wall:
                continue
            if (idle >> nb) & 1:        # point blank: taken where it stands
                idle ^= 1 << nb
                car |= 1 << nb
                continue
            path = self.ray[p][dj]
            for m in range(2, len(path) + 1):
                cell = path[m - 1]
                if cell in self.wall:
                    break               # the cast dies on the wall
                if (idle >> cell) & 1:
                    idle ^= 1 << cell
                    car |= 1 << nb
                    travels.append((path[:m], m))
                    break

        new = (p, idle, car)
        if not travels:
            return new, False

        static = idle | car
        for path, _m in travels:
            static &= ~(1 << path[0])   # the ones still in the air
        for t in range(1, max(2 * m - 2 for _p, m in travels) + 1):
            cur = static
            for path, m in travels:
                cur |= 1 << path[max(1, m - max(0, t - m + 2)) - 1]
            if (cur & self.tmask) == self.tmask:
                return new, True
        return new, False

    def _throw(self, state):
        """X with a box in hand: fling EVERY box touching the player at once.

        A box left behind by a shear is not touching the player, so it is not
        thrown -- and while it exists the player counts as carrying, so X cannot
        summon either. With no box touching the player at all the press is an
        exact no-op, which is the soft trap the module docstring describes.
        """
        p, idle, car = state
        adj = self._adjacent_carried(state)
        if not adj:
            return state
        blockers = idle | car
        for q, _dj in adj:
            blockers &= ~(1 << q)       # a box in flight blocks nothing
        for q, dj in adj:
            car ^= 1 << q
            cur = q
            while True:
                nx = self.nbr[cur][dj]
                if nx < 0 or nx in self.wall or (blockers >> nx) & 1:
                    break
                cur = nx
            idle |= 1 << cur
        return (p, idle, car)

    # -- pretty-printing (reports only) --------------------------------------
    def ascii(self, state) -> str:
        p, idle, car = state
        out = []
        for r in range(self.h):
            line = []
            for c in range(self.w):
                i = r * self.w + c
                t = (self.tmask >> i) & 1
                if i in self.wall:
                    ch = "#"
                elif i == p:
                    ch = "P" if car else "p"
                elif (idle >> i) & 1:
                    ch = "B" if t else "b"
                elif (car >> i) & 1:
                    ch = "C" if t else "c"
                else:
                    ch = "t" if t else "."
                line.append(ch)
            out.append("".join(line))
        return "\n".join(out)


# ---------------------------------------------------------------------------
# The exact distance field
# ---------------------------------------------------------------------------

class _Field:
    """Exact presses-to-win for every state a level can reach.

    TWO SWEEPS, AND WHY IT IS NOT ONE. `PSExpert`'s siblings that own a
    reversible mechanic (ps:swap_sokoban, ps:time_reversed_microban) get the
    distance-to-win from a single BFS out of the goal states, because there the
    forward mechanic enumerates predecessors just as well as successors. Nothing
    in this game is invertible: a throw sends a box down a ray with no record of
    where it started, and a shear cannot be un-sheared. So the space is
    enumerated FORWARD from the level start, recording every edge, and the edges
    are then swept BACKWARD from the winning ones. Both halves are exact; what
    is lost is laziness, which is why the cap below is a real number rather than
    a formality.

    Level 3 is the size of this: 1,152,831 states and 5.76M edges, 16 s and
    ~0.4 GB of peak RSS to build, against ~1.7 hours to walk the same space
    through the interpreter. Levels 0, 1, 2 and 4 are 7, 8, 83 and 5,999 states,
    and together take under a second.

    ``edges[5 * i + di]`` is the index of the state pressing ``DIRS[di]`` lands
    in, ``_WIN`` when that press wins, or ``_NONE`` when it changes nothing at
    all (a no-move: no shortest path contains one). ``dist[i]`` is presses to a
    win, or -1 for a state that can never reach one.
    """

    _WIN = -2
    _NONE = -1

    __slots__ = ("board", "start", "index", "edges", "dist", "capped")

    def __init__(self, board: _Board, start, cap: int):
        self.board = board
        self.start = start
        self.capped = False
        self.index: dict = {start: 0}
        order = [start]
        #: Raw int32 rather than a Python list: level 3 has 5.76M edges, and a
        #: list of that many boxed ints is ~200 MB of peak on its own.
        edges = array("i")
        assert edges.itemsize == 4, "array('i') is not int32 on this platform"
        head = 0
        while head < len(order):
            st = order[head]
            head += 1
            for di in range(5):
                nxt, win = board.step(st, di)
                if win:
                    edges.append(self._WIN)
                    continue
                if nxt == st:
                    edges.append(self._NONE)
                    continue
                j = self.index.get(nxt)
                if j is None:
                    if len(order) >= cap:
                        self.capped = True
                        self.edges = self.dist = None
                        return
                    j = len(order)
                    self.index[nxt] = j
                    order.append(nxt)
                edges.append(j)
        self.edges = np.frombuffer(
            edges, dtype=np.int32).reshape(len(order), 5)
        self.dist = self._sweep(self.edges)

    @staticmethod
    def _sweep(edges: np.ndarray) -> np.ndarray:
        """Presses-to-win for every state, by backward BFS over the recorded
        edges. Vectorised because the frontier of level 3's sweep runs to a
        hundred thousand states a layer and the whole point of the model was to
        stop paying Python per state."""
        n = edges.shape[0]
        dst = edges.ravel()
        keep = dst >= 0
        dst_k = dst[keep]
        src_k = np.repeat(np.arange(n, dtype=np.int32), 5)[keep]
        order = np.argsort(dst_k, kind="stable")
        indices = src_k[order]                       # predecessors, grouped
        indptr = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(np.bincount(dst_k, minlength=n), out=indptr[1:])

        dist = np.full(n, -1, dtype=np.int32)
        frontier = np.nonzero((edges == _Field._WIN).any(axis=1))[0]
        dist[frontier] = 1
        d = 1
        while frontier.size:
            starts, ends = indptr[frontier], indptr[frontier + 1]
            lens = ends - starts
            total = int(lens.sum())
            if not total:
                break
            offs = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens)
            preds = indices[np.repeat(starts, lens) + offs]
            preds = np.unique(preds[dist[preds] < 0])
            if not preds.size:
                break
            d += 1
            dist[preds] = d
            frontier = preds
        return dist

    # -- queries -------------------------------------------------------------
    def get(self, state) -> "int | None":
        """Presses to a win from ``state``, 0 if it is already won, or None when
        the state is unreachable from this level's start or cannot win at all."""
        if self.capped:
            return None
        if self.board.won(state):
            return 0
        i = self.index.get(state)
        if i is None:
            return None
        return None if self.dist[i] < 0 else int(self.dist[i])

    def optimal(self, state) -> list:
        """``[(direction index, successor state), ...]`` for every press on a
        shortest path from ``state``, in `DIRS` order.

        Exact and nothing inferred: a press is optimal iff its successor is one
        step nearer the win, and both distances come from the same sweep."""
        rest = self.get(state)
        if not rest:                    # None (unreachable / dead) or 0 (won)
            return []
        i = self.index[state]
        out = []
        for di in range(5):
            e = int(self.edges[i, di])
            if e == self._NONE:
                continue
            if e == self._WIN:
                if rest == 1:
                    out.append((di, None))
                continue
            if self.dist[e] > 0 and self.dist[e] + 1 == rest:
                out.append((di, self.board.step(state, di)[0]))
        return out

    def plan(self, state) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal set
        at every step, or None if this level cannot be won from there."""
        rest = self.get(state)
        if rest is None:
            return None
        presses, optsets, cur = [], [], state
        for _ in range(rest):
            best = self.optimal(cur)
            if not best:
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class WizardSchoolExpert(PSExpert):
    """`PSExpert`'s plan memo, disk cache and snapshot discipline around a
    `_Field`.

    The base class keeps the memo, the level scoping and the disk layer; only the
    strategy underneath changes, the way `PSEnumExpert` replaces it. Here the
    strategy is "read the live board and walk down the exact distance field", so
    `heuristic` is never called and asserts rather than returning a number
    nothing would use.

    `_search` reads the ENGINE's grid every time, so a re-plan from an arbitrary
    state -- the board an exploration prefix left behind, an epsilon detour --
    is answered exactly, including the answer "this board is lost", which this
    game really can produce (throw the last box down the wrong ray).
    """

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, which is canonical WITHIN a level but
    #: not across them: walls and targets are static per level and differ between
    #: levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Runaway guard on one level's forward enumeration. The largest shipped
    #: level that is attempted is level 3 at 1.15M states; level 5 (895k, and
    #: skipped) is the only other one over ten thousand. Past the cap `get`
    #: answers None, which `record_level` turns into a RESET.
    field_cap: int = 4_000_000

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.target_ids = set(g.resolve_object_name("target"))
        self.idle_ids = {g.obj_name_to_idx["boxidle"]}
        #: ``boxsummoned`` only ever survives a turn the adapter cut short on the
        #: win, so it is folded in with the carried boxes -- where `_Board.won`
        #: counts it, which is the only thing such a state is ever asked.
        self.carried_ids = {g.obj_name_to_idx["boxcarried"],
                            g.obj_name_to_idx["boxsummoned"]}
        #: `_Board`s and `_Field`s by STATIC signature, not by level index, so
        #: the tables are built once per level however many states are read.
        self._boards: dict = {}
        self._fields: dict = {}
        #: Each level's START state, by board signature. Read HERE, at
        #: construction, and never again: `_search` has to know which state a
        #: level's field was swept from, and re-seating levels to find that out
        #: while a recording is in flight would reset the adapter's per-level
        #: press budget and its `GameState` under the recorder's feet.
        self._starts: dict = {}
        game = self.game
        here = game._current_level_index
        for level in range(game.n_levels):
            game.set_level(level)
            board, start = self.read(game._engine)
            self._starts.setdefault(board.sig, start)
        game.set_level(here)

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "WizardSchoolExpert reads an exact distance field; heuristic is unused")

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid."""
        h, w = eng.height, eng.width
        walls, targets = [], []
        idle = car = 0
        player = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.target_ids:
                    targets.append(i)
                if cell & self.idle_ids:
                    idle |= 1 << i
                if cell & self.carried_ids:
                    car |= 1 << i
                if cell & self.player_ids:
                    player = i
        # Built exactly the way `_Board.sig` is, so the board cache, the field
        # cache and `_starts` are all keyed by the same tuple.
        sig = (h, w, tuple(sorted(walls)), tuple(sorted(targets)))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, targets)
        return board, (player, idle, car)

    def field(self, board: _Board) -> _Field:
        """``board``'s field, swept forward from the LEVEL START it was read at.

        A field only knows the states reachable from the start it was built at,
        and that is the right domain for every query the recorder makes: the
        level start itself, whatever an exploration prefix left behind (the
        prefix ends in a RESET back to the start), and whatever an epsilon detour
        landed in -- all reachable by construction. A state that is not in it
        gets an honest None."""
        got = self._fields.get(board.sig)
        if got is None:
            got = self._fields[board.sig] = _Field(
                board, self._starts[board.sig], self.field_cap)
        return got

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples: the player (0), the idle boxes (1) and
        the carried ones (2).

        Built from the MODEL's reading rather than from raw object ids, so the
        key is exactly what a state consists of -- which is also what the
        ``plan_cache_path`` signature is stored as, so a cached plan is matched
        against the board it was solved from and an edited level is a miss rather
        than a wrong plan."""
        _board, (p, idle, car) = self.read(eng)
        if p is None:
            return frozenset()
        w = eng.width
        out = {(p // w, p % w, 0)}
        for tag, mask in ((1, idle), (2, car)):
            i = 0
            while mask:
                if mask & 1:
                    out.add((i // w, i % w, tag))
                mask >>= 1
                i += 1
        return frozenset(out)

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if state[0] is None:                 # no player on the board
            return None
        if board.won(state):
            return Plan([], [])
        if board.sig not in self._starts:     # a board from no shipped level
            return None
        return self.field(board).plan(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class WizardSchoolSolver(PSAStarSolver):
    game_id = "puzzlescript_wizard_school"
    game_name = GAME_NAME
    expert_cls = WizardSchoolExpert

    #: `games/ps:wizard_school/ps:wizard_school.py` patches the two box sprites
    #: so a covered target is visible (see that file and ``--audit``), so the
    #: generator MUST build the adapter through it or it would tape frames the
    #: live agent never sees.
    game_module_id = GAME_ID

    #: Level 5 asks for eight covered targets and ships four boxes; no rule in
    #: the file creates a box or a target, so it is unwinnable by counting. It is
    #: skipped rather than discovered so no startup pays for the 895k-state
    #: enumeration that says so. ``--proof`` runs that enumeration on demand.
    skip_levels = frozenset({5})

    #: Unused: `WizardSchoolExpert._search` never calls `_astar`. Left at the base
    #: values so nothing reads a lie off them; `WizardSchoolExpert.field_cap` is
    #: the knob that actually bounds the sweep.
    weight = 1
    node_cap = 400_000

    #: The longest plan is 19 presses (level 3); the rest is headroom for the
    #: re-plans an epsilon detour costs. The adapter's own 200-press per-level
    #: budget is separate and is reset by the `set_level` that ends the
    #: exploration prefix, so a plan starts it from zero.
    max_steps = 120

    #: On top of the RESET prefix: roughly one press in twelve is a random legal
    #: alternative, after which the expert re-plans from wherever it landed --
    #: the taken action is the mistake and ``optimal`` is the recovery, which is
    #: the signal a policy needs after its own error. `record_level` only keeps a
    #: detour that leaves the level still winnable, which matters here in a way
    #: it does not for the reversible games in this family: throwing the last box
    #: down the wrong ray really can lose a board, and the field is what says so.
    #:
    #: The cost of a non-zero epsilon is that every worker eventually builds
    #: level 3's field (16 s, ~0.4 GB of peak RSS) instead of only reading its
    #: start plan out of `PLAN_CACHE` -- so `parallelize_generator`'s default 8
    #: workers want ~3 GB between them. Set it to 0.0 for a fan-out that cannot
    #: afford that: the RESET prefix still supplies recovery data, just not the
    #: mid-plan mistake-and-correct kind.
    epsilon = 0.08

    def epsilon_for(self, level: int) -> float:
        """No detours on the two levels that are over in one or two presses --
        there is no "recover from" there, and a detour on level 0 is just a
        wasted press in front of the only press the level has."""
        return 0.0 if level <= 1 else self.epsilon

    def explore_level(self, level: int) -> bool:
        """No exploration prefix on the two TUTORIAL levels.

        They are one and two presses long, and the prefix is a random walk that
        stops the moment it terminates the level, so it simply won them: over 20
        recorded episodes it took level 0 in 18 and level 1 in 6, and a level the
        prefix wins is taped entirely as unlabelled ``phase="explore"`` steps
        with no expert plan behind them -- i.e. the two levels that teach the
        game's only two verbs were training nothing. The other three levels are
        7 to 19 presses and keep the prefix. See `PSAStarSolver.explore_level`.
        """
        return level > 1


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    """The solver, its adapter and an expert -- without planning anything.

    The reports below each want a different slice (``--audit`` needs no plan at
    all, ``--fuzz`` needs only the model), so planning is left to whoever asks
    for it rather than paid on every entry point."""
    solver = WizardSchoolSolver()
    game = solver.make_game(seed)
    return solver, game, WizardSchoolExpert(game, node_cap=solver.node_cap)


def _levels(solver, game) -> list:
    """The levels a recording would actually keep."""
    return [lvl for lvl in range(game.n_levels)
            if lvl not in solver.skip_levels]


def _plans(verbose: bool = True) -> int:
    """Per level: the field's size, d*, the plan, how many optimal-action
    labels it carries in total, and whether it fits the adapter's press
    budget."""
    solver, game, expert = _new()
    bad = 0
    if verbose:
        print(f"{'lvl':>3}  {'states':>9}  {'d*':>3}  {'labels':>6}  "
              f"{'budget':>6}  plan")
    for level in _levels(solver, game):
        game.set_level(level)
        board, _start = expert.read(game._engine)
        field = expert.field(board)
        plan = expert.plan(game._engine, level)
        if plan is None:
            print(f"  level {level}: NO PLAN")
            bad += 1
            continue
        states = 0 if field.capped else len(field.index)
        ties = sum(len(s) for s in plan.optsets)
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        if verbose:
            print(f"{level:>3}  {states:>9}  {len(plan):>3}  {ties:>6}  "
                  f"{game._max_steps:>6} {room}  {' '.join(plan)}")
        if len(plan) >= game._max_steps:
            bad += 1
    return bad


def _proof(verbose: bool = True) -> int:
    """Level 5 is unwinnable -- argued from an invariant, then enumerated.

    The argument alone is enough (a win needs one box per target, the level ships
    four boxes and eight targets, and no rule in the file makes or breaks either
    one), but it is worth spending the enumeration once: it is the difference
    between "the level is unwinnable" and "our search did not find anything", and
    it also confirms the model agrees with the argument.
    """
    solver, game, expert = _new()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, start = expert.read(game._engine)
        boxes = bin(start[1] | start[2]).count("1")
        targets = bin(board.tmask).count("1")
        skipped = level in solver.skip_levels
        if verbose:
            print(f"  level {level}: {boxes} box(es), {targets} target(s)"
                  f"{'   [skipped]' if skipped else ''}")
        if boxes < targets and not skipped:
            print(f"    level {level} cannot be won and is NOT skipped")
            bad += 1
        if boxes >= targets and skipped:
            print(f"    level {level} is skipped but has enough boxes")
            bad += 1
    for level in sorted(solver.skip_levels):
        game.set_level(level)
        board, start = expert.read(game._engine)
        field = _Field(board, start, expert.field_cap)
        wins = 0 if field.capped else int((field.edges == _Field._WIN).sum())
        if verbose:
            print(f"  level {level}: enumerated "
                  f"{'CAPPED' if field.capped else f'{len(field.index)} states'}"
                  f", {wins} winning edges")
        if field.capped or wins:
            print(f"    level {level}: enumeration did not prove unwinnability")
            bad += 1
    return bad


def _install(game, board: _Board, state) -> None:
    """Write a model state onto the engine grid, so the interpreter can be asked
    about a board the model reached rather than only about boards a random walk
    happened to produce."""
    eng = game._engine
    N = game._game.obj_name_to_idx
    p, idle, car = state
    eng.grid = [[set() for _ in range(board.w)] for _ in range(board.h)]
    for i in range(board.n):
        cell = eng.grid[i // board.w][i % board.w]
        cell.add(N["background"])
        if (board.tmask >> i) & 1:
            cell.add(N["target"])
        if i in board.wall:
            cell.add(N["plainwall"])
        if (idle >> i) & 1:
            cell.add(N["boxidle"])
        if (car >> i) & 1:
            cell.add(N["boxcarried"])
    eng.grid[p // board.w][p % board.w].add(
        N["playercarrying"] if car else N["playeridle"])
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _compare(game, expert, board: _Board, state, di: int) -> str:
    """Step one press on both the interpreter and the model; '' when they agree.

    A WIN is compared as a win and not as a board: the adapter stops its ``again``
    loop the instant the win condition holds, so the grid it leaves behind is a
    frozen mid-animation frame (a ``boxsummoned`` in the middle of its flight
    home) that no settled state would ever show. That state is terminal -- the
    recorder breaks on it -- so agreeing on WIN is agreeing on everything that
    can still be acted on."""
    _install(game, board, state)
    game._engine.step(DIRS[di])
    got_win = bool(game._engine.check_win())
    _b, got = expert.read(game._engine)
    exp, exp_win = board.step(state, di)
    if got_win != exp_win:
        return (f"win disagreement on {DIRS[di]}:\n"
                f"{board.ascii(state)}\n  engine says {got_win}, "
                f"model says {exp_win}")
    if not got_win and got != exp:
        return (f"board disagreement on {DIRS[di]}:\nfrom\n{board.ascii(state)}\n"
                f"engine\n{board.ascii(got)}\nmodel\n{board.ascii(exp)}")
    return ""


def _fuzz(walks: int = 120, boards: int = 900, presses: int = 40,
          verbose: bool = True) -> int:
    """Differential model-vs-interpreter, two ways.

    RANDOM WALKS cover what a recording actually meets. SYNTHETIC BOARDS -- a
    random player square and a random idle/carried split over random squares --
    cover what it does not: the shear cases need a carried box facing another
    box, and a walk from a level start reaches those only by accident. Both
    install the board on the engine and compare one press.
    """
    _solver, game, expert = _new()
    bad = 0
    checked = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, start = expert.read(game._engine)
        rng = random.Random(f"wizard_school:fuzz:{level}")
        for _ in range(walks):
            st = start
            for _ in range(presses):
                di = rng.randrange(5)
                msg = _compare(game, expert, board, st, di)
                checked += 1
                if msg:
                    bad += 1
                    if bad <= 5:
                        print(f"  level {level}: {msg}")
                    break
                st, win = board.step(st, di)
                if win:
                    break
        free = list(board.free)
        nboxes = bin(start[1] | start[2]).count("1")
        for _ in range(boards):
            k = rng.randint(1, min(len(free) - 1, max(nboxes, 4)))
            cells = rng.sample(free, k + 1)
            p, rest = cells[0], cells[1:]
            ncar = rng.randint(0, k)
            idle = car = 0
            for i, cell in enumerate(rest):
                if i < ncar:
                    car |= 1 << cell
                else:
                    idle |= 1 << cell
            st = (p, idle, car)
            if board.won(st):
                continue          # terminal: the recorder never steps from one
            for di in range(5):
                msg = _compare(game, expert, board, st, di)
                checked += 1
                if msg:
                    bad += 1
                    if bad <= 5:
                        print(f"  level {level}: {msg}")
    if verbose:
        print(f"  {checked} transitions compared")
    return bad


def _enumerate(levels=None, sample: int = 0, verbose: bool = True) -> int:
    """The same differential check over a level's WHOLE enumerated space: every
    reachable state, every press, against the interpreter.

    This is the strongest statement available about the model, and unlike
    ``--fuzz`` it reaches the deep states -- the ones a random walk from a level
    start almost never produces. Levels 0, 1, 2 and 4 (7, 8, 83 and 5,999 states)
    take well under a minute together and are the default.

    Levels 3 and 5 are 1.15M and 895k states, i.e. ~1.7 h and ~1.3 h of
    interpreter at ~950 presses/s. ``sample`` is for them: it draws that many
    states UNIFORMLY out of the whole enumerated space, so the deep half is
    covered at a cost you can choose. Both have been run at 40,000 and 20,000
    states (300,000 compared presses, zero disagreements):

        python solvers/generate_wizard_school_training.py \
            --enumerate 3 5 --sample 40000
    """
    _solver, game, expert = _new()
    levels = levels or [0, 1, 2, 4]
    bad = 0
    for level in levels:
        game.set_level(level)
        board, start = expert.read(game._engine)
        field = _Field(board, start, expert.field_cap)
        if field.capped:
            print(f"  level {level}: field capped, not enumerated")
            bad += 1
            continue
        states = list(field.index)
        total = len(states)
        if sample and sample < total:
            states = random.Random(f"wizard_school:enumerate:{level}").sample(
                states, sample)
        for state in states:
            if board.won(state):
                continue
            for di in range(5):
                msg = _compare(game, expert, board, state, di)
                if msg:
                    bad += 1
                    if bad <= 5:
                        print(f"  level {level}: {msg}")
        if verbose:
            print(f"  level {level}: {len(states)} of {total} states x 5 "
                  f"presses checked")
    return bad


def _selfcheck(verbose: bool = True) -> int:
    """Replay every plan through the real interpreter and check every label.

    Three separate claims, all of which have to hold for the corpus to be worth
    anything:

    1. the plan WINS on the interpreter, press for press (the model found it; the
       interpreter is what certifies it);
    2. every press of the plan is in its own optimal set, and every press in that
       set really does finish in the same number of moves (re-derived from the
       field, not from the plan);
    3. no plan overruns the adapter's per-level press budget.
    """
    solver, game, expert = _new()
    bad = 0
    for level in _levels(solver, game):
        game.set_level(level)
        eng = game._engine
        board, start = expert.read(eng)
        field = expert.field(board)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  level {level}: no plan")
            bad += 1
            continue

        state = start
        for i, press in enumerate(plan):
            rest = field.get(state)
            best = [DIRS[di] for di, _ in field.optimal(state)]
            if plan.optsets[i] != best:
                print(f"  level {level} step {i}: label {plan.optsets[i]} "
                      f"!= re-derived {best}")
                bad += 1
            if press not in best:
                print(f"  level {level} step {i}: took {press}, not in {best}")
                bad += 1
            for alt in best:
                nxt, win = board.step(state, DIRS.index(alt))
                cost = 1 if win else (
                    None if field.get(nxt) is None else field.get(nxt) + 1)
                if cost != rest:
                    print(f"  level {level} step {i}: {alt} labelled optimal "
                          f"but costs {cost}, not {rest}")
                    bad += 1
            state = board.step(state, DIRS.index(press))[0]

        game.set_level(level)
        for press in plan:
            eng.step(press)
        if not eng.check_win():
            print(f"  level {level}: the plan does not win on the interpreter")
            bad += 1
        if len(plan) >= game._max_steps:
            print(f"  level {level}: {len(plan)} presses over the "
                  f"{game._max_steps} budget")
            bad += 1
        if verbose:
            print(f"  level {level}: {len(plan)} presses, WIN certified, "
                  f"{sum(len(s) for s in plan.optsets)} labels checked")
    return bad


def _recovery(trials: int = 60, verbose: bool = True) -> int:
    """Re-plan from perturbed mid-episode boards.

    ``recovery_mode = "reset"`` plus ``epsilon`` means the recorder asks for a
    plan at three kinds of state: the level start, whatever an exploration prefix
    left behind (which a RESET has already returned to the start), and whatever
    an epsilon detour landed in. Only the third can be anywhere, so that is what
    this walks: run a few plan presses, take a random press, and require the
    expert to answer -- either with a plan the interpreter then wins with, or
    with an honest None for a board that really is lost.

    A None here is not a failure. Throwing your last box down the wrong ray
    genuinely loses a level, and the field saying so is exactly what keeps
    `record_level` from recording a detour it cannot come back from.
    """
    solver, game, expert = _new()
    ok = dead = bad = 0
    for level in _levels(solver, game):
        rng = random.Random(f"wizard_school:recovery:{level}")
        game.set_level(level)
        plan = expert.plan(game._engine, level)
        for _ in range(trials):
            game.set_level(level)
            eng = game._engine
            for press in plan[:rng.randrange(0, len(plan))]:
                eng.step(press)
            for _ in range(rng.randrange(1, 4)):
                eng.step(DIRS[rng.randrange(5)])
            if eng.check_win():
                ok += 1
                continue
            got = expert.plan(eng, level)
            if got is None:
                dead += 1
                continue
            for press in got:
                eng.step(press)
            if eng.check_win():
                ok += 1
            else:
                print(f"  level {level}: re-plan did not win")
                bad += 1
    if verbose:
        print(f"  {ok} re-planned to a WIN, {dead} correctly reported lost, "
              f"{bad} failures")
    return bad


def _engine(seeds: int = 6, verbose: bool = True) -> int:
    """Drive the REAL adapter -- rotation, remap and all -- through each plan.

    `_selfcheck` steps ``eng`` directly, which skips the whole rotation contract.
    This one presses the SCREEN action `record_level` would record and asserts
    `GameState.WIN`, which is what catches a `screen_action` wired backwards (the
    bug this family was factored out over).
    """
    solver = WizardSchoolSolver()
    bad = 0
    plans = {}
    for seed in range(seeds):
        game = solver.make_game(seed)
        expert = WizardSchoolExpert(game, node_cap=solver.node_cap)
        for level in _levels(solver, game):
            game.set_level(level)
            plan = expert.plan(game._engine, level)
            plans.setdefault(level, list(plan))
            if list(plan) != plans[level]:
                print(f"  seed {seed} level {level}: plan differs between seeds")
                bad += 1
            k = (game._rotation_k, game._hflip, game._vflip)
            for press in plan:
                game.perform_action(ActionInput(
                    id=screen_action(press, *k, solver.remap_actions)))
            if game._state != GameState.WIN:
                print(f"  seed {seed} level {level} {k}: state "
                      f"{game._state}, not WIN")
                bad += 1
    if verbose:
        print(f"  {seeds} seeds x {len(plans)} levels driven through the adapter")
    return bad


def _audit(verbose: bool = True) -> int:
    """Whole-frame distinctness of every legal cell composition, per level.

    Compares WHOLE frames of a board FILLED with each composition rather than
    cropping one cell out by arithmetic -- the render is upscaled by a
    non-integer factor here (64 / 7, 64 / 9), so a cell crop lands off the sprite
    grid and would compare the wrong pixels. Two compositions that render
    identically are two boards an agent cannot tell apart; before the sprite fix
    in `games/ps:wizard_school`, "box on a target" and "box on floor" were such a
    pair, and they are the win condition.
    """
    from adapters.puzzlescript_adapter import _render_frame

    solver = WizardSchoolSolver()
    game = solver.make_game(0)
    N = game._game.obj_name_to_idx
    comps = [(), ("plainwall",), ("target",), ("boxidle",), ("boxcarried",),
             ("boxsummoned",), ("playeridle",), ("playercarrying",),
             ("boxidle", "target"), ("boxcarried", "target"),
             ("boxsummoned", "target"), ("playeridle", "target"),
             ("playercarrying", "target")]

    def fill(level, comp):
        game.set_level(level)
        eng = game._engine
        for r in range(1, eng.height - 1):
            for c in range(1, eng.width - 1):
                cell = eng.grid[r][c]
                cell.clear()
                cell.add(N["background"])
                for name in comp:
                    cell.add(N[name])
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        return np.asarray(game._present_frame(_render_frame(eng, game._game)))

    bad = 0
    for level in range(game.n_levels):
        frames = [fill(level, comp) for comp in comps]
        here = 0
        for i in range(len(comps)):
            for j in range(i + 1, len(comps)):
                if np.array_equal(frames[i], frames[j]):
                    print(f"  level {level}: {comps[i] or ('background',)} and "
                          f"{comps[j]} render identically")
                    here += 1
        bad += here
        if verbose:
            print(f"  level {level}: {len(comps)} compositions, "
                  f"{'all distinct' if not here else f'{here} collision(s)'}")
    return bad


def _symmetry(seeds: int = 200, walk_presses: int = 14,
              verbose: bool = True) -> int:
    """The frames recorded at one presentation are the transform of the frames
    recorded unaugmented -- for the plan AND for a blind random walk.

    This game takes rotation only (it is not in `PuzzleScriptAdapter._FLIP_GAMES`),
    and the adapter forward-remaps directional input, so a generator planning in
    engine space has to emit `screen_action`'s inverse remap. If that were
    backwards the plans would still be recorded, just against the wrong buttons;
    this is the check that would notice.
    """
    solver = WizardSchoolSolver()
    game0 = solver.make_game(0)
    expert0 = WizardSchoolExpert(game0, node_cap=solver.node_cap)
    plans = {}
    for level in _levels(solver, game0):
        game0.set_level(level)
        plans[level] = list(expert0.plan(game0._engine, level))

    def drive(game, level, presses):
        game.set_level(level)
        frames = [np.asarray(game._current_frame)]
        for act in presses:
            fd = game.perform_action(ActionInput(id=act))
            frames.append(np.asarray(
                fd.frame[-1] if fd.frame else game._current_frame))
        return frames

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    ref, seen, bad = {}, set(), 0
    for seed in range(seeds):
        game = solver.make_game(seed)
        for level in _levels(solver, game):
            game.set_level(level)
            k = (game._rotation_k, game._hflip, game._vflip)
            seen.add(k)
            rng = random.Random(f"wizard_school:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (
                    ("plan", [screen_action(d, *k, solver.remap_actions)
                              for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *k) for a in walk])):
                frames = drive(game, level, presses)
                if tag == "plan" and game._state != GameState.WIN:
                    print(f"  seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[key] = frames
                    continue
                if any(not np.array_equal(transform(a, *k), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"  seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


def _replay(episodes: int = 5, verbose: bool = True) -> int:
    """Record episodes IN MEMORY, then replay the RECORDED action stream through
    a fresh adapter and require the frames to come back pixel-identical.

    This is the end-to-end version of the rotation contract, and the only check
    here that covers the WHOLE recorded stream rather than the plan: the
    exploration prefix, its closing RESET and the epsilon detours are all in it,
    and every one of them is replayed as the SCREEN action the corpus stores. If
    `screen_action` were inverted, or a recorded index were the engine direction
    rather than the button, the frames would diverge on the first press at a
    non-zero rotation.
    """
    solver = WizardSchoolSolver(rng=random.Random(5))
    bad = 0
    frames = 0
    for seed in range(episodes):
        ok, levels = solver.solve_episode(seed)
        if not ok:
            print(f"  seed {seed}: did not solve every level")
            bad += 1
        game = solver.make_game(seed)
        for lv in levels:
            game.set_level(lv["level_id"])
            obs = np.asarray(lv["observations"])
            if not np.array_equal(np.asarray(game._current_frame), obs[0]):
                print(f"  seed {seed} level {lv['level_id']}: frame 0 differs")
                bad += 1
            t = 1
            for act in lv["actions"][1:]:
                if act["index"] == 0:                    # the prefix's RESET
                    game.set_level(lv["level_id"])
                    got = np.asarray(game._current_frame)
                else:
                    fd = game.perform_action(ActionInput(
                        id=_ID_TO_GAMEACTION[act["index"]]))
                    got = np.asarray(fd.frame[-1] if fd.frame
                                     else game._current_frame)
                if not np.array_equal(got, obs[t]):
                    print(f"  seed {seed} level {lv['level_id']}: frame {t} "
                          f"differs on replay")
                    bad += 1
                    break
                t += 1
                frames += 1
    if verbose:
        print(f"  {episodes} episodes, {frames} frames replayed")
    return bad


def _labels(episodes: int = 12, verbose: bool = True) -> int:
    """Record a few episodes IN MEMORY and audit what the corpus would carry.

    Writes nothing: this is the "what fraction of steps actually trains
    anything" check that belongs beside a generator rather than after a sweep.
    Two claims:

    1. **every ``phase="expert"`` step carries an ``optimal`` set** -- a step
       without one stays in the context and contributes nothing to the loss, and
       nothing errors when it is missing;
    2. **every level contributes at least one expert step**. That is not
       automatic: the exploration prefix is a random walk that stops the moment
       it terminates the level, so a level shorter than the prefix gets won by it
       and taped entirely unlabelled. It is why `WizardSchoolSolver.explore_level`
       turns the prefix off for the two tutorial levels.

    It also re-records the same seeds a second time and requires the two runs to
    be identical, which is the determinism the disk plan cache and the shared
    `parallelize_generator` fan-out both assume.
    """
    def record(seed_offset):
        solver = WizardSchoolSolver(rng=random.Random(11))
        out = []
        for seed in range(seed_offset, seed_offset + episodes):
            ok, levels = solver.solve_episode(seed)
            out.append((ok, levels))
        return out

    runs = record(0)
    bad = 0
    steps = labelled = 0
    per_level_expert = {}
    for ok, levels in runs:
        if not ok:
            print("  a seed did not solve every level")
            bad += 1
        for lv in levels:
            nexp = 0
            for i, act in enumerate(lv["actions"]):
                phase = act.get("phase")
                steps += 1
                if act.get("optimal"):
                    labelled += 1
                if phase == "expert" or (phase is None and i):
                    nexp += 1
                    if not act.get("optimal"):
                        print(f"  level {lv['level_id']} step {i}: expert step "
                              f"with no optimal set")
                        bad += 1
            per_level_expert[lv["level_id"]] = (
                per_level_expert.get(lv["level_id"], 0) + nexp)
    for level, n in sorted(per_level_expert.items()):
        if not n:
            print(f"  level {level}: no expert step in {episodes} episodes")
            bad += 1
    again = record(0)
    if [a for a, _ in runs] != [a for a, _ in again] or \
            [l for _, l in runs] != [l for _, l in again]:
        print("  two runs of the same seeds are not identical")
        bad += 1
    if verbose:
        print(f"  {episodes} episodes, {steps} steps, {labelled} labelled")
        print("  expert steps per level: "
              + ", ".join(f"L{k}={v}" for k, v in sorted(per_level_expert.items())))
    return bad


_REPORTS = {
    "--plans": ("plans", _plans),
    "--proof": ("proof", _proof),
    "--fuzz": ("fuzz", _fuzz),
    "--enumerate": ("whole-space enumeration", _enumerate),
    "--selfcheck": ("selfcheck", _selfcheck),
    "--recovery": ("recovery", _recovery),
    "--engine": ("engine sweeps", _engine),
    "--audit": ("audit", _audit),
    "--symmetry": ("symmetry", _symmetry),
    "--labels": ("labels", _labels),
    "--replay": ("replay", _replay),
}


if __name__ == "__main__":
    for flag, (label, fn) in _REPORTS.items():
        if flag in sys.argv:
            if flag == "--enumerate":
                rest = sys.argv[sys.argv.index(flag) + 1:]
                nums, sample = [], 0
                for i, a in enumerate(rest):
                    if a == "--sample":
                        sample = int(rest[i + 1])
                        break
                    if a.isdigit():
                        nums.append(int(a))
                violations = fn(nums or None, sample)
            else:
                violations = fn()
            print(f"{label}: {violations} problem(s)")
            sys.exit(1 if violations else 0)
    sys.exit(WizardSchoolSolver.main())
