"""Generate Phase-1 training data for the PuzzleScript game ps:party_demon
("Party Demon" by Peter Smyth).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, the native model of the turn, the exact distance field, and the
optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_party_demon",
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
presented view, so replaying the recorded actions reproduces the recorded frames
exactly.

The game
--------
You are the party demon and your job is to make every guest miserable. The win
condition is ``No Happy``, and a guest is happy exactly when it can see a friend,
so the whole game is *anti*-sokoban: instead of gathering the crates onto targets
you have to scatter them until no two of them touch.

  * **Two guest families, and they are not friends with each other.** Blue
    (``Normal``: Crate / HorSadCrate / VertSadCrate / SadCrate) and red
    (``Sticky``: StickyCrate / …). The solitude rules run per family, so a blue
    crate sitting next to a red one is alone and therefore SAD. Only same-colour
    contact makes a guest happy.
  * **Mood is a pure function of the layout.** The five ``late`` solitude rules
    per family walk Crate -> HorSadCrate / VertSadCrate -> SadCrate, but their
    net effect is a one-liner: a guest is HAPPY iff it has a 4-neighbour of its
    own family. So the mood objects are never state -- they are the render of a
    predicate over positions -- and the model does not carry them.
  * **Pushing is a chain of at most four.** Four rules push 1, 2, 3 and 4
    ``Moveable``s ahead of the player; a fifth in line has no rule to move it, so
    a five-long chain is a dead press. Two ``cancel`` rules abort the whole turn
    when a 1- or 2-long chain has a wall behind it -- and, being cancels, they
    take the sticky drag below down with them, which a plain block would not.
  * **Red guests hold hands (the interesting rule).** Rule 28 (``rigid``) and
    rule 29 hand the push force from a moving sticky crate to its stationary
    sticky neighbours, in all four directions, so shoving one red guest drags the
    whole cluster sideways. **A cluster is therefore separated by TEARING it on
    the scenery**: drag it past a wall stub, the guest whose way is blocked stays
    behind, and the rest walk off without it. That is the entire red mechanic and
    every red level is built on it.
  * **The ``rigid`` half is not decoration.** Rule 28 matches three sticky crates
    in a line and links the first two into a rigid group: if either is blocked
    BOTH forces are cancelled -- which, when the blocked one is a perpendicular
    drag, propagates back through the chain and cancels the player's own push. So
    a 2-long red cluster TEARS against an obstacle while a 3-long one JAMS. That
    difference is rare enough to be invisible to a casual fuzz and load-bearing
    where it does fire: ``--rigid`` re-runs every transition in every level's
    enumerated ball against a model with the grouping removed, and **229 of
    3,500,844 transitions come out different**, spread over the seven levels that
    have red guests. Print it as its own counter (as here) or a clean fuzz says
    nothing about it.
  * **The disco ball is a happiness beam.** It sprays light down its row and its
    column (through the player, who is not a ``Moveable``) until a crate or a
    wall stops it, and any fully-sad guest the beam touches cheers up again. So a
    guest is only really sad once it is BOTH alone and out of the light -- and
    since the ball is itself pushable, the usual answer is to shove the ball
    until its cross points somewhere harmless. Horizontal and vertical light
    share one collision layer, so a cell holds one or the other and never both;
    `_Board.lights` replicates the four light rules in order because of it.

Not one shipped level uses the disco ball. Levels 5-12 (see the header of
data/puzzlescript_games/Party_Demon.txt) are authored and three of them do -- and
what the beam turns out to force is stronger than it looks: the crate that STOPS
a beam is itself lit by the cell in front of it, so a guest can never be shielded
behind another. Winning a disco board means getting every guest out of the ball's
cross entirely, or shoving the ball until its cross points at nothing.

Model + search
--------------
`_Board.step` re-implements one turn natively -- force assignment, the sticky
drag with its rigid groups, and the chain-blocking resolution -- and
``--selfcheck`` drives ~82,000 random presses through BOTH the interpreter and
the model, on all thirteen levels and on 700 random boards seeded with dense
sticky clusters, comparing the full board AND the per-guest mood after every one.
The model was WRONG twice before that fuzz was clean (a CANCELLED push still
drags its neighbours, and the rigid pair jams where a loose one tears), so it is
the guard that lets the search trust it -- and both bugs came from the random
boards, not from the levels.

The state is ``(player cell, blue cells, red cells, ball cells)``. The whole
reachable space is far too big for the level with eight crates -- level 2 passes
four million states without closing -- but the BALL around the start is not:
every shortest route from the start to a win stays inside the ball of radius
``d*``, and level 2's is 72k states. So `_Board.enumerate` runs a layer-by-layer
forward BFS out to ``d* + MARGIN`` (or until `STATE_CAP`, whichever comes first),
then a backward BFS from the wins inside it.

Truncating the ball would normally make the field an upper bound rather than a
distance, so every answer carries a **certificate**: a state ``s`` is trusted iff
``g(s) + d(s) <= R``, where ``g`` is its forward depth and ``R`` the enumerated
radius. That is exactly the condition for a shortest route out of ``s`` to fit
inside the enumerated ball, so where it holds the field is provably EXACT and
where it does not `distance` returns None rather than a guess. The start state is
always trusted (``0 + d* <= R``), and so is every state on every shortest route
from it, which is the whole cone the corpus records.

Optimal-action targets and recovery
-----------------------------------
`optimal_for` reads the LIVE engine state and returns every press that lowers the
exact distance -- true tie sets measured off the field, not a heuristic. This
game needs them badly: the player spends most of its presses WALKING around a
crate to get behind it, and a walk's interleaving is free, so labelling one
arbitrary order as the only right answer would train a coin flip the policy
cannot win.

``epsilon`` is non-zero even though the mechanic is irreversible (you can push a
crate but never pull it, and a third of the reachable space is unwinnable). It is
safe because `record_level` PROBES a detour before keeping it: an alternative is
taken only if the expert can still plan from where it lands, and this expert only
plans from states the certificate above marks trusted. So a detour is accepted
exactly when the field can label the recovery exactly, and refused otherwise --
the taken action is the mistake, ``optimal`` is the way back. The
explore-then-RESET prefix runs in front of that.

Rendering
---------
Two sprite fixes, in the game FILE (see the comment at the top of
data/puzzlescript_games/Party_Demon.txt): the Player got four holes punched in it
and the two beam sprites were extended to full bars. The Player was opaque
across its whole middle row and column -- precisely where both light sprites live
-- so ``player`` , ``player+horizontal beam`` and ``player+vertical beam`` were
one and the same picture and the beam vanished under the demon. ``--audit`` is
the check, and it compares WHOLE FRAMES rather than cell crops (`_render_frame`
upscales the board to fill 64x64 and letterboxes it, so an arithmetic cell crop
reads the wrong window).

The mood objects deliberately stay indistinguishable in threes:
``crate == horsadcrate == vertsadcrate`` (all blue) and the same for red. That is
the game's own design -- happy is blue/red, sad is green/dark-red -- and it costs
nothing, since which AXIS a happy guest is happy along is a function of the
layout the frame already shows.

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (0..3) and, since
``Party_Demon`` is in `PuzzleScriptAdapter._FLIP_GAMES`, an independent
horizontal and vertical flip -- 16 presentations of each level. The flips are
measured rather than argued: ``--symmetry`` rebuilds every level's LAYOUT under
all 8 transforms, reloads it into the interpreter, and replays the level's own
plan plus a seeded random walk with the presses transformed; the board must land
where the transform says after every press. That check is not a formality here --
the sticky drag resolves ties by the order the interpreter expands a rule's four
directions, which is the shape that hid ps:gobble_rush's chirality.

Usage (run from the repo root):
    python solvers/generate_party_demon_training.py --episodes 200 \
        --out data/training_multi_level/party_demon
    python solvers/generate_party_demon_training.py --selfcheck
    python solvers/generate_party_demon_training.py --plans
    python solvers/generate_party_demon_training.py --verify
    python solvers/generate_party_demon_training.py --audit
    python solvers/generate_party_demon_training.py --symmetry
    python solvers/generate_party_demon_training.py --rigid
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (                     # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from solvers.common.ps_astar import (                           # noqa: E402
    PSAStarSolver, PSExpert, restore, snapshot)

GAME_NAME = "Party_Demon"

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The action space the search branches on. ``noaction`` is declared in the
#: prelude, so ACTION is not even a key here; ``--selfcheck`` presses it anyway
#: and asserts it is a pure no-op rather than reading that off the prelude.
_DIRS = ("up", "down", "left", "right")

#: The push rules cover chains of 1, 2, 3 and 4 ``Moveable``s. A fifth in line
#: has no rule to give it a force, so it blocks the whole chain.
_MAX_CHAIN = 4

#: How far past the first win `_Board.enumerate` keeps expanding. Only the ball
#: of radius ``d*`` is needed for the shortest plans; the margin is what lets an
#: epsilon detour off that cone still be answered exactly (see the module
#: docstring's certificate).
_MARGIN = 8

#: Layer-boundary budget for the enumeration. Hit on the eight-crate levels,
#: where each extra layer is ~1.4x the last; every state below the radius it
#: stops at is still fully expanded, which is what keeps the certificate sound.
_STATE_CAP = 400_000


# ---------------------------------------------------------------------------
# The level, natively: one turn, the ball around the start, the distance field
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry plus the exact distance-to-win field over the
    ball around its start state.

    Cells are flat ``r * w + c`` ids; a state is
    ``(player, blue crates, red crates, disco balls)`` with the three crate sets
    as sorted tuples -- guests of one colour are interchangeable, so the sets are
    unordered and the tuple is canonical.

    `enumerate` is what everything else reads (see the module docstring for the
    radius and the trust certificate); `plan` is a descent of the field,
    `optimal` a lookup, and both answer from ANY trusted state, which is what
    lets the recording take real detours and label the recovery.
    """

    __slots__ = ("h", "w", "n", "walls", "start", "_nb", "_rc",
                 "states", "index", "succ", "dist", "depth", "radius",
                 "complete")

    def __init__(self, h: int, w: int, walls, start):
        self.h, self.w, self.n = h, w, h * w
        self.walls = frozenset(walls)
        self.start = start

        # ``_nb[d][cell]`` is the cell one step along d, or -1 when that is a
        # wall or off the board -- the two cases the rules treat alike (no rule
        # can match past the edge, and every shipped level is wall-bordered).
        self._nb = {}
        for d, (dr, dc) in _DELTA.items():
            table = [-1] * self.n
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and nr * w + nc not in self.walls:
                        table[r * w + c] = nr * w + nc
            self._nb[d] = table
        self._rc = [divmod(i, w) for i in range(self.n)]

        self.states: list | None = None
        self.index: dict = {}
        self.succ: list = []
        self.dist: list = []
        self.depth: list = []
        self.radius = 0
        self.complete = False

    # -- one turn -------------------------------------------------------------
    def _drag(self, stickies: set, moving: set, groups: dict) -> None:
        """Rules 28 and 29: hand the push force to the sticky crates touching a
        moving one.

            rule 28  rigid [ moving Sticky | stationary Sticky | stationary Sticky ]
            rule 29        [ moving Sticky | stationary Sticky | no Sticky ]

        Each is run to a fixpoint over its four directions, 28 fully before 29,
        and each direction does a forward scan and then -- only if the forward
        scan fired -- a reverse one. That is what the interpreter does
        (`_apply_single_rule_forces`), and it is not cosmetic: every match of the
        RIGID rule mints a fresh group id and re-tags the two forces it links, so
        the scan order decides which pairs end up sharing a fate. `--selfcheck`
        is what says this is right.
        """
        for rigid in (True, False):
            while True:
                fired_iter = False
                for e in _DIRS:
                    nb = self._nb[e]
                    for reverse in (False, True):
                        cands = sorted((c for c in moving if c in stickies),
                                       key=self._scan_key(e, reverse))
                        fired = False
                        for x in cands:
                            y = nb[x]
                            if y < 0 or y not in stickies or y in moving:
                                continue
                            z = nb[y]
                            if rigid:
                                if z < 0 or z not in stickies or z in moving:
                                    continue
                            elif z >= 0 and z in stickies:
                                continue
                            moving.add(y)
                            fired = True
                            if rigid:
                                gid = len(groups) + 1
                                groups[x] = groups[y] = gid
                        fired_iter |= fired
                        if not reverse and not fired:
                            break                 # nothing matched, no rescan
                if not fired_iter:
                    break

    def _scan_key(self, e: str, reverse: bool):
        rc = self._rc
        if e == "up":
            return (lambda i: (-rc[i][0], rc[i][1])) if not reverse else \
                   (lambda i: rc[i])
        if e == "left":
            return (lambda i: (rc[i][0], -rc[i][1])) if not reverse else \
                   (lambda i: rc[i])
        return (lambda i: rc[i]) if not reverse else \
               (lambda i: (-rc[i][0], -rc[i][1]))

    def step(self, state, d: str):
        """One press, natively. Returns the settled state (``state`` itself when
        the press changes nothing, which is what the interpreter does for a walk
        into a wall, a cancelled push and a jammed one alike)."""
        player, normals, stickies, balls = state
        nb = self._nb[d]
        t = nb[player]
        if t < 0:                                # wall or edge: no rule matches
            return state
        nset, sset = set(normals), set(stickies)
        moveables = nset | sset | set(balls)

        chain = []
        cur = t
        while cur in moveables and len(chain) < _MAX_CHAIN:
            chain.append(cur)
            cur = nb[cur]
            if cur < 0:
                break
        k = len(chain)
        if k in (1, 2) and cur < 0:
            return state                         # rules 26 / 27: cancel the turn

        moving = {player}
        moving.update(chain)
        groups: dict = {}
        if any(c in sset for c in chain):
            self._drag(sset, moving, groups)

        occupied = moveables | {player}
        blocked: set = set()

        def block(cell: int) -> None:
            blocked.add(cell)
            gid = groups.get(cell)
            if gid is not None:
                blocked.update(o for o, g in groups.items() if g == gid)

        changed = True
        while changed:
            changed = False
            for cell in moving:
                if cell in blocked:
                    continue
                dest = nb[cell]
                if dest < 0 or (dest in occupied and
                                (dest not in moving or dest in blocked)):
                    before = len(blocked)
                    block(cell)
                    changed |= len(blocked) != before

        def shift(cells):
            return tuple(sorted(nb[c] if c in moving and c not in blocked else c
                                for c in cells))

        np_ = player if player in blocked else nb[player]
        out = (np_, shift(normals), shift(stickies), shift(balls))
        return state if out == state else out

    # -- the win predicate ----------------------------------------------------
    def lights(self, state) -> tuple[set, set]:
        """``(horizontal, vertical)`` lit cells.

        The four beam rules are replayed IN ORDER because the two beam objects
        share one collision layer: a cell holds H or V but never both, so the
        later rule overwrites the earlier one and a vertical beam crossing a
        horizontal one wins the cell. (That is also why the authored disco levels
        carry exactly ONE ball -- with one ball the row beam and the column beam
        cannot meet, so the resolution never has to be chiral; see
        ``--symmetry``.)"""
        _, normals, stickies, balls = state
        blockers = set(normals) | set(stickies) | set(balls)
        light: dict = {}
        for kind, axis in (("H", ("left", "right")), ("V", ("up", "down"))):
            for b in balls:                                  # rules H1 / V1
                for e in axis:
                    n = self._nb[e][b]
                    if n >= 0 and n not in blockers:
                        light[n] = kind
        for kind, axis in (("H", ("left", "right")), ("V", ("up", "down"))):
            for cell in [c for c, k in light.items() if k == kind]:   # H2 / V2
                for e in axis:
                    cur = self._nb[e][cell]
                    while cur >= 0 and cur not in blockers:
                        light[cur] = kind
                        cur = self._nb[e][cur]
        return ({c for c, k in light.items() if k == "H"},
                {c for c, k in light.items() if k == "V"})

    def happy(self, state) -> set:
        """The guests that are still enjoying themselves -- ``Happy`` in the win
        condition, i.e. everything the demon has left to do."""
        _, normals, stickies, balls = state
        out = set()
        for fam in (set(normals), set(stickies)):
            for cell in fam:
                if any(self._nb[e][cell] in fam for e in _DIRS):
                    out.add(cell)
        if balls:
            hor, vert = self.lights(state)
            bset = set(balls)
            for fam in (normals, stickies):
                for cell in fam:
                    if cell in out:
                        continue
                    if any(self._nb[e][cell] in bset for e in _DIRS):
                        out.add(cell)
                    elif (self._nb["left"][cell] in hor
                          or self._nb["right"][cell] in hor
                          or self._nb["up"][cell] in vert
                          or self._nb["down"][cell] in vert):
                        out.add(cell)
        return out

    def won(self, state) -> bool:
        """``No Happy``."""
        return not self.happy(state)

    # -- the ball around the start, and the field over it ---------------------
    def enumerate(self) -> tuple[int, int, bool]:
        """Build the ball around the start, its successor table and the exact
        distance-to-win field over it. Returns ``(states, radius, complete)``.

        Layer by layer, so the radius is decided at a LAYER BOUNDARY: every state
        below ``radius`` is fully expanded, which is what makes the trust
        certificate in `distance` sound. Expansion stops at the first layer that
        is both past the first win by `_MARGIN` and, or, over `_STATE_CAP`;
        ``complete`` says the frontier emptied first, in which case the ball is
        the whole reachable space and every state in it is trusted.

        Winning states are terminal: the adapter ends the level there, so they
        are never expanded.
        """
        if self.states is not None:
            return len(self.states), self.radius, self.complete

        states = [self.start]
        index = {self.start: 0}
        depth = [0]
        succ: list = [None]
        frontier = [0]
        g = 0
        first_win = None
        while frontier:
            if first_win is not None and (g > first_win + _MARGIN
                                          or len(states) > _STATE_CAP):
                break                            # radius decided at a boundary
            nxt = []
            for j in frontier:
                s = states[j]
                if self.won(s):
                    if first_win is None:
                        first_win = g
                    continue                     # terminal: the level ends
                row = []
                for d in _DIRS:
                    t = self.step(s, d)
                    i = index.get(t)
                    if i is None:
                        i = len(states)
                        index[t] = i
                        states.append(t)
                        depth.append(g + 1)
                        succ.append(None)
                        nxt.append(i)
                    row.append(i)
                succ[j] = tuple(row)
            frontier = nxt
            g += 1
        self.complete = not frontier
        self.radius = (max(depth) if self.complete else g)

        pred: list[list[int]] = [[] for _ in states]
        for src, row in enumerate(succ):
            if row is not None:
                for dst in row:
                    pred[dst].append(src)

        dist = [-1] * len(states)
        queue = deque()
        for j, s in enumerate(states):
            if self.won(s):
                dist[j] = 0
                queue.append(j)
        while queue:
            j = queue.popleft()
            for src in pred[j]:
                if dist[src] < 0:
                    dist[src] = dist[j] + 1
                    queue.append(src)

        self.states, self.index, self.succ = states, index, succ
        self.dist, self.depth = dist, depth
        return len(states), self.radius, self.complete

    # -- reading the field ----------------------------------------------------
    def _trusted(self, j: int) -> bool:
        """``g(s) + d(s) <= radius``: every shortest route out of ``s`` fits
        inside the enumerated ball, so the field's value for it is EXACT rather
        than an upper bound. See the module docstring for the proof; ``complete``
        (the frontier emptied on its own) makes it vacuous."""
        return self.dist[j] >= 0 and (
            self.complete or self.depth[j] + self.dist[j] <= self.radius)

    def distance(self, state) -> int | None:
        """Presses to a win from ``state``, or None when the field cannot answer
        EXACTLY -- the state is outside the enumerated ball, cannot win inside
        it, or fails the certificate above. None is never a guess."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or not self._trusted(j):
            return None
        return self.dist[j]

    def plan(self, state) -> list | None:
        """A SHORTEST press sequence from ``state`` to a win, or None.

        A descent of the field taking the first tied direction in ``_DIRS``
        order, so the plan is a pure function of the board and does not vary with
        the interpreter's hash seed."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or not self._trusted(j):
            return None
        out = []
        while self.dist[j] > 0:
            for k, nxt in enumerate(self.succ[j]):
                if self.dist[nxt] == self.dist[j] - 1:
                    out.append(_DIRS[k])
                    j = nxt
                    break
            else:                                    # pragma: no cover
                raise AssertionError("distance field has no descent")
        return out

    def optimal(self, state) -> list:
        """Every press that keeps the game on a SHORTEST route to the win.

        Exact wherever `distance` answers: a successor of a trusted state that
        lies one press closer is itself trusted (its ``g + d`` cannot exceed its
        parent's), so no optimal press can be hiding outside the ball. A press
        that leaves the board untouched maps the state to itself, whose distance
        is unchanged rather than one less, so dead presses drop out for free."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or not self._trusted(j) or self.dist[j] <= 0:
            return []
        here = self.dist[j]
        return [_DIRS[k] for k, nxt in enumerate(self.succ[j])
                if self.dist[nxt] == here - 1]


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class PartyDemonExpert(PSExpert):
    """Exact planner over the enumerated ball around each level's start (see
    `_Board`).

    `PSExpert` supplies the plan memo, the restore discipline and the level
    scoping; `_search` swaps in the field descent, so the interpreter is only
    ever stepped by the recorder -- which is also what verifies every plan, since
    a level is kept only when the engine reports WIN.
    """

    directions = list(_DIRS)

    #: `_key` is the dynamic objects only -- player, guests, balls -- which is
    #: canonical only WITHIN a level (walls differ between levels, and the mood
    #: objects that complete the engine grid are a function of the rest).
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        self.normal_ids = set(g.resolve_object_name("normal"))
        self.sticky_ids = set(g.resolve_object_name("sticky"))
        self.ball_ids = set(g.resolve_object_name("discoball"))
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.player_ids = set(self.game._engine._player_indices)
        self._boards: dict[int | None, _Board] = {}
        self._cur: _Board | None = None

    # -- reading the engine ---------------------------------------------------
    def read(self, eng):
        """The engine grid as a `_Board` state.

        The four mood variants of each family collapse to one set: which one an
        engine cell holds is a function of the layout (the ``late`` solitude
        rules), so carrying it would only alias one state into four."""
        w = len(eng.grid[0])
        player = -1
        normals, stickies, balls = [], [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.player_ids:
                    player = i
                if cell & self.normal_ids:
                    normals.append(i)
                elif cell & self.sticky_ids:
                    stickies.append(i)
                elif cell & self.ball_ids:
                    balls.append(i)
        if player < 0:                                       # pragma: no cover
            raise AssertionError("no Player on the board")
        return (player, tuple(normals), tuple(stickies), tuple(balls))

    def board(self, eng, level: int | None) -> _Board:
        """The level's `_Board`, built (and enumerated) once and cached.

        Built from the level's START state, which is where every seed's plan
        begins and what the ball is centred on -- so it must be called with the
        engine freshly at ``set_level``, which `prepare_expert` guarantees."""
        board = self._boards.get(level)
        if board is not None:
            return board
        h, w = len(eng.grid), len(eng.grid[0])
        walls = {r * w + c
                 for r, row in enumerate(eng.grid)
                 for c, cell in enumerate(row) if cell & self.wall_ids}
        board = _Board(h, w, walls, self.read(eng))
        board.enumerate()
        self._boards[level] = board
        return board

    # -- planning -------------------------------------------------------------
    def _key(self, eng):
        return self.read(eng)

    def plan(self, eng, level: int | None = None) -> list | None:
        self._cur = self._boards.get(level) or self.board(eng, level)
        return super().plan(eng, level)

    def heuristic(self, eng) -> int:
        """The EXACT remaining press count (the base `_astar` is never used here,
        but the contract is that this is 0 at a win and admissible, and the field
        is both)."""
        d = self._cur.distance(self.read(eng))
        return 0 if d is None else d

    def _search(self, eng) -> list | None:
        return self._cur.plan(self.read(eng))

    def optimal_dirs(self, eng, level: int | None) -> list:
        """Every press on a shortest route, from the engine's CURRENT state."""
        return self._boards[level].optimal(self.read(eng))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class PartyDemonSolver(PSAStarSolver):
    game_id = "puzzlescript_party_demon"
    game_name = GAME_NAME
    expert_cls = PartyDemonExpert

    #: Unused -- `PartyDemonExpert._search` never calls the base A* -- but left
    #: at the family default so a future subclass that does is not silently
    #: starved.
    node_cap = 2_000_000
    weight = 1
    #: The adapter GAME_OVERs at 200 actions per level; the longest plan here is
    #: well under a third of that, leaving the rest to the prefix and detours.
    max_steps = 200

    #: Non-zero despite an irreversible mechanic, which the `PSAStarSolver`
    #: default (0) is written for. `record_level` probes a detour before keeping
    #: it -- it is taken only if the expert can still plan from where it lands --
    #: and this expert plans only from states the field can answer EXACTLY. So an
    #: accepted detour is by construction one whose recovery is labellable, and
    #: one that would strand the level (or land past the enumerated ball) is
    #: refused and the expert simply plays on. The RESET prefix runs in front.
    epsilon = 0.1

    def prepare_expert(self, game, expert) -> None:
        """Enumerate every level's ball up front.

        Not just front-loading: `_Board` is centred on the state it is BUILT
        from, so it has to be built at the level start rather than at whatever
        state the first query happens to arrive with."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.board(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route to the win, read
        off the LIVE engine state.

        `record_level` asks for this before executing the press, so the engine is
        still at the state being labelled -- which is what makes the label right
        after an epsilon detour too, where the state is off the plan entirely and
        replaying the plan from the level start would label the wrong states.

        Falls back to the press about to be taken if the field has nothing to say
        -- no expert step may ship unlabelled."""
        best = expert.optimal_dirs(expert.game._engine, level)
        if best:
            return best
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with every level's ball enumerated."""
    solver = PartyDemonSolver()
    game, expert, _solvable = solver._ensure(0)
    return solver, game, expert


def _engine_state(expert, eng):
    """``(state, happy cells)`` read off the interpreter -- the second is what
    `_Board.happy` is checked against, since the mood objects are the only place
    the interpreter records the win predicate."""
    w = len(eng.grid[0])
    happy_ids = set(expert.g.resolve_object_name("happy"))
    happy = {r * w + c
             for r, row in enumerate(eng.grid)
             for c, cell in enumerate(row) if cell & happy_ids}
    return expert.read(eng), happy


def _random_map(rng, h, w, n_normal, n_sticky, n_ball, wallp):
    """A wall-bordered random board as PuzzleScript level ASCII."""
    rows = [["m" if (r in (0, h - 1) or c in (0, w - 1)
                     or rng.random() < wallp) else "."
             for c in range(w)] for r in range(h)]
    free = [(r, c) for r in range(h) for c in range(w) if rows[r][c] == "."]
    need = 1 + n_normal + n_sticky + n_ball
    if len(free) < need:
        return None
    for (r, c), ch in zip(rng.sample(free, need),
                          ["p"] + ["*"] * n_normal + ["$"] * n_sticky
                          + ["d"] * n_ball):
        rows[r][c] = ch
    return [("".join(row)) for row in rows]


def selfcheck(boards: int = 700, steps: int = 90, verbose: bool = True) -> int:
    """Drive random presses through BOTH the interpreter and `_Board.step`,
    comparing the whole board AND the per-guest mood after every one. This is the
    guard that lets the planner trust the native model.

    Two populations, because they fail differently:

      * the SHIPPED levels, walked from their own start states -- the boards the
        corpus is actually recorded on;
      * RANDOM boards, half of them seeded with dense sticky clusters, which is
        what the drag rules need: a cluster has to be big enough to contain a
        three-in-a-line (the ``rigid`` pattern) and to run into scenery from
        several sides at once. Both of the model's two real bugs -- a cancelled
        push still drags its neighbours, and a rigid pair jams where a loose one
        tears -- were found here and by nothing else.

    ``action`` is in the press alphabet even though ``noaction`` is declared and
    the search excludes it: that it is a pure no-op should be measured against
    the interpreter, not read off a prelude.

    Returns the number of mismatches."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = PartyDemonExpert(game)
    alphabet = _DIRS + ("action",)
    eng, g = game._engine, game._game
    total = presses = 0

    for level in range(game.n_levels):
        rng = random.Random(f"party_demon:selfcheck:{level}")
        bad = 0
        for _ in range(20):
            game.set_level(level)
            board = expert.board(eng, level)
            state, _ = _engine_state(expert, eng)
            for _ in range(steps):
                d = rng.choice(alphabet)
                want = state if d == "action" else board.step(state, d)
                eng.step(d)
                presses += 1
                got, happy = _engine_state(expert, eng)
                if want != got or board.happy(want) != happy \
                        or board.won(want) != eng.check_win():
                    bad += 1
                    break
                state = want
                if eng.check_win():
                    break
        total += bad
        if verbose:
            print(f"  L{level}: {'OK' if not bad else f'{bad} MISMATCHES'}")

    rng = random.Random("party_demon:selfcheck:random")
    bad = 0
    for _ in range(boards):
        h, w = rng.randint(5, 10), rng.randint(5, 10)
        dense = rng.random() < 0.5
        lines = _random_map(rng, h, w,
                            rng.randint(0, 10 if dense else 4),
                            rng.randint(0, 12 if dense else 4),
                            rng.choice([0, 0, 1, 1, 2]),
                            rng.choice([0.0, 0.08, 0.15, 0.22]))
        if lines is None:
            continue
        layout = g._build_level(lines)
        if layout is None:
            continue
        eng.load_level(layout)
        walls = {r * w + c
                 for r, row in enumerate(eng.grid)
                 for c, cell in enumerate(row) if cell & expert.wall_ids}
        state, happy = _engine_state(expert, eng)
        board = _Board(h, w, walls, state)
        if board.happy(state) != happy:
            bad += 1
            continue
        for _ in range(steps):
            d = rng.choice(alphabet)
            want = state if d == "action" else board.step(state, d)
            eng.step(d)
            presses += 1
            got, happy = _engine_state(expert, eng)
            if want != got or board.happy(want) != happy:
                bad += 1
                break
            state = want
    total += bad
    if verbose:
        print(f"  random boards: {'OK' if not bad else f'{bad} MISMATCHES'}")
        print(f"  {presses} presses compared against the interpreter")
    return total


def _walk(board: _Board, plan) -> list:
    """The states ``plan`` passes through, starting at the level start."""
    out, state = [], board.start
    for d in plan:
        out.append(state)
        state = board.step(state, d)
    return out


def _report() -> int:
    """Print each level's ball, its shortest plan and the engine's verdict -- the
    quick "is this game still fully solved" check."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    total_presses = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        n_states, radius, complete = board.enumerate()
        plan = board.plan(board.start)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            bad += 1
            continue
        for d in plan:
            eng.step(d)
        won = eng.check_win()                   # read BEFORE anything reloads
        bad += 0 if won else 1
        ties = sum(len(board.optimal(s)) for s in _walk(board, plan))
        total_presses += len(plan)
        trusted = sum(1 for j in range(len(board.states)) if board._trusted(j))
        print(f"  L{level}: {board.h:2d}x{board.w:<2d} {len(plan):3d} presses  "
              f"win={won}  ball r={radius}{'' if not complete else ' (whole space)'} "
              f"{n_states:6d} states, {trusted:6d} trusted  "
              f"{ties / len(plan):.2f} optimal presses/step")
    print(f"  total: {total_presses} presses over {game.n_levels} levels")
    return 1 if bad else 0


def _verify() -> int:
    """Double-entry check of the distance field and of every tie set it ships.

    `_Board.enumerate` answers everything from ONE pass -- a forward sweep that
    builds the successor table and a reverse sweep over the predecessor map it
    inverts to. A bug in that inversion would produce a self-consistent field, a
    plan that still wins, and tie sets that are quietly wrong; and the training
    labels ARE those tie sets, so "it wins" is not enough of a check. Three
    passes, which fail differently:

      * **Bellman fixpoint.** Re-derive every distance by relaxing
        ``d(s) = 1 + min d(succ)`` to a fixpoint over the successor table alone
        -- no predecessor map, no BFS ordering, nothing shared with the reverse
        sweep -- and require it to agree everywhere, plus ``d == 0`` exactly at
        the winning states.
      * **The certificate.** For every TRUSTED state, re-derive its distance by
        an independent BFS that is allowed to leave the ball (bounded by the
        value being checked), so a truncation the certificate wrongly blessed
        would show up as a shorter route the field never saw.
      * **Executable labels.** Walk each level's plan on the INTERPRETER and, at
        every step, actually press each direction the label calls optimal: the
        board the engine lands on must be the native successor, and its field
        distance must be exactly one less. That takes the labels out of the model
        and puts them through the real thing.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)

        # -- pass 1: Bellman fixpoint over the successor table ----------------
        check = [0 if board.won(s) else -1 for s in board.states]
        changed = True
        while changed:
            changed = False
            for j, row in enumerate(board.succ):
                if row is None:
                    continue
                best = min((check[n] for n in row if check[n] >= 0), default=-1)
                if best >= 0 and (check[j] < 0 or best + 1 < check[j]):
                    check[j] = best + 1
                    changed = True
        if check != board.dist:
            bad += 1
            print(f"  L{level}: the field disagrees with its Bellman fixpoint "
                  f"at {sum(1 for a, b in zip(check, board.dist) if a != b)} "
                  f"states")
        if any((d == 0) != board.won(s)
               for d, s in zip(board.dist, board.states)):
            bad += 1
            print(f"  L{level}: distance 0 is not exactly the winning states")

        # -- pass 2: the two structural facts the certificate rests on --------
        # ``g(s) + d(s) <= radius`` is only a proof of exactness if (a) every
        # state below the radius really was expanded, so no edge of a shortest
        # route is missing, and (b) ``depth`` really is the forward BFS distance,
        # so ``g`` is not an over-estimate that lets an untrustworthy state pass.
        # Both are checked here directly rather than assumed from the loop that
        # built them.
        if board.depth[0] != 0:
            bad += 1
            print(f"  L{level}: the start is not at depth 0")
        holes = sum(1 for j, s in enumerate(board.states)
                    if board.depth[j] < board.radius
                    and board.succ[j] is None and not board.won(s))
        if holes:
            bad += 1
            print(f"  L{level}: {holes} states below the radius were never "
                  f"expanded -- the certificate is unsound")
        skew = sum(1 for j, row in enumerate(board.succ) if row is not None
                   for i in row if board.depth[i] > board.depth[j] + 1)
        if skew:
            bad += 1
            print(f"  L{level}: {skew} edges break the BFS layering")

        # ... plus an independent re-search of a sample of short claims, which is
        # the only pass that never touches `succ` at all.
        rng = random.Random(f"party_demon:verify:{level}")
        sample = [j for j in range(len(board.states))
                  if board._trusted(j) and 0 < board.dist[j] <= 6]
        if len(sample) > 30:
            sample = rng.sample(sample, 30)
        for j in sample:
            claim = board.dist[j]
            seen = {board.states[j]}
            frontier = [board.states[j]]
            found = None
            for g in range(claim + 1):
                if any(board.won(s) for s in frontier):
                    found = g
                    break
                nxt = []
                for s in frontier:
                    for d in _DIRS:
                        t = board.step(s, d)
                        if t not in seen:
                            seen.add(t)
                            nxt.append(t)
                frontier = nxt
            if found != claim:
                bad += 1
                print(f"  L{level}: a trusted state claims d={claim} but a "
                      f"fresh BFS says {found}")
                break

        # -- pass 3: every shipped label, pressed on the interpreter ----------
        plan = board.plan(board.start)
        state = board.start
        for i, press in enumerate(plan):
            here = board.distance(state)
            best = board.optimal(state)
            if press not in best:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            before = snapshot(eng)
            for alt in best:
                eng.step(alt)
                landed, _ = _engine_state(expert, eng)
                there = board.distance(landed)
                if landed != board.step(state, alt) or there != here - 1:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but the "
                          f"interpreter does not land one press closer")
                restore(eng, before)
            eng.step(press)
            state = board.step(state, press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        print(f"  L{level}: {len(board.states):6d} states relaxed, "
              f"{len(sample):3d} certificates re-searched, "
              f"{len(plan):3d} plan steps, "
              f"{sum(len(board.optimal(s)) for s in _walk(board, plan))} labels "
              f"pressed on the interpreter -- "
              f"{'OK' if not bad else 'see above'}")
    print(f"verify: {bad} problems")
    return 1 if bad else 0


class _LooseBoard(_Board):
    """`_Board` with rule 28's RIGID grouping thrown away -- the drag still
    propagates, but a blocked crate no longer drags its partner's force down with
    it, so every red cluster tears where the real game jams. Only `_rigid` uses
    it, as the null model to difference against."""

    def _drag(self, stickies: set, moving: set, groups: dict) -> None:
        super()._drag(stickies, moving, {})


def _rigid() -> int:
    """Count the transitions the ``rigid`` keyword actually decides.

    A mechanic that fires on a fraction of a percent of transitions is one a
    clean fuzz says nothing about -- "the model agrees on 200k presses" is
    consistent with never having exercised it at all. So it gets its own coverage
    counter: replay EVERY transition of every level's enumerated ball through the
    real model and through `_LooseBoard`, and report where they differ. Anything
    that drops this to zero on the red levels has quietly deleted the mechanic.
    """
    _solver, game, expert = _levels()
    total = diff = 0
    for level in range(game.n_levels):
        board = expert._boards[level]
        loose = _LooseBoard(board.h, board.w, board.walls, board.start)
        n = d = 0
        for s in board.states:
            for x in _DIRS:
                n += 1
                d += board.step(s, x) != loose.step(s, x)
        total += n
        diff += d
        print(f"  L{level}: {d:4d} of {n:7d} transitions decided by the rigid "
              f"group{'' if d else '   (no red guests here)'}")
    print(f"rigid: {diff} of {total} transitions "
          f"({100.0 * diff / total:.4f}%)")
    return 0 if diff else 1


def _transform(k: int, mirror: bool):
    """``(cell_fn, dims_fn, direction_map)`` for one element of the 8-group:
    mirror left-right first, then turn clockwise ``k`` times.

    The direction map is DERIVED from the same linear part that moves the cells,
    so the two cannot drift apart."""
    def cell(rc, hw):
        r, c = rc
        h, w = hw
        if mirror:
            c = w - 1 - c
        for _ in range(k):
            r, c, h, w = c, h - 1 - r, w, h
        return (r, c)

    def dims(hw):
        h, w = hw
        return (w, h) if k % 2 else (h, w)

    dmap = {}
    for name, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[name] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    return cell, dims, dmap


def _replay_transformed(eng, expert, layout, presses) -> list[str]:
    """Replay ``presses`` on all seven non-identity turned/mirrored copies of
    ``layout`` and return a note per presentation that diverged.

    The transformed board is built from the LEVEL LAYOUT and reloaded, so the
    interpreter parses and resolves everything itself -- including the mood and
    beam rules, whose HORIZONTAL / VERTICAL qualifiers are exactly what a turn
    would break -- rather than being handed a grid this file transformed after
    the fact."""
    hw = (len(layout), len(layout[0]))
    eng.load_level(layout)
    ref = [expert.read(eng)]
    for d in presses:
        eng.step(d)
        ref.append(expert.read(eng))

    def cells(state, hw_, cell_fn, tw):
        player, normals, stickies, balls = state
        def m(i):
            r, c = divmod(i, hw_[1])
            tr, tc = cell_fn((r, c), hw_)
            return tr * tw + tc
        return (m(player), tuple(sorted(map(m, normals))),
                tuple(sorted(map(m, stickies))), tuple(sorted(map(m, balls))))

    notes = []
    for k, mirror in itertools.product(range(4), (False, True)):
        if (k, mirror) == (0, False):
            continue
        cell, dims, dmap = _transform(k, mirror)
        th, tw = dims(hw)
        turned = [[set() for _ in range(tw)] for _ in range(th)]
        for r, row in enumerate(layout):
            for c, objs in enumerate(row):
                tr, tc = cell((r, c), hw)
                turned[tr][tc] = set(objs)
        eng.load_level(turned)
        for i, d in enumerate(presses):
            eng.step(dmap[d])
            if expert.read(eng) != cells(ref[i + 1], hw, cell, tw):
                notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                break
    return notes


def _symmetry(walk: int = 300) -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an exact
    symmetry of the mechanic, which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation.

    The argument is the ESCAPE! one -- gravity-free, screen-relative input, a win
    condition (``No Happy``) that names no direction -- but two things here need
    measuring rather than arguing, which is why the check exists:

      * the sticky drag mints a rigid group per match and the matches are found
        in the interpreter's own per-direction scan order, the exact shape that
        hid ps:gobble_rush's chirality;
      * the mood and beam rules are DIRECTION-QUALIFIED (``late HORIZONTAL``,
        ``late VERTICAL``), and the two beam objects share a collision layer, so
        a turn swaps which of a crossing pair wins the cell.

    Both are checked twice over: each level's own PLAN, which is the sequence the
    corpus actually records, and a seeded RANDOM WALK, which presses into walls
    and jams the clusters in ways a shortest plan never does.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        layout = g.levels[level]
        rng = random.Random(f"party_demon:symmetry:{level}")
        runs = {"plan": board.plan(board.start),
                "walk": [rng.choice(_DIRS) for _ in range(walk)]}
        notes = []
        for kind, presses in runs.items():
            notes += [f"{kind}:{n}" for n in
                      _replay_transformed(eng, expert, layout, presses)]
        bad += len(notes)
        print(f"  L{level}: ({len(runs['plan'])} plan + {walk} random) presses "
              f"x 7 presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print("symmetry clean -- the flip augmentation is exact" if not bad
          else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION the game can show is distinguishable in the
    frame -- except the three the game deliberately paints alike -- at every cell
    size the levels render at.

    The pair this exists for is ``player`` against ``player + beam``: the Player
    shipped opaque across its whole middle row and column, which is exactly where
    both beam sprites live, so the demon standing in the disco light looked
    identical to the demon standing in the dark, and so did the two beam
    orientations under it. See the header of
    data/puzzlescript_games/Party_Demon.txt.

    The three-way ties ``crate == horsadcrate == vertsadcrate`` (and the red
    equivalents) are EXPECTED and asserted as such: the game paints happy blue,
    happy red, sad green and sad dark-red, and which axis a happy guest is happy
    along is a function of a layout the frame already shows.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    expected = {frozenset({("crate",), ("horsadcrate",), ("vertsadcrate",)}),
                frozenset({("stickycrate",), ("horsadstickycrate",),
                           ("vertsadstickycrate",)})}

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    comps = [(), ("wallmiddle",), ("player",),
             ("player", "discolighthorizontal"),
             ("player", "discolightvertical"),
             ("discolighthorizontal",), ("discolightvertical",),
             ("discoball",),
             ("crate",), ("horsadcrate",), ("vertsadcrate",), ("sadcrate",),
             ("stickycrate",), ("horsadstickycrate",),
             ("vertsadstickycrate",), ("sadstickycrate",)]
    clashes_total = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for objs in comps:
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[objs] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])
                   and not any({a, b} <= grp for grp in expected)]
        clashes_total += len(clashes)
        name = lambda t: "+".join(t) if t else "floor"           # noqa: E731
        print(f"{h}x{w} board (level{'s' if len(levels) > 1 else ''} "
              f"{', '.join(str(x) for x in levels)}), "
              f"cell {max(1, min(64 // h, 64 // w))}px")
        for objs in (("player",), ("player", "discolighthorizontal"),
                     ("player", "discolightvertical"), ("sadcrate",),
                     ("sadstickycrate",), ("discoball",)):
            painted = int((shots[objs] != shots[()]).sum())
            over = ""
            if objs[0] == "player" and len(objs) > 1:
                over = (f", {int((shots[objs] != shots[('player',)]).sum()):4d} "
                        f"px of beam survive under the Player")
            print(f"  {name(objs):34s} {painted:4d} px differ from bare floor"
                  f"{over}")
        for a, b in clashes:
            print(f"  IDENTICAL: {name(a)} == {name(b)}")
    print("audit clean" if not clashes_total
          else f"AUDIT FAILED: {clashes_total} indistinguishable pairs")
    return 0 if not clashes_total else 1


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        bad = selfcheck()
        print(f"selfcheck: {bad} mismatches")
        sys.exit(1 if bad else _report())
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--rigid" in sys.argv:
        sys.exit(_rigid())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    sys.exit(PartyDemonSolver.main())
