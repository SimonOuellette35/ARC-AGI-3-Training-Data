"""Generate Phase-1 training data for the PuzzleScript game ps:slidings
("Slidings", Alain Brobecker).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, the BELIEF-space distance field the expert plans with, the reports
that prove every level is winnable, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_slidings",
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
A board of identical grey BALLS, one of which is green -- that one is yours. A
direction press SLIDES it until it hits something; ACTION5 hands control to the
next ball. Park the green ball on the dark blue ring and the level is won. The
catch is the strip of dotted tiles along the top edge: it is a MOVE COUNTER, and
it is exactly long enough for the intended solution.

The whole ruleset is nine lines, and the three that matter are::

    [> Player | Item]    -> [ Player | Item] sfx1
    [> Player | No Item] -> [ | > Player] again sfx0
    [> Player]           -> [ PlayerHasMoved ]

Everything below was MEASURED against the interpreter (``--selfcheck``
re-measures it, 99k transitions over the eleven levels, and asserts the model
reproduces every one), not read off the .txt:

* **A press is a whole SLIDE.** The second rule walks the ball one cell per
  iteration of the turn's own ``again`` loop, and `PuzzleScriptAdapter.step`
  drains that loop inside a single press, so one press moves the ball until an
  Item stops it. Passing OVER the target does not win -- the win condition is
  only read once the slide has settled.
* **A press into a wall is FREE.** The first rule cancels the movement without
  ever creating `PlayerHasMoved`, so the counter does not tick and the board is
  unchanged. The engine returns a no-op, not a wasted move.
* **Everything solid is an Item**, including the counter tiles themselves and
  the other balls: `Item = Player or Wall or BallA or BallB or BallC or Count or
  CountEnd`. A ball is a wall to every other ball, and putting one where another
  can stop against it is the whole puzzle. `Target` and `OverWall` are on their
  own collision layers and are NOT Items -- the ring never blocks anything.
* **The counter is a snake of `Count` tiles that a `CountEnd` eats one per
  move**, ``[Count | CountEnd] -> [CountEnd | Wall]``, and it is checked BEFORE
  it is decremented: ``[Wall | CountEnd | Wall]`` (no `Count` left beside the
  end marker) turns the player into a `DeadPlayer` and the next rule is
  ``restart``. So the budget is exactly the number of `Count` tiles the level
  ships -- 3, 4, 4, 5, 7, 7, 8, 5, 9, 9, 13 for levels 0-10, matching the
  eleven ``message`` screens' "N moves" to the move. Level 10's counter bends
  around three sides of the board; the rule is direction-agnostic, so it walks
  the corner without any special case.
* **Running out of moves is a RESTART, not a loss.** ``[DeadPlayer] -> restart``
  and the adapter reloads the level (`perform_action`'s ``_rule_restart``
  branch). There is no way to lose this game, only to spend a level's budget --
  which is what makes replanning from ANY state possible (see "Recovery").
* **ACTION5 is free.** It sets the `action` flag, not a direction, so none of
  the three movement rules match, no `PlayerHasMoved` appears and the counter
  does not tick. Cycling costs presses and nothing else.
* **ACTION5 rotates the LABELS, not the balls**::

      [Action Player] [BallA] [BallB] [BallC] -> [BallC] [Player] [BallA] [BallB]

  The cell holding BallA becomes the Player, the Player's cell becomes BallC,
  and so on. Nothing moves. Because the labels rotate by exactly one each time,
  control walks a fixed CYCLE through the balls.
* **The corner decoys are skipped.** Every level parks its unused BallB/BallC
  under an `OverWall` in a corner (the ``D`` / ``E`` legend characters), so the
  four-bracket rotation rule always has a BallA, a BallB and a BallC to match.
  The follow-up rule ``[Player OverWall] [BallA] [BallB] [BallC] -> [BallC
  OverWall] ... again`` rotates straight past any decoy that would have become
  the Player. `OverWall` renders as a plain wall on the layer above, so a decoy
  is invisible, unreachable and permanently solid -- the model treats those
  cells as scenery, which is exactly what the frame shows.

Eleven levels, 2-4 controllable balls each: 6x4, 6x5, 7x4, 7x5, 10x4, 9x9, 8x7,
7x6, 8x7, 8x8 and 8x5 (width x height). Level indices here and in every report
are 0-BASED; the game's own ``message`` screens are 1-based and are not levels.

Why the expert plans in BELIEF space
------------------------------------
**The rotation cycle is not on the screen.** BallA, BallB and BallC share one
sprite and one palette -- ``--audit`` measures it -- so a frame with three grey
balls and one green one says nothing about which grey ball ACTION5 will hand
over next. An expert that read `balla`/`ballb`/`ballc` off the engine grid would
press ACTION5 exactly the right number of times for a reason no viewer of the
64x64 frame could reconstruct, which is the privileged-solver failure mode: the
target would be unlearnable by construction.

So the expert reads only what the render shows -- ball cells, which one is
green, how many `Count` tiles are left, where the walls and the ring are -- and
treats the cycle as a HIDDEN VARIABLE it discovers by pressing ACTION5 and
looking. That is cheap, because the balls become the Player in cycle order, so
one press reveals one new member and the last is forced:

* 2 balls (levels 0-4, 9, 10): nothing to discover, the other ball is the
  other ball;
* 3 balls (levels 5, 6, 8): one press settles it, 2 candidate cycles;
* 4 balls (level 7): two presses settle it, 6 candidate cycles.

The planning state is therefore ``(where every ball is, the prefix of the cycle
discovered so far, which member of that prefix is green, moves left)``, and it
is FULLY observable -- the hidden variable has been folded into the knowledge
component. Presses are deterministic in it except one: ACTION5 taken at the end
of the known prefix, which reveals one of the not-yet-seen balls. `_Field` plans
over that graph exactly, taking the WORST case over what a discovery can reveal.
Worst case rather than an average is the honest reading: the cycle is a fixed
property of the level, not a fresh coin flip, so "expected over a uniform prior"
would be a fiction and the number `--plans` prints is a guarantee instead.

The spaces are small enough to enumerate outright, so there is no heuristic, no
weight and no node cap that could quietly bite: 26 to 73k belief states per
level, 200k altogether, in about three seconds for all eleven -- paid once at
startup and shared by every seed. What that buys:

* plans that are provably SHORTEST in presses (against the worst discovery),
* exact optimal-action SETS -- ``cost(press) == distance`` -- so every expert
  step is labelled with every press that ties, not with one arbitrary choice,
* a PROOF that a level is winnable rather than "the search gave up".

Levels 0-10 come out at 5, 6, 8, 7, 11, 13, 16, 19, 15, 11 and 17 presses
worst-case; on the cycle the levels actually ship, the expert wins them in 5, 6,
8, 7, 11, 13, 15, 14, 12, 11 and 17. Every one of them spends its move budget
to the last tile -- 3/3, 4/4, 4/4, 5/5, 7/7, 7/7, 8/8, 5/5, 9/9, 9/9, 13/13 --
so the author's move counts are tight and the slides are forced; the only thing
left to minimise is the cycling, which is what these numbers are.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: a fixed-length exploration
prefix opens the episode, then ONE RESET returns to the level start. On top of
it ``epsilon = 0.08`` -- roughly one press in twelve is a random legal
alternative, after which the expert re-plans from wherever it landed, so the
taken action is the mistake and ``optimal`` is the recovery. That is safe here
because the field answers from ANY state: the enumeration is closed under every
press (``--selfcheck`` asserts a random walk never leaves it), nothing on this
board is irreversible, and the one genuinely bad outcome -- spending the move
budget -- restarts the level for free rather than stranding it.

What a mistake COSTS is the thing to know about this game's recovery, and it is
why the rate is below the family's usual 0.10-0.12. Every counter is exactly as
long as its solution, so a detour is never a reordering: it is always fatal, and
the way back is to burn the remaining counter and take the free `restart` --
about 30 presses on level 10. The adapter GAME_OVERs at 200 presses per level,
so those recoveries compound, and each one is long enough to attract several
more detours. `SlidingsExpert.plan` therefore refuses to plan a win that will
not FIT in the presses left, which taper the detour rate off as the budget runs
down rather than letting it livelock. Measured over 40 seeds: 0.08 wins 40/40
(269 frames per episode against 141 with no detours at all), 0.12 wins 38/40,
0.20 wins 24/40; without the guard 0.12 won only 19/24.

The knowledge component survives all of it. A RESET or a `restart` puts the
balls back but does not un-see the cycle, and `SlidingsExpert.observe` keeps
tracking identities across both (only one ball ever moves per press, so the
before/after ball sets differ in at most one cell).

Augmentation
------------
ps: games are not `AugmentedGame`s -- `PuzzleScriptAdapter` owns the
augmentation. ps:slidings takes the family default: per-(seed, level) board
ROTATION plus the chrome recolor, with directional input forward-remapped
through `_remap_action_full`, so this generator plans in engine space and
records the SCREEN press via `ps_astar.screen_action`. It is deliberately NOT in
`_FLIP_GAMES`: the ball sprite is chiral (its highlight sits top-left) and so is
the wall, so a mirror would show art the game never draws. Eleven levels x 4
rotations = 44 presentations.

The render fix
--------------
``--audit`` compares every cell composition the game can show, at every board
size, and it found one that was drawn wrong: the shipped `Target` was a 3x3 ring
in the middle of a 5x5 cell, and both the ball and the player sprite are opaque
over that whole area. So a ball standing on the ring erased the goal from the
frame -- including on the WINNING frame, where the win marker is exactly the
thing a policy has to see. `data/puzzlescript_games/slidings.txt` now draws the
`Target` as the 5x5 OUTER frame instead, which lands on the four transparent
corners of the ball sprite: 180-1024 pixels of ring survive under a piece at
every board size, and no composition is indistinguishable from another any more
(`wall` == `countend` == a parked decoy remains, correctly -- all three are just
"solid" to the player, and the moves REMAINING are the dotted `Count` tiles,
which are distinct from all of them).

Checks
------
``--selfcheck`` fuzz the native model against the interpreter (with coverage
                counters, so a rule nothing exercised is visible), then fuzz the
                observation-only identity tracker against the real cycle
``--plans``     every level's belief space, its plan and the engine's verdict
``--verify``    double-entry: a Bellman fixpoint over the successor table, and
                every shipped label pressed on the interpreter
``--replay``    re-drive recorded episodes on a fresh adapter: frame-exact, WIN,
                and identical when recorded twice
``--audit``     the render audit described above

Generate with the conda python (see the repo's env split)::

    /home/simon/anaconda3/envs/ARC-AGI-3/bin/python \\
        solvers/generate_slidings_training.py --episodes 1600 \\
        --out data/training_multi_level/puzzlescript_slidings
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

from arcengine import ActionInput, GameState                         # noqa: E402
from adapters.puzzlescript_adapter import (PuzzleScriptAdapter,      # noqa: E402
                                           _render_frame)
from solvers.base_solver import _ID_TO_GAMEACTION                    # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,  # noqa: E402
                                     restore, snapshot)

GAME_NAME = "slidings"

#: The four slides plus the free cycle press. ACTION5 is not optional here: it
#: is half the game, and the only thing that changes which ball you are.
_DIRS = ("up", "down", "left", "right")
_PRESSES = _DIRS + ("action",)

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Objects the model reads. Deliberately NOT `balla` / `ballb` / `ballc` on
#: their own -- see "Why the expert plans in belief space": telling them apart is
#: the hidden variable, and a model that could would be a privileged one.
_BALL_NAMES = ("balla", "ballb", "ballc")
_SOLID_NAMES = ("wall", "count", "countend", "overwall")

_WIN = "win"
_INF = 1 << 30


# ---------------------------------------------------------------------------
# What one frame shows
# ---------------------------------------------------------------------------

class _Level:
    """The static geometry of one level plus its opening position, read off a
    grid the way a viewer of the frame would read it.

    ``grid`` is a ``list[list[set[int]]]`` -- either ``game._game.levels[i]`` or
    a live ``engine.grid``; the two are the same shape. Cells are flattened to
    ``r * w + c``.

    The only thing this class knows about a ball is WHERE it is and whether it
    is the green one. `balla` / `ballb` / `ballc` are collapsed into one
    anonymous class on the way in, and the decoys parked under an `OverWall` are
    dropped into the scenery, because that is what the render shows: the
    `OverWall` is on the layer above and paints a plain wall over them.
    """

    __slots__ = ("h", "w", "solid", "targets", "balls", "player", "n0",
                 "rays", "sig")

    def __init__(self, grid, ids: dict[str, int]):
        self.h = len(grid)
        self.w = len(grid[0])
        solid: set[int] = set()
        targets: set[int] = set()
        balls: list[int] = []
        player = None
        n0 = 0
        ball_ids = {ids[n] for n in _BALL_NAMES}
        solid_ids = {ids[n] for n in _SOLID_NAMES}
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                k = r * self.w + c
                if ids["target"] in cell:
                    targets.add(k)
                if cell & solid_ids:
                    solid.add(k)
                if ids["count"] in cell:
                    n0 += 1
                if ids["overwall"] in cell:
                    continue            # a decoy under it is scenery, not a ball
                if ids["player"] in cell:
                    player = k
                    balls.append(k)
                elif cell & ball_ids:
                    balls.append(k)
        self.solid = frozenset(solid)
        self.targets = frozenset(targets)
        self.balls = tuple(sorted(balls))
        self.player = player
        self.n0 = n0
        self.sig = (self.h, self.w, self.solid, self.targets)
        self.rays = self._rays()

    def _rays(self) -> dict:
        """``rays[d][cell]`` -- the cells a slide would pass through in order,
        stopping at the board edge or at the first solid cell.

        Precomputed because the scenery never changes: `Count` becomes
        `CountEnd` becomes `Wall` as the counter is eaten, and all three are
        Items, so the set of permanently-solid cells is invariant. A slide is
        then "walk this list until a ball is in the way"."""
        out = {}
        for d, (dr, dc) in _DELTA.items():
            table = []
            for k in range(self.h * self.w):
                r, c = divmod(k, self.w)
                ray = []
                while True:
                    r, c = r + dr, c + dc
                    if not (0 <= r < self.h and 0 <= c < self.w):
                        break
                    j = r * self.w + c
                    if j in self.solid:
                        break
                    ray.append(j)
                table.append(tuple(ray))
            out[d] = table
        return out

    def slide(self, pos: int, d: str, occupied) -> "int | None":
        """Where a ball at ``pos`` comes to rest, or None if the press is a
        no-op (something is already against it in that direction -- which the
        interpreter cancels for free, without ticking the counter)."""
        last = -1
        for k in self.rays[d][pos]:
            if k in occupied:
                break
            last = k
        return None if last < 0 else last


def _read(eng, ids: dict[str, int]):
    """``(sig, balls, player, moves)`` from a live engine grid, reading only
    what the frame shows. ``player`` is None on a grid that has no green ball --
    a mid-`restart` board the adapter has not reloaded yet, which the caller
    treats as unplannable."""
    h, w = eng.height, eng.width
    solid: set[int] = set()
    targets: set[int] = set()
    balls: list[int] = []
    player = None
    moves = 0
    ball_ids = {ids[n] for n in _BALL_NAMES}
    solid_ids = {ids[n] for n in _SOLID_NAMES}
    for r in range(h):
        row = eng.grid[r]
        for c in range(w):
            cell = row[c]
            k = r * w + c
            if ids["target"] in cell:
                targets.add(k)
            if cell & solid_ids:
                solid.add(k)
            if ids["count"] in cell:
                moves += 1
            if ids["overwall"] in cell:
                continue
            if ids["player"] in cell:
                player = k
                balls.append(k)
            elif cell & ball_ids:
                balls.append(k)
    return (h, w, frozenset(solid), frozenset(targets)), tuple(sorted(balls)), \
        player, moves


# ---------------------------------------------------------------------------
# The belief-space distance field
# ---------------------------------------------------------------------------

class _Field:
    """Exact worst-case press-distance over every belief state of one level.

    A state is ``(cells, order, ctrl, moves)``:

    ``cells``  cell of ball ``i``, for ``i`` in the level's own canonical order
               (its opening cells, sorted -- an arbitrary but OBSERVABLE naming,
               fixed once per level and tracked by watching what moves);
    ``order``  the identities in ACTION5 cycle order, as far as they have been
               discovered. Length 1 at the level start, growing by one per
               discovery, and completed outright at ``k - 1`` because the last
               member is then forced;
    ``ctrl``   index into ``order`` of the ball that is currently green;
    ``moves``  `Count` tiles left.

    Every press is deterministic in this state except one -- ACTION5 pressed at
    the end of the known prefix, which reveals one of the balls not yet seen and
    is the only place the hidden cycle leaks in. `distance` is the number of
    presses to a win assuming the WORST of those reveals, so a plan following it
    is a guarantee rather than an average.

    ``order`` never shrinks, so the graph is layered by ``len(order)`` and the
    layers can be solved deepest-first: within one layer every edge is
    deterministic (an undiscovered ball changes nothing about where anything is,
    only about what happens next), and the nondeterministic presses point at a
    layer that is already done. One Dijkstra per layer settles it.
    """

    __slots__ = ("lvl", "init", "k", "start", "dist", "states", "n_states")

    def __init__(self, lvl: _Level):
        self.lvl = lvl
        self.init = lvl.balls                       # identity i -> opening cell
        self.k = len(self.init)
        p0 = self.init.index(lvl.player)
        order = (p0,)
        if self.k == 2:                             # the other ball is forced
            order += tuple(i for i in range(self.k) if i != p0)
        self.start = (self.init, order, 0, lvl.n0)
        self.states = self._enumerate()
        self.n_states = len(self.states)
        self.dist = self._solve(self.states)

    # -- dynamics ---------------------------------------------------------
    def step(self, s, d: str):
        """None (the press does nothing), `_WIN`, or ``(deterministic, states)``.

        ``deterministic`` is False only for the one press that reveals a new
        member of the cycle, in which case ``states`` holds one successor per
        ball it could turn out to be."""
        cells, order, ctrl, moves = s
        tgt = self.lvl.targets
        if d == "action":
            if ctrl + 1 < len(order):               # already seen, just advance
                nxt = order[ctrl + 1]
                return _WIN if cells[nxt] in tgt else (True, ((cells, order,
                                                               ctrl + 1, moves),))
            if len(order) == self.k:                # a full cycle: wrap around
                return _WIN if cells[order[0]] in tgt else (True, ((cells, order,
                                                                    0, moves),))
            rest = [i for i in range(self.k) if i not in order]
            if all(cells[i] in tgt for i in rest):
                return _WIN                         # whoever it is, it has won
            outs = []
            for i in rest:
                no = order + (i,)
                if len(no) == self.k - 1:           # the last member is forced
                    no += tuple(j for j in range(self.k) if j not in no)
                outs.append((cells, no, len(order), moves))
            return (False, tuple(outs))
        me = order[ctrl]
        here = cells[me]
        dest = self.lvl.slide(here, d, set(cells) - {here})
        if dest is None:
            return None                             # cancelled: costs nothing
        if moves == 0:
            # `[Wall|CountEnd|Wall] -> DeadPlayer -> restart`: the level reloads.
            # Kept as a real edge, not dropped -- it is how a state that has
            # spent its budget gets back to a winnable one.
            return (True, ((self.init, order, 0, self.lvl.n0),))
        nc = list(cells)
        nc[me] = dest
        if dest in tgt:
            return _WIN
        return (True, ((tuple(nc), order, ctrl, moves - 1),))

    # -- enumeration + values ---------------------------------------------
    def _enumerate(self) -> set:
        seen = {self.start}
        queue = deque([self.start])
        while queue:
            s = queue.popleft()
            for d in _PRESSES:
                r = self.step(s, d)
                if r is None or r is _WIN:
                    continue
                for ns in r[1]:
                    if ns not in seen:
                        seen.add(ns)
                        queue.append(ns)
        return seen

    def _solve(self, states: set) -> dict:
        layers: dict[int, list] = {}
        for s in states:
            layers.setdefault(len(s[1]), []).append(s)
        dist: dict = {}
        for m in sorted(layers, reverse=True):
            self._relax(layers[m], dist)
        return dist

    def _relax(self, nodes: list, dist: dict) -> None:
        """Dijkstra over one ``len(order)`` layer.

        Seeded with the presses that leave it -- a win (cost 1) and the
        cycle-revealing ACTION5 (cost ``1 + max`` over what it can reveal, whose
        values a deeper layer has already settled) -- then relaxed backwards
        over the layer's own deterministic edges."""
        inside = set(nodes)
        rev: dict = {}
        val: dict = {}
        pq: list = []
        for s in nodes:
            best = None
            for d in _PRESSES:
                r = self.step(s, d)
                if r is None:
                    continue
                if r is _WIN:
                    cand = 1
                elif r[0]:
                    assert r[1][0] in inside, "a deterministic press left its layer"
                    rev.setdefault(r[1][0], []).append(s)
                    continue
                else:
                    seen = [dist.get(x) for x in r[1]]
                    if any(v is None for v in seen):
                        continue        # one reveal cannot win: no guarantee
                    cand = 1 + max(seen)
                if best is None or cand < best:
                    best = cand
            if best is not None:
                val[s] = best
                heapq.heappush(pq, (best, s))
        while pq:
            v, s = heapq.heappop(pq)
            if v > val.get(s, _INF):
                continue
            for p in rev.get(s, ()):
                if v + 1 < val.get(p, _INF):
                    val[p] = v + 1
                    heapq.heappush(pq, (v + 1, p))
        dist.update(val)

    # -- queries ----------------------------------------------------------
    def distance(self, s) -> "int | None":
        return self.dist.get(s)

    def cost(self, s, d: str) -> "int | None":
        """Presses to a win if ``d`` is pressed at ``s``, worst case, or None if
        the press does nothing or carries no guarantee."""
        r = self.step(s, d)
        if r is None:
            return None
        if r is _WIN:
            return 1
        seen = [self.dist.get(x) for x in r[1]]
        if any(v is None for v in seen):
            return None
        return 1 + max(seen)

    def optimal(self, s) -> list:
        """Every press that is on a shortest guaranteed route from ``s``."""
        best = self.dist.get(s)
        if best is None:
            return []
        return [d for d in _PRESSES if self.cost(s, d) == best]


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class SlidingsExpert(PSExpert):
    """Plans one press at a time off the belief field, from what the frame shows.

    It is a PARTIALLY OBSERVABLE expert in the `PSExpert.observe` sense: the
    ACTION5 cycle is not on the screen, so the answer depends on what has been
    SEEN and not only on where the pieces are. `observe` is therefore where the
    knowledge lives, `plan` is pure, and every `plan` returns a ONE-PRESS `Plan`
    so `ps_astar.record_level` re-derives the decision at every step instead of
    following a cached path. For the same reason there is no ``plan_cache_path``
    and the base memo is bypassed: a cache keyed on the board would serve the
    answer of a differently-informed run.

    What IS cached is the per-level `_Field`, which is knowledge-independent
    (it enumerates every belief state the level admits) and seed-independent
    (the engine state after reset does not depend on the seed -- only the
    presentation is augmented). Eleven levels, about three seconds, once.
    """

    #: The five presses. Also what `record_level`'s epsilon detour draws from.
    directions = list(_PRESSES)

    #: Presses held back from the adapter's per-level budget when deciding
    #: whether a win still FITS -- see `plan`. Set to 0 to plan right up to the
    #: cap (and re-read `plan`'s note on why that is not self-consistent).
    step_reserve: int = 8

    def setup(self) -> None:
        names = ("background", "target", "wall", "overwall", "count", "countend",
                 "player") + _BALL_NAMES
        self.ids = {n: self.g.obj_name_to_idx[n] for n in names}
        self.levels = [_Level(lay, self.ids) for lay in self.g.levels]
        #: static signature -> level index, so `observe` can place a board
        #: without being told which level it is looking at.
        self.by_sig = {lvl.sig: i for i, lvl in enumerate(self.levels)}
        assert len(self.by_sig) == len(self.levels), \
            "two levels share a static signature"
        self.fields: dict[int, _Field] = {}
        #: (level, cells, order, moves) -- everything the expert has worked out
        #: so far about the board in front of it. Reset when the level changes.
        self._track = None
        self.rng = random.Random(0)

    def field(self, level: int) -> _Field:
        f = self.fields.get(level)
        if f is None:
            f = self.fields[level] = _Field(self.levels[level])
        return f

    # -- knowledge --------------------------------------------------------
    def _advance(self, track, eng):
        """``(new track, planning state)`` for the board in front of the engine,
        or None when there is nothing to plan from.

        Pure: `observe` keeps the track it returns, `plan` throws it away. That
        split is what makes `plan` safe to call on the speculative boards
        `record_level`'s epsilon detour probes -- those presses never happened,
        so they must not teach the expert anything.

        Identities are followed by watching the board, which is sound because
        only ONE ball can move per press: the before/after ball sets differ in
        at most one cell. A `restart` (or a RESET) puts them all back at once,
        which is recognised by the whole set matching the level's opening
        position -- and it does NOT un-see the cycle, so ``order`` survives it.
        """
        sig, balls, player, moves = _read(eng, self.ids)
        level = self.by_sig.get(sig)
        if level is None or player is None:
            return None
        fld = self.field(level)
        if track is None or track[0] != level:
            cells, order = list(fld.init), ()
        else:
            _, cells, order, pmoves = track
            cells = list(cells)
            if moves > pmoves and balls == fld.init:
                # The counter only ever shrinks, so it going UP is a `restart`
                # (or the recorder's RESET) and nothing else. This has to be
                # tested BEFORE the single-ball diff below, not after: a restart
                # from a board where exactly one ball had left home moves
                # exactly one ball back, which is indistinguishable from a slide
                # by the ball sets alone -- and reading it as a slide swaps two
                # identities silently. `_belief_fuzz` found precisely that.
                cells = list(fld.init)
            elif tuple(sorted(cells)) != balls:
                gone = sorted(set(cells) - set(balls))
                new = sorted(set(balls) - set(cells))
                if len(gone) == 1 and len(new) == 1:
                    cells[cells.index(gone[0])] = new[0]     # the ball that slid
                elif balls == fld.init:
                    cells = list(fld.init)                   # restart / RESET
                else:                                        # unreachable
                    cells, order = list(fld.init), ()
        if player not in cells:
            return None
        ident = cells.index(player)
        if ident not in order:
            order = order + (ident,)                         # a discovery
            if len(order) == fld.k - 1:
                order += tuple(i for i in range(fld.k) if i not in order)
        cells = tuple(cells)
        return ((level, cells, order, moves),
                (cells, order, order.index(ident), moves))

    def observe(self, eng) -> None:
        got = self._advance(self._track, eng)
        if got is not None:
            self._track = got[0]

    # -- planning ---------------------------------------------------------
    def plan(self, eng, level: int | None = None) -> "list | None":
        """A ONE-PRESS `Plan` carrying the full optimal SET for this state, or
        None when nothing on this board can be planned: an unwinnable state, a
        mid-`restart` grid with no green ball on it, or a win that would not FIT
        in the step budget left.

        That last one is a guard rail, and this game needs it. Every level's
        move counter is exactly as long as its solution, so a single wasted
        slide is fatal and the only way back is to burn the rest of the counter
        and take the free `restart` -- about 30 presses on level 10. The adapter
        GAME_OVERs at ``_max_steps`` presses, so without a check the epsilon
        detour keeps accepting mistakes it has no room left to recover from, and
        each recovery is long enough to attract several more: measured, 5 of 24
        seeds livelocked to the cap at ``epsilon = 0.12``. Screening on the step
        counter (which the HUD shows, so this reads nothing a viewer could not)
        makes the detour rate taper off as the budget runs down instead.

        The reserve is what makes the screening SELF-CONSISTENT rather than
        off-by-one. `record_level` probes a detour with `engine.step`, which the
        adapter's counter never hears about, so the same state is measured one
        press cheaper during the probe than on the next iteration, when the
        detour has been taken for real -- an exact check would accept a detour
        and then refuse to plan the recovery it just approved. A fixed reserve
        of `step_reserve` presses absorbs that and leaves room for the win check
        to land on the last press."""
        got = self._advance(self._track, eng)
        if got is None:
            return None
        (lvl, *_), state = got
        fld = self.field(lvl)
        best = fld.optimal(state)
        if not best:
            return None
        left = self.game._max_steps - self.game._action_count
        if fld.distance(state) + self.step_reserve > left:
            return None
        press = best[0] if len(best) == 1 else self.rng.choice(best)
        return Plan([press], [best])

    def heuristic(self, eng) -> int:
        raise AssertionError("ps:slidings plans off the enumerated belief field; "
                             "PSExpert._astar is never used")


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SlidingsSolver(PSAStarSolver):
    game_id = "puzzlescript_slidings"
    game_name = GAME_NAME
    expert_cls = SlidingsExpert

    #: Unused -- `SlidingsExpert.plan` never reaches the base A* -- but left at
    #: the family default so a future subclass that does is not silently starved.
    node_cap = 200_000
    weight = 1

    #: The adapter GAME_OVERs at 200 actions per level; the longest plan here is
    #: 17 presses, leaving the rest of the budget to the exploration prefix and
    #: the detours.
    max_steps = 200

    #: Non-zero, unlike most of this family. The `PSAStarSolver` default is 0
    #: because those games are IRREVERSIBLE, so a detour can strand them and
    #: their recovery data has to come from the explore-then-RESET prefix alone.
    #: This game has nothing to strand: the field answers from every state the
    #: enumeration reaches (``--selfcheck`` asserts a random walk never leaves
    #: it), and the one bad outcome -- spending the move budget -- is a free
    #: `restart` rather than a loss. So one press in twelve is a random legal
    #: alternative, after which the expert re-plans from wherever it landed: the
    #: taken action is the mistake and ``optimal`` is the recovery, which is the
    #: signal a policy needs after its own error.
    #:
    #: LOWER than the 0.10-0.12 the rest of the family runs, and measured rather
    #: than guessed, because every level's move counter is exactly as long as
    #: its solution: there is no slack, so a detour is never a reordering, it is
    #: always a full burn-the-counter-and-restart cycle of about 30 presses. Win
    #: rate over 40 seeds, with `SlidingsExpert.plan`'s budget guard in place:
    #: 0.08 -> 40/40 (mean 272 frames/episode against 141 with no detours at
    #: all, so a bit under half the corpus is recovery), 0.12 -> 38/40, 0.20 ->
    #: 24/40. Without the guard 0.12 was 19/24.
    epsilon = 0.08

    def prepare_expert(self, game, expert) -> None:
        """Enumerate every level's belief space up front.

        Pure front-loading -- the space does not depend on which state builds
        it -- but it means the eleven enumerations are paid once, at startup,
        rather than inside whichever query happens to need one first."""
        for level in range(game.n_levels):
            expert.field(level)

    def solve_episode(self, seed: int, explore: bool = True):
        """Re-seed the expert per episode and forget the previous one's board.

        The expert breaks ties by sampling uniformly among the equally-optimal
        presses (DESIGN.md's stochastic-optimal target), so a seed needs its own
        stream for that to be reproducible; and its knowledge of the ACTION5
        cycle is per level-entry, never carried over -- an expert that
        remembered the cycle from an earlier episode would skip the probing on
        a frame that looks exactly like the one where it probed, which is
        precisely the inconsistency a policy cannot learn."""
        _game, expert, _solvable = self._ensure(seed)
        expert.rng = random.Random(f"{self.game_id}:{seed}")
        expert._track = None
        return super().solve_episode(seed, explore)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _build(built=None):
    """``(solver, game, expert)`` with every level's belief space enumerated.

    Takes an existing triple straight through, so a caller that runs several
    checks in one process builds ONE of these. That matters: the eleven
    enumerated fields are about 150 MB, and two live solvers is 300 MB for no
    reason at all."""
    if built is not None:
        return built
    solver = SlidingsSolver()
    game, expert, _solvable = solver._ensure(0)
    return solver, game, expert


def _drive(game, expert, level: int, rng=None, cap: int = 200):
    """Follow the expert on the INTERPRETER until it wins. Returns
    ``(presses, slides, won)``.

    Drives ``engine.step`` directly (engine space, no rotation in the way) and
    reloads the level by hand on a `restart`, which is what the adapter's
    ``_rule_restart`` branch does for a real press."""
    eng = game._engine
    game.set_level(level)
    expert._track = None
    if rng is not None:
        expert.rng = rng
    expert.observe(eng)
    presses: list[str] = []
    slides = 0
    for _ in range(cap):
        plan = expert.plan(eng, level)
        if not plan:
            break
        d = plan[0]
        presses.append(d)
        if d != "action":
            slides += 1
        eng._rule_restart = False
        eng.step(d)
        if eng.check_win():
            return presses, slides, True
        if eng._rule_restart:
            eng.load_level(game._game.levels[level])
        expert.observe(eng)
    return presses, slides, eng.check_win()


def selfcheck(trials: int = 400, steps: int = 40, verbose: bool = True,
               built=None) -> int:
    """Drive random presses through BOTH the interpreter and `_Field.step`,
    comparing the ball positions, the counter, which ball is green, the win flag
    and the `restart` flag after every one.

    This is the guard that lets the planner trust the native model, so it is
    fuzzed with the ground-truth cycle (read off `balla`/`ballb`/`ballc` HERE
    and here only -- the check is allowed to look at what the expert may not,
    because it is checking the dynamics, not planning with them).

    Prints COVERAGE, because a random walk over a 13-move budget can easily
    never exhaust it and "0 mismatches" would then say nothing about the death
    rule. Also asserts the enumerated space is CLOSED: every state the walk
    reaches is one the field already knows, which is what entitles the epsilon
    detour and the exploration prefix to re-plan from wherever they land.

    Returns the number of mismatches."""
    solver, game, expert = _build(built)
    eng = game._engine
    ids = expert.ids
    slots = (ids["player"],) + tuple(ids[n] for n in _BALL_NAMES)
    total = 0
    # No `discover` counter: this pass is seeded with the whole cycle, so the
    # frontier press never fires. `_belief_fuzz` is what covers it.
    cover = dict(slide=0, blocked=0, cycle=0, restart=0, win=0)
    for level in range(game.n_levels):
        fld = expert.field(level)
        lvl = expert.levels[level]
        rng = random.Random(f"slidings:selfcheck:{level}")
        bad = seen_states = 0
        for _ in range(trials):
            game.set_level(level)
            # ground truth: the cycle is the slot order P, A, B, C, minus the
            # decoys parked under an OverWall.
            ring = []
            for obj in slots:
                for r in range(eng.height):
                    for c in range(eng.width):
                        k = r * eng.width + c
                        if obj in eng.grid[r][c] and k not in lvl.solid:
                            ring.append(k)
            order = tuple(fld.init.index(k) for k in ring)
            state = (fld.init, order, 0, lvl.n0)
            for _ in range(steps):
                d = rng.choice(_PRESSES)
                want = fld.step(state, d)
                eng._rule_restart = False
                eng.step(d)
                seen_states += 1
                won, restarted = eng.check_win(), eng._rule_restart
                if restarted:
                    eng.load_level(game._game.levels[level])
                _sig, balls, player, moves = _read(eng, ids)
                if want is None:
                    ok = (not won and not restarted and balls
                          == tuple(sorted(state[0])) and moves == state[3]
                          and player == state[0][state[1][state[2]]])
                    cover["blocked"] += 1
                elif want is _WIN:
                    ok = won
                    cover["win"] += 1
                else:
                    nxt = want[1][0]
                    ok = (not won and balls == tuple(sorted(nxt[0]))
                          and moves == nxt[3]
                          and player == nxt[0][nxt[1][nxt[2]]]
                          and restarted == (d != "action" and state[3] == 0))
                    if d == "action":
                        cover["cycle"] += 1
                    elif restarted:
                        cover["restart"] += 1
                    else:
                        cover["slide"] += 1
                    # The closure the field relies on, checked rather than
                    # argued: a state a random walk can reach is already in the
                    # enumerated space, so the field can answer for anything the
                    # exploration prefix or an epsilon detour lands on.
                    if nxt not in fld.states:
                        ok = False
                if not ok:
                    bad += 1
                    if verbose:
                        print(f"  L{level}: MISMATCH on {d} at {state}")
                    break
                if won:
                    break
                if want is not None:
                    state = want[1][0]
        total += bad
        if verbose:
            print(f"  L{level}: {'OK' if not bad else f'{bad} MISMATCHES'} "
                  f"({seen_states} presses vs the interpreter, "
                  f"{fld.n_states} belief states)")
    if verbose:
        print("  coverage: " + ", ".join(f"{k}={v}" for k, v in cover.items()))
        missing = [k for k, v in cover.items() if not v]
        if missing:
            print(f"  WARNING: never exercised {', '.join(missing)}")
    return total


def _true_cycle(game, expert, level: int) -> tuple:
    """The ACTION5 cycle of ``level``, as identities, read off the slot labels.

    The ONE place this file looks at `balla` / `ballb` / `ballc` separately, and
    it is a test fixture: `_belief_fuzz` needs a ground truth to hold the
    expert's observation-only tracker against. Nothing on the planning path may
    call it -- see "Why the expert plans in belief space"."""
    eng = game._engine
    game.set_level(level)
    lvl = expert.levels[level]
    fld = expert.field(level)
    ring = []
    for name in ("player",) + _BALL_NAMES:
        obj = expert.ids[name]
        for r in range(eng.height):
            for c in range(eng.width):
                k = r * eng.width + c
                if obj in eng.grid[r][c] and k not in lvl.solid:
                    ring.append(k)
    return tuple(fld.init.index(k) for k in ring)


def _belief_fuzz(game, expert, trials: int = 200, steps: int = 40,
                 verbose: bool = True) -> tuple[int, dict]:
    """Fuzz the OBSERVATION-ONLY tracker against that ground truth.

    `selfcheck` proves the DYNAMICS, but it hands the model the cycle for free,
    so it never exercises the part the expert actually depends on: recovering
    which ball is which from the frames alone. This does. It walks the
    interpreter at random and, after every press, requires
    `SlidingsExpert._advance` -- fed nothing but the grid -- to agree with the
    truth on where every ball is, which one is green and how many moves are
    left, and requires its ``order`` to be a genuine PREFIX of the real cycle
    (never a guess that happens to be wrong later).

    The three things that could break it are all in here: a ball whose identity
    is followed through a slide, a `restart` that moves every ball at once, and
    an ACTION5 at the frontier that reveals a new member."""
    eng = game._engine
    bad = 0
    cover = dict(follow=0, restart=0, discover=0)
    for level in range(game.n_levels):
        fld = expert.field(level)
        lvl = expert.levels[level]
        cycle = _true_cycle(game, expert, level)
        rng = random.Random(f"slidings:belief:{level}")
        for _ in range(trials):
            game.set_level(level)
            expert._track = None
            expert.observe(eng)
            truth = (fld.init, cycle, 0, lvl.n0)
            for _ in range(steps):
                got = expert._advance(expert._track, eng)
                if got is None:
                    bad += 1
                    print(f"  L{level}: the tracker lost the board")
                    break
                cells, order, ctrl, moves = got[1]
                if (cells != truth[0] or moves != truth[3]
                        or order != cycle[:len(order)]
                        or order[ctrl] != truth[1][truth[2]]):
                    bad += 1
                    if verbose:
                        print(f"  L{level}: tracker {got[1]} vs truth {truth}")
                    break
                before = len(order)
                d = rng.choice(_PRESSES)
                nxt = fld.step(truth, d)
                eng._rule_restart = False
                eng.step(d)
                if eng.check_win():
                    break
                if eng._rule_restart:
                    eng.load_level(game._game.levels[level])
                    cover["restart"] += 1
                elif d != "action":
                    cover["follow"] += 1
                expert.observe(eng)
                if nxt is None:
                    continue
                truth = nxt[1][0]
                if len(expert._track[2]) > before:
                    cover["discover"] += 1
    if verbose:
        print("  tracker: " + ("OK" if not bad else f"{bad} MISMATCHES")
              + " (" + ", ".join(f"{k}={v}" for k, v in cover.items()) + ")")
    return bad, cover


def _report(built=None) -> int:
    """Print each level's belief space, its plan and the engine's verdict -- the
    quick "is this game still fully solved" check."""
    _solver, game, expert = _build(built)
    bad = 0
    total_states = 0
    for level in range(game.n_levels):
        fld = expert.field(level)
        lvl = expert.levels[level]
        total_states += fld.n_states
        presses, slides, won = _drive(game, expert, level)
        ties = _tie_rate(game, expert, level, presses)
        bad += 0 if (won and slides == lvl.n0) else 1
        print(f"  L{level}: {lvl.w}x{lvl.h}  {len(lvl.balls)} balls  "
              f"{len(presses):3d} presses (worst case "
              f"{fld.distance(fld.start):3d})  "
              f"{slides}/{lvl.n0} moves spent  win={won}  "
              f"{fld.n_states:6d} belief states  "
              f"{ties:.2f} optimal presses/step")
    print(f"  total: {total_states} belief states over {game.n_levels} levels"
          + ("" if not bad else f"  -- {bad} LEVELS FAILED"))
    return 0 if not bad else 1


def _tie_rate(game, expert, level: int, presses) -> float:
    """Mean size of the optimal SET along ``presses`` -- how much of the label
    the corpus would throw away if it shipped one press per step."""
    eng = game._engine
    game.set_level(level)
    expert._track = None
    expert.observe(eng)
    fld = expert.field(level)
    tot = 0
    for d in presses:
        got = expert._advance(expert._track, eng)
        tot += len(fld.optimal(got[1])) if got else 0
        eng._rule_restart = False
        eng.step(d)
        if eng._rule_restart:
            eng.load_level(game._game.levels[level])
        expert.observe(eng)
    return tot / max(1, len(presses))


def _verify(sweeps: int = 200, built=None) -> int:
    """Double-entry check of the belief field and of every tie set it ships.

    `_Field` computes its values ONE way -- a layer-by-layer Dijkstra over
    reverse edges. A bug in that would produce a self-consistent field, a plan
    that still wins, and tie sets that are quietly wrong; and the training
    labels ARE those tie sets, so "it wins" is not enough of a check. Two more
    passes, which fail differently:

      * **Bellman fixpoint.** Re-derive every value by relaxing
        ``V(s) = min_press (1 + max_reveal V(succ))`` to a fixpoint over the
        FORWARD successor function alone -- no layering, no reverse edges,
        nothing shared with the Dijkstra -- and require it to agree everywhere.
        Given a transition function that matches the interpreter (which
        `selfcheck` fuzzes) and a state set closed under every press, that is a
        proof of minimality.
      * **Executable labels.** Walk each level's plan on the INTERPRETER and, at
        every step, actually press each direction the label calls optimal: the
        board the engine lands on must be the model's successor, and it must be
        strictly closer -- exactly one press closer for a deterministic press,
        at least that for a reveal, whose cost is quoted against the worst case.
    """
    _solver, game, expert = _build(built)
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        fld = expert.field(level)

        # -- pass 1: Bellman fixpoint over the successor function -------------
        states = fld.states
        check = {s: _INF for s in states}
        for _ in range(sweeps):
            changed = False
            for s in states:
                best = _INF
                for d in _PRESSES:
                    r = fld.step(s, d)
                    if r is None:
                        continue
                    if r is _WIN:
                        cand = 1
                    else:
                        vs = [check[x] for x in r[1]]
                        cand = _INF if max(vs) >= _INF else 1 + max(vs)
                    best = min(best, cand)
                if best < check[s]:
                    check[s] = best
                    changed = True
            if not changed:
                break
        else:
            bad += 1
            print(f"  L{level}: the Bellman sweep did not converge")
        wrong = [s for s in states
                 if (check[s] if check[s] < _INF else None) != fld.distance(s)]
        if wrong:
            bad += 1
            print(f"  L{level}: the field disagrees with its Bellman fixpoint "
                  f"at {len(wrong)} of {len(states)} states")

        # -- pass 2: every shipped label, pressed on the interpreter ----------
        presses, _slides, won = _drive(game, expert, level)
        if not won:
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        game.set_level(level)
        expert._track = None
        expert.observe(eng)
        labels = 0
        for i, press in enumerate(presses):
            got = expert._advance(expert._track, eng)
            state = got[1]
            here = fld.distance(state)
            best = fld.optimal(state)
            if press not in best:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            before = snapshot(eng)
            for alt in best:
                eng._rule_restart = False
                eng.step(alt)
                labels += 1
                if eng.check_win():
                    if here != 1:
                        bad += 1
                        print(f"  L{level}: step {i} labels {alt} optimal and it "
                              f"wins, but the field says {here} presses remain")
                else:
                    if eng._rule_restart:
                        eng.load_level(game._game.levels[level])
                    landed = expert._advance(expert._track, eng)
                    there = fld.distance(landed[1]) if landed else None
                    if there is None or there > here - 1:
                        bad += 1
                        print(f"  L{level}: step {i} labels {alt} optimal, but "
                              f"the interpreter does not land closer "
                              f"({here} -> {there})")
                restore(eng, before)
            eng._rule_restart = False
            eng.step(press)
            if eng._rule_restart:
                eng.load_level(game._game.levels[level])
            expert.observe(eng)
        print(f"  L{level}: {len(states):6d} states relaxed, "
              f"{len(presses):3d} plan steps, {labels:3d} labels pressed on the "
              f"interpreter -- {'OK' if not bad else 'see above'}")
    print(f"verify: {bad} problems")
    return 1 if bad else 0


def _replay(seeds: int = 2) -> int:
    """Re-drive every recorded episode on a FRESH adapter and require it to
    reproduce the taped frames exactly, press for press, and end in WIN. Then
    record each seed a second time and require the two to be identical.

    This is the end-to-end check the corpus contract rests on, and nothing else
    here can catch what it catches. The recorded index is the SCREEN press --
    what a player of the augmented view would press -- while the expert plans in
    ENGINE space, so a rotation-remap mistake shows up as a trajectory that
    replays into a different game (see `ps_astar.screen_action`, and the copy of
    that bug it was written for). The recorder cannot notice, because it drives
    the same adapter it tapes; a second adapter built from the seed alone can.

    Determinism is checked in the same pass because it is the same run: the
    exploration prefix, the epsilon detour and the expert's own tie-breaking all
    draw on seeded streams, and a stray global `random` call anywhere in the
    chain would make two shards of `parallelize_generator` disagree.

    The twin has to walk the same SEED SEQUENCE, not just the same seed. The
    epsilon detour and the tie-breaking are seeded per episode, but the
    exploration prefix draws on the solver's SESSION rng, which `PSAStarSolver`
    advances across episodes on purpose (the prefix is an episode-wide arc). So
    a fresh solver asked for seed 1 legitimately explores differently from one
    that has already recorded seed 0, and comparing those two would be measuring
    the harness rather than this generator. The twin shares the adapter and the
    enumerated fields -- they are seed-independent, and rebuilding them would
    double a 150 MB footprint for nothing."""
    solver = SlidingsSolver(rng=random.Random(0))
    twin = SlidingsSolver(rng=random.Random(0))
    twin._game, twin._expert, twin._solvable = solver._ensure(0)
    bad = 0
    for seed in range(seeds):
        ok, levels = solver.solve_episode(seed)
        levels = SlidingsSolver.normalize_levels(levels)
        again = SlidingsSolver.normalize_levels(twin.solve_episode(seed)[1])
        if again != levels:
            bad += 1
            print(f"  seed {seed}: a second recording of the same seed differs")
        frames = presses = 0
        for lvl in levels:
            game = PuzzleScriptAdapter(GAME_NAME, seed=seed)
            game.set_level(lvl["level_id"])
            got = [np.asarray(game._current_frame)]
            for act in lvl["actions"][1:]:
                fd = game.perform_action(
                    ActionInput(id=_ID_TO_GAMEACTION[int(act["index"])]))
                got.append(np.asarray(fd.frame[-1] if fd.frame
                                      else game._current_frame))
            want = lvl["observations"]
            if len(got) != len(want):
                bad += 1
                print(f"  seed {seed} L{lvl['level_id']}: replay produced "
                      f"{len(got)} frames, the tape has {len(want)}")
                continue
            for i, (g, w) in enumerate(zip(got, want)):
                if not np.array_equal(g, np.asarray(w, dtype=g.dtype)):
                    bad += 1
                    print(f"  seed {seed} L{lvl['level_id']}: frame {i} differs "
                          f"on replay")
                    break
            if game._state != GameState.WIN:
                bad += 1
                print(f"  seed {seed} L{lvl['level_id']}: replay ends in "
                      f"{game._state}, not WIN")
            frames += len(got)
            presses += len(lvl["actions"]) - 1
        print(f"  seed {seed}: {'all levels won' if ok else 'INCOMPLETE'}, "
              f"{len(levels)} levels, {presses} presses, {frames} frames "
              f"replayed -- {'OK' if not bad else 'see above'}")
    print(f"replay: {bad} problems")
    return 1 if bad else 0


def _audit() -> int:
    """Assert every cell COMPOSITION the game can show is distinct in the frame,
    at every cell size the eleven levels render at.

    The pair this exists for is ``ball+target`` against ``ball`` (and the same
    for the player): the win condition is ``All Player on Target``, so a piece
    that hides the ring it is standing on makes the WINNING board
    indistinguishable from any other, and makes the goal itself vanish whenever
    a ball happens to park on it. That is exactly what shipped -- a 3x3 ring
    inside a 5x5 cell, under a sprite that is opaque over all of it -- and
    `data/puzzlescript_games/slidings.txt` now draws the ring as the 5x5 outer
    frame so it survives in the ball sprite's four transparent corners.

    ``wall``, ``countend`` and a decoy parked under an ``overwall`` are SUPPOSED
    to be identical: all three are simply solid, and what the player has to read
    off the counter is how many ``count`` tiles are LEFT, which is a distinct
    composition from all of them.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    comps = [(), ("target",), ("wall",), ("count",), ("countend",),
             ("overwall", "ballb"), ("balla",), ("balla", "target"),
             ("player",), ("player", "target")]
    #: pairs the game means to be indistinguishable -- everything solid looks
    #: solid, and the counter is read from the `count` tiles that are left.
    allowed = {frozenset({("wall",), ("countend",)}),
               frozenset({("wall",), ("overwall", "ballb")}),
               frozenset({("countend",), ("overwall", "ballb")})}
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
                   and frozenset({a, b}) not in allowed]
        clashes_total += len(clashes)
        name = lambda t: "+".join(t) if t else "floor"           # noqa: E731
        print(f"{h}x{w} board (level{'s' if len(levels) > 1 else ''} "
              f"{', '.join(str(x) for x in levels)}), "
              f"cell {max(1, min(64 // h, 64 // w))}px")
        for objs in comps:
            painted = int((shots[objs] != shots[()]).sum())
            over = ""
            if objs and objs[-1] == "target" and len(objs) == 2:
                base = (objs[0],)
                over = (f", {int((shots[objs] != shots[base]).sum()):4d} px of "
                        f"Target survive under the {objs[0]}")
            print(f"  {name(objs):16s} {painted:4d} px differ from bare floor"
                  f"{over}")
        for a, b in clashes:
            print(f"  IDENTICAL: {name(a)} == {name(b)}")
    print("audit clean" if not clashes_total
          else f"AUDIT FAILED: {clashes_total} indistinguishable pairs")
    return 0 if not clashes_total else 1


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        # ONE build shared by all three passes -- see `_build`.
        shared = _build()
        mismatches = selfcheck(built=shared)
        mismatches += _belief_fuzz(*shared[1:])[0]
        print(f"selfcheck: {mismatches} mismatches")
        sys.exit(1 if mismatches else _report(shared))
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--replay" in sys.argv:
        sys.exit(_replay())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(SlidingsSolver.main())
