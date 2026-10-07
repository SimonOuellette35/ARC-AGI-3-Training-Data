"""Generate Phase-1 training data for the PuzzleScript game ps:crateblob
("CrateBlob" by Zachary Abel -- steer a blob of crates whose rows and columns
slide past each other onto a set of gray targets).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the `BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: a
native re-implementation of the mechanic (`_Board`, ~120 lines of dynamics), a
macro A* / beam over it (`_Planner`), the interpreter certification that makes
planning on a model safe, and the disk plan cache.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_crateblob",
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
action (post rotation remap), i.e. the button an agent presses in the presented
view, so replaying the recorded actions reproduces the recorded frames exactly.

The game
--------
You are ONE CELL of a blob of crates -- a rigid-ish amoeba whose rows and columns
shear past each other. Four arrow keys, no ACTION (nothing in the file binds it,
so it is dropped from the search). Every level is won by parking crates on all of
its targets, simultaneously, with everything at rest.

The whole mechanic is six statements, all asserted against the interpreter by
``--fuzz`` (which compares the model's FULL state after every single press):

  * **A blob is a COMPONENT, and components never merge.** Connectivity lives in
    hidden ``nbR/nbU/nbL/nbD`` pointers. They are rebuilt after every settle by
    ``right [CrateNew | CrateNew] -> [CrateNew nbR | CrateNew nbL]`` -- but that
    rule runs INSIDE the per-component ``startloop``, so it only ever links cells
    of the SAME component. Two blobs that come to touch stay two blobs forever
    (the level maps say which is which with the digits 0/1/2), and a blob that
    is torn geometrically SPLITS into two on the next settle. The seam is
    visible: the yellow ``HAdjImg``/``VAdjImg`` strips that erase the internal
    cell outlines are also drawn per component, so a component boundary keeps
    its double orange line.
  * **Pressing INTO your own blob just walks the avatar.** ``right [right Player
    nbR] -> [Player PloffR1 nbR]`` moves the player and nothing else. Only when
    the neighbouring cell is *not* the same component (empty, wall, or a foreign
    blob) does ``right [right Player no nbR] -> [Player PloffR1 CroffR1]`` fire
    and start a push. So the avatar's position inside the blob is a free
    parameter chosen by walking -- which is what makes a macro search the right
    shape here, and what makes most presses tied (see "Optimal-action sets").
  * **The push set is a four-rule closure, and the third rule is the game.**
    With ``d`` the pressed direction and ``M`` the set of cells that will move:

        push   x in M                       => x+d in M   (if x+d is a crate)
        drag   y in M, x = y-d same comp    => x in M
        shear  y in M, no nb_{-d}(y),
               B = y+e same comp (e perp),
               no nb_d(B)                   => B in M

    ``push`` is ordinary sokoban and crosses component boundaries; ``drag`` pulls
    the rest of your own row along behind you. ``shear`` is the "rows won't fully
    let go of their neighbouring rows" rule the tutorial shouts about: ``y`` is
    the trailing end of its run and ``B``, connected to it sideways, is the
    LEADING end of its own run, so one more step of shear would leave the two
    runs with zero column overlap and tear the blob -- so ``B`` comes too. The
    engine spreads this over several ``again`` ticks (``shear`` creates a
    lagging ``Croff*0`` that only propagates on the NEXT tick), but nothing
    moves until every offset has finished, and the rules are monotone, so the
    settled result is the plain fixpoint and the tick structure is invisible.
  * **A wall anywhere in the push set cancels the WHOLE turn.** Not "the blob
    stops": ``right [CroffR1 | Wall no Collision] -> [... CollisionL_f1]`` sets
    ``CollidedTrue``, which freezes every offset until the collision animation
    ends and then deletes all of them -- the avatar's own step included. Levels
    are wall-enclosed, so this is also what keeps crates on the board.
  * **Gravity is per component, and support crosses components.** A component
    can't fall if any of its cells sits directly on a wall or on a cell of a
    component that can't fall (``down [Crate | Wall]`` seeds it, ``down [Crate |
    CantFall]`` and the four ``nb`` rules spread it). Everything else falls one
    cell, re-splits, and the check runs again. Level 4's twelve-crate blob hangs
    off a single one-cell wall pillar poking into its underside.
  * **Fire deletes what enters it, player included.** ``[Player Fire] -> [Fire]``
    and ``[CrateNew Fire] -> [Fire]`` are checked after each unit of movement, so
    a crate is destroyed the moment it falls into the flames and the level is
    silently lost the moment the avatar does. `_Board.step` models both, and
    `_Planner` treats a dead avatar (and a board with fewer crates than targets)
    as a dead end.

Expert solver
-------------
The interpreter is FAR too slow to search: CrateBlob is 282 rules, two nested
``startloop`` component walks and a five-frame animation per press, which is
~0.5-0.9 s per engine step even after the speed-up below. A 40-move level would
cost more than a minute just to *replay*, never mind expand. So `_Board` is a
native re-implementation of the six statements above, `_Planner` searches THAT,
and every plan it returns is replayed through the real interpreter before it is
accepted (`CrateBlobExpert._certify`) -- a modelling slip costs a level, never a
recorded trajectory that does not win.

`_Planner`'s successors are ``walk the avatar to cell y, then press d`` MACROS,
so the search depth is the number of PUSHES rather than the number of presses,
and each walk is one BFS inside the component instead of four levels of
branching. ``g`` counts primitive moves, so plans are shortest in the metric the
agent actually pays. Two heuristics, run as a ladder (`_LADDER`) and the shortest
certified plan kept. Both measure a target's distance to a crate by BFS over the
cells a crate could ever occupy -- neither wall nor fire -- rather than in a
straight line, which is what makes them usable at all: these levels are built of
detours, and level 7's only route to its target leaves that side of the map
entirely and comes back down a shaft, so a straight-line estimate points the
search at the fire that blocks the short way.

  * ``h_max`` = max over uncovered targets of that distance to the nearest crate.
    Admissible -- one press moves any given crate by at most one cell, so every
    target's nearest crate needs at least that many presses, and they cannot be
    added because a single press may serve all of them at once. Weight 1 with
    this is the optimal pass, and it wins nine of the ten levels outright.
  * ``h_sum`` = the same distances summed over a greedy one-crate-per-target
    assignment. Not a bound, but it is the only thing that can tell "the blob is
    drifting toward the target block" from "the blob is drifting away". Used
    weighted -- and at the last rung's weight of 60 it is effectively greedy,
    which is what finally cracks level 7.

One thing the walk-and-push macro buys here that it does not buy in a sokoban:
the avatar is INSIDE the thing it is pushing, so "where do I push from" is a
choice over the blob's own cells, and the blob's shape changes under it every
press. Enumerating those cells per node is what keeps the branching honest --
about eleven macros per state on the big levels.

Plans are seed-independent (the levels are fixed ASCII maps; only the
PRESENTATION is augmented per seed), so seed 0 pays for every search and later
seeds replay them under their own rotation. The plans are also written to
``data/crateblob_plans.json`` -- otherwise every shard of `parallelize_generator`
would repeat the searches on every core, and the certification replays alone
(one engine step per plan move) are minutes. Delete the file to re-derive it;
cold that costs ~15 minutes, warm a run starts in about a second.

**Budget the RECORDING, though.** The plans total 440 presses across the ten
levels and the interpreter is the floor on replay, so taping one seed is ~5
minutes of wall clock and that does NOT shrink with a warm cache -- unlike the
other ps: games, where the plan cache makes later seeds nearly free. A 200-seed
corpus is ~17 hours on one core. Use `parallelize_generator` (the plan cache is
what makes the shards cheap to start).

Optimal-action sets
-------------------
Walking the avatar inside its own blob changes NOTHING but the avatar's cell, so
every shortest route to the cell a push is taken from is equally optimal and
leaves an identical board. `_optimal_sets` re-walks the plan on the model, groups
the free stretches by the "is this a press the level answers, into my own
component?" test, and labels each of their presses with every direction that
stays on a shortest route to the same push cell. A push is labelled with itself,
and no step ever ships unlabelled (the always-emit-optimal-targets rule).

Worth knowing what that actually yields here, because it is the opposite of the
sokoban-shaped games: about 40% of the presses are walks, but only ONE step in
the ten levels has a real tie. The blobs shear themselves into one-cell-wide
snakes within a few pushes -- that is what the game is -- and inside a snake the
route between two cells is unique. So the labelling is nearly all singletons by
the nature of the mechanic, not because the analysis is missing something; it is
still worth doing, because the handful of fat-blob walks that DO have a choice
sit at the start of the levels where every trajectory begins.

The palette / render adaptation
-------------------------------
``data/puzzlescript_games/CrateBlob.txt`` carries an ARC adaptation of the
original art and animation timing (colours, two sprites, one collision-layer
move, and the length of the tween -- no rule's LOGIC was touched). Five things
were wrong, four of them fatal:

  * **The crates wore the wall's colour.** ARC has one orange, and both
    ``brown`` (walls) and ``orange`` (crates) land on index 12. Walls are now
    ``darkgray``/``#333333`` (3/4) with black edging.
  * **The crate body wore the target's colour.** ``lightbrown`` quantizes to 2,
    mid gray -- exactly the gray of the targets. The crate fill (``Crimg*``,
    ``HAdjImg*``, ``VAdjImg*``, ``FillImg*``) is now ``yellow`` (11).
  * **A crate on a target ERASED it.** ``Crimg0`` is an opaque 5x5 and its
    collision layer is above ``Target``, so a covered target vanished and the
    win condition became invisible. The target is now the CELL BORDER, in
    purple, on the TOPMOST collision layer -- so it outlines the crate that
    covers it instead of being hidden by it. (Same layer-split trick as
    [[color-combination-solver]]; here it is a layer MOVE because the crate
    sprite is the blob's outline and cannot give up its ring.)
  * **The avatar was invisible on levels 1 and 3.** ``Plimg0``'s blue pixels sat
    only on the sprite's odd rows/columns, and those two levels render at
    ``cell_px = 4``, whose centred sampling reads sprite rows/cols 0,1,3,4 --
    every blue pixel fell in the gaps. It is now a solid 3x3 core, which
    survives every ``cell_px`` the ten levels use (4 to 7).
  * **The five-frame tween nearly overran the engine's ``again`` budget.** Every
    press animates ``Croff*1 -> ... -> Croff*5`` and every ONE-CELL fall is a
    fresh tween, so a nine-cell drop in level 5 or 9 costs ~45 of the engine's
    50 ``again`` iterations and would have been truncated MID-FALL into a frame
    no rule of the game can produce. The tween is now two frames (``Croff*0 ->
    Croff*1 -> Croff*5``), which keeps the lagging-shear semantics exactly (the
    shear still wakes one tick late) and is invisible in the record, because a
    directional press only ever emits its settled frame. It also made the engine
    ~1.8x faster.

Augmentation
------------
CrateBlob has GRAVITY, so it is not in `PuzzleScriptAdapter._FLIP_GAMES`: a
vertical flip would present a world that falls upward and a horizontal one would
mirror it without mirroring the fall. It takes the default rotation-only
augmentation (``rotation_k`` in {0,1,2,3} with the matching directional action
remap), which is a consistent presentation -- gravity simply pulls toward a
different screen edge -- giving 4 presentations of 10 levels. No colour
augmentation: the orange/yellow crate, the purple target and the red fire are
the three things the frames are supposed to teach apart.

Usage (run from the repo root):
    python solvers/generate_crateblob_training.py --episodes 200 \
        --out data/training_multi_level/crateblob

    python solvers/generate_crateblob_training.py --plans   # per-level report
    python solvers/generate_crateblob_training.py --fuzz    # model vs engine
"""

from __future__ import annotations

import heapq
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import PuzzleScriptAdapter        # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert,        # noqa: E402
                                     restore, snapshot)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "CrateBlob"

#: Disk cache of the per-level start plan AND its optimal-action sets. The
#: searches are seed-independent but every plan also pays one interpreter step
#: per move to be certified, which every shard of `parallelize_generator` would
#: otherwise repeat on every core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "crateblob_plans.json"

_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
#: The two directions perpendicular to each press -- the axis the shear rule
#: reaches along.
_PERP: dict[str, tuple[str, str]] = {
    "up": ("left", "right"), "down": ("left", "right"),
    "left": ("up", "down"), "right": ("up", "down"),
}

#: (heuristic name, weight, macro-expansion budget) tried in order; the shortest
#: plan the interpreter confirms wins is kept, and the first pass short-circuits
#: the rest because its plan is provably shortest.
#:
#: ``max`` at weight 1 is admissible and wins NINE of the ten levels outright
#: (so those nine plans are shortest), the worst of them -- level 8's seven
#: targets -- in about six seconds. Level 7 is the outlier and needs the last
#: rung, whose weight makes ``f`` effectively the heuristic alone, i.e. greedy
#: with a tie-break: 101 presses after ~4 minutes and 12M expansions. Everything
#: milder was tried on it and failed -- weights 1/3/12 out to 4M expansions, a
#: 2000-wide beam, and a plain breadth-first sweep of 6M expansions over 535k
#: states. That level is a long detour (the only way to its target leaves the
#: blob's chamber entirely and comes back down a fire-floored shaft), which is
#: exactly the shape a bounded-cost search cannot reach and a greedy one can.
_LADDER: tuple[tuple[str, int, int], ...] = (
    ("max", 1, 1_500_000),
    ("sum", 1, 1_500_000),
    ("sum", 3, 2_000_000),
    ("sum", 60, 12_000_000),
)
#: Beam fallback: (width, macro depth, macro budget). Nothing currently needs it
#: -- it is the safety net for a re-plan from an off-plan state, which is where
#: a level whose START the ladder cracks can still hand the search a shape none
#: of its weights like.
_BEAM: tuple[int, int, int] = (600, 80, 2_000_000)


# ---------------------------------------------------------------------------
# Native model of the mechanic
# ---------------------------------------------------------------------------

class _Board:
    """The static scenery of one level plus the game's dynamics.

    A state is the pair ``(player_cell, comps)``, where ``comps`` is the tuple of
    blob components -- each a sorted tuple of cells -- kept in a canonical order
    so states compare and hash directly. Walls, targets, fire and the level's
    "no vertical presses" flag never change, so they live here; `step` is the
    whole mechanic (see the module docstring for the six statements it encodes).
    """

    def __init__(self, eng, game):
        idx = game.obj_name_to_idx
        grid = eng.grid
        self.h, self.w = len(grid), len(grid[0])
        self.crate = idx["crate"]
        self.player_id = idx["player"]
        #: only nbR / nbU are read: the pointers are symmetric, so half of them
        #: already spans every same-component adjacency.
        self.nbr, self.nbu = idx["nbr"], idx["nbu"]
        wall_ids = {idx["wall0"], idx["wall1"]}
        fire_ids = {idx[n] for n in ("fire0", "fire1", "firedown0", "firedown1")}
        target_id, nov_id = idx["target"], idx["noverticals"]
        walls, fires, targets = set(), set(), set()
        self.no_verticals = False
        for r in range(self.h):
            for c, cell in enumerate(grid[r]):
                if cell & wall_ids:
                    walls.add((r, c))
                if cell & fire_ids:
                    fires.add((r, c))
                if target_id in cell:
                    targets.add((r, c))
                if nov_id in cell:
                    self.no_verticals = True
        self.walls, self.fires = frozenset(walls), frozenset(fires)
        self.targets = frozenset(targets)
        #: Presses the level actually responds to. Level 0 carries `NoVerticals`
        #: (the `!` glyph), which cancels up/down outright -- dropping them halves
        #: its branching instead of expanding two no-ops per node.
        self.dirs = (("left", "right") if self.no_verticals
                     else ("up", "down", "left", "right"))
        #: ``{target: {cell: fewest unit steps from cell to target}}`` over the
        #: cells a crate could ever occupy -- neither wall (it cannot enter one)
        #: nor fire (it would be destroyed on the way). That makes it a lower
        #: bound on the travel of whichever crate ends up filling the target, and
        #: therefore on the number of presses, which plain Manhattan is too: the
        #: point of measuring it around the walls is that these levels are full
        #: of detours (level 7's only route to its target leaves the target's
        #: side of the map entirely), and a straight-line estimate walks the
        #: search into the fire that blocks the short way.
        self.dist = {t: self._free_bfs(t) for t in self.targets}

    def _free_bfs(self, source):
        dist = {source: 0}
        queue = deque([source])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _DELTA.values():
                nxt = (r + dr, c + dc)
                if not (0 <= nxt[0] < self.h and 0 <= nxt[1] < self.w):
                    continue
                if nxt in self.walls or nxt in self.fires or nxt in dist:
                    continue
                dist[nxt] = dist[(r, c)] + 1
                queue.append(nxt)
        return dist

    # -- reading the interpreter ------------------------------------------
    def read(self, eng):
        """Engine grid -> ``(player_cell, comps)``, components taken from the
        hidden ``nb`` pointers (NOT from geometry: two blobs may touch)."""
        crate, player_id = self.crate, self.player_id
        parent: dict[tuple[int, int], tuple[int, int]] = {}
        player = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if crate in cell:
                    parent[(r, c)] = (r, c)
                if player_id in cell:
                    player = (r, c)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for (r, c) in list(parent):
            cell = eng.grid[r][c]
            for nb, other in ((self.nbr, (r, c + 1)), (self.nbu, (r - 1, c))):
                if nb in cell and other in parent:
                    a, b = find((r, c)), find(other)
                    if a != b:
                        parent[a] = b
        groups: dict[tuple[int, int], list] = {}
        for cell in parent:
            groups.setdefault(find(cell), []).append(cell)
        return player, tuple(sorted(tuple(sorted(g)) for g in groups.values()))

    # -- dynamics ----------------------------------------------------------
    def step(self, state, direction):
        """One press. Returns the settled ``(player, comps)``.

        The very same tuple is handed back when the turn is a no-op -- a dead
        avatar, a cancelled vertical on level 0, or a push that hit a wall --
        which is how the search recognises a move the engine refused."""
        player, comps = state
        if player is None or (self.no_verticals and direction in ("up", "down")):
            return state
        dr, dc = _DELTA[direction]
        comp_of = {cell: i for i, comp in enumerate(comps) for cell in comp}
        ahead = (player[0] + dr, player[1] + dc)
        if comp_of.get(ahead) == comp_of[player]:
            return (ahead, comps)                    # walk inside your own blob

        moving = self._closure(comp_of, player, direction)
        for (r, c) in moving:
            nxt = (r + dr, c + dc)
            if not (0 <= nxt[0] < self.h and 0 <= nxt[1] < self.w) \
                    or nxt in self.walls:
                return state                         # collision cancels the turn
        return self._settle(*self._shift(player, comps, moving, dr, dc))

    def _closure(self, comp_of, start, direction):
        """The set of cells the press moves: push + drag + shear, to fixpoint.

        Every rule fires off a cell that is already moving and its guards are
        static (the ``nb`` pointers cannot change mid-press), so the rules are
        monotone and a single worklist pass per added cell reaches the same
        fixpoint the engine's multi-tick animation does."""
        dr, dc = _DELTA[direction]
        br, bc = -dr, -dc
        perps = [_DELTA[e] for e in _PERP[direction]]
        moving = {start}
        stack = [start]
        while stack:
            y = stack.pop()
            mine = comp_of[y]
            ahead = (y[0] + dr, y[1] + dc)           # push
            if ahead in comp_of and ahead not in moving:
                moving.add(ahead)
                stack.append(ahead)
            behind = (y[0] + br, y[1] + bc)          # drag
            tied_back = comp_of.get(behind) == mine
            if tied_back:
                if behind not in moving:
                    moving.add(behind)
                    stack.append(behind)
                continue        # `no nb_{-d}` guards the shear rule -- not this
            for (er, ec) in perps:                   # shear
                side = (y[0] + er, y[1] + ec)
                if comp_of.get(side) != mine:
                    continue                         # needs nb_e(y)
                if comp_of.get((side[0] + dr, side[1] + dc)) == mine:
                    continue                         # side is not its run's head
                if side not in moving:
                    moving.add(side)
                    stack.append(side)
        return moving

    def _shift(self, player, comps, moving, dr, dc):
        """Displace ``moving`` by ``(dr, dc)``, burn whatever entered fire, and
        re-split each component by geometry (a component whose cells were torn
        apart becomes two the moment the engine rebuilds its ``nb`` pointers)."""
        fires = self.fires
        out = []
        for comp in comps:
            cells = []
            for cell in comp:
                if cell in moving:
                    cell = (cell[0] + dr, cell[1] + dc)
                if cell not in fires:
                    cells.append(cell)
            if cells:
                out.extend(_split(cells))
        if player in moving:
            player = (player[0] + dr, player[1] + dc)
        if player in fires:
            player = None                            # [Player Fire] -> [Fire]
        return player, tuple(sorted(out))

    def _settle(self, player, comps):
        """Drop everything unsupported one cell at a time until nothing moves."""
        for _ in range(4 * self.h + 8):
            grounded = self._grounded(comps)
            falling = {cell for i, comp in enumerate(comps) if not grounded[i]
                       for cell in comp}
            if not falling:
                return player, comps
            player, comps = self._shift(player, comps, falling, 1, 0)
        raise RuntimeError("CrateBlob gravity did not settle")

    def _grounded(self, comps):
        """Per component: does anything hold it up? Support crosses components
        (``down [Crate | CantFall]`` has no ``nb`` guard) and spreads through the
        whole of one (the four ``nb`` rules do), so this is a fixpoint over
        components rather than over cells."""
        occ = {cell: i for i, comp in enumerate(comps) for cell in comp}
        grounded = [False] * len(comps)
        changed = True
        while changed:
            changed = False
            for i, comp in enumerate(comps):
                if grounded[i]:
                    continue
                for (r, c) in comp:
                    below = (r + 1, c)
                    # Off the bottom edge counts as held: the engine's seeding
                    # rule needs a Wall below and would otherwise loop forever
                    # trying to move a crate off the grid. Every level is
                    # wall-enclosed, so this branch is unreachable in play.
                    if below[0] >= self.h or below in self.walls \
                            or (below in occ and grounded[occ[below]]):
                        grounded[i] = changed = True
                        break
        return grounded

    # -- goal --------------------------------------------------------------
    def win(self, state):
        """``all Target on Crate`` at rest -- the other three win conditions
        (``no WillFall``, ``no Collision``, ``some DoneAnimating``) are exactly
        "the board has settled", which every state this model produces has."""
        cells = {cell for comp in state[1] for cell in comp}
        return self.targets <= cells


def _split(cells):
    """``cells`` -> its 4-connected components, each a sorted tuple."""
    rest = set(cells)
    out = []
    while rest:
        seed = rest.pop()
        comp = [seed]
        stack = [seed]
        while stack:
            r, c = stack.pop()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nxt = (r + dr, c + dc)
                if nxt in rest:
                    rest.discard(nxt)
                    comp.append(nxt)
                    stack.append(nxt)
        out.append(tuple(sorted(comp)))
    return out


# ---------------------------------------------------------------------------
# Model audit
# ---------------------------------------------------------------------------

def _fuzz_check(trials: int = 400, max_len: int = 24, seed: int = 1234) -> int:
    """Replay random press sequences through the interpreter AND `_Board` and
    compare after every single press. Returns the mismatch count.

    This is what earns the right to plan on the model instead of the engine. It
    compares the FULL state -- every component's cell set, the avatar's cell,
    whether the avatar is still alive -- and the win flag, not just the win flag:
    a model that drifts only in how a blob split still returns plans the
    certification will reject, and levels go silently missing.

    Run it after any edit to `_Board` (the engine is ~0.6 s per press, so the
    default is ~45 min; pass a smaller count while iterating):
        python solvers/generate_crateblob_training.py --fuzz [trials] [max_len]
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    rng = random.Random(seed)
    per_level = max(1, trials // game.n_levels)
    bad = rollouts = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board(eng, game._game)
        start = snapshot(eng)
        origin = board.read(eng)
        if origin != board._settle(*origin):
            bad += 1
            print(f"  MISMATCH level {level}: the level START is not settled")
        for _trial in range(per_level):
            restore(eng, start)
            state = origin
            rollouts += 1
            for _ in range(rng.randint(1, max_len)):
                direction = rng.choice(list(_DELTA))
                eng.step(direction)
                state = board.step(state, direction)
                if board.read(eng) != state:
                    bad += 1
                    print(f"  MISMATCH level {level} after {direction!r}: "
                          f"engine {board.read(eng)} != model {state}")
                    break
                if eng.check_win() != board.win(state):
                    bad += 1
                    print(f"  MISMATCH level {level} after {direction!r}: "
                          f"win {eng.check_win()} != {board.win(state)}")
                    break
                if eng.check_win():
                    break
        restore(eng, start)
    print(f"fuzz: {rollouts} rollouts, {bad} mismatches")
    return bad


# ---------------------------------------------------------------------------
# Macro search over the model
# ---------------------------------------------------------------------------

class _Planner:
    """A* (or a beam) whose successors are ``walk to cell y, then press d``.

    Walking inside your own blob has no effect on anything but the avatar's
    cell, so the only branch that matters is which cell you push FROM and in
    which direction -- that is the macro. ``g`` counts primitive presses (the
    walk plus the push), so an admissible heuristic still yields plans that are
    shortest in presses, not in pushes."""

    def __init__(self, board: _Board, kind: str, weight: int, node_cap: int):
        self.b = board
        self.kind = kind
        self.weight = weight
        self.node_cap = node_cap

    # -- heuristics --------------------------------------------------------
    def heuristic(self, state) -> int:
        cells = [cell for comp in state[1] for cell in comp]
        held = set(cells)
        loose = [t for t in self.b.targets if t not in held]
        if not loose:
            return 0
        if not cells:
            return _DEAD
        tables = self.b.dist
        if self.kind == "max":
            # Admissible: one press displaces any single crate by at most one
            # cell, so each target costs at least the wall-and-fire-avoiding
            # distance to its nearest crate -- and they cannot be summed, since
            # one press may serve all of them at once.
            worst = 0
            for t in loose:
                table = tables[t]
                near = min((table[cell] for cell in cells if cell in table),
                           default=None)
                if near is None:
                    return _DEAD             # no crate can ever reach it
                worst = max(worst, near)
            return worst
        # Not a bound, but the only thing that can tell drifting-toward from
        # drifting-away on the open levels: a greedy one-crate-per-target
        # assignment, nearest first.
        total = 0
        taken: set[int] = set()
        for t in loose:
            table = tables[t]
            best = best_i = None
            for i, cell in enumerate(cells):
                if i in taken:
                    continue
                d = table.get(cell)
                if d is not None and (best is None or d < best):
                    best, best_i = d, i
            if best is None:
                return _DEAD
            taken.add(best_i)
            total += best
        return total

    def dead(self, state) -> bool:
        """Provably lost: the avatar burned (nothing can move again), or fire has
        eaten so many crates that the targets can no longer all be covered."""
        if state[0] is None:
            return not self.b.win(state)
        return sum(len(comp) for comp in state[1]) < len(self.b.targets)

    # -- successors --------------------------------------------------------
    def macros(self, state):
        """``[(presses, next_state), ...]``: every cell of the avatar's own blob
        it can walk to, times every direction that is a push from there.

        The walk BFS is over ``board.dirs``, NOT all four: on the level that
        carries `NoVerticals` an up/down press is cancelled outright, so routing
        the avatar through one would emit a plan whose walks silently do nothing
        and whose pushes are then taken from the wrong cell."""
        player, comps = state
        comp_of = {cell: i for i, comp in enumerate(comps) for cell in comp}
        mine = comp_of[player]
        dirs = self.b.dirs
        walk = {player: []}
        queue = deque([player])
        while queue:
            u = queue.popleft()
            for d in dirs:
                dr, dc = _DELTA[d]
                v = (u[0] + dr, u[1] + dc)
                if comp_of.get(v) == mine and v not in walk:
                    walk[v] = walk[u] + [d]
                    queue.append(v)
        out = []
        for y, prefix in walk.items():
            for d in dirs:
                dr, dc = _DELTA[d]
                if comp_of.get((y[0] + dr, y[1] + dc)) == mine:
                    continue                         # that press is a walk
                nxt = self.b.step((y, comps), d)
                if nxt == (y, comps):
                    continue                         # the engine refuses it
                out.append((prefix + [d], nxt))
        return out

    # -- search ------------------------------------------------------------
    def astar(self, state):
        """Paths are kept as parent pointers rather than on the queue: these
        levels run to hundreds of thousands of states and a copied press list per
        heap entry costs more memory than the states themselves."""
        if self.b.win(state):
            return []
        w = self.weight
        counter = 0
        pq = [(w * self.heuristic(state), 0, counter, state)]
        best_g = {state: 0}
        came: dict = {state: None}
        nodes = 0
        while pq:
            _f, g, _c, node = heapq.heappop(pq)
            if best_g.get(node, 1 << 30) < g:
                continue
            for presses, nxt in self.macros(node):
                nodes += 1
                if nodes >= self.node_cap:
                    return None
                if self.b.win(nxt):
                    return _unwind(came, node) + presses
                if self.dead(nxt):
                    continue
                ng = g + len(presses)
                if best_g.get(nxt, 1 << 30) <= ng:
                    continue
                best_g[nxt] = ng
                came[nxt] = (node, presses)
                counter += 1
                heapq.heappush(pq, (ng + w * self.heuristic(nxt), ng, counter,
                                    nxt))
        return None

    def beam(self, state, width, depth, node_cap):    # noqa: C901
        """Width-capped breadth-first search over the same macros.

        For the levels where the heuristic goes FLAT across the middle of the
        solution -- the blob has to be folded into a shape before any crate gets
        closer to any target -- A* has nothing to steer with and degenerates to
        uniform cost at a depth it cannot reach. The beam spends the same budget
        on breadth at every depth and only uses the heuristic to break ties."""
        if self.b.win(state):
            return []
        frontier = [(state, [])]
        seen = {state}
        nodes = 0
        for _ in range(depth):
            kids = []
            for node, path in frontier:
                for presses, nxt in self.macros(node):
                    nodes += 1
                    if nodes >= node_cap:
                        return None
                    if self.b.win(nxt):
                        return path + presses
                    if nxt in seen or self.dead(nxt):
                        continue
                    seen.add(nxt)
                    kids.append((self.heuristic(nxt), len(path) + len(presses),
                                 nxt, path + presses))
            if not kids:
                return None
            kids.sort(key=lambda kid: (kid[0], kid[1]))
            frontier = [(nxt, path) for _h, _g, nxt, path in kids[:width]]
        return None


#: Charge for a state the search should never pop. Finite so the search stays
#: complete.
_DEAD = 1 << 20


def _unwind(came, node) -> list:
    """The press list that reached ``node``, from the parent-pointer chain."""
    out: list[str] = []
    while came[node] is not None:
        node, presses = came[node]
        out = presses + out
    return out


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class Plan(list):
    """The press sequence, carrying the optimal SET for each of its steps.

    ``optsets[i]`` is every press that is equally shortest at the state the i-th
    press is taken from -- computed once beside the plan, so a disk-cached plan
    replayed in a later process still labels every step without re-deriving
    anything."""

    optsets: list

    def __init__(self, presses, optsets):
        super().__init__(presses)
        self.optsets = optsets


class CrateBlobExpert(PSExpert):
    """Plans on `_Board` and certifies every plan on the interpreter.

    The plan memo, the level scoping and the snapshot/restore discipline are
    `PSExpert`'s; only the search strategy (`_search`) is replaced."""

    directions = ["up", "down", "left", "right"]     # nothing binds ACTION

    def setup(self) -> None:
        self._disk = self._load_disk()

    def heuristic(self, eng) -> int:
        """Unused: `_search` plans on the native model, so `PSExpert._astar` --
        the only caller -- never runs."""
        raise NotImplementedError("CrateBlobExpert plans on the native model")

    # -- planning ----------------------------------------------------------
    def _search(self, eng):
        board = _Board(eng, self.g)
        state = board.read(eng)
        if state[0] is None:
            return None
        # Each rung is strictly a fallback for the one above it -- the first is
        # provably shortest and the rest degrade in plan quality as they gain
        # reach -- so the first plan found is the best available and the ladder
        # stops there rather than paying tens of millions of expansions to
        # confirm what it already has.
        searches = [(kind, weight, min(cap, self.node_cap))
                    for kind, weight, cap in _LADDER]
        for kind, weight, cap in searches:
            planner = _Planner(board, kind, weight, cap)
            plan = planner.astar(state)
            if plan is None:
                continue
            # The model is fuzz-verified, so certification is insurance rather
            # than a filter -- but a rejected plan falls through to the next rung
            # instead of costing the level.
            certified = self._certify(eng, plan)
            if certified is not None:
                return Plan(certified, self._optimal_sets(board, state, certified))
        plan = _Planner(board, "sum", 1, self.node_cap).beam(state, *_BEAM)
        if plan is not None:
            certified = self._certify(eng, plan)
            if certified is not None:
                return Plan(certified, self._optimal_sets(board, state, certified))
        return None

    @staticmethod
    def _certify(eng, plan):
        """Replay ``plan`` through the real interpreter and return it truncated at
        the press that wins, or None if it never does. This is what makes the
        native model safe to plan on: a modelling slip costs a level, never a
        recorded trajectory that does not win."""
        snap = snapshot(eng)
        try:
            for i, direction in enumerate(plan):
                eng.step(direction)
                if eng.check_win():
                    return plan[:i + 1]
            return None
        finally:
            restore(eng, snap)

    # -- optimal-action targets --------------------------------------------
    @staticmethod
    def _optimal_sets(board, state, plan):
        """The optimal action SET at each step of ``plan``.

        A press is a WALK exactly when it is a press the level responds to at all
        AND the cell ahead of the avatar belongs to its own component -- the
        engine's ``nbR`` test, and the one thing that distinguishes a press with
        no side effect from a push. Consecutive walks are grouped, and each of
        their presses is labelled with every direction that stays on a shortest
        route inside the component to the cell the group ends on: all of those
        reach the same cell in the same number of presses and leave an identical
        board, so they are interchangeable. A push is labelled with itself.

        The ``board.dirs`` guard is not cosmetic: on the `NoVerticals` level an
        up/down press is CANCELLED, so it neither walks nor pushes, and calling
        it an equally-good alternative would label a no-op as correct play.

        Nothing here claims the PLAN is shortest (the ``sum`` passes are not
        bounds); the claim is only that these alternatives are interchangeable
        with the step actually taken."""
        sets: list[list[str]] = []
        dirs = board.dirs
        i, n = 0, len(plan)
        while i < n:
            player, comps = state
            comp_of = {cell: k for k, comp in enumerate(comps) for cell in comp}
            mine = comp_of[player]

            def is_walk(cell, d):
                if d not in dirs:
                    return False
                dr, dc = _DELTA[d]
                return comp_of.get((cell[0] + dr, cell[1] + dc)) == mine

            j, here = i, player
            while j < n and is_walk(here, plan[j]):
                dr, dc = _DELTA[plan[j]]
                here = (here[0] + dr, here[1] + dc)
                j += 1
            if j > i:
                # Distances to the cell the walk ends on, inside the component.
                dist = {here: 0}
                queue = deque([here])
                while queue:
                    u = queue.popleft()
                    for d in dirs:
                        dr, dc = _DELTA[d]
                        v = (u[0] + dr, u[1] + dc)
                        if comp_of.get(v) == mine and v not in dist:
                            dist[v] = dist[u] + 1
                            queue.append(v)
                literal = dist.get(player) != j - i    # not a shortest walk
                cur = player
                for k in range(i, j):
                    best = [plan[k]]
                    if not literal:
                        best = sorted(
                            d for d in dirs
                            if dist.get((cur[0] + _DELTA[d][0],
                                         cur[1] + _DELTA[d][1]),
                                        1 << 30) == dist[cur] - 1)
                        if plan[k] not in best:
                            best = [plan[k]]
                    sets.append(best)
                    dr, dc = _DELTA[plan[k]]
                    cur = (cur[0] + dr, cur[1] + dc)
            if j < n:
                sets.append([plan[j]])
            for k in range(i, min(j + 1, n)):
                state = board.step(state, plan[k])
            i = j + 1
        return sets

    # -- disk plan cache ---------------------------------------------------
    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan`, with the level's START plan (and its optimal sets)
        also cached on disk.

        Only that one state is worth keeping: every seed, and every process,
        plans from it and from nothing else, because recovery is a RESET back to
        it. The first entry stored for a level is therefore its start; a re-plan
        from a mid-level state never overwrites it."""
        if level is None:
            return super().plan(eng, level)
        sig = sorted([r, c, o] for (r, c, o) in self._key(eng))
        entry = self._disk.get(level)
        if entry is not None:
            if entry["start"] != sig:
                return super().plan(eng, level)
            return (Plan(entry["plan"], entry["optsets"])
                    if entry["plan"] is not None else None)
        found = super().plan(eng, level)
        self._disk[level] = {
            "start": sig,
            "plan": None if found is None else list(found),
            "optsets": None if found is None else found.optsets,
        }
        self._save_disk()
        return found

    @staticmethod
    def _load_disk() -> dict:
        """``{level: {"start": sig, "plan": [...] | None, "optsets": [...]}}``,
        or empty if unreadable -- a cache that cannot be parsed is a miss, never
        a crash."""
        try:
            raw = json.loads(PLAN_CACHE.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:                                    # noqa: BLE001
            return {}

    def _save_disk(self) -> None:
        """Write atomically -- shards started together would otherwise interleave
        into a truncated file (see `parallelize_generator`)."""
        try:
            PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PLAN_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self._disk[k] for k in sorted(self._disk)}, indent=1))
            os.replace(tmp, PLAN_CACHE)
        except OSError:
            pass                                             # cache is optional


class CrateBlobSolver(PSAStarSolver):
    game_id = "puzzlescript_crateblob"
    game_name = GAME_NAME
    expert_cls = CrateBlobExpert

    #: `weight` is unused -- `_LADDER` carries the weight of each search -- and
    #: `node_cap` is only an upper clamp on the per-search budgets there, so it
    #: sits at the largest of them.
    node_cap = 12_000_000
    weight = 1

    #: Room for the longest plan plus a re-plan after the exploration prefix;
    #: `epsilon` is 0 for this family, so nothing else lengthens a replay.
    max_steps = 400

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup
        rather than as a mysteriously slow first seed, and it fills the disk
        cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The full set of equally-shortest presses at this step -- see
        `CrateBlobExpert._optimal_sets`. Falls back to the press about to be
        taken, so no expert step ever ships unlabelled (the
        always-emit-optimal-targets rule)."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets):
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


def _report() -> int:
    """Per-level plan report: length, push count, and how many steps carry a real
    tie set."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = CrateBlobExpert(game, node_cap=CrateBlobSolver.node_cap)
    total = ties = 0
    for level in range(game.n_levels):
        if level in CrateBlobSolver.skip_levels:
            print(f"level {level}: skipped")
            continue
        game.set_level(level)
        plan = expert.plan(game._engine, level)
        if plan is None:
            print(f"level {level}: UNSOLVED")
            continue
        sets = getattr(plan, "optsets", []) or []
        tied = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        ties += tied
        print(f"level {level}: {len(plan):3d} presses, "
              f"{tied:3d} with a tie set")
    print(f"total {total} presses, {ties} tied")
    return 0


if __name__ == "__main__":
    if "--fuzz" in sys.argv:
        rest = [int(a) for a in sys.argv[sys.argv.index("--fuzz") + 1:]
                if a.isdigit()]
        sys.exit(1 if _fuzz_check(*rest) else 0)
    if "--plans" in sys.argv:
        sys.exit(_report())
    sys.exit(CrateBlobSolver.main())
