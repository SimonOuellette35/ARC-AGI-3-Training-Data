"""Generate Phase-1 training data for the PuzzleScript game ps:kicking_walls
("Kicking walls" by Jere Majava).

The harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: its name, the edge-wall /
kick model its search needs, and its goal heuristic.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_kicking_walls",
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
exactly.

THE GAME (it looks like a sokoban and it is not one)
----------------------------------------------------
Four levels, one player, crates to be pushed onto targets, ``All Target on
Crate``. Two things make it its own mechanic, and a stock `PSSokobanExpert`
gets BOTH of them wrong.

**1. The walls are EDGES, not cells.** ``WallH`` and ``WallV`` sit IN a cell
(stacked with whatever else is there -- a crate, a target, the player) and each
one blocks exactly one boundary of it:

    WallH in (r, c)  blocks  (r, c) <-> (r+1, c)      (its sprite is the cell's
                                                       bottom pixel row)
    WallV in (r, c)  blocks  (r, c) <-> (r, c-1)      (its sprite is the cell's
                                                       left pixel column)

which is what the four blocking rules say, once in each direction per axis::

    up    [ > Movable | WallH ] -> [ Movable | WallH ]   ( into a WallH cell )
    down  [ > Movable WallH | ] -> [ Movable WallH | ]   ( out of one )
    left  [ > Movable WallV | ] -> [ Movable WallV | ]
    right [ > Movable | WallV ] -> [ Movable | WallV ]

So NOTHING blocks a cell: the player walks onto wall cells all day, and the
level's ``L``/``A``/``F`` legend entries are literally "a crate that also has a
wall on its left" and so on. `PSSokobanExpert`'s whole world model is cell
blockers (`blocker_names`), so it is set EMPTY here and the adjacency is
supplied instead by `KickingWallsExpert._open`.

**2. The kick.** The push rule fires BEFORE the blocking rules::

    [ > Player | Crate ] -> [ > Player | > Crate ]

so by the time the wall rules run, the crate is already a ``> Movable`` in its
own right -- and each of the four rules tests one mover against one boundary.
The player's boundary (S <-> P) and the crate's (P <-> X) are DIFFERENT
boundaries, tested independently. A wall between the player and the crate
therefore stops the player and NOT the crate: the crate flies over the wall and
the player stays where it stood. That is the title, and it is not a detail --
level 1's whole solution turns on it. Concretely (level 1, step 21 of the
recorded plan) the player at (1,1) presses RIGHT into (1,2), which carries a
WallV on its left edge; the player is cancelled, the crate at (1,2) sails to
(1,3), and that crate could not have been moved any other way.

The search itself steps the real interpreter, so it gets the kick for free. The
two places the mechanic has to be MODELLED are the two static maps
`KickingWallsExpert` derives from the board, and both live in that class:

  * the push-distance tables (`_build_tables`), whose relaxation must drop the
    standard sokoban test "the player can get behind the crate" and keep only
    "the stand cell is on the board" -- a wall behind a crate does not stop it
    being pushed, only being followed;
  * the player's walk graph (`_walk_distances`), which steps over BOUNDARIES
    rather than onto free cells, since every cell is free and only crates are
    in the way.

Get the first backwards and nothing raises: the tables call the crate frozen,
the heuristic charges `DEAD_COST`, and a solvable level looks unsolvable. That
is what the first cut of this file did on three of the four levels while
cheerfully solving the first.

Level 1 is also where the mechanic is TAUGHT: its crate at (2,2) carries both a
WallH and a WallV, so three of its four pushes are dead and the only way it ever
moves is a kick upward through a wall the player cannot cross.

THE LEVELS
----------
  * **Level 0** ("One of these again? This is getting tedious...") -- 6x5. One
    loose crate, one bare target, and a wall maze that turns four cells of
    separation into 33 moves.
  * **Level 1** ("Wait... What?") -- 6x5, the same board plus a second target
    and the double-walled crate above. 35 moves. This is the reveal level.
  * **Level 2** ("Such a mess") -- 6x6, five crates and five targets, so every
    crate is committed. 41 moves.
  * **Level 3** ("Last one..") -- 5x8, seven crates for six targets, so one
    crate is spare and the search has to work out which. 38 moves.

EXPERT SOLVER
-------------
`PSSokobanExpert` -- its per-target PUSH-DISTANCE tables as the heuristic (and,
for free, as the deadlock test: a crate whose cell is in no target's table can
never reach any target, so its subtree sinks to the back of the queue) -- with
the tables and the walk graph re-derived over the edge/kick model above, and
with the SEARCH swapped back to `PSExpert`'s primitive A* over an exact
``(player, crates)`` key.

That last swap is the one interesting choice, because it undoes the thing the
class exists for. `PSPushExpert`'s macros pay for themselves by deduping on the
player's reachable REGION rather than its cell, an approximation that can
mis-cost a state by the region's diameter -- and on these four boards (30 to 48
cells, the player free to roam nearly all of them) that diameter is most of the
board. It showed: the macro search returned 43 presses on level 2 against a
proven optimum of 41. The boards are also small enough that the depth reduction
macros buy is simply not needed. The primitive search solves all four in ~220s
together, once, into `PLAN_CACHE`.

The plans are engine-verified: ``--plans`` replays each one and requires the
interpreter to win on the LAST press and no earlier. Levels 0, 1 and 2 are
additionally certified SHORTEST against an independent primitive BFS over the
interpreter that shares nothing with the A* -- ``--verify`` prices them at
33 / 35 / 41 over 571 / 2336 / 59622 states and agrees with all three. Level 3
is the one plan with no independent certificate: a BFS to its nearest win runs
past 150k states, so ``--verify`` prints it UNPRICED rather than claiming it.

THE OPTIMAL SETS, and the one place they are conservative. Every step of every
plan carries a set, and every press in every set is genuinely optimal --
``--verify`` checks that as a HARD failure. Walk steps get every direction that
stays on a shortest route to the cell the next push is taken from. Push steps
are labelled with themselves alone, which is `annotate_walks`' standard
sokoban reading ("which crate you shove where IS the puzzle, so a sibling push
is a different plan, not a reordering of this one") -- and here it is
occasionally too strong, because the KICK decouples a push from the player
moving, so a kick can genuinely tie with a plain step. ``--verify`` measures
exactly that against an exhaustive field and reports it: over levels 0 and 1
(68 presses, priced to the last state) it is ONE step. It is a narrowing, never
an error -- the label is always a shortest-path press.

Measuring it away is not affordable, and the two ways of trying were both
checked rather than guessed. `PSPushExpert.exact_optsets` re-solves per
candidate and leans on the heuristic to prune the hopeless ones; this
heuristic is too loose for that (it counts pushes, while most of a plan here is
the player walking a maze -- its gap to the true distance averages 12 of ~38
presses), so 85 to 107 candidates per level survive the prune and each wants a
full A*, which is hours on levels 2 and 3. And pricing an exhaustive field per
level, the way ``--verify`` does for levels 0 and 1, needs the WHOLE reachable
space rather than the part a BFS crosses on its way to a win: level 2 reaches
its first win at 59622 states but a standalone sweep of it was still running
past 450k after twelve minutes, and level 3 is bigger again.

AUGMENTATION
------------
The engine state after reset is identical for every seed (the levels are fixed
ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group -- 16 presentations per level instead of 4, which
matters a lot when the game is four levels long.

The flips are sound here by construction rather than by luck. The four blocking
rules name absolute directions, but they name them in AXIS-SYMMETRIC PAIRS: one
rule for entering a WallH cell and one for leaving one, which are the same
boundary stated from its two sides, and likewise for WallV. A mirror maps that
boundary to a boundary and the sprite that draws it to the same edge of the
mirrored cell, so the picture stays a faithful picture of the rules. And there
is no rule-order chirality to argue around (the shape that Gobble Rush and
I'm Sick Today have to measure their way out of): one player, one force per
crate, and no rule in the game where two matches can contest a cell. ``--symmetry``
measures it anyway over a bounded slice of each level's reachable space.

The game takes no colour augmentation: crate-on-target is read as the
``Happycrate`` recolour (uniform orange) against a plain crate (orange over
maroon), which is the only cue there is -- the crate sprite covers the target
sprite completely -- and a flattening recolour would erase it. See ``--audit``.

VERIFICATION MODES (run from the repo root, with the conda python)
------------------------------------------------------------------
    python solvers/generate_kicking_walls_training.py --plans     # engine-certify
    python solvers/generate_kicking_walls_training.py --fuzz      # model vs engine
    python solvers/generate_kicking_walls_training.py --verify    # shortest + ties
    python solvers/generate_kicking_walls_training.py --symmetry  # flip safety
    python solvers/generate_kicking_walls_training.py --audit     # rendering
    python solvers/generate_kicking_walls_training.py --replay    # record + replay

Usage (run from the repo root):
    python solvers/generate_kicking_walls_training.py --episodes 200 \
        --out data/training_multi_level/kicking_walls
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from arcengine import ActionInput, GameAction, GameState         # noqa: E402
# `GameAction(5)` does NOT work: the enum is keyed by (value, action class), so
# a recorded index has to be looked up through this map, not constructed.
from solvers.base_solver import _ID_TO_GAMEACTION                # noqa: E402
from adapters.puzzlescript_adapter import _render_frame          # noqa: E402
from solvers.common.ps_astar import (                            # noqa: E402
    _DELTA, Plan, PSAStarSolver, PSExpert, PSSokobanExpert, restore, snapshot)

GAME_NAME = "Kicking_walls"

#: Where each level's START plan (and its optimal sets) is kept between
#: processes. The searches are the whole cost of generation here -- tens of
#: seconds per level, and seed-independent -- so without a file on disk every
#: shard `parallelize_generator` starts would re-derive all four.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "kicking_walls_plans.json"

#: The four presses, in the order every report in this file uses.
_ORDER = ("up", "down", "left", "right")


class KickingWallsExpert(PSSokobanExpert):
    """`PSSokobanExpert` re-derived over EDGE walls and the kick.

    The shared class assumes the two things this game does not have: that an
    obstacle occupies a CELL, and that a push needs the player to be able to
    step into the cell it is pushing out of. Both assumptions live in exactly
    two derived maps -- the per-target push-distance tables (`_build_tables`,
    which are the heuristic AND the deadlock test) and the player walk-distance
    field the tie sets are read off (`_walk_distances`) -- so those two are
    overridden here against `_open`. The memo, the plan cache, the deadlock
    argument and the heuristic arithmetic are inherited untouched, and the
    SEARCH is swapped for the primitive one in `_search`.

    `_open(r, c, dr, dc)` is the whole model: is the boundary between (r, c)
    and (r + dr, c + dc) crossable. It reads the two wall layers, which no rule
    in the game ever creates or destroys, so it is bound ONCE per board in
    `plan` rather than re-scanned per node.

    (This is the first ps: game in the tree whose walls are edges. If a second
    one turns up, these overrides are what should move into
    `solvers/common/ps_astar.py` as a passability hook -- adding the hook now
    would put a per-edge indirection into the hot BFS of forty generators that
    have no use for it.)
    """

    pushable_names = ("crate",)
    target_names = ("target",)
    #: EMPTY on purpose: nothing in this game blocks a cell, the walls block
    #: boundaries (see the module docstring). A crate still blocks its own
    #: cell, which the interpreter enforces and `_walk_distances` models from
    #: `push_ids`.
    blocker_names = ()

    #: Label each walk step with every direction that keeps it on a shortest
    #: route to the next push cell, instead of training one arbitrary
    #: interleaving of the two axes as the only right answer.
    #:
    #: Pushes stay labelled with themselves alone. That is `annotate_walks`'
    #: sokoban reading, and in THIS game it is occasionally conservative: a
    #: kick moves a crate without moving the player, so it can tie with a plain
    #: step in a way a normal sokoban push cannot. ``--verify`` prices that
    #: exactly where a field fits and finds one such step in levels 0 and 1
    #: together. It is left as is deliberately -- see the module docstring for
    #: the two measured reasons `exact_optsets` and a per-level field are both
    #: unaffordable here -- and the labels stay sound either way: a narrowed
    #: set is still a set of shortest-path presses.
    annotate = True

    #: Keep each level's start plan (and its optimal sets) on disk.
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        super().setup()
        self.wallh_id = self.g.obj_name_to_idx["wallh"]
        self.wallv_id = self.g.obj_name_to_idx["wallv"]
        self._walls: tuple | None = None

    # -- the edge model -------------------------------------------------------
    def bind_walls(self, eng) -> tuple:
        """Read the two wall layers off the grid and bind them for the search.

        Called from `plan` (and by the report modes, which search without going
        through it). No rule creates or destroys a wall, so this is a static
        per-board fact and must not be re-scanned inside `heuristic`, which
        runs on every node."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        wh = tuple(tuple(self.wallh_id in grid[r][c] for c in range(w))
                   for r in range(h))
        wv = tuple(tuple(self.wallv_id in grid[r][c] for c in range(w))
                   for r in range(h))
        self._walls = (h, w, wh, wv)
        return self._walls

    def _open(self, r: int, c: int, dr: int, dc: int) -> bool:
        """Can a mover cross from (r, c) to (r + dr, c + dc)?

        False off the board, and False when the boundary carries its wall:
        ``WallH`` in the UPPER of the two cells for a vertical boundary,
        ``WallV`` in the RIGHTER of the two for a horizontal one -- which is
        where each sprite is drawn (bottom pixel row / left pixel column)."""
        h, w, wh, wv = self._walls
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return False
        if dr == 1:
            return not wh[r][c]
        if dr == -1:
            return not wh[nr][c]
        if dc == 1:
            return not wv[r][nc]
        return not wv[r][c]

    def plan(self, eng, level: int | None = None):
        self.bind_walls(eng)
        return super().plan(eng, level)

    # -- the search -----------------------------------------------------------
    def _search(self, eng):
        """`PSExpert`'s PRIMITIVE A*, not `PSPushExpert`'s macro one -- plus the
        same inferred optimal sets the macro version would have produced.

        WHY THE PRIMITIVE SEARCH WINS HERE, when the whole point of
        `PSPushExpert` is that it does not. The macro search dedups by the
        player's REACHABLE REGION rather than by its cell, which is what makes
        branching on pushes pay, and which can mis-cost a state by up to the
        region's diameter. These four boards are 30 to 48 cells with the player
        free to roam nearly all of them, so that diameter is most of the board:
        on level 2 the macro search returned 43 presses against a proven
        optimum of 41. And they are SMALL -- level 2's whole reachable space is
        ~60k states -- so the thing the macro abstraction buys, depth in the
        number of pushes rather than of presses, is not needed. An exact
        (player, crates) key with the same push-distance heuristic solves all
        four in ~220s together (0.4 / 1.9 / 41 / 174), once, into `PLAN_CACHE`,
        and gets level 2 right.

        Everything else about `PSSokobanExpert` is kept: the heuristic, the
        tables that are also the deadlock test, the dynamic-only `_key` (hence
        `scope_by_level`), and `annotate_walks`, which needs no macro structure
        -- it re-reads the plan off the board, calling a step that moved a
        crate a push and a step that moved only the player a walk, and labels
        each maximal walk run with every direction that keeps it on a shortest
        route to the cell the next push is taken from."""
        start = snapshot(eng)
        found = PSExpert._astar(self, eng)
        restore(eng, start)
        if found is None:
            return found
        return Plan(found, self.annotate_walks(eng, found))

    # -- per-target push-distance tables --------------------------------------
    def _build_tables(self, eng) -> dict:
        """``{target: {cell: pushes to get a crate from cell onto target}}``.

        Reverse BFS from the target over push edges, ignoring the other crates
        (the standard sokoban relaxation), with the game's two changes: the
        crate's own boundary ``P -> X`` must be open, and the player's stand
        cell ``S = P - d`` only has to EXIST. There is deliberately no test
        that the player can cross ``S -> P``, because the kick does not need
        it -- putting one back is exactly the bug that made three of the four
        levels look unsolvable.

        A cell that lands in no target's table is a cell a crate can never
        leave for anywhere useful, which is the deadlock test `heuristic`
        charges `DEAD_COST` for."""
        h, w, wh, wv = self._walls
        grid = eng.grid
        targets = tuple((r, c)
                        for r in range(h) for c in range(w)
                        if grid[r][c] & self.target_ids)
        sig = (wh, wv, targets)
        cached = self._table_cache.get(sig)
        if cached is not None:
            return cached

        tables = {}
        for target in targets:
            steps = {target: 0}
            queue = deque([target])
            while queue:
                r, c = queue.popleft()
                cost = steps[(r, c)]
                for dr, dc in _DELTA.values():
                    pr, pc = r - dr, c - dc          # where the crate comes from
                    sr, sc = pr - dr, pc - dc        # where the player stands
                    if not (0 <= pr < h and 0 <= pc < w):
                        continue
                    if not (0 <= sr < h and 0 <= sc < w):
                        continue
                    if not self._open(pr, pc, dr, dc):
                        continue
                    if (pr, pc) not in steps:
                        steps[(pr, pc)] = cost + 1
                        queue.append((pr, pc))
            tables[target] = steps
        self._table_cache[sig] = tables
        return tables

    def _walk_distances(self, eng, target) -> dict:
        """BFS distance to ``target`` over the cells the player may WALK on:
        any cell without a crate, reachable across open boundaries."""
        grid = eng.grid
        block = self.push_ids
        dist = {target: 0}
        queue = deque([target])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _DELTA.values():
                nxt = (r + dr, c + dc)
                if (self._open(r, c, dr, dc) and nxt not in dist
                        and not (grid[nxt[0]][nxt[1]] & block)):
                    dist[nxt] = dist[(r, c)] + 1
                    queue.append(nxt)
        return dist

    # -- the model, stated once, for --fuzz ----------------------------------
    def predict(self, player, crates, direction):
        """The model's own answer for one press: ``(player, crates)`` after it.

        This is not used by the search -- the search steps the interpreter --
        it is the statement of the mechanic that ``--fuzz`` holds the
        interpreter to, so that the assumptions `_build_tables` and
        `_walk_distances` are built on are checked rather than believed.

        ``player`` is a cell, ``crates`` a frozenset of cells."""
        dr, dc = _DELTA[direction]
        p = (player[0] + dr, player[1] + dc)
        h, w, _wh, _wv = self._walls
        if not (0 <= p[0] < h and 0 <= p[1] < w):
            return player, crates
        if p not in crates:
            return (p if self._open(player[0], player[1], dr, dc) else player), crates
        x = (p[0] + dr, p[1] + dc)
        moved = (self._open(p[0], p[1], dr, dc) and x not in crates)
        if not moved:
            # The crate stays, so it blocks the player too (same collision
            # layer) whatever the wall between them says.
            return player, crates
        crates = (crates - {p}) | {x}
        # ...and the player follows it only if its OWN boundary is open. When
        # it is not, that is the kick: the crate flew over a wall the player
        # cannot cross.
        return (p if self._open(player[0], player[1], dr, dc) else player), crates


class KickingWallsSolver(PSAStarSolver):
    game_id = "puzzlescript_kicking_walls"
    game_name = GAME_NAME
    game_module_id = "ps:kicking_walls"
    expert_cls = KickingWallsExpert

    #: Unweighted: a weight above 1 trades the shortest plan away, and the four
    #: searches together are a ~220s one-off cached to `PLAN_CACHE`.
    weight = 1
    #: The primitive search snapshots the grid per queued node and level 3 peaks
    #: around 350k of them, so this is a MEMORY backstop as much as a time one.
    #: It is far above what any of the four levels needs.
    node_cap = 2_000_000
    #: Plans are 33-44 moves; the ceiling only has to cover a re-plan after an
    #: epsilon detour, and `epsilon` is 0 for this family (recovery data comes
    #: from the RESET prefix).
    max_steps = 150


# ---------------------------------------------------------------------------
# Reporting / verification
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with the plan memo warm, for the report modes."""
    solver = KickingWallsSolver()
    game, expert, _solvable = solver._ensure(0)
    return solver, game, expert


def _read(eng, expert):
    """``(player, crates)`` off the engine grid, as `predict` wants them."""
    player = None
    crates = set()
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if cell & expert.player_ids:
                player = (r, c)
            if cell & expert.push_ids:
                crates.add((r, c))
    return player, frozenset(crates)


def _base_grid(eng, expert):
    """The level's static scenery: the grid with player, crates and the derived
    Happycrates stripped out. `_seat` rebuilds a state on top of it, which is
    how the bounded sweeps below hold a reachable space in memory as bare
    frozensets instead of as one grid snapshot per state (the version that did
    keep snapshots is what took the machine down)."""
    dyn = expert.dyn_ids | {expert.g.obj_name_to_idx["happycrate"]}
    return [[set(cell) - dyn for cell in row] for row in eng.grid]


def _seat(eng, base, player, crates, expert):
    """Put ``(player, crates)`` back on ``base``. Happycrate is not seated: it
    is a pure function of Crate-on-Target and the engine's own late rules
    restore it during the step, and ``check_win`` reads Target/Crate anyway."""
    eng.grid = [[set(cell) for cell in row] for row in base]
    pid = min(expert.player_ids)
    cid = min(expert.push_ids)
    eng.grid[player[0]][player[1]].add(pid)
    for (r, c) in crates:
        eng.grid[r][c].add(cid)
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _reachable(eng, base, expert, start, cap):
    """Up to ``cap`` states of the reachable space, BFS from ``start``.

    Bounded on purpose, and the bound is REPORTED by every caller: an unbounded
    sweep of level 3 is not affordable on this machine."""
    seen = {start}
    order = [start]
    queue = deque([start])
    truncated = False
    while queue:
        state = queue.popleft()
        for direction in _ORDER:
            _seat(eng, base, state[0], state[1], expert)
            eng.step(direction)
            if eng.check_win():
                continue
            nxt = _read(eng, expert)
            if nxt in seen:
                continue
            if len(seen) >= cap:
                truncated = True
                queue.clear()
                break
            seen.add(nxt)
            order.append(nxt)
            queue.append(nxt)
    return order, truncated


def _plans() -> int:
    """Certify every level's plan ON THE INTERPRETER: it must win on the LAST
    press and on no earlier one (an earlier win would mean the plan is not even
    minimal, let alone shortest), and it must fit inside `max_steps`."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        t = time.time()
        plan = expert.plan(eng, level)
        took = time.time() - t
        if plan is None:
            print(f"level {level:2d}: UNSOLVABLE")
            bad += 1
            continue
        optsets = getattr(plan, "optsets", None)
        game.set_level(level)
        won_at = None
        for i, direction in enumerate(plan):
            eng.step(direction)
            if eng.check_win():
                won_at = i
                break
        ok = won_at == len(plan) - 1
        labelled = optsets is not None and len(optsets) == len(plan) and all(
            optsets[i] and plan[i] in optsets[i] for i in range(len(plan)))
        ties = 0 if optsets is None else sum(1 for s in optsets if len(s) > 1)
        room = "ok" if len(plan) <= KickingWallsSolver.max_steps else "OVER BUDGET"
        bad += (not ok) + (not labelled)
        print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d}, "
              f"{len(plan):3d} presses (budget {KickingWallsSolver.max_steps}, "
              f"{room}), {ties:3d} tie steps, searched in {took:6.1f}s: "
              f"{'CERTIFIED' if ok else f'WON AT {won_at}, NOT THE LAST PRESS'}"
              f"{'' if labelled else ', OPTIMAL SETS MISSING OR INCONSISTENT'}")
        total += len(plan)
    print(f"total {total} presses over {game.n_levels} levels")
    return 0 if not bad else 1


def _fuzz(cap: int = 30_000) -> int:
    """Hold the INTERPRETER to `KickingWallsExpert.predict`, state by state.

    The search never uses `predict` -- it steps the real engine -- but it does
    use `_build_tables` and `_walk_distances`, and those are built on exactly
    the model `predict` states. If the model is wrong the search does not crash, it
    quietly enumerates the wrong macros and reports a level unsolvable (which
    is precisely how the edge/kick reading of the rules was found: three of the
    four levels came back None). So: over a bounded slice of each level's
    reachable space, every state x every press, the engine's answer must be the
    model's answer."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        expert.bind_walls(eng)
        base = _base_grid(eng, expert)
        start = _read(eng, expert)
        t = time.time()
        states, truncated = _reachable(eng, base, expert, start, cap)
        notes = []
        for state in states:
            for direction in _ORDER:
                _seat(eng, base, state[0], state[1], expert)
                eng.step(direction)
                got = _read(eng, expert)
                want = expert.predict(state[0], state[1], direction)
                if got != want:
                    notes.append(f"{state[0]}+{direction}: engine {got[0]} "
                                 f"{sorted(got[1])} != model {want[0]} "
                                 f"{sorted(want[1])}")
        bad += len(notes)
        print(f"level {level:2d}: {len(states):6d} states"
              f"{' (CAPPED)' if truncated else ''} x 4 presses in "
              f"{time.time() - t:5.1f}s: "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}")
    print("fuzz clean" if not bad
          else f"FUZZ FAILED: {bad} transitions the model gets wrong")
    return 0 if not bad else 1


#: Sentinel successor: this press wins outright, so it is at distance 0.
_WIN = -1


def _shortest(eng, base, expert, start, cap):
    """``(d, states, truncated)`` -- the shortest press count from ``start`` to
    a win, by a plain forward BFS over the interpreter that STOPS at the first
    win it reaches.

    Much cheaper than `_field`, which cannot stop there because it needs a
    distance for every state, not just the start's: level 2 reaches a win after
    ~60k states and has well over 120k in total. So the plan LENGTH can be
    certified on a level whose tie sets cannot be priced."""
    seen = {start}
    queue = deque([(start, 0)])
    while queue:
        state, d = queue.popleft()
        for direction in _ORDER:
            _seat(eng, base, state[0], state[1], expert)
            eng.step(direction)
            if eng.check_win():
                return d + 1, len(seen), False
            nxt = _read(eng, expert)
            if nxt in seen:
                continue
            if len(seen) >= cap:
                return None, len(seen), True
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None, len(seen), False


def _field(eng, base, expert, start, cap):
    """``(order, succ, dist, truncated)`` -- the EXACT distance-to-win field
    over the reachable space, or ``truncated`` when it does not fit in ``cap``.

    One forward enumeration records every state's four successors (``_WIN``
    for a press that wins); one backward BFS over the reverse edges then prices
    every state at once. That is ``|S| * 4`` interpreter steps for the whole
    field, against one full BFS per probe if each question were asked
    separately -- on level 2 the difference is two minutes against several
    hours, which is the only reason ``--verify`` can price its tie sets at all.
    """
    index = {start: 0}
    order = [start]
    succ: list[list[int]] = []
    i = 0
    while i < len(order):
        state = order[i]
        row = []
        for direction in _ORDER:
            _seat(eng, base, state[0], state[1], expert)
            eng.step(direction)
            if eng.check_win():
                row.append(_WIN)
                continue
            nxt = _read(eng, expert)
            j = index.get(nxt)
            if j is None:
                if len(order) >= cap:
                    return order, succ, {}, True
                j = len(order)
                index[nxt] = j
                order.append(nxt)
            row.append(j)
        succ.append(row)
        i += 1

    preds: list[list[int]] = [[] for _ in order]
    dist: dict[int, int] = {}
    queue = deque()
    for s, row in enumerate(succ):
        for j in row:
            if j == _WIN:
                if s not in dist:
                    dist[s] = 1
                    queue.append(s)
            else:
                preds[j].append(s)
    while queue:
        s = queue.popleft()
        for p in preds[s]:
            if p not in dist:
                dist[p] = dist[s] + 1
                queue.append(p)
    return order, succ, dist, False


def _verify(walk_cap: int = 150_000, field_cap: int = 120_000) -> int:
    """Re-price each plan, and each of its tie sets, from the other end.

    The plan comes out of an A* whose heuristic is a greedy matching over the
    push-distance tables -- a search GUIDE, not an admissibility certificate --
    and its optimal sets come out of `annotate_walks`, which INFERS them from
    the shape of the plan rather than measuring them. Neither is checked at all
    by ``--plans``, which only certifies that the plan wins on its last press.

    So both are recomputed by something that shares nothing with either, and
    they are recomputed SEPARATELY because they cost different amounts:

      * the plan LENGTH, by `_shortest` -- a plain forward BFS over the
        interpreter that stops at the first win. Levels 0, 1 and 2 fit
        (627 / 2544 / ~60k states) and are genuinely certified shortest.
      * the TIE SETS, by `_field` -- the exhaustive space plus a backward BFS
        from the winning transitions, which prices every state exactly. This
        cannot stop at the first win, so it needs the WHOLE space, and only
        levels 0 and 1 fit.

    Anything that does not fit is printed UNPRICED rather than silently
    skipped. Sweeping level 3 unbounded is what took this machine down once, so
    do not raise either cap without watching memory."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        expert.bind_walls(eng)
        base = _base_grid(eng, expert)
        plan = expert.plan(eng, level)
        optsets = getattr(plan, "optsets", [[d] for d in plan])

        t = time.time()
        game.set_level(level)
        start = _read(eng, expert)
        star, seen, walk_trunc = _shortest(eng, base, expert, start, walk_cap)
        if walk_trunc:
            print(f"level {level:2d}: UNPRICED -- a BFS to the nearest win "
                  f"exceeds the {walk_cap} state cap ({time.time() - t:5.1f}s)")
            continue
        notes = []
        if star != len(plan):
            notes.append(f"LENGTH {star} != {len(plan)}")

        order, succ, dist, truncated = _field(eng, base, expert, start,
                                              field_cap)
        if truncated:
            bad += len(notes)
            print(f"level {level:2d}: d*={star:3d} by a {seen}-state BFS, "
                  f"{'AGREES' if not notes else '; '.join(notes[:3])} -- tie "
                  f"sets UNPRICED, the full space exceeds the {field_cap} "
                  f"state cap ({time.time() - t:5.1f}s)")
            continue

        narrowed = 0
        s = 0
        crates = start[1]
        for i, direction in enumerate(plan):
            here = dist.get(s)
            want = [a for k, a in enumerate(_ORDER)
                    if (succ[s][k] == _WIN and here == 1)
                    or (succ[s][k] != _WIN and dist.get(succ[s][k]) is not None
                        and dist[succ[s][k]] + 1 == here)]
            got = optsets[i]
            nxt = succ[s][_ORDER.index(direction)]
            pushed = nxt == _WIN or order[nxt][1] != crates
            if set(got) - set(want) or direction not in got:
                # HARD: a label that is not optimal would train a wrong press,
                # and a set that does not contain what the expert did is a
                # bookkeeping bug either way.
                notes.append(f"step {i}: {got} not within {want}")
            elif set(want) - set(got):
                if pushed:
                    # ACCEPTED: `annotate_walks` labels a step that moved a
                    # crate with itself alone. See the note in
                    # `KickingWallsExpert`. Narrower than the truth, never
                    # wrong.
                    narrowed += 1
                else:
                    notes.append(f"step {i}: WALK {got} narrower than {want}")
            if nxt == _WIN:
                break
            s = nxt
            crates = order[s][1]
        bad += len(notes)
        print(f"level {level:2d}: d*={star:3d} by a {seen}-state BFS, "
              f"{sum(len(x) for x in optsets):3d} labelled presses re-priced "
              f"over an exhaustive {len(order)}-state field in "
              f"{time.time() - t:6.1f}s: "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}"
              f"{f' ({narrowed} push steps narrower than truth)' if narrowed else ''}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _symmetry(cap: int = 4000) -> int:
    """Measure that no transition is decided by the way the board is FACING in
    a way a MIRROR would expose and the mandatory rotation would not.

    This is the evidence behind putting the game in `_FLIP_GAMES`. The argument
    in the module docstring says there is nothing to find -- the blocking rules
    name absolute directions but name them in axis-symmetric pairs, and nothing
    in the game lets two matches contest a cell -- so this is the check that the
    argument is true of the interpreter and not just of the source.

    Over a bounded slice of each level's reachable space: step the board, then
    step each of the 8 turned/mirrored copies of it with the correspondingly
    turned press, and un-transform the result. A transition is CHIRAL when any
    copy disagrees, MIRROR-ONLY when the three rotations agree and a mirror does
    not. Zero mirror-only transitions is the claim."""
    _solver, game, expert = _levels()
    eng = game._engine
    ids = expert.g.obj_name_to_idx
    keep = (ids["player"], ids["crate"], ids["wallh"], ids["wallv"],
            ids["target"])

    def cells(grid):
        """The mechanically live objects, as ``(r, c, o)``.

        BORDER walls are dropped. A ``WallV`` in column 0 fences the boundary
        between the board and (r, -1) and a ``WallH`` in the last row fences
        the one to (h, c) -- boundaries no mover can cross anyway, so both are
        pure decoration, and both are UNREPRESENTABLE once turned: rotating the
        left border a quarter turn asks for a WallH at row -1, and writing it
        into row 0 instead would fence a real interior boundary and report a
        chirality the game does not have."""
        h = len(grid)
        wallh, wallv = ids["wallh"], ids["wallv"]
        return frozenset((r, c, o)
                         for r, row in enumerate(grid)
                         for c, cell in enumerate(row)
                         for o in cell
                         if o in keep
                         and not (o == wallv and c == 0)
                         and not (o == wallh and r == h - 1))

    def turn(cs, k, mirror, h, w):
        """``cs`` under mirror-then-k-quarter-turns-clockwise, with the board's
        height/width following along.

        WallV is drawn on its cell's LEFT edge and WallH on its BOTTOM edge, so
        a turn does not just move a wall, it RE-SEATS it: the boundary is what
        transforms, and the cell that carries it is whichever of the pair the
        new orientation puts the sprite in. That is what the wall branches
        below do -- turn the boundary, then name it from the correct side."""
        out = set()
        wallh, wallv = ids["wallh"], ids["wallv"]

        def move(r, c):
            rr, cc, hh, ww = r, (w - 1 - c) if mirror else c, h, w
            for _ in range(k):
                rr, cc, hh, ww = cc, hh - 1 - rr, ww, hh
            return rr, cc

        for (r, c, o) in cs:
            if o == wallh:                       # boundary (r,c) <-> (r+1,c)
                a, b = move(r, c), move(r + 1, c)
            elif o == wallv:                     # boundary (r,c-1) <-> (r,c)
                a, b = move(r, c - 1), move(r, c)
            else:
                out.add((*move(r, c), o))
                continue
            if a[0] == b[0]:                     # horizontal boundary -> WallV
                out.add((a[0], max(a[1], b[1]), wallv))
            else:                                # vertical boundary -> WallH
                out.add((min(a[0], b[0]), a[1], wallh))
        return frozenset(out), (w if k % 2 else h), (h if k % 2 else w)

    _CW = {"up": "right", "right": "down", "down": "left", "left": "up"}
    _MIRROR = {"left": "right", "right": "left", "up": "up", "down": "down"}

    def turn_dir(d, k, mirror):
        if mirror:
            d = _MIRROR[d]
        for _ in range(k):
            d = _CW[d]
        return d

    def seat_cells(cs, h, w):
        bg = expert.bg_id
        eng.height, eng.width = h, w
        eng.grid = [[{bg} for _ in range(w)] for _ in range(h)]
        for (r, c, o) in cs:
            eng.grid[r][c].add(o)
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    total = chiral = mirror_only = 0
    for level in range(game.n_levels):
        game.set_level(level)
        h0, w0 = eng.height, eng.width
        expert.bind_walls(eng)
        base = _base_grid(eng, expert)
        states, truncated = _reachable(eng, base, expert, _read(eng, expert), cap)
        t = time.time()
        lvl_chiral = lvl_mirror = 0
        for player, crates in states:
            _seat(eng, base, player, crates, expert)
            key = cells(eng.grid)
            for direction in _ORDER:
                seat_cells(key, h0, w0)
                eng.step(direction)
                ref, refwin = cells(eng.grid), eng.check_win()
                total += 1
                split_rot = split_mir = False
                for k in range(4):
                    for mir in (False, True):
                        if (k, mir) == (0, False):
                            continue
                        tk, th, tw = turn(key, k, mir, h0, w0)
                        seat_cells(tk, th, tw)
                        eng.step(turn_dir(direction, k, mir))
                        got = (cells(eng.grid), eng.check_win())
                        want = (turn(ref, k, mir, h0, w0)[0], refwin)
                        if got != want:
                            if mir:
                                split_mir = True
                            else:
                                split_rot = True
                if split_rot or split_mir:
                    lvl_chiral += 1
                    if split_mir and not split_rot:
                        lvl_mirror += 1
        eng.height, eng.width = h0, w0
        chiral += lvl_chiral
        mirror_only += lvl_mirror
        print(f"level {level:2d}: {len(states) * 4:6d} transitions from "
              f"{len(states):5d} states{' (CAPPED)' if truncated else ''}, "
              f"{lvl_chiral:5d} chiral, {lvl_mirror:4d} split ONLY by a mirror "
              f"({time.time() - t:5.1f}s)")
    print(f"{total} transitions, {chiral} decided by the rule order, "
          f"{mirror_only} of them split only by a mirror")
    print("the flips add no inconsistency the rotation does not already carry"
          if not mirror_only else
          f"FLIPS ARE NOT SAFE: {mirror_only} mirror-only transitions")
    return 0 if not mirror_only else 1


def _audit() -> int:
    """Assert every cell COMPOSITION a level can show is distinct at every cell
    size the levels use.

    Composition, not object: this game stacks walls INTO the cells that hold
    crates, targets and the player, so what has to be readable is the stack.
    Two of those pairs are the ones a corpus cannot survive being blind to:

      * **crate vs crate-on-target.** The crate sprite covers the target sprite
        completely, so the target is not visible under a crate at all -- the
        ONLY cue that a target is satisfied is the ``Happycrate`` recolour, and
        ``brown`` and ``orange`` are the same ARC index (12), which makes a
        happy crate a UNIFORM 12 block against a plain crate's 12-over-13. That
        one palette collision is what the two objects are told apart by, which
        is also why the game must take no colour augmentation.
      * **each wall against the cell it is drawn into.** A wall is one pixel
        wide (a column or a row) plus the two-pixel Facade the level-start rules
        paint into the neighbouring cell, so it survives the sprite downsample
        only while the boards stay small -- all four here render at cell_px 8 or
        10, where every one of the 5 sprite rows/columns is sampled.

    Whole FRAMES are compared, not cropped cells: `_render_frame` upscales a
    sub-64 render to fill the frame and then letterboxes it, so cell_px
    arithmetic does not locate a cell in the output (see [[explod-rendering-fixes]])."""
    game = KickingWallsSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    # Every stack the four levels' legends can produce, plus the facades the
    # level-start rules paint.
    bodies = {"": (), "player": ("player",), "crate": ("crate",),
              "happy": ("crate", "happycrate")}
    grounds = {"": (), "target": ("target",),
               "wallh": ("wallh",), "wallv": ("wallv",),
               "wallhv": ("wallh", "wallv"),
               "facadeh": ("facadeh",), "facadev": ("facadev",),
               "dot": ("dot",)}
    comps = {}
    for bn, bo in bodies.items():
        for gn, go in grounds.items():
            # Crate and Target are welded together by the two late rules:
            # ``[ Crate Target no Happycrate ] -> [ Crate Target Happycrate ]``
            # and ``[ Happycrate no Crate ] -> [ ]``. So a Happycrate off a
            # target and a bare crate ON one are both unreachable, and the pair
            # the audit has to separate is "crate" vs "target+crate+happy",
            # which is 12-over-13 vs a uniform 12.
            on_target = "target" in go
            if (bn == "happy") != on_target and bn in ("crate", "happy"):
                continue
            name = "+".join(x for x in (gn, bn) if x) or "floor"
            comps[name] = go + bo

    sizes: dict[tuple[int, int], list[int]] = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in comps.items():
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
              f"{len(shots)} compositions, "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


def _replay(seeds: int = 4) -> int:
    """End-to-end: record an episode, then drive the recorded SCREEN actions
    back through `PuzzleScriptAdapter.perform_action` and require the frames to
    come back byte-identical and the level to end in `GameState.WIN`.

    This is the check that the rotation contract holds (see `screen_action`):
    the expert plans in ENGINE space and records the button an agent would press
    in the augmented view, and nothing in the search would notice if those two
    drifted apart -- the frames would simply be of a different game than the
    actions."""
    solver = KickingWallsSolver()
    bad = 0
    for seed in range(seeds):
        solver.rng = random.Random(seed)
        ok, levels = solver.solve_episode(seed, explore=True)
        game = solver._game
        for lvl in levels:
            game._seed = seed
            game.set_level(lvl["level_id"])
            obs, acts = lvl["observations"], lvl["actions"]
            i = 0
            mismatch = None
            if not np.array_equal(np.asarray(game._current_frame),
                                  np.asarray(obs[0])):
                mismatch = "frame 0"
            for act in acts[1:]:
                i += 1
                gid = _ID_TO_GAMEACTION[act["index"]]
                if gid == GameAction.RESET:
                    # The prefix's closing RESET is a level reload, which is
                    # what `record_level` taped -- see `PSAStarSolver`'s
                    # recovery_mode.
                    game.set_level(lvl["level_id"])
                    frame = np.asarray(game._current_frame)
                else:
                    data = act.get("data")
                    ai = (ActionInput(id=gid, data=data) if data
                          else ActionInput(id=gid))
                    fd = game.perform_action(ai)
                    frame = np.asarray(fd.frame[-1] if fd.frame
                                       else game._current_frame)
                if mismatch is None and not np.array_equal(frame,
                                                           np.asarray(obs[i])):
                    mismatch = f"frame {i}"
            won = game._state == GameState.WIN
            if mismatch is not None or not won:
                bad += 1
                print(f"seed {seed} level {lvl['level_id']}: "
                      f"{'MISMATCH at ' + mismatch if mismatch else ''}"
                      f"{'' if won else ' NOT A WIN'}")
        print(f"seed {seed}: ok={ok}, {len(levels)} levels, "
              f"{sum(len(l['actions']) for l in levels)} actions replayed")
    print("replay clean" if not bad else f"REPLAY FAILED: {bad} levels")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--replay" in sys.argv:
        sys.exit(_replay())
    sys.exit(KickingWallsSolver.main())
