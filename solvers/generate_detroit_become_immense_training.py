"""Generate Phase-1 training data for the PuzzleScript game
ps:detroit_become_immense ("Detroit: Become Immense", Guilherme S. Tows).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: an exact model of the
mechanic, an exhaustive solver over it, and the sixteen levels.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_detroit_become_immense",
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
exactly. Every expert step also carries the full set of equally-optimal presses.

THE GAME
--------
You are a white blob on a blue-walled board, scattered with green arrows. The
blob is RIGID: a press slides the whole thing one cell, and if any cell of it
would enter a wall the entire turn is cancelled (``[ > Body | Wall ] -> cancel``
-- there is no partial move and no way to wait). When a blob cell lands on an
arrow the arrow is consumed and a NEW blob cell appears one step further along
the way the arrow points -- unless that destination is a wall or already blob,
in which case the arrow survives, uncollected, underneath you.

The win condition is the joke in the title: you win when you have eaten every
arrow AND cannot move in any of the four directions. In the .txt that is spelled
out through bookkeeping objects -- every turn stamps a ``Lose*`` marker on each
cell orthogonally adjacent to the blob, a marker that lands on a Wall is
upgraded to a ``Win*``, any surviving arrow deletes every ``Win*``, and one
``Win`` per direction is the victory -- but it reduces exactly to: no arrows
left, and some blob cell has a wall to its left, some cell has one to its right,
one above, one below.

WHY EVERY LEVEL ENDS ON AN ARROW (the shape of the whole puzzle)
---------------------------------------------------------------
"Blocked in direction d" and "could not have arrived by moving the opposite way"
are the SAME predicate: both say the cell the blob would occupy on that side
contains a wall. So a position blocked in all four directions cannot be reached
by any move at all. The winning shape is therefore never walked into -- it is
GROWN into, on the turn that collects the last arrow, and a board with no arrows
on it is unwinnable no matter how it is laid out. Every level here is built
around that: the route is a tour of the arrows, and the last one has to drop its
new cell into the one pocket that wedges the shape it has by then become.

It also means the blob's final SHAPE is a choice, not a given. Growth appends
``arrow cell + arrow direction`` in board coordinates, so which cell joins the
blob depends on where the blob was standing when it swallowed the arrow -- the
same five arrows in the same order build different polyominoes depending on the
route, and only some of them fit the pocket.

THE MODEL
---------
`_Model` re-implements the mechanic natively -- ~1 us/press against ~2.6 ms for
the interpreter, which is what makes enumerating a level affordable -- and
`--selfcheck` fuzzes it against the real thing on every level: same settled
state, same cancel decision, same win flag, over 40 random rollouts each. That
fuzz is the whole licence for planning off the engine, and the same flag then
replays every level's plan through the interpreter and re-derives every optimal
label from the distance field.

The state is small and exact -- (blob cells, surviving arrows) -- and the
reachable component of every level here is a few thousand states, so the expert
is not a heuristic search at all: it is an exhaustive forward BFS over the
component followed by a reverse BFS from the win states. That gives the true
distance-to-win of every reachable state, hence a certified SHORTEST plan and,
for free, the exact optimal-action SET at each step. Weighted A*, deadlock
tests and node caps are all beside the point at this size; the whole 16-level
solve is under two seconds, which is also why there is no on-disk plan cache
(`PSExpert.plan_cache_path` stays None -- it earns its keep at ~2 minutes, not
at ~2 seconds).

The reverse BFS has to run over the FORWARD-collected edges rather than
expanding backwards from the goal: growth is irreversible (an arrow eaten is
gone and a blob cell never leaves), so the transition relation is directed and
there is no backward successor function to expand.

THE LEVELS
----------
The shipped file holds ONE board -- level 12 below. One level is not a corpus,
so the .txt now carries sixteen, fifteen of them authored here by random
synthesis filtered through the exhaustive solver, ordered by shortest-solution
length:

    level  size    arrows  plan  states  ties   what it is
    0       5x5     1        3        5    0    one arrow, one pocket
    1       6x6     1        5       12    2    the pocket is round a corner
    2       7x7     2        7       34    0    two arrows, one corridor
    3       7x7     2        9       46    2    the arrows must be taken in order
    4       7x8     2       12       66    1    a detour to set the shape up
    5       8x9     3       15      237    7    open floor: the routing is free,
                                                so half the steps have a tie
    6       8x9     3       18       72    1    a pillared room, one route
    7       9x9     3       20      109    3    three arrows round a wall block
    8       9x9     3       23      285    6    corner start, arrows spread out
    9       9x10    4       25      558    2    four arrows, an L-shaped finish
    10     10x11    4       30      465    6    two chambers, one gap
    11     10x11    4       32      628    6    a 5-cell blob into a tight slot
    12     11x12    5       38     1954   12    THE SHIPPED LEVEL
    13     11x12    5       43     2662    8    the long way round twice
    14     11x12    5       45     2661   12    arrows on both diagonals
    15     11x12    5       46     1591   16    the widest board, the most ties

``states`` is the size of the level's whole reachable component (which is what
the expert enumerates) and ``ties`` is how many of its plan steps have more than
one equally-optimal press. Every level is engine-verified: the plan the model
produces is replayed through the interpreter and the interpreter reports the
win. None starts already won, and none can be won without collecting arrows.

Two ways a hand-drawn board goes quietly wrong, both ruled out by construction
and both caught anyway by the solver:

  * an arrow whose destination cell is a WALL can never be collected, so the
    board is unwinnable however it is played -- the arrow sits under the blob
    forever and keeps deleting the Win markers;
  * a board with no wedge for any shape the arrows can build is unwinnable in
    the same silent way. Random synthesis produces both constantly; the
    exhaustive BFS is what says so, and only boards it solves are here.

The outer ring of every level is solid wall, which is not decoration: the
author's movement rules are written as ``[ > Body | no Body ] -> [ | Body ]``,
which needs a cell on the far side to move into, so a blob cell against the
board EDGE would sit still while the rest of the blob walked out from under it
and the shape would tear. The wall makes that unreachable.

RENDERING
---------
Nothing here needed recolouring -- background BLACK (5), Wall BLUE (9), the
arrows GREEN (14) and the blob WHITE (0) are four distinct ARC indices, and the
Player's black face reads against the white blob it is always standing on. The
Win/Lose markers are deliberately invisible (black, on black floor and under
blue wall): they are pure bookkeeping, recomputed from scratch every turn, and
carry nothing the blob-and-wall geometry does not already show.

One collision-layer change WAS needed, and it is a real observability fix rather
than a cosmetic one. ``Grow`` sat BELOW ``Body``, so an arrow the blob is
standing on but cannot collect -- destination blocked by the blob itself --
rendered as plain blob: hidden state, in a game whose entire win condition is
"are there arrows left". It is rare (6 frames in 20 000 random presses on the
shipped board) and it is exactly the state an agent must not have to remember,
so the layer now sits ABOVE the body sprites and below the Player. The move is
mechanically inert -- Grow and Body were already on separate layers, so nothing
about matching or movement changes -- and that was verified: the model/engine
fuzz stays at 0 violations and the shipped level's plan is the same 38 presses
before and after. ``--audit`` is the regression test.

AUGMENTATION
------------
Engine state after reset is identical for every seed (the levels are fixed ASCII
maps), so the only per-(seed, level) variables are presentational: the frame
rotation in {0,1,2,3} plus an independent horizontal and vertical flip, each
with the matching directional action remap (`_FLIP_GAMES`). Those are exact
symmetries here even though the growth rules name absolute directions, because
the four arrows form a closed orbit under the symmetry group in BOTH their
semantics and their art -- see the note beside `_FLIP_GAMES`. 16 levels x 16
presentations = 256. The expert plan is therefore seed-independent: solved once
per level, memoized, replayed per seed with that seed's remapped screen actions.

OPTIMAL-ACTION SETS
-------------------
The reverse BFS gives distance-to-win exactly, so the optimal set at a state is
just ``{d : dist(step(d)) == dist - 1}`` -- no probing, no reordering heuristic.
84 of the 371 expert steps across the sixteen levels have more than one right
answer, which is what open floor looks like: any interleaving of two axes that
keeps the blob on a shortest route to the same arrow is the same plan. Labelling
one of them as "the" answer would train against the truth on a quarter of the
corpus. No step ever ships unlabelled.

RECOVERY
--------
``recovery_mode = "reset"`` (the family default). Exploration here is genuinely
irreversible in two ways -- an eaten arrow does not come back, and a blob can
wedge itself into a corner it can never leave with arrows still on the board --
so the episode-wide epsilon prefix explores freely and ONE RESET restores the
level start, from which the memoized plan replays a guaranteed win.

Usage (run from the repo root):
    python solvers/generate_detroit_become_immense_training.py --episodes 200 \
        --out data/training_multi_level/detroit_become_immense

    python solvers/generate_detroit_become_immense_training.py --plans
    python solvers/generate_detroit_become_immense_training.py --selfcheck
    python solvers/generate_detroit_become_immense_training.py --audit
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.common.ps_astar import (Plan, PSAStarSolver, PSExpert,     # noqa: E402
                                     _load_game_module)

GAME_NAME = "Detroit__Become_Immense"
GAME_MODULE_ID = "ps:detroit_become_immense"

#: The four presses, in the order the searches branch on them. ACTION5 is bound
#: to no rule in this game, so it is not part of the action space.
KEYS = ("up", "down", "left", "right")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Arrow object name -> index into `KEYS` of the direction it grows the blob in.
_ARROW_DIR = {"growu": 0, "growd": 1, "growl": 2, "growr": 3}

#: The order the .txt states its four growth rules in, as `KEYS` indices. A
#: growth can uncover another arrow (the new cell may itself sit on one), so the
#: cascade is run to a fixed point -- and it is run in the interpreter's own rule
#: order, because two arrows can contend for the same destination cell and then
#: only the first rule to fire gets it.
_GROW_ORDER = tuple(KEYS.index(d) for d in ("right", "left", "down", "up"))


# ---------------------------------------------------------------------------
# Native model of the mechanic
# ---------------------------------------------------------------------------

class _Model:
    """Detroit: Become Immense, as a state machine over flat cell indices.

    A state is ``(body, arrows)`` where ``body`` is a frozenset of the blob's
    cells and ``arrows`` a frozenset of ``(cell, direction index)`` pairs for
    the arrows still on the board. Walls are static and live on the model.

    Cells are flat ``r * w + c`` indices. That is safe against row wrap only
    because every level's outer ring is wall, so no blob cell is ever in column
    0 or ``w - 1``; `model_from_engine` checks that rather than trusting it.
    """

    __slots__ = ("h", "w", "walls", "offsets")

    def __init__(self, h: int, w: int, walls: frozenset):
        self.h, self.w = h, w
        self.walls = walls
        self.offsets = tuple(dr * w + dc for dr, dc in
                             (_DELTA[k] for k in KEYS))

    # -- dynamics ----------------------------------------------------------
    def _grow(self, body: set, arrows: dict) -> None:
        """Run the four growth rules to a fixed point, in place.

        An arrow fires when the blob covers it and the cell one step along its
        direction is neither wall nor blob; the arrow is then consumed. Since
        the new cell can itself carry an arrow, one firing can enable another,
        so this loops until a whole pass changes nothing."""
        walls = self.walls
        changed = True
        while changed:
            changed = False
            for di in _GROW_ORDER:
                off = self.offsets[di]
                again = True
                while again:
                    again = False
                    for cell, ad in list(arrows.items()):
                        if ad != di or cell not in body:
                            continue
                        dest = cell + off
                        if dest in walls or dest in body:
                            continue
                        body.add(dest)
                        del arrows[cell]
                        again = changed = True

    def step(self, state, di: int):
        """The settled state after pressing ``KEYS[di]``, or None if the turn
        was cancelled (some blob cell would have entered a wall -- which in this
        game is the whole turn undone, growth included, not a partial move)."""
        body, arrows = state
        off = self.offsets[di]
        moved = frozenset(cell + off for cell in body)
        if moved & self.walls:
            return None
        grown, live = set(moved), dict(arrows)
        self._grow(grown, live)
        return (frozenset(grown), frozenset(live.items()))

    def blocked(self, body, di: int) -> bool:
        """True when the blob cannot move ``KEYS[di]`` -- some cell of it has a
        wall on that side. This is also, exactly, the condition the .txt's
        Lose->Win marker chain tests for that direction."""
        off = self.offsets[di]
        walls = self.walls
        return any(cell + off in walls for cell in body)

    def won(self, state) -> bool:
        body, arrows = state
        return (not arrows
                and all(self.blocked(body, di) for di in range(len(KEYS))))

    # -- exhaustive solve ---------------------------------------------------
    def explore(self, start, cap: int):
        """Forward BFS over the whole reachable component from ``start``.

        Returns ``(edges, wins)``: ``edges[state][di]`` is the successor state
        of each press that was not cancelled, and ``wins`` is every reachable
        winning state. Returns ``(None, None)`` past ``cap`` states -- the
        levels here are a few thousand, so blowing the cap means a board that
        does not belong in this file."""
        edges: dict = {}
        wins: list = []
        seen = {start}
        queue = deque([start])
        if self.won(start):
            wins.append(start)
        while queue:
            state = queue.popleft()
            out = {}
            for di in range(len(KEYS)):
                nxt = self.step(state, di)
                if nxt is None:
                    continue
                out[di] = nxt
                if nxt not in seen:
                    seen.add(nxt)
                    if self.won(nxt):
                        wins.append(nxt)
                    else:
                        # A win is terminal: the level ends there, so its
                        # successors are unreachable in play and enumerating
                        # them would be wasted work (and would let the reverse
                        # BFS route "through" a win).
                        queue.append(nxt)
                    if len(seen) > cap:
                        return None, None
            edges[state] = out
        return edges, wins

    @staticmethod
    def distances(edges: dict, wins: list) -> dict:
        """Distance-to-win of every state that has one, by reverse BFS over the
        forward-collected edges.

        It has to be done this way round: growth is irreversible, so there is no
        backward successor function to expand from the goal -- the edges must be
        discovered forwards first and then walked backwards."""
        rev: dict = {}
        for state, out in edges.items():
            for nxt in out.values():
                rev.setdefault(nxt, []).append(state)
        dist = {state: 0 for state in wins}
        queue = deque(wins)
        while queue:
            state = queue.popleft()
            for prev in rev.get(state, ()):
                if prev not in dist:
                    dist[prev] = dist[state] + 1
                    queue.append(prev)
        return dist

    def solve(self, start, cap: int):
        """A shortest press sequence from ``start`` plus the exact optimal SET
        at each of its steps, or ``(None, None)`` if the level cannot be won."""
        edges, wins = self.explore(start, cap)
        if edges is None or start not in (dist := self.distances(edges, wins)):
            return None, None
        presses: list[str] = []
        optsets: list[list[str]] = []
        state = start
        while dist[state]:
            best = [di for di, nxt in edges[state].items()
                    if dist.get(nxt, 1 << 30) == dist[state] - 1]
            optsets.append([KEYS[di] for di in best])
            presses.append(KEYS[best[0]])
            state = edges[state][best[0]]
        return presses, optsets


def model_from_engine(eng, g) -> tuple:
    """Read ``(model, state)`` off the interpreter's live grid."""
    idx = g.obj_name_to_idx
    arrow_ids = {idx[name]: d for name, d in _ARROW_DIR.items() if name in idx}
    wall_id, body_id = idx["wall"], idx["body"]
    h, w = eng.height, eng.width
    walls, body, arrows = set(), set(), {}
    for r in range(h):
        row = eng.grid[r]
        for c in range(w):
            cell, flat = row[c], r * w + c
            if wall_id in cell:
                walls.add(flat)
            if body_id in cell:
                body.add(flat)
            for obj in cell:
                if obj in arrow_ids:
                    arrows[flat] = arrow_ids[obj]
    if body & walls or any(c in (0, w - 1) or r in (0, h - 1)
                           for r, c in ((f // w, f % w) for f in body)):
        # See `_Model`: a blob cell on the board edge has nowhere to move into
        # and the author's rules would tear the shape apart rather than stop it.
        raise ValueError("blob outside the walled interior")
    return _Model(h, w, frozenset(walls)), (frozenset(body),
                                            frozenset(arrows.items()))


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class DetroitExpert(PSExpert):
    """Exhaustive optimal expert: enumerate the level's reachable component on
    `_Model`, then read the plan and the per-step optimal sets straight off the
    distance-to-win field.

    It plugs into the shared harness at `PSExpert._search`, so the plan memo,
    the snapshot/restore discipline around it and the level scoping all come
    from the base. `heuristic` is never called -- `_astar` is not the strategy
    here -- and neither is `dead`: an exhaustive search has no use for either.
    """

    directions = list(KEYS)

    #: Every level's whole component is under 3000 states, so this is a runaway
    #: guard on a board that should not have been added, not a tuning dial.
    state_cap: int = 400_000

    def _search(self, eng):
        model, state = model_from_engine(eng, self.g)
        presses, optsets = model.solve(state, self.state_cap)
        if presses is None:
            return None
        return Plan(presses, optsets)


class DetroitSolver(PSAStarSolver):
    game_id = "puzzlescript_detroit_become_immense"
    game_name = GAME_NAME
    expert_cls = DetroitExpert
    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so the generator can never tape frames from a different
    #: build of the game than the agent plays.
    game_module_id = GAME_MODULE_ID

    #: The longest plan is 46 presses; the rest is room for the exploration
    #: prefix and the replay after its RESET. Comfortably inside the adapter's
    #: own 200-step per-level budget.
    max_steps = 150


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _plan_report() -> int:
    """Per-level size, arrow count, plan length, component size and tie count.

    This is the table in the module docstring; re-run it after touching a
    level."""
    solver = DetroitSolver()
    game = solver.make_game(0)
    expert = DetroitExpert(game)
    eng = game._engine
    total = ties_total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        model, state = model_from_engine(eng, game._game)
        edges, wins = model.explore(state, expert.state_cap)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level:2d}: NO PLAN")
            continue
        ties = sum(1 for s in plan.optsets if len(s) > 1)
        total += len(plan)
        ties_total += ties
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
              f"{len(state[1])} arrows, plan {len(plan):3d} ({room}), "
              f"{len(edges) if edges else 0:6d} states, {len(wins)} win states, "
              f"{ties:3d} steps with a tie")
    print(f"total {total} presses over {game.n_levels} levels, "
          f"{ties_total} with a tie ({ties_total / max(1, total):.0%})")
    return 0


# ---------------------------------------------------------------------------
# Model self-check (fuzz against the real interpreter)
# ---------------------------------------------------------------------------

def selfcheck(trials: int = 40, steps: int = 120) -> int:
    """Audit the claim the expert rests on: `_Model` reproduces the interpreter
    EXACTLY, on every level.

    Three things have to agree on every press, and they are the three the
    solver uses: the settled state (blob cells and surviving arrows), the
    CANCEL decision -- a press the model refuses must leave the interpreter's
    grid byte-identical, and vice versa, since a cancelled turn in this game is
    a complete no-op -- and the win flag. Every seventh press is ACTION5, which
    no rule in this game reads: it must change nothing, and a rule that started
    reading it would surface here.

    Random play is the fuzz because it goes where a plan never would: it wedges
    the blob into pockets it cannot leave, walks it over arrows whose
    destination it is itself occupying (the case that motivated the collision-
    layer fix), and spends long runs bouncing off walls."""
    game = _load_game_module(GAME_MODULE_ID).make_game(seed=0)
    eng = game._engine
    bad = cancels = wins = stuck = 0
    for level in range(game.n_levels):
        for t in range(trials):
            game.set_level(level)
            model, state = model_from_engine(eng, game._game)
            rng = random.Random(f"detroit:selfcheck:{level}:{t}")
            for i in range(steps):
                if i % 7 == 6:
                    eng.step("action")
                    _m, actual = model_from_engine(eng, game._game)
                    if actual != state:
                        bad += 1
                        print(f"  L{level}: ACTION changed the state")
                        break
                    continue
                # Mostly play on, occasionally press blind: a uniform walk
                # spends most of a rollout shoving a wedged blob into the same
                # wall, and the legal moves are where the mechanic lives. Not
                # circular -- a wrong legality claim still has to survive the
                # grid comparison below.
                legal = [di for di in range(len(KEYS))
                         if model.step(state, di) is not None]
                if not legal:
                    stuck += 1
                    break
                di = (rng.choice(legal) if rng.random() < 0.8
                      else rng.randrange(len(KEYS)))
                before = [[set(c) for c in row] for row in eng.grid]
                eng.step(KEYS[di])
                frozen = eng.grid == before
                predicted = model.step(state, di)
                expected = state if predicted is None else predicted
                _m, actual = model_from_engine(eng, game._game)
                if actual != expected:
                    bad += 1
                    print(f"  L{level}: state divergence on {KEYS[di]}")
                    break
                if (predicted is None) != frozen:
                    bad += 1
                    print(f"  L{level}: cancel divergence on {KEYS[di]}: "
                          f"model {'cancel' if predicted is None else 'move'}, "
                          f"engine {'frozen' if frozen else 'changed'}")
                    break
                if eng.check_win() != model.won(expected):
                    bad += 1
                    print(f"  L{level}: win divergence on {KEYS[di]}")
                    break
                cancels += predicted is None
                state = expected
                if eng.check_win():
                    wins += 1
                    break
    print(f"selfcheck: {game.n_levels} levels x {trials} rollouts, "
          f"{cancels} cancels, {wins} incidental wins, {stuck} wedged blobs")
    return bad


def _replay_check() -> int:
    """Every level's plan, replayed through the interpreter, must WIN -- and the
    optimal sets must be honest: at each step, every press the label calls
    optimal has to lead to a state the plan's remaining length can still be
    reached from."""
    solver = DetroitSolver()
    game = solver.make_game(0)
    expert = DetroitExpert(game)
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: no plan")
            bad += 1
            continue
        model, state = model_from_engine(eng, game._game)
        edges, wins = model.explore(state, expert.state_cap)
        dist = model.distances(edges, wins)
        for i, press in enumerate(plan):
            if press not in plan.optsets[i]:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            for alt in plan.optsets[i]:
                nxt = model.step(state, KEYS.index(alt))
                if nxt is None or dist.get(nxt, 1 << 30) != dist[state] - 1:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, "
                          f"but it is not")
            state = model.step(state, KEYS.index(press))
            eng.step(press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
    print(f"replay check: {bad} problems")
    return bad


# ---------------------------------------------------------------------------
# Rendering audit
# ---------------------------------------------------------------------------

def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    The composition, not the object, is what has to be readable: an arrow
    UNDER the blob is a different thing from bare blob (it is an arrow you
    still have to come back for), and that pair is the one this game shipped
    wrong -- Grow sat below Body, so it rendered as plain blob. Checked at
    every cell size the sixteen levels render at, because the sprite detail
    that separates two compositions can be a pixel wide and the render samples
    a 5x5 sprite down to as little as 4px.

    Two clashes are DELIBERATE and are asserted to be the only ones: the Lose
    and Win markers are black, so they vanish into the black floor and under
    the blue wall respectively. They are bookkeeping objects that the .txt
    rebuilds from the blob's geometry every single turn, and that geometry is
    fully in the frame, so nothing is hidden by their being invisible."""
    solver = DetroitSolver()
    game = solver.make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {
        "floor": (), "wall": ("wall",),
        "body": ("body",), "body_player": ("body", "player"),
        "growl": ("growl",), "growr": ("growr",),
        "growu": ("growu",), "growd": ("growd",),
        "growl_on_body": ("body", "growl"), "growr_on_body": ("body", "growr"),
        "growu_on_body": ("body", "growu"), "growd_on_body": ("body", "growd"),
        "floor_lose": ("losel", "loser", "loseu", "losed"),
        "wall_win": ("wall", "winl", "winr", "winu", "wind"),
    }
    allowed = {("floor", "floor_lose"), ("wall", "wall_win")}

    sizes: dict = {}
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
            frame = np.asarray(_render_frame(eng, g))
            cell = 64 // max(h, w)
            r, c = h // 2, w // 2
            shots[name] = frame[cell * r:cell * (r + 1),
                                cell * c:cell * (c + 1)].copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])
                   and (a, b) not in allowed]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad else
          f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plan_report())
    if "--selfcheck" in sys.argv:
        sys.exit(1 if (selfcheck() + _replay_check()) else 0)
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(DetroitSolver.main())
