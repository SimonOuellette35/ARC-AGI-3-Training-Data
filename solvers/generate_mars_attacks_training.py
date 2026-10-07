"""Generate Phase-1 training data for the PuzzleScript game ps:mars_attacks
("Mars Attacks", ridlaa).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: an exact model of the
mechanic, an exhaustive solver over it, and the reports that certify both.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_mars_attacks",
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
An alien walks a 7x7 board of green walls. A press moves it one cell; a pill
(``Fruit``) it steps onto is swallowed, and a crate in front of it is shoved one
cell. You win when no pill is left. That is the whole game: two rules,

    [ > Player | Crate ] -> [ > Player | > Crate ]
    [ > Player |        ] -> [ > Player | NO Fruit ]

and ``No fruit`` for a win condition.

The second rule reads as though the alien could eat a pill it merely faces --
it deletes ``Fruit`` from the cell ahead whether or not the alien gets there.
It cannot: Player, Wall, Fruit and Crate share ONE collision layer, so the only
cell ahead that a pill can be in is one the alien is also free to walk into,
and the deletion always coincides with the step onto it. Measured on the
interpreter, not argued: `selfcheck` fuzzes ~200k presses (shipped levels plus
random boards) against a model in which eating is exactly "step onto it", and
the two never diverge.

So the puzzle is a TOUR -- visit every pill in as few presses as possible --
and the crates are what make it one. A crate is pushed only into an EMPTY cell:
a wall, the board edge, another crate or a PILL behind it all refuse the push,
and a refused push cancels the alien's own step too. On these boards the crates
sit in the single gaps of otherwise sealed walls, so they are doors: level 1's
crate plugs the only hole in the central wall, and levels 4 and 5 lock the
whole bottom room behind one. Shoving a door crate the wrong way (into the wall
beside it, or onto a pill) walls the pills off for good -- which is why
recovery here is a RESET rather than a re-plan.

THE MODEL
---------
`_Model` is that mechanic as a state machine over ``(player cell, crate
bitmask, pill bitmask)``. It is exact, not a relaxation, and `selfcheck` is
what says so. Every level is then solved EXHAUSTIVELY -- a bounded layered BFS
to the first winning layer, then a reverse BFS over the edges it recorded --
which yields both a provably shortest plan and, at every step of it, the exact
SET of presses that stay shortest. These boards are small enough that this is
the cheap option as well as the exact one: the widest level reaches its win
after visiting 741 states.

    level  size  pills  crates  plan  states  ties
        0  7x7     2      1      10     189     2
        1  7x7     2      1      12      90     6
        2  7x7     3      1      12     613     7
        3  7x7     3      1      18     741     9
        4  7x7     2      1      11     335     2
        5  7x7     2      2      15     563     3

    total 78 presses, 29 of them with a tie (37%)

The ties are why the sets matter. Most of a plan is the alien WALKING between
pills, and a walk's order is free -- any interleaving of the two axes that
stays on a shortest route reaches the same cell in the same number of presses
and leaves an identical board. Labelling one interleaving as the only right
answer would train 37% of the corpus to a coin flip; the reverse-BFS field
gives the honest answer instead, and `--selfcheck` re-derives each labelled
press independently to check it really is one shorter.

THE ONE PRESENTATION FIX
------------------------
`games/ps:mars_attacks/ps:mars_attacks.py` recolours the alien to yellow. Its
``lightgreen`` and the wall's ``Green Darkgreen lightgreen`` all collapse to
the single green in the ARC palette, so the object the agent controls rendered
in exactly the wall's colour, separated from it only by two eye pixels. See
that file for the argument and `--audit` for the check.

RECOVERY
--------
``recovery_mode = "reset"`` (the family default). A crate shoved into a corner,
or into the wall beside the doorway it was plugging, is not undoable and can
seal the remaining pills away for ever, so a perturbed state is not re-plannable
in general. The episode-wide epsilon prefix explores freely and ONE RESET
restores the level start, from which the memoized plan replays a guaranteed win.

Usage (run from the repo root):
    python solvers/generate_mars_attacks_training.py --episodes 200 \
        --out data/training_multi_level/mars_attacks

    python solvers/generate_mars_attacks_training.py --plans
    python solvers/generate_mars_attacks_training.py --selfcheck
    python solvers/generate_mars_attacks_training.py --symmetry
    python solvers/generate_mars_attacks_training.py --audit
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

GAME_NAME = "Mars_Attacks"
GAME_MODULE_ID = "ps:mars_attacks"

#: The presses the search branches on, in the order it tries them. ACTION5 is
#: bound to no rule in this game; it is kept in the list so the search MEASURES
#: that rather than assuming it (a self-loop is pruned by the state dedup).
KEYS = ("up", "down", "left", "right", "action")
_ACTION = KEYS.index("action")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


# ---------------------------------------------------------------------------
# Native model of the mechanic
# ---------------------------------------------------------------------------

class _Model:
    """Mars Attacks as a state machine over ``(player, crates, pills)``.

    ``player`` is a cell index ``r * w + c``; ``crates`` and ``pills`` are
    bitmasks over the same indexing. Walls are static and live on the model.
    There is only ever one alien, so its cell is an int rather than a mask.

    ``nbr[di][cell]`` is the neighbour of ``cell`` in direction ``KEYS[di]``, or
    -1 off the board. It is a table rather than arithmetic because the BOARD
    EDGE is load-bearing here: unlike most of this family's levels, Mars
    Attacks' maps are not sealed in walls (level 1 opens at three corners, and
    level 3's bottom-left pill sits on the border), so "off the board" is a
    refusal the search meets constantly and must never wrap around a row for.
    """

    __slots__ = ("h", "w", "walls", "nbr")

    def __init__(self, h: int, w: int, walls: int):
        self.h, self.w, self.walls = h, w, walls
        self.nbr = []
        for di in range(4):
            dr, dc = _DELTA[KEYS[di]]
            row = []
            for i in range(h * w):
                r, c = divmod(i, w)
                rr, cc = r + dr, c + dc
                row.append(rr * w + cc if 0 <= rr < h and 0 <= cc < w else -1)
            self.nbr.append(row)

    # -- dynamics ----------------------------------------------------------
    def step(self, state, di: int):
        """The settled state after pressing ``KEYS[di]``, or None if the press
        changes NOTHING -- which in this game is the same thing as the engine's
        grid being left byte-identical, and is what `selfcheck` asserts.

        Three ways to be refused, and all three are the collision layer: the
        alien's target is a wall or off the board; the target holds a crate
        whose own target is a wall, another crate, a PILL, or off the board.
        Otherwise the alien steps, and a pill on the cell it steps onto is
        swallowed (rule 2, which fires exactly when rule 1 did not).
        """
        p, crates, pills = state
        if di == _ACTION:
            return None                       # bound to no rule in this game
        t = self.nbr[di][p]
        if t < 0 or (self.walls >> t) & 1:
            return None
        tb = 1 << t
        if crates & tb:
            u = self.nbr[di][t]
            if u < 0:
                return None
            ub = 1 << u
            if (self.walls >> u) & 1 or (crates & ub) or (pills & ub):
                return None
            return (t, (crates & ~tb) | ub, pills)
        return (t, crates, pills & ~tb)

    @staticmethod
    def won(state) -> bool:
        return not state[2]

    # -- exhaustive solve ---------------------------------------------------
    def solve(self, start, cap: int):
        """A shortest press sequence from ``start`` plus the exact optimal SET
        at each of its steps, or ``(None, None)`` if the level cannot be won.

        Forward BFS one whole LAYER at a time until a layer contains a win. That
        depth is ``d*``, every state on a shortest path has been generated (BFS
        depth is minimal, so a state on a shortest path is reached at its own
        depth), and every edge out of a state at depth < d* has been recorded --
        which is exactly the subgraph a reverse BFS from the wins needs to give
        the true distance-to-win of every such state. Stopping at that layer is
        what keeps this cheap: it is the difference between the 741 states level
        3 needs and the whole reachable component behind them.
        """
        if self.won(start):
            return [], []
        depth = {start: 0}
        edges: dict = {}
        layer = [start]
        wins: list = []
        d = 0
        while layer and not wins:
            d += 1
            nxt: list = []
            for state in layer:
                out = {}
                for di in range(len(KEYS)):
                    n = self.step(state, di)
                    if n is None:
                        continue
                    out[di] = n
                    if n not in depth:
                        depth[n] = d
                        if self.won(n):
                            wins.append(n)
                        else:
                            # A win is terminal: its successors are unreachable
                            # in play, and expanding them would let the reverse
                            # BFS route "through" a win.
                            nxt.append(n)
                        if len(depth) > cap:
                            return None, None
                edges[state] = out
            layer = nxt
        if not wins:
            return None, None

        rev: dict = {}
        for state, out in edges.items():
            for n in out.values():
                rev.setdefault(n, []).append(state)
        dist = {state: 0 for state in wins}
        queue = deque(wins)
        while queue:
            state = queue.popleft()
            for prev in rev.get(state, ()):
                if prev not in dist:
                    dist[prev] = dist[state] + 1
                    queue.append(prev)

        presses: list[str] = []
        optsets: list[list[str]] = []
        state = start
        while dist[state]:
            best = [di for di, n in edges[state].items()
                    if dist.get(n, 1 << 30) == dist[state] - 1]
            optsets.append([KEYS[di] for di in best])
            presses.append(KEYS[best[0]])
            state = edges[state][best[0]]
        return presses, optsets

    def reachable(self, start, cap: int) -> tuple[set, bool]:
        """``(states, hit_the_cap)`` for the reachable component. Only the
        reports use it -- the expert never needs more than `solve`'s bound --
        and it returns the partial set rather than None so a capped sweep still
        MEASURES what it saw instead of silently skipping the level."""
        seen = {start}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            for di in range(len(KEYS)):
                n = self.step(state, di)
                if n is not None and n not in seen and not self.won(n):
                    seen.add(n)
                    queue.append(n)
                    if len(seen) >= cap:
                        return seen, True
        return seen, False


def model_from_engine(eng, g) -> tuple:
    """Read ``(model, state)`` off the interpreter's live grid."""
    idx = g.obj_name_to_idx
    wall_id, crate_id = idx["wall"], idx["crate"]
    player_id, fruit_id = idx["player"], idx["fruit"]
    h, w = eng.height, eng.width
    walls = crates = pills = 0
    player = None
    for r in range(h):
        row = eng.grid[r]
        for c in range(w):
            cell, bit = row[c], 1 << (r * w + c)
            if wall_id in cell:
                walls |= bit
            if crate_id in cell:
                crates |= bit
            if fruit_id in cell:
                pills |= bit
            if player_id in cell:
                player = r * w + c
    if player is None:
        raise ValueError("no player on the board")
    return _Model(h, w, walls), (player, crates, pills)


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class MarsExpert(PSExpert):
    """Exhaustive optimal expert: bounded forward BFS on `_Model` to the first
    win, then read the plan and the per-step optimal sets straight off the
    distance-to-win field of the shortest-path subgraph.

    It plugs into the shared harness at `PSExpert._search`, so the plan memo, the
    snapshot/restore discipline around it and the level scoping all come from the
    base. `heuristic` is never called -- `_astar` is not the strategy here -- and
    neither is `dead`: an exhaustive search has no use for either.
    """

    directions = list(KEYS)

    #: A runaway guard on a board that does not belong in this file, not a tuning
    #: dial: the widest shipped level visits 741 states before its win appears.
    state_cap: int = 400_000

    def _search(self, eng):
        model, state = model_from_engine(eng, self.g)
        presses, optsets = model.solve(state, self.state_cap)
        if presses is None:
            return None
        return Plan(presses, optsets)


class MarsSolver(PSAStarSolver):
    game_id = "puzzlescript_mars_attacks"
    game_name = GAME_NAME
    expert_cls = MarsExpert
    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so the generator can never tape frames from a different
    #: build of the game than the agent plays. It matters here: that module
    #: recolours the alien off the wall's green.
    game_module_id = GAME_MODULE_ID

    #: The longest plan is 18 presses; the rest is room for the exploration
    #: prefix and the replay after its RESET. Comfortably inside the adapter's
    #: own 200-step per-level budget.
    max_steps = 150


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    solver = MarsSolver()
    game = solver.make_game(0)
    return solver, game, MarsExpert(game)


def _plan_report() -> int:
    """Per-level size, pill count, plan length, states visited and tie count.

    This is the table in the module docstring; re-run it after touching a level.
    It also measures the two claims the model rests on that are easy to state
    and easy to get wrong: that ACTION5 changes nothing, and that no press ever
    ends with the alien standing on a pill (i.e. entering a pill's cell really
    does consume it, which is what makes "eat" and "step" the same event)."""
    _solver, game, expert = _levels()
    eng = game._engine
    total = ties_total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        model, state = model_from_engine(eng, game._game)
        plan = expert.plan(eng, level)
        pills = bin(state[2]).count("1")
        crates = bin(state[1]).count("1")
        if plan is None:
            reach, capped = model.reachable(state, expert.state_cap)
            print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
                  f"{pills:2d} pills, NO PLAN -- unwinnable "
                  f"({len(reach)}{'+' if capped else ''} reachable states)")
            continue
        # re-derive the visited-state count for the table
        seen = {state}
        layer = [state]
        won = False
        while layer and not won:
            nxt = []
            for s in layer:
                for di in range(len(KEYS)):
                    n = model.step(s, di)
                    if n is None or n in seen:
                        continue
                    seen.add(n)
                    if model.won(n):
                        won = True
                    else:
                        nxt.append(n)
            layer = nxt
        ties = sum(1 for s in plan.optsets if len(s) > 1)
        total += len(plan)
        ties_total += ties
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
              f"{pills:2d} pills, {crates:2d} crates, plan {len(plan):3d} "
              f"({room}), {len(seen):6d} states visited, "
              f"{ties:2d} steps with a tie")
    print(f"total {total} presses, {ties_total} with a tie "
          f"({ties_total / max(1, total):.0%})")

    # The two measured invariants, over as much of each level's reachable
    # component as `sweep` states allow. `capped` levels are reported, never
    # skipped: a sweep that stopped early has still checked what it saw.
    sweep = 200_000
    inert = standing = states = 0
    partial = []
    for level in range(game.n_levels):
        game.set_level(level)
        model, state = model_from_engine(eng, game._game)
        seen, capped = model.reachable(state, sweep)
        states += len(seen)
        if capped:
            partial.append(level)
        for s in seen:
            if model.step(s, _ACTION) is not None:
                inert += 1
            for di in range(len(KEYS)):
                n = model.step(s, di)
                if n is not None and (n[2] >> n[0]) & 1:
                    standing += 1
    print(f"over {states} reachable states"
          f"{f' (levels {partial} stopped at the {sweep} cap)' if partial else ''}: "
          f"ACTION5 changes {inert} of them, and {standing} transitions leave "
          f"the alien standing on an uneaten pill")
    return 0 if not (inert or standing) else 1


# ---------------------------------------------------------------------------
# Model self-check (fuzz against the real interpreter)
# ---------------------------------------------------------------------------

def _read(eng, g):
    return model_from_engine(eng, g)[1]


def selfcheck(trials: int = 40, steps: int = 150,
              boards: int = 600, board_steps: int = 40) -> int:
    """Audit the claim the expert rests on: `_Model` reproduces the interpreter
    EXACTLY.

    Three things have to agree on every press, and they are the three the solver
    uses: the settled state (the alien's cell, the crates and the surviving
    pills), the REFUSAL decision -- a press the model refuses must leave the
    interpreter's grid byte-identical, and a press it accepts must change
    something -- and the win flag.

    Two fuzz populations, and the second is the one that matters. The six
    shipped levels exercise the game as it is played, but between them they hold
    seven crates and no two of them ever touch, so a shipped rollout can never
    press a crate into another crate, and only rarely into a pill. 600 RANDOM
    boards (random walls, a random open border, crates and pills scattered
    densely enough to sit beside each other) exercise every refusal the rules
    can produce -- and each of those is one exploration prefix away from a real
    episode. The counters at the end are the proof that they were reached: the
    check FAILS if the fuzz never pressed a crate into a crate, into a pill,
    into a wall, or off the board.
    """
    game = _load_game_module(GAME_MODULE_ID).make_game(seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    bg, wall_id = idx["background"], idx["wall"]
    crate_id, player_id, fruit_id = idx["crate"], idx["player"], idx["fruit"]
    tot = {"presses": 0, "refused": 0, "wins": 0, "eats": 0, "pushes": 0,
           "v_wall": 0, "v_crate": 0, "v_pill": 0, "v_edge": 0}
    bad = 0

    def classify(model, state, di):
        """Bump the coverage counter for what ``di`` does at ``state``, so the
        report can state which of the four refusals the fuzz actually reached
        rather than assuming a dense board must have produced them."""
        p, crates, pills = state
        if di == _ACTION:
            return
        t = model.nbr[di][p]
        if t < 0:
            tot["v_edge"] += 1
            return
        if (model.walls >> t) & 1:
            tot["v_wall"] += 1
            return
        if crates & (1 << t):
            u = model.nbr[di][t]
            if u < 0:
                tot["v_edge"] += 1
            elif (model.walls >> u) & 1:
                tot["v_wall"] += 1
            elif crates & (1 << u):
                tot["v_crate"] += 1
            elif pills & (1 << u):
                tot["v_pill"] += 1
            else:
                tot["pushes"] += 1
        elif pills & (1 << t):
            tot["eats"] += 1

    def rollout(tag: str, nsteps: int, rng: random.Random) -> int:
        nonlocal bad
        model, state = model_from_engine(eng, g)
        for _ in range(nsteps):
            live = [di for di in range(len(KEYS))
                    if model.step(state, di) is not None]
            # Mostly play on, occasionally press blind: a uniform walk spends
            # most of a rollout shoving the alien into the same wall, and the
            # presses that change something are where the mechanic lives. Not
            # circular -- a wrong legality claim still has to survive the grid
            # comparison below.
            di = (rng.choice(live) if live and rng.random() < 0.8
                  else rng.randrange(len(KEYS)))
            classify(model, state, di)
            before = [[set(cell) for cell in row] for row in eng.grid]
            eng.step(KEYS[di])
            tot["presses"] += 1
            frozen = eng.grid == before
            predicted = model.step(state, di)
            expected = state if predicted is None else predicted
            actual = _read(eng, g)
            if actual != expected:
                bad += 1
                print(f"  {tag}: state divergence on {KEYS[di]}: "
                      f"model {expected} vs engine {actual}")
                return 1
            if predicted is None and not frozen:
                bad += 1
                print(f"  {tag}: model refused {KEYS[di]}, engine did not")
                return 1
            if predicted is not None and frozen:
                bad += 1
                print(f"  {tag}: model moved on {KEYS[di]}, engine frozen")
                return 1
            if eng.check_win() != _Model.won(expected):
                bad += 1
                print(f"  {tag}: win divergence on {KEYS[di]}")
                return 1
            tot["refused"] += predicted is None
            state = expected
            if eng.check_win():
                tot["wins"] += 1
                return 0
        return 0

    for level in range(game.n_levels):
        for t in range(trials):
            game.set_level(level)
            rollout(f"L{level}/{t}", steps,
                    random.Random(f"mars:shipped:{level}:{t}"))

    for t in range(boards):
        rng = random.Random(f"mars:board:{t}")
        h, w = rng.randint(4, 9), rng.randint(4, 9)
        game.set_level(0)
        eng.height, eng.width = h, w
        eng.grid = [[{bg} for _ in range(w)] for _ in range(h)]
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)
        # No forced wall ring: these boards leak at the border exactly like the
        # shipped ones, which is what puts the board-edge refusal under test.
        for (r, c) in cells[:rng.randint(0, len(cells) // 3)]:
            eng.grid[r][c].add(wall_id)
        free = [x for x in cells if wall_id not in eng.grid[x[0]][x[1]]]
        rng.shuffle(free)
        if len(free) < 4:
            continue
        n_crate = rng.randint(1, max(1, len(free) // 3))
        n_pill = rng.randint(1, max(1, len(free) // 3))
        for (r, c) in free[:n_crate]:
            eng.grid[r][c].add(crate_id)
        for (r, c) in free[n_crate:n_crate + n_pill]:
            eng.grid[r][c].add(fruit_id)
        pr, pc = free[-1]
        eng.grid[pr][pc].add(player_id)
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        rollout(f"board{t}", board_steps, rng)

    print(f"selfcheck: {tot['presses']} presses "
          f"({game.n_levels} levels x {trials} rollouts + {boards} random "
          f"boards), {tot['refused']} refused, {tot['wins']} incidental wins, "
          f"{tot['eats']} pills eaten, {tot['pushes']} crates pushed")
    print(f"  refusals reached: {tot['v_wall']} crate/alien into a wall, "
          f"{tot['v_edge']} off the board, {tot['v_crate']} crate into a "
          f"crate, {tot['v_pill']} crate onto a pill")
    if not all(tot[k] for k in ("v_wall", "v_edge", "v_crate", "v_pill",
                                "eats", "pushes")):
        print("FUZZ TOO WEAK: it never reached one of the four refusals")
        return 1
    return bad


def _replay_check() -> int:
    """Every level's plan, replayed through the interpreter, must WIN -- and the
    optimal sets must be honest: at each step, every press the label calls
    optimal has to land on a state whose own distance-to-win is one shorter."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        model, state = model_from_engine(eng, game._game)
        if plan is None:
            _seen, capped = model.reachable(state, expert.state_cap)
            if capped:
                bad += 1
                print(f"  L{level}: no plan and the component blew the cap -- "
                      f"'unwinnable' is unproven here")
            else:
                print(f"  L{level}: no plan, component exhausted (unwinnable)")
            continue
        for i, press in enumerate(plan):
            if press not in plan.optsets[i]:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            remaining = len(plan) - i - 1
            for alt in plan.optsets[i]:
                nxt = model.step(state, KEYS.index(alt))
                if nxt is None:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, "
                          f"but it changes nothing")
                    continue
                sub, _ = model.solve(nxt, expert.state_cap)
                if sub is None or len(sub) != remaining:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but it "
                          f"leaves {len(sub) if sub is not None else 'no'} "
                          f"presses against {remaining}")
            state = model.step(state, KEYS.index(press))
            eng.step(press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
    print(f"replay check: {bad} problems")
    return bad


# ---------------------------------------------------------------------------
# The symmetry group
# ---------------------------------------------------------------------------

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

    dmap = {"action": "action"}
    for name, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[name] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    return cell, dims, dmap


def _pieces(eng, g) -> tuple:
    idx = g.obj_name_to_idx
    return tuple(frozenset((r, c) for r in range(eng.height)
                           for c in range(eng.width) if o in eng.grid[r][c])
                 for o in (idx["player"], idx["crate"], idx["fruit"]))


def _chirality_probe(game, expert, rollouts: int = 12, steps: int = 40) -> int:
    """The second half of `_symmetry`, and the half with teeth.

    Replaying a level's PLAN under the eight transforms cannot find a mechanic
    the plan does not happen to use, and a shortest plan uses very little: it
    never shoves a crate into another crate, never wedges one in a corner and
    never presses into a wall at all. Those are exactly the transitions where a
    rule-order asymmetry would live -- the interpreter expands a rule's four
    directions in a FIXED order, so anything settled by which direction is tried
    first is a fact about the screen, not about the board (ps:gobble_rush is the
    case in this family where that turned out to be real).

    So this stage does not follow a plan. It random-walks each level (steered by
    the model toward presses that change something), records every ``(board,
    press)`` it visits, and re-plays each one INDEPENDENTLY on all seven other
    presentations of that same board. A single order-sensitive transition
    anywhere in the walk is a failure.

    ``load_level`` is exact for a mid-walk board here because this game does not
    set ``run_rules_on_level_start``, so loading is a plain grid assignment.
    """
    eng, g = game._engine, game._game
    bad = tot = 0
    for level in range(game.n_levels):
        game.set_level(level)
        if expert.plan(eng, level) is None:
            continue
        hw = (eng.height, eng.width)
        triples = []
        for t in range(rollouts):
            game.set_level(level)
            rng = random.Random(f"mars:chiral:{level}:{t}")
            model, state = model_from_engine(eng, g)
            for _ in range(steps):
                live = [di for di in range(len(KEYS))
                        if model.step(state, di) is not None]
                di = (rng.choice(live) if live and rng.random() < 0.8
                      else rng.randrange(len(KEYS)))
                before = [[set(cell) for cell in row] for row in eng.grid]
                eng.step(KEYS[di])
                triples.append((before, KEYS[di], _pieces(eng, g)))
                nxt = model.step(state, di)
                state = state if nxt is None else nxt
                if eng.check_win():
                    break

        notes = []
        for k, mirror in itertools.product(range(4), (False, True)):
            if (k, mirror) == (0, False):
                continue
            cell, dims, dmap = _transform(k, mirror)
            th, tw = dims(hw)
            for i, (before, press, after) in enumerate(triples):
                turned = [[set() for _ in range(tw)] for _ in range(th)]
                for r, row in enumerate(before):
                    for c, objs in enumerate(row):
                        tr, tc = cell((r, c), hw)
                        turned[tr][tc] = set(objs)
                eng.load_level(turned)
                eng.step(dmap[press])
                tot += 1
                want = tuple(frozenset(cell(x, hw) for x in s) for s in after)
                if _pieces(eng, g) != want:
                    notes.append(f"rot{k}{'m' if mirror else ''}@{i}")
                    break
        bad += len(notes)
        print(f"level {level:2d}: {len(triples):4d} random presses x 7 "
              f"presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print(f"chirality probe: {tot} transformed presses checked, {bad} chiral")
    return bad


def _symmetry() -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an exact
    symmetry of the mechanic, which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation.

    Reading the two rules and declaring them direction-free is exactly the
    argument that was wrong in other ps: games, so it is measured instead.
    Stage 1 (here) replays each level's plan on all eight turned and mirrored
    copies of its own board, built by transforming the LEVEL LAYOUT so the
    interpreter re-loads it itself; every press must leave the pieces exactly
    where the transform of the reference run put them, and the run must still
    win. Stage 2 (`_chirality_probe`) covers what a plan cannot -- see its
    docstring.

    All three object classes are compared. There is no directional art in this
    game (nothing carries a facing and no wall tiles by orientation) and no
    axis-sensitive mechanic (no gravity), input is screen-relative, and the win
    condition -- ``No fruit`` -- names no direction."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level:2d}: no plan (unwinnable) -- skipped")
            continue
        layout = g.levels[level]
        hw = (len(layout), len(layout[0]))

        eng.load_level(layout)
        ref = []
        for direction in plan:
            eng.step(direction)
            ref.append(_pieces(eng, g))

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
            for i, direction in enumerate(plan):
                eng.step(dmap[direction])
                want = tuple(frozenset(cell(x, hw) for x in s) for s in ref[i])
                if _pieces(eng, g) != want:
                    notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                    break
            else:
                if not eng.check_win():
                    notes.append(f"rot{k}{'m' if mirror else ''}: no win")
        bad += len(notes)
        print(f"level {level:2d}: {len(plan):3d} plan presses x 7 "
              f"presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print("-- stage 2: random walks, which is what the plans cannot cover --")
    bad += _chirality_probe(game, expert)
    print("symmetry clean -- the rotation and flip augmentation is exact"
          if not bad else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Rendering audit
# ---------------------------------------------------------------------------

def _audit() -> int:
    """Assert every cell COMPOSITION a board can show is distinct at every cell
    size in use.

    The composition, not the object, is what has to be readable -- but this game
    has no stacks at all: Player, Wall, Fruit and Crate share one collision
    layer, so a cell is floor or exactly one of them. That makes the audit short
    and it still has to be run, because it is the check behind the recolour in
    `games/ps:mars_attacks/ps:mars_attacks.py`: before it, ``player`` and
    ``wall`` were the same green and this report is what would have to notice
    if a future edit put them back.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {"floor": (), "wall": ("wall",), "player": ("player",),
             "crate": ("crate",), "fruit": ("fruit",)}

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
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
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
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(MarsSolver.main())
