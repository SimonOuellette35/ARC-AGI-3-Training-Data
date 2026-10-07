"""Generate Phase-1 training data for the PuzzleScript game ps:opposition
("Opposition", Matthew VanDevander).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: an exact model of the mechanic, an exhaustive optimal
solver over it, and the reports that certify both.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_opposition",
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
One arrow key drives two teams in opposite directions. The red Players take the
press; every blue BluePlayer takes its opposite. You win when ALL reds stand on
red exits and ALL blues stand on blue exits AT THE SAME TIME. The entire game is
one rule:

    [ > Player ] [ BluePlayer ] -> [ > Player ] [ < BluePlayer ]

Part One (levels 0-6) is one red and one blue; Part Two (levels 7-13) is the
same seven boards rebuilt with TWO of each, which is where the game gets its
teeth -- two reds are two independent bodies that happen to share a key, so a
level is solved by finding presses that are wall-blocked for exactly the bodies
that must stay put.

Four facts are the mechanic, and the last three are all consequences of the rule
matching on the player's FORCE rather than on the player having MOVED:

  * **A wall is how you steer.** There is no `rigid` and no cancel, so a press a
    red cannot take still hands every blue its opposite force: walking into a
    wall moves the other team alone. With two bodies per team the same trick
    desynchronises teammates -- press into a wall that blocks one red and the
    other red walks on by itself.
  * **HEAD-ON is a total no-op.** A blue directly ahead of a red is, by
    construction, a mover heading straight back at it: each one's chain ends on
    the other, `_resolve_forces` DEFERS both, nothing moves that iteration and
    the turn is dropped. The two bodies never swap and never pass -- which is
    what makes a corridor a hard wall between them (see LEVEL 5 below).
  * **A CONTESTED CELL blocks both claimants.** When a red and a blue are two
    apart and press toward the gap between them, both chains are free and both
    want the same cell, so phase 3 of `_resolve_forces` blocks both and nothing
    moves. `R . b` pressed right is as much a no-op as `R b` is.
  * **Same-team bodies in line form ONE chain.** Two reds abreast along the
    press direction are traced as a single push chain (the blocker carries the
    same force, so the trace walks through it) and move together or not at all.
    They cannot contest each other: one direction translates every red, and a
    translation is injective.

There is no way to lose -- no rule deletes a body and nothing is destructible --
but there is no ACTION either: the game declares no action rule, so ACTION5 is a
strict no-op and the expert branches on the four arrows only. The one thing a
press can cost is position, so recovery is a RESET (`PSAStarSolver.recovery_mode`).

THE LEVELS
----------
All fourteen are recorded except LEVEL 5, which is **unwinnable**, and that is
proved rather than assumed: its reachable space is 187 states and the
breadth-first sweep exhausts every one of them without a win. The reason is the
head-on rule above. Level 5's floor is a single 17-cell corridor (two dead-end
pockets at the far end apart), the red starts at index 9 counting from the red
exit and the blue at index 7 -- so the blue sits BETWEEN the red and its goal
while its own goal is at the far end. The two must trade ends of a corridor they
can neither swap on nor pass on. `discover_solvable` drops it once at startup and
every seed skips it. (Level 12 is the same map rebuilt for Part Two, and the
author widened it: it is winnable, in 56 presses.)

Every plan here is proved SHORTEST rather than merely found -- `_Model.solve` is
a breadth-first sweep of the level's own state space, so the length it returns is
the true distance. `--plans` prints this table and CERTIFIES every plan by
replaying it through the interpreter:

    level  bodies    reachable   visited    d*   ties
      0      1+1             8         7     7   1.0
      1      1+1             8         7     7   1.0
      2      1+1            65        58    11   1.0
      3      1+1           234       125     9   1.1
      4      1+1           234       214    12   1.2
      5      1+1           187         -     -   unwinnable
      6      1+1          1234       131     7   1.1
      7      2+2             8         7     7   1.0
      8      2+2            25        23     9   1.0
      9      2+2           117       111    13   1.0
     10      2+2         96470       421     9   1.1
     11      2+2         96470      3313    12   1.2
     12      2+2         36542     28650    56   1.1
     13      2+2        387818     83916    15   1.2

``reachable`` is the level's ENTIRE reachable state space and ``visited`` what
the depth-bounded search actually touches -- the gap is the point of stopping at
the first winning LAYER rather than sweeping the space: level 10 answers from 421
of its 96470 states. ``ties`` is the mean size of the optimal-press set along the
plan; 10 of the 174 presses across the thirteen levels have one, which is how
much of the labelling would be a lie if the recorder trained one arbitrary
shortest path as the only right answer. Ties are rare here for a reason worth
knowing: a press moves BOTH teams, so two different presses almost never leave
the same board, and the ones that do are the presses one team spends face-first
against a wall. The longest plan is level 12's 56 presses, comfortably inside the
adapter's 200-step per-level budget even after the exploration prefix.

Level 12 is the outlier and worth a look: 56 presses is six times the median,
because its two reds and two blues share one narrow shaft and almost every press
is a wall-bump that advances a single body one cell. It is also the level where
the search is least able to prune -- 28650 of its 36542 reachable states are
visited, because d* is deep enough that the winning layer is most of the space.

EXPERT SOLVER
-------------
`_Model` is the mechanic over two frozensets of cell indices (the reds and the
blues), and it is a faithful re-implementation of `PSEngine._resolve_forces`
restricted to this game's one collision layer -- the chain trace, the
defer-on-a-mover rule, the subsume pass and the multi-way conflict pass, in that
order, iterated to a fixed point. It is not an argument, it is fuzz-verified
against the real interpreter by `--selfcheck` on the shipped levels AND on random
boards built to hold the configurations no shipped level reaches often enough to
trust (bodies nose to nose, two chains contesting one cell, three-body teams,
same-team chains shoved into walls).

`_Model.solve` then does the exact thing:

  1. a breadth-first sweep from the level start, layer by layer, stopping at the
     first layer that contains a winning press -- so ``d*`` is the true shortest
     distance and an exhausted sweep with no win is a PROOF of unwinnability
     (winning is a transition, not a state: `check_win` is read after a press
     and a won level is terminal, so those edges lead nowhere);
  2. a backward pass over those same layers keeping only the states that reach a
     win in exactly the remaining number of presses -- the shortest-path DAG;
  3. the plan is any walk down it, and the optimal SET at each step is every
     press whose successor is still in the DAG.

Keeping LAYERS rather than a predecessor map is what makes step 2 cheap: it needs
only the states, re-deriving each successor as it goes, so nothing ever stores an
edge. The thirteen searches together take about five seconds, most of it level
13, which is why this expert DOES carry an on-disk plan cache
(`data/opposition_plans.json`): the searches are seed-independent and are the
whole cost of generation, so without a file every shard `parallelize_generator`
starts would re-derive all of them.

RENDERING
---------
No .txt change was needed, which is rare enough here to state. Both bodies are
drawn with eleven transparent pixels, so a body standing on an exit shows the
exit's colour through them and "who is home" is legible at every cell size the
game uses (6, 7, 8 and 9 px); the two exits differ in hue, the two bodies differ
in hue, and the four cross combinations are all distinct. `--audit` asserts that
over every composition the board can paint, and asserts separately that no level
seals an exit under the opaque wall sprite.

Usage
-----
    python solvers/generate_opposition_training.py --episodes 200 \
        --out data/opposition
    python solvers/generate_opposition_training.py --plans
    python solvers/generate_opposition_training.py --selfcheck
    python solvers/generate_opposition_training.py --symmetry
    python solvers/generate_opposition_training.py --audit
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame                 # noqa: E402
from solvers.common.ps_astar import (Plan, PSAStarSolver, PSExpert,      # noqa: E402
                                     _load_game_module)

GAME_NAME = "Opposition"
GAME_MODULE_ID = "ps:opposition"

#: The presses the search branches on. ACTION5 is deliberately absent: the game
#: declares no action rule, so an action press hands nobody a force and cannot
#: change the board -- it is a wait move in a game with no clock, and a wait move
#: is never on a shortest path.
KEYS = ("up", "down", "left", "right")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
#: index of the opposite of each direction -- the whole game, as a table
_OPP = (1, 0, 3, 2)

#: `_resolve_forces`' own iteration bound, mirrored so the model cannot settle a
#: board the interpreter would have abandoned half-resolved.
_MAX_ITERS = 20


# ---------------------------------------------------------------------------
# Native model of the mechanic
# ---------------------------------------------------------------------------

class _Model:
    """Opposition as a state machine over two frozensets of cell indices.

    A state is ``(reds, blues)``: the cells ``r * w + c`` holding a Player and
    those holding a BluePlayer. Walls and the two exit sets are static and live
    on the model.

    ``nbr[di][cell]`` is the neighbour of ``cell`` in direction ``di``, or -1 off
    the board, precomputed so `step` never does row/column arithmetic (which is
    where an index model wraps around an edge without noticing).
    """

    __slots__ = ("h", "w", "walls", "rexits", "bexits", "nbr")

    def __init__(self, h: int, w: int, walls: frozenset,
                 rexits: frozenset, bexits: frozenset):
        self.h, self.w = h, w
        self.walls = walls
        self.rexits = rexits
        self.bexits = bexits
        self.nbr = []
        for di in range(4):
            dr, dc = _DELTA[KEYS[di]]
            table = [-1] * (h * w)
            for r in range(h):
                for c in range(w):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        table[r * w + c] = rr * w + cc
            self.nbr.append(table)

    def step(self, state, di: int, stats: dict | None = None):
        """The settled board after one press. Never None: this game has no
        cancel, and a press nothing can act on returns ``state`` itself.

        This is `PSEngine._resolve_forces` restricted to the one collision layer
        Player, BluePlayer and Wall share, and it follows the interpreter's four
        phases in order rather than reasoning about them:

          1. trace each mover's chain -- through same-force bodies (teammates in
             line), stopping at a wall or the board edge (BLOCKED) or at a mover
             carrying a different force (DEFERRED, because the interpreter gives
             it a chance to vacate) or at empty floor (FREE);
          2. subsume chains whose head is already inside a longer chain;
          3. block every pair of surviving chains that claim the same cell;
          4. move what is left, and iterate until a pass moves nothing.

        Phase 1 reads the LIVE force table, so a chain's verdict can depend on
        whether a chain popped earlier in the same pass is still in it. That
        ordering cannot change the settled board here -- a blocked chain's head
        is a body that is not going to move, so the chain behind it is blocked
        either immediately or one iteration later -- but the loop is written the
        interpreter's way regardless, and `--symmetry`'s chirality probe is what
        turns that argument into a measurement.

        ``stats`` is an optional counter dict for `selfcheck`'s coverage report;
        it is filled from the same trace `step` acts on, so what the counters
        claim was exercised is what the model actually did.
        """
        reds, blues = state
        occ = {}
        for x in reds:
            occ[x] = True
        for x in blues:
            occ[x] = False
        forces = {}
        for x in reds:
            forces[x] = di
        for x in blues:
            forces[x] = _OPP[di]

        for _iter in range(_MAX_ITERS):
            moved_any = False
            resolved: set[int] = set()
            movable: list[tuple[list[int], int]] = []

            # Phase 1: trace.
            for cell, dv in list(forces.items()):
                if cell in resolved:
                    continue
                if cell not in occ:                # its body moved earlier
                    forces.pop(cell, None)
                    resolved.add(cell)
                    continue
                chain = [cell]
                cur = cell
                chain_free = False
                blocker_is_mover = False
                table = self.nbr[dv]
                while True:
                    nxt = table[cur]
                    if nxt < 0 or nxt in self.walls:
                        break                      # edge / wall: BLOCKED
                    if nxt not in occ:
                        chain_free = True          # empty floor: FREE
                        break
                    bf = forces.get(nxt)
                    if bf == dv:
                        chain.append(nxt)          # teammate in line: extend
                        cur = nxt
                    else:
                        blocker_is_mover = bf is not None
                        break
                if chain_free:
                    movable.append((chain, dv))
                elif blocker_is_mover:
                    if stats is not None and _iter == 0:
                        stats["headon"] += 1
                else:
                    if stats is not None and _iter == 0:
                        stats["walled"] += 1
                    for x in chain:
                        forces.pop(x, None)
                        resolved.add(x)

            # Phase 2: subsume.
            non_head = {e for chain, _ in movable for e in chain[1:]}
            live = [i for i, (chain, _) in enumerate(movable)
                    if chain[0] not in non_head]

            # Phase 3: multi-way conflicts.
            claims: dict[int, int] = {}
            conflicting: set[int] = set()
            for i in live:
                chain, dv = movable[i]
                table = self.nbr[dv]
                for x in chain:
                    k = table[x]
                    prev = claims.get(k)
                    if prev is not None and prev != i:
                        conflicting.add(i)
                        conflicting.add(prev)
                    else:
                        claims[k] = i

            # Phase 4: move.
            for i in live:
                chain, dv = movable[i]
                if i in conflicting:
                    if stats is not None and _iter == 0:
                        stats["contested"] += 1
                    for x in chain:
                        forces.pop(x, None)
                        resolved.add(x)
                    continue
                if stats is not None and _iter == 0:
                    stats["moves"] += 1
                    if len(chain) > 1:
                        stats["chains"] += 1
                table = self.nbr[dv]
                for x in reversed(chain):
                    occ[table[x]] = occ.pop(x)
                    forces.pop(x, None)
                    resolved.add(x)
                    moved_any = True

            if not moved_any:
                break

        return (frozenset(x for x, red in occ.items() if red),
                frozenset(x for x, red in occ.items() if not red))

    def won(self, state) -> bool:
        """``All Player on RedExit`` and ``All BluePlayer on BlueExit``, which is
        a SUBSET test in both directions: a level may ship more exits than
        bodies, and it is the bodies the condition quantifies over."""
        reds, blues = state
        return reds <= self.rexits and blues <= self.bexits

    def solve(self, start, cap: int):
        """``(presses, optsets)`` for a SHORTEST win, or ``(None, None)``.

        Breadth-first by layers from ``start`` (a won state is terminal and is
        never expanded), stopping at the first layer holding a winning press;
        then a backward pass keeps, in each layer, only the states that reach a
        win in exactly the presses remaining -- the shortest-path DAG. Walking
        down it gives the plan, and the optimal SET at each step is every press
        whose successor is still in the DAG, which is exact because a successor
        that reaches the win in the remaining count cannot sit in any other
        layer (its depth would make ``d*`` smaller than it is).

        Returning ``(None, None)`` from an EXHAUSTED sweep is a proof of
        unwinnability, and level 5 is the level that needs it. ``cap`` is the
        separate runaway guard for a board that does not belong to this game.
        """
        if self.won(start):
            return [], []
        layers = [[start]]
        seen = {start}
        depth = 0
        while True:
            nxt = []
            hit = False
            for s in layers[-1]:
                for di in range(4):
                    ns = self.step(s, di)
                    if self.won(ns):
                        hit = True
                        continue
                    if ns in seen:
                        continue
                    seen.add(ns)
                    nxt.append(ns)
            if hit:
                break
            if not nxt:
                return None, None           # component exhausted: unwinnable
            if len(seen) > cap:
                return None, None
            layers.append(nxt)
            depth += 1

        # Backward pass over the layers: `good[k]` is the states of layer k that
        # win in exactly `depth - k + 1` presses.
        good = [None] * (depth + 1)
        good[depth] = {s for s in layers[depth]
                       if any(self.won(self.step(s, di)) for di in range(4))}
        for k in range(depth - 1, -1, -1):
            ahead = good[k + 1]
            good[k] = {s for s in layers[k]
                       if any(self.step(s, di) in ahead for di in range(4))}

        presses, optsets = [], []
        s = start
        for k in range(depth + 1):
            if k == depth:
                best = [KEYS[di] for di in range(4)
                        if self.won(self.step(s, di))]
            else:
                ahead = good[k + 1]
                best = [KEYS[di] for di in range(4)
                        if self.step(s, di) in ahead]
            optsets.append(best)
            presses.append(best[0])
            s = self.step(s, KEYS.index(best[0]))
        return presses, optsets

    def reachable(self, start, cap: int):
        """``(states, capped)`` for the level's whole reachable space, won states
        included as leaves. Used only by the reports."""
        seen = {start}
        q = deque([start])
        while q:
            s = q.popleft()
            for di in range(4):
                ns = self.step(s, di)
                if ns in seen:
                    continue
                seen.add(ns)
                if len(seen) > cap:
                    return seen, True
                if not self.won(ns):
                    q.append(ns)
        return seen, False


def model_from_engine(eng, g) -> tuple:
    """``(_Model, state)`` read off the interpreter's live grid.

    Everything static (walls, both exit sets) goes on the model and everything
    that moves goes in the state, which is what lets a state be two frozensets.
    The bodies are asserted present: a win condition quantified over an empty
    team is vacuously true, so a board that lost one would report an instant win
    rather than an error.
    """
    idx = g.obj_name_to_idx
    h, w = eng.height, eng.width
    walls, rexits, bexits = set(), set(), set()
    reds, blues = set(), set()
    for r in range(h):
        for c in range(w):
            cell = eng.grid[r][c]
            i = r * w + c
            if idx["wall"] in cell:
                walls.add(i)
            if idx["redexit"] in cell:
                rexits.add(i)
            if idx["blueexit"] in cell:
                bexits.add(i)
            if idx["player"] in cell:
                reds.add(i)
            if idx["blueplayer"] in cell:
                blues.add(i)
    if not reds or not blues:
        raise ValueError("board has no Player or no BluePlayer")
    model = _Model(h, w, frozenset(walls), frozenset(rexits), frozenset(bexits))
    return model, (frozenset(reds), frozenset(blues))


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class OppositionExpert(PSExpert):
    """Exhaustive optimal expert: `_Model.solve` from the current board, with the
    plan and the per-step optimal sets read off the shortest-path DAG.

    It plugs into the shared harness at `PSExpert._search`, so the plan memo, its
    disk cache, the snapshot/restore discipline and the level scoping all come
    from the base. `heuristic` is never called -- `_astar` is not the strategy
    here -- and neither is `dead`: nothing in this game is lethal, and a sweep
    that exhausts the component has already proved what `dead` would guess.
    """

    directions = list(KEYS)

    #: The thirteen searches are ~5 s, nearly all of it level 13, and they are
    #: seed-independent -- so every shard `parallelize_generator` starts would
    #: otherwise re-derive them. The base checks the stored start layout, so an
    #: edited level is a cache MISS rather than a wrong plan.
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "opposition_plans.json")

    #: A runaway guard on a board that does not belong in this file, not a tuning
    #: dial -- the widest shipped level stores 118855 states.
    state_cap: int = 5_000_000

    def _search(self, eng):
        model, state = model_from_engine(eng, self.g)
        presses, optsets = model.solve(state, self.state_cap)
        if presses is None:
            return None
        return Plan(presses, optsets)


class OppositionSolver(PSAStarSolver):
    game_id = "puzzlescript_opposition"
    game_name = GAME_NAME
    expert_cls = OppositionExpert
    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so the generator can never tape frames from a different
    #: build of the game than the agent plays.
    game_module_id = GAME_MODULE_ID

    #: The longest plan is level 12's 56 presses; the rest is room for the
    #: exploration prefix and the replay after its RESET. Comfortably inside the
    #: adapter's own 200-step per-level budget.
    max_steps = 100


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    solver = OppositionSolver()
    game = solver.make_game(0)
    return solver, game, OppositionExpert(game)


def _plan_report() -> int:
    """Per-level body count, reachable-space size, states the search visits, plan
    length and tie richness -- the table in the module docstring.

    Every plan is CERTIFIED by replaying it through the real interpreter, and
    every optimal label is checked the expensive way: each press the label calls
    equally-shortest is taken on the model and re-solved from there, and it has
    to leave exactly one press fewer than the step it was offered at. A level
    with no plan is required to be a PROVED dead end (an exhausted component)
    rather than a search that ran out of budget."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    total = ties_total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        model, state = model_from_engine(eng, game._game)
        plan = expert.plan(eng, level)
        bodies = f"{len(state[0])}+{len(state[1])}"
        space, capped = model.reachable(state, expert.state_cap)
        if plan is None:
            bad += capped
            why = ("unproven, the sweep hit the cap" if capped
                   else "unwinnable (component exhausted)")
            print(f"level {level}: {eng.height}x{eng.width} {bodies} bodies, "
                  f"{len(space):6d} reachable, NO PLAN -- {why}")
            continue

        # Re-derive what the depth-bounded search touches, for the table.
        seen = {state}
        layer = [state]
        for _ in range(len(plan) - 1):
            nxt = []
            for s in layer:
                for di in range(4):
                    ns = model.step(s, di)
                    if model.won(ns) or ns in seen:
                        continue
                    seen.add(ns)
                    nxt.append(ns)
            layer = nxt

        # Certify the plan on the interpreter and the labels on the model.
        s = state
        for i, press in enumerate(plan):
            remaining = len(plan) - i - 1
            if press not in plan.optsets[i]:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            for alt in plan.optsets[i]:
                nxt = model.step(s, KEYS.index(alt))
                if model.won(nxt):
                    if remaining:
                        bad += 1
                        print(f"  L{level}: step {i} labels {alt} optimal, but "
                              f"it wins with {remaining} presses to spare")
                    continue
                sub, _ = model.solve(nxt, expert.state_cap)
                if sub is None or len(sub) != remaining:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but it "
                          f"leaves {len(sub) if sub else 'no'} presses against "
                          f"{remaining}")
            s = model.step(s, KEYS.index(press))
            eng.step(press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")

        ties = sum(len(o) for o in plan.optsets) / len(plan)
        total += len(plan)
        ties_total += sum(1 for o in plan.optsets if len(o) > 1)
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"level {level}: {eng.height}x{eng.width} {bodies} bodies, "
              f"{len(space):6d} reachable, {len(seen):6d} visited, "
              f"d* {len(plan):2d} ({room}), ties {ties:.1f}")
    print(f"total {total} presses, {ties_total} with a tie "
          f"({ties_total / max(1, total):.0%}), {bad} problems")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# Model self-check (fuzz against the real interpreter)
# ---------------------------------------------------------------------------

def _read(eng, g):
    """The dynamic state as the interpreter has it."""
    return model_from_engine(eng, g)[1]


def selfcheck(trials: int = 40, steps: int = 60,
              boards: int = 500, board_steps: int = 25) -> int:
    """Audit the claim the expert rests on: `_Model` reproduces the interpreter
    EXACTLY, on the settled state and on the win flag, for every press.

    Two fuzz populations, and the second is the one with teeth. The fourteen
    shipped levels exercise the game as it is played; random boards -- random
    walls, one to three bodies per team dropped anywhere, random exits --
    exercise the configurations no shipped level reaches often enough to trust: a
    body walking nose-first into an opponent, two bodies two cells apart
    contesting the gap, teammates in line shoved as one chain, a chain whose head
    is walled while its tail is not.

    The coverage counters below are printed and ASSERTED non-zero, because a fuzz
    that never built those boards would report a clean model while having tested
    none of the four phases of `_Model.step`.
    """
    game = _load_game_module(GAME_MODULE_ID).make_game(seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    bg = idx["background"]
    tot = {"presses": 0, "noops": 0, "wins": 0,
           "headon": 0, "contested": 0, "walled": 0, "moves": 0, "chains": 0,
           "desync": 0, "body_on_wrong_exit": 0}
    bad = 0

    def rollout(tag: str, nsteps: int, rng: random.Random) -> int:
        nonlocal bad
        model, state = model_from_engine(eng, g)
        for _ in range(nsteps):
            di = rng.randrange(4)
            before = state
            di_name = KEYS[di]
            eng.step(di_name)
            tot["presses"] += 1
            expected = model.step(state, di, stats=tot)
            actual = _read(eng, g)
            if actual != expected:
                bad += 1
                print(f"  {tag}: state divergence on {di_name}: "
                      f"model {expected} engine {actual}")
                return 1
            if eng.check_win() != model.won(expected):
                bad += 1
                print(f"  {tag}: win divergence on {di_name}")
                return 1
            tot["noops"] += expected == before
            # One team moved and the other did not -- the desync the whole game
            # is played through.
            tot["desync"] += ((expected[0] != before[0])
                              != (expected[1] != before[1]))
            tot["body_on_wrong_exit"] += bool(
                (expected[0] & model.bexits) or (expected[1] & model.rexits))
            state = expected
            if eng.check_win():
                tot["wins"] += 1
                return 0
        return 0

    for level in range(game.n_levels):
        for t in range(trials):
            game.set_level(level)
            rollout(f"L{level}/{t}", steps,
                    random.Random(f"opposition:shipped:{level}:{t}"))

    for t in range(boards):
        rng = random.Random(f"opposition:board:{t}")
        h, w = rng.randint(5, 9), rng.randint(5, 9)
        game.set_level(0)
        eng.height, eng.width = h, w
        eng.grid = [[{bg} for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if r in (0, h - 1) or c in (0, w - 1):
                    eng.grid[r][c].add(idx["wall"])
        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        rng.shuffle(cells)
        # A sparse wall field keeps the interior open enough for the bodies to
        # meet; a dense one is what puts a chain's head against a wall.
        share = 3 if rng.random() < 0.5 else 8
        for (r, c) in cells[:rng.randint(0, len(cells) // share)]:
            eng.grid[r][c].add(idx["wall"])
        free = [x for x in cells if idx["wall"] not in eng.grid[x[0]][x[1]]]
        nred = nblue = rng.randint(1, 3)
        if len(free) < 2 * nred + 2:
            continue
        rng.shuffle(free)
        for (r, c) in free[:nred]:
            eng.grid[r][c].add(idx["player"])
        for (r, c) in free[nred:nred + nblue]:
            eng.grid[r][c].add(idx["blueplayer"])
        # Exits go anywhere including under a body: the win flag has to agree
        # with the interpreter on a board that starts half-solved too.
        pool = [x for x in cells]
        rng.shuffle(pool)
        for (r, c) in pool[:nred]:
            eng.grid[r][c].add(idx["redexit"])
        for (r, c) in pool[nred:nred + nblue]:
            if idx["redexit"] not in eng.grid[r][c]:
                eng.grid[r][c].add(idx["blueexit"])
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        rollout(f"board{t}", board_steps, rng)

    print(f"selfcheck: {tot['presses']} presses "
          f"({game.n_levels} levels x {trials} rollouts + {boards} random "
          f"boards), {tot['noops']} no-ops, {tot['wins']} incidental wins")
    print(f"  coverage: {tot['headon']} head-on stand-offs, "
          f"{tot['contested']} contested cells, {tot['walled']} chains stopped "
          f"by a wall or an edge, {tot['moves']} chains moved of which "
          f"{tot['chains']} were same-team chains, {tot['desync']} presses that "
          f"moved one team only, {tot['body_on_wrong_exit']} states with a body "
          f"parked on the other team's exit")
    weak = [k for k in ("headon", "contested", "walled", "moves", "chains",
                        "desync", "body_on_wrong_exit") if not tot[k]]
    if weak:
        print(f"FUZZ TOO WEAK: never exercised {weak}")
        return 1
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

    dmap = {}
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
                 for o in (idx["player"], idx["blueplayer"]))


def _chirality_probe(game, expert, rollouts: int = 12, steps: int = 25) -> int:
    """The second half of `_symmetry`, and the half with teeth.

    Replaying a level's PLAN under the eight transforms cannot find a mechanic
    the plan does not happen to use, and this game has exactly one place a
    mirror could bite: phase 1 of `_resolve_forces` reads the LIVE force table
    while it is being emptied, so which chain is traced FIRST is a fact about the
    order the interpreter enumerates forces -- an ENGINE fact, which no
    presentation transform can change. (The argument that it cannot matter is in
    `_Model.step`: a chain blocked behind a blocked chain is blocked either way.
    This stage is what turns that argument into a measurement.)

    So this stage does not follow a plan. It random-walks each level, records
    every ``(board, press)`` it visits, and re-plays each one INDEPENDENTLY on
    all seven other presentations of that same board. A single order-sensitive
    transition anywhere in the walk is a failure.

    ``load_level`` is exact for a mid-walk board here: the game does not set
    ``run_rules_on_level_start``, so loading is a plain grid assignment.
    """
    eng, g = game._engine, game._game
    bad = tot = 0
    for level in range(game.n_levels):
        game.set_level(level)
        hw = (eng.height, eng.width)
        triples = []
        for t in range(rollouts):
            game.set_level(level)
            rng = random.Random(f"opposition:chiral:{level}:{t}")
            for _ in range(steps):
                before = [[set(cell) for cell in row] for row in eng.grid]
                press = KEYS[rng.randrange(4)]
                eng.step(press)
                triples.append((before, press, _pieces(eng, g)))
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
        print(f"level {level}: {len(triples):4d} random presses x 7 "
              f"presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print(f"chirality probe: {tot} transformed presses checked, {bad} chiral")
    return bad


def _symmetry() -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an exact
    symmetry of the mechanic, which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation.

    Reading the rules and declaring them direction-free is exactly the argument
    that was wrong in ps:gobble_rush and ps:lovendpieces. Here the rule really is
    written with the relative `>` and `<` forces and the win condition names no
    direction, so the risk is entirely in the interpreter's force resolution --
    which is what stage 2 goes after.

    Stage 1 (here) replays each level's plan on all eight turned and mirrored
    copies of its own board, built by transforming the LEVEL LAYOUT so the
    interpreter re-loads it itself; every press must leave both teams exactly
    where the transform of the reference run put them, and the run must still
    win. Stage 2 (`_chirality_probe`) is what can actually reach the contested
    configurations -- see its docstring."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level}: no plan (unwinnable) -- skipped")
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
        print(f"level {level}: {len(plan):3d} plan presses x 7 presentations: "
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
    """Assert every cell COMPOSITION a board can show is distinct at the cell
    size in use.

    The composition, not the object, is what has to be readable: RedExit and
    BlueExit sit on a layer BELOW Player, BluePlayer and Wall, so six of the ten
    cells this game can paint are stacks -- and the win condition is read off
    four of them ("is the red body showing the RED exit through its gaps, or the
    blue one"). Both bodies ship with eleven transparent pixels, which is why no
    sprite change was needed here.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    lower = {"": (), "redexit": ("redexit",), "blueexit": ("blueexit",)}
    upper = {"": (), "wall": ("wall",), "red": ("player",),
             "blue": ("blueplayer",)}
    # An exit UNDER a wall is the one stack left out, and it is left out because
    # no board should show it: walls are static, and an exit sealed under one is
    # a win condition no body could ever satisfy. That is asserted below rather
    # than assumed -- it is the only thing standing between this audit and the
    # opaque wall sprite, which does hide what is beneath it.
    comps = {f"{a or 'floor'}+{b or 'nothing'}": lower[a] + upper[b]
             for a in lower for b in upper if not (a and b == "wall")}

    sizes: dict = {}
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)
        for r in range(eng.height):
            for c in range(eng.width):
                cell = eng.grid[r][c]
                if idx["wall"] in cell and (idx["redexit"] in cell
                                            or idx["blueexit"] in cell):
                    bad += 1
                    print(f"level {level}: an exit is sealed under a wall at "
                          f"({r},{c}) -- that stack is unreadable")

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
        print(f"{h}x{w} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(shots)} compositions: "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad else
          f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plan_report())
    if "--selfcheck" in sys.argv:
        sys.exit(1 if selfcheck() else 0)
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(OppositionSolver.main())
