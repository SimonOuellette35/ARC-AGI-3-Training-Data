"""Generate Phase-1 training data for the PuzzleScript game
ps:the_observable_universe ("The Observable Universe", Paul Jeffrey).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: an exact model of the
mechanic, an exhaustive optimal solver over it, and the reports that certify
both.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_the_observable_universe",
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
Two universes are painted side by side on ONE grid and you steer both at once.
A press sends the pink universe's body (`Player`) that way and the white
universe's body (`Player2`) the OPPOSITE way, and you win only when both are
home AT THE SAME TIME -- `Some Target on Player and Some Target2 on Player2`.
There is no ACTION rule at all, so ACTION5 is a strict no-op and the whole
vocabulary is the four arrows.

The two universes are not two boards. They are one board with two independent
COLLISION LAYERS, and that is the entire mechanic:

    layer 3   Wall   Player   Crate2      <- the pink universe
    layer 4   Wall2  WallR  Player2  Crate3   <- the white universe

Nothing on layer 3 can block anything on layer 4 by collision, so `Player` walks
straight through `Wall2` and `Player2` walks straight through `Wall`. The only
way the universes touch each other is through four rules that fire ACROSS them,
and each of the four is worth stating because each one is a fact the model has
to get exactly right:

  * **A CRATE OF THE OTHER UNIVERSE IS A WALL.** ``[ > Player | Crate3 ] ->
    [ Player | Crate3 ]`` (and the same for `Crate2` against `Player2`, and for
    each crate against the other crate) strips the FORCE off the mover instead
    of cancelling. So being stopped by a foreign crate stops ONE body and the
    other one still takes its half of the press. Being stopped by your own wall
    does not: that is a ``cancel``, and a cancel drops the whole turn for both.
  * **YOU CANNOT PARK A CRATE ON A GOAL.** ``[ > Crate2 | Target ] -> cancel``
    and its mirror. The goals are the only squares in the game a crate may not
    enter, and they are enforced by dropping the turn -- so a press that would
    have shoved a crate onto a goal is a no-op for BOTH bodies.
  * **`PlayerR` IS THE PLAYER'S SHADOW, AND IT CAN COME OFF.** `R` seats a
    `Player` and a `PlayerR` in one cell; the shadow lives on its own layer, is
    handed the same force by ``[ > Player ] [ PlayerR ] -> [ > Player ] [ >
    PlayerR ]``, and carries the game's wall cancels (``[ > PlayerR | Wall ]``
    and ``[ > PlayerR | Crate2 | Wall ]``) on the Player's behalf -- the Player
    itself has no cancel rule anywhere. That indirection has a hole in it. When
    the Player is stopped by a `Crate2` whose own force was stripped by a
    `Crate3` behind it, nothing cancels, the Player is blocked by plain layer
    collision, and `PlayerR` -- which shares a layer with nothing on these
    boards -- walks on alone. The shadow then carries the wall cancels from the
    WRONG cell. It is a bug in the shipped game, it is reachable, and the model
    keeps `PlayerR` as its own coordinate rather than assuming the two stay
    together -- `--selfcheck` counts the states its fuzz reaches with them
    apart (1023 of 11705 presses), and `--plans` reports that no shortest plan
    needs one.
  * **`WallR` IS A `Wall2` UNDER ANOTHER NAME.** Same layer, same two cancels,
    no other rule mentions it. It is the (originally invisible -- see the .txt
    header) diagonal seam the author draws between the two halves.

There is no way to lose and no lethal state: no rule deletes anything, no rule
creates anything, and the only irreversible move is a crate pushed into a
pocket. Recovery is therefore a RESET (`PSAStarSolver.recovery_mode`).

THE LEVELS
----------
Six levels (the seven `Message` screens between them are not levels), all six
winnable, all six recorded, and every plan here is proved SHORTEST rather than
merely found -- `_Model.solve` is a breadth-first sweep of the level's own state
space, so the length it returns is the true distance. `--plans` prints this
table and CERTIFIES every plan by replaying it through the interpreter:

    level   size    crates   reachable   visited    d*   debt   ties
      0     3x10      0+0            4         3     3      3   1.0
      1    10x10      5+5      >200000        19     6      6   1.2
      2    10x10      6+7      >200000        31     7      5   1.4
      3    10x10      5+6          714        22     6      4   1.0
      4    10x15      9+9      >200000       531    11      3   1.1
      5    10x15      8+7      >200000     17546    15     15   1.1

``visited`` is what the depth-bounded search actually touches and ``reachable``
the level's whole reachable state space, counted only up to `_REPORT_CAP` --
four of the six run into the millions, and the gap is exactly the point of
stopping at the first winning LAYER instead of sweeping: level 5 answers from
17546 states. ``ties`` is the mean size of the optimal-press set along the plan,
i.e. how much of the labelling would be a lie if the recorder trained one
arbitrary shortest path as the only right answer. The whole run of six searches
is under a second, which is why this expert carries no on-disk plan cache -- the
other end of the trade the ps: family makes when a search is the whole cost of
generation.

``debt`` is the level's DIRECTION DEBT and it is where the difficulty lives. A
press moves `Player` by ``d`` and `Player2` by ``-d``, so the ``down`` presses
have to pay for the Player's downward need AND `Player2`'s upward one out of the
same count; summing the four per-direction minima is therefore a lower bound on
any win, and `--plans` asserts d* never falls below it. On levels 0, 1 and 5 the
bound is TIGHT -- the author lined the two goals up so the two universes ask for
the same thing and the shortest route is a straight direction budget, crates or
no crates. On levels 2, 3 and especially 4 (debt 3, d* 11) it is not: those
boards put a wall between a body and the only route that pays both debts at
once, and the extra presses are the ones spent walking one body into a wall so
the OTHER one can move alone. That is the whole game, and it is why a search is
needed rather than an arithmetic.

THE .txt REPAINT
----------------
Every sprite was repainted, and the reasoning is in the header comment of
`data/puzzlescript_games/The_Observable_Universe.txt`. In one line: `Wall2` was
solid Pink on a solid Pink `Background` and `WallR` was fully transparent, so
half the game's walls did not exist on screen; and `Target` and `Player2` shipped
as pixel-identical pictures (as did `Target2` and `Player`), so the ONE thing
`WINCONDITIONS` names could not be read off a frame. Each collision layer now
owns one 2x2 corner of the cell, which is the only scheme that survives the 4px
cells levels 4 and 5 render at while still showing all four objects that stack
on the Player's goal square. No rule, legend entry, layer, win condition or
level was touched. `--audit` is the check.

EXPERT SOLVER
-------------
`_Model` is the mechanic over five ints -- `Player`, `PlayerR` and `Player2` as
cell indices plus one bitmask per crate colour -- fuzz-verified against the real
interpreter by `--selfcheck` on the shipped levels AND on random boards built to
hold the configurations no shipped level reaches (a crate jammed by a foreign
crate, the shadow walking off on its own, a crate run shoved at a goal, the two
crate colours stacked in one cell). It exists because the interpreter runs at
**~50 presses per second** on these boards: level 5's sweep expands 17546 states
= ~88000 presses, which is half an hour through `eng.step` and 0.33 s through
the model.

`_Model.solve` then does the exact thing:

  1. a breadth-first sweep from the level start, layer by layer, stopping at the
     first layer that contains a winning press -- so ``d*`` is the true shortest
     distance and an exhausted sweep with no win would be a PROOF of
     unwinnability (winning is a transition, not a state: `check_win` is read
     after a press and a won level is terminal, so those edges lead nowhere);
  2. a backward pass over those same layers keeping only the states that reach a
     win in exactly the remaining number of presses -- the shortest-path DAG;
  3. the plan is any walk down it, and the optimal SET at each step is every
     press whose successor is still in the DAG.

Keeping LAYERS rather than a predecessor map is what makes step 2 cheap: it
needs only the states, re-deriving each successor as it goes, so nothing ever
stores an edge.

Usage
-----
    python solvers/generate_the_observable_universe_training.py --episodes 200 \
        --out data/the_observable_universe
    python solvers/generate_the_observable_universe_training.py --plans
    python solvers/generate_the_observable_universe_training.py --selfcheck
    python solvers/generate_the_observable_universe_training.py --symmetry
    python solvers/generate_the_observable_universe_training.py --audit
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

GAME_NAME = "The_Observable_Universe"
GAME_MODULE_ID = "ps:the_observable_universe"

#: The presses the search branches on, in the order it tries them. ACTION5 is in
#: the list only to be MEASURED: the game has no `[ Action ... ]` rule, so it
#: assigns no force and cannot change the board. `--plans` asserts that.
KEYS = ("up", "down", "left", "right", "action")
_ACTION = KEYS.index("action")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
#: index of the opposite of each of the four directions (ACTION maps to itself)
_OPP = (1, 0, 3, 2, 4)


# ---------------------------------------------------------------------------
# Native model of the mechanic
# ---------------------------------------------------------------------------

class _Model:
    """The Observable Universe as a state machine over three cell indices and
    two bitmasks.

    A state is ``(p, pr, q, m2, m3)``: the `Player`, its `PlayerR` shadow and
    `Player2` as cell indices ``r * w + c``, and one int per crate colour whose
    set bits are that colour's crates. Everything static -- `Wall`, `Wall2`,
    `WallR` and the two goals -- lives on the model.

    ``nbr[di][cell]`` is the neighbour of ``cell`` in direction ``di``, or -1 off
    the board, precomputed so `step` never does row/column arithmetic (which is
    where an index model wraps around an edge without noticing).

    The two layers are modelled separately because the interpreter resolves them
    separately: `Wall`/`Player`/`Crate2` share one collision layer and
    `Wall2`/`WallR`/`Player2`/`Crate3` share another, so a chain on one can never
    contest a cell with a chain on the other (`PSEngine._resolve_forces` keys its
    conflict table by ``(row, col, LAYER)``). `PlayerR` is a third layer holding
    nothing else at all on these boards, which is exactly why it can never be
    blocked and can walk off on its own.
    """

    __slots__ = ("h", "w", "wall", "wall2", "wallr", "block3",
                 "target", "target2", "nbr")

    def __init__(self, h: int, w: int, wall: int, wall2: int, wallr: int,
                 target: int, target2: int):
        self.h, self.w = h, w
        self.wall = wall
        self.wall2, self.wallr = wall2, wallr
        #: `Wall2` and `WallR` are the same object as far as every rule that
        #: mentions either is concerned -- same layer, same two cancels -- so the
        #: model carries their union and never asks which one it hit.
        self.block3 = wall2 | wallr
        self.target, self.target2 = target, target2
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

    # -- the press ------------------------------------------------------------
    def step(self, state, di: int):
        """The settled board after one press, in the interpreter's own order:
        assign forces, run the push rules, run the block rules, run the cancels,
        then resolve each layer's chains.

        Returns ``state`` itself for a press that changes nothing -- including
        ACTION, which assigns no force at all because no rule in this game
        mentions it.
        """
        if di == _ACTION:
            return state
        p, pr, q, m2, m3 = state
        nA, nB = self.nbr[di], self.nbr[_OPP[di]]

        # -- the `+` rule group: Player and PlayerR take the press, Player2
        #    takes its opposite. Everything below is a consequence of those.
        forced_p = forced_pr = forced_q = True

        # -- push rules, own-universe. Seeded from BOTH the Player and its
        #    shadow, because `[ > PlayerR | Crate2 ]` is its own rule: once the
        #    two are apart the shadow shoves a second, independent crate run.
        push2 = set()
        for seed in (p, pr):
            cur = nA[seed]
            while cur >= 0 and (m2 >> cur) & 1 and cur not in push2:
                push2.add(cur)
                cur = nA[cur]
        push3 = set()
        cur = nB[q]
        while cur >= 0 and (m3 >> cur) & 1 and cur not in push3:
            push3.add(cur)
            cur = nB[cur]

        # -- block rules, foreign-universe. These run AFTER the pushes, so a
        #    crate that was already handed a force keeps it even when the mover
        #    behind it is stopped: pushing a Crate2 that shares its cell with a
        #    Crate3 moves the crate and leaves the Player standing.
        if nA[p] >= 0 and (m3 >> nA[p]) & 1:
            forced_p = False
        if nA[pr] >= 0 and (m3 >> nA[pr]) & 1:
            forced_pr = False
        push2 = {x for x in push2 if not (nA[x] >= 0 and (m3 >> nA[x]) & 1)}
        if nB[q] >= 0 and (m2 >> nB[q]) & 1:
            forced_q = False
        push3 = {x for x in push3 if not (nB[x] >= 0 and (m2 >> nB[x]) & 1)}

        # -- cancels. Any one of them drops the WHOLE turn, both universes with
        #    it, and every one asks whether the mover still has its force -- a
        #    body or crate the block rules just stopped cancels nothing.
        if forced_pr:
            n = nA[pr]
            if n >= 0 and (self.wall >> n) & 1:
                return state                              # [ > PlayerR | Wall ]
            if n >= 0 and (m2 >> n) & 1:
                n2 = nA[n]
                if n2 >= 0 and (self.wall >> n2) & 1:
                    return state                # [ > PlayerR | Crate2 | Wall ]
        if forced_q:
            n = nB[q]
            if n >= 0 and (self.block3 >> n) & 1:
                return state       # [ > Player2 | Wall2 ] / [ ... | WallR ]
        for x in push2:
            n = nA[x]
            if n >= 0 and ((self.wall >> n) & 1 or (self.target >> n) & 1):
                return state       # [ > Crate2 | Wall ] / [ > Crate2 | Target ]
        for x in push3:
            n = nB[x]
            if n >= 0 and ((self.block3 >> n) & 1 or (self.target2 >> n) & 1):
                return state       # [ > Crate3 | Wall2 ] / [ ... | Target2 ]

        # -- resolve. Within one layer every force points the same way, so the
        #    movers form disjoint runs along that axis and no two runs can ever
        #    claim the same cell (their destinations are shifted by one from
        #    sets already a cell apart). That is what makes phases 2 and 3 of
        #    `_resolve_forces` -- subsumption and multi-way conflict -- vacuous
        #    here, and it is why `_advance` only has to answer one question per
        #    run: is the cell past its front free?
        forced2 = set(push2)
        if forced_p:
            forced2.add(p)
        moved2 = self._advance(forced2, nA,
                               lambda cell: ((self.wall >> cell) & 1
                                             or (m2 >> cell) & 1 or cell == p))
        forced3 = set(push3)
        if forced_q:
            forced3.add(q)
        moved3 = self._advance(forced3, nB,
                               lambda cell: ((self.block3 >> cell) & 1
                                             or (m3 >> cell) & 1 or cell == q))

        np_, nq, npr = p, q, pr
        nm2, nm3 = m2, m3
        # Clear every departing crate BEFORE placing any arrival: a run moves as
        # a unit, so the second crate's origin is the first crate's destination
        # and a clear-then-set per crate deletes one of them.
        for x in moved2:
            if x == p:
                np_ = nA[p]
            else:
                nm2 &= ~(1 << x)
        for x in moved2:
            if x != p:
                nm2 |= 1 << nA[x]
        for x in moved3:
            if x == q:
                nq = nB[q]
            else:
                nm3 &= ~(1 << x)
        for x in moved3:
            if x != q:
                nm3 |= 1 << nB[x]
        # The shadow shares its layer with nothing on these boards, so the only
        # thing that can stop it is the edge of the grid.
        if forced_pr and nA[pr] >= 0:
            npr = nA[pr]
        return (np_, npr, nq, nm2, nm3)

    @staticmethod
    def _advance(forced, nbr, occupied):
        """The forced cells that actually move: walk each one forward through
        the run of same-forced cells ahead of it and ask whether the cell past
        the front is on the board and free of its own layer."""
        moved = set()
        for cell in forced:
            cur = cell
            while nbr[cur] in forced:
                cur = nbr[cur]
            front = nbr[cur]
            if front >= 0 and not occupied(front):
                moved.add(cell)
        return moved

    def won(self, state) -> bool:
        return bool((self.target >> state[0]) & 1
                    and (self.target2 >> state[2]) & 1)

    # -- the exhaustive optimal solver ---------------------------------------
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

        ``cap`` is a runaway guard on a board that does not belong to this game,
        not a tuning dial: the widest shipped level stores 33041 states, because
        the sweep stops at the layer that wins and its ``d*`` is 15.
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
                for di in range(len(KEYS)):
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
                       if any(self.won(self.step(s, di))
                              for di in range(len(KEYS)))}
        for k in range(depth - 1, -1, -1):
            ahead = good[k + 1]
            good[k] = {s for s in layers[k]
                       if any(self.step(s, di) in ahead
                              for di in range(len(KEYS)))}

        presses, optsets = [], []
        s = start
        for k in range(depth + 1):
            if k == depth:
                best = [KEYS[di] for di in range(len(KEYS))
                        if self.won(self.step(s, di))]
            else:
                ahead = good[k + 1]
                best = [KEYS[di] for di in range(len(KEYS))
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
            for di in range(len(KEYS)):
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

    Everything static (the three wall kinds and the two goals) goes on the model
    and everything that moves goes in the state.

    Two assertions rather than assumptions. `Player2R` -- the second universe's
    shadow -- has a legend entry (``*``) that no shipped level uses, and the
    model does not carry it; a board that seated one would silently lose its
    cancels, so reading one is an error rather than a shrug. And the three
    bodies are asserted to exist, because "no Player" would make `won` read
    False forever instead of raising.
    """
    idx = g.obj_name_to_idx
    h, w = eng.height, eng.width
    wall = wall2 = wallr = target = target2 = m2 = m3 = 0
    p = pr = q = -1
    for r in range(h):
        for c in range(w):
            cell = eng.grid[r][c]
            i = r * w + c
            if idx["wall"] in cell:
                wall |= 1 << i
            if idx["wall2"] in cell:
                wall2 |= 1 << i
            if idx["wallr"] in cell:
                wallr |= 1 << i
            if idx["target"] in cell:
                target |= 1 << i
            if idx["target2"] in cell:
                target2 |= 1 << i
            if idx["crate2"] in cell:
                m2 |= 1 << i
            if idx["crate3"] in cell:
                m3 |= 1 << i
            if idx["player"] in cell:
                p = i
            if idx["playerr"] in cell:
                pr = i
            if idx["player2"] in cell:
                q = i
            if idx["player2r"] in cell:
                raise ValueError("a Player2R is on the board; the model has no "
                                 "coordinate for the second universe's shadow")
    if p < 0 or pr < 0 or q < 0:
        raise ValueError("board is missing Player, PlayerR or Player2")
    return (_Model(h, w, wall, wall2, wallr, target, target2),
            (p, pr, q, m2, m3))


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class ObservableUniverseExpert(PSExpert):
    """Exhaustive optimal expert: `_Model.solve` from the current board, with
    the plan and the per-step optimal sets read off the shortest-path DAG.

    It plugs into the shared harness at `PSExpert._search`, so the plan memo, the
    snapshot/restore discipline and the level scoping all come from the base.
    `heuristic` is never called -- `_astar` is not the strategy here -- and
    neither is `dead`: nothing in this game is lethal, and a sweep that exhausts
    the component has already proved what `dead` would guess.
    """

    directions = list(KEYS)

    # No `plan_cache_path`: all six searches together are under a second, so a
    # disk cache would only add a staleness surface. The base's in-memory memo
    # (keyed on the whole grid, which is also what makes it exact across levels)
    # is enough -- every seed replans from the same level starts and hits it.

    #: A runaway guard on a board that does not belong in this file, not a
    #: tuning dial -- see `_Model.solve`.
    state_cap: int = 2_000_000

    def _search(self, eng):
        model, state = model_from_engine(eng, self.g)
        presses, optsets = model.solve(state, self.state_cap)
        if presses is None:
            return None
        return Plan(presses, optsets)


class ObservableUniverseSolver(PSAStarSolver):
    game_id = "puzzlescript_the_observable_universe"
    game_name = GAME_NAME
    expert_cls = ObservableUniverseExpert
    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so the generator can never tape frames from a different
    #: build of the game than the agent plays.
    game_module_id = GAME_MODULE_ID

    #: The longest plan is 15 presses; the rest is room for the exploration
    #: prefix and the replay after its RESET. Comfortably inside the adapter's
    #: own 200-step per-level budget.
    max_steps = 100


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    solver = ObservableUniverseSolver()
    game = solver.make_game(0)
    return solver, game, ObservableUniverseExpert(game)


#: How much of a level's reachable space `--plans` bothers to count. Four of the
#: six are in the millions and the number is a curiosity, not a claim the solver
#: rests on -- the sweep that DOES matter stops at the winning layer.
_REPORT_CAP = 200_000


def _debt(model, state) -> int:
    """The DIRECTION DEBT of a state: the presses it would take if no wall ever
    got in the way.

    A press moves the `Player` by ``d`` and `Player2` by ``-d``, so the count of
    ``down`` presses must cover both the Player's downward need and Player2's
    upward one, and likewise for each of the other three directions. Summing the
    four per-direction minima is a LOWER BOUND on any win -- it ignores every
    wall, every crate and the shadow -- and `--plans` asserts d* never falls
    below it. The slack it leaves is the level's whole difficulty; see the
    module docstring.
    """
    w = model.w
    p, _pr, q, _m2, _m3 = state
    t1 = next(i for i in range(model.h * w) if (model.target >> i) & 1)
    t2 = next(i for i in range(model.h * w) if (model.target2 >> i) & 1)
    dr1, dc1 = t1 // w - p // w, t1 % w - p % w
    dr2, dc2 = t2 // w - q // w, t2 % w - q % w
    return (max(dr1, -dr2, 0) + max(-dr1, dr2, 0)      # down + up
            + max(dc1, -dc2, 0) + max(-dc1, dc2, 0))   # right + left


def _plan_report() -> int:
    """Per-level size, crate counts, reachable-space size, states the search
    visits, plan length and tie richness -- the table in the module docstring.

    Every plan is CERTIFIED by replaying it through the real interpreter, and
    every optimal label is checked the expensive way: each press the label calls
    equally-shortest is taken on the model and re-solved from there, and it has
    to leave exactly one press fewer than the step it was offered at.

    Three further claims the module docstring makes are asserted here rather
    than left as prose: ACTION5 never changes the board, ``d*`` equals the
    level's direction debt, and both goals are unique.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    total = ties_total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        model, state = model_from_engine(eng, game._game)
        plan = expert.plan(eng, level)
        c2, c3 = bin(state[3]).count("1"), bin(state[4]).count("1")
        space, capped = model.reachable(state, _REPORT_CAP)
        if plan is None:
            bad += 1
            why = ("unproven, the sweep hit the cap" if capped
                   else "unwinnable (component exhausted)")
            print(f"level {level}: {eng.height}x{eng.width} {c2}+{c3} crates, "
                  f"NO PLAN -- {why} ({len(space)} states)")
            continue

        for name, mask in (("Target", model.target), ("Target2", model.target2)):
            if bin(mask).count("1") != 1:
                bad += 1
                print(f"  L{level}: {bin(mask).count('1')} {name}s, expected 1")

        # Re-derive what the depth-bounded search touches, for the table.
        seen = {state}
        layer = [state]
        for _ in range(len(plan) - 1):
            nxt = []
            for s in layer:
                for di in range(len(KEYS)):
                    ns = model.step(s, di)
                    if model.won(ns) or ns in seen:
                        continue
                    seen.add(ns)
                    nxt.append(ns)
            layer = nxt

        # Certify the plan on the interpreter and the labels on the model.
        s = state
        apart = 0
        for i, press in enumerate(plan):
            remaining = len(plan) - i - 1
            if press not in plan.optsets[i]:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            if model.step(s, _ACTION) != s:
                bad += 1
                print(f"  L{level}: step {i} -- ACTION5 changed the board, and "
                      f"no rule in this game mentions it")
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
            apart += s[0] != s[1]
            eng.step(press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        debt = _debt(model, state)
        if debt > len(plan):
            bad += 1
            print(f"  L{level}: d* is {len(plan)} against a direction debt of "
                  f"{debt}, which is supposed to be a lower bound -- either the "
                  f"search or the bound is wrong")

        ties = sum(len(o) for o in plan.optsets) / len(plan)
        total += len(plan)
        ties_total += sum(1 for o in plan.optsets if len(o) > 1)
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"level {level}: {eng.height}x{eng.width} {c2}+{c3} crates, "
              f"{'>' if capped else ' '}{len(space):6d} reachable, "
              f"{len(seen):6d} visited, d* {len(plan):2d} ({room}) vs "
              f"debt {debt:2d}, ties {ties:.1f}, "
              f"{apart} shadow-apart states on the plan")
    print(f"total {total} presses, {ties_total} with a tie "
          f"({ties_total / max(1, total):.0%}), {bad} problems")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# Model self-check (fuzz against the real interpreter)
# ---------------------------------------------------------------------------

def _read(eng, g):
    """The dynamic state as the interpreter has it."""
    return model_from_engine(eng, g)[1]


def selfcheck(trials: int = 30, steps: int = 40,
              boards: int = 250, board_steps: int = 20) -> int:
    """Audit the claim the expert rests on: `_Model` reproduces the interpreter
    EXACTLY, on the settled state and on the win flag, for every press.

    Two fuzz populations, and the second is the one with teeth. The six shipped
    levels exercise the game as it is played; random boards -- both wall kinds
    scattered, both crate colours scattered, the three bodies dropped anywhere,
    and both goals placed where a crate can actually be shoved at them --
    exercise the configurations the shipped boards reach rarely or never: the
    Player jammed by a foreign crate while `Player2` walks on, the shadow coming
    off the Player, a crate run pushed into a goal, the two universes' crates
    stacked in one cell. The coverage counters below are printed and ASSERTED
    non-zero, because a fuzz that never built those boards would report a clean
    model while having tested none of the branches that make this game more than
    two mazes side by side.

    Note the interpreter runs at ~50 presses/second here, so the default
    populations are sized to a couple of minutes rather than to a number that
    looks impressive: the point is coverage of the branches, and the counters
    say whether it was had.
    """
    game = _load_game_module(GAME_MODULE_ID).make_game(seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    bg = idx["background"]
    tot = {"presses": 0, "noops": 0, "wins": 0, "cancels": 0, "desync": 0,
           "apart": 0, "jammed": 0, "pushes": 0, "chains": 0, "both_push": 0,
           "goal_cancel": 0, "stacked_crates": 0, "actions": 0}
    bad = 0

    def rollout(tag: str, nsteps: int, rng: random.Random) -> int:
        nonlocal bad
        model, state = model_from_engine(eng, g)
        for _ in range(nsteps):
            di = rng.randrange(len(KEYS))
            p, pr, q, m2, m3 = state

            # Coverage, classified BEFORE the press from the same quantities
            # `step` computes -- so what the counters report is what the model
            # did, not a separate guess at it.
            tot["apart"] += p != pr
            tot["stacked_crates"] += bool(m2 & m3)
            if di == _ACTION:
                tot["actions"] += 1
            else:
                nA, nB = model.nbr[di], model.nbr[_OPP[di]]
                jam = ((nA[p] >= 0 and (m3 >> nA[p]) & 1)
                       or (nB[q] >= 0 and (m2 >> nB[q]) & 1))
                tot["jammed"] += jam
                run2, cur = 0, nA[p]
                while cur >= 0 and (m2 >> cur) & 1:
                    run2 += 1
                    cur = nA[cur]
                run3, cur = 0, nB[q]
                while cur >= 0 and (m3 >> cur) & 1:
                    run3 += 1
                    cur = nB[cur]
                tot["pushes"] += run2 + run3 > 0
                tot["chains"] += run2 > 1 or run3 > 1
                tot["both_push"] += run2 > 0 and run3 > 0
                if run2 and (model.target >> _end(nA, nA[p], run2)) & 1:
                    tot["goal_cancel"] += 1
                if run3 and (model.target2 >> _end(nB, nB[q], run3)) & 1:
                    tot["goal_cancel"] += 1

            eng.step(KEYS[di])
            tot["presses"] += 1
            expected = model.step(state, di)
            actual = _read(eng, g)
            if actual != expected:
                bad += 1
                print(f"  {tag}: state divergence on {KEYS[di]}: "
                      f"model {expected} engine {actual}")
                return 1
            if eng.check_win() != model.won(expected):
                bad += 1
                print(f"  {tag}: win divergence on {KEYS[di]}")
                return 1
            tot["noops"] += expected == state
            if di != _ACTION and expected == state:
                tot["cancels"] += 1
            tot["desync"] += (expected[0] != state[0]) != \
                             (expected[2] != state[2])
            state = expected
            if eng.check_win():
                tot["wins"] += 1
                return 0
        return 0

    for level in range(game.n_levels):
        for t in range(trials):
            game.set_level(level)
            rollout(f"L{level}/{t}", steps,
                    random.Random(f"tou:shipped:{level}:{t}"))

    for t in range(boards):
        rng = random.Random(f"tou:board:{t}")
        h, w = rng.randint(5, 8), rng.randint(5, 8)
        game.set_level(0)
        eng.height, eng.width = h, w
        eng.grid = [[{bg} for _ in range(w)] for _ in range(h)]
        # A full border of BOTH wall kinds, so neither universe can walk off the
        # grid: an edge blocks silently where a wall cancels, and mixing the two
        # would test the edge case instead of the rule.
        for r in range(h):
            for c in range(w):
                if r in (0, h - 1) or c in (0, w - 1):
                    eng.grid[r][c] |= {idx["wall"], idx["wall2"]}
        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        rng.shuffle(cells)
        n_inner = len(cells)
        for (r, c) in cells[:rng.randint(0, n_inner // 5)]:
            eng.grid[r][c].add(rng.choice([idx["wall"], idx["wall2"],
                                           idx["wallr"]]))
        free = [x for x in cells
                if not (eng.grid[x[0]][x[1]] & {idx["wall"], idx["wall2"],
                                                idx["wallr"]})]
        if len(free) < 5:
            continue
        rng.shuffle(free)
        # Dense crate fields on half the boards: that is what makes a run long
        # enough to chain, and long runs are what put a crate against a goal.
        share = 2 if rng.random() < 0.5 else 5
        for (r, c) in free[3:3 + rng.randint(0, len(free) // share)]:
            for name in ("crate2", "crate3"):
                if rng.random() < 0.6:
                    eng.grid[r][c].add(idx[name])
        (pr_, pc), (qr, qc) = free[0], free[1]
        eng.grid[pr_][pc].add(idx["player"])
        # The shadow starts on the Player on every shipped level, but a fifth of
        # the boards start it apart so the desynced cancels are exercised from
        # the first press instead of only after a jam happens to occur.
        sr, sc = (pr_, pc) if rng.random() < 0.8 else free[2]
        eng.grid[sr][sc].add(idx["playerr"])
        eng.grid[qr][qc].add(idx["player2"])
        # Goals go on cells a crate CAN be shoved at, which is the only way
        # `[ > Crate2 | Target ] -> cancel` ever fires.
        goals = [x for x in free if x not in (free[0], free[1], (sr, sc))]
        if len(goals) < 2:
            continue
        eng.grid[goals[0][0]][goals[0][1]].add(idx["target"])
        eng.grid[goals[1][0]][goals[1][1]].add(idx["target2"])
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        rollout(f"board{t}", board_steps, rng)

    print(f"selfcheck: {tot['presses']} presses "
          f"({game.n_levels} levels x {trials} rollouts + {boards} random "
          f"boards), {tot['noops']} no-ops, {tot['wins']} incidental wins")
    print(f"  coverage: {tot['cancels']} cancelled turns, {tot['jammed']} "
          f"presses into a foreign crate, {tot['desync']} that moved one "
          f"universe only, {tot['apart']} states with the shadow off the "
          f"Player, {tot['pushes']} pushes of which {tot['chains']} were "
          f"multi-crate chains and {tot['both_push']} pushed in both universes, "
          f"{tot['goal_cancel']} crate runs aimed at a goal, "
          f"{tot['stacked_crates']} states with the two crate colours stacked, "
          f"{tot['actions']} ACTION presses")
    weak = [k for k in ("cancels", "jammed", "desync", "apart", "pushes",
                        "chains", "both_push", "goal_cancel", "stacked_crates",
                        "actions") if not tot[k]]
    if weak:
        print(f"FUZZ TOO WEAK: never exercised {weak}")
        return 1
    return bad


def _end(nbr, first: int, run: int) -> int:
    """The cell one past a run of ``run`` crates starting at ``first``."""
    cur = first
    for _ in range(run):
        cur = nbr[cur]
    return cur


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
                 for o in (idx["player"], idx["playerr"], idx["player2"],
                           idx["crate2"], idx["crate3"]))


def _chirality_probe(game, expert, rollouts: int = 8, steps: int = 20) -> int:
    """The second half of `_symmetry`, and the half with teeth.

    Replaying a level's PLAN under the eight transforms cannot find a mechanic
    the plan does not happen to use, and this game's plans use almost none of
    it: they are 3 to 15 presses long and never need to jam a crate or shed the
    shadow. What is at risk is exactly that kind of mechanic -- the order
    `_apply_rules_with_forces` walks a rule's matches decides which crate in a
    contested run keeps its force, and that is a fact about the ENGINE's scan
    direction, which no presentation transform can change.

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
        if expert.plan(eng, level) is None:
            continue
        hw = (eng.height, eng.width)
        triples = []
        for t in range(rollouts):
            game.set_level(level)
            rng = random.Random(f"tou:chiral:{level}:{t}")
            for _ in range(steps):
                before = [[set(cell) for cell in row] for row in eng.grid]
                press = KEYS[rng.randrange(len(KEYS))]
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
    that was wrong in ps:gobble_rush and ps:lovendpieces. Here the suspicious
    part is that a press means TWO opposite directions at once, so a mirror maps
    the pair onto itself with the roles swapped -- if anything in the engine
    resolved the two universes in a fixed order, a flip would show it.

    Stage 1 (here) replays each level's plan on all eight turned and mirrored
    copies of its own board, built by transforming the LEVEL LAYOUT so the
    interpreter re-loads it itself; every press must leave all three bodies and
    every crate exactly where the transform of the reference run put them, and
    the run must still win. Stage 2 (`_chirality_probe`) is what can actually
    reach the configurations a 3-to-15-press plan never visits -- see its
    docstring."""
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

#: The three collision layers a MOBILE object can occupy, what each can be
#: showing, and the static objects that keep it empty. Everything else on a cell
#: -- the floor, the arrows, the three wall kinds and the two goals -- never
#: moves, so it comes from the level layout rather than from a cross product.
#: Objects in one row are in one collision layer and so are mutually exclusive;
#: that is what makes the product below the complete set of compositions this
#: game can paint.
_MOBILE = (
    (("player", "crate2"), ("wall",)),
    (("player2", "crate3"), ("wall2", "wallr")),
    # Nothing static shares the shadows' layer. A `PlayerR` cannot in fact
    # reach a `Wall` cell (``[ > PlayerR | Wall ] -> cancel`` is what stops it,
    # and it is the Player's only wall check), and `Player2R` has a legend
    # entry no level uses -- so allowing both everywhere audits a SUPERSET of
    # what the game can show, which is the safe direction to be wrong in.
    (("playerr", "player2r"), ()),
)


def _audit() -> int:
    """Assert every cell COMPOSITION a board can show is distinct at the cell
    size in use.

    The composition, not the object, is what has to be readable: four layers
    stack on the Player's goal square at the moment the game is won (the body,
    the `Wall2` the `Target` is bolted to, the `Target` itself and the shadow),
    and the win condition is read off exactly that stack. This is the check the
    corner scheme in the .txt header exists to pass.

    The composition set is built from the STATIC stacks the shipped levels
    actually contain, crossed with everything the mobile objects can add on top,
    and it is built PER BOARD SIZE -- the two sizes render at 6px and 4px and
    they do not ship the same statics. That distinction is not cosmetic: the
    four Arrows are decoration that exists only on level 1 (10x10, 6px), and at
    4px the corner blocks cover their glyphs. Auditing an arrow at a size no
    board carrying one is ever rendered at would report a clash that cannot
    happen.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor).
    """
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    mobile = {"player", "playerr", "player2", "player2r", "crate2", "crate3"}

    # The statics each board size ships, read off the levels rather than guessed.
    statics: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        size = (eng.height, eng.width)
        for r in range(eng.height):
            for c in range(eng.width):
                names = {g.obj_idx_to_name[o] for o in eng.grid[r][c]}
                statics.setdefault(size, set()).add(
                    frozenset(names - mobile))

    bad = 0
    for size, stacks in sorted(statics.items()):
        h, w = size
        comps = {}
        for stack in stacks:
            slots = [("",) + opts if not (set(stack) & set(blockers)) else ("",)
                     for opts, blockers in _MOBILE]
            for choice in itertools.product(*slots):
                objs = set(stack) | {x for x in choice if x}
                comps["+".join(sorted(objs))] = objs
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx[o] for o in objs} for _ in range(w)]
                        for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h}x{w} (cell {min(64 // h, 64 // w)}px): {len(shots)} "
              f"compositions from {len(stacks)} static stacks: "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes[:4])}")
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
    sys.exit(ObservableUniverseSolver.main())
