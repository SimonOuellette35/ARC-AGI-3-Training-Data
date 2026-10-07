"""Generate Phase-1 training data for the PuzzleScript game ps:mimic_translation
("Mimic Translation", CNIAngel).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: an exact model of the mechanic, an exhaustive optimal
solver over it, and the reports that certify both.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_mimic_translation",
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
Two bodies, one keyboard. The blue-cored Player and the red-cored Doppler stand
on the same board and a direction press sends the Player that way and the
Doppler the OPPOSITE way. You win only when both are home at once: the Player on
the solid blue Target and the Doppler on the red MimicTarget. Five rules:

    [ > Player ] [ Doppler ]      -> [ > Player ] [ < Doppler ]
    [ Action Player ] [ Doppler ] -> [ Doppler ] [ Player ]
    [ > Player | Crate ]          -> [ > Player | > Crate ]
   +[ > Doppler | Crate ]         -> [ > Doppler | > Crate ]
   +[ > Crate | Crate ]           -> [ > Crate | > Crate ]

and two `[ > X | Goal ] -> [ > X | Goal ]` identities that do nothing at all.

Four facts are the whole puzzle, and three of them are consequences of the first
rule matching on the player's FORCE rather than on the player having MOVED:

  * **A wall is how you steer.** There is no `rigid` here and no cancel: a press
    the Player cannot take still hands the Doppler its opposite force, so
    walking into a wall moves the Doppler alone. Every level is solved by
    choosing which of the two is allowed to move.
  * **ACTION teleport-swaps the pair**, at any distance, through anything. Half
    the shipped levels end on it -- the mirror walks each body onto the OTHER's
    goal and one press exchanges them. It is also the only press that is never
    refused, so there is no wait move: ACTION always changes the board.
  * **Both bodies push crates**, each along its own force, and pushes chain
    through a run of crates. So one press can shove two separate stacks in
    opposite directions.
  * **They cannot pass through each other, and they contest cells.** Player and
    Doppler share a collision layer with Wall and Crate. Two consequences the
    model has to get exactly right, both measured on the interpreter rather than
    read off the rules (`--selfcheck` counts how often each fires):
      - HEAD-ON: when the cells between them are all crates (or they are
        adjacent), each one's push chain ends on the other, the interpreter
        defers both chains, nothing moves that iteration and it drops the whole
        turn. `P C D` pressed toward each other is a total no-op, and so is
        `P D`. That is also why the rule group's ambiguity is harmless: the
        contested crates get a force from both rules and whichever wins, the run
        is walled in by two bodies moving inward and cannot go anywhere.
      - CONTESTED CELL: when the two chains are free but their destination sets
        overlap, `_resolve_forces` blocks BOTH -- `. P C C . D .` pressed right
        moves nothing at all, because the far crate and the Doppler both want
        the gap. One clear cell more and everything moves at once.

There is no way to lose: no rule deletes a body, and a crate shoved into a
corner only costs the states behind it. The one irreversible move is a push, so
recovery is a RESET (`PSAStarSolver.recovery_mode`).

THE ONE .txt CHANGE
-------------------
The Crate's middle 3x3 is transparent, because a solid crate parked on a goal
hid the square the win condition names -- 120 / 118 / 119786 of the reachable
states of the three crate levels have exactly that. See the header comment in
`data/puzzlescript_games/Mimic_Translation.txt` and `--audit`, which asserts
every cell composition the board can show is distinct at the 7px cells all seven
levels render at. No mechanic is touched: transparency is a paint property.

THE LEVELS
----------
All seven are 7x9, all seven are winnable, all seven are recorded, and every
plan here is proved SHORTEST rather than merely found -- `_Model.solve` is a
breadth-first sweep of the level's own state space, so the length it returns is
the true distance. `--plans` prints this table and CERTIFIES every plan by
replaying it through the interpreter:

    level  crates   reachable   visited   d*   ties
      0       0            14        13    7   1.9
      1       0           830       417   11   1.1
      2       0           224       141   11   1.9
      3       0           722       294   10   1.3
      4       1          2942        50    5   1.6
      5       5           272        42    6   1.2
      6      15        246344       854    9   1.9

``reachable`` is the level's ENTIRE reachable state space and ``visited`` what
the depth-bounded search actually touches -- the gap is the point of stopping at
the first winning LAYER rather than sweeping the space: level 6 answers from 854
of its quarter-million states. ``ties`` is the mean size of the optimal-press set
along the plan, i.e. how much of the labelling would be a lie if the recorder
trained one arbitrary shortest path as the only right answer (54% of the 59
presses across the seven levels have one). The longest plan is 11 presses against
the adapter's 200-step per-level budget.

Level 6 is worth a look: it is exactly 180-degree symmetric about its centre --
the crate field, both bodies and both goals -- so the mirror is an involution of
the whole board and the Doppler reaching the MimicTarget is the same event as
the Player reaching the Target. It is also the one level where the interpreter
is unusable as a search engine: fifteen crates make the chain-push rule group
rescan the board until it settles, and `eng.step` drops from ~4600/s on the
other six to 30/s. That, not the state count, is why this file carries a native
model -- the same measurement that decided [[entrepotphage-demake-solver]] and
[[escaping-limbo-solver]].

EXPERT SOLVER
-------------
`_Model` is the mechanic over three ints (Player cell, Doppler cell, crate
bitmask), fuzz-verified against the real interpreter by `--selfcheck` on the
shipped levels AND on random boards built to hold the configurations no shipped
level reaches (bodies facing each other across a run of crates, two chains
contesting one cell). `_Model.solve` then does the exact thing:

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
edge. All seven searches together take under a second, which is why this expert
carries no on-disk plan cache -- the other end of the trade the ps: family makes
when a search is the whole cost of generation.

Usage
-----
    python solvers/generate_mimic_translation_training.py --episodes 200 \
        --out data/mimic_translation
    python solvers/generate_mimic_translation_training.py --plans
    python solvers/generate_mimic_translation_training.py --selfcheck
    python solvers/generate_mimic_translation_training.py --symmetry
    python solvers/generate_mimic_translation_training.py --audit
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

GAME_NAME = "Mimic_Translation"
GAME_MODULE_ID = "ps:mimic_translation"

#: The presses the search branches on, in the order it tries them. ACTION5 is
#: the swap and is never a no-op, so unlike most games here it is not in the
#: list merely to be measured.
KEYS = ("up", "down", "left", "right", "action")
_ACTION = KEYS.index("action")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
#: index of the opposite of each of the four directions (ACTION maps to itself)
_OPP = (1, 0, 3, 2, 4)


# ---------------------------------------------------------------------------
# Native model of the mechanic
# ---------------------------------------------------------------------------

class _Model:
    """Mimic Translation as a state machine over cell indices and one bitmask.

    A state is ``(player, doppler, crates)``: two cell indices ``r * w + c`` and
    an int whose set bits are the crates. Walls and the two goals are static and
    live on the model.

    ``nbr[di][cell]`` is the neighbour of ``cell`` in direction ``di``, or -1 off
    the board, precomputed so `step` never does row/column arithmetic (which is
    where an index model wraps around an edge without noticing).
    """

    __slots__ = ("h", "w", "walls", "target", "mimic", "nbr")

    def __init__(self, h: int, w: int, walls: int, target: int, mimic: int):
        self.h, self.w = h, w
        self.walls = walls
        self.target = target
        self.mimic = mimic
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

    def _chain(self, start: int, di: int, crates: int):
        """``(cells, end)``: the mover at ``start`` plus the unbroken run of
        crates ahead of it, and the first cell beyond that run (-1 off-board).

        This is the push chain `PSEngine._resolve_forces` traces: it stops at the
        first cell that is not a crate, and whether the chain may move is then a
        question about that one cell.
        """
        cells = [start]
        cur = start
        table = self.nbr[di]
        while True:
            nxt = table[cur]
            if nxt < 0:
                return cells, -1
            if (crates >> nxt) & 1:
                cells.append(nxt)
                cur = nxt
                continue
            return cells, nxt

    def step(self, state, di: int):
        """The settled board after one press. Never None: this game has no
        cancel, and a press that nothing can act on returns ``state`` itself."""
        p, d, crates = state
        if di == _ACTION:
            return (d, p, crates)

        pcells, pend = self._chain(p, di, crates)
        dcells, dend = self._chain(d, _OPP[di], crates)

        # HEAD-ON. Each chain ends on the other body, which is a mover heading
        # the other way, so the interpreter defers both, moves nothing, and
        # drops the turn. The two tests are equivalent -- a chain that ends on
        # the far body means every cell between them is a crate, so the far
        # body's chain covers the same run and ends here -- but both are asked
        # because the model must not depend on that argument being right.
        if pend == d or dend == p:
            return state

        p_ok = pend >= 0 and not (self.walls >> pend) & 1
        d_ok = dend >= 0 and not (self.walls >> dend) & 1

        if p_ok and d_ok:
            # CONTESTED CELL: two independent chains claiming the same
            # destination are BOTH blocked (`_resolve_forces` phase 3).
            ptab, dtab = self.nbr[di], self.nbr[_OPP[di]]
            pdst = {ptab[x] for x in pcells}
            if pdst & {dtab[x] for x in dcells}:
                return state

        np_, nd_, ncr = p, d, crates
        if p_ok:
            table = self.nbr[di]
            np_ = table[p]
            for x in pcells[1:]:
                ncr &= ~(1 << x)
            for x in pcells[1:]:
                ncr |= 1 << table[x]
        if d_ok:
            table = self.nbr[_OPP[di]]
            nd_ = table[d]
            for x in dcells[1:]:
                ncr &= ~(1 << x)
            for x in dcells[1:]:
                ncr |= 1 << table[x]
        return (np_, nd_, ncr)

    def won(self, state) -> bool:
        return state[0] == self.target and state[1] == self.mimic

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
        not a tuning dial: the widest shipped level stores 854 states, because
        the sweep stops at the layer that wins and its ``d*`` is 9.
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

    Everything static (walls, the two goals) goes on the model and everything
    that moves goes in the state, which is what lets the state be three ints.
    The goals are asserted unique: both win conditions are ``All <Goal> on
    <body>``, so a second Target would silently make the model's
    "player stands here" test wrong rather than merely incomplete.
    """
    idx = g.obj_name_to_idx
    h, w = eng.height, eng.width
    walls = crates = 0
    p = d = target = mimic = -1
    found = {"target": [], "mimictarget": []}
    for r in range(h):
        for c in range(w):
            cell = eng.grid[r][c]
            i = r * w + c
            if idx["wall"] in cell:
                walls |= 1 << i
            if idx["crate"] in cell:
                crates |= 1 << i
            if idx["player"] in cell:
                p = i
            if idx["doppler"] in cell:
                d = i
            if idx["target"] in cell:
                target = i
                found["target"].append(i)
            if idx["mimictarget"] in cell:
                mimic = i
                found["mimictarget"].append(i)
    for name, cells in found.items():
        if len(cells) != 1:
            raise ValueError(f"{len(cells)} {name}s on the board, expected 1")
    if p < 0 or d < 0:
        raise ValueError("board has no Player or no Doppler")
    return _Model(h, w, walls, target, mimic), (p, d, crates)


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class MimicExpert(PSExpert):
    """Exhaustive optimal expert: `_Model.solve` from the current board, with
    the plan and the per-step optimal sets read off the shortest-path DAG.

    It plugs into the shared harness at `PSExpert._search`, so the plan memo, its
    disk cache, the snapshot/restore discipline and the level scoping all come
    from the base. `heuristic` is never called -- `_astar` is not the strategy
    here -- and neither is `dead`: nothing in this game is lethal, and a sweep
    that exhausts the component has already proved what `dead` would guess.
    """

    directions = list(KEYS)

    # No `plan_cache_path`: all seven searches together are under a second, so a
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


class MimicTranslationSolver(PSAStarSolver):
    game_id = "puzzlescript_mimic_translation"
    game_name = GAME_NAME
    expert_cls = MimicExpert
    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so the generator can never tape frames from a different
    #: build of the game than the agent plays.
    game_module_id = GAME_MODULE_ID

    #: The longest plan is 11 presses; the rest is room for the exploration
    #: prefix and the replay after its RESET. Comfortably inside the adapter's
    #: own 200-step per-level budget.
    max_steps = 100


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    solver = MimicTranslationSolver()
    game = solver.make_game(0)
    return solver, game, MimicExpert(game)


def _plan_report() -> int:
    """Per-level crate count, reachable-space size, states the search visits,
    plan length and tie richness -- the table in the module docstring.

    Every plan is CERTIFIED by replaying it through the real interpreter, and
    every optimal label is checked the expensive way: each press the label calls
    equally-shortest is taken on the model and re-solved from there, and it has
    to leave exactly one press fewer than the step it was offered at."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    total = ties_total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        model, state = model_from_engine(eng, game._game)
        plan = expert.plan(eng, level)
        crates = bin(state[2]).count("1")
        space, capped = model.reachable(state, expert.state_cap)
        if plan is None:
            bad += capped
            why = ("unproven, the sweep hit the cap" if capped
                   else "unwinnable (component exhausted)")
            print(f"level {level}: {eng.height}x{eng.width} {crates:2d} crates, "
                  f"NO PLAN -- {why} ({len(space)} states)")
            continue

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
        if model.step(s, _ACTION) == s:
            bad += 1
            print(f"  L{level}: ACTION5 is a no-op, which this game never is")

        ties = sum(len(o) for o in plan.optsets) / len(plan)
        total += len(plan)
        ties_total += sum(1 for o in plan.optsets if len(o) > 1)
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"level {level}: {eng.height}x{eng.width} {crates:2d} crates, "
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


def selfcheck(trials: int = 30, steps: int = 60,
              boards: int = 400, board_steps: int = 25) -> int:
    """Audit the claim the expert rests on: `_Model` reproduces the interpreter
    EXACTLY, on the settled state and on the win flag, for every press.

    Two fuzz populations, and the second is the one with teeth. The seven
    shipped levels exercise the game as it is played; random boards -- random
    walls, random crates, the two bodies dropped anywhere -- exercise the
    configurations no shipped level reaches often enough to trust: a body
    walking into the other, the two facing each other across a run of crates,
    two push chains contesting one cell, a crate stack shoved from both ends.
    The coverage counters below are printed and ASSERTED non-zero, because a
    fuzz that never built those boards would report a clean model while having
    tested none of the three paragraphs of `_Model.step`.

    Level 6 is fuzzed too, at 30 interpreter steps per second against ~4600 for
    the rest -- that ratio is why the model exists, so the level that motivates
    it does not get to skip the audit.
    """
    game = _load_game_module(GAME_MODULE_ID).make_game(seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    bg = idx["background"]
    tot = {"presses": 0, "noops": 0, "wins": 0, "headon": 0, "contested": 0,
           "pushes": 0, "chains": 0, "both_push": 0, "desync": 0,
           "crate_on_goal": 0, "swaps": 0}
    bad = 0

    def rollout(tag: str, nsteps: int, rng: random.Random) -> int:
        nonlocal bad
        model, state = model_from_engine(eng, g)
        for _ in range(nsteps):
            di = rng.randrange(len(KEYS))
            p, d, crates = state

            # Coverage, classified BEFORE the press, from the same chain trace
            # `step` uses -- so what the counters report is what the model did.
            if di != _ACTION:
                pcells, pend = model._chain(p, di, crates)
                dcells, dend = model._chain(d, _OPP[di], crates)
                p_ok = pend >= 0 and not (model.walls >> pend) & 1 and pend != d
                d_ok = dend >= 0 and not (model.walls >> dend) & 1 and dend != p
                if pend == d or dend == p:
                    tot["headon"] += 1
                elif p_ok and d_ok and ({model.nbr[di][x] for x in pcells}
                                        & {model.nbr[_OPP[di]][x]
                                           for x in dcells}):
                    tot["contested"] += 1
                else:
                    moved = (len(pcells) - 1 if p_ok else 0) + \
                            (len(dcells) - 1 if d_ok else 0)
                    tot["pushes"] += moved > 0
                    tot["chains"] += (p_ok and len(pcells) > 2) or \
                                     (d_ok and len(dcells) > 2)
                    tot["both_push"] += (p_ok and len(pcells) > 1) and \
                                        (d_ok and len(dcells) > 1)
                    tot["desync"] += p_ok != d_ok
            else:
                tot["swaps"] += 1

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
            tot["crate_on_goal"] += bool(expected[2] >> model.target & 1
                                         or expected[2] >> model.mimic & 1)
            state = expected
            if eng.check_win():
                tot["wins"] += 1
                return 0
        return 0

    for level in range(game.n_levels):
        for t in range(trials):
            game.set_level(level)
            rollout(f"L{level}/{t}", steps,
                    random.Random(f"mimic:shipped:{level}:{t}"))

    for t in range(boards):
        rng = random.Random(f"mimic:board:{t}")
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
        for (r, c) in cells[:rng.randint(0, len(cells) // 5)]:
            eng.grid[r][c].add(idx["wall"])
        free = [x for x in cells if idx["wall"] not in eng.grid[x[0]][x[1]]]
        if len(free) < 4:
            continue
        rng.shuffle(free)
        # Half the boards get a dense crate field: that is what makes a run long
        # enough for the two bodies to contest one, which is the case the
        # shipped levels reach only in level 6.
        share = 2 if rng.random() < 0.5 else 5
        for (r, c) in free[2:2 + rng.randint(0, len(free) // share)]:
            eng.grid[r][c].add(idx["crate"])
        (pr, pc), (dr_, dc_) = free[0], free[1]
        eng.grid[pr][pc].add(idx["player"])
        eng.grid[dr_][dc_].add(idx["doppler"])
        goals = [x for x in cells if x not in (free[0], free[1])]
        eng.grid[goals[0][0]][goals[0][1]].add(idx["target"])
        eng.grid[goals[1][0]][goals[1][1]].add(idx["mimictarget"])
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        rollout(f"board{t}", board_steps, rng)

    print(f"selfcheck: {tot['presses']} presses "
          f"({game.n_levels} levels x {trials} rollouts + {boards} random "
          f"boards), {tot['noops']} no-ops, {tot['wins']} incidental wins")
    print(f"  coverage: {tot['headon']} head-on stand-offs, "
          f"{tot['contested']} contested cells, {tot['pushes']} pushes of which "
          f"{tot['chains']} were multi-crate chains and {tot['both_push']} "
          f"pushed at both ends, {tot['desync']} presses that moved one body "
          f"only, {tot['swaps']} ACTION swaps, {tot['crate_on_goal']} states "
          f"with a crate on a goal")
    weak = [k for k in ("headon", "contested", "pushes", "chains", "both_push",
                        "desync", "swaps", "crate_on_goal") if not tot[k]]
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
                 for o in (idx["player"], idx["doppler"], idx["crate"]))


def _chirality_probe(game, expert, rollouts: int = 10, steps: int = 25) -> int:
    """The second half of `_symmetry`, and the half with teeth.

    Replaying a level's PLAN under the eight transforms cannot find a mechanic
    the plan does not happen to use, and the mechanic at risk here is exactly
    that kind: when the two push chains contest a run of crates, the force each
    crate ends up carrying is decided by the order the interpreter walks a rule
    group -- a fact about the ENGINE, which no presentation transform can
    change. (The argument that it cannot matter is in the module docstring: a
    contested run is walled in by two bodies moving inward. This stage is what
    turns that argument into a measurement.)

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
            rng = random.Random(f"mimic:chiral:{level}:{t}")
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
    that was wrong in ps:gobble_rush and ps:lovendpieces, and this game has two
    places it could go the same way: the rule group that resolves a crate run
    pushed from both ends, and `_resolve_forces`' iteration over chains that
    contest a cell. Neither is stated in terms of the board's absolute
    orientation, but neither was theirs.

    Stage 1 (here) replays each level's plan on all eight turned and mirrored
    copies of its own board, built by transforming the LEVEL LAYOUT so the
    interpreter re-loads it itself; every press must leave both bodies and every
    crate exactly where the transform of the reference run put them, and the run
    must still win. Stage 2 (`_chirality_probe`) is what can actually reach the
    contested configurations -- see its docstring."""
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

    The composition, not the object, is what has to be readable: Target and
    MimicTarget sit on a layer BELOW Player, Doppler and Crate, so eleven of the
    thirteen cells this game can paint are stacks -- and the win condition is
    read off two of them ("is the blue goal showing under the blue-cored body").
    This is the check the Crate's transparent middle exists to pass; see the
    header comment in the .txt.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    lower = {"": (), "target": ("target",), "mimic": ("mimictarget",)}
    upper = {"": (), "wall": ("wall",), "player": ("player",),
             "doppler": ("doppler",), "crate": ("crate",)}
    # A goal UNDER a wall is the one stack left out, and it is left out because
    # no board can show it: walls are static, and a goal sealed under one is a
    # win condition no body could ever satisfy. That is asserted below rather
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
                if idx["wall"] in cell and (idx["target"] in cell
                                            or idx["mimictarget"] in cell):
                    bad += 1
                    print(f"level {level}: a goal is sealed under a wall at "
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
    sys.exit(MimicTranslationSolver.main())
