"""Generate Phase-1 training data for the Aquarium Sorter game.

A thin ``BaseSolver`` subclass (record/replay, schema, WIN-filter CLI and the
optional exploration prefix all live in ``solvers/base_solver.py``). An expert
solver (embedded below) sorts every level for a seed; each seed is a full 7-level
game, so each WIN seed yields one complete episode. Tanks are procedurally
derived from the seed, so different seeds give different episodes.

Aquarium Sorter action set (only indices 0..5 are ever emitted):
    ACTION1 (1) cursor -1     ACTION2 (2) cursor +1
    ACTION5 (5) toggle gate under cursor -- the ONLY action that advances the
                fish one step
    ACTION7 (7) undo -- NEVER emitted.
The solver only uses cursor moves (ACTION1/ACTION2) and gate toggles (ACTION5).

The A*/beam search assumes gates all-closed and the cursor at gate index 0 --
which holds at a level's INITIAL state (``set_level`` closes every gate and parks
the cursor at gate 0). It does not model an arbitrary perturbed gate/cursor
state, and fish motion is only loosely reversible, so an exploratory detour
cannot be re-planned from. Recovery is therefore RESET-mode
(``supports_recovery = True``, ``recovery_mode = "reset"``): the exploration
prefix runs, then a single RESET restores the level's initial state -- exactly
the gates-closed/cursor-0 configuration the search assumes -- and the optimal
plan replays from there.

Usage (run from the repo root):
    python solvers/generate_aquarium_sorter_training.py --episodes 1000 \
        --out data/training_multi_level/aquarium_sorter
"""

from __future__ import annotations

import heapq
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                             # noqa: E402
from games.aquarium_sorter.aquarium_sorter import (          # noqa: E402
    AquariumSorter,
    _STEP_BUDGETS,
)
from solvers.base_solver import BaseSolver                   # noqa: E402


# ---------------------------------------------------------------------------
# Pure simulation helpers (mirror the game's _advance_fish / _check_win logic)
# ---------------------------------------------------------------------------

def _sim_advance(fish, gate_states, tank_w, num_comp):
    """Advance all fish one step (mirrors the game's _advance_fish logic)."""
    out = [dict(f) for f in fish]
    for f in out:
        nx = f["x"] + f["dir"]
        ci = f["comp"]
        if nx >= tank_w - 1:
            gname = f"gate_{ci}_right"
            if ci < num_comp - 1 and gate_states.get(gname, False):
                f["comp"] = ci + 1
                f["x"] = tank_w - 2
                f["dir"] = -1
            else:
                f["dir"] = -1
        elif nx <= 0:
            if ci > 0:
                gname = f"gate_{ci - 1}_left"
                if gate_states.get(gname, False):
                    f["comp"] = ci - 1
                    f["x"] = 1
                    f["dir"] = 1
                else:
                    f["dir"] = 1
            else:
                f["dir"] = 1
        else:
            f["x"] = nx
    # Mirror the game's overlap resolution: nudge later-indexed different-species
    # fish to prevent permanent lockstep.
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            fi, fj = out[i], out[j]
            if (fi["species"] != fj["species"] and
                    fi["comp"] == fj["comp"] and
                    fi["x"] == fj["x"] and
                    fi["dir"] == fj["dir"]):
                nudge = fj["x"] + fj["dir"]
                if 1 <= nudge <= tank_w - 2:
                    fj["x"] = nudge
                else:
                    fj["dir"] = -fj["dir"]
    return out


def _is_won_sim(fish, targets):
    return all(f["species"] == targets[f["comp"]] for f in fish)


def _target_comp_map(targets):
    """species -> the compartment index that should ultimately hold it."""
    return {sp: ci for ci, sp in enumerate(targets)}


@lru_cache(maxsize=None)
def _fish_min_steps(tc, x, comp, d, tank_w, num_comp):
    """Minimum ACTION5 fish-steps for a fish ALONE to reach target compartment
    ``tc`` from state ``(x, comp, d)``, with gates controlled ideally for it. A
    hard physical lower bound; memoised over the tiny keyspace."""
    if comp == tc:
        return 0
    steps = 0
    guard = 4 * tank_w * num_comp + 10
    while comp != tc and steps < guard:
        steps += 1
        nx = x + d
        if nx >= tank_w - 1:
            if tc > comp:                      # cross DOWN toward target
                comp += 1
                x = tank_w - 2
                d = -1
            else:
                d = -1                         # bounce
        elif nx <= 0:
            if tc < comp and comp > 0:         # cross UP toward target
                comp -= 1
                x = 1
                d = 1
            else:
                d = 1                          # bounce
        else:
            x = nx
    return steps


def _h_astar(fish, tc_map, tank_w, num_comp):
    """Admissible heuristic for optimal A*: every ACTION5 advances ALL fish one
    step, so the plan needs >= as many ACTION5s as the slowest misplaced fish --
    i.e. the MAX per-fish lower bound."""
    return max((_fish_min_steps(tc_map[f["species"]], f["x"], f["comp"], f["dir"],
                                tank_w, num_comp)
                for f in fish if f["comp"] != tc_map[f["species"]]), default=0)


def _h_beam(fish, tc_map, tank_w, num_comp):
    """Greedy ranking for beam search: primarily fewer misplaced fish, then the
    smaller total remaining swim distance."""
    misplaced = 0
    total = 0
    for f in fish:
        total += _fish_min_steps(tc_map[f["species"]], f["x"], f["comp"], f["dir"],
                                 tank_w, num_comp)
        if f["comp"] != tc_map[f["species"]]:
            misplaced += 1
    return (misplaced, total)


# ---------------------------------------------------------------------------
# A* solver (small levels)
# ---------------------------------------------------------------------------

def _astar_solve(fish_init, gate_names, targets, num_comp, tank_w, budget,
                 node_cap=500000):
    """Optimal A* over (fish_state, gate_states, cursor). ACTION1/ACTION2 move
    the cursor (mod num_gates); ACTION5 toggles the gate under it and advances
    all fish one step. Goal checked on POP, so with the admissible ``_h_astar``
    the plan is minimum-length in TOTAL actions. Returns None on node_cap."""
    num_gates = len(gate_names)
    tc_map = _target_comp_map(targets)
    init_gates = {g: False for g in gate_names}

    def _key(fi, gs, c):
        fk = tuple(sorted((f["species"], f["comp"], f["x"], f["dir"]) for f in fi))
        gk = tuple(gs.get(gn, False) for gn in gate_names)
        return (fk, gk, c)

    h0 = _h_astar(fish_init, tc_map, tank_w, num_comp)
    counter = 0
    heap = [(h0, 0, counter, fish_init, init_gates, 0, [])]
    best_g = {_key(fish_init, init_gates, 0): 0}
    expanded = 0

    while heap:
        _f, g, _, cf, cg, cursor, actions = heapq.heappop(heap)
        if _is_won_sim(cf, targets):
            return actions                     # optimal: goal popped at min f=g
        sk = _key(cf, cg, cursor)
        if best_g.get(sk, float("inf")) < g:
            continue
        expanded += 1
        if expanded > node_cap:
            return None                        # too big -> caller uses beam
        if g >= budget:
            continue

        for delta, act in [(-1, GameAction.ACTION1), (1, GameAction.ACTION2)]:
            new_c = (cursor + delta) % num_gates
            nsk = _key(cf, cg, new_c)
            if g + 1 < best_g.get(nsk, float("inf")):
                best_g[nsk] = g + 1
                counter += 1
                h = _h_astar(cf, tc_map, tank_w, num_comp)
                heapq.heappush(heap, (g + 1 + h, g + 1, counter,
                                      cf, cg, new_c, actions + [act]))

        gname = gate_names[cursor]
        ng = dict(cg)
        ng[gname] = not ng[gname]
        nf = _sim_advance(cf, ng, tank_w, num_comp)
        nsk = _key(nf, ng, cursor)
        if g + 1 < best_g.get(nsk, float("inf")):
            best_g[nsk] = g + 1
            counter += 1
            h = (0 if _is_won_sim(nf, targets)
                 else _h_astar(nf, tc_map, tank_w, num_comp))
            heapq.heappush(heap, (g + 1 + h, g + 1, counter,
                                  nf, ng, cursor, actions + [GameAction.ACTION5]))

    return None


# ---------------------------------------------------------------------------
# Beam search solver (larger levels where A* exhausts memory)
# ---------------------------------------------------------------------------

def _beam_solve(fish_init, gate_names, targets, num_comp, tank_w, budget,
                beam_width=600):
    """Beam search over (fish_state, gate_states, cursor). Fallback for levels A*
    can't fit in its node cap. Ranked by the distance-aware ``_h_beam``. Returns
    action list or None."""
    num_gates = len(gate_names)
    tc_map = _target_comp_map(targets)
    init_gates = {g: False for g in gate_names}

    def _key(fi, gs, c):
        fk = tuple(sorted((f["species"], f["comp"], f["x"], f["dir"]) for f in fi))
        gk = tuple(gs.get(gn, False) for gn in gate_names)
        return (fk, gk, c)

    beam = [(_h_beam(fish_init, tc_map, tank_w, num_comp),
             fish_init, init_gates, 0, [])]
    seen = {_key(fish_init, init_gates, 0)}

    for _step in range(budget):
        candidates = []
        for _h, cf, cg, cursor, actions in beam:
            for delta, act in [(-1, GameAction.ACTION1), (1, GameAction.ACTION2)]:
                new_c = (cursor + delta) % num_gates
                sk = _key(cf, cg, new_c)
                if sk not in seen:
                    seen.add(sk)
                    candidates.append(
                        (_h_beam(cf, tc_map, tank_w, num_comp),
                         cf, cg, new_c, actions + [act])
                    )
            gname = gate_names[cursor]
            ng = dict(cg)
            ng[gname] = not ng[gname]
            nf = _sim_advance(cf, ng, tank_w, num_comp)
            if _is_won_sim(nf, targets):
                return actions + [GameAction.ACTION5]
            sk = _key(nf, ng, cursor)
            if sk not in seen:
                seen.add(sk)
                candidates.append(
                    (_h_beam(nf, tc_map, tank_w, num_comp),
                     nf, ng, cursor, actions + [GameAction.ACTION5])
                )
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        beam = candidates[:beam_width]

    return None


class AquariumSorterSolver(BaseSolver):
    game_id = "aquarium_sorter"
    # The search assumes gates-closed / cursor-at-0 (the initial state) and fish
    # motion is only loosely reversible, so an exploratory detour is NOT safely
    # re-planned from -- recovery is RESET-mode: explore the prefix, then RESET to
    # the level's initial state (which the search's gates-closed/cursor-0
    # assumption exactly matches) and replay the optimal plan from there.
    supports_recovery = True
    recovery_mode = "reset"

    def make_game(self, seed: int):
        return AquariumSorter(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 5]              # cursor -/+ and gate toggle

    def solve_from(self, game, level_idx: int, seed: int):
        """Plan a sort from the live level. Reads the static per-level config
        (targets / geometry / budget) and the LIVE fish positions
        (``game._fish_states``); the search itself assumes gates-closed and
        cursor 0, which is the level's initial state. Small tanks use optimal A*,
        larger tanks fall back to the distance-aware beam. Returns [] on
        failure (fails the episode)."""
        data = game._level_data[level_idx]
        targets = data["targets"]
        num_comp = data["num_comp"]
        tank_w = data["tank_w"]
        budget = _STEP_BUDGETS[level_idx]

        gate_names = [
            f"gate_{ci}_{side}"
            for ci in range(num_comp - 1)
            for side in ("left", "right")
        ]
        fish_init = [dict(f) for f in game._fish_states]

        plan = None
        if level_idx <= 2:
            plan = _astar_solve(fish_init, gate_names, targets, num_comp, tank_w,
                                budget)
        if plan is None:
            plan = _beam_solve(fish_init, gate_names, targets, num_comp, tank_w,
                               budget)
        return plan or []


if __name__ == "__main__":
    sys.exit(AquariumSorterSolver.main())
