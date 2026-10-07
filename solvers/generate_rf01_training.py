"""Generate Phase-1 training data for the rf01 ("Mirror Half") game.

A thin `BaseSolver` subclass over the local rf01 game (games/rf01/rf01.py). rf01
is a 4-key movement puzzle on an 8..10 wide grid: walk the player onto the
target, walls block, nothing can kill you and there is no step budget. The twist
is the mechanic the corpus is meant to teach -- while the player stands at
``x >= gw // 2`` (the RIGHT half of the board) the horizontal component of the
move is NEGATED, so "left" walks right and "right" walks left there:

    ACTION1 up (dy-1)     ACTION2 down (dy+1)
    ACTION3 left (dx-1)   ACTION4 right (dx+1)      ... dx *= -1 on the right half

PLANNING SPACE. Two remaps sit between the key an agent presses and the cell the
player ends on: the per-(seed, level) display rotation (`AugmentedGame`, which
maps the pressed SCREEN direction back to game space) and then the mirror. Both
are folded into `_succ`, so the search runs directly over SCREEN action ids --
the plan this generator emits is exactly what a live agent would have to press,
at the live rotation, with nothing pinned to k=0.

OPTIMALITY. The state is just the player cell (walls are static, nothing else
moves), so one reverse BFS from the target over the mirrored move graph gives the
exact distance-to-target field for the whole board. `solve_from` descends it (a
shortest plan from the LIVE cell) and `optimal_set_from` returns every key whose
successor drops the distance by one -- the full optimal tie set, computed in O(1)
game copies, which is what BaseSolver samples the recorded action from and stores
as the training target.

supports_recovery is True: every move is reversible (the mirror swaps the two
horizontal keys, it never removes one, so both directions stay available on both
halves), there is no death and no step limit, and `solve_from` re-plans from the
live cell. So the exploration prefix and the perturbation bursts can wander
anywhere and the expert simply walks back onto a shortest path -- recovery data,
never a discarded episode.

Usage (run from the repo root, with the conda env python):
    python solvers/generate_rf01_training.py --episodes 1000 \
        --out data/training_multi_level/rf01
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

# Repo root (parent of solvers/) so games/ + arcengine resolve.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction  # noqa: E402

from games.rf01.rf01 import _DELTAS, Rf01, step_cell  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402
from utils.rotation import remap_action  # noqa: E402

# The four movement keys, as the agent presses them (SCREEN space).
_SCREEN_ACTIONS = (1, 2, 3, 4)


def _read_state(game):
    """The whole level state, read off the LIVE engine (never rebuilt from seed)."""
    level = game.current_level
    player = level.get_sprites_by_tag("player")[0]
    target = level.get_sprites_by_tag("target")[0]
    walls = frozenset((s.x, s.y) for s in level.get_sprites_by_tag("wall"))
    gw, gh = level.grid_size
    return (player.x, player.y), (target.x, target.y), walls, gw, gh


def _succ(cell, screen_id: int, k: int, walls, gw: int, gh: int):
    """Where pressing SCREEN key ``screen_id`` takes the player from ``cell``.

    Composes the two remaps the engine applies, in the engine's own order: the
    display rotation (screen key -> game direction, `AugmentedGame.step`) and then
    the mirror + wall/bounds check (`games.rf01.rf01.step_cell`). A blocked move
    returns ``cell`` unchanged, exactly as the game no-ops it.
    """
    game_action = remap_action(GameAction.from_id(screen_id), k)
    dx, dy = _DELTAS.get(game_action, (0, 0))
    return step_cell(cell[0], cell[1], dx, dy, walls, gw, gh)


def _dist_field(target, k: int, walls, gw: int, gh: int) -> dict:
    """Distance (in key presses) from every cell to ``target``.

    A reverse BFS: build the forward move graph over the free cells, invert it,
    and flood out from the target. The graph is <=100 nodes with 4 edges each, so
    the whole field costs about as much as one forward BFS -- cheap enough to
    recompute at every step of every level.
    """
    cells = [(x, y) for x in range(gw) for y in range(gh) if (x, y) not in walls]
    rev: dict = {c: [] for c in cells}
    for c in cells:
        for a in _SCREEN_ACTIONS:
            nxt = _succ(c, a, k, walls, gw, gh)
            if nxt != c:
                rev[nxt].append(c)
    dist = {target: 0}
    q = deque([target])
    while q:
        cur = q.popleft()
        for prev in rev.get(cur, ()):        # prev --key--> cur
            if prev not in dist:
                dist[prev] = dist[cur] + 1
                q.append(prev)
    return dist


class Rf01Solver(BaseSolver):
    #: Plans in SCREEN space by design: the search folds rotation and mirror into
    #: one transition and enumerates SCREEN keys, so its plan is already what a
    #: player would press. The base must not convert again when recording, and
    #: ``drive`` de-rotates before ``_set_action`` (which bypasses the wrapper).
    plans_in_screen_space = True
    game_id = "rf01"
    # Movement is fully reversible, nothing kills the player and there is no step
    # budget, and solve_from re-plans from the LIVE cell -> the exploration prefix
    # and the perturbation bursts are always recoverable by simply walking back.
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        # Rf01(seed) makes the grid size, the layout, the colours AND the display
        # rotation a pure function of (seed, level_index); the demos are therefore
        # recorded at the live per-(seed, level) rotation, never pinned to k=0.
        return Rf01(seed=seed)

    def available_actions(self, game) -> list[int]:
        return list(_SCREEN_ACTIONS)

    # ── the expert ──────────────────────────────────────────────────────────
    def _field(self, game):
        """``(start, dist)`` for the live state -- the state the plan starts from
        and the distance-to-target field it descends."""
        start, target, walls, gw, gh = _read_state(game)
        k = game.rotation_k
        return (start, target, walls, gw, gh, k,
                _dist_field(target, k, walls, gw, gh))

    def solve_from(self, game, level_idx: int, seed: int):
        """A shortest plan of SCREEN key presses from the player's LIVE cell.

        Descends the distance field: at each step take any key whose successor is
        one closer to the target. Empty means "no progress possible from here"
        (an unreachable target -- which the game's layout certificate rules out),
        which fails the episode.
        """
        start, target, walls, gw, gh, k, dist = self._field(game)
        d = dist.get(start)
        if not d:                                  # already on the target / stranded
            return []
        cur, plan = start, []
        while cur != target:
            for a in _SCREEN_ACTIONS:
                nxt = _succ(cur, a, k, walls, gw, gh)
                if dist.get(nxt, 1 << 30) == dist[cur] - 1:
                    plan.append(a)
                    cur = nxt
                    break
            else:                                  # cannot happen on a live field
                return []
        return plan

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal key at the live cell: those whose successor sits
        one step closer to the target. This is the training target BaseSolver
        records (and samples the taken action from), so a board with two shortest
        routes teaches both instead of one arbitrary tie-break."""
        start, _target, walls, gw, gh, k, dist = self._field(game)
        d = dist.get(start)
        if not d:
            return []
        return [a for a in _SCREEN_ACTIONS
                if dist.get(_succ(start, a, k, walls, gw, gh), 1 << 30) == d - 1]


if __name__ == "__main__":
    sys.exit(Rf01Solver.main())
