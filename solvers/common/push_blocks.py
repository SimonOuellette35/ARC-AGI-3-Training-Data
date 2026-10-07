"""Shared expert + training-data generator for the pb01/pb02/pb03 push-block family.

The three games are one Sokoban-style mechanic with one knob each (one crate / two
crates / a lethal decoy pad), all implemented by ``utils/push_puzzle.py``, so they
share one solver. Each ``generate_pb0N_training.py`` is a thin wrapper naming the
game class.

THE EXPERT
----------
`utils.push_puzzle.PushModel` -- A* over ``(player, crates)`` states with a
crate->pad Manhattan-matching heuristic and static deadlock pruning. The solver
rebuilds the model from the LIVE level's sprites on every call (`solve_from`'s
contract), so it plans from wherever an exploratory detour left the board rather
than replaying a plan cached at the level start. That is what earns
``supports_recovery`` with ``recovery_mode = "replan"``:

  * every non-push move is reversible, and a push is reversible whenever the
    crate's far side is walkable -- so most perturbations are simply re-planned
    around, and the burst data is recovery data;
  * the cases that are NOT reversible (a crate shoved into a corner, or pb03's
    instant decoy loss) make `solve_from` return ``[]`` / the state dead, which the
    base turns into a recorded RESET or a burst rollback. Both are honest: a human
    who wedges a crate resets the level too.

`optimal_set_from` returns the full co-optimal action set (every action that leaves
the win exactly one move closer), so the base samples the taken action among them
and each seed's board yields many distinct, still-optimal routes -- Sokoban has a
lot of co-optimal walking. The model's plan cache makes that affordable: the state
actually stepped into already has its plan cached as a suffix of the current one.

ROTATION
--------
Boards are presented at the per-(seed, level) rotation `AugmentedGame` draws, and
the game maps a directional press back through it. The model plans in GAME space,
so every action is converted to the SCREEN press a player would have to make
(`inverse_remap_action_full`) before it is driven or recorded -- never pinned to
k=0, which would put 3/4 of live orientations outside the corpus.

Usage (repo root, ARC-AGI-3 conda python):
    python solvers/generate_pb01_training.py --episodes 500
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from solvers.base_solver import BaseSolver, _ID_TO_GAMEACTION  # noqa: E402
from utils.push_puzzle import PushModel                        # noqa: E402
from utils.rotation import inverse_remap_action_full           # noqa: E402


class PushBlocksSolver(BaseSolver):
    """BaseSolver over `utils.push_puzzle.PushPuzzleGame`. Subclasses set
    ``game_id`` + ``game_cls``."""

    game_cls = None                       # set by the per-game wrapper

    supports_recovery = True
    recovery_mode = "replan"
    #: A wedged crate can only be undone by RESET, and the exploration prefix is
    #: long enough to wedge one now and then, so allow a few more than the default.
    max_resets = 5

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._model_key = None
        self._model: PushModel | None = None

    # ── engine glue ──────────────────────────────────────────────────────────
    def make_game(self, seed: int):
        return self.game_cls(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    # ── live board -> (model, state) ─────────────────────────────────────────
    def _live(self, game):
        """Read the LIVE level into ``(model, state)``.

        The model (geometry + its plan cache) is memoised on the level's static
        geometry, so one A* serves a whole level -- and it survives a RESET, whose
        re-cloned level has the same walls and pads."""
        level = game.current_level
        width, height = level.grid_size
        walls, targets, decoys, crates, player = set(), set(), set(), [], None
        for sprite in level.get_sprites():
            cell = (sprite.x, sprite.y)
            tags = sprite.tags
            if "wall" in tags:
                walls.add(cell)
            elif "target" in tags:
                targets.add(cell)
            elif "decoy" in tags:
                decoys.add(cell)
            elif "block" in tags:
                crates.append(cell)
            elif "player" in tags:
                player = cell
        key = (game.level_index, width, height, frozenset(walls),
               frozenset(targets), frozenset(decoys))
        if key != self._model_key:
            self._model = PushModel(width, height, walls, targets, decoys)
            self._model_key = key
        return self._model, (player, tuple(sorted(crates)))

    def _screen(self, action_id: int, game):
        """The game-space direction, unconverted.

        Solvers plan and emit in the core game's UPRIGHT space; `BaseSolver` does the
        one conversion to screen space when it records. Converting here as well rotated
        twice and pushed the block the wrong way at k != 0."""
        return _ID_TO_GAMEACTION[int(action_id)]

    # ── the BaseSolver hooks ─────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        model, state = self._live(game)
        return [self._screen(a, game) for a in model.plan(state)]

    def optimal_set_from(self, game, level_idx: int, seed: int):
        model, state = self._live(game)
        return [self._screen(a, game) for a in model.optimal_actions(state)]
