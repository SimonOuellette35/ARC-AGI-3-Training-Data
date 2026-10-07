"""Generate Phase-1 training data for ``sokoban`` (procedurally generated Sokoban).

Seven levels per seed, 11x9/2 crates up to 19x17/5 crates, every board drawn from
the seed -- so each WIN seed is a genuinely different multi-level game.

THE EXPERT
----------
`utils.sokoban_model.SokobanModel` -- push-level A* over ``(player, boxes)`` with
edge weights in real MOVES (walk + 1 per push), a min-cost box->target assignment
heuristic over true push distances, and dead-square + freeze deadlock pruning. See
that module for why the search is over pushes rather than moves and where it
trades exactness for time on the big boards.

The model is rebuilt from the LIVE level's sprites on every call (``solve_from``'s
contract), so it plans from wherever an exploratory detour left the board instead
of replaying a plan cached at the level start.

RECOVERY (``supports_recovery = True``, ``recovery_mode = "replan"``)
--------------------------------------------------------------------
Walking is free to undo and most pushes are re-pushable, so the ordinary
perturbation is simply re-planned around -- that is the whole recovery story for
the exploration prefix and for most bursts.

When a detour actually wedges a crate (a corner, a wall lane with no target on
it, a frozen pair) the level is lost from that state and `solve_from` returns an
empty plan. The base then recovers the only way this generator allows: a burst is
rolled back through the recorder's own snapshot (invisible -- it never happened),
and a wedge from the exploration prefix costs a RESET.

The engine DOES have an undo of its own (``Sokoban.step``'s ACTION7 branch,
backed by ``game._history``), and this generator used to rewind with it: walk
back to the most recent winnable ancestor, emit that many ACTION7s, carry on. It
does not any more, and must not. **ACTION7 is outside the policy's action
vocabulary** -- `train_policy.NUM_ACTION_TYPES` is 7, i.e. indices 0..6 (RESET,
ACTION1-5, and ACTION6 == click) -- so an ACTION7 in the trajectory is an action
the model cannot represent, let alone predict: it indexes the action embedding
out of bounds and kills training with a CUDA device-side assert. That also
matches the recorder's own contract in `BaseSolver` -- undo is not a real
competition action and must leave no trace in the trajectory.

Losing the rewind costs a little recovery data (a rewind teaches "notice the
deadlock, back out exactly far enough, carry on"; a RESET only teaches "start
over") and a few more RESETs per episode, which `max_resets` absorbs.

`optimal_set_from` returns the co-optimal action set -- every first step of a
shortest walk to the next push -- so the base samples among them and one board
yields many distinct, equally good routes across episodes.

Usage (repo root, ARC-AGI-3 conda python):
    python solvers/generate_sokoban_training.py --episodes 1000
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.sokoban.sokoban import Sokoban          # noqa: E402
from solvers.base_solver import BaseSolver         # noqa: E402
from utils.sokoban_model import SokobanModel       # noqa: E402


class SokobanSolver(BaseSolver):
    game_id = "sokoban"

    supports_recovery = True
    recovery_mode = "replan"
    #: Wedging a crate is easy for a random walk and the prefix + bursts get
    #: several chances per episode. Bursts roll back invisibly, but a wedge from
    #: the exploration prefix now costs a RESET (the ACTION7 rewind is gone --
    #: see the module docstring), so allow a few more before abandoning the seed.
    max_resets = 8

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._model_key = None
        self._model: SokobanModel | None = None

    # ── engine glue ──────────────────────────────────────────────────────────
    def make_game(self, seed: int):
        return Sokoban(seed=seed)

    def available_actions(self, game) -> list[int]:
        # 1-4 move. The engine also has an undo on ACTION7, but it is NOT offered
        # here: index 7 is outside the policy's action vocabulary, so a random
        # press of it would poison the corpus (module docstring).
        return [1, 2, 3, 4]

    # ── live board -> (model, player, boxes) ─────────────────────────────────
    def _live(self, game):
        """Read the LIVE level into ``(model, player, boxes)``.

        The model (geometry + its plan caches) is memoised on the level's STATIC
        geometry -- walls and targets -- so one instance serves the whole level
        and survives a RESET, whose re-cloned level has the same walls and
        targets. Crates and the player come from the live sprites every call."""
        level = game.current_level
        width, height = level.grid_size
        walls, targets, boxes, player = set(), set(), [], None
        for sprite in level.get_sprites():
            cell = (sprite.x, sprite.y)
            if sprite.name == "wall":
                walls.add(cell)
            elif "target" in sprite.tags:
                targets.add(cell)
            elif "box" in sprite.tags:
                boxes.append(cell)
            elif "player" in sprite.tags:
                player = cell
        key = (width, height, frozenset(walls), frozenset(targets))
        if key != self._model_key:
            self._model = SokobanModel(width, height, walls, targets)
            self._model_key = key
        return self._model, player, tuple(sorted(boxes))

    # ── burst snapshot: the moving parts only ────────────────────────────────
    def _game_snapshot(self, game):
        """Override the base's whole-game ``deepcopy`` with the handful of
        mutable fields this game actually has.

        A Sokoban instance carries all seven levels (~300 wall/target sprites
        each), and a burst snapshot happens ~10x per episode plus once more for
        every `BaseSolver._burst_recovery_wins` probe -- deep-copying the lot cost
        ~20% of total generation time. Everything that can change inside a level
        is: the player cell, the crate cells, the engine's undo stack, the step
        counter, and the engine's own status flags."""
        level = game.current_level
        player = level.get_sprites_by_tag("player")[0]
        return ("sokoban",
                (player.x, player.y),
                [(b.x, b.y) for b in level.get_sprites_by_tag("box")],
                list(game._history),
                game._step_counter.steps_remaining,
                game._state,
                game._next_level,
                game._current_level_index)

    def _game_restore(self, game, snap) -> None:
        (_kind, player_xy, boxes, history, steps, state, next_level,
         level_index) = snap
        if level_index != game._current_level_index:   # pragma: no cover
            raise RuntimeError("sokoban snapshot restored across a level change")
        level = game.current_level
        level.get_sprites_by_tag("player")[0].set_position(*player_xy)
        for sprite, cell in zip(level.get_sprites_by_tag("box"), boxes):
            sprite.set_position(*cell)
        game._refresh_box_colors()
        game._history[:] = history
        game._step_counter.steps_remaining = steps
        game._state = state
        game._next_level = next_level

    # ── the BaseSolver hooks ─────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        model, player, boxes = self._live(game)
        # Empty == wedged, and with no legal way to rewind that IS the whole
        # answer: the base rolls a burst back through its snapshot, or RESETs.
        return model.moves(player, boxes)

    def optimal_set_from(self, game, level_idx: int, seed: int):
        model, player, boxes = self._live(game)
        return model.optimal_moves(player, boxes)


if __name__ == "__main__":
    sys.exit(SokobanSolver.main())
