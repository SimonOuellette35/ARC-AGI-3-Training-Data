"""PuzzleScript game: tricky_tower

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Tricky_Tower.txt

Play with: python solver_client.py ps:tricky_tower
"""

from arcengine import ActionInput, GameAction, GameState, FrameDataRaw
from adapters.puzzlescript_adapter import (
    PuzzleScriptAdapter, _render_frame, _ACTION_MAP,
)
from utils.rotation import remap_action as _remap_action


class TrickyTowerAdapter(PuzzleScriptAdapter):
    """Tricky Tower with rotation disabled and animation frames.

    The game uses absolute directional forces (left/up/right) to activate
    specific columns, so rotation augmentation breaks the controls.
    Captures intermediate grid states during rule cascades so the solver
    can see how block activations propagate up the tower.
    """

    def _do_reset(self):
        super()._do_reset()
        self._rotation_k = 0
        self._engine._capture_snapshots = True
        self._current_frame = _render_frame(self._engine, self._game)

    def perform_action(self, action_input: ActionInput, raw: bool = False) -> FrameDataRaw:
        if action_input.id == GameAction.RESET:
            self.level_reset()
            return self._make_frame_data(action_input)

        if self._state in (GameState.WIN, GameState.GAME_OVER):
            return self._make_frame_data(action_input)

        effective_id = _remap_action(action_input.id, self._rotation_k)
        direction = _ACTION_MAP.get(effective_id, "up")

        self._engine.step(direction)
        self._action_count += 1

        if self._engine._rule_restart:
            level_idx = self._current_level_index % max(1, len(self._game.levels))
            if self._game.levels:
                self._engine.load_level(self._game.levels[level_idx])
            self._current_frame = self._present_frame(
                _render_frame(self._engine, self._game)
            )
            return self._make_frame_data(action_input)

        frames = []
        for grid_snap in self._engine._snapshots:
            saved_grid = self._engine.grid
            self._engine.grid = grid_snap
            frames.append(
                self._present_frame(_render_frame(self._engine, self._game))
            )
            self._engine.grid = saved_grid

        if not frames:
            frames.append(
                self._present_frame(_render_frame(self._engine, self._game))
            )

        self._current_frame = frames[-1]

        if self._engine.check_win():
            self._state = GameState.WIN
        elif self._engine.check_game_over():
            self._state = GameState.GAME_OVER
        elif self._action_count >= self._max_steps:
            self._state = GameState.GAME_OVER
        else:
            self._state = GameState.NOT_FINISHED

        fd = self._make_frame_data(action_input)
        fd.frame = frames
        return fd


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return TrickyTowerAdapter("Tricky_Tower", seed=seed)
