"""PuzzleScript game: bridge

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Bridge.txt

Play with: python solver_client.py ps:bridge
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


class BridgeAdapter(PuzzleScriptAdapter):
    """Bridge with per-level step limits.

    The later levels ("journey", "ocean") are large sokoban-style maps that
    need substantially more moves than the 200-step default. Limits scale
    with level index so each puzzle is actually solvable.
    """

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = 300 + self._current_level_index * 300


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return BridgeAdapter("Bridge", seed=seed)
