"""PuzzleScript game: rush_hour

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Rush_Hour.txt

Play with: python solver_client.py ps:rush_hour
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Rush_Hour", seed=seed)
