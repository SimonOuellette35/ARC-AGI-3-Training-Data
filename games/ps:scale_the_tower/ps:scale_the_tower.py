"""PuzzleScript game: scale_the_tower

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Scale_the_Tower.txt

Play with: python solver_client.py ps:scale_the_tower
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Scale_the_Tower", seed=seed)
