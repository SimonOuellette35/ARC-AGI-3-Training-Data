"""PuzzleScript game: everything_antimatters

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Everything_Antimatters.txt

Play with: python solver_client.py ps:everything_antimatters
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Everything_Antimatters", seed=seed)
