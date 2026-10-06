"""PuzzleScript game: roller_boi

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Roller_Boi.txt

Play with: python solver_client.py ps:roller_boi
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Roller_Boi", seed=seed)
