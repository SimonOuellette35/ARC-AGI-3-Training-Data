"""PuzzleScript game: the_blob

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/The_Blob.txt

Play with: python solver_client.py ps:the_blob
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("The_Blob", seed=seed)
