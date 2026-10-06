"""PuzzleScript game: escape_the_void_full

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Escape_the_Void_Full_.txt

Play with: python solver_client.py ps:escape_the_void_full
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Escape_the_Void_Full_", seed=seed)
