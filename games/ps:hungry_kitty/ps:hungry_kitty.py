"""PuzzleScript game: hungry_kitty

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Hungry_Kitty.txt

Play with: python solver_client.py ps:hungry_kitty
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Hungry_Kitty", seed=seed)
