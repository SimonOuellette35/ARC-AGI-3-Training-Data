"""PuzzleScript game: the_trouble_with_toasters

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/The_Trouble_with_Toasters.txt

Play with: python solver_client.py ps:the_trouble_with_toasters
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("The_Trouble_with_Toasters", seed=seed)
