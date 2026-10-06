"""PuzzleScript game: the_nuevo_asylum

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/The_Nuevo_Asylum.txt

Play with: python solver_client.py ps:the_nuevo_asylum
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("The_Nuevo_Asylum", seed=seed)
