"""PuzzleScript game: towers_of_saigon

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Towers_of_Saigon.txt

Play with: python solver_client.py ps:towers_of_saigon
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Towers_of_Saigon", seed=seed)
