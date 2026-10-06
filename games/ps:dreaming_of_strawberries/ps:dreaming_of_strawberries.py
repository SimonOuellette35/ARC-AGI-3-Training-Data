"""PuzzleScript game: dreaming_of_strawberries

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Dreaming_of_Strawberries.txt

Play with: python solver_client.py ps:dreaming_of_strawberries
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Dreaming_of_Strawberries", seed=seed)
