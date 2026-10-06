"""PuzzleScript game: towers_of_hanoi

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Towers_of_Hanoi.txt

Play with: python solver_client.py ps:towers_of_hanoi
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Towers_of_Hanoi", seed=seed)
