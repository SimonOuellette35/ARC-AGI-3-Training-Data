"""PuzzleScript game: back_home

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Back_home.txt

Play with: python solver_client.py ps:back_home
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Back_home", seed=seed)
