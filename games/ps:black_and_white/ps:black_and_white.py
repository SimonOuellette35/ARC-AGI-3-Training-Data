"""PuzzleScript game: black_and_white

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Black_&_White.txt

Play with: python solver_client.py ps:black_and_white
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Black_&_White", seed=seed)
