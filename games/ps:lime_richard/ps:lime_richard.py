"""PuzzleScript game: lime_richard

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Lime_Richard.txt

Play with: python solver_client.py ps:lime_richard
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Lime_Richard", seed=seed)
