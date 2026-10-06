"""PuzzleScript game: two_faced

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Two-faced.txt

Play with: python solver_client.py ps:two_faced
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Two-faced", seed=seed)
