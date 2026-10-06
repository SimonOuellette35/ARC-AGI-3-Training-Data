"""PuzzleScript game: two_level_puzzle

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Two_level_puzzle.txt

Play with: python solver_client.py ps:two_level_puzzle
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Two_level_puzzle", seed=seed)
