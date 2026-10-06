"""PuzzleScript game: stand_iii

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Stand_III.txt

Play with: python solver_client.py ps:stand_iii
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Stand_III", seed=seed)
