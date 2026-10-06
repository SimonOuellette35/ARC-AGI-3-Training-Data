"""PuzzleScript game: stand_ii

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Stand_II.txt

Play with: python solver_client.py ps:stand_ii
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Stand_II", seed=seed)
