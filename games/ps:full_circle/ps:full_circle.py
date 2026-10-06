"""PuzzleScript game: full_circle

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/FULL_CIRCLE.txt

Play with: python solver_client.py ps:full_circle
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("FULL_CIRCLE", seed=seed)
