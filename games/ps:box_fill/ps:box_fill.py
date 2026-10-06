"""PuzzleScript game: box_fill

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Box_Fill.txt

Play with: python solver_client.py ps:box_fill
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Box_Fill", seed=seed)
