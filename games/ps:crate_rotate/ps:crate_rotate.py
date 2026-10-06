"""PuzzleScript game: crate_rotate

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Crate_Rotate.txt

Play with: python solver_client.py ps:crate_rotate
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Crate_Rotate", seed=seed)
