"""PuzzleScript game: slippy_penguin

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Slippy_Penguin.txt

Play with: python solver_client.py ps:slippy_penguin
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Slippy_Penguin", seed=seed)
