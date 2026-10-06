"""PuzzleScript game: sorx_aubi

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Sorx-Aubi.txt

Play with: python solver_client.py ps:sorx_aubi
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Sorx-Aubi", seed=seed)
