"""PuzzleScript game: pitman_mz_700

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Pitman_MZ-700.txt

Play with: python solver_client.py ps:pitman_mz_700
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Pitman_MZ-700", seed=seed)
