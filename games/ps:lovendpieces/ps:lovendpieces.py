"""PuzzleScript game: lovendpieces

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/lovendpieces.txt

Play with: python solver_client.py ps:lovendpieces
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("lovendpieces", seed=seed)
