"""PuzzleScript game: sokolor_ex

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Sokolor_EX.txt

Play with: python solver_client.py ps:sokolor_ex
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Sokolor_EX", seed=seed)
