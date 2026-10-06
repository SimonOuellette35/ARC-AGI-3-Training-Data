"""PuzzleScript game: sokoslam

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Sokoslam.txt

Play with: python solver_client.py ps:sokoslam
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Sokoslam", seed=seed)
