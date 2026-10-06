"""PuzzleScript game: weird_dave

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Weird_Dave.txt

Play with: python solver_client.py ps:weird_dave
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Weird_Dave", seed=seed)
