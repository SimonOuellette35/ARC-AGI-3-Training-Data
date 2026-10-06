"""PuzzleScript game: filler

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Filler.txt

Play with: python solver_client.py ps:filler
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Filler", seed=seed)
