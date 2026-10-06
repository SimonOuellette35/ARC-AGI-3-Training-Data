"""PuzzleScript game: sweet_hints

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Sweet_Hints.txt

Play with: python solver_client.py ps:sweet_hints
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Sweet_Hints", seed=seed)
