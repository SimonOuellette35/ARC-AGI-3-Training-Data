"""PuzzleScript game: stained_glass

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Stained_Glass.txt

Play with: python solver_client.py ps:stained_glass
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Stained_Glass", seed=seed)
