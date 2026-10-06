"""PuzzleScript game: color_combination

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Color_Combination.txt

Play with: python solver_client.py ps:color_combination
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Color_Combination", seed=seed)
