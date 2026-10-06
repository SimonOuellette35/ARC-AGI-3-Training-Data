"""PuzzleScript game: im_sick_today

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/I'm_Sick_Today.txt

Play with: python solver_client.py ps:im_sick_today
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("I'm_Sick_Today", seed=seed)
