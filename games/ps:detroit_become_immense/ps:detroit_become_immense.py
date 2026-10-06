"""PuzzleScript game: detroit_become_immense

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Detroit__Become_Immense.txt

Play with: python solver_client.py ps:detroit_become_immense
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Detroit__Become_Immense", seed=seed)
