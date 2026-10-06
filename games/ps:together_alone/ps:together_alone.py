"""PuzzleScript game: together_alone

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Together_Alone.txt

Play with: python solver_client.py ps:together_alone
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Together_Alone", seed=seed)
