"""PuzzleScript game: mini_nomerads

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Mini_Nomerads.txt

Play with: python solver_client.py ps:mini_nomerads
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Mini_Nomerads", seed=seed)
