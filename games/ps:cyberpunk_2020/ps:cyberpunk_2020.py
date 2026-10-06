"""PuzzleScript game: cyberpunk_2020

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Cyberpunk_2020.txt

Play with: python solver_client.py ps:cyberpunk_2020
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Cyberpunk_2020", seed=seed)
