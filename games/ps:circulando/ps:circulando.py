"""PuzzleScript game: circulando

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Circulando.txt

Play with: python solver_client.py ps:circulando
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Circulando", seed=seed)
