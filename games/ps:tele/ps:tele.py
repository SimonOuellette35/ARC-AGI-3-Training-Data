"""PuzzleScript game: tele

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Tele.txt

Play with: python solver_client.py ps:tele
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Tele", seed=seed)
