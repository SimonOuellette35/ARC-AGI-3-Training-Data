"""PuzzleScript game: travelling_salesman

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Travelling_salesman.txt

Play with: python solver_client.py ps:travelling_salesman
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Travelling_salesman", seed=seed)
