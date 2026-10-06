"""PuzzleScript game: the_observable_universe

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/The_Observable_Universe.txt

Play with: python solver_client.py ps:the_observable_universe
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("The_Observable_Universe", seed=seed)
