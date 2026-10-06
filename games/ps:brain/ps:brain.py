"""PuzzleScript game: brain

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Brain.txt

Play with: python solver_client.py ps:brain
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Brain", seed=seed)
