"""PuzzleScript game: the_workshop

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/The_Workshop.txt

Play with: python solver_client.py ps:the_workshop
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("The_Workshop", seed=seed)
