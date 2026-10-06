"""PuzzleScript game: enqueue

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Enqueue.txt

Play with: python solver_client.py ps:enqueue
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Enqueue", seed=seed)
