"""PuzzleScript game: ball_bros

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Ball_Bros.txt

Play with: python solver_client.py ps:ball_bros
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Ball_Bros", seed=seed)
