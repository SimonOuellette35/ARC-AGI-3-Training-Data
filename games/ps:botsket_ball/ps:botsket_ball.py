"""PuzzleScript game: botsket_ball

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Botsket_Ball.txt

Play with: python solver_client.py ps:botsket_ball
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Botsket_Ball", seed=seed)
