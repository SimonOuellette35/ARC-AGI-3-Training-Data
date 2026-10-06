"""PuzzleScript game: velocity_castle

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Velocity_Castle.txt

Play with: python solver_client.py ps:velocity_castle
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Velocity_Castle", seed=seed)
