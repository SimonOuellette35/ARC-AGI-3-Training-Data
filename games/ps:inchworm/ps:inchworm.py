"""PuzzleScript game: inchworm

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Inchworm.txt

Play with: python solver_client.py ps:inchworm
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Inchworm", seed=seed)
