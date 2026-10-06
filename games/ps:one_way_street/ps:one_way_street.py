"""PuzzleScript game: one_way_street

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/One_Way_Street.txt

Play with: python solver_client.py ps:one_way_street
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("One_Way_Street", seed=seed)
