"""PuzzleScript game: spiders_hollow

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Spider's_Hollow.txt

Play with: python solver_client.py ps:spiders_hollow
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Spider's_Hollow", seed=seed)
