"""PuzzleScript game: atlas_shrank_3

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Atlas_Shrank.txt

Play with: python solver_client.py ps:atlas_shrank_3
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Atlas_Shrank", seed=seed)
