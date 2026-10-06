"""PuzzleScript game: tile_tiler

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Tile_Tiler.txt

Play with: python solver_client.py ps:tile_tiler
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Tile_Tiler", seed=seed)
