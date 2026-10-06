"""PuzzleScript game: strange_warehouse

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Strange_Warehouse.txt

Play with: python solver_client.py ps:strange_warehouse
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Strange_Warehouse", seed=seed)
