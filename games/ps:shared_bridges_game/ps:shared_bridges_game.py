"""PuzzleScript game: shared_bridges_game

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Shared_Bridges_Game.txt

Play with: python solver_client.py ps:shared_bridges_game
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Shared_Bridges_Game", seed=seed)
