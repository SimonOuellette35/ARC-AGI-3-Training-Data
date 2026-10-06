"""PuzzleScript game: darkness_sokoban

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Darkness_Sokoban.txt

Play with: python solver_client.py ps:darkness_sokoban
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Darkness_Sokoban", seed=seed)
