"""PuzzleScript game: swap_sokoban

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Swap_Sokoban.txt

Play with: python solver_client.py ps:swap_sokoban
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Swap_Sokoban", seed=seed)
