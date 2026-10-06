"""PuzzleScript game: sokoban_match3

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/sokoban_match3.txt

Play with: python solver_client.py ps:sokoban_match3
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("sokoban_match3", seed=seed)
