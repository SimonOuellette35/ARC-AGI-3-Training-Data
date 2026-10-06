"""PuzzleScript game: the_dungeon_of_squeamish_chickens

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/The_Dungeon_of_Squeamish_Chickens.txt

Play with: python solver_client.py ps:the_dungeon_of_squeamish_chickens
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("The_Dungeon_of_Squeamish_Chickens", seed=seed)
