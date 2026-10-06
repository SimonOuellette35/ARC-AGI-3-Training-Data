"""PuzzleScript game: dont_play_on_the_ice

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Don't_Play_On_the_Ice.txt

Play with: python solver_client.py ps:dont_play_on_the_ice
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Don't_Play_On_the_Ice", seed=seed)
