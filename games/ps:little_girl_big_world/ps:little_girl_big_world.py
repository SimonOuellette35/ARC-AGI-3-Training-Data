"""PuzzleScript game: little_girl_big_world

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Little_Girl,_Big_World.txt

Play with: python solver_client.py ps:little_girl_big_world
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Little_Girl,_Big_World", seed=seed)
