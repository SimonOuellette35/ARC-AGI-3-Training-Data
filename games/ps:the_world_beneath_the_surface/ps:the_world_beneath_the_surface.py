"""PuzzleScript game: the_world_beneath_the_surface

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/The_World_Beneath_the_Surface.txt

Play with: python solver_client.py ps:the_world_beneath_the_surface
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("The_World_Beneath_the_Surface", seed=seed)
