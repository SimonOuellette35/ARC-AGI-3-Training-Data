"""PuzzleScript game: this_adventure_world

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/This_Adventure_World.txt

Play with: python solver_client.py ps:this_adventure_world
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("This_Adventure_World", seed=seed)
