"""PuzzleScript game: touchdown_heroes_prototype

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Touchdown_Heroes_(Prototype).txt

Play with: python solver_client.py ps:touchdown_heroes_prototype
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Touchdown_Heroes_(Prototype)", seed=seed)
