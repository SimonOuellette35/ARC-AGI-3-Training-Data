"""PuzzleScript game: sticky_cubes

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Sticky_Cubes.txt

Play with: python solver_client.py ps:sticky_cubes
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Sticky_Cubes", seed=seed)
