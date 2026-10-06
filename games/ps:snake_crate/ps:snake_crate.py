"""PuzzleScript game: snake_crate

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Snake_Crate.txt

Play with: python solver_client.py ps:snake_crate
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Snake_Crate", seed=seed)
