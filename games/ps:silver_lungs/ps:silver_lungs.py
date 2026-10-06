"""PuzzleScript game: silver_lungs

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/silver_lungs.txt

Play with: python solver_client.py ps:silver_lungs
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("silver_lungs", seed=seed)
