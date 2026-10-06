"""PuzzleScript game: strata_gems

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/STRATA-GEMS.txt

Play with: python solver_client.py ps:strata_gems
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("STRATA-GEMS", seed=seed)
