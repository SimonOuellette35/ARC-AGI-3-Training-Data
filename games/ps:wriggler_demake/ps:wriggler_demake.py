"""PuzzleScript game: wriggler_demake

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Wriggler_demake.txt

Play with: python solver_client.py ps:wriggler_demake
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Wriggler_demake", seed=seed)
