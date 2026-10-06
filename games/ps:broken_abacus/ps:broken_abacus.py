"""PuzzleScript game: broken_abacus

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Broken_Abacus.txt

Play with: python solver_client.py ps:broken_abacus
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Broken_Abacus", seed=seed)
