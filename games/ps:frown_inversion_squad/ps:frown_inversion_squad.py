"""PuzzleScript game: frown_inversion_squad

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/FROWN_INVERSION_SQUAD.txt

Play with: python solver_client.py ps:frown_inversion_squad
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("FROWN_INVERSION_SQUAD", seed=seed)
