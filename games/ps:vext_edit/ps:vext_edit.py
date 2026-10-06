"""PuzzleScript game: vext_edit

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/VEXT_EDIT.txt

Play with: python solver_client.py ps:vext_edit
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("VEXT_EDIT", seed=seed)
