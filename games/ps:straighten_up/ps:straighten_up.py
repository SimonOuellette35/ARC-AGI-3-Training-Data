"""PuzzleScript game: straighten_up

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Straighten_Up.txt

Play with: python solver_client.py ps:straighten_up
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Straighten_Up", seed=seed)
