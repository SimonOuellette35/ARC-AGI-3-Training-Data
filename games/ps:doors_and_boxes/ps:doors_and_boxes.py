"""PuzzleScript game: doors_and_boxes

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Doors_and_Boxes.txt

Play with: python solver_client.py ps:doors_and_boxes
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Doors_and_Boxes", seed=seed)
