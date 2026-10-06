"""PuzzleScript game: sokubunny_and_the_colored_boxes

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Sokubunny_and_the_colored_Boxes.txt

Play with: python solver_client.py ps:sokubunny_and_the_colored_boxes
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Sokubunny_and_the_colored_Boxes", seed=seed)
