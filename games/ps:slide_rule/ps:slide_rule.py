"""PuzzleScript game: slide_rule

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Slide_Rule.txt

Play with: python solver_client.py ps:slide_rule
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Slide_Rule", seed=seed)
