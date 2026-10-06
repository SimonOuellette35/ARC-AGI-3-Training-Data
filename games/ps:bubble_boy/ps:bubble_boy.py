"""PuzzleScript game: bubble_boy

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Bubble_Boy.txt

Play with: python solver_client.py ps:bubble_boy
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Bubble_Boy", seed=seed)
