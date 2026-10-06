"""PuzzleScript game: clean_up

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Clean_Up.txt

Play with: python solver_client.py ps:clean_up
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Clean_Up", seed=seed)
