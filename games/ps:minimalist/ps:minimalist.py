"""PuzzleScript game: minimalist

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Minimalist.txt

Play with: python solver_client.py ps:minimalist
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Minimalist", seed=seed)
