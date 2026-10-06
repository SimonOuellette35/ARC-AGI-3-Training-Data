"""PuzzleScript game: fireproof_bomber

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Fireproof_bomber.txt

Play with: python solver_client.py ps:fireproof_bomber
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Fireproof_bomber", seed=seed)
