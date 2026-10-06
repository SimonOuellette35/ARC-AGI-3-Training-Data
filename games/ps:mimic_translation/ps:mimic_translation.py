"""PuzzleScript game: mimic_translation

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Mimic_Translation.txt

Play with: python solver_client.py ps:mimic_translation
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Mimic_Translation", seed=seed)
