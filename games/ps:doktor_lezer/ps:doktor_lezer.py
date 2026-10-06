"""PuzzleScript game: doktor_lezer

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Doktor_Lezer.txt

Play with: python solver_client.py ps:doktor_lezer
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Doktor_Lezer", seed=seed)
