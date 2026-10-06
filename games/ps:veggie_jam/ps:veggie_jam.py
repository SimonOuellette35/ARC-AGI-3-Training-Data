"""PuzzleScript game: veggie_jam

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Veggie_Jam.txt

Play with: python solver_client.py ps:veggie_jam
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Veggie_Jam", seed=seed)
