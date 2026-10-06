"""PuzzleScript game: beams_and_flowers

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Beams_and_Flowers.txt

Play with: python solver_client.py ps:beams_and_flowers
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Beams_and_Flowers", seed=seed)
