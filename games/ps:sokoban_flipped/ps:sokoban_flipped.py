"""PuzzleScript game: sokoban_flipped

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Sokoban_Flipped.txt

Play with: python solver_client.py ps:sokoban_flipped
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Sokoban_Flipped", seed=seed)
