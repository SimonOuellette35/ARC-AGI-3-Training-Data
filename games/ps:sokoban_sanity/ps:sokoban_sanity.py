"""PuzzleScript game: sokoban_sanity

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/sokoban_sanity.txt

Play with: python solver_client.py ps:sokoban_sanity
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("sokoban_sanity", seed=seed)
