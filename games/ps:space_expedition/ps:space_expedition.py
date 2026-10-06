"""PuzzleScript game: space_expedition

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Space_Expedition.txt

Play with: python solver_client.py ps:space_expedition
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Space_Expedition", seed=seed)
