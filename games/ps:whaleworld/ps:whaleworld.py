"""PuzzleScript game: whaleworld

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/whaleworld.txt

Play with: python solver_client.py ps:whaleworld
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("whaleworld", seed=seed)
