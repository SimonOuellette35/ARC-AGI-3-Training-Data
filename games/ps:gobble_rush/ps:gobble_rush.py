"""PuzzleScript game: gobble_rush

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Gobble_Rush!.txt

Play with: python solver_client.py ps:gobble_rush
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Gobble_Rush!", seed=seed)
