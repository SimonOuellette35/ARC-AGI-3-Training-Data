"""PuzzleScript game: twinpush

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/TwinPush.txt

Play with: python solver_client.py ps:twinpush
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("TwinPush", seed=seed)
