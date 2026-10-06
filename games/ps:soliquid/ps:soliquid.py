"""PuzzleScript game: soliquid

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Soliquid.txt

Play with: python solver_client.py ps:soliquid
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Soliquid", seed=seed)
