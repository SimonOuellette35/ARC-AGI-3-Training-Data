"""PuzzleScript game: castle_elsewhere

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Castle_Elsewhere.txt

Play with: python solver_client.py ps:castle_elsewhere
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Castle_Elsewhere", seed=seed)
