"""PuzzleScript game: escaping_limbo

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Escaping_Limbo.txt

Play with: python solver_client.py ps:escaping_limbo
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Escaping_Limbo", seed=seed)
