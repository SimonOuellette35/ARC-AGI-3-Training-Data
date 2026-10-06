"""PuzzleScript game: collect_gnocchi

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Collect_Gnocchi.txt

Play with: python solver_client.py ps:collect_gnocchi
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Collect_Gnocchi", seed=seed)
