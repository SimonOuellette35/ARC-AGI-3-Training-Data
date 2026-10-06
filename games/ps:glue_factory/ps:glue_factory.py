"""PuzzleScript game: glue_factory

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Glue_Factory.txt

Play with: python solver_client.py ps:glue_factory
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Glue_Factory", seed=seed)
