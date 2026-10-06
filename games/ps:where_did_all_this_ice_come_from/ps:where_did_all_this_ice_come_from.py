"""PuzzleScript game: where_did_all_this_ice_come_from

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/where_did_all_this_ice_come_from_.txt

Play with: python solver_client.py ps:where_did_all_this_ice_come_from
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("where_did_all_this_ice_come_from_", seed=seed)
