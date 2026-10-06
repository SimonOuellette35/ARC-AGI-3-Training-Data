"""PuzzleScript game: ad_infinitum_v0_4_2_implemented_highfloor_finished

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Ad_Infinitum_v0.4.2_[Implemented_HighFloor,_finished].txt

Play with: python solver_client.py ps:ad_infinitum_v0_4_2_implemented_highfloor_finished
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Ad_Infinitum_v0.4.2_[Implemented_HighFloor,_finished]", seed=seed)
