"""PuzzleScript game: every_three_steps_you_hit_a_wall_out_of_nowhere

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Every_Three_Steps_You_Hit_a_Wall_Out_of_Nowhere.txt

Play with: python solver_client.py ps:every_three_steps_you_hit_a_wall_out_of_nowhere
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Every_Three_Steps_You_Hit_a_Wall_Out_of_Nowhere", seed=seed)
