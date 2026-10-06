"""PuzzleScript game: stand_aside_everyone_i_take_large_steps

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Stand_aside,_everyone!_I_take_large_steps!.txt

Play with: python solver_client.py ps:stand_aside_everyone_i_take_large_steps
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Stand_aside,_everyone!_I_take_large_steps!", seed=seed)
