"""PuzzleScript game: there_is_no_one_here_to_help_you

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/There_is_no-one_here_to_help_you.txt

Play with: python solver_client.py ps:there_is_no_one_here_to_help_you
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("There_is_no-one_here_to_help_you", seed=seed)
