"""PuzzleScript game: time_reversed_microban

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Time-reversed_Microban.txt

Play with: python solver_client.py ps:time_reversed_microban
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Time-reversed_Microban", seed=seed)
