"""PuzzleScript game: tour_de_four

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/_Tour_de_Four.txt

Play with: python solver_client.py ps:tour_de_four
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("_Tour_de_Four", seed=seed)
