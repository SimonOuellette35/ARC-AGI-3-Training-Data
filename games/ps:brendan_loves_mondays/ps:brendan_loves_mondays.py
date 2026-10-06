"""PuzzleScript game: brendan_loves_mondays

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Brendan_loves_mondays.txt

Play with: python solver_client.py ps:brendan_loves_mondays
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Brendan_loves_mondays", seed=seed)
