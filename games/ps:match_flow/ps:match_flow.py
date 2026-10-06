"""PuzzleScript game: match_flow

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Match_Flow.txt

Play with: python solver_client.py ps:match_flow
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Match_Flow", seed=seed)
