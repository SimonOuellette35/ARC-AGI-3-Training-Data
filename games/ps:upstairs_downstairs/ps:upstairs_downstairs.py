"""PuzzleScript game: upstairs_downstairs

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Upstairs_Downstairs.txt

Play with: python solver_client.py ps:upstairs_downstairs
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Upstairs_Downstairs", seed=seed)
