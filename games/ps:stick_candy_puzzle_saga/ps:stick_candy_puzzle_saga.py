"""PuzzleScript game: stick_candy_puzzle_saga

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/stick_candy_puzzle_saga.txt

Play with: python solver_client.py ps:stick_candy_puzzle_saga
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("stick_candy_puzzle_saga", seed=seed)
