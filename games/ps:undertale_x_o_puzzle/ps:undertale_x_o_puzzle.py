"""PuzzleScript game: undertale_x_o_puzzle

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Undertale_X_O_Puzzle.txt

Play with: python solver_client.py ps:undertale_x_o_puzzle
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Undertale_X_O_Puzzle", seed=seed)
