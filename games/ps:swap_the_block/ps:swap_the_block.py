"""PuzzleScript game: swap_the_block

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/_Swap_the_block!.txt

Play with: python solver_client.py ps:swap_the_block
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("_Swap_the_block!", seed=seed)
