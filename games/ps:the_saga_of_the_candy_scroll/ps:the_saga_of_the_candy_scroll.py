"""PuzzleScript game: the_saga_of_the_candy_scroll

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/The_Saga_of_the_Candy_Scroll.txt

Play with: python solver_client.py ps:the_saga_of_the_candy_scroll
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("The_Saga_of_the_Candy_Scroll", seed=seed)
