"""PuzzleScript game: candy_bomb

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Candy_Bomb.txt

Play with: python solver_client.py ps:candy_bomb
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Candy_Bomb", seed=seed)
