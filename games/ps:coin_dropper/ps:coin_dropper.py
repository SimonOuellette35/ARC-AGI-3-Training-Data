"""PuzzleScript game: coin_dropper

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Coin_Dropper.txt

Play with: python solver_client.py ps:coin_dropper
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Coin_Dropper", seed=seed)
