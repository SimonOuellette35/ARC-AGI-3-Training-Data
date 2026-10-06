"""PuzzleScript game: silly_rabbit

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Silly_Rabbit.txt

Play with: python solver_client.py ps:silly_rabbit
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Silly_Rabbit", seed=seed)
