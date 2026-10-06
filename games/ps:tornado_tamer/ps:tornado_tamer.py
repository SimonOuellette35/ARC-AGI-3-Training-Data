"""PuzzleScript game: tornado_tamer

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Tornado_Tamer.txt

Play with: python solver_client.py ps:tornado_tamer
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Tornado_Tamer", seed=seed)
