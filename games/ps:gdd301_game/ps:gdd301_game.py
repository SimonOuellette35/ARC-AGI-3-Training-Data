"""PuzzleScript game: gdd301_game

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/GDD301_Game.txt

Play with: python solver_client.py ps:gdd301_game
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("GDD301_Game", seed=seed)
