"""PuzzleScript game: cakemonsters

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/cakemonsters.txt

Play with: python solver_client.py ps:cakemonsters
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("cakemonsters", seed=seed)
