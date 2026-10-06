"""PuzzleScript game: futuristic_block_pushing_game

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Futuristic_Block_Pushing_Game.txt

Play with: python solver_client.py ps:futuristic_block_pushing_game
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Futuristic_Block_Pushing_Game", seed=seed)
