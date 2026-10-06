"""PuzzleScript game: bridge_toggle_maze

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Bridge-toggle_Maze.txt

Play with: python solver_client.py ps:bridge_toggle_maze
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Bridge-toggle_Maze", seed=seed)
