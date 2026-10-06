"""PuzzleScript game: drop_maze

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Drop_Maze.txt

Play with: python solver_client.py ps:drop_maze
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Drop_Maze", seed=seed)
