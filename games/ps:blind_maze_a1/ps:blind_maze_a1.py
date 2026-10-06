"""PuzzleScript game: blind_maze_a1

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Blind_Maze_a1.txt

Play with: python solver_client.py ps:blind_maze_a1
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Blind_Maze_a1", seed=seed)
