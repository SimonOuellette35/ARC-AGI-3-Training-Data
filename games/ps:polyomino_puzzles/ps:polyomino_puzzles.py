"""PuzzleScript game: polyomino_puzzles

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Polyomino_Puzzles.txt

Play with: python solver_client.py ps:polyomino_puzzles
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Polyomino_Puzzles", seed=seed)
