"""PuzzleScript game: four_room_tilt_mazes

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Four-room_tilt_mazes.txt

Play with: python solver_client.py ps:four_room_tilt_mazes
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Four-room_tilt_mazes", seed=seed)
