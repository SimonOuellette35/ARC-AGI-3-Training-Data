"""PuzzleScript game: aperture_science_sokoban_testing_initiative

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Aperture_Science_Sokoban_Testing_Initiative.txt

Play with: python solver_client.py ps:aperture_science_sokoban_testing_initiative
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Aperture_Science_Sokoban_Testing_Initiative", seed=seed)
