"""PuzzleScript game: tractor_beam_sokoban9

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Tractor_Beam_Sokoban9.txt

Play with: python solver_client.py ps:tractor_beam_sokoban9
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Tractor_Beam_Sokoban9", seed=seed)
