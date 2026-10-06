"""PuzzleScript game: rock_paper_scissors_v0_90_eq_v1_alpha

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Rock,_Paper,_Scissors_(v0.90_=_v1.alpha).txt

Play with: python solver_client.py ps:rock_paper_scissors_v0_90_eq_v1_alpha
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Rock,_Paper,_Scissors_(v0.90_=_v1.alpha)", seed=seed)
