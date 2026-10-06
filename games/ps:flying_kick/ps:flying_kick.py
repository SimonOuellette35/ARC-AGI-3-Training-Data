"""PuzzleScript game: flying_kick

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Flying_Kick.txt

Play with: python solver_client.py ps:flying_kick
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Flying_Kick", seed=seed)
