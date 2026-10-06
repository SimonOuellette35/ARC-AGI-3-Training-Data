"""PuzzleScript game: drop_kick

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Drop_Kick.txt

Play with: python solver_client.py ps:drop_kick
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Drop_Kick", seed=seed)
