"""PuzzleScript game: zombie_rescue

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Zombie_Rescue.txt

Play with: python solver_client.py ps:zombie_rescue
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Zombie_Rescue", seed=seed)
