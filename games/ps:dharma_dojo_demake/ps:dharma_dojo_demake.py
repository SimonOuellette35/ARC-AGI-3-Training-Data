"""PuzzleScript game: dharma_dojo_demake

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Dharma_Dojo_demake.txt

Play with: python solver_client.py ps:dharma_dojo_demake
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Dharma_Dojo_demake", seed=seed)
