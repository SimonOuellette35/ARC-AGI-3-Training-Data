"""PuzzleScript game: boolean_bloom_0_37

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Boolean_Bloom_0.37.txt

Play with: python solver_client.py ps:boolean_bloom_0_37
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Boolean_Bloom_0.37", seed=seed)
