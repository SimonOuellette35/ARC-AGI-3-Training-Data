"""PuzzleScript game: party_demon

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Party_Demon.txt

Play with: python solver_client.py ps:party_demon
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Party_Demon", seed=seed)
