"""PuzzleScript game: bal_rus_curse_cruelty_free_demake

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Bal_Ru's_Curse_cruelty-free_demake.txt

Play with: python solver_client.py ps:bal_rus_curse_cruelty_free_demake
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Bal_Ru's_Curse_cruelty-free_demake", seed=seed)
