"""PuzzleScript game: bouphas_candle_quest

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Boupha's_Candle_Quest.txt

Play with: python solver_client.py ps:bouphas_candle_quest
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Boupha's_Candle_Quest", seed=seed)
