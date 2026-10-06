"""PuzzleScript game: circuit_breaker

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Circuit_Breaker.txt

Play with: python solver_client.py ps:circuit_breaker
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("Circuit_Breaker", seed=seed)
