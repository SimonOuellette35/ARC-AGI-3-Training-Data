"""PuzzleScript game: a_knights_tour

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/A_Knight's_Tour.txt

Play with: python solver_client.py ps:a_knights_tour
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


class AKnightsTourAdapter(PuzzleScriptAdapter):
    """A Knight's Tour with per-level step limits.

    Reaching the limit flips the game to GameState.GAME_OVER (the engine's
    lose state). Limits start at 50 for level 1 and grow by 15 per level,
    yielding 50, 65, 80, ..., 200 across the 11 levels.

    The slope used to be 10 per level, which made level 4 UNWINNABLE: a jump
    costs exactly 4 turns (3 cursor presses + ACTION, the cursor being on the
    knight after every capture), and its 12 pawns need 23 jumps = 92 turns
    against a 90-turn budget. Optimal play (see
    ``solvers/generate_a_knights_tour_training.py``, which searches the
    interpreter with an admissible heuristic) needs 40, 40, 64, 64, 92, 24, 60,
    116, 64, 88, 64 turns on levels 0..10, so a slope of 15 leaves every level a
    real margin while keeping the limit a genuine pressure on the rook levels.
    """

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = 50 + self._current_level_index * 15


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return AKnightsTourAdapter("A_Knight's_Tour", seed=seed)
