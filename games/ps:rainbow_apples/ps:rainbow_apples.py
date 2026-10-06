"""PuzzleScript game: rainbow_apples

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Rainbow_Apples.txt

Play with: python solver_client.py ps:rainbow_apples
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Per-level step limit, keyed by level index. Level 8 is a six-apple sokoban in
#: a maze and the best plan found for it is 189 presses (see
#: `solvers/generate_rainbow_apples_training.py` -- the level is beyond the macro
#: A*, so its plan comes from a beam and is not proved shortest); at the
#: adapter's 200-step default it would flip to GAME_OVER before the last apple
#: landed for any agent that so much as hesitates. Every entry is 3x the expert's
#: plan rounded up, floored at the 200 default so no level is made TIGHTER than
#: stock -- a real margin for a policy that wanders, while still ending one that
#: is looping. The shipped plan lengths are
#:
#:     level    0   1   2   3   4   5   6   7    8   9
#:     presses  7  22   5  13  31  60   7  25  189  10
#:
#: See `games/ps:cancel` and `games/ps:bridge` for the same treatment.
_STEP_LIMITS = {5: 200, 8: 600}

#: Levels outside the table keep the stock default.
_DEFAULT_STEPS = 200


class RainbowApplesAdapter(PuzzleScriptAdapter):
    """Rainbow Apples with per-level step limits; see `_STEP_LIMITS`."""

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = _STEP_LIMITS.get(self._current_level_index,
                                           _DEFAULT_STEPS)


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return RainbowApplesAdapter("Rainbow_Apples", seed=seed)
