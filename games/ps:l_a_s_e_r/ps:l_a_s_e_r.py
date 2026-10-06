"""PuzzleScript game: l_a_s_e_r

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/L.A.S.E.R.txt

Play with: python solver_client.py ps:l_a_s_e_r
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Per-level step limit, keyed by level index. L.A.S.E.R.'s stage-3 boards are
#: big sokoban puzzles whose emitters have to be shoved around as well as the
#: boxes, and the expert's shortest PROVED plans are
#:
#:   level    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
#:   presses  5  22  15  25  18   7  46  55  26  39  20  10  32  33  47  29  30
#:   level   17  18  19
#:   presses 184  68  56
#:
#: so level 17 cannot be finished inside the adapter's 200-step default by
#: anything that wanders even slightly, and level 18 leaves almost no margin --
#: both would flip to GAME_OVER with the last box still in hand, i.e. they would
#: be unwinnable as shipped for any real agent. Each limit here is 3x the expert
#: plan rounded up, floored at the 200 default so no level is made TIGHTER than
#: stock. See `games/ps:cancel` and `games/ps:bridge` for the same treatment.
_STEP_LIMITS = {17: 600, 18: 250}

#: Levels not in the table keep the stock default (3x their plan is under it).
_DEFAULT_STEPS = 200


class LaserAdapter(PuzzleScriptAdapter):
    """L.A.S.E.R. with per-level step limits; see `_STEP_LIMITS`."""

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = _STEP_LIMITS.get(self._current_level_index,
                                           _DEFAULT_STEPS)


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return LaserAdapter("L.A.S.E.R", seed=seed)
