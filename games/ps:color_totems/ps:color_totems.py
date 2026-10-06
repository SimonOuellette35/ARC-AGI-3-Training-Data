"""PuzzleScript game: color_totems

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Color_Totems.txt

Play with: python solver_client.py ps:color_totems
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Per-level step limit, keyed by level index.
#:
#: Color Totems is a collection game: the golem has to pick up EVERY coin and
#: then stand on the exit, so a plan is a tour of the whole board and the later
#: levels carry dozens of coins (level 7 has 45). `PuzzleScriptAdapter` gives a
#: level 200 actions and flips to GAME_OVER on the 201st, which several of these
#: boards cannot be finished inside -- and the failure is SILENT: the plan
#: replays, the frames are right, the level simply never reaches WIN.
#:
#: Each limit is 3x the expert's shortest known plan rounded up, floored at the
#: 200 default so no level is made TIGHTER than stock -- real margin for a policy
#: that wanders, while still ending one that is looping:
#:
#:     level    0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
#:     presses 57  35  63  37  73 121  37  99 115  93  79  77 173  65  49 107  77
#:
#: `--plans` prints the budget beside the plan length for exactly this check. See
#: `games/ps:cancel` and `games/ps:bridge` for the same treatment.
#:
#: Level 17 is the one the expert cannot solve, so it has no measured plan to
#: scale from -- but it carries 272 coins, so ANY solution is at least 272
#: presses and the stock 200 makes it unwinnable outright. It gets a flat 2000.
_STEP_LIMITS = {
    0: 200, 1: 200, 2: 200, 3: 200, 4: 250, 5: 400, 6: 200, 7: 300,
    8: 350, 9: 300, 10: 250, 11: 250, 12: 550, 13: 300, 14: 200,
    15: 350, 16: 250, 17: 2000,
}

#: Levels past the table keep the stock default.
_DEFAULT_STEPS = 200


class ColorTotemsAdapter(PuzzleScriptAdapter):
    """Color Totems with per-level step limits; see `_STEP_LIMITS`."""

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = _STEP_LIMITS.get(self._current_level_index,
                                           _DEFAULT_STEPS)


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return ColorTotemsAdapter("Color_Totems", seed=seed)
