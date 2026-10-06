"""PuzzleScript game: leo

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Leo.txt

Play with: python solver_client.py ps:leo
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Per-level step limit, keyed by level index. Leo asks for two chores at once --
#: every water tile under an elephant AND every poacher annihilated against an
#: opposite one -- so its later boards are long: the expert's plans run to
#: several hundred presses on the levels that carry eight elephants and ten
#: poachers, well past the adapter's 200-step default, and a level whose plan
#: does not fit flips to GAME_OVER with the last poacher still standing, i.e. it
#: is unwinnable as shipped for any real agent.
#:
#: Each limit is ~3x the expert plan rounded up, floored at the 200 default so no
#: level is made TIGHTER than stock. See `games/ps:l_a_s_e_r`, `games/ps:cancel`
#: and `games/ps:bridge` for the same treatment.
_STEP_LIMITS = {
    4: 400, 5: 700, 6: 700, 7: 900, 8: 700, 9: 900, 10: 700, 11: 900,
}

#: Levels not in the table keep the stock default (3x their plan is under it).
_DEFAULT_STEPS = 200


class LeoAdapter(PuzzleScriptAdapter):
    """Leo with per-level step limits; see `_STEP_LIMITS`."""

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = _STEP_LIMITS.get(self._current_level_index,
                                           _DEFAULT_STEPS)


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return LeoAdapter("Leo", seed=seed)
