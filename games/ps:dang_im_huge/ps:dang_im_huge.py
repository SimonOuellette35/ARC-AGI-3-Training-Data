"""PuzzleScript game: dang_im_huge

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Dang_I'm_Huge.txt

Play with: python solver_client.py ps:dang_im_huge
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Per-level step limit, keyed by level index. The adapter's 200-press default
#: is not enough for the last two boards, whose expert plans are
#:
#:     level     0   1   2   3   4   5   6   7
#:     presses  30  28  32  55  78  74  76 120
#:
#: -- level 7 cannot even be FINISHED inside 200 with anything but a nearly
#: perfect line of play, i.e. it would be unwinnable as shipped for any agent.
#: Every limit here is 3x the expert plan, floored at the 200 default so no
#: level is made tighter than stock: a real margin for a policy that wanders,
#: while still ending an agent that is looping. See `games/ps:count_mover` and
#: `games/ps:crocodiles_love_cookies` for the same treatment.
_STEP_LIMITS: dict[int, int] = {4: 234, 5: 222, 6: 228, 7: 360}

#: Every other level, and any level added past the table.
_DEFAULT_STEPS = 200


class DangImHugeAdapter(PuzzleScriptAdapter):
    """Dang I'm Huge with per-level step limits; see `_STEP_LIMITS`. Exceeding a
    level's limit without reaching the win state flips the game to
    GameState.GAME_OVER."""

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = _STEP_LIMITS.get(self._current_level_index,
                                           _DEFAULT_STEPS)


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return DangImHugeAdapter("Dang_I'm_Huge", seed=seed)
