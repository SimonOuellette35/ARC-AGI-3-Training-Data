"""PuzzleScript game: count_mover

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Count_Mover.txt

Play with: python solver_client.py ps:count_mover
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Per-level step limit, keyed by level index. Count Mover is a game ABOUT
#: efficiency ("you can count on me to move your boxes as efficiently as
#: possible"), so it runs tighter than the adapter's 200-step default: the
#: shipped board is 35 moves and was given 60, a 1.7x margin.
#:
#: The twelve levels added around it keep that contract. Their expert plans are
#:
#:     level    0   1   2   3   4   5   6   7   8   9  10  11  12
#:     moves    9  11  13  15  20  22  31  33  35  42  45  61  69
#:
#: (level 8 is the shipped board), so 60 still leaves at least the shipped
#: level's margin everywhere up to level 8 and is kept unchanged there. It does
#: NOT for the last four -- levels 11 and 12 cannot even be finished inside it,
#: i.e. they would be unwinnable as shipped for any agent -- so those get 2x
#: their expert plan. See `games/ps:cancel` for the same treatment.
_STEP_LIMITS = {9: 84, 10: 90, 11: 122, 12: 138}

#: Every other level, and any level added past the table.
_DEFAULT_STEPS = 60


class CountMoverAdapter(PuzzleScriptAdapter):
    """Count Mover with per-level step limits; see `_STEP_LIMITS`. Exceeding a
    level's limit without reaching the win state flips the game to
    GameState.GAME_OVER.
    """

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = _STEP_LIMITS.get(self._current_level_index,
                                           _DEFAULT_STEPS)


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return CountMoverAdapter("Count_Mover", seed=seed)
