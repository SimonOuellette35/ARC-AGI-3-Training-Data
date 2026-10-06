"""PuzzleScript game: cancel

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Cancel.txt

Play with: python solver_client.py ps:cancel
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Per-level step limit, keyed by level index. Cancel's later boards are big
#: carrying puzzles -- every element moves one cell per press and only ever into
#: the cell the player just left, so repositioning between piece-moves is most
#: of the solution -- and three of them simply cannot be finished inside the
#: adapter's 200-step default: the expert's shortest known plans are
#:
#:     level  0   1   2   3   4    5    6    7
#:     presses 25  26  58  34  88  287  212  180
#:
#: so levels 5-7 would flip to GAME_OVER before the last pair cancelled, i.e.
#: they would be unwinnable as shipped for any agent. Each limit here is 3x the
#: expert plan rounded up, floored at the 200 default so no level is made
#: TIGHTER than stock -- a real margin for a policy that wanders, while still
#: ending an agent that is looping. See `games/ps:bridge` and
#: `games/ps:a_knights_tour` for the same treatment.
_STEP_LIMITS = {0: 200, 1: 200, 2: 200, 3: 200, 4: 300,
                5: 900, 6: 650, 7: 550}

#: Levels past the table (there are none today) keep the stock default.
_DEFAULT_STEPS = 200


class CancelAdapter(PuzzleScriptAdapter):
    """Cancel with per-level step limits; see `_STEP_LIMITS`."""

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = _STEP_LIMITS.get(self._current_level_index,
                                           _DEFAULT_STEPS)


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return CancelAdapter("Cancel", seed=seed)
