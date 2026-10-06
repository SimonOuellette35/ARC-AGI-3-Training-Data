"""PuzzleScript game: time_reversed_minicosmos

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Time-Reversed_Minicosmos.txt

Play with: python solver_client.py ps:time_reversed_minicosmos

Forty pull-sokoban levels -- Aymeric du Peloux's Minicosmos with the push rule
replaced by a pull, so a crate is dragged behind you instead of shoved ahead of
you. Pulling is expensive: to move a crate one square you have to walk all the
way around to its far side first, so the shortest wins here are long by sokoban
standards -- 12 to 187 presses, with three levels (indices 23, 27 and 29) over
170. The adapter's stock 200-action budget would leave those three with a margin
of 13 presses, i.e. no room for a single wrong turn, and an agent that explores
at all would hit GAME_OVER with the puzzle nearly solved. `_STEP_LIMIT` raises it
the way `games/ps:cancel`, `games/ps:bridge`, `games/ps:a_knights_tour` and
`games/ps:five_pulloban_puzzles` already do; nothing else about the adapter
changes.
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Actions allowed per level. The longest provably shortest win is 187 presses
#: (level 27; see solvers/generate_time_reversed_minicosmos_training.py
#: --plans), so this is ~3.2x it -- a real margin for a policy that wanders, and
#: wandering is expensive here: a pull cannot be undone by pressing back, so
#: recovering from one wrong drag costs tens of presses, not two. Still ends a
#: policy that loops. Never below the adapter's 200 default, so no level is made
#: TIGHTER than stock.
_STEP_LIMIT = 600


class TimeReversedMinicosmosAdapter(PuzzleScriptAdapter):
    """Time-Reversed Minicosmos with a raised step limit; see `_STEP_LIMIT`."""

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = max(_STEP_LIMIT, self._max_steps)


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return TimeReversedMinicosmosAdapter("Time-Reversed_Minicosmos", seed=seed)
