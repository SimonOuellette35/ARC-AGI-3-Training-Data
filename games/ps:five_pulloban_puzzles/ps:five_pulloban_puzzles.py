"""PuzzleScript game: five_pulloban_puzzles

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Five_Pulloban_Puzzles.txt

Play with: python solver_client.py ps:five_pulloban_puzzles

The whole game is ONE PuzzleScript level: a 55x10 archipelago cut into five
11x10 flickscreens, each an independent pull-sokoban island, joined by raft
bridges that only open once the island behind you is finished. So the single
"level" is really five puzzles plus four bridge crossings back to back, and the
shortest known win takes 231 presses -- comfortably past the adapter's stock
200-action budget, which would flip the episode to GAME_OVER somewhere on the
fourth island. `_STEP_LIMIT` raises it the way `games/ps:cancel`,
`games/ps:bridge` and `games/ps:a_knights_tour` already do; nothing else about
the adapter changes.
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Actions allowed for the game's single level. The expert's shortest plan is
#: 231 presses (33 + 9 + 15 + 19 + 60 + 17 + 23 + 22 + 33), so this is ~3x it --
#: a real margin for a policy that wanders, while still ending one that loops.
#: Never below the adapter's 200 default, so no level is made TIGHTER than stock.
_STEP_LIMIT = 700


class FivePullobanPuzzlesAdapter(PuzzleScriptAdapter):
    """Five Pulloban Puzzles with a raised step limit; see `_STEP_LIMIT`."""

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = max(_STEP_LIMIT, self._max_steps)


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return FivePullobanPuzzlesAdapter("Five_Pulloban_Puzzles", seed=seed)
