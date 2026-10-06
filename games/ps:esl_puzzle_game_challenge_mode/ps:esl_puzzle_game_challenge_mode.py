"""PuzzleScript game: esl_puzzle_game_challenge_mode

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/ESL_Puzzle_Game_Challenge_Mode.txt

Play with: python solver_client.py ps:esl_puzzle_game_challenge_mode
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Per-level step limit, keyed by level index. This is the CHALLENGE MODE half
#: of the game (the author's stages 11-20) and its last two boards do not fit
#: the adapter's 200-step default: the expert's shortest known plans are
#:
#:     level    0   1   2   3   4   5   6   7   8   9
#:     presses 48  34  64  55  28  52  35  98  65  157
#:
#: so stage 20 (level 9) could not be finished at all inside 200 presses, and
#: stage 18 (level 7) would leave a perfect agent 102 presses of slack for a
#: 98-press solution. Each limit here is 3x the expert plan rounded up, floored
#: at the 200 default so no level is made TIGHTER than stock -- a real margin
#: for a policy that explores, while still ending an agent that is looping. See
#: `games/ps:entrepotphage_demake` and `games/ps:count_mover` for the same
#: treatment.
_STEP_LIMITS = {7: 294, 9: 471}

#: Every other level, and any level added past the table.
_DEFAULT_STEPS = 200


class ESLPuzzleGameAdapter(PuzzleScriptAdapter):
    """ESL Puzzle Game -- CHALLENGE MODE with per-level step limits; see
    `_STEP_LIMITS`. Exceeding a level's limit without reaching the win state
    flips the game to GameState.GAME_OVER.
    """

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = _STEP_LIMITS.get(self._current_level_index,
                                           _DEFAULT_STEPS)


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return ESLPuzzleGameAdapter("ESL_Puzzle_Game_Challenge_Mode", seed=seed)
