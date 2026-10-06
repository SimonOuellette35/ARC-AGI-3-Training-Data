"""PuzzleScript game: ims445_puzzlescript_game_buoy_deploy

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/IMS445_Puzzlescript_Game__Buoy_Deploy_.txt

Play with: python solver_client.py ps:ims445_puzzlescript_game_buoy_deploy
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Per-level step limit, keyed by level index; every other level keeps the
#: adapter's 200-press default.
#:
#: Buoy Deploy is a PULL sokoban on 17-wide boards, and moving a buoy one cell
#: costs two or three presses rather than one: after each push the player has to
#: walk back around to the far side of the buoy. Its expert plans are
#:
#:     level    0   1    2   3   4    5    6
#:     presses 136  30  158  60  67  177  126
#:
#: (see solvers/generate_ims445_buoy_deploy_training.py --report), so four of the
#: seven do not fit the default at all once an agent spends a single press
#: exploring -- they would be unwinnable as shipped. Those four get 2x their
#: expert plan, the same contract as `games/ps:count_mover` and
#: `games/ps:cancel`; the other three keep 200, which is already more than 3x.
_STEP_LIMITS = {0: 280, 2: 320, 5: 360, 6: 260}

#: Every other level, and any level added past the table.
_DEFAULT_STEPS = 200


class BuoyDeployAdapter(PuzzleScriptAdapter):
    """Buoy Deploy with per-level step limits; see `_STEP_LIMITS`. Exceeding a
    level's limit without reaching the win state flips the game to
    GameState.GAME_OVER.
    """

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = _STEP_LIMITS.get(self._current_level_index,
                                           _DEFAULT_STEPS)


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return BuoyDeployAdapter("IMS445_Puzzlescript_Game__Buoy_Deploy_", seed=seed)
