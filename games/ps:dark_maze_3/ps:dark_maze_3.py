"""PuzzleScript game: dark_maze_3

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Dark_Maze_3.txt

Play with: python solver_client.py ps:dark_maze_3
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Step limit for the single (randomly generated) level. Dark Maze 3 is a
#: 12x8-cell maze seen through a torch beam: the board is painted black except
#: for the corridor the player is standing in, so the exit cannot be walked to
#: until it has been FOUND, and finding it is most of the trajectory. The board
#: is 25x17 engine cells, i.e. two presses per maze cell, and an honest explorer
#: has to sweep the corridors until the exit falls inside its line of sight.
#:
#: Measured over 200 seeds, the frontier-exploration expert in
#: `solvers/generate_dark_maze_3_training.py` needs a mean of 91 presses, a 95th
#: percentile of 208 and a worst case of 288 -- so the adapter's 200-step default
#: would flip roughly a quarter of seeds to GAME_OVER before the exit had even
#: been seen, i.e. they would be unwinnable as shipped for any agent that cannot
#: see through walls. 900 is ~3x the measured worst case: real margin for a
#: policy that wanders (and for the perturbation bursts data generation adds),
#: while still ending an agent that is looping. See `games/ps:cancel` and
#: `games/ps:count_mover` for the same treatment.
_STEP_LIMIT = 900


class DarkMaze3Adapter(PuzzleScriptAdapter):
    """Dark Maze 3 with a step limit sized for a blind search; see `_STEP_LIMIT`.
    Exceeding it without reaching the exit flips the game to
    GameState.GAME_OVER."""

    def _do_reset(self):
        super()._do_reset()
        self._max_steps = _STEP_LIMIT


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return DarkMaze3Adapter("Dark_Maze_3", seed=seed)
