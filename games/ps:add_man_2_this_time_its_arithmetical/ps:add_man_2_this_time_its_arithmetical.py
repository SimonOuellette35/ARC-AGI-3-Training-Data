"""PuzzleScript game: add_man_2_this_time_its_arithmetical

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Add_Man_2__This_Time_It's_Arithmetical.txt

Play with: python solver_client.py ps:add_man_2_this_time_its_arithmetical
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    adapter = PuzzleScriptAdapter("Add_Man_2__This_Time_It's_Arithmetical", seed=seed)

    # The original game uses Blue + DarkBlue for the goal indicators, but the
    # ARC palette has no distinct dark blue, so both map to the same index and
    # the 0/1 indicator becomes invisible. Recolor the indicator to light blue
    # so the goal type is readable.
    for goal_name in ("zerogoal", "onegoal"):
        obj = adapter._game.objects.get(goal_name)
        if obj is not None and len(obj.colors) >= 2:
            obj.colors[1] = 10  # light blue

    adapter._do_reset()
    return adapter
