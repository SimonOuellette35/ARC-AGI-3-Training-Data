"""pb01 -- One-Box Push. Push the single crate onto the goal pad.

Actions: ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right (screen space --
the board is presented at a per-(seed, level) rotation).

Win : the crate is on the goal pad.
Lose: the level's step budget runs out.

The mechanic, the HUD and the per-seed augmentation (layout, walls, colours,
rotation) live in ``utils/push_puzzle.py``, shared with pb02 (two crates) and pb03
(decoy pad). This file is just pb01's five level shapes: the difficulty curve runs
open floor -> scattered blocks -> a corridor barrier -> a bigger board with a stub
-> a multi-gap barrier.
"""
from utils.push_puzzle import LevelSpec, PushPuzzleGame


class Pb01(PushPuzzleGame):
    game_name = "pb01"
    step_budget_base = 60
    step_budget_per_difficulty = 15
    specs = (
        LevelSpec(grid=(8, 8), difficulty=1),
        LevelSpec(grid=(8, 8), difficulty=2, scatter=4),
        LevelSpec(grid=(8, 8), difficulty=3, barrier_gaps=1),
        LevelSpec(grid=(10, 10), difficulty=4, barrier_gaps=2, stub=3),
        LevelSpec(grid=(8, 8), difficulty=5, barrier_gaps=3, scatter=2),
    )
