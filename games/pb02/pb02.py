"""pb02 -- Two-Box Push. Two crates, two goal pads; every pad must be covered.

Actions: ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right (screen space --
the board is presented at a per-(seed, level) rotation).

Win : both crates are on pads.
Lose: the level's step budget runs out.

Same engine as pb01/pb03 (``utils/push_puzzle.py``); the only difference is two
crates, which is what makes the ordering matter -- parking the first crate can wall
off the route the second one needs.
"""
from utils.push_puzzle import LevelSpec, PushPuzzleGame


class Pb02(PushPuzzleGame):
    game_name = "pb02"
    step_budget_base = 80
    step_budget_per_difficulty = 20
    specs = (
        LevelSpec(grid=(8, 8), difficulty=1, crates=2),
        LevelSpec(grid=(8, 8), difficulty=2, crates=2, scatter=2),
        LevelSpec(grid=(8, 8), difficulty=3, crates=2, barrier_gaps=1),
        LevelSpec(grid=(10, 8), difficulty=4, crates=2, barrier_gaps=1),
        LevelSpec(grid=(8, 8), difficulty=5, crates=2, barrier_gaps=1, scatter=2),
    )
