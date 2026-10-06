"""pb03 -- Decoy Push. Push the crate onto the REAL pad; the decoy pad is a loss.

Actions: ACTION1 up, ACTION2 down, ACTION3 left, ACTION4 right (screen space --
the board is presented at a per-(seed, level) rotation).

Win : the crate is on the real pad.
Lose: the crate is pushed onto the decoy pad (instant), or the step budget runs out.

The decoy always sits within two cells of the real pad, so the two are told apart by
COLOUR, not by position -- and the bottom-left HUD tile is painted in the decoy's
colour as the on-screen key. Same engine as pb01/pb02
(``utils/push_puzzle.py``).
"""
from utils.push_puzzle import LevelSpec, PushPuzzleGame


class Pb03(PushPuzzleGame):
    game_name = "pb03"
    step_budget_base = 70
    step_budget_per_difficulty = 15
    specs = (
        LevelSpec(grid=(8, 8), difficulty=1, decoy=True),
        LevelSpec(grid=(8, 8), difficulty=2, decoy=True),
        LevelSpec(grid=(8, 8), difficulty=3, decoy=True, scatter=2),
        LevelSpec(grid=(10, 8), difficulty=4, decoy=True, barrier_gaps=1),
        LevelSpec(grid=(8, 8), difficulty=5, decoy=True, barrier_gaps=2),
    )
