"""Solve ball_sifting from the live paddles, balls and wall geometry.

The expert empties the lowest occupied paddle first, keeping its route clear of
other balls. It tries direct drops (including skips through aligned holes), then
positions a receiving paddle if needed. Each proposed transfer is checked with
the game's own falling/stacking rules on a lightweight, non-rendering copy.
Among safe transfers it prefers fewer immediate actions; this is a constructive
expert, not a claim of globally shortest play across every possible ball merge.

BaseSolver supplies rotation, full drop-animation recording, recovery and the
usual CLI. Importing this module or calling solve_from writes no training data.
To generate data explicitly, run from the repository root:
    python solvers/generate_ball_sifting_training.py --episodes 1000
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameState  # noqa: E402
from games.ball_sifting.ball_sifting import (  # noqa: E402
    BOARD, PAD_CLAMP, PW, BallSifting,
)
from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import Action  # noqa: E402


class _Preview(BallSifting):
    """Only the state used by native movement and drop physics; no sprites.

    Inheriting the mechanics avoids an approximate second collision model. The
    engine lifecycle/rendering hooks below are deliberately replaced for search.
    """

    def __init__(self, game):
        self._N = game._N
        self._hw = game._hw
        self._hx = tuple(game._hx)
        self._wy = tuple(game._wy)
        self._padY = tuple(game._padY)
        self._pads = [{"x": p["x"]} for p in game._pads]
        self._balls = [{k: v for k, v in b.items() if k != "sprite"}
                       for b in game._balls]
        self._sel = game._sel
        self._falling = []
        self._solid = set()
        self.dead = False
        self.solved = False

    def _render(self):
        pass

    def complete_action(self):
        pass

    def lose(self):
        self.dead = True

    def next_level(self):
        self.solved = True

    def drop(self):
        if not self._begin_drop():
            return False
        # Every tick moves or settles a ball. This also bounds malformed states.
        for _ in range(BOARD * len(self._balls) + 1):
            self._fall_tick()
            if not self._falling:
                return not self.dead
        return False


def _select(game, tier, actions):
    if game._sel != tier:
        actions.append(Action(6, (game._padY[tier], game._pads[tier]["x"] + PW // 2)))
        game._sel = tier


def _move(game, tier, x, actions):
    old = game._pads[tier]["x"]
    if old == x:
        return
    _select(game, tier, actions)
    if abs(x - old) == 1:
        actions.append(Action(3 if x < old else 4))
    else:
        # Row zero cannot select a paddle. Clicks address its centre, not edge.
        actions.append(Action(6, (0, x + PW // 2)))
    game._set_pad_x(tier, x)


def _fit_bounds(game, tier):
    balls = [b for b in game._balls if b["on"] == tier]
    left = min(b["dx"] for b in balls)
    right = max(b["dx"] + 3 for b in balls)
    lo = max(0, game._hx[tier] - left)
    hi = min(PAD_CLAMP, game._hx[tier] + game._hw - right)
    return left, right, lo, hi


def _transfer(game, tier):
    """Return a verified downhill transfer and its resulting preview state."""
    left, right, lo, hi = _fit_bounds(game, tier)
    if lo > hi:
        return None  # A rigid cluster cannot fit the wall below this paddle.

    candidates = []
    for x in range(lo, hi + 1):
        # First try the lower paddles as they are. A drop may skip several tiers.
        catches = [None]
        if tier + 1 < game._N:
            # Catch every ball fully on the next paddle, avoiding overhangs.
            catch_lo = max(0, x + right - PW)
            catch_hi = min(PAD_CLAMP, x + left)
            catches += [cx for cx in range(catch_lo, catch_hi + 1)
                        if cx != game._pads[tier + 1]["x"]]
        for catch in catches:
            trial = _Preview(game)
            actions = []
            if catch is not None:
                _move(trial, tier + 1, catch, actions)
            _move(trial, tier, x, actions)
            _select(trial, tier, actions)
            actions.append(Action(5))
            candidates.append((actions, trial))

    # Stable ties prefer leaving the receiving paddle alone. Arrow nudges and
    # arbitrary-distance clicks each count as one action.
    candidates.sort(key=lambda candidate: len(candidate[0]))
    for actions, trial in candidates:
        if not trial.drop():
            continue
        # A catch fixes new ball offsets. Near the board edges, a paddle can
        # catch successfully yet be unable to align those offsets with its hole.
        # Reject that placement before it strands the real game.
        occupied = {b["on"] for b in trial._balls if b["on"] >= 0}
        if any(_fit_bounds(trial, k)[2] > _fit_bounds(trial, k)[3]
               for k in occupied):
            continue
        if all(b["on"] == -1 or b["on"] > tier for b, old in
               zip(trial._balls, game._balls) if old["on"] == tier):
            return actions, trial
    return None


class BallSiftingSolver(BaseSolver):
    game_id = "ball_sifting"
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        return BallSifting(seed=seed)

    def available_actions(self, game) -> list[int]:
        # Selecting any paddle by click covers ACTION7's cycling effect, and
        # follows the shared corpus convention of action ids 0 through 6.
        return [1, 2, 3, 4, 5, 6]

    def solve_from(self, game, level_idx: int, seed: int) -> list[Action]:
        if game._state in (GameState.GAME_OVER, GameState.WIN) or game._falling:
            return []
        if any(b["on"] < -1 for b in game._balls):
            return []
        preview = _Preview(game)
        plan = []
        # Each transfer moves at least one ball strictly down. Already-banked
        # balls stay in the model so basin collisions are checked as well.
        for _ in range(game._N * len(game._balls)):
            occupied = [b["on"] for b in preview._balls if b["on"] >= 0]
            if not occupied:
                return plan
            transfer = _transfer(preview, max(occupied))
            if transfer is None:
                return []
            actions, preview = transfer
            plan.extend(actions)
            if preview.solved:
                return plan
        return []


if __name__ == "__main__":
    sys.exit(BallSiftingSolver.main())
