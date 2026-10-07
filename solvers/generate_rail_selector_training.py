"""Generate Phase-1 training data for the rail_selector game.

Rail Selector (``games/rail_selector/rail_selector.py``) is a *selection* puzzle:
n horizontal rails, each carrying exactly one coloured tile and one same-coloured
target marker. Only the SELECTED rail's tile responds to a slide, so every action
is either re-aiming the selector or moving the aimed tile::

    ACTION1  selector up      (r -> max(0, r-1))
    ACTION2  selector down    (r -> min(n-1, r+1))
    ACTION3  slide left       (p[r] -> p[r]-1, clamped to the rail)
    ACTION4  slide right      (p[r] -> p[r]+1, clamped to the rail)
    ACTION7  undo the last slide

The level is won when every tile sits on its target. Tiles never interact (one per
rail, non-collidable), so there is nothing to search: the cost of a state is
CLOSED FORM, and `solve_from` is a greedy descent on it.

The exact metric
----------------
With ``S = {r : p[r] != t[r]}`` the still-wrong rails, a plan must spend at least
``sum |p[r]-t[r]|`` slides (one slide changes one tile by one) and at least
``selector_cost(r, S)`` selector presses (the shortest walk on a line that starts
at ``r`` and visits every rail in ``S``: go straight to one end, or double back
past the near end first). Slides and selector presses are disjoint action sets, so
the sum is a lower bound -- and the "walk to one end, finishing each rail as you
pass it" route achieves it. Hence::

    dist(r, p) = sum_S |p[r]-t[r]| + selector_cost(r, S)

is exact, which makes the optimal-action SET simply every action whose successor
has ``dist == dist - 1``. (Both claims are checked against a brute-force BFS over
3000 random boards in the smoke tests: 0 mismatches on the metric, 0 on the set.)

Where the ties are -- and are NOT
--------------------------------
The shortest plan from a level's INITIAL state is UNIQUE, so stochastic-optimal
sampling is a no-op there and every episode records the same canonical route per
board (measured: 1950/1950 singleton tie sets). The reason is that the selector
always starts at rail 0 and the optimal route never moves it back up: while
``sel < min(S)`` only "selector down" descends the metric, and once ``sel ==
min(S)`` only the one slide direction that closes that rail's gap does. So the
selector is always at or above every unfinished rail, which is exactly the
condition under which the walk has no choice to make.

Ties DO exist -- once the selector sits strictly between the extremes of ``S``,
walking to either end can be equally good, and so can "slide the current rail
now" versus "keep walking" -- but that only happens after something has moved the
selector off the optimal walk. In practice that means the base's exploration
prefix and perturbation bursts, which is where `optimal_set_from` earns its keep:
it labels those recovery states with the FULL set of correct answers instead of one
arbitrary tie-break. Trajectory diversity in this corpus therefore comes from
exploration, not from optimal-route sampling.

The one engine quirk
--------------------
``RailSelector.step`` tests the win condition ONLY on the branch that performed a
successful slide, so an already-solved board is not detected as a win. That state
is reachable -- ``_generate`` draws tile and target positions independently, so a
level can open with every tile already on target. There, the optimal play is to
nudge the selected tile off its target and straight back (2 steps): the return
slide runs the win check. `_optimal_game_actions` handles that case explicitly;
everywhere else the metric above applies.

State is read from the LIVE sprites (``tile_r`` / ``target_r`` / ``selector``
positions, measured relative to the leftmost rail sprite so the solver does not
hard-code the playfield offset), never rebuilt from ``(seed, level)`` -- so
``supports_recovery`` is True and the base's exploration prefix / perturbation
bursts re-plan from wherever the detour left the board. Every state is solvable
(slides and selector moves are both reversible and nothing can block), so a
detour can never brick a level and RESET is never needed.

Plans are emitted in UPRIGHT (core game) space, per the `BaseSolver` rotation
contract: `drive` submits the upright action straight to the core game and the
base's ``_to_screen`` converts it to the SCREEN action when RECORDING, so the
demo pairs the rotated frame with the key a player would actually press. The
solver reads sprites, not the frame, so `plans_in_screen_space` stays False.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_rail_selector_training.py --episodes 1000 \
        --out data/training_multi_level/rail_selector
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                                    # noqa: E402
from games.rail_selector.rail_selector import CELL, RailSelector     # noqa: E402
from solvers.base_solver import BaseSolver                           # noqa: E402

SEL_UP, SEL_DOWN = GameAction.ACTION1, GameAction.ACTION2
SLIDE_L, SLIDE_R = GameAction.ACTION3, GameAction.ACTION4


# ── live state ──────────────────────────────────────────────────────────────
def _read_state(game):
    """``(sel, positions, targets, n_rails, rail_length)`` from the LIVE level.

    Columns are measured against the leftmost rail sprite rather than absolute
    pixels, so the solver is independent of the playfield's left gutter.
    """
    level = game.current_level
    rails = level.get_sprites_by_tag("rail")
    x0 = min(s.x for s in rails)
    rail_length = (max(s.x for s in rails) - x0) // CELL + 1

    def _col(sprite):
        return (sprite.x - x0) // CELL

    tiles = level.get_sprites_by_tag("tile")
    n_rails = len(tiles)
    positions = [0] * n_rails
    targets = [0] * n_rails
    for s in tiles:
        positions[int(s.name.split("_")[1])] = _col(s)
    for s in level.get_sprites_by_tag("target"):
        targets[int(s.name.split("_")[1])] = _col(s)

    sel_sprite = level.get_sprites_by_name("selector")[0]
    # Rails are laid out at y = r * RAIL_SPACING + 2 and the selector shares its
    # rail's y, so the sorted rail ordinate list inverts the mapping without
    # re-deriving the spacing constant.
    rail_ys = sorted({s.y for s in rails})
    sel = rail_ys.index(sel_sprite.y)
    return sel, positions, targets, n_rails, rail_length


# ── the exact cost-to-go ────────────────────────────────────────────────────
def _selector_cost(sel: int, wrong: list[int]) -> int:
    """Shortest walk on the rail stack from ``sel`` visiting every rail in
    ``wrong`` (non-empty): straight to the far end, or double back past the near
    end first when the selector starts between the two extremes."""
    lo, hi = wrong[0], wrong[-1]          # `wrong` is built in ascending order
    if sel <= lo:
        return hi - sel
    if sel >= hi:
        return sel - lo
    return (hi - lo) + min(sel - lo, hi - sel)


def _dist(sel: int, positions: list[int], targets: list[int]) -> int:
    """Minimum number of actions to reach the all-tiles-on-target board."""
    wrong = [r for r in range(len(positions)) if positions[r] != targets[r]]
    if not wrong:
        return 0
    return (sum(abs(positions[r] - targets[r]) for r in wrong)
            + _selector_cost(sel, wrong))


def _successor(action, sel, positions, n_rails, rail_length):
    """``(sel, positions)`` after ``action``, or None if it is a no-op (a clamped
    selector move or a slide into the rail's end wall). A no-op burns a step
    without changing the state, so it is never optimal."""
    if action is SEL_UP:
        return (sel - 1, positions) if sel > 0 else None
    if action is SEL_DOWN:
        return (sel + 1, positions) if sel < n_rails - 1 else None
    step = -1 if action is SLIDE_L else 1
    new_pos = positions[sel] + step
    if not 0 <= new_pos < rail_length:
        return None
    nxt = list(positions)
    nxt[sel] = new_pos
    return sel, nxt


def _optimal_game_actions(sel, positions, targets, n_rails, rail_length):
    """Every equally-optimal next action, in GAME space (empty == already won)."""
    if positions == targets:
        # The engine only evaluates the win condition on the branch that
        # performed a slide, so an already-solved opening board needs a nudge off
        # the target and straight back. Either legal direction costs the same 2
        # steps; a rail is at least 6 long, so at least one is always legal.
        return [a for a in (SLIDE_L, SLIDE_R)
                if _successor(a, sel, positions, n_rails, rail_length)]

    target_d = _dist(sel, positions, targets) - 1
    optimal = []
    for action in (SEL_UP, SEL_DOWN, SLIDE_L, SLIDE_R):
        succ = _successor(action, sel, positions, n_rails, rail_length)
        if succ is not None and _dist(succ[0], succ[1], targets) == target_d:
            optimal.append(action)
    return optimal


class RailSelectorSolver(BaseSolver):
    game_id = "rail_selector"
    # `solve_from` reads the live sprites and every reachable state is solvable
    # (both slides and selector moves are reversible; tiles cannot block each
    # other), so an exploratory detour is always recoverable by re-planning.
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        return RailSelector(seed=seed)

    def solve_from(self, game, level_idx: int, seed: int):
        """Greedy descent on the exact metric, from the board's LIVE state.

        The plan ends the moment the simulated board is all-on-target, which the
        engine detects because the step that got it there is necessarily a slide
        (nothing else moves a tile). The one exception is the board that is
        ALREADY all-on-target on entry -- undetected, per the module docstring --
        which is why the loop only checks for completion after acting.
        """
        sel, positions, targets, n_rails, rail_length = _read_state(game)
        plan = []
        for _ in range(self.step_guard):
            actions = _optimal_game_actions(sel, positions, targets,
                                            n_rails, rail_length)
            if not actions:                    # no legal slide on a solved board
                return []
            action = actions[0]
            plan.append(action)
            sel, positions = _successor(action, sel, positions,
                                        n_rails, rail_length)
            if positions == targets:           # that slide triggered the win check
                return plan
        return []                              # unreachable: the metric descends

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """The full optimal tie set at the live state. O(n_rails) -- no game copy,
        no search. A singleton on the optimal walk (see the module docstring), so
        this matters for the RECOVERY states exploration creates, where several
        answers really are equally correct."""
        sel, positions, targets, n_rails, rail_length = _read_state(game)
        return _optimal_game_actions(sel, positions, targets, n_rails, rail_length)


if __name__ == "__main__":
    sys.exit(RailSelectorSolver.main())
