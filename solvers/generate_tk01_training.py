"""Generate Phase-1 training data for tk01 (Telekinetic Tug).

tk01 (games/tk01/tk01.py) is a 10x10 board with one player, one block, one goal
and some walls: walking into the block PUSHES it one cell further, ACTION5 TUGS it
one cell towards the player, and the level ends when a WALK leaves the block on
the goal. Each *seed* is a 5-level game, so every WIN seed yields one multi-level
episode.

tk01 action set (simple actions only):
    ACTION1 (up)  ACTION2 (down)  ACTION3 (left)  ACTION4 (right)  ACTION5 (tug)

THE EXPERT IS A DISTANCE FIELD, NOT A SEARCH PER QUERY. The state is just
``(player_cell, block_cell)`` -- at most 100*100 = 10k states -- so
``utils.tug_puzzle.TugField`` precomputes every successor and runs ONE reverse BFS
from the winning transitions, giving the exact number of actions to the win for
EVERY state at ~10 ms per level (the same field the game itself uses to sample a
solvable board on the right difficulty rung). Then:

  * ``solve_from`` is a greedy descent down that field, so it is optimal from ANY
    state at ~zero cost -- which is what makes ``supports_recovery`` cheap here: an
    exploration prefix or a perturbation burst simply lands on a different cell of
    the same precomputed field, and the plan for it is a lookup;
  * ``optimal_set_from`` is a 5-way lookup, so the recorded target is the FULL set
    of equally-optimal actions (and the base samples the taken action from it) at
    no extra cost;
  * a block walled into a pocket no tug can reach out of is a genuine dead end --
    the field is simply unset there, so ``solve_from`` returns ``[]`` and the base
    rolls the burst back or records a RESET.

WHY RECOVERY IS EASY HERE. Push and tug are inverses (shove the block away, drag
it back), so unlike sokoban almost every detour is undoable and the field is
defined nearly everywhere -- an exploratory burst that scatters the board costs a
few extra actions, not the episode. The field is exact either way, so the rare
genuinely-dead board is still reported as dead rather than guessed at.

THE WIN IS A TRANSITION. The engine only advances the level after ACTION1..4, so
tugging the block onto the goal does not finish the level; the descent naturally
ends on the directional press that does (and never on the one press that would
shove the block back off the goal). See utils/tug_puzzle.py.

Rotation: tk01 is an ``AugmentedGame``, so each level is displayed at a per-(seed,
level) rotation. The field is computed in GAME space; every emitted direction is
converted to the SCREEN press the agent must make via
``inverse_remap_action_full``, so the game's own ``screen_action_to_game`` maps it
back to the intended game-space move. ACTION5 is not directional and is emitted
as-is.

Usage (run from the repo root):
    python solvers/generate_tk01_training.py --episodes 1000 \
        --out data/training_multi_level/tk01
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                                  # noqa: E402
from games.tk01.tk01 import Tk01                                  # noqa: E402
from solvers.base_solver import BaseSolver                        # noqa: E402
from utils.rotation import inverse_remap_action_full              # noqa: E402
from utils.tug_puzzle import TUG, TugField                        # noqa: E402

# Index-aligned with the field's action indices (0..3 walk, 4 tug).
_ACTIONS = (GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4, GameAction.ACTION5)


class Tk01Solver(BaseSolver):
    game_id = "tk01"
    # ``solve_from`` is a lookup in an exact distance-to-win field over the LIVE
    # (player, block) state, so it re-plans optimally from any board -- including
    # one an exploratory detour or a perturbation burst left behind.
    supports_recovery = True
    recovery_mode = "replan"
    max_resets = 4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # One-entry cache: the field depends only on the level GEOMETRY (walls,
        # goal), never on where the player/block currently are, so a single entry
        # serves a whole level -- every replan, burst and RESET inside it.
        self._cache_key = None
        self._cache_field: TugField | None = None

    def make_game(self, seed: int):
        # Tk01 is an AugmentedGame: the seed fixes each level's layout, colours and
        # display rotation, so a recorded episode replays byte-for-byte.
        return Tk01(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, 5]

    # ── live state / field ──────────────────────────────────────────────────
    @staticmethod
    def _live_state(game):
        """The LIVE board read off the game's sprites: geometry + moving parts."""
        level = game.current_level
        gw, gh = level.grid_size
        walls = tuple(sorted((w.x, w.y) for w in level.get_sprites_by_tag("wall")))
        goal = (game._g.x, game._g.y)
        player = (game._p.x, game._p.y)
        block = (game._blocks[0].x, game._blocks[0].y)
        return gw, gh, walls, goal, player, block

    def _field(self, game, level_idx: int) -> TugField:
        gw, gh, walls, goal, _player, _block = self._live_state(game)
        key = (level_idx, gw, gh, walls, goal)
        if key != self._cache_key:
            self._cache_field = TugField(gw, gh, walls, goal)
            self._cache_key = key
        return self._cache_field

    @staticmethod
    def _screen(game, actions) -> list:
        """Field action indices -> the SCREEN actions to press, through this
        level's live rotation. ACTION5 is not directional, so it passes through."""
        k = game.rotation_k
        return [_ACTIONS[TUG] if a == TUG
                else _ACTIONS[a] for a in actions]

    # ── the solver API ──────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """An optimal plan from the game's LIVE state: greedy descent down the
        distance field, breaking ties at random so repeated episodes of the same
        seed take different (equally optimal) routes. ``[]`` when the live board is
        unwinnable."""
        field = self._field(game, level_idx)
        _gw, _gh, _walls, _goal, player, block = self._live_state(game)
        p, b = field.cell(*player), field.cell(*block)

        remaining = field.steps_to_win(p, b)
        if not remaining:
            return []
        plan = []
        while remaining:
            actions = field.optimal_actions(p, b)
            if not actions:                        # unreachable: the field is exact
                return []
            a = self.rng.choice(actions)
            plan.append(a)
            if field.wins(p, b, a):                # this press ends the level
                break
            p, b = field.step(p, b, a)
            remaining -= 1
        return self._screen(game, plan)

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next action at the LIVE state -- the actions that
        strictly descend the distance field (and, one step from the end, exactly
        the presses that win). A 5-way lookup, so the full tie set is recorded as
        the training target at no extra cost."""
        field = self._field(game, level_idx)
        _gw, _gh, _walls, _goal, player, block = self._live_state(game)
        actions = field.optimal_actions(field.cell(*player), field.cell(*block))
        return self._screen(game, actions)


if __name__ == "__main__":
    sys.exit(Tk01Solver.main())
