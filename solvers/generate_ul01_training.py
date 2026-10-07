"""Generate Phase-1 training data for ul01 ("Key & Door").

ul01 (games/ul01/ul01.py) is an 8x8 open board with a player, a key and a door:
ACTION1..4 walk one cell, walking onto the key collects it, and walking onto the
door finishes the level -- but only while carrying the key, since the locked door
is a wall. Each *seed* is a 5-level game (levels differ by how long the walk is),
so every WIN seed yields one multi-level episode.

ul01 action set (simple actions only):
    ACTION1 (up)  ACTION2 (down)  ACTION3 (left)  ACTION4 (right)

THE EXPERT IS A DISTANCE FIELD, NOT A SEARCH PER QUERY. The abstract state is
``(player_cell, has_key)`` -- 128 states on an 8x8 board -- and one reverse BFS
from the win gives the exact distance-to-win for every one of them
(``utils.keydoor_puzzle.KeyDoorField``, the same model the game itself samples its
per-seed layouts with, so the two cannot drift). Then:

  * ``solve_from`` is a greedy descent down that field -- optimal from ANY state,
    at ~zero cost;
  * ``optimal_set_from`` is a 4-way lookup, so the recorded target is the FULL set
    of equally-optimal walks (and the base samples the taken action from it) with
    no extra solve.

WHY RECOVERY IS TOTAL HERE (``supports_recovery``, "replan"). ul01 has no hazard,
no step budget and no dead end: walking is reversible, and the one irreversible
event -- collecting the key -- strictly *helps*, because a carried key turns the
door from a wall into the goal. A single blocked cell cannot disconnect a
rectangular board either, so the field is finite everywhere: from every state an
exploration prefix or a perturbation burst can reach, including one that wandered
into the key early, ``solve_from`` returns a genuinely optimal plan and the RESET
path is never needed. That is exactly the recovery signal this game has to teach --
"you are somewhere unplanned, with or without the key; here is the optimal
continuation" -- and it is the whole trajectory outside the explore/burst spans.

Rotation: ul01 is an ``AugmentedGame``, so each level is displayed at a per-(seed,
level) rotation. The field is computed in GAME space; every emitted action is
converted to the SCREEN direction the agent must press via
``inverse_remap_action_full``, so the game's own ``screen_action_to_game`` maps it
back to the intended game-space walk. (Never pin ``_rotation_k = 0`` when
recording -- plan in game space, record in screen space.)

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_ul01_training.py --episodes 1000 \
        --out data/training_multi_level/ul01
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                                # noqa: E402
from games.ul01.ul01 import Ul01                                # noqa: E402
from solvers.base_solver import BaseSolver                      # noqa: E402
from utils.keydoor_puzzle import KeyDoorField                   # noqa: E402
from utils.rotation import inverse_remap_action_full            # noqa: E402

# Index-aligned with ``keydoor_puzzle.DELTAS`` and with ul01's own ACTION1..4
# -> (dx, dy) mapping in ``Ul01.step``.
_ACTIONS = (GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4)


class Ul01Solver(BaseSolver):
    game_id = "ul01"

    #: ``solve_from`` is a lookup in an exact distance-to-win field over the LIVE
    #: ``(player, has_key)`` state, and every state of this game is winnable (see
    #: the module docstring), so an exploratory detour or a burst simply lands on a
    #: different cell of the same precomputed field and is re-planned from there.
    supports_recovery = True
    recovery_mode = "replan"
    #: Nothing in ul01 can strand or kill the agent, so a RESET should never fire;
    #: one is kept as a backstop rather than as a routine recovery path.
    max_resets = 1

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # One-entry cache. The field depends only on the level GEOMETRY (where the
        # key and the door are), never on where the player stands or whether the key
        # is carried, so a single entry serves a whole level -- every replan, burst
        # and RESET inside it. Content-addressed on that geometry, so a burst
        # rollback cannot leave it stale.
        self._cache_key = None
        self._cache_field: KeyDoorField | None = None

    def make_game(self, seed: int):
        # Ul01 is an AugmentedGame: the seed fixes each level's layout, colours and
        # display rotation, so a recorded episode replays byte-for-byte.
        return Ul01(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    # ── live state / field ──────────────────────────────────────────────────
    @staticmethod
    def _live_state(game):
        """The LIVE board: grid, the player cell, and the key/door cells that are
        still on it. ``key`` is None once the key has been collected -- which is
        also how ``has_key`` is read, since the two are the same fact (the engine
        removes the sprite as it sets the flag).

        ``door`` is None only after a winning move, i.e. only when the level is
        already over."""
        level = game.current_level
        gw, gh = level.grid_size
        player = (game._player.x, game._player.y)
        key = (game._key[0].x, game._key[0].y) if game._key else None
        door = (game._door[0].x, game._door[0].y) if game._door else None
        return gw, gh, player, key, door

    def _field(self, game, level_idx: int) -> KeyDoorField | None:
        gw, gh, _player, key, door = self._live_state(game)
        if door is None:                            # the level is already won
            return None
        cache_key = (level_idx, gw, gh, key, door)
        if cache_key != self._cache_key:
            self._cache_field = KeyDoorField(gw, gh, key, door)
            self._cache_key = cache_key
        return self._cache_field

    def _screen(self, game, dirs) -> list:
        """Game-space direction indices -> the SCREEN actions to press, through this
        level's live rotation."""
        k = game.rotation_k
        return [_ACTIONS[d] for d in dirs]

    # ── the solver API ──────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """An optimal plan from the game's LIVE state: greedy descent down the
        distance field, breaking ties at random so repeated episodes of the same
        seed walk different (equally optimal) routes."""
        field = self._field(game, level_idx)
        if field is None:
            return []
        _gw, _gh, player, key, _door = self._live_state(game)
        p, has_key = field.cell(*player), key is None

        dirs: list[int] = []
        remaining = field.steps_to_win(p, has_key)
        if not remaining:
            return []                               # unreachable (or already won)
        while remaining:
            opts = field.optimal_actions(p, has_key)
            if not opts:                            # the field is broken
                return []
            d = self.rng.choice(opts)
            dirs.append(d)
            if field.wins(p, has_key, d):
                break
            p, has_key = field.step(p, has_key, d)
            remaining = field.steps_to_win(p, has_key)
        return self._screen(game, dirs)

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """STOCHASTIC OPTIMAL: every walk that drops steps-to-win by exactly one.

        Away from the door that is usually two of the four directions (any
        monotone step of an L-shaped walk), so a seed's episodes spread over many
        distinct optimal routes rather than replaying one canonical zig-zag."""
        field = self._field(game, level_idx)
        if field is None:
            return None
        _gw, _gh, player, key, _door = self._live_state(game)
        dirs = field.optimal_actions(field.cell(*player), key is None)
        return self._screen(game, dirs) or None


if __name__ == "__main__":
    sys.exit(Ul01Solver.main())
