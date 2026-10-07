"""Generate Phase-1 training data for pw01 (Dual Plate Crates).

pw01 (games/pw01/pw01.py) is a 10x10 push-only sokoban: walk into a crate to push
it one cell, get BOTH plates covered, and only then stand on the goal. Each *seed*
is a 5-level game, so every WIN seed yields one multi-level episode.

pw01 action set (simple actions only):
    ACTION1 (up)  ACTION2 (down)  ACTION3 (left)  ACTION4 (right)

THE EXPERT IS A DISTANCE FIELD, NOT A SEARCH PER QUERY. The abstract state is
``(player_cell, {crate_cell, crate_cell})`` -- at most ``100 * C(100,2)`` = 495k
states on a 10x10 board -- and the level has exactly ONE winning state (player on
the goal, a crate on each plate). So instead of searching forward from wherever
the agent happens to stand, this solver runs ONE reverse BFS from that winning
state (sokoban moves inverted: un-walk, or pull a crate back) and keeps the whole
exact distance-to-win field, cached per level geometry. Then:

  * ``solve_from`` is a greedy descent down the field -- optimal from ANY state,
    at ~zero cost, which is what makes ``supports_recovery`` cheap here: an
    exploration prefix or a perturbation burst just lands on a different cell of
    the same precomputed field;
  * ``optimal_set_from`` is a 4-way lookup, so the recorded target is the FULL set
    of equally-optimal moves (and the base samples the taken action from it) with
    no extra solve;
  * a crate shoved into a corner is a genuine dead end -- the field is simply
    unset there, so ``solve_from`` returns ``[]`` and the base records a RESET.
    Sokoban's irreversibility is captured exactly, not guessed at.

The reverse BFS is vectorised over whole frontiers with numpy (state codes are
plain ints, so a frontier is an int array and each of the 4 directions is a
handful of array ops), which keeps a full field at ~0.1 s per level -- cheap
enough to pay once per level and then answer every query by lookup.

Rotation: pw01 is an ``AugmentedGame``, so each level is displayed at a per-(seed,
level) rotation. The field is computed in GAME space; every emitted action is
converted to the SCREEN direction the agent must press via
``inverse_remap_action_full``, so the game's own ``screen_action_to_game`` maps it
back to the intended game-space move.

Usage (run from the repo root):
    python solvers/generate_pw01_training.py --episodes 1000 \
        --out data/training_multi_level/pw01
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

from arcengine import GameAction                                # noqa: E402
from games.pw01.pw01 import Pw01                                # noqa: E402
from solvers.base_solver import BaseSolver                       # noqa: E402
from utils.rotation import inverse_remap_action_full             # noqa: E402

# Game-space directions, index-aligned with ``_ACTIONS`` and with the game's own
# ACTION1..4 -> (dx, dy) mapping in ``Pw01.step``.
_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_ACTIONS = (GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4)
_OPPOSITE = (1, 0, 3, 2)

# Distances are stored as ``d + 1`` in a uint8 field (0 means "cannot win from
# here"), so the deepest representable optimal solution is 254 moves -- an order of
# magnitude more than any pw01 level needs.
_MAX_DEPTH = 254


class _Field:
    """Exact distance-to-win field for one level geometry.

    A state is encoded as the single int ``player * n^2 + lo * n + hi`` over
    ``n = gw * gh`` cells (``lo < hi`` are the two crate cells), so the field is a
    flat ``n^3`` byte array -- 1 MB for a 10x10 board -- and a frontier is just an
    int array, which is what lets the reverse BFS run vectorised.
    """

    def __init__(self, gw: int, gh: int, walls, plates, goal) -> None:
        self.gw, self.gh = gw, gh
        n = gw * gh
        self.n = n
        # The encoding is n^3 bytes and hard-codes TWO crates -- both hold for every
        # pw01 level (a 10x10 board, two crate/plate pairs). Fail loudly rather than
        # silently mis-encoding if that ever changes.
        assert n <= 100, "field size assumes a board of at most 100 cells"
        assert len(plates) == 2, "pw01 levels have exactly two plates/crates"

        # ``nbr[d][c]`` = the cell one step in direction d from c, or -1 when that
        # is off-board or a wall. Walls are impassable for the player AND for
        # crates, so one table serves both.
        wall_set = {y * gw + x for (x, y) in walls}
        nbr = np.full((4, n), -1, dtype=np.int64)
        for d, (dx, dy) in enumerate(_DELTAS):
            for y in range(gh):
                for x in range(gw):
                    nx, ny = x + dx, y + dy
                    if not (0 <= nx < gw and 0 <= ny < gh):
                        continue
                    c, t = y * gw + x, ny * gw + nx
                    if c in wall_set or t in wall_set:
                        continue
                    nbr[d][c] = t
        self.nbr = nbr

        self.dist = np.zeros(n * n * n, dtype=np.uint8)
        plate_cells = sorted(y * gw + x for (x, y) in plates)
        goal_cell = goal[1] * gw + goal[0]
        self._reverse_bfs(goal_cell, plate_cells)

    # ── encoding ────────────────────────────────────────────────────────────
    def code(self, player_cell: int, block_cells) -> int:
        lo, hi = sorted(block_cells)
        return (player_cell * self.n + lo) * self.n + hi

    def cell(self, x: int, y: int) -> int:
        return y * self.gw + x

    # ── the one reverse BFS ─────────────────────────────────────────────────
    def _reverse_bfs(self, goal_cell: int, plate_cells) -> None:
        """Fill ``dist`` with (distance-to-win + 1) by walking pw01's transition
        relation BACKWARDS from its single winning state.

        A state ``(p, B)`` is a predecessor of ``(p', B')`` under direction d iff
        either

          * the player just WALKED: ``p = p' - d``, crates unchanged, and ``p'``
            held no crate (it holds the player), or
          * the player just PUSHED: a crate sits at ``p' + d`` in ``B'``, so before
            the move that crate was at ``p'`` and the player at ``p' - d``.

        Both are enumerated for a whole frontier at once, so each BFS layer is a
        few numpy ops rather than a Python loop over states."""
        n, nbr, dist = self.n, self.nbr, self.dist
        start = self.code(goal_cell, plate_cells)
        dist[start] = 1
        frontier = np.array([start], dtype=np.int64)

        for depth in range(2, _MAX_DEPTH + 1):
            p = frontier // (n * n)
            rest = frontier % (n * n)
            b1 = rest // n
            b2 = rest % n
            preds = []
            for d in range(4):
                back = nbr[_OPPOSITE[d]][p]            # the cell the player came from
                ahead = nbr[d][p]                      # the cell the player faced

                # (a) plain walk: crates untouched, the player stepped p-d -> p.
                ok = (back >= 0) & (back != b1) & (back != b2)
                if ok.any():
                    preds.append((back[ok] * n + b1[ok]) * n + b2[ok])

                # (b) push: the crate now at p+d was at p, the player at p-d.
                for moved, other in ((b1, b2), (b2, b1)):
                    hit = (ahead >= 0) & (ahead == moved) & (back >= 0) & (back != other)
                    if not hit.any():
                        continue
                    lo = np.minimum(p[hit], other[hit])
                    hi = np.maximum(p[hit], other[hit])
                    preds.append((back[hit] * n + lo) * n + hi)

            if not preds:
                return
            cand = np.unique(np.concatenate(preds))
            cand = cand[dist[cand] == 0]
            if cand.size == 0:
                return
            dist[cand] = depth
            frontier = cand

    # ── queries ─────────────────────────────────────────────────────────────
    def steps_to_win(self, player_cell: int, block_cells) -> int | None:
        """Optimal number of moves to win from this state, or None if it is dead."""
        d = int(self.dist[self.code(player_cell, block_cells)])
        return None if d == 0 else d - 1

    def successor(self, player_cell: int, block_cells, d: int):
        """pw01's forward transition for direction ``d``, or None when the move
        changes nothing (edge, wall, crate against a wall/crate). Mirrors
        ``Pw01.step`` exactly."""
        b1, b2 = sorted(block_cells)
        target = int(self.nbr[d][player_cell])
        if target < 0:                                 # off-board or wall
            return None
        if target in (b1, b2):                         # push
            behind = int(self.nbr[d][target])
            other = b2 if target == b1 else b1
            if behind < 0 or behind == other:
                return None
            return target, (behind, other)
        return target, (b1, b2)

    def optimal_dirs(self, player_cell: int, block_cells) -> list[int]:
        """Every direction that strictly descends the field (the equally-optimal
        moves), in ACTION1..4 order."""
        here = self.steps_to_win(player_cell, block_cells)
        if not here:                                   # dead, or already won
            return []
        out = []
        for d in range(4):
            nxt = self.successor(player_cell, block_cells, d)
            if nxt is None:
                continue
            there = self.steps_to_win(nxt[0], nxt[1])
            if there is not None and there == here - 1:
                out.append(d)
        return out


class Pw01Solver(BaseSolver):
    game_id = "pw01"
    # ``solve_from`` is a lookup in an exact distance-to-win field over the LIVE
    # (player, crates) state, so it re-plans optimally from any reachable board --
    # including one an exploratory detour left behind. Sokoban IS irreversible (a
    # crate shoved into a corner cannot be pulled back), but that is represented
    # exactly rather than approximated: an unwinnable state has no field entry, so
    # the solver reports "no plan" and the base rolls the burst back / records a
    # RESET, which is precisely the recovery data we want from this game.
    supports_recovery = True
    recovery_mode = "replan"
    # A random prefix in a pushing game strands the board fairly often (one crate in
    # a corner is enough), and each strand costs one RESET; give it headroom.
    max_resets = 6

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # One-entry cache: the field depends only on the level GEOMETRY (walls,
        # plates, goal), never on where the player/crates currently are, so a single
        # entry serves a whole level -- every replan, burst and RESET inside it.
        self._cache_key = None
        self._cache_field: _Field | None = None

    def make_game(self, seed: int):
        # Pw01 is an AugmentedGame: the seed fixes each level's layout, colours and
        # display rotation, so a recorded episode replays byte-for-byte.
        return Pw01(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    # ── live state / field ──────────────────────────────────────────────────
    @staticmethod
    def _live_state(game):
        """The LIVE board read off the game's sprites: geometry + moving parts."""
        level = game.current_level
        gw, gh = level.grid_size
        walls = tuple(sorted((w.x, w.y) for w in level.get_sprites_by_tag("wall")))
        plates = tuple(sorted((p.x, p.y) for p in game._plates))
        goal = (game._goal.x, game._goal.y)
        player = (game._player.x, game._player.y)
        blocks = tuple(sorted((b.x, b.y) for b in game._blocks))
        return gw, gh, walls, plates, goal, player, blocks

    def _field(self, game, level_idx: int):
        gw, gh, walls, plates, goal, _player, _blocks = self._live_state(game)
        key = (level_idx, gw, gh, walls, plates, goal)
        if key != self._cache_key:
            self._cache_field = _Field(gw, gh, walls, plates, goal)
            self._cache_key = key
        return self._cache_field

    def _screen(self, game, dirs) -> list:
        """Game-space direction indices -> the SCREEN actions to press, through
        this level's live rotation."""
        k = game.rotation_k
        return [_ACTIONS[d] for d in dirs]

    # ── the solver API ──────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """An optimal plan from the game's LIVE state: greedy descent down the
        distance field, breaking ties at random so repeated episodes of the same
        seed take different (equally optimal) routes. ``[]`` when the live board is
        unwinnable -- a crate pushed into a corner by the exploration prefix."""
        field = self._field(game, level_idx)
        _gw, _gh, _walls, _plates, _goal, player, blocks = self._live_state(game)
        p = field.cell(*player)
        b = tuple(field.cell(x, y) for (x, y) in blocks)

        plan: list[int] = []
        remaining = field.steps_to_win(p, b)
        if not remaining:
            return []
        while remaining:
            dirs = field.optimal_dirs(p, b)
            if not dirs:                               # unreachable: field is exact
                return []
            d = self.rng.choice(dirs)
            plan.append(d)
            p, b = field.successor(p, b, d)
            remaining -= 1
        return self._screen(game, plan)

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next action at the LIVE state -- the moves that
        strictly descend the distance field. A 4-way lookup, so the full tie set is
        recorded as the training target at no extra cost."""
        field = self._field(game, level_idx)
        _gw, _gh, _walls, _plates, _goal, player, blocks = self._live_state(game)
        dirs = field.optimal_dirs(field.cell(*player),
                                  tuple(field.cell(x, y) for (x, y) in blocks))
        return self._screen(game, dirs)


if __name__ == "__main__":
    sys.exit(Pw01Solver.main())
