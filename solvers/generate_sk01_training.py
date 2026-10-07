"""Generate Phase-1 training data for sk01 (Sokoban Push).

sk01 (games/sk01/sk01.py) is a push-only sokoban on an 8x8 (or 12x12) board: walk
into a block to shove it one cell, and cover every target pad at once. Each *seed*
is a 5-level game, so every WIN seed yields one multi-level episode.

sk01 action set (simple actions only):
    ACTION1 (up)  ACTION2 (down)  ACTION3 (left)  ACTION4 (right)

THE EXPERT IS A DISTANCE FIELD, NOT A SEARCH PER QUERY -- the same shape as the
pw01 generator, generalised to k blocks. The abstract state is ``(player_cell,
{block_cell, ...})`` and a level has exactly ONE winning block configuration (the
blocks are interchangeable, so "every pad covered" is the single set equality
``blocks == targets``), with the player free to be on any other cell. So instead of
searching forward from wherever the agent happens to stand, this solver runs ONE
reverse BFS from that winning configuration (sokoban moves inverted: un-walk, or
pull a block back) and keeps the whole exact distance-to-win field, cached per level
geometry. Then:

  * ``solve_from`` is a greedy descent down the field -- optimal from ANY state, at
    ~zero cost, which is what makes ``supports_recovery`` cheap here: an exploration
    prefix or a perturbation burst just lands on a different cell of the same
    precomputed field;
  * ``optimal_set_from`` is a 4-way lookup, so the recorded target is the FULL set of
    equally-optimal moves (and the base samples the taken action from it) with no
    extra solve;
  * a block shoved into a corner is a genuine dead end -- the field is simply unset
    there, so ``solve_from`` returns ``[]`` and the base rolls the burst back or
    records a RESET. Sokoban's irreversibility is captured exactly, not guessed at.

The reverse BFS is vectorised over whole frontiers with numpy (state codes are plain
ints, so a frontier is an int array and each of the 4 directions is a handful of
array ops), which keeps a full field at well under a second even for the 3-block
level -- cheap enough to pay once per level and then answer every query by lookup.

Rotation: sk01 is an ``AugmentedGame``, so each level is displayed at a per-(seed,
level) rotation. The field is computed in GAME space; every emitted action is
converted to the SCREEN direction the agent must press via
``inverse_remap_action_full``, so the game's own ``screen_action_to_game`` maps it
back to the intended game-space move.

Usage (run from the repo root):
    python solvers/generate_sk01_training.py --episodes 1000 \
        --out data/training_multi_level/sk01
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402

from arcengine import GameAction                                # noqa: E402
from games.sk01.sk01 import Sk01                                # noqa: E402
from solvers.base_solver import BaseSolver                      # noqa: E402
from utils.rotation import inverse_remap_action_full            # noqa: E402

# Game-space directions, index-aligned with ``_ACTIONS`` and with the game's own
# ACTION1..4 -> (dx, dy) mapping in ``Sk01.step``.
_DELTAS = ((0, -1), (0, 1), (-1, 0), (1, 0))
_ACTIONS = (GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4)
_OPPOSITE = (1, 0, 3, 2)

# Distances are stored as ``d + 1`` in a uint8 field (0 means "cannot win from
# here"), so the deepest representable optimal solution is 254 moves -- several
# times what any sk01 level needs (the stock levels peak at 63).
_MAX_DEPTH = 254

# Guard on the flat field size ``n**(k+1)`` bytes. sk01's levels are 8x8 with up to
# 3 blocks (64**4 = 16 MB) and 12x12 with 2 (144**3 = 3 MB); a 12x12 board with 3
# blocks would want 430 MB, so fail loudly rather than swap the machine to death if
# the level table ever grows that way.
_MAX_FIELD = 32_000_000


class _Field:
    """Exact distance-to-win field for one level geometry.

    A state is encoded as the single int ``((player * n + b_0) * n + b_1) ...`` over
    ``n = gw * gh`` cells with the block cells sorted ascending (blocks are
    interchangeable), so the field is a flat ``n**(k+1)`` byte array and a frontier
    is just an int array -- which is what lets the reverse BFS run vectorised.
    """

    def __init__(self, gw: int, gh: int, walls, targets) -> None:
        self.gw, self.gh = gw, gh
        n = gw * gh
        self.n = n
        self.k = len(targets)
        assert self.k >= 1, "a level needs at least one block"
        assert n ** (self.k + 1) <= _MAX_FIELD, (
            f"field for a {gw}x{gh} board with {self.k} blocks would need "
            f"{n ** (self.k + 1)} bytes")

        # ``nbr[d][c]`` = the cell one step in direction d from c, or -1 when that is
        # off-board or a wall. Walls are impassable for the player AND for blocks, so
        # one table serves both. (Target pads are not collidable and never appear
        # here -- the player and blocks walk over them freely.)
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

        self.dist = np.zeros(n ** (self.k + 1), dtype=np.uint8)
        target_cells = tuple(sorted(y * gw + x for (x, y) in targets))
        free = [c for c in range(n) if c not in wall_set]
        self._reverse_bfs(target_cells, free)

    # ── encoding ────────────────────────────────────────────────────────────
    def code(self, player_cell: int, block_cells) -> int:
        c = int(player_cell)
        for b in sorted(block_cells):
            c = c * self.n + int(b)
        return c

    def cell(self, x: int, y: int) -> int:
        return y * self.gw + x

    def _encode(self, player, blocks) -> np.ndarray:
        """Vectorised `code` -- ``blocks`` is a (k, m) array, already sorted."""
        c = player
        for i in range(self.k):
            c = c * self.n + blocks[i]
        return c

    def _decode(self, codes: np.ndarray):
        """Vectorised inverse of `_encode`: ``(player, (k, m) block array)``."""
        rest = codes
        blocks = np.empty((self.k, codes.size), dtype=np.int64)
        for i in range(self.k - 1, -1, -1):
            blocks[i] = rest % self.n
            rest = rest // self.n
        return rest, blocks

    # ── the one reverse BFS ─────────────────────────────────────────────────
    def _reverse_bfs(self, target_cells, free_cells) -> None:
        """Fill ``dist`` with (distance-to-win + 1) by walking sk01's transition
        relation BACKWARDS from its winning configuration.

        The level is won as soon as the blocks cover the pads, wherever the player
        happens to stand, so the BFS is seeded with EVERY state whose block set is
        the target set (one per legal player cell) rather than a single state.

        A state ``(p, B)`` is a predecessor of ``(p', B')`` under direction d iff
        either

          * the player just WALKED: ``p = p' - d``, blocks unchanged, and ``p'`` held
            no block (it holds the player), or
          * the player just PUSHED: a block sits at ``p' + d`` in ``B'``, so before
            the move that block was at ``p'`` and the player at ``p' - d``.

        Both are enumerated for a whole frontier at once, so each BFS layer is a few
        numpy ops rather than a Python loop over states."""
        n, nbr, dist, k = self.n, self.nbr, self.dist, self.k
        tset = set(target_cells)
        seeds = [self.code(c, target_cells) for c in free_cells if c not in tset]
        if not seeds:
            return
        frontier = np.array(sorted(seeds), dtype=np.int64)
        dist[frontier] = 1

        for depth in range(2, _MAX_DEPTH + 1):
            p, bs = self._decode(frontier)
            preds = []
            for d in range(4):
                back = nbr[_OPPOSITE[d]][p]           # the cell the player came from
                ahead = nbr[d][p]                     # the cell the player faced

                # (a) plain walk: blocks untouched, the player stepped p-d -> p.
                ok = back >= 0
                for i in range(k):
                    ok &= back != bs[i]
                if ok.any():
                    preds.append(self._encode(back[ok], bs[:, ok]))

                # (b) push: the block now at p+d was at p, the player at p-d.
                for i in range(k):
                    hit = (ahead >= 0) & (ahead == bs[i]) & (back >= 0)
                    for j in range(k):
                        if j != i:
                            hit &= back != bs[j]
                    if not hit.any():
                        continue
                    moved = bs[:, hit].copy()
                    moved[i] = p[hit]                 # the block was where p is now
                    moved.sort(axis=0)                # blocks are interchangeable
                    preds.append(self._encode(back[hit], moved))

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
        """sk01's forward transition for direction ``d``, or None when the move
        changes nothing (edge, wall, block against a wall/block). Mirrors
        ``Sk01.step`` exactly."""
        blocks = tuple(sorted(block_cells))
        target = int(self.nbr[d][player_cell])
        if target < 0:                                 # off-board or wall
            return None
        if target in blocks:                           # push
            behind = int(self.nbr[d][target])
            if behind < 0 or behind in blocks:
                return None
            moved = tuple(sorted([c for c in blocks if c != target] + [behind]))
            return target, moved
        return target, blocks

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


class Sk01Solver(BaseSolver):
    game_id = "sk01"
    # ``solve_from`` is a lookup in an exact distance-to-win field over the LIVE
    # (player, blocks) state, so it re-plans optimally from any reachable board --
    # including one an exploratory detour left behind. Sokoban IS irreversible (a
    # block shoved into a corner cannot be pulled back), but that is represented
    # exactly rather than approximated: an unwinnable state has no field entry, so
    # the solver reports "no plan" and the base rolls the burst back / records a
    # RESET, which is precisely the recovery data we want from this game.
    supports_recovery = True
    recovery_mode = "replan"
    # A random prefix in a pushing game strands the board fairly often (one block in
    # a corner is enough), and each strand costs one RESET; give it headroom.
    max_resets = 6

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # One-entry cache: the field depends only on the level GEOMETRY (walls,
        # targets), never on where the player/blocks currently are, so a single entry
        # serves a whole level -- every replan, burst and RESET inside it.
        self._cache_key = None
        self._cache_field: _Field | None = None

    def make_game(self, seed: int):
        # Sk01 is an AugmentedGame: the seed fixes each level's layout, colours and
        # display rotation, so a recorded episode replays byte-for-byte.
        return Sk01(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    # ── live state / field ──────────────────────────────────────────────────
    @staticmethod
    def _live_state(game):
        """The LIVE board read off the game's sprites: geometry + moving parts."""
        level = game.current_level
        gw, gh = level.grid_size
        walls = tuple(sorted((w.x, w.y) for w in level.get_sprites_by_tag("wall")))
        targets = tuple(sorted((t.x, t.y) for t in game._targets))
        player = (game._player.x, game._player.y)
        blocks = tuple(sorted((b.x, b.y) for b in game._blocks))
        return gw, gh, walls, targets, player, blocks

    def _field(self, game, level_idx: int):
        gw, gh, walls, targets, _player, _blocks = self._live_state(game)
        key = (level_idx, gw, gh, walls, targets)
        if key != self._cache_key:
            self._cache_field = _Field(gw, gh, walls, targets)
            self._cache_key = key
        return self._cache_field

    def _screen(self, game, dirs) -> list:
        """Game-space direction indices -> the SCREEN actions to press, through this
        level's live rotation."""
        k = game.rotation_k
        return [_ACTIONS[d] for d in dirs]

    # ── the solver API ──────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """An optimal plan from the game's LIVE state: greedy descent down the
        distance field, breaking ties at random so repeated episodes of the same seed
        take different (equally optimal) routes. ``[]`` when the live board is
        unwinnable -- a block pushed into a corner by the exploration prefix."""
        field = self._field(game, level_idx)
        _gw, _gh, _walls, _targets, player, blocks = self._live_state(game)
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
        _gw, _gh, _walls, _targets, player, blocks = self._live_state(game)
        dirs = field.optimal_dirs(field.cell(*player),
                                  tuple(field.cell(x, y) for (x, y) in blocks))
        return self._screen(game, dirs)


if __name__ == "__main__":
    sys.exit(Sk01Solver.main())
