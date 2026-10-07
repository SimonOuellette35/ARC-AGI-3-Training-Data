"""Generate Phase-1 training data for the FT09 game (games/ft09/ft09.py).

FT09 is a *mouse-click* colour-constraint puzzle (a "tile-dyeing" variant of
Lights-Out). Each level is a sparse grid of coloured tiles -- ``Hkx`` blocks and
``NTi`` "hub" tiles -- interleaved with fixed ``bsT`` *clue* sprites. Every clue
sits between tiles; its centre pixel is a TARGET colour and each of its 8 grid
neighbours carries a requirement: MATCH (clue pixel ``0``) means the tile in that
direction must equal the target colour, NO-MATCH (non-zero) means it must differ.
The level is won (``cgj``) once every requirement of every clue holds. The only
action is an ACTION6 click:

  * clicking an ``Hkx`` tile cycles ITS OWN colour by one palette step;
  * clicking an ``NTi`` tile cycles itself PLUS the tiles under its ``6``-pattern
    (the plain ``NTi`` is a 4-neighbour plus; the ``ZkU`` variant is up-only).

All tiles start on ``gqb[0]`` and cycle within the per-level palette ``gqb``
(length 2 or 3). A non-winning move costs one budget point (``kCv``).

Solver
------
Because each click adds ``+1 (mod L)`` to a FIXED set of tiles, a whole solution
is one vector of click-counts and the order is irrelevant -- the reachable colour
states form the column space of an effect matrix ``A`` over ``GF(L)`` (``L`` is 2
or 3, both prime). ``_plan_level`` reads the palette + tiles + clue-derived allowed
colour sets, builds an equation per singleton-allowed tile, solves over ``GF(L)``
by Gaussian elimination (particular + null-space basis), and samples null-space
offsets until every free tile is legal. It returns tile positions to click;
``solve_from`` maps each to a screen click. Only an engine-verified win is
recorded, so an unsolvable seed is simply skipped.

Determinism / augmentation
--------------------------
``Ft09(seed=seed)`` draws the palette + clue layout from the global ``random``
module inside ``on_set_level``; this generator seeds ``random`` once with
``ft09:<seed>`` (in ``make_game``) and plays the WHOLE game on a SINGLE instance,
letting the engine's ``next_level`` advance (via ``set_level`` calling
``_really_set_next_level``) so each ``on_set_level`` runs exactly once (a fresh
``set_level`` would re-apply the colour remap and corrupt the board).

Display rotation is NOT pinned: ``Ft09`` is an ``AugmentedGame`` whose per-level
rotation is ``random_rotation_k(seed, level)`` -- the orientation live play shows.
Rotation is a pure DISPLAY transform, so the plan is unaffected; only the *clicks*
move, which ``_grid_to_display`` does by inverse-rotating each cell centre.

Recovery is REPLAN: ``_plan_level`` reads each tile's CURRENT colour and re-bases
the per-tile requirement to ``allowed - current (mod L)``, so the GF(L) solve
yields a valid click vector from any perturbed-but-solvable state (not just the
clean ``gqb[0]`` initial board). At the initial state every current colour is 0,
so this reduces to the original baseline.

Action schema (mixed simple + mouse, matching cn04 / cd82 / bp35)
    RESET / simple :  {"type": "simple", "index": k}
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_ft09_training.py --episodes 1000 \
        --out data/training_multi_level/ft09
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.ft09.ft09 import Ft09                            # noqa: E402
from solvers.base_solver import BaseSolver                  # noqa: E402
from utils.explore import Action, CLICK_ACTION              # noqa: E402
from utils.rotation import remap_click                      # noqa: E402

# 3x3 local-neighbour offsets, indexed [row][col], matching the engine's ``GBS``.
_GBS = [
    [(-1, -1), (0, -1), (1, -1)],
    [(-1, 0), (0, 0), (1, 0)],
    [(-1, 1), (0, 1), (1, 1)],
]


# ── State reading ───────────────────────────────────────────────────────────
def _gather(game):
    """(level, palette, L, {(x,y): (kind, sprite)}) for the current level."""
    level = game.current_level
    palette = list(game.gqb)
    tiles = {}
    for s in level.get_sprites_by_tag("Hkx"):
        tiles[(s.x, s.y)] = ("Hkx", s)
    for s in level.get_sprites_by_tag("NTi"):
        tiles[(s.x, s.y)] = ("NTi", s)
    return level, palette, len(palette), tiles


def _click_mask(kind, sprite):
    """Grid offsets (dx, dy) that a click on this tile increments by +1 (mod L).

    Hkx cycles only itself. NTi seeds its effect with the centre and adds every
    ``6``-pattern cell, so it cycles itself PLUS those neighbours."""
    if kind == "Hkx":
        return [(0, 0)]
    offs = [(0, 0)]
    for r in range(3):
        for c in range(3):
            if sprite.pixels[r][c] == 6:
                dx, dy = _GBS[r][c]
                offs.append((dx * 4, dy * 4))
    return offs


# ── GF(p) linear algebra ────────────────────────────────────────────────────
def _solve_gfp(rows, rhs, p, n):
    """Solve ``A x = b`` over GF(p) with ``n`` variables. Returns
    ``(particular, null_basis)`` or ``None`` if inconsistent."""
    m = len(rows)
    M = [rows[i][:] + [rhs[i]] for i in range(m)]
    pivot_col = {}
    r = 0
    for c in range(n):
        piv = next((rr for rr in range(r, m) if M[rr][c] % p != 0), None)
        if piv is None:
            continue
        M[r], M[piv] = M[piv], M[r]
        inv = pow(M[r][c], p - 2, p)
        M[r] = [(v * inv) % p for v in M[r]]
        for rr in range(m):
            if rr != r and M[rr][c] % p != 0:
                f = M[rr][c]
                M[rr] = [(M[rr][k] - f * M[r][k]) % p for k in range(n + 1)]
        pivot_col[r] = c
        r += 1
        if r == m:
            break
    for rr in range(r, m):
        if M[rr][n] % p != 0:
            return None
    pivots = set(pivot_col.values())
    x = [0] * n
    for row, c in pivot_col.items():
        x[c] = M[row][n] % p
    basis = []
    for fc in (c for c in range(n) if c not in pivots):
        v = [0] * n
        v[fc] = 1
        for row, c in pivot_col.items():
            v[c] = (-M[row][fc]) % p
        basis.append(v)
    return x, basis


# ── Planning ────────────────────────────────────────────────────────────────
def _plan_level(game):
    """A list of tile positions to click (in order) that solves the current level,
    or ``None`` if no click vector satisfies the clues."""
    level, palette, L, tiles = _gather(game)
    tile_list = list(tiles.items())
    idx_of = {pos: i for i, (pos, _) in enumerate(tile_list)}
    N = len(tile_list)
    if N == 0:
        return []

    # RE-BASE on the LIVE state (replan): read each tile's CURRENT palette-index
    # colour from its centre pixel (the engine reads it the same way,
    # ``gqb.index(pixels[1][1])``). At a clean initial state every ``cur`` is 0
    # (all tiles on ``gqb[0]``) and this reduces to the old baseline; from a
    # perturbed state the per-tile delta re-bases to ``allowed - cur (mod L)``.
    cur = [palette.index(s.pixels[1][1]) if s.pixels[1][1] in palette else 0
           for (_pos, (_kind, s)) in tile_list]

    allowed = [list(range(L)) for _ in range(N)]
    forced = {}
    for etf in level.get_sprites_by_tag("bsT"):
        target = etf.pixels[1][1]
        for r in range(3):
            for c in range(3):
                if r == 1 and c == 1:
                    continue
                dx, dy = _GBS[r][c]
                pos = (etf.x + dx * 4, etf.y + dy * 4)
                if pos not in idx_of:
                    continue
                ti = idx_of[pos]
                if etf.pixels[r][c] == 0:  # MATCH -> pin to target colour
                    if target not in palette:
                        return None
                    if ti in forced and forced[ti] != target:
                        return None
                    forced[ti] = target
                else:  # NO-MATCH -> remove the target colour
                    if target in palette:
                        allowed[ti] = [k for k in allowed[ti]
                                       if palette[k] != target]
    for ti, target in forced.items():
        allowed[ti] = [palette.index(target)]
    if any(len(a) == 0 for a in allowed):
        return None

    A = [[0] * N for _ in range(N)]
    for j, (pos, (kind, s)) in enumerate(tile_list):
        for (dx, dy) in _click_mask(kind, s):
            npos = (pos[0] + dx, pos[1] + dy)
            if npos in idx_of:
                t = idx_of[npos]
                A[t][j] = (A[t][j] + 1) % L

    rows, rhs = [], []
    for i in range(N):
        if len(allowed[i]) == 1:
            rows.append(A[i][:])
            # required TOTAL click-delta on tile i = (allowed - current) mod L,
            # so the clicks carry the tile from its LIVE colour to the target.
            rhs.append((allowed[i][0] - cur[i]) % L)
    sol = _solve_gfp(rows, rhs, L, N)
    if sol is None:
        return None
    x0, basis = sol

    def resulting_colours(c):
        # live baseline + accumulated click-delta, matching the RHS re-base above.
        return [(cur[t] + sum(A[t][j] * c[j] for j in range(N))) % L
                for t in range(N)]

    def legal(c):
        col = resulting_colours(c)
        return all(col[i] in allowed[i] for i in range(N))

    chosen = None
    if legal(x0):
        chosen = x0
    elif basis:
        rng = random.Random(0xF709 + N)  # private RNG -> no effect on game state
        for _ in range(6000):
            c = x0[:]
            for b in basis:
                k = rng.randrange(L)
                if k:
                    for t in range(N):
                        c[t] = (c[t] + k * b[t]) % L
            if legal(c):
                chosen = c
                break
    if chosen is None:
        return None

    plan = []
    for j, (pos, _) in enumerate(tile_list):
        plan.extend([pos] * (chosen[j] % L))
    return plan


class Ft09Solver(BaseSolver):
    game_id = "ft09"
    step_guard = 6000
    # REPLAN-recovery: ``_plan_level`` re-bases the GF(L) RHS on each tile's LIVE
    # colour (``allowed - current mod L``), so it computes a valid click vector
    # from ANY perturbed-but-solvable state -- no reset to the initial board is
    # needed, and burst-strewn states re-plan cleanly.
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        # Seed the global stream BEFORE construction so level 0's on_set_level
        # palette/clue layout is reproducible, exactly as the original did.
        random.seed(f"{self.game_id}:{seed}")
        return Ft09(seed=seed)

    def set_level(self, game, level_idx: int) -> None:
        """Single-instance advance: level 0 is set up by the constructor; later
        levels advance via the engine's own ``next_level`` so each ``on_set_level``
        (palette + clue remap) runs exactly once."""
        if level_idx == 0:
            return
        if game._next_level:
            game._really_set_next_level()

    def reset_level(self, game, level_idx: int, seed: int) -> None:
        """Restore ``level_idx`` to its INITIAL state (the frame ``record_level``
        captured as ``observations[0]``). ``set_level`` ADVANCES the single engine
        instance in place, so the base ``reset_level`` cannot restore it. Instead
        rebuild the level the same way it was first recorded: a fresh ``make_game``
        (which re-seeds the global stream) advanced to ``level_idx`` through the
        engine's own in-place ``_really_set_next_level`` -- running each level's
        ``on_set_level`` exactly once, in order, so the palette/clue roll is
        reproduced -- then transplant that fresh state into ``game``."""
        fresh = self.make_game(seed)
        for _ in range(level_idx):
            fresh._really_set_next_level()
        game.__dict__.update(fresh.__dict__)

    def available_actions(self, game) -> list[int]:
        return [CLICK_ACTION]

    def _grid_to_display(self, game, gx: int, gy: int) -> Action:
        """Centre-of-cell SCREEN click Action for grid cell (gx, gy). The game
        maps a click back with ``remap_click(x, y, k)`` (a rotation), so its
        inverse is ``remap_click(..., -k)`` (identity at k=0)."""
        scale, offx, offy = game.camera._calculate_scale_and_offset()
        gpx = int(gx * scale + offx + scale // 2)
        gpy = int(gy * scale + offy + scale // 2)
        sx, sy = gpx, gpy
        return Action(CLICK_ACTION, (sy, sx))              # click_rc = (row=y, col=x)

    def solve_from(self, game, level_idx: int, seed: int):
        """One click per unit of each tile's GF(L) click-count. Order is
        irrelevant; the engine wins the instant the last requirement is met."""
        plan = _plan_level(game)
        if plan is None:
            return []
        return [self._grid_to_display(game, pos[0] + 1, pos[1] + 1) for pos in plan]


if __name__ == "__main__":
    sys.exit(Ft09Solver.main())
