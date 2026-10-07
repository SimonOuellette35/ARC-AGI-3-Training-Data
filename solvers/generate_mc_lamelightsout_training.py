"""Generate Phase-1 training data for mc:lamelightsout (games/mc:lamelightsout/).

The game is classic **Lights Out** driven by the mouse: every cell of a
``cols x rows`` board is a light that is ON (white, colour 0) or OFF (grey,
colour 2); an ACTION6 click toggles the clicked cell *and its four orthogonal
neighbours* (the "plus" stamp, clipped at the border); the level is won the
instant no OFF light remains. ACTION1-5 do nothing -- clicks are the whole
action space.

Solver
------
Clicking is an involution (two clicks on the same cell cancel) and the toggle
stamps commute, so a whole solution is just a SET of cells and its order is
irrelevant. Over GF(2) the board is a linear system::

    A x = b,   A[i][j] = 1 iff clicking cell j toggles cell i,
               b[i]    = 1 iff cell i is currently OFF (needs an odd toggle count)

`_solutions` solves it by Gaussian elimination and enumerates the whole solution
coset ``x0 + span(nullspace)``; the boards here are at most 4x4 = 16 cells, so
that enumeration is exhaustive and the answer is EXACTLY minimal, not greedy:

  * ``dist(state)``        = the minimum number of clicks that wins,
  * ``optimal_set(state)`` = the union of the supports of every minimum-weight
    solution -- i.e. every click that reduces ``dist`` by one.

Both come from one solve, so `optimal_set_from` is essentially free and the
recorded target is the FULL optimal tie set rather than one arbitrary
tie-break. With the base's stochastic-optimal sampling that turns each board
into many distinct, still perfectly-minimal click orders.

Recovery (REPLAN)
-----------------
`solve_from` reads the LIVE board off the level's light sprites and re-solves
from scratch, and Lights Out is fully reversible (any click is undone by
repeating it), so *every* reachable state is still solvable: an exploration
prefix or a perturbation burst can click anywhere and the very next expert step
re-plans optimally from wherever it landed. No RESET is needed.

Rotation
--------
``McLamelightsout`` is an ``AugmentedGame``: each level is displayed at
``random_rotation_k(seed, level)`` and maps clicks back with
``remap_click(x, y, k)``. Rotation is a pure display transform, so the plan is
unaffected -- only the emitted click pixels move, which `_cell_click` handles by
inverse-rotating each cell centre. Nothing is pinned to k=0.

Action schema (mixed simple + mouse, matching cn04 / ft09 / cd82)
    RESET / simple :  {"type": "simple", "index": k}
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_mc_lamelightsout_training.py --episodes 1000 \
        --out data/training_multi_level/mc_lamelightsout
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from solvers.base_solver import BaseSolver              # noqa: E402
from utils.explore import Action, CLICK_ACTION          # noqa: E402
from utils.rotation import remap_click                  # noqa: E402

# The game id carries a ':' namespace, so the module cannot be imported by name
# (``games.mc:lamelightsout`` is not a legal identifier) -- load it by path, the
# same way ``game_envs._load_local_env`` does for live play.
_GAME_ID = "mc:lamelightsout"
_GAME_PATH = _REPO_ROOT / "games" / _GAME_ID / f"{_GAME_ID}.py"
_spec = importlib.util.spec_from_file_location("mc_lamelightsout_game", _GAME_PATH)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
McLamelightsout = _module.McLamelightsout

# The engine's toggle stamp: the clicked cell plus its four orthogonal neighbours.
_STAMP = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


# ── reading the live board ──────────────────────────────────────────────────
def _read_board(game) -> tuple[int, int, list[int]]:
    """``(cols, rows, off_flags)`` for the level as it stands RIGHT NOW.

    Read off the level's ``light`` sprites (an ON light is tagged ``on``), not
    from any cached solver state -- that is what makes `solve_from` valid from an
    arbitrary perturbed state. ``off_flags`` is row-major, 1 = the light is OFF
    and therefore needs an odd number of toggles."""
    cols, rows = game.current_level.grid_size
    off = [1] * (cols * rows)
    for sp in game.current_level.get_sprites_by_tag("light"):
        if 0 <= sp.x < cols and 0 <= sp.y < rows:
            off[sp.y * cols + sp.x] = 0 if "on" in sp.tags else 1
    return cols, rows, off


# ── GF(2) solve ─────────────────────────────────────────────────────────────
def _stamp_masks(cols: int, rows: int) -> list[int]:
    """For each cell j, the bitmask of the cells one click on j toggles."""
    masks = []
    for j in range(cols * rows):
        x, y = j % cols, j // cols
        m = 0
        for dx, dy in _STAMP:
            nx, ny = x + dx, y + dy
            if 0 <= nx < cols and 0 <= ny < rows:
                m |= 1 << (ny * cols + nx)
        masks.append(m)
    return masks


def _solutions(cols: int, rows: int, off: list[int]) -> list[int]:
    """Every click SET (as a cell bitmask) that turns the board all-ON.

    Solves ``A x = b`` over GF(2) with the columns of ``A`` packed as ints
    (column j is the stamp of cell j, so ``A`` is symmetric) and returns the
    complete coset ``x0 + span(null basis)``. The coset is enumerated in full,
    which is what makes the minimum below exact; boards are <= 16 cells here, so
    at most 2**16 candidates and in practice a handful. Empty if unsolvable."""
    n = cols * rows
    masks = _stamp_masks(cols, rows)
    b = 0
    for i, v in enumerate(off):
        if v:
            b |= 1 << i

    # Row-reduce the augmented system. Row i is the equation for cell i: its
    # coefficient of variable j is bit j of ``rows_eq[i]``.
    rows_eq = []
    for i in range(n):
        coeff = 0
        for j in range(n):
            if masks[j] >> i & 1:
                coeff |= 1 << j
        rows_eq.append((coeff, b >> i & 1))

    pivot_of: dict[int, int] = {}          # variable -> its pivot row index
    r = 0
    for j in range(n):
        piv = next((rr for rr in range(r, n) if rows_eq[rr][0] >> j & 1), None)
        if piv is None:
            continue
        rows_eq[r], rows_eq[piv] = rows_eq[piv], rows_eq[r]
        cr, cb = rows_eq[r]
        for rr in range(n):
            if rr != r and rows_eq[rr][0] >> j & 1:
                orr, obb = rows_eq[rr]
                rows_eq[rr] = (orr ^ cr, obb ^ cb)
        pivot_of[j] = r
        r += 1
    if any(rows_eq[rr][0] == 0 and rows_eq[rr][1] for rr in range(n)):
        return []                          # 0 = 1 -> inconsistent

    particular = 0
    for j, rr in pivot_of.items():
        if rows_eq[rr][1]:
            particular |= 1 << j
    free = [j for j in range(n) if j not in pivot_of]
    basis = []
    for fj in free:
        v = 1 << fj
        for j, rr in pivot_of.items():
            if rows_eq[rr][0] >> fj & 1:
                v |= 1 << j
        basis.append(v)

    out = [particular]
    for v in basis:                        # sweep the null space
        out += [s ^ v for s in out]
    return out


def _plan_cells(cols: int, rows: int, off: list[int]) -> tuple[list[int], list[int]]:
    """``(minimal_click_cells, optimal_first_cells)`` for this board.

    The first is one minimum-weight click set (order-free, so listed in cell
    order); the second is every cell that appears in ANY minimum-weight set --
    the equally-optimal next clicks. Both empty if the board is unsolvable, and
    the plan is empty when the board is already won."""
    sols = _solutions(cols, rows, off)
    if not sols:
        return [], []
    best = min(bin(s).count("1") for s in sols)
    minimal = [s for s in sols if bin(s).count("1") == best]
    union = 0
    for s in minimal:
        union |= s
    n = cols * rows
    plan = [j for j in range(n) if minimal[0] >> j & 1]
    return plan, [j for j in range(n) if union >> j & 1]


class McLamelightsoutSolver(BaseSolver):
    # Filesystem-safe id (the invocation id is ``mc:lamelightsout``; see
    # common_utils.game_to_folder_mapping), used for both the output folder and
    # the ``game_id`` field of every episode.
    game_id = "mc_lamelightsout"

    # REPLAN recovery: `solve_from` re-solves the GF(2) system against the LIVE
    # board, and every click is reversible, so any state an exploratory click or
    # a burst can reach is still solvable -- the next expert step simply re-plans
    # from there. Nothing is cached per level, so no RESET is ever required.
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        # A seed switches the game to its randomised progression (7 boards, sizes
        # growing 3x1 -> 4x4) and fixes the per-(seed, level) display rotation.
        return McLamelightsout(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [CLICK_ACTION]

    # ── clicks ──────────────────────────────────────────────────────────────
    def _cell_click(self, game, cell: int) -> Action:
        """A SCREEN click on the centre of board cell ``cell`` (row-major index).

        The camera is resized to the level's ``grid_size``, so a cell is
        ``scale`` pixels wide inside the letterbox; the game maps the click back
        with ``remap_click(x, y, k)``, whose inverse is ``remap_click(..., -k)``
        (identity at k=0)."""
        cols = game.current_level.grid_size[0]
        scale, offx, offy = game.camera._calculate_scale_and_offset()
        gpx = (cell % cols) * scale + offx + scale // 2
        gpy = (cell // cols) * scale + offy + scale // 2
        sx, sy = gpx, gpy
        return Action(CLICK_ACTION, (sy, sx))          # click_rc = (row=y, col=x)

    # ── the expert ──────────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """A minimum-length click sequence that wins from the CURRENT board."""
        cols, rows, off = _read_board(game)
        plan, _ = _plan_cells(cols, rows, off)
        return [self._cell_click(game, c) for c in plan]

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next click: the cells belonging to some
        minimum-weight solution. Free -- it falls out of the same solve as the
        plan -- and it is the real tie set, since click order never matters."""
        cols, rows, off = _read_board(game)
        _, opts = _plan_cells(cols, rows, off)
        if not opts:
            return None                                # unsolved-by-solver / won
        return [self._cell_click(game, c) for c in opts]


if __name__ == "__main__":
    sys.exit(McLamelightsoutSolver.main())
