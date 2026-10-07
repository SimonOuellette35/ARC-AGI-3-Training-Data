"""Generate Phase-1 training data for the CD82 game (games/cd82/cd82.py).

CD82 is a *mouse-click* colour-reproduction puzzle. A blank 10x10 canvas
(sprite ``xytrjjbyib``) must be repainted to match a fixed 10x10 target pattern
(sprite ``eoqnvkspoa-pqwmeN-1``), scored only on the 80 NON-diagonal cells (the
two main diagonals of the canvas are "don't care" -- see the ``poqpfcjieu`` mask
in ``wvrremwltt``). Painting is done with a small set of region "brushes":

  * 8 *pour* brushes -- one per position on the octagonal wheel (state
    ``xwmfgtlso`` 0..7), triggered by ACTION5. Each floods a fixed region of the
    canvas (top/right/bottom/left half, or one of the four diagonal triangles)
    with the currently-selected colour. Navigate the wheel with ACTION1..4.
  * 4 *scoop* brushes -- the ``ctwspzkygu`` corner scoops present only on levels
    with them (3..6). At wheel positions 0/2/4/6 an ACTION6 click on the scoop
    floods a central edge block (top/right/bottom/left centre 3x4 / 4x3).

The selected colour is chosen by ACTION6-clicking one of the palette buttons
(``pqkenviek``) along the top; the available colours are exactly the button
colours (level 1: {0, c}; levels 2..6: {0} + N randomised colours). The target
sprite is recoloured with the SAME per-seed colour map, so its colour set always
matches the buttons.

Solver
------
A target is a *painter's-algorithm* stack of these brushes: the last brush laid
over a cell wins. ``_decompose`` peels that stack top-down -- repeatedly finding a
brush whose still-unexplained masked cells are a single target colour -- until
every non-zero masked cell is explained (colour-0 cells are the blank base
canvas). ``solve_from`` reads the LIVE game (its target sprite, palette buttons,
scoops and current wheel/colour state), decomposes, then compiles the abstract
plan into the concrete screen-space input sequence a player would press -- palette
clicks + wheel navigation + pours / scoop clicks -- by driving a throwaway
DEEPCOPY of the live game so the scoop-click geometry and wheel state come from
the engine verbatim. The base replays that sequence against the real engine and
only records an engine-verified level solve.

The peel reads the LIVE canvas paint as its base (``_canvas_pixels``): a cell only
needs a brush when the current paint differs from the target there, and colour-0
"blank" brushes scrub stray paint away. Because painting is overpaintable this
re-plans from ANY perturbed state, so recovery uses the REPLAN paradigm
(``supports_recovery = True``, ``recovery_mode = "replan"``): after the exploration
prefix (or a burst) ``solve_from`` is called from wherever the game currently is.
See ``Cd82Solver``.

Determinism / augmentation
--------------------------
``Cd82(seed=S)`` bakes every level's palette + target colours at construction
(``_build_cd82_randomized_levels(self._rng)``), so colours are a pure function of
``S`` and identical whether levels are reached fresh or by ``set_level`` on one
instance. ``on_set_level`` draws the display rotation from
``random_rotation_k(S, level)`` -- also pure. Levels are recorded at their natural
rotation; ``_screen_click`` / ``_screen_action`` convert each game-space decision
into the screen-space input a player would press.

Action schema (mixed simple + mouse, matching gp01 / bp35)
    RESET / simple :  {"type": "simple", "index": k}          # k in 0..5
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_cd82_training.py --episodes 1000 \
        --out data/training_multi_level/cd82
"""

from __future__ import annotations

import copy
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from arcengine import ActionInput, GameAction  # noqa: E402
from games.cd82.cd82 import Cd82  # noqa: E402
from utils.rotation import inverse_remap_action_full, remap_click  # noqa: E402

from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import Action, CLICK_ACTION  # noqa: E402

_STEP_GUARD = 6000
_N = 10  # canvas edge length

# Simple directional / pour actions -> GameAction.
_AMAP = {1: GameAction.ACTION1, 2: GameAction.ACTION2, 3: GameAction.ACTION3,
         4: GameAction.ACTION4, 5: GameAction.ACTION5}

# Wheel position (xwmfgtlso) -> (row, col) in the 3x3 ring (centre (1,1) excluded).
_NFHY = {0: (0, 1), 1: (0, 2), 2: (1, 2), 3: (2, 2),
         4: (2, 1), 5: (2, 0), 6: (1, 0), 7: (0, 0)}
_FBNQ = {v: k for k, v in _NFHY.items()}


# ── Brush geometry ────────────────────────────────────────────────────────────
def _pour_cells(pos: int) -> frozenset:
    """Masked cells a pour at wheel position ``pos`` (ACTION5) floods."""
    cells = set()
    for r in range(_N):
        for c in range(_N):
            if pos == 0:
                hit = r <= 4
            elif pos == 1:
                hit = c >= r
            elif pos == 2:
                hit = c >= 5
            elif pos == 3:
                hit = c >= 9 - r
            elif pos == 4:
                hit = r >= 5
            elif pos == 5:
                hit = c <= r
            elif pos == 6:
                hit = c <= 4
            else:  # pos == 7
                hit = c <= 9 - r
            if hit:
                cells.add((r, c))
    return frozenset(cells)


def _scoop_cells(pos: int) -> frozenset:
    """Masked cells a corner-scoop click at wheel position ``pos`` (0/2/4/6) floods."""
    if pos == 0:
        rr, cc = range(0, 3), range(3, 7)
    elif pos == 4:
        rr, cc = range(7, 10), range(3, 7)
    elif pos == 6:
        rr, cc = range(3, 7), range(0, 3)
    elif pos == 2:
        rr, cc = range(3, 7), range(7, 10)
    else:
        return frozenset()
    return frozenset((r, c) for r in rr for c in cc)


# Masked (scored) cells: everything except the two 10x10 diagonals.
_MASK = frozenset(
    (r, c) for r in range(_N) for c in range(_N) if c != r and c != 9 - r
)
_POUR_BRUSHES = [("pour", p, _pour_cells(p) & _MASK) for p in range(8)]
_SCOOP_BRUSHES = [("scoop", p, _scoop_cells(p) & _MASK) for p in (0, 2, 4, 6)]


# ── Peel decomposition solver ─────────────────────────────────────────────────
def _decompose(target: np.ndarray, button_colors: frozenset, has_scoop: bool,
               base: np.ndarray | None = None):
    """Return a paint plan (execution order) reproducing ``target`` on the masked
    cells starting from ``base`` (the LIVE canvas paint; defaults to blank), or
    None. Each plan item is ``(kind, pos, colour)``.

    Painting is overpaintable, so a cell only needs a brush when the base paint
    already differs from the target there (``base[c] != tgt[c]``); cells that
    already match are free. A brush is always topmost for its still-unresolved
    cells, so every such cell -- match-or-not -- must share the brush colour,
    which is why the per-brush single-colour constraint below is base-agnostic.
    With a blank (all-zero) base this reduces to the original ``tgt[c] != 0``
    condition, so no-noise plans are unchanged; from a perturbed canvas the peel
    may pick colour-0 (blank) brushes to scrub stray paint back to the target."""
    brushes = _POUR_BRUSHES + (_SCOOP_BRUSHES if has_scoop else [])
    tgt = {cell: int(target[cell[0], cell[1]]) for cell in _MASK}
    if base is None:
        cur = {cell: 0 for cell in _MASK}
    else:
        cur = {cell: int(base[cell[0], cell[1]]) for cell in _MASK}
    full = _MASK
    fail: set = set()

    def dfs(resolved: frozenset):
        remaining_nonzero = [c for c in full
                             if c not in resolved and cur[c] != tgt[c]]
        if not remaining_nonzero:
            return []
        if resolved in fail:
            return None
        candidates = []
        for kind, pos, cells in brushes:
            region_unres = [c for c in cells if c not in resolved]
            if not region_unres:
                continue
            colors = {tgt[c] for c in region_unres}
            if len(colors) != 1:
                continue
            color = next(iter(colors))
            if color not in button_colors:
                continue
            candidates.append((len(region_unres), kind, pos, color, cells))
        candidates.sort(key=lambda t: -t[0])
        for _n, kind, pos, color, cells in candidates:
            new_resolved = resolved | cells
            if new_resolved == resolved:
                continue
            sub = dfs(new_resolved)
            if sub is not None:
                return [(kind, pos, color)] + sub
        fail.add(resolved)
        return None

    peel = dfs(frozenset())
    if peel is None:
        return None
    return list(reversed(peel))


# ── Navigation on the wheel ───────────────────────────────────────────────────
def _nav_step(pos: int, action: int) -> int:
    row, col = _NFHY[pos]
    if action == 1:
        nr, nc = max(0, row - 1), col
    elif action == 2:
        nr, nc = min(2, row + 1), col
    elif action == 3:
        nr, nc = row, max(0, col - 1)
    else:  # 4
        nr, nc = row, min(2, col + 1)
    if (nr, nc) == (1, 1):
        return pos
    nxt = _FBNQ.get((nr, nc))
    if nxt is not None and nxt != pos:
        return nxt
    return pos


def _nav_path(start: int, goal: int):
    """Shortest ACTION1..4 sequence moving the wheel from ``start`` to ``goal``."""
    if start == goal:
        return []
    seen = {start}
    q = deque([(start, [])])
    while q:
        pos, path = q.popleft()
        for a in (1, 2, 3, 4):
            nxt = _nav_step(pos, a)
            if nxt in seen:
                continue
            if nxt == goal:
                return path + [a]
            seen.add(nxt)
            q.append((nxt, path + [a]))
    return None


# ── Engine helpers ────────────────────────────────────────────────────────────
def _button_clicks(game) -> dict:
    """colour -> game-space display (x, y) for each palette button."""
    scale, offx, offy = game.camera._calculate_scale_and_offset()
    out = {}
    for s in game.current_level.get_sprites():
        if s.name.startswith("pqkenviek"):
            color = int(s.pixels[2, 2])
            out[color] = (int((s.x + 2) * scale + offx),
                          int((s.y + 2) * scale + offy))
    return out


def _target_pixels(game) -> np.ndarray:
    for s in game.current_level.get_sprites():
        if s.name.startswith("eoqnvkspoa-"):
            return np.asarray(s.pixels)
    raise RuntimeError("no target sprite")


def _canvas_pixels(game) -> np.ndarray:
    """The LIVE 10x10 canvas paint (``xytrjjbyib``) -- the peel's base state."""
    for s in game.current_level.get_sprites():
        if s.name.startswith("xytrjjbyib"):
            return np.asarray(s.pixels)
    raise RuntimeError("no canvas sprite")


def _screen_click(game, gx: int, gy: int) -> tuple:
    """Game-space display coords -> the screen coords to click."""
    return (int(gx), int(gy))


def _screen_action(game, action_id: GameAction) -> GameAction:
    """Game-space action -> the screen action to press (identity for ACTION5/6)."""
    return action_id


def _sim_step(sim, action_input: ActionInput) -> None:
    """Advance ``sim`` by one action WITHOUT rendering (state evolution only)."""
    sim._full_reset = False
    sim._set_action(action_input)
    guard = 0
    while not sim.is_action_complete():
        guard += 1
        if guard > _STEP_GUARD or sim._next_level:
            break
        sim.step()


# ── Solver ────────────────────────────────────────────────────────────────────
class Cd82Solver(BaseSolver):
    game_id = "cd82"
    step_guard = _STEP_GUARD
    # The peel decomposition reads the LIVE canvas paint as its base, so it can
    # re-plan from an arbitrary painted (perturbed) state -- painting is
    # overpaintable, so every cell is recoverable. REPLAN-recovery calls
    # solve_from from wherever the game currently is instead of resetting.
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # (seed, level) -> does this level have the scoop brush? See `_has_scoop`.
        self._scoop: dict[tuple[int, int], bool] = {}

    def make_game(self, seed: int):
        return Cd82(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4, 5, CLICK_ACTION]

    def set_level(self, game, level_idx: int) -> None:
        # The base's default drive leaves ``_next_level`` set after a solve and never
        # advances; clear it so the next level records as a real (multi-step) level
        # instead of a 2-frame instant "win".
        game._next_level = False
        game.set_level(level_idx)

    def _has_scoop(self, game, level_idx: int, seed: int) -> bool:
        """Does this LEVEL offer the scoop brush?

        A level constant, but NOT one that can be read off the live sprite list:
        the game deletes the scoop sprite on every step and only re-adds it while
        the wheel sits on an even position (``cd82.eanmnpxtyi``). Reading it live
        therefore reports False whenever a perturbation happens to leave the wheel
        odd, which silently strips ``_SCOOP_BRUSHES`` from the peel and makes an
        otherwise-solvable target undecomposable -- `solve_from` then returns no
        plan from a perfectly recoverable state. So latch it per (seed, level):
        the first observation is taken at the level's initial state (wheel 0, so
        the sprite is present), and once seen it is sticky."""
        key = (seed, level_idx)
        seen = any(s.name == "ctwspzkygu" for s in game.current_level.get_sprites())
        if seen:
            self._scoop[key] = True
        return self._scoop.setdefault(key, False)

    def solve_from(self, game, level_idx: int, seed: int):
        """Optimal paint plan from ``game``'s live state, compiled to the concrete
        screen-space action sequence by driving a deepcopy. The peel reads the LIVE
        canvas paint as its base, so a plan is produced from any (possibly already
        painted / perturbed) state -- overpaint makes every cell recoverable."""
        target = _target_pixels(game)
        base = _canvas_pixels(game)
        button_colors = frozenset(_button_clicks(game).keys())
        has_scoop = self._has_scoop(game, level_idx, seed)

        plan = _decompose(target, button_colors, has_scoop, base)
        if plan is None:
            return []

        sim = copy.deepcopy(game)
        actions: list = []
        cur_color = int(sim.knqmgavuh)
        cur_pos = int(sim.xwmfgtlso)

        for kind, pos, color in plan:
            # 1) select the brush colour (palette click) if not already held.
            if color != cur_color:
                clicks = _button_clicks(sim)
                if color not in clicks:
                    return []
                bx, by = _screen_click(sim, *clicks[color])
                actions.append(Action(CLICK_ACTION, (by, bx)))
                _sim_step(sim, ActionInput(id=GameAction.ACTION6,
                                           data={"x": bx, "y": by}))
                if sim._next_level:
                    return actions
                cur_color = color

            # 2) navigate the wheel to the brush position.
            nav = _nav_path(cur_pos, pos)
            if nav is None:
                return []
            for a in nav:
                sa = _screen_action(sim, _AMAP[a])
                actions.append(Action(int(sa.value)))
                _sim_step(sim, ActionInput(id=sa))
                if sim._next_level:
                    return actions
            cur_pos = int(sim.xwmfgtlso)
            if cur_pos != pos:
                return []

            # 3) apply the brush: pour (ACTION5) or scoop click (ACTION6).
            if kind == "pour":
                sa = _screen_action(sim, GameAction.ACTION5)
                actions.append(Action(int(sa.value)))
                _sim_step(sim, ActionInput(id=sa))
            else:
                scoop = sim.bmwcxxvjum()
                if not scoop:
                    return []
                data = scoop[0].data or {}
                sx, sy = _screen_click(sim, data["x"], data["y"])
                actions.append(Action(CLICK_ACTION, (sy, sx)))
                _sim_step(sim, ActionInput(id=GameAction.ACTION6,
                                           data={"x": sx, "y": sy}))
            if sim._next_level:
                return actions

        return actions


if __name__ == "__main__":
    sys.exit(Cd82Solver.main())
