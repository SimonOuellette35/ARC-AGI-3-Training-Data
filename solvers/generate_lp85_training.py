"""Generate Phase-1 training data for the LP85 game (mouse-click only).

LP85 is a **cyclic-ring permutation puzzle** -- a Hungarian-Rings variant. The board
is a set of overlapping closed loops ("rings") of cells; every cell holds a 2x2 tile.
Each ring has a pair of buttons (``button_<ring>_L`` / ``button_<ring>_R``); clicking
one rotates that ring's tiles one cell backwards / forwards. Rings share cells, so
rotating one ring re-orders the tiles another ring will later carry -- that coupling
is the whole puzzle.

Mechanics (obfuscated ``games/lp85/lp85.py``; ``crxpafuiwp`` = 3 px per ring cell):

  * Ring layouts live in the module-level ``izutyjcpih``: ``level_name -> ring_name ->``
    a grid of cell indices; index ``n`` cycles to ``n+1`` (R) or ``n-1`` (L), wrapping
    at ``oxbwsencfv`` (the ring's highest index). ``chmfaflqhy`` turns that into a list
    of (from_cell, to_cell) moves, and ``step`` applies them: the sprite sitting exactly
    on ``from_cell`` is teleported to ``to_cell``. A cell with no sprite (a "hole", e.g.
    level 6) simply moves nothing.
  * Two tile roles matter: ``goal`` tiles must land on ``bghvgbtwcb`` targets and
    ``goal-o`` tiles on ``fdgmtkfrxl`` targets (fit test: the goal sits at
    ``target + (1, 1)``). Plain ``tile`` sprites are pure decoration -- the win check
    ``khartslnwa`` never looks at them.
  * **Only button clicks cost a step**; a click that hits no button is free
    (``if not vctdsvnwjd: complete_action(); return``). The win is checked *before* the
    step counter decrements, so a plan of exactly ``StepCounter`` clicks still wins.
    The only loss is step exhaustion.
  * **Buttons can overlap.** ``on_set_level`` strips the ``sys_click`` tag from buttons
    that share an origin, but ``step`` dispatches on ``"button" in tags[0]`` over *every*
    sprite under the cursor -- so one click on a shared cell rotates several rings at
    once. Level 5 leans on this hard: 36 rings, but only 7 distinct click positions,
    each firing a compound multi-ring rotation.

Action schema (same corpus convention as ft09 / cn04 / cd82):

    RESET :  {"type": "simple", "index": 0}      (leading action for obs[0])
    click :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Three LP85-specific decisions
-----------------------------
1. **The solver plans over an abstract permutation model, not the engine.** Because a
   ring's (from_cell -> to_cell) map is fixed by the level's ring data, every click is a
   *state-independent permutation of cells* -- including the compound overlapping-button
   clicks, which are just a composition of such permutations. ``_build_model`` recovers
   each click's permutation empirically, by driving the REAL engine once per candidate
   click cell and identity-tracking where each tile went, so the compound clicks and any
   engine quirk are captured verbatim rather than re-derived from the obfuscated source.
   ``verify`` then replays random click walks against the engine and asserts the model
   reproduces it exactly (all 8 levels, 0 mismatches) -- run it with ``--verify``.

2. **Search state is just the goal positions -> BFS gives OPTIMAL plans, instantly.**
   Decoration tiles cannot affect the win check, and goals are interchangeable (each
   target only needs *some* goal), so a state collapses to
   ``(frozenset(goal cells), frozenset(goal-o cells))``. That is at most a few tens of
   thousands of states (level 5, 3 goals over 75 cells, is the worst at ~14k), so a plain
   BFS over de-duplicated click actions returns a minimum-click plan in milliseconds --
   no plan cache, no heuristic, and no post-hoc minimizer needed (contrast ka59, whose
   engine-space A* needs both). Every level is solved comfortably inside its budget:

       L0  5/13   L1  8/60   L2 16/80   L3 12/150
       L4  9/80   L5 19/80   L6  5/80   L7  5/80   (clicks / StepCounter)

3. **Geometry is seed-invariant -> solve once, replay per seed.** The only per-seed
   randomness is the colour permutation (``_randomize_colors``) and the display rotation;
   ring layouts, tiles and targets are identical for every seed (asserted by
   ``--verify``). So each level is solved ONCE against a reference instance and the plan
   -- a list of button *grid cells* -- is replayed for every seed, that seed's colours
   baked into its frames. Rotation is pinned to k=0 (as in ft09 / ka59) so recorded
   clicks need no inverse-rotation; the colour permutation is the per-seed variety.

Determinism (fixed in the game, not worked around here)
------------------------------------------------------
``Lp85`` used to be unseedable: ``__init__`` assigned an **unseeded** ``random.Random()``
to ``self._rng`` *before* the base ``__init__`` ran ``set_level(0)`` -> ``_randomize_colors``,
so level 0's palette came from OS entropy and no amount of ``random.seed`` fixed it. And
``_randomize_colors`` was **not idempotent** -- it rewrites ``sprite.pixels`` in place
(``pixels[pixels == 11] = c_target``) while ``set_level`` never re-clones from
``_clean_levels``, so re-selecting a level silently corrupted its palette.

``games/lp85/lp85.py`` now takes a ``seed`` (as ka59 does) and is a **pure function of
(seed, level)**: the palette draws from an RNG fixed by ``(seed, level_index)`` rather than
from the shared advancing ``self._rng``, the remap restores a pristine pixel snapshot first
(so it is idempotent), and the rotation uses the seeded ``random_rotation_k(seed,
level_index)``. An unseeded ``Lp85()`` keeps the original nondeterministic behaviour.

So this generator just calls ``Lp85(seed=seed)`` -- no monkeypatch, no reliance on visit
order. Episodes are still recorded as a **single-instance playthrough** (the ft09 pattern,
letting the engine's own ``next_level`` advance), which is simply the faithful way to play;
correctness no longer depends on it. A given ``--start-seed`` is reproducible byte-for-byte,
and every emitted episode is an engine-verified win.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_lp85_training.py --episodes 500
    # check the abstract model against the engine + assert seed-invariant geometry:
    python solvers/generate_lp85_training.py --verify
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

# Repo root (parent of solvers/) -- that's where the games/ package lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from games.lp85.lp85 import Lp85  # noqa: E402
from utils.rotation import remap_click  # noqa: E402

from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import Action, CLICK_ACTION  # noqa: E402

GAME_ID = "lp85"

_RESET_INDEX = int(GameAction.RESET.value)
_RESET_ACTION = {"type": "simple", "index": _RESET_INDEX}
_ACTION6 = int(GameAction.ACTION6.value)
_STEP_GUARD = 2000

# Sprite tags: the movable tiles, the two goal roles, and their targets.
_TILE_TAGS = ("tile", "goal", "goal-o")
_GOAL_TAG, _GOALO_TAG = "goal", "goal-o"
_TARGET_TAG, _TARGETO_TAG = "bghvgbtwcb", "fdgmtkfrxl"


# ── engine helpers ─────────────────────────────────────────────────────────────
def _pin_rotation(game) -> None:
    """Pin an instance's display rotation to k=0.

    ONLY for the *planning* reference instances: the plan is a list of grid cells,
    which rotation (a pure display transform) leaves untouched, so planning at k=0
    keeps plans seed- and rotation-invariant. The instances that RECORD episodes are
    deliberately left at their natural ``random_rotation_k(seed, level)`` orientation
    -- pinning those was what made every training frame k=0 while live play draws
    k in {0,1,2,3}, so ~3/4 of eval boards had an orientation the policy never saw."""
    game._rotation_k = 0


def _is_tile(sprite) -> bool:
    return any(t in sprite.tags for t in _TILE_TAGS)


def _tiles(game):
    return [s for s in game.current_level.get_sprites() if _is_tile(s)]


def _cells_with_tag(game, tag, *, offset=0):
    return frozenset(
        (s.x + offset, s.y + offset)
        for s in game.current_level.get_sprites()
        if tag in s.tags
    )


def _drive(game, x: int, y: int):
    """Click display (x, y), stopping the instant a level solve is queued so the
    captured frame is the CLEAN solved board of the CURRENT level (not the next one,
    whose fresh on_set_level re-rolls the palette). Returns (frame, solved, dead)."""
    game._full_reset = False
    game._set_action(ActionInput(id=GameAction.ACTION6, data={"x": int(x), "y": int(y)}))
    last = None
    guard = 0
    while not game.is_action_complete():
        guard += 1
        if guard > _STEP_GUARD or game._next_level:
            break
        game.step()
        last = game.camera.render(game.current_level.get_sprites())
    if last is None:
        last = game.camera.render(game.current_level.get_sprites())
    solved = game._next_level or game._state == GameState.WIN
    dead = game._state == GameState.GAME_OVER
    return last, solved, dead


def _grid_to_display(game, gx: int, gy: int):
    """Centre-of-cell SCREEN coordinate for grid cell (gx, gy).

    ``_calculate_scale_and_offset`` is game space (rotation is applied later, to the
    composited frame), so the cell centre it yields must be inverse-rotated into the
    screen space a player clicks in: the game maps a click back with
    ``remap_click(x, y, k)``, and that map is a rotation, so its inverse is
    ``remap_click(..., -k)`` (identity at k=0, i.e. the pinned planner is unaffected)."""
    scale, offx, offy = game.camera._calculate_scale_and_offset()
    gpx = int(gx * scale + offx + scale // 2)
    gpy = int(gy * scale + offy + scale // 2)
    return (gpx, gpy)


def _mouse_action(x: int, y: int) -> dict:
    return {"type": "mouse", "index": _ACTION6, "data": {"x": int(x), "y": int(y)}}


# ── abstract model ─────────────────────────────────────────────────────────────
#
# Every click is a fixed permutation of ring cells (a composition of per-ring cycles
# when buttons overlap), independent of which tile sits where. We recover it from the
# REAL engine rather than from the obfuscated ring tables: click once, see where each
# tile went. ``verify`` (--verify) proves the recovered model is exact.
def _button_click_cells(game):
    """Every grid cell covered by a button sprite -- the candidate click positions.

    Enumerating the full footprint (not just each button's origin) is what makes the
    compound clicks visible: a cell covered by two overlapping buttons rotates both
    rings in a single step, and no single-button cell can express that."""
    cells = set()
    for s in game.current_level.get_sprites():
        if s.tags and "button" in s.tags[0]:
            for dy in range(s.height):
                for dx in range(s.width):
                    cells.add((s.x + dx, s.y + dy))
    return sorted(cells)


def _build_model(game):
    """Return (actions, goals, goalo, targets, targeto) for the current level.

    ``actions`` is a list of (click_cell, permutation) with permutations de-duplicated
    -- levels ship several buttons wired to the same ring (level 3 has 16 buttons for
    4 distinct rotations), and identical actions would only widen the BFS fan-out."""
    perms = {}
    for cell in _button_click_cells(game):
        probe = copy.deepcopy(game)
        tiles = _tiles(probe)
        before = [(s.x, s.y) for s in tiles]
        _drive(probe, *_grid_to_display(probe, *cell))
        after = [(s.x, s.y) for s in tiles]
        perm = {b: a for b, a in zip(before, after) if b != a}
        if perm:  # a click that moves nothing is not a useful action
            perms[cell] = perm

    uniq = {}
    for cell, perm in perms.items():
        uniq.setdefault(tuple(sorted(perm.items())), (cell, perm))
    actions = sorted(uniq.values(), key=lambda cell_perm: cell_perm[0])

    return (
        actions,
        _cells_with_tag(game, _GOAL_TAG),
        _cells_with_tag(game, _GOALO_TAG),
        _cells_with_tag(game, _TARGET_TAG, offset=1),
        _cells_with_tag(game, _TARGETO_TAG, offset=1),
    )


def _apply(perm, cells):
    return frozenset(perm.get(c, c) for c in cells)


# ── solver (BFS -> optimal click plan) ─────────────────────────────────────────
def solve_level(game, *, max_states=4_000_000):
    """Return a minimum-length list of button grid cells that wins the current level,
    or None. State = (goal cells, goal-o cells): decoration tiles cannot affect the win
    check and goals are interchangeable, which keeps the space small enough for BFS to
    be both exhaustive and instant."""
    actions, goals, goalo, targets, targeto = _build_model(game)

    def won(state):
        return targets <= state[0] and targeto <= state[1]

    start = (goals, goalo)
    if won(start):
        return []

    seen = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for cell, perm in actions:
            nxt = (_apply(perm, state[0]), _apply(perm, state[1]))
            if nxt in seen:
                continue
            seen[nxt] = (state, cell)
            if won(nxt):
                plan = []
                cur = nxt
                while seen[cur] is not None:
                    prev, click = seen[cur]
                    plan.append(click)
                    cur = prev
                return plan[::-1]
            queue.append(nxt)
        if len(seen) > max_states:
            return None
    return None


# ── plans (solved once -- geometry is seed-invariant) ──────────────────────────
def build_plans(verbose=True):
    """Solve every level once against a reference instance. Returns {level_idx: plan}.

    ``set_level`` is used here (rather than the episodes' single-instance playthrough) to
    reach each level without first solving its predecessor -- safe now that the game's
    palette is idempotent and fixed by (seed, level). The reference instance is read only
    for *geometry* (ring cells, tiles, targets, button footprints), which is seed-invariant
    anyway; colours never reach a plan, they are baked in per seed at replay."""
    plans = {}
    ref = Lp85(seed=0)
    for level_idx in range(len(ref._levels)):
        if level_idx:
            ref.set_level(level_idx)
        _pin_rotation(ref)
        budget = ref.current_level.get_data("StepCounter")
        plan = solve_level(ref)
        if plan is None:
            if verbose:
                print(f"  L{level_idx}: UNSOLVED -- level will be skipped")
            continue
        if len(plan) > budget:
            if verbose:
                print(f"  L{level_idx}: plan {len(plan)} exceeds budget {budget} -- skipped")
            continue
        plans[level_idx] = plan
        if verbose:
            print(f"  L{level_idx}: solved, {len(plan)} clicks (budget {budget})")
    return plans


# ── Solver ─────────────────────────────────────────────────────────────────────
class Lp85Solver(BaseSolver):
    game_id = GAME_ID
    step_guard = _STEP_GUARD
    # solve_from BFS-solves the LIVE ring state (solve_level) each call, so it
    # re-plans from any perturbed board -> replan mode. Ring rotations are fully
    # reversible, so a burst never strands the level (burst-undo is a safety net).
    supports_recovery = True
    recovery_mode = "replan"

    #: {level_idx: button-cell plan}, solved once (geometry is seed-invariant).
    _plans: dict[int, list] | None = None

    def _ensure_plans(self) -> dict[int, list]:
        if Lp85Solver._plans is None:
            Lp85Solver._plans = build_plans(verbose=False)
        return Lp85Solver._plans

    def make_game(self, seed: int):
        # Lp85(seed=S) is a pure function of (seed, level); recorded at its natural
        # per-(seed, level) rotation (NOT pinned -- pinning would make every frame k=0
        # while live play draws k in {0,1,2,3}).
        return Lp85(seed=seed)

    def available_actions(self, game) -> list[int]:
        return [CLICK_ACTION]

    def set_level(self, game, level_idx: int) -> None:
        game._next_level = False   # base drive never advances; clear the stale flag
        game.set_level(level_idx)

    def solve_from(self, game, level_idx: int, seed: int):
        """Live BFS (``solve_level``) over ``game``'s CURRENT ring state -> a
        minimum-click plan, each grid cell mapped to a centre-of-cell screen click
        at the live rotation. Re-solving from the live board (rather than replaying
        the cached ``build_plans`` plan) is what makes recovery replan-capable."""
        plan = solve_level(game)
        if not plan:
            return []
        out = []
        for cell in plan:
            dx, dy = _grid_to_display(game, *cell)
            out.append(Action(CLICK_ACTION, (dy, dx)))   # click_rc = (row=y, col=x)
        return out


if __name__ == "__main__":
    sys.exit(Lp85Solver.main())
