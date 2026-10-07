"""Generate Phase-1 training data for the SB26 game (games/sb26/sb26.py).

SB26 is a *mouse-click* PROGRAM-SYNTHESIS puzzle ("subroutines"). Each level shows

  * a TARGET SEQUENCE of coloured tokens along the top (``quhhhthrri`` sprites,
    read left-to-right, top-to-bottom);
  * one or more FRAMES (sprites tagged ``pkpgflvjel``) -- these are *procedures*.
    A frame holds ``int(name[-1])`` ordered slots, slot ``i`` anchored at
    ``(frame.x + 2 + 6*i, frame.y + 2)``. Sorted by ``(y, x)``, the FIRST frame is
    MAIN (the entry point); every other frame is a subroutine identified by its
    border colour ``frame.pixels[0, 0]``;
  * a STOCK of draggable blocks along the bottom row (``y > 53``). Two kinds, told
    apart by sprite NAME (never by colour): ``lngftsryyw`` (solid square) = EMIT
    its colour; ``vgszefyyyp`` (hollow ring) = CALL the frame whose border matches.

Running the program (ACTION5) walks MAIN slot by slot: an emit block outputs its
colour and must match the next target token (a mismatch, empty slot, or MAIN
running out = failure + reset); a call block pushes its frame. The level is WON
the instant the LAST target token is emitted.

Interaction / cost model
------------------------
ACTION6 click on a stock block SELECTS it; a second ACTION6 on an empty slot
PLACES it there (2 clicks). ACTION5 RUNS the program. The grid is 64x64 so the
camera scale is 1 -- click coordinates are just grid coordinates.

Solver
------
``_plan_program`` searches the EXECUTION rather than the layout: it interprets
MAIN exactly as the engine does and fills each empty slot LAZILY, the moment
control first reaches it. At any empty slot only two kinds of item can work -- an
emit of the awaited colour, or a call to any frame still in stock -- so DFS with
backtracking settles every level (including the level-7 nested call and the
level-8 recursion) in well under a millisecond. Because slots are filled only when
REACHED, the plan is automatically minimal. The engine's anti-recursion guard is
replicated so the search never proposes a program the engine would reject.

``solve_from`` turns the plan into a flat click stream (select-click, place-click
per block, then the ACTION5 run). Plans are REPLAYED on the real engine and only
an engine-verified win is recorded, so a mis-modelled level is skipped, never
leaked.

Determinism / augmentation
--------------------------
``Sb26(seed=seed)`` draws a per-level COLOUR PERMUTATION from the global ``random``
module in ``on_set_level``; this generator seeds ``random`` once with
``sb26:<seed>`` (in ``make_game``) and plays the WHOLE game on a SINGLE instance,
letting the engine's ``next_level`` advance (via ``set_level`` below calling
``_really_set_next_level``) so each level's ``on_set_level`` runs exactly once.

Episodes are recorded at the level's REAL display rotation (``Sb26`` is an
``AugmentedGame``: each level's rotation is ``random_rotation_k(seed, level)``).
Rotation is a pure DISPLAY transform, so nothing the solver reads changes; only
the CLICK COORDINATES move, which ``_screen_click`` handles (the game undoes it
with ``remap_click``). One episode == one seed, reproducible byte-for-byte,
recorded all-or-nothing over the 8 levels.

Recovery is OFF: placing blocks consumes stock and running a wrong program resets
the level, so the search is not robust to arbitrary perturbation.

Action schema (mixed simple + mouse, matching ft09 / cn04 / cd82 / bp35)
    RESET / simple :  {"type": "simple", "index": k}
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_sb26_training.py --episodes 1000 \
        --out data/training_multi_level/sb26
"""

from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import GameAction                            # noqa: E402
from games.sb26.sb26 import Sb26                            # noqa: E402
from solvers.base_solver import BaseSolver                  # noqa: E402
from utils.explore import Action, CLICK_ACTION              # noqa: E402
from utils.rotation import remap_click                      # noqa: E402

# Engine geometry (mirrors sb26.py: rfdjlhefnd / kojduumcap / evrmzyfopo).
_SLOT_INSET = 2    # first slot sits 2px in from the frame's top-left corner
_SLOT_PITCH = 6    # kojduumcap -- horizontal distance between consecutive slots
_BOTTOM_Y = 53     # evrmzyfopo -- blocks below this row are movable stock
_BLOCK_MID = 3     # click offset to a 6x6 block's centre

# Search bounds -- real solutions are a handful of steps deep; these only stop a
# degenerate call cycle (one that emits nothing) from spinning forever.
_MAX_STACK = 64
_MAX_SEARCH_STEPS = 200_000


# ── State reading ───────────────────────────────────────────────────────────
def _read_level(game):
    """Static structure of the current level: target sequence, per-frame slot
    counts + positions, FIXED (pre-placed, unclickable) slot contents and the
    movable stock. Items are ``("emit", colour)`` or ``("call", frame_index)``."""
    level = game.current_level

    frames = sorted(level.get_sprites_by_tag("pkpgflvjel"), key=lambda s: (s.y, s.x))
    nslots = [int(f.name[-1]) for f in frames]
    frame_of_colour = {int(f.pixels[0, 0]): fi for fi, f in enumerate(frames)}
    slot_pos = [
        [(f.x + _SLOT_INSET + si * _SLOT_PITCH, f.y + _SLOT_INSET) for si in range(nslots[fi])]
        for fi, f in enumerate(frames)
    ]

    seq = sorted(level.get_sprites_by_name("quhhhthrri"), key=lambda s: (s.y, s.x))
    target = [int(s.pixels[0, 0]) for s in seq]

    def item_of(sprite):
        """Block sprite -> program item. Kind comes from the NAME, never the colour."""
        colour = int(sprite.pixels[1, 1])
        if sprite.name == "vgszefyyyp":
            if colour not in frame_of_colour:
                return None  # dangling call pointer -- treat the level as unreadable
            return ("call", frame_of_colour[colour])
        return ("emit", colour)

    blocks = level.get_sprites_by_tag("lngftsryyw")
    at = {(b.x, b.y): b for b in blocks}

    fixed: dict[tuple[int, int], tuple] = {}
    for fi in range(len(frames)):
        for si, pos in enumerate(slot_pos[fi]):
            sprite = at.get(pos)
            if sprite is None:
                continue  # empty slot -- for the search to fill
            item = item_of(sprite)
            if item is None:
                return None
            fixed[(fi, si)] = item

    stock = []
    for b in blocks:
        if b.y <= _BOTTOM_Y:
            continue
        item = item_of(b)
        if item is None:
            return None
        stock.append((item, b.x, b.y))

    return {"nslots": nslots, "slot_pos": slot_pos, "target": target,
            "fixed": fixed, "stock": stock}


# ── Program search ──────────────────────────────────────────────────────────
def _plan_program(info):
    """Assign stock items to empty slots so MAIN emits the target sequence.

    Interprets the program exactly as the engine's ``dbfxrigdqx`` does, choosing
    each empty slot's content the first time control reaches it. Returns
    ``{(frame, slot): item}`` for the slots the winning run actually visits, or
    ``None`` if no assignment wins."""
    nslots, target, fixed = info["nslots"], info["target"], info["fixed"]
    L = len(target)
    if L == 0 or not nslots:
        return None

    avail = Counter(item for item, _, _ in info["stock"])
    assign: dict[tuple[int, int], tuple] = {}
    steps = 0

    def advance(stack):
        st = list(stack)
        while st:
            fi, si = st.pop()
            if si + 1 < nslots[fi]:
                st.append((fi, si + 1))
                return st
        return None

    def apply_item(stack, tok, item):
        fi, si = stack[-1]
        if item[0] == "emit":
            if item[1] != target[tok]:
                return False
            if tok + 1 == L:
                return True
            return run(advance(stack), tok + 1)
        if si == 0 and (fi, si) in stack[:-1] and len(stack) >= 2 and stack[-2][1] == 0:
            return False
        return run(stack + [(item[1], 0)], tok)

    def run(stack, tok):
        nonlocal steps
        steps += 1
        if steps > _MAX_SEARCH_STEPS or not stack or len(stack) > _MAX_STACK:
            return False
        fi, si = stack[-1]
        item = fixed.get((fi, si)) or assign.get((fi, si))
        if item is not None:
            return apply_item(stack, tok, item)

        candidates = []
        need = ("emit", target[tok])
        if avail[need] > 0:
            candidates.append(need)
        candidates.extend(("call", cf) for cf in range(len(nslots)) if avail[("call", cf)] > 0)

        for cand in candidates:
            assign[(fi, si)] = cand
            avail[cand] -= 1
            if apply_item(stack, tok, cand):
                return True
            del assign[(fi, si)]
            avail[cand] += 1
        return False

    return assign if run([(0, 0)], 0) else None


def _plan_level(game):
    """Click plan for the current level: ``[(from_xy, to_xy), ...]`` placements, or
    ``None`` if the level can't be read or solved."""
    info = _read_level(game)
    if info is None:
        return None
    assign = _plan_program(info)
    if assign is None:
        return None

    pool: dict[tuple, list[tuple[int, int]]] = {}
    for item, x, y in info["stock"]:
        pool.setdefault(item, []).append((x, y))
    for positions in pool.values():
        positions.sort()

    placements = []
    for (fi, si) in sorted(assign):
        item = assign[(fi, si)]
        if not pool.get(item):
            return None  # search over-committed the stock (shouldn't happen)
        placements.append((pool[item].pop(0), info["slot_pos"][fi][si]))
    return placements


class Sb26Solver(BaseSolver):
    game_id = "sb26"
    step_guard = 6000
    # solve_from -> _plan_level(game) reads the LIVE stock + already-placed slots and
    # re-runs the DFS synthesis, so it re-plans from a perturbed board -> replan mode.
    # Placing blocks consumes stock and a wrong ACTION5 run resets the level; if a
    # burst wedges the config unsolvable, _plan_level returns [] and the burst-undo
    # rollback erases it (RESET is the last resort).
    supports_recovery = True
    recovery_mode = "replan"

    def make_game(self, seed: int):
        # Seed the global stream BEFORE construction so level 0's on_set_level
        # colour permutation is reproducible, exactly as the original did.
        random.seed(f"{self.game_id}:{seed}")
        return Sb26(seed=seed)

    def set_level(self, game, level_idx: int) -> None:
        """Single-instance advance: level 0 is already set up by the constructor;
        later levels advance via the engine's own ``next_level`` so each
        ``on_set_level`` (colour permutation) runs exactly once. A fresh
        ``set_level`` would re-roll the colours."""
        if level_idx == 0:
            return
        if game._next_level:
            game._really_set_next_level()

    def reset_level(self, game, level_idx: int, seed: int) -> None:
        """Restore ``level_idx`` to its INITIAL state (the frame ``record_level``
        captured as ``observations[0]``). ``set_level`` ADVANCES the single engine
        instance in place, so the base ``reset_level`` cannot restore it. Instead
        rebuild the level the way it was first recorded: a fresh ``make_game``
        (re-seeding the global stream) advanced to ``level_idx`` through the
        engine's own in-place ``_really_set_next_level`` -- running each level's
        ``on_set_level`` exactly once, in order, so the colour permutation is
        reproduced -- then transplant that fresh state into ``game``."""
        fresh = self.make_game(seed)
        for _ in range(level_idx):
            fresh._really_set_next_level()
        game.__dict__.update(fresh.__dict__)

    def available_actions(self, game) -> list[int]:
        return [CLICK_ACTION, int(GameAction.ACTION5.value)]

    def _screen_click(self, game, x: int, y: int) -> Action:
        """Game-space pixel -> the SCREEN click Action a player must press to hit
        it. The game maps a click back with ``remap_click(x, y, k)`` (a rotation),
        so its inverse is ``remap_click(..., -k)`` (identity at k=0)."""
        sx, sy = int(x), int(y)
        return Action(CLICK_ACTION, (sy, sx))              # click_rc = (row=y, col=x)

    def solve_from(self, game, level_idx: int, seed: int):
        """Flat click stream: for each block, select it on the stock row then drop
        it in its slot (both centres in game space, screen-mapped), then ACTION5 to
        run the program."""
        plan = _plan_level(game)
        if plan is None:
            return []
        seq: list = []
        for (sx, sy), (tx, ty) in plan:
            seq.append(self._screen_click(game, sx + _BLOCK_MID, sy + _BLOCK_MID))
            seq.append(self._screen_click(game, tx + _BLOCK_MID, ty + _BLOCK_MID))
        seq.append(GameAction.ACTION5)
        return seq


if __name__ == "__main__":
    sys.exit(Sb26Solver.main())
