"""Generate Phase-1 training data for the SP80 game (keyboard + mouse-click).

SP80 is a **liquid-spill puzzle**. Spouts (``syaipsfndp``, the colour-4 pixels) drip liquid
(``nkrtlkykwe``) straight down when the player spills (ACTION5). The board is a set of
paddles (``ksmzdcblcz``) and diagonal deflectors (``hfjpeygkxy``) that the player drags
around *before* spilling; the spill itself is a non-interactive cascade. You win when every
basket (``xsrqllccpx``) has caught liquid and nothing has touched the floor
(``uzunfxpwmd``).

Mechanics (obfuscated ``games/sp80/sp80.py``):

  * **Editing.** One piece is selected at a time. ACTION6 clicks a piece to select it
    (bounding-box hit test over ``rxjmwfcjyw()``, i.e. paddles then deflectors -- *not*
    pixel-perfect); ACTION1/2/3/4 nudge the selection up/down/left/right. On entering a
    level the game auto-selects the piece nearest the origin (``ckahxkcgfi``), so that
    one's click is free. Every action in "change" mode costs 1 step, clicks included.
  * **Legality.** A move needs ``aqltiyljgy`` (row >= 3, a 1-cell keep-out around every
    basket) and a collision sweep. Crucially ``step()`` force-moves through collisions that
    are *only* with other paddles/deflectors, so **pieces pass through each other** and a
    piece's legal squares depend on nothing but static geometry.
  * **The cascade.** Drops do not move -- each spill tick spawns a *new* drop one cell
    ahead, leaving the old one behind as a solid trail. A drop that meets a paddle
    respawns one cell to each side, still falling, so a stream **splits and walks off both
    paddle ends**: a paddle at column p of width w turns one falling stream into two, at
    columns ``p-1`` and ``p+w``. A basket (a 3x2 U-cup, hole at ``bx+1``) fills when the
    cells flanking the drop are both that same basket; hit anywhere else it splits like a
    paddle. Deflectors rotate a stream 90 degrees. The frame is untagged, so a stream that
    reaches it just stops -- harmless. The floor is fatal.
  * **Budget.** ``steps`` actions per level; 4 failed spills also lose. Every level's
    budget has enormous slack (L1: 30 steps for a 4-step win).

Action schema (same corpus convention as ka59 / m0r0 / dc22 / sc25):

    RESET :  {"type": "simple", "index": 0}      (leading action for obs[0])
    move  :  {"type": "simple", "index": 1|2|3|4}
    spill :  {"type": "simple", "index": 5}
    click :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Four SP80-specific decisions
----------------------------
1. **The engine IS the model.** The cascade is a pile of special cases (split-on-paddle,
   fill-vs-rim on a basket, a deflector whose ``if/if/else`` both turns *and* splits in one
   branch, drops blocking their own trail) and re-deriving it from the obfuscated source
   would be guesswork. Every candidate layout is judged by running the real spill, so a
   plan is correct by construction.

   That is affordable because the spill needs **no copying**. A failed spill already
   restores itself -- ``yxidiymutj`` deletes the spawned drops, repaints the baskets and
   returns to "change" mode -- so the oracle spills in place, reads the verdict off
   ``enlvswjeov``/``srwrqoodsc``, and rolls back through the game's own reset path (undoing
   only its failed-spill counter). That is 2.5x faster than deepcopy (0.34ms vs 0.85ms) and
   leaves the board pristine; ``--verify`` checks it against deepcopy+drive outcome-for-
   outcome.

2. **Plans are seed-independent, so there is one per level -- not one per seed.** The only
   thing a seed changes about the geometry is each paddle's starting *row* (the game's
   ``_randomize_paddle_heights`` jitter) plus the palette. So the search runs once at the
   **authored rows** and returns a target layout; per seed the plan is just "walk every
   piece from wherever it woke up to that target, then spill". This is sound because a
   piece's legal squares are a static predicate (see Legality above) -- moves are therefore
   reversible and the reachable set is the connected component of the start, which the
   jitter never leaves. Undoing a nudge costs |dy| moves against budgets with 3-10x slack.

3. **The search is over columns only, in cost order.** Given the cascade above, what a
   paddle does to a stream is set by its column (it emits at ``p-1`` and ``p+w``); its row
   only *orders* the cascade, and the authored rows already order it. That collapses a
   ~10^9 (column, row) space to 12-1,035,776 column combos per level -- small enough to be
   exhaustive rather than heuristic. Candidates are spilled in nondecreasing move cost (a
   k-smallest-sums walk over the product, valid because cost is separable), so the first
   win is the cheapest and the sweep stops there: L4 lands cost 23 after 379k of its 1.04M
   combos. An independent full sweep of all 1.04M agreed exactly -- 44 wins, cheapest 23.

   Only *reachable* columns are searched. Many columns are perfectly legal but walled off
   behind the frame border (a paddle parked entirely off-grid collides with nothing), so
   treating legality as reachability would emit plans whose moves silently never happen.

4. **Levels 5 and 6 come from hand-recorded demos, not the search.** Both must turn a stream
   into a *rotated*, side-facing basket, and a deflector only turns at its own row, which
   pins that row to the basket's hole row and forces the paddle feeding it above -- so
   decision 3's column-only premise fails. Measured, not assumed:
     * columns at authored rows: 0 wins, exhaustively (40,320 / 23,520 combos, 93s / 41s);
     * a +-3 row window is 88.8M / 27.5M layouts, and random sampling wins 0 in 30,000 on
       each -- wins are rarer than 1/30,000, so no cheap sweep finds them.
   But these levels ARE solvable (a win needs every stream to end in a basket with nothing
   on the floor; that a deflector can be fed by two streams is why the naive "not enough
   deflectors" argument is wrong). So rather than build a bespoke cascade-directed search,
   a human arranges the pieces once with ``solvers/play_sp80.py``, which writes a target
   layout to ``data/sp80_demos/sp80_level{idx}_plan.json`` after verifying it wins across
   seeds. ``PlanCache`` loads those demos with precedence over the search (see ``load_demos``),
   so the plan for a level is a demo where one exists and a column-search result otherwise --
   and because plans are seed-independent (decision 2), one demo covers every seed. With demos
   present for 5 and 6, episodes are full six-level playthroughs; with none, coverage falls
   back to the levels-1-4 prefix (a level cannot be skipped mid-playthrough). ``solve_level``'s
   ``windows`` argument remains the hook if anyone later wants to search 5-6 outright.

Rotation is pinned to k=0 (the ft09 / ka59 / lp85 / bp35 convention) so recorded clicks and
directional actions need no inverse-rotation. Colour augmentation stays live, and is
episode-scoped: one palette for the whole playthrough.

Game changes this generator required (in ``games/sp80/sp80.py``)
---------------------------------------------------------------
``Sp80`` was unseedable: ``_randomize_colors`` built a **fresh unseeded** ``Random()`` on
every call and the display rotation drew from the **global** RNG, so neither pinned. Both
now come from ``_look_rng``, and the paddle-row RNG is re-seeded per (seed, level) in
``on_set_level`` so a level's rows do not depend on how many levels preceded it.
``Sp80(seed=S)`` is now a pure function of (seed, level); an unseeded ``Sp80()`` keeps the
original nondeterministic behaviour. The seed is held in ``_board_seed`` -- ``_seed`` is
``ARCBaseGame``'s own attribute and silently overwrote it.

Two further fixes, both load-bearing here:
  * The palette was re-rolled **per level**, so one playthrough changed colours between
    levels. It is now drawn once per episode (level-independent ``_look_rng`` + a cached
    ``_look``, cleared on ``full_reset``) -- the house rule for episode-scoped look.
  * The liquid colour was a **global**: ``_randomize_colors`` recoloured the shared
    module-level drop sprite, which every spill cloned. Two live games (this generator
    solves on a probe instance while recording on another) would overwrite each other's,
    and whichever rolled last coloured both of their drops -- corrupting recorded frames.
    Each game now owns a ``_drop_proto``.

Coverage: the column search solves levels 1-4; levels 5-6 are supplied as hand-recorded demos
under ``data/sp80_demos/`` (see ``solvers/play_sp80.py`` and decision 4). With those demos in
place every episode is a full six-level playthrough; without them coverage is the 1-4 prefix.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_sp80_training.py --episodes 500
    # determinism, oracle agreement, and that every plan (searched + demo) replays across seeds:
    python solvers/generate_sp80_training.py --verify
"""

from __future__ import annotations

import argparse
import copy
import heapq
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

# Repo root (parent of solvers/) -- that's where the games/ package lives.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction, GameState  # noqa: E402
from games.sp80.sp80 import Sp80  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402

GAME_ID = "sp80"

_RESET_ACTION = {"type": "simple", "index": int(GameAction.RESET.value)}
_ACTION6 = int(GameAction.ACTION6.value)
_STEP_GUARD = 2000

# up, down, left, right -- matches step()'s ACTION1/2/3/4 dispatch.
_MOVES = {
    (0, -1): GameAction.ACTION1,
    (0, 1): GameAction.ACTION2,
    (-1, 0): GameAction.ACTION3,
    (1, 0): GameAction.ACTION4,
}

_PADDLE = "ksmzdcblcz"
_DEFLECTOR = "hfjpeygkxy"
_MOVABLE = (_PADDLE, _DEFLECTOR)

# How far a piece's row may stray from the authored one, tried narrowest first. 0 -- columns
# only -- solves levels 1-4 exhaustively in seconds. Widening to (0, 3) is the obvious way to
# reach levels 5-6, and it is not enough: see decision 4.
_ROW_WINDOWS = (0,)

_DEFAULT_PLAN_CACHE = Path("data/sp80_plans.json")
_DEFAULT_DEMO_DIR = Path("data/sp80_demos")


# ── engine helpers ─────────────────────────────────────────────────────────────
def _pin_rotation(game) -> None:
    """Pin the display rotation to k=0, so a recorded click lands where it was aimed and
    ACTION1-4 are not remapped by ``lnqtlqefzv``. on_set_level re-rolls this from the
    episode look, so it must be re-pinned after every level change."""
    game.sywpxxgfq = 0
    game.nmcpyttlk.set_rotation(0)


def _drive(game, action: ActionInput, *, render: bool = False):
    """Run one action to completion. Returns (frames_or_None, solved, dead), where
    ``frames`` is a LIST of every frame the action rendered -- its whole animation.

    A spill spans many frames, and `solver.py` hands the policy the live
    ``FrameData.frame`` LIST, so recording only the settled board trained a token
    layout inference never sees (live sp80 produces 22- and 28-frame spans that
    appear nowhere in the corpus). Rendering happens INSIDE the loop for exactly
    that reason.

    The loop still breaks on ``game._next_level``, so the LAST frame captured is
    still this level's solved state, not the next level's board. ``render=False``
    (the search paths) skips rendering entirely, which is why it stays a kwarg.
    """
    game._full_reset = False
    game._set_action(action)
    guard = 0
    frames: list = []
    while not game.is_action_complete():
        if guard > _STEP_GUARD or game._next_level:
            break
        game.step()
        guard += 1
        if render:
            frames.append(
                np.asarray(game.camera.render(game.current_level.get_sprites())).tolist())
    if render and not frames:          # no-op action rendered nothing new
        frames.append(
            np.asarray(game.camera.render(game.current_level.get_sprites())).tolist())
    solved = bool(game._next_level) or game._state == GameState.WIN
    return (frames if render else None), solved, game._state == GameState.GAME_OVER


def _fast_copy(game):
    """Deep-copy the game, sharing the levels the caller cannot touch. Only ``--verify``
    needs this; the search itself never copies (see decision 1)."""
    memo = {id(game._clean_levels): game._clean_levels}
    for i, level in enumerate(game._levels):
        if i != game._current_level_index:
            memo[id(level)] = level
    return copy.deepcopy(game, memo)


# ── geometry ───────────────────────────────────────────────────────────────────
def _pieces(game):
    """Movable pieces in ``rxjmwfcjyw()`` order -- paddles then deflectors. That order is
    also the click hit-test priority, so it must be preserved everywhere."""
    return game.rxjmwfcjyw()


def _authored(game, level_idx):
    """Each movable piece's authored (name, x, y), read from the pristine ``_clean_levels``
    descriptor -- i.e. before the per-seed row jitter."""
    sprites = game._clean_levels[level_idx].get_sprites()
    return [(s.name, s.x, s.y) for tag in _MOVABLE for s in sprites if tag in s.tags]


def _legal(game, piece, x, y) -> bool:
    """Can ``piece`` rest at (x, y)? Mirrors step()'s move gate exactly: aqltiyljgy, and
    every collision must be with another paddle/deflector (which step() force-moves
    through). Depends only on static geometry, never on where other pieces sit."""
    if not game.aqltiyljgy(piece, x, y):
        return False
    ox, oy = piece.x, piece.y
    piece.set_position(x, y)
    try:
        for other in game.current_level.get_sprites():
            if other is piece or not piece.collides_with(other):
                continue
            if any(t in other.tags for t in _MOVABLE):
                continue
            return False
    finally:
        piece.set_position(ox, oy)
    return True


def _reach(game, piece, start):
    """BFS distances from ``start`` over ``piece``'s legal squares.

    Reachability, not legality, is what a plan can use: the frame border walls off squares
    that are perfectly legal (a paddle parked entirely off-grid to the left collides with
    nothing) but that no sequence of moves can enter.
    """
    gw, gh = game.current_level.grid_size
    dist = {start: 0}
    q = deque([start])
    while q:
        x, y = q.popleft()
        for dx, dy in _MOVES:
            n = (x + dx, y + dy)
            if n in dist:
                continue
            if not (-piece.width - 1 <= n[0] <= gw + 1 and -1 <= n[1] <= gh + 1):
                continue
            if not _legal(game, piece, *n):
                continue
            dist[n] = dist[(x, y)] + 1
            q.append(n)
    return dist


def _path(game, piece, start, goal):
    """A shortest legal move path start -> goal, as a list of (dx, dy)."""
    dist = _reach(game, piece, start)
    if goal not in dist:
        return None
    out = []
    cur = goal
    while cur != start:
        for dx, dy in _MOVES:
            prev = (cur[0] - dx, cur[1] - dy)
            if dist.get(prev, -1) == dist[cur] - 1:
                out.append((dx, dy))
                cur = prev
                break
        else:
            return None
    out.reverse()
    return out


# ── the spill oracle ───────────────────────────────────────────────────────────
def _spill(game) -> bool:
    """Spill the current layout and report whether it wins -- then undo it.

    No copying: a failed spill already restores the board (``yxidiymutj`` removes every
    spawned drop, repaints baskets and floor, returns to "change" mode), so this runs the
    cascade to its verdict, reads that verdict directly, and takes the same reset path
    home. Only the failed-spill counter needs undoing.
    """
    game.tadqvfdobr()
    guard = 0
    while not game.epilwznfbr and guard < _STEP_GUARD:
        game.step()
        guard += 1
    won = (not game.enlvswjeov) and all(b in game.srwrqoodsc for b in game.mldlhgjtqi())
    game.yxidiymutj()
    game.zzocrmvox -= 1
    return won


# ── search ─────────────────────────────────────────────────────────────────────
def _probe(level_idx, seed=0):
    """A game parked on ``level_idx`` with every piece back at its authored square, ready
    to be used as a spill oracle."""
    game = Sp80(seed=seed)
    if level_idx:
        game.set_level(level_idx)
    _pin_rotation(game)
    game._set_action(ActionInput(id=GameAction.ACTION5))
    pieces = _pieces(game)
    auth = _authored(game, level_idx)
    assert len(pieces) == len(auth), f"L{level_idx}: {len(pieces)} pieces vs {len(auth)} authored"
    for p, (name, ax, ay) in zip(pieces, auth):
        assert p.name == name, f"L{level_idx}: piece order drifted ({p.name} vs {name})"
        p.set_position(ax, ay)
    return game, pieces, auth


def _combos_by_cost(options):
    """Yield (cost, combo) over the product of ``options`` in nondecreasing total cost.

    ``options`` is one sorted [(cost, value), ...] list per piece. Total cost is separable
    (a sum), so this is the classic k-smallest-sums walk: start at every piece's cheapest
    choice and repeatedly pop the cheapest frontier node, pushing the nodes that advance
    one piece by one option. Cost order is what makes the sweep stoppable -- the first
    winning combo it yields is provably the cheapest one.
    """
    start = (0,) * len(options)
    total = sum(o[0][0] for o in options)
    heap = [(total, start)]
    seen = {start}
    while heap:
        cost, idx = heapq.heappop(heap)
        yield cost, tuple(options[k][i][1] for k, i in enumerate(idx))
        for k in range(len(options)):
            if idx[k] + 1 < len(options[k]):
                nxt = idx[:k] + (idx[k] + 1,) + idx[k + 1:]
                if nxt not in seen:
                    seen.add(nxt)
                    heapq.heappush(heap, (cost - options[k][idx[k]][0]
                                          + options[k][idx[k] + 1][0], nxt))


def solve_level(level_idx, *, windows=_ROW_WINDOWS, time_limit=1800.0, verbose=False,
                progress_every=50_000):
    """Search for a cheap winning layout. Returns (targets, cost, proved).

    ``targets`` is one (x, y) per piece in ``_pieces()`` order; cost counts moves from the
    authored layout (per seed the jitter only adds the rows back). ``proved`` says whether
    the answer is final -- True when a plan was found or the search space was exhausted,
    False on a timeout, which is NOT evidence of unsolvability and so must not be cached
    as one.

    Two things make this tractable (see decisions 2 and 3):

    * **Row windows, widest last.** What a paddle does to a stream is set by its column
      (it emits at ``p-1`` and ``p+w``); its row only *orders* the cascade. Levels 1-4 are
      already ordered by their authored rows, so window 0 -- columns only -- solves them in
      seconds out of a ~10^9 (column, row) space. Levels 5 and 6 are not: each has to turn
      a stream into a side-facing basket at that basket's own hole row, which pins the
      deflector's row and forces the paddle feeding it above that. So the window widens
      only when it has to, and the first window that wins is the one that pays.
    * **Cost order.** Candidates are spilled in nondecreasing cost order, so the sweep can
      stop at the first win -- which is then the cheapest in that window.

    Because a narrower window is searched first, the result is the cheapest layout of the
    first window that wins, not necessarily the global cheapest. That is deliberate: every
    budget here has 3-10x slack, so a wider search buys nothing but time.
    """
    game, pieces, auth = _probe(level_idx)
    reaches = [_reach(game, p, (ax, ay)) for p, (_, ax, ay) in zip(pieces, auth)]
    # Leave room for one click per piece plus the spill itself.
    cap = game.tunzhnhfa - len(pieces) - 1
    t0 = time.time()

    for window in windows:
        options = [sorted((c, (x, y)) for (x, y), c in d.items() if abs(y - ay) <= window)
                   for d, (_, _, ay) in zip(reaches, auth)]
        total = 1
        for o in options:
            total *= len(o)
        if verbose:
            print(f"  L{level_idx + 1}: row window +-{window} -> {total:,} reachable "
                  f"layouts, spilling in cost order (cap {cap}) ...", flush=True)
        for n, (cost, combo) in enumerate(_combos_by_cost(options), 1):
            if cost > cap:
                break
            if time.time() - t0 > time_limit:
                if verbose:
                    print(f"    TIMEOUT after {n:,} spills -- not cached, will retry",
                          flush=True)
                return None, None, False
            for p, (x, y) in zip(pieces, combo):
                p.set_position(x, y)
            if _spill(game):
                if verbose:
                    print(f"    win at cost {cost} after {n:,} spills "
                          f"({time.time() - t0:.0f}s): "
                          f"{[(auth[i][0],) + xy for i, xy in enumerate(combo)]}", flush=True)
                return list(combo), cost, True
            if verbose and n % progress_every == 0:
                print(f"    {n:,}/{total:,} spills, cost now {cost} "
                      f"({time.time() - t0:.0f}s)", flush=True)
        if verbose:
            print(f"    no win within +-{window} ({time.time() - t0:.0f}s)", flush=True)
    return None, None, True


class PlanCache:
    """Target layouts keyed by level index. Plans do not depend on the seed (decision 2),
    so this is at most one entry per level and is reused by every episode.

    A level's plan comes from one of two sources, demos winning over search:
      * a hand-recorded demo (``solvers/play_sp80.py`` -> ``<demos>/sp80_level{idx}_plan.json``),
        for the levels the automatic search cannot crack -- currently 5 and 6; or
      * the automatic column search (``solve_level``), for the rest.
    Demos are loaded LAST and overwrite anything the cache file holds, so they are authoritative
    and survive a plan-cache rebuild -- deleting the cache re-runs the search for levels 1-4 but
    never discards a recorded demo.
    """

    def __init__(self, path, demo_dir=None):
        self.path = path
        self.plans = {}
        self.misses = 0
        # Levels whose search timed out this run. A timeout is not proof of unsolvability,
        # so it must not reach the on-disk cache -- but it must not be retried per seed
        # either, or every episode would re-pay a search measured in minutes.
        self.timed_out = set()
        # Demo-sourced levels: never searched, never overwritten by a search result.
        self.demo_levels = set()
        if path and path.exists():
            self.plans = {k: [tuple(t) for t in v] if v else []
                          for k, v in json.loads(path.read_text()).items()}
        if demo_dir:
            self.load_demos(demo_dir)

    def load_demos(self, demo_dir):
        """Merge hand-recorded demos, taking precedence over the cache file."""
        demo_dir = Path(demo_dir)
        if not demo_dir.exists():
            return
        for demo_path in sorted(demo_dir.glob("sp80_level*_plan.json")):
            data = json.loads(demo_path.read_text())
            key = str(data["level"])
            self.plans[key] = [tuple(t) for t in data["targets"]]
            self.demo_levels.add(key)

    def get(self, level_idx, *, verbose=False):
        key = str(level_idx)
        if key in self.demo_levels:
            return self.plans[key] or None
        if key in self.timed_out:
            return None
        if key not in self.plans:
            targets, _cost, proved = solve_level(level_idx, verbose=verbose)
            self.misses += 1
            if not proved:
                self.timed_out.add(key)
                return None
            # Cache a proved-unsolvable level as [] too: it will not solve on a later seed
            # either, and re-searching it per seed would cost many minutes each time.
            self.plans[key] = targets or []
            self.save()
        return self.plans[key] or None

    def save(self):
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({k: [list(t) for t in v] for k, v in self.plans.items()}))


# ── plan -> actions ────────────────────────────────────────────────────────────
def _display_of(game, gx, gy):
    """Display (64x64) coordinate whose centre lands on grid cell (gx, gy). Inverts
    camera.display_to_grid, and is checked against it by round-trip."""
    cam = game.camera
    scale = min(int(64 / cam.width), int(64 / cam.height))
    padx = int((64 - cam.width * scale) / 2)
    pady = int((64 - cam.height * scale) / 2)
    dx = gx * scale + padx + scale // 2
    dy = gy * scale + pady + scale // 2
    if cam.display_to_grid(dx, dy) != (gx, gy):
        return None
    return dx, dy


def _click_cell(game, piece):
    """A grid cell that selects ``piece``. step()'s hit test is a bounding-box scan over
    rxjmwfcjyw() that takes the FIRST match, so an overlapping earlier piece would shadow
    this one; pick a cell where ``piece`` itself wins the scan."""
    order = _pieces(game)
    gw, gh = game.current_level.grid_size
    for cy in range(piece.y, piece.y + piece.height):
        for cx in range(piece.x, piece.x + piece.width):
            if not (0 <= cx < gw and 0 <= cy < gh):
                continue
            hit = next((p for p in order
                        if p.x <= cx < p.x + p.width and p.y <= cy < p.y + p.height), None)
            if hit is piece and _display_of(game, cx, cy):
                return cx, cy
    return None


def _plan_actions(game, targets):
    """Expand a target layout into the (ActionInput, record) pairs the agent sends.

    Pieces are walked one at a time, each contiguously, so each costs at most one click --
    and the auto-selected piece goes first, since its click is free. Ordering is otherwise
    free: pieces pass through each other, so no piece can ever block another's path.

    Planned on a copy that is advanced piece by piece, because a click is aimed at the
    board as it will be *at that moment*: the hit test takes the first piece whose box
    covers the cell, so aiming at the starting layout could select a piece that an earlier
    move has since slid over the target.
    """
    sim = _fast_copy(game)
    pieces = _pieces(sim)
    selected = sim.dpkgglmdup
    todo = [i for i, p in enumerate(pieces) if (p.x, p.y) != tuple(targets[i])]
    todo.sort(key=lambda i: 0 if pieces[i] is selected else 1)

    steps = []
    for i in todo:
        p = pieces[i]
        if p is not selected:
            cell = _click_cell(sim, p)
            if cell is None:
                return None
            dx, dy = _display_of(sim, *cell)
            steps.append((ActionInput(id=GameAction.ACTION6, data={"x": dx, "y": dy}),
                          {"type": "mouse", "index": _ACTION6, "data": {"x": dx, "y": dy}}))
            selected = p
        path = _path(sim, p, (p.x, p.y), tuple(targets[i]))
        if path is None:
            return None
        for mv in path:
            action = _MOVES[mv]
            steps.append((ActionInput(id=action), {"type": "simple", "index": int(action.value)}))
        p.set_position(*targets[i])        # advance the sim so later clicks aim true
    spill = GameAction.ACTION5
    steps.append((ActionInput(id=spill), {"type": "simple", "index": int(spill.value)}))
    return steps


# ── episode assembly ───────────────────────────────────────────────────────────
def record_level(game, targets):
    """Replay a target layout on the current level, recording (observations, actions), or
    (None, None) if it does not win. On a win the game is advanced in place, so the caller
    keeps playing the same instance (one on_set_level -> one board per level)."""
    observations = [np.asarray(game.camera.render(game.current_level.get_sprites())).tolist()]
    actions = [dict(_RESET_ACTION)]

    steps = _plan_actions(game, targets)
    if not steps:
        return None, None
    if len(steps) >= game.tunzhnhfa:
        # rpnnowtzay loses the moment the counter reaches 0, and it charges the step
        # *before* the spill runs -- so the budget must be strictly greater than the plan.
        return None, None

    solved = False
    for action, record in steps:
        span, solved, dead = _drive(game, action, render=True)
        observations.extend(span)                  # the action's whole animation
        actions.append({**record, "n_obs": len(span)})
        if dead:
            return None, None
        if solved:
            break
    if not solved:
        return None, None

    if game._next_level:
        game._really_set_next_level()
        _pin_rotation(game)
    return observations, actions


# ── BaseSolver subclass ─────────────────────────────────────────────────────────
class Sp80Solver(BaseSolver):
    """Thin ``BaseSolver`` binding for SP80 (harness = record/replay + schema + save +
    CLI). The expert is the column-only spill search above, plus hand-recorded demos for
    the levels it cannot crack; a single ``Sp80(seed)`` instance is played through its
    solvable-level prefix, re-pinned to k=0 at every level.

    ``supports_recovery`` is True with ``recovery_mode = "replan"``: the plan's TARGET layout
    is a seed-independent goal, but ``solve_from`` -> ``_plan_actions`` re-paths every piece
    from its LIVE position to that target on each call, so it re-plans from a perturbed board.
    Moves are reversible (pieces pass through each other; a spill self-restores), so the target
    stays reachable from any explored state. SP80 advances levels IN PLACE (``set_level`` calls
    ``_really_set_next_level``), so the default ``reset_level`` -- which would advance PAST the
    level -- is overridden to re-seat the engine at this level's initial state instead (used as
    the last-resort RESET recovery). Coverage is the solvable prefix -- ``num_levels`` reports
    how many consecutive levels from 0 have a plan (all six once the level-5/6 demos are
    present, the 1-4 prefix otherwise)."""

    game_id = GAME_ID
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, *args, plan_cache_path: Path = _DEFAULT_PLAN_CACHE,
                 demo_dir: Path = _DEFAULT_DEMO_DIR, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = PlanCache(plan_cache_path, demo_dir=demo_dir)
        self._prefix = None

    def make_game(self, seed: int):
        return Sp80(seed=seed)

    def available_actions(self, game) -> list:
        """Actions the exploration/burst policy may draw. Four moves + selection
        click -- deliberately **excluding ACTION5 (spill)**. A spill is a one-shot
        *commit*, not a reversible probe: every failed spill permanently increments
        ``zzocrmvox`` (the 4-attempt budget), which no move can undo, so a burst that
        spills enough eventually makes the level unwinnable and forces a RESET. Moves
        + clicks keep exploration entirely in reversible territory (piece geometry is
        a static predicate), while the expert still emits the genuine winning spill.
        The winning-plan path (``solve_from``/``_plan_actions``) is independent of
        this list, so the real spill is unaffected."""
        return [int(a.value) for a in _MOVES.values()] + [_ACTION6]

    def _step_counter_probe(self, game):
        """Expose sp80's live step budget so ``_lift_step_limit`` can suppress
        step-exhaustion death during generation.

        The base probe scans for ``steps_remaining`` / ``_step_counter`` / a camera
        interface exposing ``steps_remaining`` -- none of which sp80 has: its budget
        lives in ``tunzhnhfa`` (and the ``gxetqmbwgi`` HUD interface). So the base
        probe returns None, ``_lift_step_limit`` silently no-ops, and the real budget
        stays live -- exploration/burst overhead then eats it until ``rpnnowtzay``
        hits 0 and calls ``lose()``, a step-exhaustion GAME_OVER that forces a
        spurious mid-demonstration RESET. Returning the live counter here restores
        the intended protection (the HUD still ticks down and floors at 0; only the
        exhaustion *loss* is suppressed -- real losses fire while steps remain)."""
        return lambda: getattr(game, "tunzhnhfa", None)

    def num_levels(self, game) -> int:
        """The solvable-level prefix: consecutive levels from 0 that have a plan. A level
        cannot be skipped mid-playthrough, so coverage stops at the first level without a
        plan -- reproducing the original prefix coverage."""
        if self._prefix is None:
            n = len(game._levels)
            count = 0
            for i in range(n):
                if not self._cache.get(i):
                    break
                count += 1
            self._prefix = count
        return self._prefix

    def set_level(self, game, level_idx: int) -> None:
        """Position the SINGLE instance at ``level_idx`` and re-pin its rotation to k=0
        (on_set_level re-rolls it from the episode look). Level 0 is where a fresh game
        already sits; later levels are reached by advancing IN PLACE through the win
        ``drive`` just queued -- matching the original recorder."""
        if level_idx > 0:
            game._really_set_next_level()
        _pin_rotation(game)

    def reset_level(self, game, level_idx: int, seed: int) -> None:
        """Restore ``level_idx`` to its INITIAL state for RESET-recovery. The default base
        ``reset_level`` runs ``set_level``, which here ADVANCES in place -- wrong. Instead
        re-seat the engine at ``level_idx`` with the engine's own ``set_level`` (a pure
        function of (seed, level) via ``on_set_level``: same jittered rows, palette and
        rotation), then re-pin the rotation to k=0 -- reproducing the frame ``record_level``
        captured as ``observations[0]`` and the authored (jittered) start ``solve_from``
        plans against."""
        game.set_level(level_idx)
        _pin_rotation(game)

    def solve_from(self, game, level_idx: int, seed: int) -> list:
        """The plan for ``game``'s current level: the cached target layout expanded into
        the click/move actions that walk every piece to it, then the spill (ACTION5)."""
        targets = self._cache.get(level_idx)
        if not targets:
            return []
        steps = _plan_actions(game, targets)
        if not steps:
            return []
        return [action for action, _record in steps]

    # ── CLI: keep the plan-cache + demo paths ────────────────────────────────────
    @classmethod
    def build_argparser(cls) -> argparse.ArgumentParser:
        p = super().build_argparser()
        p.add_argument("--plans", type=Path, default=_DEFAULT_PLAN_CACHE,
                       help="Plan cache (one target layout per level; seed-independent).")
        p.add_argument("--demos", type=Path, default=_DEFAULT_DEMO_DIR,
                       help="Hand-recorded demos (play_sp80.py); override the cache for "
                            "levels the automatic search cannot crack (5, 6).")
        p.add_argument("--verify", action="store_true",
                       help="Check determinism, oracle agreement, and that every plan "
                            "(searched + demo) replays across seeds, then exit.")
        return p

    @classmethod
    def main(cls, argv=None) -> int:
        args = cls.build_argparser().parse_args(argv)
        if args.verify:
            ok = verify(plans=args.plans, demos=args.demos)
            print("VERIFY:", "PASS" if ok else "FAIL")
            return 0 if ok else 1
        solver = cls(rng=random.Random(args.seed), burst_prob=args.noise,
                     burst_mean=args.burst, plan_cache_path=args.plans,
                     demo_dir=args.demos)
        out = args.out or (Path("data/training_multi_level") / cls.game_id)
        rc = solver.run(args.episodes, out, start_seed=args.start_seed,
                        max_attempts=args.max_attempts,
                        progress_every=args.progress_every,
                        explore=not args.no_exploration)
        solver._cache.save()
        return rc


# ── verification ───────────────────────────────────────────────────────────────
def _drive_spill_copy(game):
    """Ground-truth spill: deepcopy the game and drive a real ACTION5 through the engine's
    own action loop. Used only to audit the no-copy oracle."""
    probe = _fast_copy(game)
    _, solved, _ = _drive(probe, ActionInput(id=GameAction.ACTION5))
    return solved


def verify(seeds=range(12), plans=_DEFAULT_PLAN_CACHE, demos=_DEFAULT_DEMO_DIR) -> bool:
    ok = True

    print("1. Sp80(seed=S) is a pure function of (seed, level)")

    def render_all(seed, revisit=False):
        g = Sp80(seed=seed)
        out = []
        for i in range(len(g._levels)):
            if i:
                g.set_level(i)
            if revisit:                       # detour through another level and back
                g.set_level(0)
                g.set_level(i)
            out.append(np.asarray(g.camera.render(g.current_level.get_sprites())).tolist())
        return out

    for seed in (0, 3, 7):
        base = render_all(seed)
        repeat = all(render_all(seed) == base for _ in range(2))
        idem = render_all(seed, revisit=True) == base
        varies = render_all(seed + 100) != base
        print(f"   seed {seed}: reproducible={repeat}  revisit-safe={idem}  "
              f"varies-by-seed={varies}")
        ok = ok and repeat and idem and varies
    unseeded = Sp80()._look != Sp80()._look
    print(f"   unseeded Sp80() still nondeterministic (legacy behaviour preserved): {unseeded}")

    print("2. One palette + rotation for every level of an episode (episode-scoped look)")
    bad_look = []
    for seed in seeds:
        g = Sp80(seed=seed)
        looks = set()
        for i in range(len(g._levels)):
            if i:
                g.set_level(i)
            looks.add((g._bg_color, g._liquid_color, g._paddle_color, g._basket_color,
                       g.sywpxxgfq))
        if len(looks) != 1:
            bad_look.append(seed)
    print(f"   {len(list(seeds))} seeds: {len(bad_look)} with a look that changed between levels")
    ok = ok and not bad_look

    print("3. The no-copy spill oracle agrees with deepcopy+drive, and leaves no residue")
    disagree = 0
    tested = 0
    for level_idx in range(6):
        game, pieces, auth = _probe(level_idx)
        before = sorted((s.name, s.x, s.y) for s in game.current_level.get_sprites())
        rng = np.random.default_rng(level_idx)
        for _ in range(40):
            for p, (_, ax, ay) in zip(pieces, auth):
                cand = [x for x in range(ax - 4, ax + 5) if _legal(game, p, x, ay)]
                p.set_position(int(rng.choice(cand)) if cand else ax, ay)
            if _spill(game) != _drive_spill_copy(game):
                disagree += 1
            tested += 1
        for p, (_, ax, ay) in zip(pieces, auth):
            p.set_position(ax, ay)
        after = sorted((s.name, s.x, s.y) for s in game.current_level.get_sprites())
        if before != after:
            print(f"   L{level_idx + 1}: board NOT pristine after spilling")
            ok = False
    print(f"   {tested} random layouts across all 6 levels: {disagree} disagreements, "
          f"failed-spill counter clean")
    ok = ok and not disagree

    print("4. Every plan (searched levels + recorded demos) wins on every seed, within budget")
    print("   (uses the on-disk cache + demos; an empty cache solves L1-L4 first, which is slow)")
    cache = PlanCache(plans, demo_dir=demos)
    solved = [i for i in range(6) if cache.get(i, verbose=True)]
    print(f"   plan sources: demos={sorted(int(k) + 1 for k in cache.demo_levels)}, "
          f"searched/cached={[i + 1 for i in solved if str(i) not in cache.demo_levels]}")
    losses = 0
    costs = []
    for seed in seeds:
        game = Sp80(seed=seed)
        _pin_rotation(game)
        for _ in solved:
            level_id = game._current_level_index
            targets = cache.get(level_id)
            if not targets:
                break
            budget = game.tunzhnhfa
            steps = _plan_actions(game, targets)
            obs, _acts = record_level(game, targets)
            if obs is None:
                print(f"   seed {seed} L{level_id + 1}: plan did NOT win")
                losses += 1
                break
            costs.append((level_id, len(steps), budget))
    for level_id in solved:
        got = [c for lid, c, _ in costs if lid == level_id]
        bud = next((b for lid, _, b in costs if lid == level_id), None)
        if got:
            print(f"   L{level_id + 1}: {min(got)}-{max(got)} of {bud} actions used "
                  f"({len(got)} seeds)")
    print(f"   levels with a plan: {[i + 1 for i in solved]}  |  "
          f"{len(list(seeds))} seeds: {losses} failures")
    ok = ok and not losses and solved

    return bool(ok)


if __name__ == "__main__":
    sys.exit(Sp80Solver.main())
