"""Generate Phase-1 training data for the BP35 game (keyboard + mouse-click).

BP35 is a gravity-faller: the player moves left/right (ACTION3/ACTION4) and
"falls" in the current gravity direction until it lands; clicking (ACTION6)
interactive tiles flips gravity, drops the player, or toggles platforms. Reaching
the gem wins the level. It is thus the first *mixed* keyboard+click game in the
corpus and emits BOTH action shapes:

    move  :  {"type": "simple", "index": 3|4}
    click :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}
    RESET :  {"type": "simple", "index": 0}     (leading action for obs[0])

``index`` is the discrete action id (still the policy's action-type target);
``data.{x,y}`` is the display-space click coordinate for the spatial head.

Two BP35-specific decisions
---------------------------
1. **Rotation: planned at k=0, RECORDED at the natural per-(seed, level) k.**
   Rotation is a pure DISPLAY transform (``RotationDisplay`` rotates the final
   composited frame; sprites, grid cells and physics are untouched), so the
   *search* pins k=0 (``_make``) and its plans are plain game-space actions --
   rotation-invariant, and thus still reusable across seeds. Only the RECORDING
   instance (``make_game``/``_make_live``) is left at the game's own
   ``random_rotation_k(seed, level)``. Each planned action is converted to the
   SCREEN-space input a player presses (``_to_screen``), because the game converts
   it straight back (``remap_action`` / ``remap_click``): moves via
   ``_screen_action``, clicks via ``_screen_click``. The recorded action is the
   screen-space one issued.
2. **Per-level recording, three sources (all replay-verified).** No shipped
   oracle solver exists, so each level's winning action sequence comes from:
     * **idx 0-2** (procedural, vary per seed): a BFS over the *real* engine,
       solved fresh for every seed (occasionally missed within the wall-clock
       budget, in which case that level is skipped for that seed).
     * **idx 3** (level 4): the fast A* solver (``solve_level_astar``, in this
       module), solved ONCE -- the grid is seed-invariant -- and replayed for all seeds.
     * **idx 4-8** (levels 5-9): hand-recorded demos from ``--plan-dir``, also
       seed-invariant, replayed for all seeds.
   The BFS/A*/demo search space is only paid once for the seed-invariant levels;
   every seed then just REPLAYS the cached plan, and the per-seed colour
   permutation is baked into that seed's rendered frames. Every plan is validated
   by an actual replay-to-win before use.

Migration notes (this file is now a ``BaseSolver`` subclass)
-----------------------------------------------------------
Unlike the all-or-nothing single-instance games, BP35 emits a SUBSET of levels:
an episode is written with whatever levels were produced (level_id preserved), and
each level is recorded on a FRESH instance (its geometry is a pure function of
(seed, level); the levels don't advance in place). So ``solve_episode`` is
overridden to reproduce that subset model rather than the base all-or-nothing one,
and the seed-invariant plans for idx 3-8 are built once (lazily) and cached on the
solver. ``solve_from`` returns the level's plan already converted to screen space;
``drive`` is the inherited one, so every action records the FULL animation it
produced (see below). ``supports_recovery`` is True in RESET mode: the plans come
from search over the real engine at a level's INITIAL state, not from an arbitrary
perturbed one, so exploration is followed by one RESET and then the plan replays.

Multi-frame observations
------------------------
BP35 animates: one move/click makes the player fall, flips gravity or slides a
platform over many engine steps. The inherited ``drive`` renders after each internal
``step()`` and returns that whole burst, so an action owns a *span* of the flat
observation stream (``n_obs`` frames) rather than a single settled frame -- the same
convention TN36 uses. The in-flight frames are what carry the physics; consumers must
use ``common_utils.action_spans`` rather than assuming 1:1 alignment.

Determinism note: BP35 is seed-deterministic (the random grid, the rotation, and
the per-episode colour permutation all derive from the seed). Colour augmentation
is EPISODE-scoped (one permutation per seed) and left ACTIVE during generation, as
rotation is: both are solvability-safe (gameplay/BFS key on sprite names, not
colour) and are exactly the variation we want in the training frames.

Usage (run from the repo root, with the arcengine env):
    python solvers/generate_bp35_training.py --episodes 500
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from arcengine import ActionInput, GameAction, GameState  # noqa: E402
import games.bp35.bp35 as _bp35  # noqa: E402
from games.bp35.bp35 import Bp35  # noqa: E402
from solvers.base_solver import BaseSolver  # noqa: E402
from utils.explore import (  # noqa: E402
    Action, CLICK_ACTION, EpsilonSchedule, ExplorationPolicy)
from utils.rotation import inverse_remap_action_full, remap_click  # noqa: E402

_UNDO = ActionInput(id=GameAction.ACTION7)
_MOVES = [GameAction.ACTION3, GameAction.ACTION4]
# Interactive tiles a click can act on (drop-pad, spawner, toggle on/off, gravity flip).
_INTERACTIVE = {"qclfkhjnaac", "etlsaqqtjvn", "yuuqpmlxorv", "oonshderxef", "lrpkmzabbfa"}


# ---------------------------------------------------------------------------
# Engine helpers
# ---------------------------------------------------------------------------
def _make(seed: int, level_idx: int) -> Bp35:
    """PLANNING instance: display rotation pinned to identity (k=0).

    Only the search (BFS here, A* in ``solve_level_astar``, and the demo replays) runs
    on this. Rotation is a pure display transform, so a plan built at k=0 is a
    plain game-space action list -- rotation- (and hence seed-) invariant."""
    g = Bp35(seed=seed)
    g.set_level(level_idx)
    g._rotation_k = 0                 # planner only; see module docstring
    return g


def _screen_click(g: Bp35, x: int, y: int) -> tuple[int, int]:
    """Game-space click (x, y) -> the SCREEN coordinate a player would click.

    The game maps a click back with ``remap_click(x, y, k)``; that map is a
    rotation, so its inverse is ``remap_click(..., -k)`` (identity at k=0)."""
    return remap_click(int(x), int(y), (-g._rotation_k) % 4)


def _screen_action(g: Bp35, action_id: GameAction) -> GameAction:
    """Game-space directional action -> the SCREEN key a player would press.

    ``urzvqcxbsz`` runs every action through ``remap_action(id, k)``, so the press
    must be pre-inverted. Non-directional ids pass through unchanged."""
    return inverse_remap_action_full(action_id, g._rotation_k)


def _to_screen(g: Bp35, action):
    """Convert a planned game-space action tuple into the screen-space tuple that
    is both issued to the engine and written to the JSON."""
    kind, val = action
    if kind == "m":
        return ("m", _screen_action(g, val))
    return ("c", _screen_click(g, val[0], val[1]))


def _state_key(g: Bp35):
    """Cheap search key: player cell + gravity dir + move-parity + on-screen
    interactive-tile layout. Coarse (ignores off-screen moving-platform heights),
    so BFS may *miss* some solutions -- but every path BFS builds is a real
    simulated sequence, so a returned solution is always a genuine win."""
    core = g.oztjzzyqoek
    hd = core.hdnrlfmyrj
    inter = tuple(sorted((cell, s.name) for cell, ss in hd.muocdhlsktl.items()
                         for s in ss if s.name in _INTERACTIVE))
    return (tuple(core.twdpowducb.qumspquyus), core.vivnprldht,
            core.wjidupyeoa % 2, inter)


def _click_targets(g: Bp35) -> list[tuple[int, int]]:
    """Click coords hitting each on-screen interactive tile. Called on the PINNED
    planner (``_make``), so these are game-space coords; ``_to_screen``
    inverse-rotates them into screen space at record time."""
    hd = g.oztjzzyqoek.hdnrlfmyrj
    k = g._rotation_k
    off = g.oztjzzyqoek.camera.rczgvgfsfb[1]
    inter = {c for c, ss in hd.muocdhlsktl.items()
             if any(s.name in _INTERACTIVE for s in ss)}
    picked: dict[tuple[int, int], tuple[int, int]] = {}
    for dy in range(0, 64, 2):
        for dx in range(0, 64, 2):
            rx, ry = remap_click(dx, dy, k)
            cell = hd.hyntnfvpgl(rx, ry + off)
            if cell in inter and cell not in picked:
                picked[cell] = (dx, dy)
    return list(picked.values())


def _candidates(g: Bp35):
    """Actions to try at this state: both moves + one click per visible tile."""
    return [("m", a) for a in _MOVES] + [("c", v) for v in _click_targets(g)]


def _to_input(action) -> ActionInput:
    kind, val = action
    if kind == "m":
        return ActionInput(id=val)
    return ActionInput(id=GameAction.ACTION6, data={"x": val[0], "y": val[1]})


def _apply(g: Bp35, action) -> str:
    """Perform one action; return 'win' / 'lose' / 'ongoing'. A win shows up as the
    engine advancing its level index (or GameState.WIN on the last level)."""
    pre = g._current_level_index
    res = g.perform_action(_to_input(action), raw=True)
    if res.state == GameState.WIN or g._current_level_index != pre:
        return "win"
    if res.state == GameState.GAME_OVER:
        return "lose"
    return "ongoing"


def _plan_record_to_action(rec):
    """Convert a play_bp35 demo record -> internal action tuple.
    ["move", 3|4] -> ("m", ACTION3|ACTION4);  ["click", x, y] -> ("c", (x, y))."""
    if rec[0] == "move":
        return ("m", GameAction.ACTION3 if rec[1] == 3 else GameAction.ACTION4)
    return ("c", (rec[1], rec[2]))


# ---------------------------------------------------------------------------
# BFS over the real engine (shortest winning action sequence, or None)
# ---------------------------------------------------------------------------
def solve_level(seed: int, level_idx: int, node_cap: int, time_cap: float):
    """Return a shortest winning action list for (seed, level_idx), or None.

    BFS. Each frontier node is reconstructed once (fresh game + replay), then its
    children are tried via apply+undo (undo is reliable). Bounded by node_cap and
    time_cap."""
    t0 = time.time()
    root = _make(seed, level_idx)
    seen = {_state_key(root)}
    frontier: deque[list] = deque([[]])
    nodes = 0
    while frontier:
        if time.time() - t0 > time_cap:
            return None
        path = frontier.popleft()
        g = _make(seed, level_idx)
        broke = False
        for a in path:                       # replay to this node
            if _apply(g, a) != "ongoing":
                broke = True
                break
        if broke:
            continue
        for a in _candidates(g):
            st = _apply(g, a)
            if st == "win":
                return path + [a]
            nodes += 1
            if nodes > node_cap:
                return None
            if st != "lose":
                h = _state_key(g)
                if h not in seen:
                    seen.add(h)
                    frontier.append(path + [a])
            g.perform_action(_UNDO, raw=True)   # revert child, back to node state
    return None


# ---------------------------------------------------------------------------
# Seed-invariant plans (idx 3-8), built once and replayed for every seed
# ---------------------------------------------------------------------------
def _plan_wins(seed: int, level_idx: int, plan) -> bool:
    """Replay ``plan`` on a fresh (seed, level) instance; True iff it reaches the win."""
    g = Bp35(seed=seed)
    g.set_level(level_idx)
    for a in plan:
        pre = g._current_level_index
        res = g.perform_action(_to_input(_to_screen(g, a)), raw=True)
        if res.state == GameState.WIN or g._current_level_index != pre:
            return True
        if res.state == GameState.GAME_OVER:
            return False
    return False


def build_plans(plan_dir, num_levels: int, solve_l4: bool):
    """Precompute one reusable winning plan per SEED-INVARIANT level (idx 3-8).

    Levels 4-9 (idx 3-8) use the fixed base grid, so a single plan replays across
    every seed. Sources: hand-recorded demos from ``plan_dir`` for idx 4-8, and the
    fast A* solver for idx 3. Each plan is validated by an actual replay-to-win
    before being kept; for a level with several demo files the shortest winning one
    is chosen."""
    plans: dict[int, list] = {}
    if plan_dir is not None and Path(plan_dir).exists():
        for f in sorted(Path(plan_dir).glob("bp35_level*_plan.json")):
            try:
                d = json.loads(f.read_text())
            except Exception:
                continue
            idx = int(d["level"])
            if idx >= num_levels or not d.get("won"):
                continue
            plan = [_plan_record_to_action(r) for r in d["plan"]]
            if not _plan_wins(int(d["seed"]), idx, plan):
                continue                     # stale / doesn't win on the current grid
            if idx not in plans or len(plan) < len(plans[idx]):
                plans[idx] = plan
    if solve_l4 and 3 < num_levels and 3 not in plans:
        raw, _ = solve_level_astar(0, 3, time_cap=90.0, weight=8.0)
        if raw and _plan_wins(0, 3, raw):
            plans[3] = raw
    return plans


# ---------------------------------------------------------------------------
# Fast solver for the mechanically-rich levels (idx >= 3)
# ---------------------------------------------------------------------------
# The stock BFS above (``solve_level``) reconstructs every frontier node from
# scratch and each ``perform_action`` pays a full fall-animation + undo-snapshot
# deepcopy, so it never clears the search space for levels 4-9. ``solve_level_dfs``
# / ``solve_level_astar`` run in an *instant-simulation* mode with a logic-only
# state save/restore, ~15x faster per action. Three engine facts make this correct:
#
# * Instant actions: the module global ``_bp35.GRAPH_BUILDER=True`` skips the
#   per-action fall-animation stepping and the undo-snapshot deepcopy
#   (drop 5ms -> ~0.33ms), producing byte-identical state transitions.
# * Logic-only restore: in instant mode there is no undo, so we back-track with the
#   engine snapshot (``vrguokymel`` / ``mfwbyhvbpc``), patching the tile sprite's
#   ``esktperyuto`` to skip the ``ajbncqttkm`` render rebuild and stripping the large
#   STATIC appearance attrs from each snapshot.
# * Parity is dead here: the move-parity counter only drives the moving-platform
#   crush check, which early-returns for levels 4-9, so states key on physical state.
#
# GRAPH_BUILDER is flipped ONLY inside the ``_instant_mode`` context manager and
# restored on exit, so instant mode never leaks into the generator's normal
# frame-recording / drive path. Route idx < 3 to the stock ``solve_level`` (parity
# matters there). Every returned plan is independently replay-validated by the caller.
_WIN = "__WIN__"


# ---------------------------------------------------------------------------
# Instant-simulation mode: GRAPH_BUILDER kills per-action animation + undo
# snapshotting (apply ~0.33ms vs ~5ms), and we patch the tile sprite's
# esktperyuto to skip the ajbncqttkm render rebuild during a state restore
# (mfwbyhvbpc), keeping only gijxrelfht (the tile-map update the game logic
# needs). Both are verified to preserve state-transition LOGIC exactly.
# ---------------------------------------------------------------------------
def _tile_class(g):
    for s in g.oztjzzyqoek.wcbpvlolmf().keunykhwkoi.values():
        if type(s).__name__ == "ogtmlfjejir":
            return type(s)
    return None


@contextlib.contextmanager
def _instant_mode(tile_cls):
    orig_gb = _bp35.GRAPH_BUILDER
    orig_esk = tile_cls.esktperyuto

    def _fast_esktperyuto(self):
        # original guard, minus the ajbncqttkm (render) rebuild
        if self.hilopxwoqvn is not None and self not in self.hilopxwoqvn.esishrsguis:
            self.hilopxwoqvn = None
            return
        self.gijxrelfht()

    _bp35.GRAPH_BUILDER = True
    tile_cls.esktperyuto = _fast_esktperyuto
    try:
        yield
    finally:
        _bp35.GRAPH_BUILDER = orig_gb
        tile_cls.esktperyuto = orig_esk


# Per-sprite attrs that are large STATIC appearance data (a dict + a list of
# tile pixels). They never change during play (verified: not among the in-place-
# mutated attrs), yet the engine deep-copies them on every restore -- ~69% of
# restore time. We strip them from each snapshot so restore skips them entirely;
# the live sprites keep their (unchanging) values. This is the single biggest
# restore speedup and is logic-preserving (verified vs ground-truth trajectories).
_STATIC_ATTRS = ("vyicipsdbdd", "ypsmynreigg")


def _lighten(snap):
    """Drop the static appearance attrs from a snapshot so restore skips them."""
    for st in snap.obvfwimxjit.values():
        a = st.axbduooyehz
        for k in _STATIC_ATTRS:
            a.pop(k, None)
    return snap


# --- parity-free physical state key (valid for levels idx >= 3) --------------
def _phys_key(g):
    core = g.oztjzzyqoek
    hd = core.hdnrlfmyrj
    inter = tuple(sorted((cell, s.name) for cell, ss in hd.muocdhlsktl.items()
                         for s in ss if s.name in _INTERACTIVE))
    return (tuple(core.twdpowducb.qumspquyus), core.vivnprldht, inter)


def _sm(g):
    return g.oztjzzyqoek.wcbpvlolmf()


def _bfs_shortest(adj, root, win_nodes):
    """Shortest action path root -> a node with a _WIN edge (over discovered adj)."""
    if not win_nodes:
        return None
    prev = {root: None}
    q = deque([root])
    target = None
    while q:
        u = q.popleft()
        if u in win_nodes:
            target = u
            break
        for a, v in adj.get(u, ()):
            if v is _WIN or v in prev:
                continue
            prev[v] = (u, a)
            q.append(v)
    if target is None:
        return None
    seq = []
    cur = target
    while prev[cur] is not None:
        u, a = prev[cur]
        seq.append(a)
        cur = u
    seq.reverse()
    seq.append(next(a for a, v in adj[target] if v is _WIN))
    return seq


def solve_level_fast(seed, level_idx, node_cap=5_000_000, time_cap=30.0):
    """Return a shortest (over the discovered graph) winning action list + stats.

    Intended for level_idx >= 3; route earlier levels to the stock BFS (parity
    matters there and undo drifts it). Search stops as soon as a win is found;
    the returned plan is the shortest path to a win *through the subgraph
    explored so far*, then independently replay-validated by the caller.

    Navigation is pure cheap apply/undo: ``ongoing`` moves undo directly;
    ``lose`` un-latches the game-over (scene flags + wrapper ``_state``) and
    undoes the frame(s) it pushed back to the node baseline; the search halts on
    the first ``win`` (which crosses a level boundary and can't be undone)."""
    t0 = time.time()
    g = _make(seed, level_idx)
    sm = _sm(g)
    stack_frames = sm.zogplfgbcbm

    root = _phys_key(g)
    visited = {root}
    adj = {root: []}
    win_nodes = set()

    def unlatch():
        g.oztjzzyqoek.jrhqdvdwpsb = False
        g.oztjzzyqoek.nkuphphdgrp = False
        g._state = GameState.NOT_FINISHED

    # DFS frame: [key, candidates, next_index, undo_stack_baseline]
    dfs = [[root, _candidates(g), 0, len(stack_frames)]]
    nodes = 0
    aborted = False
    found = False

    while dfs:
        if nodes > node_cap or time.time() - t0 > time_cap:
            aborted = True
            break
        fr = dfs[-1]
        key, cands, i, pre_len = fr
        if i >= len(cands):
            dfs.pop()
            if dfs:
                g.perform_action(_UNDO, raw=True)  # ascend (ongoing => cheap undo)
            continue
        fr[2] = i + 1
        a = cands[i]
        st = _apply(g, a)
        nodes += 1
        if st == "win":
            adj[key].append((a, _WIN))
            win_nodes.add(key)
            found = True
            break
        if st == "lose":
            unlatch()                              # un-latch, then cheap-undo
            while len(stack_frames) > pre_len:
                g.perform_action(_UNDO, raw=True)
            continue
        child = _phys_key(g)
        adj[key].append((a, child))
        if child in visited:
            g.perform_action(_UNDO, raw=True)      # cheap: revert ongoing revisit
            continue
        visited.add(child)
        adj.setdefault(child, [])
        dfs.append([child, _candidates(g), 0, len(stack_frames)])

    plan = _bfs_shortest(adj, root, win_nodes)
    stats = {"nodes": nodes, "states": len(visited), "elapsed": time.time() - t0,
             "exhausted": not aborted and not found, "win": bool(win_nodes)}
    return plan, stats


def _gem_cell(g):
    """The (static) goal-gem cell, or None if not found."""
    hd = g.oztjzzyqoek.hdnrlfmyrj
    for cell, ss in hd.muocdhlsktl.items():
        for s in ss:
            if s.name == "fjlzdjxhant":
                return cell
    return None


def solve_level_astar(seed, level_idx, node_cap=20_000_000, time_cap=60.0,
                      weight=3.0):
    """Goal-directed search in instant mode: weighted A* toward the gem.

    The blind DFS is search-bound on levels 5-9 -- large graphs, and the goal can
    sit *above* the player (needs gravity flips), so column/row wandering wastes
    the budget. Here each state is scored ``f = depth + weight * manhattan(player,
    gem)`` and expanded best-first, which steers the search at the gem (a click
    that flips gravity and drops the player toward it lowers h, so productive
    clicks float to the top and death-clicks are pruned as ``lose``).

    ``weight``: 1.0 = plain A* (shortest, slower); higher = greedier/faster to a
    win. Returns (plan, stats); plan is replay-validated by the caller. idx>=3."""
    import heapq
    t0 = time.time()
    g = _make(seed, level_idx)
    sm = g.oztjzzyqoek.wcbpvlolmf()
    tile_cls = _tile_class(g)
    gem = _gem_cell(g)

    def h():
        # Vertical-dominant: in a gravity-faller, progress is closing the row
        # gap to the gem (along the gravity axis); reaching a gap often needs a
        # horizontal move that raises Manhattan, so weight columns only lightly
        # (a tie-breaker for final alignment), never enough to veto a descent.
        if gem is None:
            return 0
        px, py = g.oztjzzyqoek.twdpowducb.qumspquyus
        return abs(py - gem[1]) + 0.25 * abs(px - gem[0])

    def unlatch():
        g.oztjzzyqoek.jrhqdvdwpsb = False
        g.oztjzzyqoek.nkuphphdgrp = False
        g._state = GameState.NOT_FINISHED

    plan = None
    nodes = 0
    aborted = False
    with _instant_mode(tile_cls):
        root = _phys_key(g)
        snaps = {root: _lighten(sm.vrguokymel())}
        gscore = {root: 0}
        pred = {root: None}
        win_from = None
        ctr = 0
        pq = [(weight * h(), 0, root)]
        while pq:
            if nodes > node_cap or time.time() - t0 > time_cap:
                aborted = True
                break
            _, _, key = heapq.heappop(pq)
            snap = snaps.get(key)
            if snap is None:
                continue                                   # stale pq entry
            depth = gscore[key]
            sm.mfwbyhvbpc(snap); unlatch()                 # position at node
            cands = _candidates(g)
            for j, a in enumerate(cands):
                st = _apply(g, a)
                nodes += 1
                if st == "win":
                    win_from = (key, a)
                    break
                if st == "lose":
                    sm.mfwbyhvbpc(snap); unlatch()
                    continue
                child = _phys_key(g)
                nd = depth + 1
                if child not in gscore or nd < gscore[child]:
                    gscore[child] = nd
                    pred[child] = (key, a)
                    snaps[child] = _lighten(sm.vrguokymel())
                    ctr += 1
                    heapq.heappush(pq, (nd + weight * h(), ctr, child))
                if j != len(cands) - 1:
                    sm.mfwbyhvbpc(snap); unlatch()
            snaps.pop(key, None)                           # done with this node
            if win_from is not None:
                break

        if win_from is not None:
            seq = [win_from[1]]
            cur = win_from[0]
            while pred[cur] is not None:
                pk, pa = pred[cur]
                seq.append(pa)
                cur = pk
            seq.reverse()
            plan = seq
        n_states = len(gscore)

    stats = {"nodes": nodes, "states": n_states, "elapsed": time.time() - t0,
             "exhausted": not aborted and plan is None, "win": plan is not None}
    return plan, stats


def solve_level_dfs(seed, level_idx, node_cap=20_000_000, time_cap=60.0):
    """Depth-first search in instant-simulation mode; stop at first win.

    This is the workhorse: the levels' solutions are DEEP (~30-60 actions), so a
    goal-directed DFS reaches a win after exploring far fewer states than BFS,
    and it holds only O(depth) snapshots (one per node on the current path)
    instead of a snapshot per frontier node. Backtracking restores the parent's
    lightened snapshot (fast); no engine undo is used (there is none in instant
    mode). Returns (plan, stats); the plan is a genuine winning action sequence
    (not necessarily shortest) that the caller replay-validates. idx >= 3 only."""
    t0 = time.time()
    g = _make(seed, level_idx)
    sm = g.oztjzzyqoek.wcbpvlolmf()
    tile_cls = _tile_class(g)

    def unlatch():
        g.oztjzzyqoek.jrhqdvdwpsb = False
        g.oztjzzyqoek.nkuphphdgrp = False
        g._state = GameState.NOT_FINISHED

    plan = None
    nodes = 0
    aborted = False
    with _instant_mode(tile_cls):
        root = _phys_key(g)
        visited = {root}
        # DFS frame: [key, candidates, next_index, node_snapshot, action_in]
        dfs = [[root, _candidates(g), 0, _lighten(sm.vrguokymel()), None]]
        while dfs:
            if nodes > node_cap or time.time() - t0 > time_cap:
                aborted = True
                break
            fr = dfs[-1]
            key, cands, i, snap, _ = fr
            if i >= len(cands):
                dfs.pop()
                if dfs:
                    sm.mfwbyhvbpc(dfs[-1][3]); unlatch()   # ascend: restore parent
                continue
            fr[2] = i + 1
            a = cands[i]
            st = _apply(g, a)
            nodes += 1
            if st == "win":
                plan = [f[4] for f in dfs[1:]] + [a]        # actions along the path
                break
            if st == "lose":
                sm.mfwbyhvbpc(snap); unlatch()              # back to node
                continue
            child = _phys_key(g)
            if child in visited:
                sm.mfwbyhvbpc(snap); unlatch()              # back to node
                continue
            visited.add(child)
            dfs.append([child, _candidates(g), 0, _lighten(sm.vrguokymel()), a])  # descend
        n_states = len(visited)

    stats = {"nodes": nodes, "states": n_states, "elapsed": time.time() - t0,
             "exhausted": not aborted and plan is None, "win": plan is not None}
    return plan, stats


def solve_level_bfs(seed, level_idx, node_cap=20_000_000, time_cap=60.0):
    """Shortest-path BFS in instant-simulation mode. Returns (plan, stats).

    Keeps one game; each frontier node holds an engine snapshot (vrguokymel) and
    is restored (skip-render mfwbyhvbpc) to expand it. BFS + predecessor links
    give a genuinely shortest action path. Intended for level_idx >= 3."""
    t0 = time.time()
    g = _make(seed, level_idx)                    # normal mode: populate state mgr
    sm = g.oztjzzyqoek.wcbpvlolmf()
    tile_cls = _tile_class(g)

    def unlatch():
        g.oztjzzyqoek.jrhqdvdwpsb = False
        g.oztjzzyqoek.nkuphphdgrp = False
        g._state = GameState.NOT_FINISHED

    plan = None
    nodes = 0
    aborted = False
    with _instant_mode(tile_cls):
        root = _phys_key(g)
        root_snap = _lighten(sm.vrguokymel())
        snaps = {root: root_snap}                 # snapshot per live frontier node
        pred = {root: None}                       # key -> (prev_key, action)
        visited = {root}
        frontier = deque([root])
        win_from = None

        while frontier:
            if nodes > node_cap or time.time() - t0 > time_cap:
                aborted = True
                break
            key = frontier.popleft()
            snap = snaps.pop(key)
            sm.mfwbyhvbpc(snap); unlatch()        # position game at this node
            cands = _candidates(g)
            for j, a in enumerate(cands):
                st = _apply(g, a)
                nodes += 1
                if st == "win":
                    win_from = (key, a)
                    break
                if st == "lose":
                    sm.mfwbyhvbpc(snap); unlatch()   # lose corrupts -> restore node
                    continue
                child = _phys_key(g)
                if child not in visited:
                    visited.add(child)
                    pred[child] = (key, a)
                    snaps[child] = _lighten(sm.vrguokymel())
                    frontier.append(child)
                if j != len(cands) - 1:
                    sm.mfwbyhvbpc(snap); unlatch()   # return to node for next cand
            if win_from is not None:
                break

        if win_from is not None:
            # reconstruct shortest path root -> win_from[0], then the winning action
            seq = [win_from[1]]
            cur = win_from[0]
            while pred[cur] is not None:
                pk, pa = pred[cur]
                seq.append(pa)
                cur = pk
            seq.reverse()
            plan = seq
        n_states = len(visited)

    stats = {"nodes": nodes, "states": n_states, "elapsed": time.time() - t0,
             "exhausted": not aborted and plan is None, "win": plan is not None}
    return plan, stats


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class Bp35Solver(BaseSolver):
    #: Drives the engine through ``game.perform_action`` -- the AGENT-facing entry
    #: point, which de-rotates whatever it is handed. So this generator's plan is
    #: already in SCREEN space (it converts it itself), and the base must NOT
    #: convert again when recording. Generators that use the base's ``drive``
    #: (``_set_action``, which bypasses the wrapper) leave this False.
    plans_in_screen_space = True
    game_id = "bp35"
    # RESET-recovery: ``solve_from`` re-plans a level from its INITIAL state only
    # (levels 0-2 re-run BFS on a fresh ``_make(seed, level)``; 3-8 replay a cached
    # seed-invariant plan), so the exploration prefix perturbs, then ONE RESET
    # restores the initial state (``reset_level`` -> the AugmentedGame ``set_level``
    # re-clones + re-seeds the grid), and the plan replays from there.
    supports_recovery = True
    recovery_mode = "reset"

    def __init__(self, *, plan_dir=Path("data/bp35_demos"), num_levels: int = 9,
                 solve_l4: bool = True, node_cap: int = 20000, time_cap: float = 15.0,
                 **kwargs):
        super().__init__(**kwargs)
        self._plan_dir = plan_dir
        self._num_levels = num_levels
        self._solve_l4 = solve_l4
        self._node_cap = node_cap
        self._time_cap = time_cap
        self._max_bfs_level = 2       # idx 0-2 are per-seed BFS; 3-8 use cached plans
        self._plans: dict[int, list] | None = None    # built lazily on first episode

    def _ensure_plans(self) -> None:
        if self._plans is None:
            self._plans = build_plans(self._plan_dir, self._num_levels, self._solve_l4)

    def make_game(self, seed: int):
        return Bp35(seed=seed)        # RECORDING instance: natural per-(seed, level) rotation

    def available_actions(self, game) -> list[int]:
        return [3, 4, CLICK_ACTION]   # left/right moves + interactive-tile clicks

    @staticmethod
    def _to_action(sa) -> Action:
        """Screen-space internal tuple -> ``explore.Action`` (move or click)."""
        kind, val = sa
        if kind == "m":
            return Action(int(val.value))
        x, y = val
        return Action(CLICK_ACTION, (int(y), int(x)))   # click_rc = (row=y, col=x)

    def solve_from(self, game, level_idx: int, seed: int):
        """The level's winning plan, converted to THIS instance's screen space.

        idx 3-8 replay a precomputed seed-invariant plan; idx 0-2 are solved fresh
        by BFS. Empty list -> the level is skipped for this seed (subset model)."""
        self._ensure_plans()
        if level_idx in self._plans:
            plan = self._plans[level_idx]
        elif level_idx <= self._max_bfs_level:
            plan = solve_level(seed, level_idx, self._node_cap, self._time_cap)
            if plan is None:
                return []
        else:
            return []                 # no demo/solver for this hard level
        return [self._to_action(_to_screen(game, a)) for a in plan]

    # NOTE: `drive` is deliberately NOT overridden -- the base implementation is
    # what makes BP35's frames worth recording. BP35 is a gravity-faller: a single
    # move/click plays out an ANIMATION (the fall, the gravity flip, the platform
    # toggle) across many engine steps, and that in-flight motion is precisely the
    # mechanic an agent has to infer. The base `drive` renders after every internal
    # `step()` and returns the WHOLE span (recorded as one action with `n_obs` ==
    # len(frames), tn36-style), instead of only the settled frame.
    #
    # It also stops the loop the instant a level solve is queued (`_next_level`),
    # which the old `perform_action(raw=True)` override could not do: the engine's
    # own loop runs `_really_set_next_level()` and renders one more frame, so its
    # LAST frame was the next level's board rather than this level's solved one.

    def solve_episode(self, seed: int, explore: bool = True):
        """SUBSET model: record whatever levels solve, on a FRESH instance per level
        (each level's geometry is a pure function of (seed, level); levels don't
        advance in place). Returns (ok, levels) with ok == "at least one level
        produced", mirroring the original generator.

        Mirrors the base ``solve_episode`` wiring: the epsilon schedule + coverage
        policy are built ONCE (when ``explore and supports_recovery``) so the
        exploration arc is episode-wide, and each level's ``record_level`` runs the
        RESET-recovery loop (explore prefix -> ONE reset to the level's initial
        state -> replay the plan). A skipped level does not consume the horizon
        unfairly -- the schedule keeps decaying across the whole episode."""
        self._ensure_plans()
        do_explore = explore and self.supports_recovery
        schedule = (EpsilonSchedule(self.rng, center=self._explore_center,
                                    jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(self.make_game(seed)),
                                         self.rng)
                       if do_explore else None)
        levels: list[dict] = []
        for level_idx in range(self._num_levels):
            game = self.make_game(seed)
            try:
                obs, acts = self.record_level(game, level_idx, seed,
                                              schedule=schedule,
                                              exploration=exploration)
            except Exception:                    # noqa: BLE001
                obs, acts = None, None
            if obs is None:
                continue                         # subset: skip an unsolved/timed-out level
            levels.append({"level_id": level_idx, "observations": obs, "actions": acts})
        return bool(levels), levels


if __name__ == "__main__":
    sys.exit(Bp35Solver.main())
