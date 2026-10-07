"""Generate Phase-1 training data for the PuzzleScript game ps:bridge_toggle_maze
(Andrea Gilbert's "Bridge-toggle Maze", the clickmazes puzzle).

The harness -- the trajectory recorder, the rotation contract and the BaseSolver
plumbing -- lives in `solvers/common/ps_astar.py`, shared with the other ps:
generators. This file is the game-specific part: a re-implemented MODEL of the
mechanic, a search ladder over that model, and an engine verification pass that
throws away any plan the real interpreter does not actually win with.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema::

    {"game_id": "puzzlescript_bridge_toggle_maze",
     "levels": [{"level_id": 0,
                 "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette
                 "actions":      [a_0, ..., a_{T-1}]},      # length T
                ...]}

actions[0] is the RESET that produced obs[0]; actions[i] for i>=1 is the *screen*
action (post rotation-remap) that took the agent from obs[i-1] to obs[i].

The game
========
A grid of one-way BRIDGES over a road network. Every cell is one of

  * a **crossroad** (``Xroad``, a bare black-cornered square) -- passable on
    both axes, and inert; or
  * a **bridge**, drawn with a coloured centre pixel and a black frame whose
    open side shows which axis it carries: ``greenNS``/``redNS`` run
    north-south, ``greenEW``/``redEW`` run east-west.

One key press moves the player along that axis to the NEAREST cell it can stand
on -- a crossroad, or a bridge aligned with the direction of travel. Bridges
lying ACROSS the direction of travel are passed straight over (that is what
makes them bridges), and if there is no landable cell at all in that direction
the turn is cancelled. Standing ON a bridge, the perpendicular keys are dead:
you can only continue along it or turn back.

The mechanic is what the player LEAVES BEHIND. Departing a bridge rotates it a
quarter turn AND flips its colour, in a fixed 2-cycle::

    greenNS <-> redEW          redNS <-> greenEW

Departing a crossroad does nothing. The win is ``No redNS`` + ``No redEW`` +
``All Player on StartPos``: turn every bridge green, then come home.

Two consequences drive the whole solver:

  * **Colour and orientation are locked together, so each bridge is ONE BIT.**
    Each cell stays in its own 2-cycle forever -- a ``redEW`` is only ever
    ``redEW`` or ``greenNS`` -- so the class is static per cell and the live
    state is just "flipped an even or an odd number of times". Every bridge
    ships red, so the goal is "every bridge departed an ODD number of times".
    The full state is therefore ``(player cell, one bit per bridge)``, which is
    what the search below enumerates.
  * **Every move departs exactly one cell**, so a plan can never be shorter
    than the number of still-red bridges. That is the search heuristic, and it
    is admissible but LOOSE: solutions run about 2.7x the bridge count, because
    the walk keeps landing on crossroads (free moves) and has to re-cross
    bridges it already fixed (a green bridge departed again goes back to red,
    costing two).

An engine bug this game exposed
==============================
The mechanic is written as one ellipsis rule -- ``vertical [ > player | ... |
OkNS ] -> [ prevPos | ... | player OkNS ]``, i.e. "slide to the nearest cell you
can stand on". `PuzzleScriptAdapter` enumerated EVERY binding of that ellipsis
(the right thing for propagation rules, which is why it does it) and applied all
of them without re-checking, so a single key press left a COPY of the player on
every landable cell down the column. ``All Player on StartPos`` was then
unreachable and the game was unplayable. The adapter now re-validates each
binding against the already-mutated grid before firing it -- the same discipline
its multi-bracket path already used, and what PuzzleScript's reference engine
does. See `PSEngine._match_still_valid`; the fix is inert for propagation rules,
whose RHS does not consume their LHS.

Why a model, not engine A*
==========================
The family default is to search the real interpreter (`PSExpert`). Solutions
here run 30-120 moves over a state space of ``cells x 2^bridges`` (level 6 is
49 cells and 32 bridges) and the searches need millions of nodes; an
interpreter step is ~1 ms and the model steps in ~2 us. The risk a model carries
is silent drift, and it is paid for twice, exactly as in
`generate_autumn_training.py`:

  * a differential fuzz test (random play, every level, full state compared
    after every single step) ships as ``--verify-model`` -- currently 0
    mismatches over 7 levels x 60 episodes x 60 steps -- and is what caught the
    player-cloning bug above; and
  * every plan is REPLAYED ON THE REAL INTERPRETER before it is returned
    (`BridgeExpert._verify`), so model drift can only ever cost coverage, never
    emit a broken demonstration.

The search ladder
=================
`LADDER`, tried in order until a rung wins (see it for the per-rung numbers):

  * **A\\*, weight 1** -- optimal. Solves levels 0, 1, 2 and 4 in under a second.
  * **A\\*, weight 2** -- the same search leaning on the heuristic. Levels 3 and
    5 fall out in ~2 s each; plans are winning but no longer provably shortest.
  * **BEAM, width 40000** -- level 6 (7x7, 32 bridges) resists A\\* at every
    weight tried, including greedy: ``h`` counts red bridges and goes flat
    across the long stretches where the walk is repositioning rather than
    fixing anything, so A\\* degenerates to uniform-cost at a depth where that is
    hopeless. A beam spends its budget on breadth instead and wins in ~80 s.

Beam plans wander, so a beam rung is followed by `shorten`, a
longest-block-first deletion pass over the model (the same trick the Aperture
Science generator uses). All 7 levels solve: 5, 15, 27, 51, 40, 51 and 94
moves, in ~2 min of search for the whole game.

The searches are paid ONCE ever: plans are cached to
``data/bridge_toggle_maze_plans.json`` keyed by the level's start layout, which
is what matters under `parallelize_generator.py`, where every shard would
otherwise redo them.

Optimal-action targets
======================
`optimal_for` labels every expert step with the plan's own move. There is no
affordable exact distance field to derive a tie SET from -- the state space is
``cells x 2^bridges``, so "is this alternative also optimal?" is a second full
search per step -- and on the weighted / beam levels the plan is not provably
shortest anyway, so the honest target is what the expert did. What matters is
that no expert step goes out unlabelled.

Augmentation
============
The engine state after reset is identical for every seed (levels are fixed ASCII
maps); only the presentation varies per (seed, level) -- the frame rotation
(``rotation_k`` in {0,1,2,3}). ``Bridge-toggle_Maze`` is not in
`PuzzleScriptAdapter._RECOLOR_GAMES`, and deliberately stays out: red-vs-green
IS the win condition here, so a chrome recolor is the one augmentation this
board cannot afford. The plan is seed-independent: solved once per level, cached,
and replayed per seed with that seed's rotation-remapped screen actions.

Usage (run from the repo root)::

    python solvers/generate_bridge_toggle_maze_training.py --report
    python solvers/generate_bridge_toggle_maze_training.py --verify-model
    python solvers/generate_bridge_toggle_maze_training.py --episodes 200 \
        --out data/training_multi_level/bridge_toggle_maze
"""

from __future__ import annotations

import heapq
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from solvers.common.ps_astar import (                      # noqa: E402
    PSAStarSolver, PSExpert, restore, snapshot,
)

GAME_NAME = "Bridge-toggle_Maze"

#: The four engine directions. ACTION5 is bound to nothing in this game (no rule
#: mentions ``action``), so it is not in the search and not in `MOVES`.
MOVES: tuple[str, ...] = ("up", "down", "left", "right")
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
#: Indices into `MOVES` that travel on the north-south axis.
_VERTICAL: tuple[bool, ...] = tuple(d in ("up", "down") for d in MOVES)

#: Seed-independent plans, in the repo's usual ``data/<game>_plans.json`` shape.
#: Each entry records the start layout it was solved from and is ignored if that
#: no longer matches, so editing the level in the .txt cannot serve a stale plan.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "bridge_toggle_maze_plans.json"

#: ``(kind, weight_or_width, budget)`` rungs, tried in order until one wins.
#:
#: The A* budgets are node counts; the beam's is its width (its depth cap is
#: `BEAM_DEPTH`). Rung 1 is the optimal one and is deliberately given a SMALL
#: budget: it wins levels 0/1/2/4 in well under a second and there is no point
#: burning 20 s per level proving it cannot reach the other three. Rung 3 is
#: only ever needed by level 6.
LADDER: tuple[tuple[str, int, int], ...] = (
    ("astar", 1, 600_000),
    ("astar", 2, 4_000_000),
    ("beam", 40_000, 0),
)

#: Layers the beam explores before giving up. A beam plan is exactly as long as
#: the layer it was found on, so this doubles as the longest plan a beam rung
#: can return. Level 6's comes in at 94, and every layer of a FAILING beam costs
#: ~2 s at width 40000, so there is a real price for headroom nobody uses.
BEAM_DEPTH = 250


def _ladder_id() -> list:
    """Fingerprint of the search's STRENGTH, stored beside a cached "this level
    is unsolvable" so that strengthening the ladder retries it instead of being
    stopped by its own earlier failure. It covers the dials, not the code: edit
    `beam`'s ranking or `heuristic` and the cached "no" has to be dropped by
    hand (delete `PLAN_CACHE`)."""
    return [list(map(list, LADDER)), BEAM_DEPTH]


# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------

class Model:
    """The whole mechanic, as flat arrays over cell indices ``r * W + c``.

    A state is the pair ``(pos, mask)``: the player's cell and one bit per
    bridge, set when that bridge has been departed an ODD number of times (i.e.
    when it is currently GREEN -- see the module docstring on why colour and
    orientation collapse to a single bit).

    ``bit[cell]`` is the bridge's mask bit (0 for a crossroad) and
    ``vgreen[cell]`` says whether that bridge runs north-south when green. So a
    cell carries traffic on the vertical axis exactly when
    ``bool(mask & bit[cell]) == vgreen[cell]`` -- and a crossroad, with no bit,
    carries both axes.
    """

    __slots__ = ("H", "W", "bit", "vgreen", "start", "rays", "full", "n_bridges")

    def __init__(self, H, W, bit, vgreen, start, full, n_bridges):
        self.H, self.W = H, W
        self.bit = bit
        self.vgreen = vgreen
        self.start = start
        self.full = full              # mask with every bridge green
        self.n_bridges = n_bridges
        # rays[di][cell] = the cells beyond `cell` in direction `di`, nearest
        # first. Precomputed because the successor loop walks one on every
        # single node expansion, and there are millions of those.
        self.rays = [
            [self._ray(cell, _DELTA[MOVES[di]]) for cell in range(H * W)]
            for di in range(4)
        ]

    def _ray(self, cell, delta) -> tuple[int, ...]:
        dr, dc = delta
        r, c = divmod(cell, self.W)
        out = []
        r, c = r + dr, c + dc
        while 0 <= r < self.H and 0 <= c < self.W:
            out.append(r * self.W + c)
            r, c = r + dr, c + dc
        return tuple(out)

    def step(self, pos: int, mask: int, di: int):
        """Apply direction ``MOVES[di]``; return the new ``(pos, mask)``, or
        ``None`` when the engine would cancel the turn."""
        vert = _VERTICAL[di]
        bit, vgreen = self.bit, self.vgreen
        b = bit[pos]
        if b and ((mask & b != 0) == vgreen[pos]) != vert:
            return None                    # side-step off a bridge: cancelled
        for cell in self.rays[di][pos]:
            cb = bit[cell]
            if not cb or ((mask & cb != 0) == vgreen[cell]) == vert:
                return cell, (mask ^ b) if b else mask
        return None                        # nothing landable that way

    def won(self, pos: int, mask: int) -> bool:
        return mask == self.full and pos == self.start

    def replay(self, pos: int, mask: int, plan) -> int:
        """Run ``plan`` and return the 1-based length of its winning prefix, or
        0 if it never wins. A cancelled turn is not a failure -- the engine
        simply ignores it -- so the replay carries on."""
        for i, d in enumerate(plan):
            nxt = self.step(pos, mask, MOVES.index(d) if isinstance(d, str) else d)
            if nxt is not None:
                pos, mask = nxt
                if self.won(pos, mask):
                    return i + 1
        return 0


def read_state(eng, gm) -> tuple[Model, int, int]:
    """Build ``(Model, pos, mask)`` from the interpreter's live grid.

    Read off the engine rather than re-parsed from the .txt so the model always
    starts from the state the adapter actually presents, whatever its level
    loader did on the way in.
    """
    N = gm.obj_name_to_idx
    H, W = eng.height, eng.width
    ns_ids = {N["greenns"], N["redns"]}
    ew_ids = {N["greenew"], N["redew"]}
    green_ids = {N["greenns"], N["greenew"]}
    bit = [0] * (H * W)
    vgreen = [False] * (H * W)
    mask = 0
    n_bridges = 0
    pos = start = None
    for r in range(H):
        for c in range(W):
            i = r * W + c
            cell = eng.grid[r][c]
            if N["player"] in cell:
                pos = i
            if N["startpos"] in cell:
                start = i
            is_ns = bool(cell & ns_ids)
            if not (is_ns or cell & ew_ids):
                continue                   # a crossroad: no bit, no class
            green = bool(cell & green_ids)
            bit[i] = 1 << n_bridges
            # The cell's class, as "which axis does it run on when GREEN": that
            # is its current axis if it is green now, and the other one if not.
            vgreen[i] = is_ns if green else not is_ns
            if green:
                mask |= bit[i]
            n_bridges += 1
    full = (1 << n_bridges) - 1
    return Model(H, W, bit, vgreen, start, full, n_bridges), pos, mask


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def heuristic(model: Model, pos: int, mask: int) -> int:
    """Admissible lower bound on the moves left.

    Every move departs exactly ONE cell and so flips at most one bridge, and
    every still-red bridge has to be departed at least once more -- hence the
    red count. One refinement is free and pays for itself: the FIRST move
    departs the cell the player is standing on, so if that cell is not itself a
    red bridge (a crossroad, or a bridge already green) then that move cannot be
    one of the fixes and the bound goes up by one.
    """
    k = model.n_bridges - bin(mask).count("1")
    if not k:
        return 0 if pos == model.start else 1
    b = model.bit[pos]
    return k if (b and not mask & b) else k + 1


def astar(model: Model, pos0: int, mask0: int, weight: int, node_cap: int):
    """(Weighted) A* over ``(pos, mask)``. ``weight == 1`` is optimal; above
    that the plan is still a genuine win, just not provably shortest."""
    if model.won(pos0, mask0):
        return []
    start = (pos0, mask0)
    pq = [(weight * heuristic(model, pos0, mask0), 0, start)]
    best = {start: 0}
    parent: dict[tuple[int, int], tuple] = {}
    nodes = 0
    while pq:
        _f, g, state = heapq.heappop(pq)
        if best.get(state, 1 << 30) < g:
            continue                       # a stale heap entry
        pos, mask = state
        ng = g + 1
        for di in range(4):
            nxt = model.step(pos, mask, di)
            if nxt is None:
                continue
            nodes += 1
            if best.get(nxt, 1 << 30) <= ng:
                continue
            best[nxt] = ng
            parent[nxt] = (state, di)
            if model.won(*nxt):
                return _unwind(parent, nxt)
            heapq.heappush(
                pq, (ng + weight * heuristic(model, *nxt), ng, nxt))
        if nodes >= node_cap:
            return None
    return None


def _unwind(parent, state) -> list[str]:
    out = []
    while state in parent:
        state, di = parent[state]
        out.append(MOVES[di])
    out.reverse()
    return out


def beam(model: Model, pos0: int, mask0: int, width: int, depth: int):
    """Width-capped breadth-first beam, ranked by ``(red bridges, away from
    home)``.

    WHY, when A* is right there: ``heuristic`` counts red bridges, and over the
    long stretches where the walk is repositioning rather than fixing anything
    it does not move at all. A* has nothing to steer with there and degenerates
    to uniform-cost search at a depth level 6 puts well out of reach -- greedy
    weights do not help, because the flat region is flat at every weight. A beam
    spends the same budget on breadth at every depth and only uses the
    heuristic to break ties, which is all a flat heuristic is good for.

    The second rank key is not decoration: the win needs two things -- every
    bridge green AND the player home -- and the red count only sees the first,
    so within one red count the beam would otherwise be choosing blind.

    WIDTH IS THE DIAL, and level 6 is what set it. At width 20000 the beam is
    marginal there: it wins at layer 119 with this ranking and does not win at
    all within 250 layers on the red count alone. At 40000 every ranking tried
    wins, and wins SHORTER -- 94 moves against 116 -- for about 35 s more
    search, paid once ever into the plan cache and replayed by every seed.

    Dedup is kept across the WHOLE search, not per depth, so a state re-reached
    later never re-expands. The trade is plan length; see `shorten`.
    """
    if model.won(pos0, mask0):
        return []
    frontier = [((pos0, mask0), [])]
    seen = {(pos0, mask0)}
    for _layer in range(depth):
        kids = []
        for (pos, mask), path in frontier:
            for di in range(4):
                nxt = model.step(pos, mask, di)
                if nxt is None or nxt in seen:
                    continue
                if model.won(*nxt):
                    return path + [MOVES[di]]
                seen.add(nxt)
                kids.append(((model.n_bridges - bin(nxt[1]).count("1"),
                              0 if nxt[0] == model.start else 1),
                             nxt, path + [MOVES[di]]))
        if not kids:
            return None                    # the reachable space closed
        kids.sort(key=lambda kid: kid[0])
        frontier = [(state, path) for _k, state, path in kids[:width]]
    return None


def shorten(model: Model, pos0: int, mask0: int, plan: list[str]) -> list[str]:
    """Delete contiguous blocks from ``plan``, longest first, keeping every
    deletion the model still wins with.

    A beam plan wanders -- it is the shortest path the beam happened to keep,
    not the shortest path -- and the wandering is usually a few excursions that
    lift straight out. Spans are tried large-to-small and a successful deletion
    re-scans the same span, so one pass costs O(len^3) model steps: a few
    seconds on the only plan that needs it, against a plan the agent has to
    replay on every seed.
    """
    plan = list(plan)
    span = len(plan) - 1
    while span >= 1:
        i = 0
        while i + span <= len(plan):
            cand = plan[:i] + plan[i + span:]
            won = model.replay(pos0, mask0, cand)
            if won:
                plan = cand[:won]
                span = min(span, max(len(plan) - 1, 1))
            else:
                i += 1
        span -= 1
    return plan


def solve(model: Model, pos0: int, mask0: int) -> list[str] | None:
    """Walk `LADDER` until a rung returns a plan."""
    for kind, dial, budget in LADDER:
        if kind == "astar":
            plan = astar(model, pos0, mask0, weight=dial, node_cap=budget)
        else:
            plan = beam(model, pos0, mask0, width=dial, depth=BEAM_DEPTH)
            if plan is not None:
                plan = shorten(model, pos0, mask0, plan)
        if plan is not None:
            return plan
    return None


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class BridgeExpert(PSExpert):
    """Plan on the model, then prove the plan on the real interpreter.

    `PSExpert`'s A* is replaced wholesale (see the module docstring on why the
    interpreter is too slow to search); what is inherited is the plan memo, the
    snapshot/restore discipline and the `record_level` contract that a plan is a
    flat list of engine directions. The base's ``node_cap`` / ``weight`` are
    unused -- `LADDER` carries both, per rung.
    """

    scope_by_level = True
    #: No rule in this game mentions ``action``, so ACTION5 is a dead key.
    directions = list(MOVES)

    def setup(self) -> None:
        self._disk = self._load_disk()

    # -- disk cache --------------------------------------------------------
    @staticmethod
    def _signature(model: Model, pos: int, mask: int) -> list:
        """Cheap fingerprint of a level's start layout. A cached plan is only
        served when this still matches, so editing Bridge-toggle_Maze.txt
        cannot silently replay a plan for a level that no longer exists."""
        return [model.H, model.W, pos, model.start, mask,
                [i for i, b in enumerate(model.bit) if b],
                [i for i, b in enumerate(model.bit) if b and model.vgreen[i]]]

    def _load_disk(self) -> dict:
        """``{level: {"start": signature, "plan": "<digits>"}}`` -- the plan is
        one digit per move, indexing `MOVES`. Empty if unreadable: a cache that
        cannot be parsed is a miss, never a crash."""
        try:
            raw = json.loads(PLAN_CACHE.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:                                      # noqa: BLE001
            return {}

    def _save_disk(self) -> None:
        """Write atomically: shards started together (`parallelize_generator`)
        would otherwise interleave into a truncated file."""
        try:
            PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PLAN_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self._disk[k] for k in sorted(self._disk)}, indent=1))
            os.replace(tmp, PLAN_CACHE)
        except OSError:
            pass                                              # cache is optional

    # -- planning ----------------------------------------------------------
    def plan(self, eng, level: int | None = None) -> list | None:
        key = (level, self._key(eng))
        if key in self.cache:
            return self.cache[key]
        sol = self._solve(eng, level)
        self.cache[key] = sol
        return sol

    def _solve(self, eng, level: int | None) -> list | None:
        model, pos, mask = read_state(eng, self.g)
        sig = self._signature(model, pos, mask)

        cached = self._disk.get(level)
        if cached is not None and cached.get("start") == sig:
            raw = cached.get("plan")
            if raw is None:
                # A level the ladder could not win. Only trusted while the
                # LADDER is the one that failed: a cached "no" must never be
                # what stops a strengthened search from being tried.
                if cached.get("ladder") == _ladder_id():
                    return None
            else:
                checked = self._verify(eng, [MOVES[int(ch)] for ch in raw])
                if checked is not None:
                    return checked
                # The cache disagrees with the interpreter -- search afresh.

        raw = solve(model, pos, mask)
        checked = None if raw is None else self._verify(eng, raw)
        self._remember(level, sig, checked)
        return checked

    def _remember(self, level, sig, plan) -> None:
        if level is None:
            return
        entry = {"start": sig,
                 "plan": (None if plan is None
                          else "".join(str(MOVES.index(d)) for d in plan))}
        if plan is None:
            entry["ladder"] = _ladder_id()
        self._disk[level] = entry
        self._save_disk()

    def _verify(self, eng, plan: list[str]) -> list[str] | None:
        """Replay ``plan`` on the real interpreter; return it truncated at the
        winning step, or ``None`` if the engine does not win with it. This is
        what keeps model drift from ever reaching the corpus.

        Leaves the engine exactly as it was -- including the win/restart FLAGS,
        which `ps_astar.restore` does not touch. `step` clears them on entry, so
        they only matter to a `check_win` asked before the next step, which is
        precisely what `record_level` does on its first iteration: leaving a
        winning flag behind there would end the recording before it took a
        single action."""
        snap = snapshot(eng)
        flags = (eng._rule_win, eng._rule_restart)
        won = -1
        for i, d in enumerate(plan):
            eng.step(d)
            if eng.check_win():
                won = i
                break
        restore(eng, snap)
        eng._rule_win, eng._rule_restart = flags
        return plan[:won + 1] if won >= 0 else None

    def heuristic(self, eng) -> int:            # pragma: no cover - unused
        raise NotImplementedError("BridgeExpert plans on the model, not the engine")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

#: Process-wide adapter + plan cache + solvable-level set. `test_datagen.py`
#: builds a fresh solver per run, which would otherwise re-pay every search.
_SHARED: dict = {}


class BridgeToggleMazeSolver(PSAStarSolver):
    game_id = "puzzlescript_bridge_toggle_maze"
    game_name = GAME_NAME
    expert_cls = BridgeExpert

    #: Level 6's plan is the long one; the ceiling also has to cover the
    #: RESET-recovery exploration prefix that runs before it.
    max_steps = 400

    def _ensure(self, seed: int):
        if self._game is None and "game" in _SHARED:
            self._game = _SHARED["game"]
            self._expert = _SHARED["expert"]
            self._solvable = _SHARED["solvable"]
        game, expert, solvable = super()._ensure(seed)
        _SHARED.update(game=game, expert=expert, solvable=solvable)
        return game, expert, solvable

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The optimal action set at plan step ``pi``: the plan's own move.

        Deriving a genuine TIE set would need the exact distance-to-win of each
        alternative successor, i.e. a second full search per step over a
        ``cells x 2^bridges`` space -- and on the weighted and beam levels the
        plan being labelled is not provably shortest anyway, so "as good as what
        the expert did" is the honest claim. What this override buys is that no
        expert step ships unlabelled: `train_policy` v2 supervises ``optimal``
        only, so a step without one contributes nothing to the loss."""
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Differential model test / report
# ---------------------------------------------------------------------------

def _engine_state(eng, gm) -> tuple:
    """The slice of the interpreter's grid the model claims to reproduce: where
    the player is, which cells are green, and which run north-south."""
    N = gm.obj_name_to_idx
    W = eng.width
    green_ids = {N["greenns"], N["greenew"]}
    ns_ids = {N["greenns"], N["redns"]}
    pos = None
    green, ns = set(), set()
    for r in range(eng.height):
        for c in range(W):
            i = r * W + c
            cell = eng.grid[r][c]
            if N["player"] in cell:
                pos = i
            if cell & green_ids:
                green.add(i)
            if cell & ns_ids:
                ns.add(i)
    return pos, tuple(sorted(green)), tuple(sorted(ns))


def _model_state(model: Model, pos: int, mask: int) -> tuple:
    green, ns = set(), set()
    for i, b in enumerate(model.bit):
        if not b:
            continue
        is_green = bool(mask & b)
        if is_green:
            green.add(i)
        if is_green == model.vgreen[i]:
            ns.add(i)
    return pos, tuple(sorted(green)), tuple(sorted(ns))


def _verify_model(episodes: int = 60, steps: int = 60) -> int:
    """Differential test: play randomly and compare the model to the
    interpreter after EVERY step, on every level.

    This is the test that built `Model`, and it is what caught the adapter's
    ellipsis bug (see the module docstring) rather than a reading of the rules.
    It is in the repo rather than in a scratch file because the model's fidelity
    is the whole safety argument for planning off the engine. Re-run it after
    any change to the adapter or to ``Bridge-toggle_Maze.txt``. Expected result:
    zero mismatches.
    """
    import random

    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    bad = 0
    for level in range(game.n_levels):
        for ep in range(episodes):
            rng = random.Random(level * 1000 + ep)
            game.set_level(level)
            eng = game._engine
            model, pos, mask = read_state(eng, game._game)
            for t in range(steps):
                di = rng.randrange(4)
                eng.step(MOVES[di])
                nxt = model.step(pos, mask, di)
                if nxt is not None:
                    pos, mask = nxt
                if (_engine_state(eng, game._game) != _model_state(model, pos, mask)
                        or eng.check_win() != model.won(pos, mask)):
                    print(f"  MISMATCH level {level} episode {ep} step {t} "
                          f"({MOVES[di]})")
                    bad += 1
                    break
                if eng.check_win():
                    break
        print(f"  level {level:2d}: checked {episodes} episodes")
    print("model matches the interpreter" if not bad
          else f"{bad} MISMATCHING episodes")
    return 1 if bad else 0


def _report() -> int:
    """Print per-level plan lengths -- the coverage check for this game."""
    solver = BridgeToggleMazeSolver()
    game, expert, solvable = solver._ensure(0)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        model, _pos, _mask = read_state(game._engine, game._game)
        plan = expert.plan(game._engine, level)
        tag = ("unsolved" if plan is None else f"{len(plan):3d} moves")
        if plan is not None:
            total += len(plan)
        print(f"  level {level:2d}: {model.H}x{model.W}, "
              f"{model.n_bridges:2d} bridges -> {tag}")
    print(f"{len(solvable)}/{game.n_levels} levels solved, {total} moves total")
    return 0


if __name__ == "__main__":
    if "--report" in sys.argv:
        sys.exit(_report())
    if "--verify-model" in sys.argv:
        sys.exit(_verify_model())
    sys.exit(BridgeToggleMazeSolver.main())
