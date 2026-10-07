"""Generate Phase-1 training data for the PuzzleScript game ps:doktor_lezer
("Doktor Lezer" by Ax -- a laser range, a handful of mirrors on castors, and
one doctor to burn with them).

The harness -- the rotation contract, the trajectory recorder, the
RESET-recovery prefix and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file
is the game-specific part: the mechanic notes, a native model of the
interpreter, the search over it, and the optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_doktor_lezer",
      "levels": [
        {
          "level_id": 0,
          "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
          "actions":      [a_0, a_1, ..., a_{T-1}], # length T
        },
        ...
      ]
    }

actions[0] is the RESET that produced obs[0]; actions[i] for i>=1 is the action
that took the agent from obs[i-1] to obs[i]. The recorded index is the *screen*
action (post rotation remap), i.e. the button an agent presses in the presented
view, so replaying the recorded actions reproduces the recorded frames exactly.

The game
--------
Win is ``no target`` and ``no playerDead``: a beam has to land on the doctor
on a turn the player is NOT standing in one.

  * **THE PLAYER IS A TANK.** ``[up player no playerU no grabbing] ->
    [playerU]`` turns it and cancels the move, so a direction it is not
    already facing costs a press just to face. That single rule is most of
    what these plans look like -- almost nothing commutes, and "walk right
    then up" and "walk up then right" are different lengths unless the player
    happens to be facing right.

  * **GRAB AND DRAG IS THE ONLY THING THAT MOVES ANYTHING.** ACTION grabs the
    mirror in the square the player faces (and a second ACTION lets go); while
    it is held, the turn rules are suppressed and ``[moving player grabbing |
    grabbed movable]`` copies the player's force onto the mirror. Both move by
    the same delta, so **the offset between player and mirror never changes**
    -- which is why `Board` does not store the held mirror at all, it is
    always at ``player + facing``. Nothing else on the board can be moved: no
    emitter, no wall, no doctor.

  * **A BLOCKED DRAG CANCELS THE WHOLE TURN**, by the two ``-> cancel`` rules,
    so pushing a mirror into a wall/emitter/doctor/another mirror is a press
    that changes nothing at all.

  * **THE BEAM IS REDRAWN FROM SCRATCH EVERY TURN.** Each emitter re-emits and
    the ray walks until a wall or another emitter stops it. ``/`` (mirrorZ)
    swaps up<->right and down<->left, ``\\`` (mirrorS) swaps up<->left and
    down<->right, and a mirror's diagonal is FIXED -- the puzzle is where each
    one stands, never which way it leans. The doctor and the player do not
    block light; they burn in it.

  * **BEAMS CROSS FREELY.** Every one of the eight laser/emit objects sits on
    its own collision layer -- in a PuzzleScript ``COLLISIONLAYERS`` block one
    LINE is one layer, and the ``(')`` / ``(- )`` tails in this .txt are
    comments, not layer-mates -- so two beams share a square happily. That is
    what the later levels, built around head-on emitters, are made of, and it
    is what lets the whole late-rule laser block be a plain ray trace.

  * **DEATH IS INSTANT AND FINAL.** ``late [laser player] -> [laser
    playerDead]`` deletes the player object, so `check_game_over` goes true on
    that very press and the adapter refuses every action afterwards (the
    ``[playerDead] -> restart`` rule only raises the restart flag on the NEXT
    press, which never comes). Burning the doctor and the player on the same
    turn is therefore a loss, not a win.

Reading the frame
-----------------
Blue floor, grey wall, white-and-green mirrors, red beams, a blue emitter
block per laser, the doctor in his brown coat. A beam is drawn as two half
sprites -- the ``laser`` half it arrives on and the ``emit`` half it leaves on
-- so a square light merely passes through is a full bar, and the last square
of a beam is a half one. ``--audit`` checks all 30 cell compositions render
distinctly at both cell sizes the levels use (5 px and 7 px); they do, with no
recolour needed, so ``games/ps:doktor_lezer`` stays a plain passthrough.

Coverage
--------
**Fourteen of the sixteen levels solve**, 5 to 59 presses, 280 presses in all,
every plan replayed through the real interpreter to a WIN. Thirteen of the
fourteen are answered by the ladder's uniform-cost rung, so those plans are
PROVEN shortest; level 9 (59 presses) is the shortest found. Levels 10 and 11
are unwinnable rather than hard -- see `DoktorLezerSolver.skip_levels`. A cold
build searches everything in ~80 s and caches it to
``data/doktor_lezer_plans.json``; warm startup is instant.

Verified: ``--selfcheck`` runs 640 random rollouts (every level, 60 presses
each, ~450 deaths and ~5000 frozen presses among them) with ZERO divergences
from the interpreter on state, cancels, lit cells, win and death; every plan
replays through the interpreter to a WIN; and every recorded episode replays
frame-for-frame from its recorded SCREEN presses on a fresh adapter, ending in
``GameState.WIN`` on all 14 levels. ``test_datagen.py doktor_lezer`` gives 3/3
with 3 distinct trajectories.

**Augmentation is rotation ONLY** (the game is in neither
`PuzzleScriptAdapter._RECOLOR_GAMES` nor `_FLIP_GAMES`). A recolour would be
risky -- red is the beam, and reading the board is reading where the red goes.
A FLIP would be perfectly sound and is simply not enabled: there is no gravity,
input is screen-relative, and ``/`` and ``\\`` are drawn as mirror images of
each other AND reflect as mirror images, so a flipped ``/`` both looks and
bounces exactly like a ``\\``. That is where to go for more presentation
variety; the plans themselves are near-unique (see `DoktorExpert._optsets`),
so per-seed variety comes from the orientation and the exploration prefix.

Usage
-----
    python solvers/generate_doktor_lezer_training.py --episodes 200 \\
        --out data/training_multi_level/doktor_lezer

    --selfcheck   fuzz `Board` against the real interpreter
    --audit       check every cell composition renders distinctly
    --plans       print (and interpreter-verify) every level's plan
"""

from __future__ import annotations

import heapq
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                          # noqa: E402
    Plan, PSAStarSolver, PSExpert)
from adapters.puzzlescript_adapter import PuzzleScriptAdapter  # noqa: E402

GAME_NAME = "Doktor_Lezer"

PLAN_CACHE = (Path(__file__).resolve().parent.parent
              / "data" / "doktor_lezer_plans.json")

#: The five live keys, in engine-direction form.
KEYS = ("up", "down", "left", "right", "action")

#: (dr, dc) per direction index -- the order KEYS' first four are in.
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Mirror types.
MZ, MS = 0, 1          # '/' and '\'

INF = 1 << 30

#: Cache sentinel -- `Heuristic.field` stores None for "dead", so a plain
#: ``get(key)`` cannot tell a miss from a cached dead state.
_MISSING = object()


def _reflect(kind: int, d: int) -> int:
    """The direction a beam travelling ``d`` leaves a mirror of ``kind``.

    ``mirrorZ`` ('/') swaps up<->right and down<->left, which on this
    direction encoding (0 up, 1 down, 2 left, 3 right) is ``3 - d``;
    ``mirrorS`` ('\\') swaps up<->left and down<->right, which is ``d ^ 2``.
    """
    return (3 - d) if kind == MZ else (d ^ 2)


# ---------------------------------------------------------------------------
# Native model of the interpreter
# ---------------------------------------------------------------------------

class Board:
    """Doktor Lezer's rules as a pure function on immutable states.

    A state is ``(player, facing, grabbing, mZ, mS)``: the player's cell index
    (``r * w + c``), the direction it faces, whether it is holding a mirror,
    and the two mirror bitmasks. The grabbed mirror is never stored separately
    -- a grab only ever happens in the facing direction and the turn rules
    cannot fire while ``grabbing``, so the held mirror is always at
    ``player + facing`` and both move by the same delta.

    The scenery (walls, emitters, the target) is a level constant.
    """

    def __init__(self, h: int, w: int, walls: int, pews: dict, target: int):
        self.h, self.w = h, w
        self.n = h * w
        self.pews = pews                      # cell -> direction
        self.target = target
        pew_mask = 0
        for cell in pews:
            pew_mask |= 1 << cell
        #: A beam dies on a wall or an emitter; everything else it passes
        #: through (mirrors deflect it, the target and the player are burned).
        self.laser_block = walls | pew_mask
        #: ``playerObstacle`` minus the mirrors, which move: what neither the
        #: player nor a dragged mirror may ever enter. ``target < 0`` is the
        #: already-burned board the fuzz walks into, never a playable one.
        self.static_block = walls | pew_mask
        if target >= 0:
            self.static_block |= 1 << target
        self.mv = [[-1] * 4 for _ in range(self.n)]
        for cell in range(self.n):
            r, c = divmod(cell, w)
            for d, (dr, dc) in enumerate(_DELTA):
                if 0 <= r + dr < h and 0 <= c + dc < w:
                    self.mv[cell][d] = (r + dr) * w + (c + dc)
        self._beams: dict = {}
        self.dist = self._all_pairs()

    # -- geometry -------------------------------------------------------------
    def _all_pairs(self) -> list:
        """``dist[a][b]`` -- steps between two cells over the STATIC scenery
        only (mirrors ignored, beams ignored). The heuristic's only distance
        table: optimistic by construction, and built once per level."""
        out: list = [None] * self.n
        for src in range(self.n):
            if (self.static_block >> src) & 1:
                continue
            seen = {src: 0}
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for nxt in self.mv[cur]:
                    if nxt >= 0 and not (self.static_block >> nxt) & 1 \
                            and nxt not in seen:
                        seen[nxt] = seen[cur] + 1
                        queue.append(nxt)
            out[src] = seen
        return out

    # -- lasers ---------------------------------------------------------------
    def beams(self, mZ: int, mS: int) -> int:
        """Bitmask of the lit cells for a mirror layout, memoized.

        The eight laser/emit objects are each on their OWN collision layer (in
        the .txt every layer is one line; the ``(')`` / ``(- )`` tails are
        comments), so beams cross each other freely and two beams may share a
        cell -- which is what the later levels, built around emitters aimed
        straight at each other, are made of. That is what lets the whole
        late-rule laser block be a plain ray trace.
        """
        key = (mZ, mS)
        got = self._beams.get(key)
        if got is None:
            got = self._trace(mZ, mS)
            self._beams[key] = got
        return got

    def _trace(self, mZ: int, mS: int) -> int:
        lit = 0
        seen = set()
        stack = list(self.pews.items())
        block, mv = self.laser_block, self.mv
        while stack:
            state = stack.pop()
            if state in seen:               # a mirror loop: the rules stop
                continue                    # when no new laser cell appears
            seen.add(state)
            x, d = state
            y = mv[x][d]
            if y < 0 or (block >> y) & 1:
                continue
            lit |= 1 << y
            if (mZ >> y) & 1:
                nd = _reflect(MZ, d)
            elif (mS >> y) & 1:
                nd = _reflect(MS, d)
            else:
                nd = d
            stack.append((y, nd))
        return lit

    # -- the turn -------------------------------------------------------------
    def step(self, st, k: int):
        """One press. ``None`` when nothing at all happens -- a turn the rules
        cancel outright, a move the collision layer refuses, an ACTION with no
        mirror in front of the player.

        The .txt's own order: grab / let go, then turn (which cancels the move,
        so a press the player is not already facing only turns it), then the
        drag rule that copies the player's force onto the held mirror, then the
        two cancel rules that reject a drag into an obstacle.
        """
        p, f, grab, mZ, mS = st
        if k == 4:                                    # ACTION
            if grab:
                return (p, f, 0, mZ, mS)              # let go
            m = self.mv[p][f]
            if m >= 0 and ((mZ | mS) >> m) & 1:
                return (p, f, 1, mZ, mS)              # grab what it faces
            return None
        if not grab:
            if f != k:
                return (p, k, 0, mZ, mS)              # turn in place
            nxt = self.mv[p][k]
            if nxt < 0 or (self.static_block >> nxt) & 1 \
                    or ((mZ | mS) >> nxt) & 1:
                return None
            return (nxt, k, 0, mZ, mS)
        # Dragging: the player and the held mirror both take the pressed
        # direction, so their offset never changes.
        m = self.mv[p][f]
        p2, m2 = self.mv[p][k], self.mv[m][k]
        if p2 < 0 or m2 < 0:
            return None
        others = (mZ | mS) & ~(1 << m)
        if (self.static_block >> m2) & 1 or (others >> m2) & 1:
            return None                               # [> grabbed | obstacle]
        if p2 != m and ((self.static_block >> p2) & 1 or (others >> p2) & 1):
            return None                               # [> grabbing | obstacle]
        if (mZ >> m) & 1:
            mZ = (mZ & ~(1 << m)) | (1 << m2)
        else:
            mS = (mS & ~(1 << m)) | (1 << m2)
        return (p2, f, 1, mZ, mS)

    # -- verdicts -------------------------------------------------------------
    def verdict(self, st) -> int:
        """0 = playable, 1 = won, -1 = the player has been burned.

        ``no playerDead`` is half the win condition, so a beam that lands on
        the doctor and the player in the same turn is a LOSS -- and a fatal
        one: the player object is replaced by a corpse, `check_game_over` goes
        true on that very press and the adapter refuses every action after it.
        """
        lit = self.beams(st[3], st[4])
        if (lit >> st[0]) & 1:
            return -1
        return 1 if (lit >> self.target) & 1 else 0


# ---------------------------------------------------------------------------
# Reading a model state out of the interpreter
# ---------------------------------------------------------------------------

_FACINGS = ("playeru", "playerd", "playerl", "playerr")


def read_scene(eng, game) -> tuple:
    """``(h, w, walls, pews, target, state)`` for the interpreter's current
    grid: the level constants and the model state, read separately so a caller
    that only wants the state (the fuzz, once per press) does not pay for a
    `Board` and its distance table.

    A grid whose doctor has already burned comes back with ``target = -1``.
    That is a finished board, not a playable one -- `Board.verdict` is not
    defined on it -- and only the fuzz ever sees it.
    """
    idx = game.obj_name_to_idx
    h, w = len(eng.grid), len(eng.grid[0])
    walls = mZ = mS = 0
    pews: dict = {}
    target = player = -1
    facing, grab = 0, 0
    pew_dirs = {idx["pewu"]: 0, idx["pewd"]: 1, idx["pewl"]: 2, idx["pewr"]: 3}
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if not cell:
                continue
            here = r * w + c
            if idx["wall"] in cell:
                walls |= 1 << here
            if idx["mirrorz"] in cell:
                mZ |= 1 << here
            if idx["mirrors"] in cell:
                mS |= 1 << here
            if idx["target"] in cell:
                target = here
            for pid, d in pew_dirs.items():
                if pid in cell:
                    pews[here] = d
            for d, name in enumerate(_FACINGS):
                if idx[name] in cell:
                    player, facing = here, d
            if idx["grabbing"] in cell:
                grab = 1
    return h, w, walls, pews, target, (player, facing, grab, mZ, mS)


def read_board(eng, game) -> tuple:
    """``(Board, state)`` for the interpreter's current grid."""
    h, w, walls, pews, target, state = read_scene(eng, game)
    return Board(h, w, walls, pews, target), state


def _cells(mask: int):
    out = []
    while mask:
        bit = mask & -mask
        mask ^= bit
        out.append(bit.bit_length() - 1)
    return out


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

class Heuristic:
    """Presses-to-go estimate, built on a relaxed count of the mirror moves
    the beam still needs.

    `field` answers the routing question -- how many mirrors have to be put
    somewhere (or taken away) for a beam to reach the doctor -- by a 0-1 BFS
    over beam states ``(cell, direction)``: following the board as it stands
    is free, and bending the beam at a cell it passes through costs one
    "operation". That count is what makes the middle of a level rank at all;
    the raw board tells the search nothing until the beam already lands.

    The estimate then prices the CHEAPEST of those operations properly -- walk
    to a mirror of the right diagonal, grab it, drag it to the cell the route
    wants -- and charges a flat `BASE` for each of the others. It is a search
    guide, not a bound (the drag distance ignores the other mirrors and the
    beams, and `BASE` is a guess), so plans are the shortest FOUND.
    """

    #: Presses charged per relaxed operation past the one being priced.
    BASE = 8

    def __init__(self, board: Board):
        self.b = board
        self._fields: dict = {}

    def field(self, mZ: int, mS: int):
        key = (mZ, mS)
        got = self._fields.get(key, _MISSING)
        if got is _MISSING:
            got = self._field(mZ, mS)
            self._fields[key] = got
        return got

    def _field(self, mZ: int, mS: int):
        """``(ops, [(cell, kind), ...])`` -- the number of relaxed operations
        a beam still needs and, for each of them, where and what. ``kind`` is
        `MZ` / `MS` for "a mirror of this diagonal has to be here" and None
        for "the mirror here has to go". None when no routing exists at all,
        which is a dead state.
        """
        b = self.b
        mirrors = mZ | mS
        dist = {}
        parent: dict = {}
        queue: deque = deque()
        for cell, d in b.pews.items():
            state = cell * 4 + d
            if state not in dist:
                dist[state] = 0
                parent[state] = None
                queue.append(state)
        goal = None
        while queue:
            state = queue.popleft()
            cost = dist[state]
            if goal is not None and cost > dist[goal]:
                break
            x, d = divmod(state, 4)
            y = b.mv[x][d]
            if y < 0 or (b.laser_block >> y) & 1:
                continue
            if y == b.target:
                if goal is None or cost < dist.get(goal, INF):
                    goal = state
                continue
            edges = []
            if (mirrors >> y) & 1:
                kind = MZ if (mZ >> y) & 1 else MS
                edges.append((_reflect(kind, d), 0, None))
                edges.append((_reflect(1 - kind, d), 1, (y, 1 - kind)))
                edges.append((d, 1, (y, None)))
            else:
                edges.append((d, 0, None))
                edges.append((_reflect(MZ, d), 1, (y, MZ)))
                edges.append((_reflect(MS, d), 1, (y, MS)))
            for nd, price, op in edges:
                nxt = y * 4 + nd
                ncost = cost + price
                if ncost >= dist.get(nxt, INF):
                    continue
                dist[nxt] = ncost
                parent[nxt] = (state, op)
                (queue.appendleft if price == 0 else queue.append)(nxt)
        if goal is None:
            return None
        ops = []
        cur = goal
        while parent[cur] is not None:
            prev, op = parent[cur]
            if op is not None:
                ops.append(op)
            cur = prev
        return len(ops), ops

    def __call__(self, st):
        """Estimated presses to a win, or None when the state is PROVABLY
        dead.

        The only thing that returns None is a board `field` cannot route a
        beam across at all, and that verdict is sound: every real winning
        layout draws the beam as straight segments joined by 90-degree turns
        at mirrors, which is exactly a path in `field`'s relaxed graph (where
        a mirror may be conjured at any cell the beam crosses). So no path
        there means no layout here. Everything else -- including "the route I
        priced wants a diagonal this level does not own" -- comes back as a
        large finite number, because a costlier route may still exist and a
        search that pruned on it would lose real solutions.
        """
        p, f, grab, mZ, mS = st
        got = self.field(mZ, mS)
        if got is None:
            return None
        n_ops, ops = got
        if n_ops == 0:
            return 0
        b = self.b
        held = b.mv[p][f] if grab else -1
        table_p = b.dist[p] or {}
        best = INF
        for cell, kind in ops:
            if kind is None:                    # take the mirror at `cell` away
                cands = [cell]
            else:
                cands = _cells(mZ if kind == MZ else mS)
            for m in cands:
                drag = 1 if kind is None else (b.dist[m] or {}).get(cell, INF)
                if drag >= INF:
                    continue
                if m == held:
                    cost = drag
                else:
                    walk = table_p.get(m, INF)
                    if walk >= INF:
                        continue
                    # -1: the player grabs from BESIDE the mirror, +1 for the
                    # ACTION that grabs it, +1 for the one that lets go when
                    # this is not the last operation.
                    cost = max(0, walk - 1) + 1 + drag + (1 if grab else 0)
                if n_ops > 1:
                    cost += 1
                if cost < best:
                    best = cost
        if best >= INF:
            return n_ops * self.BASE
        return best + (n_ops - 1) * self.BASE


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class DoktorExpert(PSExpert):
    """Weighted A* over `Board`, plugged into `PSExpert` as its search
    strategy. The interpreter is 2.5-15 ms per press here (the laser rules
    rebuild every beam every turn, and the `+` group runs to its iteration
    cap), so nothing but the finished plan touches it."""

    directions = list(KEYS)
    plan_cache_path = PLAN_CACHE

    #: (weight, edge cap) rungs, tried in order; one edge is ~5 us.
    #:
    #: THE FIRST RUNG IS ``weight = 0``, i.e. a plain uniform-cost search, and
    #: it is there because it PROVES the plan shortest -- `Heuristic` is a
    #: guide, not a bound, so a weighted rung can and does return a longer
    #: plan (levels 2, 3 and 4 came back 2-3 presses long under ``w = 1``).
    #: Thirteen of the fourteen solvable levels are answered by it in well
    #: under a second, so the guarantee is nearly free. Level 9 is the one
    #: that is not: its optimum is around 59 presses and a UCS does not reach
    #: that depth in 20M edges (27 s, 2 GB), so it falls through to ``w = 1``,
    #: which finds 59 in 21 s. That plan is the shortest FOUND, not proven
    #: shortest.
    ladder = ((0, 3_000_000), (1, 4_000_000), (3, 2_000_000), (8, 2_000_000))

    def setup(self) -> None:
        self._how: dict = {}
        self._level: int | None = None

    # -- search ---------------------------------------------------------------
    @staticmethod
    def _astar(board: Board, start: tuple, h: Heuristic, weight: int,
               cap: int):
        """Weighted A* over `Board`, or a uniform-cost search at
        ``weight == 0``. Returns the press sequence, or None when the budget
        runs out OR the reachable space closes with no win in it -- the caller
        cannot tell those apart, which is why the ladder's rungs are tried in
        order and `DoktorLezerSolver.skip_levels` records the levels where the
        second one is what happened.

        Parent pointers rather than a path per heap entry: at a few million
        nodes, a copied tuple per node is most of the memory. A burned player
        is dropped rather than queued (there is no move back from it), and so
        is a state `Heuristic` can prove has no beam routing left at all.
        """
        if board.verdict(start) == 1:
            return ()
        best = {start: 0}
        parent: dict = {start: None}
        counter = 0
        nodes = 0
        h0 = h(start)
        if h0 is None:
            return None
        pq = [(weight * h0, 0, 0, start)]
        while pq:
            _f, g, _c, st = heapq.heappop(pq)
            if best.get(st, -1) != g:
                continue
            for k in range(5):
                nxt = board.step(st, k)
                if nxt is None:
                    continue
                nodes += 1
                verdict = board.verdict(nxt)
                if verdict == 1:
                    out = [k]
                    cur = st
                    while parent[cur] is not None:
                        cur, pk = parent[cur]
                        out.append(pk)
                    out.reverse()
                    return tuple(out)
                if verdict < 0:
                    continue                       # burned: a terminal loss
                ng = g + 1
                if best.get(nxt, INF) <= ng:
                    continue
                hv = h(nxt)
                if hv is None:
                    continue                       # no routing survives here
                best[nxt] = ng
                parent[nxt] = (st, k)
                counter += 1
                heapq.heappush(pq, (ng + weight * hv, ng, counter, nxt))
            if nodes >= cap:
                return None
        return None

    def _search(self, eng):
        """`PSExpert`'s strategy hook: read a model off the interpreter's
        current grid and search THAT, returning a `Plan` (presses plus their
        optimal sets) for the shared memo and disk cache to round-trip. The
        snapshot/restore discipline around this call is the base class's."""
        board, start = read_board(eng, self.g)
        h = Heuristic(board)
        for weight, cap in self.ladder:
            path = self._astar(board, start, h, weight, cap)
            if path is not None:
                self._how[self._level] = f"A* w={weight}"
                return Plan([KEYS[k] for k in path],
                            self._optsets(board, h, start, path))
        self._how[self._level] = "unsolved"
        return None

    # -- optimal sets ---------------------------------------------------------
    #: Edge budget for one alternative-press probe. A probe that runs out
    #: simply does not widen that step's set, so this is a speed dial and not
    #: a correctness one.
    probe_cap = 80_000

    @classmethod
    def _optsets(cls, board: Board, h: Heuristic, start: tuple,
                 path: tuple) -> list:
        """Per-step optimal sets: every press at least as good as the one the
        expert took.

        A press ``k`` qualifies when a win exists within one press FEWER than
        the expert still needs, checked by running a uniform-cost search (the
        ladder's own first rung, so the answer is a true distance and not a
        weighted guess) from the state ``k`` leads to. That is exact whenever
        the probe finishes inside `probe_cap`; where it does not, the step
        keeps the expert's press as its only label -- conservative, never
        wrong. Level 9 is where that bites: nothing uniform-cost reaches its
        depth, so its steps are labelled singly.

        No cheaper commutation test would do here. The turn rule means almost
        nothing commutes: a press the player is not already facing only turns
        it, so "walk right two then up three" and any interleaving of it are
        NOT the same length, and a plan's order is mostly forced. The ties
        that do exist are real choices -- which axis to walk first, which side
        of a mirror to grab it from, which of two symmetric mirrors to move --
        and only a distance-to-win test finds those.

        A press the rules freeze (into a wall, or ACTION with nothing in
        front) is never in the set: it burns a turn and changes nothing.
        """
        states = [start]
        for k in path:
            states.append(board.step(states[-1], k))
        out = []
        for i, taken in enumerate(path):
            remaining = len(path) - i
            best = {taken}
            for k in range(5):
                if k == taken:
                    continue
                nxt = board.step(states[i], k)
                if nxt is None:
                    continue
                verdict = board.verdict(nxt)
                if verdict < 0:
                    continue
                if verdict == 1:
                    best.add(k)
                    continue
                if remaining <= 1:
                    continue
                sub = cls._astar(board, nxt, h, 0, cls.probe_cap)
                if sub is not None:
                    if len(sub) <= remaining - 1:
                        best.add(k)
                elif cls._pull_forward_wins(board, states[i], path[i:], k):
                    best.add(k)
            out.append([KEYS[k] for k in sorted(best)])
        return out

    @staticmethod
    def _pull_forward_wins(board: Board, state: tuple, suffix: tuple,
                           k: int) -> bool:
        """Cheap SUFFICIENT test for "``k`` is as good as the expert's press",
        used only where the exact probe ran out of budget.

        Take ``k`` now and then replay the rest of the plan with its own first
        occurrence of ``k`` deleted: if that still wins, it wins in at most as
        many presses as the plan had left, which is all the label claims. It
        proves nothing when it fails -- that is what makes it a fallback and
        not the test.
        """
        if k not in suffix:
            return False
        rest = list(suffix)
        rest.pop(rest.index(k))
        for press in [k] + rest:
            nxt = board.step(state, press)
            if nxt is not None:
                state = nxt
            verdict = board.verdict(state)
            if verdict == 1:
                return True
            if verdict < 0:
                return False
        return False

    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan`, remembering which level it is planning.

        The base does the work: the in-memory memo and (because
        `plan_cache_path` is set) the on-disk cache of each level's START
        plan, which is the only state any seed ever plans from -- recovery is
        a RESET back to it. All this override adds is the level number
        `describe` reports against."""
        self._level = level
        return super().plan(eng, level)

    def describe(self, level) -> str:
        """Which rung answered for this level (for ``--plans``); "cached" when
        the plan came off disk and nothing ran."""
        return self._how.get(level, "cached")


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class DoktorLezerSolver(PSAStarSolver):
    """`PSAStarSolver` for Doktor Lezer. Recovery is the family default: an
    epsilon-decayed exploration prefix, then ONE RESET back to the level start
    -- which is the state the cached plan was solved from, and the only state
    it is valid from. That matters more here than in most of these games,
    because the flail can leave a mirror somewhere the plan does not expect,
    and (unlike a death) that is perfectly survivable and perfectly wrong."""

    game_id = "puzzlescript_doktor_lezer"
    game_name = GAME_NAME
    expert_cls = DoktorExpert

    #: The two levels with NO mirrors on them. They are not hard, they are
    #: unwinnable: the only thing the player can move in this game is a
    #: mirror, so a board without one is frozen for good, and on both of these
    #: the doctor sits in the one row and column no emitter points along. The
    #: search agrees for the right reason -- with nothing to push around, the
    #: reachable state space is a few hundred (cell, facing) pairs and the
    #: frontier EXHAUSTS instantly, which is a proof and not a timeout. They
    #: are named here so that verdict is recorded rather than rediscovered at
    #: every cold start.
    skip_levels: frozenset[int] = frozenset({10, 11})

    #: Plans run to 59 presses and the adapter GAME_OVERs a level at 200
    #: counted moves, so neither cap is the binding one.
    max_steps = 200

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup
        rather than as a mysteriously slow first seed."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)

    def optimal_for(self, expert, level: int, plan, pi: int):
        """The optimal press set at this step, off the plan's own optsets.
        Falls back to the press about to be taken so no expert step ever ships
        unlabelled -- `train_policy` v2 supervises ``optimal`` only, so a step
        without one contributes nothing to the loss."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets) and sets[pi]:
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Model self-check (fuzz against the real interpreter)
# ---------------------------------------------------------------------------

def _engine_state(eng, game) -> tuple:
    return read_scene(eng, game)[-1]


def _engine_lit(eng, game) -> int:
    idx = game.obj_name_to_idx
    ids = [idx["laseru"], idx["laserd"], idx["laserl"], idx["laserr"]]
    w = len(eng.grid[0])
    lit = 0
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if any(i in cell for i in ids):
                lit |= 1 << (r * w + c)
    return lit


def selfcheck(trials: int = 40, steps: int = 60, verbose: bool = True) -> int:
    """Audit the claim every plan rests on: `Board` reproduces the
    interpreter exactly -- the settled state, the cancel decision, the lit
    cells, the win and the death.

    Random play is the fuzz: it drags mirrors into walls and into each other,
    walks the player through beams it is about to create, and grabs and lets
    go in places no plan would.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, parsed = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        cancels = deaths = wins = 0
        for t in range(trials):
            game.set_level(level)
            board, state = read_board(eng, parsed)
            rng = random.Random(f"doktor:{level}:{t}")
            for _ in range(steps):
                k = rng.randrange(5)
                before = [[set(c) for c in row] for row in eng.grid]
                eng.step(KEYS[k])
                frozen = eng.grid == before
                predicted = board.step(state, k)
                expected = state if predicted is None else predicted
                verdict = board.verdict(expected)
                # Death is checked in BOTH directions, and the second one
                # matters as much as the first: the search DROPS every state
                # the model calls dead, so a model that invents a death loses
                # real solutions just as quietly as one that misses a real
                # death produces plans that cannot replay.
                corpse = parsed.obj_name_to_idx["playerdead"] in {
                    o for row in eng.grid for cell in row for o in cell}
                if corpse != (verdict < 0):
                    bad += 1
                    print(f"  L{level}: death divergence on {KEYS[k]}: "
                          f"engine {corpse}, model {verdict < 0}")
                    break
                if corpse:
                    # A burned player ends the rollout: the engine has replaced
                    # the objects the reader looks for, so there is no state
                    # left to compare below.
                    deaths += 1
                    break
                actual = _engine_state(eng, parsed)
                if actual != expected:
                    bad += 1
                    print(f"  L{level}: state divergence on {KEYS[k]}: "
                          f"model {expected} engine {actual}")
                    break
                if predicted is None and not frozen:
                    bad += 1
                    print(f"  L{level}: the model froze {KEYS[k]} but the "
                          f"interpreter changed the grid")
                    break
                if board.beams(expected[3], expected[4]) != _engine_lit(
                        eng, parsed):
                    bad += 1
                    print(f"  L{level}: beam divergence after {KEYS[k]}")
                    break
                if eng.check_win() != (board.verdict(expected) == 1):
                    bad += 1
                    print(f"  L{level}: win divergence on {KEYS[k]}")
                    break
                if predicted is None:
                    cancels += 1
                state = expected
                if eng.check_win():
                    wins += 1
                    break
        if verbose:
            print(f"  L{level}: {'OK' if not bad else 'VIOLATIONS'} "
                  f"({trials} rollouts, {cancels} frozen presses, "
                  f"{deaths} deaths, {wins} accidental wins)")
        if bad:
            break
    return bad


# ---------------------------------------------------------------------------
# Render audit
# ---------------------------------------------------------------------------

#: Every cell composition a settled frame of this game can show, as the object
#: stack that draws it. ``target`` + a laser is deliberately absent: the doctor
#: is replaced by a corpse on the very turn a beam touches it, and that turn
#: is the win, so no frame ever shows a lit doctor.
_AUDIT_CASES = {
    "floor": ["background"],
    "wall": ["background", "wall"],
    "wall lip": ["background", "wall", "wallside"],
    "mirror /": ["background", "mirrorz"],
    "mirror \\": ["background", "mirrors"],
    "mirror / lit": ["background", "mirrorz", "laseru", "emitr", "frontr"],
    "mirror \\ lit": ["background", "mirrors", "laseru", "emitl", "frontl"],
    "doctor": ["background", "target"],
    "doctor head": ["background", "targeth"],
    "emitter up": ["background", "pewu", "emitu"],
    "emitter down": ["background", "pewd", "emitd"],
    "emitter left": ["background", "pewl", "emitl"],
    "emitter right": ["background", "pewr", "emitr"],
    "player up": ["background", "playeru"],
    "player down": ["background", "playerd"],
    "player left": ["background", "playerl"],
    "player right": ["background", "playerr"],
    "player up holding": ["background", "playeru", "grabbing"],
    "player down holding": ["background", "playerd", "grabbing"],
    "player left holding": ["background", "playerl", "grabbing"],
    "player right holding": ["background", "playerr", "grabbing"],
    "beam passing up": ["background", "laseru", "emitu"],
    "beam passing down": ["background", "laserd", "emitd"],
    "beam passing left": ["background", "laserl", "emitl"],
    "beam passing right": ["background", "laserr", "emitr"],
    "beam crossing": ["background", "laseru", "emitu", "laserr", "emitr"],
    "beam end up": ["background", "laseru"],
    "beam end down": ["background", "laserd"],
    "beam end left": ["background", "laserl"],
    "beam end right": ["background", "laserr"],
}

#: Compositions the game's own art draws IDENTICALLY, on purpose. A beam is
#: two half sprites -- the ``laser`` half it arrives on and the ``emit`` half
#: it leaves on -- so a square light merely passes THROUGH is one full bar
#: whichever way the light is travelling. Which way it travels is read off the
#: emitter at the end of the line, exactly as a human plays it, and no
#: decision in this game depends on telling the two apart. (The last square of
#: a beam is a different matter and is NOT excused here: it is only the half
#: sprite, so "beam end up" and "beam end down" are the two halves of the bar
#: and the audit does require them to differ.)
_AUDIT_TWINS = ({"beam passing up", "beam passing down"},
                {"beam passing left", "beam passing right"})


def audit(verbose: bool = True) -> int:
    """Check that every cell composition is pixel-distinct at every cell size
    the levels render at (5 px for the 11- and 12-row boards, 7 px for the
    9-row ones). Sprites are 5x5, so nothing is decimated here -- what this
    catches is a PALETTE collision, two different colours landing on the same
    ARC index and quietly deleting a piece from the frame."""
    from adapters.puzzlescript_adapter import _render_cell_sprite

    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    parsed = game._game
    layers = game._engine._obj_layers
    sizes = set()
    for level in range(game.n_levels):
        game.set_level(level)
        h, w = len(game._engine.grid), len(game._engine.grid[0])
        sizes.add(max(1, min(64 // h, 64 // w)))

    def stack(names):
        objs = [(layers.get(parsed.obj_name_to_idx[n], -1), parsed.objects[n])
                for n in names]
        objs.sort(key=lambda pair: pair[0])
        return objs

    def twinned(a, b) -> bool:
        return any(a in group and b in group for group in _AUDIT_TWINS)

    bad = 0
    names = list(_AUDIT_CASES)
    for px in sorted(sizes):
        blocks = {k: _render_cell_sprite(stack(v), px)
                  for k, v in _AUDIT_CASES.items()}
        hits = 0
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if (blocks[a] == blocks[b]).all() and not twinned(a, b):
                    hits += 1
                    print(f"  cell_px={px}: {a} and {b} render identically")
        bad += hits
        if verbose:
            print(f"  cell_px={px}: {len(names)} cell types, "
                  f"{'all distinct' if not hits else 'COLLISIONS'}")
    return bad


def _plan_report() -> None:
    solver = DoktorLezerSolver()
    game, expert, solvable = solver._ensure(0)
    total = ties = 0
    for level in range(game.n_levels):
        if level in solver.skip_levels:
            print(f"  L{level}: skipped")
            continue
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            continue
        for direction in plan:
            eng.step(direction)
            if eng.check_win():
                break
        step_ties = sum(len(s) - 1 for s in plan.optsets)
        total += len(plan)
        ties += step_ties
        print(f"  L{level:2d}: {len(plan):3d} presses  win={eng.check_win()}  "
              f"{expert.describe(level):>10}  {step_ties:3d} tie-presses")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second right answer")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit()
        print(f"audit: {collisions} colliding cell types")
        sys.exit(1 if collisions else 0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(DoktorLezerSolver.main())
