"""Generate Phase-1 training data for the PuzzleScript game ps:brotherhood
("Brotherhood" by Son Nguyen).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other search-the-interpreter ps: generators. This file is the game-specific part:
a native model of the mechanic, the A* that plans on it, the optimal-action-set
extraction, and the engine replay that certifies every plan before it is used.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_brotherhood",
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
action (post rotation remap), i.e. the button an agent presses in the augmented
view, so replaying the recorded actions reproduces the recorded frames exactly.

The game
--------
Three (on four levels, a different three) princes walk the board TOGETHER: one
arrow key moves EVERY prince one cell in that direction at once. There is no
"select a character" key, so the whole puzzle is finding a single sequence of
presses that funnels three bodies into three goals -- a prince who is already
where he wants to be still moves, unless a wall, an enemy or a brother happens
to stand in his way. Walls are what let the group break formation.

Win: **every** prince stands on **a** goal. The title screen says "same colour",
but the win condition the engine checks is `all Player on Goal`, so any prince
on any goal counts -- which is what makes several of the later levels solvable
at all.

Four more mechanics, in the order the engine resolves them each turn:

  * **Portals.** A prince standing on his OWN colour's entry portal is teleported
    to its exit portal at the START of the next turn, whichever key is pressed,
    and that consumes his move -- so a portal costs one press to step onto and
    one press to ride. ACTION (X) does nothing else at all, which makes it the
    "ride the portal but let nobody walk" key, and it is the only way to keep the
    other princes still. Portals are one-way and per-prince: prince 2's portal is
    scenery to everyone else.
  * **Fighting.** Prince 1 kills both enemy kinds, prince 2 kills only Enemy1
    (the black soldier), prince 3 only Enemy2 (the white yeti), prince 4 kills
    nothing. Walking INTO a killable enemy destroys it and cancels that prince's
    move (so it takes a second press to step onto the freed cell); walking into
    an enemy you cannot kill is simply blocked, as is walking into a wall or a
    brother.
  * **Blocking.** Princes occupy each other's cells. Since they all move the same
    direction at once, a blocked prince blocks everyone queued behind him, and a
    prince whose move was cancelled (by a kill or a teleport) becomes a wall for
    that turn.
  * **Eviction (a quirk, deliberately never used).** A teleport is a rule, not a
    move, so it is not collision-checked: a prince who rides a portal onto a cell
    where a brother is standing DELETES that brother, and the win condition then
    only has two princes left to satisfy. `_Level.dead` marks any state with a
    missing prince and the search refuses to enter one, so no recorded plan ever
    solves a level by killing a brother. The model reproduces the eviction
    faithfully anyway -- the fuzz below compares against the interpreter, which
    does it.

Rendering fixes (`data/puzzlescript_games/Brotherhood.txt`)
-----------------------------------------------------------
Seven of the fifteen levels are 21 cells wide, which renders at **3 px per cell**
-- a 5x5 sprite point-sampled down to its rows and columns 0, 2 and 4. Four
things were invisible or ambiguous at that resolution and one at 6 px too. All
five fixes are sprite/colour edits; no rule was touched.

  * **Wall2 was invisible.** Its `darkgreen` and Background3's olive both map to
    the single ARC green, and its sprite's left/right columns are transparent, so
    on levels 1-3 and 15 the walls differed from the grass by ONE pixel per cell.
    It is now an opaque grey stone block.
  * **Portal2a was a wall.** Its blue body sampled to a solid blue 3x3 -- exactly
    Wall1's. The entry portals are now a black cross on their colour and the exit
    portals their colour ringed by black corners, so the two are also no longer
    the same block as each other.
  * **The goals had no colour.** All four sampled to a flat black square. They are
    now a solid tile of their own colour with a black centre and YELLOW corners.
  * **A prince hid whatever he stood on.** The renderer draws players last (on top
    of their whole cell regardless of layer), and the only transparent pixels of a
    prince sprite were the two ends of his hat row -- so "on a goal", "on an entry
    portal" and "on an exit portal" were all the same picture. The corners above
    now read yellow / the portal's colour / black respectively, and the princes'
    bottom-row corners were opened up as well so the tile underneath shows through
    at four pixels instead of two.
  * **Background2 swallowed the soldiers.** The cave floor mapped to
    very-dark-grey and Enemy1 -- the enemy every cave level is built out of -- is
    a black ring round one skin-coloured pixel, so a soldier was a single bright
    dot on an almost identical surround. The cave floor is now muted purple.

Expert solver
-------------
The interpreter is far too slow to search here -- engine A* on the game's Level 7
burns 200k nodes in 137 s and still loses, where the model below solves it in
10 s -- so this generator plans on a NATIVE model of
the mechanic (the board as static `bytearray`s, the state as ONE packed int:
a byte per prince's cell plus an enemies-alive bitmask) and then certifies each
plan by replaying it through the real interpreter and requiring `check_win`.

The model was fuzz-verified against the interpreter first: every one of the five
successors of every state on 60 random walks per level, comparing prince
positions, surviving enemies AND the win flag -- 269,940 transitions over all
fifteen levels, zero mismatches.

Nothing it reads is privileged. Brotherhood is fully observable, and after the
rendering fixes above every distinction the search uses -- wall vs floor, which
prince, which enemy kind, goal vs entry portal vs exit portal, and all three of
those WITH a prince standing on them -- is a distinct block of pixels at both
cell sizes the game renders at. Reading the engine grid is a faster way to see
the frame, not a way to see past it.

The search (`solve`) is A* over the five keys, guided by

    h(s) = max over princes of (that prince's shortest walk to any goal)

where the walk ignores enemies and brothers but honours walls and counts a portal
as one edge. Every turn advances each prince by at most one edge of his own
graph, so the max is a genuine lower bound and the plans are SHORTEST. Fourteen
of the fifteen levels fall to it, the worst in ~25 s / 2.5M states. The
fifteenth, level index 12, is the one place ``max`` goes flat -- the middle of
its solution is one prince walking to a portal while the prince the bound is
watching stands still -- and 12M states is not enough to leave the opening;
`beam` takes it instead. See `beam` for what that costs.

Optimal-action SETS
-------------------
Lockstep movement makes ties everywhere: "seven rights then two downs" and any
interleaving of them usually cost the same, and training one arbitrary order as
the only right answer teaches a tie-break nobody needs. Because `h` is consistent
(one move changes any prince's field value by at most one), A* has exact `g` for
every state it pops, so continuing to pop until `f > D` yields exact `g*` for
EVERY state on an optimal path. One backward sweep over those layers marks the
states that still reach a win in `D - g*` moves, and the optimal set at each step
is every key landing on a marked state. See `solve`.

Only the beamed level has no sets; its steps train on the action the expert took,
which `BaseSolver` fills in for an action with no ``optimal`` key.

Recovery
--------
``recovery_mode = "reset"`` (inherited). The episode-wide exploration prefix
flails, ONE RESET restores the level's initial state, and the cached plan replays
a guaranteed win from there.

Augmentation
------------
Brotherhood's engine state after reset is identical for every seed (the levels
are fixed ASCII maps), so the only per-(seed, level) variable is presentation:
the frame rotation plus its matching directional action remap. There is
deliberately no flip: the entry and exit portals are told apart by an
ASYMMETRIC pair of corner pixels (see the rendering fixes above), which a mirror
would swap; rotation is a rigid motion and leaves the pair readable.

The expert plan is therefore seed-independent: solved once per level, cached, and
replayed per seed with that seed's remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_brotherhood_training.py --episodes 200 \\
        --out data/training_multi_level/brotherhood
"""

from __future__ import annotations

import heapq
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (DIRECTIONS, PSAStarSolver,  # noqa: E402
                                     PSExpert, restore, snapshot)

GAME_NAME = "Brotherhood"

#: Index into `DIRECTIONS` -> (dr, dc). Index 4 ("action") moves nobody.
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Prince class index (0 = Player1 ... 3 = Player4) -> the enemy kinds
#: (0 = Enemy1, 1 = Enemy2) he destroys by walking into them. Read straight off
#: the four `[ > PlayerN | EnemyM ] -> [ PlayerN | ]` rules.
_KILLS = (frozenset({0, 1}), frozenset({0}), frozenset({1}), frozenset())

#: A prince deleted by the eviction quirk. 0xFF is outside any real cell index
#: (the widest board is 21x9 = 189 cells), so it packs into the same byte.
_DEAD = 0xFF

INF = 1 << 30


class _Level:
    """The native Brotherhood model for one level.

    Static geometry is read once off the engine grid into flat `bytearray`s;
    everything that moves lives in a single packed int -- one byte per prince
    (his flat cell, or `_DEAD`) plus the still-alive enemy bitmask above them.
    The state is a dict key on every node of the search, so a plain int (hashed
    in C, no tuple allocation per successor) is worth the shifting.

    Fuzz-verified against the interpreter; see the module docstring.
    """

    _OBJECTS = ("player1", "player2", "player3", "player4")
    _BLOCKERS = ("wall1", "wall2", "wall3", "king", "soldier", "banner")

    def __init__(self, eng, game):
        idx = game.obj_name_to_idx
        self.h, self.w = len(eng.grid), len(eng.grid[0])
        n = self.h * self.w
        pid = [idx[o] for o in self._OBJECTS]
        eid = (idx["enemy1"], idx["enemy2"])
        blockers = {idx[o] for o in self._BLOCKERS}
        goals = {idx[f"goal{i}"] for i in (1, 2, 3, 4)}

        self.blocked = bytearray(n)
        self.goal = bytearray(n)
        #: cell -> bit index in the alive mask; parallel `enemy_kind` list.
        self.enemy_bit: dict[int, int] = {}
        self.enemy_kind: list[int] = []
        entry, exit_, start = {}, {}, {}
        for r in range(self.h):
            for c in range(self.w):
                cell = eng.grid[r][c]
                if not cell:
                    continue
                f = r * self.w + c
                if cell & blockers:
                    self.blocked[f] = 1
                if cell & goals:
                    self.goal[f] = 1
                for kind, e in enumerate(eid):
                    if e in cell:
                        self.enemy_bit[f] = len(self.enemy_kind)
                        self.enemy_kind.append(kind)
                for i in range(4):
                    if pid[i] in cell:
                        start[i] = f
                    if idx[f"portal{i + 1}a"] in cell:
                        entry[i] = f
                    if idx[f"portal{i + 1}b"] in cell:
                        exit_[i] = f

        #: The prince classes this level actually uses, ascending -- which is
        #: also the order the four portal rules fire in, so eviction resolves
        #: the way the interpreter resolves it.
        self.pidx = sorted(start)
        self.n = len(self.pidx)
        #: Per present prince: {entry cell: exit cell}, empty when his colour's
        #: pair is not on this board (levels 10-12 ship only two of the three).
        self.portal = [{entry[i]: exit_[i]} if i in entry and i in exit_ else {}
                       for i in self.pidx]
        self.kills = [_KILLS[i] for i in self.pidx]
        self.mask_shift = 8 * self.n
        self.all_alive = (1 << len(self.enemy_kind)) - 1
        self.start = self.pack([start[i] for i in self.pidx], self.all_alive)

    # -- state packing -------------------------------------------------------
    def pack(self, cells, mask: int) -> int:
        s = mask << self.mask_shift
        for i, f in enumerate(cells):
            s |= f << (8 * i)
        return s

    def cells(self, state: int) -> list[int]:
        return [(state >> (8 * i)) & 0xFF for i in range(self.n)]

    # -- dynamics ------------------------------------------------------------
    def won(self, state: int) -> bool:
        """`all Player on Goal`, exactly as the engine checks it: every prince
        still on the board stands on a goal, and at least one prince is left
        (the interpreter refuses a vacuous win -- see `_check_single_win_condition`)."""
        alive = False
        for i in range(self.n):
            f = (state >> (8 * i)) & 0xFF
            if f == _DEAD:
                continue
            alive = True
            if not self.goal[f]:
                return False
        return alive

    def dead(self, state: int) -> bool:
        """True when a prince has been evicted off the board by a teleport.

        Only the SEARCH cares: the state is legal and the interpreter reaches it,
        but a plan that goes through it wins by deleting a brother rather than by
        solving the level, which is not what this game is teaching."""
        for i in range(self.n):
            if (state >> (8 * i)) & 0xFF == _DEAD:
                return True
        return False

    def step(self, state: int, a: int) -> int:
        """One turn: portals, then kills, then movement -- the order the
        interpreter's rule list and movement resolution impose."""
        n = self.n
        w, h = self.w, self.h
        pl = [(state >> (8 * i)) & 0xFF for i in range(n)]
        mask = state >> self.mask_shift
        moving = [a < 4 and pl[i] != _DEAD for i in range(n)]

        for i in range(n):                       # portal rules
            dest = self.portal[i].get(pl[i])
            if dest is None:
                continue
            for j in range(n):                   # the eviction quirk
                if j != i and pl[j] == dest:
                    pl[j] = _DEAD
                    moving[j] = False
            pl[i] = dest
            moving[i] = False

        if a < 4:
            dr, dc = _DELTA[a]
            tgt = [None] * n
            for i in range(n):
                if not moving[i]:
                    continue
                r, c = divmod(pl[i], w)
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    tgt[i] = nr * w + nc
                else:
                    moving[i] = False           # the board edge

            for i in range(n):                   # kill rules
                if not moving[i]:
                    continue
                b = self.enemy_bit.get(tgt[i])
                if (b is not None and (mask >> b) & 1
                        and self.enemy_kind[b] in self.kills[i]):
                    mask &= ~(1 << b)
                    moving[i] = False            # the rule cancels the move

            # Resolve front-to-back along the press: everyone moves the same way,
            # so the prince nearest the wall decides for the ones behind him, and
            # one ordered pass is the whole (fixpoint) resolution.
            occupied = {f for f in pl if f != _DEAD}
            for i in sorted(range(n),
                            key=lambda i: -(dr * (pl[i] // w) + dc * (pl[i] % w))):
                if not moving[i]:
                    continue
                t = tgt[i]
                if self.blocked[t] or t in occupied:
                    continue
                b = self.enemy_bit.get(t)
                if b is not None and (mask >> b) & 1:
                    continue                     # an enemy he cannot kill
                occupied.discard(pl[i])
                occupied.add(t)
                pl[i] = t

        return self.pack(pl, mask)

    def read(self, eng, game) -> int:
        """Pack the engine's CURRENT grid into a model state.

        Used to plan from wherever the recorder actually is, and by the fuzz."""
        idx = game.obj_name_to_idx
        pid = [idx[o] for o in self._OBJECTS]
        eids = (idx["enemy1"], idx["enemy2"])
        cells = [_DEAD] * self.n
        where = {p: i for i, p in enumerate(self.pidx)}
        mask = 0
        for r in range(self.h):
            for c in range(self.w):
                cell = eng.grid[r][c]
                if not cell:
                    continue
                f = r * self.w + c
                for i in range(4):
                    if pid[i] in cell and i in where:
                        cells[where[i]] = f
                b = self.enemy_bit.get(f)
                if b is not None and (cell & set(eids)):
                    mask |= 1 << b
        return self.pack(cells, mask)

    # -- heuristic -----------------------------------------------------------
    def goal_fields(self) -> list[list[int]]:
        """Per prince: the shortest walk from every cell to the nearest goal,
        over non-wall cells, with his own entry->exit portal as a cost-1 edge.

        Enemies and brothers are ignored, so this only ever under-estimates, and
        one turn moves a prince along at most one edge of this graph -- which
        makes ``max`` over the princes an admissible AND consistent heuristic."""
        n = self.h * self.w
        out = []
        for i in range(self.n):
            back = {v: k for k, v in self.portal[i].items()}
            dist = [INF] * n
            q = deque()
            for f in range(n):
                if self.goal[f] and not self.blocked[f]:
                    dist[f] = 0
                    q.append(f)
            while q:
                f = q.popleft()
                d = dist[f] + 1
                r, c = divmod(f, self.w)
                nbrs = []
                if r:
                    nbrs.append(f - self.w)
                if r + 1 < self.h:
                    nbrs.append(f + self.w)
                if c:
                    nbrs.append(f - 1)
                if c + 1 < self.w:
                    nbrs.append(f + 1)
                if f in back:
                    nbrs.append(back[f])
                for g in nbrs:
                    if not self.blocked[g] and dist[g] > d:
                        dist[g] = d
                        q.append(g)
            out.append(dist)
        return out


class _Plan(list):
    """A flat list of engine directions that also carries, per step, the SET of
    directions that are equally optimal there.

    A list subclass so `record_level` -- which only ever indexes it, measures it
    and truth-tests it -- needs to know nothing about the sets, while
    `BrotherhoodSolver.optimal_for` can read them straight off the plan object
    it is handed. (Keying a side table by ``id(plan)`` would be a use-after-free
    waiting to happen.)"""

    __slots__ = ("optsets",)


def solve(lv: _Level, state: int, node_cap: int) -> _Plan | None:
    """A* from ``state`` to a win, returning a SHORTEST plan whose steps carry
    their optimal-action sets, or None if unsolvable within ``node_cap``.

    Two things beyond a textbook A*, both bought by `goal_fields` being
    CONSISTENT (so a popped state's ``g`` is already ``g*``):

      * the search does not stop at the first goal POPPED, it records that depth
        as ``D`` and keeps popping while ``f <= D``. That expands exactly
        ``{s : g*(s) + h(s) <= D}``, which contains every state on every optimal
        path, with an exact ``g*`` for each.
      * one backward sweep over those states, grouped by ``g*``, marks the ones
        that still reach a win in ``D - g*`` moves. The optimal set at a marked
        state is then every key that lands on another marked state, and the
        returned plan is one walk through the marks.
    """
    if lv.won(state):
        p = _Plan()
        p.optsets = []
        return p
    fields = lv.goal_fields()
    rng = range(lv.n)

    def h(s: int) -> int:
        return max(fields[i][(s >> (8 * i)) & 0xFF] for i in rng)

    h0 = h(state)
    if h0 >= INF:
        return None
    pq = [(h0, 0, state)]
    best = {state: 0}
    gstar: dict[int, int] = {}
    layers: list[list[int]] = []
    depth = None
    nodes = 0
    while pq:
        f, g, s = heapq.heappop(pq)
        if depth is not None and f > depth:
            break
        if s in gstar:
            continue
        gstar[s] = g
        while len(layers) <= g:
            layers.append([])
        layers[g].append(s)
        if lv.won(s):
            if depth is None:
                depth = g
            continue
        if depth is not None and g >= depth:
            continue
        for a in range(5):
            ns = lv.step(s, a)
            nodes += 1
            if ns in gstar or lv.dead(ns):
                continue
            ng = g + 1
            if best.get(ns, INF) <= ng:
                continue
            hn = h(ns)
            if hn >= INF or (depth is not None and ng + hn > depth):
                continue
            best[ns] = ng
            heapq.heappush(pq, (ng + hn, ng, ns))
        if nodes >= node_cap:
            return None
    if depth is None:
        return None

    # Backward sweep: `good[t]` is the states at g* == t that still win in
    # depth - t. Only the marked states are kept, which is the optimal-path DAG
    # and a small fraction of what the search expanded.
    good = [set() for _ in range(depth + 1)]
    good[depth] = {s for s in layers[depth] if lv.won(s)}
    for t in range(depth - 1, -1, -1):
        nxt = good[t + 1]
        good[t] = {s for s in layers[t]
                   if any(lv.step(s, a) in nxt for a in range(5))}
    if state not in good[0]:                     # cannot happen; cheap guard
        return None

    plan = _Plan()
    plan.optsets = []
    cur = state
    for t in range(depth):
        opts = [a for a in range(5) if lv.step(cur, a) in good[t + 1]]
        plan.optsets.append([DIRECTIONS[a] for a in opts])
        plan.append(DIRECTIONS[opts[0]])
        cur = lv.step(cur, opts[0])
    return plan


def beam(lv: _Level, state: int, width: int, max_depth: int) -> _Plan | None:
    """Breadth-first beam over the five keys, ``width`` states surviving each
    depth, ranked by the SUM of the princes' goal distances.

    WHY (what A* runs out of). `solve`'s ``max`` is the tightest admissible bound
    there is, but on the deepest level it is also nearly FLAT: the whole middle
    of the solution is one prince walking to a portal while the prince who is
    furthest from his goal -- and so the only one ``max`` can see -- does not
    move at all. A* then has nothing to steer with, degenerates to uniform-cost
    at depth 50-something, and 12M states later has not left the opening.
    The ``sum`` a beam ranks on is not a bound and could never drive A*, but it
    does notice the prince who IS making progress, which is all a survivor
    ranking has to do.

    The trade is optimality: the plan is engine-verified and winning but longer
    than the shortest, so it carries NO optimal sets (``optsets = None``) and its
    steps train on the action the expert took. `shorten` claws back most of the
    wandering."""
    fields = lv.goal_fields()
    rng = range(lv.n)

    def rank(s: int) -> int:
        return sum(fields[i][(s >> (8 * i)) & 0xFF] for i in rng)

    frontier = [(state, [])]
    seen = {state}
    for _ in range(max_depth):
        kids = []
        for s, path in frontier:
            for a in range(5):
                ns = lv.step(s, a)
                if ns in seen or lv.dead(ns):
                    continue
                seen.add(ns)
                if lv.won(ns):
                    return _finish(lv, state, path + [a])
                r = rank(ns)
                if r < INF:
                    kids.append((r, ns, path + [a]))
        if not kids:
            return None
        kids.sort(key=lambda kid: kid[0])
        frontier = [(s, p) for _r, s, p in kids[:width]]
    return None


def _finish(lv: _Level, state: int, actions: list[int]) -> _Plan:
    plan = _Plan()
    plan.optsets = None
    plan.extend(DIRECTIONS[a] for a in shorten(lv, state, actions))
    return plan


def shorten(lv: _Level, state: int, actions: list[int]) -> list[int]:
    """Delete the longest block of the plan that the model says is redundant,
    repeatedly, until nothing can come out.

    A beam plan wanders: it survives on a ranking, not a bound, so it collects
    detours that the win does not need. Cutting whole blocks (longest first, so
    one pass removes a wasted round trip rather than nibbling at its ends) is
    the cheap fix -- a model step is microseconds, so re-simulating the plan
    O(n^2) times costs less than the beam's first depth."""
    def wins(seq) -> bool:
        s = state
        for a in seq:
            s = lv.step(s, a)
            if lv.won(s):
                return True
        return False

    cur = list(actions)
    changed = True
    while changed:
        changed = False
        for size in range(len(cur) - 1, 0, -1):
            for i in range(len(cur) - size + 1):
                cand = cur[:i] + cur[i + size:]
                if cand and wins(cand):
                    cur = cand
                    changed = True
                    break
            if changed:
                break
    return cur


class BrotherhoodExpert(PSExpert):
    """Plans on `_Level` and certifies every plan through the interpreter.

    `PSExpert.plan`'s state-keyed memo is replaced wholesale: the model is the
    thing that is fast here, so a plan is keyed by (level, packed model state)
    and the engine is touched only to read the current state and to replay the
    finished plan once."""

    directions = DIRECTIONS

    def setup(self) -> None:
        self._levels: dict[int, _Level] = {}
        self._plans: dict[tuple[int, int], _Plan | None] = {}

    def level(self, eng, level: int) -> _Level:
        """The model for ``level``, built from the engine's CURRENT grid.

        The first call for a level therefore has to happen at that level's start,
        or a mid-game grid would bake a killed enemy out of the static tables.
        `PSAStarSolver._ensure` guarantees it: `discover_solvable` does
        ``set_level`` then ``plan`` for every level before anything is recorded."""
        lv = self._levels.get(level)
        if lv is None:
            lv = self._levels[level] = _Level(eng, self.g)
        return lv

    #: Levels handed straight to `beam`, skipping the exact search. Only the
    #: last cave level needs it, and letting A* prove it cannot get there first
    #: would cost ~90 s of every process's startup for nothing -- which a
    #: sharded run (`parallelize_generator.py`) pays once per core.
    beam_levels: frozenset[int] = frozenset({12})
    beam_width: int = 20_000
    beam_depth: int = 200

    def plan(self, eng, level: int | None = None) -> _Plan | None:
        lv = self.level(eng, level)
        state = lv.read(eng, self.g)
        key = (level, state)
        if key in self._plans:
            return self._plans[key]
        if level in self.beam_levels:
            plan = beam(lv, state, self.beam_width, self.beam_depth)
        else:
            plan = solve(lv, state, self.node_cap)
        if plan is not None and not self._certify(eng, plan):
            plan = None
        self._plans[key] = plan
        return plan

    def _certify(self, eng, plan) -> bool:
        """Replay ``plan`` through the real interpreter and require a win.

        The model is fuzz-verified, so this never fires -- which is exactly why
        it is here: a model that silently drifts from the engine (a sprite edit
        that changes a collision layer, an engine fix) would otherwise be found
        only as a mysteriously empty corpus."""
        snap = snapshot(eng)
        try:
            for d in plan:
                eng.step(d)
                if eng.check_win():
                    return True
            return eng.check_win()
        finally:
            restore(eng, snap)


class BrotherhoodSolver(PSAStarSolver):
    """`PSAStarSolver` wiring for ps:brotherhood."""

    game_id = "puzzlescript_brotherhood"
    game_name = GAME_NAME
    game_module_id = "ps:brotherhood"
    expert_cls = BrotherhoodExpert

    #: Every level is recorded. Fourteen of the fifteen are solved OPTIMALLY by
    #: `solve`; level index 12 (the game's "Level 13") is the one the beam has
    #: to take -- see `BrotherhoodExpert.beam_levels`.
    skip_levels = frozenset()

    #: Peak for `solve` is level index 9 at ~2.5M expanded states / ~25 s /
    #: ~0.25 GB, so this is roughly a 4x margin rather than a working figure.
    node_cap = 12_000_000
    max_steps = 200

    def optimal_for(self, expert, level: int, plan, pi: int):
        """The full tie set the search proved optimal at this step.

        Lockstep movement makes those ties the common case, not the exception --
        see the module docstring."""
        sets = getattr(plan, "optsets", None)
        if sets is None or pi >= len(sets):
            return None
        return sets[pi]


if __name__ == "__main__":
    BrotherhoodSolver.main()
