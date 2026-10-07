"""Generate Phase-1 training data for the PuzzleScript game
ps:crocodiles_love_cookies ("Crocodiles Love Cookies" by Ethan Clark -- stick
your tongue out, shove a cookie into your own mouth, and try not to trap
yourself behind your own tongue).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, a native model of the interpreter, the search over it, and the
optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_crocodiles_love_cookies",
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
action (post rotation/flip remap), i.e. the button an agent presses in the
presented view, so replaying the recorded actions reproduces the recorded frames
exactly.

The game
--------
Win is ``all goal on cookie``: every goal square (the yellow ring inside a
crocodile's mouth) has to be covered by a cookie. The player is ``toungeP``, the
red TIP of the tongue, and it starts in a mouth one square below that mouth's
goal. Only the four arrow keys do anything -- no rule in the .txt mentions
``action``, so ACTION5 is a dead key that burns a turn.

  * **THE TONGUE IS A TRAIL.** ``[moving toungeP] -> [moving toungeP
    prevToungePosition]`` marks the square the tip is leaving and
    ``late [prevToungePosition] -> [tounge]`` turns it into a tongue segment.
    Tongue is ``solid``, so the tip can never re-enter a square it has left:
    between resets the tongue is a SELF-AVOIDING WALK out of the mouth, and a
    tip that walls itself in has lost the level with no way to say so.

  * **A blocked press CANCELS the whole turn.** ``late [prevToungePosition
    toungeP] -> cancel`` fires when the tip is still standing on the square it
    tried to leave, and the interpreter reverts the turn wholesale. So there is
    no such thing as waiting: pressing into a wall changes nothing at all
    (except the adapter's step counter).

  * **The tip PUSHES cookies**, any number in a line
    (``[> moveable | moveable] -> [> moveable | > moveable]``), including
    UPWARDS -- a cookie shoved up stays up, because the tip that pushed it is
    now underneath holding it.

  * **COOKIES FALL.** ``late down [cookie | no solid] -> [cookie goDown | ]``
    plus ``late [goDown] -> again`` drops every unsupported cookie one square
    per ``again`` tick until something solid is underneath. Tongue is solid, so
    **the tongue is the scaffolding**: the only way to park a cookie in mid-air
    (and most goals can only be reached along a row in mid-air) is to lay
    tongue under it first. That is the whole game.

  * **TOUCHING A GOAL RETRACTS THE TONGUE.** ``late [toungeP goal] ->
    [destroyTounge goal]``, a flood fill over the board, and
    ``late [destroyTounge originalToungeP] -> [toungeP]`` erase every tongue
    segment and put the tip back in its starting square. It is the level's undo
    -- the trail is cleared and any cookie the trail was holding up falls -- and
    the .txt announces it with "I wonder what happens if I eat my own tounge?".
    ``originalToungeP`` is pinned to the start square by the two bookkeeping
    rules at the top of the file, so the tip always comes home to the same
    square, and the retract cannot be aimed anywhere else.

  * The win is checked after EVERY gravity tick (the interpreter breaks its
    ``again`` loop the moment ``check_win`` is true), so a cookie that would
    fall THROUGH a goal wins on the way past. `Model.step` reproduces that
    exactly -- see its docstring.

Reading the frame
-----------------
Croc green, tongue pink, tip red, cookie orange, wall black, floor light grey.
The goal's own ``#a3a300`` lands on the ARC palette's ORANGE -- the cookie's
colour -- so ``games/ps:crocodiles_love_cookies`` recolours it to yellow; see
that file. ``--audit`` checks every cell composition is pixel-distinct at every
cell size the levels render at.

Coverage
--------
**Seven of the twelve levels solve** (0-4, 6, 9), 14-28 presses each, every plan
replayed through the interpreter to a WIN and every recorded episode replayed
back from its recorded screen presses frame-for-frame. The other five are
skipped; `CrocodilesLoveCookiesSolver.skip_levels` says what makes them hard and
what was tried. A cold build searches all seven in ~26s and caches them to
``data/crocodiles_love_cookies_plans.json``; warm startup is under a second.

Usage
-----
    python solvers/generate_crocodiles_love_cookies_training.py --episodes 200 \\
        --out data/training_multi_level/crocodiles_love_cookies

    --selfcheck   fuzz `Model` against the real interpreter
    --audit       check every cell composition renders distinctly
    --plans       print (and interpreter-verify) every level's plan
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_cell_sprite  # noqa: E402
from solvers.common.ps_astar import (                          # noqa: E402
    Plan, PSAStarSolver, PSExpert, _load_game_module)

GAME_NAME = "Crocodiles_Love_Cookies"
GAME_MODULE_ID = "ps:crocodiles_love_cookies"

#: Where each level's start plan is cached between processes. The searches are
#: the entire cost of generation and are seed-independent, so without this file
#: every `parallelize_generator` shard re-derives all of them.
PLAN_CACHE = (Path(__file__).resolve().parent.parent
              / "data" / "crocodiles_love_cookies_plans.json")

#: The four live keys, in engine-direction form. ACTION is deliberately absent:
#: no rule in the .txt reads it, so it is a turn that does nothing, and offering
#: it to the search would only widen the branching.
KEYS = ("up", "down", "left", "right")

#: (dr, dc) per key index.
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Object names read out of a level.
_READ_NAMES = ("croc", "tounge", "cookie", "wall", "toungep", "goal",
               "originaltoungep")


# ---------------------------------------------------------------------------
# Native model of the interpreter
# ---------------------------------------------------------------------------

class Model:
    """Crocodiles Love Cookies' rules as a pure function on immutable states.

    A state is ``(tip, tongue, cookies)`` where ``tip`` is a cell index
    ``r * w + c`` and the other two are int BITMASKS over the same indexing.
    The board shape, the immovable scenery (wall + croc), the goals and the
    tongue's home square are level constants and live on the model.

    `step` returns the next state, or ``None`` when the turn CANCELS -- which
    the interpreter reverts whole, so a cancel means "nothing happened at all",
    not "a turn passed".

    WHY A MODEL AT ALL. The interpreter is ~700us per press and this is a
    self-avoiding walk, where the state IS the path: the searches below run
    into the millions of presses. The model is ~4us. It is verified against the
    real interpreter by `selfcheck`, and that fuzz is the only reason it is
    allowed to exist -- a model that drifts from the engine produces plans that
    do not replay.
    """

    def __init__(self, h: int, w: int, solid: int, goals: int, home: int):
        self.h, self.w = h, w
        self.n = h * w
        self.solid = solid          # wall | croc: never moves, never removed
        self.goals = goals
        self.home = home
        self.mv: list[list[int]] = []       # cell -> neighbour per key, -1 off-board
        for cell in range(self.n):
            r, c = divmod(cell, w)
            self.mv.append([
                (r + dr) * w + (c + dc)
                if 0 <= r + dr < h and 0 <= c + dc < w else -1
                for dr, dc in _DELTA])
        # Masks for `Heuristic._reach`'s bitmask flood: everything the tip can
        # ever stand on, and the two column guards that stop a sideways shift
        # from wrapping around a row edge into the next row.
        self.free = ((1 << self.n) - 1) & ~solid
        self.not_first_col = sum(1 << cell for cell in range(self.n)
                                 if cell % w)
        self.not_last_col = sum(1 << cell for cell in range(self.n)
                                if cell % w != w - 1)

    def won(self, st) -> bool:
        """``all goal on cookie`` -- every goal square covered."""
        return (st[2] & self.goals) == self.goals

    # -- the turn -------------------------------------------------------------
    def step(self, st, di: int):
        """One press; ``None`` if the turn cancels.

        The order below is the .txt's own, and it matters. The push resolves
        first (main rules), then the trail is laid, then the goal test retracts
        the whole tongue, and only then does gravity run -- so a tongue that was
        holding a cookie up is gone before the cookie is asked whether it is
        supported, and the retract drops it.

        The win is tested after the move and again after every single-square
        fall, because the interpreter breaks its ``again`` loop as soon as
        ``check_win`` holds: a cookie falling PAST a goal wins there and freezes
        in mid-air. Reproducing that is not pedantry -- it is a real (and short)
        way to win some boards, and a model that let the cookie continue would
        hand the recorder a plan the engine ends early.
        """
        tip, tongue, cookies = st
        nxt = self.mv[tip][di]
        if nxt < 0:
            return None                       # off the board: blocked, cancel
        # Push chain: every cookie in a line ahead of the tip moves with it. It
        # goes nowhere unless the square past the last cookie is free, and then
        # the tip is blocked too and the turn cancels.
        end = nxt
        while cookies >> end & 1:
            end = self.mv[end][di]
            if end < 0:
                return None
        if (self.solid | tongue) >> end & 1:
            return None
        if cookies >> nxt & 1:                # contiguous run: drop head, add tail
            cookies = (cookies & ~(1 << nxt)) | (1 << end)

        tongue |= 1 << tip                    # the square left becomes tongue
        tip = nxt
        if self.goals >> tip & 1:             # eat your own tongue: full retract
            tongue = 0
            tip = self.home

        st = (tip, tongue, cookies)
        if self.won(st):
            return st
        # Gravity: mark every unsupported cookie, then drop them all one square,
        # exactly as the ``goDown`` marker plus the ``again`` tick does.
        w, n = self.w, self.n
        while True:
            blocked = self.solid | tongue | cookies | (1 << tip)
            fall = 0
            rest = cookies
            while rest:
                bit = rest & -rest
                rest ^= bit
                below = bit << w
                if (bit.bit_length() - 1) + w < n and not blocked & below:
                    fall |= bit
            if not fall:
                return st
            cookies = (cookies & ~fall) | (fall << w)
            st = (tip, tongue, cookies)
            if self.won(st):
                return st


# ---------------------------------------------------------------------------
# Reading a model state out of the interpreter
# ---------------------------------------------------------------------------

def read_state(game, eng) -> tuple:
    """``(h, w, solid, goals, home, state)`` from a live interpreter grid.

    ``home`` is where a retract puts the tip back. ``originalToungeP`` marks it
    from the first press onwards; before that press -- and in the one window
    where a retract has consumed the marker and no ``again`` tick has re-pinned
    it -- there is no marker and the tip is standing on the home square itself,
    so the fallback is exact rather than a guess.
    """
    idx = game.obj_name_to_idx
    ids = {name: idx[name] for name in _READ_NAMES}
    h, w = len(eng.grid), len(eng.grid[0])
    solid = goals = tongue = cookies = 0
    tip = home = -1
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if not cell:
                continue
            bit = 1 << (r * w + c)
            if ids["wall"] in cell or ids["croc"] in cell:
                solid |= bit
            if ids["goal"] in cell:
                goals |= bit
            if ids["tounge"] in cell:
                tongue |= bit
            if ids["cookie"] in cell:
                cookies |= bit
            if ids["toungep"] in cell:
                tip = r * w + c
            if ids["originaltoungep"] in cell:
                home = r * w + c
    return h, w, solid, goals, (home if home >= 0 else tip), (tip, tongue,
                                                              cookies)


def model_from_engine(eng, game) -> tuple:
    """``(Model, state)`` for the interpreter's current grid."""
    h, w, solid, goals, home, state = read_state(game, eng)
    return Model(h, w, solid, goals, home), state


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

class Heuristic:
    """Distance-to-go estimate in presses, plus the two dead tests.

    WHAT THE ESTIMATE HAS TO SEE. A plan here is two things at once: a route
    the COOKIE takes to the goal, and a walk the TIP takes to make that route
    happen -- which means walking every square the cookie will have to rest on
    ON THE WAY, because the tongue is the only thing that stops a cookie in
    mid-air. An estimate that watches only the cookie is flat for the twenty
    presses the tip spends laying that scaffolding, and an estimate that
    watches only the tip points it straight at the cookie, which is usually the
    wrong way round the board. So `__call__` prices the pair:

      1. `_route` plans the cookie's cheapest RELAXED route to the goal --
         sideways pushes cost 1, falls are free, and stopping a fall in mid-air
         costs `LAY` on top, because the tongue under it has to be laid. The
         route knows the current tongue, so laying a square the route wanted
         makes the estimate drop: that is the gradient the search walks down.
      2. The route hands back the squares still needing tongue and the square
         the tip must stand on to start pushing. A greedy nearest-neighbour
         TOUR of those, over the walls-only distance table, is what the tip
         still owes. This is the term that makes the difference on the boards
         where the mouth is on the wrong side of the crocodile -- see the
         `_stand` note.

    It is a search guide, not a bound: `LAY` is a guess and the tour is greedy,
    so it can overestimate, and a plan is the shortest FOUND rather than the
    shortest. On the boards that solve at ``weight = 1`` it agrees with a
    Dijkstra on length anyway.

    TWO DEAD TESTS, and they do as much work as the estimate.

      * A goal no cookie can reach in the TONGUE-FREE relaxed field can never
        be covered in the real one either. That kills the branch where a cookie
        has been shoved into a pocket it can never leave. It has to ignore the
        tongue: touching a goal erases the whole tongue and hands the board
        back, so a square the tongue is blocking today is not blocked forever.
      * A tip that can reach neither a cookie nor a goal has no move left that
        will ever matter: pushing needs a cookie and clearing the trail needs a
        goal, so the board is frozen for the rest of the level. This is the
        common way to lose Crocodiles Love Cookies -- walking the tongue into a
        corner and sealing the exit behind it -- and it is a large slice of the
        branching on every open board.

    Both stay NECESSARY conditions (a state they kill really is lost) by being
    optimistic about everything they do not model: cookies are passable in the
    tip's flood, and the tongue is invisible to the cookie field.
    """

    #: Presses charged for parking a cookie in MID-AIR: the tongue underneath it
    #: does not lay itself. Only the ordering matters, not the units -- it is
    #: what makes the relaxed route prefer a route the board can actually hold a
    #: cookie on, over one that needs a shelf built under every square.
    LAY = 2

    def __init__(self, model: Model):
        self.model = model
        self.fields = {goal: self._field(model, goal)
                       for goal in _cells(model.goals)}
        self.dist = self._all_pairs(model)

    @staticmethod
    def _field(model: Model, goal: int) -> dict:
        """``{cell: relaxed presses to bring a cookie from cell onto goal}``,
        ignoring the tongue entirely -- the dead test's field.

        A 0-1 BFS over the REVERSED relaxed graph: from a square ``y`` a cookie
        could have arrived from its left/right neighbour (cost 1), from above
        by falling (cost 0), or from below by being pushed up (cost 1)."""
        w, solid = model.w, model.solid
        dist = {goal: 0}
        queue = deque([goal])
        while queue:
            y = queue.popleft()
            d = dist[y]
            for di, cost in ((0, 0), (1, 1), (2, 1), (3, 1)):
                # `mv[y][di]` is the square the cookie came FROM: up for a fall,
                # down for a push up, and either side for a sideways push.
                x = model.mv[y][di]
                if x < 0 or solid >> x & 1:
                    continue
                nd = d + cost
                if dist.get(x, 1 << 30) <= nd:
                    continue
                dist[x] = nd
                (queue.appendleft if cost == 0 else queue.append)(x)
        return dist

    def _route(self, src: int, goal: int, supports: int):
        """The cheapest relaxed route for a cookie at ``src`` to reach ``goal``
        over a board whose solid squares are ``supports``, as
        ``(pushes, squares still needing tongue, the tip's first stand square)``
        -- or None when there is no route at all.

        One push moves the cookie one square sideways and then gravity takes
        it, so a push's successors are every square down the column it lands
        in: the bottom of the fall for free, anything above that for `LAY`. A
        goal square reached anywhere in that fall is free -- the interpreter
        checks the win after every gravity tick, so a cookie that merely falls
        PAST a goal has already won."""
        m = self.model
        w, n, goals, scen = m.w, m.n, m.goals, m.solid
        dist = {src: 0}
        parent: dict = {src: None}
        pq = [(0, src)]
        while pq:
            d, x = heapq.heappop(pq)
            if d != dist.get(x):
                continue                      # a stale heap entry
            if x == goal:
                break
            if goals >> x & 1:
                continue                      # a cookie parked on another goal
            for di in (2, 3):                 # the push, and then the fall
                y = m.mv[x][di]
                if y < 0 or scen >> y & 1:
                    continue
                cur = y
                while True:
                    below = cur + w
                    landed = below >= n or supports >> below & 1
                    lay = not (landed or goals >> cur & 1)
                    nd = d + 1 + (self.LAY if lay else 0)
                    if nd < dist.get(cur, 1 << 30):
                        dist[cur] = nd
                        parent[cur] = (x, below if lay else -1)
                        heapq.heappush(pq, (nd, cur))
                    if landed:
                        break
                    cur = below
            y = m.mv[x][0]                    # shoved up, and held there
            if y >= 0 and not scen >> y & 1:
                nd = d + 1 + self.LAY
                if nd < dist.get(y, 1 << 30):
                    dist[y] = nd
                    parent[y] = (x, -1)       # the tip under it IS the support
                    heapq.heappush(pq, (nd, y))
        if goal not in dist:
            return None
        need, cur, pushes, first = [], goal, 0, None
        while parent[cur] is not None:
            prev, lay = parent[cur]
            pushes += 1
            if lay >= 0:
                need.append(lay)
            first = (prev, cur)
            cur = prev
        return pushes, need, self._stand(*first) if first else src

    def _stand(self, src: int, dest: int) -> int:
        """Where the tip has to stand to make the route's FIRST push: the
        square on the far side of the cookie from where it is going.

        Charging the walk to THIS square rather than to the cookie is most of
        what the tour term is worth. A crocodile is a horseshoe of solid
        squares with one opening, so the square beside a cookie and the square
        on its other side are routinely twenty presses apart, and an estimate
        that cannot tell them apart is blind to which way round the board the
        tongue has to go."""
        w = self.model.w
        if dest % w == src % w:               # straight up: stand underneath
            stand = src + w
        else:                                 # sideways (the fall follows it)
            stand = src - 1 if dest % w > src % w else src + 1
            if not 0 <= stand < self.model.n or stand // w != src // w:
                return src
        if not 0 <= stand < self.model.n or self.model.solid >> stand & 1:
            return src                        # unstandable: fall back to the
        return stand                          # cookie itself

    @staticmethod
    def _all_pairs(model: Model) -> list:
        """``dist[a][b]`` -- presses to walk the tip from ``a`` to ``b`` over
        the scenery only. Built once per level; a board is a few hundred cells,
        so this is a few hundred BFS over a few hundred cells."""
        out: list = [None] * model.n
        for src in range(model.n):
            if model.solid >> src & 1:
                continue
            dist = {src: 0}
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for nxt in model.mv[cur]:
                    if nxt >= 0 and not model.solid >> nxt & 1 and nxt not in dist:
                        dist[nxt] = dist[cur] + 1
                        queue.append(nxt)
            out[src] = dist
        return out

    def _reach(self, st) -> int:
        """Bitmask of the squares the tip could still walk to, treating cookies
        as passable (it may be able to push them out of the way, and being
        optimistic is what keeps the dead test sound).

        A bitmask flood rather than a BFS: this runs on every node the search
        touches, and four shifts per iteration over one big int is about what a
        single `Model.step` costs, where a per-cell BFS would be twenty times
        that."""
        model = self.model
        free = model.free & ~st[1]
        region = 1 << st[0]
        while True:
            grown = (region | region >> model.w | region << model.w
                     | (region & model.not_first_col) >> 1
                     | (region & model.not_last_col) << 1) & free
            if grown == region:
                return region
            region = grown

    def _tour(self, tip: int, need: list, stand: int) -> int:
        """Presses for the tip to visit every square in ``need`` and finish on
        ``stand``, as a greedy nearest-neighbour tour over the walls-only
        distance table. A route needs at most a handful of squares, so the
        greedy order is within a press or two of the best one and costs
        nothing to compute."""
        cur, walk = tip, 0
        left = list(dict.fromkeys(need))
        while left:
            table = self.dist[cur] or {}
            nxt = min(left, key=lambda cell: table.get(cell, self.model.n))
            walk += table.get(nxt, self.model.n)
            cur = nxt
            left.remove(nxt)
        table = self.dist[cur] or {}
        return walk + table.get(stand, self.model.n)

    def __call__(self, st):
        """Estimated presses to a win, or None when the state is dead."""
        tip, tongue, cookies = st
        model = self.model
        bare = [g for g in self.fields if not cookies >> g & 1]
        if not bare:
            return 0
        reach = self._reach(st)
        if not reach & (cookies | model.goals):
            return None                       # frozen: no push and no retract
        loose = [c for c in _cells(cookies) if not model.goals >> c & 1]
        for goal in bare:                     # the tongue-free reachability test
            if not any(self.fields[goal].get(c) is not None for c in loose):
                return None                   # no cookie can ever cover it
        supports = model.solid | tongue
        total = 0
        taken: set[int] = set()
        for goal in bare:
            best = best_c = None
            for cell in loose:
                if cell in taken:
                    continue
                route = self._route(cell, goal, supports)
                if route is not None and (best is None or route[0] < best[0]):
                    best, best_c = route, cell
            if best is None:
                return None
            taken.add(best_c)
            pushes, need, stand = best
            total += pushes + self._tour(tip, need, stand)
        return total


def _cells(mask: int):
    """The set bits of ``mask`` as cell indices, low to high."""
    out = []
    while mask:
        bit = mask & -mask
        mask ^= bit
        out.append(bit.bit_length() - 1)
    return out


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class CrocodilesExpert(PSExpert):
    """Weighted A* over `Model`, plugged into `PSExpert` as its search strategy.

    `PSExpert`'s own engine-blackbox A* is not used: at ~700us per interpreter
    press a search that has to enumerate self-avoiding walks never finishes.
    Everything here runs on the native model, and only the finished plan touches
    the engine (in `--plans`, and in the recorder that replays it). Only
    `_search` is overridden, so the memo, the disk cache and the restore
    discipline around it stay the shared ones -- this file does not carry an
    eighth hand-rolled copy of `plan_cache_path`."""

    directions = list(KEYS)
    plan_cache_path = PLAN_CACHE

    #: (weight, node cap) rungs, tried in order. Every level this solves is
    #: solved on the first or second rung in seconds; the third is there for a
    #: level added later. The caps are edges expanded, and one is ~90us.
    ladder = ((1, 400_000), (2, 400_000), (3, 1_000_000))

    def setup(self) -> None:
        self._how: dict = {}
        self._level: int | None = None

    # -- searches -------------------------------------------------------------
    @staticmethod
    def _astar(model: Model, start: tuple, h: Heuristic, weight: int,
               cap: int):
        """Weighted A*. Parent pointers rather than a path per heap entry: at a
        million nodes a copied tuple per node is most of the memory."""
        if model.won(start):
            return ()
        best = {start: 0}
        parent: dict = {start: None}
        counter = 0
        nodes = 0
        pq = [(0, 0, 0, start)]
        while pq:
            _f, g, _c, st = heapq.heappop(pq)
            if best.get(st, -1) != g:
                continue                      # a stale heap entry
            for di in range(4):
                nxt = model.step(st, di)
                if nxt is None:
                    continue
                nodes += 1
                if model.won(nxt):
                    out = [di]
                    cur = st
                    while parent[cur] is not None:
                        cur, pd = parent[cur]
                        out.append(pd)
                    out.reverse()
                    return tuple(out)
                hv = h(nxt)
                if hv is None:
                    continue                  # dead: a goal is unreachable
                ng = g + 1
                if best.get(nxt, 1 << 30) <= ng:
                    continue
                best[nxt] = ng
                parent[nxt] = (st, di)
                counter += 1
                heapq.heappush(pq, (ng + weight * hv, ng, counter, nxt))
            if nodes >= cap:
                return None
        return None

    def _search(self, eng):
        """`PSExpert`'s strategy hook: plan from the interpreter's current grid
        by reading a model state off it and searching THAT. Returns a `Plan`
        (presses plus their optimal sets), which is what the shared memo and
        disk cache round-trip."""
        model, start = model_from_engine(eng, self.g)
        h = Heuristic(model)
        for weight, cap in self.ladder:
            path = self._astar(model, start, h, weight, cap)
            if path is not None:
                self._how[self._level] = f"A* w={weight}"
                return Plan([KEYS[d] for d in path],
                            self._reorder_optsets(model, start, path))
        self._how[self._level] = "unsolved"
        return None

    # -- optimal sets ---------------------------------------------------------
    @staticmethod
    def _reorder_optsets(model: Model, start: tuple, path: tuple) -> list:
        """Per-step optimal sets: the press taken, plus every LATER press of the
        plan that could have been taken first.

        A press ``d`` from further down the plan is as good as the one being
        taken when pulling it to the front and replaying the rest still wins by
        the recorded plan's last step -- the two commute. That is the widening
        available without an exact distance-to-win field, which a self-avoiding
        walk does not admit: the trail is part of the state, so two routes to
        the same square are almost never interchangeable. It never claims a
        press is optimal in the absolute sense -- these plans are the shortest
        FOUND -- only that it is at least as good as what the expert did, which
        is what the demonstration teaches either way.

        The whole suffix is replayed rather than a two-press commutation test: a
        press that commutes with the next one but seals a corridor ten turns
        later is not equally good, and on the model a full replay costs
        microseconds. A press the model CANCELS is replayed as a wasted turn,
        exactly as the interpreter treats it, so a permutation needing one can
        never come out shorter than the plan."""
        states = [start]
        for di in path:
            states.append(model.step(states[-1], di))
        optsets = []
        for i, taken in enumerate(path):
            suffix = path[i:]
            best = {taken}
            for j in range(1, len(suffix)):
                if suffix[j] in best:
                    continue
                st = states[i]
                for di in (suffix[j],) + suffix[:j] + suffix[j + 1:]:
                    nxt = model.step(st, di)
                    if nxt is not None:
                        st = nxt
                    if model.won(st):
                        best.add(suffix[j])
                        break
            optsets.append([KEYS[d] for d in sorted(best)])
        return optsets

    # -- planning -------------------------------------------------------------
    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan`, remembering which level it is planning.

        The base does the work: the in-memory memo, and (because
        `plan_cache_path` is set) the on-disk cache of each level's START plan,
        which is the only state any seed ever plans from -- recovery is a RESET
        back to it. Both round-trip the `Plan`'s optimal sets, so a cached plan
        replayed in a later process still labels every step. All this override
        adds is the level number `describe` reports against."""
        self._level = level
        return super().plan(eng, level)

    def describe(self, level) -> str:
        """Which search answered for this level (for `--plans`); "cached" when
        the plan came off disk and nothing ran."""
        return self._how.get(level, "cached")


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class CrocodilesLoveCookiesSolver(PSAStarSolver):
    game_id = "puzzlescript_crocodiles_love_cookies"
    game_name = GAME_NAME
    #: The wrapper recolours the goal off the cookie's palette index, and that
    #: patched adapter is what `game_envs` hands a live agent -- so it has to be
    #: what this generator tapes.
    game_module_id = GAME_MODULE_ID
    expert_cls = CrocodilesExpert

    #: The five levels the search cannot reach. Skipped up front so a cold
    #: build does not spend its whole budget on them; the other seven solve in
    #: about half a minute all told, at 14-28 presses each.
    #:
    #: WHY THEY ARE HARD, and it is not the node budget. On the seven that
    #: solve, the tongue lays its scaffolding on the way OUT of the mouth and
    #: the cookie comes back along the row above it, so the estimate starts
    #: dropping within a few presses and A* has something to follow. These five
    #: put the cookie on the far side of the crocodile from its mouth, and the
    #: only solutions are of the shape "walk twenty squares laying a shelf,
    #: walk twenty more the long way round to reach the cookie's far side, push
    #: it onto the shelf, then come back". Over the first forty of those
    #: presses NOTHING the estimate can see improves -- the cookie has not
    #: moved and no square of the route it wants is any nearer being tongue --
    #: and the trail makes every one of those forty presses a distinct state,
    #: so the frontier is a self-avoiding walk enumeration with no gradient.
    #: Measured, not assumed: 3M expanded edges at w=2 does not reach any of
    #: them (nor does 4M under an earlier, weaker estimate), and neither does a
    #: width-20k beam or a greedy best-first pass. A waypoint decomposition
    #: (plan the cookie's route first, then route the tongue through the
    #: squares it reserves) was also prototyped and solved a strict subset of
    #: what the ladder already solves, so it is not in this file.
    skip_levels: frozenset[int] = frozenset({5, 7, 8, 10, 11})

    #: Plans run to 28 presses and the adapter gives a level 200 actions (the
    #: RESET in the recovery prefix zeroes that counter, so only the post-reset
    #: plan is charged against it), so this cap is never the binding one.
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
        """The optimal press set at this step, off the plan's own optsets: the
        press taken, widened by the reordering probe. Falls back to the press
        about to be taken so no expert step ever ships unlabelled --
        `train_policy` v2 supervises ``optimal`` only, so a step without one
        contributes nothing to the loss."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets) and sets[pi]:
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Model self-check (fuzz against the real interpreter)
# ---------------------------------------------------------------------------

def selfcheck(trials: int = 60, steps: int = 80, verbose: bool = True) -> int:
    """Audit the claim the whole solver rests on: `Model` reproduces the
    interpreter EXACTLY. On random rollouts from every level start, the settled
    state, the cancel/no-cancel decision and the win flag all have to agree.
    That is what licenses planning off the engine.

    Random play reaches the corners a plan never would -- shoving a cookie into
    the board edge, walking the tip onto its own goal to retract a tongue that
    was holding two cookies up, sealing the tip in a pocket of its own trail --
    so the rollouts are the fuzz. Every ninth press is ACTION, which no rule in
    this game reads: it must leave the observable state untouched, and a rule
    that started reading it would show up here as a divergence.

    Three claims are checked per press, and they are the three the searches
    rest on: the settled state agrees; a press the MODEL cancels leaves the
    interpreter's grid byte-identical (the converse is deliberately not
    asserted -- stepping onto the goal from the home square retracts a
    one-segment tongue and lands back where it started, which is a real turn
    with no net effect, not a cancel); and the tongue's HOME square never
    moves, which is what lets `Model` keep it as a level constant."""
    game = _load_game_module(GAME_MODULE_ID).make_game(seed=0)
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        cancels = wins = retracts = 0
        for t in range(trials):
            game.set_level(level)
            model, state = model_from_engine(eng, game._game)
            rng = random.Random(f"crocodiles:selfcheck:{level}:{t}")
            sealed = False
            for i in range(steps):
                if i % 9 == 8:                       # the dead key
                    eng.step("action")
                    _m, actual = model_from_engine(eng, game._game)
                    if actual != state:
                        bad += 1
                        print(f"  L{level}: ACTION changed the state")
                        break
                    continue
                # Mostly play ON, occasionally press blind. A uniformly random
                # walk seals the tip behind its own tongue within a dozen
                # presses and then spends the rollout bouncing off it, so three
                # presses in four are drawn from the moves the model believes
                # are legal -- which is not circular, since a wrong legality
                # claim still has to survive the state comparison below.
                legal = [d for d in range(len(KEYS))
                         if model.step(state, d) is not None]
                if not legal and sealed:
                    # The tip has walled itself in and the board can never
                    # change again. A few presses into the wall are worth
                    # checking (they are the cancel path); a hundred are not.
                    break
                sealed = not legal
                di = (rng.choice(legal) if legal and rng.random() < 0.75
                      else rng.randrange(len(KEYS)))
                before = [[set(c) for c in row] for row in eng.grid]
                eng.step(KEYS[di])
                frozen = eng.grid == before
                predicted = model.step(state, di)
                expected = state if predicted is None else predicted
                _h, _w, _s, _g, home, actual = read_state(game._game, eng)
                if actual != expected:
                    bad += 1
                    print(f"  L{level}: state divergence on {KEYS[di]}: "
                          f"model {expected} engine {actual}")
                    break
                if predicted is None and not frozen:
                    bad += 1
                    print(f"  L{level}: the model cancelled {KEYS[di]} but the "
                          f"interpreter changed the grid")
                    break
                if home != model.home:
                    bad += 1
                    print(f"  L{level}: home moved {model.home} -> {home}")
                    break
                if eng.check_win() != model.won(expected):
                    bad += 1
                    print(f"  L{level}: win divergence on {KEYS[di]}")
                    break
                if predicted is not None and predicted[1] == 0 and state[1]:
                    retracts += 1
                if predicted is None:
                    cancels += 1
                else:
                    state = predicted
                if eng.check_win():
                    wins += 1
                    break
        if verbose:
            print(f"  L{level}: {'OK' if not bad else 'VIOLATIONS'} "
                  f"({trials} rollouts, {cancels} cancelled turns, "
                  f"{retracts} retracts, {wins} accidental wins)")
        if bad:
            break
    return bad


# ---------------------------------------------------------------------------
# Render audit
# ---------------------------------------------------------------------------

#: Every cell composition a level can show, as the object stack that draws it.
#: ``goal + toungeP`` is not here on purpose: the tip touching a goal retracts
#: the tongue and teleports home inside the same turn, so no settled frame ever
#: shows it.
_AUDIT_CASES = {
    "floor": ["background"],
    "wall": ["background", "wall"],
    "croc": ["background", "croc"],
    "goal": ["background", "goal"],
    "cookie": ["background", "cookie"],
    "cookie on goal": ["background", "goal", "cookie"],
    "tongue": ["background", "tounge"],
    "tongue tip": ["background", "toungep"],
}


def audit(verbose: bool = True) -> int:
    """Check that every cell composition is pixel-distinct at every cell size
    the levels use.

    These boards render at 4 to 6 pixels per cell, which is where a sparse
    sprite decimates to nothing and a palette collision silently deletes a
    piece from the frame. The goal is exactly that case -- see the module
    docstring -- and this is the check that the wrapper's recolour holds.
    Returns the number of colliding pairs."""
    game = _load_game_module(GAME_MODULE_ID).make_game(seed=0)
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
        objs.sort(key=lambda x: x[0])
        return objs

    bad = 0
    for px in sorted(sizes):
        blocks = {k: _render_cell_sprite(stack(v), px)
                  for k, v in _AUDIT_CASES.items()}
        for a, b in itertools.combinations(_AUDIT_CASES, 2):
            if (blocks[a] == blocks[b]).all():
                bad += 1
                print(f"  cell_px={px}: {a} and {b} render identically")
        if verbose:
            print(f"  cell_px={px}: {len(_AUDIT_CASES)} cell types, "
                  f"{'all distinct' if not bad else 'COLLISIONS'}")
    return bad


def _plan_report() -> None:
    """Print every level's plan, how it was found and how many of its steps have
    more than one right answer -- the quick "is this game still fully solved"
    check. Every plan is replayed through the real interpreter, so this is also
    the model's end-to-end test."""
    solver = CrocodilesLoveCookiesSolver()
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
        for direction in plan:                          # interpreter-verify it
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
    sys.exit(CrocodilesLoveCookiesSolver.main())
