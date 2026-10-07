"""Generate Phase-1 training data for the PuzzleScript game
ps:shared_bridges_game ("Shared Bridges Game", a 2015 PuzzleScript Tennis
volley by Jere / sheepolution / Connorses / sfiera).

The harness -- the rotation contract, the trajectory recorder, the
RESET-recovery prefix and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: the mechanic notes, the native model of the turn, the
macro search over it, and the optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_shared_bridges_game",
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
Two characters live in two INTERLEAVED worlds painted over one board -- red land
and blue land (the file calls the second one green; see the rendering note) --
and exactly one of the two worlds is RAISED at a time. The raised world is the
one belonging to whoever you are steering; the other has sunk into a chasm.

  * **You walk on raised land, or on a platform lying in the chasm.** The engine
    says it in one rule: ``[ > Player | no WorldMid no PlatformLow ] -> [Player|]``.
    A platform tracks the land under it (``late [PlatformLow WorldMid] ->
    [PlatformMid ...]`` and its converse), so the SAME object is a chest-high
    CRATE when it sits on raised land and a BRIDGE when it sits in the chasm.
  * **So a platform is a crate for one character and a bridge for the other**,
    and that is the whole game. Red can only shove the platforms standing on RED
    land; the moment one is pushed over the edge it becomes a bridge red can walk
    on -- and a crate blue will have to deal with after the next switch.
  * **ACTION swaps which world is up and hands control to the other character.**
    The sleeper is frozen where it stands: it blocks walking (it shares the Char
    collision layer), it cancels any push into its cell (``[ > PlatformMid |
    Sleeper ] -> cancel``) and it cannot have the platform pulled out from under
    it (``[ > PlatformMid Sleeper ] -> cancel``). That last rule is why the newly
    woken character is always standing somewhere it may legally stand.
  * **Platforms STACK, and that is a mechanic rather than an accident.** Shoving
    a crate onto a cell that already holds a bridge leaves PlatformMid on top of
    PlatformLow; step onto the bridge and you can shove the passenger one further
    into the chasm. It is the only way to cross a gap wider than one cell, and
    level 0 -- one character, a two-wide chasm and two platforms -- is a tutorial
    for exactly it. While any such stack exists ACTION is refused
    (``[ action Player ] [ PlatformMid PlatformLow ] -> [ Player ] ...``), which
    is also what keeps two platforms from ever landing on the same layer.
  * **Levels 0 and 1 ship an invisible NoSwitching object and no blue
    character**, so ACTION there is a pure no-op. The tell an agent can see is
    the missing second character, not the object.
  * **The win is a SNAPSHOT of both characters** (``all Red on RedGoal`` and
    ``all Green on GreenGoal``), and a sleeper never moves, so it is: park one on
    its goal, switch, walk the other one home. Goals sit in either world, so
    "walk home" routinely means crossing the other world on borrowed bridges.
  * **No ``restart``, no ``again``, no ``random``** -- a press is one
    deterministic board rewrite. `_Board.step` mirrors it and ``--selfcheck``
    fuzzes the two against each other.

Model + search
--------------
The interpreter runs at ~200 steps/s here (25 of its 39 rules are ``late`` rules
that repaint the fake-3D caps over the whole board every turn), so an
engine-blackbox search is out -- see the EntrepotPhage note in
[[ps-astar-generator-family]]. `_Board` is the native model. Its state is
``(who is steering, red cell, blue cell, platform cells)``; the raised world
follows from who is steering, and a platform's HEIGHT follows from the land it
is on, so a multiset of cells is the whole platform state (a cell holds at most
two: a bridge and one passenger).

Two search tiers, strongest first, reported per level by ``--plans``:

  * **exact** -- A* over MACROS (walk to a cell, then push / press ACTION), with
    an admissible heuristic. Walking has no side effect, so a shortest press
    sequence is exactly a shortest macro sequence and the macro optimum IS the
    press optimum. A second, bounded pass (``g + h <= d*``) then records the
    shortest-path subgraph and a backward Dijkstra over it gives the exact
    distance-to-win for every state a shortest plan can contain -- so the plans
    are proved shortest and the per-step optimal SETS are measured, not inferred
    ([[ps-fractured-identity-solver]]'s bounded pass, applied to the labelling).
  * **beam** -- when that runs out of budget: a width-capped beam over the same
    macros, ordered by a crate-delivery progress score (an unbridged cell on your
    route costs a turn-charged PUSH distance from the nearest platform, which is
    what gives the search a gradient at all -- ordering by the ADMISSIBLE
    heuristic instead finds nothing, see [[idols-to-the-burnt-god-solver]]).
    Beam plans wander, so `_shorten` deletes verified blocks longest-first. Such
    a level ships a plan that WINS but is not proved shortest, and its labels are
    the walk ties along the plan's own macros rather than measured sets.

``--plans`` names the tier per level, so a level whose labels are inferred says
so in the report rather than hiding among the proved ones. Measured, the tiers
answer levels 0-2, 4 and 6 exactly (d* = 17, 27, 15, 35, 33) and levels 3 and 5
with the beam (41 and 93 presses).

**Level 7 is skipped**, and it is the only one. It gives both characters a goal
in the other's world, eleven platforms and a mutual dependency -- each has to
build the other's bridges before the other can move -- and neither tier finds a
win: the exact tier exhausts its budget and every rung of the beam ladder runs
its layers dry. That is stated in `SharedBridgesSolver.skip_levels` and printed
by every check command rather than left to be inferred from a level count.

Rendering
---------
Two changes, both in data/puzzlescript_games/Shared_Bridges_Game.txt (see the
comments there), on top of the earlier green->blue recolor:

  * **Raised red land was the same ARC palette entry as sunk red land** (darkred
    and ``#300`` both quantize to 13), so a field of red land looked identical
    whichever world was up -- and which world is up is the one bit this game
    turns on. Raised red is now orange (12).
  * **A sunk platform was opaque**, so a bridge over sunk red land was
    pixel-identical to a bridge over sunk blue land -- i.e. you could not see
    which character will be able to shove it next. It is now a plank with its
    left and right columns transparent.

``--audit`` is the check. It enumerates every cell composition the mechanic can
produce, renders each one both ISOLATED and as a whole FIELD of itself (the
field is what caught the red-land collision: a raised cell's only distinguishing
mark was the cap it paints into the cell above, which is invisible when that
cell is the same colour), and it asserts that the engine never produces a
composition outside the enumerated list.

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (0..3). The game is not in
`PuzzleScriptAdapter._FLIP_GAMES`: nothing here needs the mirrors, and the art
is a fake-3D elevation view -- every raised object paints a "cap" into the cell
ABOVE it -- so the flips are left off rather than argued for.

Usage (run from the repo root):
    python solvers/generate_shared_bridges_game_training.py --episodes 200 \
        --out data/training_multi_level/puzzlescript_shared_bridges_game
    python solvers/generate_shared_bridges_game_training.py --selfcheck
    python solvers/generate_shared_bridges_game_training.py --plans
    python solvers/generate_shared_bridges_game_training.py --verify
    python solvers/generate_shared_bridges_game_training.py --audit

``--selfcheck`` / ``--plans`` / ``--verify`` take ``--all`` to include the
skipped level (minutes: it re-runs the whole beam ladder to re-establish that
nothing wins it).
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (                     # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from solvers.common.ps_astar import (                           # noqa: E402
    Plan, PSAStarSolver, PSExpert, restore, snapshot)

GAME_NAME = "Shared_Bridges_Game"

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}

#: The four walk keys, and the full press alphabet. ACTION is the world switch,
#: so unlike most of this family it is a real move and stays in the search.
WALK = ("up", "down", "left", "right")
DIRS = WALK + ("action",)

RED, GREEN = 0, 1                      # who is steering / which world is raised
INF = 1 << 30

#: The objects `_Board` is read off. The `*cap` decorations are pure art (the
#: only rule that reads one deletes it), so they are not part of the state.
_CHARS = {RED: ("redmid", "redlow", "redhi"),
          GREEN: ("greenmid", "greenlow", "greenhi")}
_WORLDS = {RED: ("redworldmid", "redworldlow"),
           GREEN: ("greenworldmid", "greenworldlow")}
_GOALS = {RED: ("redgoalmid", "redgoallow", "redgoalhi"),
          GREEN: ("greengoalmid", "greengoallow", "greengoalhi")}


# ---------------------------------------------------------------------------
# The level, natively
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry plus the turn, natively.

    Cells are flat ``r * w + c`` ids. ``world[cell]`` is 0 for the void (the
    ``#`` border, which is Background and nothing else), 1 for red land and 2 for
    blue land, and it never changes -- what changes is which of the two is
    RAISED, and that is exactly ``turn``.

    A state is ``(turn, red cell, blue cell, platform cells)``; the platform
    tuple is sorted and may hold a cell TWICE, which is the stack (a bridge with
    a crate riding on it). Everything the interpreter draws follows from those
    four: a platform on raised land is a PlatformMid, on sunk land the first is a
    PlatformLow and a second rides on top as a PlatformMid; the steered character
    is a CharMid, the sleeper a CharLow, or a CharHi when it is standing on a
    PlatformMid.
    """

    __slots__ = ("h", "w", "world", "rgoal", "ggoal", "noswitch", "start",
                 "nbr", "_adm", "_prog", "_push")

    def __init__(self, h, w, world, rgoal, ggoal, noswitch, start):
        self.h, self.w = h, w
        self.world = world
        self.rgoal, self.ggoal = frozenset(rgoal), frozenset(ggoal)
        self.noswitch = noswitch
        self.start = start
        self.nbr = {}
        for d, (dr, dc) in _DELTA.items():
            table = [-1] * (h * w)
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        table[r * w + c] = nr * w + nc
            self.nbr[d] = table
        self._adm: dict = {}           # crate multiset -> admissible fields
        self._prog: dict = {}          # crate multiset -> progress fields
        self._push: dict = {}          # crate multiset -> push-distance table

    # -- the win --------------------------------------------------------------
    def won(self, s) -> bool:
        """``all Red on RedGoal`` and ``all Green on GreenGoal``.

        A condition naming a character that the level does not ship is vacuously
        true (the adapter's guard only fires when NO player object at all is
        left, which cannot happen here), so levels 0-1 win on the red goal
        alone."""
        _turn, red, green, _cr = s
        return red in self.rgoal and (green < 0 or green in self.ggoal)

    def switchable(self, crates) -> bool:
        """ACTION does something only when nothing forbids it: the invisible
        NoSwitching object, a platform stacked on a platform, or -- vacuously --
        having nobody to switch to."""
        return (not self.noswitch and self.start[2] >= 0
                and len(set(crates)) == len(crates))

    # -- one turn -------------------------------------------------------------
    def step(self, s, d: str):
        """One press, natively. Mirrors the interpreter rule for rule; the
        ordering below is the rules section's own ordering.

        The push chain: the pressed direction hands a force to the PlatformMid in
        front, which hands it on down the line of PlatformMids. A platform under
        a sleeper is never given one (``no sleeper`` in both push rules), and the
        whole turn is CANCELLED -- the character does not step either -- if the
        far end of the chain would leave the board or hit a sleeper.
        """
        turn, red, green, crates = s
        if d == "action":
            if not self.switchable(crates):
                return s
            return (1 - turn, red, green, crates)
        me, other = (red, green) if turn == RED else (green, red)
        cnt: dict = {}
        for x in crates:
            cnt[x] = cnt.get(x, 0) + 1
        world = self.world
        nb = self.nbr[d]
        n = nb[me]
        if n < 0:
            return s
        # the chain of pushed PlatformMids, and where it ends
        chain = []
        c = n
        while c >= 0 and c != other:
            k = cnt.get(c, 0)
            if not k or not (world[c] - 1 == turn or k == 2):
                break                              # no PlatformMid here
            chain.append(c)
            c = nb[c]
        if chain:
            if c < 0 or not world[c] or c == other:
                return s                           # cancel: the void, or a sleeper
            new = list(crates)
            for x in reversed(chain):
                new.remove(x)
                new.append(nb[x])
            return (turn, *((n, green) if turn == RED else (red, n)),
                    tuple(sorted(new)))
        # no push: the step needs raised land or a bridge, and an empty cell
        if n == other or not world[n]:
            return s
        if not (world[n] - 1 == turn or cnt.get(n, 0)):
            return s
        return (turn, *((n, green) if turn == RED else (red, n)), crates)

    # -- the macro move -------------------------------------------------------
    def expand(self, s):
        """``(dist, ends, goals, macros, win)`` for one state.

        ``dist`` is the walk distance from the steered character to every cell it
        can reach WITHOUT an event -- a bare move fires no rule, so walking is
        free-form and reversible and every interleaving of it is the same board.
        ``macros`` are the events: ``(cost, cell to stand on, press, next
        state)`` for every push it could reach, and -- when a switch is legal --
        one per cell it could park on. ``win`` is ``(cost, cell)`` when simply
        walking home ends the level, i.e. when the OTHER character is already on
        its goal.

        The goal cell is a SINK in ``dist`` when that is so: arriving there ends
        the episode, so no route may pass through it and nothing may be done
        from it.
        """
        turn, red, green, crates = s
        me, other = (red, green) if turn == RED else (green, red)
        cnt: dict = {}
        for x in crates:
            cnt[x] = cnt.get(x, 0) + 1
        world, nbrs = self.world, self.nbr
        goals = self.rgoal if turn == RED else self.ggoal
        ends = other < 0 or other in (self.ggoal if turn == RED else self.rgoal)

        def walkable(x):
            w = world[x]
            if not w or x == other:
                return False
            k = cnt.get(x, 0)
            raised = w - 1 == turn
            if not (raised or k):
                return False
            return not (k and (raised or k == 2))  # a PlatformMid is a push

        dist = {me: 0}
        order = [me]
        qi = 0
        while qi < len(order):
            x = order[qi]
            qi += 1
            if ends and x in goals:
                continue                           # arriving here ends the level
            for d in WALK:
                y = nbrs[d][x]
                if y >= 0 and y not in dist and walkable(y):
                    dist[y] = dist[x] + 1
                    order.append(y)

        win = None
        if ends:
            hit = min(((dist[x], x) for x in goals if x in dist), default=None)
            if hit is not None:
                win = hit

        macros = []
        for c in sorted(set(crates)):
            k = cnt[c]
            if not (world[c] - 1 == turn or k == 2):
                continue                           # not a PlatformMid: not pushable
            for d in WALK:
                back = nbrs[_OPP[d]][c]
                if back < 0 or back not in dist or (ends and back in goals):
                    continue
                src = (turn, *((back, green) if turn == RED else (red, back)),
                       crates)
                ns = self.step(src, d)
                if ns != src:
                    macros.append((dist[back] + 1, back, d, ns))
        if self.switchable(crates):
            for x in order:
                if ends and x in goals:
                    continue
                ns = (1 - turn, *((x, green) if turn == RED else (red, x)),
                      crates)
                macros.append((dist[x] + 1, x, "action", ns))
        return dist, ends, goals, macros, win

    # -- walking --------------------------------------------------------------
    def to_dists(self, dist, ends, goals, target):
        """Walk distance FROM every reachable cell TO ``target``.

        Not just the forward BFS reversed: when ``ends`` the goal cells have no
        outgoing edges, so they may be a destination but never an intermediate.
        """
        dt = {target: 0}
        q = deque([target])
        while q:
            x = q.popleft()
            for d in WALK:
                y = self.nbr[d][x]
                if (y in dist and y not in dt
                        and not (ends and y in goals)):
                    dt[y] = dt[x] + 1
                    q.append(y)
        return dt

    def first_steps(self, me, dist, dt) -> list:
        """Every press that starts a SHORTEST walk from ``me`` toward the cell
        ``dt`` was built from -- the tie set for one step of a walk."""
        here = dt[me]
        return [d for d in WALK
                if (y := self.nbr[d][me]) in dist and dt.get(y, INF) == here - 1]

    # -- heuristics -----------------------------------------------------------
    def push_table(self, crates):
        """Turn-charged pushes to get SOME platform onto each land cell.

        A platform only moves with the character directly behind it, so every
        change of pushing side costs the walk round it -- 2 presses to swap to a
        perpendicular side, 3 to reach the opposite one. Dijkstra over
        ``(cell, incoming direction)`` with those penalties, relaxed: it ignores
        who may push, what is in the way and whether the pusher can get there.
        Only the progress score reads it, so the relaxation costs nothing but
        sharpness (see [[ps-pegs-solver]] for the same table used exactly)."""
        got = self._push.get(crates)
        if got is not None:
            return got
        n = self.h * self.w
        d = [INF] * (n * 4)
        pq = []
        for c in set(crates):
            for i in range(4):
                d[c * 4 + i] = 0
                heapq.heappush(pq, (0, c * 4 + i))
        while pq:
            dv, key = heapq.heappop(pq)
            if dv > d[key]:
                continue
            x, i = divmod(key, 4)
            for j, dd in enumerate(WALK):
                y = self.nbr[dd][x]
                if y < 0 or not self.world[y]:
                    continue
                turn_cost = 0 if j == i else (3 if WALK[j] == _OPP[WALK[i]] else 2)
                nd = dv + 1 + turn_cost
                if d[y * 4 + j] > nd:
                    d[y * 4 + j] = nd
                    heapq.heappush(pq, (nd, y * 4 + j))
        got = [min(d[x * 4:x * 4 + 4]) for x in range(n)]
        self._push[crates] = got
        return got

    def _fields(self, crates, cost_fn):
        """Per-character distance-to-goal arrays under a per-cell entry cost."""
        cnt: dict = {}
        for x in crates:
            cnt[x] = cnt.get(x, 0) + 1

        def field(goals, mine):
            d = [INF] * (self.h * self.w)
            pq = []
            for x in goals:
                d[x] = 0
                heapq.heappush(pq, (0, x))
            while pq:
                dv, x = heapq.heappop(pq)
                if dv > d[x]:
                    continue
                for dd in WALK:
                    y = self.nbr[dd][x]
                    if y < 0 or not self.world[y]:
                        continue
                    w = (1 if (self.world[y] == mine or cnt.get(y, 0))
                         else cost_fn(y))
                    if d[y] > dv + w:
                        d[y] = dv + w
                        heapq.heappush(pq, (d[y], y))
            return d
        return (field(self.rgoal, 1),
                field(self.ggoal, 2) if self.ggoal else None)

    def h_adm(self, s) -> int:
        """A LOWER bound on the presses left, so A* with it is exact.

        Every press either moves the steered character exactly one cell -- a
        push moves it too, the character steps into the cell the platform just
        left -- or switches the world, or is refused. So the presses left are at
        least ``red's travel + blue's travel + the switches``, travel is bounded
        below by a plain walk over the land graph (every move goes between
        adjacent land cells whatever is standing on them), and one switch is
        needed whenever the character NOT being steered still has to move.

        The tempting sharpening -- charge 2 rather than 1 for stepping onto a
        cell of the other world, since it is not bridged now and some press must
        put a platform there -- is NOT a bound, and ``--verify``'s exhaustive
        pass is what says so. The bridging press is itself a step: shoving the
        passenger off a stack one deeper into the chasm moves the character onto
        the bridge in the same press, which is exactly how a gap is crossed, so
        the extra 1 is already counted. Measured on level 0 it over-estimated 105
        of 2112 live states and pruned the shortest plan out of the search."""
        turn, red, green, crates = s
        got = self._adm.get(crates)
        if got is None:
            got = self._fields(crates, lambda _y: 1)
            self._adm[crates] = got
        fr, fg = got
        dr = min(fr[red], INF)
        dg = 0 if green < 0 else min(fg[green], INF)
        other = dg if turn == RED else dr
        return min(dr + dg + (1 if other else 0), INF)

    def h_prog(self, s, k: int = 3) -> int:
        """The beam's ordering score. NOT a bound: an unbridged cell on the route
        is charged the turn-charged push distance of getting a platform to it, so
        the score DROPS as a platform is shoved toward the gap it is needed at.
        The admissible bound above cannot do that -- it charges the same 2 for a
        platform one push away and one across the board -- and a beam ordered by
        it finds nothing."""
        turn, red, green, crates = s
        got = self._prog.get(crates)
        if got is None:
            pd = self.push_table(crates)
            got = self._fields(crates, lambda y: 1 + k * min(pd[y], 200))
            self._prog[crates] = got
        fr, fg = got
        dr = min(fr[red], INF)
        dg = 0 if green < 0 else min(fg[green], INF)
        other = dg if turn == RED else dr
        return min(dr + dg + (1 if other else 0), INF)


# ---------------------------------------------------------------------------
# Search tier 1: exact -- macro A*, then the shortest-path subgraph
# ---------------------------------------------------------------------------

def _astar(board: _Board, cap: int):
    """The shortest press count from ``board.start``, or None past ``cap``
    macro states.

    A win is pushed onto the frontier as a node and returned only when it is
    POPPED. Macros cost different amounts -- a walk-then-push can be 12 presses
    and a switch 1 -- so returning the first win GENERATED returns a plan that is
    merely early, not short (the ps:esl_puzzle_game bug in this family)."""
    start = board.start
    if board.won(start):
        return 0
    pq = [(board.h_adm(start), 0, 0, start)]
    best = {start: 0}
    ctr = 0
    while pq:
        f, g, _c, s = heapq.heappop(pq)
        if s is None:
            return g                                    # the win, popped in order
        if g > best.get(s, INF):
            continue
        _dist, _ends, _goals, macros, win = board.expand(s)
        if win is not None:
            ctr += 1
            heapq.heappush(pq, (g + win[0], g + win[0], ctr, None))
        for cost, _tgt, _press, ns in macros:
            ng = g + cost
            if board.won(ns):
                ctr += 1
                heapq.heappush(pq, (ng, ng, ctr, None))
                continue
            if best.get(ns, INF) <= ng:
                continue
            best[ns] = ng
            ctr += 1
            heapq.heappush(pq, (ng + board.h_adm(ns), ng, ctr, ns))
        if len(best) > cap:
            return None
    return None


def _field(board: _Board, dstar: int):
    """``(distance, edges, wins)`` -- exact distance-to-win for every macro state
    a SHORTEST plan can contain, and the subgraph it was derived from.

    One bounded forward sweep keeping ``g + h <= d*`` -- which is exactly the
    states A* had to look at -- recording every edge it keeps, then one backward
    Dijkstra over those edges from the winning ones. A state kept by the bound
    has its whole shortest continuation inside the sweep, so its distance is
    exact; a state outside can only be over-priced, because a truncated subgraph
    omits edges rather than inventing them. That is what makes the labels
    MEASURED rather than inferred."""
    start = board.start
    edges: dict = {}
    wins: dict = {}
    best = {start: 0}
    pq = [(board.h_adm(start), 0, 0, start)]
    ctr = 0
    while pq:
        _f, g, _c, s = heapq.heappop(pq)
        if g > best.get(s, INF):
            continue
        _dist, _ends, _goals, macros, win = board.expand(s)
        out = []
        if win is not None and g + win[0] <= dstar:
            wins[s] = win[0]
        for cost, _tgt, _press, ns in macros:
            ng = g + cost
            if board.won(ns):
                if ng <= dstar:
                    wins[s] = min(wins.get(s, INF), cost)
                continue
            if ng + board.h_adm(ns) > dstar:
                continue
            out.append((ns, cost))
            if best.get(ns, INF) <= ng:
                continue
            best[ns] = ng
            ctr += 1
            heapq.heappush(pq, (ng + board.h_adm(ns), ng, ctr, ns))
        edges[s] = out

    rev: dict = {}
    for u, out in edges.items():
        for v, cost in out:
            rev.setdefault(v, []).append((u, cost))
    dist: dict = {}
    pq = [(c, i, u) for i, (u, c) in enumerate(sorted(wins.items()))]
    heapq.heapify(pq)
    ctr = len(pq)
    while pq:
        dv, _i, u = heapq.heappop(pq)
        if u in dist:
            continue
        dist[u] = dv
        for p, cost in rev.get(u, ()):
            if p not in dist:
                ctr += 1
                heapq.heappush(pq, (dv + cost, ctr, p))
    return dist, edges, wins


def _emit_exact(board: _Board, dist: dict) -> Plan:
    """The shortest plan, press by press, with the EXACT optimal set at each
    step.

    At every state the candidates are the macros that keep the total at the
    state's own distance-to-win; a candidate you are already standing on
    contributes its event press, one you are not contributes every first step of
    a shortest walk toward it. That union IS the optimal press set: a shortest
    plan decomposes into macros, so an optimal first press is the first press of
    some optimal macro."""
    s = board.start
    presses, optsets = [], []
    while not board.won(s):
        d, ends, goals, macros, win = board.expand(s)
        turn, red, green, _cr = s
        me = red if turn == RED else green
        cands = []
        if win is not None:
            cands.append((win[0], win[1], None))
        for cost, tgt, press, ns in macros:
            rest = 0 if board.won(ns) else dist.get(ns)
            if rest is not None:
                cands.append((cost + rest, tgt, press))
        if not cands:                                    # pragma: no cover
            raise AssertionError("no optimal macro at a state on a shortest path")
        bestv = min(c[0] for c in cands)
        opts: set = set()
        for total, tgt, press in cands:
            if total != bestv:
                continue
            if d[tgt] == 0:
                opts.add(press)
            else:
                opts |= set(board.first_steps(
                    me, d, board.to_dists(d, ends, goals, tgt)))
        best = sorted(opts, key=DIRS.index)
        presses.append(best[0])
        optsets.append(best)
        s = board.step(s, best[0])
    return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# Search tier 2: a width-capped beam over the same macros
# ---------------------------------------------------------------------------

def _beam(board: _Board, width: int, k: int, max_depth: int = 90):
    """A macro plan (not proved shortest), or None.

    Layered by macro depth, each layer kept to ``width`` states ordered by
    ``g + h_prog``. States are deduplicated across the whole run, so a layer
    inherits only what no shallower layer already reached."""
    start = board.start
    if board.won(start):
        return []
    layer = [(0, start, ())]
    seen = {start}
    for _depth in range(max_depth):
        nxt = []
        for g, s, path in layer:
            _d, _ends, _goals, macros, win = board.expand(s)
            if win is not None:
                return list(path) + [(None, win[1])]
            for cost, tgt, press, ns in macros:
                if board.won(ns):
                    return list(path) + [(press, tgt)]
                if ns in seen:
                    continue
                seen.add(ns)
                nxt.append((g + cost, ns, path + ((press, tgt),)))
        if not nxt:
            return None
        nxt.sort(key=lambda t: t[0] + board.h_prog(t[1], k))
        layer = nxt[:width]
    return None


def _replay(board: _Board, plan):
    """``(press count, macros used)`` for a macro plan, or None if some macro's
    cell is no longer reachable or the plan does not win -- which is how
    `_shorten` tests a deletion.

    ``macros used`` is how far it got before the win, because a deletion can make
    the plan win EARLIER: the macros after that point are never executed, and
    leaving them on would have `_emit_beam` pressing on past the end of the
    episode."""
    s = board.start
    total = 0
    for i, (press, tgt) in enumerate(plan):
        d, _ends, _goals, _macros, _win = board.expand(s)
        if tgt not in d:
            return None
        total += d[tgt]
        turn, red, green, crates = s
        s = (turn, *((tgt, green) if turn == RED else (red, tgt)), crates)
        if board.won(s):
            return total, i + 1
        if press is None:
            return None                          # walked home, nobody home yet
        ns = board.step(s, press)
        if ns == s:
            return None
        total += 1
        s = ns
        if board.won(s):
            return total, i + 1
    return None


def _shorten(board: _Board, plan):
    """Delete verified blocks of macros, longest first.

    A beam plan wins but wanders -- it will shove a platform somewhere, change
    its mind and shove it back -- and every such excursion is a contiguous run of
    macros whose removal leaves a plan that still replays to a win. Cheap:
    `_replay` is the native model."""
    plan = list(plan)
    changed = True
    while changed:
        changed = False
        for size in range(len(plan) - 1, 0, -1):
            for i in range(len(plan) - size):        # never drop the last macro
                cand = plan[:i] + plan[i + size:]
                got = _replay(board, cand)
                if got is not None:
                    plan = cand[:got[1]]
                    changed = True
                    break
            if changed:
                break
    return plan


def _emit_beam(board: _Board, plan) -> Plan:
    """A macro plan, press by press, with INFERRED optimal sets: every step of a
    walk is labelled with every direction that keeps it on a shortest route to
    the same cell (no rule fires on a bare move, so those really are the same
    plan), and an event press is labelled with itself."""
    s = board.start
    presses, optsets = [], []
    for press, tgt in plan:
        while not board.won(s):
            d, ends, goals, _macros, _win = board.expand(s)
            turn, red, green, _cr = s
            me = red if turn == RED else green
            if me == tgt:
                break
            dt = board.to_dists(d, ends, goals, tgt)
            best = sorted(board.first_steps(me, d, dt), key=DIRS.index)
            presses.append(best[0])
            optsets.append(best)
            s = board.step(s, best[0])
        if press is None or board.won(s):
            break
        presses.append(press)
        optsets.append([press])
        s = board.step(s, press)
    if not board.won(s):                                 # pragma: no cover
        raise AssertionError("the emitted press sequence does not win")
    return Plan(presses, optsets)


#: Beam configurations, tried in order until one returns. Wider and cheaper
#: first: (width, push-distance weight).
_BEAM_LADDER = ((20_000, 3), (40_000, 2), (60_000, 1))


def solve(board: _Board, node_cap: int, beam: bool = True):
    """``(Plan, tier)`` for one level, strongest tier first, or ``(None, None)``."""
    dstar = _astar(board, node_cap)
    if dstar is not None:
        plan = _emit_exact(board, _field(board, dstar)[0])
        assert len(plan) == dstar, (len(plan), dstar)
        return plan, "exact"
    if not beam:
        return None, None
    for width, k in _BEAM_LADDER:
        macro = _beam(board, width, k)
        if macro is not None:
            return _emit_beam(board, _shorten(board, macro)), "beam"
    return None, None


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class SharedBridgesExpert(PSExpert):
    """`PSExpert`'s plan memo, disk cache and restore discipline around the
    native search above.

    `_search` is replaced the way ps:dotsnake and ps:escaping_limbo replace it:
    the base class keeps the memo, the level scoping and the snapshot discipline,
    and the strategy underneath becomes "read the board off the engine, solve it
    natively". The interpreter is stepped only by the recorder -- which is also
    what certifies every plan, since a level is kept only when the engine reports
    WIN.
    """

    directions = list(DIRS)
    #: Set: the searches ARE the cost of generation and they are seed-independent
    #: (the engine state after reset is the same for every seed; only the
    #: presentation is augmented), so without a file on disk every shard
    #: `parallelize_generator` starts re-derives all of them -- and the beam
    #: levels are ~1 minute each.
    plan_cache_path = (Path(__file__).resolve().parent.parent / "data"
                       / "shared_bridges_plans.json")

    #: Macro states the exact tier may keep before it gives up and the beam runs.
    exact_node_cap = 400_000

    def setup(self) -> None:
        g = self.g
        self.ids = {n: g.obj_name_to_idx[n] for n in
                    ("background", "noswitching", "platformlow", "platformmid")}
        for kind, table in (("char", _CHARS), ("world", _WORLDS),
                            ("goal", _GOALS)):
            for who, names in table.items():
                self.ids[f"{kind}{who}"] = {g.obj_name_to_idx[n] for n in names}
        self.tiers: dict = {}
        self._level: int | None = None

    def heuristic(self, eng) -> int:                      # pragma: no cover
        raise AssertionError(
            "SharedBridgesExpert plans natively; the base A* is never used")

    # -- reading the engine ---------------------------------------------------
    def read(self, eng):
        """The ``(turn, red, blue, platforms)`` state off the engine grid.

        ``turn`` is read from which character is a CharMid -- the steered one --
        which is the same bit as which world is raised."""
        w = len(eng.grid[0])
        red = green = -1
        turn = None
        crates = []
        idx = self.g.obj_name_to_idx
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                k = r * w + c
                if cell & self.ids["char0"]:
                    red = k
                    if idx["redmid"] in cell:
                        turn = RED
                if cell & self.ids["char1"]:
                    green = k
                    if idx["greenmid"] in cell:
                        turn = GREEN
                if self.ids["platformlow"] in cell:
                    crates.append(k)
                if self.ids["platformmid"] in cell:
                    crates.append(k)
        if turn is None:                                  # pragma: no cover
            raise AssertionError("no character is being steered")
        return (turn, red, green, tuple(sorted(crates)))

    def board(self, eng) -> _Board:
        """The `_Board` for the engine's CURRENT grid: static geometry off the
        world/goal objects, start state off `read`."""
        h, w = eng.height, eng.width
        world = [0] * (h * w)
        rgoal, ggoal = [], []
        noswitch = False
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                k = r * w + c
                if cell & self.ids["world0"]:
                    world[k] = 1
                if cell & self.ids["world1"]:
                    world[k] = 2
                if cell & self.ids["goal0"]:
                    rgoal.append(k)
                if cell & self.ids["goal1"]:
                    ggoal.append(k)
                if self.ids["noswitching"] in cell:
                    noswitch = True
        start = self.read(eng)
        # A level with one character and no NoSwitching would let ACTION sink the
        # world out from under it without handing control anywhere, breaking the
        # "raised world == whoever is steering" invariant `_Board` is built on.
        # No shipped level does that; asserted rather than assumed.
        assert start[2] >= 0 or noswitch, "a lone character that can still switch"
        assert not ggoal or start[2] >= 0, "a blue goal with no blue character"
        return _Board(h, w, world, rgoal, ggoal, noswitch, start)

    # -- planning -------------------------------------------------------------
    def plan(self, eng, level: int | None = None):
        """Remember which level `_search` is being asked about, and keep the tier
        that answered it beside the cached plan (the base's disk entry ignores
        the extra key), so a warm process still reports honestly."""
        self._level = level
        found = super().plan(eng, level)
        entry = self._disk.get(level) if self.plan_cache_path else None
        if entry is not None:
            if level in self.tiers:
                if entry.get("tier") != self.tiers[level]:
                    entry["tier"] = self.tiers[level]
                    self._save_disk()
            elif entry.get("tier"):
                self.tiers[level] = entry["tier"]
        return found

    def _search(self, eng):
        board = self.board(eng)
        plan, tier = solve(board, self.exact_node_cap)
        if tier is not None:
            self.tiers[self._level] = tier
        return plan


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SharedBridgesSolver(PSAStarSolver):
    game_id = "puzzlescript_shared_bridges_game"
    game_name = GAME_NAME
    expert_cls = SharedBridgesExpert

    #: Level 7 gives both characters a goal in the other's world, eleven
    #: platforms and a mutual dependency -- each has to build the other's bridges
    #: -- and neither tier finds a win for it (the exact tier runs out of budget,
    #: and every beam configuration in `_BEAM_LADDER` exhausts). Skipped up front
    #: so discovery does not pay for that at every cold start; the other seven
    #: levels are an episode's worth on their own. See `_report`.
    skip_levels = frozenset({7})

    #: Unused -- the expert never calls the base A* -- but left at the family
    #: default so a future subclass that does is not silently starved.
    node_cap = 400_000
    weight = 1
    #: The adapter GAME_OVERs at 200 actions per level, and RESET (which ends the
    #: exploration prefix) zeroes that counter, so the whole budget is the plan's.
    #: The longest plan here is level 5's beam plan.
    max_steps = 200

    #: Zero, the family default. These mechanics are irreversible in the way that
    #: matters -- a platform shoved into a corner of the chasm can only be
    #: recovered by the OTHER character, from the other world -- and outside the
    #: exact tier the expert cannot answer "is this still winnable" for the state
    #: a detour lands in. Recovery data comes from the explore-then-RESET prefix
    #: that opens every episode instead.
    epsilon = 0.0


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _setup(seed: int = 0):
    """``(game, expert)`` without the solver's discovery pass."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=seed)
    return game, SharedBridgesExpert(game)


def selfcheck(argv=(), trials: int = 12, steps: int = 60,
               verbose: bool = True) -> int:
    """Drive random presses through BOTH the interpreter and `_Board.step`,
    comparing the whole state AND the win flag after every one. This is the guard
    that lets the search trust the native model.

    Two kinds of walk, because they reach different mechanics:

      * from a COLD level start, with no warm-up press -- turn-one bookkeeping is
        precisely the class of thing a warm-up hides;
      * from every 8th press of the level's own PLAN, which is what gets the fuzz
        into the states a random walk from the start almost never finds: a
        platform stacked on a bridge, a character asleep on a platform in the
        other world, a chain of three crates against the void.

    ``action`` is in the alphabet throughout, including on the levels where
    NoSwitching makes it a no-op: that it is one is measured, not read off the
    rules section.

    Returns the number of mismatches."""
    game, expert, planned = _levels(argv)
    eng = game._engine
    total = presses = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        # The expert's own (disk-cached) plan, so the beam levels contribute
        # prefixes too -- their plans are where the deep states live. Every
        # level is FUZZED, including the one no tier solves; it just has no plan
        # to take prefixes from, and asking for one would run the whole beam
        # ladder to re-learn that.
        plan = expert.plan(eng, level) if level in planned else None
        prefixes = [[]]
        if plan is not None:
            prefixes += [list(plan[:i]) for i in range(8, len(plan), 8)]
        rng = random.Random(f"sbg:selfcheck:{level}")
        bad = 0
        for start_i in range(trials):
            prefix = prefixes[start_i % len(prefixes)]
            game.set_level(level)
            s = board.start
            for d in prefix:
                s = board.step(s, d)
                eng.step(d)
            if s != expert.read(eng):
                bad += 1
                print(f"  L{level}: the plan prefix itself diverged")
                break
            for _ in range(steps):
                d = rng.choice(DIRS)
                want = board.step(s, d)
                eng.step(d)
                presses += 1
                got = expert.read(eng)
                if want != got or board.won(want) != eng.check_win():
                    bad += 1
                    print(f"  L{level}: MISMATCH on {d!r}\n"
                          f"    from   {s}\n    model  {want}\n"
                          f"    engine {got}\n"
                          f"    win model={board.won(want)} "
                          f"engine={eng.check_win()}")
                    break
                s = want
                if eng.check_win():
                    break
        total += bad
        if verbose:
            print(f"  L{level}: {'OK' if not bad else f'{bad} MISMATCHES'} "
                  f"({len(prefixes)} start states)")
    print(f"selfcheck: {presses} presses vs the interpreter, {total} mismatches")
    return total


def _levels(argv) -> list:
    """The levels a check command looks at: everything the solver actually
    records, plus the skipped ones only when ``--all`` is passed.

    ``skip_levels`` is stated rather than silent: level 7 is skipped because no
    tier wins it, and re-deriving that verdict costs the whole beam ladder --
    several minutes -- every time anyone asks for a report."""
    solver = SharedBridgesSolver()
    game, expert = _setup()
    levels = [lvl for lvl in range(game.n_levels)
              if "--all" in argv or lvl not in solver.skip_levels]
    skipped = sorted(set(range(game.n_levels)) - set(levels))
    if skipped:
        print(f"  (levels {skipped} skipped: no tier finds a win for them; "
              f"pass --all to re-derive that)")
    return game, expert, levels


def _report(argv=()) -> int:
    """Plan every level, replay each plan through the interpreter and print what
    shipped -- the "is this game still solved" check.

    Reads the expert's plan cache, so it is seconds warm and pays for the
    searches once cold; the TIER that answered each level is cached beside the
    plan, which is what keeps a level whose labels are INFERRED from hiding
    among the proved ones."""
    game, expert, levels = _levels(argv)
    eng = game._engine
    bad = 0
    solved = 0
    for level in levels:
        game.set_level(level)
        board = expert.board(eng)
        t0 = time.time()
        plan = expert.plan(eng, level)
        dt = time.time() - t0
        if plan is None:
            print(f"  L{level}: {eng.height}x{eng.width}  UNSOLVED "
                  f"({len(board.start[3])} platforms, {dt:.0f}s)")
            continue
        solved += 1
        game.set_level(level)
        for d in plan:
            eng.step(d)
        won = eng.check_win()
        bad += not won
        ties = sum(len(x) for x in plan.optsets) / max(1, len(plan))
        print(f"  L{level}: {eng.height}x{eng.width}  {len(plan):3d} presses  "
              f"win={won}  {len(board.start[3]):2d} platforms  "
              f"tier={expert.tiers.get(level, '?'):5s}  "
              f"{ties:.2f} optimal presses/step  {dt:.1f}s")
    print(f"  {solved}/{len(levels)} levels solved; "
          f"'exact' plans are proved shortest with measured labels, "
          f"'beam' plans win with inferred ones")
    return 1 if bad else 0


def _exhaustive(board: _Board, cap: int = 120_000):
    """Every state reachable by PRIMITIVE presses with its exact distance to a
    win, or None past ``cap`` states.

    The independent answer to everything the macro search claims: it uses no
    macros, no heuristic and no bound, so it settles d*, admissibility and the
    tie sets in one object. Only the small levels fit."""
    seen = {board.start}
    order = [board.start]
    qi = 0
    while qi < len(order):
        s = order[qi]
        qi += 1
        if board.won(s):
            continue                       # winning states are terminal
        for d in DIRS:
            n = board.step(s, d)
            if n not in seen:
                if len(seen) >= cap:
                    return None
                seen.add(n)
                order.append(n)
    pred: dict = {}
    for s in order:
        if board.won(s):
            continue
        for d in DIRS:
            pred.setdefault(board.step(s, d), []).append(s)
    dist = {s: 0 for s in order if board.won(s)}
    q = deque(dist)
    while q:
        s = q.popleft()
        for p in pred.get(s, ()):
            if p not in dist:
                dist[p] = dist[s] + 1
                q.append(p)
    return dist


def _verify(argv=()) -> int:
    """Take the labels out of the model and press them on the interpreter.

    Three passes, which fail differently:

      * **Exhaustive primitive field** (the levels small enough for one). It
        settles the three things the macro search only argues: that the macro
        optimum IS the press optimum (same d*), that `_Board.h_adm` really is a
        lower bound at every live state -- the sharper version of it was not, and
        this is what caught it -- and that the shipped optimal SETS are exactly
        the presses that lower the true distance.
      * **Bellman fixpoint** (exact levels only). The distance field is built by
        one forward sweep and one backward Dijkstra over the edges it collected;
        a bug in that inversion would give a self-consistent field, a plan that
        still wins, and tie sets that are quietly wrong -- and the training
        labels ARE those tie sets. Re-derive every distance by relaxing
        ``D(u) = min(win(u), min over edges cost + D(v))`` to a fixpoint over the
        recorded edge list and require agreement everywhere. It must be that
        SAME list: the sweep is bounded by ``g + h <= d*``, so a state off every
        shortest path can legitimately hold an over-estimate (its better
        continuation was never explored), and relaxing over the unbounded macro
        graph instead just re-measures the states the field never promised
        anything about. What IS promised -- exactness along the shortest path --
        is what the third pass presses.
      * **Executable labels** (every level). Walk the plan on the INTERPRETER and
        at each step actually press every direction the label calls optimal: the
        board the engine lands on must be the native model's successor, and -- on
        an exact level -- must be exactly one press closer to the win.
    """
    game, expert, levels = _levels(argv)
    eng = game._engine
    bad = 0
    for level in levels:
        game.set_level(level)
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        tier = expert.tiers.get(level, "?")
        if plan is None:
            print(f"  L{level}: unsolved, nothing to verify")
            continue

        exact = _exhaustive(board)
        if exact is not None:
            if exact.get(board.start) != len(plan):
                bad += 1
                print(f"  L{level}: the plan is {len(plan)} presses where an "
                      f"exhaustive BFS proves {exact.get(board.start)}")
            over = [s for s in exact if board.h_adm(s) > exact[s]]
            if over:
                bad += 1
                print(f"  L{level}: h_adm over-estimates at {len(over)}/"
                      f"{len(exact)} live states -- it is not admissible")
            s = board.start
            for i, (press, opts) in enumerate(zip(plan, plan.optsets)):
                here = exact.get(s)
                want = sorted((d for d in DIRS
                               if exact.get(board.step(s, d)) == here - 1),
                              key=DIRS.index)
                if want != sorted(opts, key=DIRS.index):
                    bad += 1
                    print(f"  L{level}: step {i} ships {opts} where the "
                          f"exhaustive field says {want}")
                s = board.step(s, press)

        dist = None
        if tier == "exact":
            dist, edges, wins = _field(board, len(plan))
            check = {u: wins.get(u, INF) for u in edges}
            changed = True
            while changed:
                changed = False
                for u, out in edges.items():
                    for v, cost in out:
                        if v in check and check[v] + cost < check[u]:
                            check[u] = check[v] + cost
                            changed = True
            wrong = sum(1 for u in edges
                        if check[u] != dist.get(u, INF))
            if wrong:
                bad += 1
                print(f"  L{level}: the field disagrees with its Bellman "
                      f"fixpoint at {wrong}/{len(edges)} states")
            if dist.get(board.start) != len(plan):
                bad += 1
                print(f"  L{level}: the field's start distance is not d*")

        game.set_level(level)
        s = board.start
        pressed = 0
        for i, (press, opts) in enumerate(zip(plan, plan.optsets)):
            if press not in opts:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            here = None if dist is None else _dist_at(board, dist, s)
            before = snapshot(eng)
            for alt in opts:
                eng.step(alt)
                landed = expert.read(eng)
                want = board.step(s, alt)
                pressed += 1
                if landed != want:
                    bad += 1
                    print(f"  L{level}: step {i} label {alt!r}: the interpreter "
                          f"and the model disagree")
                elif dist is not None and not board.won(want):
                    there = _dist_at(board, dist, want)
                    if there is None or there != here - 1:
                        bad += 1
                        print(f"  L{level}: step {i} labels {alt!r} optimal, but "
                              f"it does not land one press closer")
                restore(eng, before)
            eng.step(press)
            s = board.step(s, press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        print(f"  L{level}: tier={tier:5s} {len(plan):3d} steps, {pressed} "
              f"labels pressed on the interpreter"
              f"{'' if dist is None else f', {len(dist)} macro states relaxed'}"
              f"{'' if exact is None else f', {len(exact)} states enumerated'}")
    print(f"verify: {bad} problems")
    return 1 if bad else 0


def _dist_at(board: _Board, dist: dict, s):
    """Presses to a win from ANY state, read off the macro field: the best macro
    from here plus that macro's own distance."""
    if board.won(s):
        return 0
    d, _ends, _goals, macros, win = board.expand(s)
    best = win[0] if win is not None else None
    for cost, _tgt, _press, ns in macros:
        rest = 0 if board.won(ns) else dist.get(ns)
        if rest is not None and (best is None or cost + rest < best):
            best = cost + rest
    return best


# ---------------------------------------------------------------------------
# Rendering audit
# ---------------------------------------------------------------------------

def _compositions(g):
    """Every cell composition the mechanic can produce, as ``{name: object set}``.

    Derived from the same rules `_Board` models -- a platform's height follows
    the land under it, a character's follows the platform under it, a goal
    floats on whatever it is standing on -- so the list doubles as a statement of
    what the model believes the engine draws. `_audit`'s third pass holds the
    engine to it."""
    I = g.obj_name_to_idx
    comps = {"void": {I["background"]}}
    for world, raised, crates in itertools.product("RG", (False, True), (0, 1, 2)):
        if raised and crates == 2:
            continue          # two PlatformMids in one cell: same layer, impossible
        mid = (crates == 1) if raised else (crates == 2)
        low = bool(crates) and not raised
        for char, goal in itertools.product((None, "red", "green"),
                                            (None, "red", "green")):
            objs = {I["background"],
                    I[("redworld" if world == "R" else "greenworld")
                      + ("mid" if raised else "low")]}
            if low:
                objs.add(I["platformlow"])
            if mid:
                objs.add(I["platformmid"])
            name = f"{world}{'^' if raised else 'v'}{'x' * crates}"
            if char:
                # a character is steered exactly when its own world is up
                active = raised if char[0] == world.lower() else not raised
                objs.add(I[char + ("mid" if active else
                                   ("hi" if mid else "low"))])
                name += f"+{char[0].upper()}{'A' if active else 's'}"
            if goal:
                objs.add(I[goal + "goal" + ("hi" if mid else
                                            ("mid" if (raised or low) else "low"))])
                name += f"@{goal[0]}"
            comps[name] = objs
    # The NoSwitching flag levels 0-1 carry is a Transparent object on a void
    # cell, so it renders as bare void and is DELIBERATELY not part of the
    # comparison below: it would be reported as a clash with plain void, which
    # is true and is the design. What an agent can see instead is that those two
    # levels have no second character, and ACTION is a no-op exactly there.
    invisible = {"void+noswitching": {I["background"], I["noswitching"]}}
    return comps, invisible


def _audit(walk: int = 300) -> int:
    """Assert every cell composition the game can show is distinct in the frame,
    and that the game never shows one this file has not enumerated.

    Three passes:

      * each composition ISOLATED in a board of void, whole frames compared --
        `_render_frame` upscales the board to fill 64x64 and letterboxes it, so
        an arithmetic cell crop reads the wrong window;
      * each composition as a whole FIELD of itself, which is the adversarial
        surround and the one that convicted raised-vs-sunk red land: a raised
        cell's only mark was the cap it paints into the cell above, and that cap
        is invisible when the cell above is the same colour;
      * a random walk per level, asserting every cell the engine produces is in
        the enumerated list (minus the ``*cap`` decorations, which are art the
        late rules recreate every turn).
    """
    game, expert = _setup()
    eng, g = game._engine, game._game
    comps, invisible = _compositions(g)
    caps = {i for n, i in g.obj_name_to_idx.items() if n.endswith("cap")}

    def frame(cellobjs, fill):
        h, w = 11, 12
        if fill:
            layout = [[set(cellobjs) for _ in range(w)] for _ in range(h)]
        else:
            layout = [[{g.obj_name_to_idx["background"]} for _ in range(w)]
                      for _ in range(h)]
            layout[5][5] = set(cellobjs)
        eng.load_level(layout)
        return np.asarray(_render_frame(eng, g)).copy()

    bad = 0
    for label, fill in (("isolated", False), ("field", True)):
        shots = {k: frame(v, fill) for k, v in comps.items()}
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        for a, b in clashes:
            print(f"  IDENTICAL ({label}): {a} == {b}")
        print(f"  {label}: {len(shots)} compositions, {len(clashes)} clashes")

    known = {frozenset(v) for v in (*comps.values(), *invisible.values())}
    unknown: set = set()
    for level in range(game.n_levels):
        game.set_level(level)
        rng = random.Random(f"sbg:audit:{level}")
        for _ in range(walk):
            eng.step(rng.choice(DIRS))
            for row in eng.grid:
                for cell in row:
                    key = frozenset(cell) - caps
                    if key not in known:
                        unknown.add(tuple(sorted(key)))
            if eng.check_win():
                game.set_level(level)
    bad += len(unknown)
    names = {i: n for n, i in g.obj_name_to_idx.items()}
    for key in sorted(unknown):
        print("  UNENUMERATED cell: " + "+".join(names[i] for i in key))
    print(f"  live: {len(unknown)} compositions the enumeration does not cover")
    print("audit clean" if not bad else f"AUDIT FAILED: {bad} problems")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        sys.exit(1 if selfcheck(sys.argv) else 0)
    if "--plans" in sys.argv:
        sys.exit(_report(sys.argv))
    if "--verify" in sys.argv:
        sys.exit(_verify(sys.argv))
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(SharedBridgesSolver.main())
