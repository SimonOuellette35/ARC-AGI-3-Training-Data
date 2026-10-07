"""Generate Phase-1 training data for the PuzzleScript game ps:puzzleboi
("Puzzleboi" by Tristan Shawn Den Ouden).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, the native model of the turn, the macro search, the exact
distance field it certifies, and the optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_puzzleboi",
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
A sokoban where the crates are not the goal -- they are the PARTS. Five levels,
and the arc of all five is the same: shove three lettered pieces into a row so
they weld into one useful object, then shove THAT into the thing it opens.

  * **The two forges.** ``right [ Crate | Crate2 | Crate3 ] -> [ | Key | ]``
    replaces a left-to-right triple of the three crate shapes with a single Key
    in the MIDDLE cell (the outer two cells are emptied), and
    ``right [ Bridge1 | Bridge2 | Bridge3 ] -> [ | Bridge | ]`` does the same for
    the three plank shapes. Both are ordinary rules, so they are checked BEFORE
    anything moves: lining the triple up does not weld it, the NEXT press does,
    whatever that press is. That one-turn delay is a real part of the mechanic
    and the model reproduces it rather than smoothing it away.
  * **What the two products open.** ``[ > Key | LDoor ] -> [ | Door ]`` spends
    the key and turns the locked door into the Door the win condition names;
    ``[ > Key | LWall ] -> [ | ]`` spends it to blow a hole in a locked wall
    instead. ``[ > Bridge | Water ] -> [ > BridgeA | ]`` lays a plank over
    water: the Water is deleted and the Bridge becomes a BridgeA, which lives on
    the Target/Door collision layer -- i.e. it stops being an obstacle and
    becomes FLOOR. Each of the three is a one-shot spend of an irreversible
    resource, which is what makes this game a puzzle rather than a maze.
  * **Ordering, because rules run in order.** The four push rules come first, so
    an object gets its force from the PRE-weld board; then the welds fire; then
    the key/bridge conversions; then everything moves. The visible consequence:
    on the press that welds a key, the player CANNOT push that key, because rule
    4 already ran and the Key did not exist yet. It can only walk into the two
    cells the weld emptied.
  * **No chain pushes** (no ``[ > Crate | Crate ]`` rule), so an object shoved
    into another object cancels the whole press, the player included. The five
    presses are the four arrows and ACTION; **ACTION is a WAIT that still ticks
    the welds**, which is why it is in the search alphabet and in the labels
    rather than assumed away (see ps:ouroboros).
  * **Dead rules.** EvilSneevil and Target have legend letters, rules and a
    collision-layer slot, and no shipped level places either. ``--plans``
    asserts that, because half the rules section is about a monster that is not
    in the game.
  * **No ``restart``, no ``again``, no ``random``** -- one press is one
    deterministic board rewrite. (`--selfcheck` measures that too.)

Model + search
--------------
`_Board.step` re-implements one turn natively. ``--selfcheck`` drives ~114k
random presses through BOTH the interpreter and the model -- 60k over the five
shipped levels and 54k over six hand-built boards whose only job is to make the
rare rules fire -- and compares every object's cells and the win flag after each
one. The synthetic boards are the point: a random walk on the shipped levels
never once welds a key or opens a door (``--selfcheck`` prints the per-rule
coverage counters, and that is exactly what they showed), so a model verified on
those alone would be verified on the push rules only.

The reachable space is far too large to enumerate (three crates loose in an open
room is already millions of states), so `_Level` searches MACROS -- ``walk to a
push cell, then push``, plus ``walk onto a Door`` -- costed in PRIMITIVE presses.
That is sound here because a bare move changes nothing on the board except the
pending weld, and the weld fires on the FIRST press of any macro, so a macro's
walk is a shortest path on the POST-weld free grid from beginning to end. The
heuristic is a sum of independent push-distance lower bounds (relaxed so that
only Walls block): the pushes that must build and deliver a key, plus -- when the
door is provably unreachable without laying a plank -- the pushes that must
build and deliver a bridge. Those two never share a press with each other or
with the final step onto the Door, so the sum is admissible and weight-1 A*
returns a PROVED shortest plan. ``d*`` is 26, 21, 62, 78 and 24.

That proof leans on the heuristic being admissible, so it is corroborated by
something that does not: ``--verify`` runs an exhaustive primitive BFS -- no
heuristic, no macros -- and gets the same five numbers. Three levels fit its
default budget in seconds; the other two need ``--verify --cap 12000000``, which
reaches level 0's win at depth 26 after 1.2M states and level 2's at depth 62
after 10.0M (~7 GB, ~10 min). It also measures ``h(s) <= d(s)`` directly, on
every state the field has an exact distance for.

The exact field, and what it is for
-----------------------------------
Finding one shortest plan is not enough to LABEL one. After the A* has proved
``d*``, `_Level.close` re-runs the macro search to exhaustion under the bound
``g + h <= d* + slack`` and inverts the edge set it collects, which gives the
EXACT distance-to-win for every macro state a near-shortest route can contain
(the bound is sound: if ``g(s) + d(s) <= bound`` then the whole of s's shortest
path satisfies it too, so the backward sweep can never miss an edge of it).
Mid-walk states are not macro states, and re-searching for each of them would
cost more than the level did; instead `_Level._walk_field` seeds every push cell
of a board with ``1 + d(state after the push)`` and every Door cell with 0 and
runs one Dijkstra over the walk graph, which answers "how far from a win is the
player standing HERE on THIS board" for every cell at once. `optimal` is then a
comparison of exact distances, and it is the labels -- not just the plan -- that
``--verify`` re-derives and presses on the interpreter.

The ``slack`` is what buys recovery. With it, the field also covers the states an
epsilon detour lands in, so ``epsilon = 0.10`` ships: the recorder takes a random
legal alternative roughly one press in ten, the detour is kept only if the field
still has a finite distance for it, and the step records the mistake as the
action taken and the MEASURED recovery as ``optimal``. That is on top of the
explore-then-RESET prefix that opens every episode.

Rendering
---------
Two objects recoloured and four sprites opened up, in the game FILE (see the
comment at the top of data/puzzlescript_games/Puzzleboi.txt). LWall and LDoor
shipped byte-identical -- same colours, same sprite -- although a key spent on
one is a key not spent on the other, and the fourth level ships one of each.
Bridge and BridgeA shipped byte-identical too, which is the difference between a
wall and a floor. ``--audit`` is the check and it compares WHOLE FRAMES rather
than cell crops (`_render_frame` upscales the board to fill 64x64, so an
arithmetic cell crop reads the wrong window).

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (0..3); ``Puzzleboi`` is NOT
a `PuzzleScriptAdapter._FLIP_GAMES` member and ``--symmetry`` is the measurement
that says why. Both weld rules are written with the ABSOLUTE ``right`` prefix, so
turning or mirroring the LEVEL changes the game: a triple that welds in one
presentation does not in another. The rotation augmentation is unaffected by
that -- it rotates the FRAME and remaps the buttons, so the interpreter still
plays the shipped board -- and the presentation stays readable because the
asymmetric Wall art and the crates' own interlocking end caps turn with it.

Usage (run from the repo root):
    python solvers/generate_puzzleboi_training.py --episodes 200 \
        --out data/training_multi_level/puzzleboi
    python solvers/generate_puzzleboi_training.py --selfcheck
    python solvers/generate_puzzleboi_training.py --plans
    python solvers/generate_puzzleboi_training.py --verify [--cap 12000000]
    python solvers/generate_puzzleboi_training.py --audit
    python solvers/generate_puzzleboi_training.py --symmetry
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
import time
from array import array
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (                     # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from solvers.common.ps_astar import PSAStarSolver, PSExpert     # noqa: E402

GAME_NAME = "Puzzleboi"

INF = 1 << 29

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
_DIRS = ("up", "down", "left", "right")

#: The full press alphabet. ACTION carries no movement force, but the two weld
#: rules fire on ANY press, so it is a WAIT -- and a wait is a move. It is in the
#: search, in the plans and in the optimal sets; `selfcheck` fuzzes it too rather
#: than reading "no rule mentions action" off the rules section (ps:ouroboros).
_PRESSES = ("up", "down", "left", "right", "action")

#: Every object whose cells can change. Walls never do and are held once per
#: level; Background is in every parsed cell and carries no information. A state
#: is ``(player cell,) + one sorted cell tuple per slot``, in this order.
_SLOTS = ("crate", "crate2", "crate3", "key",
          "bridge1", "bridge2", "bridge3", "bridge", "bridgea",
          "water", "lwall", "ldoor", "door", "block")
_SI = {n: i + 1 for i, n in enumerate(_SLOTS)}

#: The objects the player can shove, and the rules that do it (rules 1-4, 8-12).
#: Order is the rule order, which is also the order `_Board.step` resolves a tie
#: in -- there can only ever be one object in the cell ahead, since they all
#: share a collision layer, so the order is documentation rather than a decision.
_PUSHABLE = ("crate", "crate2", "crate3", "key", "block",
             "bridge1", "bridge2", "bridge3", "bridge")

#: Collision layer 3 -- these block the player and each other.
_LAYER3 = ("crate", "crate2", "crate3", "key", "bridge1", "bridge2", "bridge3",
           "bridge", "water", "lwall", "ldoor", "block")
#: Collision layer 2 -- walked over freely, and the only things that can stop a
#: BridgeA sliding into the water cell it just drained.
_LAYER2 = ("bridgea", "door")


# ---------------------------------------------------------------------------
# One turn, natively
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry (walls, dimensions) plus `step`, one press.

    Cells are flat ``r * w + c`` ids. The state is a plain tuple so it doubles as
    its own canonical key: ``(player,) + (cells of each `_SLOTS` entry)``, each
    slot sorted. Nothing else in the level moves.
    """

    __slots__ = ("h", "w", "walls", "start", "_next")

    def __init__(self, h: int, w: int, walls, start):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.start = start
        # ``_next[d][cell]`` is the neighbour one step along d, or -1 off the
        # board. Walls are NOT folded in here: several rules need to know which
        # object sits in the cell ahead, and a wall is just another blocker.
        self._next = {}
        for d, (dr, dc) in _DELTA.items():
            table = [-1] * (h * w)
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        table[r * w + c] = nr * w + nc
            self._next[d] = table

    def won(self, st) -> bool:
        """``All Player on Door`` -- one Player on every shipped level."""
        return st[0] in st[_SI["door"]]

    def step(self, st, d: str):
        """One press, in rule order.

        The order is the whole subtlety of this game, so it is spelled out:
        the push rules assign forces from the PRE-weld board (1-4 for the crates
        and the key, 8-12 for the block and the planks); then the crate weld
        (5); then the key's two conversions (6, 7); then the plank weld (13);
        then the bridge-into-water conversion (19); and only then does anything
        move. So a weld can delete the very object this press was shoving, which
        leaves the player pushing nothing and walking into the emptied cell.
        """
        p = st[0]
        S = {n: set(st[_SI[n]]) for n in _SLOTS}
        nxt = self._next[d] if d != "action" else None
        ahead = -1 if nxt is None else nxt[p]

        # -- rules 1-4, 8-12: whatever is in the cell ahead is given the force --
        pushed = None
        if ahead >= 0:
            for n in _PUSHABLE:
                if ahead in S[n]:
                    pushed = (n, ahead)
                    break

        # -- rule 5: right [ Crate | Crate2 | Crate3 ] -> [ | Key | ] ----------
        w = self.w
        for c1 in sorted(S["crate"]):
            if (c1 % w <= w - 3 and c1 + 1 in S["crate2"] and c1 + 2 in S["crate3"]):
                S["crate"].discard(c1)
                S["crate2"].discard(c1 + 1)
                S["crate3"].discard(c1 + 2)
                S["key"].add(c1 + 1)
                if (pushed is not None and pushed[0] in ("crate", "crate2", "crate3")
                        and pushed[1] in (c1, c1 + 1, c1 + 2)):
                    pushed = None            # the force outlived its object

        # -- rules 6 and 7: the key is spent on a locked door, or a locked wall -
        if pushed is not None and pushed[0] == "key":
            tgt = nxt[pushed[1]]
            if tgt >= 0 and tgt in S["ldoor"]:
                S["key"].discard(pushed[1])
                S["ldoor"].discard(tgt)
                S["door"].add(tgt)
                pushed = None
            elif tgt >= 0 and tgt in S["lwall"]:
                S["key"].discard(pushed[1])
                S["lwall"].discard(tgt)
                pushed = None

        # -- rule 13: right [ Bridge1 | Bridge2 | Bridge3 ] -> [ | Bridge | ] --
        for b1 in sorted(S["bridge1"]):
            if (b1 % w <= w - 3 and b1 + 1 in S["bridge2"] and b1 + 2 in S["bridge3"]):
                S["bridge1"].discard(b1)
                S["bridge2"].discard(b1 + 1)
                S["bridge3"].discard(b1 + 2)
                S["bridge"].add(b1 + 1)
                if (pushed is not None
                        and pushed[0] in ("bridge1", "bridge2", "bridge3")
                        and pushed[1] in (b1, b1 + 1, b1 + 2)):
                    pushed = None

        # -- rule 19: [ > Bridge | Water ] -> [ > BridgeA | ] ------------------
        # The plank changes class in place and KEEPS the force, so it is the
        # BridgeA that slides into the cell the Water just vacated -- on the
        # other collision layer, which is what makes the crossing walkable.
        if pushed is not None and pushed[0] == "bridge":
            tgt = nxt[pushed[1]]
            if tgt >= 0 and tgt in S["water"]:
                S["bridge"].discard(pushed[1])
                S["water"].discard(tgt)
                S["bridgea"].add(pushed[1])
                pushed = ("bridgea", pushed[1])

        # -- movement ---------------------------------------------------------
        def free3(cell):
            if cell in self.walls:
                return False
            return not any(cell in S[n] for n in _LAYER3)

        if ahead >= 0:
            if pushed is None:
                if free3(ahead):
                    p = ahead
            else:
                n, cell = pushed
                tgt = nxt[cell]
                if n == "bridgea":
                    ok = tgt >= 0 and not any(tgt in S[k] for k in _LAYER2)
                else:
                    ok = tgt >= 0 and free3(tgt)
                if ok:
                    S[n].discard(cell)
                    S[n].add(tgt)
                    p = ahead
                elif n == "bridgea":
                    # A plank that converted but could not slide (its target's
                    # layer-2 slot is taken). The player's own destination is
                    # free on layer 3 either way -- the Bridge left it.
                    if free3(ahead):
                        p = ahead
        return (p,) + tuple(tuple(sorted(S[n])) for n in _SLOTS)


def _read(eng, slot_of: dict, player_ids: set):
    """The state tuple from an engine grid."""
    w = len(eng.grid[0])
    p = None
    sets: dict[str, list] = {n: [] for n in _SLOTS}
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            for i in cell:
                if i in player_ids:
                    p = r * w + c
                    continue
                n = slot_of.get(i)
                if n is not None:
                    sets[n].append(r * w + c)
    if p is None:                                        # pragma: no cover
        raise AssertionError("the board has no Player")
    return (p,) + tuple(tuple(sorted(sets[n])) for n in _SLOTS)


# ---------------------------------------------------------------------------
# The level: macro search, the bounded exact field, the labels
# ---------------------------------------------------------------------------

class _Level:
    """One level's search and its exact distance-to-win field.

    `solve` finds ``d*`` with weight-1 A* over macros; `close` then re-runs the
    same search to exhaustion under ``g + h <= d* + slack`` and inverts the edges
    it collects, which is what turns "a shortest plan" into "the distance from
    every state a near-shortest route can contain". Everything the recorder asks
    for -- the plan, the optimal set at each step, and whether an epsilon detour
    is still winnable -- is a lookup in that field.
    """

    #: How far past ``d*`` the field is extended. Its whole job is recovery: an
    #: epsilon detour is one press off the shortest path, and undoing a detour
    #: that shoved something takes the player a walk around it, so the states a
    #: single mistake can reach sit a handful of presses above ``d*``. Six costs
    #: roughly twice the bare ``d*`` closure (level 2: 154k macro states and 106s
    #: against 83k and 60s) and buys an 88% detour acceptance rate, measured over
    #: three seeds. A detour past it is refused by the recorder's probe, which is
    #: a conservative answer and never a wrong one.
    slack = 6

    def __init__(self, board: _Board):
        self.b = board
        h, w, N = board.h, board.w, board.h * board.w
        self.h, self.w, self.N = h, w, N
        self.nx = board._next
        self.free = [c not in board.walls for c in range(N)]
        st = board.start

        # -- push-distance tables, relaxed so that only Walls block ------------
        # A relaxation can only make an object cheaper to move, so every table
        # below is a lower bound on real pushes, which is what the heuristic
        # needs. `_pt[t]` answers "pushes to bring an object from x to t".
        self._pt = {c: self._push_table([c]) for c in range(N) if self.free[c]}
        #: Cells where a horizontal triple fits at all (both welds need one).
        self.asm = [c for c in range(N)
                    if c % w <= w - 3 and all(self.free[c + i] for i in range(3))]
        #: Pushes to shove a key into SOME locked door, the shove included.
        self.keydel = self._deliver(st[_SI["ldoor"]])
        #: Pushes to shove a plank into SOME water cell, the shove included.
        self.brdel = self._deliver(st[_SI["water"]])

        self.d_star: int | None = None
        self.bound = 0
        self._index: dict = {}
        self._g = array("i")
        self._dist = array("i")
        self._wf: dict = {}
        self._memo: dict = {}

    # -- geometry -------------------------------------------------------------
    def _push_table(self, targets) -> list:
        """Reverse BFS: min pushes to bring an object from each cell to
        ``targets``. A push from x to x+d needs x+d free (where it lands) and
        x-d free (where the player must stand)."""
        d = [INF] * self.N
        q = deque()
        for t in targets:
            if self.free[t] and d[t] == INF:
                d[t] = 0
                q.append(t)
        while q:
            y = q.popleft()
            for dd in _DIRS:
                x = self.nx[_OPP[dd]][y]
                if x < 0 or not self.free[x]:
                    continue
                stand = self.nx[_OPP[dd]][x]
                if stand < 0 or not self.free[stand]:
                    continue
                if d[x] > d[y] + 1:
                    d[x] = d[y] + 1
                    q.append(x)
        return d

    def _deliver(self, sinks) -> list:
        """Min pushes to shove an object INTO one of ``sinks`` -- the table above
        to a cell beside a sink, plus the shove itself."""
        appr = set()
        for s in sinks:
            for dd in _DIRS:
                x = self.nx[_OPP[dd]][s]
                if x >= 0 and self.free[x]:
                    appr.add(x)
        if not appr:
            return [INF] * self.N
        t = self._push_table(appr)
        return [v + 1 if v < INF else INF for v in t]

    # -- heuristic ------------------------------------------------------------
    def heuristic(self, st) -> int:
        """A lower bound on the presses left, in three disjoint parts.

        Every press moves at most one object, so counting the pushes an object
        still owes is a lower bound, and two different objects never owe the same
        press. The parts:

          * **the key.** With no Door on the board one must be made: either shove
            an existing Key into a locked door, or weld the three crates
            somewhere and shove the resulting key in. The weld's own cost is the
            three crates' push distances to a triple of cells.
          * **the bridge**, but only when it is PROVABLY needed: flood the board
            with Water blocking and everything movable treated as passable (an
            over-approximation of where the player can get), and if no door is in
            that flood then a water cell must be drained, which no rule but
            ``[ > Bridge | Water ]`` can do. Its cost is the same shape as the
            key's.
          * **one press** for the final step onto the Door, which is a walk: the
            key that opened it was consumed doing so, leaving the cell clear.

        Relaxed push tables (only Walls block) keep every part a lower bound, so
        the sum is admissible and weight-1 A* proves ``d*``.
        """
        p = st[0]
        if p in st[_SI["door"]]:
            return 0

        if st[_SI["door"]]:
            need_key = 0
        else:
            need_key = INF
            for k in st[_SI["key"]]:
                if self.keydel[k] < need_key:
                    need_key = self.keydel[k]
            c1, c2, c3 = st[_SI["crate"]], st[_SI["crate2"]], st[_SI["crate3"]]
            if c1 and c2 and c3:
                x1, x2, x3 = c1[0], c2[0], c3[0]
                for a in self.asm:
                    v = self._pt[a][x1] + self._pt[a + 1][x2] + self._pt[a + 2][x3]
                    if v >= need_key:
                        continue
                    v += self.keydel[a + 1]
                    if v < need_key:
                        need_key = v
            if need_key >= INF:
                return INF                       # no key can ever be delivered

        need_bridge = 0
        water = st[_SI["water"]]
        if water and not self._door_reachable(st, water):
            need_bridge = INF
            for bx in st[_SI["bridge"]]:
                if self.brdel[bx] < need_bridge:
                    need_bridge = self.brdel[bx]
            b1, b2, b3 = st[_SI["bridge1"]], st[_SI["bridge2"]], st[_SI["bridge3"]]
            if b1 and b2 and b3:
                y1, y2, y3 = b1[0], b2[0], b3[0]
                for a in self.asm:
                    v = self._pt[a][y1] + self._pt[a + 1][y2] + self._pt[a + 2][y3]
                    if v >= need_bridge:
                        continue
                    v += self.brdel[a + 1]
                    if v < need_bridge:
                        need_bridge = v
            if need_bridge >= INF:
                return INF                       # the far side is sealed off
        return need_key + need_bridge + 1

    def _door_reachable(self, st, water) -> bool:
        """Could the player reach a door cell if every movable thing were simply
        walked through? Walls and Water are the only two blockers that no rule in
        this game can shift out of the way except by draining the water, so a
        'no' here PROVES that a plank still has to be laid."""
        blocked = set(water) | self.b.walls
        goal = set(st[_SI["door"]]) or set(st[_SI["ldoor"]])
        seen = {st[0]}
        q = deque([st[0]])
        while q:
            x = q.popleft()
            if x in goal:
                return True
            for dd in _DIRS:
                y = self.nx[dd][x]
                if y < 0 or y in blocked or y in seen:
                    continue
                seen.add(y)
                q.append(y)
        return False

    # -- macros ---------------------------------------------------------------
    def _welded(self, st):
        """The board as the NEXT press will see it: ACTION carries no force, so
        `step(st, "action")` is exactly the two weld rules and nothing else."""
        return self.b.step(st, "action")

    def _walk_map(self, view):
        """``(dist, parent)`` for the player over the free cells of ``view``.

        Walking is what a macro spends most of its presses on, and it is a plain
        BFS because a bare move changes nothing -- the one thing a press can
        change without moving anything is the weld, and ``view`` has it already
        applied, which is legitimate because the weld fires on the FIRST press of
        the macro whatever that press is."""
        occ = set(self.b.walls)
        for n in _LAYER3:
            occ.update(view[_SI[n]])
        p = view[0]
        dist = {p: 0}
        par: dict = {}
        q = deque([p])
        while q:
            x = q.popleft()
            for dd in _DIRS:
                y = self.nx[dd][x]
                if y < 0 or y in occ or y in dist:
                    continue
                dist[y] = dist[x] + 1
                par[y] = (x, dd)
                q.append(y)
        return dist, par

    def macros(self, st):
        """``(cost, presses)`` for every ``walk then push`` and every
        ``walk onto a Door``.

        Completeness: a press either moves the player alone, moves the player and
        one object, or is refused; and the only board change a walk can carry is
        the pending weld, which every macro's first press performs. So any press
        sequence is a sequence of these, and taking a non-shortest walk between
        two pushes is never cheaper."""
        view = self._welded(st)
        dist, par = self._walk_map(view)

        def path(cell):
            out = []
            while cell != view[0]:
                x, dd = par[cell]
                out.append(dd)
                cell = x
            out.reverse()
            return out

        out = []
        for n in _PUSHABLE:
            for x in view[_SI[n]]:
                for dd in _DIRS:
                    pc = self.nx[_OPP[dd]][x]
                    if pc < 0 or pc not in dist:
                        continue
                    out.append((dist[pc] + 1, path(pc) + [dd]))
        for dc in view[_SI["door"]]:
            if dist.get(dc, 0) > 0:
                out.append((dist[dc], path(dc)))
        return out

    def succ(self, st):
        """``(cost, presses, next state)`` per macro, self-loops dropped."""
        out = []
        for cost, presses in self.macros(st):
            nx = st
            for d in presses:
                nx = self.b.step(nx, d)
            if nx != st:
                out.append((cost, presses, nx))
        return out

    # -- the search and the field ---------------------------------------------
    def solve(self, node_cap: int = 4_000_000):
        """Weight-1 A* over macros. Returns ``(d*, presses)`` or ``(None, None)``.

        The win is tested at POP, not when a winning node is generated: macros
        cost different amounts, so a node popped at ``g = 40`` can generate an
        11-press win (51) while a node popped at ``g = 45`` generates a 3-press
        one (48) -- see ps:esl_puzzle_game, where taking the first generated win
        cost three presses on level 1."""
        b = self.b
        start = b.start
        g = {start: 0}
        parent: dict = {}
        pq = [(self.heuristic(start), 0, 0, start)]
        cnt = gen = 0
        while pq:
            _f, gg, _c, st = heapq.heappop(pq)
            if g.get(st, INF) < gg:
                continue
            if b.won(st):
                seq = []
                while st in parent:
                    pst, presses = parent[st]
                    seq = presses + seq
                    st = pst
                return gg, seq
            for cost, presses, nx in self.succ(st):
                ng = gg + cost
                if ng >= g.get(nx, INF):
                    continue
                hh = self.heuristic(nx)
                if hh >= INF:
                    continue
                g[nx] = ng
                parent[nx] = (st, presses)
                cnt += 1
                gen += 1
                heapq.heappush(pq, (ng + hh, ng, cnt, nx))
                if gen > node_cap:                       # pragma: no cover
                    return None, None
        return None, None

    def close(self, d_star: int):
        """Exhaust the macro search under ``g + h <= d* + slack`` and invert it.

        Soundness of the bound: for any state s on a shortest path out of a state
        t that satisfies it, ``g(s) + d(s) <= g(t) + d(t) <= bound`` and
        ``h(s) <= d(s)``, so s satisfies it too and so does every edge along the
        way. The backward sweep therefore sees the WHOLE of the shortest path of
        every state it reports on -- which is why `dist_macro` reports a distance
        only for the states whose ``g + d`` is inside the bound, and INF (i.e.
        "ask something else") for the rest.
        """
        b = self.b
        bound = self.bound = d_star + self.slack
        start = b.start
        index = {start: 0}
        states = [start]
        g = array("i", [0])
        src = array("i")
        dst = array("i")
        cost_a = array("i")
        pq = [(self.heuristic(start), 0, 0)]
        cnt = 0
        while pq:
            _f, gg, si = heapq.heappop(pq)
            if g[si] < gg:
                continue
            st = states[si]
            if b.won(st):
                continue                                 # terminal: no successors
            for cost, _presses, nx in self.succ(st):
                ng = gg + cost
                if ng > bound:
                    continue
                hh = self.heuristic(nx)
                if hh >= INF or ng + hh > bound:
                    continue
                j = index.get(nx)
                if j is None:
                    j = len(states)
                    index[nx] = j
                    states.append(nx)
                    g.append(INF)
                src.append(si)
                dst.append(j)
                cost_a.append(cost)
                if ng < g[j]:
                    g[j] = ng
                    cnt += 1
                    heapq.heappush(pq, (ng + hh, ng, j))

        # Backward Dijkstra over the reversed edge set (CSR, because a list of
        # per-node lists costs ~100 MB of empty lists before any contents).
        n = len(states)
        deg = array("i", bytes(4 * (n + 1)))
        for j in dst:
            deg[j + 1] += 1
        for i in range(n):
            deg[i + 1] += deg[i]
        pos = array("i", deg[:n])
        radj = array("i", bytes(4 * len(dst)))
        rcost = array("i", bytes(4 * len(dst)))
        for e in range(len(dst)):
            j = dst[e]
            radj[pos[j]] = src[e]
            rcost[pos[j]] = cost_a[e]
            pos[j] += 1

        dist = array("i", [INF]) * n
        heap = []
        for i, st in enumerate(states):
            if b.won(st):
                dist[i] = 0
                heap.append((0, i))
        heapq.heapify(heap)
        while heap:
            dd, i = heapq.heappop(heap)
            if dd > dist[i]:
                continue
            for e in range(deg[i], deg[i + 1]):
                j = radj[e]
                nd = dd + rcost[e]
                if nd < dist[j]:
                    dist[j] = nd
                    heapq.heappush(heap, (nd, j))

        # Keep only what the bound proves exact.
        for i in range(n):
            if dist[i] < INF and g[i] + dist[i] > bound:
                dist[i] = INF
        self._index, self._g, self._dist = index, g, dist
        self.d_star = d_star
        return n, len(dst)

    def build(self) -> bool:
        """Search, then close. False when the level has no plan at all."""
        d_star, seq = self.solve()
        if d_star is None:
            return False
        self.close(d_star)
        assert self.dist_of(self.b.start) == d_star, (
            "the closed field disagrees with the A* it was bounded by")
        return True

    # -- reading the field ----------------------------------------------------
    def dist_macro(self, st) -> int:
        """Exact presses-to-win when ``st`` is a macro state the bound covers."""
        i = self._index.get(st)
        return INF if i is None else self._dist[i]

    def _walk_field(self, board) -> list:
        """Presses-to-win for the player standing on each cell of ``board``.

        This is what makes labelling affordable. A state in the middle of a walk
        is not a macro state and re-searching from every one of them would cost
        more than the level did; but its board is a macro state's board, and from
        any cell the player's options are exactly "walk to a push cell and push"
        or "walk onto a Door". So seed each push cell with ``1 + d(state after
        the push)`` and each Door cell with 0, and one Dijkstra over the walk
        graph answers every cell at once.
        """
        hit = self._wf.get(board)
        if hit is not None:
            return hit
        occ = set(self.b.walls)
        for n in _LAYER3:
            occ.update(board[_SI[n] - 1])
        val = [INF] * self.N
        heap = []
        for dc in board[_SI["door"] - 1]:
            if dc not in occ:
                val[dc] = 0
                heap.append((0, dc))
        for n in _PUSHABLE:
            for x in board[_SI[n] - 1]:
                for dd in _DIRS:
                    pc = self.nx[_OPP[dd]][x]
                    if pc < 0 or pc in occ:
                        continue
                    after = self.b.step((pc,) + board, dd)
                    dm = self.dist_macro(after)
                    if dm >= INF:
                        continue
                    if 1 + dm < val[pc]:
                        val[pc] = 1 + dm
                        heap.append((val[pc], pc))
        heapq.heapify(heap)
        while heap:
            dd, x = heapq.heappop(heap)
            if dd > val[x]:
                continue
            for d in _DIRS:
                y = self.nx[d][x]
                if y < 0 or y in occ:
                    continue
                if dd + 1 < val[y]:
                    val[y] = dd + 1
                    heapq.heappush(heap, (val[y], y))
        self._wf[board] = val
        return val

    def dist_of(self, st) -> int:
        """Exact presses-to-win, INF when the field does not cover ``st``."""
        hit = self._memo.get(st)
        if hit is not None:
            return hit
        if self.b.won(st):
            return 0
        d = self.dist_macro(st)
        if d >= INF:
            welded = self._welded(st)
            if welded != st:
                # A weld is pending. It fires on the next press whatever that
                # press is, but the press that fires it CANNOT also shove the
                # object the weld just created (rule order: the push rules ran
                # first). So this one state is answered by an explicit one-press
                # lookahead rather than off the walk field, which would offer
                # that shove.
                d = 1 + min(self.dist_of(self.b.step(st, p)) for p in _PRESSES)
                d = min(d, INF)
            else:
                d = self._walk_field(st[1:])[st[0]]
        self._memo[st] = d
        return d

    def optimal(self, st) -> list:
        """Every press that lands one press closer to the win.

        Exact: a press that is refused maps the state to itself (or to itself
        plus a weld that was going to happen anyway), and a press that spends a
        key or a plank where it cannot be spent again leaves the field with no
        finite distance for the result -- so both fall out without a special
        case."""
        here = self.dist_of(st)
        if here >= INF or here == 0:
            return []
        return [p for p in _PRESSES
                if self.dist_of(self.b.step(st, p)) == here - 1]

    def plan_from(self, st) -> list | None:
        """A SHORTEST press sequence to a win, or None when the field cannot see
        one. A descent of the field taking the first tied press in `_PRESSES`
        order (arrows before ACTION), so a plan is a pure function of the board.
        """
        if self.dist_of(st) >= INF:
            return None
        out = []
        while not self.b.won(st):
            best = self.optimal(st)
            if not best:                                 # pragma: no cover
                raise AssertionError("the field has no descent")
            out.append(best[0])
            st = self.b.step(st, best[0])
        return out


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class PuzzleboiExpert(PSExpert):
    """Exact planner over `_Level`'s bounded distance field.

    `PSExpert` supplies the plan memo, the snapshot/restore discipline and the
    level scoping; `_search` swaps in the field descent, so the interpreter is
    only ever stepped by the recorder -- which is also what certifies every plan,
    since a level is kept only when the engine itself reports WIN.
    """

    directions = list(_PRESSES)

    #: `_key` is the state tuple, which names no wall and is therefore canonical
    #: only WITHIN a level.
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        self.slot_of: dict[int, str] = {}
        for n in _SLOTS:
            for i in g.resolve_object_name(n):
                self.slot_of[i] = n
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.player_ids = set(self.game._engine._player_indices)
        self._levels: dict[int | None, _Level] = {}
        self._cur: _Level | None = None

    def read(self, eng):
        return _read(eng, self.slot_of, self.player_ids)

    def level(self, eng, level: int | None) -> _Level:
        """The level's `_Level`, searched and closed once and cached.

        The walls are static -- no rule in this game creates or destroys one --
        so whichever state happens to build it describes the same geometry."""
        lv = self._levels.get(level)
        if lv is not None:
            return lv
        w = len(eng.grid[0])
        walls = {r * w + c
                 for r, row in enumerate(eng.grid)
                 for c, cell in enumerate(row) if cell & self.wall_ids}
        lv = _Level(_Board(len(eng.grid), w, walls, self.read(eng)))
        lv.build()
        self._levels[level] = lv
        return lv

    # -- planning -------------------------------------------------------------
    def _key(self, eng):
        return self.read(eng)

    def plan(self, eng, level: int | None = None) -> list | None:
        self._cur = self.level(eng, level)
        return super().plan(eng, level)

    def heuristic(self, eng) -> int:
        """The EXACT remaining press count (the base `_astar` is never reached
        here, but the contract is 0 at a win and admissible, and this is both)."""
        d = self._cur.dist_of(self.read(eng))
        return 0 if d >= INF else d

    def _search(self, eng) -> list | None:
        return self._cur.plan_from(self.read(eng))

    def optimal_dirs(self, eng, level: int | None) -> list:
        return self.level(eng, level).optimal(self.read(eng))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class PuzzleboiSolver(PSAStarSolver):
    game_id = "puzzlescript_puzzleboi"
    game_name = GAME_NAME
    expert_cls = PuzzleboiExpert

    #: Unused -- `PuzzleboiExpert._search` never calls the base A* -- but left at
    #: the family default so a future subclass that does is not silently starved.
    node_cap = 2_000_000
    weight = 1
    #: The adapter GAME_OVERs at 200 actions per level; the longest plan here is
    #: 78 presses, which leaves the exploration prefix and the detours room.
    max_steps = 200

    #: Non-zero in an IRREVERSIBLE game, and it is the bounded field that earns
    #: it. `record_level`'s detour probe steps a random alternative, asks the
    #: expert whether a win is still planned, and keeps the detour only if it is;
    #: for a bounded SEARCH that test is a guess that can decline a live state,
    #: but here it is a lookup in a field that is exact out to ``d* + slack``, so
    #: a kept detour provably still wins and its recovery is the measured
    #: shortest one. Roughly one press in ten is then a random legal alternative,
    #: after which the expert re-plans from wherever it landed: the taken action
    #: is the mistake and ``optimal`` is the recovery. The RESET prefix still runs
    #: in front of it.
    epsilon = 0.10

    def prepare_expert(self, game, expert) -> None:
        """Search and close every level up front.

        Pure front-loading -- the field does not depend on which state builds it
        -- but it means the searches are paid once, at startup, rather than
        inside whichever query happens to need them first."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.level(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route, read off the LIVE
        engine state.

        `record_level` asks for this before executing the press, so the engine is
        still at the state being labelled -- which is what makes the label right
        after an epsilon detour too, where the state is off the plan entirely and
        replaying the plan from the level start would label the wrong states.

        Falls back to the press about to be taken if the field has nothing to say
        -- no expert step may ship unlabelled."""
        best = expert.optimal_dirs(expert.game._engine, level)
        if best:
            return best
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with every level searched and closed."""
    solver = PuzzleboiSolver()
    game, expert, _solvable = solver._ensure(0)
    return solver, game, expert


#: Six hand-built boards whose only job is to make the RARE rules fire. A random
#: walk on the shipped levels welds a key roughly never (measured: zero times in
#: 60k presses), so a model checked on those alone would be checked on the push
#: rules and nothing else -- the ps:idols_to_the_burnt_god lesson, which is that
#: the plan-replay certification cannot find a mechanic the plans do not use.
_CH = {".": (), "#": ("wall",), "P": ("player",), "1": ("crate",),
       "2": ("crate2",), "3": ("crate3",), "K": ("key",), "a": ("bridge1",),
       "b": ("bridge2",), "c": ("bridge3",), "B": ("bridge",),
       "A": ("bridgea",), "~": ("water",), "L": ("lwall",), "l": ("ldoor",),
       "D": ("door",), "%": ("block",),
       # composite cells: a layer-3 object standing on a layer-2 one
       "E": ("bridge", "bridgea"), "F": ("water", "door")}

_FUZZ_BOARDS = (
    ["#######",                      # crates a shove from welding, both locks
     "#P123.#",
     "#..K.l#",
     "#.1.2L#",
     "#..3..#",
     "#######"],
    ["######",                       # two keys beside a door and a wall
     "#PK.l#",
     "#..KL#",
     "#1.2.#",
     "#.3..#",
     "######"],
    ["#######",                      # planks, water and a laid crossing
     "#Pabc~#",
     "#B~~~l#",
     "#~.A.D#",
     "#.%.%.#",
     "#######"],
    ["########",                     # everything at once
     "#P1233~#",
     "#KabcB~#",
     "#lL~%.A#",
     "#12.3K.#",
     "#..~..D#",
     "########"],
    ["#########",                    # overlapping triples: the weld fixpoint
     "#P123123#",
     "#.123...#",
     "#..l.L..#",
     "#K.....K#",
     "#########"],
    ["######",                       # a plank that converts but cannot slide:
     "#PEF.#",                       # the water it drains has a Door under it
     "#B~D.#",
     "#.~..#",
     "######"],
)


def _load_ascii(game, rows):
    """Load an ASCII board straight into the interpreter, and return the
    `_Board` for it. The interpreter parses and resolves it itself, so this is a
    real level and not a grid poked in from outside."""
    idx = game._game.obj_name_to_idx
    layout = [[{idx["background"]} | {idx[n] for n in _CH[ch]} for ch in row]
              for row in rows]
    game._engine.load_level(layout)
    w = len(rows[0])
    walls = {r * w + c for r, row in enumerate(rows)
             for c, ch in enumerate(row) if ch == "#"}
    return layout, walls


def selfcheck(trials: int = 200, steps: int = 60, verbose: bool = True) -> int:
    """Drive random presses through BOTH the interpreter and `_Board.step`,
    comparing every object's cells AND the win flag after every one.

    Two stages, because they find different things. The shipped levels exercise
    the pushes and the geometry; the synthetic boards exercise the welds, the two
    key conversions and the plank-into-water conversion, which random play on the
    shipped levels does not reach at all. The per-rule coverage counters are
    printed for exactly that reason -- a fuzz that reports "no mismatches"
    without having fired a rule has not tested it.

    ACTION is in the press alphabet throughout: the claim that it does nothing is
    false (it ticks the welds), and the kind of claim that should be measured
    rather than read off a rules section.

    The walks run from a COLD level start with no warm-up press, because turn-one
    bookkeeping is precisely the class of mechanic a warm-up hides.

    Returns the number of mismatches."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = PuzzleboiExpert(game)
    eng = game._engine
    cov: dict[str, int] = {}
    total = 0

    def run(tag, restore_fn, board, rng):
        nonlocal total
        bad = presses = 0
        first = None
        for _ in range(trials):
            restore_fn()
            st = board.start
            for _ in range(steps):
                d = rng.choice(_PRESSES)
                want = board.step(st, d)
                _count(cov, st, want)
                eng.step(d)
                presses += 1
                got = expert.read(eng)
                if want != got or board.won(want) != eng.check_win():
                    bad += 1
                    if first is None:
                        first = (d, st, want, got)
                    break
                st = want
                if eng.check_win():
                    break
        total += bad
        if verbose:
            print(f"  {tag}: {'OK' if not bad else f'{bad} MISMATCHES'} "
                  f"({presses} presses vs the interpreter)")
        if first and verbose:
            d, st, want, got = first
            print(f"    first divergence on {d}:")
            for i, n in enumerate(("player",) + _SLOTS):
                if want[i] != got[i]:
                    print(f"      {n}: was {st[i]} model {want[i]} engine {got[i]}")

    for level in range(game.n_levels):
        game.set_level(level)
        run(f"L{level}", lambda lv=level: game.set_level(lv),
            _board_of(eng, expert),
            random.Random(f"puzzleboi:selfcheck:{level}"))
    for bi, rows in enumerate(_FUZZ_BOARDS):
        layout, walls = _load_ascii(game, rows)
        board = _Board(len(rows), len(rows[0]), walls, expert.read(eng))
        run(f"B{bi}", lambda lay=layout: eng.load_level(lay), board,
            random.Random(f"puzzleboi:fuzz:{bi}"))

    if verbose:
        print("  rule coverage: " + ", ".join(
            f"{k}={v}" for k, v in sorted(cov.items())))
        for rule in ("weld_key", "weld_bridge", "key_opens_door",
                     "key_breaks_wall", "plank_drains_water", "plank_stuck"):
            if not cov.get(rule):
                print(f"  WARNING: {rule} never fired -- it is untested")
    return total


def _board_of(eng, expert) -> _Board:
    w = len(eng.grid[0])
    walls = {r * w + c for r, row in enumerate(eng.grid)
             for c, cell in enumerate(row) if cell & expert.wall_ids}
    return _Board(len(eng.grid), w, walls, expert.read(eng))


def _count(cov, st, want):
    """Which of the rules this transition fired -- read off the two states rather
    than instrumented into `_Board.step`, so a counter cannot drift away from the
    model it is counting (or, worse, report coverage a refactor removed)."""
    def bump(k):
        cov[k] = cov.get(k, 0) + 1

    if len(want[_SI["key"]]) > len(st[_SI["key"]]):
        bump("weld_key")
    if len(want[_SI["bridge"]]) > len(st[_SI["bridge"]]):
        bump("weld_bridge")
    if len(want[_SI["door"]]) > len(st[_SI["door"]]):
        bump("key_opens_door")
    if len(want[_SI["lwall"]]) < len(st[_SI["lwall"]]):
        bump("key_breaks_wall")
    drained = set(st[_SI["water"]]) - set(want[_SI["water"]])
    if drained:
        bump("plank_drains_water")
        if not drained & set(want[_SI["bridgea"]]):
            # the plank changed class but its target's layer-2 slot was taken,
            # so it converted in place instead of sliding
            bump("plank_stuck")
    for n in _PUSHABLE:
        if (len(want[_SI[n]]) == len(st[_SI[n]])
                and want[_SI[n]] != st[_SI[n]]):
            bump("push_" + n)


def _report() -> int:
    """Print each level's search, its shortest plan and the engine's verdict --
    the quick "is this game still fully solved" check."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        lv = expert.level(eng, level)
        board = lv.b
        plan = lv.plan_from(board.start)
        if plan is None:
            bad += 1
            print(f"  L{level}: UNSOLVED")
            continue
        st = board.start
        ties = 0
        for d in plan:
            ties += len(lv.optimal(st))
            st = board.step(st, d)
            eng.step(d)
        won = eng.check_win()
        bad += not won
        measured = sum(1 for d in lv._dist if d < INF)
        print(f"  L{level}: {eng.height}x{eng.width}  {len(plan):3d} presses  "
              f"win={won}  d*={lv.d_star}  "
              f"{len(lv._index):6d} macro states searched, "
              f"{measured:5d} with a measured distance  "
              f"{ties / len(plan):.2f} optimal presses/step")
    for name in ("evilsneevil", "target"):
        used = 0
        for level in range(game.n_levels):
            game.set_level(level)
            ids = set(game._game.resolve_object_name(name))
            used += sum(1 for row in eng.grid for cell in row if cell & ids)
        assert not used, f"a level ships a {name} -- the model does not handle it"
    print("  (EvilSneevil and Target have rules and legend letters and appear on "
          "no level)")
    return 1 if bad else 0


def _exhaustive_field(board: _Board, bound: int, cap: int):
    """The primitive ground truth: layered BFS from the level start to depth
    ``bound`` keeping every edge, then one backward sweep from the winning states
    over those edges.

    No heuristic, no macros, no bound on which successors count -- just
    `_Board.step` over all five presses. The result is the exact distance-to-win
    for every state whose ``g + d`` is inside ``bound`` (for such a state the
    whole of its shortest path is inside too, by the same argument `_Level.close`
    rests on), which is exactly the set `_Level`'s field claims to answer for.

    Returns ``({state: distance}, states seen)`` or ``(None, states seen)`` when
    the level's space blows past ``cap`` first."""
    index = {board.start: 0}
    states = [board.start]
    g = [0]
    src: list[int] = []
    dst: list[int] = []
    frontier = [0]
    for depth in range(bound):
        nxt = []
        for i in frontier:
            st = states[i]
            if board.won(st):
                continue                     # terminal: the adapter ends here
            for p in _PRESSES:
                t = board.step(st, p)
                j = index.get(t)
                if j is None:
                    j = len(states)
                    index[t] = j
                    states.append(t)
                    g.append(depth + 1)
                    nxt.append(j)
                src.append(i)
                dst.append(j)
        if len(states) > cap:
            return None, len(states)
        frontier = nxt
        if not frontier:
            break

    radj: list[list[int]] = [[] for _ in states]
    for e in range(len(src)):
        radj[dst[e]].append(src[e])
    dist = [INF] * len(states)
    q = deque()
    for i, st in enumerate(states):
        if board.won(st):
            dist[i] = 0
            q.append(i)
    while q:
        i = q.popleft()
        for j in radj[i]:
            if dist[j] > dist[i] + 1:
                dist[j] = dist[i] + 1
                q.append(j)
    truth = {states[i]: dist[i] for i in range(len(states))
             if dist[i] < INF and g[i] + dist[i] <= bound}
    return truth, len(states)


def _verify(cap: int = 600_000, sample: int = 20_000) -> int:
    """Four passes over each level, which fail differently.

      * **The whole field against a primitive ground truth.** An exhaustive
        layered BFS over all five presses, run to ``d* + slack`` and inverted, on
        the levels whose space fits under ``cap`` (retried at ``d*`` alone, which
        still settles ``d*`` and the shortest-path subgraph, before giving up).
        That is not a check on the plan, it is a check on every LABEL: the
        training targets are differences of field distances, so a field that is
        self-consistent and wrong would still produce a plan that wins. Three of
        the five levels fit the default cap; the other two say so rather than
        passing quietly, and ``--cap 12000000`` reaches them too -- levels 0 and
        2 need 1.2M and 10.0M states (~7 GB, ~9 min) and confirm 26 and 62. That
        is the run that settles ``d*`` for every level of this game without
        appealing to the heuristic at all.
      * **Admissibility, measured.** ``h(s) <= d(s)`` on every state the field
        has an exact distance for. That is the one assumption weight-1 A* rests
        on for the two levels the BFS cannot reach, and it is the assumption a
        careless extra term in `heuristic` would break -- so it is checked on
        real states rather than argued from the derivation.
      * **A Bellman fixpoint on the plan's own states.** Re-derive the distance
        at every state the plan passes through from its five successors alone
        (``d(s) = 1 + min d(succ)``) and require agreement.
      * **Every shipped label, pressed on the interpreter.** At each plan step,
        press each direction the label calls optimal on the REAL engine and
        require the board it lands on to be the native successor and its field
        distance to be exactly one less. That takes the labels out of the model
        and puts them through the thing they will be replayed against.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    rng = random.Random("puzzleboi:verify")
    for level in range(game.n_levels):
        game.set_level(level)
        lv = expert.level(eng, level)
        board = lv.b
        plan = lv.plan_from(board.start)

        t0 = time.time()
        truth = None
        for bound in (lv.d_star + lv.slack, lv.d_star):
            truth, n_seen = _exhaustive_field(board, bound, cap)
            if truth is not None:
                break
        if truth is None:
            note = f"space > {n_seen} states, NOT enumerated"
        elif truth.get(board.start) != lv.d_star:
            bad += 1
            note = (f"BFS says d*={truth.get(board.start)}, "
                    f"the search says {lv.d_star}")
        else:
            items = list(truth.items())
            if len(items) > sample:
                items = rng.sample(items, sample)
            wrong = sum(1 for st, d in items if lv.dist_of(st) != d)
            bad += bool(wrong)
            note = (f"BFS agrees on d* to depth {bound}; {len(items)} of "
                    f"{len(truth)} field states compared, {wrong} disagree "
                    f"({n_seen} explored, {time.time() - t0:.1f}s)")

        measured = [(st, lv._dist[i]) for st, i in lv._index.items()
                    if lv._dist[i] < INF]
        if len(measured) > sample:
            measured = rng.sample(measured, sample)
        loose = sum(1 for st, d in measured if lv.heuristic(st) > d)
        bad += bool(loose)
        note += (f"; h <= d on {len(measured) - loose}/{len(measured)} "
                 f"measured states")

        st = board.start
        labels = 0
        for i, press in enumerate(plan):
            here = lv.dist_of(st)
            best = lv.optimal(st)
            if press not in best:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            relaxed = 1 + min(lv.dist_of(board.step(st, p)) for p in _PRESSES)
            if relaxed != here:
                bad += 1
                print(f"  L{level}: step {i} distance {here} != Bellman {relaxed}")
            for alt in best:
                labels += 1
                after = board.step(st, alt)
                eng.step(alt)
                landed = expert.read(eng)
                if landed != after or lv.dist_of(after) != here - 1:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal but the "
                          f"interpreter does not land one press closer")
                # walk the engine back to the state under test
                game.set_level(level)
                for d in plan[:i]:
                    eng.step(d)
            eng.step(press)
            st = board.step(st, press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        print(f"  L{level}: d*={lv.d_star:3d}  {note};  "
              f"{labels} labels pressed on the interpreter")
    print(f"verify: {bad} problems")
    return 1 if bad else 0


def _transform(k: int, mirror: bool):
    """``(cell_fn, dims_fn, direction_map)`` for one element of the 8-group:
    mirror left-right first, then turn clockwise ``k`` times. The direction map
    is DERIVED from the same linear part that moves the cells, so the two cannot
    drift apart."""
    def cell(rc, hw):
        r, c = rc
        h, w = hw
        if mirror:
            c = w - 1 - c
        for _ in range(k):
            r, c, h, w = c, h - 1 - r, w, h
        return (r, c)

    def dims(hw):
        h, w = hw
        return (w, h) if k % 2 else (h, w)

    dmap = {}
    for name, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[name] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    dmap["action"] = "action"
    return cell, dims, dmap


def _symmetry(walk: int = 300) -> int:
    """Measure whether the 8-element presentation group is a symmetry of the
    MECHANIC, which is what would entitle this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation.

    It is not, and this is the measurement that says so rather than an argument:
    both weld rules are written with the absolute ``right`` prefix, so a triple
    that welds on the shipped board does not weld on the same board turned. Each
    level's LAYOUT is rebuilt under all eight transforms and reloaded, so the
    interpreter re-derives everything itself, and both the plan and a seeded
    random walk are replayed with the presses transformed; every object is
    compared, not just the player.

    The mandatory ROTATION augmentation is untouched by the result -- the adapter
    rotates the rendered FRAME and remaps the buttons, so the interpreter always
    plays the board as shipped. What this rules out is turning the LEVEL, which
    is what a flip would amount to.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    chiral = 0
    for level in range(game.n_levels):
        game.set_level(level)
        lv = expert.level(eng, level)
        layout = g.levels[level]
        hw = (len(layout), len(layout[0]))
        rng = random.Random(f"puzzleboi:symmetry:{level}")
        runs = {"plan": lv.plan_from(lv.b.start),
                "walk": [rng.choice(_PRESSES) for _ in range(walk)]}
        notes = []
        for kind, presses in runs.items():
            eng.load_level(layout)
            ref = [expert.read(eng)]
            for d in presses:
                eng.step(d)
                ref.append(expert.read(eng))
            for k, mirror in itertools.product(range(4), (False, True)):
                if (k, mirror) == (0, False):
                    continue
                cell, dims, dmap = _transform(k, mirror)
                th, tw = dims(hw)
                turned = [[set() for _ in range(tw)] for _ in range(th)]
                for r, row in enumerate(layout):
                    for c, objs in enumerate(row):
                        tr, tc = cell((r, c), hw)
                        turned[tr][tc] = set(objs)
                eng.load_level(turned)
                for i, d in enumerate(presses):
                    eng.step(dmap[d])
                    if expert.read(eng) != _map_state(ref[i + 1], cell, hw, tw):
                        notes.append(f"{kind}:rot{k}{'m' if mirror else ''}"
                                     f"@press{i}")
                        break
        chiral += len(notes)
        print(f"  L{level}: ({len(runs['plan'])} plan + {walk} random) presses "
              f"x 7 presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print("symmetry clean" if not chiral else
          f"CHIRAL in {chiral} presentations -- as expected: the two weld rules "
          f"name the absolute direction `right`, so Puzzleboi is NOT a "
          f"_FLIP_GAMES candidate")
    return 0


def _map_state(state, cell, hw, tw):
    """``state`` with every cell id carried through one transform."""
    def m(x):
        r, c = cell(divmod(x, hw[1]), hw)
        return r * tw + c
    return (m(state[0]),) + tuple(tuple(sorted(m(x) for x in s))
                                  for s in state[1:])


def _audit() -> int:
    """Assert every cell COMPOSITION this game can show is distinct in the
    frame, at every cell size the five levels render at.

    Two pairs are what this exists for, and both shipped BROKEN (see the header
    of data/puzzlescript_games/Puzzleboi.txt): LWall against LDoor -- the two
    things a Key can be spent on, with opposite outcomes -- and Bridge against
    BridgeA, which is the difference between an obstacle and floor.

    The compositions it does NOT list are the ones the game cannot produce, and
    they are asserted rather than assumed: a Wall, an LWall, an LDoor and a Water
    cell are the four things nothing in this game ever moves onto the
    Target/Door layer, and no level ships one on top of a Door or a BridgeA.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where that made an audit report every
    composition identical to floor)."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    # The four immovable layer-3 objects, and the proof that nothing puts a
    # layer-2 object under one of them: no level ships that, and the only rules
    # that create a Door or a BridgeA (6 and 19) name an LDoor and a Water cell
    # they simultaneously delete.
    static3 = ("wall", "lwall", "ldoor", "water")
    for level in range(game.n_levels):
        game.set_level(level)
        for row in eng.grid:
            for c in row:
                over = {n for n in static3 if idx[n] in c}
                under = {n for n in ("door", "bridgea", "target") if idx[n] in c}
                assert not (over and under), \
                    f"level {level} ships {over} on {under}"

    movable3 = ("player", "crate", "crate2", "crate3", "key",
                "bridge1", "bridge2", "bridge3", "bridge", "block")
    comps = [()]
    comps += [(n,) for n in static3]
    comps += [(n,) for n in movable3]
    comps += [(n,) for n in ("door", "bridgea")]
    comps += [(a, b) for a in movable3 for b in ("door", "bridgea")]

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    clashes_total = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for objs in comps:
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[objs] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(comps, 2)
                   if np.array_equal(shots[a], shots[b])]
        clashes_total += len(clashes)
        name = lambda t: "+".join(t) if t else "floor"           # noqa: E731
        print(f"{h}x{w} board (level{'s' if len(levels) > 1 else ''} "
              f"{', '.join(str(x) for x in levels)}), "
              f"cell {max(1, min(64 // h, 64 // w))}px: "
              f"{len(comps)} compositions")
        for a, b in clashes:
            print(f"  IDENTICAL: {name(a)} == {name(b)}")
    print("audit clean" if not clashes_total
          else f"AUDIT FAILED: {clashes_total} indistinguishable pairs")
    return 0 if not clashes_total else 1


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        bad = selfcheck()
        print(f"selfcheck: {bad} mismatches")
        sys.exit(1 if bad else _report())
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        # ``--cap N`` raises the ground-truth BFS budget; the default keeps the
        # check to a couple of minutes and reports the two levels it cannot
        # reach rather than skipping them silently.
        argv_cap = (int(sys.argv[sys.argv.index("--cap") + 1])
                    if "--cap" in sys.argv else 600_000)
        sys.exit(_verify(cap=argv_cap))
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    sys.exit(PuzzleboiSolver.main())
