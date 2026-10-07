"""Generate Phase-1 training data for the PuzzleScript game
ps:dreaming_of_strawberries ("Dreaming of Strawberries" by Yifan Zheng & Rachel
Han -- a wall of blue targets stands between you and the strawberry, and the
only thing that can put a hole in it is an orange crate, which dies doing it).

The harness -- the rotation contract, the trajectory recorder, the
RESET-recovery prefix and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: the mechanic notes, a native model of the interpreter, a
macro A* over that model, and the optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_dreaming_of_strawberries",
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
exactly. Every step also carries the full set of equally-optimal presses.

The game
--------
Win is ``All Player on Target2``: stand on the strawberry. Four arrow keys;
ACTION is bound to nothing, so `StrawberriesExpert.directions` drops it. Six
rules, and every one of them matters:

  * **You PUSH and you PULL.** ``[> Player | Crate] -> [> Player | > Crate]``
    is the ordinary push, but ``[< Player | Crate] -> [< Player | < Crate]`` is
    its mirror: the piece BEHIND you, on the axis you are moving along, gets
    dragged into the square you just left. Both rules are repeated for Crate2.
    So a piece is not a thing you shove around a maze, it is a thing you can
    also tow out of a dead end -- which is the only way to move half the crates
    in this game (level 5's first crate has walls above and below it and a
    Crate2 on its left; the sole legal first move on it is a pull).

    The pull is also why a WALK is not free here. Every other sokoban in this
    tree can move the player anywhere in its region for the price of the
    distance; here, stepping away from a piece you happen to be standing next
    to drags it with you. The walk graph is therefore DIRECTED -- ``p -> p+d``
    is a free step only when ``p-d`` holds no piece -- and it is not symmetric:
    from a square next to a crate you can step towards it (a push) but not away
    (a pull), so reachability is one-way and the usual "player region" dedup key
    of `PSPushExpert` would be unsound. This expert keys on the player's exact
    square instead.

  * **A crate ANNIHILATES the target it lands on**: ``[crate Target] ->
    [Background]`` deletes both. Crate and Target sit on different collision
    layers, so a crate can enter a target square -- and having entered it, both
    are gone. One crate buys exactly one hole. A crate therefore cannot be
    parked inside the barrier and cannot be pushed through it: the second
    square of a two-thick wall is opened by a SECOND crate, pushed in from the
    same outside square and travelling one square further than the first. Every
    level ships exactly as many crates as its barrier is thick, so the whole
    puzzle is "get all of them to the same entry square, in some order".

  * **A Crate2 evaporates NEXT TO a target.** ``[ target | crate2 ] -> [ Target
    | ]`` is a two-cell pattern, not a stack: the yellow crate does not have to
    reach the barrier, only to become orthogonally adjacent to it, and then it
    vanishes and the target STAYS. Crate2 is a decoy -- it looks like ammunition
    and is worth none, it pushes and pulls exactly like a crate, and its only
    real role is as an obstacle that happens to be self-disposing if you can
    shove it at the wall.

  * **Standing on a target RESTARTS the level** (``late [ player Target ] ->
    restart``). The barrier is lethal to walk into, so the player's own walk
    graph excludes every target square, and the search never generates a move
    onto one -- a restart returns to a state already at the root of the search,
    so it can never be part of a shortest plan. NB the restart is executed by
    `PuzzleScriptAdapter.perform_action`, not by ``engine.step``; a search that
    stepped the engine directly would silently see the player standing on the
    barrier.

  * **The deletions happen at the TOP of the next turn**, not at the end of the
    one that caused them: rules run before movement resolves, so a crate shoved
    onto a target is rendered sitting on it for exactly one frame and is gone
    before the next press lands. `Model.step` reproduces that ordering, and the
    audit below keeps the one-frame stack readable.

Expert solver
-------------
`Model` is the whole interpreter, natively, in four sets (player, crates,
crate2s, live targets) -- 40 lines, and ``--selfcheck`` fuzzes it against the
real thing. `Search` then plans on the model, so the interpreter only has to
CERTIFY the plan (``--plans`` replays it), not produce it: a model step is ~1000x
cheaper than ``engine.step``, and the searches need millions of them.

`Search` is an A* whose successors are ``walk to a square beside a piece, then
push or pull it once`` MACROS, plus one ``walk to the strawberry`` macro; ``g``
counts primitive presses, so plans are costed in what the agent actually pays.

The heuristic is where the game is. Distance-to-strawberry alone is FLAT over
the whole middle of every level -- the player spends forty presses fetching
crates without getting one square nearer the goal -- so it has to price the
barrier. It does that by building, per state:

  * a per-crate BFS over the squares a crate can be shuffled through (a crate
    moves to ``q+d`` when ``q+d`` is clear and either ``q-d`` is clear, so it can
    be pushed, or ``q+2d`` is, so it can be pulled). Targets are terminal: a
    crate that enters one dies there;
  * every BREACH of the barrier -- an outside square ``e``, a direction, the run
    of ``T`` target squares beyond it and the square ``x`` where the run ends --
    priced at ``sum of the T nearest crates' distances to e`` (each crate has to
    get to ``e``) ``+ T(T+1)/2`` (the j-th crate is then pushed j squares) ``+
    T`` (the player walks through);
  * a Dijkstra back from the strawberry over open squares, unit cost, with each
    breach as one extra edge at that price.

``h`` is that field read at the player's square, and a state whose field never
reaches the player is DEAD (not enough crates can still reach any breach) and is
dropped rather than queued. The breach list depends only on the target set and
the field only on ``(crates, targets)``, so both are memoised and the cost per
node is a dictionary lookup on the common path. Every term is a relaxation --
pieces do not block the walk, crate distances ignore the player, and no crate is
charged twice -- so at ``weight = 1`` the plans are shortest up to those
relaxations. Like the rest of this family that is a search guide, not an
optimality certificate: the one place it can over-count is that the player's walk
to a breach and the presses that move a crate there are added even where they
overlap.

**All 7 levels solve, in 10 to 87 presses (411 in total, ~31 s cold for all
seven, instant warm off the disk cache)**, and every plan is replayed through
the real interpreter by ``--plans``.

Optimal-action sets
-------------------
A macro is a walk followed by one acting press. The acting press is labelled
with itself alone -- which piece to shove where is the puzzle, and a sibling
push is a different plan, not a reordering of this one. Each step of a walk is
labelled with every direction that keeps the player on a shortest route to the
same square, recomputed on the live board at that step (the directed walk graph
above, so a step that would tow a piece is never offered as an alternative, and
a Crate2 that evaporates mid-walk is accounted for). No step ever ships
unlabelled.

Three art fixes ship with it, in
``data/puzzlescript_games/Dreaming_of_Strawberries.txt``; ``--audit`` is the
regression test. All three are the "an opaque body hides the ground tile it is
standing on" shape, and the first of them made the game unlearnable:

  1. **The winning frame was pixel-identical to a losing one.** Player was a
     solid 5x5 with no transparent pixel anywhere, so the player standing on the
     strawberry rendered as the player standing on grass -- the ONE composition
     the win condition is about was not in the frame. The player now has
     transparent corners and one transparent dimple beside its mouth, placed so
     that a hole survives every sampling grid the renderer uses: ``cell_px`` 2,
     3 and 4 read sprite rows/columns {1,3}, {0,2,4} and {0,1,3,4}, so a body
     transparent only at its corners is still solid on the smallest boards.
     Same constraint on the opaque side, which is why the eyes sit on row 1 --
     the player keeps both its pink and its face at all three sizes.
  2. **The strawberry was mostly transparent**, so those new holes would have
     shown grass through it anyway. Its background pixels are now painted red,
     keeping the green cap, the white seeds and the silhouette's colours.
  3. **Crate and Crate2 were solid too**, so the one frame in which a crate sits
     on the target it is about to annihilate looked exactly like a crate on
     grass -- the game's central event was invisible. Both now carry two
     transparent nicks, on opposite diagonals from each other, which is also
     what keeps them apart at ``cell_px = 2`` where their orange and yellow are
     the only other difference.

Augmentation
------------
The engine state after reset is identical for every seed (the levels are fixed
ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game is now in). Both are exact
symmetries here: there is no gravity, every rule is stated with the relative
``>``/``<`` force, the win condition is positional in no way, input is
screen-relative, and no sprite encodes a direction, so no mirror can turn one
object's art into another's. 7 levels x 16 presentations = 112.

The expert plan is therefore seed-independent: solved once per level, cached to
``data/dreaming_of_strawberries_plans.json``, and replayed per seed with that
seed's remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_dreaming_of_strawberries_training.py --episodes 200 \
        --out data/training_multi_level/dreaming_of_strawberries

    python solvers/generate_dreaming_of_strawberries_training.py --selfcheck
    python solvers/generate_dreaming_of_strawberries_training.py --audit
    python solvers/generate_dreaming_of_strawberries_training.py --plans
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from adapters.puzzlescript_adapter import (                    # noqa: E402
    PuzzleScriptAdapter, _render_cell_sprite)
from solvers.common.ps_astar import (                          # noqa: E402
    Plan, PSAStarSolver, PSExpert)

#: PuzzleScript game name (``data/puzzlescript_games/Dreaming_of_Strawberries.txt``).
GAME_NAME = "Dreaming_of_Strawberries"

#: Where each level's start plan is kept between processes. The searches are the
#: whole cost of generation and are seed-independent, so without a file on disk
#: every shard `parallelize_generator` starts re-derives all seven.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "dreaming_of_strawberries_plans.json"

#: The four engine directions, in the order the labeller reports them. ACTION is
#: bound to no rule in this game.
DIRS = ("up", "down", "left", "right")
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
DV = tuple(DELTA.values())

INF = 1 << 30


# ---------------------------------------------------------------------------
# Native model of the interpreter
# ---------------------------------------------------------------------------

class Model:
    """The whole game, in four sets.

    A state is ``(player, crates, crate2s, targets)``; walls, the board size and
    the strawberry squares never change and live on the model. `step` returns
    the next state for one press, reproducing the interpreter exactly (see
    `selfcheck`).

    The turn order is the one thing here that is not obvious and is not
    negotiable: PuzzleScript runs the RULES and then resolves MOVEMENT, so the
    two deletion rules fire on the board as it stands at the start of the press
    -- i.e. a piece shoved onto (or beside) a target survives, visibly, until the
    *following* press. Doing the deletions after the move instead would make
    every "push the crate in, then walk through the hole" plan one press shorter
    than the interpreter agrees with.
    """

    def __init__(self, eng, idx):
        self.h, self.w = eng.height, eng.width
        self._idx = (idx["player"], idx["crate"], idx["crate2"], idx["target"])
        walls, berries = set(), set()
        for r in range(self.h):
            for c in range(self.w):
                cell = eng.grid[r][c]
                if idx["wall"] in cell:
                    walls.add((r, c))
                if idx["target2"] in cell:
                    berries.add((r, c))
        self.walls = frozenset(walls)
        self.berries = frozenset(berries)
        self.start = self.read(eng)

    def read(self, eng) -> tuple:
        """``(player, crates, crate2s, targets)`` off the interpreter's grid."""
        pi, ci, c2i, ti = self._idx
        player = None
        crates, crate2s, targets = set(), set(), set()
        for r in range(self.h):
            row = eng.grid[r]
            for c in range(self.w):
                cell = row[c]
                if not cell:
                    continue
                if pi in cell:
                    player = (r, c)
                if ci in cell:
                    crates.add((r, c))
                if c2i in cell:
                    crate2s.add((r, c))
                if ti in cell:
                    targets.add((r, c))
        return (player, frozenset(crates), frozenset(crate2s), frozenset(targets))

    def inb(self, p) -> bool:
        return 0 <= p[0] < self.h and 0 <= p[1] < self.w

    def step(self, st, d: str) -> tuple:
        player, crates, crate2s, targets = st

        # -- rule phase, on the board as it stands ---------------------------
        # A Crate2 orthogonally ADJACENT to a target evaporates and the target
        # survives; a Crate sharing a square WITH a target annihilates it.
        crate2s = frozenset(
            q for q in crate2s
            if not any((q[0] + dr, q[1] + dc) in targets for dr, dc in DV))
        hit = crates & targets
        if hit:
            crates = crates - hit
            targets = targets - hit
        if d == "action":
            return (player, crates, crate2s, targets)

        # -- movement --------------------------------------------------------
        dr, dc = DELTA[d]
        front = (player[0] + dr, player[1] + dc)
        back = (player[0] - dr, player[1] - dc)
        if not self.inb(front) or front in self.walls:
            return (player, crates, crate2s, targets)
        if front in crates or front in crate2s:
            beyond = (front[0] + dr, front[1] + dc)
            if (not self.inb(beyond) or beyond in self.walls
                    or beyond in crates or beyond in crate2s):
                return (player, crates, crate2s, targets)   # the push is refused
            if front in crates:
                crates = (crates - {front}) | {beyond}
            else:
                crate2s = (crate2s - {front}) | {beyond}
        if self.inb(back):                                  # the pull
            if back in crates:
                crates = (crates - {back}) | {player}
            elif back in crate2s:
                crate2s = (crate2s - {back}) | {player}
        return (front, crates, crate2s, targets)

    def restarts(self, st) -> bool:
        """``late [ player Target ] -> restart`` -- the barrier is lethal."""
        return st[0] in st[3]

    def won(self, st) -> bool:
        return st[0] in self.berries


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

#: One planned step: the walk that leads to it, the acting press (a push or a
#: pull, or None for the final walk onto the strawberry) and the square the walk
#: ends on -- which is what the labeller needs to say which walk steps were ties.
class Macro(tuple):
    __slots__ = ()

    def __new__(cls, walk, act, stand):
        return super().__new__(cls, (tuple(walk), act, stand))

    @property
    def presses(self):
        return list(self[0]) + ([self[1]] if self[1] is not None else [])


class Search:
    """A* over ``walk + one push-or-pull`` macros on `Model`.

    ``g`` counts primitive presses. Dedup is on the exact state, player square
    included: the walk graph is directed (see the module docstring), so the
    reachable set is not an equivalence class and `PSPushExpert`'s region key
    would merge states with different futures.
    """

    def __init__(self, model: Model, weight: int = 1, node_cap: int = 2_000_000):
        self.m = model
        self.weight = weight
        self.node_cap = node_cap
        self._breaches: dict = {}
        self._fields: dict = {}
        self.nodes = 0

    # -- heuristic ----------------------------------------------------------
    def breaches(self, targets) -> list:
        """``(entry, exit, thickness)`` for every straight run of target squares
        that a crate could be fed into. Depends only on the target set, which
        changes only when a crate is spent."""
        got = self._breaches.get(targets)
        if got is not None:
            return got
        m = self.m
        out = []
        for r in range(m.h):
            for c in range(m.w):
                entry = (r, c)
                if entry in m.walls or entry in targets:
                    continue
                for dr, dc in DV:
                    cur = (r + dr, c + dc)
                    if cur not in targets:
                        continue
                    thickness = 0
                    while cur in targets:
                        thickness += 1
                        cur = (cur[0] + dr, cur[1] + dc)
                    if m.inb(cur) and cur not in m.walls:
                        out.append((entry, cur, thickness))
        self._breaches[targets] = out
        return out

    def crate_fields(self, crates, targets) -> list:
        """One BFS per crate over the squares that crate can be shuffled
        through: it reaches ``q+d`` when ``q+d`` is clear and it can either be
        pushed there (``q-d`` clear) or pulled there (``q+2d`` clear). Targets
        are excluded -- a crate that enters one is annihilated, so it can end on
        a target but never pass through."""
        m = self.m
        fields = []
        for origin in crates:
            dist = {origin: 0}
            queue = deque([origin])
            while queue:
                cur = queue.popleft()
                for dr, dc in DV:
                    nxt = (cur[0] + dr, cur[1] + dc)
                    if (not m.inb(nxt) or nxt in m.walls or nxt in targets
                            or nxt in dist):
                        continue
                    push_from = (cur[0] - dr, cur[1] - dc)
                    pull_to = (nxt[0] + dr, nxt[1] + dc)
                    if not ((m.inb(push_from) and push_from not in m.walls
                             and push_from not in targets)
                            or (m.inb(pull_to) and pull_to not in m.walls
                                and pull_to not in targets)):
                        continue
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
            fields.append(dist)
        return fields

    def field(self, crates, targets) -> dict:
        """Cost-to-strawberry for the player: a unit-cost walk over open squares
        plus one priced edge per breach. Memoised on ``(crates, targets)``, which
        is what makes the heuristic a dict lookup for every node that only moved
        the player."""
        key = (crates, targets)
        got = self._fields.get(key)
        if got is not None:
            return got
        m = self.m
        cfs = self.crate_fields(crates, targets)
        portals: dict = {}
        for entry, exit_, thickness in self.breaches(targets):
            reach = sorted(d for d in (f.get(entry) for f in cfs) if d is not None)
            if len(reach) < thickness:
                continue                       # not enough crates can get there
            cost = (sum(reach[:thickness])
                    + thickness * (thickness + 1) // 2
                    + thickness)
            portals.setdefault(exit_, []).append((entry, cost))

        dist = {b: 0 for b in m.berries}
        queue = [(0, b) for b in m.berries]
        heapq.heapify(queue)
        while queue:
            d, cur = heapq.heappop(queue)
            if d > dist.get(cur, INF):
                continue
            for dr, dc in DV:
                nxt = (cur[0] + dr, cur[1] + dc)
                if not m.inb(nxt) or nxt in m.walls or nxt in targets:
                    continue
                if d + 1 < dist.get(nxt, INF):
                    dist[nxt] = d + 1
                    heapq.heappush(queue, (d + 1, nxt))
            for entry, cost in portals.get(cur, ()):
                if d + cost < dist.get(entry, INF):
                    dist[entry] = d + cost
                    heapq.heappush(queue, (d + cost, entry))
        self._fields[key] = dist
        return dist

    def hfun(self, st) -> int:
        return self.field(st[1], st[3]).get(st[0], INF)

    # -- macros -------------------------------------------------------------
    def walk_tree(self, st) -> dict:
        """Parent pointers for the shortest FREE walks from the player.

        A step ``p -> p+d`` is free only when ``p+d`` is walkable (no wall, no
        piece, and no target -- stepping on one restarts the level) AND ``p-d``
        holds no piece, because moving away from a piece tows it. That second
        condition is what makes this graph directed."""
        m = self.m
        player, crates, crate2s, targets = st
        pieces = crates | crate2s
        blocked = pieces | m.walls
        parent = {player: None}
        queue = deque([player])
        while queue:
            cur = queue.popleft()
            for d, (dr, dc) in DELTA.items():
                nxt = (cur[0] + dr, cur[1] + dc)
                if (not m.inb(nxt) or nxt in blocked or nxt in targets
                        or nxt in parent):
                    continue
                if (cur[0] - dr, cur[1] - dc) in pieces:
                    continue
                parent[nxt] = (cur, d)
                queue.append(nxt)
        return parent

    @staticmethod
    def _walk_to(parent, cell) -> list:
        out = []
        while parent[cell] is not None:
            cell, d = parent[cell]
            out.append(d)
        out.reverse()
        return out

    def macros(self, st) -> list:
        """Every ``walk + push`` and ``walk + pull`` available right now, plus
        the walk onto the strawberry. To push a piece at ``q`` in direction
        ``d`` the player stands at ``q-d``; to pull it, at ``q+d``."""
        parent = self.walk_tree(st)
        out = []
        # Sorted, not set order: the macro order decides which of two equal-cost
        # plans the search returns, and a plan that depends on set iteration is
        # a plan that can change under the reader's feet.
        for q in sorted(st[1] | st[2]):
            for d, (dr, dc) in DELTA.items():
                for stand in ((q[0] - dr, q[1] - dc), (q[0] + dr, q[1] + dc)):
                    if stand in parent:
                        out.append(Macro(self._walk_to(parent, stand), d, stand))
        for berry in sorted(self.m.berries):
            if berry in parent:
                out.append(Macro(self._walk_to(parent, berry), None, berry))
        return out

    def run(self, st, macro):
        """Apply ``macro`` to ``st``. Returns ``(state, macro_actually_taken,
        won)``, or ``(None, None, False)`` when the macro is not executable --
        a press the engine refuses, or one that would step onto the barrier.

        The win is checked after EVERY press: a walk to a piece can cross the
        strawberry, and a pull can land the player on it, so a macro that wins
        partway through has to be truncated there -- both to keep the plan
        shortest and because walking on past the strawberry un-wins it."""
        m = self.m
        cur = st
        presses = macro.presses
        for i, d in enumerate(presses):
            nxt = m.step(cur, d)
            if nxt == cur or m.restarts(nxt):
                return None, None, False
            cur = nxt
            if m.won(cur):
                if i + 1 >= len(presses):
                    return cur, macro, True
                return cur, Macro(presses[:i + 1], None, cur[0]), True
        return cur, macro, False

    def solve(self):
        """A shortest press sequence to the strawberry as a list of `Macro`, or
        None if the node cap is hit first."""
        m = self.m
        start = m.start
        if m.won(start):
            return []
        # Parent pointers rather than a path per heap entry: these searches run
        # to ~10^6 nodes with ~50-press paths, and copying the path into every
        # child is what makes that a memory problem rather than a time one.
        trail: list = [(-1, None)]
        heap = [(self.weight * self.hfun(start), 0, 0, start, 0)]
        best = {start: 0}
        self.nodes = 0
        # Strictly increasing, so no two heap entries ever tie all the way down
        # to the state -- comparing states would fall through to `frozenset.__lt__`,
        # which is the SUBSET test and orders nothing.
        counter = 0
        while heap:
            _f, g, _c, st, node = heapq.heappop(heap)
            for macro in self.macros(st):
                nxt, taken, won = self.run(st, macro)
                self.nodes += 1
                if nxt is None:
                    continue
                trail.append((node, taken))
                child = len(trail) - 1
                if won:
                    return self._reconstruct(trail, child)
                ng = g + len(taken.presses)
                if best.get(nxt, INF) <= ng:
                    trail.pop()
                    continue
                h = self.hfun(nxt)
                if h >= INF:                   # no breach can still be fed
                    trail.pop()
                    continue
                best[nxt] = ng
                counter += 1
                heapq.heappush(heap, (ng + self.weight * h, ng,
                                      counter, nxt, child))
            if self.nodes >= self.node_cap:
                return None
        return None

    @staticmethod
    def _reconstruct(trail, node) -> list:
        out = []
        while node > 0:
            parent, macro = trail[node]
            out.append(macro)
            node = parent
        out.reverse()
        return out

    # -- optimal-action sets --------------------------------------------------
    def optsets(self, macros) -> list:
        """Per-press optimal SETS for a macro plan.

        An acting press is labelled with itself alone. Each step of a walk is
        labelled with every direction that keeps the player on a shortest route
        to the square the walk ends on -- read off a BFS on the REVERSED walk
        graph, recomputed at each step on the live board, because a Crate2 can
        evaporate mid-walk and open a shortcut. The recorded press is on a
        shortest route by construction; if a reconstruction ever disagreed, the
        press actually taken is used, so no step ships unlabelled."""
        st = self.m.start
        out = []
        for macro in macros:
            walk, act, stand = macro
            for d in walk:
                dist = self._reverse_walk_distances(st, stand)
                here = st[0]
                d0 = dist.get(here)
                alts = [] if d0 is None else [
                    a for a in DIRS
                    if dist.get(self._succ(st, here, a)) == d0 - 1]
                out.append(alts if d in alts else [d])
                st = self.m.step(st, d)
            if act is not None:
                out.append([act])
                st = self.m.step(st, act)
        return out

    def _succ(self, st, cell, d):
        """The square a free walk step in direction ``d`` lands on, or None when
        that step is not a free walk (blocked, lethal, or a tow)."""
        m = self.m
        dr, dc = DELTA[d]
        nxt = (cell[0] + dr, cell[1] + dc)
        pieces = st[1] | st[2]
        if (not m.inb(nxt) or nxt in m.walls or nxt in pieces or nxt in st[3]):
            return None
        if (cell[0] - dr, cell[1] - dc) in pieces:
            return None
        return nxt

    def _reverse_walk_distances(self, st, target) -> dict:
        """BFS distance TO ``target`` over the directed free-walk graph."""
        m = self.m
        pieces = st[1] | st[2]
        targets = st[3]
        dist = {target: 0}
        queue = deque([target])
        while queue:
            cur = queue.popleft()
            for dr, dc in DV:
                prev = (cur[0] - dr, cur[1] - dc)        # prev -> cur via (dr,dc)
                if (not m.inb(prev) or prev in m.walls or prev in pieces
                        or prev in targets or prev in dist):
                    continue
                if (prev[0] - dr, prev[1] - dc) in pieces:
                    continue                             # that step would tow
                if cur in pieces or cur in targets:
                    continue
                dist[prev] = dist[cur] + 1
                queue.append(prev)
        return dist


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class StrawberriesExpert(PSExpert):
    """`PSExpert` with the whole search replaced: planning happens on `Model`,
    not by stepping the interpreter, so `heuristic` -- the hook a stepping A*
    would call -- is never reached.

    What is inherited is everything around the search: the in-memory memo, the
    on-disk start-plan cache with its staleness check, and the snapshot/restore
    discipline `PSExpert.plan` wraps `_search` in."""

    directions = list(DIRS)          # ACTION is bound to no rule in this game
    plan_cache_path = PLAN_CACHE

    def __init__(self, game, node_cap: int = 2_000_000, weight: int = 1):
        super().__init__(game, node_cap=node_cap, weight=weight)
        self._how: dict = {}         # level -> one line for `describe`
        self._level: int | None = None

    def heuristic(self, eng) -> int:                        # pragma: no cover
        raise AssertionError("StrawberriesExpert plans on Model, not on the engine")

    def _search(self, eng):
        model = Model(eng, self.g.obj_name_to_idx)
        if model.start[0] is None:
            return None
        search = Search(model, weight=self.weight, node_cap=self.node_cap)
        started = time.time()
        macros = search.solve()
        if macros is None:
            self._how[self._level] = f"{search.nodes} macros, GAVE UP"
            return None
        presses = [p for macro in macros for p in macro.presses]
        self._how[self._level] = (f"{search.nodes} macros "
                                  f"{time.time() - started:.2f}s")
        return Plan(presses, search.optsets(macros))

    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan`, remembering which level `describe` is reporting on."""
        self._level = level
        return super().plan(eng, level)

    def describe(self, level) -> str:
        """How this level's plan was found, for ``--plans``; "cached" when it
        came off disk and nothing ran."""
        return self._how.get(level, "cached")


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class StrawberriesSolver(PSAStarSolver):
    game_id = "puzzlescript_dreaming_of_strawberries"
    game_name = GAME_NAME
    expert_cls = StrawberriesExpert

    #: Level 1's search settles at ~0.8M macros; the cap is a runaway guard.
    node_cap = 2_000_000
    weight = 1

    #: The longest plan is 87 presses; the rest is room for the exploration
    #: prefix and a re-plan after it. Stays under the adapter's own 200-press
    #: per-level budget (the RESET in the recovery prefix zeroes that counter,
    #: so only the post-reset plan is charged against it).
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup and
        fills the disk cache in one pass."""
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

def selfcheck(trials: int = 200, steps: int = 80, verbose: bool = True) -> int:
    """Audit the claim the solver rests on: `Model` reproduces the interpreter
    EXACTLY, so a search that never touches the engine is not searching a
    different game.

    Random play is the fuzz, and it is the right fuzz here: it pushes pieces
    into walls and into each other (the refused press), tows them out of dead
    ends (the pull rule, which a plan uses on purpose and which nothing else in
    this tree has), shoves Crate2s at the barrier until they evaporate, feeds
    crates into targets, and walks onto the barrier -- the restart, which
    ``engine.step`` does NOT execute (`PuzzleScriptAdapter.perform_action` does),
    so the check reloads the level itself and asserts the model agrees the level
    went back to its start.

    ACTION is included in the presses even though the search drops it: it is a
    real turn, the deletion rules fire on it, and the model has to say so."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, parsed = game._engine, game._game
    presses = list(DIRS) + ["action"]
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        model = Model(eng, parsed.obj_name_to_idx)
        restarts = wins = 0
        for t in range(trials):
            game.set_level(level)
            st = model.start
            rng = random.Random(f"strawberries:selfcheck:{level}:{t}")
            for _ in range(steps):
                d = rng.choice(presses)
                eng.step(d)
                if eng._rule_restart:               # the adapter's job, not step()
                    eng.load_level(parsed.levels[level])
                    eng._rule_restart = False
                    expected = model.start
                    restarts += 1
                else:
                    expected = model.step(st, d)
                    if model.restarts(expected):
                        bad += 1
                        print(f"  L{level}: model says restart, engine did not "
                              f"({d})")
                        break
                actual = model.read(eng)
                if actual != expected:
                    bad += 1
                    print(f"  L{level}: state divergence on {d}")
                    for i, (a, b) in enumerate(zip(expected, actual)):
                        if a != b:
                            print(f"    field {i}: model {a} engine {b}")
                    break
                st = expected
                if model.won(st) != eng.check_win():
                    bad += 1
                    print(f"  L{level}: win divergence "
                          f"(model {model.won(st)}, engine {eng.check_win()})")
                    break
                if model.won(st):
                    wins += 1
                    break
        if verbose:
            print(f"  L{level}: {trials}x{steps} presses, {restarts} restarts, "
                  f"{wins} accidental wins, {'ok' if not bad else 'DIVERGED'}")
    return bad


# ---------------------------------------------------------------------------
# Rendering audit
# ---------------------------------------------------------------------------

#: Every cell composition these levels can show, as the objects stacked in it.
#: Composition, not object: all three art bugs this checks for live in the
#: STACK -- an opaque body standing on a ground tile that the win condition, or
#: the game's central event, is about.
_AUDIT_CASES = {
    "grass": (),
    "wall": ("wall",),
    "barrier": ("target",),
    "strawberry": ("target2",),
    "player": ("player",),
    "crate": ("crate",),
    "crate2": ("crate2",),
    "crate on barrier": ("target", "crate"),          # the one frame before both die
    "crate2 on barrier": ("target", "crate2"),
    "crate on strawberry": ("target2", "crate"),
    "crate2 on strawberry": ("target2", "crate2"),
    "player on barrier": ("target", "player"),        # the frame before a restart
    "player on strawberry": ("target2", "player"),    # THE WINNING FRAME
}


def audit(verbose: bool = True) -> int:
    """Check that every cell composition is pixel-distinct at every cell size
    the levels use.

    This is the regression test for the three art fixes in the .txt. The one
    that matters most: "player on strawberry" is the winning frame, and it
    shipped pixel-identical to "player". Returns the number of colliding pairs.
    """
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
        objs.sort(key=lambda x: x[0])
        return objs

    bad = 0
    for px in sorted(sizes):
        blocks = {k: _render_cell_sprite(stack(("background",) + v), px)
                  for k, v in _AUDIT_CASES.items()}
        clashes = [(a, b) for a, b in itertools.combinations(_AUDIT_CASES, 2)
                   if bool((blocks[a] == blocks[b]).all())]
        bad += len(clashes)
        for a, b in clashes:
            print(f"  cell_px={px}: {a} and {b} render identically")
        if verbose:
            print(f"  cell_px={px}: {len(_AUDIT_CASES)} cell types, "
                  f"{'all distinct' if not clashes else 'COLLISIONS'}")
    return bad


# ---------------------------------------------------------------------------
# Plan report
# ---------------------------------------------------------------------------

def _plan_report() -> int:
    """Print every level's plan length, how it was found and how many of its
    presses have more than one right answer -- the quick "is this game still
    fully solved" check. Every plan is replayed through the real interpreter, so
    this is also the search's end-to-end test."""
    solver = StrawberriesSolver()
    game, expert, solvable = solver._ensure(0)
    total = ties = 0
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            bad += 1
            continue
        won = False
        for press in plan:                              # interpreter-verify it
            eng.step(press)
            if eng._rule_restart:
                break
            if eng.check_win():
                won = True
                break
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        step_ties = sum(len(s) - 1 for s in sets)
        total += len(plan)
        ties += step_ties
        bad += 0 if won else 1
        print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  "
              f"{len(plan):3d} presses  win={won}  "
              f"{expert.describe(level):>22}  {step_ties:3d} tie-presses")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second right answer")
    return bad


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
        failures = _plan_report()
        sys.exit(1 if failures else 0)
    sys.exit(StrawberriesSolver.main())
