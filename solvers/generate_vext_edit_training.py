"""Generate Phase-1 training data for the PuzzleScript game ps:vext_edit
("VEXT EDIT", Jack).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. The MECHANICS live in `solvers/vext_model.py`: a native
model of the game, because the interpreter runs this one at 22-123 steps/s (273
rules, most of them re-derived every tick) and no search here fits in that. The
model is fuzz-verified against the interpreter press-for-press (``--selfcheck``),
and the interpreter still certifies every plan (``--plans``).

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_vext_edit",
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
Every expert step carries an optimal-action set.

THE GAME, IN ONE PARAGRAPH
--------------------------
A sokoban you are allowed to EDIT while you play it. ``Player1`` shoves welded
``Crate1`` polyominoes onto ``Target1`` and stands on ``PlayerTarget1``; press
ACTION and you become a CURSOR that flies through the level's walls, lifts an
item off a palette (a row of pickables, each above a digit that is its stock) and
stamps copies of it into the board, one press and one unit of stock each. Press
ACTION on a player and you are that player again. Later books add a BLUE cursor
that edits the red cursor's palette, twin levels that drive two of everything off
one key, pits that drown crates into bridges and players into corpses, and a
level whose palette stamps the DIGITS themselves, so you mint new palette slots
under whatever you like. `solvers/vext_model.py` documents the mechanics.

HOW THE LEVELS ARE SOLVED
-------------------------
Every level goes down the same ladder, strongest claim first (`solve_level`):

1. **Exhaustive layered BFS** over the native model. When it finds a win the
   depth IS d*, and a backward sweep over the stored layers gives the EXACT
   optimal-action set for every step -- which matters here, because most of a
   plan is a cursor flying to a cell and every interleaving of the two axes is
   equally right.
2. **Macro Dijkstra.** A plan is a sequence of side-effect-free WALKS each ending
   in one real press, so searching only the decision points, at their true press
   cost, loses nothing and is what makes the editing levels reachable at all.
   Because macros cost different numbers of presses, the win is recognised when
   a state is POPPED, never when it is generated -- level 8 comes back at 30 the
   other way and is 28. Ties come from `annotate`: every direction that keeps the
   body on a shortest route to the same cell.
3. **Weighted macro A***, under three stamp-prunes (see `stamp_region`), then a
   **macro beam**. These are genuine wins -- the interpreter replays every one --
   but are not claimed shortest.

RESULTS: 14 of the 32 levels, 406 presses, 7 of them PROVABLY SHORTEST.

    level  0   10 presses   exhaustive BFS (330 states)     shortest
    level  1   26 presses   exhaustive BFS (71 states)      shortest
    level  2   27 presses   exhaustive BFS (10139 states)   shortest
    level  3   34 presses   exhaustive BFS (528 states)     shortest
    level  4   23 presses   macro Dijkstra                  shortest
    level  8   28 presses   macro Dijkstra                  shortest
    level  9   40 presses   macro Dijkstra                  shortest
    level 12   28 presses   macro A* (w=1, prune=tight)
    level 16   43 presses   macro A* (w=1, prune=tight)
    level 17   28 presses   macro beam (width=1500)
    level 18   51 presses   macro A* (w=1, prune=tight)
    level 19   32 presses   macro A* (w=1, prune=near)
    level 21   19 presses   macro A* (w=1, prune=tight)
    level 31   17 presses   macro A* (w=1, prune=tight)

The other eighteen are in `VextEditSolver.skip_levels`. They are not skipped for
being slow -- the whole ladder was run on each of them (~13 minutes apiece) and
came back with nothing. They are the levels where the EDIT is the puzzle rather
than the delivery: "FORM" wants a welded crate scaffold built so that the one
crate blocking the goal can be shoved from a different row; "TRAP" wants the
player sealed inside a ring of crates it has to be pushed into; "IDEA" wants the
palette's own icons shoved between stock digits to turn one crate into three.
The heuristic here scores DELIVERY (how far a piece is from the goal it must
meet), and every one of those is a long stretch of moves that improves it by
nothing at all. That is the wall, and it is a search problem, not a model one:
the model simulates all eighteen exactly.

Level 23 additionally cannot be recorded honestly even if it were solved: at
20x23 the renderer gets 2 pixels per cell, and `--audit` shows Target1 and
PlayerTarget1 (and Wall0 and Wall1) collapsing onto the same block there.

VERIFICATION
------------
* ``--selfcheck`` fuzzes the native model against the interpreter press for
  press, on every level, from guided random play, with coverage counters.
* ``--plans`` replays every cached plan on the real interpreter: 14/14 WIN.
* ``--audit`` renders every level and fails when two mechanically different cell
  compositions produce the same pixels. It found three real collisions, all
  fixed in the .txt (see `AUDIT_NAMES`); 31 of 32 levels are clean, level 23 is
  the exception noted above.
* ``--report`` re-derives every plan from scratch and prints which rung answered.
"""

from __future__ import annotations

import argparse
import heapq
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (Plan, PSAStarSolver, PSExpert,  # noqa: E402
                                     screen_action)
from solvers.vext_model import PLACE_OF, Model                     # noqa: E402

ACTS = ["up", "down", "left", "right", "action"]

#: The adapter cuts a level off at 200 presses, so a longer plan can never be
#: recorded even though it wins.
PRESS_CAP = 190


def _manh(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _field(model, goals, blocked, costly=(), toll=1):
    """Dijkstra distance from a set of goal cells, with a per-cell surcharge.

    The surcharge is what turns a flat heuristic into a gradient: an unfilled
    Pit1 between the player and its goal is not impassable (a crate shoved in
    fills it), it is EXPENSIVE, so the search is rewarded for filling it -- and
    a plain BFS, which treats the pit as one step, actively penalises the walk
    away from the goal that setting the crate up requires."""
    dist = {}
    heap = [(0, g) for g in goals if g not in blocked]
    heapq.heapify(heap)
    while heap:
        d, cell = heapq.heappop(heap)
        if cell in dist:
            continue
        dist[cell] = d
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nb = (cell[0] + dr, cell[1] + dc)
            if not model.inb(nb) or nb in blocked or nb in dist:
                continue
            heapq.heappush(heap, (d + 1 + (toll if nb in costly else 0), nb))
    return dist


class Ctx:
    """The distance fields the heuristic reads, cached per board shape.

    The walls come off the level's START board -- they are static in every level
    of this game that has a wall at all, and a heuristic that is a little stale
    is a heuristic, not a bug. What is NOT taken from the start is the surcharge
    set (pits still open, crates still in the way): those are the things a plan
    is FOR, so their field is re-derived whenever they change, and cached on
    them so that costs nothing on the states where they have not.

    It also serves the other half of the gradient: while the board is still
    EMPTY of the piece a win condition wants, the distance from the cursor to the
    palette slot that sells it is the only signal the editing half of a level
    has."""

    MISSING = 8          # flat toll for "you still have to stamp one of these"
    TOLL = 6             # surcharge for a cell the plan has to clear first

    def __init__(self, model, st0):
        self.model = model
        self.nums = set(st0.num0) | set(st0.num2)
        self.walk_blocked = model.black | st0.s["wall1"] | self.nums
        self.cursor_blocked = model.black | st0.s["wall0"]
        self.goals = (st0.s["target1"] | st0.s["playertarget1"]
                      | st0.s["target0"])
        self._cache = {}
        self.slot_of = {}
        for slot in list(st0.num0) + list(st0.num2):
            src = (slot[0] - 1, slot[1])
            if not model.inb(src):
                continue
            for mk in model.slot_mark_types(st0, src):
                obj = PLACE_OF.get(mk)
                if obj:
                    self.slot_of.setdefault(obj, []).append(src)
        self.slot_field = {obj: _field(model, cells, self.cursor_blocked)
                           for obj, cells in self.slot_of.items()}

    def field(self, goal, costly):
        key = (goal, costly)
        f = self._cache.get(key)
        if f is None:
            if len(self._cache) > 4000:
                self._cache.clear()
            f = _field(self.model, [goal], self.walk_blocked, costly, self.TOLL)
            self._cache[key] = f
        return f

    def to_goal(self, pieces, goal, costly=frozenset()):
        if not pieces:
            return self.MISSING
        f = self.field(goal, costly)
        best = min((f[p] for p in pieces if p in f), default=None)
        return best if best is not None else min(_manh(goal, p)
                                                 for p in pieces) + 6

    def to_slot(self, obj, bodies):
        """How far the controlled body is from the palette slot selling ``obj``."""
        f = self.slot_field.get(obj)
        if not f or not bodies:
            return self.MISSING
        best = min((f[b] for b in bodies if b in f), default=None)
        return self.MISSING if best is None else best


def _bodies(st):
    return (st.s["active"] | st.s["activefell"] | st.s["activecurse0"]
            | st.s["dead0a"] | st.s["cursor2a"] | st.s["cursor2b"])


def heuristic(model, st, ctx):
    """Informed, deliberately NOT admissible: it is the beam's score and A*'s
    steering, and the exact answers come from the two shortest-proving stages
    instead."""
    h = 0
    bodies = _bodies(st)
    crates = st.s["crate1"] | st.s["fell1"]
    open_pits = frozenset(st.s["pit1"] - st.s["fell1"])
    for t in st.s["target1"]:
        if t in crates:
            continue
        h += 2 + (ctx.to_goal(crates, t) if crates
                  else 4 + ctx.to_slot("crate1", bodies))
    pm = set(st.s["player1"])
    for cell in (st.curse0a() | st.s["cursor0b"]
                 | st.s["cursor2a"] | st.s["cursor2b"]):
        if "player1mark" in model.slot_marks_for(st, cell):
            pm.add(cell)
    blockers = open_pits | frozenset(st.s["crate1"])
    for pt in st.s["playertarget1"]:
        if pt in pm:
            continue
        h += 2 + (ctx.to_goal(pm, pt, blockers) if pm
                  else 4 + ctx.to_slot("player1", bodies))
    for t0 in st.s["target0"]:
        if t0 in st.s["crate0"]:
            continue
        h += 2 + (ctx.to_goal(st.s["crate0"], t0) if st.s["crate0"]
                  else 4 + ctx.to_slot("crate0", bodies))
    if st.cursors0() & (st.s["target1"] | st.s["playertarget1"]):
        h += 1
    return h


# ---------------------------------------------------------------------------
# stage 1 -- exhaustive layered BFS, which is also the exact-tie oracle
# ---------------------------------------------------------------------------

def bfs_exact(model, st0, cap=500_000, tlimit=60.0):
    """Layered BFS to the first winning depth, then a backward sweep for the
    per-step optimal SETS.

    The forward pass finishes the layer the first win appears in before it
    stops, so ``keep[win_depth]`` really is EVERY winning state at that depth;
    the sweep then walks the layers down, keeping a state only when one of its
    presses lands in the layer above's keep-set. What comes out is the exact set
    of shortest presses at every state on the path -- which matters a lot here,
    because most of a plan is a cursor flying to a cell and every interleaving of
    the two axes is equally right.

    Returns ``(Plan, n_states)`` or ``(None, n_states)``."""
    t0 = time.time()
    k0 = model.pack(st0)
    if model.won(st0):
        return Plan([], []), 1
    dist = {k0: 0}
    layers = [[k0]]
    depth = 0
    win_depth = None
    while layers[-1] and win_depth is None:
        if len(dist) > cap or time.time() - t0 > tlimit or depth >= PRESS_CAP:
            return None, len(dist)
        nxt = []
        for k in layers[-1]:
            st = model.unpack(k)
            for a in ACTS:
                ns = model.step(st, a)
                if model.ambiguous:
                    continue
                nk = model.pack(ns)
                if nk in dist:
                    continue
                dist[nk] = depth + 1
                nxt.append(nk)
                if model.won(ns):
                    win_depth = depth + 1
        layers.append(nxt)
        depth += 1
    if win_depth is None:
        return None, len(dist)

    keep = {k for k in layers[win_depth] if model.won(model.unpack(k))}
    ups = [keep]
    for d in range(win_depth - 1, -1, -1):
        above = ups[-1]
        cur = set()
        for k in layers[d]:
            st = model.unpack(k)
            for a in ACTS:
                ns = model.step(st, a)
                if model.ambiguous:
                    continue
                if model.pack(ns) in above:
                    cur.add(k)
                    break
        ups.append(cur)
    ups.reverse()                     # ups[d] == the keep-set at depth d

    presses, optsets = [], []
    st = st0
    for d in range(win_depth):
        above = ups[d + 1]
        good, succ = [], {}
        for a in ACTS:
            ns = model.step(st, a)
            if model.ambiguous:
                continue
            if model.pack(ns) in above:
                good.append(a)
                succ[a] = ns
        if not good:                  # cannot happen; a keep-set state has a move
            return None, len(dist)
        presses.append(good[0])
        optsets.append(good)
        st = succ[good[0]]
    return Plan(presses, optsets), len(dist)


# ---------------------------------------------------------------------------
# macros -- walk the controlled body somewhere, then press one key
# ---------------------------------------------------------------------------
#
# A primitive search cannot reach the editor levels: a plan there is "fly the
# cursor eight squares to the palette, press ACTION, fly it eleven squares into
# the room, press ACTION, ..." and the presses in between change nothing a
# heuristic can see. The fix is the standard one for this shape -- make the walk
# free and search over what happens at the END of it.
#
# Walking really is free here, under conditions this module CHECKS rather than
# assumes (`control_mode`): one body under control, and nothing on its layer it
# could shove or drown in. When they do not hold -- the twin levels drive two
# cursors off one key, ps:vext_edit's "LOCK" gives the cursor boxes to push and
# pits to fall into -- the search drops back to primitive presses for that level.
# Either way the plan that comes out is a flat press list the interpreter
# replays, so a broken assumption is caught, not shipped.

def control_mode(model, st):
    """``(kind, cell, free_cells)`` when exactly one body is under control and
    its walk is a pure change of position, else None."""
    curse0 = sorted(st.s["activecurse0"])
    dead = sorted(st.s["dead0a"])
    c2 = sorted(st.s["cursor2a"] | st.s["cursor2b"])
    act = sorted(st.s["active"])
    fell = sorted(st.s["activefell"])
    if dead or len(curse0) + len(c2) + len(act) + len(fell) != 1:
        return None
    if c2:
        cell = c2[0]
        return "curse2", cell, _reach(model, cell,
                                      _blocked_of(model, st, "curse2"))
    if curse0:
        cell = curse0[0]
        if st.s["crate0"] or st.s["pit0"]:
            return None                     # boxes to shove, pits to drown in
        if len(st.s["cursor0a"] | st.s["cursor0b"]) != 1:
            return None                     # a second cursor is pushable scenery
        return "curse0", cell, _reach(model, cell,
                                      _blocked_of(model, st, "curse0"))
    if act:
        cell = act[0]
        return "player", cell, _reach(model, cell,
                                      _blocked_of(model, st, "player"))
    return None


def _reach(model, start, blocked):
    """Shortest walk from ``start`` to every cell it can reach: cell -> presses."""
    out = {start: []}
    queue = [start]
    while queue:
        nxt = []
        for cell in queue:
            for d in ("up", "down", "left", "right"):
                nb = (cell[0] + (d == "down") - (d == "up"),
                      cell[1] + (d == "right") - (d == "left"))
                if not model.inb(nb) or nb in blocked or nb in out:
                    continue
                out[nb] = out[cell] + [d]
                nxt.append(nb)
        queue = nxt
    return out


def _teleport(model, st, kind, src, dst):
    if src == dst:
        return st
    o = st.clone()
    if kind == "curse2":
        for k in ("cursor2a", "cursor2b"):
            if src in o.s[k]:
                o.s[k].discard(src)
                o.s[k].add(dst)
    elif kind == "curse0":
        for k in ("cursor0a", "cursor0b", "activecurse0"):
            if src in o.s[k]:
                o.s[k].discard(src)
                o.s[k].add(dst)
    else:
        for k in ("player1", "active"):
            if src in o.s[k]:
                o.s[k].discard(src)
                o.s[k].add(dst)
    if src in o.holding:
        o.holding.discard(src)
        o.holding.add(dst)
    return o


def stamp_region(model, st, near_slots=True):
    """Where a RED cursor's stamp can possibly matter.

    Left unrestricted, "press ACTION here" is a legal macro at every empty cell
    on the board, and the search drowns in boards that differ only by a crate
    dropped in the margin. A stamp is worth trying at a cell that is either

    * inside a connected region of the PLAYED level that holds something the win
      is about -- a player, a target, a crate that is not a palette entry (the
      cursor flies through Wall1, the player does not, so a crate stamped in
      another region can never be pushed anywhere that matters), or
    * within a step or two of a stock digit's slot, because the palette is part
      of the BOARD: stamping onto a slot MINTS A NEW SLOT out of whatever you put
      down (ps:vext_edit's "GUIS" turns its one crate into an infinite supply
      exactly that way), and standing a player next to the palette lets you SHOVE
      the icons between slots, which is how "IDEA" turns one crate into three.
      The region test alone would hide both.

    This is a PRUNE, not a proof: it can in principle hide a solution that needs
    scenery parked somewhere pointless. It is what makes the editing levels
    searchable at all, and every plan it yields is still replayed on the real
    interpreter.
    """
    nums = set(st.num0) | set(st.num2)
    blocked = model.black | st.s["wall1"] | nums
    slots = {c for c in model.cells if (c[0] + 1, c[1]) in nums}
    seeds = ((st.s["player1"] | st.s["crate1"] | st.s["fell1"] | st.s["ded1"]
              | st.s["target1"] | st.s["playertarget1"] | st.s["target0"])
             - slots)
    region = {c for c in seeds if c not in blocked}
    queue = list(region)
    while queue:
        nxt = []
        for cell in queue:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (cell[0] + dr, cell[1] + dc)
                if model.inb(nb) and nb not in blocked and nb not in region:
                    region.add(nb)
                    nxt.append(nb)
        queue = nxt
    if not near_slots:
        return region | slots
    near = set()
    for r, c in slots:
        for dr in range(-2, 3):
            for dc in range(-2 + abs(dr), 3 - abs(dr)):
                cell = (r + dr, c + dc)
                if model.inb(cell):
                    near.add(cell)
    return region | slots | near


def macro_succ(model, st, prune=None):
    """``[(presses, state), ...]``: every walk-then-one-press this state allows."""
    mode = control_mode(model, st)
    if mode is None:
        out = []
        for a in ACTS:
            ns = model.step(st, a)
            if model.ambiguous or model.pack(ns) == model.pack(st):
                continue
            out.append(([a], ns))
        return out
    kind, cell, reach = mode
    finals = (["action"] if kind != "player"
              else ["action", "up", "down", "left", "right"])
    allowed = (stamp_region(model, st, prune == "near")
               if (kind == "curse0" and prune) else None)
    out = []
    for dst, path in reach.items():
        if (allowed is not None and dst not in allowed
                and dst not in st.s["player1"] and dst not in st.s["ded1"]
                and dst != cell):
            continue
        walked = _teleport(model, st, kind, cell, dst)
        if path and model.won(walked):
            out.append((path, walked))
            continue
        for a in finals:
            if a != "action" and (dst[0] + (a == "down") - (a == "up"),
                                  dst[1] + (a == "right") - (a == "left")) in reach:
                continue                    # a plain step: another macro covers it
            ns = model.step(walked, a)
            if model.ambiguous or model.pack(ns) == model.pack(walked):
                continue
            out.append((path + [a], ns))
    return out


# ---------------------------------------------------------------------------
# stage 2 / 3 -- weighted A* and a beam, for the levels stage 1 cannot close
# ---------------------------------------------------------------------------

def macro_astar(model, st0, ctx, weight=2, cap=200_000, tlimit=90.0,
                prune=None):
    """Weighted A* over `macro_succ`, costed in PRESSES (not in macros), so a
    plan it returns is comparable with a primitive one."""
    t0 = time.time()
    if model.won(st0):
        return []
    k0 = model.pack(st0)
    pq = [(weight * heuristic(model, st0, ctx), 0, 0, k0, [])]
    best = {k0: 0}
    counter = 0
    while pq:
        if len(best) > cap or time.time() - t0 > tlimit:
            return None
        _f, g, _c, k, path = heapq.heappop(pq)
        if g > best.get(k, 1 << 30):
            continue
        st = model.unpack(k)
        # The win is recognised on POP, not on generation. A macro is a walk
        # plus one press, so macros cost different numbers of presses: a node
        # popped at g=40 can win with an 11-press macro (51) while a node popped
        # at g=45 wins with a 3-press one (48), and returning the first win the
        # search GENERATED hands back the longer of the two. ps:vext_edit level
        # 8 is the case -- 30 presses generated where 28 exists. Because
        # `heuristic` is 0 at a win, popping one under Dijkstra (weight 0) is a
        # proof that nothing shorter exists.
        if model.won(st):
            return path
        for presses, ns in macro_succ(model, st, prune):
            g2 = g + len(presses)
            if g2 > PRESS_CAP:
                continue
            nk = model.pack(ns)
            if g2 >= best.get(nk, 1 << 30):
                continue
            best[nk] = g2
            npath = path + presses
            counter += 1
            heapq.heappush(pq, (g2 + weight * heuristic(model, ns, ctx), g2,
                                counter, nk, npath))
    return None


def macro_beam(model, st0, ctx, width=2000, depth=24, tlimit=180.0,
               prune=None):
    """Beam over macros. The depth is in MACROS -- a dozen of them is a whole
    level here -- which is what makes a wide beam affordable."""
    t0 = time.time()
    if model.won(st0):
        return []
    seen = {model.pack(st0)}
    frontier = [(st0, [])]
    for _ in range(depth):
        if time.time() - t0 > tlimit:
            return None
        scored = []
        for st, path in frontier:
            for presses, ns in macro_succ(model, st, prune):
                if len(path) + len(presses) > PRESS_CAP:
                    continue
                nk = model.pack(ns)
                if nk in seen:
                    continue
                seen.add(nk)
                npath = path + presses
                if model.won(ns):
                    return npath
                scored.append((heuristic(model, ns, ctx), len(npath), ns, npath))
        if not scored:
            return None
        scored.sort(key=lambda x: (x[0], x[1]))
        frontier = [(s, p) for _h, _g, s, p in scored[:width]]
    return None


def shorten(model, st0, presses):
    """Delete engine-verified blocks, longest-first. A beam plan wins but
    wanders, and every press it wastes is a press of training data that teaches
    a detour."""
    presses = list(presses)
    changed = True
    while changed:
        changed = False
        n = len(presses)
        for size in range(min(n, 24), 0, -1):
            for i in range(0, n - size + 1):
                cand = presses[:i] + presses[i + size:]
                st = st0
                ok = False
                for a in cand:
                    st = model.step(st, a)
                    if model.ambiguous:
                        ok = False
                        break
                    if model.won(st):
                        ok = True
                        break
                if ok:
                    presses = cand
                    changed = True
                    break
            if changed:
                break
    return presses


def annotate(model, st0, presses):
    """Per-step optimal SETS for a plan that came out of the macro layer.

    A macro is a walk followed by one press, and the walk is where a plan spends
    most of its length. Every interleaving of the two axes that keeps the body on
    a shortest route to the same cell leaves an IDENTICAL board -- nothing fires
    on a bare move here -- so labelling one arbitrary interleaving as "the"
    answer would train a made-up preference. Each walk step is therefore labelled
    with every direction that still reaches the same destination in the same
    number of presses; the press the walk was for is labelled with itself.

    Segments are found by WATCHING the board (a press that moved only the
    controlled body and its attachments was a walk), not by trusting a macro
    boundary the flattened plan no longer carries."""
    states = [st0]
    st = st0
    for a in presses:
        st = model.step(st, a)
        states.append(st)

    def body_of(state):
        b = sorted(_bodies(state))
        return b[0] if len(b) == 1 else None

    def walked(i):
        """True when press i only moved a single controlled body."""
        if presses[i] == "action":
            return False
        a, b = states[i], states[i + 1]
        src, dst = body_of(a), body_of(b)
        if src is None or dst is None or src == dst:
            return False
        mode = control_mode(model, a)
        if mode is None or dst not in mode[2]:
            return False
        return model.pack(_teleport(model, a, mode[0], src, dst)) == model.pack(b)

    flags = [walked(i) for i in range(len(presses))]
    out = []
    i = 0
    while i < len(presses):
        if not flags[i]:
            out.append([presses[i]])
            i += 1
            continue
        j = i
        while j < len(presses) and flags[j]:
            j += 1
        dest = body_of(states[j])
        mode = control_mode(model, states[i])
        back = _reach(model, dest, _blocked_of(model, states[i], mode[0]))
        for k in range(i, j):
            here = body_of(states[k])
            need = len(back.get(here, ()))
            opts = []
            for d in ("up", "down", "left", "right"):
                nb = (here[0] + (d == "down") - (d == "up"),
                      here[1] + (d == "right") - (d == "left"))
                if nb in back and len(back[nb]) == need - 1:
                    opts.append(d)
            out.append(opts or [presses[k]])
        i = j
    return out


def _blocked_of(model, st, kind):
    if kind == "curse2":
        return model.wall2 | model.black
    if kind == "curse0":
        return model.black | st.s["wall0"]
    body = sorted(st.s["active"])
    return (model.black | st.s["wall1"] | st.s["crate1"]
            | set(st.num0) | set(st.num2) | st.s["ded1"]
            | (st.s["player1"] - set(body))
            | (st.s["pit1"] - st.s["fell1"]))


def verify(model, st0, presses):
    """Replay a press list on the model and require it to reach a win, with no
    ambiguous press on the way. The macro layer teleports the body along its
    walk, so this is the check that the teleport was legitimate."""
    st = st0
    for i, a in enumerate(presses):
        st = model.step(st, a)
        if model.ambiguous:
            return None
        if model.won(st):
            return presses[:i + 1]
    return None


def solve_level(model, st0, budget):
    """The ladder, strongest-claim first. Returns ``(Plan, how)``.

    1. exhaustive primitive BFS -- shortest, with EXACT per-step tie sets;
    2. macro Dijkstra -- shortest in presses too (a plan is a sequence of
       side-effect-free walks each ending in one real press, so searching the
       decision points at unit press cost loses nothing), with walk ties;
    3. weighted macro A*, then a macro beam -- a genuine win, not claimed
       shortest.
    """
    ctx = Ctx(model, st0)
    plan, n = bfs_exact(model, st0, cap=budget["bfs_cap"],
                        tlimit=budget["bfs_time"])
    if plan is not None:
        return plan, f"exhaustive BFS ({n} states) -- shortest"
    attempts = [("macro Dijkstra -- shortest",
                 lambda: macro_astar(model, st0, ctx, weight=0,
                                     cap=budget["dijkstra_cap"],
                                     tlimit=budget["dijkstra_time"]))]
    # Three stamp-prunes, tightest first. Which one a level needs is not
    # guessable: "IDEA" has to stand a player NEXT TO the palette to shove its
    # icons between slots, and the tight prune hides that; "WALL" is only
    # searchable at all because the tight prune exists.
    for prune in ("tight", "near", None):
        attempts += [(f"macro A* (w={w}, prune={prune})",
                      lambda w=w, prune=prune:
                      macro_astar(model, st0, ctx, weight=w,
                                  cap=budget["astar_cap"],
                                  tlimit=budget["astar_time"], prune=prune))
                     for w in budget["weights"]]
    attempts += [(f"macro beam (width={width})",
                  lambda width=width: macro_beam(model, st0, ctx, width=width,
                                                 tlimit=budget["beam_time"],
                                                 prune="near"))
                 for width in budget["widths"]]
    for how, run in attempts:
        presses = run()
        if presses is None:
            continue
        presses = verify(model, st0, presses)
        if presses is None:                    # a teleport that was not legal
            continue
        if not how.endswith("shortest"):
            presses = shorten(model, st0, presses)
        return Plan(presses, annotate(model, st0, presses)), how
    return None, "unsolved"


# ---------------------------------------------------------------------------
# expert + generator
# ---------------------------------------------------------------------------

#: Per-level search budget. The default is what most levels want; the entries
#: name the levels whose space is big enough that stage 1 is hopeless and paying
#: for it twice (once at discovery, once at recording) is wasted minutes.
DEFAULT_BUDGET = {
    "bfs_cap": 400_000, "bfs_time": 20.0,
    "dijkstra_cap": 900_000, "dijkstra_time": 120.0,
    "astar_cap": 1_200_000, "astar_time": 60.0, "weights": (1, 8, 20),
    "widths": (1500,), "beam_time": 150.0,
}


class VextExpert(PSExpert):
    """Plans on `solvers.vext_model.Model`, never on the interpreter.

    `PSExpert` keeps the memo, the disk cache with its staleness check and the
    snapshot/restore discipline; all this adds is `_search`, which builds the
    native model off the engine grid and runs the ladder on it."""

    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "vext_edit_plans.json")
    scope_by_level = True

    def __init__(self, *a, **kw):
        self._level = None
        self.how = {}
        super().__init__(*a, **kw)

    def setup(self) -> None:
        pass

    def heuristic(self, eng) -> int:                       # pragma: no cover
        raise AssertionError("ps:vext_edit never searches the interpreter")

    def plan(self, eng, level=None):
        self._level = level
        return super().plan(eng, level)

    def _search(self, eng):
        model, st = Model.from_engine(self.g, eng)
        model.setup_key(st)
        budget = dict(DEFAULT_BUDGET)
        budget.update(BUDGETS.get(self._level, {}))
        plan, how = solve_level(model, st, budget)
        self.how[self._level] = how
        if plan is not None and len(plan) > PRESS_CAP:
            self.how[self._level] = f"{how} -- too long ({len(plan)}), dropped"
            return None
        return plan


#: Per-level overrides for `DEFAULT_BUDGET`. Empty: measurement said the same
#: budget suits every level, and the levels that need more than it are the ones
#: no budget reaches (see `VextEditSolver.skip_levels`).
BUDGETS: dict[int, dict] = {}


class VextEditSolver(PSAStarSolver):
    """`BaseSolver` for ps:vext_edit."""

    game_id = "puzzlescript_vext_edit"
    game_name = "VEXT_EDIT"
    game_module_id = "ps:vext_edit"
    expert_cls = VextExpert
    #: The levels the ladder cannot win. Skipped up front rather than
    #: re-attempted: the ladder spends ~13 minutes proving each of them
    #: unreachable, and it would do it again at every cold start. See the
    #: RESULTS table in the module docstring for what each one is.
    skip_levels: frozenset = frozenset(
        {5, 6, 7, 10, 11, 13, 14, 15, 20, 22, 23, 24, 25, 26, 27, 28, 29, 30})
    max_steps = 260
    epsilon = 0.0
    supports_recovery = True
    recovery_mode = "reset"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _adapter(seed=0):
    from adapters.puzzlescript_adapter import PuzzleScriptAdapter
    return PuzzleScriptAdapter("VEXT_EDIT", seed=seed)


def cmd_report(args):
    """Solve every level from scratch and print the ladder's verdict."""
    game = _adapter()
    expert = VextExpert(game)
    if args.fresh:
        expert._disk = {}                 # ignore the cache and re-derive
    total = 0
    for lvl in range(game.n_levels):
        if args.levels and lvl not in args.levels:
            continue
        game.set_level(lvl)
        t = time.time()
        plan = expert.plan(game._engine, lvl)
        how = expert.how.get(lvl, "cached (--fresh to re-derive)")
        if plan is None:
            print(f"level {lvl:2d}: UNSOLVED   {how}  ({time.time() - t:.1f}s)",
                  flush=True)
        else:
            total += len(plan)
            print(f"level {lvl:2d}: {len(plan):3d} presses   {how}"
                  f"  ({time.time() - t:.1f}s)", flush=True)
    print(f"total {total} presses")


def cmd_plans(args):
    """Replay each cached plan on the REAL interpreter and require a WIN."""
    from arcengine import ActionInput
    from solvers.common.ps_astar import DIR_TO_ACTION
    game = _adapter()
    expert = VextExpert(game)
    ok = bad = 0
    for lvl in range(game.n_levels):
        if args.levels and lvl not in args.levels:
            continue
        game.set_level(lvl)
        plan = expert.plan(game._engine, lvl)
        if plan is None:
            print(f"level {lvl:2d}: no plan")
            continue
        game.set_level(lvl)
        eng = game._engine
        for d in plan:
            eng.step(d)
        won = eng.check_win()
        ok, bad = (ok + won, bad + (not won))
        print(f"level {lvl:2d}: {len(plan):3d} presses -> "
              f"{'WIN' if won else 'NOT A WIN'}")
    print(f"{ok} verified, {bad} failed")


def cmd_selfcheck(args):
    """Fuzz the native model against the interpreter, press for press."""
    import random
    from collections import Counter
    from solvers.vext_model import CELLSETS
    game = _adapter()
    gm, eng = game._game, game._engine
    cov, failures = Counter(), 0

    def read():
        return Model.from_engine(gm, eng)[1]

    def same(a, b):
        return (all(a.s[n] == b.s[n] for n in CELLSETS)
                and a.holding == b.holding and a.picked == b.picked
                and a.num0 == b.num0 and a.num2 == b.num2
                and a.w1 == b.w1 and a.wf == b.wf)

    levels = args.levels or list(range(game.n_levels))
    for lvl in levels:
        for ep in range(args.fuzz_episodes):
            rng = random.Random(args.fuzz_seed * 7919 + lvl * 101 + ep)
            game.set_level(lvl)
            model, st = Model.from_engine(gm, eng)
            goal = None
            for t in range(args.fuzz_steps):
                bodies = sorted(st.s["cursor0a"] | st.s["cursor0b"]
                                | st.s["dead0a"] | st.s["cursor2a"]
                                | st.s["cursor2b"] | st.s["active"])
                if goal is None or rng.random() < 0.15:
                    goal = (rng.randrange(model.h), rng.randrange(model.w))
                if rng.random() < 0.35 or not bodies:
                    act = "action"
                else:
                    r, c = bodies[0]
                    opts = (["up"] * (goal[0] < r) + ["down"] * (goal[0] > r)
                            + ["left"] * (goal[1] < c) + ["right"] * (goal[1] > c))
                    act = rng.choice(opts) if opts else rng.choice(ACTS)
                prev = st
                eng.step(act)
                st = model.step(st, act)
                if model.ambiguous:
                    cov["ambiguous"] += 1
                    break
                for n in CELLSETS:
                    if prev.s[n] != st.s[n]:
                        cov[n] += 1
                for n in ("holding", "picked", "num0", "num2", "w1", "wf"):
                    if getattr(prev, n) != getattr(st, n):
                        cov[n] += 1
                ref = read()
                if not same(st, ref) or model.won(st) != eng.check_win():
                    print(f"MISMATCH level {lvl} ep {ep} step {t} ({act})")
                    failures += 1
                    break
        print(f"level {lvl:2d}: fuzzed", flush=True)
    print("coverage:", dict(sorted(cov.items(), key=lambda kv: -kv[1])))
    print("FAILURES:", failures)
    return failures


#: Object classes a policy has to tell apart. `--audit` renders every level and
#: fails if two DIFFERENT compositions of these come out as the same block of
#: pixels -- the check that found ps:vext_edit's three identical brown walls,
#: its digits that all decimated to the same three pixels, and its pit interiors
#: painted in the frame's own black.
AUDIT_NAMES = (
    "player1", "crate1", "wall1", "target1", "playertarget1", "pit1", "fell1",
    "ded1", "cursor0a", "cursor0b", "dead0a", "wall0", "crate0", "pit0",
    "fell0", "target0", "cursor2a", "cursor2b", "wall2", "black",
    "one0", "two0", "three0", "four0", "five0", "six0", "seven0", "inf0",
    "one2", "two2", "three2",
    "player1mark", "crate1mark", "target1mark", "playertarget1mark",
    "wall1mark", "pit1mark", "fell1mark", "ded1mark", "target0mark",
    "crate0mark", "cursor0mark", "one0mark", "two0mark", "three0mark",
    "inf0mark", "wall0mark", "one2mark", "two2mark", "three2mark", "holding",
)


def cmd_audit(args):
    """Whole-frame render audit.

    Cropping one cell out of the frame by arithmetic is wrong here -- the
    renderer picks a cell size, then UPSCALES the whole play area to fill 64x64
    and letterboxes it -- so the cell map is rebuilt by pushing cell IDS through
    the identical geometry and grouping the real frame's pixels by it."""
    import numpy as np
    from adapters import puzzlescript_adapter as PA
    game = _adapter()
    gm, eng = game._game, game._engine
    idx = {n: gm.obj_name_to_idx[n] for n in AUDIT_NAMES
           if n in gm.obj_name_to_idx}

    def cellmap():
        h, w = eng.height, eng.width
        px = max(1, min(64 // h, 64 // w))
        rh, rw = h * px, w * px
        ids = np.repeat(np.repeat(np.arange(h * w).reshape(h, w), px, 0), px, 1)
        if rh < 64 and rw < 64:
            scale = min(64 / rh, 64 / rw)
            nw = min(64, max(1, int(rw * scale)))
            nh = min(64, max(1, int(rh * scale)))
            if (nw, nh) != (rw, rh):
                ids = ids[np.ix_((np.arange(nh) * rh // nh).clip(0, rh - 1),
                                 (np.arange(nw) * rw // nw).clip(0, rw - 1))]
                rh, rw = nh, nw
        out = np.full((64, 64), -1)
        out[(64 - rh) // 2:(64 - rh) // 2 + rh,
            (64 - rw) // 2:(64 - rw) // 2 + rw] = ids
        return out, w, px

    bad = 0
    for lvl in range(game.n_levels):
        if args.levels and lvl not in args.levels:
            continue
        game.set_level(lvl)
        frame = np.asarray(PA._render_frame(eng, gm))
        cm, w, px = cellmap()
        seen, clash = {}, set()
        for cid in np.unique(cm):
            if cid < 0:
                continue
            r, c = divmod(int(cid), w)
            comp = tuple(sorted(n for n, i in idx.items() if i in eng.grid[r][c]))
            blk = frame[cm == cid].tobytes()
            if blk in seen and seen[blk] != comp:
                clash.add((seen[blk], comp))
            seen.setdefault(blk, comp)
        print(f"level {lvl:2d}: {eng.height}x{eng.width} cell_px={px} "
              + ("OK" if not clash else f"{len(clash)} CLASHES"))
        for a, b in sorted(clash):
            print("      ", a or "(empty)", " == ", b or "(empty)")
        bad += bool(clash)
    print(f"levels with clashes: {bad}")
    return bad


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true",
                    help="solve every level and print which rung answered")
    ap.add_argument("--fresh", action="store_true",
                    help="with --report, ignore data/vext_edit_plans.json")
    ap.add_argument("--plans", action="store_true",
                    help="replay every cached plan on the real interpreter")
    ap.add_argument("--selfcheck", action="store_true",
                    help="fuzz the native model against the interpreter")
    ap.add_argument("--audit", action="store_true",
                    help="whole-frame render audit for sprite collisions")
    ap.add_argument("--levels", type=lambda s: [int(x) for x in s.split(",")],
                    default=None, help="restrict to these levels")
    # Deliberately NOT --episodes/--seed/--out: those belong to `BaseSolver`'s
    # own parser and must reach it untouched, or the generation run silently
    # falls back to its 2000-episode default.
    ap.add_argument("--fuzz-episodes", type=int, default=20,
                    help="with --selfcheck, fuzz episodes per level")
    ap.add_argument("--fuzz-steps", type=int, default=60,
                    help="with --selfcheck, presses per fuzz episode")
    ap.add_argument("--fuzz-seed", type=int, default=0)
    args, rest = ap.parse_known_args(argv)
    if args.report:
        return cmd_report(args)
    if args.plans:
        return cmd_plans(args)
    if args.selfcheck:
        return cmd_selfcheck(args)
    if args.audit:
        return cmd_audit(args)
    return VextEditSolver.main(rest)


if __name__ == "__main__":
    raise SystemExit(main())
