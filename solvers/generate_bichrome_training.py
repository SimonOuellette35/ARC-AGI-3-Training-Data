"""Generate Phase-1 training data for the PuzzleScript game ps:bichrome
("Bichrome" by Nils Jung).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other search-the-interpreter ps: generators. This file is the game-specific part:
a native model of the mechanic, the two searches that plan on it, and the engine
replay that certifies every plan before it is used.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_bichrome",
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
augmented view, so replaying the recorded actions reproduces the recorded frames
exactly.

The game
--------
You steer TWO characters, an orange one and a blue one, but only ever one at a
time: ACTION5 hands control to the other (both toggle at once, so exactly one is
"active"). The four arrow keys move the active character. Every cell of every
board is a wall, and a wall's colour decides who it is solid to:

  * you WALK, freely, over walls of the OTHER colour -- the orange character
    lives on the blue walls, the blue character on the orange ones;
  * you PUSH walls of your OWN colour, one at a time, no chains;
  * black walls stop everyone.

A push into another wall of your own colour, into a black wall, or into the
OTHER character's target is refused and the whole turn is cancelled. A push into
a plain wall of the other colour SWAPS the two -- which is the entire mechanic:
tunnelling through your own colour by trading places with the other's. Pushing a
wall the other character is standing on is refused as well, and the two
characters can never share a cell.

Win: orange standing on the blue target AND blue standing on the orange target.
Both targets are themselves pushable walls of their own colour, so a character
can (and on the hardest levels must) CARRY the other one's goal across the board
to a place they can reach.

Twelve levels; ten of them are solved here. Levels 4 ("Roundabout") and 9 ("Two
Worlds") are the two built around carrying a target -- each is split into an
all-blue half and an all-orange half, so finishing means fetching the other
character's goal back across the barrier -- and neither search below reaches
them. They are skipped; `BichromeSolver.skip_levels` records what was measured.

Two things had to be fixed before any of this could be recorded.

**The engine could not see the win** (`adapters/puzzlescript_adapter.py`). Its
"all X on Y" check has a guard that refuses a vacuous win when X is a player
class, so a deleted player never counts as solved. Bichrome models "which
character are you steering" as PlayerOrange <-> PlayerOrangeActive, so two of its
four win conditions are ALWAYS vacuous -- every level of it was permanently
unwinnable. The guard now asks whether ANY player class survives rather than
whether that one does, which is the condition it actually meant.

**The frame could not show the win** (`data/puzzlescript_games/Bichrome.txt`).
The widest two levels render at 4 px per cell, and the 5x5 sprites are point-
sampled down to that: the only pixels of a character that survive are its ring's
four corners, and the only pixels of a target that survive are its middle -- which
is exactly what a character standing on it covers. So on those levels the winning
frame was pixel-identical to the same character standing on a plain wall, and
both active characters rendered as the same yellow block, making which one you
were steering unreadable. Two sprite edits fix both: the targets' red marking
moved from the middle to the border (the ring cannot cover it, and target-vs-wall
becomes a 16-pixel difference instead of a 1-pixel one), and the "active" yellow
moved from the ring's corners to the CELL's corners, where it survives the
downsample without covering any of the character's own colour. Both edits are
sprite-only and symmetric under the frame flips; the rules are untouched.

Expert solver
-------------
The interpreter is ~1000x too slow to search here (level 3 alone: 83s of engine
A* versus 0.1s on the model below), so this generator plans on a NATIVE model of
the mechanic -- the whole board is `bytes`, one code per cell, plus two positions
and the active flag -- and then certifies each plan by replaying it through the
real interpreter and requiring `check_win`. The model was fuzz-verified against
the interpreter first: 192k transitions over all twelve levels, comparing ALL
FIVE successors of every visited state (grid and win flag), zero mismatches.

Two searches over the five key presses, tried in order, first winner wins:

  1. **exact A\\***, guided by the shortest walk over the non-black cells. Every
     move steps one cell, whether it is a walk, a push or a swap, so that walk is
     a genuine lower bound and this search's plans are SHORTEST. It wins eight of
     the ten levels.
  2. **guided A\\***, using instead the shortest walk under the CURRENT wall
     layout -- a step into your own colour counts only when the cell behind it
     can take the push, so unlike (1) it knows that a wall of your own colour
     backed by a black wall is a dead end. Far more informative and not a bound,
     so its plans are winning but not certified shortest. It takes levels 2 and
     8, whose boards are mazes of the wrong colour that (1) cannot see into.

Recovery
--------
``recovery_mode = "reset"`` (inherited). A push that lands a wall somewhere
useless is not undoable, so a perturbed state is not re-plannable in general: the
episode-wide exploration prefix flails, ONE RESET restores the level's initial
state, and the cached plan replays a guaranteed win from there.

Augmentation
------------
Bichrome's engine state after reset is identical for every seed (the levels are
fixed ASCII maps), so the only per-(seed, level) variables are presentation: the
frame rotation plus an independent horizontal and vertical flip with the matching
directional action remap (`PuzzleScriptAdapter._FLIP_GAMES`), which re-sample the
board's 8-element symmetry group. The flips are exact symmetries here -- the game
is gravity-free, its input is screen-relative, its two movement rules name no
direction, and after the sprite edits above every sprite is mirror-symmetric.
There is deliberately no colour augmentation: orange-versus-blue IS the mechanic.

The expert plan is therefore seed-independent: solved once per level, cached, and
replayed per seed with that seed's remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_bichrome_training.py --episodes 200 \\
        --out data/training_multi_level/bichrome
"""

from __future__ import annotations

import heapq
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (PSAStarSolver, PSExpert,  # noqa: E402
                                     restore, snapshot)

GAME_NAME = "Bichrome"

#: One code per board cell. Every cell of every Bichrome level holds exactly one
#: of these, and no rule ever empties a cell (a push either swaps two walls or is
#: refused), so EMPTY is modelled for completeness and never actually occurs.
EMPTY, WALL_O, WALL_B, BLACK, TARGET_O, TARGET_B = range(6)

#: Engine action names, indexed the way the searches branch: 0-3 are the moves in
#: `_Board.delta` order, 4 is the control handover.
ACTIONS = ("up", "down", "left", "right", "action")

#: Per character (0 = orange, 1 = blue): the two wall kinds it PUSHES, the plain
#: wall of the other colour it SWAPS with, and the other's target, which refuses
#: a push (`OrangeBlockade` / `BlueBlockade` in the ruleset).
_OWN = ((WALL_O, TARGET_O), (WALL_B, TARGET_B))
_OTHER_PLAIN = (WALL_B, WALL_O)
_OTHER_TARGET = (TARGET_B, TARGET_O)

INF = 1 << 20

#: Object names read off the parsed game, in the order `_Board` unpacks them.
_OBJECTS = ("playerorange", "playerblue", "playerorangeactive", "playerblueactive",
            "wallorange", "wallblue", "wall", "targetorange", "targetblue")


class _Board:
    """The native Bichrome model: the wall layer as flat `bytes`, plus the two
    characters' cells and which one is active.

    Fuzz-verified against the interpreter (see the module docstring). The board
    is flat rather than 2-D because the state is a dict key on every node of
    every search: `bytes` hashes in C, and a push touches two cells, so a
    successor costs one bytearray copy rather than a rebuilt grid of tuples.
    Horizontal moves check `col` so a step cannot wrap around a row -- every
    shipped level is walled all the way round, but the searches must not be
    the thing that depends on it."""

    def __init__(self, eng, ids):
        po, pb, poa, pba, wo, wb, blk, to, tb = ids
        grid = eng.grid
        self.h, self.w = len(grid), len(grid[0])
        cells = bytearray(self.h * self.w)
        orange = blue = None
        active = 0
        for r in range(self.h):
            for c in range(self.w):
                cell = grid[r][c]
                i = r * self.w + c
                if blk in cell:
                    cells[i] = BLACK
                elif to in cell:
                    cells[i] = TARGET_O
                elif tb in cell:
                    cells[i] = TARGET_B
                elif wo in cell:
                    cells[i] = WALL_O
                elif wb in cell:
                    cells[i] = WALL_B
                if po in cell:
                    orange = i
                elif poa in cell:
                    orange, active = i, 0
                if pb in cell:
                    blue = i
                elif pba in cell:
                    blue, active = i, 1
        self.start = (bytes(cells), orange, blue, active)
        self.delta = (-self.w, self.w, -1, 1)
        self.col = tuple(i % self.w for i in range(self.h * self.w))
        self._walk_fields: dict[int, list[int]] = {}
        self._push_fields: dict[tuple, list[int]] = {}
        self._open = tuple(c != BLACK for c in cells)   # black walls never move

    def ok(self, src, dst, di):
        """Is ``dst`` the cell ``di`` away from ``src``, on the board?"""
        return (0 <= dst < len(self._open)
                and (di < 2 or self.col[dst] == self.col[src]
                     + (1 if di == 3 else -1)))

    # -- transition ---------------------------------------------------------
    def step(self, state, ai):
        cells, orange, blue, active = state
        if ai == 4:
            return (cells, orange, blue, 1 - active)
        d = self.delta[ai]
        me = orange if active == 0 else blue
        other = blue if active == 0 else orange
        q = me + d
        if not self.ok(me, q, ai):
            return state
        own_a, own_b = _OWN[active]
        qc = cells[q]
        if qc == BLACK:
            return state
        if qc == own_a or qc == own_b:
            if other == q:
                # The other character is standing on the wall: its move is
                # cancelled, and we cannot enter a cell it occupies either.
                return state
            s = q + d
            if not self.ok(q, s, ai):
                return state
            sc = cells[s]
            if (sc == own_a or sc == own_b or sc == BLACK
                    or sc == _OTHER_TARGET[active]):
                return state
            buf = bytearray(cells)
            buf[s] = qc
            buf[q] = sc if sc == _OTHER_PLAIN[active] else EMPTY
            cells = bytes(buf)
        elif other == q:
            return state
        return ((cells, q, blue, 0) if active == 0 else (cells, orange, q, 1))

    def won(self, state):
        cells, orange, blue, _active = state
        return cells[orange] == TARGET_B and cells[blue] == TARGET_O

    # -- distance fields ----------------------------------------------------
    def walk_field(self, goal):
        """Steps to ``goal`` over every non-black cell, ignoring wall colours.

        A move costs one whether it is a walk, a push or a swap, so this is a
        genuine lower bound on either character's remaining moves -- which is
        what makes search 1 exact. Black walls are the only static geometry in
        the game, so one field per goal cell serves the whole level."""
        field = self._walk_fields.get(goal)
        if field is not None:
            return field
        n = len(self._open)
        dist = [INF] * n
        dist[goal] = 0
        queue = deque([goal])
        while queue:
            cur = queue.popleft()
            nd = dist[cur] + 1
            for di in range(4):
                nxt = cur + self.delta[di]
                if (self.ok(cur, nxt, di) and self._open[nxt]
                        and dist[nxt] > nd):
                    dist[nxt] = nd
                    queue.append(nxt)
        return self._walk_fields.setdefault(goal, dist)

    def push_field(self, cells, colour):
        """Steps to that colour's own target under the CURRENT wall layout: a
        step into your own colour counts only when the cell behind it can take
        the push.

        Reverse BFS from the target, so one pass fields every start cell. Keyed
        by the layout, which a whole run of walking shares, so it is rebuilt once
        per push rather than once per node. Not a lower bound -- pushes rearrange
        the very walls it froze -- but it is the estimate that knows a wall of
        your own colour backed by a black wall is a dead end."""
        key = (cells, colour)
        field = self._push_fields.get(key)
        if field is not None:
            return field
        goal = cells.find(TARGET_B if colour == 0 else TARGET_O)
        n = len(cells)
        dist = [INF] * n
        if goal < 0:
            return self._push_fields.setdefault(key, dist)
        own_a, own_b = _OWN[colour]
        other_target = _OTHER_TARGET[colour]
        dist[goal] = 0
        queue = deque([goal])
        while queue:
            cur = queue.popleft()
            nd = dist[cur] + 1
            here = cells[cur]
            if here == BLACK:
                continue
            for di in range(4):
                d = self.delta[di]
                src = cur - d                    # the cell moved FROM
                if not self.ok(cur, src, di ^ 1) or dist[src] <= nd:
                    continue
                if here == own_a or here == own_b:
                    beyond = cur + d             # where the pushed wall lands
                    if not self.ok(cur, beyond, di):
                        continue
                    bc = cells[beyond]
                    if (bc == own_a or bc == own_b or bc == BLACK
                            or bc == other_target):
                        continue
                dist[src] = nd
                queue.append(src)
        return self._push_fields.setdefault(key, dist)

    #: What the guided estimate charges a character whose target the current
    #: wall layout walls off entirely. That is the ORDINARY case -- it is true of
    #: the start state of most levels, and rearranging walls is exactly what the
    #: game is about -- so those states must sort behind the ones that have a way
    #: through, NOT be pruned. (Pruning them costs the guided search most of the
    #: levels it is there to win.)
    detour = 25

    def estimate(self, state, exact):
        """Remaining-moves estimate: each character's distance to the target it
        must stand on, plus one handover when the one that still has to move is
        not the active one."""
        cells, orange, blue, active = state
        tb, to = cells.find(TARGET_B), cells.find(TARGET_O)
        if tb < 0 or to < 0:
            return INF
        if exact:
            do, db = self.walk_field(tb)[orange], self.walk_field(to)[blue]
        else:
            do = self.push_field(cells, 0)[orange]
            db = self.push_field(cells, 1)[blue]
            if do >= INF:
                do = self.walk_field(tb)[orange] + self.detour
            if db >= INF:
                db = self.walk_field(to)[blue] + self.detour
        if do >= INF or db >= INF:
            return INF          # walled off by BLACK walls: nothing can help
        if do and db:
            return do + db + 1
        if do:
            return do + (0 if active == 0 else 1)
        if db:
            return db + (1 if active == 0 else 0)
        return 0

    # -- walking ------------------------------------------------------------
    def walk(self, cells, colour, src, blocked):
        """Shortest walks (no pushes) from ``src`` for one character, as parent
        pointers ``{cell: (previous, direction) | None}``. ``blocked`` is the
        other character, which is solid."""
        n = len(cells)
        own_a, own_b = _OWN[colour]
        parent = {src: None}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for di in range(4):
                nxt = cur + self.delta[di]
                if (not self.ok(cur, nxt, di) or nxt in parent
                        or nxt == blocked):
                    continue
                nc = cells[nxt]
                if nc == BLACK or nc == own_a or nc == own_b:
                    continue
                parent[nxt] = (cur, di)
                queue.append(nxt)
        return parent

    @staticmethod
    def route(parent, cell):
        out = []
        while parent[cell] is not None:
            cell, di = parent[cell]
            out.append(di)
        out.reverse()
        return out


# ---------------------------------------------------------------------------
# Search 1 and 2: A* over primitive key presses
# ---------------------------------------------------------------------------

def _press_astar(board, exact, node_cap):
    """A* over the five key presses. ``exact`` picks `_Board.walk_field` (a
    lower bound, so plans are shortest) over `_Board.push_field` (informative but
    not a bound, so plans are merely winning)."""
    start = board.start
    if board.won(start):
        return []
    h = board.estimate(start, exact)
    if h >= INF:
        return None
    heap = [(h, 0, 0, start)]
    best = {start: 0}
    parent = {start: None}
    counter = nodes = 0
    while heap:
        _f, g, _c, state = heapq.heappop(heap)
        if best[state] < g:
            continue
        for ai in range(5):
            nxt = board.step(state, ai)
            nodes += 1
            if nxt == state or best.get(nxt, 1 << 30) <= g + 1:
                continue
            best[nxt] = g + 1
            parent[nxt] = (state, ai)
            if board.won(nxt):
                plan = []
                while parent[nxt] is not None:
                    nxt, ai = parent[nxt]
                    plan.append(ACTIONS[ai])
                plan.reverse()
                return plan
            h = board.estimate(nxt, exact)
            if h >= INF:
                continue
            counter += 1
            heapq.heappush(heap, (g + 1 + h, g + 1, counter, nxt))
        if nodes >= node_cap:
            return None
    return None


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class BichromeExpert(PSExpert):
    """Plans on the native `_Board` and certifies the plan on the interpreter.

    Every plan this returns has been replayed, press by press, through the real
    PuzzleScript engine and seen to satisfy `check_win` -- so a modelling slip
    can only ever cost a level, never emit a trajectory that does not win. The
    plan memo, the level scoping and the restore discipline are `PSExpert`'s."""

    #: (exact?, node budget) in the order the searches are tried. The exact one
    #: gets a budget generous enough to be worth asking first -- it is the one
    #: that certifies a plan SHORTEST, and 3M nodes is where it stops buying
    #: levels: it wins eight of the ten in 8s together and spends the other 13s
    #: failing on the two it never gets (levels 2 and 8, which the guided search
    #: then takes in 0.7s and 6.4s). Every level is searched once per process and
    #: memoised, so this is a one-off ~28s at startup, not a per-seed cost.
    searches = ((True, 3_000_000), (False, 3_000_000))

    def setup(self) -> None:
        self.ids = tuple(self.g.obj_name_to_idx[name] for name in _OBJECTS)

    def heuristic(self, eng) -> int:
        """Unused. `_search` plans on the native model, so `PSExpert._astar` --
        the only thing that would call this -- never runs."""
        raise NotImplementedError("BichromeExpert plans on the native model")

    def _search(self, eng):
        board = _Board(eng, self.ids)
        for exact, cap in self.searches:
            plan = _press_astar(board, exact, cap)
            if plan is None:
                continue
            plan = self._certify(eng, plan)
            if plan is not None:
                return plan
        return None

    @staticmethod
    def _certify(eng, plan):
        """Replay ``plan`` on the interpreter and return it truncated at the
        press that wins, or None if it does not win.

        Truncation matters as much as the check: `check_win` reads a per-step
        flag, so a plan that wins partway and keeps pressing keys silently
        "un-wins" itself."""
        snap = snapshot(eng)
        try:
            for i, direction in enumerate(plan):
                eng.step(direction)
                if eng.check_win():
                    return plan[:i + 1]
            return None
        finally:
            restore(eng, snap)


class BichromeSolver(PSAStarSolver):
    game_id = "puzzlescript_bichrome"
    game_name = GAME_NAME
    expert_cls = BichromeExpert

    #: The two carry-the-target levels, unsolved. Skipped up front rather than
    #: discovered, so startup does not burn both budgets on them at every launch.
    #: What was measured, so the next attempt does not start from nothing:
    #:
    #:   * A complete breadth-first sweep of level 4's PRIMITIVE state space,
    #:     8M states deep, never gets the orange character past column 5 of the
    #:     8 it must cross; it needs to reach the blue target in column 8.
    #:   * A search over walk-and-push MACROS (depth = pushes and handovers, not
    #:     key presses) reaches neither: 1.5M macro expansions on each, and a
    #:     complete macro sweep passes 6M states with a 3M-state frontier still
    #:     open, so the space is nowhere near exhausted either.
    #:   * Asking only for the NECESSARY HALF of level 4 -- can the orange
    #:     character stand on the blue target at all, never mind the blue one --
    #:     also fails at 4M macro states, while the same question for the blue
    #:     character is answered in FOUR macros. So the obstruction is specific
    #:     and one-sided: the barrier between the two halves is a column of
    #:     orange walls backed by a second column of orange walls, and every swap
    #:     that opens a cell for the orange character closes the one behind it.
    #:
    #: Whether that is a genuine deadlock or merely deep was not settled -- the
    #: two characters CAN ferry each other's walls across the barrier, which is
    #: what keeps the space open, so it is not proven unwinnable here.
    skip_levels = frozenset({4, 9})

    #: Budgets live on the expert (one per search); `node_cap` and `weight` are
    #: `PSExpert._astar`'s, which this expert replaces outright.
    #: Plans run to ~90 presses on the carry-the-target levels; the ceiling only
    #: has to leave room for a re-plan after an epsilon detour, and `epsilon` is
    #: 0 for this family.
    max_steps = 300


if __name__ == "__main__":
    sys.exit(BichromeSolver.main())
