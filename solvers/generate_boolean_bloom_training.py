"""Generate Phase-1 training data for the PuzzleScript game
ps:boolean_bloom_0_37 ("Boolean Bloom 0.37" by vexorian).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other search-the-interpreter ps: generators. This file is the game-specific part:
a native model of the mechanic, the macro search that plans on it, and the engine
replay that certifies every plan before it is used.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_boolean_bloom",
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
A gardener walks a walled garden full of PLANTS. Each plant is a head facing one
of the four directions plus the straight stem it has already laid down, and every
head grows ONE CELL PER TICK, forever, until something solid stands in its way --
and the growth ticks run to a fixpoint inside a single key press, so what you see
after every move is the plants at their full extent under the current obstacles.
Some cells are GOALS; the level is won when every goal holds a plant HEAD (a
stem lying across a goal does not count) and the arrangement is stable.

Because a head only ever stops against something solid, winning is not about
steering plants -- it is about placing the WALL THEY STOP AGAINST one cell past
each goal. The gardener's tools are:

  * **pushing blocks**, in chains (a pushed block is itself a pusher), which is
    how a stopper gets built where a goal needs one;
  * **retracting a plant**, by shoving into a head from the front while its own
    stem lies behind it: the head backs up one cell and the pusher takes its
    place. This is the ONLY way anything ever gets in front of a head -- growth
    is instantaneous, so the moment a blocker is removed the plant fills the gap
    before the gardener could walk into it. Shoving a BLOCK into a head (blocks
    are pushers too) is therefore how a stopper is planted in already-grown
    territory, and the gardener's own body doing it is how a head is pinned in
    place for the last move of a level.

The other half of the game -- the "boolean" -- is doors driven by buttons.
Anything HEAVY holds a button down: the gardener, a block, a plant head, even a
length of stem grown across it. Four door kinds read the buttons as four
different gates, and a closed door is solid to gardener, block and plant alike:

    Door       open iff EVERY button is held      (AND)
    NegaDoor   open iff SOME button is free       (NAND)
    Trap       open iff NO button is held         (NOR)
    NegaTrap   open iff SOME button is held       (OR)

A door can never close on top of something, so standing on one jams it open.

Pressing X three times in a row is the author's built-in LEVEL SKIP, and the
interpreter honours it: three presses is an instant win from any state. It is
excluded from the search's action set AND from the exploration policy
(`available_actions`), or every "solution" in this corpus would be that cheat.

Expert solver
-------------
The interpreter runs at ~670 steps/s here -- a single growth fixpoint is dozens
of rule passes -- which is far too slow to search, so this generator plans on a
NATIVE model of the mechanic (the collision layer as flat `bytes`, plus the
gardener's cell; doors are recomputed from the buttons every tick rather than
stored) and certifies each plan by replaying it through the real interpreter and
requiring `check_win`.

The model was fuzz-verified against the interpreter first: random playouts over
all nine levels, comparing the full collision layer AND the derived closed-door
set after every key press (see ``--fuzz``).

The search branches on MACROS, not key presses. In a settled state the only
moves that change anything are a push and a retraction, and everything else is
the gardener walking; a primitive search re-derives those walks at every depth
and drowns. A macro is ``walk to the standing cell, then shove k times``,
enumerated over every block, every retractable head, and every button/door cell
worth standing on, and it is SIMULATED rather than assumed -- a walk that the
plants grow across mid-route simply ends where it ends, and the state it reaches
is still a legal one for the search to expand.

What makes even that tractable is `_Board.key`: in a settled position every
empty cell is one no head could grow into (if one could, it already would have),
so walking onto an empty cell can never block a plant and therefore changes
nothing at all. Two states that differ only in where the gardener stands inside
one walkable region are the same position, and are merged. The exceptions are
the two ways his body is part of the machine -- pinning a head he has just
retracted, or holding a button down -- and there the key falls back to his exact
cell.

Seven of the nine levels are solved, at 15 / 31 / 41 / 98 / 86 / 40 / 65 presses
for levels 0, 1, 2, 3, 5, 6, 7. Levels 4 and 8 are not; `skip_levels` records
what was measured and where the search runs out.

Recovery
--------
``recovery_mode = "reset"`` (inherited). Plant growth is irreversible except by
retraction, and a block shoved into a dead end is gone for good, so a perturbed
state is not re-plannable in general: the episode-wide exploration prefix flails,
ONE RESET restores the level's initial state, and the cached plan replays a
guaranteed win from there.

Augmentation
------------
The engine state after reset is identical for every seed (the levels are fixed
ASCII maps), so the only per-(seed, level) variables are presentation: the frame
rotation plus an independent horizontal and vertical flip, each with the matching
directional action remap (`PuzzleScriptAdapter._FLIP_GAMES`, added for this game
-- the reasoning is recorded there, and it turns on the fact that this ruleset's
directional rules and directional sprites both come in mirrored pairs). The
expert plan is therefore seed-independent -- solved once per level, cached to
``data/boolean_bloom_plans.json``, and replayed per seed with that seed's
remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_boolean_bloom_training.py --episodes 200 \\
        --out data/training_multi_level/boolean_bloom
    python solvers/generate_boolean_bloom_training.py --fuzz 300
"""

from __future__ import annotations

import heapq
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from solvers.common.ps_astar import (PSAStarSolver, PSExpert,  # noqa: E402
                                     restore, snapshot)

GAME_NAME = "Boolean_Bloom_0.37"

#: Where the ladder's results are kept between runs, in the repo's usual
#: ``data/<game>_plans.json`` shape. The plans are seed-independent, so this
#: turns the one-off search cost into a one-off-EVER cost -- which matters most
#: for `parallelize_generator.py`, where every shard would otherwise redo it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "boolean_bloom_plans.json"

# --------------------------------------------------------------------------
# Cell codes for the ONE collision layer that matters
# --------------------------------------------------------------------------
# PuzzleScript puts Player, Wall, Block, every plant piece and every *closed*
# door on a single collision layer, so one code per cell describes the whole
# board. Closed doors are the exception: they are recomputed from the buttons at
# the top of every growth tick, so they are derived, never stored -- storing them
# would let two states that play identically hash apart.
EMPTY, WALL, BLOCK, PLAYER, STEM_V, STEM_H, HEAD_U, HEAD_D, HEAD_L, HEAD_R = range(10)

#: The four heads are the TOP four codes, so ``code >= FIRST_HEAD`` is exactly
#: "a plant head is here" -- the only thing that satisfies a goal, and the test
#: `won` and the macro enumeration both run per cell. Keep them last.
FIRST_HEAD = HEAD_U

#: Direction indices, and the engine action name each one drives.
ACTIONS = ("up", "down", "left", "right")
OPP = (1, 0, 3, 2)
HEAD_BY_DIR = (HEAD_U, HEAD_D, HEAD_L, HEAD_R)
STEM_BY_DIR = (STEM_V, STEM_V, STEM_H, STEM_H)

#: Growth is resolved right, left, up, down (the order the rules are written in),
#: and a goal-bound head moves before a free-space-bound one. Both passes are
#: reproduced faithfully even though the vacated cell always becomes a stem --
#: which is solid -- so no head can ever follow another and the order provably
#: cannot matter. Cheap to keep, and it is one fewer thing the fuzz has to prove.
GROW_ORDER = (3, 2, 0, 1)

#: Door kinds, in the order the rules read the buttons.
NO_DOOR, DOOR, NEGADOOR, TRAP, NEGATRAP = range(5)

INF = 1 << 20

#: Object names read off the parsed game. Each tuple is (code, names): every name
#: in the tuple maps a cell to that code. The player's four sprite variants are
#: one object as far as play is concerned (the rules that pick between them only
#: choose which way the gardener faces), and the "Off" plant heads are a head
#: mid-tick that has already grown this tick -- never seen in a settled state.
_CODE_NAMES = (
    (WALL, ("wall",)),
    (BLOCK, ("block",)),
    (PLAYER, ("playerfront", "playerleft", "playerright",
              "playerfrontskip0", "playerfrontskip1")),
    (STEM_V, ("plantvertical",)),
    (STEM_H, ("planthorizontal",)),
    (HEAD_U, ("plantheadup", "plantheadupoff")),
    (HEAD_D, ("plantheaddown", "plantheaddownoff")),
    (HEAD_L, ("plantheadleft", "plantheadleftoff")),
    (HEAD_R, ("plantheadright", "plantheadrightoff")),
)

#: Cell -> door kind. Read off the door layer, which no rule ever moves.
_DOOR_NAMES = ((DOOR, "door"), (NEGADOOR, "negadoor"),
               (TRAP, "trap"), (NEGATRAP, "negatrap"))

#: The *closed* form of each door kind, for the fuzz comparison only.
_CLOSED_NAMES = {DOOR: "closeddoor", NEGADOOR: "closednegadoor",
                 TRAP: "closedtrap", NEGATRAP: "closednegatrap"}


class _Board:
    """The native Boolean Bloom model: the collision layer as flat `bytes` plus
    the gardener's cell.

    Flat rather than 2-D because the state is a dict key on every node of every
    search: `bytes` hashes in C, and a push touches a handful of cells, so a
    successor costs one bytearray copy rather than a rebuilt grid of sets.
    Horizontal moves check the row so a step cannot wrap around the board --
    every shipped level is walled all the way round, but the searches must not be
    the thing that depends on it.

    The static geometry (walls are part of ``cells``; goals, buttons and the door
    layer are not) is read once from the interpreter's grid and shared by every
    state of the level.
    """

    def __init__(self, eng, ids, doors, goal_id, button_id):
        grid = eng.grid
        self.h, self.w = len(grid), len(grid[0])
        self.n = self.h * self.w
        cells = bytearray(self.n)
        goals, buttons, door_cells = [], [], []
        player = None
        for r in range(self.h):
            for c in range(self.w):
                cell = grid[r][c]
                i = r * self.w + c
                for code, oid in ids:
                    if oid in cell:
                        cells[i] = code
                        if code == PLAYER:
                            player = i
                        break
                if goal_id in cell:
                    goals.append(i)
                if button_id in cell:
                    buttons.append(i)
                for kind, oid in doors:
                    if oid in cell:
                        door_cells.append((i, kind))
                        break
        self.goals = tuple(goals)
        self.goal_set = frozenset(goals)
        self.buttons = tuple(buttons)
        self.door_cells = tuple(door_cells)
        #: Cells where merely STANDING does something -- the gates' inputs and
        #: the gates themselves. `key` refuses to merge the gardener away here.
        self.special = frozenset(buttons) | {c for c, _ in door_cells}
        self.delta = (-self.w, self.w, -1, 1)
        self.row = tuple(i // self.w for i in range(self.n))
        self.start = (bytes(self._settle(cells)), player)

    # -- geometry -----------------------------------------------------------
    def ok(self, src, dst, di):
        """Is ``dst``, the cell ``di`` away from ``src``, on the board?"""
        if dst < 0 or dst >= self.n:
            return False
        return di < 2 or self.row[dst] == self.row[src]

    # -- doors --------------------------------------------------------------
    def closed(self, cells):
        """Which door cells are solid right now.

        The four gates read the SAME two facts about the buttons -- does some
        button lie free, does some button lie held -- so both are computed once
        and the kinds just pick their side. A door with something standing on it
        cannot close, which is the rule that lets the gardener jam one open.
        Returns None when the level has no doors at all, which is most of them
        and saves the allocation on every tick of every node."""
        if not self.door_cells:
            return None
        some_free = some_held = False
        for b in self.buttons:
            if cells[b] in (EMPTY, WALL):
                some_free = True
            else:
                some_held = True
        shut = bytearray(self.n)
        for i, kind in self.door_cells:
            if cells[i] != EMPTY:
                continue                     # jammed open by whatever is on it
            if kind == DOOR:
                sh = some_free               # open iff every button is held
            elif kind == NEGADOOR:
                sh = not some_free           # open iff some button is free
            elif kind == TRAP:
                sh = some_held               # open iff no button is held
            else:
                sh = not some_held           # open iff some button is held
            if sh:
                shut[i] = 1
        return shut

    # -- growth -------------------------------------------------------------
    def _settle(self, cells):
        """Run the growth fixpoint in place and return ``cells``.

        One tick advances every head that has a free cell ahead of it by exactly
        one, leaving its stem behind; the interpreter repeats the tick (via
        ``again``) until nothing grows, all inside a single key press. Doors are
        re-read at the top of each tick, so a plant that grows across a button
        opens (or slams) a door for the tick after."""
        heads = [i for i in range(self.n) if cells[i] >= FIRST_HEAD]
        if not heads:
            return cells
        while True:
            shut = self.closed(cells)
            moved = False
            done = set()
            for goal_pass in (True, False):
                for di in GROW_ORDER:
                    hc, d, stem = HEAD_BY_DIR[di], self.delta[di], STEM_BY_DIR[di]
                    for k, h in enumerate(heads):
                        if k in done or cells[h] != hc:
                            continue
                        a = h + d
                        if not self.ok(h, a, di) or cells[a] != EMPTY:
                            continue
                        if shut is not None and shut[a]:
                            continue
                        if goal_pass and a not in self.goal_set:
                            continue
                        cells[a] = hc
                        cells[h] = stem
                        heads[k] = a
                        done.add(k)
                        moved = True
            if not moved:
                return cells

    # -- transition ---------------------------------------------------------
    def step(self, state, di):
        """One key press. Returns the settled successor state (``state`` itself,
        unchanged, when the move is refused).

        Three things can happen when the gardener shoves in direction ``di``.
        A run of blocks ahead is pushed as one chain (a pushed block is itself a
        pusher, so the rule cascades). If the far end of that chain is a plant
        head FACING BACK at the pusher with its own stem directly behind it, the
        head RETRACTS one cell and the chain slides into the cell it left --
        which is how anything ever gets in front of a growing head. Otherwise
        the chain moves only if the cell past it is free."""
        cells_b, p = state
        cells = bytearray(cells_b)
        shut = self.closed(cells)
        d = self.delta[di]

        q = p
        chain = []
        while True:
            nxt = q + d
            if not self.ok(q, nxt, di):
                return state                     # the chain runs off the board
            if cells[nxt] == BLOCK:
                chain.append(nxt)
                q = nxt
                continue
            q = nxt
            break

        beyond = q + d
        if (cells[q] == HEAD_BY_DIR[OPP[di]] and self.ok(q, beyond, di)
                and cells[beyond] == STEM_BY_DIR[di]):
            cells[beyond] = cells[q]             # the head backs up one cell
        elif cells[q] != EMPTY or (shut is not None and shut[q]):
            return state                         # refused: nothing moves at all

        # The chain slides forward one: the last block takes ``q``, the gardener
        # takes the first block's cell, and every block between keeps its code.
        if chain:
            cells[q] = BLOCK
            cells[chain[0]] = PLAYER
            newp = chain[0]
        else:
            cells[q] = PLAYER
            newp = q
        cells[p] = EMPTY
        return bytes(self._settle(cells)), newp

    def won(self, state):
        cells = state[0]
        return all(cells[g] >= FIRST_HEAD for g in self.goals)

    # -- state identity -----------------------------------------------------
    def key(self, state, parent=None):
        """The search's notion of "the same position". Collapsing pure walking
        is what makes this game searchable at all.

        In a SETTLED state every empty cell is a cell no head could grow into --
        if one could, it already would have. So the gardener can never block a
        plant by walking onto an empty cell, and walking changes nothing
        whatsoever: two states that differ only in where he stands inside one
        walkable region have exactly the same futures, and are merged into one
        node keyed by that region.

        The two exceptions are the two ways his body is part of the machine, and
        both fall back to keying on his exact cell:

          * he is STANDING IN FRONT OF A HEAD (only ever true just after a
            retraction, for the same reason as above) -- he is the thing holding
            that plant in place, and stepping aside releases it;
          * he is standing on a button or on a door, where his weight is holding
            a gate open or shut.

        Merging costs optimality -- ``g`` still counts presses, but two routes
        into a region are no longer distinguished -- so plans are winning and
        engine-certified, not shortest.
        """
        cells, p = state
        for di in range(4):
            nb = p + self.delta[di]
            if self.ok(p, nb, di) and cells[nb] == HEAD_BY_DIR[OPP[di]]:
                return cells, p
        if p in self.special:
            return cells, p
        buf = bytearray(cells)
        buf[p] = EMPTY
        if parent is None:
            parent = self.walk_parents(cells, p)
        return bytes(buf), min(parent)

    # -- heuristic ----------------------------------------------------------
    def _goal_cost(self, cells, gi):
        """Rough remaining work for one uncovered goal: the cheapest head that
        could still arrive there, charged for what stands in its way.

        A head only ever travels along its own axis, so for each of the four
        facings the candidates lie on one line through the goal, and there are
        exactly two of them:

          * a head SHORT of the goal, which has to grow the rest of the way --
            charged for every obstacle on the line it must be cleared of, and for
            fetching a stopper if nothing waits one cell past the goal (without
            one it grows straight through);
          * a head PAST the goal, lying along its own stem, which has to be
            retracted back -- charged per retraction. Whatever did the shoving
            ends up in front of it, so this case needs no separate stopper. It is
            the case the level-3 up-plants turn on, and without it those goals
            look unreachable to a search that only ever looks upstream.

        Not a lower bound -- it says nothing about the walking, and it ignores
        two goals competing for one plant -- so plans are winning, not shortest.
        """
        best = self.unreachable_cost
        for di in range(4):
            d = self.delta[di]
            # Upstream: the head has not arrived yet.
            cur, obstacles = gi, 0
            while True:
                nxt = cur - d
                if not self.ok(cur, nxt, OPP[di]):
                    break
                c = cells[nxt]
                if c == WALL:
                    break
                if c == HEAD_BY_DIR[di]:
                    stopper = gi + d
                    need = 1 if (self.ok(gi, stopper, di)
                                 and cells[stopper] == EMPTY) else 0
                    best = min(best, 1 + 2 * obstacles + need)
                    break
                if c != EMPTY:
                    obstacles += 1
                cur = nxt
            # Downstream: the head overshot and lies along its own stem.
            if cells[gi] == STEM_BY_DIR[di]:
                cur, back = gi, 1
                while True:
                    nxt = cur + d
                    if not self.ok(cur, nxt, di):
                        break
                    c = cells[nxt]
                    if c == HEAD_BY_DIR[di]:
                        best = min(best, 1 + 2 * back
                                   + self._retract_penalty(cells, nxt, di))
                        break
                    if c != STEM_BY_DIR[di]:
                        break
                    cur, back = nxt, back + 1
        return best

    #: Charged when a head that must be retracted has no room in FRONT of it for
    #: whatever would do the shoving. Retraction is the one move that runs
    #: against the grain of the game, and a head jammed nose-first against
    #: another plant's stem usually cannot be retracted ever again -- level 2
    #: turns on exactly that, and the goal it strands looks one cheap shove away
    #: without this. Big enough to reorder the plan, not so big that the search
    #: abandons a state it has no alternative to.
    blocked_nose_cost = 6

    def _retract_penalty(self, cells, h, di):
        """Extra charge for retracting the head at ``h``: whatever shoves it has
        to stand one cell AHEAD of it, so that cell must be free for the
        gardener or hold a block he can drive in from behind."""
        front = h + self.delta[di]
        if not self.ok(h, front, di):
            return self.blocked_nose_cost
        c = cells[front]
        return 0 if c in (EMPTY, BLOCK, PLAYER) else self.blocked_nose_cost

    #: Charged for a goal no plant can currently reach at all. That is the
    #: ORDINARY case at the start of most levels -- clearing the line is exactly
    #: what the gardener is for -- so such states must sort behind the ones with
    #: a plant already lined up, NOT be pruned.
    unreachable_cost = 9

    def estimate(self, state):
        cells = state[0]
        return sum(self._goal_cost(cells, g) for g in self.goals
                   if cells[g] < FIRST_HEAD)

    # -- macros -------------------------------------------------------------
    def walk_parents(self, cells, src):
        """Shortest walks from ``src`` over cells nothing is standing in, as
        parent pointers ``{cell: (previous, direction) | None}``.

        This is the map of where the gardener could go IF the plants held still,
        which they do not -- a route can be grown across while it is being
        walked. That is fine: macros built from it are simulated key press by
        key press, so a stale route just ends early, in a state the search
        expands like any other."""
        shut = self.closed(cells)
        parent = {src: None}
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for di in range(4):
                nxt = cur + self.delta[di]
                if (not self.ok(cur, nxt, di) or nxt in parent
                        or cells[nxt] != EMPTY
                        or (shut is not None and shut[nxt])):
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

    def macros(self, state, max_shoves):
        """Every ``walk somewhere, then shove k times`` the gardener can start
        right now, as ``(direction indices, resulting state)`` pairs.

        The stands are the places a key press can accomplish anything from in a
        settled state -- behind a block, in front of a head, on a button or on a
        door -- plus the four bare steps from where he already is, which is what
        covers a retraction he is lined up for and the sidestep that releases a
        plant he is pinning. Repeated shoves are rolled into one macro (up to
        ``max_shoves``) so a block that has to travel five cells is one node of
        search depth, not five.

        Every macro is SIMULATED here, walk included, and the state it reaches
        is handed back with it: the callers would otherwise replay each one from
        scratch, and the model step is the entire cost of these searches. It also
        means a route the plants grow across mid-walk is caught rather than
        assumed -- the walk is abandoned where it actually ended, and only the
        stands the gardener really reached get their shoves enumerated."""
        cells, p = state
        parent = self.walk_parents(cells, p)
        out = []
        for di in range(4):
            nxt = self.step(state, di)
            if nxt != state:
                out.append(([di], nxt))

        stands: dict[int, set] = {}
        for i in range(self.n):
            c = cells[i]
            if c == BLOCK:
                for di in range(4):
                    s = i - self.delta[di]
                    if self.ok(i, s, OPP[di]) and s in parent and s != p:
                        stands.setdefault(s, set()).add(di)
            elif c >= FIRST_HEAD:
                # Shove into the head from the cell it is trying to grow into.
                di = OPP[c - FIRST_HEAD]
                s = i - self.delta[di]
                if self.ok(i, s, OPP[di]) and s in parent and s != p:
                    stands.setdefault(s, set()).add(di)
        # Standing on a button (or on a door, jamming it open) is a move with no
        # shove at all, and on the gate levels it is the whole point.
        for i in self.special:
            if i in parent and i != p:
                stands.setdefault(i, set())

        for s, dirs in stands.items():
            walk = self.route(parent, s)
            walked = state
            for step_di in walk:
                walked = self.step(walked, step_di)
            out.append((walk, walked))
            if walked[1] != s:
                continue              # a plant grew across the route on the way
            for di in dirs:
                cur = walked
                shoves = []
                for _ in range(max_shoves):
                    nxt = self.step(cur, di)
                    if nxt == cur:
                        break
                    shoves.append(di)
                    out.append((walk + shoves.copy(), nxt))
                    cur = nxt
        return out


# ---------------------------------------------------------------------------
# The search
# ---------------------------------------------------------------------------

def _replay(board, plan):
    """Index of the press in ``plan`` that wins, or -1 if it never does."""
    state = board.start
    for i, di in enumerate(plan):
        state = board.step(state, di)
        if board.won(state):
            return i
    return -1


def _prune(board, plan):
    """Cut every stretch of ``plan`` the model can do without, longest first.

    The searches merge states that differ only in where the gardener stands
    (`_Board.key`), which is what makes them tractable and also what lets a
    plan keep a stroll it had no reason to take. Every candidate here is
    re-simulated end to end and kept only if it still wins, so this can shorten
    a plan but never break one. It runs once per level and the result goes to
    the disk cache, so the O(n^3) presses it costs are paid once ever."""
    i = 0
    while i < len(plan):
        for length in range(len(plan) - i, 0, -1):
            shorter = plan[:i] + plan[i + length:]
            won = _replay(board, shorter)
            if won >= 0:
                plan = shorter[:won + 1]
                break
        else:
            i += 1
    return plan


def _macro_astar(board, node_cap, weight, max_shoves, merge=True):
    """Weighted A* over `_Board.macros`, costed in KEY PRESSES. Returns a list
    of direction indices, or None.

    ``g`` counts primitive presses so plans stay short in the units the agent
    actually pays, while the DEPTH of the search is the number of shoves: the
    walking in between is inside the macros.

    ``merge`` picks how states are deduped, which is the whole difference
    between the ladder's rungs. Merged, the key is `_Board.key` -- the gardener's
    walkable REGION rather than his cell -- and the state space collapses by
    orders of magnitude, at the price of costing two routes into a region the
    same and so of shortest plans (`_prune` claws most of that back). Unmerged,
    the key is the exact state, ``g`` is honest and a ``weight`` of 1 gives a
    genuinely shortest plan, but only the small levels are reachable.

    The frontier carries a representative concrete state per key. Any two states
    sharing a merged key have the same futures, so which one is expanded cannot
    change what is reachable, only the exact walk written into the plan."""
    start = board.start
    if board.won(start):
        return []
    keyof = board.key if merge else (lambda s: s)
    start_key = keyof(start)
    heap = [(weight * board.estimate(start), 0, 0, start, start_key)]
    best = {start_key: 0}
    parent = {start_key: None}
    counter = nodes = 0
    while heap:
        _f, g, _c, state, k = heapq.heappop(heap)
        if best[k] < g:
            continue
        for macro, cur in board.macros(state, max_shoves):
            nodes += 1
            ng = g + len(macro)
            ck = keyof(cur)
            if ck == k or best.get(ck, INF) <= ng:
                continue
            best[ck] = ng
            parent[ck] = (k, macro)
            if board.won(cur):
                plan = []
                while parent[ck] is not None:
                    ck, macro = parent[ck]
                    plan = list(macro) + plan
                return plan
            heapq.heappush(heap, (ng + weight * board.estimate(cur),
                                  ng, counter, cur, ck))
            counter += 1
        if nodes >= node_cap:
            return None
    return None


def _macro_beam(board, width, depth, node_cap, max_shoves):
    """`_Board.macros` explored by a width-capped breadth-first beam.

    WHY, given the A* above. Both pay the same thing per node -- one model step
    per press in a macro. A* spends that on the frontier its estimate likes,
    which is right only while the estimate can tell good states from bad. Here
    it often cannot: `_Board._goal_cost` scores the goals and says nothing about
    the block that has to be walked three rooms over to become a stopper, so
    across the whole middle of a level the estimate is FLAT, weighting it changes
    nothing, and A* degenerates into uniform-cost search at a depth where that is
    hopeless. A beam spends the same budget on breadth at every depth and uses
    the estimate only to decide who survives -- which is all a flat estimate is
    good for.

    The trade is plan length: beam plans wander where A*'s would not, which is
    what `_prune` is for. Dedup is `_Board.key`, kept across the WHOLE search
    rather than per depth, so a state re-reached later never re-expands."""
    start = board.start
    if board.won(start):
        return []
    frontier = [(start, [])]
    seen = {board.key(start)}
    nodes = 0
    for _ in range(depth):
        kids = []
        for state, path in frontier:
            for macro, cur in board.macros(state, max_shoves):
                nodes += 1
                if board.won(cur):
                    return path + macro
                k = board.key(cur)
                if k in seen:
                    continue
                seen.add(k)
                kids.append((board.estimate(cur), cur, path + macro))
                if nodes >= node_cap:
                    return None
        if not kids:
            return None                       # the reachable space closed
        # Sort on the estimate ALONE -- the tuple also carries a state, and
        # ordering by that would rank boards by their byte pattern.
        kids.sort(key=lambda kid: kid[0])
        frontier = [(state, path) for _h, state, path in kids[:width]]
    return None


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class BooleanBloomExpert(PSExpert):
    """Plans on the native `_Board` and certifies the plan on the interpreter.

    Every plan this returns has been replayed, press by press, through the real
    PuzzleScript engine and seen to satisfy `check_win` -- so a modelling slip
    can only ever cost a level, never emit a trajectory that does not win. The
    plan memo, the level scoping and the restore discipline are `PSExpert`'s.
    """

    #: X is the author's level-skip cheat (three presses in a row wins outright),
    #: so the search never sees it. `record_level`'s epsilon detour reads this
    #: list too, and `BooleanBloomSolver.available_actions` keeps it out of the
    #: exploration prefix; between them no recorded trajectory can take the skip.
    directions = list(ACTIONS)

    #: ``(weight, macro budget, shoves per macro, merge states?)`` rungs, in
    #: order. The budget counts EXPANDED macros, not seconds, so which levels
    #: solve -- and which plan each one gets -- is a property of the code and not
    #: of how busy the machine was.
    #:
    #: The first rung is exact-keyed and cost-blind, so what it returns is a
    #: SHORTEST plan; it is asked first because a demonstration that wanders is
    #: worse training data than one that does not, and it is cheap when it works.
    #: The rest merge (see `_macro_astar`) and lean progressively harder on the
    #: estimate, for the levels that need a block driven right across a board.
    searches = ((1, 250_000, 6, False), (1, 400_000, 6, True),
                (3, 1_500_000, 8, True), (8, 1_500_000, 12, True))

    #: ``(beam width, depth in shoves, macro budget, shoves per macro)`` rungs,
    #: tried after `searches` for the levels whose middles the estimate cannot
    #: rank at all. See `_macro_beam`.
    #:
    #: WIDTH IS THE DIAL, and it is worth more than depth or budget here. Level 5
    #: is the measurement: at width 6k the beam runs the whole space it can see
    #: dry in 16s and reports nothing, and at an effectively unbounded width it
    #: wins in 80s. That is the signature of a level whose solution passes
    #: through states the estimate ranks BADLY -- the block that has to be driven
    #: three rooms over scores worse at every step until it arrives -- so the only
    #: thing that keeps them alive is refusing to throw states away.
    #:
    #: The wide rung's budget is capped by MEMORY, not patience: its dedup set
    #: keeps one key per state it has ever generated, and a key is the whole
    #: collision layer, so five million of them is already a couple of GB.
    beams = ((6_000, 40, 2_500_000, 10), (60_000, 45, 5_000_000, 12))

    def setup(self) -> None:
        g = self.g
        self.ids = tuple((code, g.obj_name_to_idx[n])
                         for code, names in _CODE_NAMES for n in names)
        self.doors = tuple((kind, g.obj_name_to_idx[n])
                           for kind, n in _DOOR_NAMES)
        self.goal_id = g.obj_name_to_idx["goal"]
        self.button_id = g.obj_name_to_idx["button"]
        self._disk = self._load_disk()

    def board(self, eng) -> _Board:
        return _Board(eng, self.ids, self.doors, self.goal_id, self.button_id)

    def heuristic(self, eng) -> int:
        """Unused. `plan` plans on the native model, so `PSExpert._astar` --
        the only thing that would call this -- never runs."""
        raise NotImplementedError("BooleanBloomExpert plans on the native model")

    # -- disk plan cache ----------------------------------------------------
    @staticmethod
    def _load_disk() -> dict:
        """``{level: {"start": <collision layer as hex>, "plan": [...] | None}}``,
        or empty if unreadable. A cache that cannot be parsed is a miss, never a
        crash."""
        try:
            raw = json.loads(PLAN_CACHE.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:                                  # noqa: BLE001
            return {}

    def _save_disk(self) -> None:
        """Write the cache atomically -- shards started together would otherwise
        interleave into a truncated file (see `parallelize_generator`)."""
        try:
            PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PLAN_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self._disk[k] for k in sorted(self._disk)}, indent=1))
            os.replace(tmp, PLAN_CACHE)
        except OSError:
            pass                                           # the cache is optional

    # -- PSExpert interface -------------------------------------------------
    def plan(self, eng, level: int | None = None) -> list | None:
        """A WIN sequence of primitive directions from the engine's current
        state, or None. Leaves the engine's grid unchanged.

        `PSExpert.plan`'s in-process memo, plus a disk one for the state every
        seed re-plans from -- the level's start. Only that state is worth
        keeping, and each entry records the layout it was solved from, so
        editing the level in the .txt cannot silently serve a stale plan."""
        memo = (level, self._key(eng))
        if memo in self.cache:
            return self.cache[memo]
        board = self.board(eng)
        layout = board.start[0].hex()
        entry = self._disk.get(level)
        if entry is not None and entry["start"] == layout:
            self.cache[memo] = entry["plan"]
            return entry["plan"]
        plan = self._search_board(board, eng)
        self.cache[memo] = plan
        if entry is None:
            # Only the level's START state is worth keeping -- it is the one
            # state every seed re-plans from, and `discover_solvable` asks for it
            # before anything else can, so the slot goes to the right state.
            self._disk[level] = {"start": layout, "plan": plan}
            self._save_disk()
        return plan

    def _search_board(self, board, eng):
        """Walk the ladder until a rung returns a plan the INTERPRETER wins on.

        Each rung's plan is pruned on the model first (cheap, and it is the
        merged rungs' wandering it exists to remove) and only then replayed
        through the real engine, which is the thing that decides."""
        rungs = [(_macro_astar, (cap, weight, shoves, merge))
                 for weight, cap, shoves, merge in self.searches]
        rungs += [(_macro_beam, args) for args in self.beams]
        for search, args in rungs:
            plan = search(board, *args)
            if plan is None:
                continue
            plan = [ACTIONS[di] for di in _prune(board, plan)]
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


class BooleanBloomSolver(PSAStarSolver):
    game_id = "puzzlescript_boolean_bloom"
    game_name = GAME_NAME
    game_module_id = "ps:boolean_bloom_0_37"
    expert_cls = BooleanBloomExpert

    #: The two levels no rung reaches, skipped up front rather than discovered,
    #: so a fresh checkout does not burn every budget on them at each launch.
    #: What was measured, so the next attempt does not start from nothing:
    #:
    #:   * Both are five-goal levels, and both run the whole ladder out --
    #:     four A* rungs and both beams, the widest of them 5M macro expansions
    #:     at width 60k -- without a win. A one-off experiment at width 200k and
    #:     a 20M budget was cut off after ~25 minutes, still without one.
    #:   * Neither looks DEAD, only deep. Level 4's goal at row 5 wants the
    #:     row-5 block shoved twice into the head beside it (each shove retracts
    #:     the plant and walks the block along), then lifted out of the row
    #:     entirely, after which the gardener walks the head left across the
    #:     board with his own body as its stopper -- and its goal at row 3 needs
    #:     a stopper block ferried out of row 6, up two rooms and back down into
    #:     a one-cell slot. That is on the order of twenty-five shoves before
    #:     anything scores, over a board with five goals competing for four
    #:     blocks, and `_Board.estimate` -- which reads the goals and nothing
    #:     else -- cannot rank a single step of it.
    #:
    #: So the obstruction is the estimate, not the model or the macros: what
    #: would move these is an estimate that scores a BLOCK's distance to the cell
    #: it has to become a stopper at, not just the plants' distance to the goals.
    skip_levels = frozenset({4, 8})

    #: Budgets live on the expert (one per search); `node_cap` / `weight` are
    #: `PSExpert._astar`'s, which this expert replaces outright. Plans run to ~120
    #: presses; the ceiling only has to leave room for a re-plan, and `epsilon` is
    #: 0 for this family.
    max_steps = 400

    def available_actions(self, game) -> list[int]:
        """The four movement keys only.

        The adapter advertises ACTION5 as well, but in this game X is the
        author's level SKIP: three presses in a row is an instant win from any
        state. Left in, the exploration prefix would eventually hit that cheat
        and tape a two-second "solution" of a level it never played."""
        return [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Model fuzz-verification against the interpreter
# ---------------------------------------------------------------------------

def _engine_view(eng, expert):
    """The interpreter's grid reduced to ``(collision codes, closed door set)``,
    for comparison with the model."""
    cells = bytearray(len(eng.grid) * len(eng.grid[0]))
    shut = set()
    w = len(eng.grid[0])
    closed_ids = {expert.g.obj_name_to_idx[n]: k
                  for k, n in _CLOSED_NAMES.items()}
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            for code, oid in expert.ids:
                if oid in cell:
                    cells[i] = code
                    break
            if cell & closed_ids.keys():
                shut.add(i)
    return bytes(cells), shut


def _fuzz(episodes: int, seed: int = 0) -> int:
    """Random playouts on the interpreter and the model in lockstep, comparing
    the whole collision layer and the derived closed-door set after every press.

    This is what licenses planning on the model at all -- see the module
    docstring. Returns a process exit code."""
    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = BooleanBloomExpert(game)
    rng = random.Random(seed)
    checked = bad = 0
    for level in range(game.n_levels):
        for _ in range(episodes):
            game.set_level(level)
            eng = game._engine
            board = expert.board(eng)
            state = board.start
            for _ in range(40):
                di = rng.randrange(4)
                eng.step(ACTIONS[di])
                state = board.step(state, di)
                ecells, eshut = _engine_view(eng, expert)
                mshut = board.closed(state[0])
                mset = ({i for i in range(board.n) if mshut[i]}
                        if mshut is not None else set())
                checked += 1
                if ecells != state[0] or eshut != mset:
                    bad += 1
                    print(f"MISMATCH level {level} after {ACTIONS[di]}")
                    for r in range(board.h):
                        lo, hi = r * board.w, (r + 1) * board.w
                        print("  eng", list(ecells[lo:hi]))
                        print("  mod", list(state[0][lo:hi]))
                    print("  eng doors", sorted(eshut), "mod doors", sorted(mset))
                    return 1
                if eng.check_win() != board.won(state):
                    bad += 1
                    print(f"WIN MISMATCH level {level}: "
                          f"eng={eng.check_win()} model={board.won(state)}")
                    return 1
                if eng.check_win():
                    break
    print(f"fuzz: {checked} transitions checked over {game.n_levels} levels, "
          f"{bad} mismatches")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    if "--fuzz" in sys.argv:
        i = sys.argv.index("--fuzz")
        n = int(sys.argv[i + 1]) if len(sys.argv) > i + 1 else 50
        sys.exit(_fuzz(n))
    sys.exit(BooleanBloomSolver.main())
