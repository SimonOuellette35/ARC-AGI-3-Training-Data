"""Generate Phase-1 training data for the PuzzleScript game ps:match_flow
("Match Flow", edderiofer).

The harness -- the rotation contract, the trajectory recorder, the plan memo, the
on-disk plan cache and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: an exact model of the mechanic, two solvers over it, and
the reports that certify both.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_match_flow",
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
exactly. Every expert step carries a set of equally-optimal presses.

THE GAME
--------
You are a pink engineer on a board of electricity towers. A press slides you one
cell, and a tower you walk into slides one cell ahead of you -- a plain sokoban
push, with NO chain: a tower with anything at all behind it does not budge, and
neither do you. Some towers are BOLTED DOWN (`fsquare` / `ksquare`, drawn on a
black base rather than a dark grey one) and cannot be pushed at all; `block` is a
broken tower that pushes like a tower but counts as nothing.

You win the instant the towers form ONE CONNECTED, NON-BRANCHING run:

  * every tower orthogonally connected to every other (one component), and
  * no tower with three or more tower neighbours (that is the `overload` the
    story keeps complaining about -- towers "can only supply so much energy"), and
  * no tower standing on a red `deactivator` X.

Connected + max degree 2 means the final layout is a simple PATH or a simple
CYCLE through the grid, pinned through every bolted tower. That is the whole
puzzle: the bolted towers are the fixed points of a wiring diagram and the
pushable ones are the wire, and there are very few ways to draw it -- 1 to 11
per level (`--plans` counts them). Getting the wire into that shape is the
sokoban.

The last level swaps the win condition out entirely: while a `dosquare` (the
green-cornered tower) is on the board, connectivity means nothing and the only
win is parking it on the green `switch` pad.

THE WIN IS A TRANSITION, NOT A STATE
------------------------------------
`PSEngine.step` clears `_rule_win` at the top of every press and this game has no
`WINCONDITIONS` block, so `check_win()` reports what the LAST press produced.
The whole win is recomputed from the board by `late` rules on every press,
including a press that moved nothing -- so a board that is already wired
correctly is won by the NEXT press, in any direction, and `load_level` explicitly
clears the flag afterwards even though `run_rules_on_level_start` re-derived it.
That is not a corner case here: "Oh no, I'm trapped!" and its epilogue twin ship
already-wired (the cage around you is the joke), and both are won in one press.
So the search's goal test is ``won(step(state, d))``, never ``won(state)``, and
those two levels come out at d* = 1 with all five presses in the optimal set.

THE .txt CHANGES
----------------
One fidelity fix and four rendering fixes; no level was touched.

**1. The connectivity flood is wrapped in `startloop` / `endloop`.** That is a
no-op in real PuzzleScript -- a rule is re-applied until it stops matching, so
``late [squares connected|squares no connected no deactivator] -> [...]`` marks
the whole component on the turn it starts. This interpreter is not:
`PSEngine._execute_late_single` iterates each of a rule's four expanded
DIRECTIONS to a fixed point and never re-runs the direction LIST, so the mark
only travelled as far as one up/down/left/right sweep carried it and any winding
run of towers read as DISCONNECTED. Measured on the shipped boards: the ring in
"Oh no, I'm trapped!" is such a run, so the level -- which is meant to be won on
the first press, that being the joke -- could not be won at all, and neither could
its epilogue twin. (Same interpreter fact as ps:lovendpieces; grep any new ps:
game for a `late` rule whose RHS can create a new LHS match in another direction.)

The obvious follow-on worry does NOT hold here and was measured rather than
assumed. The sweep order is fixed in ENGINE coordinates whatever the screen
shows, so an unlooped flood is orientation-dependent IN GENERAL -- that is what
made the same bug unlearnable in ps:lovendpieces. On these boards it is not:
re-running the `--symmetry` chirality probe against an unlooped build gives 0
chiral transitions out of 23520 transformed presses, because the shapes that
defeat a four-sweep flood defeat it equally in all eight presentations. So this
is a fidelity fix and a two-levels-unwinnable fix, and it is not what entitles
the game to the flip augmentation.

**2. `iwall` was `transparent`.** It is a wall in every respect -- same collision
layer as `vwall`, no rule tells them apart -- but it drew as nothing, so the cage
in the two "trapped" levels was invisible. Drawn as `vwall` is, because that is
what it is.

**3. `switch` was `transparent`.** It is the target of the entire last level and
it drew as bare floor, i.e. the goal was unobservable. Drawn as a green pad
outline. It sits under the piece layer, so a tower parked on it hides it -- which
is exactly the frame on which the game is won.

**4. `ksquare` shipped with `fsquare`'s mechanics and `square`'s picture.**
Neither `fsquare` nor `ksquare` is in the `msquare` push group, so neither can be
moved, but `ksquare` was drawn on the dark-grey base every PUSHABLE tower uses:
it looked pushable and was not. Drawn as `fsquare`, which is what it behaves as.

**5. `dosquare` was a byte-identical picture of a plain `square`** while
REPLACING the win condition of the level it appears on. Given green corners, in
the switch's own colour, so the piece and its pad read as a pair.

**6. `overload` was invisible.** It drew one `red` centre pixel on top of a
tower whose own centre pixel is `lightred`, and this palette maps `red` and
`lightred` to the SAME index (8) -- so the "this tower is branching" marker
repainted the pixel that was already there. Drawn as four corner pips, a pattern
nothing else here uses (the deactivator's X is the inner diagonal).

THE MODEL
---------
`_Model` re-implements the mechanic natively over BITMASKS: static walls, bolted
towers, deactivator cells and switch cells live on the model; a state is
``(player cell, pushable towers, blocks, dosquare)`` as one int and three masks.
The win predicate is four bit operations (a degree test over three-way
intersections of the four shifts, then a flood for connectivity), which is what
makes enumerating a level affordable. ``--selfcheck`` fuzzes it against the real
interpreter over the shipped levels AND random synthetic boards, checking the
settled board, the REFUSAL decision and the win flag on every press.

TWO SOLVERS, AND WHY
--------------------
Eleven of the fourteen levels are small enough to solve EXACTLY and the
generator does not search them at all in the A* sense: a layered forward BFS runs
until the first winning press appears -- that depth IS d* -- and a reverse BFS
over the edges collected on the way gives distance-to-win for every state on a
shortest path. Certified-shortest plan, and the exact optimal-action SET at every
step, for free. (Same shape as ps:lovendpieces; the BOUND is what makes it cheap.)

Three levels are too deep for that (their wins are 30 to 68 presses away), so
they get a second solver: A* whose successors are ``walk to the push cell, then
push`` MACROS, so the depth is the number of pushes rather than the number of
presses. Its heuristic is admissible and has two terms, taken as a max and
minimised over the level's goal configurations:

  * a min-cost ASSIGNMENT of pushable towers to the goal's free cells, costed in
    push-distance (each push is at least one press), and
  * the MST of the free-grid metric closure over the player plus, for every tower
    that must move, the cell it stands on and the cell it will be pushed from.
    The player's whole trajectory is one connected walk that has to visit both
    (it steps into the tower's cell as it pushes), so the MST is a lower bound
    on the total press count -- and unlike the assignment it actually charges
    for the walking, which is three quarters of the cost on these boards.

The macro search keeps a winning macro as an INCUMBENT and only stops when the
frontier's f reaches its cost, because a macro is ``walk + push`` and macros cost
different amounts -- returning the first win generated is not the shortest one
(see `PSPushExpert.exact_goal_test` in the shared module, and ps:esl_puzzle_game
where exactly that was convicted).

WHAT IS PROVED AND WHAT IS NOT
------------------------------
Level 8 closes: the bounded exact pass empties its frontier, so 30 is proved
shortest. Levels 7 and 11 do NOT close inside the budget, and the honest reason
is that the heuristic is loose by a factor of two on them: the real cost there is
the ORDER the pushes are done in, and no cheap bound sees ordering. It is not one
refactor away either -- even an EXACT travelling-salesman bound over the cells the
player is forced to visit comes to 16 on level 7, against a plan of 46. Their
plans are the best the probe found, and this file says so rather than claiming
otherwise: ``--plans`` prints a `proved` column.

That distinction is carried into the LABELS, which is the point of stating it.
Three tiers, and ``--plans`` prints which one each level got:

  * ``field`` -- the exhaustive solve's distance field. The true optimal set.
  * ``measured`` -- once d* is PROVED, "is press d optimal here" becomes a
    decision with a tight bound (re-solve its successor asking only for a plan
    one press shorter), which is affordable where the open search was not. Also
    the true optimal set; level 8 gets it, and it found 3 tie steps the walk
    labels below would have missed.
  * ``walk`` -- what can be proved equal WITHOUT knowing d*. No rule in this game
    fires on a bare move, so during a walk toward a push cell every press that
    keeps the player on a shortest route to that same cell leaves an identical
    board and an identical remaining cost. Those tie with the taken press whether
    or not the plan is optimal. A push is labelled with itself. Sound -- every
    member is exactly as good as the one taken -- but a SUBSET of the truth. This
    is what levels 7 and 11 ship.

No step ever ships unlabelled. `_replay_check` verifies each tier and only as
strongly as that tier claims.

THE LEVELS
----------
All fourteen shipped boards are used as they are -- no level was authored here.
``twr`` counts every tower, bolted and pushable; ``goals`` is how many winning
layouts the board admits (blank where the exhaustive solver made the question
moot); ``ties`` is how many of the plan's steps have more than one right answer.

    lvl  size  twr blk goals plan proved  labels ties  what it is
      0  5x5    5   0     -     6   yes    field    0  two bolted towers on one
                                                       row, three loose ones to
                                                       thread between them
      1  6x6    5   0     -    12   yes    field    1  the first open board
      2  6x6    9   0     -    25   yes    field    1  no branching allowed --
                                                       the level that teaches it
      3  5x7    9   0     -    24   yes    field    1  four bolted towers, and
                                                       the wire is a zigzag
      4  5x3    5   1     -    12   yes    field    1  one broken tower in the way
      5  5x3    3   3     -    17   yes    field    1  a WALL of broken towers and
                                                       one usable one
      6  7x5    7   0     -    32   yes    field    9  the first deactivators
      7  9x7   17   0     1    46    NO     walk    0  a 3x3 lattice of bolted
                                                       towers; the one wiring is a
                                                       Hamiltonian path over it
                                                       and the puzzle is the ORDER
      8  7x7   19   2     1    30   yes measured    3  "Too many towers!" -- two
                                                       bolted columns, nine loose
                                                       towers, two broken ones and
                                                       ten deactivators
      9  7x7   16   0     -     8   yes    field    1  a ring with one gap and one
                                                       tower to plug it
     10  7x7   16   0     -     1   yes    field    1  already wired at reset: the
                                                       cage IS the ring (see "the
                                                       win is a transition")
     11  9x7   23   0    11    66    NO     walk    0  the bonus level; thirteen
                                                       loose towers and the only
                                                       board with more than three
                                                       wirings
     12  8x7   16   0     -     1   yes    field    1  the epilogue's copy of 10,
                                                       one blank row taller
     13  8x7   16   0     -     2   yes    field    0  the true end: push the
                                                       dosquare onto the switch

282 presses over the fourteen, 20 of them with a tie. Every plan is
engine-verified -- replayed through the interpreter, which must report the win.

A cold run derives all of this in about six minutes, nearly all of it levels 7
and 11, and writes it to ``data/match_flow_plans.json``; later runs and every
`parallelize_generator` shard read it back.

AUGMENTATION
------------
Engine state after reset is identical for every seed (the levels are fixed ASCII
maps), so the only per-(seed, level) variables are presentational: the frame
rotation in {0,1,2,3} plus an independent horizontal and vertical flip with the
matching directional action remap (`PuzzleScriptAdapter._FLIP_GAMES`, which this
game is added to). Every part of the mechanic is direction-free -- push,
connectivity, the degree test, the deactivator and the switch all name no
direction -- and the four `connected*` connection sprites are a closed ORBIT of
the 8-element group (a mirror maps `connectedl` art to `connectedr` art exactly
as it maps the connection), so a flipped board shows art the game owns.
``--symmetry`` proves that on the interpreter rather than arguing it, in two
stages: every level's plan replayed on all eight turned and mirrored copies of
its own board, and then random walks re-played press by press on the other seven
presentations, which is the stage with teeth (see ps:lovendpieces, where stage 1
alone reported SYMMETRIC on a game that was chiral).

RECOVERY
--------
``recovery_mode = "reset"`` (the family default). Exploration is genuinely
irreversible here -- a tower shoved against a wall can never be pulled back, and
a board can be wedged in one press -- so the episode-wide epsilon prefix explores
freely and ONE RESET restores the level start, from which the memoized plan
replays a guaranteed win.

Usage (run from the repo root):
    python solvers/generate_match_flow_training.py --episodes 200 \
        --out data/training_multi_level/match_flow

    python solvers/generate_match_flow_training.py --plans
    python solvers/generate_match_flow_training.py --selfcheck
    python solvers/generate_match_flow_training.py --symmetry
    python solvers/generate_match_flow_training.py --audit
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

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.common.ps_astar import (Plan, PSAStarSolver, PSExpert,     # noqa: E402
                                     _load_game_module)

GAME_NAME = "Match_Flow"
GAME_MODULE_ID = "ps:match_flow"

#: The presses the searches branch on, in the order they are tried. The game
#: declares `noaction`, so ACTION5 is bound to nothing -- it is kept in the list
#: because the adapter still offers it to an agent, and because on a board that
#: is ALREADY wired it wins like any other press. The state dedup prunes it
#: everywhere else; ``--plans`` measures that rather than assuming it.
KEYS = ("up", "down", "left", "right", "action")
_ACTION = KEYS.index("action")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The object classes that make up a state. `square` and `dosquare` are pushable
#: towers, `block` is a pushable non-tower, `fsquare`/`ksquare` are bolted towers.
_TOWERS = ("square", "fsquare", "ksquare", "dosquare")

#: Everything a state depends on. Deliberately NOT the whole object list: the
#: `connected*` / `overload` / `winning` markers are rebuilt from the board by
#: the late rules on every press, and the rule that seeds the connectivity flood
#: is `random`, so they are not a function of the state alone. See
#: `MatchFlowExpert._key`, which is where that mattered.
_STATE_OBJECTS = ("player", "square", "block", "dosquare", "fsquare", "ksquare",
                  "vwall", "iwall", "deactivator", "switch")


# ---------------------------------------------------------------------------
# Native model of the mechanic
# ---------------------------------------------------------------------------

class _Model:
    """Match Flow as a state machine over cell BITMASKS.

    A state is ``(player, squares, blocks, dosquare)``: the player's cell index
    plus three masks over ``r * w + c``. Walls, bolted towers, deactivator cells
    and switch cells are static and live on the model.

    Bitmasks rather than sets of cells because the win predicate -- which the
    search evaluates once per generated state -- is entirely shifts and masks:
    the overload test is the four neighbour shifts intersected three at a time,
    and connectivity is a flood of the same four shifts.
    """

    __slots__ = ("h", "w", "full", "walls", "fixed", "deact", "switch", "solid",
                 "nc0", "ncl")

    def __init__(self, h: int, w: int, walls: int, fixed: int, deact: int,
                 switch: int):
        self.h, self.w = h, w
        self.full = (1 << (h * w)) - 1
        self.walls, self.fixed = walls, fixed
        self.deact, self.switch = deact, switch
        #: everything a piece or the player can never enter
        self.solid = walls | fixed
        col0 = sum(1 << (r * w) for r in range(h))
        coll = sum(1 << (r * w + w - 1) for r in range(h))
        self.nc0 = self.full & ~col0        # everything but the first column
        self.ncl = self.full & ~coll        # everything but the last column

    # -- geometry -----------------------------------------------------------
    def sh(self, x: int, di: int) -> int:
        """``x`` translated one cell in direction ``KEYS[di]``. The column masks
        stop a left/right shift from wrapping into the neighbouring row."""
        w = self.w
        if di == 0:
            return x >> w
        if di == 1:
            return (x << w) & self.full
        if di == 2:
            return (x & self.nc0) >> 1
        return (x & self.ncl) << 1

    def cells(self, mask: int) -> list[int]:
        out = []
        while mask:
            b = mask & -mask
            out.append(b.bit_length() - 1)
            mask ^= b
        return out

    # -- dynamics -----------------------------------------------------------
    def step(self, state, di: int):
        """The board after pressing ``KEYS[di]``. A refused press returns the
        state unchanged -- this game has no `cancel` rule and no
        `require_player_movement`, so a blocked press is a plain no-op that
        still re-runs every win rule (which is why it can win).

        The push rule is ``[> player | msquare] -> [> player | > msquare]`` and
        there is no rule giving a pushed tower's force to the tower behind it,
        so a CHAIN never moves: anything at all behind the tower refuses the
        whole press, the player included.
        """
        if di == _ACTION:
            return state
        p, sq, bl, do = state
        w = self.w
        r, c = divmod(p, w)
        dr, dc = _DELTA[KEYS[di]]
        r2, c2 = r + dr, c + dc
        if not (0 <= r2 < self.h and 0 <= c2 < w):
            return state
        t = r2 * w + c2
        tb = 1 << t
        if self.solid & tb:
            return state
        movers = sq | bl | do
        if movers & tb:
            r3, c3 = r2 + dr, c2 + dc
            if not (0 <= r3 < self.h and 0 <= c3 < w):
                return state
            t2 = r3 * w + c3
            b2 = 1 << t2
            if (self.solid | movers) & b2:
                return state
            if sq & tb:
                sq = (sq & ~tb) | b2
            elif bl & tb:
                bl = (bl & ~tb) | b2
            else:
                do = (do & ~tb) | b2
        return (t, sq, bl, do)

    # -- the win predicate ---------------------------------------------------
    def won(self, state) -> bool:
        """True when the late rules would fire ``win`` on this board.

        Order matters and follows the rule list: a `dosquare` on the board
        cancels the connectivity win outright (``late [winning][dosquare] ->
        [][dosquare]``) and the only thing that can win is ``late [dosquare i]
        -> win``.

        Otherwise a tower on a deactivator loses twice over -- explicitly, and
        because the flood skips deactivated towers so one can never be marked
        `connected` -- and the three checks reduce to: no tower on a
        deactivator, no tower with three neighbours, one component.
        """
        _p, sq, _bl, do = state
        if do:
            return bool(do & self.switch)
        allsq = self.fixed | sq
        if not allsq:
            return True
        if allsq & self.deact:
            return False
        n = (self.sh(allsq, 0), self.sh(allsq, 1),
             self.sh(allsq, 2), self.sh(allsq, 3))
        for a in range(4):
            for b in range(a + 1, 4):
                for cc in range(b + 1, 4):
                    if n[a] & n[b] & n[cc] & allsq:
                        return False                     # a branching tower
        comp = allsq & -allsq
        while True:
            grow = comp
            for d in range(4):
                grow |= self.sh(comp, d)
            grow &= allsq
            if grow == comp:
                break
            comp = grow
        return comp == allsq

    # -- exhaustive solve ----------------------------------------------------
    def solve(self, start, cap: int):
        """A shortest press sequence from ``start`` plus the exact optimal SET at
        each of its steps, or ``(None, None)`` if the win is further away than
        ``cap`` states allow (or unreachable).

        Forward BFS one whole LAYER at a time until a layer produces a winning
        press. That depth is d*: BFS depth is minimal, so every state on a
        shortest path has been generated at its own depth and every edge out of a
        state shallower than d* has been recorded -- exactly the subgraph a
        reverse BFS from the wins needs to give true distance-to-win. The bound
        is the whole trick; the reachable components here run past a million
        states while most wins are a couple of dozen presses away.

        The goal test is on the SUCCESSOR of a press, not on a state, because the
        win in this game is a transition (see the module docstring). So the field
        this builds is "presses still to spend", and its base case is the set of
        states that have a winning press -- distance 1, never 0. A board that is
        already wired is one of those, which is how the two trapped levels come
        out at d* = 1 instead of at a zero-press "win" that never reaches
        ``GameState.WIN``.
        """
        depth = {start: 0}
        edges: dict = {}
        layer = [start]
        ones: list = []          # states one press away from a win
        d = 0
        while layer and not ones:
            nxt: list = []
            for state in layer:
                out = {}
                winner = False
                for di in range(len(KEYS)):
                    n = self.step(state, di)
                    if self.won(n):
                        winner = True          # a winning press ends the game
                        continue
                    out[di] = n
                    if n not in depth:
                        depth[n] = d + 1
                        nxt.append(n)
                        if len(depth) > cap:
                            return None, None
                edges[state] = out
                if winner:
                    ones.append(state)
            d += 1
            layer = nxt
        if not ones:
            return None, None

        # Reverse BFS over the forward-collected edges, from every state that can
        # win in one. Winning successors were never generated, so nothing can
        # route "through" a win.
        rev: dict = {}
        for state, out in edges.items():
            for n in out.values():
                rev.setdefault(n, []).append(state)
        dist = {s: 1 for s in ones}
        queue = deque(ones)
        while queue:
            state = queue.popleft()
            for prev in rev.get(state, ()):
                if prev not in dist:
                    dist[prev] = dist[state] + 1
                    queue.append(prev)

        presses: list[str] = []
        optsets: list[list[str]] = []
        state = start
        while True:
            r = dist[state]
            best = []
            for di in range(len(KEYS)):
                n = self.step(state, di)
                if r == 1:
                    if self.won(n):
                        best.append(di)
                elif not self.won(n) and dist.get(n, 1 << 30) == r - 1:
                    best.append(di)
            optsets.append([KEYS[di] for di in best])
            presses.append(KEYS[best[0]])
            if r == 1:
                break
            state = self.step(state, best[0])
        return presses, optsets

    def reachable(self, start, cap: int) -> tuple[set, bool]:
        """``(states, hit_the_cap)`` for the reachable component, wins included
        as leaves. Only the reports use it; it returns the partial set rather
        than None so a capped sweep still MEASURES what it saw."""
        seen = {start}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            for di in range(len(KEYS)):
                n = self.step(state, di)
                if n in seen:
                    continue
                seen.add(n)
                if len(seen) >= cap:
                    return seen, True
                if not self.won(n):
                    queue.append(n)
        return seen, False


def model_from_engine(eng, g) -> tuple:
    """Read ``(model, state)`` off the interpreter's live grid."""
    idx = g.obj_name_to_idx
    h, w = eng.height, eng.width
    walls = fixed = deact = switch = sq = bl = do = 0
    player = None
    for r in range(h):
        row = eng.grid[r]
        for c in range(w):
            cell, bit = row[c], 1 << (r * w + c)
            if idx["vwall"] in cell or idx["iwall"] in cell:
                walls |= bit
            if idx["fsquare"] in cell or idx["ksquare"] in cell:
                fixed |= bit
            if idx["deactivator"] in cell:
                deact |= bit
            if idx["switch"] in cell:
                switch |= bit
            if idx["square"] in cell:
                sq |= bit
            if idx["block"] in cell:
                bl |= bit
            if idx["dosquare"] in cell:
                do |= bit
            if idx["player"] in cell:
                player = r * w + c
    if player is None:
        raise ValueError("no player on the board")
    return _Model(h, w, walls, fixed, deact, switch), (player, sq, bl, do)


# ---------------------------------------------------------------------------
# The goal configurations
# ---------------------------------------------------------------------------

def goal_configs(m: _Model, n_towers: int, limit: int = 5_000_000) -> list[int]:
    """Every winning TOWER LAYOUT, as a mask of the cells the towers occupy.

    "Connected and no tower with three neighbours" says the layout is a simple
    path or a simple cycle, INDUCED (a chord would give one of its ends a third
    neighbour). So this is a DFS that grows a path one cell at a time and only
    accepts a cell touching exactly one cell already in the set -- plus a second
    pass that closes cycles, whose last cell touches two (its predecessor and the
    start). Deactivator cells are excluded because a tower on one can never win,
    walls because nothing can stand there.

    Cheap because it is so constrained: the bolted towers pin the layout and the
    shipped boards admit between 1 and 11 layouts each. Each path is found twice,
    once from each end; the result is a set, so that costs only time.
    """
    h, w = m.h, m.w
    allowed = [i for i in range(h * w)
               if not (m.walls >> i) & 1 and not (m.deact >> i) & 1]
    allset = set(allowed)
    adj = {}
    for i in allowed:
        r, c = divmod(i, w)
        adj[i] = [rr * w + cc for rr, cc in
                  ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                  if 0 <= rr < h and 0 <= cc < w and rr * w + cc in allset]
    fixed = {i for i in allowed if (m.fixed >> i) & 1}
    if len(fixed) > n_towers or n_towers > len(allowed):
        return []
    out: set = set()
    budget = [limit]

    def grow(last, inset, close: bool, start: int):
        budget[0] -= 1
        if budget[0] < 0:
            raise RuntimeError("goal enumeration blew its node limit")
        k = len(inset)
        if len(fixed - inset) > n_towers - k:
            return                                    # cannot still cover them
        if k == n_towers:
            if fixed <= inset and (not close or start in adj[last]):
                out.add(frozenset(inset))
            return
        for nb in adj[last]:
            if nb in inset:
                continue
            touch = [x for x in adj[nb] if x in inset]
            if touch != [last]:
                # the only other legal shape is the cell that CLOSES a cycle
                if not (close and k + 1 == n_towers
                        and set(touch) == {last, start}):
                    continue
            inset.add(nb)
            grow(nb, inset, close, start)
            inset.discard(nb)

    for s in allowed:
        grow(s, {s}, False, s)
        grow(s, {s}, True, s)
    return [sum(1 << i for i in cfg) for cfg in out]


# ---------------------------------------------------------------------------
# A* over walk-to-and-push MACROS (for the levels the exhaustive solve cannot
# reach)
# ---------------------------------------------------------------------------

def _assignment(cost):
    """Min-cost perfect assignment (Jonker-Volgenant shortest augmenting path),
    returning ``(total, column_for_each_row)``. Square matrices only, which is
    all this file builds."""
    n = len(cost)
    inf = float("inf")
    u = [0] * (n + 1)
    v = [0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    res = [0] * n
    for j in range(1, n + 1):
        res[p[j] - 1] = j - 1
    return sum(cost[i][res[i]] for i in range(n)), res


class _Macro:
    """A* whose successors are ``walk to the push cell, then push`` MACROS.

    WHY. The exhaustive solve is exact and needs nothing else on eleven of the
    fourteen levels; on the other three the win is 30 to 68 presses away and the
    ball of that radius does not fit in memory. A macro search's depth is the
    number of PUSHES instead (12 to 16 there), and each macro's walk is a
    shortest path found by one BFS rather than by search.

    The cost stays in PRESSES -- ``walk length + 1`` per macro -- so a plan it
    proves shortest is shortest in the units the corpus is scored in, and states
    are deduped on the exact ``(player cell, pieces)`` pair rather than on a
    player REGION (a region merge mis-costs by up to its diameter, which is a
    real loss of optimality; see ps:entrepotphage_demake).
    """

    def __init__(self, m: _Model, goals: list[int]):
        self.m = m
        self.goals = goals
        n = m.h * m.w
        self.nbr = []
        for i in range(n):
            r, c = divmod(i, m.w)
            self.nbr.append([rr * m.w + cc for rr, cc in
                             ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                             if 0 <= rr < m.h and 0 <= cc < m.w])
        self.dist = self._free_distances()
        #: the free cells of each goal layout, i.e. the cells the PUSHABLE
        #: towers have to end on
        self.free_targets = [[t for t in m.cells(g) if not (m.fixed >> t) & 1]
                             for g in goals]

    def _free_distances(self) -> dict:
        """All-pairs BFS distance over the cells nothing static blocks. It
        ignores the movable pieces on purpose: that keeps it a LOWER bound on
        every real walk and every real push route, which is what the heuristic
        needs."""
        free = [i for i in range(self.m.h * self.m.w)
                if not (self.m.solid >> i) & 1]
        out = {}
        for s in free:
            d = {s: 0}
            q = deque([s])
            while q:
                x = q.popleft()
                for y in self.nbr[x]:
                    if y not in d and not (self.m.solid >> y) & 1:
                        d[y] = d[x] + 1
                        q.append(y)
            out[s] = d
        return out

    # -- successors ---------------------------------------------------------
    def walk_field(self, state):
        """BFS distances from the player over cells no piece occupies, with the
        parent pointers the plan expansion needs."""
        m = self.m
        p, sq, bl, do = state
        occ = m.solid | sq | bl | do
        d = {p: 0}
        par = {}
        q = deque([p])
        while q:
            x = q.popleft()
            for y in self.nbr[x]:
                if y not in d and not (occ >> y) & 1:
                    d[y] = d[x] + 1
                    par[y] = x
                    q.append(y)
        return d, par

    def macros(self, state):
        """``[(press cost, next state, (push cell, direction))]``.

        The plan never has to end on anything but a push: the board only ever
        changes on a push, so the press that turns the win on is one -- except on
        a board that is already wired, which the callers special-case.
        """
        m = self.m
        p, sq, bl, do = state
        movers = sq | bl | do
        reach, _par = self.walk_field(state)
        occ = m.solid | movers
        out = []
        for t in m.cells(movers):
            tb = 1 << t
            r, c = divmod(t, m.w)
            for di in range(4):
                dr, dc = _DELTA[KEYS[di]]
                pr, pc = r - dr, c - dc
                r2, c2 = r + dr, c + dc
                if not (0 <= pr < m.h and 0 <= pc < m.w):
                    continue
                if not (0 <= r2 < m.h and 0 <= c2 < m.w):
                    continue
                push_cell = pr * m.w + pc
                dest = r2 * m.w + c2
                if push_cell not in reach or (occ >> dest) & 1:
                    continue
                db = 1 << dest
                nsq, nbl, ndo = sq, bl, do
                if sq & tb:
                    nsq = (sq & ~tb) | db
                elif bl & tb:
                    nbl = (bl & ~tb) | db
                else:
                    ndo = (do & ~tb) | db
                out.append((reach[push_cell] + 1, (t, nsq, nbl, ndo),
                            (push_cell, di)))
        return out

    # -- heuristic ----------------------------------------------------------
    def h(self, state) -> int:
        """An admissible lower bound on the presses still to spend, as the max of
        two terms, minimised over the level's goal layouts.

        The ASSIGNMENT term costs each pushable tower's move to its goal cell in
        pushes, and every push is a press. It is what a sokoban heuristic
        usually is, and on these boards it is badly loose: three quarters of the
        cost is walking.

        The MST term is what charges for that. Pushing a tower puts the player in
        the cell the tower vacated, so the player's trajectory is one connected
        walk through, for every tower that must move, BOTH the cell it stands on
        and the cell it is pushed from (that one is only known when the tower's
        move is a single cell, which is when the direction is forced). A walk
        visiting a set of points is at least the MST of their metric closure, and
        the closure uses `dist`, which ignores movable obstacles and is therefore
        a lower bound on the real thing.
        """
        m = self.m
        p, sq, bl, do = state
        if do:
            return min((self.dist[t].get(s, 1 << 20)
                        for t in m.cells(do) for s in m.cells(m.switch)),
                       default=0)
        pieces = m.cells(sq)
        best = 1 << 30
        for cheap, gi, gmask in self._ranked(sq, pieces):
            if cheap >= best:
                # `cheap` is a lower bound on this layout's assignment term and
                # the list is sorted by it, so nothing left can beat `best`.
                # That is the whole reason for the two-stage evaluation: the
                # bonus level admits eleven layouts and the exact terms cost
                # ~15x the cheap one, so a level with many layouts would
                # otherwise spend the search's whole budget in the heuristic.
                break
            targets = [t for t in self.free_targets[gi] if not (sq >> t) & 1]
            nodes = [p]
            cost = 0
            if targets:
                movers = [x for x in pieces if not (gmask >> x) & 1]
                if len(movers) != len(targets):
                    continue                     # not this layout's tower count
                table = [[self.dist[x].get(t, 1 << 16) for t in targets]
                         for x in movers]
                cost, pick = _assignment(table)
                for k, x in enumerate(movers):
                    nodes.append(x)
                    t = targets[pick[k]]
                    if self.dist[x].get(t, 1 << 16) != 1:
                        continue
                    rx, cx = divmod(x, m.w)
                    rt, ct = divmod(t, m.w)
                    rs, cs = 2 * rx - rt, 2 * cx - ct
                    s = rs * m.w + cs
                    if 0 <= rs < m.h and 0 <= cs < m.w and not (m.solid >> s) & 1:
                        nodes.append(s)
            for x in m.cells(bl & gmask):
                nodes.append(x)
                cost += 1                       # a block on a goal cell must move
            v = max(cost, self._mst(nodes))
            if v < best:
                best = v
        return 0 if best == 1 << 30 else best

    def _ranked(self, sq: int, pieces):
        """``[(cheap lower bound, layout index, layout mask)]``, ascending.

        ``cheap`` is each unplaced tower's distance to its nearest free goal
        cell, summed -- a lower bound on the min-cost assignment, since every row
        of the assignment costs at least its own minimum. So `h` can evaluate
        layouts in this order and stop as soon as the bound reaches the best full
        value it already has.

        Skipped when the level has one layout, which is eleven of the fourteen:
        there is nothing to rank and the pre-pass would be pure overhead.
        """
        if len(self.goals) < 2:
            return [(0, gi, g) for gi, g in enumerate(self.goals)]
        out = []
        for gi, gmask in enumerate(self.goals):
            targets = [t for t in self.free_targets[gi] if not (sq >> t) & 1]
            if not targets:
                out.append((0, gi, gmask))
                continue
            movers = [x for x in pieces if not (gmask >> x) & 1]
            if not movers:
                continue
            out.append((sum(min(self.dist[x].get(t, 1 << 16) for t in targets)
                            for x in movers), gi, gmask))
        out.sort(key=lambda x: x[0])
        return out

    def _mst(self, nodes) -> int:
        if len(nodes) < 2:
            return 0
        rest = list(dict.fromkeys(nodes[1:]))
        if not rest:
            return 0
        near = {x: self.dist[nodes[0]].get(x, 1 << 16) for x in rest}
        total = 0
        while rest:
            x = min(rest, key=near.__getitem__)
            total += near[x]
            rest.remove(x)
            row = self.dist[x]
            for y in rest:
                d = row.get(y, 1 << 16)
                if d < near[y]:
                    near[y] = d
        return total

    # -- search -------------------------------------------------------------
    def beam(self, start, width: int, depth: int = 80, deadline=None):
        """``(press cost, macro path)`` from a layered beam over macros, ordered
        by PROGRESS (the heuristic) with the cost as a tie-break.

        This is a probe, not the answer: its only job is to hand `solve` an
        upper bound so its pruning has something to bite on. Ordering it by the
        admissible heuristic rather than by ``g + h`` is deliberate -- on the two
        levels that need it, h is loose by roughly a factor of two over the whole
        middle of the board, so an f-ordered search has nothing to steer with
        (see ps:idols_to_the_burnt_god, same shape).

        Width is not monotone in answer quality, which is why the caller tries
        several and keeps the best: measured on level 7, width 500 returns 46 and
        widths 3000 and 12000 both return 54. A wider beam reaches a win at a
        shallower PUSH depth, and pushes are not what these plans are made of.
        """
        m = self.m
        layer = [(0, start, ())]
        seen = {start}
        best = None
        for _ in range(depth):
            cand = []
            for g, state, path in layer:
                for cost, nxt, tag in self.macros(state):
                    ng = g + cost
                    if m.won(nxt):
                        if best is None or ng < best[0]:
                            best = (ng, path + (tag,))
                        continue
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    cand.append((self.h(nxt) * 4 + ng, ng, nxt, path + (tag,)))
            if best is not None or not cand:
                return best
            if deadline is not None and time.monotonic() > deadline:
                return None
            cand.sort(key=lambda x: x[0])
            layer = [(g, s, p) for _k, g, s, p in cand[:width]]
        return best

    def solve(self, start, cap: int, weight: int = 1, bound=None,
              deadline=None):
        """``(press cost, macro path, closed)`` -- the best plan STRICTLY cheaper
        than ``bound``, or ``(None, None, closed)`` if there is none.

        A winning macro is kept as an INCUMBENT rather than returned, and the
        search only stops when the frontier's f reaches its cost -- macros cost
        different amounts, so the first win GENERATED is routinely not the
        cheapest one (ps:esl_puzzle_game found that the hard way).

        ``closed`` means the frontier really was exhausted (or reached the
        incumbent). At ``weight == 1`` with an admissible h that is the
        optimality proof, and it is a proof either way: with a path, that path is
        shortest; without one, nothing beats ``bound``. A run that hits ``cap``
        or ``deadline`` returns what it has with ``closed`` False.

        ``deadline`` (a `time.monotonic` value) is the budget that matters. The
        node cap bounds MEMORY -- every queued node holds a macro list -- but
        the cost per node varies by two orders of magnitude between a level with
        one goal layout and one with eleven, so a node budget alone means either
        proving nothing on the small levels or a forty-minute search on the big
        ones.
        """
        m = self.m
        counter = 0
        pq = [(weight * self.h(start), 0, counter, start, ())]
        best = {start: 0}
        inc = bound
        inc_path = None
        nodes = 0
        while pq:
            f, g, _c, state, path = heapq.heappop(pq)
            if inc is not None and f >= inc:
                return inc if inc_path else None, inc_path, True
            if best.get(state, 1 << 30) < g:
                continue
            for cost, nxt, tag in self.macros(state):
                nodes += 1
                ng = g + cost
                if inc is not None and ng >= inc:
                    continue
                if m.won(nxt):
                    inc, inc_path = ng, path + (tag,)
                    continue
                if best.get(nxt, 1 << 30) <= ng:
                    continue
                hh = self.h(nxt)
                if inc is not None and ng + hh >= inc:
                    continue
                best[nxt] = ng
                counter += 1
                heapq.heappush(pq, (ng + weight * hh, ng, counter, nxt,
                                    path + (tag,)))
            if nodes >= cap or (deadline is not None
                                and time.monotonic() > deadline):
                return (inc if inc_path else None), inc_path, False
        return (inc if inc_path else None), inc_path, True

    # -- plan expansion ------------------------------------------------------
    def expand(self, start, macro_path):
        """Flatten a macro path to primitive presses, labelling each with the
        presses that PROVABLY tie with it.

        No rule in this game fires on a bare move, so two walks of equal length
        to the same push cell leave identical boards and identical remaining
        cost. Every press that keeps the player on a shortest route to the push
        cell of the macro being served therefore ties with the one taken,
        whatever the plan's own optimality. A push is labelled with itself.
        """
        m = self.m
        state = start
        presses: list[str] = []
        optsets: list[list[str]] = []
        for push_cell, di in macro_path:
            reach, par = self.walk_field(state)
            if push_cell not in reach:
                raise ValueError("macro plan walks to an unreachable cell")
            route = []
            x = push_cell
            while x != state[0]:
                route.append(x)
                x = par[x]
            route.reverse()
            # distance-to-push-cell field over the same walkable set
            back = self._to_field(state, push_cell)
            for nxt_cell in route:
                cur = state[0]
                di_walk = self._dir_between(cur, nxt_cell)
                alts = [KEYS[d] for d in range(4)
                        if self._walk_target(state, d) is not None
                        and back.get(self._walk_target(state, d), 1 << 20)
                        == back[cur] - 1]
                presses.append(KEYS[di_walk])
                optsets.append(alts if KEYS[di_walk] in alts else [KEYS[di_walk]])
                state = m.step(state, di_walk)
                if state[0] != nxt_cell:
                    raise ValueError("walk step did not land where planned")
            presses.append(KEYS[di])
            optsets.append([KEYS[di]])
            state = m.step(state, di)
        if not m.won(state):
            raise ValueError("expanded macro plan does not win")
        return presses, optsets, state

    def exact_optsets(self, start, presses, deadline=None):
        """The TRUE optimal set at every step of a plan already PROVED shortest,
        or None if the measurement runs out of time.

        Once ``d*`` is known, "is press ``d`` optimal here" is a decision with a
        tight bound: re-solve from its successor asking only for a plan strictly
        cheaper than the ``r`` presses that remain, and it is optimal exactly
        when one of ``r - 1`` comes back. The tightness is what makes this
        affordable at all -- an unbounded re-solve from each of a plan's
        successors is the search that could not be finished in the first place.

        This is the same measurement `_Model.solve`'s distance field gives away
        for free on the levels it can reach; it exists because the macro-planned
        levels have no field. It is only ever run when the plan is proved, since
        without a true ``d*`` "one shorter than the remaining plan" means nothing.
        """
        m = self.m
        state = start
        out = []
        for i, press in enumerate(presses):
            r = len(presses) - i
            best = []
            for di, key in enumerate(KEYS):
                nxt = m.step(state, di)
                if m.won(nxt):
                    if r == 1:
                        best.append(key)
                    continue
                if r == 1 or nxt == state:
                    continue
                if deadline is not None and time.monotonic() > deadline:
                    return None
                cost, _p, closed = self.solve(nxt, 1 << 30, weight=1, bound=r,
                                              deadline=deadline)
                if cost is None and not closed:
                    return None          # ran out of budget: proves nothing
                if cost == r - 1:
                    best.append(key)
            if press not in best:
                raise ValueError(f"step {i} of a proved plan measures as "
                                 f"non-optimal")
            out.append(best)
            state = m.step(state, KEYS.index(press))
        return out

    def _walk_target(self, state, di: int):
        """The cell a bare (non-pushing) move in ``di`` would reach, or None."""
        m = self.m
        p, sq, bl, do = state
        r, c = divmod(p, m.w)
        dr, dc = _DELTA[KEYS[di]]
        r2, c2 = r + dr, c + dc
        if not (0 <= r2 < m.h and 0 <= c2 < m.w):
            return None
        t = r2 * m.w + c2
        if (m.solid | sq | bl | do) >> t & 1:
            return None
        return t

    def _to_field(self, state, target: int) -> dict:
        m = self.m
        occ = m.solid | state[1] | state[2] | state[3]
        d = {target: 0}
        q = deque([target])
        while q:
            x = q.popleft()
            for y in self.nbr[x]:
                if y not in d and not (occ >> y) & 1:
                    d[y] = d[x] + 1
                    q.append(y)
        return d

    def _dir_between(self, a: int, b: int) -> int:
        ra, ca = divmod(a, self.m.w)
        rb, cb = divmod(b, self.m.w)
        for di, (dr, dc) in enumerate(_DELTA[k] for k in KEYS[:4]):
            if (ra + dr, ca + dc) == (rb, cb):
                return di
        raise ValueError("cells are not adjacent")


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class MatchFlowExpert(PSExpert):
    """Exhaustive-first expert: the layered BFS + reverse BFS field when the
    level fits in ``state_cap`` states, the macro A* otherwise.

    It plugs into the shared harness at `PSExpert._search`, so the plan memo,
    the on-disk cache with its staleness check, the snapshot/restore discipline
    and the level scoping all come from the base. `heuristic` and `dead` are
    never called -- `_astar` is not the strategy here.
    """

    directions = list(KEYS)

    #: MUST be on, for a reason the base class does not anticipate. `PSExpert._key`
    #: is the full set of non-background cells, which is documented as an exact
    #: canonical state -- but it does not carry the board's DIMENSIONS, and this
    #: game ships two levels that differ only by a trailing blank row ("Oh no, I'm
    #: trapped!" and its epilogue twin). Their keys are byte-identical, so without
    #: this the second one silently served the first one's plan. It happens to be
    #: the right plan here, since the boards really are equivalent for every
    #: object on them -- which is exactly why it went unnoticed until the report
    #: showed one of them with no report row at all. `_Model` reads `eng.height`
    #: and `eng.width`, so any pair of levels padded differently would alias.
    scope_by_level = True

    #: Where the level START plans live between processes. The searches are the
    #: whole cost of generation and are seed-independent, so without this every
    #: `parallelize_generator` shard would re-derive all fourteen.
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "match_flow_plans.json")

    #: The exhaustive search's budget, in generated states. Sized from MEMORY,
    #: not patience -- the search keeps every state's successor dict so the
    #: reverse pass can run over them -- and there is nothing in between to tune
    #: for: the deepest level it solves visits 293k states, and the three it
    #: cannot are past ten million (level 8's layers were measured out to 4.8M
    #: at depth 26 and its win is at 30). So this is "293k plus headroom", which
    #: also makes GIVING UP cheap, which is what it does three times a run.
    state_cap: int = 450_000

    #: Generated macros the A* may queue. This is a MEMORY budget -- each node
    #: carries its macro list -- and the wall clock below is what actually stops
    #: the searches that cannot finish.
    macro_cap: int = 2_000_000

    #: Wall-clock budget for one level's macro phase, in seconds. Level 8 proves
    #: in about ten; levels 7 and 11 cannot be proved at any budget this side of
    #: a better heuristic (see the module docstring), so the number is chosen to
    #: keep a COLD generation run to a few minutes rather than to chase them.
    #: Everything it produces is written to `plan_cache_path`, so it is paid once.
    macro_seconds: float = 90.0

    #: Beam widths tried before the A* passes, best answer kept -- width is not
    #: monotone in answer quality (see `_Macro.beam`) -- inside `beam_seconds`.
    #: The budget matters because a width that finds nothing still costs its full
    #: sweep: on the bonus level, width 500 spends a minute to return nothing and
    #: width 3000 then finds a plan.
    beam_widths = (500, 3000)
    beam_seconds: float = 240.0

    #: The weighted rung, and the share of the budget it gets. It is there to
    #: pull a loose beam bound down cheaply before the exact pass has to prune
    #: against it; the exact pass gets the rest.
    probe_weight: int = 3
    probe_share: float = 0.25

    #: Wall-clock budget for MEASURING the optimal sets of a macro plan that was
    #: proved shortest (`_Macro.exact_optsets`). Level 8 -- the only level that
    #: gets there -- needs well under this; over budget the plan keeps the walk
    #: labels, which are a sound subset.
    tie_seconds: float = 240.0

    def setup(self) -> None:
        self._state_ids = {self.g.obj_name_to_idx[n] for n in _STATE_OBJECTS}

    def _key(self, eng) -> frozenset:
        """The state key, narrowed to the object classes that DEFINE a state.

        `PSExpert._key` keys on every non-background cell, which is the right
        default and is WRONG here, in a way that costs nothing visible and five
        minutes a seed. This game's `late` rules rebuild `connected`,
        `connectedl/r/u/d`, `overload` and `winning` from the board every press --
        and the rule that seeds the connectivity flood is ``late random``, so
        which tower carries the (transparent) `connected` marker is drawn from
        the global RNG. `run_rules_on_level_start` runs those rules on every
        `set_level`, so the level's OWN START STATE hashes differently depending
        on how much RNG the episode has consumed so far.

        The effect: seed 0 hit the on-disk plan cache for all fourteen levels and
        every later seed missed it for all of them, re-deriving level 11's
        four-minute search once per episode -- silently, since the plans it found
        were the same. Measured before the fix at 345 s for seed 1 and after it at
        2 s. This is the "if it looks linear in seeds, the plan cache is missing"
        failure the shared module's docstring warns about, reached through the
        state key instead of through the rotation contract.

        Everything dropped here is a pure function of what is kept, so the
        narrowed key is still exact. Narrowing is also what makes
        ``scope_by_level`` mandatory rather than merely advisable.
        """
        ids = self._state_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in cell
            if o in ids
        )

    def __init__(self, *a, **kw):
        #: level -> (plan length, proved shortest, goal-layout count, seconds,
        #: label kind). The label kind is how the optimal sets were arrived at:
        #: ``field`` off the exhaustive distance field, ``measured`` by
        #: re-solving each press against a proved d*, ``walk`` from what a walk
        #: provably ties with. `_replay_check` verifies each differently, so
        #: this is not decoration -- see its docstring.
        self.report: dict[int, tuple] = {}
        super().__init__(*a, **kw)

    def _search(self, eng):
        model, state = model_from_engine(eng, self.g)
        t0 = time.time()
        presses, optsets = model.solve(state, self.state_cap)
        if presses is not None:
            self.report[getattr(self, "_level", -1)] = (
                len(presses), True, 0, time.time() - t0, "field")
            return Plan(presses, optsets)
        return self._macro_search(model, state, t0)

    def _macro_search(self, model, state, t0):
        """Probe, then bound, then prove -- the shape to reach for when a level
        is too deep for the exhaustive field but a PROOF is still wanted.

        A beam finds *a* win and its cost is used only as an upper bound; a
        weighted rung tries to improve it cheaply; the ``weight == 1`` pass then
        either improves it again or CLOSES, and a bounded run that closes with
        nothing shorter IS the optimality proof. Strictly better than a weight
        ladder that keeps its last answer: this one keeps optimality when it can
        get it and says so when it cannot.
        """
        n_towers = bin(model.fixed).count("1") + bin(state[1]).count("1")
        goals = [] if state[3] else goal_configs(model, n_towers)
        if not goals and not state[3]:
            return None                          # no layout of towers ever wins
        mac = _Macro(model, goals)
        bound = None
        path = None
        beam_end = time.monotonic() + self.beam_seconds
        for width in self.beam_widths:
            probe = mac.beam(state, width, deadline=beam_end)
            if probe is not None and (bound is None or probe[0] < bound):
                bound, path = probe
            if time.monotonic() > beam_end:
                break
        end = time.monotonic() + self.macro_seconds
        cost, p, _closed = mac.solve(
            state, self.macro_cap, weight=self.probe_weight, bound=bound,
            deadline=time.monotonic() + self.macro_seconds * self.probe_share)
        if p is not None:
            bound, path = cost, p
        cost, p, closed = mac.solve(state, self.macro_cap, weight=1,
                                    bound=bound, deadline=end)
        if p is not None:
            bound, path = cost, p
        if path is None:
            return None
        presses, optsets, _end = mac.expand(state, path)
        labels = "walk"
        if closed:
            # d* is known, so the labels can be MEASURED instead of derived from
            # what a walk provably ties with. Falls back to the walk labels (a
            # sound subset) if the measurement runs out of budget.
            exact = mac.exact_optsets(
                state, presses,
                deadline=time.monotonic() + self.tie_seconds)
            if exact is not None:
                optsets, labels = exact, "measured"
        self.report[getattr(self, "_level", -1)] = (
            len(presses), closed, len(goals), time.time() - t0, labels)
        return Plan(presses, optsets)

    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan` plus the per-level report row, carried THROUGH the
        on-disk plan cache.

        Without this the "is this plan proved shortest" column would read `NO`
        for every level as soon as `data/match_flow_plans.json` exists, because
        a cache hit never runs `_search` -- i.e. the report would quietly start
        lying about the corpus the moment it got fast. The base only ever reads
        ``start`` / ``plan`` / ``optsets`` out of an entry, so an extra key rides
        along harmlessly.
        """
        self._level = level
        found = super().plan(eng, level)
        entry = self._disk.get(level) if level is not None else None
        if entry is not None:
            if level in self.report:
                row = list(self.report[level])
                if entry.get("report") != row:
                    entry["report"] = row
                    self._save_disk()
            elif len(entry.get("report") or ()) == 5:
                # length-checked, not just present: a cache written by an older
                # build of this file carries a shorter row, and the reports index
                # it positionally.
                self.report[level] = tuple(entry["report"])
        return found


class MatchFlowSolver(PSAStarSolver):
    game_id = "puzzlescript_match_flow"
    game_name = GAME_NAME
    expert_cls = MatchFlowExpert
    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so the generator can never tape frames from a different
    #: build of the game than the agent plays.
    game_module_id = GAME_MODULE_ID

    #: The longest plan is 68 presses; the rest is room for the exploration
    #: prefix and the replay after its RESET. Inside the adapter's own 200-step
    #: per-level budget.
    max_steps = 150


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    solver = MatchFlowSolver()
    game = solver.make_game(0)
    return solver, game, MatchFlowExpert(game)


def _plan_report() -> int:
    """Per-level size, piece counts, goal-layout count, plan length, whether the
    plan is PROVED shortest, and how many of its steps carry a tie.

    This is the table in the module docstring; re-run it after touching a level
    or a search. It also measures the one claim the search list rests on that is
    easy to state and easy to get wrong -- that ACTION5 never changes a board --
    rather than assuming it.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    total = ties_total = 0
    unproved = []
    print(f"{'lvl':>3} {'size':>5} {'twr':>4} {'blk':>3} {'goals':>5} "
          f"{'plan':>4} {'proved':>6} {'labels':>8} {'ties':>4}  seconds")
    for level in range(game.n_levels):
        game.set_level(level)
        model, state = model_from_engine(eng, game._game)
        plan = expert.plan(eng, level)
        n_fixed = bin(model.fixed).count("1")
        n_push = bin(state[1]).count("1") + bin(state[3]).count("1")
        if plan is None:
            print(f"{level:3d} {eng.height}x{eng.width} -- NO PLAN")
            unproved.append(level)
            continue
        _len, proved, goals, secs, labels = expert.report.get(
            level, (len(plan), False, 0, 0.0, "?"))
        ties = sum(1 for s in plan.optsets if len(s) > 1)
        total += len(plan)
        ties_total += ties
        if not proved:
            unproved.append(level)
        room = "" if len(plan) < game._max_steps else "  OVER BUDGET"
        print(f"{level:3d} {eng.height}x{eng.width:<3d} {n_fixed + n_push:4d} "
              f"{bin(state[2]).count('1'):3d} {goals or '-':>5} {len(plan):4d} "
              f"{'yes' if proved else 'NO':>6} {labels:>8} {ties:4d}  "
              f"{secs:6.1f}{room}")
    print(f"total {total} presses, {ties_total} steps with a tie "
          f"({ties_total / max(1, total):.0%}); "
          f"{'every plan proved shortest' if not unproved else f'not proved shortest: {unproved}'}")

    sweep = 200_000
    inert = states = 0
    partial = []
    for level in range(game.n_levels):
        game.set_level(level)
        model, state = model_from_engine(eng, game._game)
        seen, capped = model.reachable(state, sweep)
        states += len(seen)
        if capped:
            partial.append(level)
        for s in seen:
            if model.step(s, _ACTION) != s:
                inert += 1
    print(f"over {states} reachable states"
          f"{f' (levels {partial} stopped at the {sweep} cap)' if partial else ''}: "
          f"ACTION5 changes {inert} of them")
    return 1 if inert else 0


# ---------------------------------------------------------------------------
# Model self-check (fuzz against the real interpreter)
# ---------------------------------------------------------------------------

def _read(eng, g):
    return model_from_engine(eng, g)[1]


def _overloads(model: _Model, state) -> int:
    """The cells the interpreter should be drawing an `overload` pip on: a tower
    off a deactivator with three or more tower neighbours that are also off one.

    This is the "no branching" half of `_Model.won` computed a second way, and
    the interpreter renders it as an object, so `selfcheck` gets an independent
    read on the predicate instead of only on the boolean it produces."""
    live = (model.fixed | state[1] | state[3]) & ~model.deact
    n = [model.sh(live, d) for d in range(4)]
    out = 0
    for a in range(4):
        for b in range(a + 1, 4):
            for c in range(b + 1, 4):
                out |= n[a] & n[b] & n[c]
    return out & live


def selfcheck(trials: int = 30, steps: int = 120,
              boards: int = 500, board_steps: int = 30) -> int:
    """Audit the claim both solvers rest on: `_Model` reproduces the interpreter
    EXACTLY.

    Three things have to agree on every press, and they are the three the
    solvers use: the settled board (which covers the refusal decision in both
    directions -- a press the model calls a no-op that moved a piece, or a push
    the interpreter did not make, is a board mismatch), the win flag, and the
    OVERLOAD markers, which are the "no tower with three neighbours" half of the
    win predicate rendered as objects and are therefore a free independent read
    on it.

    An earlier version also required a refused press to leave the interpreter's
    grid byte-identical. That is wrong and the random boards convicted it: the
    connection art, the `winning` token and the overload pips are DERIVED objects
    that the late rules recompute from scratch every press, so on a board that
    was hand-built rather than reached by play the first press draws all of them
    and the grid changes while nothing moved. Comparing the tracked classes is
    the check that was meant; comparing the whole grid was comparing the
    interpreter's own scratch space.

    Two fuzz populations, and the second is the one with teeth. The shipped
    levels exercise the game as it is played. The random boards exercise what no
    shipped level can reach: a `dosquare` next to a `deactivator`, a tower pushed
    ONTO a deactivator, chains of three and four towers (the shipped boards have
    few), branching junctions, and boards where the towers are already wired so
    the very first press wins. Fuzzing only the shipped levels would leave the
    whole `won` predicate tested on a handful of layouts -- and an exploration
    prefix reaches all of them.
    """
    game = _load_game_module(GAME_MODULE_ID).make_game(seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    tot = {"presses": 0, "refused": 0, "pushes": 0, "wins": 0,
           "deact_hits": 0, "overloads": 0, "already_won": 0, "chains": 0}
    bad = 0

    def rollout(tag: str, nsteps: int, rng: random.Random) -> int:
        nonlocal bad
        model, state = model_from_engine(eng, g)
        if model.won(state):
            tot["already_won"] += 1
        for _ in range(nsteps):
            live = [di for di in range(len(KEYS)) if model.step(state, di) != state]
            # Mostly play on, occasionally press blind: a uniform walk spends most
            # of a rollout shoving the player into the same wall, and the presses
            # that move a tower are where the mechanic lives. Not circular -- a
            # wrong legality claim still has to survive the grid comparison.
            di = (rng.choice(live) if live and rng.random() < 0.8
                  else rng.randrange(len(KEYS)))
            if di != _ACTION:                        # coverage: chain refusals
                p = state[0]
                dr, dc = _DELTA[KEYS[di]]
                r, c = divmod(p, model.w)
                movers = state[1] | state[2] | state[3]
                cs = [(r + dr * k, c + dc * k) for k in (1, 2)]
                if all(0 <= rr < model.h and 0 <= cc < model.w and
                       (movers >> (rr * model.w + cc)) & 1 for rr, cc in cs):
                    tot["chains"] += 1
            predicted = model.step(state, di)
            eng.step(KEYS[di])
            tot["presses"] += 1
            actual = _read(eng, g)
            if actual != predicted:
                bad += 1
                print(f"  {tag}: board divergence on {KEYS[di]}")
                return 1
            if eng.check_win() != model.won(predicted):
                bad += 1
                print(f"  {tag}: win divergence on {KEYS[di]} "
                      f"(engine {eng.check_win()}, model {model.won(predicted)})")
                return 1
            over = _overloads(model, predicted)
            drawn = sum(1 << (r * model.w + c)
                        for r in range(model.h) for c in range(model.w)
                        if idx["overload"] in eng.grid[r][c])
            if over != drawn:
                bad += 1
                print(f"  {tag}: overload divergence on {KEYS[di]}")
                return 1
            tot["refused"] += predicted == state
            tot["pushes"] += predicted[1:] != state[1:]
            allsq = model.fixed | predicted[1] | predicted[3]
            tot["deact_hits"] += bool(allsq & model.deact)
            tot["overloads"] += bool(over)
            state = predicted
            if eng.check_win():
                tot["wins"] += 1
                return 0
        return 0

    for level in range(game.n_levels):
        for t in range(trials):
            game.set_level(level)
            rollout(f"L{level}/{t}", steps,
                    random.Random(f"matchflow:shipped:{level}:{t}"))

    for t in range(boards):
        rng = random.Random(f"matchflow:board:{t}")
        h, w = rng.randint(4, 8), rng.randint(4, 8)
        game.set_level(0)
        eng.height, eng.width = h, w
        eng.grid = [[{idx["background"]} for _ in range(w)] for _ in range(h)]
        cells = [(r, c) for r in range(h) for c in range(w)]
        rng.shuffle(cells)
        pos = 0

        def take(k):
            nonlocal pos
            out = cells[pos:pos + k]
            pos += k
            return out

        for (r, c) in take(rng.randint(0, len(cells) // 6)):
            eng.grid[r][c].add(idx[rng.choice(("vwall", "iwall"))])
        for (r, c) in take(rng.randint(0, 4)):
            eng.grid[r][c].add(idx[rng.choice(("fsquare", "ksquare"))])
        for (r, c) in take(rng.randint(0, 5)):
            eng.grid[r][c].add(idx["square"])
        for (r, c) in take(rng.randint(0, 2)):
            eng.grid[r][c].add(idx["block"])
        if rng.random() < 0.25:
            for (r, c) in take(1):
                eng.grid[r][c].add(idx["dosquare"])
        for (r, c) in take(1):
            eng.grid[r][c].add(idx["player"])
        # floor decorations go anywhere, including under a piece
        for _ in range(rng.randint(0, 4)):
            r, c = rng.randrange(h), rng.randrange(w)
            eng.grid[r][c].add(idx["deactivator"])
        for _ in range(rng.randint(0, 2)):
            r, c = rng.randrange(h), rng.randrange(w)
            eng.grid[r][c].add(idx["switch"])
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        eng._rule_win = False
        if any(idx["player"] in cell for row in eng.grid for cell in row):
            rollout(f"board{t}", board_steps, rng)

    print(f"selfcheck: {tot['presses']} presses "
          f"({game.n_levels} levels x {trials} rollouts + {boards} random "
          f"boards), {tot['refused']} refused, {tot['pushes']} pushes, "
          f"{tot['wins']} wins, {tot['already_won']} boards already wired at "
          f"reset, {tot['deact_hits']} states with a tower on a deactivator, "
          f"{tot['overloads']} with a branching tower, {tot['chains']} presses "
          f"into a chain of two pieces")
    if not (tot["deact_hits"] and tot["overloads"] and tot["wins"]
            and tot["chains"]):
        print("FUZZ TOO WEAK: it never reached a deactivated tower, a branching "
              "tower, a two-piece chain or a win")
        return 1
    return bad


def _replay_check() -> int:
    """Every level's plan, replayed through the interpreter, must WIN -- and every
    press the labels call optimal has to be defensible.

    What "defensible" means depends on how the labels were arrived at (the
    report's ``labels`` column), and that is the point rather than a shortcut:

      * ``field`` -- re-solve INDEPENDENTLY from each labelled press with the
        exhaustive solver, which must leave exactly the remaining number of
        presses. The labels came off the distance field; this re-derives them a
        different way.
      * ``measured`` -- the labels ARE a measurement against a proved d*
        (`_Macro.exact_optsets`), so re-running it here would only repeat it.
        Checked for self-consistency and left alone.
      * ``walk`` -- check exactly what these labels claim and nothing more: that
        each alternative leaves the pieces byte-identical to the taken press,
        which with the equal-distance construction in `_Macro.expand` is what
        makes it tie. A stronger check would need a d* the macro search could not
        prove for those levels (see the module docstring).
    """
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        model, state = model_from_engine(eng, game._game)
        if plan is None:
            bad += 1
            print(f"  L{level}: no plan")
            continue
        labels = expert.report.get(level, (0, False, 0, 0.0, "?"))[4]
        for i, press in enumerate(plan):
            remaining = len(plan) - i - 1
            taken = model.step(state, KEYS.index(press))
            if press not in plan.optsets[i]:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            for alt in plan.optsets[i]:
                nxt = model.step(state, KEYS.index(alt))
                if labels == "measured":
                    continue
                if labels != "field":
                    if nxt[1:] != taken[1:]:
                        bad += 1
                        print(f"  L{level}: step {i} labels {alt} a walk tie, "
                              f"but it moves a piece")
                    continue
                if model.won(nxt):
                    if remaining:
                        bad += 1
                        print(f"  L{level}: step {i} labels {alt} optimal, but "
                              f"it wins with {remaining} presses left")
                    continue
                sub, _ = model.solve(nxt, expert.state_cap)
                if sub is None or len(sub) != remaining:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but it "
                          f"leaves {len(sub) if sub else 'no'} presses against "
                          f"{remaining}")
            state = taken
            eng.step(press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
    print(f"replay check: {bad} problems")
    return bad


# ---------------------------------------------------------------------------
# The symmetry group
# ---------------------------------------------------------------------------

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

    dmap = {"action": "action"}
    for name, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[name] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    return cell, dims, dmap


#: The classes `--symmetry` compares. The four `connected*` markers are left out
#: on purpose: they are PRESENTATION, and a turned board is supposed to draw the
#: same connection with a turned marker (`connectedu` art becomes `connectedr`
#: art), so requiring them to match would be requiring the wrong thing.
_SYM_CLASSES = _TOWERS + ("block", "player")


def _pieces(eng, g) -> tuple:
    idx = g.obj_name_to_idx
    return tuple(frozenset((r, c) for r in range(eng.height)
                           for c in range(eng.width) if idx[o] in eng.grid[r][c])
                 for o in _SYM_CLASSES)


def _chirality_probe(game, expert, rollouts: int = 10, steps: int = 40) -> int:
    """The second half of `_symmetry`, and the half with teeth.

    Replaying a level's PLAN under the eight transforms cannot find a mechanic
    the plan does not happen to use, and an interpreter-order defect is exactly
    that kind: on ps:lovendpieces, whose absorption rule had the same unlooped
    `late` bug this game's flood had, stage 1 reported SYMMETRIC on the broken
    build and this stage is what convicted it.

    So this stage does not follow a plan. It random-walks each level, records
    every ``(board, press)`` it visits, and re-plays each one INDEPENDENTLY on
    all seven other presentations of that same board. A single order-sensitive
    transition anywhere is a failure.

    `load_level` re-runs the level-start rules, which is what re-derives the
    connection art for the turned board -- and why only the piece classes are
    compared.
    """
    eng, g = game._engine, game._game
    bad = tot = 0
    for level in range(game.n_levels):
        game.set_level(level)
        if expert.plan(eng, level) is None:
            continue
        hw = (eng.height, eng.width)
        triples = []
        for t in range(rollouts):
            game.set_level(level)
            rng = random.Random(f"matchflow:chiral:{level}:{t}")
            model, state = model_from_engine(eng, g)
            for _ in range(steps):
                live = [di for di in range(len(KEYS))
                        if model.step(state, di) != state]
                di = (rng.choice(live) if live and rng.random() < 0.8
                      else rng.randrange(len(KEYS)))
                before = [[set(cell) for cell in row] for row in eng.grid]
                eng.step(KEYS[di])
                triples.append((before, KEYS[di], _pieces(eng, g),
                                eng.check_win()))
                state = model.step(state, di)

        notes = []
        for k, mirror in itertools.product(range(4), (False, True)):
            if (k, mirror) == (0, False):
                continue
            cell, dims, dmap = _transform(k, mirror)
            th, tw = dims(hw)
            for i, (before, press, after, win) in enumerate(triples):
                turned = [[set() for _ in range(tw)] for _ in range(th)]
                for r, row in enumerate(before):
                    for c, objs in enumerate(row):
                        tr, tc = cell((r, c), hw)
                        turned[tr][tc] = set(objs)
                eng.load_level(turned)
                eng.step(dmap[press])
                tot += 1
                want = tuple(frozenset(cell(x, hw) for x in s) for s in after)
                if _pieces(eng, g) != want or eng.check_win() != win:
                    notes.append(f"rot{k}{'m' if mirror else ''}@{i}")
                    break
        bad += len(notes)
        print(f"level {level:2d}: {len(triples):4d} random presses x 7 "
              f"presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print(f"chirality probe: {tot} transformed presses checked, {bad} chiral")
    return bad


def _symmetry() -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an exact
    symmetry of the mechanic, which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation.

    Reading the rules and declaring them direction-free is exactly the argument
    that has been wrong in other ps: games, which is why this measures instead.
    Stage 1 replays each level's plan on all eight turned and mirrored copies of
    its own board, built by transforming the LEVEL LAYOUT so the interpreter
    re-loads it itself. Stage 2 is `_chirality_probe`.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level:2d}: no plan -- skipped")
            continue
        layout = g.levels[level]
        hw = (len(layout), len(layout[0]))

        eng.load_level(layout)
        ref = []
        for direction in plan:
            eng.step(direction)
            ref.append(_pieces(eng, g))

        notes = []
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
            for i, direction in enumerate(plan):
                eng.step(dmap[direction])
                want = tuple(frozenset(cell(x, hw) for x in s) for s in ref[i])
                if _pieces(eng, g) != want:
                    notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                    break
            else:
                if not eng.check_win():
                    notes.append(f"rot{k}{'m' if mirror else ''}: no win")
        bad += len(notes)
        print(f"level {level:2d}: {len(plan):3d} plan presses x 7 "
              f"presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print("-- stage 2: random walks, which is what the plans cannot cover --")
    bad += _chirality_probe(game, expert)
    print("symmetry clean -- the rotation and flip augmentation is exact"
          if not bad else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Rendering audit
# ---------------------------------------------------------------------------

#: Every cell COMPOSITION a board of this game can show. The pieces share one
#: collision layer, so a cell holds at most one of them, but the floor markers
#: (`deactivator`, `switch`) are separate layers and DO stack under a piece --
#: which is the whole point of `@` (a tower parked on a deactivator) and of the
#: last level's target.
_COMPOSITIONS = {
    "floor": (),
    "vwall": ("vwall",),
    "iwall": ("iwall",),
    "square": ("square",),
    "block": ("block",),
    "fsquare": ("fsquare",),
    "ksquare": ("ksquare",),
    "dosquare": ("dosquare",),
    "player": ("player",),
    "deactivator": ("deactivator",),
    "switch": ("switch",),
    "square+deactivator": ("square", "deactivator"),
    "square+overload": ("square", "overload"),
    "fsquare+overload": ("fsquare", "overload"),
    "dosquare+switch": ("dosquare", "switch"),
    "player+switch": ("player", "switch"),
    "square+connectedu": ("square", "connectedu"),
    "square+connectedd": ("square", "connectedd"),
    "square+connectedl": ("square", "connectedl"),
    "square+connectedr": ("square", "connectedr"),
    "square+connectedud": ("square", "connectedu", "connectedd"),
    "square+connectedlr": ("square", "connectedl", "connectedr"),
}

#: Pairs that are ALLOWED to render identically, with the reason. `ksquare` is
#: `fsquare` -- same mechanics, and this file gave it the same picture on
#: purpose (see the module docstring); `iwall` is `vwall` for the same reason.
_AUDIT_EXPECTED = {("fsquare", "ksquare"), ("vwall", "iwall")}


def _audit() -> int:
    """Assert every cell composition a board can show is distinct at every cell
    size in use -- except the two pairs that are deliberately the same object
    wearing one picture.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor).
    """
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in _COMPOSITIONS.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])
                   and (a, b) not in _AUDIT_EXPECTED
                   and (b, a) not in _AUDIT_EXPECTED]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad else
          f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plan_report())
    if "--selfcheck" in sys.argv:
        sys.exit(1 if (selfcheck() + _replay_check()) else 0)
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(MatchFlowSolver.main())
