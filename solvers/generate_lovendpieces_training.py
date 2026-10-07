"""Generate Phase-1 training data for the PuzzleScript game ps:lovendpieces
("Love and Pieces", lexaloffle).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: an exact model of the
mechanic, an exhaustive solver over it, and the reports that certify both.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_lovendpieces",
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
exactly. Every expert step also carries the full set of equally-optimal presses.

THE GAME
--------
You are a face on a blue-walled board scattered with grey pieces. A press slides
you one cell; touching a piece turns it into another copy of YOU, and the new
copy immediately touches whatever it is beside, so an orthogonally-connected
clump joins you all at once. Everything you have collected then moves with you:
the blob is RIGID, and if any cell of it would enter a wall the whole turn is
cancelled (``[ > Player | Wall ] -> cancel`` -- there is no partial move and no
way to wait). You win when no grey piece is left.

So the puzzle is a tour: route a body that GROWS as you collect, through gaps
that stop fitting as it grows. The rule that makes it a puzzle rather than a
walk is the cancel -- a blob wide enough to touch a wall on one side can no
longer move that way at all, and the shape it has by then is a consequence of
where it was standing when it swallowed each piece, not of the order alone.

THE ONE .txt CHANGE, AND WHY IT IS A FIDELITY FIX
-------------------------------------------------
The absorption rule now sits inside a ``startloop`` / ``endloop`` block:

    startloop
    late [ Player | GrayBlock ] -> [ Player | Player ]
    endloop

In real PuzzleScript that is a no-op -- a rule is re-applied until it stops
matching, so the clump floods completely on the turn you touch it. This
interpreter is not: `PSEngine._execute_late_single` iterates each of the rule's
four expanded DIRECTIONS to a fixpoint but never re-runs the direction LIST, so
the flood advanced exactly one "shell" per turn along whichever direction came
first. Measured on the shipped boards, that is 4192 of the 1.19M transitions
out of level 4's first 400 000 states -- and level 4 only, because it is the one
board with two pieces orthogonally adjacent (it has two such pairs; no other
level has any).

It was worth fixing for a reason beyond fidelity: **the one-pass flood is not
equivariant under the presentation group**, and this repo's frame augmentation
is mandatory. The rule list is walked up, down, left, right in ENGINE
coordinates whatever the screen shows, so "the piece above the piece you just
ate joins you too" was true at some rotations and false at others -- a
difference no policy can read off the frame. Under ``startloop`` the flood is a
connected-component closure, movement is "a run of blob cells advances unless
something solid is in front of it", and the win names no direction: every part
of the mechanic is now direction-free, which is what ``--symmetry`` measures and
what entitles the game to the 16-presentation augmentation.

Nothing about the levels changes. Every shortest solution is the same length
before and after the fix (4, 14, 9, 21, 14, 21, 19, 15); the one-pass version
just made level 4 spend extra turns finishing a clump it had already reached.

(``data/puzzlescript_games/Love_and_Pieces.txt`` is a byte-duplicate of this
game under its other name, referenced by no game folder and left untouched.)

THE MODEL
---------
`_Model` re-implements the mechanic natively over BITMASKS -- players and pieces
are two ints, the flood is four shifts and an AND -- which is what makes
enumerating a level affordable, and `--selfcheck` fuzzes it against the real
interpreter: same settled state, same CANCEL decision, same win flag, over the
nine shipped levels AND 600 random synthetic boards. The random boards are not
decoration. Two mechanics are unreachable on the shipped levels and would
otherwise never be tested at all:

  * a piece can only ever be adjacent to the blob at the START of a turn if the
    level was laid out that way (the end-of-turn flood clears adjacency
    otherwise), and a blob pressed INTO a piece does not move -- it stands still
    and eats it;
  * when that happens to only PART of the blob, the blob TEARS: the run of cells
    behind the blocked one stays, and every other run advances. Nothing in the
    .txt says so -- it falls out of ``cancel`` naming only Wall, so a piece stops
    a cell silently instead of stopping the turn.

Both are modelled exactly even though a shipped level cannot reach them, because
an exploration prefix on a level that could would walk straight into them.
The fuzz reports how many times it exercised each, and fails if it exercised
neither.

The state -- (blob cells, surviving pieces) -- is exact and small, so the expert
is not a heuristic search: a layered forward BFS runs until the first WIN
appears, which is the true ``d*``, and a reverse BFS over the edges collected on
the way gives distance-to-win for every state on a shortest path. That yields a
certified SHORTEST plan and, for free, the exact optimal-action SET at each
step. The BOUND is what makes it cheap, and it is the only thing that does:
levels 2 and 4 have reachable components of hundreds of thousands of states
(both are still growing at a 200 000 cap), but their wins are 9 and 14 presses
away and 15 200 / 63 260 states are within reach of them. All nine levels solve
in 0.6 s, which is also why there is no on-disk plan cache
(`PSExpert.plan_cache_path` stays None -- it earns its keep at ~2 minutes, not
at ~0.6 seconds).

The reverse BFS has to run over the FORWARD-collected edges rather than
expanding backwards from the goal: a swallowed piece is gone and a blob cell
never leaves, so the transition relation is directed and there is no backward
successor function.

THE LEVELS
----------
All nine shipped boards are used as they are -- no level was authored here:

    level  size    pieces  plan  states  ties   what it is
    0      10x12    2        4      35     0    two pieces on one row
    1      10x12    4       14     439     1    four pieces behind wall stubs
    2      10x12   18        9   15200     1    two long diagonals of pieces:
                                                the blob grows a diagonal arm
                                                and eats them in cascades, 18
                                                pieces in 9 presses
    3      10x12    7       21    6969     1    a symmetric board, every piece
                                                tucked beside a wall stub
    4      10x12   13       14   63260     1    the dense one, and the only
                                                board with two pieces side by
                                                side
    5      10x12    7       21    8477     3    three wall slabs, a long tour
    6      10x12    8       19   12101     0    a pinwheel of four short walls
    7      10x12    7       15    2009     1    a lattice of pillars
    8      10x12    1      ---       2   ---    the heart (see below)

``states`` is how many states the bounded search actually visits and ``ties`` is
how many of the plan's steps have more than one equally-optimal press. Every
level is engine-verified: the plan the model produces is replayed through the
interpreter and the interpreter reports the win.

**Level 8 is unwinnable, and that is the joke.** It is a heart drawn in player
cells, with one grey piece parked at ``(0, 11)`` -- a corner of the wall frame
whose only two in-board neighbours are both Wall. Nothing can ever touch it, so
"No GrayBlock" can never hold. The blob can shuffle one cell right and back and
that is the entire reachable space: two states, no win. The generator does not
special-case it -- `discover_solvable` enumerates those two states, gets no
plan, and drops the level, which is a PROOF rather than an assumption. It is
left in the file as shipped.

RENDERING
---------
Nothing needed recolouring, which is worth stating explicitly because it is rare
in this tree. The four compositions a board can show -- floor (white/lightblue),
Wall (a blue/grey bevel), Player (yellow/orange/red) and GrayBlock (grey with a
green core) -- are pairwise distinct as rendered, and every level is 10x12, so
they render at 5px per cell where a 5x5 sprite is sampled 1:1 and no detail is
lost. ``--audit`` is the regression test, comparing WHOLE frames of a board
filled with each composition rather than an arithmetic cell crop (see
ps:explod). ``LitBlock`` is declared by the game, used by no rule and placed in
no level; it is checked anyway.

AUGMENTATION
------------
Engine state after reset is identical for every seed (the levels are fixed ASCII
maps), so the only per-(seed, level) variables are presentational: the frame
rotation in {0,1,2,3} plus an independent horizontal and vertical flip, each
with the matching directional action remap (`_FLIP_GAMES`). ``--symmetry``
proves that is exact rather than arguing it: every level's plan is replayed on
all eight turned and mirrored copies of its own board, built by transforming the
LEVEL LAYOUT so the interpreter re-loads it itself, and every press must leave
the pieces exactly where the transform of the reference run put them. 8 levels x
16 presentations. The expert plan is therefore seed-independent: solved once per
level, memoized, replayed per seed with that seed's remapped screen actions.

OPTIMAL-ACTION SETS
-------------------
The reverse BFS gives distance-to-win exactly on the shortest-path DAG, so the
optimal set at a state is just ``{d : dist(step(d)) == dist - 1}`` -- no probing,
no reordering heuristic. 8 of the 117 expert steps across the eight levels have
more than one right answer, and labelling one of them as "the" answer would train
against the truth on those. No step ever ships unlabelled.

ACTION5 is searched like any other press even though it is bound to no rule
here: it can only ever be a self-loop, so the state dedup prunes it and it never
enters an optimal set. ``--plans`` reports how many reachable states it changes
(0), rather than the search assuming it.

RECOVERY
--------
``recovery_mode = "reset"`` (the family default). Exploration here is genuinely
irreversible -- a swallowed piece does not come back, and a blob that has grown
too wide for a corridor can never get back through it -- so the episode-wide
epsilon prefix explores freely and ONE RESET restores the level start, from
which the memoized plan replays a guaranteed win.

Usage (run from the repo root):
    python solvers/generate_lovendpieces_training.py --episodes 200 \
        --out data/training_multi_level/lovendpieces

    python solvers/generate_lovendpieces_training.py --plans
    python solvers/generate_lovendpieces_training.py --selfcheck
    python solvers/generate_lovendpieces_training.py --symmetry
    python solvers/generate_lovendpieces_training.py --audit
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.common.ps_astar import (Plan, PSAStarSolver, PSExpert,     # noqa: E402
                                     _load_game_module)

GAME_NAME = "lovendpieces"
GAME_MODULE_ID = "ps:lovendpieces"

#: The presses the search branches on, in the order it tries them. ACTION5 is
#: bound to no rule in this game; it is kept in the list so the search MEASURES
#: that rather than assuming it (a self-loop is pruned by the state dedup).
KEYS = ("up", "down", "left", "right", "action")
_ACTION = KEYS.index("action")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
#: index of the opposite of each of the four directions
_OPP = (1, 0, 3, 2)


# ---------------------------------------------------------------------------
# Native model of the mechanic
# ---------------------------------------------------------------------------

class _Model:
    """Love and Pieces as a state machine over cell BITMASKS.

    A state is ``(body, pieces)``: two ints whose set bits are the blob's cells
    and the surviving GrayBlocks, indexed ``r * w + c``. Walls are static and
    live on the model. Bitmasks rather than frozensets because the whole
    mechanic is shifts and masks -- the flood is one OR of four shifts ANDed
    with the pieces -- and because a queued search state is then two ints
    instead of two sets of cells.

    ``blocked`` is the walls PLUS the board border. That is exact rather than
    convenient: `model_from_engine` checks that every border cell which is not
    a Wall has only Walls for in-board neighbours, so no blob cell can ever
    reach one or move into one. (Level 8's marooned piece sits on such a cell.)
    """

    __slots__ = ("h", "w", "full", "nc0", "ncl", "walls", "blocked")

    def __init__(self, h: int, w: int, walls: int):
        self.h, self.w = h, w
        self.full = (1 << (h * w)) - 1
        col0 = sum(1 << (r * w) for r in range(h))
        coll = sum(1 << (r * w + w - 1) for r in range(h))
        border = col0 | coll | ((1 << w) - 1) | (((1 << w) - 1) << ((h - 1) * w))
        self.walls = walls
        self.blocked = (walls | border) & self.full
        self.nc0 = self.full & ~col0        # everything but the first column
        self.ncl = self.full & ~coll        # everything but the last column

    # -- dynamics ----------------------------------------------------------
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

    def flood(self, body: int, pieces: int) -> tuple[int, int]:
        """Absorb every piece orthogonally connected to the blob, to a fixed
        point -- ``late [ Player | GrayBlock ] -> [ Player | Player ]`` inside
        the ``startloop``. Order-free by construction: it is the connected
        component of the pieces that touches the body."""
        while True:
            conv = (self.sh(body, 0) | self.sh(body, 1) |
                    self.sh(body, 2) | self.sh(body, 3)) & pieces
            if not conv:
                return body, pieces
            body |= conv
            pieces &= ~conv

    def step(self, state, di: int):
        """The settled state after pressing ``KEYS[di]``, or None if the turn was
        cancelled (some blob cell would have entered a wall -- which in this game
        undoes the whole turn, absorption included, rather than moving part of
        the blob).

        Movement is per RUN, not per blob: a maximal run of blob cells along the
        press direction advances only if the cell past its head is free. Past
        the cancel check the only thing that head can be is a piece, and a piece
        is only ever in front of the blob on a board that was laid out that way
        -- but when it is, that run stands still while the rest of the blob walks
        on. See the module docstring."""
        body, pieces = state
        if di != _ACTION:
            if self.sh(body, di) & self.blocked:
                return None                                  # a wall ahead
            back = _OPP[di]
            stuck = body & self.sh(pieces, back)             # head-on into a piece
            while True:                                      # ... and everyone behind it
                more = stuck | (body & self.sh(stuck, back))
                if more == stuck:
                    break
                stuck = more
            body = stuck | self.sh(body & ~stuck, di)
        return self.flood(body, pieces)

    @staticmethod
    def won(state) -> bool:
        return not state[1]

    # -- exhaustive solve ---------------------------------------------------
    def solve(self, start, cap: int):
        """A shortest press sequence from ``start`` plus the exact optimal SET
        at each of its steps, or ``(None, None)`` if the level cannot be won.

        Forward BFS one whole LAYER at a time until a layer contains a win. That
        depth is ``d*``, every state on a shortest path has been generated (BFS
        depth is minimal, so a state on a shortest path is reached at its own
        depth), and every edge out of a state at depth < d* has been recorded --
        which is exactly the subgraph a reverse BFS from the wins needs to give
        the true distance-to-win of every such state. The bound is the whole
        trick: the reachable component of these boards runs to hundreds of
        thousands of states while the win is a dozen presses away.
        """
        if self.won(start):
            return [], []
        depth = {start: 0}
        edges: dict = {}
        layer = [start]
        wins: list = []
        d = 0
        while layer and not wins:
            d += 1
            nxt: list = []
            for state in layer:
                out = {}
                for di in range(len(KEYS)):
                    n = self.step(state, di)
                    if n is None:
                        continue
                    out[di] = n
                    if n not in depth:
                        depth[n] = d
                        if self.won(n):
                            wins.append(n)
                        else:
                            # A win is terminal: its successors are unreachable
                            # in play, and expanding them would let the reverse
                            # BFS route "through" a win.
                            nxt.append(n)
                        if len(depth) > cap:
                            return None, None
                edges[state] = out
            layer = nxt
        if not wins:
            return None, None

        rev: dict = {}
        for state, out in edges.items():
            for n in out.values():
                rev.setdefault(n, []).append(state)
        dist = {state: 0 for state in wins}
        queue = deque(wins)
        while queue:
            state = queue.popleft()
            for prev in rev.get(state, ()):
                if prev not in dist:
                    dist[prev] = dist[state] + 1
                    queue.append(prev)

        presses: list[str] = []
        optsets: list[list[str]] = []
        state = start
        while dist[state]:
            best = [di for di, n in edges[state].items()
                    if dist.get(n, 1 << 30) == dist[state] - 1]
            optsets.append([KEYS[di] for di in best])
            presses.append(KEYS[best[0]])
            state = edges[state][best[0]]
        return presses, optsets

    def reachable(self, start, cap: int) -> tuple[set, bool]:
        """``(states, hit_the_cap)`` for the reachable component. Only the
        reports use it -- the expert never needs more than `solve`'s bound --
        and it returns the partial set rather than None so a capped sweep still
        MEASURES what it saw instead of silently skipping the level."""
        seen = {start}
        queue = deque([start])
        while queue:
            state = queue.popleft()
            for di in range(len(KEYS)):
                n = self.step(state, di)
                if n is not None and n not in seen and not self.won(n):
                    seen.add(n)
                    queue.append(n)
                    if len(seen) >= cap:
                        return seen, True
        return seen, False


def model_from_engine(eng, g) -> tuple:
    """Read ``(model, state)`` off the interpreter's live grid."""
    idx = g.obj_name_to_idx
    wall_id, player_id, piece_id = idx["wall"], idx["player"], idx["grayblock"]
    h, w = eng.height, eng.width
    walls = body = pieces = 0
    for r in range(h):
        row = eng.grid[r]
        for c in range(w):
            cell, bit = row[c], 1 << (r * w + c)
            if wall_id in cell:
                walls |= bit
            if player_id in cell:
                body |= bit
            if piece_id in cell:
                pieces |= bit
    for r in range(h):
        for c in range(w):
            if not (r in (0, h - 1) or c in (0, w - 1)):
                continue
            bit = 1 << (r * w + c)
            if body & bit:
                # See `_Model`: the border is treated as solid, and a blob cell
                # standing on it would make that treatment a lie.
                raise ValueError("blob cell on the board border")
            if walls & bit:
                continue
            for dr, dc in _DELTA.values():
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and not (walls >> (rr * w + cc)) & 1:
                    raise ValueError(
                        f"non-wall border cell ({r},{c}) is reachable from "
                        f"({rr},{cc}) -- the model's border-is-solid shortcut "
                        f"does not hold on this board")
    return _Model(h, w, walls), (body, pieces)


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class LovendExpert(PSExpert):
    """Exhaustive optimal expert: bounded forward BFS on `_Model` to the first
    win, then read the plan and the per-step optimal sets straight off the
    distance-to-win field of the shortest-path subgraph.

    It plugs into the shared harness at `PSExpert._search`, so the plan memo, the
    snapshot/restore discipline around it and the level scoping all come from the
    base. `heuristic` is never called -- `_astar` is not the strategy here -- and
    neither is `dead`: an exhaustive search has no use for either.
    """

    directions = list(KEYS)

    #: A runaway guard on a board that does not belong in this file, not a tuning
    #: dial: the widest level here visits 63k states before its win appears.
    state_cap: int = 400_000

    def _search(self, eng):
        model, state = model_from_engine(eng, self.g)
        presses, optsets = model.solve(state, self.state_cap)
        if presses is None:
            return None
        return Plan(presses, optsets)


class LovendSolver(PSAStarSolver):
    game_id = "puzzlescript_lovendpieces"
    game_name = GAME_NAME
    expert_cls = LovendExpert
    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so the generator can never tape frames from a different
    #: build of the game than the agent plays.
    game_module_id = GAME_MODULE_ID

    #: The longest plan is 21 presses; the rest is room for the exploration
    #: prefix and the replay after its RESET. Comfortably inside the adapter's
    #: own 200-step per-level budget.
    max_steps = 150


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    solver = LovendSolver()
    game = solver.make_game(0)
    return solver, game, LovendExpert(game)


def _plan_report() -> int:
    """Per-level size, piece count, plan length, states visited and tie count.

    This is the table in the module docstring; re-run it after touching a level.
    It also measures the two claims the search rests on that are easy to state
    and easy to get wrong: that ACTION5 changes nothing, and that no press ever
    leaves a piece touching the blob (i.e. the flood really does run to a fixed
    point, which is what makes the mechanic direction-free)."""
    _solver, game, expert = _levels()
    eng = game._engine
    total = ties_total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        model, state = model_from_engine(eng, game._game)
        plan = expert.plan(eng, level)
        pieces = bin(state[1]).count("1")
        if plan is None:
            reach, capped = model.reachable(state, expert.state_cap)
            print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
                  f"{pieces:2d} pieces, NO PLAN -- unwinnable "
                  f"({len(reach)}{'+' if capped else ''} reachable states)")
            continue
        # re-derive the visited-state count for the table
        seen = {state}
        layer = [state]
        won = False
        while layer and not won:
            nxt = []
            for s in layer:
                for di in range(len(KEYS)):
                    n = model.step(s, di)
                    if n is None or n in seen:
                        continue
                    seen.add(n)
                    if model.won(n):
                        won = True
                    else:
                        nxt.append(n)
            layer = nxt
        ties = sum(1 for s in plan.optsets if len(s) > 1)
        total += len(plan)
        ties_total += ties
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
              f"{pieces:2d} pieces, plan {len(plan):3d} ({room}), "
              f"{len(seen):6d} states visited, {ties:2d} steps with a tie")
    print(f"total {total} presses, {ties_total} with a tie "
          f"({ties_total / max(1, total):.0%})")

    # The two measured invariants, over as much of each level's reachable
    # component as `sweep` states allow. `capped` levels are reported, never
    # skipped: a sweep that stopped early has still checked what it saw.
    sweep = 200_000
    inert = leftover = states = 0
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
            for di in range(len(KEYS)):
                n = model.step(s, di)
                if n is not None and any(model.sh(n[0], d) & n[1]
                                         for d in range(4)):
                    leftover += 1
    print(f"over {states} reachable states"
          f"{f' (levels {partial} stopped at the {sweep} cap)' if partial else ''}: "
          f"ACTION5 changes {inert} of them, and {leftover} transitions leave a "
          f"piece still touching the blob")
    return 0 if not (inert or leftover) else 1


# ---------------------------------------------------------------------------
# Model self-check (fuzz against the real interpreter)
# ---------------------------------------------------------------------------

def _read(eng, g):
    return model_from_engine(eng, g)[1]


def selfcheck(trials: int = 40, steps: int = 150,
              boards: int = 600, board_steps: int = 30) -> int:
    """Audit the claim the expert rests on: `_Model` reproduces the interpreter
    EXACTLY.

    Three things have to agree on every press, and they are the three the solver
    uses: the settled state (blob cells and surviving pieces), the CANCEL
    decision -- a press the model refuses must leave the interpreter's grid
    byte-identical, and a press it accepts and expects to change something must
    change something -- and the win flag.

    Two fuzz populations, and the second is the one that matters. The nine
    shipped levels exercise the game as it is played; 600 RANDOM boards (random
    walls, random pieces, several blob cells, and pieces deliberately placed
    touching the blob at reset) exercise the two mechanics no shipped level can
    reach: a blob pressed head-on into a piece, which stands still and eats it,
    and a blob only PART of which is blocked, which tears. Fuzzing only the
    shipped levels would have left both untested -- and both are one exploration
    prefix away from a real episode on a board that has them."""
    game = _load_game_module(GAME_MODULE_ID).make_game(seed=0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    bg, wall_id, player_id, piece_id = (idx["background"], idx["wall"],
                                        idx["player"], idx["grayblock"])
    tot = {"presses": 0, "cancels": 0, "noops": 0, "wins": 0,
           "eat_in_place": 0, "tears": 0, "multi": 0}
    bad = 0

    def rollout(tag: str, nsteps: int, rng: random.Random) -> int:
        nonlocal bad
        model, state = model_from_engine(eng, g)
        for i in range(nsteps):
            live = [di for di in range(len(KEYS))
                    if model.step(state, di) not in (None, state)]
            # Mostly play on, occasionally press blind: a uniform walk spends
            # most of a rollout shoving a wedged blob into the same wall, and
            # the presses that change something are where the mechanic lives.
            # Not circular -- a wrong legality claim still has to survive the
            # grid comparison below.
            di = (rng.choice(live) if live and rng.random() < 0.8
                  else rng.randrange(len(KEYS)))
            # coverage: did this press eat something without the blob moving,
            # and did it move only part of the blob?
            if di != _ACTION:
                back = _OPP[di]
                stuck = state[0] & model.sh(state[1], back)
                if stuck:
                    tot["eat_in_place"] += 1
                    if stuck != state[0]:
                        tot["tears"] += 1
            before = [[set(cell) for cell in row] for row in eng.grid]
            eng.step(KEYS[di])
            tot["presses"] += 1
            frozen = eng.grid == before
            predicted = model.step(state, di)
            expected = state if predicted is None else predicted
            actual = _read(eng, g)
            if actual != expected:
                bad += 1
                print(f"  {tag}: state divergence on {KEYS[di]}")
                return 1
            if predicted is None and not frozen:
                bad += 1
                print(f"  {tag}: model cancelled {KEYS[di]}, engine did not")
                return 1
            if predicted is not None and predicted != state and frozen:
                bad += 1
                print(f"  {tag}: model moved on {KEYS[di]}, engine frozen")
                return 1
            if eng.check_win() != model.won(expected):
                bad += 1
                print(f"  {tag}: win divergence on {KEYS[di]}")
                return 1
            eaten = bin(state[1] & ~expected[1]).count("1")
            tot["multi"] += eaten > 1
            tot["cancels"] += predicted is None
            tot["noops"] += predicted == state
            state = expected
            if eng.check_win():
                tot["wins"] += 1
                return 0
        return 0

    for level in range(game.n_levels):
        for t in range(trials):
            game.set_level(level)
            rollout(f"L{level}/{t}", steps,
                    random.Random(f"lovend:shipped:{level}:{t}"))

    for t in range(boards):
        rng = random.Random(f"lovend:board:{t}")
        h, w = rng.randint(5, 10), rng.randint(5, 12)
        game.set_level(0)
        eng.height, eng.width = h, w
        eng.grid = [[{bg} for _ in range(w)] for _ in range(h)]
        for r in range(h):
            for c in range(w):
                if r in (0, h - 1) or c in (0, w - 1):
                    eng.grid[r][c].add(wall_id)
        cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
        rng.shuffle(cells)
        for (r, c) in cells[:rng.randint(0, len(cells) // 4)]:
            eng.grid[r][c].add(wall_id)
        free = [x for x in cells if wall_id not in eng.grid[x[0]][x[1]]]
        rng.shuffle(free)
        k = rng.randint(1, max(1, len(free) // 2))
        for (r, c) in free[:k]:
            eng.grid[r][c].add(piece_id)
        for (r, c) in free[k:k + rng.randint(1, 4)]:
            eng.grid[r][c].add(player_id)
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        rollout(f"board{t}", board_steps, rng)

    print(f"selfcheck: {tot['presses']} presses "
          f"({game.n_levels} levels x {trials} rollouts + {boards} random "
          f"boards), {tot['cancels']} cancels, {tot['noops']} no-ops, "
          f"{tot['wins']} incidental wins, {tot['multi']} multi-piece floods, "
          f"{tot['eat_in_place']} eat-in-place presses of which "
          f"{tot['tears']} tore the blob")
    if not tot["tears"] or not tot["multi"]:
        print("FUZZ TOO WEAK: it never tore the blob or never flooded a clump")
        return 1
    return bad


def _replay_check() -> int:
    """Every level's plan, replayed through the interpreter, must WIN -- and the
    optimal sets must be honest: at each step, every press the label calls
    optimal has to land on a state whose own distance-to-win is one shorter."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        model, state = model_from_engine(eng, game._game)
        if plan is None:
            _seen, capped = model.reachable(state, expert.state_cap)
            if capped:
                bad += 1
                print(f"  L{level}: no plan and the component blew the cap -- "
                      f"'unwinnable' is unproven here")
            else:
                print(f"  L{level}: no plan, component exhausted (unwinnable)")
            continue
        for i, press in enumerate(plan):
            if press not in plan.optsets[i]:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            remaining = len(plan) - i - 1
            for alt in plan.optsets[i]:
                nxt = model.step(state, KEYS.index(alt))
                if nxt is None:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, "
                          f"but it is cancelled")
                    continue
                sub, _ = model.solve(nxt, expert.state_cap)
                if sub is None or len(sub) != remaining:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but it "
                          f"leaves {len(sub) if sub else 'no'} presses against "
                          f"{remaining}")
            state = model.step(state, KEYS.index(press))
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
    mirror left-right first, then turn clockwise ``k`` times.

    The direction map is DERIVED from the same linear part that moves the cells,
    so the two cannot drift apart."""
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


def _pieces(eng, g) -> tuple:
    idx = g.obj_name_to_idx
    return tuple(frozenset((r, c) for r in range(eng.height)
                           for c in range(eng.width) if o in eng.grid[r][c])
                 for o in (idx["player"], idx["grayblock"]))


def _chirality_probe(game, expert, rollouts: int = 12, steps: int = 40) -> int:
    """The second half of `_symmetry`, and the half with teeth.

    Replaying a level's PLAN under the eight transforms cannot find a mechanic
    the plan does not happen to use, and this game's defect was exactly that
    kind: with the absorption rule unlooped, the interpreter's fixed
    up/down/left/right walk made the flood orientation-dependent, but only on
    boards where two pieces sit side by side and only when the blob reaches one
    of them from a particular side. Every shortest plan on the shipped levels
    misses it -- verified: stage 1 reports SYMMETRIC on the UNFIXED game too.

    So this stage does not follow a plan. It random-walks each level (steered by
    the model toward presses that change something, which is where the mechanic
    lives), records every ``(board, press)`` it visits, and then re-plays each
    one INDEPENDENTLY on all seven other presentations of that same board. A
    single order-sensitive transition anywhere in the walk is a failure.

    ``load_level`` is exact for a mid-walk board here because this game does not
    set ``run_rules_on_level_start``, so loading is a plain grid assignment.
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
            rng = random.Random(f"lovend:chiral:{level}:{t}")
            model, state = model_from_engine(eng, g)
            for _ in range(steps):
                live = [di for di in range(len(KEYS))
                        if model.step(state, di) not in (None, state)]
                di = (rng.choice(live) if live and rng.random() < 0.8
                      else rng.randrange(len(KEYS)))
                before = [[set(cell) for cell in row] for row in eng.grid]
                eng.step(KEYS[di])
                triples.append((before, KEYS[di], _pieces(eng, g)))
                nxt = model.step(state, di)
                state = state if nxt is None else nxt
                if eng.check_win():
                    break

        notes = []
        for k, mirror in itertools.product(range(4), (False, True)):
            if (k, mirror) == (0, False):
                continue
            cell, dims, dmap = _transform(k, mirror)
            th, tw = dims(hw)
            for i, (before, press, after) in enumerate(triples):
                turned = [[set() for _ in range(tw)] for _ in range(th)]
                for r, row in enumerate(before):
                    for c, objs in enumerate(row):
                        tr, tc = cell((r, c), hw)
                        turned[tr][tc] = set(objs)
                eng.load_level(turned)
                eng.step(dmap[press])
                tot += 1
                want = tuple(frozenset(cell(x, hw) for x in s) for s in after)
                if _pieces(eng, g) != want:
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

    Reading the two rules and declaring them direction-free is exactly the
    argument that was wrong in other ps: games -- and it WOULD have been wrong
    here before the ``startloop`` fix, because the interpreter walks a rule's
    expanded directions in a fixed order and the unlooped flood therefore
    depended on the board's absolute orientation (see the module docstring).

    Stage 1 (here) replays each level's plan on all eight turned and mirrored
    copies of its own board, built by transforming the LEVEL LAYOUT so the
    interpreter re-loads it itself; every press must leave the pieces exactly
    where the transform of the reference run put them, and the run must still
    win. Stage 2 (`_chirality_probe`) is what actually catches this game's
    defect -- see its docstring.

    Both object classes are compared -- there is no directional art in this game
    and no per-orientation wall tiling, so nothing is expected to differ."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level:2d}: no plan (unwinnable) -- skipped")
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

def _audit() -> int:
    """Assert every cell COMPOSITION a board can show is distinct at every cell
    size in use.

    The composition, not the object, is what has to be readable -- but this game
    has no stacks at all: Player, Wall, GrayBlock and LitBlock share one
    collision layer, so a cell is floor or exactly one of them. That makes the
    audit short and it still has to be run, because the whole corpus is
    "which cells are pieces and which are blob".

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {"floor": (), "wall": ("wall",), "player": ("player",),
             "piece": ("grayblock",), "litblock": ("litblock",)}

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
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
    sys.exit(LovendSolver.main())
