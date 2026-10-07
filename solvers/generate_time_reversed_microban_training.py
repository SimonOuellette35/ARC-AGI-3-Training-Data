"""Generate Phase-1 training data for the PuzzleScript game
ps:time_reversed_microban ("Time-reversed Microban", Toph Wells, after David
Skinner's Microban).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the exact distance FIELD that model is solved
with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_time_reversed_microban",
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
exactly. Every expert step carries the full set of equally-optimal presses.

The game
--------
Ten Microban levels PLAYED BACKWARDS. The entire rule list is a single line:

    [ no Wall no Player no Crate | > Player | Crate ] -> [ Player | Crate | ]

Read it as three cells along the direction of the press: the square BEHIND the
player, the player (moving in that direction), and the crate in front of it.
After the rule the player is on the square behind and the crate is on the square
the player just left. Nothing on the right hand side carries a force, so the
rule does the whole move and the movement phase then moves nothing.

So the press that moves a crate points AT the crate and moves you AWAY from it.
The mechanic that produces, stated in the direction things actually travel: to
move a crate one square, stand on the far side of it and walk further in that
direction -- the crate follows you. **This is a PULL sokoban. There is no push
in this game.** (The name is exact: reverse the time axis of a Microban push
and you get one of these pulls, which is also why the shipped levels are
Microban's levels with the crates and the goals EXCHANGED -- compare level 1
here, which is L9 in file order, against Microban's level 1: the same walls and
the same player, with the two crates standing on the squares Microban's goals
are on and the two targets on the squares its boxes are on.)

One press, completely:

  * off the board, or into a wall: nothing happens at all;
  * onto an empty square: the player walks there;
  * into a crate WITH a free square behind the player: the two move together --
    the player takes the square behind it, the crate takes the player's;
  * into a crate WITHOUT a free square behind (a wall, the board edge or another
    crate): nothing happens. Player and Crate share a collision layer, so with
    the rule unmatched the movement phase simply refuses the step.

The win is ``All Target on Crate`` -- every target square carries a crate. Extra
crates would be free to sit anywhere; on all ten shipped levels the counts match
exactly (2 to 6 of each).

**This game is NOT reversible and it is full of dead ends.** A pull is undone by
a push and there are no pushes, so the state graph is directed: shove a crate
into a corner it cannot be dragged out of and the level is over while it still
looks playable. That is not a rare corner either -- 4146 of L0's 5309
reachable boards can never win, and every one of the ten levels has some. Two things in this
file follow from it and nothing else: the exploration prefix ends in a RESET
rather than being walked back, and every epsilon detour is checked against the
field before it is taken (below).

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`) plus an exact distance-to-win
field (`_Field`). The shared `PSPushExpert` does not apply: its macro is "walk to
a push square, then push", and this game has no push -- its corner-deadlock test
is also the wrong test, because the deadlocks here are the ones a PULL cannot
undo, which is a different (and much larger) set.

`_Field` **enumerates**, and the reason it can is measured rather than assumed:
the ten levels' entire reachable spaces are 129, 223, 311, 320, 751, 1707,
5309, 5815, 10198 and 18193 states -- 43k for the whole game, and all ten
closures together are built in well under a second.
A pull sokoban prunes itself: a crate only ever moves when the player is on the
far side of it with a free square beyond, so the vast majority of the C(free,
crates) x free placements are simply not reachable. So the field is

  * one forward closure from the state it is asked about, over `_Board.step`,
    with a won board treated as TERMINAL (a win ends the episode, so no
    shortest path continues through one);
  * then a backward BFS from the won boards over the closure's own reversed
    edges, which labels every state in it with the true shortest presses to a
    win -- and leaves the states that cannot win at all unlabelled, which is a
    PROOF of deadness rather than a search giving up.

The first query is the level's start, whose closure contains every board the
recording can subsequently reach, so everything after it is a dict lookup. The
field is then read directly for everything else:

  * the PLAN is "walk down the gradient": at distance ``d``, any press whose
    successor is at ``d - 1``;
  * the OPTIMAL SETS are all of those presses, exactly, with no inference;
  * RECOVERY is the same lookup from wherever the board happens to be, which is
    the whole `supports_recovery` contract;
  * and the DEADNESS TEST that makes epsilon detours safe is the same lookup
    again, in O(1) -- which is what lets this game take detours at all.

Shortest, and how that is known
-------------------------------
Every plan is provably shortest, and the proof is three independent derivations
agreeing on all ten levels:

  * this file's closure field;
  * a plain FORWARD BFS to the first win over the same model, which is shortest
    by construction and shares no line with the field (``--selfcheck`` pass 3);
    its per-step tie sets are re-derived a second time by brute force (pass 4);
  * the same two sweeps driven by the REAL INTERPRETER (``--engine``): forward
    BFS from the level start by pressing actual buttons until a win appears,
    edges recorded, then backward over those edges. No `_Board` call of any kind
    participates -- the successor of a board is whatever `PSEngine.step` makes of
    it and the goal test is `check_win`. It agrees on all ten lengths and on
    every tie set.

The ten plans come to 476 presses, of which 9 stand at a step with a second
equally right answer -- a pull sokoban is tight, and the tie sets say so.

``--enumerate`` adds a fourth, and here it is affordable on EVERY level rather
than on a small subset: the shared `StateGraph.build` walks the interpreter's
whole reachable space and solves the graph exactly. That is also the independent
confirmation of the deadness counts above -- the states it reports as unable to
win are the states `_Field` leaves unlabelled.

Recovery
--------
`supports_recovery` with the family's RESET-mode prefix, PLUS ``epsilon = 0.06``
detours inside the expert replay. The expert re-plans from the LIVE board (a
field lookup at whatever state the engine is in), so a detour is answered
exactly: the taken action is the mistake and ``optimal`` is the recovery, which
is the signal a policy needs after its own error.

The division of labour between the two is forced by the irreversibility:

  * a DETOUR is offered a random alternative press and takes it only if the
    field still gives the resulting board a distance. That test is exact (the
    field enumerates, so "no distance" means "provably cannot win", not "the
    search gave up") and it is a dict lookup, so it is affordable at every step.
    A detour therefore never bricks a board and the replay always finishes;
  * the PREFIX is not offered that choice -- it explores freely, and on the
    tightest boards that strands it about half the time (``--selfcheck``'s
    recovery pass walks 1 to 24 random presses from a level start and finds 42
    of 80 boards already lost on L8, 39 of 80 on L9) -- so it ends in ONE RESET
    back to the level start, from which the cached plan is a guaranteed win.
    Exactly the human arc: flail, get stuck, hit reset, then solve.

Measured end to end at ``epsilon = 0.06``: 100 seeds, 100 attempts, 1000 levels,
1000 WIN, every recording replayed FRAME-EXACT through `perform_action`, max
action index 5 (no ACTION7 ever reaches the buffer), 0 of 54,518 expert steps
unlabelled, 5.5% of them a detour whose ``optimal`` is the recovery, and the
files byte-identical across two runs.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the frame rotation
(``rotation_k`` in 0..3) and the two flips, with their matching directional
action remap. 10 levels x 16 presentations = 160.

``Time-reversed_Microban`` is in `PuzzleScriptAdapter._FLIP_GAMES`; the argument
is recorded there and ``--symmetry`` measures it -- every level's plan AND a
seeded 200-press random walk (which drags crates off their targets, jams them
flat against walls where nothing can pull them out again, and presses the
unbound ACTION key) replayed at all 16 presentations, requiring every frame to
be the exact transform of the unaugmented one.

Rendering
---------
One sprite had to change, and it was the goal itself: Target shipped as an
orange ring on rows/cols 1-3, which is exactly the nine pixels the Player paints
over, so ``player on target`` rendered as a player on bare floor at all eight
board shapes -- the stock sokoban-template bug, seen before in ps:stickyban and
ps:swap_sokoban. It is at its worst in a PULL sokoban: the player has to stand
on the far side of every square a crate must end on, so it is on the target
squares constantly. The fix -- the ring plus the four corners, which the Player
leaves transparent and the Crate covers -- is the same forced one; the full
statement is the header comment in
``data/puzzlescript_games/Time-reversed_Microban.txt``.

``--audit`` renders every cell COMPOSITION a board of this game can hold -- wall,
floor, target, player, crate, and the player and the crate on a target -- as a
whole 64x64 frame of a uniform board, at every cell size the ten levels render
at (5, 7, 8 and 9 px), and requires them pairwise distinct. Whole frames rather
than one cell sliced out of a mixed board: `_render_frame` upscales and
centre-pads, so slicing by ``cell_px`` arithmetic reads the wrong pixels (the
ps:explod lesson). It also asserts no composition renders as a uniform frame of
the letterbox colour (the ps:stand_iii lesson) -- a real risk here, because
Target is ``orange`` and Wall is ``BROWN DARKBROWN``, and ARC's palette has one
entry for brown and orange both.

Usage (run from the repo root):
    python solvers/generate_time_reversed_microban_training.py --episodes 200 \
        --out data/training_multi_level/time_reversed_microban

    python solvers/generate_time_reversed_microban_training.py --plans
    python solvers/generate_time_reversed_microban_training.py --selfcheck
    python solvers/generate_time_reversed_microban_training.py --engine
    python solvers/generate_time_reversed_microban_training.py --enumerate
    python solvers/generate_time_reversed_microban_training.py --audit
    python solvers/generate_time_reversed_microban_training.py --symmetry
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                          # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION                     # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,   # noqa: E402
                                     screen_action, snapshot, restore)
from utils.rotation import inverse_remap_action_full                  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Time-reversed_Microban"

#: Engine directions, in the order ties are broken. There is no ACTION rule in
#: this game (no rule has ``action`` on a left-hand side), so the X button is not
#: a move and branching on it would cost a quarter of every sweep for nothing.
#: The exploration prefix still presses it -- a live agent has that button -- which
#: is why ``--symmetry``'s random walk presses it too.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: ``di ^ 1`` is the opposite direction, which is what the mechanic needs: the
#: square the player retreats onto is the one behind it. Stated here because the
#: whole model leans on the pairing of the `DIRS` order.
assert [DIRS[d ^ 1] for d in range(4)] == ["down", "up", "right", "left"]

#: Disk cache of every level's start plan AND its optimal-action sets. The whole
#: game's fields take well under a second, so this is not the load-bearing cache
#: it is on the big ps: sokobans -- it is here so a `parallelize_generator`
#: fan-out shares one derivation rather than repeating it on every core. Delete
#: the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "time_reversed_microban_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic.

    Cells are flat ``r * w + c`` indices and a STATE is ``(player, crate
    bitmask)`` -- the only two things any rule in this game moves. Walls and the
    target set never change, so they live here rather than in the state, and the
    crate set is a bitmask because the sweeps below key tens of thousands of
    states and a Python int is the cheapest exact spelling of a subset of
    squares.

    THE MECHANIC, as one press in direction ``d``. Let ``q`` be the square in
    front of the player and ``b`` the square behind it:

      * ``q`` off the board or a wall: nothing happens at all (no rule has a wall
        on its left, so the movement phase simply refuses the move);
      * ``q`` empty: the player walks to ``q``;
      * ``q`` a crate and ``b`` free floor: the PULL -- the player retreats to
        ``b`` and the crate takes the player's old square. Note which way the
        crate travels: it moves in the direction ``-d``, i.e. towards the button
        you pressed and away from where you end up;
      * ``q`` a crate and ``b`` a wall, the board edge or another crate: nothing
        happens. The rule's first cell is ``no Wall no Player no Crate``, so it
        does not match, and Player and Crate share a collision layer, so the
        movement phase cannot push through either.

    That is the entire game. `won` is the win condition verbatim (``All Target on
    Crate``: every target square carries a crate), and it is the only place
    targets appear -- Target is alone on its own collision layer and blocks
    nothing, so it never affects a move.
    """

    __slots__ = ("h", "w", "n", "wall", "tmask", "free", "edge", "sig")

    def __init__(self, h: int, w: int, walls, targets):
        self.h, self.w = h, w
        self.n = h * w
        self.wall = frozenset(walls)
        self.tmask = 0
        for t in targets:
            self.tmask |= 1 << t
        #: Every non-wall square, in scan order. The player and the crates live
        #: here and nowhere else.
        self.free = tuple(i for i in range(self.n) if i not in self.wall)
        #: ``edge[cell][di]`` is the square reached by moving one step in
        #: ``DIRS[di]``, or -1 when that is off the board or a wall. Folding the
        #: wall test into the table is what keeps `step` down to two lookups and
        #: a bit test -- and it is read for the square BEHIND the player too,
        #: via ``edge[p][di ^ 1]``.
        self.edge = []
        for i in range(self.n):
            r, c = divmod(i, w)
            row = []
            for d in DIRS:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                j = nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1
                row.append(-1 if j < 0 or j in self.wall else j)
            self.edge.append(tuple(row))
        self.sig = (h, w, tuple(sorted(self.wall)), tuple(sorted(targets)))

    # -- the mechanic --------------------------------------------------------
    def step(self, state, di: int):
        """The state after pressing ``DIRS[di]``, or ``state`` itself when the
        press moves nothing (which is a non-move: no shortest path contains
        one)."""
        p, mask = state
        q = self.edge[p][di]
        if q < 0:                            # off the board, or a wall
            return state
        bit = 1 << q
        if mask & bit:                       # a crate in front: try to pull it
            b = self.edge[p][di ^ 1]
            if b < 0 or (mask >> b) & 1:     # nowhere to retreat to
                return state
            return (b, (mask ^ bit) | (1 << p))
        return (q, mask)                     # a plain walk

    def won(self, state) -> bool:
        return (state[1] & self.tmask) == self.tmask

    # -- the win set, enumerated directly ------------------------------------
    def goal_states(self, crates: int):
        """Every state this board calls won, given ``crates`` crates on it.

        Used only by the reports (`_selfcheck` pass 5 checks the field's own won
        set against it): the field itself finds its goals inside the closure it
        enumerates, because on a directed game most of these are not reachable
        from the level start at all.

        ``All Target on Crate`` constrains only the target squares, so a board
        with spare crates has a goal state for every placement of them; on all
        ten shipped levels the crate count equals the target count and the spare
        loop runs once with an empty combination.

        Empty when the level cannot be won at all -- fewer crates than targets,
        or a target buried under a wall (neither happens here; both would
        otherwise be silently mis-fielded as "the search gave up")."""
        targets = [i for i in range(self.n) if (self.tmask >> i) & 1]
        if any(t in self.wall for t in targets) or crates < len(targets):
            return
        spare = [i for i in self.free if not (self.tmask >> i) & 1]
        for extra in itertools.combinations(spare, crates - len(targets)):
            mask = self.tmask
            for c in extra:
                mask |= 1 << c
            for p in self.free:
                if not (mask >> p) & 1:
                    yield (p, mask)


class _Field:
    """Exact presses-to-win for every reachable state, by ENUMERATION.

    WHY ENUMERATION AND NOT A SEARCH. Every other option was worse for a reason
    worth writing down.

    * A *backward* sweep from the enumerated win set -- the shape ps:swap_sokoban
      uses -- is legal here too (the predecessor relation of a pull is a push,
      which is easy to write). It is just far more expensive: the game is
      directed, so the ball of radius d* around the goal set is mostly boards
      that no play can ever produce. Measured on L3, the six-crate
      checkerboard: 7,749,362 states and 963 MB to answer a question whose
      forward closure is 18,193 states. Backward sweeps are the right tool for a REVERSIBLE game,
      where the two directions enumerate the same set; here they are not.
    * A forward A* would need a deadlock test to avoid drowning, and the
      deadlocks of a pull sokoban are not the corner traps a push sokoban's test
      knows -- writing one correctly is harder than enumerating the space it
      would prune.

    So: one forward closure from the state asked about, over `_Board.step`, then
    a backward BFS from the won boards over that closure's own reversed edges. A
    won board is TERMINAL in the closure (a win ends the episode, so no shortest
    path continues through one, and the boards that exist only beyond a win are
    not boards the game can be in).

    The result is exact, and exact in both directions: a state the backward BFS
    labels has that many presses to a win, and a state it does NOT label cannot
    win at all -- which is a proof, since the closure holds every board reachable
    from the query. That second half is what the epsilon detours in
    `record_level` rest on: this game is directed and most of its boards are
    already lost (4146 of level 1's 5332), so "is this alternative press still
    winnable" has to be answered exactly and in O(1), and it is.

    The closure from a state contains the closure of everything in it, so one
    solve labels every state it will ever be asked about downstream. In practice
    the first query is the level's start and every later one -- a detour, a
    re-plan, a recovery probe -- is a dict lookup. `states` counts what has
    actually been enumerated so the reports can show that.

    ``cap`` is a runaway guard, not a budget: past it `get` answers None, which
    `_search` turns into "no plan" and `record_level` turns into a RESET. The
    largest closure in the game is L3's 18,193 states, two orders of magnitude
    under it.
    """

    __slots__ = ("board", "dist", "dead", "cap", "capped", "states", "solves")

    def __init__(self, board: _Board, cap: int):
        self.board = board
        self.cap = cap
        #: state -> exact presses to the nearest win.
        self.dist: dict = {}
        #: states proved unable to win at all.
        self.dead: set = set()
        self.capped = False
        self.states = 0
        self.solves = 0

    def get(self, state) -> "int | None":
        """Presses to a win from ``state``, or None when this board can never be
        won from there (or the cap stopped the closure short of knowing)."""
        hit = self.dist.get(state)
        if hit is not None:
            return hit
        if state in self.dead:
            return None
        if not self._solve(state):
            return None
        return self.dist.get(state)

    def _solve(self, state) -> bool:
        """Enumerate the closure of ``state`` and label all of it. False if the
        cap stopped it (nothing is merged in that case -- a partial closure would
        label states that have unexplored escapes as dead)."""
        board = self.board
        succ: dict = {}
        seen = {state}
        queue = deque([state])
        while queue:
            cur = queue.popleft()
            if board.won(cur):
                succ[cur] = ()               # a win ends the episode
                continue
            row = []
            for di in range(4):
                nxt = board.step(cur, di)
                if nxt == cur:
                    continue                 # a press that does nothing
                row.append(nxt)
                if nxt not in seen:
                    if len(seen) >= self.cap:
                        self.capped = True
                        return False
                    seen.add(nxt)
                    queue.append(nxt)
            succ[cur] = tuple(row)

        rev: dict = {}
        for cur, row in succ.items():
            for nxt in row:
                rev.setdefault(nxt, []).append(cur)
        dist = {s: 0 for s in seen if board.won(s)}
        back = deque(dist)
        while back:
            cur = back.popleft()
            nd = dist[cur] + 1
            for prev in rev.get(cur, ()):
                if prev not in dist:
                    dist[prev] = nd
                    back.append(prev)

        self.dist.update(dist)
        self.dead |= seen - dist.keys()
        self.states += len(seen)
        self.solves += 1
        return True

    def optimal(self, state) -> list:
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``. Ties come out in ``DIRS`` order, which is what makes
        a re-derived plan byte-identical across processes.

        Exact, with nothing inferred: a press is optimal iff its successor is one
        step nearer the win, and both distances are the field's. Every successor
        is already labelled -- it is in the closure that labelled ``state`` -- so
        this never enumerates anything."""
        rest = self.get(state)
        if not rest:                      # None (dead / capped) or 0 (already won)
            return []
        out = []
        for di in range(4):
            nxt = self.board.step(state, di)
            if nxt != state and self.dist.get(nxt) == rest - 1:
                out.append((di, nxt))
        return out

    def plan(self, state) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal set
        at every step, or None if this board can no longer be won."""
        rest = self.get(state)
        if rest is None:
            return None
        presses, optsets = [], []
        cur = state
        for _ in range(rest):
            best = self.optimal(cur)
            if not best:                  # a labelled state always has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class TimeReversedMicrobanExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Field`.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes, the way `PSEnumExpert` replaces it. Here the
    strategy is "read the live board and walk down the exact distance field", so
    `heuristic` is never called and asserts rather than returning a number
    nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery probe, an epsilon detour) is answered exactly --
    including the answer "no", which on this game is a real and common one and
    is a proof rather than a budget running out (see `_Field`).
    """

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, which is canonical WITHIN a level but
    #: not across them -- walls and targets are static per level and differ
    #: between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Runaway guard on one closure; see `_Field`. The largest this game reaches
    #: is L3's 18,193 states.
    field_cap: int = 2_000_000

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.target_ids = set(g.resolve_object_name("target"))
        #: `_Board`s and `_Field`s by STATIC signature (see `read`), not by level
        #: index, so a board is shared by every state of its level and the
        #: closure is kept across queries.
        self._boards: dict = {}
        self._fields: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "TimeReversedMicrobanExpert reads an exact distance field; "
            "heuristic is unused")

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid.

        Boards are cached by their STATIC signature (dimensions, walls, targets),
        so the edge table is built once per level however many states are read
        from it."""
        h, w = eng.height, eng.width
        walls, targets = [], []
        mask = 0
        player = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.target_ids:
                    targets.append(i)
                if cell & self.crate_ids:
                    mask |= 1 << i
                if cell & self.player_ids:
                    player = i
        sig = (h, w, tuple(walls), tuple(targets))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, targets)
        return board, (player, mask)

    def field(self, board: _Board) -> _Field:
        got = self._fields.get(board.sig)
        if got is None:
            got = self._fields[board.sig] = _Field(board, self.field_cap)
        return got

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples for the player (0) and the crates (1).

        Built from the MODEL's reading rather than from raw object ids so the key
        is exactly the two things a state consists of -- which is also what the
        ``plan_cache_path`` signature is stored as, so a cached plan is matched
        against the board it was solved from and an edited level is a miss rather
        than a wrong plan."""
        _board, (p, mask) = self.read(eng)
        if p is None:
            return frozenset()
        w = eng.width
        return frozenset(
            {(p // w, p % w, 0)}
            | {(i // w, i % w, 1)
               for i in range(eng.height * w) if (mask >> i) & 1})

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if state[0] is None:                 # no player on the board
            return None
        if board.won(state):
            return Plan([], [])
        return self.field(board).plan(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class TimeReversedMicrobanSolver(PSAStarSolver):
    game_id = "puzzlescript_time_reversed_microban"
    game_name = GAME_NAME
    expert_cls = TimeReversedMicrobanExpert

    #: `games/ps:time_reversed_microban/ps:time_reversed_microban.py` is a plain
    #: passthrough -- it builds the adapter and nothing else, and the rendering
    #: fix is in the .txt, which both paths read. Set this if that wrapper ever
    #: grows a patch.
    game_module_id = ""

    #: Unused: `TimeReversedMicrobanExpert._search` never calls `_astar`. Left at
    #: the base values so nothing reads a lie off them;
    #: `TimeReversedMicrobanExpert.field_cap` is the knob that bounds the
    #: closures.
    weight = 1
    node_cap = 400_000

    #: Room for the longest plan (100 presses on L4, 98 on L2) plus the
    #: detours. The adapter's own 200-press per-level budget is the real ceiling
    #: and it is reset by the `set_level` that ends the exploration prefix, so
    #: the plan starts it from zero; this is set under it so the expert stops
    #: itself rather than being cut off by a GAME_OVER.
    max_steps = 190

    #: Every level is solved; nothing is skipped.
    skip_levels = frozenset()

    #: On top of the RESET prefix: about one press in sixteen is a random legal
    #: alternative, after which the expert re-plans from wherever it landed --
    #: the taken action is the mistake and ``optimal`` is the recovery, which is
    #: the signal a policy needs after its own error.
    #:
    #: Detours are safe here ONLY because the field's deadness test is exact and
    #: free: `record_level` offers each alternative to `expert.plan` first and
    #: keeps it only if the board can still be won. On a game this full of dead
    #: ends (78% of L0's reachable boards) an unchecked detour would brick most
    #: attempts.
    #:
    #: The RATE is low for a reason that is specific to a pull sokoban and was
    #: measured rather than guessed: **a detour here is expensive.** In a
    #: reversible game an off-plan press costs two (do it, undo it), which is
    #: why ps:swap_sokoban can run at 0.12; here a detour that drags a crate a
    #: square the wrong way is undone only by walking round to the far side and
    #: dragging it back, and the measured mean is ~6 presses. That multiplies
    #: against L2 and L4, which start at 98 and 100 presses, and the adapter
    #: cuts a level off at 200. Swept over 60 seeds x 10 levels, 0.10 rejects 3
    #: seeds in 60 with a worst recording of 206 presses; 0.06 rejects none, and
    #: over 100 generated episodes it rejects none either (100 seeds, 100
    #: attempts), its worst level recording is 148 presses -- 52 of margin --
    #: and 5.5% of the 54,518 expert steps carry a taken action that is not in
    #: their own optimal set. A rejected seed costs an extra attempt and nothing
    #: else (the WIN-only contract simply drops it), so this is a throughput
    #: knob, not a correctness one.
    epsilon = 0.06

    def prepare_expert(self, game, expert) -> None:
        """Enumerate every level's closure before `discover_solvable` asks for
        it -- the same work either way, but it fills the disk cache in one pass
        and makes the startup cost visible as startup."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    """The solver, its adapter and an expert -- without planning anything.

    The reports below each want a different slice (``--audit`` needs no plan at
    all, ``--selfcheck`` needs only the model), so planning is left to whoever
    asks for it rather than paid on every entry point."""
    solver = TimeReversedMicrobanSolver()
    game = solver.make_game(seed)
    return solver, game, TimeReversedMicrobanExpert(game,
                                                    node_cap=solver.node_cap)


def _ascii(board: _Board, state) -> str:
    """A model state as ASCII, so a failing check can print the board it failed
    on rather than a pair of integers."""
    p, mask = state
    out = []
    for r in range(board.h):
        line = ""
        for c in range(board.w):
            i = r * board.w + c
            on_target = (board.tmask >> i) & 1
            if i in board.wall:
                line += "#"
            elif i == p:
                line += "p" if on_target else "P"
            elif (mask >> i) & 1:
                line += "@" if on_target else "O"
            else:
                line += "*" if on_target else "."
        out.append(line)
    return "\n".join(out)


def _start(game, expert, level: int):
    """``(board, state)`` at a level's start, engine left seated there."""
    game.set_level(level)
    return expert.read(game._engine)


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once."""
    _solver, game, expert = _new()
    eng = game._engine
    total = ties = bad = 0
    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        mask = state[1]
        field = expert.field(board)
        # Force the closure: `plan` below may answer straight from the disk
        # cache, and then the space counts printed here would all read zero.
        field.get(state)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            bad += 1
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tie_steps = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        ties += tie_steps
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"  L{level:2d}: {board.h:2d}x{board.w:2d}  "
              f"{len(board.free):2d} floor  {bin(mask).count('1')} crates  "
              f"{len(plan):3d} presses  win={won}  "
              f"(budget {game._max_steps}, {room})  "
              f"{tie_steps:3d} steps with a tie set  "
              f"[space {field.states}, {len(field.dead)} of it already lost]")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _forward(board: _Board, state, limit: int) -> "int | None":
    """Presses to a win from ``state``, by a plain FORWARD BFS, or None past
    ``limit``.

    Shares nothing with `_Field` but `_Board.step` itself, which is the point: it
    is the independent answer every optimality claim below is checked against.
    Depth-bounded, so it is also the cheap way to ask "is there a win within k"
    for one successor without labelling anything."""
    if board.won(state):
        return 0
    seen = {state}
    queue = deque([(state, 0)])
    while queue:
        cur, d = queue.popleft()
        if d >= limit:
            continue
        for di in range(4):
            nxt = board.step(cur, di)
            if nxt == cur or nxt in seen:
                continue
            if board.won(nxt):
                return d + 1
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None


def _selfcheck(walk_presses: int = 600, samples: int = 80,
               verbose: bool = True) -> int:
    """Six things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's.** A seeded random walk on every
       level, comparing `_Board.step`'s board against the engine's grid after
       EVERY press -- including the presses that do nothing, which is where a
       collision-layer mistake hides. The walk is what measures the mechanic
       listed in the module docstring: it pulls crates, walks into walls, presses
       into crates with a wall behind, with the board edge behind and with a
       second crate behind, and drags crates off their targets.
    2. **The game really is DIRECTED.** The same walk counts, at every state, how
       many of the four presses that CHANGE the board are undone by the opposite
       press. In a reversible game (ps:swap_sokoban) that is all of them; here
       every pull is one-way, and the count is the measurement that says so --
       which is the fact the RESET prefix, the checked detours and the choice of
       a forward closure over a backward sweep all rest on.
    3. **The plans are shortest.** A plain forward BFS to the first win, which is
       shortest by construction and shares no code with the field, must return
       the field's plan length on every level.
    4. **The tie sets are exact.** Every step's optimal set is re-derived by
       brute force -- a fresh depth-bounded forward BFS from each of the four
       successors -- and must match the field's answer exactly. This game is
       small enough to afford that on EVERY level, so nothing here is sampled.
    5. **The field's won set is the win condition.** The states the closure
       labels 0 must be exactly the reachable states `_Board.won` accepts, and
       every one of them must be in `goal_states`'s independent enumeration of
       the win condition. A field that missed a goal would field every distance
       too high and nothing else would notice.
    6. **Recovery is answered from ANY board, and its "no" is right too.** This
       is the one that matters at generation time: the exploration prefix and the
       epsilon detours both leave the board off every shortest path, and on this
       game they often leave it unwinnable. From ``samples`` states reached by
       seeded random presses, the field must either return a plan whose length an
       independent forward BFS agrees with AND which wins when replayed through
       the interpreter, or declare the board dead -- in which case the
       independent BFS, run to the diameter of the whole space, must agree that
       there is no win. Both halves are checked; a field that called live boards
       dead would silently turn detours into resets.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0

    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        rng = random.Random(f"time_reversed_microban:selfcheck:{level}")
        drift = changing = reversible = 0
        for _ in range(walk_presses):
            here = snapshot(eng)
            for di in range(4):
                eng.step(DIRS[di])
                moved = eng.grid != here
                if moved:
                    changing += 1
                    eng.step(DIRS[di ^ 1])       # up<->down, left<->right
                    reversible += eng.grid == here
                restore(eng, here)
            di = rng.randrange(4)
            eng.step(DIRS[di])
            state = board.step(state, di)
            _b, live = expert.read(eng)
            if live != state or eng.check_win() != board.won(state):
                drift += 1
                break
        bad += drift
        if verbose:
            print(f"  L{level:2d}: model {'MATCHES' if not drift else 'DIVERGES from'}"
                  f" the interpreter over {walk_presses} random presses; "
                  f"{reversible}/{changing} changing presses are undone by the "
                  f"opposite press")

    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        shortest = _forward(board, state, len(plan) + 1 if plan else 400)
        ok = plan is not None and shortest == len(plan)
        bad += not ok
        if verbose:
            print(f"  L{level:2d}: field {len(plan) if plan else None} presses vs "
                  f"forward BFS {shortest} -- "
                  f"{'SHORTEST' if ok else 'NOT SHORTEST'}")

    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        sets = getattr(plan, "optsets", None) or []
        mismatched = 0
        cur = state
        for i, direction in enumerate(plan):
            rest = len(plan) - i
            truth = []
            for di in range(4):
                nxt = board.step(cur, di)
                if nxt == cur:
                    continue
                if _forward(board, nxt, rest - 1) == rest - 1:
                    truth.append(DIRS[di])
            if sorted(truth) != sorted(sets[i]):
                mismatched += 1
            cur = board.step(cur, DIRS.index(direction))
        bad += mismatched
        if verbose:
            print(f"  L{level:2d}: {len(plan)} tie sets "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'}"
                  f" brute force")

    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        field = expert.field(board)
        field.get(state)     # force the closure; `plan` would hit the disk cache
        crates = bin(state[1]).count("1")
        listed = set(board.goal_states(crates))
        labelled = {s for s, d in field.dist.items() if d == 0}
        by_predicate = {s for s in field.dist} | field.dead
        by_predicate = {s for s in by_predicate if board.won(s)}
        wrong = (labelled ^ by_predicate) | (labelled - listed)
        bad += len(wrong)
        if verbose:
            print(f"  L{level:2d}: {len(field.dist) + len(field.dead):6d} "
                  f"reachable states, {len(labelled)} of them won "
                  f"({len(field.dead)} already lost); "
                  f"{len(wrong)} disagreement(s) with the win condition")

    for level in range(game.n_levels):
        rng = random.Random(f"time_reversed_microban:recovery:{level}")
        board, state = _start(game, expert, level)
        field = expert.field(board)
        field.get(state)     # force the closure; `plan` would hit the disk cache
        diameter = max(field.dist.values()) + 1
        wrong = dead = won_after = 0
        for _ in range(samples):
            game.set_level(level)
            for _ in range(rng.randrange(1, 25)):
                eng.step(DIRS[rng.randrange(4)])
                if eng.check_win():
                    break
            if eng.check_win():
                continue                 # the walk already won; nothing to plan
            board, state = expert.read(eng)
            plan = expert._search(eng)
            truth = _forward(board, state, diameter)
            if plan is None:
                dead += 1
                if truth is not None:
                    wrong += 1
                    print(f"    L{level} called DEAD but a win is {truth} "
                          f"presses away:\n{_ascii(board, state)}")
                continue
            if truth != len(plan):
                wrong += 1
                continue
            for direction in plan:
                eng.step(direction)
            if eng.check_win():
                won_after += 1
            else:
                wrong += 1
        bad += wrong
        if verbose:
            print(f"  L{level:2d}: {samples} random boards -- {won_after} "
                  f"re-planned to a WIN in the interpreter, {dead} proved "
                  f"already lost, {wrong} WRONG")
    return bad


# ---------------------------------------------------------------------------
# The interpreter proofs
# ---------------------------------------------------------------------------

def _statics(eng, expert):
    """The level's static layer -- every cell with the player and the crates
    stripped out -- so a state key can be SEATED back onto the engine instead of
    keeping a snapshot of every board it reaches.

    Taken from the live grid rather than assembled from object ids, so it carries
    whatever the level actually holds (background under walls included) and a
    seated board is byte-identical to the one the level loaded."""
    dynamic = expert.player_ids | expert.crate_ids
    return [[cell - dynamic for cell in row] for row in eng.grid]


def _seat(eng, static, expert, state) -> None:
    """Put ``state`` back on the engine. The inverse of `_read_state`; that it
    IS the inverse is checked before either is used (see `_engine`)."""
    p, mask = state
    w = len(static[0])
    player = next(iter(expert.player_ids))
    crate = next(iter(expert.crate_ids))
    grid = []
    for r, row in enumerate(static):
        out = []
        for c, cell in enumerate(row):
            i = r * w + c
            new = set(cell)
            if (mask >> i) & 1:
                new.add(crate)
            if i == p:
                new.add(player)
            out.append(new)
        grid.append(out)
    eng.grid = grid
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _read_state(eng, expert):
    """``(player, crate bitmask)`` straight off the engine grid. A grid scan and
    nothing else -- no `_Board`, no mechanic -- so the sweeps below owe the native
    model nothing."""
    w = eng.width
    mask = 0
    player = None
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if cell & expert.crate_ids:
                mask |= 1 << (r * w + c)
            if cell & expert.player_ids:
                player = r * w + c
    return (player, mask)


def _engine(verbose: bool = True) -> int:
    """The proof that owes nothing to the native model: the same two sweeps,
    driven by the REAL INTERPRETER.

    Forward BFS from the level start by pressing actual buttons, keying each
    board by what is on the grid, stopping at the depth the first `check_win`
    appears -- which fixes the shortest length by construction -- with every edge
    recorded; then a backward sweep from the won boards over those edges, which
    gives the exact optimal SET at every state on a shortest path. No `_Board`
    call of any kind participates: successors come from `PSEngine.step` and the
    goal test is `check_win`.

    States are re-seated from their keys rather than snapshotted, which is what
    keeps the balls in megabytes; the seating is checked to be the exact inverse
    of the reading, and a seated board is checked to step identically to a
    snapshot-restored one, before any of it is trusted."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        _board, _state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        game.set_level(level)
        static = _statics(eng, expert)
        start = _read_state(eng, expert)

        # The decoder is the inverse of the encoder, and a seated board behaves
        # like a restored one -- both checked here rather than assumed, because a
        # decoder that is not exactly the inverse enumerates a DIFFERENT game.
        original = snapshot(eng)
        _seat(eng, static, expert, start)
        if eng.grid != original or _read_state(eng, expert) != start:
            print(f"  L{level:2d}: SEATING IS NOT THE INVERSE OF READING")
            bad += 1
            continue
        for di in range(4):
            restore(eng, original)
            eng.step(DIRS[di])
            expected = snapshot(eng)
            _seat(eng, static, expert, start)
            eng.step(DIRS[di])
            if eng.grid != expected:
                print(f"  L{level:2d}: a seated board steps differently ({DIRS[di]})")
                bad += 1
        _seat(eng, static, expert, start)

        depth = {start: 0}
        queue = deque([start])
        rev: dict = {}
        won_states: set = set()
        d_star = None
        while queue:
            cur = queue.popleft()
            d = depth[cur]
            if d_star is not None and d >= d_star:
                break
            for di in range(4):
                _seat(eng, static, expert, cur)
                eng.step(DIRS[di])
                nxt = _read_state(eng, expert)
                if nxt == cur:
                    continue                    # a press that did nothing
                rev.setdefault(nxt, []).append(cur)
                if nxt in depth:
                    continue
                depth[nxt] = d + 1
                if eng.check_win():
                    won_states.add(nxt)
                    if d_star is None:
                        d_star = d + 1
                    continue                    # a won board is terminal
                queue.append(nxt)
        if d_star is None:
            print(f"  L{level:2d}: the interpreter found no win")
            bad += 1
            continue

        dist = {s: 0 for s in won_states}
        back = deque(won_states)
        while back:
            cur = back.popleft()
            for prev in rev.get(cur, ()):
                if prev not in dist:
                    dist[prev] = dist[cur] + 1
                    back.append(prev)

        # Read the plan and its tie sets back out of the interpreter's own field.
        epresses, esets = [], []
        cur = start
        while dist.get(cur):
            rest = dist[cur]
            best = []
            for di in range(4):
                _seat(eng, static, expert, cur)
                eng.step(DIRS[di])
                nxt = _read_state(eng, expert)
                if nxt != cur and dist.get(nxt) == rest - 1:
                    best.append((di, nxt))
            epresses.append(DIRS[best[0][0]])
            esets.append(sorted(DIRS[di] for di, _ in best))
            cur = best[0][1]

        same_len = len(epresses) == len(plan) == d_star
        same_sets = esets == [sorted(s) for s in plan.optsets]
        bad += (not same_len) + (not same_sets)
        if verbose:
            print(f"  L{level:2d}: {len(depth):6d} engine boards within d*, "
                  f"{d_star:3d} presses vs the field's {len(plan):3d} -- "
                  f"{'AGREE' if same_len else 'DISAGREE'}; tie sets "
                  f"{'AGREE' if same_sets else 'DISAGREE'}")
        game.set_level(level)
    return bad


def _enumerate(levels=tuple(range(10)), verbose: bool = True) -> int:
    """Walk the interpreter's WHOLE reachable space with the shared
    `StateGraph.build` and solve the graph exactly.

    A fourth derivation, and the one that shares no algorithm with anything here
    either -- `StateGraph` is `solvers/common/ps_astar.py`'s, written for other
    games. It buys two things `--engine` cannot: plans that are shortest against
    the entire space rather than against a ball, and an independent count of the
    boards that CANNOT win, which is the claim `_Field`'s deadness test makes and
    the epsilon detours rest on.

    Affordable on every level here -- the biggest space in the game is 18k
    states -- which is unusual for this family and is the whole reason the expert
    can enumerate too."""
    from solvers.common.ps_astar import StateGraph                     # noqa: PLC0415

    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in levels:
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        field = expert.field(board)
        field.get(state)     # force the closure; `plan` above may hit the cache
        won = sum(1 for d in field.dist.values() if d == 0)
        game.set_level(level)
        static = _statics(eng, expert)
        graph = StateGraph.build(
            eng, lambda e: _read_state(e, expert), list(DIRS),
            node_cap=2_000_000,
            decode=lambda e, k: _seat(e, static, expert, k))
        if graph is None:
            print(f"  L{level:2d}: past the enumeration cap")
            bad += 1
            continue
        eplan = graph.plan(list(DIRS))
        stranded = [k for k in graph.succ if k not in graph.dist]
        same_len = eplan is not None and len(eplan) == len(plan)
        same_sets = eplan is not None and (
            [sorted(s) for s in eplan.optsets]
            == [sorted(s) for s in plan.optsets])
        # The interpreter's own count of hopeless boards, against the field's --
        # the claim `_Field`'s deadness test makes, checked against a space that
        # was walked by pressing real buttons. `StateGraph` keys a WIN as an EDGE
        # rather than as a state, so its space is the field's minus the won
        # boards, and that identity is asserted too rather than left to eyeball.
        same_dead = len(stranded) == len(field.dead)
        same_size = (len(graph.succ) + won
                     == len(field.dist) + len(field.dead))
        bad += ((not same_len) + (not same_sets) + (not same_dead)
                + (not same_size))
        if verbose:
            print(f"  L{level:2d}: {len(graph.succ):6d} engine states "
                  f"({graph.steps} presses), "
                  f"{len(eplan) if eplan else None} presses vs the field's "
                  f"{len(plan)} -- {'AGREE' if same_len else 'DISAGREE'}; "
                  f"tie sets {'AGREE' if same_sets else 'DISAGREE'}; "
                  f"{len(stranded)} state(s) that cannot win vs the field's "
                  f"{len(field.dead)} -- "
                  f"{'AGREE' if same_dead else 'DISAGREE'}; "
                  f"space +{won} won "
                  f"{'AGREE' if same_size else 'DISAGREE'}")
    return bad


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

#: Every cell stack a board of this game can hold. Background is on its own layer
#: and is everywhere; Target has a layer to itself; Player, Wall and Crate share
#: the third, so at most one of them is in a cell.
#:
#: ``wall + target`` is absent on purpose and is not an omission: the legend has
#: no character that places one, no level does, and a target buried under a wall
#: would make its level unwinnable rather than misread. It is also the one pair
#: this game cannot draw apart -- Wall is opaque.
_COMPOSITIONS = [("wall",), (), ("target",), ("player",), ("player", "target"),
                 ("crate",), ("crate", "target")]

#: `_render_frame`'s letterbox colour. A composition that renders as a uniform
#: frame of it is invisible against the pad (the ps:stand_iii lesson) -- a live
#: risk here, where Target is ``orange`` and Wall is ``BROWN DARKBROWN`` and ARC
#: has one palette entry for brown and orange both.
_PAD = 5


def _comp_name(comp) -> str:
    return "+".join(comp) or "floor"


def _audit(verbose: bool = True) -> int:
    """Three passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the ten boards use (5, 7, 8
    and 9 px; the size list is derived from the levels rather than assumed, so an
    edited level is covered). Whole 64x64 frames of uniform boards are compared
    rather than one cell out of a mixed board: `_render_frame` upscales and
    centre-pads, so slicing a cell by ``cell_px`` arithmetic reads the wrong
    pixels (the ps:explod lesson). Two uniform boards render identically iff
    their cells do.

    This is the pass that fails on the shipped .txt: ``player`` and
    ``player+target`` were the same frame at every size. See the header comment
    in ``data/puzzlescript_games/Time-reversed_Microban.txt``.

    **Pass 2 -- nothing is the letterbox.** No composition may render as a
    uniform frame of the pad colour, or it would have no seam against the border
    of the picture.

    **Pass 3 -- the group.** The game is augmented with a frame rotation AND both
    flips, so a board is drawn at any of 16 presentations, and the sprites are
    not all invariant under them (the Player is a little person, the Wall a
    masonry pattern). That is fine -- the whole frame is transformed together, so
    a turned board is a real board -- as long as no transform of one composition
    lands on a DIFFERENT one, which is what this checks: all 16 transforms of
    each, against all others. It runs on SQUARE boards, because a transform of a
    non-square board also moves the letterbox and every comparison would pass for
    the wrong reason.
    """
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, comp):
        eng.height, eng.width = h, w
        cell = {idx["background"]} | {idx[o] for o in comp}
        eng.grid = [[set(cell) for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g)).copy()

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {c: shoot(h, w, c) for c in _COMPOSITIONS}
        clashes = [(a, b) for a, b in itertools.combinations(_COMPOSITIONS, 2)
                   if np.array_equal(shots[a], shots[b])]
        pad = [c for c in _COMPOSITIONS
               if np.all(shots[c] == _PAD)]
        bad += len(clashes) + len(pad)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, {'OK' if not clashes and not pad else 'CLASH'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {_comp_name(a)}  ==  {_comp_name(b)}")
        for c in pad:
            print(f"      INVISIBLE AGAINST THE LETTERBOX: {_comp_name(c)}")

    def transforms(frame):
        for k in range(4):
            turned = np.rot90(frame, k=k)
            for hf in (False, True):
                for vf in (False, True):
                    out = turned
                    if hf:
                        out = np.fliplr(out)
                    if vf:
                        out = np.flipud(out)
                    yield (k, hf, vf), np.ascontiguousarray(out)

    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {c: shoot(n, n, c) for c in _COMPOSITIONS}
        clashes = [(a, b, t) for a, b in itertools.permutations(_COMPOSITIONS, 2)
                   for t, turned in transforms(shots[a])
                   if np.array_equal(turned, shots[b])]
        invariant = [_comp_name(c) for c in _COMPOSITIONS
                     if all(np.array_equal(t, shots[c])
                            for _k, t in transforms(shots[c]))]
        bad += len(clashes)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(clashes)} composition(s) "
                  f"whose transform is another's art; group-invariant: "
                  f"{', '.join(invariant)}")
        for a, b, t in clashes:
            print(f"      {_comp_name(a)} under {t} == {_comp_name(b)}")
    return bad


def _symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for the rotation + flip augmentation. The structural
    argument is in `PuzzleScriptAdapter._FLIP_GAMES` -- one rule, written with the
    relative force ``>``, and a pull that can never contest a square -- but Gobble
    Rush's chirality hid inside exactly that kind of argument, so it is measured.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (crates dragged off their targets, crates jammed
    flat against walls where nothing can pull them out again, two crates back to
    back) and it presses the unbound ACTION key as well."""
    solver, game, expert = _new()
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    def drive(g, level, presses):
        g.set_level(level)
        out = [np.asarray(g._current_frame)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame else g._current_frame))
        return out

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    ref, seen, bad = {}, set(), 0
    for seed in range(400):
        if len(seen) == 16 and seed > 80:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"time_reversed_microban:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (
                    ("plan", [screen_action(d, *k) for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *k) for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"    seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[key] = frames
                    continue
                if any(not np.array_equal(transform(a, *k), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"    seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--engine" in sys.argv:
        violations = _engine()
        print(f"engine sweeps: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--enumerate" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--enumerate") + 1:]
                if a.isdigit()]
        violations = _enumerate(tuple(args) or tuple(range(10)))
        print(f"whole-space enumeration: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(TimeReversedMicrobanSolver.main())
