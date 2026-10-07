"""Generate Phase-1 training data for the PuzzleScript game ps:swap_sokoban
("Swap Sokoban", Franklin P. Dyer).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the exact distance FIELD that model is solved
with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_swap_sokoban",
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
Ten levels of what looks like a sokoban and is not one. The entire rule list is
a single line:

    [ > Player | Crate ] -> [ Crate | Player ]

The player and the crate it walks into **swap places**. Neither side of the
right hand side carries a force, so the swap is done by the rule itself and the
movement phase then moves nothing; and because the crate lands on the square the
player is vacating, that square is free by construction, so **a swap can never
be blocked**. Walking into a crate always works, whatever is behind it.

Three consequences, and they are the whole game:

* **Crates move TOWARDS the player, one step, and the player ends up where the
  crate was.** There is no push in this game and no pull in the usual sense
  either -- the two bodies exchange, so after moving a crate the player is
  standing on the crate's old square with the crate directly behind it.
* **There are no deadlocks.** A sokoban's corner traps come from a crate needing
  a free square behind it; here the free square is the one the player just left,
  so a crate in a corner is moved out of it exactly as easily as any other. What
  costs presses is the WALK: to shove a crate one more step along a straight
  line the player has to get to the far side of it, which is four moves around
  in the open, whereas turning a corner with it costs two.
* **Every press is reversible.** A walk onto an empty square is undone by the
  opposite walk (the square behind is now the empty one), and a swap is undone by
  pressing back into the crate you just swapped with. So the reachable state
  space is an UNDIRECTED graph, which is what makes the distance field below a
  single backward sweep, and what makes this game impossible to strand: every
  board an exploration prefix can reach is still winnable, because the prefix
  itself can be walked backwards. ``--selfcheck`` measures the reversibility on
  the interpreter rather than taking it from the rule text, because everything
  else here rests on it.

The win is ``All Target on Crate`` -- every target square has a crate on it.
Extra crates would be free to sit anywhere; on all ten shipped levels the counts
match exactly.

Boards run from 6x8 to 12x8, with 19 to 54 floor squares and 2 to 6 crates.
Level 7 (index 6) is the interesting one: a 6x5 open room with six crates and
six targets interleaved in a checkerboard, which is 14.2 million states -- far
too many to enumerate, and it does not have to be, see below.

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`) plus an exact distance-to-win
field (`_Field`). The shared `PSSokobanExpert` does not apply at all: its macro
is "walk to a push square, then push", which presumes pushes, a free square
behind the crate, and a corner-deadlock test -- none of which exist here, and its
deadlock test would prune states that are perfectly alive.

`_Field` is ONE breadth-first sweep, BACKWARDS from the win:

  * the goal states are enumerated directly -- crates on every target, any spare
    crates anywhere, the player on any free square -- rather than searched for.
    That is 16 to 52 states per level;
  * BFS out from them over `_Board.step`, which is legitimate as a BACKWARD sweep
    precisely because every press is reversible: ``s`` is a predecessor of ``t``
    exactly when it is a successor of it. The result is the true shortest
    presses-to-win for every state it reaches.

The sweep is LAZY: it expands one BFS layer at a time and stops the moment the
state being asked about is labelled, so a level pays only for the ball around
its goal that it actually needs, and it is kept between queries so nothing is
re-derived. All ten level starts cost 0.04 seconds; thirty full episodes
(prefixes, detours and re-plans included) leave level 6's field at 156k states
out of that 14.2M space and every other level's under 101k. The reason the big
level needs no enumeration is that nothing ever asks about the far side of
it.

The field is then read directly for everything else:

  * the PLAN is "walk down the gradient": at distance ``d``, any press whose
    successor is at ``d - 1``;
  * the OPTIMAL SETS are all of those presses, exactly, with no inference;
  * RECOVERY is the same lookup from wherever the board happens to be, which is
    the whole `supports_recovery` contract -- and by reversibility it can never
    come back empty on a reachable board. ``--selfcheck``'s recovery pass finds
    0 dead boards out of 800, which is a prediction of the argument above, not a
    lucky sample.

Shortest, and how that is known
-------------------------------
Every plan is provably shortest, and the proof is three independent derivations
agreeing on all ten levels:

  * this file's backward field;
  * a plain FORWARD BFS to the first win over the same model, which is shortest
    by construction and shares no line with the backward sweep (``--selfcheck``
    pass 3); its per-step tie sets are re-derived a second time by brute force
    (pass 4);
  * the same two sweeps driven by the REAL INTERPRETER -- forward BFS from the
    level start by pressing actual buttons until a win appears, edges recorded,
    then backward over those edges (``--engine``). No `_Board` call of any kind
    participates: the successor of a board is whatever `PSEngine.step` makes of
    it and the goal test is `check_win`. It agrees on all ten lengths and on
    all 271 tie sets.

``--enumerate`` adds a fourth for the six levels small enough to afford it: the
shared `StateGraph.build` walks the interpreter's ENTIRE reachable space and
solves the graph exactly, which also confirms the reversibility claim's
consequence -- every reachable state there is winnable, 0 exceptions.

Recovery
--------
`supports_recovery` with the family's RESET-mode prefix, plus ``epsilon = 0.12``
detours inside the expert replay. The expert re-plans from the LIVE board (a
field lookup at whatever state the engine is in), so both kinds of perturbation
are answered exactly: the taken action is the mistake and ``optimal`` is the
recovery, which is the signal a policy needs after its own error.

This game is unusually good at that, and for a stated reason: reversibility means
a detour can never brick a board, so `record_level`'s "is this still winnable"
probe always passes and the detours actually happen instead of being silently
skipped. It also means the RESET at the end of the exploration prefix is the only
RESET a recording ever contains.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the frame rotation
(``rotation_k`` in 0..3) and the two flips, with their matching directional
action remap. 10 levels x 16 presentations = 160.

``Swap_Sokoban`` is in `PuzzleScriptAdapter._FLIP_GAMES`; the argument is
recorded there and ``--symmetry`` measures it -- every level's plan AND a seeded
200-press random walk (which shoves crates into walls, drags them back off their
targets and presses the unbound ACTION key) replayed at all 16 presentations,
requiring every frame to be the exact transform of the unaugmented one.

Rendering
---------
One sprite had to change, and it was the goal itself: Target shipped as a
dark-blue ring on rows/cols 1-3, which is exactly the nine pixels the Player
paints over, so ``player on target`` rendered as a player on bare floor. In a
game whose bodies SWAP, the player is on and off the target squares constantly.
The fix -- the ring plus the four corners, which the Player leaves transparent
and the Crate covers -- is the one Stickyban got for the identical bug from the
identical sokoban template; the full statement is the header comment in
``data/puzzlescript_games/Swap_Sokoban.txt``.

``--audit`` renders every cell COMPOSITION a board of this game can hold -- wall,
floor, target, player, crate, and the player and the crate on a target -- as a
whole 64x64 frame of a uniform board, at every cell size the ten levels render at
(5, 7, 8 and 9 px), and requires them pairwise distinct. Whole frames rather than
one cell sliced out of a mixed board: `_render_frame` upscales and centre-pads,
so slicing by ``cell_px`` arithmetic reads the wrong pixels (the ps:explod
lesson). ``wall + target`` is absent on purpose: the legend has no character for
it, no level places one, and a target under a wall would make the level
unwinnable rather than misread.

Usage (run from the repo root):
    python solvers/generate_swap_sokoban_training.py --episodes 200 \
        --out data/training_multi_level/swap_sokoban

    python solvers/generate_swap_sokoban_training.py --plans      # level report
    python solvers/generate_swap_sokoban_training.py --selfcheck  # model + optimality
    python solvers/generate_swap_sokoban_training.py --engine     # interpreter proof
    python solvers/generate_swap_sokoban_training.py --enumerate  # whole-space proof
    python solvers/generate_swap_sokoban_training.py --audit      # rendering
    python solvers/generate_swap_sokoban_training.py --symmetry   # augmentation
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

GAME_NAME = "Swap_Sokoban"

#: Engine directions, in the order ties are broken. There is no ACTION rule in
#: this game (no rule has ``action`` on a left-hand side), so the X button is not
#: a move and branching on it would double every sweep for nothing. The
#: exploration prefix still presses it -- a live agent has that button -- which is
#: why ``--symmetry``'s random walk presses it too.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Disk cache of every level's start plan AND its optimal-action sets. The whole
#: game's fields take well under a second, so this is not the load-bearing cache
#: it is on the big ps: sokobans -- it is here so a `parallelize_generator`
#: fan-out shares one derivation rather than repeating it on every core. Delete
#: the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "swap_sokoban_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic.

    Cells are flat ``r * w + c`` indices and a STATE is ``(player, crate
    bitmask)`` -- the only two things any rule in this game moves. Walls and the
    target set never change, so they live here rather than in the state, and the
    crate set is a bitmask because the sweeps below key millions of states and a
    Python int is the cheapest exact spelling of a subset of squares.

    THE MECHANIC, as one press. The player tries to step onto ``p + d``:

      * off the board, or a wall: nothing happens at all (there is no rule with a
        wall on its left, so the movement phase simply refuses the move);
      * an empty square: the player moves there;
      * a crate: the two SWAP -- the player takes the crate's square and the
        crate takes the player's. This can never fail. The square the crate
        moves onto is the one the player is leaving, so it is free by
        construction, and whatever stands behind the crate is not part of the
        match.

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
        #: ``edge[cell][di]`` is the square reached by pressing ``DIRS[di]``, or
        #: -1 when that press cannot move the player at all (off the board or
        #: into a wall). Folding the wall test into the table is what keeps
        #: `step` down to a lookup and a bit test.
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
        if q < 0:
            return state
        bit = 1 << q
        if mask & bit:                       # the swap
            return (q, (mask ^ bit) | (1 << p))
        return (q, mask)                     # a plain walk

    def won(self, state) -> bool:
        return (state[1] & self.tmask) == self.tmask

    # -- the win set, enumerated directly ------------------------------------
    def goal_states(self, crates: int):
        """Every state this board calls won, given ``crates`` crates on it.

        Enumerated rather than searched for, which is what lets `_Field` sweep
        BACKWARDS from the win. ``All Target on Crate`` constrains only the
        target squares, so a board with spare crates has a goal state for every
        placement of them; on all ten shipped levels the crate count equals the
        target count and the spare loop runs once with an empty combination.

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
    """Exact presses-to-win for every state, by ONE lazy backward sweep.

    WHY A BACKWARD SWEEP IS LEGAL HERE. Every press of this game is reversible
    (see the module docstring: a walk is undone by the opposite walk, a swap by
    pressing back into the crate you just swapped with), so the reachable space
    is an UNDIRECTED graph and `_Board.step` enumerates predecessors exactly as
    well as it enumerates successors. A breadth-first sweep out from the goal
    states therefore labels every state with its true distance to a win --
    no heuristic, no node cap that could quietly bite, and no forward search
    that has to be re-run from every state anything asks about.

    The distance is to the NEAREST win, which is the right answer even though a
    win ENDS the episode: a shortest path to the nearest goal cannot pass
    through another goal, because that one would be nearer.

    WHY IT IS LAZY. Level 6 is a 6x5 open room with six crates -- 14.2 million
    states -- and the whole point is that nothing ever asks about the far side of
    it. `get` expands one BFS layer at a time and returns the instant the state
    it was asked about is labelled, so over thirty generated episodes that level
    settles at 156k states (the ball of radius ~16 around its win) and the other
    nine at 224 to 101k. The sweep is kept between queries, so a re-plan after an
    epsilon detour is ordinarily a dict lookup and never repeats work.

    ``cap`` is a runaway guard, not a budget: past it `get` answers None, which
    `_search` turns into "no plan" and `record_level` turns into a RESET. It is
    never reached on the shipped levels -- ``--selfcheck``'s recovery pass walks
    much further off-plan than a recording ever does (up to 24 random presses,
    800 boards) and the largest field it grows is level 6's 1.55M.
    """

    __slots__ = ("board", "dist", "queue", "cap", "capped", "exhausted")

    def __init__(self, board: _Board, crates: int, cap: int):
        self.board = board
        self.cap = cap
        self.dist: dict = {}
        self.queue: deque = deque()
        for state in board.goal_states(crates):
            self.dist[state] = 0
            self.queue.append(state)
        self.capped = False
        self.exhausted = not self.queue

    def get(self, state) -> "int | None":
        """Presses to a win from ``state``, or None when this board can never be
        won from there (or the cap stopped the sweep short of knowing)."""
        hit = self.dist.get(state)
        if hit is not None:
            return hit
        if self.exhausted or self.capped:
            return None
        board, queue, dist = self.board, self.queue, self.dist
        while queue:
            cur = queue.popleft()
            nd = dist[cur] + 1
            for di in range(4):
                nxt = board.step(cur, di)
                if nxt == cur or nxt in dist:
                    continue
                dist[nxt] = nd
                queue.append(nxt)
            if state in dist:
                return dist[state]
            if len(dist) > self.cap:
                self.capped = True
                return None
        self.exhausted = True
        return None

    def optimal(self, state) -> list:
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``. Ties come out in ``DIRS`` order, which is what makes
        a re-derived plan byte-identical across processes.

        Exact, with nothing inferred: a press is optimal iff its successor is one
        step nearer the win, and both distances are the field's. Every successor
        asked about is already labelled -- BFS assigns the whole ``d - 1`` layer
        before any state at ``d`` -- so this never extends the sweep."""
        rest = self.get(state)
        if not rest:                      # None (unknown / dead) or 0 (already won)
            return []
        out = []
        for di in range(4):
            nxt = self.board.step(state, di)
            if nxt != state and self.dist.get(nxt) == rest - 1:
                out.append((di, nxt))
        return out

    def plan(self, state) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal set
        at every step, or None if this board cannot be won from here."""
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

class SwapSokobanExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Field`.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes, the way `PSEnumExpert` replaces it. Here the
    strategy is "read the live board and walk down the exact distance field", so
    `heuristic` is never called and asserts rather than returning a number
    nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered exactly.
    Unlike the rest of this family it can also be trusted to answer YES: this
    game is reversible, so a reachable board is always still winnable, and a None
    from here would mean the field hit its cap, never that a board was
    genuinely lost.
    """

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, which is canonical WITHIN a level but
    #: not across them -- walls and targets are static per level and differ
    #: between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Runaway guard on one level's backward sweep; see `_Field`. The largest
    #: this game reaches is level 6's -- 156k over thirty generated episodes,
    #: 1.55M under ``--selfcheck``'s deliberately harsher recovery fuzz.
    field_cap: int = 4_000_000

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.target_ids = set(g.resolve_object_name("target"))
        #: `_Board`s and `_Field`s by STATIC signature (see `read`), not by level
        #: index, so a board is shared by every state of its level and the sweep
        #: is kept across queries.
        self._boards: dict = {}
        self._fields: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "SwapSokobanExpert reads an exact distance field; heuristic is unused")

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

    def field(self, board: _Board, crates: int) -> _Field:
        key = (board.sig, crates)
        got = self._fields.get(key)
        if got is None:
            got = self._fields[key] = _Field(board, crates, self.field_cap)
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
        crates = bin(state[1]).count("1")
        return self.field(board, crates).plan(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class SwapSokobanSolver(PSAStarSolver):
    game_id = "puzzlescript_swap_sokoban"
    game_name = GAME_NAME
    expert_cls = SwapSokobanExpert

    #: `games/ps:swap_sokoban/ps:swap_sokoban.py` is a plain passthrough -- it
    #: builds the adapter and nothing else, and the rendering fix is in the .txt,
    #: which both paths read. Set this if that wrapper ever grows a patch.
    game_module_id = ""

    #: Unused: `SwapSokobanExpert._search` never calls `_astar`. Left at the base
    #: values so nothing reads a lie off them; `SwapSokobanExpert.field_cap` is
    #: the knob that actually bounds the sweep.
    weight = 1
    node_cap = 400_000

    #: Room for the longest plan (91 presses on level 5) plus the re-plans an
    #: epsilon detour costs. The adapter's own 200-press per-level budget is
    #: separate and is reset by the `set_level` that ends the exploration prefix,
    #: so the plan starts it from zero.
    max_steps = 150

    #: Every level is solved; nothing is skipped.
    skip_levels = frozenset()

    #: On top of the RESET prefix: one press in eight is a random legal
    #: alternative, after which the expert re-plans from wherever it landed --
    #: the taken action is the mistake and ``optimal`` is the recovery, which is
    #: the signal a policy needs after its own error. Safe at any rate here
    #: because the game is reversible, so a detour can never brick a board (see
    #: the module docstring); `record_level`'s own winnability probe is
    #: consequently never the thing that rejects one.
    epsilon = 0.12

    def prepare_expert(self, game, expert) -> None:
        """Sweep every level's field before `discover_solvable` asks for it --
        the same work either way, but it fills the disk cache in one pass and
        makes the startup cost visible as startup."""
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
    solver = SwapSokobanSolver()
    game = solver.make_game(seed)
    return solver, game, SwapSokobanExpert(game, node_cap=solver.node_cap)


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
                line += "P" if not on_target else "p"
            elif (mask >> i) & 1:
                line += "*" if not on_target else "@"
            else:
                line += "O" if on_target else "."
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
        board, (_p, mask) = _start(game, expert, level)
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
              f"{tie_steps:3d} steps with a tie set")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _forward(board: _Board, state, limit: int) -> "int | None":
    """Presses to a win from ``state``, by a plain FORWARD BFS, or None past
    ``limit``.

    Shares nothing with `_Field` but `_Board.step` itself, which is the point: it
    is the independent answer every optimality claim below is checked against,
    and it does not use reversibility at all -- so it is also the check on the
    property the backward sweep rests on."""
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
       listed in the module docstring: it swaps crates against walls, against
       other crates and against the board edge, and it walks into walls.
    2. **Every press is REVERSIBLE, measured on the interpreter.** At every state
       of that walk, each of the four presses that changes the engine's grid is
       undone EXACTLY by the opposite press. This is the load-bearing check of
       the whole file: `_Field` sweeps backwards from the win over `_Board.step`,
       which is only a predecessor enumeration because of this, and the recovery
       contract ("a reachable board is always still winnable") is the same fact.
    3. **The goal set is the win condition.** On the three smallest levels, the
       states `_Board.goal_states` enumerates are compared against a brute-force
       scan of the ENTIRE state space filtered by `_Board.won`. A goal set that
       is missing states would field every distance too high and nothing else
       would notice.
    4. **The plans are shortest.** A plain forward BFS to the first win, which is
       shortest by construction and shares no code with the backward sweep, must
       return the field's plan length on every level.
    5. **The tie sets are exact.** Every step's optimal set is re-derived by
       brute force -- a fresh depth-bounded forward BFS from each of the four
       successors -- and must match the field's answer exactly. This game is small
       enough to afford that on EVERY level, so nothing here is sampled.
    6. **Recovery is answered from ANY board, not just the start.** This is the
       one that matters at generation time: the exploration prefix and the
       epsilon detours both leave the board off every shortest path. From
       ``samples`` states reached by seeded random presses, the field must return
       a plan whose length an independent forward BFS agrees with and which WINS
       when replayed through the interpreter. By reversibility none of them can
       be dead, so a single dead board here would be a refutation of pass 2, not
       a rare corner.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0

    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        rng = random.Random(f"swap_sokoban:selfcheck:{level}")
        drift = irreversible = 0
        for _ in range(walk_presses):
            # 2 -- reversibility, from this state, on the interpreter.
            here = snapshot(eng)
            for di in range(4):
                eng.step(DIRS[di])
                moved = eng.grid != here
                if moved:
                    eng.step(DIRS[di ^ 1])       # up<->down, left<->right
                    irreversible += eng.grid != here
                restore(eng, here)
            di = rng.randrange(4)
            eng.step(DIRS[di])
            state = board.step(state, di)
            _b, live = expert.read(eng)
            if live != state or eng.check_win() != board.won(state):
                drift += 1
                break
        bad += drift + irreversible
        if verbose:
            print(f"  L{level:2d}: model {'MATCHES' if not drift else 'DIVERGES from'}"
                  f" the interpreter over {walk_presses} random presses; "
                  f"{irreversible} irreversible press(es) out of "
                  f"{4 * walk_presses} probed")

    for level in (7, 0, 8):                      # the three smallest spaces
        board, state = _start(game, expert, level)
        crates = bin(state[1]).count("1")
        space = {state}
        queue = deque([state])
        while queue:                             # the WHOLE reachable space
            cur = queue.popleft()
            for di in range(4):
                nxt = board.step(cur, di)
                if nxt != cur and nxt not in space:
                    space.add(nxt)
                    queue.append(nxt)
        truth = {s for s in space if board.won(s)}
        listed = set(board.goal_states(crates))
        missed = truth - listed
        bad += len(missed)
        if verbose:
            print(f"  L{level:2d}: {len(space):6d} reachable states, "
                  f"{len(truth)} of them won -- goal_states lists "
                  f"{len(listed)} (of which {len(listed & space)} reachable), "
                  f"{len(missed)} MISSED")

    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        shortest = _forward(board, state, len(plan) + 1 if plan else 200)
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
        rng = random.Random(f"swap_sokoban:recovery:{level}")
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
            truth = _forward(board, state, board.n * 8)
            if plan is None:
                dead += 1
                wrong += 1               # reversibility says this cannot happen
                print(f"    L{level} DEAD board (forward BFS says {truth}):\n"
                      f"{_ascii(board, state)}")
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
                  f"re-planned to a WIN in the interpreter, {dead} with no plan, "
                  f"{wrong} WRONG")
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

    Affordable on every level because a forward ball of radius ``d*`` is small
    even where the whole space is not (level 6 is 14.2M states and its ball is
    24.5k). States are re-seated from their keys rather than snapshotted, which
    is what keeps 77k boards in megabytes instead of gigabytes; the seating is
    checked to be the exact inverse of the reading, and a seated board is checked
    to step identically to a snapshot-restored one, before any of it is trusted.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
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


#: Levels whose ENTIRE reachable space the interpreter can be walked over in
#: seconds. The others are 365k to 14.2M states, i.e. minutes to never; pass
#: level numbers on the command line to enumerate them anyway.
_ENUMERABLE = (0, 1, 2, 3, 7, 8)


def _enumerate(levels=_ENUMERABLE, verbose: bool = True) -> int:
    """Walk the interpreter's WHOLE reachable space with the shared
    `StateGraph.build` and solve the graph exactly.

    A fourth derivation, and the one that shares no algorithm with anything here
    either -- `StateGraph` is `solvers/common/ps_astar.py`'s, written for other
    games. It buys two things `--engine` cannot: plans that are shortest against
    the entire space rather than against a ball, and a check on the reversibility
    claim's real consequence -- **every reachable state is winnable**, which is
    what the recovery contract rests on and which is reported per level here.

    Only the six small levels by default; see `_ENUMERABLE`."""
    from solvers.common.ps_astar import StateGraph                     # noqa: PLC0415

    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in levels:
        _board, _state = _start(game, expert, level)
        plan = expert.plan(eng, level)
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
        bad += (not same_len) + (not same_sets) + len(stranded)
        if verbose:
            print(f"  L{level:2d}: {len(graph.succ):7d} engine states "
                  f"({graph.steps} presses), "
                  f"{len(eplan) if eplan else None} presses vs the field's "
                  f"{len(plan)} -- {'AGREE' if same_len else 'DISAGREE'}; "
                  f"tie sets {'AGREE' if same_sets else 'DISAGREE'}; "
                  f"{len(stranded)} state(s) that cannot win")
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


def _comp_name(comp) -> str:
    return "+".join(comp) or "floor"


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the ten boards use (5, 7, 8
    and 9 px; the size list is derived from the levels rather than assumed, so an
    edited level is covered). Whole 64x64 frames of uniform boards are compared
    rather than one cell out of a mixed board: `_render_frame` upscales and
    centre-pads, so slicing a cell by ``cell_px`` arithmetic reads the wrong
    pixels (the ps:explod lesson). Two uniform boards render identically iff
    their cells do.

    This is the pass that fails on the shipped .txt: ``player`` and
    ``player+target`` were the same frame at every size. See the header comment
    in ``data/puzzlescript_games/Swap_Sokoban.txt``.

    **Pass 2 -- the group.** The game is augmented with a frame rotation AND both
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
        bad += len(clashes)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, {'OK' if not clashes else 'CLASH'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {_comp_name(a)}  ==  {_comp_name(b)}")

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
    relative force ``>``, and a swap that can never contest a square -- but Gobble
    Rush's chirality hid inside exactly that kind of argument, so it is measured.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (crates flat against walls, crates dragged back
    off their targets, two crates side by side) and it presses the unbound ACTION
    key as well."""
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
            rng = random.Random(f"swap_sokoban:symmetry:{level}")
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
        violations = _enumerate(args or _ENUMERABLE)
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
    sys.exit(SwapSokobanSolver.main())
