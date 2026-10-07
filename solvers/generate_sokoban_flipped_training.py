"""Generate Phase-1 training data for the PuzzleScript game ps:sokoban_flipped
("Sokoban Flipped", Franklin P. Dyer).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exact distance FIELD that model is
searched with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_sokoban_flipped",
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

The game -- and what "flipped" means
------------------------------------
The whole game is ONE rule:

    [ >  Player | Target ] -> [  >  Player | > Target  ]

Read the object names, not the shape: the thing the player SHOVES is the
``Target``, and the ``Crate`` -- which no rule mentions and which therefore never
moves in the entire game -- is what a Target has to end up on
(``All Target on Crate``). That is the joke in the title, and it is not cosmetic,
because the collision layers are

    Background
    Target, Player, Wall
    Crate

so the two roles really are exchanged relative to a normal sokoban:

* **Crates do not block anything.** They are alone on the top layer, so the
  player walks straight over one, and a shoved Target slides over one without
  stopping. Crates are goal SQUARES that happen to be drawn as boxes -- the
  board's geometry is walls only.
* **Targets block.** They share the player's layer, so a Target is what the
  player collides with (that collision IS the push) and a Target shoved into a
  wall or into a second Target cancels the whole turn, player included. There is
  no chain push: the rule names ``Player`` on its left, so a moving Target never
  re-triggers it.

Everything above was MEASURED against the interpreter (``--selfcheck`` is the
executable form of it), not read off the .txt. The four measurements, on the
shipped boards:

  * player + ``right`` onto a cell holding a crate -> the player is on the crate,
    the crate has not moved (level 0, ``(5,2) -> (5,3)``);
  * player + ``left`` into a Target whose far side is a wall -> nothing moves at
    all, neither piece (level 0);
  * player + ``left`` into a Target whose far side holds a second Target ->
    nothing moves (level 7, the ``T T`` pair);
  * player + ``left`` into a lone Target -> both slide one cell (level 2).

Ten levels, all winnable, none already won at reset, and every board has exactly
as many Targets as Crates -- so the win is a perfect matching, and a level that
starts with some Targets already parked on Crates (0, 1 and 3 do) starts with
part of that matching made. Nothing pins those down: a Target already on a Crate
is shoved off it exactly as easily as any other, which is what makes level 1 the
game's one-idea board: three Targets, two of them already home, and the whole
7-press answer is walking the long way round the room so as not to touch them.

Expert solver
-------------
A NATIVE model of the mechanic above (``_Board``), searched to an exact distance
field rather than heuristically. The reason not to use the shared
`PSSokobanExpert` -- which this game otherwise fits in two attribute lines -- is
measured, not stylistic: its macro A* dedups states by the player's reachable
REGION rather than by its exact cell, and on level 9, whose free floor is one
large region, that mis-costs the plan by 12 presses (83 against the true 71).
Keying the macro search by the exact player cell instead does return all ten
shortest plans, but takes 224s of interpreter steps against this field's 3.6s,
and it still cannot produce the tie sets below.

``_Board.solve`` is two breadth-first sweeps over ``(player, targets)`` states:

  1. FORWARD from the state being planned, stopping at the depth ``d*`` of the
     first win. That fixes the answer's length and collects ``F``, every state
     within ``d*`` presses of the start.
  2. BACKWARD from the winning configuration over INVERTED moves (a walk is
     undone by stepping back onto a free cell; a push is undone by stepping back
     and dragging the Target one cell towards you), pruned to ``F``.

The second sweep needs no successor table -- the inverse moves are enumerated
directly -- which is what keeps it affordable on level 6, whose forward space
passes 4 million states.

Pruning the backward sweep to ``F`` costs nothing that matters: if a state ``s``
lies on any shortest start-to-win path then its whole continuation has forward
distance at most ``d*``, so that continuation never leaves ``F`` and the field's
value at ``s`` is its true distance. States off every shortest path may be
over-estimated, and none of them is ever asked about.

Optimal-action sets
-------------------
The field gives them EXACTLY, with no inference: at a state ``d`` presses from a
win, the optimal presses are every direction whose successor is ``d - 1`` from
one. That is stronger than `PSPushExpert.annotate_walks`' rule of thumb ("a push
is forced, a walk's axis order is free"), and it disagrees with it -- these
boards are corridors, so most walks here have no second shortest route at all,
and the handful of steps that do are the only ones that should be labelled with
a tie. ``--plans`` reports the count; ``--selfcheck`` re-derives every set on the
small levels by brute force (a fresh bounded BFS from each successor) and
requires an exact match. No step ever ships unlabelled (the
always-emit-optimal-targets rule).

Shortest, and how that is known
-------------------------------
All ten plans are provably shortest, and the proof is three independent
derivations agreeing on all ten lengths (22, 7, 61, 24, 16, 38, 21, 58, 23, 71):

  * this file's field (forward BFS + backward BFS), 3.6s;
  * a plain forward BFS to the first win, which is shortest by construction and
    shares no code with the field's backward half, 6s (``--selfcheck``);
  * `PSSokobanExpert`'s macro A* over the REAL interpreter, keyed by the exact
    player cell, under a fully admissible heuristic (no greedy matching) with
    ``exact_goal_test``, 224s.

The third is the one that closes the loop back to the engine: it never uses the
native model at all. The native model itself is separately fuzz-verified against
the interpreter -- ``--selfcheck`` drives a seeded 400-press random walk on every
level and requires the model's board and the engine's grid to agree after every
single press, on all ten.

Recovery
--------
`supports_recovery` with RESET-mode recovery. The expert re-plans from the LIVE
board, not from a stored path: ``_search`` reads whatever the engine currently
holds and runs both sweeps from there, so an exploration prefix that leaves the
board anywhere -- including somewhere a shortest plan would never go -- is
answered exactly. Nothing about the field is tied to a level start; the start is
just the state the cache happens to be warmed at.

Augmentation
------------
Sokoban Flipped's engine state after reset is identical for every seed (levels
are fixed ASCII maps), so the only per-(seed, level) variables are the
presentation augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an
independent horizontal and vertical flip with the matching directional action
remap (`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group. 10 levels x 16 presentations = 160.

The structural argument for the flips is as strong as it gets in this family:
there is exactly ONE rule, it is written with the relative force ``>``, it names
no axis, one press moves at most two bodies and they land on ``p + d`` and
``p + 2d``, which can never be the same square -- so no two moves can contest a
cell and the order the interpreter expands the rule's four directions cannot
decide anything. The win condition is positional. ``--symmetry`` measures it
anyway: every level's plan AND a seeded 200-press random walk (which does what a
plan never does -- shoves Targets into walls and into each other, and presses the
unbound ACTION key) replayed at all 16 presentations, requiring every frame to be
the exact transform of the unaugmented one. 0 violations, all 16 drawn.

The art survives the group too, which is the other half of what a flip needs.
Floor, Crate, Target and Target-on-Crate are each invariant under all eight turns
and mirrors -- a flat field, a border, a ring and a ring inside a border. Wall's
brown/darkbrown weave and the little figure (left-right symmetric, but a person,
so not top-bottom) are NOT, so a mirrored board does show art the .txt does not
literally contain. That is only a problem if the mirrored art of one thing is
another thing's art, and it is not: ``--audit``'s second pass turns and mirrors
every composition eight ways and compares it against every other composition, at
every cell size the levels use, and finds no match.

There is deliberately no colour augmentation: a Target ON a Crate is read as "the
crate's orange border with the target's blue ring showing through its middle",
which a flattening recolor would erase.

Rendering
---------
No recolor was needed, which is worth stating because this game's palette looks
like it should collide. ``--audit`` renders every cell COMPOSITION the game can
show -- floor, wall, crate, target, target-on-crate (the win), player,
player-on-crate -- as a whole 64x64 frame of a uniform board, at every cell size
the ten levels actually render at (4, 5, 7, 8 and 9 px), and requires them
pairwise distinct. They are, at all five.

What makes it work is that both goal-side sprites are hollow and stack cleanly:
Crate is a 5x5 BORDER (orange, ARC 12) with a transparent 3x3 middle, and Target
is a 3x3 RING (dark blue, ARC 9) inside that middle, so target-on-crate shows
both. Player and Target share a collision layer and therefore never stack, which
is the one composition that would have been a problem -- the player sprite is
opaque across exactly the rows the Target's ring occupies. Wall is a brown /
darkbrown weave (12 and 13); the Background's ``LIGHTGREEN GREEN`` both quantize
to 14, so the floor is flat green and the wall's 13 is the only thing on the
board that is not 12, 14 or 9.

Usage (run from the repo root):
    python solvers/generate_sokoban_flipped_training.py --episodes 200 \
        --out data/training_multi_level/sokoban_flipped

    python solvers/generate_sokoban_flipped_training.py --plans      # level report
    python solvers/generate_sokoban_flipped_training.py --selfcheck  # model + optimality
    python solvers/generate_sokoban_flipped_training.py --audit      # rendering
    python solvers/generate_sokoban_flipped_training.py --symmetry   # augmentation
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
                                     screen_action)
from utils.rotation import inverse_remap_action_full                  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Sokoban_Flipped"

#: Engine directions, in the order ties are broken. ACTION5 is bound to nothing
#: in this game (there is no ``action`` on any rule's left-hand side), so it is
#: not a move and branching on it would double the search for nothing. The
#: exploration prefix still presses it -- a live agent has that button -- which
#: is why ``--symmetry``'s random walk presses it too.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Disk cache of every level's start plan AND its optimal-action sets. The two
#: sweeps are seed-independent but cost ~3.6s together (3.2 of it on level 6),
#: which every shard `parallelize_generator` starts would otherwise repeat on
#: every core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "sokoban_flipped_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic, plus the search.

    Cells are flat ``r * w + c`` indices and a STATE is
    ``(player, sorted tuple of target cells)`` -- the only two things any rule in
    this game moves. Walls and crates never change, so they live here rather than
    in the state; that is what makes a state a pair of small integers and a short
    tuple, which matters on level 6, where the forward space passes 4 million.

    The mechanic is the single rule ``[> Player | Target] -> [> Player | > Target]``
    resolved under the collision layers, i.e.:

      * step onto a wall, or off the board  -> nothing happens;
      * step onto a Target                  -> it slides one further, UNLESS that
                                               cell is a wall, is off the board or
                                               holds another Target, in which case
                                               nothing happens at all (the player
                                               shares the Target's layer, so its
                                               own move is cancelled with it);
      * anything else                       -> the player moves.

    Crates appear in exactly one place -- the win test -- because they are on
    their own collision layer and block nothing.
    """

    __slots__ = ("h", "w", "wall", "crates", "win_targets", "_edge")

    def __init__(self, h: int, w: int, walls, crates):
        self.h, self.w = h, w
        self.wall = bytearray(h * w)
        for i in walls:
            self.wall[i] = 1
        self.crates = frozenset(crates)
        #: The winning target configuration. Every board has as many Targets as
        #: Crates, so ``All Target on Crate`` has exactly one solution as a SET:
        #: the targets sit on the crates. (Targets are interchangeable -- the
        #: state is a set, not an assignment.)
        self.win_targets = tuple(sorted(crates))
        #: ``_edge[cell][d]`` is the neighbour cell, or -1 off the board. Built
        #: once so the inner loops never do bounds arithmetic.
        self._edge = []
        for i in range(h * w):
            r, c = divmod(i, w)
            row = []
            for d in DIRS:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self._edge.append(tuple(row))

    # -- the mechanic --------------------------------------------------------
    def step(self, state, di: int):
        """The state after pressing ``DIRS[di]``, or ``state`` if the press does
        nothing (which is a non-move: no shortest path contains one)."""
        p, ts = state
        n1 = self._edge[p][di]
        if n1 < 0 or self.wall[n1]:
            return state
        if n1 in ts:
            n2 = self._edge[n1][di]
            if n2 < 0 or self.wall[n2] or n2 in ts:
                return state
            s = set(ts)
            s.discard(n1)
            s.add(n2)
            return (n1, tuple(sorted(s)))
        return (n1, ts)

    def preds(self, state):
        """Every state one press BEFORE ``state``.

        A walk is undone by stepping the player back onto ``p - d``, which has to
        be on the board, not a wall and not a Target (it was free when the player
        left it). A push is undone the same way with the Target dragged one cell
        towards the player: the Target now at ``p + d`` was at ``p``, so the
        earlier board is ``targets - {p + d} + {p}`` -- and ``p - d`` has to be
        free of THAT board, not of this one.

        Enumerating the inverses directly is what lets the backward sweep run
        without a reverse adjacency table, which on level 6 would be tens of
        millions of edges."""
        p, ts = state
        out = []
        for di in range(4):
            dr, dc = _DELTA[DIRS[di]]
            r, c = divmod(p, self.w)
            qr, qc = r - dr, c - dc
            if not (0 <= qr < self.h and 0 <= qc < self.w):
                continue
            q = qr * self.w + qc
            if self.wall[q]:
                continue
            if q not in ts:
                out.append((q, ts))                       # undo a walk
            n2 = self._edge[p][di]
            if n2 >= 0 and n2 in ts:                      # undo a push
                s = set(ts)
                s.discard(n2)
                s.add(p)
                if q not in s:
                    out.append((q, tuple(sorted(s))))
        return out

    def won(self, state) -> bool:
        return state[1] == self.win_targets

    # -- the search ----------------------------------------------------------
    def field(self, state):
        """``(dist, d_star)``: presses-to-win for every state on a shortest path
        from ``state``, and the length of that path. ``(None, None)`` if this
        board can never be won from here.

        Sweep 1 is a forward BFS stopped at the depth of the first win, which
        both fixes ``d_star`` and collects ``F`` (every state within ``d_star``
        presses of the start). Sweep 2 is a backward BFS from the won boards over
        `preds`, pruned to ``F``.

        The prune is exact where it is read. If ``s`` lies on a shortest
        start-to-win path then everything after it on that path has forward
        distance at most ``d_star`` and so is in ``F``, hence the backward sweep
        finds ``s``'s true distance. States on no shortest path can come out too
        high; nothing asks about those, because `plan` and `optimal` only ever
        compare against ``d_star - i``.
        """
        if self.won(state):
            return {state: 0}, 0
        seen = {state: 0}
        queue = deque([state])
        d_star = None
        while queue:
            cur = queue.popleft()
            d = seen[cur]
            if d_star is not None and d >= d_star:
                break
            for di in range(4):
                nxt = self.step(cur, di)
                if nxt == cur or nxt in seen:
                    continue
                seen[nxt] = d + 1
                if self.won(nxt):
                    if d_star is None:
                        d_star = d + 1
                    continue                # a won board is terminal
                queue.append(nxt)
        if d_star is None:
            return None, None

        dist = {}
        queue = deque()
        for s in seen:
            if self.won(s):
                dist[s] = 0
                queue.append(s)
        while queue:
            cur = queue.popleft()
            d = dist[cur]
            for prev in self.preds(cur):
                if prev in dist or prev not in seen:
                    continue
                dist[prev] = d + 1
                queue.append(prev)
        return dist, d_star

    def optimal(self, dist, state):
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``. Ties come out in ``DIRS`` order, which is what makes
        a re-derived plan byte-identical across processes."""
        rest = dist.get(state)
        if not rest:
            return []
        out = []
        for di in range(4):
            nxt = self.step(state, di)
            if nxt != state and dist.get(nxt, -1) == rest - 1:
                out.append((di, nxt))
        return out

    def solve(self, state) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal set
        at every step, or None if this board cannot be won from here."""
        dist, d_star = self.field(state)
        if dist is None:
            return None
        presses, optsets = [], []
        cur = state
        for _ in range(d_star):
            best = self.optimal(dist, cur)
            if not best:                    # unreachable: dist[cur] > 0 has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class SokobanFlippedExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Board` field.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes, the way `PSEnumExpert` replaces it. Here the
    strategy is "read the live board, sweep it forwards then backwards, hand back
    the shortest plan and its exact tie sets", so `heuristic` is never called and
    asserts rather than returning a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered exactly and
    not by patching up a stored path."""

    directions = list(DIRS)
    #: `_key` is the dynamic objects only (player + targets), which is canonical
    #: WITHIN a level but not across them -- walls and crates are static per level
    #: and differ between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        self.player_ids = set(self.game._engine._player_indices)
        self.target_ids = set(self.g.resolve_object_name("target"))
        self.wall_ids = set(self.g.resolve_object_name("wall"))
        self.crate_ids = set(self.g.resolve_object_name("crate"))
        self.dyn_ids = self.player_ids | self.target_ids
        #: `_Board`s by STATIC signature (see `read`), not by level index.
        self._boards: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "SokobanFlippedExpert reads an exact distance field; "
            "heuristic is unused")

    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn))

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid.

        Boards are cached by their STATIC signature (dimensions, walls, crates),
        not by level index, so the two sweeps never rebuild the edge table and a
        board is shared by every state of its level."""
        h, w = eng.height, eng.width
        walls, crates, targets, player = [], [], [], None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.crate_ids:
                    crates.append(i)
                if cell & self.target_ids:
                    targets.append(i)
                if cell & self.player_ids:
                    player = i
        sig = (h, w, tuple(walls), tuple(crates))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, crates)
        return board, (player, tuple(sorted(targets)))

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if state[0] is None:                 # no player on the board
            return None
        if board.won(state):
            return Plan([], [])
        if len(state[1]) != len(board.crates):
            # Not reachable on the shipped levels (every board has as many
            # Targets as Crates) and not a crash if a level is ever edited: with
            # more targets than crates the win condition is unsatisfiable, with
            # fewer the "all targets on crates" set is not `win_targets`.
            return None
        return board.solve(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class SokobanFlippedSolver(PSAStarSolver):
    game_id = "puzzlescript_sokoban_flipped"
    game_name = GAME_NAME
    expert_cls = SokobanFlippedExpert

    #: `games/ps:sokoban_flipped/ps:sokoban_flipped.py` is a plain passthrough
    #: (it constructs the adapter and nothing else), so there is nothing to gain
    #: by routing through it -- but it IS what a live agent is handed, so if that
    #: wrapper ever grows a patch this must be set to ``"ps:sokoban_flipped"``.
    game_module_id = ""

    #: Unused: the expert enumerates an exact field rather than searching under a
    #: heuristic, so there is no weight to trade and no node budget that could
    #: quietly bite. Left at the base values so nothing reads a lie off them.
    weight = 1
    node_cap = 400_000

    #: The longest plan is 71 presses (level 9); the rest is room for a re-plan
    #: after the exploration prefix. Stays under the adapter's own 200-step
    #: per-level budget, which `set_level` resets before the plan starts anyway.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time ~3.6s visible as startup
        rather than as a mysteriously slow first seed, and it fills the disk
        cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new():
    solver = SokobanFlippedSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once."""
    _solver, game, expert, solvable = _new()
    idx = game._game.obj_name_to_idx
    total = ties = 0
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        pieces = sum(1 for row in eng.grid for cell in row
                     if idx["target"] in cell)
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
        print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  {pieces} targets  "
              f"{len(plan):3d} presses  win={won}  (budget {game._max_steps}, "
              f"{room})  {tie_steps:3d} steps with a tie set")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _selfcheck(walk_presses: int = 400, verbose: bool = True) -> int:
    """Three things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's.** A seeded random walk on every
       level, comparing `_Board.step`'s board against the engine's grid after
       EVERY press -- including the presses that do nothing, which is where a
       collision-layer mistake hides.
    2. **The plans are shortest.** A plain forward BFS to the first win, which is
       shortest by construction and shares no code with the field's backward
       half, must return the field's plan length on every level.
    3. **The tie sets are exact.** On the levels whose spaces are small enough to
       afford it, every step's optimal set is re-derived by brute force -- a
       fresh depth-bounded BFS from each of the four successors -- and must match
       the field's answer exactly.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0

    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        rng = random.Random(f"sokoban_flipped:selfcheck:{level}")
        drift = 0
        for _ in range(walk_presses):
            di = rng.randrange(4)
            eng.step(DIRS[di])
            state = board.step(state, di)
            _b, live = expert.read(eng)
            if live != state:
                drift += 1
                break
        bad += drift
        if verbose:
            print(f"  L{level:2d}: model {'MATCHES' if not drift else 'DIVERGES from'}"
                  f" the interpreter over {walk_presses} random presses")

    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        plan = expert.plan(eng, level)
        # An independent shortest: plain forward BFS, first win wins.
        seen = {state}
        queue = deque([(state, 0)])
        shortest = None
        while queue and shortest is None:
            cur, d = queue.popleft()
            for di in range(4):
                nxt = board.step(cur, di)
                if nxt == cur or nxt in seen:
                    continue
                if board.won(nxt):
                    shortest = d + 1
                    break
                seen.add(nxt)
                queue.append((nxt, d + 1))
        ok = plan is not None and shortest == len(plan)
        bad += not ok
        if verbose:
            print(f"  L{level:2d}: field {len(plan) if plan else None} presses vs "
                  f"BFS {shortest} -- {'SHORTEST' if ok else 'NOT SHORTEST'}")

    def brute(board, state, limit):
        """Presses to a win from ``state``, searched fresh, or None past
        ``limit``."""
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

    #: Levels 4, 5 and 6 are left out: their spaces run to 10^5-10^6 states and a
    #: fresh BFS per successor per step would take hours to say what the field
    #: already says. The seven checked cover every plan shape the game has
    #: (pushes on both axes, walks over crates, the already-satisfied target).
    cheap = [0, 1, 2, 3, 7, 8, 9]
    for level in cheap:
        game.set_level(level)
        board, state = expert.read(eng)
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
                if brute(board, nxt, rest - 1) == rest - 1:
                    truth.append(DIRS[di])
            if sorted(truth) != sorted(sets[i]):
                mismatched += 1
            cur = board.step(cur, DIRS.index(direction))
        bad += mismatched
        if verbose:
            print(f"  L{level:2d}: {len(plan)} tie sets "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'}"
                  f" brute force")
    return bad


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the ten boards use. Whole
    64x64 frames of uniform boards are compared rather than one cell out of a
    mixed board: `_render_frame` upscales and centre-pads, so slicing a cell by
    ``cell_px`` arithmetic reads the wrong pixels (the ps:explod lesson). Two
    uniform boards render identically iff their cells do.

    ``player + target`` is absent on purpose and is not an omission: Player and
    Target share a collision layer, so the engine can never put them in one cell.
    It is also the only stack that would have been a problem -- the player sprite
    is opaque across exactly the three rows the Target's ring occupies.

    **Pass 2 -- the group.** The game is in `_FLIP_GAMES`, so a board is drawn at
    any of the eight turns and mirrors, and two of its sprites (the wall weave,
    the little figure) are not invariant under them. That is fine as long as no
    transform of one composition is another composition's art, which is what this
    checks: all eight transforms of each, against all seven others. It runs on
    SQUARE boards, because a rotation of a non-square board also moves the
    letterbox and every comparison would pass for the wrong reason."""
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert, _solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = [(), ("wall",), ("crate",), ("target",), ("target", "crate"),
             ("player",), ("player", "crate")]

    def name(comp):
        return "+".join(comp) or "floor"

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, comp):
        eng.height, eng.width = h, w
        eng.grid = [[{idx[o] for o in comp} | {idx["background"]}
                     for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g)).copy()

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {c: shoot(h, w, c) for c in comps}
        clashes = [(a, b) for a, b in itertools.combinations(comps, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, {'OK' if not clashes else 'CLASH'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {name(a)}  ==  {name(b)}")

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {c: shoot(n, n, c) for c in comps}
        group = [(k, hf, vf) for k in range(4)
                 for hf in (False, True) for vf in (False, True)]
        clashes = [(a, b, t) for a, b in itertools.permutations(comps, 2)
                   for t in group
                   if np.array_equal(transform(shots[a], *t), shots[b])]
        invariant = [name(c) for c in comps
                     if all(np.array_equal(transform(shots[c], *t), shots[c])
                            for t in group)]
        bad += len(clashes)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(clashes)} composition(s) "
                  f"whose transform is another's art; group-invariant: "
                  f"{', '.join(invariant)}")
        for a, b, t in clashes:
            print(f"      {name(a)} under {t} == {name(b)}")
    return bad


def _symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for putting the game in `_FLIP_GAMES` (the rotation is
    mandatory and is checked here too). The structural argument is in the module
    docstring and is about as tight as this family gets -- one rule, written with
    the relative ``>``, moving at most two bodies onto squares that can never
    coincide -- but Gobble Rush's chirality hid inside exactly this kind of
    argument, so it is measured.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (a Target shoved flat against a wall, two Targets
    jammed together, the player sealed in a pocket) and it presses the unbound
    ACTION key as well."""
    solver, game, expert, _solvable = _new()
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
        if len(seen) == 16 and seed > 40:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"sokoban_flipped:symmetry:{level}")
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
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(SokobanFlippedSolver.main())
