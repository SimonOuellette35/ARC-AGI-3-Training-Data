"""Generate Phase-1 training data for the PuzzleScript game ps:sokoban_sanity
("Simple Block Pushing Game", David Skinner).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exact distance FIELD that model is
searched with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_sokoban_sanity",
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
This is the canonical PuzzleScript sokoban -- the one every other block-pushing
game in this corpus is a fork of. The whole game is ONE rule:

    [ >  Player | Crate ] -> [  >  Player | > Crate  ]

and the win is ``All Target on Crate``: every Target square must have a Crate
standing on it. The collision layers are

    Background
    Target
    Player, Wall, Crate

which is the ORTHODOX assignment, and worth stating because this game's sibling
`ps:sokoban_flipped` (same rule, names exchanged) is the one that is not:

* **Crates and walls block.** They share the player's layer, so walking into a
  wall does nothing, and walking into a crate is the push. A crate whose far
  side is a wall, the board edge, or a second crate does not move -- and neither
  does the player, because the rule's right-hand side moves both or neither.
* **There is no chain push.** The rule names ``Player`` on its left, so a moving
  crate never re-triggers it: two crates in a row are a wall.
* **Targets block nothing.** They are alone on their own layer, under the
  player's, so the player and a sliding crate both pass straight over one. A
  target is a goal SQUARE, and the only thing in the game that never moves
  besides the walls.

Every shipped board has exactly as many Crates as Targets, so a win is the
crates standing on precisely the target set -- but `_Board.won` is written as
"every target is covered", which is what the win condition literally says, so an
edited level with a spare crate would still be answered correctly.

Everything above was MEASURED against the interpreter rather than read off the
.txt; ``--selfcheck`` is the executable form of it (a seeded random walk on every
level, the model's board compared against the engine's grid after every single
press, including the presses that do nothing -- which is where a collision-layer
mistake hides). The four measurements the walk covers on the shipped boards:
a step onto a target (the player passes over it), a push of a crate onto a
target (it slides on and the level may end there), a push into a wall or off the
board (nothing moves, player included), and a push into a second crate (likewise
nothing).

Three levels, all 7x6, all winnable, none already won at reset:

  * L0 has no walls at all -- an open 7x6 field with one crate, one target, and
    the board EDGE as the only geometry. It is also the level whose plan walks
    the player over its one target repeatedly, which is what made the rendering
    bug below matter.
  * L1 is a walled room with two crates, one of which starts already parked on a
    target. Nothing pins that crate down -- shoving it off is as easy as any
    other push -- so the answer has to work around it, and at 33 presses it is
    the long one.
  * L2 is a 4x4 room with three crates, two of them already home.

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`), searched to an exact distance
field rather than heuristically.

The shared `PSSokobanExpert` fits this game in two attribute lines, and on these
three boards it is not wrong: ``--macro`` runs it and it returns the SAME three
plans, at the same lengths. (That is worth recording because its sibling
`ps:sokoban_flipped` documents the opposite result -- `PSPushExpert` dedups
states by the player's reachable REGION rather than by its exact cell, which
mis-costs a plan by up to that region's diameter, and L0's floor here is one open
region with no walls at all. The trap is real; it just does not bite on a board
this small. Which is the point of running the comparison instead of assuming
either way.)

What it cannot do is the reason for the field: the optimal-action SETS. A
`PSSokobanExpert` ships none at all by default, and the ``annotate = True``
alternative infers them from a rule of thumb ("a push is forced, a walk's axis
order is free") -- which on this game gets 57 of the 58 steps right and misses
the only one that matters (see below). The field gives the exact answer as a
by-product of already knowing every state's distance. It is also 4x faster
(0.01s against 0.04s for all three levels), because it runs on the model while
the macro A* drives the real interpreter for every node.

`_Board.solve` is two breadth-first sweeps over ``(player, crates)`` states:

  1. FORWARD from the state being planned, stopping at the depth ``d*`` of the
     first win. That fixes the answer's length and collects ``F``, every state
     within ``d*`` presses of the start.
  2. BACKWARD from the winning configurations over INVERTED moves (a walk is
     undone by stepping back onto a free cell; a push is undone by stepping back
     and dragging the crate one cell towards you), pruned to ``F``.

The inverse moves are enumerated directly, so the backward sweep needs no
reverse adjacency table. On these three boards either direction would be
affordable -- the whole game is ~17k reachable states, and all three fields
together take 0.01s -- but the shape is the one that scales, and it is what
produces the tie sets below.

Pruning the backward sweep to ``F`` costs nothing that matters: if a state ``s``
lies on any shortest start-to-win path then its whole continuation has forward
distance at most ``d*``, so that continuation never leaves ``F`` and the field's
value at ``s`` is its true distance. States off every shortest path may be
over-estimated, and none of them is ever asked about.

Optimal-action sets
-------------------
The field gives them EXACTLY, with no inference: at a state ``d`` presses from a
win, the optimal presses are every direction whose successor is ``d - 1`` from
one.

The measured answer is that this game has almost no ties: exactly ONE of its 58
expert steps has a second equally-right press -- L0's opening, where ``down`` and
``right`` both start the walk round the crate. Every other step is forced, which
is not what an open 7x6 field with no walls looks like it should give: the crate
sits in the walk's way, so all but one interleaving of the two axes either shoves
it somewhere useless or costs a press.

`PSPushExpert.annotate_walks`, the inferred alternative, gets 57 of the 58 right
and misses exactly that one (``--macro`` prints the comparison). So the error it
makes here is the opposite of the one its rule of thumb suggests -- it
UNDER-labels rather than over-labels, calling the opening forced because the
press it would pair with is the one the walk does not take. One step in 58 is
small; it is also the single step in the game where a policy has a genuine
choice, and there is no reason to ship it wrong when the field has the answer
already.

``--selfcheck`` re-derives every set on every level by brute force (a fresh
depth-bounded BFS from each of the four successors) and requires an exact match;
``--engine`` re-derives them again from the interpreter. No step ever ships
unlabelled (the always-emit-optimal-targets rule).

Shortest, and how that is known
-------------------------------
All three plans are provably shortest (9, 33 and 16 presses), and the proof is
three independent derivations agreeing:

  * this file's field (forward BFS + backward BFS);
  * a plain forward BFS to the first win, which is shortest by construction and
    shares no code with the field's backward half (``--selfcheck``);
  * a full-space enumeration of the real INTERPRETER -- `StateGraph.build`
    walking every reachable engine state by pressing real buttons, with no
    native model involved at all (``--engine``: 1681, 627 and 15135 states).
    That is the derivation that closes the loop back to the engine, and it
    agrees on all three lengths AND on every one of the 58 tie sets.

This game is small enough to afford the third pass, which the bigger ps: sokobans
are not -- `Sokoban_Flipped`'s level 6 alone passes 4 million states -- so here
the engine-side proof is exhaustive rather than a spot check. The native model is
separately fuzz-verified against the interpreter, which is what makes the first
two derivations statements about this game rather than about a model of it.

Recovery
--------
`supports_recovery` with RESET-mode recovery. The expert re-plans from the LIVE
board, not from a stored path: `_search` reads whatever the engine currently
holds and runs both sweeps from there, so an exploration prefix that leaves the
board anywhere -- including somewhere a shortest plan would never go -- is
answered exactly. A prefix CAN strand this game for good (shove a crate into a
corner and the level is dead), and the field says so by returning None, which is
exactly what `record_level` needs to hear to fall back to a RESET. That is not a
rare corner: ``--selfcheck``'s fourth pass walks 120 random boards per level and
finds 43 of the 360 genuinely unwinnable (31 of them on L1's cramped room). It
checks both halves against the interpreter -- every plan it returns from a random
board is replayed and must WIN, and every None must be a board an independent
forward BFS also proves dead, never a give-up.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group. 3 levels x 16 presentations = 48, and with only three
levels the flips are most of the corpus's variety rather than a nicety.

The structural argument for the flips is as strong as it gets in this family --
it is the same one `Sokoban_Flipped` and `sokoban_match3` carry, minus the latter's
match rule. There is exactly ONE rule, it is written with the relative force
``>``, it names no axis, one press moves at most two bodies and they land on
``p + d`` and ``p + 2d``, which can never be the same square -- so no two moves
can contest a cell and the order the interpreter expands the rule's four
directions cannot decide anything. The win condition is positional. ``--symmetry``
measures it anyway: every level's plan AND a seeded 200-press random walk (which
does what a plan never does -- shoves crates into walls, into each other and into
corners, and presses the unbound ACTION key) replayed at all 16 presentations,
requiring every frame to be the exact transform of the unaugmented one.

The art survives the group too, which is the other half of what a flip needs.
Floor, target, crate and crate-on-target are each invariant under all eight turns
and mirrors -- a flat field, a solid square, a border, and a square inside a
border. Wall's brown/darkbrown weave and the little figure (left-right symmetric,
but a person, so not top-bottom) are NOT, so a mirrored board does show art the
.txt does not literally contain. That is only a problem if the mirrored art of
one composition is another composition's art, and it is not: ``--audit``'s second
pass turns and mirrors every composition eight ways and compares it against every
other.

There is deliberately no colour augmentation: a Crate ON a Target is read as "the
crate's orange border with the target's purple square showing through its
middle", which a flattening recolor would erase.

Rendering
---------
ONE sprite had to change, and it is the one the win condition is made of. See the
header comment in ``data/puzzlescript_games/sokoban_sanity.txt`` for the full
statement; in short, Target shipped as a hollow DARKBLUE ring on rows/cols 1-3
and the Player sprite is opaque across exactly that window, so "player standing
on a target" was PIXEL-IDENTICAL to "player standing on grass" -- and on L0 the
player walks over its one target in almost every plan. Target is now a SOLID
PURPLE square: solid so it shows through the player's nine transparent pixels AND
through the crate's open 3x3 middle, purple because ``darkblue`` and the player's
own ``blue`` are the same index (9) in the 16-colour ARC palette. This is the
identical fix this game's fork ``ps:sokoban_match3`` already carries.

``--audit`` renders every cell COMPOSITION the game can show -- floor, wall,
target, crate, crate-on-target (the win), player, player-on-target -- as a whole
64x64 frame of a uniform board, at every cell size the three levels render at
(9 px; all three boards are 7x6), and requires them pairwise distinct. Whole
frames rather than one cell sliced out of a mixed board: `_render_frame` upscales
and centre-pads, so slicing by ``cell_px`` arithmetic reads the wrong pixels (the
ps:explod lesson). ``player + crate`` is absent on purpose and is not an
omission -- they share a collision layer, so the engine can never put them in one
cell.

Usage (run from the repo root):
    python solvers/generate_sokoban_sanity_training.py --episodes 200 \
        --out data/training_multi_level/sokoban_sanity

    python solvers/generate_sokoban_sanity_training.py --plans      # level report
    python solvers/generate_sokoban_sanity_training.py --selfcheck  # model + optimality
    python solvers/generate_sokoban_sanity_training.py --engine     # interpreter proof
    python solvers/generate_sokoban_sanity_training.py --macro      # the region-dedup trap
    python solvers/generate_sokoban_sanity_training.py --audit      # rendering
    python solvers/generate_sokoban_sanity_training.py --symmetry   # augmentation
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

GAME_NAME = "sokoban_sanity"

#: Engine directions, in the order ties are broken. ACTION5 is bound to nothing
#: in this game (no rule has ``action`` on its left-hand side), so it is not a
#: move and branching on it would double the search for nothing. The exploration
#: prefix still presses it -- a live agent has that button -- which is why
#: ``--symmetry``'s random walk presses it too.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Disk cache of every level's start plan AND its optimal-action sets. The three
#: boards are small enough that the sweeps cost well under a second, so this is
#: not the load-bearing cache it is on the bigger ps: sokobans -- it is here so a
#: `parallelize_generator` fan-out shares one derivation rather than repeating it
#: on every core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "sokoban_sanity_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic, plus the search.

    Cells are flat ``r * w + c`` indices and a STATE is
    ``(player, sorted tuple of crate cells)`` -- the only two things any rule in
    this game moves. Walls and targets never change, so they live here rather
    than in the state.

    The mechanic is the single rule ``[> Player | Crate] -> [> Player | > Crate]``
    resolved under the collision layers, i.e.:

      * step onto a wall, or off the board -> nothing happens;
      * step onto a crate                  -> it slides one further, UNLESS that
                                              cell is a wall, is off the board or
                                              holds another crate, in which case
                                              nothing happens at all (the rule's
                                              RHS moves both bodies or neither,
                                              so the player's own move is
                                              cancelled with the crate's);
      * anything else                      -> the player moves.

    Targets appear in exactly one place -- the win test -- because they are alone
    on their own collision layer and block nothing.
    """

    __slots__ = ("h", "w", "wall", "targets", "_edge")

    def __init__(self, h: int, w: int, walls, targets):
        self.h, self.w = h, w
        self.wall = bytearray(h * w)
        for i in walls:
            self.wall[i] = 1
        #: The win set. ``All Target on Crate`` reads as "every target square has
        #: a crate on it", so this is a SUBSET test, not an equality one -- which
        #: is the same thing on every shipped board (crates == targets in count)
        #: and still correct on an edited one with a crate to spare.
        self.targets = frozenset(targets)
        #: ``_edge[cell][d]`` is the neighbour cell, or -1 off the board. Built
        #: once so the inner loops never do bounds arithmetic. The board edge is
        #: the ONLY geometry level 0 has, so this table is level 0's whole map.
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
        p, cs = state
        n1 = self._edge[p][di]
        if n1 < 0 or self.wall[n1]:
            return state
        if n1 in cs:
            n2 = self._edge[n1][di]
            if n2 < 0 or self.wall[n2] or n2 in cs:
                return state
            s = set(cs)
            s.discard(n1)
            s.add(n2)
            return (n1, tuple(sorted(s)))
        return (n1, cs)

    def preds(self, state):
        """Every state one press BEFORE ``state``.

        A walk is undone by stepping the player back onto ``p - d``, which has to
        be on the board, not a wall and not a crate (it was free when the player
        left it). A push is undone the same way with the crate dragged one cell
        towards the player: the crate now at ``p + d`` was at ``p``, so the
        earlier board is ``crates - {p + d} + {p}`` -- and ``p - d`` has to be
        free of THAT board, not of this one (it never differs here, since ``p``
        and ``p + d`` are neither of them ``p - d``, but the test is written
        against the reconstructed board so it stays right if the mechanic is ever
        widened).

        Enumerating the inverses directly is what lets the backward sweep run
        without a reverse adjacency table."""
        p, cs = state
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
            if q not in cs:
                out.append((q, cs))                       # undo a walk
            n2 = self._edge[p][di]
            if n2 >= 0 and n2 in cs:                      # undo a push
                s = set(cs)
                s.discard(n2)
                s.add(p)
                if q not in s:
                    out.append((q, tuple(sorted(s))))
        return out

    def won(self, state) -> bool:
        return self.targets <= set(state[1])

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

class SokobanSanityExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Board` field.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes, the way `PSEnumExpert` replaces it. Here the
    strategy is "read the live board, sweep it forwards then backwards, hand back
    the shortest plan and its exact tie sets", so `heuristic` is never called and
    asserts rather than returning a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered exactly --
    including "this board is now dead", which an exploration prefix really can
    produce here by shoving a crate into a corner."""

    directions = list(DIRS)
    #: `_key` is the dynamic objects only (player + crates), which is canonical
    #: WITHIN a level but not across them -- walls and targets are static per
    #: level and differ between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        self.player_ids = set(self.game._engine._player_indices)
        self.target_ids = set(self.g.resolve_object_name("target"))
        self.wall_ids = set(self.g.resolve_object_name("wall"))
        self.crate_ids = set(self.g.resolve_object_name("crate"))
        self.dyn_ids = self.player_ids | self.crate_ids
        #: `_Board`s by STATIC signature (see `read`), not by level index.
        self._boards: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "SokobanSanityExpert reads an exact distance field; "
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

        Boards are cached by their STATIC signature (dimensions, walls, targets),
        not by level index, so the two sweeps never rebuild the edge table and a
        board is shared by every state of its level."""
        h, w = eng.height, eng.width
        walls, targets, crates, player = [], [], [], None
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
                    crates.append(i)
                if cell & self.player_ids:
                    player = i
        sig = (h, w, tuple(walls), tuple(targets))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, targets)
        return board, (player, tuple(sorted(crates)))

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if state[0] is None:                 # no player on the board
            return None
        if board.won(state):
            return Plan([], [])
        if len(state[1]) < len(board.targets):
            # Not reachable on the shipped levels (every board has as many crates
            # as targets) and not a crash if a level is ever edited: with fewer
            # crates than targets the win condition is unsatisfiable.
            return None
        return board.solve(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class SokobanSanitySolver(PSAStarSolver):
    game_id = "puzzlescript_sokoban_sanity"
    game_name = GAME_NAME
    expert_cls = SokobanSanityExpert

    #: `games/ps:sokoban_sanity/ps:sokoban_sanity.py` is a plain passthrough (it
    #: constructs the adapter and nothing else), so there is nothing to gain by
    #: routing through it -- but it IS what a live agent is handed, so if that
    #: wrapper ever grows a patch this must be set to ``"ps:sokoban_sanity"``.
    game_module_id = ""

    #: Unused: the expert enumerates an exact field rather than searching under a
    #: heuristic, so there is no weight to trade and no node budget that could
    #: quietly bite. Left at the base values so nothing reads a lie off them.
    weight = 1
    node_cap = 400_000

    #: The longest plan is 33 presses (level 1); the rest is room for a re-plan
    #: after the exploration prefix. Stays well under the adapter's own 200-step
    #: per-level budget, which `set_level` resets before the plan starts anyway.
    max_steps = 120

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it fills the disk cache in one pass and makes the
        startup cost visible as startup."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new():
    solver = SokobanSanitySolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once."""
    _solver, game, expert, solvable = _new()
    idx = game._game.obj_name_to_idx
    total = ties = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        crates = sum(1 for row in eng.grid for cell in row if idx["crate"] in cell)
        targets = sum(1 for row in eng.grid for cell in row if idx["target"] in cell)
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
        print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  {crates} crates / "
              f"{targets} targets  {len(plan):3d} presses  win={won}  "
              f"(budget {game._max_steps}, {room})  "
              f"{tie_steps:3d} steps with a tie set")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _selfcheck(walk_presses: int = 400, samples: int = 120,
               verbose: bool = True) -> int:
    """Four things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's.** A seeded random walk on every
       level, comparing `_Board.step`'s board against the engine's grid after
       EVERY press -- including the presses that do nothing, which is where a
       collision-layer mistake hides.
    2. **The plans are shortest.** A plain forward BFS to the first win, which is
       shortest by construction and shares no code with the field's backward
       half, must return the field's plan length on every level.
    3. **The tie sets are exact.** Every step's optimal set is re-derived by
       brute force -- a fresh depth-bounded BFS from each of the four successors
       -- and must match the field's answer exactly. Unlike the bigger ps:
       sokobans this game is small enough to afford that on EVERY level, so
       nothing here is checked by sampling.
    4. **Recovery is answered from ANY board, not just the start.** This is the
       one that matters at generation time: the exploration prefix really does
       strand this game (shove a crate into a corner and no press wins again),
       so `_search` has to be right about arbitrary states in both directions.
       From ``samples`` states reached by seeded random presses, the field must
       either return a plan that WINS when replayed through the interpreter, or
       return None on a board an independent forward BFS also finds no win from.
       A None that is merely a give-up would be a silent bug: `record_level`
       reads it as "reset", and a wrong plan length here would not show up in
       any of the three checks above, all of which only ever ask about starts.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0

    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        rng = random.Random(f"sokoban_sanity:selfcheck:{level}")
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

    for level in range(game.n_levels):
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

    # 4 -- recovery from arbitrary boards, checked against the interpreter.
    for level in range(game.n_levels):
        rng = random.Random(f"sokoban_sanity:recovery:{level}")
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
            # Independent: an unbounded forward BFS over the whole reachable
            # space, which either finds a win or PROVES there is none.
            truth = brute(board, state, len(board.wall) * 8)
            if plan is None:
                dead += 1
                wrong += truth is not None
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
                  f"re-planned to a WIN in the interpreter, {dead} proved dead, "
                  f"{wrong} WRONG")
    return bad


def _engine(verbose: bool = True) -> int:
    """The proof that owes nothing to the native model: enumerate the REAL
    interpreter.

    `StateGraph.build` walks every state reachable from the level start by
    pressing actual buttons on the engine and keying the resulting grid, then
    solves the graph exactly. It shares no line of code with `_Board` -- no
    modelled push, no modelled collision layer, no modelled win test -- so
    agreement on both the plan LENGTH and the per-step optimal SET is an
    independent derivation of everything this file claims.

    Affordable only because the whole game is a few thousand engine states; on a
    board like `Sokoban_Flipped`'s level 6 this pass is the one that would not
    run, which is why it exists here and not there."""
    from solvers.common.ps_astar import StateGraph                     # noqa: PLC0415

    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        game.set_level(level)
        graph = StateGraph.build(eng, expert._key, list(DIRS),
                                 node_cap=200_000)
        eplan = graph.plan(list(DIRS)) if graph is not None else None
        if eplan is None:
            print(f"  L{level:2d}: the interpreter enumeration found no win")
            bad += 1
            continue
        same_len = len(eplan) == len(plan)
        esets = [sorted(s) for s in eplan.optsets]
        fsets = [sorted(s) for s in plan.optsets]
        same_sets = esets == fsets
        bad += (not same_len) + (not same_sets)
        if verbose:
            print(f"  L{level:2d}: {len(graph.succ)}"
                  f" engine states, {len(eplan)} presses vs the field's "
                  f"{len(plan)} -- {'AGREE' if same_len else 'DISAGREE'}; tie sets "
                  f"{'AGREE' if same_sets else 'DISAGREE'}")
    return bad


def _macro(verbose: bool = True) -> int:
    """Run the shared `PSSokobanExpert` on this game and print its plans and its
    INFERRED tie sets beside the field's exact ones.

    Not a check that has to pass -- it is the measurement behind the module
    docstring's account of why this game gets its own expert, and the two halves
    of it say different things:

    * the plan LENGTHS agree on all three levels. `PSPushExpert._region_key`
      dedups by the player's reachable REGION rather than by its exact cell,
      which mis-costs a plan by up to that region's diameter and cost
      `ps:sokoban_flipped` 12 presses on its one open board -- and level 0 here
      has no walls at all, so its whole floor is one region. It happens not to
      bite at this size. Recorded rather than assumed, in either direction.
    * the tie SETS do not. `annotate_walks` is a rule of thumb and this reports
      how many steps it labels differently from the field.

    Returns the number of levels where the macro plan is longer, so a future
    change that makes the region dedup start to bite is visible as a non-zero."""
    from solvers.common.ps_astar import PSSokobanExpert                # noqa: PLC0415

    _solver, game, expert, _solvable = _new()
    eng = game._engine

    class _Macro(PSSokobanExpert):
        pushable_names = ("crate",)
        target_names = ("target",)
        #: Off by default, so a plain `PSSokobanExpert` game ships with NO sets
        #: at all; turned on here to compare the inferred answer against the
        #: exact one rather than against nothing.
        annotate = True

    macro = _Macro(game, node_cap=400_000, weight=1)
    worse = 0
    for level in range(game.n_levels):
        game.set_level(level)
        exact = expert.plan(eng, level)
        game.set_level(level)
        mplan = macro.plan(eng, level)
        n = None if mplan is None else len(mplan)
        worse += n is None or n > len(exact)
        if verbose:
            tag = ("SHORTEST" if n == len(exact) else
                   f"{'' if n is None else n - len(exact)} presses LONGER")
            fsets = [sorted(s) for s in exact.optsets]
            msets = [sorted(s) for s in (getattr(mplan, "optsets", None) or [])]
            differ = (sum(a != b for a, b in zip(fsets, msets))
                      if len(msets) == len(fsets) else len(fsets))
            print(f"  L{level:2d}: field {len(exact):3d}  macro "
                  f"{'None' if n is None else f'{n:3d}'}  -- {tag}; "
                  f"{sum(1 for s in fsets if len(s) > 1)} exact ties vs "
                  f"{sum(1 for s in msets if len(s) > 1)} inferred, "
                  f"{differ} step(s) labelled differently")
    return worse


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the three boards use (they are
    all 7x6, so that is 9 px, but the size list is derived rather than assumed so
    an edited level is covered). Whole 64x64 frames of uniform boards are
    compared rather than one cell out of a mixed board: `_render_frame` upscales
    and centre-pads, so slicing a cell by ``cell_px`` arithmetic reads the wrong
    pixels (the ps:explod lesson). Two uniform boards render identically iff
    their cells do.

    ``player + crate`` is absent on purpose and is not an omission: Player and
    Crate share a collision layer, so the engine can never put them in one cell.

    **Pass 2 -- the group.** The game is in `_FLIP_GAMES`, so a board is drawn at
    any of the eight turns and mirrors, and two of its sprites (the wall weave,
    the little figure) are not invariant under them. That is fine as long as no
    transform of one composition is another composition's art, which is what this
    checks: all eight transforms of each, against all others. It runs on SQUARE
    boards, because a rotation of a non-square board also moves the letterbox and
    every comparison would pass for the wrong reason."""
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert, _solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = [(), ("wall",), ("target",), ("crate",), ("crate", "target"),
             ("player",), ("player", "target")]

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
    boards a plan never visits (a crate flat against a wall, two crates jammed
    together, a crate dead in a corner) and it presses the unbound ACTION key as
    well."""
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
            rng = random.Random(f"sokoban_sanity:symmetry:{level}")
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
        print(f"engine enumeration: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--macro" in sys.argv:
        worse = _macro()
        print(f"macro A*: {worse} level(s) where the region dedup costs presses")
        sys.exit(0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(SokobanSanitySolver.main())
