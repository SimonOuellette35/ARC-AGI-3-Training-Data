"""Generate Phase-1 training data for the PuzzleScript game ps:strange_warehouse
("Strange Warehouse", Justas Dabrila, GGJ 2020).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exact distance FIELD that model is
searched with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_strange_warehouse",
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
Every expert step carries the full set of equally-optimal presses.

The game
--------
A sokoban whose second crate type is TELEKINETIC. Three rules matter:

    [ > player | crate ] -> [ > player | > crate ]
    [ enchanted_crate | ... | moving player ] -> [ moving enchanted_crate | ... | moving player ]

(the third block -- ``late`` rules that repaint an enchanted crate as one of
``spr_enchanted_crate_active_{up,down,left,right}`` -- moves nothing; it is a
per-frame INDICATOR, and the one place the mechanic is visible before you use
it). The win is three conditions at once: every enchanted crate on an enchanted
target, every plain crate on a crate target, and the player on the guy target.

The second rule is the whole game, and every clause of it was measured rather
than read off the .txt:

* **An enchanted crate copies the player's direction; it is not dragged towards
  it.** ``moving`` on the right-hand side is bound to the left-hand side's
  binding, which is the PLAYER's actual direction, so a linked crate steps the
  same way the player does. The ``active`` sprite points at the player, which is
  what makes this easy to get backwards; it is a which-way-is-the-player badge,
  not a which-way-will-I-go one.
* **"Linked" means sharing a row or a column, and nothing more.** The rule is
  written with an ellipsis, and an ellipsis constrains none of the cells it spans
  -- so the link reaches THROUGH walls, through other crates and through the
  player. (Measured: on level 2, standing at the one floor cell east of the wall
  pillar links the crate three cells west of that pillar, and pressing ``left``
  moves the crate while the player stands still against the wall.)
* **The ellipsis also spans ZERO cells**, so a crate directly beside the player
  is linked too.
* **A press the player cannot make still moves the crates.** The rule matches
  ``moving player``, which is a queued force, and forces are resolved only after
  every rule has run. Walking into a wall therefore moves every linked crate one
  step and leaves the player where it was -- which is not a curiosity, it is the
  primitive most of these fourteen boards are solved with.
* **Enchanted crates are not pushed and do not push.** No rule names them on the
  pushing side, and they share the player's collision layer, so walking into one
  is a dead press (for the player -- the crates still move) and a linked crate
  stops dead against a wall, a plain crate or another enchanted crate.
* **A blocked crate blocks nothing else.** Each linked crate resolves on its own,
  except where bodies are in a row along the direction of travel: they form one
  chain and move together or not at all, so a crate immediately BEHIND the player
  (in the direction of the press) is frozen whenever the player is.

Fourteen levels, 6x6 to 8x11, all winnable, none already won at reset. Levels 0
and 8 have no enchanted crate at all (0 is a bare walk, 8 a plain two-crate
sokoban) and are the game's tutorial for the two halves it then combines.

Everything above is what ``--selfcheck`` measures: a seeded random walk on every
level with the model's board compared against the engine's grid after EVERY
press, including the presses that do nothing, which is where a collision-layer
or force-resolution mistake hides.

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`), searched to an exact distance
field rather than heuristically. The shared `PSSokobanExpert` cannot be used at
all here and the reason is not tuning: its macro is "walk to a push cell, then
push", which presumes the player's walk is INERT. In this game a walk is the
mechanic -- every step of it drags every linked crate -- so there is no such
thing as a free approach and no macro to build.

`_Board.solve` is two breadth-first sweeps over ``(player, enchanted crates,
crates)`` states:

  1. FORWARD from the state being planned, stopping at the depth ``d*`` of the
     first win. That fixes the answer's length and collects ``F``, every state
     within ``d*`` presses of the start.
  2. BACKWARD from the winning configurations, pruned to ``F``.

Unlike the plain sokobans in this family, the backward sweep runs over edges
RECORDED during the forward one rather than over an inverted move. Inverting a
press here would mean inverting a simultaneous multi-body move -- an unknown
number of crates, some of them blocked, the player possibly not among the movers
-- and a predecessor enumerator that got one of those cases wrong would be a
silent wrong answer rather than a crash. The forward sweep already visits every
edge it needs, so keeping them costs one list per state and owes nothing to a
second derivation of the mechanic. These spaces are small enough for that to be
free: the whole game is 58,188 reachable boards, the largest single level 21,477.

Pruning the backward sweep to ``F`` costs nothing that matters: if a state ``s``
lies on any shortest start-to-win path then its whole continuation has forward
distance at most ``d*``, so that continuation never leaves ``F`` and the field's
value at ``s`` is its true distance. States off every shortest path may be
over-estimated, and none of them is ever asked about.

Optimal-action sets
-------------------
The field gives them EXACTLY, with no inference: at a state ``d`` presses from a
win, the optimal presses are every direction whose successor is ``d - 1`` from
one. ``--selfcheck`` re-derives every set on every level by brute force (a fresh
depth-bounded BFS from each of the four successors) and requires an exact match;
``--engine`` re-derives them a third time from the interpreter. No step ever
ships unlabelled (the always-emit-optimal-targets rule).

The ties this game has are not a sokoban's free walk-interleavings -- there are
no free walks here, since every step of one drags every linked crate. They are
the several different presses that happen to leave the board one step nearer a
win, and ``--plans`` counts them: 11 of the 160 expert steps across the fourteen
levels have a second equally-right press, spread over 8 of the levels.

Shortest, and how that is known
-------------------------------
Every plan is provably shortest, and the proof is three independent derivations
agreeing on all fourteen levels:

  * this file's field (forward BFS + backward BFS over the recorded edges);
  * a plain forward BFS to the first win, which is shortest by construction and
    shares no code with the field's backward half (``--selfcheck``);
  * a full-space enumeration of the real INTERPRETER -- `StateGraph.build`
    walking every reachable engine state by pressing real buttons, with no native
    model involved at all (``--engine``). That is the derivation that closes the
    loop back to the engine, and it agrees on every length AND on every tie set.

The third pass is exhaustive rather than a spot check because this game is small
enough to afford it, which the big ps: sokobans are not. It enumerates 58,174
engine states across the fourteen levels (11 to 21,476 per level -- one fewer
than the native count each time, because a won board is terminal and never
keyed) and returns 0 disagreements: all fourteen lengths and all 160 tie sets.

Recovery
--------
`supports_recovery` with RESET-mode recovery. The expert re-plans from the LIVE
board, not from a stored path: `_search` reads whatever the engine currently
holds and runs both sweeps from there, so an exploration prefix that leaves the
board anywhere -- including somewhere a shortest plan would never go -- is
answered exactly. A prefix CAN strand this game for good: the plain crates
deadlock in corners the ordinary way, and an enchanted crate is much harder to
strand but its TARGET can be made unreachable. The field says so by returning
None, which is exactly what `record_level` needs to hear to fall back to a RESET.
``--selfcheck``'s fourth pass walks random boards on every level and requires
both halves to be right against the interpreter: every plan it returns from a
random board is replayed and must WIN, and every None must be a board an
independent forward BFS also proves dead, never a give-up. That is not a rare
corner: 157 of its 1,120 random boards are genuinely unwinnable, 42 of them on
level 1 alone, whose single enchanted crate has a pocket it can be linked into
and never linked back out of.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variable is the frame rotation
(``rotation_k`` in 0..3) with its matching directional action remap. 14 levels x
4 rotations = 56 presentations.

No flips. The structural argument for the rotation is that the mechanic names no
axis -- the sokoban rule is written with the relative force ``>``, and the
enchanted rule is an undirected four-way expansion whose row case and column case
are the same sentence -- and ``--symmetry`` measures it: every level's plan AND a
seeded random walk (which does what a plan never does: jams crates into walls and
into each other, and presses the unbound ACTION key) replayed at all four
rotations, requiring every frame to be the exact transform of the unrotated one.
Adding flips would need the same evidence for the art, and the art is where it
would fail: the little figure is a person, and the four ``active`` badges are a
DIRECTION drawn on a crate, so a mirror turns the "player is to my left" badge
into the "player is to my right" one -- a mirrored frame would be a legal frame
of a different board. Rotation is safe for exactly the reason a mirror is not:
the four badges are a rotation orbit of each other, so a turned board is a real
board, which is what ``--audit``'s second pass checks.

Rendering
---------
Two things had to change, and both were the puzzle itself: the walls rendered in
the floor's colour, and the player erased every target it stood on. See the
header comment in ``data/puzzlescript_games/Strange_Warehouse.txt`` for the full
statement.

``--audit`` renders every cell COMPOSITION the game can show -- wall, floor, the
three targets, the player, a plain crate, an enchanted crate, its four ``active``
badges, and each of those eight bodies on each of the three targets, 33 in all --
as a whole 64x64 frame of a uniform board, at every cell size the fourteen levels
render at (5, 6, 7, 8 and 10 px), and requires them pairwise distinct. Whole
frames rather than one cell sliced out of a mixed board: `_render_frame` upscales
and centre-pads, so slicing by ``cell_px`` arithmetic reads the wrong pixels (the
ps:explod lesson). Compositions that share the body layer (player + crate, crate
+ enchanted crate) are absent on purpose and are not an omission: those objects
share a collision layer, so the engine can never put them in one cell.

Usage (run from the repo root):
    python solvers/generate_strange_warehouse_training.py --episodes 200 \
        --out data/training_multi_level/strange_warehouse

    python solvers/generate_strange_warehouse_training.py --plans      # level report
    python solvers/generate_strange_warehouse_training.py --selfcheck  # model + optimality
    python solvers/generate_strange_warehouse_training.py --engine     # interpreter proof
    python solvers/generate_strange_warehouse_training.py --audit      # rendering
    python solvers/generate_strange_warehouse_training.py --symmetry   # augmentation
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

GAME_NAME = "Strange_Warehouse"

#: Engine directions, in the order ties are broken. ACTION5 is bound to nothing
#: in this game -- the prelude says ``noaction`` and no rule has ``action`` on a
#: left-hand side -- so it is not a move and branching on it would inflate every
#: search for nothing. The exploration prefix still presses it (a live agent has
#: that button), which is why ``--symmetry``'s random walk presses it too.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Disk cache of every level's start plan AND its optimal-action sets. All
#: fourteen fields together take well under a second, so this is not the
#: load-bearing cache it is on the big ps: sokobans -- it is here so a
#: `parallelize_generator` fan-out shares one derivation rather than repeating it
#: on every core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "strange_warehouse_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic, plus the search.

    Cells are flat ``r * w + c`` indices and a STATE is ``(player, sorted tuple
    of enchanted crates, sorted tuple of plain crates)`` -- the only three things
    any rule in this game moves. Walls and the three target sets never change, so
    they live here rather than in the state.

    THE MECHANIC, as one press. Every body that moves this turn moves in the
    SAME direction ``d`` (there is no rule in this game that produces a force in
    any other one), which is what collapses the interpreter's general
    force-resolution loop into the small thing below:

      1. The player is forced.
      2. The plain crate directly ahead of the player, if there is one, is forced
         -- the sokoban rule.
      3. Every enchanted crate sharing the player's ROW or COLUMN is forced --
         the ellipsis rule, which constrains nothing between the two ends, so
         walls and other bodies in the way are irrelevant to whether the link
         exists.
      4. Resolution: walk each forced body forward. It moves unless it runs off
         the board, into a wall, or into an unforced body; a forced body in the
         way is part of the same chain and the whole chain shares that answer.

    Nothing here reads whether the PLAYER's own step succeeded, which is the
    clause that makes the game: pressing into a wall is a full turn for every
    linked crate and a no-op for the player. And a body directly behind the
    player along ``d`` is chained to it, so that same press freezes THAT crate
    too.

    Targets appear in exactly one place -- the win test -- because each target
    class is alone on its own collision layer and blocks nothing.
    """

    __slots__ = ("h", "w", "wall", "et", "gt", "ct", "_edge", "_row", "_col")

    def __init__(self, h: int, w: int, walls, etargets, gtargets, ctargets):
        self.h, self.w = h, w
        self.wall = bytearray(h * w)
        for i in walls:
            self.wall[i] = 1
        #: The three win sets. Each condition is ``all <body> on <target>``, i.e.
        #: every BODY is on a target -- a subset test, not "every target is
        #: covered". That is the same thing on all fourteen shipped boards (the
        #: counts match) and it is what the .txt literally says, so an edited
        #: level with a spare target would still be answered correctly.
        self.et = frozenset(etargets)
        self.gt = frozenset(gtargets)
        self.ct = frozenset(ctargets)
        #: ``_edge[cell][di]`` is the neighbour cell, or -1 off the board; built
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
        #: Row / column of every cell. The link test runs once per enchanted
        #: crate on every one of the ~200k successors a level's sweep generates,
        #: so it does not do division.
        self._row = [i // w for i in range(h * w)]
        self._col = [i % w for i in range(h * w)]

    # -- the mechanic --------------------------------------------------------
    def step(self, state, di: int):
        """The state after pressing ``DIRS[di]``, or ``state`` itself if the
        press moves nothing at all (which is a non-move: no shortest path
        contains one)."""
        p, es, cs = state
        edge, wall = self._edge, self.wall
        row, col = self._row, self._col
        pr, pc = row[p], col[p]

        forced = {p}
        ahead = edge[p][di]
        if ahead >= 0 and ahead in cs:          # the sokoban push
            forced.add(ahead)
        for e in es:                            # the ellipsis link
            if row[e] == pr or col[e] == pc:
                forced.add(e)

        occupied = set(es)
        occupied.update(cs)
        occupied.add(p)

        # Chain resolution. Every forced body is walked forward until the answer
        # is known; the walk strictly advances in ``di``, so it cannot loop, and
        # every body it passed shares the answer (they are one chain).
        moves: dict[int, bool] = {}
        for start in forced:
            if start in moves:
                continue
            chain = []
            cur = start
            while True:
                if cur in moves:
                    ok = moves[cur]
                    break
                chain.append(cur)
                nxt = edge[cur][di]
                if nxt < 0 or wall[nxt]:
                    ok = False                  # the board edge, or a wall
                    break
                if nxt not in occupied:
                    ok = True                   # free cell: the chain lands
                    break
                if nxt not in forced:
                    ok = False                  # a body that is not moving
                    break
                cur = nxt                       # same direction: same chain
            for cell in chain:
                moves[cell] = ok

        if not any(moves.values()):
            return state
        return (
            edge[p][di] if moves[p] else p,
            tuple(sorted(edge[e][di] if moves.get(e) else e for e in es)),
            tuple(sorted(edge[c][di] if moves.get(c) else c for c in cs)),
        )

    def won(self, state) -> bool:
        p, es, cs = state
        return (p in self.gt and self.et.issuperset(es)
                and self.ct.issuperset(cs))

    # -- the search ----------------------------------------------------------
    def field(self, state):
        """``(dist, d_star)``: presses-to-win for every state on a shortest path
        from ``state``, and the length of that path. ``(None, None)`` if this
        board can never be won from here.

        Sweep 1 is a forward BFS stopped at the depth of the first win, which
        both fixes ``d_star`` and collects ``F`` (every state within ``d_star``
        presses of the start). Sweep 2 is a backward BFS from the won boards over
        the edges sweep 1 recorded, which is every edge out of every state it
        expanded -- and it expands exactly the states with forward distance below
        ``d_star``.

        Recording the edges rather than inverting a press is deliberate; see the
        module docstring. It also makes the prune exact where it is read: if
        ``s`` lies on a shortest start-to-win path then everything after it on
        that path has forward distance at most ``d_star``, so the backward sweep
        reaches ``s`` with its true distance. States on no shortest path can come
        out too high; nothing asks about those, because `solve` and `optimal`
        only ever compare against ``d_star - i``.
        """
        if self.won(state):
            return {state: 0}, 0
        depth = {state: 0}
        queue = deque([state])
        rev: dict = {}
        d_star = None
        while queue:
            cur = queue.popleft()
            d = depth[cur]
            if d_star is not None and d >= d_star:
                break
            for di in range(4):
                nxt = self.step(cur, di)
                if nxt == cur:
                    continue
                rev.setdefault(nxt, []).append(cur)
                if nxt in depth:
                    continue
                depth[nxt] = d + 1
                if self.won(nxt):
                    if d_star is None:
                        d_star = d + 1
                    continue                    # a won board is terminal
                queue.append(nxt)
        if d_star is None:
            return None, None

        dist = {}
        back = deque()
        for s in depth:
            if self.won(s):
                dist[s] = 0
                back.append(s)
        while back:
            cur = back.popleft()
            for prev in rev.get(cur, ()):
                if prev not in dist:
                    dist[prev] = dist[cur] + 1
                    back.append(prev)
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
            if not best:                # unreachable: dist[cur] > 0 has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class StrangeWarehouseExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Board` field.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes, the way `PSEnumExpert` replaces it. Here the
    strategy is "read the live board, sweep it forwards then backwards, hand back
    the shortest plan and its exact tie sets", so `heuristic` is never called and
    asserts rather than returning a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered exactly --
    including "this board is now dead", which an exploration prefix really can
    produce here."""

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, which is canonical WITHIN a level but
    #: not across them -- walls and the three target sets are static per level
    #: and differ between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("wall"))
        #: The or-group, i.e. the plain sprite AND the four ``active`` badges.
        #: A badge is a pure function of the crate's own cell and the player's,
        #: so folding all five to one tag below loses nothing.
        self.ench_ids = set(g.resolve_object_name("enchanted_crate"))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.et_ids = set(g.resolve_object_name("enchanted_crate_target"))
        self.gt_ids = set(g.resolve_object_name("guy_target"))
        self.ct_ids = set(g.resolve_object_name("crate_target"))
        #: `_Board`s by STATIC signature (see `read`), not by level index.
        self._boards: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "StrangeWarehouseExpert reads an exact distance field; "
            "heuristic is unused")

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples for the player (0), the enchanted crates
        (1) and the plain crates (2).

        Built from the MODEL's reading rather than from raw object ids so that
        the five enchanted-crate sprites collapse to one tag. Keying the badge
        would not alias two boards (the badge is determined by the player cell,
        which is in the key too), but it would make the ``plan_cache_path``
        signature depend on presentation-only state, and the disk layer stores
        that signature to decide whether a cached plan is still this board's."""
        _board, (p, es, cs) = self.read(eng)
        if p is None:
            return frozenset()
        w = eng.width
        return frozenset(
            {(p // w, p % w, 0)}
            | {(e // w, e % w, 1) for e in es}
            | {(c // w, c % w, 2) for c in cs})

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid.

        Boards are cached by their STATIC signature (dimensions, walls, the three
        target sets), not by level index, so the sweeps never rebuild the edge
        table and a board is shared by every state of its level."""
        h, w = eng.height, eng.width
        walls, et, gt, ct = [], [], [], []
        es, cs, player = [], [], None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.et_ids:
                    et.append(i)
                if cell & self.gt_ids:
                    gt.append(i)
                if cell & self.ct_ids:
                    ct.append(i)
                if cell & self.ench_ids:
                    es.append(i)
                if cell & self.crate_ids:
                    cs.append(i)
                if cell & self.player_ids:
                    player = i
        sig = (h, w, tuple(walls), tuple(et), tuple(gt), tuple(ct))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, et, gt, ct)
        return board, (player, tuple(sorted(es)), tuple(sorted(cs)))

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if state[0] is None:                 # no player on the board
            return None
        if board.won(state):
            return Plan([], [])
        if (len(state[1]) > len(board.et) or len(state[2]) > len(board.ct)
                or not board.gt):
            # Not reachable on the shipped levels (every board has as many
            # crates of each kind as targets, and a guy target) and not a crash
            # if a level is ever edited: with more bodies than targets the
            # corresponding win condition is unsatisfiable.
            return None
        return board.solve(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class StrangeWarehouseSolver(PSAStarSolver):
    game_id = "puzzlescript_strange_warehouse"
    game_name = GAME_NAME
    expert_cls = StrangeWarehouseExpert

    #: `games/ps:strange_warehouse/ps:strange_warehouse.py` is a plain
    #: passthrough (it constructs the adapter and nothing else), so there is
    #: nothing to gain by routing through it -- but it IS what a live agent is
    #: handed, so if that wrapper ever grows a patch this must be set to
    #: ``"ps:strange_warehouse"``.
    game_module_id = ""

    #: Unused: the expert enumerates an exact field rather than searching under a
    #: heuristic, so there is no weight to trade and no node budget that could
    #: quietly bite. Left at the base values so nothing reads a lie off them.
    weight = 1
    node_cap = 400_000

    #: The longest plan is 19 presses (level 8); the rest is room for a re-plan
    #: after the exploration prefix. Stays well under the adapter's own 200-step
    #: per-level budget, which `set_level` resets before the plan starts anyway.
    max_steps = 100

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
    solver = StrangeWarehouseSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once."""
    _solver, game, expert, solvable = _new()
    eng = game._engine
    total = ties = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        _board, (_p, es, cs) = expert.read(eng)
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
        print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  "
              f"{len(es)} enchanted / {len(cs)} plain crates  "
              f"{len(plan):3d} presses  win={won}  "
              f"(budget {game._max_steps}, {room})  "
              f"{tie_steps:3d} steps with a tie set")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _brute(board, state, limit):
    """Presses to a win from ``state``, searched fresh, or None past ``limit``.

    Shares nothing with `_Board.field` but `_Board.step` itself, which is the
    point: it is the independent answer every optimality claim below is checked
    against."""
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


def _selfcheck(walk_presses: int = 400, samples: int = 80,
               verbose: bool = True) -> int:
    """Four things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's.** A seeded random walk on every
       level, comparing `_Board.step`'s board against the engine's grid after
       EVERY press -- including the presses that do nothing, which is where a
       collision-layer or force-resolution mistake hides. The walk is what
       measures the mechanic listed in the module docstring: it presses into
       walls (the crates move, the player does not), it links crates through
       walls, it jams a linked crate against a plain one, and it lines a crate up
       directly behind the player so the two share a chain.
    2. **The plans are shortest.** A plain forward BFS to the first win, which is
       shortest by construction and shares no code with the field's backward
       half, must return the field's plan length on every level.
    3. **The tie sets are exact.** Every step's optimal set is re-derived by
       brute force -- a fresh depth-bounded BFS from each of the four successors
       -- and must match the field's answer exactly. This game is small enough to
       afford that on EVERY level, so nothing here is checked by sampling.
    4. **Recovery is answered from ANY board, not just the start.** This is the
       one that matters at generation time: the exploration prefix really does
       strand this game, so `_search` has to be right about arbitrary states in
       both directions. From ``samples`` states reached by seeded random presses,
       the field must either return a plan that WINS when replayed through the
       interpreter, or return None on a board an independent forward BFS also
       finds no win from. A None that is merely a give-up would be a silent bug:
       `record_level` reads it as "reset", and a wrong plan length here would not
       show up in any of the three checks above, all of which only ever ask about
       starts.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0

    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        rng = random.Random(f"strange_warehouse:selfcheck:{level}")
        drift = 0
        for _ in range(walk_presses):
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
                  f" the interpreter over {walk_presses} random presses")

    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        plan = expert.plan(eng, level)
        shortest = _brute(board, state, board.h * board.w * 4)
        ok = plan is not None and shortest == len(plan)
        bad += not ok
        if verbose:
            print(f"  L{level:2d}: field {len(plan) if plan else None} presses vs "
                  f"BFS {shortest} -- {'SHORTEST' if ok else 'NOT SHORTEST'}")

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
                if _brute(board, nxt, rest - 1) == rest - 1:
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
        rng = random.Random(f"strange_warehouse:recovery:{level}")
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
            truth = _brute(board, state, board.h * board.w * 4)
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
    modelled link, no modelled chain, no modelled win test -- so agreement on
    both the plan LENGTH and the per-step optimal SET is an independent
    derivation of everything this file claims.

    Affordable only because the whole game is ~58k engine states; it is still the
    slowest report here by a wide margin (every state costs four interpreter
    ticks), so it is a one-off proof rather than something to run per change."""
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
            print(f"  L{level:2d}: {len(graph.succ):6d}"
                  f" engine states, {len(eplan):3d} presses vs the field's "
                  f"{len(plan):3d} -- {'AGREE' if same_len else 'DISAGREE'}; "
                  f"tie sets {'AGREE' if same_sets else 'DISAGREE'}")
    return bad


#: Every cell stack the engine can put on the board. The body layer holds at most
#: one of player / crate / enchanted crate (they share a collision layer), the
#: target layer at most one of the three targets, and a wall cell carries no
#: ground at all -- it is the one composition that is not floor-based.
_BODIES = [(), ("spr_player",), ("spr_crate",), ("spr_enchanted_crate",),
           ("spr_enchanted_crate_active_up",),
           ("spr_enchanted_crate_active_down",),
           ("spr_enchanted_crate_active_left",),
           ("spr_enchanted_crate_active_right",)]
_TARGETS = [(), ("spr_guy_target",), ("spr_enchanted_crate_target",),
            ("spr_crate_target",)]


def _compositions():
    comps = [("spr_wall_1",)]
    for body in _BODIES:
        for target in _TARGETS:
            comps.append(("spr_ground",) + target + body)
    return comps


def _comp_name(comp) -> str:
    return "+".join(o.replace("spr_", "") for o in comp
                    if o != "spr_ground") or "floor"


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the fourteen boards use (5, 6,
    7, 8 and 10 px; the size list is derived rather than assumed so an edited
    level is covered). Whole 64x64 frames of uniform boards are compared rather
    than one cell out of a mixed board: `_render_frame` upscales and centre-pads,
    so slicing a cell by ``cell_px`` arithmetic reads the wrong pixels (the
    ps:explod lesson). Two uniform boards render identically iff their cells do.

    Only ``spr_wall_1`` is shot: after the rendering fix all five wall sprites
    are the same solid block, which pass 1 also proves rather than assumes.

    **Pass 2 -- the group.** The game is augmented with a frame ROTATION, so a
    board is drawn at any of the four turns, and the sprites are not all
    invariant under them -- the four ``active`` badges are a rotation ORBIT of
    each other, which is exactly what makes turning safe: a turned badge is
    another real badge of the turned board. What would not be safe is a turn of
    one composition landing on a DIFFERENT composition of the same board, so that
    is what this checks: all four turns of each, against all others, with the
    badges' own orbit excluded because it is the mechanic being drawn correctly
    rather than a collision. It runs on SQUARE boards, because a rotation of a
    non-square board also moves the letterbox and every comparison would pass for
    the wrong reason.
    """
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert, _solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = _compositions()

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
            print(f"      IDENTICAL: {_comp_name(a)}  ==  {_comp_name(b)}")

    badges = {"spr_enchanted_crate_active_up", "spr_enchanted_crate_active_down",
              "spr_enchanted_crate_active_left",
              "spr_enchanted_crate_active_right"}

    def orbit_pair(a, b) -> bool:
        """True when ``a`` and ``b`` differ only by which badge they carry."""
        ba, bb = set(a) & badges, set(b) & badges
        return bool(ba) and bool(bb) and (set(a) - ba) == (set(b) - bb)

    for cell in sorted({min(64 // h, 64 // w) for h, w in sizes}):
        n = 64 // cell
        shots = {c: shoot(n, n, c) for c in comps}
        clashes = [(a, b, k) for a, b in itertools.permutations(comps, 2)
                   for k in range(1, 4)
                   if not orbit_pair(a, b)
                   and np.array_equal(np.ascontiguousarray(
                       np.rot90(shots[a], k=k)), shots[b])]
        invariant = [_comp_name(c) for c in comps
                     if all(np.array_equal(np.ascontiguousarray(
                         np.rot90(shots[c], k=k)), shots[c]) for k in range(4))]
        bad += len(clashes)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(clashes)} composition(s) "
                  f"whose turn is another's art; turn-invariant: "
                  f"{', '.join(invariant)}")
        for a, b, k in clashes:
            print(f"      {_comp_name(a)} turned {k} == {_comp_name(b)}")
    return bad


def _symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for the rotation augmentation. The structural argument
    is in the module docstring -- the sokoban rule is written with the relative
    force ``>``, and the enchanted rule's four expansions are one undirected
    sentence -- but Gobble Rush's chirality hid inside exactly this kind of
    argument, so it is measured.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (a linked crate flat against a wall, two crates
    jammed together, a plain crate dead in a corner) and it presses the unbound
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
        if len(seen) == 4 and seed > 40:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"strange_warehouse:symmetry:{level}")
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
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(StrangeWarehouseSolver.main())
