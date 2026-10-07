"""Generate Phase-1 training data for the PuzzleScript game ps:switcheroo
("Switcheroo", CNIAngel).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the exact distance field that model is solved
with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_switcheroo",
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
Eight levels. You walk with the arrow keys and you cannot push anything: the
black BLOCKS (`Temp` in the source) are as immovable as the walls. The X button
is the whole game -- it TELEPORTS you into a block, and the block into the
square you were standing on. The win is ``All Exit on Temp``: every green exit
square covered by a block.

Which block, and whether the button does anything at all, is decided by the
BEAMS, and the beams are the part worth stating precisely because the level art
suggests something narrower than the rule:

  * a block lights its ENTIRE row with HBeam and its ENTIRE column with VBeam
    (``late horizontal [ Temp | ] -> [ Temp | HBeam ]`` and its VBeam twin,
    then two propagation rules with the same shape, all of which run to
    fixpoint over a blank pattern cell -- and a blank pattern cell matches a
    WALL as happily as a floor). So a beam crosses walls, crosses other blocks
    and stops only at the edge of the board. Measured, not read off the rules:
    ``--selfcheck``'s first pass compares the lit set against "the rows and
    columns of the blocks" on every level;
  * so at any settled board, standing on HBeam means "a block shares my ROW"
    and standing on VBeam means "a block shares my COLUMN", and nothing else;
  * ``[ Action Player VBeam HBeam ] -> [ Player VBeam HBeam ]`` is the CANCEL:
    on a square where a row beam and a column beam cross, X does nothing. This
    is what makes the game a puzzle rather than a shuffle, and it is why the
    approach square matters more than the route to it;
  * ``[ Action Player HBeam no VBeam | ... | Temp ] -> [ Temp | ... | Player ]``
    (and its VBeam twin) is the teleport. The ellipsis means the block can be
    any distance away and WALLS BETWEEN YOU DO NOT MATTER -- you swap straight
    through them. With exactly one beam only one axis can hold a block at all,
    so the direction is forced; the block taken is the NEAREST one along it.

Two levels stretch that: level 5 and level 6 have TWO blocks and two exits (so
a row beam and a column beam can come from different blocks, and half the
squares in the level are cancel squares), and level 7 has TWO PLAYERS on one
key and a single block they have to hand between them.

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`) plus the EXACT distance to a
win for every reachable state (`_Space`). No heuristic and no search: the
reachable spaces are 181 to 19812 states, so each level's is enumerated whole
and solved by one backward breadth-first sweep from the winning states. That
buys three things a search cannot have:

  * plans that are provably SHORTEST against the entire space -- no weight, no
    node cap that could quietly bite;
  * EXACT optimal-action sets (``dist(successor) == dist - 1``), with nothing
    inferred;
  * RECOVERY as a dictionary lookup from wherever the board happens to be,
    which is the whole `supports_recovery` contract -- and here it can never
    come back empty, because the enumeration also proves the fact it rests on:
    **every reachable state of every level is still winnable**, 0 exceptions
    out of 23436 states (``--selfcheck`` pass 4, and ``--enumerate`` again over
    the real interpreter). This game strands nobody.

The whole derivation, all eight levels, costs one second.

Modelling the ACTION press exactly
----------------------------------
With one player the teleport is simple and self-limiting: you land on the
block's old square, which is lit BOTH ways (it is a block's row and a block's
column), so the cancel rule catches you and no press ever teleports twice.
Level 7's two players are what force the model to be written as the
interpreter's rule loop rather than as that summary, and it is written that
way for every board:

  * the beams are FROZEN for the whole press. They are cleared and redrawn only
    at the end of the tick, so a player is judged against where the blocks were
    when the button went down, not where they are by the time its turn comes;
  * the two teleport rules run in file order (HBeam then VBeam), each over its
    four directions in the interpreter's ``up, down, left, right`` order, each
    direction a forward scan (players taken from the far end opposite the
    direction) and then -- this engine's own second pass -- a reverse scan, the
    whole thing repeated until a pass changes nothing;
  * a player that has teleported has spent its Action and cannot be matched
    again, but a player that has NOT can be matched by a later iteration
    against a block some other player just moved. That is not a corner case in
    level 7: it is how the block is passed between the two of them.

None of this is argued from the rule text. ``--selfcheck``'s pass 1 fuzzes the
model against the interpreter on the shipped levels (3000 random presses each,
whole board compared after every one) and pass 2 does it again on 4000 RANDOM
boards with random walls, up to three players and up to three blocks -- which
is the only way to reach the multi-player, multi-block collisions the eight
shipped levels barely touch.

Shortest, and how that is known
-------------------------------
Every plan is provably shortest, and the proof is three independent derivations
agreeing on all eight levels:

  * this file's backward sweep over the enumerated space;
  * a plain FORWARD BFS to the first win over the same model, which is shortest
    by construction and shares no line with the backward sweep
    (``--selfcheck`` pass 5); its per-step tie sets are re-derived a second
    time by brute force (pass 6);
  * the same two sweeps driven by the REAL INTERPRETER -- forward BFS from the
    level start by pressing actual buttons until a win appears, edges recorded,
    then backward over those edges (``--engine``). No `_Board` call of any kind
    participates: the successor of a board is whatever `PSEngine.step` makes of
    it and the goal test is `check_win`. It agrees on all eight lengths and on
    all 176 tie sets.

``--enumerate`` adds a fourth: the shared `StateGraph.build` walks the
interpreter's ENTIRE reachable space and solves the graph exactly, which is
also where the "nothing is ever stranded" claim is re-measured without the
native model.

Recovery
--------
`supports_recovery` with the family's RESET-mode prefix, plus ``epsilon = 0.12``
detours inside the expert replay. The expert re-plans from the LIVE board (a
lookup at whatever state the engine is in), so both kinds of perturbation are
answered exactly: the taken action is the mistake and ``optimal`` is the
recovery, which is the signal a policy needs after its own error.

The detours actually happen rather than being silently skipped, and for a
stated reason: no reachable board of this game is dead, so `record_level`'s
"is this still winnable" probe always passes. It also means the RESET that ends
the exploration prefix is the only RESET a recording ever contains.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed
ASCII maps), so the only per-(seed, level) variables are the frame rotation
(``rotation_k`` in 0..3) and the two flips, with their matching directional
action remap. 8 levels x 16 presentations = 128.

``Switcheroo`` is in `PuzzleScriptAdapter._FLIP_GAMES`; the argument is recorded
there and ``--symmetry`` measures it -- every level's plan AND a seeded 200-press
random walk (which walks into walls and blocks, presses X on cancel squares
where it does nothing, and teleports blocks back off the exits they were
already on) replayed at all 16 presentations, requiring every frame to be the
exact transform of the unaugmented one.

Rendering
---------
One sprite had to change, and it was the cancel rule itself: HBeam and VBeam
shipped as the SAME four lightblue corners, on two different collision layers,
so "one beam" and "two beams" -- teleport and no-op -- were the same four
pixels. VBeam now takes the four edge MIDPOINTS instead. The full statement,
including why both pixel sets have to be orbits of the symmetry group, is the
header comment in ``data/puzzlescript_games/Switcheroo.txt``.

``--audit`` renders every cell COMPOSITION a board of this game can hold -- the
seven stacks of {nothing, exit} x {nothing, player, wall, block}, each with any
of the four beam combinations, minus wall-under-exit which no level places --
as a whole 64x64 frame of a uniform board, at every cell size the eight levels
render at (5, 6 and 9 px), and requires them pairwise distinct. Whole frames
rather than one cell sliced out of a mixed board: `_render_frame` upscales and
centre-pads, so slicing by ``cell_px`` arithmetic reads the wrong pixels (the
ps:explod lesson).

Usage (run from the repo root):
    python solvers/generate_switcheroo_training.py --episodes 200 \
        --out data/training_multi_level/switcheroo

    python solvers/generate_switcheroo_training.py --plans      # level report
    python solvers/generate_switcheroo_training.py --selfcheck  # model + optimality
    python solvers/generate_switcheroo_training.py --engine     # interpreter proof
    python solvers/generate_switcheroo_training.py --enumerate  # whole-space proof
    python solvers/generate_switcheroo_training.py --audit      # rendering
    python solvers/generate_switcheroo_training.py --symmetry   # augmentation
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

GAME_NAME = "Switcheroo"

#: Engine actions, in the order ties are broken. ACTION is the game, so unlike
#: most of this family it cannot be dropped from the branching.
DIRS: tuple[str, ...] = ("up", "down", "left", "right", "action")

#: The four moves, in the interpreter's own direction-expansion order for a rule
#: with no direction prefix (`PSGame._parse_rules`: up, down, left, right). The
#: teleport rules are such rules, and which block you land on when two are in
#: line depends on that order, so it is not an arbitrary listing.
_MOVES: tuple[str, ...] = DIRS[:4]

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Disk cache of every level's start plan AND its optimal-action sets. The whole
#: game's spaces take well under a second, so this is not the load-bearing cache
#: it is on the big ps: sokobans -- it is here so a `parallelize_generator`
#: fan-out shares one derivation rather than repeating it on every core. Delete
#: the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "switcheroo_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic.

    Cells are flat ``r * w + c`` indices and a STATE is ``(player bitmask, block
    bitmask)`` -- the only two things any rule in this game moves. Walls and the
    exit set never change, so they live here rather than in the state, and both
    dynamic sets are bitmasks because a level has up to three players and blocks
    at once and the sweeps below key tens of thousands of states.

    Players are a SET, not a list: they are one object type with no identity of
    their own, so two boards that differ by which player is which are the same
    state and must not be enumerated twice.

    Blocks are the `Temp` object of the source. They are renamed here because
    "temp" reads as scratch bookkeeping and this one is the puzzle piece.
    """

    __slots__ = ("h", "w", "n", "wall", "emask", "free",
                 "edge", "ray", "row", "col", "sig")

    def __init__(self, h: int, w: int, walls, exits):
        self.h, self.w = h, w
        self.n = h * w
        self.wall = frozenset(walls)
        self.emask = 0
        for e in exits:
            self.emask |= 1 << e
        #: Every non-wall square, in scan order. Players and blocks live here
        #: and nowhere else.
        self.free = tuple(i for i in range(self.n) if i not in self.wall)
        self.row = tuple(i // w for i in range(self.n))
        self.col = tuple(i % w for i in range(self.n))
        #: ``edge[cell][di]`` is the square a WALK reaches, or -1 when the walk
        #: cannot happen at all (off the board or into a wall).
        self.edge = []
        #: ``ray[cell][di]`` is every square beyond ``cell`` in that direction,
        #: nearest first, WALLS INCLUDED -- the teleport passes through them, so
        #: the ray that finds the block must not stop at one. This is the single
        #: place the two tables differ and it is the mechanic's whole geometry.
        self.ray = []
        for i in range(self.n):
            r, c = divmod(i, w)
            row_e, row_r = [], []
            for d in _MOVES:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                j = nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1
                row_e.append(-1 if j < 0 or j in self.wall else j)
                line = []
                rr, cc = nr, nc
                while 0 <= rr < h and 0 <= cc < w:
                    line.append(rr * w + cc)
                    rr += dr
                    cc += dc
                row_r.append(tuple(line))
            self.edge.append(tuple(row_e))
            self.ray.append(tuple(row_r))
        self.sig = (h, w, tuple(sorted(self.wall)), tuple(sorted(exits)))

    # -- the mechanic --------------------------------------------------------
    def step(self, state, ai: int):
        """The state after pressing ``DIRS[ai]``, or ``state`` itself when the
        press moves nothing (which is a non-move: no shortest path contains
        one)."""
        return self._act(state) if ai == 4 else self._move(state, ai)

    def _move(self, state, di: int):
        """A walk. Every player steps if it can; walls, blocks, the board edge
        and other players block it.

        Players are processed FRONT FIRST along the pressed axis, which is what
        lets a train of players move together -- the follower's target is free
        because the leader has already vacated it -- and what makes a train
        against a wall move nothing at all."""
        players, blocks = state
        cells = [i for i in range(self.n) if (players >> i) & 1]
        cells.sort(reverse=di in (1, 3))          # down / right: far end first
        out = players
        for p in cells:
            q = self.edge[p][di]
            if q < 0:
                continue
            if (blocks >> q) & 1 or (out >> q) & 1:
                continue
            out = (out & ~(1 << p)) | (1 << q)
        return (out, blocks)

    def _act(self, state):
        """The X button, as the interpreter's rule loop.

        The beams are computed ONCE, from the blocks as they stand when the
        button goes down, and are not touched again for the rest of the press
        (the rules that clear and redraw them run after both teleport rules).
        `_lit` is that frozen pair of "which rows are lit" and "which columns
        are lit"; every beam test below reads it and never the live blocks.

        Rule order is the file's: the cancel rule spends the Action of every
        player standing on a crossing, then the HBeam teleport runs to fixpoint
        over its four directions, then the VBeam one. See the module docstring
        for why the fixpoint is not decoration.
        """
        players, blocks = state
        P = {i for i in range(self.n) if (players >> i) & 1}
        B = {i for i in range(self.n) if (blocks >> i) & 1}
        lit_r, lit_c = self._lit(B)
        # The cancel rule: a player on a crossing has no Action left to spend.
        act = {p for p in P
               if not (self.row[p] in lit_r and self.col[p] in lit_c)}
        for want_h in (True, False):
            for _ in range(200):                  # the engine's own iteration cap
                changed = False
                for di in range(4):
                    if self._sweep(P, B, act, lit_r, lit_c, want_h, di, False):
                        # This engine follows every firing forward pass with a
                        # reverse-order one (`_apply_single_rule_forces`), and on
                        # a board with several players that second pass is
                        # observable, so the model owes it.
                        self._sweep(P, B, act, lit_r, lit_c, want_h, di, True)
                        changed = True
                if not changed:
                    break
        out_p = 0
        for p in P:
            out_p |= 1 << p
        out_b = 0
        for b in B:
            out_b |= 1 << b
        return (out_p, out_b)

    def _lit(self, blocks):
        """``(rows lit by HBeam, columns lit by VBeam)`` for a block set.

        A block lights its whole row and its whole column, edge to edge, through
        walls and through other blocks -- see the module docstring, and
        ``--selfcheck`` pass 1, which compares this against the interpreter's
        own beam objects rather than trusting the reading."""
        rows, cols = set(), set()
        for b in blocks:
            rows.add(self.row[b])
            cols.add(self.col[b])
        return rows, cols

    def _sweep(self, P, B, act, lit_r, lit_c, want_h: bool, di: int,
               rev: bool) -> bool:
        """One scan of one teleport rule in one direction. Mutates ``P`` / ``B``
        / ``act`` in place and returns whether anything fired.

        The scan ORDER is the interpreter's and it decides the outcome whenever
        two players are on the same line, so it is spelled out rather than left
        to a set's iteration order. The forward pass takes the anchor cells from
        the end the direction points AWAY from (for ``right``, leftmost first),
        the reverse pass takes them the other way; within a line that is the
        only thing that matters, and across lines nothing does, because a match
        never leaves the line it started on."""
        asc = (di in (0, 2)) if rev else (di in (1, 3))
        fired = False
        for p in sorted(P, reverse=not asc):
            if p not in P or p not in act:
                continue
            h = self.row[p] in lit_r
            v = self.col[p] in lit_c
            if not ((h and not v) if want_h else (v and not h)):
                continue
            found = None
            for q in self.ray[p][di]:
                if q in B:                        # the NEAREST block along the ray
                    found = q
                    break
            if found is None:
                continue
            P.discard(p)
            P.add(found)
            act.discard(p)                        # its Action is spent
            B.discard(found)
            B.add(p)
            fired = True
        return fired

    def won(self, state) -> bool:
        return (state[1] & self.emask) == self.emask


class _Space:
    """Every state reachable from a root, with its exact presses-to-win.

    Two sweeps and no heuristic. The forward one enumerates the reachable
    component over `_Board.step`, treating a WON state as terminal (the episode
    ends there, so nothing is reachable through it); the backward one is a
    breadth-first sweep from the won states over the recorded reverse edges,
    which labels every state with its true shortest distance.

    Enumerating the whole component rather than searching is affordable because
    the components are 181 to 19812 states -- the game is small in the way a
    slide puzzle is small, since a block only ever sits where a player has
    stood. It is also what makes the two guarantees this file leans on
    measurable rather than assumed: the plans are shortest against the ENTIRE
    space, and a state with no distance is provably unwinnable rather than
    "the search gave up". There are none of the latter (see `stranded`).

    ``ensure`` re-opens the enumeration from a state the component does not
    contain. Nothing an episode does can produce one -- every board a prefix or
    a detour reaches is reachable from the level start by construction -- so it
    is a contract, not a code path with a use: it means a lookup can never
    silently answer "unwinnable" because it was asked about the wrong root.

    ``cap`` is a runaway guard, not a budget: past it the space answers None,
    which `_search` turns into "no plan" and `record_level` turns into a RESET.
    The largest of the eight is 19812.
    """

    __slots__ = ("board", "succ", "rev", "dist", "cap", "capped", "dirty")

    def __init__(self, board: _Board, cap: int):
        self.board = board
        self.cap = cap
        self.succ: dict = {}
        self.rev: dict = {}
        self.dist: dict = {}
        self.capped = False
        self.dirty = True

    # -- the two sweeps ------------------------------------------------------
    def _expand(self, root) -> None:
        """Breadth-first closure from ``root`` over the five presses, added to
        whatever is already enumerated."""
        board = self.board
        queue = deque([root])
        self.succ.setdefault(root, [])
        while queue:
            cur = queue.popleft()
            if board.won(cur):
                self.succ[cur] = []               # a won board is terminal
                continue
            edges = []
            for ai in range(5):
                nxt = board.step(cur, ai)
                if nxt == cur:
                    continue                      # a press that did nothing
                edges.append((ai, nxt))
                self.rev.setdefault(nxt, set()).add(cur)
                if nxt not in self.succ:
                    if len(self.succ) >= self.cap:
                        self.capped = True
                        return
                    self.succ[nxt] = []
                    queue.append(nxt)
            self.succ[cur] = edges

    def _sweep(self) -> None:
        """Presses-to-win for every enumerated state, backwards from the won
        ones. The distance is to the NEAREST win, which is the right answer even
        though a win ENDS the episode: a shortest path to the nearest win cannot
        pass through another, because that one would be nearer."""
        board = self.board
        dist = {s: 0 for s in self.succ if board.won(s)}
        queue = deque(dist)
        while queue:
            cur = queue.popleft()
            for prev in self.rev.get(cur, ()):
                if prev not in dist:
                    dist[prev] = dist[cur] + 1
                    queue.append(prev)
        self.dist = dist
        self.dirty = False

    def ensure(self, state) -> None:
        if state not in self.succ and not self.capped:
            self._expand(state)
            self.dirty = True
        if self.dirty and not self.capped:
            self._sweep()

    # -- what the expert reads -----------------------------------------------
    def get(self, state) -> "int | None":
        """Presses to a win from ``state``, or None when it can never be won
        (or the cap stopped the enumeration short of knowing)."""
        self.ensure(state)
        return None if self.capped else self.dist.get(state)

    def stranded(self) -> int:
        """How many enumerated states cannot reach a win. 0 on every level of
        this game, which is what the recovery contract rests on."""
        return len(self.succ) - len(self.dist)

    def optimal(self, state) -> list:
        """``[(action index, successor), ...]`` for every press on a shortest
        path from ``state``. Ties come out in ``DIRS`` order, which is what makes
        a re-derived plan byte-identical across processes.

        Exact, with nothing inferred: a press is optimal iff its successor is one
        step nearer the win, and both distances come from the same sweep."""
        rest = self.get(state)
        if not rest:                     # None (unknown / dead) or 0 (already won)
            return []
        return [(ai, nxt) for ai, nxt in self.succ[state]
                if self.dist.get(nxt) == rest - 1]

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
            if not best:                 # a labelled state always has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[ai] for ai, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class SwitcherooExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Space`.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes, the way `PSEnumExpert` replaces it. Here the
    strategy is "enumerate the level's reachable component once and read the
    exact distance off it", so `heuristic` is never called and asserts rather
    than returning a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered exactly
    and in constant time. Unlike most of this family it can also be trusted to
    answer YES: no reachable board of this game is dead (`_Space.stranded` is 0
    on every level), so a None from here would mean the cap was hit, never that
    a board was genuinely lost.
    """

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, which is canonical WITHIN a level but
    #: not across them -- walls and exits are static per level and differ between
    #: levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Runaway guard on one level's enumeration; see `_Space`. The largest this
    #: game reaches is level 6's 19812.
    space_cap: int = 2_000_000

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.block_ids = set(g.resolve_object_name("temp"))
        self.exit_ids = set(g.resolve_object_name("exit"))
        self.hbeam_ids = set(g.resolve_object_name("hbeam"))
        self.vbeam_ids = set(g.resolve_object_name("vbeam"))
        #: `_Board`s by STATIC signature (see `read`) and `_Space`s by that plus
        #: the piece counts, so a component is shared by every state of its level
        #: and the enumeration is paid for once.
        self._boards: dict = {}
        self._spaces: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "SwitcherooExpert enumerates the reachable space; heuristic is unused")

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid.

        Boards are cached by their STATIC signature (dimensions, walls, exits),
        so the edge and ray tables are built once per level however many states
        are read from it. The beams are deliberately NOT read: they are a
        function of the blocks (`_Board._lit`), so reading them would be reading
        the same fact twice and would let a mis-seated board look like a
        different state."""
        h, w = eng.height, eng.width
        walls, exits = [], []
        players = blocks = 0
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.exit_ids:
                    exits.append(i)
                if cell & self.block_ids:
                    blocks |= 1 << i
                if cell & self.player_ids:
                    players |= 1 << i
        sig = (h, w, tuple(walls), tuple(exits))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, exits)
        return board, (players, blocks)

    def space(self, board: _Board, state) -> _Space:
        key = (board.sig, bin(state[0]).count("1"), bin(state[1]).count("1"))
        got = self._spaces.get(key)
        if got is None:
            got = self._spaces[key] = _Space(board, self.space_cap)
        return got

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples for the players (0) and the blocks (1).

        Built from the MODEL's reading rather than from raw object ids so the key
        is exactly the two things a state consists of -- which is also what the
        ``plan_cache_path`` signature is stored as, so a cached plan is matched
        against the board it was solved from and an edited level is a miss rather
        than a wrong plan."""
        _board, (players, blocks) = self.read(eng)
        w = eng.width
        return frozenset(
            {(i // w, i % w, 0)
             for i in range(eng.height * w) if (players >> i) & 1}
            | {(i // w, i % w, 1)
               for i in range(eng.height * w) if (blocks >> i) & 1})

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if not state[0]:                     # no player on the board
            return None
        if board.won(state):
            return Plan([], [])
        return self.space(board, state).plan(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class SwitcherooSolver(PSAStarSolver):
    game_id = "puzzlescript_switcheroo"
    game_name = GAME_NAME
    expert_cls = SwitcherooExpert

    #: `games/ps:switcheroo/ps:switcheroo.py` is a plain passthrough -- it builds
    #: the adapter and nothing else, and the rendering fix is in the .txt, which
    #: both paths read. Set this if that wrapper ever grows a patch.
    game_module_id = ""

    #: Unused: `SwitcherooExpert._search` never calls `_astar`. Left at the base
    #: values so nothing reads a lie off them; `SwitcherooExpert.space_cap` is
    #: the knob that actually bounds the enumeration.
    weight = 1
    node_cap = 400_000

    #: Room for the longest plan (36 presses on level 4) plus the re-plans an
    #: epsilon detour costs. The adapter's own 200-press per-level budget is
    #: separate and is reset by the `set_level` that ends the exploration prefix,
    #: so the plan starts it from zero.
    max_steps = 120

    #: Every level is solved; nothing is skipped.
    skip_levels = frozenset()

    #: On top of the RESET prefix: one press in eight is a random legal
    #: alternative, after which the expert re-plans from wherever it landed --
    #: the taken action is the mistake and ``optimal`` is the recovery, which is
    #: the signal a policy needs after its own error. Safe at any rate here
    #: because no reachable board of this game is dead (see `_Space`), so a
    #: detour can never brick one and `record_level`'s own winnability probe is
    #: never the thing that rejects it.
    epsilon = 0.12

    def prepare_expert(self, game, expert) -> None:
        """Enumerate every level's space before `discover_solvable` asks for it --
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
    solver = SwitcherooSolver()
    game = solver.make_game(seed)
    return solver, game, SwitcherooExpert(game, node_cap=solver.node_cap)


def _ascii(board: _Board, state) -> str:
    """A model state as ASCII, so a failing check can print the board it failed
    on rather than a pair of integers."""
    players, blocks = state
    out = []
    for r in range(board.h):
        line = ""
        for c in range(board.w):
            i = r * board.w + c
            on_exit = (board.emask >> i) & 1
            if i in board.wall:
                line += "#"
            elif (players >> i) & 1:
                line += "p" if on_exit else "P"
            elif (blocks >> i) & 1:
                line += "@" if on_exit else "*"
            else:
                line += "O" if on_exit else "."
        out.append(line)
    return "\n".join(out)


def _start(game, expert, level: int):
    """``(board, state)`` at a level's start, engine left seated there."""
    game.set_level(level)
    return expert.read(game._engine)


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the enumeration and the tie labelling
    at once."""
    _solver, game, expert = _new()
    eng = game._engine
    total = ties = bad = 0
    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
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
        space = expert.space(board, state)
        total += len(plan)
        ties += tie_steps
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"  L{level:2d}: {board.h:2d}x{board.w:2d}  "
              f"{len(board.free):2d} floor  "
              f"{bin(state[0]).count('1')}P {bin(state[1]).count('1')} block(s)  "
              f"{len(plan):3d} presses  win={won}  "
              f"(budget {game._max_steps}, {room})  "
              f"{len(space.succ):6d} states, {space.stranded()} dead  "
              f"{tie_steps:3d} steps with a tie set")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _forward(board: _Board, state, limit: int) -> "int | None":
    """Presses to a win from ``state``, by a plain FORWARD BFS, or None past
    ``limit``.

    Shares nothing with `_Space` but `_Board.step` itself, which is the point: it
    is the independent answer every optimality claim below is checked against."""
    if board.won(state):
        return 0
    seen = {state}
    queue = deque([(state, 0)])
    while queue:
        cur, d = queue.popleft()
        if d >= limit:
            continue
        for ai in range(5):
            nxt = board.step(cur, ai)
            if nxt == cur or nxt in seen:
                continue
            if board.won(nxt):
                return d + 1
            seen.add(nxt)
            queue.append((nxt, d + 1))
    return None


# -- the synthetic boards pass 2 fuzzes on ----------------------------------

def _seat_raw(eng, ids, h, w, walls, players, blocks, exits) -> None:
    """Put an ARBITRARY board on the engine, beams included.

    The beams are written by the model's law (a block lights its whole row and
    its whole column), which is exactly the claim pass 1 measures -- so this is
    only used AFTER that pass, and `_engine` re-checks it per level by requiring
    a seated level start to be byte-identical to the loaded one."""
    bg, wall, player, block, exit_, hbeam, vbeam = ids
    rows = {r for r, _ in blocks}
    cols = {c for _, c in blocks}
    grid = []
    for r in range(h):
        line = []
        for c in range(w):
            cell = {bg}
            if (r, c) in walls:
                cell.add(wall)
            if (r, c) in players:
                cell.add(player)
            if (r, c) in blocks:
                cell.add(block)
            if (r, c) in exits:
                cell.add(exit_)
            if r in rows:
                cell.add(hbeam)
            if c in cols:
                cell.add(vbeam)
            line.append(cell)
        grid.append(line)
    eng.height, eng.width = h, w
    eng.grid = grid
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _selfcheck(walk_presses: int = 3000, boards: int = 4000,
               samples: int = 80, verbose: bool = True) -> int:
    """Six things the expert would otherwise be trusted on.

    1. **The beams are the blocks' whole rows and columns**, and **the native
       model is the interpreter's**. On every level: the lit sets are compared
       against that law at the start, and then a seeded random walk compares
       `_Board.step`'s board against the engine's grid after EVERY press --
       including the presses that do nothing, which is where a collision-layer
       or a cancel-rule mistake hides. The walk presses X far more often than a
       plan does, on and off the crossings.
    2. **The ACTION rule loop is right on boards the eight levels never
       reach.** The same comparison on ``boards`` RANDOM boards -- random walls,
       one to three players, one to three blocks, 25 presses each. This is the
       pass that exercises two players racing for the same block, a player
       teleporting onto a square another just vacated, and the second (reverse
       order) scan this engine runs after every firing pass; with one player
       none of that is observable, and seven of the eight shipped levels have
       one player.
    3. **The enumeration is the whole component.** Each level's space is
       re-derived by an independent forward BFS that keeps no reverse edges and
       must produce the same state set.
    4. **Nothing is stranded.** Every enumerated state has a distance, i.e. every
       board reachable from a level start can still be won. This is the fact the
       recovery contract and the epsilon detours rest on, so it is measured on
       all eight levels rather than argued.
    5. **The plans are shortest.** A plain forward BFS to the first win, which is
       shortest by construction and shares no code with the backward sweep, must
       return the space's plan length on every level.
    6. **The tie sets are exact.** Every step's optimal set is re-derived by
       brute force -- a fresh depth-bounded forward BFS from each of the five
       successors -- and must match. Every level is small enough to afford that,
       so nothing here is sampled.

    A seventh pass answers the question that actually matters at generation
    time: **recovery from ANY board, not just the start.** From ``samples``
    states reached by seeded random presses, the space must return a plan whose
    length the independent forward BFS agrees with and which WINS when replayed
    through the interpreter. By pass 4 none of them can be dead, so a single
    dead board here would be a refutation of that pass, not a rare corner.
    """
    _solver, game, expert = _new()
    eng = game._engine
    ids = (expert.bg_id, next(iter(expert.wall_ids)),
           next(iter(expert.player_ids)), next(iter(expert.block_ids)),
           next(iter(expert.exit_ids)), next(iter(expert.hbeam_ids)),
           next(iter(expert.vbeam_ids)))
    bad = 0

    def beams(engine):
        h_, v_ = set(), set()
        for r, row in enumerate(engine.grid):
            for c, cell in enumerate(row):
                if cell & expert.hbeam_ids:
                    h_.add((r, c))
                if cell & expert.vbeam_ids:
                    v_.add((r, c))
        return h_, v_

    # 1 -- the beam law and the model, on the shipped levels.
    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        lit_r, lit_c = board._lit({i for i in range(board.n)
                                   if (state[1] >> i) & 1})
        want_h = {(r, c) for r in lit_r for c in range(board.w)}
        want_v = {(r, c) for c in lit_c for r in range(board.h)}
        got_h, got_v = beams(eng)
        law = (got_h == want_h and got_v == want_v)
        bad += not law
        rng = random.Random(f"switcheroo:selfcheck:{level}")
        drift = 0
        for _ in range(walk_presses):
            ai = rng.randrange(5)
            eng.step(DIRS[ai])
            state = board.step(state, ai)
            if _read_state(eng, expert) != state or eng.check_win() != board.won(state):
                drift += 1
                break
            if eng.check_win():
                _board, state = _start(game, expert, level)
        bad += drift
        if verbose:
            print(f"  L{level:2d}: beams are the blocks' rows and columns: "
                  f"{'YES' if law else 'NO'}; model "
                  f"{'MATCHES' if not drift else 'DIVERGES from'} the "
                  f"interpreter over {walk_presses} random presses")

    # 2 -- the same, on random boards the shipped levels never produce.
    rng = random.Random("switcheroo:boards")
    drift = multi = 0
    for _ in range(boards):
        h, w = rng.randint(4, 8), rng.randint(4, 9)
        cells = [(r, c) for r in range(h) for c in range(w)]
        walls = set(rng.sample(cells, rng.randint(0, len(cells) // 3)))
        free = [x for x in cells if x not in walls]
        n_p, n_b, n_e = rng.randint(1, 3), rng.randint(1, 3), rng.randint(1, 2)
        if len(free) < n_p + n_b + n_e:
            continue
        pick = rng.sample(free, n_p + n_b + n_e)
        players = set(pick[:n_p])
        blocks = set(pick[n_p:n_p + n_b])
        exits = set(pick[n_p + n_b:])
        multi += n_p > 1
        board = _Board(h, w, [r * w + c for r, c in walls],
                       [r * w + c for r, c in exits])
        state = (sum(1 << (r * w + c) for r, c in players),
                 sum(1 << (r * w + c) for r, c in blocks))
        _seat_raw(eng, ids, h, w, walls, players, blocks, exits)
        for _ in range(25):
            ai = rng.randrange(5)
            eng.step(DIRS[ai])
            state = board.step(state, ai)
            if _read_state(eng, expert) != state or eng.check_win() != board.won(state):
                print(f"    DIVERGED on a random {h}x{w} board after "
                      f"{DIRS[ai]}:\n{_ascii(board, _read_state(eng, expert))}")
                drift += 1
                break
    bad += drift
    if verbose:
        print(f"  {boards} random boards ({multi} of them with more than one "
              f"player), 25 presses each: {drift} divergence(s)")

    # 3 + 4 -- the component is the component, and nothing in it is stranded.
    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        space = expert.space(board, state)
        space.ensure(state)
        seen = {state}
        queue = deque([state])
        while queue:                                  # an independent closure
            cur = queue.popleft()
            if board.won(cur):
                continue
            for ai in range(5):
                nxt = board.step(cur, ai)
                if nxt != cur and nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        same = seen == set(space.succ)
        stranded = space.stranded()
        bad += (not same) + stranded
        if verbose:
            print(f"  L{level:2d}: {len(seen):6d} reachable states -- "
                  f"enumeration {'AGREES' if same else 'DISAGREES'}; "
                  f"{stranded} state(s) that cannot win")

    # 5 -- shortest, against an independent forward BFS.
    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        shortest = _forward(board, state, len(plan) + 1 if plan else 200)
        ok = plan is not None and shortest == len(plan)
        bad += not ok
        if verbose:
            print(f"  L{level:2d}: space {len(plan) if plan else None} presses vs "
                  f"forward BFS {shortest} -- "
                  f"{'SHORTEST' if ok else 'NOT SHORTEST'}")

    # 6 -- the tie sets, re-derived by brute force.
    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        sets = getattr(plan, "optsets", None) or []
        mismatched = 0
        cur = state
        for i, direction in enumerate(plan):
            rest = len(plan) - i
            truth = []
            for ai in range(5):
                nxt = board.step(cur, ai)
                if nxt == cur:
                    continue
                if _forward(board, nxt, rest - 1) == rest - 1:
                    truth.append(DIRS[ai])
            if sorted(truth) != sorted(sets[i]):
                mismatched += 1
            cur = board.step(cur, DIRS.index(direction))
        bad += mismatched
        if verbose:
            print(f"  L{level:2d}: {len(plan)} tie sets "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'}"
                  f" brute force")

    # 7 -- recovery from boards no plan visits.
    for level in range(game.n_levels):
        rng = random.Random(f"switcheroo:recovery:{level}")
        wrong = dead = won_after = 0
        for _ in range(samples):
            game.set_level(level)
            for _ in range(rng.randrange(1, 25)):
                eng.step(DIRS[rng.randrange(5)])
                if eng.check_win():
                    break
            if eng.check_win():
                continue                 # the walk already won; nothing to plan
            board, state = expert.read(eng)
            plan = expert._search(eng)
            truth = _forward(board, state, board.n * 4)
            if plan is None:
                dead += 1
                wrong += 1               # pass 4 says this cannot happen
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
    """The level's static layer -- every cell with the players, the blocks and
    BOTH BEAMS stripped out -- so a state key can be SEATED back onto the engine
    instead of keeping a snapshot of every board it reaches.

    The beams are dynamic here even though nothing presses them: they are a
    function of the blocks, so a static layer that kept them would seat the
    previous board's lighting onto the next one's blocks.

    Taken from the live grid rather than assembled from object ids, so it carries
    whatever the level actually holds (background under walls included) and a
    seated board is byte-identical to the one the level loaded."""
    dynamic = (expert.player_ids | expert.block_ids
               | expert.hbeam_ids | expert.vbeam_ids)
    return [[cell - dynamic for cell in row] for row in eng.grid]


def _seat(eng, static, expert, state) -> None:
    """Put ``state`` back on the engine, beams and all. The inverse of
    `_read_state`; that it IS the inverse is checked before either is used (see
    `_engine`), which is also the per-level check on the beam law."""
    players, blocks = state
    w = len(static[0])
    player = next(iter(expert.player_ids))
    block = next(iter(expert.block_ids))
    hbeam = next(iter(expert.hbeam_ids))
    vbeam = next(iter(expert.vbeam_ids))
    rows = {i // w for i in range(len(static) * w) if (blocks >> i) & 1}
    cols = {i % w for i in range(len(static) * w) if (blocks >> i) & 1}
    grid = []
    for r, row in enumerate(static):
        out = []
        for c, cell in enumerate(row):
            i = r * w + c
            new = set(cell)
            if (blocks >> i) & 1:
                new.add(block)
            if (players >> i) & 1:
                new.add(player)
            if r in rows:
                new.add(hbeam)
            if c in cols:
                new.add(vbeam)
            out.append(new)
        grid.append(out)
    eng.grid = grid
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _read_state(eng, expert):
    """``(player bitmask, block bitmask)`` straight off the engine grid. A grid
    scan and nothing else -- no `_Board`, no mechanic -- so the sweeps below owe
    the native model nothing."""
    w = eng.width
    players = blocks = 0
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if cell & expert.block_ids:
                blocks |= 1 << (r * w + c)
            if cell & expert.player_ids:
                players |= 1 << (r * w + c)
    return (players, blocks)


def _engine(verbose: bool = True) -> int:
    """The proof that owes nothing to the native model: two sweeps driven by the
    REAL INTERPRETER.

    Forward BFS from the level start by pressing actual buttons, keying each
    board by what is on the grid, stopping at the depth the first `check_win`
    appears -- which fixes the shortest length by construction -- with every edge
    recorded; then a backward sweep from the won boards over those edges, which
    gives the exact optimal SET at every state on a shortest path. No `_Board`
    call of any kind participates: successors come from `PSEngine.step` and the
    goal test is `check_win`.

    States are re-seated from their keys rather than snapshotted, which is what
    keeps tens of thousands of boards in megabytes instead of gigabytes; the
    seating is checked to be the exact inverse of the reading, and a seated board
    is checked to step identically to a snapshot-restored one, before any of it
    is trusted. That check is also where the beam law is re-measured per level:
    `_seat` writes the beams from the blocks, so a level whose lighting were not
    that function would fail the byte comparison on its own start board.
    """
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
        for ai in range(5):
            restore(eng, original)
            eng.step(DIRS[ai])
            expected = snapshot(eng)
            _seat(eng, static, expert, start)
            eng.step(DIRS[ai])
            if eng.grid != expected:
                print(f"  L{level:2d}: a seated board steps differently "
                      f"({DIRS[ai]})")
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
            for ai in range(5):
                _seat(eng, static, expert, cur)
                eng.step(DIRS[ai])
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
            for ai in range(5):
                _seat(eng, static, expert, cur)
                eng.step(DIRS[ai])
                nxt = _read_state(eng, expert)
                if nxt != cur and dist.get(nxt) == rest - 1:
                    best.append((ai, nxt))
            epresses.append(DIRS[best[0][0]])
            esets.append(sorted(DIRS[ai] for ai, _ in best))
            cur = best[0][1]

        same_len = len(epresses) == len(plan) == d_star
        same_sets = esets == [sorted(s) for s in plan.optsets]
        bad += (not same_len) + (not same_sets)
        if verbose:
            print(f"  L{level:2d}: {len(depth):6d} engine boards within d*, "
                  f"{d_star:3d} presses vs the model's {len(plan):3d} -- "
                  f"{'AGREE' if same_len else 'DISAGREE'}; tie sets "
                  f"{'AGREE' if same_sets else 'DISAGREE'}")
        game.set_level(level)
    return bad


def _enumerate(levels=None, verbose: bool = True) -> int:
    """Walk the interpreter's WHOLE reachable space with the shared
    `StateGraph.build` and solve the graph exactly.

    A fourth derivation, and the one that shares no algorithm with anything here
    either -- `StateGraph` is `solvers/common/ps_astar.py`'s, written for other
    games. It buys two things `--engine` cannot: plans that are shortest against
    the entire space rather than against a ball, and the "nothing is stranded"
    measurement over the real interpreter instead of the native model.

    Every level is affordable (181 to 19812 states), so all eight run by
    default; pass level numbers on the command line to narrow it."""
    from solvers.common.ps_astar import StateGraph                     # noqa: PLC0415

    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in (range(game.n_levels) if levels is None else levels):
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
                  f"{len(eplan) if eplan else None} presses vs the model's "
                  f"{len(plan)} -- {'AGREE' if same_len else 'DISAGREE'}; "
                  f"tie sets {'AGREE' if same_sets else 'DISAGREE'}; "
                  f"{len(stranded)} state(s) that cannot win")
    return bad


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

#: Every cell stack a board of this game can hold. Background is on its own layer
#: and is everywhere; Exit has a layer to itself; Player, Wall and Temp share the
#: third, so at most one of them is in a cell; HBeam and VBeam have a layer each,
#: so any of the four beam combinations can sit on top of any of those.
#:
#: ``wall + exit`` is absent on purpose and is not an omission: the legend has no
#: character that places one, no level does, and an exit buried under a wall
#: would make its level unwinnable (nothing can put a block there) rather than
#: misread. It is also the one pair this game cannot draw apart -- Wall is
#: opaque.
_COMPOSITIONS = [
    exit_ + mid + beam
    for exit_ in ((), ("exit",))
    for mid in ((), ("player",), ("wall",), ("temp",))
    for beam in ((), ("hbeam",), ("vbeam",), ("hbeam", "vbeam"))
    if not (exit_ and mid == ("wall",))
]


def _comp_name(comp) -> str:
    return "+".join(comp) or "floor"


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the eight boards use (5, 6 and
    9 px; the size list is derived from the levels rather than assumed, so an
    edited level is covered). Whole 64x64 frames of uniform boards are compared
    rather than one cell out of a mixed board: `_render_frame` upscales and
    centre-pads, so slicing a cell by ``cell_px`` arithmetic reads the wrong
    pixels (the ps:explod lesson). Two uniform boards render identically iff
    their cells do.

    This is the pass that fails on the shipped .txt: HBeam and VBeam were the
    same four corner pixels, so ``one beam`` and ``both beams`` -- teleport and
    the cancel that is this game's whole puzzle -- were the same picture, 21
    clashes at every board shape. See the header comment in
    ``data/puzzlescript_games/Switcheroo.txt``.

    **Pass 2 -- the group.** The game is augmented with a frame rotation AND both
    flips, so a board is drawn at any of 16 presentations, and one sprite is not
    invariant under them: the Player is a little person with a top and a bottom.
    That is fine -- the whole frame is transformed together, so a turned board is
    a real board -- as long as no transform of one composition lands on a
    DIFFERENT one. That is the load-bearing check for the beams
    specifically: a quarter turn exchanges rows with columns, hence exchanges
    which beam runs which way, so HBeam's art and VBeam's art each have to be
    invariant under the whole group or a turned board would read as a board with
    the other beam on it. It runs on SQUARE boards, because a transform of a
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
    argument is in `PuzzleScriptAdapter._FLIP_GAMES`; Gobble Rush's chirality hid
    inside exactly that kind of argument, so it is measured.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (a player pressed flat against a block, X on a
    crossing where it does nothing, blocks teleported back off the exits they had
    already reached) and it presses every button, not only the ones a plan
    uses."""
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
            rng = random.Random(f"switcheroo:symmetry:{level}")
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
        violations = _enumerate(args or None)
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
    sys.exit(SwitcherooSolver.main())
