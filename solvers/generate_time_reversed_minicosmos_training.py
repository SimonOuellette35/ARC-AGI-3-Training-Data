"""Generate Phase-1 training data for the PuzzleScript game
ps:time_reversed_minicosmos ("Time-Reversed Minicosmos", Toph Wells with
apologies to Aymeric du Peloux -- forty Minicosmos sokobans with the push rule
replaced by a PULL, so the whole set is played backwards).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the exact distance FIELD that model is solved
with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_time_reversed_minicosmos",
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
Forty levels, three rules, and only one of them moves anything:

    [Left  PlayerR] -> [Left  PlayerL]
    [Right PlayerL] -> [Right PlayerR]
    [No Crate| >  Player | Crate ] -> [ No Crate | < Player | < Crate  ]

The first two are the player's FACING and are dealt with below. The third is the
game. Read it in the direction the rule fires: the middle cell holds a player
moving that way (``>``), the far cell holds a crate, the near cell holds no
crate -- and the right-hand side turns the player round (``<``) and gives the
crate the same reversed force. So:

* **You press TOWARDS a crate and both of you move AWAY from it.** Standing at
  ``p`` with a crate at ``p + d``, pressing ``d`` puts the player at ``p - d``
  and the crate at ``p``. To move a crate one square in direction ``e`` you
  therefore stand at ``X + e`` (the square it is going to end on), press
  ``-e``, and end up at ``X + 2e``.
* **The square BEHIND you is what can refuse a pull.** ``p - d`` must be on the
  board, wall-free and crate-free: the pattern demands ``No Crate`` there, and a
  wall (or the edge) cancels the player's move, which cancels the crate's too
  because it is moving onto the square the player is failing to leave. Nothing
  moves at all in that case.
* **Pressing into a crate you cannot pull does nothing.** Crates share the
  player's collision layer and no rule pushes anything, so the move is simply
  refused.
* **Walking is inert.** With no crate at ``p + d`` the press is an ordinary
  move onto a free square -- no rule fires, nothing else on the board changes.
  That is what separates this from `ps:five_pulloban_puzzles`, whose crates
  follow the player whether it meant it or not, and it is why "walk to a square,
  then pull" is a real macro here (not that this file needs one -- see below).

The win is ``All Target on Crate`` -- every target square carries a crate. Crate
and target counts match exactly on all forty levels (1 to 3 of each).

**THE PLAYER FACES THE WAY YOU PRESSED, NOT THE WAY IT MOVED.** The two sprite
rules run BEFORE the pull, so they read the pressed direction: press left and
the little person turns to face left, and then the pull rule may well send it
walking backwards to the right. The facing changes even when the press moves
nothing at all (a left press into a wall still turns the player). It is real,
visible state -- PlayerR and PlayerL are different 5x5 sprites -- and it is
deliberately NOT part of the model state below:

* nothing reads it. It appears on no rule's left-hand side except its own two,
  it is not in the win condition, and `Player = PlayerR or PlayerL` makes the
  pull rule blind to it. Keeping it in the state key would double every search
  for nothing;
* it is DERIVED, not lost: facing after a press is ``left -> L``, ``right -> R``,
  and unchanged on up/down (and on the dead ACTION key). ``--selfcheck``'s first
  pass checks that against the interpreter after every press of a random walk on
  every level, which is the difference between "dropped as cosmetic" and
  "dropped without looking" (the ps:im_sick_today germ-stain lesson).

Boards run 7x7 to 9x11, with 26 to 57 floor squares.

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`) plus an exact distance-to-win
field (`_Field`). The shared `PSSokobanExpert` does not apply: its macro is
"walk to a push square, then push", and its deadlock test is the sokoban corner
test -- both of which are about pushes. Reversed, they are simply wrong: a crate
flat in a corner is the easiest thing in this game to move, and a crate in the
MIDDLE of an open room with the player walled away from its far side is the
thing that cannot be moved at all.

**The whole game enumerates.** 254 to 35623 states per level, 160721 in all,
0.4 seconds for the entire set, so `_Field` does not search or estimate
anything -- it builds the exact field for every state a level can reach and
reads the answers off it:

  * a FORWARD BFS from the level start over `_Board.step`, treating a won board
    as terminal (the episode ends there), recording every edge's predecessor;
  * a BACKWARD BFS from the won states over those recorded predecessors, which
    labels every state with its true shortest presses-to-win.

Two sweeps rather than one, and that is the whole difference from
`ps:swap_sokoban`, whose single backward sweep is legal because every press
there is reversible. **A pull is not reversible** -- undoing one means pushing,
and this game has no push -- so `_Board.step` does not enumerate predecessors
and a backward sweep over it would answer a different game. ``--selfcheck``
measures exactly that rather than asserting it, exhaustively: of the 376108
moving transitions in the whole game, not one of the 26307 PULLS can be undone
by a single press and every one of the 349801 walks can.

The field is then read directly for everything else:

  * the PLAN is "walk down the gradient": at distance ``d``, any press whose
    successor is at ``d - 1``;
  * the OPTIMAL SETS are all of those presses, exactly, with no inference;
  * RECOVERY is the same lookup from wherever the board happens to be, which is
    the whole `supports_recovery` contract;
  * DEADNESS is exact and it is a real feature of this game, unlike in a
    reversible one: 0 to 4755 of a level's reachable states can never win
    (32482 of the 160721 in total -- 65% of level 15's space, where a crate
    pulled into the wrong cul-de-sac can never come back out).
    `record_level` probes every epsilon detour for winnability before taking
    it, so the recording can never brick a board; the probe is a dict lookup
    here and its answer is exact.

Shortest, and how that is known
-------------------------------
Every plan is provably shortest -- 12 to 187 presses, 3186 in all -- and the
proof is four independent derivations agreeing on all forty levels:

  * this file's forward + backward sweep;
  * a plain FORWARD BFS to the first win over the same model, which is shortest
    by construction and shares no line with the field (``--selfcheck`` pass 4);
    its per-step tie sets are re-derived a second time by brute force (pass 5);
  * the same two sweeps driven by the REAL INTERPRETER -- forward BFS from the
    level start by pressing actual buttons until a win appears, edges recorded,
    then backward over those edges (``--engine``). No `_Board` call of any kind
    participates: the successor of a board is whatever `PSEngine.step` makes of
    it and the goal test is `check_win`;
  * the shared `StateGraph.build` walking the interpreter's ENTIRE reachable
    space and solving the graph exactly (``--enumerate``), which also confirms
    the dead-state counts on the interpreter rather than on the model.

Recovery
--------
`supports_recovery` with the family's RESET-mode prefix, plus ``epsilon = 0.10``
detours inside the expert replay. The expert re-plans from the LIVE board (a
field lookup at whatever state the engine is in), so both kinds of perturbation
are answered exactly: the taken action is the mistake and ``optimal`` is the
recovery, which is the signal a policy needs after its own error.

Detours are safe here for a reason worth stating, because it is not the
`ps:swap_sokoban` one. This game HAS dead states, so a detour genuinely can
brick a board -- what stops it is that the field knows which states those are,
exactly, and `record_level`'s winnability probe consults it before committing to
an alternative. ``--selfcheck``'s recovery pass measures the other half: from
random boards several hundred presses off the plan, the field's answer (a plan,
or "dead") agrees with an independent forward BFS every time, and every plan it
does return WINS when replayed through the interpreter.

Two of the detours are peculiar to this game and are kept on purpose: a press
into a wall that only turns the player, and a press into a crate that cannot be
pulled. Both change the frame (the facing) without moving anything, so
`record_level` accepts them as state-changing -- and a policy that has to learn
"pressing at this crate accomplishes nothing" is better served by seeing it than
by never being shown it.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the frame rotation
(``rotation_k`` in 0..3) and the two flips, with their matching directional
action remap. 40 levels x 16 presentations = 640.

``Time-Reversed_Minicosmos`` is in `PuzzleScriptAdapter._FLIP_GAMES`; the
argument is recorded there -- one relative-force rule that no dihedral element
can break, and a PlayerR/PlayerL pair that is an exact mirror orbit switched by
a pair of rules that is its own mirror image. ``--symmetry`` measures it: every
level's plan AND a seeded random walk (which drags crates back off their
targets, jams the player against a crate it cannot pull, bumps into walls to
turn on the spot and presses the `noaction`-dead ACTION key) replayed at all 16
presentations, requiring every frame to be the exact transform of the
unaugmented one.

Rendering
---------
One sprite had to change, and it was the goal itself: Target shipped as a yellow
plus on rows/cols 1-3, which is five of the pixels the Player paints over, so
``player on target`` rendered as a player on bare floor at every cell size the
forty boards use. That is not cosmetic in a game where crates are PULLED: the
player has to stand on the far side of every crate it moves, so it crosses and
re-crosses the target squares constantly. The fix -- the plus plus the four
CORNERS, of which each facing leaves two showing -- is the one Swap Sokoban and
Stickyban got for the identical bug; the full statement is the header comment in
``data/puzzlescript_games/Time-Reversed_Minicosmos.txt``.

``--audit`` renders every cell COMPOSITION a board of this game can hold -- wall,
floor, target, each player facing, each facing on a target, crate, and crate on
a target -- as a whole 64x64 frame of a uniform board, at every cell size the
forty levels render at (5, 6, 7 and 8 px), and requires them pairwise distinct.
Whole frames rather than one cell sliced out of a mixed board: `_render_frame`
upscales and centre-pads, so slicing by ``cell_px`` arithmetic reads the wrong
pixels (the ps:explod lesson). ``wall + target`` is absent on purpose: the legend
has no character for it, no level places one, and a target under a wall would
make the level unwinnable rather than misread.

The step budget
---------------
The three longest levels need 172, 178 and 187 presses, so the adapter's stock
200-action budget would leave them no room for a single wrong turn.
``games/ps:time_reversed_minicosmos/ps:time_reversed_minicosmos.py`` raises it
to 600 the way `games/ps:five_pulloban_puzzles` does, and `game_module_id` below
makes this generator record against THAT adapter -- the one a live agent gets.

Usage (run from the repo root):
    python solvers/generate_time_reversed_minicosmos_training.py --episodes 200 \
        --out data/training_multi_level/puzzlescript_time_reversed_minicosmos

    python solvers/generate_time_reversed_minicosmos_training.py --plans
    python solvers/generate_time_reversed_minicosmos_training.py --selfcheck
    python solvers/generate_time_reversed_minicosmos_training.py --engine
    python solvers/generate_time_reversed_minicosmos_training.py --enumerate
    python solvers/generate_time_reversed_minicosmos_training.py --audit
    python solvers/generate_time_reversed_minicosmos_training.py --symmetry

``--engine`` and ``--enumerate`` drive the real interpreter over whole state
spaces (~960 presses/s), so they take minutes; both accept a list of level
numbers, and ``--enumerate all`` forces the full set instead of the small-level
default.
"""

from __future__ import annotations

import itertools
import random
import sys
import time
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

GAME_NAME = "Time-Reversed_Minicosmos"

#: The game folder's wrapper raises the per-level action budget from 200 to 600;
#: see the module docstring and `TimeReversedMinicosmosSolver.game_module_id`.
GAME_MODULE_ID = "ps:time_reversed_minicosmos"

#: Engine directions, in the order ties are broken. There is no ACTION rule in
#: this game -- the prelude says ``noaction`` and no rule reads it -- so the X
#: button is not a move and branching on it would widen every sweep for nothing.
#: The exploration prefix still presses it (a live agent has that button), which
#: is why ``--symmetry``'s random walk presses it too.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Disk cache of every level's start plan AND its optimal-action sets. The whole
#: game's fields take 0.4 seconds, so this is not the load-bearing cache it is on
#: the big ps: sokobans -- it is here so a `parallelize_generator` fan-out shares
#: one derivation rather than repeating it on every core. Delete the file to
#: re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "time_reversed_minicosmos_plans.json"


def _face_after(face: str, di: int) -> str:
    """The player's sprite after pressing ``DIRS[di]`` while drawn as ``face``.

    The two sprite rules run BEFORE the pull rule and read the PRESSED
    direction, so this is a function of the press alone and not of what the
    press turned out to do -- a left press into a wall turns the player without
    moving it, and a left press that pulls a crate turns the player left and then
    walks it right. Not part of the model state (nothing reads the facing); kept
    here because ``--selfcheck`` checks it against the interpreter, which is what
    makes leaving it out of the state a measurement rather than an assumption."""
    return "L" if di == 2 else "R" if di == 3 else face


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic.

    Cells are flat ``r * w + c`` indices and a STATE is ``(player, crate
    bitmask)`` -- the only two things any rule in this game moves that anything
    reads (the player's facing is the third and is deliberately absent; see
    `_face_after`). Walls and the target set never change, so they live here
    rather than in the state, and the crate set is a bitmask because the sweeps
    below key hundreds of thousands of states and a Python int is the cheapest
    exact spelling of a subset of squares.

    THE MECHANIC, as one press. The player at ``p`` presses ``d``, with
    ``q = p + d`` ahead of it and ``b = p - d`` behind it:

      * ``q`` off the board or a wall: nothing happens (no rule has a wall on its
        left, so the movement phase simply refuses the move);
      * ``q`` empty: the player walks onto it and nothing else changes;
      * ``q`` a crate, and ``b`` on the board, wall-free and crate-free: the
        PULL. The player ends at ``b`` and the crate at ``p``;
      * ``q`` a crate and ``b`` anything else: nothing happens. Either the rule
        cannot match (``b`` holds a crate) or it matches and the movement phase
        cancels the player's step into the wall, which cancels the crate's step
        onto the square the player did not leave.

    `won` is the win condition verbatim (``All Target on Crate``: every target
    square carries a crate), and it is the only place targets appear -- Target is
    alone on its own collision layer and blocks nothing, so it never affects a
    move.
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
        #: -1 when that square is off the board or a wall. Folding the wall test
        #: into the table is what keeps `step` down to two lookups and a bit
        #: test. ``di ^ 1`` is the opposite direction (see `DIRS`), which is how
        #: `step` finds the square BEHIND the player.
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
        press moves nothing (which is a non-move: no shortest path contains one,
        though the recording may still tape one as an epsilon detour -- it turns
        the player, so it does change the frame)."""
        p, mask = state
        q = self.edge[p][di]
        if q < 0:
            return state
        if not (mask >> q) & 1:
            return (q, mask)                     # a plain walk
        b = self.edge[p][di ^ 1]                 # the square behind
        if b < 0 or (mask >> b) & 1:
            return state                         # the pull is refused
        return (b, (mask ^ (1 << q)) | (1 << p))

    def won(self, state) -> bool:
        return (state[1] & self.tmask) == self.tmask


class _Field:
    """Exact presses-to-win for every state reachable from one ROOT board.

    TWO sweeps, and the reason there are two is the whole difference between this
    game and `ps:swap_sokoban`. There a single breadth-first sweep BACKWARDS from
    the win labels everything, because every press is reversible and `step`
    therefore enumerates predecessors as well as successors. **A pull is not
    reversible** -- its inverse is a push and this game has none -- so here:

      * a FORWARD BFS from ``root`` over `_Board.step`, recording for every state
        the states that step INTO it. A won board is terminal and is not expanded:
        the episode ends there, so nothing beyond it is reachable by an agent;
      * a BACKWARD BFS from the won states over those recorded predecessors. Every
        state it reaches gets its true shortest presses-to-win, and every state it
        does NOT reach is provably DEAD -- the forward sweep generated the whole
        space, so "no path back from a win" is a proof, not a search giving up.

    The distance is to the NEAREST win, which is the right answer even though a
    win ENDS the episode: a shortest path to the nearest goal cannot pass through
    another goal, because that one would be nearer.

    NOT LAZY, on purpose. The largest level here is 35623 states and the whole
    forty are 160721 in 0.4 seconds, so there is nothing to defer and a complete
    field is worth more: it makes `dead` exact, which is what `record_level`'s
    detour probe needs in a game that really can be bricked.

    ``cap`` is a runaway guard, not a budget: past it the field reports itself
    ``capped`` and answers None everywhere, which `_search` turns into "no plan"
    and `record_level` turns into a skipped detour or a RESET. It is never
    approached on the shipped levels."""

    __slots__ = ("board", "root", "depth", "dist", "wins", "capped")

    def __init__(self, board: _Board, root, cap: int):
        self.board = board
        self.root = root
        self.capped = False
        self.wins: set = set()
        rev: dict = {}
        depth = {root: 0}
        queue = deque([root])
        if board.won(root):
            self.wins.add(root)
        while queue:
            cur = queue.popleft()
            if cur in self.wins:
                continue                     # a won board ends the episode
            nd = depth[cur] + 1
            for di in range(4):
                nxt = board.step(cur, di)
                if nxt == cur:
                    continue                 # a press that moved nothing
                rev.setdefault(nxt, []).append(cur)
                if nxt in depth:
                    continue
                if len(depth) >= cap:
                    self.capped = True
                    self.depth, self.dist = {}, {}
                    return
                depth[nxt] = nd
                if board.won(nxt):
                    self.wins.add(nxt)
                queue.append(nxt)
        #: Every state reachable from ``root``, with its distance FROM it. Kept
        #: (rather than thrown away with ``rev``) because "is this state in this
        #: field at all" is what `PSExpert` asks before trusting an answer.
        self.depth = depth
        dist = {s: 0 for s in self.wins}
        back = deque(self.wins)
        while back:
            cur = back.popleft()
            nd = dist[cur] + 1
            for prev in rev.get(cur, ()):
                if prev not in dist:
                    dist[prev] = nd
                    back.append(prev)
        self.dist = dist

    # -- reading the field ---------------------------------------------------
    def covers(self, state) -> bool:
        """True when ``state`` is reachable from this field's root, i.e. when the
        field has an exact answer for it.

        A CAPPED field claims to cover everything on purpose: it knows nothing,
        but rebuilding it per query would only hit the same cap again, and
        `PSExpert` would grow one dead field per lookup."""
        return self.capped or state in self.depth

    def get(self, state) -> "int | None":
        """Presses to a win from ``state``, or None when this board can never be
        won from there. Exact both ways -- a None here is a PROOF of deadness,
        not a search that gave up, because the forward sweep generated the whole
        reachable space (see the class docstring)."""
        return self.dist.get(state)

    def optimal(self, state) -> list:
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``. Ties come out in ``DIRS`` order, which is what makes
        a re-derived plan byte-identical across processes.

        Exact, with nothing inferred: a press is optimal iff its successor is one
        step nearer the win, and both distances are the field's."""
        rest = self.dist.get(state)
        if not rest:                      # None (dead) or 0 (already won)
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
        rest = self.dist.get(state)
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

class TimeReversedMinicosmosExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Field`.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes, the way `PSEnumExpert` replaces it. Here the
    strategy is "read the live board and walk down the exact distance field", so
    `heuristic` is never called and asserts rather than returning a number
    nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered exactly --
    including the answer "this board is dead", which in this game is a real
    answer and not a failure.
    """

    directions = list(DIRS)
    #: `_key` is the dynamic objects only, which is canonical WITHIN a level but
    #: not across them -- walls and targets are static per level and differ
    #: between levels.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Runaway guard on one level's forward sweep; see `_Field`. The largest this
    #: game reaches is level 23's 35623 states, so this is ~100x headroom.
    field_cap: int = 4_000_000

    def setup(self) -> None:
        g = self.g
        #: BOTH facings. `Player = PlayerR or PlayerL` in the .txt, and the
        #: engine's ``_player_indices`` carries the pair, so reading either one
        #: alone would lose the player the moment it turned.
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.target_ids = set(g.resolve_object_name("target"))
        #: `_Board`s by STATIC signature (see `read`), not by level index, so a
        #: board is shared by every state of its level and the edge table is
        #: built once.
        self._boards: dict = {}
        #: ``sig -> [field, ...]``. In practice exactly one field per level, rooted
        #: at the level start, because every board a recording can reach is
        #: reachable FROM that start (the exploration prefix ends in a RESET back
        #: to it, and the detours go forward from there). A second field is built
        #: only if something ever asks about a state outside it, which
        #: ``--selfcheck``'s recovery pass reports as 0.
        self._fields: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "TimeReversedMinicosmosExpert reads an exact distance field; "
            "heuristic is unused")

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid.

        Boards are cached by their STATIC signature (dimensions, walls, targets),
        so the edge table is built once per level however many states are read
        from it. The player's FACING is not read: it is on no rule's left-hand
        side but its own two and changes nothing (see `_face_after`)."""
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

    def field(self, board: _Board, state) -> _Field:
        """The field covering ``state``, building one rooted there if no existing
        field reaches it. Fields are kept for the process's lifetime; a level
        pays for its sweep once."""
        fields = self._fields.setdefault(board.sig, [])
        for f in fields:
            if f.covers(state):
                return f
        f = _Field(board, state, self.field_cap)
        fields.append(f)
        return f

    def _key(self, eng) -> frozenset:
        """``(row, col, tag)`` triples for the player (0) and the crates (1).

        Built from the MODEL's reading rather than from raw object ids so the key
        is exactly the two things a state consists of -- which drops the facing,
        and must: PlayerR and PlayerL are different object ids, so an id-based key
        would split every state in two and memoize the same plan twice. It is
        also what the ``plan_cache_path`` signature is stored as, so a cached plan
        is matched against the board it was solved from and an edited level is a
        miss rather than a wrong plan."""
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
        return self.field(board, state).plan(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class TimeReversedMinicosmosSolver(PSAStarSolver):
    game_id = "puzzlescript_time_reversed_minicosmos"
    game_name = GAME_NAME
    expert_cls = TimeReversedMinicosmosExpert

    #: NOT a passthrough: the wrapper raises the per-level action budget from 200
    #: to 600, without which the three longest levels have 13 presses of slack
    #: and no episode that wanders can finish them. Recording against the adapter
    #: the wrapper builds is also the rule for this family -- it is the one a live
    #: agent is handed.
    game_module_id = GAME_MODULE_ID

    #: Unused: `TimeReversedMinicosmosExpert._search` never calls `_astar`. Left
    #: at the base values so nothing reads a lie off them;
    #: `TimeReversedMinicosmosExpert.field_cap` is the knob that actually bounds
    #: the sweeps.
    weight = 1
    node_cap = 400_000

    #: Room for the longest plan (187 presses on level 27) plus the re-plans the
    #: epsilon detours cost, which in THIS game is not the usual "one press
    #: wasted, one press back": a pull is irreversible, so a detour that drags a
    #: crate the wrong way is undone by walking round it and dragging it back --
    #: tens of presses, not two. Measured over two full episodes the overhead
    #: ran from 0 to +139 presses on a 98-press level, and the longest level
    #: recording was 263. This is ~2.4x the longest plan, and the adapter's own
    #: 600-press per-level budget (see `game_module_id`) is separate and larger,
    #: and is reset by the `set_level` that ends the exploration prefix, so the
    #: plan starts it from zero.
    max_steps = 450

    #: Every level is solved; nothing is skipped.
    skip_levels = frozenset()

    #: On top of the RESET prefix: one press in ten is a random legal
    #: alternative, after which the expert re-plans from wherever it landed --
    #: the taken action is the mistake and ``optimal`` is the recovery. Safe
    #: despite this game HAVING dead states, because `record_level` probes each
    #: alternative for winnability first and the field's answer to that probe is
    #: exact (see the module docstring).
    epsilon = 0.10

    def prepare_expert(self, game, expert) -> None:
        """Sweep every level's field before `discover_solvable` asks for it --
        the same work either way, but it does it in one pass at startup rather
        than in the middle of a recording, and it fills the disk cache. 0.4
        seconds for all forty.

        The field is built EXPLICITLY, not as a side effect of `plan`: with a
        warm ``PLAN_CACHE`` the start plan comes off disk without touching a
        field at all, and then the first epsilon detour of the run would pay for
        the sweep instead."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.field(*expert.read(game._engine))
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    """The solver, its adapter and an expert -- without planning anything.

    The reports below each want a different slice (``--audit`` needs no plan at
    all, ``--selfcheck`` needs only the model), so planning is left to whoever
    asks for it rather than paid on every entry point."""
    solver = TimeReversedMinicosmosSolver()
    game = solver.make_game(seed)
    return solver, game, TimeReversedMinicosmosExpert(game,
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
                line += "@" if on_target else "*"
            else:
                line += "O" if on_target else "."
        out.append(line)
    return "\n".join(out)


def _start(game, expert, level: int):
    """``(board, state)`` at a level's start, engine left seated there."""
    game.set_level(level)
    return expert.read(game._engine)


def _plans(verbose: bool = True) -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once. Also the level report: board shape, crates, the size of the reachable
    space and how much of it is DEAD."""
    _solver, game, expert = _new()
    eng = game._engine
    total = ties = bad = states = deads = 0
    build = 0.0
    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        t0 = time.time()
        field = expert.field(board, state)
        build += time.time() - t0
        plan = expert.plan(eng, level)
        states += len(field.depth)
        deads += len(field.depth) - len(field.dist)
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
        if verbose:
            print(f"  L{level:2d}: {board.h:2d}x{board.w:2d}  "
                  f"{len(board.free):2d} floor  "
                  f"{bin(state[1]).count('1')} crates  "
                  f"{len(field.depth):6d} states "
                  f"({len(field.depth) - len(field.dist):5d} dead)  "
                  f"{len(plan):3d} presses  win={won}  "
                  f"(budget {game._max_steps}, {room})  "
                  f"{tie_steps:3d} steps with a tie set")
    print(f"  {states} states swept in {build:.2f}s, {deads} of them dead")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _forward(board: _Board, state, limit: int) -> "int | None":
    """Presses to a win from ``state``, by a plain FORWARD BFS, or None past
    ``limit``.

    Shares nothing with `_Field` but `_Board.step` itself, which is the point: it
    is the independent answer every optimality claim below is checked against."""
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


#: The levels whose ENTIRE state space is a few hundred boards, so a brute-force
#: forward BFS per candidate press per plan step is affordable on them
#: (254, 298, 457, 465, 721 and 964 states; 413 plan steps between them). Used by
#: ``--selfcheck`` pass 5, which is the tie-set check that shares no line at all
#: with the field.
_SMALL = (0, 3, 8, 30, 31, 34)


def _selfcheck(walk_presses: int = 250, samples: int = 40,
               verbose: bool = True) -> int:
    """Six things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's, the FACING included.** A seeded
       random walk on every level, comparing `_Board.step`'s board against the
       engine's grid after EVERY press -- including the presses that do nothing,
       which is where a collision-layer mistake hides -- and `_face_after`'s
       prediction against which of the two player sprites is actually on the
       board. The walk is what measures the mechanic listed in the module
       docstring: it pulls crates, presses into crates it cannot pull, presses
       into walls, and drags crates off their targets. ``won`` is checked against
       `check_win` on the same schedule, so the goal test the whole field is
       built on is measured on all forty levels rather than read off the .txt.

    2. **A pull is NOT reversible -- exactly, and not "mostly".** This is the
       load-bearing measurement of the whole file, in the negative:
       `ps:swap_sokoban` can sweep BACKWARDS from the win over its own `step`
       because every press there is reversible, and this game cannot, which is
       why `_Field` runs a forward sweep with a predecessor map first. Measured
       twice:

       * on the INTERPRETER, along the walk of pass 1: every press that moves
         something is tried against all four presses back. A random walk pulls
         rarely (it has to be standing beside a crate, pressing into it, with the
         square behind it clear), so the pull count is reported beside the
         irreversible count -- otherwise a level whose walk happened never to
         pull would look like a level where pulls are reversible;
       * over the WHOLE reachable space of every level, on the model, which turns
         the sample into a statement: of the 376108 moving transitions in the
         game, every one of the 26307 PULLS is irreversible and every one of the
         349801 walks is reversible, with no exceptions either way. The asymmetry is structural
         -- a walk is undone by stepping back onto the square you just left,
         while undoing a pull means PUSHING the crate, and this game has no push
         -- and this is the check that it is really true of all forty boards.

    3. **The plans are shortest.** A plain forward BFS to the first win, which is
       shortest by construction and shares no code with the field, must return
       the field's plan length on every level.

    4. **The whole field satisfies the BELLMAN equation, and the tie sets are its
       argmin.** For every one of the 160721 states, on every level:
       ``dist == 0`` exactly on the won boards; every other labelled state's
       successors have a minimum of ``dist - 1``; and every state the field calls
       DEAD has none but dead successors. That is a complete proof rather than a
       spot check -- a labelling with those three properties over a finite graph
       of unit-cost edges IS the shortest-distance function (follow the argmin
       from any state and it descends to a win in exactly ``dist`` presses;
       conversely ``dist(s) <= 1 + dist(s')`` along any path makes ``dist`` no
       larger than the true distance) -- and unlike a re-solve it owes nothing to
       the order the sweeps visited anything in, which is where a predecessor map
       would go wrong if it went wrong. The plan's optimal sets are then
       re-derived from the certified ``dist`` alone and compared to what
       `_Field.optimal` produced.

    5. **The tie sets by brute force**, on the six levels whose whole space fits
       in a few hundred states: every step's optimal set re-derived by a fresh
       depth-bounded forward BFS from each of the four successors, sharing
       nothing with the field at all. Pass 4 covers all forty exhaustively but
       through the field's own ``dist``; this covers 413 steps through nothing
       but `_Board.step`.

    6. **Recovery is answered from ANY board, not just the start**, and DEADNESS
       is an answer. This is the one that matters at generation time: the
       exploration prefix and the epsilon detours both leave the board off every
       shortest path, and unlike a reversible game this one really can be
       bricked. From ``samples`` states reached by seeded random presses, the
       field must agree with an independent forward BFS -- both on the length
       when there is a plan and on there being NO plan when there is not -- and
       every plan it returns must WIN when replayed through the interpreter. The
       count of boards the field had to build a NEW field for is reported too: it
       is 0, which is the claim that one field per level covers every board a
       recording can reach.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0

    for level in range(game.n_levels):                    # 1 + 2
        board, state = _start(game, expert, level)
        face = _read_face(eng, expert)
        rng = random.Random(f"time_reversed_minicosmos:selfcheck:{level}")
        drift = faces = irreversible = moved_total = pulls = 0
        for _ in range(walk_presses):
            here = snapshot(eng)
            for di in range(4):
                eng.step(DIRS[di])
                after_key = _read_state(eng, expert)
                if after_key != state:
                    moved_total += 1
                    pulls += after_key[1] != state[1]
                    after = snapshot(eng)
                    back = False
                    for dj in range(4):
                        restore(eng, after)
                        eng.step(DIRS[dj])
                        if _read_state(eng, expert) == state:
                            back = True
                            break
                    irreversible += not back
                restore(eng, here)
            di = rng.randrange(4)
            eng.step(DIRS[di])
            state = board.step(state, di)
            face = _face_after(face, di)
            live = _read_state(eng, expert)
            if live != state:
                drift += 1
                print(f"    L{level} model DIVERGES after {DIRS[di]}:\n"
                      f"{_ascii(board, state)}")
                break
            if _read_face(eng, expert) != face:
                faces += 1
                print(f"    L{level} FACING diverges after {DIRS[di]}")
                break
            if eng.check_win() != board.won(state):
                drift += 1
                print(f"    L{level} WIN CONDITION diverges")
                break
        bad += drift + faces
        if verbose:
            print(f"  L{level:2d}: model {'MATCHES' if not drift else 'DIVERGES from'}"
                  f" the interpreter over {walk_presses} random presses, facing "
                  f"{'ok' if not faces else 'WRONG'}; {irreversible} of the "
                  f"{moved_total} moving presses probed cannot be undone by any "
                  f"single press, and {pulls} of them were pulls")

    for level in range(game.n_levels):                    # 2, exhaustively
        board, state = _start(game, expert, level)
        field = expert.field(board, state)
        moves = pulls = pull_rev = walk_irrev = 0
        for s in field.depth:
            for di in range(4):
                t = board.step(s, di)
                if t == s:
                    continue
                moves += 1
                back = any(board.step(t, dj) == s for dj in range(4))
                if t[1] != s[1]:
                    pulls += 1
                    pull_rev += back
                else:
                    walk_irrev += not back
        bad += pull_rev + walk_irrev
        if verbose:
            print(f"  L{level:2d}: {moves:7d} moving transitions -- {pulls:6d} "
                  f"pulls, of which {pull_rev} undoable in one press; "
                  f"{moves - pulls:6d} walks, of which {walk_irrev} not")

    for level in range(game.n_levels):                    # 3
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        shortest = _forward(board, state, len(plan) + 1 if plan else 250)
        ok = plan is not None and shortest == len(plan)
        bad += not ok
        if verbose:
            print(f"  L{level:2d}: field {len(plan) if plan else None} presses vs "
                  f"forward BFS {shortest} -- "
                  f"{'SHORTEST' if ok else 'NOT SHORTEST'}")

    for level in range(game.n_levels):                    # 4
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        field = expert.field(board, state)
        dist, depth = field.dist, field.depth
        wrong_win = wrong_step = wrong_dead = 0
        for s in depth:
            d = dist.get(s)
            if board.won(s):
                wrong_win += d != 0
                continue
            succ = [board.step(s, di) for di in range(4)]
            succ = [t for t in succ if t != s]
            if d is None:                    # the field calls this state dead
                wrong_dead += any(t in dist for t in succ)
                continue
            wrong_win += d == 0              # only a won board may be 0 presses
            best = min((dist[t] for t in succ if t in dist), default=None)
            wrong_step += best != d - 1
        # The plan's tie sets, re-derived from the certified distances alone.
        cur, mismatched = state, 0
        for i, direction in enumerate(plan):
            truth = sorted(DIRS[di] for di in range(4)
                           if board.step(cur, di) != cur
                           and dist.get(board.step(cur, di)) == dist[cur] - 1)
            if truth != sorted(plan.optsets[i]):
                mismatched += 1
            cur = board.step(cur, DIRS.index(direction))
        bad += wrong_win + wrong_step + wrong_dead + mismatched
        if verbose:
            ok = not (wrong_win or wrong_step or wrong_dead)
            print(f"  L{level:2d}: {len(depth):6d} states -- Bellman "
                  f"{'CERTIFIED' if ok else 'VIOLATED'} "
                  f"({wrong_win} win labels, {wrong_step} steps, "
                  f"{wrong_dead} dead states wrong); {len(plan)} tie sets "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'}"
                  f" the argmin")

    for level in _SMALL:                                  # 5
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
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
            if sorted(truth) != sorted(plan.optsets[i]):
                mismatched += 1
            cur = board.step(cur, DIRS.index(direction))
        bad += mismatched
        if verbose:
            print(f"  L{level:2d}: {len(plan)} tie sets "
                  f"{'match' if not mismatched else f'{mismatched} MISMATCH'}"
                  f" brute force")

    for level in range(game.n_levels):                    # 6
        rng = random.Random(f"time_reversed_minicosmos:recovery:{level}")
        board, _s = _start(game, expert, level)
        n_fields = len(expert._fields.get(board.sig, ()))
        wrong = dead = won_after = fresh = 0
        for _ in range(samples):
            game.set_level(level)
            for _ in range(rng.randrange(1, 60)):
                eng.step(DIRS[rng.randrange(4)])
                if eng.check_win():
                    break
            if eng.check_win():
                continue                 # the walk already won; nothing to plan
            board, state = expert.read(eng)
            plan = expert._search(eng)
            fresh += len(expert._fields.get(board.sig, ())) > n_fields
            n_fields = len(expert._fields.get(board.sig, ()))
            truth = _forward(board, state, board.n * 8)
            if plan is None:
                dead += 1
                if truth is not None:    # the field called a winnable board dead
                    wrong += 1
                    print(f"    L{level} WRONGLY DEAD (forward BFS says "
                          f"{truth}):\n{_ascii(board, state)}")
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
                  f"re-planned to a WIN in the interpreter, {dead} provably "
                  f"dead, {fresh} needing a new field, {wrong} WRONG")
    return bad

# ---------------------------------------------------------------------------
# The interpreter proofs
# ---------------------------------------------------------------------------

def _read_face(eng, expert) -> str:
    """Which of the two player sprites is on the board -- ``"R"``, ``"L"``, or
    ``"?"`` when there is no player at all. Used only by the reports; the model
    does not carry it (see `_face_after`)."""
    right = expert.g.obj_name_to_idx["playerr"]
    for row in eng.grid:
        for cell in row:
            if cell & expert.player_ids:
                return "R" if right in cell else "L"
    return "?"


def _statics(eng, expert):
    """The level's static layer -- every cell with the players and the crates
    stripped out -- so a state key can be SEATED back onto the engine instead of
    keeping a snapshot of every board it reaches.

    Taken from the live grid rather than assembled from object ids, so it carries
    whatever the level actually holds (background under walls included) and a
    seated board is byte-identical to the one the level loaded."""
    dynamic = expert.player_ids | expert.crate_ids
    return [[cell - dynamic for cell in row] for row in eng.grid]


def _seat(eng, static, expert, state) -> None:
    """Put ``state`` back on the engine. The inverse of `_read_state`; that it
    IS the inverse is checked before either is used (see `_engine`).

    The player is always seated FACING RIGHT, the way every level starts it. That
    is a choice and it is sound: the facing is on no rule's left-hand side but
    its own two, so a board seated with either sprite steps to the same state --
    which `_engine` checks directly by comparing a seated step against a
    snapshot-restored one, from boards the walk has turned the player on."""
    p, mask = state
    w = len(static[0])
    player = expert.g.obj_name_to_idx["playerr"]
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


def _engine(levels=None, verbose: bool = True) -> int:
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
    keeps 35k boards in megabytes instead of gigabytes; the seating is checked to
    be the exact inverse of the reading, and a seated board is checked to step
    identically to a snapshot-restored one -- from a board whose player has been
    TURNED, which is the one thing seating deliberately does not preserve.

    ~960 interpreter presses a second and the balls are most of each level's
    space, so the full forty take about ten minutes. Pass level numbers to do
    fewer."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in (range(game.n_levels) if levels is None else levels):
        _board, _state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        game.set_level(level)
        static = _statics(eng, expert)
        start = _read_state(eng, expert)
        t0 = time.time()

        # The decoder is the inverse of the encoder, and a seated board behaves
        # like a restored one -- both checked here rather than assumed, because a
        # decoder that is not exactly the inverse enumerates a DIFFERENT game.
        original = snapshot(eng)
        _seat(eng, static, expert, start)
        if eng.grid != original or _read_state(eng, expert) != start:
            print(f"  L{level:2d}: SEATING IS NOT THE INVERSE OF READING")
            bad += 1
            continue
        eng.step("left")                    # turn the player, then re-seat it
        turned = snapshot(eng)
        turned_key = _read_state(eng, expert)
        for di in range(4):
            restore(eng, turned)
            eng.step(DIRS[di])
            expected = _read_state(eng, expert)
            won = eng.check_win()
            _seat(eng, static, expert, turned_key)
            eng.step(DIRS[di])
            if _read_state(eng, expert) != expected or eng.check_win() != won:
                print(f"  L{level:2d}: a seated board steps differently "
                      f"({DIRS[di]}) -- the facing is NOT inert")
                bad += 1
        _seat(eng, static, expert, start)

        depth = {start: 0}
        queue = deque([start])
        rev: dict = {}
        won_states: set = set()
        d_star = None
        presses = 0
        while queue:
            cur = queue.popleft()
            d = depth[cur]
            if d_star is not None and d >= d_star:
                break
            for di in range(4):
                _seat(eng, static, expert, cur)
                eng.step(DIRS[di])
                presses += 1
                nxt = _read_state(eng, expert)
                if nxt == cur:
                    continue                    # a press that did not move
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
                presses += 1
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
                  f"{presses:7d} presses in {time.time() - t0:5.1f}s, "
                  f"{d_star:3d} presses vs the field's {len(plan):3d} -- "
                  f"{'AGREE' if same_len else 'DISAGREE'}; tie sets "
                  f"{'AGREE' if same_sets else 'DISAGREE'}")
        game.set_level(level)
    return bad


#: Levels whose ENTIRE reachable space the interpreter can be walked over in a
#: few seconds each -- 254 to 2707 states, ~2 minutes for the twenty-two
#: together. The other eighteen run to 35623 states, i.e. minutes apiece; pass
#: level numbers, or ``all``, to enumerate them anyway.
_ENUMERABLE = (0, 3, 6, 8, 9, 14, 16, 18, 20, 21, 24, 26, 28, 30, 31,
               32, 33, 34, 35, 36, 37, 39)


def _enumerate(levels=_ENUMERABLE, verbose: bool = True) -> int:
    """Walk the interpreter's WHOLE reachable space with the shared
    `StateGraph.build` and solve the graph exactly.

    A fourth derivation, and the one that shares no algorithm with anything here
    either -- `StateGraph` is `solvers/common/ps_astar.py`'s, written for other
    games. It buys two things `--engine` cannot: plans that are shortest against
    the entire space rather than against a ball, and the DEAD-state count
    measured on the interpreter, which in a pull game is the number that says
    whether the detour probe is doing anything. Both are compared against the
    field's."""
    from solvers.common.ps_astar import StateGraph                     # noqa: PLC0415

    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in levels:
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        field = expert.field(board, state)
        game.set_level(level)
        static = _statics(eng, expert)
        t0 = time.time()
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
        # The graph drops won boards (they are the WIN sentinel, never keys), so
        # compare its live states against the field's live states the same way.
        mine = len(field.depth) - len(field.wins)
        mine_dead = len(field.depth) - len(field.dist)
        same_dead = len(stranded) == mine_dead and len(graph.succ) == mine
        bad += (not same_len) + (not same_sets) + (not same_dead)
        if verbose:
            print(f"  L{level:2d}: {len(graph.succ):6d} engine states vs the "
                  f"model's {mine:6d} ({graph.steps} presses, "
                  f"{time.time() - t0:5.1f}s), "
                  f"{len(eplan) if eplan else None} presses vs the field's "
                  f"{len(plan)} -- {'AGREE' if same_len else 'DISAGREE'}; tie "
                  f"sets {'AGREE' if same_sets else 'DISAGREE'}; "
                  f"{len(stranded)} state(s) that cannot win vs the model's "
                  f"{mine_dead} -- {'AGREE' if same_dead else 'DISAGREE'}")
    return bad


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

#: Every cell stack a board of this game can hold. Background is on its own layer
#: and is everywhere; Target has a layer to itself; PlayerR, PlayerL, Wall and
#: Crate share the third, so at most one of them is in a cell.
#:
#: ``wall + target`` is absent on purpose and is not an omission: the legend has
#: no character that places one, no level does, and a target buried under a wall
#: would make its level unwinnable rather than misread. It is also the one pair
#: this game cannot draw apart -- Wall is opaque.
_COMPOSITIONS = [("wall",), (), ("target",),
                 ("playerr",), ("playerr", "target"),
                 ("playerl",), ("playerl", "target"),
                 ("crate",), ("crate", "target")]


def _comp_name(comp) -> str:
    return "+".join(comp) or "floor"


def _audit(verbose: bool = True) -> int:
    """Two passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the forty boards use (5, 6, 7
    and 8 px; the size list is derived from the levels rather than assumed, so an
    edited level is covered). Whole 64x64 frames of uniform boards are compared
    rather than one cell out of a mixed board: `_render_frame` upscales and
    centre-pads, so slicing a cell by ``cell_px`` arithmetic reads the wrong
    pixels (the ps:explod lesson). Two uniform boards render identically iff
    their cells do.

    This is the pass that fails on the shipped .txt: ``playerr`` and
    ``playerr+target`` were the same frame at every size, and likewise for
    ``playerl``. See the header comment in
    ``data/puzzlescript_games/Time-Reversed_Minicosmos.txt``.

    **Pass 2 -- the group.** The game is augmented with a frame rotation AND both
    flips, so a board is drawn at any of 16 presentations, and the sprites are
    not all invariant under them -- the player is a little person that FACES one
    way. That is exactly what has to be checked rather than waved at: no
    transform of one composition may land on a DIFFERENT one, all 16 transforms
    of each against all others.

    The player pair is the interesting case and it PASSES for a reason worth
    keeping: PlayerL is PlayerR's exact fliplr, so a horizontal mirror maps the
    pair onto itself -- and it maps the two rules that pick between them onto
    each other too, so the figure faces the way the player last pressed at every
    presentation. That is reported as an orbit rather than as a clash. It runs on
    SQUARE boards, because a transform of a non-square board also moves the
    letterbox and every comparison would pass for the wrong reason.
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
        # The two player facings ARE each other's mirror -- that is the property
        # the flips rest on, not a collision -- and so are the two "on a target".
        orbit = {(("playerr",), ("playerl",)), (("playerl",), ("playerr",)),
                 (("playerr", "target"), ("playerl", "target")),
                 (("playerl", "target"), ("playerr", "target"))}
        real = [(a, b, t) for a, b, t in clashes if (a, b) not in orbit]
        bad += len(real)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(real)} composition(s) whose "
                  f"transform is another's art, "
                  f"{len(clashes) - len(real)} that are the player MIRROR PAIR; "
                  f"group-invariant: {', '.join(invariant)}")
        for a, b, t in real:
            print(f"      {_comp_name(a)} under {t} == {_comp_name(b)}")
    return bad


def _symmetry(walk_presses: int = 100, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for the rotation + flip augmentation. The structural
    argument is in `PuzzleScriptAdapter._FLIP_GAMES` -- one rule written with the
    relative forces, a pull that can never contest a square, and a player pair
    that is an exact mirror orbit switched by a pair of rules that is its own
    mirror -- but Gobble Rush's chirality hid inside exactly that kind of
    argument, so it is measured.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (crates dragged back off their targets, crates
    flat against walls, the player jammed against a crate it cannot pull) and it
    presses the ``noaction``-dead ACTION key as well. It is also the pass that
    would catch a facing bug: turning on the spot against a wall is a frame
    change with no board change, and it happens constantly in a random walk.

    The (seed, level) -> presentation map is built first with `set_level` alone
    (~1900/s) so that each level is DRIVEN once per presentation rather than once
    per seed -- forty levels times sixteen presentations times two replays is
    already ~110k adapter presses."""
    solver, game, expert = _new()
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    # Which (seed, level) draws which presentation. Cheap: set_level only.
    wanted: dict = {}
    seeds_scanned = 0
    for seed in range(200):
        g = solver.make_game(seed) if seed else game
        g._seed = seed
        seeds_scanned += 1
        for level in range(g.n_levels):
            g.set_level(level)
            wanted.setdefault(
                (level, (g._rotation_k, g._hflip, g._vflip)), seed)
        if len(wanted) == 16 * game.n_levels:
            break

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

    ref, bad, seen = {}, 0, set()
    # The unaugmented presentation of every level first: it is the reference the
    # other fifteen are compared against.
    order = sorted(wanted, key=lambda kv: (kv[0], kv[1] != (0, False, False)))
    for level, k in order:
        seed = wanted[(level, k)]
        g = solver.make_game(seed)
        rng = random.Random(f"time_reversed_minicosmos:symmetry:{level}")
        walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                for _ in range(walk_presses)]
        seen.add(k)
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
                    print(f"    L{level}: no seed under {seeds_scanned} draws "
                          f"the unaugmented presentation")
                    bad += 1
                    continue
                ref[key] = frames
                continue
            if any(not np.array_equal(transform(a, *k), b)
                   for a, b in zip(ref[key], frames)):
                print(f"    seed {seed} L{level} {k}: {tag} frames are not "
                      f"the transform of the unaugmented ones")
                bad += 1
    if verbose:
        print(f"  {len(wanted)} (level, presentation) pairs found in "
              f"{seeds_scanned} seeds; {len(seen)} presentations drawn: "
              f"{sorted(seen)}")
    return bad


def _levels_arg(flag: str) -> "list[int] | None":
    """Level numbers following ``flag`` on the command line, or None for the
    report's own default. ``all`` means every level."""
    rest = sys.argv[sys.argv.index(flag) + 1:]
    if any(a == "all" for a in rest):
        return list(range(40))
    nums = [int(a) for a in rest if a.isdigit()]
    return nums or None


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--engine" in sys.argv:
        violations = _engine(_levels_arg("--engine"))
        print(f"engine sweeps: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--enumerate" in sys.argv:
        violations = _enumerate(_levels_arg("--enumerate") or _ENUMERABLE)
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
    sys.exit(TimeReversedMinicosmosSolver.main())
