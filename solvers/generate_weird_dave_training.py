"""Generate Phase-1 training data for the PuzzleScript game ps:weird_dave
("Weird Dave", HugoBDesigner).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exact distance FIELD that model is
searched with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_weird_dave",
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
A SOKOBAN WHOSE PUSHER IS A RIGID POLYOMINO. "Dave" is not one square: he is a
plus made of five different objects (`PlayerT`/`PlayerL`/`PlayerC`/`PlayerR`/
`PlayerB`), or a 24-cell snake, or five sparkles scattered across the whole
board -- and every one of those cells is its own `Player`, so one arrow key
gives all of them the same force.

The win condition is the ordinary one, ``ALL Target ON Crate``: shove a crate
onto the goal square. What makes it a puzzle is that the thing doing the
shoving is a fixed SHAPE that cannot turn, cannot bend and cannot be split:

* the whole body translates one square, or NOTHING happens. The .txt spells
  that out in three rules -- ``[MOVING Player][STATIONARY Player] ->
  [MOVING Player][MOVING Player]`` puts every cell on the same force,
  ``[Player CantMove][Player] -> [Player CantMove][Player CantMoveUp]``
  propagates one cell's obstruction to all of them, and
  ``[MOVING Player CantMove] -> cancel`` throws the whole turn away. So a press
  is all-or-nothing and there is no shear (contrast [[ps:the_blob]], where the
  jammed part stays behind and the body comes out a different shape);
* every crate that any body cell walks into is pushed, and a crate pushed into
  a crate pushes that one too. The chain is collinear;
* if any body cell's destination is a Wall, or any pushed crate's is, the press
  cancels -- the body does not "give" and the crates do not slide out from
  under it.

Two consequences shape the whole file. The body's shape is INVARIANT, so the
only thing a press can do to it is move it: a state is one ANCHOR CELL plus the
crate set, not a set of body cells. And a press that pushes a crate is NOT
undoable -- pressing back moves the body but leaves the crate where it was
shoved -- so the space is a digraph, boards really can be dead, and the search
has to be able to say so.

The pre-pass that decides whether a crate can move is written against
``Collidable``, which is ``Wall`` alone -- the board EDGE is not Collidable and
neither is ``Void`` (the `x` legend, which paints the ragged outside of these
levels and sits on the BackTexture collision layer, not the player's). If a
crate or a body cell could ever reach the edge the interpreter and this model
would part company, because the model treats the edge and the void as walls
while the interpreter would simply fail the move for that one object. It cannot
happen: every shipped level is a closed room of `Wall`, and `_Board.__init__`
RAISES if a level is ever edited so that it is not -- see `_check_enclosed`.

``noaction`` is declared in the header and no rule has ``action`` on its
left-hand side, so ACTION5 is not a move; the search does not branch on it.
``--selfcheck`` presses it anyway and requires it to be a measured no-op, and
the exploration prefix presses it because a live agent has that button.

The three levels, all winnable, none already won at reset:

  * **L0** 13x12, a five-cell PLUS, FOUR crates in a 2x2 block, ONE target at
    the bottom of a 3x4 side room reached through a single one-square doorway.
    The plus is wider than the doorway is tall, so most of the answer is
    getting a crate to the door and the body to the right side of it. 27
    presses, and the only winning board in its entire 4,135,322-state closure.
  * **L1** 15x15, a 24-cell SNAKE, one crate, one target two squares to its
    right. Nine presses -- the body is so large that "which cell of me is
    beside the crate" is the whole question, and the first two presses shove
    the crate a square the wrong way to get a different cell alongside it.
  * **L2** 15x15, FIVE sparkles spread over the board, one crate, the target
    beside it. Three presses; it is the tutorial for "you are all of them".

Expert solver
-------------
A NATIVE model of the mechanic above (`_Board`), searched to an exact distance
field rather than heuristically. A state is ``(anchor, crates)`` packed into ONE
PYTHON INT -- the low bits index the cell the body's first (row-major) square
sits on, the high bits are a crate bitmask over ``r * w + c``. Everything else
on the board is static. The mechanic is then a handful of bit operations:

    nb      = body_mask[anchor + d]                 # None => the press cancels
    pushed  = nb & crates                           # crates the body walks into
    repeat:   pushed |= shift(pushed, d) & crates   # ...and the chain behind them
    crates  = (crates & ~pushed) | shift(pushed, d) # unless one has nowhere to go

`_Board.field` is a forward BFS kept in LAYERS plus a backward induction over
those layers, exactly as ps:the_blob does it and for the same reason: A PUSH HAS
NO ENUMERABLE INVERSE. Recovering the predecessor of a board means guessing
which crates were shoved into their current squares and from where, so the
backward BFS the ps: sokobans use is unavailable. Instead:

  1. FORWARD, layer by layer, stopping at the first depth ``d*`` that produces a
     winning board.
  2. BACKWARD, ``on_path[d*]`` being the wins found while expanding layer
     ``d* - 1``, and ``on_path[j]`` every state of layer ``j`` with a successor
     in ``on_path[j + 1]``.

`on_path[j]` is exactly the set of states whose true distance to a win is
``d* - j``: having a successor in ``on_path[j+1]`` exhibits a path of that
length, and a shorter one would give a start-to-win path beating ``d*``. So one
sweep pair hands over the shortest plan, the proof that it is shortest, and the
EXACT tie set at every step, with no heuristic and no node budget that could
quietly bite. `PSExpert`'s plan memo, snapshot discipline and disk cache are
inherited unchanged; only the strategy under `_search` is replaced.

Cost is level 0: 1,253,447 states forward and 1.4 s backward, about 4 s and
190 MB all told, once, cached to `data/weird_dave_plans.json`. Levels 1 and 2
are 105 and 22 states.

Recovery
--------
`supports_recovery = True` with the family's RESET mode: the exploration prefix
presses random buttons and is undone by one `set_level`, so generation itself
only ever plans from a level start. `_search` nevertheless reads the LIVE grid
and answers from any board, which is what the epsilon detour and every probe
below rely on -- including the answer "this board is dead", which in this game
is a real answer rather than a give-up. Level 0's crates are its own tomb: four
crates and one target, but only one crate can ever be delivered through the
doorway, so shoving the wrong one into a corner strands the level. ``--recovery``
seats states drawn from the model's own closure into the interpreter, re-plans,
and requires every plan to WIN when replayed and every None to be a board an
independent forward BFS also finds no win from.

Rendering
---------
Two fixes in `data/puzzlescript_games/Weird_Dave.txt`, both the recurring
"the win condition is invisible" shape, both measured by ``--audit``.

Every board here renders at cell_px 4 (13x12 and 15x15), where
`_render_cell_sprite` samples a 5x5 sprite at rows/cols {0,1,3,4} -- the middle
is never read. `Target`'s art is four Yellow CORNER pixels plus a Pink cross on
row/column 2, so at cell_px 4 the goal square IS its four corners and nothing
else. `PlayerC` and `PlayerB` were opaque at all four corners, and so was any
snake cell once its corner decorations were placed. Since `_render_frame` sorts
player objects LAST so the player is always on top, that made ``target+playerc``,
``target+playerb`` and ``target+snake`` pixel-identical to the same body on bare
floor: the goal vanished under Dave, on a game whose goal never moves and whose
only job is to deliver a crate to it.

  1. All five human bodies, and `PlayerSnakeHeadDeco`, give up their four corner
     pixels (the [[ps:stickyban]] / [[ps:the_blob]] fix: the Target's marker and
     the bodies' transparency are made disjoint by construction).
  2. The eight corner DECORATIONS -- `SnakeTopLeft`, `SnakeBottomLeft`,
     `SnakeTopRight`, `SnakeBottomRight` and the four `*CornerDark` -- become
     `Transparent`. They are the objects that filled the snake's corners in, and
     they were already carrying no information: the game draws them in
     `#80FF00` and `#008000`, which are the SAME ARC index 14 as `PlayerSnake`
     itself (the one-green rule), so the entire outline system rendered as a
     flat green fill whose only effect was to hide what was underneath. The
     rules that place them are untouched; the objects simply draw nothing, the
     way `Marker1` and the four `CantMove` already do.

``--audit`` asserts the pairwise distinctness of every reachable cell
COMPOSITION at every cell size the levels use, that none of them renders as the
uniform letterbox colour, and -- the invariant the second fix rests on -- that
adding any set of decorations to a snake cell changes no pixel.

Augmentation
------------
Rotation is the adapter's, per-(seed, level) as for every game. This game is
also in `PuzzleScriptAdapter._FLIP_GAMES`, so a board is drawn at any of the
eight turns and mirrors -- with three levels that is the difference between 12
presentations for the whole game and 48. The argument is the one the flip set
wants and it is a proof rather than an inspection: one press gives every Player
the same force and a crate chain is collinear, so a press is a translation of
two disjoint sets by a fixed vector. Translation is injective, so nothing can
contest a destination and there is nothing for the interpreter's rule-expansion
order to decide; the geometry is "blocked by a Wall, and blocking cancels the
turn", which is exactly as mirror-symmetric as it is rotation-symmetric; and
the win condition names no direction. The art holds up too: `PlayerL` and
`PlayerR` are EXACT horizontal mirrors of each other in ARC colours
(``--audit`` asserts it pixel for pixel, and note it is only true after the
palette collapse -- the author's two colour lists are permuted), the snake and
the sparkles are symmetric, and `PlayerT`/`PlayerB` are drawn upside down by
the rotation augmentation this game already takes, so a mirror shows nothing
categorically new. ``--symmetry`` MEASURES both halves anyway: every level's
plan and a seeded random walk replayed at all 16 presentations, engine grid AND
frame compared against the transform after every press.

Reports
-------
``--plans``      every level's plan, replayed through the real interpreter.
``--selfcheck``  the model IS the interpreter; the plans are shortest; the tie
                 sets are exact.
``--engine``     the levels whose space is small enumerated through the REAL
                 interpreter, sharing no line with the model; level 0's
                 transitions checked against it over the states nearest its
                 start.
``--recovery``   re-planning from boards seated out of the model's closure.
``--audit``      the render audit described above.
``--symmetry``   the augmentation contract at all 16 presentations.

Verified
--------
``--plans`` 3/3 replayed through the interpreter to a WIN, 39 presses, 9 of them
with a tie set, every one inside the adapter's 200-press budget. ``--selfcheck``
0 violations: 8,400 random presses of model-against-interpreter with ACTION5 in
the mix (1,445 of them cancelling, which is the branch a collision-layer mistake
hides in), all three plans re-derived by an independent forward BFS, and all 39
tie sets re-derived by brute force. ``--engine`` 0 disagreements: levels 1 and 2
enumerated whole through the real interpreter (759 and 113 states) agreeing on
plan length AND tie sets, level 0's 12,676 transitions out of the 6,000 states
nearest its start all matching the model. ``--recovery`` 0 wrong over 152 seated
boards -- 55 re-planned to a WIN in the interpreter and 97 PROVED dead, which is
this game's shape rather than a defect. ``--audit`` 0 colliding compositions at
both board sizes, 0 pad clashes, the 13 decorations painting nothing, and
PlayerL confirmed PlayerR's exact mirror. ``--symmetry`` 0 violations over all
48 (level, presentation) pairs.

End to end: 6 episodes replayed FRAME-EXACT through `perform_action`, 18/18
level-wins, 323 frames, max action index 5, 234 expert steps ALL carrying an
optimal set (the 65 unlabelled steps are the exploration prefix the closing
RESET discards, plus each level's action-0 RESET), and two processes at the same
rng seed producing byte-identical episodes. About 2.3 s per episode after a
0.1 s warm start off `PLAN_CACHE`.

Corpus
------
    python solvers/generate_weird_dave_training.py --episodes N \\
        --out data/training_multi_level/weird_dave
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
                                     restore, screen_action, snapshot)
from utils.rotation import inverse_remap_action_full                  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Weird_Dave"

#: Engine directions, in the order ties are broken. ACTION5 is bound to nothing
#: (the game declares ``noaction`` and no rule reads it), so it is not a move and
#: branching on it would inflate every sweep for nothing.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

#: ``(dr, dc)`` per entry of `DIRS`.
_DELTA: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: How many random presses the model-vs-interpreter walks take per level, and
#: how long the mirrored replays are. Level 1 gets a tenth of the others because
#: ITS INTERPRETER STEP COSTS 224 ms against 14 ms on level 0 and 10 ms on level
#: 2 (measured): its snake is 24 cells and the decoration block is written as
#: neighbour rules over the whole body, so a press there is a couple of hundred
#: rule applications rather than a couple of dozen. That is a property of the
#: .txt, not of anything here -- the NATIVE model steps all three levels in
#: microseconds, which is exactly why the searches run on it -- but every mode
#: that drives the real interpreter pays it, so the budgets are split rather
#: than averaged. Generation itself is unaffected: level 1 is a nine-press plan.
_WALK_PRESSES: dict[int, int] = {0: 4000, 1: 400, 2: 4000}
_MIRROR_PRESSES: dict[int, int] = {0: 200, 1: 40, 2: 200}

#: How many boards ``--recovery`` seats per level. Level 0 gets fewer because
#: the INDEPENDENT check on a dead board there is a full forward enumeration of
#: whatever is reachable from it -- up to 4,135,322 states -- which is the only
#: way to turn "the field found no win" into "there is none".
_RECOVERY_SAMPLES: dict[int, int] = {0: 24, 1: 104, 2: 24}

#: Disk cache of every level's start plan AND its optimal-action sets. Level 0's
#: field is 1.25M states and ~4 s; the other two are milliseconds. This exists so
#: a `parallelize_generator` fan-out derives level 0 once rather than once per
#: core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "weird_dave_plans.json"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic, plus the search.

    A STATE is ``(crates << ashift) | anchor``: ``anchor`` is the flat index
    ``r * w + c`` of the cell the body's first (row-major) square occupies, and
    ``crates`` is a bitmask over the same indexing. Nothing else on the board
    moves -- walls, void and targets are fixed, and the body's SHAPE is an
    invariant of the game rather than an assumption (a press either translates
    every cell of it or cancels; `WeirdDaveExpert.read` re-checks it against the
    level's start shape on every read).

    THE MECHANIC, in full:

      * the destination cells of the body are ``body_mask[anchor + d]``. If any
        of them is a Wall or off the board, ``body_ok`` is False and the press
        does nothing at all;
      * every crate standing on one of those cells is PUSHED, and so is every
        crate in the collinear chain behind one, to fixpoint;
      * if any pushed crate's own destination is a Wall or off the board, the
        press does nothing -- the crates are not left behind and the body does
        not move without them;
      * otherwise the body lands on ``anchor + d`` and the pushed crates each
        advance one square.

    Nothing can ever contest a destination, which is why the resolution above
    has no ordering in it: body and crates translate by the same fixed vector,
    which is injective; a pushed crate cannot land on a body cell (that would
    make its source a body cell, and Player and Crate share a collision layer);
    and two crates cannot land on the same square. That argument is also this
    game's ticket into `PuzzleScriptAdapter._FLIP_GAMES`.

    Note the asymmetry that makes the game a puzzle: a press that pushes
    nothing is a pure translation and the opposite press undoes it, but a press
    that shoves a crate is NOT undoable. So `field` has to be able to answer
    "no win from here" -- and on level 0, whose four crates serve one target
    behind a one-square doorway, it genuinely has to (see ``--recovery``).
    """

    __slots__ = ("h", "w", "n", "shape", "targets", "ashift", "amask",
                 "_sh", "_block", "body_mask", "body_ok", "_nbr")

    def __init__(self, h: int, w: int, walls, voids, shape, targets):
        self.h, self.w = h, w
        self.n = n = h * w
        self.shape = tuple(shape)
        all_mask = (1 << n) - 1

        col_first = sum(1 << (r * w) for r in range(h))
        col_last = sum(1 << (r * w + w - 1) for r in range(h))
        row_first = (1 << w) - 1
        row_last = row_first << ((h - 1) * w)

        #: ``_sh[di] == (keep, amount, towards_high)``: shifting ``P & keep`` by
        #: ``amount`` moves every set cell one square in direction ``DIRS[di]``.
        #: ``keep`` drops the rank that would wrap around the board edge, which
        #: is what makes the shift total -- nothing after this bounds-checks.
        self._sh = (
            (all_mask & ~row_first, w, False),   # up
            (all_mask & ~row_last, w, True),     # down
            (all_mask & ~col_first, 1, False),   # left
            (all_mask & ~col_last, 1, True),     # right
        )

        solid = set(walls) | set(voids)
        _check_enclosed(h, w, solid, set(walls))

        blocked_mask = 0
        for i in solid:
            blocked_mask |= 1 << i
        free = all_mask & ~blocked_mask
        #: ``_block[di]``: cells whose neighbour in direction ``di`` is solid or
        #: off the board -- i.e. the crates a press in that direction cannot
        #: move. Derived as "not the cells that CAN move there", so the walls and
        #: the board edge are one rule rather than two.
        self._block = tuple(all_mask & ~self._shift(free, _OPP[di])
                            for di in range(4))

        self.targets = 0
        for i in targets:
            self.targets |= 1 << i

        self.ashift = n.bit_length()
        self.amask = (1 << self.ashift) - 1

        #: The body, precomputed at every anchor it could sit on. ``body_ok[i]``
        #: is False when some cell of the shape would be solid or off the board,
        #: and ``body_mask[i]`` is 0 there -- so the whole "can the body move"
        #: test is one array lookup.
        self.body_mask = [0] * n
        self.body_ok = [False] * n
        for ar in range(h):
            for ac in range(w):
                mask, ok = 0, True
                for sr, sc in self.shape:
                    r, c = ar + sr, ac + sc
                    if not (0 <= r < h and 0 <= c < w) or (r * w + c) in solid:
                        ok = False
                        break
                    mask |= 1 << (r * w + c)
                i = ar * w + ac
                self.body_ok[i] = ok
                self.body_mask[i] = mask if ok else 0

        #: ``_nbr[di][i]``: the flat index one square in direction ``di``, or -1
        #: off the board. Kept as a table rather than arithmetic so a left/right
        #: step cannot silently wrap onto the next row.
        self._nbr = [[-1] * n for _ in range(4)]
        for r in range(h):
            for c in range(w):
                i = r * w + c
                for di, (dr, dc) in enumerate(_DELTA):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        self._nbr[di][i] = nr * w + nc

    # -- packing -------------------------------------------------------------
    def pack(self, anchor: int, crates: int) -> int:
        return (crates << self.ashift) | anchor

    def unpack(self, state: int) -> tuple[int, int]:
        return state & self.amask, state >> self.ashift

    # -- the mechanic --------------------------------------------------------
    def _shift(self, mask: int, di: int) -> int:
        keep, amount, high = self._sh[di]
        return ((mask & keep) << amount) if high else ((mask & keep) >> amount)

    def step(self, state: int, di: int) -> int:
        """The board after pressing ``DIRS[di]``. Returns ``state`` unchanged
        when the press cancels, which is a non-move: no shortest path contains
        one, and every sweep here treats a self-loop as no edge at all."""
        anchor = state & self.amask
        nxt = self._nbr[di][anchor]
        if nxt < 0 or not self.body_ok[nxt]:
            return state
        crates = state >> self.ashift
        pushed = self.body_mask[nxt] & crates
        if pushed:
            while True:
                grown = pushed | (self._shift(pushed, di) & crates)
                if grown == pushed:
                    break
                pushed = grown
            if pushed & self._block[di]:
                return state
            crates = (crates & ~pushed) | self._shift(pushed, di)
        return (crates << self.ashift) | nxt

    def won(self, state: int) -> bool:
        """``ALL Target ON Crate``: every target square holds a crate.

        The interpreter expresses the same thing by consuming both objects into
        a `CrateTarget` (``LATE [Crate Target] -> [CrateTarget]``) and asking
        whether any bare `Target` is left, which is why a satisfied goal has no
        Target on it to find -- see `WeirdDaveExpert.read`."""
        return not (self.targets & ~(state >> self.ashift))

    # -- the search ----------------------------------------------------------
    def field(self, state: int, cap: int):
        """``(on_path, d_star)``: for every ``j``, the states exactly ``j``
        presses from ``state`` whose distance to a win is exactly
        ``d_star - j``. ``(None, None)`` if this board cannot be won, or if the
        sweep passed ``cap`` states without finding a win.

        Sweep 1 is a forward BFS kept in LAYERS, stopped at the first depth that
        produces a win. Sweep 2 walks those layers backwards: ``on_path[d_star]``
        is the winning boards found while expanding the last layer, and
        ``on_path[j]`` is every state of layer ``j`` with a successor in
        ``on_path[j + 1]``.

        Why layers instead of a backward BFS over inverse moves: a push has no
        enumerable inverse. Undoing one means knowing which crates were shoved
        and from where, and that is a function of the PREDECESSOR, not of the
        board in hand. The induction above needs no inverse and costs one extra
        forward sweep.

        Exactness. ``on_path[j]`` exhibits a win path of length ``d_star - j``,
        so the true distance is at most that; and it cannot be less, because a
        state at forward distance ``j`` with a win ``h`` presses away gives a
        start-to-win path of ``j + h``, which cannot beat ``d_star``. States on
        no shortest path are simply absent, and nothing ever asks about them.
        """
        if self.won(state):
            return [{state}], 0
        seen = {state}
        layers = [[state]]
        wins: set = set()
        while True:
            frontier = []
            for cur in layers[-1]:
                for di in range(4):
                    nxt = self.step(cur, di)
                    if nxt in seen:
                        continue
                    seen.add(nxt)
                    if self.won(nxt):
                        wins.add(nxt)          # a won board is terminal
                    else:
                        frontier.append(nxt)
            if wins:
                break
            if not frontier or len(seen) > cap:
                return None, None
            layers.append(frontier)

        d_star = len(layers)
        on_path: list = [None] * (d_star + 1)
        on_path[d_star] = wins
        for j in range(d_star - 1, -1, -1):
            ahead = on_path[j + 1]
            on_path[j] = {s for s in layers[j]
                          if any(self.step(s, di) in ahead for di in range(4))}
        return on_path, d_star

    def optimal(self, on_path, state: int, j: int):
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``, which sits at forward distance ``j``. Ties come out
        in `DIRS` order, which is what makes a re-derived plan byte-identical
        across processes."""
        ahead = on_path[j + 1]
        out = []
        for di in range(4):
            nxt = self.step(state, di)
            if nxt != state and nxt in ahead:
                out.append((di, nxt))
        return out

    def solve(self, state: int, cap: int) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal
        set at every step, or None if this board cannot be won from here (or is
        past ``cap``)."""
        on_path, d_star = self.field(state, cap)
        if on_path is None:
            return None
        presses, optsets = [], []
        cur = state
        for j in range(d_star):
            best = self.optimal(on_path, cur, j)
            if not best:                    # unreachable: on_path[j] has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)

    def closure(self, state: int, cap: int) -> "list | None":
        """Every state reachable from ``state``, wins included and terminal, or
        None past ``cap``. Only the probes use it -- `field` never needs the
        whole space, and on level 0 the whole space is 4,135,322 states."""
        seen = {state}
        order = [state]
        queue = deque([state])
        while queue:
            cur = queue.popleft()
            if self.won(cur):
                continue
            for di in range(4):
                nxt = self.step(cur, di)
                if nxt in seen:
                    continue
                if len(seen) >= cap:
                    return None
                seen.add(nxt)
                order.append(nxt)
                queue.append(nxt)
        return order


#: ``_OPP[di]`` is the index of the opposite direction. Used once, to turn "the
#: cells that can move into me" into "the cells I can move into".
_OPP: tuple[int, ...] = (1, 0, 3, 2)


def _check_enclosed(h: int, w: int, solid: set, walls: set) -> None:
    """Raise unless the playable region is a closed room of `Wall`.

    THE MODEL'S ONE GEOMETRIC ASSUMPTION. `Collidable` in the .txt is `Wall`
    alone, so the interpreter's "can this crate move" pre-pass says nothing
    about the board EDGE or about `Void` (which lives on the BackTexture
    collision layer, not the player's). This model treats both as walls -- i.e.
    as things that CANCEL the press. If a body cell or a crate could ever reach
    them the two would part company: the interpreter would fail that one
    object's move and let the rest of the board slide, shearing the body.

    It cannot happen on any shipped level -- all three are closed rooms, no free
    cell touches the border and none touches a void square -- and this is where
    that stops being an assumption. Editing a level so that it is not true is a
    crash, not a wrong plan.
    """
    bad = []
    for r in range(h):
        for c in range(w):
            if (r * w + c) in solid:
                continue
            if r in (0, h - 1) or c in (0, w - 1):
                bad.append(((r, c), "on the board edge"))
                continue
            for dr, dc in _DELTA:
                nb = (r + dr) * w + (c + dc)
                if nb in solid and nb not in walls:
                    bad.append(((r, c), "next to a Void square"))
                    break
    if bad:
        raise AssertionError(
            f"Weird_Dave: the playable region is not a closed room of Wall -- "
            f"{len(bad)} free cell(s) escape it, e.g. {bad[0][0]} {bad[0][1]}. "
            "The model treats the board edge and Void as cancelling walls; the "
            "interpreter does not (see _check_enclosed).")


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class WeirdDaveExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Board` field.

    The base class keeps the memo, the level scoping and the disk cache; only
    the strategy underneath changes, the way `PSEnumExpert` replaces it. Here
    the strategy is "read the live board, sweep it forwards then backwards, hand
    back the shortest plan and its exact tie sets", so `heuristic` is never
    called and asserts rather than returning a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered exactly
    -- including "this board is now dead", which really does happen here.

    `_key` is INHERITED (every non-background cell), not narrowed to the moving
    pieces: that key is canonical across levels because the walls and targets
    are in it too, so no ``scope_by_level`` is needed and the
    ``plan_cache_path`` signature covers the whole board -- an edited level is
    then a cache MISS rather than a wrong plan.
    """

    directions = list(DIRS)
    plan_cache_path = PLAN_CACHE

    #: Runaway guard on `_Board.field`. Level 0's whole reachable space is
    #: 4,135,322 states and its field from the start visits 1,253,447, so this
    #: is "the largest sweep this game can possibly need, plus room": a field
    #: that hits it is answering about a board no press sequence can reach.
    #: Generation only ever plans from a level start (RESET recovery restores
    #: it); it is the probes' seated boards that can wander somewhere expensive.
    state_cap: int = 5_000_000

    def setup(self) -> None:
        self.player_ids = set(self.game._engine._player_indices)
        self.wall_ids = set(self.g.resolve_object_name("wall"))
        self.void_ids = set(self.g.resolve_object_name("void"))
        self.target_ids = set(self.g.resolve_object_name("target"))
        self.cratetarget_ids = set(self.g.resolve_object_name("cratetarget"))
        #: A satisfied goal is ONE object, so `CrateTarget` counts as both.
        self.crate_ids = set(self.g.resolve_object_name("crate")) | self.cratetarget_ids
        self.goal_ids = self.target_ids | self.cratetarget_ids
        #: `_Board`s by STATIC signature (see `read`), not by level index.
        self._boards: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "WeirdDaveExpert reads an exact distance field; heuristic is unused")

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng) -> tuple:
        """``(board, state)`` for the engine's current grid.

        Boards are cached by their STATIC signature (dimensions, walls, void,
        targets), not by level index, so the sweeps never rebuild the shift
        tables and one board is shared by every state of its level. THE BODY'S
        SHAPE IS PART OF THE BOARD, not of the state, so it is asserted against
        the cached one rather than silently re-keyed: the rigid-translation
        argument is the model's central claim, and a board whose shape changed
        would be a board this file is wrong about.
        """
        h, w = eng.height, eng.width
        walls, voids, targets, body = [], [], [], []
        crates = 0
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.void_ids:
                    voids.append(i)
                if cell & self.goal_ids:
                    targets.append(i)
                if cell & self.crate_ids:
                    crates |= 1 << i
                if cell & self.player_ids:
                    body.append((r, c))
        assert targets, "Weird_Dave: no Target on the board -- nothing to win"
        assert body, "Weird_Dave: no Player on the board"
        assert not (crates & self._body_mask(body, w)), (
            "Weird_Dave: a Crate and a Player share a cell, which their common "
            "collision layer forbids")

        ar, ac = body[0]
        shape = tuple((r - ar, c - ac) for r, c in body)
        sig = (h, w, tuple(walls), tuple(voids), tuple(targets))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, voids, shape, targets)
        assert board.shape == shape, (
            f"Weird_Dave: the body's shape changed from {board.shape} to "
            f"{shape} -- a press is supposed to translate it or cancel")
        return board, board.pack(ar * w + ac, crates)

    @staticmethod
    def _body_mask(body, w: int) -> int:
        mask = 0
        for r, c in body:
            mask |= 1 << (r * w + c)
        return mask

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if board.won(state):
            return Plan([], [])
        return board.solve(state, self.state_cap)

    # -- seating a model state back into the interpreter ---------------------
    def seating(self, eng):
        """``(static_grid, player_by_offset)`` for the CURRENT level, i.e. what
        `seat` needs to write any model state into the engine.

        The static grid is this level with every Player, Crate and CrateTarget
        lifted off and the bare `Target` put back where a `CrateTarget` had
        consumed it. `player_by_offset` maps each of the body's shape offsets to
        the object id sitting there, because Dave's cells are DIFFERENT objects
        (the plus is five of them) and a decode that painted one type would be
        drawing a board the game cannot make.

        The transient markers (`CantMove*`, `Marker*`) are deliberately NOT
        reproduced: the first four rules of every turn delete all of them before
        anything reads one, so a seated board steps identically without them.
        `_recovery` checks that claim rather than asserting it -- it seats the
        level start and requires all four presses to land where they land from
        the real one.
        """
        w = eng.width
        moving = self.player_ids | self.crate_ids
        static = [[cell - moving for cell in row] for row in eng.grid]
        target_id = min(self.target_ids)
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & self.goal_ids:
                    static[r][c] = static[r][c] | {target_id}
        body = [(r, c) for r, row in enumerate(eng.grid)
                for c, cell in enumerate(row) if cell & self.player_ids]
        ar, ac = body[0]
        by_offset = {(r - ar, c - ac): min(eng.grid[r][c] & self.player_ids)
                     for r, c in body}
        return static, by_offset

    def seat(self, eng, board: _Board, state: int, seating) -> None:
        """Write ``state`` onto the engine's grid, in place."""
        static, by_offset = seating
        w = board.w
        eng.grid = [[set(cell) for cell in row] for row in static]
        anchor, crates = board.unpack(state)
        ar, ac = divmod(anchor, w)
        for (sr, sc), obj in by_offset.items():
            eng.grid[ar + sr][ac + sc].add(obj)
        crate_id = min(self.crate_ids - self.cratetarget_ids)
        cratetarget_id = min(self.cratetarget_ids)
        target_id = min(self.target_ids)
        i = 0
        while crates:
            if crates & 1:
                r, c = divmod(i, w)
                if target_id in eng.grid[r][c]:
                    eng.grid[r][c].discard(target_id)
                    eng.grid[r][c].add(cratetarget_id)
                else:
                    eng.grid[r][c].add(crate_id)
            crates >>= 1
            i += 1
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class WeirdDaveSolver(PSAStarSolver):
    game_id = "puzzlescript_weird_dave"
    game_name = GAME_NAME
    expert_cls = WeirdDaveExpert

    #: `games/ps:weird_dave/ps:weird_dave.py` is a plain passthrough (it
    #: constructs the adapter and nothing else), so there is nothing to gain by
    #: routing through it -- but it IS what a live agent is handed, so if that
    #: wrapper ever grows a patch this must be set to ``"ps:weird_dave"``.
    game_module_id = ""

    #: Unused: the expert enumerates an exact field rather than searching under
    #: a heuristic, so there is no weight to trade and no node budget that could
    #: quietly bite. Left at the base values so nothing reads a lie off them --
    #: the cap that IS real is `WeirdDaveExpert.state_cap`.
    weight = 1
    node_cap = 400_000

    #: The longest plan is 27 presses (level 0); the rest is room for a re-plan
    #: after the exploration prefix. Stays well under the adapter's own 200-step
    #: per-level budget, which `set_level` resets before the plan starts anyway.
    max_steps = 80

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it fills the disk cache in one pass and makes level
        0's four-second field visible as startup rather than as a mysterious
        pause."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new():
    solver = WeirdDaveSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _brute(board: _Board, state: int, limit: int):
    """Presses to a win from ``state``, searched fresh by forward BFS, or None
    if there is no win at all (the sweep exhausts the space) or none within
    ``limit``. Shares no code with `_Board.field`'s backward half, which is the
    point of it."""
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


def _plans() -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once."""
    _solver, game, expert, solvable = _new()
    total = ties = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board, state = expert.read(eng)
        anchor, crates = board.unpack(state)
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
              f"{len(board.shape):2d} body cells / {bin(crates).count('1')} "
              f"crates / {bin(board.targets).count('1')} target(s)  "
              f"{len(plan):3d} presses  win={won}  "
              f"(budget {game._max_steps}, {room})  "
              f"{tie_steps:3d} steps with a tie set")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second equally-right answer")
    print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _selfcheck(walks: "dict | None" = None, verbose: bool = True) -> int:
    """Three things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's.** A seeded random walk on every
       level, comparing `_Board.step`'s packed state against the engine's grid
       after EVERY press -- including the presses that cancel, which is where a
       collision-layer mistake hides, and including ACTION5, which must be a
       measured no-op. `WeirdDaveExpert.read` re-asserts the body's shape and
       the crate/player disjointness on every one of those reads, so the walk
       is also the rigidity check.
    2. **The plans are shortest.** A plain forward BFS to the first win, which
       is shortest by construction and shares no code with the field's backward
       half, must return the field's plan length on every level.
    3. **The tie sets are exact.** Every step's optimal set is re-derived by
       brute force (a fresh depth-bounded BFS from each of the four successors)
       and must match the field's answer exactly.

    Level 0 is 1.25M states, so 2 and 3 are minutes rather than seconds there;
    that is the price of proving the number rather than asserting it. The walk
    lengths come from `_WALK_PRESSES`, which is not uniform -- see the note
    there on level 1's interpreter cost.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    walks = _WALK_PRESSES if walks is None else walks
    bad = 0

    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        rng = random.Random(f"weird_dave:selfcheck:{level}")
        walk_presses = walks.get(level, 400)
        drift = cancels = wins = 0
        for _ in range(walk_presses):
            direction = rng.choice(DIRS + ("action",))
            eng.step(direction)
            if direction != "action":
                nxt = board.step(state, DIRS.index(direction))
                cancels += nxt == state
                state = nxt
            _b, live = expert.read(eng)      # also re-asserts the body's shape
            if live != state:
                drift += 1
                break
            if eng.check_win():              # terminal; start the walk again
                wins += 1
                game.set_level(level)
                _b, state = expert.read(eng)
        bad += drift
        if verbose:
            print(f"  L{level:2d}: model {'MATCHES' if not drift else 'DIVERGES from'}"
                  f" the interpreter over {walk_presses} random presses "
                  f"({cancels} of them cancelled, {wins} win(s) along the way)")

    for level in range(game.n_levels):
        game.set_level(level)
        board, state = expert.read(eng)
        plan = expert.plan(eng, level)
        shortest = _brute(board, state, board.n)
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
                if board.won(nxt):
                    if rest == 1:
                        truth.append(DIRS[di])
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
    return bad


def _recovery(samples: "dict | None" = None, verbose: bool = True) -> int:
    """Re-planning from boards that are NOT the level start, checked against the
    interpreter in both directions.

    This is the one that matters at generation time. A press that shoves a crate
    is irreversible, so `_search` has to be right about arbitrary states BOTH
    ways: a plan it returns has to WIN when replayed through the real
    interpreter, and a None has to be a board an independent forward BFS also
    finds no win from. A None that were merely a give-up would not show up in
    any of `--selfcheck`'s three passes, all of which only ever ask about level
    starts -- and `record_level` reads it as "reset", so it would silently ship
    a shorter corpus.

    Boards come from the MODEL and are then SEATED into the interpreter through
    `WeirdDaveExpert.seat`, so every one of them is checked end to end. Where
    the closure enumerates (levels 1 and 2, at 105 and 22 states) that IS the
    sample -- every reachable board is tried. Level 0's 4,135,322 do not, so it
    falls back to `_sample_walk`, a spread of random walks of random length from
    the start, which is the same population `record_level`'s exploration prefix
    draws from.

    The seating itself is verified first: seat the level's own start and require
    all four presses to land exactly where they land from the untouched board.
    That is the claim that the transient `CantMove`/`Marker` objects a seated
    grid lacks are re-derived before anything reads them -- the first four rules
    of every turn delete all of them.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    samples = _RECOVERY_SAMPLES if samples is None else samples
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, start = expert.read(eng)
        seating = expert.seating(eng)

        # -- the seating is faithful --
        truth = {}
        for di, direction in enumerate(DIRS):
            game.set_level(level)
            eng.step(direction)
            truth[di] = expert.read(eng)[1]
        seat_bad = 0
        for di, direction in enumerate(DIRS):
            game.set_level(level)
            expert.seat(eng, board, start, seating)
            eng.step(direction)
            seat_bad += expert.read(eng)[1] != truth[di]
        bad += seat_bad
        if verbose and seat_bad:
            print(f"  L{level:2d}: SEATING IS NOT FAITHFUL ({seat_bad}/4 presses)")

        rng = random.Random(f"weird_dave:recovery:{level}")
        space = board.closure(start, 200_000)
        if space is None:            # level 0: walk to a spread instead
            space = _sample_walk(board, start, rng, 1_000)
        live = [s for s in space if not board.won(s)]
        want = samples.get(level, 24)
        picks = live if len(live) <= want else rng.sample(live, want)
        wrong = dead = won_after = 0
        for state in picks:
            game.set_level(level)
            expert.seat(eng, board, state, seating)
            plan = expert._search(eng)
            # Independent: an unbounded forward BFS over the whole reachable
            # space, which either finds a win or PROVES there is none.
            shortest = _brute(board, state, board.n * len(board.shape))
            if plan is None:
                dead += 1
                wrong += shortest is not None
                continue
            if shortest != len(plan):
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
            print(f"  L{level:2d}: {len(picks)} seated boards -- {won_after} "
                  f"re-planned to a WIN in the interpreter, {dead} proved dead, "
                  f"{wrong} WRONG")
    return bad


def _sample_walk(board: _Board, start: int, rng: random.Random, walks: int):
    """A spread of states from a board whose closure is past the cap `_recovery`
    is willing to hold: ``walks`` random walks of random length from the start,
    deduped. Level 0 is the only caller (4,135,322 states)."""
    out = []
    seen = set()
    for _ in range(walks):
        cur = start
        for _ in range(rng.randrange(1, 120)):
            cur = board.step(cur, rng.randrange(4))
            if board.won(cur):
                break
        if cur not in seen:
            seen.add(cur)
            out.append(cur)
    return out


#: Levels ``--engine`` enumerates in full. Level 0 is excluded and only from
#: that half: 4,135,322 reachable states is not an enumeration, it is an
#: afternoon. It gets the BOUNDED transition check below instead, which is the
#: same comparison over the states nearest its start.
_ENGINE_LEVELS: tuple[int, ...] = (1, 2)

#: How many interpreter states level 0's bounded check walks. Four presses per
#: state at 14 ms each, so this is a five-minute sweep -- the number is a time
#: budget, not a property of the level, and raising it only widens the ball.
_ENGINE_L0_CAP: int = 6_000


def _engine(verbose: bool = True) -> int:
    """The proof that owes nothing to the native model: drive the REAL
    interpreter.

    Two halves, because the three levels are three orders of magnitude apart.

    * **Levels 1 and 2 are ENUMERATED.** `StateGraph.build` walks every state
      reachable by pressing actual buttons on the engine, keys the resulting
      grid whole (players, crates, decorations, the transient `CantMove`
      markers and all) and solves the graph exactly. It shares no line of code
      with `_Board` -- no modelled push, no modelled cancel, no modelled win
      test -- so agreement on both the plan LENGTH and the per-step optimal SET
      is an independent derivation of everything this file claims about them.
    * **Level 0 is BOUNDED.** A BFS over the interpreter from its start, seated
      through `WeirdDaveExpert.seat` so no snapshots are kept, comparing all
      four of `_Board.step`'s answers against the engine's at every state it
      expands. That is the same claim as `--selfcheck`'s walk but swept rather
      than sampled: every transition out of the ball of states nearest the
      start, including every one that cancels. The sweep stops once
      ``_ENGINE_L0_CAP`` states have been DISCOVERED, so the last ones found sit
      in the queue unexpanded -- the transition count it reports is what was
      actually checked, not four times the state count.

    THE RESTING GRID REMEMBERS WHICH WAY YOU LAST PRESSED, which is why the two
    enumerations report several times the model's state count (759 against 105
    on level 1, 113 against 22 on level 2). ``[ > Player | Collidable ] ->
    [ Player CantMoveUp | Collidable ]`` and the rule that propagates it leave
    `CantMove` markers ON THE PLAYER'S OWN CELLS, and nothing deletes them until
    the first rules of the NEXT turn -- so a board carries an invisible record
    of the last direction pressed. It is `Transparent` and mechanically inert
    (every rule that reads it runs after the rules that rebuild it), and it does
    not affect these enumerations' answers, which is what the AGREE is saying.
    But it is exactly why `_Board` keys on ``(anchor, crates)`` rather than on
    the grid: the grid over-counts states by roughly the branching factor, and
    `PSExpert._key` -- which is the plan memo's key -- inherits that.
    """
    from solvers.common.ps_astar import StateGraph                    # noqa: PLC0415

    _solver, game, expert, _solvable = _new()
    eng = game._engine
    bad = 0

    for level in _ENGINE_LEVELS:
        game.set_level(level)
        plan = expert.plan(eng, level)
        game.set_level(level)

        def key_fn(e):
            return tuple(tuple(frozenset(cell) for cell in row) for row in e.grid)

        graph = StateGraph.build(eng, key_fn, list(DIRS), node_cap=200_000)
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
            print(f"  L{level:2d}: {len(graph.succ)} engine states, "
                  f"{len(eplan)} presses vs the field's {len(plan)} -- "
                  f"{'AGREE' if same_len else 'DISAGREE'}; tie sets "
                  f"{'AGREE' if same_sets else 'DISAGREE'}")

    for level in sorted(set(range(game.n_levels)) - set(_ENGINE_LEVELS)):
        game.set_level(level)
        board, start = expert.read(eng)
        seating = expert.seating(eng)
        seen = {start}
        queue = deque([start])
        checked = mismatches = cancels = 0
        while queue and len(seen) < _ENGINE_L0_CAP:
            cur = queue.popleft()
            if board.won(cur):
                continue
            for di, direction in enumerate(DIRS):
                expert.seat(eng, board, cur, seating)
                eng.step(direction)
                live = expert.read(eng)[1]
                checked += 1
                model = board.step(cur, di)
                if live != model:
                    mismatches += 1
                    if mismatches == 1:
                        print(f"  L{level:2d}: FIRST MISMATCH at anchor "
                              f"{board.unpack(cur)[0]} pressing {direction}")
                cancels += live == cur
                if live not in seen:
                    seen.add(live)
                    queue.append(live)
        bad += mismatches
        if verbose:
            print(f"  L{level:2d}: {len(seen)} engine states discovered, "
                  f"{checked} transitions checked ({cancels} cancelled) -- "
                  f"{'AGREE' if not mismatches else f'{mismatches} MISMATCH'}")
    return bad


# ---------------------------------------------------------------------------
# Render audit
# ---------------------------------------------------------------------------

#: The decorations. Every one of them renders NOTHING after the fix in the .txt
#: (`Transparent`, or -- the four `*Line` -- painting pixels `PlayerSnake`
#: already paints in the same ARC green), and `PlayerSnakeHeadDeco` is a
#: non-player object so `_render_frame` draws it UNDER the snake that is always
#: on the same square. `_audit` requires exactly that: adding any of them to a
#: cell must change no pixel. They are what used to fill the snake's corners in
#: and hide the Target.
_DECOR: tuple[str, ...] = (
    "topline", "bottomline", "leftline", "rightline",
    "snaketopleft", "snaketopright", "snakebottomleft", "snakebottomright",
    "topleftcornerdark", "toprightcornerdark",
    "bottomleftcornerdark", "bottomrightcornerdark",
    "playersnakeheaddeco",
)

#: The bodies, in the order the .txt declares them.
_BODIES: tuple[str, ...] = ("playert", "playerl", "playerc", "playerr",
                            "playerb", "playersnake", "playersparkle")

#: The one art ORBIT the mirror augmentation is allowed to trade on: `PlayerL`
#: and `PlayerR` are each other's horizontal mirror, so an orientation-reversing
#: transform drawing one where the other stood is the flip working correctly,
#: not two objects rendering as each other. `_audit`'s fourth pass excludes
#: exactly these, and flags everything else.
_MIRROR_PAIRS: dict[str, str] = {"playerl": "playerr", "playerr": "playerl"}


def _partner(comp: tuple) -> tuple:
    """``comp`` with its bodies swapped for their `_MIRROR_PAIRS` partners."""
    return tuple(_MIRROR_PAIRS.get(o, o) for o in comp)


def _reverses(t) -> bool:
    """True when the group element ``(k, hflip, vflip)`` reverses orientation --
    a quarter turn does not, each flip does, and two flips cancel."""
    _k, hflip, vflip = t
    return hflip != vflip


def _audit(verbose: bool = True) -> int:
    """Four passes over every reachable cell COMPOSITION.

    **Pass 1 -- distinctness**, at every cell size the boards use (the size list
    is derived rather than assumed, so an edited level is covered). WHOLE 64x64
    frames of uniform boards are compared rather than one cell out of a mixed
    board: `_render_frame` upscales and centre-pads, so slicing a cell by
    ``cell_px`` arithmetic reads the wrong pixels (the ps:explod lesson). Two
    uniform boards render identically iff their cells do.

    The compositions are floor, void, wall, the goal, a crate, a SATISFIED goal,
    each of the seven bodies, and each body standing ON the goal. That last
    family is the one this game's art got wrong and the one a colour-only audit
    would skip: the win condition is delivered to a square Dave walks over
    constantly, so ``target+body`` versus ``body`` is the pair that has to
    differ. ``crate + target`` is absent because the interpreter cannot make it
    -- ``LATE [Crate Target] -> [CrateTarget]`` consumes both into one object,
    which is why `cratetarget` is in the list instead. So is ``player + wall``
    (one collision layer) and ``player + crate`` (likewise).

    **Pass 2 -- the letterbox.** No composition may render as the uniform pad
    colour, because a board's border is painted with it and there would then be
    no seam between the terrain and the edge of the picture. This is the
    ps:stand_iii check, and a pairwise matrix passes it every time -- the pad is
    not a composition.

    **Pass 3 -- the decorations draw nothing.** Adding any subset of `_DECOR` to
    a snake cell, with or without a Target under it, must change no pixel. That
    is the invariant the .txt fix rests on: the decoration objects still exist
    and the rules still place them, they simply no longer paint over the goal.

    **Pass 4 -- the group.** The game is in `_FLIP_GAMES`, so a board is drawn
    at any of the eight turns and mirrors. What this requires is that no
    transform of one composition is another's art -- with ONE declared orbit,
    `_MIRROR_PAIRS`: an orientation-REVERSING element may take `PlayerL` onto
    `PlayerR` and back, because that is the pairing the flip argument leans on
    rather than a collision. `PlayerL` and `PlayerR` are exact horizontal
    mirrors of each other, so a mirrored frame draws mirrored art in the
    mirrored place, exactly as it mirrors the motion (the ps:snakeoban
    argument). They are only mirrors after the palette collapse: the author
    gave them permuted colour lists, ``#FFC080 #408000 #800000`` against
    ``#FFC080 #800000 #408000``, which land on the same two ARC indices in the
    opposite order. Anything else that maps onto another composition IS a
    violation, and the pass also reports which compositions are invariant under
    the whole group rather than assuming any are. It runs on SQUARE boards,
    because a rotation of a non-square board also moves the letterbox and every
    comparison would pass for the wrong reason.
    """
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert, _solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    floor = ("background", "backtexture")
    comps = [floor, ("background", "void"), floor + ("wall",),
             floor + ("target",), floor + ("crate",), floor + ("cratetarget",)]
    for body in _BODIES:
        comps.append(floor + (body,))
        comps.append(floor + ("target", body))
    pad = 5

    def name(comp):
        return "+".join(o for o in comp if o != "background") or "floor"

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    def shoot(h, w, comp):
        eng.height, eng.width = h, w
        eng.grid = [[{idx[o] for o in comp} for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g)).copy()

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {c: shoot(h, w, c) for c in comps}
        clashes = [(a, b) for a, b in itertools.combinations(comps, 2)
                   if np.array_equal(shots[a], shots[b])]
        blank = [c for c in comps if not (shots[c] != pad).any()]
        bad += len(clashes) + len(blank)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): {len(shots)} "
                  f"compositions, {'OK' if not clashes and not blank else 'BAD'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {name(a)}  ==  {name(b)}")
        for c in blank:
            print(f"      {name(c)} renders as the uniform letterbox colour")

    # -- pass 3: the decorations are invisible --
    rng = random.Random("weird_dave:audit:decor")
    for (h, w), _levels in sorted(sizes.items()):
        visible = 0
        for base in (floor + ("playersnake",), floor + ("target", "playersnake")):
            plain = shoot(h, w, base)
            subsets = [_DECOR] + [tuple(d for d in _DECOR if rng.random() < 0.5)
                                  for _ in range(12)]
            for extra in subsets:
                if not np.array_equal(shoot(h, w, base + extra), plain):
                    visible += 1
        bad += visible
        if verbose:
            print(f"  {h:2d}x{w:2d}: the {len(_DECOR)} decorations paint "
                  f"{'nothing' if not visible else f'{visible} VISIBLE pixel set(s)'}")

    # -- pass 4: the 8-element group --
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
        hits = [(a, b, t) for a, b in itertools.permutations(comps, 2)
                for t in group
                if np.array_equal(transform(shots[a], *t), shots[b])]
        clashes = [(a, b, t) for a, b, t in hits
                   if not (_reverses(t) and _partner(a) == b)]
        orbit = len(hits) - len(clashes)
        invariant = [name(c) for c in comps
                     if all(np.array_equal(transform(shots[c], *t), shots[c])
                            for t in group)]
        mirrored = np.array_equal(np.fliplr(shots[floor + ("playerl",)]),
                                  shots[floor + ("playerr",)])
        bad += len(clashes) + (not mirrored)
        if verbose:
            print(f"  cell {cell}px ({n}x{n}): {len(clashes)} composition(s) "
                  f"whose transform is another's art ({orbit} declared "
                  f"PlayerL/PlayerR orbit hit(s) allowed); PlayerL is "
                  f"{'' if mirrored else 'NOT '}PlayerR mirrored; "
                  f"group-invariant: {', '.join(invariant)}")
        for a, b, t in clashes:
            print(f"      {name(a)} under {t} == {name(b)}")
    return bad


def _symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every presentation it can draw
    and require both halves of the augmentation contract.

    * The ENGINE GRID must be IDENTICAL at all 16 presentations. This game has
      no in-engine rotation mechanic, so driving the same engine directions (via
      `screen_action`, which pre-inverts the adapter's own remap) must produce
      the same board whatever the frame is doing. A failure here would mean the
      mechanic is not symmetric -- which is the claim `_FLIP_GAMES` membership
      rests on.
    * The FRAMES must be exactly the transform of the unaugmented ones, which is
      the claim the ART has to satisfy.

    Both the PLANS and a seeded random walk are replayed: the walk reaches the
    boards a plan never visits (crates shoved into every corner, the body jammed
    against each wall) and it presses the unbound ACTION key as well.

    Seeds are drawn only until every (level, presentation) pair has been
    compared once, and each pair is driven once. Rotation and the two flips are
    per-(seed, level), so a handful of seeds covers all 16 -- and the pairs are
    what the contract is about, so re-driving a presentation a later seed
    repeats would only re-pay level 1's 224 ms press for nothing."""
    solver, game, expert, _solvable = _new()
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    def grid_key(g):
        return tuple(tuple(frozenset(cell) for cell in row)
                     for row in g._engine.grid)

    def drive(g, level, presses):
        g.set_level(level)
        frames = [np.asarray(g._current_frame)]
        grids = [grid_key(g)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            frames.append(np.asarray(fd.frame[-1] if fd.frame
                                     else g._current_frame))
            grids.append(grid_key(g))
        return frames, grids

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    ref, seen, done, bad = {}, set(), set(), 0
    n_levels = game.n_levels
    for seed in range(400):
        if len(done) == 16 * n_levels:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            if (level, k) in done:
                continue
            if k != (0, False, False) and (level, "plan") not in ref:
                continue            # keep the reference run first
            done.add((level, k))
            rng = random.Random(f"weird_dave:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(_MIRROR_PRESSES.get(level, walk_presses))]
            for tag, presses in (
                    ("plan", [screen_action(d, *k) for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, *k) for a in walk])):
                frames, grids = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"    seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    ref[key] = (frames, grids)
                    continue
                rframes, rgrids = ref[key]
                if rgrids != grids:
                    print(f"    seed {seed} L{level} {k}: {tag} engine grids "
                          f"differ from the unaugmented run")
                    bad += 1
                if any(not np.array_equal(transform(a, *k), b)
                       for a, b in zip(rframes, frames)):
                    print(f"    seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
        print(f"  {len(done)} (level, presentation) pairs compared "
              f"of {16 * n_levels}")
    bad += len(done) != 16 * n_levels
    return bad


def _speed(episodes: int = 3) -> int:
    """Wall-clock for a handful of episodes, into memory. The first pays for
    level 0's field (or reads it off `PLAN_CACHE`); the rest are re-renders."""
    solver = WeirdDaveSolver()
    t0 = time.time()
    solver._ensure(0)
    warm = time.time() - t0
    frames = 0
    for seed in range(episodes):
        t1 = time.time()
        ok, levels = solver.solve_episode(seed)
        n = sum(len(lv["observations"]) for lv in levels)
        frames += n
        print(f"  seed {seed}: ok={ok} {len(levels)} levels {n:4d} frames "
              f"{time.time() - t1:.2f}s")
    print(f"  startup {warm:.1f}s, {frames} frames over {episodes} episodes")
    return 0


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--recovery" in sys.argv:
        violations = _recovery()
        print(f"recovery: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--engine" in sys.argv:
        violations = _engine()
        print(f"engine cross-check: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--speed" in sys.argv:
        sys.exit(_speed())
    sys.exit(WeirdDaveSolver.main())
