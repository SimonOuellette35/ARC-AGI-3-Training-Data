"""Generate Phase-1 training data for the PuzzleScript game
ps:rock_paper_scissors_v0_90_eq_v1_alpha ("Rock, Paper, Scissors" by
chaotic_iak, v0.90 = v1.alpha).

The harness -- the rotation contract, the trajectory recorder, the plan cache
and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared
with the other ps: generators. This file is the game-specific part: a native
model of the mechanic, a macro A* over it, the optimal-action annotation, the
render audit and the differential fuzz that licenses planning natively at all.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_rock_paper_scissors",
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
augmented view, so replaying the recorded actions reproduces the recorded
frames exactly. Every expert step carries an ``optimal`` set (see
"Optimal-action sets").

The game
--------
A pushing puzzle whose pieces beat each other in a cycle. Win: **collect every
gem, then stand on the exit** (``No Gem`` and ``All Player on Exit``). The star
in most levels is a bonus collectible that no win condition reads.

Four piece classes, each with its own movement law -- the laws are the puzzle,
not the destruction:

* **Rock is heavy.** Only ``[> Player | Rock]`` exists, so a rock moves *only*
  when the player shoves it directly. A rock never pushes a second rock, and
  nothing else ever moves one. That makes rocks the immovable furniture of the
  game and every rock pushed into a pocket is furniture forever.
* **Scissors have a facing, and blades stop a push.** ``down [> Object |
  Scissors no ScissorsU] -> [> Object | > Scissors]``: anything moving down
  pushes any scissors *except* the one pointing back up at it. The two
  exceptions are hand-written -- ``down [> Scissors | ScissorsU]`` and
  ``down [> Gem | ScissorsU]`` -- so a scissors facing you can still be shoved
  by another scissors or by a gem, and by nothing else. Scissors chain into
  each other, so one shove can drive a whole row.
* **Paper is pushed by everything except a rock and the blade aimed at it**
  (``rigid down [> Object no Rock no ScissorsD | Paper]``). Papers chain along
  the push line.
* **Gems and stars are light**: ``[> Object | Gem] -> [> Object | > Gem]``
  moves them for any pusher at all. The player's own rules come first in the
  loop, so a player never pushes a gem -- it *collects* it
  (``[> Player | Gem] -> [> Player | ]``), and collects a star the same way.

Then the cycle, applied to whatever is about to collide (all of these run
before the movement is resolved, so a destroyed piece leaves its cell free for
the pusher to walk into on the same press):

* **Paper covers rock** -- both ways round. A moving rock shoved into a
  stationary paper is destroyed (``[> Rock | stationary Paper] -> [ | Paper]``);
  a moving paper driven into a stationary rock destroys the rock.
* **Scissors cut paper** -- a paper driven into the blade facing it dies, and a
  scissors driven blade-first into a paper kills that paper.
* **Rock crushes scissors** -- a scissors driven blade-first into a rock dies,
  and a rock shoved into the blade facing it kills the scissors.

The exit is guarded by ``ExitHolder``, an invisible object on the piece layer
that everything except the player collides with (``[> Player | ExitHolder] ->
[> Player | ]``, and ``late [Exit no Player] -> [Exit ExitHolder]`` puts it
back the moment the player steps off). So the exit cell is walkable for the
player and a wall for every piece -- which is, on its own, what makes six of the
eighteen shipped levels unwinnable; see below.

**ACTION5 is a level-skip cheat and is excluded from this corpus.** The game
ships ``[Action Player no SkipCheck] -> [Player SkipCheck]`` followed by
``[Action Player SkipCheck] -> [Player] win``: pressing ACTION twice in a row
wins any level outright in two presses (verified against the adapter). It is
the author's escape hatch, not a mechanic -- taking it would make the shortest
plan for all 18 levels "ACTION, ACTION" and the corpus worth nothing. The
expert therefore searches the four directions only, and `available_actions`
keeps ACTION5 out of the exploration prefix as well, so no recorded episode can
stumble into the skip. The plans here are shortest among **skip-free**
solutions; that is the claim, and it is the only useful one.

What the interpreter actually plays (paper is NOT sticky)
---------------------------------------------------------
The source ships a fifth paper rule inside the rigid group::

    + rigid [orthogonal Paper | Paper] -> [orthogonal Paper | orthogonal Paper]

which is the "PAPER is STICKY" the level 4 message announces: a clump of papers
should move as one body in any direction. **`PuzzleScriptAdapter` does not
implement it** -- ``orthogonal`` as a movement modifier on the RHS never
resolves to a direction, so the rule is inert and papers chain only along the
push line, exactly like crates in a plain sokoban. This is measured, not
assumed: the first `--selfcheck` run diverged on levels 3, 6 and 7 with the
model dragging laterally-adjacent papers the interpreter left standing, and
deleting the sticky propagation from `_Board` made all 18 levels agree.

The model reproduces the interpreter, because the interpreter is the game the
agent is handed by `game_envs`. It is worth knowing that this is not the game
the author wrote: the levels built on stickiness are easier here than intended,
and level 3 -- named "Think Before You Stick" -- is unwinnable either way (see
`skip_levels`).

The `rigid` grouping itself IS implemented and is kept: every force the four
directional paper rules touch shares one group id, so if any paper in a press
is blocked, every paper force in that press and the pusher's own are cancelled
together. With the sticky rule inert that only differs from a plain chain push
in rare multi-clump presses, but it is cheap to model and the fuzz covers it.

Six levels are unwinnable
-------------------------
Levels 3, 6, 13, 14, 15 and 17 cannot be won by any agent, so they are
`skip_levels` and the corpus is the other twelve.

Five of them are the same shape, and it is ExitHolder that causes it. The exit
sits in a pocket whose only entrance holds a piece, and **every way of clearing
that piece puts something else in its place**: a moving paper driven into a rock
destroys the rock and then *occupies its cell*, which is exactly the intended
solution to levels 6, 13 and 15 ("PAPER covers ROCK") and leaves the paper one
cell short of the exit, with nothing able to shove it the last step because
ExitHolder refuses every piece. Level 6 was driven by hand through the real
interpreter to confirm it rather than trusting the model: the paper eats the
rocks along row 1, stops at (1, 7), and no further press changes the board at
all.

`_Geom.frozen` mechanizes the argument -- a piece can leave its cell only while
it is moving, which needs a free cell ahead AND a pusher cell behind, and
neither may be a wall or the exit -- and `--proof` prints it. It settles levels
6, 13, 15 and 17 outright.

The other two need one extra step each, given here so they can be checked by
hand; the searches agree (nothing found at any weight rung, or by beams up to
width 20_000):

* **Level 3** ("Think Before You Stick"). The exit at (1, 9) is walled on three
  sides, so it can only be entered from (2, 9), which holds a paper. Pushing
  that paper up needs the exit free of ExitHolder, down needs the player already
  on the exit, and right needs the player standing on (2, 8). Left is the case
  `frozen` misses: it chains into (2, 8), whose own left neighbour is the wall
  at (2, 7), so the chain is refused -- and (2, 8) is itself immovable forever
  (its four pushes are a wall, a wall, a wall, and "stand on the wall at
  (2, 7)"). Only a scissors destroys paper, and this level ships none.
* **Level 14** ("A-maze-ing Paper"). The exit's region (rows 7-11, columns
  13-15) is sealed off by row 6, and its one entrance is (9, 12), which holds a
  ScissorsL. Pushing it left needs the player at (9, 13) -- already inside.
  Pushing it right is the one case the rules single out: a scissors facing back
  at the pusher moves only for another scissors or a gem
  (``right [> Scissors | ScissorsL]``, ``right [> Gem | ScissorsL]``), and the
  level's only other scissors is inside the region and it has no gems at all.
  Up and down are walls.

Expert solver
-------------
`_Board` is a native re-implementation of one press: assign forces to a
fixpoint, run the six interaction rules in source order, then resolve the push
chains with the rigid cancellation. It runs at **35_000-77_000 presses/s**
against the interpreter's **1-373**, which is the whole reason it exists: the
rigid paper rules make ``eng.step`` cost ~250 ms on level 3 and ~1 s on levels
14 and 16, so an engine-blackbox search cannot even replay a plan there, let
alone find one. (Measure `eng.step` throughput before choosing an
engine-blackbox expert for a ps: game -- the same lesson as EntrepotPhage
Demake, for a different rule shape.)

On top of it, an A* whose successors are ``walk to the push cell, then push``
macros, plus one ``walk to the exit`` macro because the win is *reaching*
somewhere and no push expresses that. Two things make it affordable:

* **Walks are inert, so they are not simulated.** No rule in this game fires on
  a bare player move onto an empty cell; the only thing a walk touches is
  ExitHolder, which is removed and re-grown within the same press and is a pure
  function of where the player stands. A macro therefore teleports the player
  to the stand cell and simulates the single push -- O(1) model presses per
  successor instead of O(walk length). ``--truth`` proves the teleport agrees
  with the full primitive replay on every step of every shipped plan.
* **Cost is counted in primitive MOVES and the dedup key is the exact ``(board,
  player cell)`` pair**, not the player's reachable region. Region dedup is the
  standard sokoban canonicalisation and it is exact when cost is *pushes*; with
  move cost it keeps whichever state arrived cheaper rather than whichever
  stands nearer the next push, and silently loses moves.

The heuristic is admissible: with ``k`` gems left it is ``d(player -> nearest
gem) + (k - 1) + min_gem d(gem -> exit)``, all over the wall-only distance field
(pieces relaxed away, which can only under-count since shifting one costs
moves); with no gems left it is ``d(player -> exit)``. Every gem needs its own
distinct entering move, which is where the ``k - 1`` is sound.

Levels that do not fall to ``weight = 1`` are retried up a weight ladder
``(1, 2, 3, 5, 10, 25)``, then by a BEAM (`_beam`), whose plan is handed back to
the exhaustive search as an INCUMBENT so the bound can prune it shorter.
`--plans` prints which stage each level used, and 8 of the 12 come out proved
shortest:

    level     0    1    2    4    5    7    8    9   10   11   12   16
    presses  25   38   45  114   17   10   28  162   62   32   50   32
    proved    y    y    y    .    y    y    y    .    y    y    .    y

615 presses over the twelve. Levels 4 and 9 are the beam's (they plateau A*
completely -- see `_beam`), 12 fell to weight 2. Every plan is inside the
adapter's 200-press episode cap, which `--plans` checks; the longest, level 9's
162, is what that column is there to watch.

One bug worth keeping in view, because it produced a plan that WON and was
still wrong: a macro's goal test fires after the whole ``walk + press``, so a
press the model refused reads as a win whenever the walk had already reached
the exit -- level 9's beam plan came out 162 presses long and won on 160.
`_walk_win` tests the walk itself, which is exact because a walk cannot change
the gem count.

Optimal-action sets
-------------------
Every expert step ships a set, never a bare single action:

* A **walk** step is labelled with every direction that keeps the player on a
  shortest route to that macro's stand cell (or to the exit, for the trailing
  walk). Walk order is genuinely free here -- a walk changes nothing but the
  player's own cell -- so labelling one arbitrary interleaving as the only right
  answer would train a coin flip the policy cannot win.
* A **push** step is labelled with itself. Which piece to shove where is the
  puzzle, and a sibling push is a different plan rather than a reordering of
  this one.

That is a SOUND set, not a complete one. ``--ties`` re-solves from every
labelled press at every step of every shipped plan and compares the exact
remaining cost with the plan's own: **zero labels anywhere cost more**. On the
nine levels whose plan is certified shortest, every label matches exactly, so
every one of them is on a shortest win path. On levels 4, 9 and 12 -- the beam's
and the weighted one -- some labels finish SOONER than the shipped plan does,
which is a statement about the plan rather than the label, and some re-solves do
not finish inside the check's budget and are reported unknown rather than
passed. The reverse is never claimed: an unlabelled press that ties is a sibling
plan, not a bug.

The one trap worth recording: `_Geom.walk_field` floods over cells the player
can WALK on, and the player's own cell holds `PLAYER`, which is not one of
them. Reading the field without blanking that cell first gives the opening step
of every walk run no distance at all, so its tie set silently collapses to the
single direction the expert happened to take -- a labelling bug that looks
exactly like "this game has no ties".

Rendering
---------
Audited per the usual palette-collision rule, and two fixes were needed in
`data/puzzlescript_games/Rock,_Paper,_Scissors_(v0.90_=_v1.alpha).txt`. Both are
colour/sprite only: the collision layers and every rule are untouched.

**Wall vs floor.** Wall was ``darkgreen green`` and Background is
``lightgreen yellow``. The ARC
palette has ONE green, so darkgreen, green and lightgreen all quantize to 14
and a wall cell rendered as a solid block of 14 while a floor cell rendered as
the same 14 with two yellow pixels in it -- the entire wall/floor distinction
was 2 pixels out of 25, and levels 14 and 16 are wide enough (17 and 23 cells)
that the frame is scaled to under 3 px per cell, where those two pixels do not
reliably survive. Wall is now ``purple pink`` (ARC 15 and 6, both unused by any
other object here), which also makes the level-number digits and the star
tracker along the bottom wall row -- ``darkgreen green``, previously invisible
14-on-14 -- readable.

**The player standing on the exit.** That is the WINNING frame, and on level 16
it was byte-identical to the player standing on bare floor. Level 16 is 23 cells
wide, so ``cell_px`` is 2 and `_render_cell_sprite`'s centred sampling reads a
5x5 sprite at rows and columns {1, 3} only -- four pixels, all of them opaque in
the player sprite, so nothing underneath ever showed. Its right shoulder pixel
(row 1, column 3) is now transparent, which is one of the four and lets the
exit's lightblue through. The obvious fix -- punching the sprite's CENTRE out --
does nothing here: (2, 2) is not sampled at ``cell_px = 2``.

Everything else survives quantization on its own: rock 13/12, paper 1 with a
gray 2 dog-ear and white 0 fold lines, the four scissors 3/2/8/13 (told apart
by their four distinct blade sprites), gem 9/10 and exit 9/10 (same two colours,
different shapes), star 13/8, player 5/12/8/9. ``--audit`` renders every
composition on every board size in use and asserts they are pairwise distinct.

Augmentation
------------
The engine state after reset is identical for every seed (the levels are fixed
ASCII maps), so the only per-(seed, level) variable is the presentation: the
adapter's frame rotation ``rotation_k in {0,1,2,3}`` with the matching
directional action remap (`screen_action`). 12 recorded levels x 4 rotations =
48 presentations. The plans are therefore seed-independent: searched once,
cached on disk under ``data/rock_paper_scissors_plans.json``, and replayed per
seed with that seed's remapped screen actions. **Run ``--plans`` once before
`parallelize_generator`** -- a cold cache costs a few minutes (levels 4 and 9
are ~45s each), and every shard would otherwise re-derive it.

No colour augmentation and no flips: the four scissors sprites encode a
DIRECTION that the rules read, and a flip that mirrored the art without
mirroring the rules would turn every scissors into a lie.

Self-checks
-----------
Six CLI modes, and they are the reason to trust any of the above:

* ``--selfcheck`` -- differential fuzz of `_Board` against the real interpreter,
  400 random presses on each of the 18 levels, comparing the whole piece grid
  and the win flag after every one. This is what found the sticky-paper
  divergence.
* ``--macros`` -- the MACRO successor set is complete: from each of 400 states
  per level, every board one press can produce from any cell the player could
  have walked to first is compared against what the macros produce. This is the
  regression test for the bug this generator was written around -- a prune that
  dropped every push whose target was a wall also dropped every GEM standing
  against one, which made levels 2 and 12 come out unwinnable *with the search
  reporting a cleanly closed frontier*, i.e. looking exactly like a proof.
  Fuzzing the model cannot catch a successor generator that loses moves; only
  comparing the two enumerations can.
* ``--truth`` -- every shipped plan replayed press by press through `_Board`
  wins, and wins on its last press. This is what licenses the teleport.
* ``--ties`` -- every labelled press re-solved and confirmed to cost no more
  than the plan it belongs to. Clean: 0 of 546 labels anywhere cost more.
* ``--proof`` -- the unwinnable-level argument, printed per level.
* ``--audit`` -- every cell composition distinct at every board size in use.

``--selfcheck`` takes ~25 minutes: it drives the *interpreter*, and the
interpreter is the slow thing. The rest are seconds to a few minutes.

Verified
--------
12/12 levels win on every seed; 4 seeds x 12 levels replay byte-exactly through
`games/ps:rock_paper_scissors_v0_90_eq_v1_alpha` to `GameState.WIN` (2507
actions, zero frame mismatches), all 2460 expert steps carry an optimal set, two
runs are `diff -rq` identical, ``--selfcheck`` agrees with the interpreter on
7200 presses across all 18 levels, ``--macros`` is complete on 7200 states, and
``--audit`` is clean on all eight board sizes.

Usage (run from the repo root):
    python solvers/generate_rock_paper_scissors_training.py --episodes 200 \\
        --out data/training_multi_level/rock_paper_scissors

    python solvers/generate_rock_paper_scissors_training.py --plans      # level report
    python solvers/generate_rock_paper_scissors_training.py --selfcheck  # fuzz vs interpreter
    python solvers/generate_rock_paper_scissors_training.py --macros     # successors complete
    python solvers/generate_rock_paper_scissors_training.py --truth      # teleport == replay
    python solvers/generate_rock_paper_scissors_training.py --ties       # labels are optimal
    python solvers/generate_rock_paper_scissors_training.py --proof      # unwinnable levels
    python solvers/generate_rock_paper_scissors_training.py --audit      # rendering
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

from adapters.puzzlescript_adapter import _render_frame            # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, Plan, PSExpert   # noqa: E402

GAME_NAME = "Rock,_Paper,_Scissors_(v0.90_=_v1.alpha)"
GAME_MODULE_ID = "ps:rock_paper_scissors_v0_90_eq_v1_alpha"

# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

#: Codes for the ONE piece-layer object a cell may hold. The whole mechanic
#: lives on a single collision layer (``object``), so a cell holds at most one
#: of these and the board is a flat list of codes -- which is what makes the
#: model as fast as it is.
EMPTY, WALL, PLAYER, ROCK, PAPER, SU, SL, SD, SR, GEM, STAR, HOLDER = range(12)

SCISSORS = frozenset((SU, SL, SD, SR))
#: Everything a ``walk to it and press`` macro may target.
PIECES = frozenset((ROCK, PAPER, SU, SL, SD, SR, GEM, STAR))
#: What the player may walk onto. ExitHolder is walkable for the player alone
#: -- ``[> Player | ExitHolder] -> [> Player | ]`` deletes it -- and a wall for
#: every piece, which no push macro needs to know because a push into it is
#: simply refused.
WALKABLE = frozenset((EMPTY, HOLDER))

DIRS = ("up", "down", "left", "right")
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}
#: The scissors whose blades point ALONG the press: it cuts what it runs into
#: instead of pushing it, and paper refuses to be pushed by it.
S_FWD = {"up": SU, "down": SD, "left": SL, "right": SR}
#: The scissors whose blades point BACK at the pusher: a plain push cannot
#: move it (only another scissors or a gem can), and a rock shoved into it
#: destroys it.
S_BACK = {"up": SD, "down": SU, "left": SR, "right": SL}

#: The object names the model reads out of an engine grid.
_NAME_TO_CODE = {
    "wall": WALL, "player": PLAYER, "rock": ROCK, "paper": PAPER,
    "scissorsu": SU, "scissorsl": SL, "scissorsd": SD, "scissorsr": SR,
    "gem": GEM, "star": STAR, "exitholder": HOLDER,
}
_CODE_TO_CHAR = {EMPTY: ".", WALL: "#", PLAYER: "o", ROCK: "r", PAPER: "p",
                 SU: "w", SL: "a", SD: "s", SR: "d", GEM: "g", STAR: "*",
                 HOLDER: "E"}

INF = 1 << 20


class _Board:
    """One press of Rock, Paper, Scissors, natively.

    ``g`` is a flat list of `EMPTY`..`HOLDER` codes; ``exit`` is the flat index
    of the exit cell (its ``Exit`` marker is on a layer below the pieces and
    never moves, so it is geometry rather than state -- what varies is whether
    the cell holds `HOLDER` or the player).

    `step` follows the source's four phases in order, because the order is
    load-bearing: the interaction rules run BEFORE the pushes resolve, so a
    piece destroyed this press leaves its cell free for the pusher to enter on
    the same press."""

    __slots__ = ("h", "w", "g", "exit")

    def __init__(self, h: int, w: int, cells, exit_cell: int | None):
        self.h, self.w = h, w
        self.g = list(cells)
        self.exit = exit_cell

    def copy(self) -> "_Board":
        b = _Board.__new__(_Board)
        b.h, b.w, b.g, b.exit = self.h, self.w, list(self.g), self.exit
        return b

    def player(self) -> int | None:
        try:
            return self.g.index(PLAYER)
        except ValueError:
            return None

    def won(self) -> bool:
        """``No Gem`` and ``All Player on Exit``."""
        return self.exit is not None and self.g[self.exit] == PLAYER \
            and GEM not in self.g

    def show(self) -> str:
        return "\n".join(
            "".join(_CODE_TO_CHAR[self.g[r * self.w + c]] for c in range(self.w))
            for r in range(self.h))

    def step(self, d: str) -> None:
        h, w, g = self.h, self.w, self.g
        dr, dc = DELTA[d]
        s_fwd, s_back = S_FWD[d], S_BACK[d]

        def ahead(i: int) -> int:
            r, c = divmod(i, w)
            nr, nc = r + dr, c + dc
            return nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1

        p = self.player()
        if p is None:
            return
        forces = {p}
        #: Forces the four rigid paper rules touched. They all share one group
        #: id in the interpreter, so one blocked paper cancels the lot.
        rigid: set[int] = set()

        # -- MOVEMENT: the startloop, run to a fixpoint ----------------------
        # Every force in a turn points the same way (the press direction): the
        # only rule that could produce another is the inert sticky one, so the
        # whole phase is "which cells get shoved", never "which way".
        changed = True
        while changed:
            changed = False
            for a in list(forces):
                b = ahead(a)
                if b < 0 or b in forces:
                    continue
                oa, ob = g[a], g[b]
                if ob == ROCK:
                    if oa == PLAYER:            # nothing else ever moves a rock
                        forces.add(b)
                        changed = True
                elif ob in SCISSORS:
                    if ob != s_back or oa in SCISSORS or oa == GEM:
                        forces.add(b)
                        changed = True
                elif ob == PAPER:
                    if oa != ROCK and oa != s_fwd:
                        forces.add(b)
                        rigid.add(a)
                        rigid.add(b)
                        changed = True
                elif ob == GEM or ob == STAR:
                    if oa == PLAYER:            # collected, not pushed
                        g[b] = EMPTY
                        changed = True
                    else:
                        forces.add(b)
                        changed = True

        # -- the exit swallows its holder, for the player only ----------------
        b = ahead(p)
        if b >= 0 and g[b] == HOLDER:
            g[b] = EMPTY

        # -- INTERACTION, in source order -------------------------------------
        # ``stationary`` means "carries no force", which is why these read
        # ``b not in forces``: a piece riding the same chain is not a collision.
        def kill_pusher(pred) -> None:
            for a in list(forces):
                b = ahead(a)
                if b < 0 or b in forces:
                    continue
                if pred(g[a], g[b]):
                    g[a] = EMPTY
                    forces.discard(a)
                    rigid.discard(a)

        def kill_target(pred) -> None:
            for a in list(forces):
                b = ahead(a)
                if b < 0 or b in forces:
                    continue
                if pred(g[a], g[b]):
                    g[b] = EMPTY

        kill_pusher(lambda oa, ob: oa == ROCK and ob == PAPER)     # paper covers
        kill_target(lambda oa, ob: oa == PAPER and ob == ROCK)     #   rock
        kill_pusher(lambda oa, ob: oa == PAPER and ob == s_back)   # scissors cut
        kill_target(lambda oa, ob: oa == s_fwd and ob == PAPER)    #   paper
        kill_pusher(lambda oa, ob: oa == s_fwd and ob == ROCK)     # rock crushes
        kill_target(lambda oa, ob: oa == ROCK and ob == s_back)    #   scissors

        # -- FORCE RESOLUTION --------------------------------------------------
        # One direction and one collision layer, so this is a plain chain walk:
        # a force moves iff the run of forced cells ahead of it ends on an empty
        # one. The loop re-runs after a rigid cancellation, because a chain that
        # was riding a paper is blocked once that paper's force is gone.
        while True:
            state: dict[int, bool] = {}

            def free(a: int) -> bool:
                got = state.get(a)
                if got is None:
                    b = ahead(a)
                    if b < 0:
                        got = False
                    elif g[b] == EMPTY:
                        got = True
                    elif b in forces:
                        state[a] = False            # defensive cycle break
                        got = free(b)
                    else:
                        got = False
                    state[a] = got
                return got

            blocked = {a for a in forces if not free(a)}
            if blocked & rigid and (rigid & forces):
                forces -= rigid
                continue
            break
        # Far end of the press first, so each cell is vacated before it is
        # claimed.
        for a in sorted(forces - blocked, reverse=(dr + dc) > 0):
            b = ahead(a)
            g[b] = g[a]
            g[a] = EMPTY

        # -- LATE: the exit re-grows its holder --------------------------------
        if self.exit is not None and g[self.exit] == EMPTY:
            g[self.exit] = HOLDER


def _read_board(game) -> _Board:
    """The engine's current grid, as a `_Board`."""
    eng, g = game._engine, game._game
    code = {g.obj_name_to_idx[n]: v for n, v in _NAME_TO_CODE.items()
            if n in g.obj_name_to_idx}
    exit_id = g.obj_name_to_idx["exit"]
    h, w = len(eng.grid), len(eng.grid[0])
    cells = [EMPTY] * (h * w)
    exit_cell = None
    for r in range(h):
        for c in range(w):
            for o in eng.grid[r][c]:
                if o in code:
                    cells[r * w + c] = code[o]
                elif o == exit_id:
                    exit_cell = r * w + c
    return _Board(h, w, cells, exit_cell)


# ---------------------------------------------------------------------------
# Per-level geometry (walls never change, so this is built once per board)
# ---------------------------------------------------------------------------

class _Geom:
    """Static, per-level: neighbours, the wall mask, the exit cell and the
    wall-only all-pairs distance field the heuristic reads.

    All-pairs rather than one field per query because the heuristic runs on
    every node and the biggest board here is 23x17 -- a few hundred BFS runs
    once, against a BFS per node forever."""

    __slots__ = ("h", "w", "n", "exit", "nbr", "wall", "dist", "back_idx",
                 "to_exit")

    def __init__(self, board: _Board):
        h, w = board.h, board.w
        self.h, self.w, self.n = h, w, h * w
        self.exit = board.exit
        self.nbr = []
        for i in range(h * w):
            r, c = divmod(i, w)
            row = []
            for d in DIRS:
                dr, dc = DELTA[d]
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self.nbr.append(tuple(row))
        #: index into `DIRS` of the direction OPPOSITE each direction, so the
        #: stand cell for a push is ``nbr[piece][back_idx[k]]``.
        self.back_idx = tuple(DIRS.index(OPP[d]) for d in DIRS)
        self.wall = tuple(v == WALL for v in board.g)
        self.dist: dict[int, dict[int, int]] = {}
        for s in range(h * w):
            if self.wall[s]:
                continue
            seen = {s: 0}
            queue = deque([s])
            while queue:
                x = queue.popleft()
                for y in self.nbr[x]:
                    if y >= 0 and not self.wall[y] and y not in seen:
                        seen[y] = seen[x] + 1
                        queue.append(y)
            self.dist[s] = seen
        #: ``cell -> wall-only distance to the exit``, read once per gem per
        #: node; the heuristic must not re-derive static facts.
        exit_field = self.dist.get(self.exit, {}) if self.exit is not None else {}
        self.to_exit = tuple(exit_field.get(i, INF) for i in range(h * w))

    # -- the pieces of the search that only need geometry --------------------
    def heuristic(self, g, player: int) -> int:
        """Admissible lower bound on the presses left, in primitive moves.

        Every remaining gem needs its own move to be entered, and after the last
        one the player still has to reach the exit -- so
        ``d(player -> nearest gem) + (k - 1) + min_gem d(gem -> exit)``. All
        distances are over walls only: relaxing the pieces away can only
        under-count, since shifting one out of the way costs moves of its own."""
        here = self.dist[player]
        to_exit = self.to_exit
        near = back = INF
        count = 0
        for i, v in enumerate(g):
            if v != GEM:
                continue
            count += 1
            d = here.get(i)
            if d is not None and d < near:
                near = d
            if to_exit[i] < back:
                back = to_exit[i]
        if not count:
            return to_exit[player]
        if near >= INF:
            return INF
        return near + count - 1 + back

    def walk_tree(self, g, player: int) -> dict:
        """``{cell: (previous cell, direction) | None}`` over the cells the
        player can walk to right now. A walk changes nothing, so this map is
        the whole of a macro's prefix."""
        parent = {player: None}
        queue = deque([player])
        while queue:
            x = queue.popleft()
            for k, d in enumerate(DIRS):
                y = self.nbr[x][k]
                if y >= 0 and g[y] in WALKABLE and y not in parent:
                    parent[y] = (x, d)
                    queue.append(y)
        return parent

    @staticmethod
    def walk_to(parent: dict, cell: int) -> list:
        out = []
        while parent[cell] is not None:
            cell, d = parent[cell]
            out.append(d)
        out.reverse()
        return out

    def walk_field(self, g, target: int) -> dict:
        """BFS distance to ``target`` over the player's current walkable map --
        what the walk steps' optimal sets are read off."""
        dist = {target: 0}
        queue = deque([target])
        while queue:
            x = queue.popleft()
            for y in self.nbr[x]:
                if y >= 0 and g[y] in WALKABLE and y not in dist:
                    dist[y] = dist[x] + 1
                    queue.append(y)
        return dist

    def frozen(self, g) -> set:
        """Cells holding a piece that can NEVER be vacated, hence cells the
        player can never stand on.

        Two facts drive it, and both are properties of the rules rather than of
        a position:

          * **No piece can ever enter the exit cell.** ExitHolder is on the
            piece layer and only the player deletes it, so a push aimed at the
            exit is always refused. (Verified directly against the interpreter:
            level 6's paper stops one cell short of the exit and no further
            press moves anything.)
          * **A piece leaves its cell only while it is MOVING.** Every rule that
            empties a cell -- an ordinary push, and the three "the mover dies"
            interactions (rock into paper, paper into the blade facing it,
            scissors blade-first into rock) -- needs a force on that cell, which
            needs a pusher standing directly behind it.

        So a piece at ``x`` is stuck forever unless some axis ``d`` has BOTH a
        cell to move into (``x + d`` neither wall nor exit) AND a cell to be
        pushed from (``x - d`` neither wall nor exit). That test is static: no
        rule creates or destroys a wall, and the exit never moves."""
        out = set()
        for x, v in enumerate(g):
            if v == WALL or v == EMPTY or x == self.exit:
                continue
            if v == GEM or v == STAR:
                continue                      # collected in place, never pushed
            movable = False
            for k in range(4):
                fwd, back = self.nbr[x][k], self.nbr[x][self.back_idx[k]]
                if (fwd >= 0 and not self.wall[fwd] and fwd != self.exit
                        and back >= 0 and not self.wall[back]
                        and back != self.exit):
                    movable = True
                    break
            if not movable:
                out.add(x)
        return out

    def reachable(self, g, player: int) -> set:
        """Over-approximation of every cell the player could EVER stand on:
        flood from the start over cells that are not walls and not `frozen`.

        Over-approximate on purpose -- it is used to prove a level UNWINNABLE
        (the exit is outside it), so it may only ever be too generous."""
        stuck = self.frozen(g)
        seen = {player}
        queue = deque([player])
        while queue:
            x = queue.popleft()
            for y in self.nbr[x]:
                if y >= 0 and not self.wall[y] and y not in stuck and y not in seen:
                    seen.add(y)
                    queue.append(y)
        return seen

    def macros(self, g, player: int):
        """``(walk tree, [(stand cell, push direction | None)])``.

        One macro per (piece, direction) whose stand cell the player can reach,
        plus the bare walk to the exit -- the win is *reaching* somewhere and no
        push expresses that.

        The one prune is that a piece with a WALL behind it cannot be shoved
        into it: the chain is refused, and no interaction rule reads a wall, so
        the press is a guaranteed no-op. **Gems and stars are exempt**, and the
        exemption is the whole point -- the player does not push them, it
        COLLECTS them (``[> Player | Gem] -> [> Player | ]``), which fires no
        matter what is on the far side. Pruning them cost every gem standing
        against a wall, which is most of them: it silently made levels 2 and 12
        unwinnable while the search still reported its frontier cleanly closed.
        ``--macros`` is the regression test."""
        parent = self.walk_tree(g, player)
        out = []
        for i, v in enumerate(g):
            if v not in PIECES:
                continue
            collectible = v == GEM or v == STAR
            for k, d in enumerate(DIRS):
                tgt = self.nbr[i][k]
                if not collectible and tgt >= 0 and self.wall[tgt]:
                    continue
                stand = self.nbr[i][self.back_idx[k]]
                if stand >= 0 and stand in parent:
                    out.append((stand, d))
        if self.exit is not None and self.exit in parent and self.exit != player:
            out.append((self.exit, None))
        return parent, out

    def apply(self, g, player: int, stand: int, push: str | None):
        """Run one macro: teleport the player to ``stand``, then press.

        States are ``bytes`` rather than tuples: a 391-cell board costs ~420
        bytes against ~3.2 kB as a tuple of ints, and the closed set holds
        hundreds of thousands of them (the same lesson as EntrepotPhage
        Demake's crate bitmasks -- a search dies of memory long before it dies
        of time)."""
        ng = bytearray(g)
        ng[player] = HOLDER if player == self.exit else EMPTY
        ng[stand] = PLAYER
        if push is None:
            return bytes(ng), stand
        board = _Board(self.h, self.w, ng, self.exit)
        board.step(push)
        return bytes(board.g), board.player()


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

#: Tried in order; the first rung that returns a plan wins. A rung above 1 is a
#: greedy search: the plan is still an engine-verified win, just not certified
#: shortest.
WEIGHTS = (1, 2, 3, 5, 10, 25)

#: Beam widths tried after the weight ladder gives up, first that returns.
BEAM_WIDTHS = (200, 1000, 4000)


class RPSExpert(PSExpert):
    """Macro A* over `_Board`, memoized per level and cached on disk.

    Conforms to `PSExpert` so `record_level` can drive it unchanged: `plan`
    takes the live engine, reads it into a `_Board`, searches natively and
    returns a flat list of primitive engine directions (a `Plan` carrying the
    per-step optimal sets)."""

    #: ACTION5 is the level-skip cheat -- see the module docstring. The search
    #: never offers it and `RPSSolver.available_actions` keeps it out of the
    #: exploration prefix, so it cannot reach the corpus by either route.
    directions = ["up", "down", "left", "right"]

    #: `_key` is the piece board, which is canonical within a level only.
    scope_by_level = True

    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "rock_paper_scissors_plans.json")

    #: Generated macros per level search, per weight rung.
    macro_cap = 400_000
    #: Wall-clock ceiling per rung, so a hopeless rung falls through to the next
    #: one instead of pinning the run.
    rung_seconds = 90.0
    #: Macro layers a beam explores, and its own wall-clock ceiling.
    beam_depth = 60
    beam_seconds = 240.0

    def setup(self) -> None:
        self._geom: dict[int, _Geom] = {}
        self._level: int | None = None
        #: Why the last search stopped. Starts as "cached" because a plan (or a
        #: cached None) served off disk never runs a search at all.
        self.last_status = "cached"
        #: ``level -> (weight rung, was it PROVED shortest)``, for `--plans`.
        #: Rung 1 alone does not mean shortest: a rung-1 search that ran out of
        #: budget with a win in hand returns that win, which is a valid plan and
        #: an unproven one.
        self.rung: dict[int, tuple[int, bool]] = {}

    # -- state key ------------------------------------------------------------
    def _key(self, eng) -> frozenset:
        """Every non-empty piece cell. Walls are static per level (hence
        ``scope_by_level``) but are cheap to carry and make the on-disk cache's
        start signature self-describing, so they stay in."""
        board = _read_board(self.game)
        w = board.w
        return frozenset((i // w, i % w, v)
                         for i, v in enumerate(board.g) if v != EMPTY)

    def geom(self, board: _Board, level: int | None) -> _Geom:
        got = self._geom.get(level)
        if got is None or got.n != board.h * board.w:
            got = self._geom[level] = _Geom(board)
        return got

    # -- planning -------------------------------------------------------------
    def plan(self, eng, level: int | None = None):
        self._level = level
        return super().plan(eng, level)

    def _search(self, eng) -> list | None:
        board = _read_board(self.game)
        geo = self.geom(board, self._level)
        if geo.exit is None or geo.exit not in geo.reachable(bytes(board.g),
                                                             board.player()):
            # The exit is walled off behind a cell nothing can ever vacate, so
            # the level cannot be won -- see `_Geom.frozen`. Answering here
            # rather than after six A* rungs and three beams is worth it: six of
            # the eighteen shipped levels are in this shape.
            self.last_status = "exit unreachable (see --proof)"
            return None
        for weight in WEIGHTS:
            found = self._astar_macro(board, geo, weight, want_proof=True)
            if found is not None:
                steps, optsets, proved = found
                self.rung[self._level] = (weight, proved)
                return Plan(steps, optsets)
        for width in BEAM_WIDTHS:
            found = self._beam(board, geo, width)
            if found is None:
                continue
            # Hand the beam's plan back to A* as an INCUMBENT. A beam plan
            # wanders (level 4 came out at 114 presses), and an upper bound is
            # exactly what the exhaustive search was missing: every node with
            # ``f >= that cost`` is now pruned, which on these boards is most of
            # them. It either shortens the plan or closes the frontier and
            # proves the beam's plan shortest.
            tight = self._astar_macro(board, geo, 1, want_proof=True,
                                      incumbent=found)
            if tight is not None:
                steps, optsets, proved = tight
                self.rung[self._level] = (
                    "beam, then exhaustive" if proved else f"beam {width}, tightened",
                    proved)
                return Plan(steps, optsets)
            self.rung[self._level] = (f"beam {width}", False)
            steps, optsets = self._flatten(found, board, geo)
            return Plan(steps, optsets)
        return None

    def _beam(self, board: _Board, geo: _Geom, width: int):
        """Breadth-first BEAM over the same macros, ``width`` states per depth.

        WHY, and what A* runs out of. Both searches pay the same thing per
        node, and A* spends its budget wherever the heuristic points -- which is
        the right call only while the heuristic can tell good states from bad.
        Here it frequently cannot: the estimate is "reach the gems, then the
        exit", and the middle of a solution on levels 4, 6 and 9 is a dozen
        macros that shove scissors out of a corridor without the player getting
        one step nearer anything. Over that stretch the heuristic is FLAT,
        weighting it changes nothing, and A* degenerates to uniform-cost search
        at a depth where that is hopeless. A beam spends the same budget on
        breadth at every depth and uses the heuristic only to decide who
        survives, which is all a flat heuristic is good for.

        The trade is optimality: a beam plan is engine-verified and winning but
        wanders where A* would not. It is the last rung for exactly that
        reason."""
        g0, p0 = bytes(board.g), board.player()
        if p0 is None:
            return None
        started = time.time()
        frontier = [(g0, p0, ())]
        seen = {g0}
        for _depth in range(self.beam_depth):
            kids = []
            for g, p, path in frontier:
                parent, succ = geo.macros(g, p)
                for stand, push in succ:
                    walk = geo.walk_to(parent, stand)
                    cut = self._walk_win(geo, g, p, walk)
                    if cut is not None:
                        return path + ((geo.exit, None, tuple(walk[:cut])),)
                    ng, np_ = geo.apply(g, p, stand, push)
                    here = path + ((stand, push, tuple(walk)),)
                    if np_ == geo.exit and GEM not in ng:
                        return here
                    if ng in seen:
                        continue
                    seen.add(ng)
                    # Rank on the heuristic ALONE: the rest of the tuple holds
                    # bytes and a path, which have no useful ordering.
                    kids.append((geo.heuristic(ng, np_), ng, np_, here))
            if not kids or time.time() - started > self.beam_seconds:
                return None
            kids.sort(key=lambda kid: kid[0])
            frontier = [(g, p, path) for _h, g, p, path in kids[:width]]
        return None

    @staticmethod
    def _walk_win(geo: _Geom, g, player: int, walk) -> int | None:
        """How many steps of ``walk`` are needed before it has already WON, or
        None if it never does.

        A walk changes nothing, so the gem count is whatever it was when the
        walk started: the win is simply "no gems left, and this walk sets foot
        on the exit". Checking it is not an optimisation, it is a correctness
        fix. The macro's win test fires after the whole ``walk + press``, so a
        press that the model REFUSED still reads as a win whenever the walk had
        already arrived -- and the plan ships with a dead press on the end.
        (Level 9's beam plan did exactly that: 163 presses, won at 161.)"""
        if GEM in g:
            return None
        cur = player
        for i, d in enumerate(walk):
            cur = geo.nbr[cur][DIRS.index(d)]
            if cur == geo.exit:
                return i + 1
        return None

    @staticmethod
    def _cost(macros) -> int:
        """Primitive presses in a macro list."""
        return sum(len(walk) + (0 if push is None else 1)
                   for _stand, push, walk in macros)

    def _astar_macro(self, board: _Board, geo: _Geom, weight: int,
                     want_proof: bool = False, incumbent=None):
        """A* over walk-and-push macros. Returns ``(presses, optimal sets)`` --
        plus a "was it proved shortest" flag when ``want_proof`` -- or None.

        ``g`` counts primitive MOVES and the dedup key is the exact
        ``(board, player cell)`` pair -- see the module docstring for why the
        usual region canonicalisation is unsound here. At ``weight == 1`` a win
        is kept as an incumbent and the search runs until the frontier's ``f``
        reaches its cost, which proves it shortest under the heuristic; above 1
        the first win is returned."""
        g0, p0 = bytes(board.g), board.player()
        if p0 is None:
            return None
        if geo.exit is not None and p0 == geo.exit and GEM not in g0:
            return ([], [], True) if want_proof else ([], [])
        exact = weight == 1
        started = time.time()
        counter = nodes = 0
        pq = [(weight * geo.heuristic(g0, p0), 0, 0, g0, p0, ())]
        best_g = {(g0, p0): 0}
        best: tuple | None = incumbent
        best_cost = INF if incumbent is None else self._cost(incumbent)
        proved = False
        #: "closed" means the frontier emptied inside the budget: every state
        #: reachable from the start was expanded, so a level with no plan is
        #: provably unwinnable rather than merely unsolved. `--plans` prints it.
        self.last_status = "budget"

        def out(macros):
            flat = self._flatten(macros, board, geo)
            return (flat[0], flat[1], proved) if want_proof else flat

        while pq:
            f, cost, _c, g, p, path = heapq.heappop(pq)
            if best is not None and f >= best_cost:
                proved = True
                break
            if best_g.get((g, p), INF) < cost:
                continue
            parent, succ = geo.macros(g, p)
            for stand, push in succ:
                nodes += 1
                walk = geo.walk_to(parent, stand)
                cut = self._walk_win(geo, g, p, walk)
                if cut is not None:
                    here = path + ((geo.exit, None, tuple(walk[:cut])),)
                    total = cost + cut
                    if not exact:
                        return out(here)
                    if total < best_cost:
                        best_cost, best = total, here
                    continue
                ng, np_ = geo.apply(g, p, stand, push)
                here = path + ((stand, push, tuple(walk)),)
                total = cost + len(walk) + (0 if push is None else 1)
                if np_ == geo.exit and GEM not in ng:
                    if not exact:
                        return out(here)
                    if total < best_cost:
                        best_cost, best = total, here
                    continue
                if ng == g and np_ == p:
                    continue                    # a press the model refused
                key = (ng, np_)
                if best_g.get(key, INF) <= total:
                    continue
                best_g[key] = total
                counter += 1
                heapq.heappush(pq, (total + weight * geo.heuristic(ng, np_),
                                    total, counter, ng, np_, here))
            if nodes >= self.macro_cap or (
                    time.time() - started > self.rung_seconds):
                break
        else:
            proved = exact          # the frontier closed: nothing shorter exists
            self.last_status = "closed"
        return out(best) if best is not None else None

    def _flatten(self, macros, board: _Board, geo: _Geom):
        """``[(stand, push, walk)]`` -> ``(flat presses, per-step optimal sets)``,
        replayed from ``board``.

        The sets come straight out of the macro structure rather than being
        re-derived by watching the board: each walk run is measured against the
        stand cell it ends on (or the exit, for a trailing bare walk), and the
        push that closes a macro is labelled with itself. See the module
        docstring for why that is sound but not complete.

        The start board is a parameter rather than "whatever the engine is
        showing": `_ties` re-solves from states in the MIDDLE of a plan, and an
        implicit start would have flattened those against the level's opening
        position."""
        presses: list[str] = []
        optsets: list[list[str]] = []
        g, p = bytes(board.g), board.player()
        for stand, push, walk in macros:
            # The field is built with the player's OWN cell blanked: it holds
            # `PLAYER`, which `walk_field` reads as unwalkable, so leaving it in
            # gives the run's first step no distance and silently collapses its
            # tie set to the one direction the expert happened to pick.
            free = bytearray(g)
            free[p] = HOLDER if p == geo.exit else EMPTY
            dist = geo.walk_field(bytes(free), stand)
            cur = p
            for d in walk:
                here = dist.get(cur)
                alts = [] if here is None else [
                    e for k, e in enumerate(DIRS)
                    if geo.nbr[cur][k] >= 0 and dist.get(geo.nbr[cur][k]) == here - 1]
                # The recorded step is on a shortest route by construction; an
                # empty or disagreeing set means the reconstruction drifted, so
                # fall back to what the expert actually did.
                presses.append(d)
                optsets.append(alts if d in alts else [d])
                cur = geo.nbr[cur][DIRS.index(d)]
            if push is not None:
                presses.append(push)
                optsets.append([push])
            g, p = geo.apply(g, p, stand, push)
        return presses, optsets


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class RPSSolver(PSAStarSolver):
    game_id = "puzzlescript_rock_paper_scissors"
    game_name = GAME_NAME
    game_module_id = GAME_MODULE_ID
    expert_cls = RPSExpert

    #: Six of the eighteen shipped levels cannot be won by ANY agent -- see the
    #: module docstring's "Six levels are unwinnable". Four of them (6, 13, 15,
    #: 17) are proved so mechanically by `_Geom.frozen`; 3 and 14 rest on a
    #: hand argument that is spelled out there, corroborated by every rung of
    #: the ladder and beams up to width 20_000 finding nothing.
    #:
    #: Skipping them up front matters: `discover_solvable` would otherwise burn
    #: the full search budget on each of them at every cold start.
    skip_levels = frozenset({3, 6, 13, 14, 15, 17})

    #: Plans are long (the boards are large and gems are collected one at a
    #: time); the adapter cuts an episode off at 200 presses, which `--plans`
    #: checks every level against.
    max_steps = 300
    #: The expert owns its own cap/weight ladder; the base's dials are unread.
    node_cap = 400_000
    weight = 1

    def available_actions(self, game) -> list[int]:
        """The four directions. ACTION5 is the level-skip cheat -- two presses
        win any level -- so it is kept out of the exploration prefix as well as
        out of the search; otherwise a prefix that happened to press it twice
        would tape a two-press "solution" and the RESET that ends the prefix
        would never happen (`BaseSolver._reset_prefix` stops on a win)."""
        return [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# CLI self-checks
# ---------------------------------------------------------------------------

def _levels(game):
    return [lvl for lvl in range(game.n_levels)]


def _report() -> int:
    """Per-level report: the plan, the rung it came from, and the press budget."""
    solver = RPSSolver()
    game = solver.make_game(0)
    expert = RPSExpert(game)
    total = 0
    for level in _levels(game):
        game.set_level(level)
        started = time.time()
        found = expert.plan(game._engine, level)
        elapsed = time.time() - started
        if found is None:
            skipped = " [in skip_levels]" if level in RPSSolver.skip_levels else ""
            print(f"level {level:2d}: NO PLAN, search {expert.last_status} "
                  f"({elapsed:.1f}s){skipped}")
            continue
        rung = expert.rung.get(level)
        ties = sum(1 for s in getattr(found, "optsets", []) if len(s) > 1)
        total += len(found)
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        note = ("cached" if rung is None else
                "PROVED shortest" if rung == (1, True) else
                f"{rung[0]}, not certified shortest")
        print(f"level {level:2d}: {len(found):3d} presses "
              f"(budget {game._max_steps}, {room}), {note}, "
              f"{ties:3d} steps with a tie set "
              f"({ties / max(1, len(found)):.0%}), {elapsed:.1f}s")
    print(f"total {total} presses")
    return 0


def _selfcheck(presses: int = 400, seed: int = 0) -> int:
    """Differential fuzz: random presses played move-for-move against the real
    interpreter on every shipped level, comparing the whole piece grid and the
    win flag after each one.

    This is the only thing standing between `_Board` and a silently wrong
    corpus. It is also how the sticky-paper finding was made -- see the module
    docstring."""
    solver = RPSSolver()
    game = solver.make_game(0)
    eng = game._engine
    bad = 0
    for level in _levels(game):
        rng = random.Random(seed * 1000 + level)
        game.set_level(level)
        board = _read_board(game)
        for t in range(presses):
            d = rng.choice(DIRS)
            before = board.show()
            eng.step(d)
            board.step(d)
            truth = _read_board(game)
            if truth.g != board.g:
                bad += 1
                print(f"level {level} press {t} dir={d} DIVERGE\n"
                      f"before:\n{before}\nengine:\n{truth.show()}\n"
                      f"model:\n{board.show()}")
                break
            if eng.check_win() != board.won():
                bad += 1
                print(f"level {level} press {t}: win {eng.check_win()} "
                      f"vs model {board.won()}")
                break
        else:
            print(f"level {level:2d}: {presses} presses agree")
    print("selfcheck clean" if not bad else f"SELFCHECK FAILED: {bad} levels")
    return 0 if not bad else 1


def _truth() -> int:
    """Prove the macro teleport agrees with the full primitive replay.

    The search never simulates a walk -- it drops the player on the stand cell
    and presses -- which is exact only while walking is inert. This replays
    every shipped plan press by press through `_Board` and checks it reaches the
    same win, i.e. that the plan the teleporting search built is a real one."""
    solver = RPSSolver()
    game = solver.make_game(0)
    expert = RPSExpert(game)
    bad = 0
    for level in _levels(game):
        if level in RPSSolver.skip_levels:
            continue
        game.set_level(level)
        found = expert.plan(game._engine, level)
        if found is None:
            print(f"level {level:2d}: no plan")
            continue
        board = _read_board(game)
        won_at = None
        for i, d in enumerate(found):
            board.step(d)
            if board.won():
                won_at = i
                break
        ok = won_at == len(found) - 1
        bad += not ok
        print(f"level {level:2d}: primitive replay "
              f"{'wins on the last press' if ok else f'WRONG (won at {won_at} of {len(found)})'}")
    print("truth clean" if not bad else f"TRUTH FAILED: {bad} levels")
    return 0 if not bad else 1


def _ties(seconds: float = 4.0) -> int:
    """Check every LABELLED press costs no more than the plan it belongs to.

    At each step of each shipped plan, re-solve from the state each labelled
    press leads to and compare ``1 + that distance`` with the presses the plan
    has left. Equal means the press is on a shortest continuation; a level whose
    own plan is PROVED shortest therefore has every label verified optimal
    outright.

    The reverse is deliberately not claimed: an unlabelled press that ties is a
    SIBLING PLAN, not a bug (see the module docstring).

    Each re-solve gets ``seconds`` of exhaustive search; a step whose re-solve
    does not finish is reported UNKNOWN rather than passed. Those are the deep
    levels, where an exact distance is the very thing the expert could not
    compute in the first place."""
    solver = RPSSolver()
    game = solver.make_game(0)
    expert = RPSExpert(game)
    expert.rung_seconds = seconds
    bad = 0
    for level in _levels(game):
        game.set_level(level)
        found = expert.plan(game._engine, level)
        if found is None:
            continue
        sets = getattr(found, "optsets", None)
        if sets is None:
            print(f"level {level:2d}: NO OPTIMAL SETS")
            bad += 1
            continue
        board = _read_board(game)
        geo = expert.geom(board, level)
        tight = short = worse = unknown = 0
        for i, taken in enumerate(found):
            remaining = len(found) - i
            for d in sets[i]:
                probe = board.copy()
                probe.step(d)
                if probe.won():
                    got = 1
                else:
                    sub = expert._astar_macro(probe, geo, 1)
                    got = None if sub is None else 1 + len(sub[0])
                if got is None:
                    unknown += 1
                elif got == remaining:
                    tight += 1
                elif got < remaining:
                    # The label finishes SOONER than the shipped plan does. That
                    # is a statement about the plan, not the label: it only ever
                    # happens on the levels `--plans` already reports as not
                    # certified shortest, where a mid-plan state has a better
                    # continuation than the one the beam took.
                    short += 1
                else:
                    worse += 1
            board.step(taken)
        bad += worse > 0
        print(f"level {level:2d}: {tight} labelled presses match the plan's own "
              f"remaining cost, {short} beat it, {unknown} unknown (budget), "
              f"{worse} WORSE")
    print("ties clean" if not bad else f"TIES FAILED: {bad} levels")
    return 0 if not bad else 1


def _proof() -> int:
    """Per level, report whether the exit is walled off for good.

    `_Geom.frozen` is the argument; this prints the cells it names so the claim
    can be checked by eye against the map. A level reported UNWINNABLE here is
    unwinnable for any agent, not merely unsolved by this expert."""
    solver = RPSSolver()
    game = solver.make_game(0)
    expert = RPSExpert(game)
    doomed = []
    for level in _levels(game):
        game.set_level(level)
        board = _read_board(game)
        geo = expert.geom(board, level)
        g = bytes(board.g)
        stuck = geo.frozen(g)
        reach = geo.reachable(g, board.player())
        ok = geo.exit is not None and geo.exit in reach
        gate = ([geo.nbr[geo.exit][k] for k in range(4)]
                if geo.exit is not None else [])
        gate = [x for x in gate if x >= 0 and not geo.wall[x]]
        if not ok:
            doomed.append(level)
        print(f"level {level:2d}: exit {divmod(geo.exit, geo.w)} "
              f"reached from {[divmod(x, geo.w) for x in gate]}, "
              f"{len(stuck)} frozen cells, "
              f"{'winnable so far as this test can tell' if ok else 'UNWINNABLE: every route into the exit is frozen'}")
    print(f"provably unwinnable: {doomed}")
    print(f"skip_levels: {sorted(RPSSolver.skip_levels)}")
    return 0


def _macros(states_per_level: int = 400, seed: int = 0) -> int:
    """Prove the MACRO successor set is complete, on every shipped level.

    The search never presses a direction: it enumerates ``walk to a cell, then
    press`` macros and teleports the player through the walk. That is only
    sound if, from any state, the set of boards reachable by one press from
    ANY cell the player can walk to is exactly the set the macros produce.
    This check builds that set both ways and compares them.

    It is the regression test for the bug this generator was written around: a
    prune that dropped every push whose target cell was a wall also dropped
    every GEM standing against one, so the two densest levels came out
    unwinnable -- and came out with the search reporting a cleanly closed
    frontier, which reads exactly like a real proof of unwinnability. A
    successor generator that silently loses moves cannot be caught by fuzzing
    the model, only by comparing the two enumerations."""
    solver = RPSSolver()
    game = solver.make_game(0)
    expert = RPSExpert(game)
    rng = random.Random(seed)
    bad = 0
    for level in _levels(game):
        game.set_level(level)
        board = _read_board(game)
        geo = expert.geom(board, level)
        # A press that moved only the player is a WALK, and walks are what the
        # macros abstract over rather than enumerate -- so successors are
        # compared on the PIECE layout, with the player and the exit cell
        # blanked out (stepping on and off the exit swaps PLAYER for
        # ExitHolder there and is not a piece move).
        def layout(bs: bytes) -> bytes:
            out = bytearray(bs)
            for i, v in enumerate(out):
                if v == PLAYER:
                    out[i] = EMPTY
            if geo.exit is not None:
                out[geo.exit] = EMPTY
            return bytes(out)

        state = (bytes(board.g), board.player())
        missing = extra = checked = 0
        for _ in range(states_per_level):
            g, p = state
            base = layout(g)
            # Every board one press can produce, from every cell the player
            # could have walked to first.
            truth = set()
            for cell in geo.walk_tree(g, p):
                seat = bytearray(g)
                seat[p] = HOLDER if p == geo.exit else EMPTY
                seat[cell] = PLAYER
                for d in DIRS:
                    probe = _Board(geo.h, geo.w, seat, geo.exit)
                    probe.step(d)
                    out = bytes(probe.g)
                    if layout(out) != base:
                        truth.add(out)
            got = {geo.apply(g, p, stand, push)[0]
                   for stand, push in geo.macros(g, p)[1]}
            got = {x for x in got if layout(x) != base}
            missing += len(truth - got)
            extra += len(got - truth)
            checked += 1
            nxt = [x for x in geo.macros(g, p)[1]]
            if not nxt:
                break
            stand, push = rng.choice(nxt)
            state = geo.apply(g, p, stand, push)
        ok = not missing
        bad += not ok
        print(f"level {level:2d}: {checked} states, {missing} macro successors "
              f"missing, {extra} not reachable by a single press "
              f"{'OK' if ok else '*** INCOMPLETE ***'}")
    print("macros clean" if not bad else f"MACROS FAILED: {bad} levels")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct on every board size in use.

    Composition, not object: the player standing on the exit is the winning
    frame and has to differ from the player on bare floor, and the four scissors
    have to differ from each other or their facing -- which is what decides
    whether a push is legal -- is invisible.

    The whole board is filled with one composition and whole frames are
    compared, rather than cropping a cell: `_render_frame` rescales a sub-64
    render to fill the frame, so cell-size arithmetic lands in the wrong cell on
    boards whose dimensions do not divide 64."""
    solver = RPSSolver()
    game = solver.make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {
        "floor": (), "wall": ("wall",), "exit": ("exit", "exitholder"),
        "player": ("player",), "player_on_exit": ("exit", "player"),
    }
    for name in ("rock", "paper", "scissorsu", "scissorsl", "scissorsd",
                 "scissorsr", "gem", "star"):
        comps[name] = (name,)

    sizes: dict[tuple[int, int], list[int]] = {}
    for level in _levels(game):
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
            shots[name] = np.asarray(_render_frame(eng, g))
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (levels {','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--selfcheck" in sys.argv:
        sys.exit(_selfcheck())
    if "--truth" in sys.argv:
        sys.exit(_truth())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--macros" in sys.argv:
        sys.exit(_macros())
    if "--proof" in sys.argv:
        sys.exit(_proof())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(RPSSolver.main())
