"""Generate Phase-1 training data for the PuzzleScript game ps:towers_of_saigon
("Towers of Saigon", Theta Games).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the searches that model is driven with, the
tie labelling, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_towers_of_saigon",
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
Every expert step carries an optimal-action set.

The game
--------
45 of the 48 levels are solved (2229 presses), 15 of them provably shortest.
The three that are not -- 41, 44 and 46 -- are ChangePad levels and are listed
in `TowersOfSaigonSolver.skip_levels`; see "The searches" below.

Hanoi played with your feet. Three tower pieces -- ``Tower1`` (a big yellow
box), ``Tower2`` (a medium orange ring), ``Tower3`` (a small red dot) -- are
scattered over a 9x9 walled room, one player walks among them, and the win is

    all Tower3 on Tower2
    all Tower2 on Tower1

Every shipped level has the SAME COUNT of each piece, so that reads as "build N
complete towers": N cells each holding a Tower1 with a Tower2 on it with a
Tower3 on it. 48 levels, N from 1 to 8.

The collision layers are what make the stacking work:

    Background
    PadEnt, PadExit, ChangePad
    Player, Wall, Tower1
    Tower2
    Tower3

Tower1 shares the PLAYER's layer, so the player can never stand on one and two
Tower1s can never share a cell -- but Tower2 and Tower3 each have a layer to
themselves, so a cell may hold any subset of the three pieces at once. That is
the stack.

The mechanics, all MEASURED against the interpreter rather than read off the
.txt (``--selfcheck`` is the executable form):

* **A press pushes whatever is in the next cell, whole.** ``[ > Player | TowerN
  | ] -> [ > Player | > TowerN | ]`` fires once per piece, so shoving a cell
  that holds a Tower1 and a Tower2 moves BOTH of them one square. There is no
  chain push: only the cell directly in front of the player is ever given a
  force.
* **You may stack a smaller piece onto a bigger one and never the reverse.**
  Five rules strip the force off the player AND the offending piece when the
  destination already holds something the piece may not sit on: Tower2 onto
  Tower3, Tower1 onto Tower3, Tower1 onto Tower2, Tower3 onto Tower3, Tower2
  onto Tower2. A sixth (``[ > Tower1 | Tower3 ]``) catches a Tower1 that was
  still moving after one of the others had already cancelled the player.
* **A piece shoved into a WALL cancels the entire turn.** ``[ > TowerN | Wall ]
  -> cancel``, so the player does not move either -- and, because a cancel
  reverts before the `late` rules run, a cancelled turn is the one press in this
  game with literally no effect at all.
* **Pushing a two-piece stack into a stack it half-fits ONTO SHEARS it.** Shove
  a Tower2+Tower3 into a cell holding a Tower1+Tower2 and the Tower2 is stopped
  (it may not sit on a Tower2) while the Tower3 rides on: the destination
  becomes a finished tower and the player does not move. The .txt spells this
  out in a dedicated rule, but the rule is dead code -- the generic "Tower2 onto
  Tower2" cancel fires first and leaves the Tower3's force alone, which produces
  exactly the same board. `_Board.step` reproduces the mechanic by running the
  rules in file order rather than by special-casing the outcome.
* **ChangePads rotate the piece TYPES.** Walk onto an empty ChangePad while
  three other pads hold, one each, a lone Tower1, a lone Tower2 and a lone
  Tower3, and the three swap sizes at once: Tower1 -> Tower3, Tower2 -> Tower1,
  Tower3 -> Tower2. It is the only way to change what a piece IS, and on the pad
  levels it is the whole puzzle: you deliver pieces onto pads to have them
  re-cast, then walk onto the fourth pad to fire the machine.
* **PadEnt teleports to PadExit** on 17 levels, as four `late` rules with
  different clearances: the player needs the exit empty of pieces, a Tower1
  needs it empty, a Tower2 tolerates a Tower1 already sitting there and a
  Tower3 tolerates a Tower1 and a Tower2. Run in that order they carry a whole
  stack across, one piece per rule, in a single turn. The late rules run at the
  END of every turn that was not cancelled -- including a turn where the player
  merely walked into a wall -- and the level's START state has not had them
  applied, which is the one and only thing the (``noaction``-dead) ACTION key
  can change in this game.

Nothing in the game is random and nothing depends on the player's facing, so
the state is exactly ``(player cell, the three piece sets)``.

The native model
----------------
`_Board` is that state plus the rules above, run in the interpreter's order:
assign the player's force, let the three push rules hand it on, let the five
stack rules and the Tower1 catcher strip it again, cancel on a wall, fire the
ChangePad rotation, resolve the surviving forces per collision layer, then run
the four `late` pad rules. It runs at about 1.1 MILLION presses a second against
the interpreter's ~580 on the same boards -- both measured -- which is the whole
reason a native model exists here: level 22's exact field alone is over a million
states.

``--selfcheck`` is what makes it a statement about this game rather than about a
model of it: seeded random walks on all 48 shipped levels AND on thousands of
randomly SEATED boards (pieces sprinkled in a tight cluster around a random
anchor, with the pads always in the pool so the rotation fires often), with the
model's board compared against the engine's grid after EVERY press -- including
the presses that do nothing, which is where a collision-layer mistake hides.

**The one transition the model refuses to predict.** The ChangePad rule is a
four-bracket rule, and the interpreter picks which pad each bracket binds to by
iterating a `set` of candidate positions -- an order that depends on the whole
mutation history of the position index, not on anything about the board. That
only matters if TWO pads could bind the same bracket, i.e. if two pads hold a
lone Tower1 (or two a lone Tower2, or two a lone Tower3) while a third pad is
empty for the player to step onto. Every pad level ships exactly FOUR pads, so
firing the rule needs one empty pad plus one pad of each of the three types and
there is nothing left over to be ambiguous with -- the situation is impossible,
`--selfcheck` counts zero of it over its whole fuzz, and `_Board.step` returns
None (an unavailable move) rather than guess if a future edit ever creates it.

The searches
------------
Three, tried in order of how much they prove, all over the native model:

1. **The exact distance FIELD.** A forward BFS from the state being planned,
   stopped at the depth ``d*`` of the first win, which fixes the answer's length
   and collects ``F`` (every state within ``d*`` presses of the start), followed
   by a backward BFS from the won boards over the predecessor map the forward
   sweep recorded. That gives a provably shortest plan AND the EXACT optimal set
   at every step -- at a state ``d`` presses from a win, the optimal presses are
   exactly those whose successor is ``d - 1`` from one. It also proves a board
   DEAD when the forward sweep runs dry without a win, which is what recovery
   needs to hear. Affordable on the levels whose ``d*``-ball fits in
   ``FIELD_CAP`` states.

   This game does not admit `ps:sokoban_sanity`'s analytic ``preds``: a press
   here can teleport a piece across the board, re-cast three pieces into other
   types, or shear a stack, and none of those has an enumerable inverse. So the
   forward sweep BUILDS the reverse adjacency as it goes and the backward sweep
   reads it, which costs memory but is exact for the same reason -- the prune to
   ``F`` is sound because a state on a shortest start-to-win path has its whole
   continuation inside ``F``.

2. **Weighted A*** on a min-cost-matching heuristic (below), weights tried
   smallest-first so the best plan a rung can find is the one that is kept.

3. **Beam search** on the same heuristic, widths tried smallest-first.

The shortest plan any rung returns is the one recorded. A* and the beam are
searched under a heuristic that is NOT admissible, so their plans are wins and
not proofs; the field's are both.

The heuristic
-------------
``h = matching(Tower2 -> Tower1) + matching(Tower3 -> Tower2) + reach``, where
each matching is a min-cost perfect assignment (Hungarian, N <= 8) over floor
distances and ``reach`` is how far the player still has to walk to touch
anything that needs moving. The floor distances are a 0-1 BFS with a directed
ZERO-cost edge from PadEnt to PadExit, because arriving at the entry pad puts
you on the exit pad in the same turn -- level 17 is two rooms joined by nothing
else, and without that edge half of its board reads as unreachable and the
heuristic returns infinity.

It is a good ranking and a bad bound: one press can move a whole stack, so two
matched distances can fall at once and the estimate is not a lower bound. It is
also type-blind -- it cannot see that a piece on a ChangePad is about to become
a different size -- which is why the pad levels are the ones that need the
widest beams.

Optimal-action sets
-------------------
Never absent (the always-emit-optimal-targets rule), and derived two ways:

* on a field level, EXACTLY, as a by-product of already knowing every state's
  distance;
* elsewhere by REJOINING. Press ``alt`` instead of the plan's ``d`` at step
  ``i``; if some ``k <= REJOIN_HORIZON`` further presses land exactly on the
  state the plan itself reaches after ``k + 1`` presses, then "alt, those k, and
  the plan's own tail" is a plan of the SAME length, so ``alt`` is as good as
  ``d``. That is sound with no reference to optimality: every press in a set is
  demonstrably worth the same as the recorded one. (It is not complete -- a tie
  that rejoins later than the horizon, or never, is missed -- so on a non-field
  level the sets are a floor, not the whole truth.) ``--selfcheck`` verifies
  every set the cheap way round: it BUILDS the alternative press sequence and
  replays it through the interpreter, requiring a WIN in the same number of
  presses.

The field levels are what says how big the floor's gap is, because there both
answers exist. ``--selfcheck``'s pass 4c re-runs the cheap labeller on all 421
field steps and measures it: at ``REJOIN_HORIZON = 11`` it agrees with the exact
field on 418 of them and finds FEWER on 3, and it never once labels a press the
field does not (which it cannot: an equally-long alternative to a shortest plan
is itself shortest). At horizon 5 the gap is much wider -- level 24 alone loses
two of its three ties. That is not a proof about the beam levels, whose plans
are longer and whose ties may rejoin further out, but it is the calibration that
set the horizon.

Shortest, and how that is known
-------------------------------
Only the field levels are claimed shortest, and there the claim is two
independent derivations agreeing: this file's field (forward BFS + backward BFS
over the recorded predecessors) and, in ``--selfcheck``, a plain forward BFS to
the first win, which is shortest by construction and shares no code with the
field's backward half. Every level's plan, field or not, is replayed through the
REAL interpreter in ``--plans`` and must reach a WIN.

Recovery
--------
`supports_recovery` with RESET-mode recovery. The expert re-plans from the LIVE
board, not from a stored path: `_search` reads whatever the engine currently
holds and runs the ladder from there, so an exploration prefix that leaves the
board anywhere is answered from where it actually is. A prefix CAN strand this
game for good -- shove a Tower1 into a corner behind a Tower2, or fire the
rotation one time too many -- and the field says so by returning None, which is
exactly what `record_level` needs to hear to fall back to a RESET.

Re-plans run a deliberately CHEAPER ladder than the start plans do (see
``_LADDER_FULL`` vs ``_LADDER_FAST``): a start plan is derived once per level
ever, memoized to `PLAN_CACHE` and shared by every seed and every
`parallelize_generator` shard, so it can afford a 60000-wide beam; a re-plan
happens inside the recording loop and cannot.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variable is the presentation: the frame
rotation (rotation_k in {0,1,2,3}) with the matching directional action remap.
48 levels x 4 rotations is plenty of variety without the flips.

The game is deliberately NOT in `PuzzleScriptAdapter._FLIP_GAMES`, and the
reason is the ChangePad rule again. Which pad binds which bracket is settled by
a scan the interpreter runs in SCREEN order; a rotation is orientation
preserving and carries that scan to a rotated scan of a rotated board, but a
MIRROR does not -- a mirrored presentation would be a game whose rotation
machine picks the other pad, which is not this game. The situation is
unreachable on the shipped boards (four pads, see above), so the argument is
about the shape of the rule rather than about a measured divergence; with 48
levels the flips were not worth spending a rule-order argument on.
``--symmetry`` measures the rotation contract that IS taken: every level's plan
and a seeded 200-press random walk replayed at all four rotations, requiring
every frame to be the exact transform of the unaugmented one.

Rendering
---------
FIVE sprites were redrawn; the full account is the header comment in
``data/puzzlescript_games/Towers_of_Saigon.txt``. In short:

* DARKGREEN and the GREEN background are the same ARC index, which reduced
  PadExit to a single pixel in the middle of its cell -- exactly where Tower3
  draws and inside the window the Player was opaque across -- so the exit pad
  was invisible under either, and level 46 STARTS with the player on it.
* DARKBLUE and BLUE are also one index, so all three pads were the same colour
  as each other.
* Tower1's sprite was a full 5x5 border, which made a FINISHED tower
  (Tower1 + Tower2 + Tower3, i.e. the win condition itself) paint all 25 pixels
  of its cell and hide any pad under it.
* The Player was opaque across the whole middle 3x3, which is a superset of
  Tower2's ring AND of Tower3's one pixel, so a player standing on either read
  as a player standing on grass.

The three pads are now PURPLE / LIGHTBLUE / PINK, drawn on the four corners and
the four edge midpoints -- the one pixel set that survives both the Player and a
finished tower. Tower1 leaves its corners clear, and the Player is a hollow
eight-pixel figure (four corners, four inner diagonals) that Tower2's ring arms
and Tower3's middle pixel show through.

``--audit`` renders every cell COMPOSITION the game can reach as a whole 64x64
frame of a uniform board and requires them pairwise distinct: 4 pad states x
(8 piece subsets x with/without the player, less the 4 the collision layer
forbids) = 48, plus the wall. Whole frames rather than one cell sliced out of a
mixed board, because `_render_frame` upscales and centre-pads and slicing by
``cell_px`` arithmetic reads the wrong pixels (the ps:explod lesson).

Usage (run from the repo root):
    python solvers/generate_towers_of_saigon_training.py --episodes 200 \
        --out data/training_multi_level/towers_of_saigon

    python solvers/generate_towers_of_saigon_training.py --plans      # level report
    python solvers/generate_towers_of_saigon_training.py --selfcheck  # model + sets
    python solvers/generate_towers_of_saigon_training.py --audit      # rendering
    python solvers/generate_towers_of_saigon_training.py --symmetry   # augmentation
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
                                     screen_action)
from utils.rotation import inverse_remap_action_full                  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Towers_of_Saigon"
GAME_ID = "towers_of_saigon"

#: Engine directions, in the order ties are broken. ACTION5 is bound to nothing
#: (the .txt declares ``noaction`` and no rule has ``action`` on its left-hand
#: side), so branching on it would double the search for nothing -- see
#: `_Board.act` for the one thing it can still do and why no plan needs it.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Object indices, resolved once in `TowersOfSaigonExpert.setup` and asserted
#: against the parsed game so a renamed object is a crash rather than a silent
#: mis-read.
_OBJ_NAMES = ("player", "wall", "background", "tower1", "tower2", "tower3",
              "padent", "padexit", "changepad")

#: Disk cache of every level's start plan AND its optimal-action sets. This IS
#: the load-bearing cache here: the widest beam rung costs minutes, the answer
#: is seed-independent, and every `parallelize_generator` shard would otherwise
#: re-derive all 48. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "towers_of_saigon_plans.json"

#: How many states the exact field may hold before it gives up and hands over to
#: the heuristic searches. 1.2M keeps the forward sweep plus its predecessor map
#: inside about a gigabyte, which is what a `parallelize_generator` shard can
#: afford; the levels that close do so far under it.
FIELD_CAP = 1_200_000
#: The same, for a re-plan inside the recording loop.
FIELD_CAP_FAST = 120_000

#: ``(weight, node cap)`` rungs for weighted A*, smallest weight first: a lower
#: weight searches more and returns a shorter plan, so the first rung that
#: answers is the best one this ladder will find.
_ASTAR_FULL = ((2, 250_000), (3, 400_000), (5, 600_000), (9, 600_000))
_ASTAR_FAST = ((3, 60_000), (9, 120_000))
#: ``(width, depth)`` rungs for the beam, narrowest first for the same reason.
_BEAM_FULL = ((3_000, 200), (25_000, 220), (60_000, 240))
_BEAM_FAST = ((1_500, 200),)

#: Wall-clock the heuristic rungs may spend on ONE state. The full budget is
#: paid once per level ever (the answer is memoized to `PLAN_CACHE` and shared
#: by every seed and every shard); the fast one is paid inside the recording
#: loop, where it has to stay small.
LADDER_SECONDS_FULL = 360.0
LADDER_SECONDS_FAST = 20.0

#: How many states the beam may remember it has already been to. Deduping
#: against everything seen is what stops a wide beam from re-walking the same
#: shuffle for 200 layers, but the set is the beam's whole memory footprint --
#: 60000 wide x 240 deep would otherwise be tens of millions of states and
#: several gigabytes. Past the cap the beam simply stops recording, which costs
#: some duplicated expansion and nothing else.
BEAM_SEEN_CAP = 4_000_000

#: A plan longer than this cannot be replayed: the adapter cuts a level off at
#: 200 presses (`PuzzleScriptAdapter._max_steps`), counted from the `set_level`
#: that ends the exploration prefix.
MAX_PLAN = 185

#: How far a candidate press may wander before it has to be back on the plan for
#: `rejoin_optsets` to call it equally good. The worst case is 4**horizon model
#: steps per candidate, but the sweep dedups, so in practice it is far less: on
#: the field levels, where the exact answer is known to compare against, horizon
#: 11 costs 0.04s on a 14-press plan and recovers ALL of level 24's three ties,
#: while horizon 5 finds only one of them. Every level's cached sets carry the
#: horizon they were derived at and `TowersOfSaigonExpert.plan` re-derives them
#: when this changes, so raising it does not mean deleting `PLAN_CACHE`.
REJOIN_HORIZON = 11


def _bits(mask: int) -> list[int]:
    """The set cells of a cell bitmask, ascending."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic.

    Cells are flat ``r * w + c`` indices; a STATE is
    ``(player, tower1 mask, tower2 mask, tower3 mask)`` with the three masks as
    plain Python ints used as 81-bit sets. Walls, pads and the two teleport
    squares never change, so they live here rather than in the state.
    """

    __slots__ = ("h", "w", "wall", "pads", "pad_set", "ent", "exit", "nbr")

    def __init__(self, h: int, w: int, walls, pads, ent, exit_):
        self.h, self.w = h, w
        self.wall = bytearray(h * w)
        for i in walls:
            self.wall[i] = 1
        #: Ascending, so the ChangePad scan below is deterministic.
        self.pads = tuple(sorted(pads))
        self.pad_set = frozenset(pads)
        self.ent = ent
        self.exit = exit_
        #: ``nbr[cell][d]`` is the neighbour cell, or -1 off the board.
        self.nbr = []
        for i in range(h * w):
            r, c = divmod(i, w)
            row = []
            for d in DIRS:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self.nbr.append(tuple(row))

    # -- the mechanic --------------------------------------------------------
    def late(self, p, m1, m2, m3):
        """The four `late` pad rules, in file order.

        They run at the end of EVERY turn the engine did not cancel, so this is
        called from every branch of `step` except the wall cancel. The order is
        the mechanic: the player goes first and, once it is standing on the exit,
        every piece rule is blocked by its own ``no Player``; otherwise Tower1
        goes first onto an empty exit, then Tower2 (which tolerates the Tower1
        that just landed) and then Tower3 (which tolerates both), so one turn
        carries a whole stack across.
        """
        E, X = self.ent, self.exit
        if E is None or X is None:
            return (p, m1, m2, m3)
        be, bx = 1 << E, 1 << X
        if p == E and not ((m1 | m2 | m3) & bx):
            p = X
        if (m1 & be) and p != X and not ((m1 | m2 | m3) & bx):
            m1 = (m1 & ~be) | bx
        if (m2 & be) and p != X and not ((m2 | m3) & bx):
            m2 = (m2 & ~be) | bx
        if (m3 & be) and p != X and not (m3 & bx):
            m3 = (m3 & ~be) | bx
        return (p, m1, m2, m3)

    def act(self, state):
        """The state after pressing ACTION.

        ``noaction`` is declared and no rule reads the key, so an action turn
        assigns no force and matches no main rule -- but the engine still runs
        the `late` pass, so ACTION is exactly "apply the pad rules". After any
        other turn they have already been applied and it is a no-op; the ONE
        state where it is not is a level's start, which the engine loads without
        running them. No plan needs it (every direction press applies them too),
        so the search does not branch on it -- but `--selfcheck` presses it,
        because a live agent has that button.
        """
        return self.late(*state)

    def step(self, state, di: int):
        """The state after pressing ``DIRS[di]``, ``state`` if the press does
        nothing, or None if the press is the unmodelled ChangePad ambiguity.

        The body is the interpreter's turn in the .txt's rule order: push
        forces out, strip them again, cancel on a wall, fire the rotation,
        resolve per collision layer, run the late rules.
        """
        p, m1, m2, m3 = state
        q1 = self.nbr[p][di]
        if q1 < 0 or self.wall[q1]:
            # Walking into a wall or off the board moves nothing, but it is not
            # a cancel: the turn completes and the late rules run.
            return self.late(p, m1, m2, m3)
        b1 = 1 << q1
        pieces = ((1 if m1 & b1 else 0) | (2 if m2 & b1 else 0)
                  | (4 if m3 & b1 else 0))
        q2 = self.nbr[q1][di]
        if q2 >= 0:
            b2 = 1 << q2
            ahead = ((1 if m1 & b2 else 0) | (2 if m2 & b2 else 0)
                     | (4 if m3 & b2 else 0))
            wall2 = self.wall[q2]
        else:
            b2, ahead, wall2 = 0, 0, 0

        # [ > Player | TowerN | ] -- the third cell has to EXIST, which is the
        # only thing the board edge decides here (every shipped level is walled,
        # so on those it never bites).
        pf = True
        tf = pieces if q2 >= 0 else 0
        # The five stack rules, in file order. Each strips the player's force
        # AND the offending piece's; the ones after it then see no player and
        # stop matching, which is why the order is reproduced rather than
        # collapsed into a single test.
        if pf and (tf & 2) and (ahead & 4):
            pf = False
            tf &= ~2
        if pf and (tf & 1) and (ahead & 4):
            pf = False
            tf &= ~1
        if pf and (tf & 1) and (ahead & 2):
            pf = False
            tf &= ~1
        if pf and (tf & 4) and (ahead & 4):
            pf = False
            tf &= ~4
        if pf and (tf & 2) and (ahead & 2):
            pf = False
            tf &= ~2
        # [ > Tower1 | Tower3 ]: the catcher for a Tower1 still carrying a force
        # after an earlier rule had already cancelled the player.
        if (tf & 1) and (ahead & 4):
            tf &= ~1
        # [ > TowerN | Wall ] -> cancel: the whole turn reverts, late rules and
        # all, and the player does not move.
        if tf and wall2:
            return state

        # The ChangePad rotation. Its first bracket needs the player still
        # carrying a force and the destination pad clear of pieces, which is why
        # it can only fire on a plain walk.
        if pf and pieces == 0 and q1 in self.pad_set:
            lone1 = lone2 = lone3 = -1
            ambiguous = False
            for cell in self.pads:
                bit = 1 << cell
                held = ((1 if m1 & bit else 0) | (2 if m2 & bit else 0)
                        | (4 if m3 & bit else 0))
                if held == 1:
                    ambiguous |= lone1 >= 0
                    lone1 = cell
                elif held == 2:
                    ambiguous |= lone2 >= 0
                    lone2 = cell
                elif held == 4:
                    ambiguous |= lone3 >= 0
                    lone3 = cell
            if lone1 >= 0 and lone2 >= 0 and lone3 >= 0:
                if ambiguous:
                    # Two pads could bind the same bracket and the interpreter
                    # settles it by `set` iteration order. Unreachable on every
                    # shipped board (four pads: one empty for the player plus
                    # one of each type leaves nothing spare) -- refuse rather
                    # than guess. See the module docstring.
                    return None
                # Tower1 -> Tower3, Tower2 -> Tower1, Tower3 -> Tower2, all at
                # once: the three brackets of one rule application.
                m1 = (m1 & ~(1 << lone1)) | (1 << lone2)
                m2 = (m2 & ~(1 << lone2)) | (1 << lone3)
                m3 = (m3 & ~(1 << lone3)) | (1 << lone1)
                # The rule's right-hand side PLACES the player on the pad; it
                # carries no force there, so nothing else resolves.
                return self.late(q1, m1, m2, m3)

        # Force resolution, one collision layer at a time. Nothing here can
        # contest a cell: the player and a Tower1 in front of it are ONE chain
        # on layer 2 (they move together or not at all) and the other two pieces
        # are alone on layers 3 and 4.
        if pf and (tf & 1):
            if not (m1 & b2):
                m1 = (m1 & ~b1) | b2
                p = q1
        elif tf & 1:
            # A Tower1 with a force the player has lost. No rule order reaches
            # this (every rule that strips the player either strips the Tower1
            # too or is preceded by one that does), but resolving it is what the
            # engine would do, so the model does not depend on that argument.
            if not (m1 & b2) and q2 != p:
                m1 = (m1 & ~b1) | b2
        elif pf and not (m1 & b1):
            p = q1
        if (tf & 2) and not (m2 & b2):
            m2 = (m2 & ~b1) | b2
        if (tf & 4) and not (m3 & b2):
            m3 = (m3 & ~b1) | b2
        return self.late(p, m1, m2, m3)

    def won(self, state) -> bool:
        """``all Tower3 on Tower2`` and ``all Tower2 on Tower1``: every Tower3
        cell also holds a Tower2 and every Tower2 cell also holds a Tower1.

        Written as the win condition literally says (a containment test), not as
        "N finished towers" -- which is the same thing on every shipped board,
        where the three counts are equal, and still correct on an edited one."""
        _p, m1, m2, m3 = state
        return not (m3 & ~m2) and not (m2 & ~m1)

    # -- the exact field -----------------------------------------------------
    def field(self, state, cap: int):
        """``(dist, d_star, status)`` -- presses-to-win for every state on a
        shortest path from ``state``, the length of that path, and one of
        ``"exact"`` / ``"dead"`` / ``"capped"``.

        Sweep 1 is a forward BFS stopped at the depth of the first win, which
        both fixes ``d_star`` and collects ``F``; it records the predecessors of
        everything it reaches, because this game's presses (teleports, type
        rotations, sheared stacks) have no enumerable inverse to sweep backwards
        over. Sweep 2 is a backward BFS from the won boards over those recorded
        predecessors.

        The prune to ``F`` is exact where it is read: if ``s`` lies on a shortest
        start-to-win path then everything after it on that path has forward
        distance at most ``d_star`` and so is in ``F``, hence the backward sweep
        finds ``s``'s true distance. States on no shortest path can come out too
        high; nothing asks about those, because `solve` only ever compares
        against ``d_star - i``.

        ``"dead"`` is a PROOF -- the forward queue ran dry with no win reachable
        at all -- which is what `record_level` needs to hear to fall back to a
        RESET.
        """
        if self.won(state):
            return {state: 0}, 0, "exact"
        seen = {state: 0}
        preds: dict = {}
        queue = deque([state])
        d_star = None
        while queue:
            cur = queue.popleft()
            d = seen[cur]
            if d_star is not None and d >= d_star:
                break
            for di in range(4):
                nxt = self.step(cur, di)
                if nxt is None or nxt == cur:
                    continue
                if nxt in seen:
                    if seen[nxt] == d + 1:
                        preds[nxt].append(cur)
                    continue
                seen[nxt] = d + 1
                preds[nxt] = [cur]
                if self.won(nxt):
                    if d_star is None:
                        d_star = d + 1
                    continue                    # a won board is terminal
                queue.append(nxt)
            if len(seen) > cap:
                return None, None, "capped"
        if d_star is None:
            return None, None, "dead"

        dist = {}
        bq = deque()
        for s in seen:
            if self.won(s):
                dist[s] = 0
                bq.append(s)
        while bq:
            cur = bq.popleft()
            d = dist[cur]
            for prev in preds.get(cur, ()):
                if prev in dist:
                    continue
                dist[prev] = d + 1
                bq.append(prev)
        return dist, d_star, "exact"

    def field_optimal(self, dist, state):
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``. Ties come out in `DIRS` order, which is what makes a
        re-derived plan byte-identical across processes."""
        rest = dist.get(state)
        if not rest:
            return []
        out = []
        for di in range(4):
            nxt = self.step(state, di)
            if nxt is not None and nxt != state and dist.get(nxt, -1) == rest - 1:
                out.append((di, nxt))
        return out


# ---------------------------------------------------------------------------
# The heuristic
# ---------------------------------------------------------------------------

_INF = 1 << 20


def _floor_distances(board: _Board) -> list:
    """All-pairs shortest walks over the floor, with the teleport as a free
    one-way edge.

    Reaching PadEnt puts a body on PadExit within the same turn, so the graph
    carries a directed ZERO-cost ``ent -> exit`` edge and the sweep is a 0-1 BFS.
    Level 17 is two disjoint rooms joined by nothing else: without that edge half
    its board reads as unreachable and the heuristic returns infinity on the
    start state."""
    n = board.h * board.w
    out = []
    for src in range(n):
        d = [_INF] * n
        if board.wall[src]:
            out.append(d)
            continue
        d[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            if u == board.ent and board.exit is not None and d[board.exit] > d[u]:
                d[board.exit] = d[u]
                q.appendleft(board.exit)
            for v in board.nbr[u]:
                if v >= 0 and not board.wall[v] and d[v] > d[u] + 1:
                    d[v] = d[u] + 1
                    q.append(v)
        out.append(d)
    return out


def _hungarian(cost) -> int:
    """Min-cost perfect matching on a square matrix (Jonker-Volgenant,
    O(n^3)). N is at most 8 here, so this is microseconds."""
    n = len(cost)
    if n == 0:
        return 0
    inf = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = inf
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return int(round(-v[0]))


class _Heur:
    """``matching(Tower2 -> Tower1) + matching(Tower3 -> Tower2) + reach``.

    Every Tower2 has to end up sharing a cell with some Tower1 and every Tower3
    with some Tower2, and no two of them may share the same partner, so a
    min-cost PERFECT assignment over floor distances is the natural score.
    ``reach`` adds the walk the player still owes before it can touch anything
    that needs moving, which is what stops the search from wandering.

    It is a ranking, not a bound: one press can shove a whole stack, so two
    matched distances can fall together and this is not admissible. Nor is it
    type-aware -- a piece sitting on a ChangePad is about to become a different
    size and the score cannot see it -- which is exactly why the pad levels are
    the ones that need the widest beams."""

    __slots__ = ("board", "dist")

    def __init__(self, board: _Board):
        self.board = board
        self.dist = _floor_distances(board)

    def __call__(self, state) -> int:
        p, m1, m2, m3 = state
        if not (m3 & ~m2) and not (m2 & ~m1):
            return 0
        D = self.dist
        ones, twos, threes = _bits(m1), _bits(m2), _bits(m3)
        h = _hungarian([[D[a][b] for b in ones] for a in twos])
        h += _hungarian([[D[a][b] for b in twos] for a in threes])
        need = [a for a in twos if not (m1 >> a) & 1]
        need += [a for a in threes if not (m2 >> a) & 1]
        if need:
            h += max(0, min(D[p][a] for a in need) - 1)
        return h


# ---------------------------------------------------------------------------
# The heuristic searches
# ---------------------------------------------------------------------------

def _astar(board: _Board, state, heur, weight: int, cap: int, deadline):
    """Weighted A*. Returns a press-index list, or None if the cap, the deadline
    or the reachable space ran out first."""
    import heapq                                                  # noqa: PLC0415
    if board.won(state):
        return []
    counter = 0
    pq = [(weight * heur(state), 0, 0, state, ())]
    best_g = {state: 0}
    nodes = 0
    while pq:
        _f, g, _c, cur, path = heapq.heappop(pq)
        if g > best_g.get(cur, 1 << 30):
            continue
        if g >= MAX_PLAN:
            continue
        for di in range(4):
            nxt = board.step(cur, di)
            if nxt is None or nxt == cur:
                continue
            nodes += 1
            if board.won(nxt):
                return list(path) + [di]
            ng = g + 1
            if best_g.get(nxt, 1 << 30) <= ng:
                continue
            best_g[nxt] = ng
            counter += 1
            heapq.heappush(pq, (ng + weight * heur(nxt), ng, counter, nxt,
                                path + (di,)))
        if nodes >= cap or time.monotonic() > deadline:
            return None
    return None


def _beam(board: _Board, state, heur, width: int, depth: int, deadline):
    """Greedy beam on the heuristic alone, deduped against everything already
    seen. Returns a press-index list or None."""
    if board.won(state):
        return []
    frontier = [(heur(state), state, ())]
    seen = {state}
    recording = True
    for _ in range(min(depth, MAX_PLAN)):
        nxt = []
        for _score, cur, path in frontier:
            for di in range(4):
                s2 = board.step(cur, di)
                if s2 is None or s2 == cur or s2 in seen:
                    continue
                p2 = path + (di,)
                if board.won(s2):
                    return list(p2)
                if recording:
                    seen.add(s2)
                nxt.append((heur(s2), s2, p2))
        if recording and len(seen) > BEAM_SEEN_CAP:
            recording = False
        if not nxt or time.monotonic() > deadline:
            return None
        nxt.sort(key=lambda x: x[0])
        frontier = nxt[:width]
    return None


def rejoin_optsets(board: _Board, state, presses, horizon: int = REJOIN_HORIZON):
    """Per-step optimal SETS for a plan whose length is not known to be optimal.

    A press ``alt`` is in step ``i``'s set when, taken instead of the plan's own,
    some ``k <= horizon`` further presses land exactly on the state the plan
    reaches after ``k + 1`` presses -- because then "alt, those k, and the plan's
    own tail" is a plan of the SAME length. That is sound without reference to
    optimality: every press in a set demonstrably costs what the recorded one
    costs. It is not complete -- a tie that rejoins later than the horizon, or
    that never rejoins, is missed -- so on a non-field level these sets are a
    floor rather than the whole truth.

    ``--selfcheck`` verifies them the direct way: it builds the alternative
    press sequence and replays it through the interpreter."""
    states = [state]
    for di in presses:
        states.append(board.step(states[-1], di))
    out = []
    for i, di in enumerate(presses):
        best = {di}
        for alt in range(4):
            if alt == di:
                continue
            s = board.step(states[i], alt)
            if s is None or s == states[i]:
                continue
            if board.won(s):
                if len(presses) - i == 1:
                    best.add(alt)
                continue
            if s == states[i + 1]:
                best.add(alt)
                continue
            frontier = {s}
            for k in range(1, horizon + 1):
                if i + 1 + k >= len(states):
                    break
                target = states[i + 1 + k]
                grown = set()
                hit = False
                for cur in frontier:
                    for d2 in range(4):
                        s2 = board.step(cur, d2)
                        if s2 is None or s2 == cur:
                            continue
                        if s2 == target:
                            hit = True
                            break
                        grown.add(s2)
                    if hit:
                        break
                if hit:
                    best.add(alt)
                    break
                if not grown:
                    break
                frontier = grown
        out.append([DIRS[d] for d in sorted(best)])
    return out


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class TowersOfSaigonExpert(PSExpert):
    """`PSExpert`'s plan memo and disk cache around the `_Board` ladder.

    The base class keeps the memo, the level scoping and the disk cache; only
    the strategy underneath changes. Here it is "read the live board, try the
    exact field, then weighted A*, then a beam, and keep the shortest answer",
    so `heuristic` is never called on the ENGINE and asserts rather than
    returning a number nothing would use.

    `_search` reads the engine's current grid every time, so a re-plan from an
    arbitrary state (a recovery prefix, an epsilon detour) is answered from
    where the board actually is -- including "this board is now dead", which an
    exploration prefix really can produce here."""

    directions = list(DIRS)
    #: `_key` is the dynamic objects only (player + the three piece types),
    #: which is canonical WITHIN a level but not across them -- walls, pads and
    #: the teleport squares are static per level and differ between them.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        idx = self.g.obj_name_to_idx
        missing = [n for n in _OBJ_NAMES if n not in idx]
        if missing:
            raise AssertionError(
                f"Towers_of_Saigon.txt is missing object(s) {missing}")
        self.player_ids = set(self.g.resolve_object_name("player"))
        self.wall_ids = set(self.g.resolve_object_name("wall"))
        self.t1_ids = set(self.g.resolve_object_name("tower1"))
        self.t2_ids = set(self.g.resolve_object_name("tower2"))
        self.t3_ids = set(self.g.resolve_object_name("tower3"))
        self.ent_ids = set(self.g.resolve_object_name("padent"))
        self.exit_ids = set(self.g.resolve_object_name("padexit"))
        self.pad_ids = set(self.g.resolve_object_name("changepad"))
        self.dyn_ids = (self.player_ids | self.t1_ids | self.t2_ids
                        | self.t3_ids)
        #: `_Board` + `_Heur` by STATIC signature (see `read`), not by level
        #: index, so the tables are built once per distinct geometry.
        self._boards: dict = {}
        #: Set while `prepare_expert` derives the START plans -- the one time
        #: the expensive ladder rungs are affordable (once per level ever,
        #: memoized to disk and shared by every seed and every shard).
        self.full_budget = False
        #: Filled by `_search`: how each level's plan was found, for `--plans`.
        self.provenance: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "TowersOfSaigonExpert searches a native model; the engine-side "
            "heuristic is unused")

    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan`, plus the record of WHICH rung answered.

        The provenance has to survive the disk cache or every report run after
        the first would have to re-derive it to know whether a level's tie sets
        are the field's exact ones or `rejoin_optsets`' floor -- so it is stored
        beside the plan and read back out of the cache entry."""
        self._last_how = None
        got = super().plan(eng, level)
        if level is None:
            return got
        how = self._last_how
        entry = self._disk.get(level)
        if entry is not None:
            dirty = False
            if how is None:
                how = entry.get("how")
            elif entry.get("how") != how:
                entry["how"] = how
                dirty = True
            # Re-label a cached plan whose sets were derived at a different
            # `REJOIN_HORIZON`. The presses are the expensive half and stay; the
            # sets are a pure function of (board, start, presses) and cost
            # milliseconds. Only when the engine really is AT the cached start,
            # because that is the only state those presses belong to.
            if (got is not None and how not in ("field", "won")
                    and entry.get("horizon") != REJOIN_HORIZON
                    and entry.get("start")
                    == sorted([r, c, o] for (r, c, o) in self._key(eng))):
                board, _heur, state = self.read(eng)
                sets = rejoin_optsets(
                    board, state, [self.directions.index(d) for d in got])
                entry["optsets"] = sets
                entry["horizon"] = REJOIN_HORIZON
                got = Plan(list(got), sets)
                dirty = True
            if dirty:
                self._save_disk()
        self.provenance[level] = how or "cached"
        return got

    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn))

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng):
        """``(board, heur, state)`` for the engine's current grid.

        Boards are cached by their STATIC signature (dimensions, walls, pads,
        the two teleport squares), so the distance tables are never rebuilt and
        a board is shared by every state of its level."""
        h, w = eng.height, eng.width
        walls, pads = [], []
        ent = exit_ = None
        player = None
        m1 = m2 = m3 = 0
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.pad_ids:
                    pads.append(i)
                if cell & self.ent_ids:
                    ent = i
                if cell & self.exit_ids:
                    exit_ = i
                if cell & self.player_ids:
                    player = i
                if cell & self.t1_ids:
                    m1 |= 1 << i
                if cell & self.t2_ids:
                    m2 |= 1 << i
                if cell & self.t3_ids:
                    m3 |= 1 << i
        sig = (h, w, tuple(walls), tuple(pads), ent, exit_)
        entry = self._boards.get(sig)
        if entry is None:
            board = _Board(h, w, walls, pads, ent, exit_)
            entry = self._boards[sig] = (board, _Heur(board))
        return entry[0], entry[1], (player, m1, m2, m3)

    # -- the ladder ----------------------------------------------------------
    def solve(self, board, heur, state, full: bool):
        """``(Plan, how)`` or ``(None, how)``.

        ``how`` is one of ``"won"``, ``"field"`` (provably shortest, exact tie
        sets), ``"astar:<w>"`` / ``"beam:<width>"`` (a win, tie sets by rejoin),
        ``"dead"`` (proved unwinnable) or ``"unsolved"`` (every rung ran out)."""
        if board.won(state):
            return Plan([], []), "won"
        dist, d_star, status = board.field(
            state, FIELD_CAP if full else FIELD_CAP_FAST)
        if status == "dead":
            return None, "dead"
        if status == "exact":
            presses, optsets = [], []
            cur = state
            for _ in range(d_star):
                best = board.field_optimal(dist, cur)
                if not best:                 # unreachable: dist[cur] > 0 has a step
                    return None, "unsolved"
                presses.append(DIRS[best[0][0]])
                optsets.append([DIRS[di] for di, _ in best])
                cur = best[0][1]
            if d_star > MAX_PLAN:
                return None, "unsolved"      # cannot be replayed in 200 presses
            return Plan(presses, optsets), "field"

        deadline = time.monotonic() + (LADDER_SECONDS_FULL if full
                                       else LADDER_SECONDS_FAST)
        found, how = None, "unsolved"
        for weight, cap in (_ASTAR_FULL if full else _ASTAR_FAST):
            got = _astar(board, state, heur, weight, cap, deadline)
            if got is not None:
                found, how = got, f"astar:w{weight}"
                break
        beams = list(_BEAM_FULL if full else _BEAM_FAST)
        if found is not None:
            # A* has already answered, so the beam is only being asked whether
            # it can do BETTER. One narrow rung is worth that; grinding a 60000
            # wide one for a level that is already solved is not.
            beams = beams[:1]
        for width, depth in beams:
            got = _beam(board, state, heur, width, depth, deadline)
            if got is not None:
                if found is None or len(got) < len(found):
                    found, how = got, f"beam:{width}"
                break
        if found is None or len(found) > MAX_PLAN:
            return None, "unsolved"
        presses = [DIRS[di] for di in found]
        return Plan(presses, rejoin_optsets(board, state, found)), how

    def _search(self, eng):
        board, heur, state = self.read(eng)
        if state[0] is None:                 # no player on the board
            return None
        plan, how = self.solve(board, heur, state, self.full_budget)
        self._last_how = how
        return plan


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class TowersOfSaigonSolver(PSAStarSolver):
    game_id = "puzzlescript_towers_of_saigon"
    game_name = GAME_NAME
    expert_cls = TowersOfSaigonExpert

    #: `games/ps:towers_of_saigon/ps:towers_of_saigon.py` is a plain passthrough
    #: (it constructs the adapter and nothing else), so there is nothing to gain
    #: by routing through it -- but it IS what a live agent is handed, so if that
    #: wrapper ever grows a patch this must be set to ``"ps:towers_of_saigon"``.
    game_module_id = ""

    #: Unused: the expert runs its own ladder over a native model rather than
    #: the base class's engine-side A*. Left at the base values so nothing reads
    #: a lie off them.
    weight = 1
    node_cap = 400_000

    #: The longest plan the ladder returns is under 110 presses; `MAX_PLAN`
    #: refuses anything that could not be replayed inside the adapter's own
    #: 200-step per-level budget, which `set_level` resets before the plan
    #: starts anyway.
    max_steps = MAX_PLAN + 5

    #: The three levels the ladder cannot close. They are LISTED rather than
    #: re-discovered so a run with a cold plan cache does not spend the full
    #: ladder budget (minutes each) re-failing on them; ``--plans --all``
    #: re-derives them, which is how a future search improvement gets checked.
    #: All three are ChangePad levels -- 42, 45 and 47 in the game's own
    #: numbering -- which is exactly where the type-blind heuristic is weakest
    #: (see the module docstring). A wider beam does not rescue them: 90000 and
    #: a noisy-restart beam at 4000 x 6 both come back empty.
    skip_levels = frozenset({41, 44, 46})

    def prepare_expert(self, game, expert) -> None:
        """Derive every level's start plan before `discover_solvable` asks for
        it. Same work either way, but it fills the disk cache in one pass, makes
        the startup cost visible as startup, and is the ONLY window in which the
        expensive ladder rungs are enabled."""
        expert.full_budget = True
        try:
            for level in range(game.n_levels):
                if level in self.skip_levels:
                    continue
                game.set_level(level)
                expert.plan(game._engine, level)
                expert.provenance[level] = getattr(expert, "_last_how", "cached")
        finally:
            expert.full_budget = False


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(prepare: bool = True):
    solver = TowersOfSaigonSolver()
    if not prepare:
        game = solver.make_game(0)
        expert = solver.expert_cls(game, node_cap=solver.node_cap,
                                   weight=solver.weight)
        return solver, game, expert, []
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _seat(eng, state, ids) -> None:
    """Write a model state onto the engine's grid, leaving the static geometry
    (walls, pads, teleport squares, background) exactly as the level has it."""
    player_id, t1, t2, t3 = ids
    w = eng.width
    for row in eng.grid:
        for cell in row:
            cell.discard(player_id)
            cell.discard(t1)
            cell.discard(t2)
            cell.discard(t3)
    p, m1, m2, m3 = state
    if p is not None:
        eng.grid[p // w][p % w].add(player_id)
    for obj, mask in ((t1, m1), (t2, m2), (t3, m3)):
        for i in _bits(mask):
            eng.grid[i // w][i % w].add(obj)
    eng._position_index_dirty = True
    eng._mutation_counter += 1


def _ids(expert) -> tuple:
    return (min(expert.player_ids), min(expert.t1_ids), min(expert.t2_ids),
            min(expert.t3_ids))


def _plans(verbose: bool = True) -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the searches and the tie labelling at
    once."""
    _solver, game, expert, solvable = _new()
    eng = game._engine
    total = tie_steps_all = bad = 0
    unsolved = []
    for level in range(game.n_levels):
        if level in _solver.skip_levels:
            # Not re-derived here: with a cold `PLAN_CACHE` the ladder would
            # spend its whole budget failing again. ``--plans --all`` clears
            # `skip_levels` and does re-derive them.
            unsolved.append(level)
            if verbose:
                print(f"  L{level:2d}: SKIPPED (see skip_levels; --all to retry)")
            continue
        game.set_level(level)
        _board, _heur, state = expert.read(eng)
        n = bin(state[1]).count("1")
        plan = expert.plan(eng, level)
        how = expert.provenance.get(level, "?")
        if plan is None:
            unsolved.append(level)
            if verbose:
                print(f"  L{level:2d}: {n} towers  UNSOLVED ({how})")
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tie_steps = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        tie_steps_all += tie_steps
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        if verbose:
            print(f"  L{level:2d}: {n} towers  {len(plan):3d} presses  "
                  f"win={won}  ({how}, budget {game._max_steps}, {room})  "
                  f"{tie_steps:3d}/{len(plan)} steps with a tie set")
    exact = sum(1 for h in expert.provenance.values()
                if h in ("field", "won"))
    if verbose:
        print(f"  solvable levels: {len(solvable)}/{game.n_levels}"
              f"  ({exact} provably shortest by the exact field)")
        print(f"  {total} presses, {tie_steps_all} of them with a second "
              f"equally-right answer")
        print(f"  unsolved: {unsolved}" if unsolved else "  every level solved")
        print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


def _random_state(board: _Board, n: int, rng: random.Random):
    """A board with ``n`` of each piece sprinkled in a tight cluster around a
    random anchor, with the ChangePads always in the pool.

    The cluster is the point: pieces scattered uniformly over 49 cells almost
    never end up adjacent, and every interesting rule in this game is about what
    is in the NEXT cell. Keeping the pads in the pool is what makes the rotation
    fire often enough to be worth checking."""
    free = [i for i in range(board.h * board.w) if not board.wall[i]]
    anchor = rng.choice(free)
    ar, ac = divmod(anchor, board.w)
    near = sorted(free, key=lambda i: (abs(i // board.w - ar)
                                       + abs(i % board.w - ac), i))
    pool = list(dict.fromkeys(near[:min(len(near), 4 * n + 6)]
                              + list(board.pads)))
    cells = rng.sample(pool, min(len(pool), 3 * n + 1))
    p = cells[0]
    m1 = m2 = m3 = 0
    for cell in cells[1:]:
        held = rng.randrange(1, 8)
        if held & 1:
            m1 |= 1 << cell
        if held & 2:
            m2 |= 1 << cell
        if held & 4:
            m3 |= 1 << cell
    m1 &= ~(1 << p)              # Player and Tower1 share a collision layer
    return (p, m1, m2, m3)


def _selfcheck(walk_presses: int = 300, boards: int = 60,
               recovery: int = 4, verbose: bool = True) -> int:
    """Five things the expert would otherwise be trusted on.

    1. **The native model is the interpreter's, on the shipped boards.** A
       seeded random walk on every level, comparing `_Board.step`'s state
       against the engine's grid after EVERY press -- including the presses that
       do nothing, which is where a collision-layer mistake hides -- and
       including ACTION, which `_Board.act` claims is the `late` pass and
       nothing else.
    2. **...and on boards the shipped ones never reach.** The same comparison
       from `_random_state` layouts, which is where the stack shears, the
       blocked teleports and the ChangePad rotation actually get exercised.
       Every one of these presses is also a chance for the unmodelled ChangePad
       ambiguity to appear; the count is printed and is expected to be zero.
    3. **The field plans are shortest.** A plain forward BFS to the first win,
       which is shortest by construction and shares no code with the field's
       backward half, must return the field's plan length.
    4. **Every optimal set is REAL.** For a field level, each step's set is
       re-derived by brute force (a fresh depth-bounded BFS from each successor)
       and must match exactly. For every other level, each extra press in a set
       is turned back into a whole alternative plan and REPLAYED THROUGH THE
       INTERPRETER, which must reach a WIN in the same number of presses --
       which is the entire claim `rejoin_optsets` makes.
    5. **Recovery is answered from ANY board, not just the start.** From boards
       reached by seeded random presses, the re-plan ladder (the FAST one, which
       is what generation runs) must either return a plan that WINS when
       replayed through the interpreter, or return None.
    """
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    ids = _ids(expert)
    bad = 0

    for level in range(game.n_levels):
        game.set_level(level)
        board, _heur, state = expert.read(eng)
        rng = random.Random(f"{GAME_ID}:walk:{level}")
        drift = 0
        for k in range(walk_presses):
            di = rng.randrange(5)
            if di == 4:
                eng.step("action")
                state = board.act(state)
            else:
                eng.step(DIRS[di])
                nxt = board.step(state, di)
                if nxt is None:
                    continue                 # counted in pass 2
                state = nxt
            _b, _h, live = expert.read(eng)
            if live != state:
                drift += 1
                print(f"    L{level:2d}: model DIVERGES at press {k}")
                break
        bad += drift
    if verbose:
        print(f"  pass 1: {game.n_levels} levels x {walk_presses} random presses"
              f" -- {'model matches the interpreter' if not bad else 'DIVERGED'}")

    amb = seated = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, _heur, start = expert.read(eng)
        n = bin(start[1]).count("1")
        rng = random.Random(f"{GAME_ID}:seated:{level}")
        for _ in range(boards):
            state = _random_state(board, n, rng)
            _seat(eng, state, ids)
            for _ in range(14):
                di = rng.randrange(5)
                if di == 4:
                    eng.step("action")
                    state = board.act(state)
                else:
                    eng.step(DIRS[di])
                    nxt = board.step(state, di)
                    if nxt is None:
                        amb += 1
                        _b, _h, state = expert.read(eng)
                        continue
                    state = nxt
                seated += 1
                _b, _h, live = expert.read(eng)
                if live != state:
                    bad += 1
                    print(f"    L{level:2d}: model DIVERGES on a seated board")
                    state = live
                    break
    if verbose:
        print(f"  pass 2: {seated} presses on {game.n_levels * boards} seated "
              f"boards -- {'model matches' if not bad else 'DIVERGED'}; "
              f"{amb} ChangePad-ambiguous transitions (expected 0)")
    bad += amb                                # a single one invalidates the model

    def brute(board, state, limit, cap=250_000):
        """Presses to a win from ``state``, searched fresh -- ``None`` past
        ``limit``, ``"cap"`` if the sweep outgrew ``cap`` states before it could
        say either. The cap is what keeps this affordable on the deep field
        levels (level 22's answer is 74 presses, so an uncapped sweep from its
        first step is the whole reachable space)."""
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
                if nxt is None or nxt == cur or nxt in seen:
                    continue
                if board.won(nxt):
                    return d + 1
                seen.add(nxt)
                queue.append((nxt, d + 1))
            if len(seen) > cap:
                return "cap"
        return None

    field_levels = [lv for lv, how in expert.provenance.items()
                    if how in ("field", "won")]
    for level in sorted(field_levels):
        game.set_level(level)
        board, _heur, state = expert.read(eng)
        plan = expert.plan(eng, level)
        seen = {state}
        queue = deque([(state, 0)])
        shortest = 0 if board.won(state) else None
        while queue and shortest is None:
            cur, d = queue.popleft()
            for di in range(4):
                nxt = board.step(cur, di)
                if nxt is None or nxt == cur or nxt in seen:
                    continue
                if board.won(nxt):
                    shortest = d + 1
                    break
                seen.add(nxt)
                queue.append((nxt, d + 1))
        ok = plan is not None and shortest == len(plan)
        bad += not ok
        if not ok:
            print(f"    L{level:2d}: field says {len(plan) if plan else None}, "
                  f"an independent BFS says {shortest} -- NOT SHORTEST")
    if verbose:
        print(f"  pass 3: {len(field_levels)} field level(s) re-proved shortest "
              f"by an independent forward BFS")

    exact_steps = exact_bad = exact_skipped = 0
    for level in sorted(field_levels):
        game.set_level(level)
        board, _heur, state = expert.read(eng)
        plan = expert.plan(eng, level)
        sets = getattr(plan, "optsets", None) or []
        cur = state
        for i, direction in enumerate(plan):
            rest = len(plan) - i
            truth = []
            capped = False
            for di in range(4):
                nxt = board.step(cur, di)
                if nxt is None or nxt == cur:
                    continue
                got = brute(board, nxt, rest - 1)
                if got == "cap":
                    capped = True
                    break
                if got == rest - 1:
                    truth.append(DIRS[di])
            cur = board.step(cur, DIRS.index(direction))
            if capped:
                exact_skipped += 1
                continue
            exact_steps += 1
            if sorted(truth) != sorted(sets[i]):
                exact_bad += 1
                print(f"    L{level:2d} step {i}: field says {sorted(sets[i])}, "
                      f"brute force says {sorted(truth)}")
    bad += exact_bad
    if verbose:
        print(f"  pass 4a: {exact_steps} field tie set(s) brute-forced -- "
              f"{exact_bad} mismatch(es); {exact_skipped} step(s) too deep to "
              f"re-search inside the cap")

    cross = cross_bad = cross_short = 0
    for level in sorted(field_levels):
        game.set_level(level)
        board, _heur, state = expert.read(eng)
        plan = expert.plan(eng, level)
        exact = getattr(plan, "optsets", None) or []
        cheap = rejoin_optsets(board, state,
                               [DIRS.index(d) for d in plan])
        for i in range(len(plan)):
            cross += 1
            if not set(cheap[i]) <= set(exact[i]):
                cross_bad += 1
                print(f"    L{level:2d} step {i}: rejoin labelled "
                      f"{sorted(set(cheap[i]) - set(exact[i]))} which the exact "
                      f"field does not")
            elif set(cheap[i]) != set(exact[i]):
                cross_short += 1
    bad += cross_bad
    if verbose:
        print(f"  pass 4c: rejoin re-run on {cross} field step(s) -- "
              f"{cross_bad} press(es) it labels that the field does not "
              f"(must be 0: an equally-long alternative to a SHORTEST plan is "
              f"itself shortest); {cross_short} step(s) where it found FEWER "
              f"than the field (the floor's gap, at horizon "
              f"{REJOIN_HORIZON})")

    rejoin_checked = rejoin_bad = 0
    for level in range(game.n_levels):
        how = expert.provenance.get(level)
        if how in (None, "field", "won", "dead", "unsolved"):
            continue
        game.set_level(level)
        board, _heur, state = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        sets = getattr(plan, "optsets", None) or []
        states = [state]
        for direction in plan:
            states.append(board.step(states[-1], DIRS.index(direction)))
        for i, direction in enumerate(plan):
            for alt in sets[i]:
                if alt == direction:
                    continue
                witness = _rejoin_witness(board, states, i, DIRS.index(alt))
                rejoin_checked += 1
                if witness is None:
                    rejoin_bad += 1
                    print(f"    L{level:2d} step {i}: '{alt}' is labelled "
                          f"optimal but does not rejoin")
                    continue
                seq = (list(plan[:i]) + [alt]
                       + [DIRS[d] for d in witness[0]]
                       + list(plan[i + 1 + len(witness[0]):]))
                game.set_level(level)
                for d in seq:
                    eng.step(d)
                if not eng.check_win() or len(seq) != len(plan):
                    rejoin_bad += 1
                    print(f"    L{level:2d} step {i}: the '{alt}' alternative "
                          f"did not win in {len(plan)} presses")
    bad += rejoin_bad
    if verbose:
        print(f"  pass 4b: {rejoin_checked} rejoin tie(s) rebuilt as whole "
              f"plans and replayed through the interpreter -- {rejoin_bad} bad")

    wrong = dead = won_after = 0
    for level in range(game.n_levels):
        rng = random.Random(f"{GAME_ID}:recovery:{level}")
        for _ in range(recovery):
            game.set_level(level)
            for _ in range(rng.randrange(1, 25)):
                eng.step(DIRS[rng.randrange(4)])
                if eng.check_win():
                    break
            if eng.check_win():
                continue                     # the walk already won
            plan = expert._search(eng)
            if plan is None:
                dead += 1
                continue
            for direction in plan:
                eng.step(direction)
            if eng.check_win():
                won_after += 1
            else:
                wrong += 1
    bad += wrong
    if verbose:
        print(f"  pass 5: {game.n_levels * recovery} random boards -- "
              f"{won_after} re-planned to a WIN in the interpreter, {dead} "
              f"answered None, {wrong} WRONG")
    return bad


def _rejoin_witness(board: _Board, states, i: int, alt: int,
                    horizon: int = REJOIN_HORIZON):
    """``(presses, k)`` proving ``alt`` at step ``i`` rejoins the plan, or None.

    Re-derived here rather than remembered by `rejoin_optsets`, so ``--selfcheck``
    is checking the claim and not the bookkeeping."""
    s = board.step(states[i], alt)
    if s is None or s == states[i]:
        return None
    if board.won(s):
        return ([], 0) if len(states) - 1 - i == 1 else None
    if s == states[i + 1]:
        return ([], 0)
    frontier = {s: ()}
    for k in range(1, horizon + 1):
        if i + 1 + k >= len(states):
            return None
        target = states[i + 1 + k]
        grown = {}
        for cur, path in frontier.items():
            for d2 in range(4):
                s2 = board.step(cur, d2)
                if s2 is None or s2 == cur:
                    continue
                if s2 == target:
                    return (list(path) + [d2], k)
                if s2 not in grown:
                    grown[s2] = path + (d2,)
        if not grown:
            return None
        frontier = grown
    return None


def _compositions() -> list:
    """Every cell COMPOSITION the engine can reach.

    Four pad states x the eight piece subsets x with/without the player, less
    the four the collision layer forbids (Player and Tower1 share layer 2, so no
    cell can hold both), plus the wall. ``player + tower1`` being absent is not
    an omission and is asserted by that layer, not by this list."""
    pads = [(), ("padent",), ("padexit",), ("changepad",)]
    out = [("wall",)]
    for pad in pads:
        for r in range(8):
            pieces = tuple(name for bit, name in
                           ((1, "tower1"), (2, "tower2"), (4, "tower3"))
                           if r & bit)
            out.append(pad + pieces)
            if "tower1" not in pieces:
                out.append(pad + pieces + ("player",))
    return out


def _audit(verbose: bool = True) -> int:
    """Render every reachable composition as a whole 64x64 frame of a uniform
    board and require them pairwise distinct.

    Whole frames rather than one cell sliced out of a mixed board: `_render_frame`
    upscales and centre-pads, so slicing a cell by ``cell_px`` arithmetic reads
    the wrong pixels (the ps:explod lesson). Two uniform boards render identically
    iff their cells do.

    The size list is derived rather than assumed (all 48 levels are 9x9, i.e.
    7 px a cell), so an edited level is covered."""
    from adapters.puzzlescript_adapter import _render_frame            # noqa: PLC0415

    _solver, game, _expert, _solvable = _new(prepare=False)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = _compositions()

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
            print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, "
                  f"{len(levels)} level(s)): {len(shots)} compositions, "
                  f"{'OK' if not clashes else 'CLASH'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {name(a)}  ==  {name(b)}")
    return bad


def _symmetry(walk_presses: int = 150, seeds: int = 60,
              verbose: bool = True) -> int:
    """Replay every level through the ADAPTER at every rotation it draws and
    require the frames to be the same board underneath.

    The rotation is presentation-only -- the adapter turns the rendered frame
    and inverse-remaps the directional input, leaving the engine grid alone --
    so this is really a check that the plan's screen actions are computed the
    right way round (`screen_action` vs `inverse_remap_action_full`; getting that
    backwards is the trap `solvers/common/ps_astar.py` documents, and nothing
    raises when it happens -- the adapter just executes a different direction
    than the one planned).

    Rather than wait for the ``rotation_k == 0`` draw on every level, each run's
    frames are turned BACK by its own k and compared against the first run of
    that level: every presentation must un-rotate to the same board.

    Both the PLANS and a seeded random walk are replayed. The walk reaches the
    boards a plan never visits (a piece flat against a wall, a stack that will
    not shear, a rotation fired one time too many) and it presses the
    ``noaction``-dead ACTION key as well."""
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

    def unturn(frame, k):
        return np.ascontiguousarray(np.rot90(frame, k=-k) if k else frame)

    levels = [lv for lv in range(game.n_levels) if plans[lv] is not None]
    pending = {lv: set(range(4)) for lv in levels}
    ref, bad, drawn = {}, 0, set()
    for seed in range(seeds):
        if not any(pending.values()):
            break
        g = solver.make_game(seed)
        for level in levels:
            g.set_level(level)
            k = g._rotation_k
            drawn.add((level, k))
            if k not in pending[level]:
                continue
            pending[level].discard(k)
            rng = random.Random(f"{GAME_ID}:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (
                    ("plan", [screen_action(d, k, False, False)
                              for d in plans[level]]),
                    ("walk", [inverse_remap_action_full(a, k, False, False)
                              for a in walk])):
                frames = [unturn(f, k) for f in drive(g, level, presses)]
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"    seed {seed} L{level} rot{k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    ref[key] = frames
                    continue
                if any(not np.array_equal(a, b) for a, b in zip(ref[key], frames)):
                    print(f"    seed {seed} L{level} rot{k}: {tag} frames do not "
                          f"un-rotate to the same board")
                    bad += 1
    missing = sum(len(v) for v in pending.values())
    if verbose:
        print(f"  {len(drawn)} (level, rotation) pairs drawn over {seeds} seeds; "
              f"{len(levels) * 4 - missing}/{len(levels) * 4} compared"
              + (f"; {missing} never drawn" if missing else ""))
    return bad + missing


if __name__ == "__main__":
    if "--plans" in sys.argv:
        if "--all" in sys.argv:
            # Re-derive the levels `skip_levels` lists, which is how a future
            # search improvement gets checked against them.
            TowersOfSaigonSolver.skip_levels = frozenset()
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
    sys.exit(TowersOfSaigonSolver.main())
