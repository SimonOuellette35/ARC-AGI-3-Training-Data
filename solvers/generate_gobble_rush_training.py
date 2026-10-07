"""Generate Phase-1 training data for the PuzzleScript game ps:gobble_rush
("Gobble Rush!", Mark Richardson).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: a native model of the enemy-charge mechanic, the exact
distance-to-win field that plans over it, and the checks that pin the model to
the interpreter.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_gobble_rush",
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
exactly. Each step also carries the full set of equally-optimal presses.

The game
--------
Reach the Exit without being seen. The player walks one cell per press (there is
no waiting: ``require_player_movement`` reverts a turn in which the player did
not move, and ``noaction`` makes ACTION5 a no-op, so the four directions are the
whole action space). Enemies never move on their own. An enemy that shares a row
or a column with the player, with no WALL strictly between them, is ALERTED: it
grows a red "!" and, on the player's NEXT press, charges in a straight line
towards where the player was standing, one cell per ``again`` tick, until a wall
or a stopped enemy blocks it. Anything it runs over dies -- lettuce, chickens,
and the player.

So the shape of the game is: you get exactly one turn of warning, and you spend
it stepping out of the lane. The charge resolves entirely inside the press that
follows it, and while anything is moving no new alert can be raised
(``[Move][CanMove] -> [Move][]`` wipes the marker the line-of-sight rules need),
which is what keeps a charge to one straight line rather than a homing missile.

The other pieces:

  * **Exit** -- the goal. The win condition is ``some Exit on Player`` AND
    ``no Alert`` AND ``no Move``, i.e. **standing on the exit is not enough: you
    must arrive unseen.** Every level here has exactly one exit and several
    levels place enemies with a clear line to it, so the last move is usually
    the constrained one.
  * **Lettuce** -- a wall to the player (it shares the player's collision layer,
    so walking into it is refused) but not to a charging enemy, which ploughs
    through and destroys it. It does not block line of sight. Levels 14-17 and
    19-20 are built out of it: the maze is carved by luring charges.
  * **Chicken** -- a second, stationary source of line of sight. Enemies charge
    at chickens exactly as they charge at the player, and an enemy in line with
    BOTH is alerted towards the chicken (see "the LoS flood erases Move" below).
    A chicken standing in line with an enemy therefore keeps the level from ever
    being won until the enemy eats it.
  * **Hole** -- refuses the player's move via ``[ > Player | Hole ] -> cancel``,
    which is a *main-rule* cancel and so reverts the whole turn before any
    enemy has moved. That makes a hole strictly safer than a wall: see below.

Levels 3, 5, 13 and 15 already have an alerted enemy on the start frame
(``run_rules_on_level_start``), so the first press of those levels resolves a
charge.

Four mechanics that are not readable off the .txt
------------------------------------------------
All four were found by pinning the model against the interpreter -- the first
three by the fuzz, the fourth by replaying each plan on a rotated copy of its own
board -- and each one changes which levels are winnable or how they may be
recorded.

1. **Walking into a wall is not a free pass, but walking into a hole is.**
   ``require_player_movement`` is checked at the very END of the turn: the whole
   ``again`` loop runs first, so a blocked press still lets every alerted enemy
   complete its charge, and the turn is reverted only if the player is still
   standing where it started. If a charge ate the player, the position DID
   change, there is no revert, and the player is dead. The hole rule is the
   opposite: ``cancel`` fires among the main rules and reverts immediately, so
   nothing charges at all. Pressing into a hole is the game's only "wait", and
   only three levels have one.

2. **The line-of-sight flood ERASES an enemy's charge direction.** ``LoS`` and
   the four ``Move`` markers share one collision layer, so painting ``LoS`` onto
   an enemy's cell deletes whatever ``Move`` it was carrying. The player's flood
   runs first and is harmless (nothing is moving at that point, by construction),
   but the chicken loop that follows re-floods the board once per chicken, and
   every enemy on a chicken's ray therefore has its player-assigned direction
   wiped and immediately re-assigned towards that chicken. Net effect, and this
   is the rule the model encodes: **a chicken beats the player, and the last
   chicken drawn beats the earlier ones.** Without it the model disagreed with
   the interpreter on level 12, which stands a chicken two cells below an enemy
   the player is standing directly above.

3. **The chicken loop is the game's only nondeterminism, and it never bites
   here.** ``late random [ Chicken CanMove ] -> [ Chicken LoS ]`` picks ONE
   chicken per iteration of its ``startloop``, and the loop runs its full 200
   iterations (the flood-then-clear pair always moves the mutation counter), so
   which chicken is drawn LAST decides the direction of every enemy that more
   than one chicken can see. `_Board.step` reports such a state as ambiguous and
   the search drops the transition un-expanded. Measured over the complete
   reachable space of all 21 levels: **zero ambiguous transitions**. Two chickens
   never share an enemy's sightline on any board this game can reach.

4. **Two charges that collide resolve CHIRALLY, so the mechanic is not
   rotation-invariant.** `_execute_late_single` drives each of a rule's four
   directional copies to a fixpoint IN TURN (up, down, left, right), and the
   four "move enemies" rules are four separate rules. So when two charges want
   the same cell -- head-on with one cell between them, or perpendicular -- the
   one whose direction is expanded EARLIER takes it and the other is left
   standing. "The leftward charge beats the rightward one" is a fact about the
   screen, not about the board, and under the 180-degree rotation the frames are
   presented at it reads as the opposite.

   This is not fixable in the .txt without changing the game: every symmetric
   alternative (both stop, both move) alters what happens when the player is the
   cell being contested, which is exactly the situation the levels are built
   around. So it is measured instead. `_Board.step` re-runs the turn under each
   of the 8 rule orders a symmetry of the square would present
   (`_SYMMETRY_ORDERS`) and reports the transition as CHIRAL when any disagrees;
   the recomputation is skipped unless two enemies were charging at once, which
   is the only way it can happen.

   The expert then STEERS AWAY from chirality rather than refusing it: `plan`
   takes a chiral press only when every equally-shortest press is chiral. That
   leaves **6 of 378 labelled presses** chiral, on levels 2, 17, 18 and 19.
   Refusing them outright was tried and measured, and it is a worse corpus:
   levels 17 and 19 become unwinnable (both open with a collision the player
   cannot avoid) and levels 2 and 18 lose their shortest plans (19 -> 23 and
   23 -> 27), i.e. two whole boards and the optimality guarantee traded for
   1.6% of the labels.

A fifth thing the model checks for and never finds: a charge DEADLOCK. Three
enemies charging in a cycle would each be blocked by a mover, so no stop rule
fires, nothing moves, the ``again`` loop ends -- and they keep their ``Move``
markers while ``Alert`` (cleared earlier in the same tick) is gone. That state
carries hidden information: an enemy about to charge with no "!" drawn on it.
`_Board.step` flags it and the search refuses it, so nothing this expert plans
can depend on state the frame does not show. It does not occur in any reachable
state of any level.

Expert solver
-------------
Not a search over the interpreter -- ``eng.step`` runs at ~400-1500 presses/s
here (the chicken ``startloop`` alone costs 200 iterations a turn), and a plain
BFS over engine states does not finish level 3. `_Board` is a native model of
the whole rule set instead, and it is ~100x faster, so every level gets an
EXACT distance-to-win field rather than a heuristic search:

  * a forward BFS from the level start enumerates the reachable settled states
    (fatal, refused, ambiguous and hidden-state successors are all dropped);
  * a reverse BFS from the winning states over the recorded predecessors gives
    the true distance-to-win of every state that can still win.

A reverse field cannot be built the way a Sokoban's is: a charge destroys the
lettuce and chickens it runs over, so there is no pull-move to invert. Hence the
predecessor map, and hence ``--verify``, which re-derives every plan length and
every tie set by an independent forward search that shares no code with it.

That buys the two things a forward A* would not: **every plan is proved
SHORTEST**, not merely found, and the **optimal-action SETS are measured** -- at
distance ``d``, a press is optimal iff it lands on a state at distance ``d-1``.

The interpreter is then only asked to CERTIFY: ``--plans`` replays each field
plan through the real engine and requires the win on the LAST press and no
earlier, and the recorder drives that same engine for every taped frame, so a
model that disagreed with PuzzleScript could not produce a WIN episode.

The levels
----------
All 21 shipped boards are 7x11 and all 21 are winnable; nothing was authored or
skipped. 328 presses in total, every one of them proved shortest; 45 of the 328
steps have more than one equally-optimal press (378 labelled presses in all). Field sizes, plan lengths and tie
coverage are printed by ``--plans``. The longest plan is 30 presses against the
adapter's 200-step per-level budget, so this game needs no ``games/`` step-limit
wrapper.

The four checks, and what each one can and cannot see:

  * ``--plans`` builds every field and replays its plan through the real
    interpreter, requiring the win on the LAST press and no earlier. That is the
    only check that can catch a plan which is valid but not shortest.
  * ``--fuzz`` compares `_Board` against the interpreter over 55498 transitions
    (5657 of them resolving a charge, 6855 fatal, 16359 refused, 627 lettuce and
    586 chickens eaten) by fuzzing from EVERY PREFIX of each level's own
    solution. Random play from the start is not enough: it walks into a wall or
    gets eaten within a few presses, and would exercise none of the convoy
    charges the later levels are built on. The coverage counters are printed for
    exactly that reason.
  * ``--verify`` re-derives every plan length and every tie set with an
    independent depth-bounded forward search that shares no code with the
    predecessor map the field is built from.
  * ``--symmetry`` replays each plan on all 8 turned and mirrored copies of its
    own board THROUGH THE INTERPRETER and asserts that `_Board.step`'s chirality
    flag names exactly the presses that diverge. It is the check that found
    mechanic 4, and the only one that can: the fuzz plays one board, where the
    model and the interpreter agree on a chirality neither of them can see.
  * ``--audit`` renders every cell composition at every cell size in use and
    asserts pairwise distinctness.

Rendering
---------
``--audit`` renders every cell COMPOSITION the levels can show and asserts they
are pairwise distinct, comparing WHOLE frames rather than slicing a cell out by
``cell_px`` arithmetic (`_render_frame` upscales the board to fill 64x64 and
letterboxes it, so the arithmetic crop reads the wrong window). The win
composition -- player standing on the exit -- is checked first, and it survives:
the Exit's checkerboard is fully opaque and the Player sprite is transparent at
its corners, so the exit shows through exactly where it matters. The Alert "!"
is painted on the three cells of column 4 where the Enemy sprite is transparent,
which is what makes "this enemy is about to charge" readable at all.

Augmentation
------------
The engine state after reset is identical for every seed (the levels are fixed
ASCII maps), so the only per-(seed, level) variables are presentation: the frame
rotation plus an independent horizontal and vertical flip, each with the matching
directional action remap. This game was added to
`PuzzleScriptAdapter._FLIP_GAMES`, giving 21 levels x 16 presentations = 336.

The flips need an argument, because point 4 above says the mechanic is not
symmetric. It is this: the asymmetry is ALREADY fully exposed by the rotation,
which is mandatory and was on before this generator existed. Measured over the
complete reachable space of all 21 levels, of 1009128 transitions **15881 are
settled by the rule order, and every single one of them is split by a ROTATION
-- not one is split only by a MIRROR**. So the flips add no inconsistency the
corpus does not already carry, and they quadruple the presentations. Nothing else
objects: the game is gravity-free, its win condition names no direction, its
input is screen-relative, and no sprite in it encodes a facing.

The expert plan is therefore seed-independent: solved once per level, cached to
``data/gobble_rush_plans.json``, and replayed per seed with that seed's remapped
screen actions.

Usage (run from the repo root):
    python solvers/generate_gobble_rush_training.py \
        --episodes 200 --out data/training_multi_level/gobble_rush

    python solvers/generate_gobble_rush_training.py --plans
    python solvers/generate_gobble_rush_training.py --verify
    python solvers/generate_gobble_rush_training.py --fuzz
    python solvers/generate_gobble_rush_training.py --symmetry
    python solvers/generate_gobble_rush_training.py --audit
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

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan      # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Gobble_Rush!"

#: Disk cache of each level's start plan AND its optimal-action sets. The fields
#: are seed-independent and cost ~80s together (levels 3 and 19 are nearly all of
#: it -- the chirality check triples them), which every shard of
#: `parallelize_generator` would otherwise repeat on every core. Delete the file
#: to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "gobble_rush_plans.json"

#: Engine direction names, in the order the interpreter expands a rule's four
#: directional copies. `_Board` mirrors that order exactly -- which enemy of a
#: convoy moves first inside one tick depends on it -- so this tuple is part of
#: the model, not a display convention.
_ORDER: tuple[str, ...] = ("up", "down", "left", "right")
_DELTA: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
#: Direction code -> the code pointing back the other way.
_OPP: tuple[int, ...] = (1, 0, 3, 2)
_CODE: dict[str, int] = {name: i for i, name in enumerate(_ORDER)}

#: The rule order under each of the 8 symmetries of the square, as the sequence
#: of ORIGINAL-frame directions the interpreter would drive to a fixpoint if the
#: board were presented turned or mirrored. `_Board` uses these to detect the
#: transitions whose outcome is CHIRAL -- see `_Board.step`. The first entry is
#: the identity, i.e. the real interpreter's own order.
_SYMMETRY_ORDERS: tuple[tuple[int, ...], ...] = (
    (0, 1, 2, 3),        # identity
    (3, 2, 0, 1),        # rotate 90
    (1, 0, 3, 2),        # rotate 180
    (2, 3, 1, 0),        # rotate 270
    (0, 1, 3, 2),        # mirror left-right
    (1, 0, 2, 3),        # mirror top-bottom
    (2, 3, 0, 1),        # mirror on the main diagonal
    (3, 2, 1, 0),        # mirror on the anti-diagonal
)

#: No charge can outlast the board (7x11), so the interpreter's own ``max_again``
#: budget of 50 ticks is never approached; this asserts that rather than assuming
#: it, because a truncated charge would be a silent model/engine divergence.
TICK_CAP = 50

#: Refuse to build a field larger than this. It is a MEMORY budget -- every state
#: is a tuple of tuples plus its predecessor list -- and it exists so an edited
#: level fails loudly instead of swapping. The largest level here builds 283k.
STATE_CAP = 4_000_000


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """Gobble Rush's whole rule set, over integer cell indices.

    A state is ``(player, enemies, chickens, lettuce)``:

      * ``player``   -- cell index, or -1 once the player has been eaten;
      * ``enemies``  -- sorted tuple of ``(cell, direction code)`` pairs, the
        code being -1 for an enemy that is standing still and 0-3 for one
        carrying a ``Move`` marker, i.e. one that will charge on the next press;
      * ``chickens`` / ``lettuce`` -- sorted tuples of cell indices. Both shrink
        only: a charge destroys what it runs over and nothing puts it back.

    Walls, holes and exits are static -- no rule in this game creates or destroys
    any of them -- so they live on the board rather than in the state, and the
    line-of-sight RAYS (which only walls block) are precomputed once.

    ``Alert`` is not stored: the interpreter clears it at the top of every tick
    and re-raises it together with ``Move``, so at a settled board the two are
    the same bit. Neither is ``CanMove`` (bookkeeping, invisible) or ``LoS``
    (cleared before the turn ends).
    """

    def __init__(self, h, w, walls, holes, exits, start):
        self.h, self.w = h, w
        n = h * w
        self.walls, self.holes, self.exits = walls, holes, exits
        self.nbr = [[-1] * 4 for _ in range(n)]
        for r in range(h):
            for c in range(w):
                cell = r * w + c
                for d, (dr, dc) in enumerate(_DELTA):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        self.nbr[cell][d] = rr * w + cc
        # The sight line in each direction: every cell from here up to (and not
        # including) the first wall. Only Wall blocks LoS -- enemies, chickens,
        # lettuce and holes are all see-through -- and walls never move.
        self.ray = [[() for _ in range(4)] for _ in range(n)]
        for cell in range(n):
            if cell in walls:
                continue
            for d in range(4):
                line, cur = [], cell
                while True:
                    cur = self.nbr[cur][d]
                    if cur < 0 or cur in walls:
                        break
                    line.append(cur)
                self.ray[cell][d] = tuple(line)
        self.start = start

    @classmethod
    def read(cls, eng, ids) -> "_Board":
        """Build the model, and its start state, from the interpreter's grid."""
        wall, hole, exit_, player, enemy, chicken, lettuce, moves = ids
        w = eng.width
        walls, holes, exits = set(), set(), set()
        p, en, ch, le = -1, [], [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i = r * w + c
                if cell & wall:
                    walls.add(i)
                if cell & hole:
                    holes.add(i)
                if cell & exit_:
                    exits.add(i)
                if cell & player:
                    p = i
                if cell & chicken:
                    ch.append(i)
                if cell & lettuce:
                    le.append(i)
                if cell & enemy:
                    md = -1
                    for d, oid in enumerate(moves):
                        if oid in cell:
                            md = d
                    en.append((i, md))
        start = (p, tuple(sorted(en)), tuple(sorted(ch)), tuple(sorted(le)))
        return cls(eng.height, w, walls, holes, exits, start)

    # -- dynamics -------------------------------------------------------------
    def win(self, state) -> bool:
        """``some Exit on Player`` and ``no Alert`` and ``no Move``: on the exit,
        with nothing alerted. Standing on the exit while an enemy has a line to
        you (or to a chicken) is not a win."""
        p, en, _ch, _le = state
        return p in self.exits and all(md < 0 for _c, md in en)

    def step(self, state, direction):
        """One press. Returns ``(next state, flags)``, where the next state is
        None if the player was eaten and ``state`` itself if the interpreter
        refused the turn, and ``flags`` is an ``(ambiguous, hidden, chiral)``
        triple naming the three ways a transition is unusable -- see the module
        docstring.

        ``chiral`` is decided here rather than in `_run`: the outcome is
        recomputed under each of the 8 rule orders a symmetry of the square
        would present (`_SYMMETRY_ORDERS`), and the transition is chiral when
        any of them disagrees. That can only happen while two or more enemies
        are charging at once, so `_run` reports whether it ever saw that and
        the recomputation is skipped -- which is nearly always -- when it did
        not."""
        code = _CODE[direction] if isinstance(direction, str) else direction
        nxt, ambiguous, hidden, contended = self._run(state, code,
                                                      _SYMMETRY_ORDERS[0])
        chiral = False
        if contended:
            for order in _SYMMETRY_ORDERS[1:]:
                if self._run(state, code, order)[0] != nxt:
                    chiral = True
                    break
        return nxt, (ambiguous, hidden, chiral)

    def _run(self, state, code, order):
        """The turn itself, with the stop/move rules driven to a fixpoint in
        ``order``. Returns ``(next state, ambiguous, hidden, contended)``.

        The body is the interpreter's rule list, in its order:

          1. ``[ > Player | Hole ] -> cancel`` -- a main-rule cancel, so it
             reverts the turn immediately and nothing else in it happens.
          2. the player moves, unless the destination holds a wall or anything
             on its own collision layer (enemy, chicken, lettuce) or is off the
             board. A refused move still runs the rest of the turn; only the
             ``require_player_movement`` check at the very END undoes it.
          3. the ``again`` loop: grant ``CanMove`` to everything carrying a
             ``Move`` marker, stop what is about to hit something, move what is
             not, then -- only if nothing is left moving -- recompute the alerts.
             It ends on the first tick that moved nothing.
          4. ``require_player_movement``: if the player is alive and has not
             moved, the entire turn is reverted.
        """
        p0, en0, ch0, le0 = state
        if p0 < 0:
            return None, False, False, False
        tgt = self.nbr[p0][code]
        if tgt >= 0 and tgt in self.holes:
            return state, False, False, False     # [ > Player | Hole ] -> cancel

        en = dict(en0)
        ch, le = set(ch0), set(le0)
        blocked = (tgt < 0 or tgt in self.walls or tgt in en
                   or tgt in ch or tgt in le)
        p = p0 if blocked else tgt

        ambiguous = alerted = contended = False
        for _tick in range(TICK_CAP):
            # Main rule: [ Enemy Move ] -> [ Enemy CanMove Move ]. A mover that
            # neither stops nor advances this tick keeps its marker but loses
            # CanMove, so it can only ever cover one cell per tick.
            cm = {c for c, md in en.items() if md >= 0}
            contended |= len(cm) > 1
            self._stop(en, cm, order, walls_too=True)
            # The two head-on rules: adjacent enemies charging into each other
            # both give up.
            for d in (1, 3):                      # 'down' and 'right' brackets
                for c in [c for c, md in en.items() if md == d and c in cm]:
                    n = self.nbr[c][d]
                    if n >= 0 and en.get(n, -1) == _OPP[d] and n in cm:
                        en[c] = en[n] = -1
                        cm.discard(c)
                        cm.discard(n)
            self._stop(en, cm, order, walls_too=False)  # cascades those head-ons

            moved = False
            for d in order:
                while True:
                    again = False
                    for c in [c for c, md in en.items() if md == d and c in cm]:
                        n = self.nbr[c][d]
                        if n < 0 or n in self.walls or n in en:
                            continue
                        del en[c]
                        en[n] = d
                        cm.discard(c)
                        again = moved = True
                        if p == n:                # the player was gobbled
                            p = -1
                        ch.discard(n)
                        le.discard(n)
                    if not again:
                        break

            # [ Move ] [ CanMove ] -> [ Move ] [ ]: while anything is still
            # charging, no line of sight is computed and no alert is raised.
            alerted = not any(md >= 0 for md in en.values())
            if alerted:
                ambiguous |= self._alert(p, en, ch)
            if not moved:
                break
        else:                                     # pragma: no cover
            raise AssertionError("charge outlasted the interpreter's again budget")

        if p >= 0 and p == p0:
            return state, False, False, False     # require_player_movement
        if p < 0:
            return None, ambiguous, False, contended
        nxt = (p, tuple(sorted(en.items())), tuple(sorted(ch)), tuple(sorted(le)))
        return nxt, ambiguous, not alerted, contended

    def _stop(self, en, cm, order, walls_too):
        """The ten ``(Stop enemies)`` rules, in their file order: each of the
        four directions driven to a fixpoint in turn, so a convoy whose leader
        hits a wall gives up along its whole length.

        ``walls_too`` is the difference between the first four rules
        (``| Obstacle no Move``, where ``Obstacle`` is Wall or Enemy) and the
        last four (``| Enemy no Move``). The second batch is redundant on its own
        and exists to re-cascade whatever the head-on rules between them stopped.
        Note the ``no Move`` guard: an enemy that is itself charging is NOT a
        blocker, which is what lets a convoy travel together."""
        for d in order:
            while True:
                changed = False
                for c in [c for c, md in en.items() if md == d and c in cm]:
                    n = self.nbr[c][d]
                    if n < 0:
                        continue                  # off the board: no cell, no match
                    if en.get(n, 0) < 0 or (walls_too and n in self.walls):
                        en[c] = -1
                        cm.discard(c)
                        changed = True
                if not changed:
                    break

    def _alert(self, p, en, ch) -> bool:
        """Raise the alerts for a settled board; returns True if the result was
        decided by the chicken loop's RNG.

        The player floods first and every enemy on one of its four rays is sent
        back down that ray. Then each chicken floods in turn, and because the
        flood's ``LoS`` marker shares a collision layer with the ``Move`` marker
        it lands on, a chicken OVERWRITES whatever direction an enemy already
        had -- the player's assignment included. So the chicken pass is applied
        second and unconditionally, and the transition is ambiguous exactly when
        two chickens would send the same enemy in different directions (the
        interpreter's answer is then whichever was drawn last)."""
        if p >= 0:
            for d in range(4):
                for c in self.ray[p][d]:
                    # 0 is the "not an enemy" default AND a live direction code,
                    # so this reads as "an enemy that is standing still".
                    if en.get(c, 0) < 0:
                        en[c] = _OPP[d]
        seen: dict[int, set[int]] = {}
        for k in ch:
            for d in range(4):
                for c in self.ray[k][d]:
                    if c in en:
                        seen.setdefault(c, set()).add(_OPP[d])
        ambiguous = False
        for c, dirs in seen.items():
            ambiguous |= len(dirs) > 1
            en[c] = min(dirs)
        return ambiguous

    # -- the exact distance-to-win field --------------------------------------
    def field(self, cap: int = STATE_CAP) -> dict:
        """``{state: presses to the win}`` for every reachable state that can
        still win.

        Two passes, because this game's transition relation is not invertible --
        an enemy's charge destroys lettuce and chickens, so there is no pull-move
        to run a reverse BFS over the way a Sokoban has. Instead a forward BFS
        enumerates the reachable settled states and records each one's
        predecessors, and a reverse BFS over those gives the exact
        distance-to-win. Distances are in PRIMITIVE PRESSES, the unit the agent
        pays, and nothing is canonicalised or merged.

        Four kinds of successor never enter the graph:

          * ``None`` -- the player was eaten;
          * ``state`` itself -- the interpreter refused the press (a wall, a
            piece, the board edge, or a hole's cancel). A refused press cannot
            help and would only add self-loops;
          * ambiguous -- the chicken loop's RNG decided it;
          * hidden -- it settles with enemies still carrying ``Move`` but no
            ``Alert`` drawn, i.e. it carries state the frame does not show.

        CHIRAL successors are kept, and `plan` steers away from them instead.
        Refusing them outright was measured and is a bad trade: it removes 6 of
        378 labelled presses and costs levels 17 and 19 entirely (both start
        with a collision the player cannot avoid), while lengthening levels 2
        and 18 past shortest.
        """
        pred: dict = {}
        seen = {self.start}
        queue = deque([self.start])
        wins = []
        counts = [0, 0, 0]                       # ambiguous, hidden, chiral
        while queue:
            state = queue.popleft()
            if self.win(state):
                wins.append(state)
                continue                         # the level is over
            for d in range(4):
                nxt, flags = self.step(state, d)
                for i, flag in enumerate(flags):
                    counts[i] += flag
                if nxt is None or nxt == state or flags[0] or flags[1]:
                    continue
                pred.setdefault(nxt, []).append(state)
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
            if len(seen) > cap:
                raise MemoryError(f"field exceeded {cap} states")
        dist = {state: 0 for state in wins}
        queue = deque(wins)
        while queue:
            state = queue.popleft()
            d = dist[state] + 1
            for prev in pred.get(state, ()):
                if prev not in dist:
                    dist[prev] = d
                    queue.append(prev)
        self.reachable, self.dropped = len(seen), tuple(counts)
        return dist

    def plan(self, dist: dict):
        """``(presses, optsets, chiral)`` from the start, or ``(None, None, 0)``.

        Walk the field downhill. At every state the OPTIMAL SET is exactly the
        presses that land on distance ``d - 1`` -- a measured tie, not one
        inferred from what kind of move it was.

        Which of the tied presses is TAKEN is where chirality is spent: a press
        whose outcome depends on the board's facing is chosen only when every
        equally-shortest press is one. ``chiral`` counts the steps where that
        happened, so the residue is reported rather than hidden. The set itself
        is left complete and in `_ORDER` -- a chiral press is still an optimal
        press, and dropping it from the label would teach that a shortest move
        is wrong."""
        state = self.start
        if state not in dist:
            return None, None, 0
        presses, optsets, chiral = [], [], 0
        while dist[state]:
            want = dist[state] - 1
            best = []
            for a in _ORDER:
                nxt, flags = self.step(state, a)
                if dist.get(nxt, -1) == want:
                    best.append((a, flags[2]))
            take = sorted(best, key=lambda t: t[1])[0]   # stable: _ORDER order
            chiral += take[1]
            presses.append(take[0])
            optsets.append([a for a, _c in best])
            state = self.step(state, take[0])[0]
        return presses, optsets, chiral


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class GobbleRushExpert(PSExpert):
    """Plans by building `_Board`'s exact distance-to-win field off the engine
    grid and walking it downhill; never steps the interpreter.

    `PSExpert` still owns everything around that -- the in-memory memo, the
    on-disk plan cache with its staleness check, the level scoping and the
    snapshot/restore discipline -- so the only override is `_search`. `heuristic`
    is unreachable by construction: nothing here runs A*.
    """

    #: `_key` below is dynamic-objects-only, canonical only WITHIN a level.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        g = self.g
        name = g.resolve_object_name
        self.wall_ids = set(name("wall"))
        self.hole_ids = set(name("hole"))
        self.exit_ids = set(name("exit"))
        self.player_ids = set(self.game._engine._player_indices)
        self.enemy_ids = set(name("enemy"))
        self.chicken_ids = set(name("chicken"))
        self.lettuce_ids = set(name("lettuce"))
        self.move_ids = tuple(g.obj_name_to_idx["move" + d[0]] for d in _ORDER)
        # Everything a settled frame can differ by. CanMove and LoS are left out
        # deliberately: LoS is always cleared before a turn ends, and CanMove
        # rides on the player/chickens purely as bookkeeping, so including
        # either would split states the game cannot tell apart.
        self.dyn_ids = (self.player_ids | self.enemy_ids | self.chicken_ids
                        | self.lettuce_ids | set(self.move_ids))

    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    def heuristic(self, eng) -> int:
        raise AssertionError("the field is exact; no search runs here")

    def board(self, eng) -> _Board:
        return _Board.read(eng, (self.wall_ids, self.hole_ids, self.exit_ids,
                                 self.player_ids, self.enemy_ids,
                                 self.chicken_ids, self.lettuce_ids,
                                 self.move_ids))

    def _search(self, eng) -> list | None:
        board = self.board(eng)
        presses, optsets, _chiral = board.plan(board.field())
        return None if presses is None else Plan(presses, optsets)


class GobbleRushSolver(PSAStarSolver):
    game_id = "puzzlescript_gobble_rush"
    game_name = GAME_NAME
    expert_cls = GobbleRushExpert

    #: The longest plan is 30 presses; the rest is room for the exploration
    #: prefix and the re-plan after it, well inside the adapter's per-level
    #: budget.
    max_steps = 120

    def prepare_expert(self, game, expert) -> None:
        """Build every level's field before `discover_solvable` asks for it --
        the same work either way, but it makes the one-time cost visible as
        startup rather than as a mysteriously slow first seed, and it fills the
        disk cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with the adapter parked at level 0."""
    solver = GobbleRushSolver()
    game = solver.make_game(0)
    return solver, game, GobbleRushExpert(game)


def _report() -> int:
    """Per-level piece counts, field size, plan length and tie coverage -- and
    CERTIFY each plan by replaying it through the interpreter."""
    _solver, game, expert = _levels()
    total = bad = amb = hid = chi = forced = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        _p, en, ch, le = board.start
        t = time.time()
        dist = board.field()
        presses, optsets, chiral = board.plan(dist)
        took = time.time() - t
        amb += board.dropped[0]
        hid += board.dropped[1]
        chi += board.dropped[2]
        if presses is None:
            print(f"level {level:2d}: UNSOLVABLE")
            bad += 1
            continue
        # Certification: the interpreter must win on the LAST press and no
        # earlier (an earlier win would mean the plan is not shortest).
        won_at = None
        for i, direction in enumerate(presses):
            eng.step(direction)
            if eng.check_win():
                won_at = i
                break
        ok = won_at == len(presses) - 1
        bad += not ok
        total += len(presses)
        ties = sum(1 for s in optsets if len(s) > 1)
        forced += chiral
        room = "ok" if len(presses) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {len(en)}E {len(ch)}C {len(le):2d}L, "
              f"{len(presses):3d} presses (budget {game._max_steps}, {room}), "
              f"{ties:2d} tie steps ({ties / len(presses):3.0%}), "
              f"{chiral} chiral, "
              f"field {board.reachable:6d} states in {took:5.2f}s, "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    print(f"total {total} presses over {game.n_levels} levels; dropped "
          f"{amb} ambiguous and {hid} hidden-state transitions; "
          f"{chi} of the transitions explored are chiral and {forced} plan "
          f"steps had no non-chiral alternative")
    return 0 if not bad else 1


def _verify() -> int:
    """Double-entry check of every plan length AND every tie set.

    `_Board.field` derives its answer backwards, from win states over a
    predecessor map built during the forward sweep. That bookkeeping is the one
    piece of logic here the interpreter fuzz cannot reach -- the fuzz only ever
    exercises `step`. So the answer is recomputed from the other end: for every
    successor of every state on a plan, an independent breadth-first search
    counts the presses from THAT state to a win, sharing no code with the field
    beyond `step` itself. A press is optimal iff its own search answers
    ``d* - i - 1``; every other press must answer strictly more (or never win).

    Slower than everything else here, because a state deep inside level 19's
    283k-state space costs a near-complete BFS per press."""
    _solver, game, expert = _levels()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine)
        dist = board.field()
        presses, optsets, _chiral = board.plan(dist)
        star = dist[board.start]
        t = time.time()
        notes = []
        if star != len(presses):
            notes.append(f"LENGTH {star} != {len(presses)}")
        state = board.start
        for i, direction in enumerate(presses):
            want = []
            for a in _ORDER:
                nxt, amb, hid, _cont = board._run(state, _CODE[a],
                                                  _SYMMETRY_ORDERS[0])
                if nxt is None or nxt == state or amb or hid:
                    continue
                if _independent(board, nxt, star - i - 1) == star - i - 1:
                    want.append(a)
            if want != optsets[i]:
                notes.append(f"step {i}: {optsets[i]} != {want}")
            state = board.step(state, direction)[0]
        bad += len(notes)
        print(f"level {level:2d}: d*={star:3d}, "
              f"{sum(len(s) for s in optsets):3d} labelled presses re-derived "
              f"in {time.time() - t:6.2f}s: "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _independent(board: _Board, state, limit: int) -> int | None:
    """Presses from ``state`` to a win, by a plain forward BFS that stops at the
    first winning state, or None if no win is reachable within ``limit``.

    Deliberately naive and predecessor-free: it is the second book in `_verify`'s
    double entry. It calls `_Board._run` rather than `_Board.step`, both because
    the chirality re-runs are irrelevant to a distance (`field` keeps chiral
    successors) and because that makes it share strictly less code with the
    thing it is checking. The ``limit`` is what makes it affordable: a press is
    convicted by showing no win at the expected depth, which never needs a
    deeper search than that."""
    if board.win(state):
        return 0
    seen = {state}
    frontier = [state]
    for depth in range(1, limit + 1):
        nxt_frontier = []
        for cur in frontier:
            for d in range(4):
                nxt, amb, hid, _cont = board._run(cur, d, _SYMMETRY_ORDERS[0])
                if nxt is None or nxt == cur or amb or hid or nxt in seen:
                    continue
                if board.win(nxt):
                    return depth
                seen.add(nxt)
                nxt_frontier.append(nxt)
        frontier = nxt_frontier
    return None


#: Grid transform -> the direction relabelling that goes with it. A rotation
#: does not only move the cells: an enemy carrying ``MoveU`` on the reference
#: board carries ``MoveR`` on the board turned 90 degrees clockwise, and a
#: comparison that forgets this reports a divergence on every alerted enemy.
_ROT_D: dict[str, str] = {"up": "right", "right": "down",
                          "down": "left", "left": "up"}
_FLIP_D: dict[str, str] = {"up": "up", "down": "down",
                           "left": "right", "right": "left"}


def _tdir(direction: str, k: int, mirror: bool) -> str:
    for _ in range(k):
        direction = _ROT_D[direction]
    return _FLIP_D[direction] if mirror else direction


def _tgrid(grid, k: int, mirror: bool, move_ids):
    """``grid`` rotated ``k`` quarter-turns clockwise, then optionally mirrored
    left-right, with every ``Move`` marker relabelled to match."""
    for _ in range(k):
        h = len(grid)
        grid = [[grid[h - 1 - r][c] for r in range(h)]
                for c in range(len(grid[0]))]
    if mirror:
        grid = [list(reversed(row)) for row in grid]
    out = []
    for row in grid:
        new_row = []
        for cell in row:
            c = set(cell)
            for d, oid in move_ids.items():
                if oid in cell:
                    c.discard(oid)
                    c.add(move_ids[_tdir(d, k, mirror)])
            new_row.append(c)
        out.append(new_row)
    return out


def _symmetry() -> int:
    """Confirm, ON THE INTERPRETER, exactly which plan steps are CHIRAL.

    This is the check that found mechanic 4, and it is the only one here that
    can: the fuzz compares the model to the interpreter on one board, and both
    agree on a chirality neither can see. So instead each level's plan is
    replayed on all 8 turned and mirrored copies of its own board, and the
    transformed run is compared against the transform of the reference run,
    press by press. The first press where any presentation disagrees is the
    first press whose outcome depends on the way the board is facing.

    What is asserted is that `_Board.step`'s ``chiral`` flag predicts that press
    EXACTLY -- same index, or no divergence and no flag. A model that
    over-reported would cost plan length for nothing; one that under-reported
    would let the expert lean on the screen's orientation without saying so.
    Divergence itself is expected and is not a failure; see the module docstring
    for why it is neither fixable nor worth refusing.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    move_ids = {d: g.obj_name_to_idx["move" + d[0]] for d in _ORDER}
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        presses, _optsets, _chiral = board.plan(board.field())

        # What the model says: the first plan step it calls chiral.
        state, predicted = board.start, None
        for i, direction in enumerate(presses):
            nxt, flags = board.step(state, direction)
            if flags[2] and predicted is None:
                predicted = i
            state = nxt

        # What the interpreter says: replay on every presentation.
        eng.load_level(g.levels[level])
        ref = [[[frozenset(c) for c in row] for row in eng.grid]]
        for direction in presses:
            eng.step(direction)
            ref.append([[frozenset(c) for c in row] for row in eng.grid])
        measured = None
        for k in range(4):
            for mirror in (False, True):
                if (k, mirror) == (0, False):
                    continue
                eng.load_level(_tgrid(g.levels[level], k, mirror, move_ids))
                want = [_tgrid(x, k, mirror, move_ids) for x in ref]
                for i, direction in enumerate(presses):
                    eng.step(_tdir(direction, k, mirror))
                    if eng.grid != want[i + 1]:
                        measured = i if measured is None else min(measured, i)
                        break
        ok = measured == predicted
        bad += not ok
        print(f"level {level:2d}: {len(presses):3d} presses, first chiral step "
              f"predicted {predicted}, measured {measured}: "
              f"{'AGREES' if ok else 'MODEL DISAGREES WITH THE INTERPRETER'}")
    print("symmetry check clean" if not bad
          else f"SYMMETRY FAILED: {bad} levels")
    return 0 if not bad else 1


def _fuzz(trials: int = 25, walk: int = 20) -> int:
    """Assert `_Board` reproduces the interpreter exactly.

    Random play from the level START is not enough here: it wanders into a wall
    or gets eaten within a few presses, so the long convoy charges and the
    lettuce-clearing that the later levels are made of would go unexercised.
    Fuzz from EVERY PREFIX of each level's own solution instead, which puts the
    model in the configurations the plans actually visit. Coverage counters are
    printed for the same reason -- a run that reports agreement while having
    triggered no charge has checked nothing.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    rng = random.Random(20260813)
    bad = 0
    totals = dict(presses=0, refused=0, charges=0, eaten=0, veg=0, birds=0,
                  holes=0, deaths=0)
    for level in range(game.n_levels):
        game.set_level(level)
        layout = g.levels[level]
        board = expert.board(eng)
        presses, _optsets, _chiral = board.plan(board.field())
        seen = dict(totals)
        for prefix in range(len(presses) + 1):
            for _ in range(trials):
                eng.load_level(layout)
                state = board.start
                for direction in presses[:prefix]:
                    eng.step(direction)
                    state = board.step(state, direction)[0]
                for _ in range(walk):
                    if state is None:
                        break
                    direction = rng.choice(_ORDER)
                    before = state
                    eng.step(direction)
                    # `_run` at the interpreter's own rule order, not `step`:
                    # the chirality re-runs answer a question about the OTHER
                    # seven orders, which no interpreter run can settle, so
                    # paying 8x for them here would buy nothing.
                    state, ambiguous, _hid, _cont = board._run(
                        state, _CODE[direction], _SYMMETRY_ORDERS[0])
                    if ambiguous:
                        break                     # the engine rolled a die
                    totals["presses"] += 1
                    mine = state if state is not None else None
                    theirs = _engine_state(eng, expert)
                    if mine is None:
                        ok = theirs[0] < 0
                        totals["deaths"] += 1
                    else:
                        ok = mine == theirs
                    if not ok:
                        bad += 1
                        if bad <= 3:
                            print(f"  level {level} MISMATCH after {direction}\n"
                                  f"    model  {mine}\n    engine {theirs}")
                        break
                    if state is None:
                        break
                    if board.win(state) != eng.check_win():
                        bad += 1
                        print(f"  level {level} WIN MISMATCH")
                        break
                    if state == before:
                        totals["refused"] += 1
                    if any(md >= 0 for _c, md in before[1]):
                        totals["charges"] += 1
                    totals["veg"] += len(before[3]) - len(state[3])
                    totals["birds"] += len(before[2]) - len(state[2])
                    if board.win(state):
                        break
        moves = totals["presses"] - seen["presses"]
        print(f"level {level:2d}: {moves:6d} presses, "
              f"{totals['charges'] - seen['charges']:5d} resolved a charge, "
              f"{totals['refused'] - seen['refused']:5d} refused, "
              f"{totals['deaths'] - seen['deaths']:4d} deaths, "
              f"{totals['veg'] - seen['veg']:4d} lettuce and "
              f"{totals['birds'] - seen['birds']:3d} chickens eaten")
    print(f"{totals['presses']} transitions "
          f"({totals['charges']} with a charge in flight, {totals['deaths']} "
          f"fatal, {totals['refused']} refused): "
          f"{'model matches the interpreter' if not bad else f'{bad} MISMATCHES'}")
    return 0 if not bad else 1


def _engine_state(eng, expert):
    """The interpreter's grid in `_Board`'s state encoding."""
    w = eng.width
    p, en, ch, le = -1, [], [], []
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            if cell & expert.player_ids:
                p = i
            if cell & expert.chicken_ids:
                ch.append(i)
            if cell & expert.lettuce_ids:
                le.append(i)
            if cell & expert.enemy_ids:
                md = -1
                for d, oid in enumerate(expert.move_ids):
                    if oid in cell:
                        md = d
                en.append((i, md))
    return (p, tuple(sorted(en)), tuple(sorted(ch)), tuple(sorted(le)))


#: Every cell stack the 21 levels can present. The first entry of each pair is
#: the name, the second the objects painted into the cell, bottom layer first.
#: ``player_on_exit`` is the WIN frame and is the one to check first -- it is the
#: composition a colour-only audit is most likely to skip and the one the corpus
#: cannot survive being blind to.
_COMPOSITIONS: dict[str, tuple[str, ...]] = {
    "floor": (),
    "wall": ("wall",),
    "hole": ("hole",),
    "exit": ("exit",),
    "player": ("player",),
    "player_on_exit": ("exit", "player"),
    "enemy": ("enemy",),
    "enemy_alerted": ("enemy", "alert"),
    "enemy_on_exit": ("exit", "enemy"),
    "enemy_alerted_on_exit": ("exit", "enemy", "alert"),
    "enemy_on_hole": ("hole", "enemy"),
    "chicken": ("chicken",),
    "chicken_on_exit": ("exit", "chicken"),
    "lettuce": ("lettuce",),
    "lettuce_on_exit": ("exit", "lettuce"),
    "hud0": ("level0",), "hud1": ("level1",), "hud2": ("level2",),
    "hud3": ("level3",), "hud4": ("level4",), "hud5": ("level5",),
}


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in _COMPOSITIONS.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(GobbleRushSolver.main())
