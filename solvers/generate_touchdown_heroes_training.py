"""Generate Phase-1 training data for the PuzzleScript game
ps:touchdown_heroes_prototype ("Touchdown Heroes (Prototype)", Matt Ventre).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanic, a native model of it, the exact distance FIELD that model is solved
with, the proof that every plan is shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_touchdown_heroes_prototype",
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
American football as a one-play puzzle. You are the QB at the bottom of the
field; the ball is one square in front of you; the endzone is the red band along
the top row. Pick the ball up and carry it in. Eleven defenders across ten
levels are trying to touch you, and **one touch ends the down**: the tackle rule
is `late [ Player | Defense ] -> [ Tackled | Player ] again` followed by
`late [ Tackled ] -> RESTART`, so *ending a turn orthogonally adjacent to any
defender restarts the level*. There is no health, no timer and no undo
(`noundo`), and `noaction` leaves ACTION unbound -- the four arrows are the whole
input alphabet, measured (`_selfcheck` pass 0 presses ACTION and asserts the grid
is untouched).

`require_player_movement` is the second half of the input rule and it is not
cosmetic: **a press that does not move the player reverts the ENTIRE turn**, so
walking into a sideline or a lineman does not cost you a tick and the defenders
do not advance. That makes a blocked press a genuine no-op rather than a wasted
move, which is why no shortest path ever contains one.

THE THREE DEFENDERS, which is the whole puzzle:

  * **Defense** (`[ Parallel Player ] [ Defense ] -> [ Parallel Player ]
    [ Parallel Defense ]`, plus the perpendicular twin) MIRRORS your press
    exactly -- press up and every Defense on the board steps up. Both rules
    resolve to the same thing in this interpreter, because the RHS relative
    direction is read back off the LHS match, which is the player's actual
    force. So a Defense keeps a CONSTANT offset from you and can never catch
    you... until it is stopped by a sideline or a lineman, at which point you
    keep moving and it does not. Every Defense on these levels is beaten by
    running it into the top of the field: it reaches row 0 and stalls there
    while you walk the last stretch.
  * **Blitzer** walks DOWN on every press except LEFT
    (`[ UP Player ] [ Blitzer ] -> [ UP Player ] [ DOWN Blitzer ]`, same for
    DOWN and RIGHT; the LEFT rule's RHS re-states the Blitzer with no force).
    Down is *away* from the endzone, i.e. straight at you, and the ratchet is
    one-way: the only press that does not advance it is LEFT, which is also the
    press that does not advance you up the field.
  * **Stalker** (`[ Stalker | ... | Player ] -> [ > Stalker | ... | Player ]`)
    steps one square toward you whenever it shares your ROW or COLUMN, at any
    distance and through anything -- the `...` matches walls, linemen and other
    stalkers alike (measured: three stalkers in a row all close in, and one
    behind a lineman closes in too). It is stopped only by a body standing in
    the square it is stepping into.

Every defender decides from the board BEFORE the press resolves, and all the
forces resolve together, so a turn is: read the press, hand out the forces, move
everything at once, then check the endzone and then the tackles.

Two more facts that the model had to get exactly right, both of them consequences
of `require_player_movement` being checked AFTER the late rules:

  * a tackle that puts the new Player back on the square the old one occupied
    (a defender walking into the square you just left) leaves the player POSITION
    SET unchanged, so the whole turn -- tackle included -- is reverted and the
    press is simply a no-op. Getting tackled therefore shows up as either a
    RESTART or a silent nothing, depending on geometry;
  * the win rule fires FIRST (`late [ Ball Endzone ] -> [ Celebrate Endzone ]
    again` then `late [ Celebrate Endzone ] -> WIN`), and Celebrate lands on the
    player's own collision layer, so reaching the endzone DELETES the Player
    before the tackle rules are reached. **Walking in beside a defender still
    scores** -- measured, not assumed.

The Ball is alone on its own collision layer, so nothing blocks it and nothing
but you moves it: `[ > Player Ball ] -> [ > Player > Ball ]` gives it your force
whenever you are standing on it. It follows that ball-in-endzone means
you-in-endzone, with one measurable exception the model reproduces: if your press
is BLOCKED while you are carrying, the ball still takes its step (its layer is
empty) and can land in the endzone alone -- and then the win is thrown away
again by `require_player_movement`, because you did not move. `_selfcheck`'s
fuzz found that on a random board before the search ever ran.

Expert solver
-------------
A NATIVE model of everything above (`_Board`) plus an exact distance field
(`_Field`). `PSExpert`'s engine-blackbox A* is not usable at this size (one
interpreter step is ~2 ms, and level 7's search alone touches 11k states) and
`PSPushExpert` does not apply at all -- there is nothing to push.

`_Field` does NOT enumerate. The reachable space of this game is enormous
compared with the answer: level 2's closure is 329,647 states around a 9-press
solution, because you can wander the field indefinitely while the defenders
shuffle around behind you. So the field is the CONE
([[ps-stand-off-solver]]'s construction), which needs a consistent heuristic and
one extra sweep:

  * `heuristic` = walk distance to the ball plus walk distance from the ball to
    the endzone (just the second leg once you are carrying), over the STATIC free
    squares -- sidelines and linemen removed, defenders ignored. Admissible
    because a press moves you one square and only you can carry the ball, and
    consistent because it is a shortest-path metric on a fixed graph. It is also
    tight: 15 against a d* of 15, 16, 17, 18 and 18 on the five hardest levels;
  * pass 1 is A* to the optimal cost L (20 to 11,185 closed states per level,
    0.7 s for the whole game);
  * pass 2 re-walks forward in layers keeping only `g + h <= L`, recording each
    state's successors and the states that have a WINNING press;
  * pass 3 is a backward BFS from those over the recorded edges.

The result is the exact presses-to-win for every state on an optimal path, so
each step's optimal set is just `{d : dist(succ) == dist(s) - 1}` -- exact, with
nothing inferred. A state on an optimal path has `g + dist == L`, so its whole
remaining optimal path is inside the cone; a successor that TIES is on an optimal
path by definition and so is measured exactly, and one that does not tie can only
be over-reported, which cannot add a press to the set.

RECOVERY is the same three passes run from wherever the board actually is: the
expert reads the LIVE engine grid every time it plans, so a perturbed state is
answered exactly, including the answer "no plan". Measured over 600 random walks
of 1 to 14 presses out of the ten level starts: 596 still winnable, 1 provably
lost, 0 hitting the node cap, worst probe 0.67 s.

Shortest, and how that is known
-------------------------------
All ten plans are provably shortest, and the proof is three independent
derivations agreeing on all ten:

  * this file's cone field;
  * a plain FORWARD BFS to the first win over the same model, sharing no line
    with the field and using no heuristic at all (`--selfcheck` pass 3; it closes
    73, 96, 388, 391, 218383, 173845, 9970, 554926, 49051 and 16565 states, which
    is also the measurement that says the cone was necessary). Its per-step tie
    sets are re-derived a second time by re-solving from every candidate press
    (pass 4);
  * the same two sweeps driven by the REAL INTERPRETER (`--engine`): A* and then
    the cone with `PSEngine.step` as the successor function and `check_win` as
    the goal test, no `_Board` call of any kind. It agrees on all ten lengths and
    on every tie set.

The ten plans come to 137 presses, of which 25 stand at a step with a second
equally right answer.

Recovery
--------
`supports_recovery` with the family's RESET-mode prefix, PLUS `epsilon = 0.08`
detours inside the expert replay. The expert re-plans from the LIVE board, so a
detour is answered exactly: the taken action is the mistake and ``optimal`` is
the recovery, which is the signal a policy needs after its own error.

The detour is offered a random alternative press and keeps it only if the board
can still be won, which `record_level` checks by re-planning. Two things make
that sound here rather than merely cheap:

  * a press that gets you TACKLED is not a legal successor of the model at all,
    so it is never a candidate. It is also refused a second time on the way in:
    `_search` returns None for any grid holding a Tackled or Celebrate object,
    which is what an interpreter probe of such a press leaves behind;
  * a press that leaves the level unwinnable is refused because the re-plan
    returns None, and that None is a real proof rather than a budget running out
    (A* exhausts the reachable component; the cap is two orders of magnitude
    above the largest search).

The PREFIX is not given that choice -- it explores freely, which on this game
means it gets tackled -- so it ends in ONE RESET back to the level start, from
which the cached plan is a guaranteed win. The adapter reloads the level on a
RESTART anyway, so a tackled prefix simply finds itself back at the snap.

Measured end to end at ``epsilon = 0.08``: 30 seeds, 30 attempts, 300 levels,
300 WIN, every recording replayed FRAME-EXACT through `perform_action`, max
action index 5 (ACTION only ever from the exploration prefix; no ACTION7 reaches
the buffer), 0 of 4,936 expert steps unlabelled, 8.0% of them a detour whose
``optimal`` is the recovery, longest level recording 38 presses against the
adapter's 200, and the files byte-identical across two runs. 54 seconds for the
30 episodes, of which the ten searches are under a second and cached to
``data/touchdown_heroes_plans.json``.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variable is the frame rotation
(``rotation_k`` in 0..3), with its matching directional action remap.
10 levels x 4 presentations = 40.

**This game is deliberately NOT in `PuzzleScriptAdapter._FLIP_GAMES`, and the
reason is a rule and not a tie-break.** The Blitzer's four rules name ABSOLUTE
directions, and one of them is chiral: `[ LEFT Player ] [ Blitzer ]` is the only
press that does not advance a blitzer. Rotation is orientation-preserving, so
"the press that freezes the blitzer is the one 90 degrees counter-clockwise from
the way to the endzone" is true in all four rotations and readable from the frame
(the endzone band says which way is up-field). A MIRROR swaps that hand while
leaving the endzone band where it is, so the same visible situation would carry
opposite rules in two presentations of the corpus. Rotation alone keeps the game
learnable; the flips would not. (Everything else about the game is
mirror-safe -- contested squares are resolved by blocking every chain that claims
one, so there is no rule-order chirality of the Gobble Rush kind -- but one
chiral rule is enough.)

Rendering
---------
Five sprites had to change, and the first of them is the game itself:
**Defense, Blitzer and Stalker shipped pixel-for-pixel identical.** Same five
colours, same five rows, three completely different threats. Blitzer is now
dark-red with a yellow stance and Stalker keeps the red body under a yellow head;
Defense is untouched, so it still reads as the original. The full statement is
the header comment in
``data/puzzlescript_games/Touchdown_Heroes_(Prototype).txt``.

The other three come from the SIX 16-row levels, which render at 4 pixels per
cell, and `_render_frame`'s 5->4 decimation keeps rows and columns 0, 1, 3, 4 --
it DROPS the middle row and the middle column. The Yardline was a white stripe
down the middle row (so the yard markers were bare turf on levels 4-9), the
Ball's only pixel outside the 3x3 a body covers was its bottom-middle (so
`carrying` and `not carrying` were the SAME 4x4 cell -- this game's one hidden
state bit), and the Lineman was one two-pixel row away from the Player. Only four
pixels are both transparent under every body and kept by the decimation -- the
four corners -- so the ball and the yardline SPLIT them, the ball taking the
leading diagonal and the yardline the anti-diagonal.

``--audit`` renders every cell COMPOSITION a board of this game can hold -- turf,
yardline, endzone, each with and without the ball, each with and without each of
the six bodies that can stand on it, plus the sideline -- as a whole 64x64 frame
of a uniform board, at BOTH board shapes the ten levels use (9x12 and 16x12), and
requires them pairwise distinct. Whole frames rather than one cell sliced out of
a mixed board: `_render_frame` upscales non-uniformly (a 12-wide board spends 5
or 6 pixels per column), so slicing by ``cell_px`` arithmetic reads the wrong
pixels -- the ps:explod lesson.

Reports
-------
    --plans       every level's plan replayed through the real interpreter
    --selfcheck   the model fuzzed against the interpreter, plus the independent
                  BFS, the re-derived tie sets and the recovery sweep
    --engine      d* and the tie sets re-derived with NO native model at all
    --audit       the render audit
    --symmetry    the same presses driven at all four rotations
"""

from __future__ import annotations

import heapq
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                            # noqa: E402
    Plan, PSAStarSolver, PSExpert, restore, snapshot,
)
from adapters.puzzlescript_adapter import _render_frame          # noqa: E402

GAME_NAME = "Touchdown_Heroes_(Prototype)"

#: Seed-independent start plans, so every `parallelize_generator` shard reads the
#: ten searches off disk instead of re-deriving them. See `PSExpert.plan`.
PLAN_CACHE = Path(__file__).resolve().parent.parent / "data" / "touchdown_heroes_plans.json"

#: The four arrows. ACTION is unbound (`noaction`) and is a measured no-op, so it
#: is not in the search alphabet -- branching on it would double every
#: interpreter step in `--engine` for nothing.
DIRS = ("up", "down", "left", "right")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Object classes the model reads. Sideline and Lineman are the immovable bodies
#: (Lineman has no rule that gives it a force -- "they're kind of slow. As in,
#: they don't move"); Defense / Blitzer / Stalker are the three threats.
_BLOCK_NAMES = ("sideline", "lineman")
_FOE_NAMES = ("defense", "blitzer", "stalker")


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the whole mechanic.

    Cells are flat ``r * w + c`` indices. A STATE is
    ``(player, ball, defense, blitzer, stalker)`` -- the player and the ball as
    single cells, the three defender classes as SORTED tuples of cells. Sorted
    because defenders of one class are interchangeable: every rule that reads
    them reads only where they are, so two boards differing by a permutation are
    the same board and must not be searched twice.

    Static here rather than in the state: the sidelines, the linemen (nothing can
    give either a force), the endzone squares and the ball's starting cell. The
    yardlines are not modelled at all -- no rule names Yardline, it is on its own
    collision layer so it blocks nothing, and it is not in any win condition. It
    is scenery, and the only thing this file has to say about it is that it must
    still be VISIBLE (see the module docstring's rendering section).

    THE MECHANIC, as one press in direction ``d``:

      * the PLAYER gets the force ``d``;
      * the BALL gets ``d`` too, if the player is standing on it;
      * every DEFENSE gets ``d`` -- it mirrors you;
      * every BLITZER gets ``down``, unless ``d`` is ``left``;
      * every STALKER that shares your row or column gets the one step toward
        you, at any distance and through anything;
      * everything on the shared collision layer then moves at once
        (`_resolve`), so chains push through, two claimants block each other and
        a follower can take a square that is being vacated;
      * the ball moves on its own empty layer, so nothing can block it;
      * then the late rules: ball in the endzone WINS (and deletes the player, so
        no tackle can be checked afterwards), otherwise a player orthogonally
        adjacent to any defender is TACKLED;
      * and finally `require_player_movement`: if the player's position is
        unchanged the whole turn is thrown away.

    `step` returns one of four statuses, and the search only ever follows ``ok``:

      ``win``     the ball reached the endzone with you on it;
      ``tackle``  the turn ends the down -- either a RESTART, or (when the
                  tackling defender walked into the square you left) a revert
                  that looks like nothing happening. Both are refused;
      ``noop``    the press moved nobody; the state is returned unchanged;
      ``ok``      an ordinary move, with the new state.
    """

    __slots__ = ("h", "w", "n", "blocked", "endzone", "ball_start", "edge",
                 "d_ez", "d_ball", "sig")

    def __init__(self, h, w, blocked, endzone, ball_start):
        self.h, self.w = h, w
        self.n = h * w
        self.blocked = frozenset(blocked)
        self.endzone = frozenset(endzone)
        self.ball_start = ball_start
        #: ``edge[cell][di]`` is the neighbour in ``DIRS[di]``, or -1 off the
        #: board. Blocked squares are NOT folded in here: `_resolve` has to see
        #: the body standing there to decide whether a chain pushes through it.
        self.edge = []
        for i in range(self.n):
            r, c = divmod(i, w)
            row = []
            for d in DIRS:
                dr, dc = _DELTA[d]
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self.edge.append(tuple(row))
        self.d_ez = self._walk(self.endzone)
        self.d_ball = self._walk([ball_start])
        self.sig = (h, w, tuple(sorted(self.blocked)),
                    tuple(sorted(self.endzone)), ball_start)

    def _walk(self, seeds):
        """Shortest walk distance to ``seeds`` over the free squares, ignoring
        every defender. The heuristic's two legs; see `heuristic`."""
        dist = {s: 0 for s in seeds if s not in self.blocked}
        q = deque(dist)
        while q:
            cur = q.popleft()
            for di in range(4):
                j = self.edge[cur][di]
                if j >= 0 and j not in self.blocked and j not in dist:
                    dist[j] = dist[cur] + 1
                    q.append(j)
        return dist

    # -- the heuristic -------------------------------------------------------
    def remain(self, state) -> int:
        """Admissible, consistent presses-to-win estimate.

        Walk to the ball, then carry it to the endzone. Admissible because one
        press moves you at most one square and the ball moves ONLY under you, so
        both legs have to be walked; the defenders are ignored, and they can only
        ever make a walk longer. Consistent because each leg is a shortest-path
        metric on a graph that never changes, so one press changes it by at most
        one -- including the press that picks the ball up, which ends the first
        leg (1) and starts the second exactly where it left off.

        A huge value means the player cannot reach the ball, or the ball cannot
        reach the endzone, over the STATIC free squares. That is a proof and not
        a guess (linemen and sidelines never move), so pruning on it is sound.

        Clamped to at least 1, which keeps it admissible (no state the search
        holds is already won, so every one of them is at least one press away)
        and keeps it consistent (raising a value to a floor can only close the
        gap to its successor's floor). The clamp is what lets `_Field.cost`
        return ``g + 1`` the moment it expands a state with a winning press:
        that is optimal exactly when ``g + h >= g + 1`` for every state in the
        queue, and without the floor a board handed to `read` with the ball
        already sitting in the endzone would break it by one press."""
        p, b = state[0], state[1]
        if b == p:
            return max(1, self.d_ez.get(p, 1 << 20))
        return max(1, self.d_ball.get(p, 1 << 20) + self.d_ez.get(b, 1 << 20))

    # -- one turn ------------------------------------------------------------
    def step(self, state, di: int):
        p, b, D, Z, S = state
        d = DIRS[di]
        occ = {i: "X" for i in self.blocked}
        occ[p] = "P"
        for i in D:
            occ[i] = "D"
        for i in Z:
            occ[i] = "Z"
        for i in S:
            occ[i] = "S"

        # Force assignment, in the interpreter's own rule order.
        forces = {p: d}
        for i in D:                                   # mirrors the press
            forces[i] = d
        if d != "left":                               # the one-way ratchet
            for i in Z:
                forces[i] = "down"
        pr, pc = divmod(p, self.w)
        for i in S:                                   # line of sight, any range
            sr, sc = divmod(i, self.w)
            if sr == pr and sc != pc:
                forces[i] = "right" if sc < pc else "left"
            elif sc == pc and sr != pr:
                forces[i] = "down" if sr < pr else "up"

        occ = self._resolve(occ, forces)

        # The ball is alone on collision layer 4, so it is never blocked: it
        # moves whenever the player is standing on it and the step is on the
        # board -- even when the player's own step was refused.
        nb = b
        if b == p:
            j = self.edge[b][di]
            if j >= 0:
                nb = j

        np_ = -1
        nD, nZ, nS = [], [], []
        for i, kind in occ.items():
            if kind == "P":
                np_ = i
            elif kind == "D":
                nD.append(i)
            elif kind == "Z":
                nZ.append(i)
            elif kind == "S":
                nS.append(i)

        tackled = False
        for di2 in range(4):                          # late [ Player | Defense ]
            j = self.edge[np_][di2]
            if j >= 0 and occ.get(j) in ("D", "Z", "S"):
                tackled = True
                break

        if nb in self.endzone:
            # late [ Ball Endzone ] -> [ Celebrate Endzone ] fires FIRST, and
            # Celebrate lands on the player's own collision layer. So when the
            # ball ends under you the Player is deleted, no tackle rule can
            # match, and the win stands however many defenders are beside you.
            if nb == np_:
                return "win", (np_, nb, tuple(sorted(nD)), tuple(sorted(nZ)),
                               tuple(sorted(nS)))
            # The ball got there without you, which can only happen when your
            # own press was blocked (see `_Board`'s docstring). The Player is
            # still on the board and still where it started, so
            # require_player_movement throws the win away -- unless a tackle
            # moved it, and then the down is over instead.
            return ("tackle", None) if tackled else ("noop", state)
        if tackled:
            return "tackle", None
        if np_ == p:                                  # require_player_movement
            return "noop", state
        return "ok", (np_, nb, tuple(sorted(nD)), tuple(sorted(nZ)),
                      tuple(sorted(nS)))

    # -- PSEngine._resolve_forces, restricted to the one shared layer --------
    def _resolve(self, occ, forces):
        """Move every body that can move, exactly as `PSEngine._resolve_forces`
        does it.

        Every body in this game -- player, the three defenders, linemen and the
        sidelines themselves -- is on collision layer 5, and there are no rigid
        groups, so the interpreter's algorithm reduces to: trace each force to
        the end of its push chain; a chain whose head is stopped by something
        immovable loses its forces; a chain stopped by a body that is moving
        SOMEWHERE ELSE is DEFERRED to the next pass (that is how a stalker walks
        into the square you are leaving); chains that claim the same square all
        block; everything else moves, and the loop repeats until nothing does.

        Written out rather than reused because the interpreter's version is
        threaded through object indices, rigid groups and layer lookups this game
        has none of. `_selfcheck`'s fuzz is the check that the reduction is
        faithful: 54,000 presses on random boards with up to seven of each
        defender, whole board compared after every one."""
        occ = dict(occ)
        forces = dict(forces)
        for _ in range(20):
            moved_any = False
            deferred_any = False
            start_forces = dict(forces)
            resolved = set()
            movable = []
            for cell in list(forces):
                if cell in resolved or cell not in forces:
                    continue
                d = forces[cell]
                di = DIRS.index(d)
                chain = [cell]
                cur = cell
                chain_free = False
                blocker_is_mover = False
                while True:
                    nxt = self.edge[cur][di]
                    if nxt < 0:
                        break
                    blk = occ.get(nxt)
                    if blk is None:
                        chain_free = True
                        break
                    bf = forces.get(nxt)
                    if bf == d:
                        chain.append(nxt)
                        cur = nxt
                    else:
                        blocker_is_mover = bf is not None
                        break
                if chain_free:
                    movable.append((chain, di))
                elif blocker_is_mover:
                    deferred_any = True
                else:
                    for e in chain:
                        forces.pop(e, None)
                        resolved.add(e)
            # A chain whose head is already carried by a longer chain is the same
            # push group; only the longer one is executed.
            non_head = {}
            for i, (chain, _) in enumerate(movable):
                for e in chain[1:]:
                    non_head[e] = i
            subsumed = {i for i, (chain, _) in enumerate(movable)
                        if chain[0] in non_head}
            claims = {}
            conflicting = set()
            for i, (chain, di) in enumerate(movable):
                if i in subsumed:
                    continue
                for e in chain:
                    t = self.edge[e][di]
                    if t in claims and claims[t] != i:
                        conflicting.add(i)
                        conflicting.add(claims[t])
                    else:
                        claims[t] = i
            for i, (chain, di) in enumerate(movable):
                if i in subsumed:
                    continue
                if i in conflicting:
                    for e in chain:
                        forces.pop(e, None)
                        resolved.add(e)
                    continue
                for e in reversed(chain):
                    occ[self.edge[e][di]] = occ.pop(e)
                    forces.pop(e, None)
                    resolved.add(e)
                    moved_any = True
            if not moved_any:
                if deferred_any and forces == start_forces:
                    forces.clear()
                break
        return occ


# ---------------------------------------------------------------------------
# The exact distance field
# ---------------------------------------------------------------------------

class _Field:
    """Exact presses-to-win for every state on an optimal path, by CONE.

    WHY NOT ENUMERATION. The other ps: games with an exact field enumerate their
    whole reachable space and read the distances straight off it. That is not
    available here: the closure of level 2 -- the third tutorial board, whose
    answer is 9 presses -- is 329,647 states, because nothing forces you forward
    and the defenders keep producing fresh configurations while you wander. The
    closure is not merely large, it is the wrong shape: almost all of it is
    boards no shortest path could pass through.

    So the field is the [[ps-stand-off-solver]] cone, three sweeps:

      1. A* with `_Board.remain` to the optimal cost ``L`` (exact: the heuristic is
         admissible and the weight is 1);
      2. a forward pass in layers keeping only ``g + h <= L``, recording each
         state's successors and every state that has a WINNING press;
      3. a backward BFS from those over the recorded edges.

    Why that is exact where it is used: `h` is consistent, so ``g + h`` is
    non-decreasing along any path, so every prefix of a shortest path to a state
    with ``g + h <= L`` is itself inside the cone -- pass 2's ``g`` is the true
    shortest distance from the start. And a state on an optimal path has
    ``g + dist == L``, so its whole remaining optimal path is inside the cone and
    pass 3 returns its true distance. A successor that TIES with the plan's own
    step is on an optimal path by definition and so is measured exactly; one that
    does not tie can only be over-reported, which cannot put a press into a set
    that does not belong there.

    The cost is about twice the A*, which for the whole game is under a second
    (per-level cones: 7, 7, 25, 18, 386, 654, 268, 5537, 47 and 19 states).

    ``node_cap`` is a runaway guard, not a budget: past it the field answers None
    and the caller treats the board as unplannable. The largest search in the
    game is level 7's 11,185 closed states, two orders of magnitude under it.
    """

    __slots__ = ("board", "cap", "capped")

    def __init__(self, board: _Board, cap: int):
        self.board = board
        self.cap = cap
        self.capped = False

    # -- pass 1 --------------------------------------------------------------
    def cost(self, state) -> "int | None":
        """The optimal number of presses from ``state``, or None when this board
        can never be won from there.

        None is a PROOF, not a budget running out: with no win reachable the
        priority queue drains the whole component that `h` does not prune, and
        `h` prunes only states the player provably cannot walk out of. The one
        exception is the node cap, which sets `capped` when it fires so a caller
        can tell the two apart."""
        board = self.board
        g0 = {state: 0}
        pq = [(board.remain(state), 0, state)]
        nodes = 0
        while pq:
            _f, g, s = heapq.heappop(pq)
            if g > g0.get(s, 1 << 30):
                continue
            for di in range(4):
                status, ns = board.step(s, di)
                nodes += 1
                if nodes > self.cap:
                    self.capped = True
                    return None
                if status == "win":
                    return g + 1
                if status != "ok":
                    continue
                ng = g + 1
                if ng < g0.get(ns, 1 << 30):
                    g0[ns] = ng
                    heapq.heappush(pq, (ng + board.remain(ns), ng, ns))
        return None

    # -- passes 2 and 3 ------------------------------------------------------
    def solve(self, state) -> "tuple[int, dict] | None":
        """``(L, dist)`` where ``dist[s]`` is the exact presses-to-win for every
        state on an optimal path out of ``state``, or None if unwinnable."""
        board = self.board
        L = self.cost(state)
        if L is None:
            return None
        succ = {}
        wins = []
        gmap = {state: 0}
        layer = [state]
        for g in range(L):
            nxt = []
            for s in layer:
                row = []
                row_wins = False
                for di in range(4):
                    status, ns = board.step(s, di)
                    if status == "win":
                        row_wins = True
                        continue
                    if status != "ok":
                        continue
                    if g + 1 + board.remain(ns) > L:
                        continue
                    row.append(ns)
                    if ns not in gmap:
                        gmap[ns] = g + 1
                        nxt.append(ns)
                succ[s] = row
                if row_wins:
                    wins.append(s)
            layer = nxt
        rev = {}
        for s, row in succ.items():
            for ns in row:
                rev.setdefault(ns, []).append(s)
        dist = {s: 1 for s in wins}
        q = deque(wins)
        while q:
            s = q.popleft()
            nd = dist[s] + 1
            for prev in rev.get(s, ()):
                if prev not in dist:
                    dist[prev] = nd
                    q.append(prev)
        if dist.get(state) != L:
            return None                    # cannot happen; never trust it silently
        return L, dist

    # -- the plan ------------------------------------------------------------
    def optimal(self, state, dist):
        """``[(direction index, successor or None), ...]`` for every press on a
        shortest path from ``state``; ``None`` marks a press that wins outright.
        Ties come out in ``DIRS`` order, which is what makes a re-derived plan
        byte-identical across processes."""
        d = dist.get(state)
        if not d:
            return []
        out = []
        for di in range(4):
            status, ns = self.board.step(state, di)
            if d == 1:
                if status == "win":
                    out.append((di, None))
            elif status == "ok" and dist.get(ns) == d - 1:
                out.append((di, ns))
        return out

    def plan(self, state) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal set
        at every step, or None if this board can no longer be won."""
        got = self.solve(state)
        if got is None:
            return None
        L, dist = got
        presses, optsets = [], []
        cur = state
        for _ in range(L):
            best = self.optimal(cur, dist)
            if not best:                   # a labelled state always has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class TouchdownHeroesExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Field`.

    The base class keeps the memo, the level scoping and the disk cache; only the
    strategy underneath changes, the way `PSEnumExpert` replaces it. Here the
    strategy is "read the live board and run the cone", so `heuristic` is never
    called on the ENGINE and asserts rather than returning a number nothing would
    use -- the heuristic that matters is `_Board.remain`, which is a function of a
    model state.

    `_search` reads the ENGINE's grid every time, so a re-plan from an arbitrary
    state (a recovery probe, an epsilon detour) is answered exactly -- including
    the answer "no".
    """

    directions = list(DIRS)
    #: `_key` is every object the mechanic reads, which is canonical across
    #: levels; the level scope is kept anyway so the disk cache and the memo can
    #: never serve one level's plan to another.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    #: Runaway guard on one search; see `_Field`. The largest this game reaches
    #: is level 7's 11,185 closed states (~45k model steps).
    node_cap_default = 4_000_000

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(self.game._engine._player_indices)
        self.ball_ids = set(g.resolve_object_name("ball"))
        self.endzone_ids = set(g.resolve_object_name("endzone"))
        self.block_ids = {n: set(g.resolve_object_name(n)) for n in _BLOCK_NAMES}
        self.foe_ids = {n: set(g.resolve_object_name(n)) for n in _FOE_NAMES}
        #: Objects that mean "this grid is not a state the model can step from":
        #: the down is already over (Tackled) or already scored (Celebrate).
        self.terminal_ids = (set(g.resolve_object_name("tackled"))
                             | set(g.resolve_object_name("celebrate")))
        self.key_ids = (self.player_ids | self.ball_ids | self.endzone_ids
                        | self.terminal_ids)
        for s in self.block_ids.values():
            self.key_ids |= s
        for s in self.foe_ids.values():
            self.key_ids |= s
        #: `_Board`s and `_Field`s by STATIC signature (see `read`), not by level
        #: index, so the edge table and the two walk-distance maps are built once
        #: per level however many states are read from it.
        self._boards: dict = {}
        self._fields: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "TouchdownHeroesExpert plans with an exact cone field over a native "
            "model; PSExpert.heuristic is unused (see _Board.h)")

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng):
        """``(board, state)`` for the engine's current grid, or ``(None, None)``
        when that grid is not a state this model can step from.

        The three refusals are all real states the interpreter can be left in and
        all of them must be refused rather than mis-modelled:

          * a Tackled or Celebrate object on the board -- the down is over, and
            an epsilon-detour probe that pressed into a defender leaves exactly
            that behind;
          * no Player and no Ball -- the same, one step further on;
          * the player already orthogonally adjacent to a defender. That is not
            a state legal play can produce (the tackle is checked at the END of
            every turn, so being beside a defender means the previous turn ended
            the down), so a grid showing it is a grid the model has no rules
            for."""
        h, w = eng.height, eng.width
        blocked, endzone, foes = [], [], {n: [] for n in _FOE_NAMES}
        p = b = None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.terminal_ids:
                    return None, None
                for n, ids in self.block_ids.items():
                    if cell & ids:
                        blocked.append(i)
                if cell & self.endzone_ids:
                    endzone.append(i)
                if cell & self.ball_ids:
                    b = i
                if cell & self.player_ids:
                    p = i
                for n, ids in self.foe_ids.items():
                    if cell & ids:
                        foes[n].append(i)
        if p is None or b is None:
            return None, None
        sig = (h, w, tuple(blocked), tuple(endzone), b if b != p else -1)
        board = self._boards.get(sig)
        if board is None:
            # The ball's START cell only matters while it is still lying there;
            # once it is being carried the second leg of the heuristic is
            # measured from the player, so any board of the same geometry works.
            board = self._boards[sig] = _Board(h, w, blocked, endzone,
                                               b if b != p else p)
        state = (p, b,
                 tuple(sorted(foes["defense"])),
                 tuple(sorted(foes["blitzer"])),
                 tuple(sorted(foes["stalker"])))
        for di in range(4):
            j = board.edge[p][di]
            if j >= 0 and (j in state[2] or j in state[3] or j in state[4]):
                return None, None
        return board, state

    def field(self, board: _Board) -> _Field:
        got = self._fields.get(board.sig)
        if got is None:
            got = self._fields[board.sig] = _Field(
                board, max(self.node_cap, self.node_cap_default))
        return got

    def _key(self, eng) -> frozenset:
        """Every cell holding an object the mechanic reads. Static scenery is in
        here too (sidelines, linemen, endzone), so the ``plan_cache_path``
        signature is matched against the board it was solved from and an edited
        level is a cache miss rather than a wrong plan."""
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in cell
            if o in self.key_ids
        )

    def _search(self, eng) -> "Plan | None":
        board, state = self.read(eng)
        if board is None:
            return None
        return self.field(board).plan(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class TouchdownHeroesSolver(PSAStarSolver):
    game_id = "puzzlescript_touchdown_heroes_prototype"
    game_name = GAME_NAME
    expert_cls = TouchdownHeroesExpert

    #: `games/ps:touchdown_heroes_prototype/ps:touchdown_heroes_prototype.py` is
    #: a plain passthrough -- it builds the adapter and nothing else, and the
    #: rendering fix is in the .txt, which both paths read. Set this if that
    #: wrapper ever grows a patch.
    game_module_id = ""

    #: Unused: `TouchdownHeroesExpert._search` never calls `_astar`. The knob
    #: that bounds the searches is `_Field.cap`, seeded from
    #: `TouchdownHeroesExpert.node_cap_default`.
    weight = 1
    node_cap = 400_000

    #: Room for the longest plan (21 presses on level 7) plus its detours, well
    #: under the adapter's own 200-press per-level budget -- which the
    #: `set_level` that ends the exploration prefix resets, so the plan starts it
    #: from zero.
    max_steps = 90

    #: Every level is solved; nothing is skipped.
    skip_levels = frozenset()

    #: On top of the RESET prefix: about one press in twelve is a random legal
    #: alternative, after which the expert re-plans from wherever it landed --
    #: the taken action is the mistake and ``optimal`` is the recovery.
    #:
    #: Detours are safe here because both ways of ruining a down are refused
    #: exactly and cheaply. A press that gets you tackled is not a legal model
    #: successor at all (and `_search` refuses the interpreter grid such a probe
    #: leaves behind), and a press that leaves the level unwinnable is refused by
    #: the re-plan returning None -- which is a proof, since A* exhausts the
    #: component. Measured over 600 random walks out of the ten starts, 596 were
    #: still winnable, so the detours cost throughput rather than seeds.
    #:
    #: The rate is on the high side of the family because a detour here is CHEAP:
    #: nothing about this game is irreversible except the blitzers' downward
    #: ratchet, and a wasted press typically costs two (step aside, step back).
    epsilon = 0.08

    def prepare_expert(self, game, expert) -> None:
        """Solve every level before `discover_solvable` asks for it -- the same
        work either way, but it fills the disk cache in one pass and makes the
        startup cost visible as startup."""
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
    solver = TouchdownHeroesSolver()
    game = solver.make_game(seed)
    return solver, game, TouchdownHeroesExpert(game, node_cap=solver.node_cap)


_GLYPH = {"X": "#", "P": "Q", "D": "D", "Z": "Z", "S": "S"}


def _ascii(board: _Board, state) -> str:
    """A model state as ASCII, so a failing check can print the board it failed
    on rather than a tuple of integers."""
    p, b, D, Z, S = state
    out = []
    for r in range(board.h):
        line = ""
        for c in range(board.w):
            i = r * board.w + c
            if i in board.blocked:
                ch = "#"
            elif i == p:
                ch = "@" if b == p else "Q"
            elif i in D:
                ch = "D"
            elif i in Z:
                ch = "Z"
            elif i in S:
                ch = "S"
            elif i == b:
                ch = "B"
            elif i in board.endzone:
                ch = "="
            else:
                ch = "."
            line += ch
        out.append(line)
    return "\n".join(out)


def _start(game, expert, level: int):
    """``(board, state)`` at a level's start, engine left seated there."""
    game.set_level(level)
    return expert.read(game._engine)


def _plans(verbose: bool = True) -> int:
    """Every level's plan, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once."""
    _solver, game, expert = _new()
    eng = game._engine
    total = ties = bad = 0
    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: NO PLAN")
            bad += 1
            continue
        game.set_level(level)
        for i, d in enumerate(plan):
            if d not in plan.optsets[i]:
                print(f"  L{level} step {i}: {d} not in its own optimal set")
                bad += 1
            eng.step(d)
        if not eng.check_win():
            print(f"  L{level}: plan of {len(plan)} presses did NOT win")
            bad += 1
        total += len(plan)
        ties += sum(1 for s in plan.optsets if len(s) > 1)
        if verbose:
            print(f"  L{level}: {len(plan):3d} presses, "
                  f"{sum(1 for s in plan.optsets if len(s) > 1):2d} tie steps  "
                  + " ".join("".join(x[0] for x in s) for s in plan.optsets))
    if verbose:
        print(f"  {total} presses over {game.n_levels} levels, "
              f"{ties} of them with a second equally right answer")
    return bad


# -- the interpreter, as a state source for the fuzz --------------------------

def _ids(game) -> dict:
    g = game._game
    names = ("background", "sideline", "yardline", "endzone", "ball", "player",
             "tackled", "celebrate", "lineman", "defense", "blitzer", "stalker")
    return {n: set(g.resolve_object_name(n)) for n in names}


def _seat(eng, ids, rows) -> None:
    """Seat an ASCII board (of the engine's current dimensions) on the grid."""
    ch_to_name = {"#": "sideline", "-": "yardline", "=": "endzone",
                  "B": "ball", "Q": "player", "L": "lineman",
                  "D": "defense", "Z": "blitzer", "S": "stalker"}
    for r in range(eng.height):
        for c in range(eng.width):
            cell = set(ids["background"])
            ch = rows[r][c]
            if ch == "@":
                cell |= ids["player"] | ids["ball"]
            elif ch in ch_to_name:
                cell |= ids[ch_to_name[ch]]
            eng.grid[r][c] = cell
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _grid_sig(eng) -> tuple:
    return tuple(tuple(frozenset(cell) for cell in row) for row in eng.grid)


def _random_board(rng, h, w, foes, linemen) -> list:
    """A random legal-shaped board: sidelines all round, an endzone across the
    top, the player (usually already carrying) and a scatter of bodies."""
    rows = [["." for _ in range(w)] for _ in range(h)]
    for c in range(w):
        rows[0][c] = "="
        rows[h - 1][c] = "#"
    for r in range(h):
        rows[r][0] = "#"
        rows[r][w - 1] = "#"
    rows[0][0] = rows[0][w - 1] = "#"
    free = ([(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
            + [(0, c) for c in range(1, w - 1)])
    rng.shuffle(free)
    it = iter(free)
    pr, pc = next(it)
    rows[pr][pc] = "@" if rng.random() < 0.7 else "Q"
    if rows[pr][pc] == "Q":
        br, bc = next(it)
        rows[br][bc] = "B"
    for ch, n in (("D", foes[0]), ("Z", foes[1]), ("S", foes[2]),
                  ("L", linemen)):
        for _ in range(n):
            r, c = next(it)
            rows[r][c] = ch
    return ["".join(r) for r in rows]


def _fuzz(game, expert, rng, trials, walk, level, counts, verbose) -> int:
    """Model vs interpreter: random board, random walk, whole state compared
    after every press.

    The four statuses are checked against exactly what the interpreter is allowed
    to do for each:

      ``ok``      the press moved the player, no win, no restart, and the whole
                  board matches the model's new state;
      ``win``     `check_win`;
      ``noop``    the press returned False and the grid is untouched;
      ``tackle``  either a RESTART, or the reverted no-op the engine produces
                  when the tackling defender ends up on the square the player
                  left (see `_Board.step`). Both are refused by the search, so
                  the model does not have to say which -- but it does have to be
                  right that ONE of them happened, and that is what is asserted.
    """
    ids = _ids(game)
    eng = game._engine
    bad = 0
    for t in range(trials):
        game.set_level(level)
        rows = _random_board(rng, eng.height, eng.width,
                             (rng.randint(0, 6), rng.randint(0, 4),
                              rng.randint(0, 6)), rng.randint(0, 5))
        _seat(eng, ids, rows)
        board, state = expert.read(eng)
        if board is None:
            continue
        for _ in range(walk):
            di = rng.randrange(4)
            status, nstate = board.step(state, di)
            counts[status] = counts.get(status, 0) + 1
            before = _grid_sig(eng)
            moved = eng.step(DIRS[di])
            after = _grid_sig(eng)
            win, restart = eng.check_win(), eng._rule_restart
            if status == "win":
                ok = win
            elif status == "tackle":
                ok = restart or (not moved and after == before)
            elif status == "noop":
                ok = (not moved) and after == before and not win
            else:
                _b2, got = expert.read(eng)
                ok = (moved and not win and not restart and got == nstate)
            if not ok:
                bad += 1
                print(f"  FUZZ mismatch (trial {t}, press {DIRS[di]}): model "
                      f"says {status}; engine moved={moved} win={win} "
                      f"restart={restart}")
                print(_ascii(board, state))
                if bad > 4:
                    return bad
                break
            if status in ("win", "tackle"):
                break
            state = nstate
    if verbose:
        print(f"  fuzz {eng.height}x{eng.width}: {sum(counts.values())} "
              f"presses, {bad} mismatches, {counts}")
    return bad


def _bfs(board: _Board, start, cap=4_000_000):
    """Plain forward BFS to the first win. No heuristic, no cone, no backward
    sweep -- the independent derivation of d*. Returns ``(d*, states)``."""
    dist = {start: 0}
    q = deque([start])
    while q:
        s = q.popleft()
        d = dist[s]
        for di in range(4):
            status, ns = board.step(s, di)
            if status == "win":
                return d + 1, len(dist)
            if status != "ok" or ns in dist:
                continue
            dist[ns] = d + 1
            if len(dist) > cap:
                return None, len(dist)
            q.append(ns)
    return None, len(dist)


def _selfcheck(fuzz_trials: int = 700, fuzz_walk: int = 20,
               recovery_walks: int = 40, verbose: bool = True) -> int:
    """Six passes, each an independent check on a different claim."""
    solver, game, expert = _new()
    eng = game._engine
    rng = random.Random(20260821)
    bad = 0

    if verbose:
        print("pass 0: ACTION is unbound (noaction)")
    for level in range(game.n_levels):
        game.set_level(level)
        before = _grid_sig(eng)
        moved = eng.step("action")
        if moved or _grid_sig(eng) != before:
            print(f"  L{level}: ACTION changed the board")
            bad += 1

    if verbose:
        print("pass 1: every level starts in a state the model can step from")
    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        if board is None:
            print(f"  L{level}: start state refused by read()")
            bad += 1
            continue
        if state[0] == state[1]:
            print(f"  L{level}: starts already carrying the ball")
            bad += 1

    if verbose:
        print("pass 2: the model, fuzzed against the interpreter")
    counts: dict = {}
    bad += _fuzz(game, expert, rng, fuzz_trials, fuzz_walk, 0, counts, verbose)
    counts = {}
    bad += _fuzz(game, expert, rng, fuzz_trials, fuzz_walk, 4, counts, verbose)

    if verbose:
        print("pass 3: d* re-derived by plain forward BFS")
    plans = {}
    for level in range(game.n_levels):
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        plans[level] = (board, state, plan)
        d, states = _bfs(board, state)
        got = None if plan is None else len(plan)
        if d != got:
            print(f"  L{level}: BFS says {d}, the field says {got}")
            bad += 1
        elif verbose:
            print(f"  L{level}: d*={d} agreed  (BFS closed {states} states)")

    if verbose:
        print("pass 4: every tie set re-derived by re-solving")
    for level in range(game.n_levels):
        board, state, plan = plans[level]
        if plan is None:
            continue
        field = expert.field(board)
        cur = state
        for i, taken in enumerate(plan):
            remaining = len(plan) - i
            best = []
            for di in range(4):
                status, ns = board.step(cur, di)
                if status == "win":
                    cost = 1
                elif status != "ok":
                    continue
                else:
                    sub = field.cost(ns)
                    cost = None if sub is None else sub + 1
                if cost == remaining:
                    best.append(DIRS[di])
            if sorted(best) != sorted(plan.optsets[i]):
                print(f"  L{level} step {i}: re-solve says {sorted(best)}, "
                      f"the field says {sorted(plan.optsets[i])}")
                bad += 1
            cur = board.step(cur, DIRS.index(taken))[1]

    if verbose:
        print("pass 5: recovery -- re-plan from random states off the plan")
    lost = probed = 0
    for level in range(game.n_levels):
        board, state, _plan = plans[level]
        for _ in range(recovery_walks):
            cur = state
            for _ in range(rng.randint(1, 12)):
                opts = [ns for di in range(4)
                        for st, ns in (board.step(cur, di),) if st == "ok"]
                if not opts:
                    break
                cur = rng.choice(opts)
            probed += 1
            sub = expert.field(board).plan(cur)
            if sub is None:
                lost += 1
                continue
            # replay it in the model and require a win on the last press
            s = cur
            for j, d in enumerate(sub):
                status, s = board.step(s, DIRS.index(d))
                if status == "win" and j == len(sub) - 1:
                    break
                if status != "ok":
                    print(f"  L{level}: recovery plan broke at step {j}")
                    bad += 1
                    break
            else:
                print(f"  L{level}: recovery plan of {len(sub)} did not win")
                bad += 1
    if verbose:
        print(f"  {probed} perturbed states: {probed - lost} re-planned, "
              f"{lost} provably lost")
    return bad


# -- the same two sweeps, driven by the REAL interpreter ---------------------

def _engine_state(eng, expert):
    """The interpreter's grid as a hashable key, with NO model call: every cell
    holding an object the mechanic reads."""
    return expert._key(eng)


def _engine(levels=tuple(range(10)), verbose: bool = True) -> int:
    """d* and the per-step tie sets re-derived with the interpreter as the
    successor function and `check_win` as the goal test. No `_Board` call takes
    part in the search -- the only thing shared with the field is `_Board.remain`,
    read off a state the model is asked for purely to order the queue, which
    cannot change the optimum of an admissible A*."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in levels:
        board, state = _start(game, expert, level)
        plan = expert.plan(eng, level)
        game.set_level(level)
        start_snap = snapshot(eng)
        start_key = expert._key(eng)

        def succ(snap):
            """(win presses, {press: (key, snapshot)}) from an engine state."""
            wins, out = [], {}
            for di in range(4):
                restore(eng, snap)
                moved = eng.step(DIRS[di])
                if eng.check_win():
                    wins.append(di)
                    continue
                if not moved or eng._rule_restart:
                    continue
                out[di] = (expert._key(eng), snapshot(eng))
            return wins, out

        # pass 1: A* over the interpreter
        heap = [(board.remain(state), 0, 0, start_key, start_snap)]
        g0 = {start_key: 0}
        L = None
        tick = 0
        while heap and L is None:
            _f, g, _t, key, snap = heapq.heappop(heap)
            if g > g0.get(key, 1 << 30):
                continue
            wins, out = succ(snap)
            if wins:
                L = g + 1
                break
            for di, (k2, s2) in out.items():
                if g + 1 < g0.get(k2, 1 << 30):
                    g0[k2] = g + 1
                    restore(eng, s2)
                    _b, st2 = expert.read(eng)
                    if st2 is None:
                        continue
                    tick += 1
                    heapq.heappush(heap, (g + 1 + board.remain(st2), g + 1, tick,
                                          k2, s2))
        got = None if plan is None else len(plan)
        if L != got:
            print(f"  L{level}: interpreter A* says {L}, the field says {got}")
            bad += 1
            continue

        # pass 2+3: the cone, over the interpreter
        snaps = {start_key: start_snap}
        gmap = {start_key: 0}
        layer = [start_key]
        edges = {}
        winset = []
        for g in range(L):
            nxt = []
            for key in layer:
                wins, out = succ(snaps[key])
                row = []
                for di, (k2, s2) in out.items():
                    restore(eng, s2)
                    _b, st2 = expert.read(eng)
                    if st2 is None or g + 1 + board.remain(st2) > L:
                        continue
                    row.append(k2)
                    if k2 not in gmap:
                        gmap[k2] = g + 1
                        snaps[k2] = s2
                        nxt.append(k2)
                edges[key] = row
                if wins:
                    winset.append(key)
            layer = nxt
        rev = {}
        for k, row in edges.items():
            for k2 in row:
                rev.setdefault(k2, []).append(k)
        dist = {k: 1 for k in winset}
        q = deque(winset)
        while q:
            k = q.popleft()
            for prev in rev.get(k, ()):
                if prev not in dist:
                    dist[prev] = dist[k] + 1
                    q.append(prev)
        if dist.get(start_key) != L:
            print(f"  L{level}: interpreter cone did not reach the start")
            bad += 1
            continue

        # walk the interpreter's own gradient and compare the sets
        key, snap = start_key, start_snap
        for i in range(L):
            wins, out = succ(snap)
            d = dist[key]
            best = ([DIRS[di] for di in wins] if d == 1 else
                    [DIRS[di] for di, (k2, _s) in out.items()
                     if dist.get(k2) == d - 1])
            if sorted(best) != sorted(plan.optsets[i]):
                print(f"  L{level} step {i}: interpreter says {sorted(best)}, "
                      f"the field says {sorted(plan.optsets[i])}")
                bad += 1
            if d == 1:
                break
            taken = plan[i]
            key, snap = out[DIRS.index(taken)]
        if verbose:
            print(f"  L{level}: d*={L} confirmed by the interpreter "
                  f"({len(gmap)} states in the cone)")
    return bad


# -- the render audit ---------------------------------------------------------

#: Every stack of objects a cell of this game can hold. A Sideline cell holds
#: nothing else (it is what the '#' character parses to and nothing enters it);
#: Celebrate exists only where the win rule creates it, which is an Endzone
#: square. Everything else is free: any body can stand on turf, on a yardline or
#: in the endzone, with or without the loose ball under it.
_BODIES = ("player", "celebrate", "tackled", "lineman",
           "defense", "blitzer", "stalker")
_GROUNDS = ((), ("yardline",), ("endzone",))


def _compositions() -> list:
    out = [("sideline",)]
    for gnd in _GROUNDS:
        for ball in ((), ("ball",)):
            out.append(gnd + ball)
            for body in _BODIES:
                if body == "celebrate" and gnd != ("endzone",):
                    continue
                out.append(gnd + ball + (body,))
    return out


def _audit(verbose: bool = True) -> int:
    """Render every cell composition as a whole 64x64 frame of a UNIFORM board,
    at both board shapes the ten levels use, and require them pairwise distinct.

    Whole frames, not a cell sliced out of a mixed board: `_render_frame`
    upscales non-uniformly (a 12-wide board spends 5 or 6 pixels on a column) and
    centre-pads the result, so slicing by ``cell_px`` arithmetic reads the wrong
    pixels. That is the ps:explod lesson and it is what makes this audit
    trustworthy at the 4-pixel cell size the six 16-row levels use, where
    `_render_frame` drops the sprite's middle row and middle column."""
    _solver, game, _expert = _new()
    eng = game._engine
    ids = _ids(game)
    comps = _compositions()
    bad = 0
    for level, (h, w) in ((0, (9, 12)), (4, (16, 12))):
        game.set_level(level)
        assert (eng.height, eng.width) == (h, w), (eng.height, eng.width)
        game._rotation_k, game._hflip, game._vflip = 0, False, False
        seen = {}
        for comp in comps:
            cell = set(ids["background"])
            for name in comp:
                cell |= ids[name]
            eng.grid = [[set(cell) for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            key = _render_frame(eng, game._game).tobytes()
            if key in seen:
                print(f"  {h}x{w}: {'+'.join(comp) or 'turf'} renders exactly "
                      f"like {'+'.join(seen[key]) or 'turf'}")
                bad += 1
            else:
                seen[key] = comp
        if verbose:
            print(f"  {h}x{w}: {len(comps)} compositions, {len(seen)} distinct")
    return bad


# -- the rotation argument, measured -----------------------------------------

def _symmetry(walk_presses: int = 200, verbose: bool = True) -> int:
    """Drive the same ENGINE directions at all four rotations and require every
    frame to be the exact rot90 of the unrotated one, and the engine grid to be
    identical after every press.

    That is the whole augmentation for this game: it is deliberately not in
    `_FLIP_GAMES` (the Blitzer's LEFT rule is chiral -- see the module
    docstring), so 4 presentations and no mirrors. The walk presses the unbound
    ACTION key too, and it runs long enough to get tackled repeatedly, which is
    what puts the adapter's RESTART path (reload the level, re-render) under the
    same comparison."""
    solver = TouchdownHeroesSolver()
    _s, game0, expert = _new()
    ref: dict = {}
    bad = 0
    for k in range(4):
        game = solver.make_game(seed=0)
        for level in range(game.n_levels):
            game.set_level(level)
            game._rotation_k = k
            game._current_frame = game._present_frame(
                _render_frame(game._engine, game._game))
            plan = expert.plan(game0._engine, level) if k == 0 else ref[
                ("plan", level)]
            if k == 0:
                ref[("plan", level)] = plan
            rng = random.Random(1000 + level)
            presses = list(plan) + [rng.choice(DIRS + ("action",))
                                    for _ in range(walk_presses)]
            frames, grids = [], []
            for d in presses:
                from solvers.common.ps_astar import screen_action
                from arcengine import ActionInput
                act = screen_action(d, k, False, False, True)
                fd = game.perform_action(ActionInput(id=act))
                frames.append(np.asarray(
                    fd.frame[-1] if fd.frame else game._current_frame))
                grids.append(_grid_sig(game._engine))
            if k == 0:
                ref[("frames", level)] = frames
                ref[("grids", level)] = grids
                continue
            if grids != ref[("grids", level)]:
                print(f"  k={k} L{level}: the engine took a different path")
                bad += 1
            if any(not np.array_equal(np.rot90(a, k), b)
                   for a, b in zip(ref[("frames", level)], frames)):
                print(f"  k={k} L{level}: frames are not the rot90 of k=0")
                bad += 1
    if verbose:
        print(f"  4 rotations x {game0.n_levels} levels compared")
    return bad


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--engine" in sys.argv:
        args = [int(a) for a in sys.argv[sys.argv.index("--engine") + 1:]
                if a.isdigit()]
        violations = _engine(tuple(args) or tuple(range(10)))
        print(f"engine sweeps: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(TouchdownHeroesSolver.main())
