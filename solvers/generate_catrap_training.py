"""Generate Phase-1 training data for the PuzzleScript game ps:catrap
("Catrap" by Gruntfuggly, after Yutaka Isokawa's Pitman / Catrap, 100 levels).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: a
tick-accurate MODEL of the mechanic, a macro A* over it, and the optimal-action
labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_catrap",
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
action (post rotation remap), i.e. the button an agent presses in the presented
view, so replaying the recorded actions reproduces the recorded frames exactly.

The game
--------
Win = ``no Monster``: kill every monster on the board. There is no death and no
timer -- the only way to lose is to make the board unwinnable, which is silent.

Eight rules carry the whole mechanic; the order they run in is most of the
subtlety, because ONE keypress is a whole `again` cascade (move, then fall, then
fall again, ...) that the interpreter drains before returning:

  * **Gravity.** ``down [ Gravity no Ladder | no obstacle no Ladder ]``. Boulder,
    Player, Catgirl, Demon, Troll and Phantom fall one cell per tick;
    **Ghost does not** -- it is the one monster left out of the `Gravity` group,
    so it floats. Standing ON a ladder cell suspends you (the `no Ladder` on the
    object's own cell), and a ladder cell also SUPPORTS whatever sits above it.
  * **Push.** ``horizontal [ > Player no Temp | Boulder no Temp | no Obstacle ]``
    walks the player into the boulder's cell and the boulder one further. The
    `Temp` marks (set one rule earlier by ``up [ Player | Gravity ]`` and
    ``up [ Boulder | Gravity ]``) mean **you cannot push while anything is
    stacked on your head or on the boulder's** -- and the very next rule,
    ``horizontal [ > Player | Boulder | no Obstacle ] -> [ Player | | Boulder ]``,
    then slides that boulder out from under its load WITHOUT the player
    following. Two different moves off the same keypress.
  * **Ladders.** Up is dead unless you are standing on a ladder cell
    (``up [ > Player | ] -> [ Player | ]`` eats every other up press): there is
    no jumping in this game, so height is only ever gained on a ladder.
  * **Digging.** ``horizontal [ > Player | Destructible ] -> [ > Player | ]``
    deletes Earth and walks into it -- but ONLY sideways
    (``down [ > Player | Earth ]`` cancels), so earth is a floor you can tunnel
    through but never drop through.
  * **Killing.** ``horizontal [ > Player | Monster ]`` starts a four-tick death
    animation (Monster -> Dieing1 -> 2 -> 3 -> consumed) that runs inside the
    same keypress. The last step, ``[ Player | DieingMonster3 ] -> [ | Player ]``,
    is DIRECTIONLESS: the player is pulled into the corpse's cell from whichever
    side it is on. If the player is not adjacent when the corpse ripens -- it
    fell away while the animation ran -- the corpse is stranded, and since a
    DieingMonster is still a `Monster` the level can no longer be won until the
    player walks back next to it.
  * **The catgirl** (levels 30-39, 60-69, 90-100) is a second body. ACTION swaps
    which one you drive, over three ticks, in place; the idle one is still a
    `Gravity` obstacle and can be stood on.

Expert solver
-------------
A MACRO A* over a re-implemented, tick-accurate model of those rules
(`Sim.press`), not over the interpreter. **84 of the 101 levels solve**, in 2 to
152 presses (4592 presses over the whole game, 49 of them with a tie).

  * **Why a model.** The interpreter costs ~0.4 ms per keypress; the model costs
    ~25 us. That 15x is what makes the search reach past the tutorial levels, and
    it also buys an exact, compact state key (`bytes(dyn) + bytes(earth)`)
    instead of a frozenset of engine cells.
  * **Why macros.** Almost every press only relocates the player. The successors
    here are therefore ``walk to cell X, then press the one key that changes
    something`` -- a push, a dig, a kill, a swap, or a step that un-supports a
    boulder overhead -- with the walk found by BFS over the (directed! gravity is
    one-way) walk graph. Search depth becomes the number of *decisions*, ~10-40,
    instead of the ~40-150 presses a plan actually contains, and a macro is
    applied by TELEPORTING the player to its cell (see `apply_macro`).
  * **The ladder**: A* at w=1 (shortest plans), then w=3, then a width-4000
    breadth-first beam whose wandering is trimmed by macro deletion (`_shorten`).
  * **The model is never trusted.** Every plan is replayed through the real
    interpreter and kept only if it really wins (`CatrapExpert._verify`). A model
    bug can therefore cost a level, never correctness -- and `--fuzz` compares
    model against interpreter press-by-press over random rollouts on all 101
    levels, auditing the walk analysis at sampled cells as it goes, to keep that
    from happening quietly.

Of the 17 levels left over, **40, 49 and 70 are UNWINNABLE as shipped** -- an
exhaustive BFS over the real interpreter reaches 655 / 2578 / 1006 states and no
win. All three hang on the same clause: `Obstacle` includes Ladder, so a boulder
can never be pushed onto a ladder cell, and each of them fences its monsters
behind a boulder standing next to a ladder. The other 14 are simply deep (39, 64,
65, 68, 69, 80, 87, 88, 90-93, 95, 100 -- mostly the two-body boards and the
"phantom pack" boards, 30+ monsters with a 40-decision solution). `PSAStarSolver`
keeps the levels the expert can win and skips the rest, so the corpus is only
ever real wins; `--plans` prints the current roster.

Augmentation
------------
Rotation only (`rotation_k` in {0,1,2,3}, with the matching directional action
remap in `PuzzleScriptAdapter.perform_action`). Catrap is NOT in `_FLIP_GAMES`:
gravity is an axis-sensitive mechanic, so a vertical flip would present a world
whose stones fall upwards, and its sprites are not mirror-closed. No colour
augmentation either -- earth-versus-brick (diggable versus not) is a colour
distinction the mechanic depends on.

The plans are cached on disk (`PLAN_CACHE`): they are seed-independent, a cold
build costs ~90 minutes and every shard of `parallelize_generator` would
otherwise repeat it on every core. A cached plan is re-verified on the
interpreter before it is used, and carries a signature of the level it was solved
for, so editing a level can never silently ship a stale plan.

Usage (run from the repo root):
    python solvers/generate_catrap_training.py --episodes 200 \
        --out data/training_multi_level/catrap

    python solvers/generate_catrap_training.py --plans     # level-by-level report
    python solvers/generate_catrap_training.py --plans --retry   # re-try failures
    python solvers/generate_catrap_training.py --fuzz      # model vs interpreter
"""

from __future__ import annotations

import heapq
import json
import os
import random
import sys
import time
import zlib
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import PuzzleScriptAdapter        # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert          # noqa: E402

GAME_NAME = "Catrap"

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Disk cache of the per-level plan AND its optimal-action sets. The searches are
#: seed-independent (levels are fixed ASCII maps; only the PRESENTATION is
#: augmented per seed) but cost minutes, which every shard of
#: `parallelize_generator` would otherwise repeat on every core. Delete to rebuild.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "catrap_plans.json"

DIRS = ("up", "down", "left", "right")

# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
# Two planes per cell, mirroring the two collision layers the mechanic uses:
#   `st`  static  -- brick / ladder, never created or destroyed
#   `ea`  earth   -- only ever deleted (by a sideways dig)
#   `dyn` movable -- one of the codes below
# Earth is its own plane rather than a `dyn` code because it shares its collision
# layer with the ladders, not with the boulders: a boulder and an earth block can
# never be in the same cell, but a PLAYER and a LADDER routinely are.

EMPTY, BOULDER, PLAYER, CATGIRL, MFALL, MFLOAT, DM1, DM2, DM3, SWAP = range(10)

S_NONE, S_LADDER, S_BRICK = 0, 1, 2

#: Objects the gravity rule moves. Ghost (MFLOAT) and the three dying states are
#: deliberately absent -- the game's `Gravity` group leaves them out.
_GRAV = (BOULDER, PLAYER, CATGIRL, MFALL)
#: Everything the win condition counts, i.e. the game's `Monster` group.
_MONST = (MFALL, MFLOAT, DM1, DM2, DM3)

#: Engine object name -> model code for the movable plane.
_DYN_OF = {
    "boulder": BOULDER, "player": PLAYER, "catgirl": CATGIRL,
    "demon": MFALL, "troll": MFALL, "phantom": MFALL, "ghost": MFLOAT,
    "dieingmonster1": DM1, "dieingmonster2": DM2, "dieingmonster3": DM3,
    "swap": SWAP,
}
_STATIC_OF = {"brick": S_BRICK, "halfbrick": S_BRICK, "ladder": S_LADDER}
_EARTH_NAMES = ("wholeearth", "halfearth")


class Board:
    """Static geometry of one level, padded with a ring of brick.

    The ring is what lets every neighbour be a plain index offset: no bounds
    test, no row wrap-around, and the real board already has a brick frame so the
    padding is unreachable. Cell ``(r, c)`` of the engine grid is index
    ``(r + 1) * W + (c + 1)``."""

    def __init__(self, eng, game):
        self.H, self.W = eng.height + 2, eng.width + 2
        n = self.H * self.W
        self.st = bytearray(n)
        for i in range(self.W):                      # top / bottom ring
            self.st[i] = S_BRICK
            self.st[n - 1 - i] = S_BRICK
        for r in range(self.H):                      # left / right ring
            self.st[r * self.W] = S_BRICK
            self.st[r * self.W + self.W - 1] = S_BRICK
        name = game.obj_idx_to_name
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                for o in cell:
                    s = _STATIC_OF.get(name.get(o, ""))
                    if s is not None:
                        self.st[(r + 1) * self.W + c + 1] = s
        self.delta = {"up": -self.W, "down": self.W, "left": -1, "right": 1}
        self.dirs = tuple(self.delta[d] for d in DIRS)
        #: offset -> press name, and 0 -> "action" (the swap has no direction).
        self.name = {v: k for k, v in self.delta.items()}
        self.name[0] = "action"

    def index(self, r: int, c: int) -> int:
        return (r + 1) * self.W + c + 1

    def rc(self, i: int) -> tuple[int, int]:
        return divmod(i, self.W)[0] - 1, i % self.W - 1


class Sim:
    """One mutable game state on a `Board`, with the interpreter's turn loop.

    `press` is the unit of play: it runs ticks until the `again` cascade settles,
    exactly as `PSEngine.step` does, so one call is one keypress and the state it
    leaves is a state the player can actually be handed."""

    __slots__ = ("b", "dyn", "ea", "occ", "pp", "broken")

    def __init__(self, board: Board, dyn: bytearray, ea: bytearray, pp: int):
        self.b = board
        self.dyn = dyn
        self.ea = ea
        self.occ = {i for i, v in enumerate(dyn) if v}
        self.pp = pp
        #: Set when a rule would place two movable objects in one cell -- the
        #: interpreter allows it (`_cell_add` does not evict), this model cannot
        #: represent it, so the state is abandoned rather than guessed at.
        self.broken = False

    # -- construction ---------------------------------------------------------
    @staticmethod
    def from_engine(board: Board, eng, game) -> "Sim":
        n = board.H * board.W
        dyn, ea = bytearray(n), bytearray(n)
        pp = -1
        name = game.obj_idx_to_name
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i = board.index(r, c)
                for o in cell:
                    nm = name.get(o, "")
                    if nm in _EARTH_NAMES:
                        ea[i] = 1
                    d = _DYN_OF.get(nm)
                    if d is not None:
                        dyn[i] = d
                        if d == PLAYER:
                            pp = i
        return Sim(board, dyn, ea, pp)

    def copy(self) -> "Sim":
        s = Sim.__new__(Sim)
        s.b, s.dyn, s.ea = self.b, bytearray(self.dyn), bytearray(self.ea)
        s.occ, s.pp, s.broken = set(self.occ), self.pp, self.broken
        return s

    def key(self) -> bytes:
        return bytes(self.dyn) + bytes(self.ea)

    def won(self) -> bool:
        dyn = self.dyn
        return not any(dyn[i] in _MONST for i in self.occ)

    def monsters(self) -> list[int]:
        dyn = self.dyn
        return [i for i in self.occ if dyn[i] in _MONST]

    # -- cell predicates ------------------------------------------------------
    def obstacle(self, i: int) -> bool:
        """The game's `Obstacle` group: everything except background, the digits
        and the (momentary) Swap marker."""
        v = self.dyn[i]
        return bool(self.b.st[i] or self.ea[i] or (v and v != SWAP))

    def blocked(self, i: int) -> bool:
        """Occupied in the MOVABLE collision layer -- what actually stops a move.
        Ladders and earth are on the other layer and do not block (the earth is
        deleted by the dig rule before the player is moved into it)."""
        return self.b.st[i] == S_BRICK or self.dyn[i] != EMPTY

    # -- mutation helpers -----------------------------------------------------
    def _put(self, i: int, v: int) -> None:
        if v and self.dyn[i]:
            self.broken = True
        self.dyn[i] = v
        if v:
            self.occ.add(i)
        else:
            self.occ.discard(i)

    def _move(self, i: int, j: int) -> None:
        v = self.dyn[i]
        self.dyn[i] = EMPTY
        self.occ.discard(i)
        self._put(j, v)
        if v == PLAYER:
            self.pp = j

    # -- the turn -------------------------------------------------------------
    def press(self, direction: str, max_again: int = 50) -> None:
        """Run one keypress to a standstill (`PSEngine.step`'s again loop)."""
        indir = direction
        for _ in range(max_again):
            was_dyn, was_ea = bytes(self.dyn), bytes(self.ea)
            again = self._tick(indir)
            indir = None
            if self.dyn == was_dyn and self.ea == was_ea:
                break                       # no change -> the loop cannot repeat
            if not again or self.broken or self.won():
                break

    def _tick(self, indir) -> bool:
        """One rule pass + force resolution + the late gravity probe.

        Returns whether a rule asking for `again` fired; the caller ANDs that
        with "the grid actually changed", which is what stops the late rule --
        directionless about ladders, so it matches a player hanging on one
        forever -- from spinning."""
        b = self.b
        st, dyn, ea = b.st, self.dyn, self.ea
        W = b.W
        forces: dict[int, int] = {}
        again = False

        # -- input
        if indir is not None and indir != "action":
            if self.pp >= 0:
                forces[self.pp] = b.delta[indir]

        # -- down [ Gravity no Ladder | no obstacle no Ladder ]
        for i in tuple(self.occ):
            if dyn[i] in _GRAV and st[i] != S_LADDER:
                j = i + W
                if not (st[j] or ea[j] or (dyn[j] and dyn[j] != SWAP)):
                    forces[i] = W

        # -- up [ Player | Gravity ] / up [ Boulder | Gravity ]  (the Temp marks)
        temp = set()
        for i in tuple(self.occ):
            v = dyn[i]
            if (v == PLAYER or v == BOULDER) and dyn[i - W] in _GRAV:
                temp.add(i)

        p = self.pp
        d = forces.get(p)

        # -- horizontal [ > Player no Temp | Boulder no Temp | no Obstacle ]
        if d in (-1, 1) and p not in temp:
            q = p + d
            if dyn[q] == BOULDER and q not in temp and not self.obstacle(q + d):
                self._move(q, q + d)
                self._move(p, q)
                del forces[p]
                p, d = self.pp, None

        # -- [ Temp ] -> []   (the marks live for exactly one rule)
        # -- horizontal [ > Player | Boulder | no Obstacle ] -> [ Player || Boulder ]
        d = forces.get(p)
        if d in (-1, 1):
            q = p + d
            if dyn[q] == BOULDER and not self.obstacle(q + d):
                self._move(q, q + d)
                del forces[p]

        # -- the three up rules: climb, step off the top, or nothing at all
        d = forces.get(p)
        if d == -W:
            del forces[p]
            if st[p] == S_LADDER:
                q = p - W
                if st[q] == S_LADDER:
                    if self.blocked(q):
                        self.broken = True
                    self._move(p, q)
                    p = self.pp
                elif not self.obstacle(q):
                    self._move(p, q)
                    p = self.pp

        # -- down [ > Player | Earth ] -> cancel (no digging downwards)
        if forces.get(p) == W and ea[p + W]:
            del forces[p]

        # -- [ Player | DieingMonster3 ] -> [ | Player ]   (directionless)
        # ``p`` is -1 for the one tick in the middle of a swap where no Player
        # object exists at all; nothing below can match, and the guard keeps the
        # neighbour reads from wrapping round the end of the array.
        for dd in (-W, W, -1, 1) if p >= 0 else ():
            if dyn[p + dd] == DM3:
                forces.pop(p, None)
                self._put(p + dd, EMPTY)
                self._move(p, p + dd)
                p = self.pp
                break

        # -- the death animation, one stage per tick, newest stage last
        for i in tuple(self.occ):
            if dyn[i] == DM2:
                dyn[i] = DM3
                again = True
        for i in tuple(self.occ):
            if dyn[i] == DM1:
                dyn[i] = DM2
                again = True

        # -- horizontal [ > Player | Monster ] -> [ Player | DieingMonster1 ]
        d = forces.get(p)
        if d in (-1, 1) and dyn[p + d] in _MONST:
            dyn[p + d] = DM1
            del forces[p]
            again = True

        # -- horizontal [ > Player | Destructible ] -> [ > Player | ]  (dig)
        d = forces.get(p)
        if d in (-1, 1):
            q = p + d
            if ea[q]:
                ea[q] = 0
            elif dyn[q] in _MONST:
                self._put(q, EMPTY)

        # -- the swap trio, in file order
        if self._swap_rules(indir):
            again = True

        # -- force resolution: everyone with a force moves once, if it can
        while forces:
            stuck = True
            for i, dd in tuple(forces.items()):
                j = i + dd
                if not self.blocked(j):
                    del forces[i]
                    self._move(i, j)
                    stuck = False
            if stuck:
                break

        # -- late down [ Gravity | no obstacle no Ladder ] -> again
        for i in self.occ:
            if dyn[i] in _GRAV:
                j = i + W
                if not (st[j] or ea[j] or (dyn[j] and dyn[j] != SWAP)):
                    again = True
                    break
        return again

    def _swap_rules(self, indir) -> bool:
        """``[Swap][Player] -> [Catgirl][Player]``,
        ``[Catgirl][Swap] -> [Swap][Player] again`` and, on an ACTION turn,
        ``[action Player][Catgirl] -> [Catgirl][Swap] again``.

        Read in file order they hand the body over in three ticks and leave both
        characters exactly where they stood; the middle tick has no Player object
        on the board at all (and a Swap is not `Gravity`, so nothing falls out of
        it)."""
        dyn = self.dyn
        girl = swap = player = -1
        for i in self.occ:
            v = dyn[i]
            if v == CATGIRL:
                girl = i
            elif v == SWAP:
                swap = i
            elif v == PLAYER:
                player = i
        again = False
        if swap >= 0 and player >= 0:                 # rule 1: finish the swap
            dyn[swap] = CATGIRL
            girl, swap = swap, -1
        if girl >= 0 and swap >= 0:                   # rule 2: hand the body over
            dyn[girl] = SWAP
            dyn[swap] = PLAYER
            self.pp = swap
            girl, swap, player = swap, girl, swap
            again = True
        if indir == "action" and player >= 0 and girl >= 0:   # rule 3: start it
            dyn[player] = CATGIRL
            dyn[girl] = SWAP
            self.pp = -1
            again = True
        return again


# ---------------------------------------------------------------------------
# Macro expansion: walk, then do the one thing that changes the board
# ---------------------------------------------------------------------------

def _walk_step(sim: Sim, p: int, d: int) -> int:
    """Where a player standing at ``p`` ends up if pressing ``d`` moves ONLY the
    player; -1 if that press is blocked, is a no-op, or touches anything else.
    ``sim`` must be `stripped` (no player on the board).

    This is the analytic twin of `Sim.press` restricted to the moves that leave
    the rest of the board alone, and it is what makes the search affordable: a
    walk of 20 presses costs one BFS edge each instead of 20 simulated turns.
    Anything it declines becomes a macro candidate and IS simulated, so a
    conservative "-1" only ever costs speed."""
    b = sim.b
    st, dyn, ea = b.st, sim.dyn, sim.ea
    W = b.W
    if dyn[p - W] in _GRAV:
        return -1                # leaving would drop whatever is on our head
    if d == -W:                                    # up: only ever on a ladder
        if st[p] != S_LADDER:
            return -1
        q = p - W
        if st[q] == S_LADDER:
            return -1 if sim.blocked(q) else q
        return -1 if sim.obstacle(q) else q
    if d == W:                                     # down: step off / climb down
        q = p + W
        if ea[q] or sim.blocked(q):
            return -1
    else:                                          # sideways
        q = p + d
        if ea[q] or dyn[q] != EMPTY or st[q] == S_BRICK:
            return -1                # a dig, a push, a kill -- not a plain walk
    while st[q] != S_LADDER:                       # then fall until supported
        j = q + W
        if st[j] or ea[j] or dyn[j] != EMPTY:
            break
        q = j
    return q


def stripped(sim: Sim) -> Sim:
    """A copy with the player lifted off the board, keeping ``pp`` as the cell it
    was standing on.

    Every walk query is "could the player be HERE, and where does pressing d take
    it" -- asked about cells the player is not currently on. With the player left
    on the board its own cell reads as occupied, so it blocks its own return and,
    worse, stops a fall that passes through the cell it is leaving. That is
    invisible while a macro is replayed press by press (the model gets it right
    anyway, the walk graph is merely pessimistic) and becomes a wrong board the
    moment `apply_macro` starts teleporting."""
    s = sim.copy()
    s.dyn[s.pp] = EMPTY
    s.occ.discard(s.pp)
    return s


def _reachable(base: Sim) -> dict[int, tuple[int, int]]:
    """BFS over the walk graph from ``base.pp``: ``{cell: (previous, direction)}``.
    ``base`` must be `stripped`.

    Directed on purpose -- gravity is one-way, so "walked off a ledge" is an edge
    with no reverse, and the standard sokoban trick of keying a state by the
    player's whole region would merge states that are not interchangeable."""
    parent = {base.pp: (-1, 0)}
    queue = deque((base.pp,))
    dirs = base.b.dirs
    while queue:
        p = queue.popleft()
        for d in dirs:
            q = _walk_step(base, p, d)
            if q >= 0 and q not in parent:
                parent[q] = (p, d)
                queue.append(q)
    return parent


def _path_to(parent: dict, cell: int) -> list[int]:
    out = []
    while parent[cell][0] >= 0:
        cell, d = parent[cell]
        out.append(d)
    out.reverse()
    return out


def _candidates(sim: Sim, p: int) -> list[int]:
    """The presses at ``p`` that could change something other than the player:
    a push / dig / kill sideways, a step that un-supports a load overhead, and
    ACTION while a catgirl is on the board."""
    b = sim.b
    dyn, ea = sim.dyn, sim.ea
    W = b.W
    out = []
    unsupport = dyn[p - W] in _GRAV
    for d in (-1, 1):
        q = p + d
        if ea[q] or dyn[q] in (BOULDER, MFALL, MFLOAT, DM1, DM2, DM3):
            out.append(d)
        elif unsupport and not sim.blocked(q):
            out.append(d)
    if unsupport:
        for d in (-W, W):
            if _walk_step_ignoring_load(sim, p, d):
                out.append(d)
    return out


def _walk_step_ignoring_load(sim: Sim, p: int, d: int) -> bool:
    """Whether up/down would move the player at all, ignoring the load overhead
    that made `_walk_step` refuse. Only used to decide if that refusal is a real
    macro or a plain no-op."""
    b, st = sim.b, sim.b.st
    W = b.W
    if d == -W:
        if st[p] != S_LADDER:
            return False
        q = p - W
        return not (sim.blocked(q) if st[q] == S_LADDER else sim.obstacle(q))
    q = p + W
    return not (sim.ea[q] or sim.blocked(q))


class Macro:
    """One decision: the walk that sets it up, and the press that does it.
    ``press`` is a grid offset, or 0 for ACTION (the swap has no direction)."""

    __slots__ = ("walk", "press", "target")

    def __init__(self, walk, press, target):
        self.walk = walk
        self.press = press
        self.target = target


def expand(sim: Sim) -> list[Macro]:
    """Every macro available from ``sim``: for each cell the player can walk to,
    each press there that changes the board."""
    if any(sim.dyn[i] in (DM1, DM2, DM3) for i in sim.occ):
        return _expand_simulated(sim)
    base = stripped(sim)
    parent = _reachable(base)
    has_girl = any(sim.dyn[i] == CATGIRL for i in sim.occ)
    out = []
    for cell in parent:
        walk = None
        for d in _candidates(base, cell):
            if walk is None:
                walk = _path_to(parent, cell)
            out.append(Macro(walk, d, cell))
        if has_girl:
            if walk is None:
                walk = _path_to(parent, cell)
            out.append(Macro(walk, 0, cell))
    return out


def _expand_simulated(sim: Sim) -> list[Macro]:
    """`expand` for the one state shape the analytic walk cannot describe: a
    board holding a dying monster.

    ``[ Player | DieingMonster3 ] -> [ | Player ]`` is directionless and needs no
    keypress of its own, so merely *walking past* a ripe corpse consumes it and
    yanks the player sideways -- a "walk" that changes the board. Rather than
    special-case that inside `_walk_step`, states with a corpse on them classify
    every press by simulating it. They are rare (a corpse only survives a press
    if the player fell away from it mid-animation), so the cost never shows."""
    names = sim.b.name
    parent = {sim.pp: (-1, 0)}
    sims = {sim.pp: sim}
    queue = deque((sim.pp,))
    out = []
    while queue:
        p = queue.popleft()
        cur = sims[p]
        for d in sim.b.dirs + (0,):
            s = cur.copy()
            s.press(names[d])
            if s.broken or s.key() == cur.key():
                continue
            if not s.won() and _only_player_moved(cur, s):
                if s.pp not in parent:
                    parent[s.pp] = (p, d)
                    sims[s.pp] = s
                    queue.append(s.pp)
            else:
                out.append(Macro(_path_to(parent, p), d, p))
    return out


def _only_player_moved(a: Sim, b: Sim) -> bool:
    if a.ea != b.ea or a.pp == b.pp or b.pp < 0:
        return False
    dyn = bytearray(b.dyn)
    dyn[b.pp] = EMPTY
    dyn[a.pp] = PLAYER
    return dyn == a.dyn


def apply_macro(sim: Sim, macro: Macro, board: Board) -> Sim | None:
    """Run a macro on a copy of ``sim``, or None if its press turns out to do
    nothing.

    The walk is applied by TELEPORTING the player to the macro's cell instead of
    replaying it press by press. That is exact, not an approximation: every edge
    of the walk graph is a press that provably moves nothing but the player, so
    the composition of a walk is precisely "the same board with the player
    somewhere else" -- and it turns a 20-press walk into one array write, which
    is most of the search's wall clock.

    Dropping the no-ops matters as much: without it, "walk to X and press into a
    wall" would enter the queue as a genuinely new state (the key holds the
    player's cell) for every X, and the macro decomposition would collapse back
    into the primitive search it exists to avoid."""
    s = sim.copy()
    if macro.target != s.pp:
        s._move(s.pp, macro.target)
    before = s.key()
    s.press(board.name[macro.press])
    return None if s.key() == before else s


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class Bound:
    """Admissible press-count bounds for one level.

    The relaxation keeps the two constraints that actually shape a Catrap board
    and drops every other one:

      * **bricks still block**, and **up is only possible from a ladder cell** --
        there is no jump in this game, so height is the scarce resource and a
        heuristic that ignores it is worthless on the tower levels. (The one
        other way up is the directionless corpse rule pulling the player into a
        `DieingMonster3` above it, so an up-step INTO any cell a monster could
        ever occupy is allowed too. Monsters never move sideways, so "the cell,
        and everything below it in its column" is a static superset of that.)
      * boulders, earth, the catgirl and the monsters are all walked through for
        free, and **falling costs nothing** -- one press can drop the player any
        distance, so a down edge has to be a 0-edge for the bound to hold.

    Because what is left is STATIC, the distance field from a cell is computed
    once and cached, and the field from a monster's own cell is exactly the
    "where do I go next" leg: killing a monster leaves the player standing in
    its cell.
    """

    def __init__(self, board: Board, start: Sim):
        self.b = board
        n = board.H * board.W
        self.up_ok = bytearray(n)
        st, W = board.st, board.W
        for i in start.occ:
            if start.dyn[i] in _MONST:
                j = i
                while st[j] != S_BRICK:              # ... and where it may fall
                    self.up_ok[j] = 1
                    j += W
        self.fields: dict[int, dict[int, int]] = {}

    def field(self, cell: int) -> dict[int, int]:
        """0-1 BFS cost-to-reach from ``cell``, lazily relaxed: a cell can be
        re-reached by a 0-edge after a 1-edge found it first, and treating the
        first discovery as final would over-estimate -- which makes the whole
        heuristic inadmissible rather than merely loose."""
        got = self.fields.get(cell)
        if got is not None:
            return got
        st, W, up_ok = self.b.st, self.b.W, self.up_ok
        dist: dict[int, int] = {}
        dq = deque(((cell, 0),))
        while dq:
            i, d0 = dq.popleft()
            if dist.get(i, 1 << 30) <= d0:
                continue
            dist[i] = d0
            j = i + W
            if st[j] != S_BRICK:                     # falling is free
                dq.appendleft((j, d0))
            for dd in (-1, 1):
                j = i + dd
                if st[j] != S_BRICK:
                    dq.append((j, d0 + 1))
            j = i - W
            if st[j] != S_BRICK and (st[i] == S_LADDER or up_ok[j]):
                dq.append((j, d0 + 1))
        self.fields[cell] = dist
        return dist

    def kill_cost(self, frm: int, mon: int) -> int:
        """Presses needed, at least, to be standing where ``mon`` is: walk beside
        it and press into it, or walk straight into what is left of it."""
        f = self.field(frm)
        best = f.get(mon, 1 << 30)
        for d in (-1, 1):
            v = f.get(mon + d)
            if v is not None and v + 1 < best:
                best = v + 1
        return best

    def h(self, sim: Sim) -> int:
        """Lower bound on the presses left, or -1 when a monster is walled off
        for good.

        The bound is the minimum spanning tree over {player} U {monsters} with
        edge weights that lower-bound travel in either direction: any winning
        continuation walks a connected route through all of those cells, and a
        connected subgraph costs at least its MST. It is floored by the monster
        count, since every kill is a press of its own.

        **The catgirl breaks the MST**, and it is worth being explicit about why:
        ACTION moves the *controlled body* to wherever the other one is standing,
        for one press, from anywhere on the board -- so a route between two
        monsters is not bounded by the distance between them at all. (Measured:
        the MST over-estimated by up to 11 presses on level 31, which makes the
        w=1 search's plans quietly non-optimal while they still carry
        "optimal" labels.) With a second body on the board the bound drops to the
        one shape that survives a teleport of unknown length: the cheapest FIRST
        kill -- by walking there, or by swapping and walking from the other body
        -- plus one press for each monster after it."""
        mon = sim.monsters()
        k = len(mon)
        if not k:
            return 0
        big = 1 << 30
        girl = -1
        for i in sim.occ:
            if sim.dyn[i] == CATGIRL:
                girl = i
                break
        if girl >= 0:
            first = big
            for m in mon:
                c = self.kill_cost(sim.pp, m)
                g = self.kill_cost(girl, m)
                if g < big:
                    c = min(c, g + 1)
                first = min(first, c)
            if first >= big:
                return -1
            return max(k, first + k - 1)
        nodes = [sim.pp] + mon
        d = [big] * (k + 1)
        d[0] = 0
        used = [False] * (k + 1)
        total = 0
        for _ in range(k + 1):
            u, bu = -1, big
            for i in range(k + 1):
                if not used[i] and d[i] < bu:
                    u, bu = i, d[i]
            if u < 0:
                return -1                      # some monster cannot be reached
            used[u] = True
            total += bu
            for v in range(1, k + 1):
                if not used[v]:
                    w = self.kill_cost(nodes[u], mon[v - 1])
                    if u:                      # symmetric: either leg may be the
                        w = min(w, self.kill_cost(mon[v - 1], nodes[u]))
                    if w < d[v]:
                        d[v] = w
        return max(total, k)


class Plan(list):
    """The press sequence, carrying the optimal SET for each of its steps."""

    optsets: list

    def __init__(self, presses, optsets):
        super().__init__(presses)
        self.optsets = optsets


class CatrapExpert(PSExpert):
    """Macro A* over `Sim`, with every plan re-verified on the interpreter.

    `PSExpert.plan`'s memo and the engine-blackbox `_astar` are both replaced:
    the search runs on the model, so it needs neither the engine snapshots nor
    the frozenset state key."""

    scope_by_level = True

    #: Wall-clock ceiling for one level's search. The late boards are not
    #: solvable at any budget, so this is a "give up" dial, not a quality one.
    time_budget: float = 25.0
    #: Weights tried in order. w=1 gives shortest plans; the ladder only exists
    #: for the levels a shortest-plan search cannot finish in `time_budget`.
    weights: tuple[int, ...] = (1, 3)
    #: Last resort after the weight ladder (see `_beam`). Wider and deeper than
    #: A* can afford, and correspondingly less optimal. The width is the whole
    #: dial: a narrow beam does not run out of time on these boards, it runs out
    #: of BOARD -- every survivor walks into an unwinnable position and the
    #: frontier empties (level 38 dies at depth 25 with width 250, and wins at
    #: depth 29 with width 4000, both in well under a minute).
    beam_budget: float = 180.0
    beam_width: int = 4000
    beam_depth: int = 80

    def setup(self) -> None:
        self._boards: dict[int, Board] = {}
        self._plans: dict[int, Plan | None] = {}
        self._disk = self._load_disk()
        #: A private adapter to replay candidate plans on. Verification has to
        #: reload a level and step it to the end, which would leave the RECORDING
        #: adapter mid-trajectory (`record_level` reads `_current_frame` right
        #: after it plans); a second interpreter keeps the check side-effect free.
        self._vgame: PuzzleScriptAdapter | None = None

    # -- geometry -------------------------------------------------------------
    def _vg(self) -> PuzzleScriptAdapter:
        if self._vgame is None:
            self._vgame = PuzzleScriptAdapter(GAME_NAME, seed=0)
        return self._vgame

    def board(self, level: int) -> Board:
        b = self._boards.get(level)
        if b is None:
            g = self._vg()
            g.set_level(level)
            b = self._boards[level] = Board(g._engine, g._game)
        return b

    def sim(self, level: int) -> Sim:
        """The level's START state. Read off the private adapter, not the caller's
        engine, so planning never depends on where the recording game happens to
        be standing."""
        g = self._vg()
        g.set_level(level)
        return Sim.from_engine(self.board(level), g._engine, g._game)

    # -- search ---------------------------------------------------------------
    def _astar_model(self, start: Sim, board: Board, bound: Bound, weight: int,
                     deadline: float) -> list[Macro] | None:
        if start.won():
            return []
        counter = 0
        pq = [(weight * bound.h(start), 0, counter, start, None)]
        best_g = {start.key(): 0}
        nodes = 0
        while pq:
            _f, g, _c, sim, chain = heapq.heappop(pq)
            if sim.won():
                # Returning a win at GENERATION time would cost optimality: the
                # queue can still hold a node whose f is lower than the winning
                # path's cost. Queued with h = 0 and answered on POP, w = 1 plans
                # are shortest, full stop.
                out = []
                while chain is not None:
                    out.append(chain[1])
                    chain = chain[0]
                out.reverse()
                return out
            for macro in expand(sim):
                nxt = apply_macro(sim, macro, board)
                nodes += 1
                if nxt is None or nxt.broken:
                    continue
                k = nxt.key()
                ng = g + len(macro.walk) + 1
                if best_g.get(k, 1 << 30) <= ng:
                    continue
                hh = 0 if nxt.won() else bound.h(nxt)
                if hh < 0:
                    continue          # a monster is walled off: a dead board
                best_g[k] = ng
                counter += 1
                heapq.heappush(pq, (ng + weight * hh, ng, counter,
                                    nxt, (chain, macro)))
            if nodes >= self.node_cap or time.time() > deadline:
                return None
        return None

    def _beam(self, start: Sim, board: Board, bound: Bound,
              deadline: float) -> list[Macro] | None:
        """A width-capped breadth-first beam over the same macros, for the levels
        A* cannot finish.

        What A* runs out of here is not the heuristic's quality but its shape:
        the bound is a travel estimate, and the middle of a long solution is a
        dozen decisions that rearrange boulders and park the catgirl without the
        player getting one step closer to a monster. Over that stretch the
        estimate is FLAT, weighting it changes nothing, and A* degenerates to
        uniform-cost search at a depth where that is hopeless. A beam spends the
        same budget on breadth at every depth and uses the bound only to break
        ties, which is all a flat estimate is good for. The trade is plan length:
        beam plans win, and are engine-verified, but they wander."""
        if start.won():
            return []
        frontier = [(start, None)]
        seen = {start.key()}
        for _ in range(self.beam_depth):
            kids = []
            for sim, chain in frontier:
                for macro in expand(sim):
                    nxt = apply_macro(sim, macro, board)
                    if nxt is None or nxt.broken:
                        continue
                    if nxt.won():
                        out = [macro]
                        while chain is not None:
                            out.append(chain[1])
                            chain = chain[0]
                        out.reverse()
                        return out
                    k = nxt.key()
                    if k in seen:
                        continue
                    seen.add(k)
                    hh = bound.h(nxt)
                    if hh < 0:
                        continue
                    kids.append((hh, nxt, (chain, macro)))
                if time.time() > deadline:
                    return None
            if not kids:
                return None                        # the reachable space closed
            kids.sort(key=lambda kid: kid[0])
            frontier = [(s, c) for _h, s, c in kids[:self.beam_width]]
        return None

    def _rewalk(self, start: Sim, board: Board,
                macros: list[Macro]) -> list[Macro] | None:
        """Replay a macro sequence by its TARGETS, recomputing each walk from the
        state it is actually taken in, and truncating the moment it wins.

        Returns None if some target is no longer reachable -- which is exactly
        what makes it a validity test for `_shorten`'s deletions. Recomputing the
        walks is not optional: a macro's stored walk belongs to the state it was
        generated in, and `_label` replays walks as real presses."""
        sim = start.copy()
        out: list[Macro] = []
        for macro in macros:
            parent = _reachable(stripped(sim))
            if macro.target not in parent:
                return None
            fresh = Macro(_path_to(parent, macro.target), macro.press,
                          macro.target)
            nxt = apply_macro(sim, fresh, board)
            if nxt is None or nxt.broken:
                return None
            out.append(fresh)
            sim = nxt
            if sim.won():
                return out
        return None

    def _shorten(self, start: Sim, board: Board,
                 macros: list[Macro]) -> list[Macro]:
        """Drop the decisions a beam plan did not need.

        A beam wins by breadth, so its route wanders: it parks a boulder it never
        uses, swaps bodies and swaps back. Deleting one macro and re-deriving the
        rest is a complete test of whether it mattered -- the replay either still
        reaches the win or it does not. Two greedy passes, because deleting one
        detour often makes the next one deletable; a full fixpoint is quadratic
        for a couple of extra presses.

        This is not cosmetic. `PuzzleScriptAdapter` ends a level at 200 actions,
        so a plan that wanders past `max_plan` is thrown away entirely (see
        `_verify`) and the level counts as unsolved."""
        for _ in range(2):
            i, n = 0, len(macros)
            while i < len(macros):
                shorter = self._rewalk(start, board,
                                       macros[:i] + macros[i + 1:])
                if shorter is None:
                    i += 1
                else:
                    macros = shorter
            if len(macros) == n:
                break
        return macros

    def _search_level(self, level: int) -> Plan | None:
        board = self.board(level)
        start = self.sim(level)
        bound = Bound(board, start)
        if bound.h(start) < 0:
            return None                      # unwinnable before a key is pressed
        searches = [(w, self.time_budget) for w in self.weights]
        searches.append((0, self.beam_budget))          # 0 = the beam
        for weight, budget in searches:
            deadline = time.time() + budget
            macros = (self._beam(start, board, bound, deadline) if weight == 0
                      else self._astar_model(start, board, bound, weight,
                                             deadline))
            if macros is None:
                continue
            if weight == 0:
                macros = self._shorten(start, board, macros)
            plan = self._label(start, board, macros)
            if self._verify(level, plan):
                return plan
        return None

    # -- optimal-action labelling ---------------------------------------------
    def _label(self, start: Sim, board: Board, macros: list[Macro]) -> Plan:
        """Flatten the macros into presses, tagging each with the full set of
        presses that are equally good there.

        Inside a macro's walk every shortest route to the macro's cell reaches
        the SAME state for the SAME cost, so all of their first steps are equally
        optimal -- and on these open boards there are usually several. The press
        that ends the macro is labelled with itself."""
        names = board.name
        sim = start.copy()
        presses: list[str] = []
        optsets: list[list[str]] = []
        for macro in macros:
            base = stripped(sim)
            back = _back_dists(base, macro.target)
            for d in macro.walk:
                here = back.get(sim.pp, 0)
                ties = [names[e] for e in board.dirs
                        if _tie(base, sim.pp, e, back, here)]
                optsets.append(ties or [names[d]])
                presses.append(names[d])
                sim.press(names[d])
            act = names[macro.press]
            presses.append(act)
            optsets.append([act])
            sim.press(act)
        return Plan(presses, optsets)

    #: Presses a plan may not exceed. `PuzzleScriptAdapter` gives every level a
    #: 200-action budget and turns GAME_OVER on when it runs out, so a longer
    #: plan would be recorded as a LOSS -- and since `solve_episode` needs EVERY
    #: discovered level to win, one over-long plan would fail every seed. The
    #: beam is the only search that gets anywhere near this.
    max_plan: int = 190

    def _verify(self, level: int, plan: list) -> bool:
        """Replay the plan on the real interpreter and keep it only if it wins.

        This is the contract that makes the model safe to trust: a modelling slip
        loses a level, it can never ship a trajectory that does not win. It is
        also what keeps the disk cache honest -- a cached plan is re-verified
        before it is used, so an edited level file can never silently ship a plan
        that no longer applies."""
        if len(plan) > self.max_plan:
            return False
        eng = self._vg()._engine
        self._vgame.set_level(level)
        for direction in plan:
            eng.step(direction)
            if eng.check_win():
                return True
        return eng.check_win()

    # -- PSExpert interface ---------------------------------------------------
    def _signature(self, level: int) -> str:
        """A fingerprint of the level's start, so a cache entry from a different
        version of the level file (in particular a cached "unsolvable") is a miss
        rather than a stale answer."""
        board = self.board(level)
        sim = self.sim(level)
        # zlib, not hash(): `hash` is salted per process (PYTHONHASHSEED), so a
        # signature written by one run would never match in the next.
        return (f"{board.H}x{board.W}:"
                f"{zlib.crc32(bytes(board.st) + sim.key()):08x}")

    def plan(self, eng, level: int | None = None) -> Plan | list | None:
        if eng.check_win():
            return []
        if level is None:
            return None
        if level in self._plans:
            return self._plans[level]
        found = None
        cached = self._disk.get(level)
        sig = self._signature(level)
        fresh = cached is not None and cached.get("sig") == sig
        if fresh and cached["plan"] is not None:
            found = Plan(cached["plan"], cached["optsets"])
            if not self._verify(level, found):
                found, fresh = None, False
        if not fresh:
            found = self._search_level(level)
            self._disk[level] = {
                "sig": sig,
                "plan": None if found is None else list(found),
                "optsets": None if found is None else found.optsets,
            }
            self._save_disk()
        self._plans[level] = found
        return found

    # -- disk cache -----------------------------------------------------------
    @staticmethod
    def _load_disk() -> dict:
        try:
            raw = json.loads(PLAN_CACHE.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:                                    # noqa: BLE001
            return {}

    def _save_disk(self) -> None:
        """Write atomically -- shards started together would otherwise interleave
        into a truncated file (see `parallelize_generator`)."""
        try:
            PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PLAN_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self._disk[k] for k in sorted(self._disk)}, indent=1))
            os.replace(tmp, PLAN_CACHE)
        except OSError:
            pass                                             # cache is optional


def _back_dists(base: Sim, target: int) -> dict[int, int]:
    """Walk-graph distances TO ``target`` (the graph is directed, so this is a
    BFS over the reversed edges of the forward-reachable set). ``base`` must be
    `stripped`."""
    fwd = _reachable(base)
    rev: dict[int, list[int]] = {}
    for cell in fwd:
        for d in base.b.dirs:
            q = _walk_step(base, cell, d)
            if q >= 0 and q in fwd:
                rev.setdefault(q, []).append(cell)
    dist = {target: 0}
    dq = deque((target,))
    while dq:
        i = dq.popleft()
        for j in rev.get(i, ()):
            if j not in dist:
                dist[j] = dist[i] + 1
                dq.append(j)
    return dist


def _tie(base: Sim, at: int, d: int, back: dict[int, int], here: int) -> bool:
    q = _walk_step(base, at, d)
    return q >= 0 and back.get(q, 1 << 30) == here - 1


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class CatrapSolver(PSAStarSolver):
    game_id = "puzzlescript_catrap"
    game_name = GAME_NAME
    expert_cls = CatrapExpert

    #: Macro nodes per search; the wall-clock ceiling in `CatrapExpert` is the
    #: binding limit in practice.
    node_cap = 2_000_000
    #: Plans on the solved levels run to ~150 presses.
    max_steps = 400

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The set of presses that are equally good at this step (see
        `CatrapExpert._label`). Falls back to the press about to be taken, so no
        expert step ever ships unlabelled.

        On a level solved at w=1 this is honest optimality: the macro
        decomposition is cost-preserving (every press sequence is some walk +
        effectful-press sequence, and the walks are shortest), `Bound.h` is
        admissible, and the search answers the goal on POP -- so the plan really
        is a shortest one and every tie is a real tie. On the handful of levels
        that needed w=3 or the beam it is the expert's own best line rather than
        a proven optimum, which is the same bargain the other ps: beam
        generators make."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets):
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Model audit
# ---------------------------------------------------------------------------

def _engine_view(board: Board, eng, game) -> tuple:
    """The interpreter's grid in the model's own terms, for comparison."""
    s = Sim.from_engine(board, eng, game)
    return bytes(s.dyn), bytes(s.ea)


def _audit_cell(sim: Sim, base: Sim, at: int, names: dict) -> list[str]:
    """The two analytic claims, checked at one cell by simulating every press
    from a board with the player standing there.

      * `_walk_step` says "this press moves only the player, to cell q" -- so
        simulating it must move only the player, to exactly q (otherwise a
        teleported macro lands on a board that never existed);
      * anything `_walk_step` declines must either be a no-op or be listed by
        `_candidates` (otherwise the move is invisible to the search, and the
        levels that need it are silently unsolvable)."""
    out = []
    cands = set(_candidates(base, at))
    for d in sim.b.dirs:
        s = sim.copy()
        if s.pp != at:
            s._move(s.pp, at)
        pre = s.copy()
        s.press(names[d])
        if s.broken:
            continue
        q = _walk_step(base, at, d)
        pure = _only_player_moved(pre, s)
        if q >= 0:
            if not pure or s.pp != q:
                out.append(f"walk {names[d]} at {at}: predicted {q}, "
                           f"pure={pure} landed={s.pp}")
        elif s.key() != pre.key() and d not in cands:
            out.append(f"candidate {names[d]} at {at}: press changes the board "
                       f"but is not a macro candidate")
    return out


def audit_state(sim: Sim, rng: random.Random, samples: int = 3) -> list[str]:
    """Check the search's analytic shortcuts against the model itself.

    Three claims: the two in `_audit_cell`, at the player's cell and at a random
    sample of the cells the walk graph says it can reach, plus the property the
    whole macro machinery rests on -- that WALKING to a cell and TELEPORTING to
    it leave the very same board (`apply_macro`). The sample is what makes this
    affordable to run over every level; the cells the search actually uses are
    exactly these."""
    names = sim.b.name
    base = stripped(sim)
    out = _audit_cell(sim, base, sim.pp, names)
    if any(sim.dyn[i] == CATGIRL for i in sim.occ) is False:
        s = sim.copy()
        s.press("action")
        if s.key() != sim.key():
            out.append("ACTION changed the board with no catgirl on it")
    parent = _reachable(base)
    cells = [c for c in parent if c != sim.pp]
    rng.shuffle(cells)
    for cell in cells[:samples]:
        walked = sim.copy()
        for d in _path_to(parent, cell):
            walked.press(names[d])
        ported = sim.copy()
        ported._move(ported.pp, cell)
        if walked.key() != ported.key():
            out.append(f"walk to {cell} is not a teleport: "
                       f"landed at {walked.pp}")
            continue
        out += _audit_cell(sim, base, cell, names)
    return out


def fuzz(trials: int = 8, steps: int = 60, levels=None, verbose: bool = True) -> int:
    """Play the same random presses through the interpreter and through `Sim`
    and compare the boards after every one.

    The model is a re-implementation of eight interleaved rules; this is the only
    thing that keeps it honest. Returns the number of diverging presses."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    bad = 0
    for level in (range(game.n_levels) if levels is None else levels):
        game.set_level(level)
        board = Board(eng, game._game)
        for t in range(trials):
            game.set_level(level)
            sim = Sim.from_engine(board, eng, game._game)
            rng = random.Random(f"catrap:fuzz:{level}:{t}")
            for step in range(steps):
                direction = rng.choice(DIRS + ("action",))
                eng.step(direction)
                sim.press(direction)
                if sim.broken:
                    break
                if _engine_view(board, eng, game._game) != (bytes(sim.dyn),
                                                            bytes(sim.ea)):
                    bad += 1
                    if verbose:
                        print(f"  L{level} trial {t} step {step}: "
                              f"DIVERGED on {direction}")
                        _dump(board, eng, game._game, sim)
                    break
                if eng.check_win() != sim.won():
                    bad += 1
                    print(f"  L{level} trial {t} step {step}: win disagrees")
                    break
                if sim.won():
                    break
                for msg in audit_state(sim, rng):
                    bad += 1
                    if verbose:
                        print(f"  L{level} trial {t} step {step}: {msg}")
    return bad


_GLYPH = {EMPTY: ".", BOULDER: "o", PLAYER: "C", CATGIRL: "J", MFALL: "D",
          MFLOAT: "G", DM1: "1", DM2: "2", DM3: "3", SWAP: "S"}


def _text(board: Board, dyn, ea) -> list[str]:
    rows = []
    for r in range(board.H):
        line = ""
        for c in range(board.W):
            i = r * board.W + c
            if dyn[i]:
                line += _GLYPH[dyn[i]]
            elif ea[i]:
                line += "%"
            elif board.st[i] == S_BRICK:
                line += "#"
            elif board.st[i] == S_LADDER:
                line += "L"
            else:
                line += " "
        rows.append(line)
    return rows


def _dump(board: Board, eng, game, sim: Sim) -> None:
    ed, ee = _engine_view(board, eng, game)
    left, right = _text(board, ed, ee), _text(board, sim.dyn, sim.ea)
    print("      engine" + " " * (board.W - 6) + " | model")
    for a, b in zip(left, right):
        print(f"      {a} | {b}{'   <<<' if a != b else ''}")


def _plan_report(levels=None, retry: bool = False) -> None:
    """Plan every level and print presses / wall clock -- the "how much of the
    game is solved" check and the search-tuning loop.

    ``retry`` drops the cached failures first. They are cached on purpose --
    otherwise every shard of `parallelize_generator` would re-burn the search
    budget on the unsolvable levels at startup -- so re-searching them is an
    explicit request, not the default."""
    solver = CatrapSolver()
    game = solver.make_game(0)
    expert = CatrapExpert(game, node_cap=solver.node_cap)
    if retry:
        for lvl in [k for k, v in expert._disk.items() if v["plan"] is None]:
            del expert._disk[lvl]
    solved = total = 0
    for level in (range(game.n_levels) if levels is None else levels):
        game.set_level(level)
        t0 = time.time()
        plan = expert.plan(game._engine, level)
        dt = time.time() - t0
        total += 1
        if plan is None:
            print(f"  L{level:3d}: UNSOLVED                 {dt:6.1f}s")
            continue
        solved += 1
        ties = sum(len(s) - 1 for s in getattr(plan, "optsets", []))
        over = _bound_violation(expert, level, plan)
        print(f"  L{level:3d}: {len(plan):4d} presses  win=True  "
              f"{ties:4d} tie-presses  {dt:6.1f}s"
              f"{'' if over <= 0 else f'  BOUND OVER-ESTIMATES BY {over}'}")
    print(f"  solved {solved}/{total}")


def _bound_violation(expert: CatrapExpert, level: int, plan: list) -> int:
    """How far `Bound.h` over-estimates anywhere along ``plan`` (0 = never).

    A heuristic that exceeds the true remaining cost is not merely loose, it
    makes the w=1 search's plans non-optimal while still looking optimal -- and
    nothing else would notice. Replaying a known winning line and checking the
    estimate against what it actually cost is the cheap one-sided test for
    that."""
    board = expert.board(level)
    sim = expert.sim(level)
    bound = Bound(board, sim)
    worst = 0
    for i, press in enumerate(plan):
        worst = max(worst, bound.h(sim) - (len(plan) - i))
        sim.press(press)
    return worst


if __name__ == "__main__":
    if "--fuzz" in sys.argv:
        args = [a for a in sys.argv[1:] if not a.startswith("-")]
        lv = [int(x) for x in args[0].split(",")] if args else None
        violations = fuzz(levels=lv)
        print(f"fuzz: {violations} divergences")
        sys.exit(1 if violations else 0)
    if "--plans" in sys.argv:
        args = [a for a in sys.argv[1:] if not a.startswith("-")]
        lv = [int(x) for x in args[0].split(",")] if args else None
        _plan_report(lv, retry="--retry" in sys.argv)
        sys.exit(0)
    sys.exit(CatrapSolver.main())
