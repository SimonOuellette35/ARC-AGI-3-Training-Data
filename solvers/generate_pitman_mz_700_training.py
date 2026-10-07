"""Generate Phase-1 training data for the PuzzleScript game ps:pitman_mz_700
("Pitman MZ-700" by BdR, after Yutaka Isokawa's Pitman, 50 levels).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: a
tick-accurate MODEL of the mechanic, a macro A* over it, and the optimal-action
labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_pitman_mz_700",
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
Win = ``no Gold``: collect every nugget. There is no death and no timer -- the
only way to lose is to make the board unwinnable, which is silent (dig the wrong
hole and you drop into a pit you cannot climb out of).

This is Catrap's ancestor and shares its skeleton -- gravity, ladders, boulders,
diggable earth, no jump -- but four of its rules differ in ways that change the
whole search, so it is NOT a re-parameterisation of
`solvers/generate_catrap_training.py`:

  * **Gold falls.** ``down [ STATIONARY Gold | no BlockingAll ]``. The objective
    itself is mobile: dig out from under a nugget and it drops a row. Catrap's
    monsters never move, so its heuristic could key on a fixed cell; here a
    nugget's target cell is "somewhere in its column" (see `Bound`).
  * **A push does not carry the player.** ``LEFT [ > Player | Boulder | no
    BlockingAll ] -> [ PlayerLeft | > Boulder | ]`` -- the RHS drops the player's
    force, so the boulder slides one cell and the player stays put. Shoving a
    boulder n cells therefore costs 2n-1 presses (push, step in, push, ...), not
    n.
  * **Collecting is walking into it.** ``late [ Player Gold ] -> [ Player ]``
    with Gold on the Ladder/Dirt collision layer, so the player walks straight
    into the nugget's cell (and ``late [ Player Dirt ] -> [ Player ]`` digs the
    same way). Both are cancelled vertically by ``UP/DOWN [ > Player |
    BlockingObj ]``, so gold and earth are only ever taken SIDEWAYS.
  * **A ladder cell blocks everything above it** (Ladder is in `BlockingAll`),
    so the top of a ladder is a platform you can stand on in mid-air, and a
    nugget resting on a ladder can never be dropped further.

Height is the scarce resource: ``UP [ > Player no Ladder ] -> [ Player ]`` eats
every up press that does not start on a ladder, and there is no jump.

Expert solver
-------------
A MACRO A* over a re-implemented, tick-accurate model of those rules
(`Sim.press`), not over the interpreter. **47 of the 50 levels solve**, in 14 to
97 presses (3013 presses over the whole game, 40 of them with a tie), and the
other **three are provably unwinnable as shipped** -- see below, so the expert
wins every level that can be won.

  * **Why a model.** The interpreter costs ~0.4 ms per keypress; the model ~20
    us. That is what makes the search reach past the tutorial levels, and it
    also buys an exact, compact state key (``bytes(dyn) + bytes(di)``) instead
    of a frozenset of engine cells.
  * **Why macros.** Almost every press only relocates the player. The successors
    here are ``walk to cell X, then press the one key that changes something``
    -- collect, dig, push, or a step that un-supports a load overhead -- with
    the walk found by BFS over the (directed! gravity is one-way) walk graph.
    A macro is applied by TELEPORTING the player to its cell (`apply_macro`),
    which is exact because every walk edge provably moves only the player.
  * **The ladder**: A* at w=1 (shortest plans), then w=3, then a breadth-first
    beam whose wandering is trimmed by macro deletion (`_shorten`). 46 levels
    are answered by w=1, which -- with an admissible `Bound` and the goal tested
    on POP -- means those plans are genuinely shortest and every "optimal" label
    on them is a real tie. Level 4 is the one that needs w=3.
  * **The model is never trusted.** Every plan is replayed through the real
    interpreter and kept only if it really wins (`PitmanExpert._verify`). A
    model bug can therefore cost a level, never correctness -- and ``--fuzz``
    compares model against interpreter press-by-press over random rollouts on
    all 50 levels, auditing the walk analysis at sampled cells as it goes: 0
    divergences.

**Levels 18, 20 and 49 cannot be won**, and it is `run_rules_on_level_start`
that breaks them: the header settles gravity before the first keypress, and each
of those three boards is laid out so that the settle drops the player into a
pocket it can never leave. Level 20 is the clearest -- its bottom row is empty,
so every nugget and boulder in the map falls a row, the two rows below the
player's ledge empty out completely, and the first step off that ledge is a
seven-row drop into a dead-end beside an unpushable boulder. Confirmed by an
exhaustive macro BFS over the real reachable space: 12951 / 1 / 5 states
respectively, no win in any of them. (Level 4 looks the same from the outside --
its beam closes with four nuggets left -- but it is merely deep: the same
exhaustive BFS reaches a win after 9.2 M states, and w=3 finds a 77-press
solution in under a minute. That is what `time_budget` is set by.)

`PSAStarSolver` keeps the levels the expert can win and skips the rest, so the
corpus is only ever real wins; ``--plans`` prints the current roster.

Augmentation
------------
Rotation only (``rotation_k`` in {0,1,2,3}, with the matching directional action
remap in `PuzzleScriptAdapter.perform_action`). Pitman is NOT in `_FLIP_GAMES`:
gravity is an axis-sensitive mechanic, so a vertical flip would present a world
whose stones fall upwards, and the player sprite is not mirror-closed (it faces
the way it last walked). No colour augmentation either -- ladder-versus-earth is
already a shape-only distinction (both render in palette 12, one a lattice and
one a solid block), and recolouring would put that at risk.

The plans are cached on disk (`PLAN_CACHE`): they are seed-independent, a cold
build costs ~20 minutes (most of it the three unwinnable levels and level 4) and
every shard of `parallelize_generator` would otherwise repeat it on every core.
A warm start is seconds. A cached plan is
re-verified on the interpreter before it is used, and carries a signature of the
level it was solved for, so editing a level can never silently ship a stale plan.

Usage (run from the repo root):
    python solvers/generate_pitman_mz_700_training.py --episodes 200 \
        --out data/training_multi_level/pitman_mz_700

    python solvers/generate_pitman_mz_700_training.py --plans     # level report
    python solvers/generate_pitman_mz_700_training.py --plans --retry
    python solvers/generate_pitman_mz_700_training.py --fuzz      # model vs engine
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

GAME_NAME = "Pitman_MZ-700"

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Disk cache of the per-level plan AND its optimal-action sets. The searches are
#: seed-independent (levels are fixed ASCII maps; only the PRESENTATION is
#: augmented per seed) but cost minutes, which every shard of
#: `parallelize_generator` would otherwise repeat on every core. Delete to rebuild.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "pitman_mz_700_plans.json"

DIRS = ("up", "down", "left", "right")

# ---------------------------------------------------------------------------
# The model
# ---------------------------------------------------------------------------
# Three planes per cell, mirroring the two collision layers the mechanic uses:
#   `st`  static   -- wall / ladder, never created or destroyed by any rule
#   `di`  dirt     -- only ever deleted (by a sideways dig)
#   `dyn` movable  -- boulder / gold / player
# Dirt is its own plane rather than a `dyn` code because a PLAYER and a DIRT
# block do share a cell for the rest of the tick in which the player digs it
# (the delete is a `late` rule), and because dirt sits on the Ladder/Gold layer
# while the player sits on the Wall/Boulder one.
#
# Gold lives in `dyn` even though its collision layer is the earth's: it is the
# only object on that layer that MOVES, and it can never share a cell with a
# boulder or a resting player (both are `BlockingAll`, which is exactly what
# gravity and the push rule test), so one movable code per cell is enough.

EMPTY, BOULDER, GOLD, PLAYER = range(4)

S_NONE, S_LADDER, S_WALL = 0, 1, 2

#: Objects the three gravity rules move (the player is handled separately: it is
#: the one thing a ladder cell exempts).
_FALLS = (BOULDER, GOLD)

#: Engine object name -> model code for the movable plane.
_DYN_OF = {
    "boulder": BOULDER, "gold": GOLD,
    "playerright": PLAYER, "playerleft": PLAYER, "playerladder": PLAYER,
}
_STATIC_OF = {"ladder": S_LADDER, "wall": S_WALL}


class Board:
    """Static geometry of one level, padded with a ring of wall.

    The ring is what lets every neighbour be a plain index offset: no bounds
    test and no row wrap-around. A PuzzleScript board has no implicit frame, so
    unlike Catrap's the padding is doing real work here -- it is what stops
    things falling off the bottom row and what blocks a walk off the left edge.
    Cell ``(r, c)`` of the engine grid is index ``(r + 1) * W + (c + 1)``."""

    def __init__(self, eng, game):
        self.H, self.W = eng.height + 2, eng.width + 2
        n = self.H * self.W
        self.st = bytearray(n)
        for i in range(self.W):                      # top / bottom ring
            self.st[i] = S_WALL
            self.st[n - 1 - i] = S_WALL
        for r in range(self.H):                      # left / right ring
            self.st[r * self.W] = S_WALL
            self.st[r * self.W + self.W - 1] = S_WALL
        name = game.obj_idx_to_name
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                for o in cell:
                    s = _STATIC_OF.get(name.get(o, ""))
                    if s is not None:
                        self.st[(r + 1) * self.W + c + 1] = s
        self.delta = {"up": -self.W, "down": self.W, "left": -1, "right": 1}
        self.dirs = tuple(self.delta[d] for d in DIRS)
        self.name = {v: k for k, v in self.delta.items()}

    def index(self, r: int, c: int) -> int:
        return (r + 1) * self.W + c + 1

    def rc(self, i: int) -> tuple[int, int]:
        return divmod(i, self.W)[0] - 1, i % self.W - 1


class Sim:
    """One mutable game state on a `Board`, with the interpreter's turn loop.

    `press` is the unit of play: it runs ticks until the `again` cascade settles,
    exactly as `PSEngine.step` does, so one call is one keypress and the state it
    leaves is a state the player can actually be handed."""

    __slots__ = ("b", "dyn", "di", "occ", "pp", "broken")

    def __init__(self, board: Board, dyn: bytearray, di: bytearray, pp: int):
        self.b = board
        self.dyn = dyn
        self.di = di
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
        dyn, di = bytearray(n), bytearray(n)
        pp = -1
        name = game.obj_idx_to_name
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i = board.index(r, c)
                for o in cell:
                    nm = name.get(o, "")
                    if nm == "dirt":
                        di[i] = 1
                    d = _DYN_OF.get(nm)
                    if d is not None:
                        dyn[i] = d
                        if d == PLAYER:
                            pp = i
        return Sim(board, dyn, di, pp)

    def copy(self) -> "Sim":
        s = Sim.__new__(Sim)
        s.b, s.dyn, s.di = self.b, bytearray(self.dyn), bytearray(self.di)
        s.occ, s.pp, s.broken = set(self.occ), self.pp, self.broken
        return s

    def key(self) -> bytes:
        return bytes(self.dyn) + bytes(self.di)

    def won(self) -> bool:
        dyn = self.dyn
        return not any(dyn[i] == GOLD for i in self.occ)

    def golds(self) -> list[int]:
        dyn = self.dyn
        return [i for i in self.occ if dyn[i] == GOLD]

    # -- cell predicates ------------------------------------------------------
    def blockall(self, i: int) -> bool:
        """The game's `BlockingAll` group (Ladder, Gold, Dirt, Wall, Boulder,
        Player) -- what stops a fall and what a push needs the far cell to be
        free of. Note the LADDER: a ladder cell holds up whatever is above it,
        which is why the top of a ladder is a platform."""
        return bool(self.b.st[i] or self.di[i] or self.dyn[i])

    def blockobj(self, i: int) -> bool:
        """The game's `BlockingObj` group: `BlockingAll` minus Ladder. It is what
        the two vertical cancel rules test, so it is exactly the reason gold and
        earth can only be taken sideways."""
        return bool(self.b.st[i] == S_WALL or self.di[i] or self.dyn[i])

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
        if v == PLAYER and self.dyn[j] == GOLD:
            # ``late [ Player Gold ] -> [ Player ]``. The delete really happens
            # at the end of the tick, but nothing between the move and the late
            # phase can observe the nugget: the only rules that could are the
            # gravity ones, and they ran before the move.
            self.dyn[j] = EMPTY
            self.occ.discard(j)
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
            was_dyn, was_di = bytes(self.dyn), bytes(self.di)
            again = self._tick(indir)
            indir = None
            if self.dyn == was_dyn and self.di == was_di:
                break                       # no change -> the loop cannot repeat
            if not again or self.broken or self.won():
                break

    def _tick(self, indir) -> bool:
        """One rule pass + force resolution + the late deletes, in file order.

        Returns whether a rule asking for `again` fired; the caller ANDs that
        with "the grid actually changed", which is what `PSEngine.step` does.

        The player's FACING (PlayerLeft / PlayerRight / PlayerLadder) is not
        modelled. No rule reads it -- every rule names the `Player` or-group --
        so two states differing only in facing have identical futures, and
        merging them is exact for the dynamics. It costs one thing: a press
        whose ONLY effect is to turn the player around reads as a no-op here and
        is dropped from the search, which is right, because it is one."""
        b = self.b
        st, dyn, di = b.st, self.dyn, self.di
        W = b.W
        forces: dict[int, int] = {}
        again = False

        p = self.pp
        if indir is not None and p >= 0:
            d0 = b.delta.get(indir)
            if d0 is not None:
                forces[p] = d0

        # -- LEFT/RIGHT [ > Player | Boulder | no BlockingAll ]
        #        -> [ PlayerLeft | > Boulder | ]
        # The player's force is DROPPED by the RHS: a push shoves the boulder and
        # leaves the player standing where it was.
        d = forces.get(p)
        if d in (-1, 1):
            q = p + d
            if dyn[q] == BOULDER and not self.blockall(q + d):
                del forces[p]
                forces[q] = d

        # -- UP [ > Player | BlockingObj ] -> cancel
        if forces.get(p) == -W and self.blockobj(p - W):
            del forces[p]
        # -- UP [ > Player no Ladder ] -> cancel   (there is no jump)
        if forces.get(p) == -W and st[p] != S_LADDER:
            del forces[p]
        # -- DOWN [ > Player | BlockingObj ] -> cancel  (no digging downwards)
        if forces.get(p) == W and self.blockobj(p + W):
            del forces[p]

        # -- [ MOVING Player ] -> again  /  [ MOVING Boulder ] -> again
        # Every force that exists at this point is one of those two.
        if forces:
            again = True

        # -- DOWN [ STATIONARY Player no Ladder | no BlockingAll ] -> fall
        if p >= 0 and p not in forces and st[p] != S_LADDER \
                and not self.blockall(p + W):
            forces[p] = W
            again = True
        # -- DOWN [ STATIONARY Boulder | no BlockingAll ] -> fall
        # -- DOWN [ STATIONARY Gold   | no BlockingAll ] -> fall
        for i in tuple(self.occ):
            if dyn[i] in _FALLS and i not in forces and not self.blockall(i + W):
                forces[i] = W
                again = True

        # -- force resolution. Each object is tested against ITS OWN collision
        # layer: the player and the boulders share (Wall, Boulder, Player), the
        # gold shares (Ladder, Gold, Dirt) -- which is precisely why walking
        # sideways into a nugget or a dirt block is a legal move.
        while forces:
            stuck = True
            for i, dd in tuple(forces.items()):
                j = i + dd
                if dyn[i] == GOLD:
                    blocked = bool(st[j] != S_NONE or di[j] or dyn[j] == GOLD)
                else:
                    blocked = bool(st[j] == S_WALL
                                   or dyn[j] == BOULDER or dyn[j] == PLAYER)
                if not blocked:
                    del forces[i]
                    self._move(i, j)
                    stuck = False
            if stuck:
                break

        # -- late [ Player Dirt ] -> [ Player ]
        # (the gold twin is applied inside `_move`; see there for why)
        p = self.pp
        if p >= 0 and di[p]:
            di[p] = 0
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
    st, dyn, di = b.st, sim.dyn, sim.di
    W = b.W
    # Leaving would drop whatever is on our head -- but ONLY from a bare cell. A
    # LADDER is `BlockingAll` in its own right, so a boulder parked on top of a
    # climbing player is held by the ladder and does not move when the player
    # steps off. Reading that as a load anyway would not be wrong (the press
    # becomes a macro, and macros are simulated), it would just cost the walk
    # graph most of a ladder-heavy board.
    if st[p] == S_NONE and dyn[p - W] in _FALLS:
        return -1
    if d == -W:                                    # up: only ever from a ladder
        if st[p] != S_LADDER:
            return -1
        q = p - W
        if sim.blockobj(q):
            return -1
    elif d == W:                                   # down: step off / climb down
        q = p + W
        if sim.blockobj(q):
            return -1
    else:                                          # sideways
        q = p + d
        if st[q] == S_WALL or dyn[q] != EMPTY or di[q]:
            return -1                # a wall, a push, a collect or a dig
    while st[q] != S_LADDER:                       # then fall until supported
        j = q + W
        if st[j] or di[j] or dyn[j]:
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
    if s.pp >= 0:
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


def _moves_player(sim: Sim, p: int, d: int) -> bool:
    """Would pressing ``d`` at ``p`` move the player at all, ignoring any load
    overhead? Only used to tell a real un-support macro from a plain no-op."""
    st, W = sim.b.st, sim.b.W
    if d == -W:
        return st[p] == S_LADDER and not sim.blockobj(p - W)
    if d == W:
        return not sim.blockobj(p + W)
    q = p + d
    return not (st[q] == S_WALL or sim.dyn[q] == BOULDER)


def _candidates(sim: Sim, p: int) -> list[int]:
    """The presses at ``p`` that could change something other than the player:
    a collect / dig / push sideways, and -- when a boulder or a nugget is
    resting on the player's head -- any press that steps out from under it."""
    b = sim.b
    dyn, di = sim.dyn, sim.di
    W = b.W
    out = []
    load = b.st[p] == S_NONE and dyn[p - W] in _FALLS
    for d in (-1, 1):
        q = p + d
        if di[q] or dyn[q] == GOLD:                       # dig / collect
            out.append(d)
        elif dyn[q] == BOULDER:
            if not sim.blockall(q + d):                   # push
                out.append(d)
        elif load and _moves_player(sim, p, d):
            out.append(d)
    if load:
        for d in (-W, W):
            if _moves_player(sim, p, d):
                out.append(d)
    return out


class Macro:
    """One decision: the walk that sets it up, and the press that does it.
    ``press`` is a grid offset."""

    __slots__ = ("walk", "press", "target")

    def __init__(self, walk, press, target):
        self.walk = walk
        self.press = press
        self.target = target


def expand(sim: Sim) -> list[Macro]:
    """Every macro available from ``sim``: for each cell the player can walk to,
    each press there that changes the board."""
    base = stripped(sim)
    parent = _reachable(base)
    out = []
    for cell in parent:
        walk = None
        for d in _candidates(base, cell):
            if walk is None:
                walk = _path_to(parent, cell)
            out.append(Macro(walk, d, cell))
    return out


def apply_macro(sim: Sim, macro: Macro, board: Board) -> "Sim | None":
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
# The heuristic
# ---------------------------------------------------------------------------

class Bound:
    """Admissible press-count bounds for one level.

    The relaxation keeps the two constraints that actually shape a Pitman board
    and drops every other one:

      * **walls still block**, and **up is only possible from a ladder cell** --
        there is no jump in this game, so height is the scarce resource and a
        heuristic that ignores it is worthless on the tower levels;
      * boulders, earth and the other nuggets are all walked through for free,
        and **falling costs nothing** -- one press can drop the player any
        distance, so a down edge has to be a 0-edge for the bound to hold.

    Because what is left is STATIC, the distance field from a cell is computed
    once and cached.

    **The nuggets move**, which is the one thing Catrap's version of this class
    did not have to model. Gold only ever falls, and only within its own column,
    and it can never pass a wall or a ladder (both are `BlockingAll` and neither
    is destructible) -- so ``cells(g)``, the column from ``g`` down to the first
    wall or ladder, is a static superset of everywhere that nugget can ever be.
    Every distance to a nugget is the minimum over that set, which keeps the
    bound valid on the boards where the solution is "dig the floor out and let
    the gold come to you".

    **The last step into a nugget is SIDEWAYS** (``UP/DOWN [ > Player |
    BlockingObj ]`` cancels the vertical approach), so a collect costs
    ``reach a horizontal neighbour, then press`` -- see `approach`. Charging it
    that way rather than as "reach the nugget's own cell" is worth the extra
    lookup twice over: it is tighter by the +1 on every edge, and it is the
    game's DEADLOCK TEST for free. Falling is a 0-edge, so a relaxed field walks
    down into a one-wide shaft and reports a nugget at the bottom of it as
    cheap; requiring a non-wall cell beside the nugget instead reports the whole
    column as unreachable, `h` returns -1, and the subtree is dropped rather
    than explored."""

    def __init__(self, board: Board):
        self.b = board
        self._cells: dict[int, tuple[int, ...]] = {}
        self._approach: dict[int, tuple[int, ...]] = {}
        self._fields: dict[int, dict[int, int]] = {}
        self._gfields: dict[int, dict[int, int]] = {}
        self._pair: dict[tuple[int, int], int] = {}
        self._pcost: dict[tuple[int, int], int] = {}

    # -- where a nugget can get to ------------------------------------------
    def cells(self, g: int) -> tuple[int, ...]:
        got = self._cells.get(g)
        if got is not None:
            return got
        st, W = self.b.st, self.b.W
        out = [g]
        j = g + W
        while st[j] == S_NONE:
            out.append(j)
            j += W
        got = self._cells[g] = tuple(out)
        return got

    def approach(self, g: int) -> tuple[int, ...]:
        """Every cell the player could be standing in when it presses into this
        nugget: the non-wall horizontal neighbours of everywhere it can fall to.
        Empty means the nugget is walled in on both sides for good."""
        got = self._approach.get(g)
        if got is None:
            st = self.b.st
            got = self._approach[g] = tuple(
                t + d for t in self.cells(g) for d in (-1, 1)
                if st[t + d] != S_WALL)
        return got

    # -- 0-1 BFS distance fields --------------------------------------------
    def _bfs(self, sources) -> dict[int, int]:
        """0-1 BFS cost-to-reach from ``sources``, lazily relaxed: a cell can be
        re-reached by a 0-edge after a 1-edge found it first, and treating the
        first discovery as final would over-estimate -- which makes the whole
        heuristic inadmissible rather than merely loose."""
        st, W = self.b.st, self.b.W
        dist: dict[int, int] = {}
        dq = deque((s, 0) for s in sources)
        while dq:
            i, d0 = dq.popleft()
            if dist.get(i, 1 << 30) <= d0:
                continue
            dist[i] = d0
            j = i + W
            if st[j] != S_WALL:                      # falling is free
                dq.appendleft((j, d0))
            for dd in (-1, 1):
                j = i + dd
                if st[j] != S_WALL:
                    dq.append((j, d0 + 1))
            j = i - W
            if st[j] != S_WALL and st[i] == S_LADDER:
                dq.append((j, d0 + 1))
        return dist

    def field(self, cell: int) -> dict[int, int]:
        got = self._fields.get(cell)
        if got is None:
            got = self._fields[cell] = self._bfs((cell,))
        return got

    def gold_field(self, g: int) -> dict[int, int]:
        """The field from a nugget's whole reachable column: after collecting it
        the player is standing wherever it happened to be."""
        got = self._gfields.get(g)
        if got is None:
            got = self._gfields[g] = self._bfs(self.cells(g))
        return got

    _BIG = 1 << 30

    def pcost(self, p: int, g: int) -> int:
        """Presses needed, at least, to collect the nugget at ``g`` starting from
        cell ``p``: walk beside it, then press into it."""
        k = (p, g)
        got = self._pcost.get(k)
        if got is None:
            f = self.field(p)
            got = min((f[t] + 1 for t in self.approach(g) if t in f),
                      default=self._BIG)
            self._pcost[k] = got
        return got

    def pair(self, a: int, b: int) -> int:
        """`pcost` from a nugget's own cells -- where the player is left standing
        once it has collected that one."""
        k = (a, b)
        got = self._pair.get(k)
        if got is None:
            f = self.gold_field(a)
            got = min((f[t] + 1 for t in self.approach(b) if t in f),
                      default=self._BIG)
            self._pair[k] = got
        return got

    #: Above this many nuggets the MST is replaced by the cheap bound below. The
    #: MST is O(k^2) distance lookups per NODE, and the boards that carry 20+
    #: nuggets are wide-open galleries where it barely beats the cheap bound
    #: anyway (every nugget is a step from the next).
    mst_limit: int = 14

    def h(self, sim: Sim) -> int:
        """Lower bound on the presses left, or -1 when a nugget is walled off for
        good.

        The bound is the minimum spanning tree over {player} U {nuggets} with
        edge weights that lower-bound travel in either direction: any winning
        continuation walks a connected route through all of those cells, and a
        connected subgraph costs at least its MST. Every edge already includes
        the collecting press, so the tree is never cheaper than one press per
        nugget."""
        gold = sim.golds()
        k = len(gold)
        if not k:
            return 0
        big = self._BIG
        if k > self.mst_limit:
            # Cheap fallback: reach the first nugget, then at least one press
            # each for the rest.
            first = min(self.pcost(sim.pp, g) for g in gold)
            if first >= big:
                return -1
            return first + k - 1
        nodes = [sim.pp] + gold
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
                return -1                      # some nugget cannot be reached
            used[u] = True
            total += bu
            for v in range(1, k + 1):
                if not used[v]:
                    if u:
                        w = min(self.pair(nodes[u], gold[v - 1]),
                                self.pair(gold[v - 1], nodes[u]))
                    else:
                        w = self.pcost(sim.pp, gold[v - 1])
                    if w < d[v]:
                        d[v] = w
        return total


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class Plan(list):
    """The press sequence, carrying the optimal SET for each of its steps."""

    optsets: list

    def __init__(self, presses, optsets):
        super().__init__(presses)
        self.optsets = optsets


class PitmanExpert(PSExpert):
    """Macro A* over `Sim`, with every plan re-verified on the interpreter.

    `PSExpert.plan`'s memo and the engine-blackbox `_astar` are both replaced:
    the search runs on the model, so it needs neither the engine snapshots nor
    the frozenset state key."""

    scope_by_level = True
    #: The game defines no ACTION rule at all, so branching on it would double
    #: the epsilon-detour cost for nothing.
    directions = list(DIRS)

    #: Wall-clock ceiling for one level's search at each weight. 45 of the 46
    #: solved levels finish inside 5 s of it; the number is set by level 4,
    #: whose w=3 search needs 53 s (it was the last level the old 30 s ceiling
    #: threw away, and it is winnable -- an exhaustive macro BFS reaches a win
    #: after 9.2 M states).
    time_budget: float = 90.0
    #: Weights tried in order. w=1 gives shortest plans; the ladder only exists
    #: for the levels a shortest-plan search cannot finish in `time_budget`.
    weights: tuple[int, ...] = (1, 3)
    #: Last resort after the weight ladder (see `_beam`). Wider and deeper than
    #: A* can afford, and correspondingly less optimal.
    beam_budget: float = 150.0
    beam_width: int = 3000
    beam_depth: int = 70

    def setup(self) -> None:
        self._boards: dict[int, Board] = {}
        self._bounds: dict[int, Bound] = {}
        self._plans: dict[int, "Plan | None"] = {}
        self._disk = self._load_disk()
        #: A private adapter to replay candidate plans on. Verification has to
        #: reload a level and step it to the end, which would leave the RECORDING
        #: adapter mid-trajectory (`record_level` reads `_current_frame` right
        #: after it plans); a second interpreter keeps the check side-effect free.
        self._vgame: "PuzzleScriptAdapter | None" = None

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

    def bound(self, level: int) -> Bound:
        bd = self._bounds.get(level)
        if bd is None:
            bd = self._bounds[level] = Bound(self.board(level))
        return bd

    def sim(self, level: int) -> Sim:
        """The level's START state. Read off the private adapter, not the caller's
        engine, so planning never depends on where the recording game happens to
        be standing."""
        g = self._vg()
        g.set_level(level)
        return Sim.from_engine(self.board(level), g._engine, g._game)

    # -- search ---------------------------------------------------------------
    def _astar_model(self, start: Sim, board: Board, bound: Bound, weight: int,
                     deadline: float) -> "list[Macro] | None":
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
                    continue          # a nugget is walled off: a dead board
                best_g[k] = ng
                counter += 1
                heapq.heappush(pq, (ng + weight * hh, ng, counter, nxt,
                                    (chain, macro)))
            if nodes >= self.node_cap or time.time() > deadline:
                return None
        return None

    def _beam(self, start: Sim, board: Board, bound: Bound,
              deadline: float) -> "list[Macro] | None":
        """A width-capped breadth-first beam over the same macros, for the levels
        A* cannot finish.

        What A* runs out of here is not the heuristic's quality but its shape:
        the bound is a travel estimate, and the middle of a long solution is a
        dozen decisions that dig a shaft and park a boulder without the player
        getting one step closer to a nugget. Over that stretch the estimate is
        FLAT, weighting it changes nothing, and A* degenerates to uniform-cost
        search at a depth where that is hopeless. A beam spends the same budget
        on breadth at every depth and uses the bound only to break ties, which
        is all a flat estimate is good for. The trade is plan length: beam plans
        win, and are engine-verified, but they wander."""
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
                macros: "list[Macro]") -> "list[Macro] | None":
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
                 macros: "list[Macro]") -> "list[Macro]":
        """Drop the decisions a beam plan did not need.

        A beam wins by breadth, so its route wanders: it digs a tunnel it never
        walks and shoves a boulder it never needed moved. Deleting one macro and
        re-deriving the rest is a complete test of whether it mattered -- the
        replay either still reaches the win or it does not. Two greedy passes,
        because deleting one detour often makes the next one deletable.

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

    def _search_level(self, level: int) -> "Plan | None":
        board = self.board(level)
        start = self.sim(level)
        bound = self.bound(level)
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
    def _label(self, start: Sim, board: Board, macros: "list[Macro]") -> Plan:
        """Flatten the macros into presses, tagging each with the full set of
        presses that are equally good there.

        Inside a macro's walk every shortest route to the macro's cell reaches
        the SAME state for the SAME cost, so all of their first steps are equally
        optimal -- and on these open boards there are usually several. The press
        that ends the macro is labelled with itself: which nugget you go for, and
        which boulder you shove where, is the puzzle, so a sibling press is a
        different plan rather than a reordering of this one."""
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

    def plan(self, eng, level: "int | None" = None):
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

class PitmanSolver(PSAStarSolver):
    game_id = "puzzlescript_pitman_mz_700"
    game_name = GAME_NAME
    expert_cls = PitmanExpert

    #: Macro nodes per search; the wall-clock ceiling in `PitmanExpert` is the
    #: binding limit in practice.
    node_cap = 2_000_000
    #: Plans run to ~190 presses on the beam-solved levels.
    max_steps = 400

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The set of presses that are equally good at this step (see
        `PitmanExpert._label`). Falls back to the press about to be taken, so no
        expert step ever ships unlabelled.

        On a level solved at w=1 this is honest optimality: the macro
        decomposition is cost-preserving (every press sequence is some walk +
        effectful-press sequence, and the walks are shortest), `Bound.h` is
        admissible, and the search answers the goal on POP -- so the plan really
        is a shortest one and every tie is a real tie. On the levels that needed
        w=3 or the beam it is the expert's own best line rather than a proven
        optimum, which is the same bargain the other ps: beam generators make."""
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
    return bytes(s.dyn), bytes(s.di)


def _only_player_moved(a: Sim, b: Sim) -> bool:
    if a.di != b.di or a.pp == b.pp or b.pp < 0:
        return False
    dyn = bytearray(b.dyn)
    dyn[b.pp] = EMPTY
    dyn[a.pp] = PLAYER
    return dyn == a.dyn


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

    The model is a re-implementation of fifteen interleaved rules; this is the
    only thing that keeps it honest. Returns the number of diverging presses."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    bad = 0
    for level in (range(game.n_levels) if levels is None else levels):
        game.set_level(level)
        board = Board(eng, game._game)
        for t in range(trials):
            game.set_level(level)
            sim = Sim.from_engine(board, eng, game._game)
            rng = random.Random(f"pitman:fuzz:{level}:{t}")
            for step in range(steps):
                direction = rng.choice(DIRS)
                eng.step(direction)
                sim.press(direction)
                if sim.broken:
                    break
                if _engine_view(board, eng, game._game) != (bytes(sim.dyn),
                                                            bytes(sim.di)):
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


_GLYPH = {EMPTY: ".", BOULDER: "o", GOLD: "G", PLAYER: "P"}


def _text(board: Board, dyn, di) -> list[str]:
    rows = []
    for r in range(board.H):
        line = ""
        for c in range(board.W):
            i = r * board.W + c
            if dyn[i]:
                line += _GLYPH[dyn[i]]
            elif di[i]:
                line += "%"
            elif board.st[i] == S_WALL:
                line += "#"
            elif board.st[i] == S_LADDER:
                line += "L"
            else:
                line += " "
        rows.append(line)
    return rows


def _dump(board: Board, eng, game, sim: Sim) -> None:
    ed, ee = _engine_view(board, eng, game)
    left, right = _text(board, ed, ee), _text(board, sim.dyn, sim.di)
    print("      engine" + " " * max(0, board.W - 6) + " | model")
    for a, b in zip(left, right):
        print(f"      {a} | {b}{'   <<<' if a != b else ''}")


def _plan_report(levels=None, retry: bool = False) -> None:
    """Plan every level and print presses / wall clock -- the "how much of the
    game is solved" check and the search-tuning loop.

    ``retry`` drops the cached failures first. They are cached on purpose --
    otherwise every shard of `parallelize_generator` would re-burn the search
    budget on the unsolvable levels at startup -- so re-searching them is an
    explicit request, not the default."""
    solver = PitmanSolver()
    game = solver.make_game(0)
    expert = PitmanExpert(game, node_cap=solver.node_cap)
    if retry:
        for lvl in [k for k, v in expert._disk.items() if v["plan"] is None]:
            del expert._disk[lvl]
    solved = total = 0
    presses = 0
    for level in (range(game.n_levels) if levels is None else levels):
        game.set_level(level)
        t0 = time.time()
        plan = expert.plan(game._engine, level)
        dt = time.time() - t0
        total += 1
        if plan is None:
            print(f"  L{level:3d}: UNSOLVED                 {dt:6.1f}s",
                  flush=True)
            continue
        solved += 1
        presses += len(plan)
        ties = sum(len(s) - 1 for s in getattr(plan, "optsets", []))
        over = _bound_violation(expert, level, plan)
        print(f"  L{level:3d}: {len(plan):4d} presses  win=True  "
              f"{ties:4d} tie-presses  {dt:6.1f}s"
              f"{'' if over <= 0 else f'  BOUND OVER-ESTIMATES BY {over}'}",
              flush=True)
    print(f"  solved {solved}/{total}, {presses} presses")


def _bound_violation(expert: PitmanExpert, level: int, plan: list) -> int:
    """How far `Bound.h` over-estimates anywhere along ``plan`` (0 = never).

    A heuristic that exceeds the true remaining cost is not merely loose, it
    makes the w=1 search's plans non-optimal while still looking optimal -- and
    nothing else would notice. Replaying a known winning line and checking the
    estimate against what it actually cost is the cheap one-sided test for
    that."""
    sim = expert.sim(level)
    bound = expert.bound(level)
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
    sys.exit(PitmanSolver.main())
