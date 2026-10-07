"""Generate Phase-1 training data for the PuzzleScript game ps:zombie_rescue
("Zombie Rescue", Matt Slaybaugh).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a native model of the
eleven-rule turn, the search that plans over it, and the checks that pin the
model to the interpreter.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_zombie_rescue",
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
Fifteen levels of a haunted house. Walk the Fighter onto the exit; from level 8
on there is also a Child, and the win is ``All Human on Target`` with
``Human = Fighter or Child``, so both have to end on one of the two exit squares.
Jack-o-lanterns are pushable crates, torches are decorative walls, skeletons are
harmless bodies that shuffle toward you, and zombies kill.

ONE PRESS IS TWO TICKS, and that is the whole game
--------------------------------------------------
The rule list ends with ``[ > Player ] -> [ > Player PrevPlayer ] again``. So a
DIRECTIONAL press runs the rule pass twice:

  * **Tick A** -- the player has a force, so it is not ``stationary``. Every
    monster rule is written ``[ stationary Zombie1 | ... | stationary
    PlayerLure ]`` (or is gated on ``[ stationary Player ]``), and none of them
    can match while you are moving. Tick A is therefore YOURS ALONE: you move,
    the jack-o-lantern you shoved moves, and the child takes its step. The rule
    also drops a ``PrevPlayer`` marker on the cell you are leaving, and the
    child's follow rule drops a ``PrevChild`` on the cell it is leaving.
  * **Tick B** -- the ``again`` continuation, with no input. Now you ARE
    stationary, every monster rule matches, and the whole bestiary takes exactly
    one step. It sees FOUR lure cells, not two: ``PlayerLure = Player or
    PrevPlayer`` and ``ChildLure = Child or PrevChild``, so where you *were* is
    as attractive as where you are.

Pressing ACTION (or any press the player has no force for) skips tick A: rules
run once, the monsters move once, and there is no ``Prev`` marker at all. So
ACTION is a genuine WAIT that costs the monsters one lure cell of information --
which is why the plans use it (level 7's whole answer is two of them).

And the consequences that the .txt does not show:

  * **Sight is a LINE and nothing blocks it.** The ellipsis is content-agnostic
    in this interpreter, so a monster sharing a row or column with a lure steps
    one cell down that line THROUGH WALLS, through torches, through other
    monsters, at any range. What stops it is only the MOVE: the step itself is
    an ordinary collision-layer move, so a wall in the next cell simply blocks
    it. Hiding means not sharing a line, or standing behind something.
  * **Zombies want the CHILD first.** Rule 3 (child chase) is listed before
    rule 5 (player chase) and rule 5 only matches a ``stationary Zombie1``, so
    every zombie that can see a child lure commits to it and the player-chase
    rule never gets a look. On the eight levels with a child that inverts the
    usual dynamic: you are safe exactly while the kid is bait.
  * **An in-line adjacent zombie kills you whatever you press.** Rule 6 fires
    among the RULES, on the board as it stands before anything moves, so a
    zombie whose force points at your cell bites you even as you flee. Death
    puts a ``Dying`` object on the board, which is `PSEngine.check_game_over`'s
    trigger -- the episode is over, and only RESET comes back.
  * **Dying on the exit EVICTS it**, which is why death is never a win. ``Dying``
    shares a collision layer with ``Target``, so the marker that kills you also
    deletes the exit square under you -- and the adapter checks ``check_win``
    BEFORE ``check_game_over``, so without that eviction a tick that ate the
    child while a zombie bit you (with you standing on an exit) would report a
    WIN. Measured on a board built to reach exactly that: the target is gone
    from the cell and the frame reports ``win False over True``. `_Board` folds
    death into a dead successor on the strength of it.
  * **THE CHILD CAN BE EATEN, AND THAT STILL WINS.** Rule 4 replaces the child
    with a Corpse. ``Human`` is an or-group, so with no child left ``All Human
    on Target`` is just "the fighter is on an exit" -- and the engine reports
    WIN. This is not a corner case that the shipped levels avoid: it is the
    SHORTEST answer on levels 7, 12 and 13, and on those three the child cannot
    be saved at all (``--protect`` proves it by exhausting the child-alive
    space). So the plans here are the interpreter's shortest wins, kid or no
    kid; ``--protect`` reports the child-surviving optimum beside them.
  * **Skeletons chase the PLAYER only** and cannot hurt anybody. They are
    mobile walls, and on levels 4, 10 and 14 that is the puzzle.
  * **A refused press is a real turn.** There is no ``require_player_movement``,
    so walking into a wall still fires the ``again`` rule, still drops a
    ``PrevPlayer``, still drags the child one step and still ticks the monsters.
    It is ACTION plus a child step -- and the ``PrevPlayer`` it drops lands on
    the cell you are already standing on, so when the child is NOT in line the
    two are the same move exactly. Both halves of that are in the plans and in
    the tie sets: levels 8, 10 and 13 spend blocked presses that tie with
    ACTION, and level 14's second press is a blocked one that does NOT, because
    it pulls the child a cell that ACTION would leave where it was.
  * **The child follows the same line rule you are chased by** -- ``[ Child |
    ... | moving Player ] -> [ > Child PrevChild | ... | moving Player ]`` -- so
    it closes one cell whenever your press puts it in your row or column, and
    otherwise stands still. Two humans cannot swap cells (both moves are
    refused), and two chains claiming one cell both stop, which is how you
    park the child one square behind you on the exit column.
  * **A jack-o-lantern is a one-deep push**: no rule passes the force on to a
    second crate, so a pair in line cannot be moved at all, and a crate shoved
    into a dead end -- the exit corridor included -- is stuck there for good.
    That is one of the ways an exploration prefix strands a level, and one of
    the reasons recovery here has to be a RESET.

Expert solver
-------------
Not a search over the interpreter, which runs at ~770 steps/s here (measured:
the four ellipsis rules re-scan every monster against both of its rays on every
tick, and a directional press runs the whole pass twice). `_Board` is a native
model of all eleven rules over integer cell indices -- state ``(player, child,
child_alive, jacks, zombies, skeletons)``, with walls, torches and targets
static, and being eaten folded into a dead successor -- plus a faithful port of
`PSEngine._resolve_forces` (chains, deferral, multi-way conflicts) because every
tick of this game is several objects moving at once.

The plan is A* over that model with the wall-only BFS distance from the player
to the nearest exit as the heuristic. It is admissible because a press moves any
one piece at most one cell, and it is what the search is REQUIRED to be for the
plan lengths below to mean "shortest": the goal is returned when it is POPPED,
not when it is generated.

The obvious stronger bound, ``max`` over the humans of that distance, is NOT
admissible on this game -- the child can be eaten instead of walked home, and
then its distance never has to be paid. Using it silently returned a 16-press
"optimal" for level 12 where 15 exists. It IS admissible on a board with no
zombie (nothing can eat the child) and while ``--protect`` is on, and the model
takes it in exactly those two cases.

The per-step optimal SETS are MEASURED, not inferred: at a state whose exact
remaining distance is ``d``, a press is optimal iff it survives, changes the
board, and its successor still finishes in ``d - 1`` -- re-solved with the search
bounded at ``d - 1``, so a press that strands the level costs a bounded probe
rather than an exhaustive one (that bound is the difference between 1.7s and
27s over the fifteen levels). Without them, every free stretch of the walk to
the exit would train one arbitrary interleaving of the two axes as the single
right answer.

The levels
----------
All 15 shipped levels are solved and every plan is PROVED shortest; nothing was
authored or skipped. 188 presses in all (7, 10, 10, 9, 13, 16, 14, 3, 16, 17,
24, 8, 15, 18, 8), 32 of those steps have more than one equally-optimal press
(238 labelled presses in all), and 2 of the presses are waits. The longest plan is 24
against the adapter's 200-press per-level budget, so this game needs no
``games/`` step-limit wrapper.

With ``--protect`` (the child must survive), 12 of the 15 are still solvable and
cost 163 presses against the 152 the shipped plans spend on those same twelve;
levels 7, 12 and 13 have NO child-surviving win at all,
which is the measurement behind shipping the interpreter's definition of the win
rather than the author's.

The five checks, and what each one can and cannot see:

  * ``--plans`` builds every plan and replays it through the real interpreter,
    requiring the win on the LAST press and no earlier. That is the only check
    that can catch a plan which is valid but not shortest (a shorter one would
    have won earlier).
  * ``--fuzz`` compares `_Board` against the interpreter step for step over two
    populations, because neither alone is enough. Random play from every PREFIX
    of every level's own plan puts the model in the configurations the plans
    visit -- random play from a level start usually walks into a line and dies,
    so it would never reach the later boards. And dense random 5x5..9x9 boards
    with interior walls are what reach the rules the shipped levels barely use:
    a jack pushed against a monster, a child cornered by two zombies, a
    skeleton chain. The counters are printed for exactly that reason: a run
    reporting agreement while having killed no child has not checked rule 4.
  * ``--verify`` re-derives every plan length from a search that shares no code
    with the one that produced it (a plain breadth-first sweep bounded by the
    claimed length), asserts the heuristic admissible at every state of every
    plan, and re-derives every tie set the same bounded way.
  * ``--symmetry`` replays each plan, and a seeded random walk, on all 8 turned
    and mirrored copies of its own board THROUGH THE INTERPRETER, comparing the
    engine grid after every press. It is the only check that can see a rule-order
    chirality -- `_execute_rule` drives a rule's four directional copies in the
    fixed order up, down, left, right, which is a fact about the SCREEN and not
    about the board -- and it is what justifies putting this game in
    `PuzzleScriptAdapter._FLIP_GAMES` rather than only rotating it. It measures
    zero divergence, and there is a proof of why: a monster's lure set is always
    a pair of ORTHOGONALLY ADJACENT cells (Player/PrevPlayer, Child/PrevChild),
    and being in line with both of them from different directions would put the
    monster on one of the two. So no monster ever has a direction to break a tie
    between, and the scan order is never consulted.
  * ``--audit`` renders every cell composition, in every cell of every level, at
    the cell size that level is drawn at, and asserts pairwise distinctness.
    It passes as shipped: no ``games/ps:zombie_rescue/`` sprite patch is needed,
    which is why this generator builds the adapter directly.

Augmentation
------------
The engine state after reset is identical for every seed (the levels are fixed
ASCII maps), so the only per-(seed, level) variable is presentation: the frame
rotation and flips plus the matching directional action remap. The expert plan is
therefore seed-independent -- solved once per level, memoized to
``data/zombie_rescue_plans.json``, and replayed per seed with that seed's
remapped screen actions.

The game belongs in `_FLIP_GAMES` (16 presentations per level rather than 4). It
is the ESCAPE! argument plus the one thing that has to be measured: gravity-free,
screen-relative input, a win condition that names no direction (``All Human on
Target``), and no sprite anywhere encodes a facing. What is not free is that
several objects move at once and can contest a cell every single tick, so the
rule-expansion order Gobble Rush had to steer around could in principle decide an
outcome. ``--symmetry`` measures that it does not, and the adjacent-lure argument
above says why. With fifteen levels the flips are the difference between 60
presentations for the whole game and 240.

Recovery is the shared ``recovery_mode = "reset"`` arc, and this game is one of
the sharpest cases for it: an exploration prefix here ends with the player eaten,
the child eaten, or a jack-o-lantern shoved into the exit corridor, and none of
those can be undone by playing on. ONE RESET restores the level start the cached
plan was solved from.

Usage (run from the repo root):
    python solvers/generate_zombie_rescue_training.py \
        --episodes 200 --out data/training_multi_level/zombie_rescue

    python solvers/generate_zombie_rescue_training.py --plans
    python solvers/generate_zombie_rescue_training.py --protect
    python solvers/generate_zombie_rescue_training.py --verify
    python solvers/generate_zombie_rescue_training.py --fuzz
    python solvers/generate_zombie_rescue_training.py --symmetry
    python solvers/generate_zombie_rescue_training.py --audit
"""

from __future__ import annotations

import heapq
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan      # noqa: E402

GAME_NAME = "Zombie_Rescue"

#: Refuse to search past this many expansions. It is a MEMORY budget -- every
#: state is a tuple of ints and a few tuples -- and it exists so an edited level
#: fails loudly instead of swapping. The largest shipped level expands 1159.
CAP = 2_000_000

INF = 1 << 30

#: Engine direction names in the order the interpreter expands a rule's four
#: directional copies. `_Board` mirrors that order exactly, so this tuple is part
#: of the model rather than a display convention -- see `_Board._lure_dir`.
_ORDER: tuple[str, ...] = ("up", "down", "left", "right")
_DELTA: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_CODE: dict[str, int] = {name: i for i, name in enumerate(_ORDER)}

#: The five presses, ACTION last. ACTION is a genuine WAIT (no rule reads the
#: action key) and it is the only press that skips tick A -- see the docstring.
_PRESSES: tuple[str, ...] = _ORDER + ("action",)

#: Object ids inside `_Board`'s occupancy map. Everything the game can put in a
#: cell shares ONE collision layer (``Player, Wall, Child, Zombie1, Corpse,
#: Jack1, Jack2, Skull, Torch, Skeleton``), so blocking never has to ask which
#: layer -- one object per cell is the invariant the whole model rests on.
WALL, JACK, ZOMBIE, SKEL, CORPSE, PLAYER, CHILD = range(7)

#: The player was eaten. Terminal: `PSEngine.check_game_over` sees the ``Dying``
#: object, the adapter reports GAME_OVER, and only RESET comes back.
DEAD = "dead"


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """Zombie Rescue's whole rule set, over integer cell indices.

    A state is ``(player, child, child_alive, jacks, zombies, skeletons)``:
    two cell indices, a flag and three sorted tuples of cell indices. Walls,
    torches and targets live on the board rather than in the state -- no rule
    creates or destroys any of them. ``child`` is ``-1`` on the seven levels
    that have no child; when ``child_alive`` is False it is the CORPSE's cell,
    which is a permanent obstacle and never moves again.

    Being eaten is not in the state at all: it is terminal, so `step` reports it
    as a dead successor.
    """

    def __init__(self, h, w, walls, targets):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.targets = frozenset(targets)
        n = h * w
        self.nbr = [[-1] * 4 for _ in range(n)]
        self.ray: list[list[tuple[int, ...]]] = [[() for _ in range(4)]
                                                 for _ in range(n)]
        for r in range(h):
            for c in range(w):
                cell = r * w + c
                for d, (dr, dc) in enumerate(_DELTA):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        self.nbr[cell][d] = rr * w + cc
                    line = []
                    rr, cc = r + dr, c + dc
                    while 0 <= rr < h and 0 <= cc < w:
                        line.append(rr * w + cc)
                        rr += dr
                        cc += dc
                    self.ray[cell][d] = tuple(line)
        self.dist_target = self._bfs_targets()
        self.expanded = 0

    @staticmethod
    def state_of(eng, ids) -> tuple:
        """The interpreter's grid in `_Board`'s state encoding.

        Separate from `read` because the fuzz asks for it after EVERY press and
        rebuilding the board's ray tables that often is most of its runtime --
        while the board itself (walls, torches, targets, the distance field) is
        static for the whole level by construction."""
        _wall, _torch, _target, jack, zombie, skel, corpse, player, child = ids
        w = eng.width
        p = ch = -1
        alive = False
        jacks, zs, sks = [], [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i = r * w + c
                if cell & player:
                    p = i
                if cell & child:
                    ch, alive = i, True
                if cell & corpse:
                    ch, alive = i, False
                if cell & jack:
                    jacks.append(i)
                if cell & zombie:
                    zs.append(i)
                if cell & skel:
                    sks.append(i)
        return (p, ch, alive, tuple(sorted(jacks)),
                tuple(sorted(zs)), tuple(sorted(sks)))

    @classmethod
    def read(cls, eng, ids) -> "tuple[_Board, tuple]":
        """Build the model, and its state, from the interpreter's grid."""
        wall, torch, target = ids[0], ids[1], ids[2]
        w = eng.width
        walls, targets = set(), set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i = r * w + c
                if cell & wall or cell & torch:
                    walls.add(i)
                if cell & target:
                    targets.add(i)
        return cls(eng.height, w, walls, targets), cls.state_of(eng, ids)

    # -- static geometry ------------------------------------------------------
    def _bfs_targets(self) -> list[int]:
        """Wall-only BFS distance to the nearest target, per cell.

        Jack-o-lanterns, monsters and the other human are deliberately NOT
        obstacles here: the field has to LOWER-bound the real distance for the
        heuristic built on it to be admissible, and every one of them is either
        movable or lethal rather than permanent."""
        dist = [INF] * (self.h * self.w)
        dq = deque()
        for t in self.targets:
            if t not in self.walls:
                dist[t] = 0
                dq.append(t)
        while dq:
            cell = dq.popleft()
            for d in range(4):
                nxt = self.nbr[cell][d]
                if nxt >= 0 and nxt not in self.walls and dist[nxt] > dist[cell] + 1:
                    dist[nxt] = dist[cell] + 1
                    dq.append(nxt)
        return dist

    def win(self, state) -> bool:
        """``All Human on Target``, with ``Human = Fighter or Child``.

        A dead child is a Corpse, not a Human, so it stops being part of the
        condition -- which is the whole reason levels 7, 12 and 13 are winnable
        at all."""
        p, ch, alive, _j, _z, _s = state
        if p not in self.targets:
            return False
        return not alive or ch in self.targets

    def heuristic(self, state, protect: bool = False) -> int:
        """Admissible remaining-press estimate.

        The player must walk to an exit, and one press moves it at most one
        cell, so its wall-only distance is a lower bound. The CHILD's distance
        can only be added when the child is guaranteed to have to make the trip:
        with no zombie on the board nothing can eat it, and under ``protect``
        the search refuses to let it be eaten. Adding it unconditionally is what
        made an earlier version of this file report 16 presses for level 12,
        where 15 exists."""
        p, ch, alive, _j, zs, _s = state
        if self.win(state):
            return 0
        v = self.dist_target[p]
        if alive and ch >= 0 and (protect or not zs):
            v = max(v, self.dist_target[ch])
        return v

    # -- movement -------------------------------------------------------------
    def resolve(self, occ, forces) -> None:
        """Port of `PSEngine._resolve_forces` for this game's single layer.

        ``occ`` maps cell -> object id and is mutated in place; ``forces`` maps
        cell -> direction code (one object per cell, so the cell IS the key the
        engine spells ``(r, c, obj_idx)``).

        The three behaviours that matter here, all of them reachable every turn:
        a chain of objects pushed the same way moves together (you and the
        jack-o-lantern); an object blocked by a mover heading somewhere else is
        DEFERRED to the next pass and dropped if nothing moved (that is how the
        player and the child refuse to swap cells); and two independent chains
        claiming one cell BOTH stop (two zombies converging on the same square,
        or you walking into the cell the child is stepping into)."""
        nbr = self.nbr
        for _ in range(20):
            moved_any = False
            deferred_any = False
            forces_at_start = dict(forces)
            resolved: set[int] = set()
            movable: list[tuple[list[int], int]] = []
            for cell, d in list(forces.items()):
                if cell in resolved or cell not in occ:
                    continue
                chain = [cell]
                cur = cell
                chain_free = False
                blocker_is_mover = False
                while True:
                    nxt = nbr[cur][d]
                    if nxt < 0:
                        break
                    if nxt not in occ:
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
                    movable.append((chain, d))
                elif blocker_is_mover:
                    deferred_any = True
                else:
                    for ent in chain:
                        forces.pop(ent, None)
                        resolved.add(ent)
            non_head = {ent: i for i, (chain, _d) in enumerate(movable)
                        for ent in chain[1:]}
            subsumed = {i for i, (chain, _d) in enumerate(movable)
                        if chain[0] in non_head}
            claims: dict[int, int] = {}
            conflicting: set[int] = set()
            for i, (chain, d) in enumerate(movable):
                if i in subsumed:
                    continue
                for ent in chain:
                    tgt = nbr[ent][d]
                    other = claims.get(tgt)
                    if other is not None and other != i:
                        conflicting.add(i)
                        conflicting.add(other)
                    else:
                        claims[tgt] = i
            for i, (chain, d) in enumerate(movable):
                if i in subsumed:
                    continue
                if i in conflicting:
                    for ent in chain:
                        forces.pop(ent, None)
                        resolved.add(ent)
                    continue
                for ent in reversed(chain):
                    occ[nbr[ent][d]] = occ.pop(ent)
                    forces.pop(ent, None)
                    resolved.add(ent)
                    moved_any = True
            if not moved_any:
                if deferred_any and forces == forces_at_start:
                    forces.clear()
                break

    def occupancy(self, state) -> dict[int, int]:
        p, ch, alive, jacks, zs, sks = state
        occ = dict.fromkeys(self.walls, WALL)
        for cell in jacks:
            occ[cell] = JACK
        for cell in zs:
            occ[cell] = ZOMBIE
        for cell in sks:
            occ[cell] = SKEL
        if ch >= 0:
            occ[ch] = CHILD if alive else CORPSE
        occ[p] = PLAYER
        return occ

    def _pack(self, occ, alive) -> tuple:
        p = ch = -1
        jacks, zs, sks = [], [], []
        for cell, obj in occ.items():
            if obj == PLAYER:
                p = cell
            elif obj == CHILD or obj == CORPSE:
                ch = cell
            elif obj == JACK:
                jacks.append(cell)
            elif obj == ZOMBIE:
                zs.append(cell)
            elif obj == SKEL:
                sks.append(cell)
        return (p, ch, alive, tuple(sorted(jacks)),
                tuple(sorted(zs)), tuple(sorted(sks)))

    # -- the rules ------------------------------------------------------------
    def _lure_dir(self, src, lures) -> int:
        """The direction ``[ stationary X | ... | Lure ]`` binds at ``src``.

        The ellipsis is content-agnostic in this interpreter, so a lure anywhere
        along the ray counts -- through walls, through other monsters, at any
        range. `_execute_rule` drives a rule's four directional copies in the
        fixed order ``up, down, left, right`` and a monster stops being
        ``stationary`` the moment one of them binds, so the FIRST direction with
        a lure wins.

        That order is never actually consulted on this game: every lure set is
        either one cell or a pair of ORTHOGONALLY ADJACENT cells (a piece and
        the ``Prev`` marker it just left behind), and a monster in line with two
        adjacent cells from two different directions would have to be standing
        on one of them. ``--symmetry`` measures the consequence -- the game is
        exactly rotation- and mirror-symmetric."""
        rays = self.ray[src]
        for d in range(4):
            for cell in rays[d]:
                if cell in lures:
                    return d
        return -1

    def _monster_tick(self, state, prev_player, prev_child):
        """Rules 3-7 plus movement: one tick with the player stationary.

        Returns a state, or `DEAD` when rule 6 bites the player. ``prev_player``
        / ``prev_child`` are the extra lure cells the ``again`` continuation
        sees (``-1`` on an ACTION press, which never creates them)."""
        p, ch, alive, jacks, zs, sks = state
        nbr = self.nbr
        zf: dict[int, int] = {}

        # R3  [ stationary Player ][ stationary Zombie1 | ... | ChildLure ]
        #     The player is stationary in every tick this runs in, so bracket
        #     one always matches; ChildLure is the child and where it just was.
        lures = set()
        if alive and ch >= 0:
            lures.add(ch)
        if prev_child >= 0:
            lures.add(prev_child)
        if lures:
            for z in zs:
                d = self._lure_dir(z, lures)
                if d >= 0:
                    zf[z] = d

        # R4  [ > Zombie1 | Child ] -> [ > Zombie | Corpse ]
        #     Before anything moves, so a zombie one cell short of the child on
        #     the tick it commits still gets it. The corpse then blocks the very
        #     move that made it, which is why an eaten child leaves the zombie
        #     standing where it was.
        if alive and ch >= 0:
            for z, d in zf.items():
                if nbr[z][d] == ch:
                    alive = False
                    break

        # R5  [ stationary Zombie1 | ... | stationary PlayerLure ]
        #     Only the zombies R3 left alone: chasing the child outranks
        #     chasing you, because rule 3 is listed first.
        plures = {p}
        if prev_player >= 0:
            plures.add(prev_player)
        for z in zs:
            if z in zf:
                continue
            d = self._lure_dir(z, plures)
            if d >= 0:
                zf[z] = d

        # R6  [ > Zombie1 | Player ] -> [ Zombie1 | Dying Player ]
        for z, d in zf.items():
            if nbr[z][d] == p:
                return DEAD

        # R7  [ stationary Skeleton | ... | stationary PlayerLure ]
        #     Skeletons never look at the child and can never hurt anyone.
        sf = {}
        for s in sks:
            d = self._lure_dir(s, plures)
            if d >= 0:
                sf[s] = d

        occ = self.occupancy((p, ch, alive, jacks, zs, sks))
        forces = dict(zf)
        forces.update(sf)
        self.resolve(occ, forces)
        return self._pack(occ, alive)

    def step(self, state, press):
        """One press. Returns ``(state or DEAD, won)``.

        See the module docstring: a DIRECTIONAL press is two ticks (yours, then
        the monsters'), an ACTION press is one (the monsters'). The win is
        checked between them because `PSEngine.step`'s ``again`` loop breaks the
        moment it holds -- so a press that lands both humans on the exits is not
        followed by a monster tick, and cannot be taken back by one."""
        p, ch, alive, jacks, zs, sks = state
        if press == "action" or press == 4:
            nxt = self._monster_tick(state, -1, -1)
            return (DEAD, False) if nxt is DEAD else (nxt, self.win(nxt))

        d = _CODE[press] if isinstance(press, str) else press

        # -- tick A: the player, its crate and the child. No monster can match
        #    a rule while the player has a force.
        occ = self.occupancy(state)
        forces = {p: d}
        ahead = self.nbr[p][d]
        if ahead >= 0 and occ.get(ahead) == JACK:           # R2, a one-deep push
            forces[ahead] = d
        prev_player = p                                     # R10, always fires
        prev_child = -1
        if alive and ch >= 0:                               # R11
            cd = self._lure_dir(ch, {p})
            if cd >= 0:
                forces[ch] = cd
                prev_child = ch
        self.resolve(occ, forces)
        mid = self._pack(occ, alive)
        if self.win(mid):
            return mid, True

        # -- tick B: the `again` continuation, where the bestiary moves
        nxt = self._monster_tick(mid, prev_player, prev_child)
        return (DEAD, False) if nxt is DEAD else (nxt, self.win(nxt))

    # -- search ---------------------------------------------------------------
    def astar(self, start, protect=False, bound=None, cap=CAP, want_path=True):
        """Shortest press count from ``start``, or ``(None/INF, expanded)``.

        The goal is returned when it is POPPED, not when it is generated: the
        heuristic is admissible but NOT consistent (a press that gets the child
        eaten can drop it by more than one), so returning on generation would
        return the first win found rather than the shortest.

        ``bound`` caps ``g + h``: with it, "is there a win within k presses"
        costs a bounded probe instead of exhausting the space, which is what
        makes the optimal-set measurement affordable on the levels where most
        presses strand the board.
        """
        self.expanded = 0
        h0 = self.heuristic(start, protect)
        if h0 >= INF or (bound is not None and h0 > bound):
            return (None if want_path else INF), 0
        openh = [(h0, 0, start)]
        best = {start: 0}
        parent: dict[tuple, tuple | None] = {start: None}
        while openh:
            _f, g, st = heapq.heappop(openh)
            if best.get(st, INF) < g:
                continue
            if self.win(st):
                if not want_path:
                    return g, self.expanded
                path = []
                cur = st
                while parent[cur] is not None:
                    press, prev = parent[cur]
                    path.append(press)
                    cur = prev
                return list(reversed(path)), self.expanded
            self.expanded += 1
            if self.expanded > cap:
                raise RuntimeError(f"search exceeded {cap} expansions")
            for press in _PRESSES:
                nxt, _won = self.step(st, press)
                if nxt is DEAD:
                    continue
                if protect and start[2] and not nxt[2]:
                    continue
                ng = g + 1
                if ng >= best.get(nxt, INF):
                    continue
                hv = self.heuristic(nxt, protect)
                if hv >= INF or (bound is not None and ng + hv > bound):
                    continue
                best[nxt] = ng
                parent[nxt] = (press, st)
                heapq.heappush(openh, (ng + hv, ng, nxt))
        return (None if want_path else INF), self.expanded

    def optsets(self, start, plan) -> list[list[str]]:
        """The MEASURED optimal set for every step of a shortest ``plan``.

        ``plan`` is shortest, so the exact remaining distance at step ``i`` is
        ``len(plan) - i`` and a press is optimal iff its successor finishes in
        one less. Two sound prunes keep that cheap: a press that leaves the
        state untouched wastes a move to reach where it started, and a press
        whose admissible estimate already exceeds what is left cannot make it.
        The re-solve is bounded by the same figure, so an unwinnable successor
        costs a bounded probe rather than an exhaustive search."""
        out = []
        st = start
        for i, taken in enumerate(plan):
            need = len(plan) - i
            best = []
            for press in _PRESSES:
                nxt, won = self.step(st, press)
                if nxt is DEAD or nxt == st:
                    continue
                if won:
                    if need == 1:
                        best.append(press)
                    continue
                if self.heuristic(nxt) + 1 > need:
                    continue
                d, _n = self.astar(nxt, bound=need - 1, want_path=False)
                if d + 1 == need:
                    best.append(press)
            assert taken in best, (i, taken, best)
            out.append(best)
            st = self.step(st, taken)[0]
        return out


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class ZombieRescueExpert(PSExpert):
    """Plans by building `_Board` off the engine grid and searching THAT; never
    steps the interpreter.

    `PSExpert` still owns everything around the search -- the in-memory memo,
    the disk cache with its staleness check, the level scoping and the
    snapshot/restore discipline -- so the only override is `_search`.
    `heuristic` is unreachable by construction: no A* runs over engine states
    here, only over `_Board`'s.
    """

    #: `_key` below is dynamic-objects-only, canonical only WITHIN a level.
    scope_by_level = True

    #: ACTION is a genuine WAIT, and so is a press into a wall -- but the wall
    #: press also drags the child, so neither is droppable. The plans spend both.
    directions = list(_PRESSES)

    #: The fifteen searches together take ~2s, which is small but paid by EVERY
    #: `parallelize_generator` shard at startup; the file makes them free.
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "zombie_rescue_plans.json")

    def setup(self) -> None:
        name = self.g.resolve_object_name
        self.wall_ids = set(name("wall"))
        self.torch_ids = set(name("torch"))
        self.target_ids = set(name("target"))
        self.jack_ids = set(name("jack1"))
        self.zombie_ids = set(name("zombie1"))
        self.skel_ids = set(name("skeleton"))
        self.corpse_ids = set(name("corpse"))
        self.child_ids = set(name("child"))
        self.player_ids = set(self.game._engine._player_indices)
        #: Everything a settled frame can differ by. Walls, torches and targets
        #: are left out deliberately -- no rule creates or destroys any of them,
        #: so they are static per level (hence ``scope_by_level``). ``dying``
        #: IS in: a board carrying it is terminal and must not share a key with
        #: the live board it came from.
        #: The object-id bundle `_Board` reads a grid with, in its own order.
        self.ids = (self.wall_ids, self.torch_ids, self.target_ids,
                    self.jack_ids, self.zombie_ids, self.skel_ids,
                    self.corpse_ids, self.player_ids, self.child_ids)
        self.dyn_ids = (self.player_ids | self.child_ids | self.corpse_ids
                        | self.jack_ids | self.zombie_ids | self.skel_ids
                        | set(name("dying")))

    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "ZombieRescueExpert searches the native model; the engine-state "
            "heuristic is unused")

    def board(self, eng) -> "tuple[_Board, tuple]":
        return _Board.read(eng, self.ids)

    def _search(self, eng) -> "Plan | None":
        board, start = self.board(eng)
        presses, _n = board.astar(start)
        if presses is None:
            return None
        return Plan(presses, board.optsets(start, presses))


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _levels():
    solver = ZombieRescueSolver()
    game = solver.make_game(0)
    expert = ZombieRescueExpert.__new__(ZombieRescueExpert)
    expert.game, expert.g = game, game._game
    expert.bg_id = game._game.obj_name_to_idx.get("background")
    expert.node_cap, expert.weight = CAP, 1
    expert.cache, expert._disk = {}, {}
    expert.setup()
    return solver, game, expert


def _boards(game, expert) -> dict:
    """``{level: (board, start)}`` for every level, read off the interpreter."""
    out = {}
    for level in range(game.n_levels):
        game.set_level(level)
        out[level] = expert.board(game._engine)
    return out


def _certify(game, level, presses) -> bool:
    """Replay ``presses`` through the real interpreter: the win must land on the
    LAST one and on no earlier one, and the player must never be eaten. A plan
    that wins early is a plan that is not shortest."""
    game.set_level(level)
    eng = game._engine
    for i, press in enumerate(presses):
        eng.step(press)
        if eng.check_win():
            return i == len(presses) - 1
        if eng.check_game_over():
            return False
    return False


def _report(protect: bool = False) -> int:
    """Per-level census, plan length and tie coverage -- and CERTIFY each plan by
    replaying it through the interpreter."""
    _solver, game, expert = _levels()
    bad = 0
    total = labels = waits = ties = 0
    for level, (board, start) in _boards(game, expert).items():
        t = time.time()
        presses, expanded = board.astar(start, protect=protect)
        census = (f"{len(start[3])}J {len(start[4])}Z {len(start[5])}S "
                  f"{'+kid' if start[2] else '   '}")
        if presses is None:
            print(f"level {level:2d}: {census}, NO PLAN "
                  f"({expanded} states in {time.time() - t:.2f}s)")
            bad += not protect
            continue
        sets = [] if protect else board.optsets(start, presses)
        took = time.time() - t
        ok = _certify(game, level, presses)
        bad += not ok
        total += len(presses)
        waits += sum(1 for x in presses if x == "action")
        ties += sum(1 for x in sets if len(x) > 1)
        labels += sum(len(x) for x in sets)
        print(f"level {level:2d}: {census}, {len(presses):3d} presses, "
              f"{sum(1 for x in sets if len(x) > 1):2d} tie steps, "
              f"A* {expanded:6d} states, {took:6.2f}s, "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    tag = " with the child kept alive" if protect else ""
    print(f"total {total} presses over {game.n_levels} levels{tag}, "
          f"{labels} labelled presses over {ties} tie steps, "
          f"{waits} of them waits")
    return 0 if not bad else 1


def _engine_state(eng, expert):
    """The interpreter's grid in `_Board`'s state encoding, plus the terminal
    marker the model folds into a dead successor."""
    return _Board.state_of(eng, expert.ids), eng.check_game_over()


def _random_board(rng, expert) -> list:
    """A dense random level: a wall ring, interior walls and torches, one
    fighter, usually a child, and a scatter of zombies, skeletons, crates and
    exits.

    The shipped levels are corridors with one or two monsters; these are the
    only population that reaches a jack shoved against a monster, a child
    cornered by two zombies at once, or a chain of skeletons all pushed the same
    way -- the parts of `resolve` and of rules 3-7 that a shipped board barely
    touches."""
    bg = expert.bg_id
    h, w = rng.randint(5, 9), rng.randint(5, 9)
    wall = min(expert.wall_ids)
    grid = [[{bg} if 0 < r < h - 1 and 0 < c < w - 1 else {bg, wall}
             for c in range(w)] for r in range(h)]
    free = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    rng.shuffle(free)

    def place(obj_ids, n):
        for _ in range(n):
            if not free:
                return
            r, c = free.pop()
            grid[r][c].add(min(obj_ids))

    place(expert.wall_ids, rng.randint(0, (h * w) // 8))
    place(expert.torch_ids, rng.randint(0, 2))
    place(expert.player_ids, 1)
    if rng.random() < 0.85:
        place(expert.child_ids, 1)
    place(expert.zombie_ids, rng.randint(0, 3))
    place(expert.skel_ids, rng.randint(0, 2))
    place(expert.jack_ids, rng.randint(0, 3))
    # Targets sit on their own collision layer, so they may share a cell.
    for _ in range(rng.randint(1, 3)):
        r, c = rng.choice([(r, c) for r in range(1, h - 1)
                           for c in range(1, w - 1)])
        grid[r][c].add(min(expert.target_ids))
    return grid


def _walk(board, start, game, expert, rng, length, totals) -> int:
    """Random play from ``start``, model against interpreter, step for step.

    The interpreter is already sitting at ``start``. Returns the number of
    disagreements (and stops at the first)."""
    eng = game._engine
    state = start
    for _ in range(length):
        press = rng.choice(_PRESSES)
        before = state
        nxt, won = board.step(state, press)
        eng.step(press)
        e_win, e_over = eng.check_win(), eng.check_game_over()
        totals["steps"] += 1
        if nxt is DEAD:
            totals["deaths"] += 1
            # ``not e_win`` as well as ``e_over``: the adapter reports WIN
            # ahead of GAME_OVER, so a tick that killed the player while the
            # win condition came true would be a WIN the model calls a dead
            # successor. It cannot happen -- Dying evicts the Target under the
            # player -- and this is where that is checked rather than assumed.
            if not e_over or e_win:
                print(f"  MODEL SAYS EATEN, INTERPRETER SAYS "
                      f"win={e_win} over={e_over} ({press}) from {before}")
                return 1
            return 0
        e_state, _ = _engine_state(eng, expert)
        if e_state != nxt or bool(e_win) != bool(won) or e_over:
            print(f"  DISAGREE on {press} from {before}\n"
                  f"    model  {nxt} win={won}\n"
                  f"    engine {e_state} win={e_win} over={e_over}")
            return 1
        if before[2] and not nxt[2]:
            totals["child_eaten"] += 1
        if len(before[3]) and before[3] != nxt[3]:
            totals["pushes"] += 1
        if won:
            totals["wins"] += 1
            return 0
        state = nxt
    return 0


def _fuzz(prefix_walks: int = 12, boards: int = 900, length: int = 25) -> int:
    """`_Board` against the interpreter over two populations.

    Neither alone is enough. Random play from every PREFIX of every level's own
    plan puts the model in the configurations the plans visit -- random play from
    a level start walks into a line and dies, so it would never reach the later
    boards. And the dense random boards are what reach the rules the shipped
    levels barely use. The counters are printed for exactly that reason: a run
    reporting agreement while having eaten no child has not checked rule 4."""
    _solver, game, expert = _levels()
    rng = random.Random(20260823)
    totals = dict(steps=0, deaths=0, child_eaten=0, wins=0, pushes=0)
    bad = 0

    for level, (board, start) in _boards(game, expert).items():
        presses, _n = board.astar(start)
        for cut in range(len(presses)):
            for _ in range(prefix_walks):
                game.set_level(level)
                eng = game._engine
                state = start
                for press in presses[:cut]:
                    state = board.step(state, press)[0]
                    eng.step(press)
                bad += _walk(board, state, game, expert, rng, length, totals)
                if bad:
                    return 1
    print(f"plan-prefix population: {totals['steps']} transitions")

    mid = dict(totals)
    for _ in range(boards):
        grid = _random_board(rng, expert)
        game._engine.load_level(grid)
        board, start = expert.board(game._engine)
        bad += _walk(board, start, game, expert, rng, length, totals)
        if bad:
            return 1
    print(f"random-board population: {totals['steps'] - mid['steps']} transitions")
    print(f"total {totals['steps']} transitions, 0 disagreements: "
          f"{totals['deaths']} player deaths, {totals['child_eaten']} children "
          f"eaten, {totals['pushes']} crate pushes, {totals['wins']} wins")
    return 0


def _independent(board: _Board, state, limit: int) -> "int | None":
    """Presses from ``state`` to a win, or None if that is more than ``limit``.

    A plain forward breadth-first search to the FIRST winning state, pruned by
    ``depth + heuristic > limit`` (sound, because the heuristic is admissible --
    which `_verify` asserts before trusting it here). It is the second book in
    `_verify`'s double entry: it shares nothing with `_Board.astar` but `step`
    and `heuristic` -- no priority queue, no parent map, no bound arithmetic."""
    if board.win(state):
        return 0
    seen = {state}
    frontier = [state]
    for depth in range(1, limit + 1):
        nxt_frontier = []
        for cur in frontier:
            for press in _PRESSES:
                nxt, _won = board.step(cur, press)
                if nxt is DEAD or nxt == cur or nxt in seen:
                    continue
                if board.win(nxt):
                    return depth
                if depth + board.heuristic(nxt) > limit:
                    continue
                seen.add(nxt)
                nxt_frontier.append(nxt)
        frontier = nxt_frontier
    return None


def _verify(levels: "tuple[int, ...] | None" = None) -> int:
    """Double-entry check of every plan length AND every tie set.

    Three separate claims, checked three separate ways:

      * the LENGTH: `_independent` re-derives it with a breadth-first sweep that
        shares no code with the A* that produced it. Returning exactly ``d*``
        means no shorter win exists, since the sweep would have found one first.
      * the HEURISTIC is admissible, asserted rather than argued -- at every
        state of every plan, ``h(s) <= presses remaining``. Both of `optsets`'
        prunes and `_independent`'s own prune lean on it.
      * the TIE SETS are re-derived by `_independent`, so the measurement that
        labels the corpus is checked by a search that cannot share its bug.
    """
    _solver, game, expert = _levels()
    bad = 0
    for level, (board, start) in _boards(game, expert).items():
        if levels is not None and level not in levels:
            continue
        presses, _n = board.astar(start)
        sets = board.optsets(start, presses)
        t = time.time()
        notes = []
        star = len(presses)
        again = _independent(board, start, star)
        if again != star:
            notes.append(f"LENGTH A* {star} != breadth-first {again}")
        state = start
        for i, taken in enumerate(presses):
            need = star - i
            if board.heuristic(state) > need:
                notes.append(f"step {i}: HEURISTIC {board.heuristic(state)} > {need}")
            want = []
            for press in _PRESSES:
                nxt, _won = board.step(state, press)
                if nxt is DEAD or nxt == state:
                    continue
                if _independent(board, nxt, need - 1) == need - 1:
                    want.append(press)
            if want != sets[i]:
                notes.append(f"step {i}: {sets[i]} != {want}")
            state = board.step(state, taken)[0]
        bad += len(notes)
        print(f"level {level:2d}: d*={star:3d} re-derived, heuristic admissible "
              f"at {star} states, {sum(len(x) for x in sets):3d} labelled "
              f"presses re-derived in {time.time() - t:6.2f}s: "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _tdir(direction: str, k: int, mirror: bool) -> str:
    """``direction`` as seen after ``k`` quarter-turns and an optional mirror.

    The grid transform below sends ``(r, c) -> (c, h - 1 - r)`` per turn, so a
    displacement ``(dr, dc)`` becomes ``(dc, -dr)``. Getting this backwards
    makes the check pass on the 180-degree turn and fail on the quarter ones,
    which is exactly what it looked like the first time."""
    if direction == "action":
        return direction
    dr, dc = _DELTA[_CODE[direction]]
    for _ in range(k):
        dr, dc = dc, -dr
    if mirror:
        dc = -dc
    return _ORDER[_DELTA.index((dr, dc))]


def _tgrid(grid, k: int, mirror: bool):
    out = [list(row) for row in grid]
    for _ in range(k):
        h = len(out)
        out = [[out[r][c] for r in range(h - 1, -1, -1)]
               for c in range(len(out[0]))]
    return [row[::-1] for row in out] if mirror else out


def _symmetry(walks: int = 150, length: int = 30) -> int:
    """Replay every plan -- and a seeded random walk -- on all 8 turned and
    mirrored copies of its own board, THROUGH THE INTERPRETER.

    It is the only check that can see a rule-order chirality: `_execute_rule`
    drives a rule's four directional copies in the fixed order ``up, down,
    left, right``, which is a fact about the SCREEN and not about the board, so
    a game whose monsters ever have to break a tie between two directions
    resolves the same physical situation differently under a turn. This game
    never does (see `_Board._lure_dir`), and this is the measurement of that.
    """
    _solver, game, expert = _levels()
    g = game._game
    eng = game._engine
    rng = random.Random(20260823)
    bad = 0
    checked = 0

    def replay(level, sequence):
        nonlocal bad, checked
        eng.load_level(g.levels[level])
        base = []
        for press in sequence:
            eng.step(press)
            base.append([[frozenset(cell) for cell in row] for row in eng.grid])
        for k in range(4):
            for mirror in (False, True):
                if k == 0 and not mirror:
                    continue
                eng.load_level(_tgrid(g.levels[level], k, mirror))
                for i, press in enumerate(sequence):
                    eng.step(_tdir(press, k, mirror))
                    got = [[frozenset(cell) for cell in row] for row in eng.grid]
                    checked += 1
                    if got != _tgrid(base[i], k, mirror):
                        print(f"  DIVERGE level {level} k={k} mirror={mirror} "
                              f"step {i} press {press}")
                        bad += 1
                        return

    for level, (board, start) in _boards(game, expert).items():
        presses, _n = board.astar(start)
        replay(level, presses)
    print(f"every plan replayed at all 8 presentations: {checked} presses")

    mid = checked
    for _ in range(walks):
        level = rng.randrange(game.n_levels)
        replay(level, [rng.choice(_PRESSES) for _ in range(length)])
    print(f"{walks} random walks replayed at all 8 presentations: "
          f"{checked - mid} presses")
    print("symmetry clean" if not bad else f"SYMMETRY FAILED: {bad} divergences")
    return 0 if not bad else 1


def _compositions(expert) -> dict:
    """Every cell composition this game can render, as ``{name: object set}``.

    Everything in the main collision layer is exclusive, so the only stacks are
    with the Target (its own layer) and the Dying marker that ends an episode.
    ``PrevPlayer`` / ``PrevChild`` are omitted: their sprites are entirely
    transparent, and they only ever survive a step on a board that has already
    won."""
    bg = expert.bg_id
    one = lambda ids: min(ids)                                    # noqa: E731
    target = one(expert.target_ids)
    base = {
        "empty": set(),
        "wall": {one(expert.wall_ids)},
        "torch": {one(expert.torch_ids)},
        "jack": {one(expert.jack_ids)},
        "fighter": {one(expert.player_ids)},
        "zombie": {one(expert.zombie_ids)},
        "skeleton": {one(expert.skel_ids)},
        "corpse": {one(expert.corpse_ids)},
        "child": {one(expert.child_ids)},
    }
    out = {name: {bg} | objs for name, objs in base.items()}
    out["target"] = {bg, target}
    for name in ("fighter", "child", "corpse", "jack", "zombie", "skeleton"):
        out[f"target+{name}"] = {bg, target} | base[name]
    out["dying+fighter"] = {bg, one(expert.g.resolve_object_name("dying")),
                            one(expert.player_ids)}
    return out


def _audit() -> int:
    """Render every composition in EVERY cell of every level and require the
    resulting frames to be pairwise distinct.

    Per cell rather than once per level because the renderer scales the whole
    board up to fill 64x64 after drawing it at ``min(64//H, 64//W)`` pixels per
    cell, so the pixel grid a 5x5 sprite lands on differs from cell to cell --
    a pair that survives the middle of the board can still collapse at its
    edge. Whole frames are compared for the same reason: the board origin is
    not ``(64 - H*cell_px)//2``, so indexing a cell block is not safe."""
    _solver, game, expert = _levels()
    g = game._game
    eng = game._engine
    comps = _compositions(expert)
    names = list(comps)
    bad = 0
    for level, lv in enumerate(g.levels):
        h, w = len(lv), len(lv[0])
        dups = set()
        for r in range(h):
            for c in range(w):
                frames = {}
                for name, comp in comps.items():
                    grid = [[set(cell) for cell in row] for row in lv]
                    grid[r][c] = set(comp)
                    eng.load_level(grid)
                    frames[name] = _render_frame(eng, g).copy()
                for i in range(len(names)):
                    for j in range(i + 1, len(names)):
                        if np.array_equal(frames[names[i]], frames[names[j]]):
                            dups.add((names[i], names[j]))
        bad += len(dups)
        print(f"level {level:2d}: {h}x{w} at {min(64 // h, 64 // w)} px/cell, "
              f"{len(comps)} compositions x {h * w} cells: "
              f"{'all distinct' if not dups else 'COLLIDES ' + str(sorted(dups))}")
    print("audit clean" if not bad else f"AUDIT FAILED: {bad} colliding pairs")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

class ZombieRescueSolver(PSAStarSolver):
    game_id = "puzzlescript_zombie_rescue"
    game_name = GAME_NAME
    expert_cls = ZombieRescueExpert

    #: The longest plan is 24 presses; the rest is room for the episode's
    #: exploration prefix inside the adapter's 200-press per-level budget.
    max_steps = 80

    #: The largest level expands 1159 states. The cap is a MEMORY budget, not a
    #: time one -- it is here so an edited level fails loudly.
    node_cap = CAP


if __name__ == "__main__":
    if "--verify" in sys.argv:
        i = sys.argv.index("--verify")
        rest = tuple(int(x) for x in sys.argv[i + 1:] if x.isdigit())
        sys.exit(_verify(rest or None))
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--protect" in sys.argv:
        sys.exit(_report(protect=True))
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    sys.exit(ZombieRescueSolver.main())
