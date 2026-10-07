"""Generate Phase-1 training data for the PuzzleScript game ps:zombie_invasion
("Zombie Invasion", Owen).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a native model of the
three-rule chase, the two-pass search that plans over it, and the checks that pin
the model to the interpreter.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_zombie_invasion",
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
Each step also carries the full set of equally-optimal presses.

The game
--------
Walk to the Goal without being caught. One press moves you one cell; the win is
``some Player on Goal``. There are three rules and the whole game is in them::

    [ Zombie | ... | Player ] -> [ > Zombie | ... | Player ]
    [ > Zombie | Wall ] -> [ < Zombie | Wall ]
    [ > Zombie | Player ] -> [ Zombie | Zombie ]   message You got bitten! ...

and everything below is a consequence that the .txt does not show:

  * **Sight is a LINE and nothing blocks it.** The ellipsis is content-agnostic
    in this interpreter, so a zombie sharing your row or column steps one cell
    down that line **through walls and through other zombies**, at any range.
    There is no hiding; there is only not sharing a line. A zombie that shares
    neither does not move at all, which is what makes the game a puzzle rather
    than a chase -- you are choosing which zombies to wake.
  * **An orthogonally adjacent zombie kills you whatever you press.** Adjacent
    means in line, so it has a force, and it is your cell that force points at,
    so `[ > Zombie | Player ]` fires among the RULES -- before anything moves.
    Fleeing, waiting and walking into it are the same move. (A DIAGONALLY
    adjacent zombie is in no line at all and is perfectly safe, which is what
    every dodge in these four plans is made of.)
  * **Walls REPEL zombies, they do not shield you.** Rule 2 reverses a zombie
    whose next cell is a wall, so a zombie with a wall between it and you runs
    AWAY down the line -- and if you are standing behind it, that bounce is what
    bites you. The rules run in one pass in file order, each iterated over its
    four directional copies to a fixpoint, so rule 1 never re-fires after rule 2
    and the bounce always wins. **No shipped level can reach it**: the four
    boards are one open room inside a wall ring, and a zombie's force always
    points along the ray to you, which lies inside the room. Measured: zero
    bounces across all four plans and the whole level fuzz, ~1700 on the
    synthetic boards that exist only to reach the rule.
  * **A refused press is a real WAIT.** There is no ``require_player_movement``,
    so walking into the wall ring still ticks the zombies -- it is ACTION by
    another name. ACTION is a wait too, and both stay in the press set because
    dropping either would be an unproven prune (measured: no shortest plan here
    spends one).
  * **Death is not GAME_OVER.** Nothing in this game is a ``PlayerDead``
    object -- rule 3 simply overwrites you with a Zombie -- so
    `PSEngine.check_game_over` stays False and the board FREEZES (no player, so
    no zombie is ever in line with one again). `_Board.step` reports it as a
    dead successor, so it never enters the search; the recorder's exploration
    prefix does reach it, and one RESET is the way out (see Recovery below).
  * Two objects that want the same cell in one turn both stop. That is the
    head-on case: with a zombie two cells away in line, pressing INTO it leaves
    the board exactly as it was, and it is the only way this game refuses a
    press that is not a wall.

Expert solver
-------------
Not a search over the interpreter. `_Board` is a native model of all three rules
over integer cell indices -- state ``(player, zombies)``, with walls and goals
static, and death folded into a dead successor -- and it plans in two passes:

  1. **A-star** with an admissible, consistent heuristic: the wall-only BFS
     distance from the player to the nearest Goal (zombies ignored). A press
     moves the player at most one cell, so it can never over-estimate. It proves
     ``d*`` while touching a tiny slice of the space -- 45 states on level 0,
     1211 on level 3, whose board carries eight zombies.
  2. **The exact field inside that bound**: a forward sweep pruned by
     ``g + h <= d*`` records predecessors, and a reverse sweep from the winning
     states gives the true remaining press count of every state a shortest path
     can pass through.

Two passes rather than one because the whole reachable space is far too big to
enumerate -- level 1 alone has 170k states and levels 2 and 3 pass three
million, against fields of 32 and 1288. The second pass is what makes the
**optimal-action SETS measured rather than inferred**: at distance ``d``, a
press is optimal iff it lands on a state at distance ``d - 1``. Without them
every free stretch of the walk to the Goal would train one arbitrary
interleaving of the two axes as the single right answer.

There is no ``plan_cache_path``: all four searches together take ~50 ms, so a
disk cache would only add a staleness check to something cheaper than reading
the file.

The levels
----------
All 4 shipped levels are solved and every plan is PROVED shortest; nothing was
authored or skipped. 54 presses in all (11, 11, 15, 17) over boards of 1, 2, 4
and 8 zombies, 11 of them steps with more than one equally-optimal press (65
labelled presses in all). Level
0's 11 presses are the bare Manhattan distance -- its one zombie never gets a
line on the route -- and the later levels pay 4 and 6 extra presses in dodges.
The longest plan is 17 presses against the adapter's 200-press per-level budget,
so this game needs no ``games/`` step-limit wrapper.

The five checks, and what each one can and cannot see:

  * ``--plans`` builds every field and replays its plan through the real
    interpreter, requiring the win on the LAST press and no earlier. That is the
    only check that can catch a plan which is valid but not shortest.
  * ``--fuzz`` compares `_Board` against the interpreter over ~41k transitions
    from two populations, because neither alone is enough. Random play from
    every PREFIX of every level's own plan puts the model in the configurations
    the plans visit -- random play from a level start wanders into a line and
    dies, so it would never reach the later boards. And ~3000 dense random
    5x5..8x8 boards with interior walls are the ONLY way to reach
    ``[ > Zombie | Wall ]`` at all (see above). The counters are printed for
    exactly that reason: a run reporting agreement while having bounced no
    zombie has not checked the rule.
  * ``--verify`` re-derives every plan length, asserts the heuristic admissible
    at every state of every field (``h <= dist``), and re-derives every tie set
    with a bounded forward search sharing no code with the predecessor map they
    came from.
  * ``--symmetry`` replays each plan on all 8 turned and mirrored copies of its
    own board THROUGH THE INTERPRETER. It is the only check that can see a
    rule-order chirality -- `_execute_rule` drives a rule's four directional
    copies in the fixed order up, down, left, right, which is a fact about the
    SCREEN and not about the board -- and it is what justifies putting this game
    in `PuzzleScriptAdapter._FLIP_GAMES` rather than only rotating it.
  * ``--audit`` renders every cell composition at the cell size in use and
    asserts pairwise distinctness. It is what caught the invisible wall; the two
    recolors that fix it live in ``games/ps:zombie_invasion/`` so that the
    generator and a live agent see the same board (hence ``game_module_id``).

Augmentation
------------
The engine state after reset is identical for every seed (the levels are fixed
ASCII maps), so the only per-(seed, level) variable is presentation: the frame
rotation and flips plus the matching directional action remap. The expert plan
is therefore seed-independent -- solved once per level, memoized, and replayed
per seed with that seed's remapped screen actions.

The game is in `_FLIP_GAMES` (16 presentations per level rather than 4). It is
the ESCAPE! argument: gravity-free, screen-relative input, a win condition that
names no direction, no directional sprite anywhere -- and, unlike ESCAPE!, one
rule whose matches could in principle contend, which ``--symmetry`` measures on
the interpreter instead of arguing about. Without the flips the entire corpus
would be 4 levels x 4 rotations.

Recovery is the shared ``recovery_mode = "reset"`` arc, and this game is one of
the sharpest cases for it: an exploration prefix here usually ends with the
player eaten and the board frozen (nothing can ever move again), which no amount
of playing on can undo, and ONE RESET restores the level start the cached plan
was solved from.

Usage (run from the repo root):
    python solvers/generate_zombie_invasion_training.py \
        --episodes 200 --out data/training_multi_level/zombie_invasion

    python solvers/generate_zombie_invasion_training.py --plans
    python solvers/generate_zombie_invasion_training.py --verify
    python solvers/generate_zombie_invasion_training.py --fuzz
    python solvers/generate_zombie_invasion_training.py --symmetry
    python solvers/generate_zombie_invasion_training.py --audit
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from collections import deque
from heapq import heappush, heappop
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan      # noqa: E402

GAME_NAME = "Zombie_Invasion"
GAME_MODULE = "ps:zombie_invasion"

#: Refuse to search past this many states. It is a MEMORY budget -- every state
#: is a tuple of ints -- and it exists so an edited level fails loudly instead of
#: swapping. The largest shipped level expands 1211.
CAP = 2_000_000

#: Engine direction names in the order the interpreter expands a rule's four
#: directional copies. `_Board` mirrors that order exactly, so this tuple is part
#: of the model rather than a display convention.
_ORDER: tuple[str, ...] = ("up", "down", "left", "right")
_DELTA: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
_OPP: tuple[int, ...] = (1, 0, 3, 2)
_CODE: dict[str, int] = {name: i for i, name in enumerate(_ORDER)}
#: The five presses, ACTION last. ACTION is a genuine WAIT here (there is no
#: ``require_player_movement`` and no rule reads the action key at all).
_PRESSES: tuple[str, ...] = _ORDER + ("action",)


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """Zombie Invasion's whole rule set, over integer cell indices.

    A state is ``(player, zombies)``: the player's cell index and the sorted
    tuple of zombie cells. Walls and goals are static -- no rule creates or
    destroys either -- so they live on the board rather than in the state, and
    being eaten is not in the state at all: it is terminal (the win condition is
    ``some Player on Goal`` and nothing brings the player back), so `step`
    reports it as a dead successor.
    """

    def __init__(self, h, w, walls, goals, start):
        self.h, self.w = h, w
        self.walls, self.goals = walls, goals
        n = h * w
        self.nbr = [[-1] * 4 for _ in range(n)]
        for r in range(h):
            for c in range(w):
                cell = r * w + c
                for d, (dr, dc) in enumerate(_DELTA):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        self.nbr[cell][d] = rr * w + cc
        self.start = start
        self.dist_goal = self._bfs_goals()
        self.reachable = 0

    @classmethod
    def read(cls, eng, ids) -> "_Board":
        """Build the model, and its start state, from the interpreter's grid."""
        wall, goal, player, zombie = ids
        w = eng.width
        walls, goals = set(), set()
        p, zs = -1, []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i = r * w + c
                if cell & wall:
                    walls.add(i)
                if cell & goal:
                    goals.add(i)
                if cell & player:
                    p = i
                if cell & zombie:
                    zs.append(i)
        return cls(eng.height, w, walls, goals, (p, tuple(sorted(zs))))

    # -- dynamics -------------------------------------------------------------
    def win(self, state) -> bool:
        """``some Player on Goal``."""
        return state[0] in self.goals

    def _forces(self, p, zombies) -> dict:
        """The forces rules 1 and 2 leave on the zombies, as ``{cell: dir}``.

        Rule 1 (``[ Zombie | ... | Player ]``): the ellipsis is content-agnostic
        in this interpreter, so every zombie sharing the player's row or column
        is sent one cell down that line -- through walls, through other zombies,
        at any range. A zombie sharing neither gets no force and does not move.

        Rule 2 (``[ > Zombie | Wall ]``): a zombie whose next cell is a WALL is
        reversed, i.e. it runs away from you rather than being stopped by the
        wall. One flip is enough. The interpreter iterates each rule's four
        directional copies to a fixpoint, so a zombie walled on BOTH sides of
        its line flips twice per iteration and settles on the first of the pair
        in the ``up, down, left, right`` scan -- but both of its cells are walls,
        so it cannot move either way and the only thing the final value could
        change is rule 3, which needs a PLAYER in that cell rather than a wall.
        Rule 1 does not re-fire afterwards: the three rules are one pass in file
        order, so the bounce is final."""
        w = self.w
        pr, pc = divmod(p, w)
        out = {}
        for z in zombies:
            zr, zc = divmod(z, w)
            if zr == pr:
                out[z] = 3 if zc < pc else 2
            elif zc == pc:
                out[z] = 1 if zr < pr else 0
        nbr, walls = self.nbr, self.walls
        for z, d in out.items():
            n = nbr[z][d]
            if n >= 0 and n in walls:
                out[z] = _OPP[d]
        return out

    def step(self, state, press):
        """One press. Returns ``(next state, contended)``.

        The next state is None when the player is eaten -- rule 3 fires among
        the RULES, on the board as it stands before anything moves, so a zombie
        whose force points at your cell bites you whatever you pressed -- and
        ``state`` itself when the press changed nothing.

        ``contended`` is True when two independently-moving objects wanted the
        same cell in the same turn, which is the only way this game's outcome
        could depend on the order the interpreter expands a rule's four
        directions (see the module docstring)."""
        code = _CODE.get(press, 4) if isinstance(press, str) else press
        p0, z0 = state
        zombies = set(z0)
        nbr = self.nbr
        pd = code if code < 4 else None

        zf = self._forces(p0, zombies)
        for z, d in zf.items():
            if nbr[z][d] == p0:
                return None, False            # rule 3: you are a zombie now

        p, zombies, contended = self._resolve(p0, pd, zombies, zf)
        nxt = (p, tuple(sorted(zombies)))
        return (state if nxt == state else nxt), contended

    def _resolve(self, p, pd, zombies, zf):
        """Move everything that has a force, PuzzleScript-style.

        The player, the zombies and the walls are all on one collision layer, so
        a mover is blocked by anything solid that does not vacate its cell first.
        A chain of same-direction movers (a column of zombies charging down one
        line) moves together when the far end is free; two movers claiming one
        cell both stop, which is the head-on case in the module docstring."""
        nbr, walls = self.nbr, self.walls
        # cell -> ('p' | 'z'); one solid object per cell by construction.
        occ = {z: "z" for z in zombies}
        occ[p] = "p"
        forces = {}
        if pd is not None:
            forces[p] = pd
        forces.update(zf)
        contended = False

        for _ in range(20):
            movable, moved = [], False
            for cell, d in list(forces.items()):
                chain, cur, free, blocker_moves = [cell], cell, False, False
                while True:
                    n = nbr[cur][d]
                    if n < 0:
                        break
                    if n not in walls and n not in occ:
                        free = True
                        break
                    if n not in walls and forces.get(n) == d:
                        chain.append(n)
                        cur = n
                        continue
                    if n not in walls and n in forces:
                        blocker_moves = True
                    break
                if free:
                    movable.append((chain, d))
                elif blocker_moves:
                    # The blocker is heading somewhere else and may vacate this
                    # cell later in the pass; keep the force and retry.
                    pass
                else:
                    for x in chain:
                        forces.pop(x, None)
            # A chain whose head is the tail of a longer one is the same push
            # group seen from further back; only the longer one moves.
            tails = {x for chain, _ in movable for x in chain[1:]}
            live = [m for m in movable if m[0][0] not in tails]
            claims, clash = {}, set()
            for i, (chain, d) in enumerate(live):
                for x in chain:
                    t = nbr[x][d]
                    if t in claims and claims[t] != i:
                        clash.add(i)
                        clash.add(claims[t])
                        contended = True
                    else:
                        claims[t] = i
            for i, (chain, d) in enumerate(live):
                if i in clash:
                    for x in chain:
                        forces.pop(x, None)
                    continue
                for x in reversed(chain):
                    kind = occ.pop(x)
                    occ[nbr[x][d]] = kind
                    forces.pop(x, None)
                    moved = True
            if not moved or not forces:
                break
        p = next(c for c, k in occ.items() if k == "p")
        return p, {c for c, k in occ.items() if k == "z"}, contended

    # -- planning -------------------------------------------------------------
    def _bfs_goals(self) -> list:
        """Presses from every cell to the nearest Goal, walls only. Zombies are
        ignored, which is what makes it a relaxation."""
        INF = 1 << 20
        d = [INF] * (self.h * self.w)
        q = deque()
        for g in self.goals:
            d[g] = 0
            q.append(g)
        while q:
            cur = q.popleft()
            for n in self.nbr[cur]:
                if n >= 0 and n not in self.walls and d[n] == INF:
                    d[n] = d[cur] + 1
                    q.append(n)
        return d

    def heuristic(self, state) -> int:
        """Presses that must still be spent: the wall-only distance to the
        nearest Goal.

        Admissible -- a press moves the player at most one cell and the win
        needs the player ON a goal, so no press set can be shorter -- and
        consistent, since one press changes it by at most one. Zero exactly at a
        win, which is what lets `astar` test the goal on generation."""
        return self.dist_goal[state[0]]

    def astar(self, cap: int):
        """A* over presses; returns ``(d*, expanded)`` or ``(None, expanded)``.

        The goal is tested on GENERATION rather than on the pop, which is sound
        because `heuristic` is zero exactly at a win: every non-winning node has
        ``f > g`` strictly, so a win reachable one press sooner would have a
        predecessor whose ``f`` is at most that shorter length -- strictly less
        than the ``f`` of anything being expanded -- and would have been popped,
        and its win generated, first."""
        start = self.start
        if self.win(start):
            return 0, 0
        pq = [(self.heuristic(start), 0, start)]
        best = {start: 0}
        expanded = 0
        while pq:
            _f, g, state = heappop(pq)
            if best.get(state, -1) != g:
                continue
            expanded += 1
            if expanded > cap:
                return None, expanded
            for press in range(5):
                nxt, _cont = self.step(state, press)
                if nxt is None or nxt == state:
                    continue
                ng = g + 1
                if self.win(nxt):
                    return ng, expanded
                if best.get(nxt, 1 << 30) <= ng:
                    continue
                best[nxt] = ng
                heappush(pq, (ng + self.heuristic(nxt), ng, nxt))
        return None, expanded

    def field(self, dstar: int, cap: int):
        """Exact distance-to-win over every state on a SHORTEST path.

        A forward sweep bounded by ``g + h <= dstar`` (admissible ``h``, so it
        cannot drop an optimal path) records each state's predecessors; a
        reverse sweep from the winning states then gives the true remaining
        press count. Walking that field downhill is what makes both the plan
        provably shortest and its per-step optimal SETS exact rather than
        inferred."""
        pred: dict = {}
        depth = {self.start: 0}
        frontier = [self.start]
        wins = []
        for g in range(dstar):
            nxt_frontier = []
            for state in frontier:
                for press in range(5):
                    nxt, _cont = self.step(state, press)
                    if nxt is None or nxt == state:
                        continue
                    if g + 1 + self.heuristic(nxt) > dstar:
                        continue
                    pred.setdefault(nxt, []).append(state)
                    if nxt in depth:
                        continue
                    depth[nxt] = g + 1
                    if self.win(nxt):
                        wins.append(nxt)
                    else:
                        nxt_frontier.append(nxt)
                if len(depth) > cap:
                    raise MemoryError(f"field exceeded {cap} states")
            frontier = nxt_frontier
        dist = {state: 0 for state in wins}
        queue = deque(wins)
        while queue:
            state = queue.popleft()
            d = dist[state] + 1
            for prev in pred.get(state, ()):
                if prev not in dist:
                    dist[prev] = d
                    queue.append(prev)
        self.reachable = len(depth)
        return dist

    def plan(self, dist: dict):
        """``(presses, optsets)`` from the start, or ``(None, None)``.

        At every state the optimal SET is exactly the presses landing on
        distance ``d - 1`` -- measured off the field, not inferred from what
        kind of move it was."""
        state = self.start
        if state not in dist:
            return None, None
        presses, optsets = [], []
        while dist[state]:
            want = dist[state] - 1
            best = []
            for i, name in enumerate(_PRESSES):
                nxt, _cont = self.step(state, i)
                if nxt is not None and dist.get(nxt, -1) == want:
                    best.append(name)
            if not best:
                return None, None
            presses.append(best[0])
            optsets.append(best)
            state = self.step(state, _PRESSES.index(best[0]))[0]
        return presses, optsets


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class ZombieInvasionExpert(PSExpert):
    """Plans by building `_Board` off the engine grid and searching THAT; never
    steps the interpreter.

    `PSExpert` still owns everything around the search -- the in-memory memo,
    the level scoping and the snapshot/restore discipline -- so the only
    override is `_search`. `heuristic` is unreachable by construction: no A*
    runs over engine states here, only over `_Board`'s.
    """

    #: `_key` below is dynamic-objects-only, canonical only WITHIN a level.
    scope_by_level = True

    #: ACTION is a genuine WAIT (no rule reads the action key), and so is any
    #: press into the wall ring. No shipped plan spends one, but dropping the
    #: press would be an unproven prune rather than a measured one.
    directions = list(_PRESSES)

    def setup(self) -> None:
        g = self.g
        name = g.resolve_object_name
        self.wall_ids = set(name("wall"))
        self.goal_ids = set(name("goal"))
        self.player_ids = set(self.game._engine._player_indices)
        self.zombie_ids = set(name("zombie"))
        self.dyn_ids = self.player_ids | self.zombie_ids

    def _key(self, eng) -> frozenset:
        """Every object a settled frame can differ by. Walls and goals are left
        out deliberately -- no rule in this game creates or destroys either, so
        they are static per level (hence ``scope_by_level``)."""
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "ZombieInvasionExpert searches the native model; the engine-state "
            "heuristic is unused")

    def board(self, eng) -> _Board:
        return _Board.read(eng, (self.wall_ids, self.goal_ids,
                                 self.player_ids, self.zombie_ids))

    def _search(self, eng) -> "Plan | None":
        """A* for the shortest length, then the exact field inside that bound.

        Two passes rather than one because they answer different questions. A*
        with the goal-distance heuristic proves ``d*`` while touching a tiny
        slice of the space (45 states on level 0, 1211 on level 3), and the
        whole reachable space is far too big to enumerate instead -- level 1
        alone holds 170k states and levels 2 and 3 pass three million. The
        ``g + h <= d*`` field then enumerates exactly the states a shortest path
        can pass through, which is what makes the per-step optimal SETS measured
        rather than inferred: without them every free stretch of the walk to the
        Goal would train one arbitrary interleaving of the two axes as the only
        right answer."""
        board = self.board(eng)
        dstar, _expanded = board.astar(self.node_cap)
        if dstar is None:
            return None
        presses, optsets = board.plan(board.field(dstar, self.node_cap))
        return None if presses is None else Plan(presses, optsets)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def _levels():
    solver = ZombieInvasionSolver()
    game = solver.make_game(0)
    return solver, game, ZombieInvasionExpert(game)


def _engine_state(eng, expert):
    """The interpreter's grid in `_Board`'s state encoding, plus the terminal
    marker the model folds into a dead successor (the player was eaten, so no
    Player object is left on the board)."""
    w = eng.width
    p, zs = -1, []
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            if cell & expert.player_ids:
                p = i
            if cell & expert.zombie_ids:
                zs.append(i)
    return (p, tuple(sorted(zs))), p < 0


def _report(cap: int = CAP) -> int:
    """Per-level zombie count, search size, plan length and tie coverage -- and
    CERTIFY each plan by replaying it through the interpreter."""
    _solver, game, expert = _levels()
    bad = 0
    total = labels = waits = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        t = time.time()
        dstar, expanded = board.astar(cap)
        if dstar is None:
            print(f"level {level:2d}: {len(board.start[1])}Z, "
                  f"NO PLAN ({expanded} states in {time.time() - t:.1f}s)")
            bad += 1
            continue
        dist = board.field(dstar, cap)
        presses, optsets = board.plan(dist)
        took = time.time() - t
        won_at = None
        for i, press in enumerate(presses):
            eng.step(press)
            if eng.check_win():
                won_at = i
                break
        ok = won_at == len(presses) - 1
        bad += not ok
        total += len(presses)
        waits += sum(1 for x in presses if x == "action")
        ties = sum(1 for x in optsets if len(x) > 1)
        here = sum(len(x) for x in optsets)
        labels += here
        print(f"level {level:2d}: {len(board.start[1])}Z, "
              f"{len(presses):3d} presses, {ties:2d} tie steps, "
              f"{here:3d} labels, "
              f"A* {expanded:6d} states, field {board.reachable:6d}, "
              f"{took:6.2f}s, "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    print(f"total {total} presses over {game.n_levels} levels, "
          f"{labels} labelled presses, {waits} of them waits")
    return 0 if not bad else 1


def _plans_for(game, expert) -> dict:
    """``{level: (presses, optsets)}`` -- every level's field plan, so the fuzz
    and the checks can start from the states a plan actually visits."""
    out = {}
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine)
        dstar, _n = board.astar(CAP)
        if dstar is None:
            continue
        out[level] = board.plan(board.field(dstar, CAP))
    return out


def _bounces(board, state) -> int:
    """How many zombies rule 2 reverses at ``state`` -- the counter that says
    whether a fuzz population reached the wall rule at all."""
    p, zs = state
    w = board.w
    pr, pc = divmod(p, w)
    n = 0
    for z in zs:
        zr, zc = divmod(z, w)
        if zr == pr:
            d = 3 if zc < pc else 2
        elif zc == pc:
            d = 1 if zr < pr else 0
        else:
            continue
        t = board.nbr[z][d]
        n += t >= 0 and t in board.walls
    return n


def _walk(board, eng, expert, state, rng, length, totals) -> int:
    """Random-play ``length`` presses from ``state``, asserting the model and
    the interpreter agree after every one. Returns the number of mismatches
    (0 or 1 -- the walk stops at the first)."""
    for _ in range(length):
        press = rng.choice(_PRESSES)
        before = state
        totals["bounces"] += _bounces(board, state)
        eng.step(press)
        state, contended = board.step(state, press)
        totals["presses"] += 1
        totals["contended"] += contended
        theirs, dead = _engine_state(eng, expert)
        if state is None:
            totals["deaths"] += 1
            if not dead:
                print(f"  FALSE DEATH after {press} from {before}\n"
                      f"    engine {theirs}")
                return 1
            return 0
        if dead or state != theirs:
            print(f"  MISMATCH after {press}\n    before {before}\n"
                  f"    model  {state}\n    engine {theirs} dead={dead}")
            return 1
        if board.win(state) != eng.check_win():
            print(f"  WIN MISMATCH at {state}")
            return 1
        totals["refused"] += state == before
        totals["moved"] += before[1] != state[1]
        if board.win(state):
            return 0
    return 0


def _random_board(rng, expert) -> list:
    """A dense synthetic level, and the only thing here that can reach
    ``[ > Zombie | Wall ]`` at all.

    The wall bounce is the one rule whose direction is not simply "towards the
    player", and no shipped board can ever fire it: the four levels are a single
    open room inside a wall ring, and a zombie's force always points along the
    ray to the player, which lies inside the room (measured -- zero bounces
    across all four plans and the whole level fuzz). So the configuration has to
    be manufactured: interior walls, several zombies, a small board so lines are
    shared constantly."""
    h, w = rng.randint(5, 8), rng.randint(5, 8)
    cells = [(r, c) for r in range(h) for c in range(w)]
    rng.shuffle(cells)
    grid = [[{expert.bg_id} for _ in range(w)] for _ in range(h)]

    def place(ids, n):
        oid = min(ids)
        for _ in range(n):
            if not cells:
                return
            r, c = cells.pop()
            grid[r][c].add(oid)

    place(expert.player_ids, 1)
    place(expert.zombie_ids, rng.randint(2, 5))
    place(expert.wall_ids, rng.randint(2, 8))
    place(expert.goal_ids, 1)
    return grid


def _fuzz(walks: int = 60, length: int = 40, boards: int = 3000) -> int:
    """Assert `_Board` reproduces the interpreter exactly.

    Two populations, because neither alone is enough:

      * random play from every PREFIX of every level's own plan, which puts the
        model in the configurations the shipped plans actually visit (random
        play from a level start walks into a line and dies, and would never
        reach the later boards);
      * dense random 5x5..8x8 boards WITH INTERIOR WALLS, which is the only way
        to reach ``[ > Zombie | Wall ]`` at all.

    The counters are printed for exactly that reason -- a run that reports
    agreement while having bounced no zombie has not checked the rule that
    decides what a wall does in this game."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    rng = random.Random(20260823)
    plans = _plans_for(game, expert)
    bad = 0
    totals = dict(presses=0, refused=0, deaths=0, moved=0, contended=0,
                  bounces=0)
    for level in range(game.n_levels):
        game.set_level(level)
        layout = g.levels[level]
        board = expert.board(eng)
        presses = plans.get(level, ([], []))[0]
        seen = dict(totals)
        for prefix in range(0, len(presses) + 1, max(1, len(presses) // 12)):
            for _ in range(walks):
                eng.load_level(layout)
                state = board.start
                for press in presses[:prefix]:
                    eng.step(press)
                    state = board.step(state, press)[0]
                bad += _walk(board, eng, expert, state, rng, length, totals)
        print(f"level {level:2d}: {totals['presses'] - seen['presses']:6d} presses, "
              f"{totals['deaths'] - seen['deaths']:5d} deaths, "
              f"{totals['refused'] - seen['refused']:5d} refused, "
              f"{totals['contended'] - seen['contended']:4d} contended, "
              f"{totals['bounces'] - seen['bounces']:4d} bounces")
    seen = dict(totals)
    for _ in range(boards):
        grid = _random_board(rng, expert)
        eng.load_level(grid)
        board = _Board.read(eng, (expert.wall_ids, expert.goal_ids,
                                  expert.player_ids, expert.zombie_ids))
        bad += _walk(board, eng, expert, board.start, rng, 12, totals)
    print(f"random boards: {totals['presses'] - seen['presses']:6d} presses, "
          f"{totals['deaths'] - seen['deaths']:5d} deaths, "
          f"{totals['refused'] - seen['refused']:5d} refused, "
          f"{totals['contended'] - seen['contended']:4d} contended, "
          f"{totals['bounces'] - seen['bounces']:4d} bounces")
    print(f"{totals['presses']} transitions ({totals['deaths']} fatal, "
          f"{totals['refused']} refused, {totals['contended']} contended, "
          f"{totals['bounces']} bounced): "
          f"{'model matches the interpreter' if not bad else f'{bad} MISMATCHES'}")
    return 0 if not bad else 1


def _independent(board: _Board, state, limit: int) -> "int | None":
    """Presses from ``state`` to a win, or None if that is more than ``limit``.

    A plain forward breadth-first search to the FIRST winning state, pruned by
    ``depth + heuristic > limit`` (sound, because the heuristic is admissible --
    which `_verify` asserts against the field before trusting it here). It is
    the second book in `_verify`'s double entry: `_Board.field` derives its
    answer BACKWARDS, from the win states over a predecessor map built during
    the forward sweep, and that bookkeeping is the one piece of logic here the
    interpreter fuzz cannot reach -- the fuzz only ever exercises `step`. This
    shares nothing with it but `step` and `heuristic`."""
    if board.win(state):
        return 0
    seen = {state}
    frontier = [state]
    for depth in range(1, limit + 1):
        nxt_frontier = []
        for cur in frontier:
            for press in range(5):
                nxt, _c = board.step(cur, press)
                if nxt is None or nxt == cur or nxt in seen:
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

      * the LENGTH is already double-entered by `_report`: A* proves ``d*``
        without ever touching the field, and the interpreter certifies that the
        field's plan wins on its LAST press and no earlier. Both are re-asserted
        here so one command covers everything.
      * the HEURISTIC is admissible, asserted over every state of the field
        rather than argued -- ``h(s) <= dist(s)`` everywhere. `field`'s
        ``g + h <= d*`` prune leans on it, so a heuristic that over-estimated
        would silently drop optimal paths.
      * the TIE SETS are re-derived by `_independent`, which shares no code with
        the predecessor map they come from.
    """
    _solver, game, expert = _levels()
    bad = 0
    for level in (levels if levels is not None else range(game.n_levels)):
        game.set_level(level)
        board = expert.board(game._engine)
        dstar, _n = board.astar(CAP)
        dist = board.field(dstar, CAP)
        presses, optsets = board.plan(dist)
        star = dist[board.start]
        t = time.time()
        notes = []
        if star != len(presses) or star != dstar:
            notes.append(f"LENGTH {star} != {len(presses)} / A* {dstar}")
        loose = [x for x in dist if board.heuristic(x) > dist[x]]
        if loose:
            notes.append(f"HEURISTIC INADMISSIBLE at {len(loose)} states")
        state = board.start
        for i, press in enumerate(presses):
            want = []
            for j, name in enumerate(_PRESSES):
                nxt, _c = board.step(state, j)
                if nxt is None or nxt == state:
                    continue
                if _independent(board, nxt, star - i - 1) == star - i - 1:
                    want.append(name)
            if want != optsets[i]:
                notes.append(f"step {i}: {optsets[i]} != {want}")
            state = board.step(state, _PRESSES.index(press))[0]
        bad += len(notes)
        print(f"level {level:2d}: d*={star:3d}, {len(dist):6d} field states "
              f"admissible, {sum(len(x) for x in optsets):3d} labelled presses "
              f"re-derived in {time.time() - t:6.2f}s: "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


_ROT_D: dict[str, str] = {"up": "right", "right": "down",
                          "down": "left", "left": "up"}
_FLIP_D: dict[str, str] = {"up": "up", "down": "down",
                           "left": "right", "right": "left"}


def _tdir(direction: str, k: int, mirror: bool) -> str:
    if direction == "action":
        return direction
    for _ in range(k):
        direction = _ROT_D[direction]
    return _FLIP_D[direction] if mirror else direction


def _tgrid(grid, k: int, mirror: bool):
    """``grid`` turned ``k`` quarter-turns clockwise, then optionally mirrored
    left-right. No object here is directional, so nothing has to be relabelled
    -- which is half the reason the game can take the flips at all."""
    for _ in range(k):
        h = len(grid)
        grid = [[grid[h - 1 - r][c] for r in range(h)]
                for c in range(len(grid[0]))]
    if mirror:
        grid = [list(reversed(row)) for row in grid]
    return [[set(cell) for cell in row] for row in grid]


def _symmetry() -> int:
    """Confirm ON THE INTERPRETER that no plan step depends on which way the
    board is facing.

    `_execute_rule` drives each of a rule's four directional copies to a
    fixpoint IN TURN (up, down, left, right), so a rule whose matches can
    contend is settled by an order that is a fact about the SCREEN, not about
    the board -- and the augmentation then presents the same physical situation
    resolving both ways (this is what ps:gobble_rush had to measure and steer
    around). Here it is the head-on stop and the two-sided wall bounce that
    could do it.

    So each level's plan is replayed on all 8 turned and mirrored copies of its
    own board and the result compared against the transform of the reference
    run, press by press. This is the only check that can see it: the fuzz plays
    one board, where the model and the interpreter agree on an order neither can
    perceive. It is also what justifies `_FLIP_GAMES` membership -- a mirror is
    only sound if the mechanic has no handedness."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    plans = _plans_for(game, expert)
    bad = 0
    for level in range(game.n_levels):
        presses = plans.get(level, ([], []))[0]
        eng.load_level(g.levels[level])
        ref = [[[frozenset(c) for c in row] for row in eng.grid]]
        for press in presses:
            eng.step(press)
            ref.append([[frozenset(c) for c in row] for row in eng.grid])
        measured = None
        for k in range(4):
            for mirror in (False, True):
                if (k, mirror) == (0, False):
                    continue
                eng.load_level(_tgrid(g.levels[level], k, mirror))
                want = [_tgrid(x, k, mirror) for x in ref]
                for i, press in enumerate(presses):
                    eng.step(_tdir(press, k, mirror))
                    if eng.grid != want[i + 1]:
                        measured = i if measured is None else min(measured, i)
                        break
        bad += measured is not None
        print(f"level {level:2d}: {len(presses):3d} presses on 8 presentations: "
              f"{'ORIENTATION-FREE' if measured is None else f'DIVERGES AT STEP {measured}'}")
    print("symmetry check clean" if not bad
          else f"SYMMETRY FAILED: {bad} levels depend on the screen's facing")
    return 0 if not bad else 1


#: Every cell stack the four levels can present: the Goal (or nothing) under one
#: of the layer-3 objects (or nothing). ``player_on_goal`` is the WIN frame and
#: is the one a colour-only audit is most likely to be blind to.
_LAYER2: tuple[tuple[str, tuple[str, ...]], ...] = (("", ()), ("goal", ("goal",)))
_LAYER3: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("", ()), ("wall", ("wall",)), ("player", ("player",)),
    ("zombie", ("zombie",)),
)


def _compositions() -> dict:
    """The product of the two layers, minus the one stack no level can present.

    A Wall standing ON a Goal is unreachable -- the level ASCII writes a goal
    with ``G`` and a wall with ``#``, and no rule moves either -- and it is the
    one pair that clashes, since the wall's sprite is opaque and hides whatever
    is under it. Auditing it would be auditing a board the game cannot draw."""
    out = {}
    for n2, o2 in _LAYER2:
        for n3, o3 in _LAYER3:
            if (n2, n3) == ("goal", "wall"):
                continue
            name = "_on_".join(x for x in (n3, n2) if x) or "floor"
            out[name] = o2 + o3
    return out


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor).

    This is the check that convicted the shipped art: `wall` and `floor` were
    pixel-identical, because GREEN and darkgreen are the same ARC colour. The
    fix is in ``games/ps:zombie_invasion/ps:zombie_invasion.py``, which is why
    this generator builds its adapter through ``game_module_id`` -- an audit run
    against a plain `PuzzleScriptAdapter` would be auditing a board no agent
    ever sees."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = _compositions()

    sizes: dict = {}
    for level in range(game.n_levels):
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


class ZombieInvasionSolver(PSAStarSolver):
    game_id = "puzzlescript_zombie_invasion"
    game_name = GAME_NAME
    expert_cls = ZombieInvasionExpert

    #: The adapter is built by ``games/ps:zombie_invasion/`` so the generator
    #: records the recoloured board a live agent is given -- without it the wall
    #: ring is invisible (see that file, and `_audit`).
    game_module_id = GAME_MODULE

    #: The longest plan is 17 presses; the rest is room for the episode's
    #: exploration prefix inside the adapter's 200-press per-level budget.
    max_steps = 60

    #: The largest level expands 1211 states. The cap is a MEMORY budget, not a
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
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    sys.exit(ZombieInvasionSolver.main())
