"""Generate Phase-1 training data for the PuzzleScript game ps:stand_off
("Stand Off", Mark Richardson).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: a native model of the line-of-sight / two-gun mechanic,
the exact distance field that plans over it, and the checks that pin the model to
the interpreter.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_stand_off",
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
Walk to the Exit. That is the whole win condition (``any Exit on Player``), and
it would be a plain maze except for the rule that ends every turn:

    late [ Player | ... | LoS | BadGuy ] -> [ Player | ... | Bullet | BadGuy ] again

A bad guy that shares a row or a column with you, with nothing OPAQUE strictly
between, shoots -- and the ``again`` loop walks the bullet all the way home
inside that same press, so there is no dodging and no warning. **Ending a turn
in a bad guy's line is simply death.** Bad guys never move, never look away and
never run out of bullets; the whole game is about what is standing between you
and them.

What blocks a line is ``Shroud = Wall or BadGuy or Crate or Gun or Exit or
Bullet`` -- and the list is as interesting for what is missing as for what is in
it. **Windows do not block it.** Half of these boards are built out of window
panes that stop you walking but not from being shot through, so the safe route
is rarely the open one. The Exit does block it, which is why the last step is
usually survivable at all.

**The guns are the mechanic.** ACTION5 arms you (a `GunMode` marker); the next
arrow key then DRAWS a gun into that neighbouring cell instead of walking, or
HOLSTERS one already there. So a gun costs two presses, and you may have at most
two at a time -- a third draw is refused outright by the game's own head count:

    random [stationary Gun] -> [up Gun]      (mark one)
    random [stationary Gun] -> [down Gun]    (mark a second)
    random [stationary Gun] -> cancel        (a third? the turn never happened)

A drawn gun is a Shroud, and ``[ > Player ][ stationary Gun ] -> [ > Player ][ >
Gun ]`` moves every gun with you, so the two guns are PORTABLE SHIELDS locked to
a fixed offset from your body -- until one is walked into something, because
``[ > Gun | Obstacle ] -> [ | Obstacle ]`` destroys it silently. Level 0 is the
whole game in twenty presses: cross the bad guy's column with a gun held over
your head, climb the far side, draw a second gun to your left, and step onto the
exit with both lines plugged.

Five things that are not readable off the .txt
----------------------------------------------
All five were found by pinning the model against the interpreter, and each one
changes what the search may do.

1. **Stepping onto the Exit WINS even when the exit square is itself in a bad
   guy's line.** The shot is fired -- the winning frame of such a level shows the
   bullet, and the window it has just broken -- but ``checkWin`` reads the flag
   set by the move, so the level is over before the ``again`` loop can walk the
   bullet home. Modelled as: a state whose player is on an Exit is a win, full
   stop, and its safety is never tested. Without this the search refuses the last
   press of levels whose exit is deliberately exposed.

2. **ACTION5 is not free.** It changes nothing but the mode bit, yet the turn
   still happens, so the late rules still run and you are still shot if you were
   standing in a line. There is no wait in this game: every press is checked.

3. **Drawing the third gun cancels the WHOLE turn, mode bit included.** The
   ``random`` head count above fires after the draw, so the board reverts and
   `GunMode` is still set -- the press is a genuine no-op, not "you wasted your
   mode". (Those three ``random`` rules are the only nondeterminism in the game
   and it is vacuous: with two or fewer stationary guns the first two rules mark
   them all and the fourth unmarks them, whichever order the die falls in.)

4. **A gun drawn onto the Exit is deleted the same turn** by ``late [Gun Exit] ->
   [Exit]``. So aiming at the exit from beside it costs a press and buys nothing
   -- unless you already hold two guns, in which case rule 3 fires first and the
   press does not even cost the mode bit.

5. **Standing on the Exit and pressing ACTION5 UN-WINS the level.** `GunMode`
   shares its collision layer with `Exit`, so arming yourself on the goal square
   evicts the goal square: the cell goes from ``{exit, player}`` to
   ``{player, gunmode}`` and ``any Exit on Player`` is false again. It is
   unreachable in play -- the win is read the moment you step on, and the level
   is over -- but it is exactly the kind of thing a model that treated the exit
   as "a square you may stand on" would wander into, so `_Board` treats a player
   on an Exit as TERMINAL and `_fuzz` refuses to press anything from there.

6. **BrokenWindow is Window.** It is in ``Obstacle`` and out of ``Shroud``
   exactly as Window is, so breaking one changes the picture and nothing else.
   The model does not carry it; the recorded frames come from the interpreter,
   which does.

The expert
----------
The model is three to four ORDERS OF MAGNITUDE faster than the interpreter
(`--speed` reports both, per level: 0.7-1.4M presses/s against 79-223, i.e.
3.6k-14.4kx), which is what makes an exact answer affordable -- a search that
would have taken hours in the blackbox finishes in seconds here. The budget is
spent on three passes per level:

  * **`hfield`** -- the exact distance-to-win of a RELAXED game in which crates
    are neither obstacles nor cargo, only shields, and a shield is granted
    wherever a crate could ever be pushed to (`crate_star`). It is a genuine
    lower bound (see `relaxed_field` for the mimicking argument, and note the
    optional gun destruction that argument needs), and unlike a plain
    distance-to-exit it prices the GUN CHOREOGRAPHY. What that is worth was
    measured, and it is honest to say it is worth less than it looks: on the two
    crate-free levels it is transformative (level 0 goes 14 -> 20, which is
    EXACT, and level 1 goes 10 -> 38 against a true 42), and on the five crate
    levels it is identical to the plain distance, because `crate_star` is
    generous enough there to hand out a shield on nearly every ray. Same plans,
    same lengths, 574 -> 311 and 605 -> 446 states on levels 0 and 1 and no
    change elsewhere. It is kept because it is never worse and because it is the
    field the skipped levels would need a crate-AWARE version of.
  * **``A*``** over the full state ``(player, gun offsets, mode, crates)`` with that
    heuristic. It returns on GENERATING a win rather than on popping one, which
    is still exact here because every non-win state has ``h >= 1``.
  * **the cone** -- once the optimal cost ``L`` is known, one more pass keeps
    every state with ``g + h <= L`` and runs a backward BFS over it. That is the
    exact distance-to-win for every state on any optimal path, so the per-step
    optimal SETS are measured, not inferred, for the cost of one extra search
    rather than the ~500 re-solves `PSExpert.optimal_sets` would need.

Three levels are skipped
------------------------
Levels 4, 8 and 9 are not recorded. They are winnable -- level 8's exit needs
three lines plugged at once with only two guns, which forces a crate to be walked
four corners across the board first -- but winning them is a sokoban whose goal
is a shield position, and every search tried on them (A* and a macro A* over
push-to-push edges, both under the heuristic above, plus a width-4000 beam) runs
out of budget without a plan. The heuristic is crate-blind by construction, which
is exactly the wrong blindness for those three, and a crate-aware one is a
different program. They are named in `skip_levels` so `discover_solvable` does not
re-burn the budget on them at every startup; the other seven are recorded with
plans that are PROVED shortest.

What the checks report
---------------------
  * ``--plans``  -- 342 presses over the seven recorded levels, all seven
    CERTIFIED on the interpreter (it wins on the plan's last press and on no
    earlier one, which is what makes "shortest" an end-to-end claim rather than
    a property of the model); 462 optimal labels, 244 of them forced.
  * ``--fuzz``   -- 15722 transitions from every prefix of every plan, model
    against interpreter, EXACT: 1232 gun draws, 1107 holsters, 352 crate pushes,
    4752 mode flips, 1256 deaths and 2933 refused presses, no disagreement.
  * ``--replay`` -- three seeds recorded and replayed from their recorded SCREEN
    actions, every frame equal, at rotations 0, 1, 2 and 3.
  * ``--audit``  -- 18 cell compositions pairwise distinct at all three cell
    sizes these boards use.

Reports (run from the repo root):
    python solvers/generate_stand_off_training.py --plans
    python solvers/generate_stand_off_training.py --fuzz
    python solvers/generate_stand_off_training.py --replay
    python solvers/generate_stand_off_training.py --audit
    python solvers/generate_stand_off_training.py --speed

Usage:
    python solvers/generate_stand_off_training.py \
        --episodes 200 --out data/training_multi_level/stand_off
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

from arcengine import ActionInput, GameState                          # noqa: E402
from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION                      # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan      # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Stand_Off"

#: Disk cache of each level's start plan AND its exact optimal-action sets. The
#: searches are seed-independent (only the PRESENTATION is augmented per seed)
#: and cost ~30s together, which every shard of `parallelize_generator` would
#: otherwise repeat on every core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "stand_off_plans.json"

#: Engine press names, in the order `_Board` codes them. The four directions
#: come first so a code doubles as a direction index; ACTION5 is code 4.
_ORDER: tuple[str, ...] = ("up", "down", "left", "right", "action")
_DELTA: tuple[tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
#: Direction code -> the code pointing back the other way (``d ^ 1``, spelled
#: out because `crate_star` reads it and the xor is easy to mis-read).
_OPP: tuple[int, ...] = (1, 0, 3, 2)
_ACTION = 4

#: Refuse to build a search larger than this. It is a MEMORY budget -- every
#: state is a 4-tuple plus a predecessor entry -- and it exists so a level that
#: cannot be reached fails loudly instead of swapping. The largest level
#: recorded here (7) opens 264k states; levels 4, 8 and 9 blow straight past it,
#: which is why they are in `StandOffSolver.skip_levels`.
STATE_CAP = 4_000_000


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """Stand Off's whole rule set, over integer cell indices.

    A state is ``(player, guns, mode, crates)``:

      * ``player`` -- cell index;
      * ``guns``   -- a 4-bit mask of DIRECTIONS, not of cells: bit ``d`` means a
        gun is standing in ``player + d``. That is exact, because a gun is drawn
        adjacent and then moves with the player on every press, so its offset
        never changes -- it is either beside you or destroyed. At most two bits
        are ever set (the game's own head count refuses a third);
      * ``mode``   -- the `GunMode` bit: 1 while the next arrow key will draw or
        holster instead of walking;
      * ``crates`` -- a bitmask of cells.

    Walls, windows, exits and bad guys are static (no rule creates or destroys
    any of them, and `BrokenWindow` is mechanically `Window`), so they live on
    the board rather than in the state, and the sight LINES are precomputed once
    as ``danger[cell][d]``: the mask of cells a crate would have to occupy to
    save you, or -1 when that ray reaches no bad guy at all.

    Not stored: `BrokenWindow` (see mechanic 5), `LoS` (cleared before the turn
    ends) and `Bullet`/`Corpse` (a bullet exists only inside the ``again`` loop
    that kills you, and a corpse only after it has).
    """

    def __init__(self, h, w, walls, windows, exits, badguys, start):
        self.h, self.w = h, w
        n = h * w
        self.walls, self.windows, self.exits, self.badguys = (
            walls, windows, exits, badguys)
        #: Everything static the PLAYER cannot walk onto: PuzzleScript's
        #: ``Obstacle`` minus Crate, which is dynamic.
        self.obstacle = walls | windows | badguys
        self.nbr = [[-1] * 4 for _ in range(n)]
        for r in range(h):
            for c in range(w):
                i = r * w + c
                for d, (dr, dc) in enumerate(_DELTA):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        self.nbr[i][d] = rr * w + cc
        # The sight line in each direction, stopped by the STATIC shrouds (wall,
        # exit) and by the board edge. A ray that reaches a bad guy is lethal
        # unless a gun sits in the first cell or a crate sits anywhere along it,
        # so what is stored is the crate mask that would save you.
        self.danger = [[-1] * 4 for _ in range(n)]
        for i in range(n):
            for d in range(4):
                mask, cur, hit = 0, i, False
                while True:
                    cur = self.nbr[cur][d]
                    if cur < 0 or cur in walls or cur in exits:
                        break
                    if cur in badguys:
                        hit = True
                        break
                    mask |= 1 << cur
                if hit:
                    self.danger[i][d] = mask
        self.start = start
        self.hfield = relaxed_field(self, crate_star(self, start[3]))

    @classmethod
    def read(cls, eng, ids) -> "_Board":
        """Build the model, and its start state, from the interpreter's grid."""
        wall, window, exit_, player, badguy, crate = ids
        w = eng.width
        walls, windows, exits, badguys = set(), set(), set(), set()
        cmask, p = 0, -1
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                i = r * w + c
                if cell & wall:
                    walls.add(i)
                if cell & window:
                    windows.add(i)
                if cell & exit_:
                    exits.add(i)
                if cell & badguy:
                    badguys.add(i)
                if cell & crate:
                    cmask |= 1 << i
                if cell & player:
                    p = i
        return cls(eng.height, w, walls, windows, exits, badguys,
                   (p, 0, 0, cmask))

    # -- dynamics -------------------------------------------------------------
    def win(self, state) -> bool:
        """``any Exit on Player``. Deliberately NOT conjoined with safety: see
        mechanic 1 -- the win flag is read before the shot lands."""
        return state[0] in self.exits

    def safe(self, p, guns, cm) -> bool:
        """Would the turn that left the player at ``p`` end with it alive?

        A ray is plugged by a gun in its first cell (bit ``d`` of ``guns``) or by
        a crate anywhere along it. Everything else on the ray -- window, broken
        window, corpse -- is see-through, which is the game.
        """
        dg = self.danger[p]
        for d in range(4):
            if guns >> d & 1:
                continue
            mask = dg[d]
            if mask >= 0 and not (mask & cm):
                return False
        return True

    def step(self, state, code):
        """One press. Returns ``(next, off_board)`` where ``next`` is

          * ``None``    -- the player was shot;
          * ``state``   -- the interpreter refused the turn (a wall, a jammed
            crate, a third gun): a genuine no-op, board and mode bit intact;
          * a new state -- otherwise.

        ``off_board`` flags the one configuration this model declines to
        predict: a press that would walk the player, or one of its guns, off the
        edge of the grid. PuzzleScript has no rule for it (``[ > Gun | Obstacle
        ]`` needs a cell to match against), so the outcome would be the
        interpreter's collision fallback rather than anything the .txt says.
        Every level here is walled on all four sides, so it never fires --
        `_report` asserts that rather than assuming it.
        """
        p, guns, gm, cm = state
        if code == _ACTION:
            ns = (p, guns, 1 - gm, cm)
            return (ns, False) if self.safe(p, guns, cm) else (None, False)
        d = code
        t = self.nbr[p][d]
        if t < 0:
            return state, True
        tbit = 1 << t
        if gm:
            if guns >> d & 1:                       # holster
                ng = guns & ~(1 << d)
            elif t in self.obstacle or (cm & tbit):
                return state, False                 # aimed into something solid
            elif bin(guns).count("1") >= 2:
                return state, False                 # the third gun: whole turn
            else:                                   # draw (and lose it at once
                ng = guns if t in self.exits else guns | (1 << d)   # on an exit)
            ns = (p, ng, 0, cm)
            return (ns, False) if self.safe(p, ng, cm) else (None, False)

        if t in self.obstacle:
            return state, False
        ncm = cm
        if cm & tbit:                               # push
            b = self.nbr[t][d]
            if b < 0 or b in self.obstacle or (cm >> b & 1):
                return state, False
            ncm = (cm & ~tbit) | (1 << b)
        # Every gun moves with the player, and dies if it moves into something.
        # The pushed crate is never a gun's destination -- a gun sits one cell
        # from the player and the crate's two cells are the player's own
        # destination and the one past it -- so the pre-push mask is the right
        # one to test against and no rule ordering is being papered over here.
        ng, off = 0, False
        for gd in range(4):
            if not (guns >> gd & 1):
                continue
            dst = self.nbr[self.nbr[p][gd]][d]
            if dst < 0:
                off = True
                continue
            if dst in self.obstacle or (cm >> dst & 1) or dst in self.exits:
                continue                            # destroyed (or eaten by the
            ng |= 1 << gd                           # late Gun-on-Exit rule)
        ns = (t, ng, 0, ncm)
        if self.win(ns):
            return ns, off
        return (ns, off) if self.safe(t, ng, ncm) else (None, off)

    # -- search ---------------------------------------------------------------
    def astar(self, cap: int = STATE_CAP):
        """The shortest press sequence to a win, as direction codes, or None.

        A* over the full state, with `hfield` as the heuristic. The win is
        returned when it is GENERATED rather than when it is popped, which is
        exact here: a popped node has ``g + h <= L``, and every non-win state
        has ``h >= 1``, so the win it generates costs ``g + 1 <= L``.
        """
        s0 = self.start
        if self.win(s0):
            return []
        h0 = self.hfield.get(s0[:3])
        if h0 is None:
            return None
        pq = [(h0, 0, 0, s0)]
        par: dict = {s0: None}
        best = {s0: 0}
        cnt = 0
        while pq:
            _f, g, _c, s = heapq.heappop(pq)
            if best.get(s, 1 << 30) < g:
                continue
            for code in range(5):
                nxt, off = self.step(s, code)
                if nxt is None or nxt == s or off:
                    continue
                if self.win(nxt):
                    path, k = [code], s
                    while par[k] is not None:
                        k, c = par[k]
                        path.append(c)
                    return list(reversed(path))
                hh = self.hfield.get(nxt[:3])
                if hh is None:
                    continue          # unreachable even in the relaxed game
                ng = g + 1
                if best.get(nxt, 1 << 30) <= ng:
                    continue
                best[nxt] = ng
                par[nxt] = (s, code)
                cnt += 1
                heapq.heappush(pq, (ng + hh, ng, cnt, nxt))
            if len(best) > cap:
                return None
        return None

    def cone(self, cost: int, cap: int = STATE_CAP):
        """``{state: presses to the win}``, exact for every state on an optimal
        path, given the optimal cost ``cost``.

        Forward Dijkstra keeping only states with ``g + h <= cost`` and recording
        predecessors, then a backward BFS from the wins it found. That is exact
        where it matters and the argument is short: a state ``s`` on an optimal
        path satisfies ``g(s) + d(s) == cost``, so its whole remaining optimal
        path is inside the cone, and the BFS therefore returns ``d(s)`` itself.
        For a successor OFF the optimal path the BFS can only over-report, which
        is all `plan` needs -- it asks whether ``d(succ) == d(s) - 1``, and a
        successor for which that holds is on an optimal path by definition and so
        is measured exactly.

        This is what `PSExpert.optimal_sets` buys with ~5 re-solves per plan
        step; here it is one extra search for the whole level.
        """
        s0 = self.start
        pred: dict = {}
        wins = set()
        best = {s0: 0}
        pq = [(self.hfield[s0[:3]], 0, 0, s0)]
        cnt = 0
        while pq:
            _f, g, _c, s = heapq.heappop(pq)
            if best.get(s, 1 << 30) < g:
                continue
            for code in range(5):
                nxt, off = self.step(s, code)
                if nxt is None or nxt == s or off:
                    continue
                ng = g + 1
                if self.win(nxt):
                    if ng <= cost:
                        pred.setdefault(nxt, set()).add(s)
                        wins.add(nxt)
                    continue
                hh = self.hfield.get(nxt[:3])
                if hh is None or ng + hh > cost:
                    continue
                pred.setdefault(nxt, set()).add(s)
                if best.get(nxt, 1 << 30) <= ng:
                    continue
                best[nxt] = ng
                cnt += 1
                heapq.heappush(pq, (ng + hh, ng, cnt, nxt))
            if len(best) > cap:
                return None
        dist = {w: 0 for w in wins}
        queue = deque(wins)
        while queue:
            k = queue.popleft()
            for pv in pred.get(k, ()):
                if pv not in dist:
                    dist[pv] = dist[k] + 1
                    queue.append(pv)
        return dist

    def plan(self):
        """``(presses, optsets)`` -- the shortest press sequence from the level
        start with the EXACT set of equally-shortest presses at every step, or
        ``(None, None)`` when no plan was found inside the budget.

        Ties are broken by `_ORDER`, which is what makes a re-derived plan
        byte-identical across processes and therefore makes the disk cache safe.
        """
        route = self.astar()
        if route is None:
            return None, None
        dist = self.cone(len(route))
        if dist is None or self.start not in dist:
            return None, None
        s, presses, optsets = self.start, [], []
        while dist[s]:
            want = dist[s] - 1
            best = []
            for code in range(5):
                nxt, off = self.step(s, code)
                if nxt is None or nxt == s or off:
                    continue
                if dist.get(nxt, -1) == want:
                    best.append(code)
            presses.append(_ORDER[best[0]])
            optsets.append([_ORDER[c] for c in best])
            s = self.step(s, best[0])[0]
        return presses, optsets


# ---------------------------------------------------------------------------
# The heuristic: an exact field over a crate-relaxed abstraction
# ---------------------------------------------------------------------------

def crate_star(board: _Board, cm: int) -> int:
    """Over-approximate the cells a crate can EVER stand on.

    Push reachability with the other crates ignored: a crate at ``x`` reaches
    ``x + d`` when that cell is free of static obstacles and the player has a
    free cell at ``x - d`` to shove from. Ignoring the other crates can only
    ADD cells, which is the direction this has to err in -- `relaxed_field`
    hands out a shield wherever this says a crate could be, and a shield the
    real game cannot supply would make the heuristic too small, never too big.
    """
    seen = {i for i in range(board.h * board.w) if cm >> i & 1}
    queue = deque(seen)
    while queue:
        x = queue.popleft()
        for d in range(4):
            y = board.nbr[x][d]
            stand = board.nbr[x][_OPP[d]]
            if (y < 0 or stand < 0 or y in board.obstacle
                    or stand in board.obstacle or y in seen):
                continue
            seen.add(y)
            queue.append(y)
    mask = 0
    for i in seen:
        mask |= 1 << i
    return mask


def relaxed_field(board: _Board, cstar: int) -> dict:
    """``{(player, guns, mode): presses to the win}`` in the RELAXED game R,
    which drops crates from the state entirely.

    In R a crate is never an obstacle and never cargo; instead every cell in
    ``cstar`` counts as permanently shielded. That is a lower bound on the real
    game, and the argument is a mimicking one -- take any real winning press
    sequence and follow it in R:

      * a press the real game REFUSES is skipped in R (R is free to not press
        it), so R's player stays where the real one is;
      * a press that walks (pushing a crate or not) walks in R to the same cell,
        because R has no crate obstacles;
      * a press that draws or holsters does the same in R, because R's mode
        rules are the game's own;
      * the guns are the one place the two can disagree: the real game destroys
        a gun that walks into a crate and R does not. Holding a gun you did not
        want is not free (a third draw is refused, so shedding it costs two
        presses), so R would be over-charged. Hence a move in R may destroy ANY
        SUBSET of the guns it could keep -- the branching in `_succ` below -- and
        R can always hold exactly what the real game holds.

      * finally R's safety test is weaker than the real one everywhere (the real
        crates are always inside ``cstar``), so every state the real sequence
        passes through is legal in R.

    So R has a winning sequence no longer than the real one, i.e. ``d_R <= d``.
    The space is ``cells x 11 gun sets x 2``, a couple of thousand states, so it
    is enumerated exactly by backward BFS rather than searched.
    """
    n = board.h * board.w

    def rsafe(p, guns):
        dg = board.danger[p]
        for d in range(4):
            if guns >> d & 1:
                continue
            mask = dg[d]
            if mask >= 0 and not (mask & cstar):
                return False
        return True

    def _succ(p, guns, gm):
        out = [(p, guns, 1 - gm)]
        for d in range(4):
            t = board.nbr[p][d]
            if t < 0:
                continue
            if gm:
                if guns >> d & 1:
                    out.append((p, guns & ~(1 << d), 0))
                elif t in board.obstacle or bin(guns).count("1") >= 2:
                    continue
                else:
                    out.append((p, guns if t in board.exits
                                else guns | (1 << d), 0))
                continue
            if t in board.obstacle:
                continue
            keep = 0
            for gd in range(4):
                if not (guns >> gd & 1):
                    continue
                dst = board.nbr[board.nbr[p][gd]][d]
                if dst < 0 or dst in board.obstacle or dst in board.exits:
                    continue
                keep |= 1 << gd
            sub = keep                      # any subset of the survivors
            while True:
                out.append((t, sub, 0))
                if not sub:
                    break
                sub = (sub - 1) & keep
        return [s for s in out if s[0] in board.exits or rsafe(s[0], s[1])]

    gunsets = [g for g in range(16) if bin(g).count("1") <= 2]
    pred: dict = {}
    wins = []
    for p in range(n):
        if p in board.obstacle:
            continue
        for guns in gunsets:
            # A gun set is only meaningful when each of its cells exists and can
            # actually hold a gun; an impossible set would otherwise be handed a
            # distance and let A* charge a state that cannot exist.
            if any((board.nbr[p][gd] < 0
                    or board.nbr[p][gd] in board.obstacle
                    or board.nbr[p][gd] in board.exits)
                   for gd in range(4) if guns >> gd & 1):
                continue
            for gm in (0, 1):
                if p in board.exits:
                    wins.append((p, guns, gm))
                    continue
                if not rsafe(p, guns):
                    continue
                for nxt in _succ(p, guns, gm):
                    pred.setdefault(nxt, []).append((p, guns, gm))
    dist = {w: 0 for w in wins}
    queue = deque(wins)
    while queue:
        k = queue.popleft()
        for pv in pred.get(k, ()):
            if pv not in dist:
                dist[pv] = dist[k] + 1
                queue.append(pv)
    return dist


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class StandOffExpert(PSExpert):
    """Plans by building `_Board` off the engine grid and searching THAT; never
    steps the interpreter.

    `PSExpert` still owns everything around that -- the in-memory memo, the
    on-disk plan cache with its staleness check, the level scoping and the
    snapshot/restore discipline -- so the only override is `_search`. The
    interpreter's own `step` is 79-223 presses/s on these boards against the
    model's 0.7-1.4M (`--speed`), which is the whole reason the search is not an
    engine-blackbox one; the interpreter's job here is to CERTIFY the plan
    (`--plans`) and to render the frames.
    """

    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        g = self.g
        name = g.resolve_object_name
        self.wall_ids = set(name("wall"))
        # BrokenWindow is Window for every rule that reads either (mechanic 5),
        # so the model is told they are the same thing.
        self.window_ids = set(name("window")) | set(name("brokenwindow"))
        self.exit_ids = set(name("exit"))
        self.player_ids = set(self.game._engine._player_indices)
        self.badguy_ids = set(name("badguy"))
        self.crate_ids = set(name("crate"))
        self.gunmode_ids = set(name("gunmode"))
        self.gun_ids = tuple(g.obj_name_to_idx["gun" + d[0]] for d in _ORDER[:4])

    def heuristic(self, eng) -> int:
        raise AssertionError("the field is exact; no engine-blackbox A* runs here")

    def board(self, eng) -> _Board:
        return _Board.read(eng, (self.wall_ids, self.window_ids, self.exit_ids,
                                 self.player_ids, self.badguy_ids,
                                 self.crate_ids))

    def _search(self, eng):
        presses, optsets = self.board(eng).plan()
        return None if presses is None else Plan(presses, optsets)


class StandOffSolver(PSAStarSolver):
    game_id = "puzzlescript_stand_off"
    game_name = GAME_NAME
    expert_cls = StandOffExpert

    #: Levels 4, 8 and 9 are winnable but out of reach of a crate-blind search;
    #: see the module docstring. Named here so `discover_solvable` does not spend
    #: `STATE_CAP` states on each of them at every startup.
    skip_levels = frozenset({4, 8, 9})

    #: The longest plan recorded is 114 presses (level 7); the rest is room for
    #: the exploration prefix and the re-plan after it.
    max_steps = 400

    def prepare_expert(self, game, expert) -> None:
        """Build every level's plan before `discover_solvable` asks for it --
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
    solver = StandOffSolver()
    game = solver.make_game(0)
    return solver, game, StandOffExpert(game)


def _engine_state(eng, expert):
    """The interpreter's grid in `_Board`'s state encoding, or None once the
    player is a corpse.

    The gun offsets are read back RELATIVE to the player, which is also the
    check that the model's central invariant holds on the interpreter: a gun
    that is not beside the player would come back as ``None`` and fail the fuzz
    loudly rather than being silently rounded to the nearest direction.
    """
    w = eng.width
    p, guns, gm, cm = -1, 0, 0, 0
    found: list[tuple[int, int]] = []
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            if cell & expert.player_ids:
                p = i
            if cell & expert.gunmode_ids:
                gm = 1
            if cell & expert.crate_ids:
                cm |= 1 << i
            for d, oid in enumerate(expert.gun_ids):
                if oid in cell:
                    found.append((d, i))
    if p < 0:
        return None
    for d, i in found:
        dr, dc = _DELTA[d]
        if i != p + dr * w + dc:
            return "DETACHED GUN"
        guns |= 1 << d
    return (p, guns, gm, cm)


def _report() -> int:
    """Per-level piece counts, search size, plan length and tie coverage -- and
    CERTIFY each plan by replaying it on the interpreter."""
    _solver, game, expert = _levels()
    eng = game._engine
    bad = total = labelled = forced = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        skipped = level in StandOffSolver.skip_levels
        t = time.time()
        presses, optsets = (None, None) if skipped else board.plan()
        took = time.time() - t
        crates = bin(board.start[3]).count("1")
        head = (f"level {level:2d}: {len(board.badguys):2d} bad guys, "
                f"{crates:2d} crates, {len(board.exits):2d} exits, "
                f"h0={board.hfield.get(board.start[:3])}")
        if presses is None:
            print(f"{head}: {'SKIPPED (see skip_levels)' if skipped else 'NO PLAN'}")
            bad += not skipped
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
        labelled += sum(len(s) for s in optsets)
        forced += sum(len(s) == 1 for s in optsets)
        verdict = ("interpreter WINS on the last press" if ok
                   else f"CERTIFICATION FAILED (won at {won_at})")
        print(f"{head}, {len(presses):3d} presses in {took:5.2f}s, "
              f"{sum(len(s) for s in optsets) / len(presses):.2f} optimal "
              f"presses/step: {verdict}")
    print(f"{total} presses over the recorded levels, {labelled} labels "
          f"({forced} forced): {'all plans certified' if not bad else f'{bad} BAD'}")
    return 0 if not bad else 1


def _fuzz(trials: int = 6, walk: int = 12, only=None) -> int:
    """Assert `_Board` reproduces the interpreter exactly.

    Random play from the level START is not enough: it dies within a few presses
    (this is a game where one wrong step is fatal), so the gun choreography and
    the crate pushes the plans are made of would go unexercised. Fuzz from EVERY
    PREFIX of each level's own solution instead, which puts the model in the
    configurations the plans actually visit -- including the skipped levels,
    whose starts still exercise the crate rules even though no plan is recorded
    from them.

    Coverage counters are printed for the same reason: a run reporting agreement
    while having drawn no gun and pushed no crate has checked nothing.

    The trial counts are small because the INTERPRETER is the cost (79-223
    presses/s, see `_speed`) and the prefix sweep already multiplies them by the
    plan length: level 7 alone is 115 prefixes.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    rng = random.Random(20260819)
    bad = 0
    tot = dict(presses=0, refused=0, draws=0, holsters=0, pushes=0,
               deaths=0, wins=0, modes=0)
    for level in range(game.n_levels):
        if only and level not in only:
            continue
        game.set_level(level)
        layout = g.levels[level]
        board = expert.board(eng)
        presses, _optsets = board.plan()
        presses = presses or []
        seen = dict(tot)
        for prefix in range(len(presses) + 1):
            for _ in range(trials):
                eng.load_level(layout)
                state = board.start
                for direction in presses[:prefix]:
                    eng.step(direction)
                    state = board.step(state, _ORDER.index(direction))[0]
                if board.win(state):
                    continue          # the level is over; see mechanic 6
                for _ in range(walk):
                    code = rng.randrange(5)
                    before = state
                    nxt, off = board.step(state, code)
                    if off:
                        break                    # not a transition the model
                    eng.step(_ORDER[code])       # claims to predict
                    tot["presses"] += 1
                    theirs = _engine_state(eng, expert)
                    if nxt is None:
                        ok = theirs is None
                        tot["deaths"] += 1
                    else:
                        ok = nxt == theirs
                    if not ok:
                        bad += 1
                        if bad <= 3:
                            print(f"  level {level} MISMATCH after "
                                  f"{_ORDER[code]}\n    model  {nxt}"
                                  f"\n    engine {theirs}")
                        break
                    if nxt is None:
                        break
                    if board.win(nxt) != eng.check_win():
                        bad += 1
                        cell = sorted(eng.grid[nxt[0] // eng.width]
                                      [nxt[0] % eng.width])
                        print(f"  level {level} WIN MISMATCH after "
                              f"{_ORDER[code]}: model {board.win(nxt)}, "
                              f"interpreter {eng.check_win()}\n"
                              f"    state {nxt}\n    cell  {cell}")
                        break
                    if nxt == before:
                        tot["refused"] += 1
                    tot["modes"] += nxt[2] != before[2]
                    gb, nb = bin(before[1]).count("1"), bin(nxt[1]).count("1")
                    tot["draws"] += nb > gb
                    tot["holsters"] += nb < gb
                    tot["pushes"] += nxt[3] != before[3]
                    state = nxt
                    if board.win(state):
                        tot["wins"] += 1
                        break
        print(f"level {level:2d}: {tot['presses'] - seen['presses']:6d} presses, "
              f"{tot['draws'] - seen['draws']:4d} draws, "
              f"{tot['holsters'] - seen['holsters']:4d} holsters, "
              f"{tot['pushes'] - seen['pushes']:5d} pushes, "
              f"{tot['refused'] - seen['refused']:5d} refused, "
              f"{tot['deaths'] - seen['deaths']:5d} deaths, "
              f"{tot['wins'] - seen['wins']:3d} wins", flush=True)
    print(f"{tot['presses']} transitions ({tot['draws']} draws, "
          f"{tot['holsters']} holsters, {tot['pushes']} crate pushes, "
          f"{tot['modes']} mode flips, {tot['deaths']} fatal, "
          f"{tot['refused']} refused): "
          f"{'model matches the interpreter' if not bad else f'{bad} MISMATCHES'}")
    return 0 if not bad else 1


def _replay(seeds: int = 3) -> int:
    """Record a few seeds and REPLAY their recorded screen actions through a
    fresh adapter, comparing every frame.

    This is the check for the trap the ps: family is built around
    (`ps_astar.screen_action`): the adapter FORWARD-remaps directional input by
    the seed's rotation, so an expert planning in engine space has to emit the
    INVERSE. Get it backwards and nothing raises -- the episodes still win,
    because the recorder re-plans from whatever state the wrong press produced;
    they are just slower and are labelled with presses that do not reproduce
    them. So the assertion here is frame equality, not the win.

    Every seed of these ten levels lands on one of the four rotations, and the
    report prints which, so a clean run has covered all of them.
    """
    solver = StandOffSolver()
    bad = 0
    for seed in range(seeds):
        ok_all, levels = solver.solve_episode(seed, explore=True)
        game = solver.make_game(seed)
        for lvl in levels:
            game.set_level(lvl["level_id"])
            frames = [np.asarray(game._current_frame)]
            for act in lvl["actions"][1:]:
                gact = _ID_TO_GAMEACTION[act["index"]]
                if gact.name == "RESET":
                    game.set_level(lvl["level_id"])
                    frames.append(np.asarray(game._current_frame))
                    continue
                fd = game.perform_action(ActionInput(id=gact))
                frames.append(np.asarray(fd.frame[-1] if fd.frame
                                         else game._current_frame))
            rec = [np.asarray(o, dtype=np.uint8) for o in lvl["observations"]]
            exact = (len(rec) == len(frames)
                     and all(np.array_equal(a, b) for a, b in zip(rec, frames)))
            won = game._state == GameState.WIN
            bad += not (exact and won)
            print(f"seed {seed} level {lvl['level_id']:2d}: {len(rec):3d} frames, "
                  f"rotation k={game._rotation_k}: "
                  f"{'replay EXACT' if exact else 'REPLAY DIVERGED'}, "
                  f"{'WIN' if won else 'NOT WON'}")
        if not ok_all:
            print(f"seed {seed}: not every solvable level won")
            bad += 1
    print("rotation contract holds" if not bad else f"{bad} FAILURES")
    return 0 if not bad else 1


def _speed(n: int = 300) -> int:
    """Interpreter presses/s against `_Board` presses/s, per level.

    This is the measurement the choice of expert rests on (see the
    entrepotphage rule in the ps: A* family notes: MEASURE ``eng.step`` before
    reaching for an engine-blackbox search), so it is a report rather than a
    number in a comment.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    order = list(_ORDER)
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        layout = game._game.levels[level]
        t = time.time()
        for i in range(n):
            if i % 25 == 0:
                eng.load_level(layout)
            eng.step(order[i % 5])
        engine_rate = n / (time.time() - t)
        state = board.start
        t = time.time()
        for i in range(n * 50):
            nxt, _off = board.step(state, i % 5)
            if nxt is None or board.win(nxt):
                state = board.start
            else:
                state = nxt
        model_rate = n * 50 / (time.time() - t)
        print(f"level {level:2d}: interpreter {engine_rate:8.0f} presses/s, "
              f"model {model_rate:9.0f} presses/s  ({model_rate / engine_rate:5.0f}x)")
    return 0


#: Every cell stack these ten levels can present. Bullets and corpses are in the
#: list because they are not only death frames: the corpse is what the
#: exploration prefix walks into, and a bullet is on screen in the WINNING frame
#: of a level whose exit is itself in a bad guy's line (mechanic 1).
#: ``player_on_exit`` is the win frame and is the one to check first.
_COMPOSITIONS: dict[str, tuple[str, ...]] = {
    "floor": (),
    "wall": ("wall",),
    "window": ("window",),
    "brokenwindow": ("brokenwindow",),
    "exit": ("exit",),
    "player": ("player",),
    "player_gunmode": ("player", "gunmode"),
    "player_on_exit": ("exit", "player"),
    "badguy": ("badguy",),
    "crate": ("crate",),
    "crate_on_exit": ("exit", "crate"),
    "gunu": ("gunu",), "gund": ("gund",),
    "gunl": ("gunl",), "gunr": ("gunr",),
    "corpse": ("corpse",),
    "bullet": ("bullet",),
    "bullet_on_brokenwindow": ("brokenwindow", "bullet"),
}


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor).

    This is the regression test for the rendering fixes documented at the top of
    ``data/puzzlescript_games/Stand_Off.txt``; before them it reported the mode
    bit, the bad guys, all four guns and the bullet as invisible.
    """
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
        for nm, objs in _COMPOSITIONS.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[nm] = np.asarray(_render_frame(eng, g)).copy()
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
    if "--fuzz" in sys.argv:
        rest = sys.argv[sys.argv.index("--fuzz") + 1:]
        sys.exit(_fuzz(only={int(x) for x in rest if x.isdigit()}))
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--replay" in sys.argv:
        sys.exit(_replay())
    if "--speed" in sys.argv:
        sys.exit(_speed())
    sys.exit(StandOffSolver.main())
