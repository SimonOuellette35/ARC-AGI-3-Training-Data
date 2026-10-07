"""ps:the_trouble_with_toasters -- a match-3 sokoban you play against a CLOCK:
the spawners make a new toaster roughly every other turn, for as long as the
game lasts, and the win condition is that there are none.

THE WHOLE GAME IS SEVEN RULES, and every one of them matters

    up [ ToasterGestating no Obstacle ] -> [ Toaster ]                (1)
    up [ Spawner no Obstacle ] -> [ Spawner random ToasterGestating ] (2)
    [ > Player | Toaster ] -> [ > Player | > Toaster ]                (3)
    [ > Player | Crate   ] -> [ > Player | > Crate   ]                (4)
    [ < Player | Crate   ] -> [ < Player | < Crate   ]                (5)
    late [ Crate ToasterGestating ] -> [ Crate ]                      (6)
    late [ Toaster | Toaster | Toaster ] -> [ | | ]                   (7)

with `Obstacle = Player or Toaster` and the win `No Toaster`. Five 7x7 levels,
no walls anywhere -- the board edge is the only thing you cannot walk through
-- and no way to lose: no lose condition, no restart rule, nothing lethal.

SIX CONSEQUENCES, none of them visible in the rules as written

* **ACTION is the WAIT button, not a no-op.** No rule reads the action key, so
  pressing X moves nothing -- but a press is a TURN, and rules 1 and 2 fire on
  every turn regardless. Standing still is therefore a real move, and the
  search branches on five presses rather than four. It is used once in the 91:
  step 12 of level 4, where the certified optimal set is `['action']` and
  nothing else -- the only right answer there is to stand still and let a
  toaster hatch.

* **Standing on a spawner switches it off, on the turn you ARRIVE.** `Obstacle`
  is Player or Toaster, and rules 1 and 2 both read the board BEFORE the player
  moves -- so stepping onto a spawner suppresses that turn's spawn, and
  stepping OFF one does not release it until the turn after. `selfcheck`
  measures the ordering the other way round: a model that re-arms the spawners
  before the hatchlings hatch is wrong on 2081 of 4000 random boards, because
  hatching first is exactly what makes a spawner block itself -- the toaster
  that just appeared on it is the Obstacle that stops rule 2 the same turn.

* **A toaster on a spawner buries it, and the burial is invisible.** Push a
  toaster onto a spawner and the spawner stops; push it off and the spawner
  resumes -- one turn later, or immediately if a hatchling was already sitting
  under the toaster when it arrived (rule 2 fires before a pushed toaster lands
  on the cell). Both of those states rendered as a plain toaster until the
  sprite fix; see `audit` and the header of
  `data/puzzlescript_games/The_Trouble_with_Toasters.txt`.

* **A crate on a spawner switches it off FOREVER.** A Crate is not an
  `Obstacle`, so rule 2 keeps firing under it -- and rule 6 kills the hatchling
  every single turn before it can hatch. That is the game's only permanent
  answer to a spawner, it is what level 4 means by "A crate is better than no
  crate", and it is why the crates are worth the presses it costs to drag them:
  rule 5 PULLS a crate (the only thing in the game that can be pulled -- a
  toaster cannot), so a crate can be brought back out of a corner.

* **A whole maximal RUN dies, not three of it, and the VERTICAL axis goes
  first.** Rule 7 deletes three, but the interpreter re-scans until no match is
  left and every overlapping window of a longer line matches, so a run of four,
  five or six vanishes entirely; and the rule is written without a direction,
  so it expands to all four and they are driven to a fixpoint in the order up,
  down, left, right -- columns before rows. On a plus of toasters the column
  dies and the two horizontal arms SURVIVE, two cells apart. `selfcheck`
  measures both: the three-of-a-run model is wrong on 227 of 4000 random
  boards, the rows-first model on 175.

* **A jammed push cancels the pull too.** Rules 3, 4 and 5 all fire on the same
  press, but a push into the board edge or into a second piece leaves the
  player standing still -- and then the crate behind it has nowhere to go
  either, so the whole turn is a no-op. There is no chain push: a toaster
  shoved into a toaster does not move, and neither do you.

THE SEARCH: the exact distance FIELD on four levels, a beam on the fifth

A state is `(player, toasters, crates, gestating)` -- a cell index and three
bitmasks -- which is exact and complete, because the spawners are static and
there is nothing else in the game. `ToasterBoard.field` enumerates the whole
reachable component layer by layer and then induces backwards over the layers,
which buys three things a heuristic search cannot have: a plan that is provably
SHORTEST with no weight, heuristic or node cap to get wrong; the EXACT optimal
press set at every step (`dist_to_win(successor) == dist - 1`, read straight off
the layers); and, for a level that had no plan, a proof rather than a shrug.
Levels 0-3 cost 1.5k, 34k, 293k and 1.24M states, and 0.01 s to 4 s.

Level 4 is out of reach of that and of everything like it: its space is
**39,153,617 states within 28 presses**, growing by a factor of 1.5 per press,
and its answer is 30. So it is planned by a globally-deduplicated BEAM (width
60k and 250k, shortest kept) and then handed to `certify_tail`, which walks the
plan backwards asking two exhaustive, bounded, exact questions at every state
-- is there a shorter win from here (splice it in), and which presses still
finish in the presses that are left (that is the step's optimal set). The
answers are exact where the probe budget reaches, which is the last 21 of the
30 steps; the first 9 carry the plan's own press and nothing else, a
conservative label rather than a wrong one -- and not one of the 21 certified
steps has a second optimal press, so what those 9 omit is very probably
nothing.

LEVEL 4'S PLAN IS PROVED SHORTEST ANYWAY, from the other side. `--bfs 4
--max-depth=29` enumerates the level's whole reachable space out to 28 presses
(39.15M states, 4 GB, 280 s) and then sweeps the 13.1M states of that last
layer for a winning press WITHOUT recording anything -- so "is there a win at
29?" costs the memory that depth 28 already costs. There is not. No win exists
in 29 presses, the beam's plan is 30, and 30 is therefore optimal. All five
levels are shortest.

RECOVERY is the family's RESET prefix (`recovery_mode = "reset"`). Nothing here
is reversible -- an annihilated toaster does not come back, a toaster shoved
against the edge can only be pushed further into it (there is no pull for a
toaster), and every turn spent flailing is a turn the spawners keep -- so a
perturbed board cannot be re-planned from in general, and level 4's search is
minutes rather than milliseconds in any case. One RESET restores exactly the
state the cached plan was solved from. The plans live in
`data/the_trouble_with_toasters_plans.json`.

ART. Four sprites in `data/puzzlescript_games/The_Trouble_with_Toasters.txt`.
Toaster was opaque on every pixel Spawner and ToasterGestating draw, and Crate
was opaque everywhere, so `toaster`, `toaster on spawner` and `toaster on
spawner + gestating` were ONE picture and so were `crate` and `crate on
spawner` -- i.e. every fact about the spawners in the four bullet points above
was unobservable the moment anything stood on them. Spawner and
ToasterGestating each grew four corner dots in their own colour and Toaster and
Crate had their corners punched transparent. The file's own header has the
detail; `--audit` is the regression test.

VERIFIED. All five levels, 91 presses, every one of them provably shortest.
`--bfs` re-derives that with an independent flat BFS that shares nothing with
the field but the model: 1556, 33948, 394287 and 1271317 reachable states on
levels 0-3 and shortest wins of 11, 15, 17 and 18 presses, exactly the plans,
plus the 39.15M-state lower bound that settles level 4. `--selfcheck` fuzzes
27000 presses over the five levels (4515 exact no-ops, 21926 player moves, 1353
toasters gestated, 212 annihilated) plus 4000 random boards (2590 of them with
a deletion) with 0 divergences from the interpreter, and reports the three
wrong-model counts quoted above. `--ties` re-derives 314 labelled presses with
a depth-first bounded search -- every step of levels 0-3 and 18 of level 4's 30
-- with 0 disagreements in either direction. `--audit`: 11 cell compositions
pairwise distinct at cell_px 9, the only size the five 7x7 boards use.
`--symmetry`: 5 levels x 16 presentations, every plan a WIN and every frame of
both the plan and a 200-press random walk exactly the transform of the
unaugmented one; the one chiral thing in the mechanic (columns settling before
rows) decides the outcome on 139 of 4000 random toaster fields and on 0 of the
91 boards the plans pass through, and is split by a quarter turn and never by a
mirror, which is why the flips are free. 40 seeds x 5 levels recorded
in-process: 200/200 WIN, all 4256 frames replay exactly through the adapter
from the recorded SCREEN actions, 3640/3640 expert steps labelled (1.10 optimal
presses per step, the taken press always in its own set), indices in 0..6, and
three processes at different PYTHONHASHSEEDs write byte-identical episodes.
Longest level record is 31 actions, against the adapter's 200-press cap.

CLI
---
    --selfcheck   the native model fuzzed against the interpreter
    --audit       every cell composition renders distinctly, at every cell size
    --symmetry    every presentation is an exact transform of the unaugmented
    --plans       every level's plan, replayed through the interpreter
    --ties        every optimal-action label re-derived, depth-first
                  (`--max-remaining=N` bounds how deep it will look)
    --bfs         exhaustive flat BFS over a whole state space -- the
                  independent optimality proof. `--bfs 4 --max-depth=29
                  --cap=100000000` is level 4's, and wants ~4 GB and 5 minutes.
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (        # noqa: E402
    PuzzleScriptAdapter, _render_frame)
from arcengine import ActionInput, GameState        # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION   # noqa: E402
from solvers.common.ps_astar import (              # noqa: E402
    PSAStarSolver, PSExpert, Plan, screen_action)
from utils.rotation import inverse_remap_action_full  # noqa: E402

GAME_NAME = "The_Trouble_with_Toasters"
GAME_ID = "ps:the_trouble_with_toasters"

#: Engine presses, in the order the model indexes them. ACTION is NOT a no-op
#: here even though no rule reads it: a press is a TURN, and the spawners tick
#: on every turn, so ACTION is the game's wait button.
DIRNAMES = ("up", "down", "left", "right", "action")
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))
_OPP = (1, 0, 3, 2)

#: A run this long or longer annihilates -- and, as in every PuzzleScript
#: match-3, the WHOLE maximal run dies, not this many of it.
MATCH = 3

_INF = 1 << 30

#: The object names the model reads off the interpreter's grid.
OBJECTS = ("player", "toaster", "toastergestating", "crate", "spawner")


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class ToasterBoard:
    """One level, as a model the search can step in microseconds.

    A state is ``(player, toasters, crates, gestating)`` -- a flat cell index
    and three cell BITMASKS. That is exact and complete: the spawners are
    static, there are no walls at all (the board edge is the only obstacle),
    and nothing else in the game exists. `selfcheck` is the proof, fuzzed
    against the real interpreter.

    THE TURN, in the interpreter's own rule order:

    1. every ToasterGestating whose cell holds no Obstacle (Player or Toaster)
       becomes a Toaster;
    2. every Spawner whose cell holds no Obstacle spawns a ToasterGestating --
       and step 1 runs FIRST, so a gestating that just hatched is itself the
       obstacle that stops its own spawner re-arming this turn;
    3. the player moves, pushing the one Toaster or Crate in front of it and
       PULLING the one Crate behind it (there is no pull for a toaster). A
       blocked push cancels the whole turn's movement, the pull included;
    4. LATE: a Crate deletes the ToasterGestating sharing its cell -- which is
       why a crate parked on a spawner switches that spawner off forever;
    5. LATE: every maximal run of >= 3 Toasters is deleted, the VERTICAL axis
       before the horizontal.

    Both steps 1 and 2 read the board BEFORE the player moves, which is what
    makes standing on a spawner a real move: it stops the spawn on the turn you
    arrive, not on the turn after.
    """

    def __init__(self, eng, ids):
        grid = eng.grid
        self.h, self.w = len(grid), len(grid[0])
        n = self.h * self.w
        self.n = n
        player, toasters, crates, gestating, spawn = 0, 0, 0, 0, 0
        for r in range(self.h):
            for c in range(self.w):
                cell, i = grid[r][c], r * self.w + c
                if ids["player"] in cell:
                    player = i
                if ids["toaster"] in cell:
                    toasters |= 1 << i
                if ids["crate"] in cell:
                    crates |= 1 << i
                if ids["toastergestating"] in cell:
                    gestating |= 1 << i
                if ids["spawner"] in cell:
                    spawn |= 1 << i
        self.spawn_mask = spawn
        self.state = (player, toasters, crates, gestating)

        # Neighbour table. ``-1`` off the board -- the levels have no walls, so
        # "is it free?" and "is it on the board?" are the only two questions.
        self.nb = [[-1] * 4 for _ in range(n)]
        for r in range(self.h):
            for c in range(self.w):
                i = r * self.w + c
                for k, (dr, dc) in enumerate(_DELTA):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < self.h and 0 <= cc < self.w:
                        self.nb[i][k] = rr * self.w + cc
        self.rc = [(i // self.w, i % self.w) for i in range(n)]

        # Horizontal runs are found by shifting the whole board sideways, so
        # they need a mask that drops the cells whose window would wrap onto
        # the next row. Vertical runs need none: a shift by a whole row stays
        # in its column, and a window that runs off the bottom shifts in
        # zeroes. See `settle`.
        self.hmask = 0
        for r in range(self.h):
            for c in range(self.w - MATCH + 1):
                self.hmask |= 1 << (r * self.w + c)

    # -- dynamics -----------------------------------------------------------
    def settle(self, toasters: int) -> int:
        """The match-3 LATE rule, run to fixpoint: delete every maximal run of
        >= MATCH toasters, the VERTICAL axis before the horizontal.

        The axis order is not a preference. The rule is written without a
        direction, so the interpreter expands it to all four and drives each
        expansion to a fixpoint in the order up, down, left, right -- the two
        vertical ones first. On a plus of toasters that means the column dies
        and the two horizontal arms survive, two cells apart. `selfcheck`
        measures it: a model resolving rows first is wrong on 175 of 4000
        random boards.

        Two passes are enough and a third would change nothing: the first
        leaves no vertical run of >= MATCH, the second only removes toasters,
        and removing toasters can never lengthen a run.

        WHOLE runs die, not MATCH of them -- a line of four, five or six
        vanishes entirely, because every overlapping window of it matches and
        the interpreter re-scans until none is left. That is exactly what the
        shift-and-AND below computes: ``heads`` is every cell that starts a
        window, and OR-ing MATCH shifted copies of it back covers the run end
        to end.
        """
        for shift, mask in ((self.w, -1), (1, self.hmask)):
            heads = toasters & mask
            for j in range(1, MATCH):
                heads &= toasters >> (j * shift)
            if heads:
                doomed = 0
                for j in range(MATCH):
                    doomed |= heads << (j * shift)
                toasters &= ~doomed
        return toasters

    def step(self, st, k: int):
        """One press. Returns the SAME tuple value when the press changes
        nothing at all -- which for this game means a move into the board edge
        or into a jammed push, with every spawner already blocked."""
        p, toasters, crates, gestating = st

        # 1. gestating -> toaster, where no Player and no Toaster is in the way
        obstacle = toasters | (1 << p)
        hatched = gestating & ~obstacle
        toasters |= hatched
        gestating &= obstacle

        # 2. spawn -- reading the board step 1 has already updated
        gestating |= self.spawn_mask & ~(toasters | (1 << p))

        # 3. the player's move
        if k < 4:
            tgt = self.nb[p][k]
            if tgt >= 0:
                occupied = toasters | crates
                moved = True
                if (occupied >> tgt) & 1:
                    beyond = self.nb[tgt][k]
                    if beyond >= 0 and not (occupied >> beyond) & 1:
                        if (toasters >> tgt) & 1:
                            toasters = (toasters & ~(1 << tgt)) | (1 << beyond)
                        else:
                            crates = (crates & ~(1 << tgt)) | (1 << beyond)
                    else:
                        moved = False        # jammed: nothing moves, pull too
                if moved:
                    back = self.nb[p][_OPP[k]]
                    if back >= 0 and (crates >> back) & 1:
                        crates = (crates & ~(1 << back)) | (1 << p)
                    p = tgt

        # 4. LATE: a crate kills the gestating toaster under it
        gestating &= ~crates

        # 5. LATE: match 3. Unconditional, exactly as the interpreter runs it
        #    -- an early version skipped it whenever no toaster had moved,
        #    which is only sound if the incoming board is already settled, and
        #    `selfcheck`'s random boards are not.
        return (p, self.settle(toasters), crates, gestating)

    def won(self, st) -> bool:
        """``No Toaster``. A gestating toaster is not a Toaster, so a board
        that still has one (or several) wins anyway."""
        return st[1] == 0

    # -- search: the exact distance field ------------------------------------
    def field(self, start, cap: int = 8_000_000, verbose=False):
        """Layered breadth-first enumeration from ``start``, then backward
        induction over the layers.

        Returns ``(plan, optsets, stats)`` with a PROVABLY SHORTEST plan and
        the EXACT optimal press set at every step of it, ``(None, None, stats)``
        when the whole reachable component was enumerated and none of it wins,
        or ``(False, None, stats)`` when ``cap`` states were reached first --
        three outcomes a search with a node budget cannot tell apart.

        The backward induction is what makes the optimal sets exact and free.
        ``good[d]`` is the set of states at BFS depth ``d`` that still win in
        ``d* - d`` presses; a press is optimal at a state of depth ``d`` iff it
        lands in ``good[d + 1]`` (or wins outright at ``d = d* - 1``). That is
        the same thing as ``dist_to_win(successor) == d* - d - 1``: a successor
        with that distance has ``dist_from_start <= d + 1`` and, since the two
        must sum to at least ``d*``, exactly ``d + 1`` -- so it is in layer
        ``d + 1``, and nothing optimal can hide in an earlier layer.
        """
        t0 = time.time()
        layers = [{start}]
        seen = {start}
        win_depth = None
        while layers[-1] and win_depth is None:
            nxt = set()
            for s in layers[-1]:
                for k in range(5):
                    ns = self.step(s, k)
                    if ns[1] == 0:
                        win_depth = len(layers)
                        break
                    if ns in seen:
                        continue
                    seen.add(ns)
                    nxt.add(ns)
                if win_depth is not None:
                    break
            if win_depth is not None:
                break
            if len(seen) > cap:
                return False, None, {"states": len(seen), "secs": time.time() - t0}
            layers.append(nxt)
            if verbose:
                print(f"    depth {len(layers) - 1}: {len(nxt)} new, "
                      f"{len(seen)} total, {time.time() - t0:.1f}s", flush=True)
        stats = {"states": len(seen), "secs": time.time() - t0,
                 "depth": win_depth}
        if win_depth is None:
            return None, None, stats         # exhausted: no win exists at all

        # backward induction over the layers
        good = {s for s in layers[win_depth - 1]
                if any(self.step(s, k)[1] == 0 for k in range(5))}
        goods = [good]
        for d in range(win_depth - 2, -1, -1):
            good = {s for s in layers[d]
                    if any(self.step(s, k) in good for k in range(5))}
            goods.append(good)
        goods.reverse()                      # goods[d] <-> BFS depth d

        presses, optsets, s = [], [], start
        for d in range(win_depth):
            if d == win_depth - 1:
                best = [k for k in range(5) if self.step(s, k)[1] == 0]
            else:
                best = [k for k in range(5) if self.step(s, k) in goods[d + 1]]
            presses.append(best[0])
            optsets.append(best)
            s = self.step(s, best[0])
        return presses, optsets, stats

    # -- search: bounded, for the levels the field cannot hold ---------------
    def bounded(self, start, limit: int, cap: int = 400_000):
        """Shortest win from ``start`` in at most ``limit`` presses, or None.

        None is ambiguous ON PURPOSE only when the cap bites; every caller
        checks ``capped`` in the returned stats before believing a refusal."""
        stats = {"states": 1, "capped": False}
        if self.won(start):
            return [], stats
        if limit <= 0:
            return None, stats
        seen = {start}
        frontier = [(start, ())]
        for _ in range(limit):
            nxt = []
            for s, path in frontier:
                for k in range(5):
                    ns = self.step(s, k)
                    if ns[1] == 0:
                        stats["states"] = len(seen)
                        return list(path) + [k], stats
                    if ns in seen:
                        continue
                    seen.add(ns)
                    nxt.append((ns, path + (k,)))
            if len(seen) > cap:
                stats.update(states=len(seen), capped=True)
                return None, stats
            frontier = nxt
            if not frontier:
                break
        stats["states"] = len(seen)
        return None, stats

    # -- search: beam, for the one level whose space is out of reach ---------
    def score(self, st) -> int:
        """Beam ordering: how far this board is from having its toasters in
        lines. Six per surviving toaster (the thing that has to reach zero),
        plus, for each of them, the distance to its two nearest neighbours --
        a run of three is exactly "two neighbours at distance one" -- plus how
        far the player has to walk to touch any of them at all."""
        p, toasters, _crates, _gestating = st
        ts = [self.rc[i] for i in range(self.n) if (toasters >> i) & 1]
        if not ts:
            return -1
        total = 0
        for i, a in enumerate(ts):
            ds = sorted(abs(a[0] - b[0]) + abs(a[1] - b[1])
                        for j, b in enumerate(ts) if j != i)
            total += sum(ds[:2]) + (self.h + self.w) * (2 - len(ds[:2]))
        pr, pc = self.rc[p]
        total += min(abs(pr - b[0]) + abs(pc - b[1]) for b in ts)
        return 6 * len(ts) + total

    def beam(self, start, width: int, max_depth: int = 90):
        """Best-first beam over the same dynamics, deduplicated globally.

        Deterministic: candidates are ordered by ``(score, state)`` and the
        state is a tuple of ints, so the tie-break is a total order that does
        not depend on set iteration or on ``PYTHONHASHSEED``."""
        if self.won(start):
            return []
        seen = {start}
        frontier = [(start, ())]
        for _ in range(max_depth):
            cand = []
            for s, path in frontier:
                for k in range(5):
                    ns = self.step(s, k)
                    if ns[1] == 0:
                        return list(path) + [k]
                    if ns in seen:
                        continue
                    seen.add(ns)
                    cand.append((self.score(ns), ns, path + (k,)))
            if not cand:
                return None
            cand.sort(key=lambda x: (x[0], x[1]))
            frontier = [(s, path) for _s, s, path in cand[:width]]
        return None

    def certify_tail(self, start, plan, cap: int = 400_000, verbose=False):
        """Prove (and where possible shorten) a beamed plan from its END
        backwards, and label every step it reaches.

        A beam is a guess. This is not: at each state the plan passes through,
        an exhaustive bounded BFS asks two exact questions -- is there a win
        strictly shorter than the suffix the plan spends here (splice it in and
        start over), and which presses still finish in the presses the plan has
        left (that is the step's optimal SET, exact in both directions).

        Backwards, because the bound -- and so the cost -- shrinks towards the
        end: the last steps are decided in microseconds and the first would
        need the whole state space. The first probe that hits ``cap`` ends the
        certification, and that step and every earlier one are labelled with
        the press the plan takes and nothing else. That is a conservative label
        rather than a wrong one -- it can only omit a tie -- and the report says
        exactly where the line fell.

        Returns ``(plan, optsets, certified)``: the tail of length
        ``certified`` is PROVED shortest and exactly labelled.
        """
        while True:
            states = [start]
            for k in plan:
                states.append(self.step(states[-1], k))
            sets: list = [None] * len(plan)
            certified = 0
            spliced = False
            for i in range(len(plan) - 1, -1, -1):
                remaining = len(plan) - i - 1
                shorter, stats = self.bounded(states[i], remaining, cap)
                if stats["capped"]:
                    break
                if shorter is not None:
                    if verbose:
                        print(f"    splice at {i}: {len(plan) - i} -> "
                              f"{len(shorter)} presses", flush=True)
                    plan = plan[:i] + shorter
                    spliced = True
                    break
                best, capped = [], False
                for k in range(5):
                    ns = self.step(states[i], k)
                    if ns == states[i]:
                        continue             # a press that does nothing at all
                    if ns[1] == 0:
                        if remaining == 0:
                            best.append(k)
                        continue
                    if remaining == 0:
                        continue
                    found, probe = self.bounded(ns, remaining, cap)
                    if probe["capped"]:
                        capped = True
                        break
                    if found is not None:
                        best.append(k)
                if capped or plan[i] not in best:
                    break
                sets[i] = best
                certified += 1
            if spliced:
                continue
            for i in range(len(plan)):
                if sets[i] is None:
                    sets[i] = [plan[i]]
            return plan, sets, certified

    # -- the whole strategy, in one call -------------------------------------
    def solve(self, cap: int = 3_000_000, beam_widths=(60_000, 250_000),
              probe_cap: int = 400_000, verbose=False):
        """A shortest plan with exact optimal sets when the state space fits in
        ``cap``; the best beam-and-certify plan otherwise.

        Returns ``(presses, optsets, info)``, or ``(None, None, info)`` when the
        level is provably unwinnable.
        """
        presses, optsets, stats = self.field(self.state, cap, verbose=verbose)
        if presses is not False:
            info = dict(stats, method="field", certified=len(presses or []))
            return presses, optsets, info
        # Every width is run and the shortest kept, rather than the first that
        # answers: a beam that finds a win has not finished looking, and this
        # search happens once per level for the life of the plan cache.
        plan = None
        for width in beam_widths:
            found = self.beam(self.state, width)
            if found is not None and (plan is None or len(found) < len(plan)):
                plan, best_width = found, width
        if plan is None:
            return None, None, dict(stats, method="beam")
        raw = len(plan)
        plan, optsets, certified = self.certify_tail(self.state, plan,
                                                     probe_cap, verbose=verbose)
        return plan, optsets, dict(stats, method="beam", beam_width=best_width,
                                   beam_presses=raw, certified=certified)


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class ToasterExpert(PSExpert):
    """`PSExpert`'s plan memo, restore discipline and disk cache around the
    native search. `heuristic` is never called -- nothing here is A*."""

    directions = list(DIRNAMES)
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "the_trouble_with_toasters_plans.json")

    #: `ToasterBoard.field`'s state budget. Levels 0-3 need at most 1.25M and
    #: are proved shortest; level 4 has 39M states inside 28 presses alone, so
    #: no affordable budget holds its field and it falls through to
    #: the beam and `certify_tail`. Kept just above what levels 0-3 need, because every
    #: state level 4 spends here is wasted before that fallback.
    field_cap: int = 3_000_000

    #: Bound on the exhaustive probes `certify_tail` runs for a beamed level.
    #: Every probe is exact within it; the cap only decides how far back from
    #: the end of the plan the certification reaches, and it buys less and less
    #: as it grows -- on level 4, 400k certifies 16 of the 30 steps in 8 s, 2M
    #: certifies 21 in 124 s and 6M certifies 24 in 374 s. 2M is the knee. Not
    #: one certified step has a second optimal press, so what the uncertified
    #: head loses is very probably nothing at all.
    probe_cap: int = 2_000_000

    def setup(self):
        self.ids = {n: self.g.obj_name_to_idx[n] for n in OBJECTS}
        self.info: dict = {}                 # level -> what `solve` reported
        self._last_info = None

    def plan(self, eng, level=None):
        """`PSExpert.plan`, plus a note of how the plan was found. Only a
        search that actually RAN leaves one -- a plan served from
        ``plan_cache_path`` reports itself as cached, which is the truth."""
        found = super().plan(eng, level)
        if level is not None and self._last_info is not None:
            self.info[level] = self._last_info
        self._last_info = None
        return found

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "ToasterExpert enumerates or beams; heuristic is unused")

    def board(self, eng) -> ToasterBoard:
        return ToasterBoard(eng, self.ids)

    def _search(self, eng):
        board = self.board(eng)
        presses, optsets, info = board.solve(cap=self.field_cap,
                                             probe_cap=self.probe_cap)
        self._last_info = info
        if presses is None:
            return None
        return Plan([DIRNAMES[k] for k in presses],
                    [[DIRNAMES[k] for k in s] for s in optsets])


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class ToasterSolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    game_module_id = GAME_ID
    expert_cls = ToasterExpert
    #: The longest plan is level 4's; this bounds it plus the exploration
    #: prefix and stays well inside the adapter's own 200-press level budget.
    max_steps = 120


# ---------------------------------------------------------------------------
# --selfcheck: the model against the interpreter
# ---------------------------------------------------------------------------

def _read_engine(eng, ids):
    """The interpreter's board, in the model's own terms."""
    player, toasters, crates, gestating = 0, 0, 0, 0
    w = len(eng.grid[0])
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            i = r * w + c
            if ids["player"] in cell:
                player = i
            if ids["toaster"] in cell:
                toasters |= 1 << i
            if ids["crate"] in cell:
                crates |= 1 << i
            if ids["toastergestating"] in cell:
                gestating |= 1 << i
    return (player, toasters, crates, gestating)


def _settle_scan(board, toasters: int, *, rows_first=False, whole=True) -> int:
    """A second, deliberately naive `ToasterBoard.settle`: walk each line, find
    its maximal runs, delete them.

    Two jobs. It is the INDEPENDENT implementation the shift-and-AND version is
    checked against (`selfcheck` compares the two on random boards, where the
    fast one's cleverness -- that OR-ing MATCH shifted copies of the window
    heads covers a whole run of any length -- is the thing that could be
    wrong). And its two knobs are the variants that MEASURE the axis order and
    the whole-run rule against the interpreter."""
    cols = [[r * board.w + c for r in range(board.h)] for c in range(board.w)]
    rows = [[r * board.w + c for c in range(board.w)] for r in range(board.h)]
    for lines in ([rows, cols] if rows_first else [cols, rows]):
        doomed = 0
        for line in lines:
            run = []
            for cell in line + [None]:
                if cell is not None and (toasters >> cell) & 1:
                    run.append(cell)
                    continue
                if len(run) >= MATCH:
                    for x in (run if whole else run[:MATCH]):
                        doomed |= 1 << x
                run = []
        toasters &= ~doomed
    return toasters


def _variant_step(board, st, k, *, rows_first=False, whole=True,
                  spawn_first=False):
    """`ToasterBoard.step` with one of its three load-bearing decisions
    deliberately made the other way, so `selfcheck` can MEASURE that each one
    is a fact about the interpreter rather than a preference.

    ``rows_first`` resolves the horizontal axis of the match rule before the
    vertical; ``whole=False`` deletes only the first `MATCH` toasters of a
    longer run instead of all of it; ``spawn_first`` re-arms the spawners
    before the gestating toasters hatch, which is what the two rules do if you
    read their blocks the other way round."""
    p, toasters, crates, gestating = st
    if spawn_first:
        gestating |= board.spawn_mask & ~(toasters | (1 << p))
        obstacle = toasters | (1 << p)
        toasters |= gestating & ~obstacle
        gestating &= obstacle
    else:
        obstacle = toasters | (1 << p)
        toasters |= gestating & ~obstacle
        gestating &= obstacle
        gestating |= board.spawn_mask & ~(toasters | (1 << p))
    if k < 4:
        tgt = board.nb[p][k]
        if tgt >= 0:
            occupied = toasters | crates
            moved = True
            if (occupied >> tgt) & 1:
                beyond = board.nb[tgt][k]
                if beyond >= 0 and not (occupied >> beyond) & 1:
                    if (toasters >> tgt) & 1:
                        toasters = (toasters & ~(1 << tgt)) | (1 << beyond)
                    else:
                        crates = (crates & ~(1 << tgt)) | (1 << beyond)
                else:
                    moved = False
            if moved:
                back = board.nb[p][_OPP[k]]
                if back >= 0 and (crates >> back) & 1:
                    crates = (crates & ~(1 << back)) | (1 << p)
                p = tgt
    gestating &= ~crates
    return (p, _settle_scan(board, toasters, rows_first=rows_first,
                            whole=whole), crates, gestating)


def selfcheck(trials=30, steps=90, boards=4000, verbose=True):
    """Fuzz the model against the real interpreter, two ways.

    ROLLOUTS from every level start are the honest fuzz: random play walks into
    the board edge and into toasters it cannot move (both must be exact
    no-ops), stands on spawners (which suppresses them) and off them, drags the
    crate around behind it, and occasionally lines three toasters up. Guided
    play draws mostly from the presses the model believes move a toaster, so
    the rollouts actually reach the deletion rule instead of spending
    themselves walking; uniform play stays in because it is what covers the
    no-op classification and the wait button, and on its own the guided mode
    would be circular.

    But rollouts from five hand-made levels barely exercise the parts of the
    turn that are not sokoban. So the second mode loads RANDOM boards --
    toasters, crates, spawners and gestating toasters scattered over a 7x7
    field -- and steps one random press on each. That is where the three
    decisions the model turns on are measured, each against a variant that
    makes it the other way: whole runs versus three of them, columns before
    rows, and hatching before re-arming.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    ids = {n: game._game.obj_name_to_idx[n] for n in OBJECTS}
    bad = presses = noops = moved = deleted = spawned = wins = 0

    for level in range(game.n_levels):
        for guided in (False, True):
            for t in range(trials):
                game.set_level(level)
                board = ToasterBoard(eng, ids)
                st = board.state
                rng = random.Random(f"{GAME_NAME}:selfcheck:{level}:{guided}:{t}")
                for _ in range(steps):
                    nexts = [board.step(st, k) for k in range(5)]
                    pushy = [k for k in range(4)
                             if nexts[k][1] != st[1] or nexts[k][2] != st[2]]
                    live = [k for k in range(5) if nexts[k] != st]
                    roll = rng.random()
                    if guided and pushy and roll < 0.6:
                        k = rng.choice(pushy)
                    elif guided and live and roll < 0.95:
                        k = rng.choice(live)
                    else:
                        k = rng.randrange(5)
                    before = st
                    eng.step(DIRNAMES[k])
                    st = board.step(st, k)
                    presses += 1
                    if st == before:
                        noops += 1
                    if st[0] != before[0]:
                        moved += 1
                    spawned += bin(st[3] & ~before[3]).count("1")
                    deleted += max(0, bin(before[1]).count("1")
                                   - bin(st[1]).count("1"))
                    if (_read_engine(eng, ids) != st
                            or eng.check_win() != board.won(st)):
                        bad += 1
                        if bad < 4:
                            print(f"  L{level} t{t} guided={guided} "
                                  f"{DIRNAMES[k]}: model {st} != engine "
                                  f"{_read_engine(eng, ids)}")
                        break
                    if board.won(st):
                        wins += 1
                        break

    # -- random boards, one press each: the three decisions, measured --------
    h = w = 7
    bg = game._game.obj_name_to_idx["background"]
    rng = random.Random(f"{GAME_NAME}:selfcheck:fields")
    cells = [(r, c) for r in range(h) for c in range(w)]
    variants = {"rows first": 0, "three of a run": 0, "spawn before hatch": 0}
    fields = fdel = 0
    for _ in range(boards):
        rng.shuffle(cells)
        n_t = rng.randint(2, 14)
        n_c = rng.randint(0, 3)
        n_s = rng.randint(0, 3)
        toasters = cells[:n_t]
        crates = cells[n_t:n_t + n_c]
        player = cells[n_t + n_c]
        spawners = cells[n_t + n_c + 1:n_t + n_c + 1 + n_s]
        grid = [[{bg} for _ in range(w)] for _ in range(h)]
        for (r, c) in toasters:
            grid[r][c].add(ids["toaster"])
        for (r, c) in crates:
            grid[r][c].add(ids["crate"])
        for (r, c) in spawners:
            grid[r][c].add(ids["spawner"])
            if rng.random() < 0.5:
                grid[r][c].add(ids["toastergestating"])
        grid[player[0]][player[1]].add(ids["player"])
        eng.load_level(grid)
        board = ToasterBoard(eng, ids)      # a fresh board EVERY time: the
        st = board.state                    # spawners move too, not just the
                                            # pieces
        k = rng.randrange(5)
        eng.step(DIRNAMES[k])
        live = _read_engine(eng, ids)
        mine = board.step(st, k)
        fields += 1
        if bin(st[1]).count("1") != bin(live[1]).count("1"):
            fdel += 1
        if live != mine:
            bad += 1
            if bad < 8:
                print(f"  board fuzz {DIRNAMES[k]}: model {mine} != engine {live}")
        if board.settle(st[1]) != _settle_scan(board, st[1]):
            bad += 1
            print(f"  settle: fast {board.settle(st[1])} != scan "
                  f"{_settle_scan(board, st[1])}")
        for name, kwargs in (("rows first", dict(rows_first=True)),
                             ("three of a run", dict(whole=False)),
                             ("spawn before hatch", dict(spawn_first=True))):
            if _variant_step(board, st, k, **kwargs) != live:
                variants[name] += 1
    if verbose:
        print(f"  {presses} presses fuzzed over {game.n_levels} levels "
              f"({noops} exact no-ops, {moved} player moves, {spawned} "
              f"toasters gestated, {deleted} annihilated, {wins} accidental "
              f"wins)")
        print(f"  {fields} random boards stepped ({fdel} of them with a "
              f"deletion)")
        for name, n in variants.items():
            print(f"    a model that got '{name}' the other way: "
                  f"{n} of {fields} boards wrong")
    return bad


# ---------------------------------------------------------------------------
# --audit: every cell composition, pixel-distinct at every cell size
# ---------------------------------------------------------------------------

#: Every stack of objects a square in this game can hold. Player, Toaster and
#: Crate share one collision layer, so no two of them ever meet; Spawner and
#: ToasterGestating each have their own layer underneath. A gestating toaster
#: only ever exists on a spawner (nothing else creates one), and never on a
#: crate (the LATE rule deletes it the same turn the spawner makes it), which
#: is why those two combinations are absent.
_AUDIT_CASES = {
    "floor": (),
    "spawner": ("spawner",),
    "spawner + gestating": ("spawner", "toastergestating"),
    "player": ("player",),
    "player on spawner": ("player", "spawner"),
    "player on spawner + gestating": ("player", "spawner", "toastergestating"),
    "toaster": ("toaster",),
    "toaster on spawner": ("toaster", "spawner"),
    "toaster on spawner + gestating": ("toaster", "spawner",
                                       "toastergestating"),
    "crate": ("crate",),
    "crate on spawner": ("crate", "spawner"),
}


def audit(verbose=True):
    """Assert every cell COMPOSITION renders differently at every cell size the
    levels use.

    Composition, not object: as shipped, the Toaster sprite was opaque
    everywhere the Spawner and the ToasterGestating draw, and the Crate sprite
    was opaque everywhere at all -- so `toaster`, `toaster on spawner` and
    `toaster on spawner + gestating` were ONE picture, and so were `crate` and
    `crate on spawner`. Both are load-bearing distinctions: a spawner under a
    toaster is a spawner that fires again the moment the toaster is pushed off,
    a gestating one under it fires a turn sooner than that, and a crate on a
    spawner is the game's one permanent answer to it ("A crate is better than
    no crate", says level 4). The fix is in
    `data/puzzlescript_games/The_Trouble_with_Toasters.txt`: the Spawner and
    the ToasterGestating each grew four corner dots in their own colour, and
    the Toaster and the Crate had their corners punched transparent.

    The board is FILLED with the composition and whole frames are compared
    rather than one cell being sliced out by ``cell_px`` arithmetic:
    `_render_frame` upscales a sub-64 render to fill the frame and then
    letterboxes it, so the cell grid in the output is not ``cell_px``-aligned
    and the crop lands in the wrong window. Rendering is per-cell independent,
    so a whole-frame comparison is the same test, done where the geometry
    cannot drift."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, parsed = game._engine, game._game
    idx = parsed.obj_name_to_idx
    sizes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in _AUDIT_CASES.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, parsed))
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        if verbose:
            print(f"  {h}x{w} (cell {64 // max(h, w)}px, levels "
                  f"{','.join(str(x) for x in levels)}): "
                  f"{len(_AUDIT_CASES)} compositions, "
                  f"{'all distinct' if not clashes else 'IDENTICAL ' + str(clashes)}")
    return bad


# ---------------------------------------------------------------------------
# --symmetry: the presentation augmentation is a true symmetry of the mechanic
# ---------------------------------------------------------------------------

def symmetry(walk_presses=200, boards=4000, verbose=True):
    """Replay every level through the ADAPTER at every presentation it can draw
    and require the frames to be exactly the transform of the unaugmented ones.

    This is the evidence for putting the game in `_FLIP_GAMES`. The structural
    argument is the ESCAPE! one -- gravity-free, screen-relative input, a win
    condition that names no direction ("No Toaster"), every rule that moves
    anything written with the relative ``>`` / ``<`` force, and no sprite that
    encodes a facing.

    The one asymmetry the mechanic does have is `settle`'s axis order: columns
    resolve before rows, so a plus of toasters loses its column. That is
    exposed by the mandatory ROTATION and not by the flips, and provably so --
    a mirror maps columns to columns and rows to rows, so it commutes with an
    axis-ordered settle exactly, while a quarter turn swaps the two. So flip
    membership costs nothing in consistency that rotation is not already
    costing, which is the Gobble Rush argument. The second half of this report
    MEASURES how often the order decides anything at all: how many random
    boards settle differently with the axes swapped, and whether any state the
    five plans pass through is one of them.

    For a ps: game the adapter rotates and mirrors the rendered PICTURE and
    remaps the input; the engine grid is never transformed. So the check is
    that the same engine directions driven at all 16 presentations produce
    exactly the transformed frames. Both the PLANS and a seeded random walk are
    replayed -- the walk is what reaches the boards a plan never visits
    (toasters wedged against the edge, spawners buried, lines of four
    annihilated at once), and it presses ACTION too."""
    solver = ToasterSolver()
    game, expert, _ = solver._ensure(0)
    plans = {}
    for level in range(game.n_levels):
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    def drive(g, level, presses):
        g.set_level(level)
        out = [np.asarray(g._current_frame)]
        for act in presses:
            fd = g.perform_action(ActionInput(id=act))
            out.append(np.asarray(fd.frame[-1] if fd.frame
                                  else g._current_frame))
        return out

    def transform(frame, k, hflip, vflip):
        if k:
            frame = np.rot90(frame, k=k)
        if hflip:
            frame = np.fliplr(frame)
        if vflip:
            frame = np.flipud(frame)
        return np.ascontiguousarray(frame)

    ref, seen, bad = {}, set(), 0
    for seed in range(400):
        if len(seen) == 16 and seed > 40:
            break
        g = solver.make_game(seed)
        for level in range(g.n_levels):
            g.set_level(level)
            k = (g._rotation_k, g._hflip, g._vflip)
            seen.add(k)
            rng = random.Random(f"{GAME_NAME}:symmetry:{level}")
            walk = [_ID_TO_GAMEACTION[rng.randrange(1, 6)]
                    for _ in range(walk_presses)]
            for tag, presses in (("plan", [screen_action(d, *k)
                                           for d in plans[level]]),
                                 ("walk", [inverse_remap_action_full(a, *k)
                                           for a in walk])):
                frames = drive(g, level, presses)
                if tag == "plan" and g._state != GameState.WIN:
                    print(f"  seed {seed} L{level} {k}: plan did not win")
                    bad += 1
                key = (level, tag)
                if key not in ref:
                    if k != (0, False, False):
                        continue          # nothing to compare against yet
                    ref[key] = frames
                    continue
                if any(not np.array_equal(transform(a, *k), b)
                       for a, b in zip(ref[key], frames)):
                    print(f"  seed {seed} L{level} {k}: {tag} frames are not "
                          f"the transform of the unaugmented ones")
                    bad += 1
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")

    # -- how often the axis order decides anything --------------------------
    eng = game._engine
    ids = expert.ids
    game.set_level(0)
    board = ToasterBoard(eng, ids)
    rng = random.Random(f"{GAME_NAME}:symmetry:axes")
    chiral = 0
    for _ in range(boards):
        # Toaster counts drawn from the range the levels and the plans
        # actually reach, rather than a fixed dense fill -- the point of the
        # number is how often the order decides anything in play.
        mask, cells = 0, list(range(board.n))
        rng.shuffle(cells)
        for cell in cells[:rng.randint(2, 14)]:
            mask |= 1 << cell
        if _settle_scan(board, mask) != _settle_scan(board, mask,
                                                     rows_first=True):
            chiral += 1
    plan_chiral = 0
    plan_states = 0
    for level in range(game.n_levels):
        game.set_level(level)
        b = ToasterBoard(eng, ids)
        st = b.state
        for direction in plans[level]:
            plan_states += 1
            if _settle_scan(b, st[1]) != _settle_scan(b, st[1],
                                                      rows_first=True):
                plan_chiral += 1
            st = b.step(st, DIRNAMES.index(direction))
    if verbose:
        print(f"  axis order decides the outcome on {chiral} of {boards} "
              f"random toaster fields; on {plan_chiral} of the {plan_states} "
              f"boards the five plans pass through")
    return bad


# ---------------------------------------------------------------------------
# --bfs: the independent optimality proof
# ---------------------------------------------------------------------------

def bfs_report(levels=None, cap=8_000_000, max_depth=None, verbose=True):
    """Exhaustive primitive-press BFS over the whole reachable state space, and
    the shortest win it finds compared against the plan.

    This is an INDEPENDENT implementation: a flat frontier with no layering, no
    backward induction and no optimal-set bookkeeping, so it shares nothing
    with `ToasterBoard.field` but the model itself. On a level the field
    solved, agreement is the proof that the induction did not lose a shorter
    answer. On level 4, whose space no affordable field can hold, it is the
    proof from the other side: it reports the deepest layer it completed
    WITHOUT a win, which is a lower bound on that level's answer.

    States are kept PACKED into one integer rather than as the model's tuples.
    That is not tidiness -- a tuple of four Python ints is ~250 bytes and a
    packed one ~44, and level 4 has 39.2M states inside 28 presses, which is
    the difference between ~10 GB and ~4 GB.

    ``max_depth`` expands its last layer for the win test ONLY, recording
    nothing: the deepest layer is always the biggest, so "is there a win at
    depth D?" is answerable in the memory depth D-1 already costs. That is how
    level 4's lower bound is pushed one press further than the machine could
    otherwise hold.
    """
    solver = ToasterSolver()
    game, expert, _ = solver._ensure(0)
    eng = game._engine
    bad = 0
    for level in (range(game.n_levels) if levels is None else levels):
        game.set_level(level)
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        step = board.step
        n = board.n
        sc, sg = 6 + n, 6 + 2 * n
        cell_mask = (1 << n) - 1

        def pack(st):
            return st[0] | (st[1] << 6) | (st[2] << sc) | (st[3] << sg)

        def unpack(v):
            return (v & 63, (v >> 6) & cell_mask, (v >> sc) & cell_mask,
                    v >> sg)

        t0 = time.time()
        seen = {pack(board.state)}
        frontier = [pack(board.state)]
        depth, best = 0, None
        proved = 0            # no win exists within this many presses
        exhausted = False
        while frontier and best is None:
            # The last layer is expanded for the win test only: nothing about
            # it is recorded, so asking "is there a win one press deeper?"
            # costs time and no memory at all.
            last = max_depth is not None and depth + 1 >= max_depth
            nxt = []
            for v in frontier:
                st = unpack(v)
                for k in range(5):
                    ns = step(st, k)
                    if ns[1] == 0:
                        best = depth + 1
                        break
                    if last:
                        continue
                    nv = pack(ns)
                    if nv in seen:
                        continue
                    seen.add(nv)
                    nxt.append(nv)
                if best is not None:
                    break
            if best is not None:
                break
            proved = depth + 1               # the whole layer had no winner
            if last or len(seen) > cap:
                break
            depth += 1
            frontier = nxt
            exhausted = not nxt
            if verbose:
                print(f"    L{level} depth {depth}: {len(seen)} states, "
                      f"{time.time() - t0:.0f}s", flush=True)
        if best is None:
            if exhausted and max_depth is None:
                verdict = "UNWINNABLE"
                bad += 1
            elif proved + 1 == len(plan):
                verdict = f"lower bound {proved + 1} -- PROVED SHORTEST"
            else:
                verdict = f"lower bound {proved + 1}"
            print(f"  L{level}: {len(seen)} states, NO WIN within {proved} "
                  f"presses (plan is {len(plan)}) -- {verdict}, "
                  f"{time.time() - t0:.0f}s")
            continue
        ok = best == len(plan)
        bad += 0 if ok else 1
        print(f"  L{level}: {len(seen)} reachable states, shortest win {best} "
              f"presses, plan {len(plan)} -- "
              f"{'PROVED SHORTEST' if ok else 'MISMATCH'}, "
              f"{time.time() - t0:.0f}s")
    return bad


# ---------------------------------------------------------------------------
# --ties: every optimal-action label re-derived, independently
# ---------------------------------------------------------------------------

def _wins_within(board, st, limit, budget=[0], cap=3_000_000):
    """Depth-first: can ``st`` reach a win in at most ``limit`` presses?

    Deliberately NOT the breadth-first `ToasterBoard.bounded` that produced the
    labels: this walks the tree depth-first with a best-depth-seen memo, so a
    step whose label came from a bounded BFS is re-derived by something with a
    different traversal order and a different pruning rule."""
    best = {}
    stack = [(st, 0)]
    while stack:
        s, d = stack.pop()
        if board.won(s):
            return True
        if d >= limit:
            continue
        if best.get(s, _INF) <= d:
            continue
        best[s] = d
        budget[0] += 1
        if len(best) > cap:
            raise MemoryError("ties probe out of budget")
        for k in range(4, -1, -1):
            ns = board.step(s, k)
            if ns != s:
                stack.append((ns, d + 1))
    return False


def ties_report(levels=None, max_remaining=17, verbose=True):
    """Re-derive every step's optimal SET with an independent depth-first
    bounded search and require it to equal the label the plan carries.

    `ToasterBoard.field` measures the same thing by backward induction over BFS
    layers and `optsets_bounded` by breadth-first probes, so this is the check
    that neither has quietly changed what "optimal" means. Agreement in BOTH
    directions is the claim: a press the labels omit and the search finds would
    be a target the corpus teaches is wrong, and one the labels offer and the
    search refutes would be a second-best press taught as optimal.

    ``max_remaining`` bounds how deep a re-derivation is attempted, which only
    bites on level 4 -- its plan is far longer than any tree this can walk, so
    the report says how many of its steps it managed to certify."""
    solver = ToasterSolver()
    game, expert, _ = solver._ensure(0)
    eng = game._engine
    bad = checked = skipped = 0
    for level in (range(game.n_levels) if levels is None else levels):
        game.set_level(level)
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        st = board.state
        done = 0
        for i, taken in enumerate(plan):
            left = len(plan) - i - 1
            if left > max_remaining:
                skipped += 1
                st = board.step(st, DIRNAMES.index(taken))
                continue
            truth = []
            for k in range(5):
                ns = board.step(st, k)
                if ns == st:
                    continue                    # a press that does nothing
                checked += 1
                if board.won(ns):
                    if left == 0:
                        truth.append(DIRNAMES[k])
                elif left and _wins_within(board, ns, left):
                    truth.append(DIRNAMES[k])
            if sorted(truth) != sorted(sets[i]):
                bad += 1
                print(f"  L{level} step {i}: labels {sorted(sets[i])} != "
                      f"search truth {sorted(truth)}")
            done += 1
            st = board.step(st, DIRNAMES.index(taken))
        if verbose:
            print(f"  L{level}: {done} of {len(plan)} steps re-derived")
    if verbose:
        print(f"  {checked} presses re-derived depth-first, {skipped} steps "
              f"out of reach, {bad} disagreements")
    return bad


# ---------------------------------------------------------------------------
# --plans
# ---------------------------------------------------------------------------

def _plan_report():
    """Every level's plan, replayed through the real interpreter -- the
    end-to-end test of the model, the search and the tie labelling at once."""
    solver = ToasterSolver()
    game, expert, solvable = solver._ensure(0)
    total = ties = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level:2d}: UNSOLVED")
            continue
        toasters = bin(board.state[1]).count("1")
        spawners = bin(board.spawn_mask).count("1")
        crates = bin(board.state[2]).count("1")
        for direction in plan:
            eng.step(direction)
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        step_ties = sum(len(s) - 1 for s in sets)
        total += len(plan)
        ties += step_ties
        info = expert.info.get(level)
        how = ("served from the plan cache" if info is None else
               f"{info['method']}, {info['states']} states, "
               f"{info['certified']}/{len(plan)} steps certified")
        print(f"  L{level:2d}: {eng.height}x{eng.width}, {toasters} toasters, "
              f"{spawners} spawners, {crates} crates -> {len(plan):3d} presses"
              f"  win={eng.check_win()}  {step_ties:3d} tie-presses  [{how}]")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second right answer")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--bfs" in sys.argv:
        rest = sys.argv[sys.argv.index("--bfs") + 1:]
        args = [int(a) for a in rest if a.isdigit()]
        opts = dict(a.split("=") for a in rest if "=" in a)
        violations = bfs_report(args or None,
                                cap=int(opts.get("--cap", 8_000_000)),
                                max_depth=(int(opts["--max-depth"])
                                           if "--max-depth" in opts else None))
        print(f"bfs: {violations} mismatches")
        sys.exit(1 if violations else 0)
    if "--ties" in sys.argv:
        rest = sys.argv[sys.argv.index("--ties") + 1:]
        args = [int(a) for a in rest if a.isdigit()]
        opts = dict(a.split("=") for a in rest if "=" in a)
        violations = ties_report(args or None,
                                 max_remaining=int(opts.get("--max-remaining",
                                                            17)))
        print(f"ties: {violations} disagreements")
        sys.exit(1 if violations else 0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(ToasterSolver.main())
