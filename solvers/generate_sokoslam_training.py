"""Generate Phase-1 training data for the PuzzleScript game ps:sokoslam
("Sokoslam", Aaron Steed).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a NATIVE model of them, the macro A* that searches it, the
differential fuzz that certifies the model against the real interpreter, and the
render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema::

    {
      "game_id": "puzzlescript_sokoslam",
      "levels": [
        {"level_id": 0,
         "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
         "actions":      [a_0, ..., a_{T-1}]},     # length T
        ...
      ]
    }

``actions[0]`` is the RESET that produced ``obs[0]``; ``actions[i]`` for ``i>=1``
is the action that took the agent from ``obs[i-1]`` to ``obs[i]``. The recorded
index is the *screen* action (post rotation/flip remap), so replaying the
recorded actions reproduces the recorded frames exactly. Every expert step
carries the full set of equally-optimal presses.

The game
--------
A sokoban where you do not PUSH, you KICK. Walk into a crate and it leaves --
you stay where you are and the crate flies in a straight line until something
stops it. Park a crate on every marker.

Everything below was MEASURED against the interpreter (``--fuzz`` is the
executable form of it), not read off the .txt.

* **The kick does not move the player.** ``up [ up PlayerStand | Target ] ->
  [ PlayerUp | Target MoveUp ]`` consumes the press's force and hands a ``Move``
  marker to the thing in front. So the player never advances onto a crate's old
  square, which is the single fact that makes this not a sokoban: you kick from
  the square you are already on, and after the kick you are still on it. Every
  square a crate has to be launched from must therefore be *walkable*, and the
  crate's own square never becomes one.

* **A kick travels THROUGH a chain and only the last crate flies.**
  ``[ Crate MoveUp | Target No MoveUp ] -> [ Crate CrateRest | Target MoveUp ]``
  is not ``random``, so it runs to a fixpoint before anything actually moves: a
  column of three touching crates transmits the kick to the far end and the two
  nearer ones do not budge. Level 3, kicking down into the left column: the top
  two stay, the bottom one slides to the wall.

* **What stops a flying crate is the ``Item`` group** -- Player, Wall, Crate,
  Bomb, GateClosed, BreakWall. Markers, switches, open gates and PITS are on
  other collision layers, and a crate flies straight over all of them.

* **A pit kills the crate that comes to REST on it**, and only then: ``[ Crate
  No Move Pit ] -> [ CrateFall Pit ]`` is guarded on the crate having no Move,
  so pits are lethal landing pads rather than lethal ground. Level 3, kicking
  the left crate of row 2 right: it sails over three pit squares, is stopped by
  the crate at the far end (which flies on), lands on the last pit square and is
  deleted.

* **A bomb is a crate that explodes when it stops.** ``[ Bomb CrateRest ] ->
  [ Explosion ]``, and an Explosion hands a Move to each of its four
  neighbouring crates before deleting itself -- so a bomb kicked straight into a
  wall detonates in place and scatters everything around it. Explosions chain,
  and a bomb that is merely *touched* by an arriving crate does not go off: the
  Move is passed to it and it flies, and it explodes wherever it stops.

* **A BreakWall is kickable AND blocking.** It is in ``Target`` (a kick, or a
  crate arriving, gives it a Move) and in ``Item`` (it stops a flying crate).
  ``[ BreakWall Move ] -> [ BreakDebris1 ]`` then deletes it over three cosmetic
  ticks. So kicking one destroys it and moves nothing, and a crate that hits one
  stops beside the hole it just made.

* **The switch is held by ANY item, the player included** -- ``late [ Item No
  Move Switch ] [ GateClosed ] -> ... GateOpening`` -- and the gates slam shut
  the moment it is released. The player standing on the switch can therefore
  kick a crate *through* the gates, but can never walk through them himself:
  stepping off the switch closes them again. Parking a crate on the switch is
  the other way to hold them open, and it costs a crate.

* **A gate closing DESTROYS whatever is resting on it.** ``late [ No Item
  Switch ] [ GateOpen ] -> [ Switch ] [ GateClosed ]`` writes GateClosed into
  the Item layer and overwrites the crate that was standing on the open gate.
  The model reproduces it; ``--fuzz`` covers it.

* **The win is ``All Marker on Crate``** (plus ``No Move``, which every settled
  board satisfies). Spare crates are free -- most levels ship more crates than
  markers -- and a BOMB parked on a marker counts, because ``Crate = CrateBase
  or Bomb``.

* ``noaction`` is in the prelude, so the game is FOUR presses; `_DIRS` says so
  and nothing ever branches on ACTION5.

Why a native model
------------------
``eng.step`` runs at ~530 steps/s here: a kick is an ``again`` chain one tick per
travelled square and the interpreter re-scans a 20-rule program for each. A
macro A* over the interpreter would spend its whole budget inside it (the
entrepotphage rule in this family: measure ``eng.step`` before choosing an
engine-blackbox expert). `_Board` re-implements the tick above two orders of
magnitude faster and the interpreter's only job is to CERTIFY -- ``--fuzz``
replays random play and every level's own solution against it and requires the
two to agree cell for cell, and ``--verify`` replays every shipped plan.

Two model facts are worth stating, because they are what a re-implementation
gets wrong:

* **The ``again`` loop is capped at 50 iterations** by the adapter, and a kick
  costs one iteration per travelled square plus three for any debris it makes.
  `_Board` counts the same iterations and returns None for a press that would
  overrun, so the search can never plan through a board the interpreter would
  leave half-animated. Nothing reachable on these levels comes close (the
  longest real chain is 14).

* **The two ``random`` movement rules make a press NONDETERMINISTIC when two
  crates fly the same way at once**, which happens only when explosions overlap.
  `_Board.press` counts the matches and, where a rule has more than one,
  simulates the press as a SET of boards (the flying_kick recipe) and accepts it
  only if every schedule settles on the same board. Presses that genuinely
  depend on the roll are refused, so a plan can never fail to replay.

Expert solver
-------------
A LADDER of three searches over the native model, strongest first. Every level
takes the highest rung that answers, and ``--plans`` names the rung it used:

1. **Exact macro A*** (`_Board.solve`). A successor is ``walk to a kick square,
   then kick``, so the search's depth is the number of KICKS while ``g`` still
   counts PRESSES, walk steps included -- which is what the recorded plan is
   scored on. Dedup is the exact ``(player, crates, bombs, breakwalls)`` tuple,
   not the player's reachable region: the region merge is unsound for a
   move-costed search (it mis-costs by up to the region's diameter) and here it
   would be wrong outright, because standing on the switch changes which region
   the player is in. A win is kept as an incumbent until the frontier's ``f``
   reaches its cost, because a macro is ``walk + kick`` and macros therefore
   cost different amounts -- the first win GENERATED is not the shortest one.
2. **Kick-costed A*** (`_Board.solve_kicks`) for the levels rung 1 cannot close.
   ``g`` counts kicks, which makes the depth six or eight rather than thirty,
   and states dedup on ``(pieces, player REGION)`` -- the merge that is unsound
   above is exactly right here, because inside one region the player walks free
   as far as *which kicks exist* is concerned. Winning, not shortest.
3. **A width-capped beam** (`_Board.solve_beam`) over the same macros, for the
   levels rung 2 cannot reach either. Winning, not shortest, and longer.

THE HEURISTIC is what decides rung 1, and a kick COUNT is not enough for it: on
these boards the kicks are a handful and the WALKING between them is the whole
cost, so a kick-counting estimate is flat over an entire approach and A* has
nothing to steer with. So there are two tables, both built by a reverse BFS over
the same relaxed kick -- "a crate at X kicked in direction d comes to rest
anywhere along the ray, and the player may stand anywhere on the contiguous run
behind it, because a kick travels through a chain":

* `_Board.kick_tables` counts KICKS only, per marker;
* `_Board.press_tables` counts PRESSES, over ``(crate square, player square)``
  pairs, so the player's walk to each kick square is charged for. Both relaxed
  moves cost 1, so the field is a plain BFS rather than a Dijkstra.

The estimate is the larger of "the sum of every uncovered marker's kick
minimum" and "one marker's full press distance plus the kick minimum of each
other marker", and the second is sound because the presses it counts are
disjoint: a press is either a walk or a kick, each marker ends under a crate of
its own, and one kick moves one crate. Relaxing only ever ADDS moves, so both
tables are lower bounds and rung 1's plans are shortest -- except on a BOMB
level, where an explosion moves crates nobody kicked and no count of kicks
bounds the presses at all. There the estimate is a guide, and optimality is
PROVED separately by a bounded sweep under the trivially admissible estimate
(`_Board.uniform`), or reported as unproved.

A crate square in no table is a crate that can never help again, and a marker
with no such crate is a lost level -- which is the search's dead test, and does
most of the pruning on the levels with pits.

Optimal-action sets
-------------------
For a plan proved shortest they are MEASURED, not inferred: at every step each
of the four presses is taken and the board it lands on is re-solved under a
bound of ``remaining - 1``; the press is optimal exactly when a plan that short
exists. Two sound prunes keep it affordable -- a press that changes nothing
cannot be on a shortest path, and neither can one whose admissible estimate
already exceeds what is left. ``--ties`` re-derives every label independently.

For a plan from rung 2 or 3 there is no such thing as "every equally shortest
press", so the smaller true question is answered instead: given the KICK
sequence the plan commits to, a kick is labelled with itself and each step of a
WALK with every direction that keeps it on a shortest route to the square the
next kick is taken from. No step is ever left unlabelled.

A note on the art
-----------------
One pixel-level fix ships in ``data/puzzlescript_games/Sokoslam.txt``: the
Switch sprite has gained its four corner pixels. A CrateBase is transparent at
exactly those four pixels and opaque everywhere else, and the original Switch
was transparent at all four, so a crate parked on the switch rendered
pixel-for-pixel identically to a crate on the floor -- and parking a crate on
the switch is how levels 11 and 17 hold their gates open. ``--audit`` is the
executable form of that claim and fails without the fix.

Augmentation: rotations only, NOT the mirrors
---------------------------------------------
The adapter gives this game the usual four board rotations (one per seed), and
deliberately not the extra flips of `PuzzleScriptAdapter._FLIP_GAMES`, which
would quadruple the corpus again. It meets most of that list's bar -- no
gravity, screen-relative input, a win condition that names no direction -- and
fails it on the ART. PlayerStand is not symmetric under either mirror and has no
partner sprite that is its reflection, so a flipped frame would show a player
the game does not own, and a dynamics model would be taught a sprite that can
never occur. (The four DIRECTIONAL player sprites are a different matter: Left
and Right are each other's hflip, Up and Down are NOT each other's vflip -- but
none of them can reach a recorded frame at all, because the three
"return the player to normal" rules retire them inside the same iteration.
``--fuzz`` asserts exactly that, on every press it takes.)

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the
cached plan replays to a guaranteed WIN. The reset is not a convenience here --
a crate kicked into a pit is gone forever and a bomb detonated in the wrong
place cannot be undone.

CLI
---
    --levels    print every board, with the model's reading of it
    --plans     per-level length, which rung solved it, whether it is proved
                shortest, and tie coverage
    --fuzz      differential test of the native model against the interpreter
    --ties      re-derive every optimal-action label by independent re-solve
    --verify    replay every cached plan through the real interpreter
    --audit     assert every cell composition renders distinctly, at every size

Where it stands: all 18 levels are won in 554 presses, 13 of them proved
shortest, 13% of the steps carry a tie set, the model agrees with the
interpreter over 77807 fuzzed presses, and every plan replays to a WIN in the
real engine.
"""

from __future__ import annotations

import heapq
import json
import os
import random
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                       # noqa: E402
    PSAStarSolver, PSExpert, Plan,
)

GAME_NAME = "Sokoslam"

#: The four presses, in the order that breaks every tie -- so a re-derived plan
#: is byte-identical across processes. ``noaction`` is in the game's prelude, so
#: there is no fifth.
_DIRS = ("up", "right", "down", "left")

#: Direction index -> (dr, dc). The ORDER is the order the movement rules appear
#: in the .txt, and `_Sim.tick` applies them in it -- which decides the outcome
#: when one crate is reachable by two explosions at once (the last rule wins).
_DELTA = ((-1, 0), (0, 1), (1, 0), (0, -1))

#: The adapter's ``again`` cap (`PSEngine.step`'s ``max_again``). A press needing
#: more iterations is left half-animated by the interpreter, so the model refuses
#: it rather than modelling a board no agent can reach.
_MAX_AGAIN = 50

#: Heuristic charge for a state that can no longer be won. Large enough to sink
#: it, finite so nothing overflows.
_DEAD = 1 << 20

#: Levels no rung of the search ladder can win, skipped up front so a shard does
#: not burn the whole budget per level at every startup. EMPTY: all eighteen are
#: won (thirteen of them provably shortest -- see ``--plans``). It is kept as the
#: lever to reach for if a budget here is ever tightened, rather than deleted and
#: re-derived from scratch the next time one is.
_UNSOLVED: frozenset = frozenset()


def _mask(cells) -> int:
    """A set of cell indices as one integer. See `_Sim.state` for why."""
    out = 0
    for k in cells:
        out |= 1 << k
    return out


def _popcount(mask: int) -> int:
    return bin(mask).count("1")


def _cells(mask: int) -> set:
    """The inverse of `_mask`."""
    out = set()
    while mask:
        low = mask & -mask
        out.add(low.bit_length() - 1)
        mask ^= low
    return out


class _NeedChoice(Exception):
    """Raised out of `_Sim.tick` at a ``random`` rule with several matches, so
    the caller can re-run the iteration once per choice. Carries the schedule
    built so far, which every re-run replays before branching."""

    def __init__(self, sched, options):
        super().__init__("ambiguous random rule")
        self.sched = sched
        self.options = options


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Sim:
    """The mutable board of ONE press, mid-``again``-chain.

    Separate from `_Board` (which is the level's static geometry) because a
    press whose ``random`` rules are ambiguous is simulated as a SET of these.

    Every field here is a real object on the PuzzleScript board except
    ``sched``/``pos``, which are the choice schedule described in `_NeedChoice`.
    ``CrateRest`` and ``PlayerRest`` are deliberately absent from the persistent
    state: both are created and cleared inside a single iteration, so they can
    never be part of a board the agent sees.
    """

    __slots__ = ("b", "player", "crates", "bombs", "breaks", "mv", "fall",
                 "expl", "debris", "gopen", "first", "d", "sched", "pos")

    def __init__(self, b, state=None, d=None, src=None):
        self.b = b
        if src is None:
            self.player, crates, bombs, breaks = state
            self.crates = _cells(crates)
            self.bombs = _cells(bombs)
            self.breaks = _cells(breaks)
            self.mv = {}
            self.fall = set()
            self.expl = set()
            self.debris = {}
            self.gopen = b.gates_open(state)
            self.first = True
            self.d = d
        else:
            self.player = src.player
            self.crates = set(src.crates)
            self.bombs = set(src.bombs)
            self.breaks = set(src.breaks)
            self.mv = dict(src.mv)
            self.fall = set(src.fall)
            self.expl = set(src.expl)
            self.debris = dict(src.debris)
            self.gopen = src.gopen
            self.first = src.first
            self.d = src.d
        self.sched = []
        self.pos = 0

    def fork(self):
        return _Sim(self.b, src=self)

    def state(self):
        """The four fields that survive a press: everything else is transient.

        The three object sets go out as BITMASKS. A frozenset of a dozen cells
        is ~700 bytes against ~40 for the int, and the search holds hundreds of
        thousands of these at once -- the entrepotphage lesson, and the
        difference between a level closing and the process being killed."""
        return (self.player, _mask(self.crates), _mask(self.bombs),
                _mask(self.breaks))

    def key(self):
        """A full canonical id INCLUDING the transient objects -- the dedup key
        for the set simulation, where two schedules may only be merged when the
        boards are identical down to the pending Moves."""
        return (self.player, _mask(self.crates), _mask(self.bombs),
                _mask(self.breaks), tuple(sorted(self.mv.items())),
                _mask(self.fall), _mask(self.expl),
                tuple(sorted(self.debris.items())), self.gopen, self.first)

    # -- predicates ----------------------------------------------------------
    def is_item(self, k) -> bool:
        """The ``Item`` or-group: Player, Wall, Crate, Bomb, GateClosed,
        BreakWall. Deliberately NOT CrateFall or Explosion -- both share the
        collision layer but neither is in the group, so a flying crate sails
        into their square and overwrites them."""
        b = self.b
        return (k == self.player or k in b.walls or k in self.crates
                or k in self.bombs or k in self.breaks
                or (not self.gopen and k in b.gates))

    def is_target(self, k) -> bool:
        """``Target = BreakWall or Crate or Bomb`` -- what a kick is handed to,
        and what a flying crate passes its Move on to."""
        return k in self.crates or k in self.bombs or k in self.breaks

    def occupied(self, k) -> bool:
        """The Item collision LAYER: the group plus the two transient objects.
        Governs where the player may be put, not what stops a crate."""
        return self.is_item(k) or k in self.fall or k in self.expl

    # -- the choice schedule -------------------------------------------------
    def choose(self, options):
        """Pick one of a ``random`` rule's matches.

        With a single match there is no choice to record beyond the bookkeeping.
        With several, the first run raises `_NeedChoice` so the caller can fan
        the iteration out over all of them; a re-run replays the schedule it was
        given and branches only at the NEXT undetermined point."""
        pos = self.pos
        self.pos += 1
        if pos < len(self.sched):
            return options[self.sched[pos]]
        if len(options) == 1:
            self.sched.append(0)
            return options[0]
        raise _NeedChoice(list(self.sched), len(options))

    # -- one ``again`` iteration ---------------------------------------------
    def tick(self):
        """Apply the whole rule list once, in source order, then the late rules.

        Returns the engine's continuation flag: some rule carrying ``again``
        matched AND the grid changed over the iteration. Mutates ``self``.
        """
        b = self.b
        again = False
        changed = False

        # [ BreakDebris3 ] -> [ ] / [ BreakDebris2 ] -> [ BreakDebris3 ] again /
        # [ BreakDebris1 ] -> [ BreakDebris2 ] again. Cosmetic -- the Debris
        # layer blocks nothing -- but it keeps the `again` chain alive three
        # more iterations, which is what the 50-cap is spent on.
        if self.debris:
            changed = True
            for cell, stage in list(self.debris.items()):
                if stage >= 3:
                    del self.debris[cell]
                else:
                    self.debris[cell] = stage + 1
                    again = True

        # [ CrateFall ] -> [ ]
        if self.fall:
            self.fall.clear()
            changed = True

        # The press's own force, iteration 0 only: the engine does not reassert
        # it on `again` continuations.
        if self.first:
            self.first = False
            k = b.nb[self.player][self.d]
            # [ > Player | Pit ] -> [ Player | Pit ] cancels the force outright,
            # so a player facing a pit neither steps nor kicks.
            if k >= 0 and k not in b.pits:
                if self.is_target(k):
                    self.mv[k] = self.d
                    changed = True
                elif not self.occupied(k):
                    self.player = k
                    changed = True

        # The movement rules, per direction, in source order. ``rest`` is the
        # iteration's CrateRest stamps: read by the bomb rule below, then
        # dropped by ``[ CrateRest ] -> [ ]`` before the iteration ends, which is
        # why it is a local and not part of the board.
        rest: set = set()
        if self.mv:
            for dd in range(4):
                if dd not in self.mv.values():
                    continue
                # [ Crate MoveD | Target No MoveD ] -> [ Crate CrateRest | Target
                # MoveD ] again -- not `random`, so it runs to a fixpoint and a
                # whole chain resolves inside one iteration.
                while True:
                    hit = None
                    for cell in sorted(self.mv):
                        if self.mv[cell] != dd:
                            continue
                        if cell not in self.crates and cell not in self.bombs:
                            continue
                        nxt = b.nb[cell][dd]
                        if (nxt >= 0 and self.is_target(nxt)
                                and self.mv.get(nxt) != dd):
                            hit = (cell, nxt)
                            break
                    if hit is None:
                        break
                    cell, nxt = hit
                    del self.mv[cell]
                    rest.add(cell)
                    self.mv[nxt] = dd
                    again = changed = True

                # random dir [ Crate MoveD | Item ] -> [ Crate CrateRest | Item ]
                blocked = sorted(
                    cell for cell in self.mv
                    if self.mv[cell] == dd
                    and (cell in self.crates or cell in self.bombs)
                    and b.nb[cell][dd] >= 0 and self.is_item(b.nb[cell][dd]))
                if blocked:
                    cell = self.choose(blocked)
                    del self.mv[cell]
                    rest.add(cell)
                    changed = True

                # random dir [ Crate MoveD | No Item ] -> [ | Crate MoveD ] again
                free = sorted(
                    cell for cell in self.mv
                    if self.mv[cell] == dd
                    and (cell in self.crates or cell in self.bombs)
                    and b.nb[cell][dd] >= 0 and not self.is_item(b.nb[cell][dd]))
                if free:
                    self._advance(self.choose(free), dd)
                    again = changed = True

        # up/right/down/left [ Explosion | Crate ] -> [ Explosion | Crate MoveD ]
        # One rule per direction, each applied to every explosion, in that order
        # -- so a crate caught between two explosions ends up moving the way the
        # LAST matching rule says.
        if self.expl:
            for dd in range(4):
                for e in sorted(self.expl):
                    nxt = b.nb[e][dd]
                    if nxt >= 0 and (nxt in self.crates or nxt in self.bombs):
                        if self.mv.get(nxt) != dd:
                            self.mv[nxt] = dd
                            again = changed = True
            # [ Explosion ] -> [ ]
            self.expl.clear()
            changed = True

        # [ Bomb CrateRest ] -> [ Explosion ] again
        if rest and self.bombs:
            for cell in sorted(rest & self.bombs):
                self.bombs.discard(cell)
                self.expl.add(cell)
                again = changed = True

        # [ Crate No Move Pit ] -> [ CrateFall Pit ] again
        if b.pits:
            for cell in sorted((self.crates | self.bombs) & b.pits):
                if cell in self.mv:
                    continue
                self.crates.discard(cell)
                self.bombs.discard(cell)
                self.fall.add(cell)
                again = changed = True

        # [ CrateRest ] -> [ ]  (``rest`` is per-iteration by construction)

        # [ BreakWall Move ] -> [ BreakDebris1 ] again
        for cell in sorted(self.breaks & self.mv.keys()):
            self.breaks.discard(cell)
            del self.mv[cell]
            self.debris[cell] = 1
            again = changed = True

        # The four late gate rules, collapsed: GateOpening is created and
        # consumed inside one iteration, so the gates are open exactly while a
        # STILL item rests on a switch.
        if b.switches:
            want = False
            for s in b.switches:
                if s in self.mv:
                    continue                    # `Item No Move`
                if (s == self.player or s in self.crates or s in self.bombs
                        or s in self.breaks):
                    want = True
                    break
            if want != self.gopen:
                if not want:
                    # [ GateOpen ] -> [ GateClosed ] writes into the Item layer
                    # and overwrites whatever was resting on the square.
                    for gcell in b.gates:
                        self.crates.discard(gcell)
                        self.bombs.discard(gcell)
                        self.breaks.discard(gcell)
                        self.mv.pop(gcell, None)
                        if self.player == gcell:
                            self.player = -1    # the player was overwritten
                self.gopen = want
                changed = True

        return again and changed

    def _advance(self, cell, dd):
        """``[ Crate MoveD | No Item ] -> [ | Crate MoveD ]``: the crate is
        written into the next square, overwriting anything in the Item layer that
        is not in the ``Item`` group (a CrateFall or an Explosion)."""
        nxt = self.b.nb[cell][dd]
        del self.mv[cell]
        if cell in self.crates:
            self.crates.discard(cell)
            self.crates.add(nxt)
        else:
            self.bombs.discard(cell)
            self.bombs.add(nxt)
        self.fall.discard(nxt)
        self.expl.discard(nxt)
        self.mv[nxt] = dd


class _Board:
    """One level's static geometry plus the tick-accurate kick simulation.

    A STATE is ``(player, crates, bombs, breaks)`` -- one cell index and three
    frozensets of them. Everything else is either static (walls, pits, markers,
    switches, gate squares) or a pure function of the state (a gate is open
    exactly while some item rests on a switch), so the tuple is an exact,
    canonical key.

    Cells are ``row * width + col`` and ``-1`` is off-board, which no reachable
    state ever needs: every board in this game is walled on all four sides,
    which is why the movement rules -- whose brackets need a neighbour square to
    match at all -- never have to cope with a crate flying off the edge.
    """

    def __init__(self, eng, g):
        idx = g.obj_name_to_idx
        h, w = eng.height, eng.width
        self.h, self.w, self.n = h, w, h * w

        def has(cell, *names):
            return any(idx[n] in cell for n in names)

        walls, pits, markers, switches, gates = set(), set(), set(), set(), set()
        crates, bombs, breaks = set(), set(), set()
        player = -1
        for r in range(h):
            for c in range(w):
                cell, k = eng.grid[r][c], r * w + c
                if has(cell, "wallbase", "walltop"):
                    walls.add(k)
                if has(cell, "pittop", "pitbottom"):
                    pits.add(k)
                if has(cell, "marker"):
                    markers.add(k)
                if has(cell, "switch"):
                    switches.add(k)
                if has(cell, "gateclosed", "gateopen"):
                    gates.add(k)
                if has(cell, "cratebase"):
                    crates.add(k)
                if has(cell, "bomb"):
                    bombs.add(k)
                if has(cell, "breakwall"):
                    breaks.add(k)
                if has(cell, "playerstand", "playerup", "playerdown",
                       "playerleft", "playerright"):
                    player = k
        self.walls = frozenset(walls)
        self.pits = frozenset(pits)
        self.markers = frozenset(markers)
        self.switches = frozenset(switches)
        self.gates = frozenset(gates)
        self.marker_mask = _mask(markers)
        self.switch_mask = _mask(switches)
        self.gate_mask = _mask(gates)
        self.start = (player, _mask(crates), _mask(bombs), _mask(breaks))

        self.nb = [[-1] * 4 for _ in range(self.n)]
        for r in range(h):
            for c in range(w):
                k = r * w + c
                for d, (dr, dc) in enumerate(_DELTA):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        self.nb[k][d] = rr * w + cc

        #: True if some switch is next to a gate. `walk_gates_open` treats a
        #: gate the PLAYER is holding open as shut, because stepping off the
        #: switch shuts it in the same iteration -- which is exact as long as he
        #: can never step from the switch straight onto a gate. No shipped board
        #: puts them adjacent (``--levels`` prints this), and where one did the
        #: walk graph would be conservative rather than wrong: a route it will
        #: not offer, never a route that does not exist.
        self.switch_touches_gate = any(
            self.nb[s][d] in self.gates for s in self.switches for d in range(4))

        #: A bomb moves crates without anyone kicking them, and one press can set
        #: off a whole chain of explosions -- so on a bomb level NO count of
        #: kicks is a lower bound on presses and the heuristic below is a search
        #: GUIDE only. `--plans` proves those levels separately (a bounded
        #: uniform-cost sweep) instead of claiming optimality it cannot show.
        self.certified = not bombs
        self._kick_tables = None
        self._press_tables = None
        self._press_cache: dict = {}

    # -- state predicates ----------------------------------------------------
    def gates_open(self, state) -> bool:
        """True while some item rests on a switch. The PLAYER counts, which is
        the whole trick of the switch levels: he can kick a crate through a gate
        he could never walk through."""
        if not self.switch_mask:
            return False
        player, crates, bombs, breaks = state
        held = crates | bombs | breaks
        if player >= 0:
            held |= 1 << player
        return bool(self.switch_mask & held)

    def walk_gates_open(self, state) -> bool:
        """Gates as the player may WALK through them: held by something that is
        not him, because stepping off the switch shuts them in the same
        iteration -- and, on every shipped board, no switch is next to a gate, so
        he can never be standing on one at the moment they shut."""
        if not self.switch_mask:
            return False
        return bool(self.switch_mask & (state[1] | state[2] | state[3]))

    def won(self, state) -> bool:
        """``All Marker on Crate``. The other win condition, ``No Move``, holds
        on every settled board by construction. A state whose player has been
        deleted is not a win -- the engine's own guard says so."""
        return state[0] >= 0 and not (self.marker_mask & ~(state[1] | state[2]))

    # -- the press -----------------------------------------------------------
    def press(self, state, d):
        """Simulate one press. Returns the settled state, or None when the press
        is not modellable: it would overrun the interpreter's ``again`` cap, or
        its outcome genuinely depends on the ``random`` rules' roll.

        A returned state equal to ``state`` means the press did nothing.
        """
        key = (state, d)
        got = self._press_cache.get(key, 0)
        if got != 0:
            return got
        out = self.press_set(state, d)
        got = None if out is None or len(out) != 1 else next(iter(out))
        self._press_cache[key] = got
        return got

    def press_set(self, state, d):
        """The SET of boards this press can settle on, or None on an ``again``
        overrun.

        One board while the two ``random`` movement rules have a single match
        each -- which is every press with at most one crate flying per
        direction, i.e. everything but overlapping explosions. Where they do
        not, the iteration is fanned out over every choice and the frontier is
        deduped each tick, so the schedules that reconverge (nearly all of them)
        collapse straight back to one board instead of exploding.
        """
        frontier = [_Sim(self, state, d)]
        done = set()
        for _ in range(_MAX_AGAIN):
            # The single-board fast path. It is not an optimisation of a rare
            # case: every press in this game but an overlapping explosion stays
            # on it, and it is what keeps the search out of `_Sim.key` (five
            # bitmasks) and `_Sim.state` (three) on every one of the millions of
            # iterations a level's search runs.
            if len(frontier) == 1:
                kids = self._fan(frontier[0])
                if len(kids) == 1:
                    child, again = kids[0]
                    # The engine breaks its own `again` loop the moment the win
                    # holds, so a board that wins mid-animation settles there.
                    if not again or (not child.mv and self._won_sim(child)):
                        return {child.state()}
                    frontier = [child]
                    continue
            else:
                kids = [pair for sim in frontier for pair in self._fan(sim)]
            nxt: dict = {}
            for child, again in kids:
                if not again or (not child.mv and self._won_sim(child)):
                    done.add(child.state())
                else:
                    nxt.setdefault(child.key(), child)
            frontier = list(nxt.values())
            if not frontier:
                return done
        return None

    def _won_sim(self, sim) -> bool:
        """`won`, read off a live `_Sim` -- so the settle test never has to build
        the bitmasks of a board that is going to keep animating."""
        return sim.player >= 0 and all(
            m in sim.crates or m in sim.bombs for m in self.markers)

    @staticmethod
    def _fan(sim):
        """Run one iteration of ``sim``, fanning out over the ``random`` rules'
        choices. Returns ``[(sim, again), ...]`` -- one entry unless a rule had
        several matches.

        With at most one crate in flight and no explosion on the board, no
        ``random`` rule can have a second match this iteration (its matches are
        a subset of the moving crates, and nothing but an explosion adds a Move
        mid-iteration), so the sim is ticked IN PLACE and nothing is copied.
        """
        if len(sim.mv) <= 1 and not sim.expl:
            sim.sched = []
            sim.pos = 0
            return [(sim, sim.tick())]
        out = []
        stack = [[]]
        while stack:
            sched = stack.pop()
            child = sim.fork()
            child.sched = sched
            try:
                again = child.tick()
            except _NeedChoice as exc:
                stack.extend(exc.sched + [i] for i in range(exc.options))
                continue
            out.append((child, again))
        return out

    # -- walking -------------------------------------------------------------
    def walk_field(self, state):
        """``(dist, parent)`` from the player over the squares he may walk on:
        no wall, no pit, no closed gate, no item. Walking into a crate is a
        KICK, which is a macro, not a walk, so crates block here."""
        player = state[0]
        blocked = self.walk_blocked(state)
        if player < 0:
            return {}, {}                       # the player was deleted
        dist = {player: 0}
        parent: dict = {player: None}
        queue = deque([player])
        nb = self.nb
        while queue:
            cur = queue.popleft()
            for d in range(4):
                k = nb[cur][d]
                if k >= 0 and k not in blocked and k not in dist:
                    dist[k] = dist[cur] + 1
                    parent[k] = (cur, d)
                    queue.append(k)
        return dist, parent

    def walk_blocked(self, state) -> set:
        """The squares the player may not walk onto. Factored out so the search
        (`walk_field`) and the tie annotator (`walk_optsets`) can never disagree
        about where he may go."""
        blocked = self.walls | self.pits | _cells(state[1] | state[2] | state[3])
        if not self.walk_gates_open(state):
            blocked = blocked | self.gates
        return blocked

    def walk_dist_to(self, state, target) -> dict:
        """BFS distance to ``target`` over the same walk map."""
        blocked = self.walk_blocked(state)
        dist = {target: 0}
        queue = deque([target])
        while queue:
            cur = queue.popleft()
            for d in range(4):
                k = self.nb[cur][d]
                if k >= 0 and k not in blocked and k not in dist:
                    dist[k] = dist[cur] + 1
                    queue.append(k)
        return dist

    @staticmethod
    def walk_to(parent, cell) -> list:
        out = []
        while parent[cell] is not None:
            cell, d = parent[cell]
            out.append(d)
        out.reverse()
        return out

    # -- the relaxed game the heuristic is measured in ------------------------
    def _rays(self, cell, d):
        """``(stands, rests)`` for kicking the crate on ``cell`` in direction
        ``d``, in the RELAXED game -- walls and pits only, every other object
        assumed away.

        ``stands`` is every square the player could kick from. That is not just
        the one square behind the crate: a kick travels THROUGH a chain and
        launches the far end, so standing anywhere on the contiguous run behind
        it works as long as every square in between could hold a crate (not a
        wall, not a pit -- a crate cannot rest on a pit). Leaving the chain out
        would make the table over-estimate and cost admissibility.

        ``rests`` is every square the crate could come to rest on: along the ray
        until a wall, skipping pits (it flies over them, it just cannot stop
        there).
        """
        stands, rests = [], []
        back = (d + 2) % 4
        cur = cell
        while True:
            cur = self.nb[cur][back]
            if cur < 0 or cur in self.walls or cur in self.pits:
                break
            stands.append(cur)
        cur = cell
        while True:
            cur = self.nb[cur][d]
            if cur < 0 or cur in self.walls:
                break
            if cur not in self.pits:
                rests.append(cur)
        return stands, rests

    def kick_tables(self) -> dict:
        """``{marker: {cell: kicks to get a crate from cell onto marker}}``.

        Reverse BFS over `_rays`' relaxed kick, counting KICKS only. Relaxing
        where the crate stops (in the real game that is decided by whatever
        happens to be in the way) and which squares the player may occupy only
        ever ADDS moves, so the table is a lower bound. A cell in no table is a
        crate that can never help that marker again.
        """
        if self._kick_tables is not None:
            return self._kick_tables
        # Forward kick edges, then reverse them once for all markers.
        pred: dict = {}
        for cell in range(self.n):
            if cell in self.walls:
                continue
            for d in range(4):
                stands, rests = self._rays(cell, d)
                if not stands:
                    continue
                for y in rests:
                    pred.setdefault(y, set()).add(cell)
        tables = {}
        for target in self.markers:
            dist = {target: 0}
            queue = deque([target])
            while queue:
                cur = queue.popleft()
                cost = dist[cur] + 1
                for cell in pred.get(cur, ()):
                    if cell not in dist:
                        dist[cell] = cost
                        queue.append(cell)
            tables[target] = dist
        self._kick_tables = tables
        return tables

    def press_tables(self) -> dict:
        """``{marker: {(crate cell, player cell): PRESSES to park that crate on
        that marker}}``, in the relaxed game.

        This is the table that makes the search tractable: on these boards the
        kicks are a handful and the WALKING between them is the whole cost, so a
        kick-counting bound is flat over the entire approach and A* has nothing
        to steer with. Here the relaxed game keeps the player in the state and
        charges him for every step, which is the same lower bound at a hundred
        times the resolution.

        The relaxed game is exactly two unit-cost moves over ``(crate, player)``
        pairs, so the field is a plain BFS from the goal rather than a Dijkstra:

          * WALK -- the player steps to an adjacent square that is not a wall,
            not a pit and not the crate's own square. Other crates, closed gates
            and breakwalls are ignored, so this can only under-estimate.
          * KICK -- from a square in `_rays`' ``stands`` the crate lands on any
            of its ``rests``; the player does not move, which is the one place
            this game differs from a sokoban and the reason the player's square
            has to be in the state at all.

        Not built for a level with bombs: an explosion moves crates that nobody
        kicked, so no relaxation of *kicks* bounds the presses there.
        """
        if self._press_tables is not None:
            return self._press_tables
        n = self.n
        stand_ok = [k not in self.walls and k not in self.pits for k in range(n)]
        # Reverse adjacency of the relaxed game, built once for all markers.
        rev: list = [[] for _ in range(n * n)]
        for c in range(n):
            if c in self.walls:
                continue
            for d in range(4):
                stands, rests = self._rays(c, d)
                if not stands or not rests:
                    continue
                for s in stands:
                    frm = c * n + s
                    for y in rests:
                        rev[y * n + s].append(frm)     # kick: (c,s) -> (y,s)
            for p in range(n):
                if not stand_ok[p] or p == c:
                    continue
                node = c * n + p
                for d in range(4):
                    q = self.nb[p][d]
                    if q >= 0 and stand_ok[q] and q != c:
                        rev[c * n + q].append(node)    # walk: (c,p) -> (c,q)
        tables = {}
        for target in self.markers:
            dist: dict = {}
            queue = deque()
            for p in range(n):
                if stand_ok[p] and p != target:
                    dist[target * n + p] = 0
                    queue.append(target * n + p)
            while queue:
                cur = queue.popleft()
                cost = dist[cur] + 1
                for prev in rev[cur]:
                    if prev not in dist:
                        dist[prev] = cost
                        queue.append(prev)
            tables[target] = dist
        self._press_tables = tables
        return tables

    # -- heuristic -----------------------------------------------------------
    def heuristic(self, state) -> int:
        """A lower bound on the presses left (a GUIDE only where `certified` is
        False -- see that attribute).

        Two bounds, and the larger wins:

        * the sum over uncovered markers of the fewest KICKS any crate needs to
          reach it (reuse allowed, which is what keeps it a bound rather than an
          assignment);
        * for one chosen marker, the full relaxed PRESS distance from the board
          as it stands -- walking included -- plus the kick-only bound for every
          other uncovered marker. Sound because the presses it counts are
          disjoint: a press is either a walk or a kick, each marker is covered by
          a crate of its own, and one kick moves one crate.

        `_DEAD` when a marker has no crate that can ever reach it: the level is
        lost, and the search drops the state instead of expanding it.
        """
        if state[0] < 0:
            return _DEAD                        # the player was deleted
        player, crates, bombs = state[0], state[1], state[2]
        tables = self.kick_tables()
        pieces = None
        kicks: dict = {}
        for marker in self.markers:
            if (crates | bombs) >> marker & 1:
                continue
            if pieces is None:
                pieces = _cells(crates | bombs)
            table = tables[marker]
            best = None
            for cell in pieces:
                cost = table.get(cell)
                if cost is not None and (best is None or cost < best):
                    best = cost
            if best is None:
                return _DEAD
            kicks[marker] = best
        if not kicks:
            return 0
        total = sum(kicks.values())
        if not self.certified:
            return total                        # bombs: the guide, nothing more
        ptables = self.press_tables()
        n = self.n
        best_total = total
        for marker, own in kicks.items():
            table = ptables[marker]
            best = None
            for cell in pieces:
                cost = table.get(cell * n + player)
                if cost is not None and (best is None or cost < best):
                    best = cost
            if best is not None:
                best_total = max(best_total, best + total - own)
        return best_total

    def uniform(self, state) -> int:
        """The trivially admissible estimate: one press is needed unless the
        board is already won. It is what the optimality PROOF on a bomb level
        runs under -- `heuristic` cannot be trusted there, and this can."""
        return 0 if self.won(state) else 1

    # -- search --------------------------------------------------------------
    def successors(self, state):
        """Every ``walk to a kick square, then kick`` macro, as
        ``(presses, next_state)``.

        A kick square is the square BEHIND a target, and a target that is not
        reachable from behind cannot be kicked in that direction at all. Presses
        that the model refuses (`press` returned None) and presses that change
        nothing are dropped here rather than queued.
        """
        dist, parent = self.walk_field(state)
        crates, bombs, breaks = state[1], state[2], state[3]
        out = []
        for cell in sorted(_cells(crates | bombs | breaks)):
            for d in range(4):
                stand = self.nb[cell][(d + 2) % 4]
                if stand < 0 or stand not in dist:
                    continue
                walk = self.walk_to(parent, stand)
                here = (stand, crates, bombs, breaks)
                nxt = self.press(here, d)
                if nxt is None or nxt == here:
                    continue
                out.append((walk + [d], nxt))
        return out

    def kick_bound(self, state) -> int:
        """The kick-only lower bound: the sum over uncovered markers of the
        fewest kicks any crate needs to reach one. `_DEAD` when a marker has
        none. This is `heuristic`'s first half, used on its own by
        `solve_kicks`, whose ``g`` counts KICKS."""
        crates, bombs = state[1], state[2]
        if state[0] < 0:
            return _DEAD
        tables = self.kick_tables()
        pieces = None
        total = 0
        for marker in self.markers:
            if (crates | bombs) >> marker & 1:
                continue
            if pieces is None:
                pieces = _cells(crates | bombs)
            table = tables[marker]
            best = None
            for cell in pieces:
                cost = table.get(cell)
                if cost is not None and (best is None or cost < best):
                    best = cost
            if best is None:
                return _DEAD
            total += best
        return total

    def region(self, state) -> int:
        """The canonical id of the player's walkable region: its smallest square,
        or -1 if he has been deleted. Two boards with the same pieces and the
        same region admit exactly the same kicks."""
        dist, _parent = self.walk_field(state)
        return min(dist) if dist else -1

    def solve_kicks(self, state, node_cap: int = 400_000):
        """A WINNING press sequence, found by searching KICKS instead of presses.

        The fallback for the levels the exact search cannot close. Two things
        change and they compound:

          * ``g`` counts kicks, so the depth is six or eight rather than thirty,
            and the walking -- which is most of a plan's length and none of its
            difficulty -- costs the search nothing;
          * states are deduped on ``(pieces, player REGION)`` instead of on the
            player's exact square. That merge is unsound when ``g`` counts
            presses (it mis-costs by up to the region's diameter, the
            entrepotphage lesson) and is exactly right when ``g`` counts kicks:
            the player can walk anywhere inside his region for free as far as
            *which kicks exist* is concerned.

        So the plan is winning and engine-verified but NOT shortest -- the walk
        between two kicks is whatever the shortest path happens to be, and the
        kick order was chosen without regard to it. `--plans` says so for every
        level that comes from here.
        """
        if self.won(state):
            return []
        h0 = self.kick_bound(state)
        if h0 >= _DEAD:
            return None
        counter = 0
        pq = [(h0, 0, counter, state, [])]
        seen = {(self.region(state), state[1], state[2], state[3]): 0}
        nodes = 0
        while pq:
            _f, g, _c, cur, path = heapq.heappop(pq)
            for presses, nxt in self.successors(cur):
                nodes += 1
                if nodes >= node_cap:
                    return None
                plan = path + presses
                if self.won(nxt):
                    return plan
                hn = self.kick_bound(nxt)
                if hn >= _DEAD:
                    continue
                key = (self.region(nxt), nxt[1], nxt[2], nxt[3])
                ng = g + 1
                if seen.get(key, 1 << 30) <= ng:
                    continue
                seen[key] = ng
                counter += 1
                heapq.heappush(pq, (ng + hn, ng, counter, nxt, plan))
        return None

    def solve_beam(self, state, width: int = 400, depth: int = 24,
                   node_cap: int = 400_000):
        """A WINNING press sequence, found by a width-capped breadth-first beam
        over the same kick macros. The last resort, for the levels neither exact
        A* nor `solve_kicks` can reach.

        WHY a beam and not more nodes. Both searches pay the same thing per node
        -- one press simulation per candidate kick -- and A* spends that budget
        on the frontier its estimate likes. On these boards the estimate is flat
        over the middle of a solution: nine crates in a block, three markers
        behind a wall of pits, and a dozen kicks that rearrange the block without
        any marker getting closer. Over that stretch weighting the estimate
        changes nothing and A* degenerates to uniform-cost at a depth where that
        is hopeless. A beam spends the budget on breadth at every depth instead
        and only uses the estimate to break ties, which is all a flat estimate is
        good for.

        The trade is length: a beam plan wanders where A* would not. ``--plans``
        marks every level that comes from here as winning-but-not-shortest.
        """
        if self.won(state):
            return []
        rank = self.heuristic if self.certified else self.kick_bound
        frontier = [(state, [])]
        seen = {(self.region(state), state[1], state[2], state[3])}
        nodes = 0
        for _depth in range(depth):
            # Deduped WITHIN the layer, and only the survivors are written to
            # `seen`. Marking every generated child instead -- which is what a
            # beam usually does -- burns the states the width had no room for,
            # permanently, and on level 13 that closed the whole space by depth
            # 17 with the beam having expanded a fifth of what it had blocked.
            kids: dict = {}
            for cur, path in frontier:
                for presses, nxt in self.successors(cur):
                    nodes += 1
                    if nodes >= node_cap:
                        return None
                    plan = path + presses
                    if self.won(nxt):
                        return plan
                    if self.kick_bound(nxt) >= _DEAD:
                        continue
                    key = (self.region(nxt), nxt[1], nxt[2], nxt[3])
                    if key in seen:
                        continue
                    old = kids.get(key)
                    if old is not None and old[1] <= len(plan):
                        continue
                    kids[key] = (rank(nxt), len(plan), nxt, plan)
            if not kids:
                return None                      # the reachable space closed
            best = sorted(kids.items(), key=lambda kv: (kv[1][0], kv[1][1]))
            frontier = []
            for key, (_h, _g, nxt, plan) in best[:width]:
                seen.add(key)
                frontier.append((nxt, plan))
        return None

    def solve(self, state, node_cap: int = 400_000, bound: int | None = None,
              hfn=None):
        """Shortest press sequence from ``state``, or None. Returns ``(plan,
        closed)`` where ``closed`` says the search exhausted its frontier rather
        than running out of nodes -- which, under an admissible ``hfn``, is the
        proof that the plan is shortest.

        Macro A* with ``g`` in PRESSES. A win is kept as an INCUMBENT and the
        search runs until the frontier's ``f`` reaches its cost: macros cost
        different amounts (a macro is ``walk + kick``), so the first win
        generated is not the cheapest one -- the flaw
        `PSPushExpert.exact_goal_test` exists to fix.

        ``bound`` prunes every node with ``g + h > bound`` and is what makes the
        optimal-set measurement affordable: "is there a plan of length at most
        ``r``" is far cheaper to answer than "what is the shortest".
        """
        if hfn is None:
            hfn = self.heuristic
        if self.won(state):
            return [], True
        h0 = hfn(state)
        if h0 >= _DEAD or (bound is not None and h0 > bound):
            return None, True
        counter = 0
        pq = [(h0, 0, counter, state, [])]
        best_g = {state: 0}
        best_plan = None
        best_cost = bound + 1 if bound is not None else 1 << 30
        nodes = 0
        while pq:
            f, g, _c, cur, path = heapq.heappop(pq)
            if f >= best_cost:
                return best_plan, True
            for presses, nxt in self.successors(cur):
                nodes += 1
                if nodes >= node_cap:
                    return best_plan, False
                ng = g + len(presses)
                if ng >= best_cost:
                    continue
                if self.won(nxt):
                    best_cost, best_plan = ng, path + presses
                    continue
                if best_g.get(nxt, 1 << 30) <= ng:
                    continue
                hn = hfn(nxt)
                if hn >= _DEAD or ng + hn >= best_cost:
                    continue
                best_g[nxt] = ng
                counter += 1
                heapq.heappush(pq, (ng + hn, ng, counter, nxt, path + presses))
        return best_plan, True

    def prove(self, state, cost: int, node_cap: int = 400_000) -> bool:
        """True when nothing shorter than ``cost`` presses wins from ``state``.

        Runs the same search under `uniform` -- the estimate that is admissible
        whatever the board holds -- bounded at ``cost - 1``. A closed frontier
        with no plan is the proof; a node-cap hit is not, and says so by
        returning False. Only needed where `certified` is False; elsewhere
        `heuristic` is itself admissible and `solve` closing is already the
        proof.
        """
        plan, closed = self.solve(state, node_cap, bound=cost - 1,
                                  hfn=self.uniform)
        return closed and plan is None

    # -- optimal-action sets -------------------------------------------------
    def walk_optsets(self, state, plan):
        """Per-step SETS for a plan that is winning but NOT shortest -- the ones
        the fallback searches return.

        There is no meaningful "every equally shortest press" for a plan that is
        not shortest, so the question this answers is the smaller, still true
        one: given the KICK sequence the plan commits to, which presses are
        equally good? A kick is labelled with itself, and each step of a WALK is
        labelled with every direction that keeps it on a shortest route to the
        square the next kick is taken from -- no rule in this game fires on a
        bare step onto an empty square (the switch changes the gates, but not the
        map the player walks on, because the gates it opens shut again the moment
        he leaves), so any interleaving of the two axes leaves an identical
        board. Labelling one arbitrary interleaving as the only right answer
        would train the policy to a coin flip it cannot win.

        Classification watches the BOARD rather than a macro boundary the
        flattened plan no longer carries: a press that moved a piece is a kick, a
        press that moved only the player is a walk.
        """
        steps = []
        cur = state
        for d in plan:
            nxt = self.press(cur, d)
            kind = ("kick" if nxt[1:] != cur[1:]
                    else "walk" if nxt[0] != cur[0] else "stuck")
            steps.append((kind, cur, d))
            cur = nxt
        out: list = [None] * len(plan)
        i = 0
        while i < len(steps):
            if steps[i][0] != "walk":
                out[i] = [steps[i][2]]
                i += 1
                continue
            run = i
            while i < len(steps) and steps[i][0] == "walk":
                i += 1
            if i >= len(steps):
                # A trailing walk kicks nothing, so there is no destination to
                # measure it against: keep the recorded choice.
                for j in range(run, i):
                    out[j] = [steps[j][2]]
                continue
            stand = steps[i][1][0]
            dist = self.walk_dist_to(steps[run][1], stand)
            for j in range(run, i):
                here = steps[j][1][0]
                d0 = dist.get(here)
                alts = [] if d0 is None else [
                    d for d in range(4)
                    if dist.get(self.nb[here][d]) == d0 - 1]
                # The recorded step is on a shortest route by construction, so a
                # disagreement means the reconstruction drifted -- fall back to
                # labelling what the expert did.
                out[j] = alts if steps[j][2] in alts else [steps[j][2]]
        return out

    def optimal_sets(self, state, plan, node_cap: int = 400_000, hfn=None):
        """Per-step optimal SETS for ``plan``, measured by re-solving.

        At each step every press is taken and the board it lands on is re-solved
        under a bound of ``remaining - 1``; the press is optimal exactly when a
        plan that short exists. Two sound prunes keep it affordable and neither
        is a guess: a press that leaves the board untouched cannot be on a
        shortest path (it spends a move to reach the state it came from), and
        neither can one whose ADMISSIBLE estimate already exceeds what is left.

        ``hfn`` must be admissible or the measurement is not one; a bomb level
        therefore passes `uniform`. A step whose measurement disagrees with the
        plan is labelled with the press the expert took, never with nothing --
        an unlabelled step trains on literally nothing.
        """
        if hfn is None:
            hfn = self.heuristic if self.certified else self.uniform
        memo: dict = {}
        out = []
        cur = state
        for i, taken in enumerate(plan):
            remaining = len(plan) - i
            best = []
            for d in range(4):
                nxt = self.press(cur, d)
                if nxt is None or nxt == cur:
                    continue
                if self.won(nxt):
                    if remaining == 1:
                        best.append(d)
                    continue
                if remaining == 1 or hfn(nxt) > remaining - 1:
                    continue
                got = memo.get(nxt, 0)
                if got == 0:
                    sub, closed = self.solve(nxt, node_cap, bound=remaining - 1,
                                             hfn=hfn)
                    got = None if (sub is None or not closed) else len(sub)
                    memo[nxt] = got
                if got == remaining - 1:
                    best.append(d)
            # The recorded step ties with itself by construction; a disagreement
            # means the two sides were measured differently, so fall back to
            # labelling what the expert actually did.
            out.append(sorted(best) if taken in best else [taken])
            cur = self.press(cur, taken)
        return out


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

#: Where the shipped plans live. The searches ARE the cost of generation here --
#: they are seed-independent, so without a file on disk every
#: `parallelize_generator` shard re-derives all eighteen of them.
_PLAN_CACHE = (Path(__file__).resolve().parent.parent
               / "data" / "sokoslam_plans.json")

#: Which search rung produced each level's plan, and whether it is proved
#: shortest. See `SokoslamExpert.setup`.
_NOTE_CACHE = _PLAN_CACHE.with_name("sokoslam_notes.json")


class SokoslamExpert(PSExpert):
    """`PSExpert`'s plan memo, snapshot discipline and disk cache around the
    native `_Board` search.

    `_search` is replaced the way the other native-model ps: generators replace
    it: the base class keeps the memo, the restore discipline, the level scoping
    and the on-disk plan cache, and only the strategy underneath changes. The
    interpreter is never stepped by the planner -- `_Board` is built from the
    engine's grid and everything after that is the model -- so `heuristic`
    asserts rather than returning a number nothing would use.
    """

    directions = list(_DIRS)
    plan_cache_path = _PLAN_CACHE

    #: Budget for the two fallback searches. Larger than `node_cap` because
    #: their nodes are cheaper: `solve_kicks` and `solve_beam` dedup on the
    #: player's REGION, so they hold far fewer states for the same coverage.
    fallback_cap: int = 1_500_000
    #: Beam width / depth for the last-resort search, sized from the boards it
    #: is for: level 15's twelve crates give ~30 kicks per state, so one layer
    #: costs ~90k press simulations.
    beam_width: int = 3_000
    beam_depth: int = 30

    def setup(self) -> None:
        #: Which rung of the ladder each level's plan came from, and whether it
        #: was proved shortest. Kept BESIDE the plan cache rather than inside it
        #: (`PSExpert`'s on-disk format has no room for it) so ``--plans`` tells
        #: the same story on a warm cache as on a cold one -- otherwise a report
        #: run after the first would silently claim every plan was optimal.
        self.notes: dict = self._load_notes()
        self.report: dict = {}
        self._last_note = None

    def _load_notes(self) -> dict:
        try:
            return {int(k): v for k, v in json.loads(_NOTE_CACHE.read_text()).items()}
        except Exception:                                    # noqa: BLE001
            return {}

    def _save_notes(self) -> None:
        """Written atomically, like the plan cache: shards started together
        would otherwise interleave into a truncated file."""
        try:
            _NOTE_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _NOTE_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self.notes[k] for k in sorted(self.notes)}, indent=1))
            os.replace(tmp, _NOTE_CACHE)
        except OSError:
            pass                                             # the notes are optional

    def plan(self, eng, level: int | None = None):
        self._last_note = None
        found = super().plan(eng, level)
        if level is not None:
            if self._last_note is not None:
                self.notes[level] = self._last_note
                self._save_notes()
            self.report[level] = self.notes.get(level, {"how": "cache"})
        return found

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "SokoslamExpert plans on the native model; heuristic is unused")

    def board(self, eng) -> _Board:
        return _Board(eng, self.g)

    def _search(self, eng):
        """The search LADDER, cheapest and strongest first. Each rung is tried
        only when the one above it came back empty, so every level ships the best
        plan this file can produce and `--plans` reports which rung it came from:

          1. exact macro A* (`_Board.solve`) -- proved shortest, exact tie sets;
          2. `_Board.solve_kicks` -- winning, region-deduped, kick-costed;
          3. `_Board.solve_beam` -- winning, width-capped breadth-first.

        Rungs 2 and 3 return plans that are not shortest, so their steps are
        labelled by `_Board.walk_optsets` (which presses are equally good GIVEN
        the kick sequence) rather than by the measured re-solve, which has no
        meaning for a plan that is not shortest.
        """
        board = self.board(eng)
        note = {"how": "A*", "shortest": False}
        t0 = time.time()
        plan, closed = board.solve(board.start, self.node_cap)
        if plan is not None:
            # A closed frontier under an admissible estimate proves the plan
            # shortest; a bomb level's estimate is not admissible, so it is
            # proved (or not) by the bounded uniform-cost sweep instead.
            note["shortest"] = bool(
                closed and (board.certified
                            or board.prove(board.start, len(plan),
                                           self.node_cap)))
        if plan is None:
            note["how"] = "kicks"
            plan = board.solve_kicks(board.start, self.fallback_cap)
        if plan is None:
            note["how"] = "beam"
            plan = board.solve_beam(board.start, self.beam_width,
                                    self.beam_depth, self.fallback_cap)
        note["plan"] = time.time() - t0
        if plan is None:
            note["how"] = "none"
            self._last_note = note
            return None
        t0 = time.time()
        sets = (board.optimal_sets(board.start, plan, self.node_cap)
                if note["shortest"] else board.walk_optsets(board.start, plan))
        note["ties"] = time.time() - t0
        self._last_note = note
        return Plan([_DIRS[d] for d in plan],
                    [[_DIRS[x] for x in s] for s in sets])


class SokoslamSolver(PSAStarSolver):
    game_id = "puzzlescript_sokoslam"
    game_name = GAME_NAME
    expert_cls = SokoslamExpert

    #: Record against the GAME FOLDER's adapter. ``games/ps:sokoslam`` is a plain
    #: passthrough today; it is named anyway so a sprite patch or a step cap
    #: added there later cannot silently make this generator tape a game nobody
    #: plays (the ps:count_mover trap).
    game_module_id = "ps:sokoslam"

    #: A MEMORY budget as much as a time one -- every queued node carries a
    #: state and its path. Sized from the boards: the levels this closes on all
    #: do so well inside it, and the ones it does not are in `skip_levels`.
    node_cap = 400_000

    #: Levels the macro A* cannot close within that budget. Skipped up front so
    #: `discover_solvable` does not burn the whole cap per level at every
    #: startup. See ``--plans`` for what each of them costs.
    skip_levels = frozenset(_UNSOLVED)

    #: Room for the longest plan plus the RESET exploration prefix and the
    #: re-plan after it, inside the adapter's own 200-press per-level budget.
    max_steps = 200


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = SokoslamSolver()
    game = solver.make_game(seed)
    return solver, game, SokoslamExpert(game, node_cap=SokoslamSolver.node_cap)


def _show(board: _Board, state) -> str:
    """The board as ASCII, in the model's own reading of it -- so a bug in the
    parse shows up as a picture that does not match the .txt."""
    player = state[0]
    crates, bombs, breaks = (_cells(state[1]), _cells(state[2]),
                             _cells(state[3]))
    gopen = board.gates_open(state)
    out = []
    for r in range(board.h):
        row = []
        for c in range(board.w):
            k = r * board.w + c
            if k in board.walls:
                ch = "#"
            elif k == player:
                ch = "@"
            elif k in crates:
                ch = "*" if k in board.markers else "o"
            elif k in bombs:
                ch = "%" if k in board.markers else "c"
            elif k in breaks:
                ch = "k"
            elif k in board.gates:
                ch = "-" if gopen else "g"
            elif k in board.markers:
                ch = "e"
            elif k in board.switches:
                ch = "f"
            elif k in board.pits:
                ch = "~"
            else:
                ch = "."
            row.append(ch)
        out.append("".join(row))
    return "\n".join(out)


def _levels() -> int:
    """Every board, as the model reads it, with its inventory."""
    _solver, game, expert = _new()
    eng = game._engine
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        print(f"=== level {level}: {board.h}x{board.w}, "
              f"{_popcount(board.start[1])} crates, {_popcount(board.start[2])} bombs, "
              f"{_popcount(board.start[3])} breakwalls, "
              f"{len(board.markers)} markers, "
              f"{len(board.pits)} pit squares, {len(board.switches)} switches, "
              f"{len(board.gates)} gates"
              + (", SWITCH TOUCHES A GATE" if board.switch_touches_gate else ""))
        print(_show(board, board.start))
    return 0


def _plans() -> int:
    """Per-level plan length, whether it is PROVED shortest, and tie coverage."""
    _solver, game, expert = _new()
    eng = game._engine
    total = tied = steps = 0
    solved = proved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        head = (f"level {level:2d}: {board.h:2d}x{board.w:2d}, "
                f"{_popcount(board.start[1]):2d} crates + {_popcount(board.start[2])} bombs "
                f"+ {_popcount(board.start[3])} breakwalls, "
                f"{len(board.markers)} markers")
        if level in SokoslamSolver.skip_levels:
            print(f"{head} -- SKIPPED (search does not close; see the docstring)")
            continue
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        if found is None:
            print(f"{head} -- NO PLAN")
            continue
        solved += 1
        note = expert.report.get(level, {})
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        tied += ties
        steps += len(found)
        shortest = note.get("shortest", True)
        proved += int(bool(shortest))
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(found):3d} presses via {note.get('how', 'cache'):5s} "
              f"({'PROVED SHORTEST' if shortest else 'winning, not shortest'}, "
              f"budget {game._max_steps} {room}), "
              f"{ties:3d} steps with a tie set "
              f"({ties / max(1, len(found)):3.0%}), {dt:6.2f}s")
    print(f"{solved} levels solved, {proved} proved shortest, "
          f"{total} presses total, {tied}/{steps} steps with a tie "
          f"({tied / max(1, steps):.0%})")
    return 0 if solved else 1


def _fuzz(trials: int = 60, steps: int = 40, seed: int = 7) -> int:
    """Differential test: drive the native model and the real interpreter side
    by side and require them to agree on every square, every press.

    Two sources of presses, because neither alone is enough (the escaping_limbo
    lesson -- random play reported a matching model while having exercised
    neither a gate closing on a crate nor a chain of explosions):

      * RANDOM play from each level's start, which is what reaches the odd
        corners of the board;
      * every PREFIX of each level's own shipped plan, followed by random play,
        which is what reaches the states the training data actually contains.

    Checked after every press: the four dynamic object sets, the open/closed
    state of every gate, and that the interpreter's board carries NO residue --
    no leftover Move, CrateRest, PlayerRest, Explosion, CrateFall or GateOpening,
    any of which would mean the ``again`` chain was cut short and the model and
    the engine had silently diverged in a way the object sets cannot show.

    Also prints COVERAGE counters, because "the model matches" is worth nothing
    if the run never made a crate fall, never broke a wall and never let a gate
    shut.
    """
    solver, game, expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    # Objects that must never survive a press. The four directional player
    # sprites are in the list on purpose: they are the kick animation, and the
    # three "return the player to normal" rules retire them inside the same
    # iteration -- which is what makes `--audit`'s single ``player`` composition
    # the whole truth, and what makes the mirror argument below decidable at all.
    residue = ("moveup", "movedown", "moveleft", "moveright", "craterest",
               "playerrest", "gateopening",
               "playerup", "playerdown", "playerleft", "playerright")
    #: These three CAN survive, but only on the frame that wins the level: the
    #: engine breaks its `again` loop the moment the win holds. Checked
    #: separately, and audited.
    win_only = ("explosion", "cratefall", "breakdebris1", "breakdebris2",
                "breakdebris3")

    def read(w):
        crates, bombs, breaks, player = set(), set(), set(), -1
        junk, leftover, gopen = [], [], set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                k = r * w + c
                if idx["cratebase"] in cell:
                    crates.add(k)
                if idx["bomb"] in cell:
                    bombs.add(k)
                if idx["breakwall"] in cell:
                    breaks.add(k)
                if idx["gateopen"] in cell:
                    gopen.add(k)
                for name in ("playerstand", "playerup", "playerdown",
                             "playerleft", "playerright"):
                    if idx[name] in cell:
                        player = k
                for name in residue:
                    if idx[name] in cell:
                        junk.append((r, c, name))
                for name in win_only:
                    if idx[name] in cell:
                        leftover.append((r, c, name))
        return ((player, _mask(crates), _mask(bombs), _mask(breaks)),
                frozenset(gopen), junk, leftover)

    rng = random.Random(seed)
    cov: dict = {}

    def bump(name):
        cov[name] = cov.get(name, 0) + 1

    bad = 0
    t0 = time.time()
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        plan = None
        if level not in SokoslamSolver.skip_levels:
            found = expert.plan(eng, level)
            plan = None if found is None else [_DIRS.index(x) for x in found]
        runs = [[] for _ in range(trials)]
        if plan:
            runs += [plan[:i] for i in range(len(plan) + 1)]
        for prefix in runs:
            game.set_level(level)
            state = board.start
            seq = list(prefix) + [rng.randrange(4) for _ in range(steps)]
            for d in seq:
                model = board.press(state, d)
                eng.step(_DIRS[d])
                real, gopen, junk, leftover = read(board.w)
                bump("press")
                if model is None:
                    print(f"  level {level}: model REFUSED {_DIRS[d]} "
                          f"(again overrun or ambiguous roll)")
                    bad += 1
                    break
                if model != real:
                    print(f"  level {level}: press {_DIRS[d]} MISMATCH\n"
                          f"    model {_show(board, model)}\n"
                          f"    real  {_show(board, real)}")
                    bad += 1
                    break
                if junk:
                    print(f"  level {level}: press {_DIRS[d]} left residue "
                          f"{junk} -- the `again` chain was cut short")
                    bad += 1
                    break
                if leftover:
                    # The engine breaks its own `again` loop the moment the win
                    # holds, so the WINNING press can freeze the animation
                    # part-way and leave an Explosion, a CrateFall or a
                    # BreakDebris on the final frame. That is a real frame the
                    # agent sees (``--audit`` covers those compositions); on any
                    # other press it would mean the two had parted company.
                    if board.won(model) and eng.check_win():
                        bump("win frame keeps the animation's last objects")
                    else:
                        print(f"  level {level}: press {_DIRS[d]} left "
                              f"{leftover} on a board that is not won")
                        bad += 1
                        break
                want = board.gates if board.gates_open(model) else frozenset()
                if want != gopen:
                    print(f"  level {level}: press {_DIRS[d]} gate state "
                          f"model {sorted(want)} real {sorted(gopen)}")
                    bad += 1
                    break
                if model == state:
                    bump("no-op press")
                if model[0] != state[0]:
                    bump("player walked")
                if _popcount(model[1] | model[2]) < _popcount(state[1] | state[2]):
                    bump("crate destroyed")
                if _popcount(model[2]) < _popcount(state[2]):
                    bump("bomb detonated")
                if _popcount(model[3]) < _popcount(state[3]):
                    bump("breakwall broken")
                if board.switches and board.gates_open(model) != board.gates_open(state):
                    bump("gate toggled")
                if model[0] < 0:
                    bump("player destroyed")
                    break
                state = model
                if board.won(state):
                    bump("win")
                    break
            if bad:
                break
        if bad:
            break
    print(f"coverage: " + ", ".join(f"{k} {v}" for k, v in sorted(cov.items())))
    print(f"{cov.get('press', 0)} presses in {time.time() - t0:.1f}s -- "
          + ("model agrees with the interpreter everywhere"
             if not bad else f"FUZZ FAILED: {bad} disagreements"))
    return 0 if not bad else 1


def _ties() -> int:
    """Re-derive every optimal-action label the independent way, with the check
    that matches how the label was earned.

    On a level whose plan is PROVED SHORTEST the claim is "these are exactly the
    presses that still finish in the moves remaining", and it is checked against
    a fresh bounded search from the board each candidate press lands on -- so
    the labels are compared with a new search from every state on the path
    rather than with the search that produced them.

    On a level whose plan is merely winning the claim is smaller, and so is the
    check: every claimed press must be a bare WALK that ends one step closer to
    the square the plan's next kick is taken from, measured by a fresh BFS. That
    is exactly the property `_Board.walk_optsets` asserts and nothing more; a
    "shortest" test there would be measuring a claim nobody made.

    Both modes also require the press the expert TOOK to be in its own set --
    a step whose label does not contain what was done is a step that trains the
    policy against the trajectory it is looking at.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        if level in SokoslamSolver.skip_levels:
            continue
        game.set_level(level)
        board = expert.board(eng)
        found = expert.plan(eng, level)
        if found is None:
            print(f"level {level}: no plan (skipped)")
            continue
        shortest = bool(expert.report.get(level, {}).get("shortest"))
        hfn = board.heuristic if board.certified else board.uniform
        plan = [_DIRS.index(x) for x in found]

        # Where the plan's next kick is taken from, per step -- the destination
        # a walk run is measured against in the not-shortest mode.
        states = [board.start]
        for d in plan:
            states.append(board.press(states[-1], d))
        stand_of = [None] * len(plan)
        nxt_kick = None
        for pi in range(len(plan) - 1, -1, -1):
            if states[pi][1:] != states[pi + 1][1:]:
                nxt_kick = states[pi][0]        # this press was a kick
            stand_of[pi] = nxt_kick

        checked = 0
        for pi, (taken, claimed) in enumerate(zip(plan, found.optsets)):
            state = states[pi]
            remaining = len(plan) - pi
            measured = []
            if shortest:
                for d in range(4):
                    after = board.press(state, d)
                    if after is None or after == state:
                        continue
                    if board.won(after):
                        if remaining == 1:
                            measured.append(_DIRS[d])
                        continue
                    sub, closed = board.solve(after, expert.node_cap,
                                              bound=remaining - 1, hfn=hfn)
                    if closed and sub is not None and len(sub) == remaining - 1:
                        measured.append(_DIRS[d])
                if not measured:
                    measured = [_DIRS[taken]]
            elif states[pi][1:] != states[pi + 1][1:] or stand_of[pi] is None:
                measured = [_DIRS[taken]]       # a kick, or a trailing walk
            else:
                dist = board.walk_dist_to(state, stand_of[pi])
                here = state[0]
                d0 = dist.get(here)
                measured = sorted(
                    (_DIRS[d] for d in range(4)
                     if d0 is not None and dist.get(board.nb[here][d]) == d0 - 1),
                    key=_DIRS.index)
                if not measured:
                    measured = [_DIRS[taken]]
            if sorted(measured, key=_DIRS.index) != sorted(claimed, key=_DIRS.index):
                print(f"  level {level} step {pi}: labelled {list(claimed)} "
                      f"but measured {measured}")
                bad += 1
            if _DIRS[taken] not in claimed:
                print(f"  level {level} step {pi}: took {_DIRS[taken]}, not in "
                      f"its own optimal set {list(claimed)}")
                bad += 1
            checked += 1
        print(f"level {level}: {checked} steps verified "
              f"({'exact' if shortest else 'walk-order'} mode)")
    print("ties clean" if not bad else f"TIES FAILED: {bad} mismatches")
    return 0 if not bad else 1


def _verify() -> int:
    """Replay every shipped plan through the REAL interpreter and require a WIN.

    This is what makes the native model safe to plan on: whatever `_Board`
    believes, the press sequence that ships is the one the engine was driven
    with, and a level only counts when the engine says the level is over.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        if level in SokoslamSolver.skip_levels:
            continue
        game.set_level(level)
        found = expert.plan(eng, level)
        if found is None:
            print(f"level {level}: NO PLAN")
            bad += 1
            continue
        game.set_level(level)
        won_at = None
        for i, d in enumerate(found):
            eng.step(d)
            if eng.check_win():
                won_at = i + 1
                break
        if won_at is None:
            print(f"level {level}: {len(found)} presses replayed, NO WIN")
            bad += 1
        elif won_at != len(found):
            print(f"level {level}: won after {won_at} of {len(found)} presses "
                  f"-- the plan is longer than it needs to be")
            bad += 1
        else:
            print(f"level {level}: {len(found):3d} presses -> WIN")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} levels")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every cell size the boards use.

    Whole 64x64 frames of UNIFORM boards are compared rather than one cell cut
    out of a mixed board: `_render_frame` centre-pads a non-square board, so
    indexing a cell by ``cell_px * r`` silently reads the wrong pixels on the
    wide levels (the ps:explod lesson). Two uniform boards render identically
    iff their cells do.

    The compositions are the ones the model says a settled board can hold, not a
    guess. The transient objects (Move markers, CrateRest, Explosion, CrateFall,
    the three BreakDebris stages, GateOpening) are deliberately absent: every one
    of them is created and consumed inside a single press, so no frame the
    recording tapes can contain one.

    Every board in this game is between 5x8 and 11x11, so the cell size is
    5px or more everywhere and no sprite row is ever dropped by the sampler --
    the ps:snakeoban trap does not arise here, and this report is what says so.
    """
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, _expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    comps: dict = {
        "floor": (),
        "wall_top": ("walltop",),
        "wall_base": ("wallbase",),
        "pit_top": ("pittop",),
        "pit_bottom": ("pitbottom",),
        "marker": ("marker",),
        "switch": ("switch",),
        "gate_closed": ("gateclosed",),
        "gate_open": ("gateopen",),
        "breakwall": ("breakwall",),
        "crate": ("cratebase",),
        "bomb": ("bomb",),
        "crate_on_marker": ("marker", "onmarker", "cratebase"),
        "bomb_on_marker": ("marker", "onmarker", "bomb"),
        "crate_on_switch": ("switch", "cratebase"),
        "crate_on_gate_open": ("gateopen", "cratebase"),
        "player": ("playerstand",),
        "player_on_marker": ("marker", "playerstand"),
        "player_on_switch": ("switch", "playerstand"),
        "player_on_gate_open": ("gateopen", "playerstand"),
        # Win-frame only. The engine breaks its `again` loop the moment the win
        # holds, so the last press of a level can freeze the animation with one
        # of these still on the board -- and unlike every other transient object
        # (the Move markers, CrateRest, GateOpening), they can therefore reach a
        # frame the recording tapes. They are audited for the same reason the
        # settled compositions are: a dynamics model has to be able to tell that
        # frame from any other.
        "explosion": ("explosion",),
        "crate_falling": ("cratefall",),
        "debris_1": ("breakdebris1",),
        "debris_2": ("breakdebris2",),
        "debris_3": ("breakdebris3",),
    }

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
        note = "OK" if not clashes else f"IDENTICAL {clashes}"
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} "
              f"compositions -- {note}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--levels" in sys.argv:
        sys.exit(_levels())
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(SokoslamSolver.main())
