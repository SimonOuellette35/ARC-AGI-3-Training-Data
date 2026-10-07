"""Generate Phase-1 training data for the PuzzleScript game ps:fireproof_bomber
("Fireproof bomber", MarcinK -- Bomberman where the crates are the lock and the
exit is made of the same stuff as everything else you blow up).

The harness -- the rotation contract, the trajectory recorder, the plan cache and
the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a native model of the
push / fuse / fireball ruleset, and the macro search over it.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_fireproof_bomber",
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
exactly. Every step also carries a set of equally-optimal presses.

The game
--------
The exit opens when the board holds NO crates, and the only thing that removes a
crate is fire. X drops a bomb under you and spends one round from the armoury;
the armoury is a wall-in numeral that doubles as the ammo counter, and on the
levels where you can walk into it, bumping it reloads one round per press.

FIVE RULES THAT SHAPE EVERY PLAN, none of them obvious from the art:

  * THE PUSH IS TELEKINETIC. ``[> Player | Crate] -> [Player | > Crate]`` gives
    the crate the force and takes it off the player, so shoving a crate does NOT
    move you into the cell it left -- that costs a second press. A crate with
    anything behind it does not budge, and neither do you.
  * A BOMB DOES NOT START TICKING UNTIL YOU STEP OFF IT. It is lit by
    ``late [Player | BombPlaced]``, i.e. by you standing NEXT to it, and then
    burns for exactly three more presses. So the whole fuse is: X, step away,
    two more presses, and the fireball is on the board on the press after that.
  * THE FIREBALL IS A 3x3 THAT GROWS INTO IT. On the press it appears it damages
    the centre and the four orthogonal neighbours (walls block those arms); on
    the next press it damages the full 3x3; on the third only the centre. Walls
    stop it, crates do not -- they just burn.
  * EVERYTHING BURNS, INCLUDING THE THINGS YOU NEED. Fire turns TNT into a
    fireball of its own one press later (which is what makes a chain crawl
    across the board instead of going off at once), and it deletes the ARMOURY
    and the EXIT the same way. Blowing up the exit is a silent, permanent loss:
    the win condition simply can never hold again.
  * YOU ARE FIREPROOF EXACTLY ONCE. Fire chars the Bomberman into a Cinderman
    -- four presses frozen, with no Player object on the board at all -- and
    fire kills the Cinderman outright, which is another silent softlock (this
    game has no death object, so the engine never reports GAME_OVER). The plans
    here never take the hit; the model tracks it because ``--fuzz`` does.

The expert
----------
A NATIVE model of the ruleset (`_Board.step`) and a MACRO search over it. The
model exists because the interpreter runs this game at ~1.3k steps/s where the
model runs at ~12k, and because the search wants to simulate a fuse dozens of
times per node.

The macro states are the QUIESCENT boards -- no bomb, no fire, nothing in
flight -- because that is the only place a plan has a free choice. Between two
of them the search offers:

  * ``push``   -- walk to the cell beside a crate or a stick of TNT, shove it;
  * ``reload`` -- walk up to an armoury you can reach and bump it, +1 round;
  * ``bomb``   -- walk to a cell, drop a bomb, and run until the board is quiet;
  * and, once the last crate is gone, the walk to the exit, which is the win.

The ``bomb`` macro comes in two flavours. The cheap one simulates the fuse and
its whole chain ONCE with the player removed -- nothing in a blast depends on
the player after the fuse is lit -- and then runs a (cell, turn) BFS through the
resulting danger map, which answers "where can I be standing when this is over"
for every destination at once. The expensive one (`_Board.salvo`) is a real
breadth-first search over full model states in which the player may drop FURTHER
bombs while the first is burning; level 2 is unwinnable without it, and it is
what the level's own name ("Chicken run") is about.

THE HEURISTIC. Fireballs still owed, plus the pushes needed to make them enough:
greedily cover the crates with 3x3 footprints (skipping any that would take the
exit with them), charge `BOMB_COST` per footprint used, and charge `PUSH_COST`
per SOKOBAN PUSH-DISTANCE step for every crate the budget could not cover. The
push-distance term is what makes the search shove crates together at all -- the
cover count alone is flat over the whole approach, since it only drops on the
last push of a sequence. A state whose leftover crate can never be pushed into
range, or that has burnt its exit or its last armoury, scores DEAD and is
dropped un-expanded.

The levels
----------
All five are MarcinK's own boards in shipped order (the five ``message`` screens
are not levels and the adapter does not count them).

    level  size    crates  tnt  ammo    plan  found by
    0      9x9       5      0   1         19  cheap bomb macro
    1      9x8       7      0   2         34  cheap bomb macro
    2      9x9      16      6   infinite  25  salvo (multi-bomb) macro
    3      9x9       8      2   reload    46  cheap bomb macro
    4      11x11     8     11   reload    49  cheap bomb macro

173 presses over the five, every one of them replayed through the real
interpreter to a WIN by ``--verify``, and all inside the adapter's 200-step
per-level budget with room for the exploration prefix.

Level 2 is the one that needs the salvo, and it is worth knowing why. The player
starts boxed into a fifteen-cell strip down the left; the armoury is walled in
by the crate field, so the chain that clears the field destroys it too; and the
crate at (4, 4) sits in a pocket whose only fireable cells -- (3, 5), (4, 5),
(5, 5) -- are crates until that chain has been through them. The bomb that takes
that last crate therefore has to be dropped WHILE the chain is still running:
the shipped plan drops it on (3, 5) sixteen presses after the first one, into a
gap the fire opened, with the armoury still standing. A search that insists on a
quiet board between bombs exhausts its whole reachable space (492 states) and
finds nothing.

Where the model is DELIBERATELY blind
-------------------------------------
Two fireballs born orthogonally ADJACENT on the same press fight over which of
them stays a centre, and the interpreter settles that by the iteration order of
a Python set (`_object_positions`), which no model can mirror. `_Board.step`
returns an ``ambiguous`` flag for exactly that case and every search drops such
a state un-expanded, so no plan is ever built on a coin flip. It costs nothing
on the shipped boards: no two sticks of TNT on them are orthogonally adjacent,
so a chain never sets two off side by side.

The searches are seed-independent, so they are written to
``data/fireproof_bomber_plans.json`` (start plan + optimal sets per level) and
no shard of `parallelize_generator` re-derives them. A cold run is about a
minute -- **run ``--plans`` once before a parallel run**; delete that file to
re-derive.

Optimal-action sets
-------------------
A plan step is labelled with the press taken, plus -- for the WALK stretches
inside a macro -- every other direction that keeps the player on a shortest
route to the cell that walk ends on. That is sound because a walk happens on a
QUIESCENT board: no rule fires on a bare move onto an empty cell, no fuse is
burning, so every shortest re-route reaches the same cell after the same number
of presses with an identical board behind it.

Every other step -- a push, an X, and every press of an escape -- is labelled
with itself alone. Which crate to shove where IS the puzzle, and a press one
turn earlier or later during a fuse is a different plan, not a reordering of
this one.

Augmentation
------------
Engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: frame rotation (k in {0,1,2,3}) plus an independent horizontal
and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to). The flips are
exact symmetries: there is no gravity, input is screen-relative, and every rule
is stated over all four directions at once, so a mirrored board is a legal board
of the same game.

Usage (run from the repo root):
    python solvers/generate_fireproof_bomber_training.py --episodes 200 \
        --out data/training_multi_level/fireproof_bomber

    python solvers/generate_fireproof_bomber_training.py --plans   # level report
    python solvers/generate_fireproof_bomber_training.py --verify  # replay
    python solvers/generate_fireproof_bomber_training.py --audit   # rendering
    python solvers/generate_fireproof_bomber_training.py --fuzz    # model vs engine
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame               # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan     # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Fireproof_bomber"

#: Disk cache of the per-level start plan AND its optimal-action sets.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "fireproof_bomber_plans.json"

PRESSES = ["up", "down", "left", "right", "action"]
DIRS = PRESSES[:4]
_D = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_DD = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Fireball parts, in the order the .txt declares them. Which part sits in a
#: cell decides what grows out of it next press, so the model tracks the kind
#: and not just "there is fire here".
CE, TO, BO, LE, RI, TL, TR, BL, BR = range(9)
_F1NAME = {CE: "fire1center", TO: "fire1top", BO: "fire1bottom",
           LE: "fire1left", RI: "fire1right"}
_F2NAME = {CE: "fire2center", TO: "fire2top", BO: "fire2bottom",
           LE: "fire2left", RI: "fire2right", TL: "fire2topleft",
           TR: "fire2topright", BL: "fire2bottomleft", BR: "fire2bottomright"}
_F3NAME = {CE: "fire3center", TO: "fire3top", BO: "fire3bottom",
           LE: "fire3left", RI: "fire3right", TL: "fire3topleft",
           TR: "fire3topright", BL: "fire3bottomleft", BR: "fire3bottomright"}
_N2F1 = {v: k for k, v in _F1NAME.items()}
_N2F2 = {v: k for k, v in _F2NAME.items()}
_N2F3 = {v: k for k, v in _F3NAME.items()}

#: Which corner a stage-II arm grows, per (arm part, direction it grows in).
_CORNER = {(TO, "left"): TL, (TO, "right"): TR, (BO, "left"): BL,
           (BO, "right"): BR, (LE, "up"): TL, (LE, "down"): BL,
           (RI, "up"): TR, (RI, "down"): BR}

#: Player states. C1..C4 are the four presses spent as a charred Cinder, during
#: which there is no Player object on the board at all.
BOMBER, CINDER, C1, C2, C3, C4 = range(6)
PLAYER_NAME = {BOMBER: "bomberman", CINDER: "cinderman", C1: "playercinder1",
               C2: "playercinder2", C3: "playercinder3", C4: "playercinder4"}
NAME_PLAYER = {v: k for k, v in PLAYER_NAME.items()}

#: An ArmoryI never runs out.
INF = -1

#: Presses a bomb macro cannot cost less than: the X, then the seven turns a
#: fuse and its fireball take to come and go.
BOMB_COST = 8
#: Presses a push cannot cost less than: the shove, plus a step to reach it.
PUSH_COST = 2
#: Charged instead of a distance when the board can no longer be won.
DEAD = 10 ** 6

#: Indices into a model state, for the readers below.
_TRANSIENT = slice(7, 13)          # placed, bombs, d3, fire I, fire II, fire III


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """The static half of a level (size, walls, where the exit stands) plus the
    ruleset, the macro search over it, and their caches.

    A model state is a 13-tuple::

        (player, pkind, crates, tnt, armoury, exit_alive, exit_open,
         placed, bombs, d3, fire1, fire2, fire3)

    ``placed`` is the bombs that have been dropped but not yet lit (the
    BombPlaced object, which is invisible and does not tick), ``bombs`` maps a
    cell to a fuse of 1..3, and ``d3`` is the Bomb3Dummy a burning stick of TNT
    (or a burning armoury) leaves behind for one press before it becomes a
    fireball. The three fire maps are the three collision layers the .txt splits
    the fireball across, so one cell can be carrying three different parts of
    three different blasts at once -- which is exactly what makes overlapping
    explosions render and spread the way they do.

    A QUIESCENT state -- the macro search's currency -- is one where all six of
    those are empty. ``exit_open`` is not free either: every turn re-opens the
    exit and then closes it again if any crate is left, so in a quiescent state
    it is always ``exit_alive and not crates``, which is why `quiesce` can drop
    it and `expand` can put it back.
    """

    def __init__(self, h, w, walls, exit_cell):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.exit = exit_cell
        self._blast: dict = {}
        self._sim: dict = {}
        self._walk: dict = {}
        self._h: dict = {}
        self._exitd: dict = {}
        self._salvo: dict = {}

    def inb(self, p):
        return 0 <= p[0] < self.h and 0 <= p[1] < self.w

    def signature(self):
        return (self.h, self.w, self.walls, self.exit)

    # -- dynamics ------------------------------------------------------------
    def step(self, st, press):
        """One keypress: ``(state, won, ambiguous)``.

        The rules are applied in the .txt's own order -- the main list, then
        force resolution, then the late list -- because that order is load
        bearing several times over: the exit re-opens before the crate check
        closes it again, a fuse ticks before the fireball it becomes is drawn,
        and the Bomberman is charred before the crate beside him burns.

        ``ambiguous`` marks the one case the interpreter decides by set order;
        see the module docstring.
        """
        (player, pkind, crates, tnt, armory, exit_alive, exit_open,
         placed, bombs, d3, f1, f2, f3) = st
        crates = set(crates); tnt = set(tnt); armory = dict(armory)
        placed = set(placed); bombs = dict(bombs); d3 = set(d3)
        f1 = dict(f1); f2 = dict(f2); f3 = dict(f3)
        walls = self.walls
        amb = False

        alive = player is not None and pkind in (BOMBER, CINDER)
        force = press if (press in _D and alive) else None

        # -- the main rule list ---------------------------------------------
        push = None
        if force:
            dr, dc = _D[force]
            tgt = (player[0] + dr, player[1] + dc)
            if tgt in crates:
                push, force = ("c", tgt), None      # the shove takes the force
            elif tgt in tnt:
                push, force = ("t", tgt), None
        if exit_alive and not exit_open:
            exit_open = True
        if press == "action" and alive and player not in placed:
            src, tie = self._pick_armory(armory)
            amb = amb or tie
            if src is not None:
                placed.add(player)
                if armory[src] > 0:
                    armory[src] -= 1
                # BombPlaced shares a collision layer with the exit, so dropping
                # one ON the exit deletes the exit, and with it the level.
                if exit_alive and player == self.exit:
                    exit_alive = exit_open = False
        if force:
            dr, dc = _D[force]
            tgt = (player[0] + dr, player[1] + dc)
            if tgt in armory and 0 <= armory[tgt] <= 8:
                armory[tgt] += 1                    # a bump is a reload
                force = None
        bd1 = {c for c, f in bombs.items() if f == 1}
        bd2 = {c for c, f in bombs.items() if f == 2}
        d3 |= {c for c, f in bombs.items() if f == 3}
        bombs = {}
        fd1, fd2 = f1, f2                           # ...Dummy, promoted below
        f1, f2, f3 = {}, {}, {}                     # fire III dissipates here
        cdummy = 0
        if pkind == C4:
            pkind = CINDER
        elif pkind in (C1, C2, C3):
            cdummy = pkind - C1 + 1
            pkind = None

        # -- force resolution ------------------------------------------------
        if push is not None:
            kind, cell = push
            dr, dc = _D[press]
            tgt = (cell[0] + dr, cell[1] + dc)
            if self._open(tgt, crates, tnt, armory, player):
                if kind == "c":
                    crates.discard(cell); crates.add(tgt)
                else:
                    tnt.discard(cell); tnt.add(tgt)
        elif force:
            dr, dc = _D[force]
            tgt = (player[0] + dr, player[1] + dc)
            if self._open(tgt, crates, tnt, armory, None):
                player = tgt

        # -- the late rule list ----------------------------------------------
        def put5(cell, what):
            """Write into the layer the bombs share with the charred player --
            so a fireball born under a Cinder deletes it."""
            nonlocal player, pkind, cdummy
            bombs.pop(cell, None)
            d3.discard(cell)
            if player == cell and (pkind in (C1, C2, C3, C4) or cdummy):
                player, pkind, cdummy = None, None, 0
            if what == "d3":
                d3.add(cell)
            else:
                bombs[cell] = what

        if exit_open and crates:
            exit_open = False
        if player is not None and pkind in (BOMBER, CINDER):
            for dr, dc in _DD:                      # standing beside it lights it
                nb = (player[0] + dr, player[1] + dc)
                if nb in placed:
                    placed.discard(nb)
                    put5(nb, 1)
        for cell in sorted(bd1):
            put5(cell, 2)
        for cell in sorted(bd2):
            put5(cell, 3)
        for cell, k in fd1.items():                 # fire I -> fire II
            f2[cell] = k
            fd2.pop(cell, None)                     # same collision layer
        for src_kind, sides in ((TO, ("left", "right")), (BO, ("left", "right")),
                                (LE, ("up", "down")), (RI, ("up", "down"))):
            for side in sides:                      # stage II grows its corners
                corner = _CORNER[(src_kind, side)]
                dr, dc = _D[side]
                for cell in sorted(c for c, k in f2.items() if k == src_kind):
                    tgt = (cell[0] + dr, cell[1] + dc)
                    if not self.inb(tgt) or tgt in walls:
                        continue
                    if tgt in f1 or tgt in f2 or tgt in f3:
                        continue
                    f2[tgt] = corner
                    fd2.pop(tgt, None)
        for cell, k in fd2.items():                 # fire II -> fire III
            f3[cell] = k
        centres = sorted(d3)                        # a spent fuse is a fireball
        d3.clear()                                  # TNT below refills it
        for i, a in enumerate(centres):
            for b in centres[i + 1:]:
                if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                    amb = True                      # see the module docstring
        for side, kind in (("up", TO), ("down", BO), ("left", LE), ("right", RI)):
            dr, dc = _D[side]
            for cell in centres:
                tgt = (cell[0] + dr, cell[1] + dc)
                if self.inb(tgt) and tgt not in walls:
                    f1[tgt] = kind
        # The centres go down AFTER their arms: an arm and a centre share a
        # collision layer, so two fireballs born beside each other overwrite one
        # another and the interpreter's answer depends on which match it applied
        # last. That is the ``amb`` case above -- every search drops it -- and
        # writing the centres last is simply the tidier of the two guesses.
        for cell in centres:
            f1[cell] = CE
        dmg = self._damage(f1, f2, f3)
        fire = set(f1) | set(f2) | set(f3)
        if pkind == BOMBER and player in dmg:
            bombs.pop(player, None)                 # the Cinder evicts a bomb
            d3.discard(player)
            pkind = C1
        if cdummy == 1 and player in fire:
            cdummy, pkind = 0, C1                   # still burning: stay charred
        if pkind == CINDER and player in dmg:
            player, pkind = None, None              # the Cinderman is fragile
        if cdummy:
            pkind = {1: C2, 2: C3, 3: C4}[cdummy]
            cdummy = 0
        crates -= dmg
        for cell in sorted(tnt & fire):
            tnt.discard(cell)
            put5(cell, "d3")                        # goes off NEXT press
        if exit_alive and self.exit in dmg:
            exit_alive = exit_open = False
            # The rule's RHS keeps only Fire1Center, so the fire object the
            # FireDamage group bound to is consumed along with the exit.
            e = self.exit
            if e in f1:
                del f1[e]
            elif e in f2:
                del f2[e]
            elif f3.get(e) == CE:
                del f3[e]
            f1[e] = CE
        for cell in sorted(c for c in armory if c in dmg):
            del armory[cell]
            put5(cell, "d3")

        new = (player, pkind, frozenset(crates), frozenset(tnt),
               tuple(sorted(armory.items())), exit_alive, exit_open,
               frozenset(placed), tuple(sorted(bombs.items())),
               frozenset(d3), tuple(sorted(f1.items())),
               tuple(sorted(f2.items())), tuple(sorted(f3.items())))
        won = (exit_alive and exit_open and player is not None
               and pkind in (BOMBER, CINDER) and player == self.exit)
        return new, won, amb

    def _open(self, p, crates, tnt, armory, player):
        return (self.inb(p) and p not in self.walls and p not in crates
                and p not in tnt and p not in armory and p != player)

    @staticmethod
    def _pick_armory(armory):
        """``(cell spent, is the choice order-dependent)``.

        The placement rules are listed ArmoryI, Armory1, ... Armory9, so the
        first one with a match wins: an infinite armoury if there is one, else
        the LOWEST non-zero count. Two armouries tied on that count are
        separated only by the interpreter's set order, hence the flag. Every
        shipped level has exactly one armoury, so it never fires there."""
        inf = [c for c, n in armory.items() if n == INF]
        if inf:
            return (min(inf), len(inf) > 1)
        live = [(n, c) for c, n in armory.items() if n >= 1]
        if not live:
            return (None, False)
        low = min(n for n, _ in live)
        tied = [c for n, c in live if n == low]
        return (min(tied), len(tied) > 1)

    @staticmethod
    def _damage(f1, f2, f3):
        """Cells that burn what is standing on them. Every part of fire I and
        fire II does; of fire III only the centre, which is why a dying blast's
        edges are safe to walk through."""
        out = set(f1) | set(f2)
        out |= {c for c, k in f3.items() if k == CE}
        return out

    # -- geometry ------------------------------------------------------------
    def blast(self, x):
        """The cells a fireball centred on ``x`` damages, i.e. its stage-II
        footprint: the centre, the orthogonal neighbours that are not walls,
        and each diagonal reachable from one of those arms."""
        got = self._blast.get(x)
        if got is not None:
            return got
        out = {x}
        arms = {}
        for d, (dr, dc) in _D.items():
            p = (x[0] + dr, x[1] + dc)
            if self.inb(p) and p not in self.walls:
                arms[d] = p
                out.add(p)
        for d, p in arms.items():
            for side in (("left", "right") if d in ("up", "down")
                         else ("up", "down")):
                dr, dc = _D[side]
                cn = (p[0] + dr, p[1] + dc)
                if self.inb(cn) and cn not in self.walls:
                    out.add(cn)
        out = frozenset(out)
        self._blast[x] = out
        return out

    def blockers(self, crates, tnt, armory):
        return self.walls | crates | tnt | set(armory)

    def walk(self, q):
        """BFS parents/distances over the cells the player may walk on. A cell
        holding a crate is NOT one of them: walking into a crate is a push."""
        key = (q[0], q[2], q[3], q[4])
        got = self._walk.get(key)
        if got is not None:
            return got
        block = self.blockers(q[2], q[3], dict(q[4]))
        parent = {q[0]: None}
        dist = {q[0]: 0}
        dq = deque([q[0]])
        while dq:
            cur = dq.popleft()
            for d in DIRS:
                dr, dc = _D[d]
                nxt = (cur[0] + dr, cur[1] + dc)
                if not self.inb(nxt) or nxt in block or nxt in parent:
                    continue
                parent[nxt] = (cur, d)
                dist[nxt] = dist[cur] + 1
                dq.append(nxt)
        self._walk[key] = (parent, dist)
        return parent, dist

    @staticmethod
    def path(parent, cell):
        out = []
        while parent[cell] is not None:
            cell, d = parent[cell]
            out.append(d)
        out.reverse()
        return out

    def walk_labels(self, q, start, tgt, presses):
        """Per-step optimal SETS for a walk from ``start`` to ``tgt``.

        A walk runs on a quiescent board, where a bare move onto an empty cell
        fires no rule at all, so every shortest re-route reaches ``tgt`` after
        the same number of presses with an identical board behind it. Each step
        is therefore labelled with every direction that keeps the player on a
        shortest route -- not with the one interleaving the BFS happened to
        return."""
        block = self.blockers(q[2], q[3], dict(q[4]))
        dist = {tgt: 0}
        dq = deque([tgt])
        while dq:
            cur = dq.popleft()
            for dr, dc in _DD:
                nxt = (cur[0] + dr, cur[1] + dc)
                if (self.inb(nxt) and nxt not in block and nxt not in dist):
                    dist[nxt] = dist[cur] + 1
                    dq.append(nxt)
        out = []
        cell = start
        for press in presses:
            here = dist.get(cell)
            alts = [] if here is None else [
                d for d in DIRS
                if dist.get((cell[0] + _D[d][0], cell[1] + _D[d][1])) == here - 1]
            out.append(alts if press in alts else [press])
            cell = (cell[0] + _D[press][0], cell[1] + _D[press][1])
        return out

    # -- one bomb, simulated once --------------------------------------------
    def timeline(self, q, x):
        """``(states, T)`` for a bomb dropped at ``x`` and lit on turn 1, where
        ``states[t]`` is the board at the end of turn ``t`` and ``T`` is the
        turn it all goes quiet again.

        The player is left OUT of the simulation. Once the fuse is lit nothing
        in the blast pathway depends on it, so one timeline serves every escape
        route through it -- which is the whole reason the cheap bomb macro can
        answer for every destination at once."""
        key = (q[2], q[3], q[4], q[5], x)
        if key in self._sim:
            return self._sim[key]
        armory = dict(q[4])
        src, tie = self._pick_armory(armory)
        if src is None or tie:
            self._sim[key] = None
            return None
        if armory[src] > 0:
            armory[src] -= 1
        cur = (None, None, q[2], q[3], tuple(sorted(armory.items())),
               q[5], q[5] and not q[2], frozenset(), ((x, 1),),
               frozenset(), (), (), ())
        states = [None, cur]
        for _ in range(60):
            cur, _won, amb = self.step(cur, "action")
            if amb:
                self._sim[key] = None
                return None
            states.append(cur)
            if not any(cur[_TRANSIENT]):
                break
        else:
            self._sim[key] = None               # a chain that never settles
            return None
        got = (states, len(states) - 1)
        self._sim[key] = got
        return got

    def escape(self, q, x):
        """``{quiescent state: presses}`` -- every board the player can be
        standing on when the fuse it dropped at ``x`` has finished, walking
        only. One (cell, turn) BFS through the timeline's danger map."""
        tl = self.timeline(q, x)
        if tl is None:
            return {}
        states, T = tl
        end = states[T]
        if q[5] and not end[5]:                 # the blast took the exit
            return {}
        occ = self.blockers(q[2], q[3], dict(q[4]))
        frontier = {}
        for d in DIRS:                          # turn 1 is the step that lights it
            dr, dc = _D[d]
            p = (x[0] + dr, x[1] + dc)
            if self.inb(p) and p not in occ:
                frontier[p] = [d]
        for t in range(2, T + 1):
            prev = states[t - 1]
            occ = self.blockers(prev[2], prev[3], dict(prev[4]))
            dmg = self._damage(dict(states[t][10]), dict(states[t][11]),
                               dict(states[t][12]))
            nxt = {}
            for p, presses in frontier.items():
                if p not in dmg:
                    hold = self._wait_press(prev, p)
                    if hold is not None:
                        nxt[p] = presses + [hold]
                for d in DIRS:
                    dr, dc = _D[d]
                    r = (p[0] + dr, p[1] + dc)
                    if (self.inb(r) and r not in occ and r not in dmg
                            and r not in nxt):
                        nxt[r] = presses + [d]
            frontier = nxt
            if not frontier:
                return {}
        return {(cell, q[1], end[2], end[3], end[4], end[5]): presses
                for cell, presses in frontier.items()}

    def _wait_press(self, st, p):
        """A press that passes a turn without moving anything: a wall (or the
        board edge) to walk into, a jammed crate to shove, or -- with the
        armoury empty -- X, which then does nothing at all.

        Bumping an armoury that still has room is deliberately NOT offered: it
        would reload, and the timeline this is read against was simulated
        without it."""
        crates, tnt, armory = st[2], st[3], dict(st[4])
        block = self.blockers(crates, tnt, armory)
        for d in DIRS:
            dr, dc = _D[d]
            r = (p[0] + dr, p[1] + dc)
            if not self.inb(r) or r in self.walls:
                return d
            if r in armory and armory[r] == 9:
                return d
            if r in crates or r in tnt:
                s = (r[0] + dr, r[1] + dc)
                if not self.inb(s) or s in block:
                    return d
        if self._pick_armory(armory)[0] is None:
            return "action"
        return None

    # -- several bombs at once -----------------------------------------------
    def salvo(self, q, x, cap=400_000, horizon=45):
        """``{quiescent state: presses}`` for dropping a bomb at ``x`` and then
        running -- dropping MORE bombs on the way if there is ammunition left.

        A real breadth-first search over full model states, so it is a hundred
        times the cost of `escape` and is only reached for the boards `escape`
        cannot win. Level 2 is one: its armoury is buried inside the crate field
        and dies partway through the chain, so the bomb that clears the LAST
        crate has to be dropped while the chain is still running -- into a cell
        the chain has only just cleared.

        X is offered only where it would actually burn something and would not
        take the exit with it. Without that the branching is five everywhere and
        the search drowns in bombs dropped into empty corridors."""
        # The player starts the salvo standing ON ``x``, so where it walked in
        # from is not part of the answer and does not belong in the key.
        key = (q[2], q[3], q[4], q[5], q[1], x)
        got = self._salvo.get(key)
        if got is not None:
            return got
        st = (x,) + expand(self, q)[1:]
        st, won, amb = self.step(st, "action")
        if amb or won:
            self._salvo[key] = {}
            return {}
        out: dict = {}
        seen = {st}
        frontier = [(st, [])]
        for _t in range(horizon):
            nxt = []
            for cur, presses in frontier:
                for p in PRESSES:
                    if p == "action":
                        if cur[0] in cur[7]:
                            continue
                        fp = self.blast(cur[0])
                        if not (fp & cur[2]) and not (fp & cur[3]):
                            continue
                        if cur[5] and self.exit in fp:
                            continue
                    ns, _won, amb = self.step(cur, p)
                    if amb or ns[0] is None or ns[1] not in (BOMBER, CINDER):
                        continue                # died, was charred, or a coin flip
                    if q[5] and not ns[5]:
                        continue                # the blast took the exit
                    if ns in seen:
                        continue
                    seen.add(ns)
                    route = presses + [p]
                    if not any(ns[_TRANSIENT]):
                        out.setdefault(quiesce(ns), route)
                        continue                # quiet again: a macro endpoint
                    nxt.append((ns, route))
                    if len(seen) > cap:
                        self._salvo[key] = out
                        return out
            frontier = nxt
            if not frontier:
                break
        self._salvo[key] = out
        return out

    # -- heuristic -----------------------------------------------------------
    def heuristic(self, q):
        """Fireballs still owed plus the pushes needed to make them enough.

        Everything but the walk to the exit depends on the BOARD alone, so it is
        memoized on the board: a bomb macro hands the search one successor per
        cell the player can end on, and all of them share this."""
        if not q[2]:
            if not q[5]:
                return DEAD                     # no crates left, and no exit
            d = self.exit_dist(q).get(q[0])
            return DEAD if d is None else d
        key = (q[2], q[3], q[4], q[5])
        got = self._h.get(key)
        if got is None:
            got = self._board_h(q)
            self._h[key] = got
        return got

    def exit_dist(self, q):
        """BFS distance to the exit over this board's walkable cells."""
        key = (q[2], q[3], q[4])
        got = self._exitd.get(key)
        if got is not None:
            return got
        block = self.blockers(q[2], q[3], dict(q[4]))
        dist = {} if self.exit in block else {self.exit: 0}
        dq = deque(dist)
        while dq:
            cur = dq.popleft()
            for dr, dc in _DD:
                nxt = (cur[0] + dr, cur[1] + dc)
                if self.inb(nxt) and nxt not in block and nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    dq.append(nxt)
        self._exitd[key] = dist
        return dist

    def budget(self, q):
        """How many more fireballs this board can still produce. An armoury the
        player can walk up to reloads a press at a time, so it is not ammunition
        that limits those levels -- only the walking is."""
        armory = dict(q[4])
        if any(n == INF for n in armory.values()):
            return 9
        block = self.blockers(q[2], q[3], armory)
        for cell, cnt in armory.items():
            if cnt < 9 and any(self.inb((cell[0] + dr, cell[1] + dc))
                               and (cell[0] + dr, cell[1] + dc) not in block
                               for dr, dc in _DD):
                return 9
        return sum(n for n in armory.values() if n > 0) + len(q[3])

    def centres(self, q):
        """Cells a fireball could be centred on without taking the exit with
        it: every free cell (you stand there and press X) and every stick of
        TNT (something else sets it off for you)."""
        block = self.blockers(q[2], q[3], dict(q[4]))
        out = [(r, c) for r in range(self.h) for c in range(self.w)
               if (r, c) not in block]
        out += list(q[3])
        if q[5]:
            out = [p for p in out if self.exit not in self.blast(p)]
        return out

    def _board_h(self, q):
        crates = q[2]
        cand = [(self.blast(p) & crates, p) for p in self.centres(q)]
        cand = [(cov, p) for cov, p in cand if cov]
        left = set(crates)
        covered: set = set()
        used = 0
        cap = self.budget(q)
        while left and used < cap and cand:
            cov, p = max(cand, key=lambda t: len(t[0] & left))
            if not (cov & left):
                break
            left -= cov
            covered |= self.blast(p)
            used += 1
        total = used * BOMB_COST
        if not left:
            return total
        if used == 0:
            return DEAD                         # nothing left to burn them with
        far = self._push_dist(q, covered)
        for c in left:
            d = far.get(c)
            if d is None:
                return DEAD                     # a crate that can never be moved
            total += PUSH_COST * d
        return total

    def _push_dist(self, q, targets):
        """Pushes needed to shove a crate from each cell into ``targets``,
        ignoring the other crates (the usual sokoban relaxation). This is the
        term that gives the search a gradient: the cover count alone only drops
        on the LAST push of a sequence, so on its own it cannot tell a search
        that it is halfway through lining a crate up."""
        hard = self.walls | set(dict(q[4]))
        dist = {c: 0 for c in targets if c not in hard}
        dq = deque(dist)
        while dq:
            cur = dq.popleft()
            for dr, dc in _DD:
                src = (cur[0] - dr, cur[1] - dc)      # where the crate was
                stand = (src[0] - dr, src[1] - dc)    # where the player stood
                if not self.inb(src) or src in hard or src in dist:
                    continue
                if not self.inb(stand) or stand in hard:
                    continue
                dist[src] = dist[cur] + 1
                dq.append(src)
        return dist

    # -- successors ----------------------------------------------------------
    def successors(self, q, salvo=False):
        """``[(quiescent state, presses, labels)]`` -- one entry per macro."""
        out = []
        parent, _dist = self.walk(q)
        armory = dict(q[4])
        block = self.blockers(q[2], q[3], armory)
        for pieces, which in ((q[2], "c"), (q[3], "t")):
            for piece in pieces:
                for d in DIRS:
                    dr, dc = _D[d]
                    stand = (piece[0] - dr, piece[1] - dc)
                    tgt = (piece[0] + dr, piece[1] + dc)
                    if stand not in parent:
                        continue
                    if not self.inb(tgt) or tgt in block:
                        continue
                    crates, tnt = set(q[2]), set(q[3])
                    if which == "c":
                        crates.discard(piece); crates.add(tgt)
                    else:
                        tnt.discard(piece); tnt.add(tgt)
                    ns = (stand, q[1], frozenset(crates), frozenset(tnt),
                          q[4], q[5])
                    walk = self.path(parent, stand)
                    out.append((ns, walk + [d],
                                self.walk_labels(q, q[0], stand, walk) + [[d]]))
        for cell, cnt in armory.items():
            if cnt == INF or cnt >= 9:
                continue
            for d in DIRS:
                dr, dc = _D[d]
                stand = (cell[0] - dr, cell[1] - dc)
                if stand not in parent:
                    continue
                na = dict(armory)
                na[cell] = cnt + 1
                ns = (stand, q[1], q[2], q[3], tuple(sorted(na.items())), q[5])
                walk = self.path(parent, stand)
                out.append((ns, walk + [d],
                            self.walk_labels(q, q[0], stand, walk) + [[d]]))
        if self._pick_armory(armory)[0] is not None:
            for x in parent:
                fp = self.blast(x)
                if not (fp & q[2]) and not (fp & q[3]):
                    continue                    # a bomb that burns nothing
                if q[5] and self.exit in fp:
                    continue
                ends = self.salvo(q, x) if salvo else self.escape(q, x)
                walk = self.path(parent, x)
                labels = self.walk_labels(q, q[0], x, walk) + [["action"]]
                for ns, presses in ends.items():
                    out.append((ns, walk + ["action"] + presses,
                                labels + [[p] for p in presses]))
        return out

    # -- the searches --------------------------------------------------------
    def astar(self, q0, cap=20_000, weight=1, salvo=False):
        """Weighted A* over the macros. ``cap`` counts EXPANSIONS.

        Returns ``(presses, labels)`` or None. A returned plan is a genuine WIN
        path; at ``weight`` 1 it is the shortest this macro set can express,
        which is not the same as the shortest the game admits -- the macros
        cannot start a second fuse while the first burns unless ``salvo`` is on,
        and the escape always runs to full quiet."""
        if self.heuristic(q0) >= DEAD:
            return None
        nodes = [(None, (), ())]                # (parent, presses, labels)
        pq = [(weight * self.heuristic(q0), 0, 0, q0)]
        best = {q0: 0}
        popped = 0
        while pq:
            _f, g, idx, q = heapq.heappop(pq)
            if best.get(q, -1) != g:
                continue
            popped += 1
            if popped > cap:
                return None
            done = self._finish(q, nodes, idx)
            if done is not None:
                return done
            for ns, presses, labels in self.successors(q, salvo):
                ng = g + len(presses)
                if best.get(ns, 1 << 30) <= ng:
                    continue
                h = self.heuristic(ns)
                if h >= DEAD:
                    continue
                best[ns] = ng
                nodes.append((idx, tuple(presses), tuple(map(tuple, labels))))
                heapq.heappush(pq, (ng + weight * h, ng, len(nodes) - 1, ns))
        return None

    def beam(self, q0, width=300, depth=14, salvo=False):
        """Breadth-first beam over the macros, ranked by ``g + heuristic``, for
        the boards A* cannot reach. Dedup is kept across the whole search, so a
        state re-reached at a later depth never re-expands."""
        if self.heuristic(q0) >= DEAD:
            return None
        nodes = [(None, (), ())]
        frontier = [(0, 0, q0)]
        seen = {q0}
        for _ in range(depth):
            kids = []
            for g, idx, q in frontier:
                done = self._finish(q, nodes, idx)
                if done is not None:
                    return done
                for ns, presses, labels in self.successors(q, salvo):
                    if ns in seen:
                        continue
                    h = self.heuristic(ns)
                    if h >= DEAD:
                        continue
                    seen.add(ns)
                    nodes.append((idx, tuple(presses),
                                  tuple(map(tuple, labels))))
                    ng = g + len(presses)
                    kids.append((ng + h, ng, len(nodes) - 1, ns))
            if not kids:
                return None
            kids.sort(key=lambda k: k[0])
            frontier = [(g, i, ns) for _s, g, i, ns in kids[:width]]
        return None

    def _finish(self, q, nodes, idx):
        """The walk onto the open exit, if this state is one walk from a win."""
        if q[2] or not q[5]:
            return None
        parent, _dist = self.walk(q)
        if self.exit not in parent:
            return None
        tail = self.path(parent, self.exit)
        labels = self.walk_labels(q, q[0], self.exit, tail)
        presses, sets = [], []
        while idx:
            up, p, s = nodes[idx]
            presses = list(p) + presses
            sets = [list(x) for x in s] + sets
            idx = up
        return presses + tail, sets + labels


def quiesce(st):
    """The macro-level view of a model state: everything transient dropped."""
    return (st[0], st[1], st[2], st[3], st[4], st[5])


def expand(board, q):
    """The full model state a quiescent macro state stands for."""
    return (q[0], q[1], q[2], q[3], q[4], q[5], bool(q[5] and not q[2]),
            frozenset(), (), frozenset(), (), (), ())


# ---------------------------------------------------------------------------
# Reading a board out of the interpreter
# ---------------------------------------------------------------------------

def read_board(eng, g):
    """``(_Board, state)`` for the interpreter's current grid."""
    inv = {v: k for k, v in g.obj_name_to_idx.items()}
    wall_names = set(g.or_groups.get("wall", []))
    walls, crates, tnt, placed, d3 = set(), set(), set(), set(), set()
    armory, bombs, f1, f2, f3 = {}, {}, {}, {}, {}
    player = pkind = exit_cell = None
    exit_alive = exit_open = False
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            p = (r, c)
            for o in cell:
                n = inv[o]
                if n in wall_names:
                    walls.add(p)
                elif n == "crate":
                    crates.add(p)
                elif n == "tnt":
                    tnt.add(p)
                elif n == "bombplaced":
                    placed.add(p)
                elif n == "bomb3dummy":
                    d3.add(p)
                elif n in ("bomb1", "bomb2", "bomb3"):
                    bombs[p] = int(n[-1])
                elif n == "armoryi":
                    armory[p] = INF
                elif n.startswith("armory"):
                    armory[p] = int(n[6:])
                elif n == "exitclosed":
                    exit_cell, exit_alive = p, True
                elif n == "exitopen":
                    exit_cell, exit_alive, exit_open = p, True, True
                elif n in NAME_PLAYER:
                    player, pkind = p, NAME_PLAYER[n]
                elif n in _N2F1:
                    f1[p] = _N2F1[n]
                elif n in _N2F2:
                    f2[p] = _N2F2[n]
                elif n in _N2F3:
                    f3[p] = _N2F3[n]
    board = _Board(eng.height, eng.width, walls, exit_cell)
    state = (player, pkind, frozenset(crates), frozenset(tnt),
             tuple(sorted(armory.items())), exit_alive, exit_open,
             frozenset(placed), tuple(sorted(bombs.items())), frozenset(d3),
             tuple(sorted(f1.items())), tuple(sorted(f2.items())),
             tuple(sorted(f3.items())))
    return board, state


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class FireproofExpert(PSExpert):
    """Plans read off the native model; `PSExpert` supplies the in-memory memo,
    the on-disk start-plan cache and the snapshot discipline, so only `_search`
    is overridden."""

    directions = PRESSES
    plan_cache_path = PLAN_CACHE

    #: Weights tried in order for the CHEAP (one fuse at a time) macro set; the
    #: first that returns wins, so a level solvable at 1 keeps the shortest plan
    #: that macro set can express.
    weights = (1, 2, 3, 5)
    #: Expansions per A* rung. Four of the five levels close well inside this;
    #: level 2's cheap macro space is only ~500 states and exhausts long before.
    astar_cap = 20_000
    #: The multi-bomb fallback, an order of magnitude dearer per node.
    salvo_cap = 1_500
    salvo_weights = (1, 2)
    beam_width = 120
    beam_depth = 10

    def setup(self) -> None:
        self._board = None
        self._board_sig = None

    def heuristic(self, eng) -> int:                 # pragma: no cover
        raise NotImplementedError(
            "the model owns the heuristic; see _Board.heuristic")

    def _board_for(self, board):
        """Reuse one `_Board` per level so its blast/timeline/heuristic caches
        survive across every search on that board."""
        sig = board.signature()
        if sig != self._board_sig:
            self._board, self._board_sig = board, sig
        return self._board

    def _search(self, eng) -> list | None:
        board, state = read_board(eng, self.g)
        if state[0] is None or state[1] not in (BOMBER, CINDER):
            return None                              # dead or mid-char
        board = self._board_for(board)
        q = quiesce(state)
        if any(state[_TRANSIENT]):
            return None                              # only plans from a quiet board
        found = None
        for weight in self.weights:
            found = board.astar(q, self.astar_cap, weight)
            if found is not None:
                break
        if found is None:
            for weight in self.salvo_weights:
                found = board.astar(q, self.salvo_cap, weight, salvo=True)
                if found is not None:
                    break
        if found is None:
            found = board.beam(q, self.beam_width, self.beam_depth, salvo=True)
        if found is None:
            return None
        presses, labels = found
        # The macros aim at a quiet board and only then walk onto the exit, but
        # the win can land EARLIER -- an escape route that happens to end on the
        # exit cell wins the moment the last crate burns and the exit re-opens.
        # Truncate at the first winning press so the plan (and the recording)
        # stops there instead of stepping off the exit and back on.
        cur = state
        for i, press in enumerate(presses):
            cur, won, _amb = board.step(cur, press)
            if won:
                return Plan(presses[:i + 1], labels[:i + 1])
        return Plan(presses, labels)


class FireproofSolver(PSAStarSolver):
    game_id = "puzzlescript_fireproof_bomber"
    game_name = GAME_NAME
    expert_cls = FireproofExpert

    #: Record against the game folder's adapter -- the one `game_envs` hands a
    #: live agent -- so a later step limit or sprite fix there cannot silently
    #: diverge from what is taped here.
    game_module_id = "ps:fireproof_bomber"

    #: The longest plan is level 4's 50 presses; the rest is room for the
    #: exploration prefix and its RESET, inside the adapter's 200-step budget.
    max_steps = 150


# ---------------------------------------------------------------------------
# Reports and checks
# ---------------------------------------------------------------------------

def _levels(solver=None):
    solver = solver or FireproofSolver()
    game = solver.make_game(0)
    return game, FireproofExpert(game)


def _report() -> int:
    """Per-level board size, crate/ammo count, plan length and tie coverage."""
    game, expert = _levels()
    eng = game._engine
    total = solved = 0
    for level in range(game.n_levels):
        game.set_level(level)
        _board, state = read_board(eng, game._game)
        ammo = ("inf" if any(n == INF for _c, n in state[4])
                else str(sum(n for _c, n in state[4] if n > 0)))
        found = expert.plan(eng, level)
        if found is None:
            print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
                  f"{len(state[2]):2d} crates  NO PLAN")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        solved += 1
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
              f"{len(state[2]):2d} crates {len(state[3]):2d} tnt "
              f"ammo {ammo:>3s}  {len(found):3d} presses ({room}), "
              f"{ties:3d} steps with a tie set "
              f"({ties / max(1, len(found)):.0%})")
    print(f"total {total} presses over {solved} solved levels "
          f"({game.n_levels} shipped)")
    return 0


def _verify() -> int:
    """Replay every level's plan on the REAL interpreter and require a WIN.

    The plans come off a model; this is the line that says the model and the
    interpreter agree about the boards that actually ship."""
    game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level:2d}: NO PLAN")
            bad += 1
            continue
        game.set_level(level)
        for d in plan:
            eng.step(d)
        ok = eng.check_win()
        bad += not ok
        print(f"level {level:2d}: {len(plan):3d} presses -> "
              f"{'WIN' if ok else 'NOT WON'}")
    print("verify clean" if not bad else f"VERIFY FAILED on {bad} levels")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Composition, not object: the player ON the open exit is the winning frame
    and has to differ from the player standing on the floor, each armoury digit
    has to differ from the other ten (it is the ammo counter, and the only
    readout of the one resource this game meters), and each of the three fuse
    states has to differ from the other two.

    Whole frames are compared rather than one cropped cell: `_render_frame`
    upscales any sub-64 render to fill the frame, so cropping by arithmetic
    lands in the wrong cell on every board whose pixel size does not divide 64.
    Rendering is per-cell independent, so filling the board with one composition
    tests the same thing where the geometry cannot drift.

    Bomb3Dummy is deliberately NOT in the list: it now wears Bomb3's art on
    purpose, because a burning stick of TNT leaves one and it means exactly what
    a Bomb3 means -- fire in this cell on the next press."""
    game = FireproofSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {
        "floor": (), "wall": ("wallsolid",), "wall_left": ("wallleft",),
        "crate": ("crate",), "tnt": ("tnt",),
        "exit_closed": ("exitclosed",), "exit_open": ("exitopen",),
        "bomberman": ("bomberman",), "cinderman": ("cinderman",),
        "bomberman_on_open_exit": ("exitopen", "bomberman"),
        "bomberman_on_closed_exit": ("exitclosed", "bomberman"),
        "cinderman_on_open_exit": ("exitopen", "cinderman"),
        "crate_on_closed_exit": ("exitclosed", "crate"),
    }
    for n in ("bomb1", "bomb2", "bomb3"):
        comps[n] = (n,)
        comps[n + "_under_player"] = ("bomberman", n)
    comps["bombplaced_under_player"] = ("bomberman", "bombplaced")
    for n in list(PLAYER_NAME.values())[2:]:
        comps[n] = (n,)
    for n in [f"armory{i}" for i in range(10)] + ["armoryi"]:
        comps[n] = (n,)
    for n in list(_F1NAME.values()) + list(_F2NAME.values()) + list(_F3NAME.values()):
        comps[n] = (n,)

    sizes = {}
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
            shots[name] = np.asarray(_render_frame(eng, g))
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Differential fuzz
# ---------------------------------------------------------------------------

def _write_state(eng, g, board, st):
    """Paint a model state onto the interpreter's grid."""
    idx = g.obj_name_to_idx
    grid = [[{idx["background"]} for _ in range(board.w)]
            for _ in range(board.h)]

    def put(cell, name):
        grid[cell[0]][cell[1]].add(idx[name])

    (player, pkind, crates, tnt, armory, exit_alive, _open_,
     placed, bombs, d3, f1, f2, f3) = st
    for c in board.walls:
        put(c, "wallsolid")
    for c in crates:
        put(c, "crate")
    for c in tnt:
        put(c, "tnt")
    for c, n in armory:
        put(c, "armoryi" if n == INF else f"armory{n}")
    if exit_alive:
        put(board.exit, "exitopen" if st[6] else "exitclosed")
    for c in placed:
        put(c, "bombplaced")
    for c, f in bombs:
        put(c, f"bomb{f}")
    for c in d3:
        put(c, "bomb3dummy")
    for c, k in f1:
        put(c, _F1NAME[k])
    for c, k in f2:
        put(c, _F2NAME[k])
    for c, k in f3:
        put(c, _F3NAME[k])
    if player is not None:
        put(player, PLAYER_NAME[pkind])
    eng.grid, eng.height, eng.width = grid, board.h, board.w
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()
    eng._rule_win = False


def _random_state(rng, board, cells):
    free = [c for c in cells if c not in board.walls]
    rng.shuffle(free)
    crates = set(free[:rng.randint(0, 6)]);          rest = free[len(crates):]
    tnt = set(rest[:rng.randint(0, 4)]);             rest = rest[len(tnt):]
    armory = {c: rng.choice([INF] + list(range(10)))
              for c in rest[:rng.randint(0, 2)]};    rest = rest[len(armory):]
    player = rest[0] if rest else None;              rest = rest[1:]
    pkind = rng.choice([BOMBER, BOMBER, BOMBER, CINDER, C1, C2, C3, C4])
    # BombPlaced shares a layer with the exit, so the two never legally coexist.
    placed = {c for c in rest[:rng.randint(0, 2)] if c != board.exit}
    rest = rest[2:]
    bombs = {c: rng.randint(1, 3) for c in rest[:rng.randint(0, 3)]}
    rest = rest[len(bombs):]
    d3 = set(rest[:rng.randint(0, 2)]);              rest = rest[len(d3):]
    f1 = {c: rng.randint(0, 4) for c in rest[:rng.randint(0, 3)]}
    rest = rest[len(f1):]
    f2 = {c: rng.randint(0, 8) for c in rest[:rng.randint(0, 3)]}
    rest = rest[len(f2):]
    f3 = {c: rng.randint(0, 8) for c in rest[:rng.randint(0, 3)]}
    exit_alive = rng.random() < 0.9
    return (player, pkind, frozenset(crates), frozenset(tnt),
            tuple(sorted(armory.items())), exit_alive,
            exit_alive and rng.random() < 0.5, frozenset(placed),
            tuple(sorted(bombs.items())), frozenset(d3),
            tuple(sorted(f1.items())), tuple(sorted(f2.items())),
            tuple(sorted(f3.items())))


def _fuzz(n_boards: int = 700, n_steps: int = 25, seed: int = 0,
          n_runs: int = 120, run_steps: int = 60) -> int:
    """Differential fuzz: boards played move-for-move against the real
    interpreter, comparing every crate, fuse, fire part, armoury count, the
    player's state and the win flag after every press.

    Two populations, because neither covers the other. RANDOM boards reach the
    states the shipped levels never do -- a charred player, a bomb under a
    Cinder, three blasts overlapping, an armoury on fire -- while RANDOM PLAY ON
    THE SHIPPED LEVELS is the only thing that exercises their real wall shapes,
    their reload alcoves and their TNT chains.

    Transitions the model flags AMBIGUOUS are counted and resynchronised rather
    than compared: they are the ones the interpreter decides by set order, and
    every search drops them, so holding the model to a coin flip would be
    testing the wrong thing."""
    game = FireproofSolver().make_game(0)
    eng, g = game._engine, game._game
    rng = random.Random(seed)
    bad = steps = ambs = 0

    def compare(board, st, press):
        nonlocal bad, steps, ambs
        eng.step(press)
        truth_won = eng.check_win()
        _b, truth = read_board(eng, g)
        new, won, amb = board.step(st, press)
        steps += 1
        if amb:
            ambs += 1
            return truth
        if new != truth or won != truth_won:
            bad += 1
            names = ["player", "pkind", "crates", "tnt", "armory", "exit_alive",
                     "exit_open", "placed", "bombs", "d3", "f1", "f2", "f3"]
            print(f"MISMATCH press={press} board {board.h}x{board.w} "
                  f"exit={board.exit}\n  walls {sorted(board.walls)}")
            for i, nm in enumerate(names):
                if new[i] != truth[i]:
                    print(f"   {nm}: model={new[i]} engine={truth[i]}")
            if won != truth_won:
                print(f"   won: model={won} engine={truth_won}")
            return truth
        return new

    for _ in range(n_boards):
        h, w = rng.randint(5, 8), rng.randint(5, 8)
        cells = [(r, c) for r in range(h) for c in range(w)]
        walls = {c for c in cells if rng.random() < 0.15}
        board = _Board(h, w, walls,
                       rng.choice([c for c in cells if c not in walls]))
        st = _random_state(rng, board, cells)
        _write_state(eng, g, board, st)
        _b, st = read_board(eng, g)
        for _s in range(n_steps):
            st = compare(board, st, rng.choice(PRESSES))
        if bad > 5:
            return 1

    for run in range(n_runs):
        game.set_level(run % game.n_levels)
        board, st = read_board(eng, g)
        for _s in range(run_steps):
            st = compare(board, st,
                         rng.choices(PRESSES, weights=[3, 3, 3, 3, 2])[0])
            if eng.check_win():
                break
        if bad > 5:
            return 1
    print(f"fuzz: {steps} transitions, {bad} mismatches, {ambs} ambiguous")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    sys.exit(FireproofSolver.main())
