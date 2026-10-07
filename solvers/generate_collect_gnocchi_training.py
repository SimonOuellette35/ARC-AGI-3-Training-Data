"""Generate Phase-1 training data for the PuzzleScript game ps:collect_gnocchi
("Collect Gnocchi" by Jonah Ostroff -- eat every gnocchi on the board, but every
square you have stood on is burned behind you).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, a native model of the interpreter, the search over it, and the
optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_collect_gnocchi",
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
presented view, so replaying the recorded actions reproduces the recorded frames
exactly.

The game
--------
Win is ``no Gnocchi``. Four arrow keys plus ACTION (attack). The .txt is written
as a level-EDITING puzzle -- the author's challenge is to replace each level's
key with a gnocchi and keep it solvable -- but the levels as shipped are ordinary
playable puzzles, and that is what this generator solves.

  * **THE TRAIL. Every square the player has occupied is blocked forever.**
    ``[> Player|] -> [> Player > To0|< From0]`` drops a To mark on the square
    being left and a From mark on the square being entered, and both are in
    ``Obstacle``, which ``[> Player|Obstacle] -> Cancel`` refuses to walk into.
    So a level is a SELF-AVOIDING WALK that has to pass through every gnocchi:
    there is no backtracking, no waiting (a blocked press cancels the whole turn)
    and no second chance at a square. This is the whole game, and everything
    below is a way to bend it.

  * **The stomach holds THREE.** Eating fills one of three meter cells; with all
    three full ``late [Player Gnocchi] -> Cancel`` refuses the food, which
    (because the move and the eat are one turn) means the player cannot even
    step onto it. So a level with more than three gnocchi is only winnable if it
    also hands out a way to empty the stomach:

      - **poison** (green wedge) empties it to zero, and can only be eaten with
        something in it -- a free reset that costs a square of trail;
      - **hearty** (red wedge) can only be eaten on an EMPTY stomach and fills
        all three at once, so it is a gnocchi that costs three;
      - **ACTION kills every 4-adjacent golem and spends one meter cell**, which
        is the only other way down. Golems are otherwise pure walls.

  * **Spicy** (black corners) detonates on the square you eat it on:
    ``late [Player Spicy|Explodable] -> [Player Spicy|]`` deletes every gnocchi,
    golem, adjective and key 4-adjacent to the player. It is the only way to
    remove food without eating it, so it is how the big clumps go.

  * **Herby** (magenta side bars) spawns a golem on each side PERPENDICULAR to
    the direction you ate it from, and cancels outright if both sides are
    blocked. Fresh golems are a wall you just built -- and a meter cell you can
    spend later.

  * **Magic** (purple) teleports: eating it slides you forward, in the direction
    you were walking, to the last square before the next obstacle. The squares
    flown over are NOT burned, and neither is the one you land on until you
    leave it, so magic is the one move that crosses your own trail.

  * **Frozen** (light blue) is pushed, not eaten, whenever the square beyond it
    is clear -- and the player does not move, so a push costs a turn but no
    trail. Against a wall it is eaten normally. Pushing one onto a button is how
    two of the button levels open their door.

  * **Keys and buttons.** Walking onto the key deletes every LockedDoor; a
    Weighty (player, gnocchi or golem) landing on an UnpressedButton TOGGLES
    every ButtonDoor -- except that an open door with trail or a gnocchi in it
    cannot close again, which is a real tactic and not a rounding error.

Expert solver
-------------
A NATIVE MODEL of the above (`Model`), searched instead of the interpreter --
600us per interpreter step against 3us for the model, and these searches are
nothing but steps. `selfcheck` fuzzes the model against the real interpreter on
every level over all five keys (settled state, cancel decision and win flag), so
the speed costs no fidelity; every plan is additionally replayed through the
interpreter by `--plans` before it ships.

The search is weighted A* over the model with a two-part heuristic:

  * **the dead test**, which is most of the value. From the player, flood the
    squares that are not wall, not burned and not behind a door that can never
    open again; a gnocchi outside that flood (and, if a spicy is still in play,
    outside its 4-neighbourhood) can never be reached, so the state is dropped
    rather than queued. Sealing yourself off is what almost every wrong move in
    this game does, and without this test the frontier is nothing else.
  * **a greedy nearest-neighbour tour** of the surviving gnocchi over the
    level's static all-pairs distances. It is not admissible in either direction
    (it ignores the trail, which can only lengthen a route, and it ignores
    explosions, which can shorten one), so plans are the shortest FOUND rather
    than proven optima -- but it is the only thing that tells a half-collected
    board from a hopeless one, and the plain "distance to the farthest gnocchi"
    bound leaves the middle of every level flat.

A ladder of weights runs first (w=1, then 2, then 3); the levels that ladder
cannot reach fall through to a width-capped beam over the same heuristic.
**17 of the 19 levels solve**, in 27 to 93 presses (939 total), every plan
replayed through the interpreter to a WIN. Levels 3 and 18 are skipped -- see
`CollectGnocchiSolver.skip_levels` for why level 3 is very likely unwinnable as
shipped rather than just out of reach.

Steps are labelled with the plan's own press, widened by the reordering probe
(`_reorder_optsets`): any later press of the plan that could have been taken now
and still wins by the plan's last step. On these boards that is mostly the two
axes of one diagonal walk.

Engine state after reset is seed-independent (levels are fixed ASCII maps, only
the PRESENTATION is augmented), so seed 0 pays for the searches and every later
seed replays the plan under its own rotation/flip. The start plans and their
optimal sets are cached to ``data/collect_gnocchi_plans.json`` so that every
shard of `parallelize_generator` does not repeat the cold build (~80s for the
seventeen levels; a warm seed is ~1s). Delete the file to re-derive it -- it
re-derives byte-identically, and a cached plan is only served to the state it
was solved from, so an edited level can never replay a stale one.

Augmentation
------------
``Collect_Gnocchi`` is in `PuzzleScriptAdapter._FLIP_GAMES`: rotation_k in
{0,1,2,3} x horizontal x vertical flip with the matching action remap, for 16
presentations of each level. The trail sprites are the only directional art on
the board and they form a closed orbit under that group -- see the note at the
set. No colour augmentation: telling the six gnocchi types apart IS the game.

Art changes ship with this generator, in
``data/puzzlescript_games/Collect_Gnocchi.txt``; each carries a comment there.
The load-bearing one is the TRAIL: the original drew it as one-pixel hairlines,
which the downscale to 3-6 pixel cells deleted outright, so the game's central
mechanic was not in the frame at all. Magic gnocchi were also the same palette
index as plain ones, and spicy and herby marks vanished below cell_px 5. See
`PS palette collisions` and ``--audit``.

Usage (run from the repo root):
    python solvers/generate_collect_gnocchi_training.py --episodes 200 \
        --out data/training_multi_level/collect_gnocchi

    python solvers/generate_collect_gnocchi_training.py --plans      # level report
    python solvers/generate_collect_gnocchi_training.py --selfcheck  # model fuzz
    python solvers/generate_collect_gnocchi_training.py --audit      # render check
"""

from __future__ import annotations

import heapq
import itertools
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (                          # noqa: E402
    PuzzleScriptAdapter, _render_cell_sprite)
from solvers.common.ps_astar import PSAStarSolver, PSExpert           # noqa: E402

GAME_NAME = "Collect_Gnocchi"

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Disk cache of every level's start plan AND its optimal-action sets. The
#: searches are seed-independent but the cold build is minutes, which every
#: shard of `parallelize_generator` would otherwise repeat on every core.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "collect_gnocchi_plans.json"

#: Engine direction per model direction index 0..3; index 4 is ACTION (attack).
DIRS = ("up", "down", "left", "right")
KEYS = DIRS + ("action",)

#: Adjective bits carried by one gnocchi. At most one of each category
#: (sauce / seasoning / dough) can be on a gnocchi, and the shipped levels use
#: at most one bit in total, but the model handles any combination.
POISON, HEARTY, SPICY, HERBY, MAGIC, FROZEN = 1, 2, 4, 8, 16, 32

_ADJ_BITS = (("poison", POISON), ("hearty", HEARTY), ("spicy", SPICY),
             ("herby", HERBY), ("magic", MAGIC), ("frozen", FROZEN))

#: (dr, dc) per direction index, and the two PERPENDICULAR indices per direction.
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))
_PERP = ((2, 3), (2, 3), (0, 1), (0, 1))

#: Object names read out of a level. `read_state` resolves them once per adapter.
_READ_NAMES = ("player", "gnocchi", "golem", "key", "wall", "lockeddoor",
               "buttondoor", "openbuttondoor", "unpressedbutton", "fullness")
_TRAIL_NAMES = ("to0", "tonorth", "tosouth", "toeast", "towest",
                "from0", "fromnorth", "fromsouth", "fromeast", "fromwest")


# ---------------------------------------------------------------------------
# Native model of the interpreter
# ---------------------------------------------------------------------------

class Model:
    """Collect Gnocchi's rules as a pure function on immutable states.

    A state is

        (player, fullness, marked, golems, keys, unlocked,
         dopen, dclosed, buttons, gnocchi)

    where every cell set is an int BITMASK over ``r * w + c`` and ``gnocchi`` is
    a sorted tuple of ``(cell, adjective_mask)``. Walls, locked-door cells and
    the board shape are level constants and live on the model, not in the state.

    `step` returns the next state, or ``None`` when the turn CANCELS -- which
    the interpreter reverts whole, so the caller must treat a cancel as "nothing
    happened at all" and not as "a turn passed".

    Verified against the real interpreter by `selfcheck`; that fuzz is the only
    reason this class is allowed to exist, since a model that drifts from the
    engine produces plans that do not replay.
    """

    def __init__(self, h: int, w: int, walls: int, locked: int):
        self.h, self.w = h, w
        self.walls = walls
        self.locked = locked
        self.mv: list[list[int]] = []          # cell -> neighbour per direction
        self.nbr: list[tuple[int, ...]] = []   # cell -> its on-board neighbours
        for cell in range(h * w):
            r, c = divmod(cell, w)
            row = []
            for dr, dc in _DELTA:
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self.mv.append(row)
            self.nbr.append(tuple(x for x in row if x >= 0))

    # -- the two obstacle sets the rules name ---------------------------------
    def obstacle(self, st) -> int:
        """`Obstacle` -- what cancels a player move: Block, trail, Golem."""
        (_p, _f, marked, golems, _k, unlocked, _do, dclosed, _b, _g) = st
        return (self.walls | dclosed | marked | golems
                | (0 if unlocked else self.locked))

    def gnocchi_obstacle(self, st, gcells: int) -> int:
        """`GnocchiObstacle` -- what stops a magic slide, a frozen push and a
        herby golem spawn: Block, trail, Golem, Gnocchi, Key."""
        (_p, _f, marked, golems, keys, unlocked, _do, dclosed, _b, _g) = st
        return (self.walls | dclosed | marked | golems | keys | gcells
                | (0 if unlocked else self.locked))

    # -- the turn -------------------------------------------------------------
    def step(self, st, di: int):
        """One press; ``None`` if the turn cancels.

        ``di == 4`` is ACTION. The rules below are in the .txt's own order,
        which matters: the eat happens before the magic teleport (so the
        stomach check is made where you stepped) and the spicy blast after it
        (so it goes off where you LANDED)."""
        (p, ful, marked, golems, keys, unlocked,
         dopen, dclosed, buttons, gn) = st

        if di == 4:
            # [ACTION Player|Golem][Fullness] -> spend one meter cell, then
            # [Attack|Golem] -> [Attack|] kills EVERY 4-adjacent golem at once.
            # [ACTION Player] -> Cancel covers both "no golem" and "no food".
            if ful == 0:
                return None
            hit = 0
            for n in self.nbr[p]:
                if golems >> n & 1:
                    hit |= 1 << n
            if not hit:
                return None
            return self._buttons(
                (p, ful - 1, marked, golems & ~hit, keys, unlocked,
                 dopen, dclosed, buttons, gn))

        t = self.mv[p][di]
        if t < 0:
            return None                       # off the board: nothing to enter
        if self.obstacle(st) >> t & 1:
            return None                       # [> Player|Obstacle] -> Cancel

        gd = dict(gn)
        gcells = 0
        for cell in gd:
            gcells |= 1 << cell
        mask = gd.get(t, 0)

        # -- frozen: pushed if the square beyond is clear, and the PLAYER stays
        #    put, so a push burns a turn but no trail.
        if mask & FROZEN:
            b = self.mv[t][di]
            if b >= 0 and not (self.gnocchi_obstacle(st, gcells) >> b & 1):
                del gd[t]
                gd[b] = mask                  # [> Frozen NonDough] carries the
                return self._buttons(         # sauce/seasoning along with it
                    (p, ful, marked, golems, keys, unlocked, dopen, dclosed,
                     buttons, tuple(sorted(gd.items()))))

        # -- the player moves: To on the square it leaves, From on the one it
        #    enters, and both are Obstacles from here on.
        marked |= (1 << p) | (1 << t)
        st2 = (p, ful, marked, golems, keys, unlocked, dopen, dclosed,
               buttons, gn)
        obst = self.gnocchi_obstacle(st2, gcells)

        # -- magic: the destination marker slides on until it is blocked ------
        dest = t
        if mask & MAGIC:
            cur = t
            while True:
                nxt = self.mv[cur][di]
                if nxt < 0 or (obst >> nxt & 1):
                    break
                cur = nxt
            dest = cur

        # -- herby: a golem on each side PERPENDICULAR to the travel, and no
        #    room for either is a cancelled turn. With magic the herby marker
        #    rides to the destination first, so the golems appear there.
        if mask & HERBY:
            d1, d2 = _PERP[di]
            s1, s2 = self.mv[dest][d1], self.mv[dest][d2]
            b1 = s1 < 0 or bool(obst >> s1 & 1)
            b2 = s2 < 0 or bool(obst >> s2 & 1)
            if s1 >= 0 and s2 >= 0 and b1 and b2:
                return None
            if not b1:
                golems |= 1 << s1
            if not b2:
                golems |= 1 << s2

        # -- eat (late rules, in file order: poison, hearty, everything else) --
        if t in gd:
            if mask & POISON:
                if ful == 0:
                    return None               # nothing to purge -> Cancel
                ful = 0
            elif mask & HEARTY:
                if ful > 0:
                    return None               # only on an empty stomach
                ful = 3
            else:
                if ful >= 3:
                    return None               # no room -> Cancel
                ful += 1
            del gd[t]

        p = dest                              # magic teleport

        # -- spicy: everything Explodable 4-adjacent to where the player ended
        if mask & SPICY:
            for n in self.nbr[p]:
                gd.pop(n, None)
                golems &= ~(1 << n)
                keys &= ~(1 << n)

        # -- the key opens every LockedDoor at once, and is spent either way --
        if keys >> p & 1:
            keys &= ~(1 << p)
            unlocked = True

        return self._buttons(
            (p, ful, marked, golems, keys, unlocked, dopen, dclosed, buttons,
             tuple(sorted(gd.items()))))

    def _buttons(self, st):
        """The `+late` button group: every UnpressedButton under a Weighty
        (Player, Gnocchi or Golem) presses, and each press TOGGLES every button
        door -- except that an OpenButtonDoor holding a GnocchiObstacle (trail,
        gnocchi, golem) cannot close, which is why walking through an open door
        jams it open."""
        (p, ful, marked, golems, keys, unlocked,
         dopen, dclosed, buttons, gn) = st
        gcells = 0
        for cell, _m in gn:
            gcells |= 1 << cell
        newly = buttons & ((1 << p) | golems | gcells)
        if not newly:
            return st
        buttons &= ~newly
        blockers = marked | golems | keys | gcells
        for _ in range(bin(newly).count("1")):
            stuck = dopen & blockers
            dopen, dclosed = dclosed | stuck, dopen & ~stuck
        return (p, ful, marked, golems, keys, unlocked, dopen, dclosed,
                buttons, gn)

    @staticmethod
    def won(st) -> bool:
        return not st[9]


# ---------------------------------------------------------------------------
# Reading a model state out of the interpreter
# ---------------------------------------------------------------------------

def read_state(game, eng) -> tuple:
    """``(h, w, walls, locked, state)`` from a live interpreter grid."""
    idx = game.obj_name_to_idx
    ids = {name: idx[name] for name in _READ_NAMES}
    trail = [idx[name] for name in _TRAIL_NAMES]
    adjectives = [(idx[name], bit) for name, bit in _ADJ_BITS]
    h, w = len(eng.grid), len(eng.grid[0])
    walls = locked = marked = golems = keys = dopen = dclosed = buttons = 0
    gd = {}
    player = -1
    ful = 0
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if not cell:
                continue
            bit = 1 << (r * w + c)
            if ids["wall"] in cell:
                walls |= bit
            if ids["lockeddoor"] in cell:
                locked |= bit
            if any(i in cell for i in trail):
                marked |= bit
            if ids["golem"] in cell:
                golems |= bit
            if ids["key"] in cell:
                keys |= bit
            if ids["openbuttondoor"] in cell:
                dopen |= bit
            if ids["buttondoor"] in cell:
                dclosed |= bit
            if ids["unpressedbutton"] in cell:
                buttons |= bit
            if ids["fullness"] in cell:
                ful += 1
            if ids["player"] in cell:
                player = r * w + c
            if ids["gnocchi"] in cell:
                m = 0
                for i, bitv in adjectives:
                    if i in cell:
                        m |= bitv
                gd[r * w + c] = m
    state = (player, ful, marked, golems, keys, not locked, dopen, dclosed,
             buttons, tuple(sorted(gd.items())))
    return h, w, walls, locked, state


def model_from_engine(eng, game) -> tuple:
    """``(Model, state)`` for the interpreter's current grid."""
    h, w, walls, locked, state = read_state(game, eng)
    return Model(h, w, walls, locked), state


# ---------------------------------------------------------------------------
# Heuristic
# ---------------------------------------------------------------------------

class Heuristic:
    """Distance-to-go estimate, and the dead test that does the real work.

    The dead test floods the squares the player can still stand on -- not wall,
    not burned, not behind a door whose key or button is gone -- and calls the
    state dead when a gnocchi is outside it (a spicy still in play widens that
    to the 4-neighbourhood, since a blast reaches one square). It is exact in
    the direction that matters: it never calls a live state dead, because every
    obstacle it walks through (golems, doors that can still open) really is
    removable.

    The estimate itself is a greedy nearest-neighbour tour of the surviving
    gnocchi over ``self.dist``, all-pairs shortest paths on the walls-only
    board. Inadmissible both ways -- it ignores the trail (which can only
    lengthen a route) and explosions (which can shorten one) -- so it is a
    guide, not a bound, and the plans it yields are shortest FOUND.
    """

    def __init__(self, model: Model):
        self.model = model
        self.dist = self._all_pairs(model)

    @staticmethod
    def _all_pairs(model: Model) -> list:
        """Shortest paths between open squares, ignoring everything removable.
        One BFS per square: a few hundred squares, once per level."""
        n = model.h * model.w
        inf = 1 << 20
        out: list = []
        for src in range(n):
            if model.walls >> src & 1:
                out.append(None)
                continue
            d = [inf] * n
            d[src] = 0
            queue = deque([src])
            while queue:
                cur = queue.popleft()
                for nxt in model.nbr[cur]:
                    if d[nxt] == inf and not (model.walls >> nxt & 1):
                        d[nxt] = d[cur] + 1
                        queue.append(nxt)
            out.append(d)
        return out

    def reachable(self, st) -> set:
        model = self.model
        (p, _f, marked, _g, keys, unlocked, _do, dclosed, buttons, _gn) = st
        blocked = model.walls | marked
        if not unlocked and keys == 0:
            blocked |= model.locked           # the key is gone: never opens
        if buttons == 0:
            blocked |= dclosed                # no button left: never opens
        seen = {p}
        queue = deque([p])
        while queue:
            cur = queue.popleft()
            for nxt in model.nbr[cur]:
                if nxt not in seen and not (blocked >> nxt & 1):
                    seen.add(nxt)
                    queue.append(nxt)
        return seen

    def __call__(self, st):
        """Estimated presses to the win, or ``None`` when the state is dead."""
        gn = st[9]
        if not gn:
            return 0
        seen = self.reachable(st)
        spicy = any(m & SPICY for _c, m in gn)
        targets = []
        for cell, _m in gn:
            if cell in seen:
                targets.append(cell)
            elif spicy and any(n in seen for n in self.model.nbr[cell]):
                targets.append(cell)
            else:
                return None
        cur = st[0]
        total = 0
        rest = set(targets)
        dist = self.dist
        while rest:
            row = dist[cur]
            nearest = min(rest, key=lambda t: row[t])
            total += row[nearest]
            cur = nearest
            rest.discard(nearest)
        return total


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class Plan(list):
    """A winning press list that also carries its per-step optimal sets."""

    def __init__(self, presses, optsets):
        super().__init__(presses)
        self.optsets = optsets


class CollectGnocchiExpert(PSExpert):
    """Weighted A* / beam over `Model`, memoized and cached to disk.

    `PSExpert`'s own engine-blackbox A* is not used: the interpreter is ~600us
    per step and a self-avoiding-walk search is millions of steps. Everything
    below runs on the native model and only the finished plan touches the
    engine (in `--plans`, and in the recorder that replays it)."""

    directions = list(KEYS)

    #: (weight, node cap) rungs, tried in order. w=1 gets most levels; the open
    #: boards need the extra greed before the tour heuristic stops being flat.
    ladder = ((1, 400_000), (2, 1_500_000), (3, 1_500_000))
    #: Beam widths for the levels the ladder cannot reach at all.
    beam_widths = (2_000, 8_000, 30_000)
    #: Depth cap for the beam, in presses. The longest plan any level needs is
    #: under 100; this leaves room and bounds a hopeless level's cost.
    beam_depth = 220

    def __init__(self, game, node_cap: int = 400_000, weight: int = 1):
        super().__init__(game, node_cap=node_cap, weight=weight)
        self._memo: dict = {}
        self._how: dict = {}
        self._disk = self._load_disk()

    # -- searches -------------------------------------------------------------
    @staticmethod
    def _astar(model: Model, start: tuple, h: Heuristic, weight: int,
               cap: int):
        """Weighted A*. Parent pointers rather than a path per heap entry: at a
        million nodes a copied tuple per node is most of the memory."""
        if Model.won(start):
            return ()
        best = {start: 0}
        parent: dict = {start: None}
        counter = 0
        nodes = 0
        pq = [(0, 0, 0, start)]
        while pq:
            _f, g, _c, st = heapq.heappop(pq)
            if best.get(st, -1) != g:
                continue                      # a stale heap entry
            for di in range(5):
                nxt = model.step(st, di)
                if nxt is None:
                    continue
                nodes += 1
                if Model.won(nxt):
                    out = [di]
                    cur = st
                    while parent[cur] is not None:
                        cur, pd = parent[cur]
                        out.append(pd)
                    out.reverse()
                    return tuple(out)
                hv = h(nxt)
                if hv is None:
                    continue                  # dead: a gnocchi is sealed off
                ng = g + 1
                if best.get(nxt, 1 << 30) <= ng:
                    continue
                best[nxt] = ng
                parent[nxt] = (st, di)
                counter += 1
                heapq.heappush(pq, (ng + weight * hv, ng, counter, nxt))
            if nodes >= cap:
                return None
        return None

    @classmethod
    def _beam(cls, model: Model, start: tuple, h: Heuristic, width: int):
        """Breadth-first beam over the same successors, ranked by `h`.

        What A* runs out of on the two widest levels is not nodes but guidance:
        the tour estimate barely moves while the player crosses a room, so the
        frontier spreads instead of advancing. A beam spends the same budget on
        breadth at every depth and only uses the estimate to decide who
        survives, which is all a flat estimate is good for. Plans are winning
        and engine-verified but wander more than A*'s."""
        if Model.won(start):
            return ()
        frontier = [(start, ())]
        seen = {start}
        for _depth in range(cls.beam_depth):
            kids = []
            for st, path in frontier:
                for di in range(5):
                    nxt = model.step(st, di)
                    if nxt is None or nxt in seen:
                        continue
                    if Model.won(nxt):
                        return path + (di,)
                    hv = h(nxt)
                    if hv is None:
                        continue
                    seen.add(nxt)
                    kids.append((hv, nxt, path + (di,)))
            if not kids:
                return None                   # the reachable space closed
            kids.sort(key=lambda kid: kid[0])
            frontier = [(st, path) for _h, st, path in kids[:width]]
        return None

    def _search(self, model: Model, start: tuple, level):
        h = Heuristic(model)
        for weight, cap in self.ladder:
            path = self._astar(model, start, h, weight, cap)
            if path is not None:
                self._how[level] = f"A* w={weight}"
                return path
        for width in self.beam_widths:
            path = self._beam(model, start, h, width)
            if path is not None:
                self._how[level] = f"beam {width}"
                return path
        self._how[level] = "unsolved"
        return None

    # -- optimal sets ---------------------------------------------------------
    @staticmethod
    def _reorder_optsets(model: Model, start: tuple, path: tuple) -> list:
        """Per-step optimal sets: the press taken, plus every LATER press of the
        plan that could have been taken first.

        A press ``d`` from further down the plan is as good as the one being
        taken when pulling it to the front and replaying the rest still wins by
        the recorded plan's last step -- the two commute, which on these boards
        means the two axes of one diagonal walk, or a detour whose trail the
        rest of the plan never needed. That is the widening available without an
        exact distance-to-win field, which a self-avoiding walk does not admit
        (the trail is part of the state, so no two routes to a square are
        interchangeable in general). It never claims a press is optimal in the
        absolute sense -- these plans are the shortest FOUND -- only that it is
        at least as good as what the expert did, which is what the
        demonstration teaches either way.

        The whole suffix is replayed rather than a two-press commutation test:
        a press that commutes with the next one but seals a corridor ten turns
        later is not equally good, and on the model a full replay costs
        microseconds. A press the model CANCELS is replayed as a wasted turn,
        exactly as the interpreter would treat it, so a permutation that needs
        one can never come out shorter than the plan."""
        states = [start]
        for di in path:
            states.append(model.step(states[-1], di))
        optsets = []
        for i, taken in enumerate(path):
            suffix = path[i:]
            best = {taken}
            for j in range(1, len(suffix)):
                if suffix[j] in best:
                    continue
                st = states[i]
                for di in (suffix[j],) + suffix[:j] + suffix[j + 1:]:
                    nxt = model.step(st, di)
                    if nxt is not None:
                        st = nxt
                    if Model.won(st):
                        best.add(suffix[j])
                        break
            optsets.append([KEYS[d] for d in sorted(best)])
        return optsets

    # -- planning -------------------------------------------------------------
    def plan(self, eng, level: int | None = None):
        """The winning press sequence from the interpreter's current state (a
        `Plan`, carrying its optimal sets), or None if the search cannot find
        one. Leaves the interpreter untouched -- everything runs on the model.

        The level's START state is also kept on disk: it is the only state any
        seed ever plans from (recovery is a RESET back to it), and re-deriving
        it costs minutes in every process."""
        if eng.check_win():
            return []
        model, state = model_from_engine(eng, self.g)
        memo = (level, state)
        if memo in self._memo:
            return self._memo[memo]
        cached = self._disk.get(level)
        sig = self._signature(state)
        if cached is not None and cached["start"] == sig:
            found = (Plan(cached["plan"], cached["optsets"])
                     if cached["plan"] is not None else None)
            self._memo[memo] = found
            return found
        path = self._search(model, state, level)
        found = (None if path is None else
                 Plan([KEYS[d] for d in path],
                      self._reorder_optsets(model, state, path)))
        self._memo[memo] = found
        if level is not None:
            # Unconditional: reaching here means the disk entry did not serve
            # this state, so it is either missing or STALE (the level file
            # changed). Leaving a stale entry in place would make every process
            # re-run the search it was supposed to save.
            self._disk[level] = {
                "start": sig,
                "plan": None if found is None else list(found),
                "optsets": None if found is None else found.optsets,
            }
            self._save_disk()
        return found

    @staticmethod
    def _signature(state: tuple) -> list:
        """JSON-safe form of a model state, for the disk cache's staleness
        check: a cached plan is only ever served to the state it was solved
        from, so an edited level file can never replay a stale plan."""
        (p, ful, marked, golems, keys, unlocked,
         dopen, dclosed, buttons, gn) = state
        return [p, ful, str(marked), str(golems), str(keys), bool(unlocked),
                str(dopen), str(dclosed), str(buttons),
                [[c, m] for c, m in gn]]

    def describe(self, level) -> str:
        """Which search answered for this level (for `--plans`); "cached" when
        the plan came off disk and nothing ran."""
        return self._how.get(level, "cached")

    # -- disk cache -----------------------------------------------------------
    @staticmethod
    def _load_disk() -> dict:
        try:
            raw = json.loads(PLAN_CACHE.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:                                    # noqa: BLE001
            return {}                                        # a miss, not a crash

    def _save_disk(self) -> None:
        """Write atomically -- shards started together would otherwise
        interleave into a truncated file (see `parallelize_generator`)."""
        try:
            PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PLAN_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self._disk[k] for k in sorted(self._disk)}, indent=1))
            os.replace(tmp, PLAN_CACHE)
        except OSError:
            pass                                             # cache is optional


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class CollectGnocchiSolver(PSAStarSolver):
    game_id = "puzzlescript_collect_gnocchi"
    game_name = GAME_NAME
    expert_cls = CollectGnocchiExpert

    #: Levels no rung of the ladder and no beam width reaches (the beam runs the
    #: reachable space dry around depth 70 on both). Skipped up front so a cold
    #: build does not spend its whole budget on them; the other 17 all solve.
    #:
    #: Level 3 looks unwinnable rather than merely hard, on its food economy
    #: alone: 21 gnocchi with no poison and no spicy means 21 eats and no way
    #: down but ACTION, so the stomach needs at least 18 attacks against a board
    #: holding 21 golems -- and its left room is a perfect checkerboard, where
    #: the first attack from any square kills every golem around it three or
    #: four at a time. The budget is spent before the room is a third eaten.
    #: Level 18 is the same shape of tight, without the clean argument.
    skip_levels: frozenset[int] = frozenset({3, 18})

    #: The recorder's own loop cap. Plans run to ~95 presses and the adapter
    #: gives a level 200 actions (the RESET in the recovery prefix zeroes that
    #: counter, so only the post-reset plan is charged against it).
    max_steps = 200

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup
        rather than as a mysteriously slow first seed."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)

    def optimal_for(self, expert, level: int, plan, pi: int):
        """The optimal press set at this step, off the plan's own optsets: the
        press taken, widened by the reordering probe. Falls back to the press
        about to be taken so no expert step ever ships unlabelled --
        `train_policy` v2 supervises ``optimal`` only, so a step without one
        contributes nothing to the loss."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets) and sets[pi]:
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Model self-check (fuzz against the real interpreter)
# ---------------------------------------------------------------------------

def selfcheck(trials: int = 60, steps: int = 60, verbose: bool = True) -> int:
    """Audit the claim the whole solver rests on: `Model` reproduces the
    interpreter EXACTLY. On random rollouts from every level start, over all
    five keys, the settled state, the cancel/no-cancel decision and the win flag
    all have to agree. That is what licenses planning off the engine.

    Random play reaches the corners a plan never would -- eating a herby with
    one side walled (which cancels the turn outright), shoving a frozen gnocchi
    onto a button, teleporting into a dead end -- so the rollouts are the fuzz,
    and they run long enough for the trail to seal the board off."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        cancels = wins = 0
        for t in range(trials):
            game.set_level(level)
            model, state = model_from_engine(eng, game._game)
            rng = random.Random(f"collect_gnocchi:selfcheck:{level}:{t}")
            for _ in range(steps):
                before = [[set(c) for c in row] for row in eng.grid]
                di = rng.randrange(len(KEYS))
                eng.step(KEYS[di])
                frozen = eng.grid == before
                predicted = model.step(state, di)
                expected = state if predicted is None else predicted
                _m, actual = model_from_engine(eng, game._game)
                if actual != expected:
                    bad += 1
                    print(f"  L{level}: state divergence on {KEYS[di]}")
                    for i, (a, b) in enumerate(zip(expected, actual)):
                        if a != b:
                            print(f"    field {i}: model {a} engine {b}")
                    break
                if (predicted is None) != frozen:
                    bad += 1
                    print(f"  L{level}: cancel divergence on {KEYS[di]} "
                          f"(engine frozen={frozen}, model cancel="
                          f"{predicted is None})")
                    break
                if eng.check_win() != Model.won(expected):
                    bad += 1
                    print(f"  L{level}: win divergence on {KEYS[di]}")
                    break
                if predicted is None:
                    cancels += 1
                else:
                    state = predicted
                if eng.check_win():
                    wins += 1
                    break
        if verbose:
            print(f"  L{level}: {'OK' if not bad else 'VIOLATIONS'} "
                  f"({trials} rollouts, {cancels} cancelled turns, "
                  f"{wins} accidental wins)")
        if bad:
            break
    return bad


# ---------------------------------------------------------------------------
# Render audit
# ---------------------------------------------------------------------------

#: Every cell composition a level can show, as the object stack that draws it.
_AUDIT_CASES = {
    "floor": ["floor"],
    "wall": ["floor", "wall"],
    "player": ["floor", "player"],
    "trail (to+from)": ["floor", "tonorth", "fromsouth"],
    "trail (to only)": ["floor", "toeast"],
    "trail (from only)": ["floor", "fromwest"],
    "golem": ["floor", "golem"],
    "key": ["floor", "pedestal", "key"],
    "locked door": ["floor", "lockeddoor"],
    "button door": ["floor", "buttondoor"],
    "open button door": ["floor", "openbuttondoor"],
    "button (up)": ["floor", "unpressedbutton"],
    "button (down)": ["floor", "pressedbutton"],
    "gnocchi": ["floor", "gnocchi"],
    "poison": ["floor", "gnocchi", "poison"],
    "hearty": ["floor", "gnocchi", "hearty"],
    "spicy": ["floor", "gnocchi", "spicy"],
    "herby": ["floor", "gnocchi", "herby"],
    "magic": ["floor", "gnocchi", "magic"],
    "frozen": ["floor", "gnocchi", "frozen"],
    "stomach (empty)": ["floor", "stomachmeter"],
    "stomach (full)": ["floor", "stomachmeter", "fullness"],
}


def audit(verbose: bool = True) -> int:
    """Check that every cell composition is pixel-distinct at every cell size
    the levels use.

    This game renders at 3 to 6 pixels per cell and stacks up to four sprites
    per cell, so it is exactly the case where a sprite detail or a palette
    collision silently deletes a mechanic from the frame -- which is what the
    art changes in the .txt fix. Returns the number of colliding pairs."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    parsed = game._game
    layers = game._engine._obj_layers
    sizes = set()
    for level in range(game.n_levels):
        game.set_level(level)
        h, w = len(game._engine.grid), len(game._engine.grid[0])
        sizes.add(max(1, min(64 // h, 64 // w)))

    def stack(names):
        objs = [(layers.get(parsed.obj_name_to_idx[n], -1), parsed.objects[n])
                for n in names]
        objs.sort(key=lambda x: x[0])
        return objs

    bad = 0
    for px in sorted(sizes):
        blocks = {k: _render_cell_sprite(stack(v), px)
                  for k, v in _AUDIT_CASES.items()}
        for a, b in itertools.combinations(_AUDIT_CASES, 2):
            if (blocks[a] == blocks[b]).all():
                bad += 1
                print(f"  cell_px={px}: {a} and {b} render identically")
        if verbose:
            print(f"  cell_px={px}: {len(_AUDIT_CASES)} cell types, "
                  f"{'all distinct' if not bad else 'COLLISIONS'}")
    return bad


def _plan_report() -> None:
    """Print every level's plan, how it was found and how many of its steps have
    more than one right answer -- the quick "is this game still fully solved"
    check. Every plan is replayed through the real interpreter, so this is also
    the model's end-to-end test."""
    solver = CollectGnocchiSolver()
    game, expert, solvable = solver._ensure(0)
    total = ties = 0
    for level in range(game.n_levels):
        if level in solver.skip_levels:
            print(f"  L{level}: skipped")
            continue
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            continue
        for direction in plan:                          # interpreter-verify it
            eng.step(direction)
        step_ties = sum(len(s) - 1 for s in plan.optsets)
        total += len(plan)
        ties += step_ties
        print(f"  L{level:2d}: {len(plan):3d} presses  win={eng.check_win()}  "
              f"{expert.describe(level):>10}  {step_ties:3d} tie-presses")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second right answer")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit()
        print(f"audit: {collisions} colliding cell types")
        sys.exit(1 if collisions else 0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(CollectGnocchiSolver.main())
