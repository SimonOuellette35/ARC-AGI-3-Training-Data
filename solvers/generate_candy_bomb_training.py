"""Generate Phase-1 training data for the PuzzleScript game ps:candy_bomb
("Candy Bomb" by Jonathan Brodsky -- push lit bombs next to candy, then be
somewhere else when they go off).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, a native model of the interpreter, the search over it, and the
optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_candy_bomb",
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
Four arrow keys; ACTION is a press with no target cell (``noaction``, and no rule
binds it), which makes it a permanently blocked move rather than a true no-op --
see `Model.step`. Win is
``no WinCandy`` and ``no PlayerDead``: blow up every Candy and HardCandy without
standing next to a blast. Push rules are a plain chain sokoban over
``Pushable = Candy or HardCandy or Bomb``, and everything interesting is in the
five mechanics below.

  * **A bomb is a CLOCK, and walking into it starts it.**
    ``[ > Player PlayerShadow | Bomb0 ] -> [ > Player | > Bomb1 ]`` lights a
    Bomb0 the moment the player shoves it; from there
    ``Bomb1 -> Bomb2 -> Bomb3 -> Explosion`` ticks once per turn, so lighting on
    turn ``t`` detonates on turn ``t + 3``. A Bomb1Unlit ("shorter fuse") lights
    straight to Bomb2 and detonates on ``t + 2``. **A bomb pushed by a CANDY is
    not lit** -- only the player's own shove starts a clock -- which is what the
    file's "(allow candies to push bombs)" comment is about.

  * **The turn cancels unless the player MOVES.** ``[Player] -> [Player
    PlayerShadow]`` stamps a shadow under the player before movement; the shadow
    does not move, so ``late [Player PlayerShadow] -> Cancel`` fires exactly when
    the player ended the turn where it started. A cancelled turn is reverted
    WHOLE -- no fuse tick, no detonation. **So the player cannot wait.** Killing
    a single turn costs two presses (step out and back), which is why level 12's
    source carries the comment "(correct parity offset from the top)": half the
    board's cells are the wrong colour to be standing on when a bomb goes off.

  * **...except that shoving an unlit bomb always counts.** Rules 16/17 clear the
    shadow as they light the bomb, so a Bomb0 wedged against a wall can be lit
    without the player going anywhere and the turn still stands. That is the only
    single-press "wait" in the game, and level 10's bomb stack needs it.

  * **The blast is a plus, and it reaches the player too.**
    ``late [ Player | Explosion ] -> [ PlayerDead | Explosion ]`` -- 4-adjacent
    to an explosion is death, and death is silent: no GAME_OVER, the level simply
    can never satisfy ``no PlayerDead`` again. Candy dies the same way
    (4-adjacent), a **HardCandy needs explosions on BOTH sides of one axis at the
    same instant** (two bombs, timed together), and a bomb 4-adjacent to a blast
    has its own fuse advanced one step -- which is how chain reactions ripple,
    one cell per turn. An explosion exists for exactly one turn.

  * **The lethal-turn exception to the cancel rule.** The death rule is written
    ABOVE the cancel rule, so on a turn where the player is blocked *and* a blast
    lands next to it, the Player is already PlayerDead when
    ``[Player PlayerShadow]`` is tested -- nothing matches, nothing cancels, and
    the turn stands. Walking into a wall does NOT save you from a bomb. This is
    the one rule-order subtlety in the file and the model got it wrong first;
    `selfcheck` covers it.

Expert solver
-------------
A NATIVE MODEL of the above (`Model`), searched instead of the interpreter. The
interpreter costs 560us per step and the model 1.5us -- a 370x difference, and
these searches are nothing but steps. `selfcheck` fuzzes the model against the
real interpreter (random rollouts on every level, comparing the settled state,
the cancel/no-cancel decision and the win flag), so the speed costs no fidelity;
every plan is additionally replayed through the interpreter before it ships.

Two planners over that model, in this order:

  * **The exact field.** Forward BFS from the level start, pruning states that
    can never win (the player is dead, or the last bomb has been spent with candy
    still standing -- both are permanent and both are most of the branching),
    stopping at the depth where the first win appears, then one backward BFS from
    the winning transitions. That gives the true distance-to-win for every state
    on an optimal route, so the plan is optimal by construction AND every step
    can be labelled with the COMPLETE set of equally-shortest presses rather than
    with one arbitrary tie-break. 11 of the 14 levels close this way, the biggest
    in 249k states.

  * **A-star ladder**, for the three levels whose reachable space does not
    (9, 12, 13 -- open boards where the candy itself can be pushed anywhere, so
    the space is millions of states wide). The heuristic is, per surviving candy,
    the cheapest bomb that could still reach it: a lit bomb costs its remaining
    fuse and can only be shoved as far as that fuse allows, an unlit one costs
    the walk to it plus its fuse. ``max`` over the candies is a genuine lower bound but
    goes FLAT on a board with several candies left, so the ladder re-runs with
    ``K`` presses of credit per surviving candy added on top -- inadmissible, but
    it is what makes levels 9 and 13 close at all. First rung to win, wins.

Levels 9/12/13 therefore ship a plan that is the shortest FOUND, not a proven
optimum; their steps are labelled with the plan's own press, widened by a
reordering probe: any later press of the plan that could equally have been taken
first (it commutes with everything between) is in the set too.

All 14 levels solve: 5 / 13 / 10 / 12 / 30 / 12 / 3 / 4 / 17 / 26 / 19 / 19 / 19
/ 25 presses.

Engine state after reset is seed-independent here (levels are fixed ASCII maps,
only the PRESENTATION is augmented), so seed 0 pays for the searches and every
later seed replays the plan under its own rotation/flip. The cold build is about
a minute, nearly all of it levels 9/12/13, so the start plans and their optimal
sets are also written to ``data/candy_bomb_plans.json`` -- without it every shard
of `parallelize_generator` would repeat that minute on every core. Delete the
file to re-derive it.

Augmentation
------------
Candy Bomb is gravity-free, every rule is either relative (``>``) or
direction-agnostic (the blast rules match all four sides, and the HardCandy
pattern is symmetric), and input is screen-relative -- so the board's full
8-element symmetry group is a valid presentation augmentation. ``Candy_Bomb`` is
in `PuzzleScriptAdapter._FLIP_GAMES`: rotation_k in {0,1,2,3} x horizontal x
vertical flip, each with the matching directional action remap, for 16
presentations of each of the 14 levels. No colour augmentation -- telling the
four fuse states apart is the game, and a flattening recolor could merge them.

One sprite change ships with this generator, in
``data/puzzlescript_games/Candy_Bomb.txt``: Bomb1Unlit's fuse is painted purple
instead of brown. Its art was Bomb0's art minus a single pixel, which at the 7px
cells of the biggest levels is a ONE-pixel difference standing for the whole
3-turn-versus-2-turn distinction the game teaches in level 7 and then tests with
both bomb types on one board in levels 11-13. See `PS palette collisions`.

Usage (run from the repo root):
    python solvers/generate_candy_bomb_training.py --episodes 200 \
        --out data/training_multi_level/candy_bomb

    python solvers/generate_candy_bomb_training.py --plans      # level report
    python solvers/generate_candy_bomb_training.py --selfcheck  # model fuzz
"""

from __future__ import annotations

import heapq
import json
import os
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import PuzzleScriptAdapter        # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert          # noqa: E402

GAME_NAME = "Candy_Bomb"

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Disk cache of every level's start plan AND its optimal-action sets. The
#: searches are exact and seed-independent but cost about a minute for the
#: fourteen levels, which every shard of `parallelize_generator` would otherwise
#: repeat on every core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "candy_bomb_plans.json"

#: Engine direction per model direction index. The model indexes directions
#: 0..3; everything the recorder sees is these names.
DIRS = ("up", "down", "left", "right")

#: The five keys the adapter offers. Index 4 is ACTION, which the model accepts
#: and treats as a press with no target cell -- see `Model.step`. The search
#: never branches on it (`CandyBombExpert.directions`); it is here so
#: `selfcheck` can hold the model to the interpreter on all five keys.
KEYS = DIRS + ("action",)

# Piece codes. Ordered so that ``code <= HARD`` is "a thing that must be
# destroyed" and ``code >= B0`` is "a bomb", which is every classification the
# search does.
CANDY, HARD, B0, BU, B1, B2, B3 = range(7)

#: One turn of fuse, applied before movement.
_TICK = {B1: B2, B2: B3}
#: What the player's own shove does to an unlit bomb (and it clears the shadow,
#: so the turn stands even when the bomb cannot move).
_LIGHT = {B0: B1, BU: B2}
#: What being 4-adjacent to a blast does to a bomb. Bomb3 is absent on purpose:
#: no rule advances it, it is already detonating next turn.
_BOOST = {B2: B3, B1: B2, BU: B2, B0: B1}
#: Turns until a lit bomb detonates.
_TTL = {B1: 3, B2: 2, B3: 1}
#: Turns from "the player shoves it" to the blast, per unlit bomb type.
_FUSE = {B0: 3, BU: 2}

#: Object name in the .txt -> piece code.
_NAME2CODE = {"candy": CANDY, "hardcandy": HARD, "bomb0": B0, "bomb1unlit": BU,
              "bomb1": B1, "bomb2": B2, "bomb3": B3}

#: Heuristic value for "no bomb can plausibly reach this candy any more". Big
#: enough to sink the node, finite because that judgement is only a guess (the
#: candy itself is pushable, so a pairing this test calls impossible sometimes
#: is not) and a state must never be PRUNED on it.
_BIG = 60

#: Sentinel successor for a press that ends the level.
_WIN = "win"


# ---------------------------------------------------------------------------
# Native model of the interpreter
# ---------------------------------------------------------------------------

class Model:
    """Candy Bomb's rules, as a pure function on immutable states.

    A state is ``(player_cell, pieces, explosions)`` where ``player_cell`` is a
    flat ``r * w + c`` index (-1 once the player is dead), ``pieces`` is a sorted
    tuple of ``(cell, code)`` and ``explosions`` is a tuple of cells currently
    blasting. Walls and the board shape are level constants and live on the
    model, not in the state.

    `step` returns the next state or ``None`` when the turn CANCELS -- which the
    caller must treat as "nothing happened at all", because that is what the
    interpreter does: a cancel reverts the fuse tick too.

    Verified against the real interpreter by `selfcheck`; that fuzz is the only
    reason this file is allowed to exist, since a model that drifts from the
    engine produces plans that do not replay.
    """

    def __init__(self, h: int, w: int, walls: set):
        self.h, self.w = h, w
        self.walls = walls
        self.mv = []          # cell -> [up, down, left, right] neighbour or -1
        self.nbr = []         # cell -> tuple of its on-board neighbours
        for cell in range(h * w):
            r, c = divmod(cell, w)
            row = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w else -1)
            self.mv.append(row)
            self.nbr.append(tuple(x for x in row if x >= 0))

    def step(self, state: tuple, di: int):
        """One press in direction ``di``; ``None`` if the turn cancels.

        ``di == 4`` is ACTION. The .txt declares ``noaction`` and no rule binds
        it, so it is a press with no target cell -- which is NOT the same as
        doing nothing: like walking into a wall it normally cancels the turn, but
        on a turn where a blast lands beside the player the death rule fires
        first and the turn stands. It is exactly a permanently blocked move, so
        it is dominated by every real move and the search drops it."""
        p, pieces, _expl = state
        mvd = self.mv
        walls = self.walls

        # -- tick: last turn's blast clears, fuses advance, Bomb3 detonates ----
        live = {}
        blast = []
        for cell, code in pieces:
            if code == B3:
                blast.append(cell)
            else:
                live[cell] = _TICK.get(code, code)
        blast_s = set(blast)

        # -- movement ---------------------------------------------------------
        lit = False
        moved = False
        tgt = mvd[p][di] if (p >= 0 and di < 4) else -1
        if tgt >= 0 and tgt not in walls and tgt not in blast_s:
            if tgt in live:
                chain = []
                cur = tgt
                while cur in live:
                    chain.append(cur)
                    cur = mvd[cur][di]
                    if cur < 0:
                        break
                head = live[chain[0]]
                if head in _LIGHT:
                    # Rules 16/17: the shove lights the bomb and clears the
                    # shadow whether or not the push goes through.
                    live[chain[0]] = _LIGHT[head]
                    lit = True
                if cur >= 0 and cur not in walls and cur not in blast_s:
                    for cell in reversed(chain):
                        live[mvd[cell][di]] = live.pop(cell)
                    p = tgt
                    moved = True
            else:
                p = tgt
                moved = True

        # -- late rules -------------------------------------------------------
        # Death is tested FIRST because its rule is written above the cancel
        # rule: a player killed this turn is no longer a Player, so
        # [Player PlayerShadow] finds nothing and the turn is NOT cancelled.
        # Walking into a wall is not an escape.
        dead = p < 0
        if blast_s and not dead:
            for n in self.nbr[p]:
                if n in blast_s:
                    dead = True
                    break
        if not moved and not lit and not dead:
            return None                                   # PlayerShadow cancel
        if blast_s:
            drop = []
            boost = []
            for cell, code in live.items():
                if code == CANDY:
                    if any(n in blast_s for n in self.nbr[cell]):
                        drop.append(cell)
                elif code == HARD:
                    row = mvd[cell]
                    if ((row[0] in blast_s and row[1] in blast_s) or
                            (row[2] in blast_s and row[3] in blast_s)):
                        drop.append(cell)
                elif code in _BOOST and any(n in blast_s
                                            for n in self.nbr[cell]):
                    boost.append(cell)
            for cell in drop:
                del live[cell]
            for cell in boost:
                live[cell] = _BOOST[live[cell]]

        return (-1 if dead else p, tuple(sorted(live.items())), tuple(blast))

    @staticmethod
    def won(state: tuple) -> bool:
        """``no WinCandy`` and ``no PlayerDead``."""
        p, pieces, _expl = state
        return p >= 0 and not any(code <= HARD for _cell, code in pieces)

    @staticmethod
    def doomed(state: tuple) -> bool:
        """True when this state can never win again, and both reasons are
        permanent:

          * the player is dead -- nothing ever removes PlayerDead, so
            ``no PlayerDead`` can never hold again; or
          * candy is still standing with no bomb left anywhere. Bombs are
            consumed by detonating and the game creates none, so nothing can ever
            destroy that candy. (A blast already on the board has spent its late
            rules by the time the state settles, so it does not count.)

        This is the whole reason the exact field is affordable: wasting the last
        bomb is most of what random play does, and this prunes that branch the
        move it happens instead of exploring a board that can no longer win."""
        p, pieces, _expl = state
        if p < 0:
            return True
        if not any(code <= HARD for _cell, code in pieces):
            return False                                   # already won
        return not any(code >= B0 for _cell, code in pieces)


def model_from_engine(eng, game) -> tuple:
    """``(Model, state)`` for the interpreter's current grid."""
    idx2name = {i: n for n, i in game.obj_name_to_idx.items()}
    grid = eng.grid
    h, w = len(grid), len(grid[0])
    walls = set()
    pieces = {}
    blast = []
    player = -1
    dead = False
    for r in range(h):
        row = grid[r]
        for c in range(w):
            cell = r * w + c
            for o in row[c]:
                name = idx2name[o]
                if name == "wall":
                    walls.add(cell)
                elif name == "player":
                    player = cell
                elif name == "playerdead":
                    dead = True
                elif name == "explosion":
                    blast.append(cell)
                elif name in _NAME2CODE:
                    pieces[cell] = _NAME2CODE[name]
    return (Model(h, w, walls),
            (-1 if dead else player, tuple(sorted(pieces.items())),
             tuple(sorted(blast))))


# ---------------------------------------------------------------------------
# Goal heuristic (only the A* levels need it)
# ---------------------------------------------------------------------------

class Heuristic:
    """Estimated presses to the win, over model states.

    Per surviving candy, the cheapest bomb that could still destroy it:

      * a LIT bomb costs its remaining fuse -- it detonates in exactly that many
        presses no matter what -- and can be shoved at most that many cells, so a
        candy further than the fuse allows is not this bomb's job;
      * an UNLIT bomb costs the walk to a cell beside it, plus the shove, plus
        its fuse.

    ``max`` over the candies is a lower bound (their clocks can overlap), and
    that is what ``K = 0`` uses. It is also FLAT: on a board with four candies
    left it says the same thing as on a board with one, so A* has nothing to
    steer with and degenerates to breadth-first at a depth where that is
    hopeless. ``K`` adds that many presses of credit per candy still standing --
    no longer admissible, but it is the difference between levels 9 and 13
    closing in seconds and not closing at all.
    """

    _INF = 1 << 20

    def __init__(self, model: Model, k: int = 0):
        self.m = model
        self.k = k
        self.wd = [self._wall_bfs(model, c) if c not in model.walls else None
                   for c in range(model.h * model.w)]

    @classmethod
    def _wall_bfs(cls, m: Model, src: int) -> list:
        """Shortest walk lengths from ``src`` over everything that is not a
        wall. Pushables are deliberately ignored: they move, and a distance that
        assumed they did not would stop being a lower bound."""
        dist = [cls._INF] * (m.h * m.w)
        dist[src] = 0
        queue = deque([src])
        while queue:
            cur = queue.popleft()
            for n in m.nbr[cur]:
                if n not in m.walls and dist[n] == cls._INF:
                    dist[n] = dist[cur] + 1
                    queue.append(n)
        return dist

    def __call__(self, state: tuple) -> int:
        p, pieces, _expl = state
        m, wd = self.m, self.wd
        targets = [(c, k) for c, k in pieces if k <= HARD]
        if not targets:
            return 0
        if p < 0:
            return _BIG
        walk = wd[p]
        # cell -> (presses until this bomb blows, how far it can still be shoved)
        bombs = {}
        for cell, code in pieces:
            if code in _TTL:
                bombs[cell] = (_TTL[code], _TTL[code])
            elif code in _FUSE:
                fuse = _FUSE[code]
                near = min((walk[n] for n in m.nbr[cell] if n not in m.walls),
                           default=self._INF)
                if near < self._INF:
                    bombs[cell] = (near + 1 + fuse, fuse + 1)
        if not bombs:
            return _BIG
        worst = 0
        for cell, kind in targets:
            sides = [n for n in m.nbr[cell] if n not in m.walls]
            if kind == CANDY:
                best = min((cost for b, (cost, reach) in bombs.items()
                            for a in sides if wd[b] and wd[b][a] <= reach),
                           default=_BIG)
            else:
                # A HardCandy needs two blasts on opposite sides of one axis, in
                # the same turn -- so two DIFFERENT bombs, and the estimate is
                # the later of the two.
                best = _BIG
                row = m.mv[cell]
                for i in (0, 2):
                    a1, a2 = row[i], row[i + 1]
                    if a1 < 0 or a2 < 0 or a1 in m.walls or a2 in m.walls:
                        continue
                    for b1, (c1, r1) in bombs.items():
                        if not wd[b1] or wd[b1][a1] > r1:
                            continue
                        for b2, (c2, r2) in bombs.items():
                            if b2 != b1 and wd[b2] and wd[b2][a2] <= r2:
                                best = min(best, max(c1, c2))
            worst = max(worst, best)
        return worst + self.k * (len(targets) - 1)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

class Plan(list):
    """The press sequence, carrying the optimal SET for each of its steps.

    ``optsets[i]`` is every press that is as good as the one taken at step
    ``i`` -- the complete equally-shortest set on the levels the exact field
    closed, and the plan's own press widened by the reordering probe on the
    three A* levels. Computed while the field / model is in memory, so a plan
    replayed from the disk cache in a later process still labels every step
    without rebuilding anything."""

    optsets: list

    def __init__(self, presses, optsets):
        super().__init__(presses)
        self.optsets = optsets


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class CandyBombExpert(PSExpert):
    """Plans over `Model`, not over the interpreter.

    `PSExpert.plan` is replaced outright, so `PSExpert._astar`, `_key`, `dead`
    and `heuristic` never run and are deliberately not implemented here -- the
    engine-blackbox search is 370x slower per node than the model and would not
    reach half these levels. What is inherited is the memo/restore discipline of
    the surrounding harness and the recorder that follows the plan.

    Two planners, tried in order: the exact field (`_field`), then the A* ladder
    (`_ladder`). See the module docstring for which levels need which.
    """

    #: The four arrows. ACTION is left out because it is exactly a permanently
    #: blocked move (`Model.step`) -- it either cancels the turn or lets a blast
    #: kill you, so no plan can ever want it. `selfcheck` holds the model to the
    #: interpreter on it anyway.
    directions = list(DIRS)

    #: Rungs for the levels the field cannot close, tried in order: ``(K, weight,
    #: node cap)``. K = 0 first because it is the only rung whose plan is a
    #: candidate optimum (levels 9 and 12 give 26 and 19 either way, but 12 is
    #: cheaper here), and the credited rungs after it because levels 9 and 13
    #: simply do not close without them.
    ladder = ((0, 1, 1_000_000), (2, 1, 1_000_000),
              (4, 1, 2_000_000), (6, 1, 2_000_000))

    def setup(self) -> None:
        self._memo: dict = {}                 # (level, state) -> Plan | None
        self._sizes: dict = {}                # level -> field states / A* rung
        self._disk = self._load_disk()

    # -- the exact field -------------------------------------------------------
    def _field(self, model: Model, start: tuple, level) -> Plan | None:
        """Forward BFS to the first winning depth, then one backward BFS from
        the winning transitions; the plan is the greedy descent.

        Returns None -- meaning "use the ladder" -- when the space passes
        ``node_cap`` states before a win shows up.

        WHY THE TRUNCATION IS EXACT. BFS stops after the layer that produced the
        first win, so every state at depth < L has all four of its edges
        recorded, and the backward BFS over that subgraph gives dist >= the true
        distance-to-win everywhere. For a state ON an optimal route the two
        agree: its optimal continuation never leaves depth <= L, so it lies
        entirely inside the subgraph. That is exactly the set of states the
        descent walks and the set `_descend` reads tie sets from, so both the
        plan and its optimal sets are the real thing -- and the millions of
        states beyond depth L, which only matter for routes that are already too
        long, are never enumerated."""
        if Model.won(start):
            return Plan([], [])
        cap = self.node_cap
        succ: dict = {}
        frontier = [start]
        seen = {start}
        won = False
        while frontier and not won:
            nxt = []
            for st in frontier:
                edges = [None, None, None, None]
                succ[st] = edges
                for d in range(4):
                    ns = model.step(st, d)
                    if ns is None:
                        continue
                    if Model.won(ns):
                        edges[d] = _WIN
                        won = True
                        continue
                    if Model.doomed(ns):
                        continue
                    edges[d] = ns
                    if ns not in seen:
                        seen.add(ns)
                        nxt.append(ns)
                        if len(seen) > cap:
                            return None
            frontier = nxt
        if not won:
            return None                       # the reachable space closed unwon

        # Backward BFS from the win. Every press costs 1, so plain BFS over the
        # reversed edges is the exact distance-to-win.
        preds: dict = {}
        for st, edges in succ.items():
            for ns in edges:
                if ns is not None and ns is not _WIN:
                    preds.setdefault(ns, []).append(st)
        dist = {}
        layer = [st for st, edges in succ.items() if _WIN in edges]
        for st in layer:
            dist[st] = 1
        while layer:
            nxt = []
            for st in layer:
                for pst in preds.get(st, ()):
                    if pst not in dist:
                        dist[pst] = dist[st] + 1
                        nxt.append(pst)
            layer = nxt
        self._sizes[level] = f"{len(seen)} states"
        return self._descend(succ, dist, start)

    @staticmethod
    def _descend(succ: dict, dist: dict, start: tuple) -> Plan | None:
        """Walk the field downhill to the win, collecting the COMPLETE set of
        equally-shortest presses at every step. Pure arithmetic over the field --
        it does not step the model."""
        if start not in dist:
            return None
        presses, optsets = [], []
        st = start
        while True:
            here = dist[st]
            best = []
            for d in range(4):
                ns = succ[st][d]
                if ns is None:
                    continue
                nd = 0 if ns is _WIN else dist.get(ns)
                if nd is not None and nd == here - 1:
                    best.append(DIRS[d])
            if not best:                      # unreachable: dist is consistent
                return None
            optsets.append(best)
            presses.append(best[0])
            ns = succ[st][DIRS.index(best[0])]
            if ns is _WIN:
                return Plan(presses, optsets)
            st = ns

    # -- the A* ladder ---------------------------------------------------------
    def _ladder(self, model: Model, start: tuple, level) -> Plan | None:
        """First rung of `ladder` that reaches a win, with tie sets from the
        reordering probe (`_reorder_optsets`)."""
        for k, weight, cap in self.ladder:
            path = self._astar(model, start, Heuristic(model, k), weight, cap)
            if path is not None:
                self._sizes[level] = f"A* K={k} w={weight}"
                presses = [DIRS[d] for d in path]
                return Plan(presses, self._reorder_optsets(model, start, path))
        return None

    @staticmethod
    def _astar(model: Model, start: tuple, h: Heuristic, weight: int,
               cap: int) -> tuple | None:
        """Weighted A* over the model. Parent pointers rather than a path per
        heap entry: at the couple of million nodes level 13 needs, a copied path
        tuple per node is most of the memory."""
        if Model.won(start):
            return ()
        best = {start: 0}
        parent = {start: None}
        pq = [(weight * h(start), 0, 0, start)]
        counter = 0
        nodes = 0
        while pq:
            _f, g, _c, st = heapq.heappop(pq)
            if best.get(st, -1) != g:
                continue                      # a stale heap entry
            for d in range(4):
                ns = model.step(st, d)
                nodes += 1
                if ns is None:
                    continue
                if Model.won(ns):
                    out = [d]
                    cur = st
                    while parent[cur] is not None:
                        cur, pd = parent[cur]
                        out.append(pd)
                    out.reverse()
                    return tuple(out)
                if Model.doomed(ns):
                    continue
                ng = g + 1
                if best.get(ns, 1 << 30) <= ng:
                    continue
                best[ns] = ng
                parent[ns] = (st, d)
                counter += 1
                heapq.heappush(pq, (ng + weight * h(ns), ng, counter, ns))
            if nodes >= cap:
                return None
        return None

    @staticmethod
    def _reorder_optsets(model: Model, start: tuple, path: tuple) -> list:
        """Per-step optimal sets for a plan the ladder found: the press taken,
        plus every LATER press of the plan that could have been taken first.

        A press ``d`` from further down the plan is as good as the one being
        taken when pulling it to the front and replaying the rest still wins by
        the recorded plan's last step or sooner -- the two presses commute, which
        on these boards means either the two axes of one walk, or two shoves that
        do not interact. That is the widening available without the exact field:
        it never claims a press is optimal in an absolute sense (these three
        plans are the shortest FOUND, not a proven optimum), only that it is at
        least as good as what the expert did, which is what the demonstration
        teaches either way.

        The whole suffix is replayed rather than a two-press commutation test,
        because a press that commutes with the next one but wrecks the timing
        three turns later is not equally good -- and on the model the full replay
        costs microseconds. A press the model CANCELS is replayed as a wasted
        turn, exactly as the interpreter would treat it, so a permutation that
        needs one can never come out shorter than the plan.
        """
        states = [start]
        for d in path:
            states.append(model.step(states[-1], d))
        optsets = []
        for i, taken in enumerate(path):
            suffix = path[i:]
            best = {taken}
            for j in range(1, len(suffix)):
                if suffix[j] in best:
                    continue
                st = states[i]
                for d in (suffix[j],) + suffix[:j] + suffix[j + 1:]:
                    nxt = model.step(st, d)
                    if nxt is not None:
                        st = nxt
                    if Model.won(st):
                        best.add(suffix[j])
                        break
            optsets.append([DIRS[d] for d in sorted(best)])
        return optsets

    # -- planning --------------------------------------------------------------
    def plan(self, eng, level: int | None = None) -> Plan | list | None:
        """The winning press sequence from the interpreter's current state (a
        `Plan`, carrying its optimal sets), or None if there is none. Leaves the
        interpreter untouched -- the search runs entirely on the model.

        The level's START state is also kept on disk: it is the only state any
        seed ever plans from (recovery is a RESET back to it), and re-deriving it
        costs the better part of a minute in every process."""
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
        found = self._field(model, state, level)
        if found is None:
            found = self._ladder(model, state, level)
        self._memo[memo] = found
        if level is not None and cached is None:
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
        check: a cached plan is only served to the state it was solved from."""
        p, pieces, expl = state
        return [p, [[c, k] for c, k in pieces], list(expl)]

    def describe(self, level: int | None) -> str:
        """Which planner answered for this level (for `--plans`); "cached" when
        the plan came off disk and no search ran."""
        return self._sizes.get(level, "cached")

    # -- disk cache ------------------------------------------------------------
    @staticmethod
    def _load_disk() -> dict:
        """``{level: {"start": sig, "plan": [...] | None, "optsets": [...]}}``,
        or empty if unreadable -- an unparseable cache is a miss, never a
        crash."""
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


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class CandyBombSolver(PSAStarSolver):
    game_id = "puzzlescript_candy_bomb"
    game_name = GAME_NAME
    expert_cls = CandyBombExpert

    #: Cap on FIELD states per level. Above it the level falls through to the A*
    #: ladder, so this is the memory dial between "exact plan with complete tie
    #: sets" and "shortest found". The eleven exact levels peak at 249k.
    node_cap = 400_000
    #: Plans are 3-30 presses; the ceiling only has to cover a re-plan.
    max_steps = 200

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup rather
        than as a mysteriously slow first seed."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The optimal press set at this step, off the plan's own `Plan.optsets`.

        Exact and complete on the eleven levels the field closed; on levels 9, 12
        and 13 it is the plan's press widened by the reordering probe (see
        `CandyBombExpert._reorder_optsets`). Falls back to the press about to
        be taken so that no expert step ever ships unlabelled -- `train_policy`
        v2 supervises ``optimal`` only, so a step without one contributes nothing
        to the loss."""
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
    FIVE keys, the settled state, the cancel/no-cancel decision and the win flag
    all have to agree. That is what licenses planning off the engine.

    Rollouts are steered to touch the corners the model gets wrong if it is
    wrong: they include ACTION (whose interesting case is the fatal turn it
    cannot cancel), they keep going after the player dies (the interpreter
    carries on ticking, and the death-before-cancel rule order only shows up
    there) and they are long enough for a lit bomb to reach a second one.
    Returns the number of violations."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        deaths = wins = cancels = 0
        for t in range(trials):
            game.set_level(level)
            model, state = model_from_engine(eng, game._game)
            rng = random.Random(f"candy_bomb:selfcheck:{level}:{t}")
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
                    print(f"  L{level}: state divergence on {KEYS[di]}\n"
                          f"    engine {actual}\n    model  {expected}")
                    break
                # "The grid did not change" and "the turn cancelled" are the same
                # thing only while the player is alive; a dead player's turns are
                # real turns that happen to change nothing.
                if state[0] >= 0 and (predicted is None) != frozen:
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
                if state[0] < 0:
                    deaths += 1
        if verbose:
            print(f"  L{level}: {'OK' if not bad else 'VIOLATIONS'} "
                  f"({trials} rollouts, {cancels} cancelled turns, "
                  f"{deaths} post-death steps, {wins} accidental wins)")
    return bad


def _plan_report() -> None:
    """Print every level's plan, how it was found and how many of its steps have
    more than one right answer -- the quick "is this game still fully solved"
    check. Every plan is replayed through the real interpreter, so this is also
    the model's end-to-end test."""
    solver = CandyBombSolver()
    game, expert, solvable = solver._ensure(0)
    total = ties = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            continue
        for direction in plan:                      # interpreter-verify it
            eng.step(direction)
        step_ties = sum(len(s) - 1 for s in plan.optsets)
        total += len(plan)
        ties += step_ties
        print(f"  L{level}: {len(plan):3d} presses  win={eng.check_win()}  "
              f"{expert.describe(level):>14}  {step_ties:3d} tie-presses  "
              f"{' '.join(d[0] for d in plan)}")
    print(f"  solvable levels: {solvable}")
    print(f"  {total} presses, {ties} of them with a second right answer")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(CandyBombSolver.main())
