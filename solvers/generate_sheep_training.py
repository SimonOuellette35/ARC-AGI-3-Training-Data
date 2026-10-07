"""Generate Phase-1 training data for the PuzzleScript game ps:sheep
("Sheep" by Rabbit From Hell, `data/puzzlescript_games/Sheep.txt`).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, a native model of the turn (fuzz-verified against the real
interpreter), the walk-and-scare macro search over it, and the optimal-action
labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_sheep",
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
You are a shepherd who cannot touch the sheep: they RUN from the sight of you.
Win when every Sheep has been driven onto a Target (`No Sheep`; a sheep that
lands on a target becomes an immovable PennedSheep). The player's only other
power is the crates, which push like a sokoban box and are the game's real
tool -- they block sight and they stop a bolting sheep.

The turn, as the rules actually resolve it (this is the whole mechanic):

  * A Probe is spawned in all four directions and RAY-CASTS away from the
    player: it dies on a Crate / Wall / PennedSheep, and the FIRST Sheep it
    reaches is set running one cell directly away. So each of the four rays
    scares at most one sheep, and the four rays are disjoint -- no two scared
    sheep can ever contest a cell, which is what makes the mechanic
    order-independent (and safe to mirror; see `_FLIP_GAMES` below).
  * `again` re-runs the whole turn, so a scared sheep keeps running one cell
    per tick until a Solid stops it -- or until it crosses a Target, where it
    pens mid-run. **A run therefore pens the sheep iff a target lies on the ray
    before the first blocker.**
  * THE GATE THAT IS EASY TO GET WRONG. `again` is queued by the rules that
    extend the probe (`[> Probe | no Solid]`) and set a sheep running -- and by
    nothing else. Tick 1 evaluates its rules at the player's OLD cell (rules run
    before movement), ticks 2+ at the new one. So when the player is boxed in
    tightly enough that no probe has room to extend from the cell it is standing
    on, the turn ENDS after tick 1 and the sheep never see where the player
    moved to. A model without that gate is right about 99.9% of transitions and
    wrong about the ones that matter (it was 8 mismatches in 9611 fuzz steps,
    all of them a sheep bolting when the real game leaves it alone).
  * The player can walk INTO the cell of an adjacent sheep -- it vacates as we
    step in -- but only when it has somewhere to go; a sheep with its back to a
    wall blocks the player like any other Solid.
  * ACTION (X) is bound to nothing, so it is a pure "wait" -- but a wait still
    casts the probes, which makes it a real move on a level's opening state
    (nothing has run yet there) and a no-op afterwards. The search branches on
    it and dedups by resulting state, so it prunes itself where it is idle.

The model, and why there is one
-------------------------------
`PSExpert`'s engine-blackbox A* would be the default (see
[ps_astar](common/ps_astar.py)), but an interpreter step costs ~2.7ms here and
these boards need hundreds of thousands of them. `_Board` re-implements the turn
above in ~60 lines and runs ~100x faster. It is not trusted on the strength of
having been read carefully: `--selfcheck` fuzzes it against the real interpreter
over random boards (3000 boards / 66k presses at the time of writing, 0
mismatches), and every plan it produces is then replayed on the interpreter
itself, which is what `record_level` keeps or drops the level on.

The search
----------
Primitive A* drowns: most presses are the shepherd walking, and walking is only
interesting when it changes what a sheep can see. The successors are therefore
MACROS -- *walk quietly to a cell, then take the one press that stirs the board*
-- exactly the shape `PSPushExpert` uses for the sokoban-shaped games, but with
"quietly" doing the work that "without pushing" does there. A step is quiet when
the whole turn leaves everything except the player's own cell alone, which is
decidable straight from the ray/`again` analysis (`_Board.probe`), so the walk
BFS never has to step the model at all.

The heuristic counts RUNS, not distance: per unpenned sheep, a Dijkstra over
straight-line runs to the first run that crosses a target, where a run that
stops naturally (wall / crate / penned / another sheep) costs 1 and one that
needs a blocker pushed into place first costs `place`. Two details carry it:

  * **A run is only counted if the cell BEHIND the sheep exists.** Nothing can
    scare a sheep through a wall, so a run whose launch side is solid rock is
    not a route however good it looks. Adding that one test is what took levels
    4-7 from "no plan in 60k nodes" to a few thousand nodes.
  * **Penned targets block instead of pen.** A PennedSheep occupies its target,
    so the second sheep to arrive stops in front of it.

`_Board.dead` is the sound counterpart used to prune: ignoring every crate and
every other sheep (they move; walls and penned sheep do not), can this sheep
still reach any penning run at all? It fires on ~3 states for every one
expanded on the crowded levels.

The search ladder (`SheepExpert.ladder`) runs weighted A* at increasing weight,
so the easy boards keep their short plans and the crowded ones still come out;
`weight > 1` means a plan is a genuine win but not provably shortest, which is
why the labels are walk sets rather than measured optimal sets (below).

**11 of the 12 levels solve** -- 608 presses in total, 14 on level 1 up to 108
on level 10, every one of them inside the adapter's 200-action budget. The
finale (level index 11) does not; see `SheepSolver.skip_levels` for what was
tried. The searches are seed-independent and cached to `PLAN_CACHE`, so they
are paid once (~4 minutes for the ten that need searching, plus ~25s of level
9) rather than per seed or per shard.

Optimal-action labels
---------------------
Every recorded expert press carries a target ([[always-emit-optimal-targets]]).
A macro is a walk followed by one decisive press, so:

  * a WALK step is labelled with every direction that starts an equally short
    quiet walk to the cell the macro acts from (the axis order of a walk is
    genuinely free, and labelling one arbitrary interleaving as the single right
    answer is what that memory calls a bug);
  * the decisive press is labelled with itself. Sibling macros are not searched
    for ties -- as in a sokoban, which sheep you scare where IS the puzzle, so a
    different scare is a different plan rather than a reordering of this one.

Rendering
---------
Two compositions were unreadable before this generator existed, both of the
recurring "the body hides the ground tile" kind: Crate and Player shipped as
fully opaque 5x5 sprites, so a crate on a target rendered pixel-identical to a
crate on grass, and so did the player. A crate on a target is not cosmetic here
-- it makes the target unusable and can strand the level -- and the player walks
over targets constantly. Both sprites now have transparent CORNERS, which is
exactly where Target paints its red pixels, and corners are sampled at every
cell size these boards use (only cell_px 4 drops a sprite's middle row/column).
`--audit` is the regression test: it renders every composition on a full board
and asserts the whole frames are pairwise distinct.

Augmentation
------------
Rotation is the adapter's, per (seed, level), as it is for every ps: game.
"Sheep" is also added to `PuzzleScriptAdapter._FLIP_GAMES`, which adds an
independent horizontal and vertical mirror (and the matching action remap), so
the corpus covers 8 presentations of each board instead of 4 -- 88 rather than
44 across the eleven recorded levels. `--symmetry` is what earns that: each
level's plan plus a 300-press random walk, replayed on all seven other
presentations, whole board compared after every press. Clean on all 11.

Validated
---------
  * `--selfcheck`: 3000 random boards / 66k presses, model == interpreter on
    every settled board and every win flag.
  * `--plans`: 11/12 levels, 608 presses, all inside the 200-action budget.
  * `--verify`: every plan wins ON the interpreter; every label set executed on
    it (116 presses) lands where the model says, stirs nothing, and stays on a
    shortest quiet walk; six seeds recorded and replayed through a FRESH
    adapter -- ~3800 frames byte-identical and `GameState.WIN` every time.
    (The frame count moves between runs because `BaseSolver`'s rng is unseeded
    unless `main` seeds it, so each process explores a different prefix -- which
    is the point of running the check repeatedly.)
  * `--symmetry`: clean, see above. `--audit`: clean.
  * Two full runs of the generator `diff -rq` identical.
  * Every recorded expert step carries an optimal target (608/608 per episode);
    the only unlabelled steps are the opening exploration prefix and each
    level's action-0 RESET, which is the standard.

Usage
-----
    python solvers/generate_sheep_training.py --episodes 1600 \
        --out data/training_multi_level/puzzlescript_sheep

    python solvers/generate_sheep_training.py --plans       # per-level report
    python solvers/generate_sheep_training.py --selfcheck   # model vs engine
    python solvers/generate_sheep_training.py --verify      # replay on the engine
    python solvers/generate_sheep_training.py --symmetry    # flip/rotation proof
    python solvers/generate_sheep_training.py --audit       # rendering audit
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
from arcengine import ActionInput, GameState                          # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION                     # noqa: E402
from solvers.common.ps_astar import (DIRECTIONS, Plan, PSAStarSolver,  # noqa: E402
                                     PSExpert, restore, snapshot)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Sheep"

#: Disk cache of each level's start plan AND its label sets. The searches are
#: seed-independent (only the PRESENTATION is augmented per seed) but cost a few
#: minutes on the crowded boards, which every shard of `parallelize_generator`
#: would otherwise repeat on every core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "sheep_plans.json"

#: The four moves, and the fifth press (ACTION/X) the game binds to nothing but
#: which still casts the probes -- so it is a real move exactly once per level.
_DIRS = ("up", "down", "left", "right")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
_OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}

_INF = 1 << 30


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the turn function over it.

    A STATE is ``(player, crates, sheep, penned)`` with the three object sets as
    sorted tuples of flat cell ids -- tuples rather than frozensets because the
    search holds hundreds of thousands of them and a frozenset of two ints costs
    four times what the tuple does.

    Sheep and penned sheep are tracked separately because they are different
    things to the rules: a Sheep is what the win condition counts and what the
    probes chase, a PennedSheep is scenery that blocks sight and travel (and
    occupies its target, so no second sheep can pen there).
    """

    def __init__(self, h: int, w: int, walls, targets):
        self.h, self.w = h, w
        self.walls = frozenset(walls)
        self.targets = frozenset(targets)
        # Per cell and direction: the next cell (None into a wall / off-board),
        # and the whole clear-of-walls line, which is what a probe traverses.
        self.step_cell: list[dict] = [dict() for _ in range(h * w)]
        self.ray: list[dict] = [dict() for _ in range(h * w)]
        for r in range(h):
            for c in range(w):
                p = r * w + c
                if p in self.walls:
                    continue
                for d, (dr, dc) in _DELTA.items():
                    rr, cc = r + dr, c + dc
                    q = rr * w + cc
                    ok = 0 <= rr < h and 0 <= cc < w and q not in self.walls
                    self.step_cell[p][d] = q if ok else None
                    line = []
                    while ok:
                        line.append(q)
                        rr, cc = rr + dr, cc + dc
                        q = rr * w + cc
                        ok = 0 <= rr < h and 0 <= cc < w and q not in self.walls
                    self.ray[p][d] = tuple(line)

    # -- the turn ---------------------------------------------------------
    def _cast(self, player, crates, sheep, penned):
        """``(marks, again)`` for the probes fired from ``player``.

        ``marks`` is the ``(sheep cell, direction)`` each ray sets running;
        ``again`` is whether any rule carrying the `again` keyword fired, i.e.
        whether the interpreter will run the turn once more. Only the probe
        EXTENSION and the two sheep-scaring rules carry it -- a probe that dies
        the moment it is spawned does not, which is the gate the module
        docstring describes."""
        marks = []
        again = False
        solid = sheep | crates | penned
        for d in _DIRS:
            line = self.ray[player][d]
            if not line:
                continue                       # the probe spawns inside a wall
            q = line[0]
            if q in sheep:
                marks.append((q, d))           # [Player | Sheep] -> ... again
                again = True
                continue
            if q in solid:
                continue                       # probe dies on the obstacle
            i = 1
            while i < len(line) and line[i] not in solid:
                again = True                   # [> Probe | no Solid] ... again
                i += 1
            if i < len(line) and line[i] in sheep:
                marks.append((line[i], d))     # [> Probe | Sheep] -> ... again
                again = True
        return marks, again

    def _run(self, marks, player, crates, sheep, penned):
        """Move every scared sheep one cell, penning the ones that cross a
        target. Returns whether anything moved. Blockers are read from the
        pre-move board, as the cancelling rule `[> Sheep | Solid]` does."""
        solid = sheep | crates | penned | {player}
        moved = False
        for s, d in marks:
            t = self.step_cell[s][d]
            if t is None or t in solid:
                continue
            sheep.discard(s)
            moved = True
            if t in self.targets:
                penned.add(t)
            else:
                sheep.add(t)
        return moved

    def step(self, state, action):
        """The whole turn: one press in, the settled state out."""
        player, crates, sheep, penned = state
        crates, sheep, penned = set(crates), set(sheep), set(penned)
        marks, again = self._cast(player, crates, sheep, penned)
        if action != "action":
            t = self.step_cell[player][action]
            if t is not None:
                blocked = crates | sheep | penned
                if t in crates:
                    u = self.step_cell[t][action]
                    if u is not None and u not in blocked:
                        crates.discard(t)
                        crates.add(u)
                        player = t
                elif t in sheep:
                    u = self.step_cell[t][action]
                    if u is not None and u not in blocked:
                        player = t             # it vacates as we step in
                elif t not in penned:
                    player = t
        self._run(marks, player, crates, sheep, penned)
        while again:                           # ticks 2+ see the NEW cell
            marks, again = self._cast(player, crates, sheep, penned)
            if not self._run(marks, player, crates, sheep, penned):
                break                          # nothing changed: the loop ends
        return (player, tuple(sorted(crates)), tuple(sorted(sheep)),
                tuple(sorted(penned)))

    @staticmethod
    def won(state) -> bool:
        """`No Sheep` -- every sheep has been penned."""
        return not state[2]

    # -- quiet walking ----------------------------------------------------
    def probe(self, x, crates, sheep, penned):
        """``(quiet, again)`` for a player standing at ``x``: quiet means no
        sheep this cell can see is able to move, so a press taken here disturbs
        nothing."""
        marks, again = self._cast(x, crates, sheep, penned)
        solid = sheep | crates | penned
        for s, d in marks:
            t = self.step_cell[s][d]
            if t is not None and t not in solid and t != x:
                return False, again
        return True, again

    def walk(self, state):
        """``(dist, graph, stirs)`` for the quiet-walk region around the player.

        ``dist[cell] = (presses, path)`` from the player's cell over steps that
        change nothing else; ``graph[cell][direction]`` is the quiet neighbour
        it leads to; ``stirs`` lists the ``(cell, action)`` presses on that
        region's boundary that DO change the board.

        A step ``x -> y`` is quiet when nothing can move: no sheep may be able
        to run at ``x`` (tick 1 fires from there) and, if the turn will run
        again, none at ``y`` either (ticks 2+ fire from there). Where ``x``
        cannot even extend a probe the turn stops after tick 1 and ``y`` does
        not matter at all -- which is the `again` gate, and skipping it would
        make the walk graph claim quiet steps that scatter the flock."""
        player, crates, sheep, penned = state
        crates, sheep, penned = set(crates), set(sheep), set(penned)
        occupied = crates | sheep | penned
        info: dict[int, tuple[bool, bool]] = {}

        def at(x):
            got = info.get(x)
            if got is None:
                got = info[x] = self.probe(x, crates, sheep, penned)
            return got

        dist = {player: (0, ())}
        graph: dict[int, dict] = {player: {}}
        stirs = []
        queue = deque([player])
        while queue:
            x = queue.popleft()
            d0, path = dist[x]
            quiet, again = at(x)
            if not quiet:
                stirs.extend((x, a) for a in self.actions)
                continue                       # every press here stirs the flock
            for a in _DIRS:
                y = self.step_cell[x][a]
                if y is None:
                    continue
                if y in occupied or (again and not at(y)[0]):
                    stirs.append((x, a))
                    continue
                graph[x][a] = y
                if y not in dist:
                    dist[y] = (d0 + 1, path + (a,))
                    graph.setdefault(y, {})
                    queue.append(y)
        return dist, graph, stirs

    #: The presses the search branches on. ACTION is in here for the level's
    #: opening state, where the flock has not run yet and waiting is a move.
    actions = list(DIRECTIONS)

    def successors(self, state):
        """``{next state: (presses, moves)}`` -- walk quietly, then stir."""
        dist, _graph, stirs = self.walk(state)
        out = {}
        for x, a in stirs:
            cost, path = dist[x]
            landed = self.step((x,) + state[1:], a)
            if landed == (x,) + state[1:]:
                continue                       # a blocked press: nothing happened
            old = out.get(landed)
            if old is None or old[0] > cost + 1:
                out[landed] = (cost + 1, path + (a,))
        return out

    # -- guidance ---------------------------------------------------------
    def runs_needed(self, state, place: int, cache=None) -> int:
        """Estimated RUNS left: per unpenned sheep, the cheapest chain of
        straight-line runs ending in one that crosses a target.

        A run that ends against something already on the board costs 1; one
        that has to be stopped somewhere empty costs ``place``, standing in for
        the crate somebody must push there first. Not admissible (that is what
        ``weight`` is for) -- its job is to tell the search which sheep is
        nearly herded and which is stuck behind a crate that has to move."""
        _player, crates, sheep, penned = state
        key = (crates, sheep, penned)
        if cache is not None and key in cache:
            return cache[key]
        crates, sheep, penned = set(crates), set(sheep), set(penned)
        block = crates | penned
        total = 0
        for s0 in sheep:
            others = block | (sheep - {s0})
            dist = {s0: 0}
            queue = [(0, s0)]
            best = None
            while queue:
                d0, s = heapq.heappop(queue)
                if d0 > dist.get(s, _INF):
                    continue
                if best is not None and d0 >= best:
                    break
                for d in _DIRS:
                    if self.step_cell[s][_OPP[d]] is None:
                        continue               # nothing can scare it from there
                    pens = False
                    line = []
                    for q in self.ray[s][d]:
                        if q in others:
                            break              # a penned target blocks, not pens
                        if q in self.targets:
                            pens = True
                            break
                        line.append(q)
                    if pens:
                        if best is None or d0 + 1 < best:
                            best = d0 + 1
                        continue
                    for i, cell in enumerate(line):
                        cost = d0 + (1 if i == len(line) - 1 else place)
                        if cost < dist.get(cell, _INF):
                            dist[cell] = cost
                            heapq.heappush(queue, (cost, cell))
            total += self.UNREACHABLE if best is None else best
        if cache is not None:
            cache[key] = total
        return total

    #: Charged for a sheep with no penning run in reach at all. Only a guess --
    #: `dead` is the sound version of the same question.
    UNREACHABLE = 12

    def dead(self, state, cache) -> bool:
        """True when some sheep provably can never be penned.

        Sound, and therefore safe to prune on: it asks whether the sheep could
        reach a penning run if every crate and every other sheep were out of the
        way. Only walls and penned sheep are permanent, so ignoring the movable
        rest can only over-state where a sheep can get to."""
        _player, _crates, sheep, penned = state
        for s0 in sheep:
            key = (s0, penned)
            got = cache.get(key)
            if got is None:
                got = cache[key] = self._can_pen(s0, set(penned))
            if not got:
                return True
        return False

    def _can_pen(self, s0, penned) -> bool:
        seen = {s0}
        queue = deque([s0])
        while queue:
            s = queue.popleft()
            for d in _DIRS:
                behind = self.step_cell[s][_OPP[d]]
                if behind is None or behind in penned:
                    continue
                for q in self.ray[s][d]:
                    if q in penned:
                        break
                    if q in self.targets:
                        return True
                    if q not in seen:
                        seen.add(q)
                        queue.append(q)
        return False

    # -- search -----------------------------------------------------------
    def solve(self, start, weight: int, place: int, cap: int):
        """Weighted A* over the macros. Returns ``(presses, label sets, nodes)``
        or ``(None, None, nodes)`` if the cap is reached first."""
        hcache: dict = {}
        dcache: dict = {}
        h = lambda s: self.runs_needed(s, place, hcache)      # noqa: E731
        queue = [(weight * h(start), 0, 0, start)]
        best = {start: 0}
        parent: dict = {start: None}
        nodes = tie = 0
        while queue:
            _f, g, _c, s = heapq.heappop(queue)
            if g > best.get(s, _INF):
                continue
            if self.won(s):
                presses, sets = self.annotate(start, self._macros(parent, s))
                return presses, sets, nodes
            nodes += 1
            if nodes > cap:
                return None, None, nodes
            for n, (cost, moves) in self.successors(s).items():
                ng = g + cost
                if ng >= best.get(n, _INF) or self.dead(n, dcache):
                    continue
                best[n] = ng
                parent[n] = (s, moves)
                tie += 1
                heapq.heappush(queue, (ng + weight * h(n), ng, tie, n))
        return None, None, nodes

    @staticmethod
    def _macros(parent, goal):
        """Walk the parent chain back to the start: ``[(moves, state), ...]``."""
        out = []
        cur = goal
        while parent[cur] is not None:
            prev, moves = parent[cur]
            out.append((moves, cur))
            cur = prev
        out.reverse()
        return out

    def annotate(self, start, macros):
        """``(presses, label sets)`` for a macro plan.

        Each macro is a quiet walk plus the press that stirs the board. The walk
        steps are labelled with EVERY direction that starts an equally short
        quiet walk to the cell the macro acts from -- which is the honest label,
        because the two axes of a walk can be interleaved any way at all -- and
        the stirring press with itself. See the module docstring for why sibling
        macros are not searched for ties."""
        presses: list[str] = []
        sets: list[list[str]] = []
        state = start
        for moves, nxt in macros:
            _dist, graph, _stirs = self.walk(state)
            cur = state[0]
            goal = cur
            for a in moves[:-1]:
                goal = graph[goal][a]
            back = self._reverse(graph, goal)
            for a in moves[:-1]:
                here = back.get(cur, _INF)
                alts = sorted(d for d, y in graph[cur].items()
                              if back.get(y, _INF) == here - 1)
                presses.append(a)
                sets.append(alts if a in alts else [a])
                cur = graph[cur][a]
            presses.append(moves[-1])
            sets.append([moves[-1]])
            state = nxt
        return presses, sets

    @staticmethod
    def _reverse(graph, goal):
        """Quiet-walk distances TO ``goal``, over the same graph the walk BFS
        built (its steps are not always reversible -- see `walk`)."""
        back = {c: {} for c in graph}
        for x, edges in graph.items():
            for y in edges.values():
                back.setdefault(y, {})[x] = True
        dist = {goal: 0}
        queue = deque([goal])
        while queue:
            y = queue.popleft()
            for x in back.get(y, ()):
                if x not in dist:
                    dist[x] = dist[y] + 1
                    queue.append(x)
        return dist


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class SheepExpert(PSExpert):
    """Planner over `_Board`. `PSExpert` supplies the plan memo, the disk cache
    and the restore discipline; `_search` swaps in the macro A*, so the
    interpreter is only ever stepped by the recorder -- which is also what
    verifies every plan, since a level is kept only when the engine says WIN."""

    directions = list(DIRECTIONS)
    plan_cache_path = PLAN_CACHE

    #: ``(weight, place, node cap)``, tried in order until one wins. Low weight
    #: first so the easy boards keep their short plans; the two crowded ones
    #: (levels 8 and 11, four crates and up to four sheep) only come out at the
    #: greedier settings, and paying for the cheap attempt first costs seconds.
    ladder = ((3, 3, 60_000), (5, 5, 250_000), (8, 5, 400_000))

    def setup(self) -> None:
        g = self.g
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.target_ids = set(g.resolve_object_name("target"))
        self.crate_ids = set(g.resolve_object_name("crate"))
        self.sheep_ids = set(g.resolve_object_name("sheep"))
        self.penned_ids = set(g.resolve_object_name("pennedsheep"))
        self.player_ids = set(self.game._engine._player_indices)
        self._boards: dict[int | None, _Board] = {}
        self._cur: _Board | None = None
        #: ``(weight, place, nodes)`` per search actually run, for `--plans`.
        #: Empty when every plan came from the disk cache, which is the point.
        self.searches: list[tuple[int, int, int]] = []

    # -- reading the engine ----------------------------------------------
    def board(self, eng, level: int | None) -> _Board:
        """The level's static geometry, built once. Walls and targets never
        change (no rule creates or destroys either), so whichever state builds
        it describes the whole level."""
        got = self._boards.get(level)
        if got is not None:
            return got
        h, w = len(eng.grid), len(eng.grid[0])
        walls, targets = set(), set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & self.wall_ids:
                    walls.add(r * w + c)
                if cell & self.target_ids:
                    targets.add(r * w + c)
        got = self._boards[level] = _Board(h, w, walls, targets)
        return got

    def read(self, eng):
        """The engine grid as a `_Board` state."""
        w = len(eng.grid[0])
        player = None
        crates, sheep, penned = [], [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                p = r * w + c
                if cell & self.player_ids:
                    player = p
                if cell & self.crate_ids:
                    crates.append(p)
                if cell & self.penned_ids:
                    penned.append(p)
                elif cell & self.sheep_ids:
                    sheep.append(p)
        if player is None:                                   # pragma: no cover
            raise AssertionError("no Player on the board")
        return (player, tuple(crates), tuple(sheep), tuple(penned))

    # -- planning ---------------------------------------------------------
    def plan(self, eng, level: int | None = None) -> list | None:
        self._cur = self.board(eng, level)
        return super().plan(eng, level)

    def heuristic(self, eng) -> int:
        """Runs left by `_Board.runs_needed` at the ladder's first setting.
        The base `_astar` is never used here, but the contract -- 0 at a win --
        holds, and this is the number the macro search is actually guided by."""
        return self._cur.runs_needed(self.read(eng), self.ladder[0][1])

    def _search(self, eng) -> list | None:
        board = self._cur
        state = self.read(eng)
        for weight, place, cap in self.ladder:
            presses, sets, nodes = board.solve(state, weight, place, cap)
            self.searches.append((weight, place, nodes))
            if presses is not None:
                return Plan(presses, sets)
        return None


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SheepSolver(PSAStarSolver):
    game_id = "puzzlescript_sheep"
    game_name = GAME_NAME
    expert_cls = SheepExpert

    #: Record against the GAME FOLDER's adapter rather than a bare
    #: `PuzzleScriptAdapter`. Today `games/ps:sheep/ps:sheep.py` is a plain
    #: passthrough so the two are the same object, but the one a live agent is
    #: handed is the one that must be taped -- count_mover shipped a corpus
    #: recorded against a 200-step adapter for a game whose wrapper caps levels
    #: at 60, and nothing failed loudly.
    game_module_id = "ps:sheep"

    #: Level 12 (index 11) is the game's finale: four sheep packed in a ring in
    #: the middle of the board, four crates and a pen in each corner. Every rung
    #: of the ladder exhausts on it (710k macro nodes, ~10 minutes) and so do
    #: greedier settings outside it (weight 8 / 12 / 20, and a stage search that
    #: pens one sheep at a time -- that one strands the rest, which is the
    #: level's whole difficulty: the four sheep block each other, so the order
    #: they are freed in is the puzzle). Its search space is WIDE rather than
    #: dead-ended -- only ~1% of the states the search generates are prunable --
    #: so it is a heuristic problem, not a budget one, and raising the cap is
    #: not the fix. Listed here so startup skips it outright instead of
    #: re-deriving "no" whenever the plan cache is missing.
    skip_levels = frozenset({11})

    #: Unused -- `SheepExpert._search` never calls the base A*, and its own
    #: ladder carries the caps and weights -- but left at the family defaults so
    #: a future subclass that does call it is not silently starved.
    node_cap = 400_000
    weight = 1

    #: The longest plan is 108 presses (level 9); the rest is room for a
    #: re-plan after the exploration prefix's RESET. Stays
    #: under the adapter's own 200-action per-level budget, which would
    #: otherwise flip a long level to GAME_OVER mid-plan. (`set_level` resets
    #: that counter, so the exploration prefix does not eat into it.)
    max_steps = 150

    #: Recovery is the RESET prefix alone (the family default). Penning is
    #: irreversible and a crate shoved into a corner can strand a board, so a
    #: mid-plan detour is not something the expert can always re-plan out of.
    epsilon = 0.0

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time search cost visible as
        startup rather than as a mysteriously slow first seed, and it fills the
        disk cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports and checks
# ---------------------------------------------------------------------------

def _built(seed: int = 0):
    """``(solver, game, expert)`` with nothing planned yet."""
    solver = SheepSolver()
    game = solver.make_game(seed)
    expert = SheepExpert(game)
    return solver, game, expert


def _levels(game) -> list[int]:
    """The levels this generator records -- everything but `skip_levels`, whose
    search the check modes must not walk into either (it costs ~10 minutes to
    be told no)."""
    return [lvl for lvl in range(game.n_levels)
            if lvl not in SheepSolver.skip_levels]


def _report() -> int:
    """Per-level board size, piece counts, plan length and label coverage."""
    _solver, game, expert = _built()
    eng = game._engine
    total = 0
    for level in _levels(game):
        game.set_level(level)
        state = expert.read(eng)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
                  f"{len(state[2])} sheep -- NO PLAN")
            continue
        sets = getattr(plan, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
              f"{len(state[2])} sheep, {len(state[1])} crates, "
              f"{len(expert.board(eng, level).targets)} targets, "
              f"{len(plan):3d} presses (budget {game._max_steps}, {room}), "
              f"{ties:3d} steps with a tie set "
              f"({ties / max(1, len(plan)):.0%})")
    print(f"total {total} presses over {len(_levels(game))} recorded levels "
          f"({game.n_levels} shipped, skipping "
          f"{sorted(SheepSolver.skip_levels)})")
    return 0


def _random_board(rng, h, w, n_crate, n_sheep, n_target):
    """A walled room with random rubble in it, for `_selfcheck`."""
    cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    rng.shuffle(cells)
    n_wall = rng.randint(0, len(cells) // 4)
    walls = {(r, c) for r in range(h) for c in range(w)
             if r in (0, h - 1) or c in (0, w - 1)}
    walls |= set(cells[:n_wall])
    free = cells[n_wall:]
    if len(free) < n_crate + n_sheep + n_target + 1:
        return None
    rng.shuffle(free)
    it = iter(free)
    return (walls, next(it), {next(it) for _ in range(n_crate)},
            {next(it) for _ in range(n_sheep)},
            {next(it) for _ in range(n_target)})


def _selfcheck(trials: int = 400, presses: int = 25, verbose: bool = True) -> int:
    """Fuzz `_Board` against the real interpreter.

    The model is the whole reason this generator is fast, and a model that is
    subtly wrong does not fail loudly -- it plans a win the engine never
    reaches, and `record_level` quietly drops the level. So: random rooms,
    random presses, compare the settled boards and the win flag after every one
    of them.
    """
    game = SheepSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    bg, tgt, wa, wb = (idx["background"], idx["target"], idx["walla"],
                       idx["wallb"])
    pl, cr, sh = idx["player"], idx["crate"], idx["sheep"]
    expert = SheepExpert(game)
    rng = random.Random(20260817)
    acts = list(DIRECTIONS)
    steps = bad = 0
    for _ in range(trials):
        h, w = rng.randint(4, 11), rng.randint(4, 12)
        spec = _random_board(rng, h, w, rng.randint(0, 5), rng.randint(1, 6),
                             rng.randint(1, 4))
        if spec is None:
            continue
        walls, player, crates, sheep, targets = spec
        grid = []
        for r in range(h):
            row = []
            for c in range(w):
                if (r, c) in walls:
                    cell = {wa if (r + c) % 2 == 0 else wb}
                else:
                    cell = {bg}
                    if (r, c) in targets:
                        cell.add(tgt)
                    if (r, c) == player:
                        cell.add(pl)
                    if (r, c) in crates:
                        cell.add(cr)
                    if (r, c) in sheep:
                        cell.add(sh)
                row.append(cell)
            grid.append(row)
        eng.grid, eng.height, eng.width = grid, h, w
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        board = _Board(h, w, {r * w + c for r, c in walls},
                       {r * w + c for r, c in targets})
        state = expert.read(eng)
        for _ in range(presses):
            a = rng.choice(acts)
            eng.step(a)
            state = board.step(state, a)
            steps += 1
            got = expert.read(eng)
            if got != state or eng.check_win() != board.won(state):
                bad += 1
                if verbose:
                    print(f"  MISMATCH after {a} on {h}x{w}: "
                          f"engine {got} vs model {state}")
                break
            if board.won(state):
                break
    if verbose:
        print(f"selfcheck: {steps} presses over {trials} random boards, "
              f"{bad} mismatches")
    return 1 if bad else 0


def _verify(seeds: int = 6) -> int:
    """Two checks the corpus depends on, both run on the real interpreter.

    1. **Every label is executable and free.** At each expert step, press each
       direction the label calls optimal: the board it lands on must be the
       model's successor, must have moved nothing but the player, and must be
       one quiet step closer to the cell that macro acts from. That takes the
       labels out of the model and puts them through the engine.
    2. **Every recorded episode replays.** Drive a seed, then feed the recorded
       SCREEN actions to a fresh adapter at the same seed and require the same
       frames and a `GameState.WIN` -- which is what a live agent replaying the
       corpus would get, rotation/flip augmentation included.
    """
    solver, game, expert = _built()
    eng = game._engine
    bad = 0

    for level in _levels(game):
        game.set_level(level)
        board = expert.board(eng, level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: no plan (skipped level)")
            continue
        sets = plan.optsets
        state = expert.read(eng)
        checked = 0
        for i, press in enumerate(plan):
            _dist, graph, _stirs = board.walk(state)
            labels = sets[i]
            if press not in labels:
                bad += 1
                print(f"  L{level}: step {i} is not in its own label set")
            for alt in labels:
                if alt == press and len(labels) == 1:
                    continue
                before = snapshot(eng)
                eng.step(alt)
                landed = expert.read(eng)
                restore(eng, before)
                checked += 1
                if landed != board.step(state, alt):
                    bad += 1
                    print(f"  L{level}: step {i} label {alt} -- the engine and "
                          f"the model disagree")
                elif landed[1:] != state[1:]:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but it "
                          f"stirs the board")
                elif landed[0] not in graph.get(state[0], {}).values():
                    bad += 1
                    print(f"  L{level}: step {i} label {alt} leaves the quiet "
                          f"walk region")
            eng.step(press)
            state = board.step(state, press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        print(f"  L{level}: {len(plan):3d} presses, {checked} label presses "
              f"executed on the interpreter -- {'OK' if not bad else 'see above'}")

    for seed in range(seeds):
        ok, levels = solver.solve_episode(seed)
        if not ok or not levels:
            bad += 1
            print(f"  seed {seed}: solve_episode failed")
            continue
        fresh = solver.make_game(seed)
        replayed = 0
        for lvl in levels:
            fresh.set_level(lvl["level_id"])
            frames = [np.asarray(fresh._current_frame)]
            for act in lvl["actions"][1:]:
                fd = fresh.perform_action(
                    ActionInput(id=_ID_TO_GAMEACTION[act["index"]]))
                frames.append(np.asarray(fd.frame[-1] if fd.frame
                                         else fresh._current_frame))
            if fresh._state != GameState.WIN:
                bad += 1
                print(f"  seed {seed} L{lvl['level_id']}: replay ends "
                      f"{fresh._state}, not WIN")
            recorded = [np.asarray(o) for o in lvl["observations"]]
            if len(recorded) != len(frames) or any(
                    not np.array_equal(a, b) for a, b in zip(recorded, frames)):
                bad += 1
                print(f"  seed {seed} L{lvl['level_id']}: replayed frames "
                      f"differ from the recorded ones")
            replayed += len(frames)
        print(f"  seed {seed}: {len(levels)} levels, {replayed} frames replayed "
              f"-- {'OK' if not bad else 'see above'}")
    print(f"verify: {bad} problems")
    return 1 if bad else 0


def _transform(k: int, mirror: bool):
    """``(cell_fn, dims_fn, direction_map)`` for one element of the 8-group:
    mirror left-right first, then turn clockwise ``k`` times.

    The direction map is DERIVED from the same linear part that moves the cells,
    so the two cannot drift apart."""
    def cell(rc, hw):
        r, c = rc
        h, w = hw
        if mirror:
            c = w - 1 - c
        for _ in range(k):
            r, c, h, w = c, h - 1 - r, w, h
        return (r, c)

    def dims(hw):
        h, w = hw
        return (w, h) if k % 2 else (h, w)

    dmap = {"action": "action"}
    for name, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[name] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    return cell, dims, dmap


def _replay_transformed(eng, expert, layout, presses) -> list[str]:
    """Replay ``presses`` on all seven non-identity turned/mirrored copies of
    ``layout`` and return a note per presentation that diverged.

    The transformed board is built from the LEVEL LAYOUT and reloaded, so the
    interpreter parses and resolves it itself rather than being handed a grid
    this file transformed after the fact. The WHOLE board is compared after
    every press -- player, crates, sheep and penned sheep -- because what the
    mirror could break here is which of several simultaneously scared sheep the
    interpreter settles first, and that shows up in the flock, not in the
    shepherd."""
    hw = (len(layout), len(layout[0]))
    eng.load_level(layout)
    ref = [expert.read(eng)]
    for d in presses:
        eng.step(d)
        ref.append(expert.read(eng))

    notes = []
    for k, mirror in itertools.product(range(4), (False, True)):
        if (k, mirror) == (0, False):
            continue
        cell, dims, dmap = _transform(k, mirror)
        th, tw = dims(hw)
        turned = [[set() for _ in range(tw)] for _ in range(th)]
        for r, row in enumerate(layout):
            for c, objs in enumerate(row):
                tr, tc = cell((r, c), hw)
                turned[tr][tc] = set(objs)

        def turn(state):
            def pt(p):
                tr, tc = cell(divmod(p, hw[1]), hw)
                return tr * tw + tc
            return (pt(state[0]), tuple(sorted(map(pt, state[1]))),
                    tuple(sorted(map(pt, state[2]))),
                    tuple(sorted(map(pt, state[3]))))

        eng.load_level(turned)
        for i, d in enumerate(presses):
            eng.step(dmap[d])
            if expert.read(eng) != turn(ref[i + 1]):
                notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                break
    return notes


def _symmetry(walk: int = 300) -> int:
    """Measure ON THE INTERPRETER that the 8-element presentation group is an
    exact symmetry of the mechanic -- which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation on top of the mandatory
    rotation.

    The argument is that the four probe rays are DISJOINT, so each scared sheep
    owns its own axis and no two can contest a cell -- the shape that hid Gobble
    Rush's chirality cannot arise. But this game moves up to four sheep at once,
    which is exactly the case that is not argued in this family, so it is
    measured: each level's own plan (the sequence the corpus records) plus a
    seeded random walk (which does the things a shortest plan never does --
    press into walls, shove crates into corners, scare several sheep at once),
    replayed on all seven other presentations.
    """
    _solver, game, expert = _built()
    eng, g = game._engine, game._game
    bad = 0
    for level in _levels(game):
        game.set_level(level)
        plan = expert.plan(eng, level) or []
        game.set_level(level)
        layout = g.levels[level]
        rng = random.Random(f"sheep:symmetry:{level}")
        runs = {"plan": list(plan),
                "walk": [rng.choice(_Board.actions) for _ in range(walk)]}
        notes = []
        for kind, presses in runs.items():
            notes += [f"{kind}:{n}" for n in
                      _replay_transformed(eng, expert, layout, presses)]
        bad += len(notes)
        print(f"  L{level}: ({len(runs['plan'])} plan + {walk} random) presses "
              f"x 7 presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print("symmetry clean -- the flip augmentation is exact" if not bad
          else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 1 if bad else 0


def _audit() -> int:
    """Assert every cell COMPOSITION the game can show is distinct, at every
    cell size its levels use.

    Whole boards, whole frames: the per-cell crop the older generators in this
    tree use is wrong whenever `_render_frame` upscales, which is nearly every
    board (see [[explod-rendering-fixes]]). Rendering is per-cell independent,
    so filling the board with one composition is the same test with no geometry
    to get wrong.
    """
    game = SheepSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {"floor": (), "wall_a": ("walla",), "wall_b": ("wallb",),
             "target": ("target",), "crate": ("crate",),
             "crate_on_target": ("target", "crate"), "sheep": ("sheep",),
             "penned": ("target", "pennedsheep"), "player": ("player",),
             "player_on_target": ("target", "player")}
    sizes: dict[tuple[int, int], list[int]] = {}
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
        print(f"{h:2d}x{w:2d} (levels {','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--selfcheck" in sys.argv:
        pos = sys.argv.index("--selfcheck") + 1
        trials = (int(sys.argv[pos]) if pos < len(sys.argv)
                  and sys.argv[pos].isdigit() else 400)
        sys.exit(_selfcheck(trials))
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(SheepSolver.main())
