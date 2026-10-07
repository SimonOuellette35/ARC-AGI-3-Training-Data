"""Generate Phase-1 training data for the PuzzleScript game ps:rbg
("RBG", NiGHTcapD, modded by chaotic_iak).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: an exact distance-to-win FIELD over the ball of states a
shortest solution can pass through (built by driving the real interpreter, so
there is no model to fuzz), the optimal-action oracle that falls out of it, and
the one art fix the shipped file needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_rbg",
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
exactly. Every expert step also carries the full set of equally-optimal presses.

The game
--------
Additive colour mixing, played with one key. ``Player = Red or Blue or Green or
Cyan or Magenta or Yellow``, so **every coloured blob on the board is a player
and they all move together on every press** -- there is no cursor, no selection
and no wait. The win condition is ``Some White``, and white is what you get by
bringing a primary together with the secondary that completes it.

Three mechanics, and all of them are about what happens when a moving blob is
stopped:

  * **Mixing needs one blob MOVING and the other STATIONARY.**
    ``[ > Green | stationary Blue ] -> [ | Cyan ]`` and its five siblings. Since
    one press pushes every blob the same way, two blobs travelling together
    never mix -- the front one has to be stopped first (by a wall, by a coloured
    wall that refuses it, or by the board edge), and only then does the one
    behind it run into it. So the whole game is *arranging collisions against
    scenery*: R+G -> Yellow, R+B -> Magenta, G+B -> Cyan, and the six
    primary-into-complementary-secondary pairs (R into C, B into Y, G into M and
    the three reverses) -> **White**, which ends the level.
  * **Coloured walls are FILTERS, and passing one SPLITS a secondary.**
    ``[ > Cyan | BlueWall ] -> [ Green | Blue BlueWall ]``: cyan is green+blue,
    so a cyan blob shoved at a blue wall leaves its blue component *inside the
    wall cell* and stays behind as green. The eight rules of this shape are the
    only way to take a colour apart, and they are why a blob and a coloured wall
    can share a cell (see "Rendering" -- the shipped art hid exactly that).
    A wall of the blob's own colour swallows it whole
    (``[ > Cyan | CyanWall ] -> [ | Cyan CyanWall ]``), and a PRIMARY passes
    into its own wall unchanged (no rule blocks ``Red`` at a ``RedWall``) while
    every other coloured wall stops it dead.
  * **Yellow DUPLICATES against the other two secondaries.**
    ``[ > Yellow | CyanWall ] -> [ Yellow | Green CyanWall ]`` and
    ``[ > Yellow | MagentaWall ] -> [ Yellow | Red MagentaWall ]``: the shared
    primary appears in the wall cell and the yellow blob is *unchanged*, i.e.
    matter is created. Cyan and Magenta have no such rule (they are simply
    blocked by the other secondaries' walls), so this is an asymmetry of the
    shipped game rather than a general law -- it is also what makes level 4
    winnable, so it stays.

Plain brown ``Wall`` stops everything, and a press that moves nothing is a total
no-op (there is no world to tick), so a wasted press is a self-loop and can
never be part of a shortest plan.

``noaction`` is declared and no rule mentions ``action``, so ACTION is a
self-loop on every board. That is not taken on trust -- the ouroboros precedent
in this family is a game whose rules section says exactly this while ACTION is
secretly a wait: the first stage of ``--verify`` presses it from every state of
every level's ball and requires the grid to come back unchanged.

The levels
----------
The shipped file has five, all winnable, all 7x10, all with three blobs:

    level  blobs    coloured walls            plan  ties  states  what it is
    0      R G B    --                           9   44%     489  mix in an
                                                                  empty room
    1      R G B    cyan                        11   27%     367  a doorway only
                                                                  cyan fits through
    2      R G B    blue, cyan                  17   35%     629  ... and a filter
                                                                  above it
    3      R G B    blue x2, magenta            21   38%    1008  split a magenta
                                                                  to make the red
    4      R G B    red, blue, cyan, magenta    22   32%    1140  three sealed
                                                                  rooms, joined
                                                                  only by filters

``plan`` is proved SHORTEST (the field is exact on the ball that contains every
shortest solution), ``ties`` is the share of plan steps with more than one
equally-optimal press, and ``states`` is the size of that ball. The longest plan
is 22 presses against the adapter's 200-step per-level budget, so this game
needs no ``games/`` step-limit wrapper.

Expert solver
-------------
There is no native model here and nothing to fuzz: the interpreter IS the model,
and at ~2500 ``eng.step``/s the whole game plans in about ten seconds. But the
*complete* reachable space is not small -- three blobs roaming a 8x5 room with
merges and splits runs to six figures and does not enumerate in reasonable time,
so the im_sick_today "enumerate everything" shape does not apply. What does
apply is the lovendpieces one: **the wins are SHALLOW even though the space is
wide** (d* = 9..22), so `_Field` builds only the ball a shortest solution can
live in (367-1140 states per level) --

  1. a LAYERED forward BFS from the level start over ``eng.step``, recording the
     successor of every state under each of the four directions and marking the
     presses that WIN (winning is a transition, not a state: ``check_win`` is
     read after a press and a won level is terminal, so those edges lead
     nowhere). The sweep stops at the end of the first layer that produces a
     win, i.e. after expanding every state at depth <= d* - 1;
  2. a reverse BFS over the predecessor map collected on the way, from the
     states that win in one press.

That is exact where it has to be. For a state at depth ``g`` whose true
distance-to-win is ``h``, every state on its shortest route to the win sits at
depth <= ``g + h - 1``, so as long as ``g + h <= d*`` the whole route is inside
the expanded subgraph and the reverse pass prices it correctly. Every state a
shortest plan from the start passes through satisfies that by construction, and
a state that does not can only be *over*-priced (the subgraph omits edges, it
never invents them) -- so no press is ever wrongly labelled optimal, and none is
wrongly left out. ``--verify`` re-derives all of it from the other end anyway.

States are stored as their canonical cell sets and the grid is REBUILT from the
set to re-expand a state, which keeps the queue small; the rebuild is exact
because ``PSEngine`` carries no state between presses beyond the grid (forces are
local to ``step``), and the first stage of ``--verify`` measures that too rather
than arguing it.

Optimal-action sets
-------------------
The field gives these exactly and for free: at a state ``s`` with distance ``d``,
a press is optimal iff it wins (when ``d == 1``) or reaches a state at distance
``d - 1``. A press the engine refuses maps ``s`` to itself, whose distance is
``d``, so no-ops are excluded by construction. No step ships unlabelled (the
always-emit-optimal-targets rule), and a step with a unique optimal press ships a
one-element set. This game ties a lot -- 27-44% of plan steps -- because a blob
sliding across an empty room can take the two axes of its journey in any order.

Rendering
---------
The one fix the shipped file needed, and it is the recurring (ground, body) shape
from this family. A blob and a coloured wall SHARE a cell -- that is what every
filter rule produces -- and the seven colour objects shipped as solid 5x5 blocks,
so a blob standing on a coloured wall hid it completely: ``red`` and
``red_on_bluewall`` were pixel-identical, and so were all 42 (body, coloured
wall) stacks. Which rule fires next is decided by that hidden wall, so the
mechanic was literally not in the frame. Fixed in
``data/puzzlescript_games/RBG.txt`` by drawing the bodies as a 3x3 core inside a
transparent ring, which lets the wall's checkerboard show around them; the ring
is symmetric under the whole 8-element presentation group, so it costs the
augmentation nothing. ``--audit`` renders all 57 compositions the game can show
and asserts they are pairwise distinct (they are, at the single 6px cell size
every level uses).

Deliberately NO colour augmentation: the colours ARE the mechanic here, and a
flattening recolor would merge blobs that mix differently.

Augmentation
------------
This game's engine state after reset is identical for every seed (levels are
fixed ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to). 5 levels x 16
presentations = 80.

The flips are safe on the ESCAPE! argument -- gravity-free, every rule written
with the relative ``>`` force, a win condition (``Some White``) that names no
direction, screen-relative input, and no sprite that encodes a facing. ``ps:rbg``
does have blobs contesting cells, which is the shape that makes rule-expansion
order visible, so it is measured rather than assumed: ``--symmetry`` replays
every state of every level's ball on all eight turned and mirrored copies of the
board and reports how many transitions any of them splits.

The expert plan is therefore seed-independent: solved once per level, cached to
``data/rbg_plans.json``, and replayed per seed with that seed's remapped screen
actions.

Recovery
--------
``recovery_mode = "reset"`` (the family default), and ``epsilon = 0``. The
mechanics are irreversible -- two blobs that merge are one blob forever, a
component filtered into a wall cannot be pulled back out -- so a perturbed state
generally cannot be planned out of, and the field is a ball around the START
state rather than the closed reachable set, so pricing a detour means building a
new one. The episode-wide epsilon prefix explores freely and ONE RESET restores
the level start, from which the cached plan replays a guaranteed win.

Usage (run from the repo root):
    python solvers/generate_rbg_training.py \
        --episodes 200 --out data/training_multi_level/rbg

    python solvers/generate_rbg_training.py --plans
    python solvers/generate_rbg_training.py --verify
    python solvers/generate_rbg_training.py --symmetry
    python solvers/generate_rbg_training.py --audit
"""

from __future__ import annotations

import itertools
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan      # noqa: E402

GAME_NAME = "RBG"

#: Where each level's start plan is cached between processes. The fields are
#: seed-independent, so without this every `parallelize_generator` shard would
#: re-build all five balls (~10s) before recording its first frame.
PLAN_CACHE = Path(__file__).resolve().parent.parent / "data" / "rbg_plans.json"

#: The four presses, in the order the labels list them. ACTION is a self-loop on
#: every board (``noaction`` is declared and no rule reads it), measured by
#: `_actions`, so it is not in the search.
_ORDER = ("up", "down", "left", "right")

#: `_Field.succ` entry for a press that wins. A won level is terminal -- the
#: recorder stops on it -- so the edge has no destination state.
_WIN = -1

#: The seven bodies and the six coloured walls, for `--audit` and the reports.
_BODIES = ("red", "green", "blue", "cyan", "magenta", "yellow", "white")
_CWALLS = ("redwall", "greenwall", "bluewall",
           "cyanwall", "magentawall", "yellowwall")


# ---------------------------------------------------------------------------
# The exact distance-to-win field
# ---------------------------------------------------------------------------

class _Field:
    """Every state a SHORTEST solution can pass through, with its exact distance
    to a win, computed by driving the real interpreter.

    The state is the set of ``(row, col, object)`` triples of every non-
    background cell -- i.e. the whole board minus Background, which every cell
    carries unconditionally. That is canonical: ``PSEngine`` keeps nothing
    between presses except the grid (forces live and die inside ``step``), so
    the grid is the state, and the grid is REBUILT from the set to re-expand it
    (measured over 4800 transitions by ``--verify``).

    The forward sweep is LAYERED and stops at the end of the first layer that
    produces a win, which is depth ``d* - 1``. That is exactly the ball every
    shortest solution lives in -- see the module header for why the reverse pass
    is then exact on it -- and it is what makes this affordable: the complete
    reachable space of these boards runs to six figures, the balls are 341-1101
    states.
    """

    def __init__(self, eng, bg: int, cap: int = 200_000):
        self.eng = eng
        self.bg = bg
        self.cap = cap
        self.keys: list[frozenset] = []
        self.succ: list[list[int] | None] = []
        self.dist: dict[int, int] = {}
        self.star: int | None = None      # d*, once a winning edge is found
        self.overflow = False

    # -- engine plumbing ------------------------------------------------------
    def key(self) -> frozenset:
        bg = self.bg
        return frozenset(
            (r, c, o)
            for r, row in enumerate(self.eng.grid)
            for c, cell in enumerate(row)
            for o in cell
            if o != bg
        )

    def seat(self, key: frozenset) -> None:
        """Put ``key``'s board on the engine. Every cell of a parsed level
        carries Background (asserted at build time), so the rebuild is the key
        plus Background everywhere."""
        eng, bg = self.eng, self.bg
        grid = [[{bg} for _ in range(eng.width)] for _ in range(eng.height)]
        for (r, c, o) in key:
            grid[r][c].add(o)
        eng.grid = grid
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    # -- the two sweeps -------------------------------------------------------
    def build(self) -> "_Field":
        """Layered forward BFS out to depth ``d* - 1``, then the reverse BFS that
        turns the edges it collected into a distance-to-win field."""
        eng = self.eng
        assert all(self.bg in cell for row in eng.grid for cell in row), \
            "a cell without Background: the key rebuild would drop it"
        start = self.key()
        index = {start: 0}
        self.keys = [start]
        self.succ = [None]
        layer = [0]
        depth = 0
        while layer and self.star is None:
            nxt: list[int] = []
            for i in layer:
                row = []
                for direction in _ORDER:
                    self.seat(self.keys[i])
                    eng.step(direction)
                    if eng.check_win():
                        row.append(_WIN)
                        continue
                    k = self.key()
                    j = index.get(k)
                    if j is None:
                        j = len(self.keys)
                        index[k] = j
                        self.keys.append(k)
                        self.succ.append(None)
                        nxt.append(j)
                    row.append(j)
                self.succ[i] = row
                if _WIN in row and self.star is None:
                    self.star = depth + 1     # finish the layer, then stop
            if len(self.keys) > self.cap:                     # pragma: no cover
                self.overflow = True
                return self
            depth += 1
            layer = nxt

        preds: list[list[int]] = [[] for _ in self.keys]
        frontier = []
        for i, row in enumerate(self.succ):
            if row is None:              # a depth-d* state, deliberately unexpanded
                continue
            for j in row:
                if j == _WIN:
                    if i not in self.dist:
                        self.dist[i] = 1
                        frontier.append(i)
                else:
                    preds[j].append(i)
        queue = deque(frontier)
        while queue:
            i = queue.popleft()
            for p in preds[i]:
                if p not in self.dist:
                    self.dist[p] = self.dist[i] + 1
                    queue.append(p)
        return self

    # -- reading it off -------------------------------------------------------
    def optimal(self, state: int) -> list[str]:
        """Every press that is equally shortest at ``state``."""
        d = self.dist[state]
        row = self.succ[state]
        if d == 1:
            return [a for a, j in zip(_ORDER, row) if j == _WIN]
        return [a for a, j in zip(_ORDER, row)
                if j != _WIN and self.dist.get(j, 1 << 30) == d - 1]

    def plan(self) -> tuple[list[str] | None, list[list[str]]]:
        """``(presses, optsets)`` for the walk downhill from the level start, or
        ``(None, [])`` when no win is reachable."""
        if 0 not in self.dist:
            return None, []
        presses: list[str] = []
        optsets: list[list[str]] = []
        state = 0
        while True:
            best = self.optimal(state)
            optsets.append(best)
            presses.append(best[0])
            if self.dist[state] == 1:
                return presses, optsets
            state = self.succ[state][_ORDER.index(best[0])]


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class RBGExpert(PSExpert):
    """Plans by building `_Field` -- the ball of states a shortest solution can
    pass through -- and walking its exact distance-to-win field downhill.

    `PSExpert` still owns everything around that: the in-memory memo, the
    on-disk plan cache with its staleness check, and the snapshot/restore
    discipline. The only override is `_search`. `heuristic` is unreachable by
    construction: nothing here runs A*. The inherited `_key` (the whole board
    minus Background) is already exactly `_Field`'s state, so the memo and the
    disk cache's staleness check agree with the field.
    """

    directions = list(_ORDER)
    plan_cache_path = PLAN_CACHE

    def heuristic(self, eng) -> int:                          # pragma: no cover
        raise AssertionError("the field is exact; no search runs here")

    def field(self, eng) -> _Field:
        return _Field(eng, self.bg_id).build()

    def _search(self, eng) -> list | None:
        presses, optsets = self.field(eng).plan()
        return None if presses is None else Plan(presses, optsets)


class RBGSolver(PSAStarSolver):
    game_id = "puzzlescript_rbg"
    game_name = GAME_NAME
    game_module_id = "ps:rbg"
    expert_cls = RBGExpert

    #: The longest plan is 22 presses; the rest is room for the exploration
    #: prefix and the re-plan after it, and it stays well under the adapter's
    #: 200-step per-level budget.
    max_steps = 100

    def prepare_expert(self, game, expert) -> None:
        """Build every level's field before `discover_solvable` asks for it --
        the same work either way, but it makes the one-time ~10s visible as
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
    solver = RBGSolver()
    game = solver.make_game(0)
    return solver, game, RBGExpert(game)


def _census(eng, game) -> str:
    """The blobs and coloured walls on the engine's current grid, as a string."""
    idx = game.obj_name_to_idx
    counts = {}
    for row in eng.grid:
        for cell in row:
            for name in _BODIES + _CWALLS:
                if idx[name] in cell:
                    counts[name] = counts.get(name, 0) + 1
    blobs = "".join(n[0].upper() * counts.get(n, 0) for n in _BODIES)
    walls = ",".join(f"{n[:-4]}{'x%d' % counts[n] if counts[n] > 1 else ''}"
                     for n in _CWALLS if n in counts) or "--"
    return f"{blobs:<5s} {walls}"


def _report() -> int:
    """Per-level board size, contents, plan length, tie coverage and ball size --
    and CERTIFY each plan by replaying it through the interpreter."""
    _solver, game, expert = _levels()
    total = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        census = _census(eng, game._game)
        t = time.time()
        field = expert.field(eng)
        presses, optsets = field.plan()
        took = time.time() - t
        head = (f"level {level}: {eng.height}x{eng.width} {census:<34s} "
                f"{len(field.keys):5d} states in {took:5.1f}s")
        if presses is None:
            why = ("GAVE UP AT THE STATE CAP" if field.overflow
                   else "NO WIN IS REACHABLE")
            print(f"{head}: {why}")
            bad += 1
            continue
        # Certification: the interpreter must win on the LAST press and no
        # earlier (an earlier win would mean the plan is not shortest).
        game.set_level(level)
        won_at = None
        for i, direction in enumerate(presses):
            eng.step(direction)
            if eng.check_win():
                won_at = i
                break
        ok = won_at == len(presses) - 1
        bad += not ok
        total += len(presses)
        ties = sum(1 for s in optsets if len(s) > 1)
        room = "ok" if len(presses) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(presses):3d} presses "
              f"(budget {game._max_steps}, {room}), "
              f"{ties:2d} tie steps ({ties / len(presses):3.0%}), "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    print(f"total {total} presses over {game.n_levels} levels")
    return 0 if not bad else 1


def _actions() -> int:
    """Assert the two modelling assumptions the field rests on, by measurement.

    1. **The grid IS the state.** `_Field` keys a state on the board minus
       Background and REBUILDS the grid from that key to re-expand it, so any
       engine state carried between presses (a leftover force, a pending
       ``again``, a stale position index) would make the whole enumeration --
       distances, plans and every training label -- quietly wrong. So it is
       measured against a grid the engine actually WALKED to: random-walk each
       level from ``set_level`` with no re-seating at all, then step both that
       live grid and its rebuild with each of the four presses and compare the
       resulting boards and ``check_win``.
    2. **ACTION is a self-loop.** The game declares ``noaction`` and no rule
       mentions ``action``, so it is not in `_ORDER` at all -- but ps:ouroboros
       says exactly the same thing in its rules section while ACTION there is a
       secret wait that halves every distance, so a search without it would
       label long routes optimal. Here it is pressed from every state of every
       ball and the grid must come back unchanged.
    """
    import random

    _solver, game, expert = _levels()
    eng = game._engine
    rng = random.Random(20260815)
    field = _Field(eng, expert.bg_id)
    field_bad = walked = 0
    for level in range(game.n_levels):
        for _ in range(60):
            game.set_level(level)                 # live grid from here on
            for _ in range(rng.randrange(14)):
                eng.step(rng.choice(_ORDER))
            live = [[set(cell) for cell in row] for row in eng.grid]
            key = field.key()
            for direction in _ORDER:
                eng.grid = [[set(cell) for cell in row] for row in live]
                eng._position_index_dirty = True
                eng._rule_noop_cache.clear()
                eng.step(direction)
                want = (field.key(), eng.check_win())
                field.seat(key)                   # ... and from the rebuild
                eng.step(direction)
                field_bad += (field.key(), eng.check_win()) != want
                walked += 1
    print(f"{walked} transitions stepped from a WALKED grid and from its "
          f"rebuild: {'the grid is the whole state' if not field_bad else f'{field_bad} DISAGREEMENTS'}")

    action_bad = states = 0
    for level in range(game.n_levels):
        game.set_level(level)
        lvl = expert.field(eng)
        for key in lvl.keys:
            lvl.seat(key)
            eng.step("action")
            action_bad += lvl.key() != key or eng.check_win()
            states += 1
    print(f"{states} states pressed with ACTION: "
          f"{'ACTION is a self-loop everywhere' if not action_bad else f'{action_bad} CHANGED THE BOARD'}")
    return 0 if not (field_bad or action_bad) else 1


def _verify() -> int:
    """Double-entry check of every plan length and every shipped tie set.

    `_Field` answers both from ONE sweep pair: a layered forward BFS that builds
    the successor table and a reverse BFS over its predecessor map. A bug in the
    reverse sweep (or in the ball's truncation depth) would produce a
    self-consistent field, a plan that still wins, and tie sets that are quietly
    wrong -- and the training labels ARE those tie sets, so "it wins" is not
    enough of a check.

    So each candidate press is priced again from the other end, by an
    independent forward BFS from its successor that stops at the first winning
    press: no reverse sweep, no predecessor map, no truncation, no shared state
    but ``eng.step`` itself. A press is optimal iff that distance is
    ``d* - i - 1``, and that set must be exactly what the field shipped.

    Memoized on the successor key, because the plans revisit states.
    """
    _solver, game, expert = _levels()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        field = expert.field(eng)
        presses, optsets = field.plan()

        t = time.time()
        memo: dict[frozenset, int | None] = {}

        def distance(key: frozenset) -> int | None:
            """Presses from ``key`` to a win, by a fresh BFS. None if none."""
            if key in memo:
                return memo[key]
            seen = {key}
            queue = deque([(key, 0)])
            answer = None
            while queue and answer is None:
                k, d = queue.popleft()
                for direction in _ORDER:
                    field.seat(k)
                    eng.step(direction)
                    if eng.check_win():
                        answer = d + 1
                        break
                    nxt = field.key()
                    if nxt not in seen:
                        seen.add(nxt)
                        queue.append((nxt, d + 1))
            memo[key] = answer
            return answer

        notes = []
        state = 0
        star = len(presses)
        for i, direction in enumerate(presses):
            want = []
            for a in _ORDER:
                j = field.succ[state][_ORDER.index(a)]
                if j == _WIN:
                    got = 1
                else:
                    got = distance(field.keys[j])
                    got = None if got is None else got + 1
                if got == star - i:
                    want.append(a)
            if want != optsets[i]:
                notes.append(f"step {i}: {optsets[i]} != {want}")
            state = field.succ[state][_ORDER.index(direction)]
        # The start's own distance, priced the same independent way.
        if distance(field.keys[0]) != star:
            notes.append(f"LENGTH {distance(field.keys[0])} != {star}")
        bad += len(notes)
        print(f"level {level}: d*={star:3d}, "
              f"{sum(len(s) for s in optsets):3d} labelled presses re-priced by "
              f"{len(memo):4d} independent searches in {time.time() - t:5.1f}s: "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _symmetry() -> int:
    """Measure, on the interpreter, which transitions are decided by the way the
    board is FACING -- and whether the mandatory rotation already exposes them.

    This is the evidence behind putting the game in `_FLIP_GAMES`. Nothing in
    the rules names an absolute direction, but two blobs can contest one cell,
    and PuzzleScript settles that by the order the interpreter expands a rule's
    four directions (see [[ps-engine-render-gotchas]] #7) -- which is a fact
    about the ENGINE, not about the screen, so a presentation transform cannot
    change it. The question is therefore not "is the game symmetric" but "does a
    mirror expose an inconsistency the rotation does not already carry".

    So: every state of every level's ball, every press. Step the board; then
    step each of the 8 turned/mirrored copies of it with the correspondingly
    turned press and un-transform the result. A transition is CHIRAL when any
    copy disagrees, and it is MIRROR-ONLY when the three rotations all agree and
    a mirror does not. Zero mirror-only transitions is the claim.
    """
    _solver, game, expert = _levels()
    eng = game._engine

    def turn(cells, k, mirror, h, w):
        """The state (a set of ``(r, c, o)``) under mirror-then-``k``-quarter-
        turns clockwise, with the board's own height/width following along."""
        out = set()
        for (r, c, o) in cells:
            rr, cc, hh, ww = r, (w - 1 - c) if mirror else c, h, w
            for _ in range(k):
                rr, cc, hh, ww = cc, hh - 1 - rr, ww, hh
            out.add((rr, cc, o))
        return frozenset(out), (w if k % 2 else h), (h if k % 2 else w)

    _CW = {"up": "right", "right": "down", "down": "left", "left": "up"}
    _MIRROR = {"left": "right", "right": "left", "up": "up", "down": "down"}

    def turn_dir(d, k, mirror):
        if mirror:
            d = _MIRROR[d]
        for _ in range(k):
            d = _CW[d]
        return d

    bad = total = chiral = mirror_only = 0
    for level in range(game.n_levels):
        game.set_level(level)
        h0, w0 = eng.height, eng.width
        field = expert.field(eng)
        t = time.time()
        lvl_chiral = lvl_mirror = 0
        for key in field.keys:
            for direction in _ORDER:
                eng.height, eng.width = h0, w0
                field.seat(key)
                eng.step(direction)
                ref = field.key()
                refwin = eng.check_win()
                total += 1
                split_by_rotation = split_by_mirror = False
                for k in range(4):
                    for mir in (False, True):
                        if (k, mir) == (0, False):
                            continue
                        tk, th, tw = turn(key, k, mir, h0, w0)
                        eng.height, eng.width = th, tw
                        field.seat(tk)
                        eng.step(turn_dir(direction, k, mir))
                        got = (field.key(), eng.check_win())
                        want = (turn(ref, k, mir, h0, w0)[0], refwin)
                        if got != want:
                            if mir:
                                split_by_mirror = True
                            else:
                                split_by_rotation = True
                if split_by_rotation or split_by_mirror:
                    lvl_chiral += 1
                    if split_by_mirror and not split_by_rotation:
                        lvl_mirror += 1
        eng.height, eng.width = h0, w0
        chiral += lvl_chiral
        mirror_only += lvl_mirror
        bad += lvl_mirror
        print(f"level {level}: {len(field.keys) * 4:5d} transitions, "
              f"{lvl_chiral:4d} chiral, {lvl_mirror:3d} split ONLY by a mirror "
              f"({time.time() - t:5.1f}s)")
    print(f"{total} transitions, {chiral} decided by the rule order, "
          f"{mirror_only} of them split only by a mirror")
    print("the flips add no inconsistency the rotation does not already carry"
          if not bad else
          f"FLIPS ARE NOT SAFE: {mirror_only} mirror-only transitions")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION the game can show is distinct.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor).

    The pairs that matter here are the 42 (body, coloured wall) STACKS, which
    every filter rule produces and which the shipped solid-block art rendered
    identically to the bare body -- see "Rendering" in the header. Plain Wall
    shares a collision layer with the bodies and with White, so no body can ever
    stand on it and those combinations are not compositions the game can show.
    """
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {"floor": (), "wall": ("wall",)}
    comps.update({w: (w,) for w in _CWALLS})
    comps.update({b: (b,) for b in _BODIES})
    comps.update({f"{b}_on_{w}": (w, b) for w in _CWALLS for b in _BODIES})

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
        print(f"{h}x{w} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} compositions, "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes[:4])}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_actions() or _verify())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(RBGSolver.main())
