"""Generate Phase-1 training data for the PuzzleScript game ps:im_sick_today
("I'm Sick Today", Ben Porter).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: the exact distance-to-win FIELD over the level's whole
reachable state space (built by stepping the real interpreter, so there is no
model to fuzz), the optimal-action oracle it hands out for free, and the level
work the shipped file needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_im_sick_today",
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
You are patient zero. The win condition is ``No Person``, and the only way to
remove a Person is to stand next to one: the whole game is ten rules, and four
of them are the ones that matter.

  * **You infect by TOUCHING, and touching MOVES you.**
    ``late [InfectedPlayer | Person] -> [Infected | InfectedPlayer]``. Read it
    carefully -- the cell you were standing in becomes ``Infected`` and *the
    person becomes the new player*. The rule then runs to a fixpoint, so the
    player CHAIN-HOPS through the connected cluster, leaving an Infected cell
    behind at every step; ``late [Infected | Person] -> [Infected | Infected]``
    catches whatever the chain did not walk onto, and
    ``late [Infected] -> [Dead]`` turns all of it into corpses. So one press
    into a cluster kills the ENTIRE cluster and teleports you somewhere inside
    it. Which cell you land on is decided by the interpreter's rule-expansion
    order, not by anything the level says -- see "Augmentation" below.
  * **Corpses are walls.** ``Dead`` is in ``Collidable``, and no rule removes
    it. A cluster you kill is permanent terrain, so the ORDER you take clusters
    in decides which parts of the board you can still cross afterwards. This is
    the whole puzzle on the multi-cluster levels.
  * **Doctors cure you, and being cured is losing.**
    ``late [InfectedPlayer | Doctor] -> [CuredPlayer | Doctor]``. A CuredPlayer
    infects nobody and no rule ever re-infects it, so on a level with a Person
    left the game is over -- silently, with no GAME_OVER and no lose condition.
    The field needs no explicit death test for it: a cured state simply cannot
    reach a win, so the reverse sweep never labels it and no plan can route
    through it. Note the infection rule is declared FIRST, so arriving next to a
    doctor and next to a person on the same turn infects rather than cures.
  * **Doctors chase you along rows and columns.**
    ``[Doctor | ... | stationary InfectedPlayer] -> [> Doctor | ... |
    InfectedPlayer]`` -- one step toward you whenever you share a row or a
    column, THROUGH walls (``...`` constrains nothing). Two non-obvious things
    make this the timing puzzle it is:
      - the player's own move is executed by the germ-stain rule
        (``[> InfectedPlayer | No Collidable] -> [random GermStain |
        InfectedPlayer]``), which *teleports* the player and leaves it
        ``stationary``, and that rule is declared BEFORE the doctor rule. So
        doctors step toward where you just arrived, not where you came from;
      - a REFUSED press (into a wall, a corpse, a person, a doctor) leaves the
        player with its force intact and no cell to teleport into, so the
        doctor rule does not match and **nothing happens at all**. There is no
        wait move, which makes the parity of a detour the actual mechanic on
        levels 4-6. (Same convention as ps:goblin_hooblob; the opposite one is
        just as common, so it is worth measuring in any new ps: game.)

Two smaller rules: ``[> Player | Bed] -> [ | BedWithInfected]`` plus
``[BedWithInfected] [Person] -> [BedWithInfected] [Person2]`` -- going to bed
converts every Person into a Person2, which satisfies ``No Person`` outright and
is the whole of the last level -- and ``DOWN [WallStart | No Walls] ->
[Wall_S...]``, which runs once at level start (``run_rules_on_level_start``) to
pick the wall sprites. That same level-start pass is why the doctors on levels
3, 5 and 11 have already taken one step before the first frame.

GermStain is cosmetic AND invisible: it is painted ``#010``, which quantizes to
ARC 5, and so is the Background (``#001 #000``). It sits on its own collision
layer and no rule reads it, so the state this file keys on DROPS it. That is the
only modelling assumption here and it is measured rather than argued -- the
first stage of ``--verify`` steps 3120 transitions from random positions with
and without the stains present and compares the resulting boards and
``check_win``. Without it the state key would carry the player's whole path
history, every state would be distinct, and nothing would ever dedup: level 11's
21988 states become a search that does not terminate.

The levels
----------
The shipped file has thirteen. Twelve are winnable and are recorded; level 10 --
the 27-row "these hospitals seem crowded" joke board -- is not, and is skipped:

    level  size   persons doctors  plan  ties   states   what it is
    0       7x6      2       0        6   33%        13   two rooms, two people
    1      7x10      3       0       15   13%       151   ... and a corridor
    2      7x10      4       0       24   25%       101   ... and a fourth person
    3      3x15      1       1        5    0%        40   a race down one corridor
    4       7x6      2       1        6   33%        30   the doctor arc opens
    5       7x7      4       2        7   29%        54   parity: step aside to
                                                          make the doctors miss
    6      7x10      4       1       22    9%       624   doctor in the open
    7       7x6      8       0        4   25%         8   ONE press kills eight
    8      10x8    13       0        17   12%        24   three wards
    9      8x12    21       1        11   27%       142   a crowded ward
    10    27x12   130      89         -    -         60   UNWINNABLE, skipped
    11     7x12     1       4        12   17%     21988   one patient, four
                                                          doctors, open floor
    12      3x9     1       0         4    0%         4   go back to bed

``plan`` is proved SHORTEST, not merely found: ``states`` is the size of the
level's ENTIRE reachable state space, which the field enumerates exhaustively,
so the distance it reports is the true one. ``ties`` is the share of plan steps
with more than one equally-optimal press. The longest plan is 24 presses against
the adapter's 200-step per-level budget, so this game needs no `games/`
step-limit wrapper.

**Why level 10 cannot be won here.** Its two person-masses are eleven rows deep
and threaded with doctors. The chain-hop kills only the row the player enters,
because `_execute_late_single` drives each of a rule's four directional
expansions to a fixpoint IN TURN and never re-runs the rule over all four (see
[[ps-engine-render-gotchas]] #7): the chain runs along one row, and by the time
it reaches a cell with a person above it, the "up" expansion has already been
retired. The player is then left standing next to a doctor and is cured on the
same turn. The whole reachable space is 60 states and contains no win, so this
is a property of the board and the interpreter, not of the search -- ``--plans``
re-derives it every run. It is left in the file (deleting it would renumber the
levels a live agent plays) and named in ``skip_levels``.

Expert solver
-------------
There is no native model here and nothing to fuzz: the interpreter IS the model.
Every level's reachable state space is small enough to enumerate outright (the
largest is 21988 states, 21s; all twelve together are ~23s), so `_Field` does
exactly that --

  1. a forward BFS from the level start over ``eng.step``, recording the
     successor of every state under each of the four directions and marking the
     presses that WIN (winning is a transition, not a state: ``check_win`` is
     read after a press, and a won level is terminal, so those edges lead
     nowhere);
  2. a reverse BFS over the predecessor map from the states that can win in one
     press, which labels every state that can reach a win at all with its EXACT
     distance;

and the plan is the walk downhill from the start. States are stored as their
canonical germstain-free cell sets and the grid is REBUILT from that set when a
state has to be re-expanded -- keeping a snapshot per queued state instead costs
~370 MB on level 11, and the rebuild is exact (same 3120-transition check as
above).

This is strictly better than an A* over the same interpreter, and for a reason
worth stating: a heuristic here would have to estimate the cost of visiting
several clusters in an order that does not wall you in with your own corpses,
which is a travelling-salesman problem over a board whose walls change as you
solve it. Nothing cheap is admissible, and the space is small enough that the
question never has to be asked.

Optimal-action sets
-------------------
The field gives these exactly and for free: at a state ``s`` with distance ``d``,
a press is optimal iff it wins (when ``d == 1``) or reaches a state at distance
``d - 1``. A press that the engine refuses maps ``s`` to itself, whose distance
is ``d``, so no-ops are excluded by construction. No step ships unlabelled (the
always-emit-optimal-targets rule), and a step with a unique optimal press ships a
one-element set. ``--verify`` re-derives every one of them from the other end --
an independent bounded BFS from each candidate successor -- so the labels do not
rest on the reverse pass alone.

Augmentation
------------
This game's engine state after reset is identical for every seed (levels are
fixed ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to). 12 levels x
16 presentations = 192.

The flips are safe on the ESCAPE! argument -- gravity-free, the one force in the
game is the relative ``>``, the win condition (``No Person``) names no
direction, input is screen-relative, and no sprite encodes a facing. The one
asymmetry in the game is the cell the chain-hop lands the player on, which is
decided by the interpreter's rule-expansion order and is therefore a fact about
the ENGINE rather than about the screen. That is already fully exposed by the
mandatory rotation (a rotation permutes the four expansions exactly as a mirror
does), so the flips cost nothing in consistency and buy 4x the presentations --
the same argument, measured, that put Gobble Rush in that set. There is
deliberately no colour augmentation: Person (grey body) and Doctor (white body)
differ only in colour once the sprite downsample drops their heads, and a
flattening recolor would merge the person you must touch with the doctor you
must not.

The expert plan is therefore seed-independent: solved once per level, cached to
``data/im_sick_today_plans.json``, and replayed per seed with that seed's
remapped screen actions.

Recovery
--------
``recovery_mode = "reset"`` (the family default). The mechanics are irreversible
-- a corpse is permanent, a cure is permanent -- so a perturbed state generally
cannot be planned out of. The episode-wide epsilon prefix explores freely and
ONE RESET restores the level start, from which the cached plan replays a
guaranteed win.

Usage (run from the repo root):
    python solvers/generate_im_sick_today_training.py \
        --episodes 200 --out data/training_multi_level/im_sick_today

    python solvers/generate_im_sick_today_training.py --plans
    python solvers/generate_im_sick_today_training.py --verify
    python solvers/generate_im_sick_today_training.py --symmetry
    python solvers/generate_im_sick_today_training.py --audit
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

GAME_NAME = "I'm_Sick_Today"

#: Where each level's start plan is cached between processes. The fields are
#: seed-independent, so without this every `parallelize_generator` shard would
#: re-enumerate all twelve state spaces (~23s) before recording its first frame.
PLAN_CACHE = Path(__file__).resolve().parent.parent / "data" / \
    "im_sick_today_plans.json"

#: The four presses, in the order the labels list them. ACTION5 is bound to
#: nothing (the game declares ``noaction``), so it is not in the search at all.
_ORDER = ("up", "down", "left", "right")

#: `_Field.succ` entry for a press that wins. A won level is terminal -- the
#: recorder stops on it -- so the edge has no destination state.
_WIN = -1


# ---------------------------------------------------------------------------
# The exact distance-to-win field
# ---------------------------------------------------------------------------

class _Field:
    """Every state reachable from a level start, with its exact distance to a
    win, computed by driving the real interpreter.

    The state is the set of ``(row, col, object)`` triples of every non-
    background, non-germstain cell -- i.e. the whole board minus Background,
    which every cell carries unconditionally, and minus the germ stains, which
    nothing reads. That is canonical: the grid is REBUILT from the set to
    re-expand a state, and stepping the rebuilt grid gives the same board and
    the same ``check_win`` as stepping the original (measured over 3120
    transitions from random positions on all thirteen levels by ``--verify``).

    Storing the set rather than a snapshot of the grid is what makes the
    enumeration fit: the queue on level 11 holds up to ~22k states, and a
    ``list[list[set]]`` snapshot of that board is ~17 kB each.
    """

    def __init__(self, eng, bg: int, germ: frozenset[int],
                 cap: int = 400_000):
        self.eng = eng
        self.bg = bg
        self.ignore = germ | {bg}
        self.cap = cap
        self.keys: list[frozenset] = []
        self.succ: list[list[int]] = []
        self.dist: dict[int, int] = {}
        self.overflow = False

    # -- engine plumbing ------------------------------------------------------
    def key(self) -> frozenset:
        ignore = self.ignore
        return frozenset(
            (r, c, o)
            for r, row in enumerate(self.eng.grid)
            for c, cell in enumerate(row)
            for o in cell
            if o not in ignore
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
        """Forward BFS over the whole reachable space, then the reverse BFS that
        turns it into a distance-to-win field."""
        eng = self.eng
        assert all(self.bg in cell for row in eng.grid for cell in row), \
            "a cell without Background: the key rebuild would drop it"
        start = self.key()
        index = {start: 0}
        self.keys = [start]
        self.succ = [None]                                    # type: ignore
        queue = deque([0])
        while queue:
            i = queue.popleft()
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
                    self.succ.append(None)                    # type: ignore
                    queue.append(j)
                row.append(j)
            self.succ[i] = row
            if len(self.keys) > self.cap:                     # pragma: no cover
                self.overflow = True
                return self

        preds: list[list[int]] = [[] for _ in self.keys]
        frontier = []
        for i, row in enumerate(self.succ):
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

class ImSickTodayExpert(PSExpert):
    """Plans by enumerating the level's whole reachable space with `_Field` and
    walking the exact distance-to-win field downhill.

    `PSExpert` still owns everything around that -- the in-memory memo, the
    on-disk plan cache with its staleness check and the snapshot/restore
    discipline -- so the only override is `_search`. `heuristic` is unreachable
    by construction: nothing here runs A*.
    """

    directions = list(_ORDER)
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        g = self.g
        self.germ_ids = frozenset(g.resolve_object_name("germstain"))
        self.person_ids = frozenset(g.resolve_object_name("person"))
        self.doctor_ids = frozenset(g.resolve_object_name("doctor"))
        self.cured_ids = frozenset(g.resolve_object_name("curedplayer"))

    def _key(self, eng) -> frozenset:
        # The whole board minus Background and the (cosmetic, invisible) germ
        # stains -- which is exactly `_Field`'s state, so the plan memo and the
        # disk cache's staleness check agree with the field.
        ignore = self.germ_ids | {self.bg_id}
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in cell
            if o not in ignore
        )

    def heuristic(self, eng) -> int:                          # pragma: no cover
        raise AssertionError("the field is exact; no search runs here")

    def field(self, eng) -> _Field:
        return _Field(eng, self.bg_id, self.germ_ids).build()

    def _search(self, eng) -> list | None:
        presses, optsets = self.field(eng).plan()
        return None if presses is None else Plan(presses, optsets)


class ImSickTodaySolver(PSAStarSolver):
    game_id = "puzzlescript_im_sick_today"
    game_name = GAME_NAME
    game_module_id = "ps:im_sick_today"
    expert_cls = ImSickTodayExpert

    #: Level 10 has no win in its entire 60-state reachable space -- see the
    #: header. Skipped up front so `discover_solvable` does not re-enumerate it
    #: at every startup only to reach the same answer.
    skip_levels = frozenset({10})

    #: The longest plan is 24 presses; the rest is room for the exploration
    #: prefix and the re-plan after it, and it stays well under the adapter's
    #: 200-step per-level budget.
    max_steps = 100

    def prepare_expert(self, game, expert) -> None:
        """Build every level's field before `discover_solvable` asks for it --
        the same work either way, but it makes the one-time ~23s visible as
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
    solver = ImSickTodaySolver()
    game = solver.make_game(0)
    return solver, game, ImSickTodayExpert(game)


def _census(eng, expert) -> tuple[int, int]:
    """``(persons, doctors)`` on the engine's current grid."""
    persons = doctors = 0
    for row in eng.grid:
        for cell in row:
            persons += bool(cell & expert.person_ids)
            doctors += bool(cell & expert.doctor_ids)
    return persons, doctors


def _report() -> int:
    """Per-level board size, population, plan length, tie coverage and state
    count -- and CERTIFY each plan by replaying it through the interpreter."""
    _solver, game, expert = _levels()
    total = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        persons, doctors = _census(eng, expert)
        t = time.time()
        field = expert.field(eng)
        presses, optsets = field.plan()
        took = time.time() - t
        head = (f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
                f"{persons:3d} persons {doctors:3d} doctors, "
                f"{len(field.keys):6d} states in {took:5.1f}s")
        if presses is None:
            why = ("GAVE UP AT THE STATE CAP" if field.overflow
                   else "NO WIN IS REACHABLE")
            known = level in ImSickTodaySolver.skip_levels and not field.overflow
            print(f"{head}: {why}{' (known, skipped)' if known else ''}")
            bad += not known
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
              f"{ties:3d} tie steps ({ties / len(presses):3.0%}), "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    print(f"total {total} presses over "
          f"{game.n_levels - len(ImSickTodaySolver.skip_levels)} recorded levels")
    return 0 if not bad else 1


def _stains(walk: int = 14, trials: int = 60) -> int:
    """Assert that DROPPING the germ stains is sound.

    It is the only modelling assumption in this file: `_Field` keys states on
    the board minus Background and minus GermStain, and rebuilds the grid from
    that key to re-expand a state, so a stain that influenced anything would
    make the whole enumeration -- distances, plans and every training label --
    quietly wrong. The rules say it cannot (GermStain has its own collision
    layer, appears on no rule's left-hand side, and is written only by the
    move rule), but "the rules say" is what this family gets caught by.

    So it is measured instead: walk to a random position on each level, snapshot
    the real grid (stains and all), and step both that grid and its stain-free
    rebuild with each of the four presses, comparing the resulting boards and
    ``check_win``.
    """
    import random

    _solver, game, expert = _levels()
    eng = game._engine
    rng = random.Random(20260814)
    field = _Field(eng, expert.bg_id, expert.germ_ids)
    bad = seen = stained = 0
    for level in range(game.n_levels):
        for _ in range(trials):
            game.set_level(level)
            for _ in range(rng.randrange(walk)):
                eng.step(rng.choice(_ORDER))
            snap = [[set(cell) for cell in row] for row in eng.grid]
            key = field.key()
            stained += sum(1 for row in eng.grid for cell in row
                           if cell & expert.germ_ids)
            for direction in _ORDER:
                eng.grid = [[set(cell) for cell in row] for row in snap]
                eng._position_index_dirty = True
                eng._rule_noop_cache.clear()
                eng.step(direction)
                with_stains = (field.key(), eng.check_win())
                field.seat(key)
                eng.step(direction)
                bad += with_stains != (field.key(), eng.check_win())
                seen += 1
    print(f"{seen} transitions stepped with and without the stains present "
          f"({stained} stains on the boards): "
          f"{'the stains are inert' if not bad else f'{bad} DISAGREEMENTS'}")
    return 0 if not bad else 1


def _verify() -> int:
    """Double-entry check of every plan length and every shipped tie set.

    `_Field` answers both from ONE enumeration: a forward sweep that builds the
    successor table and a reverse sweep over its predecessor map. A bug in the
    reverse sweep (or in the predecessor map it walks) would produce a
    self-consistent field, a plan that still wins, and tie sets that are quietly
    wrong -- and the training labels are exactly those tie sets, so "it wins" is
    not enough of a check.

    So each candidate press is priced again from the other end, by an
    independent forward BFS from its successor that stops at the first winning
    press: no reverse sweep, no predecessor map, no shared state but
    ``eng.step`` itself. A press is optimal iff that distance is
    ``d* - i - 1``, and that set must be exactly what the field shipped.

    Memoized on the successor key, because the plans revisit states and level
    11's sub-searches are ~20s each otherwise.
    """
    _solver, game, expert = _levels()
    bad = 0
    for level in range(game.n_levels):
        if level in ImSickTodaySolver.skip_levels:
            continue
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
        print(f"level {level:2d}: d*={star:3d}, "
              f"{sum(len(s) for s in optsets):3d} labelled presses re-priced by "
              f"{len(memo):4d} independent searches in {time.time() - t:5.1f}s: "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _symmetry() -> int:
    """Measure, on the interpreter, which transitions are decided by the way the
    board is FACING -- and whether the mandatory rotation already exposes them.

    This is the evidence behind putting the game in `_FLIP_GAMES`. Nothing in
    the rules names an absolute direction during play, but the interpreter
    drives each of a rule's four directional expansions to a fixpoint IN TURN
    (see [[ps-engine-render-gotchas]] #7), so which cell the chain-hop leaves
    the player on is a fact about the engine's expansion order rather than about
    the level. A presentation transform cannot change that -- it transforms the
    picture, not the engine -- so the question is not "is the game symmetric"
    (it is not) but "does the mirror expose an inconsistency the rotation does
    not already carry".

    So: every state of every level's complete reachable space, every press.
    Step the board; then step each of the 8 turned/mirrored copies of it with
    the correspondingly turned press and un-transform the result. A transition
    is CHIRAL when any copy disagrees, and it is MIRROR-ONLY when the three
    rotations all agree and a mirror does not. Zero mirror-only transitions is
    the claim; anything else would mean the flips add an ambiguity the corpus
    does not already have.
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
        for i, key in enumerate(field.keys):
            for a, direction in enumerate(_ORDER):
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
        print(f"level {level:2d}: {len(field.keys) * 4:6d} transitions, "
              f"{lvl_chiral:5d} chiral, {lvl_mirror:4d} split ONLY by a mirror "
              f"({time.time() - t:5.1f}s)")
    print(f"{total} transitions, {chiral} decided by the rule order, "
          f"{mirror_only} of them split only by a mirror")
    print("the flips add no inconsistency the rotation does not already carry"
          if not bad else
          f"FLIPS ARE NOT SAFE: {mirror_only} mirror-only transitions")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor).

    Two families of pairs are EXPECTED to be identical, and rather than listing
    them the check quotients the compositions by what they are ALLOWED to
    render the same as (`same_by_design`):

      * anything vs itself-with-a-germ-stain-under-it. GermStain is ``#010`` on
        a ``#001 #000`` Background, i.e. ARC 5 on ARC 5, so the trail is
        invisible on floor -- which is the right answer for a cosmetic object
        that no rule reads and that this file's state key drops. What the
        ``*_germ`` compositions are here to prove is that it is invisible UNDER
        every object too, so a stain can never disguise one piece as another.
      * ``person`` vs ``person2`` -- the same sprite and the same colours in the
        shipped art. Person2 exists only after the bed is occupied on level 12,
        i.e. only in that level's WINNING frame, where the win is read off the
        bed (which gains a green stripe) and off the player's disappearance.

    Everything else must be pairwise distinct at every cell size the levels
    render at (2, 4, 5, 6, 7 and 9 px), and is.
    """
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    bodies = ("wall", "wall_s", "infectedplayer", "curedplayer", "dead",
              "person", "person2", "doctor", "bed", "bedwithinfected")
    comps = {"floor": ()}
    comps.update({b: (b,) for b in bodies})
    # ... and each of them standing on a germ stain, which is where an
    # "invisible" trail would stop being invisible.
    comps["germ"] = ("germstain1",)
    comps.update({f"{b}_germ": ("germstain2", b) for b in bodies})

    def same_by_design(name: str) -> str:
        """The class a composition is allowed to be indistinguishable within:
        the germ stain is dropped, and Person2 is Person's twin."""
        base = "floor" if name == "germ" else name.removesuffix("_germ")
        return "person" if base == "person2" else base

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
                   if np.array_equal(shots[a], shots[b])
                   and same_by_design(a) != same_by_design(b)]
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
    if "--verify" in sys.argv:
        sys.exit(_stains() or _verify())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(ImSickTodaySolver.main())
