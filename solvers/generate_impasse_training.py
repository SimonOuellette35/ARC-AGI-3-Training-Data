"""Generate Phase-1 training data for the PuzzleScript game ps:impasse
("Impasse", RatoLibre1, a demake of Wanderlands' Flash game).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: the exact distance-to-win FIELD over each level's whole
reachable state space (built by stepping the real interpreter, so there is no
model to fuzz), the optimal-action oracle it hands out for free, and the sprite
work the shipped file needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_impasse",
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
A 10x3 corridor (with a solid Border row above and below), the player on the
left, the Goal on the right, and a field of hazards that MOVE WHEN YOU DO. The
win is ``All Player on Goal``; there are four presses and no ACTION (the game
declares ``noaction``). Five rules carry the whole thing.

  * **Your own move wraps vertically.** ``late [Border Mover | ... | | Border]
    -> [Border | ... | Mover | Border]`` teleports anything that stepped into a
    Border row to the row just inside the far Border. Every Mover -- the player
    included -- wraps, so the three playable rows are a cycle and UP from the
    top row is DOWN to the bottom row. Horizontally there is no Border, so a
    press into the left or right map edge is REFUSED.
  * **Half the hazards move on your VERTICAL presses.**
    ``[vertical Player][Upper] -> [vertical Player][up Upper]`` (and the
    mirrored ``Lower``/``down``). Orange pieces, one cell per vertical press,
    wrapping through the Border rule like everything else.
  * **The other half move on your HORIZONTAL presses.**
    ``[horizontal Player][PerpUp] -> [horizontal Player][up PerpUp]`` (and
    ``PerpDown``/``down``). Yellow pieces: the trigger axis is PERPENDICULAR to
    the direction they travel, which is the whole reason the game is a puzzle
    rather than a dodge -- you cannot advance toward the goal without winding
    the yellow pieces on, and you cannot wind the orange pieces on without
    leaving the row you wanted to be in.
  * **A REFUSED press still ticks the world.** The mover rules match on the
    player's FORCE, not on the player having moved, so pressing into the left
    or right map edge advances every yellow piece and moves nothing else. That
    is the game's only wait move, it exists on exactly two columns, and several
    levels are unwinnable without it. (The opposite convention is just as
    common in this family -- see [[goblin-hooblob-solver]] -- so it is worth
    measuring in any new ps: game.)
  * **The two toggling hazards.** ``[vertical Player][Vanisher1] ->
    [vertical Player][VanisherTemp]`` plus ``[vertical Player][Vanisher2] ->
    [vertical Player][Vanisher1]`` and the ``late [VanisherTemp] -> [Vanisher2]``
    that follows: every VERTICAL press flips every vanisher between its armed
    (``Vanisher1``, lethal) and spent (``Vanisher2``, harmless) phase, in
    lockstep across the board. The Temp object exists only to stop the two
    rules ping-ponging within one turn and never survives a press. Blockers are
    the same toggle driven by a different clock: ``late [Player Switch]
    [Blocker1] -> [Player Switch][BlockerTemp]`` flips every blocker on any
    turn the player ENDS ON a Switch, whichever direction it pressed.
  * **Death is a RESTART, not a loss.** ``late [Player Obstacle] -> restart``,
    where ``Obstacle`` is everything lethal: the fixed walls, all four movers,
    and the ARMED vanishers/blockers. Two things about it decide how this file
    is written -- see below.

Two details of the interpreter that the field has to get right:

  * the vanisher/blocker toggles are declared BEFORE the death rule, so a press
    is judged against the board AFTER the flip. Stepping onto an armed vanisher
    with a vertical press is safe (it disarms on the way in) and stepping onto a
    spent one is fatal (it arms). The rules read the other way round at a
    glance;
  * ``restart`` is handled by the ADAPTER, not by ``PSEngine.step`` -- the
    engine only raises ``_rule_restart`` and leaves the player standing on the
    obstacle, and `PuzzleScriptAdapter.perform_action` is what reloads the
    level. A field built on ``eng.step`` alone would therefore walk straight
    through every wall in the game. Here a restart edge reloads the level and
    is looked up like any other successor, so it lands on the level's start
    state (``--plans`` reports the count of edges that did not, which is 0),
    and the search learns for free that dying is never a shortcut: the start is
    at distance ``d*``, so a press that restarts is never on a shortest path.

The levels
----------
All 23 shipped levels are 10x3 (plus the two Border rows), all 23 are winnable
and all 23 are recorded. ``--plans`` prints this table and CERTIFIES every plan
by replaying it through the interpreter:

    level  hazards                       states  plan  ties
    0-2    fixed walls only                15-25  12-14  ~30%
    3-8    + orange (vertical-clock)       30-63  15-20  ~25%
    9-14   + vanishers                    37-69  14-30  ~20%
    15-20  + yellow (horizontal-clock)   131-273  17-30  ~25%
    21-22  + blockers and switches       243-692  15     ~30%

Plans are proved SHORTEST rather than merely found: ``states`` is the size of
the level's ENTIRE reachable state space, which the field enumerates
exhaustively, so the distance it reports is the true one. The longest plan is 30
presses against the adapter's 200-step per-level budget, so this game needs no
`games/` step-limit wrapper.

Expert solver
-------------
There is no native model here and nothing to fuzz: the interpreter IS the model.
Every level's reachable space is tiny (692 states at the worst, ~11s for all
twenty-three), because the board is only 30 cells and the hazards are not free
-- the orange pieces are a function of the vertical press count mod their cycle,
the yellow ones of the horizontal count, and the two toggles of one bit each. So
`_Field` enumerates outright --

  1. a forward BFS from the level start over ``eng.step``, recording the
     successor of every state under each of the four presses and marking the
     presses that WIN (winning is a transition, not a state: ``check_win`` is
     read after a press, and a won level is terminal, so those edges lead
     nowhere);
  2. a reverse BFS over the predecessor map from the states that win in one
     press, which labels every state that can reach a win at all with its EXACT
     distance;

and the plan is the walk downhill from the start. States are stored as their
canonical cell sets and the grid is REBUILT from the set to re-expand a state,
which is exact because every parsed cell carries Background (asserted at build
time) and no object in this game is cosmetic or hidden.

This is the [[im-sick-today-solver]] shape and it is chosen for the same reason:
measure the reachable space before writing a model or a heuristic. An A* here
would need a heuristic that prices "arrive in the right column on the right
parity of two independent clocks, with the vanisher bit set the right way", and
the honest distance to the goal is FLAT over most of a level -- you spend whole
stretches marking time against a wall to wind the yellow pieces round. Nothing
cheap is admissible, and the space is small enough that the question never has
to be asked.

Optimal-action sets
-------------------
The field gives these exactly and for free: at a state ``s`` with distance ``d``,
a press is optimal iff it wins (when ``d == 1``) or reaches a state at distance
``d - 1``. A press the engine refuses outright maps ``s`` to itself, whose
distance is ``d``, so no-ops are excluded by construction -- and note that in
this game a refused press is usually NOT a no-op (it winds the yellow pieces),
which is exactly why it has to be measured rather than assumed. No step ships
unlabelled (the always-emit-optimal-targets rule), and a step with a unique
optimal press ships a one-element set. ``--verify`` re-derives every one of them
from the other end -- an independent bounded BFS from each candidate successor
-- so the labels do not rest on the reverse pass alone.

Rendering (the shipped art was unreadable three ways)
-----------------------------------------------------
See the comment block at the top of ``data/puzzlescript_games/Impasse.txt``;
``--audit`` is the check. In short: Goal and Player were transparent at exactly
the same four corners and nowhere else, so the frame the whole game is ABOUT --
the player standing on the goal -- was pixel-identical to the player standing on
bare floor; Goal and Switch were both ARC 14 on sprites differing in two pixels;
and every piece was an outlined box, so nothing could be read through anything.
Now every BODY (player + four movers) is a hollow ring with a transparent 3x3
middle and every GROUND tile is a solid square, so a stack reads out of the
middle. ``--audit`` renders every cell COMPOSITION that occurs anywhere in the
23 reachable spaces -- 34 of them -- as a whole 64x64 frame and asserts they are
pairwise distinct.

Augmentation
------------
This game's engine state after reset is identical for every seed (levels are
fixed ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to). 23 levels x
16 presentations = 368.

The ESCAPE! argument does NOT apply here -- this game is full of absolute
directions (``up Upper``, ``down Lower``), and the interpreter is not equivariant
under a transform that swaps up and down, because it has no object that moves
left or right at all. What makes the augmentation sound instead is that the
absolute direction is drawn ON THE SPRITE, as an orbit of the full 8-element
symmetry group: a mover paints a black bar on its leading edge and paints the
two corners on that side and no others, so every rotation and every flip maps
direction art onto direction art exactly as it maps the motion. The two facts a
policy needs are then transform-invariant statements about the picture:

  * a piece travels the way its own bar points, one cell per triggering press;
  * ORANGE is triggered by a press along the axis it travels, YELLOW by a press
    along the perpendicular axis.

Deliberately one colour per family rather than the shipped four: "orange means
up" is true in exactly one of the four rotations, so a per-direction colour would
teach a rule that is false three times out of four (the
[[directioban-solver]] rule -- do not recolour a family apart when its SHAPE is
the mechanic).

The flips are then measured rather than argued: ``--symmetry`` rebuilds each
level's LAYOUT under each of the four transforms that preserve the up/down axis
(identity, mirror, flip, half-turn), RELABELLING Upper<->Lower and
PerpUp<->PerpDown as the transform demands, reloads it through the interpreter
so the level-start rules re-run, replays the level's own plan with the presses
transformed, and requires the whole board to land where the transform says. The
90-degree rotations cannot be tested this way -- the game owns no left/right
mover to relabel them onto -- but they are mandatory anyway, and the sprite
orbit above is what makes them consistent. There is no colour augmentation.

The expert plan is therefore seed-independent: solved once per level, cached to
``data/impasse_plans.json``, and replayed per seed with that seed's remapped
screen actions.

Recovery
--------
``recovery_mode = "reset"`` (the family default). The episode-wide epsilon prefix
explores freely -- and in this game a wrong press frequently RESTARTS the level
outright, which is worth having in the corpus -- and ONE RESET restores the level
start, from which the cached plan replays a guaranteed win.

Usage (run from the repo root):
    python solvers/generate_impasse_training.py \
        --episodes 200 --out data/training_multi_level/impasse

    python solvers/generate_impasse_training.py --plans
    python solvers/generate_impasse_training.py --verify
    python solvers/generate_impasse_training.py --symmetry
    python solvers/generate_impasse_training.py --audit
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

GAME_NAME = "Impasse"

#: Where each level's start plan is cached between processes. The fields are
#: seed-independent, so without this every `parallelize_generator` shard would
#: re-enumerate all twenty-three state spaces (~11s) before recording a frame.
PLAN_CACHE = Path(__file__).resolve().parent.parent / "data" / "impasse_plans.json"

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
    background cell -- i.e. the whole board minus Background, which every cell
    carries unconditionally. Nothing in this game is cosmetic or hidden (the two
    ``*Temp`` objects never survive the press that creates them, which
    ``--plans`` asserts), so that set is canonical: the grid is REBUILT from it
    to re-expand a state.

    ``reload`` reloads the level onto the same engine. It is what makes a
    ``restart`` edge correct: ``PSEngine.step`` only RAISES ``_rule_restart``
    and leaves the player standing on the obstacle it died to, and it is
    `PuzzleScriptAdapter.perform_action` that reloads. Without this the field
    would happily plan straight through the walls.
    """

    def __init__(self, eng, bg: int, reload, cap: int = 200_000):
        self.eng = eng
        self.bg = bg
        self.reload = reload
        self.cap = cap
        self.keys: list[frozenset] = []
        self.succ: list[list[int]] = []
        self.dist: dict[int, int] = {}
        self.overflow = False
        #: restart edges seen, and how many of them did NOT land on the level
        #: start (0 -- reported rather than assumed, see the header).
        self.restarts = 0
        self.restarts_off_start = 0

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
                if eng._rule_restart:
                    # What the adapter does with a death: reload the level. The
                    # resulting key is then looked up like any other successor,
                    # so nothing here ASSUMES it is the start state.
                    self.restarts += 1
                    self.reload()
                    k = self.key()
                    self.restarts_off_start += (k != start)
                elif eng.check_win():
                    row.append(_WIN)
                    continue
                else:
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

class ImpasseExpert(PSExpert):
    """Plans by enumerating the level's whole reachable space with `_Field` and
    walking the exact distance-to-win field downhill.

    `PSExpert` still owns everything around that -- the in-memory memo, the
    on-disk plan cache with its staleness check and the snapshot/restore
    discipline -- so the only overrides are `_search` and the two lines of
    `plan` that remember which level's layout a ``restart`` has to reload.
    `heuristic` is unreachable by construction: nothing here runs A*.
    """

    directions = list(_ORDER)
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        self._level: int | None = None

    def heuristic(self, eng) -> int:                          # pragma: no cover
        raise AssertionError("the field is exact; no search runs here")

    def plan(self, eng, level: int | None = None) -> list | None:
        self._level = level
        return super().plan(eng, level)

    def field(self, eng, level: int | None = None) -> _Field:
        """The level's field. ``level`` defaults to the one `plan` was last
        asked for, and to the adapter's current level when it is called
        directly (the reports do that)."""
        if level is None:
            level = self._level
        if level is None:
            level = self.game._current_level_index
        layout = self.g.levels[level]
        return _Field(eng, self.bg_id, lambda: eng.load_level(layout)).build()

    def _search(self, eng) -> list | None:
        presses, optsets = self.field(eng).plan()
        return None if presses is None else Plan(presses, optsets)


class ImpasseSolver(PSAStarSolver):
    game_id = "puzzlescript_impasse"
    game_name = GAME_NAME
    game_module_id = "ps:impasse"
    expert_cls = ImpasseExpert

    #: The longest plan is 30 presses; the rest is room for the exploration
    #: prefix and the re-plan after it, and it stays well under the adapter's
    #: 200-step per-level budget.
    max_steps = 100

    def prepare_expert(self, game, expert) -> None:
        """Build every level's field before `discover_solvable` asks for it --
        the same work either way, but it makes the one-time ~11s visible as
        startup rather than as a mysteriously slow first seed, and it fills the
        disk cache in one pass."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with the adapter parked at level 0."""
    solver = ImpasseSolver()
    game = solver.make_game(0)
    return solver, game, ImpasseExpert(game)


def _hazards(eng, gm) -> str:
    """A one-word census of what a level throws at the player."""
    names = {gm.obj_idx_to_name.get(o, "")
             for row in eng.grid for cell in row for o in cell}
    tags = []
    if names & {"upper", "lower"}:
        tags.append("orange")
    if names & {"perpup", "perpdown"}:
        tags.append("yellow")
    if names & {"vanisher1", "vanisher2"}:
        tags.append("vanish")
    if names & {"blocker1", "blocker2", "switch"}:
        tags.append("block")
    return "+".join(tags) if tags else "walls"


def _report() -> int:
    """Per-level board size, hazards, plan length, tie coverage and state count
    -- and CERTIFY each plan by replaying it through the interpreter."""
    _solver, game, expert = _levels()
    eng, gm = game._engine, game._game
    temps = {gm.obj_name_to_idx[n] for n in ("vanishertemp", "blockertemp")}
    total = bad = states = off = 0
    for level in range(game.n_levels):
        game.set_level(level)
        t = time.time()
        field = expert.field(eng, level)
        presses, optsets = field.plan()
        took = time.time() - t
        states += len(field.keys)
        off += field.restarts_off_start
        # The two bookkeeping objects must never survive a press -- the state
        # key would otherwise carry a mid-turn board that no frame can show.
        alive = sum(1 for k in field.keys for (_r, _c, o) in k if o in temps)
        head = (f"level {level:2d}: {eng.height}x{eng.width} "
                f"{_hazards(eng, gm):20s} {len(field.keys):4d} states "
                f"({field.restarts:4d} restart edges, {field.restarts_off_start} "
                f"off-start) in {took:4.1f}s")
        bad += alive
        if presses is None:
            why = ("GAVE UP AT THE STATE CAP" if field.overflow
                   else "NO WIN IS REACHABLE")
            print(f"{head}: {why}")
            bad += 1
            continue
        # Certification: the interpreter must win on the LAST press and no
        # earlier (an earlier win would mean the plan is not shortest), and no
        # press of the plan may restart the level.
        game.set_level(level)
        won_at = died = None
        for i, direction in enumerate(presses):
            eng.step(direction)
            if eng._rule_restart:
                died = i
                break
            if eng.check_win():
                won_at = i
                break
        ok = won_at == len(presses) - 1 and died is None
        bad += not ok
        total += len(presses)
        ties = sum(1 for s in optsets if len(s) > 1)
        room = "ok" if len(presses) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(presses):3d} presses "
              f"(budget {game._max_steps}, {room}), "
              f"{ties:3d} tie steps ({ties / len(presses):3.0%}), "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}"
              f"{'' if not alive else f' -- {alive} *Temp CELLS SURVIVED A PRESS'}")
    print(f"total {total} presses over {game.n_levels} levels, "
          f"{states} states enumerated, {off} restart edges off the level start")
    return 0 if not bad else 1


def _verify() -> int:
    """Double-entry check of every plan length and every shipped tie set.

    `_Field` answers both from ONE enumeration: a forward sweep that builds the
    successor table and a reverse sweep over its predecessor map. A bug in the
    reverse sweep (or in the predecessor map it walks) would produce a
    self-consistent field, a plan that still wins, and tie sets that are quietly
    wrong -- and the training labels ARE those tie sets, so "it wins" is not
    enough of a check.

    So each candidate press is priced again from the other end, by an
    independent forward BFS from its successor that stops at the first winning
    press: no reverse sweep, no predecessor map, nothing shared but ``eng.step``
    and the same restart handling. A press is optimal iff that distance is
    ``d* - i - 1``, and that set must be exactly what the field shipped.

    Memoized on the successor key, because the plans revisit states.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        layout = game._game.levels[level]
        field = expert.field(eng, level)
        presses, optsets = field.plan()
        if presses is None:
            print(f"level {level:2d}: NO PLAN")
            bad += 1
            continue

        t = time.time()
        memo: dict[frozenset, int | None] = {}

        def step_from(k: frozenset, direction: str):
            """``(key, won)`` after ``direction`` from ``k``, restart included."""
            field.seat(k)
            eng.step(direction)
            if eng._rule_restart:
                eng.load_level(layout)
                return field.key(), False
            return field.key(), eng.check_win()

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
                    nxt, won = step_from(k, direction)
                    if won:
                        answer = d + 1
                        break
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
        if distance(field.keys[0]) != star:
            notes.append(f"LENGTH {distance(field.keys[0])} != {star}")
        bad += len(notes)
        print(f"level {level:2d}: d*={star:3d}, "
              f"{sum(len(s) for s in optsets):3d} labelled presses re-priced by "
              f"{len(memo):4d} independent searches in {time.time() - t:4.1f}s: "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# The presentation group, measured
# ---------------------------------------------------------------------------

#: What a transform that swaps UP and DOWN has to do to the pieces whose
#: movement direction is baked into their identity. There is deliberately no
#: entry for a 90-degree rotation: this game owns no left/right mover to relabel
#: onto, which is exactly why `_symmetry` tests the mirror/flip subgroup only
#: and the rotations rest on the sprite orbit instead (see the header).
_VFLIP_SWAP = (("upper", "lower"), ("perpup", "perpdown"))


def _symmetry() -> int:
    """Replay every level's plan on the four turned/mirrored copies of its own
    LAYOUT and require the whole board to land where the transform says.

    This is the evidence behind putting Impasse in `_FLIP_GAMES`, and it is the
    [[icecrates-solver]] check with the one addition this game forces: the
    transform is applied to the level LAYOUT (so the interpreter re-runs the
    level-start rules and re-derives the board itself, rather than being handed
    a grid that no ``load_level`` would produce), and the direction-carrying
    objects are RELABELLED as it goes -- a flip that swaps up and down turns
    every Upper into a Lower. Without the relabel the test measures nothing but
    the fact that ``up Upper`` is an absolute direction, which is already known.

    A clean run says the mechanic is invariant under the mirror/flip subgroup
    once the pieces are renamed, and the sprites are drawn so that the RENAME IS
    WHAT THE RENDER DOES -- a flipped Upper is pixel-identical to a Lower. So
    the flips add no ambiguity: every presentation shows a board the game could
    have shipped.
    """
    _solver, game, expert = _levels()
    eng, gm = game._engine, game._game
    idx = gm.obj_name_to_idx
    swap = {}
    for a, b in _VFLIP_SWAP:
        swap[idx[a]] = idx[b]
        swap[idx[b]] = idx[a]

    def turn_layout(layout, mirror: bool, flip: bool):
        rows = [list(row) for row in layout]
        if mirror:
            rows = [row[::-1] for row in rows]
        if flip:
            rows = rows[::-1]
            rows = [[{swap.get(o, o) for o in cell} for cell in row]
                    for row in rows]
        return [[set(cell) for cell in row] for row in rows]

    def turn_key(cells, mirror: bool, flip: bool, h: int, w: int):
        out = set()
        for (r, c, o) in cells:
            rr = (h - 1 - r) if flip else r
            cc = (w - 1 - c) if mirror else c
            out.add((rr, cc, swap.get(o, o) if flip else o))
        return frozenset(out)

    _MIRROR = {"left": "right", "right": "left", "up": "up", "down": "down"}
    _FLIP = {"up": "down", "down": "up", "left": "left", "right": "right"}

    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        layout = gm.levels[level]
        h, w = eng.height, eng.width
        field = expert.field(eng, level)
        presses, _optsets = field.plan()
        notes = []
        for mirror, flip in itertools.product((False, True), repeat=2):
            eng.load_level(turn_layout(layout, mirror, flip))
            steps = 0
            for direction in presses:
                d = direction
                if mirror:
                    d = _MIRROR[d]
                if flip:
                    d = _FLIP[d]
                eng.step(d)
                steps += 1
                if eng._rule_restart or eng.check_win():
                    break
            got = field.key()
            eng.load_level(layout)
            for direction in presses:
                eng.step(direction)
                if eng._rule_restart or eng.check_win():
                    break
            want = turn_key(field.key(), mirror, flip, h, w)
            if got != want or steps != len(presses) or not eng.check_win():
                notes.append(f"mirror={mirror} flip={flip}")
        bad += len(notes)
        print(f"level {level:2d}: {len(presses):3d} presses replayed on 4 "
              f"presentations: {'EQUIVARIANT' if not notes else 'SPLIT BY ' + ', '.join(notes)}")
    print("the mirror/flip subgroup is a symmetry of the mechanic once the "
          "direction-carrying pieces are relabelled -- which is exactly what "
          "the sprites render" if not bad else
          f"FLIPS ARE NOT SAFE: {bad} presentations disagree")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Rendering audit
# ---------------------------------------------------------------------------

def _compositions(game, expert) -> list[tuple[str, ...]]:
    """Every cell COMPOSITION that occurs anywhere in the 23 reachable spaces,
    as sorted object-name tuples (the empty tuple is bare floor).

    Enumerated rather than listed: a hand-written list is a guess about which
    stacks the game can build, and this game stacks a mover on a wall, on a
    switch, on the goal, on either phase of either toggle, and on ANOTHER mover.
    """
    eng, gm = game._engine, game._game
    seen = set()
    for level in range(game.n_levels):
        game.set_level(level)
        field = expert.field(eng, level)
        for key in field.keys:
            cells: dict[tuple[int, int], set[int]] = {}
            for (r, c, o) in key:
                cells.setdefault((r, c), set()).add(o)
            for objs in cells.values():
                seen.add(tuple(sorted(gm.obj_idx_to_name[o] for o in objs)))
        seen.add(())
    return sorted(seen)


def _audit() -> int:
    """Assert every cell COMPOSITION the game can show is distinct in the frame.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor).

    Every level of this game is 10x3 inside its borders, so there is one cell
    size to check (6 px) -- asserted rather than assumed.
    """
    _solver, game, expert = _levels()
    eng, gm = game._engine, game._game
    idx = gm.obj_name_to_idx

    sizes = set()
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.add((eng.height, eng.width))
    assert len(sizes) == 1, f"more than one board size: {sizes}"
    (h, w), = sizes

    comps = _compositions(game, expert)
    shots = {}
    for objs in comps:
        eng.height, eng.width = h, w
        eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                     for _ in range(w)] for _ in range(h)]
        eng._position_index_dirty = True
        shots[objs] = np.asarray(_render_frame(eng, gm)).copy()

    clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
               if np.array_equal(shots[a], shots[b])]
    name = lambda t: "+".join(t) if t else "floor"            # noqa: E731
    print(f"{h}x{w} boards, cell {64 // max(h, w)}px, "
          f"{len(comps)} compositions reachable across {game.n_levels} levels")
    for objs in comps:
        painted = int((shots[objs] != shots[()]).sum())
        print(f"  {name(objs):28s} {painted:4d} px differ from bare floor")
    if clashes:
        for a, b in clashes:
            print(f"  IDENTICAL: {name(a)} == {name(b)}")
    print("audit clean" if not clashes
          else f"AUDIT FAILED: {len(clashes)} indistinguishable pairs")
    return 0 if not clashes else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(ImpasseSolver.main())
