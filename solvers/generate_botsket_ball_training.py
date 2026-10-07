#!/usr/bin/env python3
"""Training-data generator for ``ps:botsket_ball`` (PuzzleScript "Botsket Ball").

THE GAME IS A LEVEL EDITOR, NOT A SOKOBAN
-----------------------------------------
Nothing on the board moves while you hold the controls. The player is a free
CURSOR (its own collision layer -- it walks over walls, holes, the ball, the
net, everything) and every keypress is an authoring action:

  * ACTION on an empty cell drops an ``abot``; the NEXT direction key turns it
    into a bot that will forever march that way (``[abot right player] ->
    [righter right player]``).
  * ACTION twice on an empty cell makes a ``box`` plus a ``drawing`` marker;
    while the drawing is live every step onto an empty cell extrudes another box
    CONNECTED to the last one. A third ACTION cancels the drawing, so a lone box
    is ACTION-ACTION-ACTION and a chain is ACTION-ACTION-<dirs>-ACTION. This
    generator only ever authors LONE boxes -- see `Machine`.
  * ACTION on the dial in the bottom panel grabs it; left/right then drags it
    along the track and ACTION drops it. The dial sets the SIMULATION SPEED.
  * ACTION on the green play button starts the simulation, which then runs to
    completion inside that single keypress (the adapter drains the whole
    ``again`` cascade). There is no second chance: the episode ends WIN or
    GAME_OVER on that one press.

So an episode is: think, build a machine, press play once.

THE MACHINE (what the simulation actually does)
-----------------------------------------------
A hidden marker sweeps from the dial to the ``st`` end of the track, one cell
per tick, and each time it lands on ``st`` the play state advances
LEFT -> UP -> RIGHT -> DOWN -> LEFT. On every tick where the marker sits on the
dial, the bots MATCHING THE CURRENT PHASE each step one cell. So:

  * one bot move per phase, four phases per cycle, always in the order L,U,R,D;
  * the dial-to-``st`` distance is the number of ticks per phase -- dragging the
    dial next to ``st`` makes it 1 tick, which is what this generator always
    does, because the adapter only affords ~170 ticks per play press and the
    default dial spends 3 of them per phase.

Movement is Sokoban-ish but with two twists that drive every solution here:

  * A bot is blocked (``cant``) only when the run of movables in front of it
    ends at a WALL. Nothing else stops a bot, and a bot never turns -- so a bot
    that has done its job keeps marching and will happily shove the ball back
    out of position, or walk into a hole.
  * A box shoved onto a hole falls in and FILLS it, which is the only way to
    bridge a gap. (The source also chains boxes into rigid bodies that fall
    together; the interpreter does not reproduce that, so nothing here uses it --
    see `Machine`.)

A bot or the ball entering a hole raises ``fail``, which is permanent and makes
the level unwinnable. Win = the ball is on the net at the end of any tick, and
the engine stops the cascade the moment that happens -- so the ball only has to
TRANSIT the net, it does not have to stop there.

THE EXPERT
----------
Because the whole episode is one authored configuration plus one keypress, the
search is over CONFIGURATIONS, not over action sequences. `Machine` is a fast
phase-level model of the simulation above (cross-checked against the real
interpreter, board state and all, by ``--fuzz N``); `solve_level` searches
configurations and `config_actions` compiles the winning one into the cursor
choreography that builds it.

The configuration search is route-driven rather than blind. A solution is the
ball's path decomposed into straight SEGMENTS, and each segment needs exactly
three things, all of which are enumerable:

  * a pusher -- a bot of the segment's direction placed ``m`` cells behind the
    ball. ``m`` is also the CLOCK: that bot first touches the ball on cycle
    ``m``, so placing it further back is how the machine waits for the ball to
    finish an earlier segment.
  * bridging boxes -- one per hole the segment crosses, parked between the ball
    and the first hole so the ball's own chain-push feeds them in one at a time.
  * a stop -- either a wall, or a run of boxes filling the rest of the ray to
    the wall (a box only blocks when the movables behind it back onto a wall).

Every candidate is then run through `Machine`, and the winner is replayed
against the real interpreter before it is ever recorded, so a plan that the
model got wrong is dropped rather than taped.

WHICH LEVELS SOLVE
------------------
Levels 0, 1 and 2 (a straight shot, a right-then-down corner, and the first
bridged gap). The other four are skipped, and NOT for want of search budget --
the route enumeration exhausts each of them in milliseconds. They are shut out
by the one structural fact of this game: a ball only ever moves in the direction
of a bot standing directly behind it, and a bot can never wait. So a bot's
distance behind the ball IS the cycle it fires on, and the board is only nine
cells wide:

  * L3 (ball top-left, net bottom-right of a 6x6 pit): the ball needs six pushes
    to reach the net's column, so its down-pusher would have to sit six rows
    above row 2 -- off the board. Every alternative column is a pit.
  * L4 (net one cell past a full-height gap): a bridging box has to be shoved
    into the gap, and the only cells it can be shoved from are the ball's own
    cell and the net. Fed from the far side by a lefter, the bridge lands one
    cycle before that same lefter reaches the ball, and the two bots then
    oscillate it between the same two cells forever.
  * L5 (net straight below the ball, whole top row a pit): pushing DOWN needs a
    bot in row 1, which is all holes -- unauthorable, and a bot walked up there
    is shoved straight back down on the next phase.
  * L6 (three-wide gap): bridging it takes three boxes fed in one at a time, and
    the ball's side of the gap is only two cells wide.

`discover_solvable` drops them once, for every seed, which is the corpus
contract (WIN-only).

NOT FLIP-AUGMENTED
------------------
Deliberately absent from `PuzzleScriptAdapter._FLIP_GAMES`, unlike most of the
gravity-free ps: games. The phase cycle LEFT -> UP -> RIGHT -> DOWN is CHIRAL:
the four rotations all present it as the same clockwise sweep entered at a
different point, but a mirror turns it counter-clockwise -- a machine that the
real game could never hand out, timed the opposite way while looking almost
identical. Rotation alone still gives every level its four presentations.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSExpert     # noqa: E402

# ---------------------------------------------------------------------------
# geometry / phase model
# ---------------------------------------------------------------------------

#: The play-state cycle, in the order the late ``[marker st][dial][playX]``
#: rules walk it. One bot move per entry, so this order is the whole clock.
PHASES = ("L", "U", "R", "D")

DELTA = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
DIRNAME = {"U": "up", "D": "down", "L": "left", "R": "right"}

WIN, FAIL, STUCK = "win", "fail", "stuck"

#: Ticks the adapter affords one play press: 49 inside ``step()``'s ``again``
#: loop (iteration 0 is the press itself, which creates ``playl`` only AFTER the
#: bot rules have run, so nothing moves on it) plus `perform_action`'s 120-tick
#: animation drain. With the dial dragged next to ``st`` a tick IS a phase, so
#: this is also the phase budget.
MAX_PHASES = 169
#: The adapter stops the drain after this many consecutive ticks in which the
#: non-UI grid did not change (`STABLE_TICKS_REQUIRED`). A machine that idles
#: longer than this is cut off mid-plan, so the model has to honour it.
IDLE_PHASES = 20


class LevelSpec:
    """The static half of a level, read straight off the engine grid."""

    def __init__(self, eng, game):
        idx = game.obj_name_to_idx
        h, w = eng.height, eng.width

        def cells_of(name):
            i = idx.get(name)
            if i is None:
                return set()
            return {(r, c) for r in range(h) for c in range(w) if i in eng.grid[r][c]}

        self.h, self.w = h, w
        self.walls = cells_of("wall")
        self.holes = frozenset(cells_of("hole"))
        self.nets = frozenset(cells_of("net"))
        self.balls = tuple(sorted(cells_of("ball")))
        self.player = next(iter(cells_of("player")))
        self.play = next(iter(cells_of("play")))
        self.dial = next(iter(cells_of("dial")))
        self.st = next(iter(cells_of("st")))
        self.track = frozenset(cells_of("track"))
        # Dragging the dial one cell past `st` is the fastest setting the track
        # allows (`[> movin | no track] -> [|]` drops the grab at the track end).
        self.fast_dial = (self.st[0], self.st[1] + 1)

    def inside(self, cell):
        return 0 <= cell[0] < self.h and 0 <= cell[1] < self.w

    def placeable(self, cell):
        """Can a fresh bot or box be authored here?

        Mirrors the two authoring rules, which agree on the exclusion list:
        ``[action player no hole no wall no box no play no bot no ball no net
        no abot]`` for a bot and ``no wall no bot no box no ball no net no hole``
        for a drawn box."""
        return (self.inside(cell) and cell not in self.walls
                and cell not in self.holes and cell not in self.nets
                and cell not in self.balls)


# ---------------------------------------------------------------------------
# the simulation model
# ---------------------------------------------------------------------------

class Machine:
    """Phase-level model of one play press.

    State is the movable layer (ball / boxes / bots) plus the set of still-open
    holes; walls and nets never change. One `move` is one PHASE, which is the
    only granularity that matters: the fall/fail rules resolve at the top of
    every tick and the bot rules fire below them, so a box shoved onto a hole is
    always gone before anything can move again, whatever the dial says.

    ONLY LONE BOXES. The game can also draw CHAINED boxes that move and fall as
    one rigid body, and this model deliberately does not have them: ``--fuzz``
    showed the interpreter does not reproduce the connect-drag
    (``[moving movable | < connect movable]``) -- it moves the box that was
    actually pushed and shears the rest of the shape off -- so any plan built on
    rigid chains would be a plan for a different game. Lone boxes agree with the
    interpreter on every trial, and `config_actions` authors them with the
    ACTION-ACTION-ACTION form that lays down no connects at all.
    """

    __slots__ = ("spec", "holes", "kind")

    def __init__(self, spec: LevelSpec, config: "Config"):
        self.spec = spec
        self.holes = set(spec.holes)
        self.kind: dict[tuple[int, int], str] = {}
        for cell in spec.balls:
            self.kind[cell] = "ball"
        for cell, d in config.bots:
            self.kind[cell] = d
        for cell in config.boxes:
            self.kind[cell] = "box"

    # -- one phase -----------------------------------------------------------
    def _cant(self, d) -> set:
        """Cells that cannot move in ``d`` this phase.

        Seeded with the walls (``[wall] -> [wall cant]``) and grown by
        ``dir [playX][movable | cant] -> [cant movable | cant]``: a movable
        backed onto a cant cell is itself cant. That is the whole reason a lone
        box only blocks anything when the run behind it reaches a wall.
        """
        dr, dc = DELTA[d]
        cant = set(self.spec.walls)
        pending = True
        while pending:
            pending = False
            for cell in self.kind:
                if cell in cant:
                    continue
                f = (cell[0] + dr, cell[1] + dc)
                if f in cant or not self.spec.inside(f):
                    cant.add(cell)
                    pending = True
        return cant

    def move(self, d) -> bool:
        """Run the ``d`` phase. Returns True if anything actually moved."""
        dr, dc = DELTA[d]
        cant = self._cant(d)
        moving = {cell for cell, k in self.kind.items()
                  if k == d and cell not in cant}
        if not moving:
            return False
        frontier = list(moving)
        while frontier:                                  # chain push
            cell = frontier.pop()
            f = (cell[0] + dr, cell[1] + dc)
            if f in self.kind and f not in moving:
                moving.add(f)
                frontier.append(f)
        self.kind = {((cell[0] + dr, cell[1] + dc) if cell in moving else cell): k
                     for cell, k in self.kind.items()}
        return True

    def resolve(self) -> tuple[bool, bool]:
        """Top-of-tick rules. Returns ``(failed, changed)``.

        ``[hole bot] / [hole ball] -> [hole fail]`` is permanent and kills the
        run; ``[box no wont hole] -> [fell]`` drops an unsupported box into the
        hole under it, filling it.
        """
        fallen = []
        for cell, k in self.kind.items():
            if cell in self.holes:
                if k != "box":
                    return True, False
                fallen.append(cell)
        for cell in fallen:
            del self.kind[cell]
            self.holes.discard(cell)
        return False, bool(fallen)

    def won(self) -> bool:
        nets = self.spec.nets
        return all(cell in nets for cell, k in self.kind.items() if k == "ball")

    def run(self, max_phases=MAX_PHASES, idle_limit=IDLE_PHASES):
        """Play the machine out. Returns WIN / FAIL / STUCK."""
        idle = 0
        for i in range(max_phases):
            failed, changed = self.resolve()
            if failed:
                return FAIL
            moved = self.move(PHASES[i & 3])
            if self.won():
                return WIN
            if moved or changed:
                idle = 0
            else:
                idle += 1
                if idle >= idle_limit:
                    return STUCK
        return STUCK


class Config:
    """A machine to author: bots (``(cell, direction)``) plus lone boxes."""

    __slots__ = ("bots", "boxes", "_cells")

    def __init__(self, bots=(), boxes=()):
        self.bots = tuple(sorted(bots))
        self.boxes = tuple(sorted(boxes))
        self._cells = frozenset([c for c, _ in self.bots] + list(self.boxes))

    @property
    def cells(self):
        return self._cells

    def merged(self, bots=(), boxes=()):
        return Config(self.bots + tuple(bots), self.boxes + tuple(boxes))

    def key(self):
        return (self.bots, self.boxes)

    def size(self):
        return len(self._cells)

    def __repr__(self):
        return f"Config(bots={list(self.bots)}, boxes={list(self.boxes)})"


def simulate(spec: LevelSpec, config: Config) -> str:
    return Machine(spec, config).run()


# ---------------------------------------------------------------------------
# configuration search
# ---------------------------------------------------------------------------

#: Longest run of boxes the search will build to stop the ball short of a wall.
MAX_BLOCKERS = 3
#: Segments the ball's route may have.
MAX_SEGMENTS = 3


def _ray(spec, pos, d):
    """The cells the ball would travel over, from ``pos`` up to the first wall."""
    dr, dc = DELTA[d]
    out, cell = [], pos
    while True:
        cell = (cell[0] + dr, cell[1] + dc)
        if not spec.inside(cell) or cell in spec.walls:
            return out
        out.append(cell)


def _pushers(spec, pos, d, taken):
    """``[(cell, m)]`` -- where a bot of direction ``d`` can be authored so that
    it first shoves the ball on cycle ``m``.

    A hole behind the ball ends the scan: the bot would march into it and raise
    ``fail`` before ever reaching the ball. A net (or an already-used cell) is
    only unauthorable, not fatal -- the bot walks over it -- so the scan skips
    past those.
    """
    dr, dc = DELTA[d]
    out, cell = [], pos
    for m in range(1, max(spec.h, spec.w)):
        cell = (cell[0] - dr, cell[1] - dc)
        if not spec.inside(cell) or cell in spec.walls or cell in spec.holes:
            break
        if spec.placeable(cell) and cell not in taken:
            out.append((cell, m))
    return out


def _bridge(spec, ray, upto, taken):
    """Boxes needed to carry the ball over the holes in ``ray[:upto+1]``.

    They ride in front of the ball and are fed into the gaps one per push, so
    the requirement is simply that as many free cells exist between the ball and
    the first hole. Returns the cells, or None when the segment is impassable.
    """
    holes = [i for i, c in enumerate(ray[:upto + 1]) if c in spec.holes]
    if not holes:
        return ()
    first = holes[0]
    n = len(holes)
    if first < n:
        return None                      # no room to carry enough boxes
    cells = tuple(ray[first - n:first])
    if any(not spec.placeable(c) or c in taken for c in cells):
        return None
    return cells


def _finishers(spec, pos, taken):
    """Yield the pieces of a last segment: one that carries the ball across a
    net cell (the win is checked every tick, so transit is enough)."""
    for d in PHASES:
        ray = _ray(spec, pos, d)
        target = next((i for i, c in enumerate(ray) if c in spec.nets), None)
        if target is None:
            continue
        bridge = _bridge(spec, ray, target, taken)
        if bridge is None:
            continue
        used = taken | set(bridge)
        for cell, _m in _pushers(spec, pos, d, used):
            yield [(cell, d)], bridge


def _stoppers(spec, pos, taken):
    """Yield ``(stop, bots, boxes)`` for an intermediate segment: one that parks
    the ball on a chosen cell so a later segment can pick it up.

    No arrival cycle is threaded through. Every pusher distance is enumerated
    anyway -- that is the whole point, since the distance IS the delay -- so
    reasoning about when the ball lands would only prune candidates the
    simulation is about to adjudicate exactly."""
    for d in PHASES:
        ray = _ray(spec, pos, d)
        for i, stop in enumerate(ray):
            blockers = tuple(ray[i + 1:])
            if len(blockers) > MAX_BLOCKERS:
                continue
            if any(not spec.placeable(c) or c in taken for c in blockers):
                continue
            bridge = _bridge(spec, ray, i, taken)
            if bridge is None:
                continue
            used = taken | set(blockers) | set(bridge)
            for cell, _m in _pushers(spec, pos, d, used):
                yield stop, [(cell, d)], bridge + blockers


def candidate_configs(spec: LevelSpec, max_segments=MAX_SEGMENTS):
    """Every route-shaped configuration, shortest routes first."""
    ball = spec.balls[0]

    def routes(pos, cfg, depth):
        for bots, boxes in _finishers(spec, pos, cfg.cells):
            yield cfg.merged(bots, boxes)
        if depth >= max_segments - 1:
            return
        for stop, bots, boxes in _stoppers(spec, pos, cfg.cells):
            yield from routes(stop, cfg.merged(bots, boxes), depth + 1)

    # Shortest routes first: `depth` starts high so the first pass yields only
    # one-segment machines, the next two-segment ones, and so on.
    for limit in range(1, max_segments + 1):
        yield from routes(ball, Config(), max_segments - limit)


def solve_level(spec: LevelSpec, *, budget=400_000) -> Config | None:
    """Search for a configuration whose simulation wins. Returns None if none of
    the routes inside the budget does."""
    if not spec.balls:
        return None
    seen, tried = set(), 0
    for cfg in candidate_configs(spec):
        k = cfg.key()
        if k in seen:
            continue
        seen.add(k)
        tried += 1
        if tried > budget:
            break
        if simulate(spec, cfg) == WIN:
            return cfg
    return None


# ---------------------------------------------------------------------------
# compiling a configuration into cursor choreography
# ---------------------------------------------------------------------------

def _walk(pos, target):
    """Directions for a monotone cursor walk, plus the tie set at each step.

    The cursor is unobstructed, so every interleaving of the two axes is an
    equally short route: the tie set is recorded as the step's ``optimal`` so
    the policy is not taught one arbitrary zig-zag."""
    steps = []
    r, c = pos
    while (r, c) != target:
        opts = []
        if r > target[0]:
            opts.append("U")
        elif r < target[0]:
            opts.append("D")
        if c > target[1]:
            opts.append("L")
        elif c < target[1]:
            opts.append("R")
        d = opts[0]
        steps.append((d, tuple(opts)))
        r, c = r + DELTA[d][0], c + DELTA[d][1]
    return steps


def config_actions(spec: LevelSpec, config: Config):
    """Compile a configuration into ``[(engine_action, optimal_set)]``.

    The order pieces are authored in does not matter -- every piece has its own
    cell and walking the cursor over a finished one does nothing. The invariant
    that does matter is that ACTION is only ever pressed on an EMPTY cell:
    pressing it on a box or a bot deletes that piece instead of adding one.
    """
    out = []
    pos = spec.player

    def walk_to(target):
        nonlocal pos
        for d, opts in _walk(pos, target):
            out.append((DIRNAME[d], tuple(DIRNAME[o] for o in opts)))
            pos = (pos[0] + DELTA[d][0], pos[1] + DELTA[d][1])

    def press(name):
        out.append((name, (name,)))

    # 1. Drag the speed dial next to `st`: one tick per phase, which is the only
    #    setting that fits a whole machine inside one play press.
    if spec.fast_dial != spec.dial:
        walk_to(spec.dial)
        press("action")
        for _ in range(spec.dial[1] - spec.fast_dial[1]):
            press("left")
            pos = (pos[0], pos[1] - 1)
        press("action")

    # 2. Author the boxes. The second ACTION lays the box down but leaves a live
    #    `drawing`, which would extrude a fresh connected box under the cursor on
    #    every following step; the third ACTION cancels it. That also keeps the
    #    boxes UNCHAINED, which is what `Machine` models -- see its docstring.
    for cell in config.boxes:
        walk_to(cell)
        press("action")
        press("action")
        press("action")

    # 3. Author the bots. The direction press both aims the bot and moves the
    #    cursor off the cell.
    for cell, d in config.bots:
        walk_to(cell)
        press("action")
        press(DIRNAME[d])
        pos = (cell[0] + DELTA[d][0], cell[1] + DELTA[d][1])

    # 4. Run it.
    walk_to(spec.play)
    press("action")
    return out


# ---------------------------------------------------------------------------
# expert / solver harness
# ---------------------------------------------------------------------------

def _snapshot(eng):
    return [[set(cell) for cell in row] for row in eng.grid]


def _restore(eng, snap):
    eng.grid = [[set(cell) for cell in row] for row in snap]
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


#: `perform_action`'s animation drain, reproduced so a plan is verified against
#: the budget the agent will actually get: at most this many continuation ticks,
#: abandoned early once the board (ignoring the marker and the play-state cycle)
#: has been still for `IDLE_PHASES` of them.
DRAIN_TICKS = 120


def _drain(eng, ui_indices) -> None:
    """Finish a play press exactly the way `PuzzleScriptAdapter.perform_action`
    does, so `verify` cannot pass on ticks the agent would never be given."""
    def board():
        return tuple(tuple(tuple(sorted(i for i in cell if i not in ui_indices))
                           for cell in row) for row in eng.grid)

    stable, streak = board(), 0
    for _ in range(DRAIN_TICKS):
        if not eng.step_animation_once():
            return
        now = board()
        if now == stable:
            streak += 1
            if streak >= IDLE_PHASES:
                return
        else:
            stable, streak = now, 0


def verify(eng, game, plan) -> bool:
    """Replay a plan on the interpreter itself and report whether it wins.

    The model is only a model; nothing is recorded until the real rules agree.
    Leaves the engine as it found it.
    """
    ui = {game.obj_name_to_idx[n] for n in ("playl", "playu", "playr", "playd",
                                            "marker") if n in game.obj_name_to_idx}
    snap = _snapshot(eng)
    try:
        for act, _opts in plan:
            eng.step(act)
            if eng.check_win():
                return True
            if act == "action":
                _drain(eng, ui)
                if eng.check_win():
                    return True
        return eng.check_win()
    finally:
        _restore(eng, snap)


class BotsketExpert(PSExpert):
    """Configuration-search expert. `plan` returns the whole episode: the cursor
    choreography that authors the machine plus the single press that runs it.

    There is no disk plan cache (the pattern the slower ps: generators use):
    the whole seven-level search is ~20ms, so caching it would only add a way
    for a stale entry to be replayed."""

    scope_by_level = True

    def setup(self):
        self.level_plans: dict[int, list | None] = {}
        self.optimal_sets: dict[int, list] = {}
        self._level = None

    def heuristic(self, eng):                      # unused: `_search` is replaced
        return 0

    def plan(self, eng, level=None):
        self._level = level
        return super().plan(eng, level)

    def _search(self, eng):
        level = self._level
        if level not in self.level_plans:
            spec = LevelSpec(eng, self.g)
            cfg = None if level in SKIP_LEVELS else solve_level(spec)
            steps = None
            if cfg is not None:
                steps = config_actions(spec, cfg)
                if not verify(eng, self.g, steps):
                    steps = None                    # the model was wrong: drop it
            self.level_plans[level] = steps
            if steps is not None:
                # `record_level` drives a flat direction list; the per-step tie
                # sets ride alongside (see `BotsketBallSolver.optimal_for`).
                self.optimal_sets[level] = [opts for _a, opts in steps]
        plan = self.level_plans[level]
        return None if plan is None else [a for a, _o in plan]


#: Levels no configuration in this vocabulary can win -- see the module
#: docstring. Skipped up front so startup does not burn the whole route budget
#: on them at every launch.
SKIP_LEVELS = frozenset({3, 5, 6})


class BotsketBallSolver(PSAStarSolver):
    game_id = "puzzlescript_botsket_ball"
    game_name = "Botsket_Ball"
    expert_cls = BotsketExpert
    skip_levels = SKIP_LEVELS

    #: Plans run 24-55 keypresses; the rest is headroom for the exploration
    #: prefix. The adapter itself calls GAME_OVER at 200 actions, so this cannot
    #: usefully go higher.
    max_steps = 150
    #: The play press animates the ENTIRE simulation inside one action. Keeping
    #: only its settled frame would record the machine being built and then, with
    #: nothing in between, a ball already in the net -- the one thing an agent has
    #: to learn about this game would be missing from every episode.
    record_spans = True

    def optimal_for(self, expert, level, plan, pi):
        """Walking the cursor is order-free, so most steps have a two-action
        optimal set; `config_actions` recorded it alongside the plan."""
        sets = getattr(expert, "optimal_sets", {}).get(level)
        if not sets or pi >= len(sets):
            return None
        return list(sets[pi])


# ---------------------------------------------------------------------------
# model-vs-interpreter fuzz (``--fuzz N``)
# ---------------------------------------------------------------------------

def _engine_outcome(eng, game, plan):
    """Author + run a machine on the real interpreter and read the result back.

    Returns ``(verdict, movables, holes)`` in the same vocabulary `Machine`
    uses, so the two can be compared cell for cell.
    """
    ui = {game.obj_name_to_idx[n] for n in ("playl", "playu", "playr", "playd",
                                            "marker") if n in game.obj_name_to_idx}
    for act, _opts in plan:
        eng.step(act)
        if act == "action" and not eng.check_win():
            _drain(eng, ui)
    idx = game.obj_name_to_idx
    names = {idx["ball"]: "ball", idx["box"]: "box", idx["uper"]: "U",
             idx["downer"]: "D", idx["lefter"]: "L", idx["righter"]: "R"}
    fail_id, hole_id = idx["fail"], idx["hole"]
    movables, holes, failed = {}, set(), False
    for r in range(eng.height):
        for c in range(eng.width):
            cell = eng.grid[r][c]
            if hole_id in cell:
                holes.add((r, c))
            if fail_id in cell:
                failed = True
            for i in cell:
                if i in names:
                    movables[(r, c)] = names[i]
    verdict = WIN if eng.check_win() else FAIL if failed else STUCK
    return verdict, movables, holes


def _random_config(spec, rng):
    """A random machine of the shape the search actually builds: a few bots and
    a few LONE boxes. Boxes are drawn adjacent to each other about half the time
    so the fuzz exercises multi-box chain pushes and blocking runs, which is
    where the interesting disagreements would be."""
    free = [c for c in ((r, c) for r in range(spec.h) for c in range(spec.w))
            if spec.placeable(c)]
    rng.shuffle(free)
    taken, bots, boxes = set(), [], []
    for _ in range(rng.randint(1, 3)):
        if not free:
            break
        cell = free.pop()
        taken.add(cell)
        bots.append((cell, rng.choice(PHASES)))
    for _ in range(rng.randint(0, 4)):
        neighbours = [(c[0] + d[0], c[1] + d[1]) for c in boxes
                      for d in DELTA.values()]
        neighbours = [c for c in neighbours
                      if spec.placeable(c) and c not in taken]
        if neighbours and rng.random() < 0.5:
            cell = rng.choice(neighbours)
            free.remove(cell)
        elif free:
            cell = free.pop()
        else:
            break
        taken.add(cell)
        boxes.append(cell)
    return Config(bots, boxes)


def fuzz(seed=0, trials=200, levels=None) -> int:
    """Cross-check `Machine` against the interpreter on random machines.

    The model is what the search believes; this is the only thing that makes
    that belief worth anything. Every trial authors a random configuration
    through the SAME `config_actions` compiler the generator uses, so a bug in
    either the physics model or the choreography shows up here.
    """
    import random
    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    game = PuzzleScriptAdapter("Botsket_Ball", seed=0)
    rng = random.Random(seed)
    bad = 0
    for level in (levels or range(game.n_levels)):
        mismatch = 0
        for t in range(trials):
            game.set_level(level)
            eng = game._engine
            spec = LevelSpec(eng, game._game)
            cfg = _random_config(spec, rng)
            model = Machine(spec, cfg)
            verdict = model.run()
            plan = config_actions(spec, cfg)
            snap = _snapshot(eng)
            try:
                got, movs, holes = _engine_outcome(eng, game._game, plan)
            finally:
                _restore(eng, snap)
            if got != verdict:
                mismatch += 1
                print(f"  L{level} trial {t}: verdict model={verdict} "
                      f"engine={got}  {cfg}")
                continue
            if verdict == FAIL:
                continue        # the engine has swapped the doomed piece for `fail`
            if movs != model.kind or holes != model.holes:
                mismatch += 1
                print(f"  L{level} trial {t}: state differs  {cfg}\n"
                      f"    model={sorted(model.kind.items())}\n"
                      f"    engine={sorted(movs.items())}")
        print(f"level {level}: {trials - mismatch}/{trials} agree")
        bad += mismatch
    print("FUZZ OK" if not bad else f"FUZZ FAILED ({bad} mismatches)")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--fuzz" in sys.argv:
        pos = sys.argv.index("--fuzz")
        n = int(sys.argv[pos + 1]) if len(sys.argv) > pos + 1 else 200
        sys.exit(fuzz(trials=n))
    sys.exit(BotsketBallSolver.main())
