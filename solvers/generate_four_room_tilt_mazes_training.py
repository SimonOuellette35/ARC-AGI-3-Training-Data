"""Generate Phase-1 training data for ps:four_room_tilt_mazes.

"Four-room tilt mazes" (Andrea Gilbert, clickmazes.com,
``data/puzzlescript_games/Four-room_tilt_mazes.txt``). A 2x2 red block TILTS: one
press slides it in a straight line until a wall stops it, and it wins by covering
the 2x2 blue target, which vanishes on contact (``No Target``). Three 32x32 mazes.

WHAT MAKES IT ITS OWN GAME: ``flickscreen 16x16``
-------------------------------------------------
Each level is FOUR 16x16 rooms and the camera shows only the one the block stands
in, so on two of the three levels the target is off screen at the start and the
walls of three quarters of the maze have never been rendered. A generator that
planned over the whole engine grid would emit a beeline to a target the frames do
not show -- a target no observation-conditioned learner can reproduce (the
badge_placement failure mode in its pure form). So the expert here is HONEST: it
reads one room per frame through `FourRoomExpert.observe`, keeps a map of the
rooms it has been shown, and plans over that map and nothing else.

THE EXPERT
----------
Two modes, chosen from the knowledge state, re-derived at every press:

* **exploit** -- the target has been seen and the mapped part of the maze already
  contains a route that clears it: take the shortest such route.
* **explore** -- otherwise, reach the cheapest press whose OUTCOME the map cannot
  predict (a slide that leaves the rooms seen so far) in as few presses as
  possible, and take it. Each such press ends in a room the camera then shows, so
  the map grows and the game is a finite search.

Both are exact shortest paths over a tiny graph (a few hundred states), so every
step gets the full set of equally-good presses, not one arbitrary choice
(`FourRoomExpert.decide`). The block is never optimal in the FULL maze it cannot
see -- that is the point; it is optimal in what it has been shown, which is what
the policy is being asked to learn. It costs remarkably little: 31 / 35 / 26
presses against the 31 / 33 / 25 a player who could see all four rooms would need
(``--plans``).

Those sets are all SINGLETONS, and that is the game rather than a broken tie
check: a slide ends where the walls put it, so no two presses from a state lead
anywhere near each other, and the full-knowledge distance field has not one state
with a second equally-short press on any of the three levels. Episode diversity
comes from the presentation (rotation x both flips = 16 per level) and the
exploration prefix, not from the route.

THE MODEL
---------
The interpreter runs at ~34 presses/s here (one press is a whole ``again`` loop
over a 32x32 grid), and planning over a PARTIAL map is not something the
interpreter can do at all, so the tilt is re-implemented: `slide`. It is exact --
``--selfcheck`` walks the interpreter over every state reachable in all three
levels (752 transitions) and compares position, targets and the win flag.

ONE ENGINE FIX WAS NEEDED. ``late [Player Target | Player Target] -> [Player |
Player]`` names the or-group ``Target`` in both cells, and the four target cells
hold four DIFFERENT members (T1..T4). `PSEngine._apply_late_rule_match` bound the
group once per rule and dropped any cell not holding that member, so only one
target of each pair was ever cleared, the last one could never be cleared at all,
and ``No Target`` was unreachable on all three levels. Fixed in
`adapters/puzzlescript_adapter.py` to fall back to the member THIS cell holds --
which is what the non-late path already did.

Usage (run from the repo root):
    python solvers/generate_four_room_tilt_mazes_training.py --episodes 200 \
        --out data/training_multi_level/ps:four_room_tilt_mazes
    python solvers/generate_four_room_tilt_mazes_training.py --selfcheck
    python solvers/generate_four_room_tilt_mazes_training.py --plans
    python solvers/generate_four_room_tilt_mazes_training.py --audit

``--audit`` is the anti-privilege test: it checks that every cell composition is
pixel-distinct at the cell size the levels use AND that un-presenting a frame and
reading its 16x16 cells reproduces the expert's room read exactly, on every
rotation and both flips (`frame_check`).
"""
from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                             # noqa: E402
from adapters.puzzlescript_adapter import (                    # noqa: E402
    PuzzleScriptAdapter, _render_cell_sprite, _render_frame)
from solvers.common.ps_astar import (                          # noqa: E402
    Plan, PSAStarSolver, PSExpert, restore, snapshot)

GAME_NAME = "Four-room_tilt_mazes"

#: Engine direction -> (dr, dc). The game has no ACTION rule, so ACTION5 is a
#: no-op and never enters the search or the exploration probe set.
DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
DIRECTIONS = ("up", "down", "left", "right")

_INF = 1 << 30


# ---------------------------------------------------------------------------
# The tilt, as a function of a wall map
# ---------------------------------------------------------------------------

def _front(pos: tuple[int, int], d: str) -> tuple[tuple[int, int], ...]:
    """The two cells directly ahead of the 2x2 block at ``pos`` (its top-left)."""
    r, c = pos
    dr, dc = DELTA[d]
    if dc:
        cc = c + 2 if dc > 0 else c - 1
        return ((r, cc), (r + 1, cc))
    rr = r + 2 if dr > 0 else r - 1
    return ((rr, c), (rr, c + 1))


def _clear(pos: tuple[int, int], targets: frozenset) -> frozenset:
    """Apply ``late [Player Target | Player Target] -> [Player | Player]`` for the
    block standing at ``pos``.

    The rule needs TWO orthogonally adjacent cells that each hold a player and a
    target, so a lone overlapping target cell survives -- and once the engine fix
    is in, both cells of every matching pair are cleared, in every direction, to a
    fixpoint. That is exactly "drop every overlapping target that has an
    overlapping orthogonal neighbour"."""
    r, c = pos
    over = {(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)} & targets
    if len(over) < 2:
        return targets
    dead = {p for p in over
            if any((p[0] + dr, p[1] + dc) in over for dr, dc in
                   ((1, 0), (-1, 0), (0, 1), (0, -1)))}
    return targets - dead if dead else targets


def slide(wall: dict, pos: tuple[int, int], targets: frozenset, d: str):
    """One press: tilt the 2x2 block until a wall stops it. Returns the settled
    ``(pos, targets)``, or None when the outcome runs off the KNOWN map.

    ``wall`` maps a cell to True/False and is allowed to be partial -- that is the
    whole point, it is the map of the rooms the camera has shown. A press whose
    slide needs a cell that is not in it is unpredictable, not blocked.

    Three details the interpreter forced (all covered by ``--selfcheck``):

    * a press into a wall is a ``cancel``, i.e. the whole turn is undone, so it is
      a plain no-op rather than a zero-length slide with late rules;
    * the targets are cleared at EVERY cell the block passes through, not only
      where it stops -- the ``again`` loop runs the late rules once per tick;
    * the adapter breaks its ``again`` loop the moment the win condition holds, so
      a slide that clears the last target STOPS THERE instead of running on to the
      wall. That shortcut is conditioned on there having BEEN a target to clear:
      an empty ``targets`` here means "none has been seen yet", not "the level is
      won", and reading it as a win stopped every slide after one cell.
    """
    dr, dc = DELTA[d]
    clearing = bool(targets)
    while True:
        ahead = _front(pos, d)
        if any(f not in wall for f in ahead):
            return None
        if any(wall[f] for f in ahead):
            return pos, targets
        pos = (pos[0] + dr, pos[1] + dc)
        targets = _clear(pos, targets)
        if clearing and not targets:
            return pos, targets


# ---------------------------------------------------------------------------
# The honest partial-observation expert
# ---------------------------------------------------------------------------

class FourRoomExpert(PSExpert):
    """Plans from the rooms the camera has SHOWN, one press at a time.

    `plan` returns a one-press `Plan` carrying that press's optimal SET, so
    `record_level` re-derives the decision from the live state at every step --
    which is what an adaptive policy is. There is no plan cache and no disk cache:
    the answer depends on what has been seen, not only on where the block is, so a
    memo keyed on the board would serve the wrong press.
    """

    directions = list(DIRECTIONS)

    def setup(self) -> None:
        self.wall_ids = frozenset(self.g.resolve_object_name("wall"))
        self.player_ids = frozenset(self.g.resolve_object_name("player"))
        self.target_ids = frozenset(self.g.resolve_object_name("target"))
        # No flickscreen would mean the whole level is one "room" -- the expert
        # then degenerates to a fully-informed shortest-path player, which is the
        # correct reading of "the frame shows everything".
        self.room_w, self.room_h = self.g.flickscreen or (1 << 30, 1 << 30)
        self._episode = None                  # the (seed, level) knowledge is for
        self._forget()

    # -- knowledge ----------------------------------------------------------
    def _forget(self) -> None:
        self.wall: dict[tuple[int, int], bool] = {}
        self.targets: frozenset = frozenset()
        self.pos: tuple[int, int] | None = None
        self.start = None                     # the state a RESET lands back on
        self.seen: dict = {}                  # (pos, targets, dir) -> state, played
        self._prev = None
        self._rng = random.Random(0)
        self.explored = 0                     # presses spent finding the target

    def observe(self, eng) -> None:
        """Fold ONE presented frame into the map. The only engine read there is.

        It takes exactly the flickscreen window `_render_frame` renders -- the
        room holding the block -- and reads only what that 64x64 frame shows: which
        of its cells are walls, where the block is, and which target cells are
        still there. Nothing outside the window is touched, which is what makes the
        expert reproducible by a viewer of the frames. (The decorative vBar / hBar /
        xBar grid lines are #f0f0f0 on white and vanish at cell_px=4, so they are
        not read either -- see ``--audit``.)"""
        pos, window, walls, targets = self._look(eng)
        self.wall.update(walls)
        sr, sc, er, ec = window
        # The room on screen is ground truth for its own targets; targets in rooms
        # that are off screen are remembered exactly as they were last seen (only
        # the block can remove one, and it can only do that on screen).
        self.targets = frozenset(
            t for t in self.targets if not (sr <= t[0] < er and sc <= t[1] < ec)
        ) | targets
        state = (pos, self.targets)
        if self.start is None:
            self.start = state
        elif self._prev is not None and state != self._prev and state != self.start:
            # Remember what a press actually did. The map alone cannot predict a
            # slide that left the rooms seen so far, and the press is identified
            # by the displacement it produced -- unambiguous here, because the
            # cells next to the block are always inside the room on screen, so an
            # unpredictable press always MOVES (never a silent no-op).
            d = self._displacement(self._prev[0], pos)
            if d is not None:
                self.seen[(self._prev[0], self._prev[1], d)] = state
        self.pos = pos
        self._prev = state

    def _look(self, eng):
        """(block pos, window, wall map of the window, targets in the window)."""
        cells = [(r, c) for r in range(eng.height) for c in range(eng.width)
                 if eng.grid[r][c] & self.player_ids]
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        sr = (r0 // self.room_h) * self.room_h
        sc = (c0 // self.room_w) * self.room_w
        er = min(sr + self.room_h, eng.height)
        ec = min(sc + self.room_w, eng.width)
        walls, targets = {}, set()
        for r in range(sr, er):
            row = eng.grid[r]
            for c in range(sc, ec):
                cell = row[c]
                walls[(r, c)] = bool(cell & self.wall_ids)
                if cell & self.target_ids:
                    targets.add((r, c))
        return (r0, c0), (sr, sc, er, ec), walls, frozenset(targets)

    @staticmethod
    def _displacement(old, new) -> str | None:
        """Which press turns ``old`` into ``new``, if a straight tilt can."""
        dr, dc = new[0] - old[0], new[1] - old[1]
        if dr and dc:
            return None
        for d, (er, ec) in DELTA.items():
            if (dr and er and dr * er > 0) or (dc and ec and dc * ec > 0):
                return d
        return None

    # -- planning over the map ----------------------------------------------
    def _succ(self, state, d):
        """The state ``d`` leads to, or None when the map cannot say."""
        key = (state[0], state[1], d)
        if key in self.seen:
            return self.seen[key]
        return slide(self.wall, state[0], state[1], d)

    def _graph(self, s0):
        """Every state reachable from ``s0`` through presses the map CAN predict,
        plus the states from which some press it cannot lies.

        A state with no targets left is absorbing ONLY when the search started
        with some -- with none seen yet an empty target set means "the target has
        not been found", not "the level is over", and treating the two alike
        stopped the exploration search dead at its own root."""
        absorb = bool(s0[1])
        edges: dict = {}
        blind: set = set()
        q = deque([s0])
        edges[s0] = {}
        while q:
            s = q.popleft()
            if absorb and not s[1]:
                continue                      # cleared: the level ends here
            for d in DIRECTIONS:
                ns = self._succ(s, d)
                if ns is None:
                    blind.add(s)
                    continue
                edges[s][d] = ns
                if ns not in edges:
                    edges[ns] = {}
                    q.append(ns)
        return edges, blind

    @staticmethod
    def _distances(edges, seeds, base: int) -> dict:
        """Presses-to-goal for every state, by BFS backwards from ``seeds``."""
        rev: dict = {}
        for s, outs in edges.items():
            for ns in outs.values():
                rev.setdefault(ns, set()).add(s)
        dist = {s: base for s in seeds}
        q = deque(seeds)
        while q:
            s = q.popleft()
            for p in rev.get(s, ()):
                if p not in dist:
                    dist[p] = dist[s] + 1
                    q.append(p)
        return dist

    def decide(self):
        """``(press, optimal set)`` from the current knowledge, or ``(None, None)``
        when the map offers neither a clear nor anything new to see."""
        s0 = (self.pos, self.targets)
        edges, blind = self._graph(s0)

        if s0[1]:                                    # a target has been seen
            dist = self._distances(edges, [s for s in edges if not s[1]], 0)
            here = dist.get(s0)
            if here == 0:
                return None, None                    # already cleared
            if here is not None:
                best = sorted(d for d, ns in edges[s0].items()
                              if dist.get(ns, _INF) == here - 1)
                return self._pick(best), best

        # No target seen yet, or none of the mapped maze reaches it: spend the
        # fewest presses to see something new. A press is "new" when the map
        # cannot predict where it ends, which after it is taken is exactly a room
        # the camera has not shown.
        dist = self._distances(edges, blind, 1)
        here = dist.get(s0)
        if here is None:
            return None, None                        # nothing left to try
        self.explored += 1
        if here == 1:
            best = sorted(d for d in DIRECTIONS if self._succ(s0, d) is None)
        else:
            best = sorted(d for d, ns in edges[s0].items()
                          if dist.get(ns, _INF) == here - 1)
        return self._pick(best), best

    def _pick(self, best: list) -> str:
        """One of the equally-good presses, drawn from a per-(seed, level) stream
        so runs stay byte-reproducible.

        Inert on the shipped mazes -- ``best`` is always a singleton there (see
        the module docstring) -- and kept because "take a random one of the
        equally-shortest presses" is the right rule the moment a maze has a real
        tie, and picking first-in-order would then train one arbitrary member of a
        set the label says are all correct."""
        return self._rng.choice(best)

    # -- PSExpert interface --------------------------------------------------
    def plan(self, eng, level: int | None = None):
        """One press, carrying its optimal set. `record_level` therefore re-plans
        every step, which is the point: the answer moves as the map grows."""
        episode = (getattr(self.game, "_seed", 0), level)
        if episode != self._episode:
            # A new (seed, level): forget the previous maze and re-read the frame
            # the harness has already presented (the reset that `observe` saw
            # before this call belongs to the level that just ended).
            self._episode = episode
            self._forget()
            self._rng = random.Random(f"tilt:{episode[0]}:{episode[1]}")
            self.observe(eng)
        d, best = self.decide()
        if d is None:
            return None
        return Plan([d], [best])

    def heuristic(self, eng) -> int:                 # pragma: no cover
        raise AssertionError("FourRoomExpert plans over its map, not by A*")


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class FourRoomTiltMazesSolver(PSAStarSolver):
    """`PSAStarSolver` wired to the honest tilt expert."""

    game_id = "ps:four_room_tilt_mazes"
    game_name = GAME_NAME
    expert_cls = FourRoomExpert

    #: All three levels are winnable, and the expert is adaptive, so the
    #: one-shot "can the expert win from the start" discovery pass would only be
    #: measuring its first press. Require the real thing: every level must WIN.
    require_all_levels = True
    #: The stock adapter budget. Finding the target and then clearing it costs
    #: 26-35 presses on these three mazes (``--plans``) and the exploration prefix
    #: at most ~25 more, so the game needs no raised cap.
    max_steps = 200

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]                          # ACTION5 fires no rule here


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def selfcheck(verbose: bool = True) -> int:
    """Walk the interpreter over EVERY state the three mazes can reach and check
    `slide` against it -- position, remaining targets and the win flag.

    Exhaustive rather than random: the whole reachable space is a few hundred
    states, so this is the strongest statement available (it is also what proves
    the game winnable at all, which it was not before the engine fix). Returns the
    number of divergences."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    parsed = game._game
    wall_ids = frozenset(parsed.resolve_object_name("wall"))
    player_ids = frozenset(parsed.resolve_object_name("player"))
    target_ids = frozenset(parsed.resolve_object_name("target"))
    eng = game._engine

    def read():
        cells = [(r, c) for r in range(eng.height) for c in range(eng.width)
                 if eng.grid[r][c] & player_ids]
        pos = (min(r for r, _ in cells), min(c for _, c in cells))
        targets = frozenset((r, c) for r in range(eng.height)
                            for c in range(eng.width)
                            if eng.grid[r][c] & target_ids)
        return pos, targets

    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        wall = {(r, c): bool(eng.grid[r][c] & wall_ids)
                for r in range(eng.height) for c in range(eng.width)}
        start = read()
        hidden = 0
        snaps = {start: snapshot(eng)}
        dist = {start: 0}
        q = deque([start])
        checked = 0
        while q:
            s = q.popleft()
            if not s[1]:
                continue
            for d in DIRECTIONS:
                predicted = slide(wall, s[0], s[1], d)
                restore(eng, snaps[s])
                eng.step(d)
                checked += 1
                actual = read()
                if actual != predicted:
                    bad += 1
                    print(f"  L{level}: {d} from {s[0]} -> model {predicted}, "
                          f"engine {actual}")
                if eng.check_win() != (not actual[1]):
                    bad += 1
                    print(f"  L{level}: {d} from {s[0]} -> win flag disagrees")
                if actual not in dist:
                    dist[actual] = dist[s] + 1
                    snaps[actual] = snapshot(eng)
                    q.append(actual)
                    # `FourRoomExpert.observe` reads the room's targets straight
                    # off the grid, which is the same thing the frame shows ONLY
                    # while no target sits UNDER the block (the player is drawn on
                    # top of it). The partial clear that would leave one there --
                    # an overlap of a single cell, with no orthogonal partner to
                    # match the late rule against -- turns out to be unreachable
                    # in all three mazes, and this is the assertion of that.
                    (r0, c0), tg = actual
                    if tg & {(r0, c0), (r0, c0 + 1),
                             (r0 + 1, c0), (r0 + 1, c0 + 1)}:
                        hidden += 1
        wins = [dist[s] for s in dist if not s[1]]
        if verbose:
            print(f"  L{level}: {len(dist)} states, {checked} transitions, "
                  f"shortest full-knowledge win "
                  f"{min(wins) if wins else 'UNREACHABLE'}"
                  + (f", {hidden} STATES HIDE A TARGET UNDER THE BLOCK"
                     if hidden else ""))
        bad += hidden
        if not wins:
            bad += 1
    return bad


def frame_check(seeds: int = 4, verbose: bool = True) -> int:
    """Prove the expert is not privileged: everything `FourRoomExpert.observe`
    takes off the grid is legible in the 64x64 frame the policy is handed.

    For every level and several seeds (so every rotation and both flips are
    covered) the presented frame is un-presented, cut into its 16x16 cells and
    classified by colour, and the result has to equal the expert's own room read --
    same walls, same target cells, same block. Random play walks the block from
    room to room so the check covers the camera moving, not only the start.
    Returns the number of divergences."""
    bad = 0
    for seed in range(seeds):
        game = PuzzleScriptAdapter(GAME_NAME, seed=seed)
        expert = FourRoomExpert(game)
        eng = game._engine
        objs = game._game.objects
        wall_colour = objs["bar"].dominant_color
        target_colours = {objs[f"t{i}"].dominant_color for i in (1, 2, 3, 4)}
        block_colours = {c for i in (1, 2, 3, 4) for c in objs[f"p{i}"].colors}
        assert objs["edge"].dominant_color == wall_colour
        assert not (target_colours & block_colours)
        for level in range(game.n_levels):
            game.set_level(level)
            rng = random.Random(f"frames:{seed}:{level}")
            for step in range(25):
                expert._forget()
                expert.observe(eng)
                frame = np.asarray(game._current_frame)
                if game._vflip:
                    frame = np.flipud(frame)
                if game._hflip:
                    frame = np.fliplr(frame)
                frame = np.rot90(frame, k=-game._rotation_k)
                px = 64 // expert.room_h
                sr = (expert.pos[0] // expert.room_h) * expert.room_h
                sc = (expert.pos[1] // expert.room_w) * expert.room_w
                seen_walls, seen_targets, seen_block = {}, set(), set()
                for r in range(expert.room_h):
                    for c in range(expert.room_w):
                        block = frame[r * px:(r + 1) * px, c * px:(c + 1) * px]
                        colours = set(block.ravel().tolist())
                        cell = (sr + r, sc + c)
                        seen_walls[cell] = (colours == {wall_colour})
                        if colours & target_colours:
                            seen_targets.add(cell)
                        if colours & block_colours:
                            seen_block.add(cell)
                block_cells = {(expert.pos[0] + dr, expert.pos[1] + dc)
                               for dr in (0, 1) for dc in (0, 1)}
                for label, shown, read in (("walls", seen_walls, expert.wall),
                                           ("targets", seen_targets,
                                            set(expert.targets)),
                                           ("block", seen_block, block_cells)):
                    if shown != read:
                        bad += 1
                        print(f"  seed {seed} L{level} step {step}: the frame's "
                              f"{label} differ from what the expert read")
                if eng.check_win():
                    break
                eng.step(rng.choice(DIRECTIONS))
                game._current_frame = game._present_frame(
                    _render_frame(eng, game._game))
    if verbose:
        print(f"  {seeds} seeds x 3 levels x 25 frames: "
              f"{'frame and expert agree everywhere' if not bad else 'DIVERGED'}")
    return bad


def plan_report() -> None:
    """Play every level with the honest expert and report what it costs -- how
    many presses went into FINDING the target versus clearing it, against the
    shortest win a player who could see the whole maze would take."""
    solver = FourRoomTiltMazesSolver()
    game, expert, _ = solver._ensure(0)
    eng = game._engine
    for level in range(game.n_levels):
        game.set_level(level)
        expert.observe(eng)
        presses = ties = 0
        while not eng.check_win() and presses < solver.max_steps:
            plan = expert.plan(eng, level)
            if plan is None:
                print(f"  L{level}: STUCK after {presses} presses")
                break
            ties += len(plan.optsets[0]) - 1
            eng.step(plan[0])
            expert.observe(eng)
            presses += 1
        rooms = len({(r // expert.room_h, c // expert.room_w)
                     for (r, c) in expert.wall})
        print(f"  L{level}: {presses:3d} presses  win={eng.check_win()}  "
              f"({expert.explored} spent exploring, {rooms} of the rooms mapped, "
              f"{ties} steps with a second right answer)")


#: One representative cell composition per thing the frame can show. The block is
#: 2x2 P1..P4 and the target 2x2 T1..T4, so each quadrant is its own case.
_AUDIT_CASES = {
    "floor": ["background", "cell"],
    "vbar": ["background", "vbar"],
    "hbar": ["background", "hbar"],
    "xbar": ["background", "xbar"],
    "edge": ["background", "edge"],
    "bar": ["background", "bar"],
    **{f"target{i}": ["background", "cell", f"t{i}"] for i in (1, 2, 3, 4)},
    **{f"block{i}": ["background", "cell", f"p{i}"] for i in (1, 2, 3, 4)},
}

#: Collisions that are the game, not a bug:
#: * the three grid-line objects are #f0f0f0 decoration on a white floor and are
#:   meant to be invisible -- they mark cell boundaries, they do not block;
#: * Bar and Edge are both "darkgrey wall" and the rules treat them as one
#:   (``Wall = Bar or Edge``).
#: The four target quadrants and the four block quadrants survive cell_px=4
#: distinctly (each paints its own corner of the cell), so the 2x2 glyphs the
#: frame shows are whole rather than smeared -- worth checking, since the
#: decorative grid lines in the same palette do NOT survive it.
_AUDIT_ALLOWED = {
    frozenset({"floor", "vbar"}), frozenset({"floor", "hbar"}),
    frozenset({"floor", "xbar"}), frozenset({"vbar", "hbar"}),
    frozenset({"vbar", "xbar"}), frozenset({"hbar", "xbar"}),
    frozenset({"edge", "bar"}),
}


def audit(verbose: bool = True) -> int:
    """Check every cell composition renders distinctly at the cell size the levels
    actually use, so nothing the expert reads off a room is invisible in the frame
    the policy gets. Returns the number of unexpected results."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    parsed = game._game
    layers = game._engine._obj_layers
    sizes = set()
    for level in range(game.n_levels):
        game.set_level(level)
        h, w = len(game._engine.grid), len(game._engine.grid[0])
        if parsed.flickscreen:
            w, h = min(w, parsed.flickscreen[0]), min(h, parsed.flickscreen[1])
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
        for a, b in itertools.combinations(sorted(_AUDIT_CASES), 2):
            same = bool((blocks[a] == blocks[b]).all())
            if frozenset({a, b}) in _AUDIT_ALLOWED:
                if not same:
                    bad += 1
                    print(f"  cell_px={px}: {a} and {b} are now distinct -- "
                          f"drop them from _AUDIT_ALLOWED")
            elif same:
                bad += 1
                print(f"  cell_px={px}: {a} and {b} render identically")
        if verbose:
            print(f"  cell_px={px}: {len(_AUDIT_CASES)} cell types, "
                  f"{len(_AUDIT_ALLOWED)} allowed collisions, "
                  f"{'all distinct' if not bad else 'COLLISIONS'}")
    return bad


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit() + frame_check()
        print(f"audit: {collisions} unexpected collisions")
        sys.exit(1 if collisions else 0)
    if "--plans" in sys.argv:
        plan_report()
        sys.exit(0)
    sys.exit(FourRoomTiltMazesSolver.main())
