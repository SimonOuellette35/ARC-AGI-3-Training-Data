"""Shared engine for the ``pb01`` / ``pb02`` / ``pb03`` push-block family.

The three games are the same Sokoban-style mechanic with one knob each turned:

  * **pb01** -- one crate, one goal pad.
  * **pb02** -- two crates, two goal pads (all must be covered).
  * **pb03** -- one crate, one goal pad, plus a *decoy* pad that is an instant
    loss if the crate is pushed onto it.

Their three source files were byte-for-byte identical apart from those knobs (the
level tables, the win test and pb03's decoy branch), so the mechanic, the HUD, the
per-seed augmentation and the search model live here once and each game file is
reduced to its level specs plus its knob. See ``games/pb01/pb01.py``.

Two things this module owns that the original hand-written trio did not have:

**Per-seed augmentation.** The stock games shipped five hand-authored, fully
deterministic boards, so an entire training corpus would have been fifteen fixed
positions -- a memorisation target, not a distribution. `PushPuzzleGame` instead
draws, deterministically from ``(seed, level_index)``, the wall skeleton (a barrier
line with N gaps / scattered blocks, per the level's `LevelSpec`), the player /
crate / pad / decoy cells, and an episode-scoped colour palette. Every candidate
layout is accepted only if `PushModel` proves it solvable (and, where possible,
non-trivial), so difficulty is preserved while the boards vary. The presentation
rotation comes from `AugmentedGame` as usual.

Levels are built in ``__init__`` (the seed is known there), NOT in
``on_set_level``: the engine's ``_clean_levels`` snapshot is taken from what
``__init__`` passes up, so RESET / ``level_reset`` restore *this seed's* board, and
re-seating a level is idempotent for free.

**`PushModel`** -- the pure (x, y) dynamics + an A* planner over
``(player, crates)`` states. The game uses it as its solvability oracle and the
generator (`solvers/common/push_blocks.py`) uses it as its expert, so there is one
statement of "what a push does" for board generation and planning. The *authority*
on the mechanic is still ``PushPuzzleGame.step`` (it moves the sprites); the model
is verified against the engine by ``solvers/common/push_blocks.py``'s self-check.
"""
from __future__ import annotations

import heapq
import itertools
import random
from dataclasses import dataclass

from arcengine import Camera, Level, RenderableUserDisplay, Sprite

from utils.arc_game import AugmentedGame

# Camera letterbox. Never sampled as a piece colour, so the board can never blend
# into the padding around a non-square grid.
PADDING_COLOR = 4

#: action id -> (dx, dy). Matches ``utils.rotation``'s convention, which is what
#: makes the rotation remap correct: ACTION1 up, ACTION2 down, ACTION3 left,
#: ACTION4 right.
DELTAS: dict[int, tuple[int, int]] = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}
ACTION_IDS: tuple[int, ...] = (1, 2, 3, 4)


# ---------------------------------------------------------------------------
# The search model: pure geometry + push dynamics + an A* expert
# ---------------------------------------------------------------------------
class PushModel:
    """Push dynamics and an optimal planner for ONE board geometry.

    A *state* is ``(player, crates)`` with ``player = (x, y)`` and ``crates`` a
    sorted tuple of ``(x, y)``. The geometry (grid, walls, pads, decoys) is fixed
    for the level, so one model instance serves every state of that level and can
    cache plans across them -- which is what makes `optimal_actions` affordable
    (it needs a plan for each successor, and the successor actually taken already
    has its plan cached as a suffix of the current one).
    """

    def __init__(self, width: int, height: int, walls, targets, decoys=()) -> None:
        self.width = int(width)
        self.height = int(height)
        self.walls = frozenset(walls)
        self.targets = frozenset(targets)
        self.decoys = frozenset(decoys)
        self._plans: dict[tuple, tuple] = {}      # state -> optimal action tuple
        self.dead_cells = self._deadlocks()

    # ── geometry ─────────────────────────────────────────────────────────────
    def open(self, cell) -> bool:
        """Is ``cell`` on the board and not a wall? (Pads and decoys are floor.)"""
        x, y = cell
        return 0 <= x < self.width and 0 <= y < self.height and cell not in self.walls

    def _deadlocks(self) -> frozenset:
        """Non-pad cells from which a crate can provably never reach a pad. Two
        sound, static rules -- both used to prune the search AND to keep the
        board sampler from proposing hopeless layouts:

        * **corner** -- blocked on one horizontal AND one vertical side (by a wall
          or the board edge), so the crate cannot be pushed at all: pushing needs
          the player on the far side of the crate and the cell beyond it free.
        * **edge lane** -- a crate against the board edge can only ever be pushed
          ALONG that edge (pushing it off the edge line would need the player
          standing outside the board), so it is confined to that lane forever; if
          the lane holds no pad, it is dead wherever it started on it.
        """
        dead = set()
        cols = {x for x, _ in self.targets}
        rows = {y for _, y in self.targets}
        for y in range(self.height):
            for x in range(self.width):
                cell = (x, y)
                if not self.open(cell) or cell in self.targets:
                    continue
                horiz = not self.open((x - 1, y)) or not self.open((x + 1, y))
                vert = not self.open((x, y - 1)) or not self.open((x, y + 1))
                if horiz and vert:
                    dead.add(cell)
                elif ((x in (0, self.width - 1) and x not in cols)
                      or (y in (0, self.height - 1) and y not in rows)):
                    dead.add(cell)
        return frozenset(dead)

    # ── dynamics ─────────────────────────────────────────────────────────────
    def step(self, state, action_id: int):
        """One action. Returns ``(next_state, outcome)`` where outcome is
        ``"blocked"`` (nothing moved -- the engine does not even count the step),
        ``"moved"``, ``"push"``, or ``"dead"`` (pb03: the crate would land on a
        decoy pad, which loses the game -- so the state is NOT advanced)."""
        player, crates = state
        dx, dy = DELTAS[action_id]
        ahead = (player[0] + dx, player[1] + dy)
        if not self.open(ahead):
            return state, "blocked"
        if ahead in crates:
            behind = (ahead[0] + dx, ahead[1] + dy)
            if not self.open(behind) or behind in crates:
                return state, "blocked"
            if behind in self.decoys:
                return state, "dead"
            # Re-sort the WHOLE set: crates are an unordered set, and a state key
            # that depends on push order would give the search two names for one
            # state (defeating its dedup and its plan cache).
            moved = tuple(sorted([c for c in crates if c != ahead] + [behind]))
            return (ahead, moved), "push"
        return (ahead, crates), "moved"

    def is_goal(self, state) -> bool:
        return all(c in self.targets for c in state[1])

    def _stuck(self, crates) -> bool:
        return any(c in self.dead_cells for c in crates)

    def _heuristic(self, crates) -> int:
        """Lower bound on moves: the cheapest crate->pad assignment by Manhattan
        distance. Each crate needs at least that many pushes and a push costs at
        least one move, so it never over-estimates."""
        best = None
        for pads in itertools.permutations(sorted(self.targets), len(crates)):
            total = sum(abs(c[0] - p[0]) + abs(c[1] - p[1])
                        for c, p in zip(crates, pads))
            if best is None or total < best:
                best = total
        return best or 0

    # ── planning ─────────────────────────────────────────────────────────────
    def plan(self, state, limit: int | None = None) -> list[int]:
        """A shortest winning action sequence from ``state``, or ``[]`` if the
        state cannot be won (already-won states also give ``[]`` -- callers test
        `is_goal` first). Deterministic: ties break on the fixed action order, so
        the same state always yields the same canonical route.

        Every state on the returned path is cached with ITS optimal suffix, so a
        walk down the plan costs one search for the whole level.

        ``limit`` caps the number of expansions and returns ``[]`` when it is hit,
        WITHOUT caching -- so it means "no cheap plan", not "unsolvable". Board
        sampling uses it (proving a random two-crate board unsolvable costs ~100x
        more than solving a good one, and the sampler only wants to move on);
        planning for real never does."""
        cached = self._plans.get(state)
        if cached is not None:
            return list(cached)
        if self.is_goal(state):
            self._plans[state] = ()
            return []
        if self._stuck(state[1]):
            self._plans[state] = ()
            return []

        counter = itertools.count()
        start_h = self._heuristic(state[1])
        heap = [(start_h, 0, next(counter), state)]
        best_g = {state: 0}
        came: dict[tuple, tuple] = {}            # state -> (prev_state, action_id)
        goal_state = None
        expansions = 0
        while heap:
            _f, g, _c, cur = heapq.heappop(heap)
            if g > best_g.get(cur, g):
                continue                          # stale heap entry
            expansions += 1
            if limit is not None and expansions > limit:
                return []                         # give up WITHOUT caching
            if self.is_goal(cur):
                goal_state = cur
                break
            for action_id in ACTION_IDS:
                nxt, outcome = self.step(cur, action_id)
                if outcome in ("blocked", "dead"):
                    continue
                if outcome == "push" and self._stuck(nxt[1]):
                    continue
                ng = g + 1
                if ng >= best_g.get(nxt, 1 << 30):
                    continue
                best_g[nxt] = ng
                came[nxt] = (cur, action_id)
                heapq.heappush(heap, (ng + self._heuristic(nxt[1]), ng,
                                      next(counter), nxt))
        if goal_state is None:
            self._plans[state] = ()               # provably unsolvable from here
            return []

        # Walk the parent chain back, caching each state's own optimal suffix.
        rev: list[tuple] = []                     # [(state, action_id), ...]
        cur = goal_state
        while cur != state:
            prev, action_id = came[cur]
            rev.append((prev, action_id))
            cur = prev
        rev.reverse()
        suffix: tuple = ()
        for st, action_id in reversed(rev):
            suffix = (action_id,) + suffix
            self._plans[st] = suffix
        return list(self._plans[state])

    def optimal_actions(self, state) -> list[int]:
        """Every action that keeps the win exactly one move closer -- the optimal
        tie SET, which the generator records as the policy target and samples the
        taken action from. Costs at most three extra searches per step (the fourth
        successor is the one already cached by `plan`)."""
        plan = self.plan(state)
        if not plan:
            return []
        best = len(plan)
        out = []
        for action_id in ACTION_IDS:
            nxt, outcome = self.step(state, action_id)
            if outcome in ("blocked", "dead"):
                continue
            if self.is_goal(nxt):
                if best == 1:
                    out.append(action_id)
                continue
            sub = self.plan(nxt)
            if sub and len(sub) == best - 1:
                out.append(action_id)
        return out or plan[:1]


# ---------------------------------------------------------------------------
# Presentation
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Palette:
    """One episode's colours. Sampled mutually distinct so no piece can hide
    against another or against the background."""

    background: int
    player: int
    crate: int
    crate_done: int
    target: int
    wall: int
    fail: int
    decoy: int = 0

    #: Order the colours are drawn in. Kept explicit so the sampled palette is a
    #: pure function of the seed regardless of dataclass field order.
    ROLES = ("background", "player", "crate", "crate_done", "target", "wall",
             "fail", "decoy")

    @classmethod
    def sample(cls, rng: random.Random) -> "Palette":
        pool = [c for c in range(1, 16) if c != PADDING_COLOR]
        drawn = rng.sample(pool, len(cls.ROLES))
        return cls(**dict(zip(cls.ROLES, drawn)))


class PushBadge(RenderableUserDisplay):
    """Bottom-right 4x4 status tile: crate colour while pads are uncovered, the
    covered-crate colour once every pad is covered, and the fail colour when a
    crate has been pushed onto a decoy. pb03 additionally paints a bottom-left
    3x3 tile in the decoy colour -- the on-screen cue for which pad is the trap.
    """

    def __init__(self, palette: Palette, show_decoy_cue: bool = False) -> None:
        self._palette = palette
        self._show_decoy_cue = show_decoy_cue
        self._done = False
        self._decoy_fail = False

    def update(self, done: bool, *, decoy_fail: bool | None = None) -> None:
        self._done = done
        if decoy_fail is not None:
            self._decoy_fail = decoy_fail

    def render_interface(self, frame):
        import numpy as np

        if not isinstance(frame, np.ndarray):
            return frame
        h, w = frame.shape
        if self._decoy_fail:
            color = self._palette.fail
        elif self._done:
            color = self._palette.crate_done
        else:
            color = self._palette.crate
        frame[h - 4:h, w - 4:w] = color
        if self._show_decoy_cue:
            frame[h - 4:h - 1, 0:3] = self._palette.decoy
        return frame


# ---------------------------------------------------------------------------
# Level specification + randomised construction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LevelSpec:
    """The *shape* of one level -- what stays constant across seeds. Positions,
    walls within the pattern, and colours are all drawn per (seed, level)."""

    grid: tuple[int, int]
    difficulty: int
    crates: int = 1
    #: Number of openings in a full-width/height wall line (0 = no barrier line).
    barrier_gaps: int = 0
    #: Extra single-cell wall blocks scattered over the floor.
    scatter: int = 0
    #: Length of a wall stub growing perpendicular off the barrier line.
    stub: int = 0
    decoy: bool = False


def _sprite(color: int, name: str, tags: list[str], collidable: bool) -> Sprite:
    return Sprite(pixels=[[color]], name=name, visible=True,
                  collidable=collidable, tags=tags)


def _random_walls(spec: LevelSpec, rng: random.Random) -> set:
    """A wall skeleton matching ``spec``: an optional barrier line (random axis,
    random offset, ``barrier_gaps`` random openings), an optional perpendicular
    stub off it, and ``scatter`` loose blocks."""
    w, h = spec.grid
    walls: set = set()
    if spec.barrier_gaps:
        vertical = rng.random() < 0.5
        span, offs = (h, w) if vertical else (w, h)
        line = rng.randrange(2, offs - 2)         # keep >=2 columns/rows each side
        gaps = set(rng.sample(range(span), min(spec.barrier_gaps, span)))
        for i in range(span):
            if i in gaps:
                continue
            walls.add((line, i) if vertical else (i, line))
        if spec.stub:
            side = rng.choice((-1, 1))
            at = rng.randrange(span)
            for step in range(1, spec.stub + 1):
                pos = line + side * step
                if not 0 <= pos < offs:
                    break
                walls.add((pos, at) if vertical else (at, pos))
    for _ in range(spec.scatter):
        free = [(x, y) for x in range(w) for y in range(h) if (x, y) not in walls]
        if not free:
            break
        walls.add(rng.choice(free))
    return walls


#: Expansion cap for the sampler's solvability probe (see `PushModel.plan`).
#: Comfortably above what any accepted board needs (the hardest observed level
#: settles in a few thousand), so it only ever truncates hopeless candidates.
_SAMPLE_SEARCH_LIMIT = 20_000


def _random_layout(spec: LevelSpec, rng: random.Random, attempts: int = 240):
    """Draw a solvable layout for ``spec``.

    Rejection sampling against `PushModel`: a candidate is kept only if the model
    finds a winning plan (with crates never pushable onto a decoy). The first 3/4
    of the attempts additionally demand a plan long enough to match the level's
    difficulty, so an easy accident on a hard level is re-rolled; after that any
    solvable board is taken rather than failing. Returns
    ``(walls, player, crates, targets, decoy, plan_len)`` or ``None``.
    """
    w, h = spec.grid
    want_len = 4 + 2 * spec.difficulty
    for attempt in range(attempts):
        walls = _random_walls(spec, rng)
        floor = [(x, y) for x in range(w) for y in range(h) if (x, y) not in walls]
        need = 1 + 2 * spec.crates + (1 if spec.decoy else 0)
        if len(floor) < need:
            continue
        targets = rng.sample(floor, spec.crates)
        decoy = None
        if spec.decoy:
            # The decoy is a LURE: keep it near the real pad, as in the stock
            # boards, so telling them apart is a colour judgement rather than a
            # distance one.
            near = [c for c in floor
                    if c not in targets
                    and 0 < abs(c[0] - targets[0][0]) + abs(c[1] - targets[0][1]) <= 2]
            if not near:
                continue
            decoy = rng.choice(near)
        taken = set(targets) | ({decoy} if decoy else set())
        rest = [c for c in floor if c not in taken]
        if len(rest) < spec.crates + 1:
            continue
        model = PushModel(w, h, walls, targets, [decoy] if decoy else ())
        # Crates go only where a crate can still be won FROM (`dead_cells` is a
        # static, sound test), which is what keeps the sampler off the hopeless
        # boards that dominate a uniform draw. The player is unconstrained.
        placeable = [c for c in rest if c not in model.dead_cells]
        if len(placeable) < spec.crates:
            continue
        crates = tuple(sorted(rng.sample(placeable, spec.crates)))
        free_for_player = [c for c in rest if c not in crates]
        if not free_for_player:
            continue
        player = rng.choice(free_for_player)
        state = (player, crates)
        plan = model.plan(state, limit=_SAMPLE_SEARCH_LIMIT)
        if not plan:
            continue
        if len(plan) < want_len and attempt < attempts * 3 // 4:
            continue
        return walls, player, crates, tuple(sorted(targets)), decoy, len(plan)
    return None


class PushPuzzleGame(AugmentedGame):
    """The shared push-block game. A subclass supplies ``game_name``, ``specs``
    and its step-budget curve; everything else (mechanic, HUD, augmentation) is
    here.

    ``seed=None`` keeps interactive play fresh: colours, boards and rotation are
    all drawn from the global RNG, so every launch is a new game. With a seed,
    every one of them is a pure function of ``(seed, level_index)`` -- which is
    what lets a recorded episode and a later live run of the same seed line up.
    """

    game_name: str = ""
    specs: tuple[LevelSpec, ...] = ()
    #: step budget = base + per_difficulty * spec.difficulty
    step_budget_base: int = 60
    step_budget_per_difficulty: int = 15

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)             # FIRST: the Camera holds its display
        color_rng = random.Random(f"{self.game_name}:colors:{seed}"
                                  if seed is not None else None)
        self.palette = Palette.sample(color_rng)
        self._show_decoy_cue = any(s.decoy for s in self.specs)
        self._ui = PushBadge(self.palette, self._show_decoy_cue)
        levels = [self._build_level(i, spec, seed)
                  for i, spec in enumerate(self.specs)]
        super().__init__(
            self.game_name,
            levels,
            Camera(0, 0, 16, 16, self.palette.background, PADDING_COLOR,
                   [self._ui]),
            False,
            1,
            [1, 2, 3, 4],
        )

    # ── construction ─────────────────────────────────────────────────────────
    def _build_level(self, index: int, spec: LevelSpec, seed: int | None) -> Level:
        rng = random.Random(f"{self.game_name}:layout:{seed}:{index}"
                            if seed is not None else None)
        layout = _random_layout(spec, rng)
        if layout is None:                        # exhausted the sampler: retry
            layout = _random_layout(spec, random.Random(f"fallback:{seed}:{index}"))
        if layout is None:
            raise RuntimeError(
                f"{self.game_name}: no solvable layout for level {index} ({spec})")
        walls, player, crates, targets, decoy, plan_len = layout
        p = self.palette
        sprites = [_sprite(p.player, "player", ["player"], True)
                   .set_position(*player)]
        for cell in crates:
            sprites.append(_sprite(p.crate, "block", ["block"], True)
                           .set_position(*cell))
        for cell in targets:
            sprites.append(_sprite(p.target, "target", ["target"], False)
                           .set_position(*cell))
        if decoy is not None:
            sprites.append(_sprite(p.decoy, "decoy", ["decoy"], False)
                           .set_position(*decoy))
        for cell in sorted(walls):
            sprites.append(_sprite(p.wall, "wall", ["wall"], True)
                           .set_position(*cell))
        budget = (self.step_budget_base
                  + self.step_budget_per_difficulty * spec.difficulty)
        return Level(
            sprites=sprites,
            grid_size=spec.grid,
            data={"difficulty": spec.difficulty, "step_limit": budget,
                  "optimal_steps": plan_len},
        )

    # NOTE: sprite ORDER matters and is set above -- ``Level.get_sprite_at``
    # returns the first sprite at a cell in insertion order (within one layer),
    # so a crate standing on a pad must be inserted before the pads or the push
    # test below would see the pad and let a second crate through it.

    def on_set_level(self, level: Level) -> None:
        # Re-read from ``current_level``: a RESET re-clones the level, so the
        # sprite objects (and the ``level`` argument) are replaced wholesale.
        current = self.current_level
        self._player = current.get_sprites_by_tag("player")[0]
        self._crates = current.get_sprites_by_tag("block")
        self._targets = current.get_sprites_by_tag("target")
        self._decoys = current.get_sprites_by_tag("decoy")
        self._step_limit = current.get_data("step_limit")
        self._steps = 0
        self._ui.update(False, decoy_fail=False)
        self._sync_ui()

    # ── state ────────────────────────────────────────────────────────────────
    @property
    def _steps_remaining(self) -> int:
        """Steps left in the budget. Named for ``BaseSolver._step_counter_probe``,
        which uses it to suppress *step-exhaustion* losses during data generation
        (exploration overhead must not discard an otherwise-won trajectory) while
        leaving pb03's decoy loss intact."""
        return max(0, self._step_limit - self._steps)

    def _on_target(self, crate) -> bool:
        return any(crate.x == t.x and crate.y == t.y for t in self._targets)

    def _all_placed(self) -> bool:
        return all(self._on_target(c) for c in self._crates)

    def _sync_ui(self) -> None:
        for crate in self._crates:
            on = self._on_target(crate)
            if on and "done" not in crate.tags:
                crate.color_remap(self.palette.crate, self.palette.crate_done)
                crate.tags.append("done")
            elif not on and "done" in crate.tags:
                crate.color_remap(self.palette.crate_done, self.palette.crate)
                crate.tags.remove("done")
        self._ui.update(self._all_placed())

    # ── mechanic ─────────────────────────────────────────────────────────────
    def step(self) -> None:
        # Screen space -> game space: the frame is presented at this level's
        # rotation, so a directional press means whatever that rotation makes it.
        action = self.screen_action_to_game(self.action.id)
        delta = DELTAS.get(int(action.value))
        if delta is None:
            self.complete_action()
            return
        dx, dy = delta

        grid_w, grid_h = self.current_level.grid_size
        new_x, new_y = self._player.x + dx, self._player.y + dy
        if not (0 <= new_x < grid_w and 0 <= new_y < grid_h):
            self.complete_action()
            return

        sprite = self.current_level.get_sprite_at(new_x, new_y,
                                                 ignore_collidable=True)
        if sprite and "wall" in sprite.tags:
            self.complete_action()
            return

        if sprite and "block" in sprite.tags:
            crate_x, crate_y = new_x + dx, new_y + dy
            if not (0 <= crate_x < grid_w and 0 <= crate_y < grid_h):
                self.complete_action()
                return
            behind = self.current_level.get_sprite_at(crate_x, crate_y,
                                                      ignore_collidable=True)
            if behind and ("block" in behind.tags or "wall" in behind.tags):
                self.complete_action()
                return
            if behind and "decoy" in behind.tags:
                self._ui.update(self._all_placed(), decoy_fail=True)
                self.lose()
                self.complete_action()
                return
            sprite.set_position(crate_x, crate_y)
            self._player.set_position(new_x, new_y)
        elif not sprite or not sprite.is_collidable:
            self._player.set_position(new_x, new_y)

        self._steps += 1
        self._sync_ui()

        if self._all_placed():
            self.next_level()
        elif self._steps >= self._step_limit:
            self.lose()

        self.complete_action()
