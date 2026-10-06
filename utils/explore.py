"""Shared exploration behaviour for the solvers and the data generators.

This is the single source of truth for the two pieces of game-agnostic behaviour
that every live-driving consumer needs -- the offline per-game `solve()` experts,
the live `solver.py` roll-out, the training-data generators in `solvers/`, and
the goal-hypothesizer input builder `generate_goal_input.py`:

  * `ExplorationPolicy` -- the coverage-driven semi-smart exploration policy.
    Every available simple action (and a click, where the game exposes one)
    appears at least once in shuffled order before any uniform-random refill, so
    a short prefix already exercises the whole action space. Click targets are
    biased to salient (non-background) cells, because uniform-random clicks on a
    64x64 board almost never hit anything. This is exactly the scheme that used
    to live inline in `generate_goal_input.py`; that script now imports it here.

  * `ExplorationPrefix` -- an EPISODE-WIDE, fixed-length BINARY exploration
    prefix. The episode opens with exactly `length` pure-exploration steps, drawn
    ONCE per episode as `center +/- jitter` (default 10 +/- 5, so uniform in
    [5, 15]), and then hands off to the solver's optimal policy. This is a clean
    two-phase explore-then-exploit arc, not a probabilistic decay.

    "Episode-wide, not level-wide" is the load-bearing detail: the step counter
    spans the whole episode and is NOT reset when the engine auto-advances to the
    next level, so the prefix happens ONCE at the episode's opening rather than
    restarting each level. (`EpsilonSchedule` remains as a back-compat alias so
    older imports keep working.)

Action representation
---------------------
An `Action` carries the discrete action id (the `GameAction` enum value: RESET=0,
ACTION1..5, CLICK==ACTION6, ...) and, for a click, the target as `(row, col)`.
Two click conventions coexist in the codebase, so both are exposed:

  * `click_rc` -> `(row, col)`  -- array/`np.argwhere` convention.
  * `click_xy` -> `(x=col, y=row)` -- the engine's mouse convention (matches
    `solver.py`'s `LiveEnv.step(idx, xy)` and `evidence_log._step_mouse`).
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass

import numpy as np

# Discrete action ids (the GameAction enum value IS the id). Kept as plain ints
# so this module has no hard dependency on `arcengine` -- generators import it in
# environments where the engine is not on the path.
RESET_ACTION = 0
CLICK_ACTION = 6            # ACTION6 == mouse click at (row, col)
UNUSED_ACTIONS = (7,)      # ACTION7/UNDO is not an exploratory move

# Sentinel placed in the coverage queue to mean "emit a click"; the concrete
# target cell is only chosen when the action is actually drawn, against the
# frame current at that moment.
_CLICK = "click"


# ---------------------------------------------------------------------------
# Action value object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Action:
    """A chosen action: a discrete id plus, for a click, its target cell."""

    action_id: int
    click_rc: tuple[int, int] | None = None

    @property
    def is_click(self) -> bool:
        return self.click_rc is not None

    @property
    def click_xy(self) -> tuple[int, int] | None:
        """Target as the engine's `(x=col, y=row)` mouse convention, or None."""
        if self.click_rc is None:
            return None
        r, c = self.click_rc
        return (c, r)


# ---------------------------------------------------------------------------
# Salient-cell click sampling
# ---------------------------------------------------------------------------
def random_click_target(grid: np.ndarray, rng: random.Random) -> tuple[int, int]:
    """A random `(row, col)` to click, biased to content: uniform over the
    non-background cells (background = the single most frequent colour), falling
    back to any cell when the board is a solid colour. Uniform-random clicking a
    64x64 board almost always lands on background and does nothing, so the whole
    exploration prefix would degenerate to a no-op sequence without this bias."""
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[int(np.argmax(counts))])
    rows, cols = np.where(grid != bg)
    if len(rows) == 0:
        return rng.randrange(grid.shape[0]), rng.randrange(grid.shape[1])
    i = rng.randrange(len(rows))
    return int(rows[i]), int(cols[i])


# ---------------------------------------------------------------------------
# Coverage-driven exploration policy
# ---------------------------------------------------------------------------
class ExplorationPolicy:
    """Semi-smart exploration: cover the action space, then refill at random.

    On construction the available actions are split into simple (keyboard)
    actions and an optional click. The coverage queue is every simple action
    plus (if available) one click, shuffled; `action()` draws from the queue and,
    once it is exhausted, refills with a uniform-random draw over the same pool.
    So the first `len(simple) + has_click` actions guarantee each available
    action is exercised at least once, in a randomised order, before anything
    repeats -- the highest-information opening for probing unknown mechanics.

    The policy is stateful: each `action()` call advances the coverage pointer.
    Feed only the steps you actually spend exploring (an `ExploringAgent` calls
    it only on explore steps), so a mostly-optimal episode still walks the whole
    coverage list across its scattered exploratory steps.
    """

    def __init__(self, available_actions, rng: random.Random, *,
                 reset_action: int = RESET_ACTION, click_action: int = CLICK_ACTION,
                 exclude=UNUSED_ACTIONS) -> None:
        avail = [int(a) for a in (available_actions if available_actions is not None else [])]
        avail = [a for a in avail if a not in exclude]
        self.simple = [a for a in avail if a not in (reset_action, click_action)]
        self.has_click = click_action in avail
        self.click_action = click_action
        self.rng = rng
        # Pool the refill phase draws from; the click sentinel stands in for a
        # click so its target is (re)chosen against the live frame each time.
        self._pool = list(self.simple) + ([_CLICK] if self.has_click else [])
        if not self._pool:
            raise ValueError("No available gameplay actions for exploration")
        self.reset()

    def reset(self) -> None:
        """Redraw the shuffled coverage queue (e.g. at the start of an episode)."""
        once = list(self.simple) + ([_CLICK] if self.has_click else [])
        self.rng.shuffle(once)
        self._queue: deque = deque(once)

    def _next_kind(self):
        if self._queue:
            return self._queue.popleft()
        return self.rng.choice(self._pool)

    def action(self, frame: np.ndarray) -> Action:
        """Draw the next exploratory action against the current `frame`."""
        kind = self._next_kind()
        if kind == _CLICK:
            return Action(self.click_action, random_click_target(frame, self.rng))
        return Action(int(kind))


# ---------------------------------------------------------------------------
# Episode-wide exploration prefix
# ---------------------------------------------------------------------------
class ExplorationPrefix:
    """A fixed-length BINARY exploration prefix, drawn once per episode.

    The episode opens with exactly ``length`` pure-exploration steps and then
    hands off to the solver's optimal policy -- a clean two-phase
    explore-then-exploit arc, not a probabilistic decay. ``length`` is drawn at
    construction as ``center + U{-jitter..+jitter}`` (default 10 +/- 5 -> [5, 15]);
    build ONE per EPISODE and never rebuild it at a level boundary, so the prefix
    stays at the episode's opening rather than restarting each level.
    """

    def __init__(self, rng: random.Random, *, center: int = 10, jitter: int = 5) -> None:
        self.length = max(0, center + rng.randint(-jitter, jitter))
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    def explore(self) -> bool:
        """True while still inside the opening prefix (this step should explore)."""
        return self._step < self.length

    def advance(self, n: int = 1) -> None:
        """Advance the episode-wide step counter (call once per taken step)."""
        self._step += n


# Back-compat alias: generators construct the episode-scoped exploration object by
# this name and only call ``.explore()`` / ``.advance()`` on it, so swapping the
# mechanism (epsilon decay -> binary prefix) needs no change in any generator.
EpsilonSchedule = ExplorationPrefix
