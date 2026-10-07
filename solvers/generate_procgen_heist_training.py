"""Generate Phase-1 training data for the ProcGen Heist game (procgen_heist).

Drives the ProcGen "heist" environment wrapped by ``adapters/procgen_adapter.py``
through the standard ``BaseSolver`` harness -- ONE adapter instance walked through
its 7 levels by ``set_level``, so the episode/level record loop, the schema, the
WIN filter, the exploration prefix and the perturbation bursts all come from the
base. (Contrast ``generate_procgen_maze_training.py``, which predates the adapter
being drivable this way and rebuilds a raw gym env per level.)

The game
--------
A maze holding up to three coloured KEYS, up to three matching coloured LOCKS and
one GEM. Walking over a key collects it; walking into a lock opens it once the
matching key has been collected; reaching the gem wins. Nothing in heist can kill
the agent and movement is reversible -- the only irreversible events are the
pickups and unlocks, and both only ever ADD reachable space. That is what makes
``recovery_mode = "replan"`` sound: no detour an exploration prefix or a
perturbation burst can take is unrecoverable.

Why the expert is a pixel field and not a state-space search
------------------------------------------------------------
ProcGen moves the agent with continuous accelerating physics (2.8-4.5 px per step
on a 4.9 px cell in hard mode), so its state is a real-valued position plus a
velocity. Searching that by quantising the position -- the approach
``generate_procgen_maze_training.py`` takes, and the first thing tried here --
does NOT work for heist: the clearance between the agent and a corridor wall is
about a pixel, so any quantisation coarse enough to keep the search affordable
merges "fits through this gap" with "does not", and the search then reports
boards unsolvable that the engine walks straight through (measured: at 2 px, 3 of
7 levels on the first seed). Simulating the engine is also the expensive part --
one ``act``/``observe`` costs ~1.9 ms -- so a few thousand expansions per level
is already several seconds.

So the expert plans on the FRAME and only ever asks the engine what it is fast at:

  * `_passable` marks the pixels the agent may occupy -- floor, plus the locks
    it can open;
  * `_distance_field` runs a 4-connected BFS back from the target over those
    pixels, giving exact pixel distance-to-target;
  * `Plan` picks that target: the GEM whenever the field reaches it, else the
    nearest KEY still on the floor, else the nearest lock not yet proven shut.
    A lock is just a wall a key turns into floor, so keys are the only subgoal.
  * `_best_actions` then chooses, by REPLAYING each of the four moves on the
    engine (snapshot / act / observe / restore) and reading the field at the
    position the agent really lands in. Four milliseconds a step buys away every
    movement-model bug at once: acceleration, ProcGen's corner-slide, and the
    sub-pixel clearance are observed rather than predicted.

Everything is re-derived from the live frame each step -- the field is cached
only against the world signature, so it is rebuilt exactly when something is
collected or opened -- which is what makes the exploration prefix and the
perturbation bursts recoverable: after any detour the expert simply reads where
it now is and carries on.

Two things about heist are not observable and are learned by trial instead
([[no-privileged-solvers]]): whether a given lock really needs its key (in this
ProcGen build some open on contact without one), and whether the agent started
the level already holding a key (ProcGen draws the inventory as icons in a corner
of the frame). `_passable` and `_room_sizes` document how each is pinned down.

Perception is otherwise plain reading of the same 64x64 palette frame the agent
sees, whose fixed 7-symbol vocabulary the adapter's `_postprocess_heist`
guarantees (1 wall / 5 floor / 9 blue / 14 green / 8 red / 11 gem / 12 + 0
player). Nothing reads ProcGen's internal object list.

Roughly 85% of ProcGen's heist boards solve; the rest put the gem behind a turn
the agent's body cannot make, are given up on by `_STALL_LIMIT`, and cost the
seed (`solve_episode` is all-or-nothing). Expect to generate about two seeds per
episode kept.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_procgen_heist_training.py --episodes 200 \
        --out data/training_multi_level/procgen_heist
"""

from __future__ import annotations

import sys
from pathlib import Path

# Repo root (parent of solvers/) -- that's where adapters/ + arcengine live.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                              # noqa: E402
from scipy import ndimage                                       # noqa: E402

from arcengine import ActionInput, GameAction, GameState        # noqa: E402
from adapters.procgen_adapter import (                          # noqa: E402
    ProcGenAdapter, _postprocess_heist, _COMPLETION_BONUS,
    _HEIST_WALL_IDX, _HEIST_BLUE_IDX, _HEIST_GREEN_IDX, _HEIST_RED_IDX,
    _HEIST_GEM_IDX, _HEIST_PLAYER_IDX, _HEIST_PLAYER_CORE_IDX,
)
from utils.explore import ExplorationPolicy, ExplorationPrefix   # noqa: E402
from solvers.base_solver import BaseSolver, DriveResult, _as_action  # noqa: E402

# The four movement keys, in the adapter's ACTION1..ACTION4 order.
_ID_TO_GA = {
    1: GameAction.ACTION1,   # up
    2: GameAction.ACTION2,   # down
    3: GameAction.ACTION3,   # left
    4: GameAction.ACTION4,   # right
}

#: Colour indices a key/lock can take in the post-processed frame.
_OBJECT_COLOURS = (_HEIST_BLUE_IDX, _HEIST_GREEN_IDX, _HEIST_RED_IDX)

#: A connected blob of at least this many pixels is a LOCK; anything smaller is a
#: KEY. Measured across both distribution modes the two are far apart -- locks are
#: cell-sized (18-57 px), keys are 2-3 row sprites (4-11 px) -- so the split is
#: never close to the boundary.
_LOCK_MIN_PIXELS = 14
#: ...and below this it is neither: a one- or two-pixel speck is the downsampler
#: smearing a sprite edge, and treating one as a key sends the agent off to
#: collect a thing that is not there.
_KEY_MIN_PIXELS = 3

#: Smallest open region the agent could actually stand and move in. Anything
#: smaller is not part of the maze -- see `_room_sizes`.
_MIN_ROOM_PIXELS = 40

#: Extra cost, in pixels of detour, for a step that would leave the agent's centre
#: within a pixel of a wall. This is a soft preference on TOP of `Plan`'s hard
#: choice of routing space, and it is what keeps the agent down the middle of a
#: corridor between two routes of equal length -- which is the one it can take at
#: speed without catching a corner.
_HUG_WALL_PENALTY = 3

#: An object counts as still on the board while at least this fraction of its
#: original pixels are visible. Standing against something shaves a pixel or two
#: off it through ProcGen's downsampling; collecting it removes all of them, so
#: the two cases are nowhere near each other.
_PRESENT_FRACTION = 0.35

#: Cost added per previous visit to a position when ranking a candidate move.
#: The field has no local minima, so descending it always makes progress -- until
#: the agent catches a corner the field routed it round, where every move can be
#: non-improving. Charging for ground already covered walks it back out instead of
#: letting it oscillate; it stays small so it can never outweigh real progress.
_REVISIT_PENALTY = 2

#: ProcGen's Discrete(15) joystick int for each of the four movement actions.
_ID_TO_PG = {1: 5, 2: 3, 3: 1, 4: 7}       # up, down, left, right
#: (dy, dx) for each, used to work out what a move that went nowhere ran into.
_ID_TO_STEP = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}

#: How far in front of the agent to look for whatever blocked a move, in pixels
#: -- just past its own sprite, so it lands on the obstruction and not on itself.
_BUMP_REACH = 3
#: Blocked moves into the same lock before it is written off as needing a key.
_BUMPS_TO_CONDEMN = 2

#: Give the level up after this many steps without ever getting closer to the
#: current target than the best already seen. A handful of ProcGen boards put the
#: gem behind a turn the agent's body cannot make, and without this the expert
#: batters at it for the whole 1400-step guard -- ~14 s per doomed level, on a
#: seed the harness is going to discard anyway. Generous enough that a legitimate
#: detour (round a lock, off to a key) never trips it: the longest route measured
#: across both distribution modes closes on its target inside ~60 steps.
_STALL_LIMIT = 150

_SQ3 = np.ones((3, 3), dtype=bool)
_CROSS = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
_UNREACHED = np.int32(1 << 29)


# ---------------------------------------------------------------------------
# Perception -- reads ONLY the 64x64 palette frame
# ---------------------------------------------------------------------------

class Board:
    """The static reading of a level: where its keys, locks and gem are.

    Built once from the level's first frame. Objects never move, so from then on
    the only question asked of a frame is which of THESE blobs are still there --
    which keeps the tracking immune to any stray pixel the downsampler invents
    somewhere else on the board."""

    __slots__ = ("keys", "locks", "gem", "held")

    def __init__(self, frame: np.ndarray) -> None:
        self.keys: list[tuple[np.ndarray, int, int]] = []    # (mask, colour, n0)
        self.locks: list[tuple[np.ndarray, int, int]] = []
        #: Colours whose key the agent is ALREADY carrying at the start.
        self.held: set = set()
        rooms = _room_sizes(frame)
        for colour in _OBJECT_COLOURS:
            mask = frame == colour
            if not mask.any():
                continue
            lab, n = ndimage.label(mask, _SQ3)
            for i in range(1, n + 1):
                blob = lab == i
                size = int(blob.sum())
                if size >= _LOCK_MIN_PIXELS:
                    self.locks.append((blob, colour, size))
                elif size >= _KEY_MIN_PIXELS:
                    if rooms[blob].max() < _MIN_ROOM_PIXELS:
                        self.held.add(colour)        # inventory icon, not a key
                    else:
                        self.keys.append((blob, colour, size))
        # Keep only gem blobs big enough to be the sprite. A single stray pixel
        # would otherwise become a second, unreachable target and the field would
        # route the agent at it forever.
        gem = frame == _HEIST_GEM_IDX
        lab, n = ndimage.label(gem, _SQ3)
        if n > 1:
            keep = np.zeros_like(gem)
            for i in range(1, n + 1):
                blob = lab == i
                if int(blob.sum()) >= 3:
                    keep |= blob
            if keep.any():
                gem = keep
        self.gem: np.ndarray = gem

    def signature(self, frame: np.ndarray) -> tuple[int, ...]:
        """Which keys and locks are still on the board, as a hashable tuple.

        Doubles as the cache key for the distance field: the field can only change
        when this does."""
        return tuple(int(_present(frame, blob, colour, n0))
                     for blob, colour, n0 in (*self.keys, *self.locks))


def _room_sizes(frame: np.ndarray) -> np.ndarray:
    """For every pixel, the area of the open region it belongs to.

    ProcGen draws heist's key INVENTORY as small key icons in the top-right
    corner of the frame, over the border, and on some boards the agent starts
    already holding one. Read naively that icon is a key lying on the floor --
    which makes the expert wait for a key that is already in hand and, on a board
    whose only route runs through that colour's lock, declare a perfectly
    winnable level dead. The icons give themselves away geometrically: each sits
    in its own islet of non-wall pixels far too small for the agent to stand in,
    whereas a key on the floor sits in a corridor system hundreds of pixels
    across. No hard-coded HUD rectangle needed."""
    lab, n = ndimage.label(frame != _HEIST_WALL_IDX, _SQ3)
    sizes = np.zeros(n + 1, dtype=np.int32)
    if n:
        sizes[1:] = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    return sizes[lab]


def _player_centre(frame: np.ndarray):
    """The agent's position, as the centre of its sprite's bounding box.

    The bounding-box centre rather than the pixel centroid: the sprite is
    lopsided (its orange is on the feet and hat) and animates as it walks, so the
    centroid wanders by a pixel or so between frames while the box does not."""
    ys, xs = np.nonzero((frame == _HEIST_PLAYER_IDX) |
                        (frame == _HEIST_PLAYER_CORE_IDX))
    if not ys.size:
        return None
    return ((int(ys.min()) + int(ys.max())) / 2.0,
            (int(xs.min()) + int(xs.max())) / 2.0)


def _present(frame: np.ndarray, blob: np.ndarray, colour: int, n0: int) -> bool:
    return int(((frame == colour) & blob).sum()) >= max(2, _PRESENT_FRACTION * n0)


# ---------------------------------------------------------------------------
# The pixel distance field
# ---------------------------------------------------------------------------

def _passable(frame: np.ndarray, board: Board, shut: frozenset,
              optimistic: bool) -> np.ndarray:
    """Pixels the agent's centre may occupy, under one of two readings of the locks.

    The STRICT reading is heist as documented: a lock is solid until the key of
    its colour has been collected, i.e. until that key is no longer on the board.
    It is the right prior -- it plans the short route straight to the gem -- and
    it is what the expert uses whenever it yields a plan at all.

    The OPTIMISTIC reading treats every lock as a door that opens on contact,
    except the ones the agent has already walked into and failed to open. It
    exists because the strict rule is not the whole truth in this ProcGen build:
    on some boards a lock opens on contact with no key collected at all
    (verified: the blue lock of one hard board opens on the first push from a
    standing start), and under the strict reading those boards read as dead --
    no key, no gem, nothing reachable. So when strict planning finds nothing to
    aim at, the expert stops assuming and goes and finds out, condemning locks
    that really are shut as it bumps into them ([[no-privileged-solvers]]: read
    the frame, pin the hidden part down by trial and error)."""
    solid = frame == _HEIST_WALL_IDX
    collected = {colour: colour in board.held for colour in _OBJECT_COLOURS}
    existed = {colour: colour in board.held for colour in _OBJECT_COLOURS}
    for blob, colour, n0 in board.keys:
        existed[colour] = True
        if not _present(frame, blob, colour, n0):
            collected[colour] = True
    for i, (blob, colour, n0) in enumerate(board.locks):
        if not _present(frame, blob, colour, n0):
            continue                                   # already opened -> floor
        if optimistic:
            if i in shut:
                solid |= blob
        elif not (existed[colour] and collected[colour]):
            solid |= blob
    return ~solid


def _distance_field(nav: np.ndarray, target: np.ndarray) -> np.ndarray:
    """4-connected BFS distance from every navigable pixel back to ``target``.

    Frontier-at-a-time over boolean masks, so the whole field is a couple of dozen
    numpy dilations rather than a Python queue over 4096 pixels."""
    dist = np.full((64, 64), _UNREACHED, dtype=np.int32)
    frontier = target & nav
    if not frontier.any():                             # target sits inside a wall:
        frontier = ndimage.binary_dilation(target, _SQ3, iterations=2) & nav
        if not frontier.any():
            return dist
    d = 0
    dist[frontier] = 0
    reached = frontier.copy()
    while frontier.any():
        d += 1
        frontier = ndimage.binary_dilation(frontier, _CROSS) & nav & ~reached
        dist[frontier] = d
        reached |= frontier
    return dist


def _nearest_navigable(nav: np.ndarray, pos) -> tuple[int, int] | None:
    """The navigable pixel closest to ``pos``.

    The agent's centre is not always inside the eroded free space -- squeezing
    along a wall puts it a pixel outside -- so every field lookup snaps to the
    nearest pixel that IS in it rather than reporting "unreachable"."""
    y, x = int(round(pos[0])), int(round(pos[1]))
    y = min(63, max(0, y))
    x = min(63, max(0, x))
    if nav[y, x]:
        return y, x
    ys, xs = np.nonzero(nav)
    if not ys.size:
        return None
    i = int(np.argmin((ys - y) ** 2 + (xs - x) ** 2))
    return int(ys[i]), int(xs[i])


class Plan:
    """The distance field for one reading of the board, plus what it aims at.

    Two readings are tried in turn -- strict locks first, then the optimistic one
    (see `_passable`) -- and within each, three targets in order of preference:

      * the GEM, whenever the field reaches it;
      * otherwise the nearest KEY still on the board -- collecting it is what
        turns the lock in the way into floor;
      * otherwise the nearest lock not yet proven shut, so the agent goes and
        finds out rather than giving up on a board it has not finished reading.

    ``ok`` is False only when nothing at all is reachable either way, which is
    the honest answer that the level cannot be finished from here."""

    __slots__ = ("nav", "open", "dist", "goal_is_gem", "ok", "optimistic")

    def __init__(self, frame: np.ndarray, board: Board, pos, shut: frozenset) -> None:
        self.dist = np.full((64, 64), _UNREACHED, dtype=np.int32)
        self.goal_is_gem = False
        self.ok = False
        self.optimistic = False
        self.nav = _passable(frame, board, shut, optimistic=False)
        self.open = ndimage.binary_erosion(self.nav, _CROSS)
        for optimistic in (False, True):
            free = (self.nav if not optimistic
                    else _passable(frame, board, shut, optimistic=True))
            # Route with the agent's body first and only fall back to routing a
            # point. Eroding is what stops the field threading the agent through
            # a one-pixel gap it cannot fit -- which it will then batter at
            # forever, since the field insists that is the way. But eroding can
            # also pinch a corner apart and hide a route that exists, so the
            # un-eroded space is kept as the fallback rather than the default.
            for nav in (ndimage.binary_erosion(free, _CROSS), free):
                start = _nearest_navigable(nav, pos)
                if start is None:
                    continue
                for target, is_gem in self._targets(frame, board, shut):
                    field = _distance_field(nav, target)
                    if field[start] >= _UNREACHED:
                        continue
                    self.nav = nav
                    self.open = ndimage.binary_erosion(nav, _CROSS)
                    self.dist, self.goal_is_gem = field, is_gem
                    self.ok, self.optimistic = True, optimistic
                    return

    @staticmethod
    def _targets(frame: np.ndarray, board: Board, shut: frozenset):
        yield board.gem, True
        keys = np.zeros((64, 64), dtype=bool)
        for blob, colour, n0 in board.keys:
            if _present(frame, blob, colour, n0):
                keys |= blob
        if keys.any():
            yield keys, False
        untried = np.zeros((64, 64), dtype=bool)
        for i, (blob, colour, n0) in enumerate(board.locks):
            if i not in shut and _present(frame, blob, colour, n0):
                untried |= blob
        if untried.any():
            yield untried, False

    def at(self, pos) -> int:
        """Distance-to-target at ``pos``, snapped to the nearest navigable pixel.

        Hugging a wall costs a little extra, so that between two routes of equal
        length the one down the middle of the corridor wins -- which is the one
        the agent can actually take at speed without catching a corner."""
        cell = _nearest_navigable(self.nav, pos)
        if cell is None:
            return int(_UNREACHED)
        snap = abs(cell[0] - pos[0]) + abs(cell[1] - pos[1])
        hug = 0 if self.open[cell] else _HUG_WALL_PENALTY
        return int(self.dist[cell]) + int(round(snap)) + hug


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

class ProcgenHeistSolver(BaseSolver):
    """BaseSolver over `ProcGenAdapter("heist")` -- 7 levels per episode."""

    game_id = "procgen_heist"
    supports_recovery = True
    recovery_mode = "replan"
    #: The adapter gives heist a 1000-action budget; leave headroom for the
    #: exploration prefix and any bursts on top of the route itself.
    step_guard = 1400
    #: Solved levels an episode must contain to be worth writing (see
    #: `solve_episode` -- this game emits a level SUBSET, not all 7).
    min_levels = 4

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._board: Board | None = None
        self._plan: Plan | None = None
        self._plan_sig: tuple | None = None
        self._world: tuple | None = None
        #: How often each rounded position has been stood on this level. Purely a
        #: record of what happened, used to break the tie when no move improves.
        self._visits: dict = {}
        #: Locks the agent has walked into and failed to open, plus the running
        #: bump count that puts them there. Both are observations, not knowledge:
        #: they are cleared the moment anything on the board changes.
        self._shut: set = set()
        self._bumps: dict = {}
        #: Closest the agent has come to the current target, and how many steps
        #: ago -- counted per STEP TAKEN, not per query: the base asks for the
        #: optimal set more than once per step, and charging patience for being
        #: asked would trip the guard part-way through a perfectly good route.
        self._best_dist: int = int(_UNREACHED)
        self._stalled: int = 0
        #: One-entry memo of the four-move lookahead, keyed by the live state, so
        #: the several calls the base makes per step simulate the engine once.
        self._lookahead_cache: tuple | None = None

    # ── adapter plumbing ─────────────────────────────────────────────────────
    def make_game(self, seed: int):
        return ProcGenAdapter("heist", seed=seed)

    def num_levels(self, game) -> int:
        return game.n_levels

    def render(self, game) -> np.ndarray:
        return np.asarray(game._current_frame)

    def available_actions(self, game=None) -> list[int]:
        return list(_ID_TO_GA)

    def drive(self, game, action) -> DriveResult:
        before = np.asarray(game._current_frame)
        pos_before = _player_centre(before)
        game.perform_action(ActionInput(id=_ID_TO_GA[action.action_id]))
        after = np.asarray(game._current_frame)
        solved = game._state == GameState.WIN
        if not solved:
            pos = _player_centre(after)
            if pos is not None:
                self._visits[self._pos_key(pos)] = 1 + self._visits.get(
                    self._pos_key(pos), 0)
                if pos_before is not None:
                    self._note_bump(after, pos_before, pos, action.action_id)
                self._note_progress(game)
        return DriveResult(after, solved, game._state == GameState.GAME_OVER)

    def _note_progress(self, game) -> None:
        """Tick the stall guard: did this step get us closer than we have been?"""
        _frame, pos, plan = self._current_plan(game)
        if pos is None or plan is None:
            return
        here = plan.at(pos)
        if here < self._best_dist:
            self._best_dist, self._stalled = here, 0
        else:
            self._stalled += 1

    def _note_bump(self, frame, pos_before, pos_after, action_id: int) -> None:
        """Record a lock that just refused to open.

        A move that leaves the agent where it was, aimed at a lock still on the
        board, is the only evidence the game gives that the lock needs something
        we do not have. Two of them settle it -- one alone can be the agent
        catching the corner of the doorway rather than the door itself."""
        if (abs(pos_after[0] - pos_before[0]) +
                abs(pos_after[1] - pos_before[1])) >= 0.6:
            return                                  # it moved: nothing refused it
        dy, dx = _ID_TO_STEP[action_id]
        y = int(round(pos_before[0] + dy * _BUMP_REACH))
        x = int(round(pos_before[1] + dx * _BUMP_REACH))
        if not (0 <= y < 64 and 0 <= x < 64) or self._board is None:
            return
        for i, (blob, colour, n0) in enumerate(self._board.locks):
            if i in self._shut or not blob[y, x]:
                continue
            if not _present(frame, blob, colour, n0):
                continue
            self._bumps[i] = self._bumps.get(i, 0) + 1
            if self._bumps[i] >= _BUMPS_TO_CONDEMN:
                self._shut.add(i)
                self._plan_sig = None               # re-plan around it
            return

    def set_level(self, game, level_idx: int) -> None:
        game.set_level(level_idx)
        self._board = None
        self._plan = None
        self._plan_sig = None
        self._world = None
        self._visits.clear()
        self._shut.clear()
        self._bumps.clear()
        self._best_dist = int(_UNREACHED)
        self._stalled = 0
        self._lookahead_cache = None

    def reset_level(self, game, level_idx: int, seed: int) -> None:
        """RESET the level in place.

        NOT the base implementation, which routes through ``set_level`` -- for
        heist that re-runs `ProcGenAdapter._make_good_env`'s trivial-seed search
        (up to 50 fresh ``gym.make`` calls) when all a RESET means is re-seating
        the SAME env, which the adapter's own ``level_reset`` does."""
        game.level_reset()
        if game._state == GameState.GAME_OVER:
            game._state = GameState.NOT_FINISHED
        self._plan = None
        self._plan_sig = None
        self._world = None
        self._visits.clear()
        self._shut.clear()
        self._bumps.clear()
        self._best_dist = int(_UNREACHED)
        self._stalled = 0
        self._lookahead_cache = None

    # ── the expert ───────────────────────────────────────────────────────────
    @staticmethod
    def _pos_key(pos) -> tuple[int, int]:
        return int(round(pos[0])), int(round(pos[1]))

    def _current_plan(self, game):
        """The field for the LIVE frame, rebuilt only when the world changed.

        Returns ``(frame, pos, plan)``, or ``(frame, pos, None)`` when there is
        nothing left to aim at."""
        frame = self.render(game)
        pos = _player_centre(frame)
        if pos is None:
            return frame, None, None
        if self._board is None:
            self._board = Board(frame)
        sig = self._board.signature(frame)
        if sig != self._world:                      # something was collected/opened
            self._world = sig
            self._shut.clear()                      # a key may be what a lock wanted
            self._bumps.clear()
            self._visits.clear()                    # old ground is new again
            self._plan_sig = None
            self._best_dist = int(_UNREACHED)        # new target, new patience
            self._stalled = 0
        stamp = (sig, frozenset(self._shut))
        if stamp != self._plan_sig:
            self._plan = Plan(frame, self._board, pos, stamp[1])
            self._plan_sig = stamp
        return frame, pos, self._plan if self._plan and self._plan.ok else None

    def _best_actions(self, game) -> list[int]:
        """Every action that descends the field furthest from the LIVE state.

        Each candidate is scored by REPLAYING it on the engine (snapshot, act,
        observe, restore) and reading the field at the position the agent really
        lands in. That one extra ms per candidate buys away the entire class of
        bugs a movement model would introduce: acceleration, the corner-slide
        ProcGen applies when a move is partly blocked, and the sub-pixel clearance
        that decides whether the agent fits through a gap are all simply observed
        rather than predicted. A move that takes the gem outright scores below
        everything else."""
        _frame, pos, plan = self._current_plan(game)
        if pos is None or plan is None:
            return []
        if self._stalled > _STALL_LIMIT:
            return []                    # not getting anywhere: fail the level
        cached = self._lookahead_cache
        stamp = (game._action_count, hash(_frame.tobytes()))
        if cached is not None and cached[0] == stamp:
            return cached[1]
        gym3 = game._gym3
        snap = game._snapshot()
        scored: dict[int, int] = {}
        try:
            state = gym3.get_state()
            for aid, pg in _ID_TO_PG.items():
                gym3.set_state(state)
                gym3.act(np.array([pg], dtype=np.int32))
                reward, obs, first = gym3.observe()
                if float(reward[0]) >= _COMPLETION_BONUS:  # takes the gem
                    scored[aid] = -1
                    continue
                if bool(first[0]):                        # ran out of ProcGen's clock
                    continue
                nxt = _player_centre(_postprocess_heist(obs["rgb"][0]))
                if nxt is None:
                    continue
                scored[aid] = (plan.at(nxt)
                               + _REVISIT_PENALTY * self._visits.get(
                                   self._pos_key(nxt), 0))
        finally:
            game._restore(snap)
        if not scored:
            return []
        best = min(scored.values())
        moves = ([] if best >= _UNREACHED
                 else sorted(aid for aid, sc in scored.items() if sc == best))
        self._lookahead_cache = (stamp, moves)
        return moves

    def solve_from(self, game, level_idx: int, seed: int) -> list:
        """An optimal-descent plan from the game's LIVE state.

        Only the head is ever executed -- `_optimal_set` truncates the plan so the
        next step re-derives everything from the frame -- but the plan is returned
        at its full estimated length, because `_burst_recovery_wins` sizes its own
        step budget from it."""
        best = self._best_actions(game)
        if not best:
            return []
        _frame, pos, plan = self._current_plan(game)
        cell = _nearest_navigable(plan.nav, pos)
        remaining = 1
        if cell is not None and plan.dist[cell] < _UNREACHED:
            # Pixel distance -> steps, at the agent's ~3.5 px per step.
            remaining = max(1, int(plan.dist[cell] / 3.5) + 1)
        head = _ID_TO_GA[best[0]]
        return [head] * remaining

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-good next action at the LIVE state -- the training target,
        and the set the base samples the taken action from. Open stretches of the
        map genuinely tie, so this is real trajectory diversity, not a formality."""
        best = self._best_actions(game)
        return [_ID_TO_GA[a] for a in best] if best else None

    def _optimal_set(self, game, level_idx: int, seed: int, plan: list):
        """Force a re-decision every step: the tail `solve_from` returns is a
        length estimate, not a route, so truncating the cached plan to the live
        optimum keeps the base taking freshly-derived actions."""
        opts = super()._optimal_set(game, level_idx, seed, plan)
        if opts:
            plan[:] = opts[:1]
        return opts

    def _burst_recovery_wins(self, game, plan: list) -> bool:
        """Engine-verify that the post-burst state is still winnable by playing it
        out, re-deciding every step, and restoring afterwards.

        The base implementation replays ``plan`` verbatim, which here is a
        length estimate rather than a route -- it would report every burst
        unrecoverable and roll them all back, losing exactly the recovery data
        bursts exist to produce."""
        # Bound the check by the route it is checking. ``plan``'s length is the
        # whole remaining route, and the rollout follows a near-optimal one, so
        # twice that plus a margin is ample -- and the rollout is the expensive
        # part of a burst (four engine simulations per step), so a loose bound
        # here is what makes bursting dominate generation time.
        budget = min(self.step_guard, 2 * len(plan) + 40)
        snap = self._game_snapshot(game)
        memory = (dict(self._visits), set(self._shut), dict(self._bumps),
                  self._world, self._best_dist, self._stalled)
        try:
            for _ in range(budget):
                best = self._best_actions(game)
                if not best:
                    return False
                res = self.drive(game, _as_action(_ID_TO_GA[best[0]]))
                if res.solved:
                    return True
                if res.dead:
                    return False
            return False
        except Exception:                  # noqa: BLE001 -- a crash is not a win
            return False
        finally:
            self._game_restore(game, snap)
            (self._visits, self._shut, self._bumps,
             self._world, self._best_dist, self._stalled) = memory
            # Drop BOTH halves of the plan cache. Restoring the stamp while
            # clearing the Plan it stands for makes every later lookup hit a
            # cache entry holding None and report the level unplannable -- which
            # silently failed every seed the first time bursts were switched on.
            self._plan = None
            self._plan_sig = None
            self._lookahead_cache = None


    # ── episode model: a SUBSET of the 7 levels ──────────────────────────────
    def solve_episode(self, seed: int, explore: bool = True):
        """Record every level of ``seed`` that can be solved, and keep the seed if
        enough of them were.

        NOT the base's all-or-nothing model. ProcGen generates heist boards freely
        and about one in six puts the gem behind a turn the agent's body cannot
        make; the expert gives up on those (`_STALL_LIMIT`). Under all-or-nothing
        a single such board discards the whole seed -- and because consecutive
        seeds share boards (level L of seed S is ProcGen level S + L), one bad
        board poisons five seeds in a row, which measured out at a ~30% seed
        yield with most of the work thrown away. Emitting the levels that DID
        solve keeps every one of those trajectories, at the cost of episodes with
        a gap in their level ids -- the same trade the other subset-emitting
        generators make (bp35, wa30, dc22, ...).

        Everything else matches the base: the exploration prefix and policy are
        built ONCE per episode so the exploration arc spans it."""
        game = self.make_game(seed)
        n = self.num_levels(game)
        do_explore = explore and self.supports_recovery
        schedule = (ExplorationPrefix(self.rng, center=self._explore_center,
                                      jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(game), self.rng)
                       if do_explore else None)
        levels: list[dict] = []
        for level_idx in range(n):
            try:
                obs, acts = self.record_level(game, level_idx, seed,
                                              schedule=schedule,
                                              exploration=exploration)
            except Exception:                    # noqa: BLE001 -- a bad board just fails
                obs = None
            if obs is None:
                continue                         # unsolvable board: skip the level
            levels.append({"level_id": level_idx, "observations": obs,
                           "actions": acts})
        return len(levels) >= self.min_levels, levels


if __name__ == "__main__":
    sys.exit(ProcgenHeistSolver.main())
