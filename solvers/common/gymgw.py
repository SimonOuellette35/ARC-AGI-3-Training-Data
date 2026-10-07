"""Shared BaseSolver base for the Gym-Gridworlds generators.

ONE expert for the whole `gym_gridworlds` family (`adapters/gymgridworlds_adapter.py`).
Every supported env -- empty rooms, corridor, penalty grids, barrier, danger maze,
cliff walk, quicksand, four-/three-room layouts, the random maze and taxi
delivery -- is the same weighted shortest-path problem once the tile vocabulary is
written down, so the per-game generators are pure configuration (usually just
``env_id``) and carry no search code.

Absolute movement, so no heading in the state:

    ACTION1 up   ACTION2 down   ACTION3 left   ACTION4 right   ACTION5 stay

The tile rules the planner encodes (all from `gym_gridworlds.gridworld`)
--------------------------------------------------------------------
* ``WALL`` / ``PIT`` -- impassable. A PIT is instant death (-100), a wall move
  silently wastes the action; neither is ever worth an edge.
* ``QCKSND`` -- you LEAVE a quicksand tile with probability 0.1, so the edge out
  of one costs ~10 actions, not 1. That is a cost, not a barrier: the
  three-room-quicksand layouts use quicksand instead of walls as their dividers,
  so crossing is sometimes the only way through.
* ``BAD`` / ``BAD_SMALL`` -- -10 / -0.1 reward and no termination, so they are
  passable but weighted; the expert detours around a penalty tile unless the
  detour costs more than the penalty (Corridor-3x4 has no way round at all).
* directional tiles (the arrows, ids 0..8) -- standing on one, the ONLY action
  that moves you is the arrow's own direction. Modelled as a cell with a single
  outgoing edge, which is what makes the Barrier and *-Loop / *-Stuck layouts
  work: the goal there is ringed by arrows and reachable from one side only.
* ``RND_MOVE`` -- a directional tile whose direction is re-rolled after EVERY
  step. The live grid always shows the current arrow, so it plans correctly for
  one step; re-planning every step (below) is what keeps it honest.
* ``GOOD`` -- the target. There can be several (FourRooms-Wall, ThreeRooms-*);
  the reverse search starts from all of them at once.
* ``GOOD_SMALL`` -- a distractor worth +0.1 that ENDS the episode if you stand on
  it and press stay, so it is passable-but-weighted. In Taxi it is a passenger
  instead: `collect_first` makes them mandatory pickups (see below).

Winning
-------
The adapter fires WIN the moment the agent stands on a ``GOOD`` tile, so reaching
it is the whole objective -- with one exception the search cannot express: when a
level RESETS with the agent already on the goal (the random-start envs do this
about one level in twelve) no move is needed, just the action that lets the env
notice. That is ``STAY``, or any move in a ``no_stay`` env, and `_lookup` emits it
directly.

Taxi (``collect_first``) is the one env with more than position in its state: the
goal pays 0 / 1 / 3 / 15 for 0 / 1 / 2 / 3 passengers delivered and driving to the
destination empty ends the episode as a LOSS, so the state carries a bitmask of
which passengers have been picked up and the terminal is "on the goal, having
collected them all".

Re-planning and rotation
------------------------
Like the MiniGrid family this re-derives the field from the LIVE grid every step
(cached against ``grid.tobytes()``), which is what makes quicksand slips, the
re-rolled RND_MOVE arrows and the exploration prefix + perturbation bursts all
recoverable -- ``supports_recovery = True``.

The adapter applies a random k*90-degree frame rotation per level and inverse-
remaps directional inputs (`utils.rotation.remap_action`) so screen-up still moves
the agent up on screen. We plan in GRID space (the desired absolute action), then
invert that remap to recover the screen-space input to press, so each recorded
(frame, action) pair is self-consistent in display space.

Subclass: set ``env_id``. ``game_id`` is derived from it to match
``GymGridworldsAdapter.game_id``, and the action set is read off the env
(``no_stay`` layouts have no STAY action at all -- their action space is
``Discrete(4)``).
"""
from __future__ import annotations

import heapq
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np                                          # noqa: E402
from arcengine import ActionInput, GameAction, GameState    # noqa: E402
from adapters import GymGridworldsAdapter                   # noqa: E402
from utils.rotation import remap_action                     # noqa: E402
from gym_gridworlds.gridworld import (                      # noqa: E402
    ACTION_TO_VEC, BAD, BAD_SMALL, GOOD, GOOD_SMALL, PIT, QCKSND, WALL)
from solvers.base_solver import (                           # noqa: E402
    BaseSolver, DriveResult, _ID_TO_GAMEACTION, _as_action)

_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Grid move (dr, dc) -> the ABSOLUTE ("effective") GameAction the adapter maps to
# the corresponding gym_gridworlds action (ACTION1->UP, ACTION2->DOWN,
# ACTION3->LEFT, ACTION4->RIGHT).
_DELTA_TO_EFFECTIVE: dict[tuple[int, int], GameAction] = {
    (-1, 0): GameAction.ACTION1,   # up
    (1, 0): GameAction.ACTION2,    # down
    (0, -1): GameAction.ACTION3,   # left
    (0, 1): GameAction.ACTION4,    # right
}
_STAY = GameAction.ACTION5

_IMPASSABLE = frozenset({WALL, PIT})

#: Extra cost for ENTERING a tile, on top of the 1 action the move itself costs.
#: BAD is -10 reward, so it is worth a long detour; BAD_SMALL is -0.1, so it is
#: worth only a short one. GOOD_SMALL is a distractor that can end the episode
#: early for +0.1, so it is avoided where there is any alternative.
_ENTER_COST: dict[int, int] = {BAD: 30, BAD_SMALL: 3, GOOD_SMALL: 4}
#: Extra cost for LEAVING a tile: quicksand releases you with probability 0.1.
_QUICKSAND_LEAVE = 9
#: Entering an arrow tile costs a little: it takes away every exit but one.
_ARROW_ENTER = 2

_T = "WIN"                 # virtual terminal node of the search graph


def _input_for_effective(effective: GameAction, rotation_k: int) -> GameAction:
    """The screen-space input that, after the adapter's rotation remap, produces
    the desired absolute (effective) action. Inverts `remap_action` by searching
    the four directional inputs."""
    if rotation_k % 4 == 0:
        return effective
    for cand in (GameAction.ACTION1, GameAction.ACTION2,
                 GameAction.ACTION3, GameAction.ACTION4):
        if remap_action(cand, rotation_k) == effective:
            return cand
    return effective                                       # unreachable


class _Model:
    """The live gridworld plus the exact action-cost field over it.

    Built once per grid signature and then answers every step by a table lookup:
    `plan_from` descends the field for a cost-optimal move sequence and `ties_at`
    reads off EVERY equally-optimal next move, so the plan's head is always a
    member of the tie set by construction."""

    def __init__(self, grid: np.ndarray, *, collect_first: bool) -> None:
        self.grid = grid
        self.h, self.w = grid.shape
        self.goals = {(int(r), int(c))
                      for r, c in zip(*np.nonzero(grid == GOOD))}
        # Taxi: the passengers still ON the board. A collected one is erased from
        # the grid, so the signature changes and the field is rebuilt without it
        # -- which is why the live mask is always 0 (see `state_at`).
        self.picks = (sorted((int(r), int(c))
                             for r, c in zip(*np.nonzero(grid == GOOD_SMALL)))
                      if collect_first else [])
        self.pick_bit = {p: i for i, p in enumerate(self.picks)}
        self.full = (1 << len(self.picks)) - 1
        self.collect_first = collect_first
        self.edges: dict = {}
        self.togo: dict = {}

    # ── search graph ─────────────────────────────────────────────────────────
    def _exits(self, p):
        """The moves allowed FROM cell ``p``. An arrow tile permits exactly one;
        every other tile permits all four. (A tile whose arrow points at a wall or
        off the board is a dead end -- it yields nothing.)"""
        v = int(self.grid[p])
        if v in ACTION_TO_VEC:
            vec = ACTION_TO_VEC[v]
            return (vec,) if vec in _DELTA_TO_EFFECTIVE else ()
        return _DIRS

    def _succ(self, s):
        """Outgoing edges ``(next_state, cost, delta)`` of one state."""
        r, c, mask = s
        out = []
        leave = 1 + (_QUICKSAND_LEAVE if int(self.grid[r, c]) == QCKSND else 0)
        for dr, dc in self._exits((r, c)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < self.h and 0 <= nc < self.w):
                continue                       # off the board: the move is a no-op
            v = int(self.grid[nr, nc])
            if v in _IMPASSABLE:
                continue                       # wall wastes the action, pit kills
            cost = leave + _ENTER_COST.get(v, 0)
            if v in ACTION_TO_VEC:
                cost += _ARROW_ENTER
            nmask = mask
            if self.collect_first and (nr, nc) in self.pick_bit:
                nmask |= 1 << self.pick_bit[(nr, nc)]
                cost = leave                   # a passenger is a pickup, not a hazard
            if (nr, nc) in self.goals and (not self.collect_first
                                           or nmask == self.full):
                out.append((_T, cost, (dr, dc)))
            else:
                out.append(((nr, nc, nmask), cost, (dr, dc)))
        return out

    def build(self, start) -> None:
        """Enumerate the states reachable from the live cell, then reverse-
        Dijkstra from the terminal so every state carries its exact cost-to-win.
        One build serves every step until the grid changes."""
        stack = [start]
        seen = {start}
        while stack:
            s = stack.pop()
            es = self._succ(s)
            self.edges[s] = es
            for t, _c, _d in es:
                if t is not _T and t not in seen:
                    seen.add(t)
                    stack.append(t)

        pred: dict = {}
        for s, es in self.edges.items():
            for t, cost, _d in es:
                pred.setdefault(t, []).append((s, cost))
        if _T not in pred:
            return
        togo = {_T: 0}
        pq = [(0, 0, _T)]
        tick = 1
        while pq:
            d, _n, s = heapq.heappop(pq)
            if d > togo.get(s, 1 << 60):
                continue
            for u, cost in pred.get(s, ()):
                nd = d + cost
                if nd < togo.get(u, 1 << 60):
                    togo[u] = nd
                    heapq.heappush(pq, (nd, tick, u))
                    tick += 1
        self.togo = togo

    # ── read-out ─────────────────────────────────────────────────────────────
    def plan_from(self, s) -> list:
        """A cost-optimal sequence of grid moves from ``s``, by field descent."""
        d = self.togo.get(s)
        if d is None:
            return []
        plan: list = []
        guard = 0
        while d > 0 and guard < 4000:
            guard += 1
            for t, cost, delta in self.edges.get(s, ()):
                if self.togo.get(t, 1 << 60) == d - cost:
                    plan.append(delta)
                    s, d = t, d - cost
                    break
            else:
                return []
            if s is _T:
                break
        return plan

    def ties_at(self, s) -> list:
        """Every equally-optimal next move at ``s``."""
        d = self.togo.get(s)
        if not d:
            return []
        out: list = []
        for t, cost, delta in self.edges.get(s, ()):
            if self.togo.get(t, 1 << 60) == d - cost and delta not in out:
                out.append(delta)
        return out


class GymgwSolver(BaseSolver):
    """Standard multilevel BaseSolver for a Gym-Gridworlds env. Recovery-enabled.

    A "level" is one env reset: ``set_level(idx)`` reseeds at ``base + idx``, which
    re-rolls the layout (several envs get a per-level randomiser inside the
    adapter), the start cell, the goal cell and the frame rotation. Episode ``e``
    owns the block ``[e*L, e*L+L)`` so episodes never overlap."""

    env_id: str = ""

    #: Taxi: GOOD_SMALL tiles are passengers that must ALL be collected before the
    #: goal, because arriving empty-handed pays 0 and ends the episode as a loss.
    collect_first: bool = False

    levels_per_episode: int = 10
    supports_recovery = True
    #: More RESET recoveries than the base default: an exploratory step can walk
    #: into a pit or burn a tight step budget (several layouts allow only 50), and
    #: the episode-wide exploration prefix spends most of itself in level 0.
    max_resets = 6

    def __init_subclass__(cls, **kw) -> None:
        """Derive ``game_id`` from ``env_id`` exactly as ``GymGridworldsAdapter``
        does, so the generator's output directory matches the adapter's id."""
        super().__init_subclass__(**kw)
        if not cls.__dict__.get("game_id") and cls.__dict__.get("env_id"):
            clean = (cls.env_id.replace("Gym-Gridworlds/", "")
                     .replace("-v0", "").replace("-v1", ""))
            cls.game_id = "gymgridworlds_" + clean.lower().replace("-", "_")

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._cache: dict = {}          # grid signature -> _Model
        self._last = None               # (state key, plan, ties) of the last lookup

    # ── engine plumbing ──────────────────────────────────────────────────────
    def make_game(self, seed: int):
        self._cache.clear()
        self._last = None
        return GymGridworldsAdapter(self.env_id,
                                    seed=seed * self.levels_per_episode)

    def num_levels(self, game) -> int:
        return self.levels_per_episode

    def set_level(self, game, level_idx: int) -> None:
        super().set_level(game, level_idx)
        self._last = None

    def render(self, game) -> np.ndarray:
        return np.asarray(game._current_frame)

    def available_actions(self, game) -> list[int]:
        """The four moves, plus STAY where the env has it. A ``no_stay`` layout's
        action space is ``Discrete(4)``, so offering ACTION5 there would hand the
        exploration policy an action the env rejects."""
        return [1, 2, 3, 4] if getattr(game._env.unwrapped, "no_stay", False) \
            else [1, 2, 3, 4, 5]

    def drive(self, game, action) -> DriveResult:
        fd = game.perform_action(
            ActionInput(id=_ID_TO_GAMEACTION[action.action_id]))
        frame = np.asarray(fd.frame[0])
        solved = game._state == GameState.WIN
        dead = game._state == GameState.GAME_OVER
        return DriveResult(frame, solved, dead)

    # ── the field, cached against the live grid ──────────────────────────────
    def _model(self, game):
        u = game._env.unwrapped
        sig = u.grid.tobytes()
        m = self._cache.get(sig)
        if m is None:
            self._cache.clear()
            m = _Model(u.grid.copy(), collect_first=self.collect_first)
            m.build((int(u.agent_pos[0]), int(u.agent_pos[1]), 0))
            self._cache[sig] = m
        return m

    def _lookup(self, game):
        """``(plan, ties)`` at the live state as ABSOLUTE grid moves (or the STAY
        sentinel), both read off the SAME field so the plan's head is always in
        the tie set."""
        u = game._env.unwrapped
        pos = (int(u.agent_pos[0]), int(u.agent_pos[1]))
        key = (u.grid.tobytes(), pos)
        if self._last is not None and self._last[0] == key:
            return self._last[1], self._last[2]

        def answer(plan, ties):
            self._last = (key, plan, ties)
            return plan, ties

        # Already standing on the goal (the random-start layouts reset like this
        # about one level in twelve). gym_gridworlds scores the tile you are ON at
        # the START of a step, so the win needs one more action -- STAY, or in a
        # `no_stay` layout literally any move.
        if int(u.grid[pos]) == GOOD:
            if getattr(u, "no_stay", False):
                m = self._model(game)
                moves = list(m._exits(pos)) or [(-1, 0)]
                return answer([moves[0]], list(moves))
            return answer([_STAY], [_STAY])

        m = self._model(game)
        s = (pos[0], pos[1], 0)
        if s not in m.togo:
            # The live cell fell outside the enumerated set. Rebuild rooted here.
            self._cache.clear()
            m = self._model(game)
            s = (pos[0], pos[1], 0)
        return answer(m.plan_from(s), m.ties_at(s))

    def _to_inputs(self, game, moves) -> list:
        """Grid moves -> the rotation-aware screen-space inputs to press."""
        k = game._rotation_k
        return [mv if mv is _STAY
                else _input_for_effective(_DELTA_TO_EFFECTIVE[mv], k)
                for mv in moves]

    # ── the solver API ───────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """A cost-optimal plan from the agent's LIVE cell, re-derived from the
        live grid every call -- which is what makes exploration prefixes and
        perturbation bursts recoverable."""
        return self._to_inputs(game, self._lookup(game)[0])

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every equally-optimal next move at the LIVE cell. Open rooms have
        genuine ties (any interleaving of the two axes is the same length), so
        this feeds the base's STOCHASTIC OPTIMAL sampling with real diversity."""
        ties = self._to_inputs(game, self._lookup(game)[1])
        return ties or None

    def _optimal_set(self, game, level_idx: int, seed: int, plan: list):
        """Force a re-plan every step. The base otherwise pops a cached plan
        action by action, and here the world moves under it: quicksand drops most
        move attempts, and the RND_MOVE arrows are re-rolled after every step. The
        field itself is cached against the grid, so this is nearly free."""
        opts = super()._optimal_set(game, level_idx, seed, plan)
        if opts:
            plan[:] = opts[:1]
        return opts

    def _burst_recovery_wins(self, game, plan: list) -> bool:
        """Engine-verify that the post-burst state is really winnable, re-planning
        at every step rather than replaying ``plan`` blind -- a blind replay would
        spuriously fail wherever quicksand eats a move. Restores the live state,
        so this leaves no trace."""
        snap = self._game_snapshot(game)
        try:
            for _ in range(self.step_guard):
                p = [_as_action(a) for a in self.solve_from(game, 0, 0)]
                if not p:
                    return False
                res = self.drive(game, p[0])
                if res.solved:
                    return True
                if res.dead:
                    return False
            return False
        except Exception:                  # noqa: BLE001 -- a crash is not a win
            return False
        finally:
            self._cache.clear()
            self._last = None
            self._game_restore(game, snap)

    # ── CLI: expose --levels-per-episode alongside the base flags ────────────
    @classmethod
    def build_argparser(cls):
        p = super().build_argparser()
        p.add_argument("--levels-per-episode", type=int,
                       default=cls.levels_per_episode,
                       help="Distinct level resets bundled per episode.")
        return p

    @classmethod
    def main(cls, argv=None) -> int:
        args = cls.build_argparser().parse_args(argv)
        solver = cls(rng=random.Random(args.seed),
                     burst_prob=args.noise, burst_mean=args.burst)
        solver.levels_per_episode = args.levels_per_episode
        out = args.out or (Path("data/training_multi_level") / cls.game_id)
        return solver.run(args.episodes, out, start_seed=args.start_seed,
                          max_attempts=args.max_attempts,
                          progress_every=args.progress_every,
                          explore=not args.no_exploration)
