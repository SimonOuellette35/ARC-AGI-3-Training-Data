"""Generate Phase-1 training data for the ProcGen Maze game (procgen_maze).

Drives the ProcGen "maze" environment wrapped by ``adapters/procgen_adapter.py``.
The output schema, action encoding and WIN-filter CLI live in ``BaseSolver``;
this module supplies the bespoke real-dynamics expert and a ``solve_episode``
override (the native record/replay loop does not fit -- see below).

Multi-env multilevel model (custom ``solve_episode``)
-----------------------------------------------------
ONE procgen_maze base seed is a multi-level game (``N_LEVELS`` = len of the maze
seed pool), but -- unlike the native ARCBaseGame model where one instance advances
through levels in place -- EACH level is a SEPARATE single-level ProcGen env built
by ``_make_level_env(level, base_seed)`` from the curated absolute seed pools in
``adapters.procgen_adapter._LEVEL_SEEDS_ABS["maze"]`` (level L uses
pool[base_seed % len(pool)]). A base seed is kept only if ALL levels solve.
Because every level is a fresh env with its own reachability graph and its own
close(), ``solve_episode`` is overridden rather than driving through the base's
``record_level``/``set_level`` template.

Expert solver
-------------
ProcGen's agent moves with continuous physics (no clean grid step) and the adapter
exposes only the rendered frame, so a frame-space BFS controller is fragile.
Instead we drive the REAL dynamics: ProcgenGym3Env exposes get_state()/set_state(),
so we BFS over discretized agent positions (the idx-1 player-blob centroid) to map
the whole maze into a reachability graph, compute a dist-to-goal field, and follow
it greedily. Exact, guaranteed-correct. supports_recovery stays False (no
exploration): the env is not the base engine and the expert is pure shortest path.

Winning-step note: on reaching the exit ProcGen terminates and IMMEDIATELY
auto-resets the agent to the maze start, so the engine never RENDERS the
"agent on goal" frame. We synthesize that final WIN frame from the last clean grid
frame -- the player cell becomes floor and the goal cell becomes the player -- so
the trajectory ends with the agent standing on the goal.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_procgen_maze_training.py --episodes 200 \
        --out data/training_multi_level/procgen_maze
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

# Repo root (parent of solvers/) -- that's where adapters/ + arcengine live.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

import gym  # noqa: E402  (procgen registers its envs on import)
import procgen  # noqa: F401,E402

from arcengine import GameAction  # noqa: E402
from adapters.procgen_adapter import (  # noqa: E402
    _rgb_to_arc_frame, _postprocess_maze, _LEVEL_SEEDS_ABS,
    _MAZE_PLAYER_IDX, _MAZE_GOAL_IDX, _MAZE_FLOOR_IDX,
)
from utils.explore import (  # noqa: E402
    Action, EpsilonSchedule, ExplorationPolicy, RESET_ACTION)
from solvers.base_solver import BaseSolver  # noqa: E402

GAME_NAME = "maze"
N_LEVELS = len(_LEVEL_SEEDS_ABS[GAME_NAME])          # 7

# The RAW nearest-neighbour quantisation index used ONLY to locate the animated
# mouse blob for the BFS (the raw frame is where the sprite lives). The SAVED
# observations instead go through _postprocess_maze (matching ProcGenAdapter).
PLAYER_IDX = 1
PLAYER_FALLBACK = (1, 3)                              # player outline shares idx 3

# ProcGen Discrete(15) move ints (joystick 3x3): up=5, down=3, left=1, right=7.
MOVES = (5, 3, 1, 7)
_PG_TO_ACTION = {
    5: GameAction.ACTION1,   # up
    3: GameAction.ACTION2,   # down
    1: GameAction.ACTION3,   # left
    7: GameAction.ACTION4,   # right
}
# Inverse of _PG_TO_ACTION, keyed by the GameAction VALUE (1..4): what the
# exploration policy emits (an ``Action`` over the four move ids) -> the ProcGen
# joystick int to feed the real env.
_ACTION_TO_PG = {int(a.value): pg for pg, a in _PG_TO_ACTION.items()}

# Position discretization (px): agent moves ~2.5px/step; 1.5 keeps even the
# densest mazes' ~2px corridors as distinct BFS cells without exploding the graph.
DISC = 1.5
MAX_EXPLORE_NODES = 60000


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _gym3_env(env):
    """Walk the gym wrapper chain to the ProcgenGym3Env (has get_state/set_state)."""
    o = env
    while o is not None and not hasattr(o, "get_state"):
        o = getattr(o, "env", None)
    if o is None:
        raise RuntimeError("could not locate ProcgenGym3Env (get_state/set_state)")
    return o


def _make_level_env(level: int, base_seed: int):
    """Create a single-level maze env for (level, base_seed) using the curated
    absolute seed pools -- identical seed selection to ProcGenAdapter.set_level."""
    mode, pool = _LEVEL_SEEDS_ABS[GAME_NAME][level]
    start = pool[base_seed % len(pool)]
    return gym.make(
        "procgen:procgen-maze-v0",
        start_level=start,
        num_levels=1,
        distribution_mode=mode,
        use_backgrounds=False,
        render_mode="rgb_array",
    )


def _agent_pos(rgb: np.ndarray):
    """(row, col) centroid of the player blob in a 64x64x3 RGB frame, or None."""
    f = _rgb_to_arc_frame(rgb)
    m = f == PLAYER_IDX
    if not m.any():
        m = np.isin(f, PLAYER_FALLBACK)
    if not m.any():
        return None
    ys, xs = np.nonzero(m)
    return (ys.mean(), xs.mean())


def _frame(rgb: np.ndarray) -> np.ndarray:
    """64x64 palette-index frame, post-processed identically to ProcGenAdapter
    (_rgb_to_arc_frame -> _postprocess_maze) so the SAVED frames match what
    solver_client.py feeds the encoder/dynamics stack at inference time."""
    return _postprocess_maze(rgb, _rgb_to_arc_frame(rgb))


def _key(pos):
    return (int(round(pos[0] / DISC)), int(round(pos[1] / DISC)))


# ---------------------------------------------------------------------------
# Exact expert: explore the real dynamics -> reachability graph + dist-to-goal
# ---------------------------------------------------------------------------

def _explore(g3):
    """BFS the maze over discretized agent positions using state save/restore.

    Returns (graph, dist, s0, start_key) or None if the goal is unreachable
    within MAX_EXPLORE_NODES.
      graph: key -> {move_int: next_key or 'GOAL'}   (only moves that change cell)
      dist:  key -> min #moves to win                 (for greedy descent)
    """
    rew, obs, first = g3.observe()
    p0 = _agent_pos(obs["rgb"][0])
    if p0 is None:
        return None
    s0 = g3.get_state()
    start_key = _key(p0)

    graph: dict = {}
    visited = {start_key}
    queue = deque([(start_key, s0)])
    nodes = 0
    goal_seen = False

    while queue and nodes < MAX_EXPLORE_NODES:
        k, st = queue.popleft()
        edges = graph.setdefault(k, {})
        for a in MOVES:
            g3.set_state(st)
            g3.act(np.array([a], dtype=np.int32))
            rew, obs, first = g3.observe()
            nodes += 1
            if float(rew[0]) > 0:                    # this move wins
                edges[a] = "GOAL"
                goal_seen = True
                continue
            if bool(first[0]):                       # timeout/death auto-reset
                continue
            npos = _agent_pos(obs["rgb"][0])
            if npos is None:
                continue
            nk = _key(npos)
            if nk == k:                              # blocked: agent did not move
                continue
            edges[a] = nk
            if nk not in visited:
                visited.add(nk)
                queue.append((nk, g3.get_state()))

    if not goal_seen:
        return None

    # Reverse BFS for dist-to-goal: cells with a winning move are 1 move away.
    dist: dict = {}
    frontier = deque()
    pred: dict = {}
    for k, edges in graph.items():
        for a, nk in edges.items():
            if nk == "GOAL":
                if k not in dist:
                    dist[k] = 1
                    frontier.append(k)
            else:
                pred.setdefault(nk, []).append(k)
    while frontier:
        k = frontier.popleft()
        for pk in pred.get(k, []):
            if pk not in dist:
                dist[pk] = dist[k] + 1
                frontier.append(pk)

    return graph, dist, s0, start_key


class ProcgenMazeSolver(BaseSolver):
    game_id = "procgen_maze"
    # Maze navigation is fully REVERSIBLE and each level's env exposes an exact
    # state save/restore, so the epsilon exploration prefix is recoverable: explore
    # random moves, then RESET the env back to its INITIAL maze (``set_state(s0)``,
    # the same initial the greedy plan was computed from) and follow the shortest
    # path from the start. ``reset_level`` == re-seat the env at s0. Hence
    # ``recovery_mode = "reset"``.
    supports_recovery = True
    recovery_mode = "reset"
    step_guard = 500                           # per-level move cap (procgen step budget)

    # make_game / solve_from are unused: solve_episode is fully overridden because
    # each of the 7 levels is a distinct env with its own reachability graph.

    def available_actions(self, game=None) -> list[int]:
        """The four movement keys -- procgen has no click and no game object to
        introspect, so the exploration policy covers up/down/left/right."""
        return [int(a.value) for a in _PG_TO_ACTION.values()]

    def _solve_level(self, level: int, base_seed: int, *,
                     schedule: EpsilonSchedule | None = None,
                     exploration: ExplorationPolicy | None = None):
        """Solve one level and record its trajectory via ``_encode_step``.

        Returns (observations, actions) on success, or None if unsolved. The
        trajectory ends with a synthesized WIN frame (agent on the goal cell).

        When ``exploration``/``schedule`` are supplied the recording opens with the
        episode-wide epsilon prefix (random real moves), then a RESET restores the
        env to its initial maze (``set_state(s0)``) before the exact greedy plan is
        followed from the start -- the "reset" recovery paradigm. Because the greedy
        plan is recomputed from ``s0`` and the reset returns to exactly ``s0``, the
        level still wins deterministically regardless of the detour."""
        env = _make_level_env(level, base_seed)
        try:
            env.reset()
            g3 = _gym3_env(env)
            explored = _explore(g3)
            if explored is None:
                return None
            graph, dist, s0, start_key = explored

            g3.set_state(s0)
            rew, obs, first = g3.observe()
            prev = _frame(obs["rgb"][0])
            observations = [prev.tolist()]
            actions = [self._reset_step()]

            def emit(pg_move: int, frame: np.ndarray, phase: str = "expert",
                     optimal_moves=None) -> None:
                """Record one step. ``optimal_moves`` is the co-optimal ProcGen move
                set at the pre-step cell (all lead to a shortest win); ``None`` marks
                a non-target (exploration) step. STOCHASTIC OPTIMAL: the recorded
                target is that whole set, not just the taken move."""
                nonlocal prev
                taken = Action(int(_PG_TO_ACTION[pg_move].value))
                opt = (None if optimal_moves is None
                       else [Action(int(_PG_TO_ACTION[m].value)) for m in optimal_moves])
                changed = not np.array_equal(frame, prev)
                observations.append(frame.tolist())
                actions.append(self._encode_step(
                    taken, optimal=opt, phase=phase, changed=changed))
                prev = frame

            # ── epsilon exploration prefix + RESET recovery ──────────────────
            if exploration is not None and schedule is not None:
                won, perturbed = self._explore_prefix(
                    g3, s0, level, observations, actions, schedule, exploration)
                if won is not None:                  # accidental win during prefix
                    return won
                if perturbed:
                    # RESET: restore the initial maze start the greedy plan assumes.
                    g3.set_state(s0)
                    rew, obs, first = g3.observe()
                    rf = _frame(obs["rgb"][0])
                    changed = not np.array_equal(rf, prev)
                    observations.append(rf.tolist())
                    actions.append(self._encode_step(
                        Action(RESET_ACTION), optimal=[Action(RESET_ACTION)],
                        phase="reset", changed=changed))
                    prev = rf

            cur = start_key
            for _step in range(self.step_guard):
                edges = graph.get(cur)
                if not edges:
                    return None                      # dead end (should not happen)
                win_moves = [a for a, nk in edges.items() if nk == "GOAL"]
                moving = {a: nk for a, nk in edges.items() if nk != "GOAL"}

                if win_moves:
                    # The winning move steps onto the goal, but ProcGen terminates+
                    # auto-resets within that step, so the agent-on-goal frame is
                    # never rendered. Synthesize it from the last clean frame: old
                    # player cell -> floor, goal cell -> player. Every winning move
                    # is co-optimal (each wins in one step); sample which to take.
                    win_a = self.rng.choice(win_moves)
                    last = np.array(observations[-1])
                    if (last == _MAZE_GOAL_IDX).any():
                        win_frame = last.copy()
                        win_frame[last == _MAZE_PLAYER_IDX] = _MAZE_FLOOR_IDX
                        win_frame[last == _MAZE_GOAL_IDX] = _MAZE_PLAYER_IDX
                        emit(win_a, win_frame, optimal_moves=win_moves)
                    return observations, actions     # ends ON the goal (WIN frame)
                if not moving:
                    return None
                # STOCHASTIC OPTIMAL: every move whose next cell is one step closer
                # to the goal (min dist-to-goal) is co-optimal -- descending any of
                # them keeps the path a shortest path, so sample uniformly and record
                # the whole set. A perfect (tree) maze has a unique route, so this is
                # usually a single move; the code is the correct general form.
                cand = {m: dist[nk] for m, nk in moving.items() if nk in dist}
                if cand:
                    best = min(cand.values())
                    co = [m for m, dd in cand.items() if dd == best]
                else:
                    co = list(moving)                # no dist info: degrade gracefully
                a = self.rng.choice(co)

                g3.act(np.array([a], dtype=np.int32))
                rew, obs, first = g3.observe()
                emit(a, _frame(obs["rgb"][0]), optimal_moves=co)
                cur = edges[a]
                if schedule is not None:   # keep the episode-wide horizon advancing
                    schedule.advance()     # so exploration stays front-loaded

            return None                              # hit the step cap unsolved
        finally:
            env.close()

    def _explore_prefix(self, g3, s0, level, observations, actions,
                        schedule: EpsilonSchedule, exploration: ExplorationPolicy):
        """Run the episode-wide epsilon prefix as REAL random moves from ``s0``,
        recording each as a ``phase="explore"`` step. Returns ``(won, perturbed)``:
        ``won`` is ``(observations, actions)`` if a random move happens to WIN the
        level (synthesizing the agent-on-goal frame, as the greedy path does) else
        ``None``; ``perturbed`` is True if any real move was taken, so the caller
        knows whether a RESET back to ``s0`` is needed before following the plan."""
        perturbed = False
        while schedule.explore():
            schedule.advance()
            a_act = exploration.action(np.asarray(observations[-1]))
            pg_move = _ACTION_TO_PG.get(a_act.action_id)
            if pg_move is None:
                continue
            g3.act(np.array([pg_move], dtype=np.int32))
            rew, obs, first = g3.observe()
            perturbed = True
            prev = np.array(observations[-1])
            if float(rew[0]) > 0:                    # accidental win mid-prefix
                taken = Action(a_act.action_id)
                if (prev == _MAZE_GOAL_IDX).any():
                    win_frame = prev.copy()
                    win_frame[prev == _MAZE_PLAYER_IDX] = _MAZE_FLOOR_IDX
                    win_frame[prev == _MAZE_GOAL_IDX] = _MAZE_PLAYER_IDX
                    changed = not np.array_equal(win_frame, prev)
                    observations.append(win_frame.tolist())
                    actions.append(self._encode_step(
                        taken, optimal=None, phase="explore", changed=changed))
                return (observations, actions), perturbed
            if bool(first[0]):
                # The episode ended (maze can only time out) and ProcGen has
                # already auto-reset, so ``obs`` is a fresh board. RECORD the step
                # anyway, on the PRE-action frame: gym3's ``ToGymEnv.step`` hands
                # the adapter ``prev_ob`` on any terminal step (interop.py: ``if
                # first[0]: ob = prev_ob``), so an agent that ends the episode sees
                # its last frame unchanged -- and dropping the step instead left
                # the demo one action short of what the game actually does.
                taken = Action(a_act.action_id)
                observations.append(prev.tolist())
                actions.append(self._encode_step(
                    taken, optimal=None, phase="explore", changed=False))
                break                                 # stop; caller RESETs to s0
            f = _frame(obs["rgb"][0])
            taken = Action(a_act.action_id)
            changed = not np.array_equal(f, prev)
            observations.append(f.tolist())
            actions.append(self._encode_step(
                taken, optimal=None, phase="explore", changed=changed))
        return None, perturbed

    def solve_episode(self, seed: int, explore: bool = True):
        """Solve all 7 levels for one base seed (all-or-nothing).

        Mirrors the base ``solve_episode``: build the episode-wide
        ``EpsilonSchedule`` + ``ExplorationPolicy`` once (only when ``explore and
        supports_recovery``) and thread them through every level's recording, so the
        ignorant-then-informed exploration arc spans the whole episode. Recovery is
        ``reset`` (see ``_solve_level``)."""
        do_explore = explore and self.supports_recovery
        schedule = (EpsilonSchedule(self.rng, center=self._explore_center,
                                    jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(), self.rng)
                       if do_explore else None)
        levels: list[dict] = []
        for level in range(N_LEVELS):
            try:
                res = self._solve_level(level, seed, schedule=schedule,
                                        exploration=exploration)
            except Exception:                        # noqa: BLE001 -- a bad seed fails
                return False, levels
            if res is None:
                return False, levels
            observations, actions = res
            levels.append({"level_id": level, "observations": observations,
                           "actions": actions})
        return True, levels


if __name__ == "__main__":
    sys.exit(ProcgenMazeSolver.main())
