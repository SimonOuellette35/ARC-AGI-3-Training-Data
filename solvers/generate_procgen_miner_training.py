"""Generate Phase-1 training data for the ProcGen Miner game (procgen_miner).

Drives the ProcGen "miner" environment wrapped by ``adapters/procgen_adapter.py``.
The output schema, action encoding and WIN-filter CLI live in ``BaseSolver``; this
module supplies the bespoke expert and a ``solve_episode`` override (the native
record/replay loop does not fit -- see below).

The game (Boulderdash)
----------------------
A grid of dirt (10x10 in ``easy`` mode, 20x20 in ``hard``) holding boulders, gems
and one exit. Walking digs the dirt away; an unsupported boulder or gem falls one
cell per tick and rolls sideways off another rounded object; a boulder can be
pushed horizontally into an EMPTY cell only. Collect every gem (3 easy / 12 hard,
+1 reward each) and then step into the exit for the +10 completion bonus. The
miner dies under a falling boulder -- and, just as often, simply *entombs* itself:
tunnelling under a boulder field drops the boulders in behind it until no move
changes anything. That, not death, is the failure mode this expert has to plan
around.

WIN is ``reward >= 9`` -- NOT ``reward > 0``
--------------------------------------------
Two ProcGen quirks make the naive "terminated and reward > 0" test wrong here,
and both were caught producing trajectories that end in something other than a win:

  * gems pay +1 WITHOUT ending the episode, so an episode that ends on the same
    step a gem is picked up (crushed by the boulder the gem was holding up, or
    truncated at the 1000-step timeout) reports ``terminated, reward == 1``. Only
    the +10 completion bonus means "won", hence `WIN_REWARD`.
  * ``gems_needed`` is not armed until the first ``game_step``, so on ~4% of the
    curated seeds the miner starts next to the exit and the level can be won with
    ONE keypress, gems still on the board. The expert never uses that exploit:
    the exit leg is only ever searched once the board holds no gems, so the
    recorded trajectory is always the real thing.

Expert solver (engine-blackbox, leg DFS)
----------------------------------------
Boulder physics plus irreversible digging make a hand-written forward model a
liability, and the state space (which dirt is dug x where every boulder sits) is
far too large for a flat search. So the expert drives the REAL dynamics --
``ProcgenGym3Env`` exposes ``get_state``/``set_state`` -- and structures the
search the way the game is structured: one A* *leg* per gem, then one leg to the
exit, with a DFS over WHICH gem to take next so a leg that strands the miner is
backtracked instead of failing the level. A leg's heuristic is a Dijkstra field
over the decoded grid in which a boulder costs `BOULDER_COST` rather than
blocking -- boulders are obstacles, not walls -- plus a penalty on states where
the miner has walled itself into a small pocket (`_Search.mobility`).

The field depends only on the BOULDER MASK (dirt and empty floor both cost 1 to
enter, since digging is free), and boulders move rarely, so it is cached by that
mask and is essentially free per node.

Frames are decoded to the grid by colour, one cell at a time, binning each pixel
by its CENTRE ((y+0.5)*W/64): ProcGen sprites are ~6px in a 6.4px cell and start
a fraction of a pixel early, so binning by the pixel's top-left corner smears the
miner across two rows. The miner's own sprite carries a cyan lamp and a dark
outline, which read as a gem and as the exit, so its cell is masked out of those
two channels.

Multi-env multilevel model (custom ``solve_episode``)
-----------------------------------------------------
ONE procgen_miner base seed is a 7-level game, but -- unlike the native
ARCBaseGame model where one instance advances through levels in place -- EACH
level is a SEPARATE single-level ProcGen env built by ``_make_level_env(level,
base_seed)`` from the curated absolute seed pools in
``adapters.procgen_adapter._LEVEL_SEEDS_ABS["miner"]`` (level L uses
pool[base_seed % len(pool)]; levels 0-3 are ``easy``, 4-6 ``hard``). A base seed
is kept only if ALL levels solve. Because every level is a fresh env with its own
close(), ``solve_episode`` is overridden rather than driving through the base's
``record_level``/``set_level`` template.

The env is built with ``ProcgenGym3Env`` directly rather than through
``gym.make(..., render_mode="rgb_array")`` as the adapter does: the human-view
render costs 2.5ms per step against 0.08ms without it (30x), and the ``rgb``
observation the frames are built from is byte-identical either way.

Recovery: RESET (``recovery_mode = "reset"``)
---------------------------------------------
Miner is IRREVERSIBLE -- dug dirt does not come back, a collected gem is gone, a
dropped boulder cannot be lifted -- so a plan cannot be resumed after random
flailing. The episode therefore opens with the exploration prefix (real random
moves, recorded ``phase="explore"``), then ONE RESET restores the level's initial
state (``set_state(s0)``, exactly what the plan was computed from) and the plan
replays from there: precisely a human who flails, gets stuck or crushed, hits
reset, and only then solves it.

Winning-step note: on completion ProcGen terminates and IMMEDIATELY auto-resets
the level, so the engine never RENDERS the "miner on the exit" frame. It is
synthesised from the last clean frame by ``_miner_win_frame`` -- which lives in
the ADAPTER, and is used by both, so the recorded final frame is byte-identical to
the one a live agent is shown when it wins.

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_procgen_miner_training.py --episodes 200 \
        --out data/training_multi_level/procgen_miner
"""

from __future__ import annotations

import heapq
import sys
import time
from pathlib import Path

# Repo root (parent of solvers/) -- that's where adapters/ + arcengine live.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from procgen import ProcgenGym3Env  # noqa: E402

from arcengine import GameAction  # noqa: E402
from adapters.procgen_adapter import (  # noqa: E402
    _rgb_to_arc_frame, _postprocess_miner, _miner_win_frame, _LEVEL_SEEDS_ABS)
from utils.explore import (  # noqa: E402
    Action, ExplorationPolicy, ExplorationPrefix)
from solvers.base_solver import BaseSolver  # noqa: E402

GAME_NAME = "miner"
N_LEVELS = len(_LEVEL_SEEDS_ABS[GAME_NAME])          # 7

# Grid size per ProcGen difficulty mode. The whole level is always on screen
# (miner does not scroll), so 64px / WORLD_DIM is the cell pitch.
WORLD_DIM = {"easy": 10, "hard": 20}

#: A terminal step counts as a WIN only at (at least) the completion bonus; +1
#: gem rewards are NOT wins. See the module docstring.
WIN_REWARD = 9.0

# Decoded tile codes.
DIRT, EMPTY, BOULDER, DIAMOND, PLAYER, EXIT = range(6)

# ProcGen Discrete(15) move ints (joystick 3x3): up=5, down=3, left=1, right=7.
MOVES = (5, 3, 1, 7)
_PG_TO_ACTION = {
    5: GameAction.ACTION1,   # up
    3: GameAction.ACTION2,   # down
    1: GameAction.ACTION3,   # left
    7: GameAction.ACTION4,   # right
}
# Inverse, keyed by the GameAction VALUE (1..4): what the exploration policy
# emits (an ``Action`` over the four move ids) -> the ProcGen joystick int.
_ACTION_TO_PG = {int(a.value): pg for pg, a in _PG_TO_ACTION.items()}

#: Cost of ENTERING a boulder cell in the heuristic field. A boulder is an
#: obstacle, not a wall: it can be pushed sideways into empty space or undermined
#: so it falls out of the way, so treating it as impassable leaves the search with
#: no gradient at all toward a gem walled in by boulders (which is most of them).
BOULDER_COST = 4
#: Distance clamp -- anything past this is "far", and comparing two far cells by
#: their exact cost only adds noise to the tie-breaking.
FAR = 99
#: Below this many walkable cells the miner has bricked itself into a pocket;
#: states are penalised toward this bound and pruned outright at <= 1.
MOBILITY_CAP = 16


# ---------------------------------------------------------------------------
# Env helpers
# ---------------------------------------------------------------------------

def _level_mode_and_seed(level: int, base_seed: int) -> tuple[str, int]:
    """(distribution_mode, start_level) for (level, base_seed) -- identical seed
    selection to ``ProcGenAdapter.set_level`` via the curated absolute pools."""
    mode, pool = _LEVEL_SEEDS_ABS[GAME_NAME][level]
    return mode, pool[base_seed % len(pool)]


def _make_level_env(level: int, base_seed: int):
    """Create a single-level miner env for (level, base_seed)."""
    mode, start = _level_mode_and_seed(level, base_seed)
    return ProcgenGym3Env(num=1, env_name=GAME_NAME, start_level=start,
                          num_levels=1, distribution_mode=mode,
                          use_backgrounds=False)


def _arc_frame(rgb: np.ndarray) -> np.ndarray:
    """64x64 palette-index frame, post-processed identically to ProcGenAdapter
    (_rgb_to_arc_frame -> _postprocess_miner) so the SAVED frames match what
    solver_client.py feeds the encoder/dynamics stack at inference time."""
    return _postprocess_miner(rgb, _rgb_to_arc_frame(rgb))


# ---------------------------------------------------------------------------
# Frame -> grid decoding
# ---------------------------------------------------------------------------

_CELL_INDEX: dict[int, np.ndarray] = {}


def _cell_index(W: int) -> np.ndarray:
    """Flat pixel -> cell index map, binning each pixel by its CENTRE.

    ProcGen draws a ~6px sprite in a 6.4px cell starting a fraction of a pixel
    before the cell boundary, so binning by the pixel's top-left corner
    (``y*W//64``) puts that first row in the cell ABOVE and smears every sprite
    across two cells."""
    if W not in _CELL_INDEX:
        idx = ((2 * np.arange(64) + 1) * W) // 128
        _CELL_INDEX[W] = (idx[:, None] * W + idx[None, :]).ravel()
    return _CELL_INDEX[W]


def _decode(rgb: np.ndarray, W: int) -> tuple[np.ndarray, np.ndarray]:
    """(W, W) tile grid + per-cell dark-pixel counts from a 64x64x3 RGB frame.

    Colour keys: dirt is brown, boulders and the exit's frame are blue-gray, gems
    are cyan, the miner is teal-green, the exit's core is (51,51,51) dark. The
    miner's sprite contains BOTH a cyan lamp and a dark outline, so its cell is
    removed from the gem and exit channels before they are counted -- otherwise
    the miner reads as a gem sitting on the exit and both the gem tally (the leg
    goal test) and the exit's position go wrong.

    The dark counts are returned as well: the exit is the cell holding the most
    dark pixels, which is how it is told apart from the incidental dark shading
    inside boulder sprites."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    cm = _cell_index(W)
    n = W * W

    def count(mask: np.ndarray) -> np.ndarray:
        return np.bincount(cm[mask.ravel()], minlength=n)

    total = np.bincount(cm, minlength=n).astype(np.float32)
    m_dirt = (r > 110) & (r - b > 50)
    m_gray = (np.abs(g - b) < 14) & (b >= r - 20) & (r > 100) & (b > 100)
    m_cyan = (b > 170) & (b - r > 50) & (g > 140)
    m_teal = (g > r + 40) & (g > 110) & (b > r + 20) & (g > b)
    m_dark = (np.abs(r - 51) < 14) & (np.abs(g - 51) < 14) & (np.abs(b - 51) < 14)

    teal = count(m_teal)
    p_cell = int(np.argmax(teal)) if teal.max() > total.max() * 0.15 else -1
    if p_cell >= 0:
        keep = (cm != p_cell).reshape(64, 64)
        m_cyan = m_cyan & keep
        m_dark = m_dark & keep

    dirt, gray, cyan, dark = (count(m_dirt), count(m_gray),
                              count(m_cyan), count(m_dark))
    grid = np.full(n, EMPTY, dtype=np.int8)
    grid[dirt > total * 0.35] = DIRT
    grid[gray > total * 0.35] = BOULDER
    grid[dark > total * 0.18] = EXIT
    grid[cyan > total * 0.25] = DIAMOND
    if p_cell >= 0:
        grid[p_cell] = PLAYER
    return grid.reshape(W, W), dark.reshape(W, W)


def _player_cell(grid: np.ndarray) -> tuple[int, int] | None:
    ys, xs = np.nonzero(grid == PLAYER)
    return (int(ys[0]), int(xs[0])) if ys.size else None


def _field(boulders: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Cost-to-``target`` for every cell; entering a boulder costs `BOULDER_COST`,
    any other cell 1. Depends ONLY on the boulder mask -- dirt and dug floor both
    cost 1, since digging is free -- which is what makes it cacheable."""
    W = boulders.shape[0]
    dist = np.full((W, W), 1 << 20, dtype=np.int32)
    dist[target] = 0
    queue = [(0, target[0], target[1])]
    while queue:
        c, j, i = heapq.heappop(queue)
        if c > dist[j, i]:
            continue
        for nj, ni in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)):
            if 0 <= nj < W and 0 <= ni < W:
                nc = c + (BOULDER_COST if boulders[nj, ni] else 1)
                if nc < dist[nj, ni]:
                    dist[nj, ni] = nc
                    heapq.heappush(queue, (nc, nj, ni))
    return dist


# ---------------------------------------------------------------------------
# Expert: leg DFS over the real ProcGen dynamics
# ---------------------------------------------------------------------------

class _Search:
    """Engine-blackbox planner for ONE miner level.

    Owns a ProcGen env it rewinds with ``set_state``; every node it expands is a
    real step of the real game, so any plan it returns is genuine by construction
    (the recorder replays and re-verifies it anyway)."""

    def __init__(self, env, mode: str, rng) -> None:
        self.env = env
        self.W = WORLD_DIM[mode]
        self.rng = rng
        self.nodes = 0
        self._cache: dict = {}
        _, obs, _ = env.observe()
        grid, dark = _decode(obs["rgb"][0], self.W)
        self.exit = tuple(int(x) for x in np.unravel_index(np.argmax(dark),
                                                           dark.shape))
        self.s0 = env.get_state()

    # ── heuristic pieces ────────────────────────────────────────────────────
    def h(self, grid: np.ndarray, target: tuple[int, int]) -> int | None:
        """Estimated cost from the miner to ``target``, or None if the miner is
        not visible in the frame (it is briefly hidden the step it is crushed)."""
        key = (target, (grid == BOULDER).tobytes())
        field = self._cache.get(key)
        if field is None:
            field = _field(grid == BOULDER, target)
            self._cache[key] = field
        cell = _player_cell(grid)
        if cell is None:
            return None
        return min(int(field[cell]), FAR)

    def mobility(self, grid: np.ndarray, cap: int = MOBILITY_CAP) -> int:
        """How many cells the miner can still walk to WITHOUT moving a boulder
        (dirt counts -- it is dug through freely), capped at ``cap``.

        Entombment is the classic Boulderdash way to lose a level without dying,
        and it is invisible to a distance field that prices boulders as merely
        expensive; an ordinary tunnel has its whole dug-out back open, so only a
        genuinely sealed pocket comes in under the cap."""
        W = self.W
        src = _player_cell(grid)
        if src is None:
            return 0
        seen = {src}
        stack = [src]
        while stack and len(seen) < cap:
            j, i = stack.pop()
            for nj, ni in ((j - 1, i), (j + 1, i), (j, i - 1), (j, i + 1)):
                if (0 <= nj < W and 0 <= ni < W and (nj, ni) not in seen
                        and grid[nj, ni] != BOULDER):
                    seen.add((nj, ni))
                    stack.append((nj, ni))
        return len(seen)

    # ── one leg: collect any gem, or (with no gems left) reach the exit ─────
    def leg(self, state, target, want_win: bool, max_nodes: int,
            weight: float = 2.0):
        """A* on the real engine toward ``target``.

        Goal is "the gem tally dropped" for a gem leg and "the level completed"
        for the exit leg -- the miner may well collect a DIFFERENT gem than the
        one steering the heuristic (a gem it undermined can fall into its lap),
        and that is a fine outcome, so the goal test counts gems rather than
        checking arrival. Returns the action list, or None if the leg is not
        found within ``max_nodes``."""
        env = self.env
        env.set_state(state)
        _, obs, _ = env.observe()
        grid0, _ = _decode(obs["rgb"][0], self.W)
        gems0 = int((grid0 == DIAMOND).sum())
        h0 = self.h(grid0, target)
        if h0 is None:
            return None
        queue = [(weight * h0, 0, 0, [], state)]
        seen = {grid0.tobytes(): 0}
        used = 0
        counter = 0
        moves = list(MOVES)
        while queue and used < max_nodes:
            _f, g, _c, path, snap = heapq.heappop(queue)
            # STOCHASTIC OPTIMAL: shuffling the expansion order makes equal-cost
            # routes come out in a different order every episode, so one board
            # yields many distinct (still shortest-found) trajectories.
            self.rng.shuffle(moves)
            for a in moves:
                env.set_state(snap)
                env.act(np.array([a], dtype=np.int32))
                rew, obs, first = env.observe()
                used += 1
                self.nodes += 1
                if bool(first[0]):
                    # The episode ended: a completion (the only reward that
                    # reaches WIN_REWARD), or death / the 1000-step timeout.
                    if want_win and float(rew[0]) >= WIN_REWARD:
                        return path + [a]
                    continue
                grid, _ = _decode(obs["rgb"][0], self.W)
                if not want_win and int((grid == DIAMOND).sum()) < gems0:
                    return path + [a]
                key = grid.tobytes()
                ng = g + 1
                if key in seen and seen[key] <= ng:
                    continue
                seen[key] = ng
                h = self.h(grid, target)
                if h is None:
                    continue
                mob = self.mobility(grid)
                if mob <= 1:
                    continue               # entombed: no move can change anything
                counter += 1
                heapq.heappush(queue, (ng + weight * (h + MOBILITY_CAP - mob),
                                       ng, counter, path + [a], env.get_state()))
        return None

    # ── the level: DFS over which gem to take next ──────────────────────────
    def solve(self, *, node_budget: int = 200_000, leg_nodes: int = 4000,
              branch: int = 3, time_budget: float = 20.0):
        """Plan the whole level: one leg per gem, then the exit.

        A greedy nearest-gem order solves most levels outright, but "nearest" is
        exactly the order that walks the miner into a boulder field and seals it
        in, so a failed leg backtracks and the next-nearest gem is tried instead
        -- a DFS over gem orderings whose successful path is the plan."""
        deadline = time.time() + time_budget
        stack = [([], self.s0, set())]
        while stack and self.nodes < node_budget and time.time() < deadline:
            plan, state, tried = stack[-1]
            self.env.set_state(state)
            _, obs, _ = self.env.observe()
            grid, _ = _decode(obs["rgb"][0], self.W)
            gems = [(int(j), int(i)) for j, i in zip(*np.nonzero(grid == DIAMOND))]
            if gems:
                if _player_cell(grid) is None:
                    stack.pop()
                    continue
                order = sorted(gems, key=lambda c: (self.h(grid, c) or FAR))
                candidates = [c for c in order[:branch] if c not in tried]
            else:
                candidates = [] if self.exit in tried else [self.exit]
            if not candidates:
                stack.pop()
                continue
            target = candidates[0]
            tried.add(target)
            path = self.leg(state, target, not gems, leg_nodes)
            if path is None:
                continue
            if not gems:
                return plan + path
            stack.append((plan + path, self.env.get_state(), set()))
        return None


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class ProcgenMinerSolver(BaseSolver):
    game_id = "procgen_miner"
    # Miner is IRREVERSIBLE (dug dirt, taken gems, dropped boulders), so a plan
    # cannot be resumed from wherever exploration left the level. Recovery is
    # therefore RESET: explore, then restore the initial state the plan was
    # computed from (``set_state(s0)``) and follow it. See the module docstring.
    supports_recovery = True
    recovery_mode = "reset"
    #: Perturbation bursts need a solver that re-plans from a live state; a
    #: reset-mode solver is state-blind by construction, and the base's burst path
    #: is disabled for it anyway.
    default_noise = 0.0
    step_guard = 1000                      # ProcGen's own per-episode step budget

    #: Per-level planning budget. Solved levels come back in ~0.1s (easy) to ~1s
    #: (hard); anything still searching after this is a level whose gems are
    #: walled in badly enough that the DFS is thrashing, and the seed is cheaper
    #: to abandon than to finish.
    time_budget: float = 20.0

    # make_game / solve_from are unused: solve_episode is fully overridden because
    # each of the 7 levels is a distinct env with its own dynamics.

    def available_actions(self, game=None) -> list[int]:
        """The four movement keys -- procgen has no click, and ACTION5/6/7 all map
        to the ProcGen no-op, so exploration covers up/down/left/right."""
        return [int(a.value) for a in _PG_TO_ACTION.values()]

    # ── one level ───────────────────────────────────────────────────────────
    def _solve_level(self, level: int, base_seed: int, *,
                     schedule: ExplorationPrefix | None = None,
                     exploration: ExplorationPolicy | None = None):
        """Plan and record one level. Returns ``(observations, actions)`` on a
        verified win, or None.

        With ``exploration``/``schedule`` supplied the recording opens with the
        episode-wide exploration prefix (real random moves), then ONE RESET
        restores ``s0`` before the plan replays -- the ``reset`` recovery
        paradigm, driven through ``BaseSolver._reset_prefix``. The plan is
        computed from ``s0`` and the RESET returns to exactly ``s0``, so the level
        still wins deterministically however far the detour went."""
        mode, _start = _level_mode_and_seed(level, base_seed)
        env = _make_level_env(level, base_seed)
        try:
            search = _Search(env, mode, self.rng)
            plan = search.solve(time_budget=self.time_budget)
            if plan is None:
                return None
            s0 = search.s0

            env.set_state(s0)
            _, obs, _ = env.observe()
            rgb = obs["rgb"][0]
            observations: list = [_arc_frame(rgb).tolist()]
            actions: list = [self._reset_step()]
            prev = np.asarray(observations[0])
            level_won = False
            obs_holder = [rgb]         # the RGB behind observations[-1]

            def step(pg_move: int):
                """One real move -> (arc frame, terminal). ``terminal`` is "win"
                (the synthesized on-the-exit frame), "over" (death / timeout), or
                None.

                BOTH terminal frames are built from the PREVIOUS observation, not
                the one ProcGen hands back, and for the same reason: ProcGen
                auto-resets inside the terminal step, so its observation belongs
                to a fresh board. gym3's ``ToGymEnv.step`` -- the layer the adapter
                sits on -- papers over that by returning the pre-action
                observation on any terminal step (``interop.py``: ``if first[0]:
                ob = prev_ob``), so what a live agent actually sees when it dies is
                the frame it moved FROM, unchanged. Record that, or the demo
                diverges from the game at exactly the steps recovery is about."""
                nonlocal level_won
                before_rgb = obs_holder[0]
                env.act(np.array([pg_move], dtype=np.int32))
                rew, ob, first = env.observe()
                if bool(first[0]):
                    if float(rew[0]) >= WIN_REWARD:
                        level_won = True
                        return _miner_win_frame(before_rgb), "win"
                    return _arc_frame(before_rgb), "over"
                obs_holder[0] = ob["rgb"][0]
                return _arc_frame(obs_holder[0]), None

            def drive_explore(action: Action):
                pg = _ACTION_TO_PG.get(action.action_id)
                if pg is None:                 # ACTION5/6/7 -> ProcGen no-op
                    pg = 4
                return step(pg)

            def reset_to_level():
                env.set_state(s0)
                _, ob, _ = env.observe()
                obs_holder[0] = ob["rgb"][0]
                return _arc_frame(obs_holder[0])

            prev = self._reset_prefix(
                schedule=schedule, exploration=exploration, prev=prev,
                drive_explore=drive_explore, reset_to_level=reset_to_level,
                record_obs=lambda f: observations.append(np.asarray(f).tolist()),
                record_act=actions.append)
            if level_won:
                # An exploratory move happened to finish the level -- an honest
                # win, and the level is over, so the plan never runs.
                return observations, actions

            for pg_move in plan:
                taken = Action(int(_PG_TO_ACTION[pg_move].value))
                frame, terminal = step(pg_move)
                observations.append(np.asarray(frame).tolist())
                actions.append(self._encode_step(
                    taken, optimal=[taken], phase="expert",
                    changed=not np.array_equal(frame, prev)))
                prev = np.asarray(frame)
                if terminal == "win":
                    return observations, actions
                if terminal == "over":
                    return None                # the plan died: discard the seed
                if schedule is not None:
                    schedule.advance()         # keep exploration episode-wide
            return None                        # plan ran out without completing
        finally:
            env.close()

    def solve_episode(self, seed: int, explore: bool = True):
        """Solve all 7 levels for one base seed (all-or-nothing).

        Mirrors the base ``solve_episode``: build the episode-wide
        ``ExplorationPrefix`` + ``ExplorationPolicy`` once (only when ``explore
        and supports_recovery``) and thread them through every level's recording,
        so the explore-then-exploit arc spans the whole episode."""
        do_explore = explore and self.supports_recovery
        schedule = (ExplorationPrefix(self.rng, center=self._explore_center,
                                      jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(), self.rng)
                       if do_explore else None)
        levels: list[dict] = []
        for level in range(N_LEVELS):
            try:
                res = self._solve_level(level, seed, schedule=schedule,
                                        exploration=exploration)
            except Exception:                  # noqa: BLE001 -- a bad seed fails
                return False, levels
            if res is None:
                return False, levels
            observations, actions = res
            levels.append({"level_id": level, "observations": observations,
                           "actions": actions})
        return True, levels


if __name__ == "__main__":
    sys.exit(ProcgenMinerSolver.main())
