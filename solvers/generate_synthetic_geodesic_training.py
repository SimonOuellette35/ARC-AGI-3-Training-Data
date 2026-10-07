"""Generate training data from the toy V2 SYNTHETIC geodesic mazes.

These are the detour-rich, farthest-start optimal-path demonstrations used by
topology_learning/toy_value_topologyV2.py -- exactly the data that forces a value
head to learn GEODESIC (path) distance rather than the Euclidean shortcut. Adding it
to the multi-game corpus injects a strong topology signal (and clean, fixed-size
agent/goal squares, unlike procgen's animated sprites).

There is no arcengine game behind this corpus: the maze grid IS the transition
model. `utils.synthetic.SyntheticMazeGame` is the whole "engine" -- a bundle of independent mazes
(one per level) plus the agent's cell -- so the generator can subclass ``BaseSolver``
like every other game and inherit the full recording harness: the two-stream on-disk
schema (``phase`` / ``changed`` / ``optimal``), the episode-wide EXPLORATION PREFIX,
perturbation BURSTS, STOCHASTIC OPTIMAL sampling, and the CLI.

``supports_recovery = True``: a perfect maze is fully reversible and every open cell
is connected to the goal, so `solve_from` re-plans from wherever a detour left the
agent and no burst can ever strand or kill it -- RESET recovery and burst-undo exist
but never need to fire.

Output is the canonical multi-level episode schema (`BaseSolver.normalize_levels`):

    {
      "game_id": "synthetic_geodesic",
      "levels": [
        {"level_id": 0,
         "observations": [[[c, ...], ...], ...],       # [T, 64, 64] palette idx
         "actions": [{"type": "simple", "index": i,    # the action TAKEN
                      "phase": "expert",               # reset|expert|explore|burst
                      "changed": true,                 # did the frame change?
                      "optimal": [{...}, ...]}, ...]}, # the optimal action SET
        ...
      ]
    }

actions[0] is the RESET that produced obs[0]; actions[t] for t>=1 is the move that took
the agent from obs[t-1] to obs[t]. Each fresh maze is one "level"; 10 mazes are bundled
into one episode (the mazes are independent, like the bundled MiniGrid seeds), and the
mazes are a pure function of the episode seed.

Frames use the maze palette (0 floor, 1 wall, 3 player, 14 goal) at 64x64, ~4px/cell --
the same scale the (capped) procgen_maze renders at, and which the frozen encoder decodes
cleanly (wall IoU 1.0). Movement convention matches the other games: ACTION1 up,
ACTION2 down, ACTION3 left, ACTION4 right.

Usage (run from the repo root):
    python solvers/generate_synthetic_geodesic.py
    python solvers/generate_synthetic_geodesic.py --episodes 1600 --levels 10 --grid 7
    python solvers/generate_synthetic_geodesic.py --no-exploration --noise 0  # pure optimal
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

# Repo root on the path so the maze/demo utils (utils.synthetic) import.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np                                          # noqa: E402
from arcengine import GameAction                            # noqa: E402
from solvers.base_solver import BaseSolver, DriveResult     # noqa: E402
# The ENV itself lives in utils.synthetic, imported by BOTH this generator and
# solver.py's live roll-out. It used to be written twice and the two copies had
# diverged (seeding, level count, movement rule) -- see that module's docstring.
from utils.synthetic import (                                # noqa: E402
    DELTA_ACTION, DELTAS, FRAME_SIZE, GRID_N, N_LEVELS,
    MazeLevel, SyntheticMazeGame, render_game, step_agent)

GAME_ID = "synthetic_geodesic"

# Grid step (row, col) -> the 4-direction GameAction, derived from the shared int
# map so the two cannot drift apart. Looked up by NAME: GameAction does not
# support construction from a value (``GameAction(1)`` raises even though
# ``ACTION1.value == 1``), which is why every generator spells this out.
_DELTA_TO_ACTION: dict[tuple[int, int], GameAction] = {
    delta: getattr(GameAction, f"ACTION{aid}")
    for delta, aid in DELTA_ACTION.items()
}
_DELTAS = DELTAS


# ---------------------------------------------------------------------------
# The "engine" -- `MazeLevel` / `SyntheticMazeGame` / `make_maze` now live in
# utils.synthetic, shared verbatim with solver.py's live roll-out. They were
# duplicated here and had drifted from the eval copy; see that module.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
class SyntheticGeodesicSolver(BaseSolver):
    game_id = GAME_ID
    # A perfect maze is fully reversible and every open cell is connected to the
    # goal, so `solve_from` re-plans from ANY state an exploratory detour reaches.
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, *, levels: int = N_LEVELS, grid_n: int = GRID_N,
                 frame_size: int = FRAME_SIZE,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.n_levels = levels
        self.grid_n = grid_n
        self.frame_size = frame_size

    # ── engine plumbing ──────────────────────────────────────────────────────
    def make_game(self, seed: int) -> SyntheticMazeGame:
        return SyntheticMazeGame(seed, self.n_levels, self.grid_n,
                                 self.frame_size)

    def num_levels(self, game) -> int:
        return len(game.levels)

    def set_level(self, game, level_idx: int) -> None:
        game.set_level(level_idx)

    def reset_level(self, game, level_idx: int, seed: int) -> None:
        """RESET == put the agent back on this maze's start (the maze is static)."""
        self.set_level(game, level_idx)

    def render(self, game) -> np.ndarray:
        return render_game(game)

    def available_actions(self, game) -> list[int]:
        return [1, 2, 3, 4]

    # ── burst undo: the whole mutable state is (cur, agent) ──────────────────
    # The base's native path deepcopies the entire game and rebuilds ``__dict__``;
    # here the mazes are immutable for the episode's lifetime, so a two-tuple is a
    # complete and far cheaper snapshot (bursts and the burst-recovery probe run
    # this per burst step).
    def _game_snapshot(self, game):
        return (game.cur, game.agent)

    def _game_restore(self, game, snap) -> None:
        game.cur, game.agent = snap

    # ── the solver ───────────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int) -> list:
        """Shortest path from the agent's LIVE cell, by descending the geodesic
        field one step at a time (``dist == dist(cur) - 1``)."""
        lvl = game.level
        dist, cur = lvl.dist, game.agent
        d = float(dist[cur])
        if not np.isfinite(d):                               # walled in (unreachable)
            return []
        plan = []
        while cur != lvl.goal:
            nxt = self._descend(lvl, cur)
            if nxt is None:                                  # field is broken -> fail
                return []
            delta, cur = nxt
            plan.append(_DELTA_TO_ACTION[delta])
        return plan

    def optimal_set_from(self, game, level_idx: int, seed: int) -> list | None:
        """Every equally-optimal next move: the directions whose target cell is one
        step closer to the goal. Feeds the base's STOCHASTIC OPTIMAL sampling, so a
        given maze yields many distinct (still perfectly optimal) routes across
        episodes. O(1) -- no game copy / lookahead probe needed."""
        lvl = game.level
        d = float(lvl.dist[game.agent])
        if not np.isfinite(d) or d == 0.0:                   # already on the goal
            return []
        return [_DELTA_TO_ACTION[delta]
                for delta in _DELTAS
                if self._dist_at(lvl, game.agent, delta) == d - 1.0]

    def drive(self, game, action) -> DriveResult:
        """One move against the maze, via the SHARED `step_agent` rule -- the same
        function solver.py's live roll-out steps with, so recorded and evaluated
        dynamics cannot drift. Reaching the goal is the level's WIN; nothing here
        can ever kill the agent."""
        game.agent = step_agent(game.level, game.agent, action.action_id)
        return DriveResult(self.render(game), game.solved, False)

    # ── geodesic-field helpers ───────────────────────────────────────────────
    @staticmethod
    def _dist_at(lvl: MazeLevel, cell: tuple[int, int],
                 delta: tuple[int, int]) -> float:
        r, c = cell[0] + delta[0], cell[1] + delta[1]
        if not (0 <= r < lvl.grid.shape[0] and 0 <= c < lvl.grid.shape[1]):
            return float("inf")
        return float(lvl.dist[r, c])

    def _descend(self, lvl: MazeLevel, cell: tuple[int, int]):
        """One step down the geodesic field: ``(delta, next_cell)`` or None."""
        d = float(lvl.dist[cell])
        for delta in _DELTAS:
            if self._dist_at(lvl, cell, delta) == d - 1.0:
                return delta, (cell[0] + delta[0], cell[1] + delta[1])
        return None

    # ── CLI ──────────────────────────────────────────────────────────────────
    @classmethod
    def build_argparser(cls):
        p = super().build_argparser()
        p.set_defaults(episodes=1600)          # matches the other games' corpus size
        p.add_argument("--levels", type=int, default=10,
                       help="synthetic mazes bundled per episode.")
        p.add_argument("--grid", type=int, default=7,
                       help="maze is (2*grid+1)^2 cells; grid 7 -> 15x15 (~4px/cell "
                            "at 64x64).")
        p.add_argument("--frame-size", type=int, default=64)
        return p

    @classmethod
    def main(cls, argv=None) -> int:
        args = cls.build_argparser().parse_args(argv)
        solver = cls(rng=random.Random(args.seed),
                     burst_prob=args.noise, burst_mean=args.burst,
                     levels=args.levels, grid_n=args.grid,
                     frame_size=args.frame_size)
        out = args.out or (Path("data/training_multi_level") / cls.game_id)
        print(f"maze {2 * args.grid + 1}x{2 * args.grid + 1} cells, "
              f"{args.levels} levels/episode")
        return solver.run(args.episodes, out, start_seed=args.start_seed,
                          max_attempts=args.max_attempts,
                          progress_every=args.progress_every,
                          explore=not args.no_exploration)


if __name__ == "__main__":
    sys.exit(SyntheticGeodesicSolver.main())
