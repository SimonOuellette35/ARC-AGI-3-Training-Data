"""Generate Phase-1 training data for the PuzzleScript game ps:ball_bros.

The harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is the game-specific part: the state key, the goal
heuristic, the skipped levels and the search budget.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_ball_bros",
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
action (post rotation-remap), i.e. the button an agent presses in the rotated
view, so replaying the recorded actions reproduces the recorded frames exactly.

The game
--------
Two balls, one set of arrow keys. A BIG ball (a rigid 2x2 block of the four
quarter-sprites Ball1..Ball4) and a SMALL ball (Ball5) share the four directions;
ACTION (X) swaps which of the two is "active", and only the active one moves. The
Player object is an invisible controller that never moves -- it just donates its
force to whichever balls are active -- so there is nothing on the board that
corresponds to "where the agent is". The win is every exit covered at once: the
2x2 exit block by the matching quarters of the big ball and the round Exit5 by the
small ball. Cover them all and a five-tick animation eats the balls, which is what
the ``no Ball`` win condition actually tests.

The mechanics the levels are built out of, and which of the two balls each one
is aimed at:

  * WALLS block both balls. The big ball is a rigid 2x2, so it needs a 2-wide
    corridor and one blocked quarter freezes all four
    (``[stationary BallA|moving BallA]``).
  * CRATES block both balls, and only the SMALL ball can push one. Crates are
    themselves blocked by walls, closed gates, big balls and other crates.
  * GATES are pipe segments (the ╔ ═ ╝ ║ glyphs) wired into networks by the
    directions drawn on each piece. A closed gate blocks the small ball, crates
    and lasers -- but NOT the big ball, which rolls straight over one. A network
    opens while ANY of its cells holds a ball or a crate, so the recurring puzzle
    is "park the big ball on the pipe, switch, and walk the small ball through
    the hole it opens", and it slams shut the moment the big ball rolls off.
  * EMITTERS fire a white beam that is re-walked from scratch every turn, and
    what stands in it subtracts primaries from it. A crate takes the blue
    (white -> yellow) and the small ball takes the green (white -> magenta), both
    where they stand. The big ball is the odd one and is the mechanic the last
    levels are built on: a beam entering one of its quarters is COPIED onto the
    diagonally opposite quarter with the red taken out, so it lights up the
    NEIGHBOURING row or column rather than recolouring itself -- which is why the
    beam levels come with emitters in adjacent pairs. A RECEIVER opens the gate
    network it touches only for the one beam colour it is painted, and closed
    gates cut beams, which is how one network gates another.

Expert solver
-------------
Primitive-action A* over the real interpreter (`PSExpert`), run up the weight
`LADDER`. Two things make the primitive search -- which drowns on the
sokoban-shaped ps: games -- the right tool here:

  * There is nothing to make macros OUT of. A press moves the active ball exactly
    one cell, and every press changes the world: rolling the big ball across a
    pipe opens and re-closes gates under it, and moving either ball through a beam
    recolours it. There is no "walk there, it does not matter how" the way there
    is when only pushes have side effects.
  * The heuristic is tight enough that the search barely branches. Because each
    press moves the ACTIVE ball only, the presses the big ball needs and the ones
    the small ball needs are disjoint, so

        h = dist(big ball -> its exit) + dist(small ball -> Exit5) + toggles

    is admissible, with each distance a walls-only BFS field (the big ball's over
    positions where its 2x2 fits) and ``toggles`` the 1 press owed when the ball
    that still has ground to cover is the inactive one. That estimate is exact on
    an empty board and only loses to detours around crates and shut gates --
    which is exactly where the ladder's second rung earns its keep.

The state key is the balls and the crates alone. Everything else the levels
contain -- the gate open/closed state, the beams and their colours, the activated
receivers -- is recomputed from scratch by the ``late`` rules every single turn
(``late [Laser]->[]`` then re-emit; ``late [Gate]->[Raise Gate]`` then re-lower),
so it is a pure function of the ball and crate positions and carries no
information of its own. Walls and exits are static, which is why the memo is
``scope_by_level``. This was verified by replaying random walks on all ten levels
and checking that the full non-background grid really is a function of the
narrowed key -- 0 collisions.

Solvable levels
---------------
{0, 1, 2, 3, 4, 9}, at 25 / 46 / 58 / 77 / 58 / 26 moves: every gate-and-crate
level, the crate maze, the first beam level, and the last level -- a 3x3 lattice
of six differently coloured receivers fed by four parallel emitters, which is the
one that exercises the beam mechanic in full. Levels 5-8 are skipped; see
`BallBrosSolver.skip_levels` for which of them are dead and which are only out of
search budget.

The first run of a level pays for its search (seconds for levels 0 and 9, ~40
minutes for the two that need the ladder's second rung) and writes the result to
`PLAN_CACHE`; every run after that, and every `parallelize_generator.py` shard,
reads it back in milliseconds.

Augmentation
------------
Ball Bros' engine state after reset is identical for every seed, so the only
per-(seed, level) variables are the presentation augmentations the adapter owns:
the frame rotation (rotation_k in {0,1,2,3}) plus an independent horizontal and
vertical flip, each with the matching directional action remap. The board has no
gravity and its inputs are screen-relative moves, so a flip is an exact symmetry.
The expert plan is therefore identical across seeds: it is solved once per level
and replayed from cache, and only the presented frames and the recorded
screen-action encoding differ between seeds.

Recovery
--------
Inherited from `PSAStarSolver`: an epsilon-decayed exploration prefix followed by
ONE RESET back to the level's initial state, from which the cached plan replays a
guaranteed win. The RESET is what makes it safe -- a ball shoved into a corner
behind a shut gate is not always re-planable, and the crates make some states
outright dead.

Verified 2026-08-10: 8/8 seeds win all six levels; every recorded action list,
replayed on a FRESH adapter, reproduces its frames byte-for-byte and ends in
GameState.WIN (48/48 level replays); two runs at the same rng seed are
byte-identical; no expert step lacks an optimal target.

Usage (run from the repo root, with the conda interpreter):
    /home/simon/anaconda3/envs/ARC-AGI-3/bin/python \
        solvers/generate_ball_bros_training.py --episodes 200 \
        --out data/training_multi_level/ball_bros
"""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from solvers.common.ps_astar import (PSAStarSolver, PSExpert,  # noqa: E402
                                     restore, snapshot)

GAME_NAME = "Ball_Bros"

#: Where the per-level searches are kept between runs, in the repo's usual
#: ``data/<game>_plans.json`` shape. The plans are seed-independent, so this
#: turns a one-off cost per PROCESS into a one-off cost EVER -- which is what
#: makes `parallelize_generator.py` usable here, since every shard would
#: otherwise redo the same multi-minute searches. Each entry records the level
#: start it was solved from and is ignored when that no longer matches, so
#: editing the level in the .txt cannot silently serve a stale plan. Delete the
#: file to force a fresh search.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "ball_bros_plans.json"

#: (dr, dc) for the four board directions, for the heuristic's BFS fields.
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Distance stand-in for a ball whose exit no walls-only path reaches. Only
#: reachable through a level whose exit is walled off from its ball, i.e. one
#: with no solution at all, so the value never has to be tight -- it just has to
#: not pretend such a state is close to the goal.
_UNREACHABLE = 999

#: ``(weight, node budget)`` rungs, tried in order, where the budget counts
#: EXPANDED nodes rather than seconds -- so which levels solve, and which plan
#: each one gets, is a property of the code and not of how busy the machine was.
#:
#: The first rung solves levels 0, 1, 2 and 9 at near-shortest length (level 2,
#: the slowest of them, needs ~45k nodes). Level 3 does not come out of it at
#: all: it is a crate maze where the big ball's 2-wide corridor is blocked in
#: half a dozen places, so the admissible heuristic is flat across every prefix
#: that clears a crate and w=2 spreads its budget over all of them. At w=5 the
#: search commits to a crate order and finds a 77-move win. That is the whole
#: reason for the ladder: pay for shortest plans where they are cheap, and buy a
#: longer plan rather than no plan where they are not.
LADDER: tuple[tuple[int, int], ...] = ((2, 150_000), (5, 900_000))


class BallBrosExpert(PSExpert):
    """A* over the interpreter, keyed on the balls and crates only.

    See the module docstring for why the key is narrow (gates, beams and
    receivers are re-derived from those positions every turn) and why the
    two-distance heuristic is admissible (a press moves the ACTIVE ball only, so
    the big ball's presses and the small ball's never overlap).
    """

    #: `_key` is dynamic-objects-only: canonical WITHIN a level, not across them.
    scope_by_level = True

    def setup(self) -> None:
        r = self.g.resolve_object_name
        self.wall_ids = set(r("wall"))
        # Ball1 is the big ball's top-left quarter and Exit1 the exit block's
        # top-left cell, so this one pair tracks the whole rigid 2x2.
        self.ball1_ids = set(r("ball1"))
        self.ball1a_ids = set(r("ball1a"))
        self.ball5_ids = set(r("ball5"))
        self.exit1_ids = set(r("exit1"))
        self.exit5_ids = set(r("exit5"))
        self.dyn_ids = set(r("ball")) | set(r("crate"))
        self._level: int | None = None
        self._fields: dict[tuple, tuple[dict, dict]] = {}
        self._disk = self._load_disk()
        self._planned: set[int | None] = set()

    # -- canonical state ----------------------------------------------------
    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    # -- disk-backed plan memo ----------------------------------------------
    def _load_disk(self) -> dict:
        """``{level: {"start": layout, "plan": [...] | None}}``, or empty if
        unreadable. A cache that cannot be parsed is a miss, never a crash."""
        try:
            raw = json.loads(PLAN_CACHE.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:                                   # noqa: BLE001
            return {}

    def _save_disk(self) -> None:
        """Write atomically -- shards started together (`parallelize_generator`)
        would otherwise interleave into a truncated file."""
        try:
            PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PLAN_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self._disk[k] for k in sorted(self._disk)}, indent=1))
            os.replace(tmp, PLAN_CACHE)
        except OSError:
            pass                                       # the cache is optional

    def plan(self, eng, level: int | None = None):
        # `heuristic` needs the level to key its distance fields and only `plan`
        # is handed it, so stash it on the way through.
        self._level = level
        # Only the level START is worth persisting: it is the one state every
        # seed re-plans from, and it is the state of the FIRST call for a level
        # (both `discover_solvable` and `record_level` plan straight after
        # `set_level`). Claim that "first" flag here, ABOVE the two cache
        # lookups, so a mid-episode re-plan -- an epsilon detour, a recovery
        # prefix that wandered -- can never inherit it and overwrite the start
        # entry with a state no later run will ever be in.
        first = level not in self._planned
        self._planned.add(level)

        key = (level, self._key(eng))
        if key in self.cache:
            return self.cache[key]
        layout = sorted(list(cell) for cell in key[1])
        entry = self._disk.get(level)
        if entry is not None and entry["start"] == layout:
            self.cache[key] = entry["plan"]
            return entry["plan"]

        found = super().plan(eng, level)
        if first:
            self._disk[level] = {"start": layout, "plan": found}
            self._save_disk()
        return found

    def _search(self, eng) -> list | None:
        """Walk `LADDER`: the first rung that returns a plan wins.

        `_astar` reads ``self.weight`` / ``self.node_cap``, so a rung is just
        those two values.

        THE ENGINE MUST BE RESTORED BETWEEN RUNGS. `_astar` drives the real
        interpreter and leaves the grid wherever its last `restore` put it --
        some arbitrary state deep in the frontier it gave up on. Only `plan`
        puts the start state back, and that runs after the whole ladder, so
        without this the second rung searches from the first rung's wreckage:
        level 3 came back unsolvable from a generator whose expert had just
        solved it standalone."""
        start = snapshot(eng)
        for weight, cap in LADDER:
            restore(eng, start)
            self.weight, self.node_cap = weight, cap
            found = self._astar(eng)
            if found is not None:
                return found
        return None

    # -- heuristic ----------------------------------------------------------

    def _build_fields(self, eng) -> tuple[dict, dict]:
        """``(big, small)`` BFS distance-to-exit maps over the WALLS ONLY.

        Ignoring crates, gates and the other ball is what makes each field a
        lower bound rather than a guess: everything omitted can be pushed,
        opened or driven away, so no omission can make a real path shorter than
        the field says.
        """
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        wall = self.wall_ids
        blocked = [[bool(cell & wall) for cell in row] for row in grid]
        exits: dict[int, tuple[int, int] | None] = {1: None, 5: None}
        for r in range(h):
            for c in range(w):
                if grid[r][c] & self.exit1_ids:
                    exits[1] = (r, c)
                elif grid[r][c] & self.exit5_ids:
                    exits[5] = (r, c)

        def bfs(start, fits) -> dict:
            if start is None or not fits(*start):
                return {}
            dist = {start: 0}
            queue = deque([start])
            while queue:
                r, c = queue.popleft()
                for dr, dc in _DELTA:
                    nxt = (r + dr, c + dc)
                    if (0 <= nxt[0] < h and 0 <= nxt[1] < w
                            and nxt not in dist and fits(*nxt)):
                        dist[nxt] = dist[(r, c)] + 1
                        queue.append(nxt)
            return dist

        def fits_big(r, c) -> bool:
            # The big ball is at (r, c) when its top-left quarter is: all four
            # cells of the block must be off the walls for the placement to exist.
            return (r + 1 < h and c + 1 < w
                    and not blocked[r][c] and not blocked[r][c + 1]
                    and not blocked[r + 1][c] and not blocked[r + 1][c + 1])

        return (bfs(exits[1], fits_big),
                bfs(exits[5], lambda r, c: not blocked[r][c]))

    def fields(self, eng) -> tuple[dict, dict]:
        # Walls and exits never move in this game, so one build per level is
        # enough; `_level` is the memo key because it is the only thing `plan`
        # knows that distinguishes two boards.
        key = (self._level, len(eng.grid), len(eng.grid[0]))
        cached = self._fields.get(key)
        if cached is None:
            cached = self._fields[key] = self._build_fields(eng)
        return cached

    def heuristic(self, eng) -> int:
        ball1_ids, ball5_ids = self.ball1_ids, self.ball5_ids
        big = small = None
        big_active = False
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & ball1_ids:
                    big = (r, c)
                    big_active = bool(cell & self.ball1a_ids)
                elif cell & ball5_ids:
                    small = (r, c)
        big_f, small_f = self.fields(eng)
        # A ball that is gone has been eaten by the win animation: it is home.
        hb = 0 if big is None else big_f.get(big, _UNREACHABLE)
        hs = 0 if small is None else small_f.get(small, _UNREACHABLE)
        if hb == 0 and hs == 0:
            return 0
        # Exactly one of the two is active, and the inactive one cannot move
        # until an ACTION press swaps them: that press is owed and is not one of
        # the moves already counted.
        toggle = 1 if (hs if big_active else hb) > 0 else 0
        return hb + hs + toggle


class BallBrosSolver(PSAStarSolver):
    game_id = "puzzlescript_ball_bros"
    game_name = GAME_NAME
    expert_cls = BallBrosExpert

    #: Levels never attempted. Skipping them up front is not cosmetic: with the
    #: ladder, proving a level unwinnable costs BOTH rungs, and level 6 alone
    #: burns two hours doing it at every cold startup.
    #:
    #:   5, 8 -- DEAD. The search CLOSES on these (frontier exhausted, well
    #:         inside the node cap: 866s and 121s), so no ball/crate arrangement
    #:         reachable under this interpreter covers all five exits. Level 5
    #:         agrees with a hand-check of the map: its Exit5 sits behind a gate
    #:         that only the cyan-lit ReceiverC opens, and lighting that receiver
    #:         needs the beam's two intervening gate networks held open at the
    #:         same moment -- with only two movable balls, the last one shuts
    #:         before the small ball can step through.
    #:   6, 7 -- OUT OF BUDGET, not shown unwinnable. Level 6 exhausts the
    #:         900k-node second rung without closing (~2h); level 7 did not
    #:         survive the same budget on this machine at all. Both are 11x14+
    #:         boards carrying live beams, where an interpreter step costs ~8ms
    #:         and the frontier snapshots are what run out first.
    skip_levels = frozenset({5, 6, 7, 8})

    #: The expert runs `LADDER` and sets both of these per rung, so these two
    #: are only the values it starts on. Weighting even the first rung is a
    #: small speed-for-length trade rather than a lifeline: on level 0, w=2
    #: returns the same 25-move plan w=1 does, three times faster.
    weight, node_cap = LADDER[0]
    max_steps = 300


if __name__ == "__main__":
    sys.exit(BallBrosSolver.main())
