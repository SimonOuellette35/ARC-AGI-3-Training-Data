"""Generate Phase-1 training data for the PuzzleScript game ps:back_home
(Le Slo's "Back home").

The whole harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: its name, its pruned action
set, its state key, its goal heuristic and its search budget.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_back_home",
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
A side-on gravity climber. G has lost her keys and has to get back up to her
window: the win condition is `All Target on Player`, i.e. stand on the single
Target cell. The movement vocabulary is four keys and two mechanics:

  * **walk** -- LEFT/RIGHT move one cell; **climb** -- UP/DOWN move one cell as
    well, but only usefully on a `Stair`, because
  * **gravity** -- ``late down [Player no stair | no Wall no Crates no stair] ->
    [PlayerF | ] again`` turns the player into the falling variant `PlayerF`
    whenever the cell below is empty, and the `again` loop then drops it
    cell-by-cell until something catches it. A fall of ANY height therefore
    resolves inside the turn that started it. A player standing on the top cell
    of a ladder is held by the stair BELOW it, which is how you step off a ladder
    onto the target.

Crates are the second mechanic and they are stranger than a sokoban's.

  * **Pushing.** ``[ > Player | Crates ] -> [ > Player | > Crates ]`` is the
    ordinary push, and the Startloop chain rules push a whole row of crates at
    once (``[ > Crate | Crate ]``), cancelling the entire chain via the `CrateN`
    marker if the far end is against a wall. But the two rules ABOVE it,
    ``right [ right Player stair | Crate ] -> [ Player stair | right Crate ]``
    and its left twin, fire first: pushing FROM A STAIR shoves the crate and
    leaves the player standing on the ladder. That is the only way to push a
    crate without following it.
  * **Magnetism.** Crates fall (``late down [CrateL no stair | ...] ->
    [CrateNL | ] again``) unless they rest on a wall/crate/player, sit on a
    stair, or -- the surprise -- TOUCH another crate in any direction
    (``late [ CrateL | crates ] -> [ Crate | crates ]``, which is undirected and
    so matches all four). Crates therefore clump into rigid floating rafts, and
    detaching one from a raft drops it.

`Crate`/`CrateL` (and the transient `CrateN`/`CrateNL`) are the SAME crate in
different bookkeeping states -- "stuck" vs "loose" -- recomputed from scratch by
those `late` rules at the end of every turn, so at rest the variant is a pure
function of the crate positions. It is not quite invisible: the sprites differ in
their four corner pixels (green vs crimson), which is the game telling the player
which crates are about to fall.

Pruned action set
-----------------
`BackHomeExpert.directions` is only ``up / down / left / right``. ACTION (X) is a
dead key -- no rule in the game mentions it -- and dropping it cuts the branching
factor from 5 to 4 for free. Verified empirically as well as by reading the rules:
over 5200 reachable states across all 13 levels (random walks), an ACTION press
never moved the player or a crate. (It does re-settle the crate variant flags on
the very first turn after a level loads, which is why the check is on POSITIONS
and not on the raw grid.)

DOWN is kept, unlike the other gravity ps: game (`generate_atlas_shrank_training`,
where gravity subsumes it): here the player rests on ladders, from which the only
way down is to press it.

The state key
-------------
`_key` is the player cell plus the crate cells, with the four crate variants
COLLAPSED to one symbol -- and `scope_by_level` is therefore set, since walls,
stairs and the target are static per level but differ between levels.

Collapsing the variants is what the paragraph above buys: without it the state a
level loads in (crates parsed as `Crate`) and the identical position reached one
turn later (the same crates re-flagged `CrateL`) are two different keys for one
position, so the plan memoized at the start state would not be found again.
Because the variant is recomputed from the positions every turn, the collapsed
key is still Markov -- verified by grouping the reachable states of all 13 levels
by collapsed key and checking that every member of a group has identical
collapsed successors under all four directions.

The heuristic (why Manhattan is not good enough)
------------------------------------------------
`BackHomeExpert` precomputes, ONCE per level, a *fall-is-free* distance field from
the Target over the level's non-wall cells (a 0-1 BFS): moving sideways or
climbing one cell costs 1, descending one row costs 0, because a fall of any
height is a single turn. `heuristic` is then a table lookup at the player's cell.

Plain Manhattan distance both walks through walls and prices a six-row drop at 6
when it costs 1 turn, which on these tall shaft-shaped levels sends A* into
pockets that are geometrically near the window and topologically far from it. The
field respects the walls and prices the falls. It ignores crates and stairs
entirely -- it will happily climb a bare wall -- so it stays an optimistic lower
bound: no turn can move the player further than one step plus a free descent, so
the plans returned at ``weight == 1`` are genuinely shortest.

Solvable levels
---------------
All 13, optimally, at ~7 minutes of one-time search for the whole game:

    level  0   17 moves       81 nodes        level  7   63 moves    81716 nodes
    level  1   21 moves      196 nodes        level  8   38 moves   104228 nodes
    level  2   17 moves      831 nodes        level  9   46 moves    48123 nodes
    level  3   39 moves     8604 nodes        level 10   71 moves   103971 nodes
    level  4   20 moves     2479 nodes        level 11   36 moves    94772 nodes
    level  5   25 moves     1616 nodes        level 12   26 moves    25624 nodes
    level  6   48 moves    30695 nodes

`node_cap` is ~4x the worst of those, so it is headroom rather than a tuned fit.
There are no `skip_levels`: the interpreter reproduces every level faithfully and
the search reaches all of them.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps); only the presentation augmentation varies per (seed, level): the frame
rotation (rotation_k in {0,1,2,3}). Back home takes no flip augmentation -- a
`flipud` would invert gravity -- and no recolor. The expert plan is therefore
seed-independent: solved once per level, cached, and replayed per seed with that
seed's rotation-remapped screen actions.

A rotated Back home is a genuinely different-looking game -- gravity points
sideways or up on screen, and the ladders run across it -- which is exactly the
point of the augmentation: the agent has to read "which way is down" off the
frame rather than assume it.

Usage (run from the repo root):
    python solvers/generate_back_home_training.py --episodes 200 \
        --out data/training_multi_level/puzzlescript_back_home
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSExpert  # noqa: E402

GAME_NAME = "Back_home"

#: "unreachable" in the distance field. Kept well below ``1 << 30`` so that
#: ``weight * INF`` stays a small int inside the A* priority queue.
INF = 1 << 20


class BackHomeExpert(PSExpert):
    """A* over the real engine, guided by a fall-is-free distance-to-target field.

    See the module docstring for why ``directions`` drops ACTION, why `_key`
    collapses the crate variants, and why the field beats Manhattan.
    """

    #: ACTION is not mentioned by any rule in the game; the other four all matter
    #: (DOWN is how you climb back down a ladder).
    directions = ["up", "down", "left", "right"]

    #: `_key` is player + crates only -- canonical within a level, but not across
    #: levels, whose walls, stairs and target differ.
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        # The or-groups the game itself declares, so the variants stay in sync
        # with the rules: ``Players = Player or PlayerF``, ``Crates = Crate or
        # CrateN or CrateL or CrateNL``, ``Walls = Wall or WallCL or ...``.
        self.player_ids = set(g.resolve_object_name("players"))
        self.crate_ids = set(g.resolve_object_name("crates"))
        self.wall_ids = set(g.resolve_object_name("walls"))
        self.target_ids = set(g.resolve_object_name("target"))
        #: level -> distance field, built on first use.
        self._fields: dict[int | None, list[list[int]]] = {}
        self._dist: list[list[int]] | None = None

    def _key(self, eng) -> frozenset:
        # One symbol per crate regardless of variant (see "The state key" in the
        # module docstring): the variant is recomputed from the positions by the
        # `late` rules every turn, so it carries no state of its own -- but it
        # DOES differ between a freshly loaded level and the same position one
        # turn later, which would otherwise cost the start-state plan its cache
        # hit on every seed.
        crates, players = self.crate_ids, self.player_ids
        out = []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & crates:
                    out.append((r, c, 0))
                if cell & players:
                    out.append((r, c, 1))
        return frozenset(out)

    # -- the fall-is-free distance field ------------------------------------
    def _distance_field(self, eng) -> list[list[int]]:
        """0-1 BFS from the Target over non-wall cells, in units of TURNS.

        Edge costs are those of the forward move ``a -> b`` and the search runs on
        the REVERSED graph, so ``dist[a]`` is the cost of getting from ``a`` to the
        target. There are three kinds of edge, one per thing a turn can do to the
        player:

          * ``a`` directly above ``b`` -> cost **0**. Gravity is an `again` loop,
            so a drop of any height resolves inside the turn that started it and a
            column of free cells is free to descend. (A deliberate DOWN press
            costs a turn, but pricing the descent at 0 only ever under-estimates,
            which is what an admissible heuristic needs.)
          * ``a`` beside ``b`` -> cost **1** (a walk).
          * ``a`` directly below ``b`` -> cost **1** (a climb).

        Stairs and crates are ignored: the field climbs bare walls and walks
        through crate rafts. Both omissions only remove constraints, so no real
        turn can ever cost less than the field says and the plans returned at
        ``weight == 1`` are shortest.
        """
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        wall, tgt = self.wall_ids, self.target_ids
        free = [[not (grid[r][c] & wall) for c in range(w)] for r in range(h)]
        dist = [[INF] * w for _ in range(h)]
        dq: deque[tuple[int, int]] = deque()
        for r in range(h):
            for c in range(w):
                if free[r][c] and (grid[r][c] & tgt):
                    dist[r][c] = 0
                    dq.append((r, c))
        while dq:
            r, c = dq.popleft()
            d = dist[r][c]
            # Predecessors of (r, c): the cell above reaches it by falling (0),
            # the two beside it by walking and the one below it by climbing (1).
            for ar, ac, cost in ((r - 1, c, 0), (r, c - 1, 1),
                                 (r, c + 1, 1), (r + 1, c, 1)):
                if (0 <= ar < h and 0 <= ac < w and free[ar][ac]
                        and dist[ar][ac] > d + cost):
                    dist[ar][ac] = d + cost
                    if cost:
                        dq.append((ar, ac))
                    else:
                        dq.appendleft((ar, ac))
        return dist

    def plan(self, eng, level: int | None = None):
        """Select (building on first use) this level's distance field, then defer
        to the memoized A*.

        The field depends only on the walls and the target, which are static per
        level, so any state of the level yields the same table and one build per
        level is enough."""
        field = self._fields.get(level)
        if field is None:
            field = self._fields[level] = self._distance_field(eng)
        self._dist = field
        return super().plan(eng, level)

    def heuristic(self, eng) -> int:
        dist, players = self._dist, self.player_ids
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & players:
                    return dist[r][c]
        # No player on the board: not reachable in this game (nothing consumes
        # the player), but a missing player must not read as "at the goal".
        return INF


class BackHomeSolver(PSAStarSolver):
    game_id = "puzzlescript_back_home"
    game_name = GAME_NAME
    expert_cls = BackHomeExpert

    #: Recorded against the game folder's own ``make_game`` -- the same adapter
    #: `game_envs` hands a live agent -- so the taped frames are the agent's.
    game_module_id = "ps:back_home"

    #: ~4x the worst level measured (level 8, 104k nodes); see "Solvable levels".
    node_cap = 400_000
    weight = 1

    #: Comfortably above the longest plan (level 10, 71 moves).
    max_steps = 200


if __name__ == "__main__":
    sys.exit(BackHomeSolver.main())
