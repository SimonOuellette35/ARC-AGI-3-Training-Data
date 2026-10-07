"""Generate Phase-1 training data for the PuzzleScript game ps:atlas_shrank_3
(James Noeckel's "Atlas Shrank").

The whole harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: its name, its pruned action
set, its goal heuristic, its skipped levels and its search budget.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_atlas_shrank",
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
A side-on gravity platformer. Atlas walks LEFT/RIGHT along the floor of a walled
cavern and must reach the Exit (`all Exit on Player`). There is no jump: the
whole movement vocabulary is

  * **walk** one cell horizontally,
  * **auto step-up** -- walking into a ONE-cell-high ledge climbs it (the
    `H_step` helper is spawned above the player, carried along by the move, and
    the player is teleported into it when the horizontal move itself is blocked),
  * **fall** -- gravity is an `again` loop, so leaving a ledge drops the player
    all the way to the floor in the SAME turn, however far that is.

ACTION (X / ACTION5) picks up and puts down Crates. Facing a crate, ACTION lifts
it onto the player's head (`Heldcrate`, which rides along and is itself steppable);
ACTION again drops it one column ahead, where gravity takes it. Crates are also
pushed by walking into them at floor level. Since the player can only ever climb
ONE cell, crates are the game's ladder: every level that needs more than a
one-cell rise is really "carry a crate to the right column and drop it".

Switches (`Switch`) and Doors (`Door`) are the second mechanic. A Door is Solid
and blocks the way; a Switch opens every door in the level exactly while it
carries `Weight` (the player OR a crate) -- `late [Switch no Weight] [...] ->
[... Door]` re-closes them the moment the weight leaves. So a switch guarding a
door the player must walk through has to be pinned down by a *crate*: the player
cannot both stand on the switch and be at the door.

Pruned action set
-----------------
`AtlasShrankExpert.directions` is only ``left / right / action``. UP and DOWN are
provably dead keys, and dropping them cuts the branching factor from 5 to 3:

  * **UP** is cancelled outright by the rule ``[ up Player ] -> cancel``, which is
    how the game forbids jumping. The whole turn is undone; the grid never moves.
  * **DOWN** is subsumed by gravity. `Down [ Massive ] -> [ down Massive ] again`
    and the player's own gravity rule run to fixpoint at the end of every turn, so
    the player is always already resting on something and a DOWN press has nothing
    left to do.

Verified empirically as well as by reading the rules: over thousands of reachable
states across all 12 levels (random walks including ACTION presses and held
crates), neither key ever changed the grid. Pruning them is a search speed-up
only -- it cannot make a state unsolvable, it can only leave a level unsolved,
which `skip_levels` / `discover_solvable` handle as data that is never emitted.

The heuristic (why Manhattan is not good enough)
------------------------------------------------
`AtlasShrankExpert` precomputes, ONCE per level, a *fall-is-free* distance field
from the Exit over the level's non-wall cells (a 0-1 BFS): moving horizontally or
climbing costs 1, dropping one row costs 0, because a fall of any height is a
single turn. `heuristic` is then a table lookup at the player's cell.

Plain Manhattan distance is badly misleading in a cavern: it walks straight
through walls, so on the maze-shaped levels A* spends its budget in pockets that
are geometrically near the exit and topologically far from it, and it *over*
estimates every drop (a 6-row fall is 1 move, not 6). The field respects the walls
and prices the falls, which is what makes the solved levels land inside the node
budget at all. Doors are treated as PASSABLE by the field, and crates are ignored
entirely, so it stays an optimistic lower bound: the plans it returns at
``weight == 1`` are shortest.

Weight escalation
-----------------
That optimism is also the heuristic's ceiling. Wherever the answer is "first walk
AWAY from the exit, fetch a crate, and drop it under the ledge", the field is not
merely loose, it points the wrong way, and plain A* has to widen a frontier of
crate configurations to overcome it. Levels 0-3 and 5 never hit that (17-3140
nodes each, solved optimally); level 4 does, and does not finish an admissible
search in half an hour.

So `AtlasShrankExpert.escalation` runs the CHEAP, optimal search first under a
small node cap and only falls through to a greedy weighted one for the levels that
need it. Level 4 then lands 7.6k nodes into the second pass. This keeps every
level that can be solved optimally optimal -- a single global ``weight`` of 5
would have quietly lengthened the other five levels' plans to buy one -- at the
cost of the small cap being spent first on the levels that fall through (a bounded
~2 minutes each).

Solvable levels
---------------
Levels 0-5 (6 of the 12), measured at 430s of one-time search and 131MB peak:

    level 0   6 moves      17 nodes   optimal
    level 1  21 moves    2633 nodes   optimal
    level 2  33 moves    1445 nodes   optimal
    level 3  40 moves    3140 nodes   optimal
    level 4 118 moves   17647 nodes   weight 5 (10k of those are the w=1 pass)
    level 5  33 moves    1271 nodes   optimal

What bounds the set is search budget, not mechanics -- the interpreter reproduces
every level faithfully; see `AtlasShrankSolver.skip_levels` for what was measured
on the other six and why they are out of reach. Because each level is an
independent trajectory in the training schema, an episode holding only the solved
levels is valid, and ``level_id`` preserves the true (seed, level) provenance.
Clear ``skip_levels`` (with a bigger cap / weight in ``escalation``) to
force-attempt them.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps); only the presentation augmentations vary per (seed, level): the frame
rotation (rotation_k in {0,1,2,3}) and the surface recolor (see
`puzzlescript_adapter._recolor_surfaces`). The expert plan is therefore
seed-independent: solved once per level, cached, and replayed per seed with that
seed's rotation-remapped screen actions and recolored frames.

Note that a rotated Atlas Shrank is a genuinely different-looking game -- gravity
points sideways or up on screen -- which is exactly the point of the augmentation:
the agent has to read "which way is down" off the frame rather than assume it.

Usage (run from the repo root):
    python solvers/generate_atlas_shrank_training.py --episodes 200 \
        --out data/training_multi_level/puzzlescript_atlas_shrank
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSExpert  # noqa: E402

GAME_NAME = "Atlas_Shrank"

#: "unreachable" in the distance field. Kept well below ``1 << 30`` so that
#: ``weight * INF`` stays a small int inside the A* priority queue.
INF = 1 << 20


class AtlasShrankExpert(PSExpert):
    """A* over the real engine, guided by a fall-is-free distance-to-exit field.

    See the module docstring for why ``directions`` drops UP/DOWN and why the
    field beats Manhattan. ``scope_by_level`` is required because `_key` is
    narrowed to the DYNAMIC objects only -- correct within a level, but not across
    levels, whose walls and exits differ.
    """

    #: UP is `cancel`ed by a rule and DOWN is subsumed by gravity; both are
    #: no-ops in every reachable state, so the search never branches on them.
    directions = ["left", "right", "action"]

    scope_by_level = True

    #: ``(weight, node_cap)`` searches to try in order, first success wins. See
    #: "Weight escalation" in the module docstring: the admissible pass is tried
    #: first so a level that CAN be solved optimally is, and only a level that
    #: exhausts its cap pays for the greedy one. The first cap is ~3x the most
    #: any optimally-solved level has needed (level 3, 3140 nodes), so it is
    #: headroom rather than a tuned fit; a level that overruns it degrades to a
    #: slightly longer plan, never to a failure.
    escalation: tuple[tuple[int, int], ...] = ((1, 10_000), (5, 400_000))

    def setup(self) -> None:
        g = self.g
        self.player_ids = set(g.resolve_object_name("player"))
        self.exit_ids = set(g.resolve_object_name("exit"))
        self.wall_ids = set(g.resolve_object_name("wall"))
        # Everything that is NOT scenery. Derived by SUBTRACTION rather than by
        # listing the movers, so the bookkeeping helpers (H_step / H_grav /
        # H_pickup / H_drop) and every player facing / crate / door variant are
        # covered without having to keep a list in sync with the rules. Walls
        # (post-`Wallify`), the Exit and the Switches are the only static
        # objects; Doors open and close, so they stay in the key.
        static = {self.bg_id} | self.wall_ids | self.exit_ids | set(
            g.resolve_object_name("switch"))
        self.dyn_ids = set(g.obj_name_to_idx.values()) - static
        #: level -> distance field, built on first use.
        self._fields: dict[int | None, list[list[int]]] = {}
        self._dist: list[list[int]] | None = None

    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    # -- the fall-is-free distance field ------------------------------------
    def _distance_field(self, eng) -> list[list[int]]:
        """0-1 BFS from the Exit over non-wall cells, in units of TURNS.

        Edge costs are those of the forward move ``a -> b`` and the search runs
        on the REVERSED graph, so ``dist[a]`` is the cost of getting from ``a`` to
        the exit. There are exactly three kinds of edge, one per thing a turn can
        do to the player:

          * ``a`` directly above ``b`` -> cost **0**. Gravity is an `again` loop,
            so a drop of any height resolves inside the turn that started it; a
            column of free cells is free to descend.
          * ``a`` beside ``b`` -> cost **1** (a walk).
          * ``a`` one row BELOW and one column across from ``b`` -> cost **1**.
            This is the auto step-up, and it is the edge that makes the field a
            true lower bound: a climb gains a row AND a column in ONE turn, so
            pricing it as a walk plus a separate vertical move (2) would
            OVER-estimate every staircase and cost the search its optimality
            guarantee. The diagonal does not check that the intermediate cell is
            the ledge being climbed, which -- like ignoring crates and treating
            Doors as passable -- only ever makes the estimate smaller.

        So no single turn can cost less than this field says, and the plans
        returned at ``weight == 1`` are genuinely shortest.
        """
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        wall, exi = self.wall_ids, self.exit_ids
        free = [[not (grid[r][c] & wall) for c in range(w)] for r in range(h)]
        dist = [[INF] * w for _ in range(h)]
        dq: deque[tuple[int, int]] = deque()
        for r in range(h):
            for c in range(w):
                if free[r][c] and (grid[r][c] & exi):
                    dist[r][c] = 0
                    dq.append((r, c))
        while dq:
            r, c = dq.popleft()
            d = dist[r][c]
            # Predecessors of (r, c): the cell above reaches it by falling (0);
            # the sides by walking, and the two cells below-and-across by
            # climbing, for 1 each. (r+1, c) is not a predecessor in its own
            # right -- there is no straight-up move, only the diagonal climb.
            for ar, ac, cost in ((r - 1, c, 0), (r, c - 1, 1), (r, c + 1, 1),
                                 (r + 1, c - 1, 1), (r + 1, c + 1, 1)):
                if (0 <= ar < h and 0 <= ac < w and free[ar][ac]
                        and dist[ar][ac] > d + cost):
                    dist[ar][ac] = d + cost
                    if cost:
                        dq.append((ar, ac))
                    else:
                        dq.appendleft((ar, ac))
        return dist

    def plan(self, eng, level: int | None = None):
        """Select (building on first use) this level's distance field, then run
        the `escalation` searches until one returns a plan.

        The field depends only on the walls and the exit, which are static per
        level, so any state of the level yields the same table and one build per
        level is enough.

        The parent memoizes by state key -- including a failure -- so a ``None``
        from one budget is evicted before the next is tried; otherwise the cheap
        pass would veto the expensive one for the rest of the run. The final
        ``None`` (every budget exhausted) IS cached, which is what stops
        `discover_solvable` re-searching a hopeless level once per seed."""
        field = self._fields.get(level)
        if field is None:
            field = self._fields[level] = self._distance_field(eng)
        self._dist = field

        key = (level if self.scope_by_level else None, self._key(eng))
        if key in self.cache:
            return self.cache[key]
        for self.weight, self.node_cap in self.escalation:
            sol = super().plan(eng, level)
            if sol is not None:
                return sol
            del self.cache[key]
        self.cache[key] = None
        return None

    def heuristic(self, eng) -> int:
        dist, players = self._dist, self.player_ids
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & players:
                    return dist[r][c]
        # No player on the board: not reachable in this game (nothing consumes
        # the player), but a missing player must not read as "at the goal".
        return INF


class AtlasShrankSolver(PSAStarSolver):
    game_id = "puzzlescript_atlas_shrank"
    game_name = GAME_NAME
    expert_cls = AtlasShrankExpert

    #: Recorded against the game folder's own ``make_game`` -- the same adapter
    #: `game_envs` hands a live agent -- so the taped frames are the agent's.
    game_module_id = "ps:atlas_shrank_3"

    #: Levels 6-11, none of which the primitive-action A* wins. Skipped up front
    #: so discovery does not burn the whole escalation on them at every startup.
    #:
    #: 6, 7 and 8 were each given a 28-minute search at weights 3, 5 and 8 and
    #: found nothing. They are the switch-and-door levels: a door only stays open
    #: while a Switch is weighted, and the player cannot be both the weight and
    #: the one walking through, so each is gated on a multi-crate haul the
    #: distance field gives no credit for until it is finished.
    #:
    #: 9, 10 and 11 are not close, and the reason is mechanical rather than a
    #: timeout: the interpreter step is ~99.6% of node cost and scales with board
    #: area, so at 54, 72 and 36 cells wide (against the 18-wide levels that do
    #: solve) they run at a few tens of nodes/second while needing solutions
    #: hundreds of moves long through many crate placements. They are several
    #: `flickscreen 18x11` screens across; winning them needs a macro planner over
    #: "carry crate C to column X", not a bigger budget for this one.
    skip_levels = frozenset({6, 7, 8, 9, 10, 11})

    #: Per-search budgets live in `AtlasShrankExpert.escalation`; these two are
    #: the values it starts from and are overwritten on the first `plan`.
    node_cap = 10_000
    weight = 1

    #: Comfortably above the longest plan (level 4, 114 moves).
    max_steps = 400


if __name__ == "__main__":
    sys.exit(AtlasShrankSolver.main())
