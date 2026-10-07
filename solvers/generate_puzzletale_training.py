"""Generate Phase-1 training data for the PuzzleScript game ps:puzzletale
("PUZZLETALE" by Connorses -- an UNDERTALE puzzle-room pastiche).

The harness -- the engine-blackbox macro A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: its five rooms, the two
AIMED macros the shared push search does not have (grab a bridge seed, float
one out onto the water), the heuristic that guides them, and the notes on the
four hooks the shared class grew to say what this game does -- a conditional
walk map, and three facts the optimal-action labeller needed to stop being
either silent or wrong.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_puzzletale",
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
action (post rotation remap), i.e. the button an agent presses in the presented
view, so replaying the recorded actions reproduces the recorded frames exactly.
Every expert step also carries the full set of equally-optimal presses.

The game
--------
Five rooms, one win condition -- ``Some Player on Goal`` -- and three different
locks on the door.

**The spike gate (levels 0, 1, 2).** ``spike_floor`` is a strip of tiles laid
across the only corridor to the exit, and two late rules re-derive its state
from scratch every single turn::

    late [spike_floor spikes] -> [spike_floor]
    late [spike_floor] [target no crate] -> [spike_floor spikes] [target]

so the spikes are up whenever ANY target is bare and down the instant the last
one is covered. Spikes sit on the Player/Crate collision layer, so up means the
corridor is a wall. That makes these rooms a sokoban whose crates are the KEY
rather than the goal: cover every target, then walk out. Nothing about the win
condition says so, which is why the heuristic below has to combine two
estimates rather than pick one.

**The bridge seeds (level 3).** A river four cells wide, no crates, and a 2x2
patch of four seeds up a dead-end at the far corner -- so every seed is a round
trip. ACTION facing a seed patch picks one
up (one at a time -- the rule is guarded on ``no bridge_pickup``); ACTION facing
open water throws it in, and it then FLOATS in the direction you were facing
until it hits something::

    [bridge_float_r] -> [right bridge_float_r] again
    [> bridge_float|bridge_float_rest] -> [bridge_float_rest|bridge_float_rest]
    [> bridge_float|no water] -> [bridge_float_rest|  ]

and when four come to rest in a row they germinate into a bridge. All of that
resolves inside the single keypress that threw the seed, ``again`` loop and
all, so the interpreter hands back a settled board and the search never sees
the animation. Both ACTIONs read the player's FACING, which is the one thing in
this game that makes a bare walk step matter -- see "Optimal-action sets".

**The crate bridge (level 4).** ``late [crate water no bridge] -> [bridge2
water]``: shove a crate into the river and the crate BECOMES the plank. Two
crates, two water cells, and the second crate has to be pushed ACROSS the plank
the first one made.

Facing is drawn on the character, and it has to be: ``player_l`` and
``player_r`` are pixel-identical sprites, so the only thing separating "aimed
left" from "aimed right" on screen is the ``head_l`` / ``head_r`` face the game
paints in the cell ABOVE the player. ``--audit`` is the regression test for
that (and for crate-on-target, and for seed-on-water vs bridge).

The levels
----------
All five are 12x12 and all five are shipped, unedited::

    level  lock          plan  X  ties  search  proof   what it is
      0    spikes         17    0  24%    0.2s      2s  one crate, one target,
                                                        then the climb to the
                                                        door
      1    spikes         20    0  10%   50.4s    269s  two crates, two targets,
                                                        the river is scenery
      2    spikes         27    0   4%   38.0s     97s  four crates in a knot of
                                                        one-wide corridors
      3    seeds          53    8   8%   18.3s    177s  four grabs, four throws,
                                                        and the walk back each
                                                        time
      4    crate bridge   21    0   0%    0.5s      8s  push the plank, then
                                                        push the second crate
                                                        over it

``X`` counts the presses that are ACTION rather than a direction, ``ties`` the
share of steps with more than one optimal press, ``proof`` the exhaustive
primitive search ``--verify`` runs to certify the length. 138 moves over 5
levels, ~107s of one-time search, written to ``data/puzzletale_plans.json`` so
`parallelize_generator`'s shards do not each re-derive it. Delete the file to
re-derive.

Level 4 has no ties at all and that is the board, not a gap: every one of its 21
presses is either the only way round a wall or a push that has to happen in that
order, because the second crate can only cross the river on the plank the first
one became.

Expert solver
-------------
`PuzzletaleExpert` -- `PSSokobanExpert`'s walk-and-push macro A* over the real
interpreter, plus the three things this game needs that a sokoban does not:

  * **a conditional walk map.** Water blocks the player only while no bridge is
    lying on it, and the same cell is walkable the moment one is. Neither
    "water blocks" nor "water does not" is a fact about the board, so it cannot
    be said with ``blocker_names``; `_refine_free` (a new hook on the shared
    class) is where the search and the tie annotator both read it.
  * **two AIMED macros.** ``walk to a cell, aim at the thing beside it, press
    X`` -- one for the seed patches and one for the water. They are not pushes,
    and without them the macro search closes on level 3 with the river intact.
    `aim` is the whole subtlety: arriving at the aiming cell *from* the right
    direction does the aiming for free, so the macro is one press shorter when
    some shortest walk ends that way, and it costs a turn press (which the
    terrain cancels, leaving only the facing) when none does.
  * **a macro that simply walks onto the goal**, because that is the win and no
    push expresses it.

THE HEURISTIC combines three estimates and the composition is the interesting
part:

    max(walk distance to the goal, sokoban cost to cover every target)
        + 2 x (bridge seeds still to plant)

``max`` and not ``+`` for the first two: both are counted in presses and a push
IS a player move, so the two overlap and adding them would overestimate. The
seed term IS added, because it counts X presses and the other two count moves,
which are disjoint by construction. Each is admissible on its own:

  * the walk distance ignores everything but WALLS, which are the only things
    on these boards no rule can remove;
  * the cover cost is a minimum-cost perfect matching of targets to crates over
    `PSSokobanExpert`'s push-distance tables -- exact, not the inherited greedy
    match, because ``--verify`` PRUNES with this number and an overcount there
    would let the proof discard the state it is trying to rule out (see
    `PuzzletaleExpert._cover`). Note that this game's "dynamic" blockers are
    exactly right to build the tables against: spikes are up precisely while a
    target is bare, which is precisely when the estimate is consulted, and a
    crate can no more be pushed onto a spike than onto a wall;
  * ``4 - (seeds already resting or grown into bridge)`` is a floor on the
    plantings left, each of which costs a grab press and a throw press (less
    one if a seed is already in hand). It is charged only on a board that has a
    seed patch on it, so it is inert on the four rooms that have no seeds.

``weight = 1`` and every term above is admissible, so the plans are shortest --
and that is checked rather than argued: ``--verify`` re-derives each level's
optimal length by an exhaustive PRIMITIVE search over the interpreter, which
shares neither the macro vocabulary nor the region-key dedup the plans came
from. All five levels come back proved.

Optimal-action sets
-------------------
Inferred by `PSPushExpert.annotate_walks` -- pushes and the aimed X presses are
labelled with themselves alone, walk steps with every direction that keeps them
on a shortest route -- with the three hooks this game is the first to need:

  * ``trailing_walk_is_free``. Every plan here ENDS with a bare walk to the
    door -- 9 of level 0's 17 presses are its climb up the pyramid -- and the
    default labeller has no press after that run to read a destination off, so
    it labels the whole approach with the one route the expert happened to
    pick. Measured against the cell the plan ends on instead, 4 of those 9
    steps turn out to take either axis. That is the whole of what this flag is
    worth here (11 tie steps across the game with it, 7 without), and it is
    worth it: the closing walk is where a policy that has already solved the
    puzzle gets punished for a coin flip.
  * ``action_reads_facing``. It does not ADD tie sets, it REMOVES two that were
    wrong. The plan reaches each of level 3's aiming cells already facing the
    right way, so the last step of the walk before an X *is* the aim; without
    the flag, steps 24 and 37 are both labelled ``['up', 'left']`` -- and
    taking that ``up`` does not cost 29 and 16 presses like the ``left`` it is
    offered beside, it costs MORE than the whole remaining plan, because it
    arrives at the seed patch pointing somewhere else and the X then does
    nothing. ``--verify``'s tie check is what catches that; the flag is what
    stops it happening.
  * ``shadow_names``, and without it the other two would have had nothing to
    do: `annotate_walks` classifies a press by asking whether it changed
    anything but the player's own cell, and this game paints a FACE in the cell
    above the player and balances the carried seed on top of that. Both move on
    every press, so every walk step in the game read as a push and every single
    tie set came out empty -- an annotator that looked like it was working and
    was labelling 138 forced presses.

All three are opt-in on the shared class and inert for every generator that
does not set them (verified by re-deriving, from scratch, byte-identical plans
AND byte-identical optimal sets for 12 of ESCAPE!'s 13 levels, all 13 of Count
Mover's and all 5 of Color Combination's -- covering both the inferred and the
measured labeller).

What is NOT captured is cross-macro order ties -- which crate to shove first,
which of level 3's four identical seeds to fetch first -- which is
`annotate_walks`' standard, and deliberate, limitation in this family. No step
ever ships unlabelled (the always-emit-optimal-targets rule).

Augmentation
------------
PUZZLETALE's engine state after reset is identical for every seed (the levels
are fixed ASCII maps), so the only per-(seed, level) variable is the frame
rotation (rotation_k in {0,1,2,3}) with its matching directional action remap.
5 levels x 4 rotations = 20 presentations.

It is deliberately NOT in `PuzzleScriptAdapter._FLIP_GAMES`, and the reason is
the art rather than the mechanic. The mechanic would qualify on the ESCAPE!
argument -- gravity-free, the push stated with the relative ``>`` force, a win
condition that names no direction, screen-relative input -- but the character's
four facings are drawn as four hand-drawn faces, not as an orbit of the square's
symmetry group: ``head_l`` and ``head_r`` are exact mirrors of each other (fine),
while ``head_u`` and ``head_d`` are each horizontally asymmetric and are nothing
like each other's vertical mirror. A flip would therefore present a facing sprite
the game does not own, on the one cell an agent has to read to know where the
player is aimed. Rotation is safe because it permutes the four facings among
themselves: within one presentation the facing-to-art map stays a bijection, and
the presentation is fixed for a whole episode.

The expert plan is therefore seed-independent: solved once per level, cached,
and replayed per seed with that seed's remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_puzzletale_training.py --episodes 200 \
        --out data/training_multi_level/puzzletale

    python solvers/generate_puzzletale_training.py --plans    # level report
    python solvers/generate_puzzletale_training.py --verify   # replay + proof
    python solvers/generate_puzzletale_training.py --audit    # rendering audit
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (DEAD_COST, PSAStarSolver,   # noqa: E402
                                     PSSokobanExpert, _DELTA)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "PUZZLETALE"

#: Disk cache of each level's start plan and its optimal-action sets. The
#: searches are seed-independent, so without a file on disk every shard
#: `parallelize_generator` starts would re-derive all ~106s of them. Delete to
#: re-derive.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "puzzletale_plans.json"

#: Seeds a bridge takes. The rules spell it out four times over (the 7-, 6-, 5-
#: and 4-wide ``bridge_float_rest`` runs all germinate), and 4 is the shortest,
#: so four resting seeds in a row is the cheapest bridge the game will build.
_BRIDGE_SEEDS = 4


class PuzzletaleExpert(PSSokobanExpert):
    """Walk-and-push macro A* over the real interpreter, extended with the two
    AIMED X macros the bridge-seed room needs and a walk map that knows a
    bridged river is a floor.

    WHY `PSSokobanExpert` AND NOT `PSPushExpert`: the per-target PUSH-DISTANCE
    TABLES. Three of the five rooms are a sokoban in everything but the win
    condition -- the spikes are up exactly while a target is bare, so covering
    every target is not optional -- and on boards that are corridors and
    pockets the cost of a crate is its detour, not its separation, which is
    what the tables measure and a distance does not. The inherited ESTIMATE is
    not used: the win is ``Some Player on Goal``, so it hits zero with the
    level unsolved and the player still in the corridor, and its greedy
    target-to-crate matching is a search guide where `heuristic` needs a bound.
    See `_cover`. The two rooms with no targets at all (the seeds, the crate
    bridge) fall out of that half automatically.

    WHY THE TABLES ARE BUILT AGAINST THE *DYNAMIC* BLOCKERS. `PSSokobanExpert`
    builds them from ``blocker_ids``, which here includes spikes and seed
    patches -- things a board can lose. That is not a bug and not an
    approximation: spikes and patches share the crate's collision layer, so a
    crate can no more be pushed onto one than onto a wall, and the only state in
    which the table is ever read (some target bare) is exactly the state in
    which the spikes are up. The estimate is measuring the board it is standing
    on.

    THE DEADLOCK TEST comes with the tables for free (a cell no push can leave
    lands in no target's table and scores `DEAD_COST`), and it is load-bearing
    on level 2, whose four crates sit in a knot of one-wide corridors where
    nearly every wrong first push buries one permanently.
    """

    #: The whole action space. `PSPushExpert` drops ACTION because in a pure
    #: sokoban it does nothing; here it is half of level 3. The macro search
    #: branches on macros rather than on this list, but `record_level`'s
    #: epsilon detour and `PSExpert._astar` both read it, so it should name what
    #: the game actually has.
    directions = ["up", "down", "left", "right", "action"]

    pushable_names = ("crate",)
    #: Everything sharing the Player/Crate collision layer. Water is NOT here --
    #: it is conditional, and lives in `_refine_free`.
    blocker_names = ("wall", "spikes", "bridge_ground", "bell", "ring")
    target_names = ("target",)
    #: Both live in the cell ABOVE the player and are re-derived from it by a
    #: ``late`` rule every turn, so they move on every single press. Telling
    #: `annotate_walks` to ignore them is what lets it see a walk as a walk.
    shadow_names = ("head", "bridge_pickup")

    #: Keep each level's start plan (and its optimal sets) on disk.
    plan_cache_path = PLAN_CACHE

    #: Prove the plan shortest rather than returning the first win the macro
    #: search reaches: the macros here differ wildly in length (a 1-press X
    #: against an 11-press fetch-and-return), which is exactly the case the
    #: generate-time goal test gets wrong.
    exact_goal_test = True

    #: Label the order-free walk steps (`PSPushExpert.annotate_walks`).
    annotate = True
    #: ...measuring the plan's closing walk against the door it ends on, and
    #: treating the last step before an aimed X as the aim it is. See the
    #: module docstring's "Optimal-action sets".
    trailing_walk_is_free = True
    action_reads_facing = True

    def setup(self) -> None:
        super().setup()
        g = self.g

        def ids(name: str) -> set:
            return set(g.resolve_object_name(name))

        self.wall_ids = ids("wall")
        self.water_ids = ids("water")
        self.bridge_ids = ids("bridge")            # bridge1 (seeds) or bridge2 (crate)
        self.goal_ids = ids("goal")
        self.ground_ids = ids("bridge_ground")     # an unpicked seed
        self.spawn_ids = ids("bridge_spawn")       # marks a seed room
        self.pickup_ids = ids("bridge_pickup")     # the seed in hand
        self.rest_ids = ids("bridge_float_rest")   # a seed settled on the water
        self._face_of = {i: d for d in ("up", "down", "left", "right")
                         for i in g.resolve_object_name("player_" + d[0])}

        # Everything a rule here can move, create or destroy. Walls, targets,
        # water, spike_floor and the goal are static per level, and the head /
        # sound / wall-shading objects are pure functions of the rest, so this
        # is a canonical state within a level (`scope_by_level` is inherited).
        self.dyn_ids = (self.push_ids | self.player_ids | self.pickup_ids
                        | self.ground_ids | self.bridge_ids
                        | ids("bridge_float") | ids("bridge_anim")
                        | ids("spikes") | ids("bell") | ids("ring"))
        # `_region_key`'s half of that -- see there for why the carried seed
        # cannot stay in it.
        self._board_ids = self.dyn_ids - self.player_ids - self.pickup_ids
        self._static: dict = {}

    # -- the conditional walk map --------------------------------------------
    def _refine_free(self, eng, free) -> None:
        """Open water stops the player; bridged water does not.

        ``[> player|water no bridge] -> [player|water]`` is the rule, and both
        halves of it happen on these boards -- level 3 grows a bridge out of
        seeds, level 4 makes one out of a crate. The seeds RESTING on the water
        are not a bridge and do not open it."""
        water, bridge = self.water_ids, self.bridge_ids
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & water and not cell & bridge:
                    free[r][c] = False

    # -- state -----------------------------------------------------------------
    def _facing(self, eng) -> str | None:
        """Which way the player is aimed, or None if a rule has eaten it."""
        face_of = self._face_of
        for row in eng.grid:
            for cell in row:
                for o in cell:
                    got = face_of.get(o)
                    if got is not None:
                        return got
        return None

    def _carrying(self, eng) -> bool:
        pickup = self.pickup_ids
        return any(cell & pickup for row in eng.grid for cell in row)

    def _region_key(self, eng, region) -> tuple:
        """The board, whether the player's hands are full, and which region it
        is standing in.

        The carried seed cannot stay in ``board``: the interpreter parks
        `bridge_pickup` in the cell ABOVE the player, so leaving it there would
        re-attach the exact position the region canonicalisation just dropped
        and dedup nothing. It goes in as a bit instead.

        The player's FACING is dropped with its cell, which is the same
        approximation `PSPushExpert` documents and can mis-cost a state by one
        press -- a facing is always one turn away. Live here in a way it is not
        in a sokoban (level 3's X presses read it), so it is worth saying that
        it costs nothing on these boards rather than assuming: ``--verify``
        proves all five plans shortest by a search that does not merge
        anything."""
        board = frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & self._board_ids)
        )
        return (board, self._carrying(eng), region)

    # -- the macros a push search cannot express -------------------------------
    def extra_macros(self, eng, parent, walk_to, free) -> list[list[str]]:
        """``aim at a seed patch and press X``, ``aim at open water and press
        X``, and ``walk onto the door``.

        The first two are the whole of level 3 and the third is the whole win
        condition; the inherited push enumeration expresses none of them. Which
        of the two X macros is offered is decided by whether the player's hands
        are full, because the interpreter decides it the same way: the grab
        rules are guarded on ``no bridge_pickup`` and the throw rules on having
        one."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        facing = self._facing(eng)
        carrying = self._carrying(eng)

        depth: dict = {}

        def d_of(cell) -> int:
            got = depth.get(cell)
            if got is None:
                got = depth[cell] = len(walk_to(cell))
            return got

        def aim(stand, direction) -> list[str]:
            """The shortest press list leaving the player ON ``stand`` FACING
            ``direction`` -- what an aimed X needs.

            Walking IN that direction aims for free, so when some shortest
            route to ``stand`` arrives that way the macro is one press shorter
            than "walk there, then turn". When none does, the turn press is
            still only a turn: what the player is aiming at (a seed patch, open
            water) is by construction something they cannot walk into, so the
            interpreter cancels the move and keeps the facing."""
            dr, dc = _DELTA[direction]
            back = (stand[0] - dr, stand[1] - dc)
            if back in parent and d_of(back) == d_of(stand) - 1:
                return walk_to(back) + [direction]
            walk = walk_to(stand)
            if not walk and facing == direction:
                return []
            return walk + [direction]

        out: list[list[str]] = []
        for r in range(h):
            for c in range(w):
                cell = grid[r][c]
                if not cell:
                    continue
                aimable = (
                    (cell & self.ground_ids) if not carrying else
                    (cell & self.water_ids
                     and not cell & self.bridge_ids     # already a plank
                     and not cell & self.rest_ids))     # already holds a seed
                if aimable:
                    for direction, (dr, dc) in _DELTA.items():
                        stand = (r - dr, c - dc)
                        if stand in parent:
                            out.append(aim(stand, direction) + ["action"])
                if cell & self.goal_ids and (r, c) in parent:
                    walk = walk_to((r, c))
                    if walk:            # [] is the win the caller already has
                        out.append(walk)
        return out

    # -- heuristic --------------------------------------------------------------
    def _statics(self, eng) -> tuple:
        """``(wall, goals, h, w, seed_room)`` for this board, memoized on the
        wall map.

        Walls, the door and the seed patches' spawn markers are static per
        level -- no rule creates or destroys any of them -- but `heuristic`
        runs on every node and must not re-scan the grid for static facts."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        wall = tuple(tuple(bool(cell & self.wall_ids) for cell in row)
                     for row in grid)
        got = self._static.get(wall)
        if got is None:
            goals = frozenset((r, c)
                              for r, row in enumerate(grid)
                              for c, cell in enumerate(row)
                              if cell & self.goal_ids)
            seed_room = any(cell & self.spawn_ids
                            for row in grid for cell in row)
            got = (wall, goals, h, w, seed_room)
            self._static[wall] = got
        return got

    def _cover(self, crates, targets, player) -> int:
        """Pushes to get a crate onto every target, plus the walk to the first
        one -- an ADMISSIBLE lower bound, which is the whole reason it is here
        and not `PSSokobanExpert.heuristic`.

        The inherited estimate matches greedily (each target to its nearest
        unclaimed crate, in board order), which is a fine search guide and can
        overcount: two crates and two targets can be matched the expensive way
        round. That is harmless while the number is only steering a queue, and
        it is not harmless here, because ``--verify``'s optimality proof PRUNES
        with this number -- an overcount there would let the proof discard the
        very state it is trying to rule out. So the assignment is exact: a
        bitmask DP over the push-distance tables, which on a board with four
        crates is 64 operations.

        Both terms lower-bound disjoint presses. A push moves a crate one cell
        and costs one press, and the tables count pushes for one crate with the
        others ignored (the standard sokoban relaxation), so a minimum-cost
        perfect matching of targets to distinct crates can only undercount. The
        walk is the moves before the FIRST push: a bare target means some crate
        must move, so the player must first stand beside one, which is at least
        its manhattan distance less one, and none of those presses is a push.

        Every target -- not just the bare ones -- and every crate goes into the
        matching, because a crate can be pushed OFF a target on the way to
        covering another one; charging only the bare targets against only the
        loose crates would ignore the target that push uncovers. `DEAD_COST`
        when no assignment exists at all, which is the corner-deadlock test the
        tables give for free (a crate no push can ever move again appears in no
        table)."""
        if not targets:
            return 0
        rows = []
        for target in targets:
            steps = self._tables[target]
            row = [steps.get(crate) for crate in crates]
            if all(cost is None for cost in row):
                return DEAD_COST
            rows.append(row)
        size = 1 << len(crates)
        dp = [DEAD_COST] * size
        dp[0] = 0
        best = DEAD_COST
        for mask in range(size):
            if dp[mask] >= DEAD_COST:
                continue
            i = bin(mask).count("1")           # targets 0..i-1 are assigned
            if i == len(rows):
                best = min(best, dp[mask])
                continue
            for j, cost in enumerate(rows[i]):
                if cost is None or mask & (1 << j):
                    continue
                nxt = mask | (1 << j)
                if dp[mask] + cost < dp[nxt]:
                    dp[nxt] = dp[mask] + cost
        if best >= DEAD_COST:
            return DEAD_COST
        if best and crates:
            best += max(0, min(abs(player[0] - r) + abs(player[1] - c)
                               for r, c in crates) - 1)
        return best

    def heuristic(self, eng) -> int:
        wall, goals, h, w, seed_room = self._statics(eng)
        grid = eng.grid
        player = None
        planted = carrying = 0
        crates: list[tuple[int, int]] = []
        targets: list[tuple[int, int]] = []
        bare = False
        for r in range(h):
            row = grid[r]
            for c in range(w):
                cell = row[c]
                if not cell:
                    continue
                if cell & self.player_ids:
                    player = (r, c)
                if cell & self.rest_ids or cell & self.bridge_ids:
                    planted += 1
                if cell & self.pickup_ids:
                    carrying = 1
                crated = bool(cell & self.push_ids)
                if crated:
                    crates.append((r, c))
                if cell & self.target_ids:
                    targets.append((r, c))
                    bare |= not crated
        if player is None:                      # player consumed by a rule
            return DEAD_COST
        if player in goals:
            return 0

        # Moves: the player's own distance to the door over the only obstacle
        # no rule on these boards can remove. Plain BFS -- every edge costs one
        # press, so the priority queue a weighted search would need buys
        # nothing.
        walk = None
        dist = {player: 0}
        queue = deque([player])
        while queue and walk is None:
            cur = queue.popleft()
            nd = dist[cur] + 1
            for dr, dc in _DELTA.values():
                nxt = (cur[0] + dr, cur[1] + dc)
                if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                    continue
                if wall[nxt[0]][nxt[1]] or nxt in dist:
                    continue
                if nxt in goals:
                    walk = nd
                    break
                dist[nxt] = nd
                queue.append(nxt)
        if walk is None:                        # the door is walled off
            return DEAD_COST

        # Moves: the sokoban half. Zero once every target is covered, which is
        # also when the spikes come down, so the two estimates hand over to each
        # other exactly where the puzzle does.
        cover = self._cover(crates, targets, player) if bare else 0

        # X presses: one grab and one throw per seed still to plant. Disjoint
        # from both move counts above, hence added rather than maxed.
        seeds = 0
        if seed_room:
            short = max(0, _BRIDGE_SEEDS - planted)
            seeds = max(0, 2 * short - carrying)

        return max(walk, cover) + seeds


class PuzzletaleSolver(PSAStarSolver):
    game_id = "puzzlescript_puzzletale"
    game_name = GAME_NAME
    expert_cls = PuzzletaleExpert

    #: Record against the GAME FOLDER's adapter -- the object `game_envs` hands
    #: a live agent. `games/ps:puzzletale/ps:puzzletale.py` is a plain
    #: passthrough today, so this is currently the same thing a bare
    #: `PuzzleScriptAdapter(GAME_NAME)` would build; it is named anyway so that
    #: a step cap or sprite patch added there later cannot silently make this
    #: generator tape a game nobody plays (the trap ps:count_mover hit).
    game_module_id = "ps:puzzletale"

    #: Unweighted: the heuristic is admissible and the whole search is ~107s
    #: once, cached to disk, so there is nothing to buy by trading optimality
    #: away.
    weight = 1
    #: Runaway guard, not a tuning dial: the slowest level here (1) finds and
    #: PROVES its plan in 50s, so nothing comes near this.
    node_cap = 2_000_000
    #: The longest plan is 53 presses; the rest is room for the RESET
    #: exploration prefix (<= 15) and a re-plan after it. Stays under the
    #: adapter's own 200-step per-level budget, which would otherwise flip a
    #: level to GAME_OVER mid-plan.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup
        rather than as a mysteriously slow first seed, and it fills the disk
        cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _fresh():
    """A solver, its adapter and an expert, sharing one plan cache."""
    solver = PuzzletaleSolver()
    game = solver.make_game(0)
    expert = PuzzletaleExpert(game, node_cap=PuzzletaleSolver.node_cap,
                              weight=PuzzletaleSolver.weight)
    return solver, game, expert


def _report() -> int:
    """Per-level board size, plan length, X-press count and tie coverage."""
    import time

    _solver, game, expert = _fresh()
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        if found is None:
            print(f"level {level}: NO PLAN")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        acts = sum(1 for d in found if d == "action")
        total += len(found)
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level}: {eng.height:2d}x{eng.width:2d}, "
              f"{len(found):3d} moves ({acts} X) "
              f"(budget {game._max_steps}, {room}), "
              f"{ties:3d} steps with a tie set "
              f"({ties / max(1, len(found)):3.0%}), {dt:6.1f}s")
    print(f"total {total} moves over {game.n_levels} levels")
    return 0


#: Macro budget for one `_shortest_within` call. A runaway guard: the whole
#: --verify pass over all five levels closes in well under this many.
_PROOF_NODE_CAP = 3_000_000


def _verify() -> int:
    """Three checks, in rising order of what they cost and what they prove.

    1. **Every plan wins.** Replayed press by press from a fresh ``set_level``
       through the adapter's own action path -- i.e. through the rotation remap
       an agent's presses go through -- and required to reach `GameState.WIN`
       on exactly its last press and not before.
    2. **The plan is shortest**, by `_shortest_within` -- an exhaustive
       PRIMITIVE search over the interpreter, which owes nothing to the macro
       search's region-key dedup or to its macro vocabulary. Those are the two
       approximations this file's plans would otherwise rest on, so this is the
       check that can catch either of them being wrong.
    3. **Every optimal SET is real.** For each labelled alternative the expert
       did not itself take, take it instead and require `_shortest_within` to
       still finish in exactly the moves that were left. `annotate_walks`
       INFERS the ties from a walk BFS rather than measuring them, so without
       this an inference bug ships as a confidently mislabelled press.

    Both search checks still trust that `PuzzletaleExpert.heuristic` is
    admissible (they prune with it) -- that is argued term by term in the
    module docstring, and it is the one thing here no run can prove.
    """
    import time

    from arcengine import ActionInput, GameState

    from solvers.common.ps_astar import screen_action, snapshot, restore

    _solver, game, expert = _fresh()
    bad = 0

    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(game._engine, level)
        if plan is None:
            print(f"level {level}: NO PLAN")
            bad += 1
            continue
        sets = getattr(plan, "optsets", None)

        # 1. replay through the adapter, exactly as a recorded episode does
        game.set_level(level)
        rot = (game._rotation_k, game._hflip, game._vflip)
        early = None
        for i, direction in enumerate(plan):
            game.perform_action(ActionInput(id=screen_action(direction, *rot)))
            if game._state == GameState.WIN and early is None and i + 1 < len(plan):
                early = i
        won = game._state == GameState.WIN
        if not won or early is not None:
            print(f"level {level}: REPLAY FAILED "
                  f"(win={won}, early win at step {early})")
            bad += 1
            continue

        # 2. nothing shorter exists
        game.set_level(level)
        eng = game._engine
        t0 = time.time()
        shortest, nodes = _shortest_within(expert, eng, len(plan))
        dt = time.time() - t0
        if shortest == len(plan):
            proof = f"proved shortest ({nodes} nodes, {dt:.0f}s)"
        else:
            proof = (f"NOT SHORTEST: {shortest} presses exist"
                     if shortest is not None else
                     f"UNPROVED (node budget hit after {nodes})")
            bad += 1

        # 3. every labelled alternative is genuinely as short
        game.set_level(level)
        checked = wrong = 0
        for i, direction in enumerate(plan):
            left = len(plan) - i           # presses remaining, this one included
            for alt in (sets[i] if sets else [direction]):
                if alt == direction:
                    continue
                here = snapshot(eng)
                eng.step(alt)
                rest = 0 if eng.check_win() else _shortest_within(
                    expert, eng, left - 1)[0]
                restore(eng, here)
                checked += 1
                if rest is None or 1 + rest != left:
                    wrong += 1
                    print(f"level {level} step {i}: '{alt}' is labelled optimal "
                          f"but costs {'more than ' + str(left) if rest is None else 1 + rest}"
                          f", not {left}")
            eng.step(direction)
        bad += wrong

        print(f"level {level}: WIN in {len(plan)} presses, {proof}, "
              f"{checked - wrong}/{checked} tie-set alternatives confirmed")

    print("verify clean" if not bad else f"VERIFY FAILED: {bad} problems")
    return 0 if not bad else 1


def _shortest_within(expert, eng, limit: int) -> tuple[int | None, int]:
    """``(shortest winning press count from the engine's current grid, nodes
    expanded)``, or ``(None, nodes)`` when no win exists within ``limit``
    presses (or the node budget ran out -- the two are distinguished by whether
    ``nodes`` reached `_PROOF_NODE_CAP`). Leaves the grid unchanged.

    Exhaustive breadth-first over the PRIMITIVE presses, deduped on
    `PSPushExpert._key` -- the full dynamic object set, which is an exact
    canonical engine state, not the region canonicalisation the macro search
    uses. That independence is the whole point: this is the only thing in this
    file that can contradict the macro search.

    What makes it affordable is the one thing it does borrow: a successor whose
    ``g + heuristic`` already exceeds ``limit`` is dropped. The heuristic is
    admissible (see the module docstring), so that prune can never hide a
    solution, and it is the difference between a proof and a hang -- unpruned,
    the same search does not finish level 0's 17 presses in two minutes; pruned,
    the deepest board (level 3, 53 presses) closes in 35694 states.
    """
    from solvers.common.ps_astar import snapshot, restore

    start = snapshot(eng)
    if eng.check_win():
        return 0, 0
    frontier = [start]
    best_g = {expert._key(eng): 0}
    nodes = 0
    found = None
    for g in range(1, limit + 1):
        nxt = []
        for snap in frontier:
            for direction in expert.directions:
                restore(eng, snap)
                eng.step(direction)
                nodes += 1
                if eng.check_win():
                    found = g
                    break
                if g + expert.heuristic(eng) > limit:
                    continue
                key = expert._key(eng)
                if best_g.get(key, 1 << 30) <= g:
                    continue
                best_g[key] = g
                nxt.append(snapshot(eng))
            if found is not None or nodes >= _PROOF_NODE_CAP:
                break
        if found is not None or nodes >= _PROOF_NODE_CAP or not nxt:
            break
        frontier = nxt
    restore(eng, start)
    return found, nodes


def _audit() -> int:
    """Assert every cell composition this game can show renders distinctly.

    Compared as WHOLE FRAMES of a board that differs in exactly one cell, not
    as a cropped cell: these boards are 12x12, so the renderer lays them out at
    5px per cell and then NEAREST-upscales 60px to 64px, and a fixed
    ``cell * r`` crop is off by the letterbox and by the uneven upscale.

    ONE collapse is expected, and it is asserted to be the only one:
    ``background1`` and ``background2`` are two different purples that both
    resolve to ARC palette 15. They are the same thing mechanically -- plain
    floor -- and no rule distinguishes them, so this costs nothing.

    The FACING is audited as an eight-way check rather than a one-cell one, and
    that is the part worth having. ``player_l`` and ``player_r`` are
    pixel-identical in the game file, so a single-cell audit would report them
    as a collapse and be measuring the wrong thing: what carries the facing is
    the FACE the game paints in the cell above the player, and what an agent
    reads is the pair. So the four facings go in as (head, player) pairs -- and
    again with the seed the player is carrying stacked on that same head cell,
    because a carried seed that hid the face would make level 3's X unaimable
    exactly when it is being aimed. All eight must come out distinct.
    """
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, _expert = _fresh()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    cells = {
        "floor1": ("background1",),
        "floor2": ("background2",),
        "wall_a": ("background1", "wall_a"),
        "wall_b": ("background1", "wall_b"),
        "target": ("background2", "target"),
        "crate": ("background2", "crate"),
        "crate_on_target": ("background2", "target", "crate"),
        "spike_floor": ("background2", "spike_floor"),
        "spiked": ("background2", "spike_floor", "spikes"),
        "water": ("background1", "water"),
        "goal": ("background1", "goal"),
        "seed_patch": ("background1", "bridge_ground"),
        "seed_resting": ("background1", "water", "bridge_float_rest"),
        "bridge_grown": ("background1", "water", "bridge1"),
        "bridge_crate": ("background1", "water", "bridge2"),
        "player": ("background2", "player_d"),
        "player_on_target": ("background2", "target", "player_d"),
        "player_on_spikefloor": ("background2", "spike_floor", "player_d"),
    }
    #: (cell above, player cell) -- the pair that carries the facing, empty
    #: handed and holding a seed.
    facings = {
        f"facing_{d}{tag}": (("background2", "head_" + d[0]) + held,
                             ("background2", "player_" + d[0]))
        for d in ("up", "down", "left", "right")
        for tag, held in (("", ()), ("_holding", ("bridge_pickup",)))
    }

    h = w = 12

    def frame(patch: dict) -> np.ndarray:
        """The whole board rendered as plain floor, with ``patch`` painted in."""
        eng.height, eng.width = h, w
        eng.grid = [[{idx["background1"]} for _ in range(w)] for _ in range(h)]
        for (r, c), objs in patch.items():
            eng.grid[r][c] = {idx[o] for o in objs}
        eng._position_index_dirty = True
        return np.asarray(_render_frame(eng, g))

    shots = {name: frame({(6, 6): objs}) for name, objs in cells.items()}
    shots.update({name: frame({(5, 6): up, (6, 6): here})
                  for name, (up, here) in facings.items()})

    expected = {("floor1", "floor2")}
    clashes = {(a, b) for a, b in itertools.combinations(sorted(shots), 2)
               if np.array_equal(shots[a], shots[b])}
    unexpected = clashes - expected
    missing = expected - clashes
    for a, b in sorted(unexpected):
        print(f"  INDISTINGUISHABLE: {a} / {b}")
    for a, b in sorted(missing):
        print(f"  expected-collapse {a} / {b} no longer collapses "
              f"(harmless, but the audit's note is now stale)")
    print(f"{len(shots)} compositions, {len(clashes)} collapses "
          f"({len(expected)} expected)")
    print("audit clean" if not unexpected
          else f"AUDIT FAILED: {len(unexpected)} indistinguishable pairs")
    return 0 if not unexpected else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(PuzzletaleSolver.main())
