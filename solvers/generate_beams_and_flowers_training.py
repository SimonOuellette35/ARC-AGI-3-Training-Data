"""Generate Phase-1 training data for the PuzzleScript game ps:beams_and_flowers
("Beams and Flowers" by BoredMatt).

The harness -- the engine-blackbox search plumbing, the rotation contract, the
trajectory recorder and the BaseSolver CLI -- lives in `solvers/common/ps_astar.py`,
shared with the other search-the-interpreter ps: generators. This file is the
game-specific part: the macro model of its movement, the goal heuristic, and the
level bookkeeping.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_beams_and_flowers",
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
action (post rotation remap), i.e. the button an agent presses in the augmented
view, so replaying the recorded actions reproduces the recorded frames exactly.

The game
--------
A robot walks a room to pass through every green CHECKPOINT and to see every
purple FLOWER destroyed, then the level is won. Two things make it more than a
maze:

  * **The robot is a tank.** Pressing a direction it is not already facing only
    TURNS it -- the move costs a second press. So the cost metric is over
    ``(cell, facing)``, not over cells, and "walk to X" is a shortest path in
    that doubled graph.
  * **Lasers.** Every laser paints a ray each turn; the ray runs until it hits a
    mirror, a wall, a white block or a golden block, and reflects off the mirror
    diagonals. Standing in a ray kills the robot (the level restarts); a ray
    crossing a FLOWER destroys it, which is the only way to clear one -- walking
    into a flower kills you. A ray crossing a CHECKPOINT destroys it, which is
    an instant loss, so the beams have to be aimed *around* the checkpoints.

Everything else is the vocabulary the levels combine:

  * white / ice / golden blocks and all three mirror families are pushed one cell
    by walking into them (ice blocks also melt under a ray);
  * pressing ACTION reaches down the row/column the robot faces to the first
    obstacle and, if that is a RED mirror, rotates it anti-clockwise; if it is a
    GOLDEN block or a BLUE mirror, shoves it one cell away. This is the remote
    control that lets you re-aim a beam from outside its path;
  * rusted floor collapses into a hole the moment anything steps *off* it, so a
    rusted tile is a one-way door.

Expert solver
-------------
`BeamsExpert`: A* (with a breadth-first beam as the fallback for the levels A*
cannot reach) over the REAL interpreter, whose successors are MACROS rather than
single key presses:

  * *walk onto a checkpoint* -- collect it;
  * *walk to a push cell and push* -- move a block or a mirror one cell;
  * *walk into line with a red/blue mirror or golden block, face it, press
    ACTION* -- rotate or shove it from range.

WHY MACROS. Between two events that change anything, every press is just the
robot repositioning itself, and with the turn cost a five-cell walk is up to ten
presses. A primitive search therefore spends its entire budget re-deriving walks:
a 60-press solution is a depth-60 search over branching 5 where fewer than ten of
those presses did anything. Branching on macros makes the depth the number of
EVENTS and finds each walk with one Dijkstra. Costs stay in primitive presses (the
unit the agent pays) and a plan is emitted as a flat list of directions, so the
recorder is unchanged.

WHY THE WALK MODEL IS SOUND. A walk changes nothing a walk depends on: the robot
is not a laser obstacle, so the ray map is frozen for the whole walk; collapsing a
rusted tile leaves a hole, and holes do not stop beams either. So "avoid the cells
that currently hold a ray" is an exact safety test for a walk, not an
approximation -- and every macro is replayed through the interpreter anyway, with
a restart (i.e. a death, or a checkpoint shot) rejected as a dead branch. It is
also what lets `_apply_macro` apply the walking presses to the grid directly
instead of ticking the interpreter for each, which is worth ~15x; that fast path
was differentially fuzzed against the interpreter over 2400 random press
sequences on all forty levels with zero disagreements, as was the filter that
drops shoves the rules would cancel (2335 shoves, zero disagreements).

The walk Dijkstra charges a small extra for stepping onto rusted floor, so among
equal-length routes the expert takes the one that does not burn a one-way tile.
That is only a tie-break, and it is not the same thing as reasoning about the
tiles: the search gets the route right on the two boards built around the
collapsing floor (10 and 21) because the macro sequence is short enough to search,
not because the walk between macros is chosen with the later macros in mind.

For the ACTION macros the search keeps the ``action_stands`` cheapest firing
positions per (target, direction) rather than only the closest one, because the
closest one is frequently the one the newly-aimed beam sweeps through: "step back
one tile and *then* turn the mirror" is a move the game asks for often, and it is
invisible to a search that only ever fires from point blank.

Levels
------
Forty levels; the expert wins 27 of them, and the other thirteen are skipped
rather than emitted as broken data (see `PSAStarSolver`). What it does and does
not get:

  * every level up to 23 except 11 and 19, plus 26, 28, 30, 31 and -- the
    surprise -- 35 ("Not afraid", sixty static mirrors and two lasers, which
    looks like the hardest board in the game and turns out to be a 38-press walk
    once you can see which beams are already lethal);
  * levels 21 ("The old church") and 10 ("Make your way"), the two boards built
    entirely around the collapsing floor, both of which the rusted-tile penalty
    in the walk Dijkstra gets right;
  * NOT levels 11, 19, 24, 25, 27, 29, 32, 33, 34, 36, 37, 38, 39 -- see
    `skip_levels`.

Eleven of the thirteen failures are flower levels, and they fail for one reason:
the search has no idea what progress toward burning a flower looks like. Turning
a red mirror teleports a beam somewhere else entirely, so the useful sequences
are long runs of macros that each look worthless on their own, and neither the
A* frontier nor the beam's ranking has anything to prefer them by. (A heuristic
term for beam-to-flower distance was tried on exactly these thirteen and rescued
none of them; see `_heuristic`.) The other two, 24 and 25, are simply the widest
boards in the game with the most checkpoints.

Augmentation
------------
Beams and Flowers takes the plain rotation augmentation: the frame is rotated by
``rotation_k in {0,1,2,3}`` per (seed, level) and directional input is
forward-remapped by the adapter, so the expert -- which plans in ENGINE space --
emits the SCREEN press (`ps_astar.screen_action`, from `record_level`). It is in
neither `PuzzleScriptAdapter._RECOLOR_GAMES` nor `_FLIP_GAMES`.

A recolor would be wrong: the mechanic is read off colour -- green checkpoint,
purple flower, and red vs blue vs yellow mirror is precisely the difference
between a mirror ACTION turns, one it shoves, and one it ignores -- and the
recolor surfaces flatten a sprite to one colour. A FLIP, on the other hand,
would be sound and is simply not enabled: the game is gravity-free with
screen-relative input, and reflection is a symmetry of the beams, with the
sprites to match (PlayerRight/PlayerLeft and RayNE/RayNW are drawn as mirror
images of each other, so a flipped NE mirror both looks like an NW mirror and
bounces like one). Adding the game to `_FLIP_GAMES` is where to go if these seeds
ever want more presentation variety than one of four rotations.

The engine state after reset is seed-independent (levels are fixed ASCII maps),
so plans are solved once, cached, and replayed per seed at that seed's rotation.
Per-seed variety within a level is therefore the rotation plus the exploration
prefix's flail, not a different route.

Usage (run from the repo root):
    python solvers/generate_beams_and_flowers_training.py --episodes 200 \
        --out data/training_multi_level/beams_and_flowers
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from solvers.common.ps_astar import PSAStarSolver, PSExpert, snapshot, restore  # noqa: E402

GAME_NAME = "Beams_and_Flowers"

#: Where the searches are kept between runs, in the repo's usual
#: ``data/<game>_plans.json`` shape. The engine state after reset is
#: seed-independent, so a level is solved once and every later run -- and, more to
#: the point, every `parallelize_generator.py` shard, which would each otherwise
#: redo the lot -- reads the answer. A level the search could NOT win is cached
#: too: proving that took the whole node budget, and it is just as reusable.
#: Each entry records a digest of the start state it was solved from and is
#: ignored when that stops matching, so editing the level in the .txt cannot
#: silently serve a stale plan. Delete the file to force a fresh search.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "beams_and_flowers_plans.json"

#: Engine direction -> (dr, dc).
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
_DIRS = ("up", "down", "left", "right")

#: Presses are the real cost; a rusted tile adds a whisker so that among equally
#: short walks the expert prefers the one that does not collapse a one-way tile.
#: _PRESS is large enough that no realistic number of rusted tiles on a board can
#: add up to a whole extra press.
_PRESS = 1024
_RUST = 1


class BeamsExpert(PSExpert):
    """Macro search over the real Beams and Flowers interpreter.

    States are engine grids; successors are the three macros described in the
    module docstring, each a list of primitive presses replayed through the
    interpreter. `_search` runs A* first and falls back to a breadth-first beam,
    which is what gets the mirror-heavy levels where the heuristic goes flat (the
    robot is no closer to any checkpoint for the whole stretch where it is
    re-aiming a beam at a flower).
    """

    #: Presses the epsilon detour and the exploration policy may use.
    directions = ["up", "down", "left", "right", "action"]

    #: Cheapest firing positions kept per (ACTION target, direction). See the
    #: module docstring: point blank is often inside the beam you just aimed.
    action_stands: int = 4

    #: A* node budget is `node_cap` (set by the solver); the beam gets its own.
    beam_width: int = 220
    beam_depth: int = 26
    beam_node_cap: int = 40_000

    # -- object vocabulary ---------------------------------------------------
    def setup(self) -> None:
        self._disk = self._load_disk()
        g = self.g
        idx = g.obj_name_to_idx

        def ids(*names) -> frozenset:
            return frozenset(idx[n] for n in names)

        self.void = idx["void"]
        self.wall = idx["wall"]
        self.rusted = idx["rustedfloor"]
        self.checkgate = idx["checkgate"]
        self.flower = idx["flower"]
        self.goldenblock = idx["goldenblock"]

        self.collapsed = idx["collapsedrustedfloor"]
        #: The two ways to lose. The rulebook reacts to either with ``restart``,
        #: but only on the FOLLOWING turn -- the turn a beam cuts the robot down
        #: just swaps it for a wreck and raises no flag -- so the search has to
        #: recognise the wreck itself rather than wait for the flag.
        self.doomed_ids = ids("destroyedplayer", "destroyedcheckgate")

        self.facing_of = {idx["playerup"]: "up", idx["playerdown"]: "down",
                          idx["playerleft"]: "left", idx["playerright"]: "right"}
        self.player_of = {d: o for o, d in self.facing_of.items()}
        self.player_ids = frozenset(self.facing_of)
        self.mirror_ids = ids("rmirrorne", "rmirrornw", "rmirrorsw", "rmirrorse",
                              "pmirrorne", "pmirrornw", "pmirrorsw", "pmirrorse",
                              "smirrorne", "smirrornw", "smirrorsw", "smirrorse")
        self.laser_ids = ids("laserleft", "laserright", "laserdown", "laserup")
        self.ray_ids = ids("rayup", "rayright", "rayleft", "raydown",
                           "rayne", "rayen", "raynw", "raywn",
                           "raysw", "rayws", "rayse", "rayes")
        blocks = ids("whiteblock", "iceblock", "goldenblock")

        #: Pushed one cell by walking into them.
        self.piece_ids = self.mirror_ids | blocks
        #: Blue mirrors: shoved by a remote ACTION, like a golden block. (Red ones
        #: rotate instead, and the yellow ones ignore it.)
        self.pmirror_ids = ids("pmirrorne", "pmirrornw", "pmirrorsw", "pmirrorse")
        #: Affected by a remote ACTION press at all. Everything else in the way
        #: merely stops the reach, so firing at it is a wasted macro.
        self.action_target_ids = (self.pmirror_ids | {self.goldenblock}
                                  | ids("rmirrorne", "rmirrornw", "rmirrorsw",
                                        "rmirrorse"))
        #: Stops the ACTION reach (PuzzleScript's ``ActionObstacle``).
        self.action_obstacle_ids = (self.mirror_ids | blocks | self.laser_ids
                                    | {self.wall, self.checkgate, self.flower})
        #: The robot cannot enter these: its own collision layer, plus holes.
        #: ``collapsedrustedfloor`` is a tile that has just fallen away -- the
        #: first rule of the next turn turns it into a hole, well before that
        #: turn's movement resolves, so it is already impassable.
        self.solid_ids = (self.mirror_ids | blocks | self.laser_ids
                          | {self.wall, self.void, self.collapsed})
        #: What makes a step something other than a plain walk (see `_apply_macro`).
        self.no_walk_ids = self.solid_ids | self.ray_ids | {self.flower}
        #: A shove -- by hand or by ACTION -- needs the tile BEHIND the piece to be
        #: clear of everything sharing the pieces' collision layer, and the rules
        #: spell out ``no Void`` on top of that. If it is not, the interpreter
        #: cancels the whole turn, so those macros are dropped before they are
        #: tried: each one costs a full interpreter tick to learn nothing, and in a
        #: maze most of the four directions on most pieces are blocked.
        self.push_blocked_ids = self.solid_ids | {self.checkgate, self.flower}

        #: Objects that make up the canonical state. Everything omitted is either
        #: derived from these each turn (the beams, the void borders, the HUD
        #: counters and their digits) or a one-frame marker the rules clear
        #: (targets, destroyed-*, the rusted-floor bookkeeping tag).
        self.key_ids = (self.mirror_ids | blocks | self.laser_ids | self.player_ids
                        | {self.void, self.collapsed, self.rusted, self.wall,
                           self.checkgate, self.flower})
        #: A just-collapsed tile and the hole it becomes next turn are the same
        #: state -- the difference is one rule that has not fired yet -- so the key
        #: reads them as one. Without this, whether a state arrived through the
        #: interpreter or through the walk fast path would change its identity and
        #: the search would explore both copies.
        self.key_alias = {self.collapsed: self.void}

    # -- disk plan cache -----------------------------------------------------
    def plan(self, eng, level: int | None = None) -> list | None:
        """`PSExpert.plan` with `PLAN_CACHE` behind the in-memory memo."""
        key = (None, self._key(eng))
        if key in self.cache:
            return self.cache[key]
        digest = hashlib.sha1(
            repr(sorted(key[1])).encode()).hexdigest()[:16]
        entry = self._disk.get(level)
        if entry is not None and entry["start"] == digest:
            self.cache[key] = entry["plan"]
            return entry["plan"]
        start = snapshot(eng)
        sol = self._search(eng)
        restore(eng, start)
        self.cache[key] = sol
        # Only the level's START state is worth keeping -- it is the one state
        # every seed plans from -- so a mid-trajectory re-plan never evicts it.
        if level is not None and entry is None:
            self._disk[level] = {"start": digest, "plan": sol}
            self._save_disk()
        return sol

    def _load_disk(self) -> dict:
        """``{level: {"start": digest, "plan": [...] | None}}``, or empty if
        unreadable. A cache that cannot be parsed is a miss, never a crash."""
        try:
            return {int(k): v for k, v in json.loads(PLAN_CACHE.read_text()).items()}
        except Exception:                                   # noqa: BLE001
            return {}

    def _save_disk(self) -> None:
        """Write atomically -- shards started together would otherwise interleave
        into a truncated file (see `parallelize_generator`)."""
        try:
            PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PLAN_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self._disk[k] for k in sorted(self._disk)}, indent=1))
            os.replace(tmp, PLAN_CACHE)
        except OSError:
            pass                                            # the cache is optional

    # -- canonical state -----------------------------------------------------
    def _key(self, eng) -> frozenset:
        """Exact state, INCLUDING where the robot stands and which way it faces.

        This is the plan-cache key, so it has to pin the robot down: a cached plan
        opens with a walk from one specific tile, and serving it to a robot
        standing somewhere else in the same room would replay a walk that goes
        somewhere else. `_region_key` is the loose one, and it is used only for
        dedup inside a single search."""
        keep, alias = self.key_ids, self.key_alias
        return frozenset(
            (r, c, alias.get(o, o))
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & keep)
        )

    def _region_key(self, eng, region) -> tuple:
        """State with the robot's exact tile replaced by its reachable ROOM.

        Two states with the same board and the same room admit exactly the same
        macros, so merging them cannot lose a solution -- it can only mis-cost one
        by at most the room's diameter. Without it every tile the robot could have
        stopped on is its own node and the macro search collapses back into the
        primitive one. The FACING is dropped for the same reason (any facing is one
        press away, and the walk that opens the next macro pays for it)."""
        keep, alias = self.key_ids - self.player_ids, self.key_alias
        return (frozenset(
            (r, c, alias.get(o, o))
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & keep)
        ), region)

    # -- board reading -------------------------------------------------------
    def _analyze(self, eng):
        """Return ``(region, macros, h, player, facing)`` for the engine's current
        grid, or ``(None, [], _LOST, None, None)`` when there is no robot left.

        One pass builds the walkability map (solid tiles plus the tiles a beam is
        currently crossing, which are lethal), the ACTION-reach map, and the lists
        of checkpoints, flowers, pushable pieces and ACTION targets; one Dijkstra
        over ``(tile, facing)`` then prices every macro."""
        grid = eng.grid
        height, width = len(grid), len(grid[0])
        solid, ray_ids = self.solid_ids, self.ray_ids
        walkable = [[False] * width for _ in range(height)]
        reach = [[False] * width for _ in range(height)]      # no ACTION obstacle
        rusted = [[False] * width for _ in range(height)]
        player = None
        facing = "up"
        gates: list[tuple[int, int]] = []
        flowers: list[tuple[int, int]] = []
        pieces: list[tuple[int, int]] = []
        targets: list[tuple[int, int]] = []
        for r in range(height):
            row = grid[r]
            for c in range(width):
                cell = row[c]
                if not cell:
                    walkable[r][c] = True
                    reach[r][c] = True
                    continue
                if cell & self.player_ids:
                    player = (r, c)
                    facing = self.facing_of[next(iter(cell & self.player_ids))]
                if self.checkgate in cell:
                    gates.append((r, c))
                if self.flower in cell:
                    flowers.append((r, c))
                if cell & self.piece_ids:
                    pieces.append((r, c))
                if cell & self.action_target_ids:
                    targets.append((r, c))
                if self.rusted in cell:
                    rusted[r][c] = True
                reach[r][c] = not (cell & self.action_obstacle_ids)
                walkable[r][c] = not (cell & solid or cell & ray_ids
                                      or self.flower in cell)
        if player is None:
            return None, [], _LOST, None, None

        dist, parent = self._walk(walkable, rusted, player, facing, height, width)

        def path_to(state) -> list[str]:
            out: list[str] = []
            while parent[state] is not None:
                state, press = parent[state]
                out.append(press)
            out.reverse()
            return out

        macros: list[list[str]] = []
        # 1. Collect a checkpoint by walking onto it.
        for gate in gates:
            best = min(((dist[(gate[0], gate[1], f)], f) for f in _DIRS
                        if (gate[0], gate[1], f) in dist), default=None)
            if best is not None:
                macros.append(path_to((gate[0], gate[1], best[1])))
        # 2. Push a piece by walking into it.
        for (pr, pc) in pieces:
            for d, (dr, dc) in _DELTA.items():
                stand = (pr - dr, pc - dc, d)
                if stand in dist and self._shovable(grid, pr + dr, pc + dc,
                                                   height, width):
                    macros.append(path_to(stand) + [d])
        # 3. Rotate / shove a piece from range with ACTION.
        for (tr, tc) in targets:
            for d, (dr, dc) in _DELTA.items():
                if (self.goldenblock in grid[tr][tc]
                        or grid[tr][tc] & self.pmirror_ids) and not self._shovable(
                            grid, tr + dr, tc + dc, height, width):
                    continue         # a shove the interpreter would cancel
                cands = []
                r, c = tr - dr, tc - dc
                while 0 <= r < height and 0 <= c < width and reach[r][c]:
                    state = (r, c, d)
                    if state in dist:
                        cands.append((dist[state], state))
                    r, c = r - dr, c - dc
                cands.sort()
                for _cost, state in cands[:self.action_stands]:
                    macros.append(path_to(state) + ["action"])

        region = min(k[:2] for k in dist)
        return (region, macros, self._heuristic(dist, gates, flowers),
                player, facing)

    def _shovable(self, grid, r, c, height, width) -> bool:
        """Can a piece be shoved into ``(r, c)``? See `push_blocked_ids`."""
        return (0 <= r < height and 0 <= c < width
                and not (grid[r][c] & self.push_blocked_ids))

    def _walk(self, walkable, rusted, player, facing, height, width):
        """Dijkstra over ``(row, col, facing)``: one press turns, one press steps
        forward. Returns ``(dist, parent)`` in the padded cost unit ``_PRESS``."""
        start = (player[0], player[1], facing)
        dist = {start: 0}
        parent: dict[tuple, tuple | None] = {start: None}
        queue = [(0, start)]
        while queue:
            cost, state = heapq.heappop(queue)
            if cost > dist[state]:
                continue
            r, c, f = state
            for d, (dr, dc) in _DELTA.items():
                if d == f:
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < height and 0 <= nc < width) or not walkable[nr][nc]:
                        continue
                    nxt = (nr, nc, f)
                    ncost = cost + _PRESS + (_RUST if rusted[nr][nc] else 0)
                else:
                    nxt = (r, c, d)
                    ncost = cost + _PRESS
                if ncost < dist.get(nxt, 1 << 40):
                    dist[nxt] = ncost
                    parent[nxt] = (state, d)
                    heapq.heappush(queue, (ncost, nxt))
        return dist, parent

    def _heuristic(self, dist, gates, flowers) -> int:
        """Presses still owed, guessed.

        The checkpoint term is the real one: walking to the nearest checkpoint is
        a cost the robot certainly pays, and the per-checkpoint constant keeps the
        search from dawdling once it is standing next to one. Flowers get a flat
        charge only -- what it actually costs to burn one is a mirror puzzle, and
        no cheap number describes it -- which is exactly why the flower levels are
        the ones that fall through to the beam.

        A term for how far the nearest beam runs from the nearest surviving
        flower was tried here, on the theory that it would rank the mirror
        settings throwing light roughly the right way above the ones aimed at a
        wall. It rescued NONE of the thirteen unsolved levels, so it is not in the
        code: turning a mirror teleports a beam rather than sweeping it, and the
        distance it lands at says nothing about whether the next turn helps."""
        if not gates and not flowers:
            return 0
        nearest = 0
        if gates:
            reachable = [dist[(r, c, f)] // _PRESS
                         for (r, c) in gates for f in _DIRS if (r, c, f) in dist]
            nearest = min(reachable) if reachable else _STRANDED
        return nearest + 10 * len(gates) + 14 * len(flowers)

    # -- macro replay --------------------------------------------------------
    def _apply_macro(self, eng, macro, player, facing) -> int:
        """Run one macro and return the index of the press that WON, ``-1`` if it
        finished without winning, or ``-2`` if the branch is dead (see `_doomed`).

        The win is tested after EVERY press because ``check_win`` reads a per-step
        flag: a macro that wins partway and keeps walking silently un-wins.

        Losing is the game's own reaction to the robot being shot, walking into a
        flower, or a beam cutting a checkpoint, and its reaction is ``restart``.
        The interpreter only raises the flag -- the ADAPTER is what reloads the
        level -- so the search drops the branch here instead. Restarting is never
        part of a plan: it throws away every checkpoint collected so far to arrive
        somewhere already visited.

        THE WALK FAST PATH. An interpreter tick on these boards costs ~25ms -- this
        game re-propagates every beam through two rule loops on every single press
        -- and it is the entire cost of the search. But the great majority of the
        presses in a macro are the robot walking to where the interesting press
        happens, and a walk provokes exactly three changes: the robot turns or
        moves, a checkpoint it walks into disappears, and a rusted tile it walks
        off collapses. Nothing else in the rulebook can fire, because the robot
        stops no beam and neither does a hole, so the beams are bit-for-bit the
        same afterwards. So a press whose destination is plainly clear (no wall,
        block, mirror, laser, hole, beam or flower) is applied to the grid
        directly, and only the presses that can actually start something -- ACTION,
        and any move into an occupied tile, which is a push -- go through the
        interpreter. The derived objects a skipped tick would have refreshed (the
        void borders, the remaining-checkpoint and remaining-flower digits) are
        rebuilt wholesale by the next real tick and are not part of the state key.

        ``player`` / ``facing`` are where the robot starts, which the caller has
        just read off the board in `_analyze`."""
        eng._rule_restart = False
        grid = eng.grid
        height, width = len(grid), len(grid[0])
        pos, face = player, facing
        no_walk, player_ids = self.no_walk_ids, self.player_ids
        for i, press in enumerate(macro):
            if press != "action" and face != press:
                # A turn: the robot swaps to its other sprite and stays put.
                cell = grid[pos[0]][pos[1]]
                cell -= player_ids
                cell.add(self.player_of[press])
                face = press
                eng._position_index_dirty = True
                continue                       # a turn can neither win nor kill

            step_r = step_c = None
            if press != "action":
                dr, dc = _DELTA[press]
                step_r, step_c = pos[0] + dr, pos[1] + dc
            if (step_r is not None and 0 <= step_r < height and 0 <= step_c < width
                    and not (grid[step_r][step_c] & no_walk)):
                src, dst = grid[pos[0]][pos[1]], grid[step_r][step_c]
                src -= player_ids
                if self.rusted in src:         # a one-way tile, now a hole
                    src.discard(self.rusted)
                    src.add(self.void)
                dst.discard(self.checkgate)
                dst.add(self.player_of[press])
                pos = (step_r, step_c)
                eng._position_index_dirty = True
            else:
                eng.step(press)
                if eng._rule_restart or self._doomed(eng):
                    return -2
                pos, face = self._find_player(eng)
                if pos is None:
                    return -2
            if eng.check_win():
                return i
        return -1

    def _doomed(self, eng) -> bool:
        """True once this state is a loss: the robot is a wreck, or a beam has
        cut a checkpoint. Both are only *acted on* next turn (see `doomed_ids`),
        and a shot checkpoint is the nastier one -- the gate is off the board, so
        a search that waited for the ``restart`` flag would read the state as
        "checkpoint collected" and plan happily on from a lost position."""
        doomed = self.doomed_ids
        return any(cell & doomed for row in eng.grid for cell in row)

    def _find_player(self, eng):
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                here = cell & self.player_ids
                if here:
                    return (r, c), self.facing_of[next(iter(here))]
        return None, None

    # -- search --------------------------------------------------------------
    def _search(self, eng) -> list | None:
        return self._astar(eng) or self._beam(eng)

    def _astar(self, eng) -> list | None:
        if eng.check_win():
            return []
        weight = self.weight
        start = snapshot(eng)
        region, macros, h, player, facing = self._analyze(eng)
        if region is None:
            return None
        counter = nodes = 0
        queue = [(weight * h, 0, counter, start, [], macros, player, facing)]
        best_g = {self._region_key(eng, region): 0}
        while queue:
            _f, g, _c, snap, path, macros, player, facing = heapq.heappop(queue)
            for macro in macros:
                restore(eng, snap)
                won = self._apply_macro(eng, macro, player, facing)
                nodes += 1
                if won >= 0:
                    return path + macro[:won + 1]
                if won == -2 or eng.grid == snap:
                    continue          # died / restarted, or a move the engine refused
                nregion, nmacros, nh, nplayer, nfacing = self._analyze(eng)
                if nregion is None:
                    continue
                key = self._region_key(eng, nregion)
                ng = g + len(macro)
                if best_g.get(key, 1 << 30) <= ng:
                    continue
                best_g[key] = ng
                counter += 1
                heapq.heappush(queue, (ng + weight * nh, ng, counter,
                                       snapshot(eng), path + macro, nmacros,
                                       nplayer, nfacing))
                if nodes >= self.node_cap:
                    return None
        return None

    def _beam(self, eng) -> list | None:
        """Width-capped breadth-first search over the same macros.

        A* is the right tool while the heuristic can tell good states from bad; on
        the beam-and-flower levels it cannot. The middle of those solutions is a
        run of macros that walk mirrors around without the robot getting one press
        closer to a checkpoint, so the estimate is FLAT over exactly the stretch
        that matters and A* degenerates to uniform-cost search at a hopeless depth.
        A beam spends the same interpreter budget on breadth at every depth and
        uses the heuristic only to break ties, which is all a flat heuristic can
        do. The trade is optimality: these plans win and are engine-verified, but
        they wander."""
        if eng.check_win():
            return []
        region, macros, _h, player, facing = self._analyze(eng)
        if region is None:
            return None
        frontier = [(snapshot(eng), [], macros, player, facing)]
        seen = {self._region_key(eng, region)}
        nodes = 0
        for _depth in range(self.beam_depth):
            kids = []
            for snap, path, macs, player, facing in frontier:
                for macro in macs:
                    restore(eng, snap)
                    nodes += 1
                    won = self._apply_macro(eng, macro, player, facing)
                    if won >= 0:
                        return path + macro[:won + 1]
                    if won == -2 or eng.grid == snap:
                        continue
                    nregion, nmacros, nh, nplayer, nfacing = self._analyze(eng)
                    if nregion is None:
                        continue
                    key = self._region_key(eng, nregion)
                    if key in seen:
                        continue
                    seen.add(key)
                    # Sorted on the heuristic ALONE: the tuple also carries a
                    # snapshot (a list of sets), which has no ordering.
                    kids.append((nh, snapshot(eng), path + macro, nmacros,
                                 nplayer, nfacing))
                    if nodes >= self.beam_node_cap:
                        return None
            if not kids:
                return None                     # the reachable space closed
            kids.sort(key=lambda kid: kid[0])
            frontier = [kid[1:] for kid in kids[:self.beam_width]]
        return None


#: Heuristic charge for a state with no robot in it (the level is mid-restart).
_LOST = 1_000_000
#: ... and for a checkpoint the robot cannot currently walk to at all (walled off
#: by a beam it has to re-aim first). Finite, so the search stays complete.
_STRANDED = 500


class BeamsAndFlowersSolver(PSAStarSolver):
    """Generator for ps:beams_and_flowers. See the module docstring."""

    game_id = "puzzlescript_beams_and_flowers"
    game_name = GAME_NAME
    expert_cls = BeamsExpert

    #: Macro budgets, in macros tried rather than seconds, so which levels solve
    #: is a property of the code and not of how busy the machine was. Both are
    #: spent in full on a level that cannot be won -- A* first, then the beam --
    #: which is why a failure is worth caching to disk just as much as a plan is.
    node_cap = 40_000
    #: Weighted: at w=1 the searches that finish quickly still finish, and the
    #: ones that do not are not close enough for optimality to be the thing
    #: standing in the way. The heuristic is not admissible anyway (its
    #: per-checkpoint constant is a guess), so w=1 buys no guarantee here.
    weight = 3
    #: The adapter ends an episode at 200 counted moves, so a plan longer than
    #: that cannot be recorded no matter how long the search is given. The longest
    #: kept plan is level 16's at 177.
    max_steps = 200

    #: Levels the search cannot win; see "Levels" in the module docstring. Listed
    #: up front so a run that starts with no `PLAN_CACHE` does not spend the full
    #: node budget on each of them (~45 minutes apiece) to re-learn it.
    skip_levels: frozenset[int] = frozenset(
        {11, 19, 24, 25, 27, 29, 32, 33, 34, 36, 37, 38, 39})


if __name__ == "__main__":
    sys.exit(BeamsAndFlowersSolver.main())
