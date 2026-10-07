"""Generate Phase-1 training data for the PuzzleScript game ps:escaping_limbo
("Escaping Limbo", Doublepancake).

The harness -- the rotation contract, the trajectory recorder, the plan cache
and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`, shared
with the other ps: generators. This file is the game-specific part: a native
model of the mechanic, a macro A* over it, the exact optimal-action oracle, and
the two rendering/level fixes the shipped game needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_escaping_limbo",
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
action (post rotation/flip remap), i.e. the button an agent presses in the
augmented view, so replaying the recorded actions reproduces the recorded
frames exactly. Each expert step also carries the full set of equally-optimal
presses (see "Optimal-action sets" below).

The game
--------
Walk the blue square onto the blue goal square (``all player on goal``). Four
directions, no ACTION button. Between you and the goal:

* **Stones** you push, Sokoban style, and they CHAIN: a push shoves a whole
  contiguous run of them, not just the first.
* **Grates** -- light grey floor. You may walk on a grate; a stone may never
  enter one, and the rule that says so is ``cancel``, so shoving a stone at a
  grate does not just fail, it **annuls the entire turn**. Same for shoving a
  stone at a closed door.
* **Buttons and doors**, in two independent colours. Every pink door is open
  (and passable, by the player and by stones) exactly while **every** pink
  button has a stone on it -- see `_Board.closed`, because that conjunction is
  not what the rules look like they say. Only stones press buttons; standing
  on one yourself does nothing, which is the whole puzzle.

Two consequences worth stating because they drive the solver:

* **A closing door DESTROYS whatever is standing in it** -- silently, with no
  death animation and no GAME_OVER. That is faithful PuzzleScript: an object
  created on an occupied collision layer replaces what was there. In practice
  only stones are ever caught (`_Board.step` has the argument for why the
  player cannot be), so it is a way to lose a stone you needed, not a way to
  die.
* **The win is REACHING somewhere, not placing stones.** Nothing has to end up
  anywhere; a stone shoved into a dead corner is a fine outcome unless it was
  the corner you needed. So this is not a sokoban and the sokoban deadlock
  tests would be backwards here -- the same argument as ps:escape.

Why a native model (and not the shared engine-blackbox A*)
----------------------------------------------------------
Measured first, per the ps:entrepotphage_demake lesson: this game's
interpreter runs at **375-575 ``eng.step``/s**, against ~20-25k for a plain
push game. The cost is the four ``late`` door rules -- each is a two-bracket
rule that re-scans the whole board for every door cell on every turn, whether
or not a button changed. A search that pays that per node is ~50x over budget
for a mechanic that fits in forty lines, so `_Board` re-implements the turn and
the interpreter only ever **certifies**: `--fuzz` checks the model against it
cell-for-cell over random play, and `record_level` replays the finished plan
through the real adapter to a real WIN.

`_Board` is a bitmask model -- stones are one int, not a frozenset of cells --
and the whole state is ``(player cell, stones)``. Door state is a *function* of
the stones (a button is pressed iff a stone is on it), so it is not part of the
key and can never go stale.

The search
----------
`_Board.solve` is A* whose successors are ``walk to the push cell, then push``
macros plus one ``walk onto the goal`` macro -- the same shape as
`PSPushExpert`, re-implemented over the model rather than the interpreter.

* ``g`` counts primitive MOVES (what the agent pays), and every walk inside a
  macro is a shortest path, so a plan is shortest over primitives: any
  primitive solution splits into maximal walk runs each ending in a push (or,
  once, on the goal), and those are exactly the macros.
* Dedup is on the EXACT ``(player, stones)`` pair, not on the player's
  reachable region. The region key is the standard sokoban canonicalisation
  and `PSPushExpert` documents it as costing at most a region diameter -- but
  that is a real loss of optimality for a move-costed search (it cost
  ps:entrepotphage_demake 11 moves over 11 levels). After a macro the player
  always stands on the pushed stone's old cell, so the exact key barely grows
  the space.
* The heuristic is the player's own shortest walk to the goal counting only
  WALLS and permanently-immovable stones as obstacles -- closed doors are
  treated as open, because they can be opened. It is admissible: the player
  moves one non-wall cell per press. A stone is called permanently immovable
  only when a WALL forbids the stand cell (or a wall/grate forbids the landing
  cell) on every axis, which no future move can change; a stone merely blocked
  by another stone comes free when its neighbour moves and is deliberately not
  counted.

Levels
------
Nine, shipped, ordered as the author wrote them -- three stone-free mazes that
teach the goal, then stones, then grates, then one button, then the full
two-circuit boards (``--plans`` re-derives this table)::

    level  size   stones doors  plan  ties  search   what it teaches
     0      8x3      0     0      5    0%    0.0s   walk to the goal
     1      8x8      0     0     10   50%    0.0s   walk around a wall
     2     13x9      0     0     20   30%    0.0s   a maze, still no stones
     3      8x8      4     0     13   23%    0.0s   stones are furniture
     4      9x9     12     0     11   18%    0.0s   grates: a stone shoved at
                                                    one annuls the whole turn
     5      7x9      1     3     16   25%    0.0s   one button, one door bank
     6     11x17     1     3     46    0%    1.1s   the button is a long walk
                                                    through a grate labyrinth
     7      8x18     2     5     62    5%    5.9s   two circuits, two stones
     8      9x19     4     6     42   17%    4.0s   two PINK buttons, so both
                                                    must be held at once

``ties`` is the share of steps with more than one equally-optimal press. 225
moves over 9 levels, ~11s of one-time search cached to
``data/escaping_limbo_plans.json`` (cold and warm runs verified byte-identical).
Delete the file to re-derive.

**Level 8 shipped unwinnable and is fixed here.** Its route needs the pink
doors open to enter the middle pocket and the orange doors open to leave it,
and it has exactly one stone that can reach any button (the other is sealed
inside the pocket) -- so the two circuits can never be held down at once.
Exhaustive primitive BFS over the level's whole reachable space (640 states)
finds no win. One stone was added on the right-hand side, where it rides up to
the top pink button while the original goes left onto the orange one; the
board's three buttons and two door colours make that plainly the intended
shape. Nothing else about the level changed.

Optimal-action sets
-------------------
Measured, not inferred, by the same oracle ps:escape uses: at each step,
re-solve from each successor and keep every press that still finishes in the
moves remaining. It has to be measured here for the reason it does there --
`PSPushExpert.annotate_walks` calls any step that moved a stone forced, which
is right in a sokoban and wrong in a game where shoving a stone aside is just
how you walk down a corridor. Two sound prunes keep it cheap (a press that
leaves the board untouched cannot be on a shortest path, nor can one whose
admissible estimate already exceeds what is left). No step ever ships
unlabelled.

The rendering fixes
-------------------
Audited per-composition at every cell size the boards use (``--audit``), which
found three things wrong with the shipped art -- all of them the recurring
kinds catalogued for this adapter:

* **The open pink door was invisible.** ``pinkx`` (the object a pink door
  becomes when the circuit is live) was ``#f9e0ef``, a pale tint that
  quantizes onto ARC 0 -- the Background's own colour. Half of the game's
  state, on the boards the game is named for, rendered as bare floor. It is
  now ARC 7, the light tint of its own door's hue; ``orangex`` likewise moved
  off ARC 7 onto ARC 11 so the two circuits stay distinct.
* **The buttons were pixel-identical to the doors.** Both of ``pinkbutton``'s
  colours quantize to ARC 6 and both of ``orangebutton``'s to ARC 12, so each
  button's 5x5 sprite flattened into exactly the solid block its door renders
  as. They are now hollow rings, which also gives the stone standing on one
  something to show through.
* **Bodies hid the ground they stood on.** ``Goal`` was a 3x3 centre block and
  so is ``Player``, so "player on goal" -- the winning frame, the one cell
  stack the corpus cannot be blind to -- was pixel-identical to the player on
  bare floor. Goal is now a solid square and Player and Stone carry
  transparent corners, so every ``(ground, body)`` pair shows both. Corners
  specifically: `_render_cell_sprite` samples a 5x5 sprite at rows/cols
  {0,2,4} at cell_px 3 (this game's widest boards) and {0,1,3,4} at 4, and the
  corners are the only holes that survive both.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed
ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`). The game is gravity-free, its push rules
are stated with the relative ``>`` force, its win condition names no direction
and its input is screen-relative, so a flip is an exact symmetry of the
mechanic; no sprite is directional, so no mirror can show art the game does
not own. There is deliberately no colour augmentation: the pink/orange circuit
pairing IS the mechanic, and a recolor that broke the door-to-button colour
link would make the boards unreadable. 9 levels x 16 presentations = 144.

The expert plan is therefore seed-independent: solved once per level, cached to
``data/escaping_limbo_plans.json``, and replayed per seed with that seed's
remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_escaping_limbo_training.py --episodes 200 \
        --out data/training_multi_level/escaping_limbo

    python solvers/generate_escaping_limbo_training.py --plans  # level report
    python solvers/generate_escaping_limbo_training.py --audit  # render audit
    python solvers/generate_escaping_limbo_training.py --fuzz   # model vs engine
"""

from __future__ import annotations

import heapq
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (PSAStarSolver, PSExpert,      # noqa: E402
                                     Plan)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Escaping_Limbo"

#: Disk cache of each level's start plan and its optimal-action sets. The
#: searches are seed-independent, so without a file on disk every shard
#: `parallelize_generator` starts would re-derive all of them. Delete to
#: re-derive.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "escaping_limbo_plans.json"

#: Engine direction -> (dr, dc). The game has no ACTION button.
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
_DIRS = ("up", "down", "left", "right")

#: Heuristic charge for a state that can never win -- the player crushed by a
#: closing door, or the goal sealed off behind stones no push can ever move.
#: Large enough to sink the node, finite so the search stays complete.
_UNREACHABLE = 10_000


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry plus the turn rule, over ``(player, stones)``.

    Cells are flat indices ``r * w + c``; ``stones`` is a bitmask over them, so
    a state is two ints and hashes for free. Door state is NOT stored: a
    circuit is live exactly when a stone sits on one of its buttons, so it is
    recomputed from ``stones`` and can never disagree with it.

    The turn, in the order the interpreter resolves it (verified by ``--fuzz``,
    which replays random play against the real engine and compares the board
    cell-for-cell):

    1. The player's target cell is read against the CURRENT doors -- walls and
       closed doors block, grates and open doors do not.
    2. If the target holds a stone, the whole contiguous run of stones behind
       it moves with it (the chain rule). The run is refused -- meaning the
       turn does nothing at all, not that the stones stop -- if the cell past
       its end is off the board, a wall, a grate or a closed door. The last
       two are the game's ``cancel`` rules; the first two are ordinary
       collision, and both come out as "nothing happened" either way.
    3. Buttons are re-read from the NEW stone positions (the door rules are
       ``late``, i.e. after movement), and every door of a circuit that is no
       longer live closes -- **destroying any stone or player standing in it**,
       because a closing door is an object created on an occupied collision
       layer. A player destroyed this way is gone for good and the level can
       no longer be won.
    """

    __slots__ = ("h", "w", "wall", "grate", "goal", "doors", "buttons",
                 "start", "_free_cache", "_dist_cache")

    def __init__(self, h, w, wall, grate, goal, doors, buttons, start):
        self.h, self.w = h, w
        self.wall: int = wall            # bitmask
        self.grate: int = grate          # bitmask
        self.goal: int = goal            # cell index
        #: circuit -> bitmask of its door cells
        self.doors: dict[str, int] = doors
        #: circuit -> bitmask of its button cells
        self.buttons: dict[str, int] = buttons
        #: ``(player cell, stones bitmask)`` on the grid this was read from
        self.start: tuple[int, int] = start
        self._free_cache: dict = {}
        self._dist_cache: dict = {}
        # `step`'s single-pass door resolution relies on this: a crushed stone
        # is standing in a doorway, so if a doorway could also be a button the
        # crush could darken a second circuit that the late rules have already
        # been past this turn.
        for circuit, cells in self.doors.items():
            assert not (cells & self.buttons[circuit]), \
                f"{circuit} has a door and a button on the same cell"

    # -- construction ---------------------------------------------------------
    #: object name -> the board field it fills. A door contributes its cell
    #: whichever of its two forms is on the grid: reading a board from a
    #: mid-level grid would otherwise silently forget every door that happens
    #: to be OPEN at that moment, and the model would let the player walk
    #: through it forever after.
    _FIELDS = (("wall", "wall"), ("grate", "grate"), ("goal", "goal"),
               ("stone", "stones"), ("player", "player"),
               ("pinkdoor", "door.pink"), ("pinkx", "door.pink"),
               ("orangedoor", "door.orange"), ("orangex", "door.orange"),
               ("pinkbutton", "button.pink"),
               ("orangebutton", "button.orange"))

    @classmethod
    def from_engine(cls, eng, game) -> "_Board":
        """Read a board (geometry + the grid's current state) off a live
        interpreter."""
        ids = [(set(game.resolve_object_name(n)), field)
               for n, field in cls._FIELDS]
        h, w = eng.height, eng.width
        masks = {"wall": 0, "grate": 0, "stones": 0,
                 "door.pink": 0, "door.orange": 0,
                 "button.pink": 0, "button.orange": 0}
        goal = player = -1
        for r in range(h):
            for c in range(w):
                cell, bit = eng.grid[r][c], 1 << (r * w + c)
                if not cell:
                    continue
                for obj_ids, field in ids:
                    if not (cell & obj_ids):
                        continue
                    if field == "goal":
                        goal = r * w + c
                    elif field == "player":
                        player = r * w + c
                    else:
                        masks[field] |= bit
        return cls(h, w, masks["wall"], masks["grate"], goal,
                   {"pink": masks["door.pink"],
                    "orange": masks["door.orange"]},
                   {"pink": masks["button.pink"],
                    "orange": masks["button.orange"]},
                   (player, masks["stones"]))

    @staticmethod
    def read_state(eng, stone_ids, player_ids) -> tuple[int, int]:
        """``(player cell, stones)`` for a live grid of this board's level."""
        w = eng.width
        player, stones = -1, 0
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & stone_ids:
                    stones |= 1 << (r * w + c)
                if cell & player_ids:
                    player = r * w + c
        return (player, stones)

    # -- the turn -------------------------------------------------------------
    def closed(self, stones: int) -> int:
        """Bitmask of the door cells that are shut, given these stone positions.

        A circuit is live only when **every one** of its buttons holds a stone
        -- not merely one of them. That is not obvious from the rules and it is
        the single most important thing the model has to get right, so: the two
        late rules run in order, and

            late [stone pinkbutton] [pinkdoor] -> [stone pinkbutton] [pinkx]

        opens every shut door as soon as SOME button is held, but

            late [no stone pinkbutton] [pinkx] -> [no stone pinkbutton] [pinkdoor]

        then shuts every open door again as soon as SOME button is bare. With
        one button per colour the two readings coincide, which is why eight of
        the nine levels cannot tell them apart; level 8 has two pink buttons
        and it is unambiguous there (``--fuzz`` caught exactly this). A colour
        with no button at all is shut forever -- neither rule can match."""
        out = 0
        for circuit, cells in self.doors.items():
            held = self.buttons[circuit]
            if not held or (stones & held) != held:
                out |= cells
        return out

    def step(self, state, direction):
        """The state after one press. Returns ``state`` itself when the press
        does nothing (blocked, or a turn the game cancels outright)."""
        player, stones = state
        if player < 0:                      # crushed earlier: nothing responds
            return state
        w, h = self.w, self.h
        dr, dc = _DELTA[direction]
        r, c = divmod(player, w)
        nr, nc = r + dr, c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            return state
        target = nr * w + nc
        shut = self.closed(stones)
        if (self.wall | shut) & (1 << target):
            return state

        if stones & (1 << target):
            # Chain push: walk the contiguous run of stones along ``direction``
            # and find the cell past its end.
            run = []
            cur_r, cur_c = nr, nc
            while stones & (1 << (cur_r * w + cur_c)):
                run.append(cur_r * w + cur_c)
                cur_r, cur_c = cur_r + dr, cur_c + dc
                if not (0 <= cur_r < h and 0 <= cur_c < w):
                    return state            # the run is against the board edge
            end = cur_r * w + cur_c
            if (self.wall | self.grate | shut) & (1 << end):
                return state                # collision, or a cancelled turn
            step_by = dr * w + dc
            for cell in run:
                stones &= ~(1 << cell)
            for cell in run:
                stones |= 1 << (cell + step_by)
            player = target
        else:
            player = target

        # Late rules: doors of a circuit that just went dark close on whatever
        # is standing in them, because a door is an object created on an
        # occupied collision layer and that REPLACES what was there.
        #
        # One pass is exact, and two facts make it so. (a) A crushed stone
        # cannot itself release a button and cascade into a second circuit
        # closing: it was standing in a doorway, and no level puts a door and a
        # button on the same cell (asserted in `__init__`). (b) The PLAYER can
        # never be caught at all -- the only cell a move takes a stone off is
        # the head of the pushed run, which is exactly the cell the player ends
        # on, so a button that goes bare is a cell the player is standing on,
        # and by (a) that cell is not a door. The branch below is kept anyway:
        # it is what the interpreter would do, it costs one test, and an
        # authored board (or a future level) that broke the premise would
        # otherwise diverge silently.
        crush = self.closed(stones)
        if crush:
            stones &= ~crush
            if crush & (1 << player):
                player = -1
        return (player, stones)

    def won(self, state) -> bool:
        player, _stones = state
        return player >= 0 and player == self.goal

    # -- reachability ---------------------------------------------------------
    def _walk_bfs(self, state):
        """``(parent, blocked)`` for the player's walk over the current board.

        ``parent`` is the shortest-walk tree (``{cell: (prev, direction)}``,
        root mapped to None), ``blocked`` the bitmask the player may not enter:
        walls, shut doors and stones. Grates are walkable -- they stop stones,
        not people."""
        player, stones = state
        blocked = self.wall | self.closed(stones) | stones
        w, h = self.w, self.h
        parent = {player: None}
        queue = deque([player])
        while queue:
            cur = queue.popleft()
            r, c = divmod(cur, w)
            for d, (dr, dc) in _DELTA.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                nxt = nr * w + nc
                if blocked & (1 << nxt) or nxt in parent:
                    continue
                parent[nxt] = (cur, d)
                queue.append(nxt)
        return parent, blocked

    @staticmethod
    def _walk_to(parent, cell) -> list:
        out = []
        while parent[cell] is not None:
            cell, d = parent[cell]
            out.append(d)
        out.reverse()
        return out

    def macros(self, state) -> list[list[str]]:
        """``walk to a push cell then push`` for every stone the player can get
        behind, plus ``walk onto the goal``.

        The goal walk is the only winning move and no push enumeration can
        express it -- without it the search closes with the goal sitting
        unvisited in the player's own region (the ps:escape trap)."""
        player, stones = state
        if player < 0:
            return []
        parent, _blocked = self._walk_bfs(state)
        w, h = self.w, self.h
        out = []
        bits = stones
        while bits:
            low = bits & -bits
            cell = low.bit_length() - 1
            bits ^= low
            r, c = divmod(cell, w)
            for d, (dr, dc) in _DELTA.items():
                sr, sc = r - dr, c - dc
                if not (0 <= sr < h and 0 <= sc < w):
                    continue
                stand = sr * w + sc
                if stand in parent:
                    out.append(self._walk_to(parent, stand) + [d])
        if self.goal in parent and self.goal != player:
            out.append(self._walk_to(parent, self.goal))
        return out

    # -- heuristic ------------------------------------------------------------
    def _immovable(self, stones: int) -> int:
        """Stones no push can EVER move, hence permanent walls.

        Pushing the stone at ``X`` along ``d`` needs the player's stand cell
        ``X - d`` to exist and not be a wall, and the landing cell ``X + d`` to
        exist and be neither wall nor grate. When no axis offers that, nothing
        in this game can relocate the stone -- only a push moves one -- so its
        cell is impassable for the rest of the level.

        Only walls and grates are consulted, never other stones or doors: a
        stone blocked by a stone comes free when its neighbour moves, and a
        door opens, so counting either would be an unsound prune rather than a
        sharper estimate."""
        got = self._free_cache.get(stones)
        if got is not None:
            return got
        w, h = self.w, self.h
        out = 0
        bits = stones
        while bits:
            low = bits & -bits
            cell = low.bit_length() - 1
            bits ^= low
            r, c = divmod(cell, w)
            for dr, dc in _DELTA.values():
                ar, ac, br, bc = r + dr, c + dc, r - dr, c - dc
                if not (0 <= ar < h and 0 <= ac < w
                        and 0 <= br < h and 0 <= bc < w):
                    continue
                if ((self.wall | self.grate) & (1 << (ar * w + ac))
                        or self.wall & (1 << (br * w + bc))):
                    continue
                break
            else:
                out |= low
        self._free_cache[stones] = out
        return out

    def heuristic(self, state) -> int:
        """Player's shortest walk to the goal over non-wall, non-frozen cells,
        in primitive moves -- the unit ``g`` is counted in.

        Closed doors are treated as OPEN and ordinary stones as passable: the
        player crosses a stone's cell in one move by pushing it, so charging
        for either would break admissibility."""
        player, stones = state
        if player < 0:
            return _UNREACHABLE
        if player == self.goal:
            return 0
        blocked = self.wall | self._immovable(stones)
        key = (player, blocked)
        got = self._dist_cache.get(key)
        if got is not None:
            return got
        w, h = self.w, self.h
        seen = {player}
        queue = deque([(player, 0)])
        out = _UNREACHABLE
        while queue:
            cur, d = queue.popleft()
            r, c = divmod(cur, w)
            for dr, dc in _DELTA.values():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                nxt = nr * w + nc
                if blocked & (1 << nxt) or nxt in seen:
                    continue
                if nxt == self.goal:
                    out = d + 1
                    queue.clear()
                    break
                seen.add(nxt)
                queue.append((nxt, d + 1))
        self._dist_cache[key] = out
        return out

    # -- search ---------------------------------------------------------------
    def astar(self, state, node_cap: int = 2_000_000) -> list | None:
        """Shortest primitive press sequence from ``state`` to a win, or None.

        A* over the macros, ``g`` in primitive moves, dedup on the exact
        ``(player, stones)`` pair. Every walk inside a macro is a shortest
        path and the heuristic is admissible at ``weight = 1``, so the result
        is shortest over primitives: any primitive solution splits into
        maximal walk runs each ending in a push (or, once, on the goal), and
        those runs are exactly what `macros` enumerates."""
        if self.won(state):
            return []
        counter = 0
        nodes = 0
        pq = [(self.heuristic(state), 0, counter, state, [])]
        best_g = {state: 0}
        while pq:
            _f, g, _c, cur, path = heapq.heappop(pq)
            if best_g.get(cur, -1) != g:
                continue                       # superseded by a cheaper route
            for macro in self.macros(cur):
                nxt = cur
                won_at = -1
                for i, d in enumerate(macro):
                    nxt = self.step(nxt, d)
                    if self.won(nxt):
                        won_at = i             # truncate: a later step un-wins
                        break
                nodes += 1
                if won_at >= 0:
                    return path + macro[:won_at + 1]
                if nxt == cur or nxt[0] < 0:
                    continue                   # refused, or the player is dead
                ng = g + len(macro)
                if best_g.get(nxt, 1 << 30) <= ng:
                    continue
                best_g[nxt] = ng
                counter += 1
                heapq.heappush(pq, (ng + self.heuristic(nxt), ng, counter,
                                    nxt, path + macro))
            if nodes >= node_cap:
                return None
        return None

    def solve(self, state, node_cap: int = 2_000_000) -> "Plan | None":
        """The level's plan plus a MEASURED optimal set for each of its steps.

        The sets are measured rather than inferred for the reason ps:escape
        measures its own: a step that moved a stone is not automatically
        forced here, because shoving a stone aside is frequently just how you
        walk down a corridor, and several directions genuinely finish in the
        same number of moves.

        At each step, re-solve from each successor and keep every press that
        still finishes in the moves remaining. Two prunes make that cheap and
        both are sound: a press that leaves the board untouched cannot be on a
        shortest path (it wastes a move to reach the state it started from),
        and neither can one whose ADMISSIBLE estimate already exceeds what is
        left."""
        plan = self.astar(state, node_cap)
        if plan is None:
            return None
        exact: dict = {}

        def remaining_from(state) -> int | None:
            got = exact.get(state)
            if got is None and state not in exact:
                sol = self.astar(state, node_cap)
                got = None if sol is None else len(sol)
                exact[state] = got
            return got

        optsets = []
        cur = state
        for i, taken in enumerate(plan):
            left = len(plan) - i               # moves left once this is taken
            best = []
            for d in _DIRS:
                nxt = self.step(cur, d)
                if self.won(nxt):
                    cost = 1
                elif nxt == cur or self.heuristic(nxt) > left - 1:
                    cost = None                # sound prunes; see above
                else:
                    sub = remaining_from(nxt)
                    cost = None if sub is None else 1 + sub
                if cost == left:
                    best.append(d)
            # The recorded step ties with itself by construction; a
            # disagreement would mean the two sides were measured differently,
            # so fall back to labelling what the expert actually did rather
            # than shipping a set that does not contain it.
            optsets.append(best if taken in best else [taken])
            cur = self.step(cur, taken)
        return Plan(plan, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class LimboExpert(PSExpert):
    """`PSExpert`'s plan cache and snapshot discipline around a NATIVE search.

    `_search` is overridden to build a `_Board` off the interpreter grid and
    solve that; the interpreter is never stepped by the planner. The memo, the
    on-disk plan cache with its staleness check and the restore discipline all
    come from the base class unchanged, which is the pattern
    ps:crocodiles_love_cookies / ps:dotsnake / ps:epicjamgame established for
    "the interpreter is too slow to search but is still the authority".

    `heuristic` is therefore never called and asserts rather than returning a
    number nothing would use.
    """

    #: `_key` is dynamic-objects-only, canonical only WITHIN a level.
    scope_by_level = True
    #: Keep each level's start plan (and its optimal sets) on disk.
    plan_cache_path = PLAN_CACHE
    #: No ACTION button in this game.
    directions = list(_DIRS)

    def setup(self) -> None:
        self.stone_ids = set(self.g.resolve_object_name("stone"))
        self.player_ids = set(self.g.resolve_object_name("player"))
        self.wall_ids = set(self.g.resolve_object_name("wall"))
        self.dyn_ids = self.stone_ids | self.player_ids
        self._boards: dict = {}

    def _key(self, eng) -> frozenset:
        # Stones + player. Everything else (walls, grates, goal, buttons) is
        # static per level, and the DOORS are a function of the stones, so this
        # is an exact state -- hence ``scope_by_level``.
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    def heuristic(self, eng) -> int:
        raise AssertionError("LimboExpert plans natively; heuristic is unused")

    def board(self, eng) -> _Board:
        """The `_Board` for the engine's current level, memoized on its walls.

        Only the GEOMETRY is reused -- the state to plan from is read
        separately -- and the caches a board accumulates (walk distances,
        immovable-stone sets) are worth keeping across the many searches one
        level's optimal-set oracle runs. Walls are never created or destroyed
        by any rule here, so the wall map identifies a level uniquely."""
        key = tuple(tuple(bool(cell & self.wall_ids) for cell in row)
                    for row in eng.grid)
        got = self._boards.get(key)
        if got is None:
            got = _Board.from_engine(eng, self.g)
            self._boards[key] = got
        return got

    def _search(self, eng) -> "Plan | None":
        board = self.board(eng)
        state = _Board.read_state(eng, self.stone_ids, self.player_ids)
        return board.solve(state, node_cap=self.node_cap)


class LimboSolver(PSAStarSolver):
    game_id = "puzzlescript_escaping_limbo"
    game_name = GAME_NAME
    expert_cls = LimboExpert

    #: Record against the GAME FOLDER's adapter. ``games/ps:escaping_limbo``
    #: is a plain passthrough today; it is named anyway so a step cap or sprite
    #: patch added there later cannot silently make this generator tape a game
    #: nobody plays (the ps:count_mover trap).
    game_module_id = "ps:escaping_limbo"

    #: Runaway guard on the native search, not a tuning dial -- the hardest
    #: level expands four orders of magnitude fewer nodes than this.
    node_cap = 2_000_000
    #: The longest plan is 62 moves; the rest is room for the RESET exploration
    #: prefix and a re-plan after it. Stays under the adapter's own 200-step
    #: per-level budget, which would otherwise flip a level to GAME_OVER
    #: mid-plan.
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

def _report() -> int:
    """Per-level board size, piece counts, plan length and tie coverage."""
    import time
    game = LimboSolver().make_game(0)
    expert = LimboExpert(game, node_cap=LimboSolver.node_cap)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = _Board.from_engine(eng, game._game)
        stones = bin(board.start[1]).count("1")
        doors = sum(bin(m).count("1") for m in board.doors.values())
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        if found is None:
            print(f"level {level}: NO PLAN")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level}: {eng.height:2d}x{eng.width:2d} "
              f"{stones:2d} stones {doors:2d} doors, {len(found):3d} moves "
              f"(budget {game._max_steps}, {room}), "
              f"{ties:3d} steps with a tie set "
              f"({ties / max(1, len(found)):3.0%}), {dt:6.1f}s")
    print(f"total {total} moves over {game.n_levels} levels")
    return 0


#: Cell legend for `_SCENARIOS`. Upper case is a STACK -- a body standing on a
#: ground tile -- which the shipped levels have no character for because they
#: never author one, but which normal play reaches constantly.
_SCEN_LEGEND = {
    ".": (), "#": ("wall",), ":": ("grate",), "g": ("goal",),
    "p": ("player",), "h": ("stone",),
    "a": ("pinkbutton",), "b": ("pinkdoor",), "x": ("pinkx",),
    "c": ("orangebutton",), "d": ("orangedoor",), "y": ("orangex",),
    "A": ("pinkbutton", "stone"), "C": ("orangebutton", "stone"),
    "X": ("pinkx", "stone"), "P": ("pinkx", "player"),
}

#: Hand-built boards for the turns random play does not reach: chain pushes,
#: the three ways a push is refused, and a door closing on a stone. Each is
#: ``(name, rows, presses)`` and is checked against the interpreter press by
#: press exactly like the random runs. They are not levels -- they are the
#: model's unit tests, written as boards because that is what they are about.
_SCENARIOS = (
    # A push shoves a whole contiguous RUN of stones, not just the first.
    ("chain of two", ["..phh.."], "right right right"),
    ("chain of three", [".phhh.."], "right right"),
    # ...and is refused outright when the far end has nowhere to go. All three
    # of these leave the board untouched, but for different reasons: ordinary
    # collision, and the game's two `cancel` rules.
    ("chain into a wall", ["#phhh#"], "right"),
    ("chain into a grate", [".phh:."], "right"),
    ("chain into a shut door", [".phhb.a"], "right"),
    ("chain against the edge", [".phh"], "right"),
    # One button of two is not enough: the second late rule shuts the doors
    # again while any button of the colour is bare.
    ("one of two pink buttons", [".pha.a.b."], "right right"),
    ("then the second", ["..hpa", "....a", "....b"], "left left down down"),
    # A stone may be pushed onto an OPEN door -- and is destroyed if the same
    # push is what closes it. (The player cannot be caught this way: see
    # `_Board.step`.)
    ("stone crushed by its door", ["phab."], "right right"),
    ("stone rides an open door", [".pXA.."], "right"),
    # The two circuits are independent.
    ("orange ignores pink", [".phc.b.d.a"], "right"),
)


def _load_scenario(game, rows):
    """Load an ASCII scenario board onto the interpreter and return
    ``(board, state, engine)`` ready to step. The engine's grid is replaced
    wholesale, the way `_audit` does it."""
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    h, w = len(rows), max(len(r) for r in rows)
    eng.height, eng.width = h, w
    eng.grid = [[{idx["background"]}
                 | {idx[o] for o in _SCEN_LEGEND[(row + "." * w)[c]]}
                 for c in range(w)] for row in rows]
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()
    board = _Board.from_engine(eng, g)
    return board, board.start, eng


class _Referee:
    """Steps `_Board` and the interpreter side by side and compares the board.

    Everything visible is compared -- the player, the stones AND the two door
    objects the model does not store but DERIVES from the stone positions --
    so a wrong `_Board.closed` is caught as surely as a wrong push. It also
    counts what each run actually exercised, because a fuzz that never opens a
    door proves nothing about the half of the mechanic this game is named
    for."""

    def __init__(self, game):
        g = game._game
        self.game = game
        self.stone_ids = set(g.resolve_object_name("stone"))
        self.player_ids = set(g.resolve_object_name("player"))
        self.open_ids = {"pink": set(g.resolve_object_name("pinkx")),
                         "orange": set(g.resolve_object_name("orangex"))}
        self.cover = dict.fromkeys(
            ("push", "chain", "cancel", "open", "close", "crushed"), 0)

    def press(self, eng, board, state, d):
        """Take ``d`` on both. Returns ``(new state, complaint or None)``."""
        w = board.w
        before, shut_before = state, board.closed(state[1])
        # Was this a chain? Read the run off the BEFORE state: after the push
        # the stones overlap their own old cells, so counting changed bits
        # would report a chain of two as a single move.
        dr, dc = _DELTA[d]
        run = 0
        if before[0] >= 0:
            r, c = divmod(before[0], w)
            while True:
                r, c = r + dr, c + dc
                if not (0 <= r < board.h and 0 <= c < w
                        and before[1] & (1 << (r * w + c))):
                    break
                run += 1

        eng.step(d)
        state = board.step(state, d)
        shut = board.closed(state[1])
        if before[0] >= 0:
            if state == before:
                self.cover["cancel"] += run > 0
            elif run:
                self.cover["push"] += 1
                self.cover["chain"] += run > 1
        self.cover["open"] += bool(shut_before & ~shut)
        self.cover["close"] += bool(shut & ~shut_before)
        # A closing door destroys what stands in it. Only stones are ever
        # caught: see `_Board.step` for why the player cannot be.
        self.cover["crushed"] += (bin(state[1]).count("1")
                                  < bin(before[1]).count("1")
                                  or before[0] >= 0 > state[0])

        got_p, got_s, got_open = -1, 0, {"pink": 0, "orange": 0}
        for r in range(board.h):
            for c in range(w):
                cell, bit = eng.grid[r][c], 1 << (r * w + c)
                if cell & self.stone_ids:
                    got_s |= bit
                if cell & self.player_ids:
                    got_p = r * w + c
                for circuit, ids in self.open_ids.items():
                    if cell & ids:
                        got_open[circuit] |= bit
        want_open = {k: board.doors[k] & ~shut for k in board.doors}
        if (got_p, got_s) != state or got_open != want_open:
            return state, (f"after {d!r}: engine (player={got_p}, "
                           f"stones={got_s:#x}, open={got_open}) vs model "
                           f"(player={state[0]}, stones={state[1]:#x}, "
                           f"open={want_open})")
        if eng.check_win() != board.won(state):
            return state, f"after {d!r}: win disagreement"
        return state, None

    def report(self) -> str:
        return "[" + " ".join(f"{k} {v}" for k, v in self.cover.items()) + "]"


def _fuzz(episodes: int = 40, steps: int = 60, seed: int = 0) -> int:
    """Assert `_Board` reproduces the interpreter, cell for cell.

    This is what licenses planning natively at all: the model may only stand in
    for the interpreter for as long as it agrees with it. Two suites:

    * **Every level**, from its start and from every PREFIX of its own
      solution, then random presses. The prefixes are the point -- pressing a
      button takes a deliberate sequence of pushes, so purely random play on
      the big boards almost never opens a door and would "pass" while testing
      none of the door rules.
    * **`_SCENARIOS`**, hand-built boards for the turns even that does not
      reach: chain pushes, the three ways a push is refused, and a door
      closing on a stone.

    The coverage counts are printed for the same reason the scenarios exist:
    a run whose ``chain`` or ``crush`` column is zero has not tested those
    rules, whatever its verdict says. (This is how the "one button of two is
    not enough" semantics were found -- the level suite alone reported OK.)
    """
    import random as _random

    game = LimboSolver().make_game(0)
    g = game._game
    expert = LimboExpert(game, node_cap=LimboSolver.node_cap)
    rng = _random.Random(seed)
    bad = 0

    ref = _Referee(game)
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board.from_engine(game._engine, g)
        plan = expert.plan(game._engine, level) or []
        starts = list(range(len(plan) + 1))   # the start, and every prefix
        mismatches = 0
        for ep in range(episodes):
            game.set_level(level)
            eng = game._engine
            state = board.start
            presses = (list(plan[:starts[ep % len(starts)]])
                       + [rng.choice(_DIRS) for _ in range(steps)])
            for d in presses:
                state, complaint = ref.press(eng, board, state, d)
                if complaint:
                    mismatches += 1
                    print(f"  level {level}: MISMATCH {complaint}")
                    break
                if state[0] < 0 or board.won(state):
                    break             # crushed or finished: nothing more to do
        bad += mismatches
        print(f"level {level}: {episodes} runs from {len(starts)} start states"
              f" -- {'OK' if not mismatches else str(mismatches) + ' MISMATCH'}")
    print(f"   levels coverage {ref.report()}")

    scen = _Referee(game)
    for name, rows, presses in _SCENARIOS:
        board, state, eng = _load_scenario(game, rows)
        complaint = None
        for d in presses.split():
            state, complaint = scen.press(eng, board, state, d)
            if complaint:
                break
        bad += complaint is not None
        print(f"scenario {name:26s} -- "
              f"{'OK' if not complaint else 'MISMATCH ' + complaint}")
    print(f"   scenario coverage {scen.report()}")
    missing = [k for k, v in scen.cover.items() if not v and not ref.cover[k]]
    if missing:
        print(f"WARNING: never exercised: {', '.join(missing)}")
    print("model matches the interpreter" if not bad
          else f"FUZZ FAILED: {bad} mismatching runs")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Composition, not object: the bugs this exists for were all in the stack --
    an open pink door that quantized onto the Background, buttons whose 5x5
    sprites flattened into the solid block their doors render as, and a Goal
    that the Player's identical 3x3 body covered completely, so the winning
    frame was pixel-identical to the player standing on bare floor.

    Whole 64x64 frames of uniform boards are compared rather than one cell out
    of a mixed board: `_render_frame` centre-pads a non-square board, so
    indexing a cell by ``cell_px * r`` silently reads the wrong pixels on the
    wide levels. Two uniform boards render identically iff their cells do."""
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    game = LimboSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {
        "floor": (), "wall": ("wall",), "grate": ("grate",), "goal": ("goal",),
        "pinkdoor": ("pinkdoor",), "orangedoor": ("orangedoor",),
        "pinkx": ("pinkx",), "orangex": ("orangex",),
        "pinkbutton": ("pinkbutton",), "orangebutton": ("orangebutton",),
        "stone": ("stone",), "player": ("player",),
        "player_on_goal": ("goal", "player"),
        "player_on_grate": ("grate", "player"),
        "player_on_pinkbutton": ("pinkbutton", "player"),
        "player_on_orangebutton": ("orangebutton", "player"),
        "player_on_pinkx": ("pinkx", "player"),
        "player_on_orangex": ("orangex", "player"),
        "stone_on_goal": ("goal", "stone"),
        "stone_on_pinkbutton": ("pinkbutton", "stone"),
        "stone_on_orangebutton": ("orangebutton", "stone"),
        "stone_on_pinkx": ("pinkx", "stone"),
        "stone_on_orangex": ("orangex", "stone"),
    }

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    sys.exit(LimboSolver.main())
