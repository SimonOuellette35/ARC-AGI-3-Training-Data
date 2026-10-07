"""Generate Phase-1 training data for the PuzzleScript game ps:sticky_cubes
("Sticky Cubes", PuzzleScriptGamer).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, the macro A* that searches them, the optimal-action labelling, the
render fixes the game needed, and the checks behind all of it.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_sticky_cubes",
      "levels": [
        {"level_id": 0,
         "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
         "actions":      [a_0, a_1, ..., a_{T-1}]},
        ...
      ]
    }

``actions[0]`` is the RESET that produced ``obs[0]``; ``actions[i]`` for i>=1 is
the action that took the agent from ``obs[i-1]`` to ``obs[i]``. The recorded
index is the *screen* action (post rotation remap), so replaying the recorded
actions reproduces the recorded frames exactly. Every expert step carries the
full set of equally-optimal presses.

THE GAME
========
Eleven levels, ``All Target on Cube``: every black goal square must end with an
orange cube standing on it. One player, four arrow keys and X. What makes it
not a sokoban is that **the cubes are glued to each other**:

    [MOVING Cube|Cube] -> [MOVING Cube|MOVING Cube]

so a push propagates through the whole orthogonally-connected clump, and

    [> Cube|Wall] -> [STATIONARY Cube|Wall]
    [Moving Cube|STATIONARY Cube] -> [STATIONARY Cube|STATIONARY Cube]

freeze it right back if ANY member is blocked. A clump therefore moves as one
rigid body or not at all, which is the whole puzzle: a clump of five crosses a
room at one press per CELL rather than per cube, and one member of it wedged
against a wall costs you the other four as well.

THE FOUR PLAYERS ARE FOUR VERBS
-------------------------------
The player is one of four objects and the terrain rewrites which:

    LATE [Player Fire]   -> [RPlayer Fire]      red    -- X burns
    LATE [Player Water]  -> [BPlayer Water]     blue   -- plain push
    LATE [Player Magnet] -> [YPlayer Magnet]    yellow -- pulls
    LATE [Player Earth]  -> [GPlayer Earth]     green  -- launches

* **Blue** is the default and only pushes: ``[> Player|Cube] -> [> Player|> Cube]``.
* **Red** adds ``[ACTION RPlayer|Cube] -> [ACTION RPlayer| ]`` -- pressing X
  deletes EVERY cube orthogonally adjacent to you at once, in all four
  directions. It is the only way to delete a cube outright (an earth plate also
  takes one out of play, by rooting it into a wall), and the only press in the
  game that is not a direction.
* **Yellow** adds ``[< YPlayer|Cube] -> [< YPlayer|< Cube]``: a cube beside you
  follows when you walk AWAY from it. So while yellow, *walking is not inert* --
  half the steps a blue player would take as a stroll are pulls. The search and
  the tie labelling both have to know that (see `_StickyExpert._walk`).

  And a yellow press pulls even when the player CANNOT MOVE. Pressing into a
  wall still hands the cube behind you an away-force; that cube is stopped by
  the player's own body, but ``[> Cube|Wall]`` does not fire, because the
  Player is a `Block` and not a `Wall` -- so the rest of its clump shears past
  on either side. It is the shear below, performed with your own back instead
  of a magnet, and it is a legal move that changes the board from a cell the
  player never leaves.
* **Green** turns a push into a LAUNCH: the cube is rewritten to CubeU/D/L/R,
  which is a cube with one cell of motion still owed to it. That cell is spent
  on the NEXT press -- ``UP [CubeU|NO Wall] -> [> CubeU|NO Wall]`` runs before
  ``[> Player][CubeM] -> [> Player][CubeN]`` resets it -- so a green push is
  worth two cells instead of one, and the second is paid for by whatever you
  press afterwards, up to and including the press that wins.

  The game's own message calls this pushing cubes "across the room", and in
  stock PuzzleScript the rule's ``again`` would indeed chain the slide to the
  far wall. This interpreter does not chain it: one extra cell is what it
  actually buys. That is MEASURED (``--selfcheck`` presses a launched cube four
  times in an open field and it moves once), and it is why a CubeU is a
  genuinely different state from a CubeN rather than a cosmetic leftover -- see
  `_StickyExpert._key` and `_apply_macro`, both of which had to stop treating
  them as the same board.

THE PLATES ARE THE GEOMETRY
---------------------------
Six things live on the terrain layer, and each of them is a different kind of
wall depending on who is asking:

    Target  goal square; blocks nothing
    Fire    LATE [Cube Fire]  -> [Fire]        a cube that lands here is deleted
    Earth   LATE [Cube Earth] -> [CubeG Earth] a cube that lands here becomes a
                                               WALL -- and CubeG is not a Cube,
                                               so it can never satisfy a Target
                                               again. Rooting is irreversible.
    Magnet  [> Cube|Magnet] -> [STATIONARY Cube|Magnet]
                                               a cube may not enter; the clump
                                               SHEARS around it (below)
    Water   nothing, except that it makes you blue again
    Crate   [> Player|Crate] -> [STATIONARY Player|Crate]
                                               the "cube plate": cubes pass, the
                                               player may not

So for the PLAYER the impassable set is {wall, cube, crate-plate}, and for a
CUBE it is {wall, magnet, fire, earth} -- fire and earth because a cube that
arrives there stops being a usable cube. Both maps are in `heuristic`.

THE SHEAR (why "magnet plates dismantle cubes")
-----------------------------------------------
Two rules stop a moving clump, and they are not the same rule:

    [Moving Cube|STATIONARY Cube]  -> freeze     (any direction of travel)
    [> Cube|STATIONARY Cube]       -> freeze     (travelling TOWARDS it)

The first has already run by the time a magnet anchors anything, so the freeze
that a magnet causes propagates only through the second -- i.e. only along the
line of travel. A 2x2 clump with its bottom row against a magnet, shoved right,
therefore leaves that row behind and walks the top one off (``p`` the player,
``m`` the magnet plate; ``--selfcheck`` runs exactly this board):

    .cc.             ..cc
    pccm    --->     pccm

That is what the game's own message calls "magnet plates dismantle cubes". It is
one of three ways to break a clump apart: a fire plate deletes a member, a
magnet anchors one, and -- see the yellow bullet above -- a blocked yellow
player anchors one with its own body.

EXPERT
======
`_StickyExpert` is a MACRO A* over the real interpreter, in the shape of
`PSPushExpert` but with this game's three complications built in.

*Why not the primitive search.* `PSExpert`'s five-way A* solves 3 of the 11
levels in ~30s each and does not finish the other 8 in 20 minutes at a 400k node
cap: the plans are 30-50 presses of which only ~10 do anything, so the search
spends its entire budget re-deriving walks.

*The macros.* A press only matters when it pushes, pulls or burns, so a macro is
``walk to a cell, then press``:

    push   walk to a cell orthogonally beside a cube, press towards it
    pull   (yellow only) walk beside a cube, press AWAY from it
    burn   (red only) walk beside a cube, press X

*The walk graph is over (cell, colour), not cell.* Two things force that. A walk
across a plate REWRITES the player, so "where can I get to" is not a set of
cells but a set of cell-and-colour pairs -- the same cell reached the long way
round may be reached as a different verb. And while yellow, a step that moves
directly away from an adjacent cube is a PULL, not a walk, so it is struck out
of the graph as an edge (and offered as a macro instead). The graph is therefore
DIRECTED, which is why states are deduped by the whole reachable set rather than
by `PSPushExpert`'s "smallest cell in the region" shorthand: with one-way edges
that shorthand can merge two states with different futures.

*Walks are applied by SEATING, not by stepping.* On a board with no cube in
flight a bare move fires no rule at all, so `_apply_macro` writes the player
straight onto the walk's end cell and steps the interpreter once, for the press
that matters. That is the difference between ~15 interpreter ticks per node and
1, and it is what makes the search finish. On a board that DOES hold a
CubeU/D/L/R the shortcut is off and every primitive is stepped, because there
the first press of the walk also spends the launched cube's last cell -- which
can block the rest of the walk, and can win. ``--selfcheck`` replays a few
thousand macros both ways and asserts the grids are identical.

*The heuristic* matches every target to a distinct live cube and sums the
shortest cube-route between them, over the cube-impassable map above, plus the
player's walk to the nearest loose one. It is NOT admissible -- a clump of five
moves five cubes for one press, so summing per-cube routes over-charges exactly
the levels this game is about -- so plans are winning and engine-verified but
not certified shortest. The admissible variant (the MAX instead of the sum) is
worse in practice, not better: it is flat across the middle of every level here,
which leaves A* nothing to steer with.

The tables double as the deadlock test, for free: `DEAD_COST` for a target no
live cube can route to at all, and for a board with fewer live cubes than
targets, which is reachable and permanent (fire DELETES a cube, earth ROOTS one
into a wall that no longer counts).

*And it is not enough on its own.* On five of the eleven levels the cubes start
beside their targets and the estimate is 4 or 5 while the answer is forty
presses of moving the PLAYER -- onto a fire plate to burn the cube pinning a
clump, onto a magnet to shear one. Over that stretch ``f`` is essentially ``g``
and A* is uniform-cost search at a hopeless depth, so `_search` gives A* a small
node budget and then hands the level to `_beam`: the same macros, breadth-first,
width-capped, with the heuristic demoted to a tie-break. Beam plans wander, so
`_shorten` cuts verified blocks out of them afterwards. Which levels come out of
which search, and how long each takes, is what ``--plans`` reports.

THE ELEVEN LEVELS, AND WHICH SEARCH GETS THEM
---------------------------------------------
Measured cold (one process per level, no cached plan). The last column is what
the plan actually DOES, read back off the board press by press rather than
guessed at -- which colours the player takes, how many X presses land, and how
many cubes stop being cubes:

    L0  10x14   9 cubes /  5 targets  A*    33p    9s   blue only
    L2  10x14  16 cubes /  2 targets  beam  54p  212s   blue>red>yellow, 1 burn
    L3  10x14  14 cubes /  4 targets  beam  48p  305s   blue>red, 2 burns (-5 cubes)
    L4  10x14   4 cubes /  3 targets  A*    34p    6s   blue>yellow, pure shear
    L7  10x16   3 cubes /  3 targets  A*    33p    2s   blue>green, launches
    L8  10x16   7 cubes /  5 targets  A*    46p   15s   blue>green, 2 cubes ROOTED
    L1 L5 L6 L9 L10                   no plan -- see `skip_levels`

248 presses in all, 23 of them carrying a second equally-right answer. All four
colours are exercised and so is every irreversible thing the game can do: L3's
two X presses delete five cubes between them (X takes all four neighbours at
once), and L8 deliberately roots two cubes into walls on the way to filling five
targets with the other five. Every plan is engine-verified and fits inside the
adapter's 200-press per-level budget with room for the exploration prefix.

The five that do not yield are the ones whose answer is a long, unrewarded
detour, and the honest summary is arithmetic: at ~4ms per interpreter tick the
search sees ~200k boards per level, and these need more than that. ``--deep``
raises the budget 3.5x and drops the skip list; at that budget the beam has been
seen to solve L9 and to fail on L5, so it is worth one run (anything it finds is
cached) but it is not the answer either. The real fix is a native model of the
mechanic; `skip_levels` says why, and what it would cost.

OPTIMAL-ACTION SETS
-------------------
Every recorded expert step carries one. A push/pull/burn is labelled with
itself: which cube to move where is the puzzle, and a sibling push is a
different plan rather than a reordering of this one. A WALK step is labelled
with every direction that keeps it on a shortest route to the (cell, colour)
the next press is taken from -- computed by a reverse BFS in the same directed
graph the macros were built from, so the alternatives it offers respect both the
plate rewrites and the yellow no-pull rule. `PSPushExpert.annotate_walks` is the
shared version of this and is NOT usable here: it reasons over cells only, so it
would offer a detour that arrives the right place the wrong colour.

RENDERING (fixed in data/puzzlescript_games/Sticky_Cubes.txt)
=============================================================
Both board shapes (10x14 and 10x16) render at ``cell_px = 4``, where
`_render_cell_sprite` samples sprite rows/cols {0, 1, 3, 4} -- the middle is
never read. As shipped, the game was unreadable in three ways, all of them the
recurring "ground tile under an opaque body" shape:

* **The WIN was invisible.** Cube shipped with no sprite at all, i.e. a solid
  5x5 orange block, and Target painted its mark on rows 3-4. A cube standing on
  a target was therefore PIXEL-IDENTICAL to a cube standing on bare floor -- the
  one fact the win condition is made of could not be read off the frame.
* Same for ``cube on a cube-plate``, and for the player standing on ANY plate:
  the player sprite is transparent only at its two top corners.
* Magnet and Crate were both ``ORANGE``, which is ARC 12, which is the cubes'
  own colour. (CubeG, a rooted cube, is ARC 12 too, but it carries a green core
  and stayed distinct.)

Fixed by giving the cubes transparent CORNERS and every plate a mark in those
corners, in its own colour:

    Cube    .000. / 00000 / 00000 / 00000 / .000.
    plate   0...0 / ..... / ..... / ..... / 0...0   + its old body art

At cell_px 4 the four sprite corners land on output pixels (0,0), (0,3), (3,0)
and (3,3), the cube is transparent at exactly those, and the player is
transparent at the top two -- so a plate reads through both bodies. Magnet
became YELLOW and Crate PURPLE, which also makes every plate the colour of the
player it produces (fire/red, water/blue, magnet/yellow, earth/green) instead of
three of them being the cubes' orange. CubeG (a rooted cube, which is a WALL and
can never satisfy a target again) keeps its green core so it stays distinct from
a live cube. ``--audit`` is the executable form of all of that.

USAGE
-----
    python solvers/generate_sticky_cubes_training.py --plans       # the plans
    python solvers/generate_sticky_cubes_training.py --selfcheck   # mechanics
    python solvers/generate_sticky_cubes_training.py --audit       # rendering
    python solvers/generate_sticky_cubes_training.py --symmetry    # rotation
    python solvers/generate_sticky_cubes_training.py --episodes 1600 \
        --out data/training_multi_level/ps:sticky_cubes

``--deep`` may be added to any of those: it widens the beam and drops the skip
list, which is worth ONE run (``--plans --deep``) before sharded generation,
because whatever it finds lands in `_StickyExpert.plan_cache_path` and every
later run reads it from there.

The searches are seed-independent and cached, so the first invocation pays for
them (~10 minutes for the six shipped levels, one process) and every seed after
that is a re-render.
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                        # noqa: E402
from solvers.common.ps_astar import (                               # noqa: E402
    _DELTA, DEAD_COST, PSAStarSolver, PSExpert, Plan, restore, screen_action,
    snapshot,
)

GAME_NAME = "Sticky_Cubes"

#: Engine directions the search branches on: the four moves plus X. X is only
#: ever useful to a RED player (it is the burn), but it is in the list because
#: `record_level`'s epsilon detour and the exploration prefix both read it.
DIRECTIONS = ["up", "down", "left", "right", "action"]


# ---------------------------------------------------------------------------
# The board, read once per node
# ---------------------------------------------------------------------------

class _Board:
    """One scan of the engine grid, in the terms the expert reasons in.

    Read ONCE per search node and threaded through `_walk`, `_macros` and
    `_h` rather than re-derived by each: the scan is ~150 cells of Python set
    arithmetic and the three of them together run on every expansion, so
    re-scanning was a third of the search's wall clock.
    """

    __slots__ = ("h", "w", "cubes", "cubem", "walls", "crates", "targets",
                 "plates", "cube_block", "player", "ptype")

    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.cubes: set = set()        # live (pushable) cubes
        self.cubem: list = []          # the CubeU/D/L/R subset -- see `_seat`
        self.walls: set = set()        # BWall + CubeG (a rooted cube IS a wall)
        self.crates: set = set()       # cube plates: the player may not enter
        self.targets: set = set()
        self.plates: dict = {}         # cell -> the player object it rewrites to
        self.cube_block: set = set()   # cells a cube can never usefully occupy
        self.player = None
        self.ptype = None


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class _StickyExpert(PSExpert):
    """Macro A* over the real interpreter -- see the module docstring."""

    directions = DIRECTIONS

    #: `_key` names only the cubes and the player, which is canonical WITHIN a
    #: level (walls, plates and targets are static per level) but not across
    #: them.
    scope_by_level = True

    #: The searches are the entire cost of generation and they are
    #: seed-independent, so they are kept on disk: without this every shard
    #: `parallelize_generator` starts would re-derive all eleven.
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "sticky_cubes_plans.json")

    # -- setup ---------------------------------------------------------------
    def setup(self) -> None:
        g = self.g
        idx = g.obj_name_to_idx
        self.cube_ids = frozenset(g.resolve_object_name("cube"))
        self.cubem_ids = frozenset(g.resolve_object_name("cubem"))
        self.cuben = idx["cuben"]
        self.cubeg = idx["cubeg"]
        self.wall_ids = frozenset(g.resolve_object_name("wall"))
        self.player_ids = frozenset(self.game._engine._player_indices)
        self.t_target = idx["target"]
        self.t_crate = idx["crate"]
        #: plate object -> the player object standing on it becomes.
        self.transform = {idx["fire"]: idx["rplayer"], idx["water"]: idx["bplayer"],
                          idx["magnet"]: idx["yplayer"], idx["earth"]: idx["gplayer"]}
        self.yplayer = idx["yplayer"]
        self.rplayer = idx["rplayer"]
        #: Terrain no cube may ever come to rest on: a magnet refuses it, a fire
        #: plate deletes it and an earth plate roots it into a wall.
        self.cube_block_ids = frozenset(
            {idx["magnet"], idx["fire"], idx["earth"]}) | self.wall_ids
        self._tables: dict = {}        # (blocked, targets) -> per-target BFS

    # -- board scan ----------------------------------------------------------
    def _read(self, eng) -> _Board:
        b = _Board(eng.height, eng.width)
        cube_ids, wall_ids, players = self.cube_ids, self.wall_ids, self.player_ids
        cubem_ids, transform, blockers = (self.cubem_ids, self.transform,
                                          self.cube_block_ids)
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & cube_ids:
                    b.cubes.add((r, c))
                    if cell & cubem_ids:
                        b.cubem.append((r, c))
                if cell & wall_ids:
                    b.walls.add((r, c))
                if cell & blockers:
                    b.cube_block.add((r, c))
                if self.t_crate in cell:
                    b.crates.add((r, c))
                if self.t_target in cell:
                    b.targets.add((r, c))
                for plate, becomes in transform.items():
                    if plate in cell:
                        b.plates[(r, c)] = becomes
                        break
                here = cell & players
                if here:
                    # Every shipped level has exactly one player. If an edit ever
                    # adds a second, this reads the last in scan order (and the
                    # lowest-numbered object in its cell) -- deterministic, which
                    # is all the search needs to stay reproducible.
                    b.player = (r, c)
                    b.ptype = min(here)
        return b

    # -- the (cell, colour) walk graph ---------------------------------------
    def _walk(self, b: _Board) -> dict:
        """Shortest inert walks from the player, as a parent tree over
        ``(cell, player_object)`` nodes.

        Three things make this graph rather than a plain flood fill, and all
        three are load-bearing:

        * a cell holding a **cube plate** is closed to the player outright;
        * stepping onto a fire / water / magnet / earth plate REWRITES which
          player object you are, so the node carries the colour;
        * while YELLOW, a step that moves directly away from an adjacent cube
          drags it -- that is a pull, not a walk, so the edge is deleted here
          and the same press is offered as a macro by `_macros` instead.

        The last one makes the graph DIRECTED: you can step towards a cube but
        not back off it.
        """
        h, w, cubes, walls, crates = b.h, b.w, b.cubes, b.walls, b.crates
        plates, yellow = b.plates, self.yplayer
        start = (b.player, b.ptype)
        parent: dict = {start: None}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            (r, c), colour = node
            for d, (dr, dc) in _DELTA.items():
                nxt = (r + dr, c + dc)
                if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                    continue
                if nxt in walls or nxt in cubes or nxt in crates:
                    continue
                if colour == yellow and (r - dr, c - dc) in cubes:
                    continue                       # this press is a PULL
                key = (nxt, plates.get(nxt, colour))
                if key not in parent:
                    parent[key] = (node, d)
                    queue.append(key)
        return parent

    @staticmethod
    def _route(parent: dict, node) -> list:
        out = []
        while parent[node] is not None:
            node, d = parent[node]
            out.append(d)
        out.reverse()
        return out

    def _macros(self, b: _Board, parent: dict) -> list:
        """Every ``walk + one press that does something`` from this board.

        Each macro is ``(presses, end_cell, end_colour)``: the flat primitive
        list the plan is made of, plus where the walk leaves the player, which
        `_apply_macro` uses to skip replaying it.
        """
        cubes, yellow, red = b.cubes, self.yplayer, self.rplayer
        out = []
        for node in parent:
            (r, c), colour = node
            walk = None
            for d, (dr, dc) in _DELTA.items():
                if (r + dr, c + dc) in cubes:                    # push / launch
                    pass
                elif colour == yellow and (r - dr, c - dc) in cubes:   # pull
                    # Deliberately NOT pruned to the case where the player can
                    # actually move. A yellow player pressing into a WALL still
                    # pulls: the cube directly behind is stopped by the player's
                    # own body, but the Player is not a `Wall`, so
                    # ``[> Cube|Wall]`` never fires and the rest of that cube's
                    # clump shears past on both sides. Pruning these away is the
                    # one thing ``--selfcheck``'s macro-completeness pass has
                    # ever caught in this file, and it caught it on a board 250
                    # random presses deep.
                    pass
                else:
                    continue
                if walk is None:
                    walk = self._route(parent, node)
                out.append((walk + [d], (r, c), colour))
            if colour == red and any((r + dr, c + dc) in cubes
                                     for dr, dc in _DELTA.values()):
                if walk is None:
                    walk = self._route(parent, node)
                out.append((walk + ["action"], (r, c), colour))
        if b.cubem:
            # A cube is still owed a cell of travel, and ANY directional press
            # collects it -- so on these boards a bare step is a move that
            # changes the world, and it can be the press that wins. Without
            # these four the search could only ever cash a launch in as a side
            # effect of walking somewhere it wanted to go anyway, and on a board
            # where the launch IS the answer there may be nowhere it wants to go.
            out.extend(([d], b.player, b.ptype) for d in _DELTA)
        return out

    # -- applying a macro ----------------------------------------------------
    def _seat(self, eng, b: _Board, end, colour) -> None:
        """Put the player on ``end`` as ``colour`` without stepping the walk.

        Sound ONLY on a board with no cube in flight, which is what
        `_apply_macro` checks before calling this. There, no rule in the game
        matches a player that merely moved: the plate rewrites are already in
        ``colour`` (the walk graph applied them as it went), the push/pull/burn
        rules all need a cube beside the player and a walk never steps beside
        one it then leaves, and no walk can win, because the win names Targets
        and Cubes and a walk moves neither.

        Checked by ``--selfcheck``, which replays every macro of a few thousand
        random boards both ways and compares the grids cell for cell.
        """
        grid = eng.grid
        pr, pc = b.player
        grid[pr][pc] = grid[pr][pc] - self.player_ids
        er, ec = end
        grid[er][ec] = grid[er][ec] | {colour}
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    def _apply_macro(self, eng, b: _Board, macro) -> int:
        """Run one macro; return the index of the primitive that WON, else -1.

        The seating shortcut is switched off whenever a CubeU/D/L/R is on the
        board: there the walk is not inert (its first press spends the launched
        cube's last cell), the rest of the walk may be blocked or freed by where
        that cube lands, and the walk can END the level. So those boards are
        stepped primitive by primitive with the win tested after each -- which is
        also why the win test is a loop rather than a single check at the end:
        ``check_win`` reads a per-step flag, so a macro that wins partway through
        and keeps stepping silently un-wins itself.
        """
        presses, end, colour = macro
        if b.cubem:
            for i, direction in enumerate(presses):
                eng.step(direction)
                if eng.check_win():
                    return i
            return -1
        if len(presses) > 1:
            self._seat(eng, b, end, colour)
        eng.step(presses[-1])
        return len(presses) - 1 if eng.check_win() else -1

    # -- state key -----------------------------------------------------------
    def _key(self, eng) -> frozenset:
        """Cubes (live and rooted) and the player, by cell AND by object.

        The five live cube objects are kept APART, which is not obvious and was
        wrong in the first draft of this expert. A CubeR is not a CubeN that has
        stopped: it is a cube still owed one cell of travel, and the next
        directional press collects it (see the launch note in the module
        docstring). Collapsing them onto one token merges two boards whose
        futures differ by a cube's worth of position, and the plans that come
        out do not replay. ``--selfcheck`` is what caught it and what keeps it
        caught: it steps a launched cube and a settled one through all five
        presses and requires the boards to disagree.

        Rooted cubes (CubeG) get their own token because they are a different
        thing entirely: a wall that no longer counts towards the win.
        """
        cube_ids, players = self.cube_ids, self.player_ids
        out = []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                for o in (cell & cube_ids):
                    out.append((r, c, o))
                if self.cubeg in cell:
                    out.append((r, c, -2))
                here = cell & players
                if here:
                    out.append((r, c, min(here)))
        return frozenset(out)

    def _search_key(self, eng, parent: dict) -> tuple:
        """The key the SEARCH dedups on: the cubes, plus the whole set of
        ``(cell, colour)`` the player can walk to -- and deliberately NOT where
        the player is standing.

        Two boards with the same cubes and the same reachable set admit exactly
        the same macros (a macro is a walk to a node of that set plus a press),
        hence exactly the same successors, so merging them cannot lose a
        solution -- it can only mis-cost one by the length of a walk inside the
        region. That is `PSPushExpert`'s standard sokoban canonicalisation, and
        it is worth restating why it is the SET here rather than that class's
        "smallest cell of the region" shorthand: this walk graph is DIRECTED
        (while yellow you can step towards a cube but not back off it, and a
        plate rewrite is one-way), so two nodes can share a region-representative
        without sharing a future.

        `_key` -- which carries the player's cell and colour -- is the exact
        state, and stays exact because it is what the plan memo and the
        ``plan_cache_path`` signature are built from. Using it HERE was the
        first draft's mistake and it cost most of the search: every cell the
        player could have stopped on became its own node, which is precisely the
        dedup this class exists to get.
        """
        cube_ids = self.cube_ids
        cubes = frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & cube_ids)
            if True)
        rooted = frozenset(
            (r, c)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            if self.cubeg in cell)
        return (cubes, rooted, frozenset(parent))

    # -- heuristic -----------------------------------------------------------
    def _cube_tables(self, b: _Board) -> dict:
        """``{target: {cell: steps for a cube to travel from cell to target}}``.

        Plain BFS over the cells a cube may occupy -- not walls, not magnets
        (which refuse it), not fire (which deletes it), not earth (which roots
        it into a wall). Ignores the other cubes and the player, which is the
        standard relaxation, and ignores stickiness, which is what makes the sum
        of these an over-estimate rather than a bound. Rebuilt whenever the
        blocked set changes, i.e. when a cube is rooted.
        """
        sig = (frozenset(b.cube_block), frozenset(b.targets))
        cached = self._tables.get(sig)
        if cached is not None:
            return cached
        h, w, blocked = b.h, b.w, b.cube_block
        tables = {}
        for target in b.targets:
            dist = {target: 0}
            queue = deque([target])
            while queue:
                r, c = queue.popleft()
                step = dist[(r, c)] + 1
                for dr, dc in _DELTA.values():
                    nxt = (r + dr, c + dc)
                    if (0 <= nxt[0] < h and 0 <= nxt[1] < w
                            and nxt not in blocked and nxt not in dist):
                        dist[nxt] = step
                        queue.append(nxt)
            tables[target] = dist
        self._tables[sig] = tables
        return tables

    def _h(self, b: _Board) -> int:
        """Cube routes, greedily matched, in units of primitive presses.

        Every target is matched to a distinct live cube -- not just the
        UNCOVERED ones to the LOOSE ones, which is the obvious version and is
        subtly wrong: it calls a board dead when the only cube that can route to
        an empty target happens to be parked on a full one, and shoving that
        cube along and refilling behind it is an ordinary move. Matching over
        every (target, cube) pair costs nothing extra -- a covered target draws
        its own cube at distance 0 and the greedy pass takes it first -- and it
        makes `DEAD_COST` mean what it says.

        Two ways a board is dead, both one-way and both real: fewer live cubes
        than targets (fire DELETES a cube and earth ROOTS it into a wall that no
        longer counts), and a target no live cube can route to at all. The
        second is free: the tables are a BFS over the cells a cube may occupy,
        so a target walled off from every cube simply has none of them in its
        table.
        """
        targets = b.targets
        if not targets:
            return 0
        cubes = list(b.cubes)
        if len(cubes) < len(targets):
            return DEAD_COST
        bare = [t for t in targets if t not in b.cubes]
        if not bare:
            return 0
        tables = self._cube_tables(b)
        total = 0
        taken: set = set()
        # Covered targets first, so each takes the cube already standing on it
        # (distance 0) instead of stealing one a bare target needs.
        for target in sorted(targets, key=lambda t: t not in b.cubes):
            dist = tables[target]
            best = best_i = None
            for i, cell in enumerate(cubes):
                if i in taken:
                    continue
                d = dist.get(cell)
                if d is not None and (best is None or d < best):
                    best, best_i = d, i
                    if d == 0:
                        break
            if best is None:
                return DEAD_COST      # no live cube can ever route to this target
            taken.add(best_i)
            total += best
        loose = [c for c in cubes if c not in targets]
        if b.player is not None and loose:
            # The walk to the first push, as the crow flies: a small term beside
            # the cube routes, and a real BFS per node is not worth it.
            total += max(0, min(abs(b.player[0] - r) + abs(b.player[1] - c)
                                for r, c in loose) - 1)
        return total

    def heuristic(self, eng) -> int:
        """`_h` off a fresh scan. The search itself never calls this -- it
        already holds the `_Board` -- but `PSExpert.optimal_sets` and any
        caller poking at the expert do."""
        return self._h(self._read(eng))

    # -- the search ----------------------------------------------------------
    def _astar(self, eng) -> list | None:
        """Weighted A* over macros. ``g`` counts primitive presses, so the cost
        being minimised is what the agent actually pays.

        States are deduped by `_search_key` -- the cubes plus the player's whole
        reachable ``(cell, colour)`` set, with its exact cell deliberately left
        out. See that method for why the set and not a representative of it.
        """
        if eng.check_win():
            return []
        weight = self.weight
        board = self._read(eng)
        parent = self._walk(board)
        start = (snapshot(eng), [], board, self._macros(board, parent))
        pq = [(weight * self._h(board), 0, 0, start)]
        best_g = {self._search_key(eng, parent): 0}
        counter = nodes = 0
        while pq:
            _f, g, _c, (snap, path, board, macros) = heapq.heappop(pq)
            for macro in macros:
                restore(eng, snap)
                won = self._apply_macro(eng, board, macro)
                nodes += 1
                if won >= 0:
                    return path + macro[0][:won + 1]
                if eng.grid == snap:
                    continue                    # a press the engine refused
                nboard = self._read(eng)
                if nboard.player is None:
                    continue                    # the player was consumed
                est = self._h(nboard)
                if est >= DEAD_COST:
                    continue                    # unwinnable from here
                nparent = self._walk(nboard)
                key = self._search_key(eng, nparent)
                ng = g + len(macro[0])
                if best_g.get(key, 1 << 30) <= ng:
                    continue
                best_g[key] = ng
                counter += 1
                heapq.heappush(pq, (ng + weight * est, ng, counter,
                                    (snapshot(eng), path + macro[0], nboard,
                                     self._macros(nboard, nparent))))
                if nodes >= min(self.node_cap, self.astar_node_cap):
                    return None
        return None

    # -- the breadth-first fallback -----------------------------------------
    #: Macro budget for `_astar` before `_search` gives up on it and runs the
    #: beam. Deliberately far below ``node_cap``: the four levels A* wins it
    #: wins inside ~15s (a few thousand macros), and on the other seven its
    #: estimate is flat (see `_beam`) so the next 200k nodes are twenty minutes
    #: of uniform-cost search that ends in None anyway. Failing fast is what
    #: makes a cold plan cache minutes rather than hours.
    astar_node_cap: int = 40_000

    #: States kept per macro layer in `_beam`, ranked by `_h`.
    beam_width: int = 250
    #: Give up after this many macro layers.
    beam_depth: int = 25
    #: Macro budget for the whole beam (``node_cap`` stays A*'s). One macro is
    #: ~4ms, so this is the ~10-minute-per-level dial. It is a ONE-TIME cost:
    #: `plan_cache_path` keeps the answer -- including "no plan" -- on disk, and
    #: the engine state after a reset is seed-independent, so no later seed and
    #: no later shard pays it again.
    beam_node_cap: int = 200_000

    #: Interpreter ticks `_shorten` may spend tidying one beam plan.
    shorten_tick_budget: int = 30_000

    #: A plan longer than this is discarded (the level is reported unsolvable)
    #: rather than shipped. `record_level` stops at ``max_steps`` and the adapter
    #: stops at its own 200-press per-level budget, so a plan that does not fit
    #: would tape a level that never reaches WIN -- and because a seed counts
    #: only when EVERY solvable level wins, one such level would silently zero
    #: the whole corpus rather than cost one level.
    max_plan_presses: int = 140

    def _beam(self, eng) -> list | None:
        """The same macros, explored breadth-first with a width cap.

        WHY A* RUNS OUT. Both searches pay the same thing per node -- one
        interpreter tick plus a walk BFS -- and A* spends that budget where the
        heuristic points. On seven of the eleven levels it points nowhere: the
        cubes already start beside their targets and the estimate is 4 or 5,
        while the answer is forty presses of getting the PLAYER somewhere (onto
        a fire plate to burn the cube that pins the clump, onto a magnet to
        shear it). Over that stretch the estimate is flat and small, so ``f`` is
        essentially ``g`` and A* degenerates to uniform-cost search at a depth
        where that is hopeless. The beam spends the budget on breadth at every
        depth instead and uses the heuristic only to break ties, which is all a
        flat heuristic is good for. It converts two of the seven (L2 and L3,
        both of which have to walk to a fire plate and BURN something first).

        The trade is length: a beam plan wins and is engine-verified but wanders,
        so `_shorten` cuts it back afterwards.
        """
        if eng.check_win():
            return []
        board = self._read(eng)
        parent = self._walk(board)
        frontier = [(snapshot(eng), [], board, self._macros(board, parent))]
        seen = {self._search_key(eng, parent)}
        nodes = nodes_born = 0
        for _depth in range(self.beam_depth):
            kids = []
            for snap, path, board, macros in frontier:
                for macro in macros:
                    restore(eng, snap)
                    won = self._apply_macro(eng, board, macro)
                    nodes += 1
                    if won >= 0:
                        return path + macro[0][:won + 1]
                    if eng.grid == snap:
                        continue
                    nboard = self._read(eng)
                    if nboard.player is None:
                        continue
                    est = self._h(nboard)
                    if est >= DEAD_COST:
                        continue
                    nparent = self._walk(nboard)
                    key = self._search_key(eng, nparent)
                    if key in seen:
                        continue
                    seen.add(key)
                    # Sorted on (estimate, presses so far, scramble) -- the
                    # tuple also carries a snapshot, which has no ordering.
                    # The scramble matters: over the stretch where the estimate
                    # is flat (which is most of the levels the beam is here
                    # for) the first two fields tie for hundreds of states, and
                    # without a third the width cap keeps whichever ones the
                    # macro enumeration happened to emit first -- i.e. a slice
                    # biased by board scan order, all of it from the same corner
                    # of the frontier. Multiplying the birth counter by a large
                    # odd constant permutes those ties deterministically, so the
                    # survivors are spread across the layer and a re-derived
                    # plan is still byte-identical.
                    nodes_born += 1
                    kids.append((est, len(path) + len(macro[0]),
                                 (nodes_born * 2654435761) & 0xFFFFFFFF,
                                 snapshot(eng), path + macro[0], nboard,
                                 self._macros(nboard, nparent)))
                    if nodes >= self.beam_node_cap:
                        return None
            if not kids:
                return None                  # the reachable space closed
            kids.sort(key=lambda kid: kid[:3])
            frontier = [(snap, path, board, macros)
                        for _e, _g, _s, snap, path, board, macros
                        in kids[:self.beam_width]]
        return None

    def _shorten(self, eng, plan) -> list:
        """Delete the longest block of presses the plan still wins without, and
        repeat until nothing more can go or the replay budget runs out.

        A beam plan wanders: it is the first win breadth-first search reached,
        not the shortest, and it typically carries a whole macro that undoes
        another. Cutting is done against the INTERPRETER -- replay the candidate,
        ask `check_win` -- so a shorter plan is a verified win rather than an
        argument.

        The budget is the point, not a detail. One interpreter tick is ~4ms on
        these boards and it is 95% of everything this file does, so the textbook
        "try every block" pass over a 100-press plan is ~5000 replays of ~100
        ticks each, i.e. half an hour to tidy up one level. Largest-block-first
        with a tick budget gets nearly all of the gain (the cuts worth making
        are whole wasted macros, which are large) and stops on the clock rather
        than on a proof.
        """
        start = snapshot(eng)
        spent = 0

        def wins(seq) -> bool:
            nonlocal spent
            restore(eng, start)
            for i, direction in enumerate(seq):
                eng.step(direction)
                spent += 1
                if eng.check_win():
                    return i == len(seq) - 1
            return False

        changed = True
        while changed and plan and spent < self.shorten_tick_budget:
            changed = False
            for size in range(len(plan) - 1, 0, -1):
                for i in range(len(plan) - size + 1):
                    cut = plan[:i] + plan[i + size:]
                    if cut and wins(cut):
                        plan = cut
                        changed = True
                        break
                    if spent >= self.shorten_tick_budget:
                        break
                if changed or spent >= self.shorten_tick_budget:
                    break
        restore(eng, start)
        return plan

    def _search(self, eng) -> "Plan | None":
        """`_astar`, then the per-step optimal SETS beside it.

        Annotating here rather than lazily during recording is what lets the
        disk cache carry the labels: `plan` runs before the first frame is taped
        and stores the `Plan` whole, so a later process replays labelled steps
        without re-deriving anything. `_astar` leaves the grid dirty, hence the
        bracketing snapshot.
        """
        start = snapshot(eng)
        found = self._astar(eng)
        restore(eng, start)
        if found is None:
            found = self._beam(eng)
            restore(eng, start)
            if found is None:
                return None
            # Rebuild, cut, rebuild. The first pass makes the routes between
            # the presses shortest, which is both a saving in its own right and
            # what makes the cut cheap (every candidate is a full replay, so a
            # shorter plan is a cheaper search). The second repairs the routes
            # the cut just joined up.
            found = self._rebuild(eng, found)
            restore(eng, start)
            found = self._shorten(eng, found)
            restore(eng, start)
            found = self._rebuild(eng, found)
            restore(eng, start)
        if len(found) > self.max_plan_presses:
            return None
        sets = self._annotate(eng, found)
        restore(eng, start)
        return Plan(found, sets)

    # -- optimal-action sets -------------------------------------------------
    def _reverse_walk(self, b: _Board, goal) -> dict:
        """Steps-to-``goal`` for every node of the walk graph, by BFS over the
        graph's REVERSED edges. Separate from `_walk` because that graph is
        directed: a plain BFS from ``goal`` would walk edges that do not exist.
        """
        h, w, cubes, walls, crates = b.h, b.w, b.cubes, b.walls, b.crates
        plates, yellow = b.plates, self.yplayer
        dist = {goal: 0}
        queue = deque([goal])
        while queue:
            node = queue.popleft()
            (r, c), colour = node
            step = dist[node] + 1
            for d, (dr, dc) in _DELTA.items():
                prev = (r - dr, c - dc)          # the cell we would come FROM
                if not (0 <= prev[0] < h and 0 <= prev[1] < w):
                    continue
                if prev in walls or prev in cubes or prev in crates:
                    continue
                if (r, c) in walls or (r, c) in cubes or (r, c) in crates:
                    continue
                for pcolour in self._preimage(colour, (r, c), plates):
                    if pcolour == yellow and (prev[0] - dr, prev[1] - dc) in cubes:
                        continue                 # that press would be a pull
                    key = (prev, pcolour)
                    if key not in dist:
                        dist[key] = step
                        queue.append(key)
        return dist

    def _preimage(self, colour, cell, plates) -> list:
        """Which colours a player could have been BEFORE stepping onto ``cell``
        and coming out ``colour``. A plate rewrites unconditionally, so any of
        the four qualifies; a plain cell passes the colour through."""
        plate = plates.get(cell)
        if plate is None:
            return [colour]
        return list(self.player_ids) if plate == colour else []

    def _classify(self, eng, plan) -> list:
        """Replay ``plan`` and label each press ``push``/``walk``/``stuck``.

        Classification watches the BOARD rather than trusting macro boundaries
        the flattened plan no longer carries: a press that moved a cube (or
        rooted one into a wall) is a ``press``, one that moved only the player
        is a ``walk``, one that moved nothing at all is ``stuck``. Each entry
        also keeps the ``(cell, colour)`` node it was taken from and the whole
        board it was taken on, so a caller can rebuild that walk without a
        second replay. Leaves the engine at the plan's END -- both callers
        bracket it with their own snapshot.
        """
        steps = []
        board = self._read(eng)
        for direction in plan:
            before = (frozenset(board.cubes), frozenset(board.walls))
            here = (board.player, board.ptype)
            eng.step(direction)
            after = self._read(eng)
            kind = ("press"
                    if (frozenset(after.cubes), frozenset(after.walls)) != before
                    else "walk" if (after.player, after.ptype) != here else "stuck")
            steps.append((kind, here, direction, board))
            board = after
        return steps

    def _rebuild(self, eng, plan) -> list:
        """Replace every WALK run in ``plan`` with a shortest walk to the same
        ``(cell, colour)``, and keep the result only if the interpreter still
        wins on it.

        This is the other half of tidying a beam plan, and it is the half that
        `_shorten` cannot do. `_shorten` can only DELETE presses, so it leaves
        the routes between the presses exactly as the beam stumbled on them --
        and those routes are not shortest, which costs twice: presses that did
        not need taking, and a plan whose walks `_annotate` cannot label,
        because a step that is not on a shortest route has no equally-optimal
        siblings to offer and falls back to "the one the expert happened to
        take". Re-walking is free (the walk graph is already built once per
        board) and can only shorten the plan, since the run it replaces went
        between the same two nodes.

        Not applied to A* plans: their walks are shortest by construction (each
        came out of `_analyze`'s BFS), so the pass would be a no-op that could
        only introduce a difference.
        """
        start = snapshot(eng)
        steps = self._classify(eng, plan)
        restore(eng, start)
        out: list = []
        i = 0
        while i < len(steps):
            if steps[i][0] != "walk":
                out.append(steps[i][2])
                i += 1
                continue
            run = i
            while i < len(steps) and steps[i][0] == "walk":
                i += 1
            if i >= len(steps):                  # a trailing walk: nothing to
                out.extend(steps[j][2] for j in range(run, i))   # aim at
                continue
            goal = steps[i][1]
            parent = self._walk(steps[run][3])
            if goal in parent:
                out.extend(self._route(parent, goal))
            else:
                out.extend(steps[j][2] for j in range(run, i))
        won = False
        for j, direction in enumerate(out):
            eng.step(direction)
            if eng.check_win():
                won = j == len(out) - 1
                break
        restore(eng, start)
        return out if (won and len(out) <= len(plan)) else plan

    def _annotate(self, eng, plan) -> list:
        """Per-step optimal SETS for ``plan``, engine at the plan's start.

        A press that moved a cube (push, pull, burn) is labelled with itself: in
        a game whose win is where the cubes are, a sibling push is a different
        plan, not a reordering of this one. A maximal run of WALK steps ends on
        the cell the next press is taken from, and every direction that keeps
        the walk on a shortest route to that ``(cell, colour)`` is equally
        right -- a walk moves nothing, so all of them leave the identical board.
        """
        start = snapshot(eng)
        steps = self._classify(eng, plan)
        restore(eng, start)

        out: list = [None] * len(plan)
        i = 0
        while i < len(steps):
            if steps[i][0] != "walk":
                out[i] = [steps[i][2]]
                i += 1
                continue
            run = i
            while i < len(steps) and steps[i][0] == "walk":
                i += 1
            if i >= len(steps):
                # A plan always ends on the press that wins, so a walk run can
                # only ever be an approach to one. If an edit ever makes that
                # false, fall back to labelling what the expert did.
                for j in range(run, i):
                    out[j] = [steps[j][2]]
                continue
            goal = steps[i][1]                       # (cell, colour) of the press
            dist = self._reverse_walk(steps[run][3], goal)
            for j in range(run, i):
                node = steps[j][1]
                d0 = dist.get(node)
                (r, c), colour = node
                alts = [] if d0 is None else [
                    d for d, (dr, dc) in _DELTA.items()
                    if dist.get(((r + dr, c + dc),
                                 steps[run][3].plates.get((r + dr, c + dc),
                                                          colour))) == d0 - 1
                    and not (colour == self.yplayer
                             and (r - dr, c - dc) in steps[run][3].cubes)]
                # The recorded step is on a shortest route by construction, so a
                # set that does not contain it means the reconstruction drifted.
                out[j] = alts if steps[j][2] in alts else [steps[j][2]]
        restore(eng, start)
        return out


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class StickyCubesSolver(PSAStarSolver):
    game_id = "puzzlescript_sticky_cubes"
    game_name = GAME_NAME
    expert_cls = _StickyExpert

    #: `games/ps:sticky_cubes/ps:sticky_cubes.py` is a plain passthrough (it
    #: constructs the adapter and nothing else) -- the render fixes are in the
    #: .txt, which both paths read. Set this if that wrapper ever grows a patch.
    game_module_id = ""

    #: Macro nodes, not primitive ones: one node is a whole walk-and-press, and
    #: the hardest level here settles well inside this.
    node_cap = 200_000

    #: The heuristic over-charges a clump (see the module docstring), so
    #: ``weight`` is already effectively above 1 and raising it buys nothing.
    weight = 1

    #: Longest plan is 54 presses; the rest is room to re-plan after the
    #: exploration prefix. Under the adapter's own 200-step per-level budget.
    max_steps = 150

    #: Levels never attempted. Not a judgement that they are unwinnable -- they
    #: are ordinary puzzles a person solves -- but a measurement: at the shipped
    #: budget neither `_StickyExpert._astar` (40k macros) nor `_StickyExpert._beam`
    #: (250 wide, 200k macros) finds a plan, which is 6-10 minutes per level of
    #: finding nothing. Skipping them up front is what keeps a cold start
    #: minutes rather than an hour, and it matters more than it looks: the plan
    #: cache is written on completion, so N shards launched together by
    #: `parallelize_generator` would each pay the whole bill in parallel.
    #:
    #: What they need, in the order worth trying:
    #:   * ``--deep`` (a 700-wide, 700k-macro beam, no skip list) -- the same
    #:     search with the budget the shipped default cannot afford to spend at
    #:     every cold start. It is worth running: at that budget the beam DID
    #:     crack **L9** (39 presses raw, 33 after `_StickyExpert._shorten`,
    #:     engine-verified, ~8 minutes) and did NOT crack **L5** (given up after
    #:     ~16 minutes). Read the L9 result as promising rather than banked:
    #:     it was measured just before `_macros` stopped pruning blocked yellow
    #:     pulls (see the note there), so the shipped enumeration is slightly
    #:     wider than the one that found it and the re-run was not completed.
    #:     L1, L6 and L10 have not been run to completion at this budget at all;
    #:   * a NATIVE model of the sticky mechanic. The whole search is one
    #:     interpreter tick per node at ~4ms, and a re-implementation would be
    #:     ~100x that, which is the difference between 200k boards per level and
    #:     20M. It is the real fix and it is real work: the clump/shear
    #:     semantics come out of PuzzleScript's rule ORDER (see the shear note
    #:     above), so a model has to be fuzzed against the interpreter the way
    #:     ``--selfcheck`` fuzzes the shortcuts here.
    #:
    #: Why each is hard, from watching the searches: L1 and L5 need cubes burnt
    #: or sheared to an exact count before anything can be pushed home (L5 ships
    #: 8 cubes for 8 targets, so every cube lost is fatal and the heuristic
    #: cannot see which loss is which); L6 is the biggest board in the game, 23
    #: cubes and 12 targets; L9 wants cubes ROOTED on earth plates as anchors,
    #: which the heuristic scores as destroying them; L10 needs an 8-cube slab
    #: sheared into a single row to fit a 3-cell cube-plate doorway.
    skip_levels = frozenset({1, 5, 6, 9, 10})

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it fills the disk cache in one pass and makes the
        startup cost visible as startup."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(plans: bool = True):
    """The solver, its adapter and its expert.

    ``plans=False`` skips `_ensure` -- i.e. skips planning all eleven levels --
    for the reports that only need the board and the expert's board reading
    (``--audit``, and the mechanics half of ``--selfcheck``). Those searches are
    minutes on a cold disk cache and none of it is what those reports measure.
    """
    solver = StickyCubesSolver()
    if plans:
        game, expert, solvable = solver._ensure(0)
        return solver, game, expert, solvable
    game = solver.make_game(0)
    expert = solver.expert_cls(game, node_cap=solver.node_cap,
                               weight=solver.weight)
    return solver, game, expert, None


def _plans(verbose: bool = True) -> int:
    """Every level's plan, replayed through the REAL interpreter.

    The end-to-end test of the search, the macro shortcut and the tie labelling
    at once: the plan is stepped through the interpreter press by press and the
    win is the interpreter's own. A level in `StickyCubesSolver.skip_levels` is
    listed but not attempted -- see that attribute for what was tried on it.

    On a cold `plan_cache_path` this pays for every search (minutes per level);
    afterwards it is instant.
    """
    solver, game, expert, solvable = _new()
    idx = game._game.obj_name_to_idx
    cube_ids = set(game._game.resolve_object_name("cube"))
    total = ties = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        cubes = sum(1 for row in eng.grid for cell in row if cell & cube_ids)
        targets = sum(1 for row in eng.grid for cell in row if idx["target"] in cell)
        if level in solver.skip_levels:
            if verbose:
                print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  "
                      f"{cubes:2d} cubes / {targets} targets   SKIPPED "
                      f"(not attempted -- see skip_levels)")
            continue
        plan = expert.plan(eng, level)
        if plan is None:
            if verbose:
                print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  "
                      f"{cubes:2d} cubes / {targets} targets   UNSOLVED")
            continue
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        sets = getattr(plan, "optsets", None) or [[p] for p in plan]
        tie_steps = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        ties += tie_steps
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        if verbose:
            print(f"  L{level:2d}: {eng.height:2d}x{eng.width:2d}  "
                  f"{cubes:2d} cubes / {targets} targets  {len(plan):3d} presses  "
                  f"win={won}  (budget {game._max_steps}, {room})  "
                  f"{tie_steps:3d} steps with a tie set")
    if verbose:
        print(f"  solvable levels: {solvable}")
        print(f"  {total} presses, {ties} of them with a second equally-right "
              f"answer")
        print("  every plan reaches a WIN" if not bad else f"  {bad} PROBLEM(S)")
    return 1 if bad else 0


#: A press that changes nothing is not a mechanic; these are the four moves plus
#: X, which is what the expert and the exploration policy both branch on.
_ALL = DIRECTIONS


def _ascii(game) -> str:
    """The engine grid as ASCII, for a check that fails to print with.

    A failure message that names a press without showing the board it was taken
    on is a failure you have to reproduce before you can read it, and these
    boards come out of a seeded random walk hundreds of presses deep."""
    g = game._game
    name = g.obj_idx_to_name
    bodies = {"bwall": "#", "cuben": "c", "cubeu": "^", "cubed": "v",
              "cubel": "<", "cuber": ">", "cubeg": "G", "bplayer": "B",
              "rplayer": "R", "yplayer": "Y", "gplayer": "g"}
    plates = {"target": "t", "fire": "f", "magnet": "m", "water": "w",
              "earth": "e", "crate": "r"}
    out = []
    for row in game._engine.grid:
        line = ""
        for cell in row:
            body = [bodies[name[o]] for o in cell if name[o] in bodies]
            plate = [plates[name[o]] for o in cell if name[o] in plates]
            line += body[0] if body else (plate[0] if plate else ".")
        out.append("    " + line)
    return "\n".join(out)


def _reachable_states(game, level, presses, keep, rng):
    """``keep`` boards sampled from a seeded random walk of ``presses`` presses.

    Random rather than plan-driven on purpose: the plans never park a cube in a
    corner, never root one on an earth plate they did not mean to, never press X
    with nothing beside them and never leave a cube mid-launch, and those are
    exactly the boards the expert's shortcuts have to be right about.

    Sampled rather than exhaustive because one interpreter tick is ~4ms and the
    checks below replay every macro of every board TWICE: the full walk on all
    eleven levels is several hours, and the same bugs show up in a few hundred
    boards. The walk itself is still run press by press -- it is the cheap half
    -- and only which boards get checked is thinned."""
    eng = game._engine
    game.set_level(level)
    boards = [snapshot(eng)]
    for _ in range(presses):
        eng.step(rng.choice(_ALL))
        if eng.check_win():
            game.set_level(level)
        boards.append(snapshot(eng))
    if len(boards) <= keep:
        return boards
    # Evenly spaced, so the sample spans the whole walk rather than its opening.
    step = len(boards) / keep
    return [boards[int(i * step)] for i in range(keep)]


def _selfcheck(walk_presses: int = 250, keep: int = 40, per_board: int = 10,
               verbose: bool = True) -> int:
    """Everything the expert would otherwise be trusted on.

    1. **The measured mechanics.** The plate rewrites, the cube-plate rule, the
       burn's four-way reach and the magnet SHEAR, each on a board built for it
       and compared against the interpreter. This is the executable form of the
       module docstring: if a claim there is wrong, it is wrong here.
    2. **`_seat` is the walk.** Every macro of a few thousand random boards is
       applied twice -- once by seating the player and stepping only the press,
       once by replaying every primitive -- and the grids are compared cell for
       cell. This is the shortcut that makes the search finish, and it is the
       one place the expert stops asking the interpreter.
    3. **A launched cube is its own state.** `_key` keeps CubeU/D/L/R apart
       from CubeN, which costs states and is only worth it if the two really do
       behave differently. Checked by taking every board a random walk reaches
       with one on it, rewriting it to a CubeN, and counting the presses after
       which the two boards differ. (The first draft of this expert asserted the
       OPPOSITE -- that they were the same state -- and this is the check that
       disproved it.)
    4. **The walk graph is inert and complete.** Every edge it offers moves the
       player and nothing else, and lands it on the colour the graph predicted;
       every press that DOES move a cube from the player's own cell is offered
       as a zero-walk macro. The completeness half is the one that has actually
       caught something: `_macros` used to skip a yellow pull whose destination
       was blocked, on the reasonable-sounding ground that a player who cannot
       move drags nothing -- and it does, because the cube behind it is stopped
       by the player's body rather than by a `Wall`, so the rest of that cube's
       clump shears past. The board it found that on was 250 random presses
       into level 2 and no plan would ever have gone near it.

    Sampled, not exhaustive: one interpreter tick is ~4ms and 2 and 4 replay
    every macro of every board twice, so the full walk on all eleven levels is
    hours. ``keep`` boards per level, ``per_board`` macros on each; the walk
    itself is still stepped press by press.
    """
    _solver, game, expert, _solvable = _new(plans=False)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    name = g.obj_idx_to_name
    bad = 0

    def board(rows):
        """Build a level from ASCII and load it (level-start rules included)."""
        ch = {"#": "bwall", "c": "cuben", "G": "cubeg", "p": "bplayer",
              "R": "rplayer", "Y": "yplayer", "g": "gplayer", "t": "target",
              "f": "fire", "m": "magnet", "w": "water", "e": "earth",
              "r": "crate", ".": None}
        grid = []
        for line in rows:
            row = []
            for c in line:
                cell = {idx["background"]}
                if ch[c]:
                    cell.add(idx[ch[c]])
                row.append(cell)
            grid.append(row)
        eng.load_level(grid)

    def at(r, c):
        return sorted(name[o] for o in eng.grid[r][c]
                      if name[o] not in ("background",)
                      and not g._group_contains("border", name[o]))

    def expect(what, got, want):
        nonlocal bad
        ok = got == want
        bad += not ok
        if verbose or not ok:
            print(f"    {'ok  ' if ok else 'FAIL'} {what}: {got}"
                  + ("" if ok else f"  (expected {want})"))

    # -- 1. the mechanics ----------------------------------------------------
    if verbose:
        print("  measured mechanics")
    for plate, becomes, why in (("f", "rplayer", "fire"), ("w", "bplayer", "water"),
                                ("m", "yplayer", "magnet"), ("e", "gplayer", "earth")):
        board(["#####", f"#p{plate}.#", "#####"])
        eng.step("right")
        expect(f"player onto '{plate}' becomes {becomes}", at(1, 2),
               sorted([becomes, why]))

    board(["#####", "#pr.#", "#####"])
    eng.step("right")
    expect("player refused by a cube plate", at(1, 1), ["bplayer"])

    board(["#####", "#pcr#", "#####"])
    eng.step("right")
    expect("a cube passes onto a cube plate",
           at(1, 3) + at(1, 2), ["crate", "cuben", "bplayer"])

    board(["#####", "#.c.#", "#cRc#", "#.c.#", "#####"])
    eng.step("action")
    expect("X burns all four neighbours at once",
           [x for r in range(1, 4) for c in range(1, 4) for x in at(r, c)],
           ["rplayer"])

    board(["######", "#.cc.#", "#pccm#", "######"])
    eng.step("right")
    expect("a magnet SHEARS the clump it anchors",
           sorted((r, c) for r in range(4) for c in range(6)
                  if "cuben" in at(r, c)),
           [(1, 3), (1, 4), (2, 2), (2, 3)])

    board(["#####", "#pce#", "#####"])
    eng.step("right")
    expect("a cube shoved onto earth is ROOTED into a wall",
           at(1, 3), ["cubeg", "earth"])
    expect("...and a rooted cube is no longer a Cube",
           idx["cubeg"] in set(g.resolve_object_name("cube")), False)

    board(["#####", "#pcf#", "#####"])
    eng.step("right")
    expect("a cube shoved onto fire is DELETED", at(1, 3), ["fire"])

    board(["##########", "#........#", "#pe......#", "#........#", "##########"])
    eng.step("right")
    expect("the earth plate makes the player green", at(2, 2),
           ["earth", "gplayer"])

    def cube_at():
        return [(r, c, x) for r in range(5) for c in range(10)
                for x in at(r, c) if x.startswith("cube")]

    board(["##########", "#........#", "#gc......#", "#........#", "##########"])
    eng.step("right")
    expect("a green push LAUNCHES the cube (rewritten, one cell moved)",
           cube_at(), [(2, 3, "cuber")])
    eng.step("up")
    expect("...and the launch is worth ONE more cell, on the next press",
           cube_at(), [(2, 4, "cuben")])
    for _ in range(4):
        eng.step("up")
    expect("...after which it is a settled cube and travels no further",
           cube_at(), [(2, 4, "cuben")])

    board(["######", "#.c..#", "#Y...#", "######"])
    eng.step("down")
    expect("yellow PULLS the cube it walks away from", at(1, 2), ["cuben"])

    def cubes_at():
        return sorted((r, c) for r in range(5) for c in range(5)
                      if "cuben" in at(r, c))

    board(["#####", "#.Y.#", "#cc.#", "#...#", "#####"])
    eng.step("up")
    expect("a yellow player who CANNOT MOVE still pulls -- the cube behind is "
           "stopped by the player's body, the rest of its clump shears past",
           cubes_at(), [(1, 1), (2, 2)])
    board(["#####", "#.Y.#", "#.c.#", "#...#", "#####"])
    eng.step("up")
    expect("...but a lone cube behind a blocked player does not move",
           cubes_at(), [(2, 2)])
    board(["#####", "#.p.#", "#cc.#", "#...#", "#####"])
    eng.step("up")
    expect("...and a BLUE player pulls nothing at all", cubes_at(),
           [(2, 1), (2, 2)])

    # -- 2, 3, 4: the expert's shortcuts, over random boards -----------------
    seats = collapses = edges = macro_checks = 0
    rng = random.Random("sticky_cubes:selfcheck")
    for level in range(game.n_levels):
        for snap in _reachable_states(game, level, walk_presses, keep, rng):
            restore(eng, snap)
            b = expert._read(eng)
            if b.player is None:
                continue
            parent = expert._walk(b)
            macros = expert._macros(b, parent)
            if len(macros) > per_board:
                macros = rng.sample(macros, per_board)

            # 2. seating a macro == replaying its primitives
            for macro in macros:
                restore(eng, snap)
                expert._apply_macro(eng, b, macro)
                fast = [[set(cell) for cell in row] for row in eng.grid]
                restore(eng, snap)
                for direction in macro[0]:
                    eng.step(direction)
                if eng.grid != fast:
                    bad += 1
                    print(f"    FAIL L{level}: seating disagrees with the walk "
                          f"for macro {macro[0]}")
                seats += 1

            # 4a. every walk edge is inert and lands on the predicted colour
            for (cell, colour), link in parent.items():
                if link is None:
                    continue
                (prev, pcolour), direction = link
                restore(eng, snap)
                if (prev, pcolour) != (b.player, b.ptype):
                    continue                    # only the first step is cheap
                eng.step(direction)
                nb = expert._read(eng)
                if (nb.cubes != b.cubes or nb.walls != b.walls
                        or (nb.player, nb.ptype) != (cell, colour)):
                    bad += 1
                    print(f"    FAIL L{level}: walk edge {direction} is not inert")
                edges += 1

            # 4b. every press that moves a cube from HERE is offered as a macro
            zero = {m[0][0] for m in expert._macros(b, parent) if len(m[0]) == 1}
            for direction in _ALL:
                restore(eng, snap)
                eng.step(direction)
                nb = expert._read(eng)
                if (nb.cubes, nb.walls) != (b.cubes, b.walls) \
                        and direction not in zero:
                    bad += 1
                    restore(eng, snap)
                    print(f"    FAIL L{level}: '{direction}' changes the cubes "
                          f"but is not a macro (player {b.player} "
                          f"{name[b.ptype]}, zero-walk macros {sorted(zero)})")
                    print(_ascii(game))
                macro_checks += 1

            # 3. a launched cube is NOT a settled one -- the regression guard
            #    for `_key`. Rewrite every CubeU/D/L/R on the board to a CubeN
            #    and require the two boards to answer some press differently.
            #    Per board they may legitimately agree (a launch into a wall has
            #    nothing left to spend), so the assertion is on the total: if
            #    NOTHING ever disagreed, keying them apart would be dead weight
            #    and the docstring's account of the launch would be wrong.
            if b.cubem:
                plain = [[set(cell) for cell in row] for row in snap]
                for (r, c) in b.cubem:
                    plain[r][c] = ((plain[r][c] - expert.cubem_ids)
                                   | {expert.cuben})
                for direction in _ALL:
                    restore(eng, snap)
                    eng.step(direction)
                    one = expert._key(eng)
                    restore(eng, plain)
                    eng.step(direction)
                    collapses += expert._key(eng) != one
    if not collapses:
        bad += 1
        print("    FAIL a launched cube never behaved differently from a settled "
              "one -- either the walk never launched anything, or `_key` is "
              "splitting states for no reason")
    if verbose:
        print(f"  {seats} macros applied both ways, {edges} walk edges, "
              f"{macro_checks} press classifications, {collapses} presses where a "
              f"launched cube outlived a settled one")
    return bad


# ---------------------------------------------------------------------------
# Rendering audit
# ---------------------------------------------------------------------------

#: Every cell stack this game can actually show. The terrain plates are one
#: collision layer and the bodies are another, so a stack is at most one of
#: each -- but most of that product is unreachable, and listing what IS reachable
#: is the point: a pairwise audit over the full product reports a wall drawn on
#: a fire plate as a clash, which is true and means nothing.
#:
#: The exclusions, each for a reason in the rules rather than in the levels:
#:   * a plate under a WALL -- no level places one and no rule creates one;
#:   * ``cubeg`` off an earth plate -- ``LATE [Cube Earth] -> [CubeG Earth]`` is
#:     the only thing that makes one, and nothing ever moves it again;
#:   * a cube on a MAGNET (``[> Cube|Magnet]`` refuses it) or on FIRE / EARTH
#:     (both rewrite it the moment it lands, so the stack does not survive a
#:     tick);
#:   * a player on a CUBE PLATE (``[> Player|Crate]`` refuses it);
#:   * every player/plate pair but the matching one, because standing on a plate
#:     IS what sets the colour -- there is no blue player on a fire plate.
#: The four cube stacks and the twelve player stacks are the ones that matter:
#: ``cuben+target`` is the win being assembled, and the player's plate says
#: which verb the next press will be.
_STACKS = [
    (),
    ("target",), ("fire",), ("magnet",), ("water",), ("earth",), ("crate",),
    ("bwall",), ("cubeg", "earth"),
    ("cuben",), ("cuben", "target"), ("cuben", "crate"), ("cuben", "water"),
    ("bplayer",), ("bplayer", "target"), ("bplayer", "water"),
    ("rplayer",), ("rplayer", "target"), ("rplayer", "fire"),
    ("yplayer",), ("yplayer", "target"), ("yplayer", "magnet"),
    ("gplayer",), ("gplayer", "target"), ("gplayer", "earth"),
]

#: What can stand in the cell ABOVE, because every one of them paints a "3D"
#: Border object into the cell below it and that ink lands on the top rows of
#: the stack being audited.
_ABOVE = (None, "cuben", "cubeg", "bwall", "bplayer", "rplayer", "yplayer",
          "gplayer", "target", "fire", "magnet", "water", "earth", "crate")


def _audit(verbose: bool = True) -> int:
    """Every cell composition the game can show, at every board shape it uses,
    under every neighbour that draws ink into it.

    WHOLE 64x64 frames of boards that differ in ONE cell are compared, never a
    cell sliced out of one board: `_render_frame` upscales the rendered board to
    fill the frame and then letterboxes it, so the cell grid in the output is
    not ``cell_px``-aligned and slicing reads the wrong pixels (the ps:explod
    lesson). Two boards identical but for one cell render identically iff that
    cell does.

    The neighbour axis is not decoration. This game's "3D effect" is a dozen
    ``LATE DOWN [X| ] -> [X|XB]`` rules that stamp a marker into the cell BELOW
    every object, so what a cell looks like depends on what is above it, and a
    composition audit that renders each cell in isolation would be auditing
    boards the game never draws. Running the composition through `load_level`
    (this game has ``run_rules_on_level_start``) is what makes those markers
    real -- and it also NORMALISES the composition, which is how the unreachable
    pairs are dropped: the engine rewrites ``cube on fire`` to ``fire`` before
    the frame is taken, and the two are then one entry rather than a false clash.

    Also checks that no composition renders as a flat block of the letterbox
    colour, which a pairwise matrix structurally cannot catch (the other half of
    that pair is not a composition -- see ps:stand_iii).
    """
    from adapters.puzzlescript_adapter import _render_frame          # noqa: PLC0415

    _solver, game, _expert, _solvable = _new(plans=False)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    name = g.obj_idx_to_name

    shapes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        shapes.setdefault((eng.height, eng.width), []).append(level)

    def build(h, w, cell_objs, above_objs, at=(4, 6)):
        grid = [[{idx["background"]} for _ in range(w)] for _ in range(h)]
        r, c = at
        for o in cell_objs:
            grid[r][c].add(idx[o])
        for o in above_objs:
            grid[r - 1][c].add(idx[o])
        eng.load_level(grid)

    def content(r, c):
        return tuple(sorted(name[o] for o in eng.grid[r][c]
                            if name[o] != "background"
                            and not g._group_contains("border", name[o])))

    def label(comp):
        return "+".join(comp) or "floor"

    bad = 0
    for (h, w), levels in sorted(shapes.items()):
        cell = min(64 // h, 64 // w)
        clashes = []
        for above in _ABOVE:
            shots: dict = {}
            for comp in _STACKS:
                build(h, w, comp, () if above is None else (above,))
                got = content(4, 6)
                if got in shots:
                    continue                # the engine normalised it onto one
                shots[got] = np.asarray(_render_frame(eng, g)).copy()
            for a, b in itertools.combinations(sorted(shots), 2):
                if np.array_equal(shots[a], shots[b]):
                    clashes.append((above, a, b))
        bad += len(clashes)
        if verbose:
            print(f"  {h:2d}x{w:2d} (cell {cell}px, levels "
                  f"{','.join(str(x) for x in levels)}): "
                  f"{len(_ABOVE)} neighbour contexts x {len(_STACKS)} stacks -- "
                  f"{'OK' if not clashes else f'{len(clashes)} CLASH'}")
        for above, a, b in clashes:
            print(f"      under {above or 'nothing'}: IDENTICAL "
                  f"{label(a)}  ==  {label(b)}")

    # The letterbox check: a cell that renders as the pad colour is invisible
    # against the border of the picture.
    pad = int(np.asarray(_render_frame(eng, g))[0, 0])
    for (h, w) in sorted(shapes):
        for comp in _STACKS:
            grid = [[{idx["background"]} | {idx[o] for o in comp}
                     for _ in range(w)] for _ in range(h)]
            eng.load_level(grid)
            frame = np.asarray(_render_frame(eng, g))
            if np.all(frame == pad):
                bad += 1
                print(f"      {label(comp)} renders as the letterbox colour "
                      f"{pad} at {h}x{w}")
    if verbose:
        print(f"  letterbox colour is {pad}; no composition is a flat block of it"
              if not bad else "")
    return bad


# ---------------------------------------------------------------------------
# The rotation contract, end to end
# ---------------------------------------------------------------------------

def _symmetry(verbose: bool = True) -> int:
    """Replay every plan through the ADAPTER at every presentation it draws,
    and require both the WIN and that the frames are exactly the rotation of the
    unaugmented ones.

    This is the executable form of the rotation contract in
    `solvers/common/ps_astar`: the expert plans in ENGINE space and
    `screen_action` converts each direction to the button an agent presses in
    the rotated view. Get that backwards and nothing raises -- the adapter just
    executes a different direction than planned -- so it is measured rather than
    argued. Sticky Cubes takes rotation only (it is not in
    `PuzzleScriptAdapter._FLIP_GAMES`), so there are four presentations.

    It is deliberately NOT proposed for the flip set. A mirror is only a
    symmetry of a game whose left/right rules are exact mirrors of each other,
    and this rule list is not obviously that: the sliding and the green-launch
    rules are four per-direction copies evaluated in a FIXED order (up, down,
    left, right), and this game has at least one mechanic whose outcome is
    decided by rule order rather than by geometry -- the shear, which happens
    because ``[Moving Cube|STATIONARY Cube]`` runs before the magnet rule and
    ``[> Cube|STATIONARY Cube]`` after it. A chirality hiding in an argument of
    exactly that shape is what Gobble Rush turned out to have. Rotation is safe
    and is what ships; adding flips would need this check re-run over all
    sixteen presentations with the ENGINE grid compared after every press, not
    just the frames.
    """
    solver, game, expert, solvable = _new()
    plans = {}
    for level in solvable:
        game.set_level(level)
        plans[level] = expert.plan(game._engine, level)

    ref: dict = {}
    seen: set = set()
    bad = 0
    for seed in range(80):
        if len(seen) == 4 and seed > 12:
            break
        aug = solver.make_game(seed)
        for level, plan in plans.items():
            if plan is None:
                continue
            aug.set_level(level)
            k = (aug._rotation_k, aug._hflip, aug._vflip)
            seen.add(k)
            frames = [np.asarray(aug._current_frame)]
            for direction in plan:
                act = screen_action(direction, *k)
                fd = aug.perform_action(ActionInput(id=act))
                frames.append(np.asarray(fd.frame[-1] if fd.frame
                                         else aug._current_frame))
            if aug._state != GameState.WIN:
                bad += 1
                print(f"    seed {seed} L{level} {k}: the plan did not WIN")
            if k == (0, False, False):
                ref[level] = frames
                continue
            if level not in ref:
                continue
            if any(not np.array_equal(np.rot90(a, k=k[0]), b)
                   for a, b in zip(ref[level], frames)):
                bad += 1
                print(f"    seed {seed} L{level} {k}: frames are not the "
                      f"rotation of the unaugmented ones")
    if verbose:
        print(f"  {len(seen)} presentations drawn: {sorted(seen)}")
    return bad


def _deep() -> None:
    """Raise the beam budget and attempt every level, including the ones
    `StickyCubesSolver.skip_levels` gives up on by default.

    The shipped budget is chosen so a cold plan cache costs minutes per level.
    ``--deep`` trades that for reach: a wider beam, more of it, and no skip
    list. It is worth running ONCE, in one process, before any sharded
    generation -- whatever it finds lands in `plan_cache_path` and every later
    run and every shard reads it from there for free. See that attribute and
    `_StickyExpert._beam`.
    """
    _StickyExpert.beam_width = 700
    _StickyExpert.beam_depth = 35
    _StickyExpert.beam_node_cap = 700_000
    _StickyExpert.shorten_tick_budget = 60_000
    StickyCubesSolver.skip_levels = frozenset()


if __name__ == "__main__":
    if "--deep" in sys.argv:
        sys.argv.remove("--deep")          # argparse must not see it
        _deep()
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = _audit()
        print(f"audit: {collisions} colliding compositions")
        sys.exit(1 if collisions else 0)
    if "--symmetry" in sys.argv:
        violations = _symmetry()
        print(f"symmetry: {violations} violations")
        sys.exit(1 if violations else 0)
    sys.exit(StickyCubesSolver.main())
