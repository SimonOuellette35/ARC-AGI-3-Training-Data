"""Generate Phase-1 training data for ps:aperture_science_sokoban_testing_initiative.

The harness -- the engine-blackbox macro searches, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is the game-specific part: the macro set its mechanics
need, the goal heuristic, the skipped levels and the search budget.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_aperture_science_sokoban_testing_initiative",
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
Portal, as a sokoban. Chell walks the four directions and presses ACTION (X) to
fire a portal; the win is always the same -- reach the Level Evacuation Terminal
(the Target tile), or on the last level the cake. Stacked on top of that:

  * CRATES and SENTRIES are pushed (a crate can push a sentry). Either falls into
    a Hole and is destroyed, and either blocks a laser.
  * BUTTONS open DOORS while held down -- by the player standing on one, or
    permanently by a crate parked on one. Two independent circuits (the "Lie"
    circuit is a visually identical second set). A button hidden behind Metal is
    a laser target, pressed by a laser touching it. Buttons also flip "rotating
    walls" (Metal+DoorOpen) into portalable Wall.
  * SENTRIES fire a laser in their facing direction (or in all four); the beam
    walks the board every turn and kills the player on contact. Anything in
    LaserBlock (crate, metal, door, sentry, wall, portal) stops it.
  * PORTALS are fired at a bare Wall face and alternate blue/orange, so only one
    of each exists; a portal can be entered only from the face it was shot into,
    and it teleports crates, sentries, the player -- and lasers.
  * The FIZZLER (a bare NextShotBlue column) passes the player but destroys
    crates, sentries and portal shots, and cancels your portals on contact.

Levels that never place a Shootable marker under their Target have no portal
device at all, and ACTION is a dead key there.

The `SentA` fix in the game file
--------------------------------
`data/puzzlescript_games/Aperture_Science_Sokoban_Testing_Initiative.txt` used
the legend's and-composites (`SentR = Sentry and ShotR`, `SentA = Sentry and
BlueShot`, ...) inside its rules. `PuzzleScriptAdapter` flattens an and-composite
to a plain index list and then matches it with ANY, so `SentA` matched "a sentry
OR a blueshot" -- and `late [player | SentA] -> [ | SentA]` therefore killed the
player next to ANY sentry, in any direction. That made every sentry untouchable:
level 11 killed the player during the level-start rule pass (unwinnable before
the first key press) and every "push the sentry into the beam" level dead-ended.
The rules now spell the conjunction out inline (`[player | Sentry BlueShot]`),
which is the same thing in PuzzleScript and which the adapter's multi-object cell
patterns already match with AND. Only this game's file is touched -- ten games in
`data/puzzlescript_games/` use an and-composite name in their rules and none of
the others has a generator -- so no other corpus changes.

Expert solver
-------------
Macro search over the real interpreter: a successor is `walk somewhere, then do
one primitive`, so the search depth is the number of DECISIONS rather than the
number of key presses. That matters more here than in the other ps: games,
because the interpreter step is the entire cost of the search -- 8 ms on the
small boards and up to 50 ms on the ones carrying live lasers, since the game
has ~120 late rules for the beams and the two door circuits. A primitive-action
A* gets ~5k engine steps into a 2-minute budget and wins 3 levels; the same
budget in macros wins them in a few hundred steps.

Beyond the inherited push macros, `extra_macros` adds the moves this game has
that a sokoban does not:

  * WALK ONTO the exit (that IS the win) or onto a button (that opens its doors,
    for as long as she stands there).
  * STEP INTO a portal mouth -- portals sit on wall cells, so the walk BFS calls
    them blocked and would never route into one.
  * FIRE a portal, three ways: on the facing the walk arrived with; after
    turning to face a direction (a direction press turns Chell whether or not
    she can move, so firing at the wall you stand against is legal and is often
    the only shot available); and, blind, by stepping onto a button and firing
    in the same macro -- see `_shot_macros` for why that one cannot be split in
    two. Shots are deduped by the REGION THE MOUTH OPENS INTO, which is the only
    thing about a shot that matters and cuts ~20 shots per node to a handful.

The walk model is only a proposer -- it can route through a door that closes as
you step off its button, or into a beam that is about to move. Every macro is
executed on the real interpreter and the resulting state is what gets searched,
so a wrong proposal costs nodes and never correctness. Any returned plan is a
genuine WIN path.

Two strategies run in order, because one heuristic cannot cover both halves of
this game. Weighted A* handles the levels whose exit its player-distance
estimate can see. On the deep levels that estimate is FLAT -- level 3 spends
eight pushes shoving crates into pits in a side room, and level 5 spends its
whole middle placing two portals, without the player getting one step nearer the
exit -- and a flat heuristic gives A* nothing to steer with at a depth where
uniform-cost search is hopeless. There the beam takes over (`PSBeamExpert`),
spending the same budget on breadth per depth and using the heuristic only to
break ties. Beam plans win but wander, so `_shorten` deletes every block the
engine agrees is unnecessary (68 -> 42 moves on level 3, 48 -> 33 on level 6).

Solvable levels
---------------
{0, 1, 2, 3, 4, 6} plus the cake on 22 (which is free -- one walk). Together they
cover crate pushing, a button-and-door, pre-placed portals that the player and a
crate travel through, firing the portal device across a pit, and portalling a
crate out of a column it could never have been pushed out of.

LEVEL 5 IS UNWINNABLE, and not for want of searching -- it is a deadlock in the
level as this interpreter runs it. Its exit sits behind door (3,3); the only
crate sits behind door (5,8); both doors are on the one button at (5,5), and a
door is open only while a player stands on that button or a solid sits in the
doorway itself. So the crate has to reach the button, and it cannot:

  * Pushed through door (5,8), it would have to be shoved from the right room
    while the door is open -- but the only thing that opens that door is the
    player standing on the button, three cells away on the far side.
  * Carried by a portal, it lands on a portal MOUTH, and a mouth is always the
    last free cell before a wall -- so in this room always on the edge ring
    (row 4, row 7, column 1 or column 7). A crate on an edge can only slide
    along it: pushing it inward needs the player on the far side, which is the
    wall. The one gap in that ring is doorway (5,8), which closes the loop
    above.

Two searches confirm it from the other side: a full-width beam with the shot
dedup relaxed to exact wall faces closes out, and the progress-ranked beam drives
the crate to (5,9) -- hard against the door -- and stalls there for ten more
plies. The level is skipped rather than papered over.

Levels 7-21 are skipped too. They are where the sentries and live laser grids
start, which is also where the interpreter step gets ~5x more expensive, and the
beam that cracks 3 and 6 in 7-11 minutes does not reach them in an hour. Two of
them (10 and 21) EXHAUST their search space outright rather than time out.
Because each level is an independent trajectory in the training schema, an
episode holding only the solvable levels is valid, and ``level_id`` preserves the
true (seed, level) provenance -- so raising this later is a matter of finding
plans for more levels, not of changing the format.

The beam searches are not re-run at every startup: their results are stored in
`_BEAM_PLANS` and replayed for validation, so a normal run pays nothing for them
(a full 7-level episode records in ~2 s after a ~6 s startup).

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps); only the presentation varies per (seed, level) -- `PuzzleScriptAdapter`
draws a frame rotation (rotation_k in 0..3) from a private RNG. This game gets no
flip and no recolor. The expert plan is therefore seed-independent: solved once
per level, cached, and replayed per seed with that seed's rotation-remapped
screen actions and rotated frames.

Usage (run from the repo root):
    python solvers/generate_aperture_science_training.py --episodes 200 \
        --out data/training_multi_level/aperture_science
"""

from __future__ import annotations

import heapq
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                       # noqa: E402
    PSAStarSolver, PSBeamExpert, PSExpert, _DELTA, restore, snapshot,
)

GAME_NAME = "Aperture_Science_Sokoban_Testing_Initiative"

#: Plans the beam found for the levels A* cannot reach, one letter per key
#: press (u/d/l/r = the four directions, x = ACTION), already run through
#: `ApertureExpert._shorten`.
#:
#: These are CHECKED, NOT TRUSTED: `ApertureSolver._preload_plans` replays each
#: one on the interpreter at that level's start and only seeds the expert's memo
#: with the ones that actually win, so editing the game file can make a plan
#: stale but can never make it wrong -- a plan that stops winning is dropped and
#: the level is searched again from scratch. They are here rather than in a
#: cache under ``data/`` because the search that produced them costs 7-11
#: minutes a level, and that is a bad thing to re-pay on a fresh checkout for a
#: result that is a pure function of the game file.
_BEAM_PLANS: dict[int, str] = {
    3: "lldrdduuluuudddlddruuuluuruuudddrdrrrrrdrr",
    6: "uuurrlxuuldrrdxllllurdrddlrxuuldd",
}

#: Heuristic value for a state whose exit no chain of clearable obstacles
#: reaches (it is walled off, and only a portal could bridge it) and for one
#: whose player is gone (walked into a beam or a pit -- a dead branch).
UNREACHABLE = 60
DEAD = 500


class ApertureExpert(PSBeamExpert):
    """Macro search over the real interpreter, with the game's non-push moves.

    Two strategies, in order. Weighted macro A* handles the levels whose exit
    the player-distance heuristic can actually see, and its plans are short.
    When A* gives up, the beam takes over: the deep levels spend a dozen moves
    rearranging crates and portals without the player getting one step nearer
    the exit, and over that stretch the heuristic is flat and A* has nothing to
    steer with (see `PSBeamExpert`). Beam plans wander, so they go through
    `_shorten` before being returned.
    """

    pushable_names = ("crate", "sentry")
    #: Everything the player cannot step onto. Portals are included because they
    #: replace their wall (``late [portal wall] -> [portal]``) and are entered
    #: only through the dedicated macro below; Hole and Laser are lethal, so a
    #: walk must never route through them.
    blocker_names = ("wall", "metal", "door", "hole", "laser", "portal")

    def setup(self) -> None:
        super().setup()
        g = self.g
        idx = g.obj_name_to_idx
        self.wall = idx["wall"]
        self.metal = idx["metal"]
        self.door = idx["door"]
        self.hole = idx["hole"]
        self.target = idx["target"]
        self.lie = idx["lie"]
        self.button = idx["button"]
        self.wb = idx["weighedbutton"]
        self.nsb = idx["nextshotblue"]
        self.anything_ids = set(g.resolve_object_name("anything"))
        self.portal_ids = set(g.resolve_object_name("portal"))
        # A direction press that cannot move Chell still turns her, so she can
        # fire at the surface she is standing against. Only these blockers give
        # that free turn: a hole or a laser would kill her, a crate or a sentry
        # would be a push (already a macro of its own).
        self.turn_only_ids = {self.wall, self.metal, self.door} | self.portal_ids
        # A portal shot dies on any Solid or on an existing portal, and forms a
        # portal only on a bare Wall.
        self.shot_stop_ids = set(g.resolve_object_name("solid")) | self.portal_ids \
            | {self.wall}
        self.laser_ids = set(g.resolve_object_name("laser"))
        # Which way each portal spits a solid back out. A shot travelling UP
        # lands an OrPortU/BlPortU on the wall ABOVE the shooter, so that
        # portal's mouth faces DOWN, back at whoever fired it -- entry and exit
        # are both on the shooter's side. (That is also why you can never open a
        # sealed room by shooting its outside wall.)
        self.portal_spit = {
            idx["orportu"]: (1, 0), idx["blportu"]: (1, 0),
            idx["orportd"]: (-1, 0), idx["blportd"]: (-1, 0),
            idx["orportl"]: (0, 1), idx["blportl"]: (0, 1),
            idx["orportr"]: (0, -1), idx["blportr"]: (0, -1),
        }
        # Permanent scenery for the heuristic: everything else on the board can
        # be cleared, pushed, opened or blocked by some sequence of moves.
        self.impassable_ids = {self.wall, self.metal} | self.portal_ids

    # -- strategy ------------------------------------------------------------
    def _search(self, eng) -> list | None:
        """Weighted macro A* first; the beam (plus a shortening pass) after."""
        plan = self._astar(eng)
        if plan is not None:
            return plan
        plan = super()._search(eng)
        return None if plan is None else self._shorten(eng, plan)

    def _shorten(self, eng, plan: list) -> list:
        """Delete from ``plan`` everything the engine says it can do without.

        A beam plan wins but wanders -- it carries whole detours the beam took
        before it found the line, and a demonstration that wanders teaches
        wandering. Sweeping deletable blocks longest-first and keeping any cut
        that still wins takes the measured plans from 68 to 42 and 48 to 33
        moves. Every candidate is replayed on the interpreter, so a cut that
        only looks redundant is rejected: this can shorten a plan but never
        break one."""
        start = snapshot(eng)

        def win_index(seq):
            """Index of the step this sequence wins on, or None if it never
            does. ``check_win`` reads a per-step flag, so it has to be tested
            after every step rather than once at the end."""
            restore(eng, start)
            for i, direction in enumerate(seq):
                eng.step(direction)
                if eng.check_win():
                    return i
            return None

        for block in (6, 5, 4, 3, 2, 1):
            i = 0
            while i + block <= len(plan):
                candidate = plan[:i] + plan[i + block:]
                won = win_index(candidate)
                if won is not None:
                    plan = candidate[:won + 1]
                else:
                    i += 1
        restore(eng, start)
        return plan

    # -- the win -------------------------------------------------------------
    def goal_cells(self, grid) -> list[tuple[int, int]]:
        """Cells the player wins by standing on.

        ``late [player target] -> [player lie]`` then
        ``late [Player Lie no anything] -> [Player] win``, so a Target cell only
        wins when nothing else in the ``Anything`` group shares it. That test is
        load-bearing, not defensive: a pressed rotating wall on the Lie circuit
        settles as ``Wall + Hole + Target`` (``late [DoorOpen Metal Lie] ->
        [Wall nextshotblue target]``), which is a WALL that happens to carry a
        Target -- treating it as an exit would send the search at a surface the
        player can never stand on, and would also read as "this level has a
        portal device" on a level that has none.
        """
        out = []
        anyx = self.anything_ids
        tgt, lie = self.target, self.lie
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if tgt in cell:
                    if not (cell & anyx) - {tgt}:
                        out.append((r, c))
                elif lie in cell and not (cell & anyx):
                    out.append((r, c))       # the cake, on the last level
        return out

    def portals_on(self, grid) -> bool:
        """Whether the level grants the portal device. The Target carries the
        next shot's color (WeighedButton = orange, NextShotBlue = blue); a bare
        Target means ``[action ChellD] [Target shootable]`` never matches and
        ACTION is a dead key for the whole level."""
        return any(grid[r][c] & {self.wb, self.nsb} for r, c in self.goal_cells(grid))

    # -- state keys ----------------------------------------------------------
    def _key(self, eng) -> frozenset:
        # `PSPushExpert` keys on the pushed pieces plus the player, which is
        # exact for a pure sokoban. Here doors open and close, portals move
        # between walls, rotating walls become walls, buttons weigh down and
        # laser beams redraw every turn -- all of it reachable state that a
        # pieces-only key would alias. Fall back to the family's exact
        # whole-grid key.
        return PSExpert._key(self, eng)

    def _region_key(self, eng, region) -> tuple:
        # As `_key`, but with the player canonicalised to its walkable region
        # (the sokoban dedup that makes the macro search bite). Chell's facing
        # rides on which of the four Chell* objects is placed, and it drops out
        # with her -- every macro that cares sets the facing itself, on the
        # press immediately before ACTION.
        bg, players = self.bg_id, self.player_ids
        return (frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in cell
            if o != bg and o not in players
        ), region)

    # -- the moves a sokoban does not have -----------------------------------
    def extra_macros(self, eng, parent, walk_to, free) -> list[list[str]]:
        grid = eng.grid
        height, width = len(grid), len(grid[0])
        out: list[list[str]] = []
        buttons = [(r, c) for r, row in enumerate(grid)
                   for c, cl in enumerate(row) if self.button in cl]

        # Walk onto the exit (that IS the win) or onto a button (which holds its
        # doors open while she stands there -- long enough to shove a crate into
        # the doorway, which is how several levels are actually solved).
        for cell in self.goal_cells(grid):
            if cell in parent:
                out.append(walk_to(cell))
        for cell in buttons:
            if cell in parent:
                out.append(walk_to(cell))

        # Step into a portal mouth. Portals replace their wall, so the walk BFS
        # calls them blocked and would never enter one on its own.
        for r, row in enumerate(grid):
            for c, cl in enumerate(row):
                if not (cl & self.portal_ids):
                    continue
                for d, (dr, dc) in _DELTA.items():
                    stand = (r - dr, c - dc)
                    if stand in parent:
                        out.append(walk_to(stand) + [d])

        if self.portals_on(grid):
            out.extend(self._shot_macros(grid, parent, walk_to, free, buttons,
                                         height, width))
        return [m for m in out if m]

    def _shot_macros(self, grid, parent, walk_to, free, buttons,
                     height, width) -> list[list[str]]:
        """Every distinct portal shot reachable from here.

        Deduped by the REGION THE PORTAL'S MOUTH OPENS INTO, not by the wall it
        lands on. A portal's mouth faces back at the shooter, so it opens onto
        the last free cell the shot crossed; a mouth you cannot stand in front
        of is worthless, and two shots whose mouths open into the same region
        buy the same connectivity. That one change collapses ~20 near-identical
        shots per node to a handful and is what brings these levels into range.
        The origin-is-a-button flag stays in the key because firing from a
        button is a materially different outcome -- its doors are open for the
        shot.
        """
        out: list[list[str]] = []
        # Fire with the facing we already have, from where we already stand.
        out.append(["action"])

        # Step ONTO a button and fire in the same breath, from every side we can
        # reach it from. Emitted blind, WITHOUT tracing the ray: the shot leaves
        # one turn after the step, by which time standing on the button has
        # opened its doors -- so the ray we could trace here (doors shut) would
        # say the shot dies in the doorway. This is the shot level 5 turns on:
        # fire down the corridor the button itself just opened. It also has to
        # be one macro rather than "walk onto the button" then "fire", because
        # the region key drops Chell's facing, so the four ways of standing on a
        # button dedup to one and only one facing would survive.
        for (r, c) in buttons:
            for d, (dr, dc) in _DELTA.items():
                stand = (r - dr, c - dc)
                if stand in parent:
                    out.append(walk_to(stand) + [d, "action"])

        comp = self._components(free, height, width)
        shots: dict[tuple, list[str]] = {}

        def offer(mouth, d, on_button, macro):
            key = (comp[mouth[0]][mouth[1]], d, on_button)
            if key not in shots or len(macro) < len(shots[key]):
                shots[key] = macro

        # (a) fire on the facing the walk arrived with -- no direction press, so
        # she does not step off the cell she walked to.
        for cell, par in parent.items():
            if par is None:
                continue
            d = par[1]
            mouth = self._shot_mouth(grid, cell, d, free, height, width)
            if mouth is not None:
                offer(mouth, d, self.button in grid[cell[0]][cell[1]],
                      walk_to(cell) + ["action"])

        # (b) turn (or step) to face d first, then fire.
        for cell in parent:
            r, c = cell
            for d, (dr, dc) in _DELTA.items():
                nr, nc = r + dr, c + dc
                inside = 0 <= nr < height and 0 <= nc < width
                if inside and free[nr][nc]:
                    origin = (nr, nc)               # the press walks her there
                elif not inside or (grid[nr][nc] & self.turn_only_ids):
                    origin = (r, c)                 # the press only turns her
                else:
                    continue                        # a push, a hole or a beam
                mouth = self._shot_mouth(grid, origin, d, free, height, width)
                if mouth is not None:
                    offer(mouth, d, self.button in grid[origin[0]][origin[1]],
                          walk_to(cell) + [d, "action"])

        out.extend(shots.values())
        return out

    @staticmethod
    def _components(free, height, width) -> list[list[int]]:
        """Label the connected components of the walkable map."""
        comp = [[-1] * width for _ in range(height)]
        label = 0
        for r0 in range(height):
            for c0 in range(width):
                if not free[r0][c0] or comp[r0][c0] >= 0:
                    continue
                comp[r0][c0] = label
                stack = [(r0, c0)]
                while stack:
                    r, c = stack.pop()
                    for dr, dc in _DELTA.values():
                        nr, nc = r + dr, c + dc
                        if (0 <= nr < height and 0 <= nc < width
                                and free[nr][nc] and comp[nr][nc] < 0):
                            comp[nr][nc] = label
                            stack.append((nr, nc))
                label += 1
        return comp

    def _shot_mouth(self, grid, origin, d, free, height, width):
        """The cell a portal fired from ``origin`` toward ``d`` would open onto,
        or None if the shot fizzles or lands somewhere unusable."""
        ocell = grid[origin[0]][origin[1]]
        if self.nsb in ocell and self.target not in ocell:
            return None                             # cannot fire from a fizzler
        hit = self._trace_shot(grid, origin, d, height, width)
        if hit is None:
            return None
        dr, dc = _DELTA[d]
        mouth = (hit[0] - dr, hit[1] - dc)
        if not (0 <= mouth[0] < height and 0 <= mouth[1] < width):
            return None
        if not free[mouth[0]][mouth[1]]:
            return None                             # nowhere to stand to use it
        return mouth

    def _trace_shot(self, grid, origin, d, height, width):
        """Where a portal fired from ``origin`` toward ``d`` lands, as
        ``(row, col, direction)``, or None if the shot fizzles.

        The shot flies over anything that is not Solid, not already a portal and
        not a fizzler, and forms a portal on the first bare Wall it reaches; a
        wall that already carries a portal eats the shot instead."""
        dr, dc = _DELTA[d]
        r, c = origin[0] + dr, origin[1] + dc
        while 0 <= r < height and 0 <= c < width:
            cell = grid[r][c]
            if cell & self.shot_stop_ids:
                if self.wall in cell and not (cell & self.portal_ids):
                    return (r, c, d)
                return None
            if self.nsb in cell and self.target not in cell:
                return None                          # the fizzler eats it
            r += dr
            c += dc
        return None

    # -- heuristic -----------------------------------------------------------
    def _enter_cost(self, cell) -> int | None:
        """Estimated cost of getting the player THROUGH ``cell``, or None for
        permanent scenery she can never cross.

        A free cell costs 1, so as soon as the exit is walkable the estimate is
        the exact remaining walk -- which is what makes the search converge once
        the last obstacle is gone. Everything clearable costs roughly what
        clearing it costs in moves, so a state that has opened the door sorts
        ahead of one that has not. A single flat "blocked" penalty gives no such
        gradient and the search just wanders."""
        if cell & self.impassable_ids:
            return None
        if cell & self.push_ids:
            return 6                       # push it out of the way
        if self.door in cell:
            return 6                       # find and hold its button
        if cell & self.laser_ids:
            return 5                       # block the beam
        if self.hole in cell:
            return 8                       # bridge it, or portal past it
        return 1

    def _warp_edges(self, grid, height, width) -> dict:
        """The teleport shortcut the open portal pair provides, as
        ``mouth -> other mouth``.

        Without this the distance is computed on the walls-only graph, and a
        player who has just come out of a portal scores UNREACHABLE -- exactly
        the states the search needs to like. A portal's mouth is the cell it
        spits into, which is the same side the shot was fired from."""
        spit = self.portal_spit
        mouths = []
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                for obj in cell & self.portal_ids:
                    dr, dc = spit[obj]
                    mouth = (r + dr, c + dc)
                    if 0 <= mouth[0] < height and 0 <= mouth[1] < width:
                        mouths.append(mouth)
        if len(mouths) != 2:
            return {}                    # no pair open yet, or a broken state
        a, b = mouths
        return {a: b, b: a}

    def heuristic(self, eng) -> int:
        """Estimated moves to the exit: a Dijkstra over `_enter_cost`, plus the
        portal pair's teleport edge.

        Not admissible -- it under-counts what clearing an obstacle really costs
        and over-counts elsewhere -- and it is weighted on top of that; see the
        class docstring."""
        grid = eng.grid
        height, width = len(grid), len(grid[0])
        player = None
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if cell & self.player_ids:
                    player = (r, c)
                    break
            if player is not None:
                break
        if player is None:
            return DEAD                               # dead: prune this branch
        goals = set(self.goal_cells(grid))
        if not goals:
            return 0
        warp = self._warp_edges(grid, height, width)
        best = {player: 0}
        queue = [(0, player)]
        while queue:
            dist, cell = heapq.heappop(queue)
            if cell in goals:
                return dist
            if dist > best.get(cell, 1 << 30):
                continue
            r, c = cell
            neighbours = [(r + dr, c + dc, None) for dr, dc in _DELTA.values()]
            if cell in warp:
                neighbours.append((warp[cell][0], warp[cell][1], 1))
            for nr, nc, fixed in neighbours:
                if not (0 <= nr < height and 0 <= nc < width):
                    continue
                step = fixed if fixed is not None else self._enter_cost(grid[nr][nc])
                if step is None:
                    continue
                nd = dist + step
                if nd < best.get((nr, nc), 1 << 30):
                    best[(nr, nc)] = nd
                    heapq.heappush(queue, (nd, (nr, nc)))
        return UNREACHABLE


class ApertureSolver(PSAStarSolver):
    game_id = "puzzlescript_aperture_science_sokoban_testing_initiative"
    game_name = GAME_NAME
    expert_cls = ApertureExpert

    #: Level 5 is a deadlock (unwinnable as this interpreter runs it) and 7-21
    #: are out of the search's budget; see "Solvable levels" in the module
    #: docstring. Skipped up front so startup discovery does not spend the node
    #: budget on them at every run -- for level 5 that would be the full beam,
    #: ~50 minutes, to rediscover that it cannot be done.
    skip_levels = frozenset({5} | set(range(7, 22)))

    #: A*'s macro budget. Deliberately small: A* is the FIRST strategy tried on
    #: every level, and on the levels it cannot solve it should establish that
    #: in a couple of minutes and hand over to the beam rather than grind. The
    #: levels it does solve land inside ~50 macros.
    node_cap = 3_000
    weight = 4
    #: The adapter GAME_OVERs at 200 actions per level; the longest plan here is
    #: 42 moves, so even the exploration prefix in front of it has room.
    max_steps = 200

    _LETTER_TO_DIR = {"u": "up", "d": "down", "l": "left", "r": "right",
                      "x": "action"}

    def prepare_expert(self, game, expert) -> None:
        """Seed the expert's memo with `_BEAM_PLANS`, keeping only plans that
        still win.

        `PSExpert.plan` memoises by ``(level, state key)`` and, with
        ``recovery_mode = "reset"``, is only ever asked about a level's START
        state -- so one seeded entry per level skips the search entirely. A plan
        that no longer wins is left out and that level falls through to a real
        search, which is why a stale entry here can cost time but cannot
        corrupt a demonstration."""
        for level, letters in sorted(_BEAM_PLANS.items()):
            plan = [self._LETTER_TO_DIR[ch] for ch in letters]
            game.set_level(level)
            eng = game._engine
            key = (level, expert._key(eng))
            start = snapshot(eng)
            won = False
            for direction in plan:
                eng.step(direction)
                if eng.check_win():
                    won = True
                    break
            restore(eng, start)
            if won:
                expert.cache[key] = plan
            else:
                print(f"  [warn] the stored plan for level {level} no longer "
                      f"wins -- re-searching it (this takes minutes)")


if __name__ == "__main__":
    sys.exit(ApertureSolver.main())
