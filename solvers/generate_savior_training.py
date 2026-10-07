"""Generate Phase-1 training data for the PuzzleScript game ps:savior
("Savior" by Jack Lance).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix, the disk plan cache and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: the mechanic notes, the macro set the search branches on,
the goal heuristic, the interpreter speed-up that makes the search finish at all,
and the two rendering fixes the shipped game needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_savior",
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
presented view, so replaying the recorded actions reproduces the recorded frames
exactly. Every expert step also carries the full set of equally-optimal presses.

The game
--------
A sokoban whose one idea is a SAVE STATE. Push boxes, walk onto the exit
(``all target on player``) -- and press ACTION on a black diamond to bookmark
the board, ACTION anywhere else to roll back to the bookmark.

  * **Crates** push one at a time, ordinary sokoban. **Sticky blocks** push as
    one rigid blob (``[moving sticky | sticky] -> [moving sticky | moving
    sticky]``) and the whole shove is CANCELLED if any part of the blob would
    hit a wall or a shut door.
  * **Buttons and doors.** A door group toggles between shut (``wallone``,
    which blocks) and open (``walloneno``, which does not) every time a button
    of that group is pressed AND every time it is released -- so a door group
    is flipped exactly while something heavy sits on one of its buttons. The
    player is heavy, and so is a crate: parking a crate on a button is how a
    door is held open while the player is elsewhere. One special case matters:
    ``late [walloneno toggle no heavy no to] -> [wallone]`` -- a door with
    something standing IN it cannot shut, so stepping into a doorway before
    letting the button go jams that one cell open until you leave it.

The save state, and why the whole level is 25 cells wide
--------------------------------------------------------
Every level is a 12x25 grid: the 12x12 play area on the left (tagged ``from``),
a dead column, and a 12x12 MIRROR on the right (tagged ``to``). ``flickscreen
12x12`` means the agent only ever sees the play area. Saving copies every
savable object -- crates, sticky blocks, doors, buttons, and **the player** --
from the play area into the mirror 13 columns right; resetting copies them back.

Three consequences the expert has to respect:

  * **Saving clones the player.** From the first save on there are two player
    sprites, and the one in the mirror is frozen (``[moving player to] ->
    [stationary player to]``). `SaviorExpert.live_player` picks a live one;
    without it the search would build its walks from a cell no press can leave.
    The adapter's camera makes the same choice for the same reason.
  * **Save/reset is per ZONE, not per level.** The mirror is painted in up to
    three colours (``one`` / ``two`` / ``three``), and that painting is copied
    back onto the play area at level start -- which is why the floor is tinted
    green / blue / yellow. A saver bookmarks only its own zone's cells, and a
    reset only restores its own zone's cells, so the halves of a level roll back
    independently. That is the point of the whole design: level 5 has a column
    of doors running through both zones, and putting the two zones' doors out of
    phase is the only way to have that column open at one row and shut at
    another, which is the only way through.
  * **A rollback fires from anywhere, and it can DUPLICATE the player.** The
    author guarded the door rules with ``no to`` but not the reset rules, so
    ``[action player two][saver saved two]`` is matched by the FROZEN CLONE
    sitting in the mirror's blue half -- which means that once a zone is
    bookmarked, ACTION rolls it back no matter where the player is standing.
    And the sweep that clears the zone before restoring it
    (``[resetting two][savable two from] -> [resetting two][two from]``) only
    touches that zone's cells, so a rollback pressed OUTSIDE the zone leaves the
    live player where it is and restores a second one on the saver. Both then
    move on every press. This is not modelled around; it is played with (see
    `SaviorExpert.extra_macros`), because level 5's solution needs it.

Pressed inside its own zone a rollback teleports the player instead, because the
player is savable and saving requires standing on the saver: the bookmark for
zone Z always holds a player on Z's saver. That makes the ordinary rollback a
"return to the diamond, keeping every other zone's progress" move, which is how
the multi-part levels are meant to be played.

The expert
----------
`PSPushExpert`: A* over ``walk to a cell, then press`` MACROS on the real
interpreter, with ``g`` counting primitive moves so plans are shortest in the
units the agent pays. A primitive search does not work here -- level 0's player
starts in a six-cell pocket in a lattice of twenty crates, so essentially every
press that matters is a push and a primitive A* re-derives the same walks
forever. Besides the usual walk-and-push, `SaviorExpert.extra_macros` adds the
moves this game has that are not pushes:

  * **walk onto the target** -- the win is REACHING somewhere, so without it the
    macro search closes with the goal one step away;
  * **walk onto a saver, ACTION** -- bookmark that zone;
  * **ACTION** -- roll every bookmarked zone back. Three places to press it are
    materially different (in the zone, outside it, on a button), so there is a
    macro for each; see `SaviorExpert.extra_macros`;
  * **walk onto a button, then step off in each direction** -- pressing a button
    only flips the doors while you stand on it, so the single useful thing to do
    from a button is step straight into a door that just opened (which then jams
    it). Plain walks stay inert this way, which is what the walk BFS assumes.

Optimal-action sets
-------------------
``exact_optsets``: at every plan step each of the five presses is MEASURED by
re-solving from it, and the label is every press that still finishes in the
moves remaining. Inferring them instead (`PSPushExpert.annotate_walks`) is wrong
for this game twice over -- shoving a crate aside is frequently just how you
walk here, so a push genuinely ties with a sibling push, and the annotator's
push/walk classification reads the board through a single player position that
a save has just made ambiguous.

The measurement is bounded (`SaviorExpert.optset_node_cap`), so it can only miss
a tie, never invent one, and every step carries a label either way. Measured
density on the levels whose searches are seconds is 1.09-1.29 equally-optimal
presses per step: these plans are mostly forced, which is what a board where the
walls are made of crates looks like.

Making the interpreter fast enough
----------------------------------
Nine of Savior's rules are the level-start plumbing that paints the ``from`` /
``to`` regions and copies the zone colours across the mirror. They are ellipsis
rules (``[from | | ... | from]``) over a 25-wide grid, and they cost **91% of
every interpreter step** -- 0.2s a press, which is minutes per A* node. They are
also no-ops after the level has been seated: nothing in the game creates or
destroys a ``from``, ``to``, ``one``, ``two``, ``three`` or ``mark`` object, and
none of them is savable. `_install_lean_rules` therefore seats each level with
the full rule list and drops those nine immediately afterwards, which makes a
press ~30x cheaper. ``--fuzz`` is the check that this is a speed-up and not a
behaviour change: it replays the same random presses (ACTION included, so saves
and resets are exercised) on a full and a lean interpreter and compares the
grids cell for cell after every one.

Rendering fixes (in ``data/puzzlescript_games/Savior.txt``)
-----------------------------------------------------------
``--audit`` renders every cell COMPOSITION a level can show and asserts pairwise
distinctness. It found two states that were not in the frame at all:

  * **"this zone is bookmarked" was invisible.** ``onesaved`` was ``green`` on a
    ``lightgreen`` zone and ``threesaved`` was ``#CCCC00`` on a ``yellow`` one --
    both pairs collapse onto one ARC index -- and each was a single centre pixel,
    which the player's own sprite covers when it stands on the diamond. All three
    are now ``red``, drawn on the cell's corners and edge midpoints where nothing
    that can share the cell covers them.
  * **An open door under a crate was invisible.** ``walloneno`` painted only the
    four corners, and a crate's sprite is opaque except for a plus-shaped hole
    through its middle -- so a crate parked in an open doorway (the way a door is
    held open) hid the fact that the doorway was there. Both open doors now also
    paint their middle row, which is exactly what shows through that hole.

The two pairs ``--audit`` still reports are not defects: an opaque Wall hides the
zone tint under it, and a pressed button with nothing on it cannot exist (a late
rule un-presses it the same turn), so its sprite is never seen bare.

What it solves
--------------
Eight of the nine levels, every plan engine-verified and proved shortest under
`SaviorExpert`'s search (0: 14 presses, 2: 22, 3: 23, 4: 32, 5: 19, 6: 20,
7: 30, 8: 28). Two of those lengths have an independent witness: level 0 matches
a plain primitive A* and level 5 matches an exhaustive breadth-first sweep of its
whole reachable space, which is worth having because level 5 is the level whose
solution turns on the rollback quirk above.

Level 1 is skipped -- see `SaviorSolver.skip_levels`.

The searches are the whole cost of generation and they are seed-independent, so
they are paid once and kept in ``data/savior_plans.json``; level 8 alone is a few
hours of interpreter the first time and free after that.

Recovery
--------
``recovery_mode = "reset"`` (the `PSAStarSolver` default). Pushing a crate into a
corner is not undoable, so a perturbed board cannot be re-planned from in
general; the episode-wide epsilon prefix explores freely and ONE RESET restores
the level, from which the cached plan replays a guaranteed win. Note this is the
harness's RESET button, not the game's own ACTION rollback -- the latter is a
move the expert plans with, and it shows up inside the plans.

Usage
-----
    python solvers/generate_savior_training.py --episodes 1600
        # -> data/training_multi_level/puzzlescript_savior, then
        # python3 convert_training_data.py --game puzzlescript_savior --delete

    python solvers/generate_savior_training.py --plans     # per-level report
    python solvers/generate_savior_training.py --verify    # labels + replay
    python solvers/generate_savior_training.py --fuzz      # lean == full rules
    python solvers/generate_savior_training.py --audit     # sprite distinctness
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                      # noqa: E402
from adapters.puzzlescript_adapter import _render_frame           # noqa: E402
from solvers.base_solver import _ID_TO_GAMEACTION                 # noqa: E402
from solvers.common.ps_astar import (                             # noqa: E402
    DIRECTIONS, PSAStarSolver, PSPushExpert, _DELTA, restore, snapshot,
)

#: Heuristic charge for a board the player can no longer reach the target from.
#: Large enough to sink the node, finite so the search stays complete.
UNREACHABLE = 10_000

#: Objects the level-start plumbing is made of. A rule that mentions nothing
#: else is one of the nine `_install_lean_rules` drops: they paint the ``from``
#: and ``to`` halves of the grid, copy the zone colours back from the mirror and
#: clear the legend's marker sprites, all of which happens once, at level start.
#: No rule anywhere in the game creates or destroys one of these objects, and
#: none of them is ``savable``, so none of them can change again afterwards.
_STATIC_OBJECTS = frozenset({
    "from", "to", "one", "two", "three",
    "onemark", "twomark", "threemark", "frommark", "mark",
})


def _rule_objects(rule) -> set:
    """Every object (or or-group) name a parsed rule mentions, either side."""
    names = set()
    for groups in (rule.groups_lhs, rule.groups_rhs):
        for group in groups:
            for cell in group:
                for item in cell:
                    if isinstance(item, tuple) and len(item) == 3 and item[1]:
                        names.add(item[1])
    return names


def _install_lean_rules(game) -> None:
    """Make the interpreter ~30x faster by retiring the level-start rules once
    they have run.

    See the module docstring: the nine ellipsis rules that paint the regions and
    the zone colours are 91% of the cost of a press and cannot fire again. They
    still have to RUN, though -- ``run_rules_on_level_start`` is what puts the
    zone tint on the floor in the first place -- so rather than editing the game
    file this puts the full list back for the level seating and takes it away
    again straight after. ``--fuzz`` compares the two interpreters press for
    press.

    The wrap goes on ``_do_reset``, which is the one place a level is actually
    loaded: ``set_level``, ``level_reset`` (what a RESET press runs) and
    ``full_reset`` all funnel through it, and a level re-seated with the lean
    list would come up with no zone tint on the floor at all.

    ``run_rules_on_level_start`` does not always finish the fill, which is why
    the seating is followed by `_settle`: on level 7 one wall cell inside the
    blue zone is still uncoloured when the level comes up and the real
    interpreter only paints it on the FIRST PRESS. Dropping the rules before
    that press is what the first version of this did, and ``--fuzz`` caught it
    -- one cell, on one level, forever after."""
    parsed = game._game
    full = list(parsed.rules)
    lean = [r for r in full
            if not (_rule_objects(r) and _rule_objects(r) <= _STATIC_OBJECTS)]
    original = game._do_reset

    def _do_reset():
        parsed.rules = full
        try:
            result = original()
            _settle(game._engine)
            return result
        finally:
            parsed.rules = lean

    game._do_reset = _do_reset
    parsed.rules = lean          # the level seated at construction used `full`


def _settle(eng, limit: int = 20) -> None:
    """Run the main rules with NO forces until the grid stops changing.

    That is the level-start propagation reaching the fixpoint it is dropped at.
    Only rules that need no movement can fire in such a pass -- everything else
    in Savior is gated on ``>``, ``moving`` or ``action`` -- and the two that
    can (the fills, and ``[one saved] -> [one onesaved saved]``) are idempotent,
    so this settles and then does nothing. In practice it is one no-op pass on
    eight levels and one cell on the ninth."""
    for _ in range(limit):
        before = [[set(cell) for cell in row] for row in eng.grid]
        eng._apply_rules_with_forces("right", {}, False)
        if eng.grid == before:
            return


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class SaviorExpert(PSPushExpert):
    """Macro A* over the real interpreter, with the save state as a macro.

    ``g`` counts primitive presses and the heuristic is admissible (see
    `heuristic`), so with ``exact_goal_test`` every plan here is PROVED shortest
    up to this class's one documented approximation: two boards whose pieces
    agree and whose player stands anywhere in the same walkable region are
    merged, which can mis-cost a plan by at most that region's diameter.
    """

    #: ACTION is a move in this game -- it is the save and the rollback -- so it
    #: belongs in the alphabet `optimal_sets` measures over. (The macro search
    #: itself branches on `extra_macros`, not on this list.)
    directions = DIRECTIONS

    pushable_names = ("crate", "sticky")
    #: ``wallnum`` is a SHUT door; ``wallno`` (the open form) lives on the
    #: button layer and blocks nothing, so it is deliberately absent.
    blocker_names = ("wall", "wallnum")

    exact_goal_test = True
    exact_optsets = True

    #: Node budget for ONE candidate press inside `optimal_sets`, well under
    #: `SaviorSolver.node_cap`, which is sized for the level-start searches.
    #:
    #: Measuring a label re-solves the whole level from one press in, and on
    #: level 8 the level-start search alone is hours: unbounded, labelling it
    #: would cost days. A capped sub-search can only ever OVERSTATE what a press
    #: costs (it returns the best plan it reached, or nothing), so a press it
    #: cannot price is left out of the set rather than wrongly put in -- and if
    #: the press the expert actually took is the one it cannot price,
    #: `optimal_sets` falls back to labelling that press alone. Every label that
    #: ships is still a measured tie; some ties on the deep levels are simply
    #: not found.
    optset_node_cap: int = 1_000

    def _exact(self, eng) -> "int | None":
        """`PSPushExpert._exact` under ``optset_node_cap``."""
        full, self.node_cap = self.node_cap, self.optset_node_cap
        try:
            return super()._exact(eng)
        finally:
            self.node_cap = full

    #: These searches are the whole cost of generation and they are
    #: seed-independent, so every shard of `parallelize_generator` would
    #: otherwise re-derive all of them.
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "savior_plans.json")

    def setup(self) -> None:
        super().setup()
        resolve = self.g.resolve_object_name
        ids = lambda name: set(resolve(name))            # noqa: E731
        self.from_ids = ids("from")
        self.to_ids = ids("to")
        self.wall_ids = ids("wall")
        self.target_ids = ids("target")
        self.saver_ids = ids("saver")
        self.saved_ids = ids("saved")
        self.button_ids = ids("button")
        self.zone_ids = {z: ids(z) for z in ("one", "two", "three")}
        # Everything a press can change. Doors, buttons, the bookmark markers
        # and the stuck ``toggle`` of a jammed doorway are all state the search
        # must tell apart, and the mirror's frozen copies of the pieces are the
        # bookmark itself -- drop any of them from the key and two boards that
        # roll back to different places would look like one.
        self.state_ids = set()
        for name in ("player", "crate", "sticky", "wallnum", "wallno",
                     "button", "saved", "toggle"):
            self.state_ids |= ids(name)
        self._static_cache: dict = {}
        self._static: tuple = (frozenset(), frozenset())

    # -- reading the board ----------------------------------------------------
    def live_player(self, eng, positions):
        """A player OUTSIDE the mirror: the frozen clone a save leaves behind is
        in a ``to`` cell and can never move again.

        There can be more than one live player -- a rollback pressed outside its
        own zone restores a second one without sweeping the first away (see the
        module docstring) -- and from then on every press moves all of them. The
        first in scan order is taken as the one the macros are built around,
        which is an approximation and not a claim: the others still move, and
        the walk BFS does not know it. It costs only completeness, never
        soundness, because every macro is executed on the interpreter and every
        plan that comes back is a win it actually reached."""
        for cell in positions:
            if not (eng.grid[cell[0]][cell[1]] & self.to_ids):
                return cell
        return None

    def _live_players(self, eng) -> list:
        """Every player sprite outside the mirror, in scan order."""
        return [(r, c)
                for r, row in enumerate(eng.grid)
                for c, cell in enumerate(row)
                if (cell & self.player_ids) and not (cell & self.to_ids)]

    def _refine_free(self, eng, free) -> None:
        """Confine the player to the play area.

        The mirror is not scenery the walk BFS should route through: it is
        reachable on paper (the border rows are open floor) and a player that
        steps into a ``to`` cell is frozen there for good, so every walk through
        it is a walk to nowhere. No level actually leaves a gap into it -- this
        is what makes that a fact of the search rather than of the level maps."""
        from_ids = self.from_ids
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not (cell & from_ids):
                    free[r][c] = False

    def dead(self, eng) -> bool:
        """No live player left: it walked into the mirror and froze."""
        return not self._live_players(eng)

    def _region_key(self, eng, region) -> tuple:
        """Every mutable object, plus the region the player can walk in -- with
        the player's own sprite dropped while there is exactly one of it.

        Dropping it is the standard sokoban canonicalisation the base class
        does (any cell of the same region admits the same futures); it is
        withheld once a rollback has duplicated the player, because the walk
        region is then only ONE of their regions and merging on it would be
        merging boards that do not behave alike.

        Everything else in the key is there because the base class's -- the
        PUSHABLES alone -- would alias half this game: whether a door is open,
        whether a button is held, which zones are bookmarked and what the mirror
        is holding are all things a press changes and the next press depends on.
        """
        live = self._live_players(eng)
        drop = live[0] if len(live) == 1 else None
        state = self.state_ids
        player = self.player_ids
        return (frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & state)
            if not (o in player and (r, c) == drop)
        ), region)

    # -- macros ---------------------------------------------------------------
    def extra_macros(self, eng, parent, walk_to, free) -> list[list[str]]:
        """The presses that are not pushes: reach the exit, bookmark a zone,
        roll a zone back, and use a button.

        A pure sokoban needs none of these -- there, the win IS a push outcome.
        Here the win is standing somewhere, and the two ACTION moves are the
        game's whole idea, so a macro search without them closes with the board
        untouched on the levels that need a rollback."""
        grid = eng.grid
        out: list[list[str]] = []
        bookmarked = set()
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if not (cell & self.from_ids):
                    continue                       # the mirror is not played on
                if (cell & self.saver_ids) and (cell & self.saved_ids):
                    for zone, zone_ids in self.zone_ids.items():
                        if cell & zone_ids:
                            bookmarked.add(zone)
                if (r, c) not in parent:
                    continue
                if cell & self.target_ids:
                    out.append(walk_to((r, c)))
                if cell & self.saver_ids:
                    out.append(walk_to((r, c)) + ["action"])
                if cell & self.button_ids:
                    # Standing on a button flips its doors and stepping off
                    # flips them back, so the one durable thing a button buys
                    # is the step INTO a door that just opened -- which jams
                    # that cell open (``no heavy``) as the button is released.
                    walk = walk_to((r, c))
                    out.extend(walk + [d] for d in _DELTA)
        if bookmarked:
            # A rollback fires from ANYWHERE (see `live_player` -- the mirror's
            # clone is what the rule matches), so where it is pressed changes
            # only what the same rollback does to the player and to the doors:
            #
            #   * inside a bookmarked zone -- the player is one of that zone's
            #     savables, so it is swept up and comes back on the saver;
            #   * outside every bookmarked zone -- nothing sweeps the player up,
            #     so the restored copy is a SECOND player and both move from
            #     then on;
            #   * standing on a BUTTON -- the doors are flipped while it is held,
            #     the rollback overwrites the bookmarked zone's doors with their
            #     unflipped copies, and the flip comes off every OTHER door on
            #     the next move, when the player finally steps off. That is the
            #     only move in the game that puts two zones' doors out of phase
            #     with each other, and level 5 cannot be won without it.
            out.append(["action"])                    # from right here
            for cell_rc in parent:
                cell = grid[cell_rc[0]][cell_rc[1]]
                if (cell & self.button_ids) and not (cell & self.saver_ids):
                    out.append(walk_to(cell_rc) + ["action"])
            for zone in bookmarked:
                zone_ids = self.zone_ids[zone]
                best = None
                for cell_rc in parent:
                    cell = grid[cell_rc[0]][cell_rc[1]]
                    if (not (cell & zone_ids) or (cell & self.saver_ids)
                            or (cell & self.button_ids)):
                        continue
                    walk = walk_to(cell_rc)
                    if best is None or len(walk) < len(best):
                        best = walk
                if best is not None:
                    out.append(best + ["action"])
        return out

    # -- heuristic ------------------------------------------------------------
    def plan(self, eng, level: int | None = None):
        """Bind this level's static geometry, then plan as usual.

        Walls and targets are the only things `heuristic` needs that no rule in
        Savior ever creates or destroys, so they are read once per layout here
        rather than on every node."""
        self._static = self._read_static(eng)
        return super().plan(eng, level)

    def _read_static(self, eng) -> tuple:
        """``(walls, targets)`` as cell sets, memoized by layout. ``walls``
        holds the mirror and the dead column too -- they are off the board as
        far as the player is concerned."""
        walls = frozenset(
            (r, c)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            if not (cell & self.from_ids) or (cell & self.wall_ids))
        targets = frozenset(
            (r, c)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            if cell & self.target_ids)
        return self._static_cache.setdefault((walls, targets), (walls, targets))

    def heuristic(self, eng) -> int:
        """Presses for the nearest live player to reach the target, in a
        relaxation that keeps only the two facts a press can never undo.

        The relaxation: walls block, and a cell holding a crate or a sticky
        block may be walked into only if the cell straight on from it is not a
        wall -- i.e. only if that piece has somewhere to be shoved. Everything
        else is treated as free. Doors are ignored (a shut one can always be
        opened later, so counting it would not be a lower bound), other pieces
        are ignored when testing a shove, and a sticky blob is treated as if its
        cells moved one at a time. Each of those makes the estimate smaller, so
        it stays admissible; taken together they still capture what actually
        costs presses on these boards, which is the detour around a crate that
        has nowhere to go.

        This runs on every node and it is cheap -- one BFS over ~150 cells --
        next to the interpreter step the node is really paying for. The earlier
        version ignored the pieces entirely and was flat across most of the
        search: on the crate-lattice levels it left the macro A* wandering.

        The NEAREST player, not the one the macros are built around: the win is
        ``all target on player``, so any of them standing on it ends the level,
        and a press moves them all by at most one cell.
        """
        walls, targets = self._static
        starts = self._live_players(eng)
        if not starts or not targets:
            return UNREACHABLE
        grid = eng.grid
        h, w = eng.height, eng.width
        pieces = self.push_ids
        seen = set(starts)
        queue = deque((cell, 0) for cell in starts)
        while queue:
            (r, c), dist = queue.popleft()
            if (r, c) in targets:
                return dist
            for dr, dc in _DELTA.values():
                nr, nc = r + dr, c + dc
                if (nr, nc) in seen or (nr, nc) in walls:
                    continue
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if grid[nr][nc] & pieces:
                    br, bc = nr + dr, nc + dc         # where the piece would go
                    if (not (0 <= br < h and 0 <= bc < w)
                            or (grid[br][bc] & self.wall_ids)):
                        continue
                seen.add((nr, nc))
                queue.append(((nr, nc), dist + 1))
        return UNREACHABLE


# ---------------------------------------------------------------------------
# The generator
# ---------------------------------------------------------------------------

class SaviorSolver(PSAStarSolver):
    """`PSAStarSolver` for ps:savior."""

    game_id = "puzzlescript_savior"
    game_name = "Savior"
    game_module_id = "ps:savior"
    expert_cls = SaviorExpert

    #: Sized from what the levels actually needed: 7 and 8 are unreachable below
    #: ~120k nodes and land at ~180k and ~350k. The whole cost is paid once, at
    #: seed 0, and kept in ``data/savior_plans.json``.
    node_cap = 400_000
    max_steps = 300

    #: Level 1 -- the two crate rooms joined by a corridor -- is the one board
    #: this expert does not reach. It is a plain sokoban with thirteen crates and
    #: no doors, so the macro branching is ~52 wide with a heuristic that cannot
    #: see which shove strands which crate, and 400k nodes and four hours of
    #: interpreter did not close it. Skipped up front so a generation run does
    #: not pay for that discovery at every startup. `discover_solvable` would
    #: find the other eight on its own; naming them here is not needed and is
    #: deliberately not done, so a search improvement picks them up for free.
    skip_levels = frozenset({1})

    def make_game(self, seed: int):
        game = super().make_game(seed)
        _install_lean_rules(game)
        return game


# ---------------------------------------------------------------------------
# Checks (none of these write anything under data/)
# ---------------------------------------------------------------------------

def _levels():
    """A solver with its adapter, expert and solvable-level set warmed up."""
    solver = SaviorSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _report() -> int:
    """Per level: the plan, whether it uses the save state, and the engine's
    verdict on replaying it."""
    _solver, game, expert, solvable = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        if level not in solvable:
            print(f"  L{level}: no plan within {SaviorSolver.node_cap} nodes")
            continue
        plan = expert.plan(eng, level)
        for direction in plan:
            eng.step(direction)
        won = eng.check_win()
        bad += not won
        ties = sum(len(s) for s in plan.optsets) / len(plan)
        print(f"  L{level}: {len(plan):3d} presses  win={won}  "
              f"{plan.count('action')} ACTION  "
              f"{ties:.2f} optimal presses/step")
    print(f"solved {len(solvable)}/{game.n_levels} levels"
          + ("" if not bad else f" -- {bad} do not replay to a win"))
    return 1 if bad or not solvable else 0


def _verify() -> int:
    """Two checks the recorded corpus stands on.

    1. **The labels.** Every press a label calls optimal is pressed on the
       interpreter, solved from, and required to finish in exactly the presses
       remaining -- and the step the expert actually took has to be in its own
       set. This checks that every label SHIPPED is genuinely optimal, which is
       what a wrong label would break; it deliberately does not re-derive the
       sets from scratch (that is `PSPushExpert.optimal_sets`, and repeating it
       here would cost a search for each of the five presses at every step of
       every plan without checking anything new).
    2. **The recording.** Generate an episode and replay its recorded SCREEN
       actions through ``perform_action`` on a fresh adapter, asserting every
       frame matches the taped one and the level ends in ``GameState.WIN``.
       This is what catches a rotation-remap mistake, which nothing else does:
       a wrongly-remapped press still wins on the engine, it just is not the
       button the agent would have to press.
    """
    _solver, game, expert, solvable = _levels()
    eng = game._engine
    bad = 0

    for level in solvable:
        game.set_level(level)
        plan = expert.plan(eng, level)
        expert._exact_dist = {}
        for i, press in enumerate(plan):
            remaining = len(plan) - i
            if press not in plan.optsets[i]:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            here = snapshot(eng)
            for alt in plan.optsets[i]:
                eng.step(alt)
                if eng.check_win():
                    cost = 1
                else:
                    sub = expert._exact(eng)
                    cost = None if sub is None else 1 + sub
                restore(eng, here)
                if cost != remaining:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but it "
                          f"finishes in {cost} against {remaining}")
            eng.step(press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the plan does not replay to a win")
        print(f"  L{level}: {len(plan)} steps, "
              f"{sum(len(s) for s in plan.optsets)} labels re-measured")

    for seed in (0, 3):
        solver = SaviorSolver()
        ok, levels = solver.solve_episode(seed, explore=False)
        if not ok:
            bad += 1
            print(f"  seed {seed}: solve_episode reported failure")
        replay = solver.make_game(seed)
        for record in levels:
            replay.set_level(record["level_id"])
            frames = [np.asarray(f) for f in record["observations"]]
            if not np.array_equal(np.asarray(replay._current_frame), frames[0]):
                bad += 1
                print(f"  seed {seed} L{record['level_id']}: frame 0 differs")
            for i, act in enumerate(record["actions"][1:], start=1):
                result = replay.perform_action(
                    ActionInput(id=_ID_TO_GAMEACTION[act["index"]]))
                got = np.asarray(result.frame[-1] if result.frame
                                 else replay._current_frame)
                if not np.array_equal(got, frames[i]):
                    bad += 1
                    print(f"  seed {seed} L{record['level_id']}: frame {i} "
                          f"differs on replay")
                    break
            if replay._state != GameState.WIN:
                bad += 1
                print(f"  seed {seed} L{record['level_id']}: replay does not "
                      f"end in WIN")
        print(f"  seed {seed}: {len(levels)} levels replayed press for press")
    print(f"verify: {bad} problems")
    return 1 if bad else 0


def _fuzz(trials: int = 6, steps: int = 40) -> int:
    """`_install_lean_rules` must be a speed-up and nothing else: replay the
    same random presses on an untouched interpreter and on the lean one, and
    compare what the agent would SEE plus what the engine holds.

    ACTION is in the alphabet on purpose -- the rules being dropped are the ones
    that paint the mirror, and saving and resetting are the only things that
    touch it.

    The two comparisons are not the same test and neither subsumes the other.
    The FRAME is what the corpus is made of and it must match at every press,
    including the level's opening frame. The GRID is checked from the first
    press onwards, because `_settle` deliberately runs level 7's last
    propagation step early: the untouched interpreter paints that cell on its
    first press instead, so the two grids agree from then on and the one frame
    where they do not is identical anyway (the cell is under an opaque Wall)."""
    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    plain = PuzzleScriptAdapter(SaviorSolver.game_name, seed=0)
    lean = SaviorSolver().make_game(0)
    # `_install_lean_rules` edits the parsed game's rule list in place, so a
    # shared parse would make this whole check compare the lean interpreter with
    # itself and pass for the wrong reason.
    assert plain._game is not lean._game, "the two adapters share one parse"
    # Rendered straight off the engine rather than read from ``_current_frame``,
    # which only the adapter's own action path refreshes.
    frame = lambda g: _render_frame(g._engine, g._game)   # noqa: E731
    bad = 0
    for level in range(plain.n_levels):
        for trial in range(trials):
            rng = random.Random(1000 * level + trial)
            plain.set_level(level)
            lean.set_level(level)
            if not np.array_equal(frame(plain), frame(lean)):
                bad += 1
                print(f"  L{level} t{trial}: the seated levels look different")
                continue
            for step in range(steps):
                direction = rng.choice(DIRECTIONS)
                plain._engine.step(direction)
                lean._engine.step(direction)
                if not np.array_equal(frame(plain), frame(lean)):
                    bad += 1
                    print(f"  L{level} t{trial}: frames differ after step "
                          f"{step} ({direction})")
                    break
                if plain._engine.grid != lean._engine.grid:
                    bad += 1
                    print(f"  L{level} t{trial}: grids differ after step "
                          f"{step} ({direction})")
                    break
                if plain._engine.check_win() != lean._engine.check_win():
                    bad += 1
                    print(f"  L{level} t{trial}: win differs at step {step}")
                    break
        print(f"  L{level}: {trials}x{steps} presses agree")
    print(f"fuzz: {bad} divergences")
    return 1 if bad else 0


#: Every cell stack a Savior level can present, as ``name -> object names``.
#: Composition, not object: these bugs live in the stack (a marker under the
#: player, a doorway under a crate), not in any one sprite.
_STACKS = (
    ("floor", ()),
    ("wall", ("wall",)),
    ("target", ("target",)),
    ("saver", ("saver",)),
    ("saver+saved", ("saver", "saved")),
    ("crate", ("crate",)),
    ("sticky", ("sticky",)),
    ("player", ("player",)),
    ("player_on_target", ("target", "player")),
    ("player_on_saver", ("saver", "player")),
    ("player_on_saver+saved", ("saver", "saved", "player")),
    ("crate_on_target", ("target", "crate")),
    ("sticky_on_target", ("target", "sticky")),
    ("doorA_shut", ("wallone",)),
    ("doorA_open", ("walloneno",)),
    ("doorB_shut", ("walltwo",)),
    ("doorB_open", ("walltwono",)),
    ("btnA_up", ("buttonone",)),
    ("btnA_down", ("buttononep",)),
    ("btnB_up", ("buttontwo",)),
    ("btnB_down", ("buttontwop",)),
    ("crate_on_btnA", ("buttononep", "crate")),
    ("crate_on_btnB", ("buttontwop", "crate")),
    ("player_on_btnA", ("buttononep", "player")),
    ("player_on_btnB", ("buttontwop", "player")),
    ("player_in_doorA", ("walloneno", "player")),
    ("player_in_doorB", ("walltwono", "player")),
    ("crate_in_doorA", ("walloneno", "crate")),
    ("crate_in_doorB", ("walltwono", "crate")),
    ("sticky_in_doorA", ("walloneno", "sticky")),
)
#: Zone tint -> the bookmark marker that zone's saver wears. The tint is not
#: decoration: it is painted onto the floor at level start and it is how a
#: player reads which half of the level a rollback would restore.
_ZONES = (("z1", "one", "onesaved"), ("z2", "two", "twosaved"),
          ("z3", "three", "threesaved"), ("z0", None, None))

#: Pairs that render alike and are not defects. A Wall is opaque, so it hides
#: the zone tint under it (and the tint of the cell you cannot walk on is not
#: information). A pressed button with nothing on it does not exist -- ``late
#: [buttononep no heavy] -> [buttonone]`` un-presses it within the same turn --
#: so its sprite is only ever seen under a crate or the player, and those
#: compositions are distinct.
_ALLOWED_CLASHES = frozenset(
    {frozenset({f"{a}/wall", f"{b}/wall"})
     for a, b in itertools.combinations([z[0] for z in _ZONES], 2)}
    | {frozenset({f"{z}/btnA_up", f"{z}/btnA_down"}) for z, *_ in _ZONES}
)


def _audit() -> int:
    """Render every composition at the 5px cells every Savior level uses and
    assert pairwise distinctness."""
    game = SaviorSolver().make_game(0)
    parsed, eng = game._game, game._engine
    idx = parsed.obj_name_to_idx
    game.set_level(0)
    size = 12                                   # flickscreen is 12x12 throughout
    assert parsed.flickscreen == (size, size)

    shots = {}
    for zone, tint, marker in _ZONES:
        for name, objs in _STACKS:
            stack = {idx["background"]} | {idx[o] for o in objs}
            if tint:
                stack.add(idx[tint])
            if "saved" in objs and marker:
                stack.add(idx[marker])
            eng.height, eng.width = size, size
            eng.grid = [[{idx["background"]} for _ in range(size)]
                        for _ in range(size)]
            eng.grid[size // 2][size // 2] = stack
            eng._position_index_dirty = True
            eng._rule_noop_cache.clear()
            parsed.flickscreen = None           # the board IS the screen here
            shots[f"{zone}/{name}"] = np.asarray(_render_frame(eng, parsed)).copy()
    parsed.flickscreen = (size, size)

    clashes = [frozenset({a, b})
               for a, b in itertools.combinations(sorted(shots), 2)
               if np.array_equal(shots[a], shots[b])]
    unexpected = [c for c in clashes if c not in _ALLOWED_CLASHES]
    print(f"  {len(shots)} compositions, {len(clashes)} render alike "
          f"({len(clashes) - len(unexpected)} of them known-benign)")
    for clash in unexpected:
        a, b = sorted(clash)
        print(f"  IDENTICAL: {a} == {b}")
    print("audit clean" if not unexpected
          else f"AUDIT FAILED: {len(unexpected)} indistinguishable pairs")
    return 1 if unexpected else 0


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(SaviorSolver.main())
