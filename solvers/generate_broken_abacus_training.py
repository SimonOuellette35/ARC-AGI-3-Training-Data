"""Generate Phase-1 training data for the PuzzleScript game ps:broken_abacus
("Broken Abacus" by Le Slo).

The harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is the game-specific part: the mechanic model, the goal
heuristic, the search ladder, the stored plans and the optimal-set labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_broken_abacus",
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
action (post rotation remap), i.e. the button an agent presses in the rotated
view, so replaying the recorded actions reproduces the recorded frames exactly.

The game
--------
An abacus with two rods. The player is not a character but a RIGID BAR: one
horizontal bar (a run of ``BarH`` cells in a single row) and, on most levels, one
vertical bar (a run of ``BarVN`` cells in a single column), often crossing each
other. Exactly one rod is live at a time; ACTION (X) swaps which, turning
``BarH -> BarHN`` (inert, red) and ``BarVN -> BarV`` (live) in the same press.
Arrow keys translate the live rod one cell; the inert rod is pinned by
``[> playernotMove] -> [playernotMove]``.

The goal is always ``All Target on Crate`` -- cover every ring-shaped Target
with a crate -- plus ``All BarV on Crate``, which is vacuous while the vertical
rod is inert and is why a plan that finishes in vertical mode has to park that
rod on crates (or toggle back first).

Five mechanics do all the work:

  * **Push.** A moving horizontal rod shoves the crates in the cells it is
    entering, and those crates shove the crates beyond them -- so a 3-wide rod
    moving down pushes a whole row of crates at once. It also shoves the inert
    vertical rod, but only through a crate; rod-on-rod is transparent (they sit
    on different collision layers and slide through each other).
  * **Carry.** When a crate's push is refused -- it is against a wall, or behind
    another stuck crate -- the rod does NOT stop. It slides ONTO the crate's
    cell, and from then on drags that crate with it
    (``[> barh crate] -> [> barh > crate]``). This is the mechanic the levels are
    built around: a target on the far side of a wall stub is unreachable by
    pushing and trivial by shoving a crate into the wall and towing it back.
    Several levels start with a crate already mounted on a rod (the ``a`` / ``b``
    / ``2`` legend cells).
  * **Wall.** The horizontal rod stops dead at a wall; the turn is cancelled for
    every piece, not just the blocked cell.
  * **Pit.** A rod spanning a pit is fine as long as ONE of its cells is still
    over solid ground; the moment every cell is over a pit the whole rod drops
    and is gone. A crate over a pit is destroyed unless a rod cell shares its
    cell -- so a rod parked across a hole is a bridge, and pulling it away drops
    whatever it was carrying. Losing a crate usually makes the level unwinnable,
    which is what `BrokenAbacusExpert.dead` prunes.
  * **Ghost.** The LIVE vertical rod (``BarV``) is a phantom. Every movement,
    push, block and pit rule in the file is written against ``BarH`` (the live
    horizontal rod) and ``BarVN`` (the inert vertical one); ``BarV`` is named by
    nothing except the win condition, and it shares a collision layer with
    nothing solid. So while it is live it drifts through walls, crates and the
    other rod alike, stopped only by the edge of the board, and it neither
    pushes nor carries anything. ACTION therefore reads as "unstick the vertical
    rod, fly it anywhere, set it down again" -- and setting it down is what makes
    it a wall for crates once more. Verified against the interpreter, not read
    off the source: a fuzz over random play found the ghost sitting on wall
    cells and leaving a crate it was carrying behind the moment it went live.

Two consequences the expert is built around: the horizontal rod is the ONLY
piece that can ever move a crate (so losing it to a pit ends the level, and its
distance to a crate is the only distance worth estimating), and no progress at
all is possible while the vertical rod is live (so ghost mode always costs at
least one more ACTION).

Sprite fixes in the game file
-----------------------------
`data/puzzlescript_games/Broken_Abacus.txt` shipped both rods as a ONE-PIXEL line
down the middle of the 5x5 sprite (row 2 for the horizontal, column 2 for the
vertical). `_render_cell_sprite` samples a 5x5 sprite into a cell_px box with
PIL's centered-nearest formula, and at ``cell_px == 4`` that samples sprite rows
{0, 1, 3, 4} -- it drops the middle row entirely. Levels 3, 11 and 14 are 13-14
cells wide and therefore render at cell_px 4, so the PLAYER was invisible on
them: the whole rod vanished from the frame while still being the only thing the
input moves. Both rods now draw their line one pixel off-centre (row 1 / column
1), which every cell_px in this game samples, and no cell they can share gains
any occlusion: the crate (sprite rows 1-3, cols 1-3) and the Target ring keep
pixels under a rod in both orientations, verified per level in both modes.

The live vertical rod was also drawn in ``black``, which is exactly the ARC index
the Pit renders in -- so ``BarV`` was invisible on the pit levels, which are the
levels where standing a rod over a pit is the mechanic. It is now pink (index 6,
otherwise unused here), and the inert ``BarVN`` keeps its own desaturated colour,
which lands on dark grey (index 3, also otherwise unused). The four rod states
are therefore four distinct palette entries: live horizontal 12, inert
horizontal 8, live vertical 6, inert vertical 3.

None of this touches a rule, an object name or a collision layer, so the stored
plans below are unaffected by it.

Expert solver
-------------
`BrokenAbacusExpert`: search over the REAL interpreter whose successors are
``free walk, then one interacting press`` MACROS. `ps_astar`'s own macro classes
do not fit -- they assume a one-cell player that walks to a push cell, and this
player is a multi-cell rigid body whose every move is potentially a push, a
carry and a bridge at once -- so the walk model is local to this file, and
fuzz-verified against the interpreter rather than read off the rules.

Why macros. Almost every press here is a walk: the rod slides a cell and nothing
else changes. Branching on presses spends the whole budget re-deriving the same
approaches. Measured, on the same boards: primitive weighted A* plus a 600k-node
primitive beam won 3 of 16 levels.

The heuristic is a greedy nearest-crate matching over the uncovered targets
(Manhattan, since a push moves a crate one cell), plus the HORIZONTAL rod's
distance to the nearest crate that still has somewhere to be, plus one more
press whenever the vertical rod is live. It is not merely loose, it is
MISLEADING -- it prices a crate's straight-line distance while the real cost is
walking a rigid rod around behind it -- which is why the search is unweighted:
level 11 solves at weight 1, takes a longer route at weight 3 and fails outright
at weight 6.

So `_search` runs a ladder instead of trusting one search (see `_ladder`), and
its first axis is not the search at all but whether the rods may be SWAPPED.
Every landing the vertical rod can be set down on is its own macro, so allowing
the swap roughly quadruples the branching -- and most levels never need it.
Searching the no-swap space first either wins cheaply or proves the level needs
the swap by exhausting a small space in under a minute. Within each half: A* for
a shortest plan, then greedy best-first, then a beam.

Coverage
--------
Ten of the sixteen levels: 0, 1, 2, 4, 6, 11, 12, 13, 14 and 15. All but 12 and
15 come from an A* pass and are shortest for the half of the ladder that found
them; 12 and 15 come from the greedy pass and are winning but not shortest.

The other six are in `BrokenAbacusSolver.skip_levels`. They are NOT unwinnable
and the searches do not close on them -- instrumented, every one is BUDGET-capped
rather than exhausted, and A* gets within an estimated 1 to 3 moves of a win
before running out. The branching is ~40 macros a node, so a 200k-macro budget
buys only ~5000 expansions, and levels 7, 9 and 10 have been shown to need the
rod swap (their no-swap spaces exhaust with no win), which is the expensive half.
What would move them is a better heuristic, not more compute: a push-distance
table in the shape of `generate_bad_example_training.py`'s, built for a rigid
multi-cell pusher, rather than the Manhattan estimate here.

Stored plans
------------
Engine state after reset is seed-independent (the levels are fixed ASCII maps;
only the presentation is augmented per seed), so every level is searched ONCE and
replayed for every seed. The searches are slow -- the interpreter runs ~750 steps
a second on these boards and the deep levels need six figures of them, tens of
minutes each -- so the results live in `_PLANS` and `prepare_expert` seeds the
memo with them. A stored plan that no longer wins is dropped and that level falls
back to a real search, so a stale entry costs time and cannot corrupt a
demonstration.

Optimal-action targets
----------------------
`optimal_for` labels every expert step with a SET, not just the key that was
pressed. These plans are full of order-free stretches -- walking the rod three
cells left and two cells up is the same state in any interleaving -- and training
one arbitrary interleaving as the only right answer is a lie the policy has to
unlearn. The set is built by an engine-verified reordering probe: at step ``i``,
each of the next `reorder_window` presses is pulled to the front and the whole
permuted suffix is replayed; if it still wins by the same step, that press is an
equally good move here. It is computed once per (level, plan) on a private
adapter and cached, so the recording loop pays nothing for it.

Augmentation
------------
Per (seed, level) the adapter draws a frame rotation (rotation_k in 0..3) from a
private RNG; this game takes no flip and no recolor. The rotation is a genuine
symmetry of the mechanic (no gravity, screen-relative moves) and the four rod
states stay four distinct colours under it, so a 90-degree turn presents the
horizontal rod as a vertical one and the plan replays through
`inverse_remap_action_full`.

Usage (run from the repo root):
    python solvers/generate_broken_abacus_training.py --episodes 200 \
        --out data/training_multi_level/broken_abacus
"""

from __future__ import annotations

import heapq
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (            # noqa: E402
    DIRECTIONS, PSAStarSolver, PSExpert, restore, snapshot,
)

GAME_NAME = "Broken_Abacus"

#: Plan letters <-> engine directions, for `_PLANS`.
_LETTER_TO_DIR = {"u": "up", "d": "down", "l": "left", "r": "right",
                  "x": "action"}

#: Heuristic charge for a target no free crate is left for -- the state cannot
#: win, but `dead` only prunes the clear-cut case (a destroyed crate), so this
#: sinks the rest to the back of the queue while keeping the search complete.
_UNREACHABLE = 99

#: (dr, dc) per engine direction, for the free-walk BFS.
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: Engine-verified winning plans, one per level, found by the ladder in
#: `BrokenAbacusExpert._search` and stored because each one costs minutes of
#: search. Replayed and re-verified at startup by `prepare_expert`; a plan that
#: no longer wins is dropped and its level falls back to a real search.
_PLANS: dict[int, str] = {
    0: "ullurrrddldrrrlll",
    1: "rddrrrullluuurdldddrrrulrulddlu",
    2: "ulurddrrdrdlluuurdrlldllud",
    4: "rullrrruldluurddllrrddrr",
    6: "uurlddddrrllurrlluul",
    11: "rxlxdllrrrrrrrd",
    12: "drrllulduuurrrrrdduullllddlrddrrlullurdrrlluuuu",
    13: "uurrrdldrrullull",
    14: "drdlrdllrrruulllrrrrurrdrrru",
    15: "rrddlldrulllrrrrullldrrldrruluuu",
}


class BrokenAbacusExpert(PSExpert):
    """A* whose successors are ``free walk, then one interacting press`` MACROS.

    WHY (what the primitive search runs out of). Almost every press in this game
    is a walk: the rod slides a cell and nothing else on the board changes. A
    primitive search therefore spends its whole budget re-deriving the same
    approaches -- a 50-move solution is a depth-50 search over branching 5, and
    only the handful of presses that met a crate changed anything. Branching on
    macros makes the depth the number of INTERACTIONS, which is single digits on
    most of these levels. Measured: primitive weighted A* plus a 600k-node
    primitive beam won 3 of the 16 levels.

    The walks are not stepped through the interpreter, they are applied by grid
    surgery (`_translated`), because a free move is by definition a pure
    translation of the rod and the crates it tows. That makes a macro cost ONE
    interpreter step regardless of how far the rod walked, so the whole search
    costs about what the primitive one did per node while being an order of
    magnitude shallower. `_free` -- the predicate deciding what counts as a walk
    -- is fuzz-verified against the interpreter over random play on every level
    (0 false positives in 9600 moves); `_search` replays every candidate plan
    through the interpreter before returning it, so a model flaw can only ever
    cost search time.

    The ghost makes this shape fit a game that is not a sokoban: while the
    vertical rod is live, EVERY move is free, so its reachable set is the whole
    board and "fly the ghost anywhere and set it down" is one macro per landing
    cell rather than a search.
    """

    directions = DIRECTIONS

    #: `_key` is dynamic-objects-only, so it is canonical only WITHIN a level.
    scope_by_level = True

    #: States kept per macro layer by `_macro_beam`, ranked by `heuristic`.
    beam_width: int = 120
    #: Give up after this many INTERACTIONS (not presses).
    beam_depth: int = 24
    #: Macro budget for the whole beam (`node_cap` stays A*'s).
    beam_node_cap: int = 400_000

    #: How many places the horizontal rod may be standing when the rods swap.
    #: Only bites while it is TOWING -- see `_ghost_anchors`.
    ghost_anchors: int = 2

    #: Macro budget for `_macro_greedy`.
    greedy_node_cap: int = 400_000

    #: Whether the rods may be swapped at all. The ghost trip is ~70% of the
    #: branching (every landing is its own macro) and most levels never need it,
    #: so `_search` runs the whole ladder with it OFF first -- see there.
    use_ghost: bool = True

    def setup(self) -> None:
        g = self.g
        idx = g.obj_name_to_idx
        self.i_wall, self.i_pit, self.i_crate = (idx["wall"], idx["pit"],
                                                 idx["crate01"])
        self.i_barh, self.i_barv = idx["barh"], idx["barv"]
        self.i_barhn, self.i_barvn = idx["barhn"], idx["barvn"]
        self.target_ids = set(g.resolve_object_name("target"))
        self.crate_ids = {self.i_crate}
        self.barv_ids = {self.i_barv}
        # The horizontal rod in EITHER state. It is the only thing on the board
        # that can move a crate (see the ghost note in the module docstring), so
        # this -- not "whichever rod is live" -- is the piece whose distance to a
        # crate means something, and its disappearance is the end of the level.
        self.horiz_ids = {self.i_barh, self.i_barhn}
        # Everything a turn can move or destroy: the four rod states, the
        # crates, and the death animation a dropped piece leaves behind. Walls,
        # pits, targets and every floor/edge decoration derived from them are
        # STATIC on these boards -- verified over 19200 random turns -- and the
        # aux/cantMove bookkeeping objects are all cleared by the late rules
        # before the turn returns. Keying on the dynamic set instead of on every
        # non-background cell is worth real time: the key is rebuilt for every
        # node, and on the bigger boards the static scenery is most of the grid.
        self.anim_ids = set(g.resolve_object_name("anims"))
        self.dyn_ids = ({self.i_crate, self.i_barh, self.i_barv,
                         self.i_barhn, self.i_barvn} | self.anim_ids)
        self._ghost_now = False

    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    # -- reading the board -----------------------------------------------------
    def _read(self, eng) -> dict:
        """Every cell set the model and the heuristic need, in one pass.

        ``rod`` is whichever rod is LIVE and ``ghost`` says which one that is.
        An empty ``rod`` is a real state, not an error: press ACTION on a level
        with no vertical rod and the horizontal one goes inert with nothing to
        replace it, so nothing is live until ACTION is pressed again."""
        walls = set(); pits = set(); crates = set(); inert = set()
        rod = set(); horiz = set(); targets = set(); anims = set()
        ghost = False
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if self.i_wall in cell:
                    walls.add((r, c))
                if self.i_pit in cell:
                    pits.add((r, c))
                if self.i_crate in cell:
                    crates.add((r, c))
                if self.i_barvn in cell:
                    inert.add((r, c))
                if self.i_barh in cell:
                    rod.add((r, c)); horiz.add((r, c))
                if self.i_barhn in cell:
                    horiz.add((r, c))
                if self.i_barv in cell:
                    rod.add((r, c)); ghost = True
                if cell & self.target_ids:
                    targets.add((r, c))
                if cell & self.anim_ids:
                    anims.add((r, c))
        return {"walls": walls, "pits": pits, "crates": crates, "inert": inert,
                "rod": rod, "horiz": horiz, "targets": targets, "ghost": ghost,
                "anims": anims, "vert": rod if ghost else inert,
                "height": eng.height, "width": eng.width}

    def _key_board(self, board) -> tuple:
        """The dedup key straight off a board read, so a child costs ONE grid
        scan rather than one per (key, heuristic, dead) call. Same content as
        `_key`: the pieces plus which rod is live."""
        return (frozenset(board["crates"]), frozenset(board["horiz"]),
                frozenset(board["vert"]), frozenset(board["anims"]),
                board["ghost"])

    def dead(self, eng) -> bool:
        """True once the win is arithmetically out of reach.

        Two ways to get there, both irreversible and both common: a crate fell
        into a pit (nothing creates crates, so fewer crates than targets can
        never be undone), or the horizontal rod did (it is the only piece that
        can move a crate, and with the vertical rod still on the board the
        engine does not even call losing it a game over)."""
        return self._dead_board(self._read(eng))

    def _dead_board(self, board) -> bool:
        return (len(board["crates"]) < len(board["targets"])
                or not board["horiz"])

    def heuristic(self, eng) -> int:
        return self._estimate(self._read(eng))

    def _estimate(self, board) -> int:
        crates, targets = board["crates"], board["targets"]
        bare = targets - crates
        loose = crates - targets

        total = 0
        if bare:
            taken = set()
            remaining = len(bare)
            for dist, goal, crate in sorted(
                    (abs(t[0] - c[0]) + abs(t[1] - c[1]), t, c)
                    for t in bare for c in loose):
                if goal in taken or crate in taken:
                    continue
                taken.add(goal)
                taken.add(crate)
                total += dist
                remaining -= 1
                if not remaining:
                    break
            total += _UNREACHABLE * remaining
            if board["horiz"] and loose:
                # The horizontal rod's walk to the first crate it has to touch.
                # -1 because it pushes from the cell BESIDE the crate. Without
                # this the estimate is flat for the whole approach.
                total += max(0, min(abs(a[0] - c[0]) + abs(a[1] - c[1])
                                    for a in board["horiz"] for c in loose) - 1)
        # `All BarV on Crate` plus the fact that nothing can push while the
        # vertical rod is live: either way the answer is at least one more
        # ACTION. Charging the rod's LENGTH here instead (one per cell off a
        # crate) would price ghost mode out of the search, and ghost mode is how
        # several of these levels are meant to be opened.
        if board["ghost"] and not board["rod"] <= crates:
            total += 1
        return total

    def _won(self, board, crates, rod) -> bool:
        """The engine's two win conditions off the model's own cell sets, so a
        walk can stop the moment it wins without an interpreter step. Every plan
        the search returns is replayed through the interpreter anyway (see
        `_search`), so this is an accelerator, never the authority."""
        return (board["targets"] <= crates
                and (not board["ghost"] or rod <= crates))

    # -- the free-walk model ---------------------------------------------------
    def _free(self, board, rod, crates, delta) -> set | None:
        """The crates after a FREE translation of ``rod`` by ``delta``, or None
        when the move is not one.

        Free means the live rod slides one cell and nothing on the board changes
        except the crates it is towing. Everything this refuses is either a
        no-op (the rod against a wall cancels the turn), suicide (the rod
        entirely over a pit drops), or an interaction the interpreter has to
        resolve -- and those become macro END moves instead, so refusing is
        always safe and only ever costs the search a walk.

        The two rules here that are NOT guessable from the game source, both
        found by fuzzing: a towed crate behaves like a pushed one and shoves
        whatever is in front of it, and every drag rule bails out when a BarVN
        shares either cell (``[> barh crate no BarVN]`` / ``| no barvn``)."""
        dr, dc = delta
        height, width = board["height"], board["width"]
        moved = {(r + dr, c + dc) for r, c in rod}
        if any(not (0 <= r < height and 0 <= c < width) for r, c in moved):
            return None
        if board["ghost"]:
            return crates                       # phases through everything
        walls, pits, inert = board["walls"], board["pits"], board["inert"]
        if moved & walls or (moved - rod) & crates:
            return None
        if moved <= pits:
            # It MIGHT drop -- it survives only when the other rod is holding it
            # up -- so this is never a walk. It is still reachable as a macro
            # end, one press at a time, with the interpreter deciding.
            return None
        towed = crates & rod
        if not towed:
            return crates
        for cell in towed:
            ahead = (cell[0] + dr, cell[1] + dc)
            if ahead in crates or ahead in walls or ahead in inert:
                return None
            if cell in inert:
                return None
        return (crates - towed) | {(r + dr, c + dc) for r, c in towed}

    def _translated(self, snap, rod, towed, delta, rod_idx=None):
        """``snap`` with the live rod (and the crates it tows) shifted by
        ``delta``.

        This is what the free-walk model buys: a walk is applied by moving
        object indices in a copied grid instead of by stepping the interpreter
        once per cell, so a macro costs ONE interpreter step no matter how far
        the rod had to travel to make its move."""
        dr, dc = delta
        if rod_idx is None:
            rod_idx = self.i_barv if self._ghost_now else self.i_barh
        out = [[set(cell) for cell in row] for row in snap]
        for r, c in rod:
            out[r][c].discard(rod_idx)
        for r, c in towed:
            out[r][c].discard(self.i_crate)
        for r, c in rod:
            out[r + dr][c + dc].add(rod_idx)
        for r, c in towed:
            out[r + dr][c + dc].add(self.i_crate)
        return out

    # -- macro generation ------------------------------------------------------
    def _analyze(self, board) -> tuple:
        """``(win_walk, macros)``.

        ``win_walk`` is a pure walk that already wins -- the rod tows a crate
        onto the last bare target on its way past, and then there is nothing
        left to interact with.

        ``macros`` are ``("walk", primitive directions, delta, final press)``
        and ``("ghost", primitive directions, delta, None)``:

        * a WALK macro is a free approach ending in the one press the
          interpreter has to resolve, and only where the rod would actually meet
          a crate. A press into a wall is a cancelled turn and a press off the
          last solid ground is suicide; neither is worth an interpreter step.
        * a GHOST macro is the whole round trip ``ACTION, fly, ACTION`` in one
          step: pick the vertical rod up, move it (every ghost move is free, so
          its reachable set is the whole board) and set it back down as an
          obstacle.

        Two prunings keep the branching flat, and both are why this search fits
        where the primitive one did not:

        * The ghost trip is emitted only from where the rod is STANDING, never
          after a walk. Setting the vertical rod down changes nothing the
          horizontal rod can collide with (the two rods are transparent to each
          other), so the horizontal rod's free-reachable set is identical before
          and after the trip and it is still inside it -- walking first and
          walking after cost the same. Without this the branching is the product
          of the two rods' reachable sets instead of their sum.
        * A landing is kept only where the vertical rod ends up touching a
          crate, a target or a pit, since that is the only way it can matter:
          as something a crate is shoved against, as something standing on the
          square a crate has to reach, or as a bridge. This one is a heuristic
          restriction, not a theorem -- it can in principle hide a solution
          whose only idea is parking the rod in open ground.
        """
        rod, crates = board["rod"], board["crates"]
        if not rod:
            # Nothing is live (ACTION on a level with no vertical rod at all).
            # The only move that does anything is pressing it again.
            return None, [("walk", [], (0, 0), "action", None)]

        # Free-walk BFS over rod TRANSLATIONS. Each offset carries the crate set
        # it is reached with, because the rod tows crates as it walks.
        state = {(0, 0): (crates, None)}         # offset -> (crates, parent)
        queue = deque([(0, 0)])
        win_at = None
        while queue:
            off = queue.popleft()
            here, _parent = state[off]
            rod_here = {(r + off[0], c + off[1]) for r, c in rod}
            if off != (0, 0) and self._won(board, here, rod_here):
                win_at = off
                break
            for name, (dr, dc) in _DELTA.items():
                nxt = (off[0] + dr, off[1] + dc)
                if nxt in state:
                    continue
                after = self._free(board, rod_here, here, (dr, dc))
                if after is None:
                    continue
                state[nxt] = (after, (off, name))
                queue.append(nxt)

        def walk_to(off, tree):
            out = []
            while tree[off][1] is not None:
                off, name = tree[off][1]
                out.append(name)
            out.reverse()
            return out

        if win_at is not None:
            return walk_to(win_at, state), []
        walks = {off: walk_to(off, state) for off in state}

        macros = []
        if board["ghost"]:
            # Every ghost landing, as a plain solidify -- this only happens when
            # the SEARCH STARTS in ghost mode, since a ghost macro never leaves
            # the search in it.
            for off in state:
                if self._useful_landing(board, off):
                    macros.append(("walk", walks[off], off, "action", None))
            return None, macros

        walls, pits = board["walls"], board["pits"]
        height, width = board["height"], board["width"]
        for off, (here, _parent) in state.items():
            walk = walks[off]
            rod_here = {(r + off[0], c + off[1]) for r, c in rod}
            for name, (dr, dc) in _DELTA.items():
                if self._free(board, rod_here, here, (dr, dc)) is not None:
                    continue                    # already covered by the walk
                moved = {(r + dr, c + dc) for r, c in rod_here}
                if any(not (0 <= r < height and 0 <= c < width)
                       for r, c in moved):
                    continue                    # off the board: nothing happens
                if moved & walls:
                    # A press that drives the rod into a wall cancels the turn
                    # for every piece, so it is a no-op and not worth a step.
                    # Fuzz-verified: 725 such presses, none changed the grid.
                    continue
                # A press that puts every rod cell over a pit is NOT skipped,
                # even though it usually drops the rod. The two rods hold each
                # other up where they cross on a crate, so a rod entirely over
                # pits survives when it is linked to one that is not -- 4 of 35
                # in the fuzz, and it is how level 11 crosses its chasm. The
                # interpreter decides; `dead` prunes the ones that do drop.
                macros.append(("walk", walk, off, name, None))

        vert = board["vert"]
        if vert and self.use_ghost:
            landings = self._ghost_landings(board, vert)
            for anchor in self._ghost_anchors(state, walks):
                for off, ghost_walk in landings:
                    macros.append(("ghost", walks[anchor], anchor,
                                   ghost_walk, off))
        return None, macros

    def _ghost_anchors(self, state, walks) -> list:
        """The offsets worth STANDING at when the rods swap.

        Where the horizontal rod stands is almost always irrelevant: setting the
        vertical rod down changes nothing the horizontal one can collide with
        (the two are transparent to each other), so its free-reachable set is
        identical before and after the trip and it is still inside it -- walk
        first or walk after, same cost, same state. Emitting the trip only from
        where the rod already stands turns the branching from the PRODUCT of the
        two rods' reachable sets into their sum.

        The exception is a rod that is TOWING: then walking moves crates, and
        where the crates are when the vertical rod lands is exactly what the
        landing means (level 11's solution walks the towed crate out of the way
        first, then drops the rod where it used to be). So anchors are grouped
        by the CRATE SET they present -- one anchor per distinct configuration,
        reached by its shortest walk. A rod towing nothing collapses to a single
        anchor, which is every level but the handful that start with a crate
        mounted on a rod.

        On those, one anchor per configuration is still the reachable set of the
        horizontal rod, and multiplied by the landings it is several hundred
        macros a node -- enough to stall the search at a depth of about two. So
        the anchors are capped at `ghost_anchors`, nearest first. That is a
        COVERAGE CAP, not a canonicalisation: a solution that needs the towed
        crate walked a long way before the vertical rod comes down is one this
        search can miss."""
        best: dict[frozenset, tuple] = {}
        for off, (crates_here, _parent) in state.items():
            key = frozenset(crates_here)
            length = len(walks[off])
            if key not in best or length < best[key][0]:
                best[key] = (length, off)
        ranked = sorted(best.values())
        return [off for _length, off in ranked[:self.ghost_anchors]]

    def _ghost_landings(self, board, vert) -> list:
        """``[(delta, walk)]`` -- every landing worth taking the trip for.

        Three filters. It has to go somewhere (a trip that lands where it took
        off is two presses and no change). It has to matter
        (`_useful_landing`). And it has to leave the round trip a pure
        relocation, which is what lets `_expand` apply it without touching the
        interpreter: a rod that comes down with every cell over a pit drops, and
        a rod flying off a crate it was bridging over a pit drops the crate, so
        both are refused here rather than modelled."""
        pits, crates = board["pits"], board["crates"]
        if any(cell in pits and cell in crates for cell in vert):
            return []                           # it is holding a crate up
        out = []
        for delta, walk in self._ghost_walks(board, vert).items():
            if delta == (0, 0) or not self._useful_landing(board, delta):
                continue
            dr, dc = delta
            if all((r + dr, c + dc) in pits for r, c in vert):
                continue                        # it would drop
            out.append((delta, walk))
        return out

    def _ghost_walks(self, board, vert) -> dict:
        """``{delta: walk}`` for every place the ghost can be flown to.

        A ghost move is free whenever the whole rod stays on the board, so this
        is a plain rectangle of offsets and the walk to each is its L-shaped
        path -- no BFS and no interpreter needed."""
        height, width = board["height"], board["width"]
        rows = [r for r, _ in vert]
        cols = [c for _, c in vert]
        walks = {}
        for dr in range(-min(rows), height - max(rows)):
            for dc in range(-min(cols), width - max(cols)):
                walks[(dr, dc)] = (["down"] * dr + ["up"] * -dr
                                   + ["right"] * dc + ["left"] * -dc)
        return walks

    #: Where a landing has to be to be worth an interpreter step: on, or
    #: orthogonally beside, the thing it is meant to act on.
    _TOUCH = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))

    def _useful_landing(self, board, off) -> bool:
        """Whether setting the vertical rod down at ``off`` can matter.

        A vertical rod does exactly two things once it is solid: it stops a
        crate being shoved into it, and it occupies a square a crate has to
        reach. Both need it ON or BESIDE a crate or a target, so landings
        anywhere else are dropped -- and that matters, because the landings are
        the branching factor. On the widest board the whole rectangle is ~110
        landings per node and this cuts it by an order of magnitude.

        Deliberately NOT counted as useful: parking the rod over a pit as a
        bridge. It is a real use (a rod cell shares its square with a crate and
        keeps it from dropping) but on the pit-heavy boards every landing
        touches a pit, which makes the test vacuous exactly where it has to
        bite. A level whose only solution is a bare bridge is one this search
        will not find."""
        dr, dc = off
        interesting = board["crates"] | board["targets"]
        for r, c in board["vert"]:
            r, c = r + dr, c + dc
            for ar, ac in self._TOUCH:
                if (r + ar, c + ac) in interesting:
                    return True
        return False

    # -- search ----------------------------------------------------------------
    def _expand(self, eng, snap, path, g):
        """Yield ``(cost, plan, won, board)`` for every macro out of ``snap``.

        The child's board is handed back with it so the caller gets its key,
        heuristic and death test off ONE grid scan instead of one per test.
        Leaves the engine holding the child it just built."""
        restore(eng, snap)
        board = self._read(eng)
        self._ghost_now = board["ghost"]
        win_walk, macros = self._analyze(board)
        towed = set() if board["ghost"] else board["crates"] & board["rod"]

        if win_walk is not None:
            delta = (sum(_DELTA[d][0] for d in win_walk),
                     sum(_DELTA[d][1] for d in win_walk))
            restore(eng, self._translated(snap, board["rod"], towed, delta))
            yield g + len(win_walk), path + win_walk, eng.check_win(), None
            return

        # One approach grid per distinct offset, not per macro: every landing
        # shares its anchor's approach, and building it is a whole-grid copy.
        approaches = {(0, 0): snap}

        def approach_for(delta):
            grid = approaches.get(delta)
            if grid is None:
                grid = self._translated(snap, board["rod"], towed, delta)
                approaches[delta] = grid
            return grid

        for kind, walk, delta, tail, landing in macros:
            restore(eng, approach_for(delta))
            if kind == "walk":
                eng.step(tail)
                plan = walk + [tail]
            else:
                # The whole round trip -- ACTION, fly, ACTION -- resolves to one
                # grid edit: the vertical rod ends up somewhere else and every
                # other thing on the board, including which rod is live, ends up
                # exactly where it started. So it is applied by surgery and
                # costs NO interpreter steps, which is the difference between
                # ~100 and ~1000 macros a second on the big boards (the profile
                # is ~all interpreter). `_ghost_landings` refuses the two
                # landings where that identity does not hold -- a rod that would
                # come down entirely over pits drops, and a rod flying away from
                # a crate it was bridging drops the crate -- so the edit is the
                # whole story, and the plan replay in `_search` is the backstop.
                eng.grid = self._translated(approach_for(delta), board["vert"],
                                            set(), landing, self.i_barvn)
                eng._position_index_dirty = True
                eng._rule_noop_cache.clear()
                eng._rule_win = False
                plan = walk + ["action"] + tail + ["action"]
            won = eng.check_win()
            yield g + len(plan), path + plan, won, (None if won
                                                    else self._read(eng))

    def _search(self, eng) -> list | None:
        """Macro A*, then a macro beam, and the winner is replayed through the
        interpreter before it is returned.

        The replay is what makes the free-walk model safe to search over: a
        macro's END press is a real interpreter step, but its WALK is grid
        surgery, so a flaw in the model could in principle yield a plan the
        engine does not follow. Verifying turns that from a correctness risk
        into wasted search time."""
        start = snapshot(eng)
        for ghost, finder in self._ladder():
            self.use_ghost = ghost
            restore(eng, start)
            plan = finder(eng)
            if plan is None:
                continue
            restore(eng, start)
            won = -1
            for i, direction in enumerate(plan):
                eng.step(direction)
                if eng.check_win():
                    won = i
                    break
            restore(eng, start)
            if won >= 0:
                return plan[:won + 1]
            print("  [warn] the free-walk model produced a plan the "
                  "interpreter does not follow -- falling through")
        return None

    def _macro_astar(self, eng) -> list | None:
        """A* over the macros, costed in primitive MOVES (what the agent pays)."""
        if eng.check_win():
            return []
        weight = self.weight
        start = snapshot(eng)
        counter = nodes = 0
        pq = [(weight * self.heuristic(eng), 0, counter, start, [])]
        best = {self._key_board(self._read(eng)): 0}
        while pq:
            _f, g, _c, snap, path = heapq.heappop(pq)
            for cost, plan, won, board in self._expand(eng, snap, path, g):
                nodes += 1
                if won:
                    return plan
                if self._dead_board(board):
                    continue
                key = self._key_board(board)
                if best.get(key, 1 << 30) <= cost:
                    continue
                best[key] = cost
                counter += 1
                heapq.heappush(pq, (cost + weight * self._estimate(board), cost,
                                    counter, snapshot(eng), plan))
                if nodes >= self.node_cap:
                    return None
        return None

    def _ladder(self) -> list:
        """The strategies to try, in order, as ``(ghost allowed, finder)``.

        Rod-swapping OFF comes first, and that ordering is most of the
        solver. Every landing the vertical rod can be set down on is its own
        macro, so allowing the swap roughly quadruples the branching -- and most
        levels never need it. Searching the no-swap space first either wins
        cheaply (levels 6, 12, 13 and 15 all fall there, and two of them
        EXHAUST the space in under a minute) or proves the level needs the swap
        before a single node is spent on one.

        Within each half: A* for a shortest plan, then greedy best-first, then a
        beam. See `_macro_greedy` for why cost-blind search is worth reaching
        for here at all."""
        return [(False, self._macro_astar), (False, self._macro_greedy),
                (True, self._macro_astar), (True, self._macro_greedy),
                (True, self._macro_beam)]

    def _macro_greedy(self, eng) -> list | None:
        """Best-first over the macros on the heuristic ALONE, ignoring cost.

        A* is the wrong shape for the deep boards. Its branching is ~40 macros,
        so a 200k-macro budget buys only ~5000 expansions, and it spends them
        broadening a frontier whose g-values are all similar rather than
        following the one branch that is nearly done -- measured, it gets within
        an estimated 1 to 3 moves of a win on the levels it then fails to solve.
        Dropping g entirely turns the same budget into depth. The plans are
        longer than A*'s, which is the trade; the reordering probe in
        `BrokenAbacusSolver._optimal_table` still labels them honestly, as
        "equally good as what the expert did"."""
        if eng.check_win():
            return []
        counter = nodes = 0
        pq = [(self.heuristic(eng), counter, snapshot(eng), [])]
        seen = {self._key_board(self._read(eng))}
        while pq:
            _h, _c, snap, path = heapq.heappop(pq)
            for _cost, plan, won, board in self._expand(eng, snap, path, 0):
                nodes += 1
                if won:
                    return plan
                if self._dead_board(board):
                    continue
                key = self._key_board(board)
                if key in seen:
                    continue
                seen.add(key)
                counter += 1
                heapq.heappush(pq, (self._estimate(board), counter,
                                    snapshot(eng), plan))
                if nodes >= self.greedy_node_cap:
                    return None
        return None

    def _macro_beam(self, eng) -> list | None:
        """The same macros explored by a width-capped breadth-first beam.

        A* steers by the heuristic, which goes flat over the stretch where the
        rod is being manoeuvred BEHIND a crate rather than towards one -- and on
        these boards that is most of a solution. The beam spends its budget on
        breadth at every layer instead and uses the heuristic only to decide who
        survives, which is all a flat estimate is good for. Its plans win but
        wander."""
        if eng.check_win():
            return []
        frontier = [(snapshot(eng), [])]
        seen = {self._key_board(self._read(eng))}
        nodes = 0
        for _depth in range(self.beam_depth):
            kids = []
            for snap, path in frontier:
                for _cost, plan, won, board in self._expand(eng, snap, path, 0):
                    nodes += 1
                    if won:
                        return plan
                    if self._dead_board(board):
                        continue
                    key = self._key_board(board)
                    if key in seen:
                        continue
                    seen.add(key)
                    # Sort on the heuristic ALONE -- the tuple also carries a
                    # snapshot (a list of sets), which has no ordering.
                    kids.append((self._estimate(board), snapshot(eng), plan))
                    if nodes >= self.beam_node_cap:
                        return None
            if not kids:
                return None                     # the reachable space closed
            kids.sort(key=lambda kid: kid[0])
            frontier = [(snap, plan) for _h, snap, plan in kids[:self.beam_width]]
        return None


class BrokenAbacusSolver(PSAStarSolver):
    game_id = "puzzlescript_broken_abacus"
    game_name = GAME_NAME
    expert_cls = BrokenAbacusExpert

    #: Levels the ladder in `BrokenAbacusExpert._search` cannot win inside a
    #: practical budget, skipped up front so startup discovery does not burn
    #: hours proving it again on every run. See the module docstring.
    skip_levels = frozenset({3, 5, 7, 8, 9, 10})

    #: Every stored plan is replayed from the memo, so these budgets only bite
    #: when a stored plan has gone stale (or a level is being re-searched).
    #: Unweighted: measured, weighting this heuristic makes the search WORSE
    #: (level 11 solves at w=1 and fails outright at w=6), because the estimate
    #: is not merely loose, it is misleading -- it prices a crate's Manhattan
    #: distance while the real cost is walking the rod around behind it.
    node_cap = 600_000
    weight = 1
    #: The adapter GAME_OVERs at 200 actions per level; the longest plan here is
    #: well under half that, leaving room for the exploration prefix in front.
    max_steps = 200

    #: How far ahead `optimal_for`'s reordering probe looks. Presses that commute
    #: with the one being taken are almost always adjacent to it (they are the
    #: two axes of one walk); widening this costs O(window x len^2) interpreter
    #: steps per level and finds nothing.
    reorder_window: int = 8

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._probe = None                       # private adapter, see _optimal
        self._opt_cache: dict[tuple, list[list[str]]] = {}

    def prepare_expert(self, game, expert) -> None:
        """Seed the expert's memo with `_PLANS`, keeping only plans that still
        win.

        `PSExpert.plan` memoises by ``(level, state key)`` and, with
        ``recovery_mode = "reset"``, is only ever asked about a level's START
        state -- so one seeded entry per level skips the search entirely."""
        for level, letters in sorted(_PLANS.items()):
            plan = [_LETTER_TO_DIR[ch] for ch in letters]
            game.set_level(level)
            eng = game._engine
            key = (level if expert.scope_by_level else None, expert._key(eng))
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

    # -- optimal-action targets ------------------------------------------------
    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The equally-good presses at plan step ``pi`` -- see the module
        docstring. Never empty: the plan's own press is always in the set."""
        table = self._optimal_table(level, plan)
        return table[pi] if pi < len(table) else None

    def _optimal_table(self, level: int, plan: list) -> list[list[str]]:
        """One optimal SET per step of ``plan``, built by replaying reorderings
        of the remaining plan on a private adapter.

        A press ``d`` further along the plan is equally good HERE when pulling
        it to the front and replaying the rest still wins by the same step: the
        two presses commute, which on these boards means they are the two axes
        of one walk. The probe runs on its own `PuzzleScriptAdapter` rather than
        on the recording engine, because it steps hundreds of times and
        `check_win` reads a per-step flag that a bare grid restore would leave
        holding the probe's answer.

        This is a claim about the RECORDED plan, not about the game: a beam plan
        is not shortest, so the set is "as good as what the expert did", which is
        what the demonstration is teaching either way."""
        key = (level, tuple(plan))
        cached = self._opt_cache.get(key)
        if cached is not None:
            return cached

        if self._probe is None:
            self._probe = self.make_game(0)
        self._probe.set_level(level)
        eng = self._probe._engine

        table: list[list[str]] = []
        for pi, taken in enumerate(plan):
            suffix = plan[pi:]
            here = snapshot(eng)
            best = {taken}
            for j in range(1, min(len(suffix), self.reorder_window)):
                if suffix[j] in best:
                    continue
                candidate = [suffix[j]] + suffix[:j] + suffix[j + 1:]
                restore(eng, here)
                for direction in candidate:
                    # The permutation is the same LENGTH as the suffix, so any
                    # win inside it is a win by the recorded plan's last step or
                    # sooner -- this press is at least as good as the one being
                    # taken.
                    eng.step(direction)
                    if eng.check_win():
                        best.add(suffix[j])
                        break
            restore(eng, here)
            eng.step(taken)
            table.append(sorted(best))

        self._opt_cache[key] = table
        return table


if __name__ == "__main__":
    sys.exit(BrokenAbacusSolver.main())
