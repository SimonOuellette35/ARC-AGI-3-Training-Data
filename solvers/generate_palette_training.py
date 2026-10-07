"""Generate Phase-1 training data for the PuzzleScript game ps:palette
("Palette", Versial -- subtractive-looking, additive-really colour sokoban).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: a native model of the thirteen rules, the exact field
that plans over it, and the checks that pin both to the interpreter.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_palette",
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
augmented view, so replaying the recorded actions reproduces the recorded frames
exactly. Each step also carries the full set of equally-optimal presses (see
"Optimal-action sets" below).

The game
--------
Sokoban whose crates are coloured lights: shoving one into another MIXES them.
``noaction`` is in the prelude, so the four arrow keys are the whole action
space. The rule set is one push plus twelve mixes::

    [ > Player | Crate ] -> [ > Player | > Crate ]

    [ > Blue  | Red    ] -> [ | Purple ]      [ > Red    | Blue  ] -> [ | Purple ]
    [ > Blue  | Green  ] -> [ | Cyan   ]      [ > Green  | Blue  ] -> [ | Cyan   ]
    [ > Green | Red    ] -> [ | Yellow ]      [ > Red    | Green ] -> [ | Yellow ]
    [ > Blue  | Yellow ] -> [ | White  ]      [ > Yellow | Blue  ] -> [ | White  ]
    [ > Green | Purple ] -> [ | White  ]      [ > Purple | Green ] -> [ | White  ]
    [ > Cyan  | Red    ] -> [ | White  ]      [ > Red    | Cyan  ] -> [ | White  ]

    WINCONDITIONS: All TRed on Red, All TBlue on Blue, ... (one per colour)

Five facts, and the first two are the whole solver:

  * **Colour is a SET of primary lights and mixing is DISJOINT UNION.**
    red={R}, green={G}, blue={B}, yellow={R,G}, cyan={G,B}, purple={R,B},
    white={R,G,B}. The twelve rules above are *exactly* the six disjoint unions
    written in both orders, and every OVERLAPPING pair (yellow+purple, white+
    anything, X+X) has no rule at all. `_MASK` is that algebra; `_Board.step` is
    three lines because of it.
  * **A mix lands on the STATIONARY crate's cell and consumes the mover**, so
    two crates become one and the board's total light content is exactly
    CONSERVED -- nothing here spends it, because a target is not consumed when
    it is covered. That conservation is what makes the field's reachable set
    small and what makes `--verify`'s exhaustive sweep affordable.
  * **Pushes are one crate deep and never chain.** The push rule matches
    ``moving Player | Crate`` and re-firing it changes nothing, so the force
    never reaches a second crate. A crate with a wall, the board edge, or a
    non-combining crate behind it simply cannot move -- and since crates share
    the player's collision layer, the whole turn is then a silent no-op, player
    included. ``PRRB`` moving right is a total no-op even though the second R
    and the B would happily mix.
  * **Targets are inert scenery on their own collision layer.** Any crate (and
    the player) may stand on any target; crossing one disturbs nothing; a mix
    that resolves on top of a target leaves the target intact. So unlike
    ps:color_combination there is no "push it IN" rule -- a crate that is merely
    PARKED on a matching target already satisfies that target.
  * **The win is per TARGET, not per crate.** ``All TRed on Red`` asks every
    TRed cell to carry a Red; spare crates elsewhere are free, and two TReds
    need two Reds.

All five are asserted against the interpreter -- see "Checks".

The levels
----------
Nine shipped boards, none skipped, none authored (the file's own progression
already ramps 4 -> 34 presses):

    level  size   crates targets  plan ties  states   what it is
    0      5x10    1     1           4   0%     552   push the red onto the TRed
    1      5x10    1     1          11   9%     462   the wall makes you walk PAST
                                                      the crate to get behind it
    2      5x10    2     2          16   0%    1408   two independent lanes
    3      5x10    2     1          10  10%    4230   the first MIX: R+B -> purple
    4      5x10    2     2          12   8%    1722   the doorway IS the covered
                                                      TRed -- shove the red off
                                                      its target and put it back
    5      4x11    4     3          24   8%    2126   mix the two COVERED crates
                                                      into the purple, then
                                                      rebuild both from the spares
    6      7x 7    5     1          26   0%   37389   thread the blue to its target
                                                      past four reds, touching none
    7      7x 7    4     2          34   0%  167644   two R-over-B lanes: mix one
                                                      pair, deliver the other red
    8      5x 9    4     2          19   5%  173409   G+B -> the TCyan, G+R -> the
                                                      TYellow, nothing left over

``plan`` is proved SHORTEST (not merely found), ``ties`` is the share of its
steps with more than one equally-optimal press, and ``states`` is the size of the
level's whole reachable state space (388 942 over the nine). 156 presses in
total; the longest plan is 34 against the adapter's 200-step per-level budget, so
this game needs no `games/` step-limit wrapper.

Levels 4 and 5 are the ones that make the game a game rather than a sokoban with
paint: they open with targets ALREADY covered (``@`` is ``Red and TRed``, ``!``
is ``Blue and TBlue``), so the crate you need for the remaining target is one you
are currently standing a win on. Level 4 goes further and makes the covered TRed
the only doorway between its two rooms, so the red MUST come off its target and
be walked back afterwards. Spending a crate wrongly is not punished by anything
on screen -- the state simply stops having a distance, which is why `dead` would
be a guess and the field is not.

Expert solver
-------------
Not a search over the interpreter, and not the family's macro A* either: the
mechanic collapses to a small native model (`_Board` -- a 3-bit colour mask per
free cell packed into one int, plus the player's index), so `_Board.field`
computes the EXACT distance-to-win of every reachable state and `_Board.plan`
walks it downhill.

The field is built forwards, the ps:one_way_street way rather than the
ps:futuristic_block_pushing_game way:

  1. a forward BFS from the start closes the reachable set, recording each
     state's predecessors as it goes;
  2. a backward BFS from every reachable WINNING state over those predecessors
     gives the exact distance-to-win.

The reverse pull-BFS is not available here for the usual reason and for one
particular to this game: ``All TRed on Red`` with spare crates has no unique
winning configuration to seed from (level 6 has five crates for one target), and
mixing is IRREVERSIBLE, so the inverse relation would have to invent every
disjoint split of every crate -- the forward closure gets the same answer with no
inverse at all. It is exact for the same reason: every shortest path out of the
start stays inside the reachable set. Conservation of light content keeps that
set small (34k states at the worst level; all nine close in well under a second),
which is also what lets `--verify` check the model EXHAUSTIVELY rather than by
sampling.

Three things this buys over the family's macro A* (which does solve this game --
it was the first thing tried, and it agrees with the field on all nine plans):

  * **Optimality is proved, not hoped for.** `PSPushExpert` merges states by the
    player's reachable REGION, which mis-costs a move-counted plan, and its
    macro goal test returns the first winning macro unless `exact_goal_test` is
    set.
  * **The optimal-action SETS are measured.** At distance ``d``, a press is
    optimal iff it reaches ``d - 1``. `PSPushExpert.annotate_walks` can only
    ever find ties between WALK steps and calls every push forced; here a tie
    between two different pushes, or between "walk now, push later" and its
    reverse, is labelled too.
  * **Speed.** All nine fields in ~0.9s against ~15s of interpreter A*, and
    cached to disk thereafter.

The interpreter is then only asked to CERTIFY: `--plans` replays each field plan
through the real engine and requires the win on the LAST press and no earlier,
and the recorder drives that same engine for every taped frame, so a model that
disagreed with PuzzleScript could not produce a WIN episode.

Optimal-action sets
-------------------
Every recorded step ships the complete set of equally-shortest presses, read
straight off the field (`_Board.plan`). Nothing is ever labelled by inference,
and no step ships unlabelled (the always-emit-optimal-targets rule). The tie
rate here is low -- 6 of the 156 presses -- and that is a fact about the boards
rather than a shortcoming: seven of the nine are three-row corridors where a walk
has one shortest route, and the mixes on top of that have to happen in a forced
order. Every one of the six is certified on the interpreter by `--verify`'s third
stage, so the few ties there are are not taken on the model's word.

Checks
------
``--fuzz`` asserts `_Board` reproduces the interpreter EXHAUSTIVELY: every one of
the 388 942 reachable states across the nine levels is seated into the engine
grid (`_seat`) and all four presses are compared, resulting position and win flag
alike. That is 1 555 768 transitions and it is not a sample -- it is a proof of
the model over exactly the set the field is computed on. All agree. It is also
the slowest check here by an order of magnitude (~20 min), which is why it is
opt-in.

Coverage counters are printed per outcome, because "the model agrees" means
nothing if the run never mixed anything, and here they earn their keep: over the
whole reachable space the counts are 957 680 walks, 498 111 refused presses,
91 463 plain pushes and only 8514 mixes -- and those mixes are R+B, R+G and G+B
ONLY. **The three white-producing rules (R+C, G+M, B+Y) are unreachable in every
shipped level**, because each needs a Cyan, Purple or Yellow standing on the
board and no level makes one that survives to be pushed again. So the fuzz
reports them as a named gap rather than a zero in a column, and they are covered
instead by ``--selfcheck``, which builds all twelve mixes (six rules x both
written orders) by hand. Between the two, every rule in the game is exercised
against the interpreter.

``--selfcheck`` asserts the five mechanic statements above on hand-built boards,
including everything the levels never reach: the three white mixes, all ten
overlapping no-op pairs, a mix resolving on top of a non-matching target, and the
``PRRB`` non-chain.

``--verify`` is the independent check of the SEARCH -- the part the fuzz cannot
reach, since it only ever exercises `step`. Three parts, in increasing
independence:

  * the backward field recomputed WITHOUT the predecessor map, by scanning the
    whole reachable set once per BFS layer. This is what certifies the
    predecessor bookkeeping in `field`.
  * the double-entry formula: a forward BFS from the start must put the nearest
    winning state at exactly the plan length, and step ``i``'s optimal set must
    equal ``{a : fwd(next) == i + 1 and back(next) == d* - i - 1}``.
  * every LABELLED press spliced into a variant plan (prefix + that press + a
    fresh optimal continuation) and replayed on the real INTERPRETER, which must
    win on press ``d*`` and not before. So the tie labels this generator ships
    are certified by PuzzleScript, not by `_Board`.

``--symmetry`` proves on the interpreter that the 8-element presentation group is
an exact symmetry of the mechanic, which is what entitles the game to
`PuzzleScriptAdapter._FLIP_GAMES`. Unlike ps:one_way_street and ps:l_a_s_e_r
there is nothing to relabel -- a crate's colour is not a direction -- so the
transform moves cells only.

``--audit`` checks that every cell COMPOSITION the game can produce is distinct
at every cell size in use. It reports clean on the shipped art and no
``data/puzzlescript_games/Palette.txt`` change was needed, which is worth
recording because it is unusual in this family. Three things happen to be right:
the seven crate colours land on seven distinct ARC indices (blue 9, red 8,
purple 15, green 14, yellow 11, cyan 10, white 5); the target sprite is a diamond
that occupies EXACTLY the four holes in the crate's ring, so a crate on a
matching target renders as a solid block and a crate on a mismatched one shows
the target's colour through its middle; and no level renders at cell_px 3 or 4,
where that middle row would be dropped by the renderer's sampling (the boards are
7-11 wide, giving 5, 6, 7 and 9 px). The one collision is that White, the Player
and TWhite are all ``Black`` -> ARC index 5, but their sprites (ring, X, diamond)
differ everywhere it matters: the closest pair still reads 10.7 px/cell apart at
the smallest cell size.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: rotation_k in {0,1,2,3} plus an independent horizontal and
vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to). 9 levels x 16
presentations = 144. There is deliberately no colour augmentation (the game is
not in ``_RECOLOR_GAMES`` and must not be): which crate mixes with which IS the
puzzle, and a recolor that relabelled the palette would break the R/G/B
arithmetic the frames are supposed to teach.

The expert plan is therefore seed-independent: solved once per level, cached to
``data/palette_plans.json``, and replayed per seed with that seed's remapped
screen actions.

Usage (run from the repo root):
    python solvers/generate_palette_training.py \
        --episodes 200 --out data/training_multi_level/palette

    python solvers/generate_palette_training.py --plans
    python solvers/generate_palette_training.py --selfcheck
    python solvers/generate_palette_training.py --fuzz
    python solvers/generate_palette_training.py --verify
    python solvers/generate_palette_training.py --symmetry
    python solvers/generate_palette_training.py --audit
"""

from __future__ import annotations

import itertools
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame                # noqa: E402
from solvers.common.ps_astar import PSAStarSolver, PSExpert, Plan      # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Palette"

#: Disk cache of each level's start plan AND its optimal-action sets. The fields
#: are seed-independent and cheap (~0.9s together), but every shard of
#: `parallelize_generator` would otherwise repeat them on every core. Delete the
#: file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "palette_plans.json"

#: Engine direction -> (dr, dc). The four moves are the whole action space
#: (``noaction`` is in the prelude, so ACTION5 is bound to nothing).
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}
#: Fixed iteration order, so a plan (and its tie sets) is reproducible.
_ORDER: tuple[str, ...] = ("up", "down", "left", "right")

#: Crate colour -> its set of primary lights as a bitmask (R=1, G=2, B=4).
#: Mixing is DISJOINT UNION and nothing else; the game's target objects are the
#: same names with a ``t`` prefix. Masks are 1..7, so a cell's colour fits in
#: three bits and 0 means "no crate" -- which is what `_Board` packs.
_MASK: dict[str, int] = {
    "red": 1, "green": 2, "blue": 4,
    "yellow": 3, "purple": 5, "cyan": 6,
    "white": 7,
}
#: mask -> object name, for seating a model state back into the engine grid.
_UNMASK: dict[int, str] = {m: n for n, m in _MASK.items()}

#: Refuse to close a reachable set larger than this. It is a MEMORY budget, not
#: a time one -- each state is a dict entry plus its predecessor list -- and it
#: exists so an edited level that opens a board up fails loudly instead of
#: swapping. The largest level here closes at 33 498 states.
STATE_CAP = 4_000_000


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """Palette over the level's free cells.

    A state is ``(packed, player_index)``. ``packed`` holds each free cell's
    3-bit colour mask at bit ``3 * i`` -- 0 for "no crate", 1..7 for the seven
    crate colours -- and the player is an index into the same list. Walls are
    not represented at all: they are exactly the cells absent from ``free``, so
    "off the board" and "into a wall" are the same lookup miss.

    Targets never move, are never consumed and never block anything, so they are
    not part of the state at all; they live in the pair ``(tsel, tval)`` that
    `win` tests with one mask-and-compare.

    `step` is the game; `field` inverts it.
    """

    def __init__(self, walls, targets, crates, player, h, w):
        self.h, self.w = h, w
        self.free = [(r, c) for r in range(h) for c in range(w)
                     if (r, c) not in walls]
        self.idx = {cell: i for i, cell in enumerate(self.free)}
        self.n = len(self.free)
        self.targets = tuple(sorted(targets.items()))
        # ``win`` is ``(packed & tsel) == tval``: tsel is 0b111 in every target
        # cell's slot, tval the colour each of them demands.
        self.tsel = self.tval = 0
        for cell, mask in self.targets:
            self.tsel |= 7 << (3 * self.idx[cell])
            self.tval |= mask << (3 * self.idx[cell])
        packed = 0
        for cell, mask in crates.items():
            packed |= mask << (3 * self.idx[cell])
        self.start = (packed, self.idx[player])
        self.nxt = {d: [self.idx.get((r + dr, c + dc), -1)
                        for (r, c) in self.free]
                    for d, (dr, dc) in _DELTA.items()}

    @classmethod
    def read(cls, eng, ids) -> "_Board":
        """Build the model from the interpreter's current grid."""
        wall_ids, crate_mask, tgt_mask, player_ids = ids
        walls, targets, crates, player = set(), {}, {}, None
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & wall_ids:
                    walls.add((r, c))
                    continue            # a wall cell can hold nothing else
                if cell & player_ids:
                    player = (r, c)
                for obj in cell:
                    if obj in crate_mask:
                        crates[(r, c)] = crate_mask[obj]
                    elif obj in tgt_mask:
                        targets[(r, c)] = tgt_mask[obj]
        return cls(walls, targets, crates, player, eng.height, eng.width)

    def win(self, state) -> bool:
        """``All TRed on Red`` and its six siblings: every TARGET cell carries a
        crate of its own colour. Note the direction -- spare crates anywhere else
        are free, and a level with two TReds needs two Reds."""
        return state[0] & self.tsel == self.tval

    # -- dynamics -------------------------------------------------------------
    def step(self, state, direction):
        """One press. Returns the new state, or ``state`` itself when the
        interpreter would cancel the turn.

        The whole rule set, in the order PuzzleScript applies it. The push rule
        gives the player's force to the ONE crate it is walking into and stops
        there (re-firing ``[ > Player | Crate ]`` changes nothing, so the force
        never reaches a second crate). Then the mixing rules look at that single
        moving crate and the cell it is entering. Then the forces resolve, with
        Player / Wall / Crate all on one collision layer.

        So there are exactly four outcomes:

          * the cell ahead is a wall or off the board -- refused;
          * it is empty -- the player walks;
          * it holds a crate and the cell beyond it is free -- an ordinary push;
          * it holds a crate and the cell beyond holds a crate whose colour set
            is DISJOINT -- the mover is consumed, the far crate becomes the
            union and stays put, and the player advances into the vacated cell.

        Anything else (a crate against a wall, or against a crate it overlaps in
        colour) is a total no-op: the crate cannot move, so neither can the
        player behind it.

        Both moving branches are the same arithmetic -- the mover's mask leaves
        cell ``x`` and is ADDED to cell ``y`` -- because a disjoint union is a
        sum on non-overlapping bits, and an empty cell is the union with 0."""
        packed, p = state
        nxt = self.nxt[direction]
        x = nxt[p]
        if x < 0:
            return state                        # wall / off the board
        cx = (packed >> (3 * x)) & 7
        if not cx:
            return (packed, x)                  # a bare walk
        y = nxt[x]
        if y < 0:
            return state                        # the crate is against a wall
        cy = (packed >> (3 * y)) & 7
        if cx & cy:
            return state                        # overlapping colours: no rule
        return (packed + (cx << (3 * y)) - (cx << (3 * x)), x)

    # -- the exact distance-to-win field --------------------------------------
    def field(self, cap: int = STATE_CAP) -> tuple[dict, dict]:
        """``(dist, fwd)`` -- distance to the win, and distance from the start,
        for every state reachable from the level start.

        A forward closure that records predecessors as it goes, then a backward
        BFS from every reachable WINNING state over those predecessors. Exact in
        PRIMITIVE PRESSES (the unit the agent pays) and with no canonicalisation
        applied: every shortest path out of the start stays inside the reachable
        set, so restricting the backward pass to that set cannot shorten one.

        Why not the usual reverse BFS from the win over pull-moves: mixing is
        irreversible, so its inverse would have to invent every disjoint split of
        every crate, and ``All TRed on Red`` with spare crates gives no unique
        goal configuration to seed from.

        A refused press maps a state to itself; those self-loops are dropped
        rather than recorded, so no state is ever its own predecessor.

        Dead states need no test: a crate mixed into a colour no target can use
        still enters the closure, it just never receives a distance, and `plan`
        reports the level unsolvable if the START is one of those."""
        start = self.start
        pred: dict = {}
        fwd = {start: 0}
        queue: deque = deque([start])
        while queue:
            state = queue.popleft()
            d = fwd[state] + 1
            for direction in _ORDER:
                nxt = self.step(state, direction)
                if nxt == state:
                    continue                          # refused: a self-loop
                pred.setdefault(nxt, []).append(state)
                if nxt not in fwd:
                    fwd[nxt] = d
                    queue.append(nxt)
            if len(fwd) > cap:
                raise MemoryError(
                    f"reachable set exceeded {cap} states on a "
                    f"{self.h}x{self.w} board")
        dist: dict = {}
        queue = deque()
        for state in fwd:
            if self.win(state):
                dist[state] = 0
                queue.append(state)
        while queue:
            state = queue.popleft()
            d = dist[state] + 1
            for prev in pred.get(state, ()):
                if prev not in dist:
                    dist[prev] = d
                    queue.append(prev)
        return dist, fwd

    def plan(self, dist: dict):
        """``(presses, optsets)`` from the start, or ``(None, None)``.

        Walk the field downhill. At every state the OPTIMAL SET is exactly the
        presses reaching distance ``d - 1``: a refused press leaves the state
        (and so the distance) unchanged and is never in it, and a press onto a
        state the field does not hold is a press that threw the win away."""
        state = self.start
        if state not in dist:
            return None, None
        presses, optsets = [], []
        while dist[state]:
            want = dist[state] - 1
            best = [a for a in _ORDER
                    if dist.get(self.step(state, a), -1) == want]
            presses.append(best[0])
            optsets.append(best)
            state = self.step(state, best[0])
        return presses, optsets


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class PaletteExpert(PSExpert):
    """Plans by building `_Board`'s exact field off the engine grid and walking
    it downhill; never steps the interpreter.

    `PSExpert` still owns everything around that -- the in-memory memo, the
    on-disk plan cache with its staleness check, the level scoping and the
    snapshot/restore discipline -- so the only override is `_search`. `heuristic`
    is unreachable by construction: nothing here runs A*.
    """

    directions = list(_ORDER)
    #: `_key` below is dynamic-objects-only, canonical only WITHIN a level.
    scope_by_level = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        g = self.g
        self.wall_ids = set(g.resolve_object_name("wall"))
        self.player_ids = set(self.game._engine._player_indices)
        self.crate_mask = {g.obj_name_to_idx[n]: m for n, m in _MASK.items()}
        self.tgt_mask = {g.obj_name_to_idx["t" + n]: m for n, m in _MASK.items()}
        self.crate_ids = set(self.crate_mask)
        self.dyn_ids = self.crate_ids | self.player_ids

    def _key(self, eng) -> frozenset:
        # Crates + player. Walls and targets are static per level -- no rule in
        # this game creates, destroys or moves any of them -- hence
        # ``scope_by_level``.
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    def heuristic(self, eng) -> int:
        raise AssertionError("the field is exact; no search runs here")

    def board(self, eng) -> _Board:
        return _Board.read(eng, (self.wall_ids, self.crate_mask, self.tgt_mask,
                                 self.player_ids))

    def _search(self, eng) -> list | None:
        board = self.board(eng)
        dist, _fwd = board.field()
        presses, optsets = board.plan(dist)
        return None if presses is None else Plan(presses, optsets)


class PaletteSolver(PSAStarSolver):
    game_id = "puzzlescript_palette"
    game_name = GAME_NAME
    expert_cls = PaletteExpert

    #: The longest plan is 34 presses; the rest is room for the exploration
    #: prefix and the replay after it. Stays under the adapter's 200-step
    #: per-level budget, which would otherwise flip a level to GAME_OVER
    #: mid-plan.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Build every level's field before `discover_solvable` asks for it --
        the same work either way, but it makes the one-time cost visible as
        startup rather than as a mysteriously slow first seed, and it fills the
        disk cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Seating a model state back into the interpreter
# ---------------------------------------------------------------------------

def _seat(eng, g, board: _Board, state) -> None:
    """Write ``state`` into the engine grid, so the interpreter can be asked
    what it does from a state the model reached.

    This is what makes `--fuzz` a proof rather than a sample: the reachable set
    is small enough to seat and step EVERY state, instead of random-walking and
    hoping to line a mix up. Walls and targets come from ``board`` (they are
    static), the crates and player from ``state``."""
    idx = g.obj_name_to_idx
    grid = [[{idx["wall"]} for _ in range(board.w)] for _ in range(board.h)]
    for (r, c) in board.free:
        grid[r][c] = {idx["background"]}
    for cell, mask in board.targets:
        grid[cell[0]][cell[1]].add(idx["t" + _UNMASK[mask]])
    packed, p = state
    for i, (r, c) in enumerate(board.free):
        m = (packed >> (3 * i)) & 7
        if m:
            grid[r][c].add(idx[_UNMASK[m]])
    pr, pc = board.free[p]
    grid[pr][pc].add(idx["player"])
    eng.grid = grid
    eng.height, eng.width = board.h, board.w
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


def _read_state(eng, expert, board: _Board):
    """``(packed, player_index)`` read back off the interpreter grid, or None
    when the player has vanished (which no rule here can do -- if it ever
    happens the fuzz should say so rather than crash)."""
    packed, p = 0, None
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if cell & expert.player_ids:
                p = board.idx.get((r, c))
            for obj in cell & expert.crate_ids:
                i = board.idx.get((r, c))
                if i is None:            # a crate inside a wall: impossible
                    return None
                packed |= expert.crate_mask[obj] << (3 * i)
    return None if p is None else (packed, p)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with the adapter parked at level 0."""
    solver = PaletteSolver()
    game = solver.make_game(0)
    return solver, game, PaletteExpert(game)


def _report() -> int:
    """Per-level board size, piece count, plan length, tie coverage and state
    count -- and CERTIFY each plan by replaying it through the interpreter."""
    _solver, game, expert = _levels()
    total = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = expert.board(eng)
        if eng.check_win():
            print(f"level {level:2d}: STARTS ALREADY WON")
            bad += 1
            continue
        t = time.time()
        dist, fwd = board.field()
        presses, optsets = board.plan(dist)
        took = time.time() - t
        if presses is None:
            print(f"level {level:2d}: UNSOLVABLE")
            bad += 1
            continue
        # Certification: the interpreter must win on the LAST press and no
        # earlier (an earlier win would mean the plan is not shortest).
        won_at = None
        for i, direction in enumerate(presses):
            eng.step(direction)
            if eng.check_win():
                won_at = i
                break
        ok = won_at == len(presses) - 1
        bad += not ok
        total += len(presses)
        ties = sum(1 for s in optsets if len(s) > 1)
        crates = sum(1 for i in range(board.n)
                     if (board.start[0] >> (3 * i)) & 7)
        room = "ok" if len(presses) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {board.h:2d}x{board.w:2d} free={board.n:2d} "
              f"{crates} crates {len(board.targets)} targets, "
              f"{len(presses):3d} presses (budget {game._max_steps}, {room}), "
              f"{ties:3d} tie steps ({ties / len(presses):3.0%}), "
              f"{len(fwd):6d} reachable ({len(dist):6d} scored) in "
              f"{took:5.2f}s, "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    print(f"total {total} presses over {game.n_levels} levels")
    return 0 if not bad else 1


def _verify() -> int:
    """Independent check of the SEARCH -- the part `--fuzz` cannot reach.

    The fuzz only ever exercises `step`, the forward rule. Everything the plans
    actually rest on is downstream of that: the predecessor bookkeeping in
    `field`, the backward pass, and the claim that a labelled press is genuinely
    one of the shortest. Three checks, in increasing independence:

      * **the backward field, recomputed without the predecessor map.** One scan
        of the whole reachable set per BFS layer, keeping the states with a
        successor at distance ``d - 1``. Slower (O(states x d*)) and completely
        separate bookkeeping; it must agree on every state.
      * **the double-entry formula.** A forward BFS from the start must put the
        nearest winning state at exactly the plan length, and step ``i``'s
        optimal set must be ``{a : fwd(next) == i + 1 and back(next) ==
        d* - i - 1}`` -- which uses both passes and so fails if either is wrong.
      * **every labelled press replayed on the INTERPRETER.** Each press in each
        optimal set is spliced into a variant plan (the prefix, that press, then
        a fresh optimal continuation from wherever it lands) and driven through
        the real engine, which must reach ``check_win`` on press ``d*`` and not
        before. This one leaves `_Board` behind entirely: it is PuzzleScript
        certifying the tie labels this generator ships.
    """
    _solver, game, expert = _levels()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        layout = game._game.levels[level]
        board = expert.board(eng)
        back, fwd = board.field()
        presses, optsets = board.plan(back)
        star = len(presses)
        notes = []

        # 1. the backward field again, by layer scan, with no predecessor map.
        t = time.time()
        scan = {s: 0 for s in fwd if board.win(s)}
        frontier = set(scan)
        d = 0
        while frontier:
            d += 1
            frontier = {s for s in fwd if s not in scan
                        and any(board.step(s, a) in frontier for a in _ORDER)}
            for s in frontier:
                scan[s] = d
        if scan != back:
            notes.append(
                f"LAYER SCAN differs on {len(set(scan) ^ set(back))} states")
        scan_t = time.time() - t

        # 2. the double-entry formula, over both passes.
        wins = [fwd[s] for s in fwd if board.win(s)]
        if min(wins) != star:
            notes.append(f"LENGTH {min(wins)} != {star}")
        state = board.start
        for i, direction in enumerate(presses):
            want = [a for a in _ORDER
                    if fwd.get(board.step(state, a), -1) == i + 1
                    and back.get(board.step(state, a), -1) == star - i - 1]
            if want != optsets[i]:
                notes.append(f"step {i}: {optsets[i]} != {want}")
            state = board.step(state, direction)

        # 3. every labelled press, spliced into a variant plan and replayed on
        #    the real interpreter.
        replays = 0
        state = board.start
        for i, alts in enumerate(optsets):
            for alt in alts:
                nxt = board.step(state, alt)
                tail, _ = _variant(board, nxt).plan(back)
                variant = presses[:i] + [alt] + tail
                replays += 1
                eng.load_level(layout)
                won_at = None
                for j, direction in enumerate(variant):
                    eng.step(direction)
                    if eng.check_win():
                        won_at = j
                        break
                if won_at != star - 1 or len(variant) != star:
                    notes.append(f"step {i} press {alt}: interpreter won at "
                                 f"{won_at} of {len(variant)}")
            state = board.step(state, presses[i])

        bad += len(notes)
        print(f"level {level:2d}: {len(fwd):6d} reachable, layer scan "
              f"{scan_t:6.2f}s, d*={star:3d}, {replays:3d} labelled presses "
              f"replayed: {'AGREES' if not notes else '; '.join(notes[:3])}")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _variant(board: _Board, state):
    """``board`` with its start moved to ``state`` -- so `_Board.plan` can be
    reused to build an optimal CONTINUATION from mid-plan without recomputing
    the field."""
    clone = object.__new__(_Board)
    clone.__dict__ = dict(board.__dict__)
    clone.start = state
    return clone


def _fuzz() -> int:
    """Assert `_Board` reproduces the interpreter, EXHAUSTIVELY.

    Random play would be the usual shape of this check and it would be weak
    here: from a level start it wanders, and lining two crates of DISJOINT
    colour up so a mix fires is exactly what it is bad at -- the game's whole
    mechanic would go essentially unexercised. But the reachable set is small
    (34k states at the worst level), so instead every reachable state is seated
    into the engine grid (`_seat`) and all four presses are compared against the
    model, resulting position AND win flag. That is not a sample of the space,
    it is the space.

    Coverage counters are printed per OUTCOME, and the six mix counters are the
    point of them: a run that reported agreement having never merged two crates
    would have checked a sokoban, not this game."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    total = bad = 0
    kinds: dict[str, int] = {}
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        _dist, fwd = board.field()
        seen: dict[str, int] = {}
        for state in fwd:
            for direction in _ORDER:
                mine = board.step(state, direction)
                _seat(eng, g, board, state)
                eng.step(direction)
                theirs = _read_state(eng, expert, board)
                total += 1
                if theirs != mine:
                    bad += 1
                    if bad < 5:
                        print(f"  level {level} MISMATCH from {state} "
                              f"{direction}: model {mine} engine {theirs}")
                if eng.check_win() != board.win(mine):
                    bad += 1
                    if bad < 5:
                        print(f"  level {level} WIN MISMATCH from {state} "
                              f"{direction}")
                kind = _kind(board, state, mine, direction)
                seen[kind] = seen.get(kind, 0) + 1
        for k, v in seen.items():
            kinds[k] = kinds.get(k, 0) + v
        print(f"level {level:2d}: {len(fwd):6d} states x 4 presses = "
              f"{4 * len(fwd):6d} transitions  "
              + "  ".join(f"{k}={seen.get(k, 0)}" for k in _KINDS
                          if seen.get(k)))
    print("coverage: " + "  ".join(f"{k}={kinds.get(k, 0)}" for k in _KINDS))
    # A zero here is a real gap and must not read as a pass: the three
    # white-producing mixes need a Cyan, a Purple or a Yellow ON THE BOARD, and
    # no shipped level ever makes one that survives, so nothing reachable
    # exercises them. They are covered instead by `--selfcheck`, which builds
    # all twelve mixes by hand. Print it rather than leave it to be read off a
    # column of numbers.
    missing = [k for k in _KINDS if k not in ("refused", "walk", "push")
               and not kinds.get(k)]
    if missing:
        print(f"  NOT REACHABLE in any level: {', '.join(missing)} "
              f"-- these rules are covered by --selfcheck only")
    print(f"{total} transitions: "
          f"{'model matches the interpreter' if not bad else f'{bad} MISMATCHES'}")
    return 0 if not bad else 1


#: The outcome labels `_kind` can return, in report order.
_KINDS: tuple[str, ...] = ("refused", "walk", "push",
                           "R+B", "R+G", "G+B", "R+C", "G+M", "B+Y")


def _kind(board: _Board, state, nxt, direction) -> str:
    """Classify a transition for the fuzz's coverage counters. A mix is named by
    the two colour SETS that combined, so the six labels are the game's six
    rules (in both of their written orders)."""
    if nxt == state:
        return "refused"
    packed, p = state
    x = board.nxt[direction][p]
    cx = (packed >> (3 * x)) & 7
    if not cx:
        return "walk"
    y = board.nxt[direction][x]
    cy = (packed >> (3 * y)) & 7
    if not cy:
        return "push"
    lo, hi = sorted((cx, cy))
    return {(1, 4): "R+B", (1, 2): "R+G", (2, 4): "G+B",
            (1, 6): "R+C", (2, 5): "G+M", (3, 4): "B+Y"}[(lo, hi)]


# ---------------------------------------------------------------------------
# Mechanic self-check
# ---------------------------------------------------------------------------

#: Level-file legend letters used by `_selfcheck`'s hand-built boards.
_SHORT = {"P": "player", "R": "red", "G": "green", "B": "blue", "C": "cyan",
          "M": "purple", "Y": "yellow", "W": "white", "#": "wall",
          "1": "tblue", "2": "tred", "3": "tgreen", "4": "tpurple",
          "5": "tcyan", "6": "tyellow", "7": "twhite"}


def _selfcheck() -> int:
    """Audit, against the interpreter, the five mechanic statements the module
    docstring makes -- including the cases the nine levels never reach.

    `--fuzz` already proves the MODEL equals the interpreter over every
    reachable state, which subsumes most of this. What it cannot do is state the
    claims in a form a reader can check: these are the sentences the solver was
    designed from, written as boards. They also cover configurations no level
    contains at all (a White crate, a mix resolving on a mismatched target),
    where a future edit to the .txt would otherwise break something silently.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    n2i = g.obj_name_to_idx
    bad = 0
    height, width = 7, 16

    def fail(msg: str) -> None:
        nonlocal bad
        bad += 1
        print(f"  VIOLATION: {msg}")

    def build(row: str, extra=()) -> None:
        grid = [[{n2i["background"]} for _ in range(width)]
                for _ in range(height)]
        for c in range(width):
            grid[0][c].add(n2i["wall"])
            grid[height - 1][c].add(n2i["wall"])
        for r in range(height):
            grid[r][0].add(n2i["wall"])
            grid[r][width - 1].add(n2i["wall"])
        for c, ch in enumerate(row):
            if ch != ".":
                grid[3][c + 1].add(n2i[_SHORT[ch]])
        for (c, ch) in extra:
            grid[3][c].add(n2i[_SHORT[ch]])
        eng.grid = grid
        eng.height, eng.width = height, width
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    def cells(name: str) -> set:
        return {(r, c) for r, row in enumerate(eng.grid)
                for c, cell in enumerate(row) if n2i[name] in cell}

    def snap() -> list:
        return [[set(cell) for cell in row] for row in eng.grid]

    # 1. mixing is disjoint union, it lands on the STATIONARY crate's cell, and
    #    the player advances into the cell the mover vacated.
    for a, b, out in (("R", "B", "purple"), ("B", "R", "purple"),
                      ("R", "G", "yellow"), ("G", "R", "yellow"),
                      ("B", "G", "cyan"), ("G", "B", "cyan"),
                      ("B", "Y", "white"), ("Y", "B", "white"),
                      ("G", "M", "white"), ("M", "G", "white"),
                      ("C", "R", "white"), ("R", "C", "white")):
        build("P" + a + b)
        eng.step("right")
        if cells(out) != {(3, 3)}:
            fail(f"{a}+{b} did not leave a single {out} at the stationary "
                 f"crate's cell")
        if cells("player") != {(3, 2)}:
            fail(f"{a}+{b}: the player did not advance into the vacated cell")

    # 2. every OVERLAPPING pair is a dead no-op -- no rule matches it at all, so
    #    the crate cannot move and neither can the player behind it.
    for a, b in (("Y", "M"), ("Y", "C"), ("M", "C"), ("W", "R"), ("R", "W"),
                 ("R", "R"), ("W", "W"), ("C", "C"), ("M", "R"), ("Y", "G")):
        build("P" + a + b)
        before = snap()
        eng.step("right")
        if eng.grid != before:
            fail(f"{a}+{b} changed the board -- the colour algebra is wrong")

    # 3. pushes never chain. PRRB is a total no-op even though R+B would mix:
    #    the second R never receives a force.
    for row in ("PRRB", "PRR.", "PRW."):
        build(row)
        before = snap()
        eng.step("right")
        if eng.grid != before:
            fail(f"{row}: a push chained through a second crate")

    # 4. a crate against a WALL blocks the whole turn, player included.
    build("P" + "R" + "#")
    before = snap()
    eng.step("right")
    if eng.grid != before:
        fail("a crate pushed into a wall did not cancel the turn")

    # 5. targets are inert scenery on their own layer: a crate crosses one, the
    #    player stands on one, and a mix resolving on one leaves it intact.
    build("PR..", extra=[(4, "1")])
    for _ in range(3):
        eng.step("right")
    if cells("tblue") != {(3, 4)} or cells("red") != {(3, 5)}:
        fail("pushing a crate across a non-matching target disturbed it")
    build("PRB", extra=[(3, "2")])
    eng.step("right")
    if cells("tred") != {(3, 3)} or cells("purple") != {(3, 3)}:
        fail("a mix resolving on a target consumed the target")
    if eng.check_win():
        fail("a purple on a TRed counted as a win")

    # 6. the win is per TARGET: parking on it is enough, and two targets of one
    #    colour need two crates.
    build("P.R", extra=[(3, "2")])
    if not eng.check_win():
        fail("a crate already parked on its own target did not count")
    build("PR2", extra=[(5, "2")])
    eng.step("right")
    if eng.check_win():
        fail("one Red satisfied two TReds")
    return bad


# -- the symmetry group -----------------------------------------------------

def _transform(k: int, mirror: bool):
    """``(cell_fn, dims_fn, direction_map)`` for one element of the 8-group:
    mirror left-right first, then turn clockwise ``k`` times.

    The direction map is DERIVED from the same linear part that moves the cells,
    so the two cannot drift apart."""
    def cell(rc, hw):
        r, c = rc
        h, w = hw
        if mirror:
            c = w - 1 - c
        for _ in range(k):
            r, c, h, w = c, h - 1 - r, w, h
        return (r, c)

    def dims(hw):
        h, w = hw
        if k % 2:
            h, w = w, h
        return (h, w)

    dmap = {}
    for name, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[name] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    return cell, dims, dmap


def _dynamic(eng, expert):
    """``(player, crates)`` off the interpreter grid -- the only things a
    presentation transform is allowed to be asserted on. Crates carry their
    COLOUR, because a transform must not merely move them, it must leave the
    same colours in the transformed cells."""
    player, crates = None, set()
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if cell & expert.player_ids:
                player = (r, c)
            for obj in cell & expert.crate_ids:
                crates.add((r, c, expert.crate_mask[obj]))
    return player, frozenset(crates)


def _symmetry() -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an exact
    symmetry of the mechanic, which is what entitles this game to
    `PuzzleScriptAdapter._FLIP_GAMES`.

    The argument from the rules is as clean as it gets in this family -- all
    thirteen are stated with the relative ``>`` force, there is no gravity, the
    win conditions name no direction, input is screen-relative, and only ONE
    object ever carries a force in a turn, so two chains can never contest a
    cell the way ps:gobble_rush's charges do. It is measured anyway, because
    that argument has been wrong before (see the `_FLIP_GAMES` comments).

    Each level's plan is replayed on all eight turned and mirrored copies of its
    own board, built by transforming the LEVEL LAYOUT so the interpreter re-runs
    its own level-start work, and every press must leave the pieces -- with their
    COLOURS -- exactly where the transform of the reference run put them.

    Unlike ps:one_way_street and ps:l_a_s_e_r there is nothing to RELABEL: no
    object in this game encodes a direction, so moving the cells is the whole
    transform. The sprites are sound under the group for the same reason and
    then some -- the crate ring, the target diamond and the player's X are each
    invariant under all eight elements, and the only asymmetric art (the wall's
    and background's noise textures) is a per-cell constant carrying no
    information."""
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng)
        dist, _fwd = board.field()
        presses, _optsets = board.plan(dist)
        layout = g.levels[level]
        hw = (len(layout), len(layout[0]))

        eng.load_level(layout)
        ref = [_dynamic(eng, expert)]
        for direction in presses:
            eng.step(direction)
            ref.append(_dynamic(eng, expert))

        notes = []
        for k, mirror in itertools.product(range(4), (False, True)):
            if (k, mirror) == (0, False):
                continue
            cell, dims, dmap = _transform(k, mirror)
            th, tw = dims(hw)
            turned = [[set() for _ in range(tw)] for _ in range(th)]
            for r, row in enumerate(layout):
                for c, objs in enumerate(row):
                    tr, tc = cell((r, c), hw)
                    turned[tr][tc] = set(objs)
            eng.load_level(turned)
            for i, direction in enumerate(presses):
                eng.step(dmap[direction])
                player, crates = ref[i + 1]
                want = (cell(player, hw),
                        frozenset(cell(x[:2], hw) + (x[2],) for x in crates))
                if _dynamic(eng, expert) != want:
                    notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                    break
        bad += len(notes)
        print(f"level {level:2d}: {len(presses):3d} presses x 7 presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print("symmetry clean -- the flip augmentation is exact" if not bad
          else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    This game is the rare one in the family that passes on its shipped art, and
    the audit is here to KEEP it that way rather than to have found a bug: the
    seven crate colours land on seven distinct ARC indices, the target diamond
    occupies exactly the four holes in the crate's ring (so a crate on its own
    target is a solid block and a crate on a mismatched one shows the target's
    colour through its middle), and the boards are 7-11 wide, giving cell sizes
    5/6/7/9 -- none of them the 3 or 4 at which the renderer's centred sampling
    drops the sprite's middle row and column, which is exactly where those holes
    are. The one real collision is that White, TWhite and the Player are all
    ``Black`` -> index 5, told apart by shape alone, so their margins are
    reported rather than merely their distinctness.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and the arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor)."""
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {"floor": (), "wall": ("wall",), "player": ("player",)}
    for name in _MASK:
        comps[name] = (name,)
        comps["t" + name] = ("t" + name,)
        comps[f"{name}_on_t{name}"] = ("t" + name, name)
        comps[f"player_on_t{name}"] = ("t" + name, "player")
    # The two mismatched stacks that carry the game's only "wrong answer"
    # signal: a crate parked on a target of another colour must not read as a
    # win, and must not read as the bare crate either.
    comps["red_on_tblue"] = ("tblue", "red")
    comps["blue_on_tred"] = ("tred", "blue")

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
        # The win frame is the one that must not merely differ but be READABLE,
        # so its margin is reported; likewise the three Black compositions.
        win_px = min(int((shots[f"{n}_on_t{n}"] != shots[n]).sum()) / (h * w)
                     for n in _MASK)
        black = min(int((shots[a] != shots[b]).sum()) / (h * w)
                    for a, b in itertools.combinations(
                        ("player", "white", "twhite"), 2))
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} compositions, "
              f"crate-on-its-target reads >={win_px:4.1f}px/cell against the "
              f"bare crate, the Black trio >={black:4.1f}px/cell apart, "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--selfcheck" in sys.argv:
        violations = _selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(PaletteSolver.main())
