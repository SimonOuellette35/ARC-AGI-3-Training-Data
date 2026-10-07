"""Generate Phase-1 training data for the PuzzleScript game
ps:esl_puzzle_game_challenge_mode ("ESL Puzzle Game -- CHALLENGE MODE
アダムのパズルゲーム", A.R.Nakama) -- the sokoban where one of the three crate
colours can only be PULLED, three in a line annihilate, and a crate parked on a
door turns into a wall for good.

The harness -- the macro A* over the real interpreter, the rotation contract, the
trajectory recorder, the disk plan cache and the `BaseSolver` plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: the macro set (pushes AND pulls), the walk graph the pull
rule makes DIRECTED, and a heuristic that knows a crate has three different ways
to stop being a problem.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_esl_puzzle_game_challenge_mode",
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
exactly. Every step also carries the full set of equally-optimal presses.

The game
--------
Ten boards (the author's stages 11-20) built out of five rules::

    [ >  Player | CrateR ] -> [  >  Player | > CrateR  ]
    [ >  Player | CrateB ] -> [  >  Player | > CrateB  ]
    [ <  Player | CrateP ] -> [  <  Player | < CrateP  ]

    late [ CrateR | CrateR | CrateR | CrateR | CrateR ] -> win
    late [ CrateX | CrateX | CrateX ] -> [ | |]          (X = R, B, P)
    late [ CrateX TargetXWall ]       -> [ Wall ]

with ``All CrateX on TargetX`` for each of the three colours as the win
condition. Four consequences, and every level is built on one of them:

* **Red and blue are pushed; GREY IS PULLED.** ``[ < Player | CrateP ]`` fires
  when the player steps AWAY from an adjacent grey, so a grey follows you. You
  can never shove one. The rule does not care *why* you moved, so it also fires
  on an ordinary walk: **crossing the cell behind a grey drags it**, and that is
  what makes the walk graph directed (see `ESLExpert._analyze`).
* **Three of a colour in a line annihilate**, on either axis, checked after
  every turn. That is a way to *delete* crates, and since ``All CrateX on
  TargetX`` is vacuously true with no CrateX left, on some boards it is the only
  win: stage 19 has three reds, three greys and no red or grey target at all.
* **A crate on a matching WALL-target becomes a Wall.** The crate is consumed
  (so it also satisfies the win condition by disappearing) and the square is
  blocked forever. Doors are one-way and permanent, which is why half the hints
  in the file are about ORDER ("finish red second", "the gray is last", "don't
  close doors until you have to").
* **Five reds in a line wins outright**, ahead of the win conditions. It is the
  whole of stage 15, and it is a tighter needle than it looks: four in a line
  contain three in a line, which annihilate first, so the fifth red has to
  arrive PERPENDICULARLY into the gap of a 2-gap-2 column.

ACTION5 is bound to nothing, so the four directions are the whole action space.

The expert
----------
`PSPushExpert`'s macro A* -- successors are ``walk to the working cell, then act``
-- over the REAL interpreter, with four game-specific pieces:

* **The macro set** (`ESLExpert._analyze`) is pushes *and* pulls. A push is the
  familiar one: stand at ``q - m`` and press ``m``. A pull is its mirror: stand
  at ``q + m`` (beside the grey) and press ``m`` (away from it), which walks the
  player to ``q + 2m`` and drags the grey to ``q + m``. Both are prefiltered for
  legality exactly, so no macro the engine would refuse is ever queued.
* **The walk graph is DIRECTED.** The step ``u -> u + m`` is a *bare* walk only
  when ``u - m`` holds no grey; otherwise it is a pull, which is a macro, not a
  walk. So reachability is a directed BFS, and `_walk_distances` (which
  `PSPushExpert.annotate_walks` reads to label tie sets) is a REVERSE BFS over
  the same directed edges rather than the base's undirected one.
* **`_region_key` is the exact state.** The base merges states by the player's
  walkable REGION, which is only sound for an undirected walk graph and a static
  board; here the graph is directed AND rules create walls and consume targets.
  Dedup is on the full non-background grid instead -- see
  [[entrepotphage-demake-solver]] for the same call made for a different reason.

* **The search is a LADDER, then a beam.** `ESLExpert.weights` is tried in turn
  and the first plan inside the node cap is kept; the level none of them reaches
  falls through to `_beam`. And `PSPushExpert.exact_goal_test` is on, without
  which the macro A* returns the first win it *generates* rather than the
  shortest -- level 0 came back at 51 against a primitive BFS's proven 48.

THE HEURISTIC (`ESLExpert.heuristic`) is where the game's two wins meet. It is
the cheaper of

  * satisfying the three win conditions, which costs at least the sum over the
    colours of that colour's own cheapest route -- an exact min-cost ASSIGNMENT
    of its crates to its sinks (ring targets and wall-targets alike, since a
    crate is equally finished by being parked or eaten), or ANNIHILATING some
    three of them on the cheapest line of three cells and assigning the rest;
  * lining up five reds, for the boards that have five.

plus one player WALK, because ``g`` counts the player's steps and everything
above counts the crates'. Distances come from per-cell BFS tables over that
colour's own move graph -- a pushed crate needs the cell BEHIND it free, a
pulled one the *second* cell ahead, so red/blue and grey get different graphs --
cached per WALL LAYOUT, which changes only when a door closes, so the whole
heuristic is dictionary lookups on all but a handful of nodes.

Every relaxation in there only makes the number SMALLER (other crates are
ignored, and the three-cell matching lets each cell take its nearest crate), so
the estimate is an admissible lower bound and rung 1 of the ladder really does
prove its plans shortest. The two boards a primitive BFS can settle exhaustively
(0 and 1, at 141k and 5k states) agree: 48 and 34, both ways.

A board where no colour has a route left scores `DEAD_COST` and `dead()` drops
it un-expanded -- the usual sokoban corner test, generalised to a game where a
corner is survivable if two friends of your colour can reach the cells beside
you, and where the fatal move is more often annihilating a colour down to two
crates with nowhere left to put them.

The levels
----------
All ten are the author's own, in shipped order (his stages 11-20; the file is
the CHALLENGE MODE half of the game, so it opens at what is already stage 11).
``plan`` is the expert's press count, ``budget`` the level's step limit in
`games/ps:esl_puzzle_game_challenge_mode`, ``search`` how the plan was found and
``ties`` the share of steps carrying more than one optimal press::

    level  size    R  B  P  plan  budget  search   ties  time   the author's hint
    0      8x 9    2  1  -   48     200   A* w=1    19%   11s   push the middle
                                                                red down, then
                                                                back up
    1      7x 8    1  -  1   34     200   A* w=1    12%    0s   you can only pull
                                                                the gray blocks
    2      7x13    1  2  1   64     200   A* w=1    12%   58s   finish red second
    3      8x10    2  -  2   55     200   A* w=1     7%   26s   the gray will move
                                                                up and down
    4      7x 7    5  3  -   28     200   A* w=2    25%  316s   make 5 red in a
                                                                row.  Blues first
    5      8x 9    2  2  -   52     200   A* w=1    31%    8s   make a hallway
    6      8x 7    2  2  3   35     200   A* w=1     9%    0s   the red is first
    7      9x13    1  1  1   98     294   A* w=1    17%   13s   the gray is last
    8     10x 8    3  1  3   65     200   A* w=1    18%  141s   move and then
                                                                replace the blue
                                                                block
    9     12x12    3  1  2  157     471   beam      21% 1808s   don't close doors
                                                                until you have to

Every level is engine-verified solvable -- the ``plan`` column is a plan the
interpreter walked to a WIN -- and none starts already won. Levels 0-3 and 5-8
are proved SHORTEST (rung 1 of the ladder is an admissible-heuristic A* with the
goal tested on pop). Level 4's 28 is shortest too, though rung 2 is what
returned it inside the node cap: an uncapped rung-1 run proves the same 28.
Level 9's 157 is a beam plan and is only known to WIN -- see `ESLExpert._beam`.

The two independent checks on all of that: levels 0 and 1 are small enough for
an exhaustive PRIMITIVE breadth-first search over the interpreter (141k and 5k
states), and it returns exactly 48 and 34; and ``--fuzz`` verifies the macro
model against move-by-move replay on every board.

The whole set costs ~34 minutes to search once, almost all of it on levels 4 and
9, and is written to ``data/esl_puzzle_game_plans.json`` so no shard of
`parallelize_generator` re-derives it. Warm, a run starts in about a second.
Delete that file to re-search.

Optimal-action sets
-------------------
`PSPushExpert.annotate_walks` (via ``annotate``) labels every step: a push or a
pull with itself, and each step of a walk with every direction that keeps it on a
shortest route to the same working cell -- 7% to 31% of the steps per level.
Ties are the common case in the walks, which are most of every solution, and the
walk-step legality test is the same directed one the macro set is built from
(`ESLExpert._walk_step_ok`), so a direction that would drag a grey is never
offered as a free alternative. No step ever ships unlabelled (the
always-emit-optimal-targets rule).

The label a push gets is itself alone, which understates the ties on a board
where two crates of a colour are interchangeable. `PSPushExpert.exact_optsets`
is the harness's answer to that and is deliberately NOT used here: it re-solves
from every candidate press, and one solve on these boards runs from 10s to five
minutes. See `ESLExpert._search`.

Augmentation
------------
The engine state after reset is identical for every seed (the levels are fixed
ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation plus an independent horizontal and vertical
flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game is in). Every rule here is
stated with a RELATIVE force (``>`` push, ``<`` pull) and expands to all four
directions, the annihilation rule reads both axes alike, there is no gravity and
the input is screen-relative -- so a flip is an exact symmetry of the mechanic,
and no sprite is chiral enough for a mirror to land one object's art on
another's. 10 levels x 16 presentations = 160. There is deliberately no colour
augmentation: the three crate colours are not decoration, they are which rule
moves the crate and which target accepts it.

The palette and sprite fixes
----------------------------
Two of the three shapes `--audit` exists to catch were in the original art, and
both are in the class where the object is not hidden but the STACK is:

* The player is opaque across the whole middle of its 5x5, and both kinds of
  target were drawn inside that middle -- so a player standing on a goal was
  pixel-identical to a player standing on grass, at every cell size the ten
  boards render at. The goals now carry four CORNER studs and the doors four
  SIDE studs, which are exactly where the player sprite is transparent.
* At ``cell_px = 4`` (levels 2 and 7, the two widest boards) a 5x5 sprite is
  sampled at rows and columns {0, 1, 3, 4} -- the middle is simply not read --
  so ``TargetR`` (a hollow ring) and ``TargetRWall`` (a solid block) rendered
  identically, and a goal was indistinguishable from a door that eats crates.
  The two stud patterns separate them at that size too.

No rule, legend or level was touched by either edit. ``--audit`` is the check:
it renders every cell COMPOSITION the game can show -- floor, wall, player, each
crate, each goal, each door, a crate on its goal, a crate on the WRONG colour's
goal, the player on each goal -- at every cell size the levels use, and asserts
they are pairwise distinct.

Usage (run from the repo root):
    python solvers/generate_esl_puzzle_game_training.py --episodes 200 \
        --out data/training_multi_level/esl_puzzle_game

    python solvers/generate_esl_puzzle_game_training.py --plans   # level report
    python solvers/generate_esl_puzzle_game_training.py --audit   # render audit
    python solvers/generate_esl_puzzle_game_training.py --fuzz    # macro fidelity
"""

from __future__ import annotations

import itertools
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import _render_frame                 # noqa: E402
from solvers.common.ps_astar import (DEAD_COST, PSAStarSolver, Plan,    # noqa: E402
                                     PSPushExpert, _DELTA, restore,
                                     snapshot)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "ESL_Puzzle_Game_Challenge_Mode"

#: Disk cache of each level's start plan AND its optimal-action sets. The
#: searches are seed-independent but cost minutes of interpreter steps, which
#: every shard of `parallelize_generator` would otherwise repeat on every core.
#: Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "esl_puzzle_game_plans.json"

#: Crate object -> (its ring target, its wall target). The three colours differ
#: only in which rule moves them (see `ESLExpert.pull_names`), so everything else
#: in this file is a loop over this table.
COLOURS: dict[str, tuple[str, str]] = {
    "crater": ("targetr", "targetrwall"),
    "crateb": ("targetb", "targetbwall"),
    "cratep": ("targetp", "targetpwall"),
}

#: Length of the line of reds that wins outright (`late [R|R|R|R|R] -> win`).
WIN_RUN = 5
#: Length of the line that annihilates (`late [X|X|X] -> [ | |]`).
POP_RUN = 3


class ESLExpert(PSPushExpert):
    """Macro A* over the real interpreter for a push/pull/match-3 sokoban.

    See the module docstring for the mechanic and for what each override is
    for. The short version: `_analyze` emits pulls as well as pushes and builds
    its walk tree over a DIRECTED graph (stepping away from a grey is a pull,
    not a walk), `_region_key` dedups on the exact board because rules here
    create walls, and `heuristic` prices the board at the cheaper of its two
    wins -- tidy every colour away, or line up the five reds that end it.
    """

    pushable_names = tuple(COLOURS)
    blocker_names = ("wall",)

    #: The colour the player can only PULL. Its macros are the mirror of a push
    #: and its move graph is a different graph (see `_move_edge`).
    pull_names = ("cratep",)

    #: Label every step with its optimal SET (most steps are walks, whose axis
    #: order is free) and keep each level's start plan on disk.
    annotate = True
    plan_cache_path = PLAN_CACHE

    #: Prove the plan shortest instead of taking the first win generated. The
    #: macros here range from 1 to a dozen presses, so the difference is real:
    #: level 0 came back at 51 without it and at 48 -- the length a primitive
    #: BFS proves shortest over 141k states -- with it.
    exact_goal_test = True

    #: Weight LADDER: try each in turn and keep the first plan that comes back
    #: inside the node cap. Rung 1 is a proof of shortest (admissible heuristic,
    #: goal tested on pop); the later rungs are not, they just finish. Eight of
    #: the ten boards settle on rung 1, stage 15 settles on rung 2 (and returns
    #: the same 28 presses an uncapped rung-1 run proves shortest), and stage 20
    #: reaches no rung at all and falls through to `_beam`. Same idea as
    #: EntrepotPhage's ladder: pay for optimality wherever it is affordable
    #: rather than giving it up globally.
    #:
    #: Three rungs, not five. Every weight measured -- 2, 3, 5 and 10 -- fails on
    #: stage 20 by the same margin, so the rungs between them buy nothing there
    #: and cost ~200s each on the way to the beam that does work.
    weights: tuple[int, ...] = (1, 2, 5)

    #: Beam fallback, for the level no rung of the ladder reaches. Stage 20's
    #: heuristic is FLAT -- weights 1, 2, 3, 5 and 10 all burn the same node cap
    #: without returning, because on a board whose six crates each have their own
    #: door on the far side of a maze the estimate barely moves for the dozen
    #: macros in the middle of the solution, and an inflated flat estimate ranks
    #: nothing. A beam spends its budget on breadth at each depth instead and
    #: only uses the heuristic to break ties, which is all a flat one is good
    #: for. See `PSBeamExpert` for the same argument at length.
    beam_width: int = 400
    beam_depth: int = 60
    beam_node_cap: int = 400_000

    #: Walls are created by rules and targets consumed by them, so the board is
    #: NOT static within a level: the key has to carry all of it.
    scope_by_level = False

    def _search(self, eng) -> list | None:
        """The weight LADDER, then the BEAM, then the annotation pass.

        Annotation runs on the RESTORED start state, because both searches leave
        the grid dirty.

        `PSPushExpert.exact_optsets` -- the MEASURED tie sets -- is deliberately
        not wired up here and is refused rather than silently ignored. It costs
        one full re-solve per candidate press, and a single solve on these
        boards runs from 10s to 5 minutes; it also requires ``weight == 1``,
        which the ladder does not promise. So ties come from `annotate_walks`,
        which means a push is labelled with itself even where a sibling push
        would genuinely have tied. Walks -- most of every plan -- are labelled
        completely."""
        assert not self.exact_optsets, "exact_optsets is unaffordable here"
        start = snapshot(eng)
        found = None
        for weight in self.weights:
            self.weight = weight
            self.strategy = f"A* w={weight}"
            restore(eng, start)
            found = self._astar(eng)
            restore(eng, start)
            if found is not None:
                break
        if found is None:
            self.strategy = "beam"
            restore(eng, start)
            found = self._beam(eng)
            restore(eng, start)
            if found is not None:
                found = self._shorten(eng, found)
                restore(eng, start)
        if found is None:
            return None
        return (Plan(found, self.annotate_walks(eng, found))
                if self.annotate else found)

    def _beam(self, eng) -> list | None:
        """`PSBeamExpert._search` with two changes this game needs.

        It prunes on `dead()` -- most of what a beam would otherwise spend its
        width on here is boards that have annihilated a colour down to two
        crates with nowhere left to put them, which is unrecoverable and
        invisible (the level plays on exactly as before) -- and it dedups on the
        exact state rather than the region, for the same reason `_region_key`
        does. Kept local rather than pushed into `PSBeamExpert`: adding a
        `dead()` call there would change ps:bridge's plans, which are
        byte-verified."""
        if eng.check_win():
            return []
        _region, macros = self._analyze(eng)
        frontier = [(snapshot(eng), [], macros)]
        seen = {self._key(eng)}
        nodes = 0
        for _depth in range(self.beam_depth):
            kids = []
            for snap, path, macs in frontier:
                for macro in macs:
                    restore(eng, snap)
                    nodes += 1
                    won = self._apply_macro(eng, macro)
                    if won >= 0:
                        return path + macro[:won + 1]
                    if self.dead(eng):
                        continue
                    key = self._key(eng)
                    if key in seen:
                        continue
                    seen.add(key)
                    _nregion, nmacros = self._analyze(eng)
                    if not nmacros:
                        continue
                    # Rank on the heuristic, tie-break on the shorter path. The
                    # snapshot in the tuple is a list of sets and has no
                    # ordering, so it must never reach the comparison.
                    kids.append((self.heuristic(eng), len(path) + len(macro),
                                 snapshot(eng), path + macro, nmacros))
                    if nodes >= self.beam_node_cap:
                        return None
            if not kids:
                return None                       # the reachable space closed
            kids.sort(key=lambda kid: (kid[0], kid[1]))
            frontier = [(s, p, m) for _h, _g, s, p, m in kids[:self.beam_width]]
        return None

    #: Longest block `_shorten` will try to delete. Each size costs one replay
    #: per position, so the pass is ``max_cut x len(plan)`` replays; 16 covers
    #: two or three whole macros, which is the size of the detours a beam
    #: actually leaves, and keeps the pass to a few minutes on the one level
    #: that needs it.
    max_cut: int = 16

    def _shorten(self, eng, plan: list) -> list:
        """Delete the contiguous blocks of presses the plan does not need.

        A beam plan WINS but it wanders -- it is the shortest route through the
        states the beam happened to keep, not through the game. Cutting blocks
        longest-first and re-verifying each cut against the interpreter is the
        cheap half of the fix: a detour that goes somewhere and comes back is a
        block whose removal still wins, and the check is exact because the
        engine, not a model, decides. What it cannot recover is a plan that took
        a worse route rather than a longer one -- so this shortens, it does not
        optimise, and the level it runs on is labelled as such.

        (Loop removal -- cutting between two presses that leave the board in the
        same state -- is the other standard trick and is deliberately absent: the
        beam dedups on that exact key, so no state repeats along a plan it
        returns and there is never anything to cut.)"""
        start = snapshot(eng)

        def wins(seq) -> bool:
            restore(eng, start)
            eng._rule_win = False
            for d in seq:
                eng.step(d)
                if eng.check_win():
                    return True
            return False

        for size in range(min(self.max_cut, len(plan) - 1), 0, -1):
            i = 0
            while i + size <= len(plan):
                trial = plan[:i] + plan[i + size:]
                if trial and wins(trial):
                    plan = trial            # keep scanning from the same index
                else:
                    i += 1
        restore(eng, start)
        return plan

    def setup(self) -> None:
        super().setup()
        g = self.g
        self.pull_ids = {i for n in self.pull_names
                         for i in g.resolve_object_name(n)}
        #: Crate object index -> (its own index, ring target ids, wall target ids)
        self.colour_ids = {}
        for crate, (ring, wall) in COLOURS.items():
            self.colour_ids[g.obj_name_to_idx[crate]] = (
                set(g.resolve_object_name(ring)),
                set(g.resolve_object_name(wall)),
                g.obj_name_to_idx[crate] in self.pull_ids,
            )
        self.red_id = g.obj_name_to_idx["crater"]
        #: Which rung (or the beam) produced the last plan `_search` returned.
        #: Reporting only -- reading ``self.weight`` afterwards would say "w=5"
        #: for a beam plan, because that is the rung the ladder gave up on.
        self.strategy = "-"
        self._tables: dict = {}      # wall layout -> distance tables
        #: ONE-slot memo, not a dictionary: `_astar` asks `dead()` and then
        #: `heuristic()` about the same engine state (nothing between them
        #: mutates the grid), and that pair is the entire hit rate -- a state is
        #: never scored twice otherwise, because the ``best_g`` test rejects a
        #: revisit before the heuristic is reached. A growing map would hold a
        #: full board key per node for nothing.
        self._h_last: tuple = (None, 0)

    # -- the exact state ------------------------------------------------------
    def _key(self, eng) -> frozenset:
        """Every non-background cell.

        `PSPushExpert` narrows this to pieces + player, which is canonical only
        while the scenery is static. Here ``late [CrateR TargetRWall] -> [Wall]``
        both destroys a target and creates a wall, so two states with identical
        crates can have different boards and completely different futures."""
        bg = self.bg_id
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in cell
            if o != bg
        )

    def _region_key(self, eng, region):
        """Dedup on the exact state, not on the player's walkable region.

        The base's region merge assumes the walk graph is undirected (so "same
        region" really is "same futures"). The pull rule makes it directed --
        you can walk into a pocket past a grey and not walk back out without
        dragging it -- so merging on the region would drop live states."""
        return self._key(eng)

    # -- macros ---------------------------------------------------------------
    def _read(self, eng):
        """One pass over the grid: ``(player, free, grey, crates, walls)``.

        ``free`` is the player's walkability map (no wall, no crate of any
        colour), ``grey`` marks the cells whose occupant the player drags when
        it steps directly away, ``crates`` is ``{object index: [cells]}`` and
        ``walls`` is the frozenset the distance tables are cached on."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        player = None
        free = [[True] * w for _ in range(h)]
        grey = [[False] * w for _ in range(h)]
        crates: dict[int, list] = {i: [] for i in self.colour_ids}
        walls = []
        for r in range(h):
            row = grid[r]
            for c in range(w):
                cell = row[c]
                if not cell:
                    continue
                if cell & self.player_ids:
                    player = (r, c)
                if cell & self.blocker_ids:
                    free[r][c] = False
                    walls.append((r, c))
                hit = cell & self.push_ids
                if hit:
                    free[r][c] = False
                    for o in hit:
                        crates[o].append((r, c))
                    if hit & self.pull_ids:
                        grey[r][c] = True
        return player, free, grey, crates, frozenset(walls)

    def _walk_tree(self, player, free, grey, h, w):
        """Directed BFS from the player over BARE walk steps.

        The step ``u -> u + m`` moves nothing but the player only when ``u - m``
        holds no grey crate; when it does, the same press is a PULL, which the
        macro list carries separately. The graph is therefore directed, and
        `_region_key` is why that matters."""
        parent: dict[tuple[int, int], tuple | None] = {player: None}
        queue = deque([player])
        while queue:
            cur = queue.popleft()
            r, c = cur
            for d, (dr, dc) in _DELTA.items():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not free[nr][nc]:
                    continue
                if (nr, nc) in parent:
                    continue
                br, bc = r - dr, c - dc
                if 0 <= br < h and 0 <= bc < w and grey[br][bc]:
                    continue                       # this press is a pull
                parent[(nr, nc)] = (cur, d)
                queue.append((nr, nc))
        return parent

    def _analyze(self, eng) -> tuple:
        """``(None, macros)`` -- every push and every pull available right now.

        A push of the crate at ``q`` in direction ``m``: stand at ``q - m``,
        press ``m``; the crate lands on ``q + m``, which must be free (targets
        are a different collision layer, so a door or a goal counts as free).
        A pull of the grey at ``q`` in direction ``m``: stand at ``q + m``,
        press ``m``; the player lands on ``q + 2m``, which must be free, and the
        grey follows to ``q + m``.

        Both are prefiltered exactly, so every macro the search queues really
        changes the board -- the base's "the engine refused it" branch never
        fires here, which matters because a refused macro that leaves the player
        somewhere new is not a no-op it can dedup away.

        Duplicates are dropped: one press can be a push and a pull at the same
        time (shove a red while a grey sits behind you) and the enumeration
        reaches it once from each crate. The engine does both either way -- it
        is one macro, not two, and queueing it twice buys two identical nodes.

        The region slot is ``None``: `_region_key` ignores it."""
        h, w = len(eng.grid), len(eng.grid[0])
        player, free, grey, crates, _walls = self._read(eng)
        if player is None:                      # player consumed by a rule
            return None, []
        parent = self._walk_tree(player, free, grey, h, w)

        def walk_to(cell) -> list:
            out = []
            while parent[cell] is not None:
                cell, d = parent[cell]
                out.append(d)
            out.reverse()
            return out

        def inside(cell):
            return 0 <= cell[0] < h and 0 <= cell[1] < w

        macros, seen = [], set()
        for obj, cells in crates.items():
            pulled = self.colour_ids[obj][2]
            for (qr, qc) in cells:
                for m, (dr, dc) in _DELTA.items():
                    if pulled:
                        stand = (qr + dr, qc + dc)
                        land = (qr + 2 * dr, qc + 2 * dc)
                    else:
                        stand = (qr - dr, qc - dc)
                        land = (qr + dr, qc + dc)
                    if not inside(land) or not free[land[0]][land[1]]:
                        continue
                    if stand not in parent:
                        continue
                    macro = walk_to(stand) + [m]
                    if (stand, m) in seen:
                        continue
                    seen.add((stand, m))
                    macros.append(macro)
        return None, macros

    def _apply_macro(self, eng, macro) -> int:
        """Run a macro, TELEPORTING the walk instead of stepping it.

        No rule fires on a bare walk step -- that is the definition `_analyze`
        builds the walk tree from -- and the late rules have already settled, so
        relocating the player to the working cell and stepping only the final
        press leaves exactly the state the primitive replay would. It turns a
        ~9-turn macro into 1, which is the whole cost of this search;
        ``--fuzz`` is the check that it really is exact.

        The win can only be produced by the final press (a walk moves no crate),
        so the winning index is the macro's last."""
        if len(macro) > 1:
            pos = None
            for r, row in enumerate(eng.grid):
                for c, cell in enumerate(row):
                    if cell & self.player_ids:
                        pos = (r, c)
                        break
                if pos is not None:
                    break
            here = eng.grid[pos[0]][pos[1]]
            moving = here & self.player_ids
            here -= moving
            for d in macro[:-1]:
                dr, dc = _DELTA[d]
                pos = (pos[0] + dr, pos[1] + dc)
            eng.grid[pos[0]][pos[1]] |= moving
            eng._position_index_dirty = True
            eng._rule_noop_cache.clear()
        eng.step(macro[-1])
        return len(macro) - 1 if eng.check_win() else -1

    # -- tie sets -------------------------------------------------------------
    def _walk_distances(self, eng, target) -> dict:
        """Walk distance from every cell TO ``target``, over the directed graph.

        The base BFSes outward from ``target`` over undirected adjacency, which
        would offer the policy a step that actually drags a grey as a free
        alternative. This is the reverse BFS of `_walk_tree`: ``u`` precedes
        ``v = u + m`` when ``v`` is free and ``u - m = v - 2m`` holds no grey."""
        h, w = len(eng.grid), len(eng.grid[0])
        _player, free, grey, _crates, _walls = self._read(eng)
        dist = {target: 0}
        queue = deque([target])
        while queue:
            v = queue.popleft()
            for dr, dc in _DELTA.values():
                u = (v[0] - dr, v[1] - dc)
                if not (0 <= u[0] < h and 0 <= u[1] < w) or u in dist:
                    continue
                if not free[v[0]][v[1]]:
                    continue
                b = (v[0] - 2 * dr, v[1] - 2 * dc)
                if 0 <= b[0] < h and 0 <= b[1] < w and grey[b[0]][b[1]]:
                    continue
                dist[u] = dist[v] + 1
                queue.append(u)
        return dist

    def _walk_step_ok(self, eng, cell, direction) -> bool:
        """True when pressing ``direction`` at ``cell`` is a bare walk.

        `PSPushExpert.annotate_walks` calls this before offering a direction as
        an equally-short alternative. Without it a tie set could name the press
        that drags a grey along -- same walk distance, different board."""
        h, w = len(eng.grid), len(eng.grid[0])
        dr, dc = _DELTA[direction]
        b = (cell[0] - dr, cell[1] - dc)
        if not (0 <= b[0] < h and 0 <= b[1] < w):
            return True
        return not (eng.grid[b[0]][b[1]] & self.pull_ids)

    # -- heuristic ------------------------------------------------------------
    def _move_edge(self, free, h, w, cell, dr, dc, pulled):
        """Can a crate at ``cell`` move one step by ``(dr, dc)``?

        A pushed crate needs the cell it moves INTO free and the cell BEHIND it
        free, because the player has to stand there. A pulled crate needs the
        cell it moves into free and the cell BEYOND that free, because the
        player has to stand there instead -- one cell further along the same
        direction. That asymmetry is the whole difference between the two
        colours' distance tables. Other crates are ignored (this is the
        relaxation that keeps the tables cacheable per wall layout)."""
        r, c = cell
        ar, ac = r + dr, c + dc
        if not (0 <= ar < h and 0 <= ac < w) or not free[ar][ac]:
            return False
        br, bc = (ar + dr, ac + dc) if pulled else (r - dr, c - dc)
        return 0 <= br < h and 0 <= bc < w and free[br][bc]

    def _tab(self, free, walls, h, w):
        """Distance tables for one WALL LAYOUT, built once and cached.

        Walls change only when a door closes -- a handful of times per level --
        so every other node's heuristic is dictionary lookups. ``free`` here is
        the STATIC map (walls only): the crates are what we are costing, so
        letting them block each other would make the table depend on the state
        and defeat the cache."""
        tab = self._tables.get(walls)
        if tab is not None:
            return tab
        cells = [(r, c) for r in range(h) for c in range(w) if free[r][c]]
        lines = {}
        for n in (POP_RUN, WIN_RUN):
            runs = []
            for (r, c) in cells:
                for dr, dc in ((0, 1), (1, 0)):
                    run = [(r + i * dr, c + i * dc) for i in range(n)]
                    if all(0 <= y < h and 0 <= x < w and free[y][x]
                           for y, x in run):
                        runs.append(run)
            lines[n] = runs
        tab = {"cells": cells, "lines": lines, "fwd": {}, "rev": {}}
        self._tables[walls] = tab
        return tab

    def _bfs(self, free, h, w, start, pulled, backward):
        """Crate-move BFS. ``backward`` walks the edges in reverse, giving
        "distance from every cell TO ``start``"; forward gives "FROM ``start``"."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            cell = queue.popleft()
            for dr, dc in _DELTA.values():
                nxt = (cell[0] + dr, cell[1] + dc)
                if nxt in dist or not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                    continue
                ok = (self._move_edge(free, h, w, nxt, -dr, -dc, pulled)
                      if backward else
                      self._move_edge(free, h, w, cell, dr, dc, pulled))
                if not ok:
                    continue
                dist[nxt] = dist[cell] + 1
                queue.append(nxt)
        return dist

    def _fwd(self, tab, free, h, w, cell, pulled):
        key = (cell, pulled)
        out = tab["fwd"].get(key)
        if out is None:
            out = tab["fwd"][key] = self._bfs(free, h, w, cell, pulled, False)
        return out

    def _rev(self, tab, free, h, w, cell, pulled):
        key = (cell, pulled)
        out = tab["rev"].get(key)
        if out is None:
            out = tab["rev"][key] = self._bfs(free, h, w, cell, pulled, True)
        return out

    def _assign(self, dists, sinks):
        """Exact min-cost assignment of crates to distinct sinks.

        ``dists[i]`` is crate i's reverse-distance table lookup function. The
        sets are at most 3x3 on these boards, so the permutations ARE the
        Hungarian algorithm and cost less than setting one up."""
        n = len(dists)
        if n == 0:
            return 0
        if n > len(sinks):
            return DEAD_COST
        best = DEAD_COST
        for pick in itertools.permutations(sinks, n):
            total = 0
            for i, sink in enumerate(pick):
                d = dists[i].get(sink)
                if d is None:
                    total = DEAD_COST
                    break
                total += d
            best = min(best, total)
        return best

    def _line_cost(self, tab, fwds, n):
        """Cheapest line of ``n`` cells to gather ``n`` crates on.

        Each cell takes its nearest crate independently -- a relaxation of the
        matching, so the result stays a lower bound -- because this runs on
        every node and the exact version is ``n!`` per candidate line."""
        best = DEAD_COST
        for run in tab["lines"][n]:
            total = 0
            for cell in run:
                near = DEAD_COST
                for fwd in fwds:
                    d = fwd.get(cell)
                    if d is not None and d < near:
                        near = d
                if near >= DEAD_COST:
                    total = DEAD_COST
                    break
                total += near
            best = min(best, total)
            if best == 0:
                break
        return best

    def heuristic(self, eng) -> int:
        """Lower-bound crate moves left: the cheaper of the game's TWO wins.

        There are two, and they are alternatives, not stages:

        * satisfy the three ``All CrateX on TargetX`` conditions, which costs at
          least the sum over the colours of that colour's own cheapest route --
          assign every crate to a sink (ring target or door: a crate is equally
          finished by being parked on a goal or eaten by a door), or annihilate
          some three of them and assign the rest;
        * line up five reds, which fires ``-> win`` and ends the level with
          whatever else is still on the board.

        Taking the MIN is what makes the estimate a lower bound. Summing them
        would charge stage 15 for tidying three blue crates that its winning
        move ignores. `DEAD_COST` when neither route is left -- which is most of
        this game's dead ends, because a colour that annihilates down to two
        crates with no target of its own can never be finished again, and
        `dead()` drops those un-expanded."""
        key = self._key(eng)
        if self._h_last[0] == key:
            return self._h_last[1]
        h, w = len(eng.grid), len(eng.grid[0])
        player, _free, _grey, crates, walls = self._read(eng)
        static = [[(r, c) not in walls for c in range(w)] for r in range(h)]
        tab = self._tab(static, walls, h, w)

        conditions = 0
        outright = DEAD_COST
        for obj, cells in crates.items():
            if not cells:
                continue
            rings, doors, pulled = self.colour_ids[obj]
            sinks = [(r, c) for r in range(h) for c in range(w)
                     if eng.grid[r][c] & (rings | doors)]
            revs = {s: self._rev(tab, static, h, w, s, pulled) for s in sinks}
            dists = [{s: revs[s].get(cell) for s in sinks
                      if revs[s].get(cell) is not None} for cell in cells]
            fwds = [self._fwd(tab, static, h, w, cell, pulled) for cell in cells]
            n = len(cells)

            if obj == self.red_id and n >= WIN_RUN:
                outright = self._line_cost(tab, fwds, WIN_RUN)

            best = self._assign(dists, sinks)
            if n >= POP_RUN:
                for trio in itertools.combinations(range(n), POP_RUN):
                    rest = [i for i in range(n) if i not in trio]
                    if len(rest) > len(sinks):
                        continue
                    pop = self._line_cost(tab, [fwds[i] for i in trio], POP_RUN)
                    if pop >= DEAD_COST:
                        continue
                    best = min(best,
                               pop + self._assign([dists[i] for i in rest], sinks))
            conditions = min(DEAD_COST, conditions + best)

        total = min(conditions, outright)
        if 0 < total < DEAD_COST and player is not None:
            # The player has to REACH a crate before it can move one, and ``g``
            # counts its steps while everything above counts the crates'. One
            # walk is the cheapest thing that is certainly still owed, so adding
            # it keeps the bound valid and stops the estimate going flat on the
            # wide boards. Manhattan (not the real walk) so it stays O(crates),
            # and -1 because the working cell is BESIDE the crate, not on it.
            total += max(0, min(abs(player[0] - r) + abs(player[1] - c)
                                for cells in crates.values()
                                for (r, c) in cells) - 1)
        self._h_last = (key, total)
        return total

    def dead(self, eng) -> bool:
        """Neither win route is left. The usual sokoban corner test, widened
        two ways: a crate in a corner is fine here if two friends of its colour
        can reach the two cells beside it, and a board with no route at all for
        one colour is still live while five reds can still make a line."""
        return self.heuristic(eng) >= DEAD_COST


class ESLSolver(PSAStarSolver):
    game_id = "puzzlescript_esl_puzzle_game_challenge_mode"
    game_name = GAME_NAME
    expert_cls = ESLExpert
    game_module_id = "ps:esl_puzzle_game_challenge_mode"

    #: Rung 0 of `ESLExpert.weights`; the expert re-sets it per rung.
    weight = 1

    #: A cap on GENERATED nodes, chosen for MEMORY, not for time. Every queued
    #: node holds a full board snapshot (a list of sets, ~30 KB on the 12x12
    #: board) plus its macro list, so the queue is the whole footprint: an
    #: 800k-node run of stage 20 reached 20 GB of RSS before it was killed. At
    #: 250k the worst board peaks near 6 GB, the boards A* can settle all settle
    #: well inside it, and the one that cannot falls through to the beam --
    #: which is frontier-bounded and never passes 1 GB.
    node_cap = 250_000

    #: The longest plan is 157 presses (stage 20, the beam's); the rest is room
    #: for a re-plan. Stays inside that level's own 471-step budget in
    #: `games/ps:esl_puzzle_game_challenge_mode`.
    max_steps = 300

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup and
        fills the disk cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _report() -> int:
    """Per-level plan length, board size, crate counts and tie coverage."""
    import time
    solver = ESLSolver()
    game = solver.make_game(0)
    expert = ESLExpert(game, node_cap=ESLSolver.node_cap, weight=ESLSolver.weight)
    idx = game._game.obj_name_to_idx
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        counts = {c: sum(1 for row in eng.grid for cell in row if idx[c] in cell)
                  for c in COLOURS}
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        head = (f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
                f"R{counts['crater']} B{counts['crateb']} P{counts['cratep']}")
        if found is None:
            print(f"{head}: NO PLAN ({dt:.1f}s)")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(found):3d} moves (budget {game._max_steps}, {room}), "
              f"{ties:3d} tie steps ({ties / max(1, len(found)):.0%}), "
              f"{expert.strategy}, {dt:.1f}s")
    print(f"total {total} moves over {game.n_levels} levels")
    return 0


def _fuzz(trials: int = 400) -> int:
    """Assert the teleported walk in `_apply_macro` is exact.

    Every macro is generated from a board reached by random legal macros, then
    run BOTH ways -- teleport-then-press, and press-every-primitive -- and the
    resulting grids and win flags compared. This is the only thing standing
    between the search and a silently wrong model, because a teleport that
    skipped a rule would not error, it would just plan through states the engine
    never reaches."""
    import random
    solver = ESLSolver()
    game = solver.make_game(0)
    expert = ESLExpert(game, node_cap=ESLSolver.node_cap)
    eng = game._engine
    rng = random.Random(0)
    bad = 0
    for level in range(game.n_levels):
        checked = 0
        for _ in range(trials):
            game.set_level(level)
            eng._rule_win = False
            for _ in range(rng.randrange(0, 6)):        # random legal prefix
                _region, macros = expert._analyze(eng)
                if not macros:
                    break
                expert._apply_macro(eng, rng.choice(macros))
                if eng.check_win():
                    break
            if eng.check_win():
                continue
            _region, macros = expert._analyze(eng)
            if not macros:
                continue
            macro = rng.choice(macros)
            base = snapshot(eng)
            expert._apply_macro(eng, macro)
            fast, fast_win = snapshot(eng), eng.check_win()
            restore(eng, base)
            eng._rule_win = False
            slow_win = False
            for d in macro:
                eng.step(d)
                slow_win = slow_win or eng.check_win()
            slow = snapshot(eng)
            checked += 1
            if fast != slow or fast_win != slow_win:
                bad += 1
                if bad <= 3:
                    print(f"  MISMATCH level {level} macro {macro}")
        print(f"level {level}: {checked} macros checked")
    print("fuzz clean" if not bad else f"FUZZ FAILED: {bad} mismatches")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Composition, not object: what has to stay readable is the STACK -- a red
    crate on a red goal is a different thing from either, and a door that has
    swallowed its crate is a Wall. The sprite detail that separates them is a
    few pixels wide, so it has to be checked at the size each board renders at
    (``cell_px = min(64 // H, 64 // W)``), where a 5x5 sprite is sampled at only
    some of its rows."""
    game = ESLSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {"floor": (), "wall": ("wall",), "player": ("player",)}
    for crate, (ring, wall) in COLOURS.items():
        tag = crate[-1]
        comps[f"crate{tag}"] = (crate,)
        comps[f"target{tag}"] = (ring,)
        comps[f"door{tag}"] = (wall,)
        comps[f"crate{tag}_on_target"] = (ring, crate)
        comps[f"player_on_target{tag}"] = (ring, "player")
    # A crate of the WRONG colour parked on a ring target: the target has to
    # still read through it, or the board loses a goal every time one passes.
    comps["crateR_on_targetB"] = ("targetb", "crater")

    sizes = {}
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
            # The WHOLE frame, not a computed cell box: `_render_frame` upscales
            # the rendered board to fill 64x64 and then letterboxes it, so the
            # cell grid in the output is not `cell_px` aligned and slicing one
            # cell out by arithmetic samples the wrong pixels (it reported every
            # composition identical to floor). Every cell of the board holds the
            # same composition here, so comparing full frames is exact.
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--fuzz" in sys.argv:
        sys.exit(_fuzz())
    sys.exit(ESLSolver.main())
