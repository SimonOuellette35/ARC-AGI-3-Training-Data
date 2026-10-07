"""Generate Phase-1 training data for the PuzzleScript game ps:color_combination
("Color Combination" by Pancake Robot -- additive-colour sokoban).

The harness -- the rotation contract, the walk-and-push macro A*, the trajectory
recorder, the RESET-recovery prefix and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is the game-specific part: the mechanic notes, the colour
algebra the heuristic and the death test are built on, and the disk plan cache.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_color_combination",
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
exactly.

The game
--------
Sokoban with one twist: the crates are coloured lights, and shoving one into
another MIXES them. Four arrow keys, no ACTION (nothing in the file binds it, so
it is dropped from the search). Fifteen rules, in the order the engine runs them:

    [ > Player | Block ] -> [ > Player | > Block ]           (push, ONE deep)
    [ > Red | RedTarget ] -> [ | FilledTarget ]              (x7, one per colour)
    [ > Red | Green ] -> [ | Yellow ]                        (x12, both orders)
    ...
    WINCONDITIONS: No ColoredTargets

  * **Pushes are one crate deep and never chain.** The push rule matches only
    ``moving Player | Block``, and re-firing it changes nothing, so it never
    propagates to a second crate. A crate with a wall, the board edge, or a
    non-combining crate behind it simply cannot move -- and because crates share
    the player's collision layer, the whole turn is then a silent no-op.
  * **Colour is a SET of primary lights and combining is DISJOINT UNION.**
    red={R}, green={G}, blue={B}, yellow={R,G}, cyan={G,B}, magenta={R,B},
    white={R,G,B}. The twelve mixing rules are exactly the disjoint unions:
    R+G=Y, R+B=M, G+B=C, R+C=W, G+M=W, B+Y=W. Every OVERLAPPING pair (Y+M, Y+C,
    M+C, W+anything, X+X) has no rule at all, so pushing one into the other is a
    dead no-op. `_MASK` below is that algebra, and it is the whole reason this
    solver can be sharp: it makes both the heuristic and `dead` exact statements
    about colour rather than guesses.
  * **A merge lands on the STATIONARY crate's cell** and the moving one is
    consumed, so two crates become one and the primary-light content is
    conserved. The only other way a crate leaves the board is filling a target,
    which consumes exactly that target's colour set. So the board's total light
    content only ever decreases, by exactly what the targets take -- which is
    what makes `dead` sound.
  * **A crate must be pushed INTO its target from an adjacent cell.** The fill
    rule needs ``moving Red | RedTarget``, so a crate that is merely parked on
    top of its own target does nothing; it has to come off and be shoved back
    in. Targets sit on their own collision layer, so any crate may sit on any
    target, and pushing a crate across a target leaves the target untouched.
  * **The fill rule OUTRANKS the mixing rule.** Fills are listed first, so a Red
    shoved at a cell holding both a Green crate and a RedTarget fills the target
    and leaves the Green standing, rather than mixing into Yellow.

All six statements are asserted against the interpreter by ``--selfcheck``.

Expert solver
-------------
`PSPushExpert`: A* over the REAL interpreter whose successors are ``walk to the
push cell, then push`` macros, so search depth is the number of pushes and each
walk is one BFS rather than four levels of branching. ``g`` counts primitive
MOVES, so plans are shortest in the metric the agent actually pays.

The heuristic is where the colour algebra pays. For each remaining target,
choose a set of crates whose colour masks are pairwise DISJOINT and union to the
target's mask -- by the conservation argument above, no other set of crates can
ever become that colour -- and charge the half-perimeter of the bounding box of
those crates plus the target. That is admissible: the crates' push trajectories
form a connected structure spanning all of them and the target (each merge joins
two trajectories, each fill ends one on the target), a Manhattan Steiner tree
over those points is a lower bound on its total length, and the bounding-box
half-perimeter is a lower bound on that tree -- exact for the two-crate case.
Targets are assigned crate sets disjointly, so summing over targets stays a
bound. It is worth the trouble: it takes the five levels from 157s to 23s of
search and returns the same plan lengths, i.e. it only removed wasted nodes.

`dead` falls out of the same enumeration for free: when no disjoint assignment
of the surviving crates to the surviving targets exists, the level is provably
lost -- and unlike most sokoban deadlocks, this one is not geometric, it happens
the instant a wrong merge burns a primary light (shove the Red into the Green in
level 2 and the White target can never be built again, with nothing on screen to
say so). It is the single most useful fact about the game.

Plans are engine-verified wins by construction, memoized by state key, and
seed-independent (levels are fixed ASCII maps; only the PRESENTATION is
augmented per seed), so seed 0 pays for all five searches and every later seed
replays them under its own rotation/flip. Those 23s are also written to
``data/color_combination_plans.json``, or every shard of `parallelize_generator`
would repeat them on every core. Warm, a run starts in about a second. Delete
the file to re-derive it.

Optimal-action sets
-------------------
`PSPushExpert.annotate_walks` labels every step: a push with itself, and each
step of a walk with every direction that keeps it on a shortest route to the
same push cell. The ties are the common case here -- the boards are wide open
rooms and 60% of the presses are walking across them -- and any interleaving of
the two axes reaches the same cell in the same number of moves and leaves an
identical board, so labelling one of them as "the" answer would train against
the truth on most steps. No step ever ships unlabelled (see the
always-emit-optimal-targets rule).

The palette adaptation
----------------------
``data/puzzlescript_games/Color_Combination.txt`` carries an ARC recolor of the
original art (colours and sprite shapes only -- no rule was touched), because
the original does not survive quantization onto the 16-colour ARC palette: green
and yellow both landed on index 14, cyan collapsed onto the background's blue
speckles, every crate wore the WALL's sprite, and an opaque crate parked on a
target ERASED it from the frame. The fix gives each object its own ARC index and
splits the cell geometry between the two collision layers -- a target is the 1px
cell border, a crate is the solid interior -- so a crate sitting on an unmatched
target leaves that target fully visible. See the note at the top of the .txt.

Augmentation
------------
Colour Combination is gravity-free, its rules are stated entirely over relative
directions (every one of the fifteen is a ``>``), its win condition (``No
ColoredTargets``) is positional in no way, and its input is screen-relative, so
the board's full 8-element symmetry group is a valid presentation augmentation:
the game is in `PuzzleScriptAdapter._FLIP_GAMES`, giving rotation_k in {0,1,2,3}
x horizontal x vertical flip with the matching directional action remap. That is
16 presentations of 5 levels. No colour augmentation (the game is not in
``_RECOLOR_GAMES`` and must not be): which crate mixes with which IS the puzzle,
and a recolor that relabelled the palette would break the R/G/B arithmetic the
frames are supposed to teach.

Usage (run from the repo root):
    python solvers/generate_color_combination_training.py --episodes 200 \
        --out data/training_multi_level/color_combination

    python solvers/generate_color_combination_training.py --plans      # level report
    python solvers/generate_color_combination_training.py --selfcheck  # mechanic audit
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import PuzzleScriptAdapter        # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSPushExpert,    # noqa: E402
                                     snapshot)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Color_Combination"

#: Disk cache of the per-level start plan AND its optimal-action sets. The
#: searches are seed-independent but cost ~23s of interpreter steps, which every
#: shard of `parallelize_generator` would otherwise repeat on every core. Delete
#: the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "color_combination_plans.json"

#: Crate/target colour -> its set of primary lights as a bitmask (R=1, G=2, B=4).
#: Mixing is DISJOINT UNION and nothing else -- see the module docstring. The
#: target objects are the same names with ``target`` appended.
_MASK: dict[str, int] = {
    "red": 1, "green": 2, "blue": 4,
    "yellow": 3, "magenta": 5, "cyan": 6,
    "white": 7,
}

#: Charge for a state no assignment of crates to targets can win. Finite so the
#: search stays complete; large enough that such a node is never popped.
_DEAD = 1 << 20


class ColorCombinationExpert(PSPushExpert):
    """Walk-and-push A* over the real interpreter, guided (and pruned) by the
    additive-colour algebra. See the module docstring for the admissibility
    argument and for why `dead` is the most valuable part of it."""

    pushable_names = ("block",)
    blocker_names = ("wall",)
    #: Label every step with its optimal SET, and keep each level's start plan
    #: on disk -- both are `PSExpert` / `PSPushExpert` features now; see the
    #: "Optimal-action sets" section of the module docstring for why the ties
    #: matter here and `PLAN_CACHE` for why the searches are worth storing.
    annotate = True
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        super().setup()
        g = self.g
        #: object index -> colour mask, for crates and for targets separately.
        self.block_mask = {g.obj_name_to_idx[n]: m for n, m in _MASK.items()}
        self.tgt_mask = {g.obj_name_to_idx[n + "target"]: m
                         for n, m in _MASK.items()}
        self.target_ids = set(self.tgt_mask)
        # Targets belong in the state key: they vanish one at a time as they are
        # filled, and two routes can reach the same crate layout having filled
        # different ones.
        self.dyn_ids |= self.target_ids
        self._disk = self._load_disk()

    def _region_key(self, eng, region) -> tuple:
        """`PSPushExpert`'s key plus the surviving targets -- crates alone do not
        pin the state once a fill has removed one (`setup`)."""
        ids = self.push_ids | self.target_ids
        return (frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & ids)
        ), region)

    # -- the colour algebra ---------------------------------------------------
    def _scan(self, eng) -> tuple[list, list]:
        """``(crates, targets)`` as ``[(row, col, colour mask), ...]``."""
        crates, targets = [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                for o in cell:
                    m = self.block_mask.get(o)
                    if m is not None:
                        crates.append((r, c, m))
                        continue
                    m = self.tgt_mask.get(o)
                    if m is not None:
                        targets.append((r, c, m))
        return crates, targets

    def _cost(self, eng) -> int | None:
        """The heuristic value, or None when the level is provably lost.

        Assign each surviving target a set of crates whose masks are pairwise
        disjoint and union to the target's own -- the only crates that can ever
        become that colour -- with no crate serving two targets, and charge the
        half-perimeter of the bounding box of each group plus its target. The
        minimum over assignments is an admissible lower bound on the remaining
        moves (module docstring); no assignment at all means dead."""
        crates, targets = self._scan(eng)
        if not targets:
            return 0
        n = len(crates)

        # Every pairwise-disjoint subset of the crates, with its union mask. The
        # boards carry a handful of crates, so enumerating 2^n once per node is
        # cheaper than being clever about it.
        groups: list[tuple[int, int]] = []
        for bits in range(1, 1 << n):
            union, ok, rest, i = 0, True, bits, 0
            while rest:
                if rest & 1:
                    m = crates[i][2]
                    if union & m:
                        ok = False
                        break
                    union |= m
                rest >>= 1
                i += 1
            if ok:
                groups.append((bits, union))

        best: list[int | None] = [None]

        def span(cells) -> int:
            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            return (max(rows) - min(rows)) + (max(cols) - min(cols))

        def assign(i: int, used: int, acc: int) -> None:
            if best[0] is not None and acc >= best[0]:
                return                      # already worse than a full solution
            if i == len(targets):
                best[0] = acc
                return
            tr, tc, tm = targets[i]
            for bits, union in groups:
                if union != tm or (bits & used):
                    continue
                cells = [(tr, tc)] + [crates[j][:2]
                                      for j in range(n) if bits >> j & 1]
                assign(i + 1, used | bits, acc + span(cells))

        assign(0, 0, 0)
        return best[0]

    def heuristic(self, eng) -> int:
        cost = self._cost(eng)
        return _DEAD if cost is None else cost

    def dead(self, eng) -> bool:
        """True when the surviving crates can no longer be partitioned into one
        disjoint colour-exact group per surviving target.

        Sound, not a guess: a crate's colour set only ever grows by disjoint
        union, so the group that ends up filling a target must have been a
        disjoint cover of it all along. A state this rejects holds no win."""
        return self._cost(eng) is None


class ColorCombinationSolver(PSAStarSolver):
    game_id = "puzzlescript_color_combination"
    game_name = GAME_NAME
    expert_cls = ColorCombinationExpert

    #: Unweighted: the colour heuristic already brings all five levels in under
    #: 25s together, so there is nothing to buy by trading optimality away.
    weight = 1
    #: The hardest level settles in ~35k macro expansions; the cap only has to
    #: leave headroom above that.
    node_cap = 400_000
    #: Longest plan is 22 moves; the rest is room for the exploration prefix and
    #: a re-plan after it.
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

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The full set of equally-shortest presses at this step -- see
        `PSPushExpert.annotate_walks`. Falls back to the press about to be
        taken, so no expert step ever ships unlabelled (the
        always-emit-optimal-targets rule)."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets):
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Mechanic self-check
# ---------------------------------------------------------------------------

def selfcheck(trials: int = 40, steps: int = 30, verbose: bool = True) -> int:
    """Audit, against the interpreter, every claim the solver rests on.

    The six mechanic statements in the module docstring are checked on hand-built
    boards; the two the search relies on most -- that colour content is conserved
    by mixing and only ever spent on a matching target, and that a bare move onto
    an empty cell has no side effect -- are also checked over random rollouts
    from every level start, which is where a rule interaction nobody thought of
    would show up. Returns the number of violations."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = ColorCombinationExpert(game)
    eng = game._engine
    n2i = game._game.obj_name_to_idx
    bad = 0

    def fail(msg: str) -> None:
        nonlocal bad
        bad += 1
        print(f"  VIOLATION: {msg}")

    # -- hand-built boards, inside level 0's grid so the engine's own extents
    # -- stay valid (check_win indexes the grid it was loaded with).
    height, width = 8, 10

    def build(row: str, extra=()) -> None:
        game.set_level(0)
        grid = [[{n2i["background"]} for _ in range(width)]
                for _ in range(height)]
        for c in range(width):
            grid[0][c].add(n2i["wall"])
            grid[height - 1][c].add(n2i["wall"])
        for r in range(height):
            grid[r][0].add(n2i["wall"])
            grid[r][width - 1].add(n2i["wall"])
        for c, name in enumerate(row):
            if name != ".":
                grid[2][c + 1].add(n2i[_SHORT[name]])
        for (r, c, name) in extra:
            grid[r][c].add(n2i[name])
        eng.grid = grid
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    def cells(name: str) -> set:
        idx = n2i[name]
        return {(r, c) for r, row in enumerate(eng.grid)
                for c, cell in enumerate(row) if idx in cell}

    # 1. mixing is disjoint union, and lands on the stationary crate's cell.
    for a, b, out in (("R", "G", "yellow"), ("G", "R", "yellow"),
                      ("R", "B", "magenta"), ("G", "B", "cyan"),
                      ("R", "C", "white"), ("G", "M", "white"),
                      ("B", "Y", "white")):
        build("P" + a + b)
        eng.step("right")
        if cells(out) != {(2, 3)}:
            fail(f"{a}+{b} did not leave a {out} at the stationary crate's cell")

    # 2. every OVERLAPPING pair is a dead no-op (no rule matches it at all).
    for a, b in (("Y", "M"), ("Y", "C"), ("M", "C"), ("W", "R"), ("R", "R"),
                 ("W", "W"), ("C", "C")):
        build("P" + a + b)
        before = snapshot(eng)
        eng.step("right")
        if eng.grid != before:
            fail(f"{a}+{b} changed the board -- the colour algebra is wrong")

    # 3. pushes never chain: a crate with a non-combining crate behind it is a
    #    full-turn no-op, player included.
    build("PRWW")
    before = snapshot(eng)
    eng.step("right")
    if eng.grid != before:
        fail("a push chained through a second crate")

    # 4. a crate must be pushed INTO its target; parking on it does nothing.
    build("PR", extra=[(2, 2, "redtarget")])
    eng.step("right")
    if not cells("redtarget") or cells("filledtarget"):
        fail("a crate resting on its own target filled it")

    # 5. a crate crosses a non-matching target without disturbing it.
    build("PR.2")
    for _ in range(3):
        eng.step("right")
    if cells("greentarget") != {(2, 4)}:
        fail("pushing a crate over a non-matching target disturbed it")

    # 6. the fill rule outranks the mixing rule.
    build("PRG", extra=[(2, 3, "redtarget")])
    eng.step("right")
    if not cells("filledtarget") or cells("green") != {(2, 3)}:
        fail("mixing beat the fill rule on a shared cell")

    # -- random rollouts: colour conservation and side-effect-free walking.
    for level in range(game.n_levels):
        for t in range(trials):
            game.set_level(level)
            rng = random.Random(f"colorcombination:selfcheck:{level}:{t}")
            for _ in range(steps):
                crates, targets = expert._scan(eng)
                light = sum(bin(m).count("1") for (_r, _c, m) in crates)
                pos = _player_cell(eng, expert)
                before = snapshot(eng)
                direction = rng.choice(expert.directions)
                eng.step(direction)
                after_crates, after_targets = expert._scan(eng)
                after_light = sum(bin(m).count("1")
                                  for (_r, _c, m) in after_crates)
                spent = sum(bin(m).count("1") for (_r, _c, m) in targets) - \
                    sum(bin(m).count("1") for (_r, _c, m) in after_targets)
                if after_light != light - spent:
                    fail(f"L{level}: {direction} changed the board's light "
                         f"content by {light - after_light}, targets took "
                         f"{spent}")
                new_pos = _player_cell(eng, expert)
                if new_pos != pos and after_crates == crates and \
                        after_targets == targets:
                    # A bare walk: nothing but the player may have moved. This
                    # is the assumption `annotate_walks` labels ties on.
                    moved = {(r, c, o)
                             for r, row in enumerate(eng.grid)
                             for c, cell in enumerate(row) for o in cell} ^ \
                            {(r, c, o)
                             for r, row in enumerate(before)
                             for c, cell in enumerate(row) for o in cell}
                    if moved - {(pos[0], pos[1], o) for o in expert.player_ids} \
                            - {(new_pos[0], new_pos[1], o)
                               for o in expert.player_ids}:
                        fail(f"L{level}: a bare {direction} moved something "
                             f"other than the player")
                if eng.check_win():
                    break
        if verbose:
            print(f"  L{level}: {trials} rollouts checked")
    return bad


#: Level-file legend letters used by `selfcheck`'s hand-built boards.
_SHORT = {"P": "player", "R": "red", "G": "green", "B": "blue", "C": "cyan",
          "M": "magenta", "Y": "yellow", "W": "white", "#": "wall",
          "1": "redtarget", "2": "greentarget", "3": "bluetarget",
          "4": "cyantarget", "5": "magentatarget", "6": "yellowtarget",
          "0": "whitetarget", "@": "filledtarget"}


def _player_cell(eng, expert):
    for r, row in enumerate(eng.grid):
        for c, cell in enumerate(row):
            if cell & expert.player_ids:
                return (r, c)
    return None


def _plan_report() -> None:
    """Print the plan, its engine-verified win and its tie count for every level
    -- the quick "is this game still fully solved" check."""
    solver = ColorCombinationSolver()
    game, expert, solvable = solver._ensure(0)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            continue
        for direction in plan:                      # engine-verify the plan
            eng.step(direction)
        total += len(plan)
        sets = getattr(plan, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        print(f"  L{level}: {len(plan):3d} presses  win={eng.check_win()}  "
              f"{ties:3d}/{len(plan)} steps with a tie set  "
              f"{' '.join(d[0] for d in plan)}")
    print(f"  total {total} presses over {game.n_levels} levels")
    print(f"  solvable levels: {solvable}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(ColorCombinationSolver.main())
