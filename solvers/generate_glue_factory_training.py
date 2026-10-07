"""Generate Phase-1 training data for the PuzzleScript game ps:glue_factory
(Cale Bradbury's "Glue Factory", 4 levels).

The harness -- the rotation contract, the trajectory recorder, the plan memo, the
on-disk plan cache and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: the goal heuristic, the provably-unwinnable test, and the
exact optimal-action SETS the plans ship as training targets.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_glue_factory",
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
exactly.


The game
--------
The board holds a Player, a Shooter, some Holes and some Glue. The win condition
is ``No Hole`` AND ``No Glue`` -- fill every hole and leave nothing behind. Every
level ships with FEWER glue blobs than holes, so glue must be manufactured.

Four rules carry the whole game, and none of them is what it looks like:

1.  ``[ > Player | Shooter | no wall ] -> [ > Player | > Shooter | ]``
    The player shoves the Shooter one cell, exactly like a sokoban crate. This is
    the only way to make the Shooter move, and the Shooter moving is the only way
    to make Glue move.

2.  ``[MOVING Shooter|...|Glue] -> [MOVING Shooter|...|MOVING Glue]``
    The Shooter has a CROSSHAIR -- its entire row plus its entire column, which
    the `Guide` tiles draw. The rule is direction-agnostic and its ``...`` spans
    walls, so any glue anywhere on either arm is flung one cell **in the
    direction the Shooter was just pushed** -- NOT along the ray it was found on.
    A shooter shoved sideways therefore drags glue sitting five cells ABOVE it
    sideways as well, through walls, across sealed chambers. Level 0 is nothing
    but that: the hole and the only glue blob sit in a corridor the player can
    never enter.

    The useful invariant: once a blob shares the Shooter's row, pushing the
    Shooter along that row's perpendicular moves both in lockstep, so they STAY
    aligned. Alignment is sticky; getting aligned is the puzzle.

3.  ``[> Glue|Wall] -> [|Wall]``
    Glue flung INTO a wall is destroyed. This is the disposal mechanism, and the
    win needs it: ``No Glue`` means every surplus blob has to be thrown away.

4.  ``[< Glue|Wall] -> [< Glue |< Glue wall]``
    Glue flung AWAY FROM a wall directly behind it leaves a COPY on the wall
    tile. This is the game's only glue source, so a level with three holes and
    one blob is solved by shoving that blob against a wall and peeling copies off
    it, one per push.

Plus ``late [player Glue] -> cancel`` / ``late [player Hole] -> cancel`` (the
player may not stand on either) and ``late [Glue Hole] -> []`` (a blob landing in
a hole annihilates it, which is how ``No Hole`` is reached).

ACTION5 appears in no rule, so `GlueFactoryExpert.directions` is the four moves
and the search never branches on a press that cannot change anything.


Rendering: four objects were invisible, one of them a whole mechanic
--------------------------------------------------------------------
The art shipped in `data/puzzlescript_games/Glue_Factory.txt` did not survive
adaptation to the 16-colour ARC palette. ``--audit`` is the check -- it fills a
whole board with each cell COMPOSITION and compares WHOLE FRAMES, because
`_render_frame` upscales the board to fill 64x64 and letterboxes it, so a
``cell_px``-arithmetic crop reads the wrong window (the mistake that made older
audits in this tree report every composition identical to floor). What it found:

  * ``Shooter = Orange`` and ``Wall = BROWN`` both quantize to ARC **12**, and
    both were solid 5x5 blocks: the Shooter -- the piece the player pushes, and
    the piece whose position aims the entire crosshair -- was pixel-identical to
    the walls it stands between. Repainted ``Purple``.
  * Glue sits on a LOWER collision layer than Wall and Shooter, so ``wall+glue``
    rendered as a plain wall and ``shooter+glue`` as a plain shooter. A blob
    duplicated onto a wall by rule 4 was therefore INVISIBLE -- and that blob is
    a real object which the win condition will not let you leave behind. Wall and
    Shooter are now RINGS with a hollow 3x3 middle, which is where the glue under
    them shows through (white middle = loaded, grey = empty).
  * ``Guide``'s corner dots were authored ``#808080``, which quantizes to ARC 2
    -- the exact colour of the ``Gray`` Background. The crosshair, i.e. the only
    on-screen statement of which glue is in range, was drawn in the background
    colour. Repainted in the author's own unused palette entry, ``#ff0000``.
  * The Player was solid and composites LAST (over every layer, including
    Guide), so the crosshair vanished under it. Given transparent corners, which
    are exactly the pixels Guide paints.

The corners carry the Guide and the hollow middle carries the glue, so the two
readings never overwrite each other. Every composition the game can produce is
now pairwise distinct at both cell sizes its levels use, with one exception that
cannot be observed: ``hole+glue``, which ``late [Glue Hole] -> []`` annihilates
inside the same tick, so it never reaches a settled frame. ``--audit`` asserts
that exception is the only one, and ``--verify`` proves the state never settles.


The expert
----------
Engine-blackbox A* over the real interpreter, at ``weight = 1`` with an
admissible+consistent heuristic, so every plan is a shortest one. The heuristic
is::

    h = max over holes of (Manhattan distance to the nearest glue blob)

and 1 when the holes are gone but glue remains. It is a lower bound because glue
propagates at most one cell per press *from where glue already is*: rule 2 moves
a blob exactly one cell, and rule 4's copy is born on the wall tile orthogonally
adjacent to its parent. So after t presses every blob on the board is within
Manhattan t of some blob present now, and a hole D away cannot be filled in fewer
than D presses. Holes are never created and never move, which is what makes the
bound stable. It is consistent by the same argument: a hole that disappears was
at distance <= 1, so h can fall by at most 1 per press.

`dead()` drops the states with holes left and no glue at all -- irreversibly
lost, since rule 4 is the only glue source and it needs a blob to copy.

Optimal-action sets are EXACT, not "the one path the search happened to find".
Once A* has proved ``d*``, `_exact_sets` re-sweeps a layered BFS bounded by
``g + h <= d*`` -- admissible h, so this keeps every state on a shortest path and
almost nothing else (level 3: 24.6k states against the 174k an unbounded BFS
visits) -- and runs a reverse DP over it for the true distance-to-win. A press is
in the set iff it is on some shortest path. Both passes cost about one A*, and
``--ties`` re-derives the answer from scratch with an independent A* per
candidate press.

The whole search is seed-independent (the levels are fixed ASCII maps; only the
PRESENTATION is augmented per seed), so it is paid once and cached to
``data/glue_factory_plans.json``; later seeds and later `parallelize_generator`
shards replay it and only re-render.


Augmentation
------------
Per (seed, level) frame rotation k in {0,1,2,3} plus independent horizontal and
vertical flips, each with the matching directional action remap. Glue Factory is
gravity-free and every one of its rules is written in relative directions
(``>``/``<``/``MOVING``), so a flip is an exact symmetry of the mechanic -- hence
the game is added to `PuzzleScriptAdapter._FLIP_GAMES` and each level re-samples
the board's full 8-element symmetry group. The expert plans in ENGINE space and
`ps_astar.screen_action` emits the SCREEN press.


Usage (run from the repo root):
    python solvers/generate_glue_factory_training.py --episodes 200 \
        --out data/training_multi_level/glue_factory

    python solvers/generate_glue_factory_training.py --plans
    python solvers/generate_glue_factory_training.py --ties
    python solvers/generate_glue_factory_training.py --verify
    python solvers/generate_glue_factory_training.py --audit
"""

from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput                                    # noqa: E402
from adapters.puzzlescript_adapter import _render_frame              # noqa: E402
from solvers.common.ps_astar import (                                # noqa: E402
    PSAStarSolver, PSExpert, Plan, restore, screen_action, snapshot,
)

GAME_NAME = "Glue_Factory"
PLAN_CACHE = (Path(__file__).resolve().parent.parent
              / "data" / "glue_factory_plans.json")

#: Sentinels in the bounded-BFS adjacency rows built by `_exact_sets`.
_WIN = -2        # this press wins outright
_PRUNED = -1     # dead, or provably off every shortest path
_INF = 1 << 30


class GlueFactoryExpert(PSExpert):
    """A* over the interpreter, with exact optimal-action sets.

    ``directions`` drops ACTION5: it appears in no rule of this game, so
    branching on it only pays to discover a no-op four times per expansion.
    """

    directions = ["up", "down", "left", "right"]
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        self.hole_ids = set(self.g.resolve_object_name("hole"))
        self.glue_ids = set(self.g.resolve_object_name("glue"))
        # `_ckey` packs each cell's object set into ONE byte. Seven objects here;
        # the assert is what turns a future eighth into a crash rather than into
        # silently aliased states.
        assert len(self.g.obj_name_to_idx) <= 8, "cell bitmask no longer fits a byte"

    # -- goal knowledge -------------------------------------------------------
    def _scan(self, eng) -> tuple[list, list]:
        """``(hole cells, glue cells)``."""
        holes, glues = [], []
        hole_ids, glue_ids = self.hole_ids, self.glue_ids
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if cell & hole_ids:
                    holes.append((r, c))
                if cell & glue_ids:
                    glues.append((r, c))
        return holes, glues

    def heuristic(self, eng) -> int:
        """Admissible and consistent; see the module docstring for the argument.

        Distances are plain Manhattan with walls ignored, which is not a
        relaxation to apologise for: rule 2 flings glue THROUGH walls, so wall
        distance would not be a lower bound at all.
        """
        holes, glues = self._scan(eng)
        if not holes:
            return 1 if glues else 0            # only the No Glue half is left
        if not glues:
            return _INF                         # `dead` normally gets here first
        return max(min(abs(r - gr) + abs(c - gc) for gr, gc in glues)
                   for r, c in holes)

    def dead(self, eng) -> bool:
        """Holes left and no glue anywhere: provably lost.

        ``[< Glue|Wall]`` is the only rule that creates glue and it needs an
        existing blob to copy, so an empty board can never make another one. The
        state would otherwise sit in the queue forever wearing the heuristic's
        sentinel, and glue is destroyed by a wall often enough that this is a
        real slice of the branching.
        """
        holes, glues = self._scan(eng)
        return bool(holes) and not glues

    def _ckey(self, eng) -> bytes:
        """Compact canonical state for the bounded BFS: one byte per cell.

        `_key`'s frozenset of triples is what the memo and the disk cache's
        staleness check need (`PSExpert.plan` reads it as ``(r, c, o)``), but a
        24k-state sweep wants something cheaper to build, hash and hold.
        """
        return bytes(sum(1 << o for o in cell)
                     for row in eng.grid for cell in row)

    # -- planning -------------------------------------------------------------
    def _search(self, eng) -> list | None:
        """A* for the shortest length, then `_exact_sets` for the tie sets."""
        start = snapshot(eng)
        plan = self._astar(eng)                  # leaves the grid DIRTY
        restore(eng, start)
        if plan is None:
            return None
        presses, optsets = self._exact_sets(eng, len(plan))
        restore(eng, start)
        return Plan(presses, optsets)

    def _exact_sets(self, eng, dstar: int) -> tuple[list, list]:
        """Every press that is equally shortest, at every step of one plan.

        Two passes over the shortest-path subgraph:

        FORWARD -- a layered BFS from the start keeping only states with
        ``g + h <= d*``. With h admissible that bound cannot drop a state on a
        shortest path (such a state has ``g + h <= g + h* = d*``), and it drops
        nearly everything else. The layer index is each state's true minimal g,
        because BFS discovers it first at that depth.

        BACKWARD -- a DP over the layers in reverse for the exact
        distance-to-win, following only edges that advance a layer. An edge to a
        state at layer ``<= g`` can never be on a shortest path from here, so
        ignoring it costs nothing and keeps the DP acyclic.

        The distances are then exact on the shortest-path set (a state on one has
        its whole continuation inside the bound) and only ever OVER-estimates
        elsewhere, which is the safe direction: a press is admitted iff it really
        does keep the plan shortest.
        """
        if dstar == 0:
            return [], []
        dirs = self.directions
        ids = {self._ckey(eng): 0}
        depth_of = [0]                              # sid -> minimal g
        adj: dict[int, list[int]] = {}
        expanded: list[list[int]] = [[] for _ in range(dstar)]
        layer = [(0, snapshot(eng))]

        for depth in range(dstar):
            nxt = []
            for sid, snap in layer:
                row = []
                for direction in dirs:
                    restore(eng, snap)
                    eng.step(direction)
                    if eng.check_win():
                        row.append(_WIN)
                        continue
                    if self.dead(eng) or depth + 1 + self.heuristic(eng) > dstar:
                        row.append(_PRUNED)
                        continue
                    k = self._ckey(eng)
                    cid = ids.get(k)
                    if cid is None:
                        cid = len(ids)
                        ids[k] = cid
                        depth_of.append(depth + 1)
                        nxt.append((cid, snapshot(eng)))
                    row.append(cid)
                adj[sid] = row
                expanded[depth].append(sid)
            layer = nxt

        dist = [_INF] * len(ids)
        for depth in range(dstar - 1, -1, -1):
            for sid in expanded[depth]:
                best = _INF
                for cid in adj[sid]:
                    if cid == _WIN:
                        best = min(best, 1)
                    elif cid >= 0 and depth_of[cid] == depth + 1:
                        best = min(best, dist[cid] + 1)
                dist[sid] = best
        assert dist[0] == dstar, f"reverse DP says {dist[0]}, A* proved {dstar}"

        presses, optsets, sid = [], [], 0
        for depth in range(dstar):
            remaining = dist[sid]
            opts, step_to = [], {}
            for direction, cid in zip(dirs, adj[sid]):
                cost = (1 if cid == _WIN else
                        dist[cid] + 1 if cid >= 0 and depth_of[cid] == depth + 1
                        else _INF)
                if cost == remaining:
                    opts.append(direction)
                    step_to[direction] = cid
            assert opts, f"no optimal press at depth {depth}"
            presses.append(opts[0])
            optsets.append(opts)
            sid = step_to[opts[0]]
        return presses, optsets


class GlueFactorySolver(PSAStarSolver):
    game_id = "puzzlescript_glue_factory"
    game_name = GAME_NAME
    expert_cls = GlueFactoryExpert

    #: The longest plan is 31 moves; the rest is room for the exploration prefix
    #: and the re-plan after the RESET that ends it. Well under the adapter's
    #: 200-step per-level budget, which would flip a level to GAME_OVER mid-plan.
    max_steps = 120

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time A* visible as startup rather
        than as a mysteriously slow first seed, and it fills the disk cache in
        one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with the adapter parked at level 0."""
    solver = GlueFactorySolver()
    game = solver.make_game(0)
    return solver, game, GlueFactoryExpert(game)


def _report() -> int:
    """Per-level board size, piece counts, plan length and tie coverage -- and
    CERTIFY each plan by replaying it through the interpreter."""
    _solver, game, expert = _levels()
    total = bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        holes, glues = expert._scan(eng)
        t = time.time()
        plan = expert.plan(eng, level)
        took = time.time() - t
        if plan is None:
            print(f"level {level}: UNSOLVABLE")
            bad += 1
            continue
        # The interpreter must win on the LAST press and on no earlier one -- an
        # earlier win would mean the plan is not shortest.
        won_at = None
        for i, direction in enumerate(plan):
            eng.step(direction)
            if eng.check_win():
                won_at = i
                break
        ok = won_at == len(plan) - 1
        bad += not ok
        total += len(plan)
        ties = sum(1 for s in getattr(plan, "optsets", ()) if len(s) > 1)
        print(f"level {level}: {eng.height}x{eng.width} "
              f"{len(holes)} holes / {len(glues)} glue, "
              f"{len(plan):3d} moves in {took:6.2f}s, "
              f"{ties:3d} tie steps ({ties / len(plan):3.0%}), "
              f"{'CERTIFIED' if ok else 'PLAN REJECTED BY THE INTERPRETER'}")
    print(f"total {total} moves over {game.n_levels} levels")
    return 0 if not bad else 1


class _LengthOnlyExpert(GlueFactoryExpert):
    """`GlueFactoryExpert` minus `_exact_sets`: bare A*, shortest length only.

    `--ties` asks for the distance-to-win of every successor of every plan step,
    which is a few hundred searches. Running the production `_search` for each
    would re-derive a whole bounded-BFS tie sweep per probe -- the very answer
    the check is supposed to be independent of, and orders of magnitude more work
    than the question needs.
    """

    plan_cache_path = None               # never write a probe answer to disk

    def _search(self, eng) -> list | None:
        return self._astar(eng)


def _ties(levels: list[int] | None = None) -> int:
    """Double-entry check of the optimal-action SETS.

    `_exact_sets` derives them from a bounded layered BFS plus a reverse DP.
    Here the same question is answered from the other end and with no shared
    code: walk the plan, and at each state run an INDEPENDENT A* from each of
    the four successors. A press ties iff ``1 + d*(successor) == remaining``.
    Disagreement in either direction is a failure -- a missing press
    under-labels the step, a spurious one teaches a move that is not shortest.
    """
    _solver, game, expert = _levels()
    probe = _LengthOnlyExpert(game)      # its own memo, so nothing is inherited
    levels = list(range(game.n_levels)) if levels is None else levels
    bad = 0
    for level in levels:
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            continue
        t = time.time()
        mismatches, shortcuts = [], []
        for i, opts in enumerate(plan.optsets):
            # The plan is shortest and we are ON it, so the true distance to the
            # win from here is exactly this.
            remaining = len(plan) - i
            here = snapshot(eng)
            measured = []
            for direction in expert.directions:
                restore(eng, here)
                eng.step(direction)
                if eng.check_win():
                    cost = 1
                elif probe.dead(eng):
                    cost = _INF
                else:
                    sub = probe._plan_memoized(eng, level)
                    cost = _INF if sub is None else 1 + len(sub)
                if cost < remaining:
                    # A press that beats the plan: d* itself is wrong.
                    shortcuts.append((i, direction, cost, remaining))
                if cost == remaining:
                    measured.append(direction)
            restore(eng, here)
            if measured != opts:
                mismatches.append((i, opts, measured))
            eng.step(plan[i])
        bad += len(mismatches) + len(shortcuts)
        note = ("sets exact" if not mismatches and not shortcuts else
                (f"{len(mismatches)} MISMATCH: {mismatches[:3]} " if mismatches else "")
                + (f"{len(shortcuts)} SHORTCUT: {shortcuts[:3]}" if shortcuts else ""))
        print(f"level {level}: {len(plan)} steps re-solved in "
              f"{time.time() - t:6.1f}s -- {note}", flush=True)
    print("ties clean" if not bad else f"TIES FAILED: {bad} steps disagree")
    return 0 if not bad else 1


def _verify(budget: int = 120_000) -> int:
    """Two invariants the rest of the file leans on, both measured.

    1.  ``hole+glue`` -- the one cell composition `--audit` cannot tell apart --
        never reaches a settled frame. ``late [Glue Hole] -> []`` should
        annihilate it inside the tick it is created in, so a breadth-first sweep
        of the reachable space must never see the two co-located after a step.
        The sweep is budgeted (`budget` states, unpruned otherwise) and reports
        the depth it got to, because the full ball at level 3's d* = 31 is a
        half-million states and this is a check, not a search.
    2.  A press is one TICK: the adapter returns a single frame per action, so
        the recording needs no ``record_spans``. Nothing in the ruleset uses
        ``again``, and this is the check of that.
    """
    solver, game, expert = _levels()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            continue

        # (1) unpruned BFS over the reachable states, watching for a settled cell
        # holding both a hole and a glue.
        seen = {expert._ckey(eng)}
        layer = [snapshot(eng)]
        stacked = depth = 0
        while layer and len(seen) < budget and depth < len(plan):
            nxt = []
            for snap in layer:
                for direction in expert.directions:
                    restore(eng, snap)
                    eng.step(direction)
                    if any(cell & expert.hole_ids and cell & expert.glue_ids
                           for row in eng.grid for cell in row):
                        stacked += 1
                    if eng.check_win():
                        continue
                    k = expert._ckey(eng)
                    if k in seen:
                        continue
                    seen.add(k)
                    nxt.append(snapshot(eng))
            layer = nxt
            depth += 1
        bad += stacked

        # (2) one frame per press, along the plan.
        game.set_level(level)
        spans = set()
        for direction in plan:
            act = screen_action(direction, game._rotation_k, game._hflip,
                                game._vflip, solver.remap_actions)
            spans.add(len(game.perform_action(ActionInput(id=act)).frame))
        multi = spans - {1}
        bad += len(multi)
        print(f"level {level}: {len(seen):6d} states swept to depth {depth}"
              f"{'' if depth == len(plan) else ' (budget)'}, "
              f"hole+glue settled {stacked} times "
              f"({'never observable' if not stacked else 'OBSERVABLE -- FIX THE ART'}), "
              f"frames per press {sorted(spans)} "
              f"({'single tick' if not multi else 'NEEDS record_spans'})")
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} violations")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    Four separate invisibilities shipped in this game's art and not one of them
    is findable by comparing objects: the Shooter collided with the Wall in the
    palette, glue UNDER a wall or a shooter was hidden by the layer above it, and
    the Guide was drawn in the background's own colour. It is the STACK that has
    to be checked, one frame per composition.

    Whole frames are compared, not cell crops: `_render_frame` upscales the board
    to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (exactly
    what made older audits in this tree report every composition identical to
    floor).

    ``hole+glue`` is the one deliberate exception -- ``late [Glue Hole] -> []``
    destroys the pair inside the tick that creates it, so no frame can show it.
    ``--verify`` is the proof; this only records that it is expected.
    """
    _solver, game, _expert = _levels()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {
        "floor": (), "wall": ("wall",), "glue": ("glue",), "hole": ("hole",),
        "player": ("player",), "shooter": ("shooter",),
        "wall+glue": ("wall", "glue"),                 # rule 4's duplicate
        "shooter+glue": ("shooter", "glue"),
        "shooter+hole": ("shooter", "hole"),
        "hole+glue": ("hole", "glue"),                 # transient, see docstring
        "floor+guide": ("guide",), "wall+guide": ("wall", "guide"),
        "glue+guide": ("glue", "guide"), "hole+guide": ("hole", "guide"),
        "player+guide": ("player", "guide"),
        "shooter+guide": ("shooter", "guide"),
        "wall+glue+guide": ("wall", "glue", "guide"),
    }
    expected = {("hole", "hole+glue")}

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
        unexpected = [p for p in clashes if p not in expected]
        bad += len(unexpected)
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{len(comps)} compositions, {len(clashes)} identical pair(s) "
              f"{'(all expected)' if not unexpected else 'UNEXPECTED ' + str(unexpected)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(GlueFactorySolver.main())
