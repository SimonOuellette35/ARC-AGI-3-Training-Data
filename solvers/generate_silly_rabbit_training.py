"""Generate Phase-1 training data for the PuzzleScript game ps:silly_rabbit
("Silly Rabbit", Jere Majava).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators, as does the exhaustive-enumeration oracle (`StateGraph` /
`PSEnumExpert`) this game plans with. This file is the game-specific part: the
measured mechanics, the reports that prove every level is winnable and shortest,
and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_silly_rabbit",
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
exactly. Every expert step carries the full set of equally-optimal presses (see
"Optimal-action sets" below).

The game
--------
The rabbit JUMPS TWO CELLS at a time. Reach the carrot (the Target) -- but a
jump overshoots it unless you line up exactly, so the puzzle is arriving on the
right parity, and the terrain is what lets you change parity: it is the
obstacles that give the rabbit a SHORT hop.

The whole ruleset is two lines::

    [ >  Player | no Tree | no tree no rock ] -> [ | | Player ]
    late [ player water ] -> [ water splash ]

and the second one is a death. Everything below was MEASURED against the
interpreter (``--moves`` re-measures it and asserts this table), not read off
the .txt -- the first rule is only half the story, because when it does NOT
match the interpreter still runs its ordinary movement phase and the rabbit
takes a plain ONE-cell step. Landing cell by (cell+1, cell+2) content, ``-``
meaning the press does nothing:

    cell+1 \\ cell+2   floor   rock    tree    water   target
    floor              +2      +1      +1      DEAD    +2
    rock               +2      -       -       DEAD    +2
    tree               -       -       -       -       -
    water              +2      DEAD    DEAD    DEAD    +2
    target             +2      +1      +1      DEAD    +2

Read off that table:

* **A tree beside the rabbit cancels the press outright.** It blocks the jump
  (the rule needs ``no Tree`` in the cell hopped over) and it blocks the
  one-cell fallback too (Player, Tree and Rock share a collision layer), so a
  tree is a wall in both senses.
* **A rock is jumped OVER but never landed ON.** A rock at cell+1 is no
  obstacle at all (``rock floor -> +2``); a rock at cell+2 kills the jump and
  demotes the press to the one-cell step that is the game's only way to change
  parity. A rock at BOTH is a no-op.
* **The board EDGE demotes the same way.** With cell+2 off the board the jump
  rule cannot match and the rabbit steps one cell; standing on the edge itself,
  the press does nothing. So the border is a fifth "rock", and on level 0 the
  plan uses it.
* **Water is only lethal where the rabbit LANDS.** ``water floor -> +2`` clears
  a canal in one hop, and level 2's plan does exactly that. But a press demoted
  to a one-cell step by a rock (or by the edge) drops the rabbit INTO the water
  it meant to clear -- ``water rock -> DEAD`` -- which is the trap the second
  message level is built around.
* **Death is not a restart, it is a dead board.** The late rule deletes the
  Player and leaves a Splash on the water; ``check_win`` (``all Target on
  Player``) can then never be satisfied, no rule ever re-creates a Player, and
  every later press is a no-op. The adapter does not call that GAME_OVER, so a
  drowned level simply stops responding until a RESET -- which is precisely what
  the recovery prefix records (see "Recovery").
* **The target is scenery until you stand on it.** It has its own collision
  layer, so it neither blocks nor stops a jump: ``target floor -> +2`` hops
  clean over the carrot without winning. Landing on it wins.
* Splash is created by the death rule only, and Water/Target/Tree/Rock are
  never moved or destroyed by anything, so the only thing that changes in a
  reachable state is where the rabbit is (or that it is gone).
* ACTION5 exists but no rule reads it, so the game is four directions --
  `directions` says so and the enumeration never branches on it.

The four levels are 6x7, 6x9 and two 7x9 (the game's two ``message`` screens are
not levels and the adapter does not count them). Level indices here and in every
report are 0-BASED.

Why an exhaustive oracle, and not the shared A*
-----------------------------------------------
Nothing on the board moves except the rabbit, so a state IS the rabbit's cell,
plus one absorbing dead board per cell it can drown on. That caps the space at
two states per cell and reaches far fewer -- 32, 41, 43 and 49 reachable per
level (16 and 17 of the last two's are drowned), in at most 196 engine steps.
Every level enumerates in under a twentieth of a second, all four in a tenth
altogether. So this generator does not search at all: `StateGraph.build`
enumerates the whole reachable component with the interpreter as the authority,
and a backward BFS from the winning edges gives the exact distance-to-win of
every state.

That buys three things a heuristic search would not:

* plans that are provably SHORTEST (no weight, no heuristic, no node cap that
  could bite): 8, 13, 15 and 18 presses;
* exact optimal-action SETS for free -- ``dist(succ) == dist - 1`` -- rather than
  inferred ones. Every expert step is labelled;
* a PROOF that a level is winnable-or-not rather than "the search gave up", and
  -- since drowning is irreversible -- an exact count of the states that can no
  longer win, which ``--bfs`` prints (levels 2 and 3 have them; levels 0 and 1,
  which have no water at all, have none).

Because the whole thing costs a tenth of a second there is no
``plan_cache_path``: every `parallelize_generator` shard re-derives the plans,
which is cheaper than the staleness risk a cached entry would carry.

Optimal-action sets
-------------------
``optsets[i]`` is every press ``d`` with ``dist(succ(state_i, d)) == dist(
state_i) - 1``, read straight off the exhaustive distance field, so it is exact
rather than inferred: a set is the complete list of shortest continuations, and a
press missing from it provably costs at least one more. ``--ties`` re-derives
every label independently by taking each candidate press and re-solving from the
state it lands in. Ties are rare here (3 of 54 plan steps) and that is the game:
a hop of two lands somewhere specific, so on most boards exactly one direction
keeps the rabbit on a shortest route.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the plan
replays to a guaranteed WIN. Here the RESET is a NECESSITY rather than a
convenience on the water levels -- an exploring rabbit that drowns is on a board
with no winning continuation and no press that does anything, so RESET is the
only key that works, and the prefix records exactly that lesson.

CLI
---
    --plans   per-level size, plan length and tie coverage
    --bfs     the exhaustive reachable-state report (the winnability proof)
    --moves   re-measure the (cell+1, cell+2) landing table in the docstring
    --ties    re-derive every optimal-action label by independent re-solve
    --audit   assert every cell composition renders distinctly, at every size
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                       # noqa: E402
    WIN as _WIN, PSAStarSolver, PSEnumExpert, StateGraph as _Graph,
    restore, snapshot,
)

GAME_NAME = "Silly_Rabbit"

#: The four presses. ACTION5 exists but no rule reads it (see the docstring),
#: and the order here is the plan's tie-break, so it is fixed.
_DIRS = ("up", "down", "left", "right")


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class SillyRabbitExpert(PSEnumExpert):
    """`PSEnumExpert` -- the exhaustive reachable-state oracle -- on four
    presses. See that class for why enumeration and not the shared A*, and the
    module docstring for why a game whose only moving object is the player has a
    space small enough to allow it.

    `_key` is inherited (every non-background cell), which is exact and
    canonical ACROSS levels: the water and the trees are in the key too, so two
    levels can never serve each other's plans and ``scope_by_level`` is
    unnecessary. It is also what makes a drowned board its own state -- the
    Player is gone and a Splash has appeared -- rather than one that aliases a
    live one.
    """

    #: No ACTION button is read by any rule; branching on it would add a quarter
    #: more interpreter steps for edges that are always self-loops.
    directions = list(_DIRS)


class SillyRabbitSolver(PSAStarSolver):
    game_id = "puzzlescript_silly_rabbit"
    game_name = GAME_NAME
    expert_cls = SillyRabbitExpert

    #: Record against the GAME FOLDER's adapter. ``games/ps:silly_rabbit`` is a
    #: plain passthrough today; it is named anyway so a sprite patch or a step
    #: cap added there later cannot silently make this generator tape a game
    #: nobody plays (the ps:count_mover trap).
    game_module_id = "ps:silly_rabbit"

    #: Runaway guard on the enumeration, not a tuning dial: the largest level in
    #: the game reaches 49 states.
    node_cap = 200_000

    #: Room for the longest plan (18 presses) plus the RESET exploration prefix
    #: and the re-plan after it. Stays under the adapter's own 200-step
    #: per-level budget, which would otherwise flip a level to GAME_OVER
    #: mid-plan.
    max_steps = 120


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = SillyRabbitSolver()
    game = solver.make_game(seed)
    expert = SillyRabbitExpert(game, node_cap=SillyRabbitSolver.node_cap)
    return solver, game, expert


def _report() -> int:
    """Per-level board size, inventory, plan length and tie coverage."""
    _solver, game, expert = _new()
    eng = game._engine
    ids = game._game.obj_name_to_idx
    total = tied = 0
    for level in range(game.n_levels):
        game.set_level(level)
        counts = {n: sum(1 for row in eng.grid for cell in row
                         if ids[n] in cell)
                  for n in ("tree", "rock", "water", "target")}
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        head = (f"level {level}: {eng.height:2d}x{eng.width:2d}, "
                f"{counts['tree']:2d} tree(s), {counts['rock']:2d} rock(s), "
                f"{counts['water']:2d} water, {counts['target']} target(s)")
        if found is None:
            print(f"{head} -- UNWINNABLE (see --bfs)")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        tied += ties
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(found):3d} presses (budget {game._max_steps}, "
              f"{room}), {ties:3d} with a tie set "
              f"({ties / max(1, len(found)):3.0%}), {dt:5.2f}s")
    print(f"total {total} presses, {tied}/{total} steps with a tie "
          f"({tied / max(1, total):.0%})")
    return 0


def _bfs() -> int:
    """The exhaustive reachable-state report -- and therefore the proof that the
    plans are SHORTEST, and the census of the states drowning strands.

    The enumeration terminates having generated every state the interpreter
    admits, so ``reachable - can-still-win`` is an exact count of the dead
    boards, not an estimate: on the water levels those are the drowned ones (the
    Player deleted, a Splash left behind), and on the dry levels there are none
    -- every press there is undoable, so the rabbit can never strand itself."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        t0 = time.time()
        graph = _Graph.build(eng, expert._key, expert.directions,
                             expert.node_cap)
        dt = time.time() - t0
        if graph is None:
            print(f"level {level}: node cap hit -- report is not a proof")
            bad += 1
            continue
        live = len(graph.dist)
        found = graph.plan(expert.directions)
        wins = sum(1 for e in graph.succ.values() if _WIN in e.values())
        verdict = (f"shortest {len(found)} presses" if found is not None
                   else "UNWINNABLE (no winning edge is reachable)")
        print(f"level {level}: {len(graph.succ):5d} reachable states "
              f"({live} can still win, {len(graph.succ) - live} stranded, "
              f"{wins} touch the target), {graph.steps:5d} engine steps, "
              f"{dt:5.2f}s -- {verdict}")
    return 0 if not bad else 1


#: The measured landing table of the docstring: ``(cell+1, cell+2) -> cells
#: moved``, with ``"dead"`` for a press that drowns the rabbit. Asserted by
#: `_moves`, which re-measures every entry against the interpreter.
_MOVE_TABLE = {
    ("floor", "floor"): 2, ("floor", "rock"): 1, ("floor", "tree"): 1,
    ("floor", "water"): "dead", ("floor", "target"): 2,
    ("rock", "floor"): 2, ("rock", "rock"): 0, ("rock", "tree"): 0,
    ("rock", "water"): "dead", ("rock", "target"): 2,
    ("tree", "floor"): 0, ("tree", "rock"): 0, ("tree", "tree"): 0,
    ("tree", "water"): 0, ("tree", "target"): 0,
    ("water", "floor"): 2, ("water", "rock"): "dead",
    ("water", "tree"): "dead", ("water", "water"): "dead",
    ("water", "target"): 2,
    ("target", "floor"): 2, ("target", "rock"): 1, ("target", "tree"): 1,
    ("target", "water"): "dead", ("target", "target"): 2,
}


def _moves() -> int:
    """Re-measure the movement table in the module docstring.

    Every entry is driven on the real interpreter over a board built for it, so
    this is the measurement itself rather than a restatement of it -- the only
    thing `_MOVE_TABLE` adds is that a change in the engine (or in the game
    file) fails here instead of silently making the docstring a lie.

    The boards are synthetic because the shipped levels do not contain all 25
    pairs; they are otherwise ordinary boards of the same objects, laid out on
    the largest level's dimensions. The two EDGE rows are the same measurement
    for "cell+2 is off the board" and "cell+1 is off the board", which is how
    the border demotes a jump.
    """
    _solver, game, _expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    bg, player = idx["background"], idx["player"]
    contents = ("floor", "rock", "tree", "water", "target")
    h, w = 5, 9

    def board(col: int, fills: dict) -> None:
        eng.height, eng.width = h, w
        eng.grid = [[{bg} for _ in range(w)] for _ in range(h)]
        eng.grid[2][col].add(player)
        for off, name in fills.items():
            if name != "floor":
                eng.grid[2][col + off].add(idx[name])
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    def rabbit():
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if player in cell:
                    return (r, c)
        return None

    bad = 0
    print(f"{'cell+1':8s} {'cell+2':8s}  measured  documented")
    for c1 in contents:
        for c2 in contents:
            board(2, {1: c1, 2: c2})
            eng.step("right")
            pos = rabbit()
            got = "dead" if pos is None else pos[1] - 2
            want = _MOVE_TABLE[(c1, c2)]
            flag = "" if got == want else "   <-- MISMATCH"
            bad += got != want
            print(f"{c1:8s} {c2:8s}  {str(got):8s}  {str(want):8s}{flag}")

    # The board edge, measured the same way: cell+2 off the board demotes the
    # jump to a one-cell step; cell+1 off the board is a no-op.
    for off, want in ((1, 1), (0, 0)):
        board(w - 1 - off, {})
        eng.step("right")
        pos = rabbit()
        got = -1 if pos is None else pos[1] - (w - 1 - off)
        bad += got != want
        note = "cell+2 off the board" if off else "cell+1 off the board"
        print(f"{note:17s}  {str(got):8s}  {str(want):8s}"
              f"{'' if got == want else '   <-- MISMATCH'}")
    print("moves clean" if not bad else f"MOVES FAILED: {bad} mismatches")
    return 0 if not bad else 1


def _ties() -> int:
    """Re-derive every optimal-action label the independent way: take each
    candidate press, then re-solve from the state it lands in. A press is
    optimal iff ``1 + len(replan)`` equals the plan length remaining, so this
    checks the distance field and the plan-following against a fresh enumeration
    from every state on the path rather than against itself."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level}: no plan (skipped)")
            continue
        checked = 0
        for pi, (taken, claimed) in enumerate(zip(plan, plan.optsets)):
            remaining = len(plan) - pi
            measured = []
            here = snapshot(eng)
            for d in expert.directions:
                restore(eng, here)
                eng.step(d)
                if eng.check_win():
                    cost = 1
                else:
                    rest = expert.plan(eng)
                    if rest is None:
                        continue           # drowned, or no winning continuation
                    cost = 1 + len(rest)
                if cost == remaining:
                    measured.append(d)
            restore(eng, here)
            if measured != list(claimed):
                print(f"  level {level} step {pi}: labelled {list(claimed)} "
                      f"but measured {measured}")
                bad += 1
            if taken not in claimed:
                print(f"  level {level} step {pi}: took {taken}, not in "
                      f"its own optimal set {list(claimed)}")
                bad += 1
            checked += 1
            eng.step(taken)                 # advance one press along the plan
        print(f"level {level}: {checked} steps verified")
    print("ties clean" if not bad else f"TIES FAILED: {bad} mismatches")
    return 0 if not bad else 1


def _reachable_compositions(eng, game, key_fn, directions) -> set:
    """Every cell composition (as a sorted tuple of object names) that appears
    in ANY reachable state of the level the engine is loaded with.

    Derived rather than hand-listed: this game's compositions are the terrain
    crossed with where the rabbit can get to and where it can drown, and a
    hand-list would go stale the moment a level or a sprite changed. It re-walks
    the space `StateGraph` walks (which keeps distances, not boards), which
    costs another ~200 engine steps per level -- nothing next to being sure the
    audited set is the set the agent can actually see."""
    inv = {v: k for k, v in game.obj_name_to_idx.items()}
    out: set = set()
    start = snapshot(eng)
    k0 = key_fn(eng)
    snaps = {k0: start}
    queue = deque([k0])
    while queue:
        k = queue.popleft()
        for direction in directions:
            restore(eng, snaps[k])
            eng.step(direction)
            for row in eng.grid:
                for cell in row:
                    out.add(tuple(sorted(inv[o] for o in cell)))
            nk = key_fn(eng)
            if nk != k and nk not in snaps:
                snaps[nk] = snapshot(eng)
                queue.append(nk)
    for snap in snaps.values():             # the states themselves, incl. start
        for row in snap:
            for cell in row:
                out.add(tuple(sorted(inv[o] for o in cell)))
    restore(eng, start)
    return out


def _audit() -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every cell size the boards use.

    Whole 64x64 frames of uniform boards are compared rather than one cell out of
    a mixed board: `_render_frame` centre-pads a non-square board, so indexing a
    cell by ``cell_px * r`` silently reads the wrong pixels on the wide levels
    (the ps:explod lesson). Two uniform boards render identically iff their cells
    do.

    The compositions come from `_reachable_compositions`, i.e. from the
    enumeration, so the audit is about boards this game can actually reach.
    Three of the eight are worth naming:

    * ``player + target`` is the WIN frame: the rabbit's 5x5 sprite leaves the
      Target's ring of pixels showing at the corners, so the winning square does
      not read as a bare rabbit.
    * ``splash + water`` is the DEATH frame, and it has to be distinguishable
      from plain water, because it is the only thing on the board that says the
      episode now needs a RESET.
    * The PuzzleScript palette collapses (see the ps-palette-collisions note):
      Tree's darkgreen and green are one ARC green, and Background's brown is
      the same index as Tree's darkbrown trunk. The tree still reads as a green
      blob and the rock as a grey one, and this is the test of that.

    ``player + water`` is absent for an invariant rather than luck: the late
    death rule fires in the same tick the rabbit lands, so no settled frame ever
    shows a rabbit standing on water.
    """
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    comps: set = set()
    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)
        comps |= _reachable_compositions(eng, g, expert._key,
                                         expert.directions)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for comp in sorted(comps):
            eng.height, eng.width = h, w
            eng.grid = [[{idx[o] for o in comp} | {idx["background"]}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[comp] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        note = "OK" if not clashes else f"IDENTICAL {clashes}"
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(shots)} "
              f"compositions -- {note}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--bfs" in sys.argv:
        sys.exit(_bfs())
    if "--moves" in sys.argv:
        sys.exit(_moves())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(SillyRabbitSolver.main())
