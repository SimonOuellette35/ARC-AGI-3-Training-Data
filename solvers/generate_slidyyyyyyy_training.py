"""Generate Phase-1 training data for the PuzzleScript game ps:slidyyyyyyy
("Slidyyyyyyy", mokesmoe).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators, as does the exhaustive-enumeration oracle (`StateGraph` /
`PSEnumExpert`) this game plans with. This file is the game-specific part: the
measured mechanics, the reports that prove every level is winnable and shortest,
the proof that the game-folder rule patch changes nothing, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_slidyyyyyyy",
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
You cannot take a step. One press launches you and you slide until a wall stops
you. You win by coming to rest ON the target -- which means the target is only
usable from the directions that have a wall directly behind it, and a run that
merely passes over it is worth nothing. Five levels, no other object, no way to
die.

Everything below was MEASURED against the interpreter, not read off the .txt:

* **What stops the slide is Wall, and only Wall.** The move is
  ``[ > player | no wall ] -> [ | > player ]`` (see "The one game-folder patch"
  below for why it is spelled that way here): the player advances while the next
  cell is not a wall, and the force left over at the end is refused by the wall.
  Every level is fully bordered, so a slide always terminates.
* **The target does NOT stop the slide, and is not on the player's collision
  layer.** COLLISIONLAYERS puts Target on its own layer between Background and
  Player, and no rule in the file mentions Target at all -- the only thing that
  reads it is the win condition. Parked at (1,1) on level 1 with the target at
  (1,3) and floor out to (1,8), ``right`` finishes at (1,8), having crossed the
  target without noticing it. So the target is enterable only from a direction
  whose far side is a wall, and the level is the problem of arriving from one of
  those. In practice that is a needle: ``--bfs`` counts the winning edges over
  the whole reachable space and finds ONE on each of levels 1, 2, 3 and 5 and
  two on level 4 (both the same press from two cells of one corridor). Out of
  the 124 board states this game can reach, exactly six presses win.
* **Pressing into a wall is a pure no-op, not a death.** The rule declines to
  match and the leftover force is refused; the grid compares equal afterwards.
  There is no ``restart``, no ``cancel`` and no lose condition anywhere in the
  ruleset, so no press can ever end an episode.
* **But the game still has dead ends** -- see ``--bfs``. Level 3 has 24
  reachable states of which only 17 can still reach the target; the other 7 are
  a closed orbit that no press leaves, and it is three presses from the level
  start. Nothing marks it -- no death, no restart, no visual tell -- which is
  exactly why the RESET recovery arc below is not decoration on this game.
* **The win is ``All player on target`` AND ``no marker``.** With the animation
  collapsed no marker is ever created, so the second condition is trivially true
  and the win is "stopped on the target". The interpreter's
  "a deleted player is not a win" guard is never exercised: nothing in this game
  removes the player.
* ACTION5 exists in the adapter's action list but no rule reads it -- an action
  turn sets no directional force, so ``[ > player | ... ]`` cannot match and the
  board is unchanged. `directions` says so and the enumeration never branches on
  it. (The exploration prefix still presses it; a no-op press is honest
  recovery data.)

The one game-folder patch, and why the generator records through it
------------------------------------------------------------------
``games/ps:slidyyyyyyy/ps:slidyyyyyyy.py`` replaces the source's 22 rules with
the single move they animate. The source spells one slide as a 5-frames-per-cell
cel animation driven by ``again``; this interpreter caps an ``again`` chain at 50
iterations, so it can only finish a TEN-cell slide, and level 5's row 3 is a
thirteen-cell corridor. Crossing it truncates: the player stops two cells short
with a live marker on the board, the marker's sprite is blank so the frame looks
honest, ``no marker`` quietly makes that board unwinnable, and the next press --
any of the four -- resumes the abandoned slide instead of doing what it says.

``--rules`` is the proof that the replacement is a restatement and not a change:
it enumerates the whole reachable space of all five levels under BOTH rulesets
and compares every transition and every win flag. Levels 1-4 agree on all 392;
level 5 differs on exactly the 2 that overrun the cap, and there the patched
engine finishes the slide the original abandoned. ``game_module_id`` makes this
generator record against that wrapper, which is also what `game_envs` hands a
live agent, so the tape and the game are the same object.

Rendering
---------
The Target sprite is redrawn in ``data/puzzlescript_games/Slidyyyyyyy.txt``
(comment at the top of that file; ``--audit`` is the check). As shipped it was a
small ring on rows/cols 1-3 -- precisely the pixels the Player's sprite paints
over -- so ``player on target``, the winning board, rendered as a bare player and
the win was invisible. And level 5 is 8x17, which renders at cell_px=3, where the
sampling reads sprite rows/cols {0,2,4} only: every pixel of the old ring fell
between them, so on that level the goal itself rendered as bare floor. Target now
occupies the nine pixels the Player leaves transparent, five of them on rows/cols
{0,2,4}, so it is visible at all three cell sizes this game uses (3, 5, 6) and
cannot be occluded.

Why an exhaustive oracle, and not the shared A*
-----------------------------------------------
A press is a whole slide, so the only states are the cells the player can come to
REST on: 17-29 states per level, reached in at most 116 engine steps. Every level
is enumerated in under a twentieth of a second and all five together in about a
tenth. So this generator does not search at all -- `StateGraph.build` enumerates
the whole reachable component with the interpreter as the authority, and a
backward BFS from the winning edges gives the exact distance-to-win of every
state. Same argument as ps:roller_boi, the other slide game here.

That buys three things a heuristic search would not:

* plans that are provably SHORTEST (no weight, no heuristic, no node cap that
  could bite): 11, 9, 9, 11 and 13 presses;
* exact optimal-action SETS for free -- ``dist(succ) == dist - 1`` -- rather than
  inferred ones. Every expert step is labelled;
* a PROOF of which states can still win, which is what identifies level 3's
  seven-state dead orbit rather than leaving it as "the search gave up".

Level indices in the reports are 0-BASED, i.e. one less than the number in the
game's own "Level N" messages; the prose above uses the game's numbering.

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
state it lands in. There is not a single tie anywhere in this game (0 of 53 plan
steps), and that is the mechanic rather than luck: a press commits to a whole
corridor, so from a given rest cell the four presses land in four different
places and only one of them is on a shortest route.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the plan
replays to a guaranteed WIN. Unlike ps:roller_boi the RESET is doing real work
here -- level 3's dead orbit (``--bfs``) is three presses from the start and
cannot be escaped by playing on, so a prefix that wanders into it has no
continuation except the reset.

CLI
---
    --plans   per-level size, plan length and tie coverage
    --bfs     the exhaustive reachable-state report (winnability + dead ends)
    --ties    re-derive every optimal-action label by independent re-solve
    --rules   patched vs. original ruleset over the whole reachable space
    --audit   assert every cell composition renders distinctly, at every size
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                       # noqa: E402
    WIN as _WIN, PSAStarSolver, PSEnumExpert, StateGraph as _Graph,
    restore, snapshot,
)

GAME_NAME = "Slidyyyyyyy"
GAME_MODULE = "ps:slidyyyyyyy"

#: The four presses. ACTION5 exists but no rule reads it (see the docstring),
#: and the order here is the plan's tie-break, so it is fixed.
_DIRS = ("up", "down", "left", "right")


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class SlidyExpert(PSEnumExpert):
    """`PSEnumExpert` -- the exhaustive reachable-state oracle -- on four
    presses. See that class for why enumeration and not the shared A*, and the
    module docstring for why a slide-until-you-stop game's space is small enough
    to allow it.

    `_key` is inherited (every non-background cell), which is exact and
    canonical ACROSS levels: nothing in this game is created or destroyed, so
    the key is just "where the walls, the target and the player are", and the
    walls differ between every pair of levels. Two levels can therefore never
    serve each other's plans and ``scope_by_level`` is unnecessary.
    """

    #: No ACTION button is read by any rule; branching on it would add a quarter
    #: more interpreter steps for edges that are always self-loops.
    directions = list(_DIRS)


class SlidySolver(PSAStarSolver):
    game_id = "puzzlescript_slidyyyyyyy"
    game_name = GAME_NAME
    expert_cls = SlidyExpert

    #: Record against the GAME FOLDER's adapter, which is NOT a passthrough here:
    #: it collapses the slide animation the interpreter's ``again`` cap cannot
    #: finish (see the module docstring and ``--rules``). A generator that built
    #: its own `PuzzleScriptAdapter` would tape a level 5 nobody plays.
    game_module_id = GAME_MODULE

    #: Runaway guard on the enumeration, not a tuning dial: the largest level in
    #: the game reaches 29 states.
    node_cap = 200_000

    #: Room for the longest plan (13 presses) plus the RESET exploration prefix
    #: and the re-plan after it. Stays under the adapter's own 200-step
    #: per-level budget, which would otherwise flip a level to GAME_OVER
    #: mid-plan.
    max_steps = 120


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = SlidySolver()
    game = solver.make_game(seed)
    return solver, game, SlidyExpert(game, node_cap=SlidySolver.node_cap)


def _report() -> int:
    """Per-level board size, plan length and tie coverage."""
    _solver, game, expert = _new()
    eng = game._engine
    total = tied = steps = 0
    for level in range(game.n_levels):
        game.set_level(level)
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        head = f"level {level}: {eng.height:2d}x{eng.width:2d}"
        if found is None:
            print(f"{head} -- UNWINNABLE (see --bfs)")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        tied += ties
        steps += len(found)
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(found):3d} presses (budget {game._max_steps}, "
              f"{room}), {ties:3d} with a tie set "
              f"({ties / max(1, len(found)):3.0%}), {dt:5.2f}s")
    print(f"total {total} presses, {tied}/{steps} steps with a tie "
          f"({tied / max(1, steps):.0%})")
    return 0


def _bfs() -> int:
    """The exhaustive reachable-state report -- and therefore the proof that the
    plans are SHORTEST: the enumeration terminates having generated every state
    the interpreter admits, so the backward distance field is exact.

    It is also the DEAD-END report, which matters more here than on ps:roller_boi.
    Nothing in this game kills you, but a slide maze can still strand you: level
    3 (index 2) has 7 reachable states from which no sequence of presses reaches
    the target, and they are invisible -- the board looks like any other. That is
    what the RESET recovery arc exists for."""
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
        stuck = len(graph.succ) - live
        verdict = (f"shortest {len(found)} presses" if found is not None
                   else "UNWINNABLE (no winning edge is reachable)")
        print(f"level {level}: {len(graph.succ):5d} reachable states "
              f"({live} can still win, {stuck} stranded, {wins} touch the "
              f"target), {graph.steps:5d} engine steps, {dt:5.2f}s -- {verdict}")
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
                        continue
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


def _rules() -> int:
    """Proof that the game-folder rule patch is a RESTATEMENT of the animation.

    Enumerate the whole reachable space under the PATCHED ruleset (the one the
    generator and a live agent both play) and replay every one of its
    transitions through a second, UNPATCHED adapter loaded straight from the
    .txt, comparing the resulting board and the win flag. Both adapters parse
    the same object table, so a state key from one is a state key in the other.

    Expected: every transition agrees except the two on level 5 whose slide is
    longer than the interpreter's 50-iteration ``again`` budget, where the
    original truncates mid-slide (player short of the wall, a live marker still
    on the board) and the patched engine completes it. Those two are printed in
    full so the exception stays a measured fact rather than a claim, and the
    report FAILS if a difference shows up anywhere else -- including any
    difference at all on levels 1-4.
    """
    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    _solver, game, expert = _new()
    raw = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, orig = game._engine, raw._engine
    ids = game._game.obj_name_to_idx
    if raw._game.obj_name_to_idx != ids:
        print("object tables differ -- the two adapters are not comparable")
        return 1
    marker, player = ids["marker"], ids["player"]

    def cells(e, obj):
        return [(r, c) for r in range(e.height) for c in range(e.width)
                if obj in e.grid[r][c]]

    bad = expected = 0
    for level in range(game.n_levels):
        game.set_level(level)
        raw.set_level(level)
        graph = _Graph.build(eng, expert._key, expert.directions,
                             expert.node_cap)
        # Re-walk the enumerated space in the order it was built (BFS dequeue
        # order, so a state's snapshot is always taken before its own turn
        # comes up), stepping BOTH engines from every state.
        snaps = {graph.start: snapshot(eng)}
        checked = differ = skipped = 0
        for k in graph.succ:
            if k not in snaps:
                skipped += 1
                continue
            for d in expert.directions:
                restore(eng, snaps[k])
                restore(orig, snaps[k])     # same object table, same grid
                eng.step(d)
                orig.step(d)
                checked += 1
                if (expert._key(eng) != expert._key(orig)
                        or eng.check_win() != orig.check_win()):
                    differ += 1
                    stray = cells(orig, marker)
                    print(f"  level {level} press {d}: patched player "
                          f"{cells(eng, player)}, original player "
                          f"{cells(orig, player)}, original markers {stray} "
                          f"-- {'ANIMATION TRUNCATED' if stray else 'MISMATCH'}")
                    if stray:
                        expected += 1
                    else:
                        bad += 1
                nk = graph.succ[k].get(d)
                if nk is not None and nk != _WIN and nk not in snaps:
                    snaps[nk] = snapshot(eng)
        if skipped:
            print(f"  level {level}: {skipped} states never re-reached -- "
                  "the walk is incomplete, this report is not a proof")
            bad += skipped
        print(f"level {level}: {checked} transitions over "
              f"{len(graph.succ)} states -- "
              f"{checked - differ} identical, {differ} differ")
        restore(eng, snaps[graph.start])
    print(f"{expected} differences, all of them a truncated animation"
          if not bad else f"RULES FAILED: {bad} unexplained differences")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every cell size the boards use.

    Whole 64x64 frames of uniform boards are compared rather than one cell out of
    a mixed board: `_render_frame` centre-pads a non-square board, so indexing a
    cell by ``cell_px * r`` silently reads the wrong pixels on the wide level
    (the ps:explod lesson). Two uniform boards render identically iff their cells
    do.

    Five compositions, which is all a settled board of this game can hold --
    nothing is ever created, destroyed or transformed, and the marker and the
    sixteen animation parts cannot appear at all once the game-folder patch has
    collapsed the animation. ``player_on_target`` is the WIN frame and the reason
    the Target sprite is redrawn in the .txt: as shipped it rendered as a bare
    player at every size, and on the 8x17 level (cell_px=3) a bare target
    rendered as bare floor. Both clashes are what this report would show if the
    sprite were reverted.
    """
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, _expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    comps: dict = {
        "floor": (),
        "wall": ("wall",),
        "target": ("target",),
        "player": ("player",),
        "player_on_target": ("target", "player"),
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
        note = "OK" if not clashes else f"IDENTICAL {clashes}"
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} "
              f"compositions -- {note}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--bfs" in sys.argv:
        sys.exit(_bfs())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--rules" in sys.argv:
        sys.exit(_rules())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(SlidySolver.main())
