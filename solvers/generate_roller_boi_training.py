"""Generate Phase-1 training data for the PuzzleScript game ps:roller_boi
("Roller Boi", David Upshall and Itsjustkewa).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators, as does the exhaustive-enumeration oracle (`StateGraph` /
`PSEnumExpert`) this game plans with. This file is the game-specific part: the
measured mechanics, the reports that prove every level is winnable and shortest,
and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_roller_boi",
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
The boi ROLLS: he cannot take a step. One press launches him in that direction
and he travels until something stops him, then you press again. Reach the target
-- and, on four of the five levels, delete the door with the key first.

Everything below was MEASURED against the interpreter, not read off the .txt.
The mechanic is one rule, ``[ > Player | NO Obstacle] -> [ | > Player ]``, which
re-fires to a fixpoint inside a single ``eng.step``, so one press is a whole
slide and the branching factor is four:

* **What stops the roll is the ``Obstacle`` group: Target, Wall, Door,
  BlueDoor.** Everything else -- keys, teleport pads -- is on another collision
  layer and the boi rolls straight over it.
* **The target STOPS the roll and he lands ON it.** The slide rule declines to
  move him into an Obstacle, but Target has its own collision layer, so the
  movement phase that follows the rules then puts him on the target square and
  the level ends there. Parked at (4,1) on level 0 with the target at (4,4) and
  open floor out to (4,7), ``right`` finishes at (4,4). The target can never be
  rolled past.
* **Bumping a wall is a pure no-op, not a death.** ``[ > Player | Wall ] ->
  [ Player | Wall ]`` cancels the force outright; the grid compares equal
  afterwards. There is no `restart` and no `again` anywhere in the ruleset, so
  unlike ps:gdd301_game -- the other slide game here -- this one has no way to
  lose and no state it cannot back out of.
* **A key is collected only if the roll STOPS on it.** The key rule runs after
  the slide rule has already carried the boi to the end of his run, so it sees
  only the cell he came to rest on. Parked at (6,1) on level 2 and rolled right
  over the key at (6,3), he finishes at (6,7) with the key and the door both
  still standing. The game's own tip calls this "you can only collect keys or
  teleport if you are stationary".
* **The key deletes the door from anywhere on the board** --
  ``[Player Key] [Door] -> [Player] []`` is a two-bracket rule and the second
  bracket is unconstrained, so stopping on the key at (5,1) of level 3 removes
  the door at (3,1) across the map. Every level here ships exactly one key and
  one door, so the one-key-opens-one-door tie-break that costs ps:gdd301_game
  three levels never arises.
* **The teleport fires on the cell the roll ENDS on, even when a wall is what
  stopped it.** ``[ > player Teleport1 ] [ Teleport2 ]`` needs the boi to still
  carry his force, and he does: the wall-cancel rule is ordered BEFORE the slide
  rule and the interpreter does not re-run it afterwards, so a roll that halts
  against a wall arrives still moving. Level 3's ``down`` from (1,4) rolls to
  (4,4) -- stopped by the wall at (5,4) -- lands on Teleport1 and comes out at
  Teleport2 (2,7). Rolling ACROSS a pad does nothing: ``left`` from (4,6) passes
  over the same pad at (4,4) and continues to (4,1).
* **The win is two conditions, ``no Door`` AND ``all Target on Player``.**
  Standing on level 2's target with the door still up reads as not-won; deleting
  the door with the boi left where he is flips it. So the key is not optional
  scenery on levels 1-4 even where the door is not in the way.
* Teleport3/Teleport4, Key2, BlueDoor and Holes are declared but appear in no
  level, so no reachable state contains one. (Holes are unreachable by design --
  the only rule that would move one is commented out in the source.)
* ACTION5 exists but no rule reads it, so the game is four directions --
  `directions` says so and the enumeration never branches on it.

Why an exhaustive oracle, and not the shared A*
-----------------------------------------------
A press is a whole roll, so the only states are the cells the boi can come to
REST on, crossed with the two bits of progress (key taken, door gone). That
collapses the space to 13-31 states per level, reached in at most 124 engine
steps: every level in the game is enumerated in under a twentieth of a second
and all five together in 0.13s. So this generator does not search at all --
`StateGraph.build` enumerates the whole reachable component with the interpreter
as the authority, and a backward BFS from the winning edges gives the exact
distance-to-win of every state.

That buys three things a heuristic search would not:

* plans that are provably SHORTEST (no weight, no heuristic, no node cap that
  could bite): 6, 12, 10, 12 and 11 presses;
* exact optimal-action SETS for free -- ``dist(succ) == dist - 1`` -- rather than
  inferred ones. Every expert step is labelled;
* a PROOF that a level is winnable-or-not rather than "the search gave up". Here
  all five are winnable, and ``--bfs`` shows every reachable state of every level
  can still win: nothing in this game is irreversible.

Level indices here and in every report are 0-BASED, i.e. one less than the number
in the game's own "Level N" messages.

Because the whole thing costs under a second there is no ``plan_cache_path``:
every `parallelize_generator` shard re-derives the plans, which is cheaper than
the staleness risk a cached entry would carry.

Optimal-action sets
-------------------
``optsets[i]`` is every press ``d`` with ``dist(succ(state_i, d)) == dist(
state_i) - 1``, read straight off the exhaustive distance field, so it is exact
rather than inferred: a set is the complete list of shortest continuations, and a
press missing from it provably costs at least one more. ``--ties`` re-derives
every label independently by taking each candidate press and re-solving from the
state it lands in. Ties are rare here (1 of 51 plan steps) and that is the game:
a roll commits to a whole corridor, so from most rest cells exactly one direction
makes progress.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the plan
replays to a guaranteed WIN. Nothing in this game is irreversible (see ``--bfs``:
every reachable state can still win, on every level), so the RESET is a
convenience rather than a necessity -- but it keeps the recorded arc the same
"flail, reset, then solve" shape as the rest of the ps: family.

CLI
---
    --plans   per-level size, plan length and tie coverage
    --bfs     the exhaustive reachable-state report (the winnability proof)
    --ties    re-derive every optimal-action label by independent re-solve
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

GAME_NAME = "Roller_Boi"

#: The four presses. ACTION5 exists but no rule reads it (see the docstring),
#: and the order here is the plan's tie-break, so it is fixed.
_DIRS = ("up", "down", "left", "right")


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class RollerBoiExpert(PSEnumExpert):
    """`PSEnumExpert` -- the exhaustive reachable-state oracle -- on four
    presses. See that class for why enumeration and not the shared A*, and the
    module docstring for why a roll-until-you-stop game's space is small enough
    to allow it.

    `_key` is inherited (every non-background cell), which is exact and
    canonical ACROSS levels: the key and the door are destroyed as the level
    progresses and the walls are in the key too, so two levels can never serve
    each other's plans and ``scope_by_level`` is unnecessary.
    """

    #: No ACTION button is read by any rule; branching on it would add a quarter
    #: more interpreter steps for edges that are always self-loops.
    directions = list(_DIRS)


class RollerBoiSolver(PSAStarSolver):
    game_id = "puzzlescript_roller_boi"
    game_name = GAME_NAME
    expert_cls = RollerBoiExpert

    #: Record against the GAME FOLDER's adapter. ``games/ps:roller_boi`` is a
    #: plain passthrough today; it is named anyway so a sprite patch or a step
    #: cap added there later cannot silently make this generator tape a game
    #: nobody plays (the ps:count_mover trap).
    game_module_id = "ps:roller_boi"

    #: Runaway guard on the enumeration, not a tuning dial: the largest level in
    #: the game reaches 31 states.
    node_cap = 200_000

    #: Room for the longest plan (12 presses) plus the RESET exploration prefix
    #: and the re-plan after it. Stays under the adapter's own 200-step
    #: per-level budget, which would otherwise flip a level to GAME_OVER
    #: mid-plan.
    max_steps = 120


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = RollerBoiSolver()
    game = solver.make_game(seed)
    return solver, game, RollerBoiExpert(game, node_cap=RollerBoiSolver.node_cap)


def _report() -> int:
    """Per-level board size, inventory, plan length and tie coverage."""
    _solver, game, expert = _new()
    eng = game._engine
    ids = game._game.obj_name_to_idx
    total = tied = steps = 0
    for level in range(game.n_levels):
        game.set_level(level)
        counts = {n: sum(1 for row in eng.grid for cell in row
                         if ids[n] in cell)
                  for n in ("key", "door", "teleport1")}
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        head = (f"level {level}: {eng.height:2d}x{eng.width:2d}, "
                f"{counts['key']} key(s), {counts['door']} door(s), "
                f"{counts['teleport1']} pad(s)")
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
    plans are SHORTEST and that no level is a dead end: the enumeration
    terminates having generated every state the interpreter admits, and (for
    this game) every one of them still has a path to a win, so no press can
    ever strand the boi."""
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
              f"({live} can still win, {wins} touch the target), "
              f"{graph.steps:5d} engine steps, {dt:5.2f}s -- {verdict}")
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


def _audit() -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every cell size the boards use.

    Whole 64x64 frames of uniform boards are compared rather than one cell out of
    a mixed board: `_render_frame` centre-pads a non-square board, so indexing a
    cell by ``cell_px * r`` silently reads the wrong pixels on the wide levels
    (the ps:explod lesson). Two uniform boards render identically iff their cells
    do.

    The compositions are the ones the enumeration actually produces, not a guess
    -- ``--audit`` would otherwise be asserting things about boards this game
    cannot reach. Two of the ten are worth naming:

    * ``player_on_target`` is the WIN frame, and it survives the Player sprite
      being drawn over it: the Player's 5x5 leaves eight transparent pixels
      (both top corners, both mid-left/right edges, the gaps between the feet)
      through which the Target's white-and-black bars still read.
    * ``player_on_teleport2`` is where a teleport leaves him -- he arrives on the
      exit pad and stays there until the next press -- so it is a board an agent
      sees on levels 3 and 4 and has to tell apart from a bare pad.

    ``player_on_key`` and ``player_on_teleport1`` are absent for reasons that are
    invariants rather than luck: stopping on the key CONSUMES it in the same
    tick (every level has a door left to spend it on), and stopping on the
    entrance pad teleports him off it in the same tick. Neither can be seen in a
    settled frame, so neither is audited.
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
        "key": ("key",),
        "door": ("door",),
        "teleport_in": ("teleport1",),
        "teleport_out": ("teleport2",),
        "player": ("player",),
        "player_on_target": ("target", "player"),
        "player_on_teleport_out": ("teleport2", "player"),
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
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(RollerBoiSolver.main())
