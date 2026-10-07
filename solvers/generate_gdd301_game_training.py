"""Generate Phase-1 training data for the PuzzleScript game ps:gdd301_game
("GDD301 Game", Mikey Bikey).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the exhaustive
reachable-state oracle, its exact optimal-action sets, the reports that prove
which levels are winnable, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_gdd301_game",
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
augmented view, so replaying the recorded actions reproduces the recorded
frames exactly. Every expert step carries the full set of equally-optimal
presses (see "Optimal-action sets" below).

The game
--------
A "dirty guy" slides on ice and leaves a trail he must not touch, from
"Rooky Moves"'s cursor idiom (the author says so in a comment). One press is a
whole SLIDE, not a step: the `again` loop inside a single `eng.step` runs the
run to completion, so the branching factor is four and a plan is a handful of
presses.

Everything below was measured against the interpreter, not read off the .txt --
the trail rules in particular do not do what they look like they do:

* **A slide leaves a trail on every cell it DEPARTS, and none on the cell it
  stops at.** ``[ > Player ] -> [ > Player Trail ]`` paints the player's
  *current* cell while it still holds the force, i.e. before it moves off.
* **Touching a trail is death.** ``late [ Player Trail ] -> restart`` reloads
  the level, so the trail is a self-avoiding-walk constraint, mid-slide as well
  as at the end -- sliding *across* an old trail restarts just as surely as
  stopping on one.
* **Pressing into a wall is death, and this is the whole difficulty of the
  game.** The stopping rule is not "you cannot move": the run stops because the
  Cursor that carries it is eaten one cell early by ``[ Cursor Wall ] ->
  [ Wall ]``, and the player then spends a forceless `again` tick standing
  still. But on the FIRST tick the force comes from the keypress, not from a
  cursor, so a press whose adjacent cell is a wall still paints a trail under
  the player and then fails to move -- player on trail, restart. There is no
  "bump into the wall and nothing happens" move in this game.
* **A closed Door1 is worse than a wall.** ``[ Cursor Wall ]`` names Wall only,
  so the cursor survives on a door and keeps the run going; the player then
  moves into the door, is blocked by the shared collision layer, and lands on
  the trail it just painted. Sliding at a closed door is always death.
* **One key opens exactly ONE door.** ``[Player Key1] [Door1] -> [Player] []``
  is a multi-bracket rule, and both this interpreter and the reference engine
  re-validate every tuple after the first: the Key1 is gone by then, so every
  later (key, door) tuple is dropped. Levels with two doors need two keys, and
  a key is only consumed while a door remains to spend it on -- levels 7 and 8
  ship a spare (2 keys / 1 door and 5 keys / 4 doors), so an inert key really
  can end up sitting under a trail or under the player, which is why the render
  audit checks both compositions.
* **A broom (RemoveTrail) sweeps ONE trail, for the same reason.** Sliding over
  one erases the trail the same tick painted underneath the player, so the
  broom's real effect is to leave one CROSSABLE cell behind on a line you have
  already used. Stopping on one erases a single arbitrary trail instead. The
  message "Brooms sweep away dirty trails!" oversells it.
* **The win is checked between `again` ticks**, so the run STOPS the moment the
  player reaches the Target rather than sliding past it -- the target is a
  stopper, and the board it stops on is the one recorded as the win frame
  (spent Cursor and all; see `_audit`).
* ACTION5 exists but no rule reads it, so the game is four directions --
  `directions` says so and the search never branches on it.

Why an exhaustive oracle, and not the shared A*
-----------------------------------------------
Measured first, per the ps:entrepotphage_demake lesson: the interpreter runs at
232-1034 ``eng.step``/s here (a press is a whole slide, so each one is up to
fifteen rule ticks). That would be far too slow for a real search -- but there
is no real search to do. Death by trail prunes almost every press, so the
*entire* reachable state space of a level is 5 to 46 states, reached in at most
184 engine steps; the worst level in the game is exhausted in a quarter of a
second and all ten together in 1.1s. So this generator does not search at all:
the shared `StateGraph` (imported here as `_Graph`) enumerates the whole
reachable component with the interpreter as the authority, and a backward BFS
from the winning edges gives the exact distance-to-win of every state.

That buys three things a heuristic search would not:

* plans that are provably SHORTEST (no weight, no heuristic, no node cap that
  could bite): 1, 4, 5, 16, 10, 10 and 5 presses on the seven levels that have
  one;
* exact optimal-action SETS for free -- ``dist(succ) == dist - 1`` -- rather
  than inferred ones. Every expert step is labelled; two of the 51 have a
  genuine tie, which is what this game is like (most states have exactly one
  press that is not suicide);
* a PROOF of unwinnability for the levels that have none, instead of "the
  search gave up": levels 3, 8 and 9 exhaust their reachable space (12, 24 and
  21 states) without a winning edge.

Level indices here and in every report are 0-BASED, i.e. one less than the
number in the game's own "Level N" messages.

Because the whole thing costs under a second there is no ``plan_cache_path``:
every `parallelize_generator` shard re-derives the plans, which is cheaper than
the staleness risk a cached negative entry would carry.

Unwinnable levels
-----------------
Levels 3, 8 and 9 cannot be won and are dropped by `discover_solvable`; the
seed's episode holds the other seven. ``--bfs`` prints the proof. All three
die on the same hinge: which door a key spends itself on. Level 3 has one key
and two doors, and the door the interpreter opens -- (2,9) -- is the one that
does not lead anywhere, while opening (3,8) instead would put the target at
the end of an unobstructed row 3 and solve the level in seven presses.

Which door that is depends on the order the interpreter enumerates the second
bracket's matches, and that order is currently Python `set` iteration order:
`PSEngine._apply_single_rule_forces` sorts its candidate cells (with a comment
saying it does so "to preserve scan-order-dependent semantics") and
`_apply_multi_group_rule_forces` does not. Giving the multi-bracket path the
same direction-aware sort makes levels 3 and 9 winnable (7 and 14 presses) and
level 8 unwinnable -- 9/10 rather than 7/10; a plain row-major sort gives a
different 8/10, and column-major gives 10/10. That is an interpreter change
with a blast radius over every multi-bracket ps: game, so it is NOT made here:
this generator records the interpreter as it stands, and if the ordering is
ever fixed `discover_solvable` picks the extra levels up on its own with no
edit to this file.

Optimal-action sets
-------------------
``optsets[i]`` is every press ``d`` with ``dist(succ(state_i, d)) == dist(
state_i) - 1``, read straight off the exhaustive distance field, so it is
exact rather than inferred: a set is the complete list of shortest
continuations, and a press missing from it provably costs at least one more.
``--ties`` re-derives every label independently by taking each candidate press
and re-solving from the state it lands in.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed
exploration prefix flails first and ONE RESET restores the level start, from
which the plan replays to a guaranteed WIN. It fits this game unusually well:
restart *is* the game's own failure mode, so an agent that has just walked into
its own trail has been put back at the level start by the interpreter itself --
which is exactly the state the plan was solved from.

CLI
---
    --plans   per-level size, plan length and tie coverage
    --bfs     the exhaustive reachable-state report (the unwinnability proof)
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

GAME_NAME = "GDD301_Game"

#: The four presses. ACTION5 exists but no rule reads it (see the docstring),
#: and the order here is the plan's tie-break, so it is fixed.
_DIRS = ("up", "down", "left", "right")


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class GDD301Expert(PSEnumExpert):
    """`PSEnumExpert` -- the exhaustive reachable-state oracle -- on four
    presses. See that class for why enumeration and not the shared A*, and the
    module docstring for why this game's space is small enough to allow it.

    `_key` is inherited (every non-background cell), which is exact and
    canonical ACROSS levels: this game creates and destroys trails, keys, doors
    and brooms, and the walls are in the key too, so two levels can never serve
    each other's plans and ``scope_by_level`` is unnecessary.
    """

    #: No ACTION button is read by any rule; branching on it would double the
    #: interpreter steps for nothing.
    directions = list(_DIRS)


class GDD301Solver(PSAStarSolver):
    game_id = "puzzlescript_gdd301_game"
    game_name = GAME_NAME
    expert_cls = GDD301Expert

    #: Record against the GAME FOLDER's adapter. ``games/ps:gdd301_game`` is a
    #: plain passthrough today; it is named anyway so a sprite patch or a
    #: step cap added there later cannot silently make this generator tape a
    #: game nobody plays (the ps:count_mover trap).
    game_module_id = "ps:gdd301_game"

    #: Runaway guard on the enumeration, not a tuning dial: the largest level
    #: in the game reaches 46 states.
    node_cap = 200_000

    #: Room for the longest plan (16 presses) plus the RESET exploration prefix
    #: and the re-plan after it. Stays under the adapter's own 200-step
    #: per-level budget, which would otherwise flip a level to GAME_OVER
    #: mid-plan.
    max_steps = 120


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = GDD301Solver()
    game = solver.make_game(seed)
    return solver, game, GDD301Expert(game, node_cap=GDD301Solver.node_cap)


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
                  for n in ("key1", "door1", "removetrail")}
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        head = (f"level {level}: {eng.height:2d}x{eng.width:2d}, "
                f"{counts['key1']} key(s), {counts['door1']} door(s), "
                f"{counts['removetrail']} broom(s)")
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
    """The exhaustive reachable-state report -- and therefore the proof, for
    the levels that have no plan, that they are UNWINNABLE rather than merely
    unsolved: the enumeration terminates having generated every state the
    interpreter admits, and none of them has a winning edge."""
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
    checks the distance field and the plan-following against a fresh
    enumeration from every state on the path rather than against itself."""
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
                eng._rule_restart = False
                eng.step(d)
                if eng._rule_restart:
                    continue
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
            # advance one press along the plan
            eng._rule_restart = False
            eng.step(taken)
        print(f"level {level}: {checked} steps verified")
    print("ties clean" if not bad else f"TIES FAILED: {bad} mismatches")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every cell size the boards use.

    Whole 64x64 frames of uniform boards are compared rather than one cell out
    of a mixed board: `_render_frame` centre-pads a non-square board, so
    indexing a cell by ``cell_px * r`` silently reads the wrong pixels on the
    wide levels (the ps:explod lesson). Two uniform boards render identically
    iff their cells do.

    THREE collisions are expected and are listed as such; each hides something
    that provably cannot affect play. Anything else is a real bug and fails the
    audit.

    * ``player`` / ``player_on_target``: the Player sprite is opaque over the
      whole 3x3 the Target draws, so a player standing on the target renders
      exactly like a player standing on floor. Accepted rather than patched
      because it costs no information -- there is one target per level, it is
      plainly visible until the moment it is reached, and the win frame is the
      last frame of the level, so the target's disappearance under the player
      IS the terminal signal.
    * ``floor`` / ``cursor``: the Cursor's sprite is entirely transparent by
      construction -- it is the invisible token that carries a slide. It is
      audited (rather than left out) because it does survive onto the board:
      the `again` loop breaks on the win, so a spent cursor sits on two of the
      seven winning boards, and this asserts it stays invisible there.
    * ``player`` / ``player_on_key``, and only at 4px: the Key1 sprite peeks
      out from under the Player by a single pixel, which the downscale to a
      4px cell loses. The one board it happens on is level 7. It hides nothing
      that matters, and the reason is an invariant rather than luck: stepping
      on a key CONSUMES it whenever a door is left to spend it on, so a key can
      only ever be seen under the player once no door remains -- at which point
      it is inert scenery for the rest of the level and every future frame is
      identical with or without it. (``trail_on_key``, the same invariant seen
      from the other side, stays distinct at every size.)
    """
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, _expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    # Exactly the compositions that occur, enumerated over every reachable
    # state of every level plus the seven winning boards -- not a guess. The
    # two that look like guesses are the ones worth naming: a key is only spent
    # while a door remains (``[Player Key1] [Door1]`` needs both brackets), so
    # a level with a spare key really does leave one sitting under a trail, and
    # really does let the player stop on top of one.
    comps: dict = {
        "floor": (),
        "wall": ("wall",),
        "target": ("target",),
        "key": ("key1",),
        "door": ("door1",),
        "broom": ("removetrail",),
        "trail": ("trail",),
        "trail_on_key": ("key1", "trail"),
        "player": ("player",),
        "player_on_key": ("key1", "player"),
        "player_on_target": ("target", "player"),
        "cursor": ("cursor",),
    }
    # Accepted-collision CLASSES, not pairs: at 4px the player hides both the
    # target and the key it stands on, which also makes those two collide with
    # each other -- a consequence of the two facts above, not a third one.
    expected_classes = [
        {"player", "player_on_key", "player_on_target"},
        {"floor", "cursor"},
    ]

    def _expected(a: str, b: str) -> bool:
        return any(a in cls and b in cls for cls in expected_classes)

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
        unexpected = [p for p in clashes if not _expected(*p)]
        bad += len(unexpected)
        note = ("OK" if not unexpected
                else "IDENTICAL (UNEXPECTED) " + str(unexpected))
        seen = sorted(p for p in clashes if _expected(*p))
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} "
              f"compositions -- {note}"
              f"; {len(seen)} expected collision(s) {seen}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} unexpected indistinguishable pairs")
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
    sys.exit(GDD301Solver.main())
