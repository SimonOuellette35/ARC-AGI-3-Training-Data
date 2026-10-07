"""Generate Phase-1 training data for the PuzzleScript game ps:slippy_penguin
("Slippy Penguin", Ryan Woods).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators, as does the exhaustive-enumeration oracle (`StateGraph` /
`PSEnumExpert`) this game plans with. This file is the game-specific part: the
measured mechanics, the reports that prove every level is winnable and shortest,
and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_slippy_penguin",
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
Slippy is a penguin on ice: he cannot take a step. One press launches him and he
slides until a rock stops him. Slide onto the water (the Goal) to win. On the
last three levels the fish he has been eating CHASE him, and a fish is a wall
that walks.

Everything below was MEASURED against the interpreter, not read off the .txt.

**One press is a whole SLIDE.** The movement rules are
``[LEFT Player] -> [PlayerL]`` / ``left [PlayerL | Obstacle] -> [Player_Still |
Obstacle]`` / ``[PlayerL] -> [LEFT PlayerL] again`` (and the three rotations),
so the ``again`` loop runs the run to completion inside a single ``eng.step``.
The branching factor is four and plans are 1-15 presses.

* **The GOAL stops the slide, mid-run, by winning.** The win is checked between
  ``again`` ticks, and Goal1 is on its own collision layer, so Slippy is never
  blocked by it -- he simply wins on the tick he first covers it and the run
  ends there. Level 6's ``up`` from (7,2) wins at the goal (2,2) even though
  (1,2) is open floor beyond it. So a goal can never be slid past, and no plan
  ever has to arrange a rock behind it.
* **What stops a slide is ``Obstacle``, and ``Obstacle`` is Rock alone.** The
  board's outer ring is rock on every level, so a run always terminates.
* **Pressing into a rock you are already touching is a pure no-op** -- the
  cancel rule fires on the first tick and the grid compares equal. There is no
  ``restart`` and no death rule anywhere in the ruleset, and `_rule_restart` is
  never set: nothing in this game is lethal.
* **A FISH also stops the slide -- but by collision, not by the cancel rule.**
  Player, Rock and all three Fish share one collision layer, so a fish blocks
  the move; but the cancel rule names ``Obstacle`` (= Rock), so it does not
  fire, and Slippy is left holding the DIRECTIONAL sprite (PlayerL/R/U/D)
  instead of reverting to Player_Still. That is a visible, distinct board state
  and the enumeration treats it as one: pressing right from (1,1) into a fish at
  (1,4) ends with PlayerR at (1,2) and the fish drawn in at (1,3).
* **Pressing into an adjacent fish is a one-tick sprite flip, not a no-op.**
  Slippy turns to face the fish and stays put (the fish, wanting the cell he is
  standing on, is blocked in turn), so the first such press IS an edge in the
  graph and the second is the no-op.

The fish
--------
``[fish |...| Player] -> [ > Fish |...| Player]`` (plus the same rule for Fish2
behind Fish, and Fish3 behind Fish2). Four measured consequences:

* **A fish steps one cell toward its target whenever the two share a ROW or a
  COLUMN**, and it does so on every ``again`` tick, i.e. once per cell of
  Slippy's slide. Directly behind him in the corridor he is sliding down, a fish
  keeps up exactly and arrives one cell behind him; off that line it stalls
  until he lines up again.
* **The line of sight ignores rocks.** ``...`` matches any cells at all, so a
  fish two rooms away still tracks him through a wall.
* **Movement does not.** The same collision layer that lets a fish stop Slippy
  stops the fish: a rock in the cell it wants leaves it standing.
* **Fish2 chases Fish and Fish3 chases Fish2**, so the shoal strings out into a
  rook-stepping tail rather than a rigid snake.

Fish are never eaten, never removed, and win nothing -- ``All Player on Goal1``
names only Slippy. A fish CAN stand on the goal (Goal is its own collision
layer; measured), though no reachable state of any shipped level does.

**The fish are what makes this game irreversible.** Levels 0-3 have no fish and
every reachable state of them can still win. Levels 4-6 do, and they have dead
ends: 21 of level 4's 94 states, 22 of level 5's 148, and 20 of level 6's 22 --
wall yourself in with your own shoal and no press gets you out. ``--bfs`` prints
the split. This is why ``recovery_mode = "reset"`` is load-bearing here rather
than decorative, and why `record_level`'s epsilon detour (which re-plans before
committing to an alternative) must stay the only source of off-plan presses.

Other objects: Goal2 and the ``noaction`` / ``noundo`` prelude flags. Goal2 is
declared but appears in no level, and ACTION5 is read by no rule -- measured a
no-op on every level's start state -- so ``directions`` is the four moves and
the enumeration never branches on a fifth.

Why an exhaustive oracle, and not the shared A*
-----------------------------------------------
A press is a whole slide, so the states are only the boards Slippy can come to
REST on, crossed with wherever the shoal has strung itself out. That collapses
the space to 3-148 states per level, reached in at most 592 interpreter steps:
the whole game enumerates in under two seconds. So this generator does not
search -- `StateGraph.build` enumerates the entire reachable component with the
interpreter as the authority, and a backward BFS from the winning edges gives
the exact distance-to-win of every state.

That buys three things a heuristic search would not:

* plans that are provably SHORTEST (no weight, no heuristic, no node cap that
  could quietly bite): 3, 7, 13, 15, 7, 5 and 1 presses;
* exact optimal-action SETS for free -- ``dist(succ) == dist - 1`` -- rather than
  inferred ones. Every expert step is labelled;
* a PROOF that a level is winnable rather than "the search gave up", and -- on a
  game that HAS dead ends -- an exact count of the states from which it is not.
  All seven levels are winnable from their start.

Level indices here and in every report are 0-BASED, i.e. one less than the
number in the game's own "Level N" messages (the ``message`` screens between
levels are consumed by the adapter and are not levels).

Because the whole thing costs under two seconds there is no ``plan_cache_path``:
every `parallelize_generator` shard re-derives the plans, which is cheaper than
the staleness risk a cached entry would carry.

Optimal-action sets
-------------------
``optsets[i]`` is every press ``d`` with ``dist(succ(state_i, d)) == dist(
state_i) - 1``, read straight off the exhaustive distance field, so it is exact
rather than inferred: a set is the complete list of shortest continuations, and
a press missing from it provably costs at least one more. ``--ties``
re-derives every label independently by taking each candidate press and
re-solving from the state it lands in.

There are ZERO ties in this game's 51 plan presses, and that is the mechanic
rather than an accident: a slide commits to a whole corridor, so from a given
rest cell the four presses land in four different places and at most one of them
is on a shortest path. The sets are still emitted (they are what
`PSAStarSolver.optimal_for` reads) and ``--ties`` still checks them, because
"no tie" is a measurement, not an assumption.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the plan
replays to a guaranteed WIN. Unlike ps:roller_boi -- the other slide game here
-- the RESET is doing real work on levels 4-6: the flail can strand Slippy
behind his own fish in a state with no path to the water (see ``--bfs``), and
only the reset gets him out.

`SlippyPenguinSolver._reset_prefix` widens that RESET by one case the shared
harness skips: a prefix that WINS. Level 0 is three presses on a two-corridor
board and the exploration budget is at its largest on the episode's first level,
so a random flail finishes it about a quarter of the time -- and the shared
prefix, which resets only when it did NOT win, then leaves a level made
entirely of ``phase="explore"`` steps, i.e. a WIN that `train_policy` masks out
of the policy loss completely. Measured at 10 of 280 levels over 40 seeds before
the override and 0 of 280 after; expert steps went 2010 -> 2040. See the method
for why the fix lives here rather than in `BaseSolver`.

CLI
---
    --plans   per-level size, plan length and tie coverage
    --bfs     the exhaustive reachable-state report (winnability + dead ends)
    --ties    re-derive every optimal-action label by independent re-solve
    --audit   assert every cell composition renders distinctly, at every size
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                          # noqa: E402

from arcengine import GameState                             # noqa: E402
from solvers.common.ps_astar import (                       # noqa: E402
    WIN as _WIN, PSAStarSolver, PSEnumExpert, StateGraph as _Graph,
    restore, snapshot,
)
from utils.explore import Action, RESET_ACTION              # noqa: E402

GAME_NAME = "Slippy_Penguin"

#: The four presses. ACTION5 exists as a button but the game's prelude says
#: ``noaction`` and no rule reads it (measured a no-op on every level start), and
#: the order here is the plan's tie-break, so it is fixed.
_DIRS = ("up", "down", "left", "right")


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class SlippyPenguinExpert(PSEnumExpert):
    """`PSEnumExpert` -- the exhaustive reachable-state oracle -- on four
    presses. See that class for why enumeration and not the shared A*, and the
    module docstring for why a slide game's space is small enough to allow it.

    `_key` is inherited (every non-background cell), which is what this game
    needs on both counts: it separates the four DIRECTIONAL player sprites from
    Player_Still -- distinct boards an agent can see, and reachable here because
    a fish stops a slide without firing the cancel rule -- and it keeps the rocks
    in the key, so the key is canonical ACROSS levels and ``scope_by_level`` is
    unnecessary.
    """

    #: No ACTION button is read by any rule; branching on it would add a quarter
    #: more interpreter steps for edges that are always self-loops.
    directions = list(_DIRS)


class SlippyPenguinSolver(PSAStarSolver):
    game_id = "puzzlescript_slippy_penguin"
    game_name = GAME_NAME
    expert_cls = SlippyPenguinExpert

    #: Record against the GAME FOLDER's adapter. ``games/ps:slippy_penguin`` is a
    #: plain passthrough today; it is named anyway so a sprite patch or a step
    #: cap added there later cannot silently make this generator tape a game
    #: nobody plays (the ps:count_mover trap).
    game_module_id = "ps:slippy_penguin"

    #: Runaway guard on the enumeration, not a tuning dial: the largest level in
    #: the game reaches 148 states.
    node_cap = 200_000

    #: Room for the longest plan (15 presses) plus the RESET exploration prefix
    #: and the re-plan after it. Stays under the adapter's own 200-step
    #: per-level budget, which would otherwise flip a level to GAME_OVER
    #: mid-plan.
    max_steps = 120

    def _reset_prefix(self, *, reset_to_level, record_obs, record_act, **kw):
        """`BaseSolver._reset_prefix`, plus a RESET when the prefix WON.

        The shared prefix deliberately skips its closing RESET on a terminal win
        (``if perturbed and last_terminal != "win"``) -- for a game whose levels
        take a dozen presses that branch is unreachable, so no other ps:
        generator has to think about it. Level 0 here is THREE presses on a
        two-corridor board, and the opening exploration budget is at its largest
        on the first level of the episode, so a random flail wins it outright
        roughly a quarter of the time. `record_level`'s loop then sees an
        already-won level and breaks, and the level tapes a WIN made entirely of
        ``phase="explore"`` steps -- which `train_policy` masks out of the policy
        loss, so those recordings train the policy on NOTHING (measured: 10 of
        280 levels over 40 seeds, all of them level 0).

        Resetting anyway costs one frame and puts the level back at its start,
        where the expert replays the real 3-press solution. Every level then
        carries a full labelled demonstration and every episode has the same
        flail / reset / solve arc.

        The single-frame assumption is safe here: ``record_spans`` is False, so
        `record_level`'s ``reset_to_level`` hands back one ``(H, W)`` frame.
        """
        prev = super()._reset_prefix(reset_to_level=reset_to_level,
                                     record_obs=record_obs,
                                     record_act=record_act, **kw)
        if self._game is None or self._game._state != GameState.WIN:
            return prev
        frame = np.asarray(reset_to_level())
        record_obs(frame)
        record_act(self._encode_step(
            Action(RESET_ACTION), optimal=[Action(RESET_ACTION)],
            phase="reset", changed=not np.array_equal(frame, prev), n_obs=1))
        return frame


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = SlippyPenguinSolver()
    game = solver.make_game(seed)
    return solver, game, SlippyPenguinExpert(
        game, node_cap=SlippyPenguinSolver.node_cap)


def _report() -> int:
    """Per-level board size, shoal size, plan length and tie coverage."""
    _solver, game, expert = _new()
    eng = game._engine
    ids = game._game.obj_name_to_idx
    total = tied = steps = 0
    for level in range(game.n_levels):
        game.set_level(level)
        fish = sum(1 for row in eng.grid for cell in row
                   for n in ("fish", "fish2", "fish3") if ids[n] in cell)
        probe = snapshot(eng)
        before = expert._key(eng)
        eng.step("action")
        act_noop = expert._key(eng) == before
        restore(eng, probe)
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        head = (f"level {level}: {eng.height:2d}x{eng.width:2d}, "
                f"{fish} fish, ACTION5 {'no-op' if act_noop else 'ACTIVE'}")
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
    plans are SHORTEST and that every level is winnable from its start: the
    enumeration terminates having generated every state the interpreter admits.

    It also counts the DEAD ENDS, which this game has and ps:roller_boi does
    not: a state from which no press reaches the water. They appear only on the
    fish levels (4-6) and they are the reason the recorded arc's RESET matters
    -- the exploration prefix can walk into one."""
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
              f"({live} can still win, {len(graph.succ) - live} dead ends, "
              f"{wins} touch the water), {graph.steps:5d} engine steps, "
              f"{dt:5.2f}s -- {verdict}")
        if found is None:
            bad += 1
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


#: Compositions the game DRAWS identically and that nothing needs to tell apart.
#: Fish, Fish2 and Fish3 are three copies of one sprite with one palette in the
#: source -- they are all just fish, and the only thing their identity decides is
#: who chases whom, which the shoal's geometry already shows. `_audit` FAILS if
#: an accepted pair does NOT actually render identically, so the accept cannot go
#: stale and quietly stop auditing a composition an agent can now tell apart.
_ACCEPTED = [frozenset({"fish", "fish2", "fish3"})]


def _audit() -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every cell size the boards use.

    Whole 64x64 frames of uniform boards are compared rather than one cell out of
    a mixed board: `_render_frame` centre-pads a non-square board, so indexing a
    cell by ``cell_px * r`` silently reads the wrong pixels on the wide levels
    (the ps:explod lesson). Two uniform boards render identically iff their cells
    do.

    The compositions are the ones the enumeration actually produces, not a guess.
    They were collected by walking every reachable state of every level (the same
    walk ``--bfs`` proves complete), and they are more than a slide game's usual
    four:

    * **all four DIRECTIONAL player sprites are settled boards**, not just
      animation. A slide stopped by a FISH never fires the cancel rule that
      restores Player_Still, so Slippy sits facing the fish until the next press
      -- on levels 4-6 he does it often.
    * **the win frame is a directional sprite ON the goal.** The win is checked
      inside the ``again`` loop, on the tick he covers the water and while his
      force is still held, so Player_Still is never what wins. Measured across
      every winning edge of every level, exactly three of the four occur --
      ``playerl`` (levels 1, 2, 4, 5), ``playerr`` (0, 3) and ``playeru``
      (2, 6). ``playerd`` is absent because no level's water is ever entered
      from above.

    A fish on the goal is legal (Goal is its own collision layer -- measured on a
    hand-built board) but no reachable state of any shipped level produces one,
    so it is not audited.
    """
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, _expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    comps: dict = {
        "floor": (),
        "rock": ("rock",),
        "goal": ("goal1",),
        "player": ("player_still",),
        "player_left": ("playerl",),
        "player_right": ("playerr",),
        "player_up": ("playeru",),
        "player_down": ("playerd",),
        "fish": ("fish",),
        "fish2": ("fish2",),
        "fish3": ("fish3",),
        "win_left": ("goal1", "playerl"),
        "win_right": ("goal1", "playerr"),
        "win_up": ("goal1", "playeru"),
    }
    accepted = {frozenset({a, b})
                for group in _ACCEPTED
                for a, b in itertools.combinations(sorted(group), 2)}

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    accepted_seen = 0
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
        ok = [p for p in clashes if frozenset(p) in accepted]
        hard = [p for p in clashes if frozenset(p) not in accepted]
        # A STALE accept is a failure too: if a fish sprite were ever made
        # distinct, silently keeping it out of the audit would stop checking a
        # composition an agent can now see.
        stale = sorted(tuple(sorted(pair)) for pair in accepted
                       if frozenset(pair) not in {frozenset(c) for c in clashes})
        accepted_seen += len(ok)
        bad += len(hard) + len(stale)
        note = ("OK" if not hard else f"IDENTICAL {hard}")
        if ok:
            note += f" (+{len(ok)} accepted: {ok})"
        if stale:
            note += f" -- STALE ACCEPT (these render differently): {stale}"
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
    sys.exit(SlippyPenguinSolver.main())
