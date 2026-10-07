"""Generate Phase-1 training data for the PuzzleScript game ps:tumblin
("Tumblin'", Westward).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators, as does the exhaustive-enumeration oracle (`StateGraph` /
`PSEnumExpert`) this game plans with. This file is the game-specific part: the
measured mechanics, the reports that prove every level is winnable and shortest,
and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_tumblin",
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
action (post rotation remap), i.e. the button an agent presses in the augmented
view, so replaying the recorded actions reproduces the recorded frames exactly.
Every expert step carries the full set of equally-optimal presses (see
"Optimal-action sets" below).

The game
--------
"The wind blows all the tumbleweeds until they hit something." One arrow key
blows EVERY tumbleweed on the board in that direction at once, and each one
travels until a wall or a stopped weed is in its way. Park a weed on every blue
target and then press X.

Everything below was MEASURED against the interpreter, not read off the .txt.

* **One press moves every ball, and only walls and STOPPED balls stop them.**
  A press turns each `Player` into a `RightPlayer`/`LeftPlayer`/... which is
  given a force, moved one cell, and re-fires through the `late [RightPlayer]
  -> [RightPlayer] again` chain until `[ RightPlayer | Wall ]` or
  `[ RightPlayer | Player ]` converts it back. So one keypress is a whole roll,
  resolved inside a single `eng.step`.
* **A convoy keeps its formation; a blocked convoy PACKS.** Three weeds abreast
  with open floor ahead all move together (`#PPP.....#` -> `#.....PPP#`,
  spacing preserved), because the stop rules only look for a `Player` that has
  already settled. But the moment the leader converts, the follower's rule
  matches and re-fires down the line inside the same tick, so a run that ends at
  a wall arrives packed solid: `#P.PP..P#` -> `#...PPPP#` in ONE press. Gaps
  are destroyed by a press that ends at a wall and preserved by one that does
  not, which is the whole game.
* **A target does NOT stop a roll.** `Target` is on its own collision layer and
  no rule reads it, so a weed rolls straight over one (`#P..O..#` -> `#...O.P#`)
  and covering a target means arranging for a weed's run to END there. Every
  covered target in this game is a weed stopped by a wall or by another weed
  that happens to sit on it.
* **The win needs a separate X press.** `All Target on Player` is only one of
  six win conditions; the last is `Some ActionFlag2`, and `ActionFlag2` exists
  only on the tick after `[Action Player] -> [ActionFlag Player]` fired. So
  arranging the weeds is NOT a win -- the board sits there covered until X is
  pressed, and every plan in this game ends in `action`. Two consequences worth
  stating because they shape what an agent has to learn:
  - `ActionFlag` / `ActionFlag2` are `transparent` objects, so a non-winning X
    press produces a **pixel-identical frame** (verified). X is the one press
    whose effect is invisible until it is the right one.
  - The flag is deleted again by `[ActionFlag2] -> []` at the top of the next
    turn, so it carries NO information forward: pressing X is idempotent and
    can never be "banked" for a later turn. That is why `_key` drops both flag
    objects (see `TumblinExpert._key`).
* **A press with nothing to move is an exact no-op** -- the grid compares equal
  afterwards -- and there is no `restart`, no death, and nothing destructible
  anywhere in the ruleset. `--bfs` confirms the consequence: every reachable
  state of every level can still reach a win, so this game has no way to lose.
* **Each level is TWO boards.** The `-` legend is `Empty`, a transparent object
  that fills the gutter column and the corners, and each level draws a walled
  playfield on the left (the one with the targets) and a second walled playfield
  to its right. The two are separated by that gutter, so no weed can cross and
  the right-hand board contributes nothing to the win condition -- it is a
  distractor that rolls in lockstep with the real puzzle. `--boards` PROVES the
  decoupling rather than assuming it (see below); the search keys on every weed
  on the grid anyway, so nothing here depends on the proof.
* ACTION5 is read (it is the win button), so `directions` is all five presses.
  Mouse actions are read by nothing.

Why an exhaustive oracle, and not the shared A*
-----------------------------------------------
A press is a whole roll and a roll that ends at a wall PACKS the weeds against
it, so the configurations reachable from a level start collapse almost
immediately: 25 to 176 states per level, 4 to 8 weeds, the whole game enumerated
in ~2.5s and at most 880 interpreter steps for the biggest level. So this
generator does not search at all -- `StateGraph.build` enumerates the whole
reachable component with the interpreter as the authority, and a backward BFS
from the winning edges gives the exact distance-to-win of every state.

That buys three things a heuristic search would not:

* plans that are provably SHORTEST (no weight, no heuristic, no node cap that
  could bite): 4, 7, 6, 5, 9 and 4 presses;
* exact optimal-action SETS for free -- ``dist(succ) == dist - 1`` -- rather than
  inferred ones. Every expert step is labelled;
* a PROOF that a level is winnable rather than "the search gave up": the
  enumeration terminates having generated every state the interpreter admits.

Level indices here and in every report are 0-BASED. The game's two opening
MESSAGE screens and its two closing ones are not levels and the adapter does not
present them, so level 0 here is the game's first playable board.

Because the whole thing costs a couple of seconds there is no
``plan_cache_path``: every `parallelize_generator` shard re-derives the plans,
which is cheaper than the staleness risk a cached entry would carry.

Optimal-action sets
-------------------
``optsets[i]`` is every press ``d`` with ``dist(succ(state_i, d)) == dist(
state_i) - 1``, read straight off the exhaustive distance field, so it is exact
rather than inferred: a set is the complete list of shortest continuations, and a
press missing from it provably costs at least one more. ``--ties`` re-derives
every label independently by taking each candidate press and re-solving from the
state it lands in. Ties are rare (1 of 35 plan steps) and that is the game: a
press commits every weed on the board at once, so two directions almost never
leave the puzzle equally far along.

Rendering
---------
Six cell COMPOSITIONS occur in a settled frame and ``--audit`` asserts they are
pairwise distinct at every cell size the boards use (3px and 4px):

    void (Empty) = 5 | sand = 11 | wall = 12 | target = 9 over 11
    player = 2 over 11 | player_on_target = 2 over 9

The one that matters is `player_on_target`, the WIN composition: the Player's
5x5 is transparent at exactly its four corners and the Target's X paints exactly
those four corners, so a covered target still reads through the weed sitting on
it -- and those corners survive the renderer's centred sampling at both 3px
(rows/cols {0,2,4}) and 4px ({0,1,3,4}). Without that the frame could not say
whether the board was won.

`void` renders as a uniform 5, which IS the letterbox pad colour -- the
ps:stand_iii clash, and here it is deliberate rather than a bug: `Empty` is
outside both playfields' walls, no rule can put anything on it and `--boards`
confirms no reachable state ever has a weed there. It is out of bounds, and it
renders as out of bounds. `--audit` states the exception explicitly instead of
skipping the check.

One fix was needed in `data/puzzlescript_games/Tumblin'.txt`: all five ball
sprites were painted `lightbrown`, which is **not a name `_COLOR_NAME_TO_ARC`
knows** -- `_color_name_to_arc` silently returns its fallback index 2 for an
unrecognised name, so the game's only movable object was relying on a coin flip
that happened to land on a free index (the ps:box_fill trap). Repainted `gray`,
which IS index 2, so the frames are byte-identical (verified: identical
start-frame hash over all six levels) and nothing is left resting on the
fallback.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the plan
replays to a guaranteed WIN. Nothing in this game is irreversible (see
``--bfs``), so the RESET is a convenience rather than a necessity -- but it keeps
the recorded arc the same "flail, reset, then solve" shape as the rest of the
ps: family.

CLI
---
    --plans   per-level size, inventory, plan length and tie coverage
    --bfs     the exhaustive reachable-state report (the winnability proof)
    --boards  the two-playfield decoupling proof + the settled-state invariants
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

GAME_NAME = "Tumblin'"

#: The five presses. ACTION5 is the win button -- ``All Target on Player`` is
#: not enough on its own, the ruleset also demands ``Some ActionFlag2`` -- so
#: unlike the other roll-until-you-stop games here it cannot be dropped from the
#: branching. The order is the plan's tie-break, so it is fixed.
_DIRS = ("up", "down", "left", "right", "action")

#: Rendered by nothing (both are declared `transparent`) and remembered by
#: nothing: see `TumblinExpert._key`.
_FLAGS = ("actionflag", "actionflag2")

#: The four in-flight ball objects. A settled frame contains none of them --
#: `--boards` asserts it, which is also the check that the `again` chain never
#: runs out of its 50-iteration budget mid-roll.
_ROLLING = ("rightplayer", "leftplayer", "upplayer", "downplayer")


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class TumblinExpert(PSEnumExpert):
    """`PSEnumExpert` -- the exhaustive reachable-state oracle -- on all five
    presses. See that class for why enumeration and not the shared A*, and the
    module docstring for why a blow-every-weed-at-once game's space is small
    enough to allow it.
    """

    directions = list(_DIRS)

    def setup(self) -> None:
        self._skip = {self.g.obj_name_to_idx[n]
                      for n in _FLAGS if n in self.g.obj_name_to_idx}
        self._skip.add(self.bg_id)

    def _key(self, eng) -> frozenset:
        """Every non-background cell EXCEPT the two ActionFlag objects.

        The flags have to go, and dropping them is exact rather than a
        convenience. `ActionFlag2` is created on the tick X is pressed and
        deleted by `[ActionFlag2] -> []` at the top of the very next turn, so
        its presence cannot influence any future press or any future win check:
        two boards that differ only in it are the same puzzle. Keeping it would
        make a non-winning X press look like a state CHANGE, which would double
        the enumeration with a shadow copy of every state and hand the plan a
        pointless press.

        Everything else stays, walls and targets included, which makes the key
        canonical ACROSS levels -- so two levels can never serve each other's
        plans and ``scope_by_level`` is unnecessary.
        """
        skip = self._skip
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in cell
            if o not in skip
        )


class TumblinSolver(PSAStarSolver):
    game_id = "puzzlescript_tumblin"
    game_name = GAME_NAME
    expert_cls = TumblinExpert

    #: Record against the GAME FOLDER's adapter. ``games/ps:tumblin`` is a plain
    #: passthrough today; it is named anyway so a sprite patch or a step cap
    #: added there later cannot silently make this generator tape a game nobody
    #: plays (the ps:count_mover trap).
    game_module_id = "ps:tumblin"

    #: Runaway guard on the enumeration, not a tuning dial: the largest level in
    #: the game reaches 176 states.
    node_cap = 200_000

    #: Room for the longest plan (9 presses) plus the RESET exploration prefix
    #: and the re-plan after it. Stays under the adapter's own 200-step
    #: per-level budget, which would otherwise flip a level to GAME_OVER
    #: mid-plan.
    max_steps = 80


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = TumblinSolver()
    game = solver.make_game(seed)
    return solver, game, TumblinExpert(game, node_cap=TumblinSolver.node_cap)


def _count(eng, ids, names) -> int:
    want = {ids[n] for n in names if n in ids}
    return sum(1 for row in eng.grid for cell in row if cell & want)


def _report() -> int:
    """Per-level board size, inventory, plan length and tie coverage."""
    _solver, game, expert = _new()
    eng = game._engine
    ids = game._game.obj_name_to_idx
    total = tied = 0
    for level in range(game.n_levels):
        game.set_level(level)
        weeds = _count(eng, ids, ("player",))
        targets = _count(eng, ids, ("target",))
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        head = (f"level {level}: {eng.height:2d}x{eng.width:2d}, "
                f"{weeds} weed(s), {targets} target(s)")
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
              f"({ties / max(1, len(found)):3.0%}), {dt:5.2f}s "
              f"-- {' '.join(found)}")
    print(f"total {total} presses, {tied}/{total} steps with a tie "
          f"({tied / max(1, total):.0%})")
    return 0


def _bfs() -> int:
    """The exhaustive reachable-state report -- and therefore the proof that the
    plans are SHORTEST and that no level is a dead end: the enumeration
    terminates having generated every state the interpreter admits, and (for
    this game) every one of them still has a path to a win, so no press can ever
    strand the weeds.

    ``wins`` counts the states from which X ends the level, i.e. the distinct
    board arrangements that cover every target -- the thing the whole puzzle is
    looking for."""
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
        if live != len(graph.succ):
            bad += 1
        verdict = (f"shortest {len(found)} presses" if found is not None
                   else "UNWINNABLE (no winning edge is reachable)")
        print(f"level {level}: {len(graph.succ):5d} reachable states "
              f"({live} can still win, {wins} cover every target), "
              f"{graph.steps:5d} engine steps, {dt:5.2f}s -- {verdict}")
    print("bfs clean" if not bad
          else f"BFS FAILED: {bad} level(s) with a dead end or an unproved cap")
    return 0 if not bad else 1


def _components(eng, ids) -> list:
    """The level's separate PLAYFIELDS: 4-connected components of the cells that
    are not `Empty`.

    `Empty` is the `-` legend, a transparent object filling the gutter column
    and the corners. It is the ONLY thing between the two boards each level
    draws, so grouping by "not Empty" is exactly the author's own division --
    and it is derived from the board rather than from a hard-coded column, so a
    level with one playfield or three is reported as such."""
    empty = ids["empty"]
    seen = [[False] * eng.width for _ in range(eng.height)]
    out = []
    for r0 in range(eng.height):
        for c0 in range(eng.width):
            if seen[r0][c0] or empty in eng.grid[r0][c0]:
                continue
            comp, queue = [], deque([(r0, c0)])
            seen[r0][c0] = True
            while queue:
                r, c = queue.popleft()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < eng.height and 0 <= nc < eng.width
                            and not seen[nr][nc]
                            and empty not in eng.grid[nr][nc]):
                        seen[nr][nc] = True
                        queue.append((nr, nc))
            out.append(frozenset(comp))
    return out


def _boards() -> int:
    """Prove the two playfields are INDEPENDENT, and check the settled-state
    invariants.

    The module docstring claims the right-hand board is a distractor that cannot
    touch the puzzle. That is a claim about the interpreter, so it is measured:
    walk every reachable state, and for each playfield check that the weeds
    inside it after a press are a function of the weeds inside it before the
    press ALONE -- i.e. two states that agree on this board but differ on the
    other take it to the same place. If that holds for every state and every
    press, no information crosses the gutter.

    Nothing in the solver relies on the result (the enumeration keys on every
    weed on the grid), so this is documentation with a proof attached rather
    than an assumption being justified after the fact. Three invariants are
    checked on the same walk, because they need the same state dump:

    * no weed is ever on an `Empty` cell -- the gutter really is unreachable, so
      the `void`-renders-as-the-letterbox-pad note in `--audit` is harmless;
    * no settled state contains a `RightPlayer`/`LeftPlayer`/`UpPlayer`/
      `DownPlayer`, which is the check that the `late ... again` chain always
      finishes a roll inside one `eng.step` and never hits the adapter's
      50-iteration budget mid-flight (the ps:crateblob trap);
    * every target sits on `Sand`, so the win composition the audit renders is
      the one the boards actually show.
    """
    _solver, game, expert = _new()
    eng = game._engine
    ids = game._game.obj_name_to_idx
    player, empty, sand, target = (ids["player"], ids["empty"], ids["sand"],
                                   ids["target"])
    rolling = {ids[n] for n in _ROLLING if n in ids}
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        comps = _components(eng, ids)
        # Walk the whole space, keeping each state's weed set.
        start_snap = snapshot(eng)

        def weeds(e) -> frozenset:
            return frozenset((r, c)
                             for r, row in enumerate(e.grid)
                             for c, cell in enumerate(row) if player in cell)

        snaps = {expert._key(eng): start_snap}
        pos = {}
        queue = deque(snaps)
        trans: list = []          # (state key, direction, before, after)
        while queue:
            k = queue.popleft()
            restore(eng, snaps[k])
            pos[k] = weeds(eng)
            for d in expert.directions:
                restore(eng, snaps[k])
                eng.step(d)
                if rolling and any(cell & rolling
                                   for row in eng.grid for cell in row):
                    print(f"  level {level}: a roll survived the press {d} "
                          f"-- the again chain did not finish")
                    bad += 1
                nk = expert._key(eng)
                if nk == k:
                    continue
                trans.append((k, d, pos[k], weeds(eng)))
                if nk not in snaps:
                    snaps[nk] = snapshot(eng)
                    queue.append(nk)
        restore(eng, start_snap)

        # Invariants over every reachable state.
        on_void = sum(1 for p in pos.values() for cell in p
                      if empty in eng.grid[cell[0]][cell[1]])
        if on_void:
            print(f"  level {level}: {on_void} weed placement(s) on Empty")
            bad += 1
        floorless = sum(1 for r, row in enumerate(eng.grid)
                        for c, cell in enumerate(row)
                        if target in cell and sand not in cell)
        if floorless:
            print(f"  level {level}: {floorless} target(s) not on Sand")
            bad += 1

        # Per-playfield determinism.
        notes = []
        for ci, comp in enumerate(sorted(comps, key=lambda s: min(s))):
            table: dict = {}
            clash = 0
            for _k, d, before, after in trans:
                sub = (frozenset(before & comp), d)
                res = frozenset(after & comp)
                if table.setdefault(sub, res) != res:
                    clash += 1
            n_t = sum(1 for _r, _c in comp if target in eng.grid[_r][_c])
            n_w = len(pos[expert._key(eng)] & comp)
            notes.append(f"#{ci} {len(comp):3d} cells, {n_w} weed(s), "
                         f"{n_t} target(s), {len(table):4d} (config, press) "
                         f"pairs, {clash} clash(es)")
            bad += clash
        print(f"level {level}: {len(pos)} states, {len(comps)} playfield(s)")
        for note in notes:
            print(f"    {note}")
    print("boards clean" if not bad
          else f"BOARDS FAILED: {bad} violation(s)")
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
    a mixed board: `_render_frame` upscales and centre-pads the board, so
    indexing a cell by ``cell_px * r`` silently reads the wrong pixels (the
    ps:explod lesson). Two uniform boards render identically iff their cells do.

    The compositions are the six the levels are built out of (the legend is
    ``- . # P O Q``), and the one to watch is ``player_on_target`` -- the WIN
    board. The Player's 5x5 is transparent at exactly its four corners and the
    Target's X paints exactly those four corners, so a covered target reads
    through the weed standing on it; those corners are sampled at cell_px 3
    (rows/cols {0,2,4}) and at 4 ({0,1,3,4}), which is every size these boards
    use. In palette terms ``player`` is 2-over-11 and ``player_on_target`` is
    2-over-9, so the difference is a colour and not a hairline.

    ``void`` is a DELIBERATE clash with the letterbox pad (both uniform 5) and
    the check below states it rather than skipping it: `Empty` is outside both
    playfields' walls, no rule can place anything on it, and ``--boards``
    confirms no reachable state ever puts a weed there. It is out of bounds and
    it renders as out of bounds -- unlike the ps:stand_iii holes, which were
    lethal terrain wearing the pad's colour.
    """
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, _expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    #: name -> objects stacked in the cell, ON TOP of Background. Every level
    #: cell carries Sand or Empty, so Sand is in each of the playfield ones.
    comps: dict = {
        "void": ("empty",),
        "sand": ("sand",),
        "wall": ("wall", "sand"),
        "target": ("target", "sand"),
        "player": ("player", "sand"),
        "player_on_target": ("target", "player", "sand"),
    }
    #: Pairs allowed to render identically, with the reason. See the docstring.
    exceptions = {("void", "_pad")}

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
        # The ps:stand_iii check: a composition that renders as nothing but the
        # letterbox pad is invisible against the frame's own border.
        pad = [(n, "_pad") for n, s in shots.items()
               if len(set(s.flatten().tolist())) == 1
               and (n, "_pad") not in exceptions]
        bad += len(clashes) + len(pad)
        note = "OK" if not (clashes or pad) else f"IDENTICAL {clashes + pad}"
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} "
              f"compositions -- {note} "
              f"[{len(exceptions)} declared exception(s)]")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--bfs" in sys.argv:
        sys.exit(_bfs())
    if "--boards" in sys.argv:
        sys.exit(_boards())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(TumblinSolver.main())
