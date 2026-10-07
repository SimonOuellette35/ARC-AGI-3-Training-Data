"""Generate Phase-1 training data for the PuzzleScript game
ps:little_girl_big_world ("Little Girl, Big World", Franklin P. Dyer).

The harness -- the rotation contract, the trajectory recorder, the plan memo and
its disk cache, and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: the exact distance-to-win FIELD over the slice of each
level's state space that can carry a shortest solution (built by stepping the
real interpreter, so there is no model to fuzz), the optimal-action oracle that
falls out of it, and the sprite work the shipped file needed.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_little_girl_big_world",
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
exactly. Every expert step also carries the full set of equally-optimal presses.

The game
--------
A girl in a walled garden has to walk out through the one gap in the wall. Four
presses, no ACTION, and the whole thing is five rules:

  * **You push walls.** ``[ > Player | Wall ] -> [ > Player | > Wall ]``, where
    ``Wall = Wall1 or Wall2 or Wall3`` -- the blue blocks and the red blocks
    (Wall3 is declared but appears in no shipped level).
  * **A wall is a RIGID BODY, and its body is its colour.** Inside a
    ``startloop``, ``[ moving Wall1 | Wall1 ] -> [ moving Wall1 | moving Wall1 ]``
    (and the same for Wall2/Wall3) floods the movement across the whole
    4-connected component of blocks OF THE SAME COLOUR, in every direction, so
    shoving one cell of a seven-cell blue snake drags all seven. Two blocks of
    DIFFERENT colours touching are two separate bodies; two blocks of the same
    colour that are shoved into contact become one body FOREVER, because nothing
    ever splits a component. That is the whole puzzle: every push you make
    welds the garden a little more solid.
  * **Bodies push bodies.** Still inside the loop, ``[ > Wall | Wall ] ->
    [ > Wall | > Wall ]`` hands the force on to whatever is in front, so a blue
    body can shove a red one, which floods through the red body, which shoves
    the next blue one. The loop runs the flood and the hand-off to a fixpoint,
    so one press moves an arbitrarily long chain at once.
  * **The garden fence REFUSES, it does not resist.**
    ``[ > Wall | Barrier ] -> cancel`` throws the ENTIRE turn away the moment
    any wall in that chain would enter the fence -- the girl does not move
    either. This is what keeps the rigid bodies whole: there is no state in
    which half a body moved. (The girl walking into the fence herself is not a
    cancel, just a press that does nothing: no rule matches, so her move is
    dropped and everything else stands.)
  * **The gap is the win.** ``[ > Player | Goal ] -> [ | Player ]`` deletes the
    Goal as she steps into it, and the win condition is ``No Goal``. So the
    girl and the goal never coexist in a frame; the goal is on its own collision
    layer, and the one cell of the fence ring that carries it carries no
    Barrier -- which means a WALL can be shoved into the gap and plug it. That
    state is legal, reachable on every level, and recoverable (shove the wall
    back off), so the goal has to stay visible underneath one. See Rendering.

There is no death, no timer and no hidden state: the board IS the state.

The levels
----------
All five shipped levels are winnable and all five are recorded. ``--plans``
prints this table and CERTIFIES every plan twice -- by replaying it through the
bare interpreter and again through the ADAPTER an agent drives:

    level  board   bodies          states  plan  tie steps
    0       7x7    1 blue, 1 red      144    14    2 (14%)
    1       8x8    2 blue, 1 red      172    28    7 (25%)
    2      11x10   2 blue, 1 red     1822    22    4 (18%)
    3      12x11   1 blue, 1 red      663    37    7 (19%)
    4      14x11   4 blue, 1 red     8060    20   11 (55%)

121 presses in all, 10861 states enumerated, ~2 minutes for the five fields
(which is the whole cost of generation: they are seed-independent, so seed 0
pays for them and every later seed replays the cached plans at its own
presentation).

Expert solver
-------------
There is no native model here and nothing to fuzz: the interpreter IS the model.
`_Field` enumerates, per level,

  1. a forward BFS from the level start over ``eng.step``, layer by layer,
     recording the successor of every state under each of the four presses and
     marking the presses that WIN (winning is a transition, not a state:
     ``check_win`` is read after a press and a won level is terminal, so those
     edges lead nowhere). The first winning edge appears while expanding layer
     ``D - 1``, which fixes ``D`` -- the true shortest solution length -- and
     the sweep then finishes that layer and STOPS;
  2. a reverse BFS over the predecessor map of exactly that slice, which labels
     each state in it with its distance to a win.

**Why the sweep stops at ``D - 1``, and why the field is still exact.** The full
reachable space of this game is not small -- the bodies can be shuffled around
the garden more or less forever, and level 2 alone did not close after ten
minutes -- but nothing beyond depth ``D - 1`` can matter. If ``s`` lies on some
shortest solution then ``g(s) + d(s) = D``, and every state ``t`` on ``s``'s own
shortest path to the win inherits ``g(t) + d(t) <= D`` with ``d(t) >= 1``, so
``g(t) <= D - 1``: the whole path was enumerated. Hence the reverse sweep's
distance is EXACT for every state that can carry a shortest solution, and merely
an over-estimate (or absent) for the rest -- which is precisely the direction
that keeps the optimal sets below sound. ``--verify`` is what stops that being
an argument on paper: it re-derives every distance in the slice from its own
successors, and then re-prices every labelled press from the other end with a
search that shares nothing with the field.

An A* here would need a heuristic, and there is nothing honest to charge: the
girl's distance to the gap is flat over most of a solution (you spend a dozen
presses walking round a body to shove it from the other side), and "how welded
is the garden" is not a distance at all. The bounded field asks nothing and
proves the answer.

Optimal-action sets
-------------------
The field gives these exactly and for free: at a state ``s`` with distance ``d``,
a press is optimal iff it wins (when ``d == 1``) or reaches a state at distance
``d - 1``. A press the engine refuses -- walking into the fence, or a push the
``cancel`` rule threw away -- maps ``s`` to itself, whose distance is ``d``, so
no-ops are excluded by construction. No step ships unlabelled (the
always-emit-optimal-targets rule), and a step with a unique optimal press ships a
one-element set. 31 of the 121 steps have a genuine tie, and they are not spread
evenly -- 55% of level 4 against 14% of level 0, which is what you would expect
of a game whose plans are mostly the girl WALKING to the far side of a block she
wants to shove: the more open the garden, the more ways round.

Rendering (the goal was not drawn at all)
-----------------------------------------
See the comment block at the top of ``data/puzzlescript_games/Little_Girl,_Big_World.txt``;
``--audit`` is the check. In short: ``Goal`` shipped as ``Transparent`` with no
sprite, so the one square the win condition names painted nothing -- the frame
showed a gap in the fence and no way to tell it from any other stretch of it. It
is a solid pink square now. And because a wall can be shoved into that gap, all
three walls were given TRANSPARENT CORNERS so the pink shows through one; the
corners are the holes that survive `_render_cell_sprite`'s centred sampling at
cell_px 4, which is what level 4's 14-row board renders at. ``--audit`` renders
every cell COMPOSITION the collision layers permit -- as a whole 64x64 frame,
never a cell crop, because `_render_frame` upscales and letterboxes -- at every
cell size the five levels use, and asserts they are pairwise distinct.

Augmentation
------------
This game's engine state after reset is identical for every seed (levels are
fixed ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`, which this game was added to). 5 levels x 16
presentations = 80.

This is the purest form of the ESCAPE! argument in this family. Every rule is
stated with a RELATIVE force -- ``>`` and ``moving``, never ``up`` or ``left`` --
there is no gravity, the win condition names no direction, input is
screen-relative, and no sprite encodes a facing (the girl faces the reader in
every frame; the blocks are symmetric decorations). So the interpreter is
equivariant under the full 8-element symmetry group by construction, and
``--symmetry`` measures it rather than arguing it: each level's LAYOUT is
rebuilt under all eight transforms, reloaded through the interpreter so the
level-start rules re-run, and the level's own plan is replayed with the presses
transformed -- the whole board has to land where the transform says, and win on
the same press. There is no colour augmentation.

The expert plan is therefore seed-independent: solved once per level, cached to
``data/little_girl_big_world_plans.json``, and replayed per seed with that
seed's remapped screen actions.

Recovery
--------
``recovery_mode = "reset"`` (the family default). The episode-wide epsilon prefix
explores freely -- and in this game a wrong shove is frequently PERMANENT, since
two same-coloured bodies welded together never come apart, so the prefix
generates genuinely unrecoverable boards, which is exactly the data the reset
behaviour is for -- and ONE RESET restores the level start, from which the cached
plan replays a guaranteed win.

Usage (run from the repo root):
    python solvers/generate_little_girl_big_world_training.py \
        --episodes 200 --out data/training_multi_level/little_girl_big_world

    python solvers/generate_little_girl_big_world_training.py --plans
    python solvers/generate_little_girl_big_world_training.py --verify
    python solvers/generate_little_girl_big_world_training.py --symmetry
    python solvers/generate_little_girl_big_world_training.py --audit
"""

from __future__ import annotations

import itertools
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                          # noqa: E402
from adapters.puzzlescript_adapter import _render_frame               # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert, Plan,   # noqa: E402
                                     screen_action)

GAME_NAME = "Little_Girl,_Big_World"

#: Where each level's start plan is cached between processes. The fields are
#: seed-independent, so without this every `parallelize_generator` shard would
#: re-enumerate all five slices (~2 min) before recording a frame.
PLAN_CACHE = (Path(__file__).resolve().parent.parent / "data"
              / "little_girl_big_world_plans.json")

#: The four presses, in the order the labels list them. ACTION5 is bound to no
#: rule in this game, so it is a pure no-op and is not in the search at all.
_ORDER = ("up", "down", "left", "right")

#: `_Field.succ` entry for a press that wins. A won level is terminal -- the
#: recorder stops on it -- so the edge has no destination state.
_WIN = -1


def _assert_barrier_is_immovable(gm) -> None:
    """The Barrier fence is a constant of the level -- no rule ever creates,
    destroys or moves one.

    Two things rest on it and would both fail silently if it stopped being
    true: `_Field` keeps the barriers out of its state key (so a barrier that
    moved would be a state change the field could not see), and `_audit`
    excludes the ``goal+barrier`` stack as unbuildable (so a barrier that could
    be shoved into the gap would hide the goal in a frame the audit never
    renders). Cheap enough to re-check on every field build.
    """
    for rule in gm.rules:
        for cell in rule.patterns_rhs:
            for (_mod, name, _ids) in cell:
                assert name != "barrier", "a rule writes the Barrier"
        for cell in rule.patterns_lhs:
            for (mod, name, _ids) in cell:
                assert not (name == "barrier" and mod), \
                    f"a rule moves the Barrier ({mod} {name})"


# ---------------------------------------------------------------------------
# The exact distance-to-win field, over the slice that can carry a solution
# ---------------------------------------------------------------------------

class _Field:
    """Every state within ``D - 1`` presses of a level's start, with its exact
    distance to a win, computed by driving the real interpreter.

    The state is every cell of the girl, the walls and the goal -- everything
    the board holds except the Background that is in every cell and the Barrier
    fence that never moves. This game has no hidden object, no cosmetic object
    and no state that is not a position, so that set is canonical: the grid is
    REBUILT from it (plus `static`) to re-expand a state.

    See the module header for why stopping at ``D - 1`` costs no exactness where
    it is read. `overflow` reports the one way that can fail: a level whose
    slice does not close inside `cap` states, for which no plan is returned at
    all rather than an unproven one.
    """

    def __init__(self, eng, gm, cap: int = 200_000):
        self.eng = eng
        self.gm = gm
        self.bg = gm.obj_name_to_idx["background"]
        self.cap = cap
        self.keys: list[frozenset] = []
        self.succ: list[list[int] | None] = []
        self.dist: dict[int, int] = {}
        #: The true shortest solution length, or None when no win was reached.
        self.star: int | None = None
        self.overflow = False

        # A state is the DYNAMIC objects only. Background is in every cell and
        # the Barrier fence never moves (`build` asserts both), so the rest of
        # the board is a constant of the level and is carried in `self.static`
        # instead of in every one of the thousands of keys: a level-4 key holds
        # ~40 packed ints where the whole board would hold 154 tuples. That is a
        # MEMORY saving and not a speed one -- measured, these searches spend
        # essentially all of their time inside ``eng.step``, and narrowing the
        # key moved the wall clock by under 5%.
        self.n_obj = gm.n_objects
        self.dyn = {i for name, i in gm.obj_name_to_idx.items()
                    if name not in ("background", "barrier")}
        self.static = [(r, c, o)
                       for r, row in enumerate(eng.grid)
                       for c, cell in enumerate(row)
                       for o in cell
                       if o != self.bg and o not in self.dyn]

    # -- engine plumbing ------------------------------------------------------
    def key(self) -> frozenset:
        """The dynamic cells, packed one int per ``(cell, object)``."""
        dyn, n, w = self.dyn, self.n_obj, self.eng.width
        return frozenset(
            (r * w + c) * n + o
            for r, row in enumerate(self.eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    def seat(self, key: frozenset) -> None:
        """Put ``key``'s board on the engine: Background everywhere, the level's
        immovable cells, then the key."""
        eng, n, w = self.eng, self.n_obj, self.eng.width
        grid = [[{self.bg} for _ in range(w)] for _ in range(eng.height)]
        for (r, c, o) in self.static:
            grid[r][c].add(o)
        for v in key:
            cell, o = divmod(v, n)
            grid[cell // w][cell % w].add(o)
        eng.grid = grid
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    # -- the two sweeps -------------------------------------------------------
    def build(self) -> "_Field":
        """Forward BFS out to depth ``D - 1``, then the reverse BFS that turns
        it into a distance-to-win field."""
        eng = self.eng
        assert all(self.bg in cell for row in eng.grid for cell in row), \
            "a cell without Background: the key rebuild would drop it"
        _assert_barrier_is_immovable(self.gm)
        start = self.key()
        index = {start: 0}
        self.keys = [start]
        self.succ = [None]
        layer = [0]
        depth = 0
        while layer and (self.star is None or depth < self.star):
            nxt: list[int] = []
            for i in layer:
                row: list[int] = []
                for direction in _ORDER:
                    self.seat(self.keys[i])
                    eng.step(direction)
                    if eng.check_win():
                        # The FIRST layer that can win fixes D: BFS expands in
                        # depth order, so no earlier press could have won.
                        if self.star is None:
                            self.star = depth + 1
                        row.append(_WIN)
                        continue
                    k = self.key()
                    j = index.get(k)
                    if j is None:
                        j = len(self.keys)
                        index[k] = j
                        self.keys.append(k)
                        self.succ.append(None)
                        nxt.append(j)
                    row.append(j)
                self.succ[i] = row
            if len(self.keys) > self.cap:                     # pragma: no cover
                self.overflow = True
                return self
            layer = nxt
            depth += 1

        preds: list[list[int]] = [[] for _ in self.keys]
        frontier = []
        for i, row in enumerate(self.succ):
            if row is None:            # depth D: enumerated but never expanded
                continue
            for j in row:
                if j == _WIN:
                    if i not in self.dist:
                        self.dist[i] = 1
                        frontier.append(i)
                else:
                    preds[j].append(i)
        queue = deque(frontier)
        while queue:
            i = queue.popleft()
            for p in preds[i]:
                if p not in self.dist:
                    self.dist[p] = self.dist[i] + 1
                    queue.append(p)
        # The forward and the reverse sweep priced the start independently; if
        # they disagreed the slice would be the wrong shape, not merely slow.
        assert self.star is None or self.dist.get(0) == self.star, (
            f"forward D={self.star} but the field says {self.dist.get(0)}")
        return self

    # -- reading it off -------------------------------------------------------
    def optimal(self, state: int) -> list[str]:
        """Every press that is equally shortest at ``state``."""
        d = self.dist[state]
        row = self.succ[state]
        if d == 1:
            return [a for a, j in zip(_ORDER, row) if j == _WIN]
        return [a for a, j in zip(_ORDER, row)
                if j != _WIN and self.dist.get(j, 1 << 30) == d - 1]

    def plan(self) -> tuple[list[str] | None, list[list[str]]]:
        """``(presses, optsets)`` for the walk downhill from the level start, or
        ``(None, [])`` when no win is reachable."""
        if 0 not in self.dist:
            return None, []
        presses: list[str] = []
        optsets: list[list[str]] = []
        state = 0
        while True:
            best = self.optimal(state)
            optsets.append(best)
            presses.append(best[0])
            if self.dist[state] == 1:
                return presses, optsets
            state = self.succ[state][_ORDER.index(best[0])]


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class LittleGirlBigWorldExpert(PSExpert):
    """Plans by enumerating the solution-bearing slice of the level's state
    space with `_Field` and walking the exact distance-to-win field downhill.

    `PSExpert` still owns everything around that -- the in-memory memo, the
    on-disk plan cache with its staleness check and the snapshot/restore
    discipline -- so `_search` is the only real override. `heuristic` is
    unreachable by construction: nothing here runs A*.
    """

    directions = list(_ORDER)
    plan_cache_path = PLAN_CACHE

    def heuristic(self, eng) -> int:                          # pragma: no cover
        raise AssertionError("the field is exact; no search runs here")

    def field(self, eng) -> _Field:
        """The field for whatever board the engine is currently showing."""
        return _Field(eng, self.g).build()

    def _search(self, eng) -> list | None:
        presses, optsets = self.field(eng).plan()
        return None if presses is None else Plan(presses, optsets)


class LittleGirlBigWorldSolver(PSAStarSolver):
    game_id = "puzzlescript_little_girl_big_world"
    game_name = GAME_NAME
    game_module_id = "ps:little_girl_big_world"
    expert_cls = LittleGirlBigWorldExpert

    #: The longest plan is 37 presses; the rest is room for the exploration
    #: prefix and the re-plan after it, and it stays well under the adapter's
    #: 200-step per-level budget.
    max_steps = 100

    def prepare_expert(self, game, expert) -> None:
        """Build every level's field before `discover_solvable` asks for it --
        the same work either way, but it makes the one-time ~2 min visible as
        startup rather than as a mysteriously slow first seed, and it fills the
        disk cache in one pass."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with the adapter parked at level 0."""
    solver = LittleGirlBigWorldSolver()
    game = solver.make_game(0)
    return solver, game, LittleGirlBigWorldExpert(game)


def _bodies(eng, gm) -> str:
    """A census of the rigid bodies on the board: connected components per
    wall colour. It is the only structural fact about a level that matters --
    what you can shove, and how big it is."""
    out = []
    for name, tag in (("wall1", "blue"), ("wall2", "red"), ("wall3", "yellow")):
        idx = gm.obj_name_to_idx[name]
        cells = {(r, c) for r, row in enumerate(eng.grid)
                 for c, cell in enumerate(row) if idx in cell}
        n = 0
        while cells:
            n += 1
            seed = cells.pop()
            queue = deque([seed])
            while queue:
                r, c = queue.popleft()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    if (r + dr, c + dc) in cells:
                        cells.discard((r + dr, c + dc))
                        queue.append((r + dr, c + dc))
        if n:
            out.append(f"{n} {tag}")
    return ", ".join(out) if out else "none"


def _replay_on_adapter(game, level: int, presses) -> str:
    """Drive the plan through `PuzzleScriptAdapter.perform_action` -- the same
    entry point a live agent uses, at this seed's rotation and flips -- and
    report what the ADAPTER thought happened.

    The field is built on bare ``eng.step``. That is one layer below what the
    agent plays: the adapter also owns the presentation augmentation and the
    directional remap, the animation settling, and the per-level step budget. A
    plan certified only against the engine would be certified against something
    nobody drives, and the rotation contract in particular fails SILENTLY (see
    `screen_action`), so it is checked here rather than assumed.
    """
    game.set_level(level)
    rot, hf, vf = game._rotation_k, game._hflip, game._vflip
    for i, direction in enumerate(presses):
        act = screen_action(direction, rot, hf, vf)
        game.perform_action(ActionInput(id=act))
        if game._state == GameState.WIN:
            return ("WON ON THE LAST PRESS" if i == len(presses) - 1
                    else f"won early at press {i} of {len(presses)}")
        if game._state == GameState.GAME_OVER:
            return f"GAME OVER at press {i}"
    return "NEVER WON"


def _report() -> int:
    """Per-level board size, rigid-body census, plan length, tie coverage and
    slice size -- and CERTIFY each plan against the interpreter AND the
    adapter."""
    _solver, game, expert = _levels()
    eng, gm = game._engine, game._game
    total = bad = states = 0
    for level in range(game.n_levels):
        game.set_level(level)
        # Read the START board before the field is built: the BFS leaves the
        # engine seated on whichever state it expanded last, so a census taken
        # afterwards would be counting a scrambled garden.
        size, bodies = f"{eng.height:2d}x{eng.width:2d}", _bodies(eng, gm)
        t = time.time()
        field = expert.field(eng)
        presses, optsets = field.plan()
        took = time.time() - t
        states += len(field.keys)
        head = (f"level {level}: {size} {bodies:22s} "
                f"{len(field.keys):6d} states in {took:5.1f}s")
        if presses is None:
            why = ("GAVE UP AT THE STATE CAP" if field.overflow
                   else "NO WIN IS REACHABLE")
            print(f"{head}: {why}")
            bad += 1
            continue
        # Certification 1: the interpreter must win on the LAST press and no
        # earlier -- an earlier win would mean the plan is not shortest.
        game.set_level(level)
        won_at = None
        for i, direction in enumerate(presses):
            eng.step(direction)
            if eng.check_win():
                won_at = i
                break
        engine_ok = won_at == len(presses) - 1
        # Certification 2: the same plan, through the adapter an agent drives.
        verdict = _replay_on_adapter(game, level, presses)
        adapter_ok = verdict == "WON ON THE LAST PRESS"
        bad += not (engine_ok and adapter_ok)
        total += len(presses)
        ties = sum(1 for s in optsets if len(s) > 1)
        room = "ok" if len(presses) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(presses):3d} presses "
              f"(budget {game._max_steps}, {room}), "
              f"{ties:3d} tie steps ({ties / len(presses):3.0%}), "
              f"engine {'CERTIFIED' if engine_ok else 'REJECTED'}, "
              f"adapter {verdict}")
    print(f"total {total} presses over {game.n_levels} levels, "
          f"{states} states enumerated")
    return 0 if not bad else 1


def _bellman(field: "_Field") -> list[str]:
    """Complaints about the field's internal consistency, as a list (empty is
    clean). Cheap: one pass over the slice, no interpreter steps.

    `_Field` builds the successor table in one sweep and the distances in
    another, and only the SECOND is a guess about which states matter. This
    checks the two against each other: a distance function has to satisfy
    ``d(s) = 1 + min over successors`` (a winning press counting as 0), and no
    state may be labelled when none of its successors is. It catches a
    predecessor-map or reverse-BFS bug in a second, over the WHOLE slice rather
    than over the ~100 states one plan walks through -- which is the coverage
    the expensive re-pricing below cannot afford.

    What it deliberately cannot catch is the depth bound being wrong, since it
    only ever reads states the bound already admitted. That is exactly what
    `_verify`'s independent searches are for, and why both run.
    """
    notes = []
    for i, row in enumerate(field.succ):
        if row is None or i not in field.dist:
            continue
        best = min((0 if j == _WIN else field.dist.get(j, 1 << 30))
                   for j in row)
        if field.dist[i] != best + 1:
            notes.append(f"state {i}: d={field.dist[i]} but 1+min succ={best + 1}")
        if len(notes) > 3:
            break
    return notes


def _verify() -> int:
    """Double-entry check of every plan length and every shipped tie set.

    `_Field` answers both from ONE enumeration: a forward sweep that builds the
    successor table and a reverse sweep over its predecessor map, run only over
    the states the depth bound admitted. A bug in the reverse sweep, or in the
    bound argument itself, would produce a self-consistent field, a plan that
    still wins, and tie sets that are quietly wrong -- and the training labels
    ARE those tie sets, so "it wins" is not enough of a check.

    Two independent passes, because they fail differently:

      * `_bellman` re-derives every distance in the slice from its own
        successors. Whole-slice coverage, one second, blind to the bound.
      * then each candidate press along the plan is priced again from the other
        end, by a fresh forward BFS: no reverse sweep, no predecessor map, no
        successor table, no depth bound derived from ``D``, nothing shared with
        the field but ``eng.step`` and the key encoder. Even the plan's own
        states are re-walked through the interpreter rather than read out of
        ``succ``. A press is optimal iff that distance is ``d* - i - 1``, and
        that set must be exactly what the field shipped.

    Two prunes keep the second pass affordable and neither is heuristic. The
    BFS is cut off at the presses the plan has left -- a successor that cannot
    win inside the remaining budget is not optimal, and one that can is priced
    exactly. And a press that leaves the board UNCHANGED (walking into the
    fence, or a push the ``cancel`` rule threw away) is dropped without a
    search: it spends a move to reach the state it started from, so it can
    never be on a shortest path.

    It is still the slow mode -- ~26 minutes of searching for the five levels
    (8s / 40s / 204s / 830s / 500s), against two minutes to build the fields --
    because a bounded BFS from a state one press off the plan explores nearly
    the whole slice again, once per candidate press. Note it is level 3 that
    costs the most, not the level with the biggest slice: its ``d*`` is 37, and
    the bound on every one of those searches is what is left of it. An offline
    check, not part of generation.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        field = expert.field(eng)
        presses, optsets = field.plan()
        if presses is None:
            print(f"level {level}: NO PLAN")
            bad += 1
            continue
        notes = _bellman(field)

        t = time.time()
        searches = 0

        def distance(key: frozenset, bound: int) -> int | None:
            """Presses from ``key`` to a win by a fresh BFS, or None when that
            is more than ``bound``."""
            nonlocal searches
            searches += 1
            seen = {key}
            layer = [key]
            for d in range(bound):
                nxt = []
                for k in layer:
                    for direction in _ORDER:
                        field.seat(k)
                        eng.step(direction)
                        if eng.check_win():
                            return d + 1
                        nk = field.key()
                        if nk not in seen:
                            seen.add(nk)
                            nxt.append(nk)
                layer = nxt
            return None

        star = len(presses)
        game.set_level(level)
        cur = field.key()                 # the plan's states, re-walked here
        for i, taken in enumerate(presses):
            remaining = star - i
            want = []
            for a in _ORDER:
                field.seat(cur)
                eng.step(a)
                if eng.check_win():
                    got = 1
                else:
                    nxt = field.key()
                    if nxt == cur:        # a no-op press; see the docstring
                        got = None
                    else:
                        sub = distance(nxt, remaining - 1)
                        got = None if sub is None else sub + 1
                if got == remaining:
                    want.append(a)
            if want != optsets[i]:
                notes.append(f"step {i}: {optsets[i]} != {want}")
            field.seat(cur)
            eng.step(taken)
            cur = field.key()
        bad += len(notes)
        print(f"level {level}: d*={star:3d}, "
              f"{sum(len(s) for s in optsets):3d} labelled presses re-priced by "
              f"{searches:3d} independent searches in {time.time() - t:5.1f}s "
              f"(+ {len(field.dist)} distances re-derived by the Bellman pass): "
              f"{'AGREES' if not notes else '; '.join(notes[:3])}", flush=True)
    print("verify clean" if not bad else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# The presentation group, measured
# ---------------------------------------------------------------------------

#: The 8-element symmetry group of the square, generated by one clockwise
#: quarter turn and one left-right mirror. The adapter's own augmentation is
#: rotation x hflip x vflip = 16 presentations, but a vertical flip is a mirror
#: composed with a half turn, so those 16 are these 8 transforms twice over --
#: testing the group tests every presentation.
_MIRROR = {"left": "right", "right": "left", "up": "up", "down": "down"}
_ROT = {"up": "right", "right": "down", "down": "left", "left": "up"}


def _symmetry() -> int:
    """Replay every level's plan on all EIGHT turned/mirrored copies of its own
    LAYOUT and require the whole board to land where the transform says.

    This is the evidence behind putting the game in `_FLIP_GAMES` (and behind
    the mandatory rotation). Unlike Impasse or Goblin Hooblob there is nothing
    to relabel: no object in this game carries a direction, in its rules or in
    its art, so the transform is applied to the LAYOUT and to the PRESSES and to
    nothing else. Applying it to the layout rather than to a grid matters --
    ``load_level`` re-runs the level-start rules, so the test measures a board
    the interpreter itself built.

    A clean run says the mechanic is equivariant under the full 8-element
    symmetry group, i.e. every one of the 16 presentations shows a board this
    game could have shipped, played by the same rules.
    """
    _solver, game, expert = _levels()
    eng, gm = game._engine, game._game
    #: Read the board as ``(row, col, object)`` triples, in COORDINATES rather
    #: than in `_Field`'s width-packed integers -- a quarter turn swaps the
    #: board's height and width, so the packing means something different on
    #: each side of the comparison. Barriers are read too: they are static, but
    #: "static" is a claim about the rules and this is the test that would catch
    #: it being wrong.
    def read():
        return frozenset((r, c, o)
                         for r, row in enumerate(eng.grid)
                         for c, cell in enumerate(row)
                         for o in cell
                         if o != gm.obj_name_to_idx["background"])

    def turn_layout(layout, rot: int, mirror: bool):
        rows = [[set(cell) for cell in row] for row in layout]
        for _ in range(rot):
            rows = [list(r) for r in zip(*rows[::-1])]     # clockwise quarter
        if mirror:
            rows = [row[::-1] for row in rows]
        return rows

    def turn_key(cells, rot: int, mirror: bool, h: int, w: int):
        out = set()
        for (r, c, o) in cells:
            rr, cc, hh, ww = r, c, h, w
            for _ in range(rot):
                rr, cc, hh, ww = cc, hh - 1 - rr, ww, hh
            if mirror:
                cc = ww - 1 - cc
            out.add((rr, cc, o))
        return frozenset(out)

    def turn_press(direction: str, rot: int, mirror: bool) -> str:
        for _ in range(rot):
            direction = _ROT[direction]
        return _MIRROR[direction] if mirror else direction

    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        layout = gm.levels[level]
        h, w = eng.height, eng.width
        field = expert.field(eng)
        presses, _optsets = field.plan()

        # The reference: the plan played straight through, and where it leaves
        # the board. (The win deletes the Goal, so this is a real board.)
        eng.load_level(layout)
        for direction in presses:
            eng.step(direction)
            if eng.check_win():
                break
        reference = read()

        notes = []
        for rot, mirror in itertools.product(range(4), (False, True)):
            eng.load_level(turn_layout(layout, rot, mirror))
            steps = won = 0
            for direction in presses:
                eng.step(turn_press(direction, rot, mirror))
                steps += 1
                if eng.check_win():
                    won = 1
                    break
            got = read()
            want = turn_key(reference, rot, mirror, h, w)
            if not won or steps != len(presses) or got != want:
                notes.append(f"rot={rot} mirror={mirror}")
        bad += len(notes)
        print(f"level {level}: {len(presses):3d} presses replayed on 8 "
              f"presentations: "
              f"{'EQUIVARIANT' if not notes else 'SPLIT BY ' + ', '.join(notes)}")
    print("the full 8-element symmetry group is a symmetry of the mechanic: "
          "every rule is relative, no sprite encodes a facing" if not bad else
          f"THE AUGMENTATION IS NOT SAFE: {bad} presentations disagree")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Rendering audit
# ---------------------------------------------------------------------------

#: Layer 2 holds at most one of these; layer 1 holds the Goal or nothing. The
#: audit set is their product, minus two pairs that cannot be built -- and it is
#: deliberately the LEGAL set rather than the set the plans walk through, since
#: the RESET exploration prefix presses at random and can shove a wall anywhere
#: the rules allow, including into the gap the Goal sits in.
_LAYER2 = (None, "player", "wall1", "wall2", "wall3", "barrier")

#: The two impossible stacks, and why each is impossible.
#:
#: ``goal+player``: the rule that steps the girl into the gap
#: (``[ > Player | Goal ] -> [ | Player ]``) DELETES the goal in the same turn,
#: so the two never share a frame -- and that turn is the win, after which the
#: episode is over.
#:
#: ``goal+barrier``: no rule in the game moves a Barrier, so a barrier is only
#: ever where the level LAYOUT put it, and no shipped layout puts one on the
#: goal cell. `_audit` asserts both halves of that rather than trusting this
#: comment -- it matters, because the Barrier sprite is a solid 5x5 and would
#: hide the goal completely if the two could ever stack.
_IMPOSSIBLE = {("goal", "player"), ("barrier", "goal")}


def _compositions() -> list[tuple[str, ...]]:
    out = []
    for goal in (False, True):
        for other in _LAYER2:
            objs = tuple(sorted((["goal"] if goal else [])
                                + ([other] if other else [])))
            if objs in _IMPOSSIBLE:
                continue
            out.append(objs)
    return out


def _audit() -> int:
    """Assert every cell COMPOSITION the game can show is distinct in the frame,
    at every cell size the five levels render at.

    Whole frames are compared, not cell crops: `_render_frame` upscales the
    board to fill 64x64 and letterboxes it, so the output's cell grid is not
    ``cell_px``-aligned and an arithmetic crop reads the wrong window (see
    ps:esl_puzzle_game and ps:explod, where exactly that made an audit report
    every composition identical to floor).

    The pair this game exists to check is ``goal+wall*`` against ``wall*``: a
    wall shoved into the gap must not hide where the gap was.
    """
    _solver, game, expert = _levels()
    eng, gm = game._engine, game._game
    idx = gm.obj_name_to_idx

    # The two claims `_IMPOSSIBLE` rests on, checked here so the exclusion is a
    # measurement rather than a comment: no rule can put a Barrier in motion,
    # and no shipped layout starts one on the goal cell.
    _assert_barrier_is_immovable(gm)
    goal_i, barrier_i = idx["goal"], idx["barrier"]
    for level in range(game.n_levels):
        game.set_level(level)
        for row in eng.grid:
            for cell in row:
                assert not (goal_i in cell and barrier_i in cell), \
                    "a level ships a Barrier on the goal cell"

    sizes = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    comps = _compositions()
    clashes_total = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for objs in comps:
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[objs] = np.asarray(_render_frame(eng, gm)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        clashes_total += len(clashes)
        name = lambda t: "+".join(t) if t else "floor"        # noqa: E731
        print(f"{h}x{w} board (level{'s' if len(levels) > 1 else ''} "
              f"{', '.join(str(x) for x in levels)}), cell {64 // max(h, w)}px, "
              f"{len(comps)} compositions")
        for objs in comps:
            painted = int((shots[objs] != shots[()]).sum())
            over = ""
            if "goal" in objs and len(objs) > 1:
                bare = tuple(x for x in objs if x != "goal")
                over = (f", {int((shots[objs] != shots[bare]).sum()):4d} px "
                        f"of goal survive under the {bare[0]}")
            print(f"  {name(objs):20s} {painted:4d} px differ from bare floor"
                  f"{over}")
        for a, b in clashes:
            print(f"  IDENTICAL: {name(a)} == {name(b)}")
    print("audit clean" if not clashes_total
          else f"AUDIT FAILED: {clashes_total} indistinguishable pairs")
    return 0 if not clashes_total else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(LittleGirlBigWorldSolver.main())
