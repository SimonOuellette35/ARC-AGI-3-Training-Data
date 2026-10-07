"""Generate Phase-1 training data for the PuzzleScript game ps:box_fill.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema::

    {
      "game_id": "puzzlescript_box_fill",
      "levels": [
        {"level_id": 0,
         "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
         "actions":      [a_0, ..., a_{T-1}]},     # length T
        ...
      ]
    }

``actions[0]`` is the RESET that produced ``observations[0]``; ``actions[i]`` for
i>=1 is the action that took the agent from ``observations[i-1]`` to
``observations[i]``. The recorded index is the *screen* action (post
rotation/flip remap), i.e. the button an agent presses in the presented view, so
replaying the recorded actions reproduces the recorded frames exactly.


THE GAME
========
Box Fill (Pichusuperlover) is a walled 5x5 room with an orange **box creator**
bolted into one corner. The goal is to cover every Air cell in the room with a
box -- ``All Air on Box``.

The twist is that **the creator never moves**. Its whole rule set is::

    [ > BoxCreator | Air      ] -> [ BoxCreator | Air TempBox      ]
    [ > BoxCreator | GateDown ] -> [ BoxCreator | GateDown TempBox ]
    [ > BoxCreator            ] -> [ BoxCreator                    ]   (cancel)
    late [ TempBox ] -> [ Box ]

so pressing a direction strips the creator's own movement and instead EXTRUDES a
new box into the neighbouring cell. But ``Player = BoxCreator or Box``, so every
box already on the board gets that same directional force and simply *moves*.
One press therefore does two things at once:

  1. every box slides one cell in the pressed direction (a box against a wall,
     the creator, or another blocked box just stays; a train of boxes all shuffle
     along together if the leader can move), and
  2. a fresh box appears in the cell next to the creator.

The order matters and is not the one the source reads like: PuzzleScript runs the
rules BEFORE movement resolution, so the TempBox is *marked* while the old box is
still there, the old box then slides away, and only the late rule turns the mark
into a Box. The net effect is **move first, then spawn**, which is why holding
one direction lays down a solid line rather than jamming after one box.

A press whose neighbouring cell is a wall spawns nothing, but still moves every
box on the board -- that is the entire cost model. The room needs one box per Air
cell, each press yields at most one box, so a level with ``A`` uncovered Air
cells needs at least ``A`` presses and a "wasted" press is a press that only
shuffled.

**Gates and buttons** (levels 3-7) add the one irreversible mechanic::

    late [ Box Button ] [ GateDown ] -> [ Box Button ] [ GateUp ]

GateDown (light blue) sits *below* the box layer, so boxes slide straight over it
and the creator can even extrude into it; GateUp (dark blue) sits *on* the box
layer, so it is a wall. The moment any box touches any button every gate in the
level latches up, permanently -- and because GateUp lands on the box layer, a box
standing on a gate at that instant is **destroyed**. Gate cells carry no Air, so
they never need covering; a button cell does. So the shape of every gated level
is: thread boxes through the gates first, and only tap the button once nothing is
left on the far side.


THE EXPERT: ONE EXACT DISTANCE FIELD PER LEVEL
==============================================
The whole state is ``(which cells hold a box, have the gates latched)``. Boxes
are indistinguishable, the rooms are 5x5, and nothing else on the board can
change -- so the state is a 25-bit mask plus one bit, and the reachable set is
small enough to enumerate outright. Measured, from each level's start:

    level    0       1      2      3       4      5      6     7
    states  126612  67141  5061  129597  6752  2493  16823  186

`Field` walks that set forward from the level start, then runs a backward BFS
from every winning state to get the EXACT number of presses still needed from
anywhere in it. That single table answers all three questions the recorder asks,
in O(1) each and with no search at runtime:

  * `Field.plan` -- an optimal press sequence from the live state, so
    ``recovery_mode = "replan"``: an exploratory detour or a perturbation burst
    just lands on another state the table already knows the answer for.
  * `Field.optimal` -- the FULL set of equally-optimal presses, which is the
    training target and the source of trajectory diversity. Ties are abundant
    here (a bare room can be filled column-first or row-first and everything in
    between), so stochastic-optimal sampling makes each seed a genuinely
    different perfect playthrough rather than a re-rendering of one route.
  * dead states -- a level really can be bricked (shove a box into a corner the
    conveyor can no longer reach past, or latch the gates with boxes still to
    thread through). Those get ``dist = -1``, `solve_from` returns ``[]``, and
    the recorder rolls the burst back or RESETs.

`Level.step` is the one place this generator re-implements PuzzleScript instead
of calling it, so it is differentially tested against the real interpreter --
`verify_model` (``--verify-model``) replays random walks on all eight levels and
compares the box set, the gate latch AND the win verdict after every single
press.


PRESENTATION FIXES MADE FOR THIS GAME
=====================================
* ``BoxCreator`` was ``orange``, which quantizes to the SAME ARC index (12) as
  the wall's ``brown``: the creator sat inside the wall frame and was literally
  invisible, while *where it is* determines every spawn in the game. It is now
  ``purple`` (15), which nothing else here uses. ``Box``/``TempBox`` were
  ``lightbrown``, a name the quantizer does not know, so they reached index 2 via
  the generic fallback -- named ``gray`` now so the colour is a decision, not an
  accident (same rendered index, no pixel changes). See
  [[ps-palette-collisions]].
* ``Box_Fill`` joins ``PuzzleScriptAdapter._FLIP_GAMES``: it is gravity-free with
  screen-relative moves, no rule or win condition names a direction, and no
  object has a sprite at all (every one is a solid colour block), so a reflected
  trajectory is one the game could really hand out. With eight levels that takes
  the corpus from 32 presentations to 128.
"""
from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                      # noqa: E402
from solvers.base_solver import (                                 # noqa: E402
    BaseSolver, DriveResult, _ID_TO_GAMEACTION)
from solvers.common.ps_astar import _load_game_module, screen_action  # noqa: E402
from utils.explore import Action                                  # noqa: E402

GAME_NAME = "Box_Fill"
_GAME_ID = "ps:box_fill"

#: The four presses. The prelude declares ``noaction`` and no rule mentions
#: ACTION, so ACTION5 cannot change a pixel and is left out of the search and of
#: the exploration action set alike -- offering it would only spend exploration
#: budget on a guaranteed null transition.
DIRS = ("up", "down", "left", "right")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


# ---------------------------------------------------------------------------
# The mechanic, compiled per level
# ---------------------------------------------------------------------------
#
# STATE ENCODING. A state is one integer: ``(box_mask << 1) | latched``, where
# ``box_mask`` has bit ``r * width + c`` set for every cell holding a box, and
# ``latched`` is 1 once a box has touched a button and the gates are up. Boxes
# are interchangeable and every other object on the board is static (Air, walls,
# buttons, the creator; gates only ever make the one-way GateDown -> GateUp
# transition, which ``latched`` records), so this is an EXACT canonical state --
# two boards with the same integer admit exactly the same futures.
#
# ``latched`` cannot be derived from the mask: a box can latch the gates and then
# slide off the button on a later press, leaving the gates up with no box on any
# button.

class Level:
    """The static geometry of one Box Fill level, compiled into press tables.

    Everything `step` needs is precomputed here, in two variants -- gates open
    and gates latched -- so a press is a walk over at most 25 cells with no
    conditionals about the board itself.
    """

    def __init__(self, engine, parsed) -> None:
        n2i = parsed.obj_name_to_idx
        self.height, self.width = engine.height, engine.width
        air = gates = buttons = blocked = 0
        boxes = 0
        latched = 0
        creator = None
        for r, row in enumerate(engine.grid):
            for c, cell in enumerate(row):
                bit = 1 << (r * self.width + c)
                if n2i["air"] in cell:
                    air |= bit
                if n2i["gatedown"] in cell or n2i["gateup"] in cell:
                    gates |= bit
                if n2i["gateup"] in cell:
                    latched = 1
                if n2i["button"] in cell:
                    buttons |= bit
                if n2i["wall"] in cell:
                    blocked |= bit
                if n2i["boxcreator"] in cell:
                    blocked |= bit
                    creator = (r, c)
                if n2i["box"] in cell:
                    boxes |= bit
        if creator is None:
            raise ValueError("Box Fill level with no BoxCreator")
        self.air, self.gates, self.buttons = air, gates, buttons
        self.creator = creator
        #: The board AS COMPILED, in the state encoding above. Compiled at a
        #: level's initial state this is the state a `Field` is rooted at.
        self.start = (boxes << 1) | latched
        #: Everything a press table depends on. Two levels with the same
        #: signature are the same puzzle, which is what lets a `Field` be cached
        #: across seeds -- and what makes a layout that ever becomes
        #: seed-dependent show up as a cache miss instead of a wrong answer.
        self.geometry = (self.height, self.width, air, gates, buttons,
                         blocked, creator)

        # Per direction: where each cell's box would go (-1 = it cannot move),
        # the front-first order to resolve them in, and where a press extrudes.
        # A latched gate is on the box layer, so it blocks movement AND, having
        # replaced GateDown with an object carrying no Air, kills the spawn.
        self.dest = {}          # (direction, latched) -> list[int] over cells
        self.order = {}         # direction -> cell indices, front-first
        self.spawn = {}         # (direction, latched) -> cell index or -1
        n_cells = self.height * self.width
        for d, (dr, dc) in _DELTA.items():
            self.order[d] = sorted(range(n_cells),
                                   key=lambda i: -((i // self.width) * dr
                                                   + (i % self.width) * dc))
            for lat in (0, 1):
                shut = blocked | (gates if lat else 0)
                dest = []
                for i in range(n_cells):
                    r, c = divmod(i, self.width)
                    nr, nc = r + dr, c + dc
                    j = nr * self.width + nc
                    dest.append(j if (0 <= nr < self.height
                                      and 0 <= nc < self.width
                                      and not (shut >> j) & 1) else -1)
                self.dest[(d, lat)] = dest
                # Rule 1 extrudes into any cell containing Air; rule 2 into a
                # GateDown cell (which carries no Air of its own). Once latched
                # there is no GateDown left, so only the Air branch survives.
                cr, cc = creator[0] + dr, creator[1] + dc
                tgt = -1
                if 0 <= cr < self.height and 0 <= cc < self.width:
                    bit = 1 << (cr * self.width + cc)
                    if (air & bit) or (not lat and (gates & bit)):
                        tgt = cr * self.width + cc
                self.spawn[(d, lat)] = tgt

    def step(self, state: int, d: str) -> int:
        """One press. Boxes move first, then the creator extrudes, then the gate
        latch fires -- the order the interpreter actually resolves them in (see
        the module docstring), not the order the source file reads in."""
        mask, lat = state >> 1, state & 1
        dest = self.dest[(d, lat)]
        occupied = mask
        moved = 0
        for i in self.order[d]:
            bit = 1 << i
            if not mask & bit:
                continue
            j = dest[i]
            if j >= 0 and not (occupied >> j) & 1:
                occupied = (occupied & ~bit) | (1 << j)
                moved |= 1 << j
            else:
                moved |= bit
        tgt = self.spawn[(d, lat)]
        if tgt >= 0:
            moved |= 1 << tgt
        if not lat and (moved & self.buttons):
            # Every GateDown becomes a GateUp on the box layer, so a box caught
            # standing on a gate at that moment is overwritten and gone.
            lat = 1
            moved &= ~self.gates
        return (moved << 1) | lat

    def won(self, state: int) -> bool:
        """``All Air on Box``: every Air cell carries a box. Gate cells hold no
        Air of their own, so latched gates never stand in the way of a win."""
        return ((state >> 1) & self.air) == self.air


# ---------------------------------------------------------------------------
# Exact distance-to-win over the whole reachable state space
# ---------------------------------------------------------------------------

class Field:
    """Every state reachable from a level's start, with the exact number of
    presses still needed from each.

    Built once per level and reused for every seed: the engine state after a
    reset is seed-independent for this game (the levels are fixed ASCII maps and
    only the PRESENTATION is augmented), so the table is a property of the level
    alone.

    Forward BFS enumerates the reachable set and records the successor of every
    (state, press) pair; the presses are then reversed into CSR form and a
    backward BFS from the winning states fills in the distances. Winning states
    are terminal -- the episode ends there -- so they get no outgoing presses.
    """

    #: Refuse to build a field bigger than this rather than exhaust memory on a
    #: level someone adds later whose room does not fit the premise. All eight
    #: shipped levels come in under 130k.
    state_cap: int = 4_000_000

    def __init__(self, level: Level, start: int | None = None) -> None:
        self.level = level
        self.start = level.start if start is None else start
        self._enumerate()
        self._solve()

    def _enumerate(self) -> None:
        level = self.level
        ids: dict[int, int] = {self.start: 0}
        states: list[int] = [self.start]
        succ: list[list[int]] = []
        queue = deque([self.start])
        while queue:
            state = queue.popleft()
            # The queue is FIFO and ids are handed out on first discovery, so
            # popleft order IS id order and ``succ`` can be built by appending.
            assert ids[state] == len(succ)
            if level.won(state):
                succ.append([-1] * len(DIRS))     # terminal: the level is over
                continue
            row = []
            for d in DIRS:
                nxt = level.step(state, d)
                sid = ids.get(nxt)
                if sid is None:
                    sid = ids[nxt] = len(states)
                    states.append(nxt)
                    queue.append(nxt)
                    if len(states) > self.state_cap:
                        raise MemoryError(
                            f"Box Fill reachable set exceeded {self.state_cap} "
                            f"states -- this level does not fit the premise "
                            f"that the whole space can be enumerated")
                row.append(sid)
            succ.append(row)
        self.ids = ids
        self.states = states
        self.succ = np.array(succ, dtype=np.int32)   # row i = state i's presses

    def _solve(self) -> None:
        succ = self.succ
        n = len(self.states)
        # Reverse the press graph into CSR (sorted by destination), skipping the
        # -1 rows terminal states carry.
        src = np.repeat(np.arange(n, dtype=np.int32), len(DIRS))
        dst = succ.ravel()
        keep = dst >= 0
        src, dst = src[keep], dst[keep]
        order = np.argsort(dst, kind="stable")
        src, dst = src[order], dst[order]
        ptr = np.searchsorted(dst, np.arange(n + 1, dtype=np.int32))

        dist = np.full(n, -1, dtype=np.int32)
        frontier = np.flatnonzero(
            np.fromiter((self.level.won(s) for s in self.states),
                        dtype=bool, count=n)).astype(np.int32)
        dist[frontier] = 0
        depth = 0
        while frontier.size:
            depth += 1
            cand = np.unique(np.concatenate(
                [src[ptr[v]:ptr[v + 1]] for v in frontier]))
            cand = cand[dist[cand] < 0]
            dist[cand] = depth
            frontier = cand
        self.dist = dist

    # ── the queries the recorder makes ──────────────────────────────────────
    def knows(self, state: int) -> bool:
        """Is ``state`` in the set this field was built over?

        Every state legal play can produce IS in it by construction, so a miss
        means the caller handed over a board the field was not built for (a
        different level, or a level whose layout became seed-dependent) -- which
        is a different thing from a state that is merely unwinnable."""
        return state in self.ids

    def distance(self, state: int) -> int | None:
        """Presses still needed, or None for a state that can never be won."""
        sid = self.ids.get(state)
        if sid is None:
            return None
        d = int(self.dist[sid])
        return None if d < 0 else d

    def optimal(self, state: int) -> list[str]:
        """Every press that strictly shortens the remaining distance -- the full
        optimal tie set, and the training target."""
        sid = self.ids.get(state)
        if sid is None:
            return []
        here = int(self.dist[sid])
        if here <= 0:
            return []
        return [d for i, d in enumerate(DIRS)
                if 0 <= self.succ[sid, i]
                and int(self.dist[self.succ[sid, i]]) == here - 1]

    def plan(self, state: int) -> list[str] | None:
        """An optimal press sequence from ``state``, or None if it is dead."""
        sid = self.ids.get(state)
        if sid is None or self.dist[sid] < 0:
            return None
        out: list[str] = []
        for _ in range(int(self.dist[sid])):
            here = int(self.dist[sid])
            for i, d in enumerate(DIRS):
                nxt = int(self.succ[sid, i])
                if nxt >= 0 and int(self.dist[nxt]) == here - 1:
                    out.append(d)
                    sid = nxt
                    break
            else:                                  # unreachable: dist is exact
                raise AssertionError("distance field has no descending press")
        return out


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class BoxFillSolver(BaseSolver):
    """`BaseSolver` for ps:box_fill, driving the real PuzzleScript adapter.

    Every hook reads the LIVE engine grid, and the field covers the entire
    reachable space, so replan-mode recovery works end to end: any detour lands
    on a state the table already has the answer for, and the only states that
    need a RESET are the genuinely dead ones.
    """

    game_id = "puzzlescript_box_fill"

    supports_recovery = True
    recovery_mode = "replan"
    #: Random pressing bricks a gated level fast -- on level 7 three quarters of
    #: the reachable states are dead and two wrong presses from the start reach
    #: one -- so the exploration prefix needs far more RESET headroom than the
    #: base default of 3. A failed recovery only costs a seed, but a reset is
    #: also perfectly good "I bricked it, start over" data, so err generous.
    max_resets = 8

    #: `solve_from` / `optimal_set_from` already emit the SCREEN press (they plan
    #: in engine space and invert the adapter's remap via `screen_action`), so
    #: the base must not convert them again. `drive` is overridden and hands the
    #: press to the adapter untouched, which applies its own forward remap.
    plans_in_screen_space = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #: level index -> Field. Seed-independent, so this is built once per
        #: process and reused by every later seed.
        self._fields: dict[int, Field] = {}

    # ── engine glue ─────────────────────────────────────────────────────────
    def make_game(self, seed: int):
        """Build the adapter through ``games/ps:box_fill/ps:box_fill.py`` -- the
        same entry point `game_envs` hands a live agent -- so the generator can
        never tape frames from a differently-configured adapter."""
        return _load_game_module(_GAME_ID).make_game(seed=seed)

    def num_levels(self, game) -> int:
        return game.n_levels

    def set_level(self, game, level_idx: int) -> None:
        super().set_level(game, level_idx)
        # Build the field HERE, at the level's INITIAL state: the reachable set
        # is defined relative to that state, and one built from a partly-filled
        # board would cover only the tail of the level and answer None for every
        # state behind it. This is also the one place the cache's premise -- that
        # the post-reset engine state is the same for every seed -- can be
        # CHECKED rather than assumed, so a level that ever gains a per-seed
        # layout rebuilds instead of quietly answering for the wrong board.
        level = Level(game._engine, game._game)
        field = self._fields.get(level_idx)
        if (field is None or field.start != level.start
                or field.level.geometry != level.geometry):
            self._fields[level_idx] = Field(level)

    def reset_level(self, game, level_idx: int, seed: int) -> None:
        """The adapter's ``set_level`` re-seats the level from its clean template
        (`PuzzleScriptAdapter._do_reset`), which is exactly the engine RESET, so
        the base implementation's ``_clean_levels`` branch is simply not needed
        here -- go straight through `set_level` (which also re-validates the
        cached field) and clear any terminal state."""
        self.set_level(game, level_idx)
        game._state = GameState.NOT_FINISHED

    def render(self, game) -> np.ndarray:
        return np.asarray(game._current_frame)

    def available_actions(self, game) -> list[int]:
        """The four presses; ACTION5 is excluded -- see `DIRS`."""
        return [1, 2, 3, 4]

    def drive(self, game, action: Action) -> DriveResult:
        fd = game.perform_action(
            ActionInput(id=_ID_TO_GAMEACTION[action.action_id]))
        frames = (np.asarray(fd.frame) if fd.frame
                  else np.asarray(game._current_frame)[None])
        solved = game._state == GameState.WIN
        # There is no lose condition -- no rule kills the creator and the board
        # cannot reach a terminal failure -- so GAME_OVER here only ever means
        # the adapter's own step budget ran out. An unwinnable board is reported
        # as a dead END by `solve_from` returning [], not as a death.
        return DriveResult(frames, solved,
                           not solved and game._state == GameState.GAME_OVER)

    # ── reading the live board ──────────────────────────────────────────────
    def _live_state(self, game, level_idx: int) -> tuple[Field, int] | None:
        """``(field, state)`` for the board as it stands, or None if the field
        cannot answer for it."""
        field = self._fields.get(level_idx)
        if field is None:
            return None
        n2i = game._game.obj_name_to_idx
        width = field.level.width
        mask = 0
        latched = 0
        for r, row in enumerate(game._engine.grid):
            for c, cell in enumerate(row):
                if n2i["box"] in cell:
                    mask |= 1 << (r * width + c)
                if n2i["gateup"] in cell:
                    latched = 1
        state = (mask << 1) | latched
        return (field, state) if field.knows(state) else None

    def _press(self, game, direction: str) -> Action:
        """Engine direction -> the SCREEN button that makes the adapter execute
        it, inverting the adapter's rotation + flip remap
        ([[ps-astar-generator-family]])."""
        return Action(int(screen_action(direction, game._rotation_k,
                                        game._hflip, game._vflip).value))

    # ── the two BaseSolver hooks ────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int) -> list:
        live = self._live_state(game, level_idx)
        if live is None:
            return []
        field, state = live
        presses = field.plan(state)
        if not presses:
            return []
        return [self._press(game, d) for d in presses]

    def optimal_set_from(self, game, level_idx: int, seed: int):
        live = self._live_state(game, level_idx)
        if live is None:
            return None
        field, state = live
        presses = field.optimal(state)
        if not presses:
            return None
        return [self._press(game, d) for d in presses]


# ---------------------------------------------------------------------------
# Differential test: `Level.step` vs the real PuzzleScript interpreter
# ---------------------------------------------------------------------------

def verify_model(trials: int = 150, steps: int = 120, seed: int = 99,
                 verbose: bool = True) -> int:
    """Cross-check `Level` against the engine on random walks. Returns the number
    of mismatches (0 is the only acceptable answer).

    `Level.step` is the ONE place this generator re-implements PuzzleScript
    instead of calling it, and it encodes four things that are easy to get subtly
    wrong: that boxes MOVE before the creator extrudes (so holding a direction
    lays a line instead of jamming), that a train of boxes shuffles along
    together only when its leader can move, that latching the gates DESTROYS any
    box standing on one, and that a latched gate kills the extrude as well as the
    walk. Random walking hits all four constantly -- the run below logs how many
    latch events and how many latches-with-a-box-on-a-gate it actually exercised,
    so a change that stops reaching them is visible rather than silently vacuous.

    Every press is compared on the full box set, the latch bit AND the win
    verdict, so a model that drifts by one cell fails on the next step.
    """
    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    engine, parsed = game._engine, game._game
    n2i = parsed.obj_name_to_idx
    rng = random.Random(seed)
    bad = latches = destroyed = wins = 0

    for lvl in range(game.n_levels):
        for _ in range(trials):
            game.set_level(lvl)
            level = Level(engine, parsed)
            state = level.start
            for _ in range(steps):
                d = rng.choice(DIRS)
                on_gate = bool((state >> 1) & level.gates) and not state & 1
                engine.step(d)
                state = level.step(state, d)
                if state & 1:
                    latches += 1
                    if on_gate:
                        destroyed += 1
                width = level.width
                eng_mask = 0
                eng_lat = 0
                for r, row in enumerate(engine.grid):
                    for c, cell in enumerate(row):
                        if n2i["box"] in cell:
                            eng_mask |= 1 << (r * width + c)
                        if n2i["gateup"] in cell:
                            eng_lat = 1
                eng_state = (eng_mask << 1) | eng_lat
                if eng_state != state or engine.check_win() != level.won(state):
                    bad += 1
                    if verbose and bad <= 5:
                        print(f"  MISMATCH level {lvl} press={d}: "
                              f"engine boxes={eng_mask:#x} latched={eng_lat} "
                              f"win={engine.check_win()} / model "
                              f"boxes={state >> 1:#x} latched={state & 1} "
                              f"win={level.won(state)}")
                    break
                if level.won(state):
                    wins += 1
                    break
    if verbose:
        print(f"verify_model: {game.n_levels * trials} walks, {bad} mismatches "
              f"({latches} latched presses, {destroyed} of them destroying a box "
              f"on a gate, {wins} walks that stumbled into a win)")
    return bad


if __name__ == "__main__":
    if "--verify-model" in sys.argv:
        sys.exit(1 if verify_model() else 0)
    sys.exit(BoxFillSolver.main())
