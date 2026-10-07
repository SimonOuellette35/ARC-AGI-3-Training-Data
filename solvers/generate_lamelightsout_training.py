"""Generate Phase-1 training data for the PuzzleScript game ps:lamelightsout.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema::

    {
      "game_id": "puzzlescript_lamelightsout",
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
LameLightsOut (Matthew VanDevander) is Lights Out with a WALKING cursor. The
board is a full rectangle of lights, each White (on) or Grey (off); the red
cursor moves one cell per arrow press, and ACTION flips the light it stands on
plus the (clipped) four orthogonal neighbours -- a PLUS. Win is ``No LightOff``.

The whole rule set is seven lines, and the interesting thing about them is that
not one carries a direction::

    [Action Player]          -> [Player RuleApplier]
    [RuleApplier | LightOff] -> [RuleApplier | TempOn]
    [RuleApplier | LightOn]  -> [RuleApplier | TempOff]
    [RuleApplier LightOn]    -> [TempOff]
    [RuleApplier LightOff]   -> [TempOn]
    [TempOn]  -> [LightOn]
    [TempOff] -> [LightOff]

A bare ``|`` pattern is expanded by the interpreter to all four directions, so
the two neighbour rules ARE the arms of the plus. The Temp objects are the
double-buffer that makes the flip simultaneous: they live on their own collision
layer, so an arm's new state is written beside the old light rather than over
it, and the two same-cell rules then consume the RuleApplier (which is why a
press leaves nothing behind and the board is quiescent between presses).

That the mechanic is direction-free at every level -- rules, sprites and win
condition alike -- is what puts the game in `PuzzleScriptAdapter._FLIP_GAMES`;
the argument is written out there.


THE COST MODEL (why this is not just Lights Out)
================================================
Classic Lights Out is pure GF(2) linear algebra: presses commute, each is its
own inverse, so a solution is a SET of cells and the only question is which set
is smallest. Here the cursor has to WALK to each cell it presses, so the cost of
a solution set S is ``|S| + (length of the shortest walk visiting all of S)`` --
a routing problem stapled onto the algebra, and the walk is the bigger half of
it on six of the sixteen levels -- level 5's optimal plan is 3 toggles and 6
steps, level 12's is 6 and 10.

On the shipped boards the two halves never actually fight -- the optimal plan
always uses a minimum-weight click set, because the alternative cosets here are
several clicks larger and no walk is that expensive. What routing decides is
WHICH minimum-weight set: where there is more than one, they differ in walking
cost by up to three presses (level 2: 5 vs 7; level 11: 15 vs 16 vs 18), and the
expert takes the cheapest -- which is not a thing the linear system can express.

So the expert does NOT solve the linear system. It solves the real game.


THE EXPERT: ONE EXACT DISTANCE FIELD PER LEVEL
==============================================
The whole state is ``(which lights are on, where the cursor is)`` -- there is no
hidden bookkeeping between presses, and the board is a bare rectangle with no
walls -- so a board of ``n`` cells has at most ``2^n * n`` states. Every shipped
board is 16 cells or fewer, and the reachable set is a coset of the toggle
matrix's column space, so it is smaller still (65536 states on the 4x4, 57344 on
the 2x7). `Field` enumerates that set forward from the level start, then runs a
backward BFS from every winning state to get the EXACT number of presses still
needed from anywhere in it.

That single table answers all three questions the recorder asks, in O(1) each
and with no search at runtime:

  * `Field.plan` -- an optimal press sequence from the LIVE state, so
    ``recovery_mode = "replan"``: an exploratory detour or a perturbation burst
    just lands on another state the table already knows the answer for.
  * `Field.optimal` -- the FULL set of equally-optimal presses, which is the
    training target and the source of trajectory diversity. Ties are dense here
    (about 40% of live states have two or more optimal presses: every walk
    between two click cells can interleave its two axes freely, and a plus that
    has to be pressed twice can be pressed at either end of the route), so
    stochastic-optimal sampling makes each seed a genuinely different perfect
    playthrough rather than a re-rendering of one route.
  * dead states -- there are NONE, and that is a fact about this game rather
    than an accident. Every action is invertible (a walk retraces, a toggle is
    its own inverse), so the state graph is undirected and every state in a
    level's component can reach every other one, the winning one included. The
    field still carries ``dist = -1`` handling because `Field` is built from the
    level START and an off-component board would be genuinely unwinnable, but
    ``--report`` asserts the count is zero on all sixteen levels: no press an
    agent can make will ever brick this game, which is exactly why the recovery
    mode is replan and RESET is only ever the last resort.

`Level.step` is the one place this generator re-implements PuzzleScript instead
of calling it, so it is differentially tested against the real interpreter --
`verify_model` (``--verify-model``) replays random walks on all sixteen levels
and compares the light mask, the cursor cell AND the win verdict after every
single press.


PRESENTATION FIXES MADE FOR THIS GAME
=====================================
* The lights shipped as FLAT squares, so two adjacent cells in the same state
  rendered as one bar with no boundary in it -- and the 2x2 level, which starts
  all-dark and fills the whole 64x64 frame, was a single uniform grey field with
  nothing in it but the cursor. Both lights are now a solid centre inside a
  one-pixel DarkGrey frame, which draws a gridline between every pair of cells.
  The state colours (White on, Grey off) are untouched. See the sprite contract
  at the head of data/puzzlescript_games/LameLightsOut.txt, and `_audit`, which
  asserts that all four cell compositions render distinctly at every board size
  in use.
* Twelve levels were authored on top of the shipped four. The originals are 1x3,
  1x4, 1x5 and 2x2 -- the biggest has five cells -- which is a game to memorise
  rather than a mechanic to learn. The new boards run 1x6 up to 4x4 and 8 up to
  23 presses; every one is scrambled backwards from the all-on win (so it is
  solvable by construction) and then proved solvable AND shortest by the field
  here. See the note in the game file's LEVELS block.
"""
from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameState                          # noqa: E402
from adapters.puzzlescript_adapter import _render_frame               # noqa: E402
from solvers.base_solver import (                                     # noqa: E402
    BaseSolver, DriveResult, _ID_TO_GAMEACTION)
from solvers.common.ps_astar import _load_game_module, screen_action  # noqa: E402
from utils.explore import Action                                      # noqa: E402

GAME_NAME = "LameLightsOut"
_GAME_ID = "ps:lamelightsout"

#: The five presses: walk the cursor, or flip the plus under it. Unlike most of
#: the ps: generators here ACTION is the *point* of the game, so it is in the
#: search and in the exploration action set alike.
DIRS = ("up", "down", "left", "right", "action")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


# ---------------------------------------------------------------------------
# The mechanic, compiled per level
# ---------------------------------------------------------------------------
#
# STATE ENCODING. A state is one integer: ``lights * n_cells + cursor``, where
# ``lights`` has bit ``r * width + c`` set for every cell whose light is ON and
# ``cursor`` is the cell index the player stands on. Nothing else exists between
# presses -- RuleApplier and the two Temp objects are created and consumed
# inside a single ACTION (see the module docstring) -- so this is an EXACT
# canonical state: two boards with the same integer admit exactly the same
# futures.

class Level:
    """The static geometry of one LameLightsOut board, compiled into tables.

    The board is a bare rectangle: every cell holds a light and nothing blocks
    the cursor, so the only per-level facts are its shape, the plus each cell
    toggles and where a walk in each direction lands.
    """

    def __init__(self, engine, parsed) -> None:
        n2i = parsed.obj_name_to_idx
        self.height, self.width = engine.height, engine.width
        self.n_cells = self.height * self.width
        self.full = (1 << self.n_cells) - 1

        lights = 0
        cursor = None
        for r, row in enumerate(engine.grid):
            for c, cell in enumerate(row):
                i = r * self.width + c
                if n2i["lighton"] in cell:
                    lights |= 1 << i
                elif n2i["lightoff"] not in cell:
                    raise ValueError(
                        f"LameLightsOut cell ({r},{c}) holds no light -- this "
                        f"model assumes a full rectangle of lights (a bare cell "
                        f"would also strand a RuleApplier, since the two rules "
                        f"that consume it both require a light underneath)")
                if n2i["player"] in cell:
                    cursor = i
        if cursor is None:
            raise ValueError("LameLightsOut level with no Player")

        #: cell -> the light mask ACTION flips when the cursor stands there:
        #: the cell itself plus its orthogonal neighbours, clipped at the edges.
        self.toggle = []
        for i in range(self.n_cells):
            r, c = divmod(i, self.width)
            mask = 1 << i
            for dr, dc in _DELTA.values():
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    mask |= 1 << (nr * self.width + nc)
            self.toggle.append(mask)

        #: (direction, cell) -> the cell a walk lands on. Walking into the edge
        #: is a self-loop, exactly as the interpreter refuses the move.
        self.walk = {}
        for d, (dr, dc) in _DELTA.items():
            dest = []
            for i in range(self.n_cells):
                r, c = divmod(i, self.width)
                nr, nc = r + dr, c + dc
                dest.append(nr * self.width + nc
                            if 0 <= nr < self.height and 0 <= nc < self.width
                            else i)
            self.walk[d] = dest

        #: The board AS COMPILED, in the state encoding above. Compiled at a
        #: level's initial state this is the state a `Field` is rooted at.
        self.start = lights * self.n_cells + cursor
        #: Everything the tables depend on. Two levels with the same signature
        #: are the same puzzle, which is what lets a `Field` be cached across
        #: seeds -- and what makes a layout that ever becomes seed-dependent
        #: show up as a cache miss instead of a wrong answer.
        self.geometry = (self.height, self.width)

    def step(self, state: int, d: str) -> int:
        """One press: walk the cursor, or flip the plus under it."""
        lights, cursor = divmod(state, self.n_cells)
        if d == "action":
            return (lights ^ self.toggle[cursor]) * self.n_cells + cursor
        return lights * self.n_cells + self.walk[d][cursor]

    def won(self, state: int) -> bool:
        """``No LightOff``: every cell is lit. The cursor is irrelevant -- it
        stands on a light rather than replacing one."""
        return state // self.n_cells == self.full


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
    #: board someone adds later that does not fit the premise that the whole
    #: space can be enumerated. The reachable set is ``2^rank * n_cells``, so a
    #: full-rank 16-cell board (2x8) would be 1048576 and a 5x5 would be 210
    #: million; every shipped level comes in under 66k. See the ceiling note in
    #: the game file's LEVELS block.
    state_cap: int = 2_000_000

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
                            f"LameLightsOut reachable set exceeded "
                            f"{self.state_cap} states -- this board does not "
                            f"fit the premise that the whole space can be "
                            f"enumerated")
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

class LameLightsOutSolver(BaseSolver):
    """`BaseSolver` for ps:lamelightsout, driving the real PuzzleScript adapter.

    Every hook reads the LIVE engine grid, and the field covers the entire
    reachable space, so replan-mode recovery works end to end. Nothing in this
    game is irreversible, so no detour can ever need the RESET -- see the "dead
    states" note in the module docstring.
    """

    game_id = "puzzlescript_lamelightsout"

    supports_recovery = True
    recovery_mode = "replan"

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
        """Build the adapter through ``games/ps:lamelightsout/ps:lamelightsout.py``
        -- the same entry point `game_envs` hands a live agent -- so the
        generator can never tape frames from a differently-configured adapter."""
        return _load_game_module(_GAME_ID).make_game(seed=seed)

    def num_levels(self, game) -> int:
        return game.n_levels

    def set_level(self, game, level_idx: int) -> None:
        super().set_level(game, level_idx)
        # Build the field HERE, at the level's INITIAL state: the reachable set
        # is defined relative to that state, and one built from a half-solved
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
        """All five: the four walks and ACTION5, which is the toggle."""
        return [1, 2, 3, 4, 5]

    def drive(self, game, action: Action) -> DriveResult:
        fd = game.perform_action(
            ActionInput(id=_ID_TO_GAMEACTION[action.action_id]))
        frames = (np.asarray(fd.frame) if fd.frame
                  else np.asarray(game._current_frame)[None])
        solved = game._state == GameState.WIN
        # There is no lose condition -- no rule removes the player and no board
        # state is unrecoverable -- so GAME_OVER here only ever means the
        # adapter's own step budget ran out.
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
        lights = 0
        cursor = None
        for r, row in enumerate(game._engine.grid):
            for c, cell in enumerate(row):
                if n2i["lighton"] in cell:
                    lights |= 1 << (r * width + c)
                if n2i["player"] in cell:
                    cursor = r * width + c
        if cursor is None:
            return None
        state = lights * field.level.n_cells + cursor
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

def verify_model(walks: int = 60, steps: int = 80, seed: int = 99,
                 verbose: bool = True) -> int:
    """Cross-check `Level` against the engine on random walks. Returns the number
    of mismatches (0 is the only acceptable answer).

    `Level.step` is the ONE place this generator re-implements PuzzleScript
    instead of calling it, and it encodes three things that are easy to get
    subtly wrong: that ACTION flips a PLUS *clipped* at the board edge rather
    than wrapping or reaching off-grid, that the flip is SIMULTANEOUS (the
    double-buffer through TempOn/TempOff means an arm is decided from the
    pre-press board, so two arms never see each other's new value), and that
    walking into the edge is a no-op rather than a wrapped move. Random walking
    hits all three constantly -- the run below logs how many presses were edge
    toggles, how many were refused walks and how many walks stumbled into a win,
    so a change that stops reaching them is visible rather than silently vacuous.

    Every press is compared on the full light mask, the cursor cell AND the win
    verdict, so a model that drifts by one cell fails on the next step.
    """
    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    engine, parsed = game._engine, game._game
    n2i = parsed.obj_name_to_idx
    rng = random.Random(seed)
    bad = edge_toggles = refused = wins = presses = 0

    for lvl in range(game.n_levels):
        for _ in range(walks):
            game.set_level(lvl)
            level = Level(engine, parsed)
            state = level.start
            for _ in range(steps):
                d = rng.choice(DIRS)
                cursor = state % level.n_cells
                if d == "action":
                    # A plus is 5 cells only in the interior; anywhere else the
                    # engine's `|` match simply finds no neighbour to flip.
                    if bin(level.toggle[cursor]).count("1") < 5:
                        edge_toggles += 1
                elif level.walk[d][cursor] == cursor:
                    refused += 1
                engine.step(d)
                state = level.step(state, d)
                presses += 1

                eng_lights = 0
                eng_cursor = None
                for r, row in enumerate(engine.grid):
                    for c, cell in enumerate(row):
                        if n2i["lighton"] in cell:
                            eng_lights |= 1 << (r * level.width + c)
                        if n2i["player"] in cell:
                            eng_cursor = r * level.width + c
                eng_state = eng_lights * level.n_cells + eng_cursor
                if eng_state != state or engine.check_win() != level.won(state):
                    bad += 1
                    if verbose and bad <= 5:
                        print(f"  MISMATCH level {lvl} press={d}: engine "
                              f"lights={eng_lights:#x} cursor={eng_cursor} "
                              f"win={engine.check_win()} / model "
                              f"lights={state // level.n_cells:#x} "
                              f"cursor={state % level.n_cells} "
                              f"win={level.won(state)}")
                    break
                if level.won(state):
                    wins += 1
                    break
    if verbose:
        print(f"verify_model: {game.n_levels * walks} walks, {presses} presses, "
              f"{bad} mismatches ({edge_toggles} toggles on a clipped plus, "
              f"{refused} walks refused at the edge, {wins} walks that stumbled "
              f"into a win)")
    return bad


# ---------------------------------------------------------------------------
# Rendering audit
# ---------------------------------------------------------------------------

def _audit() -> int:
    """Assert every cell COMPOSITION renders distinctly at every board size in
    use, and that a board of identical cells still shows its gridlines.

    There are exactly four compositions -- a light is on or off, and the cursor
    is on it or not -- and all four matter: the win frame is "every cell lit",
    the cursor has to be findable on both light states, and the state UNDER the
    cursor has to stay readable (which is why the cursor is four corner pixels
    and not a filled square). The gridline check is the one the shipped art
    failed: flat squares made a run of same-state cells one featureless bar,
    so a 2x2 all-dark board was a uniform grey frame.

    Whole frames, not cropped cells: `_render_frame` upscales a sub-64 board to
    fill the frame, so ``cell_px * r`` arithmetic lands in the wrong cell
    ([[explod-rendering-fixes]])."""
    game = LameLightsOutSolver().make_game(0)
    eng, parsed = game._engine, game._game
    idx = parsed.obj_name_to_idx
    comps = {"off": ("lightoff",), "on": ("lighton",),
             "cursor-on-off": ("lightoff", "player"),
             "cursor-on-on": ("lighton", "player")}

    sizes: dict[tuple[int, int], list[int]] = {}
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
            shots[name] = np.asarray(_render_frame(eng, parsed))
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        # A board of identical cells must still be legible AS a board, which is
        # what the shipped flat squares failed. The gridline is the light
        # sprite's one-pixel frame, so the test is that BOTH of a light's
        # colours -- frame and interior -- survive to the screen. That is a real
        # risk and not a formality: the renderer resamples a 5x5 sprite down to
        # cell_px, and a one-pixel edge is exactly the feature that vanishes
        # when the sampler skips a row ([[ps-engine-render-gotchas]]).
        flat = [n for n in ("off", "on")
                if not set(parsed.objects[f"light{n}"].colors)
                <= set(np.unique(shots[n]).tolist())]
        bad += len(clashes) + len(flat)
        note = ("OK" if not clashes and not flat
                else " ".join(filter(None, [
                    f"IDENTICAL {clashes}" if clashes else "",
                    f"NO GRIDLINE {flat}" if flat else ""])))
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {note}")
    print("audit clean" if not bad else f"AUDIT FAILED: {bad} problems")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Field report
# ---------------------------------------------------------------------------

def _report() -> int:
    """Print each level's exact field: plan length, reachable / dead states, and
    how much of a choice the expert ever has.

    The dead count is an ASSERTION as much as a statistic -- every press in this
    game is invertible, so a level with any dead state at all would mean the
    model has an irreversible transition the mechanic does not."""
    solver = LameLightsOutSolver()
    game = solver.make_game(0)
    total = dead_total = 0
    for level in range(game.n_levels):
        solver.set_level(game, level)
        field = solver._fields[level]
        lvl = field.level
        plan = field.plan(field.start)
        live = [sid for sid, d in enumerate(field.dist) if d > 0]
        opts = [len(field.optimal(field.states[sid])) for sid in live]
        branch = sum(1 for k in opts if k > 1)
        dead = sum(1 for d in field.dist if d < 0)
        total += len(field.states)
        dead_total += dead
        print(f"level {level:2d}: {lvl.height}x{lvl.width}, plan "
              f"{len(plan):2d} presses, {len(field.states):6d} reachable states "
              f"({dead} dead), {branch}/{len(live)} states with a choice "
              f"(optimal-set size {min(opts)}-{max(opts)}, mean "
              f"{sum(opts) / len(opts):.2f})")
    print(f"total: {total} reachable states, {dead_total} dead")
    if dead_total:
        print("REPORT FAILED: every press in this game is invertible, so no "
              "reachable state can be dead")
    return 1 if dead_total else 0


if __name__ == "__main__":
    if "--verify-model" in sys.argv:
        sys.exit(1 if verify_model() else 0)
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--report" in sys.argv:
        sys.exit(_report())
    sys.exit(LameLightsOutSolver.main())
