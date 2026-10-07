"""Generate Phase-1 training data for the PuzzleScript game ps:hungry_kitty
("Hungry Kitty", Adrian Arias -- a remake of lexaloffle's Neko).

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_hungry_kitty",
      "levels": [
        {
          "level_id": 0,
          "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
          "actions":      [a_0, a_1, ..., a_{T-1}], # length T
        },
        ...
      ]
    }

``actions[0]`` is the RESET that produced ``obs[0]``; ``actions[i]`` for i >= 1 is
the action that took the agent from ``obs[i-1]`` to ``obs[i]``. The recorded index
is the *screen* action (post rotation / flip remap), so replaying the recorded
actions against the same seed reproduces the recorded frames exactly. Every step
also carries the full set of equally-optimal presses.


THE GAME
========
A cat on a bare 8x7 (or 8x8) board with 5-9 fish on it. Win is ``No Fish``: eat
all of them. The whole rule set is two lines::

    [ > Player | ... | Fish ] -> [ | ... | Player ] sfx0
    [ > Player ] -> [ Player]

which read as: **a press makes the cat POUNCE to the nearest fish in that
direction, eating it; a press with no fish in that direction does nothing at
all.** The cat never takes a single step -- it teleports along the row or column
it is standing in, and it stops at the first fish it meets (the second rule
strips the force of any press the first one did not consume, so the cat cannot
drift). There is no wall, no hazard, and no ACTION5 rule, so the four direction
keys are the entire action space.

That makes the game a pure *ordering* puzzle over the fish, and it has three
consequences the expert below is built on:

1. **Every press that does anything eats exactly one fish**, so a level with
   ``n`` fish is won in exactly ``n`` presses or not at all -- there are no
   longer routes to compare, only orderings that work and orderings that do not.
   `Field` asserts this (``dist == popcount(mask)`` for every live state); it is
   a free, sharp check that the model has no transition the game does not have.

2. **A wrong order BRICKS the level, silently.** Eating the last fish of a row
   can leave the cat somewhere with nothing in any of the four lines, and then no
   press changes a pixel ever again: the win condition is simply unreachable and
   the game never reports anything. 72 of the 167 states reachable across the ten
   levels are dead this way -- 17 of level 8's 25 and 15 of level 9's 25. That is
   what the exploration prefix walks into, so `max_resets` is raised well above
   the base default (see `HungryKittySolver`).

3. **The optimal move is usually forced** -- 1.08 equally-optimal presses per live
   state, averaged over the 79 live states of all ten levels (only levels 2, 4, 5
   and 7 ever offer a choice, and never more than two). The learnable content is
   therefore perception, not search: which of the four rays the cat is standing on
   holds a fish, and which of those choices does not strand it.

Levels 0-9 are all winnable, in 5/5/6/6/7/7/7/7/7/9 presses.


EXPERT SOLVER
=============
Not a search over the interpreter: the mechanic is small enough to model exactly
and the entire reachable state space of a level is 6-34 states, so every level
gets an EXACT distance-to-win field instead of a heuristic.

A state is ``(cat cell, set of fish still on the board)``, packed into one int as
``(pos << n_fish) | fish_mask`` -- exact and canonical, since fish are never
created, never move, and nothing else on the board has state. `Field` enumerates
the reachable set forward from the level start, then reverse-BFSes from the
winning states for the exact distance-to-win everywhere. That one table answers
all three questions the harness asks:

  * ``plan(state)``   -- an optimal press sequence from *any* live state, which is
    what makes ``recovery_mode = "replan"`` honest: after an exploratory detour
    the cat is simply somewhere else in the same table.
  * ``optimal(state)`` -- every press whose successor is one step closer, i.e. the
    exact optimal-action SET for the step's training target, ties included.
  * ``dist(state) < 0`` -- the state is dead, so the only recovery is a RESET.

The field is seed-independent (levels are fixed ASCII maps; only the presentation
is augmented per seed), so it is built once per level in the first seed that
needs it and reused by every later one -- marginal cost per seed is a re-render.


VERIFICATION
============
`Level.step` is the one place this generator re-implements PuzzleScript rather
than calling it, and everything above rests on one claim the .txt does NOT state:
that the pounce stops at the NEAREST fish. It is not a property of the rule --
every fish further down the ray is an equally valid ellipsis binding, and it is
`PSEngine._match_still_valid` (the fix behind [[ps-ellipsis-clones-the-subject]])
that keeps the later ones from firing too. So ``--verify-model`` differentially
tests the model against the real interpreter, on the shipped levels AND on random
dense boards built to make multi-fish rays the common case, comparing the cat's
cell, the whole fish set and the win verdict after every single press. It reports
how many multi-fish rays and adjacent-fish pounces it actually exercised, so a
change that stops reaching them is visible rather than silently vacuous.

``--audit`` is the rendering check ([[ps-palette-collisions]]): it renders every
cell composition the game can show, at every board size the levels use, and
asserts they are pairwise distinct. Whole frames, not cropped cells -- the
renderer upscales a sub-64 board, so cell arithmetic lands in the wrong place
([[explod-rendering-fixes]]).

Usage::

    python solvers/generate_hungry_kitty_training.py --episodes 30 --out data/hk
    python solvers/generate_hungry_kitty_training.py --verify-model
    python solvers/generate_hungry_kitty_training.py --audit


PRESENTATION NOTES
==================
* No palette fix was needed. Background ``darkgreen`` -> 14, Fish
  ``lightblue darkblue`` -> 10 / 9, Player ``white gray lightblueblue`` ->
  0 / 2 / 10 (+9: the typo'd third token splits into ``lightblue`` + ``blue``,
  and the sprite only indexes the first three). The cat is a mostly-white 5x5
  and the fish a blue lozenge with a dark eye, so at the 8 px cells these boards
  render at they are distinct from each other and from the background --
  ``--audit`` asserts it.
* ``Hungry_Kitty`` joins ``PuzzleScriptAdapter._FLIP_GAMES``: gravity-free,
  screen-relative input, a direction-free win condition, and a mechanic ("pounce
  to the nearest fish along the pressed ray") that is invariant under every
  symmetry of the square -- there is only one moving object, so nothing can
  contest a cell and no rule-order chirality exists ([[ps-engine-render-gotchas]]
  7b). The cat sprite is asymmetric art but encodes no facing (the game has a
  single Player object and no direction states), so a mirrored frame is a frame
  the game could really hand out. Ten levels: 32 presentations -> 128.
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

GAME_NAME = "Hungry_Kitty"
_GAME_ID = "ps:hungry_kitty"

#: The four presses. No rule mentions ACTION, so ACTION5 cannot change a pixel;
#: offering it to the exploration policy would only spend budget on a guaranteed
#: null transition.
DIRS = ("up", "down", "left", "right")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


# ---------------------------------------------------------------------------
# The mechanic, compiled per level
# ---------------------------------------------------------------------------
#
# STATE ENCODING. One integer, ``(pos << n_fish) | fish_mask``, where ``pos`` is
# ``row * width + col`` of the cat and bit ``i`` of ``fish_mask`` is set while the
# level's i-th fish (fish are numbered in row-major order of their START cells) is
# still on the board. Fish are never created and never move, and there is nothing
# else on the board, so this is an EXACT canonical state: two boards with the same
# integer admit exactly the same futures.

class Level:
    """The static geometry of one Hungry Kitty level, compiled into ray tables.

    A press is answered by walking the precomputed list of fish along that ray
    and taking the first one still present -- no scanning of the board, and no
    conditionals about geometry at all.
    """

    def __init__(self, engine, parsed) -> None:
        n2i = parsed.obj_name_to_idx
        fish_id, player_id = n2i["fish"], n2i["player"]
        self.height, self.width = engine.height, engine.width

        fish: list[tuple[int, int]] = []
        start_cell: tuple[int, int] | None = None
        for r, row in enumerate(engine.grid):
            for c, cell in enumerate(row):
                if fish_id in cell:
                    fish.append((r, c))
                if player_id in cell:
                    start_cell = (r, c)
        if start_cell is None:
            raise ValueError("level has no player")
        self.fish = tuple(fish)                     # bit i <-> self.fish[i]
        self.n = len(self.fish)
        self.mask_all = (1 << self.n) - 1
        self.bit_of = {cell: i for i, cell in enumerate(self.fish)}

        # rays[pos][dir] = the fish bits along that ray, NEAREST FIRST. Built for
        # every cell, not just the ones the cat can occupy: it is 4 short lists
        # per cell on a <= 64-cell board, and a table with no reachability
        # assumption in it cannot go stale when a detour puts the cat somewhere
        # the level start could not.
        self.rays: list[dict[str, tuple[int, ...]]] = []
        for pos in range(self.height * self.width):
            r0, c0 = divmod(pos, self.width)
            per_dir = {}
            for d, (dr, dc) in _DELTA.items():
                bits = []
                r, c = r0 + dr, c0 + dc
                while 0 <= r < self.height and 0 <= c < self.width:
                    if (r, c) in self.bit_of:
                        bits.append(self.bit_of[(r, c)])
                    r, c = r + dr, c + dc
                per_dir[d] = tuple(bits)
            self.rays.append(per_dir)

        self.start = (start_cell[0] * self.width + start_cell[1]) << self.n \
            | self.mask_all

    @property
    def geometry(self) -> tuple:
        """Everything the compiled tables depend on -- used to detect a level that
        has changed under a cached `Field` (see `HungryKittySolver.set_level`)."""
        return (self.height, self.width, self.fish)

    def step(self, state: int, direction: str) -> int:
        """The settled state after one press. A press with no fish left on that
        ray returns ``state`` unchanged: the second rule cancels the force, so
        the turn is a true no-op, not a step."""
        mask = state & self.mask_all
        for bit in self.rays[state >> self.n][direction]:
            if mask >> bit & 1:                     # the nearest SURVIVING fish
                r, c = self.fish[bit]
                return ((r * self.width + c) << self.n) | (mask & ~(1 << bit))
        return state

    def won(self, state: int) -> bool:
        return not state & self.mask_all


# ---------------------------------------------------------------------------
# Exact distance-to-win field over the whole reachable state space
# ---------------------------------------------------------------------------

class Field:
    """Forward-enumerate every state reachable from a level's start, then
    reverse-BFS from the wins for the exact distance-to-win of each.

    The spaces are 6-34 states, so this is exhaustive and instant, and it makes
    every answer below exact rather than heuristic: the plan is shortest, the
    optimal set is complete (no equally-good press is missed and none is invented),
    and "dead" means provably dead rather than "the search gave up".
    """

    def __init__(self, level: Level) -> None:
        self.level = level
        self.start = level.start

        # -- forward: the reachable set + its successor table ------------------
        ids: dict[int, int] = {level.start: 0}
        states: list[int] = [level.start]
        succ: list[list[int]] = []
        queue = deque([level.start])
        while queue:
            state = queue.popleft()
            row = []
            for d in DIRS:
                nxt = state if level.won(state) else level.step(state, d)
                if nxt not in ids:
                    ids[nxt] = len(states)
                    states.append(nxt)
                    queue.append(nxt)
                row.append(ids[nxt])
            # ``succ`` is indexed by state id, and ids are handed out in the order
            # states are discovered, so a row can only be appended for the state at
            # the head of the queue -- which is exactly the loop's invariant.
            succ.append(row)
        self.ids, self.states, self.succ = ids, states, succ

        # -- backward: exact distance to a win --------------------------------
        pred: list[list[int]] = [[] for _ in states]
        for sid, row in enumerate(succ):
            for nid in row:
                if nid != sid:                      # a no-op press is not an edge
                    pred[nid].append(sid)
        self.dist = [-1] * len(states)
        wins = deque()
        for sid, state in enumerate(states):
            if level.won(state):
                self.dist[sid] = 0
                wins.append(sid)
        while wins:
            sid = wins.popleft()
            for pid in pred[sid]:
                if self.dist[pid] < 0:
                    self.dist[pid] = self.dist[sid] + 1
                    wins.append(pid)

        # Every press that changes anything eats exactly one fish, so a live
        # state's distance to a win IS its fish count. Asserting it costs nothing
        # and pins the model's single claim about the game's cost structure: a
        # transition that removed two fish, or none, or added one, breaks it.
        for sid, state in enumerate(states):
            if self.dist[sid] >= 0:
                assert self.dist[sid] == bin(state & level.mask_all).count("1"), \
                    f"state {state:#x}: dist {self.dist[sid]} != fish count"

    def knows(self, state: int) -> bool:
        return state in self.ids

    def alive(self, state: int) -> bool:
        sid = self.ids.get(state)
        return sid is not None and self.dist[sid] >= 0

    def optimal(self, state: int) -> list[str]:
        """Every press that strictly closes the distance to a win. Empty at a win
        and at a dead state."""
        sid = self.ids.get(state)
        if sid is None or self.dist[sid] <= 0:
            return []
        here = self.dist[sid]
        return [d for i, d in enumerate(DIRS)
                if self.dist[self.succ[sid][i]] == here - 1]

    def plan(self, state: int) -> list[str] | None:
        """An optimal press sequence from ``state``; ``[]`` at a win, ``None`` if
        the state is dead or unknown."""
        sid = self.ids.get(state)
        if sid is None or self.dist[sid] < 0:
            return None
        out: list[str] = []
        while self.dist[sid] > 0:
            here = self.dist[sid]
            for i, d in enumerate(DIRS):
                nid = self.succ[sid][i]
                if self.dist[nid] == here - 1:
                    out.append(d)
                    sid = nid
                    break
            else:                                   # unreachable: dist is exact
                raise AssertionError("distance field has no descending press")
        return out


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class HungryKittySolver(BaseSolver):
    """`BaseSolver` for ps:hungry_kitty, driving the real PuzzleScript adapter.

    Every hook reads the LIVE engine grid and the field covers the level's whole
    reachable space, so replan-mode recovery works end to end: an exploratory
    detour lands on a state the table already has the answer for, and only the
    genuinely dead states need a RESET.
    """

    game_id = "puzzlescript_hungry_kitty"

    supports_recovery = True
    recovery_mode = "replan"
    #: Random pressing strands this game fast -- eating fish in the wrong order
    #: leaves the cat with nothing in any of its four lines, and 72 of the 187
    #: states reachable across the ten levels are that (17 of 25 on level 8, 15
    #: of 25 on level 9). A stranded prefix is perfectly good "I bricked it,
    #: start over" data, so give the recorder plenty of RESET headroom instead of
    #: discarding those episodes.
    max_resets = 8

    #: `solve_from` / `optimal_set_from` already emit the SCREEN press (they plan
    #: in engine space and invert the adapter's remap via `screen_action`), so the
    #: base must not convert them again. `drive` is overridden and hands the press
    #: to the adapter untouched, which applies its own forward remap.
    plans_in_screen_space = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #: level index -> Field. Seed-independent, so this is built once per
        #: process and reused by every later seed.
        self._fields: dict[int, Field] = {}

    # -- engine glue ---------------------------------------------------------
    def make_game(self, seed: int):
        """Build the adapter through ``games/ps:hungry_kitty/ps:hungry_kitty.py``
        -- the same entry point `game_envs` hands a live agent -- so the generator
        can never tape frames from a differently-configured adapter."""
        return _load_game_module(_GAME_ID).make_game(seed=seed)

    def num_levels(self, game) -> int:
        return game.n_levels

    def set_level(self, game, level_idx: int) -> None:
        super().set_level(game, level_idx)
        # Build the field HERE, at the level's INITIAL state: the reachable set is
        # defined relative to that state, and one built from a half-eaten board
        # would cover only the tail of the level and answer None for every state
        # behind it. This is also the one place the cache's premise -- that the
        # post-reset engine state is the same for every seed -- can be CHECKED
        # rather than assumed, so a level that ever gains a per-seed layout
        # rebuilds instead of quietly answering for the wrong board.
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
        # There is no lose condition -- nothing in the game can remove the cat --
        # so GAME_OVER here only ever means the adapter's own step budget ran out.
        # A stranded board is reported as a dead END by `solve_from` returning [],
        # not as a death.
        return DriveResult(frames, solved,
                           not solved and game._state == GameState.GAME_OVER)

    # -- reading the live board ----------------------------------------------
    def _live_state(self, game, level_idx: int) -> tuple[Field, int] | None:
        """``(field, state)`` for the board as it stands, or None if the field
        cannot answer for it."""
        field = self._fields.get(level_idx)
        if field is None:
            return None
        level = field.level
        n2i = game._game.obj_name_to_idx
        fish_id, player_id = n2i["fish"], n2i["player"]
        mask = 0
        pos = None
        for r, row in enumerate(game._engine.grid):
            for c, cell in enumerate(row):
                if fish_id in cell:
                    bit = level.bit_of.get((r, c))
                    if bit is None:                 # a fish the level never had
                        return None
                    mask |= 1 << bit
                if player_id in cell:
                    pos = r * level.width + c
        if pos is None:                             # no cat: not a state we model
            return None
        state = (pos << level.n) | mask
        return (field, state) if field.knows(state) else None

    def _press(self, game, direction: str) -> Action:
        """Engine direction -> the SCREEN button that makes the adapter execute
        it, inverting the adapter's rotation + flip remap
        ([[ps-astar-generator-family]])."""
        return Action(int(screen_action(direction, game._rotation_k,
                                        game._hflip, game._vflip).value))

    # -- the two BaseSolver hooks --------------------------------------------
    def solve_from(self, game, level_idx: int, seed: int) -> list:
        live = self._live_state(game, level_idx)
        if live is None:
            return []
        field, state = live
        presses = field.plan(state)
        if not presses:                             # dead, or already won
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

def _read_board(engine, fish_id: int, player_id: int):
    """``(cat cell, frozenset of fish cells)`` straight off the engine grid."""
    cat = None
    fish = set()
    for r, row in enumerate(engine.grid):
        for c, cell in enumerate(row):
            if player_id in cell:
                cat = (r, c)
            if fish_id in cell:
                fish.add((r, c))
    return cat, frozenset(fish)


def _seat_board(engine, bg_id: int, fish_id: int, player_id: int,
                cat, fish) -> None:
    """Overwrite the engine grid with an arbitrary board (used by the fuzz to
    reach configurations the ten shipped levels never present)."""
    engine.grid = [[{bg_id} for _ in range(engine.width)]
                   for _ in range(engine.height)]
    for (r, c) in fish:
        engine.grid[r][c].add(fish_id)
    engine.grid[cat[0]][cat[1]].add(player_id)
    engine._position_index_dirty = True
    engine._rule_noop_cache.clear()


def verify_model(walks: int = 200, steps: int = 40, boards: int = 2000,
                 seed: int = 99, verbose: bool = True) -> int:
    """Cross-check `Level` against the real interpreter. Returns the number of
    mismatches (0 is the only acceptable answer).

    `Level.step` encodes ONE claim, and it is not written down anywhere in the
    .txt: the cat pounces to the NEAREST fish along the pressed ray. Every fish
    further along that ray is an equally valid binding of the rule's ellipsis, and
    what stops those bindings from firing as well is an engine detail
    ([[ps-ellipsis-clones-the-subject]]) -- so it is measured here, not assumed.
    The second claim, that a press with no fish on its ray is a complete no-op, is
    checked by the same comparison.

    Two populations, because the shipped levels are sparse and their reachable
    spaces are tiny (6-34 states, exhausted in a handful of walks):

      * random walks on all ten levels, which covers what the corpus will contain;
      * random DENSE boards (up to a third of the cells holding fish), which is
        what makes multi-fish rays and adjacent fish the common case rather than a
        rarity -- the two configurations the "nearest" claim actually turns on.

    The cat's cell, the whole fish set and the win verdict are compared after
    every press, so a model that drifts by one cell fails on the next step.
    """
    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    engine, parsed = game._engine, game._game
    n2i = parsed.obj_name_to_idx
    bg_id, fish_id, player_id = n2i["background"], n2i["fish"], n2i["player"]
    rng = random.Random(seed)
    bad = presses = noops = far_rays = adjacent = wins = stranded = 0

    def _run(level: Level, state: int, n_steps: int) -> None:
        """Play one random walk, comparing model and interpreter every press."""
        nonlocal bad, presses, noops, far_rays, adjacent, wins, stranded
        for _ in range(n_steps):
            # Mostly press a direction the MODEL thinks has a fish on it. Uniform
            # walking is 83% no-ops (the cat strands itself early and then every
            # press is a null turn), which compares the two implementations on
            # the one case they cannot disagree about. The remaining fifth stays
            # uniform, so a direction the model wrongly believes empty is still
            # pressed -- the bias narrows what is tested most, it does not close
            # anything off.
            live_dirs = [d for d in DIRS if level.step(state, d) != state]
            d = (rng.choice(live_dirs) if live_dirs and rng.random() < 0.8
                 else rng.choice(DIRS))
            mask = state & level.mask_all
            stuck = not live_dirs
            ray = [b for b in level.rays[state >> level.n][d] if mask >> b & 1]
            if len(ray) > 1:
                far_rays += 1                    # a genuine nearest-vs-farther test
            if ray:
                r, c = level.fish[ray[0]]
                pr, pc = divmod(state >> level.n, level.width)
                if abs(r - pr) + abs(c - pc) == 1:
                    adjacent += 1                # the zero-cell ellipsis binding
            engine.step(d)
            nxt = level.step(state, d)
            presses += 1
            if nxt == state:
                noops += 1
            cat, fish = _read_board(engine, fish_id, player_id)
            want_cat = divmod(nxt >> level.n, level.width)
            want_fish = frozenset(level.fish[b] for b in range(level.n)
                                  if nxt >> b & 1)
            if (cat != want_cat or fish != want_fish
                    or engine.check_win() != level.won(nxt)):
                bad += 1
                if verbose and bad <= 5:
                    print(f"  MISMATCH press={d}: engine cat={cat} "
                          f"fish={sorted(fish)} win={engine.check_win()} / model "
                          f"cat={want_cat} fish={sorted(want_fish)} "
                          f"win={level.won(nxt)}")
                return
            state = nxt
            if level.won(state):
                wins += 1
                return
            if stuck:
                # The cat has nothing on any of its four rays, so the rest of the
                # walk is the same null turn over and over -- one comparison of it
                # is the test, the other 39 are filler. End the walk here (the
                # press above was that one comparison).
                stranded += 1
                return

    for lvl in range(game.n_levels):
        for _ in range(walks):
            game.set_level(lvl)
            level = Level(engine, parsed)
            _run(level, level.start, steps)

    game.set_level(0)
    height, width = engine.height, engine.width
    for _ in range(boards):
        cells = [(r, c) for r in range(height) for c in range(width)]
        rng.shuffle(cells)
        n_fish = rng.randint(1, max(1, (height * width) // 3))
        cat, fish = cells[0], set(cells[1:1 + n_fish])
        game.set_level(0)
        _seat_board(engine, bg_id, fish_id, player_id, cat, fish)
        level = Level(engine, parsed)
        _run(level, level.start, steps)

    if verbose:
        print(f"verify_model: {game.n_levels * walks} level walks + {boards} "
              f"random boards, {presses} presses, {bad} mismatches "
              f"({far_rays} presses onto a ray holding 2+ fish, {adjacent} onto "
              f"an adjacent fish, {noops} no-op presses, {wins} walks that "
              f"reached a win, {stranded} that stranded the cat)")
    return bad


# ---------------------------------------------------------------------------
# Rendering audit
# ---------------------------------------------------------------------------

def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every board size in use.

    There are only three -- the board is bare and Fish and Player share a
    collision layer, so nothing ever stacks -- but the win frame IS "the cat
    alone on the background", so the cat having to differ from both the fish it
    ate and the ground it stands on is the whole readability requirement of this
    game.

    Whole frames, not cropped cells: `_render_frame` upscales a sub-64 board to
    fill the frame, so ``cell_px * r`` arithmetic lands in the wrong cell
    ([[explod-rendering-fixes]])."""
    game = HungryKittySolver().make_game(0)
    eng, parsed = game._engine, game._game
    idx = parsed.obj_name_to_idx
    comps = {"background": (), "fish": ("fish",), "cat": ("player",)}

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
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Field report
# ---------------------------------------------------------------------------

def _report() -> int:
    """Print each level's exact field: plan length, reachable / dead states, and
    how much of a choice the expert ever has."""
    solver = HungryKittySolver()
    game = solver.make_game(0)
    total = dead_total = 0
    for level in range(game.n_levels):
        solver.set_level(game, level)
        field = solver._fields[level]
        plan = field.plan(field.start)
        live = [sid for sid, d in enumerate(field.dist) if d > 0]
        opts = [len(field.optimal(field.states[sid])) for sid in live]
        dead = sum(1 for d in field.dist if d < 0)
        total += len(field.states)
        dead_total += dead
        print(f"level {level}: {field.level.n} fish, plan {len(plan)} presses, "
              f"{len(field.states)} reachable states ({dead} dead), "
              f"optimal-set size {min(opts)}-{max(opts)} "
              f"(mean {sum(opts) / len(opts):.2f}), plan {' '.join(plan)}")
    print(f"total: {total} reachable states, {dead_total} dead")
    return 0


if __name__ == "__main__":
    if "--verify-model" in sys.argv:
        sys.exit(1 if verify_model() else 0)
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--report" in sys.argv:
        sys.exit(_report())
    sys.exit(HungryKittySolver.main())
