"""Generate Phase-1 training data for the PuzzleScript game
ps:escape_the_void_full ("Escape the Void Full" by Croubble -- a sokoban with
two crate colours and opposite verbs: the orange ones you PUSH, the yellow ones
you PULL, and you can never do the other thing to either).

The harness -- the rotation contract, the trajectory recorder, the
RESET-recovery prefix and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other ps: generators. This file is
the game-specific part: the mechanic notes, a native model of the interpreter,
the exact distance FIELD computed over it, and the optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_escape_the_void_full",
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
presented view, so replaying the recorded actions reproduces the recorded frames
exactly. Every expert step also carries the full set of equally-optimal presses.

The game
--------
Two rules, and they are mirror images of each other::

    [ < Player | Crate1 ] -> [ < Player | < Crate1 ]   (yellow: PULL only)
    [ > Player | Crate2 ] -> [ > Player | > Crate2 ]   (orange: PUSH only)

Win is ``All Target on Cratey``, where ``Cratey = Crate1 or Crate2``: every red
square has to end up under a crate of *either* colour. Every level ships exactly
as many crates as it has targets, so the win is a perfect matching -- no crate
is spare, and burying one anywhere it cannot reach a target loses the level.

  * **Orange is a normal sokoban crate.** Step into it and it slides one square
    ahead of you, if the square beyond is empty.

  * **Yellow is the opposite piece.** ``<`` is "moving against the rule's
    direction", so the pattern is *player with a Crate1 behind it, walking
    away*: the crate is towed into the square the player just left. Walking
    *into* a yellow crate does nothing at all -- Player, Crate1, Crate2 and
    Water share one collision layer, so the press is simply refused. A yellow
    crate can therefore only ever be dragged **towards** the player, which means
    the player has to get **past** it first and then retreat.

  * **A WALK IS NOT FREE, and the walk graph is DIRECTED.** This is the trap
    (the same one ps:dreaming_of_strawberries has, from the same ``<`` rule):
    there is no such thing as repositioning the player without side effects,
    because any step taken with a yellow crate immediately behind you drags that
    crate one square. Stepping *toward* a yellow crate is free; stepping away
    from it is not, and is not undoable -- you cannot push it back.

  * **Both rules can fire on one press.** A player with a yellow crate behind
    and an orange crate ahead moves all three pieces at once, and if *either*
    crate is blocked the whole turn is refused and nothing moves at all
    (verified against the interpreter, not assumed -- see ``--selfcheck``).

  * ACTION5 is bound to nothing, so the four arrows are the whole action space
    (`VoidExpert.directions` drops it). ``Wall`` is declared but never placed:
    the ``#`` in every level is ``Water``, which is a plain blocker.

Solver
------
There is no A* here and no heuristic. Because a yellow crate makes plain walking
state-changing, the walk-to-and-push MACRO abstraction the rest of this family
uses (`PSPushExpert`) does not apply -- a macro's walk would silently drag
crates around. But the boards are small (12-30 open squares, 1-6 crates), and
the whole reachable state space turns out to fit in memory: the biggest level
closes at 30k states, and every level in the game closes in under a tenth of a
second.

So `Board` builds the exact field instead, which is strictly better than a
search:

  1. forward BFS from the level start over ``(player, yellow mask, orange
     mask)``, treating a won state as terminal, giving the reachable closure;
  2. reverse BFS from every won state in that closure, giving ``h(s)``, the
     EXACT number of presses from ``s`` to a win (``INF`` for the states that
     have been ruined -- a crate dragged somewhere no target can be reached
     from);
  3. the plan is a greedy descent of ``h`` from the start, and the optimal
     press set at each step is *every* press ``d`` with ``h(step(s, d)) ==
     h(s) - 1``.

Both the plan and the labels are therefore exact rather than approximate: no
weight to tune, no admissibility argument to make, no tie-set oracle that
re-solves (the field already answers every "is this press also shortest?"
question for free). A press that changes nothing scores ``h(s)``, not
``h(s) - 1``, so refused presses drop out of the sets automatically.

The levels
----------
All 17 shipped levels solve, 655 presses in total, ~0.3s of one-time search for
the whole game, cached to ``data/escape_the_void_full_plans.json`` so
`parallelize_generator`'s shards do not each re-derive it. Delete the file to
re-derive.

    level  size    open  crates(y/o)  targets  plan  ties  states
      0     9x7     16      0/1          1      24     0      225   "Push" tutorial
      1     9x7     19      1/0          1      24     0      120   "Pull" tutorial
      2     7x7     16      0/2          2      21     0      769
      3     9x6     17      2/0          2      27     1      196
      4     8x9     24      0/2          2      67     1     5897
      5     9x10    30      2/0          2      70     0      879
      6     8x6     14      0/2          2      33     0      629
      7     8x6     16      2/0          2      32     0      192
      8     7x10    25      0/3          3      82     1    17922
      9     8x8     18      2/0          2      41     1      496
     10     6x6     12      1/1          2       8     0       76
     11     7x8     16      4/1          5      19     0     2341
     12     6x7     16      3/1          4      30     0     4135
     13     9x7     23      1/2          3      55     2    16762
     14     9x6     17      1/1          2      32     0      362
     15    11x6     15      1/2          3      28     1      669
     16     9x8     25      5/1          6      62     0    30244   the finale

``ties`` is how many of the plan's presses have more than one equally-shortest
alternative, and it is genuinely almost always zero: these boards are corridors
and pockets where a 24-press solution lives on a 16-square level, so the shortest
route is usually unique down to the press. That number is measured off the exact
field, not guessed -- where a tie does exist it is labelled, and where none does,
none is invented.

The art fixes
-------------
Two, both in ``data/puzzlescript_games/Escape_the_Void_Full_.txt``, both of the
recurring kinds (see the palette/render notes this tree keeps):

  * **Target was a hollow RING, and every body that can stand on one is opaque
    exactly there.** The player sprite covers all eight ring pixels, so a player
    standing on a target rendered pixel-identical to a player standing on bare
    floor -- at every cell size in the game. Target is now a solid square, which
    shows through the player's transparent corners and through each crate's open
    middle.
  * **The player was drawn in the walls' colour.** Its body was ``White`` and
    ``Water`` -- which is what every ``#`` on every board is -- is ``#FAFAFA``,
    i.e. the same ARC index 0. The body is now ``Green``, and the one head pixel
    ``Purple`` rather than ``Red`` so that red means "target" and nothing else.

``--audit`` is the regression test: every cell COMPOSITION the game can show
(floor, wall, target, each crate, each crate on a target, player, player on a
target) rendered at every cell size the 17 boards use, asserted pairwise
distinct. No rule, sprite shape or level was touched by either edit.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (``rotation_k`` in {0,1,2,3}) plus an
independent horizontal and vertical flip, each with the matching directional
action remap (`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the
board's 8-element symmetry group. The game is gravity-free, both its rules are
stated with the relative ``<`` / ``>`` forces, its win condition names no
direction and its input is screen-relative, so a flip is an exact symmetry of
the mechanic. There is deliberately no colour augmentation: yellow-vs-orange
*is* the mechanic (which verb the crate answers to), and crate-on-target is read
as the target's red showing through a crate's open middle -- a flattening
recolor would erase both. 17 levels x 16 presentations = 272.

The expert plan is therefore seed-independent: solved once per level, cached,
and replayed per seed with that seed's remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_escape_the_void_full_training.py --episodes 200 \
        --out data/training_multi_level/escape_the_void_full

    python solvers/generate_escape_the_void_full_training.py --plans      # levels
    python solvers/generate_escape_the_void_full_training.py --verify     # replay
    python solvers/generate_escape_the_void_full_training.py --ties       # labels
    python solvers/generate_escape_the_void_full_training.py --audit      # render
    python solvers/generate_escape_the_void_full_training.py --selfcheck  # fuzz

``--selfcheck`` fuzzes `Board` against the interpreter (305k random presses, 0
divergences), ``--verify`` replays every plan through the interpreter to a WIN,
``--ties`` re-derives all 655 optimal-action sets by an independent forward BFS
and compares them with the field's, ``--audit`` is the rendering regression test
and ``--plans`` prints the table above.
"""

from __future__ import annotations

import random
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import PuzzleScriptAdapter      # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert,      # noqa: E402
                                     Plan, _DELTA)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "Escape_the_Void_Full_"

#: Disk cache of each level's start plan and its optimal-action sets. The whole
#: game is ~0.3s of search, so this is a convenience rather than a necessity --
#: it exists so that `parallelize_generator`'s shards do not each redo it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "escape_the_void_full_plans.json"

#: Engine direction names, in the order the model indexes them. ACTION5 is bound
#: to no rule in this game and is left out entirely.
DIRS: tuple[str, ...] = ("up", "down", "left", "right")

_INF = 1 << 30


# ---------------------------------------------------------------------------
# Native model of the two rules
# ---------------------------------------------------------------------------

class Board:
    """The level as a bitmask state machine, plus the exact distance field.

    A state is ``(p, m1, m2)``: the player's open-square index, the bitmask of
    squares holding a yellow ``Crate1``, and the bitmask holding an orange
    ``Crate2``. Only the open squares are indexed, so a board of 25 walkable
    cells needs 25 bits per mask and the whole state is three small ints -- which
    is what makes the full-closure BFS below cheap enough to be the plan.

    Walls (``Water``, and the never-placed ``Wall``) and targets are static:
    nothing in this game creates or destroys either, so they live on the board
    rather than in the state.
    """

    def __init__(self, eng, idx: dict):
        h, w = eng.height, eng.width
        cells: list[tuple[int, int]] = []
        yellow: list[tuple[int, int]] = []
        orange: list[tuple[int, int]] = []
        targets: list[tuple[int, int]] = []
        player: tuple[int, int] | None = None
        for r in range(h):
            for c in range(w):
                cell = eng.grid[r][c]
                if idx["water"] in cell or idx["wall"] in cell:
                    continue                       # a blocker is not a square
                cells.append((r, c))
                if idx["crate1"] in cell:
                    yellow.append((r, c))
                if idx["crate2"] in cell:
                    orange.append((r, c))
                if idx["target"] in cell:
                    targets.append((r, c))
                if idx["player"] in cell:
                    player = (r, c)
        self.cells = cells
        self.n = len(cells)
        self.height, self.width = h, w
        pos = {cell: i for i, cell in enumerate(cells)}

        # ahead[d][i] / behind[d][i]: the square one step along / against
        # direction d from square i, or -1 when that is a wall or off the board.
        # Both are needed because a press reads BOTH sides of the player: the
        # square ahead (is there an orange crate to push?) and the square behind
        # (is there a yellow crate to tow?).
        self.ahead: list[list[int]] = []
        self.behind: list[list[int]] = []
        for name in DIRS:
            dr, dc = _DELTA[name]
            self.ahead.append([pos.get((r + dr, c + dc), -1) for r, c in cells])
            self.behind.append([pos.get((r - dr, c - dc), -1) for r, c in cells])

        self.target_mask = 0
        for cell in targets:
            self.target_mask |= 1 << pos[cell]
        m1 = m2 = 0
        for cell in yellow:
            m1 |= 1 << pos[cell]
        for cell in orange:
            m2 |= 1 << pos[cell]
        self.n_targets = len(targets)
        self.n_yellow = len(yellow)
        self.n_orange = len(orange)
        self.start = (pos[player], m1, m2) if player is not None else None

    # -- dynamics ------------------------------------------------------------
    def step(self, state: tuple, d: int) -> tuple:
        """The settled state after pressing direction ``d``. Returns ``state``
        itself when the interpreter would refuse the press.

        The order below is the order the rules resolve in, and the early returns
        are the refusals: the player cannot enter a wall, cannot enter a yellow
        crate (nothing pushes one), and cannot enter an orange crate whose
        landing square is occupied -- and in that last case the yellow crate
        behind does NOT get towed either, because the whole turn is cancelled.
        """
        p, m1, m2 = state
        n = self.ahead[d][p]
        if n < 0:                                  # wall or off the board
            return state
        bit = 1 << n
        if m1 & bit:                               # yellow: never pushable
            return state
        if m2 & bit:                               # orange: push it one square
            beyond = self.ahead[d][n]
            if beyond < 0:
                return state
            landing = 1 << beyond
            if (m1 | m2) & landing:
                return state
            m2 = (m2 ^ bit) | landing
        back = self.behind[d][p]
        if back >= 0 and (m1 >> back) & 1:         # yellow: towed into our cell
            m1 = (m1 ^ (1 << back)) | (1 << p)
        return (n, m1, m2)

    def won(self, state: tuple) -> bool:
        """``All Target on Cratey`` -- every target square under a crate of
        either colour."""
        return (state[1] | state[2]) & self.target_mask == self.target_mask

    # -- the exact field -----------------------------------------------------
    def field(self) -> tuple[dict, list, list]:
        """``(index, succ, dist)`` -- the whole reachable state space and the
        exact press-count from each of its states to a win.

        ``index`` maps a state to its id, ``succ[i][d]`` is the id the press
        ``d`` leads to (``-1`` when it is refused, since a state that maps to
        itself can never be on a shortest path), and ``dist[i]`` is the number of
        presses from state ``i`` to the nearest win, or ``_INF`` for a state that
        has already been ruined.

        A won state is a SINK: the level ends there, so its successors do not
        exist and expanding it would let a plan walk *through* a win and out the
        other side. That is also why the forward pass is what bounds the space --
        the reachable-and-not-yet-won closure is the only part of the state graph
        this game can ever be in.
        """
        index: dict[tuple, int] = {self.start: 0}
        order: list[tuple] = [self.start]
        succ: list[list[int]] = []
        i = 0
        while i < len(order):
            state = order[i]
            i += 1
            row = [-1, -1, -1, -1]
            if not self.won(state):
                for d in range(4):
                    nxt = self.step(state, d)
                    if nxt == state:
                        continue                   # a refused press
                    j = index.get(nxt)
                    if j is None:
                        j = len(order)
                        index[nxt] = j
                        order.append(nxt)
                    row[d] = j
            succ.append(row)

        pred: list[list[int]] = [[] for _ in order]
        for j, row in enumerate(succ):
            for k in row:
                if k >= 0:
                    pred[k].append(j)

        dist = [_INF] * len(order)
        queue: deque[int] = deque()
        for j, state in enumerate(order):
            if self.won(state):
                dist[j] = 0
                queue.append(j)
        while queue:
            j = queue.popleft()
            step = dist[j] + 1
            for p in pred[j]:
                if dist[p] > step:
                    dist[p] = step
                    queue.append(p)
        return index, succ, dist

    def solve(self) -> tuple[list[str], list[list[str]], int] | None:
        """``(presses, optsets, closure_size)`` for a shortest solution from the
        level start, or None if the level cannot be won.

        The descent takes the lowest-indexed optimal press at each step purely to
        be deterministic; *which* one is recorded does not matter, because
        ``optsets[i]`` carries every press that is equally shortest there and
        `train_policy` supervises the whole set.
        """
        if self.start is None:
            return None
        index, succ, dist = self.field()
        if dist[0] >= _INF:
            return None
        presses: list[str] = []
        optsets: list[list[str]] = []
        state, here = self.start, 0
        while dist[here] > 0:
            best = [d for d in range(4)
                    if succ[here][d] >= 0
                    and dist[succ[here][d]] == dist[here] - 1]
            # The field is exact, so a state with a finite distance always has at
            # least one press that realises it; an empty set would mean the two
            # passes disagreed.
            assert best, "state with finite distance has no optimal press"
            optsets.append([DIRS[d] for d in best])
            presses.append(DIRS[best[0]])
            here = succ[here][best[0]]
            state = self.step(state, best[0])
        assert self.won(state)
        return presses, optsets, len(index)


def board_from_engine(eng, game) -> Board | None:
    """Read the engine's current grid into a `Board`, or None if the level has
    no player (nothing in this game can remove one -- the guard is for a level
    that simply never placed one)."""
    board = Board(eng, game.obj_name_to_idx)
    return board if board.start is not None else None


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class VoidExpert(PSExpert):
    """`PSExpert` with the whole search replaced by `Board`'s exact field.

    What is inherited is everything around the search: the in-memory plan memo,
    the on-disk start-plan cache with its staleness check, and the
    snapshot/restore discipline `PSExpert.plan` wraps `_search` in. `heuristic`
    -- the hook a stepping A* would call -- is never reached, because nothing
    here searches the interpreter.

    WHY NOT `PSPushExpert`. Its macros are "walk to the push square, then push",
    which assumes a walk has no side effects. Half the crates in this game are
    yellow, and a yellow crate is towed by any step the player takes away from
    it, so a macro's walk would drag pieces around the board without the search
    ever modelling it. The same rule also makes the region-based dedup that
    makes those macros affordable unsound (reachability is one-way: the player
    can step toward a yellow crate but not back away from it for free).
    """

    directions = list(DIRS)          # ACTION is bound to no rule in this game
    plan_cache_path = PLAN_CACHE

    def __init__(self, game, node_cap: int = 400_000, weight: int = 1):
        super().__init__(game, node_cap=node_cap, weight=weight)
        self._how: dict = {}         # level -> one line for `describe`
        self._level: int | None = None

    def heuristic(self, eng) -> int:                        # pragma: no cover
        raise AssertionError("VoidExpert plans on Board, not on the engine")

    def _search(self, eng):
        board = board_from_engine(eng, self.g)
        if board is None:
            return None
        started = time.time()
        found = board.solve()
        if found is None:
            return None
        presses, optsets, closure = found
        self._how[self._level] = (f"{closure} states "
                                  f"{time.time() - started:.2f}s")
        return Plan(presses, optsets)

    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan`, remembering which level `describe` is reporting on."""
        self._level = level
        return super().plan(eng, level)

    def describe(self, level) -> str:
        """How this level's plan was found, for ``--plans``; "cached" when it
        came off disk and nothing ran."""
        return self._how.get(level, "cached")


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class VoidSolver(PSAStarSolver):
    game_id = "puzzlescript_escape_the_void_full"
    game_name = GAME_NAME
    expert_cls = VoidExpert

    #: Record against the GAME FOLDER's adapter. That file is a plain
    #: passthrough today, so this is the same object a bare
    #: `PuzzleScriptAdapter(GAME_NAME)` would be -- it is set anyway because it
    #: is what `game_envs` hands a live agent, and a step cap or sprite patch
    #: added there later would otherwise silently make this generator tape a
    #: game nobody plays (the trap ps:count_mover hit).
    game_module_id = "ps:escape_the_void_full"

    #: The longest plan is 82 presses (level 8), so the adapter's 200-press
    #: per-level default leaves a comfortable margin. The RESET that ends the
    #: recovery prefix zeroes that counter, so only the post-reset plan is
    #: charged against it.
    max_steps = 200

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it fills the disk cache in one pass and makes the
        (small) one-time cost visible as startup rather than as a slow first
        seed."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)

    def optimal_for(self, expert, level: int, plan, pi: int):
        """The optimal press set at this step, off the plan's own optsets.
        Falls back to the press about to be taken so no expert step ever ships
        unlabelled -- `train_policy` v2 supervises ``optimal`` only, so a step
        without one contributes nothing to the loss."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets) and sets[pi]:
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Model self-check (fuzz against the real interpreter)
# ---------------------------------------------------------------------------

def _read_state(board: Board, eng, idx: dict) -> tuple:
    """The engine's current grid as a `Board` state."""
    pos = {cell: i for i, cell in enumerate(board.cells)}
    p = 0
    m1 = m2 = 0
    for r in range(eng.height):
        for c in range(eng.width):
            cell = eng.grid[r][c]
            i = pos.get((r, c))
            if i is None:
                continue
            if idx["player"] in cell:
                p = i
            if idx["crate1"] in cell:
                m1 |= 1 << i
            if idx["crate2"] in cell:
                m2 |= 1 << i
    return (p, m1, m2)


def selfcheck(trials: int = 300, steps: int = 60, verbose: bool = True) -> int:
    """Audit the claim the whole solver rests on: `Board` reproduces the
    interpreter EXACTLY.

    Random play is the fuzz, and it is a good one for this game: it walks into
    walls and into yellow crates (the two refusals), shoves orange crates into
    each other and into walls (the cancelled push, which must also cancel the
    tow behind it), drags yellow crates around by accident (the mechanic a
    player has to learn), and runs long enough to ruin the board in ways a plan
    never does. Every settled state and every win flag has to agree.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, idx = game._engine, game._game.obj_name_to_idx
    bad = checked = 0
    for level in range(game.n_levels):
        refusals = wins = 0
        for t in range(trials):
            game.set_level(level)
            board = board_from_engine(eng, game._game)
            state = board.start
            rng = random.Random(f"void:selfcheck:{level}:{t}")
            for _ in range(steps):
                d = rng.randrange(4)
                eng.step(DIRS[d])
                predicted = board.step(state, d)
                actual = _read_state(board, eng, idx)
                checked += 1
                if predicted == state:
                    refusals += 1
                if actual != predicted:
                    bad += 1
                    print(f"  L{level}: state divergence on {DIRS[d]}: "
                          f"model {predicted} engine {actual}")
                    break
                if board.won(predicted) != eng.check_win():
                    bad += 1
                    print(f"  L{level}: win divergence on {DIRS[d]}")
                    break
                state = predicted
                if board.won(state):
                    wins += 1
                    break
        if verbose:
            print(f"  L{level}: {refusals} refused presses, "
                  f"{wins} accidental wins")
    print(f"selfcheck: {checked} presses, {bad} divergences")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _report() -> int:
    """Per-level board size, piece counts, plan length and tie coverage."""
    solver = VoidSolver()
    game = solver.make_game(0)
    expert = VoidExpert(game, node_cap=VoidSolver.node_cap)
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = board_from_engine(eng, game._game)
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        if found is None:
            print(f"level {level:2d}: NO PLAN")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"level {level:2d}: {eng.height:2d}x{eng.width:2d} "
              f"{board.n:2d} open, {board.n_yellow}y/{board.n_orange}o crates, "
              f"{board.n_targets} targets, {len(found):3d} presses "
              f"(budget {game._max_steps}, {room}), "
              f"{ties:2d} ties ({ties / max(1, len(found)):3.0%}), "
              f"{dt:5.2f}s [{expert.describe(level)}]")
    print(f"total {total} presses over {game.n_levels} levels")
    return 0


def _verify_plans() -> int:
    """Replay every cached plan through the real interpreter to a WIN.

    The plans come out of a native model, so this is the certification step: the
    model is what searched, the interpreter is what decides.
    """
    solver = VoidSolver()
    game = solver.make_game(0)
    expert = VoidExpert(game, node_cap=VoidSolver.node_cap)
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level:2d}: NO PLAN")
            bad += 1
            continue
        game.set_level(level)
        eng = game._engine
        won_at = None
        for i, direction in enumerate(plan):
            eng.step(direction)
            if eng.check_win():
                won_at = i
                break
        ok = won_at == len(plan) - 1
        bad += not ok
        print(f"level {level:2d}: {len(plan):3d} presses -> "
              f"{'WIN' if ok else f'FAILED (won_at={won_at})'}")
    print("all plans win" if not bad else f"PLAN REPLAY FAILED on {bad} levels")
    return 0 if not bad else 1


def _verify_ties() -> int:
    """Re-derive every step's optimal set the OTHER way round and compare.

    `Board.solve` reads its labels off one reverse BFS from the winning states.
    This re-measures them with an independent forward BFS from each of the four
    successors -- "how many presses does a win take from *here*" asked once per
    successor per step, sharing nothing with the field but `Board.step` itself.
    Agreement on all 655 steps is the check that the reverse pass is not quietly
    labelling states it merely reached rather than states it can win from.
    """
    solver = VoidSolver()
    game = solver.make_game(0)
    expert = VoidExpert(game, node_cap=VoidSolver.node_cap)

    def shortest(board: Board, state: tuple) -> int:
        """Presses from ``state`` to a win, by plain forward BFS."""
        if board.won(state):
            return 0
        seen = {state}
        queue = deque([(state, 0)])
        while queue:
            cur, d = queue.popleft()
            for direction in range(4):
                nxt = board.step(cur, direction)
                if nxt == cur or nxt in seen:
                    continue
                if board.won(nxt):
                    return d + 1
                seen.add(nxt)
                queue.append((nxt, d + 1))
        return _INF

    bad = steps = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        board = board_from_engine(eng, game._game)
        plan = expert.plan(eng, level)
        state = board.start
        here = shortest(board, state)
        for i, taken in enumerate(plan):
            expected = sorted(
                DIRS[d] for d in range(4)
                if board.step(state, d) != state
                and shortest(board, board.step(state, d)) == here - 1)
            got = sorted(plan.optsets[i])
            steps += 1
            if got != expected or taken not in expected:
                bad += 1
                print(f"  L{level} step {i}: field says {got}, "
                      f"forward BFS says {expected}, took {taken}")
            state = board.step(state, DIRS.index(taken))
            here -= 1
        if here != 0 or not board.won(state):
            bad += 1
            print(f"  L{level}: plan does not land on a win")
    print(f"ties: {steps} steps cross-checked, {bad} disagreements")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    The bug this exists for is in the original art: Target was a hollow ring and
    the Player sprite is opaque at exactly those eight pixels, so a player
    standing on a target was pixel-identical to a player on bare floor -- at
    every board size in the game. Target is now a solid square. The same pass
    catches the second fix: the player's body was White, which is the ARC index
    Water (every wall on every board) renders as.
    """
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    game = VoidSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {"floor": (), "wall": ("water",), "target": ("target",),
             "yellow": ("crate1",), "orange": ("crate2",),
             "yellow_on_target": ("target", "crate1"),
             "orange_on_target": ("target", "crate2"),
             "player": ("player",), "player_on_target": ("target", "player")}

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
            frame = np.asarray(_render_frame(eng, g))
            cell = 64 // max(h, w)
            r, c = h // 2, w // 2
            shots[name] = frame[cell * r:cell * (r + 1),
                                cell * c:cell * (c + 1)].copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"{h:2d}x{w:2d} (cell {64 // max(h, w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{'OK' if not clashes else 'IDENTICAL ' + str(clashes)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--verify" in sys.argv:
        sys.exit(_verify_plans())
    if "--ties" in sys.argv:
        sys.exit(_verify_ties())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    if "--selfcheck" in sys.argv:
        sys.exit(selfcheck())
    sys.exit(VoidSolver.main())
