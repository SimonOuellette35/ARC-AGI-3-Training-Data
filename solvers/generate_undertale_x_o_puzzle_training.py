"""Generate Phase-1 training data for the PuzzleScript game
ps:undertale_x_o_puzzle ("Undertale X/O Puzzle", Toby Fox, transcribed by PKRB).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a measured model of
the mechanic, an EXACT distance-to-win field over that model, the reports that
prove both, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_undertale_x_o_puzzle",
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
exactly. Every expert step carries the full set of equally-optimal presses.

The game
--------
Walk over every X exactly once and finish standing on the button. Two levels.
Everything below was MEASURED against the interpreter (``--model`` is the check),
not read off the .txt:

* **A target has three states and you drive it with your feet.** The two late
  rules are ``[Player TargetBaked] -> [Player TargetBurnt]`` and then
  ``[Player TargetFresh] -> [Player TargetBaked]``, in that order, and they fire
  AFTER the move resolves. So stepping onto a fresh X bakes it to an O (rule 2
  matches, rule 1 does not), and being on a baked O at the end of any later turn
  burns it. Burnt is absorbing: no rule turns it back.
* **The win is "every target baked".** ``No TargetFresh`` + ``No TargetBurnt`` +
  ``All Player on Button``, so each target must be visited an EXACT number of
  times: once. That makes the puzzle a self-avoiding tour of the X's -- free
  floor may be re-crossed as often as you like, an X may not.
* **A press that fails to move you still burns the tile under you.** The late
  rules do not care whether the move happened, so standing on a baked O and
  pressing into a wall burns it, and so does ACTION (which no rule reads, so it
  never moves anything). *You cannot wait on a target.* The corollary is the
  trap on level 1: the cell (2,3) is a dead end whose only exit is the baked O
  at (2,2), so every press available there either does nothing or costs you a
  whole restart.
* **The button is a RESET, not a checkpoint, and it fires on the way OUT.**
  ``[ Player Button ] [ Target ] -> [ Player Button ] [ TargetFresh ]`` is a
  normal rule, so it is applied BEFORE the move resolves: it fires on the turn
  you press a key *while standing on* the button, not on the turn you arrive.
  Arriving with everything baked therefore wins before anything can be undone,
  and arriving early costs you the entire tour -- which makes the button a
  one-way terminal, and on level 1 it sits in the middle of the only corridor
  joining the two halves of the top row.
* **Nothing can kill you and nothing is unrecoverable.** There is no lose
  condition, no ``restart`` and no ``cancel`` in the ruleset, and the button
  un-burns everything, so *every* state either wins or can walk to the button
  and start the tour again. The field below measures that as a fact: 0 of the
  2293880 board configurations it covers are dead.
* ACTION5 is in the adapter's action list and no rule reads it. It is still a
  real press here, because a press that does not move you is exactly the thing
  that burns the tile you are standing on -- so the search branches on it.

Level indices in the reports are 0-BASED, i.e. one less than the number in the
game's own "Level N" messages; the prose above uses the game's numbering.

Why an exact field, and not the shared A*
-----------------------------------------
The state is the player's cell plus one trit per target, and the burnt trit
COLLAPSES: a board with anything burnt can only be won through the button, and
from the button a burnt board and a fresh board have identical successors (the
reset rule fires on the way out either way). So

    dist(burnt board at q) = walk(q -> button) + dist(button, nothing baked)

exactly, and the whole game reduces to the LIVE space -- player cell x baked
SUBSET -- which is 30 x 4 = 120 states on level 1 and 70 x 32768 = 2293760 on
level 2. That is small enough to hold as one numpy array, so this generator does
not search at all: `_Board.enumerate` builds the successor table for all five
presses and sweeps the Bellman backup to convergence, giving the exact
distance-to-win of EVERY configuration, not just the ones on some plan.

That buys three things a heuristic search would not:

* plans that are provably SHORTEST -- 28 and 37 presses -- with no weight, no
  heuristic and no node cap that could quietly bite;
* exact optimal-action SETS (``dist(succ) == dist - 1``) at every step, read off
  the field from the LIVE engine state, so a step after an epsilon detour is
  labelled from where the game actually is;
* a PROOF that no state is dead, which is what makes the epsilon detours below
  free.

The dead-edge cost is the one place this is subtler than a plain BFS, and it is
the one place it was wrong first: a burn edge must be charged the walk distance
of the cell you END UP on, not the one you pressed from. Charging the source
inflated level 1's trap state from 41 to 42 and the Bellman check in ``--verify``
is what caught it.

Rendering (the game file was edited, see its header)
----------------------------------------------------
``data/puzzlescript_games/Undertale_X_O_Puzzle.txt`` has one sprite redrawn and
two colour pairs re-picked; no rule, level, layer or win condition is touched.
Button was drawn on rows 2-3 / columns 1-3, exactly under the Player's opaque
middle, so ``player on button`` -- a third of the win condition and the frame
every episode aims at -- was PIXEL-IDENTICAL to a player on bare floor. It now
also paints the four corners, which are transparent in the Player and are a
closed orbit of the rotation group. And `blue` == `darkblue` == 9 while
`green` == `darkgreen` == 14, so TargetFresh's diagonal X and TargetBaked's ring
both quantized to flat blocks -- with the X landing on the WALL's own colour, on
boards that are mostly wall. Their second colour is now `white` (the
Background's own paper colour), which lets the author's X and O shapes through,
and the X is `purple`. ``--audit`` is the check.

Symmetry and the augmentation
-----------------------------
Not one rule in this game names a direction: the three rules are position
predicates over a single Player, the movement is the interpreter's own, and the
win condition names no axis. So the 8-element presentation group is an exact
symmetry, and ``--symmetry`` measures it rather than asserting it -- every
level's plan is replayed on all eight turned and mirrored copies of its own
LEVEL LAYOUT and every press must leave the board where the transform says.
That is what entitles the game to `PuzzleScriptAdapter._FLIP_GAMES` (16
presentations) on top of the mandatory rotation.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the plan
replays to a guaranteed WIN.

On top of that the expert replay runs at ``epsilon > 0``, and the detours are
the most valuable data this game has: burning an O is the mistake a player
actually makes, and the label on the very next press is the walk back to the
button, i.e. the recovery.

`record_level` keeps a random alternative only if it changes the board AND
leaves the level winnable -- and here that guard is VACUOUS, because no state of
this game is dead. What bites instead is cost: a burn is a 34-press repair, a
longer run draws more detours, and measured over 150 seeds the compounding ran
level 2 into the adapter's 200-press cut-off on 9% of seeds at epsilon 0.08 and
21% at 0.12. So `UndertaleExpert.plan` refuses a detour it cannot afford -- it
answers a HYPOTHETICAL (which is all `record_level` ever asks it for when
vetting one) only when the state it lands in can still win with a press to
spare, and the state the recorder is actually in unconditionally. That turns
"survivable" into "affordable" without touching the shared harness: no seed is
lost at any rate, so the rate is chosen for the shape of the episode instead
(``epsilon_for`` has the table).

CLI
---
    --plans     per-level size, field build, plan length and tie coverage
    --field     the exact-field report: state counts, dead states, the traps
    --model     fuzz the model against the interpreter, press by press
    --verify    prove the field: Bellman + scalar-successor + mutation + replay
    --symmetry  replay every plan on all 8 presentations of its own board
    --audit     assert every cell composition renders distinctly, at every size
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSExpert   # noqa: E402

GAME_NAME = "Undertale_X_O_Puzzle"

#: The five presses, in the order that breaks plan ties -- so a re-derived plan
#: is byte-identical across processes. ACTION is last because it never moves
#: anything: where it ties it is the least interesting way to spend the press.
_DIRS = ("up", "down", "left", "right", "action")

#: Engine direction -> (dr, dc). ACTION is the zero step: it takes a turn and
#: moves nothing, which is precisely how you burn the tile you stand on.
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
          "action": (0, 0)}

#: Target trits.
FRESH, BAKED, BURNT = 0, 1, 2

_INF = np.int32(1 << 29)
_DEADEDGE = -1        # this press burns a target: leaves the live space
_WINEDGE = -2         # this press wins


# ---------------------------------------------------------------------------
# The board: static layout + the exact distance-to-win field
# ---------------------------------------------------------------------------

def _layout_of(eng, ids) -> tuple:
    """The level's STATIC layout, read off any grid of it.

    Walls, the button and WHICH cells are targets never change -- no rule
    creates, destroys or moves any of them, the three target objects only
    transform into each other in place -- so this is the same tuple whatever
    state of the level it is read from, which is what lets `_Board` be cached by
    it and shared across seeds and mid-game re-plans."""
    h, w = eng.height, eng.width
    tset = ids["fresh"] | ids["baked"] | ids["burnt"]
    walls, targets, button = [], [], None
    for r in range(h):
        for c in range(w):
            cell = eng.grid[r][c]
            if cell & ids["wall"]:
                walls.append((r, c))
            if cell & tset:
                targets.append((r, c))
            if cell & ids["button"]:
                assert button is None, "two buttons: the reset is not a point"
                button = (r, c)
    assert button is not None, "no button: the win condition is unreachable"
    return (h, w, tuple(walls), tuple(targets), button)


class _Board:
    """One level's exact distance-to-win field.

    THE STATE SPACE. A configuration is the player's cell plus one trit per
    target (fresh / baked / burnt). The burnt trits collapse out (see the module
    docstring), so the array below is indexed by the LIVE space only --
    ``cell_index * 2**n_targets + baked_bitmask`` -- and `distance` routes a
    board with anything burnt through the closed form instead.

    Not every index is a configuration the game can present: an index whose
    player cell is a target with its own bit CLEAR would be a player standing on
    a fresh X, which the late rule makes impossible. Those indices are still
    filled in (they are only ever sources, never successors, so they cannot
    corrupt anything) and simply never queried.

    THE FIELD. `enumerate` builds the successor table for all five presses and
    then sweeps ``d <- min(d, 1 + min_press cost)`` from INF to convergence, so
    after k sweeps the array holds "presses to a win using at most k presses"
    and at the fixpoint it is the exact distance by construction. The dead-edge
    cost ``walk(q -> button) + d(button, nothing baked)`` depends on the field
    being swept, so it is recomputed inside the loop; it is monotone, so the
    sweep still converges downward to the true distances.
    """

    #: A wider board would need a different encoding rather than a bigger array
    #: -- 2**24 masks x the cells is already ~10 GB. Both shipped levels are far
    #: under it (2 and 15 targets); the assert is so a future level says so.
    MAX_TARGETS = 22

    def __init__(self, layout: tuple):
        self.layout = layout
        self.h, self.w, walls, self.targets, self.button = layout
        self.walls = frozenset(walls)
        self.cells = [(r, c) for r in range(self.h) for c in range(self.w)
                      if (r, c) not in self.walls]
        self.cid = {p: i for i, p in enumerate(self.cells)}
        self.tindex = {p: i for i, p in enumerate(self.targets)}
        self.n = len(self.targets)
        assert self.n <= self.MAX_TARGETS, (
            f"{self.n} targets: the dense mask encoding does not fit")
        self.M = 1 << self.n
        self.full = self.M - 1
        self.btn = self.cid[self.button]
        self.N = len(self.cells) * self.M
        self.dbtn = self._walk_field(self.button)
        self.succ: np.ndarray | None = None
        self.qcell: np.ndarray | None = None
        self.dist: np.ndarray | None = None
        self.sweeps = 0
        self.build_s = 0.0

    # -- geometry -------------------------------------------------------------
    def _walk_field(self, goal) -> np.ndarray:
        """Presses to WALK to ``goal``, ignoring the targets entirely.

        Sound because nothing in this game blocks the player except walls: a
        target is on a lower collision layer and the button is enterable, so the
        only thing a walk can cost is the wall it has to go round."""
        seen = {goal: 0}
        queue = deque([goal])
        while queue:
            p = queue.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                q = (p[0] + dr, p[1] + dc)
                if q in self.cid and q not in seen:
                    seen[q] = seen[p] + 1
                    queue.append(q)
        out = np.full(len(self.cells), _INF, np.int32)
        for p, d in seen.items():
            out[self.cid[p]] = d
        return out

    def _dest(self, p: tuple, direction: str) -> tuple:
        """Where a press from cell ``p`` leaves the player. A blocked press
        leaves them where they were, which is not a no-op: the late rules still
        fire on the tile underneath."""
        dr, dc = _DELTA[direction]
        q = (p[0] + dr, p[1] + dc)
        return q if q in self.cid else p

    # -- the scalar model (the authority the fuzz test compares) ---------------
    def start_state(self, player: tuple) -> tuple:
        return (player, (FRESH,) * self.n)

    def step(self, state: tuple, direction: str) -> tuple:
        """One press, as the interpreter executes it: the button rule fires
        BEFORE the move (it is a normal rule and the move resolves after the
        rule pass), then the move, then the two late bake rules on whatever the
        player is standing on."""
        pos, ts = state
        if pos == self.button and any(t != FRESH for t in ts):
            ts = (FRESH,) * self.n
        pos = self._dest(pos, direction)
        i = self.tindex.get(pos)
        if i is not None and ts[i] != BURNT:
            ts = ts[:i] + (BURNT if ts[i] == BAKED else BAKED,) + ts[i + 1:]
        return (pos, ts)

    def win(self, state: tuple) -> bool:
        pos, ts = state
        return pos == self.button and all(t == BAKED for t in ts)

    # -- the field ------------------------------------------------------------
    def enumerate(self) -> None:
        """Build the successor table and sweep the field. Idempotent."""
        if self.dist is not None:
            return
        t0 = time.time()
        M, ncell = self.M, len(self.cells)
        masks = np.arange(M, dtype=np.int32)
        succ = np.empty((5, self.N), np.int32)
        qcell = np.empty((5, ncell), np.int32)
        for di, direction in enumerate(_DIRS):
            for pi, p in enumerate(self.cells):
                qi = self.cid[self._dest(p, direction)]
                qcell[di, pi] = qi
                # Standing on the button, the reset rule has already made every
                # target fresh by the time the move resolves.
                m1 = np.zeros(M, np.int32) if pi == self.btn else masks
                t = self.tindex.get(self.cells[qi])
                if t is None:
                    nxt = qi * M + m1
                    m2 = m1
                else:
                    m2 = m1 | np.int32(1 << t)
                    nxt = np.where(((m1 >> t) & 1).astype(bool),
                                   _DEADEDGE, qi * M + m2)
                if qi == self.btn:
                    nxt = np.where(m2 == self.full, _WINEDGE, nxt)
                succ[di, pi * M:(pi + 1) * M] = nxt
        self.succ, self.qcell = succ, qcell

        deadm = [succ[d] == _DEADEDGE for d in range(5)]
        winm = [succ[d] == _WINEDGE for d in range(5)]
        take = [np.where(succ[d] >= 0, succ[d], 0) for d in range(5)]
        dist = np.full(self.N, _INF, np.int32)
        sweeps = 0
        while True:
            # The cost of burning something: walk to the button from where the
            # burn LEAVES you, then run the whole tour again. Recomputed every
            # sweep because it is a function of the field being swept.
            reset = int(dist[self.btn * M])
            deadc = (np.minimum(self.dbtn.astype(np.int64) + reset, int(_INF))
                     .astype(np.int32) if reset < _INF
                     else np.full(ncell, _INF, np.int32))
            best = np.full(self.N, _INF, np.int32)
            for d in range(5):
                c = dist[take[d]]
                c = np.where(deadm[d], np.repeat(deadc[qcell[d]], M), c)
                c = np.where(winm[d], 0, c)
                np.minimum(best, c, out=best)
            new = np.minimum(dist, np.where(best >= _INF, _INF, best + 1))
            sweeps += 1
            if np.array_equal(new, dist):
                break
            dist = new
        self.dist = dist
        self.sweeps = sweeps
        self.build_s = time.time() - t0

    # -- queries --------------------------------------------------------------
    def index(self, state: tuple) -> int:
        """The live-space index of ``state``. Only valid when nothing is burnt."""
        pos, ts = state
        mask = 0
        for i, t in enumerate(ts):
            if t == BAKED:
                mask |= 1 << i
        return self.cid[pos] * self.M + mask

    def reset_cost(self) -> "int | None":
        d = int(self.dist[self.btn * self.M])
        return None if d >= _INF else d

    def distance(self, state: tuple) -> "int | None":
        """Exact presses to a win from ``state``, or None if it can never win.

        A board with anything burnt goes through the closed form: it can only be
        cleared at the button, and from the button a burnt board and an all-fresh
        one have the same successors, so its distance is the walk plus the
        all-fresh tour."""
        self.enumerate()
        pos, ts = state
        if BURNT in ts:
            walk = int(self.dbtn[self.cid[pos]])
            rest = self.reset_cost()
            if walk >= _INF or rest is None:
                return None
            return walk + rest
        d = int(self.dist[self.index(state)])
        return None if d >= _INF else d

    def cost(self, state: tuple, direction: str) -> "int | None":
        """Presses to a win if ``direction`` is pressed at ``state``."""
        nxt = self.step(state, direction)
        if self.win(nxt):
            return 1
        d = self.distance(nxt)
        return None if d is None else d + 1

    def optimal(self, state: tuple) -> list:
        """Every press on a shortest route from ``state``."""
        best = self.distance(state)
        if best is None:
            return []
        return [d for d in _DIRS if self.cost(state, d) == best]

    def plan(self, state: tuple) -> "list | None":
        """The shortest press sequence from ``state``, ties broken by `_DIRS`."""
        best = self.distance(state)
        if best is None:
            return None
        out: list[str] = []
        cur = state
        while not self.win(cur):
            d = self.optimal(cur)[0]
            out.append(d)
            cur = self.step(cur, d)
        assert len(out) == best, (len(out), best)
        return out


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class UndertaleExpert(PSExpert):
    """`PSExpert`'s plan memo and restore discipline around `_Board`'s field.

    `_search` is replaced the way ps:tile_tiler replaces it -- the base class
    keeps the memo, the snapshot discipline and the level scoping, and only the
    strategy underneath changes. Here the strategy is a field lookup, so the
    interpreter is stepped only by the RECORDER, which is also what verifies
    every plan: a level is kept only when the engine itself reports WIN.

    `_key` is inherited (every non-background cell), which is exact and
    canonical ACROSS levels -- the walls are in the key and no two levels share
    theirs -- so ``scope_by_level`` is unnecessary. The board is likewise found
    from the grid rather than from a level index, so `plan` works from any state
    the recorder hands it, with or without a level argument.
    """

    directions = list(_DIRS)

    def setup(self) -> None:
        g = self.g
        self.ids = {
            "wall": set(g.resolve_object_name("wall")),
            "button": set(g.resolve_object_name("button")),
            "fresh": set(g.resolve_object_name("targetfresh")),
            "baked": set(g.resolve_object_name("targetbaked")),
            "burnt": set(g.resolve_object_name("targetburnt")),
            "player": set(self.game._engine._player_indices),
        }
        self._boards: dict[tuple, _Board] = {}
        #: Presses left before the adapter cuts the level off, and the state
        #: they were counted at. Both None outside a recording, where nothing
        #: has a budget (`discover_solvable` and the reports plan freely).
        self.budget: int | None = None
        self._anchor: tuple | None = None

    # -- reading the engine ---------------------------------------------------
    def board(self, eng) -> _Board:
        """The level's `_Board`, built (and swept) once and cached by its STATIC
        layout -- so every state of a level finds the same field, and a level
        re-visited at another seed does not rebuild it."""
        key = _layout_of(eng, self.ids)
        board = self._boards.get(key)
        if board is None:
            board = _Board(key)
            board.enumerate()
            self._boards[key] = board
        return board

    def read(self, eng, board: _Board) -> tuple:
        """The live configuration off the engine grid. Nothing is inferred: the
        player is located and every target cell reports its own trit."""
        ids = self.ids
        player = None
        for r in range(board.h):
            for c in range(board.w):
                if eng.grid[r][c] & ids["player"]:
                    assert player is None, "two players"
                    player = (r, c)
        assert player is not None, "the player was deleted"
        ts = []
        for (r, c) in board.targets:
            cell = eng.grid[r][c]
            ts.append(BAKED if cell & ids["baked"] else
                      BURNT if cell & ids["burnt"] else FRESH)
        return (player, tuple(ts))

    def state(self, eng) -> tuple:
        return self.read(eng, self.board(eng))

    # -- planning -------------------------------------------------------------
    def heuristic(self, eng) -> int:
        """The EXACT remaining press count. The base `_astar` is never reached
        here, but the contract -- 0 at a win, admissible -- holds, and the field
        is both."""
        board = self.board(eng)
        d = board.distance(self.read(eng, board))
        return 0 if d is None else d

    def observe(self, eng) -> None:
        """A frame has just been presented: re-read how many presses the level
        has left, and remember the state they were counted at.

        The adapter flips a level to GAME_OVER at 200 counted presses, which is
        generous against a 37-press plan and NOT generous against the detours:
        stepping onto an O you already baked costs the whole tour again (34
        presses), and one such detour lengthens the run, which draws more
        detours. Left ungoverned the tail runs into the cut-off on ~10% of seeds
        at a useful epsilon; `plan` spends this budget to stop that, and the
        rate in `UndertaleSolver.epsilon_for` is set on the assumption that it
        does."""
        board = self.board(eng)
        self._anchor = self.read(eng, board)
        game = self.game
        self.budget = max(0, int(game._max_steps) - int(game._action_count))

    def plan(self, eng, level: int | None = None) -> "list | None":
        """`PSExpert.plan`, refusing a HYPOTHETICAL that cannot be finished
        inside the presses this level has left.

        `record_level` vets an epsilon detour by stepping the engine, calling
        this, and keeping the alternative only if it gets a plan back -- so this
        is the one hook a generator has for saying "that mistake is affordable"
        rather than merely "that mistake is survivable". Every state of this game
        is survivable (``--field``), so without the budget the guard passes
        unconditionally and the detours walk the level into the adapter's
        cut-off.

        Only a hypothetical is refused. A call at the state the recorder is
        ACTUALLY in -- the one `observe` last saw -- is answered unconditionally,
        because that is the exploit path and returning None there would end the
        level unsolved. The two stay consistent by construction: a detour is
        accepted only when the state it lands in can still win with a press to
        spare, so the invariant ``distance <= budget`` survives the press that
        takes it, and an optimal press spends one of each.

        Outside a recording ``budget`` is None and nothing is refused."""
        if self.budget is not None:
            board = self.board(eng)
            state = self.read(eng, board)
            if state != self._anchor:
                d = board.distance(state)
                if d is None or d + 1 > self.budget:
                    return None
        return super().plan(eng, level)

    def _search(self, eng) -> "list | None":
        board = self.board(eng)
        return board.plan(self.read(eng, board))

    def optimal_dirs(self, eng) -> list:
        """Every press on a shortest route, from the engine's CURRENT state."""
        board = self.board(eng)
        return board.optimal(self.read(eng, board))


class UndertaleSolver(PSAStarSolver):
    game_id = "puzzlescript_undertale_x_o_puzzle"
    game_name = GAME_NAME
    expert_cls = UndertaleExpert

    #: The game folder's ``make_game`` is a plain passthrough to
    #: `PuzzleScriptAdapter`, so the family default (build the adapter directly)
    #: records against exactly the object a live agent is handed.
    game_module_id = ""

    #: Unused: the expert reads an exact field rather than searching under a
    #: heuristic, so there is no weight to trade and no node budget that could
    #: quietly bite. Left at the family default so a future subclass that does
    #: search is not silently starved.
    node_cap = 400_000
    weight = 1

    #: The adapter GAME_OVERs at 200 actions per level and `record_level`'s
    #: closing ``set_level`` gives the plan the whole budget back, so this only
    #: has to cover the plan (28 / 37) plus whatever the detours add.
    max_steps = 200

    #: `record_level` keeps a random alternative only if it changes the board
    #: AND leaves the level winnable -- and ``--field`` proves NO state of this
    #: game is dead, so that guard never fails and a detour can never strand the
    #: recording. What governs the rate instead is COST, and
    #: `UndertaleExpert.plan`'s press budget is what enforces it; see
    #: `epsilon_for` for the measurements behind this number.
    epsilon = 0.05

    def epsilon_for(self, level: int) -> float:
        """One flat rate for both levels, chosen from measurement rather than
        from the family default.

        A detour here is not the usual one-press-wasted: stepping onto an O you
        already baked burns it, and the only repair is the whole tour again (32
        presses on level 1, 34 on level 2). Against 28- and 37-press plans in a
        200-press level that COMPOUNDS -- a burn lengthens the run, a longer run
        draws more detours -- so without `UndertaleExpert.plan`'s budget guard
        the rate is bounded by the cut-off rather than by what the data wants.
        Level 2, 150 seeds, recorded length T (which includes the RESET
        exploration prefix) and seeds lost to the 200-press cut-off:

            eps    no guard                with the guard
            0.03   T~67   2 lost           T~69    0 lost
            0.04   T~74   3 lost
            0.05   T~82   8 lost           T~88    0 lost
            0.08   T~105  14 lost          T~125   0 lost
            0.12   T~117  32 lost          T~163   0 lost

        With the guard nothing is ever lost, so the rate is free to be chosen
        for the SHAPE of the episode instead. 0.05 puts level 2 at about twice
        its optimal length: roughly one mistake every twenty presses, each
        followed by the walk back to the button and a fresh tour, all of it
        expert-labelled. Higher rates keep working but pin every episode against
        the cut-off (p90 of T is 212 of 200 counted presses at 0.12), which
        makes the budget rather than the mechanic the thing the corpus is
        mostly about."""
        return self.epsilon

    def prepare_expert(self, game, expert) -> None:
        """Build every level's field up front.

        Pure front-loading -- the field does not depend on which state builds it
        -- but it means level 2's 2.3M-state sweep is paid once, at startup,
        rather than inside the first query that needs it."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.board(game._engine)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route to the win, read
        off the LIVE engine state.

        `record_level` asks for this before executing the press, so the engine is
        still at the state being labelled -- which is what makes the label right
        after an epsilon detour too, where the state is off the plan entirely and
        replaying the plan from the level start would label the wrong states.

        Falls back to the press about to be taken if the field has nothing to say
        -- no expert step may ship unlabelled."""
        best = expert.optimal_dirs(expert.game._engine)
        if best:
            return best
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = UndertaleSolver()
    game = solver.make_game(seed)
    return solver, game, UndertaleExpert(game, node_cap=UndertaleSolver.node_cap)


def _boards():
    """``(solver, game, expert, [(level, board, start_state), ...])``."""
    solver, game, expert = _new()
    out = []
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(game._engine)
        out.append((level, board, expert.read(game._engine, board)))
    return solver, game, expert, out


def _report() -> int:
    """Per-level board size, field build, plan length and tie coverage."""
    solver, game, expert, levels = _boards()
    total = tied = 0
    for level, board, start in levels:
        plan = board.plan(start)
        if plan is None:
            print(f"level {level}: UNWINNABLE (see --field)")
            continue
        sets = [board.optimal(s) for s in _walk_states(board, start, plan)]
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        tied += ties
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        eps = solver.epsilon_for(level)
        print(f"level {level}: {board.h:2d}x{board.w:2d}, {board.n:2d} targets, "
              f"{len(board.cells):2d} free cells, field {board.N:8d} states in "
              f"{board.sweeps:3d} sweeps / {board.build_s:5.2f}s -- "
              f"{len(plan):3d} presses (budget {game._max_steps}, {room}), "
              f"{ties:3d} with a tie set ({ties / max(1, len(plan)):3.0%}), "
              f"eps {eps:.2f}")
    print(f"total {total} presses, {tied}/{total} steps with a tie "
          f"({tied / max(1, total):.0%})")
    return 0


def _walk_states(board: _Board, start: tuple, plan: list) -> list:
    """The state each press of ``plan`` is taken from."""
    out, cur = [], start
    for d in plan:
        out.append(cur)
        cur = board.step(cur, d)
    return out


def _field() -> int:
    """The exact-field report -- and therefore the proof that the plans are
    SHORTEST: the field covers every configuration the live space admits, so its
    backward sweep is exact rather than a search that gave up.

    It is also the DEAD-STATE report, and the answer is the unusual one: there
    are none. The button un-burns everything and it can be walked to from every
    cell of both levels, so no press of this game can ever make it unwinnable --
    which is exactly what makes the epsilon detours in `UndertaleSolver` free.

    The traps it does have are FORCED BURNS: cells from which every available
    press either changes nothing or burns the O underneath, i.e. a dead end whose
    only exit is the target you came in over. Nothing marks them -- no death, no
    restart, no visual tell -- and stepping into one costs the whole tour."""
    _solver, _game, _expert, levels = _boards()
    bad = 0
    for level, board, start in levels:
        dist = board.dist
        live = int((dist < _INF).sum())
        far = int(dist[dist < _INF].max())
        reset = board.reset_cost()
        d0 = board.distance(start)
        # FORCED BURNS, counted exactly over the whole live space: a
        # configuration where every press either leaves the board untouched or
        # burns something. Read straight off the successor table -- a press is a
        # non-move when its column points back at the state itself.
        me = np.arange(board.N, dtype=np.int32)
        stuck = np.ones(board.N, bool)
        burns = np.zeros(board.N, bool)
        for d in range(5):
            col = board.succ[d]
            burns |= col == _DEADEDGE
            stuck &= (col == _DEADEDGE) | (col == me)
        trapped = stuck & burns
        print(f"level {level}: {board.N:8d} live configurations, "
              f"{live} can win ({board.N - live} dead), deepest {far} presses, "
              f"walk-to-button max {int(board.dbtn.max())}, "
              f"restart costs {reset}, d*(start) = {d0}")
        idxs = np.flatnonzero(trapped)
        if idxs.size:
            cells = sorted({board.cells[int(i) // board.M] for i in idxs})
            print(f"          {idxs.size} forced-burn configurations, on cells "
                  f"{cells} -- every press there either does nothing or costs "
                  f"you the tour")
        if board.N != live:
            bad += 1
        if d0 is None:
            bad += 1
    print("field clean -- every configuration of this game can still win"
          if not bad else f"FIELD FAILED: {bad} levels have dead states")
    return 0 if not bad else 1


def _model() -> int:
    """Fuzz `_Board.step` against the interpreter, press by press.

    The field is only as good as the model underneath it, and the model is a
    re-implementation of three rules whose ORDER and whose position in the turn
    (the button reset is a normal rule and fires BEFORE the move; the two bake
    rules are late and fire after) are the whole mechanic. So rather than argue
    from the .txt: drive random walks on the real engine and require the model's
    (player, trits) and its win flag to agree after EVERY press.

    The walks deliberately include the states a plan never visits -- burnt
    boards, wasted ACTION presses, presses into walls, and the button pressed
    from with work in progress -- because those are exactly the transitions the
    closed form for burnt boards rests on."""
    _solver, game, expert, levels = _boards()
    eng = game._engine
    bad = checked = 0
    for level, board, start in levels:
        rng = random.Random(1000 + level)
        for trial in range(80):
            game.set_level(level)
            state = start
            for _ in range(150):
                d = rng.choice(_DIRS)
                eng.step(d)
                state = board.step(state, d)
                checked += 1
                live = expert.read(eng, board)
                if live != state or eng.check_win() != board.win(state):
                    print(f"  level {level} trial {trial}: after {d!r} the "
                          f"engine says {live} win={eng.check_win()} and the "
                          f"model says {state} win={board.win(state)}")
                    bad += 1
                    break
                if eng.check_win():
                    break
        print(f"level {level}: 80 random walks agree press for press")
    print(f"model clean -- {checked} presses" if not bad
          else f"MODEL FAILED: {bad} divergences in {checked} presses")
    return 0 if not bad else 1


def _verify() -> int:
    """Prove the field, four ways.

    PASS 1 -- BELLMAN, EXHAUSTIVE. Check ``d(s) == 1 + min_press cost(s, press)``
    at EVERY live configuration of both levels. That single equality is the whole
    proof and it is worth spelling out why: the ``<=`` direction makes ``d`` an
    upper bound by induction on the true optimal path, and the ``>=`` direction
    means every state has a successor one step nearer, so descending strictly
    decreases ``d`` and must reach a win in exactly ``d`` presses -- an
    achievable route. Upper bound plus achievable is equality. This is also the
    pass that caught the dead-edge bug: charging a burn the SOURCE cell's walk
    distance instead of the destination's satisfies nothing here, and level 1's
    trap state read 42 where the truth is 41.

    PASS 2 -- SCALAR SUCCESSORS. The table Pass 1 checks was built vectorised,
    one broadcast per (cell, press); re-derive a large sample of its entries
    through the SCALAR `_Board.step` -- the same function ``--model`` fuzzes
    against the interpreter -- and compare. That closes the loop
    table == model == interpreter.

    PASS 3 -- MUTATION. Roll one direction's successor column by one and require
    Pass 1 to FAIL, so a clean report is evidence rather than a tautology.

    PASS 4 -- REPLAY. Press every plan into the real interpreter and require
    ``check_win()``, and at every step require the field's optimal set to contain
    the press taken and to consist only of presses that re-solve in the same
    number of moves.
    """
    _solver, game, expert, levels = _boards()
    eng = game._engine
    bad = 0

    def bellman(board: _Board, dist: np.ndarray) -> int:
        """Number of live configurations violating the Bellman equality."""
        M = board.M
        reset = int(dist[board.btn * M])
        deadc = (np.minimum(board.dbtn.astype(np.int64) + reset, int(_INF))
                 .astype(np.int32) if reset < _INF
                 else np.full(len(board.cells), _INF, np.int32))
        best = np.full(board.N, _INF, np.int32)
        for d in range(5):
            col = board.succ[d]
            c = dist[np.where(col >= 0, col, 0)]
            c = np.where(col == _DEADEDGE, np.repeat(deadc[board.qcell[d]], M), c)
            c = np.where(col == _WINEDGE, 0, c)
            np.minimum(best, c, out=best)
        want = np.where(best >= _INF, _INF, best + 1)
        return int((want != dist).sum())

    for level, board, start in levels:
        # PASS 1
        off = bellman(board, board.dist)
        print(f"level {level}: pass 1 Bellman over {board.N} configurations -- "
              f"{'exact' if not off else str(off) + ' VIOLATIONS'}")
        bad += off

        # PASS 2
        rng = random.Random(7 + level)
        sample = (range(board.N) if board.N <= 200_000 else
                  [rng.randrange(board.N) for _ in range(200_000)])
        n_sample = board.N if isinstance(sample, range) else len(sample)
        wrong = 0
        for s in sample:
            pi, mask = divmod(int(s), board.M)
            pos = board.cells[pi]
            ts = [BAKED if (mask >> i) & 1 else FRESH for i in range(board.n)]
            state = (pos, tuple(ts))
            for di, d in enumerate(_DIRS):
                nxt = board.step(state, d)
                col = int(board.succ[di][s])
                if board.win(nxt):
                    ok = col == _WINEDGE
                elif BURNT in nxt[1]:
                    ok = (col == _DEADEDGE
                          and int(board.qcell[di][pi]) == board.cid[nxt[0]])
                else:
                    ok = col == board.index(nxt)
                if not ok:
                    wrong += 1
        print(f"level {level}: pass 2 scalar successors on {n_sample} "
              f"configurations x 5 presses -- "
              + ("agree" if not wrong else f"{wrong} MISMATCHES"))
        bad += wrong

        # PASS 3
        mutated = board.succ.copy()
        keep, board.succ = board.succ, mutated
        mutated[2] = np.roll(mutated[2], 1)
        caught = bellman(board, board.dist)
        board.succ = keep
        print(f"level {level}: pass 3 mutation (one press column rolled) -- "
              + (f"DETECTED at {caught} states" if caught else "NOT DETECTED"))
        if not caught:
            bad += 1

        # PASS 4
        plan = board.plan(start)
        game.set_level(level)
        state = start
        mislabelled = 0
        for i, d in enumerate(plan):
            best = board.optimal(state)
            remaining = len(plan) - i
            if d not in best:
                mislabelled += 1
            for cand in best:
                if board.cost(state, cand) != remaining:
                    mislabelled += 1
            eng.step(d)
            state = board.step(state, d)
        won = eng.check_win()
        print(f"level {level}: pass 4 replay {len(plan)} presses -- "
              f"interpreter {'WIN' if won else 'NO WIN'}, "
              + ("labels exact" if not mislabelled
                 else f"{mislabelled} BAD LABELS"))
        if not won:
            bad += 1
        bad += mislabelled

    print("verify clean -- the field is the exact distance-to-win"
          if not bad else f"VERIFY FAILED: {bad} problems")
    return 0 if not bad else 1


def _transform(k: int, mirror: bool):
    """``(cell_fn, dims_fn, direction_map)`` for one element of the 8-group:
    mirror left-right first, then turn clockwise ``k`` times. The direction map
    is DERIVED from the same linear part that moves the cells, so the two cannot
    drift apart. ACTION is its own image: it has no direction to turn."""
    def cell(rc, hw):
        r, c = rc
        h, w = hw
        if mirror:
            c = w - 1 - c
        for _ in range(k):
            r, c, h, w = c, h - 1 - r, w, h
        return (r, c)

    def dims(hw):
        h, w = hw
        return (w, h) if k % 2 else (h, w)

    dmap = {}
    for name, (dr, dc) in _DELTA.items():
        if mirror:
            dc = -dc
        for _ in range(k):
            dr, dc = dc, -dr
        dmap[name] = next(n for n, d in _DELTA.items() if d == (dr, dc))
    return cell, dims, dmap


def _symmetry() -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an
    exact symmetry of the mechanic, which is what entitles this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation on top of the mandatory
    rotation.

    The argument from the source is about as strong as it gets -- not one of the
    three rules names a direction, there is a single Player so no two forces can
    contest a cell, and the win condition names no axis -- but reading rules and
    declaring them symmetric is exactly the argument that was wrong in
    ps:gobble_rush. So instead each level's plan is replayed on all eight turned
    and mirrored copies of its own LEVEL LAYOUT (transforming the layout, so the
    interpreter re-derives the board itself), and every press must leave the
    player and all three target populations where the transform says."""
    _solver, game, expert, levels = _boards()
    eng, g = game._engine, game._game
    ids = expert.ids
    bad = 0

    def read(hw):
        return (
            next((r, c) for r in range(eng.height) for c in range(eng.width)
                 if eng.grid[r][c] & ids["player"]),
            frozenset((r, c) for r in range(eng.height) for c in range(eng.width)
                      if eng.grid[r][c] & ids["fresh"]),
            frozenset((r, c) for r in range(eng.height) for c in range(eng.width)
                      if eng.grid[r][c] & ids["baked"]),
            frozenset((r, c) for r in range(eng.height) for c in range(eng.width)
                      if eng.grid[r][c] & ids["burnt"]),
        )

    for level, board, start in levels:
        # A plain plan is a tour and never burns anything, so drive a longer
        # WALK on top of it: the burn and the button reset have to be inside the
        # thing being tested.
        rng = random.Random(500 + level)
        presses = board.plan(start) + [rng.choice(_DIRS) for _ in range(60)]
        layout = g.levels[level]
        hw = (len(layout), len(layout[0]))

        eng.load_level(layout)
        ref = []
        for d in presses:
            eng.step(d)
            ref.append(read(hw))

        notes = []
        for k, mirror in itertools.product(range(4), (False, True)):
            if (k, mirror) == (0, False):
                continue
            cell, dims, dmap = _transform(k, mirror)
            th, tw = dims(hw)
            turned = [[set() for _ in range(tw)] for _ in range(th)]
            for r, row in enumerate(layout):
                for c, objs in enumerate(row):
                    tr, tc = cell((r, c), hw)
                    turned[tr][tc] = set(objs)
            eng.load_level(turned)
            for i, d in enumerate(presses):
                eng.step(dmap[d])
                player, fresh, baked, burnt = ref[i]
                want = (cell(player, hw),
                        frozenset(cell(x, hw) for x in fresh),
                        frozenset(cell(x, hw) for x in baked),
                        frozenset(cell(x, hw) for x in burnt))
                if read(hw) != want:
                    notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                    break
        bad += len(notes)
        print(f"level {level}: {len(presses):3d} presses x 7 presentations -- "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print("symmetry clean -- the flip augmentation is exact" if not bad
          else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every cell size the boards use.

    Whole 64x64 frames of uniform boards are compared rather than one cell out of
    a mixed board: `_render_frame` upscales and then centre-pads, so indexing a
    cell by ``cell_px * r`` silently reads the wrong pixels (the ps:explod
    lesson). Two uniform boards render identically iff their cells do.

    Ten compositions, which is all a settled board of this game can hold. Target
    and Button share a collision layer so they can never stack, Player and Wall
    likewise, and ``player + targetfresh`` is unreachable for a subtler reason:
    entering a fresh target BAKES it in the same turn (the late rule), and the
    only rule that makes a target fresh again requires the player to be standing
    on the button, which no target can be. ``player + button`` is the WIN frame
    and the reason Button is redrawn in the .txt -- it rendered as a bare player
    at both sizes, which is what this report would show if the sprite were
    reverted.

    Also checks no composition renders as a uniform block of the letterbox pad
    colour (the ps:stand_iii check) -- this game has no void terrain, but the
    check is free and the failure it catches is invisible to a pairwise matrix.
    """
    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, _expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    comps: dict = {
        "floor": (),
        "wall": ("wall",),
        "button": ("button",),
        "fresh": ("targetfresh",),
        "baked": ("targetbaked",),
        "burnt": ("targetburnt",),
        "player": ("player",),
        "player+button": ("button", "player"),
        "player+baked": ("targetbaked", "player"),
        "player+burnt": ("targetburnt", "player"),
    }

    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)

    bad = 0
    for (h, w), lvls in sorted(sizes.items()):
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"]} | {idx[o] for o in objs}
                         for _ in range(w)] for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        padded = [a for a, f in shots.items()
                  if len(np.unique(f)) == 1 and int(f.flat[0]) == 5]
        bad += len(clashes) + len(padded)
        note = "OK" if not clashes else f"IDENTICAL {clashes}"
        if padded:
            note += f" / PAD-COLOURED {padded}"
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in lvls)}): {len(comps)} "
              f"compositions -- {note}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--field" in sys.argv:
        sys.exit(_field())
    if "--model" in sys.argv:
        sys.exit(_model())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(UndertaleSolver.main())
