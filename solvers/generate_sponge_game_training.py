"""Generate Phase-1 training data for the PuzzleScript game ps:sponge_game
("Sponge Game", Merge).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: the measured
mechanics, a native model of them, the exhaustive distance FIELD that model is
searched with, the proof that the plan is shortest, the proof that the model IS
the interpreter, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_sponge_game",
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
One level, one sponge, one target, and a pond. You shove the sponge; the sponge
DRINKS the water it is shoved into. You win by parking it on the target.

Everything below was MEASURED against the interpreter (``--verify`` is the
proof), not read off the .txt:

* **Water is ice.** ``[ up Player ] -> [ up Player up WentU ] again`` drops an
  invisible direction marker in the player's cell, and
  ``[ Water Player WentU ] -> [ Water up Player up WentU ] again`` re-applies
  that direction for as long as the player is standing on water. So a step onto
  water is a SLIDE that runs until the player lands on a dry cell or something
  stops it. ``[Went no Water] -> []`` is what ends it: the marker is deleted the
  moment it is off water.
* **The sponge does not slide.** No rule ever gives a Went marker to DrySponge.
  A push moves it exactly one cell, whatever it lands on.
* **The sponge drinks.** ``late [ DrySponge water ] -> [ DrySponge ]`` deletes
  the water under the sponge at the end of the turn it arrives. Water is never
  created by any rule, so the pond only ever SHRINKS, and the shrinking is
  monotone and irreversible -- which is what makes the RESET recovery arc below
  the only way back from a mistake.
* **A push always lands the player on dry ground.** The player steps into the
  cell the sponge just vacated, and the sponge dried that cell when it arrived
  there. That is why shoving the sponge along a row of water works at all: the
  sponge lays a dry towpath one cell ahead of the player.
* **You cannot push while you are sliding.**
  ``[ no Went > Player | DrySponge ] -> [ no Went > Player | > DrySponge ]``
  refuses to fire when the player's cell holds a Went marker, and a sliding
  player always has one. A slide that runs into the sponge therefore does not
  shove it: the player stops dead, ON the water, one cell short.
* **A player at rest never has a marker -- even a player resting on water.**
  This is the one genuinely surprising thing in the file, and the whole native
  model depends on it. Went is its OWN collision layer, so when a slide is
  stopped by a wall the player is blocked but its marker is not: the marker
  advances into the wall cell, ``[Went no Water]`` finds it standing on no water
  and deletes it. So the invisible state cleans itself up between turns, a
  player parked on water can push again, and a settled board is exactly
  ``(player, sponge, water)``. ``--verify`` asserts it on every board it checks.
  The single exception is a press that WINS: the interpreter breaks its
  ``again`` loop the moment the win condition fires, one iteration before the
  marker would have been swept up, so the winning frame can carry one. Nothing
  reads a board after that -- the episode is over.
* **Pressing into a wall, or into a sponge with a wall behind it, is a pure
  no-op** -- the marker is created, escapes into the blocking cell and is
  deleted, and the board compares equal. There is no ``restart``, no ``cancel``
  and no lose condition anywhere in the ruleset, so no press can ever end an
  episode.
* **ACTION5 is a no-op too.** An action turn sets no directional force, so
  neither the push rule nor the four ``[dir Player]`` rules can match, and with
  no marker on the board the four slide rules cannot either. `directions` says
  so and the enumeration never branches on it; ``--verify`` presses it from
  every state it samples and asserts the board is unchanged. (The exploration
  prefix still presses it -- a no-op press is honest recovery data.)
* **A slide is ONE frame, not an animation.** The adapter captures per-iteration
  snapshots only for ACTION presses, so a directional press returns exactly one
  frame however many cells the slide covered (measured: 1 frame on every press
  of the plan). There is no span to record and `record_spans` stays off -- the
  player crossing the pond is a jump in the tape because it is a jump in the
  game a live agent is shown.
* **But the game is a minefield.** ``--bfs``: of the 109080 boards reachable
  from the start, only 26381 can still be won. Three quarters of this game's
  state space is a dead end, it is reached by ordinary-looking presses, and
  nothing marks it -- no death, no restart, no visual tell. Drink the wrong
  puddle and the level is over while the frame still looks fine. That is what
  the RESET recovery arc exists for.

Why the sponge has to climb, and what the level is actually asking
-----------------------------------------------------------------
The target is at (1,5), in the dry strip along the top of the board, and the
pond fills rows 2-5 of columns 4-6 beneath it. There are exactly TWO winning
presses in the whole 109080-board space (counted over the enumeration, 59
winning edges in all):

    up    with the sponge at (2,5) and the player at (3,5)   -- 42 of them
    left  with the sponge at (1,6) and the player at (1,7)   -- 17 of them

A right-push into the target would need the player standing at (1,3), which is
a wall, so the third direction does not exist. Both surviving routes need the
sponge lifted UP out of the pond, and every cell under it is water, and a player
standing on water cannot push. The only reason it is possible at all is the
towpath: an up-push dries the cell the sponge LEAVES, and that cell is exactly
where the player has to stand for the next one, so the pair climbs the column
one rung at a time. The 23-press plan is the cheapest way to build the first
rung -- most of it is spent laying dry ground the player will need later.

The native model, and why the interpreter is not searched directly
------------------------------------------------------------------
109080 states at ~2.6 ms per interpreter step is a ~24-minute enumeration and
tens of thousands of full grid snapshots; the same enumeration over `_Board`
takes about a third of a second. So the search runs on a native model of the
bullets above -- a state is ``(player cell, sponge cell, water bitmask)`` -- and
``--verify`` is the receipt: it re-seats sampled model states onto the real
engine, presses all five actions, and compares the resulting board and win flag
against the model's answer, asserting no marker survives. Run it as
``--verify full`` to check every reachable state instead of a sample. That has
been run: 545105 presses over all 109021 playable boards, not one disagreement,
in 630 seconds.

The 59 already-WON boards are the one thing it does not press from, and the
reason is the same ``again``-loop shortcut that leaves a marker behind: once the
win holds, the interpreter stops re-running the rules, so a press that should
slide moves one cell and stops. The engine is not running the mechanic there.
Nothing ever presses from a won board either -- the recorder, the exploration
prefix and the field sweep all treat it as terminal, because it is.

Because the pond only shrinks, the water bitmask is monotone and the space is
finite and small, so the model is enumerated EXHAUSTIVELY rather than searched:
a forward BFS for the whole reachable component and a backward BFS from the won
boards give the exact distance-to-win of every state. That buys the same three
things it buys ps:slidyyyyyyy:

* a plan that is provably SHORTEST (no heuristic, no weight, no node cap that
  could quietly bite): 23 presses, of which exactly one step has a tie
  (``down``/``left`` are both on a shortest path at step 5);
* exact optimal-action SETS for free -- ``dist(succ) == dist - 1`` -- rather
  than inferred ones. Every expert step is labelled;
* a PROOF of which boards can still win, which is what turns "the search gave
  up" into the 82699-state dead zone reported above.

Rendering
---------
Two sprites are redrawn and one colour name changed in
``data/puzzlescript_games/Sponge_Game.txt`` (the comment at the top of that file
is the full account; ``--audit`` is the check, and it FAILS on the unfixed
file). As shipped, DrySponge was a bare colour name -- an opaque 5x5 fill on the
TOP collision layer -- so ``drysponge on target``, the only winning board this
game has, rendered as the same yellow square as a sponge anywhere else and the
win was invisible. ``player on target`` collided the same way. Target now owns
the four corner pixels, which neither occluder paints, and DrySponge opens them.
Target also moves DarkBlue -> Purple: ``_COLOR_NAME_TO_ARC`` maps "darkblue" and
"blue" to the SAME ARC index 9, which is Water's colour and the Player's fourth,
so the goal was drawn in the pond's exact palette entry.

Optimal-action sets
-------------------
``optsets[i]`` is every press ``d`` with ``dist(succ(state_i, d)) == dist(
state_i) - 1``, read straight off the exhaustive distance field, so it is exact
rather than inferred: a set is the complete list of shortest continuations, and
a press missing from it provably costs at least one more. ``--ties`` re-derives
every label independently, by taking each candidate press and running a FRESH
enumeration from the state it lands in.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the plan
replays to a guaranteed WIN. The RESET is doing real work on this game -- three
quarters of the reachable space cannot be won, the sponge drinking a puddle is
irreversible, and the prefix has a large chance of stranding the level.

There is no ``plan_cache_path``: the whole enumeration is a third of a second,
which is cheaper than the staleness risk a cached entry would carry. It is paid
once per process and every later seed replays the memoized plan at its own
rotation -- 50 seeds is about 10 seconds end to end, all 50 winning.

CLI
---
    --plans      the plan, its tie coverage, replayed through the interpreter
    --bfs        the exhaustive reachable-state report (winnability + dead ends)
    --ties       re-derive every optimal-action label by independent re-solve
    --verify     model vs. interpreter; ``--verify full`` checks every state
    --audit      assert every cell composition renders distinctly
"""

from __future__ import annotations

import random
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                       # noqa: E402
    PSAStarSolver, PSExpert, Plan,
)

GAME_NAME = "Sponge_Game"
GAME_ID = "puzzlescript_sponge_game"

#: The four presses, in the order that breaks every tie -- fixed, so a plan
#: re-derived in another process is byte-identical.
DIRS = ("up", "down", "left", "right")
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """One level's static geometry, plus the mechanic, plus the exhaustive
    distance field over it.

    A state is ``(player, sponge, water)``: two cell indices (``r * w + c``) and
    a BITMASK of the cells that still hold water, bit ``i`` for cell ``i``. The
    mask is a plain Python int, so it is canonical, hashable and free to copy at
    any board size, and no water-cell index table has to be threaded around.

    Nothing else in the game is dynamic: no rule creates or destroys a Wall, a
    Target or the Player, water is only ever DELETED (by the sponge), and the
    Went markers -- the one thing that would not fit in this state -- are proven
    absent from every settled board anything ever plans from (``--verify``; the
    lone exception is a board that has already been WON, which is terminal).
    """

    __slots__ = ("h", "w", "wall", "targets", "edge", "_fields")

    def __init__(self, h: int, w: int, walls, targets):
        self.h, self.w = h, w
        self.wall = [False] * (h * w)
        for i in walls:
            self.wall[i] = True
        self.targets = tuple(sorted(targets))
        # ``edge[i][di]`` is the neighbour of cell ``i`` in direction ``di``, or
        # -1 off the board. Precomputed once: the enumeration reads it a few
        # million times.
        self.edge = []
        for r in range(h):
            for c in range(w):
                row = []
                for d in DIRS:
                    dr, dc = _DELTA[d]
                    nr, nc = r + dr, c + dc
                    row.append(nr * w + nc if 0 <= nr < h and 0 <= nc < w
                               else -1)
                self.edge.append(tuple(row))
        #: ``[(seen, dist), ...]`` -- one exhaustive sweep per state it was
        #: asked from. See `field`.
        self._fields: list = []

    # -- the mechanic --------------------------------------------------------
    def step(self, state, di: int):
        """The state after pressing ``DIRS[di]``, or ``state`` itself if the
        press does nothing (a non-move: no shortest path contains one).

        Three cases, in the order the ruleset resolves them:

        1. the sponge is in the way -> PUSH it one cell (and it drinks whatever
           water is there), the player takes its place. Refused, and the whole
           press is a no-op, when the cell behind the sponge is a wall or off
           the board;
        2. a wall is in the way -> nothing happens;
        3. otherwise the player steps, and then SLIDES on for as long as it
           keeps landing on water, stopping when it lands on a dry cell or when
           the next cell is a wall or the sponge. Note which way round that is:
           the player stops ON the water it was sliding over, and cannot push
           the sponge it stopped against, because it stopped mid-slide.
        """
        p, s, water = state
        n1 = self.edge[p][di]
        if n1 < 0 or self.wall[n1]:
            return state
        if n1 == s:
            n2 = self.edge[s][di]
            if n2 < 0 or self.wall[n2]:
                return state
            return (n1, n2, water & ~(1 << n2))
        cur = n1
        while (water >> cur) & 1:
            nxt = self.edge[cur][di]
            if nxt < 0 or self.wall[nxt] or nxt == s:
                break
            cur = nxt
        return (cur, s, water)

    def won(self, state) -> bool:
        return self.targets == (state[1],)

    # -- the exhaustive field ------------------------------------------------
    def field(self, state):
        """``(seen, dist)`` for the component ``state`` lives in: every board
        reachable from it, and the exact presses-to-win of each board that can
        still be won. Boards in ``seen`` but not in ``dist`` are PROVEN dead.

        Sweep 1 is a forward BFS over `step`; a won board is terminal and is
        never expanded (the episode ends there). Sweep 2 is a backward BFS from
        the won boards over the reverse edges sweep 1 recorded. There is no
        node cap and no heuristic: the component is finite because the water
        mask only ever loses bits and the two pieces have finitely many cells,
        and it is small enough to hold whole (see ``--bfs``).

        Cached per board, keyed by nothing: a later query is answered by
        whichever completed sweep already contains the state, and only a state
        no sweep has seen pays for another one. Everything the recorder asks
        about is reachable from the level start by definition, so in practice
        exactly one sweep is ever run.
        """
        for seen, dist in self._fields:
            if state in seen:
                return seen, dist
        ids = {state: 0}
        states = [state]
        rev: list = [[]]
        queue = deque([0])
        while queue:
            u = queue.popleft()
            here = states[u]
            if self.won(here):
                continue                      # a won board is terminal
            for di in range(4):
                nxt = self.step(here, di)
                if nxt == here:
                    continue
                v = ids.get(nxt)
                if v is None:
                    v = len(states)
                    ids[nxt] = v
                    states.append(nxt)
                    rev.append([])
                    queue.append(v)
                rev[v].append(u)
        depth = [-1] * len(states)
        queue = deque()
        for v, st in enumerate(states):
            if self.won(st):
                depth[v] = 0
                queue.append(v)
        while queue:
            v = queue.popleft()
            for u in rev[v]:
                if depth[u] < 0:
                    depth[u] = depth[v] + 1
                    queue.append(u)
        dist = {states[v]: d for v, d in enumerate(depth) if d >= 0}
        seen = set(ids)
        self._fields.append((seen, dist))
        return seen, dist

    def optimal(self, dist, state):
        """``[(direction index, successor), ...]`` for every press on a shortest
        path from ``state``. Ties come out in ``DIRS`` order."""
        rest = dist.get(state)
        if not rest:
            return []
        out = []
        for di in range(4):
            nxt = self.step(state, di)
            if nxt != state and dist.get(nxt, -1) == rest - 1:
                out.append((di, nxt))
        return out

    def solve(self, state) -> "Plan | None":
        """The shortest press sequence from ``state``, with the EXACT optimal
        set at every step, or None if this board can no longer be won."""
        _seen, dist = self.field(state)
        if state not in dist:
            return None
        presses, optsets = [], []
        cur = state
        while not self.won(cur):
            best = self.optimal(dist, cur)
            if not best:                      # unreachable: dist>0 has a step
                return None
            presses.append(DIRS[best[0][0]])
            optsets.append([DIRS[di] for di, _ in best])
            cur = best[0][1]
        return Plan(presses, optsets)


# ---------------------------------------------------------------------------
# The expert
# ---------------------------------------------------------------------------

class SpongeExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `_Board` field.

    The base class keeps the memo, the level scoping and the recording contract;
    only the strategy underneath changes, the way `PSEnumExpert` replaces it.
    Here the strategy is "read the live board, enumerate its component, hand
    back the shortest plan and its exact tie sets", so `heuristic` is never
    called and asserts rather than returning a number nothing would use.

    `_search` reads the ENGINE's current grid every time, so a re-plan from an
    arbitrary state (the state a recovery prefix left behind) is answered
    exactly and not by patching up a stored path.
    """

    #: ACTION5 is read by no rule (see the module docstring and ``--verify``);
    #: branching on it would add a fifth of the enumeration for self-loops.
    directions = list(DIRS)

    #: `_key` is the dynamic objects only (player, sponge, water, and the
    #: markers for safety), which is canonical WITHIN a level but not across
    #: them -- walls and targets are static per level and would differ between
    #: levels if this game ever grew a second one.
    scope_by_level = True

    def setup(self) -> None:
        self.player_ids = set(self.game._engine._player_indices)
        self.sponge_ids = set(self.g.resolve_object_name("drysponge"))
        self.water_ids = set(self.g.resolve_object_name("water"))
        self.wall_ids = set(self.g.resolve_object_name("wall"))
        self.target_ids = set(self.g.resolve_object_name("target"))
        self.went_ids = set()
        for name in ("wentu", "wentd", "wentr", "wentl"):
            self.went_ids |= set(self.g.resolve_object_name(name))
        self.dyn_ids = (self.player_ids | self.sponge_ids | self.water_ids
                        | self.went_ids)
        #: `_Board`s by STATIC signature (see `read`), not by level index.
        self._boards: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "SpongeExpert reads an exact distance field; heuristic is unused")

    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn))

    # -- engine <-> model ----------------------------------------------------
    def read(self, eng):
        """``(board, state, markers)`` for the engine's current grid.

        Boards are cached by their STATIC signature (dimensions, walls,
        targets), so the sweeps never rebuild the edge table and one board is
        shared by every state of its level. ``markers`` is the count of live
        Went objects. Every settled board has none EXCEPT one that has already
        been won: the interpreter breaks its ``again`` loop the instant the win
        condition fires, one iteration before ``[Went no Water]`` would have
        swept the marker up. The episode ends on that board, so nothing plans
        from it -- `_search` checks the win first and refuses any other board
        that carries a marker rather than modelling a state it has never been
        shown (see ``--verify``)."""
        h, w = eng.height, eng.width
        walls, targets = [], []
        player = sponge = None
        water = 0
        markers = 0
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                i = r * w + c
                if cell & self.wall_ids:
                    walls.append(i)
                if cell & self.target_ids:
                    targets.append(i)
                if cell & self.water_ids:
                    water |= 1 << i
                if cell & self.player_ids:
                    player = i
                if cell & self.sponge_ids:
                    sponge = i
                if cell & self.went_ids:
                    markers += 1
        sig = (h, w, tuple(walls), tuple(targets))
        board = self._boards.get(sig)
        if board is None:
            board = self._boards[sig] = _Board(h, w, walls, targets)
        return board, (player, sponge, water), markers

    def seat(self, eng, board, state) -> None:
        """Write a model state back onto the engine as a settled board. Used
        only by ``--verify``, which is the whole point of it: a decoder that is
        not exactly `read`'s inverse would silently check a different game, so
        the two are kept next to each other."""
        bg = self.bg_id
        p, s, water = state
        eng.grid = [[{bg} for _ in range(board.w)] for _ in range(board.h)]
        for i, is_wall in enumerate(board.wall):
            if is_wall:
                eng.grid[i // board.w][i % board.w] |= self.wall_ids
        for i in board.targets:
            eng.grid[i // board.w][i % board.w] |= self.target_ids
        for i in range(board.h * board.w):
            if (water >> i) & 1:
                eng.grid[i // board.w][i % board.w] |= self.water_ids
        eng.grid[p // board.w][p % board.w] |= self.player_ids
        eng.grid[s // board.w][s % board.w] |= self.sponge_ids
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()

    def _search(self, eng) -> "Plan | None":
        board, state, markers = self.read(eng)
        if state[0] is None or state[1] is None:
            return None                       # no player, or no sponge left
        if board.won(state):
            return Plan([], [])
        if markers:
            # Never happens on a board anything plans from: the ONLY settled
            # board that keeps a marker is one that has already been won (see
            # `read` and ``--verify``), and that is caught on the line above.
            return None
        if len(board.targets) != 1:
            # Not reachable on the shipped level (one target, one sponge) and
            # not a crash if the level is ever edited: with a second target the
            # single sponge cannot satisfy `All Target on DrySponge` at all.
            return None
        return board.solve(state)


# ---------------------------------------------------------------------------
# The solver
# ---------------------------------------------------------------------------

class SpongeGameSolver(PSAStarSolver):
    game_id = GAME_ID
    game_name = GAME_NAME
    expert_cls = SpongeExpert

    #: `games/ps:sponge_game/ps:sponge_game.py` is a plain passthrough (it
    #: constructs the adapter and nothing else), so there is nothing to gain by
    #: routing through it -- but it IS what a live agent is handed, so if that
    #: wrapper ever grows a patch this must be set to ``"ps:sponge_game"``.
    game_module_id = ""

    #: Unused: the expert enumerates an exact field rather than searching under
    #: a heuristic, so there is no weight to trade and no node budget that could
    #: quietly bite. Left at the base values so nothing reads a lie off them.
    weight = 1
    node_cap = 400_000

    #: The plan is 23 presses; the rest is room for the exploration prefix and
    #: the re-plan after it. Stays well under the adapter's own 200-step
    #: per-level budget, which `set_level` resets before the plan starts anyway.
    max_steps = 120

    def prepare_expert(self, game, expert) -> None:
        """Sweep the level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time ~0.3s visible as startup
        rather than as a mysteriously slow first seed."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new():
    solver = SpongeGameSolver()
    game, expert, solvable = solver._ensure(0)
    return solver, game, expert, solvable


def _plans() -> int:
    """The plan for every level, replayed through the REAL interpreter -- the
    end-to-end test of the native model, the field and the tie labelling at
    once. A plan that does not win the interpreter fails the report."""
    _solver, game, expert, _solvable = _new()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        t0 = time.time()
        plan = expert.plan(eng, level)
        dt = time.time() - t0
        head = f"level {level}: {eng.height}x{eng.width}"
        if plan is None:
            print(f"{head} -- UNWINNABLE (see --bfs)")
            bad += 1
            continue
        for d in plan:
            eng.step(d)
        won = eng.check_win()
        bad += 0 if won else 1
        sets = getattr(plan, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(plan)} presses (budget {game._max_steps}, {room}), "
              f"{ties}/{len(plan)} steps with a tie set, {dt:.2f}s -- "
              f"interpreter says {'WIN' if won else 'NOT WON'}")
        print(f"  {' '.join(plan)}")
    print("plans clean" if not bad else f"PLANS FAILED: {bad} level(s)")
    return 0 if not bad else 1


def _bfs() -> int:
    """The exhaustive reachable-state report -- and therefore the proof that the
    plan is SHORTEST: the sweep terminates having generated every board the
    model admits, so the backward distance field is exact.

    It is also the DEAD-END report, which is the headline fact about this game.
    Nothing here kills you and nothing restarts the level, but the sponge
    drinking a puddle is irreversible, so most of the space is boards that look
    perfectly ordinary and can never be won again."""
    _solver, game, expert, _solvable = _new()
    for level in range(game.n_levels):
        game.set_level(level)
        board, state, _m = expert.read(game._engine)
        # A FRESH board, so the printed time is the sweep and not a cache hit on
        # the one `prepare_expert` already ran at startup.
        fresh = _Board(board.h, board.w,
                       [i for i, x in enumerate(board.wall) if x], board.targets)
        t0 = time.time()
        seen, dist = fresh.field(state)
        dt = time.time() - t0
        board = fresh
        wins = sum(1 for s in seen if board.won(s))
        d = dist.get(state)
        verdict = (f"shortest {d} presses" if d is not None
                   else "UNWINNABLE (no won board is reachable)")
        print(f"level {level}: {len(seen)} reachable boards "
              f"({len(dist)} can still win, {len(seen) - len(dist)} stranded, "
              f"{wins} already won), {dt:.2f}s -- {verdict}")
    return 0


def _ties() -> int:
    """Re-derive every optimal-action label the independent way: take each
    candidate press and run a FRESH exhaustive sweep from the board it lands in.
    A press is optimal iff ``1 + dist(successor)`` equals the presses remaining,
    so this checks the distance field and the plan-following against a sweep
    that shares no state with the one that produced the labels."""
    _solver, game, expert, _solvable = _new()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state, _m = expert.read(game._engine)
        plan = expert.plan(game._engine, level)
        if plan is None:
            print(f"level {level}: no plan (skipped)")
            continue
        for pi, (taken, claimed) in enumerate(zip(plan, plan.optsets)):
            remaining = len(plan) - pi
            measured = []
            for di, d in enumerate(DIRS):
                nxt = board.step(state, di)
                if nxt == state:
                    continue
                if board.won(nxt):
                    cost = 1
                else:
                    fresh = _Board(board.h, board.w,
                                   [i for i, x in enumerate(board.wall) if x],
                                   board.targets)
                    _s2, d2 = fresh.field(nxt)
                    if nxt not in d2:
                        continue
                    cost = 1 + d2[nxt]
                if cost == remaining:
                    measured.append(d)
            if measured != list(claimed):
                print(f"  level {level} step {pi}: labelled {list(claimed)} "
                      f"but measured {measured}")
                bad += 1
            if taken not in claimed:
                print(f"  level {level} step {pi}: took {taken}, not in its "
                      f"own optimal set {list(claimed)}")
                bad += 1
            state = board.step(state, DIRS.index(taken))
        print(f"level {level}: {len(plan)} steps verified")
    print("ties clean" if not bad else f"TIES FAILED: {bad} mismatches")
    return 0 if not bad else 1


def _verify(argv) -> int:
    """The receipt for the native model: seat a model board onto the REAL
    interpreter, press all FIVE actions, and compare the board and the win flag
    against `_Board.step`.

    Three things are asserted per press, and the third is the one that licenses
    the model's state at all:

    * the resulting ``(player, sponge, water)`` matches -- ``step`` is the
      mechanic;
    * ``check_win()`` matches ``won()``;
    * NO Went object survives the press, unless the press WON. The markers are
      transparent, they are the only hidden state this game has, and the whole
      native model rests on them being gone by the time the turn settles. The
      one exception is measured, not assumed: the interpreter breaks its
      ``again`` loop the instant the win fires, one iteration before
      ``[Went no Water]`` would have deleted the marker, so a won board can
      carry one. The episode ends there and nothing plans from it.

    ACTION5 is included in the five and is asserted to leave the board
    untouched, which is what lets `directions` drop it.

    ``--verify`` samples (default 4000 boards, reproducibly, from the whole
    reachable component); ``--verify full`` checks every one of them, which is
    545105 presses and about ten minutes for this game. ``--verify <N>`` sets
    the sample size."""
    _solver, game, expert, _solvable = _new()
    eng = game._engine
    arg = next((a for a in argv if a not in ("--verify",)
                and not a.startswith("--")), None)
    full = arg == "full"
    want = 4000 if arg is None or full else int(arg)

    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, start, _m = expert.read(eng)
        seen, _dist = board.field(start)
        # ALREADY-WON boards are excluded, and that is a fact about the game
        # rather than a convenience. The interpreter breaks its ``again`` loop
        # the instant the win condition holds, so on a won board a press that
        # should slide moves ONE cell and stops -- the engine is no longer
        # running the mechanic. Nothing ever presses from there: `record_level`
        # and `_reset_prefix` both stop on the win, `field` treats a won board
        # as terminal, and the adapter has ended the episode.
        won = [k for k in seen if board.won(k)]
        pool = sorted(k for k in seen if not board.won(k))
        live = len(pool)
        if not full and len(pool) > want:
            pool = random.Random(12345).sample(pool, want)
        t0 = time.time()
        checked = 0
        for state in pool:
            for di, d in enumerate(("up", "down", "left", "right", "action")):
                expert.seat(eng, board, state)
                eng.step(d)
                got_board, got_state, markers = expert.read(eng)
                want_state = (state if d == "action"
                              else board.step(state, di))
                checked += 1
                if got_state != want_state:
                    print(f"  level {level} {state} {d}: engine {got_state}, "
                          f"model {want_state}")
                    bad += 1
                if eng.check_win() != board.won(want_state):
                    print(f"  level {level} {state} {d}: engine win "
                          f"{eng.check_win()}, model {board.won(want_state)}")
                    bad += 1
                if markers and not board.won(want_state):
                    print(f"  level {level} {state} {d}: {markers} Went "
                          f"marker(s) survived the press")
                    bad += 1
                if got_board is not board:
                    print(f"  level {level} {state} {d}: statics changed")
                    bad += 1
        dt = time.time() - t0
        scope = "EVERY playable board" if full else f"{len(pool)} sampled boards"
        print(f"level {level}: {checked} presses over {scope} "
              f"(of {live} playable, {len(won)} already-won boards skipped, "
              f"{len(seen)} reachable), {dt:.1f}s")
    print("verify clean -- the model IS the interpreter" if not bad
          else f"VERIFY FAILED: {bad} disagreements")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct.

    Whole 64x64 frames of uniform boards are compared rather than one cell out
    of a mixed board: `_render_frame` centre-pads a non-square board, so
    indexing a cell by ``cell_px * r`` silently reads the wrong pixels (the
    ps:explod lesson). Two uniform boards render identically iff their cells do.

    ``sponge_on_target`` is the WIN frame and the reason the .txt is patched: as
    shipped it rendered as a bare sponge, i.e. the game's only winning board was
    invisible. ``sponge_on_water`` is here for completeness -- the late rule
    dries the sponge's cell in the turn it arrives, so no settled board holds
    it -- and the Went markers cannot appear at all (``--verify``)."""
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, _expert, _solvable = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    comps = {
        "floor": (),
        "wall": ("wall",),
        "water": ("water",),
        "target": ("target",),
        "player": ("player",),
        "player_on_water": ("player", "water"),
        "player_on_target": ("player", "target"),
        "sponge": ("drysponge",),
        "sponge_on_water": ("drysponge", "water"),
        "sponge_on_target": ("drysponge", "target"),
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
        print(f"{h}x{w} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} "
              f"compositions -- {note}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_plans())
    if "--bfs" in sys.argv:
        sys.exit(_bfs())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--verify" in sys.argv:
        sys.exit(_verify(sys.argv[sys.argv.index("--verify") + 1:]))
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(SpongeGameSolver.main())
