"""Generate Phase-1 training data for the PuzzleScript game
ps:five_pulloban_puzzles ("Five Pulloban Puzzles" by Croubble -- crates you can
only ever DRAG behind you, on five islands joined by bridges that open one at a
time and burn behind you).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, a native model of the interpreter, the exact value function over
it, and the optimal-action labeller.

Each solved seed yields one single-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_five_pulloban_puzzles",
      "levels": [
        {
          "level_id": 0,
          "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
          "actions":      [a_0, a_1, ..., a_{T-1}], # length T
        }
      ]
    }

actions[0] is the RESET that produced obs[0]; actions[i] for i>=1 is the action
that took the agent from obs[i-1] to obs[i]. The recorded index is the *screen*
action (post rotation/flip remap), i.e. the button an agent presses in the
presented view, so replaying the recorded actions reproduces the recorded frames
exactly.

The game
--------
There is ONE level: a 55x10 map cut by ``flickscreen 11x10`` into five screens,
each a self-contained puzzle. The camera shows only the screen the player is
standing in, so the agent sees one 11x10 island at a time.

  * **EVERY crate is a PULL crate.** The only piece rule that can ever fire is
    ``[ < Player | PullCrate ] -> [ < Player | < PullCrate ]``: a crate one cell
    BEHIND the player follows it when the player steps away. (``Crate`` and its
    push rule exist in the source but no level character maps to them, so
    nothing in this game is ever pushed.) Concretely, to shift a crate at ``X``
    one cell in direction ``d`` the player must stand at ``X+d`` and press
    ``d``, ending at ``X+2d``; ``X+2d`` has to be free or nothing moves at all.

  * **WALKING IS NOT INERT**, which is what makes this not a sokoban. Any step
    that happens to move away from an adjacent crate drags it, whether you
    meant to or not -- so there is no such thing as a side-effect-free route
    across a board, and macro (walk-to-then-act) search does not apply. The
    search here is over PRIMITIVE presses.

  * **Pressing into a crate does nothing.** Crates share the player's collision
    layer and no rule pushes them, so the move is simply refused (and the crate
    behind you does not move either -- it has nowhere to go).

  * **AN ISLAND IS SOLVED THE INSTANT ALL OF ITS TARGETS ARE COVERED.** Each
    turn a ``Pulse`` floods the player's wall-free region; a target inside it
    with no crate raises ``Failure``, and with none raised the region raises
    ``Success``, which DELETES every crate, target and respawn token in it.
    There is no partial credit and no undoing it.

  * **SUCCESS OPENS THE BRIDGE.** ``Raft`` cells are walls carrying a dark ring;
    on Success every raft the pulse reached (it chains along a whole raft run)
    drops its wall and becomes a walkable ``ActiveRaft`` plank. Standing on a
    plank the player emits no pulse at all, so nothing is judged mid-crossing --
    but the plank is also a ``Save`` cell, which RESETS every crate on the board
    to its starting square. **Stepping off the far end onto the next island
    raises that island's Failure, which turns every plank back into plain wall.**
    The crossing is one-way and the bridge is gone for good.

So the level is five pull-puzzles and four crossings in a fixed order, and the
whole thing is one trajectory. The win condition is ``no Target``, i.e. all five
islands finished.

Two fixes to `data/puzzlescript_games/Five_Pulloban_Puzzles.txt`
---------------------------------------------------------------
**The pulse could not turn a corner.** `PuzzleScriptAdapter._execute_late_single`
applies a rule ONE DIRECTION AT A TIME, each to fixpoint, and never re-runs the
rule over all four -- so ``late [Pulse | no Pulse no Wall] -> [Pulse|Pulse]``
flooded along straight runs and stopped at every bend. Two consequences, both
silent: an L-shaped island was only partly judged, so a target the flood never
reached could not raise ``Failure`` and the island "solved" with it still
uncovered (and its respawn token survived to be re-materialised by the next
Save); and a bridge whose raft run bends -- three of the four do -- only ever
opened its first straight leg, leaving the game unwinnable. Fixed in the .txt,
not the adapter, by wrapping the two spread rules in ``startloop``/``endloop``,
which the adapter DOES iterate to a fixpoint. That is what real PuzzleScript
does with any rule anyway, so it is a faithfulness fix, and it is scoped to this
file. See [[ps-engine-render-gotchas]].

**Palette / sprite collisions**, all of the kinds this tree keeps hitting:
``Target`` shipped ``DarkBlue`` and ``Wall`` ``Blue`` -- both ARC index 9, so
every goal square rendered as a piece of wall; and ``Target`` was a hollow ring
drawn exactly under the player's opaque cells, so a player standing on a target
hid it completely. Target is now a solid ``Purple`` square (shows through both
the crate's open middle and the player's transparent corners). ``ActiveRaft``
used ``LightBrown``, a colour name `_COLOR_NAME_TO_ARC` does not know, which
silently falls back to index 2; it is a solid ``Brown`` plank now. ``Raft`` was a
single centre pixel and is now a ring, so "this wall will become a bridge" is
legible. ``--audit`` renders all ten cell compositions this game can show and
asserts they are pairwise distinct.

Why a native model
------------------
`PSExpert`'s engine-blackbox A* is not used. The interpreter runs at ~520
presses/s here (the pulse re-floods the whole 55x10 board every single turn),
and the labels this corpus wants are EXACT optimal-action sets, which means a
distance-to-win FIELD rather than one A* path. The islands are tiny once you
look at them as state graphs -- 1.7k, 2.0k, 11k, 19k and 78k ``(player,
crates)`` states, ~112k in all -- so `Fields` enumerates every one of them and
builds the exact field by backward Dijkstra in about a second, and the
interpreter is left to CERTIFY the plan (``--plans`` replays it) rather than
produce it. `--selfcheck` fuzzes the model against the real interpreter, phase
transitions included.

Coverage
--------
The single level solves in **231 presses**, all of them labelled with their
exact optimal SET: 33 + 9 + 15 + 19 + 60 + 17 + 23 + 22 + 33 (solve, cross,
solve, ...), and only 4 of the 231 have a second right answer -- the boards are
corridors, so nearly every press is forced.

The plan is provably shortest, and ``--verify`` is the proof: it enumerates the
3251 states reachable from the start using nothing but `Fields.step`, BFSes back
from the win over the reversed graph with no island/terminal bookkeeping at all,
and checks that against `Fields` state by state. That is a genuinely independent
recomputation -- if the backward, island-by-island construction threaded a
terminal value wrongly it would still be self-consistent, and only this catches
it. Cold build is ~1.5s and caches to ``data/five_pulloban_puzzles_plans.json``.

The game folder's wrapper raises the adapter's 200-action budget to 700, without
which the shipped level is unwinnable for any agent (and the failure is silent:
the plan replays fine and the episode just never reaches WIN). See
`games/ps:five_pulloban_puzzles/ps:five_pulloban_puzzles.py`.

Usage
-----
    python solvers/generate_five_pulloban_puzzles_training.py --episodes 200 \\
        --out data/training_multi_level/five_pulloban_puzzles

    --selfcheck   fuzz the native model against the real interpreter
    --audit       check every cell composition renders distinctly
    --verify      re-derive the whole value function a second way
    --plans       print (and interpreter-verify) the plan
"""

from __future__ import annotations

import heapq
import itertools
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import (                    # noqa: E402
    PuzzleScriptAdapter, _render_cell_sprite, _render_frame)
from solvers.common.ps_astar import (                          # noqa: E402
    Plan, PSAStarSolver, PSExpert, _load_game_module)

GAME_NAME = "Five_Pulloban_Puzzles"
GAME_MODULE_ID = "ps:five_pulloban_puzzles"

#: Where the level's start plan is cached between processes. Building the exact
#: fields is the entire cost of generation and is seed-independent, so without
#: this file every `parallelize_generator` shard re-derives all five.
PLAN_CACHE = (Path(__file__).resolve().parent.parent
              / "data" / "five_pulloban_puzzles_plans.json")

#: The four live keys, in engine-direction form. ACTION is deliberately absent:
#: no rule in the .txt reads it, so it is a turn that does nothing (`selfcheck`
#: asserts that), and offering it to the search would only widen the branching.
KEYS = ("up", "down", "left", "right")

#: (dr, dc) per key index. Ordered so ``d ^ 1`` is the opposite direction, which
#: is how the pull rule finds the cell BEHIND the player.
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

INF = float("inf")


def _nbrs(cell):
    return [(cell[0] + dr, cell[1] + dc) for dr, dc in _DELTA]


def _components(cells: set) -> list:
    """4-connected components of ``cells``, each returned as a frozenset."""
    out, seen = [], set()
    for start in sorted(cells):
        if start in seen:
            continue
        comp, q = {start}, deque([start])
        seen.add(start)
        while q:
            for n in _nbrs(q.popleft()):
                if n in cells and n not in seen:
                    seen.add(n)
                    comp.add(n)
                    q.append(n)
        out.append(frozenset(comp))
    return out


# ---------------------------------------------------------------------------
# Static geometry of the level
# ---------------------------------------------------------------------------

class Board:
    """The level's fixed skeleton: five islands in play order, the raft run
    between each consecutive pair, and each island's starting crates/targets.

    All of it is read off the level's INITIAL grid and none of it changes: walls
    only ever appear or vanish on raft cells, and a raft cell is by construction
    not part of any island. That is what lets the fields below be built once and
    reused from any mid-game state -- `state_from_engine` locates the live
    player in this static map rather than re-deriving the map from a grid whose
    bridges have already burned.
    """

    def __init__(self, eng, parsed):
        idx = parsed.obj_name_to_idx
        cells = [(r, c) for r in range(eng.height) for c in range(eng.width)]
        free = {p for p in cells if idx["wall"] not in eng.grid[p[0]][p[1]]}
        rafts = {p for p in cells if idx["raft"] in eng.grid[p[0]][p[1]]}
        player = next(p for p in cells if idx["player"] in eng.grid[p[0]][p[1]])

        islands = _components(free)
        chains = _components(rafts)
        home = {p: i for i, isl in enumerate(islands) for p in isl}

        # Each raft run touches exactly two islands; the resulting graph is a
        # path, and the play order is that path walked from the player's island.
        link: dict = {}
        for chain in chains:
            touch = sorted({home[n] for p in chain for n in _nbrs(p) if n in home})
            assert len(touch) == 2, f"raft run touching {touch}, expected 2 islands"
            link[(touch[0], touch[1])] = chain
            link[(touch[1], touch[0])] = chain
        order = [home[player]]
        while len(order) < len(islands):
            nxt = [j for j in range(len(islands))
                   if (order[-1], j) in link and j not in order]
            assert len(nxt) == 1, f"island {order[-1]} has {len(nxt)} ways on"
            order.append(nxt[0])

        self.islands = [tuple(sorted(islands[i])) for i in order]
        self.chains = [tuple(sorted(link[(order[k], order[k + 1])]))
                       for k in range(len(order) - 1)]
        self.island_of = {p: k for k, isl in enumerate(self.islands) for p in isl}
        self.chain_of = {p: k for k, ch in enumerate(self.chains) for p in ch}
        self.crates0 = [frozenset(p for p in isl
                                  if idx["pullcrate"] in eng.grid[p[0]][p[1]])
                        for isl in self.islands]
        self.targets = [frozenset(p for p in isl
                                  if idx["target"] in eng.grid[p[0]][p[1]])
                        for isl in self.islands]
        self.bit = [{p: 1 << i for i, p in enumerate(isl)} for isl in self.islands]
        self.tmask = [sum(self.bit[k][t] for t in self.targets[k])
                      for k in range(len(self.islands))]
        self.crates0_mask = [sum(self.bit[k][p] for p in self.crates0[k])
                             for k in range(len(self.islands))]

    @property
    def n(self) -> int:
        return len(self.islands)


# ---------------------------------------------------------------------------
# Native model + exact distance-to-win field
# ---------------------------------------------------------------------------
#
# A global state is one of
#   ("solve", k, player, crate-mask)  -- k's targets are still up
#   ("walk",  k, player)              -- k is finished; crossing to k+1
#   ("done",)                         -- every target gone, i.e. WIN
#
# `Fields.step` is the interpreter's whole observable behaviour on these states
# and `Fields.value` is the exact number of presses left, so the optimal SET at
# any state is just the presses whose successor is one cheaper.

class Fields:
    """Exact distance-to-win for every state the game can reach.

    Built backwards, because that is the only direction in which the terminal
    values are known: the last island's Success ends the game, so its field is
    computed against a terminal of 0; the walk field before it is a Dijkstra
    whose sources are "step off the bridge into island k+1 and start solving
    it"; and that walk field is in turn the terminal for island k's own field.

    The islands are small enough that this is exhaustive rather than searched --
    1.7k to 78k ``(player, crates)`` states each, ~112k in total -- which is
    what makes the optimal-action sets EXACT rather than "at least as good as
    what the expert did".
    """

    def __init__(self, board: Board):
        self.b = board
        n = board.n
        self.solve: list = [None] * n
        self.walk: list = [None] * n          # walk[k]: island k -> island k+1
        self.solve[n - 1] = self._island_field(n - 1, {p: 0 for p in board.islands[n - 1]})
        for k in range(n - 2, -1, -1):
            self.walk[k] = self._walk_field(k)
            self.solve[k] = self._island_field(k, self.walk[k])

    # -- construction ---------------------------------------------------------
    def _island_field(self, k: int, terminal: dict) -> dict:
        """``(player, crate-mask) -> presses to win`` for island ``k``.

        ``terminal[p]`` is what remains to be paid once the island's Success
        fires with the player standing on ``p``. Solved configurations are not
        states here: Success deletes the crates and the targets in the same
        turn, so a state that reaches one is charged ``1 + terminal[p]`` and the
        island's graph ends there."""
        cells = self.b.islands[k]
        bit = self.b.bit[k]
        tmask = self.b.tmask[k]
        ncr = len(self.b.crates0[k])
        nb = {p: tuple((q if q in bit else None) for q in _nbrs(p)) for p in cells}

        preds: dict = {}
        heap: list = []
        for combo in itertools.combinations(cells, ncr):
            mask = 0
            for p in combo:
                mask |= bit[p]
            if (mask & tmask) == tmask:       # already solved: not a state
                continue
            occupied = set(combo)
            for p in cells:
                if p in occupied:
                    continue
                st = (p, mask)
                for d in range(4):
                    q = nb[p][d]
                    if q is None or (mask & bit[q]):
                        continue              # wall or crate: the press is refused
                    back = nb[p][d ^ 1]
                    nmask = mask
                    if back is not None and (mask & bit[back]):
                        nmask = (mask & ~bit[back]) | bit[p]     # it follows
                    if (nmask & tmask) == tmask:
                        heapq.heappush(heap, (1 + terminal[q], st))
                    else:
                        preds.setdefault((q, nmask), []).append(st)

        dist: dict = {}
        while heap:
            dv, st = heapq.heappop(heap)
            if st in dist:
                continue
            dist[st] = dv
            for prev in preds.get(st, ()):
                if prev not in dist:
                    heapq.heappush(heap, (dv + 1, prev))
        return dist

    def _walk_field(self, k: int) -> dict:
        """``cell -> presses to win`` while crossing from island ``k`` to ``k+1``.

        The walkable set is island ``k`` (emptied of crates and targets by its
        own Success) plus the raft run, every plank of which is open. Walking is
        inert here -- there is nothing left on the island to drag -- so this is
        a plain Dijkstra whose sources are the planks that touch island ``k+1``,
        each charged the one press onto the island plus the cost of solving it
        from that entry square. Island ``k+1``'s crates are guaranteed to be at
        their starting squares: every plank is a ``Save`` cell, so the crossing
        itself resets them."""
        cells = set(self.b.islands[k]) | set(self.b.chains[k])
        nxt = self.solve[k + 1]
        nxt_mask = self.b.crates0_mask[k + 1]
        heap: list = []
        for plank in self.b.chains[k]:
            for entry in _nbrs(plank):
                if self.b.island_of.get(entry) == k + 1:
                    cost = nxt.get((entry, nxt_mask), INF)
                    if cost < INF:
                        heapq.heappush(heap, (1 + cost, plank))
        dist: dict = {}
        while heap:
            dv, p = heapq.heappop(heap)
            if p in dist:
                continue
            dist[p] = dv
            for q in _nbrs(p):
                if q in cells and q not in dist:
                    heapq.heappush(heap, (dv + 1, q))
        return dist

    # -- dynamics -------------------------------------------------------------
    def step(self, st, d: int):
        """The state after pressing ``KEYS[d]``. A refused press returns ``st``
        itself, which is exactly what the interpreter does (the turn passes and
        nothing on the board moves)."""
        kind = st[0]
        if kind == "done":
            return st
        b = self.b
        if kind == "solve":
            _, k, p, mask = st
            bit = b.bit[k]
            q = (p[0] + _DELTA[d][0], p[1] + _DELTA[d][1])
            if q not in bit or (mask & bit[q]):
                return st
            back = (p[0] - _DELTA[d][0], p[1] - _DELTA[d][1])
            nmask = mask
            if back in bit and (mask & bit[back]):
                nmask = (mask & ~bit[back]) | bit[p]
            if (nmask & b.tmask[k]) == b.tmask[k]:
                return ("done",) if k == b.n - 1 else ("walk", k, q)
            return ("solve", k, q, nmask)
        _, k, p = st                                   # "walk"
        q = (p[0] + _DELTA[d][0], p[1] + _DELTA[d][1])
        if b.island_of.get(q) == k or b.chain_of.get(q) == k:
            return ("walk", k, q)
        if b.island_of.get(q) == k + 1:
            return ("solve", k + 1, q, b.crates0_mask[k + 1])
        return st

    def value(self, st) -> float:
        kind = st[0]
        if kind == "done":
            return 0
        if kind == "solve":
            return self.solve[st[1]].get((st[2], st[3]), INF)
        return self.walk[st[1]].get(st[2], INF)

    def optimal(self, st) -> list:
        """Every press that is on a shortest win from ``st``, as key indices.

        A refused press leaves the state alone and so scores ``value + 1``; it
        can never be in the set. Empty when the state is dead."""
        v = self.value(st)
        if v == INF:
            return []
        return [d for d in range(4) if 1 + self.value(self.step(st, d)) == v]

    def descend(self, st):
        """A shortest win from ``st`` as ``(presses, per-step optimal sets)``,
        or ``None`` if the state is dead. Ties are broken by key order, so the
        plan is deterministic and the disk cache stays stable."""
        if self.value(st) == INF:
            return None
        presses, optsets = [], []
        while st[0] != "done":
            best = self.optimal(st)
            presses.append(KEYS[best[0]])
            optsets.append([KEYS[d] for d in best])
            st = self.step(st, best[0])
        return presses, optsets


# ---------------------------------------------------------------------------
# Reading the live interpreter
# ---------------------------------------------------------------------------

def state_from_engine(eng, parsed, board: Board):
    """The global state the interpreter's grid is in.

    The phase is decided by where the player is in the STATIC map and by what is
    still on the board: standing on an island that still has targets is a solve
    phase, and anywhere else -- an emptied island, or a plank -- is a crossing.
    Both are unambiguous because the game is strictly one-way: a finished island
    never gets its targets back, and a burned bridge never re-opens."""
    idx = parsed.obj_name_to_idx
    player = None
    live_targets, live_crates = set(), set()
    for r in range(eng.height):
        for c in range(eng.width):
            cell = eng.grid[r][c]
            if idx["player"] in cell:
                player = (r, c)
            if idx["target"] in cell:
                live_targets.add((r, c))
            if idx["pullcrate"] in cell:
                live_crates.add((r, c))
    if not live_targets:
        return ("done",)
    k = board.island_of.get(player)
    if k is not None and (live_targets & set(board.islands[k])):
        mask = 0
        for p in live_crates & set(board.islands[k]):
            mask |= board.bit[k][p]
        return ("solve", k, player, mask)
    gap = board.chain_of.get(player, k)
    # The islands are finished strictly in order, so the only way to be off a
    # live board with targets still standing is to be crossing to the next one.
    assert gap is not None and gap < board.n - 1, (
        f"player at {player} is past the last bridge with targets still up")
    return ("walk", gap, player)


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class FivePullobanExpert(PSExpert):
    """`Fields` plugged into `PSExpert` as its search strategy.

    Only `_search` is overridden, so the in-memory memo, the on-disk plan cache
    (with its staleness check) and the snapshot/restore discipline around them
    stay the shared ones -- this file carries no copy of any of that. The
    engine is never stepped by the planner; it only certifies the finished plan.
    """

    directions = list(KEYS)
    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        self._fields: Fields | None = None
        self._board: Board | None = None
        self._how: dict = {}
        self._level: int | None = None

    def build(self):
        """The static board and its exact fields, built once and cached.

        Read off a private pristine adapter rather than off the live engine:
        `_search` can be called from a mid-game grid whose bridges have already
        burned, and the skeleton has to be the level's ORIGINAL one."""
        if self._fields is None:
            fresh = PuzzleScriptAdapter(GAME_NAME, seed=0)
            fresh.set_level(0)
            self._board = Board(fresh._engine, fresh._game)
            self._fields = Fields(self._board)
        return self._board, self._fields

    def _search(self, eng):
        """`PSExpert`'s strategy hook: read a model state off the interpreter's
        grid and walk the exact field down from it. Returns a `Plan` (presses
        plus their optimal sets), which is what the shared memo and disk cache
        round-trip."""
        board, fields = self.build()
        found = fields.descend(state_from_engine(eng, self.g, board))
        self._how[self._level] = "exact field" if found else "dead state"
        if found is None:
            return None
        presses, optsets = found
        return Plan(presses, optsets)

    def plan(self, eng, level: int | None = None):
        """`PSExpert.plan`, remembering which level it is planning so `describe`
        can report against it. The base does all the actual work."""
        self._level = level
        return super().plan(eng, level)

    def describe(self, level) -> str:
        """How the plan was found (for `--plans`); "cached" when it came off
        disk and nothing ran."""
        return self._how.get(level, "cached")


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class FivePullobanPuzzlesSolver(PSAStarSolver):
    game_id = "puzzlescript_five_pulloban_puzzles"
    game_name = GAME_NAME
    #: The wrapper raises the adapter's 200-action budget, without which the
    #: level cannot be finished at all -- and that patched adapter is what
    #: `game_envs` hands a live agent, so it has to be what this generator tapes.
    game_module_id = GAME_MODULE_ID
    expert_cls = FivePullobanExpert

    #: The plan is 231 presses and the recorder needs room for the whole of it;
    #: the exploration prefix that precedes it ends in a RESET, which zeroes the
    #: adapter's own counter, so only the post-reset plan is charged against the
    #: wrapper's 700.
    max_steps = 400

    def prepare_expert(self, game, expert) -> None:
        """Build the fields before `discover_solvable` asks for a plan -- the
        same work either way, but it makes the one-time cost visible as startup
        rather than as a mysteriously slow first seed."""
        expert.build()

    def optimal_for(self, expert, level: int, plan, pi: int):
        """The exact optimal press set at this step, off the plan's own optsets
        (which round-trip through the disk cache, so a warm start never rebuilds
        the fields). Falls back to the press about to be taken so no expert step
        ever ships unlabelled -- `train_policy` v2 supervises ``optimal`` only,
        so a step without one contributes nothing to the loss."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets) and sets[pi]:
            return sets[pi]
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Model self-check (fuzz against the real interpreter)
# ---------------------------------------------------------------------------

def selfcheck(trials: int = 40, steps: int = 70, verbose: bool = True) -> int:
    """Audit the claim the whole solver rests on: the native model reproduces
    the interpreter EXACTLY, phase transitions included.

    Rollouts start from a random prefix of the canonical plan -- which is the
    only cheap way to reach the later islands at all -- and then press at
    random, so every island, both crossings' ends and the deletion/activation
    turns are all exercised. Four claims per press, and they are the four the
    field rests on: the settled ``(phase, island, player, crates)`` agrees with
    the model; the interpreter's win flag agrees with the model's ``done``; a
    press the model refuses leaves the grid byte-identical; and ACTION -- which
    no rule in the .txt reads -- never changes anything, so a rule that started
    reading it would show up here as a divergence."""
    game = _load_game_module(GAME_MODULE_ID).make_game(seed=0)
    eng = game._engine
    expert = FivePullobanExpert(game)
    board, fields = expert.build()
    game.set_level(0)
    start = state_from_engine(eng, game._game, board)
    plan = fields.descend(start)[0]

    # The presses at which the plan changes phase -- an island's Success (which
    # deletes its board and opens a bridge) and the step off the far plank
    # (which burns it). Each happens exactly once per island, so a uniformly
    # random prefix reaches them rarely; half the rollouts are seeded a few
    # presses short of one instead, and press at random THROUGH it.
    st, marks = start, []
    for i, direction in enumerate(plan):
        nxt = fields.step(st, KEYS.index(direction))
        if nxt[0] != st[0] or nxt[1:2] != st[1:2]:
            marks.append(i)
        st = nxt

    bad = 0
    seen_phases: set = set()
    refused = wins = opened = crossed = 0
    for t in range(trials):
        rng = random.Random(f"pulloban:selfcheck:{t}")
        game.set_level(0)
        cut = (max(0, rng.choice(marks) - rng.randrange(4)) if t % 2 and marks
               else rng.randrange(len(plan) + 1))
        for direction in plan[:cut]:
            eng.step(direction)
        state = state_from_engine(eng, game._game, board)
        for i in range(steps):
            if state[0] == "done":
                wins += 1
                break
            seen_phases.add((state[0], state[1]))
            if i % 9 == 8:                              # the dead key
                before = [[set(c) for c in row] for row in eng.grid]
                eng.step("action")
                if eng.grid != before:
                    bad += 1
                    print(f"  trial {t}: ACTION changed the grid")
                    break
                continue
            d = rng.randrange(4)
            before = [[set(c) for c in row] for row in eng.grid]
            eng.step(KEYS[d])
            predicted = fields.step(state, d)
            actual = state_from_engine(eng, game._game, board)
            if predicted != actual:
                bad += 1
                print(f"  trial {t} step {i}: {KEYS[d]} -> model {predicted} "
                      f"engine {actual}")
                break
            if predicted == state and eng.grid != before:
                bad += 1
                print(f"  trial {t} step {i}: model refused {KEYS[d]} but the "
                      f"interpreter changed the grid")
                break
            if eng.check_win() != (actual[0] == "done"):
                bad += 1
                print(f"  trial {t} step {i}: win divergence on {KEYS[d]}")
                break
            if predicted == state:
                refused += 1
            elif predicted[0] != state[0]:
                opened += state[0] == "solve"      # Success: board gone, bridge open
                crossed += state[0] == "walk"      # stepped ashore: bridge burned
            state = actual
    if verbose:
        solve = sorted(k for kind, k in seen_phases if kind == "solve")
        walk = sorted(k for kind, k in seen_phases if kind == "walk")
        print(f"  {'OK' if not bad else 'VIOLATIONS'} ({trials} rollouts, "
              f"{refused} refused presses, {wins} accidental wins)")
        print(f"  islands solved-in: {solve}   crossings walked: {walk}")
        print(f"  {opened} islands finished mid-fuzz, {crossed} bridges crossed")
    return bad


# ---------------------------------------------------------------------------
# Render audit
# ---------------------------------------------------------------------------

#: Every cell composition a level can show, as the object stack that draws it.
#: A burned bridge is deliberately absent: `late [Failure][ActiveRaft] ->
#: [Failure][Wall]` leaves a plain wall behind, which is the truth -- the plank
#: is gone and will never come back.
_AUDIT_CASES = {
    "floor": ["background"],
    "wall": ["background", "wall"],
    "raft (closed bridge)": ["background", "wall", "raft"],
    "plank (open bridge)": ["background", "activeraft"],
    "player on plank": ["background", "activeraft", "player"],
    "target": ["background", "target"],
    "target + crate": ["background", "target", "pullcrate"],
    "target + player": ["background", "target", "player"],
    "crate": ["background", "pullcrate"],
    "player": ["background", "player"],
}


def audit(verbose: bool = True) -> int:
    """Check that every cell composition is pixel-distinct, twice over.

    The flickscreen is 11x10, so cells are always 5px -- the size at which all
    five sprite rows and columns survive the nearest-neighbour downsample. Two
    of these compositions used to collide (target/wall, and target under the
    player), which is what the .txt recolour in the module docstring fixed;
    this is the check that it holds.

    The first pass compares the 5x5 sprite blocks. That is not sufficient on its
    own: a 55x50 render is UPSCALED to 63x58 to fill the frame, so cell
    boundaries stop being multiples of 5 and two blocks that differ could in
    principle sample the same. The second pass therefore fills the whole board
    with one composition and compares the finished 64x64 frames -- the picture
    the agent is actually shown. Returns the number of colliding pairs found by
    either. See [[explod-rendering-fixes]]."""
    game = _load_game_module(GAME_MODULE_ID).make_game(seed=0)
    parsed = game._game
    eng = game._engine
    layers = eng._obj_layers
    game.set_level(0)
    px = max(1, min(64 // parsed.flickscreen[1], 64 // parsed.flickscreen[0]))

    def stack(names):
        objs = [(layers.get(parsed.obj_name_to_idx[n], -1), parsed.objects[n])
                for n in names]
        objs.sort(key=lambda x: x[0])
        return objs

    bad = 0
    blocks = {k: _render_cell_sprite(stack(v), px) for k, v in _AUDIT_CASES.items()}
    for a, b in itertools.combinations(_AUDIT_CASES, 2):
        if (blocks[a] == blocks[b]).all():
            bad += 1
            print(f"  cell_px={px}: {a} and {b} render identical sprite blocks")

    frames = {}
    for name, names in _AUDIT_CASES.items():
        fill = {parsed.obj_name_to_idx[n] for n in names}
        for r in range(eng.height):
            for c in range(eng.width):
                eng.grid[r][c] = set(fill)
        eng._position_index_dirty = True
        frames[name] = _render_frame(eng, parsed).copy()
    for a, b in itertools.combinations(_AUDIT_CASES, 2):
        if (frames[a] == frames[b]).all():
            bad += 1
            print(f"  whole frame: {a} and {b} render identically")
    if verbose:
        print(f"  cell_px={px}: {len(_AUDIT_CASES)} cell types, "
              f"{'all distinct' if not bad else 'COLLISIONS'} "
              f"(sprite blocks and whole 64x64 frames)")
    return bad


def verify_field(verbose: bool = True) -> int:
    """Re-derive the distance-to-win for EVERY reachable state a second way, and
    check it against `Fields`.

    `Fields` builds its value function island by island, backwards, threading a
    terminal value through each crossing -- so a mistake in that bookkeeping
    would produce a self-consistent field that is simply wrong, and the plan
    read off it would be neither shortest nor correctly tie-labelled. This
    instead enumerates the reachable global states forward from the start using
    only `Fields.step`, then runs a plain BFS back from ``("done",)`` over the
    reversed graph -- no islands, no terminals, no phases. Agreement everywhere
    is what licenses calling the plan provably shortest and the optimal sets
    exact. Returns the number of states where the two disagree."""
    fresh = PuzzleScriptAdapter(GAME_NAME, seed=0)
    fresh.set_level(0)
    board = Board(fresh._engine, fresh._game)
    fields = Fields(board)
    start = state_from_engine(fresh._engine, fresh._game, board)

    preds: dict = {}
    seen = {start}
    queue = deque([start])
    while queue:
        st = queue.popleft()
        if st[0] == "done":
            continue
        for d in range(4):
            nxt = fields.step(st, d)
            if nxt == st:                       # a refused press: not an edge
                continue
            preds.setdefault(nxt, []).append(st)
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)

    dist = {("done",): 0}
    queue = deque([("done",)])
    while queue:
        st = queue.popleft()
        for prev in preds.get(st, ()):
            if prev not in dist:
                dist[prev] = dist[st] + 1
                queue.append(prev)

    bad = 0
    for st in seen:
        if dist.get(st, INF) != fields.value(st):
            bad += 1
            if bad <= 5:
                print(f"  {st}: BFS {dist.get(st, INF)} != field "
                      f"{fields.value(st)}")
    if verbose:
        dead = sum(1 for st in seen if st not in dist)
        print(f"  {len(seen)} reachable states, {dead} of them dead ends, "
              f"{'field exact' if not bad else 'FIELD WRONG'}; "
              f"shortest win from the start = {dist[start]} presses")
    return bad


def _plan_report() -> None:
    """Print the plan, how it was found and how many of its steps have more than
    one right answer -- the quick "is this game still fully solved" check. The
    plan is replayed through the real interpreter, so this is also the model's
    end-to-end test, and the press count is checked against the budget the game
    folder's wrapper grants."""
    solver = FivePullobanPuzzlesSolver()
    game, expert, solvable = solver._ensure(0)
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        budget = game._max_steps
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"  L{level}: UNSOLVED")
            continue
        for direction in plan:                          # interpreter-verify it
            eng.step(direction)
            if eng.check_win():
                break
        ties = sum(len(s) - 1 for s in plan.optsets)
        print(f"  L{level}: {len(plan)} presses  win={eng.check_win()}  "
              f"budget={budget}  {expert.describe(level)}  "
              f"{ties} tie-presses on {sum(1 for s in plan.optsets if len(s) > 1)}"
              f"/{len(plan)} steps")
    print(f"  solvable levels: {solvable}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--audit" in sys.argv:
        collisions = audit()
        print(f"audit: {collisions} colliding cell types")
        sys.exit(1 if collisions else 0)
    if "--verify" in sys.argv:
        wrong = verify_field()
        print(f"verify: {wrong} states where the field is wrong")
        sys.exit(1 if wrong else 0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(FivePullobanPuzzlesSolver.main())
