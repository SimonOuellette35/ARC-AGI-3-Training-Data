"""Generate Phase-1 training data for the PuzzleScript game ps:whaleworld
("2D Whale World", increpare -- the whale section of English Country Tune, in
eight lines of rules).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a NATIVE model of the
mechanic (fuzz-verified against the interpreter), the exact distance fields that
model makes affordable, the stratified search the one level too big for a field
needs, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_whaleworld",
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
Every expert step carries an optimal-action SET (see "Optimal-action sets").

THE GAME
--------
You are a blue dot in a walled pond. Brown whales are beached in it, and a whale
is FREED by shoving it off the edge of the map into the orange void. You may not
step into the void yourself. Win when no whale is left.

You never touch a whale. Instead each whale broadcasts two BEAMS -- an `HBeam`
down its whole row and a `VBeam` down its whole column, both straight through
walls and void to the grid edge -- and you move whales by walking into a beam:

    horizontal [ > Player | VBeam ] -> [ > Player | > VBeam ]
    vertical   [ > Player | HBeam ] -> [ > Player | > HBeam ]
    [ Whale | ... | perpendicular Beam ] -> [ perpendicular Whale | ... | perpendicular Beam ]

Everything below was MEASURED against the interpreter, not read off the .txt.

* **A press moves whales that share a LINE with the cell you step INTO, in your
  own direction.** Step left or right into cell X and every whale in X's COLUMN
  slides one cell that way; step up or down into X and every whale in X's ROW
  slides one that way. It is the cell you arrive at that matters, never the one
  you leave, so the same walk taken backwards is not the same move.

* **A blocked press is a total no-op -- the whales do not move either.** The
  prelude declares `require_player_movement`, and the adapter implements it by
  reverting the WHOLE turn (grid included) when the player ends where it began.
  So walking into a wall, into a whale, or into the void (where
  `[ > player | void ] -> [ player | void ]` eats the force outright) leaves the
  board untouched: there is no wait move in this game, and no way to nudge a
  whale by bumping a wall.

* **A whale entering the void is DELETED (`late [ Whale Void ] -> [ Void ]`);
  a whale entering a wall or another whale simply does not move.** Whales are on
  the player's collision layer, so they also block you. Nothing pushes a whale
  chain: two whales in a line both get the force, and the one whose destination
  is occupied stays put while the other leaves.

* **THE TIE-BREAK THAT IS THE WHOLE PUZZLE: a horizontal press does nothing at
  all when the cell you step into is on a whale's ROW as well as a whale's
  column.** `Beam = HBeam or VBeam`, and a rule matching an or-group binds its
  FIRST declared member that is present in the cell. Step sideways into a cell
  carrying both beams and `perpendicular Beam` binds the HBeam -- which the
  `horizontal` rule never gave a force to -- so the whale rule finds no match and
  no whale moves. The vertical press is not symmetric: it forces the HBeam, which
  is the member that gets bound, so it fires whether or not a VBeam is there too.
  ps:whaleworld level 7 is built on this: the player starts on the one cell of
  the board where both beams cross, and half of the level is spent standing
  where a sideways step is dead. Nothing in the .txt says so; it is the adapter's
  or-group binding order, measured (``--model`` counts the cancellations).

* ACTION5 exists but no rule reads it, and with `require_player_movement` it
  cannot even end a turn, so the game is four presses. `directions` says so and
  no search ever branches on it.

* `check_win()` is False on every level's start frame (each ships at least one
  whale), so `vacuous_start_win` is not needed.

Rendering (see ``--audit``)
---------------------------
Three fixes in `data/puzzlescript_games/whaleworld.txt`, all forced:

* **The whale was the void.** Whale shipped `brown` and Void `orange`, which are
  BOTH ARC palette 12, and both are solid 5x5 blocks -- so the piece you have to
  move was pixel-identical to the terrain you have to move it into, i.e. to the
  win condition itself. Repainted `purple` (15).
* **Both beams vanished at cell_px 4.** HBeam painted row 2 and VBeam column 2,
  and `_render_cell_sprite` takes centred samples: at cell_px 4 it reads
  rows/cols {0,1,3,4} and never touches the middle, so on the four boards that
  render at 4px the game's entire signalling layer was blank floor. Redrawn on
  rows {0,4} (HBeam) and columns {0,4} (VBeam), inset to the middle three cells.
  The inset is what keeps `wall` readable: a beam crossing paints a ring, the
  Wall sprite IS a ring, and a full-width beam ring over a wall would have
  rendered exactly like one over floor. Leaving the four corners to the wall
  keeps the pair apart at every size.
* The two beam sprites are a CLOSED ORBIT of the 8-element symmetry group
  (rotate HBeam 90 degrees and you get VBeam exactly), which is what lets the
  game keep the adapter's rotation augmentation without a turned frame showing
  art the game does not own.

A beam UNDER the player is invisible -- `_render_frame` sorts the player last and
the Player sprite is solid -- and that is left alone deliberately: a beam is a
pure function of where the whales are, and the whales are visible, so the frame
loses no state. The cell an agent has to read the tie-break off is the one it is
about to step INTO, which it is not standing on yet.

How the levels are solved
-------------------------
The state is (player cell, set of whale cells) and nothing else -- walls, void
and the beams are all functions of it -- so the model is a dozen lines and runs
~2000x faster than the interpreter (`--model` fuzzes the two against each other
from random play AND from every prefix of every level's own solution, with
per-mechanic coverage counters, because random play alone never exercises a
chain block).

That speed buys the EXACT thing on seven of the eight levels: `_Field` enumerates
the whole reachable component forward and reads a backward BFS from the winning
edges, which gives the provably shortest plan, the exact optimal-action SET at
every step, and a proof of winnability rather than "the search gave up". The
components are 279 to 447k states.

Level 6 (the game's "level 7 of 8", four whales in a 13x16 pond with exactly two
cells you can shove one out of) does not close: its four-whale stratum alone
passes 12M states at depth 36. It is planned by a STRATIFIED search instead --
whales are only ever destroyed, so any path crosses the strata "4 whales, then 3,
then 2, then 1" monotonically, and a bucket-queue BFS per stratum carries the
exact distances of the states that exit it into the next one. Strata 2 and 1
close outright (219k and 6.3k states); strata 4 and 3 are cut off at
``stratum_cap``, so the 80 presses it returns are the shortest solution that
leaves those inside the searched depth rather than a proved global optimum.
Nothing finds shorter: raising ``stratum_cap`` to 5M (69s, 2.3 GB peak) closes
stratum 3 outright at 4.83M states and still returns 80, and so does an
independent width-20000 beam search over the same model. See ``--plans`` and
``--bfs``.

Optimal-action sets
-------------------
``optsets[i]`` is the complete set of equally-shortest presses wherever a field
covers the step -- ``dist(succ) == dist - 1``, read straight off the exhaustive
backward BFS, so it is measured rather than inferred. That is all of levels 0-5
and 7, and the tail of level 6 from the moment two whales are left (the field
closes there), which is 36 of its 80 steps.

The head of level 6 is labelled by the WALK inference instead. Split the plan at
the presses that actually move a whale; between two of them the board is frozen
and the player is just walking, so every shortest walk that is SAFE (moves no
whale) and ends on the same cell reaches the identical state and therefore ties.
That is sound given the plan is shortest -- it can only under-report, never name
a press that is worse -- and it is the same inference `PSPushExpert.
annotate_walks` makes for the sokoban-shaped games here.

``--ties`` checks all of it three ways: every field is BELLMAN-certified end to
end (which proves the sets for the whole space, not only along a plan); on the
levels small enough to afford it each step is re-solved from scratch from every
successor; and each walk-inferred label is checked by CONSTRUCTION -- the
alternative route it claims is built explicitly and replayed through the real
interpreter, which has to win in exactly the same number of presses.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the plan
replays to a guaranteed WIN. The RESET is a NECESSITY here, not the convenience it
is in most of this family. Whaleworld is savagely irreversible -- a whale shoved
into a pocket it cannot be lined up out of is gone for good -- and ``--bfs``
counts exactly how savagely: of the reachable states, 27% (level 1) to 95% (level
5) and 98.4% (level 7) can no longer reach a win. A few random presses really do
kill the pond, which is precisely the arc DESIGN.md wants recorded: flail, get
stuck, reset, then solve.

The expert re-plans from whatever state it is handed rather than replaying a plan
cached at the level start, so it is usable for perturbation data too. On the seven
field levels that re-plan is a dict lookup and free -- the field already holds the
distance of every state, stranded ones included, and returns None for those, which
is what `record_level` needs to know it must reset rather than press on.

Because the enumerations are the whole cost of generation and they are
seed-independent, the START plan of every level is cached to
``data/whaleworld_plans.json`` (`PSExpert.plan_cache_path`): without it every
`parallelize_generator` shard re-derives level 6's stratified search.

Level indices here and in every report are 0-BASED, i.e. one less than the number
in the game's own "level N of 8" messages.

CLI
---
    --plans   per-level size, search kind, plan length and tie coverage
    --model   fuzz the native model against the interpreter, with coverage
    --bfs     the exhaustive reachable-state report (the winnability proof)
    --ties    re-derive every optimal-action label by independent re-solve
    --audit   assert every cell composition renders distinctly, at every size
"""

from __future__ import annotations

import itertools
import random
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                       # noqa: E402
    PSAStarSolver, PSExpert, Plan,
)

GAME_NAME = "whaleworld"

#: The four presses, in the order that breaks every tie -- so a re-derived plan
#: is byte-identical across processes. ACTION5 is read by no rule (see the
#: module docstring) and never enters a search.
_DIRS = ("up", "down", "left", "right")

#: Press index -> (dr, dc). Searches work in indices; `_DIRS` converts at the
#: boundary, because the engine and `Plan` both speak direction names.
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))


# ---------------------------------------------------------------------------
# The native model
# ---------------------------------------------------------------------------

class _Board:
    """The static half of a level -- shape, walls and void -- plus the one-press
    transition over the dynamic half.

    A state is ``(player_cell, whale_cell, ...)``: a flat tuple whose head is the
    player's cell index and whose tail is the SORTED tuple of whale cells, which
    makes it hashable, canonical (whales are interchangeable) and small. Cells are
    ``row * width + col``.

    `step` is the whole game. Read it against the four measured facts in the
    module docstring: the destination cell decides which whales move, a press the
    player cannot complete is a no-op for everything, a whale on the void is
    deleted, and a sideways press into a doubly-beamed cell moves nothing.
    """

    __slots__ = ("h", "w", "walls", "voids", "free")

    def __init__(self, h: int, w: int, walls: frozenset, voids: frozenset):
        self.h = h
        self.w = w
        self.walls = walls
        self.voids = voids
        #: Cells the player may stand on. Whales may stand on these too, and may
        #: additionally enter a void cell -- which deletes them.
        self.free = frozenset(i for i in range(h * w)
                              if i not in walls and i not in voids)

    # -- construction ------------------------------------------------------
    @classmethod
    def read(cls, eng, ids) -> tuple["_Board", tuple]:
        """Build the board and its current state from a live engine grid."""
        h, w = eng.height, eng.width
        walls, voids, whales, player = set(), set(), [], None
        wall_i, void_i = ids["wall"], ids["void"]
        whale_i, player_i = ids["whale"], ids["player"]
        for r in range(h):
            row = eng.grid[r]
            for c in range(w):
                cell = row[c]
                if wall_i in cell:
                    walls.add(r * w + c)
                if void_i in cell:
                    voids.add(r * w + c)
                if whale_i in cell:
                    whales.append(r * w + c)
                if player_i in cell:
                    player = r * w + c
        board = cls(h, w, frozenset(walls), frozenset(voids))
        return board, (player,) + tuple(sorted(whales))

    # -- the mechanic ------------------------------------------------------
    def step(self, state: tuple, d: int) -> tuple:
        """One press. Returns the new state, or ``state`` itself when the press
        is a no-op (which is what a blocked player means here -- see the
        docstring: `require_player_movement` reverts the whole turn)."""
        w = self.w
        player, whales = state[0], state[1:]
        dr, dc = _DELTA[d]
        pr, pc = divmod(player, w)
        nr, nc = pr + dr, pc + dc
        if not (0 <= nr < self.h and 0 <= nc < w):
            return state
        dest = nr * w + nc
        # Void eats the force, wall and whale block the body; either way the
        # player does not move and the turn is reverted whole.
        if dest not in self.free or dest in whales:
            return state
        if dc:
            # Sideways: the destination's VBeam is the one that gets the force,
            # but an HBeam in the same cell wins the or-group binding and kills
            # the rule outright. An HBeam is there exactly when a whale shares
            # the destination's row.
            if any(x // w == nr for x in whales):
                movers = ()
            else:
                movers = tuple(x for x in whales if x % w == nc)
        else:
            # Up/down: the destination's HBeam is forced AND is the bound member,
            # so this fires whenever a whale shares the destination's row.
            movers = tuple(x for x in whales if x // w == nr)
        if not movers:
            return (dest,) + whales
        survivors = set(whales)
        for x in movers:
            xr, xc = divmod(x, w)
            yr, yc = xr + dr, xc + dc
            if not (0 <= yr < self.h and 0 <= yc < w):
                continue
            y = yr * w + yc
            # Walls and whales block; note the test is against the ORIGINAL whale
            # set, which is right: two movers are never in each other's way (they
            # share the line they were forced along, so their tracks are
            # parallel), and a stationary whale blocks whatever it is standing in
            # front of.
            if y in self.walls or y in whales:
                continue
            survivors.discard(x)
            if y not in self.voids:
                survivors.add(y)          # else: freed, into the void
        return (dest,) + tuple(sorted(survivors))

    def safe_moves(self, whales: tuple) -> dict:
        """``cell -> {press: cell}`` over the presses that move the player and
        leave every whale where it is, for a fixed whale set.

        This is the "walking" half of the game, and it is DIRECTED: whether a
        press disturbs a whale depends on the cell it lands on, so A->B being
        safe says nothing about B->A. Used to label the walk stretches of a plan
        (see `_Analysis._walk_sets`)."""
        out: dict = {}
        occupied = set(whales)
        for cell in self.free:
            if cell in occupied:
                continue
            edges = {}
            state = (cell,) + whales
            for d in range(4):
                nxt = self.step(state, d)
                if nxt is not state and nxt[0] != cell and nxt[1:] == whales:
                    edges[d] = nxt[0]
            out[cell] = edges
        return out


# ---------------------------------------------------------------------------
# Exact distance field over a closed reachable component
# ---------------------------------------------------------------------------

class _Field:
    """Every state reachable from one board state, with its exact distance to a
    win.

    Forward BFS to close the component (the model is the authority; the
    interpreter only certifies the finished plan), then a backward BFS from the
    winning edges. Because the component is closed under the transition, the
    backward distances are globally exact, which buys the three things a
    heuristic search cannot have: plans that are provably SHORTEST, optimal-action
    SETS that are measured rather than inferred, and a proof that a state can or
    cannot win at all.

    `build` returns None past ``cap`` states -- for this game that is a real
    outcome, not a runaway guard: level 6's component does not fit and is planned
    by `_Analysis._layered` instead.
    """

    __slots__ = ("board", "dist", "seen")

    def __init__(self, board: _Board, dist: dict, seen: set):
        self.board = board
        self.dist = dist          # state -> presses to win (winnable states only)
        self.seen = seen          # every reachable state, winnable or not

    @classmethod
    def build(cls, board: _Board, start: tuple, cap: int) -> "_Field | None":
        step = board.step
        seen = {start}
        queue = deque([start])
        # rev[s] lists the states that reach s in one press; wins are the states
        # with a press that empties the pond.
        rev: dict = defaultdict(list)
        wins: list = []
        while queue:
            s = queue.popleft()
            for d in range(4):
                nxt = step(s, d)
                if nxt is s:
                    continue                      # the press did nothing at all
                if len(nxt) == 1:
                    wins.append(s)
                    continue                      # the win is terminal, not a state
                rev[nxt].append(s)
                if nxt not in seen:
                    if len(seen) >= cap:
                        return None
                    seen.add(nxt)
                    queue.append(nxt)
        dist: dict = {}
        frontier = deque()
        for s in wins:
            if s not in dist:
                dist[s] = 1
                frontier.append(s)
        while frontier:
            s = frontier.popleft()
            nd = dist[s] + 1
            for p in rev.get(s, ()):
                if p not in dist:
                    dist[p] = nd
                    frontier.append(p)
        return cls(board, dist, seen)

    # -- reading the field --------------------------------------------------
    def cost(self, state: tuple, d: int) -> "int | None":
        """Presses to win if ``d`` is pressed at ``state``, or None if that press
        is unavailable or leads somewhere that cannot win."""
        nxt = self.board.step(state, d)
        if nxt is state:
            return None
        if len(nxt) == 1:
            return 1
        rest = self.dist.get(nxt)
        return None if rest is None else rest + 1

    def optimal(self, state: tuple) -> list:
        """Every press index on a shortest path from ``state``."""
        best = self.dist.get(state)
        if best is None:
            return []
        return [d for d in range(4) if self.cost(state, d) == best]

    def presses(self, state: tuple) -> "list | None":
        """The shortest press sequence from ``state``, ties broken by `_DIRS`
        order, with the exact optimal SET at every step."""
        if state not in self.dist:
            return None
        out, sets = [], []
        s = state
        while len(s) > 1:
            best = self.optimal(s)
            out.append(best[0])
            sets.append(best)
            s = self.board.step(s, best[0])
        return out, sets


# ---------------------------------------------------------------------------
# One level's planner: a field where the component fits, strata where it does not
# ---------------------------------------------------------------------------

class _Analysis:
    """Everything one static board knows, built lazily and reused across every
    state the recorder asks about.

    `solve` answers from an exact `_Field` whenever the component containing the
    state fits in ``state_cap`` -- which is seven of the eight levels, and every
    state of them, so a re-plan after a perturbation is a dict lookup. The one
    level that does not fit falls through to `_layered`.
    """

    def __init__(self, board: _Board, state_cap: int, stratum_cap: int):
        self.board = board
        self.state_cap = state_cap
        self.stratum_cap = stratum_cap
        self.fields: list = []
        self.notes: dict = {}          # per start state: how it was solved

    # -- entry point --------------------------------------------------------
    def solve(self, state: tuple) -> "Plan | None":
        field = self._field_for(state)
        if field is not None:
            found = field.presses(state)
            if found is None:
                return None
            presses, sets = found
            self.notes.setdefault(state, "field")
            return Plan([_DIRS[d] for d in presses],
                        [[_DIRS[d] for d in s] for s in sets])
        presses = self._layered(state)
        if presses is None:
            return None
        self.notes.setdefault(state, "strata")
        sets = self._label(state, presses)
        return Plan([_DIRS[d] for d in presses],
                    [[_DIRS[d] for d in s] for s in sets])

    def _field_for(self, state: tuple) -> "_Field | None":
        """The exact field covering ``state``, building one if the component
        fits. A failed build is remembered by whale count, so the second call at
        a four-whale state does not pay for the same doomed enumeration twice."""
        for f in self.fields:
            if state in f.seen:
                return f
        n = len(state) - 1
        if n in getattr(self, "_too_big", ()):
            return None
        field = _Field.build(self.board, state, self.state_cap)
        if field is None:
            self._too_big = getattr(self, "_too_big", set()) | {n}
            return None
        self.fields.append(field)
        return field

    # -- the stratified search ---------------------------------------------
    def _layered(self, start: tuple) -> "list | None":
        """Shortest presses to a win, searched one WHALE-COUNT stratum at a time.

        A whale is only ever destroyed, never created, so every path from a
        k-whale state crosses the strata k, k-1, ... 0 in order and never comes
        back. That makes the whole search decomposable: BFS inside stratum k from
        its sources (each carrying the exact distance at which it was entered --
        a bucket queue, i.e. Dijkstra with non-zero initial distances), collect
        the states it EXITS into with their distances, and hand those to stratum
        k-1 as its sources.

        The decomposition is exact when no stratum is capped; when one is, the
        answer is the shortest solution that leaves that stratum inside the
        searched depth, which is the honest claim `--plans` reports. Level 6 caps
        only its first stratum and its three later ones close, so a shorter
        answer would have to keep all four whales for longer than the search
        looked -- and neither a wider cap nor an independent beam search finds
        one.
        """
        k = len(start) - 1
        step = self.board.step
        dist = {start: 0}
        parent: dict = {}
        sources = {start: 0}
        while k > 0:
            if not sources:
                return None
            buckets: dict = defaultdict(list)
            for s, d0 in sources.items():
                buckets[d0].append(s)
            seen = set(sources)
            exits: dict = {}
            depth = min(buckets)
            deepest = max(buckets)
            count = 0
            capped = False
            while depth <= deepest and not capped:
                for s in buckets.get(depth, ()):
                    if dist[s] != depth:
                        continue
                    for d in range(4):
                        nxt = step(s, d)
                        if nxt is s:
                            continue
                        nd = depth + 1
                        if len(nxt) - 1 < k:
                            if nxt not in exits or nd < exits[nxt]:
                                exits[nxt] = nd
                                dist[nxt] = nd
                                parent[nxt] = (s, d)
                            continue
                        if nxt in seen:
                            continue
                        seen.add(nxt)
                        dist[nxt] = nd
                        parent[nxt] = (s, d)
                        buckets[nd].append(nxt)
                        if nd > deepest:
                            deepest = nd
                        count += 1
                        if count >= self.stratum_cap:
                            capped = True
                            break
                    if capped:
                        break
                depth += 1
            self.notes.setdefault("strata_report", []).append(
                (k, len(seen), len(exits), depth - 1, capped))
            sources = exits
            k -= 1
        if not sources:
            return None
        goal = min(sources, key=lambda s: sources[s])
        presses = []
        cur = goal
        while cur in parent:
            prev, d = parent[cur]
            presses.append(d)
            cur = prev
        presses.reverse()
        return presses

    # -- optimal-action sets for a stratified plan --------------------------
    def _label(self, start: tuple, presses: list) -> list:
        """Per-step optimal SETS for a plan the strata produced.

        Exact from the first state a `_Field` closes at (which for level 6 is the
        moment two whales are left), inferred from the walk structure before it.
        Attempts are made only where the whale count first changes -- trying every
        step would pay for the same doomed enumeration dozens of times."""
        states = [start]
        for d in presses:
            states.append(self.board.step(states[-1], d))
        # Candidate hand-over points: the plan's start, and each state right
        # after a whale is freed.
        candidates = [0]
        for i in range(1, len(states)):
            if len(states[i]) < len(states[i - 1]):
                candidates.append(i)
        field = None
        handover = len(presses)
        for i in candidates:
            if i >= len(presses):
                break
            f = self._field_for(states[i])
            if f is not None:
                field, handover = f, i
                break
        sets: list = [None] * len(presses)
        for i in range(handover, len(presses)):
            best = field.optimal(states[i])
            sets[i] = best if presses[i] in best else [presses[i]]
        for i, s in enumerate(self._walk_sets(states, presses, handover)):
            sets[i] = s
        return sets

    def _walk_sets(self, states: list, presses: list, upto: int) -> list:
        """Label the first ``upto`` steps by the WALK inference.

        Between two presses that move a whale the board is frozen and the player
        is only repositioning, so every SAFE (whale-preserving) shortest walk that
        ends on the same cell reaches the identical state and ties exactly. Sound
        whenever the plan is shortest; it can only under-report."""
        sets: list = []
        i = 0
        while i < upto:
            j = i
            while j < upto and states[j + 1][1:] == states[j][1:]:
                j += 1
            # [i, j) is a run of pure walking; the player must arrive at
            # states[j][0] -- either to make the whale press at j, or to hand
            # over to the exact field there.
            whales = states[i][1:]
            back = self._safe_dist(whales, states[j][0])
            for t in range(i, j):
                here = back.get(states[t][0])
                best = []
                if here is not None:
                    for d in range(4):
                        nxt = self.board.step(states[t], d)
                        if nxt is states[t] or nxt[1:] != whales:
                            continue
                        if back.get(nxt[0], -1) == here - 1:
                            best.append(d)
                sets.append(best if presses[t] in best else [presses[t]])
            if j < upto:
                sets.append([presses[j]])       # the whale press itself
            i = j + 1
        return sets[:upto]

    def _safe_dist(self, whales: tuple, target: int) -> dict:
        """Safe-walk distance from every cell TO ``target``, for a fixed whale
        set. A backward BFS, because `safe_moves` is directed."""
        edges = self.board.safe_moves(whales)
        rev: dict = defaultdict(list)
        for cell, outs in edges.items():
            for nxt in outs.values():
                rev[nxt].append(cell)
        out = {target: 0}
        queue = deque([target])
        while queue:
            cell = queue.popleft()
            nd = out[cell] + 1
            for p in rev.get(cell, ()):
                if p not in out:
                    out[p] = nd
                    queue.append(p)
        return out


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class WhaleWorldExpert(PSExpert):
    """`PSExpert`'s plan memo, disk cache and snapshot discipline around the
    native model.

    `_search` is replaced the way ps:dotsnake and ps:escaping_limbo replace it:
    the base class keeps the memo, the restore discipline and the level scoping,
    and only the strategy underneath changes. Here the strategy is "read the board
    off the engine, hand it to `_Analysis`, return its `Plan`" -- the interpreter
    is never stepped by the planner at all, only by the recorder that certifies
    the plan. `heuristic` is therefore never called and asserts rather than
    returning a number nothing would use.

    `_key` is inherited (every non-background cell), which is exact and canonical
    ACROSS levels -- the walls and the void are in the key -- so no
    ``scope_by_level`` is needed. It includes the beams, which are a pure function
    of the whale positions and so alias nothing.
    """

    directions = list(_DIRS)

    #: The searches are the whole cost of generation and are seed-independent, so
    #: each level's START plan is kept between processes. Without it every
    #: `parallelize_generator` shard re-derives level 6's stratified search.
    plan_cache_path = (Path(__file__).resolve().parent.parent
                       / "data" / "whaleworld_plans.json")

    #: Ceiling on one exhaustive component. Sized from the boards: the largest
    #: that closes is level 7's at 447k states, and level 6's four-whale stratum
    #: alone is past 12M, so anything in between separates them cleanly. It is a
    #: MEMORY budget -- every state is a 5-tuple plus its reverse-edge list.
    state_cap: int = 1_200_000

    #: Ceiling on ONE stratum of the stratified search. Level 6's stratum 4 is
    #: the only one that ever reaches it.
    stratum_cap: int = 1_500_000

    def setup(self) -> None:
        self.ids = self.g.obj_name_to_idx
        self._analyses: dict = {}

    def heuristic(self, eng) -> int:
        raise AssertionError(
            "WhaleWorldExpert plans on the native model; heuristic is unused")

    def analysis(self, board: _Board) -> _Analysis:
        key = (board.h, board.w, board.walls, board.voids)
        got = self._analyses.get(key)
        if got is None:
            got = _Analysis(board, self.state_cap, self.stratum_cap)
            self._analyses[key] = got
        return got

    def _search(self, eng) -> "Plan | None":
        board, state = _Board.read(eng, self.ids)
        return self.analysis(board).solve(state)


class WhaleWorldSolver(PSAStarSolver):
    game_id = "puzzlescript_whaleworld"
    game_name = GAME_NAME
    expert_cls = WhaleWorldExpert

    #: Record against the GAME FOLDER's adapter. ``games/ps:whaleworld`` is a
    #: plain passthrough today; it is named anyway so a sprite patch or a step cap
    #: added there later cannot silently make this generator tape a game nobody
    #: plays (the ps:count_mover trap).
    game_module_id = "ps:whaleworld"

    #: Room for the longest plan (level 6's 80 presses) with margin. The
    #: exploration prefix is not counted against it -- the ``set_level`` that ends
    #: the prefix resets the adapter's own 200-press per-level budget.
    max_steps = 140


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = WhaleWorldSolver()
    game = solver.make_game(seed)
    return solver, game, WhaleWorldExpert(game)


def _state_of(game):
    return _Board.read(game._engine, game._game.obj_name_to_idx)


def _replay(game, presses) -> bool:
    """Certify a plan by stepping the real interpreter through it."""
    eng = game._engine
    for d in presses:
        eng.step(d)
        if eng.check_win():
            return True
    return eng.check_win()


def _report() -> int:
    """Per-level size, inventory, how the plan was found, its length (certified
    against the interpreter) and its tie coverage."""
    _solver, game, expert = _new()
    eng = game._engine
    total = tied = 0
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = _state_of(game)
        t0 = time.time()
        plan = expert.plan(eng, level)
        dt = time.time() - t0
        head = (f"level {level}: {eng.height:2d}x{eng.width:2d}, "
                f"{len(board.free):3d} open cells, {len(state) - 1} whale(s)")
        if plan is None:
            print(f"{head} -- UNWINNABLE")
            bad += 1
            continue
        game.set_level(level)
        won = _replay(game, plan)
        analysis = expert.analysis(board)
        field = analysis._field_for(state)
        if field is not None:
            how = f"exact field ({len(field.seen)} states, SHORTEST)"
        else:
            rep = analysis.notes.get("strata_report")
            how = ("strata " + " ".join(f"{k}:{n}{'*' if cap else ''}"
                                        for k, n, _e, _d, cap in rep)
                   if rep else
                   "strata (plan served from the disk cache; --bfs re-searches)")
        sets = plan.optsets
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(plan)
        tied += ties
        room = "ok" if len(plan) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(plan):3d} presses "
              f"({'WIN certified' if won else 'REPLAY FAILED'}, budget "
              f"{game._max_steps} {room}), {ties:3d} tie sets "
              f"({ties / max(1, len(plan)):3.0%}), {dt:6.2f}s -- {how}")
        if not won:
            bad += 1
    print(f"total {total} presses, {tied} steps with a tie "
          f"({tied / max(1, total):.0%})   (* = stratum capped)")
    return 0 if not bad else 1


def _model() -> int:
    """Fuzz the native model against the interpreter, with per-mechanic coverage.

    Two sources of trajectories, because neither is enough on its own: seeded
    RANDOM play from the level start (broad, but it almost never gets a whale
    near a wall it can be blocked against) and random play from every PREFIX of
    each level's own solution (which is where the chain blocks, the tie-break
    cancellations and the frees actually live). Every press compares the full
    grid -- player, whales, and that the walls and void never moved -- plus the
    win predicate.

    The coverage counters are the point of the report: a fuzz that reports "model
    matches" having exercised no whale-blocked-by-whale is not evidence about the
    rule that decides levels 6 and 7."""
    _solver, game, expert = _new()
    eng, ids = game._engine, game._game.obj_name_to_idx
    cover = defaultdict(int)
    checked = mism = 0

    def observe(board, state, d, nxt):
        w = board.w
        player, whales = state[0], state[1:]
        pr, pc = divmod(player, w)
        dr, dc = _DELTA[d]
        nr, nc = pr + dr, pc + dc
        if nxt is state:
            cover["press refused (whole turn reverted)"] += 1
            return
        moved = len(whales) - len(set(whales) & set(nxt[1:]))
        if len(nxt) < len(state):
            cover["whale freed into the void"] += len(state) - len(nxt)
        if dc and any(x // w == nr for x in whales) \
                and any(x % w == nc for x in whales):
            cover["sideways press CANCELLED by the HBeam tie-break"] += 1
        if moved:
            cover["whale moved by a %s press"
                  % ("sideways" if dc else "vertical")] += 1
            if moved > 1:
                cover["one press moved 2+ whales"] += 1
        for x in whales:
            xr, xc = divmod(x, w)
            forced = ((dc and not any(y // w == nr for y in whales)
                       and xc == nc) or (not dc and xr == nr))
            if not forced:
                continue
            y = (xr + dr) * w + (xc + dc)
            if y in board.walls:
                cover["whale blocked by a wall"] += 1
            elif y in whales:
                cover["whale blocked by another whale"] += 1

    def grid_state():
        _b, st = _state_of(game)
        return st

    def sweep(level, prefix, trials, length, rng, forced=None):
        """``trials`` random walks of ``length`` presses from ``prefix``, or one
        walk down ``forced`` when a fixed press list is supplied."""
        nonlocal checked, mism
        game.set_level(level)
        board, _st = _state_of(game)
        walls0, voids0 = board.walls, board.voids
        for _ in range(trials):
            game.set_level(level)
            for d in prefix:
                eng.step(d)
            state = grid_state()
            for _k in range(length):
                d = _DIRS.index(forced[_k]) if forced else rng.randrange(4)
                nxt = board.step(state, d)
                observe(board, state, d, nxt)
                eng.step(_DIRS[d])
                b2, live = _state_of(game)
                checked += 1
                if b2.walls != walls0 or b2.voids != voids0:
                    print(f"  L{level}: the static board MOVED")
                    mism += 1
                    return
                if live != nxt:
                    print(f"  L{level} prefix{len(prefix)} press {_DIRS[d]}: "
                          f"engine {live} != model {nxt}")
                    mism += 1
                    return
                if eng.check_win() != (len(nxt) == 1):
                    print(f"  L{level}: win predicate disagrees")
                    mism += 1
                    return
                state = nxt
                if len(state) == 1:
                    break

    for level in range(game.n_levels):
        game.set_level(level)
        rng = random.Random(1000 + level)
        sweep(level, (), 250, 40, rng)
        game.set_level(level)          # the sweeps left the engine mid-flail
        plan = expert.plan(eng, level)
        if plan is not None:
            # The plan itself, press for press -- the only trajectory that is
            # guaranteed to exercise every whale FREE, which random play from a
            # prefix almost never completes.
            sweep(level, (), 1, len(plan), rng, forced=list(plan))
            for cut in range(0, len(plan), 2):
                sweep(level, plan[:cut], 4, 12, rng)
        print(f"level {level}: fuzzed")
    print(f"\n{checked} presses compared, {mism} mismatches")
    print("coverage:")
    for name in sorted(cover):
        print(f"  {cover[name]:7d}  {name}")
    missing = [n for n in ("whale freed into the void",
                           "whale blocked by a wall",
                           "whale blocked by another whale",
                           "sideways press CANCELLED by the HBeam tie-break",
                           "whale moved by a sideways press",
                           "whale moved by a vertical press",
                           "press refused (whole turn reverted)",
                           "one press moved 2+ whales")
               if not cover.get(n)]
    if missing:
        print(f"UNEXERCISED (the fuzz proves nothing about these): {missing}")
    print("model clean" if not mism and not missing
          else "MODEL FUZZ FAILED" if mism else "model matched but incomplete")
    return 0 if not mism else 1


def _bfs() -> int:
    """The exhaustive reachable-state report: for every level whose component
    closes, how big it is, how much of it can still win, and the proof that the
    plan is shortest. The level whose component does not close reports what its
    strata did instead."""
    _solver, game, expert = _new()
    eng = game._engine
    for level in range(game.n_levels):
        game.set_level(level)
        board, state = _state_of(game)
        analysis = expert.analysis(board)
        t0 = time.time()
        # Go through `_Analysis`, not through `expert.plan`: the disk plan cache
        # would short-circuit the search and this report would have nothing to
        # say about how the answer was reached.
        field = analysis._field_for(state)
        dt = time.time() - t0
        if field is None:
            print(f"level {level}: component does not close under "
                  f"{expert.state_cap} states ({dt:.1f}s)")
            plan = analysis.solve(state)
            for k, n, e, d, cap in analysis.notes.get("strata_report", []):
                print(f"    stratum {k}: {n:8d} states, {e:5d} exits, "
                      f"depth {d}{' CAPPED' if cap else ' (closed)'}")
            print(f"    -> {len(plan)} presses; shortest solution that leaves "
                  f"the capped strata inside the searched depth")
            continue
        live = len(field.dist)
        stranded = len(field.seen) - live
        found = field.presses(state)
        verdict = (f"shortest {len(found[0])} presses" if found
                   else "UNWINNABLE (no winning edge is reachable)")
        print(f"level {level}: {len(field.seen):7d} reachable states "
              f"({live} can still win, {stranded} stranded = "
              f"{stranded / max(1, len(field.seen)):.1%}), {dt:5.2f}s "
              f"-- {verdict}")
    return 0


def _certify(field: _Field) -> int:
    """Bellman-certify a whole distance field, and return the error count.

    For every reachable state the field must satisfy ``dist(s) == 1 + min over
    presses of dist(succ)``, with a winning press counting 1 and a press into a
    state that cannot win counting nothing -- and a state absent from ``dist``
    must have no such press at all. That is a complete proof of the field (and
    therefore of every plan and every optimal SET read off it) in one sweep over
    the space, which is strictly cheaper and strictly stronger than re-solving
    from each state: it checks EVERY state, not only the ones a plan visits.
    """
    board = field.board
    errs = 0
    for s in field.seen:
        best = None
        for d in range(4):
            nxt = board.step(s, d)
            if nxt is s:
                continue
            if len(nxt) == 1:
                cost = 1
            elif nxt in field.dist:
                cost = field.dist[nxt] + 1
            else:
                continue
            if best is None or cost < best:
                best = cost
        if field.dist.get(s) != best:
            errs += 1
    return errs


def _segment_end(analysis, states: list, plan: list, pi: int) -> int:
    """Index the walk stretch starting at ``pi`` has to arrive at: the step whose
    press moves a whale, or the step an exact field takes the plan over."""
    j = pi
    while j < len(plan):
        if states[j + 1][1:] != states[j][1:]:
            break                                   # plan[j] is the whale press
        if analysis._field_for(states[j]) is not None:
            break                                   # the field takes over here
        j += 1
    return j


def _walk_route(analysis, whales: tuple, src: int, target: int) -> "list | None":
    """A shortest SAFE (whale-preserving) press sequence from ``src`` to
    ``target``, as press indices."""
    back = analysis._safe_dist(whales, target)
    if src not in back:
        return None
    edges = analysis.board.safe_moves(whales)
    out = []
    cell = src
    while cell != target:
        here = back[cell]
        for d in range(4):
            nxt = edges[cell].get(d)
            if nxt is not None and back.get(nxt, -1) == here - 1:
                out.append(d)
                cell = nxt
                break
        else:
            return None
    return out


def _ties() -> int:
    """Re-derive every optimal-action label independently of the labels.

    Three checks, because the two kinds of label are established two different
    ways:

    * every exact field the plans lean on is BELLMAN-CERTIFIED end to end
      (`_certify`), which proves the distances -- and so every set read off them
      -- for the whole space rather than for the plan's own path;
    * on the levels whose component is small enough to afford it, each step is
      additionally re-solved the fully independent way: take each candidate
      press and build a FRESH field from the state it lands in, keeping the press
      iff ``1 + len(replan)`` is exactly the plan length remaining;
    * a walk-INFERRED label (level 6's head, where no field exists) is checked by
      CONSTRUCTION: for every press it names, the alternative route is built
      explicitly -- that press, then a shortest safe walk back to the cell the
      plan's next whale press is made from -- and the resulting sequence is
      replayed through the real interpreter and required to win in exactly the
      same number of presses as the plan. That is the strongest statement
      available where "is anything shorter" is the question the search cannot
      close.
    """
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board, start = _state_of(game)
        analysis = expert.analysis(board)
        plan = expert.plan(eng, level)
        if plan is None:
            print(f"level {level}: no plan (skipped)")
            continue
        states = [start]
        for name in plan:
            states.append(board.step(states[-1], _DIRS.index(name)))

        for field in analysis.fields:
            errs = _certify(field)
            bad += errs
            print(f"level {level}: field of {len(field.seen)} states -- "
                  f"{'Bellman-certified' if not errs else f'{errs} BROKEN'}")

        exact = inferred = resolved = alts = 0
        for pi, (taken, claimed) in enumerate(zip(plan, plan.optsets)):
            s = states[pi]
            remaining = len(plan) - pi
            if taken not in claimed:
                print(f"  level {level} step {pi}: took {taken}, not in its "
                      f"own optimal set {list(claimed)}")
                bad += 1
            field = analysis._field_for(s)
            if field is not None:
                measured = [_DIRS[d] for d in field.optimal(s)]
                if measured != list(claimed):
                    print(f"  level {level} step {pi}: labelled "
                          f"{list(claimed)} but the field measures {measured}")
                    bad += 1
                exact += 1
                if len(field.seen) <= 20_000:
                    indep = []
                    for d in range(4):
                        nxt = board.step(s, d)
                        if nxt is s:
                            continue
                        if len(nxt) == 1:
                            cost = 1
                        else:
                            fresh = _Field.build(board, nxt, expert.state_cap)
                            got = None if fresh is None else fresh.presses(nxt)
                            cost = None if got is None else 1 + len(got[0])
                        if cost == remaining:
                            indep.append(_DIRS[d])
                    if indep != list(claimed):
                        print(f"  level {level} step {pi}: labelled "
                              f"{list(claimed)} but a fresh re-solve says "
                              f"{indep}")
                        bad += 1
                    resolved += 1
                continue

            inferred += 1
            j = _segment_end(analysis, states, plan, pi)
            if j == pi:                       # the whale press itself
                if list(claimed) != [taken]:
                    print(f"  level {level} step {pi}: a whale press labelled "
                          f"{list(claimed)}")
                    bad += 1
                continue
            target = states[j][0]
            for name in claimed:
                d = _DIRS.index(name)
                nxt = board.step(s, d)
                if nxt is s or nxt[1:] != s[1:]:
                    print(f"  level {level} step {pi}: {name} is not a safe "
                          f"walk move")
                    bad += 1
                    continue
                route = _walk_route(analysis, s[1:], nxt[0], target)
                if route is None or len(route) != j - pi - 1:
                    print(f"  level {level} step {pi}: {name} does not rejoin "
                          f"the plan in the same number of presses")
                    bad += 1
                    continue
                if name == taken:
                    continue
                seq = ([_DIRS.index(x) for x in plan[:pi]] + [d] + route
                       + [_DIRS.index(x) for x in plan[j:]])
                game.set_level(level)
                if len(seq) != len(plan) or not _replay(
                        game, [_DIRS[x] for x in seq]):
                    print(f"  level {level} step {pi}: the {name} alternative "
                          f"does not win in {len(plan)} presses")
                    bad += 1
                else:
                    alts += 1
        print(f"level {level}: {exact} steps measured off a certified field "
              f"({resolved} of them re-solved from scratch), {inferred} "
              f"walk-inferred ({alts} alternatives replayed to a WIN)")
    print("ties clean" if not bad else f"TIES FAILED: {bad} mismatches")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every cell size the boards use.

    Whole 64x64 frames of uniform boards are compared rather than one cell out of
    a mixed board: `_render_frame` centre-pads a non-square board, so indexing a
    cell by ``cell_px * r`` silently reads the wrong pixels on the wide levels
    (the ps:explod lesson). Two uniform boards render identically iff their cells
    do.

    Every terrain can carry either beam or both -- the beams sweep the whole grid
    line, straight through walls and void -- so the matrix is the four opaque
    things crossed with the four beam states, which is what caught the ring-over-a-
    ring clash the redraw fixes. The one deliberate exception is the player, who
    is drawn last and solid and therefore hides the beams under him (see the
    module docstring for why that costs the frame no state). Each composition is
    also checked not to render as the uniform letterbox colour, which is the one
    clash a pairwise matrix structurally cannot see (the ps:stand_iii lesson --
    and Wall here is `black`, ARC 5, which IS the pad colour)."""
    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, _expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx

    comps: dict = {}
    for base in ("floor", "void", "wall", "whale"):
        objs = () if base == "floor" else (base,)
        for tag, beams in (("", ()), (" +h", ("hbeam",)), (" +v", ("vbeam",)),
                           (" +hv", ("hbeam", "vbeam"))):
            comps[base + tag] = objs + beams
    comps["player"] = ("player",)
    comps["player +hv"] = ("player", "hbeam", "vbeam")

    #: The player is rendered last and is a solid 5x5, so a beam beneath him
    #: cannot show. Deliberate: a beam is a function of the whale positions and
    #: the whales are visible, so no state is lost.
    expected = {("player", "player +hv")}

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
        unexpected = [c for c in clashes
                      if tuple(sorted(c)) not in {tuple(sorted(e))
                                                  for e in expected}]
        pad = [n for n, f in shots.items()
               if len(np.unique(f)) == 1 and int(f.flat[0]) == 5]
        bad += len(unexpected) + len(pad)
        note = "OK" if not unexpected else f"IDENTICAL {unexpected}"
        if pad:
            note += f"  PAD-COLOURED {pad}"
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): {len(comps)} "
              f"compositions -- {note}")
    print("audit clean" if not bad else f"AUDIT FAILED: {bad} problems")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--model" in sys.argv:
        sys.exit(_model())
    if "--bfs" in sys.argv:
        sys.exit(_bfs())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(WhaleWorldSolver.main())
