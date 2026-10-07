"""Generate Phase-1 training data for the PuzzleScript game ps:brendan_loves_mondays
("Brendan loves mondays", Benjy Bates).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: a
native model of the mechanic, a macro A* over it, and the interpreter
certification that makes a modelling slip cost a level rather than emit a
trajectory that does not win.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_brendan_loves_mondays",
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
exactly.

The game
--------
Sokoban with a MOOD. Brendan is drawn in one of four colours and can only push
the balls of the colour he currently is; ACTION5 advances him one step around his
level's mood cycle, and the cycle differs per level:

    level 0, 1     love <-> hate                                   (2 moods)
    levels 2, 3    orange -> hate -> love -> orange                (3 moods)
    levels 4, 5    green -> love -> orange -> hate -> green        (4 moods)
    levels 6-9     "Me", which has no ACTION rule at all           (1 mood)

Balls of one colour chain-push each other (`[> LoveBall|LoveBall]`), so a run of
them shifts as a unit and cancels the turn if the far end is blocked. Everything
else in the crate layer -- a ball of the wrong colour, a ball against a wall --
simply blocks, and the turn is a no-op.

The goal object is never a ball. Levels 0-5 have a Key, which EVERY mood can push
(`[> Player|Key]`, and Player is the or-group of all ten Brendans), and the win is
``all key on lock``: the balls are purely the obstacle course between the key and
the lock in the wall. Levels 6-8 are the same puzzle stripped bare -- Me pushes a
bottle of Lucozade four cells up into the lock (`[> lucozade|lock] -> win`) -- and
level 9 is Me walking two cells onto the bed (`[> me|bed] -> win`). Those last
four are the game's ending cutscene and are genuinely two to four presses long.

Note both endings win by RULE, not by win condition: levels 6-8 leave ``all me on
bed`` false (there is a Me and no bed), so the conjunction of the three win
conditions is never satisfied there and only ``_rule_win`` ends the level. The
model reproduces both paths.

Expert solver
-------------
The interpreter is too slow to search: this game has ~26 rules and cancel-checks
dominate, costing **14 ms per `step()`**, which is ~145 macro expansions per
second -- level 1 does not fall out of that in half an hour. So `_Board` is a
native re-implementation of the mechanic above (~40 lines of dynamics) and the
search runs on that at ~25k expansions per second, a 170x speedup. It is
FUZZ-VERIFIED, not trusted: `_fuzz_check` replays random action sequences from
every level start through both the interpreter and the model and compares the
full crate map, the player cell, the mood AND the win flag after every single
press. Any plan the search returns is then replayed press-by-press through the
real interpreter and only accepted if `check_win` fires (`_certify`), so the
worst a modelling slip can do is lose a level.

`_macro_astar` branches on ``walk to the push cell, then push`` MACROS rather
than key presses -- a push is the only move with an effect, so a primitive search
burns its whole budget re-deriving walks -- plus one ``press ACTION`` macro per
node for the mood cycle, plus (level 9 only) ``walk onto the bed``. Costs are
counted in primitive MOVES, the unit the agent pays, and states are deduped by
(crates, the player's reachable REGION, mood): two states with the same crates,
the same region and the same mood admit exactly the same futures. The mood MUST
be in that key -- it is the one thing about this game that a stock sokoban
canonicalisation gets wrong, and dropping it silently merges "can push blue" with
"can push purple".

The heuristic is a backwards Dijkstra from the lock over the goal piece's PUSH
edges (a piece reaches ``X`` from ``X-d`` only when ``X-d`` and the player's stand
cell ``X-2d`` are on the board and not walls), charging `_CLUTTER` extra for every
other crate the push would have to displace first. Two things that buys:

  * **Deadlock detection.** A cell with no outgoing push at all is in no table, so
    a key shoved into a corner scores `_DEAD` and the subtree is dropped. On these
    boards that is most of the pruning: row 1 of level 1, for instance, is a
    complete trap -- row 0 is solid wall, so a key anywhere in row 1 can only ever
    be pushed sideways and can never come back down.
  * **Direction.** Without the clutter term the estimate is flat across the dozen
    moves that clear balls out of the key's corridor without the key itself
    moving, and A* degenerates to uniform-cost search at a depth where that is
    hopeless.

The clutter term is not a lower bound, so no plan here is certified shortest at
any weight. That is why `_WEIGHTS` runs the search at more than one weight and
keeps the SHORTEST plan rather than the first: weighted A* is not monotone in the
weight on these boards (level 1 comes out at 54 moves under weight 2, 68 under
weight 3 and 52 under weight 4; level 4 at 61, 46 and 54 respectively), so no
weight dominates and each of the three is cheap. Contiguous-block deletion, which
shortens the wandering plans of a beam search, buys exactly nothing here --
deleting any block of a sokoban plan invalidates every push after it, and it was
measured returning all five hard plans unchanged -- so it is not attempted.

All 10 levels solve, in ~75s of one-time search at seed 0:

    level  0    1    2    3    4    5    6    7    8    9
    moves 10   52   29   86   46   78    4    4    4    2

Augmentation
------------
The engine state after reset is identical for every seed (the levels are fixed
ASCII maps), so the only per-(seed, level) variable is the presentation: the frame
rotation (``rotation_k in {0,1,2,3}``) plus an independent horizontal and vertical
flip, each with the matching directional action remap, all owned by
`PuzzleScriptAdapter`. Together they re-sample the board's 8-element symmetry
group. The flips are sound here for the same reason they are in Bad Example --
gravity-free, screen-relative moves, every rule written with the relative ``>``
force, and no sprite that encodes a direction -- and this game was added to
`PuzzleScriptAdapter._FLIP_GAMES` for them; see the comment there. There is no
colour augmentation, so the shipped hexes are what an agent sees; they were
checked against the ARC palette collapse before any of this was trusted (see
`_PALETTE_NOTES`). The expert plan is seed-independent: solved once per level,
cached, and replayed per seed with that seed's remapped screen actions. Seed 0
pays the ~75s; every later seed replays from cache.

Usage (run from the repo root):
    python solvers/generate_brendan_loves_mondays_training.py --episodes 200 \
        --out data/training_multi_level/brendan_loves_mondays
"""

from __future__ import annotations

import heapq
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                             # noqa: E402
    PSAStarSolver, PSExpert, restore, snapshot,
)

GAME_NAME = "Brendan_loves_mondays"

#: (name, dr, dc) for the four moves, in the engine's direction vocabulary.
_D4 = (("up", -1, 0), ("down", 1, 0), ("left", 0, -1), ("right", 0, 1))
_DELTA = {name: (dr, dc) for name, dr, dc in _D4}

#: Brendan's mood cycle: pressing ACTION replaces him with the next mood. Three
#: disjoint cycles ship (a 2-, a 3- and a 4-cycle) plus "Me", who has no ACTION
#: rule and so maps to himself. Read off the ``[action JeffX] -> [JeffY]`` rules
#: and confirmed by pressing ACTION six times on one level of each family.
_MOOD_NEXT = {
    "jeffhate": "jefflove", "jefflove": "jeffhate",
    "jefforange": "jeffhate2", "jeffhate2": "jefflove2", "jefflove2": "jefforange",
    "jeffgreen": "jefflove3", "jefflove3": "jefforange2",
    "jefforange2": "jeffhate3", "jeffhate3": "jeffgreen",
    "me": "me",
}

#: The one crate each mood can push. The Key is pushable by every mood and is
#: handled separately; everything else in the crate layer just blocks.
_MOOD_PUSH = {
    "jeffhate": "hateball", "jeffhate2": "hateball", "jeffhate3": "hateball",
    "jefflove": "loveball", "jefflove2": "loveball", "jefflove3": "loveball",
    "jefforange": "orangeball", "jefforange2": "orangeball",
    "jeffgreen": "greenball", "me": "lucozade",
}

_CRATES = ("hateball", "loveball", "orangeball", "greenball", "key", "lucozade")

#: Heuristic charge for a state that can never win -- the goal piece parked where
#: no push sequence can ever bring it to the lock. Large enough to sink the node,
#: finite so the search stays complete.
_DEAD = 10_000

#: Extra estimated cost per crate the goal piece's push route has to displace
#: first. Purely a search guide, and the reason the heuristic is not a bound.
_CLUTTER = 4

#: ``(weight, node budget)`` pairs. EVERY pair is run and the SHORTEST winning
#: plan is kept -- see the module docstring on why "first that succeeds" is the
#: wrong rule here. Each of the three is the sole best weight on at least one
#: level (2 on levels 3 and 5, 3 on level 4, 4 on level 1) and none of them wins
#: everywhere, which is the whole argument for running all three. One-time cost
#: at seed 0, ~75s for all ten levels together.
_WEIGHTS = ((2, 400_000), (3, 400_000), (4, 400_000))

#: Why the rendering is legible after the adapter collapses PuzzleScript colours
#: onto the 16 ARC indices (see the palette-collision check every ps: game needs).
#: Board cells are 5px and the sprites are 5x5, so nothing is lost to downsampling.
#:
#:   background 4/13/14 dither   terrain (wall) 1/2      lock 4 + 12
#:   JeffHate 6                  HateBall 6 + a 0 pip    (stick figure vs disc)
#:   JeffLove 9/10               LoveBall 9/10 + 0 pip
#:   JeffOrange 8/12             OrangeBall 8/12 + 0 pip
#:   JeffGreen 14/11             GreenBall 14/11 + 0 pip
#:   Me 7/0/15                   Lucozade 8/12           Key 12   Bed 13/4/1/15
#:
#: The colour pairs that DO collapse never share a level: JeffOrange and Lucozade
#: are both 8/12 but Lucozade only ever appears with Me, and Key (solid 12) only
#: ever appears with the Jeffs. Player-vs-its-own-ball is the pair that matters
#: most and is safe by shape, not by colour: every Brendan is an open stick figure
#: with a two-pixel gap along its bottom row, every ball is a filled disc.
_PALETTE_NOTES = None


# ---------------------------------------------------------------------------
# Native model of the mechanic
# ---------------------------------------------------------------------------

class _Board:
    """The static scenery of one level plus the game's dynamics.

    A state is the triple ``(player_cell, mood_id, {cell: crate_id})``. Walls,
    locks and beds never change, so they live here; `step` is the whole mechanic.
    """

    def __init__(self, eng, game):
        idx = game.obj_name_to_idx
        grid = eng.grid
        self.h, self.w = len(grid), len(grid[0])
        self.key = idx["key"]
        self.luco = idx["lucozade"]
        self.me = idx["me"]
        self.crate_ids = frozenset(idx[n] for n in _CRATES)
        self.player_ids = frozenset(idx[n] for n in _MOOD_NEXT)
        self.next_mood = {idx[a]: idx[b] for a, b in _MOOD_NEXT.items()}
        self.mood_push = {idx[a]: idx[b] for a, b in _MOOD_PUSH.items()}
        cells = [(r, c) for r in range(self.h) for c in range(self.w)]
        self.walls = frozenset(p for p in cells if idx["terrain"] in grid[p[0]][p[1]])
        self.locks = frozenset(p for p in cells if idx["lock"] in grid[p[0]][p[1]])
        self.beds = frozenset(p for p in cells if idx["bed"] in grid[p[0]][p[1]])

    def read(self, eng):
        """Engine grid -> ``(player_cell, mood_id, {cell: crate_id})``."""
        player = mood = None
        crates = {}
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                hit = cell & self.player_ids
                if hit:
                    player, mood = (r, c), next(iter(hit))
                hit = cell & self.crate_ids
                if hit:
                    crates[(r, c)] = next(iter(hit))
        return player, mood, crates

    # -- dynamics ----------------------------------------------------------
    def step(self, player, mood, crates, direction):
        """One turn. Returns ``(player, mood, crates, rule_win)``. ``crates`` is a
        new dict only when something actually moved, so the common no-op and
        plain-walk cases allocate nothing.

        ``rule_win`` is the ``-> win`` RULE firing (Lucozade shoved into the lock,
        Me stepping onto the bed), which is a TRANSITION and not visible in the
        resulting state -- levels 6-8 win only this way. `condition_win` covers
        the other half.
        """
        if direction == "action":
            return player, self.next_mood[mood], crates, False
        dr, dc = _DELTA[direction]
        t = (player[0] + dr, player[1] + dc)
        if not (0 <= t[0] < self.h and 0 <= t[1] < self.w) or t in self.walls:
            return player, mood, crates, False
        occ = crates.get(t)
        if occ is None:
            return t, mood, crates, mood == self.me and t in self.beds
        if occ == self.key:
            chain = [t]                       # [> Player|Key], any mood, no chain
        elif occ == self.luco:
            if mood != self.me:
                return player, mood, crates, False
            chain = [t]                       # [> Me|Lucozade], no chain rule
        elif occ == self.mood_push.get(mood):
            chain = []                        # [> Ball|Ball] chains same colour
            cell = t
            while crates.get(cell) == occ:
                chain.append(cell)
                cell = (cell[0] + dr, cell[1] + dc)
        else:
            return player, mood, crates, False   # wrong colour: turn cancelled
        end = (chain[-1][0] + dr, chain[-1][1] + dc)
        if not (0 <= end[0] < self.h and 0 <= end[1] < self.w) \
                or end in self.walls or end in crates:
            return player, mood, crates, False
        moved = dict(crates)
        for cell in reversed(chain):
            moved[(cell[0] + dr, cell[1] + dc)] = moved.pop(cell)
        return t, mood, moved, occ == self.luco and end in self.locks

    def condition_win(self, player, mood, crates):
        """The three ``all X on Y`` win conditions, ANDed, with PuzzleScript's
        vacuous-truth rule for an absent X (an absent Key is not a failure)."""
        for cell, o in crates.items():
            if (o == self.key or o == self.luco) and cell not in self.locks:
                return False
        return not (mood == self.me and player not in self.beds)


def _fuzz_check(trials: int = 4000, max_len: int = 25, seed: int = 1234) -> int:
    """Replay random action sequences through the interpreter AND `_Board` and
    compare after every single press. Returns the mismatch count.

    This is what earns the right to plan on the model instead of the engine. It
    compares the FULL state -- every crate cell, the player cell, the mood -- and
    the win flag, not just the win flag, because a model that drifts only in
    where the balls ended up still returns plans the certification will reject
    and levels will silently go missing.

    Run it after any edit to `_Board.step`:
        python solvers/generate_brendan_loves_mondays_training.py --fuzz
    """
    import random

    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    rng = random.Random(seed)
    presses = ["up", "down", "left", "right", "action"]
    per_level = max(1, trials // game.n_levels)
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = _Board(eng, game._game)
        start = snapshot(eng)
        origin = board.read(eng)
        for _trial in range(per_level):
            restore(eng, start)
            eng._rule_win = False
            player, mood, crates = origin[0], origin[1], dict(origin[2])
            won = False
            for _ in range(rng.randint(1, max_len)):
                direction = rng.choice(presses)
                eng.step(direction)
                player, mood, crates, rule_win = board.step(
                    player, mood, crates, direction)
                won = won or rule_win
                if (board.read(eng) != (player, mood, crates)
                        or eng.check_win() != (won or board.condition_win(
                            player, mood, crates))):
                    bad += 1
                    print(f"MISMATCH on level {level} after {direction!r}")
                    break
                if eng.check_win():
                    break
        restore(eng, start)
    print(f"fuzz: {per_level * game.n_levels} rollouts, {bad} mismatches")
    return bad


# ---------------------------------------------------------------------------
# Macro A* over the model
# ---------------------------------------------------------------------------

class _Planner:
    """A* whose successors are ``walk to the push cell, then push`` macros, plus
    ``press ACTION`` and (level 9) ``walk onto the bed``. See the module
    docstring for the state key and the heuristic."""

    def __init__(self, board, weight, node_cap):
        self.b = board
        self.weight = weight
        self.node_cap = node_cap
        self.piece = None
        self.goals = ()
        self.kind = "none"
        self.table = {}

    # -- goal + distance tables --------------------------------------------
    def _set_goal(self, crates):
        """Pick the level's objective and build its static distance table.

        Which objective a level has is read off the board, not hardcoded: a lock
        plus a key is the Sokoban proper, a lock plus a bottle is the Lucozade
        ending, and a bed is the walk-to-bed ending."""
        b = self.b
        present = set(crates.values())
        if b.locks and b.key in present:
            self.piece, self.goals, self.kind = b.key, b.locks, "push"
        elif b.locks and b.luco in present:
            self.piece, self.goals, self.kind = b.luco, b.locks, "push"
        elif b.beds:
            self.piece, self.goals, self.kind = None, b.beds, "walk"
        else:
            self.piece, self.goals, self.kind = None, (), "none"
        self.table = (self._push_table() if self.kind == "push"
                      else self._walk_table() if self.kind == "walk" else {})

    def _push_table(self):
        """``{cell: fewest pushes to get the goal piece from cell onto a goal}``,
        ignoring the other crates (the standard sokoban relaxation). A cell
        MISSING from this table can never reach the lock even with the board
        cleared, so it is a true deadlock and `_heuristic` may prune it."""
        table = {}
        for goal in self.goals:
            steps = {goal: 0}
            queue = deque([goal])
            while queue:
                r, c = queue.popleft()
                cost = steps[(r, c)]
                for _d, dr, dc in _D4:
                    src = (r - dr, c - dc)          # where the piece comes from
                    stand = (r - 2 * dr, c - 2 * dc)  # where the player stands
                    if not self._open(src) or not self._open(stand):
                        continue
                    if src not in steps:
                        steps[src] = cost + 1
                        queue.append(src)
            for cell, d in steps.items():
                if d < table.get(cell, 1 << 30):
                    table[cell] = d
        return table

    def _walk_table(self):
        steps = {g: 0 for g in self.goals}
        queue = deque(self.goals)
        while queue:
            r, c = queue.popleft()
            for _d, dr, dc in _D4:
                nxt = (r + dr, c + dc)
                if self._open(nxt) and nxt not in steps:
                    steps[nxt] = steps[(r, c)] + 1
                    queue.append(nxt)
        return steps

    def _open(self, cell):
        b = self.b
        return (0 <= cell[0] < b.h and 0 <= cell[1] < b.w
                and cell not in b.walls)

    # -- heuristic ---------------------------------------------------------
    def _heuristic(self, player, crates):
        if self.kind == "none":
            return 0
        if self.kind == "walk":
            return self.table.get(player, _DEAD)
        pieces = [cell for cell, o in crates.items() if o == self.piece]
        if not pieces:
            return 0
        if any(p not in self.table for p in pieces):
            return _DEAD                    # buried where no push route exists
        field = self._clutter_field(crates)
        total = sum(field.get(p, self.table[p]) for p in pieces)
        # The player still has to walk to the first push, from BESIDE the piece.
        return total + max(0, min(abs(player[0] - r) + abs(player[1] - c)
                                  for r, c in pieces) - 1)

    def _clutter_field(self, crates):
        """`_push_table` re-run as a Dijkstra on the LIVE board, charging
        `_CLUTTER` for each other crate sitting on the pushed-through cell or on
        the cell the player must stand on. This is the term that keeps the
        estimate from going flat while balls are being cleared out of the way."""
        piece = self.piece
        dist = {g: 0 for g in self.goals}
        pq = [(0, g) for g in self.goals]
        heapq.heapify(pq)
        while pq:
            d, (r, c) = heapq.heappop(pq)
            if d > dist.get((r, c), 1 << 30):
                continue
            for _d, dr, dc in _D4:
                src = (r - dr, c - dc)
                stand = (r - 2 * dr, c - 2 * dc)
                if not self._open(src) or not self._open(stand):
                    continue
                nxt = d + 1
                if crates.get((r, c), piece) != piece:
                    nxt += _CLUTTER
                if crates.get(stand, piece) != piece:
                    nxt += _CLUTTER
                if nxt < dist.get(src, 1 << 30):
                    dist[src] = nxt
                    heapq.heappush(pq, (nxt, src))
        return dist

    # -- macros ------------------------------------------------------------
    def _macros(self, player, mood, crates):
        """``(region, macros)`` for one state: the canonical id of the player's
        walkable region, and every macro available from it as a list of primitive
        directions."""
        b = self.b
        parent = {player: None}
        queue = deque([player])
        while queue:
            cur = queue.popleft()
            r, c = cur
            for d, dr, dc in _D4:
                nxt = (r + dr, c + dc)
                if self._open(nxt) and nxt not in crates and nxt not in parent:
                    parent[nxt] = (cur, d)
                    queue.append(nxt)

        def walk_to(cell):
            out = []
            while parent[cell] is not None:
                cell, d = parent[cell]
                out.append(d)
            out.reverse()
            return out

        pushable = b.mood_push.get(mood)
        macros = []
        for (r, c), o in crates.items():
            if o != b.key and o != pushable:
                continue
            for d, dr, dc in _D4:
                stand = (r - dr, c - dc)
                if stand in parent:
                    macros.append(walk_to(stand) + [d])
        if b.next_mood[mood] != mood:
            macros.append(["action"])
        if self.kind == "walk":
            # Level 9's win is the player MOVING onto the bed, which no push
            # macro can express; without this the macro search closes with the
            # goal one step away.
            for goal in self.goals:
                if goal in parent:
                    walk = walk_to(goal)
                    if walk:
                        macros.append(walk)
        return min(parent), macros

    # -- search ------------------------------------------------------------
    def plan(self, player, mood, crates):
        b = self.b
        self._set_goal(crates)
        if b.condition_win(player, mood, crates):
            return []
        region, macros = self._macros(player, mood, crates)
        weight = self.weight
        counter = nodes = 0
        pq = [(weight * self._heuristic(player, crates), 0, counter,
               (player, mood, crates), [], macros)]
        best = {(frozenset(crates.items()), region, mood): 0}
        while pq:
            _f, g, _c, state, path, macros = heapq.heappop(pq)
            for macro in macros:
                p, m, cr = state
                won = -1
                for i, direction in enumerate(macro):
                    p, m, cr, rule_win = b.step(p, m, cr, direction)
                    if rule_win or b.condition_win(p, m, cr):
                        won = i
                        break
                nodes += 1
                if won >= 0:
                    # Truncate at the winning press: a plan that wins partway and
                    # keeps stepping silently un-wins itself on the interpreter,
                    # whose win flag is per-step.
                    return path + macro[:won + 1]
                if (p, m, cr) == state:
                    continue              # a move the mechanic refused outright
                nregion, nmacros = self._macros(p, m, cr)
                key = (frozenset(cr.items()), nregion, m)
                ng = g + len(macro)
                if best.get(key, 1 << 30) <= ng:
                    continue
                best[key] = ng
                counter += 1
                heapq.heappush(pq, (ng + weight * self._heuristic(p, cr), ng,
                                    counter, (p, m, cr), path + macro, nmacros))
                if nodes >= self.node_cap:
                    return None
        return None


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class BrendanExpert(PSExpert):
    """Plans on `_Board` and certifies every plan on the interpreter.

    The plan memo, the level scoping and the snapshot/restore discipline are
    `PSExpert`'s; only the search strategy (`_search`) is replaced."""

    #: `_key` is `PSExpert`'s full-grid canonical key, so plans are safe to share
    #: across levels and this stays False.
    scope_by_level = False

    def setup(self) -> None:
        self._opt: dict[tuple, list[list[str]]] = {}

    def heuristic(self, eng) -> int:
        """Unused: `_search` plans on the native model, so `PSExpert._astar` --
        the only caller -- never runs."""
        raise NotImplementedError("BrendanExpert plans on the native model")

    def _search(self, eng):
        board = _Board(eng, self.g)
        player, mood, crates = board.read(eng)
        if player is None:
            return None
        shortest = None
        for weight, cap in _WEIGHTS:
            plan = _Planner(board, weight, min(cap, self.node_cap)).plan(
                player, mood, dict(crates))
            if plan is None:
                continue
            plan = self._certify(eng, plan)
            if plan is not None and (shortest is None or len(plan) < len(shortest)):
                shortest = plan
        return shortest

    @staticmethod
    def _certify(eng, plan):
        """Replay ``plan`` through the real interpreter and return it truncated at
        the press that wins, or None if it never does. This is what makes the
        native model safe to plan on: a modelling slip costs a level, never a
        recorded trajectory that does not win."""
        snap = snapshot(eng)
        try:
            for i, direction in enumerate(plan):
                eng.step(direction)
                if eng.check_win():
                    return plan[:i + 1]
            return None
        finally:
            restore(eng, snap)

    # -- optimal-action targets --------------------------------------------
    def plan(self, eng, level=None):
        """`PSExpert.plan`, plus the per-step optimal SETS for the plan it
        returns. `PSExpert.plan` restores the engine before returning, so ``eng``
        is still at the state the plan starts from when they are computed."""
        sol = super().plan(eng, level)
        if sol:
            key = (level, tuple(sol))
            if key not in self._opt:
                self._opt[key] = self._optimal_sets(eng, sol)
        return sol

    def optimal_sets(self, level, plan):
        return self._opt.get((level, tuple(plan)))

    def _optimal_sets(self, eng, plan):
        """The optimal action SET at each step of ``plan``.

        Every macro is ``walk to a cell, then act``, and the walk is a shortest
        path. Every OTHER shortest walk to the same cell reaches exactly the same
        state for exactly the same cost, so all of them are equally optimal and
        the recorder must not train one arbitrary interleaving of the two axes as
        the only right answer. The acting press -- the push, the ACTION mood flip,
        the step onto the bed -- has no such freedom, so its set is just itself.

        Nothing here claims the PLAN is shortest (the heuristic is not a bound);
        the claim is only that these alternatives are interchangeable with the
        step actually taken."""
        board = _Board(eng, self.g)
        player, mood, crates = board.read(eng)
        sets: list[list[str]] = []
        i, n = 0, len(plan)
        while i < n:
            # Scan forward to the first press with an EFFECT. `step` hands back
            # the very same crate dict when nothing moved, so "acts" is exactly
            # "won, or the mood changed, or the crates object is a new one".
            end, j = player, i
            probe = (player, mood, crates)
            while j < n:
                nxt = board.step(probe[0], probe[1], probe[2], plan[j])
                if nxt[3] or nxt[1] != probe[1] or nxt[2] is not probe[2] \
                        or nxt[0] == probe[0]:
                    break
                probe, j, end = nxt[:3], j + 1, nxt[0]
            walk = plan[i:j]
            if walk:
                # Distances TO the cell the walk ends on, over the cells the walk
                # may use. The crates cannot move during a walk, so one BFS does
                # for the whole stretch.
                dist = self._walk_dist(board, crates, end)
                here = player
                literal = dist.get(player) != len(walk)   # not a shortest walk
                for direction in walk:
                    best = [direction]
                    if not literal:
                        best = sorted(
                            d for d, dr, dc in _D4
                            if dist.get((here[0] + dr, here[1] + dc), 1 << 30)
                            == dist[here] - 1)
                        if direction not in best:
                            best = [direction]
                    sets.append(best)
                    dr, dc = _DELTA[direction]
                    here = (here[0] + dr, here[1] + dc)
            if j < n:
                sets.append([plan[j]])
            for k in range(i, min(j + 1, n)):
                player, mood, crates, _won = board.step(
                    player, mood, crates, plan[k])
            i = j + 1
        return sets

    @staticmethod
    def _walk_dist(board, crates, target):
        """BFS from ``target`` over the walkable cells (no wall, no crate)."""
        dist = {target: 0}
        queue = deque([target])
        while queue:
            r, c = queue.popleft()
            for _d, dr, dc in _D4:
                nxt = (r + dr, c + dc)
                if (0 <= nxt[0] < board.h and 0 <= nxt[1] < board.w
                        and nxt not in board.walls and nxt not in crates
                        and nxt not in dist):
                    dist[nxt] = dist[(r, c)] + 1
                    queue.append(nxt)
        return dist


class BrendanLovesMondaysSolver(PSAStarSolver):
    game_id = "puzzlescript_brendan_loves_mondays"
    game_name = GAME_NAME
    expert_cls = BrendanExpert

    #: `weight` is unused -- `_WEIGHTS` carries the weight of each search -- and
    #: `node_cap` is only an upper clamp on the per-search budgets in `_WEIGHTS`,
    #: so it is set above the largest of them.
    node_cap = 2_000_000
    weight = 1

    #: The longest plan is 86 moves; the ceiling only has to cover a re-plan after
    #: an epsilon detour, and `epsilon` is 0 for this family.
    max_steps = 250

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        sets = expert.optimal_sets(level, plan)
        if not sets or pi >= len(sets):
            return None
        return sets[pi]


if __name__ == "__main__":
    if "--fuzz" in sys.argv:
        sys.exit(1 if _fuzz_check() else 0)
    sys.exit(BrendanLovesMondaysSolver.main())
