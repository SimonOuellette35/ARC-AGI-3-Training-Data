"""Generate Phase-1 training data for the PuzzleScript game ps:silver_lungs
("silver lungs", zuza).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators. This file is the game-specific part: a native model of one
turn, the exact distance field over every level's whole reachable space, and the
reports that prove the model, the plans and the labels.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_silver_lungs",
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
Two islands of grey Ground float in a blue Void, and the Crate has to be pushed
onto the dark-red Target -- which is on the other island. Nothing in the game
lets you cross. What lets you cross is the PLATFORM: the dark-grey Ground2 tiles
carrying a purple Magic marker are the only walkable thing over the Void, and
they SLIDE. **Pushing the special crate one cell slides every magic platform
tile one cell the same way, anywhere on the board.** Line the platform up with
the gap in the island and it is a bridge; step onto it and shove the special
crate again and it is a ferry -- except it does not carry you, only crates.

That is the whole mechanic, and the .txt does not say it. The rules read as a
chain that ought to be confined to the special crate's own row or column: the
crate hands a force to the Beam behind it and to the Beams ahead of it, the
Beams pass it along `[ > Beam | Beam ]`, and a moving Beam hands it to a Magic
"somewhere ahead". Beam is an invisible object that the level-start rules breed
over EVERY cell of the board, and in this interpreter the ``...`` gap is not
restricted to the beam's line, so in practice the force reaches every Magic on
the board -- measured (``--moves``), not read off the rules.

One turn, measured
------------------
Everything below was driven on the interpreter (``--moves`` re-drives all
fifteen scenarios and asserts them; ``--selfcheck`` fuzzes the model against it
over random walks):

* **Grounds gate every move.** `Grounds` is `Ground or Ground2`, so a platform
  tile is as walkable as the island. A press into a cell with no Grounds is
  refused outright, and so is a push whose crate would land on one -- and the
  refusal cancels the player's own step with it, because the push rule binds
  both movements into one match.
* **The special crate moves the platform, but only if the way is clear of a
  plain crate.** The beam rules carry `no Crates` / `Crate Magic` guards on the
  cell ahead of it, so shoving the special crate into an ordinary crate is a
  total no-op -- neither crate moves and the platform does not budge. A crate
  that is itself standing ON a platform tile is the exception: it rides.
* **A crate standing on a platform tile rides with it, the player does NOT.**
  `[ > Magic Ground2 Crates | ] -> [ > Magic > Ground2 > Crates | ]` names
  `Crates`, which is Crate-or-CrateSpecial and not Player. So a crate can be
  ferried across the Void, and a player standing on a tile that slides away is
  left standing on the bare Void -- legally: the rule that blocks the Void
  blocks moving INTO it, not standing on it, and the player can step back off
  onto anything with Grounds.
* **The platform and the crate that rides it can COME APART**, which is the
  trap. Magic and Ground2 have their own collision layers and nothing on the
  board can block them; the rider is on the crate layer and can be blocked by
  any crate. So a riding crate wedged against another crate stays put while the
  tile slides out from under it, and the same happens to the special crate
  itself when its rider cannot move: the whole crate layer freezes and the
  platform still slides. Displacement of the platform is therefore NOT locked to
  displacement of the special crate, which is why the model carries the magic
  cells explicitly instead of an offset.
* **The board EDGE stops a tile without stopping the others.** A tile with no
  cell ahead of it cannot move and blocks the tile behind it (Magic collides
  with Magic), so a platform pressed into the edge compresses rather than
  translating. Ground2 does the same thing for the same reason, which is what
  keeps it exactly co-located with Magic in every reachable state (asserted).
* **ACTION does nothing.** No rule reads it and there is no `late` rule for it
  to tick, so it cannot change the board -- ``--selfcheck`` presses it against
  the interpreter rather than trusting the rules section (ps:ouroboros's ACTION
  is a shrink button and its rules section does not mention ``action`` either).
* There is no `restart` and no `again` in the game, nothing is ever created or
  destroyed, and `check_win` (``All Target on Crate``) is False on every level's
  start frame, so none of this family's opt-in escapes (`vacuous_start_win`, the
  ps:impasse restart trap) applies.

Seven levels, 10x13, 10x14 and 8x9. Level indices here and in every report are
0-BASED.

Why a native model, and not the shared A*
-----------------------------------------
``eng.step`` runs at ~70 presses/second on these boards: the invisible Beam
covers every cell, and the four ``[ > Beam | ... | Magic ]`` rules re-scan every
beam against every other cell of its line on every turn. That is a sixth of
ps:entrepotphage_demake's rate, which is the game this family already decided
was too slow to search through (see the ps-astar-generator-family note), so the
interpreter here CERTIFIES rather than plans.

The model pays for itself immediately, because the reachable space is tiny:
3318 / 7129 / 7537 / 11678 / 3355 / 5881 / 15397 states, the whole game
enumerated in about a second. So there is no search at all. `_Board.enumerate`
is one forward BFS from the level start collecting every state and its four
successors and one backward BFS from the winning states over the edges it
inverts, and everything else reads that field:

* plans that are provably SHORTEST -- no weight, no heuristic, no node cap that
  could quietly bite: 19, 29, 29, 19, 35, 17 and 12 presses;
* exact optimal-action SETS, ``dist(succ) == dist - 1``, rather than inferred
  ones;
* a proof of winnability, and an exact census of the states that can no longer
  win -- large here, because a crate shoved off the platform's line, or a
  platform slid past the gap it was meant to bridge, does not come back.

Because that costs about a second there is no ``plan_cache_path``: every
`parallelize_generator` shard re-derives the field, which is cheaper than the
staleness risk a cached file would carry.

Optimal-action sets
-------------------
``optimal_for`` reads the field at the LIVE engine state before each press, so
the label is right after an epsilon detour too -- where the state is off the
plan entirely and a set derived from the plan's own step index would be
labelling the wrong board. A press that is refused maps the state to itself,
whose distance is unchanged rather than one less, so no-ops are excluded for
free, and so is every shove that kills the level, since a dead state has no
distance at all. ``--verify`` re-derives the whole field by an independent
Bellman fixpoint and then presses every shipped label on the interpreter.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the plan
replays to a guaranteed WIN. ``epsilon = 0.12`` on top of it, which is a
departure from most of this family and is earned by the exact field:
`record_level`'s detour probe keeps a random alternative only if the expert can
still plan a win from it, and here that test is a dict lookup over the closed
reachable set rather than a bounded search that might decline a live state. So
roughly one press in eight is a deliberate mistake the recording then recovers
from, with the correct set labelled at every step of the recovery.

Rendering
---------
`data/puzzlescript_games/silver_lungs.txt` ships with six sprites redrawn for
this corpus (its header explains each one; ``--audit`` is the check). The win
was not drawn at all -- Crate is the higher collision layer and its solid block
covered the Target's small ring exactly, so the winning board rendered as a
plain crate -- and Ground2's solid square hid whether the platform is over
Ground or over Void. Every layer now owns disjoint pixels of the cell (Target
the corners, Ground2 and Magic the side midpoints, the crates and the player the
interior), chosen to survive the cell_px 4 and 7 sampling these board sizes
produce, so nothing occludes anything. No colour name, object, collision layer,
rule, legend entry, level or win condition was touched.

CLI
---
    --plans      per-level space, dead count, plan length and tie coverage
    --selfcheck  fuzz the native model against the interpreter (all 5 presses)
    --moves      re-drive the fifteen measured mechanics scenarios
    --verify     Bellman fixpoint + every shipped label pressed on the engine
    --symmetry   prove the 8-element presentation group is exact
    --audit      assert every reachable cell composition renders distinctly
"""

from __future__ import annotations

import itertools
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np                                            # noqa: E402

from adapters.puzzlescript_adapter import (                   # noqa: E402
    PuzzleScriptAdapter, _render_frame,
)
from solvers.common.ps_astar import (                         # noqa: E402
    PSAStarSolver, PSExpert, restore, snapshot,
)

GAME_NAME = "silver_lungs"

#: Engine direction -> (dr, dc).
_DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

#: The presses the search branches on, and the plan's tie-break order (which is
#: what makes a re-derived plan byte-identical across processes). ``action``
#: parses but no rule reads it -- `selfcheck` asserts that against the
#: interpreter rather than trusting the rules section.
_DIRS = ("up", "down", "left", "right")


# ---------------------------------------------------------------------------
# The level, natively: one turn, the whole state space, the distance field
# ---------------------------------------------------------------------------

class _Board:
    """A level's static geometry plus its COMPLETE state space and the exact
    distance-to-win field over it.

    Cells are flat ``r * w + c`` ids. A state is

        ``(player, crates, cratespecial, magic)``

    with the two sets held as sorted tuples. Ground, Void, Target (and Wall,
    which no level of this game ships) are static -- no rule creates, destroys
    or moves any of them -- and Ground2 is exactly co-located with Magic in every
    reachable state, so those four fields ARE the state.

    The magic cells are carried explicitly rather than as one offset from the
    special crate, even though a press normally moves both by the same step: a
    riding crate that cannot move freezes the whole crate layer while the
    platform slides on regardless (see the module docstring), so the two
    displacements genuinely come apart.
    """

    __slots__ = ("h", "w", "ground", "targets", "start", "_next",
                 "states", "index", "succ", "dist")

    def __init__(self, h: int, w: int, ground, targets, start):
        self.h, self.w = h, w
        #: ``ground[cell]`` -- the static Ground object. Ground and Void
        #: partition the board (asserted in `from_engine`), so this doubles as
        #: "has Grounds" once the magic cells of the state are added to it, and
        #: that one predicate answers both the walk rule and the push rule.
        self.ground = list(ground)
        self.targets = tuple(sorted(targets))
        self.start = start

        # ``_next[d][cell]`` is the cell one step along d, or -1 off the board.
        # Precomputed because it is the inner loop of the enumeration.
        self._next = {}
        for d, (dr, dc) in _DELTA.items():
            table = [-1] * (h * w)
            for r in range(h):
                for c in range(w):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        table[r * w + c] = nr * w + nc
            self._next[d] = table

        self.states: list | None = None
        self.index: dict = {}
        self.succ: list = []
        self.dist: list = []

    # -- construction ---------------------------------------------------------
    @classmethod
    def from_engine(cls, eng, ids) -> "_Board":
        """Read the level off the engine grid.

        The invariants this game's model rests on are asserted here rather than
        assumed, because each of them would make the model WRONG (not slow) if a
        level ever broke it: Ground and Void partition the board, Ground2 is
        exactly the Magic cells, no Target stands on the Void, and there is
        exactly one Player, one CrateSpecial and no Wall.
        """
        h, w = eng.height, eng.width
        cells = {n: set() for n in ("ground", "void", "target", "wall",
                                    "player", "crate", "cratespecial",
                                    "magic", "ground2")}
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                for name, bucket in cells.items():
                    if ids[name] in cell:
                        bucket.add(r * w + c)
        every = set(range(h * w))
        assert cells["ground"] | cells["void"] == every, \
            "a cell is neither Ground nor Void"
        assert not (cells["ground"] & cells["void"]), \
            "a cell is both Ground and Void"
        assert cells["ground2"] == cells["magic"], \
            "Ground2 is not exactly the Magic cells"
        assert not (cells["target"] - cells["ground"]), \
            "a Target stands on the Void"
        assert not cells["wall"], "this level ships a Wall"
        assert len(cells["player"]) == 1, "not exactly one Player"
        assert len(cells["cratespecial"]) == 1, "not exactly one CrateSpecial"
        ground = [False] * (h * w)
        for cell in cells["ground"]:
            ground[cell] = True
        start = (next(iter(cells["player"])),
                 tuple(sorted(cells["crate"])),
                 next(iter(cells["cratespecial"])),
                 tuple(sorted(cells["magic"])))
        return cls(h, w, ground, cells["target"], start)

    # -- one turn -------------------------------------------------------------
    def won(self, state) -> bool:
        """``All Target on Crate`` -- the plain Crate, never the special one."""
        crates = state[1]
        return all(t in crates for t in self.targets)

    @staticmethod
    def _chain(nxt, forces, occupied) -> set:
        """The subset of ``forces`` that actually moves, by the interpreter's
        chain rule: an object moves iff the run of occupied cells ahead of it is
        entirely made of objects pushed the same way and ends at a free cell on
        the board. A blocked run cancels every object in it, which is what makes
        a push into a wedged crate a complete no-op rather than a partial one.

        Every force in one turn of this game points the same way (there is one
        press and every rule passes it straight through), so two objects can
        never claim the same cell from different sides and the multi-way
        conflict case of `PSEngine._resolve_forces` cannot arise.
        """
        verdict: dict = {}
        for cell in forces:
            chain = []
            cur = cell
            while True:
                if cur in verdict:
                    v = verdict[cur]
                    break
                chain.append(cur)
                nex = nxt[cur]
                if nex < 0 or (nex in occupied and nex not in forces):
                    v = False
                    break
                if nex not in occupied:
                    v = True
                    break
                cur = nex
            for x in chain:
                verdict[x] = v
        return {c for c in forces if verdict[c]}

    def resolve(self, state, d: str):
        """``(crate-layer movers, platform movers, the platform moved)`` for one
        press, or None when the press is refused outright.

        Split out of `step` so the reports can read the branch the turn actually
        took (`_classify`) rather than guess it back out of a before/after diff
        -- the interesting cases here are the ones where the diff is misleading
        (a shove into a crate and an ordinary walk into a wall are both "nothing
        moved", and only one of them is the beam guard firing).
        """
        p, crates, sp, magic = state
        nxt = self._next[d]
        ground, mset = self.ground, set(magic)
        n1 = nxt[p]
        if n1 < 0 or not (ground[n1] or n1 in mset):
            return None                        # no Grounds: the walk is refused
        cset = set(crates)
        forces = {p}
        magic_move = False
        if n1 == sp or n1 in cset:
            n2 = nxt[n1]
            if n2 < 0 or not (ground[n2] or n2 in mset):
                return None                    # the shove has nowhere to land
            forces.add(n1)
            if n1 == sp:
                # The beam guards: a plain crate in the way stops the platform
                # dead, unless that crate is itself standing on a platform tile.
                magic_move = n2 not in cset or n2 in mset
        if magic_move:
            # `[ > Magic Ground2 Crates | ]` -- a crate on a tile rides it. The
            # bounds test is the rule's own trailing ``|``: a tile against the
            # edge hands nothing on (and could not move anyway).
            for m in magic:
                if nxt[m] >= 0 and (m == sp or m in cset):
                    forces.add(m)
        movers = self._chain(nxt, forces, cset | {sp, p})
        # Magic and Ground2 own their collision layers, so the only thing that
        # can stop a tile is another tile or the board edge -- and both layers
        # answer that identically, which is what keeps them co-located.
        slid = self._chain(nxt, mset, mset) if magic_move else set()
        return movers, slid, magic_move

    def step(self, state, d: str):
        """One press, natively. Returns the same state object for a no-op."""
        out = self.resolve(state, d)
        if out is None:
            return state
        movers, slid, magic_move = out
        p, crates, sp, magic = state
        nxt = self._next[d]
        shift = lambda cell: nxt[cell] if cell in movers else cell   # noqa: E731
        new_magic = (tuple(sorted(nxt[m] if m in slid else m for m in magic))
                     if magic_move else magic)
        return (shift(p), tuple(sorted(shift(x) for x in crates)),
                shift(sp), new_magic)

    # -- the whole space ------------------------------------------------------
    def enumerate(self) -> tuple[int, int]:
        """Build the reachable state set, its successor table and the exact
        distance-to-win field. Returns ``(states, dead states)``.

        The set is CLOSED under every press (each state contributes all four of
        its successors), so any state the agent can reach by any sequence of
        presses -- plan, exploration prefix, epsilon detour, or a RESET back to
        the start -- is already in it, and the field can answer for all of them.
        Winning states are terminal: the adapter ends the level there, so they
        are recorded but never expanded.

        The dead count is large and that is the mechanic rather than a bug: a
        crate pushed off the line the platform can ferry it along, or a platform
        slid past the gap it was meant to bridge, is not recoverable.
        """
        if self.states is not None:
            return len(self.states), sum(1 for x in self.dist if x < 0)

        states = [self.start]
        index = {self.start: 0}
        succ: list = []
        i = 0
        while i < len(states):
            state = states[i]
            if self.won(state):
                succ.append(None)                    # terminal: the level ends
            else:
                row = []
                for d in _DIRS:
                    nxt = self.step(state, d)
                    j = index.get(nxt)
                    if j is None:
                        j = len(states)
                        index[nxt] = j
                        states.append(nxt)
                    row.append(j)
                succ.append(tuple(row))
            i += 1

        pred: list[list[int]] = [[] for _ in states]
        for src, row in enumerate(succ):
            if row is not None:
                for dst in row:
                    pred[dst].append(src)

        dist = [-1] * len(states)
        queue = deque()
        for j, state in enumerate(states):
            if self.won(state):
                dist[j] = 0
                queue.append(j)
        while queue:
            j = queue.popleft()
            for src in pred[j]:
                if dist[src] < 0:
                    dist[src] = dist[j] + 1
                    queue.append(src)

        self.states, self.index, self.succ, self.dist = states, index, succ, dist
        return len(states), sum(1 for x in dist if x < 0)

    # -- reading the field ----------------------------------------------------
    def distance(self, state) -> "int | None":
        """Presses to a win from ``state``, or None when it can no longer win
        (or is not reachable from the level start, which cannot happen -- see
        `enumerate`)."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] < 0:
            return None
        return self.dist[j]

    def plan(self, state) -> "list | None":
        """A SHORTEST press sequence from ``state`` to a win, or None.

        A descent of the field taking the first tied direction in ``_DIRS``
        order, so the plan is a pure function of the board and does not vary
        with the interpreter's hash seed."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] < 0:
            return None
        out = []
        while self.dist[j] > 0:
            for k, nxt in enumerate(self.succ[j]):
                if self.dist[nxt] == self.dist[j] - 1:
                    out.append(_DIRS[k])
                    j = nxt
                    break
            else:                                    # pragma: no cover
                raise AssertionError("distance field has no descent")
        return out

    def optimal(self, state) -> list:
        """Every press that keeps the game on a SHORTEST route to the win."""
        self.enumerate()
        j = self.index.get(state)
        if j is None or self.dist[j] <= 0:
            return []
        here = self.dist[j]
        return [_DIRS[k] for k, nxt in enumerate(self.succ[j])
                if self.dist[nxt] == here - 1]


# ---------------------------------------------------------------------------
# Expert
# ---------------------------------------------------------------------------

class SilverLungsExpert(PSExpert):
    """Exact planner over the enumerated state space (see `_Board`).

    `PSExpert` supplies the plan memo, the restore discipline and the level
    scoping; `_search` swaps in the field descent, so the interpreter is only
    ever stepped by the recorder -- which is also what certifies every plan,
    since a level is kept only when the engine reports WIN.
    """

    directions = list(_DIRS)

    #: `_key` is the piece tuple, canonical only WITHIN a level: the Ground,
    #: the Void and the Target that complete the state are static per level but
    #: differ between them.
    scope_by_level = True

    def setup(self) -> None:
        self.ids = self.g.obj_name_to_idx
        self._boards: dict[int | None, _Board] = {}
        self._cur: _Board | None = None

    # -- reading the engine ---------------------------------------------------
    def read(self, eng):
        """The ``(player, crates, cratespecial, magic)`` state from the grid."""
        w = eng.width
        ids = self.ids
        player = special = None
        crates, magic = [], []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if ids["player"] in cell:
                    player = r * w + c
                if ids["cratespecial"] in cell:
                    special = r * w + c
                if ids["crate"] in cell:
                    crates.append(r * w + c)
                if ids["magic"] in cell:
                    magic.append(r * w + c)
        if player is None or special is None:            # pragma: no cover
            raise AssertionError("the board is missing a Player or a "
                                 "CrateSpecial")
        return (player, tuple(crates), special, tuple(magic))

    def board(self, eng, level: "int | None") -> _Board:
        """The level's `_Board`, built (and enumerated) once and cached. The
        static geometry is the same whichever state happens to build it."""
        board = self._boards.get(level)
        if board is None:
            board = _Board.from_engine(eng, self.ids)
            board.enumerate()
            self._boards[level] = board
        return board

    # -- planning -------------------------------------------------------------
    def _key(self, eng):
        return self.read(eng)

    def plan(self, eng, level: "int | None" = None) -> "list | None":
        self._cur = self.board(eng, level)
        return super().plan(eng, level)

    def heuristic(self, eng) -> int:
        """The EXACT remaining press count. The base `_astar` is never used
        here, but the contract is that this is 0 at a win and admissible, and
        the field is both."""
        d = self._cur.distance(self.read(eng))
        return 0 if d is None else d

    def _search(self, eng) -> "list | None":
        return self._cur.plan(self.read(eng))

    def optimal_dirs(self, eng, level: "int | None") -> list:
        """Every press on a shortest route, from the engine's CURRENT state."""
        return self.board(eng, level).optimal(self.read(eng))


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class SilverLungsSolver(PSAStarSolver):
    game_id = "puzzlescript_silver_lungs"
    game_name = GAME_NAME
    expert_cls = SilverLungsExpert

    #: Record against the GAME FOLDER's adapter. ``games/ps:silver_lungs`` is a
    #: plain passthrough today; it is named anyway so a sprite patch or a step
    #: cap added there later cannot silently make this generator tape a game
    #: nobody plays (the ps:count_mover trap).
    game_module_id = "ps:silver_lungs"

    #: Unused -- `SilverLungsExpert._search` never calls the base A* -- but left
    #: at a family default so a future subclass that does is not starved.
    node_cap = 2_000_000
    weight = 1

    #: The adapter GAME_OVERs at 200 actions per level; the longest plan here is
    #: 35 presses, leaving the rest of the budget to the exploration prefix and
    #: the detours.
    max_steps = 200

    #: Non-zero even though this game is IRREVERSIBLE, which is a departure from
    #: most of this family and is earned by the exact field: `record_level`'s
    #: detour probe steps a random alternative, asks the expert whether it can
    #: still plan a win, and keeps the detour only if it can. For a bounded
    #: search that test is a guess that can decline a live state; here it is a
    #: dict lookup in a field covering the whole reachable space, so a detour
    #: provably cannot strand the platform or the crate. Roughly one press in
    #: eight is then a random legal alternative, after which the expert re-plans
    #: from wherever it landed: the taken action is the mistake and ``optimal``
    #: is the recovery, which is the signal a policy needs after its own error.
    #: The RESET prefix still runs in front of it.
    epsilon = 0.12

    def prepare_expert(self, game, expert) -> None:
        """Enumerate every level's state space up front -- pure front-loading,
        so the per-level BFS is paid once at startup rather than inside the
        first query that needs it."""
        for level in range(game.n_levels):
            game.set_level(level)
            expert.board(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """Every press that keeps the game on a SHORTEST route to the win, read
        off the LIVE engine state.

        `record_level` asks for this before executing the press, so the engine
        is still at the state being labelled -- which is what makes the label
        right after an epsilon detour too, where the state is off the plan
        entirely and replaying the plan from the level start would label the
        wrong states.

        Falls back to the press about to be taken if the field has nothing to
        say -- no expert step may ship unlabelled."""
        best = expert.optimal_dirs(expert.game._engine, level)
        if best:
            return best
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Scenes: a board written down, for the measured-mechanics table
# ---------------------------------------------------------------------------

#: Scene characters. The four "on a tile" forms exist because a piece standing
#: on a platform tile behaves differently from the same piece on the island,
#: and that difference is most of what `_moves` measures.
_SCENE = {
    "_": (),                                   # bare Void
    "g": ("ground",),
    "o": ("ground", "target"),
    "m": ("magic",),                           # a platform tile over the Void
    "M": ("ground", "magic"),                  # a platform tile over an island
    "P": ("ground", "player"),
    "p": ("magic", "player"),
    "c": ("ground", "crate"),
    "C": ("magic", "crate"),                   # a crate riding a tile
    "*": ("ground", "cratespecial"),
    "x": ("magic", "cratespecial"),
}


def _scene(rows) -> tuple:
    """``(board, state)`` for a hand-written scene -- see `_SCENE`."""
    h, w = len(rows), len(rows[0])
    ground = [False] * (h * w)
    targets, crates, magic = set(), [], []
    player = special = None
    for r, line in enumerate(rows):
        assert len(line) == w, "ragged scene"
        for c, ch in enumerate(line):
            cell = r * w + c
            parts = _SCENE[ch]
            if "ground" in parts:
                ground[cell] = True
            if "target" in parts:
                targets.add(cell)
            if "magic" in parts:
                magic.append(cell)
            if "player" in parts:
                player = cell
            if "crate" in parts:
                crates.append(cell)
            if "cratespecial" in parts:
                special = cell
    state = (player, tuple(sorted(crates)), special, tuple(sorted(magic)))
    return _Board(h, w, ground, targets, state), state


def _seat(eng, ids, board: _Board, state) -> None:
    """Load a model state onto the interpreter's grid.

    Beam is painted over every cell because that is what the level-start rules
    leave on every real board -- ``--selfcheck`` re-checks that invariant after
    every press rather than only assuming it here."""
    p, crates, sp, magic = state
    cset, mset = set(crates), set(magic)
    eng.height, eng.width = board.h, board.w
    grid = []
    for r in range(board.h):
        row = []
        for c in range(board.w):
            cell = r * board.w + c
            objs = {ids["background"], ids["beam"]}
            objs.add(ids["ground"] if board.ground[cell] else ids["void"])
            if cell in board.targets:
                objs.add(ids["target"])
            if cell in mset:
                objs.add(ids["magic"])
                objs.add(ids["ground2"])
            if cell == p:
                objs.add(ids["player"])
            if cell in cset:
                objs.add(ids["crate"])
            if cell == sp:
                objs.add(ids["cratespecial"])
            row.append(objs)
        grid.append(row)
    eng.grid = grid
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


#: The measured mechanics table. Each entry is ``(name, scene, press,
#: expectation)``; the expectation is prose, and what `_moves` actually asserts
#: is that the INTERPRETER and `_Board.step` agree cell for cell -- the prose is
#: there so a reader can see what the scenario was built to catch.
#:
#: The shipped levels do not reach several of these (nothing in them wedges a
#: riding crate, and no platform tile ever touches the board edge), which is the
#: whole point: a certification pass cannot find a mechanic the plans do not
#: happen to use, so the corner cases get boards of their own.
_SCENES = (
    ("walk", ["__________",
              "_gggggggg_",
              "_P*gggggg_",
              "_gggggggg_",
              "____m_____",
              "__________"], "up",
     "a plain step; the platform does not move unless the special crate does"),
    ("push the special", ["__________",
                          "_gggggggg_",
                          "_P*gggggg_",
                          "_gggggggg_",
                          "____m_____",
                          "__________"], "right",
     "the tile slides one cell, in a row of its own"),
    ("the platform is not the crate's line", ["__________",
                                             "_gggggggg_",
                                             "_g*gggggg_",
                                             "_gPgggggg_",
                                             "____m_____",
                                             "_m________",
                                             "________m_",
                                             "__________"], "up",
     "every tile on the board slides, not only the ones in line with it"),
    ("a tile against the edge", ["_________",
                                 "_ggggggg_",
                                 "_P*ggggg_",
                                 "_ggggggg_",
                                 "________m",
                                 "m________",
                                 "_________"], "right",
     "the edge tile stays, the other one moves: the platform compresses"),
    ("tiles compress", ["__________",
                        "_gggggggg_",
                        "_P*gggggg_",
                        "_gggggggg_",
                        "____mm____",
                        "________mm",
                        "__________"], "right",
     "a tile blocked by a tile blocks the one behind it, so nothing merges"),
    ("a crate rides", ["__________",
                       "_gggggggg_",
                       "_P*gggggg_",
                       "_gggggggg_",
                       "____C_____",
                       "__________"], "right",
     "`Crates` in the ride rule, so the crate goes with the tile"),
    ("a rider is wedged", ["__________",
                           "_gggggggg_",
                           "_P*gggggg_",
                           "_gggggggg_",
                           "____Cc____",
                           "__________"], "right",
     "the tile slides out from under the crate: they come apart"),
    ("a rider off the edge", ["__________",
                              "_gggggggg_",
                              "_P*gggggg_",
                              "_gggggggg_",
                              "________C_",
                              "_________C",
                              "__________"], "right",
     "the tile at the edge holds its rider, the other carries its own"),
    ("the special into a crate", ["__________",
                                  "_gggggggg_",
                                  "_P*cggggg_",
                                  "_gggggggg_",
                                  "____m_____",
                                  "__________"], "right",
     "the beam guard fails: a total no-op, the platform does not budge"),
    ("the special into a rider", ["__________",
                                  "_gggggggg_",
                                  "_P*Cggggg_",
                                  "_gggggggg_",
                                  "______m___",
                                  "__________"], "right",
     "`Crate Magic` ahead: the platform moves and the crate rides"),
    ("a wedged rider freezes the crates", ["__________",
                                           "_gggggggg_",
                                           "_P*Ccgggg_",
                                           "_gggggggg_",
                                           "____m_____",
                                           "__________"], "right",
     "nothing on the crate layer moves and the platform slides anyway"),
    ("the special onto the Void", ["__________",
                                   "_ggggg____",
                                   "_P*_______",
                                   "_ggggg____",
                                   "____m_____",
                                   "__________"], "right",
     "no Grounds to land on: refused, and the player's own step with it"),
    # The two scenes below carry an idle CrateSpecial off to one side purely so
    # the board is a legal one (every level of this game has exactly one, and
    # `SilverLungsExpert.read` asserts it); nothing reaches it.
    ("a crate onto the Void", ["__________",
                               "_gggg*____",
                               "_Pc_______",
                               "_ggggg____",
                               "__________"], "right",
     "the same refusal for a plain crate"),
    ("the player onto the Void", ["__________",
                                  "_gggg*____",
                                  "_P________",
                                  "_ggggg____",
                                  "__________"], "right",
     "and for a bare step"),
    ("the special rides its own tile", ["__________",
                                        "_gggggggg_",
                                        "_Pxmggggg_",
                                        "_gggggggg_",
                                        "__________"], "right",
     "Ground2 ahead IS Grounds, so it can be pushed along the platform"),
    ("the player is left on the Void", ["__________",
                                        "_ggggg____",
                                        "_g*pm_____",
                                        "_ggggg____",
                                        "__________"], "left",
     "the player does not ride, but here the platform follows it anyway"),
    ("a crate rides onto a target", ["__________",
                                     "_gggggggg_",
                                     "_P*gggggg_",
                                     "_gggggggg_",
                                     "____Co____",
                                     "__________"], "right",
     "the win can be ferried in, not only pushed in"),
)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _levels():
    """``(solver, game, expert)`` with every level's space enumerated."""
    solver = SilverLungsSolver()
    game, expert, _solvable = solver._ensure(0)
    return solver, game, expert


def _walk(board: _Board, plan) -> list:
    """The states ``plan`` passes through, starting at the level start."""
    out, state = [], board.start
    for d in plan:
        out.append(state)
        state = board.step(state, d)
    return out


def _report() -> int:
    """Print each level's space, its shortest plan and the engine's verdict --
    the quick "is this game still fully solved" check.

    The dead column is expected to be large: a crate shoved off the line the
    platform can ferry it along never comes back, and neither does a platform
    slid past the gap it was meant to bridge. What must hold is that every
    level's START is live, which is the ``win=True`` column.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    bad = states_total = dead_total = presses_total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        n_states, n_dead = board.enumerate()
        states_total += n_states
        dead_total += n_dead
        plan = board.plan(board.start)
        if plan is None:
            bad += 1
            print(f"  L{level}: UNWINNABLE")
            continue
        for d in plan:
            eng.step(d)
        won = eng.check_win()                   # read BEFORE anything reloads
        bad += not won
        presses_total += len(plan)
        ties = sum(len(board.optimal(s)) for s in _walk(board, plan))
        print(f"  L{level}: {eng.height:2d}x{eng.width:2d}  {len(plan):3d} "
              f"presses  win={won}  {n_states:6d} states "
              f"({n_dead:6d} dead, {n_dead / n_states:3.0%})  "
              f"{ties / len(plan):.2f} optimal presses/step")
    print(f"  total: {presses_total} presses over {states_total} states, "
          f"{dead_total} dead ({dead_total / states_total:.0%})")
    return 1 if bad else 0


#: Every mechanic `_classify` can name, in the order the coverage report prints
#: them. Listed rather than collected so a branch that NO fuzz reaches is
#: printed as a zero instead of quietly not appearing.
_MECHANICS = (
    "a step refused", "a shove refused", "a total no-op",
    "the special shoved", "the platform slid",
    "the platform held by a crate", "a crate rode", "a rider came apart",
    "a tile blocked", "the crate layer froze",
    "the player on the Void", "a crate on the Void",
)


def _classify(board: _Board, state, d: str) -> set:
    """Which of `_MECHANICS` one ``(state, press)`` exercises.

    Read off `_Board.resolve`, i.e. off the branch the turn really took, so the
    coverage report cannot be fooled by two different mechanics producing the
    same before/after diff.
    """
    p, crates, sp, magic = state
    nxt = board._next[d]
    ground, mset = board.ground, set(magic)
    tags = set()
    out = board.resolve(state, d)
    if out is None:
        n1 = nxt[p]
        tags.add("a step refused" if n1 < 0 or not (ground[n1] or n1 in mset)
                 else "a shove refused")
    else:
        movers, slid, magic_move = out
        if nxt[p] == sp:
            tags.add("the special shoved")
            tags.add("the platform slid" if magic_move
                     else "the platform held by a crate")
        if magic_move:
            riders = [c for c in crates if c in mset]
            if any(c in movers for c in riders):
                tags.add("a crate rode")
            if any(c in slid and c not in movers for c in riders):
                tags.add("a rider came apart")
            if any(m not in slid for m in magic):
                tags.add("a tile blocked")
            if not movers:
                tags.add("the crate layer froze")
    want = board.step(state, d)
    if want == state:
        tags.add("a total no-op")
    after = set(want[3])
    if not (ground[want[0]] or want[0] in after):
        tags.add("the player on the Void")
    if any(not ground[c] and c not in after for c in want[1]):
        tags.add("a crate on the Void")
    return tags


def selfcheck(walks: int = 10, steps: int = 60, samples: int = 120,
              verbose: bool = True) -> int:
    """Drive presses through BOTH the interpreter and `_Board.step`, comparing
    every piece, the platform, the terrain and the win flag after every one.
    This is the guard that lets the planner trust the native model.

    Five things are checked past "the pieces agree", each an assumption the
    model is built on:

      * Ground2 stays exactly on the Magic cells -- the model carries one set
        for both, and a press that separated them would make the walkable map
        wrong rather than merely mislabelled;
      * Ground, Void and Target never move;
      * Beam still covers every cell, which is what `_seat` relies on;
      * the interpreter never raises a restart (the game has no `restart` rule,
        and the ps:impasse trap is that the ADAPTER, not `eng.step`, would
        execute one);
      * every state reached is inside the enumerated space, i.e. the closure
        `_Board.enumerate` claims, so the field can answer for anything an
        exploration prefix or an epsilon detour lands on.

    Two phases, because they fail differently:

      1. **Random walks from a COLD level start**, no warm-up press, because
         turn-one bookkeeping is precisely the class of mechanic a warm-up
         hides (the ps:idols_to_the_burnt_god lesson). This is the only phase
         that drives the interpreter forward the way a player does.
      2. **Sampled states, seated onto the engine.** A random walk on these
         boards spends nearly all its presses walking -- it shoves the special
         crate in about 2% of them -- so a uniform sample of the level's whole
         enumerated space is taken and every one of the five presses is driven
         from it. That is what reaches the states a plan never visits. The
         sample is then TOPPED UP with witnesses for each mechanic the level's
         space can reach, because the rarest of them (freezing the whole crate
         layer: 38 of one level's 217k pairs) is precisely what a uniform draw
         misses and precisely what is worth checking.

    ``action`` is in the press alphabet even though the search excludes it: the
    claim that it is a pure no-op is exactly the kind of thing that should be
    measured against the interpreter rather than read off a rules section,
    however short (ps:ouroboros's ACTION is a shrink button and its rules
    section does not mention ``action`` either).

    The report ends with a coverage table of every mechanic `_classify` can
    name, in two columns. The first is a cheap EXHAUSTIVE classification of
    every ``(state, press)`` pair in every level's enumerated space, run on the
    model alone -- so "the fuzz never wedged a rider" and "no shipped level can
    wedge a rider" are told apart instead of both showing as a zero. The second
    is what the two phases above actually drove through the interpreter. A
    mechanic the space reaches and the fuzz missed is flagged; a mechanic the
    space cannot reach at all is `_moves`' job, and it builds a board for each
    one on purpose.

    Returns the number of mismatches.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    expert = SilverLungsExpert(game)
    ids = expert.ids
    alphabet = _DIRS + ("action",)
    counts = {name: 0 for name in _MECHANICS}
    space = {name: 0 for name in _MECHANICS}
    total = presses = 0

    def compare(board, state, d) -> str:
        """Press ``d`` on the interpreter (already at ``state``) and return the
        first disagreement with the model, or "" when there is none."""
        want = state if d == "action" else board.step(state, d)
        eng._rule_restart = False
        eng.step(d)
        if eng._rule_restart:
            return "the interpreter raised a restart"
        if expert.read(eng) != want:
            return f"pieces disagree: model {want}, engine {expert.read(eng)}"
        if board.won(want) != eng.check_win():
            return "the win flag disagrees"
        w = eng.width
        seen = {n: {r * w + c for r, row in enumerate(eng.grid)
                    for c, cell in enumerate(row) if ids[n] in cell}
                for n in ("ground2", "ground", "void", "target")}
        if seen["ground2"] != set(want[3]):
            return "Ground2 has come off the Magic"
        if seen["ground"] != {c for c, g in enumerate(board.ground) if g} \
                or seen["ground"] | seen["void"] != set(range(len(board.ground))):
            return "the terrain moved"
        if seen["target"] != set(board.targets):
            return "a Target moved"
        if sum(1 for row in eng.grid for cell in row
               if ids["beam"] in cell) != eng.height * eng.width:
            return "the Beam no longer covers the board"
        if want not in board.index:
            return "the state is outside the enumerated space"
        return ""

    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        rng = random.Random(f"silver_lungs:selfcheck:{level}")
        bad = 0

        # -- what the level's whole space can even do (model only) ------------
        # The witnesses are what make the sample REPRESENTATIVE rather than
        # merely large: a mechanic 38 pairs of the level's 217k can reach is
        # one a uniform draw of a hundred states will simply miss, and it is
        # exactly the sort of state worth putting through the interpreter.
        witness: dict = {}
        for state in board.states:
            if board.won(state):
                continue                         # terminal: never pressed from
            for d in _DIRS:
                for tag in _classify(board, state, d):
                    space[tag] += 1
                    seen = witness.setdefault(tag, [])
                    if len(seen) < 4 and state not in seen:
                        seen.append(state)

        # -- phase 1: random walks from a cold start --------------------------
        for _ in range(walks):
            game.set_level(level)
            state = expert.read(eng)
            for _ in range(steps):
                d = rng.choice(alphabet)
                if d != "action":
                    for tag in _classify(board, state, d):
                        counts[tag] += 1
                note = compare(board, state, d)
                presses += 1
                if note:
                    bad += 1
                    if verbose:
                        print(f"    L{level} walk press={d} from {state}: "
                              f"{note}")
                    break
                state = state if d == "action" else board.step(state, d)
                if eng.check_win():
                    break

        # -- phase 2: sampled states, seated -----------------------------------
        pool = [s for s in board.states if not board.won(s)]
        rng.shuffle(pool)
        pool = pool[:samples]
        for states in witness.values():          # top up the rare mechanics
            pool += [s for s in states if s not in pool]
        for state in pool:
            for d in alphabet:
                if d != "action":
                    for tag in _classify(board, state, d):
                        counts[tag] += 1
                _seat(eng, ids, board, state)
                note = compare(board, state, d)
                presses += 1
                if note:
                    bad += 1
                    if verbose:
                        print(f"    L{level} seat press={d} from {state}: "
                              f"{note}")

        total += bad
        if verbose:
            print(f"  L{level}: {'OK' if not bad else f'{bad} MISMATCHES'}")
    if verbose:
        print(f"  {presses} presses driven through both")
        print(f"    {'mechanic':30s} {'in the space':>12s} {'fuzzed':>8s}")
        for name in _MECHANICS:
            note = ""
            if not space[name]:
                note = "   no level can reach it -- see --moves"
            elif not counts[name]:
                note = "   <-- REACHABLE BUT NEVER FUZZED"
            print(f"    {name:30s} {space[name]:12d} {counts[name]:8d}{note}")
    return total


def _moves() -> int:
    """Re-drive the measured mechanics scenarios of `_SCENES` and assert the
    model and the interpreter agree on every one.

    The boards are synthetic because the shipped levels do not contain all of
    these situations -- see `_SCENES`. They are otherwise ordinary boards of the
    game's own objects, seated the way a level start leaves them.
    """
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng = game._engine
    expert = SilverLungsExpert(game)
    ids = expert.ids
    bad = 0
    covered: set = set()
    for name, rows, press, note in _SCENES:
        board, state = _scene(rows)
        want = board.step(state, press)
        tags = _classify(board, state, press)
        covered |= tags
        _seat(eng, ids, board, state)
        eng._rule_restart = False
        eng.step(press)
        got = expert.read(eng)
        agree = got == want and not eng._rule_restart
        bad += not agree
        print(f"  {name:36s} {press:5s}  {'OK' if agree else 'MISMATCH'}")
        print(f"      {note}")
        print(f"      exercises: {', '.join(sorted(tags))}")
        if not agree:
            print(f"      model {want}\n      engine {got}")
    missing = [m for m in _MECHANICS if m not in covered]
    if missing:
        bad += len(missing)
        print(f"  NOT COVERED by any scenario: {', '.join(missing)}")
    else:
        print(f"  all {len(_MECHANICS)} mechanics of _MECHANICS are covered")
    print("moves clean" if not bad else f"MOVES FAILED: {bad} problems")
    return 0 if not bad else 1


def _verify() -> int:
    """Double-entry check of the distance field and of every tie set it ships.

    `_Board.enumerate` answers everything from ONE pass -- a forward sweep that
    builds the successor table and a reverse sweep over the predecessor map it
    inverts to. A bug in that inversion would produce a self-consistent field, a
    plan that still wins, and tie sets that are quietly wrong; and the training
    labels ARE those tie sets, so "it wins" is not enough of a check. Three
    passes, which fail differently:

      * **Bellman fixpoint.** Re-derive every distance by relaxing
        ``d(s) = 1 + min d(succ)`` to a fixpoint over the successor table alone
        -- no predecessor map, no BFS ordering, nothing shared with the reverse
        sweep -- and require it to agree everywhere, plus ``d == 0`` exactly at
        the winning states. Given a transition function that matches the
        interpreter (which `selfcheck` fuzzes) this is a proof of minimality,
        because the enumerated set is closed under every press.
      * **State well-formedness.** No state may hold two platform tiles or two
        crates on one cell, or a piece on a cell with no Grounds under it other
        than the player (which can be marooned, and legally). A merge would be
        the model quietly losing an object, and it would still enumerate.
      * **Executable labels.** Walk each level's plan on the INTERPRETER and, at
        every step, actually press each direction the label calls optimal: the
        board the engine lands on must be the native successor, and its field
        distance must be exactly one less. That takes the labels out of the
        model and puts them through the real thing.
    """
    _solver, game, expert = _levels()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)

        # -- pass 1: Bellman fixpoint over the successor table ----------------
        check = [0 if board.won(s) else -1 for s in board.states]
        changed = True
        while changed:
            changed = False
            for j, row in enumerate(board.succ):
                if row is None:
                    continue
                best = min((check[n] for n in row if check[n] >= 0), default=-1)
                if best >= 0 and (check[j] < 0 or best + 1 < check[j]):
                    check[j] = best + 1
                    changed = True
        if check != board.dist:
            bad += 1
            print(f"  L{level}: the field disagrees with its Bellman fixpoint "
                  f"at {sum(1 for a, b in zip(check, board.dist) if a != b)} "
                  f"states")
        if any((d == 0) != board.won(s)
               for d, s in zip(board.dist, board.states)):
            bad += 1
            print(f"  L{level}: distance 0 is not exactly the winning states")

        # -- pass 2: no state has quietly lost a piece ------------------------
        malformed = 0
        for p, crates, sp, magic in board.states:
            if len(set(magic)) != len(magic) or len(set(crates)) != len(crates):
                malformed += 1
            elif sp in crates or p in crates or p == sp:
                malformed += 1
            elif not all(board.ground[c] or c in magic for c in crates) \
                    or not (board.ground[sp] or sp in magic):
                malformed += 1
        if malformed:
            bad += 1
            print(f"  L{level}: {malformed} malformed states")

        # -- pass 3: every shipped label, pressed on the interpreter ----------
        plan = board.plan(board.start)
        state = board.start
        labels = 0
        for i, press in enumerate(plan):
            here = board.distance(state)
            best = board.optimal(state)
            if press not in best:
                bad += 1
                print(f"  L{level}: step {i} is not in its own optimal set")
            before = snapshot(eng)
            for alt in best:
                eng.step(alt)
                landed = expert.read(eng)
                if landed != board.step(state, alt) \
                        or board.distance(landed) != here - 1:
                    bad += 1
                    print(f"  L{level}: step {i} labels {alt} optimal, but the "
                          f"interpreter does not land one press closer")
                labels += 1
                restore(eng, before)
            eng.step(press)
            state = board.step(state, press)
        if not eng.check_win():
            bad += 1
            print(f"  L{level}: the interpreter does not report a win")
        print(f"  L{level}: {len(board.states):6d} states relaxed, "
              f"{malformed} malformed, {len(plan):3d} plan steps, "
              f"{labels} labels pressed on the interpreter")
    print(f"verify: {bad} problems")
    return 1 if bad else 0


# ---------------------------------------------------------------------------
# Symmetry: is the 8-element presentation group exact?
# ---------------------------------------------------------------------------

def _transform(k: int, mirror: bool):
    """``(cell_fn, dims_fn, direction_map)`` for one element of the 8-group:
    mirror left-right first, then turn clockwise ``k`` times.

    The direction map is DERIVED from the same linear part that moves the cells,
    so the two cannot drift apart."""
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


def _replay_transformed(eng, expert, layout, presses) -> list:
    """Replay ``presses`` on all seven non-identity turned/mirrored copies of
    ``layout`` and return a note per presentation that diverged.

    Every piece is compared, the platform included: a rotation that slid the
    tiles somewhere else while leaving the player where the transform says would
    be exactly the chirality this is looking for.

    The transformed board is built from the LEVEL LAYOUT and reloaded, so the
    interpreter parses it and re-runs the level-start rules (which is what
    breeds the Beam) itself, rather than being handed a grid this file
    transformed after the fact."""
    hw = (len(layout), len(layout[0]))

    def relabel(state, cell, tw):
        p, crates, sp, magic = state
        m = lambda x: (lambda rc: rc[0] * tw + rc[1])(   # noqa: E731
            cell(divmod(x, hw[1]), hw))
        return (m(p), tuple(sorted(m(x) for x in crates)), m(sp),
                tuple(sorted(m(x) for x in magic)))

    eng.load_level(layout)
    ref = [expert.read(eng)]
    for d in presses:
        eng.step(d)
        ref.append(expert.read(eng))

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
            if expert.read(eng) != relabel(ref[i + 1], cell, tw):
                notes.append(f"rot{k}{'m' if mirror else ''}@press{i}")
                break
    return notes


def _symmetry(walk: int = 300) -> int:
    """Prove ON THE INTERPRETER that the 8-element presentation group is an
    exact symmetry of the mechanic, which is what would entitle this game to the
    `PuzzleScriptAdapter._FLIP_GAMES` augmentation on top of the mandatory
    rotation.

    The argument is strong -- every rule states its force with the relative
    ``>``, there is no gravity, the win condition (``All Target on Crate``)
    names no direction, and no sprite encodes a facing -- but this game also has
    four per-direction copies of the beam rules, which is the shape that hid the
    chirality in ps:gobble_rush, so it is measured rather than argued. Two runs
    per level, because a shortest plan never presses into the Void and never
    wedges a rider, and those are the two things the interpreter has to decide
    here:

      * each level's own PLAN, the sequence the corpus actually records;
      * a seeded RANDOM WALK per level, which does little else.
    """
    _solver, game, expert = _levels()
    eng, g = game._engine, game._game
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        layout = g.levels[level]
        rng = random.Random(f"silver_lungs:symmetry:{level}")
        runs = {"plan": board.plan(board.start),
                "walk": [rng.choice(_DIRS) for _ in range(walk)]}
        notes = []
        for kind, presses in runs.items():
            notes += [f"{kind}:{n}" for n in
                      _replay_transformed(eng, expert, layout, presses)]
        bad += len(notes)
        print(f"  L{level}: ({len(runs['plan'])} plan + {walk} random) presses "
              f"x 7 presentations: "
              f"{'SYMMETRIC' if not notes else 'CHIRAL ' + ', '.join(notes[:3])}")
    print("symmetry clean -- the flip augmentation is exact" if not bad
          else f"SYMMETRY FAILED: {bad} chiral presentations")
    return 0 if not bad else 1


# ---------------------------------------------------------------------------
# Render audit
# ---------------------------------------------------------------------------

def _reachable_compositions(board: _Board) -> set:
    """Every cell composition (a sorted tuple of object names) that appears in
    ANY reachable state of ``board``, WINNING STATES INCLUDED -- the win frame
    is taped like any other and is the one the corpus exists to teach.

    Derived from the enumeration rather than hand-listed: a hand-list would go
    stale the moment a level or a sprite changed, and it would be the thing
    deciding what the audit is allowed to see. Only the cells that can differ
    between states are re-read per state (the pieces and the platform); every
    other cell shows its static composition, which is collected once.
    """
    board.enumerate()
    static = {}
    for cell in range(board.h * board.w):
        parts = ["ground"] if board.ground[cell] else ["void"]
        if cell in board.targets:
            parts.append("target")
        static[cell] = tuple(parts)
    comps = set(static.values())
    for p, crates, sp, magic in board.states:
        occ = {p: "player", sp: "cratespecial"}
        for x in crates:
            occ[x] = "crate"
        mset = set(magic)
        for cell in set(occ) | mset:
            extra = ["magic", "ground2"] if cell in mset else []
            if cell in occ:
                extra.append(occ[cell])
            comps.add(static[cell] + tuple(extra))
    return comps


def _audit() -> int:
    """Assert every REACHABLE cell composition renders distinctly, at every cell
    size this game's boards produce.

    Whole frames of uniform boards are compared, not cell crops: `_render_frame`
    upscales the board to fill 64x64 and letterboxes it, so the output's cell
    grid is not ``cell_px``-aligned and an arithmetic crop reads the wrong
    window (the ps:explod lesson). Two uniform boards render identically iff
    their cells do.

    Both orientations of every board size are audited, because the mandatory
    rotation presents a 10x13 level as 13x10 and the cell size is
    ``min(64 // h, 64 // w)`` -- a square-ish board and its turn can land on
    different sizes, and the sprites have to survive both samplings.

    The pair this exists for is ``crate+target`` against ``crate``: the win
    condition is ``All Target on Crate``, so a Crate that hides the Target it is
    standing on makes the winning board indistinguishable from any other. That
    is exactly what shipped -- see the header of
    data/puzzlescript_games/silver_lungs.txt for that and the four other
    collisions this measured before the sprites were redrawn.
    """
    solver = SilverLungsSolver()
    game = solver.make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    expert = SilverLungsExpert(game)

    comps: set = set()
    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        board = expert.board(eng, level)
        comps |= _reachable_compositions(board)
        sizes.setdefault((eng.height, eng.width), []).append(level)
        sizes.setdefault((eng.width, eng.height), []).append(level)

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for comp in sorted(comps):
            eng.height, eng.width = h, w
            eng.grid = [[{idx["background"], idx["beam"]}
                         | {idx[o] for o in comp} for _ in range(w)]
                        for _ in range(h)]
            eng._position_index_dirty = True
            shots[comp] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(sorted(comps), 2)
                   if np.array_equal(shots[a], shots[b])]
        bad += len(clashes)
        print(f"  {h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in sorted(set(levels)))}): "
              f"{len(shots)} reachable compositions -- "
              f"{'OK' if not clashes else 'CLASHES'}")
        for a, b in clashes:
            print(f"      IDENTICAL: {'+'.join(a)} == {'+'.join(b)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        n = selfcheck()
        print(f"selfcheck: {n} mismatches")
        sys.exit(1 if n else 0)
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--moves" in sys.argv:
        sys.exit(_moves())
    if "--verify" in sys.argv:
        sys.exit(_verify())
    if "--symmetry" in sys.argv:
        sys.exit(_symmetry())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(SilverLungsSolver.main())
