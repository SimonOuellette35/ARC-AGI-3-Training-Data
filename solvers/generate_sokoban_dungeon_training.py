"""Generate Phase-1 training data for the PuzzleScript game ps:sokoban_dungeon
("Sokoban Dungeon", Joseph King).

The harness -- the rotation contract, the trajectory recorder and the
`BaseSolver` plumbing -- lives in `solvers/common/ps_astar.py`, shared with the
other ps: generators, as does the exhaustive-enumeration oracle (`StateGraph` /
`PSEnumExpert`) this game plans with. This file is the game-specific part: the
measured mechanics, the packed state key that makes the enumeration fit in
memory, the proof that the one thing the key drops can never matter, the reports
that prove every level is winnable and shortest, and the render audit.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_sokoban_dungeon",
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
exactly. Every expert step carries the full set of equally-optimal presses (see
"Optimal-action sets" below).

The game
--------
A three-level sokoban with a locked door and a monster. There is exactly ONE
crate and ONE blue target on every level: putting the crate on the target
unlocks the exit, and the win is standing on the unlocked exit. In between are
skeletons, which are not a decoration -- two of the three levels are about them.

Everything below was MEASURED against the interpreter (``--mechanics`` is the
executable form of this section), not read off the .txt:

* **A skeleton CHARGES. It does not step.** The rule is
  ``[Skeleton | ... | Player] -> [> Skeleton | ... | Player] again``. The
  ``again`` re-runs the whole turn, the rule matches again from the skeleton's
  new cell, and the loop only stops when the alignment breaks -- so from the
  far end of an open row a skeleton crosses the entire row in ONE press and
  ``[> Skeleton | Player] -> [Burst | Skeleton]`` deletes the player. Sharing a
  row or a column with a skeleton at the START of a turn is therefore lethal
  unless YOUR press is the thing that breaks the alignment: the first
  again-iteration is the only one in which you move at all.
* **Line of sight ignores walls and crates.** ``...`` is an unconstrained gap,
  so a skeleton four cells away behind a wall still sees you and still starts
  moving. What stops it is not sight but the ordinary movement refusal -- a
  skeleton whose next cell holds a wall, a crate or another skeleton simply
  does not move, and the again-loop terminates because nothing changed.
* **A pushed crate kills a skeleton**: ``[> Crate | Skeleton] -> [Skeledust |
  Crate]``. The crate ends up on the skeleton's cell and a Skeledust decal is
  left behind on the crate's old cell. This is the only offensive move in the
  game, and level 3 is built on it -- the skeleton is standing ON the exit.
* **A skeleton does NOT push the crate.** Only ``[> Player | Crate]`` pushes;
  a charging skeleton that meets the crate is just blocked by it.
* **Death is a RESTART, not a game over.** ``late [Burst] -> restart`` and the
  adapter reloads the level in place (`PuzzleScriptAdapter.perform_action`),
  so a lethal press costs the whole level and returns the board to its start.
  `StateGraph` drops those edges: a restart can never be on a shortest path.
* **The lock is a pair of rules and the unlock is `late`.**
  ``[Exit] [Crate no Target] -> [Exitlocked] ...`` locks the door whenever the
  crate is off the target (so the exit is already locked after the first press
  of every level), and ``late [Exitlocked] [Crate Target] -> [Exit] ...``
  unlocks it AFTER movement resolves -- which is what lets the press that
  shoves the crate onto the target and lands you on the door win outright,
  as level 3's fifth and last press does.
* **You cannot stand on a locked door**: ``[Player Exitlocked | no Obstacle] ->
  [Exitlocked | Player]`` shoves you back off it on the following turn. The
  frame in between is a real observable state, and it renders differently from
  the winning ``player on exit`` (``--audit``).
* ACTION5 exists in the adapter's action list but no rule reads it -- an action
  turn sets no directional force, so nothing can match and the board is
  unchanged. `directions` says so and the enumeration never branches on it.
  (The exploration prefix still presses it; a no-op press is honest recovery
  data.)
* ``check_win()`` is False on all three level starts, so no ``vacuous_start_win``.

Why an exhaustive oracle, and not the shared A*
-----------------------------------------------
Three levels of 39, 19693 and 5384 reachable states, enumerated with the
interpreter as the authority, in about a hundred seconds altogether. A backward
BFS from the winning edges then gives the exact distance-to-win of every state,
so this generator does not search at all. That buys three things a heuristic
search would not:

* plans that are provably SHORTEST (no weight, no heuristic, no node cap that
  could quietly bite): 13, 40 and 5 presses;
* exact optimal-action SETS for free -- ``dist(succ) == dist - 1`` -- rather
  than inferred ones. Every expert step is labelled;
* a PROOF of which states can still win. It is the interesting half of the
  report here: level 2 can reach 19693 states of which only 7959 can still
  win, and level 3 5384 of which only 1304 -- three fifths of one board and
  three quarters of the other is a dead end you cannot see, because shoving the
  one crate somewhere useless is silent and permanent. That is what the RESET
  recovery arc is for.

The packed key, and the one thing it drops
------------------------------------------
An enumeration keyed the shared way (`PSExpert._key`, a frozenset of
``(row, col, obj)``) has to keep a `snapshot` per state so it can be re-seated,
and a snapshot of a 6x12 grid is a fresh `set` per cell -- ~64 KB. Level 2 then
costs 1.3 GB, which is not a size to hand to `parallelize_generator`. So this
expert overrides the `PSEnumExpert.enum_key` / `enum_decode` pair added for it:
the key is the board packed to two bytes per cell (a bitmask of the object ids
present), the enumeration keeps no snapshots at all and re-seats a state by
DECODING its key, and the same searches run in ~100 MB. `_key` itself is
untouched -- it is what the plan memo and the ``plan_cache_path`` signature are
built from, and the disk layer reads it as ``(r, c, obj)`` triples.

The packed key drops exactly one object: **Skeledust**, the decal a crate leaves
where it killed a skeleton. Dropping it is what keeps level 2 enumerable at all
(the decal can be stamped on many different cells, and each one splits every
state that follows), and it is sound for a reason that is checked rather than
asserted -- ``--decode`` does it two ways:

* Skeledust appears on the RIGHT-hand side of exactly one rule and on no LHS,
  and in no win condition. Nothing in the game can ever READ it, so two boards
  differing only in their decals have identical futures. That is a mechanical
  scan of the parsed ruleset, not a reading of the .txt.
* On the two levels whose FULL space is enumerable (0 and 2), the report builds
  the lossless graph as well and checks that dropping the decal is an exact
  graph homomorphism: same states after projection, same edges, same win flags,
  same distances. On level 1, whose lossless space does not terminate in
  practical time, it fuzzes instead -- a random walk on the real interpreter,
  comparing every successor of a real (decal-carrying) board against the
  successors of its decoded twin, and reporting how many of the boards it
  visited actually carried a decal, so the check cannot pass vacuously.

Rendering
---------
Three sprites are redrawn in ``data/puzzlescript_games/Sokoban_Dungeon.txt``
(comment at the top of that file; ``--audit`` is the check). Target and Exit
both live UNDER the collision layer that Player, Skeleton and Crate share, and
both were being erased: ``player on target`` rendered as a bare player, and
``crate on exit`` as a bare crate. Neither could be fixed by moving a sprite
into "the pixels the occluder leaves transparent", because Player is
transparent only on its border and Crate only in its interior -- the two holes
are disjoint -- so each sprite now claims pixels in both regions. Exitlocked
gains its two top corners for a smaller reason: it and Exit differed by exactly
one pixel under the Player, i.e. the winning board and the "the door shoves you
back off" board were one pixel apart at cell_px=5.

``--audit`` compares whole 64x64 frames of uniform boards rather than cropping
one cell out of a mixed one (`_render_frame` centre-pads a non-square board, so
indexing a cell by ``cell_px * r`` reads the wrong pixels on the 6x12 levels),
and it scores only the compositions the enumeration proves REACHABLE, listing
the rest separately. Three are added to that set by hand because the
enumeration structurally cannot see them, and one of the three is the important
one: ``player on exit`` is the WIN, and `StateGraph` records a winning press as
an EDGE without ever storing the board it lands on -- so the one frame the agent
has to recognise is precisely the one a measured-reachability audit omits. The
only indistinguishable pairs left in this game are ``<anything> + Wall`` --
Wall's 5x5 is fully opaque -- and no level places anything under a wall.

Presentation
------------
Rotation augmentation is the adapter's and is mandatory, so it needs nothing
here beyond emitting the SCREEN press -- which `record_level` does through
`screen_action`, with ``remap_actions`` left at its default True (this game is
not in ``PuzzleScriptAdapter._NO_ACTION_REMAP_GAMES``).

This game is deliberately NOT added to ``_FLIP_GAMES``, and that is a decision
rather than an oversight. Most of the usual argument holds -- no gravity,
screen-relative input, a win condition naming no direction, no sprite encoding a
facing -- but the skeleton charge does not obviously survive a mirror: level 2
has two skeletons, the charge rule is expanded over the four directions in a
fixed order, and which of two skeletons wins a contested cell is therefore a
fact about the screen rather than about the board (the ps:gobble_rush problem).
Settling it needs the whole reachable space measured at all sixteen
presentations, which this generator does not do. Adding the flips is worth 4x
the presentations if someone wants to pay for that measurement first.

Level indices in the reports are 0-BASED, i.e. one less than the number in the
game's own "Level N" messages; the prose above uses the game's numbering.

Optimal-action sets
-------------------
``optsets[i]`` is every press ``d`` with ``dist(succ(state_i, d)) == dist(
state_i) - 1``, read straight off the exhaustive distance field, so it is exact
rather than inferred: a set is the complete list of shortest continuations, and
a press missing from it provably costs at least one more. ``--ties`` re-derives
every label from the REAL interpreter -- it replays the plan on the adapter and,
at each step, steps every candidate press from the live (decal-carrying) board
and looks its successor's distance up -- so the labels are checked against the
game rather than against the graph they came from, and the replay doubles as the
proof that the plan wins.

Recovery
--------
``recovery_mode = "reset"`` from `PSAStarSolver`: an epsilon-decayed exploration
prefix flails first and ONE RESET restores the level start, from which the plan
replays to a guaranteed WIN. The RESET is doing real work on this game rather
than decorating it -- two thirds of both big levels is unwinnable (``--bfs``),
one crate is all it takes to get there, and nothing on screen says so.

CLI
---
    --plans      per-level size, plan length and tie coverage
    --bfs        the exhaustive reachable-state report (winnability + dead ends)
    --ties       re-derive every optimal-action label against the interpreter
    --decode     the proof that dropping the Skeledust decal changes nothing
    --mechanics  the measured rules of the game, asserted on crafted boards
    --audit      assert every reachable cell composition renders distinctly
"""

from __future__ import annotations

import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (                       # noqa: E402
    WIN as _WIN, PSAStarSolver, PSEnumExpert, StateGraph as _Graph,
    restore, snapshot,
)

GAME_NAME = "Sokoban_Dungeon"

#: The four presses. ACTION5 exists but no rule reads it (see the docstring),
#: and the order here is the plan's tie-break, so it is fixed.
_DIRS = ("up", "down", "left", "right")

#: The level START plans, cached between processes. The enumerations are the
#: whole cost of generation (~100 s cold, and seed-independent), so without a
#: file on disk every `parallelize_generator` shard would re-derive all three.
PLAN_CACHE = (Path(__file__).resolve().parent.parent
              / "data" / "sokoban_dungeon_plans.json")

#: The one object the enumeration key drops. See the module docstring, and
#: ``--decode`` for the two checks that it cannot matter.
_DROPPED = "skeledust"


# ---------------------------------------------------------------------------
# Expert + solver
# ---------------------------------------------------------------------------

class DungeonExpert(PSEnumExpert):
    """`PSEnumExpert` -- the exhaustive reachable-state oracle -- on four
    presses, with a PACKED enumeration key so the biggest level fits in ~100 MB
    instead of 1.3 GB.

    `enum_key` is the board as two bytes per cell (a bitmask of the object ids
    present, minus Background and minus the Skeledust decal); `enum_decode` is
    its exact inverse up to that decal, which nothing in the game can read.
    `StateGraph.build` therefore keeps no snapshots at all.

    `_key` is left as inherited (the frozenset of every non-background cell):
    it is what the plan memo and the ``plan_cache_path`` start-signature are
    built from, and the disk layer reads it as ``(r, c, obj)`` triples. It is
    also canonical ACROSS levels -- the walls are in it and the three levels
    have different walls -- so no ``scope_by_level`` is needed.
    """

    #: No ACTION button is read by any rule; branching on it would add a
    #: quarter more interpreter steps for edges that are always self-loops.
    directions = list(_DIRS)

    plan_cache_path = PLAN_CACHE

    def setup(self) -> None:
        self.dust_id = self.g.obj_name_to_idx[_DROPPED]
        self._skip = {self.bg_id, self.dust_id}
        #: mask -> the object ids it holds, memoised across the whole
        #: enumeration (a board of tens of thousands of states shows a handful
        #: of distinct cell contents).
        self._unpack: dict[int, frozenset] = {}

    # -- the packed, decodable enumeration key -------------------------------
    def enum_key(self, eng) -> bytes:
        """The whole board as two bytes per cell, Skeledust excluded.

        Two bytes and not one: the object ids run to 9 here, so a byte-wide
        bitmask would silently drop Exit and Exitlocked -- i.e. it would drop
        the win condition, and the enumeration would report every level
        unwinnable rather than crash."""
        skip = self._skip
        out = bytearray(2 * eng.height * eng.width)
        i = 0
        for row in eng.grid:
            for cell in row:
                mask = 0
                for o in cell:
                    if o not in skip:
                        mask |= 1 << o
                out[i] = mask & 0xFF
                out[i + 1] = mask >> 8
                i += 2
        return bytes(out)

    def enum_decode(self, eng, key: bytes) -> None:
        """Seat a packed key back onto the engine. Mirrors `restore`: same fresh
        per-cell `set`s, same two invalidations."""
        bg, unpack = self.bg_id, self._unpack
        width = eng.width
        grid = []
        i = 0
        for _r in range(eng.height):
            row = []
            for _c in range(width):
                mask = key[i] | (key[i + 1] << 8)
                i += 2
                objs = unpack.get(mask)
                if objs is None:
                    objs = unpack[mask] = frozenset(
                        o for o in range(mask.bit_length()) if mask >> o & 1)
                row.append({bg} | objs)
            grid.append(row)
        eng.grid = grid
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()


class DungeonSolver(PSAStarSolver):
    game_id = "puzzlescript_sokoban_dungeon"
    game_name = GAME_NAME
    expert_cls = DungeonExpert

    #: Runaway guard on the enumeration, not a tuning dial: the largest level in
    #: the game reaches 19693 states.
    node_cap = 200_000

    #: Room for the longest plan (40 presses) plus the RESET exploration prefix
    #: (10 +/- 5) and the re-plan after it. Stays well under the adapter's own
    #: 200-step per-level budget, which would otherwise flip a level to
    #: GAME_OVER mid-plan.
    max_steps = 120


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _new(seed: int = 0):
    solver = DungeonSolver()
    game = solver.make_game(seed)
    return solver, game, DungeonExpert(game, node_cap=DungeonSolver.node_cap)


def _graphs(game, expert, levels=None) -> dict:
    """``{level: StateGraph}``, built once and reused by the reports that need
    more than the plan."""
    out = {}
    for level in (range(game.n_levels) if levels is None else levels):
        game.set_level(level)
        out[level] = expert.graph(game._engine)
    return out


def _report() -> int:
    """Per-level board size, plan length and tie coverage."""
    _solver, game, expert = _new()
    eng = game._engine
    total = tied = 0
    for level in range(game.n_levels):
        game.set_level(level)
        t0 = time.time()
        found = expert.plan(eng, level)
        dt = time.time() - t0
        head = f"level {level}: {eng.height:2d}x{eng.width:2d}"
        if found is None:
            print(f"{head} -- UNWINNABLE (see --bfs)")
            continue
        sets = getattr(found, "optsets", []) or []
        ties = sum(1 for s in sets if len(s) > 1)
        total += len(found)
        tied += ties
        room = "ok" if len(found) < game._max_steps else "OVER BUDGET"
        print(f"{head}, {len(found):3d} presses (budget {game._max_steps}, "
              f"{room}), {ties:3d} with a tie set "
              f"({ties / max(1, len(found)):3.0%}), {dt:6.2f}s")
    print(f"total {total} presses, {tied}/{total} steps with a tie "
          f"({tied / max(1, total):.0%})")
    return 0


def _bfs() -> int:
    """The exhaustive reachable-state report -- and therefore the proof that the
    plans are SHORTEST: the enumeration terminates having generated every state
    the interpreter admits, so the backward distance field is exact.

    It is also the DEAD-END report, which is the fact worth knowing about this
    game: about two thirds of each big level's reachable space can no longer
    win. There is one crate, the exit only unlocks with it on the target, and
    the ways to strand it (a corner, a wall it can no longer be walked around,
    a skeleton wedged behind it) are silent -- no death, no restart, no visual
    tell. That is what the RESET recovery arc exists for."""
    _solver, game, expert = _new()
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        t0 = time.time()
        graph = expert.graph(game._engine)
        dt = time.time() - t0
        if graph is None:
            print(f"level {level}: node cap hit -- report is not a proof")
            bad += 1
            continue
        live = len(graph.dist)
        found = graph.plan(expert.directions)
        wins = sum(1 for e in graph.succ.values() if _WIN in e.values())
        stuck = len(graph.succ) - live
        verdict = (f"shortest {len(found)} presses" if found is not None
                   else "UNWINNABLE (no winning edge is reachable)")
        print(f"level {level}: {len(graph.succ):6d} reachable states "
              f"({live} can still win, {stuck} stranded, {wins} touch the "
              f"exit), {graph.steps:6d} engine steps, {dt:6.2f}s -- {verdict}")
    return 0 if not bad else 1


def _ties() -> int:
    """Re-derive every optimal-action label against the INTERPRETER.

    The plan is replayed on the real interpreter, so every state this checks is
    a live board complete with the Skeledust decals the enumeration key drops.
    At each step every candidate press is taken from that board and its
    successor's distance-to-win is looked up: a press is optimal iff it lands
    one closer. That checks three things the graph cannot check about itself --
    that the plan's presses do on the real engine what the graph says they do,
    that the labels are the complete set, and (because the replay runs to the
    end) that the plan actually wins."""
    _solver, game, expert = _new()
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        game.set_level(level)
        graph = expert.graph(eng)
        plan = graph.plan(expert.directions) if graph else None
        if plan is None:
            print(f"level {level}: no plan (skipped)")
            continue
        game.set_level(level)
        checked = won = 0
        for pi, (taken, claimed) in enumerate(zip(plan, plan.optsets)):
            remaining = len(plan) - pi
            here = snapshot(eng)
            measured = []
            for d in expert.directions:
                eng._rule_restart = False
                eng.step(d)
                if eng._rule_restart:
                    cost = None                     # a restart is never on a path
                elif eng.check_win():
                    cost = 1
                else:
                    rest = graph.dist.get(expert.enum_key(eng))
                    cost = None if rest is None else 1 + rest
                restore(eng, here)
                eng._rule_restart = False
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
            checked += 1
            eng._rule_restart = False
            eng.step(taken)
            if eng.check_win():
                won = 1
                break                     # the plan's last press is the win
        if not won:
            print(f"  level {level}: the plan did not reach a win")
            bad += 1
        print(f"level {level}: {checked} steps verified, plan "
              f"{'WINS' if won else 'FAILED'}")
    print("ties clean" if not bad else f"TIES FAILED: {bad} mismatches")
    return 0 if not bad else 1


def _decode() -> int:
    """Proof that the packed enumeration key may drop the Skeledust decal.

    Three parts, in increasing cost:

    1. A mechanical scan of the PARSED ruleset and win conditions: Skeledust
       must appear on no LHS and in no win condition, i.e. nothing in the game
       can read it. (It does appear on one RHS -- that is the rule that makes
       it.) This is the actual argument; the other two are evidence that the
       argument was implemented correctly.
    2. On every level whose LOSSLESS space terminates, build that graph too and
       check the projection is an exact graph homomorphism -- every lossless
       state maps onto an enumerated one, every edge maps onto the same edge,
       and the distances agree. That is complete, not sampled.
    3. Where it does not terminate, fuzz: walk the REAL interpreter and compare
       every successor of the live decal-carrying board against the successors
       of its decoded twin. The walk restarts from every PREFIX of the level's
       own solution rather than only from the level start, because the decal
       exists only after a crate has been shoved into a skeleton and a walk
       that has to find that by chance mostly checks boards where there was
       nothing to disagree about. Reports how many visited boards actually
       carried a decal, so a pass cannot be vacuous.
    """
    _solver, game, expert = _new()
    eng, g = game._engine, game._game
    dust = expert.dust_id
    bad = 0
    packed_graphs = _graphs(game, expert)     # one enumeration per level, reused

    # 1 -- nothing reads it.
    readers = []
    for i, rule in enumerate(g.rules):
        for cell in rule.patterns_lhs:
            for _mod, _name, ids in cell:
                if dust in ids:
                    readers.append(f"rule {i} LHS")
    # A win condition names its objects by NAME, and the name may be an
    # or-group, so both sides are resolved through the object table rather than
    # string-matched -- a scan that only compared strings would pass vacuously.
    for i, wc in enumerate(g.win_conditions):
        named = [n for n in (wc.obj_name, wc.on_obj) if n]
        if any(dust in g.resolve_object_name(n) for n in named):
            readers.append(f"win condition {i} ({wc.quantifier} "
                           f"{' on '.join(named)})")
    if not g.win_conditions:
        readers.append("NO WIN CONDITION AT ALL -- this scan proves nothing")
    makers = [i for i, rule in enumerate(g.rules)
              if any(dust in ids for cell in rule.patterns_rhs
                     for _m, _n, ids in cell)]
    conds = ", ".join(f"{wc.quantifier} {wc.obj_name} on {wc.on_obj}"
                      for wc in g.win_conditions)
    print(f"[1] win conditions: {conds}")
    print(f"[1] {_DROPPED}: written by rules {makers}, read by "
          f"{readers or 'NOTHING'}")
    if readers:
        print("    -- the key may NOT drop it")
        bad += 1

    # 2 -- exact homomorphism where the lossless space terminates.
    def lossless_key(e):
        return frozenset((r, c, o) for r, row in enumerate(e.grid)
                         for c, cell in enumerate(row) for o in cell
                         if o != expert.bg_id)

    def project(k):
        return frozenset((r, c, o) for (r, c, o) in k if o != dust)

    #: A lossless state is a `snapshot`, ~10-64 KB depending on the board, so
    #: this cap is a memory budget (~0.8 GB) rather than a patience one.
    slow_cap = 12_000
    for level in range(game.n_levels):
        packed = packed_graphs[level]
        game.set_level(level)
        t0 = time.time()
        full = _Graph.build(eng, lossless_key, expert.directions, slow_cap)
        if full is None:
            print(f"[2] level {level}: lossless space exceeds {slow_cap} states "
                  f"-- fuzzed instead (part 3)")
            continue
        # Map every lossless state to the packed state it projects onto, by
        # re-deriving the packed key from the projected lossless key.
        def repack(k):
            cells = {}
            for (r, c, o) in project(k):
                cells.setdefault((r, c), 0)
                cells[(r, c)] |= 1 << o
            out = bytearray(2 * eng.height * eng.width)
            for (r, c), m in cells.items():
                i = 2 * (r * eng.width + c)
                out[i] = m & 0xFF
                out[i + 1] = m >> 8
            return bytes(out)

        miss = edge_bad = 0
        for k, edges in full.succ.items():
            pk = repack(k)
            if pk not in packed.succ:
                miss += 1
                continue
            pe = packed.succ[pk]
            for d, nk in edges.items():
                want = _WIN if nk == _WIN else repack(nk)
                if pe.get(d) != want:
                    edge_bad += 1
            for d, nk in pe.items():
                if d not in edges:
                    edge_bad += 1
            if full.dist.get(k) != packed.dist.get(pk):
                edge_bad += 1
        bad += miss + edge_bad
        print(f"[2] level {level}: {len(full.succ)} lossless states -> "
              f"{len(packed.succ)} packed, {time.time() - t0:5.1f}s -- "
              f"{miss} unmapped, {edge_bad} edge/distance disagreements")

    # 3 -- fuzz the levels part 2 could not close.
    #
    # Anchored on the level's own SOLUTION, not only on the level start. A pure
    # random walk from the start reached a decal on 64 of 4000 level-2 boards:
    # the decal only exists after a crate has been shoved into a skeleton, which
    # is a deliberate act several dozen presses in, and a walk that has to find
    # it by chance mostly reports "the decoder agrees" about boards where there
    # was nothing to disagree about. So the walk restarts from every PREFIX of
    # the plan (which does kill both skeletons) and wanders from there.
    rng = random.Random(20260818)
    walk = 25
    for level in range(game.n_levels):
        graph = packed_graphs[level]
        plan = graph.plan(expert.directions) or []
        dusty = visited = mismatch = 0
        for prefix in range(len(plan) + 1):
            game.set_level(level)
            for d in plan[:prefix]:
                eng._rule_restart = False
                eng.step(d)
            for _ in range(walk):
                board = snapshot(eng)
                if any(dust in cell for row in board for cell in row):
                    dusty += 1
                key = expert.enum_key(eng)
                visited += 1
                for d in expert.directions:
                    restore(eng, board)
                    eng._rule_restart = False
                    eng.step(d)
                    real = (None if eng._rule_restart else
                            _WIN if eng.check_win() else expert.enum_key(eng))
                    expert.enum_decode(eng, key)
                    eng._rule_restart = False
                    eng.step(d)
                    twin = (None if eng._rule_restart else
                            _WIN if eng.check_win() else expert.enum_key(eng))
                    if real != twin:
                        mismatch += 1
                # Step on, preferring a press that changes something -- a walk
                # that spends its budget bumping into walls covers nothing.
                restore(eng, board)
                eng._rule_restart = False
                moves = [d for d in expert.directions
                         if graph.succ.get(key, {}).get(d) is not None]
                if not moves:
                    break
                eng.step(rng.choice(moves))
                if eng._rule_restart or eng.check_win():
                    break
        bad += mismatch
        print(f"[3] level {level}: {visited} boards fuzzed from "
              f"{len(plan) + 1} plan prefixes ({dusty} carried a decal), "
              f"{mismatch} successor mismatches")
        if not dusty:
            print("    -- no decal was ever reached; this level's fuzz proves "
                  "nothing about the decal (parts 1 and 2 still apply)")

    print("decode clean" if not bad else f"DECODE FAILED: {bad} problems")
    return 0 if not bad else 1


def _mechanics() -> int:
    """The measured rules of the game, asserted on crafted boards.

    Every claim in the module docstring's "The game" section that a reader could
    otherwise only take on trust, executed. The boards are built by hand rather
    than reached by play so that each one isolates exactly one rule."""
    _solver, game, expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    chars = {".": (), "#": ("wall",), "S": ("skeleton",), "P": ("player",),
             "*": ("crate",), "o": ("target",), "E": ("exit",),
             "L": ("exitlocked",), "@": ("crate", "target"),
             "Z": ("exit", "skeleton"), "d": ("skeledust",)}

    def build(rows):
        eng.height, eng.width = len(rows), len(rows[0])
        eng.grid = [[{idx["background"]} | {idx[o] for o in chars[ch]}
                     for ch in row] for row in rows]
        eng._position_index_dirty = True
        eng._rule_noop_cache.clear()
        eng._rule_restart = False

    def cells(name):
        o = idx[name]
        return sorted((r, c) for r in range(eng.height) for c in range(eng.width)
                      if o in eng.grid[r][c])

    def press(rows, d):
        build(rows)
        eng.step(d)
        return {"player": cells("player"), "skeleton": cells("skeleton"),
                "crate": cells("crate"), "skeledust": cells("skeledust"),
                "exit": cells("exit"), "exitlocked": cells("exitlocked"),
                "restart": bool(eng._rule_restart), "win": bool(eng.check_win())}

    bad = 0

    def check(name, got, **want):
        nonlocal bad
        wrong = {k: (v, got[k]) for k, v in want.items() if got[k] != v}
        print(f"  {'ok  ' if not wrong else 'FAIL'} {name}")
        for k, (w, x) in wrong.items():
            print(f"         {k}: wanted {w}, measured {x}")
        bad += len(wrong)

    print("the skeleton")
    # One press crosses the whole row: the again-loop keeps re-matching.
    check("charges the full length of a line and kills, in ONE press",
          press(["########", "#......#", "#S....P#", "#......#", "########"],
                "right"),
          restart=True)
    # Your own press is the only chance to break the alignment.
    check("is escaped by breaking the alignment; it gains one cell",
          press(["#####", "#...#", "#.S.#", "#...#", "#.P.#", "#####"], "left"),
          skeleton=[(3, 2)], player=[(4, 1)], restart=False)
    # `...` constrains nothing, so a wall between the two is not cover.
    check("sees THROUGH walls and crates (`...` matches anything)",
          press(["########", "#......#", "#S..#.P#", "#......#", "########"],
                "down"),
          skeleton=[(2, 2)], restart=False)
    # What stops it is the ordinary movement refusal, not sight.
    check("is stopped by whatever blocks its next cell, and the again-loop "
          "then terminates",
          press(["########", "#......#", "#S#...P#", "#......#", "########"],
                "down"),
          skeleton=[(2, 1)], restart=False)
    check("does not push the crate",
          press(["#######", "#.....#", "#S.*.P#", "#.....#", "#######"], "down"),
          crate=[(2, 3)], skeleton=[(2, 2)])

    print("the crate")
    check("kills a skeleton when PUSHED into it, leaving a decal behind",
          press(["#######", "#.....#", "#..P*S#", "#.....#", "#######"], "right"),
          crate=[(2, 5)], skeleton=[], skeledust=[(2, 4)], player=[(2, 4)])

    print("the door")
    check("locks the moment the crate is off the target",
          press(["#######", "#.....#", "#Eo*P.#", "#.....#", "#######"], "up"),
          exit=[], exitlocked=[(2, 1)])
    check("unlocks LATE, so the press that seats the crate can also win",
          press(["#######", "#.....#", "#Lo*P.#", "#.....#", "#######"], "left"),
          crate=[(2, 2)], exit=[(2, 1)], exitlocked=[], win=False)
    check("cannot be stood on while locked -- it shoves you back off",
          press(["#######", "#.....#", "#.o*..#", "#..L..#", "#######"], "up"),
          player=[], exitlocked=[(3, 3)])
    check("is the win the moment it is unlocked and you are on it",
          press(["#######", "#.....#", "#.@E..#", "#..P..#", "#######"], "up"),
          player=[(2, 3)], win=True)

    print("the level starts")
    for level in range(game.n_levels):
        game.set_level(level)
        print(f"  {'ok  ' if not eng.check_win() else 'FAIL'} level {level} "
              f"does not read as already won")
        bad += int(bool(eng.check_win()))

    print("mechanics clean" if not bad else f"MECHANICS FAILED: {bad} claims")
    return 0 if not bad else 1


def _audit() -> int:
    """Assert every cell COMPOSITION a settled frame can show is distinct, at
    every cell size the boards use.

    Whole 64x64 frames of uniform boards are compared rather than one cell out
    of a mixed board: `_render_frame` centre-pads a non-square board, so
    indexing a cell by ``cell_px * r`` silently reads the wrong pixels on the
    6x12 levels (the ps:explod lesson). Two uniform boards render identically
    iff their cells do.

    Which compositions matter is not guessed: the enumeration is the authority.
    Every reachable state of every level is scanned for the cell contents it
    holds, those are the ones required to be pairwise distinct, and the rest of
    the 30-composition cross product is listed as unreachable. That is what
    lets the report pass with the ``<anything> + Wall`` pairs still identical --
    Wall's 5x5 is opaque, and no level puts anything under a wall."""
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    _solver, game, expert = _new()
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    inv = {v: k for k, v in idx.items()}
    bg = idx["background"]

    layer2 = ["target", "burst", "skeledust", "exit", "exitlocked"]
    layer3 = ["player", "skeleton", "wall", "crate"]
    comps = {}
    for a in [None] + layer2:
        for b in [None] + layer3:
            objs = tuple(x for x in (a, b) if x)
            comps["+".join(objs) or "floor"] = objs

    # Which compositions the game can actually show, measured off the
    # enumeration. Three have to be added by hand, and each is named rather
    # than assumed because the enumeration structurally cannot see it:
    #   * ``exit+player`` is the WIN. `StateGraph` records a winning press as
    #     an edge and never stores the board it lands on, so the one frame the
    #     agent has to recognise is exactly the one the measured set omits --
    #     which is how ps:slidyyyyyyy shipped a game whose win was invisible.
    #   * Burst never survives the tick it is created in (it restarts the
    #     level), so no enumerated state holds it -- but it IS rendered.
    #   * Skeledust is the object the packed key drops. It is stamped on the
    #     cell a pushed crate just left, which the pushing player immediately
    #     enters, and after that anything on the upper collision layer can
    #     cross it -- everything except Wall, which is never created here.
    reachable = {"exit+player", "burst"}
    reachable |= {"skeledust"} | {f"skeledust+{b}" for b in layer3
                                  if b != "wall"}
    sizes: dict = {}
    for level in range(game.n_levels):
        game.set_level(level)
        sizes.setdefault((eng.height, eng.width), []).append(level)
        graph = expert.graph(eng)
        for key in graph.succ:
            for i in range(0, len(key), 2):
                mask = key[i] | (key[i + 1] << 8)
                objs = tuple(sorted(
                    (inv[o] for o in range(mask.bit_length())
                     if mask >> o & 1 and o != bg),
                    key=lambda n: (n not in layer2, n)))
                reachable.add("+".join(objs) or "floor")

    bad = 0
    for (h, w), levels in sorted(sizes.items()):
        shots = {}
        for name, objs in comps.items():
            eng.height, eng.width = h, w
            eng.grid = [[{bg} | {idx[o] for o in objs} for _ in range(w)]
                        for _ in range(h)]
            eng._position_index_dirty = True
            shots[name] = np.asarray(_render_frame(eng, g)).copy()
        clashes = [(a, b) for a, b in itertools.combinations(shots, 2)
                   if np.array_equal(shots[a], shots[b])]
        fatal = [(a, b) for a, b in clashes
                 if a in reachable and b in reachable]
        bad += len(fatal)
        note = "OK" if not fatal else f"IDENTICAL {fatal}"
        print(f"{h:2d}x{w:2d} (cell {min(64 // h, 64 // w)}px, levels "
              f"{','.join(str(x) for x in levels)}): "
              f"{len(reachable)} reachable of {len(comps)} compositions -- "
              f"{note}")
        if clashes and not fatal:
            print(f"     {len(clashes)} clashing pairs, none reachable: "
                  f"{sorted({a for a, _ in clashes} | {b for _, b in clashes})}")
    print(f"reachable compositions: {sorted(reachable)}")
    print("audit clean" if not bad
          else f"AUDIT FAILED: {bad} indistinguishable reachable pairs")
    return 0 if not bad else 1


if __name__ == "__main__":
    if "--plans" in sys.argv:
        sys.exit(_report())
    if "--bfs" in sys.argv:
        sys.exit(_bfs())
    if "--ties" in sys.argv:
        sys.exit(_ties())
    if "--decode" in sys.argv:
        sys.exit(_decode())
    if "--mechanics" in sys.argv:
        sys.exit(_mechanics())
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(DungeonSolver.main())
