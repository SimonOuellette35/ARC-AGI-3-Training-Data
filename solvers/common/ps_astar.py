"""Shared engine-blackbox A* expert + generator harness for PuzzleScript games.

The ps: generators that plan by *searching the real interpreter* (rather than a
re-implemented model of the mechanic) were all the same program with a different
heuristic bolted on: ``generate_enqueue_training.py`` and
``generate_antimatters_training.py`` were ~95% byte-identical. This module is that
program; each generator is now a ~40-line subclass naming its PuzzleScript game and
its goal heuristic.

WHY IT IS SHARED (the bug that motivated it)
--------------------------------------------
The duplicated ``_screen_action`` helper was edited in one place and copy-pasted
wrong into both: the ACTION5 / directional branches got swapped, so directions were
recorded and driven WITHOUT the inverse rotation remap. The adapter then applied its
own forward remap (`PuzzleScriptAdapter.perform_action`) and executed a different
direction than the expert planned. Nothing crashed -- the expert simply re-planned
from the unexpected state every single step, missing its plan cache every time, so
generation went from seconds per seed to minutes per seed and won 0 seeds. One copy
of `screen_action` means that class of drift can happen only once.

THE ROTATION CONTRACT (do not "simplify" this)
----------------------------------------------
PuzzleScript games are NOT `AugmentedGame`s -- `utils/arc_game.py` never sees them.
The ADAPTER owns the augmentation: it rotates/flips the rendered frame and, for
every game except ``Drop_Maze``, forward-remaps directional input through
`remap_action_full`. So a generator that plans in ENGINE space must emit the SCREEN
press (`inverse_remap_action_full`) both to drive the adapter and to record, which
is the exact opposite of the native games, where `BaseSolver` owns the conversion.

THE EXPERT
----------
A* (optionally weighted) over the real engine dynamics. The state key is the full
set of non-background object cells, which is an exact canonical state for these
games -- they mutate almost everything, including hidden bookkeeping objects, so
anything narrower would alias distinct states. Successors are the settled grids
after ``step(direction)`` for each of the five actions; any returned plan is a
genuine WIN path. Plans are memoized by state key, and because the engine state
after reset is seed-independent for these games (levels are fixed ASCII maps, and
only the PRESENTATION is augmented per seed), the first seed pays for the search
and every later seed replays from cache with its own rotation/recolor.

`PSPushExpert` is the same search for the SOKOBAN-SHAPED games, where the only
move with any effect is a push and the primitive search drowns in re-derived
walks: its successors are ``walk to the push cell, then push`` macros, so the
search depth is the number of pushes. It emits the same flat list of primitive
directions, so everything below this line is unaffected by which one a game uses.

RECOVERY
--------
``recovery_mode = "reset"``: these mechanics are irreversible (an enqueued block, a
solidified antimatter pair), so a perturbed state cannot be re-planned from in
general. The episode-wide epsilon prefix explores freely and ONE RESET restores the
level's initial state, from which the cached plan replays a guaranteed WIN -- the
human "flail, hit reset, then solve" arc. See `BaseSolver._reset_prefix`.
"""

from __future__ import annotations

import heapq
import importlib.util
import json
import os
import random
import re
import sys
from collections import deque
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from arcengine import ActionInput, GameAction, GameState        # noqa: E402
from adapters.puzzlescript_adapter import PuzzleScriptAdapter   # noqa: E402
from solvers.base_solver import BaseSolver, _ID_TO_GAMEACTION   # noqa: E402
from utils.explore import EpsilonSchedule, ExplorationPolicy    # noqa: E402
from utils.rotation import inverse_remap_action_full            # noqa: E402

# The five engine actions these games use: directions plus ACTION (X). The search
# tries each and dedups by resulting state, so no-op actions prune themselves.
DIRECTIONS = ["up", "down", "left", "right", "action"]

#: Engine direction -> the *game-coordinate* GameAction it corresponds to.
#:
#: ``"reset"`` is not a direction and is never searched over (it is absent from
#: `DIRECTIONS`): it is there so an EXPERT can put a level restart in its plan
#: and have `record_level` drive and record it like any other press. The
#: adapter answers `GameAction.RESET` with `PuzzleScriptAdapter.level_reset`,
#: i.e. the game's own R key, and does NOT refund the level's press budget. An
#: honest expert on an IRREVERSIBLE game needs it -- ps:velocity_castle is the
#: case: it explores a castle it can only see one screen of, and a roll into an
#: unseen screen can strand the level, exactly the situation the game's own
#: opening message tells the player to press R for.
DIR_TO_ACTION: dict[str, GameAction] = {
    "up": GameAction.ACTION1,
    "down": GameAction.ACTION2,
    "left": GameAction.ACTION3,
    "right": GameAction.ACTION4,
    "action": GameAction.ACTION5,
    "reset": GameAction.RESET,
}


def screen_action(direction: str, rotation_k: int,
                  hflip: bool = False, vflip: bool = False,
                  remap: bool = True) -> GameAction:
    """Return the *screen* action to press so the adapter -- which forward-remaps
    directional input by its own ``rotation_k`` / ``hflip`` / ``vflip`` -- executes
    ``direction`` on the engine. ACTION5 is non-directional and passes through.

    This is the whole rotation contract for ps: generators; see the module
    docstring. Get it backwards and nothing raises, it just silently drives (and
    records) the wrong direction on 3 of every 4 orientations.

    ``remap=False`` is for the games in `PuzzleScriptAdapter._NO_ACTION_REMAP_GAMES`
    (Drop_Maze), whose rotation is a native in-engine mechanic: the adapter passes
    their input through untouched, so left/right must mean the same thing on screen
    and in the engine.
    """
    game_act = DIR_TO_ACTION[direction]
    if direction in ("action", "reset") or not remap:
        return game_act
    return inverse_remap_action_full(game_act, rotation_k, hflip, vflip)


def _load_game_module(game_id: str):
    """Import ``games/<game_id>/<game_id>.py`` and return the module.

    Loaded by path, not by ``import``: the folder and file are named after the
    client id (``ps:add_man_2_...``), and a colon is not a legal module name.
    This is the same file `game_envs._load_puzzlescript_env` runs for a live
    agent, so a generator using it records the agent's real game."""
    path = (Path(__file__).resolve().parent.parent.parent
            / "games" / game_id / f"{game_id}.py")
    spec = importlib.util.spec_from_file_location(
        f"psgame_{re.sub(r'[^0-9a-zA-Z_]', '_', game_id)}", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"no game module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Engine snapshot / restore (A* needs cheap state save+restore)
# ---------------------------------------------------------------------------

def snapshot(eng) -> list:
    return [[set(cell) for cell in row] for row in eng.grid]


def restore(eng, snap) -> None:
    eng.grid = [[set(cell) for cell in row] for row in snap]
    eng._position_index_dirty = True
    eng._rule_noop_cache.clear()


# ---------------------------------------------------------------------------
# A* expert over the real engine dynamics
# ---------------------------------------------------------------------------

class Plan(list):
    """A press sequence carrying the optimal SET for each of its steps.

    ``optsets[i]`` is every press that is equally shortest at the state the i-th
    press is taken from. It is computed once beside the plan (not lazily during
    recording), so a disk-cached plan replayed in a later process still labels
    every step without re-deriving anything -- which is also why it survives the
    JSON round-trip in `PSExpert.plan`."""

    optsets: list

    def __init__(self, presses, optsets):
        super().__init__(presses)
        self.optsets = optsets


class PSExpert:
    """(Weighted) A* planner over PuzzleScript engine states, memoized by
    canonical state key.

    Subclasses implement `heuristic`; everything else is game-independent. A
    ``weight`` above 1 trades optimality for search speed -- any returned plan is
    still a genuine WIN path, just not guaranteed shortest.
    """

    #: Engine actions the search branches on. Narrow it when a game cancels some
    #: of them outright (Drop_Maze ignores up/down and has no ACTION), which cuts
    #: the branching factor straight out of the search.
    directions: list[str] = DIRECTIONS

    #: Scope the plan memo by level. Needed when `_key` is narrowed to the DYNAMIC
    #: objects only: that key is canonical within one level but not across levels
    #: (walls/targets are static per level yet differ between them), so two levels
    #: sharing a piece layout would otherwise serve each other's plans.
    scope_by_level: bool = False

    #: Return `Plan`s whose per-step optimal SETS were MEASURED by
    #: `optimal_sets` -- every press that still finishes in the moves the
    #: recorded one leaves. Turn it on for any game recorded with
    #: `PSAStarSolver`, whose `optimal_for` picks the sets up automatically:
    #: without them a plan with order-free stretches (a walk to a cell, where
    #: any interleaving of the two axes is equally shortest) trains one
    #: arbitrary interleaving as the single right answer. Costs a re-solve per
    #: candidate press; see `optimal_sets` for the two prunes that keep that
    #: affordable, and note that both of them (and the measurement itself)
    #: require an ADMISSIBLE heuristic and ``weight == 1``.
    exact_optsets: bool = False

    #: Where to keep each level's START plan between processes. Set it (to a
    #: ``data/<game>_plans.json`` path) for a game whose searches are the whole
    #: cost of generation: they are seed-independent, so without a file on disk
    #: every shard `parallelize_generator` starts re-derives all of them. Left
    #: None the cache is purely in-memory, which is right when the searches are
    #: seconds. See `plan` for what is stored and when it is trusted.
    plan_cache_path: "Path | None" = None

    def __init__(self, game: PuzzleScriptAdapter, node_cap: int = 400_000,
                 weight: int = 1):
        self.game = game
        self.g = game._game
        self.bg_id = self.g.obj_name_to_idx.get("background")
        self.node_cap = node_cap
        self.weight = weight
        self.cache: dict[tuple[int | None, frozenset], list | None] = {}
        self._disk: dict = self._load_disk() if self.plan_cache_path else {}
        self.setup()

    def setup(self) -> None:
        """Hook for subclasses to resolve the object ids their heuristic needs
        (``self.g.resolve_object_name(...)``). Called once at construction."""

    def heuristic(self, eng) -> int:
        """Estimated remaining cost from the engine's current grid. Must be 0 at a
        win. Cheap and goal-aware beats tight: these searches are dominated by the
        interpreter step, not by node count."""
        raise NotImplementedError

    def observe(self, eng) -> None:
        """Hook: a frame has just been presented to the agent. Default no-op.

        `record_level` calls it once per FRAME the recording tapes -- the level
        start, every exploration-prefix press, the RESET that ends the prefix and
        every expert press -- so a PARTIALLY OBSERVABLE game's expert can build
        its knowledge from exactly the frames the policy will be trained on, and
        no others. Four-room tilt mazes is the case: `flickscreen 16x16` crops a
        32x32 level to the one room the block is standing in, so the target is
        off-screen at the start of two of its three levels, and an expert that
        planned from the whole engine grid would emit a beeline no
        observation-conditioned learner could reproduce (see the badge_placement
        failure mode). Its `observe` reads the room window only.

        Without this hook such an expert would see only the states its own `plan`
        is called at and would miss everything the exploration prefix uncovered
        -- i.e. it would be trained to re-explore rooms the policy has already
        been shown."""

    def dead(self, eng) -> bool:
        """Hook: True when this state can never reach a win, so the successor is
        dropped instead of queued.

        Default False -- a puzzle whose moves are all recoverable has no such
        state, and its search is unaffected. Games with a LETHAL failure (Boupha's
        Candle Quest: a monster steps on the player and replaces it with a corpse)
        need it: the win condition is then vacuously half-true and the state sits
        in the queue forever with whatever the heuristic guessed. Charging a huge
        heuristic instead would sink those nodes but still pay to snapshot and heap
        every one of them, and death is a large slice of the branching on the
        levels that have it."""
        return False

    def _key(self, eng) -> frozenset:
        # These games mutate nearly every object class -- pieces, targets (removed
        # on match), walls created/destroyed by rules, the player/cursor, and all
        # the hidden queue / aiming / mode bookkeeping objects. Only Background is
        # invariant, so keying on every non-background cell is an exact canonical
        # state; anything narrower would alias distinct states and return plans
        # that do not apply. A game whose static scenery really is static can
        # override this with a cheaper dynamic-objects-only key -- and must then
        # set ``scope_by_level``.
        bg = self.bg_id
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in cell
            if o != bg
        )

    def _astar(self, eng) -> list | None:
        if eng.check_win():
            return []
        w = self.weight
        start = snapshot(eng)
        start_key = self._key(eng)
        counter = 0
        pq = [(w * self.heuristic(eng), 0, counter, start, [])]
        best_g = {start_key: 0}
        nodes = 0
        while pq:
            _f, g, _c, snap, path = heapq.heappop(pq)
            for direction in self.directions:
                restore(eng, snap)
                eng.step(direction)
                nodes += 1
                if eng.check_win():
                    return path + [direction]
                if self.dead(eng):
                    continue
                k = self._key(eng)
                ng = g + 1
                if best_g.get(k, 1 << 30) <= ng:
                    continue
                best_g[k] = ng
                counter += 1
                heapq.heappush(
                    pq,
                    (ng + w * self.heuristic(eng), ng, counter,
                     snapshot(eng), path + [direction]),
                )
                if nodes >= self.node_cap:
                    return None
        return None

    def plan(self, eng, level: int | None = None) -> list | None:
        """Return a WIN action sequence from the engine's current state (or None
        if unsolvable within the node cap). Leaves the engine grid unchanged.

        ``level`` scopes the memo when ``scope_by_level`` is set; it is ignored
        otherwise, so callers can always pass it.

        With ``plan_cache_path`` set this also consults (and fills) the on-disk
        cache, but only for a level's START state: every seed, and every
        process, plans from that one state and from nothing else, because
        recovery is a RESET back to it. A stored entry carries the start layout
        it was solved from and is used only when the board still matches, so an
        edited level is a miss rather than a wrong plan."""
        if self.plan_cache_path is None or level is None:
            return self._plan_memoized(eng, level)
        sig = sorted([r, c, o] for (r, c, o) in self._key(eng))
        entry = self._disk.get(level)
        if entry is not None:
            if entry["start"] != sig:      # not the start state (or a stale file)
                return self._plan_memoized(eng, level)
            if entry["plan"] is None:
                return None
            return (Plan(entry["plan"], entry["optsets"])
                    if entry.get("optsets") is not None else list(entry["plan"]))
        found = self._plan_memoized(eng, level)
        self._disk[level] = {
            "start": sig,
            "plan": None if found is None else list(found),
            "optsets": getattr(found, "optsets", None),
        }
        self._save_disk()
        return found

    def _plan_memoized(self, eng, level: int | None = None) -> list | None:
        """`plan` without the disk layer: the in-memory memo around `_search`."""
        k = (level if self.scope_by_level else None, self._key(eng))
        if k in self.cache:
            return self.cache[k]
        start = snapshot(eng)
        sol = self._search(eng)
        restore(eng, start)
        self.cache[k] = sol
        return sol

    # -- disk plan cache ------------------------------------------------------
    def _load_disk(self) -> dict:
        """``{level: {"start": sig, "plan": [...] | None, "optsets": [...] | None}}``,
        or empty if unreadable -- a cache that cannot be parsed is a miss, never
        a crash."""
        try:
            raw = json.loads(Path(self.plan_cache_path).read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:                                    # noqa: BLE001
            return {}

    def _save_disk(self) -> None:
        """Write atomically -- shards started together would otherwise interleave
        into a truncated file (see `parallelize_generator`)."""
        try:
            path = Path(self.plan_cache_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self._disk[k] for k in sorted(self._disk)}, indent=1))
            os.replace(tmp, path)
        except OSError:
            pass                                             # cache is optional

    def _search(self, eng) -> list | None:
        """The plan-finding strategy. `_astar` by default; subclasses swap the
        whole strategy here (see `PSBeamExpert`) rather than reimplementing the
        memo, the restore discipline and the level scoping in `plan`.

        When a subclass wants per-step optimal SETS (`_wants_optsets`), the
        search is bracketed by a snapshot/restore so `_optsets` runs with the
        engine back AT the state the plan starts from -- `_astar` itself leaves
        the grid dirty. Annotating here rather than lazily during recording is
        what lets a game reuse the recording adapter: `plan` runs before the
        first frame is taped, and the resulting `Plan` is what the disk cache
        stores, so a later process replays labelled steps without re-deriving
        anything."""
        if not self._wants_optsets():
            return self._astar(eng)
        start = snapshot(eng)
        found = self._astar(eng)
        restore(eng, start)
        if found is None:
            return found
        sets = self._optsets(eng, found)
        return found if sets is None else Plan(found, sets)

    def _wants_optsets(self) -> bool:
        """True when `_search` should call `_optsets`. Kept separate from the
        flags themselves so a subclass can add its own way of asking for sets
        (`PSPushExpert.annotate`) without re-copying `_search`."""
        return self.exact_optsets

    def _optsets(self, eng, plan) -> "list[list[str]] | None":
        """The per-step optimal SETS for ``plan``, engine at its start state.
        None means "no sets" -- the plan is returned unlabelled."""
        self._exact_dist = {}          # scoped to this level's search
        return self.optimal_sets(eng, plan)

    # -- optimal-action sets, MEASURED ---------------------------------------
    def optimal_sets(self, eng, plan) -> list[list[str]]:
        """Per-step optimal SETS for ``plan``, measured by re-solving. Engine at
        the plan's start state; left unchanged.

        At each step, re-solve from each successor and keep every press that
        still finishes in the moves remaining. Two prunes keep that affordable,
        and both are sound rather than heuristic -- a press that leaves the
        board untouched cannot be on a shortest path (it wastes one move to
        reach the state it started from), and neither can one whose ADMISSIBLE
        estimate already exceeds what is left. Between them only the genuine
        candidates are ever searched.

        The measurement is as exact as the search it calls: both sides of the
        comparison come from the same `_astar`, so the sets are exactly the
        presses that tie with the plan's own step under it -- which makes it
        exact when `heuristic` is admissible and ``weight == 1``, and a guess
        otherwise.

        `PSPushExpert.annotate_walks` is the cheap, INFERRED answer to the same
        question, and it is the wrong one for a game where a piece-move can
        TIE: it labels a step that moved a piece with itself alone -- sound in
        a sokoban, where which crate you shove where *is* the puzzle, so a
        sibling push is a different plan rather than a reordering of this one.
        In a game whose win is REACHING somewhere, shoving a piece aside is
        frequently just how you walk, and several directions genuinely finish
        in the same number of moves."""
        start = snapshot(eng)
        out = []
        for i, taken in enumerate(plan):
            remaining = len(plan) - i      # moves left once this one is taken
            here = snapshot(eng)
            best = []
            for direction in self.directions:
                eng.step(direction)
                if eng.check_win():
                    cost = 1
                elif eng.grid == here or self.heuristic(eng) > remaining - 1:
                    cost = None            # sound prunes; see above
                else:
                    sub = self._exact(eng)
                    cost = None if sub is None else 1 + sub
                restore(eng, here)
                if cost == remaining:
                    best.append(direction)
            # The recorded step ties with itself by construction; a disagreement
            # would mean the two sides were measured differently, so fall back
            # to labelling what the expert actually did rather than shipping a
            # set that does not contain it.
            out.append(best if taken in best else [taken])
            eng.step(taken)
        restore(eng, start)
        return out

    def _exact(self, eng) -> int | None:
        """Shortest remaining move count from the current grid, memoized.

        Calls `_astar` rather than `plan`, which would recurse straight back
        into `_search` and into this."""
        key = self._key(eng)
        if key in self._exact_dist:
            return self._exact_dist[key]
        snap = snapshot(eng)
        sol = self._astar(eng)
        restore(eng, snap)
        got = None if sol is None else len(sol)
        self._exact_dist[key] = got
        return got


# ---------------------------------------------------------------------------
# A* over walk-to-and-push MACROS (the sokoban-shaped ps: games)
# ---------------------------------------------------------------------------

#: Engine direction -> (dr, dc) on the grid. Only the four moves; ACTION never
#: participates in a push macro.
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}


class PSPushExpert(PSExpert):
    """A* whose successors are ``walk to the push cell, then push`` MACROS.

    WHY (the primitive search does not scale here). In a sokoban-shaped game the
    only state-changing move is a push: every other move just relocates the
    player, with no side effect at all. `PSExpert`'s primitive A* therefore spends
    its whole budget re-deriving the same walks: a 50-move solution is a depth-50
    search over branching 4, and only the handful of steps that were pushes
    changed anything. Branching on macros makes the depth the number of PUSHES,
    and each macro's walk is a shortest path found by one BFS instead of by
    search.

    The plan is still a flat list of PRIMITIVE directions, so everything
    downstream (`record_level`, the plan cache, the epsilon detour) is unchanged.

    OPTIMALITY. ``g`` counts primitive MOVES, not pushes, so the cost metric is
    the one the agent actually pays and (at ``weight == 1``) plans stay shortest
    up to one approximation: dedup keys the player by its REACHABLE REGION rather
    than by its exact cell (the standard sokoban canonicalisation). Two states
    with the same pieces and the same player region admit exactly the same
    futures, so merging them can never lose a solution -- it can only mis-cost one
    by at most the region's diameter. That is what makes the dedup bite: without
    it, every cell the player could have stopped on is its own state and the macro
    search degenerates back to the primitive one.

    Subclasses set ``pushable_names`` / ``blocker_names`` and implement
    `heuristic` (in units of primitive moves). A game with moves that are not
    pushes -- stepping onto the exit, holding a button down, firing a portal --
    adds them through `extra_macros`.
    """

    directions = ["up", "down", "left", "right"]

    #: `_key` is dynamic-objects-only, canonical only WITHIN a level.
    scope_by_level = True

    #: Return `Plan`s carrying `annotate_walks`' per-step optimal SETS instead of
    #: a bare direction list. Opt-in only because it costs one extra replay of
    #: the plan per search; turn it on for any game recorded with
    #: `PSAStarSolver`, whose `optimal_for` picks the sets up automatically --
    #: without them every WALK step trains one arbitrary interleaving of the two
    #: axes as the single right answer.
    annotate: bool = False

    #: ``exact_optsets`` (inherited from `PSExpert`) takes precedence over
    #: ``annotate``: turn it on for a game where shoving a piece aside is just
    #: how the player WALKS, so a push genuinely ties with a sibling push (or
    #: with a plain move) -- `annotate_walks` calls every push forced, which is
    #: right in a sokoban and wrong there.

    #: Test the goal when a winning node is POPPED, not when it is generated.
    #:
    #: The default (False) returns the first winning macro the search reaches,
    #: which is what every generator on this class was written against and is
    #: exactly right when all macros cost the same -- but they do not. A macro
    #: is ``walk + act``, so its cost is the length of its walk, and a node
    #: popped at ``g = 40`` can win with an 11-step macro (51) while a node
    #: popped at ``g = 45`` wins with a 3-step one (48). Generating the first
    #: win therefore does NOT prove it shortest. With this on, a win is kept as
    #: the incumbent and the search runs until the frontier's ``f`` reaches its
    #: cost, which (at ``weight == 1``, with an admissible heuristic) is a proof.
    #:
    #: Opt-in because turning it on changes the plans of every game here that
    #: never noticed -- ESL Puzzle Game was 51 against a primitive BFS's proven
    #: 48 -- and those plans are byte-verified. Cost is the tail of the search
    #: between the first win and the bound; on these boards it is small.
    exact_goal_test: bool = False

    #: The ACTION press reads the player's FACING, so the last walk step before
    #: one is FORCED (it is the press that aimed the player) and the tie
    #: analysis for the rest of that walk run targets the cell that press was
    #: taken FROM. See `annotate_walks`.
    #:
    #: Off by default because in every game here whose ACTION is aimed, the
    #: aiming press is a separate blocked turn the flattened plan already shows
    #: as its own step. PUZZLETALE is the case: it can arrive at the aiming cell
    #: already facing the right way, so the last step of the WALK doubles as the
    #: aim, and offering its siblings as equally optimal would be offering
    #: presses after which the ACTION does nothing at all.
    action_reads_facing: bool = False

    #: A TRAILING walk run (one that ends the plan rather than setting up a
    #: press) is labelled against the cell the plan ENDS on, instead of having
    #: every step of it labelled with the one direction the expert happened to
    #: pick.
    #:
    #: Set it for a game whose win is REACHING somewhere: there the plan's last
    #: macro is a bare walk to the goal, so the default -- which has no press
    #: after the run to read a destination off -- leaves the whole approach
    #: labelled as a single forced route when any shortest route to the same
    #: cell is equally right. Wrong for a game whose trailing walk is
    #: positioning for something the recording does not contain, and wrong for
    #: any game whose walks are not inert, hence opt-in.
    trailing_walk_is_free: bool = False

    #: Object (or or-group) names the player pushes.
    pushable_names: tuple[str, ...] = ()
    #: Object (or or-group) names the player cannot walk onto. Pushables block the
    #: player too -- walking into one is a push, which is a macro, not a walk --
    #: and are added automatically.
    blocker_names: tuple[str, ...] = ("wall",)
    #: Object (or or-group) names that SHADOW the player: pure functions of
    #: where it stands and which way it faces, parked in a neighbouring cell by
    #: a ``late`` rule. PUZZLETALE has two -- the face it draws in the cell
    #: above the player, and the seed it balances on top of that.
    #:
    #: `annotate_walks` classifies a press by asking whether it changed
    #: anything but the player's own cell, and a shadow changes on EVERY press,
    #: so without this every walk step in such a game reads as a push and gets
    #: labelled forced -- the tie sets come out empty and the annotator is
    #: silently doing nothing. Nothing else consults this: a shadow is not a
    #: blocker (it has its own collision layer) and it carries no state the
    #: player's own does not, so the search keys are unaffected.
    shadow_names: tuple[str, ...] = ()

    def setup(self) -> None:
        g = self.g
        self.push_ids = {i for n in self.pushable_names
                         for i in g.resolve_object_name(n)}
        self.blocker_ids = {i for n in self.blocker_names
                            for i in g.resolve_object_name(n)}
        self.shadow_ids = {i for n in self.shadow_names
                           for i in g.resolve_object_name(n)}
        self.player_ids = set(self.game._engine._player_indices)
        self.dyn_ids = self.push_ids | self.player_ids

    def _key(self, eng) -> frozenset:
        # Pieces + player. Walls / targets / background are static per level in
        # this family, so they are redundant in the key -- hence ``scope_by_level``.
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    # -- macro generation ----------------------------------------------------
    def _refine_free(self, eng, free) -> None:
        """Hook: strike cells off (or back onto) the player's walkability map
        that the id sets cannot express. Mutates ``free`` in place; default
        no-op.

        ``blocker_names`` answers "does this object block", which is every game
        in this family but the ones whose terrain is CONDITIONAL: PUZZLETALE's
        water stops the player only while no bridge is lying on it, and the very
        same cell becomes walkable the moment one is -- neither "water blocks"
        nor "water does not" is true of the board.

        Both `_analyze` (the search's macros) and `_walk_distances` (the tie
        annotator's alternatives) build their map here, so the two can never
        disagree about where the player may walk -- which is the whole reason
        this is one hook rather than an override of each."""

    def _analyze(self, eng) -> tuple:
        """Return ``(region, macros)`` for the engine's current grid.

        ``region`` is the canonical id of the player's walkable region (its
        smallest cell), and ``macros`` is every ``walk + push`` the player can
        reach right now, each as a list of primitive directions.

        A piece sharing its cell with a blocker is NOT pushable: the push would
        need the player to step onto that cell, which the blocker forbids. (In
        Add Man that is a real, permanent trap -- a digit shoved into a wall can
        only ever be changed by an arithmetic carry, never pushed again.)"""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        push_ids, blocker_ids, player_ids = (self.push_ids, self.blocker_ids,
                                             self.player_ids)
        players: list[tuple[int, int]] = []
        pieces: list[tuple[int, int]] = []
        free = [[True] * w for _ in range(h)]
        for r in range(h):
            row = grid[r]
            for c in range(w):
                cell = row[c]
                if not cell:
                    continue
                if cell & player_ids:
                    players.append((r, c))
                blocked = bool(cell & blocker_ids)
                if cell & push_ids:
                    free[r][c] = False
                    if not blocked:
                        pieces.append((r, c))
                elif blocked:
                    free[r][c] = False
        self._refine_free(eng, free)
        player = self.live_player(eng, players)
        if player is None:                       # player consumed by a rule
            return None, []

        # Shortest walks from the player over free cells, kept as parent pointers
        # and materialised only for the cells a macro actually uses: this runs on
        # every node, and building a path list per reachable cell would allocate
        # the whole board's worth of them to use four.
        parent: dict[tuple[int, int], tuple | None] = {player: None}
        queue = deque([player])
        while queue:
            cur = queue.popleft()
            r, c = cur
            for d, (dr, dc) in _DELTA.items():
                nxt = (r + dr, c + dc)
                if (0 <= nxt[0] < h and 0 <= nxt[1] < w
                        and free[nxt[0]][nxt[1]] and nxt not in parent):
                    parent[nxt] = (cur, d)
                    queue.append(nxt)

        def walk_to(cell) -> list:
            out = []
            while parent[cell] is not None:
                cell, d = parent[cell]
                out.append(d)
            out.reverse()
            return out

        macros = []
        for (pr, pc) in pieces:
            for d, (dr, dc) in _DELTA.items():
                stand = (pr - dr, pc - dc)
                if stand in parent:
                    macros.append(walk_to(stand) + [d])
        macros.extend(self.extra_macros(eng, parent, walk_to, free))
        return min(parent), macros

    def live_player(self, eng, positions) -> "tuple[int, int] | None":
        """Which of ``positions`` -- every cell holding a player sprite, in scan
        order -- is the one the presses actually drive.

        Default: the last one, which is what `_analyze` read before this hook
        existed and is trivially right for the games here, all of which have
        exactly one player on the board. It is a hook because a game can CLONE
        the player: Savior's save state copies every savable object -- the
        player included -- into an off-screen mirror region, so from the first
        save on there are two player sprites and only one of them moves. An
        expert that planned around the frozen clone would build its walks from
        a cell no press can leave. (The adapter's flickscreen camera makes the
        same choice for the same reason; see `_render_frame`.)"""
        return positions[-1] if positions else None

    def extra_macros(self, eng, parent, walk_to, free) -> list[list[str]]:
        """Hook: macros this game needs BESIDES ``walk to the push cell, push``.

        A pure sokoban needs none -- a push is the only move with an effect, and
        the win is a push outcome. Games where the player itself does something
        (walks onto the exit, holds a button down, fires a portal at a wall) have
        moves the push enumeration cannot express, and without them the macro
        search closes with the goal still unreached. Aperture Science is the
        motivating case; see `solvers/generate_aperture_science_training.py`.

        ``parent`` is the walk BFS tree (``{cell: (prev_cell, direction) | None}``,
        so ``cell in parent`` means "reachable"), ``walk_to(cell)`` materialises
        the shortest walk to a reachable cell, and ``free[r][c]`` is the
        walkability map both were built from. Return a list of macros, each a
        list of primitive engine directions."""
        return []

    def _apply_macro(self, eng, macro) -> int:
        """Run one macro on ``eng`` and return the index of the primitive that
        WON, or -1 if the macro finished without winning.

        The win is checked after EVERY primitive, not just at the macro's end:
        ``check_win`` reads a per-step flag, so a macro that wins partway through
        and keeps stepping silently "un-wins" and the solution is lost. Truncating
        at the winning step also keeps the plan shortest. (A pure sokoban only
        ever wins on a macro's final push, so this is a no-op there.)

        Shared by `_astar` and `PSBeamExpert._search`, which ran byte-identical
        copies of this loop, and it is the ONLY place a macro touches the
        interpreter. Since that step is the entire cost of these searches, it is
        also the hook to override for a game that can reach a macro's end state
        for less -- one whose rules do nothing at all on a bare player move can
        relocate the player instead of stepping the walk. An override must leave
        the engine in exactly the state the primitive replay would have."""
        for i, direction in enumerate(macro):
            eng.step(direction)
            if eng.check_win():
                return i
        return -1

    def _wants_optsets(self) -> bool:
        """`PSExpert`'s measured sets, plus this class's INFERRED ones."""
        return self.exact_optsets or self.annotate

    def _optsets(self, eng, plan) -> "list[list[str]] | None":
        """Measured sets when ``exact_optsets`` is on, else `annotate_walks`'
        inferred ones. See `PSExpert.optimal_sets` for why a sokoban wants the
        cheap inference and a walk-shaped game does not."""
        if self.exact_optsets:
            return super()._optsets(eng, plan)
        return self.annotate_walks(eng, plan)

    # -- the region-key dedup (see the class docstring) is the one place the
    # measured sets of `PSExpert.optimal_sets` are an approximation rather than
    # a proof: two states merged by it can be mis-costed by at most the
    # region's diameter.
    def _region_key(self, eng, region) -> tuple:
        push_ids = self.push_ids
        return (frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & push_ids)
        ), region)

    # -- optimal-action sets --------------------------------------------------
    def annotate_walks(self, eng, plan) -> list[list[str]]:
        """Per-step optimal-direction SETS for a flat macro ``plan``, with the
        engine at the state the plan starts from. Leaves ``eng`` unchanged.

        Most of a macro plan is not pushes, it is the player WALKING to the next
        push cell, and a walk's order is FREE: any interleaving of the two axes
        that stays on a shortest route to the same cell costs the same and
        leaves an identical state, because no rule in a push game fires on a
        bare move onto an empty cell. Labelling one arbitrary interleaving as
        the only right answer trains the policy to a coin flip it cannot win and
        turns every walk into a place to compound error -- so each walk step is
        labelled with every direction that keeps it on a shortest route. Pushes
        are labelled with themselves alone: which piece to shove where is the
        puzzle, and a sibling push is a different plan, not a reordering of this
        one.

        Classification watches the BOARD rather than trusting a macro boundary
        the flattened plan no longer carries: a step that changed anything but
        the player's cell is a push, a step that moved only the player is a
        walk. A maximal run of walks always ends on the stand cell of the push
        that follows it -- that is how `_analyze` builds a macro -- so one BFS
        over the run's (constant, since a walk moves nothing) walkable map gives
        the distance-to-stand field the alternatives are read off.

        Two opt-in flags change which cell a run is measured against, and both
        are no-ops unless a game sets them: ``action_reads_facing`` makes the
        last step before an ACTION forced (it aimed the player, so its siblings
        reach the same cell facing the wrong way), and ``trailing_walk_is_free``
        measures a run that ENDS the plan against the cell the plan ends on.
        """
        start = snapshot(eng)
        player_ids = self.player_ids
        shadows = self.shadow_ids

        def read():
            pos, rest = None, []
            for r, row in enumerate(eng.grid):
                for c, cell in enumerate(row):
                    for o in cell:
                        if o in player_ids:
                            pos = (r, c)
                        elif o not in shadows:
                            rest.append((r, c, o))
            return frozenset(rest), pos

        # One replay, keeping each step's kind, the cell it was taken from, and
        # the board it was taken on (so a walk run's map needs no second replay).
        steps = []
        rest, pos = read()
        for direction in plan:
            snap_before, before, here = snapshot(eng), rest, pos
            eng.step(direction)
            rest, pos = read()
            kind = ("push" if rest != before
                    else "walk" if pos != here else "stuck")
            steps.append((kind, here, direction, snap_before))
        final_pos = pos                    # where the plan LEFT the player

        out: list = [None] * len(plan)
        i = 0
        while i < len(steps):
            if steps[i][0] != "walk":
                out[i] = [steps[i][2]]
                i += 1
                continue
            run = i
            while i < len(steps) and steps[i][0] == "walk":
                i += 1
            end = i                        # one past the last step to label here
            if i < len(steps):
                stand = steps[i][1]
                if (self.action_reads_facing and steps[i][2] == "action"
                        and end > run):
                    # The ACTION about to be pressed reads the player's FACING,
                    # and the run's last step is what set it -- so that step is
                    # forced, and the rest of the run is a walk to the cell it
                    # was taken FROM. Offering its siblings instead would offer
                    # presses that arrive on the same cell aimed elsewhere,
                    # after which the ACTION does nothing at all.
                    end -= 1
                    out[end] = [steps[end][2]]
                    stand = steps[end][1]
            elif self.trailing_walk_is_free:
                stand = final_pos          # the run IS the approach to the goal
            else:
                stand = None
            if stand is None:              # a trailing walk pushes nothing: keep
                for j in range(run, end):  # the recorded choice as the only one
                    out[j] = [steps[j][2]]
                continue
            restore(eng, steps[run][3])
            dist = self._walk_distances(eng, stand)
            for j in range(run, end):
                here = steps[j][1]
                d0 = dist.get(here)
                alts = [] if d0 is None else [
                    d for d, (dr, dc) in _DELTA.items()
                    if dist.get((here[0] + dr, here[1] + dc)) == d0 - 1
                    and self._walk_step_ok(eng, here, d)]
                # The recorded step is on a shortest route by construction, so
                # an empty (or disagreeing) `alts` means the reconstruction has
                # drifted -- fall back to labelling what the expert did.
                out[j] = alts if steps[j][2] in alts else [steps[j][2]]
        restore(eng, start)
        return out

    def _walk_step_ok(self, eng, cell, direction) -> bool:
        """Hook: is pressing ``direction`` at ``cell`` a BARE walk?

        `annotate_walks` offers a direction as an equally-short alternative only
        when this says yes. True for every game whose walk graph is undirected
        (a press either moves the player or does nothing), which is all of them
        but the ones where the player DRAGS something -- ESL Puzzle Game's grey
        crates follow the player, so there the same press can be a walk from one
        cell and a pull from its neighbour, and a tie set that named it would be
        naming a different board."""
        return True

    def _walk_distances(self, eng, target) -> dict:
        """BFS distance to ``target`` over the cells the player may walk on:
        no blocker, no pushable (walking into one is a push, not a walk), and
        whatever else `_refine_free` strikes off -- the same map `_analyze`
        gives the search."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        block = self.blocker_ids | self.push_ids
        free = [[not (grid[r][c] & block) for c in range(w)] for r in range(h)]
        self._refine_free(eng, free)
        dist = {target: 0}
        queue = deque([target])
        while queue:
            r, c = queue.popleft()
            for dr, dc in _DELTA.values():
                nxt = (r + dr, c + dc)
                if (0 <= nxt[0] < h and 0 <= nxt[1] < w
                        and free[nxt[0]][nxt[1]]
                        and nxt not in dist):
                    dist[nxt] = dist[(r, c)] + 1
                    queue.append(nxt)
        return dist

    def _astar(self, eng) -> list | None:
        if eng.check_win():
            return []
        w = self.weight
        start = snapshot(eng)
        region, macros = self._analyze(eng)
        counter = 0
        nodes = 0
        pq = [(w * self.heuristic(eng), 0, counter, start, [], macros)]
        best_g = {self._region_key(eng, region): 0}
        best_plan: list | None = None
        best_cost = 1 << 30
        while pq:
            _f, g, _c, snap, path, macros = heapq.heappop(pq)
            if best_plan is not None and _f >= best_cost:
                return best_plan            # see `exact_goal_test`
            for macro in macros:
                restore(eng, snap)
                won = self._apply_macro(eng, macro)
                nodes += 1
                if won >= 0:
                    plan = path + macro[:won + 1]
                    if not self.exact_goal_test:
                        return plan
                    if len(plan) < best_cost:
                        best_cost, best_plan = len(plan), plan
                    continue
                if eng.grid == snap:
                    # A push the engine refused -- a piece against the board edge,
                    # or a turn its rules cancelled outright (Add Man 2 cancels any
                    # turn leaving an unresolvable carry). Dedup would drop it a
                    # step later anyway; catching it here saves the reachability
                    # BFS, and these no-ops are a large slice of the branching.
                    continue
                if self.dead(eng):
                    # Same contract as `PSExpert._astar`: a state that can never
                    # reach a win is dropped rather than queued. Inert for the
                    # games that do not override `dead` (the base returns False),
                    # and the reachability BFS below is skipped for the ones that
                    # do -- Color Combination's colour algebra makes most wrong
                    # merges instantly, provably fatal.
                    continue
                nregion, nmacros = self._analyze(eng)
                k = self._region_key(eng, nregion)
                ng = g + len(macro)
                if best_g.get(k, 1 << 30) <= ng:
                    continue
                best_g[k] = ng
                counter += 1
                heapq.heappush(
                    pq,
                    (ng + w * self.heuristic(eng), ng, counter,
                     snapshot(eng), path + macro, nmacros),
                )
                if nodes >= self.node_cap:
                    return best_plan
        return best_plan


# ---------------------------------------------------------------------------
# The PLAIN sokoban (push crates onto targets, and nothing else)
# ---------------------------------------------------------------------------

#: Heuristic charge for a state that can no longer be won -- a piece parked where
#: no target can be reached from (a corner, or a pocket the player cannot get
#: behind). Large enough to sink the node to the back of the queue, finite so the
#: search stays complete.
DEAD_COST = 10_000


class PSSokobanExpert(PSPushExpert):
    """`PSPushExpert` plus the standard sokoban heuristic, for the games whose
    whole mechanic is ``[> Player | Piece] -> [> Player | > Piece]`` with an
    ``All Piece on Target`` win.

    THE HEURISTIC. Per-target PUSH-DISTANCE tables, built by a reverse BFS from
    the target over push edges: a piece reaches ``X`` from ``X - d`` only when
    ``X - d`` (where the piece comes from) and ``X - 2d`` (where the player has
    to stand) are both on the board and not blockers. The estimate is the sum
    over BARE targets of the table entry of a greedily matched free piece, plus
    the player's walk to the nearest loose piece.

    WHY THE TABLES AND NOT A DISTANCE. Counting uncovered targets, or summing
    manhattan distances, guides nothing on these boards: they are corridors and
    pockets, where the cost of a piece is its detour, not its separation. And the
    tables are also the DEADLOCK TEST, for free -- a cell with two adjacent walls
    has no outgoing push at all, so the reverse BFS never reaches it and it lands
    in no target's table. A piece pushed into a corner therefore scores
    `DEAD_COST` and its subtree sinks to the back of the queue instead of being
    explored, which is most of what makes a dense board tractable, since nearly
    every wrong early push buries a piece somewhere it can never leave.

    The greedy matching can overcount (it commits each target to its nearest free
    piece in board order rather than solving the assignment), so like the rest of
    this family the estimate is a search guide, not an optimality certificate --
    ``weight = 1`` plans are shortest in practice on the boards here but are not
    certified so. Where that matters, re-run under the fully admissible variant
    (drop the matching, take the nearest piece per target with reuse allowed) and
    compare, as `solvers/generate_bad_example_training.py` documents.

    Subclasses set ``pushable_names`` / ``target_names`` (and ``blocker_names``
    if the game's walls are not called ``wall``); nothing else is required.
    """

    #: Object (or or-group) names a piece has to end up on.
    target_names: tuple[str, ...] = ()

    def setup(self) -> None:
        super().setup()
        self.target_ids = {i for n in self.target_names
                           for i in self.g.resolve_object_name(n)}
        self._table_cache: dict = {}
        self._tables: dict = {}

    # -- per-target push-distance tables --------------------------------------
    def plan(self, eng, level: int | None = None):
        """Bind this board's distance tables, then plan as usual.

        The tables depend only on the blockers and the targets, neither of which
        any rule in a plain sokoban touches, so they are built ONCE per layout
        here rather than inside `heuristic` -- which runs on every node and must
        not re-scan the board for static facts."""
        self._tables = self._build_tables(eng)
        return super().plan(eng, level)

    def _build_tables(self, eng) -> dict:
        """``{target: {cell: pushes to get a piece from cell onto target}}``.

        Reverse BFS over push edges only, ignoring the other pieces (the standard
        sokoban relaxation). A cell absent from every table is a cell a piece can
        never leave -- see the deadlock note in the class docstring."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        wall = tuple(tuple(bool(cell & self.blocker_ids) for cell in row)
                     for row in grid)
        targets = tuple((r, c)
                        for r, row in enumerate(grid)
                        for c, cell in enumerate(row)
                        if cell & self.target_ids)
        sig = (wall, targets)
        cached = self._table_cache.get(sig)
        if cached is not None:
            return cached

        tables = {}
        for target in targets:
            steps = {target: 0}
            queue = deque([target])
            while queue:
                r, c = queue.popleft()
                cost = steps[(r, c)]
                for dr, dc in _DELTA.values():
                    pr, pc = r - dr, c - dc          # where the piece comes from
                    sr, sc = pr - dr, pc - dc        # where the player stands
                    if not (0 <= pr < h and 0 <= pc < w) or wall[pr][pc]:
                        continue
                    if not (0 <= sr < h and 0 <= sc < w) or wall[sr][sc]:
                        continue
                    if (pr, pc) not in steps:
                        steps[(pr, pc)] = cost + 1
                        queue.append((pr, pc))
            tables[target] = steps
        self._table_cache[sig] = tables
        return tables

    # -- heuristic ------------------------------------------------------------
    def heuristic(self, eng) -> int:
        push_ids, target_ids = self.push_ids, self.target_ids
        player = None
        loose: list[tuple[int, int]] = []     # pieces not yet on a target
        bare: list[tuple[int, int]] = []      # targets not yet covered
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & self.player_ids:
                    player = (r, c)
                on_target = bool(cell & target_ids)
                if cell & push_ids:
                    if not on_target:
                        loose.append((r, c))
                elif on_target:
                    bare.append((r, c))
        if not bare:
            return 0

        total = 0
        taken: set[int] = set()
        for goal in bare:
            steps = self._tables[goal]
            best = best_i = None
            for i, cell in enumerate(loose):
                if i in taken:
                    continue
                d = steps.get(cell)
                if d is not None and (best is None or d < best):
                    best, best_i = d, i
            if best is None:
                return DEAD_COST     # no free piece can ever reach this target
            taken.add(best_i)
            total += best
        if player is not None:
            # The walk to the first push. -1 because the player pushes from the
            # cell BESIDE the piece, not from the piece's cell.
            total += max(0, min(abs(player[0] - r) + abs(player[1] - c)
                                for r, c in loose) - 1)
        return total


# ---------------------------------------------------------------------------
# Breadth-first BEAM over the same macros (for the games A* cannot reach)
# ---------------------------------------------------------------------------

class PSBeamExpert(PSPushExpert):
    """`PSPushExpert`'s macros, explored by a width-capped breadth-first beam.

    WHY (what A* runs out of). Both searches pay the same thing per node -- one
    interpreter step per primitive in a macro, and that step is the entire cost
    of these searches. A* spends that budget on the frontier its heuristic
    likes, which is the right call only while the heuristic can actually tell
    good states from bad. In the deep puzzle levels it cannot: the estimate is
    the player's distance to the exit, and the middle of a solution is a dozen
    moves that reposition crates and portals without the player getting one step
    closer. Over that stretch the heuristic is FLAT, weighting it changes
    nothing, and A* degenerates to uniform-cost search at a depth where that is
    hopeless. A beam spends the same budget on breadth at every depth instead,
    and only uses the heuristic to decide who survives a tie -- which is all a
    flat heuristic is good for.

    The trade is optimality: plans are winning and engine-verified but longer
    than A*'s (a beam plan wanders where A* would not). Prefer `PSPushExpert`
    and fall back to this only for levels A* cannot reach at all.

    ``beam_width`` states survive each depth, ranked by `heuristic`; dedup is
    the same region key, kept across the WHOLE search rather than per depth, so
    a state re-reached later never re-expands.
    """

    #: States kept per depth. Cost is roughly width x branching x macro length
    #: interpreter steps per depth, so this is the wall-clock dial.
    beam_width: int = 300
    #: Give up after this many macro layers.
    beam_depth: int = 30
    #: Macro budget for the whole beam (`node_cap` stays A*'s).
    beam_node_cap: int = 100_000

    def _search(self, eng) -> list | None:
        if eng.check_win():
            return []
        region, macros = self._analyze(eng)
        if region is None:
            return None
        frontier = [(snapshot(eng), [], macros)]
        seen = {self._region_key(eng, region)}
        nodes = 0
        for _depth in range(self.beam_depth):
            kids = []
            for snap, path, macs in frontier:
                for macro in macs:
                    restore(eng, snap)
                    nodes += 1
                    won = self._apply_macro(eng, macro)
                    if won >= 0:
                        return path + macro[:won + 1]
                    if eng.grid == snap:
                        continue                  # a move the engine refused
                    nregion, nmacros = self._analyze(eng)
                    if nregion is None:
                        continue                  # the player died
                    key = self._region_key(eng, nregion)
                    if key in seen:
                        continue
                    seen.add(key)
                    # Sort on the heuristic ALONE -- the tuple also carries a
                    # snapshot (a list of sets), which has no ordering.
                    kids.append((self.heuristic(eng), snapshot(eng),
                                 path + macro, nmacros))
                    if nodes >= self.beam_node_cap:
                        return None
            if not kids:
                return None                       # the reachable space closed
            kids.sort(key=lambda kid: kid[0])
            frontier = [(snap, path, macs)
                        for _h, snap, path, macs in kids[:self.beam_width]]
        return None


# ---------------------------------------------------------------------------
# EXHAUSTIVE enumeration (for the games whose reachable space is TINY)
# ---------------------------------------------------------------------------

#: Sentinel successor for "this press wins". Deliberately not a state key: a win
#: ends the episode, and the interpreter can leave mid-tick bookkeeping on the
#: board there (a spent Cursor, a half-run `again` loop), so keying it would
#: split one goal into several noise states.
WIN = "win"


class StateGraph:
    """Every state reachable from one board, with its exact distance to a win.

    Built by stepping the real interpreter. Use it INSTEAD of a heuristic search
    for a ps: game whose mechanic prunes the space down to tens of states -- a
    lethal move (ps:gdd301_game restarts the level the moment you touch your own
    trail) or a coarse move (ps:roller_boi's one press is a whole SLIDE, so only
    the cells you can come to rest on are states at all). Measure before
    choosing: enumeration is only cheap because the space is small, and it has
    no node budget to fall back on.

    What it buys over `PSExpert._astar`, all three of which need the WHOLE space
    and so are unavailable to a search:

    * plans that are provably SHORTEST -- no weight, no heuristic, no node cap
      that could quietly bite;
    * EXACT optimal-action sets (``dist(succ) == dist - 1``) rather than the
      inferred ones `PSPushExpert.annotate_walks` produces;
    * a PROOF of unwinnability for a level that has no plan, instead of "the
      search gave up": the enumeration terminates having generated every state
      the interpreter admits, and none of them has a winning edge.

    ``succ[k]`` maps a press to the state key it lands in, or `WIN`. A press is
    ABSENT from ``succ[k]`` when it cannot be taken: it restarted the level or it
    changed nothing at all. Both are non-moves for planning -- a restart returns
    to the level start, so no shortest path ever contains one.
    """

    __slots__ = ("succ", "dist", "start", "steps")

    def __init__(self, succ: dict, dist: dict, start, steps: int):
        self.succ = succ
        self.dist = dist
        self.start = start
        self.steps = steps

    @classmethod
    def build(cls, eng, key_fn, directions, node_cap: int = 200_000,
              decode=None) -> "StateGraph | None":
        """Enumerate from the engine's current board. Restores it before
        returning; returns None past ``node_cap`` states (a runaway guard -- the
        spaces this is for are tens of states, not thousands).

        ``decode(eng, k)`` is an opt-in MEMORY trade: supply it when ``key_fn``
        is a lossless encoding of the board, and a state is re-seated by
        decoding its key instead of by keeping a snapshot of it. The snapshot is
        what makes enumeration expensive in RAM rather than in time -- it is a
        fresh ``set`` per cell, ~64 KB on a 6x12 grid against ~150 bytes for a
        packed key -- so a space of tens of thousands of states goes from
        gigabytes to megabytes and stops being the reason to reach for a
        heuristic search instead. Left None every state is snapshotted, which is
        the right default: most `_key`s here drop something (they scope to the
        dynamic objects) and could not be decoded back.

        A decoder that is not exactly ``key_fn``'s inverse silently enumerates a
        DIFFERENT game, so a game using this owes a report that re-walks the
        space and compares a decoded step against a snapshotted one."""
        start_snap = snapshot(eng)
        start = key_fn(eng)
        # ``seen`` is the state set; ``snaps`` is empty when the keys decode.
        snaps = {} if decode is not None else {start: start_snap}
        seen = {start}
        succ: dict = {}
        steps = 0
        queue = deque([start])
        while queue:
            k = queue.popleft()
            edges: dict = {}
            for d in directions:
                if decode is not None:
                    decode(eng, k)
                else:
                    restore(eng, snaps[k])
                eng._rule_restart = False
                eng.step(d)
                steps += 1
                if eng._rule_restart:
                    continue                 # death: back to the level start
                if eng.check_win():
                    edges[d] = WIN
                    continue
                nk = key_fn(eng)
                if nk == k:
                    continue                 # a press that did nothing at all
                if nk not in seen:
                    if len(seen) >= node_cap:
                        restore(eng, start_snap)
                        eng._rule_restart = False
                        return None
                    seen.add(nk)
                    if decode is None:
                        snaps[nk] = snapshot(eng)
                    queue.append(nk)
                edges[d] = nk
            succ[k] = edges
        restore(eng, start_snap)
        eng._rule_restart = False
        return cls(succ, cls._distances(succ), start, steps)

    @staticmethod
    def _distances(succ: dict) -> dict:
        """Presses-to-win for every state, by backward BFS from the winning
        edges over the reversed graph. States absent from the result cannot win
        at all."""
        rev: dict = {}
        dist: dict = {}
        frontier = []
        for k, edges in succ.items():
            for d, nk in edges.items():
                if nk is WIN or nk == WIN:
                    if k not in dist:
                        dist[k] = 1
                        frontier.append(k)
                else:
                    rev.setdefault(nk, []).append(k)
        queue = deque(frontier)
        while queue:
            k = queue.popleft()
            for p in rev.get(k, ()):
                if p not in dist:
                    dist[p] = dist[k] + 1
                    queue.append(p)
        return dist

    def cost(self, k, d) -> "int | None":
        """Presses to win if ``d`` is pressed at ``k``, or None if that press is
        unavailable or leads nowhere."""
        nk = self.succ[k].get(d)
        if nk is None:
            return None
        if nk == WIN:
            return 1
        rest = self.dist.get(nk)
        return None if rest is None else rest + 1

    def optimal(self, k, directions) -> list:
        """Every press that is on a shortest path from ``k``."""
        best = self.dist.get(k)
        if best is None:
            return []
        return [d for d in directions if self.cost(k, d) == best]

    def plan(self, directions) -> "Plan | None":
        """The shortest press sequence from the start, with the exact optimal SET
        at every step. Ties are broken by ``directions`` order, which is what
        makes a re-derived plan byte-identical across processes."""
        if self.start not in self.dist:
            return None
        presses, optsets = [], []
        k = self.start
        while True:
            best = self.optimal(k, directions)
            presses.append(best[0])
            optsets.append(best)
            nk = self.succ[k][best[0]]
            if nk == WIN:
                return Plan(presses, optsets)
            k = nk


class PSEnumExpert(PSExpert):
    """`PSExpert`'s plan memo and snapshot discipline around a `StateGraph`.

    `_search` is replaced the way ps:dotsnake / ps:full_circle replace it -- the
    base class keeps the memo, the restore discipline and the level scoping, and
    only the strategy underneath changes. Here the strategy is "enumerate the
    whole reachable component and read the distance field", so `heuristic` is
    never called and asserts rather than returning a number nothing would use.

    Subclasses normally set only `directions` (drop the presses no rule reads --
    branching on them doubles the interpreter steps for nothing). `_key` is
    inherited: every non-background cell, which is exact and canonical ACROSS
    levels for these games because the walls are in the key too, so no
    ``scope_by_level`` is needed.

    A game whose space runs to tens of thousands of states can additionally
    override `enum_key` / `enum_decode` -- a PACKED, decodable spelling of the
    same board, used only inside the enumeration -- and `StateGraph.build` then
    keeps no snapshots at all. `_key` is deliberately left alone by that: it is
    what the plan memo and the ``plan_cache_path`` signature are built from, and
    the disk layer reads it as ``(r, c, obj)`` triples.
    """

    def heuristic(self, eng) -> int:
        raise AssertionError(
            f"{type(self).__name__} enumerates the reachable space; "
            "heuristic is unused")

    def enum_key(self, eng):
        """The state key the ENUMERATION uses. `_key` by default."""
        return self._key(eng)

    def enum_decode(self, eng, key) -> None:
        """Seat ``key`` (from `enum_key`) back onto the engine, or None when the
        keys are not decodable and `StateGraph.build` must keep snapshots."""
        return None

    def graph(self, eng) -> "StateGraph | None":
        decode = (None if type(self).enum_decode is PSEnumExpert.enum_decode
                  else self.enum_decode)
        return StateGraph.build(eng, self.enum_key, self.directions,
                                self.node_cap, decode)

    def _search(self, eng) -> "Plan | None":
        graph = self.graph(eng)
        return None if graph is None else graph.plan(self.directions)


# ---------------------------------------------------------------------------
# Trajectory recording
# ---------------------------------------------------------------------------

def _frame_to_list(frame) -> list:
    return np.asarray(frame).tolist()


def record_level(expert: PSExpert, game: PuzzleScriptAdapter, level: int,
                 epsilon: float, rng: random.Random, max_steps: int,
                 *, remap: bool = True,
                 schedule: EpsilonSchedule | None = None,
                 exploration: ExplorationPolicy | None = None,
                 solver: BaseSolver | None = None,
                 spans: bool = False, optimal_fn=None,
                 vacuous_start_win: bool = False):
    """Reset to a (seed, level) start and drive the expert to a WIN, recording the
    presented (rotated / recolored) display frames. Returns ``(final_state, obs,
    actions)``, or ``(None, None, None)`` if the start is unwinnable.

    The plan is fetched ONCE from the start state and then followed, rather than
    re-planned every step: a fresh A* from an early state costs real time, and the
    start state is the one entry guaranteed to be in the seed-independent cache. We
    only re-plan when an epsilon detour knocks us off the cached path.

    When ``exploration`` is supplied (recovery ON), an epsilon-decayed exploration
    prefix runs first and -- unless it wins outright -- is undone by ONE RESET back
    to the level's initial state (adapter ``set_level`` reloads the level fresh and
    clears any GAME_OVER), which is exactly the state the cached plan was solved
    from, so it stays a valid WIN path. See `BaseSolver._reset_prefix`.

    THREE OPT-IN KNOBS, all no-ops when unset so every game already recorded here
    stays byte-identical:

      * ``spans`` records an action's WHOLE animation instead of only its settled
        frame, tagging the step with ``n_obs`` (see `BaseSolver.record_level` for
        the alignment rule). Needed by any game where the interesting mechanic
        happens INSIDE one keypress -- botsket_ball's play button runs the entire
        simulation in the press that starts it, so the settled-frame-only record
        would show the machine being built and then, with nothing in between, the
        ball sitting in the net.
      * ``optimal_fn(plan, pi) -> [engine directions] | None`` supplies the
        optimal action SET for a step. It is called with the engine still AT the
        state being labelled (before the press is executed), so an oracle that
        reads the live grid can answer from it directly rather than replaying the
        plan from the level start. It is what the games whose plans contain
        genuinely order-free stretches need (a free-roaming cursor walking to a
        cell can take any interleaving of the two axes): without it every step
        trains a single arbitrary choice as the only right answer.
      * ``vacuous_start_win`` says the win predicate is already satisfied on the
        level's START frame, so it must not be believed until a press has
        happened. Flood is the case: its condition is ``No Wave3 and Some
        Player`` and nothing has spread yet at reset, so without this the level
        tapes a zero-press "win", never reaches ``GameState.WIN``, and is
        dropped. Off by default because for every other game here a start that
        reads as won IS won.
    """
    game.set_level(level)
    eng = game._engine
    expert.observe(eng)
    rot_k, hflip, vflip = game._rotation_k, game._hflip, game._vflip

    # Plan before recording anything, so a level whose start is unwinnable (e.g.
    # enqueue's two-queue levels, which the interpreter cannot simulate) is cleanly
    # skipped rather than half-taped.
    if eng.check_win() and not vacuous_start_win:
        plan: list | None = []
    else:
        plan = expert.plan(eng, level)
        if plan is None:
            return None, None, None

    observations = [_frame_to_list(game._current_frame)]
    actions = [{"type": "simple", "index": int(GameAction.RESET.value)}]

    # RESET-recovery exploration prefix (no-op unless recovery is enabled).
    if exploration is not None and solver is not None:
        def _drive_explore(act):
            if act.is_click:
                ai = ActionInput(id=GameAction.ACTION6,
                                 data={"x": act.click_xy[0], "y": act.click_xy[1]})
            else:
                ai = ActionInput(id=_ID_TO_GAMEACTION[act.action_id])
            fd = game.perform_action(ai)
            expert.observe(eng)
            fr = (list(fd.frame) if (spans and fd.frame)
                  else fd.frame[-1] if fd.frame else game._current_frame)
            term = ("win" if game._state == GameState.WIN
                    else "over" if game._state == GameState.GAME_OVER else None)
            return np.asarray(fr), term

        def _reset_to_level():
            game.set_level(level)
            expert.observe(eng)
            return np.asarray(game._current_frame)

        solver._reset_prefix(
            schedule=schedule, exploration=exploration,
            prev=np.asarray(game._current_frame),
            drive_explore=_drive_explore, reset_to_level=_reset_to_level,
            record_obs=lambda fr: observations.append(_frame_to_list(fr)),
            record_act=lambda d: actions.append(d))
        # Drop the plan fetched above so the loop re-plans from the state the
        # prefix left behind. That state IS the pre-fetch state (the prefix ends
        # in a RESET back to the level start, unless it won outright, which the
        # loop's win check catches first), so for a memoized expert this is a
        # cache hit and the recording is byte-identical. It matters for an expert
        # whose plan depends on what it has SEEN rather than only on where the
        # pieces are (see `PSExpert.observe`): the prefix has just shown it
        # frames, and re-planning is what lets those count.
        plan = None

    pi = 0
    pressed = False
    for _ in range(max_steps):
        if eng.check_win() and (pressed or not vacuous_start_win):
            break
        if plan is None or pi >= len(plan):
            plan = expert.plan(eng, level)           # (re)plan from current state
            pi = 0
            if not plan:                             # None (stuck) or [] (won)
                break
        direction = plan[pi]

        # Epsilon detour: take a random alternative action, but only keep it if it
        # changes the state AND leaves the level still winnable, so we never trap
        # the puzzle unrecoverably. A detour invalidates the cached path, so it
        # forces a re-plan on the next iteration.
        detoured = False
        if epsilon > 0.0 and rng.random() < epsilon:
            alts = [d for d in expert.directions if d != direction]
            rng.shuffle(alts)
            probe = snapshot(eng)
            for alt in alts:
                eng.step(alt)
                new_snap = snapshot(eng)
                still_ok = expert.plan(eng, level) is not None
                restore(eng, probe)
                if still_ok and new_snap != probe:
                    direction = alt
                    detoured = True
                    break

        # The label is asked for BEFORE the press is executed, so ``optimal_fn``
        # sees the engine at the state it is labelling. Every generator here
        # derives its set from ``(level, plan, pi)`` alone and is unaffected by
        # the ordering (verified byte-identical), but a generator with a live
        # distance oracle needs the pre-press grid to answer "which presses are
        # optimal HERE" -- and after the press that state is gone (a crate shoved
        # into a corner is not undoable by stepping back).
        best = optimal_fn(plan, pi) if optimal_fn is not None else None

        act = screen_action(direction, rot_k, hflip, vflip, remap)
        fd = game.perform_action(ActionInput(id=act))
        expert.observe(eng)
        step_frames = (list(fd.frame) if (spans and fd.frame)
                       else [fd.frame[-1] if fd.frame else game._current_frame])
        for fr in step_frames:
            observations.append(_frame_to_list(fr))
        record = {"type": "simple", "index": int(act.value)}
        if spans:
            record["n_obs"] = len(step_frames)
        if best:
            record["optimal"] = [
                {"type": "simple",
                 "index": int(screen_action(d, rot_k, hflip, vflip,
                                            remap).value)}
                for d in best]
        actions.append(record)
        pressed = True
        if detoured:
            plan = None          # off the cached path -> re-plan next step
        else:
            pi += 1

    return game._state, observations, actions


def discover_solvable(expert: PSExpert, game: PuzzleScriptAdapter,
                      levels: list[int],
                      vacuous_start_win: bool = False) -> list[int]:
    """Return the subset of ``levels`` whose start state the expert can win. Engine
    state is seed-independent for these games, so this holds for every seed.

    ``vacuous_start_win`` drops the "already won" shortcut -- see `record_level`."""
    solvable: list[int] = []
    for level in levels:
        game.set_level(level)
        if ((game._engine.check_win() and not vacuous_start_win)
                or expert.plan(game._engine, level) is not None):
            solvable.append(level)
    return solvable


# ---------------------------------------------------------------------------
# BaseSolver harness
# ---------------------------------------------------------------------------

class PSAStarSolver(BaseSolver):
    """`BaseSolver` for a PuzzleScript game solved by engine-blackbox A*.

    These generators drive an external interpreter (not the native `ARCBaseGame`),
    and an episode keeps only the SOLVABLE levels -- levels the interpreter cannot
    simulate, or that the primitive-action search cannot win within the node
    budget, are cleanly skipped rather than emitted as broken data. Because each
    level is an independent trajectory in the training schema, an episode holding
    only the solvable levels is valid, and ``level_id`` preserves the true
    (seed, level) provenance. Neither of those fits the base's native record/replay
    loop, so `solve_episode` is overridden and only the CLI (``main``/``run``) and
    the ``{"game_id", "levels": [...]}`` save schema are inherited.

    Subclasses set ``game_id``, ``game_name``, ``expert_cls`` and, if needed,
    ``skip_levels`` / ``node_cap`` / ``weight`` / ``max_steps``.
    """

    #: PuzzleScript game name passed to `PuzzleScriptAdapter`.
    game_name: str = ""
    #: `PSExpert` subclass supplying the goal heuristic.
    expert_cls: type[PSExpert] = PSExpert
    #: Levels never attempted (the search cannot win them within a practical
    #: budget). Skipped up front so discovery does not burn ~node_cap nodes per
    #: level fruitlessly at every startup.
    skip_levels: frozenset[int] = frozenset()

    #: Require EVERY level to win for the seed to count, instead of discovering a
    #: seed-independent solvable subset once. Set this when solvability is itself
    #: seed-dependent (Drop_Maze's rotation-start can strand the ball), where a
    #: one-shot discovery at seed 0 would wrongly drop a level for every later
    #: seed. It also skips that startup discovery pass entirely.
    require_all_levels: bool = False

    #: False for `PuzzleScriptAdapter._NO_ACTION_REMAP_GAMES` -- see `screen_action`.
    remap_actions: bool = True

    #: ``ps:<name>`` folder under ``games/`` whose ``make_game`` builds the
    #: adapter. Set it whenever that wrapper does more than construct the adapter
    #: (a recolor, a sprite fix); see `make_game`.
    game_module_id: str = ""

    #: Record each action's WHOLE animation (with ``n_obs``) rather than only its
    #: settled frame. Opt-in: see `record_level`.
    record_spans: bool = False

    #: The win predicate already holds on the level's START frame, so nothing
    #: may believe it until a press has happened. Opt-in: see `record_level`.
    vacuous_start_win: bool = False

    node_cap: int = 400_000
    weight: int = 1
    max_steps: int = 300
    #: Epsilon-detour rate inside the expert replay. Recovery data comes from the
    #: RESET prefix instead (see the module docstring), so this stays 0.
    epsilon: float = 0.0

    supports_recovery = True
    recovery_mode = "reset"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._game = None        # PuzzleScriptAdapter, reused across seeds
        self._expert = None      # expert + its seed-independent plan cache
        self._solvable = None    # levels the expert can win (seed-independent)

    def _ensure(self, seed: int):
        """Lazily build (and cache across seeds) the adapter, expert and the
        seed-independent solvable-level set. Sharing one adapter + one plan cache
        across every seed is what makes generation cheap: seed 0 pays for the
        searches, later seeds only re-render them at their own augmentation."""
        if self._game is None:
            self._game = self.make_game(seed)
            self._expert = self.expert_cls(self._game, node_cap=self.node_cap,
                                           weight=self.weight)
            self.prepare_expert(self._game, self._expert)
            levels = [lvl for lvl in range(self._game.n_levels)
                      if lvl not in self.skip_levels]
            self._solvable = (levels if self.require_all_levels else
                              discover_solvable(self._expert, self._game, levels,
                                                self.vacuous_start_win))
        return self._game, self._expert, self._solvable

    def prepare_expert(self, game, expert) -> None:
        """Hook: run once, after the expert is built and BEFORE the solvable
        levels are discovered. For pre-seeding ``expert.cache`` with plans that
        are too expensive to re-derive at every startup -- discovery is the
        first thing that would pay for them, so a hook that ran later would be
        too late to help."""

    def epsilon_for(self, level: int) -> float:
        """The epsilon-detour rate to record ``level`` at. Defaults to the flat
        ``epsilon`` for every level, so nothing recorded before this existed
        changes.

        Override it for a game whose levels do not share a press budget. The
        adapter cuts a level off at 200 presses and `record_level` resets that
        counter with the ``set_level`` that ends the exploration prefix, so the
        margin a detour can spend is ``200 - len(plan)`` -- which is generous on
        a 13-press level and nearly nothing on a 169-press one. ps:tornado_tamer
        is the case: five of its six levels take 10 to 46 presses and its fourth
        takes 169, because the player is sealed in a room and has to relay
        tornadoes through four splits to reach the far leaf.
        """
        return self.epsilon

    def explore_level(self, level: int) -> bool:
        """Whether ``level`` gets the RESET-recovery exploration prefix. True for
        every level by default, so nothing recorded before this existed changes.

        Override it for a game with a level SHORTER THAN THE PREFIX. The prefix
        is a random walk that ends either at its own length or the moment it
        terminates the level, and on a one- or two-press level it usually wins
        outright -- at which point `_reset_prefix` skips its closing RESET, the
        expert is never asked for a plan, and the whole level is taped as
        unlabelled ``phase="explore"`` steps. That is a valid trajectory but it
        carries no ``optimal``, so the level trains nothing (see
        `train_policy`: the taken action is context, the target is ``optimal``).
        ps:wizard_school is the case: its first two levels are the game's
        tutorial and are one and two presses long, and the prefix was winning
        level 0 in 18 recordings out of 20.
        """
        return True

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The optimal ENGINE-direction SET at plan step ``pi``, or None to fall
        back to "the action taken is the only optimal one".

        Default: the step's entry in a `Plan`'s ``optsets`` when the expert
        produced one (`PSPushExpert.annotate`), else None -- which is right for
        a search whose plan is a single shortest path with no order-free
        stretches. Override where the sets have to be derived some other way --
        see `record_level`'s ``optimal_fn``."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets):
            return sets[pi]
        return None

    def make_game(self, seed: int):
        """The adapter this generator records against.

        When ``game_module_id`` is set, the adapter is built by that game folder's
        ``make_game`` instead of directly. Some ps: games need it: their
        ``games/<id>/<id>.py`` patches the parsed game before play (Add Man 2
        recolors the target indicator, whose original Blue + DarkBlue collapse to
        a single ARC palette index and render the 0-vs-1 marker invisible), and
        that patched adapter is exactly what `game_envs` hands a live agent. A
        generator that built its own adapter would tape frames the agent never
        sees. Left empty for the games whose wrapper is a plain passthrough."""
        if self.game_module_id:
            return _load_game_module(self.game_module_id).make_game(seed=seed)
        return PuzzleScriptAdapter(self.game_name, seed=seed)

    def solve_episode(self, seed: int, explore: bool = True):
        """Solve every SOLVABLE level for one seed; succeed only when all of them
        win (the WIN-only corpus contract). Returns ``(ok, levels)``."""
        game, expert, solvable = self._ensure(seed)
        if not solvable:
            return False, []
        game._seed = seed        # reuse the parsed game; augmentation re-derives
        do_explore = explore and self.supports_recovery
        schedule = (EpsilonSchedule(self.rng, center=self._explore_center,
                                    jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(game), self.rng)
                       if do_explore else None)
        rng = random.Random(seed)
        levels_out = []
        for level in solvable:
            # `explore_level` defaults to True, so this is the same pair of
            # objects every generator here was already passing.
            wants = self.explore_level(level)
            state, obs, acts = record_level(
                expert, game, level, self.epsilon_for(level), rng, self.max_steps,
                remap=self.remap_actions,
                schedule=schedule if wants else None,
                exploration=exploration if wants else None, solver=self,
                spans=self.record_spans,
                vacuous_start_win=self.vacuous_start_win,
                optimal_fn=(lambda plan, pi, _lvl=level:
                            self.optimal_for(expert, _lvl, plan, pi)))
            if state is None or state != GameState.WIN:
                continue
            levels_out.append({"level_id": level, "observations": obs,
                               "actions": acts})
        return len(levels_out) == len(solvable), levels_out
