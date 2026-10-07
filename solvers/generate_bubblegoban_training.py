"""Generate Phase-1 training data for the PuzzleScript game ps:bubblegoban
("Bubblegoban" by Franklin P. Dyer -- bubblegum sokoban).

The harness -- the rotation contract, the trajectory recorder, the RESET-recovery
prefix and the BaseSolver plumbing -- lives in `solvers/common/ps_astar.py`,
shared with the other ps: generators. This file is the game-specific part: the
mechanic notes, the exact distance field over the real interpreter, and the
optimal-action labeller.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_bubblegoban",
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
exactly.

The game
--------
Walk the player onto the single Target (``All Target on Player``). Four arrow
keys, no ACTION -- nothing in the file binds it, so it is dropped from the search.
What makes it a puzzle is the Gum, and it is worth reading the three rules in the
order the engine runs them:

    [ > Stickable | Stickable ] -> [ > Stickable | > Stickable ]     (push)
    startloop
      [ moving Gum | Stickable ] -> [ moving Gum | moving Stickable ]
      [ moving Stickable | Gum ] -> [ moving Stickable | moving Gum ] (stick)
      [ > Stickable | Stickable ] -> [ > Stickable | > Stickable ]
    endloop
    [ Moving Gum | Wall ] -> cancel                                  (the trap)
    [ > Player | Wall ] -> cancel

``Stickable = Player or Gum or Box``, so:

  * **Gum is glue in all four directions.** The two sticky rules have no
    direction prefix, so they fire on every 4-neighbour pair with a Gum on either
    end. Whatever the player is touching moves with it -- sideways, backwards,
    any direction -- and the loop propagates that through a whole cluster. Pick a
    gum up and you carry it for the rest of the level: **there is no way to put it
    down**, and nothing in the game deletes gum.
  * **A Box is dead weight, not glue.** It is pushed like a sokoban crate (the
    push rule) and it is *carried* when a gum is stuck to it, but a box next to
    the player with no gum between them does not follow. A box shoved into a wall
    simply fails to move (same collision layer, no cancel rule) -- the turn is a
    silent no-op, not a trap.
  * **The trap: a moving gum ADJACENT to a wall cancels the whole turn.** The
    rule is directionless, so it is not "gum pushed into a wall" -- it is "gum
    with a wall anywhere beside it, moving at all". Since every gum in the
    player's cluster moves on every press, a cluster gum that ends up beside a
    wall makes *every* key cancel, forever. **That is a permanent, silent loss:**
    no GAME_OVER, no message, the board just stops responding. `dead()` below is
    exactly that predicate, and it is why the reachable state space is small
    enough to solve exhaustively.

So the whole game is: route to the target through a minefield of gum, where the
mine only arms once you touch it, and every gum you touch has to be dragged
through open ground for the rest of the level.

Expert solver
-------------
An EXACT distance field over the real interpreter, not a search. Forward BFS from
the level's start over engine states (`_key` = the player / gum / box cells;
walls and the target are static, hence ``scope_by_level``), pruning dead states,
then one backward BFS from the winning transitions gives ``dist`` for every
reachable state. The plan is the greedy descent, and it is optimal by
construction.

Why the whole field rather than `PSExpert`'s A*: the reachable space is *tiny*
(62 / 145 / 258 / 658 / ... states per level) because `dead()` prunes the instant
the player glues itself to a wall, and almost every wrong move does that. For the
price of one A* -- the field costs 4 engine steps per state, ~30s for the biggest
level -- it buys the thing A* cannot give: ``dist`` at EVERY state, so
`optimal_for` labels each step with the complete set of equally-shortest presses
instead of one arbitrary tie-break. On these open boards the ties are the common
case (any interleaving of two axes crosses an empty field in the same number of
moves), so labelling one of them as "the" answer would train against the truth on
most steps.

The field is built once per level at seed 0 and reused: engine state after reset
is seed-independent for this game (levels are fixed ASCII maps, only the
PRESENTATION is augmented), so every later seed replays it under its own
rotation/flip. The interpreter step is 7ms and is ~98% of that cost (the whole
build is 227s for the seven levels), so the start plan and its optimal sets are
also written to ``data/bubblegoban_plans.json``: without it every shard of
`parallelize_generator` would rebuild all seven fields on every core. Warm, a run
starts in about a second. Delete the file to re-derive it.

Augmentation
------------
Bubblegoban is gravity-free with direction-agnostic rules and screen-relative
input, so the board's full 8-element symmetry group is a valid presentation
augmentation: ``Bubblegoban`` is in `PuzzleScriptAdapter._FLIP_GAMES`, giving
rotation_k in {0,1,2,3} x horizontal x vertical flip, each with the matching
directional action remap. No colour augmentation (the game is not in
``_RECOLOR_GAMES``): gum-versus-wall is the mechanic, and a flattening recolor
could merge them.

Usage (run from the repo root):
    python solvers/generate_bubblegoban_training.py --episodes 200 \
        --out data/training_multi_level/bubblegoban

    python solvers/generate_bubblegoban_training.py --plans      # level report
    python solvers/generate_bubblegoban_training.py --selfcheck  # mechanic audit
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters.puzzlescript_adapter import PuzzleScriptAdapter        # noqa: E402
from solvers.common.ps_astar import (PSAStarSolver, PSExpert,        # noqa: E402
                                     restore, snapshot)

GAME_NAME = "Bubblegoban"

_REPO_ROOT = Path(__file__).resolve().parent.parent

#: Disk cache of the per-level start plan AND its optimal-action sets. The field
#: is exact and seed-independent but costs ~4 minutes of interpreter steps for
#: the seven levels, which every shard of `parallelize_generator` would otherwise
#: repeat on every core. Delete the file to re-derive it.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "bubblegoban_plans.json"

_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Sentinel successor for a press that ends the level.
_WIN = "win"


class Plan(list):
    """The press sequence, carrying the optimal SET for each of its steps.

    ``optsets[i]`` is every press that is equally shortest at the state the i-th
    press is taken from -- computed from the field while it is in memory, so a
    cached plan replayed in a later process still labels every step without
    rebuilding anything."""

    optsets: list

    def __init__(self, presses, optsets):
        super().__init__(presses)
        self.optsets = optsets


class BubblegobanExpert(PSExpert):
    """Exact shortest-path oracle: a full BFS distance field over the real engine
    dynamics, with the gum-glued-to-a-wall states pruned as dead.

    See the module docstring for why the field is affordable and what it buys
    over `PSExpert`'s A*. `plan` is overridden outright, so `PSExpert._astar`
    never runs and this class deliberately implements no ``heuristic`` -- the
    field IS the exact distance, and a second, approximate copy of the same
    answer would only be something to keep in sync.
    """

    #: ACTION is bound to nothing in this game -- no rule mentions it, so it is a
    #: pure no-op and only costs an engine step per node.
    directions = ["up", "down", "left", "right"]

    #: `_key` keeps the dynamic objects only; walls and the target are static per
    #: level but differ between levels.
    scope_by_level = True

    def setup(self) -> None:
        g = self.g
        self.wall = g.obj_name_to_idx["wall"]
        self.gum = g.obj_name_to_idx["gum"]
        self.box = g.obj_name_to_idx["box"]
        self.player_ids = set(self.game._engine._player_indices)
        #: Everything a rule can move. Sticky partners are gum/box/player.
        self.dyn_ids = {self.gum, self.box} | self.player_ids
        #: level -> {state key: (dist, {direction: successor key or _WIN})}
        self._fields: dict[int | None, dict] = {}
        #: level -> the cached start plan (see `PLAN_CACHE`)
        self._disk = self._load_disk()

    # -- state ----------------------------------------------------------------
    def _key(self, eng) -> frozenset:
        dyn = self.dyn_ids
        return frozenset(
            (r, c, o)
            for r, row in enumerate(eng.grid)
            for c, cell in enumerate(row)
            for o in (cell & dyn)
        )

    def _cluster(self, eng) -> set | None:
        """The cells that move when the player moves: the connected component of
        the player under "the two cells are 4-adjacent and at least one of them
        holds a Gum".

        That edge relation is exactly the pair of sticky rules -- both need a Gum
        on one end -- and the ``startloop`` around them takes the transitive
        closure, so the whole component is set moving by any press. Returns None
        if the player is gone (it never is in this game; a rule that consumed the
        player would make every state dead, which is the safe reading)."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        gum, dyn = self.gum, self.dyn_ids
        start = None
        for r in range(h):
            row = grid[r]
            for c in range(w):
                if row[c] & self.player_ids:
                    start = (r, c)
                    break
            if start is not None:
                break
        if start is None:
            return None
        seen = {start}
        queue = deque([start])
        while queue:
            r, c = queue.popleft()
            here_gum = gum in grid[r][c]
            for dr, dc in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w) or (nr, nc) in seen:
                    continue
                cell = grid[nr][nc]
                if not (cell & dyn):
                    continue
                if here_gum or gum in cell:
                    seen.add((nr, nc))
                    queue.append((nr, nc))
        return seen

    def dead(self, eng) -> bool:
        """True when the level can never be won again: some gum in the player's
        cluster has a wall beside it.

        Every press moves the whole cluster (see `_cluster`), and
        ``[ Moving Gum | Wall ] -> cancel`` then cancels the turn -- in every
        direction, forever. The player is welded to the board. This is the single
        most important fact about the game and the reason a full field fits: it
        prunes the branch the moment the mistake is made, instead of letting the
        search wander a dead board.

        Sound by construction (a cluster gum beside a wall really does cancel
        everything; `selfcheck` asserts it against the interpreter), and it is a
        pure loss test -- a state it calls dead can hold no win."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        cells = self._cluster(eng)
        if cells is None:
            return True
        for (r, c) in cells:
            if self.gum not in grid[r][c]:
                continue
            for dr, dc in _NEIGHBOURS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and self.wall in grid[nr][nc]:
                    return True
        return False

    # -- the distance field ----------------------------------------------------
    def _build_field(self, eng, level: int | None) -> dict:
        """Forward-explore every state reachable from the engine's current state,
        then backward-BFS the distances from the winning transitions.

        Returns ``{key: (dist, {direction: successor})}`` where ``successor`` is
        another key or `_WIN`, and ``dist`` is None for a state from which no win
        is reachable. Leaves the engine grid unchanged."""
        root = snapshot(eng)
        root_key = self._key(eng)
        succ: dict[frozenset, dict] = {}
        snaps = {root_key: root}
        queue = deque([root_key])
        while queue:
            key = queue.popleft()
            if key in succ:
                continue
            edges: dict[str, object] = {}
            succ[key] = edges
            for direction in self.directions:
                restore(eng, snaps[key])
                eng.step(direction)
                if eng.check_win():
                    edges[direction] = _WIN
                    continue
                if self.dead(eng):
                    continue
                nkey = self._key(eng)
                if nkey == key:
                    continue                 # a cancelled turn: a self-loop
                edges[direction] = nkey
                if nkey not in snaps:
                    snaps[nkey] = snapshot(eng)
                    queue.append(nkey)
                    if len(snaps) > self.node_cap:
                        restore(eng, root)
                        raise RuntimeError(
                            f"bubblegoban: level {level} has more than "
                            f"{self.node_cap} reachable states")
        restore(eng, root)

        # Backward BFS from the win. Every press costs 1, so plain BFS over the
        # reversed edges gives the exact distance-to-win.
        preds: dict[frozenset, list] = {}
        for key, edges in succ.items():
            for nkey in edges.values():
                if nkey is not _WIN:
                    preds.setdefault(nkey, []).append(key)
        dist: dict[frozenset, int] = {}
        frontier = [k for k, edges in succ.items()
                    if any(n is _WIN for n in edges.values())]
        for key in frontier:
            dist[key] = 1
        while frontier:
            nxt = []
            for key in frontier:
                for pkey in preds.get(key, ()):
                    if pkey not in dist:
                        dist[pkey] = dist[key] + 1
                        nxt.append(pkey)
            frontier = nxt
        return {key: (dist.get(key), edges) for key, edges in succ.items()}

    def _field(self, eng, level: int | None) -> dict:
        """The field for ``level``, built on first use and extended if a state
        outside it ever turns up.

        The extension branch is belt-and-braces: `record_level` only ever plans
        from the level's start (the RESET-recovery prefix rewinds to exactly that
        state), so in practice the field is built once and hit forever."""
        field = self._fields.get(level)
        if field is None:
            field = self._fields[level] = self._build_field(eng, level)
        elif self._key(eng) not in field:
            field = self._fields[level] = self._build_field(eng, level)
        return field

    # -- planning --------------------------------------------------------------
    def _descend(self, field: dict, key) -> Plan | None:
        """Walk the field downhill from ``key`` to the win: the shortest press
        sequence, plus the full optimal SET at every step along it. Pure field
        arithmetic -- it does not step the interpreter."""
        entry = field.get(key)
        if entry is None or entry[0] is None:
            return None
        presses, optsets = [], []
        while True:
            edges = entry[1]
            best = self._best_dirs(field, entry)
            optsets.append(best)
            step = best[0]
            presses.append(step)
            nxt = edges[step]
            if nxt is _WIN:
                return Plan(presses, optsets)
            entry = field[nxt]

    def _best_dirs(self, field: dict, entry) -> list:
        """Every press from ``entry`` that stays on a shortest route."""
        dist, edges = entry
        best = []
        for direction in self.directions:
            nxt = edges.get(direction)
            if nxt is None:
                continue
            ndist = 0 if nxt is _WIN else field[nxt][0]
            if ndist is not None and ndist == dist - 1:
                best.append(direction)
        return best

    def plan(self, eng, level: int | None = None) -> Plan | list | None:
        """The shortest winning press sequence from the engine's current state
        (a `Plan`, carrying its optimal sets), or None if there is none. Leaves
        the engine unchanged.

        The level's START plan is also kept on disk: it is the only state any
        seed ever plans from -- recovery is a RESET back to it -- and rebuilding
        the field for it costs minutes of interpreter steps in every process."""
        if eng.check_win():
            return []
        sig = sorted([r, c, o] for (r, c, o) in self._key(eng))
        cached = self._disk.get(level)
        if cached is not None and cached["start"] == sig:
            return (Plan(cached["plan"], cached["optsets"])
                    if cached["plan"] is not None else None)
        found = self._descend(self._field(eng, level), self._key(eng))
        if level is not None and cached is None:
            self._disk[level] = {
                "start": sig,
                "plan": None if found is None else list(found),
                "optsets": None if found is None else found.optsets,
            }
            self._save_disk()
        return found

    def optimal_dirs(self, eng, level: int | None = None) -> list:
        """Every press that keeps the level on a SHORTEST route to the win, read
        off the live engine state. Empty when the state is unwinnable.

        Only used when the plan's own ``optsets`` cannot answer (an off-plan
        state), because it needs the field -- which a cached run has not built."""
        if eng.check_win():
            return []
        entry = self._field(eng, level).get(self._key(eng))
        if entry is None or entry[0] is None:
            return []
        return self._best_dirs(self._fields[level], entry)

    # -- disk cache ------------------------------------------------------------
    @staticmethod
    def _load_disk() -> dict:
        """``{level: {"start": sig, "plan": [...] | None, "optsets": [...]}}``,
        or empty if unreadable -- an unparseable cache is a miss, never a
        crash."""
        try:
            raw = json.loads(PLAN_CACHE.read_text())
            return {int(k): v for k, v in raw.items()}
        except Exception:                                    # noqa: BLE001
            return {}

    def _save_disk(self) -> None:
        """Write atomically -- shards started together would otherwise interleave
        into a truncated file (see `parallelize_generator`)."""
        try:
            PLAN_CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = PLAN_CACHE.with_suffix(f".json.tmp{os.getpid()}")
            tmp.write_text(json.dumps(
                {str(k): self._disk[k] for k in sorted(self._disk)}, indent=1))
            os.replace(tmp, PLAN_CACHE)
        except OSError:
            pass                                             # cache is optional


class BubblegobanSolver(PSAStarSolver):
    game_id = "puzzlescript_bubblegoban"
    game_name = GAME_NAME
    expert_cls = BubblegobanExpert

    #: Cap on FIELD states per level (the expert plans by exhaustive BFS, not by
    #: A*, so this bounds memory rather than search effort). The shipped levels
    #: peak at a few thousand; anything near this ceiling is a level that wants
    #: redesigning, so it raises instead of silently returning no plan.
    node_cap = 200_000
    #: Plans are 4-50 presses; the ceiling only has to cover a re-plan, and this
    #: family runs `epsilon` at 0.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Build every level's field before `discover_solvable` asks for it --
        the same work either way, but it makes the one-time cost visible as
        startup rather than as a mysteriously slow first seed."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)

    def optimal_for(self, expert, level: int, plan: list, pi: int):
        """The full set of equally-shortest presses at this step.

        The field makes this EXACT rather than a guess, and the ties are real:
        crossing open ground costs the same in any interleaving of the two axes,
        so labelling one of them as "the" answer would train against the truth.
        The plan carries its own sets (so a disk-cached run needs no field), and
        the live state answers when a step is somehow off-plan. Falls back to the
        press about to be taken, so no expert step ever ships unlabelled (see the
        always-emit-optimal-targets rule)."""
        sets = getattr(plan, "optsets", None)
        if sets is not None and pi < len(sets):
            return sets[pi]
        best = expert.optimal_dirs(expert.game._engine, level)
        if best:
            return best
        return [plan[pi]] if pi < len(plan) else None


# ---------------------------------------------------------------------------
# Mechanic self-check
# ---------------------------------------------------------------------------

def selfcheck(trials: int = 60, steps: int = 40, verbose: bool = True) -> int:
    """Audit the two claims the whole solver rests on, against the interpreter.

      1. `BubblegobanExpert.dead` is SOUND: in a state it calls dead, all five
         keys (including ACTION) really are no-ops, so the board is frozen.
      2. ACTION is a no-op in EVERY state, dead or not -- which is what lets the
         search drop it from `directions`.

    Random rollouts from every level start; returns the number of violations."""
    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    expert = BubblegobanExpert(game)
    eng = game._engine
    bad = 0
    for level in range(game.n_levels):
        dead_seen = 0
        for t in range(trials):
            game.set_level(level)
            rng = random.Random(f"bubblegoban:selfcheck:{level}:{t}")
            for _ in range(steps):
                before = snapshot(eng)
                eng.step("action")
                if eng.grid != before or eng.check_win():
                    bad += 1
                    print(f"  L{level}: ACTION was not a no-op")
                    restore(eng, before)
                if expert.dead(eng):
                    dead_seen += 1
                    for direction in ("up", "down", "left", "right", "action"):
                        eng.step(direction)
                        if eng.grid != before:
                            bad += 1
                            print(f"  L{level}: dead() state moved on "
                                  f"{direction}")
                            restore(eng, before)
                    break
                eng.step(rng.choice(expert.directions))
                if eng.check_win():
                    break
        if verbose:
            print(f"  L{level}: {'OK' if not bad else 'VIOLATIONS'} "
                  f"({trials} rollouts, {dead_seen} reached a dead board)")
    return bad


def _plan_report() -> None:
    """Print the shortest plan and the field size for every level -- the quick
    "is this game still fully solved" check, and the level-design loop."""
    solver = BubblegobanSolver()
    game, expert, solvable = solver._ensure(0)
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        plan = expert.plan(eng, level)
        field = expert._fields.get(level, {})
        live = sum(1 for dist, _ in field.values() if dist is not None)
        size = (f"{len(field):5d} states ({live} winnable)" if field
                else "        cached          ")
        if plan is None:
            print(f"  L{level}: UNSOLVED   {size}")
            continue
        for direction in plan:                      # engine-verify the plan
            eng.step(direction)
        ties = sum(len(s) - 1 for s in plan.optsets)
        print(f"  L{level}: {len(plan):3d} presses  win={eng.check_win()}  "
              f"{size}  {ties} tie-presses  {' '.join(d[0] for d in plan)}")
    print(f"  solvable levels: {solvable}")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        violations = selfcheck()
        print(f"selfcheck: {violations} violations")
        sys.exit(1 if violations else 0)
    if "--plans" in sys.argv:
        _plan_report()
        sys.exit(0)
    sys.exit(BubblegobanSolver.main())
