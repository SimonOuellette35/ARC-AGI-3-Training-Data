"""Generate Phase-1 training data for the PuzzleScript game ps:escape
("ESCAPE!", Adrian's dungeon of crates).

The harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: its levels, the "walk
onto the exit" macro the shared push search does not have, the distance
heuristic that guides it, and an exact optimal-action oracle.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_escape",
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
augmented view, so replaying the recorded actions reproduces the recorded
frames exactly. Each expert step also carries the full set of equally-optimal
presses (see "Optimal-action sets" below).

The game
--------
One rule, and it is the Sokoban rule:

    [ > Player | Crate ] -> [ > Player | > Crate ]

but the win condition is ``All Player on Target``, and that one word changes
the whole puzzle. **The crates are not the goal, they are the door.** Nothing
has to end up anywhere; the player has to *get* somewhere, and the crates are
the furniture standing in the corridor. A crate shoved into a corner is a
perfectly good outcome as long as it is not the corner you needed to walk
through -- which is the exact opposite of the deadlock rule every other push
game in this tree is built around.

Wall, Player and Crate share a collision layer, so a crate shoved into a wall
or into a second crate cancels the turn (pushes never chain); Target sits on
its own layer below them. ACTION5 is bound to nothing, so the four directions
are the whole action space (`PSPushExpert.directions` already drops it).

The levels
----------
The shipped file holds 5. That is 80 presentations over the game's 16-fold
augmentation, and its five boards are three cramped mazes plus one 44-crate
field -- enough to memorise, not enough to learn a push from. So the file now
carries 13, eight of them authored here, ordered by expert plan length:

    level  size   crates  plan  push  ties  search   what it is
     0      8x8     7       10    3    20%    0.1s   pockets off one room
     1      9x9     9       11    3    27%    0.2s   crates in the doorways
     2      9x9     9       11    4    36%    1.4s   open floor, four ways
     3      8x11    9       11    4     0%    0.6s   one forced route
     4      9x11   11       12    3    17%    2.0s   two ways in, one clears
     5      6x6     1       13    8     0%    0.0s   SHIPPED -- ONE crate, and
                                                     you shove it eight times
     6     10x5     3       15    6     0%    0.1s   SHIPPED -- a five-wide
                                                     shaft, a crate per landing
     7     10x10   12       15    5     7%    1.0s   the widest board's maze
     8     11x11   14       16    3    44%    0.6s   wide open: 7 of 16 steps
                                                     take either axis
     9     12x12   44       18    9    33%   78.4s   SHIPPED -- the crate
                                                     field. A straight diagonal
                                                     to the exit, half its
                                                     presses pushes
    10      6x7     2       19    2     0%    0.0s   SHIPPED -- the opening
                                                     board: two crates, two
                                                     doors, no slack
    11     11x11   14       24    2     8%    4.1s   the long way round
    12     10x9     7       27   22     0%    1.1s   SHIPPED -- the spiral, and
                                                     the game's real finale:
                                                     22 of 27 presses push

``push`` is how many of the plan's presses move a crate, ``ties`` the share of
steps with more than one optimal press. 202 moves over 13 levels, ~90s of
one-time search (78s of it level 9), written to
``data/escape_plans.json`` so `parallelize_generator`'s shards do not each
re-derive it. Delete the file to re-derive.

The authored boards are open caves rather than corridors, and that is
structural, not taste: **in a one-cell-wide corridor a crate can only ever be
pushed straight ahead**, so it jams at the first wall and takes the level with
it. Every level added here was checked to be walk-UNsolvable (the exit is
unreachable without moving something) and then engine-verified: the ``plan``
column is an A* plan the interpreter actually walked to a WIN.

Expert solver
-------------
`EscapeExpert` -- `PSPushExpert`'s walk-and-push macro A* over the real
interpreter, plus the two things this game needs that a sokoban does not: a
macro that walks the player ONTO the exit (nothing else can win), and a
heuristic that measures the PLAYER's distance to it rather than any crate's
distance to anything. See the class docstring, including why a crate wedged in
a wall pocket is soundly treated as a wall.

``weight = 1`` and the heuristic is admissible, so plans are shortest up to the
one approximation the shared macro search makes (the region-key dedup; see
`PSPushExpert`).

Optimal-action sets
-------------------
Measured, not inferred: at each step the expert re-solves from each successor
and keeps every press that still finishes in the moves remaining. The shared
`PSPushExpert.annotate_walks` cannot do this job here -- it calls a step that
moved a crate forced, which is right in a sokoban and wrong in a game where
pushing a crate aside IS walking. On level 9 that is the difference between
labelling six of eighteen steps honestly and teaching one arbitrary staircase
through a 44-crate field. No step ever ships unlabelled (the
always-emit-optimal-targets rule).

The palette fix
---------------
The original Crate is a solid 5x5 block, so a crate standing on the Target hid
it completely: the one square the win condition names vanished from the frame
whenever anything was parked on it, and the RESET exploration prefix parks
things on it. The crate's four corners are now transparent, so the target's
corner pixels show through -- the same fix, for the same reason, as the crate
in ps:dang_i'm_huge and the exit in ps:dotsnake. Its fill colour was also named
``lightbrown``, which `_color_name_to_arc` does not know and silently resolves
to the fallback index 2; it now says ``gray`` (the same index) on purpose
rather than by luck. No rule, sprite shape or level was touched by that edit.

``--audit`` is the regression test: every cell COMPOSITION the game can show
(floor, wall, target, crate, crate-on-target, player, player-on-target)
rendered at every cell size the 13 boards use, asserted pairwise distinct.

Augmentation
------------
ESCAPE!'s engine state after reset is identical for every seed (levels are
fixed ASCII maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group. The game is gravity-free, its one rule is stated with
the relative ``>`` force, its win condition names no direction and its input is
screen-relative, so a flip is an exact symmetry of the mechanic; no sprite is
directional, so no mirror can show art the game does not own. There is
deliberately no colour augmentation: crate-on-target is read as the target's
corners showing through the crate's transparent ones, which a flattening
recolor would erase. 13 levels x 16 presentations = 208.

The expert plan is therefore seed-independent: solved once per level, cached,
and replayed per seed with that seed's remapped screen actions.

Usage (run from the repo root):
    python solvers/generate_escape_training.py --episodes 200 \
        --out data/training_multi_level/escape

    python solvers/generate_escape_training.py --plans   # level report
    python solvers/generate_escape_training.py --audit   # rendering audit
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import (PSAStarSolver, PSPushExpert,   # noqa: E402
                                     _DELTA)

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "ESCAPE!"

#: Disk cache of each level's start plan and its optimal-action sets. The
#: searches are seed-independent, so without a file on disk every shard
#: `parallelize_generator` starts would re-derive all of them. Delete to
#: re-derive.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "escape_plans.json"


#: Heuristic charge for a state whose exit is walled off behind frozen crates.
#: Large enough to sink the node to the back of the queue, finite so the search
#: stays complete (the frozen test only reads WALLS, so it can never be wrong,
#: but the exit can genuinely be sealed by an early push).
_UNREACHABLE = 10_000


class EscapeExpert(PSPushExpert):
    """Walk-and-push A* over the real interpreter, guided by the player's own
    distance to the exit.

    WHY NOT `PSSokobanExpert`. ESCAPE! looks like a sokoban and is not one: its
    win condition is ``All Player on Target``, so the crates are never the
    goal, only the furniture in the way. There is nothing to match pieces to
    targets, no piece has to end anywhere in particular, and a crate shoved
    into a corner is a perfectly fine outcome as long as it is not standing
    between the player and the exit. So the sokoban heuristic (per-target push
    distance tables + greedy matching) has nothing to measure here, and its
    corner-deadlock test -- the part that makes a dense sokoban board tractable
    -- would be actively wrong: burying crates in corners is usually how you
    clear a path.

    THE MACRO SET. `PSPushExpert` enumerates ``walk to the push cell, then
    push``, which in a sokoban is every move that matters. Here it is every
    move but the winning one: the player also has to be able to simply *walk
    onto the exit*, and no push expresses that. `extra_macros` adds it -- one
    macro per reachable target cell -- and without it the search closes with
    the exit sitting unvisited in the player's own region.

    THE HEURISTIC is the player's shortest walk to the exit over cells that are
    not walls, in primitive moves, which is the unit ``g`` is counted in. It is
    admissible: every real solution moves the player one non-wall cell per
    press, so it can never do better than the wall-only shortest path. Crates
    are ignored because they are not an obstacle in the lower-bound sense --
    the player can cross a crate's cell in a single move by pushing it, so
    charging anything for one would break the bound.

    FROZEN CRATES ARE WALLS, and that is the one place the estimate gets to be
    sharper than "ignore the crates". A crate at ``X`` can be pushed along
    ``d`` only if the player can stand at ``X - d`` and the crate can land on
    ``X + d``; when a WALL forbids that in every direction the crate can never
    move again, by any sequence of any length, so its cell is permanently
    impassable and removing it from the distance graph stays admissible. That
    is not a corner case on these boards -- ESCAPE!'s dungeon is drawn as
    crates wedged into wall pockets, and on level 9's 12x12 field it is what
    stops the estimate from cheerfully routing the player through a solid rank
    of them.

    Only walls are consulted for that test, never other crates: a crate blocked
    by a crate can come free the moment its neighbour is pushed away, and
    calling it frozen would be an unsound prune rather than an inadmissible
    estimate.

    THE OPTIMAL SETS ARE MEASURED (``exact_optsets``), not inferred.
    `PSPushExpert.annotate_walks` labels a step that moved a crate with itself
    ALONE, which is right in a sokoban and wrong here, where a push is
    frequently just how you walk. Level 9 is a field of 44 crates between the
    player and the exit and its solution is nine LEFTs and nine DOWNs -- the
    straight diagonal, half of whose presses shove a crate out of the way. Six
    of those eighteen steps take either axis and still finish in eighteen;
    `annotate_walks` sees a crate move and calls every one of them forced,
    which would train one arbitrary staircase as the only right answer through
    the hardest board in the game. `PSPushExpert.optimal_sets` re-solves from
    each successor instead and keeps every press that still finishes in the
    moves remaining.
    """

    pushable_names = ("crate",)
    blocker_names = ("wall",)

    #: Keep each level's start plan (and its optimal sets) on disk.
    plan_cache_path = PLAN_CACHE

    #: MEASURE the optimal sets (`PSPushExpert.optimal_sets`) instead of
    #: inferring them: in this game a push is frequently just how you walk, so
    #: `annotate_walks`' "a step that moved a crate is forced" is wrong here.
    #: See the class docstring's note below on level 9.
    exact_optsets = True

    def setup(self) -> None:
        super().setup()
        self.target_ids = set(self.g.resolve_object_name("target"))
        self._static: dict = {}

    # -- static per-board facts ----------------------------------------------
    def _statics(self, eng) -> tuple:
        """``(wall, targets, h, w)`` for this board, memoized on the wall map.

        Walls and targets are static in this game -- no rule creates or
        destroys either -- but `heuristic` runs on every node, so re-scanning
        the grid for them per node is the one avoidable cost in the search."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        wall = tuple(tuple(bool(cell & self.blocker_ids) for cell in row)
                     for row in grid)
        got = self._static.get(wall)
        if got is None:
            targets = tuple((r, c)
                            for r, row in enumerate(grid)
                            for c, cell in enumerate(row)
                            if cell & self.target_ids)
            got = (wall, targets, h, w)
            self._static[wall] = got
        return got

    def _frozen(self, eng, wall, h, w) -> set:
        """Crates no push can EVER move, hence permanent walls.

        A push of the crate at ``(r, c)`` along ``d`` needs the player's stand
        cell ``(r, c) - d`` and the crate's landing cell ``(r, c) + d`` both on
        the board and free of wall. If no axis offers that, no rule in this
        game can ever relocate the crate (nothing but a push moves one), so its
        cell is impassable for the rest of the level."""
        out = set()
        grid = eng.grid
        for r in range(h):
            row = grid[r]
            for c in range(w):
                if not (row[c] & self.push_ids):
                    continue
                for dr, dc in _DELTA.values():
                    ar, ac, br, bc = r + dr, c + dc, r - dr, c - dc
                    if (0 <= ar < h and 0 <= ac < w and not wall[ar][ac]
                            and 0 <= br < h and 0 <= bc < w
                            and not wall[br][bc]):
                        break
                else:
                    out.add((r, c))
        return out

    # -- heuristic ------------------------------------------------------------
    def heuristic(self, eng) -> int:
        wall, targets, h, w = self._statics(eng)
        frozen = self._frozen(eng, wall, h, w)
        grid = eng.grid
        player = None
        for r in range(h):
            row = grid[r]
            for c in range(w):
                if row[c] & self.player_ids:
                    player = (r, c)
                    break
            if player is not None:
                break
        if player is None:                      # player consumed by a rule
            return _UNREACHABLE
        goals = set(targets)
        if player in goals:
            return 0

        # Plain BFS: every edge costs one press, so the priority queue a
        # weighted search would need buys nothing.
        dist = {player: 0}
        queue = deque([player])
        while queue:
            cur = queue.popleft()
            d = dist[cur] + 1
            r, c = cur
            for dr, dc in _DELTA.values():
                nxt = (r + dr, c + dc)
                if not (0 <= nxt[0] < h and 0 <= nxt[1] < w):
                    continue
                if wall[nxt[0]][nxt[1]] or nxt in frozen or nxt in dist:
                    continue
                if nxt in goals:
                    return d
                dist[nxt] = d
                queue.append(nxt)
        return _UNREACHABLE

    # -- the winning move -----------------------------------------------------
    def extra_macros(self, eng, parent, walk_to, free) -> list[list[str]]:
        """Walk onto the exit, when the exit is in the player's region.

        The whole win condition, and the one move `PSPushExpert`'s push
        enumeration cannot express. ``walk_to`` returns [] for the player's own
        cell, which would be an empty macro the search would loop on forever;
        that state is a win the search has already returned from, so dropping
        it is not a case, it is a guard."""
        _wall, targets, _h, _w = self._statics(eng)
        walks = [walk_to(t) for t in targets if t in parent]
        return [walk for walk in walks if walk]

class EscapeSolver(PSAStarSolver):
    game_id = "puzzlescript_escape"
    game_name = GAME_NAME
    expert_cls = EscapeExpert

    #: Record against the GAME FOLDER's adapter. `games/ps:escape/ps:escape.py`
    #: is a plain passthrough today, so this is currently the same object a
    #: bare `PuzzleScriptAdapter(GAME_NAME)` would be -- it is set anyway
    #: because that file is what `game_envs` hands a live agent, and a step cap
    #: or sprite patch added to it later would otherwise silently make this
    #: generator tape a game nobody plays (the trap ps:count_mover hit).
    game_module_id = "ps:escape"

    #: Unweighted, and the heuristic is admissible, so plans are shortest (up
    #: to `PSPushExpert`'s region-key dedup). Level 9 is the only board that
    #: costs anything and it is cached to disk, so there is nothing to buy by
    #: trading optimality away.
    weight = 1
    #: Level 9's optimal-set oracle re-solves the crate field ~20 times over;
    #: the cap is a runaway guard, not a tuning dial (no level comes near it).
    node_cap = 2_000_000
    #: The longest plan is 27 moves; the rest is room for the RESET exploration
    #: prefix and a re-plan after it. Stays under the adapter's own 200-step
    #: per-level budget, which would otherwise flip a level to GAME_OVER
    #: mid-plan.
    max_steps = 150

    def prepare_expert(self, game, expert) -> None:
        """Plan every level before `discover_solvable` asks for it -- the same
        work either way, but it makes the one-time cost visible as startup
        rather than as a mysteriously slow first seed, and it fills the disk
        cache in one pass."""
        for level in range(game.n_levels):
            if level in self.skip_levels:
                continue
            game.set_level(level)
            expert.plan(game._engine, level)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def _report() -> int:
    """Per-level board size, crate count, plan length and tie coverage."""
    import time
    game = EscapeSolver().make_game(0)
    expert = EscapeExpert(game, node_cap=EscapeSolver.node_cap)
    crate = game._game.obj_name_to_idx["crate"]
    total = 0
    for level in range(game.n_levels):
        game.set_level(level)
        eng = game._engine
        crates = sum(1 for row in eng.grid for cell in row if crate in cell)
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
              f"{crates:2d} crates, {len(found):3d} moves "
              f"(budget {game._max_steps}, {room}), "
              f"{ties:3d} steps with a tie set "
              f"({ties / max(1, len(found)):3.0%}), {dt:6.1f}s")
    print(f"total {total} moves over {game.n_levels} levels")
    return 0


def _audit() -> int:
    """Assert every cell COMPOSITION is distinct at every cell size in use.

    The bug this exists for is in the original art: the Crate sprite is a solid
    5x5 block, so a crate standing on the Target hid it completely and the one
    square the win condition names disappeared from the frame whenever anything
    was parked on it -- which is a state the exploration prefix reaches on its
    own. The crate's four corners are now transparent, so the target's corner
    pixels show through it. Checked at the cell size each board actually
    renders at, because that detail is one pixel wide at the 12x12 boards."""
    import itertools

    import numpy as np

    from adapters.puzzlescript_adapter import _render_frame

    game = EscapeSolver().make_game(0)
    eng, g = game._engine, game._game
    idx = g.obj_name_to_idx
    comps = {"floor": (), "wall": ("wall",), "target": ("target",),
             "crate": ("crate",), "crate_on_target": ("target", "crate"),
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
    if "--audit" in sys.argv:
        sys.exit(_audit())
    sys.exit(EscapeSolver.main())
