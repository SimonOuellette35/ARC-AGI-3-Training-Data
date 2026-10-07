"""Generate Phase-1 training data for the PuzzleScript game ps:blockfaker
(Droqen's "Block Faker" -- a match-3 crossed with a push puzzle).

The harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: its name, the macro search
it uses, its goal heuristic and its disk plan cache.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_blockfaker",
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
Walk the black player onto the EndPoint. Four rules do all the work:

    [ > Moveable | Moveable ] -> [ > Moveable | > Moveable ]   (Moveable = Player or Block)
    [ > Block | Grille ]      -> [ Block | Grille ]
    late [ XBlock | XBlock | XBlock ] -> [ | | ]               (one per colour)

so:

  * **Pushes are CHAIN pushes.** Walking into a block shoves the whole run of
    blocks behind it, and the turn only happens if the far end of that run lands
    on an empty cell. Blocks and walls share the player's collision layer, so a
    refused push is a cancelled turn (``require_player_movement``) -- a literal
    no-op, not a partial move.
  * **Grilles are one-way filters.** The player walks over a Grille freely (it is
    on its own layer); a block moving *into* one has its force cancelled, which
    cancels the whole push. Two blocks with a grille behind them and a wall in
    front are therefore FROZEN FOREVER, and level 3 is built almost entirely out
    of that trick: of its 14 green blocks only 3 can ever move.
  * **Three of a colour in a line vanish**, horizontally or vertically, checked
    after every turn. That is the only way to remove a block, and on levels 2 and
    3 it is the only way through: the corridor to the EndPoint is plugged by
    blocks that cannot be pushed anywhere useful, and the solution is to walk a
    third block of the same colour across the board to line them up.

Blocks may sit ON the EndPoint (separate collision layer), which is how a level
is lost rather than won -- shove the wrong block onto the goal and it can no
longer be stood on. The search sees that as a plain dead end.

Palette fix (see the ps: palette-collision note)
------------------------------------------------
``EndPoint`` was ``green`` and ``GreenBlock`` was ``lightgreen``; the ARC palette
has exactly ONE green, so both quantised to index 14. GreenBlock's sprite is a
filled square with a single transparent pixel in the middle, which the 4 px cell
downsample erases -- so on levels 2-4 the goal was pixel-for-pixel identical to
the eleven green blocks scattered around it. ``data/puzzlescript_games/blockfaker.txt``
now paints EndPoint ``yellow`` (11, used by nothing else in the game), leaving the
five block colours as shipped: blue 9, pink 8, purple 15, orange 12, green 14,
against white background 0, gray walls/grilles 2 and the black player 5.

Expert solver
-------------
`PSPushExpert`: A* over the REAL engine dynamics whose successors are
``walk to the push cell, then push`` MACROS rather than single key presses,
costed in primitive MOVES. Two game-specific changes on top of it:

  * **Push-legality prefilter** (`_analyze`). The base enumerates a macro per
    (piece, direction) and lets the engine reject the illegal ones, which here is
    most of them -- a board is mostly frozen blocks, and every refused push still
    costs a walk's worth of interpreter steps before it is thrown away. Scanning
    the run of blocks ahead of the piece and requiring its far end to be an
    on-board, non-wall, non-grille cell reproduces the engine's decision exactly
    and drops those branches for free. On level 3 it cuts 14 pieces down to the
    3 greens and 3 oranges that can actually move.
  * **Teleport walks** (`_apply_macro`). No rule in this game reacts to the player
    moving through empty space, so the walk part of a macro is a pure relocation
    of the player object; only the final push has to go through `eng.step`. That
    turns a macro from ~8 interpreter turns into 1. This is the documented
    `_apply_macro` override contract ("must leave the engine in exactly the state
    the primitive replay would have") and it is fuzz-verified: 400 random macros
    per level, teleported vs primitively replayed, produced identical grids and
    identical win-truncation indices on all five levels.

The heuristic is a Dijkstra from the EndPoint back to the player over non-wall
cells where stepping ONTO a block costs ``block_cost`` extra -- "how far away the
goal is, counting each block in the way as work". Plain walk distance is useless
here (the player is usually a few cells from a goal it cannot reach), and
counting blocks alone is flat; the mix ranks "one block to clear" above "two"
while still preferring the short way round. ``block_cost = 4`` was picked by
measurement: 10 and a weighted (w=2) A* both did WORSE on levels 2 and 3, which
is the usual sign that an inflated heuristic is mis-ranking states whose blocking
pieces are about to disappear anyway.

All five levels solve, in 26 / 24 / 65 / 68 / 14 moves. Levels 2 and 3 are the
expensive ones (~5.5 and ~4 minutes of one-time search, ~9 minutes for the whole
board); everything else is under a second.

Plan cache
----------
Engine state after a reset is seed-independent (levels are fixed ASCII maps; only
the PRESENTATION is augmented per seed), so a level is searched once and every
later seed replays the same plan with its own rotation/flip remap. Because levels
2-3 cost minutes, that cache is also written to ``data/blockfaker_plans.json`` so
a fresh process -- in particular each shard of `parallelize_generator.py`, which
would otherwise repeat the whole 9 minutes per core -- starts warm. Delete that
file to force a fresh search.

Augmentation
------------
Rotation (k in 0..3) plus independent horizontal and vertical flips, each with
the matching directional action remap (`PuzzleScriptAdapter._FLIP_GAMES`, which
this game is added to), re-sampling the board's 8-element symmetry group over
only five levels. Legal here for the same reason as Bad Example: no gravity, no
axis-sensitive rule (the match-3 rules expand to both axes, the grille rule to
all four directions), screen-relative input, and no sprite that encodes a
direction. There is no colour augmentation -- which colour a block is IS the
mechanic.

Usage (run from the repo root):
    python solvers/generate_blockfaker_training.py --episodes 200 \
        --out data/training_multi_level/blockfaker
"""

from __future__ import annotations

import heapq
import json
import os
import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSPushExpert  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_NAME = "blockfaker"

#: Where the seed-independent per-level plans are kept between processes.
PLAN_CACHE: Path = _REPO_ROOT / "data" / "blockfaker_plans.json"

#: Engine direction -> (dr, dc) on the grid.
_DELTA: dict[str, tuple[int, int]] = {
    "up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1),
}

#: Charge for a state with no player or no EndPoint left on the board (a block
#: was shoved onto the goal, or a rule ate the player). Large enough to sink the
#: node, finite so the search stays complete.
_DEAD = 500


class BlockFakerExpert(PSPushExpert):
    """Walk-and-push A* over the real interpreter, guided by a block-penalised
    distance from the EndPoint. See the module docstring for the prefilter, the
    teleported walks and why ``block_cost`` is 4."""

    pushable_names = ("block",)
    blocker_names = ("wall", "wallblock")

    #: Extra heuristic charge for a path cell occupied by a block.
    block_cost = 4

    def setup(self) -> None:
        super().setup()
        g = self.g
        self.endpoint = g.obj_name_to_idx["endpoint"]
        self.grille = g.obj_name_to_idx["grille"]
        self.player = next(iter(self.player_ids))
        self._disk = self._load_disk()

    # -- heuristic -----------------------------------------------------------
    def heuristic(self, eng) -> int:
        """Cheapest route from the EndPoint back to the player, charging
        ``block_cost`` extra for every block cell it has to pass through.

        Recomputed per node rather than cached per layout: the blocks ARE the
        state here, so a table keyed on the static walls would answer the same
        thing for every board."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        goal = player = None
        for r in range(h):
            row = grid[r]
            for c in range(w):
                cell = row[c]
                if not cell:
                    continue
                if self.endpoint in cell:
                    goal = (r, c)
                if self.player in cell:
                    player = (r, c)
        if goal is None or player is None:
            return _DEAD

        penalty = self.block_cost
        dist = {goal: 0}
        pq = [(0, goal)]
        while pq:
            d, cur = heapq.heappop(pq)
            if cur == player:
                return d
            if d > dist.get(cur, 1 << 30):
                continue
            r, c = cur
            for dr, dc in _DELTA.values():
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                cell = grid[nr][nc]
                if cell & self.blocker_ids:
                    continue
                nd = d + 1 + (penalty if (cell & self.push_ids) else 0)
                if nd < dist.get((nr, nc), 1 << 30):
                    dist[(nr, nc)] = nd
                    heapq.heappush(pq, (nd, (nr, nc)))
        return _DEAD                     # the goal is walled off from the player

    # -- macro generation ----------------------------------------------------
    def _analyze(self, eng) -> tuple:
        """``(region, macros)`` -- the base's walk BFS, but emitting only pushes
        the engine will actually perform, plus the walk onto the EndPoint.

        A push moves the whole RUN of blocks ahead of the piece, and happens only
        if that run ends on an on-board cell that is neither a wall nor a grille
        (``[ > Block | Grille ]`` cancels the block's force, and a block that
        cannot move cancels the player's move with it). Checking that here rather
        than letting the engine refuse it is what makes the mostly-frozen boards
        cheap: a refused push still costs a walk's worth of interpreter steps.

        The EndPoint walk is the reason this game needs a macro the base does not
        generate: the win is the PLAYER standing somewhere, not a push outcome, so
        a search branching only on pushes closes with the goal never entered."""
        grid = eng.grid
        h, w = len(grid), len(grid[0])
        push_ids, blocker_ids = self.push_ids, self.blocker_ids
        player = None
        pieces: list[tuple[int, int]] = []
        free = [[True] * w for _ in range(h)]
        for r in range(h):
            row = grid[r]
            for c in range(w):
                cell = row[c]
                if not cell:
                    continue
                if self.player in cell:
                    player = (r, c)
                if cell & push_ids:
                    free[r][c] = False
                    pieces.append((r, c))
                elif cell & blocker_ids:
                    free[r][c] = False
        if player is None:
            return None, []

        # Shortest walks from the player, kept as parent pointers and
        # materialised only for the cells a macro actually uses.
        parent: dict[tuple[int, int], tuple | None] = {player: None}
        queue = deque([player])
        while queue:
            r, c = queue.popleft()
            for d, (dr, dc) in _DELTA.items():
                nxt = (r + dr, c + dc)
                if (0 <= nxt[0] < h and 0 <= nxt[1] < w
                        and free[nxt[0]][nxt[1]] and nxt not in parent):
                    parent[nxt] = ((r, c), d)
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
                if stand not in parent:
                    continue                  # cannot get behind the piece
                er, ec = pr + dr, pc + dc     # walk off the end of the chain
                while 0 <= er < h and 0 <= ec < w and grid[er][ec] & push_ids:
                    er, ec = er + dr, ec + dc
                if not (0 <= er < h and 0 <= ec < w):
                    continue                  # chain runs off the board
                dest = grid[er][ec]
                if (dest & blocker_ids) or self.grille in dest:
                    continue                  # the engine would cancel the turn
                macros.append(walk_to(stand) + [d])

        for r in range(h):
            for c in range(w):
                if self.endpoint in grid[r][c] and (r, c) in parent:
                    macros.append(walk_to((r, c)))
        return min(parent), macros

    # -- macro execution -----------------------------------------------------
    def _apply_macro(self, eng, macro) -> int:
        """Run one macro, TELEPORTING the walk and stepping only the push.

        Nothing in this game reacts to the player crossing empty space: the push
        rule needs a Moveable in front, the grille rule needs a moving Block, and
        the match-3 rules see a block layout the walk does not touch. So moving
        the player object straight to the stand cell leaves the engine in exactly
        the state the primitive replay would have (fuzz-verified per the module
        docstring) for a fraction of the cost -- and the interpreter step is the
        entire cost of this search.

        Returns the index of the primitive that WON, or -1. A walk step can win
        (the EndPoint macro, or a walk that happens to route across the goal), so
        the win is still checked after every primitive -- but only when the cell
        entered holds the EndPoint, since ``some Player on Endpoint`` cannot
        become true any other way and `check_win` scans the whole board."""
        grid = eng.grid
        pid, push_ids, blocker_ids = self.player, self.push_ids, self.blocker_ids
        pos = None
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                if pid in cell:
                    pos = (r, c)
        if pos is None:
            return -1

        moved = False                    # a teleport the engine has not seen yet

        def _sync():
            eng._position_index_dirty = True
            eng._rule_noop_cache.clear()

        for i, direction in enumerate(macro):
            dr, dc = _DELTA[direction]
            nr, nc = pos[0] + dr, pos[1] + dc
            cell = grid[nr][nc]
            if (cell & push_ids) or (cell & blocker_ids):
                if moved:
                    _sync()
                    moved = False
                eng.step(direction)
                grid = eng.grid          # a cancelled turn rebinds the grid
                pos = None
                for r, row in enumerate(grid):
                    for c, cc in enumerate(row):
                        if pid in cc:
                            pos = (r, c)
                if eng.check_win():
                    return i
                if pos is None:
                    return -1            # a rule consumed the player
            else:
                grid[pos[0]][pos[1]].discard(pid)
                cell.add(pid)
                pos = (nr, nc)
                moved = True
                if self.endpoint in cell and eng.check_win():
                    _sync()
                    return i
        if moved:
            _sync()
        return -1

    # -- disk plan cache -----------------------------------------------------
    def plan(self, eng, level: int | None = None) -> list | None:
        """`PSPushExpert.plan`, with the level's START plan also cached on disk.

        Only that one state is worth keeping: every seed, and every process,
        re-plans from it and from nothing else (recovery is a RESET back to it).
        The first entry stored for a level is therefore its start; a later
        re-plan from a mid-level state never overwrites it."""
        if level is None:
            return super().plan(eng, level)
        sig = sorted([r, c, o] for (r, c, o) in self._key(eng))
        entry = self._disk.get(level)
        if entry is not None:
            if entry["start"] == sig:
                return entry["plan"]
            return super().plan(eng, level)
        found = super().plan(eng, level)
        self._disk[level] = {"start": sig, "plan": found}
        self._save_disk()
        return found

    @staticmethod
    def _load_disk() -> dict:
        """``{level: {"start": sig, "plan": [...] | None}}``, or empty if
        unreadable -- a cache that cannot be parsed is a miss, never a crash."""
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


class BlockFakerSolver(PSAStarSolver):
    game_id = "puzzlescript_blockfaker"
    game_name = GAME_NAME
    expert_cls = BlockFakerExpert

    #: Unweighted: weighting the heuristic measured strictly worse on the two
    #: hard levels (see the module docstring), so there is nothing to buy.
    weight = 1
    #: Level 2 needs ~430k macro expansions; the cap only has to leave headroom
    #: above that, and no level is close to it from the other side.
    node_cap = 600_000
    #: Longest plan is 68 moves; the rest is room for the exploration prefix and
    #: a re-plan after it.
    max_steps = 250


if __name__ == "__main__":
    sys.exit(BlockFakerSolver.main())
