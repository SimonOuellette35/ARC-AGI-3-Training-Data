"""Generate Phase-1 training data for the PuzzleScript game
ps:bouphas_candle_quest ("Boupha's Candle Quest").

The harness -- the engine-blackbox A* expert, the rotation contract, the
trajectory recorder and the BaseSolver plumbing -- lives in
`solvers/common/ps_astar.py`, shared with the other search-the-interpreter ps:
generators. This file is only the game-specific part: its name and its goal
heuristic.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_bouphas_candle_quest",
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
Collect every Candle, then stand on the Rocket (``No Candle`` + ``All Player On
Rocket`` + ``All Rocket On Player``). A candle is picked up by walking into it,
and that is the only part of the game that is simple. Twelve levels bolt on six
independent mechanics, all of them direction-agnostic (no rule carries an
explicit ``up``/``down``/``left``/``right``, so the whole game is isotropic):

  * **Pumpkin** -- ``[> Player | Pumpkin] -> [> Player | > Pumpkin]``. Ordinary
    sokoban push.
  * **Unpumpkin** -- ``[< Player | Unpumpkin] -> [< Player | < Unpumpkin]``. The
    mirror image: the player DRAGS it. It moves only when the player is adjacent
    and steps directly AWAY from it, into the cell the player just left -- so a
    pull can never be blocked, and a piece can be walked out of a pocket a push
    could never empty.
  * **Switch / Door / AntiDoor** -- a switch is held down by any ``Weight``
    (player, any *kin, zombie, brain). The two door rules run in order (open
    every Door if SOME switch is weighted, then close every OpenDoor if SOME
    switch is not), so the stable reading is: **doors are open exactly when ALL
    switches are weighted, and antidoors are open exactly when they are not.**
    The player is itself a weight, which is the trap the puzzles are built on --
    standing on the last switch opens the door and stepping off it closes the
    door again, so a *kin has to be parked there instead. (Antidoors are guarded
    with ``No Weight`` and so never close on the player; doors are not.)
  * **ACTION5 (Gourdslay)** -- ``[Action Player | *kin] -> [Player | ]`` smashes
    EVERY adjacent pumpkin/unpumpkin/qumpkin at once. Irreversible, and levels 5
    and 6 need it (a *kin parked in the only corridor).
  * **Qumpkin** -- ``late [Player |...| Qumpkin] -> [Qumpkin |...| Temp]``
    teleport-swaps the player with any qumpkin sharing its row or column. It is
    ``late`` and unconditional, so on level 11 it fires whether or not the player
    wants it: a step that lines up with a qumpkin ends somewhere else entirely.
  * **Zombie / Skull / Brain** -- ``late [Monster | Player] -> [Monster | Dead]``
    kills the player on contact, and zombies home in on any Brain or Player
    sharing a row or column (``|...|`` matches across walls, so their line of
    sight ignores the geometry even though their movement does not). A zombie is
    also a Weight, which is the only way level 10's switches can be held down.

Note the win condition's asymmetry: with the player dead there are no Player
objects left, so ``All Player On Rocket`` is vacuously true while ``All Rocket On
Player`` still fails -- death is a permanent loss, never a spurious win. That is
what `CandleQuestExpert.dead` prunes.

Expert solver
-------------
`PSExpert`: A* over the REAL interpreter, branching on the five primitive
presses. Unlike the sokoban-shaped ps: games there is no macro worth naming here
-- pushes, pulls, candle pickups, smashes, teleports and the zombies' reaction to
where the player stands are all different kinds of move, and the walk between
them is itself state-changing (every step advances the zombies and re-evaluates
the doors). Searching primitives keeps the model of the game at exactly zero
lines.

The heuristic is two terms:

  * **Travel.** The remaining route is "visit every candle, end on the rocket",
    so charge the MST of the metric closure over {player} + candles + rocket
    (walls only, doors optimistically open). A path visiting all of those nodes
    is a spanning tree of them, so its MST is a genuine lower bound -- and unlike
    "distance to the nearest candle" it does not collapse to nothing when three
    candles sit in three different corners, which is most of the boards here.
  * **Switches, but only when they are load-bearing.** A closed door is invisible
    to the travel term (it plans through it), so on the switch levels the
    heuristic is flat across the whole pumpkin-shuffling middlegame and A* has
    nothing to steer with. The fix is to charge for the switches -- but ONLY when
    the current door state actually blocks the goal, which is a reachability BFS
    over the doors as they stand right now. Levels 7 and 10 are why the condition
    matters: their antidoors are open BECAUSE the switches are bare, so a term
    that always pushed toward weighting switches would be pushing the wrong way.
    When the goal is blocked, charge ``2 x`` the distance from the nearest
    movable *kin to each unweighted switch: a piece-step costs the player one
    move to make plus roughly one to reposition for the next.

That second term is a guide, not a bound -- so plans are winning and engine-
verified but not certified shortest. Measured against the travel-only ablation it
costs nothing and buys a lot: on the ten levels travel-only can also solve, both
return the SAME plan length (12/27/34/28/20/22/48/19/20/24 moves), it turns the
other two from "no plan inside 200k nodes" into 27s and 41s, and it cuts the ones
travel-only does reach by an order of magnitude (level 2 42.7s -> 3.7s, level 4
52.9s -> 4.5s, level 6 23.2s -> 1.7s). Whole-corpus search: 429s and 10/12 levels
without it, 89s and 12/12 with it.

All twelve levels solve, in ~89s of search at seed 0 (levels 3 and 10 are
three-quarters of it), 12 to 111 moves each. That is the whole cost of the
corpus: every later seed replays the cached plans at ~1.5s per episode.

Augmentation
------------
The engine state after reset is identical for every seed (levels are fixed ASCII
maps), so the only per-(seed, level) variables are the presentation
augmentations: the frame rotation (rotation_k in {0,1,2,3}) plus an independent
horizontal and vertical flip, each with the matching directional action remap
(`PuzzleScriptAdapter._FLIP_GAMES`), which together re-sample the board's
8-element symmetry group. Every rule in this game is direction-agnostic and the
input is screen-relative, so a flip is an exact symmetry of the mechanic; the
only asymmetric sprites (the player's head-over-legs, the candle's flame) are
decoration, and the rotation augmentation already permutes their orientation.
There is deliberately no recolor: with eighteen object classes sharing a small
palette (Wall and Pumpkin are already both Black+Brown/Orange, Switch and Door
are both Blue+LightBlue, distinguished only by their sprite shape) a recolor has
nowhere safe to move.

The expert plan is therefore seed-independent: solved once per level, cached, and
replayed per seed with that seed's remapped screen actions. Seed 0 pays for all
twelve searches; every later seed replays them from cache.

Usage (run from the repo root):
    python solvers/generate_bouphas_candle_quest_training.py --episodes 200 \
        --out data/training_multi_level/bouphas_candle_quest
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from solvers.common.ps_astar import PSAStarSolver, PSExpert  # noqa: E402

GAME_NAME = "Boupha's_Candle_Quest"

#: (dr, dc) per grid direction, for the walk BFS.
_DELTA = ((-1, 0), (1, 0), (0, -1), (0, 1))

#: Stand-in distance for "unreachable even with every door open". Finite so a
#: board the walls really do cut in two still gets ordered sensibly rather than
#: overflowing into the dead-state band.
_FAR = 300

#: Player moves charged per piece-step in the switch term: one to make the
#: push/pull, roughly one more to walk around for the next.
_PIECE_COST = 2


class CandleQuestExpert(PSExpert):
    """Primitive A* over the real interpreter, guided by a travel MST plus a
    switch term that only fires when a door is actually in the way. See the
    module docstring for both terms and for why the second one is conditional."""

    def setup(self) -> None:
        idx = self.g.obj_name_to_idx
        self.wall = idx["wall"]
        self.candle = idx["candle"]
        self.rocket = idx["rocket"]
        self.switch = idx["switch"]
        self.player_ids = set(self.game._engine._player_indices)
        # Anything that holds a switch down (the LEGEND's ``Weight`` group).
        self.weight_ids = {idx[n] for n in ("player", "pumpkin", "unpumpkin",
                                            "zombie", "qumpkin", "brain")}
        # ...of which only the *kin can be relocated onto a switch. A brain is
        # scenery until a zombie eats it and a zombie goes where it likes.
        self.movable_ids = {idx[n] for n in ("pumpkin", "unpumpkin", "qumpkin")}
        # Shut doors. The open forms live on their own collision layer and are
        # walked over freely, so only these two ever block anybody.
        self.shut_ids = {idx["door"], idx["antidoor"]}
        self._free: list[list[bool]] = []
        self._dist: dict = {}

    # -- per-board setup -------------------------------------------------------
    def plan(self, eng, level: int | None = None):
        """Bind this board's static geometry, then plan as usual.

        Walls and the rocket are the only things no rule in this game can move,
        and they are what the walk BFS runs over -- so they are read ONCE here
        rather than inside `heuristic`, which runs on every node."""
        self._prep(eng)
        return super().plan(eng, level)

    def _prep(self, eng) -> None:
        grid = eng.grid
        self.h, self.w = len(grid), len(grid[0])
        self._free = [[self.wall not in cell for cell in row] for row in grid]
        self._rocks = [(r, c)
                       for r, row in enumerate(grid)
                       for c, cell in enumerate(row)
                       if self.rocket in cell]
        self._dist = {}

    def _bfs(self, src, free=None) -> dict:
        """Walk distances from ``src``. With no ``free`` map this is the
        doors-optimistic geometry, which is static per board and so memoized;
        callers that pass their own map (the door-state reachability test) get an
        uncached run, since that map changes every node."""
        cached = free is None
        if cached:
            hit = self._dist.get(src)
            if hit is not None:
                return hit
            free = self._free
        dist = {src: 0}
        queue = deque([src])
        while queue:
            r, c = queue.popleft()
            step = dist[(r, c)] + 1
            for dr, dc in _DELTA:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.h and 0 <= nc < self.w and free[nr][nc]
                        and (nr, nc) not in dist):
                    dist[(nr, nc)] = step
                    queue.append((nr, nc))
        if cached:
            self._dist[src] = dist
        return dist

    # -- state read ------------------------------------------------------------
    def _scan(self, eng) -> tuple:
        """One pass over the grid for everything both terms need."""
        player = None
        candles: list[tuple[int, int]] = []
        bare: list[tuple[int, int]] = []      # switches with nothing on them
        movable: list[tuple[int, int]] = []
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                if not cell:
                    continue
                if cell & self.player_ids:
                    player = (r, c)
                if self.candle in cell:
                    candles.append((r, c))
                if self.switch in cell and not cell & self.weight_ids:
                    bare.append((r, c))
                if cell & self.movable_ids:
                    movable.append((r, c))
        return player, candles, bare, movable

    def dead(self, eng) -> bool:
        """A monster ate the player: the Player object is gone for good and only
        a Dead corpse is left, so no future reaches the win. Worth its own scan --
        on the zombie levels most branches end here."""
        for row in eng.grid:
            for cell in row:
                if cell & self.player_ids:
                    return False
        return True

    # -- heuristic -------------------------------------------------------------
    def heuristic(self, eng) -> int:
        player, candles, bare, movable = self._scan(eng)
        if player is None:                     # only reachable via `plan`, not A*
            return _FAR * 10
        travel = self._travel(player, candles)
        if not bare or not self._blocked(eng, player, candles):
            return travel
        return travel + _PIECE_COST * self._switch_cost(bare, movable)

    def _travel(self, player, candles) -> int:
        """Lower bound on "visit every candle, end on the rocket": the MST of the
        metric closure over those nodes plus the player (a route through them all
        IS a spanning tree of them). With the candles gone it degenerates to the
        walk to the rocket, which is then the whole remaining game."""
        if not candles:
            dist = self._bfs(player)
            return min((dist.get(x, _FAR) for x in self._rocks), default=0)
        nodes = [player] + candles + self._rocks
        pair = {}
        for a in nodes:
            da = self._bfs(a)
            for b in nodes:
                pair[(a, b)] = da.get(b, _FAR)
        joined = {nodes[0]}
        rest = set(nodes[1:])
        total = 0
        while rest:
            cost, nxt = min((pair[(a, b)], b) for a in joined for b in rest)
            total += cost
            joined.add(nxt)
            rest.discard(nxt)
        return total

    def _blocked(self, eng, player, candles) -> bool:
        """True when some remaining goal is unreachable through the doors AS THEY
        STAND -- i.e. when the switches are load-bearing right now. This is the
        guard that keeps the switch term from firing on the levels whose antidoors
        want the switches left bare."""
        grid = eng.grid
        free = [[self._free[r][c] and not (grid[r][c] & self.shut_ids)
                 for c in range(self.w)]
                for r in range(self.h)]
        reach = self._bfs(player, free)
        return any(goal not in reach for goal in candles + self._rocks)

    def _switch_cost(self, bare, movable) -> int:
        """Piece-steps to hold every bare switch down: each one charged its
        nearest movable *kin, with reuse allowed. Reuse keeps it cheap and keeps
        it from over-charging a board with one pumpkin and two switches into
        looking hopeless -- it is a direction to walk in, not a bound."""
        total = 0
        for cell in bare:
            dist = self._bfs(cell)
            total += min((dist.get(m, _FAR) for m in movable), default=_FAR)
        return total


class BouphasCandleQuestSolver(PSAStarSolver):
    game_id = "puzzlescript_bouphas_candle_quest"
    game_name = GAME_NAME
    expert_cls = CandleQuestExpert

    #: Unweighted: every level falls out at w=1, the two slow ones in 27s and
    #: 41s, so there is nothing to buy by trading plan length away.
    weight = 1
    #: Level 3 (four unpumpkins onto four switches) is the deepest search; the
    #: cap is a runaway backstop, not a budget anything here comes near.
    node_cap = 400_000
    #: Level 10's plan is 111 moves -- the longest by a factor of two.
    max_steps = 300


if __name__ == "__main__":
    sys.exit(BouphasCandleQuestSolver.main())
