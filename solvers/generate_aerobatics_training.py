"""Generate Phase-1 training data for the PuzzleScript game ps:aerobatics.

Each solved seed yields one multi-level episode JSON in the shared
encoder/dynamics schema:

    {
      "game_id": "puzzlescript_aerobatics",
      "levels": [
        {
          "level_id": 0,
          "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
          "actions":      [a_0, a_1, ..., a_{T-1}], # length T
        },
        ...
      ]
    }

``actions[0]`` is the RESET that produced ``observations[0]``; ``actions[i]`` for
i>=1 is the action that took the agent from ``observations[i-1]`` to
``observations[i]``. The recorded index is the *screen* action (post
rotation/flip remap), i.e. the button an agent presses in the presented view, so
replaying the recorded actions reproduces the recorded frames exactly.

The game
--------
Aerobatics (Mark Richardson) is a stunt-plane puzzle. A plane flies over a
walled course and **never stops**: every action moves it. The four arrow keys do
not steer it directly -- they name the direction you want to be flying, and the
plane performs the corresponding manoeuvre. Writing ``f`` for the plane's
current heading and ``q`` for the pressed direction, one action sweeps the plane
through this cell list (in order) and leaves it as noted:

  * ``q == f``   -- LEVEL FLIGHT. Sweeps ``[p+f]``; ends at ``p+f`` still heading
    ``f``.
  * ``q == -f``  -- LOOP-DE-LOOP. Sweeps ``[p+f, p+2f]``; ends BACK at ``p+f``
    now heading ``-f``. It costs one action to reverse, and it needs two clear
    cells ahead even though it only advances one.
  * ``q ⊥ f``    -- BANKING TURN. Sweeps ``[p+q, p+q+f, p+q+f+q]``; ends at
    ``p+f+2q`` heading ``q``. A turn is three cells of travel, so it needs much
    more room than it looks like on the board.

Touching a Wall anywhere along that sweep destroys the plane (Explosion), and
the level can never be won afterwards -- the win condition includes ``some
Plane``. Flying through a Ring collects it, and the sweep collects rings at
EVERY cell it crosses, not just where it stops -- so a well-aimed turn can take
three rings in one action. The level is won when the last ring is gone:
``no Ring`` and ``no Turn`` (never mid-manoeuvre) and ``some Plane`` (alive).

ACTION5 toggles a "flight plan" HUD that paints the plane's projected path in
green. It moves nothing and cannot affect the win condition, and at these grid
sizes the markers are 1-2 pixel sprites that the 64x64 downsample mostly eats
(the toggle changes 0 pixels on 5 of the 10 levels), so it is a no-op here and
is left out of both the search and the exploration action set. Nothing is lost:
the heading it visualises is plainly readable off the plane sprite itself, which
renders distinctly for all four headings on every level.

The expert: one exact distance field per level
----------------------------------------------
The plane's whole state is ``(cell, heading, which rings are left)`` -- 4 x
(free cells) x 2^R, at most 756 x 4096 on Down Town. That is small enough to
solve EXACTLY and, more importantly, to solve BACKWARDS: `FlightField` builds
``dist[remaining_rings][cell, heading]`` = the minimum number of actions to
finish the level from that state, for EVERY state at once.

Building it backwards rather than searching forwards from the start is what
makes the generator replan-capable ([[solvers-must-support-recovery]]). A
forward A* answers one question -- "how do I win from the start?" -- and is
useless the moment an exploratory action puts the plane somewhere the plan did
not anticipate. The field answers every such question in O(1): `solve_from`
reads the LIVE engine grid (plane cell + sprite heading + surviving rings),
indexes the field, and either returns an optimal continuation or reports the
state dead (crashed, or alive but with a ring it can no longer reach). So the
recorder runs in ``recovery_mode = "replan"`` with perturbation bursts on, and
every explore/burst step is labelled with what the expert would have done.

It also gives `optimal_set_from` for free: the co-optimal actions at a state are
exactly those whose successor's distance is one smaller. In THIS game that turns
out to buy almost nothing -- across the ten optimal routes only 15 of 454 steps
have a tie at all (levels 8 and 9 have none), because a fixed-speed vehicle whose
cheapest turn already costs three cells of travel has essentially one best way
through a tight course. Stochastic-optimal sampling is therefore close to a no-op
here, like sl01 and synthetic_geodesic. It stays on because it is free and
correct, but the per-seed variety in this corpus comes from the presentation
augmentation and from the exploration prefix / perturbation bursts, NOT from
route diversity -- see Augmentation below.

The field construction (see `FlightField`): masks are processed in increasing
order of rings-remaining, so an action that COLLECTS something lands in an
already-finished layer and is read off directly; the actions that collect
nothing stay inside the current layer and are settled by a Bellman-Ford
relaxation over that layer's unit-cost edges. All ten levels build in ~10s
total (Down Town, the 12-ring one, is 6s of it) and the field is cached on the
solver: the engine state after a reset is seed-independent (the levels are fixed
ASCII maps and only the PRESENTATION is augmented), so seed 0 pays for it and
every later seed replays against the same field at its own augmentation.

Optimal lengths, levels 0-9: 8, 10, 41, 129, 38, 45, 42, 39, 27, 75 actions --
optimal, not merely winning, because the field is an exact backward BFS.

`sweep` is a re-implementation of the mechanic, not the interpreter, so it is
differentially tested against the real PuzzleScript engine -- run
``python solvers/generate_aerobatics_training.py --verify-model``: 4000 random
dense-wall / dense-ring boards x random (heading, press) must agree exactly on
crash / final cell / final heading / rings taken. They do. The recorded
trajectories are then driven through the real adapter on top of that, so only a
genuine engine WIN is ever written.

Augmentation
------------
The engine state after a reset is identical for every seed, so the per-(seed,
level) variables are entirely presentational: the frame rotation
(``rotation_k`` in 0..3) plus an independent horizontal and vertical flip. The
flips were added for this game (``PuzzleScriptAdapter._FLIP_GAMES``): Aerobatics
has no axis-sensitive mechanic -- no gravity, no stacking -- and its inputs are
screen-relative, so a mirror is an exact symmetry. It is exact right down to the
art: ``flipud(PlaneU)`` is pixel-identical to ``PlaneD`` and
``fliplr(PlaneL)`` to ``PlaneR`` (likewise the diagonals and the wall-edge
variants), so a flipped frame is itself a frame the game could really have
produced. That takes the presentation group from 4 orientations to 16, which
matters more here than usual: this game gets no colour augmentation, so without
the flips seeds differ ONLY by one of four rotations.

The expert plans in ENGINE space and emits the SCREEN press via
`ps_astar.screen_action`, which inverts the adapter's own forward remap -- see
that module for why this direction is the opposite of the native games.

Usage (run from the repo root):
    python solvers/generate_aerobatics_training.py --episodes 200 \
        --out data/training_multi_level/aerobatics
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from arcengine import ActionInput, GameState                        # noqa: E402
from solvers.base_solver import (                                   # noqa: E402
    Action, BaseSolver, DriveResult, _ID_TO_GAMEACTION)
from solvers.common.ps_astar import (                               # noqa: E402
    _load_game_module, screen_action)

GAME_NAME = "Aerobatics"
_GAME_ID = "ps:aerobatics"

#: The four manoeuvre keys. ACTION5 only toggles the flight-plan HUD -- it moves
#: nothing and cannot affect the win condition -- so it is not a search branch
#: and not an exploration action either.
DIRS = ("up", "down", "left", "right")
FWD = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}

#: Engine object names, resolved once against the parsed game.
_WALL_NAMES = ("wallx", "wallu", "walld", "walll", "wallr",
               "wallul", "wallur", "walldl", "walldr", "wallo")
#: Settled plane sprite -> heading. The eight-way diagonal sprites (planeul,
#: planeur, planedl, planedr) are drawn only WHILE a Turn object exists, and a
#: turn always resolves inside one `PSEngine.step` (its rules chain on ``again``
#: and the win condition itself requires ``no Turn``), so a settled grid never
#: shows one. `AerobaticsSolver._scan` reports them as an unreadable state rather
#: than guessing a heading from one.
_PLANE_HEADING = {"planeu": "up", "planed": "down",
                  "planel": "left", "planer": "right"}

_INF = np.int32(1 << 20)


def sweep(pos: tuple[int, int], heading: str, press: str):
    """One manoeuvre, as pure geometry.

    Returns ``(cells, end_pos, end_heading)`` where ``cells`` is every cell the
    plane passes through IN ORDER -- so the first wall in that list is where it
    explodes, and every ring in it is collected. The start cell is not included
    (the plane is already there).

    See the module docstring for the three cases. Verified against the real
    interpreter by `verify_model`.
    """
    r, c = pos
    fr, fc = FWD[heading]
    if press == heading:                            # level flight
        return [(r + fr, c + fc)], (r + fr, c + fc), heading
    if press == OPP[heading]:                       # loop-de-loop
        return ([(r + fr, c + fc), (r + 2 * fr, c + 2 * fc)],
                (r + fr, c + fc), press)
    qr, qc = FWD[press]                             # banking turn
    a = (r + qr, c + qc)
    b = (r + qr + fr, c + qc + fc)
    end = (r + qr + fr + qr, c + qc + fc + qc)
    return [a, b, end], end, press


class FlightField:
    """Exact backward BFS over ``(cell, heading, rings remaining)`` for one level.

    ``self.dist[mask, node]`` is the minimum number of actions needed to clear
    the level from that state, where ``node`` indexes ``(cell, heading)`` and
    ``mask`` is the bitmask of rings NOT yet collected. ``_INF`` means the state
    is dead -- the plane can still fly, but some ring is unreachable from it
    (every route to it goes through a wall).

    WHY BACKWARD. A forward search answers one question, "how do I win from the
    start", and is stale as soon as anything unexpected happens. The whole field
    answers it from ANY state, which is what makes `solve_from` a true live-state
    replanner and gives the co-optimal tie set for free.

    HOW. Process masks in increasing popcount. For a given ``mask``, an action
    either
      * collects at least one ring still in ``mask`` -> its successor lives in a
        strictly smaller mask, which was finished on an earlier pass, so its cost
        is just ``1 + dist[smaller][succ]``; or
      * collects nothing new -> it stays inside this mask's layer.
    So each layer starts from the best cross-layer exit at every node and is then
    settled by relaxing the layer's own unit-cost edges to a fixpoint
    (Bellman-Ford; the arrays are one layer wide, so this is a handful of numpy
    ops per round). ``mask == 0`` is the goal layer and is 0 everywhere.
    """

    def __init__(self, walls: set, rings: list, height: int, width: int):
        self.walls = frozenset(walls)
        self.rings = list(rings)
        self.height, self.width = height, width

        free = [(r, c) for r in range(height) for c in range(width)
                if (r, c) not in walls]
        self.node_id = {(cell, h): i * 4 + hi
                        for i, cell in enumerate(free)
                        for hi, h in enumerate(DIRS)}
        self.ring_id = {cell: i for i, cell in enumerate(self.rings)}
        n_nodes = len(free) * 4
        self.n_rings = len(self.rings)
        self.full_mask = (1 << self.n_rings) - 1

        # succ[node, press] -> successor node, or -1 when the manoeuvre crashes.
        # coll[node, press] -> bitmask of rings the sweep passes through.
        succ = np.full((n_nodes, 4), -1, np.int32)
        coll = np.zeros((n_nodes, 4), np.int32)
        for cell in free:
            for heading in DIRS:
                u = self.node_id[(cell, heading)]
                for pi, press in enumerate(DIRS):
                    cells, end, end_h = sweep(cell, heading, press)
                    got = 0
                    for x in cells:
                        if not (0 <= x[0] < height and 0 <= x[1] < width):
                            break                   # off the map == crash
                        if x in walls:
                            break                   # exploded here
                        if x in self.ring_id:
                            got |= 1 << self.ring_id[x]
                    else:                           # no break: the sweep is clear
                        succ[u, pi] = self.node_id[(end, end_h)]
                        coll[u, pi] = got
        self.succ, self.coll = succ, coll
        self.dist = self._solve(n_nodes)

    def _solve(self, n_nodes: int) -> np.ndarray:
        n_masks = 1 << self.n_rings
        dist = np.full((n_masks, n_nodes), _INF, np.int32)
        dist[0, :] = 0                              # every ring collected: done
        flies = self.succ >= 0
        # -1 would index the last row, so park crashing moves on node 0 and mask
        # them out of every min() instead.
        safe_succ = np.where(flies, self.succ, 0)
        for mask in sorted(range(n_masks), key=lambda m: bin(m).count("1")):
            if mask == 0:
                continue
            left = mask & ~self.coll                # rings still owed afterwards
            exits = flies & (left != mask)          # collects something new
            # Clamped at _INF so "dead" stays exactly _INF instead of drifting
            # upward one per layer as unreachable costs get +1'd along the way.
            cur = np.minimum(
                np.where(exits, dist[left, safe_succ] + 1, _INF).min(axis=1),
                _INF).astype(np.int32)
            inside = flies & (left == mask)         # stays in this layer
            inside_succ = np.where(inside, safe_succ, 0)
            for _ in range(n_nodes + 1):
                relaxed = np.where(inside, cur[inside_succ] + 1, _INF).min(axis=1)
                nxt = np.minimum(cur, relaxed).astype(np.int32)
                if np.array_equal(nxt, cur):
                    break
                cur = nxt
            dist[mask] = cur
        return dist

    # ── queries ─────────────────────────────────────────────────────────────
    def mask_of(self, live_rings) -> int:
        """Bitmask of the rings still on the board. Raises `KeyError` if a ring
        sits somewhere this field never knew about -- which would mean the field
        was built for a different layout, so silently mis-indexing it is the one
        outcome worth crashing on."""
        m = 0
        for cell in live_rings:
            m |= 1 << self.ring_id[cell]
        return m

    def steps_to_win(self, mask: int, node: int) -> int:
        return int(self.dist[mask, node])

    def optimal_presses(self, mask: int, node: int) -> list[str]:
        """Every press whose successor is exactly one step closer to the win --
        the co-optimal tie set. Empty when the state is dead."""
        here = int(self.dist[mask, node])
        if here >= _INF:
            return []
        out = []
        for pi, press in enumerate(DIRS):
            nxt = int(self.succ[node, pi])
            if nxt < 0:
                continue
            left = mask & ~int(self.coll[node, pi])
            if int(self.dist[left, nxt]) + 1 == here:
                out.append(press)
        return out

    def plan(self, mask: int, node: int) -> list[str] | None:
        """An optimal press sequence to the win, or None if the state is dead."""
        if int(self.dist[mask, node]) >= _INF:
            return None
        out: list[str] = []
        while mask:
            presses = self.optimal_presses(mask, node)
            if not presses:                          # unreachable: dist said finite
                return None
            press = presses[0]
            pi = DIRS.index(press)
            mask &= ~int(self.coll[node, pi])
            node = int(self.succ[node, pi])
            out.append(press)
        return out


class AerobaticsSolver(BaseSolver):
    """`BaseSolver` for ps:aerobatics, driving the real PuzzleScript adapter.

    Every hook that reads the world reads the LIVE engine grid, so the recorder's
    replan mode works end to end: an exploratory detour or a perturbation burst
    just moves the plane to another state the field already knows the answer for.
    """

    game_id = "puzzlescript_aerobatics"

    supports_recovery = True
    #: The field covers every state, so a perturbed plane is re-planned in place;
    #: RESET is only the last resort, for a state that is genuinely dead (the
    #: plane exploded, or a ring is walled off from it).
    recovery_mode = "replan"
    #: Crashing is easy -- random flying finds a wall within a few actions on the
    #: tighter courses -- so the exploration prefix needs more RESET headroom than
    #: the base default of 3.
    max_resets = 8

    #: `solve_from` / `optimal_set_from` already emit the SCREEN press (they plan
    #: in engine space and invert the adapter's remap via `screen_action`), so the
    #: base must not convert them again. `drive` is overridden and submits the
    #: press to the adapter untouched, which applies its own forward remap.
    plans_in_screen_space = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        #: level index -> FlightField. The engine state after a reset is
        #: seed-independent, so this is built once per process and reused by
        #: every later seed.
        self._fields: dict[int, FlightField] = {}
        self._names = None          # object-name -> index, resolved per game

    # ── engine glue ─────────────────────────────────────────────────────────
    def make_game(self, seed: int):
        """Build the adapter through ``games/ps:aerobatics/ps:aerobatics.py`` --
        the same entry point `game_envs` hands a live agent -- so the generator
        can never tape frames from a differently-configured adapter."""
        self._names = None
        return _load_game_module(_GAME_ID).make_game(seed=seed)

    def num_levels(self, game) -> int:
        return game.n_levels

    def set_level(self, game, level_idx: int) -> None:
        super().set_level(game, level_idx)
        # Build the field HERE, at the level's initial state, where the ring set
        # is complete. Building it lazily from the first `solve_from` would work
        # only if that call happened before any ring was collected; a field built
        # from a partially-cleared board would silently index the wrong rings.
        #
        # This is also the one place the cache's premise -- that the engine state
        # after a reset is the same for every seed, so a field built at seed 0 is
        # valid for all of them -- can be CHECKED rather than assumed. If a level
        # ever gains a per-seed layout, the cached field is rebuilt here instead
        # of quietly answering for the wrong board.
        walls, rings, _, _ = self._scan(game)
        fld = self._fields.get(level_idx)
        if fld is None or fld.walls != frozenset(walls) or fld.rings != rings:
            fld = FlightField(walls, rings, game._engine.height,
                              game._engine.width)
            self._fields[level_idx] = fld

    def render(self, game) -> np.ndarray:
        return np.asarray(game._current_frame)

    def available_actions(self, game) -> list[int]:
        """The four manoeuvre keys. ACTION5 (flight-plan HUD) is excluded: it
        cannot move the plane or clear a ring, and on half the levels it does not
        change a single pixel, so offering it to the exploration policy would
        only spend exploration budget on a null transition."""
        return [1, 2, 3, 4]

    def drive(self, game, action: Action) -> DriveResult:
        fd = game.perform_action(ActionInput(id=_ID_TO_GAMEACTION[action.action_id]))
        frames = np.asarray(fd.frame) if fd.frame else np.asarray(
            game._current_frame)[None]
        solved = game._state == GameState.WIN
        # The game has no lose condition: a plane that hits a wall is replaced by
        # an Explosion and the board just sits there, permanently unwinnable
        # (``some Plane`` is part of the win). Report that as dead so the recorder
        # rolls a burst back / RESETs immediately, instead of flying an
        # unrecoverable state until `solve_from` happens to be consulted.
        dead = not solved and (game._state == GameState.GAME_OVER
                               or not self._alive(game))
        return DriveResult(frames, solved, dead)

    # ── reading the live board ──────────────────────────────────────────────
    def _ids(self, game) -> dict:
        if self._names is None:
            n2i = game._game.obj_name_to_idx
            self._names = {
                "walls": frozenset(n2i[n] for n in _WALL_NAMES if n in n2i),
                "ring": n2i["ring"],
                "planes": {n2i[n]: h for n, h in _PLANE_HEADING.items()
                           if n in n2i},
            }
        return self._names

    def _alive(self, game) -> bool:
        planes = frozenset(self._ids(game)["planes"])
        return any(cell & planes
                   for row in game._engine.grid for cell in row)

    def _scan(self, game):
        """``(walls, rings, plane_cell, heading)`` off the live engine grid.

        ``plane_cell``/``heading`` are None when the plane is gone (exploded) or
        showing a mid-turn diagonal sprite -- see `_PLANE_HEADING`."""
        ids = self._ids(game)
        wall_ids, ring_id, plane_ids = ids["walls"], ids["ring"], ids["planes"]
        walls, rings = set(), []
        plane_cell = heading = None
        for r, row in enumerate(game._engine.grid):
            for c, cell in enumerate(row):
                if cell & wall_ids:
                    walls.add((r, c))
                if ring_id in cell:
                    rings.append((r, c))
                for pid, h in plane_ids.items():
                    if pid in cell:
                        plane_cell, heading = (r, c), h
        return walls, rings, plane_cell, heading

    def _state(self, game, level_idx: int):
        """``(field, mask, node)`` for the live board, or None if it is not a
        state the field can answer for (plane destroyed, or mid-turn).

        The field is always already built: `set_level` builds it at the level's
        initial state, which is the only moment the full ring set is on the board
        (see there)."""
        fld = self._fields[level_idx]
        _, rings, cell, heading = self._scan(game)
        if cell is None or heading is None:
            return None
        node = fld.node_id.get((cell, heading))
        if node is None:                      # plane inside a wall: impossible
            return None
        return fld, fld.mask_of(rings), node

    def _press(self, game, direction: str) -> Action:
        """Engine direction -> the SCREEN button that makes the adapter execute
        it, inverting the adapter's rotation + flip remap ([[ps-astar-generator-family]])."""
        return Action(int(screen_action(direction, game._rotation_k,
                                        game._hflip, game._vflip).value))

    # ── the two BaseSolver hooks ────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int) -> list:
        st = self._state(game, level_idx)
        if st is None:
            return []
        fld, mask, node = st
        presses = fld.plan(mask, node)
        if presses is None:
            return []
        return [self._press(game, d) for d in presses]

    def optimal_set_from(self, game, level_idx: int, seed: int):
        st = self._state(game, level_idx)
        if st is None:
            return None
        fld, mask, node = st
        presses = fld.optimal_presses(mask, node)
        if not presses:
            return None
        return [self._press(game, d) for d in presses]


# ---------------------------------------------------------------------------
# Differential test: `sweep` vs the real PuzzleScript interpreter
# ---------------------------------------------------------------------------

def verify_model(trials: int = 4000, size: int = 13, seed: int = 12345,
                 verbose: bool = True) -> int:
    """Cross-check `sweep` against the engine on random boards. Returns the
    number of mismatches (0 is the only acceptable answer).

    `sweep` is the ONE place this generator re-implements PuzzleScript instead of
    calling it, and it encodes four things that are easy to get subtly wrong: the
    order of the cells a banking turn crosses (``q, q+f, q+2q+f`` -- not
    ``f, q, q``, which sweeps the same displacement through different cells), the
    loop-de-loop's overshoot to ``p+2f`` before settling back on ``p+f``, that a
    wall ANYWHERE in the sweep is fatal rather than blocking, and that a ring is
    taken at every swept cell rather than only where the plane stops. So it is
    tested against the interpreter directly, on boards dense enough (18% walls,
    ~25% rings) that every one of those cases fires constantly, rather than only
    on the ten shipped courses.

    Each trial loads a random board straight into the engine, presses one key,
    and compares crash / final cell / final heading / surviving rings.
    """
    import random

    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    game = PuzzleScriptAdapter(GAME_NAME, seed=0)
    eng, parsed = game._engine, game._game
    n2i, i2n = parsed.obj_name_to_idx, parsed.obj_idx_to_name
    bg, wall_id, ring_id = n2i["background"], n2i["wallx"], n2i["ring"]
    sprite = {h: n2i[n] for n, h in _PLANE_HEADING.items()}
    rng = random.Random(seed)
    bad = 0

    for _ in range(trials):
        walls, rings = set(), set()
        for r in range(size):
            for c in range(size):
                if r in (0, size - 1) or c in (0, size - 1):
                    walls.add((r, c))
                elif rng.random() < 0.18:
                    walls.add((r, c))
                elif rng.random() < 0.25:
                    rings.add((r, c))
        free = [(r, c) for r in range(size) for c in range(size)
                if (r, c) not in walls]
        pos = rng.choice(free)
        rings.discard(pos)
        heading, press = rng.choice(DIRS), rng.choice(DIRS)

        level = [[({bg, wall_id} if (r, c) in walls else
                   {bg, ring_id} if (r, c) in rings else {bg})
                  for c in range(size)] for r in range(size)]
        level[pos[0]][pos[1]] = {bg, sprite[heading], n2i["player"]}
        eng.load_level(level)
        eng.step(press)

        eng_pos = eng_head = None
        eng_crash = False
        eng_rings = set()
        for r, row in enumerate(eng.grid):
            for c, cell in enumerate(row):
                for o in cell:
                    name = i2n[o]
                    if name in _PLANE_HEADING:
                        eng_pos, eng_head = (r, c), _PLANE_HEADING[name]
                    elif name.startswith("explosion"):
                        eng_crash = True
                    elif name == "ring":
                        eng_rings.add((r, c))

        cells, end, end_head = sweep(pos, heading, press)
        swept, crash = [], False
        for x in cells:
            if not (0 <= x[0] < size and 0 <= x[1] < size) or x in walls:
                crash = True
                break
            swept.append(x)
        ok = crash == eng_crash
        if not crash and not eng_crash:
            ok = (eng_pos == end and eng_head == end_head
                  and eng_rings == rings - set(swept))
        if not ok:
            bad += 1
            if verbose and bad <= 5:
                print(f"  MISMATCH from {pos} heading={heading} press={press}: "
                      f"engine={eng_pos},{eng_head},crash={eng_crash} "
                      f"model={end},{end_head},crash={crash}")
    if verbose:
        print(f"verify_model: {trials} trials, {bad} mismatches")
    return bad


if __name__ == "__main__":
    if "--verify-model" in sys.argv:
        sys.exit(1 if verify_model() else 0)
    sys.exit(AerobaticsSolver.main())
