"""
Toy Train Routes
================
Drive a toy train along colored track segments to pick up passengers and
deliver them to their matching colored destination stations.

Track movement is constrained: the train can only move in a direction if
the current cell connects in that direction AND the destination cell
connects back.  Switch cells have a lever: ACTION5 when standing on a
switch toggles it between two bit-masks of connected directions.
Lever 0 keeps the main-line connections only; Lever 1 opens an extra
branch direction so the train can detour onto branch tracks.

Color key:
  0  (#FFFFFF) — empty background
  4  (#333333) — rail track (dark gray)
  7  (#E6E6E6) — switch / lever junction (light gray)
  9  (#1E93FF) — train locomotive (blue)
  8  (#F93C31) — red  passenger / destination
  11 (#FFDC00) — yellow passenger / destination
  12 (#FF851B) — orange passenger / destination
  6  (#E36EF6) — pink  passenger / destination
  1  (#0074D9) — blue  passenger / destination
  14 (#4FCC30) — station platform (green)
  5  (#000000) — letterbox border

Actions:
  ACTION1 (↑) — move train north (if track connects)
  ACTION2 (↓) — move train south
  ACTION3 (←) — move train west
  ACTION4 (→) — move train east
  ACTION5     — toggle switch lever (when at switch); OR
                  pick up waiting passenger (when at occupied station); OR
                  drop off matching passenger (when at destination)
  ACTION7     — undo last move or lever flip (not pickups / drop-offs)

Win condition : All passengers delivered to their matching destinations.
Lose condition: Step budget exhausted.
Levels        : 4 puzzles — 2 passengers, 1 switch (easy) →
                             3 passengers, 2 switches (medium) →
                             4 passengers, 3 switches (hard) →
                             5 passengers, 4 switches (expert).
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
from utils.arc_game import AugmentedGame
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

# ── Colors ───────────────────────────────────────────────────────────────────
C_EMPTY   = 0
C_TRACK   = 4
C_SWITCH  = 7
C_TRAIN   = 9
C_STATION = 14
C_BORDER  = 5

_PASS_COLORS = [8, 11, 12, 6, 1, 2, 3, 10]   # red, yellow, orange, pink, blue, magenta, teal, cyan

_STEP_BUDGETS = [100, 180, 300, 440, 620, 840, 1100]
_TRAIN_CAPACITY = 2
CELL = 3

# ── Direction bits (N=1, E=2, S=4, W=8) ─────────────────────────────────────
_N, _E, _S, _W = 1, 2, 4, 8
_DIR_TO_BIT = {(0, -1): _N, (1, 0): _E, (0, 1): _S, (-1, 0): _W}
_BIT_TO_DX  = {_N: (0, -1), _E: (1, 0), _S: (0, 1), _W: (-1, 0)}

# ── Pixel art  (all CELL×CELL = 3×3) ────────────────────────────────────────
_B = -1  # transparent

def _track_pixels(bits: int) -> list[list[int]]:
    """3×3 rail sprite matching the given connection bitmask."""
    n = bool(bits & _N); e = bool(bits & _E)
    s = bool(bits & _S); w = bool(bits & _W)
    ctr = C_TRACK if (n or e or s or w) else _B
    return [
        [_B,              C_TRACK if n else _B, _B             ],
        [C_TRACK if w else _B, ctr,             C_TRACK if e else _B],
        [_B,              C_TRACK if s else _B, _B             ],
    ]

def _switch_pixels(active_bits: int) -> list[list[int]]:
    """3×3 switch sprite — active connections bright, inactive dark."""
    n = bool(active_bits & _N); e = bool(active_bits & _E)
    s = bool(active_bits & _S); w = bool(active_bits & _W)
    def _c(active):
        return C_SWITCH if active else C_TRACK
    return [
        [_B,        _c(n), _B   ],
        [_c(w),     C_SWITCH, _c(e)],
        [_B,        _c(s), _B   ],
    ]

_TRAIN_PIXELS = [
    [C_TRAIN,  _B,      C_TRAIN],
    [C_TRAIN,  C_TRAIN, C_TRAIN],
    [5,        C_TRAIN, 5      ],
]

def _train_with_pass_pixels(c: int) -> list[list[int]]:
    return [
        [c,       _B,     C_TRAIN],
        [C_TRAIN, C_TRAIN, C_TRAIN],
        [5,       C_TRAIN, 5      ],
    ]

def _passenger_pixels(c: int) -> list[list[int]]:
    """Small 3×3 waiting-passenger figure."""
    return [
        [_B, c,  _B],
        [c,  c,  c ],
        [_B, c,  _B],
    ]

def _station_pixels(c: int) -> list[list[int]]:
    """Platform with colored sign."""
    return [
        [c,        c,        C_STATION],
        [C_STATION, C_STATION, C_STATION],
        [C_STATION, C_STATION, C_STATION],
    ]

def _dest_pixels(c: int) -> list[list[int]]:
    """Destination marker: colored diamond on platform."""
    return [
        [C_STATION, c,        C_STATION],
        [c,         C_STATION, c       ],
        [C_STATION, c,        C_STATION],
    ]


# ── HUD ──────────────────────────────────────────────────────────────────────

class StepCounter(RenderableUserDisplay):
    def __init__(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.steps_remaining = max_steps

    def decrement(self) -> None:
        if self.steps_remaining > 0:
            self.steps_remaining -= 1

    def reset(self, max_steps: int) -> None:
        self.max_steps = max_steps
        self.steps_remaining = max_steps

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        if self.max_steps == 0:
            return frame
        filled = round(64 * self.steps_remaining / self.max_steps)
        for x in range(64):
            frame[63, x] = 7 if x < filled else 4
        return frame


# ── Level generation ─────────────────────────────────────────────────────────
# (width, height, num_passengers, num_switches)
_CONFIGS = [
    (11,  9, 2, 1),
    (13, 11, 3, 2),
    (15, 13, 4, 3),
    (17, 15, 5, 4),
    (19, 17, 5, 4),
    (21, 19, 5, 4),
    (21, 19, 5, 5),
]


def _make_level(w: int, h: int, num_pass: int, num_sw: int,
                rng: random.Random) -> dict:
    """
    Generate a ToyTrain level.

    Layout:
      • Outer oval loop (clockwise: top→right→bottom←left↑).
      • num_sw switches sampled randomly across all four sides of the loop.
      • Each switch grows an inward branch (toward the center), ending in a
        dead-end track cell.
      • Lever 0: switch acts as main-line side track only.
      • Lever 1: switch adds the inward branch direction.

    Stations (passengers) are placed at branch dead-ends first, then on the
    oval.  Destinations are placed on the oval (not on branches).
    """
    grid = [[0] * w for _ in range(h)]

    lx, rx = 1, w - 2
    ty, by = 1, h - 2

    # Oval corners
    grid[ty][lx] = _S | _E   # SE (top-left): connects south+east
    grid[ty][rx] = _S | _W   # SW (top-right)
    grid[by][lx] = _N | _E   # NE (bottom-left)
    grid[by][rx] = _N | _W   # NW (bottom-right)

    # Top and bottom horizontal tracks
    for x in range(lx + 1, rx):
        grid[ty][x] = _E | _W
        grid[by][x] = _E | _W

    # Left and right vertical tracks
    for y in range(ty + 1, by):
        grid[y][lx] = _N | _S
        grid[y][rx] = _N | _S

    switch_positions: list[tuple[int, int]] = []
    # (bits_lever0, bits_lever1) for each switch
    switch_bits: list[tuple[int, int]] = []
    branch_ends: list[tuple[int, int]] = []
    side_candidates: list[tuple[int, int, int, int, int, int, int]] = []
    # (sx, sy, main_bits, branch_bit, dx, dy, max_depth)
    for x in range(lx + 2, rx - 1):
        side_candidates.append((x, ty, _E | _W, _S, 0, 1, by - ty - 1))   # top -> inward south
        side_candidates.append((x, by, _E | _W, _N, 0, -1, by - ty - 1))  # bottom -> inward north
    for y in range(ty + 2, by - 1):
        side_candidates.append((lx, y, _N | _S, _E, 1, 0, rx - lx - 1))   # left -> inward east
        side_candidates.append((rx, y, _N | _S, _W, -1, 0, rx - lx - 1))  # right -> inward west

    rng.shuffle(side_candidates)
    chosen_switches = side_candidates[:num_sw]

    for si, (sx, sy, main_bits, branch_bit, dx, dy, max_depth) in enumerate(chosen_switches):
        grid[sy][sx] = 100 + si
        switch_positions.append((sx, sy))
        switch_bits.append((main_bits, main_bits | branch_bit))

        # Build inward branch; stop before collisions with other branches.
        min_depth = 2
        if max_depth < min_depth:
            continue
        branch_len = rng.randint(min_depth, max_depth)
        path: list[tuple[int, int]] = []
        cx, cy = sx, sy
        for _ in range(branch_len):
            cx += dx
            cy += dy
            if grid[cy][cx] != 0:
                break
            path.append((cx, cy))

        if not path:
            continue

        back_bit = _DIR_TO_BIT[(-dx, -dy)]
        fwd_bit = _DIR_TO_BIT[(dx, dy)]
        for px, py in path[:-1]:
            grid[py][px] = back_bit | fwd_bit
        tx, ty2 = path[-1]
        grid[ty2][tx] = back_bit
        branch_ends.append((tx, ty2))

    # ── Collect oval cells for station/dest placement ───────────────────────
    # Walk the oval in CW order; exclude corners and switch cells.  A switch
    # cell MUST be excluded: ACTION5 there flips the lever and returns, so a
    # passenger waiting on a switch could never be picked up (nor a passenger
    # dropped off), which makes the level unsolvable.
    def _plain(x: int, y: int) -> bool:
        return 0 < grid[y][x] < 100

    oval_cells: list[tuple[int, int]] = []
    for x in range(lx + 1, rx):
        if _plain(x, ty):
            oval_cells.append((x, ty))
    for y in range(ty + 1, by):
        if _plain(rx, y):
            oval_cells.append((rx, y))
    for x in range(rx - 1, lx, -1):
        if _plain(x, by):
            oval_cells.append((x, by))
    for y in range(by - 1, ty, -1):
        if _plain(lx, y):
            oval_cells.append((lx, y))

    rng.shuffle(oval_cells)
    colors = rng.sample(_PASS_COLORS, k=num_pass)

    # Stations: branch dead-ends first, then oval
    rng.shuffle(branch_ends)
    stations: list[tuple[int, int]] = list(branch_ends[:num_pass])
    oval_pool = [c for c in oval_cells if c not in set(stations)]
    while len(stations) < num_pass and oval_pool:
        stations.append(oval_pool.pop(0))
    rng.shuffle(stations)

    # Destinations: oval only, not at station cells
    station_set = set(stations)
    dests: list[tuple[int, int]] = []
    for c in oval_cells:
        if c not in station_set and len(dests) < num_pass:
            dests.append(c)
    if len(dests) < num_pass:
        for x in range(lx + 1, rx):
            c = (x, ty)
            if _plain(x, ty) and c not in station_set and c not in dests:
                dests.append(c)
            if len(dests) == num_pass:
                break

    train_start = (lx, (ty + by) // 2)

    return {
        "grid":             grid,
        "width":            w,
        "height":           h,
        "switch_positions": switch_positions,
        "switch_bits":      switch_bits,   # [(bits_0, bits_1), ...]
        "branch_ends":      branch_ends,
        "stations":         stations,      # [(gx, gy), ...] per passenger
        "dests":            dests,
        "colors":           colors,
        "num_passengers":   num_pass,
        "train_start":      train_start,
    }


def _build_base_level(data: dict) -> Level:
    """Build track/switch/station/dest sprites (no train or waiting passengers)."""
    grid   = data["grid"]
    w, h   = data["width"], data["height"]
    colors = data["colors"]
    switch_positions = set(data["switch_positions"])
    sw_idx = {pos: i for i, pos in enumerate(data["switch_positions"])}
    station_idx = {pos: i for i, pos in enumerate(data["stations"])}
    dest_idx    = {pos: i for i, pos in enumerate(data["dests"])}

    sprites: list[Sprite] = []

    for y in range(h):
        for x in range(w):
            val = grid[y][x]
            if val == 0:
                continue
            px, py = x * CELL, y * CELL

            if val < 100:
                # Regular track sprite
                sprites.append(
                    Sprite(pixels=_track_pixels(val), name=f"track_{x}_{y}",
                           collidable=False, tags=["track"], layer=0)
                    .set_position(px, py)
                )
            else:
                # Switch cell: initial active_bits = pair 0 (lever=0)
                si = val - 100
                bits0 = data["switch_bits"][si][0]
                sprites.append(
                    Sprite(pixels=_switch_pixels(bits0), name=f"switch_{si}",
                           collidable=False,
                           tags=["switch", f"sw_{si}"], layer=0)
                    .set_position(px, py)
                )

            # Overlay station sprite if this cell is a station
            if (x, y) in station_idx:
                pi = station_idx[(x, y)]
                c = colors[pi] if pi < len(colors) else C_STATION
                sprites.append(
                    Sprite(pixels=_station_pixels(c), name=f"station_{pi}",
                           collidable=False, tags=["station", f"station_{pi}"], layer=1)
                    .set_position(px, py)
                )

            # Overlay destination sprite if this cell is a destination
            if (x, y) in dest_idx:
                pi = dest_idx[(x, y)]
                c = colors[pi] if pi < len(colors) else C_STATION
                sprites.append(
                    Sprite(pixels=_dest_pixels(c), name=f"dest_{pi}",
                           collidable=False, tags=["destination", f"dest_{pi}"], layer=1)
                    .set_position(px, py)
                )

    return Level(sprites=sprites, grid_size=(w * CELL, h * CELL))


# ── Game class ────────────────────────────────────────────────────────────────

class ToyTrain(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base rotates the board on set_level
        rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])
        self._level_data: list[dict] = []
        levels: list[Level] = []

        for w, h, num_pass, num_sw in _CONFIGS:
            data = _make_level(w, h, num_pass, num_sw, rng)
            self._level_data.append(data)
            levels.append(_build_base_level(data))

        # Runtime state (set in on_set_level)
        self._lever_states:       list[int] = []
        self._pass_on_train:      list[int] = []  # passenger indices on train
        self._pass_delivered:     set[int]  = set()
        self._pass_at_station:    list[bool] = []  # passenger i still waiting?
        self._history: list[tuple] = []

        super().__init__(
            game_id="toy_train",
            levels=levels,
            camera=Camera(
                background=C_EMPTY,
                letter_box=C_BORDER,
                interfaces=[self._step_counter],
            ),
            available_actions=[1, 2, 3, 4, 5, 7],
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _data(self) -> dict:
        return self._level_data[self._current_level_index]

    def _train(self) -> Sprite:
        return self.current_level.get_sprites_by_tag("train")[0]

    def _switch_sprite(self, si: int) -> Sprite:
        return self.current_level.get_sprites_by_tag(f"sw_{si}")[0]

    def _passenger_sprite(self, pi: int) -> Sprite | None:
        sprites = self.current_level.get_sprites_by_tag(f"waiting_{pi}")
        return sprites[0] if sprites else None

    def _active_bits(self, gx: int, gy: int) -> int:
        """Return the effective connection bitmask for grid cell (gx, gy)."""
        grid = self._data()["grid"]
        val = grid[gy][gx]
        if val == 0:
            return 0
        if val < 100:
            return val
        si = val - 100
        lever = self._lever_states[si]
        return self._data()["switch_bits"][si][lever]

    def _can_move(self, gx: int, gy: int, dx: int, dy: int) -> bool:
        """True if the train at (gx,gy) may move in direction (dx,dy)."""
        data = self._data()
        nx, ny = gx + dx, gy + dy
        if not (0 <= nx < data["width"] and 0 <= ny < data["height"]):
            return False
        bit  = _DIR_TO_BIT[(dx, dy)]
        obit = _DIR_TO_BIT[(-dx, -dy)]
        return bool(self._active_bits(gx, gy) & bit) and \
               bool(self._active_bits(nx, ny) & obit)

    def _grid_pos(self) -> tuple[int, int]:
        t = self._train()
        return t.x // CELL, t.y // CELL

    def _snapshot(self) -> tuple:
        gx, gy = self._grid_pos()
        return (
            gx, gy,
            list(self._pass_on_train),
            set(self._pass_delivered),
            list(self._pass_at_station),
            list(self._lever_states),
        )

    def _restore(self, snap: tuple) -> None:
        gx, gy, on_train, delivered, at_station, levers = snap
        self._train().set_position(gx * CELL, gy * CELL)
        self._pass_on_train   = list(on_train)
        self._pass_delivered  = set(delivered)
        self._pass_at_station = list(at_station)
        self._lever_states    = list(levers)
        self._refresh_train()
        self._refresh_waiting_passengers()
        self._refresh_switches()

    def _refresh_train(self) -> None:
        train = self._train()
        if self._pass_on_train:
            c = self._data()["colors"][self._pass_on_train[0]]
            train.pixels = np.array(_train_with_pass_pixels(c), dtype=np.int8)
        else:
            train.pixels = np.array(_TRAIN_PIXELS, dtype=np.int8)

    def _refresh_waiting_passengers(self) -> None:
        data = self._data()
        for pi in range(data["num_passengers"]):
            sp = self._passenger_sprite(pi)
            if sp is not None:
                sp.set_visible(self._pass_at_station[pi])

    def _refresh_switches(self) -> None:
        data = self._data()
        for si in range(len(data["switch_positions"])):
            lever = self._lever_states[si]
            bits  = data["switch_bits"][si][lever]
            sw    = self._switch_sprite(si)
            sw.pixels = np.array(_switch_pixels(bits), dtype=np.int8)

    def _check_win(self) -> bool:
        return len(self._pass_delivered) == self._data()["num_passengers"]

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def on_set_level(self, level: Level) -> None:
        self._history.clear()
        idx = self._current_level_index
        self._step_counter.reset(_STEP_BUDGETS[idx])
        data = self._data()

        # Idempotency: remove dynamic sprites (train, waiting passengers)
        for sp in level.get_sprites():
            if "train" in sp.tags or "waiting" in sp.tags:
                level.remove_sprite(sp)

        # Reset runtime state
        self._lever_states      = list(data.get("lever_defaults",
                                                  [0] * len(data["switch_positions"])))
        self._pass_on_train     = []
        self._pass_delivered    = set()
        self._pass_at_station   = [True] * data["num_passengers"]

        # Reset switch visuals
        self._refresh_switches()

        # Add waiting-passenger sprites at each station
        for pi, (sx, sy) in enumerate(data["stations"]):
            c = data["colors"][pi]
            level.add_sprite(
                Sprite(pixels=_passenger_pixels(c), name=f"waiting_pass_{pi}",
                       collidable=False, tags=["waiting", f"waiting_{pi}"], layer=2)
                .set_position(sx * CELL, sy * CELL)
            )

        # Add train
        tx, ty = data["train_start"]
        level.add_sprite(
            Sprite(pixels=_TRAIN_PIXELS, name="train",
                   collidable=False, tags=["train"], layer=3)
            .set_position(tx * CELL, ty * CELL)
        )

    # ── Step ─────────────────────────────────────────────────────────────────

    _MOVE_DELTAS = {
        GameAction.ACTION1: (0,  -1),
        GameAction.ACTION2: (0,   1),
        GameAction.ACTION3: (-1,  0),
        GameAction.ACTION4: (1,   0),
    }

    def step(self) -> None:
        self._step_counter.decrement()
        action = self.action.id

        # ── Undo ──────────────────────────────────────────────────────────────
        if action == GameAction.ACTION7:
            if self._history:
                self._restore(self._history.pop())
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # ── Interact (ACTION5) ────────────────────────────────────────────────
        if action == GameAction.ACTION5:
            data = self._data()
            gx, gy = self._grid_pos()
            grid   = data["grid"]
            val    = grid[gy][gx]

            # At a switch: flip lever
            if val >= 100:
                si = val - 100
                self._history.append(self._snapshot())
                self._lever_states[si] ^= 1
                self._refresh_switches()
                self.complete_action()
                if not self._step_counter.steps_remaining:
                    self.lose()
                return

            # At a station: pick up passenger (if waiting and capacity allows)
            if (gx, gy) in (pos for pos in data["stations"]):
                pi = data["stations"].index((gx, gy))
                if (self._pass_at_station[pi]
                        and pi not in self._pass_on_train
                        and len(self._pass_on_train) < _TRAIN_CAPACITY):
                    self._history.append(self._snapshot())
                    self._pass_at_station[pi] = False
                    self._pass_on_train.append(pi)
                    self._refresh_train()
                    self._refresh_waiting_passengers()
                    self.complete_action()
                    if not self._step_counter.steps_remaining:
                        self.lose()
                    return

            # At a destination: drop off matching passenger
            if (gx, gy) in (pos for pos in data["dests"]):
                pi = data["dests"].index((gx, gy))
                if pi in self._pass_on_train and pi not in self._pass_delivered:
                    self._history.append(self._snapshot())
                    self._pass_on_train.remove(pi)
                    self._pass_delivered.add(pi)
                    self._refresh_train()
                    if self._check_win():
                        self.next_level()
                        self.complete_action()
                        return

            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # ── Move train ────────────────────────────────────────────────────────
        delta = self._MOVE_DELTAS.get(action)
        if delta is None:
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        dx, dy = delta
        gx, gy = self._grid_pos()

        if not self._can_move(gx, gy, dx, dy):
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        self._history.append(self._snapshot())
        self._train().set_position((gx + dx) * CELL, (gy + dy) * CELL)

        self.complete_action()
        if not self._step_counter.steps_remaining:
            self.lose()
