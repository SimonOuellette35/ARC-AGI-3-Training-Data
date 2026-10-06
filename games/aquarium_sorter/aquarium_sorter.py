"""
Aquarium Sorter
===============
Sort fish into the correct aquarium compartments by opening and closing
gates. Fish swim left and right automatically after each player action.
When a fish reaches an open gate it swims into the adjacent compartment.
When it hits a closed gate or wall it reverses direction. A cursor moves
along the gate positions. ACTION5 toggles the gate under the cursor.
ACTION7 undoes the last gate toggle and reverts fish one step.

Color key:
  1  (#1E93FF) — water (compartment background)
  4  (#333333) — partition wall (impassable)
  0  (#FFFFFF) — open gate (gap in wall)
  7  (#E6E6E6) — closed gate (solid segment)
  8  (#F93C31) — clownfish (orange-red)
  11 (#FFDC00) — goldfish (yellow)
  14 (#4FCC30) — angelfish (green)
  12 (#FF851B) — neon tetra (orange)
  9  (#1E93FF) — cursor position (bright blue)
  5  (#000000) — letterbox border

Target compartment indicators shown as colored strips on right wall.

Actions:
  ACTION1 (↑) — move cursor up
  ACTION2 (↓) — move cursor down
  ACTION3 (←) — move cursor to left wall
  ACTION4 (→) — move cursor to right wall
  ACTION5     — toggle gate at cursor position
  ACTION7     — undo last gate toggle (reverts fish)

Win condition : Each compartment contains exactly its target fish species.
Lose condition: Step budget exhausted.
Levels        : 4 tanks — 2 compartments 2 species (easy) → 4 compartments 4 species.
"""

from __future__ import annotations

import random

import numpy as np
from utils.arc_game import AugmentedGame
from arcengine import ARCBaseGame, Camera, GameAction, Level, RenderableUserDisplay, Sprite

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------
C_WATER   = 1   # blue water
C_WALL    = 4   # dark partition
C_GATE_OPEN   = 0  # white gap
C_GATE_CLOSED = 7  # light grey closed
C_CURSOR  = 9   # blue cursor
C_BORDER  = 5   # black

_FISH_COLORS = [8, 11, 14, 12]   # clownfish, goldfish, angelfish, neon

_STEP_BUDGETS = [60, 200, 300, 400, 550, 700, 900]

# ---------------------------------------------------------------------------
# Cell size: CELL=3 gives max viewport 48×51 for 16×17 grid.
# ---------------------------------------------------------------------------
CELL = 3

# ---------------------------------------------------------------------------
# Sprite pixel art  (all CELL×CELL = 3×3 pixels)
# ---------------------------------------------------------------------------

def _fish_pixels(c: int) -> list[list[int]]:
    """3×3 fish: body + eye + tail."""
    return [
        [0,  c,  c],  # eye (white) + top fin
        [c,  c,  c],  # body
        [-1, c, -1],  # tail fork
    ]

# Cursor: 3×3 bracket indicator
_CURSOR_PIXELS = [
    [C_CURSOR, -1,       C_CURSOR],
    [-1,       C_CURSOR, -1      ],
    [C_CURSOR, -1,       C_CURSOR],
]

# Level configs: (tank_width, compartments, fish_per_comp)
_CONFIGS = [
    (12, 2, 2),
    (14, 3, 2),
    (14, 4, 2),
    (16, 4, 3),
    (18, 4, 4),
    (20, 4, 4),
    (20, 4, 5),
]


# ---------------------------------------------------------------------------
# HUD
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Level generator
# ---------------------------------------------------------------------------

def _generate(tank_w: int, num_comp: int, fish_per: int,
              rng: random.Random) -> dict:
    """
    Generate a guaranteed-solvable tank configuration.

    Fish placement strategy (ensures solvability without BFS check):
    - Each fish of species ``sp`` is placed in a compartment *adjacent* to its
      target compartment (±1 hop away).
    - The fish's initial direction points **toward** its target compartment so
      that solutions are reachable within a small step budget.

    Fish positions: (x, comp_idx, direction)  direction = +1 (right) or -1 (left)
    """
    comp_h = 4  # height of each compartment
    total_h = num_comp * comp_h + 1  # +1 for bottom wall
    color_pool = list(_FISH_COLORS)
    rng.shuffle(color_pool)
    colors = color_pool[:num_comp]

    # Randomise targets (which species each compartment should ultimately hold)
    species_list = list(range(num_comp))
    rng.shuffle(species_list)
    targets = species_list

    target_to_comp = {targets[ci]: ci for ci in range(num_comp)}
    all_fish: list[dict] = []
    for sp in range(num_comp):
        tc = target_to_comp[sp]
        # Only place fish in compartments directly adjacent to the target
        adjacent = [c for c in [tc - 1, tc + 1] if 0 <= c < num_comp]
        # All fish of same species go into the SAME starting compartment so
        # the scripted solver can sort each group cleanly.
        sc = rng.choice(adjacent)
        # Direction always points toward the target compartment
        d = 1 if sc < tc else -1
        for _ in range(fish_per):
            # Ensure no initial same-cell overlap with fish of a different species
            # in the same starting compartment (prevents visual confusion from the start).
            for _attempt in range(200):
                x_pos = rng.randint(1, tank_w - 2)
                if not any(
                    f["comp"] == sc and f["x"] == x_pos and f["species"] != sp
                    for f in all_fish
                ):
                    break
            all_fish.append({
                "comp": sc,
                "x": x_pos,
                "dir": d,
                "species": sp,
            })

    return {
        "tank_w": tank_w,
        "comp_h": comp_h,
        "num_comp": num_comp,
        "total_h": total_h,
        "targets": targets,
        "colors": colors,
        "fish": all_fish,
        "fish_per": fish_per,
    }


def _comp_y_range(data: dict, ci: int) -> tuple[int, int]:
    """Return (top_y, bottom_y) of compartment ci interior."""
    comp_h = data["comp_h"]
    top = ci * comp_h
    return top + 1, top + comp_h - 1  # interior rows


def _build_level(data: dict) -> Level:
    tank_w = data["tank_w"]
    num_comp = data["num_comp"]
    comp_h = data["comp_h"]
    total_h = data["total_h"]
    colors = data["colors"]
    targets = data["targets"]
    sprites: list[Sprite] = []

    # Build wall structure
    for y in range(total_h):
        for x in range(tank_w):
            # Left and right border walls
            if x == 0 or x == tank_w - 1:
                sprites.append(
                    Sprite(pixels=[[C_WALL]*CELL]*CELL, name=f"bwall_{x}_{y}",
                           collidable=True, tags=["wall"], layer=-1)
                    .set_position(x * CELL, y * CELL)
                )
            # Partition walls between compartments
            elif y % comp_h == 0:
                sprites.append(
                    Sprite(pixels=[[C_WALL]*CELL]*CELL, name=f"pwall_{x}_{y}",
                           collidable=True, tags=["wall"], layer=-1)
                    .set_position(x * CELL, y * CELL)
                )
            else:
                # Water
                sprites.append(
                    Sprite(pixels=[[C_WATER]*CELL]*CELL, name=f"water_{x}_{y}",
                           collidable=False, tags=["water"], layer=0)
                    .set_position(x * CELL, y * CELL)
                )

    # Add target indicators on the right border for each compartment
    for ci in range(num_comp):
        ti = targets[ci]
        tc = colors[ti] if ti < len(colors) else C_WATER
        mid_y = ci * comp_h + comp_h // 2
        sprites.append(
            Sprite(pixels=[[tc]*CELL]*CELL, name=f"target_ind_{ci}",
                   collidable=False, tags=["target_ind", f"target_{ci}"], layer=2)
            .set_position((tank_w - 1) * CELL, mid_y * CELL)
        )

    # Add gates (closed by default) on partition walls
    # Each partition y = comp_i * comp_h for comp_i > 0
    for ci in range(num_comp - 1):
        gy = (ci + 1) * comp_h  # partition row (grid)
        # Left gate: x=1, Right gate: x=tank_w-2
        for gx, side in [(1, "left"), (tank_w - 2, "right")]:
            sprites.append(
                Sprite(pixels=[[C_GATE_CLOSED]*CELL]*CELL,
                       name=f"gate_{ci}_{side}",
                       collidable=True,
                       tags=["gate", f"gate_{ci}_{side}", "gate_closed"],
                       layer=2)
                .set_position(gx * CELL, gy * CELL)
            )

    # Cursor starts at first gate position (pixel coords)
    sprites.append(
        Sprite(pixels=_CURSOR_PIXELS, name="cursor",
               collidable=False, tags=["cursor"], layer=3)
        .set_position(1 * CELL, comp_h * CELL)
    )

    return Level(sprites=sprites, grid_size=(tank_w * CELL, total_h * CELL))


# ---------------------------------------------------------------------------
# Game class
# ---------------------------------------------------------------------------

class AquariumSorter(AugmentedGame):

    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: the base draws this level's rotation
        rng = random.Random(seed)
        self._step_counter = StepCounter(_STEP_BUDGETS[0])
        self._level_data: list[dict] = []
        levels: list[Level] = []

        for tank_w, num_comp, fish_per in _CONFIGS:
            data = _generate(tank_w, num_comp, fish_per, rng)
            self._level_data.append(data)
            levels.append(_build_level(data))

        self._fish_states: list[dict] = []  # current fish positions
        self._gate_states: dict[str, bool] = {}  # gate_name -> open
        self._cursor_pos: tuple[int, int] = (0, 0)
        self._history: list[tuple[list[dict], dict[str, bool], tuple[int, int]]] = []

        super().__init__(
            game_id="aquarium_sorter",
            levels=levels,
            camera=Camera(
                background=C_WATER,
                letter_box=C_BORDER,
                interfaces=[self._step_counter],
            ),
            available_actions=[1, 2, 3, 4, 5, 7],
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cursor(self) -> Sprite:
        return self.current_level.get_sprites_by_tag("cursor")[0]

    def _get_gates(self) -> list[Sprite]:
        return self.current_level.get_sprites_by_tag("gate")

    def _gate_at_cursor(self) -> Sprite | None:
        cx, cy = self._cursor_pos
        return next((g for g in self._get_gates() if g.x == cx and g.y == cy), None)

    def _set_gate(self, gate: Sprite, open_state: bool) -> None:
        self._gate_states[gate.name] = open_state
        if open_state:
            gate.set_collidable(False)
            gate.pixels = np.array([[C_GATE_OPEN]*CELL]*CELL, dtype=np.int8)
            gate._tags[:] = [t for t in gate.tags if t != "gate_closed"]
            if "gate_open" not in gate.tags:
                gate._tags.append("gate_open")
        else:
            gate.set_collidable(True)
            gate.pixels = np.array([[C_GATE_CLOSED]*CELL]*CELL, dtype=np.int8)
            gate._tags[:] = [t for t in gate.tags if t != "gate_open"]
            if "gate_closed" not in gate.tags:
                gate._tags.append("gate_closed")

    def _is_wall_or_closed_gate(self, x: int, y: int) -> bool:
        for s in self.current_level.get_sprites():
            if s.x == x and s.y == y:
                if "wall" in s.tags:
                    return True
                if "gate" in s.tags and "gate_closed" in s.tags:
                    return True
        return False

    def _advance_fish(self) -> None:
        """Move all fish one step. Fish at gate columns cross into adjacent
        compartments if the corresponding gate is open; otherwise they bounce."""
        data = self._level_data[self._current_level_index]
        tank_w = data["tank_w"]
        comp_h = data["comp_h"]
        num_comp = data["num_comp"]

        out = [dict(f) for f in self._fish_states]
        for fish in out:
            nx = fish["x"] + fish["dir"]
            ci = fish["comp"]

            if nx >= tank_w - 1:
                # Hit right wall: check gate_{ci}_right to cross DOWN to comp_{ci+1}
                gate_name = f"gate_{ci}_right"
                if ci < num_comp - 1 and self._gate_states.get(gate_name, False):
                    fish["comp"] = ci + 1
                    fish["x"] = tank_w - 2
                    fish["dir"] = -1  # reverse: now swimming left in new comp
                else:
                    fish["dir"] = -1  # bounce
            elif nx <= 0:
                # Hit left wall: check gate_{ci-1}_left to cross UP to comp_{ci-1}
                if ci > 0:
                    gate_name = f"gate_{ci - 1}_left"
                    if self._gate_states.get(gate_name, False):
                        fish["comp"] = ci - 1
                        fish["x"] = 1
                        fish["dir"] = 1  # reverse: now swimming right in new comp
                    else:
                        fish["dir"] = 1  # bounce
                else:
                    fish["dir"] = 1  # bounce off outer left wall
            else:
                fish["x"] = nx

        # Resolve permanent overlap: if two fish of different species land at the
        # same (comp, x, dir) they would move in lockstep forever and become
        # inseparable.  Nudge the later-indexed fish by one cell to break the tie.
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                fi, fj = out[i], out[j]
                if (fi["species"] != fj["species"] and
                        fi["comp"] == fj["comp"] and
                        fi["x"] == fj["x"] and
                        fi["dir"] == fj["dir"]):
                    nudge = fj["x"] + fj["dir"]
                    if 1 <= nudge <= tank_w - 2:
                        fj["x"] = nudge
                    else:
                        fj["dir"] = -fj["dir"]  # reverse at boundary instead

        self._fish_states = out

        # Update fish sprite positions (pixel coords)
        fish_sprites = self.current_level.get_sprites_by_tag("fish")
        for fs, fstate in zip(fish_sprites, self._fish_states):
            cy_top, _ = _comp_y_range_static(fstate["comp"], comp_h)
            fs.set_position(fstate["x"] * CELL, cy_top * CELL)

    def _gate_positions(self) -> list[tuple[int, int]]:
        """Return gate positions in pixel coords."""
        data = self._level_data[self._current_level_index]
        comp_h = data["comp_h"]
        num_comp = data["num_comp"]
        tank_w = data["tank_w"]
        pos = []
        for ci in range(num_comp - 1):
            gy = (ci + 1) * comp_h
            pos.append((1 * CELL, gy * CELL))
            pos.append(((tank_w - 2) * CELL, gy * CELL))
        return pos

    def _check_win(self) -> bool:
        data = self._level_data[self._current_level_index]
        targets = data["targets"]
        num_comp = data["num_comp"]
        # Check each compartment has only its target species
        for ci in range(num_comp):
            fish_in_comp = [f for f in self._fish_states if f["comp"] == ci]
            for f in fish_in_comp:
                if f["species"] != targets[ci]:
                    return False
        # All fish accounted for
        total_fish = sum(len([f for f in self._fish_states if f["comp"] == ci])
                         for ci in range(num_comp))
        return total_fish == len(self._fish_states)

    def _snapshot(self):
        return (
            [dict(f) for f in self._fish_states],
            dict(self._gate_states),
            self._cursor_pos,
        )

    def on_set_level(self, level: Level) -> None:
        self._history.clear()
        idx = self._current_level_index
        self._step_counter.reset(_STEP_BUDGETS[idx])
        data = self._level_data[idx]

        # Clear fish sprites from any previous call (idempotency)
        for sp in list(level.get_sprites_by_tag("fish")):
            level.remove_sprite(sp)

        # Reset all gates to closed (updates visual state + gate_states dict)
        for g in level.get_sprites_by_tag("gate"):
            self._set_gate(g, False)

        # Init fish states from level data
        self._fish_states = [dict(f) for f in data["fish"]]

        # Place fish sprites (pixel coords)
        comp_h = data["comp_h"]
        colors = data["colors"]
        for i, fish in enumerate(self._fish_states):
            cy_top, _ = _comp_y_range_static(fish["comp"], comp_h)
            fc = colors[fish["species"]] if fish["species"] < len(colors) else C_WATER
            level.add_sprite(
                Sprite(pixels=_fish_pixels(fc), name=f"fish_{i}",
                       collidable=False, tags=["fish", f"fish_{i}"], layer=2)
                .set_position(fish["x"] * CELL, cy_top * CELL)
            )

        # Cursor starts at first gate (pixel coords)
        gate_pos = self._gate_positions()
        if gate_pos:
            self._cursor_pos = gate_pos[0]
        else:
            self._cursor_pos = (CELL, CELL)
        self._cursor().set_position(*self._cursor_pos)

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self) -> None:
        self._step_counter.decrement()

        # --- Undo -------------------------------------------------------
        if self.action.id == GameAction.ACTION7:
            if self._history:
                fish_st, gate_st, cursor = self._history.pop()
                self._fish_states = fish_st
                self._cursor_pos = cursor
                self._cursor().set_position(*cursor)
                # Restore gate states
                for g in self._get_gates():
                    self._set_gate(g, gate_st.get(g.name, False))
                self._gate_states = dict(gate_st)
                # Restore fish sprites (pixel coords)
                fish_sprites = self.current_level.get_sprites_by_tag("fish")
                data = self._level_data[self._current_level_index]
                comp_h = data["comp_h"]
                for fs, fstate in zip(fish_sprites, self._fish_states):
                    cy_top, _ = _comp_y_range_static(fstate["comp"], comp_h)
                    fs.set_position(fstate["x"] * CELL, cy_top * CELL)
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # --- Toggle gate (ACTION5) -------------------------------------
        if self.action.id == GameAction.ACTION5:
            gate = self._gate_at_cursor()
            if gate is not None:
                self._history.append(self._snapshot())
                cur_state = self._gate_states.get(gate.name, False)
                self._set_gate(gate, not cur_state)
                # Advance fish
                self._advance_fish()
                if self._check_win():
                    self.next_level()
                    self.complete_action()
                    return
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        # --- Move cursor -----------------------------------------------
        gate_pos = self._gate_positions()
        if not gate_pos:
            self.complete_action()
            if not self._step_counter.steps_remaining:
                self.lose()
            return

        if self.action.id in (GameAction.ACTION1, GameAction.ACTION2,
                               GameAction.ACTION3, GameAction.ACTION4):
            cx, cy = self._cursor_pos
            # Cycle through gate positions
            idx = gate_pos.index((cx, cy)) if (cx, cy) in gate_pos else 0
            if self.action.id == GameAction.ACTION1:
                idx = (idx - 1) % len(gate_pos)
            elif self.action.id == GameAction.ACTION2:
                idx = (idx + 1) % len(gate_pos)
            elif self.action.id == GameAction.ACTION3:
                # Move to left-wall gates (pixel x = 1*CELL)
                left = [(x, y) for (x, y) in gate_pos if x == CELL]
                if left:
                    self._cursor_pos = left[0]
            elif self.action.id == GameAction.ACTION4:
                # Move to right-wall gates (pixel x != 1*CELL)
                right = [(x, y) for (x, y) in gate_pos if x != CELL]
                if right:
                    self._cursor_pos = right[0]
            if self.action.id in (GameAction.ACTION1, GameAction.ACTION2):
                self._cursor_pos = gate_pos[idx]
            self._cursor().set_position(*self._cursor_pos)

        self.complete_action()
        if not self._step_counter.steps_remaining:
            self.lose()


def _comp_y_range_static(comp_i: int, comp_h: int) -> tuple[int, int]:
    top = comp_i * comp_h
    return top + 1, top + comp_h - 1
