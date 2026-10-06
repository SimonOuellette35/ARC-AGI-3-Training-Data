"""ball_sifting -- sift falling balls down through tiers of walls-with-holes.

Balls rest on movable paddles.  Slide the SELECTED paddle, then press ACTION5 to
open it: every ball on it falls straight down.  A ball that is lined up with the
hole in the wall below passes through and lands on the next tier's paddle; a
ball that misses the hole -- or overshoots a paddle that is not there to catch
it -- is trapped on top of a wall and the level is lost.  With several balls on
one paddle they all drop together, so part of the puzzle is nudging them
side-by-side into a cluster that fits through the hole.  Win when every ball has
dropped past the last wall into the basin.

Controls (available_actions = [1, 2, 3, 4, 5, 6, 7]):
  ACTION6 (click)    the fast move: click a paddle to select it; click anywhere
                     else to jump the selected paddle's centre to that column.
  ACTION3 / ACTION4  fine-nudge the selected paddle one column left / right.
                     ACTION1 / ACTION2 are inert upright -- offered only so the
                     board's per-seed rotation stays playable (the wrapper maps
                     a screen arrow onto whichever game arrow rotation made it).
  ACTION7           select the next paddle down (wraps).
  ACTION5           open the selected paddle -- drop its ball(s) one tier.

Layout (hole and paddle x-positions, every colour) is drawn per
(seed, level_index) from AugmentedGame.level_rng.  Each hole is 3*B + 2 wide, so
a full cluster of the level's B balls fits through it, and every level is
solvable one tier at a time.
"""

import numpy as np

from utils.arc_game import AugmentedGame, clear_dynamic_sprites
from arcengine import Camera, Level, Sprite

BOARD = 64
PW = 14                 # paddle width -- holds a cluster of up to 4 balls
TOP = 4
BOTTOM = 54             # deepest a wall is placed
FLOOR_Y = 62            # 2-row basin floor at rows 62-63
PAD_CLAMP = BOARD - PW  # 50 -- max paddle x
PAD_STEP = 1            # columns an arrow press nudges the paddle (fine control;
                       #   a click jumps it anywhere instantly)
FALL_SPEED = 3          # rows a ball drops per animation frame

_POOL = list(range(16))

# N tiers (N walls, N paddles), B balls.
_CFG = [
    {"N": 2, "B": 1},
    {"N": 3, "B": 1},
    {"N": 3, "B": 2},
    {"N": 4, "B": 2},
    {"N": 4, "B": 3},
    {"N": 5, "B": 3},
]

LEVELS = [Level(sprites=[], grid_size=(BOARD, BOARD), data={"config": c})
          for c in _CFG]


class BallSifting(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: base rotates the board on set_level
        super().__init__(
            "ball_sifting",
            LEVELS,
            Camera(0, 0, BOARD, BOARD, 0, 0, []),
            False,
            len(LEVELS),
            [1, 2, 3, 4, 5, 6, 7],
        )

    # ── level setup ───────────────────────────────────────────────────────
    def on_set_level(self, level: Level) -> None:
        clear_dynamic_sprites(level, tags=["bsift"])
        cfg = level.get_data("config")
        N, B = cfg["N"], cfg["B"]
        self._N = N
        self._hw = 3 * B + 2          # a cluster of all B balls fits any hole
        rl = self.level_rng("layout")
        rc = self.level_rng("colors")

        tier_h = (BOTTOM - TOP) // N
        self._wy = [TOP + (k + 1) * tier_h for k in range(N)]
        self._padY = [TOP + 2] + [self._wy[k - 1] + 5 for k in range(1, N)]
        # bounds keep any hole reachable by a paddle clamped to [0, PAD_CLAMP]
        # for any spawn offset, and a full B-ball cluster placeable inside it.
        self._hx = [rl.randint(8, 44) for _ in range(N)]

        pool = list(_POOL)
        rc.shuffle(pool)
        self._c_bg, self._c_wall, self._c_pad, self._c_sel, self._c_floor = pool[:5]
        ball_cols = pool[5:5 + B]
        if self.camera is not None:
            self.camera.background = self._c_bg
            self.camera.letter_box = self._c_bg

        for k in range(N):
            px = np.full((2, BOARD), self._c_wall, dtype=np.int8)
            px[:, self._hx[k]:self._hx[k] + self._hw] = -1
            level.add_sprite(Sprite(
                pixels=px, name="bsift_wall", visible=True, collidable=False,
                layer=0, tags=["bsift"],
            ).set_position(0, self._wy[k]))
        level.add_sprite(Sprite(
            pixels=np.full((2, BOARD), self._c_floor, dtype=np.int8),
            name="bsift_floor", visible=True, collidable=False, layer=0,
            tags=["bsift"],
        ).set_position(0, FLOOR_Y))

        self._pads = []
        for k in range(N):
            spr = Sprite(
                pixels=np.full((2, PW), self._c_pad, dtype=np.int8),
                name="bsift_pad", visible=True, collidable=False, layer=1,
                tags=["bsift"],
            )
            level.add_sprite(spr)
            self._pads.append({"x": rl.randint(0, PAD_CLAMP), "sprite": spr})

        self._balls = []
        for i, k in enumerate(sorted(rl.sample(range(N), B))):
            dx = rl.randint(0, PW - 3)
            spr = Sprite(
                pixels=np.full((3, 3), ball_cols[i], dtype=np.int8),
                name="bsift_ball", visible=True, collidable=False, layer=2,
                tags=["bsift"],
            )
            level.add_sprite(spr)
            self._balls.append({
                "x": self._pads[k]["x"] + dx, "y": self._padY[k] - 3,
                "on": k, "dx": dx, "color": ball_cols[i], "sprite": spr,
            })

        self._sel = self._balls[0]["on"]
        self._falling: list[dict] = []
        self._solid: set[tuple[int, int]] = set()
        self._render()

    # ── rendering ────────────────────────────────────────────────────────
    def _render(self) -> None:
        dropping = bool(self._falling)
        for k, p in enumerate(self._pads):
            sel = k == self._sel
            p["sprite"].pixels = np.full(
                (2, PW), self._c_sel if sel else self._c_pad, dtype=np.int8)
            p["sprite"].set_position(p["x"], self._padY[k])
            p["sprite"].set_visible(not (sel and dropping))
        for b in self._balls:
            b["sprite"].set_position(b["x"], b["y"])

    # ── geometry ─────────────────────────────────────────────────────────
    @staticmethod
    def _ball_cells(b: dict) -> list[tuple[int, int]]:
        return [(b["x"] + dx, b["y"] + dy) for dx in range(3) for dy in range(3)]

    def _set_pad_x(self, k: int, x: int) -> None:
        nx = max(0, min(PAD_CLAMP, int(x)))
        self._pads[k]["x"] = nx
        for b in self._balls:
            if b["on"] == k:
                b["x"] = nx + b["dx"]
        self._render()

    def _click(self, gx: int, gy: int) -> None:
        for k, p in enumerate(self._pads):
            if p["x"] <= gx < p["x"] + PW and self._padY[k] <= gy <= self._padY[k] + 1:
                self._sel = k
                self._render()
                return
        self._set_pad_x(self._sel, gx - PW // 2)

    # ── the drop ─────────────────────────────────────────────────────────
    def _build_solid(self) -> set[tuple[int, int]]:
        s: set[tuple[int, int]] = set()
        for k in range(self._N):
            wy = self._wy[k]
            for x in range(BOARD):
                if self._hx[k] <= x < self._hx[k] + self._hw:
                    continue
                s.add((x, wy))
                s.add((x, wy + 1))
        for k, p in enumerate(self._pads):
            if k == self._sel:
                continue
            for x in range(p["x"], p["x"] + PW):
                s.add((x, self._padY[k]))
                s.add((x, self._padY[k] + 1))
        for x in range(BOARD):
            s.add((x, FLOOR_Y))
            s.add((x, FLOOR_Y + 1))
        return s

    def _begin_drop(self) -> bool:
        drop = [b for b in self._balls if b["on"] == self._sel]
        if not drop:
            return False
        changed = True
        while changed:                       # a ball stacked on a dropper rides down
            changed = False
            for b in self._balls:
                if b in drop:
                    continue
                if any(b["y"] + 3 == d["y"] and b["x"] < d["x"] + 3
                       and d["x"] < b["x"] + 3 for d in drop):
                    drop.append(b)
                    changed = True
        for b in drop:
            b["on"] = -2
        self._falling = drop
        self._solid = self._build_solid()
        self._render()
        return True

    def _fall_substep(self) -> None:
        """Advance every falling ball by one row (walls are 2px, so we never skip
        a row -- speed comes from doing several of these per rendered frame)."""
        rest: set[tuple[int, int]] = set()
        for b in self._balls:
            if b["on"] != -2:
                rest.update(self._ball_cells(b))
        fall_cells: set[tuple[int, int]] = set()
        for b in self._falling:
            fall_cells.update(self._ball_cells(b))

        settled = []
        for b in sorted(self._falling, key=lambda b: -b["y"]):
            own = set(self._ball_cells(b))
            others = fall_cells - own
            blocked = any(
                b["y"] + 3 > BOARD - 1
                or (b["x"] + dx, b["y"] + 3) in self._solid
                or (b["x"] + dx, b["y"] + 3) in rest
                or (b["x"] + dx, b["y"] + 3) in others
                for dx in range(3)
            )
            if blocked:
                settled.append(b)
                rest.update(own)
                fall_cells -= own
            else:
                fall_cells -= own
                b["y"] += 1
                fall_cells.update(self._ball_cells(b))
        for b in settled:
            self._falling.remove(b)

    def _fall_tick(self) -> None:
        for _ in range(FALL_SPEED):
            if not self._falling:
                break
            self._fall_substep()
        self._render()
        if not self._falling:
            self._resolve()

    def _resolve(self) -> None:
        lose = False
        for b in sorted((x for x in self._balls if x["on"] == -2),
                        key=lambda b: -b["y"]):
            if b["y"] >= self._wy[-1]:            # past the last wall -> basin
                b["on"] = -1
                continue
            sup = None
            for dx in range(3):
                cx, cy = b["x"] + dx, b["y"] + 3
                for k, p in enumerate(self._pads):
                    if k != self._sel and self._padY[k] == cy \
                            and p["x"] <= cx < p["x"] + PW:
                        sup = ("pad", k)
                        break
                if sup:
                    break
                for ob in self._balls:
                    if ob is not b and ob["y"] == cy and ob["x"] <= cx < ob["x"] + 3:
                        sup = ("ball", ob)
                        break
                if sup:
                    break
            if sup is None:                       # trapped on top of a wall
                lose = True
            elif sup[0] == "pad":
                b["on"] = sup[1]
                b["dx"] = b["x"] - self._pads[sup[1]]["x"]
            else:
                ob = sup[1]
                if ob["on"] == -1:
                    b["on"] = -1
                elif ob["on"] >= 0:
                    b["on"] = ob["on"]
                    b["dx"] = b["x"] - self._pads[ob["on"]]["x"]
                else:
                    lose = True

        self._render()
        if lose:
            self.complete_action()
            self.lose()
            return
        if all(b["on"] == -1 for b in self._balls):
            self.complete_action()
            self.next_level()
            return
        on_pads = [b["on"] for b in self._balls if b["on"] >= 0]
        if on_pads:
            self._sel = min(on_pads)
        self._render()
        self.complete_action()

    # ── step ─────────────────────────────────────────────────────────────
    def step(self) -> None:
        if self._falling:
            self._fall_tick()
            return
        a = self.action.id.value
        if a == 3:
            self._set_pad_x(self._sel, self._pads[self._sel]["x"] - PAD_STEP)
        elif a == 4:
            self._set_pad_x(self._sel, self._pads[self._sel]["x"] + PAD_STEP)
        elif a == 7:
            self._sel = (self._sel + 1) % self._N
            self._render()
        elif a == 6:
            d = self.action.data or {}
            if "x" in d and "y" in d:
                self._click(int(d["x"]), int(d["y"]))
        elif a == 5:
            if self._begin_drop():
                return
        self.complete_action()
