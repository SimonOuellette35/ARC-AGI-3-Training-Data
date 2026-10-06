"""color_sorter -- inspired by ar25.

One or two FIXED axes split the 64x64 board into 2 regions (a single
horizontal or single vertical axis) or 4 regions (both axes -- a cross).
Coloured shape tokens sit on either side of every axis at mirrored
positions.  A cursor rides along the axes; ACTION5 swaps the pair (or the
group) of tokens that straddle the axis at the cursor's current station.

Win when every region is a single colour and the regions' colours are all
different -- i.e. all colours have been sorted onto one side / into one
region.

Controls (available_actions = [1, 2, 3, 4, 5]):
  ACTION1 / ACTION2  -- move the cursor up / down along the VERTICAL axis
                        (selects a row station; swaps act left<->right).
  ACTION3 / ACTION4  -- move the cursor left / right along the HORIZONTAL
                        axis (selects a column station; swaps act top<->bottom).
  ACTION5           -- swap the tokens straddling the axis at the current
                        station.
Up/down are inert on a horizontal-only level, left/right on a vertical-only
level.

Every level is generated solvable: the solved board is scrambled with a
bounded sequence of legal swaps, so reversing that sequence always wins.
Layout, colours and the scramble are pure functions of (seed, level_index)
via AugmentedGame.level_rng.
"""

from utils.arc_game import AugmentedGame, clear_dynamic_sprites
from arcengine import Camera, Level, Sprite

BACKGROUND_COLOR = 0
AXIS_COLOR = 8
CURSOR_COLOR = 4

# Token colours -- excludes background (0), cursor (4) and axis (8).
PALETTE = [1, 2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15]

TOK = 7          # token icon is TOK x TOK pixels
CUR = 9          # cursor ring is CUR x CUR pixels
AXIS_HALF = 1    # axis line is 2*AXIS_HALF+1 pixels thick

# 7x7 icon patterns; one is picked per colour so tokens read as distinct shapes.
ICON_PATTERNS = [
    ["#######", "#######", "#######", "#######", "#######", "#######", "#######"],
    ["#######", "#.....#", "#.....#", "#.....#", "#.....#", "#.....#", "#######"],
    ["...#...", "..###..", ".#####.", "#######", ".#####.", "..###..", "...#..."],
    ["...#...", "...#...", "...#...", "#######", "...#...", "...#...", "...#..."],
    ["#.....#", "##...##", ".##.##.", "..###..", ".##.##.", "##...##", "#.....#"],
    ["#######", ".......", "#######", ".......", "#######", ".......", "#######"],
    ["#.#.#.#", "#.#.#.#", "#.#.#.#", "#.#.#.#", "#.#.#.#", "#.#.#.#", "#.#.#.#"],
    ["#.#.#.#", ".#.#.#.", "#.#.#.#", ".#.#.#.", "#.#.#.#", ".#.#.#.", "#.#.#.#"],
]


def _icon_pixels(color: int) -> list[list[int]]:
    pattern = ICON_PATTERNS[color % len(ICON_PATTERNS)]
    return [[color if ch == "#" else -1 for ch in row] for row in pattern]


def _cursor_pixels() -> list[list[int]]:
    px = [[-1] * CUR for _ in range(CUR)]
    for i in range(CUR):
        for j in range(CUR):
            if i < 2 or i >= CUR - 2 or j < 2 or j >= CUR - 2:
                px[i][j] = CURSOR_COLOR
    return px


def _level(config: dict) -> Level:
    return Level(sprites=[], grid_size=(64, 64), data={"config": config})


# axc / axr: pixel centre of the vertical / horizontal axis (None = no such axis).
# xcols: token top-left x positions (>=2 entries mirrored about axc when axc set).
# yrows: token top-left y positions (>=2 entries mirrored about axr when axr set).
# tokens sit at the full xcols x yrows grid.
LEVELS = [
    # 1) vertical axis, 3 lanes, 2 regions
    _level({"axc": 32, "axr": None, "xcols": [10, 47], "yrows": [8, 28, 48],
            "scramble": 6}),
    # 2) horizontal axis, 4 lanes, 2 regions
    _level({"axc": None, "axr": 32, "xcols": [6, 20, 37, 51], "yrows": [11, 46],
            "scramble": 8}),
    # 3) both axes -- 4 quadrants, two tokens each (8 total), 4 regions
    _level({"axc": 32, "axr": 32, "xcols": [8, 18, 39, 49], "yrows": [16, 41],
            "scramble": 10}),
    # 4) vertical axis, 5 lanes, 2 regions
    _level({"axc": 32, "axr": None, "xcols": [8, 49], "yrows": [3, 15, 27, 39, 51],
            "scramble": 14}),
    # 5) both axes -- 4 quadrants, 4 tokens each (16 total), 4 regions
    _level({"axc": 32, "axr": 32, "xcols": [8, 18, 39, 49], "yrows": [8, 18, 39, 49],
            "scramble": 20}),
    # 6) horizontal axis, 5 lanes, 2 regions
    _level({"axc": None, "axr": 32, "xcols": [3, 15, 27, 39, 51], "yrows": [8, 49],
            "scramble": 16}),
]


class ColorSorter(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: base rotates the board on set_level
        super().__init__(
            "color_sorter",
            LEVELS,
            Camera(0, 0, 64, 64, BACKGROUND_COLOR, BACKGROUND_COLOR, []),
            False,
            len(LEVELS),
            [1, 2, 3, 4, 5],
        )

    # ── level setup ─────────────────────────────────────────────────────────
    def on_set_level(self, level: Level) -> None:
        # Re-seating a level re-runs this hook on the same Level object: drop the
        # sprites the previous pass added before rebuilding them.
        clear_dynamic_sprites(level, tags=["cs_dyn"])

        cfg = level.get_data("config")
        self._axc = cfg["axc"]
        self._axr = cfg["axr"]
        self._xcols = sorted(cfg["xcols"])
        self._yrows = sorted(cfg["yrows"])

        # Cursor stations: vertical-axis rows, horizontal-axis columns.
        self._vstations = list(self._yrows) if self._axc is not None else []
        self._hstations = list(self._xcols) if self._axr is not None else []
        self._mode = "V" if self._vstations else "H"
        self._vi = 0
        self._hi = 0

        self._build_axes(level)
        self._build_tokens(level, cfg["scramble"])
        self._cursor = Sprite(
            pixels=_cursor_pixels(), name="cs_cursor", visible=True,
            collidable=False, layer=10, tags=["cs_dyn", "cs_cursor"],
        )
        level.add_sprite(self._cursor)
        self._place_cursor()

    def _build_axes(self, level: Level) -> None:
        if self._axc is not None:
            w = 2 * AXIS_HALF + 1
            level.add_sprite(Sprite(
                pixels=[[AXIS_COLOR] * w for _ in range(64)],
                name="cs_axis_v", visible=True, collidable=False, layer=-10,
                tags=["cs_dyn", "cs_axis"],
            ).set_position(self._axc - AXIS_HALF, 0))
        if self._axr is not None:
            h = 2 * AXIS_HALF + 1
            level.add_sprite(Sprite(
                pixels=[[AXIS_COLOR] * 64 for _ in range(h)],
                name="cs_axis_h", visible=True, collidable=False, layer=-10,
                tags=["cs_dyn", "cs_axis"],
            ).set_position(0, self._axr - AXIS_HALF))

    def _region_key(self, x: int, y: int) -> tuple[int, int]:
        kx = 0 if self._axc is None else (-1 if x < self._axc else 1)
        ky = 0 if self._axr is None else (-1 if y < self._axr else 1)
        return (kx, ky)

    def _build_tokens(self, level: Level, scramble: int) -> None:
        rng = self.level_rng("layout")

        kxs = [0] if self._axc is None else [-1, 1]
        kys = [0] if self._axr is None else [-1, 1]
        regions = [(kx, ky) for kx in kxs for ky in kys]
        colors = rng.sample(PALETTE, len(regions))
        region_color = dict(zip(regions, colors))

        self._tokens: list[Sprite] = []
        for x in self._xcols:
            for y in self._yrows:
                color = region_color[self._region_key(x, y)]
                s = Sprite(
                    pixels=_icon_pixels(color), name="cs_token", visible=True,
                    collidable=False, layer=0, tags=["cs_dyn", "cs_token"],
                )
                s.set_position(x, y)
                s._cs_color = color
                level.add_sprite(s)
                self._tokens.append(s)

        legal: list[tuple[str, int]] = []
        legal += [("V", y) for y in self._vstations]
        legal += [("H", x) for x in self._hstations]
        for _ in range(scramble):
            self._apply_swap(*rng.choice(legal))
        guard = 0
        while self._is_solved() and guard < 64:
            self._apply_swap(*rng.choice(legal))
            guard += 1

    # ── token lookup / swapping ─────────────────────────────────────────────
    def _find(self, x: int, y: int) -> Sprite | None:
        for s in self._tokens:
            if s.x == x and s.y == y:
                return s
        return None

    def _apply_swap(self, mode: str, station: int) -> None:
        if mode == "V":
            axis = self._xcols
            pair_at = lambda a, b: (self._find(a, station), self._find(b, station))
            move = lambda s, other: s.set_position(other, station)
        else:
            axis = self._yrows
            pair_at = lambda a, b: (self._find(station, a), self._find(station, b))
            move = lambda s, other: s.set_position(station, other)
        n = len(axis)
        for i in range(n // 2):
            a, b = axis[i], axis[n - 1 - i]
            sa, sb = pair_at(a, b)
            if sa is None or sb is None:
                continue
            move(sa, b)
            move(sb, a)

    # ── win test ───────────────────────────────────────────────────────────
    def _is_solved(self) -> bool:
        regions: dict[tuple[int, int], set[int]] = {}
        for s in self._tokens:
            regions.setdefault(self._region_key(s.x, s.y), set()).add(s._cs_color)
        if any(len(cs) != 1 for cs in regions.values()):
            return False
        picked = [next(iter(cs)) for cs in regions.values()]
        return len(set(picked)) == len(picked)

    # ── cursor ─────────────────────────────────────────────────────────────
    def _place_cursor(self) -> None:
        if self._mode == "V":
            cx = self._axc
            cy = self._vstations[self._vi] + TOK // 2
        else:
            cx = self._hstations[self._hi] + TOK // 2
            cy = self._axr
        x = max(0, min(64 - CUR, cx - CUR // 2))
        y = max(0, min(64 - CUR, cy - CUR // 2))
        self._cursor.set_position(x, y)

    # ── step ───────────────────────────────────────────────────────────────
    def step(self) -> None:
        a = self.action.id.value
        if a == 1 and self._vstations:
            self._mode = "V"
            self._vi = max(0, self._vi - 1)
        elif a == 2 and self._vstations:
            self._mode = "V"
            self._vi = min(len(self._vstations) - 1, self._vi + 1)
        elif a == 3 and self._hstations:
            self._mode = "H"
            self._hi = max(0, self._hi - 1)
        elif a == 4 and self._hstations:
            self._mode = "H"
            self._hi = min(len(self._hstations) - 1, self._hi + 1)
        elif a == 5:
            if self._mode == "V":
                self._apply_swap("V", self._vstations[self._vi])
            else:
                self._apply_swap("H", self._hstations[self._hi])

        self._place_cursor()

        if self._is_solved():
            self.next_level()

        self.complete_action()
