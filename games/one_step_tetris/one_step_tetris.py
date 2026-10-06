"""one_step_tetris -- inspired by Tetris, reduced to a single decisive drop.

A stack of blocks sits at the bottom of the board with gaps carved into it.
One or more polyominoes hang near the top.  Align them with the gaps, then
press ACTION5: every piece drops straight down at once.  If the drop completes
the level's required number of full rows the level is cleared; otherwise it is
lost.

Controls (available_actions = [1, 2, 3, 4, 5, 6]):
  ACTION3 / ACTION4  -- move the active piece one column left / right.  The
                        piece has no vertical freedom, so ACTION1 / ACTION2 do
                        nothing upright; they are offered only so the board's
                        per-seed rotation stays playable (the wrapper maps a
                        screen arrow onto whichever game arrow the rotation
                        turned it into).
  ACTION6 (click)    -- click on a piece to make it active; click on empty
                        space to slide the active piece so its left edge sits
                        at the clicked column.
  ACTION5           -- commit: drop every piece and judge the result.

Everything except the well positions is drawn per (seed, level_index) from
AugmentedGame.level_rng and is a pure function of it: the piece SHAPES and their
matching holes, the colour scheme (background, stack, active-piece highlight,
every idle piece), the start columns, and -- from level 4 on -- the start
heights, so pieces hang at different distances from the ground.  Multi-well
levels force pairwise-distinct shapes, so no piece can plug another's hole.
Levels 1-2 have 1-2 wells; levels 3-8 have >= 3.  Levels 7-8 leave only a
1-3 cell orientation hint in each hole -- you place the whole polyomino.

Every level stays solvable: each hole is directly reachable by a vertical drop
from its piece's solution column (piece.y == BASE_ROW - L is column-independent,
so the piece seats flush with no mid-descent collision); a wall across the
completed band makes any horizontal misalignment settle high and lose.
"""

import numpy as np

from utils.arc_game import AugmentedGame, clear_dynamic_sprites
from arcengine import Camera, Level, Sprite

W = H = 20
SCALE = min(64 // W, 64 // H)          # 3 px per cell
X_OFF = (64 - W * SCALE) // 2          # letterbox offset of the board in the frame
Y_OFF = (64 - H * SCALE) // 2
BASE_ROW = H - 1                       # the floor row every well bottoms out on

BACKGROUND_COLOR = 0
STACK_COLOR = 8
ACTIVE_COLOR = 4   # the piece currently accepting arrow moves

# Every colour on the board -- background, the stack, the active-piece
# highlight, and each idle piece -- is redrawn per (seed, level) from this pool
# in on_set_level via level_rng("colors"), so no two seeds look alike while the
# scheme stays a pure function of (seed, level).  The constants above are only
# the fall-back for an unseeded interactive game.
_COLOR_POOL = list(range(16))


def _piece_pixels(offsets: list[tuple[int, int]], color: int) -> np.ndarray:
    w = max(ox for ox, _ in offsets) + 1
    h = max(oy for _, oy in offsets) + 1
    px = np.full((h, w), -1, dtype=np.int8)
    for ox, oy in offsets:
        px[oy, ox] = color
    return px


# ── shapes / holes ────────────────────────────────────────────────────────
#
# A level RECIPE lists WELLS as (x0, cols, max_drop, max_h).  Per (seed, level)
# a concrete SPEC is drawn for each well: a per-column (drop, height) where the
# piece has `height` cells in that column and its lowest cell sits `drop` rows
# above the piece's own bottom line.  The matching notch is carved into the
# stack so the piece drops straight in and seats flush in every column at once
# (piece.y = BASE_ROW - L is column-independent -> no mid-descent collision).
# Every non-well column is walled from the top completed row down, so any
# horizontal misalignment strikes the wall, settles high, and leaves the target
# rows open -> loss.  With `min_notch` the draw keeps the pre-placed stack to a
# 1-3 cell orientation hint, so the level is "place the whole polyomino".

def _skyline_offsets(spec: list[tuple[int, int]]) -> list[list[int]]:
    L = max(d + h - 1 for d, h in spec)
    offs = [[dx, r]
            for dx, (d, h) in enumerate(spec)
            for r in range(L - d - h + 1, L - d + 1)]
    miny = min(r for _, r in offs)
    return sorted([[c, r - miny] for c, r in offs])


def _connected(offs: list[list[int]]) -> bool:
    s = {tuple(o) for o in offs}
    seen = {next(iter(s))}
    stk = list(seen)
    while stk:
        c, r = stk.pop()
        for nb in ((c + 1, r), (c - 1, r), (c, r + 1), (c, r - 1)):
            if nb in s and nb not in seen:
                seen.add(nb)
                stk.append(nb)
    return len(seen) == len(s)


def _drop_cells(x0: int, offs: list, start_y: int,
                occ: set[tuple[int, int]]) -> list[tuple[int, int]]:
    d = 0
    while all(start_y + oy + d + 1 < H and (x0 + ox, start_y + oy + d + 1) not in occ
              for ox, oy in offs):
        d += 1
    return [(x0 + ox, start_y + oy + d) for ox, oy in offs]


def _gen_spec(rng, cols: int, max_drop: int, max_h: int,
              min_notch: bool, min_req: int) -> list[tuple[int, int]]:
    """A random connected skyline spec for one well.

    Every column fills at least ``min_req`` cells down to the floor
    (drop + height >= min_req), so the completed band is >= min_req rows deep
    and the whole piece shape -- not just its bottom edge -- has to match.
    With ``min_notch`` the pre-placed notch is 1-3 cells: place the whole
    polyomino.  Never a plain rectangle.
    """
    for _ in range(600):
        drops = [rng.randint(0, max_drop) for _ in range(cols)]
        if min(drops) != 0:                       # anchor the piece to the floor
            continue
        heights = [rng.randint(1, max_h) for _ in range(cols)]
        spec = list(zip(drops, heights))
        if any(d + h < min_req for d, h in spec):
            continue
        if min_notch and not (1 <= sum(drops) <= 3):
            continue
        if cols > 1 and len(set(spec)) == 1:      # not a rectangle
            continue
        if not _connected(_skyline_offsets(spec)):
            continue
        return spec
    return [(0, max(1, min_req))] * cols


def _carve(wells: list) -> tuple[set, set]:
    notch: set[tuple[int, int]] = set()
    well_cols: set[int] = set()
    for x0, spec, _o in wells:
        for dx, (d, _h) in enumerate(spec):
            well_cols.add(x0 + dx)
            for y in range(BASE_ROW - d + 1, H):
                notch.add((x0 + dx, y))
    return notch, well_cols


def _solved_band(wells: list, notch: set, well_cols: set) -> tuple[int, int]:
    """Drop every piece at its solution column; return (ry0, required rows)."""
    occ = set(notch)
    for x0, _s, offs in sorted(wells, key=lambda w: w[0]):
        occ.update(_drop_cells(x0, offs, 0, occ))
    ry0 = BASE_ROW + 1
    for y in range(BASE_ROW, -1, -1):
        if all((c, y) in occ for c in well_cols):
            ry0 = y
        else:
            break
    return ry0, max(1, BASE_ROW - ry0 + 1)


def _perm_safe(wells: list, stack: set, required: int) -> bool:
    """True iff NO non-identity assignment of pieces to wells completes
    ``required`` rows -- i.e. every hole needs its own piece."""
    import itertools
    n = len(wells)
    sols = [w[0] for w in wells]
    offs = [w[2] for w in wells]
    pw = [max(c for c, _ in o) + 1 for o in offs]
    order = sorted(range(n), key=lambda k: sols[k])
    for perm in itertools.permutations(range(n)):
        if perm == tuple(range(n)):
            continue
        occ = set(stack)
        for k in order:
            pi = perm[k]
            x = min(max(0, sols[k]), W - pw[pi])
            occ.update(_drop_cells(x, offs[pi], 0, occ))
        if sum(1 for y in range(H)
               if all((c, y) in occ for c in range(W))) >= required:
            return False
    return True


def _build_layout(rng, rec: dict):
    """(wells, stack, well_cols, ry0, required) -- redrawn until every hole is
    deep enough and no piece fits another's hole."""
    last = None
    for _ in range(60):
        wells, seen, ok = [], [], True
        for (x0, cols, mxd, mxh) in rec["wells"]:
            chosen = None
            for _ in range(80):
                spec = _gen_spec(rng, cols, mxd, mxh,
                                 rec["min_notch"], rec["min_req"])
                offs = _skyline_offsets(spec)
                shape = tuple(map(tuple, offs))
                sig = (len(offs), max(c for c, _ in offs) + 1)
                if any(sh == shape or si == sig for sh, si in seen):
                    continue
                chosen = (x0, spec, offs)
                seen.append((shape, sig))
                break
            if chosen is None:
                ok = False
                break
            wells.append(chosen)
        if not ok:
            continue

        notch, well_cols = _carve(wells)
        ry0, required = _solved_band(wells, notch, well_cols)
        if required < rec["min_req"]:
            continue
        stack = set(notch)
        for x in range(W):
            if x not in well_cols:
                for y in range(ry0, H):
                    stack.add((x, y))
        last = (wells, stack, well_cols, ry0, required)
        if len(wells) > 1 and not _perm_safe(wells, stack, required):
            continue
        return last
    if last is not None:
        return last
    # extremely defensive: no distinct set found in 60 tries -- take any.
    wells = [(x0, _gen_spec(rng, cols, mxd, mxh, rec["min_notch"], rec["min_req"]),
              None) for (x0, cols, mxd, mxh) in rec["wells"]]
    wells = [(x0, spec, _skyline_offsets(spec)) for x0, spec, _ in wells]
    notch, well_cols = _carve(wells)
    ry0, required = _solved_band(wells, notch, well_cols)
    stack = set(notch)
    for x in range(W):
        if x not in well_cols:
            for y in range(ry0, H):
                stack.add((x, y))
    return wells, stack, well_cols, ry0, required


def _level(recipe: dict) -> Level:
    return Level(sprites=[], grid_size=(W, H), data={"config": recipe})


# Recipe: wells [(x0, cols, max_drop, max_h)] + min_req (hole depth = how many
# rows clear) + flags.  Shapes/holes, colours, start columns and (from L4) start
# heights are all drawn per (seed, level).  Levels 1-2 have 1-2 wells; 3-8 have
# >= 3.  Multi-well levels redraw until no piece can plug another's hole.
LEVELS = [
    # 1) one flat-bottomed 4-wide piece
    _level({"wells": [(8, 4, 0, 2)], "min_req": 1,
            "vary_h": False, "min_notch": False}),
    # 2) two distinct pieces, widths 3 and 4
    _level({"wells": [(3, 3, 1, 3), (11, 4, 1, 3)], "min_req": 2,
            "vary_h": False, "min_notch": False}),
    # 3) three wells (2/3/4 wide)
    _level({"wells": [(2, 2, 1, 3), (7, 3, 1, 3), (13, 4, 2, 3)], "min_req": 2,
            "vary_h": False, "min_notch": False}),
    # 4) three wells + varied start heights
    _level({"wells": [(2, 3, 2, 4), (9, 3, 2, 4), (15, 4, 2, 4)], "min_req": 3,
            "vary_h": True, "min_notch": False}),
    # 5) three wider wells, deeper holes
    _level({"wells": [(2, 3, 2, 4), (8, 4, 2, 4), (14, 4, 2, 4)], "min_req": 3,
            "vary_h": True, "min_notch": False}),
    # 6) four wells
    _level({"wells": [(1, 2, 1, 4), (5, 3, 2, 4), (10, 4, 2, 4), (16, 3, 2, 4)],
            "min_req": 3, "vary_h": True, "min_notch": False}),
    # 7) three tall polyominoes, only a 1-3 cell orientation hint each
    _level({"wells": [(2, 3, 2, 6), (8, 4, 2, 6), (15, 4, 2, 6)], "min_req": 4,
            "vary_h": True, "min_notch": True}),
    # 8) four tall polyominoes to place, almost from scratch
    _level({"wells": [(1, 2, 2, 6), (5, 3, 2, 6), (10, 3, 2, 6), (15, 4, 2, 6)],
            "min_req": 4, "vary_h": True, "min_notch": True}),
]


class OneStepTetris(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: base rotates the board on set_level
        super().__init__(
            "one_step_tetris",
            LEVELS,
            Camera(0, 0, W, H, BACKGROUND_COLOR, BACKGROUND_COLOR, []),
            False,
            len(LEVELS),
            [1, 2, 3, 4, 5, 6],
        )

    # ── level setup ────────────────────────────────────────────────────────
    def on_set_level(self, level: Level) -> None:
        clear_dynamic_sprites(level, tags=["ost_dyn"])
        rec = level.get_data("config")
        rp = self.level_rng("start")
        cc = self.level_rng("colors")

        # shapes / holes / walls: redrawn until every hole is deep enough and no
        # piece fits another's hole.
        wells, stack, well_cols, ry0, required = _build_layout(
            self.level_rng("shape"), rec)
        self._req = required
        self._stack = stack
        self._before_full = self._count_full(stack)
        self._committed = False
        self._active = 0

        # 5) per-(seed, level) colour augmentation -- distinct indices for the
        #    background, the stack, the active highlight and every piece.
        pool = list(_COLOR_POOL)
        cc.shuffle(pool)
        bg, self._stack_color, self._active_color = pool[0], pool[1], pool[2]
        piece_palette = pool[3:]
        if self.camera is not None:
            self.camera.background = bg
            self.camera.letter_box = bg

        sp = np.full((H, W), -1, dtype=np.int8)
        for x, y in stack:
            sp[y, x] = self._stack_color
        level.add_sprite(Sprite(
            pixels=sp, name="ost_stack", visible=True, collidable=False,
            layer=0, tags=["ost_dyn"],
        ).set_position(0, 0))

        # 6) place the pieces.  Each piece spawns inside its own horizontal
        #    "territory" (split at the midpoints between neighbouring wells) so
        #    no two pieces ever share a cell at spawn -- start column is random
        #    within it but never the solution.  From L4 the start height is also
        #    random, so pieces hang at different distances from the ground.
        rw = rec["wells"]
        self._pieces = []
        for i, (x0, _spec, offs) in enumerate(wells):
            toffs = [tuple(o) for o in offs]
            L = max(oy for _, oy in toffs)
            pw = max(ox for ox, _ in toffs) + 1
            left_split = -1 if i == 0 else (rw[i - 1][0] + rw[i - 1][1] - 1 + x0) // 2
            right_split = W - 1 if i == len(wells) - 1 else \
                (x0 + rw[i][1] - 1 + rw[i + 1][0]) // 2
            lo = max(0, left_split + 1, x0 - 3)
            hi = min(W - pw, right_split - pw + 1, x0 + 3)
            if hi < lo:
                lo = hi = max(0, min(x0, W - pw))
            choices = [x for x in range(lo, hi + 1) if x != x0] \
                or [x for x in range(lo, hi + 1)] or [x0]
            sx = rp.choice(choices)
            cap = max(0, ry0 - L - 1)
            sy = rp.randint(0, cap) if rec["vary_h"] else 0
            idle = piece_palette[i % len(piece_palette)]
            spr = Sprite(
                pixels=_piece_pixels(toffs, idle), name="ost_piece",
                visible=True, collidable=False, layer=5, tags=["ost_dyn"],
            )
            level.add_sprite(spr)
            self._pieces.append({
                "offsets": toffs, "x": sx, "y": sy,
                "color": idle, "sprite": spr, "solution_x": x0,
            })
        self._render_pieces()

    # ── geometry helpers ──────────────────────────────────────────────────
    @staticmethod
    def _width(p: dict) -> int:
        return max(ox for ox, _ in p["offsets"]) + 1

    def _cells(self, p: dict, x: int | None = None, y: int | None = None) -> list[tuple[int, int]]:
        px = p["x"] if x is None else x
        py = p["y"] if y is None else y
        return [(px + ox, py + oy) for ox, oy in p["offsets"]]

    @staticmethod
    def _count_full(occ: set[tuple[int, int]]) -> int:
        return sum(1 for y in range(H) if all((x, y) in occ for x in range(W)))

    # ── rendering ─────────────────────────────────────────────────────────
    def _render_pieces(self) -> None:
        for i, p in enumerate(self._pieces):
            color = self._active_color if i == self._active else p["color"]
            p["sprite"].pixels = _piece_pixels(p["offsets"], color)
            p["sprite"].set_position(p["x"], p["y"])

    # ── controls ──────────────────────────────────────────────────────────
    #
    # Pieces are suspended and pass freely through each other while being
    # arranged; they only interact on the ACTION5 drop.  Movement is a plain
    # horizontal clamp to the board -- the spawn height keeps every piece above
    # the wall band, so nothing can wedge it.
    def _move(self, dx: int) -> None:
        p = self._pieces[self._active]
        p["x"] = max(0, min(W - self._width(p), p["x"] + dx))
        self._render_pieces()

    def _click(self, gx: int, gy: int) -> None:
        for i, p in enumerate(self._pieces):
            if (gx, gy) in self._cells(p):
                self._active = i
                self._render_pieces()
                return
        p = self._pieces[self._active]
        p["x"] = min(max(gx, 0), W - self._width(p))
        self._render_pieces()

    def _commit(self) -> None:
        occ = set(self._stack)
        order = sorted(range(len(self._pieces)),
                       key=lambda i: (self._pieces[i]["x"], self._pieces[i]["y"]))
        for i in order:
            p = self._pieces[i]
            d = 0
            while all(cy + 1 < H and (cx, cy + 1) not in occ
                      for cx, cy in self._cells(p, y=p["y"] + d)):
                d += 1
            p["y"] += d
            occ.update(self._cells(p))
        self._render_pieces()

        if self._count_full(occ) - self._before_full >= self._req:
            self.next_level()
        else:
            self.lose()

    # ── step ──────────────────────────────────────────────────────────────
    def step(self) -> None:
        a = self.action.id.value
        if not self._committed:
            if a == 3:
                self._move(-1)
            elif a == 4:
                self._move(1)
            elif a == 6:
                data = self.action.data or {}
                if "x" in data and "y" in data:
                    self._click((int(data["x"]) - X_OFF) // SCALE,
                                (int(data["y"]) - Y_OFF) // SCALE)
            elif a == 5:
                self._committed = True
                self._commit()
        self.complete_action()
