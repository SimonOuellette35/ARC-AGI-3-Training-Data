"""procgen_adapter.py — Wraps selected ProcGen environments as ARCBaseGame-compatible objects.

Why these ProcGen games fit ARC-AGI-3:
  - Native 64×64 RGB output — no scaling needed, preserving all visual detail.
  - Turn-based step interface: one discrete action → one frame, matching ARC-AGI-3.
  - Goal inference required: agent must infer the win condition from visual state alone.
  - Three games selected for grid-reasoning alignment:
      maze    — navigate a generated maze to the exit (top-down, grid-aligned)
      miner   — collect diamonds and reach the exit, avoid falling boulders (Boulderdash)
      heist   — collect coloured keys, unlock matching gates, reach the exit

Frame pipeline:
  ProcGen obs (64×64×3 RGB uint8)
  → nearest-neighbour quantise to ARC 16-colour palette in RGB space
  → 64×64 uint8 ndarray of palette indices

Action mapping (4-direction layout, consistent with all other ARC-AGI-3 games):
  ACTION1 (up/w)    → ProcGen 5  (up)
  ACTION2 (down/s)  → ProcGen 3  (down)
  ACTION3 (left/a)  → ProcGen 1  (left)
  ACTION4 (right/d) → ProcGen 7  (right)
  ACTION5 (e/space) → ProcGen 4  (noop — no meaningful button in these games)
  ACTION6           → ProcGen 4  (noop)
  ACTION7 (z/u)     → ProcGen 4  (noop — no undo in ProcGen)

  ProcGen joystick grid layout (Discrete 15):
    0=↙  1=←  2=↖  /  3=↓  4=·  5=↑  /  6=↘  7=→  8=↗  /  9-14: same + D-button

WIN / GAME_OVER semantics:
  terminated=True AND reward >= _COMPLETION_BONUS  →  GameState.WIN  (reached exit / goal)
  terminated=True AND reward <  _COMPLETION_BONUS  →  GameState.GAME_OVER (death)
  truncated=True                                   →  GameState.GAME_OVER (budget)

  The threshold, rather than ``reward > 0``, is load-bearing for miner: gems pay
  +1 WITHOUT ending the episode, so an episode ending on the same step a gem is
  picked up (crushed by the boulder that gem was propping up, or truncated at
  ProcGen's 1000-step timeout) terminates with reward 1 -- a loss that
  ``reward > 0`` would report as a WIN. Only the completion bonus means "won".

Level concept:
  procgen_maze and procgen_miner use curated absolute seeds (7 tiers each,
  sorted by floor-pixel / boulder count) so the difficulty is always monotone
  regardless of the base seed.
  Other games use set_level(idx) which maps to start_level = base_seed + within_tier_seed.

Installation:
  pip install procgen

Usage:
    from adapters import ProcGenAdapter
    from arcengine import GameAction, ActionInput, GameState

    game = ProcGenAdapter("maze", seed=0)
    game.set_level(0)
    result = game.perform_action(ActionInput(id=GameAction.ACTION1))
    print(result.state)           # GameState.NOT_FINISHED
    print(result.frame[0].shape)  # (64, 64)
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
from scipy import ndimage

# "Warning: early reset ignored" is a plain Python print() in gym3/interop.py,
# emitted when env.reset() is called mid-episode.  Suppress it by redirecting
# sys.stdout during seed-validation resets.
_null_stream = io.StringIO()


@contextlib.contextmanager
def _quiet_stdout():
    """Suppress Python-level stdout for the duration of the block."""
    with contextlib.redirect_stdout(_null_stream):
        yield

from arcengine import ActionInput, GameAction, GameState, FrameDataRaw

from adapters.base import BaseAdapter


# ---------------------------------------------------------------------------
# ARC 16-colour palette — RGB values derived from arc_agi.rendering.COLOR_MAP
#
#   0  #FFFFFF  white          8  #F93C31  red
#   1  #CCCCCC  light gray     9  #1E93FF  blue
#   2  #999999  mid gray      10  #88D8F1  light blue
#   3  #666666  dark gray     11  #FFDC00  yellow
#   4  #333333  very dark     12  #FF851B  orange
#   5  #000000  black         13  #921231  dark red
#   6  #E53AA3  magenta       14  #4FCC30  green
#   7  #FF7BCC  light pink    15  #A356D6  purple
# ---------------------------------------------------------------------------

_ARC_PALETTE_RGB: np.ndarray = np.array(
    [
        [255, 255, 255],  #  0  white
        [204, 204, 204],  #  1  light gray
        [153, 153, 153],  #  2  mid gray
        [102, 102, 102],  #  3  dark gray
        [ 51,  51,  51],  #  4  very dark gray
        [  0,   0,   0],  #  5  black
        [229,  58, 163],  #  6  magenta
        [255, 123, 204],  #  7  light pink
        [249,  60,  49],  #  8  red
        [ 30, 147, 255],  #  9  blue
        [136, 216, 241],  # 10  light blue
        [255, 220,   0],  # 11  yellow
        [255, 133,  27],  # 12  orange
        [146,  18,  49],  # 13  dark red
        [ 79, 204,  48],  # 14  green
        [163,  86, 214],  # 15  purple
    ],
    dtype=np.int32,
)


def _rgb_to_arc_frame(rgb: np.ndarray) -> np.ndarray:
    """Quantise a (64, 64, 3) RGB uint8 frame to a (64, 64) ARC palette index array.

    Uses nearest-neighbour L2 distance in RGB space — same strategy as the ALE
    adapter, appropriate for ProcGen's sprite-based pixel art.
    """
    pixels = rgb.reshape(-1, 3).astype(np.int32)                   # (4096, 3)
    diff = pixels[:, None, :] - _ARC_PALETTE_RGB[None, :, :]       # (4096, 16, 3)
    dist = (diff * diff).sum(axis=-1)                               # (4096, 16)
    return dist.argmin(axis=-1).reshape(64, 64).astype(np.uint8)   # (64, 64)


# Palette indices for the reconstructed maze.
_MAZE_PLAYER_IDX = 9      # blue   -- the agent cell
_MAZE_GOAL_IDX = 14       # green  -- the goal cell
_MAZE_FLOOR_IDX = 5       # black  -- a navigable (corridor) cell
_MAZE_WALL_IDX = 2        # gray   -- a wall cell (fixed, so the repainted palette is deterministic)
_MAZE_MIN_PITCH = 3.5     # below this the maze has too many cells for 64px to render cleanly
                          # (corridors < ~4px alias/merge) -> reconstruction is unreliable; leave raw


def _maze_period(edges: np.ndarray, lo: float = 2.0, hi: float = 7.0, step: float = 0.02):
    """Recover the grid pitch P and offset O from a set of wall/corridor edge positions.

    The maze is a regular grid, so all edges sit on grid lines O + i*P. The period whose
    complex exponential sum has the largest magnitude is the pitch; its phase gives O."""
    edges = np.asarray(edges, dtype=np.float64)
    if edges.size < 4:
        return None, None
    periods = np.arange(lo, hi, step)
    z = np.exp(2j * np.pi * np.outer(1.0 / periods, edges)).sum(axis=1)
    i = int(np.argmax(np.abs(z)))
    P = float(periods[i])
    O = float((-np.angle(z[i]) / (2 * np.pi) * P) % P)
    return P, O


def _postprocess_maze(rgb: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Reconstruct procgen_maze on its own grid and re-render it as CLEAN, grid-aligned cells.

    procgen scales an N x N maze grid to fit 64px, so cells land on NON-integer pixel
    boundaries (~4.3px easy, ~2.55px hard) and the player/goal sprites are small animated
    blobs that re-quantise differently every step -- impossible to align by pixel-painting.
    Instead we (1) recover the grid pitch+offset from the wall/corridor edge periodicity,
    (2) classify each grid cell as wall / floor (majority navigable) / agent / goal, and
    (3) repaint each cell as a solid block. The result is a crisp maze whose agent and goal
    are full grid cells that move cleanly cell-to-cell with no lateral jitter.

    Detect by distinctive RGB: cheese = bright orange (253,155,37); mouse = light bluish-gray
    (187,203,204), where B>=R, unlike the brown tan walls where R >> B."""
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)
    tan = (r > 150) & (g > 100) & (b > 50) & (r - b > 40)            # brown maze walls
    nav = ~tan                                                       # navigable: corridor + sprites
    cheese = (r > 190) & (g > 110) & (b < 110) & (r - b > 110)       # goal: orange cheese (+edges)
    mouse = (b >= r - 8) & (r > 115) & (g > 115) & (b > 115)         # player: light bluish-gray mouse

    # grid pitch/offset from the periodicity of the navigable<->wall edges in x and y
    nx = nav.astype(np.int8)
    tx = np.concatenate([np.where(np.diff(row))[0] + 0.5 for row in nx]) if nx.any() else np.array([])
    ty = np.concatenate([np.where(np.diff(col))[0] + 0.5 for col in nx.T]) if nx.any() else np.array([])
    Px, Ox = _maze_period(tx)
    Py, Oy = _maze_period(ty)
    if Px is None or Py is None or Px < _MAZE_MIN_PITCH or Py < _MAZE_MIN_PITCH:
        return frame                       # grid undetectable or too dense for 64px; leave raw

    out = np.full_like(frame, _MAZE_WALL_IDX)        # fixed wall idx -> deterministic palette

    def _span(O, P, k, n):
        return max(0, int(round(O + k * P))), min(n, int(round(O + (k + 1) * P)))

    for j in range(int(np.floor(-Oy / Py)), int(np.ceil((64 - Oy) / Py))):
        y0, y1 = _span(Oy, Py, j, 64)
        if y1 <= y0:
            continue
        for i in range(int(np.floor(-Ox / Px)), int(np.ceil((64 - Ox) / Px))):
            x0, x1 = _span(Ox, Px, i, 64)
            if x1 <= x0:
                continue
            if nav[y0:y1, x0:x1].sum() > (y1 - y0) * (x1 - x0) * 0.5:
                out[y0:y1, x0:x1] = _MAZE_FLOOR_IDX                  # navigable corridor cell

    # agent / goal occupy the grid cell their sprite centroid falls in
    for mask, colour in ((cheese, _MAZE_GOAL_IDX), (mouse, _MAZE_PLAYER_IDX)):
        if not mask.any():
            continue
        ys, xs = np.where(mask)
        y0, y1 = _span(Oy, Py, int(np.floor((ys.mean() - Oy) / Py)), 64)
        x0, x1 = _span(Ox, Px, int(np.floor((xs.mean() - Ox) / Px)), 64)
        if y1 > y0 and x1 > x0:
            out[y0:y1, x0:x1] = colour
    return out


def _postprocess_miner(rgb: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """Enhance player, exit, and diamond visibility in miner frames.

    - Player teal sprite → yellow (idx 11) with dark outline (idx 4)
    - Exit/window sprite → green (idx 14) — the exit is a dark-gray square
      (RGB≈51,51,51) surrounded by blue-gray (RGB≈187,203,204) frame
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    # --- Exit detection ---
    # The exit sprite ("window.png") renders as a compact cluster of dark
    # (51,51,51) pixels surrounded by blue-gray (187,203,204).
    # Detect by finding connected pairs of the dark center color.
    dark_center = (np.abs(r - 51) < 8) & (np.abs(g - 51) < 8) & (np.abs(b - 51) < 8)

    # Find dark pixels that have at least one orthogonal dark neighbor.
    # The exit center is always a connected cluster of 2+ dark pixels.
    has_neighbor = np.zeros((64, 64), dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        has_neighbor |= np.roll(np.roll(dark_center, dy, axis=0), dx, axis=1)
    exit_core = dark_center & has_neighbor

    if exit_core.any():
        # Flood-fill to include adjacent dark pixels connected to the core
        changed = True
        while changed:
            changed = False
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    expanded = np.roll(np.roll(exit_core, dy, axis=0), dx, axis=1)
                    new_pixels = dark_center & expanded & ~exit_core
                    if new_pixels.any():
                        exit_core |= new_pixels
                        changed = True

        # Paint the exit: core dark pixels + surrounding blue-gray frame → green
        ey, ex = np.where(exit_core)
        ymin = max(0, ey.min() - 2)
        ymax = min(64, ey.max() + 3)
        xmin = max(0, ex.min() - 2)
        xmax = min(64, ex.max() + 3)
        roi_r = r[ymin:ymax, xmin:xmax]
        roi_g = g[ymin:ymax, xmin:xmax]
        roi_b = b[ymin:ymax, xmin:xmax]
        roi_core = exit_core[ymin:ymax, xmin:xmax]
        roi_frame = (roi_g > roi_r + 8) & (roi_b > roi_r + 8) & (np.abs(roi_g - roi_b) < 5) & (roi_r > 80)
        roi_black = (roi_r < 5) & (roi_g < 5) & (roi_b < 5)
        # Only include frame/black pixels adjacent to the core
        near_core = np.zeros_like(roi_core)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                near_core |= np.roll(np.roll(roi_core, dy, axis=0), dx, axis=1)
        exit_mask = roi_core | ((roi_frame | roi_black) & near_core)
        frame[ymin:ymax, xmin:xmax][exit_mask] = 14  # green — exit marker

    # --- Player detection ---
    player = (g > r + 40) & (g > 150) & (b > r + 20) & (g > b)
    frame[player] = 11  # yellow

    # Player outline within bounding box
    py, px = np.where(player)
    if len(py) > 0:
        ymin, ymax = py.min() - 1, py.max() + 2
        xmin, xmax = px.min() - 1, px.max() + 2
        ymin, xmin = max(0, ymin), max(0, xmin)
        ymax, xmax = min(64, ymax), min(64, xmax)
        roi = frame[ymin:ymax, xmin:xmax]
        r_roi = r[ymin:ymax, xmin:xmax]
        g_roi = g[ymin:ymax, xmin:xmax]
        b_roi = b[ymin:ymax, xmin:xmax]
        dark_roi = (r_roi < 80) & (g_roi < 80) & (b_roi < 80) & (r_roi + g_roi + b_roi > 0)
        roi[dark_roi] = 4  # very dark gray outline

    return frame




# Palette indices `_postprocess_miner` paints, reused to synthesise the win frame.
_MINER_PLAYER_IDX = 11     # yellow          -- the miner
_MINER_OUTLINE_IDX = 4     # very dark gray  -- the miner's outline
_MINER_EXIT_IDX = 14       # green           -- the exit
_MINER_FLOOR_IDX = 5       # black           -- dug-out floor


def _miner_win_frame(prev_rgb: np.ndarray) -> np.ndarray:
    """The winning frame ProcGen never renders: the miner standing on the exit.

    Same problem (and same remedy) as `_heist_win_frame`: ProcGen terminates and
    auto-resets INSIDE the winning step, so the observation handed back belongs to
    a brand-new board and the "you won" state is never shown. Synthesise it from
    the last frame actually played -- the miner's cell becomes dug-out floor, and
    the exit becomes the miner. Working in palette space is enough here because
    `_postprocess_miner` has already painted the miner (and only the miner) yellow
    and the exit (and only the exit) green."""
    frame = _postprocess_miner(prev_rgb, _rgb_to_arc_frame(prev_rgb))
    out = frame.copy()
    player = frame == _MINER_PLAYER_IDX
    if player.any():
        ys, xs = np.nonzero(player)
        # `_postprocess_miner` draws the outline inside the miner's bounding box
        # grown by one pixel; clear it there and nowhere else, so dark pixels
        # elsewhere on the board are left alone.
        box = np.zeros_like(player)
        box[max(0, ys.min() - 1):ys.max() + 2, max(0, xs.min() - 1):xs.max() + 2] = True
        out[player | (box & (frame == _MINER_OUTLINE_IDX))] = _MINER_FLOOR_IDX
    out[frame == _MINER_EXIT_IDX] = _MINER_PLAYER_IDX
    return out

# ---------------------------------------------------------------------------
# heist: semantic re-render
#
# ProcGen draws heist at a higher internal resolution and downsamples to 64px,
# so the naive nearest-neighbour ARC quantisation loses the game outright:
#
#   * the GEM (the goal) is a pale-gold sprite whose pixels land on
#     (254,228,129) / (191,187,174) / (255,250,233) -- all of which quantise to
#     ARC idx 1, the SAME index as the blue-gray brick wall. On a "hard" board
#     the entire goal survives as a single yellow pixel; a frame-only policy
#     simply cannot see what it is meant to reach.
#   * the RED key/lock (232,106,23) and the PLAYER's orange (255,153,0) both
#     quantise to ARC idx 12, so the agent and one of the three lock colours are
#     indistinguishable.
#   * downsampling fringes scatter stray palette indices along every sprite edge.
#
# So heist is re-rendered from the RGB frame onto a fixed 7-symbol palette --
# the same treatment `_postprocess_maze` / `_postprocess_miner` give their games:
#
#     1  wall        9  blue key / lock      8  red key / lock
#     5  floor      14  green key / lock     11 gem (the goal)
#                   12  player (0 = its white core)
#
# The classification is pure colour geometry (no grid model, no game state), so
# it is exact for every distribution mode and costs one pass of numpy per frame.
# ---------------------------------------------------------------------------

_HEIST_WALL_IDX = 1
_HEIST_FLOOR_IDX = 5
_HEIST_BLUE_IDX = 9        # blue key / lock
_HEIST_GREEN_IDX = 14      # green key / lock
_HEIST_RED_IDX = 8         # red key / lock   (NOT 12 -- that is the player)
_HEIST_GEM_IDX = 11        # the gem: what the level is won by reaching
_HEIST_PLAYER_IDX = 12     # the player's body
_HEIST_PLAYER_CORE_IDX = 0  # the player's white centre

_SQ3 = np.ones((3, 3), dtype=bool)
_SQ5 = np.ones((5, 5), dtype=bool)


def _heist_parts(rgb: np.ndarray):
    """Split a heist RGB frame into its semantic parts.

    Returns ``(wall, blue, green, red, gem, player, core)`` boolean 64x64 masks.

    The discriminators, in the order they are applied:

    * **wall** -- the brick texture is the only COOL desaturated family
      (B >= G >= R, e.g. (187,203,204)); every sprite colour is warm or saturated.
    * **blue / green** -- saturated hue tests; nothing else in the game is cool
      and saturated.
    * **red vs player** -- both are warm, and separating them is the whole
      difficulty: the lock is (232,106,23) (G/R = 0.46) and the player is
      (255,153,0) (G/R = 0.60). The ratio splits them, and an extra ``B < 60``
      guard rejects the lock-over-wall downsampling blends, which drift up into
      the player's ratio band but always carry the wall's blue with them.
    * **gem** -- bright and yellow (R >= 240, G >= 190); its pale rim
      ((191,187,174), (255,250,233)) is too desaturated for that test but is
      warm-NEUTRAL (R >= G >= B), which the wall never is, so the rim is grown
      back by adjacency to the bright core.
    * **player** -- the orange pixels plus the white core and the dark-gray
      outline they touch, keyed as ONE connected blob so a stray lock blend
      cannot masquerade as the agent (ties break on white-pixel count: the
      player sprite always carries a white core, the locks never do).
    """
    a = rgb.astype(np.int16)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    mx = a.max(axis=2)
    white = (r >= 245) & (g >= 245) & (b >= 245)

    wall = (b >= g - 3) & (g >= r - 3) & (r > 120) & ~white
    warm = r > b + 40
    ratio = g / np.maximum(r, 1)

    blue = (b > r + 40) & (b > g)
    green = (g > r + 40) & (g > b + 40)
    red = warm & (ratio < 0.55)
    orange = warm & (ratio >= 0.55) & (ratio < 0.75) & (r > 150) & (b < 60)

    # gem: bright yellow core, grown into its pale warm-neutral rim
    gem = (r >= 240) & (g >= 190) & (b <= 200) & ~orange
    rim = (r >= g) & (g >= b) & (r - b >= 8) & (r > 100) & ~wall & ~orange
    for _ in range(2):
        grown = ndimage.binary_dilation(gem, _SQ3) & rim
        if (grown == gem).all():
            break
        gem = grown

    # player: the orange blob plus the white core / dark outline it touches
    player = np.zeros((64, 64), dtype=bool)
    core = np.zeros((64, 64), dtype=bool)
    if orange.any():
        near = ndimage.binary_dilation(orange, _SQ5)
        dark = (mx < 175) & (mx - a.min(axis=2) < 40) & (mx > 20)
        cand = orange | (near & (white | dark))
        lab, n = ndimage.label(cand, _SQ3)
        if n:
            idx = np.arange(1, n + 1)
            # prefer the component holding the most white (the player's core);
            # break ties on size, so a white-less lock blend can never win.
            score = (100 * np.array(ndimage.sum(white & cand, lab, idx))
                     + np.array(ndimage.sum(cand, lab, idx)))
            player = lab == int(np.argmax(score)) + 1
            core = player & white

    # The agent's own anti-aliasing can pass the gem test: where its orange meets
    # its white core the downsampler lands on pale gold. Left in, the frame shows
    # a gem pixel travelling with the agent -- a phantom goal, both for a policy
    # and for anything planning off the frame -- so the gem never overlaps the
    # agent or the ring of blend pixels around it.
    gem &= ~ndimage.binary_dilation(player, _SQ3)
    return wall, blue, green, red, gem, player, core


def _postprocess_heist(rgb: np.ndarray, frame: np.ndarray | None = None) -> np.ndarray:
    """Re-render a heist frame onto the fixed 7-symbol palette (see above).

    ``frame`` (the naive quantisation) is accepted for signature-compatibility
    with `_postprocess_maze` / `_postprocess_miner` but discarded rather than
    patched: every
    sprite in this game needs correcting, and rebuilding from the wall/floor
    split drops the downsampling fringe that would otherwise scatter stray
    indices through the corridors."""
    wall, blue, green, red, gem, player, core = _heist_parts(rgb)
    out = np.where(wall, _HEIST_WALL_IDX, _HEIST_FLOOR_IDX).astype(np.uint8)
    out[gem] = _HEIST_GEM_IDX
    out[blue] = _HEIST_BLUE_IDX
    out[green] = _HEIST_GREEN_IDX
    out[red] = _HEIST_RED_IDX
    out[player] = _HEIST_PLAYER_IDX
    out[core] = _HEIST_PLAYER_CORE_IDX
    return out


def _heist_win_frame(prev_rgb: np.ndarray) -> np.ndarray:
    """The winning frame ProcGen never renders: the agent standing on the gem.

    ProcGen terminates and IMMEDIATELY auto-resets inside the winning step, so
    the observation handed back is the FIRST frame of a brand-new level. Painting
    that into ``_current_frame`` would end every won episode -- in training data
    and at inference alike -- on a board that has nothing to do with the one just
    solved. Instead the win is synthesised from the LAST frame of the level that
    was actually played: the gem is replaced by the agent, and the agent's old
    position becomes plain floor. (`_postprocess_maze`'s generator does the same
    for procgen_maze, but only in its recorder -- here it is done in the adapter,
    so the recorded frame and the live one agree.)"""
    wall, blue, green, red, gem, player, _core = _heist_parts(prev_rgb)
    out = np.where(wall, _HEIST_WALL_IDX, _HEIST_FLOOR_IDX).astype(np.uint8)
    out[blue] = _HEIST_BLUE_IDX
    out[green] = _HEIST_GREEN_IDX
    out[red] = _HEIST_RED_IDX
    if gem.any():                       # the agent has arrived on the gem's cell
        out[gem] = _HEIST_PLAYER_IDX
        ys, xs = np.nonzero(gem)
        cy, cx = ys.mean(), xs.mean()
        i = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        out[ys[i], xs[i]] = _HEIST_PLAYER_CORE_IDX
    elif player.any():                  # no gem in view: leave the agent put
        out[player] = _HEIST_PLAYER_IDX
    return out


# ---------------------------------------------------------------------------
# Supported games and action mapping
# ---------------------------------------------------------------------------

SUPPORTED_GAMES: tuple[str, ...] = ("maze", "miner", "heist")

# 7-level difficulty schedule: (distribution_mode, within-tier seed) per level index.
# heist:  levels 0-1 use "easy" (1 key/lock), levels 2-6 use "hard" (2 keys/locks).
#         "extreme" mode is not supported by this procgen build (only chaser/dodgeball/
#         leaper/starpilot support it).  Mode transition is the primary difficulty driver;
#         within-tier seeds vary the layout only.
# The within-tier seed is added to self._seed as an offset.
# Beyond index 6, the last entry is repeated.
_LEVEL_SCHEDULE: dict[str, tuple[tuple[str, int], ...]] = {
    "heist":  (("easy",0),("easy",1),("hard",0),("hard",1),("hard",2),("hard",3),("hard",4)),
}

# Curated seed pools for games with monotone difficulty ordering.
# Each level entry is (distribution_mode, (seed_pool,)) where the pool holds
# all validated non-trivial seeds that share the same floor-pixel tier.
# _preferred_start picks pool[self._seed % len(pool)] so different base seeds
# yield different mazes while difficulty ordering is always preserved.
#
# ARC-palette floor-pixel count (index 2 = mid-gray = open path) is the proxy:
#   more floor pixels  →  more open maze  →  easier navigation
#   fewer floor pixels →  denser maze     →  harder navigation
#
# Pools scanned from seeds 0-699; all entries validated non-trivial (T=0).
# Capped to the 5 easy tiers (sparsest -> densest); hard tiers dropped (don't render cleanly).
# Top easy tiers (fp≈3984, fp≈3748) dropped — too simple / only rotations.
_LEVEL_SEEDS_ABS: dict[str, tuple[tuple[str, tuple[int, ...]], ...]] = {
    "maze": (
        # easy fp≈3568  level 1 — 80 seeds
        ("easy", (5, 25, 32, 54, 82, 88, 100, 111, 120, 138, 141, 147, 151, 156, 165, 173,
                  192, 206, 218, 226, 231, 236, 244, 255, 256, 261, 277, 278, 285, 303,
                  305, 316, 317, 320, 324, 329, 333, 342, 343, 352, 356, 367, 372, 379,
                  385, 393, 406, 412, 419, 422, 425, 449, 452, 460, 471, 474, 488, 489,
                  504, 509, 543, 547, 558, 568, 574, 577, 583, 589, 605, 608, 617, 632,
                  643, 649, 652, 653, 666, 676, 690, 695)),
        # easy fp≈3184  level 2 — 76 seeds
        ("easy", (3, 8, 13, 15, 47, 63, 75, 78, 81, 90, 98, 101, 114, 125, 146, 150, 158,
                  166, 171, 177, 181, 223, 232, 237, 249, 279, 287, 302, 319, 323, 326,
                  330, 346, 371, 375, 386, 396, 404, 416, 421, 440, 441, 446, 451, 453,
                  464, 479, 480, 486, 487, 496, 497, 502, 517, 518, 524, 539, 546, 551,
                  570, 576, 578, 582, 585, 600, 604, 626, 627, 630, 637, 640, 642, 660,
                  671, 697, 698)),
        # easy fp≈2904  level 3 — 99 seeds
        ("easy", (1, 10, 18, 21, 27, 49, 53, 58, 62, 71, 76, 77, 80, 92, 94, 110, 112,
                  116, 118, 135, 144, 145, 153, 160, 161, 167, 170, 174, 175, 188, 191,
                  193, 205, 209, 214, 216, 224, 225, 228, 229, 233, 241, 246, 248, 253,
                  258, 264, 269, 271, 275, 280, 281, 283, 291, 292, 304, 325, 331, 340,
                  348, 354, 358, 359, 392, 405, 409, 426, 436, 442, 445, 448, 454, 455,
                  466, 473, 490, 505, 511, 519, 520, 521, 528, 533, 559, 579, 581, 601,
                  615, 616, 619, 620, 623, 648, 665, 672, 679, 681, 691, 694)),
        # easy fp≈2315  level 4 — 91 seeds
        ("easy", (4, 9, 19, 23, 31, 60, 69, 79, 86, 91, 121, 128, 132, 137, 152, 155, 182,
                  184, 207, 222, 235, 250, 262, 265, 282, 290, 293, 295, 298, 299, 307,
                  309, 310, 311, 318, 321, 327, 350, 360, 361, 364, 368, 373, 376, 387,
                  394, 397, 399, 400, 402, 411, 417, 418, 424, 427, 431, 458, 462, 469,
                  481, 482, 483, 485, 493, 494, 500, 503, 514, 516, 525, 530, 545, 554,
                  563, 565, 566, 571, 572, 590, 602, 609, 625, 644, 654, 659, 664, 667,
                  669, 675, 684, 685)),
        # easy fp≈1912  level 5 — 112 seeds
        ("easy", (6, 7, 12, 16, 26, 28, 34, 35, 37, 38, 50, 55, 64, 66, 73, 74, 83, 89,
                  96, 108, 109, 119, 129, 130, 131, 136, 140, 142, 143, 149, 163, 172,
                  176, 180, 195, 201, 204, 215, 234, 238, 251, 263, 266, 267, 274, 300,
                  337, 338, 344, 345, 349, 362, 363, 366, 369, 370, 374, 378, 390, 398,
                  407, 410, 413, 415, 428, 430, 435, 437, 438, 443, 459, 461, 468, 470,
                  476, 477, 491, 492, 499, 526, 527, 529, 531, 532, 537, 541, 550, 557,
                  560, 561, 562, 573, 584, 587, 588, 593, 594, 599, 603, 610, 624, 631,
                  634, 641, 646, 651, 657, 663, 668, 674, 677, 687)),
        # NOTE: the two former "hard" tiers (levels 6-7) are dropped -- their mazes are large
        # (~25x25 cells) and procgen crams them into 64px at ~2.5px/cell, so corridors alias and
        # the player/goal sprites become undetectable. Only easy-mode mazes (~4.3px/cell) render
        # cleanly and are reconstructable, so the schedule is capped at the 5 easy tiers.
    ),
    "miner": (
        # easy bld≈2773-2822  level 1 — 125 seeds (fewest boulders, easiest)
        ("easy", (212, 49, 360, 94, 15, 137, 227, 110, 210, 21, 383, 238, 176, 96, 481, 298,
                  302, 497, 143, 197, 120, 338, 123, 218, 352, 310, 392, 422, 2, 434, 185,
                  430, 26, 192, 364, 450, 93, 131, 296, 324, 119, 121, 245, 251, 307, 127,
                  136, 157, 201, 205, 483, 55, 175, 255, 263, 346, 496, 248, 314, 492, 171,
                  29, 130, 169, 278, 351, 124, 152, 269, 287, 387, 470, 54, 371, 446, 196,
                  421, 495, 173, 184, 46, 256, 437, 158, 177, 257, 95, 325, 72, 116, 239,
                  326, 386, 426, 47, 221, 236, 299, 487, 112, 397, 33, 295, 27, 103, 247,
                  447, 25, 148, 183, 331, 465, 17, 111, 250, 428, 449, 44, 48, 68, 174,
                  411, 451, 254, 400)),
        # easy bld≈2822-2846  level 2 — 125 seeds
        ("easy", (440, 469, 69, 115, 291, 398, 90, 280, 389, 246, 369, 407, 491, 32, 91, 8,
                  52, 379, 438, 456, 464, 28, 471, 0, 109, 189, 12, 182, 258, 341, 350, 7,
                  138, 172, 180, 217, 253, 277, 194, 391, 419, 53, 135, 214, 297, 317, 353,
                  42, 164, 213, 220, 233, 403, 423, 467, 41, 84, 132, 223, 234, 301, 311,
                  336, 412, 489, 80, 114, 319, 322, 395, 494, 207, 275, 18, 57, 125, 144,
                  163, 187, 388, 405, 102, 105, 162, 202, 244, 273, 283, 334, 357, 393, 416,
                  420, 272, 401, 462, 479, 70, 159, 267, 304, 332, 399, 230, 354, 37, 237,
                  486, 56, 191, 329, 333, 367, 58, 126, 165, 260, 358, 380, 35, 43, 118,
                  195, 284, 290)),
        # easy bld≈2847-2874  level 3 — 125 seeds
        ("easy", (5, 9, 274, 285, 425, 38, 45, 413, 442, 229, 243, 382, 134, 361, 414, 432,
                  249, 348, 436, 190, 203, 211, 318, 76, 81, 241, 108, 128, 222, 288, 268,
                  276, 365, 475, 476, 498, 99, 129, 186, 216, 378, 480, 490, 78, 315, 435,
                  85, 429, 20, 34, 36, 60, 160, 215, 266, 286, 320, 204, 31, 73, 231, 242,
                  431, 147, 170, 198, 363, 394, 208, 342, 345, 484, 485, 240, 306, 327, 11,
                  23, 146, 225, 337, 445, 87, 167, 265, 270, 453, 156, 359, 373, 472, 477,
                  133, 140, 166, 366, 478, 219, 279, 294, 305, 308, 309, 381, 404, 463, 4,
                  82, 448, 24, 199, 313, 356, 385, 454, 461, 468, 75, 117, 206, 418, 474,
                  281, 343, 19)),
        # easy bld≈2874-2976  level 4 — 125 seeds (most boulders in easy)
        ("easy", (62, 122, 6, 66, 154, 161, 328, 402, 410, 97, 408, 14, 145, 493, 155, 344,
                  396, 417, 289, 362, 406, 424, 153, 226, 340, 460, 92, 101, 188, 262, 323,
                  355, 443, 77, 79, 200, 16, 193, 261, 13, 63, 83, 499, 30, 252, 282, 372,
                  67, 259, 339, 349, 64, 88, 104, 293, 368, 427, 415, 433, 228, 22, 86, 141,
                  235, 1, 98, 151, 335, 482, 488, 178, 466, 51, 107, 457, 459, 224, 455, 40,
                  473, 452, 3, 458, 89, 181, 139, 444, 61, 441, 377, 390, 321, 376, 39, 106,
                  150, 179, 292, 375, 409, 312, 209, 74, 113, 59, 71, 347, 65, 303, 370, 316,
                  439, 300, 384, 10, 149, 264, 330, 50, 168, 271, 142, 100, 374, 232)),
        # hard bld≈2760-2794  level 5 — 166 seeds
        ("hard", (45, 245, 43, 259, 161, 15, 436, 74, 127, 112, 360, 382, 252, 272, 93, 117,
                  121, 150, 86, 279, 416, 443, 122, 61, 214, 354, 393, 110, 175, 198, 219,
                  228, 253, 323, 133, 226, 305, 322, 364, 422, 454, 455, 169, 199, 478, 9,
                  23, 98, 176, 304, 390, 410, 398, 446, 47, 400, 8, 46, 167, 230, 295, 367,
                  30, 126, 148, 227, 391, 7, 25, 54, 99, 152, 178, 196, 331, 340, 369, 424,
                  444, 16, 31, 229, 250, 257, 338, 350, 484, 63, 128, 194, 327, 407, 459, 21,
                  132, 363, 441, 83, 103, 105, 158, 180, 183, 233, 235, 239, 306, 356, 372,
                  376, 377, 409, 427, 487, 100, 210, 213, 225, 251, 379, 1, 12, 55, 60, 71,
                  101, 108, 204, 220, 266, 404, 470, 4, 116, 182, 217, 246, 248, 263, 293,
                  373, 397, 458, 29, 40, 139, 143, 231, 247, 375, 438, 450, 457, 64, 97,
                  187, 240, 261, 296, 318, 328, 428, 431, 32, 35, 147)),
        # hard bld≈2794-2810  level 6 — 166 seeds
        ("hard", (160, 166, 174, 195, 242, 294, 345, 493, 17, 70, 106, 164, 170, 193, 202,
                  221, 232, 278, 320, 383, 419, 461, 37, 62, 165, 184, 452, 462, 68, 84, 87,
                  123, 205, 348, 399, 414, 439, 448, 460, 28, 77, 118, 131, 209, 211, 234,
                  238, 299, 300, 464, 499, 137, 146, 236, 264, 456, 496, 497, 51, 67, 79,
                  102, 243, 277, 287, 387, 482, 114, 190, 216, 270, 289, 352, 423, 437, 480,
                  495, 82, 120, 260, 262, 283, 73, 80, 90, 107, 168, 297, 349, 351, 384, 401,
                  483, 49, 134, 138, 140, 173, 191, 192, 203, 207, 321, 392, 394, 406, 468,
                  488, 58, 59, 197, 241, 275, 311, 403, 435, 463, 485, 22, 34, 66, 69, 179,
                  284, 292, 310, 329, 405, 467, 469, 475, 3, 27, 258, 303, 347, 411, 432, 44,
                  48, 78, 96, 154, 155, 157, 172, 185, 201, 381, 434, 38, 156, 171, 177, 215,
                  224, 302, 330, 335, 342, 388, 389, 42, 104, 181, 208)),
        # hard bld≈2810-2854  level 7 — 168 seeds (most boulders, hardest)
        ("hard", (237, 298, 309, 336, 366, 402, 39, 92, 94, 273, 280, 313, 334, 421, 113,
                  325, 365, 433, 465, 472, 494, 88, 149, 159, 162, 218, 222, 244, 271, 281,
                  359, 396, 466, 476, 2, 11, 75, 85, 249, 269, 301, 319, 370, 417, 453, 6,
                  14, 52, 57, 136, 282, 315, 353, 362, 486, 141, 254, 324, 339, 426, 429,
                  440, 471, 50, 125, 326, 346, 415, 425, 26, 65, 163, 200, 256, 267, 268,
                  361, 413, 430, 0, 5, 76, 135, 286, 355, 408, 447, 129, 130, 276, 312, 343,
                  449, 451, 477, 489, 10, 56, 91, 255, 378, 474, 498, 119, 308, 344, 385,
                  386, 479, 481, 81, 144, 151, 341, 368, 395, 13, 19, 142, 153, 95, 145, 20,
                  53, 115, 442, 206, 212, 33, 89, 274, 307, 314, 316, 473, 333, 418, 358,
                  420, 18, 72, 186, 317, 412, 111, 124, 332, 337, 492, 41, 357, 189, 36, 265,
                  109, 188, 291, 491, 288, 24, 374, 371, 380, 445, 290, 223, 490, 285)),
    ),
}

# ProcGen Discrete(15) action indices — joystick 3×3 grid layout:
#   0=↙  1=←  2=↖
#   3=↓  4=·   5=↑
#   6=↘  7=→  8=↗
#   (9-14: same directions + D-button)
_ACTION_MAP: dict[GameAction, int] = {
    GameAction.ACTION1: 5,  # up
    GameAction.ACTION2: 3,  # down
    GameAction.ACTION3: 1,  # left
    GameAction.ACTION4: 7,  # right
    GameAction.ACTION5: 4,  # noop (no meaningful button interaction in these games)
    GameAction.ACTION6: 4,  # noop
    GameAction.ACTION7: 4,  # noop (no undo)
}

# ---------------------------------------------------------------------------
# Step budget tracker
# ---------------------------------------------------------------------------

# Per-game step budgets.  ProcGen's built-in timeout is 1 000 for all games,
# but these tighter per-game limits give better feedback via the status bar
# and create a real LOSE condition before the engine truncates the episode.
#   maze  — navigation only, well-bounded by maze size
#   miner — collect diamonds + reach exit; boulder dodging takes more steps
#   heist — 1-2 key/lock pairs; moderate navigation overhead
#: Reward ProcGen pays for COMPLETING a level (identical for maze/miner/heist).
#: The WIN test thresholds on it rather than on ``reward > 0`` because miner also
#: pays +1 per gem mid-episode -- see the module docstring.
_COMPLETION_BONUS = 10.0

_PROCGEN_MAX_STEPS: dict[str, int] = {
    "maze":  500,
    "miner": 2000,
    "heist": 1000,
}


# ---------------------------------------------------------------------------
# ProcGenAdapter
# ---------------------------------------------------------------------------

class ProcGenAdapter(BaseAdapter):
    """Wraps a ProcGen environment as an ARCBaseGame-compatible object.

    Seven levels (0–6) for most games; procgen_maze is capped at 5 (see below).

    For procgen_maze the 5 levels use curated absolute easy-mode seed pools (bypassing
    self._seed) so maze complexity increases monotonically while every maze still renders
    cleanly at ~4.3px/cell:
        levels 0–4 — easy mode  (fp≈3568 → fp≈1912, increasingly dense)
    (The former hard-mode levels 5–6 are dropped: their ~25x25 mazes compress to ~2.5px/cell
    in the 64px frame, aliasing the corridors and hiding the player/goal sprites.)

    For procgen_miner the 7 levels use curated absolute seed pools sorted by
    boulder count (more boulders = harder):
        levels 0–3 — easy mode  (bld≈2773 → bld≈2976, increasing boulders)
        levels 4–6 — hard mode  (bld≈2760 → bld≈2854, largest/most boulders)

    For heist, levels use relative seeds (self._seed + within_seed) with
    distribution_mode stepping.  Mode controls key/lock count monotonically:
        levels 0–1  — easy    (1 key/lock pair, small maps)
        levels 2–6  — hard    (2 key/lock pairs, larger maps; extreme mode is
                               not supported by this procgen build)

    Args:
        game_name:  One of "maze", "miner", "heist".
        seed:       Base seed offset added to the within-tier seed index.
    """

    def __init__(self, game_name: str, seed: int = 0) -> None:
        if game_name not in SUPPORTED_GAMES:
            raise ValueError(
                f"Unsupported game {game_name!r}. Supported: {SUPPORTED_GAMES}"
            )

        self._game_name = game_name
        self._seed = seed
        self._game_id = f"procgen_{game_name}"

        self._current_mode: str = "easy"
        self._env = None
        self._gym3 = None  # ProcgenGym3Env, located after each env creation
        # Opening gym3 state of the CURRENT level, captured the first time that
        # level is seated; `_do_reset` restores it (see there). Cleared whenever a
        # new env is built, since that env has an opening state of its own.
        self._level_s0 = None
        self._prev_rgb = None  # last pre-terminal RGB (heist's synthesised WIN frame)

        # Shared adapter scaffolding (state, step counter, undo stack). The frame
        # is produced by set_level → _do_reset below.
        self._init_base(self._game_id,
                        max_steps=_PROCGEN_MAX_STEPS.get(game_name, 1000))

        self.set_level(0)

    # ------------------------------------------------------------------
    # Environment lifecycle
    # ------------------------------------------------------------------

    def _level_to_mode_and_seed(self, idx: int) -> tuple[str, int]:
        """Map a level index (0-based) to (distribution_mode, seed).

        For games in _LEVEL_SEEDS_ABS the seed is chosen from the tier's pool
        using self._seed as the selector (pool[self._seed % len(pool)]), giving
        variety across base seeds while preserving difficulty ordering.
        For other games the seed is a within-tier offset added to self._seed.
        Clamps to the last entry for indices beyond the schedule length.
        """
        if self._game_name in _LEVEL_SEEDS_ABS:
            schedule = _LEVEL_SEEDS_ABS[self._game_name]
            mode, pool = schedule[min(idx, len(schedule) - 1)]
            return mode, pool[self._seed % len(pool)]
        schedule = _LEVEL_SCHEDULE[self._game_name]
        return schedule[min(idx, len(schedule) - 1)]

    # Minimum number of straight-line steps before a seed is considered non-trivial.
    _MIN_STEPS: dict[str, int] = {"easy": 10, "hard": 20, "extreme": 30}

    def _create_raw_env(self, start_level: int, mode: str):
        """Create a ProcGen gym env without touching self._env."""
        try:
            import gym  # type: ignore[import]
            import procgen  # noqa: F401
        except ImportError as exc:
            raise ImportError("ProcGen is required: pip install procgen") from exc
        return gym.make(
            f"procgen:procgen-{self._game_name}-v0",
            start_level=start_level,
            num_levels=1,
            distribution_mode=mode,
            use_backgrounds=False,
            render_mode="rgb_array",
        )

    def _make_env(self, start_level: int, mode: str):
        """Close self._env and create a fresh one (no seed validation)."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        return self._create_raw_env(start_level, mode)

    def _is_trivial(self, env, mode: str) -> bool:
        """Return True if the level can be won by walking straight in any one direction.

        Tests each cardinal direction for up to _MIN_STEPS[mode] steps.  Seeds
        where the exit is right next to the spawn are rejected this way.
        """
        threshold = self._MIN_STEPS.get(mode, 10)
        with _quiet_stdout():
            for direction in (5, 3, 1, 7):  # up, down, left, right
                env.reset()
                for _ in range(threshold):
                    result = env.step(direction)
                    reward, done = result[1], result[2]
                    if done and reward > 0:
                        return True
        return False

    def _make_good_env(self, preferred_start: int, mode: str, max_search: int = 50):
        """Search for a non-trivial seed, trying up to max_search candidates."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

        for offset in range(max_search):
            env = self._create_raw_env(start_level=preferred_start + offset, mode=mode)
            if not self._is_trivial(env, mode):
                return env
            try:
                env.close()
            except Exception:
                pass

        # Fallback: return the last candidate regardless
        return self._create_raw_env(start_level=preferred_start + max_search, mode=mode)

    @staticmethod
    def _find_gym3(env):
        """Walk the gym wrapper chain to the ProcgenGym3Env exposing
        get_state/set_state (the only layer that can snapshot the C++ core)."""
        cur = getattr(env, "unwrapped", env)
        while cur is not None:
            if type(cur).__name__ == "ProcgenGym3Env":
                return cur
            cur = getattr(cur, "env", None)
        return None

    def _do_reset(self) -> None:
        """Put the current level back at its opening position.

        ``self._env.reset()`` CANNOT do this: gym3's ``ToGymEnv.reset`` only
        re-observes and prints "Warning: early reset ignored" when the episode is
        still running (which is exactly when RESET is pressed), so going through
        it left the level mid-play and made the engine's RESET action a silent
        no-op for every ProcGen game. So the opening state is snapshotted the
        first time a level is seated and restored through the gym3 layer
        thereafter -- the same ``get_state``/``set_state`` pair `_snapshot` uses.
        """
        self._gym3 = self._find_gym3(self._env)
        self._undo_stack.clear()
        if self._level_s0 is not None and self._gym3 is not None:
            self._gym3.set_state(self._level_s0)
            _rew, obs, _first = self._gym3.observe()
            rgb = self._extract_rgb(obs)[0]
        else:
            with _quiet_stdout():
                result = self._env.reset()
            obs = result[0] if isinstance(result, tuple) else result
            rgb = self._extract_rgb(obs)
            if self._gym3 is not None:
                self._level_s0 = self._gym3.get_state()
        self._current_frame = _rgb_to_arc_frame(rgb)
        if self._game_name == "miner":
            self._current_frame = _postprocess_miner(rgb, self._current_frame)
        elif self._game_name == "maze":
            self._current_frame = _postprocess_maze(rgb, self._current_frame)
        elif self._game_name == "heist":
            self._current_frame = _postprocess_heist(rgb, self._current_frame)
        self._prev_rgb = rgb
        self._state = GameState.NOT_FINISHED
        self._action_count = 0
        self._step_counter.reset()

    @staticmethod
    def _extract_rgb(obs) -> np.ndarray:
        """Extract the (64, 64, 3) RGB array from a ProcGen observation."""
        return obs["rgb"] if isinstance(obs, dict) else obs

    # ------------------------------------------------------------------
    # ARCBaseGame interface (game_id/perform_action/_make_frame_data + RESET +
    # ACTION7-undo + terminal short-circuit come from BaseAdapter). set_level /
    # full_reset / level_reset are overridden here because ProcGen needs a fresh
    # gym env + seed search rather than BaseAdapter's _reset_env(seed).
    # ------------------------------------------------------------------

    def _preferred_start(self, idx: int) -> tuple[str, int]:
        """Return (mode, start_level) for the given level index.

        For games in _LEVEL_SEEDS_ABS the start_level is already the absolute
        pool-selected seed (self._seed used only as pool index, not offset).
        For other games self._seed is added as an offset.
        """
        mode, seed = self._level_to_mode_and_seed(idx)
        if self._game_name in _LEVEL_SEEDS_ABS:
            return mode, seed           # already absolute
        return mode, self._seed + seed  # relative offset

    def set_level(self, idx: int) -> None:
        """Pin to a specific level, skipping trivially easy seeds."""
        mode, start = self._preferred_start(idx)
        self._current_level_index = idx
        self._current_mode = mode
        if self._game_name in _LEVEL_SEEDS_ABS:
            # Curated pools are already validated — use the exact seed.
            self._env = self._make_env(start_level=start, mode=mode)
        else:
            self._env = self._make_good_env(preferred_start=start, mode=mode)
        self._level_s0 = None                 # fresh env -> new opening state
        self._do_reset()

    def full_reset(self) -> None:
        """Reset back to level 0."""
        self._current_level_index = 0
        mode, start = self._preferred_start(0)
        self._current_mode = mode
        if self._game_name in _LEVEL_SEEDS_ABS:
            self._env = self._make_env(start_level=start, mode=mode)
        else:
            self._env = self._make_good_env(preferred_start=start, mode=mode)
        self._level_s0 = None                 # fresh env -> new opening state
        self._do_reset()

    def level_reset(self) -> None:
        """Reset the current level in-place (same seed, no re-search needed)."""
        self._do_reset()

    def _apply(self, action_input: ActionInput) -> None:
        """Execute one action.

        WIN:       terminated=True  AND  reward >= _COMPLETION_BONUS
        GAME_OVER: terminated=True  AND  reward <  _COMPLETION_BONUS  (death)
                   truncated=True                                     (step limit)"""
        pg_action = _ACTION_MAP.get(action_input.id, 0)

        result = self._env.step(pg_action)

        # Handle both old gym (4-tuple) and new gym ≥0.26 (5-tuple) APIs
        if len(result) == 4:
            obs, reward, done, _info = result
            terminated, truncated = bool(done), False
        else:
            obs, reward, terminated, truncated, _info = result
            terminated, truncated = bool(terminated), bool(truncated)

        rgb = self._extract_rgb(obs)
        won = terminated and reward >= _COMPLETION_BONUS
        if won and self._game_name in ("heist", "miner"):
            # ProcGen already auto-reset inside this step, so ``obs`` belongs to a
            # NEW level. Synthesise the win from the last frame of the level that
            # was actually played (see `_heist_win_frame` / `_miner_win_frame`).
            self._current_frame = (_heist_win_frame if self._game_name == "heist"
                                   else _miner_win_frame)(self._prev_rgb)
        else:
            self._current_frame = _rgb_to_arc_frame(rgb)
            if self._game_name == "miner":
                self._current_frame = _postprocess_miner(rgb, self._current_frame)
            elif self._game_name == "maze":
                self._current_frame = _postprocess_maze(rgb, self._current_frame)
            elif self._game_name == "heist":
                self._current_frame = _postprocess_heist(rgb, self._current_frame)
            self._prev_rgb = rgb
        self._action_count += 1
        self._step_counter.steps_remaining = max(0, self._step_counter.max_steps - self._action_count)

        if terminated:
            self._state = (GameState.WIN if reward >= _COMPLETION_BONUS
                           else GameState.GAME_OVER)
        elif truncated or self._step_counter.steps_remaining == 0:
            self._state = GameState.GAME_OVER
        else:
            self._state = GameState.NOT_FINISHED

    # ── snapshot / restore for generic UNDO ──────────────────────────────────
    def _snapshot(self):
        """Snapshot the ProcGen C++ core via the gym3 layer's get_state (a list
        of immutable bytes), plus adapter bookkeeping. If the gym3 layer wasn't
        located (older build), snapshotting is disabled and UNDO is a no-op —
        procgen moves are walk-back reversible, so this is safe."""
        core = self._gym3.get_state() if self._gym3 is not None else None
        return (core, self._action_count, self._step_counter.steps_remaining,
                self._state, self._current_frame, self._prev_rgb)

    def _restore(self, snap) -> None:
        core, self._action_count, self._step_counter.steps_remaining, \
            self._state, self._current_frame, self._prev_rgb = snap
        if core is not None and self._gym3 is not None:
            self._gym3.set_state(core)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        """Total number of levels in the difficulty schedule (5 for maze, 7 otherwise)."""
        if self._game_name in _LEVEL_SEEDS_ABS:
            return len(_LEVEL_SEEDS_ABS[self._game_name])
        return len(_LEVEL_SCHEDULE[self._game_name])

    @staticmethod
    def list_available_games() -> tuple[str, ...]:
        """Return the supported ProcGen game names."""
        return SUPPORTED_GAMES

    def close(self) -> None:
        """Release the underlying gym environment."""
        if self._env is not None:
            self._env.close()

    def __repr__(self) -> str:
        return (
            f"ProcGenAdapter(game={self._game_name!r}, "
            f"level={self._current_level_index}, state={self._state})"
        )
