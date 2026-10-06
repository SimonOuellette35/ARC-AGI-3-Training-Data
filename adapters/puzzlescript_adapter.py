"""puzzlescript_adapter.py — Pure-Python PuzzleScript interpreter + ARCBaseGame adapter.

Wraps PuzzleScript game definition files (.txt) as ARCBaseGame-compatible objects.
Implements a subset of the PuzzleScript language sufficient for grid-based puzzle
games (sokoban variants, block-pushing, path-clearing, etc.).

Supported PuzzleScript features:
  - Objects with solid colors or 5×5 pixel sprites
  - Legend entries (character → object mapping, including "and"/"or" composites)
  - Collision layers (objects on the same layer block each other)
  - Rules with directional movement propagation (>, <, ^, v)
  - Late rules (applied after main rule pass)
  - Win conditions: all/no/some X [on Y]
  - Multiple levels per game file
  - require_player_movement prelude flag

Frame pipeline:
  Grid state (H×W, each cell has a set of object indices)
  → render each cell: pick top-visible object's dominant color
  → map to nearest ARC 16-color palette index
  → integer-scale + center-pad to 64×64
  → 64×64 uint8 ndarray of palette indices

Action mapping (5 actions, absolute directions):
  GameAction.ACTION1 (W / ↑) → UP    (direction index 0)
  GameAction.ACTION2 (S / ↓) → DOWN  (direction index 1)
  GameAction.ACTION3 (A / ←) → LEFT  (direction index 2)
  GameAction.ACTION4 (D / →) → RIGHT (direction index 3)
  GameAction.ACTION5 (E / sp) → ACTION (direction index 4)

WIN / GAME_OVER semantics:
  Win conditions met      → GameState.WIN
  Step limit exceeded     → GameState.GAME_OVER

Usage:
    from adapters.puzzlescript_adapter import PuzzleScriptAdapter

    game = PuzzleScriptAdapter("sokoban_basic", seed=0)
    game.set_level(0)
    result = game.perform_action(ActionInput(id=GameAction.ACTION1))
    print(result.state)           # GameState.NOT_FINISHED
    print(result.frame[0].shape)  # (64, 64)
"""

from __future__ import annotations

import copy as _copy
import itertools as _itertools
import os
import re
import random as _random_module
from typing import Optional

import numpy as np
from PIL import Image

from arcengine import ActionInput, GameAction, GameState, FrameDataRaw
from utils.rotation import remap_action_full as _remap_action_full
from adapters.base import BaseAdapter

# ---------------------------------------------------------------------------
# ARC 16-color palette (RGB values)
# ---------------------------------------------------------------------------

ARC_PALETTE_RGB = [
    (255, 255, 255),  # 0  white
    (204, 204, 204),  # 1  light gray
    (153, 153, 153),  # 2  mid gray
    (102, 102, 102),  # 3  dark gray
    (51, 51, 51),     # 4  very dark gray
    (0, 0, 0),        # 5  black
    (229, 58, 163),   # 6  pink/magenta
    (255, 123, 204),  # 7  light pink
    (249, 60, 49),    # 8  red
    (30, 147, 255),   # 9  blue
    (136, 216, 241),  # 10 light blue
    (255, 220, 0),    # 11 yellow
    (255, 133, 27),   # 12 orange
    (146, 18, 49),    # 13 dark red/maroon
    (79, 204, 48),    # 14 green
    (163, 86, 214),   # 15 purple
]

# Named CSS/PuzzleScript colors → closest ARC palette index
_COLOR_NAME_TO_ARC: dict[str, int] = {
    "black": 5,
    "white": 0,
    "lightgray": 1, "lightgrey": 1, "light gray": 1, "light grey": 1,
    "gray": 2, "grey": 2,
    "darkgray": 3, "darkgrey": 3, "dark gray": 3, "dark grey": 3,
    "red": 8,
    "darkred": 13,
    "lightred": 8,
    "brown": 12, "darkbrown": 13,
    "orange": 12,
    "yellow": 11, "lightyellow": 11,
    "green": 14, "darkgreen": 14,
    "lightgreen": 14,
    "blue": 9, "darkblue": 9,
    "lightblue": 10,
    "purple": 15,
    "pink": 6, "magenta": 6,
    "transparent": -1,
}


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    """Convert hex color (#RGB or #RRGGBB) to (R, G, B)."""
    h = hex_str.lstrip("#")
    if len(h) == 3:
        h = h[0]*2 + h[1]*2 + h[2]*2
    elif len(h) == 6:
        pass
    else:
        return (128, 128, 128)  # fallback
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _rgb_to_arc(r: int, g: int, b: int) -> int:
    """Find nearest ARC palette index for an RGB color.

    Uses a hue-aware penalty so saturated colors (greens, blues, etc.)
    are never mapped to achromatic grays.
    """
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    chroma = max_c - min_c
    has_hue = chroma > 40  # input color is clearly chromatic

    best_idx = 0
    best_dist = float("inf")
    for i, (pr, pg, pb) in enumerate(ARC_PALETTE_RGB):
        d = (r - pr)**2 + (g - pg)**2 + (b - pb)**2
        # Penalise matching a chromatic input to an achromatic palette entry
        if has_hue:
            p_chroma = max(pr, pg, pb) - min(pr, pg, pb)
            if p_chroma < 30:
                d += 15000
        if d < best_dist:
            best_dist = d
            best_idx = i
    return best_idx


def _color_name_to_arc(name: str) -> int:
    """Convert a PuzzleScript color name or hex to ARC palette index."""
    name = name.strip().lower()
    if name in _COLOR_NAME_TO_ARC:
        return _COLOR_NAME_TO_ARC[name]
    if name.startswith("#"):
        r, g, b = _hex_to_rgb(name)
        return _rgb_to_arc(r, g, b)
    # Try stripping underscores/spaces
    normalized = name.replace(" ", "").replace("_", "")
    if normalized in _COLOR_NAME_TO_ARC:
        return _COLOR_NAME_TO_ARC[normalized]
    return 2  # fallback: gray


_KNOWN_COLOR_NAMES = sorted(_COLOR_NAME_TO_ARC.keys(), key=len, reverse=True)


def _split_named_colors(text: str) -> list[str]:
    """Split concatenated named colors like 'BlackWhite' into ['Black','White']."""
    result: list[str] = []
    remaining = text
    while remaining:
        matched = False
        lower = remaining.lower()
        for name in _KNOWN_COLOR_NAMES:
            if lower.startswith(name):
                result.append(remaining[:len(name)])
                remaining = remaining[len(name):]
                matched = True
                break
        if not matched:
            result.append(remaining)
            break
    return result


def _split_color_tokens(color_line: str) -> list[str]:
    """Split a PuzzleScript color line into individual color tokens.

    Handles concatenated hex codes like '#872e2e#802b2b#9e3636Black'
    and concatenated named colors like 'BlackWhite'.
    """
    tokens: list[str] = []
    for part in re.split(r'\s+', color_line.strip()):
        if not part:
            continue
        if '#' not in part:
            tokens.extend(_split_named_colors(part))
            continue
        remainder = part
        while remainder:
            if remainder.startswith('#'):
                m = re.match(r'#[0-9a-fA-F]{6}', remainder)
                if not m:
                    m = re.match(r'#[0-9a-fA-F]{3}', remainder)
                if m:
                    tokens.append(m.group())
                    remainder = remainder[m.end():]
                else:
                    tokens.append(remainder)
                    break
            else:
                next_hash = remainder.find('#')
                if next_hash == -1:
                    tokens.extend(_split_named_colors(remainder))
                    remainder = ''
                else:
                    tokens.extend(_split_named_colors(remainder[:next_hash]))
                    remainder = remainder[next_hash:]
    return tokens


# ---------------------------------------------------------------------------
# Direction helpers
# ---------------------------------------------------------------------------

# Direction deltas: (dr, dc) for UP, DOWN, LEFT, RIGHT
_DIR_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_DIR_NAMES = ["up", "down", "left", "right"]

# Opposite directions
_OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}

# Direction symbols used in rules
_DIR_SYMBOLS = {">": "right", "<": "left", "^": "up", "v": "down"}


# ---------------------------------------------------------------------------
# PuzzleScript Parser
# ---------------------------------------------------------------------------

class PSObject:
    """A PuzzleScript object definition."""
    def __init__(self, name: str, colors: list[int], sprite: Optional[list[list[int]]] = None):
        self.name = name
        self.colors = colors  # list of ARC palette indices
        self.sprite = sprite  # 5x5 grid of color indices (into self.colors), or None
        # Dominant color for single-cell rendering: most-used non-transparent
        # sprite color, falling back to colors[0].
        self.dominant_color = colors[0] if colors else 2
        if sprite and colors:
            from collections import Counter
            counts: Counter = Counter()
            for row in sprite:
                for ci in row:
                    if 0 <= ci < len(colors) and colors[ci] >= 0:
                        counts[ci] += 1
            if counts:
                best_ci = counts.most_common(1)[0][0]
                self.dominant_color = colors[best_ci]


class PSRule:
    """A parsed PuzzleScript rule."""
    def __init__(self, groups_lhs: list, groups_rhs: list,
                 directions: list[str], is_late: bool = False,
                 again: bool = False, cancel: bool = False,
                 requires_action: bool = False, rigid: bool = False,
                 win: bool = False, restart: bool = False,
                 group_id: int = -1, random: bool = False,
                 loop_id: int = -1):
        self.groups_lhs = groups_lhs    # list of bracket groups, each is list of cell patterns
        self.groups_rhs = groups_rhs
        self.patterns_lhs = groups_lhs[0] if groups_lhs else []  # backward compat
        self.patterns_rhs = groups_rhs[0] if groups_rhs else []
        self.directions = directions  # which directions this rule applies to
        self.is_late = is_late
        self.again = again
        self.cancel = cancel
        self.requires_action = requires_action
        self.rigid = rigid
        self.win = win
        self.restart = restart
        self.group_id = group_id
        self.random = random
        self.loop_id = loop_id


class PSWinCondition:
    """A parsed win condition."""
    def __init__(self, quantifier: str, obj_name: str, on_obj: Optional[str] = None):
        self.quantifier = quantifier  # "all", "no", "some"
        self.obj_name = obj_name
        self.on_obj = on_obj  # for "X on Y" conditions


class PSGame:
    """Parsed PuzzleScript game definition."""

    def __init__(self, game_text: str):
        self.title = "Untitled"
        self.require_player_movement = False
        self.run_rules_on_level_start = False
        self.flickscreen: Optional[tuple[int, int]] = None  # (width, height) or None
        self.objects: dict[str, PSObject] = {}
        self.legend: dict[str, list[str]] = {}  # char/name → list of object names
        self.or_groups: dict[str, list[str]] = {}  # name → list of alternatives
        self.collision_layers: list[list[str]] = []
        self.rules: list[PSRule] = []
        self.win_conditions: list[PSWinCondition] = []
        self.levels: list[list[list[set[int]]]] = []  # parsed levels
        self.obj_name_to_idx: dict[str, int] = {}
        self.obj_idx_to_name: dict[int, str] = {}
        self.n_objects: int = 0
        self._resolve_cache: dict[str, list[int]] = {}
        self._parse(game_text)

    def _parse(self, text: str):
        """Parse a PuzzleScript game file."""
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        lines = text.split("\n")

        # Remove comments (lines starting with parentheses or inline comments).
        # A line that consisted entirely of a comment is dropped — keeping it as
        # a blank line would falsely act as a section separator (e.g. splitting
        # an object's name from its color line in OBJECTS).
        cleaned_lines = []
        comment_depth = 0
        for line in lines:
            original_nonblank = line.strip() != ""
            # PuzzleScript's parser (parser.js parseLevelsToken) captures the
            # rest of a "message" line verbatim as message text — parentheses
            # in messages are NOT comment delimiters. Without this, an
            # unclosed `(` inside a message (e.g. Castle Elsewhere's
            # "(utter beginner set only") opens a multi-line comment that
            # silently eats the following level.
            if comment_depth == 0 and line.lstrip().lower().startswith("message"):
                cleaned_lines.append(line)
                continue
            result = ""
            i = 0
            while i < len(line):
                # PuzzleScript comments nest: ``( outer ( inner ) outer )``
                # is one comment, not two. Track depth so an inner ``)``
                # doesn't reopen emission inside the outer block.
                if line[i] == "(":
                    comment_depth += 1
                    i += 1
                elif line[i] == ")" and comment_depth > 0:
                    comment_depth -= 1
                    i += 1
                elif comment_depth == 0:
                    result += line[i]
                    i += 1
                else:
                    i += 1
            if original_nonblank and result.strip() == "":
                continue
            cleaned_lines.append(result)
        lines = cleaned_lines

        # Find sections
        sections: dict[str, list[str]] = {}
        current_section = "prelude"
        sections["prelude"] = []

        section_headers = {
            "objects", "legend", "sounds", "collisionlayers",
            "rules", "winconditions", "levels"
        }

        for line in lines:
            stripped = line.strip().lower()
            # Section headers are lines of === or the section name
            if stripped.replace("=", "").strip() in section_headers:
                current_section = stripped.replace("=", "").strip()
                if current_section not in sections:
                    sections[current_section] = []
            elif stripped and all(c == "=" for c in stripped):
                # Skip separator lines
                continue
            else:
                if current_section not in sections:
                    sections[current_section] = []
                sections[current_section].append(line)

        # Parse prelude
        self._parse_prelude(sections.get("prelude", []))
        # Parse objects
        self._parse_objects(sections.get("objects", []))
        # Parse legend
        self._parse_legend(sections.get("legend", []))
        # Parse collision layers
        self._parse_collision_layers(sections.get("collisionlayers", []))
        # Parse rules
        self._parse_rules(sections.get("rules", []))
        # Parse win conditions
        self._parse_win_conditions(sections.get("winconditions", []))
        # Parse levels
        self._parse_levels(sections.get("levels", []))
        # Precompute per-cell-pattern metadata for hot paths
        self._compile_patterns()

    def _compile_patterns(self):
        """Precompute per-cell-pattern metadata used by the hot match paths.

        For each unique cell pattern (identified by ``id``), we store:
          - has_ellipsis : bool
          - required_idx_lists : list[list[int]]
                Each inner list is the resolved object indices for one
                "positive presence" constraint (modifier is "" or one of the
                directional/state modifiers that imply presence). At least
                one index from each list must be present in the cell for
                a match. Used as a fast pre-filter.
          - first_required_set : set[int] | None
                Union of indices of the first positive constraint, used as
                a quick candidate test against ``self.grid[r][c]``. None
                means "no fast filter possible" (e.g. only ``no`` modifiers).
        """
        meta: dict[int, dict] = {}

        _MOVEMENT_MODS = frozenset((
            ">", "<", "^", "v",
            "right", "left", "up", "down", "moving",
            "perpendicular", "parallel",
            "horizontal", "vertical",
        ))

        def _compile_cell(cell_pattern):
            key = id(cell_pattern)
            if key in meta:
                return
            has_ellipsis = False
            required_lists: list[list[int]] = []
            has_movement_mod = False
            has_or_group = False
            has_action_mod = False
            # In-place rewrite of cell pattern items from (mod, name) to
            # (mod, name, idxs_tuple), so hot iteration sites can read the
            # resolved object indices without calling resolve_object_name.
            for i in range(len(cell_pattern)):
                item = cell_pattern[i]
                if len(item) == 3:
                    mod, name, _ = item
                else:
                    mod, name = item
                if mod == "...":
                    has_ellipsis = True
                    cell_pattern[i] = (mod, name, ())
                    continue
                if mod in _MOVEMENT_MODS:
                    has_movement_mod = True
                if mod == "action":
                    has_action_mod = True
                if name and name in self.or_groups:
                    has_or_group = True
                if name:
                    idxs_tuple = tuple(self.resolve_object_name(name))
                else:
                    idxs_tuple = ()
                cell_pattern[i] = (mod, name, idxs_tuple)
                # Modifiers that imply the object must be present in the cell.
                if mod in ("", ">", "<", "^", "v",
                           "right", "left", "up", "down",
                           "stationary", "moving", "action",
                           "perpendicular", "parallel",
                           "horizontal", "vertical"):
                    if name and idxs_tuple:
                        required_lists.append(list(idxs_tuple))
            first_required: set[int] | None = None
            if required_lists:
                first_required = set(required_lists[0])
            meta[key] = {
                "has_ellipsis": has_ellipsis,
                "required_idx_lists": required_lists,
                "first_required": first_required,
                "has_movement_mod": has_movement_mod,
                "has_or_group": has_or_group,
                "has_action_mod": has_action_mod,
            }

        def _walk(groups):
            for group in groups:
                for cell_pattern in group:
                    _compile_cell(cell_pattern)

        def _any_flag(groups, flag_key):
            for group in groups:
                for cell_pattern in group:
                    m = meta.get(id(cell_pattern))
                    if m is not None and m[flag_key]:
                        return True
            return False

        for rule in self.rules:
            _walk(rule.groups_lhs)
            _walk(rule.groups_rhs)
            # Per-rule cache of "first cell" required-set unioned across the
            # entire LHS first cell — usable by the engine to pre-select
            # starting positions.
            rule._first_cell_required_per_group = []
            for group_lhs in rule.groups_lhs:
                if group_lhs:
                    fc_meta = meta.get(id(group_lhs[0]))
                    rule._first_cell_required_per_group.append(
                        fc_meta["first_required"] if fc_meta else None
                    )
                else:
                    rule._first_cell_required_per_group.append(None)

            # Per-rule feature flags so the hot apply path can skip dead
            # code branches when the rule doesn't use that feature.
            rule._lhs_has_or_group = _any_flag(rule.groups_lhs, "has_or_group")
            rule._rhs_has_or_group = _any_flag(rule.groups_rhs, "has_or_group")
            rule._lhs_has_movement_mod = _any_flag(rule.groups_lhs, "has_movement_mod")
            rule._rhs_has_movement_mod = _any_flag(rule.groups_rhs, "has_movement_mod")
            rule._lhs_has_action_mod = _any_flag(rule.groups_lhs, "has_action_mod")
            rule._rhs_has_action_mod = _any_flag(rule.groups_rhs, "has_action_mod")

            # "Trivial" rule: pure object substitution at each matched cell,
            # no force handling, no or-group binding, no action state, no
            # random, no 'no' / 'random' modifiers on RHS, no ellipsis, no
            # rigid grouping. The fast apply path replaces the per-cell LHS
            # index set with the per-cell RHS index set (with same-layer
            # eviction) — bypassing the ~10-pass generic apply function.
            rule._is_trivial = self._compute_is_trivial(rule)

            # "Safely cacheable" rule: matching depends only on grid state
            # (no movement / action / random / rigid involvement), and any
            # side effects on the forces dict or action set can only occur
            # together with a grid mutation. So a previous call that left
            # ``_mutation_counter`` unchanged is provably a complete no-op
            # — the engine can replay its cached ``fired`` flag without
            # re-scanning the grid. This kills the dominant per-step cost
            # in + groups whose LHS keeps matching at steady state but
            # whose RHS makes no change (CrateBlob's Checker tiling and
            # Croff propagation each spin 200 such no-op iterations per
            # step).
            rule._is_safely_cacheable = self._compute_is_safely_cacheable(rule)
            if rule._is_trivial:
                rule._trivial_lhs_per_group = [
                    [self._compute_cell_indices(cell, side="lhs") for cell in group]
                    for group in rule.groups_lhs
                ]
                rule._trivial_rhs_per_group = [
                    [self._compute_cell_indices(cell, side="rhs") for cell in group]
                    for group in rule.groups_rhs
                ]

        self._pattern_meta = meta

    def _compute_is_trivial(self, rule) -> bool:
        """A rule is 'trivial' if applying it at a match is just a per-cell
        object replacement. ``no <or-group>`` on LHS is allowed (it does not
        require binding, the match guarantees the negative condition).
        """
        if rule.rigid or rule.random:
            return False
        _SPECIAL_MODS = frozenset((
            ">", "<", "^", "v",
            "right", "left", "up", "down",
            "moving", "perpendicular", "parallel",
            "horizontal", "vertical", "action",
        ))
        # ``stationary`` is match-only on the LHS (``_cell_matches_forces``
        # asserts the object has no pending force), so it does not disqualify
        # the fast path there. On the RHS it *writes*: it drops the object's
        # pending force. The fast path only substitutes objects, so a rule
        # whose RHS merely re-states its LHS with a ``stationary`` marker —
        # VEXT_EDIT's ``[player1 cant1] -> [stationary player1 cant1]`` — looks
        # like an identity substitution and is skipped entirely. The blocked
        # pusher then keeps its force, and a rigid multi-cell shape shears
        # apart when only part of it is up against a wall.
        _RHS_SPECIAL_MODS = _SPECIAL_MODS | {"stationary"}
        for group in rule.groups_lhs:
            for cell in group:
                for item in cell:
                    mod = item[0]
                    name = item[1]
                    if mod == "...":
                        return False
                    if mod == "random":
                        return False
                    if mod == "no":
                        # 'no X' on LHS just asserts absence; no binding needed
                        # even if X is an or-group.
                        continue
                    if mod in _SPECIAL_MODS:
                        return False
                    if name and name in self.or_groups:
                        return False
        for group in rule.groups_rhs:
            for cell in group:
                for item in cell:
                    mod = item[0]
                    name = item[1]
                    if mod == "...":
                        return False
                    if mod == "no" or mod == "random":
                        return False
                    if mod in _RHS_SPECIAL_MODS:
                        return False
                    if name and name in self.or_groups:
                        return False
        return True

    def _compute_is_safely_cacheable(self, rule) -> bool:
        """A rule is 'safely cacheable' iff its matching depends only on
        the grid (no movement / action / ellipsis modifiers, no
        random/rigid, no command flags) AND any forces.pop / action_objects
        side effects occur only when an LHS object is actually present
        in the cell — which in turn requires a grid mutation
        (``_cell_discard`` is what fires forces.pop in those code paths).

        Under those conditions, a previous call that left
        ``_mutation_counter`` unchanged from its start is provably a
        complete no-op: no grid change, no forces change, no action
        change. Replaying its cached ``fired`` flag is exact.

        Excludes any rule that uses a force / action / movement modifier
        on either side (forces and the action set both affect matching
        and may be written), random / rigid (RNG and rigid-group side
        effects), ``...`` ellipsis (complex matching kept on the slow
        path), and command flags (cancel / win / restart — the cheap
        path, not worth caching).
        """
        if rule.rigid or rule.random:
            return False
        if rule.cancel or rule.win or rule.restart:
            return False
        _DISALLOWED = frozenset((
            "action",
            ">", "<", "^", "v",
            "right", "left", "up", "down",
            "moving", "stationary",
            "perpendicular", "parallel",
            "horizontal", "vertical",
        ))
        for group in rule.groups_lhs:
            for cell in group:
                for item in cell:
                    mod = item[0]
                    if mod == "...":
                        return False
                    if mod in _DISALLOWED:
                        return False
        for group in rule.groups_rhs:
            for cell in group:
                for item in cell:
                    mod = item[0]
                    if mod == "...":
                        return False
                    if mod == "no":
                        return False
                    if mod == "random":
                        return False
                    if mod in _DISALLOWED:
                        return False
        return True

    def _compute_cell_indices(self, cell_pattern, side: str) -> frozenset:
        """For a trivial cell pattern, return the set of object indices the
        cell carries after matching/applying. For LHS this is the set of
        objects required-present (positive-presence items, excluding 'no').
        For RHS this is the set of objects the cell ends up containing.
        """
        indices: set = set()
        for item in cell_pattern:
            mod = item[0]
            if mod == "no" or mod == "...":
                continue
            if len(item) >= 3:
                _name = item[1]
                idxs = item[2]
            else:
                _name = item[1]
                idxs = ()
            if _name and idxs:
                indices.update(idxs)
        return frozenset(indices)

    def _parse_prelude(self, lines: list[str]):
        for line in lines:
            stripped = line.strip().lower()
            if stripped.startswith("title"):
                self.title = line.strip()[6:].strip()
            elif "require_player_movement" in stripped:
                self.require_player_movement = True
            elif "run_rules_on_level_start" in stripped:
                self.run_rules_on_level_start = True
            elif stripped.startswith("flickscreen"):
                m = re.match(r'flickscreen\s+(\d+)x(\d+)', stripped)
                if m:
                    self.flickscreen = (int(m.group(1)), int(m.group(2)))

    def _parse_objects(self, lines: list[str]):
        """Parse OBJECTS section."""
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if not line:
                i += 1
                continue

            # Object name line: "ObjectName [CharAlias]"
            # CharAlias is a single character used in levels to represent this object
            tokens = line.split()
            obj_name = tokens[0].lower()
            char_alias = None
            if len(tokens) >= 2 and len(tokens[1]) == 1:
                char_alias = tokens[1].lower()
            i += 1

            if i >= len(lines):
                break

            # Color line
            color_line = lines[i].strip()
            if not color_line:
                i += 1
                continue

            colors_strs = _split_color_tokens(color_line)
            colors = [_color_name_to_arc(c) for c in colors_strs]
            i += 1

            # Check for sprite lines (5 lines of digits/dots)
            sprite = None
            sprite_lines = []
            while i < len(lines) and len(lines[i].strip()) > 0:
                sline = lines[i].strip()
                if all(c in "0123456789." for c in sline):
                    sprite_lines.append(sline)
                    i += 1
                else:
                    break

            if sprite_lines:
                sprite = []
                for sl in sprite_lines:
                    row = []
                    for ch in sl:
                        if ch == ".":
                            row.append(-1)  # transparent
                        else:
                            row.append(int(ch))
                    sprite.append(row)

            obj = PSObject(obj_name, colors, sprite)
            self.objects[obj_name] = obj

            # Assign index
            idx = len(self.obj_name_to_idx)
            self.obj_name_to_idx[obj_name] = idx
            self.obj_idx_to_name[idx] = obj_name

            # Register character alias in legend (used in level data)
            if char_alias is not None:
                self.legend[char_alias] = [obj_name]

        self.n_objects = len(self.obj_name_to_idx)

    def _parse_legend(self, lines: list[str]):
        """Parse LEGEND section."""
        for line in lines:
            line = line.strip()
            if not line or "=" not in line:
                continue

            parts = line.split("=", 1)
            char_or_name = parts[0].strip().lower()
            definition = parts[1].strip().lower()

            if " and " in definition:
                # Composite: char = ObjA and ObjB
                obj_names = [n.strip() for n in definition.split(" and ")]
                self.legend[char_or_name] = obj_names
            elif " or " in definition:
                # Group: Name = ObjA or ObjB or ObjC
                obj_names = [n.strip() for n in definition.split(" or ")]
                self.or_groups[char_or_name] = obj_names
                self.legend[char_or_name] = obj_names  # first as default
            else:
                self.legend[char_or_name] = [definition.strip()]

    def _parse_collision_layers(self, lines: list[str]):
        """Parse COLLISIONLAYERS section."""
        for line in lines:
            line = line.strip()
            if not line:
                continue
            objs = [o.strip().lower() for o in re.split(r'[,\s]+', line) if o.strip()]
            self.collision_layers.append(objs)

    def _parse_rules(self, lines: list[str]):
        """Parse RULES section."""
        next_group_id = 0
        current_group_id = -1
        in_loop = False
        next_loop_id = 0
        current_loop_id = -1
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Skip message lines
            if line.lower().startswith("message"):
                continue

            # Handle startloop/endloop
            if line.lower() == "startloop":
                in_loop = True
                current_loop_id = next_loop_id
                next_loop_id += 1
                current_group_id = -1
                continue
            if line.lower() == "endloop":
                in_loop = False
                current_loop_id = -1
                current_group_id = -1
                continue

            # Handle '+' prefix for grouped rules (loop until stable)
            is_grouped = False
            if line.startswith("+"):
                is_grouped = True
                line = line[1:].strip()

            is_late = False
            if line.lower().startswith("late"):
                is_late = True
                line = line[4:].strip()

            is_rigid = False
            if line.lower().startswith("rigid"):
                is_rigid = True
                line = line[5:].strip()

            is_random = False
            if line.lower().startswith("random"):
                is_random = True
                line = line[6:].strip()

            # Determine directions
            directions = []
            rule_line = line
            for dir_prefix in ["up", "down", "left", "right", "horizontal", "vertical"]:
                lower = rule_line.lower()
                if lower.startswith(dir_prefix + " ") or lower.startswith(dir_prefix + "["):
                    if dir_prefix == "horizontal":
                        directions = ["left", "right"]
                    elif dir_prefix == "vertical":
                        directions = ["up", "down"]
                    else:
                        directions = [dir_prefix]
                    rule_line = rule_line[len(dir_prefix):].strip()
                    break

            if not directions:
                directions = ["up", "down", "left", "right"]

            # Parse rule pattern: [ ... ] -> [ ... ]
            if "->" not in rule_line:
                continue

            arrow_idx = rule_line.index("->")
            lhs_str = rule_line[:arrow_idx].strip()
            rhs_str = rule_line[arrow_idx + 2:].strip()

            # Remove sound effects from rhs (case-insensitive)
            rhs_str = re.sub(r'\s+sfx\d+', '', rhs_str, flags=re.IGNORECASE)

            # Check for 'again' keyword
            again = False
            if re.search(r'\bagain\b', rhs_str, re.IGNORECASE):
                again = True
                rhs_str = re.sub(r'\bagain\b', '', rhs_str, flags=re.IGNORECASE).strip()

            # Detect trailing command keywords (win, restart, cancel) that follow
            # the final bracket pattern, e.g. "[A] -> [B] Win". Strip them and set
            # flags; the rule will both transform AND fire the command.
            trailing_win = False
            trailing_restart = False
            trailing_cancel = False
            if ']' in rhs_str:
                last_bracket = rhs_str.rfind(']')
                trailing_part = rhs_str[last_bracket + 1:]
                kept_tokens = []
                for tok in trailing_part.split():
                    low = tok.lower()
                    if low == 'win':
                        trailing_win = True
                    elif low == 'restart':
                        trailing_restart = True
                    elif low == 'cancel':
                        trailing_cancel = True
                    else:
                        kept_tokens.append(tok)
                rhs_str = rhs_str[:last_bracket + 1]
                if kept_tokens:
                    rhs_str = rhs_str + ' ' + ' '.join(kept_tokens)
                rhs_str = rhs_str.strip()

            lhs_patterns = self._parse_rule_side(lhs_str)

            # Detect if any LHS cell uses the "action" modifier on the Player.
            # The "action" modifier has dual semantics in PuzzleScript: on the
            # Player it gates the rule on the action key being pressed, but on
            # any other object it is just an internal force set by another
            # rule's RHS (e.g. laser propagation in Ad Infinitum, where
            # [action LaserU] is set by the firing rule and consumed by the
            # propagation rule on the same turn). Only the player-action form
            # should require an action turn.
            player_indices = set(self.resolve_object_name("player"))
            req_action = False
            for group in lhs_patterns:
                for cell in group:
                    for modifier, obj_name in cell:
                        if modifier == "action" and obj_name:
                            obj_indices = set(self.resolve_object_name(obj_name))
                            if obj_indices & player_indices:
                                req_action = True
                                break
                    if req_action:
                        break
                if req_action:
                    break

            # Assign group_id: '+' groups rules into a sub-loop
            if is_grouped and current_group_id >= 0:
                gid = current_group_id
            else:
                gid = next_group_id
                next_group_id += 1
                current_group_id = gid

            lid = current_loop_id if in_loop else -1

            # Handle cancel rules (RHS is just "cancel", no brackets)
            if rhs_str.strip().lower() == "cancel":
                if lhs_patterns:
                    self.rules.append(PSRule(lhs_patterns, [], directions, is_late,
                                            cancel=True, requires_action=req_action,
                                            rigid=is_rigid, group_id=gid,
                                            random=is_random, loop_id=lid))
                continue

            rhs_lower = rhs_str.strip().lower()

            if rhs_lower == "win" or (not rhs_lower and trailing_win and not trailing_restart and not trailing_cancel):
                if lhs_patterns:
                    self.rules.append(PSRule(lhs_patterns, [], directions, is_late,
                                            win=True, requires_action=req_action,
                                            rigid=is_rigid, group_id=gid,
                                            random=is_random, loop_id=lid))
                continue

            if rhs_lower == "restart" or (not rhs_lower and trailing_restart and not trailing_win and not trailing_cancel):
                if lhs_patterns:
                    self.rules.append(PSRule(lhs_patterns, [], directions, is_late,
                                            restart=True, requires_action=req_action,
                                            rigid=is_rigid, group_id=gid,
                                            random=is_random, loop_id=lid))
                continue

            rhs_patterns = self._parse_rule_side(rhs_str)

            if lhs_patterns and rhs_patterns:
                self.rules.append(PSRule(lhs_patterns, rhs_patterns, directions, is_late,
                                        again, requires_action=req_action,
                                        rigid=is_rigid, group_id=gid,
                                        random=is_random, loop_id=lid,
                                        cancel=trailing_cancel,
                                        win=trailing_win,
                                        restart=trailing_restart))
            elif lhs_patterns and not rhs_patterns and again:
                # Bare ``[X] -> again`` rule: no grid transformation, just
                # re-trigger the rule loop while the LHS is present (used
                # e.g. by botsket_ball's playing-state animation cycle).
                self.rules.append(PSRule(lhs_patterns, lhs_patterns, directions, is_late,
                                        again=True, requires_action=req_action,
                                        rigid=is_rigid, group_id=gid,
                                        random=is_random, loop_id=lid,
                                        cancel=trailing_cancel,
                                        win=trailing_win,
                                        restart=trailing_restart))

    def _parse_rule_side(self, s: str) -> list[list[list[tuple[str, str]]]]:
        """Parse one side of a rule: [ cell1 | cell2 ] [ cell3 | ... ] ...

        Returns list of bracket groups.  Each group is a list of cells
        (separated by ``|`` inside one ``[…]``).  Each cell is a list of
        ``(modifier, object_name)`` tuples.
        modifier can be: "", ">", "<", "^", "v", "no", "...", "action",
        "stationary", "moving", "right", "left", "up", "down"
        """
        # Find ALL bracket groups
        bracket_matches = re.findall(r'\[(.*?)\]', s)
        if not bracket_matches:
            return []

        groups = []
        for bracket_content in bracket_matches:
            content = bracket_content.strip()
            cells = content.split("|")
            group = []

            for cell in cells:
                cell = cell.strip()
                tokens = cell.split()
                cell_items = []

                i = 0
                while i < len(tokens):
                    token = tokens[i].lower()

                    if token in (">", "<", "^", "v"):
                        # Direction modifier followed by object name
                        if i + 1 < len(tokens):
                            obj = tokens[i + 1].lower()
                            cell_items.append((token, obj))
                            i += 2
                        else:
                            i += 1
                    elif token in ("right", "left", "up", "down", "perpendicular", "parallel",
                                   "horizontal", "vertical"):
                        if i + 1 < len(tokens):
                            obj = tokens[i + 1].lower()
                            cell_items.append((token, obj))
                            i += 2
                        else:
                            cell_items.append(("", token))
                            i += 1
                    elif token == "no":
                        if i + 1 < len(tokens):
                            obj = tokens[i + 1].lower()
                            cell_items.append(("no", obj))
                            i += 2
                        else:
                            i += 1
                    elif token in ("stationary", "moving", "action"):
                        if i + 1 < len(tokens):
                            obj = tokens[i + 1].lower()
                            cell_items.append((token, obj))
                            i += 2
                        else:
                            i += 1
                    elif token == "random":
                        if i + 1 < len(tokens):
                            obj = tokens[i + 1].lower()
                            cell_items.append(("random", obj))
                            i += 2
                        else:
                            i += 1
                    elif token == "...":
                        cell_items.append(("...", ""))
                        i += 1
                    else:
                        cell_items.append(("", token))
                        i += 1

                group.append(cell_items)
            groups.append(group)

        return groups

    def _parse_win_conditions(self, lines: list[str]):
        """Parse WINCONDITIONS section."""
        for line in lines:
            line = line.strip().lower()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 2:
                continue

            quantifier = parts[0]  # all, no, some, any
            if quantifier not in ("all", "no", "some", "any"):
                continue
            if quantifier == "any":
                quantifier = "some"

            obj_name = parts[1]
            on_obj = None

            if "on" in parts:
                on_idx = parts.index("on")
                if on_idx + 1 < len(parts):
                    on_obj = parts[on_idx + 1]

            self.win_conditions.append(PSWinCondition(quantifier, obj_name, on_obj))

    def _parse_levels(self, lines: list[str]):
        """Parse LEVELS section into grid states."""
        current_level_lines = []

        for line in lines:
            # Message lines separate levels
            if line.strip().lower().startswith("message"):
                if current_level_lines:
                    level = self._build_level(current_level_lines)
                    if level is not None:
                        self.levels.append(level)
                    current_level_lines = []
                continue

            if not line.strip():
                if current_level_lines:
                    level = self._build_level(current_level_lines)
                    if level is not None:
                        self.levels.append(level)
                    current_level_lines = []
                continue

            current_level_lines.append(line)

        if current_level_lines:
            level = self._build_level(current_level_lines)
            if level is not None:
                self.levels.append(level)

    def _build_level(self, lines: list[str]) -> Optional[list[list[set[int]]]]:
        """Convert level text lines to grid of object index sets."""
        if not lines:
            return None

        # Strip trailing whitespace but preserve internal structure
        height = len(lines)
        width = max(len(line) for line in lines)

        if width == 0 or height == 0:
            return None

        grid = [[set() for _ in range(width)] for _ in range(height)]

        for r, line in enumerate(lines):
            for c, ch in enumerate(line):
                if c >= width:
                    break
                obj_names = self._char_to_objects(ch)
                for name in obj_names:
                    if name in self.obj_name_to_idx:
                        grid[r][c].add(self.obj_name_to_idx[name])
                    elif name in self.or_groups:
                        # For "or" groups in legend, place the first option
                        for sub in self.or_groups[name]:
                            if sub in self.obj_name_to_idx:
                                grid[r][c].add(self.obj_name_to_idx[sub])
                                break
                    elif name in self.legend:
                        for sub in self.legend[name]:
                            if sub in self.obj_name_to_idx:
                                grid[r][c].add(self.obj_name_to_idx[sub])
                            elif sub in self.or_groups:
                                for subsub in self.or_groups[sub]:
                                    if subsub in self.obj_name_to_idx:
                                        grid[r][c].add(self.obj_name_to_idx[subsub])
                                        break

        # PuzzleScript: Background is implicitly present on every cell.
        # Add it to cells that don't already have an object on the background layer.
        bg_idx = self.obj_name_to_idx.get("background")
        bg_default_idx = bg_idx
        if bg_default_idx is None and "background" in self.or_groups:
            first_bg = self.or_groups["background"][0]
            bg_default_idx = self.obj_name_to_idx.get(first_bg)
        if bg_default_idx is None and "background" in self.legend:
            for sub in self.legend["background"]:
                if sub in self.obj_name_to_idx:
                    bg_default_idx = self.obj_name_to_idx[sub]
                    break
        if bg_default_idx is not None:
            bg_layer = self.get_collision_layer(bg_default_idx)
            bg_layer_indices = set()
            if bg_layer >= 0:
                for obj_name in (self.collision_layers[bg_layer] if bg_layer < len(self.collision_layers) else []):
                    if obj_name in self.obj_name_to_idx:
                        bg_layer_indices.add(self.obj_name_to_idx[obj_name])
                    elif obj_name in self.or_groups:
                        for sub in self.or_groups[obj_name]:
                            if sub in self.obj_name_to_idx:
                                bg_layer_indices.add(self.obj_name_to_idx[sub])
            for r in range(height):
                for c in range(width):
                    if not (grid[r][c] & bg_layer_indices):
                        grid[r][c].add(bg_default_idx)

        return grid

    def _char_to_objects(self, ch: str) -> list[str]:
        """Map a level character to object names."""
        ch_lower = ch.lower()

        # Check legend first
        if ch_lower in self.legend:
            return self.legend[ch_lower]

        # Check direct object name match (single char objects)
        if ch_lower in self.obj_name_to_idx:
            return [ch_lower]

        # Background fallback for spaces
        if ch == " ":
            if "background" in self.obj_name_to_idx:
                return ["background"]
            if "background" in self.legend:
                return self.legend["background"][:1]
            if "background" in self.or_groups:
                # Use first object in the or-group
                for sub in self.or_groups["background"]:
                    if sub in self.obj_name_to_idx:
                        return [sub]

        return []

    def resolve_object_name(self, name: str) -> list[int]:
        """Resolve an object name (possibly a group/legend entry) to object indices."""
        cached = self._resolve_cache.get(name)
        if cached is not None:
            return cached

        lname = name.lower()
        cached = self._resolve_cache.get(lname)
        if cached is not None:
            self._resolve_cache[name] = cached
            return cached

        if lname in self.obj_name_to_idx:
            result = [self.obj_name_to_idx[lname]]
        elif lname in self.or_groups:
            result = []
            for sub in self.or_groups[lname]:
                result.extend(self.resolve_object_name(sub))
        elif lname in self.legend:
            result = []
            for sub in self.legend[lname]:
                result.extend(self.resolve_object_name(sub))
        else:
            result = []

        self._resolve_cache[lname] = result
        if name != lname:
            self._resolve_cache[name] = result
        return result

    def _group_contains(self, group_name: str, target: str, visited: set[str] | None = None) -> bool:
        """Recursively check if *target* is a member of the OR-group or legend entry *group_name*."""
        if visited is None:
            visited = set()
        if group_name in visited:
            return False
        visited.add(group_name)

        members: list[str] = []
        if group_name in self.or_groups:
            members = self.or_groups[group_name]
        elif group_name in self.legend:
            members = self.legend[group_name]
        else:
            return False

        for m in members:
            if m == target:
                return True
            if self._group_contains(m, target, visited):
                return True
        return False

    def get_collision_layer(self, obj_idx: int) -> int:
        """Get which collision layer an object is on (-1 if not found).

        When an object appears on multiple layers (directly or via groups),
        the last layer wins — standard PuzzleScript semantics.
        """
        name = self.obj_idx_to_name.get(obj_idx, "")
        result = -1
        for layer_i, layer in enumerate(self.collision_layers):
            if name in layer:
                result = layer_i
                continue
            for layer_obj in layer:
                if self._group_contains(layer_obj, name):
                    result = layer_i
                    break
        return result


# ---------------------------------------------------------------------------
# Game State Engine
# ---------------------------------------------------------------------------

class PSEngine:
    """Runs a PuzzleScript game: applies rules, checks win conditions.

    Movement model (faithful to PuzzleScript semantics):
      1. Player receives a directional force from the input.
      2. Rules are evaluated: patterns with ">" match objects that have a
         pending force. Rules can propagate forces (e.g., [ > Player | Crate ]
         -> [ > Player | > Crate ] gives the crate a force too).
      3. All objects with forces attempt to move. Movement is blocked if a
         same-collision-layer object exists in the target cell (and that object
         doesn't also have a compatible force).
    """

    def __init__(self, game: PSGame):
        self.game = game
        self.grid: list[list[set[int]]] = []
        self.height: int = 0
        self.width: int = 0
        self._player_indices: list[int] = []
        self._init_player_indices()
        # Precompute collision layer for each object
        self._obj_layers: dict[int, int] = {}
        for idx in range(game.n_objects):
            self._obj_layers[idx] = game.get_collision_layer(idx)
        self._rule_win = False
        self._rule_restart = False
        self._late_cancel = False
        self._again_triggered = False
        self._capture_snapshots = False
        self._snapshots: list[list[list[set[int]]]] = []
        # Inverted index: object idx -> set of (r, c) positions.
        # Maintained incrementally by _cell_add / _cell_discard; rebuilt
        # only after full grid replacement (load_level, cancel revert).
        self._object_positions: dict[int, set[tuple[int, int]]] = {}
        self._position_index_dirty: bool = True
        # Monotonic counter bumped by _cell_add / _cell_discard whenever the
        # grid actually changes. Used in place of deep-copy + cell-compare
        # to detect "did this iteration change anything?".
        self._mutation_counter: int = 0
        # Per-rule cache: id(rule) -> (mut_counter_at_call, fired). Used
        # only for rules where ``_is_safely_cacheable`` is True. A cache
        # entry is created when a call returns with ``_mutation_counter``
        # unchanged from its start (i.e. the call was a complete no-op on
        # the grid; for safely-cacheable rules that implies no forces /
        # action_objects change either). Subsequent calls at the same
        # ``_mutation_counter`` replay the cached ``fired`` flag without
        # re-scanning. Cleared on level reload and at the start of every
        # ``_apply_rules_with_forces`` pass.
        self._rule_noop_cache: dict[int, tuple[int, bool]] = {}
        # Memoized or-group co-membership for force inheritance during
        # same-layer swaps. (a, b) is True iff a and b are alternate
        # members of the same or-group (e.g., PlaneU/PlaneD in Plane).
        self._share_or_group_cache: dict[tuple[int, int], bool] = {}
        # Rigid rule tracking: forces produced by a single rigid rule
        # application share a group id. If any force in the group is blocked
        # during resolution, all forces in the group are cancelled together.
        self._force_rigid_group: dict[tuple[int, int, int], int] = {}
        self._rigid_group_counter: int = 0

    def _build_position_index(self):
        """Rebuild the object-position inverted index from the grid."""
        idx: dict[int, set[tuple[int, int]]] = {}
        grid = self.grid
        for r in range(self.height):
            row = grid[r]
            for c in range(self.width):
                for obj_idx in row[c]:
                    s = idx.get(obj_idx)
                    if s is None:
                        s = set()
                        idx[obj_idx] = s
                    s.add((r, c))
        self._object_positions = idx
        self._position_index_dirty = False

    def _cell_add(self, r: int, c: int, obj_idx: int) -> None:
        """Add obj_idx to grid[r][c], keeping the position index in sync
        and bumping the mutation counter (used for cheap change detection).
        """
        cell = self.grid[r][c]
        if obj_idx in cell:
            return
        cell.add(obj_idx)
        self._mutation_counter += 1
        if self._position_index_dirty:
            return
        s = self._object_positions.get(obj_idx)
        if s is None:
            s = set()
            self._object_positions[obj_idx] = s
        s.add((r, c))

    def _cell_discard(self, r: int, c: int, obj_idx: int) -> None:
        """Remove obj_idx from grid[r][c], keeping the position index in sync
        and bumping the mutation counter.
        """
        cell = self.grid[r][c]
        if obj_idx not in cell:
            return
        cell.discard(obj_idx)
        self._mutation_counter += 1
        if self._position_index_dirty:
            return
        s = self._object_positions.get(obj_idx)
        if s is not None:
            s.discard((r, c))

    def _candidate_positions(self, required_set):
        """Return iterable of (r, c) positions that could match a rule whose
        first cell requires at least one of the indices in ``required_set``.

        If ``required_set`` is None or empty, returns None (caller should
        fall back to scanning every cell).
        """
        if not required_set:
            return None
        if self._position_index_dirty:
            self._build_position_index()
        positions: set[tuple[int, int]] = set()
        idx = self._object_positions
        for obj_idx in required_set:
            s = idx.get(obj_idx)
            if s:
                positions.update(s)
        return positions

    def _init_player_indices(self):
        """Find which object indices correspond to 'player'."""
        self._player_indices = self.game.resolve_object_name("player")

    def _objects_share_or_group(self, idx_a: int, idx_b: int) -> bool:
        """True if idx_a and idx_b are alternate members of the same or-group."""
        if idx_a == idx_b:
            return False
        key = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
        cached = self._share_or_group_cache.get(key)
        if cached is not None:
            return cached
        name_a = self.game.obj_idx_to_name.get(idx_a)
        name_b = self.game.obj_idx_to_name.get(idx_b)
        result = False
        if name_a and name_b and name_a != name_b:
            for group_name in self.game.or_groups:
                if (self.game._group_contains(group_name, name_a)
                        and self.game._group_contains(group_name, name_b)):
                    result = True
                    break
        self._share_or_group_cache[key] = result
        return result

    def load_level(self, level_data: list[list[set[int]]]):
        """Load a level grid (deep copy)."""
        self.height = len(level_data)
        self.width = len(level_data[0]) if level_data else 0
        self.grid = [[cell.copy() for cell in row] for row in level_data]
        self._rule_win = False
        self._rule_restart = False
        self._late_cancel = False
        self._position_index_dirty = True
        self._force_rigid_group = {}
        self._rule_noop_cache.clear()
        if self.game.run_rules_on_level_start:
            # Run exactly ONE tick at level start (regular rules → forces →
            # late rules), repeating only when a rule triggers `again`.
            # Looping until the grid stabilises would over-fire turn-scoped
            # late rules — e.g. Autumn's bunny-walks-toward-player rule
            # would step the bunny once per outer iteration and end up
            # adjacent to the player at level start instead of staying at
            # the far edge.
            forces: dict[tuple[int, int, int], str] = {}
            action_objects: set[tuple[int, int, int]] = set()
            max_again = 50
            for _again_iter in range(max_again):
                self._again_triggered = False
                iter_start_grid = [[frozenset(cell) for cell in row] for row in self.grid]
                _, any_main_again = self._apply_rules_with_forces(
                    "action", forces, late=False,
                    is_action_turn=False,
                    action_objects=action_objects,
                )
                self._resolve_forces(forces)
                self._apply_late_rules("action")
                any_late_again = self._again_triggered
                # `again` re-triggers only when the grid actually changed
                # this iteration (PuzzleScript reference semantics).
                self._again_triggered = False
                if any_main_again or any_late_again:
                    for r in range(self.height):
                        if self._again_triggered:
                            break
                        row_before = iter_start_grid[r]
                        row_now = self.grid[r]
                        for c in range(self.width):
                            if row_now[c] != row_before[c]:
                                self._again_triggered = True
                                break
                if self._late_cancel or not self._again_triggered:
                    break
                forces = {}
                self._force_rigid_group = {}
                action_objects.clear()
            self._rule_win = False
            self._rule_restart = False
            self._late_cancel = False

    def step(self, direction: str) -> bool:
        """Execute one game step in the given direction. Returns True if state changed."""
        if direction not in _DIR_NAMES and direction != "action":
            return False

        self._rule_win = False
        self._rule_restart = False
        self._late_cancel = False

        # Save state for require_player_movement check and cancel
        old_player_positions = self._get_player_positions()
        old_grid = [[cell.copy() for cell in row] for row in self.grid]

        # Phase 1: Assign forces — player gets a force in the pressed direction
        # forces maps (row, col, obj_idx) -> direction
        forces: dict[tuple[int, int, int], str] = {}
        self._force_rigid_group = {}

        # Track per-object action state (consumed when a rule matches
        # [Action X] on LHS but RHS has [X] without Action)
        action_objects: set[tuple[int, int, int]] = set()

        if direction != "action":
            for r in range(self.height):
                for c in range(self.width):
                    for pi in self._player_indices:
                        if pi in self.grid[r][c]:
                            forces[(r, c, pi)] = direction
        else:
            for r in range(self.height):
                for c in range(self.width):
                    for pi in self._player_indices:
                        if pi in self.grid[r][c]:
                            action_objects.add((r, c, pi))

        # Phase 2+3: Apply rules once, then resolve forces.
        # PuzzleScript execution model (per-turn):
        #   1. Apply all rules in order (single pass — force propagation
        #      happens via each rule's internal scan order).
        #   2. Resolve forces (actually move objects).
        #   3. If the grid changed, goto 1 (PuzzleScript re-runs rules
        #      until the grid stabilises within a single turn).
        # Cancel rules fire during step 1; because they appear before
        # gravity in the rule list, they only see forces from earlier rules.
        is_action_turn = (direction == "action")
        if self._capture_snapshots:
            self._snapshots = []
        max_again = 50
        for _again_iter in range(max_again):
            self._again_triggered = False
            iter_start_grid = [[frozenset(cell) for cell in row] for row in self.grid]

            cancelled, any_main_again = self._apply_rules_with_forces(
                direction, forces, late=False,
                is_action_turn=(is_action_turn and _again_iter == 0),
                action_objects=action_objects,
            )
            if cancelled:
                # Cancel reverts the entire turn: revert the grid and drop
                # all pending forces (PuzzleScript semantics — cancel undoes
                # any rule-driven grid changes made earlier in the turn,
                # e.g. Stand Off's "validate number of guns" cancel that
                # rejects a 3rd gun drawn by an earlier rule).
                self.grid = [[cell.copy() for cell in row] for row in old_grid]
                self._position_index_dirty = True
                forces.clear()
                self._force_rigid_group = {}
                self._rule_restart = False
                self._rule_win = False
                return False

            self._resolve_forces(forces)

            if self._capture_snapshots:
                self._snapshots.append(
                    [[cell.copy() for cell in row] for row in self.grid]
                )

            self._apply_late_rules(direction)
            any_late_again = self._again_triggered

            # PuzzleScript re-triggers `again` only when the grid actually
            # changed over the full iteration (main rules + forces + late
            # rules), not just when a rule with `again` matched. Without
            # this, rules like Dharma Dojo's
            # ``[Block no MatchToken | Hopper] -> [> Block … | Hopper] again``
            # spin forever once a block in the hopper is blocked: the rule
            # keeps matching, sets a RIGHT force, force resolution cancels
            # it (no movement, no grid change), repeat. Similarly the late
            # ``[Block Chute] [PlayerSmacking] -> ... AssertGravityToken``
            # rule keeps re-adding the token after the main pass removes it,
            # leaving the net iteration a no-op.
            self._again_triggered = False
            if any_main_again or any_late_again:
                for r in range(self.height):
                    if self._again_triggered:
                        break
                    row_before = iter_start_grid[r]
                    row_now = self.grid[r]
                    for c in range(self.width):
                        if row_now[c] != row_before[c]:
                            self._again_triggered = True
                            break

            if self._late_cancel or not self._again_triggered:
                break

            # Stop the again loop the moment the win condition is satisfied,
            # so games whose animation keeps applying forces (e.g. botsket_ball
            # pushing the ball with bots while the ``[playl]->again`` cycle
            # runs) don't shove the ball *past* the net before we notice.
            if self.check_win():
                break

            forces = {}
            self._force_rigid_group = {}
            # PuzzleScript only applies the action input on the first call;
            # the `again` continuation re-runs rules with no input (dir=-1
            # in the reference engine). So the action force is consumed
            # after iteration 0 and not reasserted. Without this, Dharma
            # Dojo's ``[Block | Action PlayerSmacking]`` keeps re-firing on
            # every iteration and conflicts with the gravity DOWN force on
            # the chute block (both forces on the same cell cancel each
            # other and the block never falls).
            action_objects.clear()

        # Late cancel reverts the entire turn
        if self._late_cancel:
            self.grid = old_grid
            self._position_index_dirty = True
            self._rule_restart = False
            self._rule_win = False
            return False

        # Check require_player_movement
        if self.game.require_player_movement:
            new_player_positions = self._get_player_positions()
            if new_player_positions == old_player_positions:
                # Revert
                self.grid = old_grid
                self._position_index_dirty = True
                self._rule_restart = False
                self._rule_win = False
                return False

        return True

    def _get_player_positions(self) -> set[tuple[int, int]]:
        """Get current positions of all player objects."""
        positions = set()
        for r in range(self.height):
            for c in range(self.width):
                for pi in self._player_indices:
                    if pi in self.grid[r][c]:
                        positions.add((r, c))
        return positions

    def _apply_rules_with_forces(self, direction: str,
                                  forces: dict[tuple[int, int, int], str],
                                  late: bool,
                                  is_action_turn: bool = False,
                                  action_objects: set = None) -> tuple[bool, bool]:
        """Apply rules, using and modifying the forces dict.

        A ">" in the LHS matches objects that have a pending force.
        A ">" in the RHS assigns a force to the matched object.
        An empty RHS cell removes the object.

        Returns (cancelled, any_again) tuple.
        ``is_action_turn`` is True only when the player pressed action AND we
        are on the first turn (not an ``again`` continuation).  Rules that
        require the ``action`` keyword are skipped when this is False.
        ``action_objects`` tracks per-object action state; consumed when a rule
        matches [Action X] on LHS but RHS has [X] without Action.
        """
        any_again = False
        any_cancel = False
        if action_objects is None:
            action_objects = set()

        # The rule-result cache is keyed on ``_mutation_counter`` only,
        # which is valid because safely-cacheable rules touch neither the
        # forces dict nor the action set without bumping the mutation
        # counter. Outside of this method, however, ``_resolve_forces``
        # and ``step()`` mutate forces (and action_objects starts/ends a
        # turn re-populated by ``step``) without touching the mutation
        # counter — so cache entries from a prior pass would silently
        # become stale. Clearing at every pass start sidesteps that
        # entirely.
        self._rule_noop_cache.clear()

        # Collect eligible rules (respecting late/action filters)
        eligible = []
        for rule in self.game.rules:
            if rule.is_late != late:
                continue
            if rule.requires_action and not is_action_turn:
                continue
            eligible.append(rule)

        # Build sequential segments: each segment is either a single rule,
        # a '+' group, or a startloop block (which itself contains sub-segments).
        segments = self._build_rule_segments(eligible)

        for seg in segments:
            c, a = self._execute_segment(seg, direction, forces, action_objects)
            if c:
                any_cancel = True
            if a:
                any_again = True

        return (any_cancel, any_again)

    def _build_rule_segments(self, rules: list) -> list:
        """Organize rules into segments for execution.

        Returns a list of segments. Each segment is one of:
        - ('single', rule)                  — standalone rule
        - ('plus', [rule, ...])             — '+'-grouped rules (loop until stable)
        - ('loop', [segment, ...])          — startloop/endloop block (repeat all until stable)
        """
        segments = []
        i = 0
        while i < len(rules):
            rule = rules[i]

            if rule.loop_id >= 0:
                lid = rule.loop_id
                loop_end = i + 1
                while loop_end < len(rules) and rules[loop_end].loop_id == lid:
                    loop_end += 1
                loop_rules = rules[i:loop_end]
                sub_segments = self._build_plus_segments(loop_rules)
                segments.append(('loop', sub_segments))
                i = loop_end
            else:
                group_end = i + 1
                while group_end < len(rules) and rules[group_end].group_id == rule.group_id:
                    group_end += 1
                if group_end - i > 1:
                    segments.append(('plus', rules[i:group_end]))
                else:
                    segments.append(('single', rule))
                i = group_end

        return segments

    def _build_plus_segments(self, rules: list) -> list:
        """Build sub-segments within a startloop block, respecting '+' groups."""
        segments = []
        i = 0
        while i < len(rules):
            rule = rules[i]
            group_end = i + 1
            while group_end < len(rules) and rules[group_end].group_id == rule.group_id:
                group_end += 1
            if group_end - i > 1:
                segments.append(('plus', rules[i:group_end]))
            else:
                segments.append(('single', rule))
            i = group_end
        return segments

    def _execute_segment(self, seg, direction, forces, action_objects):
        """Execute a rule segment. Returns (any_cancel, any_again)."""
        any_cancel = False
        any_again = False

        kind = seg[0]
        if kind == 'single':
            rule = seg[1]
            c, a = self._execute_rule(rule, direction, forces, action_objects)
            return c, a

        elif kind == 'plus':
            group_rules = seg[1]
            for _ in range(200):
                any_fired = False
                for grule in group_rules:
                    c, a, fired = self._execute_rule_check_fired(
                        grule, direction, forces, action_objects)
                    if c:
                        any_cancel = True
                    if a:
                        any_again = True
                    if fired:
                        any_fired = True
                if not any_fired:
                    break
            return any_cancel, any_again

        elif kind == 'loop':
            sub_segments = seg[1]
            for _ in range(200):
                mut_at_start = self._mutation_counter
                old_forces = dict(forces)
                for sub_seg in sub_segments:
                    c, a = self._execute_segment(sub_seg, direction, forces, action_objects)
                    if c:
                        any_cancel = True
                    if a:
                        any_again = True
                grid_changed = self._mutation_counter != mut_at_start
                forces_changed = (forces != old_forces)
                if not grid_changed and not forces_changed:
                    break
            return any_cancel, any_again

        return any_cancel, any_again

    def _execute_rule(self, rule, direction, forces, action_objects):
        """Execute a single rule. Returns (any_cancel, any_again).

        Transformation rules iterate over the direction list until the grid
        and forces stabilise — matches PuzzleScript's per-rule iteration
        semantics, needed for propagation rules (e.g. wallHit infection
        through an L-shaped sticky cluster) where one direction's match
        creates a new match for a subsequent direction.
        """
        any_cancel = False
        any_again = False

        if (rule.cancel or rule.win or rule.restart) and not rule.groups_rhs:
            for dir_name in rule.directions:
                if rule.cancel:
                    if self._check_cancel_match(rule, dir_name, forces,
                                                input_dir=direction,
                                                action_objects=action_objects):
                        any_cancel = True
                elif rule.win:
                    if self._check_cancel_match(rule, dir_name, forces,
                                                input_dir=direction,
                                                action_objects=action_objects):
                        self._rule_win = True
                elif rule.restart:
                    if self._check_cancel_match(rule, dir_name, forces,
                                                input_dir=direction,
                                                action_objects=action_objects):
                        self._rule_restart = True
            return any_cancel, any_again

        # Cache fast path: a previous call that left the grid unchanged is
        # provably a no-op replay for safely-cacheable rules (see
        # `_compute_is_safely_cacheable`).
        if rule._is_safely_cacheable:
            cached = self._rule_noop_cache.get(id(rule))
            if cached is not None and cached[0] == self._mutation_counter:
                if cached[1] and rule.again:
                    any_again = True
                return any_cancel, any_again

        mut_at_call = self._mutation_counter

        max_iters = 1 if rule.random else 200
        rule_fired = False
        for _ in range(max_iters):
            mut_at_start = self._mutation_counter
            old_forces = dict(forces)
            iter_had_matches = False
            for dir_name in rule.directions:
                had_matches = self._apply_single_rule_forces(
                    rule, dir_name, forces, input_dir=direction,
                    action_objects=action_objects)
                if had_matches:
                    iter_had_matches = True
                    rule_fired = True
                    if rule.again:
                        any_again = True
                    if rule.random:
                        break
            grid_changed = self._mutation_counter != mut_at_start
            forces_changed = (forces != old_forces)
            if not (grid_changed or forces_changed):
                break
            if not iter_had_matches:
                break

        # Apply trailing command flags on transformation rules that fired.
        if rule_fired:
            if rule.cancel:
                any_cancel = True
            if rule.win:
                self._rule_win = True
            if rule.restart:
                self._rule_restart = True

        # Cache iff the entire call left the grid untouched (which, for
        # safely-cacheable rules, also implies no forces/action changes).
        if rule._is_safely_cacheable and self._mutation_counter == mut_at_call:
            self._rule_noop_cache[id(rule)] = (mut_at_call, rule_fired)

        return any_cancel, any_again

    def _execute_rule_check_fired(self, rule, direction, forces, action_objects):
        """Execute a single rule, also tracking whether it fired. Returns (cancel, again, fired)."""
        any_cancel = False
        any_again = False
        any_fired = False

        if (rule.cancel or rule.win or rule.restart) and not rule.groups_rhs:
            for dir_name in rule.directions:
                if rule.cancel:
                    if self._check_cancel_match(rule, dir_name, forces,
                                                input_dir=direction,
                                                action_objects=action_objects):
                        any_cancel = True
                elif rule.win:
                    if self._check_cancel_match(rule, dir_name, forces,
                                                input_dir=direction,
                                                action_objects=action_objects):
                        self._rule_win = True
                elif rule.restart:
                    if self._check_cancel_match(rule, dir_name, forces,
                                                input_dir=direction,
                                                action_objects=action_objects):
                        self._rule_restart = True
            return any_cancel, any_again, any_fired

        # Cache fast path: replay a previous no-op call. The cache is
        # populated only for safely-cacheable rules and only when the call
        # ended with ``_mutation_counter`` unchanged from its start —
        # which provably implies the call was a complete no-op (no grid
        # mutation, no forces.pop, no action_objects.add/discard). This is
        # the main + group win: e.g. CrateBlob's Checker rules keep
        # LHS-matching at steady state, so every iteration past the first
        # is a cache hit.
        if rule._is_safely_cacheable:
            cached = self._rule_noop_cache.get(id(rule))
            if cached is not None and cached[0] == self._mutation_counter:
                cached_fired = cached[1]
                if cached_fired:
                    any_fired = True
                    if rule.again:
                        any_again = True
                return any_cancel, any_again, any_fired

        mut_at_call = self._mutation_counter

        max_iters = 1 if rule.random else 200
        for _ in range(max_iters):
            mut_at_start = self._mutation_counter
            old_forces = dict(forces)
            iter_had_matches = False
            for dir_name in rule.directions:
                had_matches = self._apply_single_rule_forces(
                    rule, dir_name, forces, input_dir=direction,
                    action_objects=action_objects)
                if had_matches:
                    iter_had_matches = True
                    any_fired = True
                    if rule.again:
                        any_again = True
                    if rule.random:
                        break
            grid_changed = self._mutation_counter != mut_at_start
            forces_changed = (forces != old_forces)
            if not (grid_changed or forces_changed):
                break
            if not iter_had_matches:
                break

        # Apply trailing command flags on transformation rules that fired.
        if any_fired:
            if rule.cancel:
                any_cancel = True
            if rule.win:
                self._rule_win = True
            if rule.restart:
                self._rule_restart = True

        # Cache iff the entire call left the grid untouched (which, for
        # safely-cacheable rules, also implies no forces/action changes).
        if rule._is_safely_cacheable and self._mutation_counter == mut_at_call:
            self._rule_noop_cache[id(rule)] = (mut_at_call, any_fired)

        return any_cancel, any_again, any_fired

    def _check_cancel_match(self, rule: PSRule, direction: str,
                             forces: dict, input_dir: str = None,
                             action_objects: set = None) -> bool:
        """Check if a cancel rule's LHS pattern matches anywhere."""
        dr, dc = _DIR_DELTAS.get(direction, (0, 0))
        dir_symbol_map = self._get_dir_symbol_map(direction)

        # Multi-group: all groups must match independently
        for group_lhs in rule.groups_lhs:
            found = False
            for r in range(self.height):
                if found:
                    break
                for c in range(self.width):
                    match = self._check_rule_match_forces(
                        rule, r, c, dr, dc, dir_symbol_map, forces,
                        input_dir=input_dir, lhs_patterns=group_lhs,
                        action_objects=action_objects)
                    if match is not None:
                        found = True
                        break
            if not found:
                return False
        return True

    def _tag_rigid_forces(self, positions, forces, group_id: int):
        """Tag all forces at the given match positions with a rigid group id.

        Used to implement RIGID rule semantics: if any tagged force is
        blocked during resolution, every other force in the same group is
        cancelled along with it.
        """
        if not positions:
            return
        pos_set = {(r, c) for (r, c) in positions}
        for key in forces:
            if (key[0], key[1]) in pos_set:
                self._force_rigid_group[key] = group_id

    def _apply_single_rule_forces(self, rule: PSRule, direction: str,
                                   forces: dict[tuple[int, int, int], str],
                                   input_dir: str = None,
                                   action_objects: set = None) -> bool:
        """Apply a single rule in a specific direction, managing forces.

        Returns True if any matches were found and applied.
        """
        dr, dc = _DIR_DELTAS.get(direction, (0, 0))
        dir_symbol_map = self._get_dir_symbol_map(direction)

        if len(rule.groups_lhs) > 1:
            return self._apply_multi_group_rule_forces(
                rule, direction, forces, input_dir,
                action_objects=action_objects)

        had_match = False
        r_range = range(self.height - 1, -1, -1) if direction == "up" else range(self.height)
        c_range = range(self.width - 1, -1, -1) if direction == "left" else range(self.width)

        first_required = (rule._first_cell_required_per_group[0]
                          if rule._first_cell_required_per_group else None)

        def _iter_positions():
            cands = self._candidate_positions(first_required)
            if cands is None:
                for r in r_range:
                    for c in c_range:
                        yield r, c
            else:
                # Sort candidates to preserve scan-order-dependent semantics.
                if direction == "up":
                    sorted_cands = sorted(cands, key=lambda p: (-p[0], p[1]))
                elif direction == "left":
                    sorted_cands = sorted(cands, key=lambda p: (p[0], -p[1]))
                else:
                    sorted_cands = sorted(cands)
                for pos in sorted_cands:
                    yield pos

        if rule.random:
            matches = []
            for (r, c) in _iter_positions():
                for m in self._iter_rule_matches_forces(
                        rule, r, c, dr, dc, dir_symbol_map, forces,
                        input_dir=input_dir,
                        action_objects=action_objects):
                    matches.append(m)
            if not matches:
                return False
            chosen = _random_module.choice(matches)
            self._apply_rule_match_forces(
                rule, chosen, dr, dc, dir_symbol_map, forces,
                action_objects=action_objects)
            if rule.rigid:
                gid = self._rigid_group_counter
                self._rigid_group_counter += 1
                self._tag_rigid_forces(chosen, forces, gid)
            return True

        for (r, c) in _iter_positions():
            for mi, match in enumerate(list(self._iter_rule_matches_forces(
                    rule, r, c, dr, dc, dir_symbol_map, forces,
                    input_dir=input_dir,
                    action_objects=action_objects))):
                if mi and not self._match_still_valid(
                        rule.patterns_lhs, match, dir_symbol_map, forces,
                        input_dir=input_dir, action_objects=action_objects):
                    continue
                self._apply_rule_match_forces(rule, match, dr, dc, dir_symbol_map, forces,
                                              action_objects=action_objects)
                if rule.rigid:
                    gid = self._rigid_group_counter
                    self._rigid_group_counter += 1
                    self._tag_rigid_forces(match, forces, gid)
                had_match = True

        if had_match:
            r_range_rev = range(self.height) if direction == "up" else range(self.height - 1, -1, -1)
            c_range_rev = range(self.width) if direction == "left" else range(self.width - 1, -1, -1)

            def _iter_positions_rev():
                cands = self._candidate_positions(first_required)
                if cands is None:
                    for r in r_range_rev:
                        for c in c_range_rev:
                            yield r, c
                else:
                    if direction == "up":
                        sorted_cands = sorted(cands, key=lambda p: (p[0], p[1]))
                    elif direction == "left":
                        sorted_cands = sorted(cands, key=lambda p: (p[0], p[1]))
                    else:
                        sorted_cands = sorted(cands, reverse=True)
                    for pos in sorted_cands:
                        yield pos

            for (r, c) in _iter_positions_rev():
                for mi, match in enumerate(list(self._iter_rule_matches_forces(
                        rule, r, c, dr, dc, dir_symbol_map, forces,
                        input_dir=input_dir,
                        action_objects=action_objects))):
                    if mi and not self._match_still_valid(
                            rule.patterns_lhs, match, dir_symbol_map, forces,
                            input_dir=input_dir, action_objects=action_objects):
                        continue
                    self._apply_rule_match_forces(rule, match, dr, dc, dir_symbol_map, forces,
                                                  action_objects=action_objects)
                    if rule.rigid:
                        gid = self._rigid_group_counter
                        self._rigid_group_counter += 1
                        self._tag_rigid_forces(match, forces, gid)

        return had_match

    def _apply_multi_group_rule_forces(self, rule: PSRule, direction: str,
                                        forces: dict, input_dir: str = None,
                                        action_objects: set = None) -> bool:
        """Apply a multi-bracket-group rule.  Each group is matched
        independently against the entire grid.  All groups must match
        for the rule to fire.
        """
        dr, dc = _DIR_DELTAS.get(direction, (0, 0))
        dir_symbol_map = self._get_dir_symbol_map(direction)

        # Collect all matches per group; bail if any group has none
        all_group_matches: list[list] = []
        for gi, group_lhs in enumerate(rule.groups_lhs):
            matches = []
            first_required = (rule._first_cell_required_per_group[gi]
                              if gi < len(rule._first_cell_required_per_group) else None)
            candidates = self._candidate_positions(first_required)
            if candidates is None:
                for r in range(self.height):
                    for c in range(self.width):
                        m = self._check_rule_match_forces(
                            rule, r, c, dr, dc, dir_symbol_map, forces,
                            input_dir=input_dir, lhs_patterns=group_lhs,
                            action_objects=action_objects)
                        if m is not None:
                            matches.append(m)
            else:
                for (r, c) in candidates:
                    m = self._check_rule_match_forces(
                        rule, r, c, dr, dc, dir_symbol_map, forces,
                        input_dir=input_dir, lhs_patterns=group_lhs,
                        action_objects=action_objects)
                    if m is not None:
                        matches.append(m)
            if not matches:
                return False
            all_group_matches.append(matches)

        # Overlap preference: when a later group has multiple matches and
        # some share cells with earlier groups' matches, restrict to those
        # overlapping ones. This handles patterns like [Bdoor][target] where
        # the target is colocated with the Bdoor (legend: D = Bdoor and target).
        # Skipped when the group references an or-group (e.g. Ball = BallA or
        # BallI), because such patterns are meant to fire for every match —
        # restricting to the overlap with a prior group's cell drops the
        # other matches (e.g. Ball Bros' [action Player][Ball]).
        chosen_cells: set[tuple[int, int]] = set()
        for m in all_group_matches[0]:
            chosen_cells.update(m)
        for gi in range(1, len(all_group_matches)):
            group_lhs = rule.groups_lhs[gi]
            uses_or_group = any(
                obj_name in self.game.or_groups
                for cell_pattern in group_lhs
                for modifier, obj_name, _idxs in cell_pattern
                if obj_name and modifier not in ("no", "...")
            )
            if not uses_or_group and len(all_group_matches[gi]) > 1:
                overlapping = [m for m in all_group_matches[gi]
                               if any(pos in chosen_cells for pos in m)]
                if overlapping:
                    all_group_matches[gi] = overlapping
            for m in all_group_matches[gi]:
                chosen_cells.update(m)

        # For random rules, pick exactly one match per group so the rule
        # fires once total (not once per non-anchored group match).
        if rule.random:
            for gi in range(len(all_group_matches)):
                if len(all_group_matches[gi]) > 1:
                    all_group_matches[gi] = [_random_module.choice(all_group_matches[gi])]

        # Collect all LHS force references across ALL groups so that
        # 'moving' in one group's RHS can inherit forces from another group.
        all_lhs_force_refs: list[tuple[int, int, int]] = []
        for gi, matches in enumerate(all_group_matches):
            g_lhs = rule.groups_lhs[gi]
            for match in matches:
                for i, (r, c) in enumerate(match):
                    if i < len(g_lhs):
                        for modifier, obj_name, idxs in g_lhs[i]:
                            if modifier in ("moving", ">", "<", "^", "v",
                                            "right", "left", "up", "down",
                                            "perpendicular", "parallel",
                                            "horizontal", "vertical"):
                                cell = self.grid[r][c]
                                for idx in idxs:
                                    if idx in cell:
                                        all_lhs_force_refs.append((r, c, idx))

        # Build cross-group or-bindings from ALL groups' LHS matches
        cross_group_or_bindings: dict[str, int] = {}
        for gi, matches in enumerate(all_group_matches):
            g_lhs = rule.groups_lhs[gi]
            for match in matches:
                for i, (r, c) in enumerate(match):
                    if i < len(g_lhs):
                        for modifier, obj_name, idxs in g_lhs[i]:
                            if modifier in ("no", "...") or not obj_name:
                                continue
                            if obj_name in cross_group_or_bindings:
                                continue
                            if obj_name in self.game.or_groups:
                                cell = self.grid[r][c]
                                for idx in idxs:
                                    if idx in cell:
                                        cross_group_or_bindings[obj_name] = idx
                                        break

        # Apply each combination in the Cartesian product of per-group
        # matches, re-validating each tuple against the (now-mutated) grid
        # so consumed LHS state (e.g. ``ACTION Player`` in Collect Gnocchi)
        # blocks stale tuples from firing. Matches PuzzleScript's reference
        # engine, which iterates over generateTuples(matches) with per-tuple
        # re-check. Without this, rules like Dharma Dojo's
        # ``[SetsToken1][Hopper]`` seed only the first hopper, and gravity
        # rules like ``[AssertGravityToken][Block | no Wall no Hopper]`` give
        # the force to whichever block scans first instead of every eligible
        # block. For RIGID rules, all groups in a tuple share a single rigid
        # group id so they cancel together.
        any_applied = False
        for tuple_idx, match_tuple in enumerate(_itertools.product(*all_group_matches)):
            if tuple_idx > 0:
                # Re-validate this tuple against the current grid.
                valid = True
                for gi, match in enumerate(match_tuple):
                    if not match:
                        valid = False
                        break
                    r0, c0 = match[0]
                    new_m = self._check_rule_match_forces(
                        rule, r0, c0, dr, dc, dir_symbol_map, forces,
                        input_dir=input_dir,
                        lhs_patterns=rule.groups_lhs[gi],
                        action_objects=action_objects)
                    if new_m is None or list(new_m) != list(match):
                        valid = False
                        break
                if not valid:
                    continue
            rigid_gid = None
            if rule.rigid:
                rigid_gid = self._rigid_group_counter
                self._rigid_group_counter += 1
            for gi, match in enumerate(match_tuple):
                g_lhs = rule.groups_lhs[gi]
                g_rhs = rule.groups_rhs[gi] if gi < len(rule.groups_rhs) else []
                self._apply_rule_match_forces(
                    rule, match, dr, dc, dir_symbol_map, forces,
                    lhs_patterns=g_lhs, rhs_patterns=g_rhs,
                    cross_group_force_refs=all_lhs_force_refs,
                    action_objects=action_objects,
                    cross_group_or_bindings=cross_group_or_bindings)
                if rigid_gid is not None:
                    self._tag_rigid_forces(match, forces, rigid_gid)
            any_applied = True
        return any_applied

    def _match_still_valid(self, lhs_patterns, match, dir_map: dict,
                           forces: dict, input_dir: str = None,
                           action_objects: set = None) -> bool:
        """Does ``match`` STILL hold against the (already mutated) grid?

        `_iter_rule_matches_forces` enumerates every ellipsis binding from one
        anchor in a single pass over the grid as it was BEFORE any of them was
        applied. Applying the first one can destroy the very cells the others
        matched on, and PuzzleScript's reference engine re-checks each tuple
        before firing it -- which is what the multi-bracket path here already
        does (see `_apply_multi_group_rule_forces`).

        Without this check a teleport rule CLONES its subject: Bridge-toggle
        Maze's ``[ > player | ... | OkNS ] -> [ prevPos | ... | player OkNS ]``
        means "slide to the nearest cell you can stand on", and every further
        landable cell down that column is also a valid binding, so one keypress
        left a copy of the player on each of them and its ``All Player on
        StartPos`` win became unreachable.

        Propagation rules -- the reason all the bindings are enumerated in the
        first place, e.g. ``[left coin|...|moveable] -> [left coin|...|left
        moveable]`` -- are untouched: their RHS does not consume the LHS, so
        every later binding re-validates and still fires.

        Ellipsis cells are skipped: their recorded position is an internal
        placeholder marking where the ellipsis began, not a matched cell.
        """
        meta_map = self.game._pattern_meta
        for i, cell_pattern in enumerate(lhs_patterns):
            if i >= len(match):
                return False
            cell_meta = meta_map.get(id(cell_pattern))
            has_ellipsis = (cell_meta["has_ellipsis"] if cell_meta
                            else any(item[0] == "..." for item in cell_pattern))
            if has_ellipsis:
                continue
            mr, mc = match[i]
            if not (0 <= mr < self.height and 0 <= mc < self.width):
                return False
            if not self._cell_matches_forces(
                    cell_pattern, mr, mc, dir_map, forces,
                    input_dir=input_dir, action_objects=action_objects):
                return False
        return True

    def _iter_rule_matches_forces(self, rule: PSRule, r: int, c: int,
                                   dr: int, dc: int, dir_map: dict,
                                   forces: dict, input_dir: str = None,
                                   lhs_patterns: list = None,
                                   action_objects: set = None):
        """Yield every valid LHS match starting at (r, c).

        Standard PuzzleScript fires the rule once per valid bound of the
        ellipsis target. Without this, propagation rules like
        ``[left coin|...|moveable]`` would only set a force on the first
        moveable scanned, leaving the rest of the row unmoved — so a coin
        cannot push a wall of blocks even when the wrap rules would
        otherwise complete the rotation.
        """
        lhs = lhs_patterns if lhs_patterns is not None else rule.patterns_lhs
        meta_map = self.game._pattern_meta

        def _match(i: int, cr: int, cc: int, positions: list[tuple[int, int]]):
            while i < len(lhs):
                cell_pattern = lhs[i]
                cell_meta = meta_map.get(id(cell_pattern))
                has_ellipsis = (cell_meta["has_ellipsis"] if cell_meta
                                else any(item[0] == "..." for item in cell_pattern))

                if has_ellipsis:
                    next_i = i + 1
                    if next_i >= len(lhs):
                        # Ellipsis at end — record dummy position and finish
                        yield positions + [(cr, cc)]
                        return

                    next_pattern = lhs[next_i]
                    search_r, search_c = cr, cc
                    while 0 <= search_r < self.height and 0 <= search_c < self.width:
                        if self._cell_matches_forces(next_pattern, search_r, search_c,
                                                      dir_map, forces,
                                                      input_dir=input_dir,
                                                      action_objects=action_objects):
                            yield from _match(next_i, search_r, search_c,
                                              positions + [(cr, cc)])
                        search_r += dr
                        search_c += dc
                    return

                # Normal cell matching
                if cr < 0 or cr >= self.height or cc < 0 or cc >= self.width:
                    return

                if not self._cell_matches_forces(cell_pattern, cr, cc, dir_map, forces,
                                                  input_dir=input_dir,
                                                  action_objects=action_objects):
                    return

                positions = positions + [(cr, cc)]
                cr += dr
                cc += dc
                i += 1

            yield positions

        yield from _match(0, r, c, [])

    def _check_rule_match_forces(self, rule: PSRule, r: int, c: int,
                                  dr: int, dc: int, dir_map: dict,
                                  forces: dict, input_dir: str = None,
                                  lhs_patterns: list = None,
                                  action_objects: set = None) -> Optional[list[tuple[int, int]]]:
        """Return the first valid LHS match starting at (r, c), or None.

        Backward-compatible single-match wrapper around
        :meth:`_iter_rule_matches_forces`. Used by cancel/win/restart command
        checks and the multi-bracket matcher, which only need to know whether
        a match exists, not enumerate all of them.
        """
        for m in self._iter_rule_matches_forces(
                rule, r, c, dr, dc, dir_map, forces,
                input_dir=input_dir, lhs_patterns=lhs_patterns,
                action_objects=action_objects):
            return m
        return None

    def _cell_matches_forces(self, cell_pattern,
                              r: int, c: int, dir_map: dict,
                              forces: dict, input_dir: str = None,
                              action_objects: set = None) -> bool:
        """Check if a cell pattern matches, considering forces for directional modifiers."""
        cell = self.grid[r][c]
        for modifier, obj_name, obj_indices in cell_pattern:
            if modifier == "...":
                continue

            if modifier == "no":
                for idx in obj_indices:
                    if idx in cell:
                        return False
            elif modifier == "action":
                # The "action" force can be set on any object by rules (RHS
                # `action X`), not only on the player when the action key is
                # pressed.  Match if the object is present AND its (r, c, idx)
                # is in action_objects.
                found = False
                for idx in obj_indices:
                    if idx in cell:
                        if action_objects is not None and (r, c, idx) in action_objects:
                            found = True
                            break
                if not found:
                    return False
            elif modifier in (">", "<", "^", "v"):
                # Object must be present AND have a force in the specified direction
                actual_dir = dir_map.get(modifier, "right")
                found = False
                for idx in obj_indices:
                    if idx in cell:
                        obj_force = forces.get((r, c, idx))
                        if obj_force == actual_dir:
                            found = True
                            break
                if not found:
                    return False
            elif modifier in ("right", "left", "up", "down"):
                found = False
                for idx in obj_indices:
                    if idx in cell:
                        obj_force = forces.get((r, c, idx))
                        if obj_force == modifier:
                            found = True
                            break
                if not found:
                    return False
            elif modifier == "perpendicular":
                forward_dir = dir_map.get(">", "right")
                if forward_dir in ("left", "right"):
                    perp_dirs = ("up", "down")
                else:
                    perp_dirs = ("left", "right")
                found = False
                for idx in obj_indices:
                    if idx in cell:
                        obj_force = forces.get((r, c, idx))
                        if obj_force in perp_dirs:
                            found = True
                            break
                if not found:
                    return False
            elif modifier == "parallel":
                forward_dir = dir_map.get(">", "right")
                backward_dir = dir_map.get("<", "left")
                par_dirs = (forward_dir, backward_dir)
                found = False
                for idx in obj_indices:
                    if idx in cell:
                        obj_force = forces.get((r, c, idx))
                        if obj_force in par_dirs:
                            found = True
                            break
                if not found:
                    return False
            elif modifier == "stationary":
                # Object must be present and have NO pending force
                found = False
                for idx in obj_indices:
                    if idx in cell:
                        if (r, c, idx) not in forces:
                            found = True
                            break
                if not found:
                    return False
            elif modifier == "moving":
                # Object must be present and HAVE a pending force
                found = False
                for idx in obj_indices:
                    if idx in cell:
                        if (r, c, idx) in forces:
                            found = True
                            break
                if not found:
                    return False
            elif modifier in ("horizontal", "vertical"):
                # Object must be present and have force on the matching axis
                axis_dirs = ("left", "right") if modifier == "horizontal" else ("up", "down")
                found = False
                for idx in obj_indices:
                    if idx in cell:
                        obj_force = forces.get((r, c, idx))
                        if obj_force in axis_dirs:
                            found = True
                            break
                if not found:
                    return False
            else:
                # Plain object reference — must be present
                if obj_name and obj_indices:
                    found = False
                    for idx in obj_indices:
                        if idx in cell:
                            found = True
                            break
                    if not found:
                        return False

        return True

    def _apply_rule_match_forces(self, rule: PSRule, positions: list[tuple[int, int]],
                                  dr: int, dc: int, dir_map: dict,
                                  forces: dict, lhs_patterns: list = None,
                                  rhs_patterns: list = None,
                                  cross_group_force_refs: list = None,
                                  action_objects: set = None,
                                  cross_group_or_bindings: dict = None):
        """Apply the RHS of a matched rule, updating forces.

        Handles two movement models:
        1. Force propagation: RHS uses ">" to assign/keep forces for later resolution
        2. Direct placement: RHS places objects in different cells than LHS (teleport)
           In this case, the object's force is consumed.
        """
        eff_lhs = lhs_patterns if lhs_patterns is not None else rule.patterns_lhs
        eff_rhs = rhs_patterns if rhs_patterns is not None else rule.patterns_rhs

        # Fast path for trivial rules: per-cell object replacement only.
        # Avoids ~10 passes over positions and all the force/or-group
        # bookkeeping the generic path needs.
        if rule._is_trivial:
            # Pick the correct precomputed index sets — multi-bracket rules
            # call this once per group with the resolved per-group pattern.
            lhs_idx_per_cell = None
            rhs_idx_per_cell = None
            if lhs_patterns is None and rhs_patterns is None:
                lhs_idx_per_cell = rule._trivial_lhs_per_group[0] if rule._trivial_lhs_per_group else []
                rhs_idx_per_cell = rule._trivial_rhs_per_group[0] if rule._trivial_rhs_per_group else []
            else:
                # Multi-group: locate the matching group by identity.
                for gi, group in enumerate(rule.groups_lhs):
                    if group is lhs_patterns:
                        lhs_idx_per_cell = rule._trivial_lhs_per_group[gi]
                        rhs_idx_per_cell = (rule._trivial_rhs_per_group[gi]
                                            if gi < len(rule._trivial_rhs_per_group) else [])
                        break
            if lhs_idx_per_cell is not None and rhs_idx_per_cell is not None:
                n_lhs = len(lhs_idx_per_cell)
                n_rhs = len(rhs_idx_per_cell)
                obj_layers = self._obj_layers
                for i, (r, c) in enumerate(positions):
                    if i >= n_rhs:
                        # No RHS for this cell: if LHS had content, remove it
                        if i < n_lhs:
                            for idx in lhs_idx_per_cell[i]:
                                self._cell_discard(r, c, idx)
                                forces.pop((r, c, idx), None)
                        continue
                    cell_lhs = lhs_idx_per_cell[i] if i < n_lhs else frozenset()
                    cell_rhs = rhs_idx_per_cell[i]
                    # Remove LHS objects not kept in RHS
                    for idx in cell_lhs:
                        if idx not in cell_rhs:
                            self._cell_discard(r, c, idx)
                            forces.pop((r, c, idx), None)
                    # Add RHS objects not already present (with same-layer eviction)
                    cell = self.grid[r][c]
                    for idx in cell_rhs:
                        if idx in cell:
                            continue
                        obj_layer = obj_layers.get(idx, -1)
                        if obj_layer >= 0:
                            for x in list(cell):
                                if x != idx and obj_layers.get(x, -1) == obj_layer:
                                    self._cell_discard(r, c, x)
                                    forces.pop((r, c, x), None)
                        self._cell_add(r, c, idx)
                return

        # Build or-group binding only when the rule actually references
        # or-groups; otherwise the dict is empty and every later lookup
        # against it would always miss.
        lhs_group_member: dict[str, int] = {}
        if rule._lhs_has_or_group or rule._rhs_has_or_group or cross_group_or_bindings:
            for i, (r, c) in enumerate(positions):
                if i < len(eff_lhs):
                    cell = self.grid[r][c]
                    for modifier, obj_name, obj_indices in eff_lhs[i]:
                        if modifier in ("no", "...") or not obj_name:
                            continue
                        if obj_name in lhs_group_member:
                            continue
                        if obj_name in self.game.or_groups:
                            if modifier in (">", "<", "^", "v"):
                                actual_dir = dir_map.get(modifier, "right")
                                for idx in obj_indices:
                                    if idx in cell and forces.get((r, c, idx)) == actual_dir:
                                        lhs_group_member[obj_name] = idx
                                        break
                            elif modifier in ("right", "left", "up", "down"):
                                for idx in obj_indices:
                                    if idx in cell and forces.get((r, c, idx)) == modifier:
                                        lhs_group_member[obj_name] = idx
                                        break
                            elif modifier == "moving":
                                for idx in obj_indices:
                                    if idx in cell and (r, c, idx) in forces:
                                        lhs_group_member[obj_name] = idx
                                        break
                            elif modifier == "stationary":
                                for idx in obj_indices:
                                    if idx in cell and (r, c, idx) not in forces:
                                        lhs_group_member[obj_name] = idx
                                        break
                            elif modifier == "action":
                                for idx in obj_indices:
                                    if idx in cell and (action_objects and (r, c, idx) in action_objects):
                                        lhs_group_member[obj_name] = idx
                                        break
                            if obj_name not in lhs_group_member:
                                for idx in obj_indices:
                                    if idx in cell:
                                        lhs_group_member[obj_name] = idx
                                        break

            # Merge cross-group or-bindings (from other groups' LHS matches)
            if cross_group_or_bindings:
                for name, idx in cross_group_or_bindings.items():
                    if name not in lhs_group_member:
                        lhs_group_member[name] = idx

        # First pass: determine what LHS objects exist in each cell.
        # For or-groups, only include the bound member so that other
        # members sharing the cell are not accidentally removed.
        lhs_objs_per_cell: list[set[int]] = []
        for i, (r, c) in enumerate(positions):
            lhs_objs = set()
            if i < len(eff_lhs):
                cell = self.grid[r][c]
                for modifier, obj_name, obj_indices in eff_lhs[i]:
                    if modifier != "no" and modifier != "..." and obj_name:
                        if obj_name in lhs_group_member:
                            bound_idx = lhs_group_member[obj_name]
                            if bound_idx in cell:
                                lhs_objs.add(bound_idx)
                            else:
                                for idx in obj_indices:
                                    if idx in cell:
                                        lhs_objs.add(idx)
                        else:
                            for idx in obj_indices:
                                if idx in cell:
                                    lhs_objs.add(idx)
            lhs_objs_per_cell.append(lhs_objs)

        # Second pass: determine what RHS wants in each cell
        rhs_objs_per_cell: list[dict] = []  # list of {obj_idx: modifier} per cell
        rhs_remove_per_cell: list[set] = []  # objects explicitly marked 'no' for removal
        rhs_has_content_per_cell: list[bool] = []

        for i, (r, c) in enumerate(positions):
            cell_rhs = {}
            remove_objs: set[int] = set()
            has_content = False
            if i < len(eff_rhs):
                cell = self.grid[r][c]
                for modifier, obj_name, obj_indices in eff_rhs[i]:
                    if modifier == "..." or not obj_name:
                        continue
                    has_content = True
                    if modifier == "no":
                        # RHS 'no' means explicitly remove this object
                        for idx in obj_indices:
                            remove_objs.add(idx)
                        continue
                    if modifier == "random":
                        # RHS 'random X' picks a random member of property X
                        # (or just X if not a property). Use the or_group
                        # members directly so we sample uniformly.
                        if obj_name in self.game.or_groups:
                            choices = [self.game.obj_name_to_idx[n]
                                       for n in self.game.or_groups[obj_name]
                                       if n in self.game.obj_name_to_idx]
                        else:
                            choices = list(obj_indices)
                        matched_idx = _random_module.choice(choices) if choices else None
                        if matched_idx is not None:
                            cell_rhs[matched_idx] = ""
                        continue
                    # For or-groups, prefer the member from this cell's LHS
                    # match so each cell keeps its own or-group member.
                    # Fall back to global binding for cells where the
                    # or-group wasn't present in the LHS (e.g. teleport).
                    matched_idx = None
                    if obj_name in self.game.or_groups and i < len(lhs_objs_per_cell):
                        for idx in obj_indices:
                            if idx in lhs_objs_per_cell[i]:
                                matched_idx = idx
                                break
                    if matched_idx is None:
                        if obj_name in lhs_group_member:
                            matched_idx = lhs_group_member[obj_name]
                        else:
                            for idx in obj_indices:
                                if idx in cell:
                                    matched_idx = idx
                                    break
                            if matched_idx is None:
                                matched_idx = obj_indices[0] if obj_indices else None
                    if matched_idx is not None:
                        cell_rhs[matched_idx] = modifier
            rhs_objs_per_cell.append(cell_rhs)
            rhs_remove_per_cell.append(remove_objs)
            rhs_has_content_per_cell.append(has_content)

        # Third pass: apply changes
        # Save forces from LHS objects before removal so they can be
        # transferred to replacement or-group members (e.g. avatarup → avatardown)
        saved_forces_per_cell: list[dict[int, str]] = []
        for i, (r, c) in enumerate(positions):
            cell_saved: dict[int, str] = {}
            for idx in lhs_objs_per_cell[i]:
                f = forces.get((r, c, idx))
                if f is not None:
                    cell_saved[idx] = f
            saved_forces_per_cell.append(cell_saved)

        # Clear forces for objects explicitly matched with movement modifiers
        # in the LHS.  The LHS "consumed" these forces; the RHS will assign
        # new ones.  Without this, assigning a different direction in the RHS
        # (e.g. [> Player] -> [< Player]) would conflict and cancel.
        if rule._lhs_has_movement_mod:
            for i, (r, c) in enumerate(positions):
                if i < len(eff_lhs):
                    for lhs_mod, lhs_name, lhs_idxs in eff_lhs[i]:
                        if lhs_mod in (">", "<", "^", "v", "moving",
                                       "right", "left", "up", "down",
                                       "perpendicular", "parallel",
                                       "horizontal", "vertical") and lhs_name:
                            for lhs_idx in lhs_idxs:
                                forces.pop((r, c, lhs_idx), None)

        # Track objects that were moved by the rule (placed in a different cell)
        objects_moved_by_rule: set[int] = set()

        # Determine objects that appear in RHS at a different position than LHS
        for i, (r, c) in enumerate(positions):
            for obj_idx, modifier in rhs_objs_per_cell[i].items():
                if obj_idx not in lhs_objs_per_cell[i]:
                    # Check if this is a same-layer swap (e.g. avatarup→avatardown)
                    obj_layer = self._obj_layers.get(obj_idx, -1)
                    is_swap = False
                    if obj_layer >= 0:
                        for old_idx in lhs_objs_per_cell[i]:
                            if self._obj_layers.get(old_idx, -1) == obj_layer:
                                is_swap = True
                                break
                    if not is_swap:
                        objects_moved_by_rule.add(obj_idx)

        # Apply: explicitly remove objects marked with 'no' in RHS
        for i, (r, c) in enumerate(positions):
            for idx in rhs_remove_per_cell[i]:
                self._cell_discard(r, c, idx)
                forces.pop((r, c, idx), None)

        # Apply: remove objects from cells where LHS had them but RHS doesn't
        for i, (r, c) in enumerate(positions):
            if not rhs_has_content_per_cell[i] and lhs_objs_per_cell[i]:
                # Empty RHS cell — remove all LHS objects
                for idx in lhs_objs_per_cell[i]:
                    self._cell_discard(r, c, idx)
                    forces.pop((r, c, idx), None)
            else:
                # Remove LHS objects not in RHS (excluding already-removed 'no' objects)
                for idx in lhs_objs_per_cell[i]:
                    if idx not in rhs_objs_per_cell[i] and idx not in rhs_remove_per_cell[i]:
                        self._cell_discard(r, c, idx)
                        forces.pop((r, c, idx), None)

        # Apply: add objects where RHS has them but they're not present
        for i, (r, c) in enumerate(positions):
            for obj_idx, modifier in rhs_objs_per_cell[i].items():
                if obj_idx not in self.grid[r][c]:
                    obj_layer = self._obj_layers.get(obj_idx, -1)
                    if obj_layer >= 0:
                        for x in list(self.grid[r][c]):
                            if x != obj_idx and self._obj_layers.get(x, -1) == obj_layer:
                                self._cell_discard(r, c, x)
                                forces.pop((r, c, x), None)
                    self._cell_add(r, c, obj_idx)

                # Handle force assignment / clearing
                if modifier in (">", "<", "^", "v"):
                    actual_dir = dir_map.get(modifier, "right")
                    existing = forces.get((r, c, obj_idx))
                    if existing is not None and existing != actual_dir:
                        # Conflicting forces cancel — object stays stationary
                        forces.pop((r, c, obj_idx), None)
                    else:
                        forces[(r, c, obj_idx)] = actual_dir
                elif modifier in ("right", "left", "up", "down"):
                    existing = forces.get((r, c, obj_idx))
                    if existing is not None and existing != modifier:
                        forces.pop((r, c, obj_idx), None)
                    else:
                        forces[(r, c, obj_idx)] = modifier
                elif modifier == "perpendicular":
                    forward_dir = dir_map.get(">", "right")
                    if forward_dir in ("left", "right"):
                        perp_dirs = ("up", "down")
                    else:
                        perp_dirs = ("left", "right")
                    matched_perp = None
                    for j, (pr, pc) in enumerate(positions):
                        if j < len(eff_lhs):
                            for lhs_mod, lhs_name, lhs_idxs in eff_lhs[j]:
                                if lhs_mod == "perpendicular" and lhs_name:
                                    for lhs_idx in lhs_idxs:
                                        # The LHS-movement-mod pass above already
                                        # popped this force, so read the copy taken
                                        # before it: unlike ">" / "up", the RHS
                                        # direction here IS the old force, and
                                        # losing it silently drops the force from
                                        # the RHS object. Collect Gnocchi's herby
                                        # rule then spawned a golem on one side of
                                        # the gnocchi instead of both, because the
                                        # second rule direction no longer matched.
                                        # (Same pre-pop read the `moving` and
                                        # horizontal/vertical branches below
                                        # already do.)
                                        f = forces.get((pr, pc, lhs_idx))
                                        if f is None and j < len(saved_forces_per_cell):
                                            f = saved_forces_per_cell[j].get(lhs_idx)
                                        if f in perp_dirs:
                                            matched_perp = f
                                            break
                                    if matched_perp:
                                        break
                        if matched_perp:
                            break
                    if matched_perp is None and cross_group_force_refs:
                        for ref_r, ref_c, ref_idx in cross_group_force_refs:
                            f = forces.get((ref_r, ref_c, ref_idx))
                            if f in perp_dirs:
                                matched_perp = f
                                break
                    if matched_perp:
                        existing = forces.get((r, c, obj_idx))
                        if existing is not None and existing != matched_perp:
                            forces.pop((r, c, obj_idx), None)
                        else:
                            forces[(r, c, obj_idx)] = matched_perp
                elif modifier == "parallel":
                    forward_dir = dir_map.get(">", "right")
                    backward_dir = dir_map.get("<", "left")
                    par_dirs = (forward_dir, backward_dir)
                    matched_par = None
                    for j, (pr, pc) in enumerate(positions):
                        if j < len(eff_lhs):
                            for lhs_mod, lhs_name, lhs_idxs in eff_lhs[j]:
                                if lhs_mod == "parallel" and lhs_name:
                                    for lhs_idx in lhs_idxs:
                                        # Same pre-pop read as `perpendicular`.
                                        # Note the fallback below assigns the
                                        # RULE direction, which for a parallel
                                        # match is a coin flip between the
                                        # object's real direction and its exact
                                        # opposite -- reaching it at all was the
                                        # bug, not the tie-break.
                                        f = forces.get((pr, pc, lhs_idx))
                                        if f is None and j < len(saved_forces_per_cell):
                                            f = saved_forces_per_cell[j].get(lhs_idx)
                                        if f in par_dirs:
                                            matched_par = f
                                            break
                                    if matched_par:
                                        break
                            if matched_par:
                                break
                    if matched_par is None and cross_group_force_refs:
                        for ref_r, ref_c, ref_idx in cross_group_force_refs:
                            f = forces.get((ref_r, ref_c, ref_idx))
                            if f in par_dirs:
                                matched_par = f
                                break
                    if matched_par is None:
                        matched_par = forward_dir
                    existing = forces.get((r, c, obj_idx))
                    if existing is not None and existing != matched_par:
                        forces.pop((r, c, obj_idx), None)
                    else:
                        forces[(r, c, obj_idx)] = matched_par
                elif modifier == "stationary":
                    forces.pop((r, c, obj_idx), None)
                elif modifier == "moving":
                    # Preserve existing force; if this object has no force,
                    # inherit from a co-located LHS 'moving' object in the
                    # same cell, then fall back to any matched moving object.
                    # When forces were consumed by LHS matching, also check
                    # saved_forces_per_cell as a fallback.
                    if (r, c, obj_idx) not in forces:
                        # First try: co-located moving object in same cell
                        if i < len(eff_lhs):
                            for lhs_mod, lhs_name, lhs_idxs in eff_lhs[i]:
                                if lhs_mod == "moving" and lhs_name:
                                    for lhs_idx in lhs_idxs:
                                        f = forces.get((r, c, lhs_idx))
                                        if f is None and i < len(saved_forces_per_cell):
                                            f = saved_forces_per_cell[i].get(lhs_idx)
                                        if f is not None:
                                            forces[(r, c, obj_idx)] = f
                                            break
                                    if (r, c, obj_idx) in forces:
                                        break
                        # Second try: any matched moving object across cells
                        if (r, c, obj_idx) not in forces:
                            for j, (pr, pc) in enumerate(positions):
                                for oidx in lhs_objs_per_cell[j]:
                                    f = forces.get((pr, pc, oidx))
                                    if f is None and j < len(saved_forces_per_cell):
                                        f = saved_forces_per_cell[j].get(oidx)
                                    if f is not None:
                                        forces[(r, c, obj_idx)] = f
                                        break
                                if (r, c, obj_idx) in forces:
                                    break
                        # Third try: cross-group force refs (multi-bracket rules)
                        if (r, c, obj_idx) not in forces and cross_group_force_refs:
                            for ref_r, ref_c, ref_idx in cross_group_force_refs:
                                f = forces.get((ref_r, ref_c, ref_idx))
                                if f is not None:
                                    forces[(r, c, obj_idx)] = f
                                    break
                elif modifier in ("horizontal", "vertical"):
                    # Preserve the axis-matching force from the LHS-matched
                    # object. The LHS clear pass at line 2693 already popped
                    # the original force into saved_forces_per_cell; without
                    # restoring it here, rules like Dotsnake's
                    # ``[HORIZONTAL Player No Tail] -> [HORIZONTAL Player Tail2]``
                    # would strip the player's move force every other turn
                    # (the turn the No-Tail guard matches), and the snake
                    # would only advance on alternating keypresses.
                    axis_dirs = ("left", "right") if modifier == "horizontal" else ("up", "down")
                    existing = forces.get((r, c, obj_idx))
                    if existing not in axis_dirs:
                        matched = None
                        if i < len(saved_forces_per_cell):
                            saved = saved_forces_per_cell[i].get(obj_idx)
                            if saved in axis_dirs:
                                matched = saved
                            else:
                                for old_idx, old_force in saved_forces_per_cell[i].items():
                                    if old_force in axis_dirs and self._objects_share_or_group(old_idx, obj_idx):
                                        matched = old_force
                                        break
                        if matched is None:
                            for j, (pr, pc) in enumerate(positions):
                                if j >= len(eff_lhs):
                                    continue
                                for lhs_mod, lhs_name, lhs_idxs in eff_lhs[j]:
                                    if lhs_mod == modifier and lhs_name:
                                        for lhs_idx in lhs_idxs:
                                            f = forces.get((pr, pc, lhs_idx))
                                            if f is None and j < len(saved_forces_per_cell):
                                                f = saved_forces_per_cell[j].get(lhs_idx)
                                            if f in axis_dirs:
                                                matched = f
                                                break
                                        if matched:
                                            break
                                if matched:
                                    break
                        if matched is None and cross_group_force_refs:
                            for ref_r, ref_c, ref_idx in cross_group_force_refs:
                                f = forces.get((ref_r, ref_c, ref_idx))
                                if f in axis_dirs:
                                    matched = f
                                    break
                        if matched is not None:
                            if existing is not None and existing != matched:
                                forces.pop((r, c, obj_idx), None)
                            else:
                                forces[(r, c, obj_idx)] = matched
                else:
                    # Empty modifier in RHS.  Only clear the object's force
                    # if the LHS explicitly referenced it with a movement
                    # modifier (>, <, ^, v, moving) — that means the rule
                    # intentionally removed the movement.  If both sides use
                    # a plain reference the force is preserved (no-op).
                    should_clear = False
                    if i < len(eff_lhs):
                        for lhs_mod, lhs_name, lhs_idxs in eff_lhs[i]:
                            if lhs_name and lhs_mod in (">", "<", "^", "v", "moving",
                                                          "right", "left", "up", "down",
                                                          "perpendicular", "parallel",
                                                          "horizontal", "vertical"):
                                if obj_idx in lhs_idxs:
                                    should_clear = True
                                    break
                    if should_clear:
                        forces.pop((r, c, obj_idx), None)
                    elif (r, c, obj_idx) not in forces and i < len(saved_forces_per_cell):
                        moved_from_elsewhere = any(
                            obj_idx in lhs_objs_per_cell[j]
                            for j in range(len(positions)) if j != i
                        )
                        if not moved_from_elsewhere:
                            # Same-layer swap: a force on an LHS object that's
                            # being removed transfers to the new RHS object on
                            # the same collision layer (e.g. avatarup → avatardown).
                            # Forces never cross collision layers — without this
                            # check, an unrelated object on a different layer can
                            # absorb a force meant for its layer-mate (e.g.
                            # Aerobatics' [right Plane Turn] -> [planeur Turn]
                            # would otherwise hand the plane's force to the Turn
                            # marker, which is on a different layer).
                            # Restrict further to alternate forms of the same
                            # avatar (sharing an or-group) so that pure markers
                            # like Bridge-toggle Maze's prevPos don't inherit
                            # the player's force and slide off the original
                            # cell during force resolution.
                            obj_layer = self._obj_layers.get(obj_idx, -1)
                            if obj_layer >= 0:
                                # Forces matched by an LHS movement modifier
                                # (e.g. ``DOWN To0``) are consumed by the
                                # match — they must not transfer to the new
                                # RHS object on the same layer, or
                                # ``[DOWN To0] -> [ToSouth]`` would leave
                                # ToSouth still carrying To0's down force
                                # and the directional sprite would drift
                                # during force resolution.
                                consumed_by_lhs: set[int] = set()
                                if i < len(eff_lhs):
                                    for lhs_mod, lhs_name, lhs_idxs in eff_lhs[i]:
                                        if (lhs_mod in (">", "<", "^", "v", "moving",
                                                        "right", "left", "up", "down",
                                                        "perpendicular", "parallel",
                                                        "horizontal", "vertical")
                                                and lhs_name):
                                            consumed_by_lhs.update(lhs_idxs)
                                for old_idx, old_force in saved_forces_per_cell[i].items():
                                    if old_idx == obj_idx:
                                        continue
                                    if old_idx in rhs_objs_per_cell[i]:
                                        continue
                                    if self._obj_layers.get(old_idx, -1) != obj_layer:
                                        continue
                                    if not self._objects_share_or_group(old_idx, obj_idx):
                                        continue
                                    if old_idx in consumed_by_lhs:
                                        continue
                                    forces[(r, c, obj_idx)] = old_force
                                    break

        # Or-group force propagation: in standard PuzzleScript, or-groups
        # in rules are expanded so each member is matched independently.
        # Approximate this by propagating forces from the bound member to
        # all other members of the or-group present in the same cell.
        if rule._rhs_has_or_group and rule._rhs_has_movement_mod:
            for i, (r, c) in enumerate(positions):
                if i < len(eff_rhs):
                    cell = self.grid[r][c]
                    for modifier, obj_name, obj_indices in eff_rhs[i]:
                        if not obj_name or obj_name not in self.game.or_groups:
                            continue
                        if modifier not in (">", "<", "^", "v", "moving",
                                            "right", "left", "up", "down",
                                            "perpendicular", "parallel",
                                            "horizontal", "vertical"):
                            continue
                        bound_force = None
                        for idx in obj_indices:
                            f = forces.get((r, c, idx))
                            if f is not None:
                                bound_force = f
                                break
                        if bound_force is not None:
                            for idx in obj_indices:
                                if idx in cell and (r, c, idx) not in forces:
                                    forces[(r, c, idx)] = bound_force

        # Consume forces for objects moved by the rule (only at matched positions).
        # Excludes ellipsis-dummy positions: those are just internal placeholders
        # marking where the ellipsis began, not real LHS-matched cells. Unrelated
        # objects (with forces set by an earlier rule) happen to live there and
        # must not have their forces silently cleared — e.g. Coin Dropper's wrap
        # rule ``[side|...|right moveable|side]`` whose ellipsis dummy lands on a
        # block that other rules just moved, which would otherwise leave the block
        # stranded inside the side wall.
        ellipsis_position_set: set[tuple[int, int]] = set()
        for i, (r, c) in enumerate(positions):
            if i < len(eff_lhs):
                cell_pattern = eff_lhs[i]
                if any(item[0] == "..." for item in cell_pattern):
                    ellipsis_position_set.add((r, c))
        position_set = {(r, c) for (r, c) in positions} - ellipsis_position_set
        for obj_idx in objects_moved_by_rule:
            keys_to_remove = [k for k in forces
                              if k[2] == obj_idx and (k[0], k[1]) in position_set]
            for k in keys_to_remove:
                r, c, _ = k
                i_pos = None
                for i, pos in enumerate(positions):
                    if pos == (r, c) and i < len(eff_lhs) and not any(
                            item[0] == "..." for item in eff_lhs[i]):
                        i_pos = i
                        break
                if i_pos is not None and obj_idx in rhs_objs_per_cell[i_pos]:
                    mod = rhs_objs_per_cell[i_pos][obj_idx]
                    if mod not in (">", "<", "^", "v",
                                   "up", "down", "left", "right",
                                   "moving",
                                   "perpendicular", "parallel",
                                   "horizontal", "vertical"):
                        del forces[k]
                elif i_pos is not None:
                    del forces[k]

        # Consume action state: if LHS has 'action' modifier for an object
        # but RHS doesn't, remove the object from action_objects so subsequent
        # rules requiring [Action X] won't match it.
        if action_objects and rule._lhs_has_action_mod:
            for i, (r, c) in enumerate(positions):
                if i < len(eff_lhs):
                    cell = self.grid[r][c]
                    lhs_action_objs = set()
                    for modifier, obj_name, obj_indices in eff_lhs[i]:
                        if modifier == "action" and obj_name:
                            for idx in obj_indices:
                                if idx in cell:
                                    lhs_action_objs.add(idx)
                    if lhs_action_objs and i < len(eff_rhs):
                        rhs_has_action = set()
                        for modifier, obj_name, obj_indices in eff_rhs[i]:
                            if modifier == "action" and obj_name:
                                rhs_has_action.update(obj_indices)
                        for idx in lhs_action_objs - rhs_has_action:
                            action_objects.discard((r, c, idx))

        # Produce action state: when RHS has 'action X' modifier, set the
        # action force on X at that cell so subsequent rules requiring
        # [Action X] can match it.  This is how the action force propagates
        # through rules (e.g. laser propagation in Ad Infinitum).
        if action_objects is not None and rule._rhs_has_action_mod:
            for i, (r, c) in enumerate(positions):
                if i < len(eff_rhs):
                    cell = self.grid[r][c]
                    for modifier, obj_name, obj_indices in eff_rhs[i]:
                        if modifier == "action" and obj_name:
                            for idx in obj_indices:
                                if idx in cell:
                                    action_objects.add((r, c, idx))

    def _is_chain_blocked(self, r: int, c: int, obj_idx: int,
                          direction: str, forces: dict) -> bool:
        """Check if an object's push chain is blocked (read-only, no grid changes)."""
        if obj_idx not in self.grid[r][c]:
            return True

        mdr, mdc = _DIR_DELTAS[direction]
        cur_r, cur_c, cur_obj = r, c, obj_idx
        visited: set[tuple[int, int, int]] = set()

        while True:
            if (cur_r, cur_c, cur_obj) in visited:
                return True
            visited.add((cur_r, cur_c, cur_obj))

            nr, nc = cur_r + mdr, cur_c + mdc
            if not (0 <= nr < self.height and 0 <= nc < self.width):
                return True

            cur_layer = self._obj_layers.get(cur_obj, -1)
            blocker = None
            if cur_layer >= 0:
                for existing in self.grid[nr][nc]:
                    if self._obj_layers.get(existing, -1) == cur_layer:
                        blocker = existing
                        break

            if blocker is None:
                return False

            blocker_force = forces.get((nr, nc, blocker))
            if blocker_force == direction:
                cur_r, cur_c, cur_obj = nr, nc, blocker
            else:
                return True

    def _resolve_forces(self, forces: dict[tuple[int, int, int], str]):
        """Resolve all pending forces: move objects that can move.

        Uses chain-based resolution with multi-way conflict detection:
        traces each push chain to its end, checks the endpoint is free,
        then detects when multiple chains target the same cell and blocks
        all of them (PuzzleScript semantics).  Non-conflicting chains
        move simultaneously per iteration; the loop repeats until stable
        so that cells freed in one pass can be claimed in the next.

        RIGID rules: forces produced by a single rigid rule application
        are linked via ``_force_rigid_group``. If any force in the group
        is blocked or conflicts, every other force in the same group is
        cancelled too.
        """
        if not forces:
            return

        def _cancel_rigid_cascade(seed_keys, forces, resolved):
            """Cancel any force keys that share a rigid group with seed_keys.
            Returns the (possibly extended) set of cancelled keys."""
            if not self._force_rigid_group:
                return set(seed_keys)
            dead_groups = set()
            for key in seed_keys:
                gid = self._force_rigid_group.get(key)
                if gid is not None:
                    dead_groups.add(gid)
            if not dead_groups:
                return set(seed_keys)
            extra = set(seed_keys)
            for key, gid in list(self._force_rigid_group.items()):
                if gid in dead_groups and key in forces:
                    extra.add(key)
            for key in extra:
                forces.pop(key, None)
                resolved.add(key)
                self._force_rigid_group.pop(key, None)
            return extra

        max_iters = 20
        for _iter in range(max_iters):
            moved_any = False
            deferred_any = False
            forces_at_iter_start = dict(forces)
            resolved: set[tuple[int, int, int]] = set()

            # Phase 1: Plan — trace every chain and determine if its
            # endpoint is free on the *current* grid.
            movable = []   # (chain, mdr, mdc, endpoint_target)
            blocked_keys: set[tuple[int, int, int]] = set()
            for (r, c, obj_idx), direction in list(forces.items()):
                if (r, c, obj_idx) in resolved:
                    continue
                if obj_idx not in self.grid[r][c]:
                    forces.pop((r, c, obj_idx), None)
                    resolved.add((r, c, obj_idx))
                    self._force_rigid_group.pop((r, c, obj_idx), None)
                    continue

                mdr, mdc = _DIR_DELTAS[direction]

                chain = [(r, c, obj_idx)]
                cur_r, cur_c, cur_obj = r, c, obj_idx
                chain_free = False
                blocker_is_mover = False
                while True:
                    nr, nc = cur_r + mdr, cur_c + mdc
                    if not (0 <= nr < self.height and 0 <= nc < self.width):
                        break

                    cur_layer = self._obj_layers.get(cur_obj, -1)
                    blocker = None
                    if cur_layer >= 0:
                        for existing in self.grid[nr][nc]:
                            if self._obj_layers.get(existing, -1) == cur_layer:
                                blocker = existing
                                break

                    if blocker is None:
                        chain_free = True
                        break

                    blocker_force = forces.get((nr, nc, blocker))
                    if blocker_force == direction:
                        chain.append((nr, nc, blocker))
                        cur_r, cur_c, cur_obj = nr, nc, blocker
                    else:
                        if blocker_force is not None:
                            blocker_is_mover = True
                        break

                if chain_free:
                    last_r, last_c, _ = chain[-1]
                    endpoint_target = (last_r + mdr, last_c + mdc)
                    movable.append((chain, mdr, mdc, endpoint_target))
                elif blocker_is_mover:
                    # Blocker has a force in a different direction. It may
                    # vacate the cell during this resolution pass, so leave
                    # this chain's forces in place and retry next iteration.
                    # Without this, e.g. Circulando's clockwise rotation
                    # would drop the "follower" crate's force the moment the
                    # leader (heading a different direction) was traced.
                    deferred_any = True
                else:
                    # Pop force but defer removing the rigid-group tag until
                    # after the cascade has had a chance to observe it.
                    for cr, cc, oidx in chain:
                        forces.pop((cr, cc, oidx), None)
                        resolved.add((cr, cc, oidx))
                        blocked_keys.add((cr, cc, oidx))

            # Rigid cascade for blocked chains: if any blocked key was in a
            # rigid group, cancel every other force in that group and drop
            # any planned chain that contained one of those forces.
            if blocked_keys and self._force_rigid_group:
                extra_cancelled = _cancel_rigid_cascade(blocked_keys, forces, resolved)
                if extra_cancelled - blocked_keys:
                    movable = [m for m in movable
                               if not any(ent in extra_cancelled for ent in m[0])]
            # Drop rigid-group tags for the originally blocked keys.
            for key in blocked_keys:
                self._force_rigid_group.pop(key, None)

            # Phase 2: Subsume shorter chains whose head entity is
            # already part of a longer chain (same logical push group).
            all_non_head: dict[tuple[int,int,int], int] = {}
            for i, (chain, _, _, _) in enumerate(movable):
                for ent in chain[1:]:
                    all_non_head[ent] = i
            subsumed: set[int] = set()
            for i, (chain, _, _, _) in enumerate(movable):
                if chain[0] in all_non_head:
                    subsumed.add(i)

            # Phase 3: Detect multi-way conflicts among independent
            # (non-subsumed) chains. Two chains conflict only if any of
            # their entity destinations land on the same cell AND the
            # same collision layer — chains targeting the same cell on
            # different layers (e.g. Player layer-2 and PlayerR layer-5
            # moving together) coexist legally.
            target_claims: dict[tuple[int, int, int], int] = {}
            conflicting: set[int] = set()
            for i, (chain, mdr, mdc, _) in enumerate(movable):
                if i in subsumed:
                    continue
                for cr, cc, oidx in chain:
                    nr, nc = cr + mdr, cc + mdc
                    layer = self._obj_layers.get(oidx, -1)
                    key = (nr, nc, layer)
                    existing = target_claims.get(key)
                    if existing is not None and existing != i:
                        conflicting.add(i)
                        conflicting.add(existing)
                    else:
                        target_claims[key] = i

            # Rigid cascade for conflicts: extend the conflicting set with
            # any chain sharing a rigid group with a conflicting chain.
            if conflicting and self._force_rigid_group:
                conflict_keys = set()
                for i in conflicting:
                    for ent in movable[i][0]:
                        conflict_keys.add(ent)
                conflict_groups = {
                    self._force_rigid_group[k]
                    for k in conflict_keys
                    if k in self._force_rigid_group
                }
                if conflict_groups:
                    extended_keys = set(conflict_keys)
                    for key, gid in self._force_rigid_group.items():
                        if gid in conflict_groups:
                            extended_keys.add(key)
                    for i, (chain, _, _, _) in enumerate(movable):
                        if i in subsumed or i in conflicting:
                            continue
                        if any(ent in extended_keys for ent in chain):
                            conflicting.add(i)

            # Phase 4: Move non-conflicting, non-subsumed chains;
            # block conflicting and subsumed ones.
            for i, (chain, mdr, mdc, ep) in enumerate(movable):
                if i in subsumed:
                    continue
                if i in conflicting:
                    for cr, cc, oidx in chain:
                        forces.pop((cr, cc, oidx), None)
                        resolved.add((cr, cc, oidx))
                        self._force_rigid_group.pop((cr, cc, oidx), None)
                    continue

                for cr, cc, oidx in reversed(chain):
                    self._cell_discard(cr, cc, oidx)
                    self._cell_add(cr + mdr, cc + mdc, oidx)
                    forces.pop((cr, cc, oidx), None)
                    resolved.add((cr, cc, oidx))
                    self._force_rigid_group.pop((cr, cc, oidx), None)
                    moved_any = True

            if not moved_any:
                # If the only remaining work is deferred chains and no
                # mover changed state this iteration, we're deadlocked
                # (e.g. two crates trying to swap places). Drop the
                # deferred forces so they don't survive to late rules.
                if deferred_any and forces == forces_at_iter_start:
                    for key in list(forces.keys()):
                        forces.pop(key, None)
                        self._force_rigid_group.pop(key, None)
                break

    def _apply_late_rules(self, direction: str):
        """Apply late rules, handling startloop/endloop groups.

        Rules sharing the same ``loop_id`` (from startloop/endloop) are
        applied together in a loop until no rule in the group fires.
        Rules sharing the same ``group_id`` (from ``+`` prefix) are
        applied together as a sub-loop within their context.
        Standalone rules are applied until individually stable.
        ``direction`` is the player's input (forwarded for ``action`` modifier).
        """
        eligible = [r for r in self.game.rules if r.is_late]

        segments = self._build_late_rule_segments(eligible)
        for seg in segments:
            self._execute_late_segment(seg, direction)

    def _build_late_rule_segments(self, rules: list) -> list:
        """Organize late rules into segments respecting loop_id and group_id."""
        segments = []
        i = 0
        while i < len(rules):
            rule = rules[i]

            if rule.loop_id >= 0:
                lid = rule.loop_id
                loop_end = i + 1
                while loop_end < len(rules) and rules[loop_end].loop_id == lid:
                    loop_end += 1
                loop_rules = rules[i:loop_end]
                sub_segments = self._build_late_plus_segments(loop_rules)
                segments.append(('loop', sub_segments))
                i = loop_end
            else:
                gid = rule.group_id
                group_end = i + 1
                while group_end < len(rules) and rules[group_end].group_id == gid:
                    group_end += 1
                if group_end - i > 1:
                    segments.append(('plus', rules[i:group_end]))
                else:
                    segments.append(('single', rule))
                i = group_end
        return segments

    def _build_late_plus_segments(self, rules: list) -> list:
        """Build sub-segments within a startloop block for late rules."""
        segments = []
        i = 0
        while i < len(rules):
            rule = rules[i]
            gid = rule.group_id
            group_end = i + 1
            while group_end < len(rules) and rules[group_end].group_id == gid:
                group_end += 1
            if group_end - i > 1:
                segments.append(('plus', rules[i:group_end]))
            else:
                segments.append(('single', rule))
            i = group_end
        return segments

    def _execute_late_segment(self, seg, direction: str):
        """Execute a late-rule segment."""
        kind = seg[0]

        if kind == 'single':
            rule = seg[1]
            self._execute_late_single(rule, direction)

        elif kind == 'plus':
            group_rules = seg[1]
            for _ in range(200):
                any_changed = False
                for grule in group_rules:
                    if self._execute_late_single_check_fired(grule, direction):
                        any_changed = True
                if not any_changed:
                    break

        elif kind == 'loop':
            sub_segments = seg[1]
            for _ in range(200):
                mut_at_start = self._mutation_counter
                for sub_seg in sub_segments:
                    self._execute_late_segment(sub_seg, direction)
                if self._mutation_counter == mut_at_start:
                    break

    def _execute_late_single(self, rule, direction: str):
        """Execute a single late rule (loop until stable per direction)."""
        if (rule.cancel or rule.win or rule.restart) and not rule.groups_rhs:
            if rule.cancel:
                if self._check_late_command_match(rule, input_dir=direction):
                    self._late_cancel = True
            elif rule.win:
                if self._check_late_command_match(rule, input_dir=direction):
                    self._rule_win = True
            elif rule.restart:
                if self._check_late_command_match(rule, input_dir=direction):
                    self._rule_restart = True
            return

        # For transformation rules with trailing command flags, pre-check the
        # LHS match — the rule's RHS may be a no-op (LHS == RHS) so the
        # apply_late_rule "grid changed" return cannot be used to detect firing.
        command_matched = False
        if rule.cancel or rule.win or rule.restart:
            command_matched = self._check_late_command_match(rule, input_dir=direction)

        for dir_name in rule.directions:
            for _ in range(200):
                matched, changed = self._apply_late_rule(rule, dir_name,
                                                input_dir=direction)
                if matched and rule.again:
                    self._again_triggered = True
                if not changed or rule.random:
                    break
            if rule.random:
                break

        if command_matched:
            if rule.cancel:
                self._late_cancel = True
            if rule.win:
                self._rule_win = True
            if rule.restart:
                self._rule_restart = True

    def _execute_late_single_check_fired(self, rule, direction: str) -> bool:
        """Execute a single late rule, returning True if it fired."""
        if (rule.cancel or rule.win or rule.restart) and not rule.groups_rhs:
            if rule.cancel:
                if self._check_late_command_match(rule, input_dir=direction):
                    self._late_cancel = True
            elif rule.win:
                if self._check_late_command_match(rule, input_dir=direction):
                    self._rule_win = True
            elif rule.restart:
                if self._check_late_command_match(rule, input_dir=direction):
                    self._rule_restart = True
            return False

        command_matched = False
        if rule.cancel or rule.win or rule.restart:
            command_matched = self._check_late_command_match(rule, input_dir=direction)

        any_fired = False
        for dir_name in rule.directions:
            matched, changed = self._apply_late_rule(rule, dir_name, input_dir=direction)
            if matched:
                if rule.again:
                    self._again_triggered = True
            if changed:
                any_fired = True
                if rule.random:
                    break

        if command_matched:
            if rule.cancel:
                self._late_cancel = True
            if rule.win:
                self._rule_win = True
            if rule.restart:
                self._rule_restart = True
            any_fired = any_fired or True

        if rule.random:
            return any_fired
        return any_fired

    def _check_late_command_match(self, rule: PSRule, input_dir: str = None) -> bool:
        """Check if a late win/restart rule's LHS matches anywhere."""
        for dir_name in rule.directions:
            dr, dc = _DIR_DELTAS.get(dir_name, (0, 0))
            dir_symbol_map = self._get_dir_symbol_map(dir_name)
            all_groups_match = True
            for group_lhs in rule.groups_lhs:
                found = False
                for r in range(self.height):
                    if found:
                        break
                    for c in range(self.width):
                        m = self._check_late_rule_match(
                            rule, r, c, dr, dc, dir_symbol_map,
                            input_dir=input_dir, lhs_patterns=group_lhs)
                        if m is not None:
                            found = True
                            break
                if not found:
                    all_groups_match = False
                    break
            if all_groups_match:
                return True
        return False

    def _apply_late_rule(self, rule: PSRule, direction: str,
                          input_dir: str = None) -> tuple[bool, bool]:
        """Apply a late rule once. Returns (matched, changed).

        ``matched`` is True if the LHS matched anywhere (even if RHS == LHS so
        the grid did not change — this is what ``again`` keys off in
        PuzzleScript). ``changed`` is True if the grid actually changed.
        """
        dr, dc = _DIR_DELTAS.get(direction, (0, 0))
        dir_symbol_map = self._get_dir_symbol_map(direction)

        if len(rule.groups_lhs) > 1:
            return self._apply_multi_group_late_rule(
                rule, direction, input_dir=input_dir)

        # Find all matches (single-group path), expanding ... to all positions
        matches = []
        first_required = rule._first_cell_required_per_group[0] if rule._first_cell_required_per_group else None
        candidates = self._candidate_positions(first_required)
        if candidates is None:
            for r in range(self.height):
                for c in range(self.width):
                    all_m = self._find_all_late_rule_matches(
                        rule, r, c, dr, dc, dir_symbol_map,
                        input_dir=input_dir)
                    matches.extend(all_m)
        else:
            for (r, c) in candidates:
                all_m = self._find_all_late_rule_matches(
                    rule, r, c, dr, dc, dir_symbol_map,
                    input_dir=input_dir)
                matches.extend(all_m)

        if not matches:
            return False, False

        if rule.random:
            matches = [_random_module.choice(matches)]

        # Snapshot mutation counter to detect actual changes
        mut_at_start = self._mutation_counter

        # Apply all matches
        for match in matches:
            self._apply_late_rule_match(rule, match)

        if self._mutation_counter != mut_at_start:
            return True, True
        return True, False

    def _apply_multi_group_late_rule(self, rule: PSRule, direction: str,
                                      input_dir: str = None) -> tuple[bool, bool]:
        """Apply a multi-bracket-group late rule once. Returns (matched, changed).

        Iterates the Cartesian product of per-group matches, re-validating
        each tuple against the (now-mutated) grid. Matches PuzzleScript's
        reference engine semantics.
        """
        dr, dc = _DIR_DELTAS.get(direction, (0, 0))
        dir_symbol_map = self._get_dir_symbol_map(direction)

        # Each group must match independently
        all_group_matches: list[list] = []
        for gi, group_lhs in enumerate(rule.groups_lhs):
            matches = []
            first_required = (rule._first_cell_required_per_group[gi]
                              if gi < len(rule._first_cell_required_per_group) else None)
            candidates = self._candidate_positions(first_required)
            if candidates is None:
                for r in range(self.height):
                    for c in range(self.width):
                        all_m = self._find_all_late_rule_matches(
                            rule, r, c, dr, dc, dir_symbol_map,
                            input_dir=input_dir, lhs_patterns=group_lhs)
                        matches.extend(all_m)
            else:
                for (r, c) in candidates:
                    all_m = self._find_all_late_rule_matches(
                        rule, r, c, dr, dc, dir_symbol_map,
                        input_dir=input_dir, lhs_patterns=group_lhs)
                    matches.extend(all_m)
            if not matches:
                return False, False
            all_group_matches.append(matches)

        # Overlap preference: when a later group has multiple matches and
        # some share cells with the first match of an earlier group, prefer
        # the overlapping one. Mirrors _apply_multi_group_rule_forces —
        # skipped when the group references an or-group.
        chosen_cells: set[tuple[int, int]] = set(all_group_matches[0][0])
        for gi in range(1, len(all_group_matches)):
            group_lhs = rule.groups_lhs[gi]
            uses_or_group = any(
                obj_name in self.game.or_groups
                for cell_pattern in group_lhs
                for modifier, obj_name, _idxs in cell_pattern
                if obj_name and modifier not in ("no", "...")
            )
            if not uses_or_group and len(all_group_matches[gi]) > 1:
                overlapping = [m for m in all_group_matches[gi]
                               if any(pos in chosen_cells for pos in m)]
                if overlapping:
                    all_group_matches[gi] = overlapping
            chosen_cells.update(all_group_matches[gi][0])

        mut_at_start = self._mutation_counter

        # Build cross-group or-bindings from ALL groups' LHS matches
        cross_group_or_bindings: dict[str, int] = {}
        for gi, matches in enumerate(all_group_matches):
            g_lhs = rule.groups_lhs[gi]
            for match in matches:
                for i, (r, c) in enumerate(match):
                    if i < len(g_lhs):
                        cell = self.grid[r][c]
                        for modifier, obj_name, obj_indices in g_lhs[i]:
                            if modifier in ("no", "...") or not obj_name:
                                continue
                            if obj_name in cross_group_or_bindings:
                                continue
                            if obj_name in self.game.or_groups:
                                for idx in obj_indices:
                                    if idx in cell:
                                        cross_group_or_bindings[obj_name] = idx
                                        break

        for tuple_idx, match_tuple in enumerate(_itertools.product(*all_group_matches)):
            if tuple_idx > 0:
                valid = True
                for gi, match in enumerate(match_tuple):
                    if not match:
                        valid = False
                        break
                    r0, c0 = match[0]
                    new_m = self._check_late_rule_match(
                        rule, r0, c0, dr, dc, dir_symbol_map,
                        input_dir=input_dir,
                        lhs_patterns=rule.groups_lhs[gi])
                    if new_m is None or list(new_m) != list(match):
                        valid = False
                        break
                if not valid:
                    continue
            for gi, match in enumerate(match_tuple):
                g_lhs = rule.groups_lhs[gi]
                g_rhs = rule.groups_rhs[gi] if gi < len(rule.groups_rhs) else []
                self._apply_late_rule_match(
                    rule, match, lhs_patterns=g_lhs, rhs_patterns=g_rhs,
                    cross_group_or_bindings=cross_group_or_bindings)

        if self._mutation_counter != mut_at_start:
            return True, True
        return True, False

    def _check_late_rule_match(self, rule: PSRule, r: int, c: int,
                                dr: int, dc: int, dir_map: dict,
                                input_dir: str = None,
                                lhs_patterns: list = None) -> Optional[list[tuple[int, int]]]:
        """Check late rule match (no force checking)."""
        positions = []
        cr, cc = r, c

        lhs = lhs_patterns if lhs_patterns is not None else rule.patterns_lhs
        meta_map = self.game._pattern_meta
        i = 0
        while i < len(lhs):
            cell_pattern = lhs[i]
            cell_meta = meta_map.get(id(cell_pattern))

            # Check if this cell is a pure ellipsis (...) — wildcard gap
            if cell_meta["has_ellipsis"] if cell_meta else any(item[0] == "..." for item in cell_pattern):
                next_i = i + 1
                if next_i >= len(lhs):
                    positions.append((cr, cc))
                    i = next_i
                    continue

                next_pattern = lhs[next_i]

                # Search forward from (cr, cc) for a cell matching next_pattern
                search_r, search_c = cr, cc
                found = False
                while 0 <= search_r < self.height and 0 <= search_c < self.width:
                    if self._late_cell_matches(next_pattern, search_r, search_c,
                                                input_dir=input_dir):
                        found = True
                        break
                    search_r += dr
                    search_c += dc

                if not found:
                    return None

                positions.append((cr, cc))
                cr, cc = search_r, search_c
                i = next_i
                continue

            # Normal cell matching
            if cr < 0 or cr >= self.height or cc < 0 or cc >= self.width:
                return None

            if not self._late_cell_matches(cell_pattern, cr, cc,
                                            input_dir=input_dir):
                return None

            positions.append((cr, cc))
            cr += dr
            cc += dc
            i += 1

        return positions

    def _find_all_late_rule_matches(self, rule: PSRule, r: int, c: int,
                                     dr: int, dc: int, dir_map: dict,
                                     input_dir: str = None,
                                     lhs_patterns: list = None) -> list[list[tuple[int, int]]]:
        """Find ALL matches expanding ellipsis (...) to every valid gap length.

        In PuzzleScript, rules with ``...`` are effectively unrolled into
        one match per possible gap length.  This method returns all of them
        so that, e.g., ``late [ Whale | ... | ] -> [ Whale | ... | HBeam ]``
        places a beam at *every* cell from the whale to the grid edge, not
        just the nearest cell.
        """
        lhs = lhs_patterns if lhs_patterns is not None else rule.patterns_lhs
        meta_map = self.game._pattern_meta

        has_ellipsis = False
        for cp in lhs:
            cell_meta = meta_map.get(id(cp))
            if cell_meta["has_ellipsis"] if cell_meta else any(item[0] == "..." for item in cp):
                has_ellipsis = True
                break
        if not has_ellipsis:
            m = self._check_late_rule_match(rule, r, c, dr, dc, dir_map,
                                             input_dir, lhs_patterns)
            return [m] if m is not None else []

        # Match elements before the ellipsis
        positions_before: list[tuple[int, int]] = []
        cr, cc = r, c
        ellipsis_idx: Optional[int] = None

        for i in range(len(lhs)):
            cell_pattern = lhs[i]
            cell_meta = meta_map.get(id(cell_pattern))
            if cell_meta["has_ellipsis"] if cell_meta else any(item[0] == "..." for item in cell_pattern):
                ellipsis_idx = i
                break
            if cr < 0 or cr >= self.height or cc < 0 or cc >= self.width:
                return []
            if not self._late_cell_matches(cell_pattern, cr, cc,
                                            input_dir=input_dir):
                return []
            positions_before.append((cr, cc))
            cr += dr
            cc += dc

        if ellipsis_idx is None:
            return []

        after_patterns = lhs[ellipsis_idx + 1:]
        ellipsis_dummy = (cr, cc)

        results: list[list[tuple[int, int]]] = []
        sr, sc = cr, cc
        while 0 <= sr < self.height and 0 <= sc < self.width:
            rest_r, rest_c = sr, sc
            valid = True
            rest_pos: list[tuple[int, int]] = []
            for cp in after_patterns:
                if rest_r < 0 or rest_r >= self.height or rest_c < 0 or rest_c >= self.width:
                    valid = False
                    break
                if not self._late_cell_matches(cp, rest_r, rest_c,
                                                input_dir=input_dir):
                    valid = False
                    break
                rest_pos.append((rest_r, rest_c))
                rest_r += dr
                rest_c += dc

            if valid and len(rest_pos) == len(after_patterns):
                full = list(positions_before) + [ellipsis_dummy] + rest_pos
                results.append(full)

            sr += dr
            sc += dc

        return results

    def _late_cell_matches(self, cell_pattern,
                            r: int, c: int, input_dir: str = None) -> bool:
        """Check if a single cell pattern matches for late rules (no forces)."""
        cell = self.grid[r][c]
        for modifier, obj_name, obj_indices in cell_pattern:
            if modifier == "...":
                continue
            if modifier == "no":
                for idx in obj_indices:
                    if idx in cell:
                        return False
            elif modifier == "action":
                if input_dir != "action":
                    return False
                found = any(idx in cell for idx in obj_indices)
                if not found:
                    return False
            elif modifier == "moving":
                # Nothing is moving in the late phase — always fails
                return False
            elif modifier == "perpendicular":
                return False
            elif obj_name and obj_indices:
                # stationary or plain — just check presence (no forces in late phase)
                found = any(idx in cell for idx in obj_indices)
                if not found:
                    return False
        return True

    def _apply_late_rule_match(self, rule: PSRule, positions: list[tuple[int, int]],
                                lhs_patterns: list = None, rhs_patterns: list = None,
                                cross_group_or_bindings: dict = None):
        """Apply a late rule match: directly modify the grid based on RHS."""
        eff_lhs = lhs_patterns if lhs_patterns is not None else rule.patterns_lhs
        eff_rhs = rhs_patterns if rhs_patterns is not None else rule.patterns_rhs

        # Build or-group binding from LHS (before any grid modifications)
        lhs_group_member: dict[str, int] = {}
        for i, (r, c) in enumerate(positions):
            if i < len(eff_lhs):
                cell = self.grid[r][c]
                for modifier, obj_name, obj_indices in eff_lhs[i]:
                    if modifier in ("no", "...") or not obj_name:
                        continue
                    if obj_name in lhs_group_member:
                        continue
                    if obj_name in self.game.or_groups:
                        for idx in obj_indices:
                            if idx in cell:
                                lhs_group_member[obj_name] = idx
                                break

        # Merge cross-group or-bindings (from other groups' LHS matches)
        if cross_group_or_bindings:
            for name, idx in cross_group_or_bindings.items():
                if name not in lhs_group_member:
                    lhs_group_member[name] = idx

        for i, (r, c) in enumerate(positions):
            if i >= len(eff_rhs):
                break

            rhs_cell = eff_rhs[i]
            lhs_cell = eff_lhs[i]
            cell = self.grid[r][c]

            # Determine LHS objects at this position.
            # For or-groups, prefer the rule-wide bound member, but fall back to
            # whichever member THIS cell actually holds when the binding (taken
            # from the first cell that matched the group) is not here. An
            # or-group binds per cell in PuzzleScript, so a rule naming the same
            # group in two cells matches a different member in each: without the
            # fallback the second cell's member is not in ``lhs_objs``, so an RHS
            # that drops the group never removes it -- and the RHS re-add below
            # then plants the FIRST cell's member on top of it. Four-room tilt
            # mazes' ``late [Player Target | Player Target] -> [Player | Player]``
            # is the case: each 2x2 target cell holds a different T1..T4, so only
            # one of every pair was ever cleared, one target could never be
            # removed at all, and its ``No Target`` win was unreachable on every
            # level. Mirrors the non-late path in _apply_rule_match_forces.
            # BLAST RADIUS: a static scan of all 810 parsed games finds 27 whose
            # late rules name an or-group in two cells of one bracket, and random
            # play reaches this branch in Castle Elsewhere, Dharma Dojo demake,
            # Match Flow, Party Demon, Straighten Up and stick candy puzzle saga
            # -- none of which has a generator here, so no recorded corpus moves.
            # The one that does (FROWN INVERSION SQUAD) re-generates byte-
            # identical, verified seed 0 and 1 across the change.
            lhs_objs = set()
            for modifier, obj_name, obj_indices in lhs_cell:
                if modifier != "no" and modifier != "..." and obj_name:
                    if obj_name in lhs_group_member:
                        bound_idx = lhs_group_member[obj_name]
                        if bound_idx in cell:
                            lhs_objs.add(bound_idx)
                        else:
                            for idx in obj_indices:
                                if idx in cell:
                                    lhs_objs.add(idx)
                    else:
                        for idx in obj_indices:
                            if idx in cell:
                                lhs_objs.add(idx)

            # Determine RHS objects
            rhs_objs = set()
            rhs_remove = set()
            rhs_has_content = False
            for modifier, obj_name, obj_indices in rhs_cell:
                if modifier == "..." or not obj_name:
                    continue
                rhs_has_content = True
                if modifier == "no":
                    # RHS 'no' means explicitly remove this object
                    for idx in obj_indices:
                        rhs_remove.add(idx)
                    continue
                if modifier == "random":
                    if obj_name in self.game.or_groups:
                        choices = [self.game.obj_name_to_idx[n]
                                   for n in self.game.or_groups[obj_name]
                                   if n in self.game.obj_name_to_idx]
                    else:
                        choices = list(obj_indices)
                    if choices:
                        rhs_objs.add(_random_module.choice(choices))
                    continue
                # For or-groups, prefer the member this cell's own LHS
                # matched, then the rule-wide LHS binding.  Reading the
                # member "already in the cell" first is wrong for teleport
                # rules, where the RHS cell has no LHS occurrence of the
                # group: an unrelated member sitting in the destination
                # would capture the binding and the object named by the
                # LHS would be deleted instead of moved.  Impasse's screen
                # wrap — ``[Border Mover|...| |Border] -> [Border|...|Mover|
                # Border]`` — did exactly that: with another Mover in the
                # landing cell, the Player was discarded from the border
                # cell and never re-added, so it vanished from the grid.
                # Mirrors the non-late path in _apply_rule_match_forces.
                matched_idx = None
                if obj_name in self.game.or_groups:
                    for idx in obj_indices:
                        if idx in lhs_objs:
                            matched_idx = idx
                            break
                if matched_idx is None and obj_name in lhs_group_member:
                    matched_idx = lhs_group_member[obj_name]
                if matched_idx is None:
                    for idx in obj_indices:
                        if idx in cell:
                            matched_idx = idx
                            break
                    if matched_idx is None:
                        matched_idx = obj_indices[0] if obj_indices else None
                if matched_idx is not None:
                    rhs_objs.add(matched_idx)

            # Explicitly remove objects marked with 'no'
            for idx in rhs_remove:
                self._cell_discard(r, c, idx)

            if not rhs_has_content and lhs_objs:
                # Empty RHS cell — remove all LHS objects
                for idx in lhs_objs:
                    self._cell_discard(r, c, idx)
            elif rhs_has_content:
                # Remove LHS objects not in RHS (excluding already-removed 'no' objects)
                for idx in lhs_objs:
                    if idx not in rhs_objs and idx not in rhs_remove:
                        self._cell_discard(r, c, idx)
                # Add RHS objects not already in cell
                for idx in rhs_objs:
                    if idx not in self.grid[r][c]:
                        obj_layer = self._obj_layers.get(idx, -1)
                        if obj_layer >= 0:
                            for x in list(self.grid[r][c]):
                                if x != idx and self._obj_layers.get(x, -1) == obj_layer:
                                    self._cell_discard(r, c, x)
                        self._cell_add(r, c, idx)

    def _get_dir_symbol_map(self, direction: str) -> dict[str, str]:
        """Map direction symbols (>, <, ^, v) to actual directions based on rule direction."""
        if direction == "right":
            return {">": "right", "<": "left", "^": "up", "v": "down"}
        elif direction == "left":
            return {">": "left", "<": "right", "^": "down", "v": "up"}
        elif direction == "up":
            return {">": "up", "<": "down", "^": "left", "v": "right"}
        elif direction == "down":
            return {">": "down", "<": "up", "^": "right", "v": "left"}
        return {">": "right", "<": "left", "^": "up", "v": "down"}

    def check_game_over(self) -> bool:
        """Check if a player death has occurred (PlayerDead, DeadPlayer, or
        Dying on grid). Without ``deadplayer`` here, Dharma Dojo's death
        state — triggered when a block reaches the ceiling and the screen
        fills with the red DeathCover overlay — leaves the engine reporting
        NOT_FINISHED even though the game is unrecoverably over.
        """
        # list(...) is load-bearing: `resolve_object_name` hands back the
        # MEMOISED list, so `+=` used to extend the cache entry for
        # "playerdead" in place -- it grew by one index on every call, making
        # this scan quadratic in the number of calls (2.5 ms per call on a
        # 10x10 grid after a few thousand) and handing the same polluted list
        # to any other caller. The only external reader is
        # generate_squeamish_chickens_training.py, whose game resolves both
        # "deadplayer" and "dying" to nothing, so the fix appends nothing
        # there and is inert.
        dead_indices = list(self.game.resolve_object_name("playerdead"))
        dead_indices += self.game.resolve_object_name("deadplayer")
        dead_indices += self.game.resolve_object_name("dying")
        if not dead_indices:
            return False
        for r in range(self.height):
            for c in range(self.width):
                for idx in dead_indices:
                    if idx in self.grid[r][c]:
                        return True
        return False

    def step_animation_once(self) -> bool:
        """Run a single ``again`` continuation tick.

        Returns True if the animation is still ongoing after this tick (i.e.
        another tick should follow). Returns False once the cycle has
        terminated — either because no rule fired ``again``, a late cancel
        fired, or the win condition is now satisfied.
        """
        if self.check_win():
            return False
        self._again_triggered = False
        forces: dict[tuple[int, int, int], str] = {}
        self._force_rigid_group = {}
        action_objects: set[tuple[int, int, int]] = set()
        _, any_again = self._apply_rules_with_forces(
            "action", forces, late=False,
            is_action_turn=False,
            action_objects=action_objects,
        )
        if any_again:
            self._again_triggered = True
        self._resolve_forces(forces)
        self._apply_late_rules("action")
        if self._late_cancel or not self._again_triggered:
            return False
        if self.check_win():
            return False
        return True

    def continue_animation(self, max_iter: int = 300) -> None:
        """Drain an in-progress ``again`` animation without applying new input.

        Runs the regular + late rule pipeline with no forces and no action
        signal, exactly as the engine would on each ``again`` continuation.
        Stops when the win condition is met, no rule fires ``again``, or
        ``max_iter`` ticks have elapsed (the latter guarding against rules
        that keep emitting ``again`` indefinitely, like botsket_ball's
        playl→playu→playr→playd cycle).
        """
        for _ in range(max_iter):
            if not self.step_animation_once():
                return

    def check_win(self) -> bool:
        """Check if all win conditions are satisfied."""
        if self._rule_win:
            return True
        if not self.game.win_conditions:
            return False

        for wc in self.game.win_conditions:
            if not self._check_single_win_condition(wc):
                return False
        return True

    def _any_player_on_grid(self) -> bool:
        """True while at least one object of any player class is on the board.

        Used only by the "the player was deleted, that is not a win" guard in
        `_check_single_win_condition`; see the comment there."""
        players = set(self._player_indices)
        if not players:
            return False
        return any(cell & players for row in self.grid for cell in row)

    def _check_single_win_condition(self, wc: PSWinCondition) -> bool:
        """Check a single win condition."""
        obj_indices = self.game.resolve_object_name(wc.obj_name)
        on_indices = self.game.resolve_object_name(wc.on_obj) if wc.on_obj else []

        if wc.quantifier == "all":
            if wc.on_obj:
                # All X must be on Y
                found_any = False
                for r in range(self.height):
                    for c in range(self.width):
                        has_obj = any(idx in self.grid[r][c] for idx in obj_indices)
                        if has_obj:
                            found_any = True
                            has_on = any(idx in self.grid[r][c] for idx in on_indices)
                            if not has_on:
                                return False
                if (not found_any
                        and any(idx in self._player_indices
                                for idx in obj_indices)
                        and not self._any_player_on_grid()):
                    # "All Player on Goal" is vacuously true when no player
                    # is left on the grid.  A player that has been deleted
                    # (rather than moved) is a broken state, never a win —
                    # without this guard an engine slip that loses the
                    # player instantly "solves" the level and poisons any
                    # recorded demonstration.  Only the player is guarded:
                    # other objects (crates, gems, …) are legitimately
                    # consumed by some games.
                    #
                    # The guard asks whether ANY player object survives, not
                    # whether THIS one does.  A game with several player
                    # classes routinely turns one into another, leaving the
                    # class named by a win condition legitimately absent while
                    # the player is very much still on the board: Bichrome
                    # models "which of the two characters you are steering" as
                    # PlayerOrange <-> PlayerOrangeActive, so two of its four
                    # win conditions are ALWAYS vacuous and a per-class guard
                    # made every level of it permanently unwinnable.
                    return False
                return True
            else:
                return True

        elif wc.quantifier == "no":
            # No X present anywhere (or no X on Y)
            for r in range(self.height):
                for c in range(self.width):
                    if any(idx in self.grid[r][c] for idx in obj_indices):
                        if wc.on_obj:
                            has_on = any(idx in self.grid[r][c] for idx in on_indices)
                            if has_on:
                                return False
                        else:
                            return False
            return True

        elif wc.quantifier == "some":
            # At least one X exists (optionally on Y)
            for r in range(self.height):
                for c in range(self.width):
                    if any(idx in self.grid[r][c] for idx in obj_indices):
                        if wc.on_obj:
                            if any(idx in self.grid[r][c] for idx in on_indices):
                                return True
                        else:
                            return True
            return False

        return True


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def _render_cell_sprite(cell_objects: list[tuple[int, PSObject]], cell_px: int,
                        value_fn=None, empty: int = 0) -> np.ndarray:
    """Render a single cell as a cell_px × cell_px block with composited sprites.

    cell_objects: list of (layer, PSObject) sorted by layer ascending (bottom first).

    value_fn: optional (obj, cell_objects) -> int. When given, each object is painted with
    that LABEL id over its sprite footprint (transparency still taken from the sprite), instead
    of the sprite's palette colours -- this is how ground-truth label maps (type/color/state)
    are rendered through the exact same geometry/compositing as the colour frame. cell_objects
    is passed so value_fn can express per-cell attributes (e.g. a block is 'selected' iff the
    selector shares its cell). `empty` fills pixels no object drew.

    Returns: (cell_px, cell_px) int16 array (colour indices by default; label ids with value_fn).
    """
    block = np.full((cell_px, cell_px), empty, dtype=np.int16)
    filled = np.zeros((cell_px, cell_px), dtype=bool)

    for _layer, obj in cell_objects:
        if obj.sprite is not None:
            # Render the 5×5 sprite scaled into cell_px × cell_px
            sprite_h = len(obj.sprite)
            sprite_w = max((len(row) for row in obj.sprite), default=0)
            if sprite_h == 0 or sprite_w == 0:
                continue

            # Centered nearest-neighbor sampling: when cell_px < sprite size,
            # floor-based mapping (pr * sprite_h // cell_px) drops the last
            # sprite row/col, turning a 5×5 box outline into an L-shape at
            # cell_px=4. Centered sampling (PIL's NEAREST formula) keeps
            # both edges.
            for pr in range(cell_px):
                sr = (pr * 2 + 1) * sprite_h // (2 * cell_px)
                row = obj.sprite[sr]
                for pc in range(cell_px):
                    sc = (pc * 2 + 1) * sprite_w // (2 * cell_px)
                    if sc >= len(row):
                        continue  # ragged sprite row — treat as transparent
                    color_idx = row[sc]
                    if color_idx == -1:
                        continue  # transparent pixel — show layer below
                    if value_fn is not None:
                        block[pr, pc] = value_fn(obj, cell_objects); filled[pr, pc] = True
                        continue
                    # Map sprite color index to ARC palette
                    if 0 <= color_idx < len(obj.colors):
                        arc_color = obj.colors[color_idx]
                    else:
                        arc_color = obj.dominant_color
                    if arc_color >= 0:  # skip transparent colors
                        block[pr, pc] = arc_color; filled[pr, pc] = True
        else:
            # No sprite — fill solid (only non-transparent)
            if value_fn is not None:
                block[:, :] = value_fn(obj, cell_objects); filled[:, :] = True
            elif obj.dominant_color >= 0:
                # Fill only pixels not yet drawn by a higher-priority sprite
                # (since we go bottom-to-top, we overwrite)
                block[:, :] = obj.dominant_color; filled[:, :] = True

    # Pixels no object drew -> `empty` (0/background by default).
    return np.where(filled, block, empty)


def _render_frame(engine: PSEngine, game: PSGame,
                  value_fn=None, empty: int = 0, pad: int = 5) -> np.ndarray:
    """Render current grid state to 64×64 ARC palette frame.

    Each cell is rendered with its actual 5×5 PuzzleScript sprite (scaled),
    compositing multiple objects from bottom layer to top with transparency.
    For grids larger than 64 in either dimension, renders at 1px per cell
    (using dominant color) then scales down via nearest-neighbor.
    Supports flickscreen: only renders the screen containing the player.

    value_fn/empty/pad: when value_fn is given, render a per-pixel LABEL map instead of the
    colour frame -- same geometry, flickscreen, HUD crop, scaling and layering -- with objects
    painted by value_fn(obj, cell_objects), unfilled play-area pixels = `empty`, and the
    letterbox border = `pad`. Returns int16 in that case (labels may be negative); default
    (value_fn=None, empty=0, pad=5) returns the byte-identical uint8 colour frame."""
    H, W = engine.height, engine.width
    grid = engine.grid

    if H == 0 or W == 0:
        return np.full((64, 64), pad, dtype=np.int16 if value_fn else np.uint8)

    # Flickscreen: render only the screen containing the player
    if game.flickscreen:
        scr_w, scr_h = game.flickscreen
        # Find the player position to centre the flickscreen window on. Most
        # games have exactly one player, but a few produce several: Savior's
        # "save state" mechanic clones the whole board — the player included —
        # into an off-screen mirror region tagged with the `to` object. If the
        # camera centres on whichever player is scanned first (top-left-most),
        # an ordinary move can flip the view onto the frozen clone and snap the
        # screen to that off-map region. Prefer the *live* player: the one not
        # co-located with a `to`-region marker. Falls back to first-found when
        # there is a single player (byte-identical to the old behaviour) or when
        # every player sits in a `to` cell.
        player_positions: list[tuple[int, int]] = []
        for r in range(H):
            for c in range(W):
                cell = engine.grid[r][c]
                for pi in engine._player_indices:
                    if pi in cell:
                        player_positions.append((r, c))
                        break
        pr, pc = 0, 0
        if player_positions:
            pr, pc = player_positions[0]
            if len(player_positions) > 1:
                to_idxs = set(game.resolve_object_name("to"))
                if to_idxs:
                    for (r, c) in player_positions:
                        if not (engine.grid[r][c] & to_idxs):
                            pr, pc = r, c
                            break
        sr = (pr // scr_h) * scr_h
        sc = (pc // scr_w) * scr_w
        er = min(sr + scr_h, H)
        ec = min(sc + scr_w, W)
        grid = [row[sc:ec] for row in engine.grid[sr:er]]
        H = er - sr
        W = ec - sc

    # Crop HUD rows (Board/Turn display objects) from edges of the grid.
    hud_indices = set()
    for group_name in ("board", "turn"):
        if group_name in game.or_groups:
            for obj_name in game.or_groups[group_name]:
                if obj_name in game.obj_name_to_idx:
                    hud_indices.add(game.obj_name_to_idx[obj_name])
    if "turn0" in game.obj_name_to_idx:
        hud_indices.add(game.obj_name_to_idx["turn0"])
    if hud_indices:
        crop_bottom = 0
        for r in range(H - 1, -1, -1):
            if all(grid[r][c] & hud_indices for c in range(W)):
                crop_bottom += 1
            else:
                break
        crop_top = 0
        for r in range(H):
            if all(grid[r][c] & hud_indices for c in range(W)):
                crop_top += 1
            else:
                break
        # A HUD is a thin status strip at the top/bottom edge. If the crop would
        # consume every row, the "hud" group is actually the playfield itself
        # (e.g. Rook Game's `Board = LightSpace or DarkSpace`, where every cell
        # is a board tile) — cropping it would blank the whole frame. Skip it.
        if crop_top + crop_bottom >= H:
            crop_top = crop_bottom = 0
        if crop_top > 0 or crop_bottom > 0:
            grid = grid[crop_top:H - crop_bottom]
            H = len(grid)

    # Compute cell pixel size — largest integer scaling that fits within 64×64
    cell_px = max(1, min(64 // H, 64 // W))

    rh, rw = H * cell_px, W * cell_px

    # If rendered size still exceeds 64 (very large grids), fall back to 1px/cell
    # and use PIL to downscale
    oversized = (rh > 64 or rw > 64)
    if oversized:
        cell_px = 1
        rh, rw = H, W

    # Build the rendered frame at (rh, rw)
    rendered = np.full((rh, rw), empty, dtype=np.int16)

    player_index_set = set(engine._player_indices)

    # Identify "shadow-band" border objects (PuzzleScript "3D effect" decals
    # like BorderD/BorderW/BPlayerB whose sprite is a solid top band). At the
    # ARC palette resolution they collapse to the same color as their parent
    # and visually extend it into the cell below — making cubes look 1.4×
    # taller and players look like they overlap a cube above. Decorative
    # 'B' siblings (CrateB/TargetB/FireB/...) have sparse top rows and are
    # kept since they complete a multi-cell glyph.
    shadow_indices: set[int] = set()
    if "border" in game.or_groups:
        for obj_name in game.or_groups["border"]:
            idx = game.obj_name_to_idx.get(obj_name)
            if idx is None:
                continue
            obj = game.objects.get(obj_name)
            if obj is None or obj.sprite is None or not obj.sprite[0]:
                continue
            top_row = obj.sprite[0]
            if len(top_row) >= 5 and all(c >= 0 for c in top_row[:5]):
                shadow_indices.add(idx)

    for r in range(H):
        for c in range(W):
            # Collect objects in this cell, separating player from non-player
            cell_objs = []
            player_objs = []
            for obj_idx in grid[r][c]:
                if obj_idx in shadow_indices:
                    continue
                obj_name = game.obj_idx_to_name.get(obj_idx, "")
                obj = game.objects.get(obj_name)
                if obj is None:
                    continue
                layer = engine._obj_layers.get(obj_idx, 0)
                if obj_idx in player_index_set:
                    player_objs.append((layer, obj))
                else:
                    cell_objs.append((layer, obj))

            if not cell_objs and not player_objs:
                # Empty cell — background
                rendered[r * cell_px:(r + 1) * cell_px,
                         c * cell_px:(c + 1) * cell_px] = empty
                continue

            # Sort non-player objects by layer (bottom first), then append
            # player objects last so the player is always visible on top
            cell_objs.sort(key=lambda x: x[0])
            player_objs.sort(key=lambda x: x[0])
            final_objs = cell_objs + player_objs

            if cell_px == 1:
                # Single pixel — use top (last) object's value
                top_obj = final_objs[-1][1]
                if value_fn is not None:
                    rendered[r, c] = value_fn(top_obj, final_objs)
                else:
                    rendered[r, c] = top_obj.dominant_color if top_obj.dominant_color >= 0 else empty
            else:
                # Render cell sprite
                block = _render_cell_sprite(final_objs, cell_px, value_fn, empty)
                rendered[r * cell_px:(r + 1) * cell_px,
                         c * cell_px:(c + 1) * cell_px] = block

    # For oversized grids, scale down to fit 64×64
    # NEAREST-neighbour resize preserves exact label ids. PIL needs uint8; label ids are small
    # non-negative (callers offset any negative sentinel before rendering), so the round-trip is
    # lossless. int16 kept so the caller can post-map (e.g. shift a colour sentinel back to -1).
    def _resize(arr, new_w, new_h):
        img = Image.fromarray(arr.astype(np.uint8), mode="L").resize((new_w, new_h), Image.NEAREST)
        return np.asarray(img, dtype=np.int16)

    if oversized:
        scale = 64 / max(rh, rw)
        new_w = max(1, int(rw * scale)); new_h = max(1, int(rh * scale))
        rendered = _resize(rendered, new_w, new_h); rh, rw = new_h, new_w
    elif rh < 64 and rw < 64:
        # Upscale (nearest-neighbor) so the play area fills more of the 64×64
        # frame instead of sitting in a small black-bordered patch.
        scale = min(64 / rh, 64 / rw)
        new_w = min(64, max(1, int(rw * scale))); new_h = min(64, max(1, int(rh * scale)))
        if new_w != rw or new_h != rh:
            rendered = _resize(rendered, new_w, new_h); rh, rw = new_h, new_w

    # Center-pad to 64×64 with the border value (letterbox palette 5 by default).
    pad_r = (64 - rh) // 2
    pad_c = (64 - rw) // 2
    frame = np.full((64, 64), pad, dtype=np.int16)
    frame[pad_r:pad_r + rh, pad_c:pad_c + rw] = rendered

    return frame if value_fn is not None else frame.astype(np.uint8)


# ---------------------------------------------------------------------------
# Game data directory
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "puzzlescript_games")


# ---------------------------------------------------------------------------
# Available games (auto-discovered from data/puzzlescript_games/*.txt)
# ---------------------------------------------------------------------------

def _discover_puzzlescript_games() -> tuple[str, ...]:
    """Scan the data directory for available PuzzleScript game files."""
    if not os.path.isdir(_DATA_DIR):
        return ()
    games = []
    for f in sorted(os.listdir(_DATA_DIR)):
        if f.endswith(".txt"):
            games.append(f[:-4])  # strip .txt extension
    return tuple(games)


_AVAILABLE_PUZZLESCRIPT_GAMES: tuple[str, ...] = _discover_puzzlescript_games()


# GameAction → direction string
_ACTION_MAP: dict[GameAction, str] = {
    GameAction.ACTION1: "up",
    GameAction.ACTION2: "down",
    GameAction.ACTION3: "left",
    GameAction.ACTION4: "right",
    GameAction.ACTION5: "action",
}


# ---------------------------------------------------------------------------
# PuzzleScriptAdapter
# ---------------------------------------------------------------------------

class PuzzleScriptAdapter(BaseAdapter):
    """Wraps a PuzzleScript game as an ARCBaseGame-compatible object.

    Args:
        game_name: Name of the game (matches filename without .txt extension
                   in data/puzzlescript_games/).
        seed:      Base random seed for level selection.
    """

    def __init__(self, game_name: str, seed: int = 0) -> None:
        self._game_name = game_name
        self._seed = seed
        self._game_id = f"puzzlescript_{game_name.lower().replace(' ', '_').replace('-', '_')}"

        # Load and parse game file
        game_path = os.path.join(_DATA_DIR, game_name + ".txt")
        if not os.path.exists(game_path):
            raise FileNotFoundError(f"PuzzleScript game not found: {game_path}")

        with open(game_path, "r", encoding="utf-8") as f:
            game_text = f.read()

        self._game = PSGame(game_text)
        self._engine = PSEngine(self._game)

        self._max_steps: int = 200  # reasonable default for puzzle games

        # Shared adapter scaffolding (state, step counter, undo stack). The first
        # frame is produced by set_level → _do_reset in the tail of __init__.
        self._init_base(self._game_id, max_steps=self._max_steps,
                        available_actions=[1, 2, 3, 4, 5])

        # Rotation augmentation
        self._rotation_k: int = 0

        # Horizontal-flip augmentation (see _HFLIP_GAMES / _do_reset). A
        # left-right mirror of the presented frame; for the gravity games it
        # applies to, this is chosen because it preserves the vertical axis (so
        # gravity still points down and the ball still rests on the floor below
        # it). Combined with the native rotation it spans the full 8-element
        # symmetry group, including the vertically-flipped (mirror) maze.
        self._hflip: bool = False

        # Vertical-flip augmentation (see _FLIP_GAMES / _do_reset). A top-bottom
        # mirror (np.flipud) of the presented frame. Only enabled for games with
        # no axis-sensitive mechanic (no gravity) whose directional inputs are
        # screen-relative moves, so the flip is remapped through the action path
        # (up ↔ down) and stays an exact symmetry. Off (False) for every other
        # game — in particular the gravity games that use _hflip, where a
        # vertical flip would invert gravity.
        self._vflip: bool = False

        # Object indices for the "playing" animation states (botsket_ball-style
        # play-button games). When any of these appear on the grid mid-turn,
        # the entire end-of-episode animation is drained inside one
        # perform_action call and the result becomes WIN or GAME_OVER.
        self._playing_indices: list[int] = []
        for name in ("playl", "playu", "playr", "playd"):
            if name in self._game.obj_name_to_idx:
                self._playing_indices.append(self._game.obj_name_to_idx[name])

        # UI-cycle indices that always change every tick (speed slider marker,
        # play-button cycle states). Excluded from the "is the game state
        # stable?" check used to terminate the play animation early.
        self._ui_cycle_indices: set[int] = set(self._playing_indices)
        for name in ("marker",):
            if name in self._game.obj_name_to_idx:
                self._ui_cycle_indices.add(self._game.obj_name_to_idx[name])

        # Chrome recolor augmentation. Precompute (once, from the original colors,
        # before any reset mutates them) a plan of the surfaces to recolor and the
        # palette each may take. Every surface excludes the black frame border
        # (palette 5), its own original color (so the recolor is always a real
        # change) and any per-surface "reserved" colors. At reset the surfaces
        # are colored in order, each additionally excluding the colors already
        # assigned this reset, so all surfaces end up mutually distinct. Each
        # surface's objects are flattened to one flat color.
        self._bg_recolor_idx: Optional[int] = None
        self._wall_recolor_idx: Optional[int] = None
        self._ball_recolor_idx: Optional[int] = None
        self._goal_recolor_idx: Optional[int] = None
        self._selector_recolor_idx: Optional[int] = None
        self._cursor_recolor_idx: Optional[int] = None
        self._recolor_plan: list[dict] = []
        if game_name in self._RECOLOR_GAMES:
            for surf in self._recolor_surfaces():
                # Surfaces are (key, objects, reserved[, spec]). ``spec`` is
                # optional; when present with mode "remap" the surface maps a
                # single original color to a new one across possibly-multicolor
                # objects (preserving their other pixels), rather than flattening
                # every object to one flat color.
                key, objs, reserved = surf[0], surf[1], surf[2]
                spec = surf[3] if len(surf) > 3 else None
                if not objs:
                    continue
                if spec and spec.get("mode") == "remap":
                    # Only the class's original color is being replaced, so it
                    # (not the object's full palette) is the "own" color to avoid
                    # picking again.
                    own = {spec["old"]}
                else:
                    own = {c for o in objs for c in o.colors if c >= 0}
                allowed = [i for i in range(len(ARC_PALETTE_RGB))
                           if i != 5 and i not in own and i not in reserved]
                entry = {"key": key, "objs": objs, "allowed": allowed,
                         "mode": "flatten"}
                if spec and spec.get("mode") == "remap":
                    entry["mode"] = "remap"
                    entry["old"] = spec["old"]
                self._recolor_plan.append(entry)

        # Snapshot each recolored object's ORIGINAL colors so every reset
        # re-applies the augmentation from a clean base. This matters for
        # "remap" surfaces: they replace a specific original color, which a
        # previous reset would already have overwritten — restoring first keeps
        # the remap idempotent across resets (and is a harmless no-op for
        # flatten). Dedup by object identity, since an object (e.g. the
        # two-tone annihilation flash) can belong to more than one surface.
        self._recolor_originals: list = []
        _seen_ids: set = set()
        for surf in self._recolor_plan:
            for obj in surf["objs"]:
                if id(obj) not in _seen_ids:
                    _seen_ids.add(id(obj))
                    self._recolor_originals.append(
                        (obj, list(obj.colors), obj.dominant_color))

        self._do_reset()

    def _recolor_surfaces(self) -> list[tuple]:
        """Per-game list of chrome surfaces to recolor, as
        ``(key, objects, reserved_colors)`` tuples. ``key`` names the surface
        (also the ``_{key}_recolor_idx`` attribute set at reset); ``objects`` is
        the PSObjects flattened to the chosen color; ``reserved_colors`` are
        palette indices this surface must additionally avoid (on top of the
        frame border and its own original color)."""
        def _objs(names):
            return [self._game.objects[n] for n in names
                    if n in self._game.objects]
        g = self._game

        if self._game_name == "Everything_Antimatters":
            # Chrome surfaces (each FLATTENED to one random color):
            #   * "wall"   — the Wall frame bordering every level (burgundy).
            #   * "bg"     — the Background field (light blue).
            #   * "cursor" — the Cursor square the player moves (white). Also
            #     recolors CursorStop, the object the Cursor briefly becomes on
            #     the action key (rule ``[action cursor] -> [cursorStop ...]``),
            #     so the controllable square keeps its randomized color on
            #     ACTION5 instead of snapping back to burgundy.
            #
            # Block color CLASSES (each REMAPPED — one original color → one new
            # color across every object that uses it): the matter/antimatter
            # crates and the "sparks" the player generates share exactly two
            # source colors — pink (Positive/PositiveAux/TargetP/PFAnim…) and
            # purple (Negative/NegativeAux/TargetN/NFAnim…). We remap per color
            # class, not per object, so ALL pink pieces recolor together to one
            # color and ALL purple pieces to another — a consistent palette swap
            # that preserves the block↔spark↔target visual grouping (and the
            # win condition, which matches by object index). The annihilation
            # flashes Anim01/Anim02 carry BOTH source colors; because we remap
            # (not flatten) they keep their two-tone look, each pixel following
            # its own class.
            #
            # All five surfaces are colored from a shared "already used" set at
            # reset (see _do_reset), so every final color is mutually distinct —
            # no crate can vanish into the floor/wall/cursor and the two matter
            # types stay visually separable. Each block class additionally
            # reserves BOTH original block colors so a remapped class never
            # lands on the other class's source color (which, applied in
            # sequence over the shared Anim objects, would corrupt the two-tone).
            chrome_names = {"background", "wall", "cursor", "cursorstop", "pit"}
            bg_objs = _objs(["background"])
            wall_objs = _objs(["wall"])
            cursor_objs = _objs(["cursor", "cursorstop"])
            block_objs = [o for n, o in g.objects.items()
                          if n not in chrome_names]
            block_colors = sorted({c for o in block_objs
                                   for c in o.colors if c >= 0})
            surfaces = [
                ("bg", bg_objs, set(), None),
                ("wall", wall_objs, set(), None),
                ("cursor", cursor_objs, set(), None),
            ]
            for col in block_colors:
                members = [o for o in block_objs if col in o.colors]
                surfaces.append(
                    (f"block_{col}", members, set(block_colors),
                     {"mode": "remap", "old": col}))
            return surfaces

        if self._game_name == "A_Knight's_Tour":
            # Every surface here is FLATTENED to one random color, and the shared
            # "already used" set at reset keeps all eight mutually distinct. That
            # distinctness is not just variety — it is what makes the game
            # observable at all:
            #
            #   * Player is `red` and Playeroption is `lightred`, and BOTH map to
            #     ARC palette 8. The cursor's sprite (a square broken at the four
            #     edge midpoints) is a strict SUBSET of the highlight's (a solid
            #     square), and Player is the topmost collision layer — so a cursor
            #     standing ON a legal knight destination rendered pixel-identical
            #     to that destination with no cursor on it. The one state where
            #     the ACTION key matters was the one state you could not see.
            #   * Knight, Hole and Background3 are all `black`, i.e. the hole cell
            #     (Hole over Background3) was a uniform black block and the knight
            #     was black on gray. Splitting hole from knight keeps a captured
            #     square from reading as a piece.
            #
            # The two checkerboard shades are separate surfaces so the board keeps
            # its light/dark pattern instead of flattening to one field, and the
            # pawns and rooks are separate surfaces because they play completely
            # differently (objective vs. hazard) despite both shipping `white`.
            # Colors never affect the engine — every rule and the win condition
            # match by object index — so this is a pure relabel and any plan is
            # byte-for-byte unaffected.
            return [
                ("bg", _objs(["background1"]), set()),
                ("bg2", _objs(["background2"]), set()),
                ("hole", _objs(["background3", "hole"]), set()),
                ("knight", _objs(["knight"]), set()),
                ("pawn", _objs(["pawn"]), set()),
                ("rook", _objs(g.or_groups.get("rook", [])), set()),
                ("option", _objs(g.or_groups.get("option", [])), set()),
                ("cursor", _objs(["player"]), set()),
            ]

        if self._game_name == "Enqueue":
            surfaces = [
                # The section backdrop is a grid-like pattern of Wall glyphs over
                # the Background: recolor both. "bg" also paints the solid section
                # separator band, which is the same Background object.
                ("bg", _objs(["background"]), set()),
                ("wall", _objs(["wall"]), set()),
                # The selection cursor (hollow square marking the active slot).
                ("selector", _objs(["selector"]), set()),
            ]
            # Puzzle content: each COLOR CLASS is a Block and its matching Pad
            # (RedBlock ↔ RedPad, …). Recolor the pair as ONE unit, so every
            # object of a class takes the SAME new color and the block↔pad match
            # stays visually readable. The engine matches by object index, not
            # color (rule ``[RedBlock RedPad] -> [RedBlock]``), so this is a pure
            # relabel — the solution / A* plan is byte-for-byte identical. Colors
            # are kept distinct across classes (and vs the chrome above) by the
            # reset-time "already used" exclusion, so no two classes ever collapse
            # to one color (which would make the board visually ambiguous).
            for bname in g.or_groups.get("block", []):
                if not bname.endswith("block"):
                    continue
                prefix = bname[:-len("block")]
                objs = _objs([bname, prefix + "pad"])
                if objs:
                    surfaces.append((f"content_{prefix}", objs, set()))
            return surfaces

        # Drop_Maze (default): background floor, walls (+ the wall-colored
        # player pixel), ball and goal cell.
        wall_names = list(g.or_groups.get("wall", []))
        if "player" in g.objects:
            # The Player is a single pixel drawn in the wall color; recolor
            # it with the walls so it stays blended into the border.
            wall_names.append("player")
        return [
            ("bg", _objs(["background"]), set()),
            ("wall", _objs(wall_names), set()),
            ("ball", _objs(g.or_groups.get("ball", [])), set()),
            ("goal", _objs(["target"]), set()),
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    # Games whose mechanics are inherently axis-sensitive (gravity, row-only
    # slides, vertical-only stacking). Rotating the frame would map the
    # player's intuitive arrow press to a perpendicular direction in the
    # engine and break the mental model — e.g. Coin Dropper's row-rotation
    # cue is the side-flanking ``+`` indicator, which only makes sense in
    # the unrotated frame.
    _NO_ROTATION_GAMES = frozenset({"Coin_Dropper", "Crate_Rotate"})

    # Games where directional inputs encode an operation on the board itself
    # (e.g. rotate-the-maze) rather than a screen-relative move. Action keys
    # must pass through unchanged regardless of frame rotation so left/right
    # consistently mean CCW/CW rotation of the board.
    _NO_ACTION_REMAP_GAMES = frozenset({"Drop_Maze"})

    # Games that already implement rotation as a native in-engine mechanic, so
    # they must NOT get the np.rot90 frame-rotation augmentation (it would rotate
    # an already-correctly-oriented view and desync the mechanic). Drop_Maze is
    # the canonical case: each level is a 2×2 block of quadrants holding four
    # pre-rotated copies of the maze (with balls D / L / R / U kept in sync by
    # the Mark cycle rules); pressing left/right walks the Player to an adjacent
    # quadrant and the flickscreen shows that rotation. We augment the *initial*
    # orientation instead — by applying a random number of native rotate-steps at
    # reset (see _do_reset), which yields the 4 valid rotations for free and
    # keeps gravity and the ball's resting position consistent.
    _NATIVE_ROTATION_GAMES = frozenset({"Drop_Maze"})

    # The engine direction that performs one native rotation step for each
    # native-rotation game (any consistent choice works — it just walks the
    # Player one quadrant around the border).
    _NATIVE_ROTATION_DIR = {"Drop_Maze": "left"}

    # Rotation-start phases that leave a specific (game, level) unsolvable and
    # so must be excluded from the native-rotation augmentation. Drop_Maze's
    # gravity is irreversible, so rotating the board before play can trap the
    # ball with no way back to its target; on level 1, starting phase 3 (three
    # native rotate-steps) is such a dead end. Excluding it keeps every
    # generated start winnable. Discovered empirically with the A* expert (see
    # solvers/generate_drop_maze_training.py). Keyed by game -> {level_index:
    # {forbidden phase, ...}}. Levels not listed allow all four phases (0-3);
    # for those, choice([0,1,2,3]) draws identically to the old randint(0,3),
    # so their rotation-starts are unchanged.
    _UNSOLVABLE_NATIVE_ROTATIONS = {
        "Drop_Maze": {1: {3}},
    }

    # Games that additionally get a horizontal-flip augmentation. Restricted to
    # a LEFT-RIGHT mirror (np.fliplr): for gravity games it preserves the
    # vertical axis, so gravity stays "down" and the ball keeps resting on the
    # floor — unlike a vertical flip (np.flipud), which would invert gravity and
    # leave the ball floating against a ceiling. Because the native rotation
    # already supplies all four rotations (each with gravity down), the h-flip
    # combined with them spans the whole dihedral group, so the vertically
    # mirrored maze is covered too — with correct downward gravity. This is a
    # pure observation mirror: only the rendering is flipped, the engine and the
    # action mapping are untouched, so every augmented episode is an exact
    # pixel-mirror of a real one under the same action sequence.
    _HFLIP_GAMES = frozenset({"Drop_Maze"})

    # Games that get INDEPENDENT horizontal AND vertical flip augmentations, each
    # remapped through the directional action path (unlike _HFLIP_GAMES, whose
    # flip is a pure observation mirror on games that don't remap actions). This
    # is valid only for games with no axis-sensitive mechanic (no gravity, no
    # up-only stacking) whose directional inputs are screen-relative moves: there
    # a flip is a true symmetry — mirror the frame, swap the corresponding axis
    # of the input, and the engine plays out an exact reflection of a real
    # trajectory. Enqueue (queue-walking + push, gravity-free) qualifies. Add Man
    # 2 (sokoban + binary arithmetic) qualifies too: its ONE directional rule is
    # the carry, which acts on the BOARD ("the cell to the left of a Two"), not on
    # the input, so a mirror just presents a board whose carry runs the other way
    # — exactly what the rotation augmentation already does — and both digit
    # sprites (a ring and a bar) are mirror-symmetric, so the 0-vs-1 distinction
    # survives. Aerobatics qualifies as well: the plane flies over a walled course
    # with no gravity and no stacking, and its rules are stated in terms of the
    # plane's own heading vs the pressed direction, so mirroring the board and the
    # matching input axis reflects a real flight exactly. Its art is mirror-closed
    # too — flipud(PlaneU) is pixel-identical to PlaneD, fliplr(PlaneL) to PlaneR,
    # and likewise for the diagonals and the WallUL/UR/DL/DR edge variants — so a
    # flipped frame is one the game could itself have produced. It gets no color
    # augmentation, so the flips are what stop its seeds differing by nothing but
    # one of four rotations. Drawn per (seed, level) from private RNG streams, so
    # the four rotations × 2 hflip × 2 vflip re-sample the board's 8-element
    # symmetry group. Bad Example is the plainest case of all: a textbook Sokoban
    # whose single rule is direction-agnostic, so every reflection of a trajectory
    # is itself a legal trajectory and no sprite encodes a direction. It has three
    # levels and no color augmentation, so the flips are the difference between 12
    # and 24 distinct presentations across the whole corpus.
    # Baguettes qualifies as well: rigid-piece sokoban, gravity-free, screen-
    # relative moves. Its ruleset is mirror-closed -- bagleft/bagright and the
    # `late right`/`late left` bake rules are exact mirror images of each other,
    # as are bagup/bagdown and `late up`/`late down`, and so are the target
    # classes they have to be matched against -- so a mirrored board is one this
    # game could really hand out. Its art is mirror-closed in ARC palette space,
    # which is the space that matters here: every sprite flattens to a single
    # colour plus transparent CORNER pixels, and fliplr of bagleft's two
    # left-corner holes is exactly bagright's two right-corner holes (likewise
    # targetleft/targetright, and flipud for bagup/bagdown and
    # targetup/targetdown), while the mids, the balls, the oven, the dough and
    # the player are corner-symmetric already. It gets no color augmentation --
    # the recolor surfaces flatten a sprite to one flat colour, which would erase
    # the corner code that is the ONLY bagleft-vs-bagright cue -- so as with
    # Aerobatics the flips are what stop its seeds differing by nothing but one
    # of four rotations.
    # Bichrome is the Bad Example case again: gravity-free, screen-relative moves,
    # and its two movement rules ([> Player | wall of my colour] -> push, and the
    # push/swap resolution) name no direction at all, so every reflection of a
    # trajectory is itself a legal trajectory. No sprite encodes a direction and
    # every one of them is mirror-symmetric (its "active character" marker sits on
    # all four cell corners precisely so it survives both the 4 px downsample and
    # the flips). It gets no color augmentation -- orange-versus-blue IS the
    # mechanic there -- so without the flips its twelve levels would differ by
    # nothing but one of four rotations.
    # Blind Maze a1 is the Bad Example case once more: gravity-free, screen-relative
    # moves, and every one of its movement rules ships as a matched left/right +
    # up/down quartet (the wall-bump wrap and the slide loop), so a reflection of a
    # trajectory is itself a legal trajectory. The one asymmetric sprite is the
    # flag, whose orientation carries no meaning (it is "a flag", not a directional
    # marker), and the wall-edge shading the rotation augmentation already permutes
    # anyway. It gets no colour augmentation -- red-versus-white IS the mechanic --
    # and its boards, flag placements and player start are drawn per (seed, level),
    # so the flips are pure extra presentation diversity on top of that.
    # Block Faker is the Bad Example case yet again: gravity-free, screen-relative
    # moves, and not one of its four rules names a direction -- the chain push and
    # the grille cancel are written with a bare `>` (every direction), and each
    # match-3 rule is an unprefixed three-in-a-line that already expands to both
    # axes. So a reflected trajectory is one the game could really hand out, and a
    # reflected board is one it could really ship. No sprite encodes a direction;
    # the only asymmetric one is WallBlock (a black edge along one side), which is
    # decoration -- it is mechanically identical to Wall -- and whose orientation
    # the rotation augmentation already permutes. It gets no colour augmentation
    # (which colour a block is IS the mechanic), so with only five levels the flips
    # are the difference between 20 and 40 presentations across the whole corpus.
    # Box Fill is the cleanest instance of the same argument: gravity-free,
    # screen-relative moves, and every one of its five rules is written with a
    # bare `>` or is direction-free (the two spawn rules, the movement cancel,
    # the TempBox promotion, the gate latch), so no rule and no win condition can
    # tell a reflected trajectory from an original one. Nothing on the board has
    # a sprite at all -- every object is a solid colour block -- so there is no
    # orientation to preserve. It gets no colour augmentation, and with eight
    # levels the flips take the corpus from 32 presentations to 128.
    # Boupha's Candle Quest is isotropic in the strongest sense available: not
    # one of its rules carries an explicit up/down/left/right, so the push, the
    # pull, the qumpkin swap, the zombies' line-of-sight homing and both door
    # latches all expand over the four directions symmetrically, and its moves
    # are screen-relative with no gravity. The only asymmetric sprites (the
    # player's head-over-legs, the candle's flame) are decoration, and the
    # rotation augmentation already permutes their orientation. It gets no
    # colour augmentation -- eighteen object classes share a small palette, and
    # Wall/Pumpkin and Switch/Door are already colour-twins told apart by sprite
    # shape alone -- so over twelve levels the flips are the difference between
    # 48 presentations and 96.
    # Boolean Bloom is the one case here whose rules are FULL of explicit
    # directions -- and it still reflects exactly, because those directions come
    # in mirrored pairs that the flip maps onto each other. Every rule naming
    # `right` has a `left` twin and every `up` a `down` one (the four retractions,
    # the four goal-priority growths, the four plain growths), the objects they
    # name pair off the same way (PlantHeadRight <-> PlantHeadLeft, PlayerRight
    # <-> PlayerLeft), and the rest of the game -- the chain push, the four
    # button gates, the goal tick, the win condition -- names no direction at
    # all. So a horizontally reflected trajectory is one the game could really
    # hand out, with each plant playing its mirror twin's part. The sprites make
    # that literal rather than merely mechanical: PlantHeadRight's sprite mirrors
    # pixel-for-pixel onto PlantHeadLeft's and PlayerRight's onto PlayerLeft's,
    # and the up/down heads are each self-mirror once you notice their palette
    # lists #3C2E47 twice, so indices 4 and 5 are the same colour. The vertical
    # flip is the same argument with the axes swapped. The only sprites that do
    # not land on a shipped sprite are the two stems, whose leafy vine is
    # decoration -- horizontal stays horizontal under either flip, so no
    # mechanical class is ever confused. No colour augmentation, and with nine
    # levels the flips take the corpus from 36 presentations to 144.
    # Bridge is as clean a case as Bad Example: sokoban whose crates double as
    # bridge planks, gravity-free, screen-relative moves, and all THREE of its
    # rules are written with the relative `>` ("the cell the mover is heading
    # into"), so not one of them names a compass direction and every reflection
    # of a trajectory is itself a legal trajectory. Its art is mirror-closed in
    # ARC palette space: wall, background, lava and the bridge tile are solid
    # single-colour squares, the pad is a symmetric X, and the crate's and the
    # player's transparent holes sit on the two diagonals, so a flip maps each
    # onto the other's motif — in the other object's COLOUR, which is what tells
    # them apart, so no mechanical class is ever confused with another. Four
    # levels, no colour augmentation, so the flips are the difference between 16
    # and 64 presentations.
    # Brendan loves mondays is the Bad Example case with a colour lock on top:
    # gravity-free sokoban, screen-relative moves, and every rule stated with the
    # relative `>` force ("the thing I am moving into"), including the mood flips,
    # which name no direction at all. So a reflected trajectory is one the game
    # could really hand out. Nothing in its art encodes a direction either — the
    # Brendans are one stick figure re-coloured, the balls are discs, and the key,
    # bottle, lock and bed are asymmetric but static scenery whose mirror lands on
    # no other object's sprite, so no mechanical class is ever confused. Ten
    # levels and no colour augmentation, so the flips take it from 40
    # presentations to 160.
    # Broken Maze is the rare case where the reflection is an EXACT symmetry of
    # the rule file rather than merely a plausible one. Its two rules bind the
    # four walkers by compass name -- horizontally NE and SE mirror NW while SW
    # copies it, vertically NE copies while SE and SW mirror -- and a reflection
    # relabels those names in exactly the pairs that leave both rules fixed:
    # hflip swaps NW<->NE and SW<->SE, which after negating the horizontal axis
    # maps `[> NW][< NE][< SE][> SW]` onto itself and leaves the vertical rule's
    # `[> NW][> NE][< SE][< SW]` untouched; vflip is the same argument with the
    # axes exchanged. The crate push and the win condition name no direction at
    # all. So a flipped frame is a board this game could really ship, with
    # another quadrant playing the orange reference part -- which the rotation
    # augmentation already does. Every sprite is mirror-symmetric (the walkers
    # are squares with the four corners cut, the crate is a ring, everything else
    # is a solid block), so nothing lands on another class's art. No colour
    # augmentation -- orange-versus-grey is what marks the reference walker -- so
    # over eight levels the flips take the corpus from 32 presentations to 128.
    # Bubblegoban is the strongest form of the Bad Example argument: not one of
    # its rules names a direction. The two sticky rules and the
    # `[ Moving Gum | Wall ] -> cancel` trap are stated without any prefix, so
    # they already fire in all four directions on every turn; the push rule and
    # the player-into-wall cancel use the relative `>` force; and the win
    # condition (`All Target on Player`) is positional. There is no gravity and
    # input is screen-relative, so a reflected board is one the game could really
    # ship. No sprite lands on another class's art under a mirror -- gum, box and
    # target are symmetric rings/blocks, the wall is a symmetric brick, and the
    # player is the only asymmetric one (head over legs), which mirrors onto
    # nothing else. Seven levels and no colour augmentation, so the flips take
    # the corpus from 28 presentations to 112.
    # Cake Monsters is the same argument again: not one rule names a direction
    # (every one of them is stated with the relative `>` force), the win is
    # `No Cake` -- positional in no way at all -- there is no gravity, and input
    # is screen-relative. So a reflected board is one the game could really ship.
    # No sprite lands on another class's art under a mirror either: the cake,
    # the wall and the destroyer are symmetric left-to-right, and the monster is
    # the only chiral one (a black eye on the left, a white one on the right),
    # which mirrors onto nothing else -- and the rotation augmentation already
    # moves those two pixels around all four sides. 36 levels and no colour
    # augmentation, so the flips take the corpus from 144 presentations to 576.
    # Candy Bomb splits cleanly the same way: the push chain and the two
    # light-the-fuse rules are all stated with the relative `>` force, and every
    # rule that matters for the blast -- the player's death, the candy shatter,
    # the fuse chaining onto a neighbouring bomb -- is written without a prefix,
    # so it already fires on all four sides every turn. The one three-cell
    # pattern, `[ Explosion | HardCandy | Explosion ]`, is its own mirror. The
    # win is `no WinCandy` / `no PlayerDead`, positional in no way; there is no
    # gravity and input is screen-relative. Under a mirror no sprite lands on
    # another class's art: the candies, the explosion and the bomb bodies are
    # symmetric, and the two chiral marks -- the bomb's fuse trailing off one
    # corner and the shatter's debris -- mirror onto nothing else (and the
    # rotation augmentation already swings the fuse around all four corners).
    # 14 levels and no colour augmentation, so the flips take the corpus from 56
    # presentations to 224.
    # Cancel's five rules are all stated over RELATIVE directions (`>` push,
    # `<` pull, the swap, and the `v`/`^` pair that drags an air), and that pair
    # is what makes the mirror exact: the two rules cover the two perpendiculars
    # symmetrically, so reflecting the board maps each onto the other rather
    # than onto nothing. The two late rules name no direction at all, the win is
    # `No Earth / No Water / No Fire / No Air` -- positional in no way -- there
    # is no gravity and input is screen-relative. Every object is a plain
    # coloured block with no sprite art, so a mirror cannot land one class's art
    # on another's. 8 levels and no colour augmentation (the four elements are
    # told apart by colour and nothing else, so a flattening recolor could only
    # take information away), so the flips take the corpus from 32 presentations
    # to 128.
    #
    # Circuit_Breaker is the one entry here whose sprites are DIRECTIONAL, and it
    # is still exact: a connector is a green stub drawn from the middle of a crate
    # out to one wall of its cell, and it wins by meeting the opposite stub across
    # that wall. A mirror moves the stub art with the board, so a pair that met
    # still meets -- and the U/D and L/R rules are each other's transpose, so the
    # mechanic maps onto itself. No gravity, screen-relative input.
    #
    # Colour Combination qualifies on the Cake Monsters grounds: all fifteen
    # rules are stated with the relative `>` force (the push, the seven target
    # fills and the twelve mixes), the win is `No ColoredTargets` -- positional
    # in no way -- there is no gravity and input is screen-relative. Under the
    # ARC adaptation of its art (see the note at the top of the .txt) no sprite
    # can land on another class's under a mirror either: a target is the cell
    # border, a crate the solid interior and the player a diamond, all three
    # symmetric on both axes, and the wall's brick texture is the only chiral
    # sprite -- it mirrors onto nothing else. 5 levels and no colour
    # augmentation (which crate mixes with which IS the puzzle, so a recolor
    # would break the R/G/B arithmetic the frames teach), so the flips take the
    # corpus from 20 presentations to 80.
    #
    # Collect Gnocchi is the one entry whose whole board is DIRECTIONAL art and
    # is still exact, because that art is a closed orbit: every cell the player
    # has walked keeps a To mark whose four variants (a filled centre with an arm
    # out one side) are exact rotations and mirrors of each other, and the From
    # marks are one direction-free ring. A mirror therefore maps each trail mark
    # onto the variant that direction really becomes on screen, never onto
    # another class. Everything else is relative or side-agnostic: the push, the
    # magic slide and the trail are `>` rules, the herby golems spawn
    # PERPENDICULAR (both sides at once, so the pattern is its own mirror), and
    # the eat / poison / spicy-blast / key / button rules name no direction. The
    # win is `no Gnocchi`, positional in no way; no gravity, screen-relative
    # input. The only absolute-direction rules build and left-pack the stomach
    # HUD, which is a strip of three cells the player can never reach -- it just
    # renders at the mirrored corner. 19 levels and no colour augmentation
    # (which gnocchi is which IS the puzzle), so the flips take the corpus from
    # 76 presentations to 304.
    # Dang I'm Huge is gravity-free with screen-relative input, and its win
    # (`all Target on Pushable`) is positional in no way. It DOES have
    # absolute-direction rules, but every one of them -- the growth rules and the
    # eight rigid propagation rules -- only asserts the internal geometry of a
    # 2x2 body (quadrant 1 is top-left of 2, 3 is below 1, ...), which a mirror
    # carries with the body it is drawn on. Its chiral sprites are the quadrants,
    # and under a mirror each maps INSIDE its own class or onto nothing:
    # Player1<->Player2 and Player3<->Player4 under a horizontal flip, the pads
    # are symmetric squares, and the crate quadrants' asymmetry is a shading
    # highlight that mirrors to art no rule produces but that is still plainly a
    # blue 2x2 crate. No mirror can land one object's art on another's. 8 levels
    # and no colour augmentation (crate blue vs target purple vs armed/disarmed
    # pad is exactly what the frames have to teach), so the flips take the corpus
    # from 32 presentations to 128.
    # Count Mover qualifies on exactly the Bad Example grounds -- it is the same
    # textbook Sokoban: gravity-free, screen-relative input, one relative-force
    # rule, an `All Crate on Goal` win that is positional in no way, and not one
    # chiral sprite (crate, goal and player are all symmetric on both axes), so
    # no mirror can land one object's art on another's. 13 levels and no colour
    # augmentation (crate-on-goal is read as the goal ring showing around the
    # crate, which a flattening recolor would erase), so the flips take the
    # corpus from 52 presentations to 208.
    # Darkness Sokoban qualifies on the same grounds and needs it most: the same
    # textbook Sokoban (gravity-free, screen-relative input, every rule stated
    # over the relative force `>`, an `All Target on Crate` win that is
    # positional in no way) plus a lighting layer whose rules are the same
    # relative-adjacency pattern expanded over all four directions, and no
    # object's mirror image is another object's art. It ships only 2 levels, so
    # the flips are what take its corpus from 8 presentations to 32.
    # Dotsnake is the Boolean Bloom case: its rules are full of absolute
    # directions and it still reflects EXACTLY, because those directions come in
    # mirrored sets that the flip maps onto each other, and so do the objects
    # they name. The four head rules ([UP Player] -> [Up HeadUp], ...) pair off
    # under each flip exactly as HeadUp/HeadDown and HeadLeft/HeadRight do, and
    # the eight tail-drawing rules are one orbit: "tail connecting from above +
    # leaving left -> the up-left corner piece" mirrors horizontally onto "from
    # above + leaving right -> up-right" and vertically onto "from below +
    # leaving left -> down-left", which are two of the other seven, while
    # Tail3/Tail4/Tail5/Tail6 map the same way. The rest of the game names no
    # direction at all: both cancels use the relative `>`, the eat is a bare
    # late rule, and the win ("no Dot", "all Player on Exit") is positional in no
    # way. There is no gravity and input is screen-relative.
    # The art is a closed orbit to match, which is the part the .txt had to be
    # fixed for: the six tail pieces are already exact rotations and mirrors of
    # each other (vertical<->horizontal, and the four corners onto each other),
    # and the four heads now are too -- they shipped with a single eye pixel off
    # the facing axis, so a mirrored head was art this game does not own. So a
    # flipped frame is a board Dotsnake could really hand out. No colour
    # augmentation, and with 12 levels the flips take the corpus from 48
    # presentations to 192.
    # Detroit: Become Immense is the same "mirrored sets that map onto each
    # other" argument in its purest form. Its four growth rules are stated with
    # absolute directions and its four Grow arrows are CHIRAL sprites -- but
    # GrowL's art is exactly GrowR's mirrored, GrowU's exactly GrowD's, and each
    # is the other pair rotated 90 degrees, so every symmetry of the frame maps
    # the arrow set onto itself and lands each arrow's art on the arrow whose
    # direction the action remap has just made it mean. Everything else is
    # relative (`>` movement, the four Lose/Win adjacency probes as one orbit,
    # a win that is "blocked on all four sides" and so orientation-free) and
    # there is no gravity. The Player's face is not symmetric, but it is a pure
    # marker -- no rule reads its position -- so a mirrored face misleads
    # nothing. 16 levels x 16 presentations = 256.
    # Directioban is gravity-free with screen-relative input and an
    # `All Target on Crate` win that is positional in no way. It is the one game
    # here whose rules are AXIS-prefixed -- `VERTICAL [ > Player | Crate1 ]`,
    # `HORIZONTAL [ ... Crate2 ]`, and the perpendicular Crate4 telekinesis --
    # and that is exactly why the flips are safe: a mirror maps each axis onto
    # itself, so a vertical-only crate is still vertical-only in the flipped
    # view, while the matching action remap keeps "the key that moves me down"
    # pointing down. (A rotation, which every ps: game already gets, swaps the
    # axes instead, and stays consistent because the crate's sprite -- bars
    # along the axis it travels -- rotates with the board.) Every sprite is
    # symmetric on both axes except the player, whose mirror is still plainly
    # the player and lands on no other object's art. 22 levels and no colour
    # augmentation (crate-on-target is read as the target showing through the
    # crate's open middle), so the flips take the corpus from 88 presentations
    # to 352.
    #
    # Four-room tilt mazes is isotropic by construction: not one of its five
    # rules names a direction (the slide chain, the wall cancel, the cursor
    # advance and the target clear are all written with a bare `>` or with no
    # modifier at all), input is screen-relative with no gravity, and `No Target`
    # names no direction either. The walls are one class -- `Wall = Bar or Edge`,
    # both plain darkgrey squares -- so a mirrored maze is a maze the game could
    # have shipped. Its two multi-cell glyphs are the interesting part: the blue
    # target is four corner quadrants (T1..T4) that compose into a CENTRED square,
    # which every symmetry of the frame maps onto itself, and the red block is
    # four quadrants of a face (P1..P4) whose mirror is not a shipped composition
    # -- but the face is a pure marker, no rule reads its orientation, and the
    # rotation augmentation already turns it. It gets no colour augmentation
    # (blue target versus red block versus grey wall is the whole reading), and
    # with only THREE levels the flips are the difference between 12 presentations
    # for the entire game and 48.
    #
    # ESCAPE! is the same argument in its simplest form: gravity-free, one
    # direction-agnostic rule (`[ > Player | Crate ] -> [ > Player | > Crate ]`),
    # screen-relative input, and an `All Player on Target` win that names no
    # direction. Nothing in it is chiral either -- no sprite is directional, so
    # a mirror can never show art the game does not own. It takes no colour
    # augmentation (crate-on-target is read as the target's corners showing
    # through the crate's transparent ones), so without the flips its whole
    # corpus would be one of four rotations per level.
    _FLIP_GAMES = frozenset({"Directioban",
                             # Snakeoban is the ESCAPE! argument plus one extra
                             # check. Gravity-free, screen-relative input, and a
                             # win condition that names no direction ("All Crate
                             # on Target" / "No Apple"). Its art IS directional
                             # -- the four PlayerUp/Down/Left/Right connectors
                             # that draw the snake as a joined body -- but they
                             # are an exact orbit of the mirror group (Left and
                             # Right are each other's hflip, Up and Down each
                             # other's vflip), so a flip maps connector art to
                             # connector art exactly as it maps the motion. And
                             # nothing can contest a cell: there is one player,
                             # each body segment carries exactly one marker
                             # (solvers/generate_snakeoban_training.py asserts no
                             # two aim at the same cell), and a crate chain is
                             # collinear -- so the rule-order chirality that
                             # Gobble Rush has to argue around cannot arise.
                             # That generator's --symmetry MEASURES it: the same
                             # engine directions driven at all 16 presentations,
                             # engine grid compared after every press.
                             "Snakeoban",
                             "Enqueue", "Add_Man_2__This_Time_It's_Arithmetical",
                             "Aerobatics", "Bad_Example", "Baguettes", "Bichrome",
                             "Blind_Maze_a1", "blockfaker", "Boolean_Bloom_0.37",
                             "Box_Fill", "Boupha's_Candle_Quest", "Bridge",
                             "Brendan_loves_mondays", "Broken_Abacus",
                             "Broken_Maze", "Bubblegoban", "cakemonsters",
                             "Cancel", "Candy_Bomb", "Circuit_Breaker",
                             "Clean_Up", "Color_Combination", "Collect_Gnocchi",
                             "Color_Totems", "Count_Mover", "Dang_I'm_Huge",
                             "Darkness_Sokoban", "Detroit__Become_Immense",
                             "Dotsnake", "Dreaming_of_Strawberries",
                             "EntrepotPhage_Demake", "EpicJamGame",
                             "ESCAPE!", "Escape_the_Void_Full_",
                             "Escaping_Limbo",
                             "ESL_Puzzle_Game_Challenge_Mode", "Explod",
                             "Fireproof_bomber", "Five_Pulloban_Puzzles",
                             "Four-room_tilt_mazes", "Fractured_Identity",
                             "FULL_CIRCLE",
                             "Futuristic_Block_Pushing_Game",
                             "Glue_Factory",
                             # Gobble Rush's charge resolution is NOT
                             # mirror-symmetric -- two charges contesting one
                             # cell are settled by the order the interpreter
                             # expands a rule's four directions, so "the
                             # leftward charge wins" is a fact about the screen
                             # (see solvers/generate_gobble_rush_training.py).
                             # It is in here anyway because that asymmetry is
                             # ALREADY fully exposed by the mandatory rotation:
                             # measured over the whole reachable space of all 21
                             # levels, every one of the 15881 order-sensitive
                             # transitions out of 1009128 is split by a rotation
                             # alone, and not one is split only by a mirror. So
                             # the flips cost nothing in consistency and buy 4x
                             # the presentations. Otherwise the ESCAPE! argument
                             # holds: gravity-free, direction-free win
                             # condition, screen-relative input, and no sprite
                             # that encodes a facing.
                             "Gobble_Rush!",
                             # Goblin Hooblob is the first game here whose art
                             # IS directional: each Hooman facing carries a pip
                             # on the side it walks toward (see the header of
                             # data/puzzlescript_games/Goblin_Hooblob.txt --
                             # without it the four facings render identically
                             # and the game is unobservable). That is still
                             # mirror-safe, because the pips were drawn as an
                             # orbit of the full 8-element group: each pair
                             # straddles the sprite's centre and is symmetric
                             # about its own axis, so every flip maps facing
                             # art to facing art exactly as it maps the motion
                             # -- a hooman's pip points the way it moves on
                             # screen at all 16 presentations. The mechanic
                             # itself is the ESCAPE! argument: gravity-free,
                             # every rule stated with the relative `>` force,
                             # a win condition that names no direction, and
                             # screen-relative input.
                             "Goblin_Hooblob",
                             # Hungry Kitty is the purest case of the ESCAPE!
                             # argument in this set: gravity-free, screen-
                             # relative input, a win condition that names no
                             # direction ("No Fish"), and a single moving object
                             # -- so nothing can ever contest a cell and the
                             # rule-order chirality that Gobble Rush has to
                             # argue around cannot arise. Its one rule ("pounce
                             # to the nearest fish along the pressed ray") is
                             # invariant under every symmetry of the square. The
                             # cat sprite is asymmetric art, but it encodes no
                             # facing: the game has one Player object and no
                             # direction states, so a mirrored cat is a cat.
                             "Hungry_Kitty",
                             # IceCrates states its slide as four per-direction
                             # rule blocks rather than with the relative `>`
                             # force, which is the shape that hid Gobble Rush's
                             # chirality -- so rather than argue from the rules,
                             # solvers/generate_icecrates_training.py --symmetry
                             # replays every level's plan on all 8 turned and
                             # mirrored copies of its own board and requires the
                             # pieces to land where the transform says. All six
                             # levels are exactly symmetric. The rest is the
                             # ESCAPE! argument: sliding on ice is gravity-free,
                             # the win condition ("Some PlayerStill on Goal")
                             # names no direction, input is screen-relative, and
                             # the only directional art -- the four sliding
                             # player facings -- is never on screen at rest.
                             "IceCrates",
                             # Buoy Deploy is the ESCAPE! argument with nothing
                             # to qualify: gravity-free, every one of its five
                             # live rules stated with the relative `>` / `<`
                             # force, a win condition ("All Target on Buoy")
                             # that names no direction, screen-relative input,
                             # and one moving object -- the player -- so nothing
                             # can ever contest a cell and the rule-order
                             # chirality Gobble Rush has to argue around cannot
                             # arise. Its octopi do not move (`randomDir` is not
                             # a token this parser knows, so that rule is an
                             # identity -- see the generator), which removes the
                             # last thing on the board that could have had a
                             # heading. No sprite encodes a facing.
                             "IMS445_Puzzlescript_Game__Buoy_Deploy_",
                             # Idols to the Burnt God is direction-blind in a
                             # way none of the others quite are: its whole
                             # mechanic is "everything orthogonally beside the
                             # cell you moved into burns one step", written as
                             # eighteen rules with no direction prefix, so each
                             # expands to all four directions and every statue
                             # advances exactly once however the interpreter
                             # orders them -- there is no cell for two matches
                             # to contest and so no chirality to arise. That is
                             # measured rather than assumed:
                             # generate_idols_to_the_burnt_god_training.py
                             # --symmetry replays all 27 levels' plans on all 8
                             # turned and mirrored copies of their own boards
                             # and every one lands where the transform says.
                             # The rest is the ESCAPE! argument: gravity-free,
                             # pushes stated with the relative `>` force, win
                             # conditions that are pure object counts (`no
                             # Ash`, `no Heretic`, `no Unburnt`), screen-
                             # relative input, and no sprite that encodes a
                             # facing.
                             "IDOLS_TO_THE_BURNT_GOD",
                             # Impasse is the one game in this set the ESCAPE!
                             # argument does NOT cover, and it is worth reading
                             # for that reason. Its hazards move in ABSOLUTE
                             # directions baked into their identity (`[vertical
                             # Player][Upper] -> [vertical Player][up Upper]`),
                             # and the game owns no left/right mover at all, so
                             # the interpreter cannot be equivariant under a
                             # quarter turn -- there is nothing to relabel onto.
                             # What makes the augmentation sound is that the
                             # direction is drawn ON THE SPRITE as an orbit of
                             # the full 8-element group: a mover paints a black
                             # bar on its leading edge and the two corners on
                             # that side and no others, so a flipped Upper is
                             # pixel-identical to a Lower and every transform
                             # maps direction art onto direction art exactly as
                             # it maps the motion. The rules a policy needs are
                             # then transform-invariant statements about the
                             # picture ("a piece travels the way its bar points;
                             # orange is triggered by a press along that axis,
                             # yellow by the perpendicular one"), which is also
                             # why the four shipped per-direction colours were
                             # collapsed to one per family (see the sprite
                             # contract at the top of the .txt). For the flips
                             # specifically that is measured, not argued:
                             # generate_impasse_training.py --symmetry rebuilds
                             # every level's LAYOUT under the mirror, the flip
                             # and the half-turn with Upper<->Lower and
                             # PerpUp<->PerpDown relabelled, reloads it so the
                             # level-start rules re-run, replays the level's own
                             # plan with the presses transformed, and all 23
                             # land exactly where the transform says.
                             "Impasse",
                             # I'm Sick Today is the Gobble Rush case again,
                             # measured the same way. Nothing in it names an
                             # absolute direction during play, but the cell the
                             # infection chain-hop leaves the player on is
                             # decided by the order the interpreter expands a
                             # rule's four directions -- a fact about the
                             # ENGINE, which no presentation transform can
                             # change. Over the COMPLETE reachable space of all
                             # 13 levels, 92 of 92956 transitions are settled
                             # that way and every one of them is split by a
                             # ROTATION; not one is split only by a mirror
                             # (solvers/generate_im_sick_today_training.py
                             # --symmetry). So the flips cost nothing in
                             # consistency and buy 4x the presentations.
                             # Otherwise the ESCAPE! argument holds: gravity-
                             # free, the one force in the game is the relative
                             # `>`, the win condition ("No Person") names no
                             # direction, input is screen-relative, and no
                             # sprite encodes a facing. (Wall_S, the shaded
                             # wall top picked by the game's one absolute-
                             # direction rule, is static scenery chosen once at
                             # level start -- the mandatory rotation already
                             # shows it facing all four ways.)
                             "I'm_Sick_Today",
                             # Kicking walls is the ESCAPE! argument with one
                             # wrinkle worth stating, because its four blocking
                             # rules DO name absolute directions:
                             #   up    [ > Movable | WallH ] ...
                             #   down  [ > Movable WallH | ] ...
                             #   left  [ > Movable WallV | ] ...
                             #   right [ > Movable | WallV ] ...
                             # They name them in AXIS-SYMMETRIC PAIRS -- one
                             # rule for entering a WallH cell and one for
                             # leaving it, which are the same boundary stated
                             # from its two sides, and likewise for WallV. A
                             # mirror maps that boundary to a boundary, and
                             # maps the sprite that draws it (WallH's bottom
                             # pixel row, WallV's left pixel column) onto the
                             # correct side of the mirrored pair, so the
                             # picture stays a faithful picture of the rules.
                             # There is no rule-order chirality to argue around
                             # either: one player, one force per crate (the
                             # push rule gives the force to a single crate and
                             # crates never chain), so no two matches can ever
                             # contest a cell. Measured anyway over a bounded
                             # slice of every level's reachable space by
                             # solvers/generate_kicking_walls_training.py
                             # --symmetry. The rest is ESCAPE!: gravity-free,
                             # screen-relative input, an `All Target on Crate`
                             # win that names no direction, and no sprite that
                             # encodes a facing.
                             "Kicking_walls",
                             # L.A.S.E.R. is the Impasse case: its emitters bake
                             # an ABSOLUTE direction into object identity
                             # (LaserUp / LaserDown / LaserLeft / LaserRight are
                             # four objects, and the six beam-propagation rules
                             # are written as per-direction blocks -- LATE RIGHT,
                             # LATE HORIZONTAL, ... -- which is the shape that
                             # hid Gobble Rush's chirality). What makes the
                             # augmentation sound is that the facing is drawn ON
                             # THE SPRITE as an orbit of the full 8-element
                             # group: an emitter is a symmetric core with two
                             # bright muzzle pixels on its firing edge and none
                             # elsewhere, so a mirrored LaserRight is pixel-
                             # identical to a LaserLeft and a quarter-turned one
                             # to a LaserUp -- exactly as the transform maps the
                             # beam. The beams themselves are the same orbit
                             # (Horizap is rows 1 and 3, Vertzap is columns 1
                             # and 3), so a turned horizontal beam IS the
                             # vertical sprite. See the art note in
                             # data/puzzlescript_games/L.A.S.E.R.txt. That the
                             # MECHANIC is equivariant is measured rather than
                             # argued: generate_l_a_s_e_r_training.py --symmetry
                             # rebuilds every level's LAYOUT under all 8
                             # transforms with the four emitters relabelled,
                             # reloads it so run_rules_on_level_start re-lights
                             # the beams, replays the level's own plan with the
                             # presses transformed, and all 20 land the pieces
                             # AND both zap layers exactly where the transform
                             # says. The rest is the ESCAPE! argument: gravity-
                             # free, both push rules stated with the relative
                             # `>` force on a single object (so no two matches
                             # can contest a cell), screen-relative input, and a
                             # win condition ("All target on box", "No player on
                             # zap") that names no direction.
                             "L.A.S.E.R",
                             # LameLightsOut is the ESCAPE! argument in its
                             # purest form -- it is the only game in this set
                             # with no directional mechanic AT ALL. Not one of
                             # its seven rules carries a direction modifier, and
                             # the neighbour toggles are written as bare
                             # `[ RuleApplier | LightOff ]`, which the
                             # interpreter expands to all four directions at
                             # once: pressing ACTION flips a PLUS, a shape that
                             # every element of the 8-fold symmetry group maps
                             # onto itself. No two matches can contest a cell
                             # either (the four expansions of a toggle rule read
                             # and write four different cells, and TempOn/TempOff
                             # are on their own collision layer, so the flip is
                             # computed from the pre-press board no matter what
                             # order the expansions run in) -- so there is no
                             # rule-order chirality of the Gobble Rush kind to
                             # argue around. The cursor's own movement is the
                             # engine's default screen-relative walk, the win
                             # ("No LightOff") names no direction, and both
                             # sprites are symmetric: the lights are framed
                             # squares and the cursor is four corner pixels.
                             # With 16 levels the flips take the corpus from 64
                             # presentations to 256.
                             "LameLightsOut",
                             # Little Girl, Big World is the ESCAPE! argument
                             # with nothing left over to argue about. Every one
                             # of its five rules is stated with a RELATIVE force
                             # -- `>` and `moving`, never `up` or `left` -- so
                             # the rigid-body flood that drags a whole connected
                             # block of walls along names no direction at all.
                             # There is no gravity, input is screen-relative, the
                             # win condition is `No Goal`, and no sprite encodes
                             # a facing: the girl faces the reader in every
                             # frame and the blocks are decorations. Nor is there
                             # a rule-order chirality of the Gobble Rush kind to
                             # worry about -- the only rule two matches could
                             # contest is `[ > Wall | Barrier ] -> cancel`, whose
                             # RHS is the same whichever match fires first.
                             # Measured, not assumed:
                             # generate_little_girl_big_world_training.py
                             # --symmetry rebuilds each level's LAYOUT under all
                             # 8 transforms, reloads it so
                             # run_rules_on_level_start re-runs, replays the
                             # level's own plan with the presses transformed, and
                             # all five land every wall exactly where the
                             # transform says and win on the same press. With 5
                             # levels the flips take the corpus from 20
                             # presentations to 80.
                             "Little_Girl,_Big_World",
                             # Love and Pieces is the LameLightsOut case -- two
                             # rules, neither of which names a direction: the
                             # cancel is stated with the relative `>` force
                             # (`[ > Player | Wall ] -> cancel`) and the
                             # absorption is a bare `[ Player | GrayBlock ]`,
                             # which the interpreter expands to all four
                             # directions. The win ("No GrayBlock") names no
                             # direction, input is screen-relative, there is no
                             # gravity, and no sprite encodes a facing (the
                             # player's little face is asymmetric but it is a
                             # pure marker -- every blob cell is the same object
                             # and no rule reads it).
                             # It only became sound when the absorption rule was
                             # wrapped in startloop/endloop, and that is the
                             # cautionary half of this note. `_execute_late_single`
                             # iterates each of a rule's expanded directions to a
                             # fixpoint but never re-runs the direction LIST, so
                             # the unlooped flood advanced one shell per turn in
                             # whichever of up/down/left/right came first -- i.e.
                             # "the piece above the piece you just ate joins you
                             # too" was true at some rotations and false at
                             # others, which is the Gobble Rush chirality with a
                             # core mechanic instead of a rare contested cell.
                             # Looped, the flood is a connected-component closure
                             # and the order cannot matter. Measured, not argued:
                             # generate_lovendpieces_training.py --symmetry
                             # rebuilds every level's LAYOUT under all 8
                             # transforms, reloads it, replays the level's own
                             # plan with the presses transformed, and all eight
                             # winnable levels land the blob and the pieces
                             # exactly where the transform says. With 8 levels
                             # the flips take the corpus from 32 presentations to
                             # 128.
                             "lovendpieces",
                             # Match Flow is the ESCAPE! argument with one extra
                             # thing to check. Gravity-free, screen-relative
                             # input, a single relative-force push rule, and a
                             # win condition stated entirely as a property of
                             # the tower GRAPH -- one component, no tower with
                             # three neighbours, none on a deactivator -- which
                             # names no direction; the last level's "dosquare on
                             # a switch" names none either. The extra thing is
                             # that it DOES own directional art: the four
                             # `connected*` link sprites. They are a closed
                             # ORBIT of the 8-element group (`connectedl` is
                             # `connectedr` mirrored, `connectedu` is
                             # `connectedd` flipped, and a quarter turn maps the
                             # horizontal pair onto the vertical one), and the
                             # two rules that draw them are one `up` and one
                             # `right` over the same symmetric pattern, so a
                             # turned board draws the turned link -- the art
                             # follows the presentation instead of contradicting
                             # it. Measured rather than argued, and in both the
                             # ways this family has learned to (see lovendpieces
                             # directly above, whose stage-1-only check was
                             # blind): generate_match_flow_training.py
                             # --symmetry replays every level's plan on all 8
                             # turned and mirrored copies of its own board, AND
                             # random-walks each level and re-plays every
                             # (board, press) it visits independently on the
                             # other seven presentations, requiring the same
                             # pieces AND the same win flag. Both clean: 14
                             # levels x 7 presentations, 39200 transformed
                             # presses. (The .txt's startloop fix is a fidelity
                             # fix -- without it two levels are unwinnable --
                             # but it is NOT what earns the flips: the unlooped
                             # flood was measured over the same 23520
                             # transformed presses and is chiral on none of
                             # them. It reaches too few cells, equally in
                             # every orientation, on these particular boards.)
                             "Match_Flow",
                             # Mars Attacks is the purest sokoban-shaped case
                             # of the ESCAPE! argument: gravity-free, screen-
                             # relative input, a win condition ("No fruit")
                             # that names no direction, and two rules stated
                             # entirely with the relative `>` force -- a push
                             # and a "delete the Fruit ahead of you". Only the
                             # alien ever moves, so nothing can contest a cell
                             # and the rule-order chirality Gobble Rush has to
                             # argue around cannot arise. No sprite encodes a
                             # facing (one Player object, no direction states,
                             # no per-orientation wall tiling).
                             # Measured rather than argued, in both the ways
                             # this family has learned to: generate_mars_
                             # attacks_training.py --symmetry replays every
                             # level's plan on all 8 turned and mirrored
                             # copies of its own board, AND -- because a
                             # shortest plan never presses into a wall, never
                             # shoves a crate into a crate and never wedges
                             # one -- random-walks each level and re-plays all
                             # 20069 of the transitions it finds on the other
                             # seven presentations. Not one is chiral. With 6
                             # levels the flips take the corpus from 24
                             # presentations to 96.
                             "Mars_Attacks",
                             # Miner To Miner Empire has an EMPTY rules section
                             # -- four objects, no rules at all -- so the whole
                             # mechanic is the interpreter's own movement
                             # resolution and there is no per-direction rule
                             # block of the kind that hid Gobble Rush's and
                             # lovendpieces' chirality. The rest of the ESCAPE!
                             # argument is unqualified: no gravity, screen-
                             # relative input, and a win condition ("All Player
                             # on Target") that names no direction. No sprite
                             # encodes a facing either -- one Player object, no
                             # direction states.
                             # The one thing a mirror could have caught is the
                             # order a TRAIN of miners settles in (one arrow
                             # moves every miner, and they block each other), so
                             # it is measured rather than argued: generate_miner_
                             # to_miner_empire_training.py --symmetry rebuilds
                             # every level's LAYOUT under all 8 transforms,
                             # reloads it, and replays both the level's own plan
                             # and a 400-press random walk -- the walk precisely
                             # to bump into walls and pile miners up, which a
                             # shortest plan does not. All 8 levels are exactly
                             # symmetric. With 8 levels the flips take the corpus
                             # from 32 presentations to 128.
                             "Miner_To_Miner_Empire",
                             # Mimic Translation is the ESCAPE! argument for a
                             # game with TWO bodies on one key: gravity-free,
                             # screen-relative input, win conditions ("All
                             # Target on Player", "All MimicTarget on Doppler")
                             # that name no direction, and every one of its five
                             # live rules stated with the relative `>` / `<`
                             # force -- including the one that defines the game,
                             # `[ > Player ][ Doppler ] -> [ > Player ][ <
                             # Doppler ]`, whose mirror image is itself. A
                             # mirror maps "the Doppler goes the other way" onto
                             # "the Doppler goes the other way".
                             # It is the contesting that had to be measured,
                             # because unlike Mars Attacks this game has two
                             # movers and they meet: a run of crates pushed from
                             # both ends gets a force from two rules of one `+`
                             # group, and two free push chains claiming the same
                             # cell are settled inside `_resolve_forces`. Both
                             # are engine order, which no presentation transform
                             # can change -- so the argument that neither can
                             # matter (a contested run is walled in by two
                             # bodies moving inward, and a contested cell blocks
                             # both chains symmetrically) is checked rather than
                             # trusted: generate_mimic_translation_training.py
                             # --symmetry replays every level's plan on all 8
                             # turned and mirrored copies of its own board AND
                             # random-walks each level, re-playing all 12250 of
                             # the transitions it finds on the other seven
                             # presentations. Not one is chiral.
                             # No sprite encodes a facing: Player and Doppler
                             # are the same picture in two colours and no rule
                             # reads either one's orientation. The two goals ARE
                             # vertical mirrors of each other in shape, which
                             # sounds like the trap -- but Blue and DarkBlue
                             # both land on ARC index 9, so the Target renders
                             # as a plain solid square that every symmetry maps
                             # onto itself, and the MimicTarget is told from it
                             # by being red. With 7 levels the flips take the
                             # corpus from 28 presentations to 112.
                             "Mimic_Translation",
                             # Minimalist is the smallest mechanic in this set
                             # and the ESCAPE! argument holds unqualified for
                             # it. Its ONLY rule is `[ > Player ] -> [ > Player
                             # ]`, an identity stated with the relative `>`
                             # force -- there is no per-direction rule block of
                             # the kind that hid Gobble Rush's and
                             # lovendpieces' chirality, and nothing at all left
                             # for a mirror to disagree with. Everything else
                             # the game does is the interpreter's own movement
                             # resolution against a wall, and there is exactly
                             # ONE mover, so no two forces can ever contest a
                             # cell and no settle order exists to be handed.
                             # No gravity, screen-relative input, and a win
                             # condition (`all Player on Finish`) that names no
                             # direction. The Player's sprite is a centred 3x3
                             # square (see the header of
                             # data/puzzlescript_games/Minimalist.txt) and the
                             # other three objects are solid colour, so every
                             # sprite is invariant under the whole 8-element
                             # group and none encodes a facing.
                             # Measured anyway, since it is nearly free:
                             # generate_minimalist_training.py --symmetry
                             # rebuilds every level's LAYOUT under all 8
                             # transforms, reloads it, and replays both the
                             # level's own plan and a 400-press random walk
                             # (the walk precisely to bump into walls and the
                             # board edge, which a shortest plan never does).
                             # All 10 levels are exactly symmetric. With 10
                             # levels the flips take the corpus from 40
                             # presentations to 160.
                             "Minimalist",
                             # Modality is the ESCAPE! argument unqualified.
                             # All four of its rules are stated with the
                             # relative `>` force (two pushes and two CANCELs),
                             # so there is no per-direction rule block of the
                             # kind that hid Gobble Rush's and lovendpieces'
                             # chirality; there is no gravity; input is
                             # screen-relative; the win condition (`All Crate
                             # on Target`) names no direction; and there is
                             # exactly ONE mover, so no two forces can contest
                             # a cell and no settle order exists to be handed.
                             # The mechanic is a property of the floor COLOUR
                             # under the pieces, which no symmetry touches.
                             # Every sprite was redrawn invariant under the
                             # whole 8-element group (the tiles are flat, the
                             # Target is four corners plus a centre pixel, the
                             # Player an inner square ring and the Crate a
                             # diamond outline -- see the header of
                             # data/puzzlescript_games/Modality.txt), and none
                             # encodes a facing.
                             # Measured anyway: generate_modality_training.py
                             # --symmetry rebuilds every level's LAYOUT under
                             # all 8 transforms, reloads it, and replays both
                             # the level's own plan and a 400-press random walk
                             # -- the walk precisely to press into the board
                             # edge and to try the pushes the CANCEL rules
                             # refuse, which a shortest plan never does. Every
                             # level is exactly symmetric.
                             "Modality",
                             # Opposition is the ESCAPE! argument with the one
                             # qualification this set keeps running into, and it
                             # is measured rather than argued. Its whole game is
                             # `[ > Player ][ BluePlayer ] -> [ > Player ][ <
                             # BluePlayer ]` -- both forces relative, no
                             # per-direction rule block of the kind that hid
                             # Gobble Rush's and lovendpieces' chirality, no
                             # gravity, screen-relative input, and win
                             # conditions (`All Player on RedExit`, `All
                             # BluePlayer on BlueExit`) that name no direction.
                             # The qualification is that it has up to four
                             # movers, so two chains CAN contest a cell and the
                             # interpreter's force-resolution order is an engine
                             # fact no transform can relabel. That is what
                             # generate_opposition_training.py --symmetry goes
                             # after: stage 1 replays all 13 winnable levels'
                             # plans on all 8 turned and mirrored copies of
                             # their own boards, and stage 2 re-plays 4200
                             # random-walk transitions (which is what reaches
                             # the nose-to-nose and contested-cell
                             # configurations a shortest plan never does)
                             # independently on the other seven presentations.
                             # 29400 transformed presses, 0 chiral. No sprite
                             # encodes a facing: the exits are 8-fold symmetric
                             # rings and the two bodies are pure markers whose
                             # own art is left-right symmetric already.
                             "Opposition",
                             # One Way Street is the ESCAPE! argument on its two
                             # push rules and a measurement on the other eight.
                             # `[ > Player | Box ]` and `[ > Box | Box ]` are
                             # stated with the relative `>` force; there is no
                             # gravity, input is screen-relative, the win
                             # condition (`All goal on box`) names no direction
                             # and there is exactly ONE mover, so no two chains
                             # can contest a cell. But the one-way tiles are
                             # eight DIRECTION-QUALIFIED cancels
                             # (`VERTICAL[ > Movable | Lefttile ]`,
                             # `RIGHT[ > Movable | Lefttile ]`, ...), which is
                             # precisely the per-direction rule-block shape that
                             # hid Gobble Rush's and lovendpieces' chirality, so
                             # it is measured:
                             # generate_one_way_street_training.py --symmetry
                             # rebuilds every level's LAYOUT under all 8
                             # transforms with the four TILE OBJECTS RELABELLED
                             # by the transform (their direction is baked into
                             # object identity, so moving only the cells would
                             # build a different puzzle -- the trap that
                             # reported ps:l_a_s_e_r chiral), reloads it and
                             # replays the level's own plan. All 10 levels land
                             # exactly where the transform says. The art is
                             # sound under the group for the same reason: the
                             # four arrow sprites are an ORBIT of the 8-element
                             # group (a mirrored LeftTile is pixel-identical to
                             # a RightTile, a quarter-turned one to an UpTile),
                             # so every transform maps tile art onto tile art
                             # exactly as it maps the entry direction, and the
                             # other four objects are invariant under the whole
                             # group. With 10 levels the flips take the corpus
                             # from 40 presentations to 160.
                             "One_Way_Street",
                             # Ouroboros is the strongest case in this set for
                             # measuring rather than arguing, and it measures
                             # clean. Both of its load-bearing rules -- the
                             # marker the head stamps behind itself and the
                             # `late [tail|u]` family that walks the tail along
                             # those markers -- are written as FOUR
                             # per-direction blocks rather than with the
                             # relative `>` force, which is exactly the shape
                             # that hid Gobble Rush's chirality, and its body
                             # segments are directional ART (see the header of
                             # data/puzzlescript_games/Ouroboros.txt: without
                             # the black flow bars all four render alike and the
                             # snake's own future is not in the frame). The bars
                             # were drawn as an orbit of the full 8-element
                             # group -- the top edge turned clockwise is the
                             # right edge, the left edge mirrored is the right
                             # edge -- so every transform maps segment art onto
                             # the segment it maps the motion onto.
                             # solvers/generate_ouroboros_training.py --symmetry
                             # replays each level's plan plus a 400-press random
                             # walk (ACTION included) on all 8 turned and
                             # mirrored copies of its own board, RELABELLING the
                             # segments through the same transform, and requires
                             # every object to land where the transform says:
                             # all six levels are exactly symmetric. The rest is
                             # the ESCAPE! argument -- no gravity, screen-
                             # relative input, and a win condition
                             # (`[> player|tail] -> win`) that names no
                             # direction. With only 6 levels the flips are the
                             # difference between 24 presentations for the whole
                             # game and 96.
                             "Ouroboros",
                             # Palette is the ESCAPE! argument with nothing to
                             # qualify, and it is still measured. All thirteen
                             # rules are stated with the relative `>` force
                             # (one push, twelve colour mixes), there is no
                             # gravity, input is screen-relative, and the win
                             # conditions (`All TRed on Red`, ...) name no
                             # direction. The usual qualification -- two chains
                             # contesting a cell, settled by the order the
                             # interpreter expands a rule's four directions
                             # (Gobble Rush, lovendpieces) -- cannot arise:
                             # `[ > Player | Crate ]` never re-fires, so
                             # exactly ONE crate carries a force in a turn and
                             # there is nothing for it to contest. No object
                             # encodes a direction either, so unlike
                             # One_Way_Street and L.A.S.E.R there is nothing to
                             # relabel; generate_palette_training.py --symmetry
                             # rebuilds each level's LAYOUT under all 8
                             # transforms, reloads it and replays the level's
                             # own plan, and all 9 land exactly where the
                             # transform says, colours included. The art is
                             # sound under the group too: the crate ring, the
                             # target diamond and the player's X are each
                             # invariant under all eight elements, and the only
                             # asymmetric sprites (the wall and background
                             # noise textures) are per-cell constants carrying
                             # no information. With 9 levels the flips take the
                             # corpus from 36 presentations to 144.
                             "Palette",
                             # Party Demon is the ESCAPE! argument -- no
                             # gravity, screen-relative input, a win condition
                             # (`No Happy`) that names no direction -- with two
                             # things that had to be measured rather than
                             # argued, which is why they were. (1) The sticky
                             # drag mints a fresh RIGID group per match and
                             # finds its matches in the interpreter's own
                             # per-direction scan order, the exact shape that
                             # hid Gobble Rush's chirality; it decides 229 of
                             # the 3.5M transitions in the levels' state spaces
                             # (generate_party_demon_training.py --rigid), so a
                             # chirality there would be real but nearly
                             # invisible. (2) The mood and beam rules are
                             # DIRECTION-QUALIFIED (`late HORIZONTAL` /
                             # `late VERTICAL`) and the two beam objects share a
                             # collision layer, so a turn swaps which of a
                             # crossing pair wins the cell -- which is why every
                             # authored disco level carries exactly ONE ball,
                             # whose row beam and column beam cannot meet. The
                             # art is clean: HorSad/VertSad render identically
                             # to their happy parent by the author's own design,
                             # the two beams are an exact orbit of the group (a
                             # turned row bar IS the column bar), and nothing
                             # encodes a facing.
                             # --symmetry rebuilds each level's LAYOUT under all
                             # 8 transforms, reloads it and replays the level's
                             # own plan plus a 300-press random walk with the
                             # presses transformed: all 13 levels exactly
                             # symmetric. The flips take the corpus from 52
                             # presentations to 208.
                             "Party_Demon",
                             # Pegs is the ESCAPE! argument plus one .txt edit
                             # and one art question. Every rule that matters is
                             # stated with the relative `>` force (the push,
                             # the four combine rules, the four hole rules),
                             # there is no gravity, input is screen-relative,
                             # and the win condition (`No Block`) names no
                             # direction. The hole-boundary decals are written
                             # as four per-direction rules but as a complete
                             # set, so the group permutes them among
                             # themselves. The one absolute-direction mechanic
                             # -- the replacement selection, which used to
                             # cycle forward on DOWN and backward on UP -- is
                             # now direction-free: any arrow advances it by
                             # one, so it reads the same at every presentation
                             # (see the header of
                             # data/puzzlescript_games/Pegs.txt). The art
                             # question is the TRIANGLE peg, whose sprite is
                             # chiral, and it is Hungry Kitty's answer: the
                             # shape encodes no facing -- the game has one
                             # triangle object and no direction states, so a
                             # mirrored triangle is a triangle -- and since
                             # the recolor each peg type carries its own ARC
                             # index (triangle green), so type is read off the
                             # colour and the shape is not even load-bearing.
                             # solvers/generate_pegs_training.py --symmetry
                             # random-walks each level and replays every
                             # (board, press) it visits INDEPENDENTLY on all 8
                             # turned and mirrored copies of that board: 25200
                             # transitions over the nine levels, all exactly
                             # symmetric. With only 9 levels the flips are the
                             # difference between 36 presentations for the
                             # whole game and 72.
                             "Pegs",
                             # Piedra is the ESCAPE! argument with nothing to
                             # qualify, and the shortest statement of it in
                             # this set: the ENTIRE rules section is one line,
                             # `[ > Player | Crate ] -> [ > Player | > Crate ]`,
                             # stated with the relative `>` force. There is no
                             # gravity (the title is a Sisyphus joke, not a
                             # mechanic -- nothing rolls back down), input is
                             # screen-relative, the win condition ("All Target
                             # on Crate") names no direction, there is one
                             # mover plus the crate it shoves so nothing can
                             # ever contest a cell, and the two sprites the
                             # game has -- added for the render fix, see the
                             # header of data/puzzlescript_games/Piedra.txt --
                             # are centred 3x3-in-5x5 bodies, invariant under
                             # all eight turns and mirrors.
                             # solvers/generate_piedra_training.py --symmetry
                             # measures it anyway: each level's own plan plus a
                             # 400-press random walk, replayed on all 8 turned
                             # and mirrored copies of its own layout, with both
                             # the player AND the crate required to land where
                             # the transform says -- 4/4 levels exactly
                             # symmetric. With only four levels the flips are
                             # the difference between 16 presentations for the
                             # whole game and 64.
                             "Piedra",
                             # Polyomino Puzzles is a packing puzzle -- lift a
                             # piece, walk it somewhere, set it down -- and it
                             # is gravity-free, screen-relative, and its win
                             # ("all Target on LoSquare") names no direction.
                             # What needs the check rather than the argument is
                             # that its two central rule blocks are NOT written
                             # with the relative `>` force: the pickup flood
                             # and the edge merge are each given as a `right`
                             # rule plus a `down` rule, the shape that hid
                             # Gobble Rush's chirality. The pair covers both
                             # orientations of each axis (the flood is stated
                             # once with the raised cell on the left and once
                             # with it on the right), so the argument does go
                             # through -- but
                             # solvers/generate_polyomino_puzzles_training.py
                             # --symmetry measures it: every level's plan plus
                             # a 400-press random walk on all 8 turned and
                             # mirrored copies of its own layout, comparing the
                             # cursor, the carried piece AND the full partition
                             # of dropped pieces after every press. All seven
                             # levels are exactly symmetric. No sprite encodes
                             # a facing (EdgeN/S/W/E are four objects, not four
                             # facings of one, so a turn maps the set onto
                             # itself), and with only six usable levels the
                             # flips are the difference between 24
                             # presentations for the whole game and 48.
                             "Polyomino_Puzzles",
                             # RBG is additive colour mixing played with one
                             # key: every coloured blob is a Player, so they
                             # all move together and a mix only happens when
                             # one of them has been STOPPED by scenery. That
                             # makes contested cells the norm rather than the
                             # exception -- the shape that exposed Gobble
                             # Rush's rule-expansion chirality -- so the
                             # ESCAPE! argument (gravity-free, every rule
                             # written with the relative `>` force, a win
                             # condition "Some White" that names no direction,
                             # screen-relative input, and sprites that are
                             # centred 3x3-in-5x5 bodies or a checkerboard,
                             # all invariant under the eight turns and
                             # mirrors) is not taken on its own here.
                             # solvers/generate_rbg_training.py --symmetry
                             # measures it: every state of every level's
                             # shortest-solution ball, every press, replayed on
                             # all 8 turned and mirrored copies of the board --
                             # 14532 transitions over the five levels, NOT ONE
                             # of them decided by the way the board faces. With
                             # only five levels the flips are the difference
                             # between 20 presentations for the whole game and
                             # 80.
                             "RBG",
                             # Rainbow Apples is the ESCAPE! argument --
                             # gravity-free, every one of its 37 rules written
                             # with the relative `>` / `<` force, win conditions
                             # ("All RedTarget on RedApple" and four siblings)
                             # that name no direction, screen-relative input,
                             # and no sprite that encodes a facing (the two
                             # players' faces are asymmetric art, but neither
                             # has a direction state). Its one qualification is
                             # level 9, where a mirror twin walks the opposite
                             # way and so two movers CAN contest a cell -- the
                             # shape that hid Gobble Rush's chirality. So
                             # solvers/generate_rainbow_apples_training.py
                             # --symmetry measures it twice: every level's plan
                             # replayed on all 8 turned and mirrored boards,
                             # and then EVERY reachable state and press of the
                             # six levels small enough to sweep -- level 9
                             # included, 5712 transitions, not one of them
                             # decided by the way the board faces.
                             "Rainbow_Apples",
                             # Santa's Great Escape is the ESCAPE! argument on
                             # its face -- gravity-free, every rule written
                             # with the relative `>` force or direction-free
                             # (the twelve delivery rules are single-cell, the
                             # dog's leap is one symmetric three-cell pattern),
                             # win conditions ("No Target", "No Player") that
                             # name no direction, screen-relative input, and no
                             # sprite that encodes a facing (Santa, the boy and
                             # the dog each have exactly one object and no
                             # direction states, so a mirrored santa is a
                             # santa). Its qualification is that up to three
                             # bodies move independently -- Santa plus a boy
                             # and two dogs chasing him -- which is the shape
                             # that hid Gobble Rush's chirality, so it is not
                             # argued.
                             # solvers/generate_santas_great_escape_training.py
                             # --symmetry replays each level's plan on all 8
                             # turned and mirrored copies of that level's own
                             # board and requires the whole board to land where
                             # the transform says after EVERY press, not just
                             # at the end: 6/6 levels exactly symmetric across
                             # all 141 moves. With only six levels the flips
                             # are the difference between 24 presentations for
                             # the whole game and 96.
                             "Santa's_Great_Escape",
                             # Sheep is gravity-free, screen-relative, and every
                             # rule is written with the relative `>` force or is
                             # direction-free, so the mechanic names no axis; its
                             # win condition is "No Sheep" and no sprite encodes
                             # a facing. It moves up to four bodies at once --
                             # the shape that hid Gobble Rush's chirality -- so
                             # that half is measured, not argued. The reason to
                             # expect it clean is structural: the four probe rays
                             # out of the shepherd are DISJOINT, so each scared
                             # sheep owns its own axis and no two can ever
                             # contest a cell, which is the only place a rule's
                             # four-direction expansion order could decide
                             # anything. solvers/generate_sheep_training.py
                             # --symmetry replays each level's plan AND a seeded
                             # 300-press random walk (which does what a plan
                             # never does: presses into walls, shoves crates into
                             # corners, scares several sheep at once) on all 8
                             # turned and mirrored copies of that level's board,
                             # comparing the WHOLE board after every press.
                             "Sheep",
                             # silver lungs is gravity-free and
                             # screen-relative, every rule that moves anything
                             # is written with the relative `>` force, its win
                             # condition ("All Target on Crate") names no
                             # direction, and after the render fix (see the
                             # header of
                             # data/puzzlescript_games/silver_lungs.txt) every
                             # sprite is a centred body, a corner set or a
                             # side-midpoint pair -- each of which the eight
                             # turns and mirrors map onto itself. What needs
                             # measuring rather than arguing is the beam chain:
                             # the force reaches the sliding platform through
                             # four per-direction `[ > Beam | ... | Magic ]`
                             # rules, which is the shape that hid Gobble Rush's
                             # chirality. The reason to expect it clean is
                             # structural -- there is one press per turn and
                             # every rule passes that same direction straight
                             # through, so every force on the board points the
                             # same way and two objects can never contest a
                             # cell, which is the only place a rule's
                             # four-direction expansion order could decide
                             # anything. solvers/generate_silver_lungs_training
                             # .py --symmetry replays each level's plan AND a
                             # seeded 300-press random walk (which does what a
                             # plan never does: presses into the Void, wedges a
                             # crate riding the platform, drives tiles into the
                             # board edge) on all 8 turned and mirrored copies
                             # of that level's own layout, requiring the
                             # player, both crates AND every platform tile to
                             # land where the transform says after every press:
                             # 7/7 levels exactly symmetric. With only seven
                             # levels the flips are the difference between 28
                             # presentations for the whole game and 112.
                             "silver_lungs",
                             # Slide Rule is gravity-free and screen-relative,
                             # every movement rule is written with the relative
                             # `>` force or with a per-direction marker the
                             # input stamps, and its win rules ("Player on
                             # Finish", "stationary Player on FinishX") name no
                             # direction. After the sprite pass (see the note
                             # at the top of
                             # data/puzzlescript_games/Slide_Rule.txt) every
                             # sprite is a centred body, a corner set, a
                             # side-midpoint pair or a rope lane whose four
                             # variants are each other's mirrors, so the eight
                             # turns and mirrors map the art onto itself; the
                             # crate's slack pips read by COUNT, which is
                             # invariant, exactly as they already must be under
                             # the mandatory rotation. What needs measuring is
                             # the pullback reel, which is four separate rules
                             # run to a fixpoint each in the fixed order left,
                             # right, up, down -- the shape that hid Gobble
                             # Rush's chirality, and here it really does bite:
                             # solvers/generate_slide_rule_training.py
                             # --symmetry turns and mirrors every ball state of
                             # every level (636k transitions) and finds 28
                             # decided by that rule order. They are in here
                             # anyway on the Gobble Rush argument, and more
                             # cheaply: all 28 are split by a ROTATION, which
                             # is mandatory, so not one is exposed only by a
                             # mirror -- and none of the 28 is a press the
                             # corpus labels, because they live in the
                             # double-move region no shortest plan enters.
                             "Slide_Rule",
                             # Snake Crate: every rule is written with the
                             # RELATIVE forces `>` / `<` (push, pull, the trail
                             # and the cancel), so not one of them names an
                             # axis, and the win is positional. It moves two
                             # bodies in one press -- the crate in front and the
                             # crate behind -- which is the shape that hid
                             # Gobble Rush's chirality, so that half is measured
                             # rather than argued; structurally the two land on
                             # `p + 2d` and `p`, which can never be the same
                             # square, so no two moves contest a cell and the
                             # order the interpreter expands a rule's four
                             # directions decides nothing. The trail sprite is
                             # symmetric and no sprite encodes a facing.
                             # solvers/generate_snake_crate_training.py
                             # --symmetry replays each level's plan AND a seeded
                             # 200-press random walk at all 16 presentations and
                             # requires exact frame equality with the transform.
                             "Snake_Crate",
                             # Sokoban Flipped is the tightest case for the
                             # flips in this whole set, because the game is ONE
                             # rule: `[ > Player | Target ] -> [ > Player | >
                             # Target ]`. It is written with the relative force
                             # `>`, so it names no axis; the win condition
                             # ("All Target on Crate") is positional; there is
                             # one press per turn and it moves at most two
                             # bodies, which land on `p + d` and `p + 2d` and
                             # so can never contest a cell -- which is the only
                             # place the order the interpreter expands a rule's
                             # four directions could decide anything (the shape
                             # that hid Gobble Rush's chirality). No sprite
                             # encodes a facing, and the two that are not
                             # symmetric under the group -- the wall's
                             # brown/darkbrown weave and the little figure --
                             # are each the only object drawn in their colours,
                             # so no mirror can land one object's art on
                             # another's. Measured anyway:
                             # solvers/generate_sokoban_flipped_training.py
                             # --symmetry replays each level's plan AND a seeded
                             # 200-press random walk (which shoves Targets into
                             # walls and into each other, and presses the unbound
                             # ACTION key) at all 16 presentations and requires
                             # exact frame equality with the transform. With ten
                             # levels the flips are the difference between 40
                             # presentations for the whole game and 160.
                             "Sokoban_Flipped",
                             # Match 3 Block Push is two rules: the ordinary
                             # sokoban push, written with the relative force
                             # `>`, and a LATE `[Crate|Crate|Crate] -> [| |]`
                             # written with no direction at all. One press
                             # moves at most two bodies, landing on `p + d` and
                             # `p + 2d`, so nothing can contest a cell; the win
                             # ("All Crate on Target") is positional; no sprite
                             # encodes a facing, and after the Target fix (see
                             # the header of
                             # data/puzzlescript_games/sokoban_match3.txt)
                             # every sprite but the wall's weave and the little
                             # figure is symmetric under the group, and each of
                             # those two is the only object drawn in its
                             # colours. The one asymmetry the mechanic has is
                             # that the interpreter resolves the undirected
                             # match rule up/down BEFORE left/right, so a
                             # crate at the centre of a plus loses its column
                             # and keeps its row -- but that is a fact about
                             # the ENGINE grid, which no presentation ever
                             # touches: for a ps: game the adapter transforms
                             # the rendered picture and remaps the input, and
                             # nothing else. Measured:
                             # solvers/generate_sokoban_match3_training.py
                             # --symmetry replays each level's plan AND a
                             # seeded 200-press random walk (which wedges
                             # crates, annihilates lines of four and presses
                             # the unbound ACTION key) at all 16 presentations
                             # and requires exact frame equality with the
                             # transform.
                             "sokoban_match3",
                             # Sokolor EX is a sokoban whose win condition is a
                             # WIRING rather than a place: every push relights
                             # the board from scratch, seeding one random block
                             # per colour and flooding the light along
                             # same-type and Connector adjacency. Every clause
                             # of the ESCAPE! argument holds -- gravity-free,
                             # screen-relative input, one moving body per press
                             # (the player and at most one block, landing on
                             # `p + d` and `p + 2d`, so nothing can contest a
                             # cell), and a win condition that names no
                             # direction ("All Target on Block", "All Color on
                             # Active"). Its spread rules are written with no
                             # direction and run to a FIXPOINT, so unlike
                             # sokoban_match3 there is not even an engine-grid
                             # rule order to argue around. The six block
                             # sprites are asymmetric, but a hole pattern is a
                             # picture and not a heading: it maps to its own
                             # mirror exactly as the motion does. The
                             # randomness is drawn from the module-global
                             # `random` and the engine grid is never
                             # transformed, so the same presses draw the same
                             # dice at every presentation. Measured:
                             # solvers/generate_sokolor_ex_training.py
                             # --symmetry replays each level's plan AND a
                             # seeded 200-press random walk (which wedges
                             # blocks into corners, splits every colour apart
                             # and presses the `noaction`-dead ACTION key) at
                             # all 16 presentations, with random.seed pinned
                             # per drive, and requires exact frame equality
                             # with the transform.
                             "Sokolor_EX",
                             # Simple Block Pushing Game is the ANCESTOR of the
                             # two above -- the canonical PuzzleScript sokoban,
                             # ONE rule: `[ > Player | Crate ] -> [ > Player |
                             # > Crate ]`, win `All Target on Crate`. It is
                             # sokoban_match3 minus the match rule, so the
                             # argument is that entry's with one clause fewer:
                             # the rule is written with the relative force `>`
                             # and names no axis, one press moves at most two
                             # bodies and they land on `p + d` and `p + 2d`,
                             # which can never be the same square, so nothing
                             # can contest a cell and the order the interpreter
                             # expands the rule's four directions decides
                             # nothing. The win is positional, no sprite
                             # encodes a facing, and after the Target fix (see
                             # the header of
                             # data/puzzlescript_games/sokoban_sanity.txt)
                             # every sprite but the wall's weave and the little
                             # figure is symmetric under the group, with each
                             # of those two the only object drawn in its
                             # colours. Measured:
                             # solvers/generate_sokoban_sanity_training.py
                             # --symmetry replays each level's plan AND a
                             # seeded 200-press random walk (which wedges
                             # crates against walls, against each other and
                             # into dead corners, and presses the unbound
                             # ACTION key) at all 16 presentations and requires
                             # exact frame equality with the transform. With
                             # only three levels the flips are the difference
                             # between 12 presentations for the whole game and
                             # 48.
                             "sokoban_sanity",
                             # Sokubunny is sokoban_sanity with the crates
                             # TYPED: the same single rule
                             # `[ > Player | Crate ] -> [ > Player | > Crate ]`
                             # with `Crate = RedCrate or BlueCrate`, and a win
                             # that is a conjunction of two positional
                             # conditions ("All RedTarget on RedCrate", "All
                             # BlueTarget on BlueCrate"). So the argument is
                             # that entry's, verbatim -- the rule is written
                             # with the relative force `>` and names no axis,
                             # one press moves at most two bodies and they land
                             # on `p + d` and `p + 2d`, which can never be the
                             # same square, so nothing can contest a cell and
                             # the order the interpreter expands the rule's
                             # four directions decides nothing -- plus one
                             # clause for the typing, which costs nothing: a
                             # crate's colour is what decides the win and no
                             # turn or mirror touches a colour. No sprite
                             # encodes a facing, and after the Target fix (see
                             # the header of data/puzzlescript_games/
                             # Sokubunny_and_the_colored_Boxes.txt) every
                             # sprite but the wall's weave and the little bunny
                             # is symmetric under the group, with each of those
                             # two the only object drawn in its colours.
                             # Measured: solvers/generate_sokubunny_training.py
                             # --symmetry replays each of the 23 levels' plans
                             # AND a seeded 200-press random walk (which wedges
                             # crates against walls, against each other and
                             # into dead corners, parks the wrong colour on a
                             # target, and presses the unbound ACTION key) at
                             # all 16 presentations and requires exact frame
                             # equality with the transform.
                             "Sokubunny_and_the_colored_Boxes",
                             # Something is the ESCAPE! argument with nothing
                             # left over. Gravity-free, screen-relative input;
                             # BOTH of its movement rules are stated with the
                             # relative `>` force and name no axis
                             # (`rigid [ > Player | Wall ]` and its `+` partner
                             # for the sleeping character); its other three
                             # rules -- the ACTION paradigm swap and the two
                             # `late` refloors -- are whole-board rewrites with
                             # no geometry at all; and the win condition is
                             # positional ("All APlayer on ATarget"). Nothing
                             # can contest a cell: one character is awake, and
                             # the at-most-two bodies a press moves land on p
                             # and p + d, which are never the same square, so
                             # the rule-order chirality Gobble Rush has to
                             # argue around cannot arise. No sprite encodes a
                             # facing: the two characters are a centred 3x3
                             # ring and plus, and the two targets are four
                             # marks on the cell rim (the corners, and the edge
                             # midpoints), so all six drawn sprites are
                             # invariant under the whole 8-element group --
                             # measured, along with the pairwise distinctness
                             # of all 21 reachable cell compositions, by
                             # solvers/generate_something_training.py --audit.
                             # That generator's --symmetry replays each of the
                             # ten recorded levels' plans AND a seeded 200-press
                             # random walk (which jams walls against the board
                             # edge, ferries the sleeper into dead pockets and
                             # presses the swap far more often than a plan
                             # does) at all 16 presentations, requiring exact
                             # frame equality with the transform.
                             "Something",
                             # Spacekoban is the ESCAPE! argument with one
                             # wrinkle worth naming. Gravity-free, screen-
                             # relative input, and a win condition that names
                             # no direction ("all crate on target" / "no dir").
                             # The wrinkle is that it HAS directional objects:
                             # the U/D/L/R momentum markers that carry a flying
                             # body's velocity. They are drawn with a fully
                             # TRANSPARENT sprite, so a mirror can never show
                             # art the game does not own -- and they exist only
                             # inside one keypress, since every settled frame
                             # this generator records has none (that is the
                             # `no dir` win clause, and the reachable-space
                             # census in solvers/generate_spacekoban_training.py
                             # measures zero stranded markers over all 207003
                             # states). Rule-order chirality is what is left,
                             # and that generator's --symmetry MEASURES it: each
                             # of the 20 levels' plans AND a seeded 150-press
                             # random walk (which jams trains into corridors and
                             # presses the unbound ACTION key) driven at all 16
                             # presentations, requiring exact frame equality
                             # with the transform.
                             "Spacekoban",
                             # Sorx-Aubi is here on the GOBBLE RUSH argument
                             # rather than the ESCAPE! one, because its
                             # dynamics are genuinely NOT symmetric. Its rigid
                             # rule spreads a force sideways through a clump of
                             # AubiiCrates, the adapter cancels the resulting
                             # forces pairwise rather than per component, and
                             # which cells of a partly-blocked clump end up
                             # moving therefore depends on the order
                             # `_apply_single_rule_forces` scanned the board in
                             # -- which is chirality-dependent (UP scans rows
                             # bottom-to-top, LEFT scans columns
                             # right-to-left). Measured over 6000 (state,
                             # press) pairs across all 15 levels by
                             # solvers/generate_sorx_aubi_training.py
                             # --symmetry: 55 of them are order-sensitive, and
                             # every single one is already split by a ROTATION
                             # alone -- none by a mirror only. So the mirrors
                             # cost nothing in consistency the mandatory
                             # rotation does not already cost, and buy 4x the
                             # presentations. Everything else is the ESCAPE!
                             # argument: gravity-free (the gravity block in the
                             # .txt is commented out), screen-relative input,
                             # a win condition that names no direction ("All
                             # Target on Player"), and -- after the Target and
                             # SorxxCrate fixes documented in the header of
                             # data/puzzlescript_games/Sorx-Aubi.txt -- every
                             # sprite invariant under the whole 8-element
                             # group. The only directional rules in the game
                             # are the two `late` ones that stamp AubiCorner
                             # seam decals between adjacent AubiiCrates, and
                             # that decal is an exact orbit of the group: it
                             # marks the two corners either side of a seam, so
                             # a turn or a mirror maps seam art to seam art
                             # exactly as it maps the seam. That generator's
                             # --audit measures it, transforming all 18
                             # reachable cell compositions eight ways at every
                             # cell size the levels use and finding no
                             # transform of one that is another's art.
                             "Sorx-Aubi",
                             # Stand is the ESCAPE! argument with the one
                             # wrinkle that a press moves up to SIX bodies at
                             # once -- the shape that hid Gobble Rush's
                             # rule-order chirality. Gravity-free,
                             # screen-relative input, a win condition that names
                             # no direction ("No Target1", i.e. every light
                             # covered at once), and its four sprites (a person,
                             # a light, and two paint jobs for a wall) encode no
                             # facing: no rule in the game reads a direction
                             # except through the relative `>` force. The reason
                             # to expect the many-bodies half clean is
                             # structural: every force a press assigns points
                             # the SAME way, so two chains can never claim one
                             # cell and the order `_resolve_forces` traced them
                             # in cannot decide anything. Measured anyway by
                             # solvers/generate_stand_training.py --symmetry:
                             # each of the five levels' plans AND a seeded
                             # 200-press random walk (which jams whole crowds
                             # into corners and presses the unbound ACTION key)
                             # driven at all 16 presentations, requiring exact
                             # frame equality with the transform. That
                             # generator's --audit adds the art half, turning
                             # and mirroring all 8 reachable cell compositions
                             # eight ways at every cell size the levels use and
                             # finding no transform of one that is another's
                             # picture.
                             "Stand",
                             # Stand II is Stand's argument plus two crate
                             # rules and one directional rule, and both
                             # additions are safe. The crates are pushed only
                             # through the relative `>` force ("[ > Player |
                             # Crate ] -> [ > Player | > Crate ]" and the
                             # inverted BigCrate one that drops the pusher's
                             # force), so no push names a screen direction, and
                             # neither rule chains -- a crate stopped by
                             # anything solid stops its pusher, in every
                             # direction alike. The one rule in the file that
                             # DOES name a direction, "late up [crate|no
                             # crateTop] -> [Crate|Cratetop]", paints a decal on
                             # the engine-up side of every crate purely so the
                             # author can draw a crate a square taller: it is
                             # deleted and re-derived every turn, sits in a
                             # collision layer of its own, and is read by no
                             # rule and no win condition, so a mirrored
                             # presentation simply draws crate tops on the
                             # mirrored side, consistently and as an exact
                             # transform. Up to SEVEN bodies move per press
                             # (level 9), clean for the same structural reason
                             # as Stand -- every force a press assigns points
                             # the same way. Measured by
                             # solvers/generate_stand_ii_training.py
                             # --symmetry (all 11 levels' plans AND a seeded
                             # 200-press random walk at all 16 presentations,
                             # exact frame equality with the transform) and
                             # --audit (all 30 reachable cell compositions,
                             # turned and mirrored eight ways at every cell
                             # size the levels use, no transform of one being
                             # the picture of another that MEANS something
                             # else).
                             "Stand_II",
                             # Stand III is Stand II's argument plus the hole
                             # rules, and those add nothing to it: falling is
                             # decided by CO-LOCATION, not by any direction.
                             # "late [falls hole] -> [hole]" and "late [player
                             # hole] -> [lose hole]" each name one cell and no
                             # relative or absolute direction, and the restart
                             # they lead to reloads the level, which a flipped
                             # presentation draws flipped like any other board.
                             # Everything else is inherited: gravity-free,
                             # screen-relative input, a win condition that
                             # names no direction ("No Target1"), no sprite
                             # that encodes a facing, the two "> Player |
                             # Crate" push rules that name no screen direction
                             # and do not chain, and the one directional rule
                             # ("late up [crate|no crateTop]") whose decal is
                             # deleted and re-derived every turn, sits in a
                             # collision layer of its own and is read by
                             # nothing. Up to SEVEN bodies move per press
                             # (level 13), clean for the same structural reason
                             # as Stand -- every force a press assigns points
                             # the same way, so two chains can never contest a
                             # cell. Measured by
                             # solvers/generate_stand_iii_training.py
                             # --symmetry (all 15 levels' plans AND a seeded
                             # 200-press random walk at all 16 presentations,
                             # exact frame equality with the transform, the
                             # walk deliberately left free to fall in holes and
                             # restart the level mid-replay) and --audit (all
                             # 36 reachable cell compositions, turned and
                             # mirrored eight ways at every cell size the
                             # levels use).
                             "Stand_III",
                             # Stained Glass is the ESCAPE! argument on the
                             # mechanic -- gravity-free, screen-relative input,
                             # win conditions that name no direction ("No
                             # UpGlass", ...), and only one moving body per
                             # press (only `[ > Player | <pane> ]` grants a
                             # force, and there are no chain pushes), so nothing
                             # can contest a cell and the rule-order chirality
                             # Gobble Rush has to argue around cannot arise.
                             # Its art IS directional twice over and both are
                             # safe. Each pane is an arrow pointing the way it
                             # must be pushed, and the four are an exact orbit
                             # of the group AS SHAPES (LeftGlass/RightGlass are
                             # each other's hflip, UpGlass/DownGlass each
                             # other's vflip), so a mirrored pane still points
                             # where the remapped press sends it; the frame
                             # draws its filled slots as bands on the side each
                             # pane entered from, which a flip carries with the
                             # entry. What does NOT follow the mirror is
                             # COLOUR -- a mirrored LeftGlass is a yellow
                             # right-pointing arrow, which the unmirrored game
                             # never shows -- but the mandatory rotation already
                             # permutes colour against direction the same way,
                             # so the flips add no ambiguity that was not there
                             # and the 16 variants stay pairwise distinct (a
                             # flip is a bijection on pixels).
                             # solvers/generate_stained_glass_training.py
                             # --symmetry measures the mechanic half: all four
                             # levels' plans plus a seeded 60-press tail off the
                             # plan, driven at all 16 presentations, engine grid
                             # compared after every press -- 0 boards differ.
                             "Stained_Glass",
                             # Stickyban is the ESCAPE! argument, and the reason
                             # to state it carefully is that two of its rules
                             # carry NO direction prefix at all --
                             # `[moving Player|Crate]` and `[moving Crate|Crate]`
                             # -- so they are expanded over all four directions
                             # and drag the whole orthogonally-connected
                             # component of crates the player is touching. That
                             # is MORE symmetric than a relative `>`, not less:
                             # the closure they compute is a graph fixpoint over
                             # the 4-neighbourhood, which every element of the
                             # dihedral group maps to the corresponding closure.
                             # The braking rules (`[> Crate|Wall]`, the wallHit
                             # flood, `[> player|wallHit]`) are the same two
                             # shapes again. Gravity-free, screen-relative input,
                             # and "All Target on Crate" names no direction.
                             # Nothing can contest a cell either: everything that
                             # moves in a turn moves by the SAME vector, so the
                             # rule-order chirality Gobble Rush has to argue
                             # around cannot arise. No sprite is directional
                             # (Crate and Target are 4-fold symmetric; Player,
                             # Wall and Background are not, but no rule reads
                             # their orientation and the mandatory rotation
                             # already turns them). It takes no colour
                             # augmentation -- crate-on-target is read as the
                             # target ring showing through the crate's hollow
                             # middle -- and with only SEVEN levels the flips are
                             # the difference between 28 presentations for the
                             # whole game and 112.
                             # solvers/generate_stickyban_training.py --symmetry
                             # measures it: every level's plan plus a seeded
                             # 200-press random walk, driven at all 16
                             # presentations, frames compared against the
                             # transform of the unaugmented ones.
                             "Stickyban",
                             # "Stand aside, everyone! I take large steps!" is
                             # the ESCAPE! argument with nothing left over.
                             # Gravity-free; screen-relative input; a win
                             # condition that names no direction ("All Target
                             # on Crate" and "All Player on VictoryStand"); and
                             # every one of its eight rules is written with the
                             # relative `>` force and no axis word, so each is
                             # expanded over all four directions and a turn or
                             # a mirror maps a match to a match. Nothing can
                             # contest a cell: there is ONE player and a press
                             # moves at most it and one crate, onto cells that
                             # are never the same square (the player lands on
                             # p+d or p+2d and the crate ahead of it), so the
                             # rule-order chirality Gobble Rush has to argue
                             # around cannot arise. No sprite encodes a facing
                             # -- crate, rock, target and stand are all
                             # centred and symmetric, and the only two that are
                             # not (the wall's weave and the little figure) are
                             # each the only object drawn in their colours.
                             # Measured: solvers/generate_large_steps_training
                             # .py --symmetry replays each of the 17 levels'
                             # plans AND a seeded 60-press random walk (which
                             # wedges crates against walls, eats rocks and
                             # presses the unbound ACTION key) at all 16
                             # presentations and requires exact frame equality
                             # with the transform.
                             "Stand_aside,_everyone!_I_take_large_steps!",
                             # Sticky Candy Puzzle Saga is the ESCAPE! argument
                             # plus one check on its decals. Gravity-free;
                             # screen-relative input; three win conditions that
                             # name no direction ("All Yellow on YellowTarget",
                             # ...); and the push is the relative `>` force with
                             # the stickiness stated as the axis-free
                             # `[ moving Candy | stationary Candy ]`, so every
                             # rule is expanded over all four directions and a
                             # turn or a mirror maps a match to a match. Nothing
                             # can contest a cell: one press translates the
                             # player and exactly one candy component, all by
                             # the same vector, so no two bodies are ever
                             # offered the same square and the rule-order
                             # chirality Gobble Rush has to argue around cannot
                             # arise. The only directional art is the join
                             # decals, and they are an exact orbit of the
                             # 8-element group -- JoinL's two pixels are the
                             # left column's corners and JoinR's the right
                             # column's, JoinU's the top row's and JoinD's the
                             # bottom row's, JoinBoth's all four -- placed by
                             # rules that are equivariant too (a candy is marked
                             # on each side that has a neighbour, and "a
                             # neighbour on both sides of an axis" renders as
                             # all four corners whichever axis it is). Measured:
                             # solvers/generate_stick_candy_puzzle_saga_training
                             # .py --symmetry replays each of the ten solved
                             # levels' plans AND a seeded 120-press random walk
                             # (which shoves candies into walls, welds the wrong
                             # colours together and presses the unbound ACTION
                             # key) at all 16 presentations and requires exact
                             # frame equality with the transform -- 320 replays,
                             # 0 violations.
                             "stick_candy_puzzle_saga",
                             # Straighten Up is the ESCAPE! argument with one
                             # extra step, and the extra step is the win
                             # condition rather than the art. Gravity-free;
                             # screen-relative input; the only rule with a
                             # player on its left is the relative-`>` push, and
                             # one press moves at most two bodies, onto `p + d`
                             # and `p + 2d`, which can never be the same square
                             # -- so nothing can contest a cell and the
                             # rule-order chirality Gobble Rush has to argue
                             # around cannot arise. What needs saying is that
                             # "All Crate on Path" LOOKS axis-bound: Path is
                             # computed every tick by rules written `late right
                             # [Crate|Crate|Crate]` and `late down [...]`, with
                             # PathHoriz and PathVert as separate objects. They
                             # come as a PAIR that is a complete orbit under a
                             # quarter turn (a rotation exchanges the two
                             # exactly as it exchanges the axes), the strip and
                             # repaint rules are the same pair, and the win
                             # condition reads `Path = PathHoriz or PathVert`,
                             # so the predicate "every crate is in a run of
                             # three" is invariant under the whole 8-element
                             # group. Neither path object draws anything
                             # (both TRANSPARENT); the blue a finished crate
                             # wears comes from BlueCrate, which is axis-blind.
                             # The one piece of directional art is the victory
                             # pose -- `late right [no Arm|Player|no Arm] ->
                             # [RightArm|Player|LeftArm]` -- and it is a proper
                             # mirror orbit (each arm is one pixel column
                             # hugging the player, so an hflip maps the pair
                             # onto itself), it encodes no facing, and it is on
                             # screen only on the frame the level ends.
                             # Measured: solvers/generate_straighten_up_
                             # training.py --symmetry replays all six levels'
                             # plans AND a seeded 200-press random walk (which
                             # shoves crates into walls and into each other,
                             # strands them where no line of three can reach
                             # them, and presses the unbound ACTION key) at all
                             # 16 presentations and requires exact frame
                             # equality with the transform.
                             "Straighten_Up",
                             # Swap Sokoban is the ESCAPE! argument with the
                             # least left over of any game in this set: it has
                             # exactly ONE rule,
                             # `[ > Player | Crate ] -> [ Crate | Player ]`,
                             # written with the relative force `>`, so it is
                             # expanded over all four directions and every
                             # element of the dihedral group maps a match to a
                             # match. Gravity-free; screen-relative input; the
                             # win condition "All Target on Crate" names no
                             # direction. Nothing can contest a cell -- a press
                             # moves at most two bodies and they SWAP, the crate
                             # taking exactly the square the player vacates, so
                             # no two bodies are ever offered the same square and
                             # the rule-order chirality Gobble Rush has to argue
                             # around cannot arise. Background, Crate and Target
                             # are symmetric under the whole group (Target's
                             # ring-plus-corners is, after the rendering fix in
                             # data/puzzlescript_games/Swap_Sokoban.txt);
                             # Player and Wall are not, but no rule reads their
                             # orientation and the mandatory rotation already
                             # turns them. With ten levels the flips are the
                             # difference between 40 presentations for the whole
                             # game and 160.
                             # solvers/generate_swap_sokoban_training.py
                             # --symmetry measures it: every level's plan plus a
                             # seeded 200-press random walk (which shoves crates
                             # into walls, drags them back off their targets and
                             # presses the unbound ACTION key), driven at all 16
                             # presentations, frames compared against the
                             # transform of the unaugmented ones.
                             "Swap_Sokoban",
                             # Swap the block! is Swap Sokoban's argument
                             # verbatim: the same single rule,
                             # `[ > Player | Crate ] -> [ Crate | Player ]`,
                             # from the same sokoban template, written with the
                             # relative force `>`; gravity-free; screen-relative
                             # input; "All Target on Crate" names no direction.
                             # The one difference is that it has two to five
                             # PLAYERS, all of them moving on every press, and
                             # that only strengthens the case rather than
                             # weakening it -- the players are one object type
                             # with no orientation, they resolve as independent
                             # runs along the pressed axis, and a swap still
                             # puts the crate on exactly the square its player
                             # vacates, so no two bodies ever contest a cell.
                             # Its Target is a full 5x5 frame (symmetric under
                             # the whole group), Background and Crate likewise;
                             # Player and Wall are not, the same way Swap
                             # Sokoban's are not. With five levels the flips are
                             # the difference between 20 presentations for the
                             # whole game and 80.
                             # solvers/generate_swap_the_block_training.py
                             # --symmetry measures it: every level's plan plus a
                             # seeded 200-press random walk (which shoves crates
                             # into walls, drags them back off their targets,
                             # trains players up behind a crate and presses the
                             # unbound ACTION key), driven at all 8
                             # presentations, the players and crates compared
                             # against the transform of the unaugmented ones.
                             "_Swap_the_block!",
                             # Sweet Hints is a Witness panel drawn with the
                             # arrow keys, and its whole state is a LINE. The
                             # ESCAPE! argument holds: gravity-free, screen-
                             # relative input, one moving object (the head of
                             # the line, so nothing can ever contest a cell),
                             # and win conditions -- `all player on target`,
                             # `no error` -- that name no direction. The
                             # constraint that decides the win is a flood with
                             # no direction prefix at all, so it expands over
                             # all four and reaches the same fixpoint from any
                             # orientation.
                             # What it does NOT get for free is the art: the
                             # four lineX segments carry a real direction (a
                             # lineX names its cell's SUCCESSOR, and only the
                             # neighbour pointing back at the head can be
                             # retreated onto), and the four targetX differ
                             # cosmetically. Both are exact orbits of the full
                             # 8-element group after the rendering fix in
                             # data/puzzlescript_games/Sweet_Hints.txt -- a
                             # quarter turn carries each onto the next and each
                             # is symmetric about its own axis -- so every
                             # transform maps direction art onto direction art
                             # exactly as it maps the motion. The two pictures
                             # that could NOT be made into orbits were removed
                             # as pictures instead: the five playerX facings are
                             # picked by a fixed L,R,U,D rule order that no
                             # rotation can carry (and decide nothing -- the
                             # facing is a function of the neighbouring lines,
                             # which are on screen and now carry the direction
                             # themselves), and `dot` was a chiral blob marking
                             # a square that has no orientation. Both are now
                             # group-invariant.
                             # solvers/generate_sweet_hints_training.py
                             # --symmetry measures the mechanic rather than
                             # arguing it, in the strong IceCrates form: every
                             # level's plan and a seeded random walk replayed on
                             # all eight turned and mirrored copies of the LEVEL
                             # LAYOUT, with the line, the targets and the ERROR
                             # cells required to land where the transform says.
                             # The errors matter: they are what the flood and
                             # the dot/square rules produce, i.e. the half of
                             # the mechanic a movement-only comparison would
                             # never touch. --audit measures the art, orbit and
                             # all. With five levels the flips are the
                             # difference between 20 presentations for the whole
                             # game and 80.
                             "Sweet_Hints",
                             # Switcheroo is the ESCAPE! argument with the
                             # beams as the one thing that needs saying.
                             # Gravity-free; screen-relative input; the win
                             # condition "All Exit on Temp" names no direction;
                             # and the walk has no rule at all behind it --
                             # nothing in this game is pushable, so a press
                             # translates the players and nothing else, no two
                             # bodies are ever offered the same square, and the
                             # rule-order chirality Gobble Rush has to argue
                             # around cannot arise. The teleport rules carry no
                             # direction prefix, so each is expanded over all
                             # four and every element of the dihedral group maps
                             # a match to a match.
                             # The beams are the part that is not free. A block
                             # lights its whole ROW with HBeam and its whole
                             # COLUMN with VBeam, and a quarter turn exchanges
                             # rows with columns -- so it exchanges the two
                             # objects, and the pair has to be an ORBIT of the
                             # group for a turned board to be a board. It is, on
                             # both halves. The rules are the same shape with the
                             # axes swapped (`late horizontal [Temp|]` against
                             # `late vertical [Temp|]`, and their two propagation
                             # twins), the cancel rule `[Action Player VBeam
                             # HBeam]` names both symmetrically, and the two
                             # teleport rules are each other with the beams
                             # exchanged. After the rendering fix in
                             # data/puzzlescript_games/Switcheroo.txt the ART is
                             # an orbit too, in the strongest form: HBeam is the
                             # four corners and VBeam the four edge midpoints,
                             # and each of those pixel sets is invariant under
                             # the WHOLE eight-element group, so neither beam can
                             # land on the other's picture under any turn or
                             # mirror. Background, Exit, Temp and even Wall
                             # are group-invariant as well; the Player is the
                             # only sprite that is not, the same way Swap
                             # Sokoban's is not, and no rule reads its
                             # orientation.
                             # solvers/generate_switcheroo_training.py
                             # --symmetry measures it: every level's plan plus a
                             # seeded 200-press random walk (which walks into
                             # walls and blocks, presses X on the crossings where
                             # it does nothing, and teleports blocks back off the
                             # exits they had already reached), driven at all 16
                             # presentations, frames compared against the
                             # transform of the unaugmented ones. --audit
                             # measures the art, orbit and all. With eight levels
                             # the flips are the difference between 32
                             # presentations for the whole game and 128.
                             "Switcheroo",
                             # The Blob is the ESCAPE! argument at its very
                             # simplest, and the one game here where the
                             # rule-order chirality Gobble Rush has to argue
                             # around is impossible by construction rather than
                             # by inspection. Every Player cell takes the SAME
                             # force from the input, so a press translates a set
                             # of cells by one -- an injective map, so no two
                             # bodies can ever contest a destination and there is
                             # nothing for a rule expansion order to decide. The
                             # only geometry is "blocked by a Wall or by the
                             # board edge, and that blocking propagates back
                             # along the pressed axis", which is exactly as
                             # mirror-symmetric as it is rotation-symmetric.
                             # Gravity-free, screen-relative input, and the win
                             # condition names no direction ("All Target on
                             # Player" / "No AntiTarget on Player"). Its rules
                             # never fire on the shipped levels at all (they are
                             # the Life/Poison growth-and-death machinery, and no
                             # level places either object), so the mechanic IS
                             # the movement resolution. The art is a full orbit
                             # of the 8-element group after the rendering fix in
                             # data/puzzlescript_games/The_Blob.txt -- solid
                             # blocks, a ring plus four corners, and a block
                             # minus four corners. --symmetry in
                             # solvers/generate_the_blob_training.py measures it
                             # in the strong form: every level's plan and a
                             # seeded random walk replayed at all 16
                             # presentations, engine grid AND frame compared
                             # against the transform after every press. With
                             # five levels the flips are the difference between
                             # 20 presentations for the whole game and 80.
                             "The_Blob",

                             # The Observable Universe is two universes on one
                             # grid: a press moves the Player one way and
                             # Player2 the opposite way, and every rule that
                             # couples them ([ > Player | Crate3 ], the crate
                             # pushes, and the eleven cancels) is written with
                             # relative directions only. Nothing is gravity and
                             # nothing reads an absolute axis, so a flip maps
                             # the mechanic onto itself with the two universes'
                             # roles preserved. That is exactly the argument
                             # that was wrong for ps:gobble_rush, so it is
                             # MEASURED: --symmetry in
                             # solvers/generate_the_observable_universe_training.py
                             # replays all six plans on the 8 presentations and
                             # then re-plays 929 random-walk transitions on the
                             # other seven -- 6503 transformed presses, 0
                             # chiral. The repainted art (see the header of
                             # data/puzzlescript_games/The_Observable_Universe.txt)
                             # is one 2x2 corner per collision layer, which no
                             # rotation or flip can make ambiguous because the
                             # four corners map onto each other as a set.
                             "The_Observable_Universe",

                             # The Trouble with Toasters is the ESCAPE!
                             # argument plus one measurement. Gravity-free,
                             # screen-relative input, a win condition that
                             # names no direction ("No Toaster"), and every
                             # rule that moves anything written with the
                             # relative force -- the push `[ > Player |
                             # Toaster ]`, its crate twin, and the pull
                             # `[ < Player | Crate ]`. The two spawner rules
                             # carry an `up` prefix but match a SINGLE cell, so
                             # the direction decides nothing.
                             # The one asymmetry is the match-3 rule: it is
                             # undirected, so the interpreter expands it to
                             # up, down, left, right and drives each to a
                             # fixpoint in that order, which resolves COLUMNS
                             # before rows -- a plus of toasters loses its
                             # column and keeps its two arms. That is split by
                             # a quarter turn and never by a mirror (a mirror
                             # maps columns to columns and rows to rows, so it
                             # commutes with an axis-ordered settle exactly),
                             # so the flips add no inconsistency the mandatory
                             # rotation is not already adding -- the Gobble
                             # Rush argument, here provable rather than
                             # measured. --symmetry in
                             # solvers/generate_the_trouble_with_toasters_training.py
                             # measures it anyway, and replays all five plans
                             # plus a 200-press random walk at the 16
                             # presentations with the frames compared against
                             # the transform. The art (see the header of
                             # data/puzzlescript_games/The_Trouble_with_Toasters.txt)
                             # is corner dots and centred blocks, a full orbit
                             # of the 8-element group, and no sprite encodes a
                             # facing.
                             "The_Trouble_with_Toasters",

                             # Time-Reversed Minicosmos is the ESCAPE! argument
                             # with its directional ART as the one thing that
                             # needs saying. The mechanic could not be simpler:
                             # a single piece rule,
                             # `[No Crate| > Player | Crate ] -> [No Crate|
                             # < Player | < Crate ]`, written with the relative
                             # forces and no direction prefix, so it expands
                             # over all four and every element of the dihedral
                             # group maps a match to a match. Gravity-free;
                             # screen-relative input; "All Target on Crate"
                             # names no direction. Nothing can contest a cell:
                             # a press moves at most two bodies, the crate onto
                             # exactly the square the player vacates and the
                             # player onto the square behind it, so no two are
                             # ever offered the same one and the rule-order
                             # chirality Gobble Rush has to argue around cannot
                             # arise. Background, Wall, Crate and Target are all
                             # symmetric under the whole group (Target's
                             # plus-plus-corners is, after the rendering fix in
                             # data/puzzlescript_games/Time-Reversed_Minicosmos
                             # .txt).
                             # The art that is NOT invariant is the player, and
                             # it encodes a real FACING: PlayerR and PlayerL are
                             # a little person drawn looking right and looking
                             # left, picked by the game's other two rules,
                             # `[Left PlayerR] -> [Left PlayerL]` and
                             # `[Right PlayerL] -> [Right PlayerR]`. They are an
                             # exact mirror orbit -- PlayerL is PlayerR's
                             # fliplr, pixel for pixel -- and the pair of rules
                             # that switches them is its own mirror image, so a
                             # horizontal flip maps facing art to facing art
                             # exactly as it maps the motion: the figure faces
                             # the way the player last pressed along the screen
                             # axis those rules read, at every one of the 16
                             # presentations. A vertical flip touches neither
                             # (it preserves the horizontal axis the rules read
                             # and turns the sprite upside down), and the
                             # quarter turns are the mandatory rotation, which
                             # every game here already takes.
                             # solvers/generate_time_reversed_minicosmos_
                             # training.py --symmetry measures it: all forty
                             # levels' plans plus a seeded 200-press random walk
                             # (which pulls crates off their targets, bumps them
                             # into walls, jams the player against a crate it
                             # cannot pull and presses the `noaction`-dead
                             # ACTION key) driven at all 16 presentations, the
                             # frames compared against the transform of the
                             # unaugmented ones. With forty levels the flips are
                             # the difference between 160 presentations for the
                             # whole game and 640.
                             "Time-Reversed_Minicosmos",

                             # Time-reversed Microban is the ESCAPE! argument in
                             # the one-rule form Swap Sokoban has, with the rule
                             # reading one cell wider:
                             # `[ no Wall no Player no Crate | > Player | Crate ]
                             #  -> [ Player | Crate | ]`. It is written with the
                             # relative force `>` and carries no direction
                             # prefix, so the interpreter expands it over all
                             # four directions and every element of the dihedral
                             # group maps a match to a match. Gravity-free;
                             # screen-relative input; the win condition "All
                             # Target on Crate" names no direction; there is no
                             # ACTION rule at all and no second rule of any
                             # kind, so there is no rule ORDER to be chiral.
                             # Nothing can contest a cell either: a press moves
                             # at most two bodies, the crate onto exactly the
                             # square the player is vacating and the player onto
                             # the square the rule's first cell already required
                             # to be empty, so no two bodies are ever offered
                             # the same square and the chirality Gobble Rush has
                             # to argue around cannot arise.
                             # Background (a flat block), Crate (a hollow 5x5
                             # box) and Target are symmetric under the whole
                             # group -- Target's ring-plus-corners is, after the
                             # rendering fix in
                             # data/puzzlescript_games/Time-reversed_Microban.txt
                             # -- while Player and Wall are not, exactly as in
                             # Swap Sokoban; no rule reads their orientation and
                             # the mandatory rotation already turns them.
                             # With ten levels the flips are the difference
                             # between 40 presentations for the whole game and
                             # 160.
                             # solvers/generate_time_reversed_microban_
                             # training.py --symmetry measures it: every level's
                             # plan plus a seeded 200-press random walk (which
                             # drags crates back off their targets, jams them
                             # flat against walls where no pull can reach them
                             # again, and presses the unbound ACTION key),
                             # driven at all 16 presentations, the frames
                             # compared against the transform of the unaugmented
                             # ones.
                             "Time-reversed_Microban",

                             # Together Alone is the ESCAPE! argument with one
                             # extra thing to check, and the check is what makes
                             # it safe. Gravity-free; screen-relative input;
                             # every movement rule is written with the relative
                             # force `>` and carries no direction prefix, so the
                             # interpreter expands each over all four directions
                             # and every element of the dihedral group maps a
                             # match to a match; and the win condition ("No
                             # BreakableTile / All Ben on BrownExit / All
                             # Isabelle on BlueExit") names no direction.
                             # The extra thing is that this game has TWO player
                             # objects, which is where rule-order chirality
                             # normally enters (see Gobble Rush). It cannot
                             # here: the two rules `[ > Ben ] [ IsaActive ] ->
                             # ...` and `[ > Isabelle ] [ BenActive ] -> ...`
                             # cancel the INACTIVE character's force before any
                             # movement rule is reached, so exactly one body
                             # ever carries a force and no two bodies are ever
                             # offered the same square. Ben's three-cell jump
                             # rule is the widest match in the file and it too
                             # is `>`-relative. ACTION is bound (it is the
                             # character switch) but it is non-directional and
                             # passes through the remap untouched.
                             # The art is direction-free after the rendering fix
                             # in data/puzzlescript_games/Together_Alone.txt:
                             # each collision layer owns a disjoint region of
                             # the sprite, and no sprite encodes a facing --
                             # there are no facing states in the game at all.
                             # Ground's two horizontal bands and Ben's left
                             # block are not group-invariant, exactly as Player
                             # and Wall are not in Swap Sokoban; no rule reads
                             # their orientation and the mandatory rotation
                             # already turns them. --audit checks all 16
                             # transforms of all 43 compositions against each
                             # other.
                             # With four levels the flips are the difference
                             # between 16 presentations for the whole game and
                             # 64.
                             # solvers/generate_together_alone_training.py
                             # --symmetry measures it: every level's plan plus a
                             # seeded 200-press random walk (which breaks tiles
                             # in orders no plan would, strands a character on a
                             # lone tile, parks both on one square and presses
                             # the switch), driven at all 16 presentations, the
                             # frames compared against the transform of the
                             # unaugmented ones.
                             "Together_Alone",

                             # Tour de Four is a sokoban with NO WALL OBJECT and
                             # a match-four clear. Both of its rules are as
                             # orientation-blind as a rule gets: the push,
                             # `[ > Movable | Movable ] -> [ > Movable |
                             # > Movable ]`, is written with the relative force
                             # and names no axis, and the clearing flood
                             # (`[Checking Red | Unchecked Red] -> [Checking Red
                             # | Checking Red]` and its two siblings) names no
                             # direction and, since the '+' group fix in
                             # data/puzzlescript_games/_Tour_de_Four.txt, runs
                             # to a FIXPOINT -- so what dies is the connected
                             # COMPONENT and cannot depend on which of the four
                             # expanded directions the interpreter takes first.
                             # That is one clause stronger than sokoban_match3,
                             # which has to argue that its columns-before-rows
                             # order is a fact about the engine grid; here there
                             # is no order to argue about at all. Gravity-free,
                             # screen-relative input, and the win ("No Block,
                             # No Cleared") names no direction. Nothing can
                             # contest a cell either: one press moves the player
                             # and one contiguous run of blocks, all of them one
                             # square along the same axis, so no two bodies are
                             # ever offered the same square. Red is a symmetric
                             # 5x5 with a centre dot; Blue's two dots and
                             # Yellow's three are not rotation-invariant, but a
                             # dot pattern is a picture and not a heading -- no
                             # rule reads it, the block is one object whichever
                             # way it is turned, and each is the only object
                             # drawn in its colours. The board's only geometric
                             # asymmetry, the EDGE that refuses a push, is
                             # mapped to an edge by every element of the group.
                             # Measured: solvers/generate_tour_de_four_training
                             # .py --symmetry replays every level's plan AND a
                             # seeded 200-press random walk (which jams runs of
                             # blocks against all four edges, strands blocks in
                             # corners, clears components of six at once and
                             # presses the unbound ACTION key) at all 16
                             # presentations and requires exact frame equality
                             # with the transform. With thirteen levels the
                             # flips take the corpus from 52 presentations to
                             # 208.
                             "_Tour_de_Four",

                             # TwinPush is the ESCAPE! argument with the
                             # two-player clause discharged from the RULES
                             # rather than from the level geometry. Gravity-free
                             # (a crate stays where it is put); screen-relative
                             # input; the entire rule listing is
                             # `[ > Player | Crate ] -> [ > Player | > Crate ]`
                             # plus a sound-only `[ > Player ] -> [ > Player ]`,
                             # both written with the relative force `>` and
                             # naming no axis; and the win condition ("All
                             # Target on Crate") names no direction.
                             # Two players is where rule-order chirality
                             # normally enters (see Gobble Rush), and it cannot
                             # here. The only force on the board is the single
                             # input direction, handed to every Player, so of
                             # the four directions the interpreter expands the
                             # push rule over, only one can match anything at
                             # all -- every body that moves on a press moves the
                             # SAME way. Two bodies are therefore never offered
                             # one square from different sides: a crate has a
                             # single cell behind it and so a single possible
                             # pusher, and a player whose own move is refused
                             # blocks the player behind it (they share a
                             # collision layer) rather than swapping with it.
                             # On top of that, every shipped level puts the two
                             # players in components separated by a solid double
                             # column of Wall, so they cannot come near each
                             # other in the first place.
                             # The art carries no facing -- there is one Player
                             # object and no direction states -- and after the
                             # rendering fix in
                             # data/puzzlescript_games/TwinPush.txt the Crate
                             # and Player sprites each leave their (3,3) pixel
                             # transparent, which is one point of a rotation-
                             # and-mirror orbit of four, so which one is open is
                             # a picture and not a heading. --audit checks all
                             # eight transforms of all seven compositions
                             # against each other at the 2px cell these boards
                             # render at.
                             # Measured: solvers/generate_twinpush_training.py
                             # --symmetry replays every level's plan AND a
                             # seeded 200-press random walk (which jams both
                             # crates into corners and against walls, walks both
                             # players over their targets and presses the
                             # unbound ACTION key) at all 16 presentations and
                             # requires exact frame equality with the transform.
                             # With four levels the flips take the corpus from
                             # 16 presentations to 64.
                             "TwinPush",

                             # Two-faced is the ESCAPE! argument with one extra
                             # thing to show, and this game shows it more
                             # strongly than any other entry here. Gravity-free;
                             # screen-relative input; a win condition ("All
                             # TargetH on CrateH" / "All TargetV on CrateV") that
                             # names no direction; and no two bodies contesting a
                             # square -- the only force on the board is the
                             # single pressed direction, handed to both halves of
                             # the one player, and a crate that cannot move jams
                             # rather than swapping with anything.
                             # The mechanic IS axis-flavoured -- a press across
                             # the body's long axis pushes CrateH, a press along
                             # it pushes CrateV -- but "across" and "along" are
                             # what every element of the dihedral group
                             # preserves, so a mirrored board plays the mirrored
                             # game with CrateH still CrateH. The 16 push rules
                             # come in horizontal/vertical pairs that are each
                             # other with the axes swapped, and the two rules
                             # that re-create the player at the end of a turn
                             # (`up [auxM|auxM] -> [playerB|playerT]` and
                             # `right [auxM|auxM] -> [playerL|playerR]`) are the
                             # same statement on the two axes.
                             # The ART is an exact orbit of the group, in the
                             # strongest form: after the rendering fix in
                             # data/puzzlescript_games/Two-faced.txt the
                             # ANTI-DIAGONAL reflection (rot90 then fliplr)
                             # carries PlayerT's sprite pixel-for-pixel onto
                             # PlayerR's and PlayerB's onto PlayerL's, and back
                             # -- which is exactly the relabelling that
                             # reflection makes on the board, since it turns a
                             # vertical body into a horizontal one and sends its
                             # top half to the right and its bottom half to the
                             # left. Wall, floor, the outside pockets and BOTH
                             # bare targets are group-INVARIANT, so no turn can
                             # rename a goal, and the crate art carries no
                             # heading. A frame is never ambiguous either: a
                             # board holds EITHER a T/B pair or an L/R pair and
                             # never one of each.
                             # Measured: solvers/generate_two_faced_training.py
                             # --symmetry replays every level's plan AND a seeded
                             # 200-press random walk (which pivots the body off
                             # jammed crates, shoves runs of crates into walls,
                             # presses into walls where the turn cancels, and
                             # presses the unbound ACTION key) at all 16
                             # presentations and requires exact frame equality
                             # with the transform. --audit checks the art, orbit
                             # and all. With nine levels the flips take the
                             # corpus from 36 presentations to 144.
                             "Two-faced",

                             # Undertale X/O Puzzle is the ESCAPE! argument with
                             # nothing left over to argue about, and it is the
                             # purest case in this set. Not one of its three
                             # rules names a direction -- they are position
                             # predicates over the single Player ("standing on a
                             # baked target burns it", "standing on a fresh one
                             # bakes it", "pressing a key while on the button
                             # makes every target fresh again") -- the movement
                             # is the interpreter's own, input is screen-
                             # relative, and the win condition ("No TargetFresh"
                             # / "No TargetBurnt" / "All Player on Button")
                             # names no axis. There is one moving object, so the
                             # rule-order chirality Gobble Rush has to argue
                             # around cannot arise at all, and no sprite encodes
                             # a facing: Player, Wall, Button and the three
                             # target states are one object each, with no
                             # directional siblings a mirror could rename.
                             # Measured anyway:
                             # solvers/generate_undertale_x_o_puzzle_training.py
                             # --symmetry replays each level's plan plus a
                             # seeded 60-press random tail (which burns targets,
                             # presses into walls, wastes the unbound ACTION key
                             # and trips the button reset with work in progress)
                             # on all eight turned and mirrored copies of the
                             # LEVEL LAYOUT, and requires the player and all
                             # three target populations to land where the
                             # transform says. Both levels are exactly
                             # symmetric. With two levels the flips take the
                             # corpus from 8 presentations to 32.
                             "Undertale_X_O_Puzzle",
                             # Veggie Jam is the ESCAPE! argument with nothing
                             # to qualify. The entire rules section is three
                             # copies of the canonical push,
                             # `[ > Player | Cauliflower ] ->
                             #  [ > Player | > Cauliflower ]` and the same for
                             # Carrot and Peas, every one of them stated with
                             # the relative `>` force and none naming an axis.
                             # There is no gravity, input is screen-relative,
                             # and the win condition ("All Cauliflower on
                             # Target" / "All Carrot on Target" / "All Peas on
                             # Target") is positional. One press moves at most
                             # two bodies, onto p+d and p+2d, which can never be
                             # the same square, so no two moves can contest a
                             # cell and the order the interpreter expands a
                             # rule's four directions cannot decide anything.
                             # The ART is the part that needs the check rather
                             # than the argument -- the Carrot's stalk points
                             # up-left, the Peas sit on a diagonal and the
                             # Target's bottom row is inset, so a mirrored board
                             # shows art the .txt does not literally contain.
                             # That is only a problem if the mirrored art of one
                             # composition is another composition's art, and
                             # solvers/generate_veggie_jam_training.py --audit
                             # turns and mirrors all 11 compositions eight ways
                             # at both cell sizes (7px and, for the 7x17 level,
                             # 3px) and compares each against every other: 1540
                             # transformed comparisons, all distinct.
                             # --symmetry measures the mechanic the same way:
                             # each level's own plan plus a 400-press random
                             # walk (which shoves veggies into walls and corners
                             # and presses the unbound ACTION key), replayed on
                             # all 8 turned and mirrored copies of its own
                             # LAYOUT, with both the player AND the veggie
                             # required to land where the transform says -- 3/3
                             # levels exactly symmetric. With only three levels
                             # the flips are the difference between 12
                             # presentations for the whole game and 48.
                             "Veggie_Jam",
                             # VRPS is the game in this set with the MOST
                             # directional art -- four scooters, four avatar
                             # facings, four player facings, four combined
                             # avatar-on-scooter sprites, four boundary markers
                             # and sixteen warning ticks -- and the whole of it
                             # was drawn as an exact orbit of the 8-element
                             # group: turning the scooterright sprite clockwise
                             # IS the scooterdown sprite, pixel for pixel, so in
                             # a turned view every piece points the way the
                             # remapped key drives it.
                             # solvers/generate_vrps_training.py --audit
                             # measures all 36 of them at all 8 presentations
                             # (288 comparisons, no exceptions) and asserts that
                             # the 231 cell compositions the game can show stay
                             # pairwise distinct at all five of its board sizes.
                             # The MECHANIC is not stated relatively -- the
                             # scooter carry, the "riding freezes the player"
                             # rule and the warning marks are all written out as
                             # four per-direction blocks, which is the shape
                             # that hid Gobble Rush's chirality -- so --symmetry
                             # measures that too: each level's plan plus a
                             # 300-press random walk, replayed on all 8 turned
                             # and mirrored copies of the LAYOUT (cells moved
                             # and every directional object renamed), with the
                             # avatar, the player, every crate and every scooter
                             # required to land where the transform says. All 8
                             # levels exactly symmetric.
                             "VRPS",

                             # Weird Dave is a sokoban whose pusher is a RIGID
                             # polyomino: every cell of the body is its own
                             # Player, one arrow key gives all of them the same
                             # force, and if any of them is obstructed
                             # [ MOVING Player CantMove ] -> cancel throws the
                             # whole turn away. So a press translates the body
                             # and the crates it walks into by one fixed vector,
                             # and a crate chain is collinear. Translation by a
                             # fixed vector is INJECTIVE, so nothing can ever
                             # contest a destination and there is nothing for
                             # the interpreter's rule-expansion order to decide
                             # -- Gobble Rush's chirality cannot arise. The
                             # geometry is "blocked by a Wall, and blocking
                             # cancels", as mirror-symmetric as it is
                             # rotation-symmetric; the game is gravity-free with
                             # screen-relative input; and the win condition
                             # names no direction ("ALL Target ON Crate").
                             # The art holds up after the rendering fix in
                             # data/puzzlescript_games/Weird_Dave.txt: PlayerL
                             # and PlayerR are EXACT horizontal mirrors of each
                             # other once the palette collapses their permuted
                             # colour lists, which is the one pairing a mirror
                             # trades on, and the snake and the sparkles are
                             # symmetric. PlayerT/PlayerB are not each other's
                             # vflip, but the rotation augmentation this game
                             # already takes draws both of them upside down, so
                             # a mirror shows nothing categorically new.
                             # solvers/generate_weird_dave_training.py --audit
                             # measures the art (0 unexplained cross-hits over
                             # the 8 turns and mirrors, the PlayerL/PlayerR
                             # orbit declared) and --symmetry measures the
                             # mechanic in the strong form: every level's plan
                             # and a seeded random walk replayed at all 16
                             # presentations, engine grid AND frame compared
                             # against the transform after every press. With
                             # three levels the flips are the difference between
                             # 12 presentations for the whole game and 48.
                             "Weird_Dave",
                             # Zombie Invasion is the ESCAPE! argument with one
                             # thing that has to be measured rather than argued.
                             # Gravity-free, screen-relative input, a win
                             # condition that names no direction ("some Player
                             # on Goal"), and no sprite anywhere encodes a
                             # facing -- the player and the zombie are the same
                             # 5x5 glyph in two colours. What is not free is
                             # that several objects move at once and can contest
                             # a cell (a zombie two cells away in line, walked
                             # into head-on, is the common case), so the
                             # rule-expansion order Gobble Rush had to steer
                             # around could in principle decide an outcome.
                             # solvers/generate_zombie_invasion_training.py
                             # --symmetry measures it: every level's plan
                             # replayed on all 8 turned and mirrored copies of
                             # its own board, engine grid compared after every
                             # press, zero divergence. With four levels the
                             # flips are the difference between 16 presentations
                             # for the whole game and 64.
                             "Zombie_Invasion",
                             # Zombie Rescue is the same argument as Zombie
                             # Invasion, with a stronger answer to the one part
                             # that has to be measured. Gravity-free,
                             # screen-relative input, a win condition that names
                             # no direction ("All Human on Target"), and no
                             # sprite anywhere encodes a facing. Several objects
                             # move at once on EVERY tick here -- the player,
                             # the crate it shoves, the child following it, and
                             # then the whole bestiary -- so the rule-expansion
                             # order Gobble Rush had to steer around could in
                             # principle decide an outcome. It cannot: a
                             # monster's lure set is always a piece and the
                             # Prev marker it just left behind, i.e. two
                             # ORTHOGONALLY ADJACENT cells, and being in line
                             # with both from different directions would put the
                             # monster on one of them -- so no monster ever has
                             # two directions to break a tie between and the
                             # up/down/left/right scan order is never consulted.
                             # solvers/generate_zombie_rescue_training.py
                             # --symmetry measures it: all fifteen plans plus
                             # 150 random walks replayed on all 8 turned and
                             # mirrored copies of their own board, engine grid
                             # compared after every press, zero divergence over
                             # 32816 presses. With fifteen levels the flips are
                             # the difference between 60 presentations for the
                             # whole game and 240.
                             "Zombie_Rescue"})

    # Games that get a color augmentation: a set of surfaces are each recolored
    # to a random ARC palette entry, coordinated so they end up mutually distinct
    # (and none matches the black frame border). Deterministic per (seed, level).
    # The concrete surface list per game is built by _recolor_surfaces(); see also
    # _do_reset. Drop_Maze recolors bg/wall/ball/goal. Enqueue recolors the
    # grid-pattern background, the grid-pattern walls, the selection cursor AND
    # each block↔pad color class as one unit (so the match logic stays readable
    # while the palette is randomized — a pure relabel, engine matches by index).
    # Everything_Antimatters recolors its border (Wall frame), background field
    # and the player-controlled Cursor square, each avoiding the colors of the
    # crates/pit/marker it must not blend into (see _recolor_surfaces).
    _RECOLOR_GAMES = frozenset({"Drop_Maze", "Enqueue", "Everything_Antimatters",
                                "A_Knight's_Tour"})

    def _do_reset(self):
        """Reset to current level."""
        self._undo_stack.clear()
        level_idx = self._current_level_index % max(1, len(self._game.levels))
        if self._game.levels:
            self._engine.load_level(self._game.levels[level_idx])
        if self._game_name in self._NO_ROTATION_GAMES or \
                self._game_name in self._NATIVE_ROTATION_GAMES:
            # Native-rotation games carry their own in-engine rotation mechanic,
            # so the np.rot90 frame augmentation stays off (rotation_k = 0) to
            # avoid rotating an already-correctly-oriented view.
            self._rotation_k = 0
        else:
            # Draw the rotation from a private RNG seeded deterministically from
            # (base seed, level index). The global ``random`` module can't be
            # used here: its state is shared process-wide (any other consumer
            # desyncs it) and ``load_level`` above may itself draw from it via
            # level-start rules — both of which made "same seed → same rotation"
            # fail. A per-reset ``Random`` instance is reproducible regardless
            # of call site or surrounding randomness.
            self._rotation_k = _random_module.Random(
                self._seed + self._current_level_index
            ).randint(0, 3)

        # Rotation augmentation for native-rotation games: start the episode at a
        # random one of the game's own rotations by applying a deterministic
        # number (0–3) of in-engine rotate-steps. This uses the game's real
        # mechanic (e.g. Drop_Maze walking the Player to another quadrant), so
        # gravity and the ball's resting position stay consistent — no floating,
        # no desynced ball.
        if self._game_name in self._NATIVE_ROTATION_GAMES:
            forbidden = self._UNSOLVABLE_NATIVE_ROTATIONS.get(
                self._game_name, {}).get(self._current_level_index, set())
            allowed = [k for k in range(4) if k not in forbidden]
            # choice([0,1,2,3]) matches the old randint(0,3) exactly, so levels
            # with no forbidden phase keep their previous rotation-starts.
            n_rot = _random_module.Random(
                f"native-rot:{self._seed}:{self._current_level_index}"
            ).choice(allowed)
            rot_dir = self._NATIVE_ROTATION_DIR.get(self._game_name, "left")
            for _ in range(n_rot):
                self._engine.step(rot_dir)

        # Horizontal-flip augmentation: deterministically decide whether to
        # left-right mirror the presented frame (own RNG stream, reproducible).
        # _HFLIP_GAMES get an action-free observation mirror; _FLIP_GAMES get an
        # independent horizontal + vertical flip whose axis swaps are applied to
        # the directional action too (see perform_action).
        if self._game_name in self._HFLIP_GAMES or self._game_name in self._FLIP_GAMES:
            self._hflip = bool(_random_module.Random(
                f"hflip:{self._seed}:{self._current_level_index}"
            ).randint(0, 1))
        else:
            self._hflip = False

        # Vertical-flip augmentation: independent of the horizontal flip, drawn
        # from its own RNG stream. Only _FLIP_GAMES (gravity-free, screen-relative
        # moves) get it — flipud on a gravity game would invert gravity.
        if self._game_name in self._FLIP_GAMES:
            self._vflip = bool(_random_module.Random(
                f"vflip:{self._seed}:{self._current_level_index}"
            ).randint(0, 1))
        else:
            self._vflip = False

        # Color augmentation: recolor each planned surface in order, each from
        # its own reproducible RNG stream and excluding the colors already
        # assigned this reset, so all surfaces stay mutually distinct (e.g. a
        # ball / block is never the same color as the background it sits on).
        # A "flatten" surface paints every object solid with the chosen color;
        # a "remap" surface replaces just one original color across (possibly
        # multi-color) objects, leaving their other pixels intact.
        #
        # Restore original colors first so the augmentation always applies to a
        # clean base — remap surfaces search for an original color a prior reset
        # would otherwise have already replaced.
        for obj, orig_colors, orig_dom in self._recolor_originals:
            obj.colors = list(orig_colors)
            obj.dominant_color = orig_dom
        used: set[int] = set()
        for surf in self._recolor_plan:
            choices = [i for i in surf["allowed"] if i not in used]
            if not choices:
                continue
            idx = _random_module.Random(
                f'{surf["key"]}-recolor:{self._seed}:{self._current_level_index}'
            ).choice(choices)
            used.add(idx)
            setattr(self, f'_{surf["key"]}_recolor_idx', idx)
            if surf.get("mode") == "remap":
                old = surf["old"]
                for obj in surf["objs"]:
                    obj.colors = [idx if c == old else c for c in obj.colors]
                    if obj.dominant_color == old:
                        obj.dominant_color = idx
            else:
                for obj in surf["objs"]:
                    obj.dominant_color = idx
                    obj.colors = [idx for _ in obj.colors] if obj.colors else [idx]

        self._current_frame = self._present_frame(_render_frame(self._engine, self._game))
        self._state = GameState.NOT_FINISHED
        self._action_count = 0

    def _present_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply the presentation augmentations (rotation, then horizontal flip,
        then vertical flip) to a freshly rendered engine frame. The flips are
        applied after the rotation, i.e. in screen space; the matching action
        remap in perform_action inverts them in the same order."""
        if self._rotation_k != 0:
            frame = np.rot90(frame, k=self._rotation_k)
        if self._hflip:
            frame = np.fliplr(frame)
        if self._vflip:
            frame = np.flipud(frame)
        # Return a contiguous copy: np.rot90 / np.fliplr / np.flipud yield views,
        # and callers store / mutate the frame independently of the engine's next
        # render.
        return np.ascontiguousarray(frame)

    # ------------------------------------------------------------------
    # ARCBaseGame interface (game_id comes from BaseAdapter; PuzzleScript keeps
    # its own perform_action + _make_frame_data — too specialised to fold into
    # the base template (rule-restart, drained play animations, per-tick
    # extra_frames) — but wires in the base's generic snapshot UNDO below).
    # ------------------------------------------------------------------

    def set_level(self, idx: int) -> None:
        """Select a level by index."""
        self._current_level_index = idx
        # Seed the random module for reproducible rotation
        _random_module.seed(self._seed + idx)
        self._do_reset()

    def full_reset(self) -> None:
        """Reset to level 0."""
        self._current_level_index = 0
        _random_module.seed(self._seed)
        self._do_reset()

    def level_reset(self) -> None:
        """Reset the current level."""
        _random_module.seed(self._seed + self._current_level_index)
        self._do_reset()

    def perform_action(
        self,
        action_input: ActionInput,
        raw: bool = False,
    ) -> FrameDataRaw:
        """Execute one action and return a FrameDataRaw."""
        if action_input.id == GameAction.RESET:
            self.level_reset()
            return self._make_frame_data(action_input)

        # Generic snapshot-based UNDO (before the terminal short-circuit so it can
        # revive the agent from a GAME_OVER / WIN reached during a burst).
        if action_input.id == GameAction.ACTION7:
            if self._undo_stack:
                self._restore(self._undo_stack.pop())
            return self._make_frame_data(action_input)

        if self._state in (GameState.WIN, GameState.GAME_OVER):
            return self._make_frame_data(action_input)

        # Snapshot the pre-action state (bounded) so ACTION7 can reverse this step.
        self._undo_stack.append(self._snapshot())
        if len(self._undo_stack) > self.max_undo:
            self._undo_stack.pop(0)

        # Remap action for rotation (+ flip) augmentation. _hflip / _vflip are
        # only ever set for _FLIP_GAMES here (the _HFLIP_GAMES observation mirror
        # belongs to _NO_ACTION_REMAP_GAMES, which take the pass-through branch),
        # so this reduces to plain rotation remapping for every other game.
        if self._game_name in self._NO_ACTION_REMAP_GAMES:
            effective_id = action_input.id
        else:
            effective_id = _remap_action_full(
                action_input.id, self._rotation_k, self._hflip, self._vflip)
        direction = _ACTION_MAP.get(effective_id, "up")

        # Capture per-iteration snapshots when the player presses ACTION. The
        # action turn's `again` loop drains up to 50 iterations inside step(),
        # which is where most of botsket_ball's end-of-episode physics actually
        # happens (the ball hopping cell-by-cell toward the net). Without
        # snapshots, perform_action would only see the final frame of those
        # iterations.
        capture_during_step = (direction == "action")
        if capture_during_step:
            self._engine._capture_snapshots = True
            self._engine._snapshots = []

        # Step the engine
        moved = self._engine.step(direction)
        if moved:
            self._action_count += 1

        in_step_snapshots = self._engine._snapshots if capture_during_step else []
        if capture_during_step:
            self._engine._capture_snapshots = False

        # Handle restart (rule-triggered)
        if self._engine._rule_restart:
            level_idx = self._current_level_index % max(1, len(self._game.levels))
            if self._game.levels:
                self._engine.load_level(self._game.levels[level_idx])
            self._current_frame = self._present_frame(_render_frame(self._engine, self._game))
            return self._make_frame_data(action_input)

        # If a "play"-button animation has started this turn (e.g. botsket_ball
        # pressing the green play button), drain the animation to completion in
        # one perform_action call so the user sees the full episode and an
        # immediate WIN / LOSE verdict instead of having to spam more inputs to
        # advance the cycle. Capture an intermediate frame after each tick so
        # the renderer can play the cycle back step-by-step.
        play_animation_active = self._has_playing_objects()
        animation_frames: list[np.ndarray] = []
        if play_animation_active:
            # Render each in-step snapshot as a frame so the user sees the
            # whole simulation, not just the post-step settled state. The
            # initial pre-action grid is not included — the renderer already
            # showed it before the player pressed action.
            if in_step_snapshots:
                saved_grid = self._engine.grid
                try:
                    for snap in in_step_snapshots:
                        self._engine.grid = snap
                        animation_frames.append(
                            self._present_frame(_render_frame(self._engine, self._game))
                        )
                finally:
                    self._engine.grid = saved_grid
            else:
                animation_frames.append(
                    self._present_frame(_render_frame(self._engine, self._game))
                )

            # The play-button cycle (playu → playr → playd → playl → ...) can
            # spin forever when nothing else is moving. The play state only
            # advances when the speed-slider marker reaches `st` (a few ticks
            # apart), and a directional bot only moves on its matching play
            # state — so a full 4-state cycle can span well over a dozen
            # ticks. Stop once enough ticks have passed with no change to the
            # non-UI grid that we've definitely seen a full cycle without
            # further progress: the simulation has settled into a LOSE.
            stable_grid = self._grid_snapshot_excluding_playing()
            stable_streak = 0
            STABLE_TICKS_REQUIRED = 20
            for _ in range(120):
                more = self._engine.step_animation_once()
                animation_frames.append(self._present_frame(_render_frame(self._engine, self._game)))
                if not more:
                    break
                snapshot = self._grid_snapshot_excluding_playing()
                if snapshot == stable_grid:
                    stable_streak += 1
                    if stable_streak >= STABLE_TICKS_REQUIRED:
                        break
                else:
                    stable_grid = snapshot
                    stable_streak = 0

        # Update frame
        self._current_frame = self._present_frame(_render_frame(self._engine, self._game))

        # Check win / game over
        if self._engine.check_win():
            self._state = GameState.WIN
        elif self._engine.check_game_over():
            self._state = GameState.GAME_OVER
        elif play_animation_active:
            # The end-of-episode animation ran but the ball never landed on the
            # net — that's a LOSE.
            self._state = GameState.GAME_OVER
        elif self._action_count >= self._max_steps:
            self._state = GameState.GAME_OVER
        else:
            self._state = GameState.NOT_FINISHED

        return self._make_frame_data(action_input, extra_frames=animation_frames)

    def _has_playing_objects(self) -> bool:
        """True if any "playing" animation-state object is on the grid."""
        if not self._playing_indices:
            return False
        grid = self._engine.grid
        for r in range(self._engine.height):
            row = grid[r]
            for c in range(self._engine.width):
                cell = row[c]
                for idx in self._playing_indices:
                    if idx in cell:
                        return True
        return False

    def _grid_snapshot_excluding_playing(self) -> tuple:
        """Return a hashable snapshot of the grid with UI-cycle indices removed.

        Used to detect when the play-button animation cycle is the only thing
        still changing on the board, so the adapter can stop early.
        """
        ui_cycle = self._ui_cycle_indices
        grid = self._engine.grid
        rows: list[tuple] = []
        for r in range(self._engine.height):
            row = grid[r]
            rows.append(tuple(
                tuple(sorted(idx for idx in row[c] if idx not in ui_cycle))
                for c in range(self._engine.width)
            ))
        return tuple(rows)

    def _make_frame_data(
        self,
        action_input: ActionInput,
        extra_frames: list[np.ndarray] | None = None,
    ) -> FrameDataRaw:
        fd = FrameDataRaw()
        fd.game_id = self._game_id
        fd.state = self._state
        fd.levels_completed = self._current_level_index
        fd.win_levels = 1
        fd.action_input = action_input
        fd.full_reset = False
        fd.available_actions = self._available_actions
        if extra_frames:
            # Replay the per-tick animation frames; the final frame is the
            # current grid state (already appended above as the last entry).
            fd.frame = list(extra_frames)
        elif self._current_frame is not None:
            fd.frame = [self._current_frame]
        return fd

    # ── generic UNDO hooks (perform_action is custom, so _apply is unused) ─────
    def _apply(self, action_input: ActionInput) -> None:  # pragma: no cover
        raise NotImplementedError(
            "PuzzleScriptAdapter overrides perform_action directly")

    def _snapshot(self):
        """Capture the engine's mutable state (grid + rule flags + mutation
        counter) plus adapter bookkeeping. The derived position index / caches
        are NOT stored — they are rebuilt lazily on restore. rotation_k / flips
        are per-level constants, so undo (which never crosses a reset) leaves
        them untouched."""
        e = self._engine
        return (
            _copy.deepcopy(e.grid), e._rule_win, e._rule_restart,
            e._late_cancel, e._again_triggered, e._mutation_counter,
            self._action_count, self._state, self._current_frame,
        )

    def _restore(self, snap) -> None:
        e = self._engine
        (grid, e._rule_win, e._rule_restart, e._late_cancel, e._again_triggered,
         e._mutation_counter, self._action_count, self._state,
         self._current_frame) = snap
        e.grid = _copy.deepcopy(grid)
        # Force the object-position index + per-rule no-op cache to rebuild from
        # the restored grid rather than trusting the stale post-burst indices.
        e._position_index_dirty = True
        e._rule_noop_cache.clear()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_levels(self) -> int:
        """Total number of levels in this PuzzleScript game."""
        return len(self._game.levels)

    @staticmethod
    def list_available_games() -> list[str]:
        """Return names of available PuzzleScript games."""
        return list(_AVAILABLE_PUZZLESCRIPT_GAMES)

    def close(self) -> None:
        """No resources to release."""
        pass

    def __repr__(self) -> str:
        return (
            f"PuzzleScriptAdapter(game={self._game_name!r}, "
            f"level={self._current_level_index}, state={self._state})"
        )
