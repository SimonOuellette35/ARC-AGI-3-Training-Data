"""Native model of the PuzzleScript game VEXT EDIT ("ps:vext_edit", Jack).

WHY A MODEL AT ALL
------------------
The .txt carries 273 rules, most of them re-derived every tick (wall boundaries,
pit rims, palette highlighting, the held-item preview). The interpreter runs the
game at 22-123 steps/s depending on the board, which is one to three orders of
magnitude below what any search here needs -- see the note in
`solvers/common/ps_astar.py` about measuring `eng.step` before choosing an
engine-blackbox expert. So the mechanics are re-implemented natively and the
interpreter is kept only as the ORACLE the model is fuzz-verified against
(``--selfcheck``) and as the thing that finally replays a plan.

THE GAME
--------
Three worlds stacked on one grid, and one key set that drives whichever of them
currently owns the ``player`` group:

* **layer 1 -- the sokoban.** ``Player1`` walks, shoves ``Crate1`` and shoves
  other players. Crates are WELDED into rigid polyominoes (the ``crateu1`` /
  ``crated1`` / ``cratel1`` / ``crater1`` markers, seeded by the level legend and
  by the editor). ``Wall1`` blocks. A crate group with no cell off a ``Pit1``
  falls and becomes ``Fell1`` -- which fills the pit, counts as a crate for the
  win, and is walkable. A player that walks onto an unfilled ``Pit1`` becomes a
  ``Ded1`` corpse that can only swim WITHIN the pits, shoving ``Fell1`` around.
* **layer 0 -- the editor.** ``Cursor0a`` / ``Cursor0b`` fly over the level
  (``Wall1`` is not their wall; ``Wall0`` and the black frame are). They pick an
  item off a PALETTE -- a row of pickables each sitting directly above a
  ``number0`` digit that is its remaining stock -- and stamp copies of it into
  the level, one press each, decrementing the digit. ``inf0`` never decrements.
  Standing on a ``Player1`` and pressing ACTION hands control to it; pressing
  ACTION while playing hands control back to a cursor.
* **layer 2 -- the meta editor.** ``Cursor2a`` / ``Cursor2b`` edit the EDITOR:
  their palette stocks ``Cursor0a``, ``Wall0`` and ``Crate0`` on top of
  everything layer 0 can place, so the blue cursor writes the red cursor's world.
  ``Wall2`` and the frame are its only walls.

The win is all four conditions at once: every ``Target1`` under a crate or a
fallen crate, every ``PlayerTarget1`` under a player OR under the ``player1mark``
PREVIEW a cursor draws while holding one, and NO ``Cursor0`` parked on either
kind of target (which is what rules the preview cheese out for the red cursor and
leaves it open for the blue one).

MECHANICS THIS FILE ENCODES THAT ARE NOT READABLE OFF THE .txt
--------------------------------------------------------------
* **``cant`` is the whole movement model, and it runs BEFORE force
  propagation.** The game floods a "cannot move in the input direction" marker
  backwards from every wall through push-adjacency AND through crate welds, then
  makes only the CONTROLLED entity stationary. Because the flood's relation is
  the exact inverse of the forward push/weld relation, a rigid group with one
  blocked cell never gets a force at all -- which is why groups never tear. Model
  it the other way round (propagate forces, then block) and welded groups shear.
* **The ACTION flag is a per-object force that the rules CONSUME.** Only the
  ``player`` group (``active``, ``activecurse0``, ``activefell``, ``dead0a``,
  ``cursor2a``, ``cursor2b``) is handed one; the first rule in the file exists
  purely to pass it from ``activecurse0`` to the ``cursor0`` sharing its cell.
  Every rule whose RHS omits ``action`` eats it, so exactly one of pick / switch
  / drop / place / mode-change happens per press, in file order.
* **The held item's preview is a real object.** ``[placemark <x>mark] ->
  [placemark <x> <x>mark]`` is what actually places, so an item with no ``*mark``
  sprite (``four0``..``seven0``) can be held and never placed -- and
  ``player1mark`` satisfies a ``PlayerTarget1`` on its own.
* **Ghosts gate the mode switch.** While a player is active and no cursor0 is on
  the board, every layer-0 object turns into its transparent twin; a player
  standing on ``wall0ghost`` or ``crate0ghost`` may NOT switch to cursor mode,
  and a moving ``Crate1`` DRAGS a ``crate0ghost`` with it.
"""

from __future__ import annotations

DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
OPP = {"up": "down", "down": "up", "left": "right", "right": "left"}

#: ``showthis`` rules, in file order: palette object -> the preview it draws.
MARK_OF = [
    ("player1", "player1mark"), ("pit1", "pit1mark"), ("crate1", "crate1mark"),
    ("wall1", "wall1mark"), ("one0", "one0mark"), ("two0", "two0mark"),
    ("three0", "three0mark"), ("one2", "one2mark"), ("two2", "two2mark"),
    ("three2", "three2mark"), ("wall0", "wall0mark"), ("fell1", "fell1mark"),
    ("ded1", "ded1mark"), ("target0", "target0mark"), ("crate0", "crate0mark"),
    ("cursor0", "cursor0mark"), ("inf0", "inf0mark"), ("target1", "target1mark"),
    ("playertarget1", "playertarget1mark"),
]

#: Collision layers the previews sit on -- a second preview on the same layer
#: evicts the first, which is the only reason a multi-object palette slot is
#: well defined at all.
MARK_LAYERS = [
    ("pit1mark",),
    ("fell1mark", "ded1mark", "target0mark"),
    ("target1mark", "playertarget1mark"),
    ("player1mark", "crate1mark"),
    ("wall1mark",),
    ("one0mark", "two0mark", "three0mark", "wall0mark", "inf0mark"),
    ("one2mark", "two2mark", "three2mark"),
    ("cursor0mark", "crate0mark"),
]
_MARK_LAYER = {m: i for i, layer in enumerate(MARK_LAYERS) for m in layer}

#: ``[placemark <x>mark] -> [placemark <x> <x>mark]``, in file order.
PLACE_OF = {
    "player1mark": "player1", "pit1mark": "pit1", "crate1mark": "crate1",
    "wall1mark": "wall1", "one0mark": "one0", "two0mark": "two0",
    "three0mark": "three0", "one2mark": "one2", "two2mark": "two2",
    "three2mark": "three2", "wall0mark": "wall0", "fell1mark": "fell1",
    "ded1mark": "ded1", "target0mark": "target0", "crate0mark": "crate0",
    "target1mark": "target1", "inf0mark": "inf0",
    "playertarget1mark": "playertarget1",
}

#: ``x1`` -- what blocks a RED cursor from stamping into a cell.
X1 = ("player1", "crate1", "wall1", "target1", "playertarget1", "pit1")
#: ``x0`` -- the layer-0 half of ``xx``; ``xx`` additionally covers the cursors.
X0 = ("crate0", "wall0", "pit0", "fell0")

#: ``pickable`` / ``pickable2``: what each cursor may lift off a palette slot.
PICKABLE = ("player1", "pit1", "crate1", "wall1", "fell1", "ded1", "target0",
            "playertarget1", "target1")          # + number0 / number2, handled apart
PICKABLE2_EXTRA = ("wall0", "crate0", "cursor0a", "dead0a")

#: Which sets are plain cell sets in the working state.
CELLSETS = ("player1", "crate1", "fell1", "ded1", "wall1", "target1",
            "playertarget1", "pit1",
            "cursor0a", "cursor0b", "dead0a", "wall0", "crate0", "pit0",
            "fell0", "target0",
            "cursor2a", "cursor2b",
            # the control flags. They are SETS, not scalars: ps:vext_edit's
            # twin levels seat an `activecurse0` on each of two cursors and
            # drive both off the same key.
            "active", "activefell", "activecurse0")

NUMS = {"one0": 1, "two0": 2, "three0": 3, "four0": 4, "five0": 5,
        "six0": 6, "seven0": 7, "inf0": 0}
NUM_NAME = {v: k for k, v in NUMS.items()}
NUMS2 = {"one2": 1, "two2": 2, "three2": 3}
NUM2_NAME = {v: k for k, v in NUMS2.items()}


def _add(cell, d):
    dr, dc = DIRS[d]
    return (cell[0] + dr, cell[1] + dc)


class State:
    """Mutable working state. `key()` freezes it for the search."""

    __slots__ = ("s", "num0", "num2", "w1", "wf", "holding", "picked",
                 "ghost0", "ghost2")

    def __init__(self):
        self.s = {n: set() for n in CELLSETS}
        self.num0 = {}          # cell -> 1..7, or 0 for inf0
        self.num2 = {}          # cell -> 1..3
        self.w1 = set()         # crate1 welds: frozenset({a, b})
        self.wf = set()         # fell1 welds
        self.holding = set()    # cells carrying the `holding` object
        self.picked = {"a0": None, "b0": None, "a2": None, "b2": None}
        self.ghost0 = False
        self.ghost2 = False

    def clone(self):
        o = State.__new__(State)
        o.s = {k: set(v) for k, v in self.s.items()}
        o.num0 = dict(self.num0)
        o.num2 = dict(self.num2)
        o.w1 = set(self.w1)
        o.wf = set(self.wf)
        o.holding = set(self.holding)
        o.picked = dict(self.picked)
        o.ghost0 = self.ghost0
        o.ghost2 = self.ghost2
        return o

    def key(self):
        return (
            tuple(tuple(sorted(self.s[n])) for n in CELLSETS),
            tuple(sorted(self.num0.items())),
            tuple(sorted(self.num2.items())),
            tuple(sorted(tuple(sorted(e)) for e in self.w1)),
            tuple(sorted(tuple(sorted(e)) for e in self.wf)),
            tuple(sorted(self.holding)),
            (self.picked["a0"], self.picked["b0"],
             self.picked["a2"], self.picked["b2"]),
            self.ghost0, self.ghost2,
        )

    # -- convenience -----------------------------------------------------
    def cursors0(self):
        return self.s["cursor0a"] | self.s["cursor0b"]

    def curse0a(self):
        """The `curse0a` or-group: a red cursor OR a drowned one."""
        return self.s["cursor0a"] | self.s["dead0a"]


class Model:
    """The level's static scenery plus the `step` the search runs on."""

    def __init__(self, h, w, black, wall2, partb, partc):
        self.h, self.w = h, w
        self.black = frozenset(black)
        self.wall2 = frozenset(wall2)
        self.partbc = bool(partb or partc)
        self.partc = bool(partc)
        self.cells = [(r, c) for r in range(h) for c in range(w)]

    # -- construction ----------------------------------------------------
    @classmethod
    def from_engine(cls, game, eng):
        """Read a Model + its start State off a freshly-reset interpreter."""
        name = game.obj_idx_to_name
        h, w = eng.height, eng.width
        black, wall2, partb, partc = set(), set(), set(), set()
        st = State()
        # weld markers, resolved into undirected edges once the grid is read
        marks = {"crateu1": set(), "crated1": set(), "cratel1": set(),
                 "crater1": set(), "fellu1": set(), "felld1": set(),
                 "felll1": set(), "fellr1": set()}
        for r in range(h):
            for c in range(w):
                cell = (r, c)
                for o in eng.grid[r][c]:
                    n = name[o]
                    if n in CELLSETS:
                        st.s[n].add(cell)
                    elif n in marks:
                        marks[n].add(cell)
                    elif n in NUMS:
                        st.num0[cell] = NUMS[n]
                    elif n in NUMS2:
                        st.num2[cell] = NUMS2[n]
                    elif n == "black":
                        black.add(cell)
                    elif n == "wall2":
                        wall2.add(cell)
                    elif n == "wall2ghost":
                        wall2.add(cell)
                        st.ghost2 = True
                    elif n == "partb":
                        partb.add(cell)
                    elif n == "partc":
                        partc.add(cell)
                    elif n == "holding":
                        st.holding.add(cell)
                    elif n == "pickeda0":
                        st.picked["a0"] = cell
                    elif n == "pickedb0":
                        st.picked["b0"] = cell
                    elif n == "pickeda2":
                        st.picked["a2"] = cell
                    elif n == "pickedb2":
                        st.picked["b2"] = cell
                    elif n in ("wall0ghost", "crate0ghost", "pit0ghost",
                               "fell0ghost"):
                        st.s[n[:-5]].add(cell)
                        st.ghost0 = True
        for a, b, d in (("crateu1", "crated1", "up"),
                        ("cratel1", "crater1", "left")):
            for cell in marks[a]:
                st.w1.add(frozenset((cell, _add(cell, d))))
            for cell in marks[b]:
                st.w1.add(frozenset((cell, _add(cell, OPP[d]))))
        for a, b, d in (("fellu1", "felld1", "up"),
                        ("felll1", "fellr1", "left")):
            for cell in marks[a]:
                st.wf.add(frozenset((cell, _add(cell, d))))
            for cell in marks[b]:
                st.wf.add(frozenset((cell, _add(cell, OPP[d]))))
        m = cls(h, w, black, wall2, partb, partc)
        m.prune_welds(st)
        return m, st

    def inb(self, cell):
        return 0 <= cell[0] < self.h and 0 <= cell[1] < self.w

    def prune_welds(self, st):
        """``late [crateu1 no crate1] -> []`` + its per-direction twin: a weld
        survives only while BOTH of its cells still carry the piece."""
        st.w1 = {e for e in st.w1
                 if all(x in st.s["crate1"] for x in e)}
        st.wf = {e for e in st.wf
                 if all(x in st.s["fell1"] for x in e)}

    # -- derived ---------------------------------------------------------
    def groups(self, st, kind):
        """Connected components of ``crate1`` / ``fell1`` under their welds."""
        cells = st.s[kind]
        welds = st.w1 if kind == "crate1" else st.wf
        adj = {}
        for e in welds:
            a, b = tuple(e)
            adj.setdefault(a, []).append(b)
            adj.setdefault(b, []).append(a)
        seen, out = set(), []
        for cell in cells:
            if cell in seen:
                continue
            comp, stack = set(), [cell]
            seen.add(cell)
            while stack:
                x = stack.pop()
                comp.add(x)
                for y in adj.get(x, ()):
                    if y in cells and y not in seen:
                        seen.add(y)
                        stack.append(y)
            out.append(comp)
        return out

    def slot_marks(self, st, which):
        """The preview a cursor draws: the ``*mark`` set of the palette object
        directly above the selected count digit, with same-layer eviction."""
        pk = st.picked[which]
        if pk is None:
            return frozenset()
        src = _add(pk, "up")
        if not self.inb(src):
            return frozenset()
        by_layer = {}
        for obj, mk in MARK_OF:
            if obj == "cursor0":
                hit = src in st.cursors0()
            elif obj in NUMS:
                hit = st.num0.get(src) == NUMS[obj]
            elif obj in NUMS2:
                hit = st.num2.get(src) == NUMS2[obj]
            else:
                hit = src in st.s.get(obj, ())
            if hit:
                by_layer[_MARK_LAYER[mk]] = mk
        return frozenset(by_layer.values())

    # -- the turn --------------------------------------------------------
    def step(self, st, inp):
        """One press. Returns a NEW state (the old one when the turn cancels).

        Sets `self.ambiguous` when this press went through one of the game's
        four `random` rules with more than one match -- the global RNG then
        decides which body becomes Cursor0a and which becomes Cursor0b, so the
        press is not reproducible and the search must not take it."""
        self.ambiguous = False
        new = st.clone()
        if inp == "action":
            self._action_phase(new)
        self._ghost_phase(new)
        if inp != "action":
            self._movement_phase(new, inp)
        if self._late_phase(new):
            return st            # ``late [player1 ded1] -> cancel``
        return new

    # ---- action --------------------------------------------------------
    def _pickable(self, st, cell, blue=False):
        for n in PICKABLE:
            if cell in st.s[n]:
                return True
        if cell in st.num0 or cell in st.num2:
            return True
        if blue:
            for n in PICKABLE2_EXTRA:
                if cell in st.s[n]:
                    return True
        return False

    def _has_x1(self, st, cell):
        return any(cell in st.s[n] for n in X1)

    def _has_xx(self, st, cell):
        return (self._has_x1(st, cell)
                or any(cell in st.s[n] for n in X0)
                or cell in st.s["cursor0a"] or cell in st.s["cursor0b"]
                or cell in st.s["dead0a"])

    def _action_phase(self, st):
        # The ``player`` or-group is what the ACTION force is handed to.
        act = set()
        for kind in ("active", "activefell", "activecurse0", "dead0a",
                     "cursor2a", "cursor2b"):
            for cell in st.s[kind]:
                act.add((kind, cell))

        # [action activecurse0 curse0] -> [activecurse0 action curse0]
        for k, cell in list(act):
            if k != "activecurse0":
                continue
            for c0 in ("cursor0a", "cursor0b"):
                if cell in st.s[c0]:
                    act.discard((k, cell))
                    act.add((c0, cell))
                    break

        def held(*kinds):
            return sorted(cell for k, cell in act if k in kinds)

        # ---- pick up ready icon
        for kinds, mk, blue, nums in ((("cursor0a", "dead0a"), "a0", False, st.num0),
                                      (("cursor0b",), "b0", False, st.num0),
                                      (("cursor2a",), "a2", True, st.num2),
                                      (("cursor2b",), "b2", True, st.num2)):
            for cell in held(*kinds):
                below = _add(cell, "down")
                if (cell not in st.holding and self._pickable(st, cell, blue)
                        and below in nums):
                    st.holding.add(cell)
                    st.picked[mk] = below
                    self._consume(act, kinds, cell)

        # ---- switch to a different palette slot
        for kinds, mk, blue, nums in ((("cursor0a", "dead0a"), "a0", False, st.num0),
                                      (("cursor0b",), "b0", False, st.num0),
                                      (("cursor2a",), "a2", True, st.num2),
                                      (("cursor2b",), "b2", True, st.num2)):
            for cell in held(*kinds):
                below = _add(cell, "down")
                if (self._pickable(st, cell, blue) and below in nums
                        and st.picked[mk] is not None and st.picked[mk] != below):
                    st.picked[mk] = below
                    self._consume(act, kinds, cell)

        # ---- put the held icon back down
        for kinds, mks, blue in ((("cursor0a", "cursor0b"),
                                  ("a0", "b0", "a2", "b2"), False),
                                 (("cursor2a",), ("a2",), True),
                                 (("cursor2b",), ("b2",), True)):
            for cell in held(*kinds):
                below = _add(cell, "down")
                if (cell in st.holding and self._pickable(st, cell, blue)
                        and any(st.picked[m] == below for m in mks)):
                    st.holding.discard(cell)
                    for m in mks:
                        if st.picked[m] == below:
                            st.picked[m] = None
                    self._consume(act, kinds, cell)

        # ---- weld a crate about to be stamped to the crates beside it.
        #      (The `fell1mark` twins of these rules are DEAD: they ask for
        #      `no xx`, and `xx` contains the very cursor that must match.)
        for cell in held("cursor0a", "cursor0b"):
            if (cell in st.holding and "crate1mark" in self.slot_marks_for(st, cell)
                    and not self._has_x1(st, cell) and cell not in st.num0):
                for d in ("up", "down", "left", "right"):
                    nb = _add(cell, d)
                    if self.inb(nb) and nb in st.s["crate1"]:
                        st.w1.add(frozenset((cell, nb)))

        # ---- stamp. Three passes, because the interpreter runs the whole
        #      `placemark` rule, then the whole placement block, then the whole
        #      decrement block -- so two cursors stamping on the same press both
        #      read the PRE-stamp board (which is how ps:vext_edit level 23 ends
        #      up with two Cursor0a rather than an a and a b).
        pending = []
        for kinds, mk, blue, nums in ((("cursor0a", "dead0a"), "a0", False, st.num0),
                                      (("cursor0b",), "b0", False, st.num0),
                                      (("cursor2a",), "a2", True, st.num2),
                                      (("cursor2b",), "b2", True, st.num2)):
            for cell in held(*kinds):
                marks = self.slot_marks_for(st, cell)
                slot = st.picked[mk]
                if not (cell in st.holding and marks and slot is not None):
                    continue
                if blue:
                    if self._has_xx(st, cell) or cell in st.num0 or cell in st.num2:
                        continue
                    if slot not in st.num2:
                        continue
                else:
                    if self._has_x1(st, cell) or cell in st.num0:
                        continue
                    if slot not in st.num0:
                        continue
                pending.append((cell, marks, slot, mk, blue, kinds))
                self._consume(act, kinds, cell)
        # `down [placemark cursor0mark][curse0a|no one2]` is judged once, on the
        # board as it stood before any of this press's stamps landed.
        as_b = any(st.num2.get(_add(x, "down")) != 1 for x in st.curse0a())
        for cell, marks, _slot, _mk, _blue, _kinds in pending:
            self._place(st, cell, marks, as_b)
        for cell, _marks, slot, mk, blue, _kinds in pending:
            self._decrement(st, slot, mk, cell, blue)

        # ---- mode switches
        self._mode_switches(st, act, held)

    @staticmethod
    def _consume(act, kinds, cell):
        for k in kinds:
            act.discard((k, cell))

    def slot_marks_for(self, st, cell):
        """The preview(s) drawn at ``cell``.

        ``showhere`` is stamped on EVERY cursor of each kind, so a cell holding
        two cursors (a blue one parked on a red one) shows both previews and the
        later one evicts the earlier on any shared layer."""
        by_layer = {}
        srcs = []
        if cell in st.curse0a():
            srcs.append("a0")
        if cell in st.s["cursor0b"]:
            srcs.append("b0")
        if cell in st.s["cursor2a"]:
            srcs.append("a2")
        if cell in st.s["cursor2b"]:
            srcs.append("b2")
        for which in srcs:
            for mk in self.slot_marks(st, which):
                by_layer[_MARK_LAYER[mk]] = mk
        return frozenset(by_layer.values())

    #: Collision layers a stamped object shares, so placing evicts.
    _LAYER_OF = {
        "player1": "P1", "wall1": "P1", "crate1": "P1",
        "cursor0a": "C0", "cursor0b": "C0", "wall0": "C0", "crate0": "C0",
        "fell1": "F1", "ded1": "F1",
        "dead0a": "D0", "fell0": "D0",
        "pit1": "PIT1", "pit0": "PIT0",
        "target1": "TGT", "playertarget1": "TGT", "target0": "TGT",
    }

    def _put(self, st, cell, obj):
        layer = self._LAYER_OF.get(obj)
        if layer:
            for other, lay in self._LAYER_OF.items():
                if lay == layer and other != obj:
                    st.s[other].discard(cell)
        st.s[obj].add(cell)

    def _place(self, st, cell, marks, as_b=False):
        """``[placemark <x>mark] -> [placemark <x> <x>mark]``, in file order."""
        for mk, obj in PLACE_OF.items():
            if mk not in marks:
                continue
            if obj in NUMS:
                st.num0[cell] = NUMS[obj]
            elif obj in NUMS2:
                st.num2[cell] = NUMS2[obj]
            else:
                self._put(st, cell, obj)
        if "cursor0mark" in marks:
            self._put(st, cell, "cursor0b" if as_b else "cursor0a")

    def _decrement(self, st, slot, mk, cursor_cell, blue):
        nums = st.num2 if blue else st.num0
        val = nums.get(slot)
        if val is None:
            return
        if val == 1:
            # the stock ran out: the palette entry itself is deleted, and the
            # cursor stops holding anything
            src = _add(slot, "up")
            if self.inb(src):
                self._strip_slot(st, src, blue)
            del nums[slot]
            key = {"a0": "cursor0a", "b0": "cursor0b",
                   "a2": "cursor2a", "b2": "cursor2b"}[mk]
            if cursor_cell in st.s[key]:
                st.holding.discard(cursor_cell)
                st.picked[mk] = None
        elif val == 0:
            pass                                   # inf0 never decrements
        else:
            nums[slot] = val - 1

    def _strip_slot(self, st, src, blue):
        """Empty the palette slot when its last unit is spent.

        EVERY pickable goes, not one: the rule that does it
        (``down [xmark holding cursor2a][pickable2|one2 decrement pickeda2] ->
        [...][|one2 decrement pickeda2]``) binds a single or-group member per
        application, but the interpreter re-runs a rule until it stops matching,
        and the slot still matches while anything pickable is left in it. It
        shows up only where a slot stacks two of them -- ps:vext_edit's "HOLD"
        parks a Cursor0a on top of an inf0, i.e. a palette entry that is itself a
        palette entry."""
        for n in PICKABLE:
            st.s[n].discard(src)
        if blue:
            for n in PICKABLE2_EXTRA:
                st.s[n].discard(src)
        st.num0.pop(src, None)
        st.num2.pop(src, None)

    def _mode_switches(self, st, act, held):
        ghost_block = st.ghost0
        # ---- playing -> editing
        cands = []
        for flag, piece in (("active", "player1"), ("activefell", "ded1")):
            for cell in held(flag):
                if (cell in st.s[flag] and cell in st.s[piece] and self.partbc
                        and not (ghost_block and (cell in st.s["wall0"]
                                                  or cell in st.s["crate0"]))):
                    cands.append(cell)
        if len(cands) > 1 and not st.curse0a():
            self.ambiguous = True
        for flag, piece in (("active", "player1"), ("activefell", "ded1")):
            for cell in held(flag):
                if cell not in st.s[flag] or cell not in st.s[piece]:
                    continue
                if not self.partbc:
                    continue
                if ghost_block and (cell in st.s["wall0"] or cell in st.s["crate0"]):
                    continue
                self._put(st, cell, "cursor0b" if st.curse0a() else "cursor0a")
                st.s[flag].discard(cell)
                st.s["activecurse0"].add(cell)
                self._consume(act, (flag,), cell)

        # ---- editing -> playing (holding first, then empty-handed)
        for kinds, mk in ((("cursor0a", "dead0a"), "a0"), (("cursor0b",), "b0")):
            for cell in held(*kinds):
                if cell not in st.s["activecurse0"] or cell not in st.holding:
                    continue
                if not self.slot_marks_for(st, cell):
                    continue
                slot = st.picked[mk]
                if slot is None or slot not in st.num0:
                    continue
                if cell in st.s["player1"]:
                    st.s["active"].add(cell)
                elif cell in st.s["ded1"]:
                    st.s["activefell"].add(cell)
                else:
                    continue
                for k in kinds:
                    st.s[k].discard(cell)
                st.s["activecurse0"].discard(cell)
                st.holding.discard(cell)
                st.picked[mk] = None
                self._consume(act, kinds, cell)
        for cell in held("cursor0a", "cursor0b"):
            if cell not in st.s["activecurse0"] or cell in st.holding:
                continue
            if cell in st.s["player1"]:
                st.s["active"].add(cell)
            elif cell in st.s["ded1"]:
                st.s["activefell"].add(cell)
            else:
                continue
            st.s["cursor0a"].discard(cell)
            st.s["cursor0b"].discard(cell)
            st.s["activecurse0"].discard(cell)
            self._consume(act, ("cursor0a", "cursor0b"), cell)

        # ---- red -> blue
        if self.partc:
            for kind, mk, blue in (("cursor0a", "a0", "cursor2a"),
                                   ("cursor0b", "b0", "cursor2b")):
                for cell in held(kind):
                    if cell not in st.s["activecurse0"] or cell not in st.s[kind]:
                        continue
                    if (cell in st.holding and self.slot_marks_for(st, cell)
                            and st.picked[mk] is not None):
                        st.holding.discard(cell)
                        st.picked[mk] = None
                    st.s[blue].add(cell)
                    st.s["activecurse0"].discard(cell)
                    self._consume(act, (kind,), cell)

        # ---- blue -> red
        blues = [cell for kind in ("cursor2a", "cursor2b") for cell in held(kind)
                 if cell in st.s[kind] and (cell in st.s["cursor0a"]
                                            or cell in st.s["cursor0b"])]
        # `[action curse2 curse0 ...][cursor0a] -> [activecurse0 cursor0b][cursor0a]`
        # is applied once per (bracket1, bracket2) PAIR, and whether a cursor
        # keeps its name depends on whether the pair happened to bind it to
        # ITSELF -- which is decided by set-iteration order, i.e. by nothing a
        # model should pretend to mirror. With one blue and one Cursor0a the
        # pairing is forced and the outcome is well defined; past that it is not.
        if len(blues) > 1 or (blues and len(st.s["cursor0a"]) > 1):
            self.ambiguous = True
        for kind in ("cursor2a", "cursor2b"):
            for cell in held(kind):
                if cell not in st.s[kind]:
                    continue
                on = ("cursor0a" if cell in st.s["cursor0a"]
                      else "cursor0b" if cell in st.s["cursor0b"] else None)
                if on is None:
                    continue
                if cell in st.holding:
                    if not self.slot_marks_for(st, cell):
                        continue
                    slot = st.picked["a2"]
                    if slot is None or slot not in st.num2:
                        continue
                    st.holding.discard(cell)
                    st.picked["a2"] = None
                # `[action curse2 curse0 no holding][cursor0a] -> [activecurse0
                # cursor0b][cursor0a]` looks like it renames the cursor, and it
                # does not: `cursor0a` is a CONCRETE object, so the engine's
                # overlap preference restricts the second bracket to a cell the
                # first bracket already claimed whenever one exists, and its RHS
                # puts the cursor0a straight back. The rename therefore only
                # happens through the `random` fallback, i.e. only when no
                # cursor0a is left on the board at all. (ps:vext_edit level 23
                # drives two blue cursors onto two red ones on one key, which is
                # what exposed this.)
                st.s[kind].discard(cell)
                if not st.s["cursor0a"]:
                    st.s[on].discard(cell)
                    self._put(st, cell, "cursor0a")
                st.s["activecurse0"].add(cell)
                self._consume(act, (kind,), cell)

    # ---- ghosts --------------------------------------------------------
    def _ghost_phase(self, st):
        if st.s["active"] & st.s["player1"]:
            st.ghost0 = True
        if st.cursors0():
            st.ghost0 = False
        if st.s["activecurse0"] & st.cursors0():
            st.ghost2 = True
        if st.s["cursor2a"] or st.s["cursor2b"]:
            st.ghost2 = False

    # ---- movement ------------------------------------------------------
    LAYERS = {
        "P1": ("player1", "wall1", "crate1"),
        "C0": ("cursor0a", "cursor0b", "wall0", "crate0"),
        "F1": ("fell1", "ded1"),
        "D0": ("dead0a", "fell0"),
        "C2": ("cursor2a", "cursor2b"),
    }

    def _movement_phase(self, st, d):
        F = set()
        for kind in ("active", "activefell", "activecurse0", "dead0a",
                     "cursor2a", "cursor2b"):
            for cell in st.s[kind]:
                F.add((kind, cell))

        # (move attached items)
        for cell in st.s["activecurse0"]:
            for k in ("cursor0a", "cursor0b"):
                if cell in st.s[k]:
                    F.add((k, cell))
        for cell in st.s["active"]:
            if cell in st.s["player1"]:
                F.add(("player1", cell))
        for cell in st.s["activefell"]:
            if cell in st.s["ded1"]:
                F.add(("ded1", cell))
        for k in ("cursor0a", "cursor0b", "cursor2a", "cursor2b"):
            for cell in st.s[k]:
                if (k, cell) in F and cell in st.holding:
                    F.add(("holding", cell))

        cant0, cant1, fc0, fc1 = self._cant(st, F, d)

        # stationary
        for kind, blocked in (("cursor0a", cant0), ("cursor0b", cant0),
                              ("player1", cant1), ("ded1", fc1),
                              ("dead0a", fc0)):
            for cell in list(st.s[kind]):
                if cell in blocked:
                    F.discard((kind, cell))
        for kind in ("cursor2a", "cursor2b"):
            for cell in st.s[kind]:
                nb = _add(cell, d)
                if nb in self.wall2 or nb in self.black:
                    F.discard((kind, cell))

        self._propagate(st, F, d, fc1)

        # attachments follow the entity's stationary-ness
        for cell in st.s["active"]:
            if ("player1", cell) not in F:
                F.discard(("active", cell))
        for cell in st.s["activecurse0"]:
            if not any((k, cell) in F for k in ("cursor0a", "cursor0b")):
                F.discard(("activecurse0", cell))
                F.discard(("holding", cell))
        for cell in st.s["activefell"]:
            if ("ded1", cell) not in F:
                F.discard(("activefell", cell))
        for k in ("cursor2a", "cursor2b"):
            for cell in st.s[k]:
                if (k, cell) not in F:
                    F.discard(("holding", cell))

        self._resolve(st, F, d)

    def _cant(self, st, F, d):
        cant0 = set(self.black) | set(st.s["wall0"])
        cant1 = (set(self.black) | set(st.s["wall1"])
                 | set(st.num0) | set(st.num2))
        fc1 = {c for c in self.cells if c not in st.s["pit1"]}
        fc0 = {c for c in self.cells if c not in st.s["pit0"]}
        mov_ded = any(("ded1", c) in F for c in st.s["ded1"])
        mov_p1 = any(("player1", c) in F for c in st.s["player1"])
        mov_c0 = any((k, c) in F for k in ("cursor0a", "cursor0b")
                     for c in st.s[k])
        mov_d0 = any(("dead0a", c) in F for c in st.s["dead0a"])
        w1 = [tuple(e) for e in st.w1]
        wf = [tuple(e) for e in st.wf]
        while True:
            n0, n1, f0, f1 = len(cant0), len(cant1), len(fc0), len(fc1)
            if mov_ded:
                for kind in ("fell1", "ded1"):
                    for cell in st.s[kind]:
                        if _add(cell, d) in fc1:
                            fc1.add(cell)
            for a, b in wf:
                if a in fc1 or b in fc1:
                    fc1.add(a)
                    fc1.add(b)
            if mov_p1:
                for kind in ("crate1", "player1"):
                    for cell in st.s[kind]:
                        if _add(cell, d) in cant1:
                            cant1.add(cell)
            for a, b in w1:
                if a in cant1 or b in cant1:
                    cant1.add(a)
                    cant1.add(b)
            for cell in st.s["crate1"] & st.s["crate0"]:
                if cell in cant0 or cell in cant1:
                    cant0.add(cell)
                    cant1.add(cell)
            if mov_d0:
                for cell in st.s["dead0a"]:
                    if _add(cell, d) in fc0:
                        fc0.add(cell)
            if mov_c0:
                for kind in ("cursor0a", "cursor0b"):
                    for cell in st.s[kind]:
                        if _add(cell, d) in cant0:
                            cant0.add(cell)
                for cell in st.s["crate0"]:
                    nb = _add(cell, d)
                    if nb in cant0 or nb in st.num0:
                        cant0.add(cell)
            if (len(cant0), len(cant1), len(fc0), len(fc1)) == (n0, n1, f0, f1):
                return cant0, cant1, fc0, fc1

    def _propagate(self, st, F, d, fc1):
        # [ > Ded1 | Fell1 no fellcant1 ]
        changed = True
        while changed:
            changed = False
            for cell in st.s["ded1"]:
                if ("ded1", cell) not in F:
                    continue
                nb = _add(cell, d)
                if nb in st.s["fell1"] and nb not in fc1 and ("fell1", nb) not in F:
                    F.add(("fell1", nb))
                    changed = True
        # the '+' group: cursor0/crate0 and player1/crate1 push chains + welds
        pairs = (("cursor0a", "crate0"), ("cursor0b", "crate0"),
                 ("cursor0a", "cursor0a"), ("cursor0a", "cursor0b"),
                 ("cursor0b", "cursor0a"), ("cursor0b", "cursor0b"),
                 ("crate0", "crate0"),
                 ("player1", "crate1"), ("player1", "player1"),
                 ("crate1", "player1"), ("crate1", "crate1"))
        w1 = [tuple(e) for e in st.w1]
        changed = True
        while changed:
            changed = False
            for src, dst in pairs:
                for cell in st.s[src]:
                    if (src, cell) not in F:
                        continue
                    nb = _add(cell, d)
                    if nb in st.s[dst] and (dst, nb) not in F:
                        F.add((dst, nb))
                        changed = True
            for a, b in w1:
                for x, y in ((a, b), (b, a)):
                    if (("crate1", x) in F and y in st.s["crate1"]
                            and ("crate1", y) not in F):
                        F.add(("crate1", y))
                        changed = True
        # fell1 chains + fell welds
        wf = [tuple(e) for e in st.wf]
        changed = True
        while changed:
            changed = False
            for cell in st.s["fell1"]:
                if ("fell1", cell) not in F:
                    continue
                nb = _add(cell, d)
                if nb in st.s["fell1"] and ("fell1", nb) not in F:
                    F.add(("fell1", nb))
                    changed = True
            for a, b in wf:
                for x, y in ((a, b), (b, a)):
                    if (("fell1", x) in F and y in st.s["fell1"]
                            and ("fell1", y) not in F):
                        F.add(("fell1", y))
                        changed = True
        # [moving crate1 crate0ghost] -> the crate drags the ghosted layer-0 box
        if st.ghost0:
            for cell in st.s["crate1"] & st.s["crate0"]:
                if ("crate1", cell) in F:
                    F.add(("crate0", cell))

    def _resolve(self, st, F, d):
        """Chain resolution. Every force this turn points the same way, so a
        maximal RUN of movers moves iff the cell past its head is free of that
        run's collision layer (or the run walks off the board, which blocks)."""
        moved = {}
        occs = [{cell: k for k in kinds for cell in st.s[k]}
                for kinds in self.LAYERS.values()]
        for occ in occs:
            mov = {cell for cell, k in occ.items() if (k, cell) in F}
            for cell in mov:
                run = [cell]
                x = cell
                while True:
                    nb = _add(x, d)
                    if not self.inb(nb):
                        ok = False
                        break
                    if nb not in occ:
                        ok = True
                        break
                    if nb in mov:
                        run.append(nb)
                        x = nb
                        continue
                    ok = False
                    break
                if ok:
                    for y in run:
                        moved[(occ[y], y)] = _add(y, d)
        for kind in ("active", "activefell", "activecurse0"):
            for cell in st.s[kind]:
                if (kind, cell) in F and self.inb(_add(cell, d)):
                    moved[(kind, cell)] = _add(cell, d)
        for cell in list(st.holding):
            if ("holding", cell) in F and self.inb(_add(cell, d)):
                moved[("holding", cell)] = _add(cell, d)
        if not moved:
            return
        for (kind, cell), dest in moved.items():
            if kind in st.s:
                st.s[kind].discard(cell)
            elif kind == "holding":
                st.holding.discard(cell)
        for (kind, cell), dest in moved.items():
            if kind in st.s:
                st.s[kind].add(dest)
            elif kind == "holding":
                st.holding.add(dest)
        shift = {cell for (kind, cell) in moved if kind == "crate1"}
        st.w1 = {frozenset((_add(a, d) if a in shift else a,
                            _add(b, d) if b in shift else b))
                 for a, b in (tuple(e) for e in st.w1)}
        shiftf = {cell for (kind, cell) in moved if kind == "fell1"}
        st.wf = {frozenset((_add(a, d) if a in shiftf else a,
                            _add(b, d) if b in shiftf else b))
                 for a, b in (tuple(e) for e in st.wf)}

    # ---- late ----------------------------------------------------------
    def _late_phase(self, st):
        """Support, drowning, falling, weld upkeep. True when the turn cancels."""
        sup = set()
        for cell in st.s["crate1"]:
            if (cell in st.s["ded1"] or cell not in st.s["pit1"]
                    or cell in st.s["fell1"]):
                sup.add(cell)
        w1 = [tuple(e) for e in st.w1]
        while True:
            n = len(sup)
            for a, b in w1:
                if a in sup or b in sup:
                    sup.add(a)
                    sup.add(b)
            if len(sup) == n:
                break

        # late [cursor0 pit0 no fell0] -> [pit0 dead0a]
        for kind in ("cursor0a", "cursor0b"):
            for cell in list(st.s[kind]):
                if cell in st.s["pit0"] and cell not in st.s["fell0"]:
                    st.s[kind].discard(cell)
                    st.s["fell0"].discard(cell)
                    st.s["dead0a"].add(cell)
        # late [player1 active pit1 no fell1 no ded1] -> [pit1 ded1 activefell]
        for cell in list(st.s["player1"]):
            if (cell in st.s["pit1"] and cell not in st.s["fell1"]
                    and cell not in st.s["ded1"]):
                st.s["player1"].discard(cell)
                st.s["ded1"].add(cell)
                if cell in st.s["active"]:
                    st.s["active"].discard(cell)
                    st.s["activefell"].add(cell)
        # late [crate0 pit0 no supported] -> [pit0 fell0]
        for cell in list(st.s["crate0"]):
            if cell in st.s["pit0"] and cell not in sup:
                st.s["crate0"].discard(cell)
                st.s["dead0a"].discard(cell)
                st.s["fell0"].add(cell)
        # late [crate1 pit1 no supported] -> [pit1 fell1]
        fallen = set()
        for cell in list(st.s["crate1"]):
            if cell in st.s["pit1"] and cell not in sup:
                st.s["crate1"].discard(cell)
                st.s["ded1"].discard(cell)
                st.s["fell1"].add(cell)
                fallen.add(cell)
        if fallen:
            for e in list(st.w1):
                a, b = tuple(e)
                if a in fallen and b in fallen:
                    st.wf.add(e)
        self.prune_welds(st)

        # late [player1 ded1] -> cancel
        return bool(st.s["player1"] & st.s["ded1"])

    # ---- compact keys --------------------------------------------------
    #: Cell sets some rule can create or destroy no matter what the palette
    #: holds. Everything else is static unless the level's palette can stamp it.
    ALWAYS_DYNAMIC = ("player1", "crate1", "fell1", "ded1", "cursor0a",
                      "cursor0b", "dead0a", "crate0", "fell0",
                      "cursor2a", "cursor2b",
                      "active", "activefell", "activecurse0")

    def setup_key(self, st):
        """Decide which cell sets this level can actually change, and freeze the
        rest into a template. Keys are then ~5x smaller, which is the difference
        between a level whose space enumerates and one that runs out of RAM."""
        dyn = set(self.ALWAYS_DYNAMIC)
        numbery = False
        for slot in list(st.num0) + list(st.num2):
            src = _add(slot, "up")
            if not self.inb(src):
                continue
            for mk in self.slot_mark_types(st, src):
                obj = PLACE_OF.get(mk)
                if obj in NUMS or obj in NUMS2:
                    numbery = True
                if obj and obj in st.s:
                    dyn.add(obj)
            for n in CELLSETS:
                if src in st.s[n]:
                    dyn.add(n)
        if numbery:
            # A palette that stamps DIGITS can mint new palette slots under
            # anything, so nothing on the board is static any more. That is the
            # whole of ps:vext_edit's "PLAN".
            dyn = set(CELLSETS)
        self.dyn = tuple(n for n in CELLSETS if n in dyn)
        self.static = tuple(n for n in CELLSETS if n not in dyn)
        self.nb = (self.h * self.w + 7) // 8
        self.template = st.clone()

    def slot_mark_types(self, st, src):
        by_layer = {}
        for obj, mk in MARK_OF:
            if obj == "cursor0":
                hit = src in st.cursors0()
            elif obj in NUMS:
                hit = st.num0.get(src) == NUMS[obj]
            elif obj in NUMS2:
                hit = st.num2.get(src) == NUMS2[obj]
            else:
                hit = src in st.s.get(obj, ())
            if hit:
                by_layer[_MARK_LAYER[mk]] = mk
        return frozenset(by_layer.values())

    def _mask(self, cells):
        w, m = self.w, 0
        for r, c in cells:
            m |= 1 << (r * w + c)
        return m.to_bytes(self.nb, "little")

    def _cells(self, blob):
        w = self.w
        m = int.from_bytes(blob, "little")
        out = set()
        i = 0
        while m:
            if m & 1:
                out.add(divmod(i, w))
            m >>= 1
            i += 1
        return out

    def pack(self, st):
        w = self.w
        out = bytearray()
        for n in self.dyn:
            out += self._mask(st.s[n])
        out += self._mask(st.holding)
        for d in (st.num0, st.num2):
            for cell, v in sorted(d.items()):
                out += (cell[0] * w + cell[1]).to_bytes(2, "little")
                out += bytes((v,))
            out += b"\xff\xff"
        for ws in (st.w1, st.wf):
            for e in sorted(tuple(sorted(x)) for x in ws):
                for cell in e:
                    out += (cell[0] * w + cell[1]).to_bytes(2, "little")
            out += b"\xff\xff"
        for k in ("a0", "b0", "a2", "b2"):
            p = st.picked[k]
            out += (65535 if p is None else p[0] * w + p[1]).to_bytes(2, "little")
        out += bytes((st.ghost0 * 2 + st.ghost2,))
        return bytes(out)

    def unpack(self, blob):
        w = self.w
        st = self.template.clone()
        i = 0
        for n in self.dyn:
            st.s[n] = self._cells(blob[i:i + self.nb])
            i += self.nb
        st.holding = self._cells(blob[i:i + self.nb])
        i += self.nb
        for d in ("num0", "num2"):
            out = {}
            while blob[i:i + 2] != b"\xff\xff":
                idx = int.from_bytes(blob[i:i + 2], "little")
                out[divmod(idx, w)] = blob[i + 2]
                i += 3
            i += 2
            setattr(st, d, out)
        for name in ("w1", "wf"):
            out = set()
            while blob[i:i + 2] != b"\xff\xff":
                a = int.from_bytes(blob[i:i + 2], "little")
                b = int.from_bytes(blob[i + 2:i + 4], "little")
                out.add(frozenset((divmod(a, w), divmod(b, w))))
                i += 4
            i += 2
            setattr(st, name, out)
        for k in ("a0", "b0", "a2", "b2"):
            idx = int.from_bytes(blob[i:i + 2], "little")
            st.picked[k] = None if idx == 65535 else divmod(idx, w)
            i += 2
        flags = blob[i]
        st.ghost0 = bool(flags & 2)
        st.ghost2 = bool(flags & 1)
        return st

    # ---- win -----------------------------------------------------------
    def won(self, st):
        cr = st.s["crate1"] | st.s["fell1"]
        if not st.s["target1"] <= cr:
            return False
        pm = set(st.s["player1"])
        for cell in (st.curse0a() | st.s["cursor0b"]
                     | st.s["cursor2a"] | st.s["cursor2b"]):
            if "player1mark" in self.slot_marks_for(st, cell):
                pm.add(cell)
        if not st.s["playertarget1"] <= pm:
            return False
        if not st.s["target0"] <= st.s["crate0"]:
            return False
        cur = st.cursors0()
        if cur & (st.s["target1"] | st.s["playertarget1"]):
            return False
        return True
