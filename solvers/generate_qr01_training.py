"""Generate Phase-1 training data for the QR01 game (games/qr01/qr01.py).

QR01 ("Quad Twist") is a *mouse-click* tile-permutation puzzle. The 8x8 board is
carved into rectangular **panels** of free tiles (everything else is wall); an
ACTION6 click on cell ``(gx, gy)`` rotates the 2x2 block anchored there clockwise,
so the four tile colours cycle ``(gx,gy) -> (gx+1,gy) -> (gx+1,gy+1) -> (gx,gy+1)``.
A block that leaves the grid or touches a wall is a free no-op. The level is won the
moment every free cell matches the target pattern drawn in the top-right reference
panel. Colours are only ever permuted, so the target is always a rearrangement of
the board -- see the game docstring for the per-seed layout/colour/scramble
augmentation and for why the original five hand-written levels (two of which were
outright impossible) were replaced.

Solver: an exact distance oracle per panel
------------------------------------------
Panels are separated by at least one wall cell, so no 2x2 block ever spans two of
them: each panel is an **independent** sub-puzzle, and the number of clicks left is
the SUM of the per-panel distances. A panel is small by construction (at most 16
cells over at most 4 colours, e.g. 3x3 over 4 colours = 7560 reachable colourings,
4x3 over 3 = 34650, 4x4 over 2 = 12870), and a click is an invertible permutation of
its cells, so `_dist_map` simply BFSes the panel's whole orbit *backwards from its
target* (applying the inverse, counter-clockwise, permutations) and gets the exact
distance of every reachable state. That single BFS -- memoised per
``(w, h, target)`` and so paid once per panel per episode -- then answers everything:

  * `solve_from` walks the distance gradient down to zero, one panel at a time;
  * `optimal_set_from` returns EVERY globally-optimal next click (any panel's
    distance-reducing click is globally optimal, because the panels are additive),
    which is the optimal-action-SET target the policy trains on and is what makes
    the base's stochastic-optimal sampling real here rather than one canonical route;
  * recovery is total: reachability is symmetric (three more clicks on an anchor undo
    one), so every state the exploration prefix or a perturbation burst can reach is
    in the map with an exact distance. There is no state this solver cannot solve
    from, which is why ``supports_recovery`` / ``recovery_mode = "replan"`` hold
    unconditionally.

The board and target are read off the live game (``_g`` / ``_target``); both are
fully visible on screen (the playfield and the reference panel), so this is a
convenience, not privileged information.

Display rotation is NOT pinned: ``Qr01`` is an `AugmentedGame`, so each level is
recorded at its live ``random_rotation_k(seed, level)`` orientation. Rotation is a
pure display transform -- the plan is computed in game space and only the *clicks*
move, which `_click` handles by inverse-rotating each cell centre.

Action schema (mixed simple + mouse, matching ft09 / cn04 / cd82)
    RESET / simple :  {"type": "simple", "index": k}
    mouse click    :  {"type": "mouse",  "index": 6, "data": {"x": x, "y": y}}

Usage (run from the repo root, with the ARC-AGI-3 conda python):
    python solvers/generate_qr01_training.py --episodes 1000 \
        --out data/training_multi_level/qr01
    python solvers/generate_qr01_training.py --verify 25    # engine-replay check
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from games.qr01.qr01 import Qr01                             # noqa: E402
from solvers.base_solver import BaseSolver                    # noqa: E402
from utils.explore import Action, CLICK_ACTION                # noqa: E402
from utils.rotation import remap_click                        # noqa: E402

# Hard stop on the backward BFS. Every panel this game builds has an orbit far
# below it (the largest is 34650 states), so the cap only exists so a future,
# bigger panel degrades to "solver declines, base records a RESET" instead of
# eating all the memory on the machine.
_NODE_CAP = 400_000


# ── panel move tables ───────────────────────────────────────────────────────
def _move_table(w: int, h: int):
    """For a ``w x h`` panel: one entry per rotatable anchor, as
    ``((ax, ay), cw, ccw)`` where ``cw``/``ccw`` are source-index tuples --
    ``tuple(state[i] for i in cw)`` applies the clockwise rotation.

    Clockwise moves the *contents* BL -> TL -> TR -> BR -> BL, so the cell that
    ENDS at TL is the one that started at BL, and so on; ``ccw`` is that inverted.
    """
    table = []
    for ay in range(h - 1):
        for ax in range(w - 1):
            tl = ay * w + ax
            tr = tl + 1
            bl = tl + w
            br = bl + 1
            cw = list(range(w * h))
            cw[tl], cw[tr], cw[br], cw[bl] = bl, tl, tr, br
            ccw = list(range(w * h))
            ccw[bl], ccw[tl], ccw[tr], ccw[br] = tl, tr, br, bl
            table.append(((ax, ay), tuple(cw), tuple(ccw)))
    return table


class Qr01Solver(BaseSolver):
    game_id = "qr01"
    # An exact distance oracle over each panel's whole orbit: `solve_from` reads the
    # LIVE board and returns an optimal plan from ANY reachable state, so an
    # exploratory detour or a burst is just a different (still solvable) start.
    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self._moves: dict[tuple[int, int], list] = {}
        self._dist: dict[tuple, dict[tuple, int]] = {}

    # ── engine plumbing ─────────────────────────────────────────────────────
    def make_game(self, seed: int):
        # Distance maps are per (panel shape, target) and so per seed; drop them at
        # the episode boundary to keep the sweep's memory flat.
        self._dist.clear()
        return Qr01(seed=seed)

    def available_actions(self, game) -> list[int]:
        # The four movement keys are advertised by the game and do nothing; keeping
        # them in the exploration pool is what puts null transitions in the corpus.
        return [1, 2, 3, 4, CLICK_ACTION]

    def _click(self, game, gx: int, gy: int) -> Action:
        """Centre-of-cell UPRIGHT click for board cell ``(gx, gy)``.

        Deliberately NOT converted to screen space here. Every solver plans in the core
        game's upright space and the base does the one conversion -- it submits the
        upright action through ``_set_action`` (which bypasses the rotation wrapper) and
        records the screen action. Converting here as well double-rotated the click and
        landed it on the wrong cell at k=1/2/3."""
        scale, offx, offy = game.camera._calculate_scale_and_offset()
        px = gx * scale + offx + scale // 2
        py = gy * scale + offy + scale // 2
        return Action(CLICK_ACTION, (py, px))          # click_rc = (row=y, col=x)

    # ── the oracle ──────────────────────────────────────────────────────────
    def _move_table_for(self, w: int, h: int):
        table = self._moves.get((w, h))
        if table is None:
            table = self._moves[(w, h)] = _move_table(w, h)
        return table

    def _dist_map(self, w: int, h: int, target: tuple) -> dict[tuple, int]:
        """``{panel colouring: minimum clicks to reach ``target``}``, for every
        colouring in the target's orbit. BFS backwards from the target over the
        inverse (counter-clockwise) moves: ``s`` is one clockwise click from ``t``
        exactly when ``t``'s counter-clockwise image is ``s``."""
        key = (w, h, target)
        dist = self._dist.get(key)
        if dist is not None:
            return dist
        table = self._move_table_for(w, h)
        dist = {target: 0}
        frontier = [target]
        depth = 0
        while frontier and len(dist) < _NODE_CAP:
            depth += 1
            nxt = []
            for s in frontier:
                for _anchor, _cw, ccw in table:
                    t = tuple(s[i] for i in ccw)
                    if t not in dist:
                        dist[t] = depth
                        nxt.append(t)
            frontier = nxt
        self._dist[key] = dist
        return dist

    @staticmethod
    def _panel_states(game, box) -> tuple[tuple, tuple]:
        """``(current, target)`` colourings of one panel, in row-major order."""
        x0, y0, w, h = box
        cur = tuple(game._g[y0 + dy][x0 + dx]
                    for dy in range(h) for dx in range(w))
        tgt = tuple(game._target[y0 + dy][x0 + dx]
                    for dy in range(h) for dx in range(w))
        return cur, tgt

    def _downhill(self, table, dist, cur, d):
        """Every ``(anchor, successor)`` that takes colouring ``cur`` (at distance
        ``d``) one click closer to the target."""
        out = []
        for anchor, cw, _ccw in table:
            nxt = tuple(cur[i] for i in cw)
            if dist.get(nxt, -1) == d - 1:
                out.append((anchor, nxt))
        return out

    def _panel_best(self, game, box):
        """``(distance, [(anchor, successor state), ...])`` for one panel: its exact
        remaining click count and every click that reduces it. ``None`` if the panel's
        state is outside the (capped) oracle."""
        _x0, _y0, w, h = box
        cur, tgt = self._panel_states(game, box)
        dist = self._dist_map(w, h, tgt)
        d = dist.get(cur)
        if d is None:
            return None
        table = self._move_table_for(w, h)
        best = self._downhill(table, dist, cur, d) if d else []
        if d and not best:                     # cannot happen with an exact map
            return None
        return d, best

    # ── BaseSolver hooks ────────────────────────────────────────────────────
    def solve_from(self, game, level_idx: int, seed: int):
        """An optimal click plan from the game's CURRENT board: walk each panel's
        distance gradient to zero. Panel order (and each tie among equally-optimal
        clicks) is drawn from ``self.rng``, so repeated episodes of the same seed
        take different -- still minimum-length -- routes."""
        boxes = list(game._panels)
        self.rng.shuffle(boxes)
        plan: list[Action] = []
        for box in boxes:
            x0, y0, w, h = box
            cur, tgt = self._panel_states(game, box)
            dist = self._dist_map(w, h, tgt)
            table = self._move_table_for(w, h)
            d = dist.get(cur)
            if d is None:
                return []                      # outside the oracle -> let the base RESET
            while d:
                step = self._downhill(table, dist, cur, d)
                if not step:
                    return []
                (ax, ay), cur = self.rng.choice(step)
                plan.append(self._click(game, x0 + ax, y0 + ay))
                d -= 1
        return plan

    def optimal_set_from(self, game, level_idx: int, seed: int):
        """Every globally-optimal next click. Distances are additive over panels, so
        a click that reduces ANY panel's distance by one reduces the total by one."""
        acts: list[Action] = []
        for box in game._panels:
            got = self._panel_best(game, box)
            if got is None:
                return None                    # fall back to the plan head
            _d, best = got
            for (ax, ay), _nxt in best:
                acts.append(self._click(game, box[0] + ax, box[1] + ay))
        return acts

    # ── verification ────────────────────────────────────────────────────────
    def verify(self, seeds: int, start_seed: int = 0) -> int:
        """Replay each recorded episode's actions into a FRESH game through the real
        ``perform_action`` and require a WIN -- the only check with teeth for a
        rotation-bearing generator (a plan recorded at the wrong orientation wins
        roughly the 1-in-4 levels whose k happens to be 0)."""
        from arcengine import ActionInput, GameAction, GameState

        from utils.rotation import random_rotation_k

        id_to_action = {int(a.value): a for a in GameAction}
        bad = 0
        for seed in range(start_seed, start_seed + seeds):
            ok, levels = self.solve_episode(seed, explore=True)
            if not ok:
                print(f"  seed {seed}: NOT SOLVED")
                bad += 1
                continue
            game = self.make_game(seed)
            game.set_level(0)
            clicks = plan_len = 0
            for lvl in levels:
                acts = self.normalize_levels([lvl])[0]["actions"]
                for rec in acts[1:]:
                    if rec["index"] == 0:
                        ai = ActionInput(id=GameAction.RESET)
                    elif rec["index"] == CLICK_ACTION:
                        ai = ActionInput(id=GameAction.ACTION6, data=rec["data"])
                        clicks += 1
                    else:
                        ai = ActionInput(id=id_to_action[rec["index"]])
                    game.perform_action(ai)
                plan_len += len(acts) - 1
            if game._state != GameState.WIN:
                print(f"  seed {seed}: replay ended in {game._state} "
                      f"(expected WIN) after {plan_len} actions")
                bad += 1
            else:
                rots = [random_rotation_k(seed, i) for i in range(len(levels))]
                print(f"  seed {seed}: WIN  actions={plan_len} clicks={clicks} "
                      f"rot={rots}")
        print(f"{seeds - bad}/{seeds} seeds replay to WIN")
        return 1 if bad else 0

    @classmethod
    def build_argparser(cls):
        p = super().build_argparser()
        p.add_argument("--verify", type=int, default=0,
                       help="Replay N seeds through the engine and assert WIN "
                            "instead of writing a corpus.")
        return p

    @classmethod
    def main(cls, argv=None) -> int:
        import random
        args = cls.build_argparser().parse_args(argv)
        if args.verify:
            solver = cls(rng=random.Random(args.seed),
                         burst_prob=args.noise, burst_mean=args.burst)
            return solver.verify(args.verify, start_seed=args.start_seed)
        return super().main(argv)


if __name__ == "__main__":
    sys.exit(Qr01Solver.main())
