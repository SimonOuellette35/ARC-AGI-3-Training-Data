"""Generate Phase-1 training data for the PuzzleScript game ps:dark_maze_3
("Dark Maze 3" by Adam Gates) -- a PARTIALLY OBSERVABLE maze search.

Each solved seed yields one single-level episode JSON in the shared schema::

    {
      "game_id": "puzzlescript_dark_maze_3",
      "levels": [
        {"level_id": 0,
         "observations": [[[c, ...], ...], ...],   # [T, 64, 64] palette idx
         "actions":      [a_0, ..., a_{T-1}]},     # length T
      ]
    }

``actions[0]`` is the RESET that produced ``observations[0]``; ``actions[i]`` for
i>=1 took the agent from ``observations[i-1]`` to ``observations[i]``. The
recorded index is the *screen* action (post rotation remap), so replaying the
recorded actions reproduces the recorded frames exactly.


THE GAME
========
One level, generated afresh from the seed. The board is a 25x17 engine grid
holding a 12x8 maze: cells sit at (odd, odd), the wall positions between them at
(odd, even) / (even, odd), and inert pillars at (even, even). At level start
``run_rules_on_level_start`` fires a randomized-Prim carve --
``random [New | Wall | No New] -> [ | random Opening | ]`` -- which drops one
player, one exit, and turns 95 of the 172 wall positions into an ``Opening``,
i.e. a spanning tree over the 96 cells. An Opening is a ``Hall`` three times in
four and a ``Door`` the fourth; both are on the collision layer BELOW the player,
so **both are walkable** -- the difference is that a Door is opaque and a Hall is
not (its sprite is all transparent, so a Hall renders as plain floor). Movement
is stock PuzzleScript with no rules attached, the prelude says ``noaction``, so
the whole game is four arrow keys, and the win condition is
``All Exit on Player``: walk onto the exit.

What makes it a puzzle is that you cannot see the maze. Every turn the late
rules repaint the WHOLE board black and then carve a torch beam out of it::

    late [ ] -> [ Dark ]
    late HORIZONTAL [ | Player | ] -> [ GazeH | Player | GazeH ]
    late HORIZONTAL [ GazeH No Wall No Door | Dark ] -> [ GazeH | GazeH ]
    late VERTICAL   [ | GazeH | ] -> [ No Dark | | No Dark ]
    ... and the same three rules again with the axes swapped

so what is visible is the run of cells the player can see along its row and
along its column (each run ending ON the first Wall or Door, which is lit but
stops the beam), plus the cells immediately perpendicular to that cross.
Everything else is ``Dark``, which renders as ARC palette 5 -- the same grey as
the letterbox border -- so an unexplored board is literally not there.

Two consequences that drive the whole solver, both established by reproducing
the beam exactly against the interpreter (a model of the rules above predicted
the lit set on 1600/1600 steps over 40 seeds):

  * **The four cells around the player are ALWAYS lit** -- the two GazeH and two
    GazeV seeds are placed unconditionally, even into walls. So the agent always
    learns its own local connectivity, and dead-reckoning a move is exact: it
    already knows whether the cell it is stepping into is a wall.
  * **The player is sometimes INVISIBLE.** The beam is a flood, not a ray: a
    lit cell that is not a Wall/Door spreads to any adjacent Dark cell -- back
    towards the player included. So the player's own cell is lit iff at least
    one of its four neighbours is not a Wall and not a Door. Stand in a cell
    whose only ways out are Doors and the blue square is swallowed by the dark
    (2.5% of steps; 2 of 40 seeds start that way). It is still localizable --
    the player is then the one dark cell whose four neighbours are all lit --
    and after the first step dead reckoning carries it.

The 64x64 frame is therefore a small window of five colours floating in grey:
black walls (0), green floor (3), brown doors (12), the blue player (9) and the
yellow exit (11).

KNOWN ENGINE DEVIATION (not fixed here). ``_execute_rule`` applies a ``random``
rule to the first of ``["up","down","left","right"]`` that has a match, instead
of drawing one match uniformly across all four as PuzzleScript does. The carve
rule is directionless, so every vertical link is opened before the first
horizontal one is considered, and the maze comes out as a comb -- 8 fully open
columns joined by exactly 7 rungs -- on every seed (88 vertical + 7 horizontal
openings, measured over 200 seeds). It is still a perfect maze with a genuinely
hidden layout (which rungs, where the doors are, where the exit is), and the
expert below neither knows nor cares about the bias; but a fix in the engine
would give this game properly varied mazes. It is left alone because the same
code path is shared with other shipped games (Four Colour Theorem among them),
whose corpora would silently change.


THE EXPERT: FRONTIER SEARCH OVER WHAT HAS BEEN SEEN
===================================================
This is a partially observable game and it is solved as one -- see
[[no-privileged-solvers]]. The expert never reads a cell the frame does not
show: `FrameReader` skips every cell carrying ``Dark`` and classifies the rest
by their top rendering object, and `DarkMazeExpert`'s map starts empty and only
ever grows from what has been lit.

Two modes, decided by whether the exit has been seen yet:

  * **Search.** Frontier exploration: BFS from the player over the cells known
    to be walkable, and walk to a NEAREST cell that still has an unseen
    neighbour. Standing on such a cell necessarily lights that neighbour (the
    four cells around the player are always lit), so every arrival strictly
    grows the map and the search terminates.
  * **Approach.** Once the exit has been lit, BFS to it over the known cells.

The WALK is exactly optimal in both modes, and in this game "shortest known
route" is also "shortest route": the carve produces a spanning tree, the set of
cells lit in one turn is connected and contains the player, so the known region
is always a CONNECTED SUBTREE -- and a path between two nodes of a tree is
unique. There is no shorter way round hiding in the dark (verified: 200/200
seeds carve a spanning tree). So the approach phase is optimal outright, and the
search phase is an optimal walk to a greedily chosen frontier -- which is the
honest ceiling, since no policy can know which frontier hides the exit.

The optimal SET recorded as the target is every direction that starts a shortest
path to some nearest target. In the approach phase that is always a single
direction (unique tree path); in the search phase it is a real tie whenever two
frontiers sit the same distance away in different directions (~2.5% of steps).


RECOVERY  (``recovery_mode = "replan"``)
========================================
Nothing in this game is irreversible: there is no death, no lose condition, no
one-way door, and every move can be undone by pressing the opposite arrow. A
map-building expert is the ideal replan-mode expert -- it is driven by
(action, observation) pairs and does not care who chose the action, so an
exploration step is not damage to be undone, it is a free observation that makes
the map bigger. Concretely:

  * the episode-wide exploration PREFIX runs at level start and is simply
    absorbed: no RESET is ever recorded, because there is nothing to recover
    from;
  * perturbation BURSTS are kept for the same reason, labelled with the target
    the expert would have chosen at that state.

Both prefix and burst steps carry the expert's own optimal set as ``optimal``,
so every recorded step has a target to train on.

The one failure mode left is the step budget, and the game folder's
`DarkMaze3Adapter` sizes it for a blind search (900, ~3x the measured worst
case) rather than the adapter's 200-step default -- which a searching agent
would blow through on roughly a quarter of seeds before ever seeing the exit.
"""

from __future__ import annotations

import collections
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcengine import ActionInput, GameAction, GameState            # noqa: E402
from solvers.base_solver import BaseSolver, _ID_TO_GAMEACTION       # noqa: E402
from solvers.common.ps_astar import _load_game_module, screen_action  # noqa: E402
from utils.explore import (ExplorationPolicy, ExplorationPrefix,    # noqa: E402
                           RESET_ACTION)
from utils.rotation import remap_action_full                        # noqa: E402

GAME_NAME = "Dark_Maze_3"
GAME_MODULE = "ps:dark_maze_3"

DIRECTIONS = ("up", "down", "left", "right")
DELTA = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
ACTION_TO_DIR = {GameAction.ACTION1: "up", GameAction.ACTION2: "down",
                 GameAction.ACTION3: "left", GameAction.ACTION4: "right",
                 GameAction.ACTION5: "action"}

#: What a lit cell can be. ``BLOCK`` is the only one the player cannot enter;
#: ``DOOR`` is walkable but opaque, ``FREE`` covers plain floor and the
#: transparent ``Hall`` openings alike (they render identically, and the
#: distinction does not exist for a viewer of the frame).
BLOCK, FREE, DOOR, EXIT = "block", "free", "door", "exit"
WALKABLE = frozenset({FREE, DOOR, EXIT})

#: ``Dark`` is what hides the board, so it MUST render as a solid opaque cell.
#: If a future sprite edit ever punched a hole in it, the objects underneath
#: would leak into the frame and reading the grid the way `FrameReader` does
#: would stop matching what a viewer sees.
MUST_STAY_OPAQUE = frozenset({"dark"})

#: Level-start / mid-turn bookkeeping objects that must all be consumed before a
#: frame is rendered. ``New`` is the carve marker, ``GazeH``/``GazeV`` are the
#: torch beam; all three are opaque red or opaque, so one surviving into an
#: observed frame would both repaint the board and invalidate the cell
#: classification below. Asserted on every read rather than assumed.
MUST_BE_CONSUMED = frozenset({"new", "gazeh", "gazev"})


# ---------------------------------------------------------------------------
# Observation: everything a viewer of the 64x64 frame can see, and nothing else
# ---------------------------------------------------------------------------

@dataclass
class Obs:
    """One frame's worth of *renderable* facts.

    ``kind`` holds an entry only for the cells the torch beam lit this turn --
    the rest of the board is painted ``Dark`` and is, for a viewer, not there.
    ``player`` is None on the turns the player's own cell is dark (see the
    module docstring); that is a real hidden state, not an omission."""

    kind: dict                  #: lit cell -> BLOCK / FREE / DOOR / EXIT
    player: object              #: the player's cell, or None when it is dark
    height: int
    width: int


class FrameReader:
    """Turns the interpreter's grid into an `Obs` using only what renders.

    The honesty contract is one rule: a cell holding ``Dark`` is skipped
    entirely, because ``Dark`` is an opaque black fill that covers everything
    below it -- exactly what the frame shows. Lit cells are then classified by
    their top rendering object, walking the collision layers downwards
    (Dark/Gaze > Player/Wall > Exit/Opening > Background)."""

    def __init__(self, game) -> None:
        self.game = game
        ps = game._game
        self._idx = dict(ps.obj_name_to_idx)
        missing = sorted({"dark", "wall", "fixedwall", "door", "hall",
                          "player", "exit"} - set(self._idx))
        if missing:
            raise RuntimeError(f"{GAME_NAME}: objects {missing} are gone -- "
                               "re-audit FrameReader before generating")
        for name in sorted(MUST_STAY_OPAQUE):
            obj = ps.objects[name]
            opaque = (obj.sprite is None or
                      all(px >= 0 for row in obj.sprite for px in row))
            if not (opaque and obj.dominant_color >= 0):
                raise RuntimeError(
                    f"{GAME_NAME}: '{name}' no longer hides the cell it covers, "
                    "so reading the grid would expose unexplored board")
        self._consumed = [self._idx[n] for n in sorted(MUST_BE_CONSUMED)
                          if n in self._idx]

    def read(self) -> Obs:
        eng = self.game._engine
        grid = eng.grid
        i = self._idx
        dark, wall, fixed = i["dark"], i["wall"], i["fixedwall"]
        door, exit_, player = i["door"], i["exit"], i["player"]
        kind, ppos = {}, None
        for r, row in enumerate(grid):
            for c, cell in enumerate(row):
                for leftover in self._consumed:
                    if leftover in cell:
                        raise RuntimeError(
                            f"{GAME_NAME}: bookkeeping object survived into a "
                            f"rendered frame at {(r, c)} -- the beam model and "
                            "the cell classification are both invalid")
                if dark in cell:
                    continue                      # painted out: not observable
                if wall in cell or fixed in cell:
                    kind[(r, c)] = BLOCK
                elif player in cell:
                    kind[(r, c)] = FREE           # the player draws over its cell
                    ppos = (r, c)
                elif exit_ in cell:
                    kind[(r, c)] = EXIT
                elif door in cell:
                    kind[(r, c)] = DOOR
                else:
                    kind[(r, c)] = FREE
        return Obs(kind=kind, player=ppos, height=eng.height, width=eng.width)


# ---------------------------------------------------------------------------
# The expert: a map of what has been lit, and a search over it
# ---------------------------------------------------------------------------

class DarkMazeExpert:
    """Frontier search over the cells the torch beam has revealed so far.

    Holds two pieces of state, both derived only from observations: ``known``
    (cell -> kind, monotone) and ``pos`` (where the player is). It is re-usable
    across a level via `start_level`."""

    def __init__(self, game) -> None:
        self.reader = FrameReader(game)
        self.start_level()

    # -- state ------------------------------------------------------------
    def start_level(self) -> None:
        self.known: dict = {}
        self.pos = None
        self.obs = None
        self.observe(None)

    def observe(self, direction: str | None) -> None:
        """Fold the current frame into the map. ``direction`` is the move that
        was just made (None at level start), used to dead-reckon through the
        turns where the player's own cell is dark."""
        predicted = self._predict(direction)
        obs = self.reader.read()
        self.obs = obs
        self.known.update(obs.kind)
        self.pos = self._locate(obs, predicted)
        # Standing somewhere proves it is walkable, which matters on exactly the
        # turns the player is invisible: the cell is dark, so nothing else says
        # so, and without it the search would treat its own position as a wall.
        if self.pos is not None and self.known.get(self.pos) is None:
            self.known[self.pos] = FREE

    def _predict(self, direction: str | None):
        """Where the previous position moves to under ``direction``.

        Exact: the four cells around the player are always lit, so the cell
        being stepped into was classified last turn and a bump into a wall is
        known to be a no-op."""
        if direction is None or self.pos is None or direction not in DELTA:
            return None
        dr, dc = DELTA[direction]
        ahead = (self.pos[0] + dr, self.pos[1] + dc)
        return ahead if self.known.get(ahead) in WALKABLE else self.pos

    def _locate(self, obs: Obs, predicted):
        if obs.player is not None:
            if predicted is not None and predicted != obs.player:
                # The frame disagrees with dead reckoning, so one of the two
                # things it rests on is wrong: the beam always lighting all four
                # neighbours, or a bump into a wall being a no-op. Both are
                # load-bearing on the turns the player is invisible, so stop
                # rather than record a trajectory labelled from a broken model.
                raise RuntimeError(
                    f"{GAME_NAME}: dead reckoning said {predicted} but the "
                    f"player is drawn at {obs.player} -- movement model broken")
            return obs.player
        if predicted is not None and predicted not in obs.kind:
            return predicted            # dark cell, as an invisible player must be
        # First turn of a level with the player already swallowed by the dark:
        # it is the one unlit cell whose every in-bounds neighbour is lit (the
        # beam seeds all four, so nothing else on the board looks like that).
        holes = [p for p in self._dark_cells(obs)
                 if all(n in obs.kind for n in self._neighbours(p, obs))]
        if len(holes) == 1:
            return holes[0]
        return None

    @staticmethod
    def _neighbours(cell, obs: Obs):
        for dr, dc in DELTA.values():
            r, c = cell[0] + dr, cell[1] + dc
            if 0 <= r < obs.height and 0 <= c < obs.width:
                yield (r, c)

    @staticmethod
    def _dark_cells(obs: Obs):
        return [(r, c) for r in range(obs.height) for c in range(obs.width)
                if (r, c) not in obs.kind]

    # -- policy -----------------------------------------------------------
    def decide(self) -> tuple[str, list]:
        """``(mode, directions)`` at the current state.

        ``mode`` is ``"approach"`` once the exit has been seen and ``"search"``
        while it has not; ``directions`` is the full optimal tie set, empty when
        there is nothing left to do (which cannot happen on a connected maze and
        so fails the level rather than guessing)."""
        if self.pos is None:
            return "lost", []
        dist = self._bfs([self.pos])
        targets = [p for p, k in self.known.items() if k == EXIT and p in dist]
        mode = "approach"
        if not targets:
            targets = [p for p in dist if self._is_frontier(p)]
            mode = "search"
        if not targets:
            return "stuck", []
        best = min(dist[p] for p in targets)
        targets = [p for p in targets if dist[p] == best]
        # Backward BFS from the whole target set: a direction is optimal iff it
        # steps onto a cell one closer to SOME nearest target.
        back = self._bfs(targets)
        here = back.get(self.pos)
        if here is None:
            return "stuck", []
        dirs = [d for d in DIRECTIONS
                if back.get((self.pos[0] + DELTA[d][0],
                             self.pos[1] + DELTA[d][1]), 1 << 30) == here - 1]
        return (mode, dirs) if dirs else ("stuck", [])

    def _bfs(self, sources: list) -> dict:
        """Step distances from ``sources`` over the cells known to be walkable."""
        dist = {s: 0 for s in sources}
        queue = collections.deque(sources)
        while queue:
            cell = queue.popleft()
            for d in DIRECTIONS:
                n = (cell[0] + DELTA[d][0], cell[1] + DELTA[d][1])
                if n not in dist and self.known.get(n) in WALKABLE:
                    dist[n] = dist[cell] + 1
                    queue.append(n)
        return dist

    def _is_frontier(self, cell) -> bool:
        """A walkable cell with an unseen neighbour. Walking onto it lights that
        neighbour, so every frontier visit strictly grows the map."""
        obs = self.obs
        for dr, dc in DELTA.values():
            r, c = cell[0] + dr, cell[1] + dc
            if 0 <= r < obs.height and 0 <= c < obs.width and (r, c) not in self.known:
                return True
        return False


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _frame_to_list(frame) -> list:
    return np.asarray(frame).tolist()


class DarkMaze3Solver(BaseSolver):
    """`BaseSolver` for ps:dark_maze_3.

    Like the other ps: generators this drives an external interpreter rather
    than a native `ARCBaseGame`, and the expert is a policy over a knowledge
    state rather than a plan over a visible board -- so `solve_episode` is
    overridden and only the CLI (`main`/`run`) and the
    ``{"game_id", "levels": [...]}`` save schema come from the base."""

    game_id = "puzzlescript_dark_maze_3"
    game_name = GAME_NAME

    #: Hard cap on RECORDED presses per level -- prefix, bursts and expert moves
    #: alike, so a burst chain cannot run the game past the adapter's own limit.
    #: Measured: the expert alone needs a mean of 91 and a worst case of 288 over
    #: 200 seeds; with the default ``--noise 0.1`` bursts that becomes a mean of
    #: 148 and a worst case of 456 over 80 seeds. 700 leaves half again on top of
    #: that and still sits under the adapter's 900, so a run that is looping is
    #: abandoned here (and re-tried on the next seed) rather than dying of
    #: GAME_OVER mid-trajectory.
    max_steps: int = 700

    supports_recovery = True
    recovery_mode = "replan"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._game = None
        self._expert = None

    # -- plumbing ----------------------------------------------------------
    def make_game(self, seed: int):
        """The adapter this generator records against -- built through the game
        folder's ``make_game``, which raises the step limit to something a blind
        search can actually finish inside. That patched adapter is exactly what
        `game_envs` hands a live agent, so the frames taped here are the frames
        the agent sees."""
        return _load_game_module(GAME_MODULE).make_game(seed=seed)

    def solve_from(self, game, level_idx: int, seed: int) -> list:
        """Not used: this generator drives its own knowledge-state loop (the
        expert is a policy over what has been seen, not a plan over a visible
        board), so `solve_episode` is overridden and the base record loop never
        runs."""
        raise NotImplementedError(
            "dark_maze_3 records through its own frontier-search loop; "
            "see DarkMaze3Solver.solve_episode")

    def available_actions(self, game) -> list[int]:
        """Four arrows. The prelude says ``noaction``, so ACTION5 is a guaranteed
        no-op and would only pad the trajectory with null transitions."""
        return [1, 2, 3, 4]

    def _ensure(self, seed: int):
        if self._game is None:
            self._game = self.make_game(seed)
            self._expert = DarkMazeExpert(self._game)
        self._game._seed = seed        # maze draw + augmentation re-derive
        return self._game, self._expert

    # -- recording ---------------------------------------------------------
    @staticmethod
    def _step(action: GameAction, *, phase: str, changed: bool,
              optimal: list | None) -> dict:
        return {"type": "simple", "index": int(action.value), "phase": phase,
                "changed": bool(changed), "n_obs": 1,
                "optimal": None if optimal is None else
                [{"type": "simple", "index": int(a.value)} for a in optimal]}

    def _screen(self, game, direction: str) -> GameAction:
        return screen_action(direction, game._rotation_k, game._hflip,
                             game._vflip)

    def _engine_dir(self, game, action: GameAction) -> str:
        return ACTION_TO_DIR[remap_action_full(action, game._rotation_k,
                                               game._hflip, game._vflip)]

    def record_level(self, game, expert, level: int, *,
                     schedule=None, exploration=None):
        """Drive the level to a WIN with the frontier expert, recording as we go.

        Returns ``(observations, actions)``, or ``(None, None)`` if the level was
        not won (the step cap, or a board the expert could not localize on)."""
        game.set_level(level)
        expert.start_level()

        observations = [_frame_to_list(game._current_frame)]
        actions = [{"type": "simple", "index": RESET_ACTION, "phase": "reset",
                    "changed": False, "n_obs": 1, "optimal": None}]
        prev = np.asarray(game._current_frame)
        steps = 0

        def optimal_screen():
            """The expert's target at the CURRENT state, as screen actions."""
            _, dirs = expert.decide()
            return [self._screen(game, d) for d in dirs]

        def drive(action: GameAction, phase: str, optimal):
            nonlocal prev, steps
            steps += 1
            fd = game.perform_action(ActionInput(id=action))
            frame = np.asarray(fd.frame[-1] if fd.frame else game._current_frame)
            observations.append(_frame_to_list(frame))
            actions.append(self._step(action, phase=phase,
                                      changed=not np.array_equal(frame, prev),
                                      optimal=optimal))
            prev = frame
            expert.observe(self._engine_dir(game, action))

        # ---- exploration prefix (episode-wide budget) ----------------------
        # Nothing here is irreversible, so the detour is simply absorbed: it
        # feeds the map like any other step and no RESET is ever needed.
        while (schedule is not None and exploration is not None
               and schedule.explore()):
            schedule.advance()
            act = exploration.action(prev)
            if act.is_click:                     # this game exposes no click
                continue
            targets = optimal_screen()
            if not targets:
                return None, None
            drive(_ID_TO_GAMEACTION[act.action_id], "explore", targets)
            if game._state in (GameState.WIN, GameState.GAME_OVER):
                break

        # ---- expert loop ---------------------------------------------------
        # ``steps`` counts every recorded press, prefix and bursts included, so
        # the cap bounds the whole trajectory rather than only the expert's own
        # moves -- a burst chain must not be able to run the game past the
        # adapter's own step limit and lose the level to GAME_OVER.
        while steps < self.max_steps:
            if game._state == GameState.WIN:
                break
            if game._state == GameState.GAME_OVER:
                return None, None

            _mode, dirs = expert.decide()
            if not dirs:
                return None, None
            targets = [self._screen(game, d) for d in dirs]

            # Perturbation burst: a random walk is always survivable here (no
            # death, no one-way move), and it is more observations, so it is
            # kept and labelled rather than rolled back.
            if self._burst_prob > 0.0 and self.rng.random() < self._burst_prob:
                length = max(1, int(self.rng.expovariate(1.0 / self._burst_mean)))
                step_targets = targets
                for _ in range(length):
                    if not step_targets:
                        return None, None
                    drive(self._screen(game, self.rng.choice(DIRECTIONS)),
                          "burst", step_targets)
                    if game._state != GameState.NOT_FINISHED:
                        break
                    step_targets = optimal_screen()
                continue

            drive(self.rng.choice(targets), "expert", targets)

        if game._state != GameState.WIN:
            return None, None
        return observations, actions

    # -- episode -----------------------------------------------------------
    def solve_episode(self, seed: int, explore: bool = True):
        game, expert = self._ensure(seed)
        do_explore = explore and self.supports_recovery
        schedule = (ExplorationPrefix(self.rng, center=self._explore_center,
                                      jitter=self._explore_jitter)
                    if do_explore else None)
        exploration = (ExplorationPolicy(self.available_actions(game), self.rng)
                       if do_explore else None)

        out = []
        for level in range(game.n_levels):
            obs, acts = self.record_level(game, expert, level,
                                          schedule=schedule,
                                          exploration=exploration)
            if obs is None:
                return False, out
            out.append({"level_id": level, "observations": obs,
                        "actions": acts})
        return len(out) == game.n_levels, out


if __name__ == "__main__":
    raise SystemExit(DarkMaze3Solver.main())
