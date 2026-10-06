"""One implementation of the repo's per-(seed, level) presentation augmentation.

THE ROTATION LIVES AT THE GAME'S BOUNDARY, AND NOWHERE ELSE::

    agent ---- screen action ----> [de-rotate] ----> core game (upright)
    agent <--- rotated frame ----- [rotate]    <---- core game (upright)

`AugmentedGame` is a WRAPPER. The core game inside it is never exposed to rotation: it
builds its board, caches coordinates off it, and bakes in layout assumptions ("the slots
run to the right of the frame", "respawn at the start cell") in one fixed upright space,
exactly as it was written to. A game cannot have a rotation bug if it cannot see
rotation, so no game file needs any rotation code at all.

Two designs were tried and rejected before this one, both because they put the rotation
somewhere that made it EVERY caller's problem:

  * a ``RotationDisplay`` camera interface turned the frame while the board stayed
    upright, and each game had to un-rotate its own input. Most games never did, so they
    showed a turned board and took upright keys.
  * turning the level DATA put every game INSIDE the rotated world, so each baked-in
    assumption became its own bug to hunt (sb26's slot row, tb01's respawn,
    projectile_dodge's wall grid, stale coordinate caches).

Keep the rotation in the two wrapper methods below. If a fix ever seems to need rotation
logic inside a game, the wrapper is wrong -- fix the wrapper.

Every game in ``games/`` is an ``ARCBaseGame``. The augmentation each of them needs --
draw a display rotation for this (seed, level), render through it, and map screen input
back to game space -- was copy-pasted into each game instead, and it drifted:

  * seven games got it right, but spelled the seed three ways (``_seed_value``,
    ``_base_seed``) and the level index two (``level_index``,
    ``_current_level_index``);
  * four games (cn04, vc33, ka59, lf52) accept a ``seed=`` and use it for colours and
    layout, but drew the rotation from ``random_rotation_k()`` -- the UNSEEDED global
    fallback -- so their orientation was not reproducible from the seed;
  * six games (m0r0, r11l, sb26, tr87, ft09, s5i5) had no seed at all.

The consequence was measurable: generators recorded demos at a pinned k=0 while live play
showed a random orientation, so ~3/4 of evaluation boards were in an orientation absent
from the whole training corpus. Subclassing ``AugmentedGame`` is what stops that class of
bug from being reintroduced one game at a time.

Usage in a game::

    class Xx(AugmentedGame):
        def __init__(self, seed: int | None = None) -> None:
            self._init_augmentation(seed)      # FIRST -- see the note below
            ...
            camera = Camera(..., interfaces=[self._hud])   # no rotation interface
            super().__init__("xx", levels, camera, ...)

        def on_set_level(self, level):
            ...     # NO rotation code anywhere in the game, and no input remapping in
                    # step(): the wrapper de-rotated the action before step() ran, and
                    # rotates the frame after the camera renders it.

``_init_augmentation`` must run before ``super().__init__``, because the base
constructor calls ``set_level`` -- which is where the rotation is drawn.
"""
from __future__ import annotations

import random

from arcengine import ARCBaseGame

# Imported as a MODULE, not `from ... import random_rotation_k`, so the draw is looked
# up late. `game_envs._load_local_env` re-executes each game file on every `make_env`,
# so a game's own `from utils.rotation import ...` re-binds every time and therefore
# sees a monkeypatched `random_rotation_k`; this module is cached and imported once, so
# a by-name import would freeze whatever the name meant at first import and silently
# ignore the patch -- pinning the rotation in a test harness would then render every
# level at the first level's k. Test harnesses pin `utils.rotation.random_rotation_k`;
# keep this indirection so they still work.
from utils import rotation as _rotation

# Every ARC-AGI-3 frame is 64x64, and clicks are addressed in THAT space -- not in
# ``camera.width``, which is the camera's GRID size (16 for sp80, 10 for cq01, 8 for
# gp01). De-rotating a click inside a 16-wide box lands it somewhere else entirely.
_FRAME_SIZE = 64


class _RotatedRender:
    """The camera-level render hook, as an OBJECT rather than a closure.

    It must survive ``copy.deepcopy`` of the game, because the generators clone games
    constantly (burst snapshot/restore in `solvers.base_solver`, blackbox search in a
    dozen solvers). A closure would NOT: ``deepcopy`` treats function objects as atoms
    and returns them unchanged, so a cloned camera kept a wrapper still bound to the
    ORIGINAL camera and the ORIGINAL game -- every frame rendered off the clone was
    then drawn from the original camera's interfaces (its HUD/UI state) at the
    original game's rotation. For a game that paints its board through a camera
    interface (tb01's bridges + player, and every other ``RenderableUserDisplay``)
    that means the clone renders a GHOST of the state the original happens to be in.

    Holding the game and the camera as plain attributes makes the wrapper an ordinary
    object: ``deepcopy`` copies it, and its memo re-points ``game``/``camera`` at
    whichever pair the copy belongs to. ``type(self.camera).render`` (rather than a
    bound method captured at install time) reaches the engine's own renderer without
    re-entering this hook.
    """

    __slots__ = ("game", "camera")

    def __init__(self, game, camera) -> None:
        self.game = game
        self.camera = camera

    def __call__(self, sprites):
        cam = self.camera
        frame = type(cam).render(cam, sprites)
        k = self.game._rotation_k % 4
        if not k:
            return frame
        import numpy as _np
        return _np.rot90(_np.asarray(frame), k).copy()


# ── on_set_level idempotency ────────────────────────────────────────────────
def clear_dynamic_sprites(level, *, tags=(), names=()) -> None:
    """Drop the actor sprites a PREVIOUS ``on_set_level`` pass left on ``level``.

    ``ARCBaseGame.set_level`` does NOT clone the level -- it re-seats the same
    persistent ``Level`` object and calls ``on_set_level`` again. So any game that
    builds its movable actors there (``level.add_sprite(player)``) silently
    DUPLICATES them every time the level is seated twice, and the base constructor
    already seats level 0 once: a generator's ``BaseSolver.set_level`` (or a
    viewer's jump-to-level) is the second seat. The duplicates stack at the same
    start cell, so frame 0 looks correct and the ghost only becomes visible once
    the live actor moves off it -- and, if the actor is collidable, it also blocks
    its own start cell for the rest of the level.

    Call this FIRST in ``on_set_level``, naming the sprites that ``on_set_level``
    itself adds (never ones baked into the level template)::

        def on_set_level(self, level):
            clear_dynamic_sprites(level, tags=["player", "box"])
            level.add_sprite(_player_sprite(...))
    """
    doomed = []
    for s in level.get_sprites():
        if s.name in names or any(t in s._tags for t in tags):
            doomed.append(s)
    for s in doomed:
        level.remove_sprite(s)


def seed_game(game, seed: int | None) -> bool:
    """Install ``seed`` on a freshly-constructed game; True if it took.

    The single choke point `game_envs._load_local_env` calls so that EVERY native
    game ends up seeded, not just the ones whose ``__init__`` happens to declare a
    ``seed`` parameter. A game that is not an `AugmentedGame` has no augmentation
    state to seed and is reported as False -- which is the signal that it still
    needs migrating.
    """
    if seed is None or not isinstance(game, AugmentedGame):
        return False
    game.adopt_seed(seed)
    return True


class AugmentedGame(ARCBaseGame):
    """``ARCBaseGame`` wrapped in a deterministic per-(seed, level) rotation.

    Frames go out turned by k; actions and clicks come in turned back by -k. The core
    game only ever sees its own upright space and needs no rotation code of its own.
    """

    _seed_value: int | None = None
    _rotation_k: int = 0

    # ── construction ────────────────────────────────────────────────────────
    def _init_augmentation(self, seed: int | None = None) -> None:
        """Prepare the augmentation state. Call FIRST in the subclass ``__init__``.

        ``seed=None`` keeps the legacy behaviour of an unseeded global draw, so a
        bare ``Xx()`` still varies run to run for interactive play; pass a seed and
        every level's orientation becomes a pure function of (seed, level_index).
        """
        self._seed_value = seed
        self._rotation_k = 0

    # ── rotation ────────────────────────────────────────────────────────────
    def set_level(self, index: int) -> None:
        """Draw this level's rotation, then hand off to the engine.

        ``set_level`` is the single choke point for every level change -- the
        constructor's first seat, ``_really_set_next_level``, ``level_reset``,
        ``full_reset``, a generator jumping straight to a level -- so drawing k here
        means every path gets the right orientation. The level itself is untouched.
        """
        self._refresh_rotation(index)
        self._install_render_hook()
        super().set_level(index)
        # Clear any pending level-advance flag. The engine only clears ``_next_level``
        # inside ``_really_set_next_level`` (the normal in-play advance); a generator
        # that jumps straight to a level with ``set_level`` would otherwise inherit a
        # stale ``_next_level=True`` left over from the previous level's win, making the
        # next level report an immediate (false) win on its first action.
        self._next_level = False

    def _refresh_rotation(self, level_index: int) -> None:
        self._rotation_k = _rotation.random_rotation_k(self._seed_value, level_index)

    # ── boundary 1: observation out ─────────────────────────────────────────
    def _install_render_hook(self) -> None:
        """Rotate EVERY frame the camera produces, once, at the camera itself.

        Hooking ``camera.render`` rather than ``perform_action`` is deliberate: plenty
        of code renders directly (``game.camera.render(level.get_sprites())`` all over
        the generators and viewers) and would otherwise see an upright board while the
        agent sees a turned one -- two coordinate spaces again, by the back door.

        The hook is a `_RotatedRender` object, not a closure, so that a deep-copied
        game renders through ITS OWN camera; see that class.
        """
        cam = self.camera
        if cam is None:
            return
        hook = cam.__dict__.get("render")
        if isinstance(hook, _RotatedRender):
            # Already hooked. Re-point it at the live pair anyway: a hand-rolled
            # restore that rebuilt ``game.__dict__`` (base_solver's burst UNDO) can
            # leave a wrapper copied from a clone.
            hook.game, hook.camera = self, cam
            return
        cam.render = _RotatedRender(self, cam)
        cam._augment_render_hooked = True

    # ── boundary 2: action in ───────────────────────────────────────────────
    def perform_action(self, action_input, raw: bool = False):
        """De-rotate the incoming action, then hand it to the core game untouched.

        The agent presses and clicks in SCREEN space -- the space the rotated frame
        showed it. The game underneath only ever thinks in its own upright space, so
        both the direction and the click coordinate are mapped back here, at the single
        point every action enters through.
        """
        if self._rotation_k % 4:
            action_input = self._augment_derotate_action(action_input)
        return super().perform_action(action_input, raw)

    def _augment_derotate_action(self, action_input):
        """Screen ActionInput -> the equivalent one in the core game's upright space."""
        k = self._rotation_k % 4
        action_input.id = _rotation.remap_action(action_input.id, k)
        data = getattr(action_input, "data", None)
        if data and "x" in data and "y" in data:
            x, y = _rotation.remap_click(int(data["x"]), int(data["y"]), k, _FRAME_SIZE)
            data["x"], data["y"] = x, y
        return action_input

    def adopt_seed(self, seed: int | None) -> None:
        """Install the augmentation seed on an ALREADY-CONSTRUCTED game.

        ``game_envs._load_local_env`` can only pass ``seed=`` to a game whose
        ``__init__`` declares the parameter; every other game was silently built
        unseeded, so its rotation came from the global fallback draw and no amount
        of ``random.seed()`` in the harness could pin it. This is the catch-all:
        call it right after construction and the CURRENT level's orientation is
        re-drawn as a pure function of (seed, level).

        It cannot retroactively fix layout randomness a game already consumed in
        its own ``__init__`` -- for that the game must accept ``seed=`` and draw
        through `level_rng`. Rotation, which is drawn per level and re-drawn on
        every `set_level`, it fixes completely.
        """
        if seed is None:
            return
        self._seed_value = int(seed)
        self._refresh_rotation(getattr(self, "_current_level_index", 0))

    # ── seeded randomness ───────────────────────────────────────────────────
    def level_rng(self, tag: str = "") -> random.Random:
        """A FRESH RNG, deterministic in (seed, current level, ``tag``).

        Use this instead of ``random.Random()`` for anything a level's content
        depends on -- layout, colours, hidden assignments. Two properties matter:

        * **Fresh per call, keyed on the level.** A game that keeps one
          long-lived ``self._rng`` makes level N's board depend on which levels
          were visited before it, because that is what determines the stream
          position. So ``make_env(g, seed, 3)`` (a direct jump) and playing
          0->1->2->3 produce DIFFERENT level 3s, and a recorded demo cannot be
          replayed. Keying on the level index removes the path dependence.
        * **``tag`` separates independent draws** so adding a new one does not
          shift every existing stream (pass e.g. ``"layout"`` / ``"colors"``).

        With no seed (bare ``Xx()`` for interactive play) it returns an unseeded
        RNG, preserving the old varies-every-run behaviour.
        """
        if self._seed_value is None:
            return random.Random()
        return random.Random(f"{tag}:{self._seed_value}:"
                             f"{getattr(self, '_current_level_index', 0)}")

    @property
    def rotation_k(self) -> int:
        """The rotation the CURRENT level is being displayed at.

        Informational -- for logging, or for a generator that needs to convert the
        upright plan it just built into the screen actions it should RECORD.
        """
        return self._rotation_k

    # ── deprecated: the wrapper already did this ────────────────────────────
    #
    # These used to convert a screen press back to game space, once per game, in each
    # game's own ``step``. `perform_action` now does it for every game at the single
    # point actions enter through, so by the time a game sees ``self.action`` it is
    # ALREADY in game space. They are kept as identities so the games that still call
    # them keep working -- dropping the call is a tidy-up, not a fix.
    #
    # Do NOT restore a real conversion here: that would de-rotate a second time, and
    # the double-remap is silent (it looks right at k=0 and k=2, wrong at k=1 and k=3).
    def screen_click_to_game(self, x: int, y: int) -> tuple[int, int]:
        """Identity. `perform_action` already de-rotated this click."""
        return (int(x), int(y))

    def screen_action_to_game(self, action_id):
        """Identity. `perform_action` already de-rotated this action."""
        return action_id
