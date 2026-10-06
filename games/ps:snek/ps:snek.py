"""PuzzleScript game: snek

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Snek.txt

Play with: python solver_client.py ps:snek
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: Where `_show_covered_under_the_head` puts its two extra dots, and the colour
#: index it paints them with (3 = Blue in Covered's own palette, the same colour
#: as the five dots the sprite already has).
_CORNERS = ((0, 4), (4, 0))
_BLUE = 3


def _show_covered_under_the_head(game) -> None:
    """Make the "this target is covered" dot visible under EVERY head facing.

    Snek's win condition is ``all target on obj``, and ``Covered`` is the marker
    the game paints on a target once something is standing on it -- so it is the
    only thing in the picture that says how close the level is to being won.

    Two things conspire to hide it. The game's own collision layers put
    ``covered`` on top, but `_render_frame` overrides that: it sorts player
    objects LAST, deliberately, "so the player is always visible on top". And
    two of the four head sprites -- ``pup`` and ``pright`` -- have no
    transparent pixel anywhere the five Covered dots land. A head facing up or
    right, standing on a target, is therefore drawn pixel-for-pixel like a head
    standing on bare floor.

    A Crate is worse: it is opaque over the whole cell, so a crate parked on a
    target renders exactly like a crate on bare floor and the dot was the only
    thing saying otherwise.

    That is not a cosmetic complaint. ``solvers/generate_snek_training.py
    --audit`` replays every level to its win and, for each target, re-renders
    the board with that one target's ``Covered`` mark removed. Without this fix
    the two frames are equal on EVERY level -- i.e. the frame each episode WINS
    on is identical to a frame that has not won. A corpus recorded that way
    teaches a goal that is not in the picture (the ps:g2020 / ps:silver_lungs
    failure). With it, all of them differ.

    The fix is two pixels. No single cell position is transparent in all four
    head sprites, but ``(4, 0)`` is clear in ``pup`` and ``pright`` and
    ``(0, 4)`` is clear in ``pdown`` and ``pleft``, so a dot in each opposite
    corner leaves at least one visible whichever way the head is pointing --
    and the corners clear the Crate sprite too. The original five dots are left
    exactly as they are, so this only ever ADDS a mark; nothing that used to be
    distinguishable stops being so.
    """
    covered = game._game.objects["covered"]
    for r, c in _CORNERS:
        covered.sprite[r][c] = _BLUE


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    game = PuzzleScriptAdapter("Snek", seed=seed)
    _show_covered_under_the_head(game)
    return game
