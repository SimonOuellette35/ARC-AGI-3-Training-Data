"""PuzzleScript game: mars_attacks

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Mars_Attacks.txt

Play with: python solver_client.py ps:mars_attacks
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    adapter = PuzzleScriptAdapter("Mars_Attacks", seed=seed)

    # The player is drawn in `lightgreen` and the wall in `Green Darkgreen
    # lightgreen` -- three greens the ARC 16-colour palette has exactly one
    # entry for. So the alien and the walls it walks between render in the SAME
    # colour (14), and the only thing separating the one object the agent
    # controls from the scenery is that the wall's 5x5 is solid while the
    # player's has two background-coloured eye pixels. That is a shape the
    # agent has to learn before it can even locate itself, on a board where
    # walls are the single most common cell.
    #
    # Recoloured to yellow: the nearest free hue to the `lightgreen` the
    # original was reaching for, and used by nothing else here (background is
    # black + blue, crate orange + maroon + grey, fruit white + grey). The
    # eyes stay black, so the alien is still the same glyph -- it is simply no
    # longer the same colour as the walls. `--audit` in
    # `solvers/generate_mars_attacks_training.py` checks every cell
    # composition stays pixel-distinct at the 9px cells these levels use.
    player = adapter._game.objects.get("player")
    if player is not None and player.colors:
        player.colors[0] = player.dominant_color = 11             # yellow

    adapter._do_reset()
    return adapter
