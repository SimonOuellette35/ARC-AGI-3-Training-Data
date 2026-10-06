"""PuzzleScript game: crocodiles_love_cookies

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Crocodiles_Love_Cookies.txt

Play with: python solver_client.py ps:crocodiles_love_cookies
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    adapter = PuzzleScriptAdapter("Crocodiles_Love_Cookies", seed=seed)

    # The goal's own colour is #a3a300 -- a dark yellow -- but its nearest ARC
    # palette entry is ORANGE, which is also where the cookie's `brown` lands.
    # Target and piece then render in the same colour, separated only by a
    # sparse 5x5 dot sprite that decimates to four corner pixels at the 4px
    # cells the later levels use: the square you are trying to fill and the
    # thing you fill it with become the same orange smudge. Recoloured to
    # yellow, which is what the original hex was reaching for and which nothing
    # else in this game uses. `--audit` in
    # `solvers/generate_crocodiles_love_cookies_training.py` checks every cell
    # composition stays pixel-distinct at every cell size the levels use.
    goal = adapter._game.objects.get("goal")
    if goal is not None and goal.colors:
        goal.colors[0] = goal.dominant_color = 11                 # yellow

    adapter._do_reset()
    return adapter
