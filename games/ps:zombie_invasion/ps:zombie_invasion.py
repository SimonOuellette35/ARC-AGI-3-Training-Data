"""PuzzleScript game: zombie_invasion

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Zombie_Invasion.txt

Play with: python solver_client.py ps:zombie_invasion
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game.

    Two recolors, both of which `solvers/generate_zombie_invasion_training.py
    --audit` checks at the 4px cells these levels render at.

    * **The Wall was INVISIBLE.** It is declared `GREEN` and the Background
      `darkgreen`, and the ARC 16-colour palette has exactly one green (14).
      Both objects are solid 5x5s, so the wall ring that encloses every level
      rendered pixel-identical to the floor it encloses -- the board looked
      like an unbounded field, and the one press that is a genuine no-op
      (walking into the border) looked like the world had frozen for no reason.
      Recoloured to blue: unused by anything else here (player pink, zombie
      grey, goal black, floor green).
    * **The Goal was the letterbox.** `_render_frame` pads the 64x64 frame out
      to the board's aspect with colour 5 -- black -- which is exactly the
      Goal's own colour, so the one square the game asks you to reach rendered
      in the same colour as the dead border around the play area. Recoloured to
      yellow, which nothing else uses; the win frame (the pink player's
      transparent pixels showing yellow through) is then unmistakable.
    """
    adapter = PuzzleScriptAdapter("Zombie_Invasion", seed=seed)

    wall = adapter._game.objects.get("wall")
    if wall is not None and wall.colors:
        wall.colors[0] = wall.dominant_color = 9                  # blue
    goal = adapter._game.objects.get("goal")
    if goal is not None and goal.colors:
        goal.colors[0] = goal.dominant_color = 11                 # yellow

    adapter._do_reset()
    return adapter
