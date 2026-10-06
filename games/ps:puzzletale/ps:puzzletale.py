"""PuzzleScript game: puzzletale

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/PUZZLETALE.txt

Play with: python solver_client.py ps:puzzletale

Five UNDERTALE-flavoured puzzle rooms behind one win condition -- ``Some Player
on Goal`` -- and three different locks on the door:

  * **spikes (levels 0, 1, 2).** Two `late` rules re-derive the spike strip from
    scratch every turn, so it is up whenever ANY target is bare and down the
    instant the last one is covered. Spikes share the Player/Crate collision
    layer, so up means the only corridor to the exit is a wall: these rooms are
    a sokoban whose crates are the KEY, not the goal.
  * **bridge seeds (level 3).** ACTION facing a seed patch picks one up (one at
    a time); ACTION facing open water throws it in, and it floats the way you
    were facing until it hits something. Four at rest in a row germinate into a
    bridge. All of that -- the float, the collision, the germination -- resolves
    inside the single keypress that threw the seed, so the interpreter hands
    back a settled board.
  * **a crate bridge (level 4).** ``late [crate water no bridge] -> [bridge2
    water]``: shove a crate into the river and the crate BECOMES the plank. The
    second crate has to be pushed across the one the first made.

Facing is load-bearing in level 3 and it is drawn on the CHARACTER: ``player_l``
and ``player_r`` are pixel-identical sprites, so what separates "aimed left"
from "aimed right" on screen is the ``head_l`` / ``head_r`` face the game paints
in the cell ABOVE the player. That is also why the game is not in
`PuzzleScriptAdapter._FLIP_GAMES` -- ``head_u`` and ``head_d`` are hand-drawn
asymmetric faces rather than an orbit of the square's symmetry group, so a
mirror would show a facing sprite the game does not own. Rotation is safe (it
permutes the four facings among themselves), which is what the game gets.

No per-level step cap is needed here, and that is a checked fact rather than an
omission: `PuzzleScriptAdapter` gives every level 200 actions and flips to
GAME_OVER on the 201st, and the shortest plans are

    level      0   1   2   3   4
    presses   17  20  27  53  21

every one of them proved shortest by an exhaustive search over the interpreter.
``python solvers/generate_puzzletale_training.py --plans`` prints that row and
``--verify`` re-derives the proof.

The game FILE is unedited -- no sprite or palette fix was needed. ``--audit``
is the regression test for that: every cell composition the game can show
renders distinctly except the two purples it uses for plain floor, which are
the same thing mechanically.
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    return PuzzleScriptAdapter("PUZZLETALE", seed=seed)
