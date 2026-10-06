"""PuzzleScript game: wizard_school

This game is loaded via the PuzzleScriptAdapter from
data/puzzlescript_games/Wizard_School!.txt

Play with: python solver_client.py ps:wizard_school

Press X to summon every box that shares your row or column to your side, walk it
somewhere, and press X again to fling all of them away at once. The win is
``all target on box``.

THE ONE RENDERING FIX (and why it is needed)
--------------------------------------------
``Target`` sits on its own collision layer UNDER the box layer, and both box
sprites (``boxidle``, ``boxcarried``) are opaque 5x5 squares. So a box standing
on a target rendered pixel-identically to a box standing anywhere else, and the
win condition -- which is entirely about which targets are covered -- was
invisible in the frame the agent is shown.

``make_game`` punches the four edge-midpoint pixels of both box sprites out to
transparent. Those are exactly the four cells the target's black diamond has its
tips in, so a box on a target now shows four black pips and a box on plain floor
shows four yellow ones. Nothing mechanical changes -- sprites are cosmetic --
and `solvers/generate_wizard_school_training.py --audit` checks the whole-frame
distinctness of all thirteen legal cell compositions on every level, which is
the check that caught this in the first place.

The generator records against THIS adapter (`WizardSchoolSolver.game_module_id`
is set), so the frames it tapes are the frames a live agent sees.
"""

from adapters.puzzlescript_adapter import PuzzleScriptAdapter

#: The four sprite pixels the target's diamond has its tips in. Punched out of
#: the box sprites so the target underneath shows through; see the module
#: docstring.
_TARGET_TIPS = ((0, 2), (2, 0), (2, 4), (4, 2))


def make_game(seed: int = 0):
    """Create an instance of this PuzzleScript game."""
    adapter = PuzzleScriptAdapter("Wizard_School!", seed=seed)

    for name in ("boxidle", "boxcarried"):
        obj = adapter._game.objects.get(name)
        if obj is None or not obj.sprite:
            continue
        for (r, c) in _TARGET_TIPS:
            obj.sprite[r][c] = -1

    adapter._do_reset()
    return adapter
