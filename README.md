# ARC-AGI-3-style Games, Solvers and training data generators

## Introduction

This repository contains 429 distinct games in the style of ARC-AGI-3 games. That is, its action set is a subset of the ARC-AGI-3 action set, and the observation space is defined as a 64x64 matrix.

Additionally, most games have an element of randomization/data augmentation builtin. For example, if you run ar25 several times in a row using play.py, you will see different shapes and different starting positions. If desired, determinism can be preserved by specifying the seed number.

## Playing the games

To list the available games:

`python3 play.py`

To manually play a game: 

`python3 play.py (game name)`

example: `python3 play.py ps:vrps`

Actions:

- A, W, S, D for movement in most games
- Spacebar is ACTION 5
- r for reset
- u for undo

## Using the solvers to generate data

Each game has a solver associated with it, which can be used to generate training demonstrations for the purpose or Imitation learning, or bootstapping hybrid solutions ("Learning from demonstrations" + RL, for example).

`python3 master_data_generator.py --game_list list.csv --episodes 1000 --n_workers 6`

This generates 1000 training episodes per game in the game list.

## Format of the generated demonstration files

The `solvers/generate_*_training.py` scripts write one JSON file per successful
episode. By default, files are stored under
`data/training_multi_level/<game_id>/episode_00000_seed0.json`. The five-digit
number counts episodes written in that run; `seed0` identifies the game seed.
Failed attempts can leave gaps in the seed sequence. The seed is encoded in the
filename, not in the JSON object.

Each file contains a `game_id` and a list of independent level trajectories.
`game_id` is the generator's corpus identifier, for example `puzzlescript_wrap`
for the playable game `ps:wrap`. A level contains:

| Field | Meaning |
| --- | --- |
| `level_id` | The original zero-based level index. Some solvers emit only a supported, solvable subset, so indices need not be consecutive. |
| `observations` | An ordered list of frames, each a 64×64 matrix of integer palette indices, indexed as `[row][column]`. These are the rendered observations, including the game's visual augmentation. |
| `actions` | An ordered list of action records, including the initial RESET and any exploration or recovery steps. |

The following example illustrates the structure using **2×2 placeholder frames**
for readability; actual frames are always 64×64:

```json
{
  "game_id": "puzzlescript_wrap",
  "levels": [
    {
      "level_id": 0,
      "observations": [
        [[0, 0], [1, 0]],
        [[0, 0], [0, 1]]
      ],
      "actions": [
        {
          "type": "simple",
          "index": 0,
          "phase": "reset",
          "changed": false,
          "n_obs": 1,
          "optimal": null
        },
        {
          "type": "simple",
          "index": 4,
          "phase": "expert",
          "changed": true,
          "n_obs": 1,
          "optimal": [{"type": "simple", "index": 4}]
        }
      ]
    }
  ]
}
```

Action records describe both the action taken and, when available, the expert's
training target:

| Field | Meaning |
| --- | --- |
| `type` | `"simple"` for a discrete action, or `"mouse"` for a click. |
| `index` | The action ID: `0` is RESET, `1`–`5` are ACTION1–ACTION5, `6` is a click, and `7` is ACTION7 (typically undo). Available actions and their meaning depend on the game. |
| `data` | Present for mouse actions: `{"x": column, "y": row}` in the displayed 64×64 frame, with zero-based coordinates. |
| `phase` | `"reset"` for initialization/recovery, `"explore"` for exploratory play, `"burst"` for a perturbation burst, or `"expert"` for solver-guided play. |
| `changed` | Whether the settled observation differs from the preceding settled observation. The initial RESET uses `false`. This describes visible change, not necessarily a change in hidden game state. |
| `n_obs` | The number of consecutive frames produced by this action; usually `1`, but can be larger for animations. |
| `optimal` | A list of acceptable expert target actions at the state **before** the action, or `null` when unlabelled. Each entry contains `type`, `index`, and optional `data`. A solver may supply multiple equally good choices or a single selected action. |

Actions and click coordinates use the displayed screen's coordinate system,
including any rotation or reflection. The top-level `type`/`index`/`data` fields
record what was actually executed. During exploration or a burst, that action
can differ from `optimal`; do not treat every executed action as an expert
target. Use `phase` and the presence of `optimal` to select training examples.
Recovery RESETs may have a RESET target, while the initial RESET is unlabelled.

**Frame/action alignment:** `actions[0]` is the RESET that produces the initial
observation. When every `n_obs` is `1`, `actions[i]` for `i > 0` takes the game
from `observations[i-1]` to `observations[i]`. For animated actions, consume the
flat frame list using each action's frame count:

```python
offset = 0
previous = None
for action in level["actions"]:
    count = action.get("n_obs", 1)  # Older records may omit n_obs.
    frames = level["observations"][offset:offset + count]
    # `previous` is the pre-action observation (None for the initial RESET).
    # `frames` contains the resulting animation; frames[-1] is the settled frame.
    previous = frames[-1]
    offset += count
assert offset == len(level["observations"])
```

Thus `sum(action["n_obs"] for action in actions) == len(observations)`;
the number of actions need not equal the number of frames. Levels are separate
trajectories: do not create a transition between the end of one and the start
of the next. The format does not include per-step rewards, terminal flags, or
per-pixel labels; the generators save successful demonstrations.

These details describe the solvers' JSON output. The master launcher also
attempts conversion to `.ep.zst` files through `convert_training_data.py`, deleting
the original JSON after successful verification by default. Use `--keep-json`
to retain the JSON alongside compressed output, or `--no-compress` to generate
JSON only. The conversion script must be present to use compression.
