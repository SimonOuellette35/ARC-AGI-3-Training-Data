# ARC-AGI-3-style Games, Solvers and training data generators

This repository contains 429 distinct games in the style of ARC-AGI-3 games. That is, its action set is a subset of the ARC-AGI-3 action set, and the observation space is defined as a 64x64 matrix.

Additionally, most games have an element of randomization/data augmentation builtin. For example, if you run ar25 several times in a row using play.py, you will see different shapes and different starting positions. If desired, determinism can be preserved by specifying the seed number.

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
