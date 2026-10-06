# MIT License
#
# Copyright (c) 2026 ARC Prize Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import copy as _copy_module
import random as _random_module
from collections import deque as _deque

import numpy as np
from arcengine import (
    ActionInput,
    ARCBaseGame,
    Camera,
    GameAction,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from utils.arc_game import AugmentedGame

sprites = {
    "aidclcbjcv": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="aidclcbjcv",
        visible=True,
        collidable=True,
    ),
    "byigobxzpg": Sprite(
        pixels=[
            [12, 12, 12, 12],
            [12, 12, 12, 12],
            [12, 12, 12, 12],
            [12, 12, 12, 12],
        ],
        name="byigobxzpg",
        visible=True,
        collidable=True,
        tags=["kdweefinfi"],
        layer=1,
    ),
    "cwefnfvjhr": Sprite(
        pixels=[
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        ],
        name="cwefnfvjhr",
        visible=True,
        collidable=True,
    ),
    "doijajrgdi": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="doijajrgdi",
        visible=True,
        collidable=False,
        tags=["fsjjayjoeg"],
    ),
    "geffskzhqq": Sprite(
        pixels=[
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ],
        name="geffskzhqq",
        visible=True,
        collidable=False,
        tags=["zqxwgacnue"],
    ),
    "ghklglzjuf": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="ghklglzjuf",
        visible=True,
        collidable=False,
        tags=["fsjjayjoeg"],
    ),
    "jigtxgzhwt": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="jigtxgzhwt",
        visible=True,
        collidable=False,
        tags=["fsjjayjoeg"],
    ),
    "jqzhxgbmtz": Sprite(
        pixels=[
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
            [15, 15, 15, 15],
        ],
        name="jqzhxgbmtz",
        visible=True,
        collidable=True,
        tags=["ysysltqlke"],
        layer=1,
    ),
    "ktghqrydvd": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="ktghqrydvd",
        visible=True,
        collidable=False,
        tags=["fsjjayjoeg"],
    ),
    "ofwegeqknn": Sprite(
        pixels=[
            [9, 9, 9, 9],
            [9, 2, 2, 9],
            [9, 2, 2, 9],
            [9, 2, 2, 9],
            [9, 2, 2, 9],
            [9, 2, 2, 9],
            [9, 2, 2, 9],
            [9, 9, 9, 9],
        ],
        name="ofwegeqknn",
        visible=True,
        collidable=False,
        tags=["fsjjayjoeg"],
    ),
    "ooaamfpvqr": Sprite(
        pixels=[
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2],
        ],
        name="ooaamfpvqr",
        visible=True,
        collidable=False,
        tags=["zqxwgacnue"],
    ),
    "peimznrlqd": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="peimznrlqd",
        visible=True,
        collidable=False,
        tags=["fsjjayjoeg"],
    ),
    "pktgsotzmw": Sprite(
        pixels=[
            [4, 4, 4, 4],
            [4, 9, 9, 4],
            [4, 9, 9, 4],
            [4, 4, 4, 4],
        ],
        name="pktgsotzmw",
        visible=True,
        collidable=True,
        tags=["geezpjgiyd"],
        layer=1,
    ),
    "pmargquscu": Sprite(
        pixels=[
            [2, -2, 2, 2],
            [-2, 2, 2, 2],
            [2, 2, 2, -2],
            [2, 2, -2, 2],
        ],
        name="pmargquscu",
        visible=True,
        collidable=False,
        tags=["bnzklblgdk"],
    ),
    "uasmnkbzmm": Sprite(
        pixels=[
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
            [5, 5, 5, 5],
        ],
        name="uasmnkbzmm",
        visible=True,
        collidable=True,
        tags=["debyzcmtnr"],
    ),
    "vikkhnsrzd": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="vikkhnsrzd",
        visible=True,
        collidable=False,
        tags=["fsjjayjoeg"],
    ),
    "wkmuwhjqyo": Sprite(
        pixels=[
            [9, 9, 9, 9, 9, 9, 9, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 2, 2, 2, 2, 2, 2, 9],
            [9, 9, 9, 9, 9, 9, 9, 9],
        ],
        name="wkmuwhjqyo",
        visible=True,
        collidable=False,
        tags=["fsjjayjoeg"],
    ),
    "wppuejnwhl": Sprite(
        pixels=[
            [0, 0, 0, 0],
            [14, 14, 14, 14],
            [14, 14, 14, 14],
            [14, 14, 14, 14],
        ],
        name="wppuejnwhl",
        visible=True,
        collidable=True,
        tags=["wbmdvjhthc"],
        layer=1,
    ),
    "xqaqifquaw": Sprite(
        pixels=[
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        ],
        name="xqaqifquaw",
        visible=True,
        collidable=False,
        tags=["zqxwgacnue"],
    ),
    "xxmzyqktqy": Sprite(
        pixels=[
            [9, 9, 9, 9],
            [9, 2, 2, 9],
            [9, 2, 2, 9],
            [9, 9, 9, 9],
        ],
        name="xxmzyqktqy",
        visible=True,
        collidable=False,
        tags=["fsjjayjoeg"],
    ),
}
levels = [
    # Level 1
    Level(
        sprites=[
            sprites["jigtxgzhwt"].clone().set_position(28, 28),
            sprites["pktgsotzmw"].clone().set_position(44, 24),
            sprites["pktgsotzmw"].clone().set_position(16, 28),
            sprites["pktgsotzmw"].clone().set_position(32, 36),
            sprites["wppuejnwhl"].clone().set_position(32, 48),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 200,
        },
    ),
    # Level 2
    Level(
        sprites=[
            sprites["byigobxzpg"].clone().set_position(24, 36),
            sprites["doijajrgdi"].clone().set_position(12, 28),
            sprites["pktgsotzmw"].clone().set_position(48, 32),
            sprites["pktgsotzmw"].clone().set_position(36, 28),
            sprites["pktgsotzmw"].clone().set_position(40, 20),
            sprites["pktgsotzmw"].clone().set_position(48, 24),
            sprites["pktgsotzmw"].clone().set_position(44, 40),
            sprites["wppuejnwhl"].clone().set_position(12, 8),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 70,
        },
    ),
    # Level 3
    Level(
        sprites=[
            sprites["byigobxzpg"].clone().set_position(48, 12),
            sprites["ktghqrydvd"].clone().set_position(52, 24),
            sprites["pktgsotzmw"].clone().set_position(32, 32),
            sprites["pktgsotzmw"].clone().set_position(20, 20),
            sprites["pktgsotzmw"].clone().set_position(12, 44),
            sprites["pktgsotzmw"].clone().set_position(8, 16),
            sprites["pktgsotzmw"].clone().set_position(32, 12),
            sprites["pmargquscu"].clone().set_position(32, 0),
            sprites["pmargquscu"].clone().set_position(32, 4),
            sprites["pmargquscu"].clone().set_position(32, 8),
            sprites["pmargquscu"].clone().set_position(32, 12),
            sprites["pmargquscu"].clone().set_position(32, 16),
            sprites["pmargquscu"].clone().set_position(32, 20),
            sprites["pmargquscu"].clone().set_position(32, 24),
            sprites["pmargquscu"].clone().set_position(32, 28),
            sprites["pmargquscu"].clone().set_position(32, 32),
            sprites["pmargquscu"].clone().set_position(32, 36),
            sprites["pmargquscu"].clone().set_position(32, 40),
            sprites["pmargquscu"].clone().set_position(32, 44),
            sprites["pmargquscu"].clone().set_position(32, 48),
            sprites["pmargquscu"].clone().set_position(32, 52),
            sprites["pmargquscu"].clone().set_position(32, 56),
            sprites["pmargquscu"].clone().set_position(32, 60),
            sprites["wppuejnwhl"].clone().set_position(16, 36),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 100,
        },
    ),
    # Level 4
    Level(
        sprites=[
            sprites["byigobxzpg"].clone().set_position(56, 4),
            sprites["byigobxzpg"].clone().set_position(8, 12),
            sprites["byigobxzpg"].clone().set_position(24, 56),
            sprites["ofwegeqknn"].clone().set_position(4, 24),
            sprites["pktgsotzmw"].clone().set_position(32, 36),
            sprites["pktgsotzmw"].clone().set_position(24, 24),
            sprites["pktgsotzmw"].clone().set_position(36, 40),
            sprites["pktgsotzmw"].clone().set_position(24, 40),
            sprites["pktgsotzmw"].clone().set_position(32, 24),
            sprites["pktgsotzmw"].clone().set_position(36, 24),
            sprites["pktgsotzmw"].clone().set_position(24, 4),
            sprites["pmargquscu"].clone().set_position(28, 20),
            sprites["pmargquscu"].clone().set_position(24, 20),
            sprites["pmargquscu"].clone().set_position(20, 20),
            sprites["pmargquscu"].clone().set_position(20, 24),
            sprites["pmargquscu"].clone().set_position(20, 28),
            sprites["pmargquscu"].clone().set_position(20, 32),
            sprites["pmargquscu"].clone().set_position(20, 36),
            sprites["pmargquscu"].clone().set_position(20, 40),
            sprites["pmargquscu"].clone().set_position(20, 44),
            sprites["pmargquscu"].clone().set_position(32, 20),
            sprites["pmargquscu"].clone().set_position(36, 20),
            sprites["pmargquscu"].clone().set_position(40, 20),
            sprites["pmargquscu"].clone().set_position(40, 24),
            sprites["pmargquscu"].clone().set_position(24, 44),
            sprites["pmargquscu"].clone().set_position(28, 44),
            sprites["pmargquscu"].clone().set_position(32, 44),
            sprites["pmargquscu"].clone().set_position(36, 44),
            sprites["pmargquscu"].clone().set_position(40, 44),
            sprites["pmargquscu"].clone().set_position(40, 40),
            sprites["pmargquscu"].clone().set_position(40, 36),
            sprites["pmargquscu"].clone().set_position(40, 32),
            sprites["pmargquscu"].clone().set_position(40, 28),
            sprites["uasmnkbzmm"].clone().set_position(16, 48),
            sprites["uasmnkbzmm"].clone().set_position(12, 52),
            sprites["uasmnkbzmm"].clone().set_position(8, 56),
            sprites["uasmnkbzmm"].clone().set_position(4, 60),
            sprites["uasmnkbzmm"].clone().set_position(44, 48),
            sprites["uasmnkbzmm"].clone().set_position(48, 52),
            sprites["uasmnkbzmm"].clone().set_position(52, 56),
            sprites["uasmnkbzmm"].clone().set_position(56, 60),
            sprites["uasmnkbzmm"].clone().set_position(28, 4),
            sprites["uasmnkbzmm"].clone().set_position(28, 0),
            sprites["uasmnkbzmm"].clone().set_position(28, 8),
            sprites["uasmnkbzmm"].clone().set_position(28, 12),
            sprites["uasmnkbzmm"].clone().set_position(28, 16),
            sprites["wkmuwhjqyo"].clone().set_position(36, 56),
            sprites["wppuejnwhl"].clone().set_position(28, 32),
            sprites["xxmzyqktqy"].clone().set_position(8, 36),
            sprites["xxmzyqktqy"].clone().set_position(52, 28),
            sprites["xxmzyqktqy"].clone().set_position(56, 20),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 100,
        },
    ),
    # Level 5
    Level(
        sprites=[
            sprites["byigobxzpg"].clone().set_position(20, 28),
            sprites["ktghqrydvd"].clone().set_position(8, 24),
            sprites["pktgsotzmw"].clone().set_position(52, 48),
            sprites["pktgsotzmw"].clone().set_position(60, 52),
            sprites["pktgsotzmw"].clone().set_position(48, 4),
            sprites["pktgsotzmw"].clone().set_position(56, 8),
            sprites["pktgsotzmw"].clone().set_position(44, 56),
            sprites["pktgsotzmw"].clone().set_position(44, 28),
            sprites["uasmnkbzmm"].clone().set_position(24, 24),
            sprites["uasmnkbzmm"].clone().set_position(28, 24),
            sprites["uasmnkbzmm"].clone().set_position(32, 24),
            sprites["uasmnkbzmm"].clone().set_position(36, 24),
            sprites["uasmnkbzmm"].clone().set_position(36, 20),
            sprites["uasmnkbzmm"].clone().set_position(36, 16),
            sprites["uasmnkbzmm"].clone().set_position(36, 12),
            sprites["uasmnkbzmm"].clone().set_position(36, 8),
            sprites["uasmnkbzmm"].clone().set_position(36, 4),
            sprites["uasmnkbzmm"].clone().set_position(36, 0),
            sprites["uasmnkbzmm"].clone().set_position(36, 60),
            sprites["uasmnkbzmm"].clone().set_position(36, 56),
            sprites["uasmnkbzmm"].clone().set_position(36, 52),
            sprites["uasmnkbzmm"].clone().set_position(36, 48),
            sprites["uasmnkbzmm"].clone().set_position(36, 44),
            sprites["uasmnkbzmm"].clone().set_position(36, 40),
            sprites["uasmnkbzmm"].clone().set_position(36, 36),
            sprites["uasmnkbzmm"].clone().set_position(24, 36),
            sprites["uasmnkbzmm"].clone().set_position(28, 36),
            sprites["uasmnkbzmm"].clone().set_position(32, 36),
            sprites["wppuejnwhl"].clone().set_position(44, 36),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 125,
        },
    ),
    # Level 6
    Level(
        sprites=[
            sprites["ghklglzjuf"].clone().set_position(28, 12),
            sprites["jqzhxgbmtz"].clone().set_position(16, 16),
            sprites["ooaamfpvqr"].clone().set_position(52, 24),
            sprites["pktgsotzmw"].clone().set_position(56, 28),
            sprites["pktgsotzmw"].clone().set_position(28, 16),
            sprites["uasmnkbzmm"].clone().set_position(44, 0),
            sprites["uasmnkbzmm"].clone().set_position(44, 4),
            sprites["uasmnkbzmm"].clone().set_position(44, 8),
            sprites["uasmnkbzmm"].clone().set_position(44, 12),
            sprites["uasmnkbzmm"].clone().set_position(44, 16),
            sprites["uasmnkbzmm"].clone().set_position(44, 20),
            sprites["uasmnkbzmm"].clone().set_position(48, 0),
            sprites["uasmnkbzmm"].clone().set_position(48, 4),
            sprites["uasmnkbzmm"].clone().set_position(48, 8),
            sprites["uasmnkbzmm"].clone().set_position(48, 12),
            sprites["uasmnkbzmm"].clone().set_position(48, 16),
            sprites["uasmnkbzmm"].clone().set_position(48, 20),
            sprites["uasmnkbzmm"].clone().set_position(44, 28),
            sprites["uasmnkbzmm"].clone().set_position(44, 32),
            sprites["uasmnkbzmm"].clone().set_position(44, 36),
            sprites["uasmnkbzmm"].clone().set_position(44, 40),
            sprites["uasmnkbzmm"].clone().set_position(44, 44),
            sprites["uasmnkbzmm"].clone().set_position(44, 48),
            sprites["uasmnkbzmm"].clone().set_position(48, 28),
            sprites["uasmnkbzmm"].clone().set_position(48, 32),
            sprites["uasmnkbzmm"].clone().set_position(48, 36),
            sprites["uasmnkbzmm"].clone().set_position(48, 40),
            sprites["uasmnkbzmm"].clone().set_position(48, 44),
            sprites["uasmnkbzmm"].clone().set_position(48, 48),
            sprites["uasmnkbzmm"].clone().set_position(44, 52),
            sprites["uasmnkbzmm"].clone().set_position(44, 56),
            sprites["uasmnkbzmm"].clone().set_position(48, 52),
            sprites["uasmnkbzmm"].clone().set_position(48, 56),
            sprites["uasmnkbzmm"].clone().set_position(44, 60),
            sprites["uasmnkbzmm"].clone().set_position(48, 60),
            sprites["wppuejnwhl"].clone().set_position(20, 52),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 75,
        },
    ),
    # Level 7
    Level(
        sprites=[
            sprites["aidclcbjcv"].clone().set_position(0, 44),
            sprites["cwefnfvjhr"].clone(),
            sprites["geffskzhqq"].clone().set_position(48, 28),
            sprites["jqzhxgbmtz"].clone().set_position(40, 32),
            sprites["ofwegeqknn"].clone().set_position(12, 28),
            sprites["pktgsotzmw"].clone().set_position(32, 24),
            sprites["pktgsotzmw"].clone().set_position(28, 32),
            sprites["uasmnkbzmm"].clone().set_position(0, 16),
            sprites["uasmnkbzmm"].clone().set_position(4, 16),
            sprites["uasmnkbzmm"].clone().set_position(8, 16),
            sprites["uasmnkbzmm"].clone().set_position(12, 16),
            sprites["uasmnkbzmm"].clone().set_position(16, 16),
            sprites["uasmnkbzmm"].clone().set_position(20, 16),
            sprites["uasmnkbzmm"].clone().set_position(24, 16),
            sprites["uasmnkbzmm"].clone().set_position(28, 16),
            sprites["uasmnkbzmm"].clone().set_position(32, 16),
            sprites["uasmnkbzmm"].clone().set_position(36, 16),
            sprites["uasmnkbzmm"].clone().set_position(40, 16),
            sprites["uasmnkbzmm"].clone().set_position(44, 16),
            sprites["uasmnkbzmm"].clone().set_position(48, 16),
            sprites["uasmnkbzmm"].clone().set_position(52, 16),
            sprites["uasmnkbzmm"].clone().set_position(56, 16),
            sprites["uasmnkbzmm"].clone().set_position(60, 16),
            sprites["uasmnkbzmm"].clone().set_position(0, 40),
            sprites["uasmnkbzmm"].clone().set_position(4, 40),
            sprites["uasmnkbzmm"].clone().set_position(8, 40),
            sprites["uasmnkbzmm"].clone().set_position(12, 40),
            sprites["uasmnkbzmm"].clone().set_position(16, 40),
            sprites["uasmnkbzmm"].clone().set_position(20, 40),
            sprites["uasmnkbzmm"].clone().set_position(24, 40),
            sprites["uasmnkbzmm"].clone().set_position(28, 40),
            sprites["uasmnkbzmm"].clone().set_position(32, 40),
            sprites["uasmnkbzmm"].clone().set_position(36, 40),
            sprites["uasmnkbzmm"].clone().set_position(40, 40),
            sprites["uasmnkbzmm"].clone().set_position(44, 40),
            sprites["uasmnkbzmm"].clone().set_position(48, 40),
            sprites["uasmnkbzmm"].clone().set_position(52, 40),
            sprites["uasmnkbzmm"].clone().set_position(56, 40),
            sprites["uasmnkbzmm"].clone().set_position(60, 40),
            sprites["wppuejnwhl"].clone().set_position(20, 32),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 125,
        },
    ),
    # Level 8
    Level(
        sprites=[
            sprites["byigobxzpg"].clone().set_position(28, 4),
            sprites["byigobxzpg"].clone().set_position(36, 44),
            sprites["jqzhxgbmtz"].clone().set_position(32, 16),
            sprites["jqzhxgbmtz"].clone().set_position(32, 56),
            sprites["peimznrlqd"].clone().set_position(48, 48),
            sprites["pktgsotzmw"].clone().set_position(32, 12),
            sprites["pktgsotzmw"].clone().set_position(28, 8),
            sprites["pktgsotzmw"].clone().set_position(40, 52),
            sprites["pktgsotzmw"].clone().set_position(36, 48),
            sprites["pktgsotzmw"].clone().set_position(28, 52),
            sprites["pktgsotzmw"].clone().set_position(32, 52),
            sprites["pktgsotzmw"].clone().set_position(36, 8),
            sprites["pktgsotzmw"].clone().set_position(24, 12),
            sprites["pktgsotzmw"].clone().set_position(4, 8),
            sprites["pktgsotzmw"].clone().set_position(8, 12),
            sprites["pktgsotzmw"].clone().set_position(8, 8),
            sprites["pktgsotzmw"].clone().set_position(4, 12),
            sprites["pktgsotzmw"].clone().set_position(24, 44),
            sprites["uasmnkbzmm"].clone().set_position(24, 24),
            sprites["uasmnkbzmm"].clone().set_position(28, 24),
            sprites["uasmnkbzmm"].clone().set_position(8, 24),
            sprites["uasmnkbzmm"].clone().set_position(12, 24),
            sprites["uasmnkbzmm"].clone().set_position(0, 24),
            sprites["uasmnkbzmm"].clone().set_position(4, 24),
            sprites["uasmnkbzmm"].clone().set_position(32, 24),
            sprites["uasmnkbzmm"].clone().set_position(36, 24),
            sprites["uasmnkbzmm"].clone().set_position(40, 24),
            sprites["uasmnkbzmm"].clone().set_position(44, 24),
            sprites["uasmnkbzmm"].clone().set_position(48, 24),
            sprites["uasmnkbzmm"].clone().set_position(52, 24),
            sprites["uasmnkbzmm"].clone().set_position(56, 24),
            sprites["uasmnkbzmm"].clone().set_position(60, 24),
            sprites["uasmnkbzmm"].clone().set_position(44, 36),
            sprites["uasmnkbzmm"].clone().set_position(48, 36),
            sprites["uasmnkbzmm"].clone().set_position(52, 36),
            sprites["uasmnkbzmm"].clone().set_position(56, 36),
            sprites["uasmnkbzmm"].clone().set_position(60, 36),
            sprites["uasmnkbzmm"].clone().set_position(0, 36),
            sprites["uasmnkbzmm"].clone().set_position(4, 36),
            sprites["uasmnkbzmm"].clone().set_position(8, 36),
            sprites["uasmnkbzmm"].clone().set_position(12, 36),
            sprites["uasmnkbzmm"].clone().set_position(16, 36),
            sprites["uasmnkbzmm"].clone().set_position(20, 36),
            sprites["uasmnkbzmm"].clone().set_position(24, 36),
            sprites["uasmnkbzmm"].clone().set_position(28, 36),
            sprites["uasmnkbzmm"].clone().set_position(32, 36),
            sprites["vikkhnsrzd"].clone().set_position(44, 8),
            sprites["wppuejnwhl"].clone().set_position(4, 32),
            sprites["xqaqifquaw"].clone().set_position(4, 8),
            sprites["xqaqifquaw"].clone().set_position(12, 48),
        ],
        grid_size=(64, 64),
        data={
            "StepCounter": 150,
        },
    ),
    # Level 9
    Level(
        sprites=[
            sprites["byigobxzpg"].clone().set_position(16, 28),
            sprites["byigobxzpg"].clone().set_position(44, 4),
            sprites["jqzhxgbmtz"].clone().set_position(60, 56),
            sprites["ooaamfpvqr"].clone().set_position(4, 28),
            sprites["peimznrlqd"].clone().set_position(20, 12),
            sprites["pktgsotzmw"].clone().set_position(12, 20),
            sprites["pktgsotzmw"].clone().set_position(8, 28),
            sprites["pktgsotzmw"].clone().set_position(4, 20),
            sprites["pktgsotzmw"].clone().set_position(4, 32),
            sprites["pktgsotzmw"].clone().set_position(56, 32),
            sprites["pktgsotzmw"].clone().set_position(44, 20),
            sprites["pktgsotzmw"].clone().set_position(8, 12),
            sprites["pktgsotzmw"].clone().set_position(48, 28),
            sprites["pktgsotzmw"].clone().set_position(4, 28),
            sprites["pmargquscu"].clone().set_position(52, 12),
            sprites["pmargquscu"].clone().set_position(40, 12),
            sprites["pmargquscu"].clone().set_position(56, 12),
            sprites["pmargquscu"].clone().set_position(60, 12),
            sprites["pmargquscu"].clone().set_position(48, 12),
            sprites["pmargquscu"].clone().set_position(44, 12),
            sprites["uasmnkbzmm"].clone().set_position(40, 40),
            sprites["uasmnkbzmm"].clone().set_position(36, 40),
            sprites["uasmnkbzmm"].clone().set_position(32, 40),
            sprites["uasmnkbzmm"].clone().set_position(56, 40),
            sprites["uasmnkbzmm"].clone().set_position(52, 40),
            sprites["uasmnkbzmm"].clone().set_position(48, 40),
            sprites["uasmnkbzmm"].clone().set_position(44, 40),
            sprites["uasmnkbzmm"].clone().set_position(60, 40),
            sprites["uasmnkbzmm"].clone().set_position(32, 44),
            sprites["uasmnkbzmm"].clone().set_position(32, 48),
            sprites["uasmnkbzmm"].clone().set_position(40, 56),
            sprites["uasmnkbzmm"].clone().set_position(40, 60),
            sprites["uasmnkbzmm"].clone().set_position(48, 44),
            sprites["uasmnkbzmm"].clone().set_position(48, 48),
            sprites["uasmnkbzmm"].clone().set_position(56, 48),
            sprites["uasmnkbzmm"].clone().set_position(56, 52),
            sprites["uasmnkbzmm"].clone().set_position(24, 56),
            sprites["uasmnkbzmm"].clone().set_position(24, 60),
            sprites["uasmnkbzmm"].clone().set_position(24, 52),
            sprites["uasmnkbzmm"].clone().set_position(24, 48),
            sprites["uasmnkbzmm"].clone().set_position(56, 56),
            sprites["uasmnkbzmm"].clone().set_position(56, 60),
            sprites["uasmnkbzmm"].clone().set_position(48, 52),
            sprites["uasmnkbzmm"].clone().set_position(48, 56),
            sprites["uasmnkbzmm"].clone().set_position(40, 48),
            sprites["uasmnkbzmm"].clone().set_position(40, 52),
            sprites["uasmnkbzmm"].clone().set_position(32, 56),
            sprites["uasmnkbzmm"].clone().set_position(32, 52),
            sprites["uasmnkbzmm"].clone().set_position(24, 40),
            sprites["uasmnkbzmm"].clone().set_position(28, 40),
            sprites["uasmnkbzmm"].clone().set_position(16, 40),
            sprites["uasmnkbzmm"].clone().set_position(20, 40),
            sprites["uasmnkbzmm"].clone().set_position(16, 44),
            sprites["uasmnkbzmm"].clone().set_position(16, 48),
            sprites["uasmnkbzmm"].clone().set_position(16, 56),
            sprites["uasmnkbzmm"].clone().set_position(16, 52),
            sprites["uasmnkbzmm"].clone().set_position(36, 0),
            sprites["uasmnkbzmm"].clone().set_position(36, 4),
            sprites["uasmnkbzmm"].clone().set_position(36, 8),
            sprites["uasmnkbzmm"].clone().set_position(36, 16),
            sprites["uasmnkbzmm"].clone().set_position(36, 12),
            sprites["uasmnkbzmm"].clone().set_position(36, 20),
            sprites["uasmnkbzmm"].clone().set_position(36, 24),
            sprites["uasmnkbzmm"].clone().set_position(36, 28),
            sprites["uasmnkbzmm"].clone().set_position(8, 60),
            sprites["uasmnkbzmm"].clone().set_position(8, 56),
            sprites["uasmnkbzmm"].clone().set_position(8, 52),
            sprites["uasmnkbzmm"].clone().set_position(8, 48),
            sprites["uasmnkbzmm"].clone().set_position(8, 44),
            sprites["uasmnkbzmm"].clone().set_position(8, 40),
            sprites["wkmuwhjqyo"].clone().set_position(52, 24),
            sprites["wkmuwhjqyo"].clone().set_position(52, 8),
            sprites["wppuejnwhl"].clone().set_position(32, 32),
        ],
        grid_size=(64, 64),
        data={
            # Bumped from 70: this labyrinth level has 9 blocks and only helper movers can
            # deliver some of them; 70 steps was unwinnable even on solvable spawns.
            "StepCounter": 110,
        },
    ),
]
BACKGROUND_COLOR = 1
PADDING_COLOR = 0
celomdfhbh = 4
wvrpthjfsv = 4
vlzkmytlgh = 3
hgqlwjikqr = 0
qrmfayeqpo = 5
ansrsmsvzs = 15
lajjveeqzb = 11

# Valid colors for randomization (excludes 0=transparent, 1=background,
# 3=adjacent-block-border, 5=block-outline-border, 11=adjacent-ysysltqlke-border)
_WA30_COLOR_POOL = [2, 4, 6, 7, 8, 9, 12, 13, 14, 15]


class etuniyewsy(RenderableUserDisplay):
    """."""

    def __init__(self, dbdarsgrbj: int):
        """."""
        self.dbdarsgrbj = dbdarsgrbj
        self.current_steps = dbdarsgrbj

    def uwwwedmhqv(self, yfgynsqzmu: int) -> None:
        """."""
        self.current_steps = max(0, min(yfgynsqzmu, self.dbdarsgrbj))

    def pfakmupgbr(self) -> bool:
        """."""
        if self.current_steps > 0:
            self.current_steps -= 1
        return self.current_steps > 0

    def ububboesmh(self) -> None:
        """."""
        self.current_steps = self.dbdarsgrbj

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        """."""
        if self.dbdarsgrbj == 0:
            return frame
        icsbqzymhf = self.current_steps / self.dbdarsgrbj
        eagwaqduxs = round(64 * icsbqzymhf)
        for x in range(64):
            if x < eagwaqduxs:
                frame[63, x] = 7
            else:
                frame[63, x] = 4
        return frame


class wjdpciselr(Camera):
    """."""

    def _raw_render(self, sprites: list[Sprite]) -> np.ndarray:
        output = np.full((self._height, self._width), self._background, dtype=np.int8)
        for sprite in sprites:
            sprite_pixels = sprite.pixels
            sprite_height, sprite_width = sprite_pixels.shape
            rel_x = sprite._x - self._x
            rel_y = sprite._y - self._y
            dest_x_start = max(0, rel_x)
            dest_x_end = min(self._width, rel_x + sprite_width)
            dest_y_start = max(0, rel_y)
            dest_y_end = min(self._height, rel_y + sprite_height)
            if dest_x_end <= dest_x_start or dest_y_end <= dest_y_start:
                continue
            sprite_x_start = max(0, -rel_x)
            sprite_x_end = sprite_width - max(0, rel_x + sprite_width - self._width)
            sprite_y_start = max(0, -rel_y)
            sprite_y_end = sprite_height - max(0, rel_y + sprite_height - self._height)
            sprite_region = sprite_pixels[sprite_y_start:sprite_y_end, sprite_x_start:sprite_x_end]
            visible_mask = sprite_region >= 0
            output[dest_y_start:dest_y_end, dest_x_start:dest_x_end][visible_mask] = sprite_region[visible_mask]
        return output


def hbipqrhvbm(rfwebuoepa: tuple[int, int], ixzwlkvsbs: tuple[int, int]) -> int:
    return abs(rfwebuoepa[0] - ixzwlkvsbs[0]) + abs(rfwebuoepa[1] - ixzwlkvsbs[1])


def mdbwmdaxuu(rfwebuoepa: tuple[int, int], ixzwlkvsbs: tuple[int, int]) -> bool:
    return hbipqrhvbm(rfwebuoepa, ixzwlkvsbs) == celomdfhbh


def vwiozbtqgi(fqdniajpfh: Sprite, sprite: Sprite) -> bool:
    if fqdniajpfh.rotation == 0:
        return fqdniajpfh.x == sprite.x and fqdniajpfh.y - celomdfhbh == sprite.y
    elif fqdniajpfh.rotation == 180:
        return fqdniajpfh.x == sprite.x and fqdniajpfh.y + celomdfhbh == sprite.y
    elif fqdniajpfh.rotation == 90:
        return fqdniajpfh.x + celomdfhbh == sprite.x and fqdniajpfh.y == sprite.y
    else:
        return fqdniajpfh.x - celomdfhbh == sprite.x and fqdniajpfh.y == sprite.y


def uxricavavq(sprite: Sprite, clhazarisu: int) -> None:
    sprite.pixels[:, 0] = clhazarisu
    sprite.pixels[:, 3] = clhazarisu
    sprite.pixels[0, :] = clhazarisu
    sprite.pixels[3, :] = clhazarisu


def pjedoipwee(dx: int, dy: int) -> int:
    if dy < 0:
        return 0
    elif dx > 0:
        return 90
    elif dy > 0:
        return 180
    return 270


def anojofkynf(v: tuple[int, int], sprite: Sprite) -> bool:
    return v[0] >= sprite.x and v[0] < sprite.x + sprite.width and (v[1] >= sprite.y) and (v[1] < sprite.y + sprite.height)


class Wa30(AugmentedGame):
    kuncbnslnm: etuniyewsy
    nsevyuople: dict[Sprite, Sprite]
    zmqreragji: dict[Sprite, Sprite]
    pkbufziase: set[tuple[int, int]]
    wyzquhjerd: set[tuple[int, int]]
    lqctaojiby: set[tuple[int, int]]
    qthdiggudy: set[tuple[int, int]]
    lkvghqfwan: set[tuple[int, int]]
    uuorgjazmj: set[tuple[int, int]]
    # (block cell, pusher cell) pairs a helper has proved undeliverable from that side. Dragging
    # keeps a FIXED pusher/block offset, so which side a helper grabs from decides which targets it
    # can ever reach; without this a helper that grabs a bad side re-grabs it forever.
    _bad_grabs: set[tuple[tuple[int, int], tuple[int, int]]]
    _bad_grab_sig: tuple | None

    def __init__(self, seed: int | None = None) -> None:
        # ``seed`` makes the game a pure function of (seed, level): the block layout,
        # the palette and the display rotation are each drawn from an RNG fixed by the
        # seed, so Wa30(seed=S) renders and lays out identically every run. This is what
        # lets a recorded episode and a later live run of the same seed match. Optional --
        # an unseeded Wa30() keeps the original nondeterministic behaviour.
        self._seed_value = seed
        self._rng = _random_module.Random()
        # Tracks the current ysysltqlke normal-border color, used by zzppkjnqgk
        self._ysysltqlke_color = ansrsmsvzs  # 15 (default)
        self.kuncbnslnm = etuniyewsy(0)
        self._init_augmentation(seed)
        sfahrdviv = Camera(
            background=BACKGROUND_COLOR,
            letter_box=PADDING_COLOR,
            interfaces=[self.kuncbnslnm],
        )
        # Deep-copy the module-level levels so each Wa30 instance gets its own
        # sprite objects; this ensures color/position randomization in on_set_level
        # doesn't bleed across concurrent game instances in the same process.
        _instance_levels = _copy_module.deepcopy(levels)
        super().__init__(
            game_id="wa30",
            levels=_instance_levels,
            camera=sfahrdviv,
            available_actions=[1, 2, 3, 4, 5],
        )

    def _randomize_colors(self) -> None:
        """Pick 5 distinct random colors and remap sprite pixels for each object type.

        Each object type is remapped from its known original (module-load-time) color
        value, so this works correctly when advancing to a new level whose sprites are
        always freshly cloned with the original colors.
        """
        c1, c2, c3, c4, c5 = self._rng.sample(_WA30_COLOR_POOL, 5)
        for sprite in self.current_level.get_sprites():
            tags = sprite.tags
            if "geezpjgiyd" in tags:
                # Original inner color is 9 (pktgsotzmw pixels[1,1] etc.)
                sprite.pixels[sprite.pixels == 9] = c1
            if "fsjjayjoeg" in tags or "zqxwgacnue" in tags:
                # Original fill color is 2
                sprite.pixels[sprite.pixels == 2] = c2
            if "debyzcmtnr" in tags:
                # Original obstacle color is 5
                sprite.pixels[sprite.pixels == 5] = c3
            if "kdweefinfi" in tags:
                # Original kdweefinfi color is 12
                sprite.pixels[sprite.pixels == 12] = c4
            if "ysysltqlke" in tags:
                # Original ysysltqlke color is 15
                sprite.pixels[sprite.pixels == 15] = c5
            if "bnzklblgdk" in tags:
                # Original dot-obstacle color is 2
                sprite.pixels[sprite.pixels == 2] = c3
        # Keep ysysltqlke_color in sync so zzppkjnqgk uses the right border color
        self._ysysltqlke_color = c5

    def _fixed_obstacle_positions(self) -> set:
        """Return grid-aligned positions of static collidable obstacles (debyzcmtnr)."""
        blocked: set = set()
        for sprite in self.current_level.get_sprites_by_tag("debyzcmtnr"):
            blocked.add((sprite.x, sprite.y))
        return blocked

    def _bnzklblgdk_positions(self) -> set:
        """Return positions of bnzklblgdk dot obstacles."""
        positions: set = set()
        for sprite in self.current_level.get_sprites_by_tag("bnzklblgdk"):
            positions.add((sprite.x, sprite.y))
        return positions

    def _target_grid_cells(self) -> set:
        """Return grid-aligned positions (multiples of celomdfhbh) within target sprites."""
        cells: set = set()
        step = celomdfhbh
        for sprite in self.current_level.get_sprites_by_tag("fsjjayjoeg"):
            for dy in range(sprite.height):
                for dx in range(sprite.width):
                    px, py = sprite.x + dx, sprite.y + dy
                    if px % step == 0 and py % step == 0:
                        cells.add((px, py))
        for sprite in self.current_level.get_sprites_by_tag("zqxwgacnue"):
            for dy in range(sprite.height):
                for dx in range(sprite.width):
                    px, py = sprite.x + dx, sprite.y + dy
                    if px % step == 0 and py % step == 0:
                        cells.add((px, py))
        return cells

    def _player_reachable_region(self, bnzklblgdk: set, fixed_blocked: set) -> set:
        """BFS from the player (wbmdvjhthc) treating bnzklblgdk + fixed obstacles as barriers."""
        player_sprites = self.current_level.get_sprites_by_tag("wbmdvjhthc")
        if not player_sprites or not bnzklblgdk:
            return set()
        step = celomdfhbh
        grid_max = 64 - step
        start = (player_sprites[0].x, player_sprites[0].y)
        barriers = bnzklblgdk | fixed_blocked
        visited = {start}
        queue = [start]
        while queue:
            cx, cy = queue.pop(0)
            for ddx, ddy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                nx, ny = cx + ddx, cy + ddy
                if (nx, ny) in visited:
                    continue
                if nx < 0 or ny < 0 or nx > grid_max or ny > grid_max:
                    continue
                if (nx, ny) in barriers:
                    continue
                visited.add((nx, ny))
                queue.append((nx, ny))
        return visited

    def _bfs_reaches_target(self, start: tuple, target_cells: set, blocked: set) -> bool:
        """Return True if a block at start can reach any target cell via grid BFS."""
        step = celomdfhbh
        grid_max = 64 - step  # = 60
        if start in target_cells:
            return True
        visited = {start}
        queue = [start]
        while queue:
            cx, cy = queue.pop(0)
            for ddx, ddy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                nx, ny = cx + ddx, cy + ddy
                if (nx, ny) in visited:
                    continue
                if nx < 0 or ny < 0 or nx > grid_max or ny > grid_max:
                    continue
                if (nx, ny) in blocked:
                    continue
                if (nx, ny) in target_cells:
                    return True
                visited.add((nx, ny))
                queue.append((nx, ny))
        return False

    def _randomize_target_positions(self, fixed_blocked: set, player_region: set, bnzklblgdk: set) -> None:
        """Shift all target sprites by a random grid-aligned offset."""
        step = celomdfhbh
        grid_size = 64
        target_sprites = (
            self.current_level.get_sprites_by_tag("fsjjayjoeg")
            + self.current_level.get_sprites_by_tag("zqxwgacnue")
        )
        if not target_sprites:
            return
        # Build candidate offsets: multiples of step in [-16, 16]
        offsets = list(range(-16, 17, step))
        pairs = [(dx, dy) for dx in offsets for dy in offsets if dx != 0 or dy != 0]
        pairs.append((0, 0))  # ensure original is a fallback
        self._rng.shuffle(pairs)
        transfer_zone = set(bnzklblgdk)
        for bx, by in bnzklblgdk:
            for ddx, ddy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                nx, ny = bx + ddx, by + ddy
                if (nx, ny) not in player_region and (nx, ny) not in bnzklblgdk:
                    transfer_zone.add((nx, ny))
        forbidden = fixed_blocked | player_region | transfer_zone
        for dx, dy in pairs:
            valid = True
            for sprite in target_sprites:
                nx, ny = sprite.x + dx, sprite.y + dy
                # Check sprite stays fully within grid
                if nx < 0 or ny < 0:
                    valid = False
                    break
                if nx + sprite.width > grid_size or ny + sprite.height > grid_size:
                    valid = False
                    break
                # Check no grid-aligned corner overlaps a fixed obstacle or player region
                for sy in range(0, sprite.height, step):
                    for sx in range(0, sprite.width, step):
                        if (nx + sx, ny + sy) in forbidden:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                if dx != 0 or dy != 0:
                    for sprite in target_sprites:
                        sprite.set_position(sprite.x + dx, sprite.y + dy)
                return

    def _grid_regions(self, blocked: set) -> list:
        """Flood-fill the open 4px lattice into connected regions under ``blocked`` barriers.

        Returns a list of sets of (x, y) cells. Used so block placement respects the SAME
        barriers every mover obeys at runtime (static obstacles AND dot obstacles): the old
        placement check treated dots as passable, which on the dot-maze levels spawned blocks in
        pockets no mover could ever deliver from -- an unwinnable board.
        """
        step = celomdfhbh
        grid_max = 64 - step  # = 60
        open_cells = {
            (x, y)
            for x in range(0, grid_max + 1, step)
            for y in range(0, grid_max + 1, step)
            if (x, y) not in blocked
        }
        regions: list = []
        seen: set = set()
        for cell in open_cells:
            if cell in seen:
                continue
            comp: set = {cell}
            stack = [cell]
            seen.add(cell)
            while stack:
                cx, cy = stack.pop()
                for ddx, ddy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                    nb = (cx + ddx, cy + ddy)
                    if nb in open_cells and nb not in seen:
                        seen.add(nb)
                        comp.add(nb)
                        stack.append(nb)
            regions.append(comp)
        return regions

    def _randomize_block_positions(self, fixed_blocked: set, bnzklblgdk: set, target_cells: set, player_region: set) -> None:
        """Move each geezpjgiyd block to a random valid, *deliverable* start position.

        Solvability by construction: a block is only ever placed in a connected region (under the
        real obstacle + dot barriers) that contains a MOVER able to deliver it -- the avatar
        (``wbmdvjhthc``) or a helper mover (``kdweefinfi``) -- and that still has a free win-target
        to spare, and no region is given more blocks than it has targets. Adversary movers
        (``ysysltqlke``) do not count as deliverers. This replaces the previous quota logic, whose
        dots-blind reachability check could strand blocks in target-less pockets.
        """
        step = celomdfhbh
        grid_max = 64 - step  # = 60
        blocks = self.current_level.get_sprites_by_tag("geezpjgiyd")
        if not blocks or not target_cells:
            return

        # Positions occupied by AI movers and player (blocks can't start on them).
        reserved: set = set()
        mover_cells: set = set()
        for tag in ("wbmdvjhthc", "kdweefinfi", "ysysltqlke"):
            for sprite in self.current_level.get_sprites_by_tag(tag):
                reserved.add((sprite.x, sprite.y))
        for tag in ("wbmdvjhthc", "kdweefinfi"):  # only deliverers, not adversaries
            for sprite in self.current_level.get_sprites_by_tag(tag):
                mover_cells.add((sprite.x, sprite.y))

        blocked_true = set(fixed_blocked) | set(bnzklblgdk)
        regions = self._grid_regions(blocked_true)

        # Keep only regions that can actually deliver: a deliverer sits in them AND they hold at
        # least one grid-aligned win target. Capacity = number of such targets in the region.
        grid_targets = {c for c in target_cells if c[0] % step == 0 and c[1] % step == 0}
        deliverable = []  # (region_cells, capacity, target_cells_in_region)
        for reg in regions:
            if not (reg & mover_cells):
                continue
            tin = reg & grid_targets
            if not tin:
                continue
            deliverable.append([reg, len(tin), tin])

        # Fallback: if the level geometry has no dot/obstacle partitioning that this model can use
        # (e.g. everything is one big region, or no deliverable region found), fall back to the
        # simple dots-and-obstacles-aware reachability placement over all open cells.
        if not deliverable:
            self._place_blocks_fallback(blocks, blocked_true, grid_targets, reserved)
            return

        # Build a shuffled pool of region slots, capped per region at its target capacity, and
        # assign each block a region. If total capacity < #blocks the surplus blocks fall back.
        self._rng.shuffle(deliverable)
        slots = []
        for reg, cap, _tin in deliverable:
            slots.extend([id(reg)] * cap)
        self._rng.shuffle(slots)
        region_by_id = {id(reg): (reg, tin) for reg, _cap, tin in deliverable}

        placed: set = set(reserved)
        leftover_blocks = []
        for block, slot in zip(blocks, slots):
            reg, tin = region_by_id[slot]
            # Prefer cells not already on a target so the puzzle isn't pre-solved, and require the
            # block can still reach an in-region target given everything already placed.
            open_here = [c for c in reg if c not in placed and c not in bnzklblgdk]
            non_target = [c for c in open_here if c not in grid_targets]
            self._rng.shuffle(non_target)
            self._rng.shuffle(open_here)
            chosen = None
            for pool in (non_target, open_here):
                for pos in pool:
                    if self._bfs_reaches_target(pos, tin, blocked_true | (placed - {pos})):
                        chosen = pos
                        break
                if chosen is not None:
                    break
            if chosen is None:
                leftover_blocks.append(block)
                continue
            block.set_position(chosen[0], chosen[1])
            placed.add(chosen)

        # Any block without a region slot (more blocks than total capacity, or a region that
        # filled up): place it wherever it can still reach any target under the true barriers.
        for block in list(blocks)[len(slots):] + leftover_blocks:
            self._place_one_fallback(block, blocked_true, grid_targets, placed)

    def _place_one_fallback(self, block, blocked_true: set, grid_targets: set, placed: set) -> None:
        """Place a single block on any open cell that can reach a target under true barriers."""
        step = celomdfhbh
        grid_max = 64 - step
        open_cells = [
            (x, y)
            for x in range(0, grid_max + 1, step)
            for y in range(0, grid_max + 1, step)
            if (x, y) not in blocked_true and (x, y) not in placed
        ]
        self._rng.shuffle(open_cells)
        for pos in open_cells:
            if self._bfs_reaches_target(pos, grid_targets, blocked_true | (placed - {pos})):
                block.set_position(pos[0], pos[1])
                placed.add(pos)
                return
        placed.add((block.x, block.y))  # leave in place if nothing valid

    def _place_blocks_fallback(self, blocks, blocked_true: set, grid_targets: set, reserved: set) -> None:
        """Dots/obstacle-aware version of the original placement, used when the region model finds
        no deliverable region to work with."""
        placed = set(reserved)
        for block in blocks:
            self._place_one_fallback(block, blocked_true, grid_targets, placed)

    def _drag_bfs(self, pusher, block, obs_pusher: set, obs_block: set, avoid: set, cap: int = 25000):
        """Explore the REAL drag model from a pusher standing near a block, exactly as the engine
        moves things (``fuykgiiwit``): the pusher may not enter ``obs_pusher`` (obstacles + dots +
        other movers/blocks) but a dragged block may be pushed onto a dot -- so the block only
        collides with ``obs_block`` (obstacles + other movers/blocks, NOT dots).

        Returns the set of cells the block can be released on (grid-aligned), excluding ``avoid``.
        Used to invert delivery: any returned cell is a start from which this pusher can drag the
        block back to where it started (``block``). Capped so a level load stays cheap.
        """
        step = celomdfhbh
        dirs = ((0, -step, 0), (0, step, 180), (-step, 0, 270), (step, 0, 90))
        face = {0: (0, -step), 180: (0, step), 90: (step, 0), 270: (-step, 0)}
        start = (pusher[0], pusher[1], block[0], block[1], 0, 0)
        seen = {start}
        q = _deque([start])
        rest = set()
        n = 0
        while q:
            n += 1
            if n > cap:
                break
            px, py, bx, by, gr, rot = q.popleft()
            if gr == 0 and (bx, by) not in avoid:
                rest.add((bx, by))
            for dx, dy, dr in dirs:
                if gr == 0:
                    nx, ny = px + dx, py + dy
                    if (nx, ny) in obs_pusher or (nx, ny) == (bx, by) or not (0 <= nx <= 60 and 0 <= ny <= 60):
                        ns = (px, py, bx, by, 0, dr)
                    else:
                        ns = (nx, ny, bx, by, 0, dr)
                else:
                    nx, ny = px + dx, py + dy
                    nbx, nby = bx + dx, by + dy
                    okp = ((nx, ny) not in obs_pusher or (nx, ny) == (bx, by)) and 0 <= nx <= 60 and 0 <= ny <= 60
                    okb = ((nbx, nby) not in obs_block or (nbx, nby) == (px, py)) and 0 <= nbx <= 60 and 0 <= nby <= 60
                    ns = (nx, ny, nbx, nby, 1, rot) if (okp and okb) else (px, py, bx, by, 1, rot)
                if ns not in seen:
                    seen.add(ns)
                    q.append(ns)
            if gr == 0:
                fdx, fdy = face[rot]
                if (px + fdx, py + fdy) == (bx, by):
                    ns = (px, py, bx, by, 1, rot)
                    if ns not in seen:
                        seen.add(ns)
                        q.append(ns)
            else:
                ns = (px, py, bx, by, 0, rot)
                if ns not in seen:
                    seen.add(ns)
                    q.append(ns)
        return rest

    def _reverse_drag_placement(self, fixed_blocked: set, bnzklblgdk: set) -> None:
        """Place blocks by REVERSE-DRAGGING each one out from a win target, guaranteeing the board is
        drag-solvable even in a tight labyrinth (blocks are placed at the far end of a real drag path
        a mover can retrace). Solves the L8 deadlock the point-reachability check could not detect.

        Per connected region (under obstacle + dot barriers): assign blocks to distinct win targets,
        and for each, BFS the drag model outward from the target with a region mover as pusher,
        treating earlier-placed blocks as barriers and every OTHER target as off-limits. Delivering
        in reverse placement order then wins with no deadlock.
        """
        step = celomdfhbh
        blocks = self.current_level.get_sprites_by_tag("geezpjgiyd")
        if not blocks:
            return
        # Win targets = grid-aligned cells of fsjjayjoeg only (NOT the zqxwgacnue decoys); computed
        # directly because self.wyzquhjerd is not built until later in on_set_level.
        win_targets = set()
        for sprite in self.current_level.get_sprites_by_tag("fsjjayjoeg"):
            for dy in range(sprite.height):
                for dx in range(sprite.width):
                    px, py = sprite.x + dx, sprite.y + dy
                    if px % step == 0 and py % step == 0:
                        win_targets.add((px, py))
        if not win_targets:
            return

        blocked_true = set(fixed_blocked) | set(bnzklblgdk)
        delivs = [(s.x, s.y) for s in self.current_level.get_sprites_by_tag("wbmdvjhthc")]
        for s in self.current_level.get_sprites_by_tag("kdweefinfi"):
            delivs.append((s.x, s.y))
        mover_cells = set(delivs)
        for s in self.current_level.get_sprites_by_tag("ysysltqlke"):
            mover_cells.add((s.x, s.y))

        regions = self._grid_regions(blocked_true)
        reg_info = []  # [region_cells, [deliverers in it], [win targets in it]]
        for reg in regions:
            rd = [d for d in delivs if d in reg]
            rt = [t for t in win_targets if t in reg]
            if rd and rt:
                reg_info.append((reg, rd, rt))
        # NB: confining placement to avatar-containing regions was TRIED AND REVERTED -- it made
        # things worse (avatar-undeliverable seeds 7/32 -> 9/32) and voided the L8 demos. The stuck
        # blocks are NOT simply in mover-only regions; `_grid_regions` connectivity (walls+dots)
        # does not match what the avatar can actually walk (`kblzhbvysd` also excludes movers and
        # blocks), so region membership is the wrong lens for this. Root cause still open.
        if not reg_info:
            self._place_blocks_fallback(blocks, blocked_true, win_targets, mover_cells)
            return

        # Assign each block a region slot, capped per region at its win-target count.
        self._rng.shuffle(reg_info)
        slots = []
        for i, (_reg, _rd, rt) in enumerate(reg_info):
            slots.extend([i] * len(rt))
        self._rng.shuffle(slots)
        region_blocks: dict = {}
        for block, slot in zip(blocks, slots):
            region_blocks.setdefault(slot, []).append(block)

        placed_blocks: set = set()
        for i, (reg, rd, rt) in enumerate(reg_info):
            bl = region_blocks.get(i, [])
            avail_t = list(rt)
            self._rng.shuffle(avail_t)
            for j, block in enumerate(bl):
                if j >= len(avail_t):
                    self._place_one_fallback(block, blocked_true, win_targets, mover_cells | placed_blocks)
                    continue
                target = avail_t[j]
                rest = None
                # NB: restricting this to avatar-only pushers was tried and is a NO-OP -- `delivs`
                # already lists the avatar first and it is the successful pusher on every seed
                # checked, so the helper fallback never fires. The ~22% of seeds carrying an
                # avatar-undeliverable block are NOT caused by helper-pusher placement; the retrace
                # verifies a drag path exists but never checks the avatar can WALK to the grab cell
                # to start it, which is the remaining suspect.
                for pusher in rd:
                    # Only the PUSHER's own start cell is excluded from the pusher's barriers (it
                    # stands there). The BLOCK must never rest on a mover cell -- two collidables on
                    # one cell give pkbufziase a single entry, and the second sprite to move off it
                    # raises KeyError. The drag model still lets the block swap into the pusher's
                    # CURRENT cell via the `== (px, py)` exception, so nothing is over-restricted.
                    obs_p = (blocked_true | placed_blocks | mover_cells) - {pusher}
                    obs_b = set(fixed_blocked) | placed_blocks | mover_cells
                    restset = self._drag_bfs(pusher, target, obs_p, obs_b, avoid=win_targets)
                    restset -= mover_cells
                    restset -= placed_blocks
                    if restset:
                        # Prefer a cell dragged well away from the target (non-trivial start), with a
                        # deterministic order so the layout stays a pure function of the seed.
                        cands = sorted(restset, key=lambda c: (abs(c[0] - target[0]) + abs(c[1] - target[1]), c[0], c[1]))
                        half = cands[len(cands) // 2:] or cands
                        rest = half[self._rng.randrange(len(half))]
                        break
                if rest is None and target not in mover_cells and target not in placed_blocks:
                    rest = target  # pre-solved fallback (still a win; only if the mover can't move it)
                if rest is None:
                    # never stack a block on a mover/another block: pkbufziase keys on the cell
                    self._place_one_fallback(block, blocked_true, win_targets, mover_cells | placed_blocks)
                    continue
                block.set_position(rest[0], rest[1])
                placed_blocks.add(rest)

        # Any block that never got a slot (more blocks than total capacity) -- shouldn't happen when
        # win targets >= blocks, but stay safe.
        for block in list(blocks)[len(slots):]:
            self._place_one_fallback(block, blocked_true, win_targets, mover_cells | placed_blocks)

    def on_set_level(self, level: Level) -> None:
        # When seeded, re-seed the layout RNG deterministically from (seed, level) so the
        # palette + block placement below are a pure function of the seed. The frame rotation
        # for this (seed, level) is drawn by AugmentedGame.set_level before this runs.
        if self._seed_value is not None:
            self._rng = _random_module.Random(
                f"wa30:{int(self._seed_value)}:{int(self._current_level_index)}"
            )
        # Randomize colors first (remaps sprite pixels based on tracked current colors)
        self._randomize_colors()
        # Compute fixed obstacle sets before moving anything
        _fixed = self._fixed_obstacle_positions()
        _dots = self._bnzklblgdk_positions()
        _player_region = self._player_reachable_region(_dots, _fixed)
        _targets = self._target_grid_cells()
        # Randomize block starting positions (with region-based distribution).
        # Skip for level 6 (index 5): block must start in solution region for adversarial move.
        # The labyrinth level (index 8) uses reverse-drag placement instead, which guarantees the
        # tight corridors can never spawn a drag-deadlock the region check would miss.
        if self._current_level_index == 8:
            self._reverse_drag_placement(_fixed, _dots)
        elif self._current_level_index != 5:
            self._randomize_block_positions(_fixed, _dots, _targets, _player_region)
        self.xcuqvqnmiu()
        self.nsevyuople = dict()
        self.zmqreragji = dict()
        self.pkbufziase = set()
        self.wyzquhjerd = set()
        self.lqctaojiby = set()
        self.qthdiggudy = set()
        self.lkvghqfwan = set()
        self.uuorgjazmj = set()
        self._bad_grabs = set()
        self._bad_grab_sig = None
        xdatcqhbr = self.current_level.get_sprites()
        for wunpnzavk in xdatcqhbr:
            if wunpnzavk.is_collidable:
                self.pkbufziase.add((wunpnzavk.x, wunpnzavk.y))
        for i in range(0, 64, celomdfhbh):
            self.pkbufziase.add((-celomdfhbh, i))
            self.pkbufziase.add((64, i))
            self.pkbufziase.add((i, -celomdfhbh))
            self.pkbufziase.add((i, 64))
        qiitczxxug = self.current_level.get_sprites_by_tag("fsjjayjoeg")
        for yotccnlopr in qiitczxxug:
            for i in range(yotccnlopr.height):
                for iiffofdbmb in range(yotccnlopr.width):
                    self.wyzquhjerd.add((yotccnlopr.x + iiffofdbmb, yotccnlopr.y + i))
        djpjnippwv = self.current_level.get_sprites_by_tag("zqxwgacnue")
        for tsrsytlfgb in djpjnippwv:
            for i in range(tsrsytlfgb.height):
                for iiffofdbmb in range(tsrsytlfgb.width):
                    self.lqctaojiby.add((tsrsytlfgb.x + iiffofdbmb, tsrsytlfgb.y + i))
        gflnevqmdv = self.current_level.get_sprites_by_tag("bnzklblgdk")
        for adxrpmucwx in gflnevqmdv:
            self.qthdiggudy.add((adxrpmucwx.x, adxrpmucwx.y))
        self.vyltpasvhc()
        self.lgirylubbp()

    def vyltpasvhc(self) -> None:
        """."""
        self.lkvghqfwan = set()
        fyfxmnwzhp = self.current_level.get_sprites_by_tag("geezpjgiyd")
        # A proved-bad grab side only stays valid while the board is unchanged -- once any block
        # moves, the drag paths change and a side that failed before may now work.
        sig = tuple(sorted((b.x, b.y) for b in fyfxmnwzhp))
        if sig != self._bad_grab_sig:
            self._bad_grabs = set()
            self._bad_grab_sig = sig
        for ijudbtgsll in fyfxmnwzhp:
            if ijudbtgsll not in self.zmqreragji and (not self.shbxbhnhjc((ijudbtgsll.x, ijudbtgsll.y))):
                qrcmkiozlr = [
                    (ijudbtgsll.x - celomdfhbh, ijudbtgsll.y),
                    (ijudbtgsll.x + celomdfhbh, ijudbtgsll.y),
                    (ijudbtgsll.x, ijudbtgsll.y - celomdfhbh),
                    (ijudbtgsll.x, ijudbtgsll.y + celomdfhbh),
                ]
                for v in qrcmkiozlr:
                    # don't route a mover to a side it has already proved cannot deliver
                    if ((ijudbtgsll.x, ijudbtgsll.y), v) not in self._bad_grabs:
                        self.lkvghqfwan.add(v)

    def lgirylubbp(self) -> None:
        """."""
        self.uuorgjazmj = set()
        fyfxmnwzhp = self.current_level.get_sprites_by_tag("geezpjgiyd")
        for ijudbtgsll in fyfxmnwzhp:
            if not self.jrrltylxpp(ijudbtgsll) and (not self.ahzqkfjpsc((ijudbtgsll.x, ijudbtgsll.y))):
                qrcmkiozlr = [
                    (ijudbtgsll.x - celomdfhbh, ijudbtgsll.y),
                    (ijudbtgsll.x + celomdfhbh, ijudbtgsll.y),
                    (ijudbtgsll.x, ijudbtgsll.y - celomdfhbh),
                    (ijudbtgsll.x, ijudbtgsll.y + celomdfhbh),
                ]
                for v in qrcmkiozlr:
                    self.uuorgjazmj.add(v)

    def xcuqvqnmiu(self) -> None:
        """."""
        drogceccgh = self.current_level.get_data("StepCounter")
        self.kuncbnslnm.dbdarsgrbj = drogceccgh
        self.kuncbnslnm.ububboesmh()

    def wqwsvmhhzj(self, fqdniajpfh: Sprite, x: int, y: int) -> None:
        if fqdniajpfh in self.nsevyuople:
            tkrlgpoppf = self.nsevyuople[fqdniajpfh]
            if self.fuykgiiwit(fqdniajpfh, tkrlgpoppf, (x, y)):
                dx, dy = (tkrlgpoppf.x - fqdniajpfh.x, tkrlgpoppf.y - fqdniajpfh.y)
                self.pkbufziase.remove((fqdniajpfh.x, fqdniajpfh.y))
                self.pkbufziase.remove((tkrlgpoppf.x, tkrlgpoppf.y))
                fqdniajpfh.set_position(x, y)
                tkrlgpoppf.set_position(x + dx, y + dy)
                self.pkbufziase.add((fqdniajpfh.x, fqdniajpfh.y))
                self.pkbufziase.add((tkrlgpoppf.x, tkrlgpoppf.y))
                self.lgirylubbp()
        elif self.kblzhbvysd((x, y)):
            self.pkbufziase.remove((fqdniajpfh.x, fqdniajpfh.y))
            fqdniajpfh.set_position(x, y)
            self.pkbufziase.add((fqdniajpfh.x, fqdniajpfh.y))

    def qnmfimgpwc(self, fqdniajpfh: Sprite, dx: int, dy: int) -> None:
        if fqdniajpfh not in self.nsevyuople:
            rotation = pjedoipwee(dx, dy)
            fqdniajpfh.set_rotation(rotation)
        self.wqwsvmhhzj(fqdniajpfh, fqdniajpfh.x + dx, fqdniajpfh.y + dy)

    def kblzhbvysd(self, v: tuple[int, int]) -> bool:
        return v not in self.pkbufziase and v not in self.qthdiggudy

    def fuykgiiwit(self, fqdniajpfh: Sprite, tkrlgpoppf: Sprite, v: tuple[int, int]) -> bool:
        dx = tkrlgpoppf.x - fqdniajpfh.x
        dy = tkrlgpoppf.y - fqdniajpfh.y
        tsvynrmpmm = (v[0] + dx, v[1] + dy)
        vouzrxjshl = (v not in self.pkbufziase or v == (tkrlgpoppf.x, tkrlgpoppf.y)) and v not in self.qthdiggudy and (tsvynrmpmm not in self.pkbufziase or tsvynrmpmm == (fqdniajpfh.x, fqdniajpfh.y))
        return vouzrxjshl

    def shbxbhnhjc(self, v: tuple[int, int]) -> bool:
        return v in self.wyzquhjerd

    def ahzqkfjpsc(self, v: tuple[int, int]) -> bool:
        return v in self.lqctaojiby

    def czrprbohhe(self, fqdniajpfh: Sprite) -> list[tuple[int, int]] | None:
        kpiyvjksem = (fqdniajpfh.x, fqdniajpfh.y)
        qkmekaxaqf = set()
        qkmekaxaqf.add(kpiyvjksem)
        mbkpalarob = [[kpiyvjksem]]
        while mbkpalarob:
            sfivrisylh = mbkpalarob.pop(0)
            omhkabyvpg = sfivrisylh[-1]
            if omhkabyvpg in self.lkvghqfwan:
                return sfivrisylh
            qrcmkiozlr = [
                (omhkabyvpg[0] - celomdfhbh, omhkabyvpg[1]),
                (omhkabyvpg[0] + celomdfhbh, omhkabyvpg[1]),
                (omhkabyvpg[0], omhkabyvpg[1] - celomdfhbh),
                (omhkabyvpg[0], omhkabyvpg[1] + celomdfhbh),
            ]
            for yweobqdxhg in qrcmkiozlr:
                if yweobqdxhg not in qkmekaxaqf and self.kblzhbvysd(yweobqdxhg):
                    qkmekaxaqf.add(yweobqdxhg)
                    mbkpalarob.append(sfivrisylh + [yweobqdxhg])
        return None

    def cyjrduhzmz(self, fqdniajpfh: Sprite) -> list[tuple[int, int]] | None:
        return self._drag_path(fqdniajpfh, self.nsevyuople[fqdniajpfh])

    def _drag_path(self, fqdniajpfh: Sprite, tkrlgpoppf: Sprite) -> list[tuple[int, int]] | None:
        """Shortest pusher path that seats ``tkrlgpoppf`` on a win cell, at the offset the pair
        currently has. Split out of ``cyjrduhzmz`` so a mover can ask "could I deliver this block
        if I grabbed it from here?" BEFORE committing to the grab -- the offset is frozen at grab
        time, so asking afterwards is too late."""
        kpiyvjksem = (fqdniajpfh.x, fqdniajpfh.y)
        qkmekaxaqf = set()
        qkmekaxaqf.add(kpiyvjksem)
        mbkpalarob = [[kpiyvjksem]]
        dx = tkrlgpoppf.x - fqdniajpfh.x
        dy = tkrlgpoppf.y - fqdniajpfh.y
        while mbkpalarob:
            sfivrisylh = mbkpalarob.pop(0)
            omhkabyvpg = sfivrisylh[-1]
            if self.shbxbhnhjc((omhkabyvpg[0] + dx, omhkabyvpg[1] + dy)):
                return sfivrisylh
            qrcmkiozlr = [
                (omhkabyvpg[0] - celomdfhbh, omhkabyvpg[1]),
                (omhkabyvpg[0] + celomdfhbh, omhkabyvpg[1]),
                (omhkabyvpg[0], omhkabyvpg[1] - celomdfhbh),
                (omhkabyvpg[0], omhkabyvpg[1] + celomdfhbh),
            ]
            for yweobqdxhg in qrcmkiozlr:
                if yweobqdxhg not in qkmekaxaqf and self.fuykgiiwit(fqdniajpfh, tkrlgpoppf, yweobqdxhg):
                    qkmekaxaqf.add(yweobqdxhg)
                    mbkpalarob.append(sfivrisylh + [yweobqdxhg])
        return None

    def zauouvdhta(self, fqdniajpfh: Sprite) -> list[tuple[int, int]] | None:
        kpiyvjksem = (fqdniajpfh.x, fqdniajpfh.y)
        qkmekaxaqf = set()
        qkmekaxaqf.add(kpiyvjksem)
        mbkpalarob = [[kpiyvjksem]]
        while mbkpalarob:
            sfivrisylh = mbkpalarob.pop(0)
            omhkabyvpg = sfivrisylh[-1]
            if omhkabyvpg in self.uuorgjazmj:
                return sfivrisylh
            qrcmkiozlr = [
                (omhkabyvpg[0] - celomdfhbh, omhkabyvpg[1]),
                (omhkabyvpg[0] + celomdfhbh, omhkabyvpg[1]),
                (omhkabyvpg[0], omhkabyvpg[1] - celomdfhbh),
                (omhkabyvpg[0], omhkabyvpg[1] + celomdfhbh),
            ]
            for yweobqdxhg in qrcmkiozlr:
                if yweobqdxhg not in qkmekaxaqf and self.kblzhbvysd(yweobqdxhg):
                    qkmekaxaqf.add(yweobqdxhg)
                    mbkpalarob.append(sfivrisylh + [yweobqdxhg])
        return None

    def egqayvffim(self, fqdniajpfh: Sprite) -> list[tuple[int, int]] | None:
        kpiyvjksem = (fqdniajpfh.x, fqdniajpfh.y)
        qkmekaxaqf = set()
        qkmekaxaqf.add(kpiyvjksem)
        mbkpalarob = [[kpiyvjksem]]
        tkrlgpoppf = self.nsevyuople[fqdniajpfh]
        dx = tkrlgpoppf.x - fqdniajpfh.x
        dy = tkrlgpoppf.y - fqdniajpfh.y
        while mbkpalarob:
            sfivrisylh = mbkpalarob.pop(0)
            omhkabyvpg = sfivrisylh[-1]
            if self.ahzqkfjpsc((omhkabyvpg[0] + dx, omhkabyvpg[1] + dy)):
                return sfivrisylh
            qrcmkiozlr = [
                (omhkabyvpg[0] - celomdfhbh, omhkabyvpg[1]),
                (omhkabyvpg[0] + celomdfhbh, omhkabyvpg[1]),
                (omhkabyvpg[0], omhkabyvpg[1] - celomdfhbh),
                (omhkabyvpg[0], omhkabyvpg[1] + celomdfhbh),
            ]
            for yweobqdxhg in qrcmkiozlr:
                if yweobqdxhg not in qkmekaxaqf and self.fuykgiiwit(fqdniajpfh, tkrlgpoppf, yweobqdxhg):
                    qkmekaxaqf.add(yweobqdxhg)
                    mbkpalarob.append(sfivrisylh + [yweobqdxhg])
        return None

    def xpcvspllwr(self, fqdniajpfh: Sprite, sprite: Sprite) -> None:
        if sprite in self.zmqreragji:
            self.kqrtstlzkg(self.zmqreragji[sprite])
        self.nsevyuople[fqdniajpfh] = sprite
        self.zmqreragji[sprite] = fqdniajpfh
        self.vyltpasvhc()
        self.lgirylubbp()

    def kqrtstlzkg(self, fqdniajpfh: Sprite) -> None:
        if fqdniajpfh in self.nsevyuople:
            sprite = self.nsevyuople[fqdniajpfh]
            del self.zmqreragji[sprite]
            del self.nsevyuople[fqdniajpfh]
            self.vyltpasvhc()
            self.lgirylubbp()

    def zzppkjnqgk(self) -> None:
        xfkkaooadb = self.current_level.get_sprites_by_tag("wbmdvjhthc")
        fyfxmnwzhp = self.current_level.get_sprites_by_tag("geezpjgiyd")
        for ijudbtgsll in fyfxmnwzhp:
            if ijudbtgsll in self.zmqreragji:
                fqdniajpfh = self.zmqreragji[ijudbtgsll]
                if fqdniajpfh in xfkkaooadb:
                    uxricavavq(ijudbtgsll, hgqlwjikqr)
                elif any([vwiozbtqgi(olxydpaqfn, ijudbtgsll) for olxydpaqfn in xfkkaooadb]):
                    uxricavavq(ijudbtgsll, vlzkmytlgh)
                else:
                    uxricavavq(ijudbtgsll, qrmfayeqpo)
            elif any([vwiozbtqgi(olxydpaqfn, ijudbtgsll) for olxydpaqfn in xfkkaooadb]):
                uxricavavq(ijudbtgsll, vlzkmytlgh)
            else:
                uxricavavq(ijudbtgsll, wvrpthjfsv)
        zyjsukrrpd = self.current_level.get_sprites_by_tag("ysysltqlke")
        for kkkwtxdnov in zyjsukrrpd:
            if any([vwiozbtqgi(olxydpaqfn, kkkwtxdnov) for olxydpaqfn in xfkkaooadb]):
                uxricavavq(kkkwtxdnov, lajjveeqzb)
            else:
                uxricavavq(kkkwtxdnov, self._ysysltqlke_color)

    def ynmgxjqkgh(self) -> None:
        makxrwhhlc = self.current_level.get_sprites_by_tag("kdweefinfi")
        for dhptvmzfpx in makxrwhhlc:
            if dhptvmzfpx in self.nsevyuople:
                tkrlgpoppf = self.nsevyuople[dhptvmzfpx]
                if self.shbxbhnhjc((tkrlgpoppf.x, tkrlgpoppf.y)):
                    self.kqrtstlzkg(dhptvmzfpx)
                else:
                    sfivrisylh = self.cyjrduhzmz(dhptvmzfpx)
                    if sfivrisylh and len(sfivrisylh) > 1:
                        czvpytgvaw = sfivrisylh[1]
                        self.wqwsvmhhzj(dhptvmzfpx, czvpytgvaw[0], czvpytgvaw[1])
                    else:
                        # No delivery from this side: remember it so the mover walks around and
                        # tries another one instead of re-grabbing here every tick forever.
                        self._bad_grabs.add(((tkrlgpoppf.x, tkrlgpoppf.y),
                                             (dhptvmzfpx.x, dhptvmzfpx.y)))
                        self.kqrtstlzkg(dhptvmzfpx)
            else:
                fyfxmnwzhp = self.current_level.get_sprites_by_tag("geezpjgiyd")
                for ijudbtgsll in fyfxmnwzhp:
                    if mdbwmdaxuu((dhptvmzfpx.x, dhptvmzfpx.y), (ijudbtgsll.x, ijudbtgsll.y)) and ijudbtgsll not in self.zmqreragji and (not self.shbxbhnhjc((ijudbtgsll.x, ijudbtgsll.y))):
                        pair = ((ijudbtgsll.x, ijudbtgsll.y), (dhptvmzfpx.x, dhptvmzfpx.y))
                        if pair in self._bad_grabs:
                            continue
                        if self._drag_path(dhptvmzfpx, ijudbtgsll) is None:
                            self._bad_grabs.add(pair)
                            continue
                        self.xpcvspllwr(dhptvmzfpx, ijudbtgsll)
                        return
                sfivrisylh = self.czrprbohhe(dhptvmzfpx)
                # len 1 = the mover already stands on an approach cell but declined the grab (bad
                # side); there is no step to take, so idle rather than index past the path end.
                if sfivrisylh and len(sfivrisylh) > 1:
                    czvpytgvaw = sfivrisylh[1]
                    self.wqwsvmhhzj(dhptvmzfpx, czvpytgvaw[0], czvpytgvaw[1])

    def jrrltylxpp(self, sprite: Sprite) -> bool:
        if sprite in self.zmqreragji:
            fqdniajpfh = self.zmqreragji[sprite]
            return "ysysltqlke" in fqdniajpfh.tags
        return False

    def aoeyzovteg(self) -> None:
        zyjsukrrpd = self.current_level.get_sprites_by_tag("ysysltqlke")
        for kkkwtxdnov in zyjsukrrpd:
            if kkkwtxdnov in self.nsevyuople:
                tkrlgpoppf = self.nsevyuople[kkkwtxdnov]
                if self.ahzqkfjpsc((tkrlgpoppf.x, tkrlgpoppf.y)):
                    self.kqrtstlzkg(kkkwtxdnov)
                else:
                    sfivrisylh = self.egqayvffim(kkkwtxdnov)
                    if sfivrisylh and len(sfivrisylh) > 1:
                        czvpytgvaw = sfivrisylh[1]
                        self.wqwsvmhhzj(kkkwtxdnov, czvpytgvaw[0], czvpytgvaw[1])
                    else:
                        self.kqrtstlzkg(kkkwtxdnov)
            else:
                fyfxmnwzhp = self.current_level.get_sprites_by_tag("geezpjgiyd")
                for ijudbtgsll in fyfxmnwzhp:
                    if mdbwmdaxuu((kkkwtxdnov.x, kkkwtxdnov.y), (ijudbtgsll.x, ijudbtgsll.y)) and (not self.jrrltylxpp(ijudbtgsll)) and (not self.ahzqkfjpsc((ijudbtgsll.x, ijudbtgsll.y))):
                        self.xpcvspllwr(kkkwtxdnov, ijudbtgsll)
                        return
                sfivrisylh = self.zauouvdhta(kkkwtxdnov)
                if sfivrisylh:
                    czvpytgvaw = sfivrisylh[1]
                    self.wqwsvmhhzj(kkkwtxdnov, czvpytgvaw[0], czvpytgvaw[1])

    def ymzfopzgbq(self) -> bool:
        fyfxmnwzhp = self.current_level.get_sprites_by_tag("geezpjgiyd")
        return all([self.shbxbhnhjc((ijudbtgsll.x, ijudbtgsll.y)) and ijudbtgsll not in self.zmqreragji for ijudbtgsll in fyfxmnwzhp])

    def dhrikuybfo(self) -> None:
        self.ynmgxjqkgh()
        self.aoeyzovteg()
        self.zzppkjnqgk()

    def yygfcvqoyx(self, action: ActionInput) -> None:
        _action_id = self.screen_action_to_game(action.id)
        olxydpaqfn = self.current_level.get_sprites_by_tag("wbmdvjhthc")[0]
        if _action_id == GameAction.ACTION1:
            self.kuncbnslnm.pfakmupgbr()
            dx = 0
            dy = -celomdfhbh
            self.qnmfimgpwc(olxydpaqfn, dx, dy)
            self.dhrikuybfo()
        elif _action_id == GameAction.ACTION2:
            self.kuncbnslnm.pfakmupgbr()
            dx = 0
            dy = celomdfhbh
            self.qnmfimgpwc(olxydpaqfn, dx, dy)
            self.dhrikuybfo()
        elif _action_id == GameAction.ACTION3:
            self.kuncbnslnm.pfakmupgbr()
            dx = -celomdfhbh
            dy = 0
            self.qnmfimgpwc(olxydpaqfn, dx, dy)
            self.dhrikuybfo()
        elif _action_id == GameAction.ACTION4:
            self.kuncbnslnm.pfakmupgbr()
            dx = celomdfhbh
            dy = 0
            self.qnmfimgpwc(olxydpaqfn, dx, dy)
            self.dhrikuybfo()
        elif _action_id == GameAction.ACTION5:
            self.kuncbnslnm.pfakmupgbr()
            if olxydpaqfn in self.nsevyuople:
                self.kqrtstlzkg(olxydpaqfn)
            else:
                fyfxmnwzhp = self.current_level.get_sprites_by_tag("geezpjgiyd")
                for ijudbtgsll in fyfxmnwzhp:
                    if vwiozbtqgi(olxydpaqfn, ijudbtgsll):
                        self.xpcvspllwr(olxydpaqfn, ijudbtgsll)
                        break
                zyjsukrrpd = self.current_level.get_sprites_by_tag("ysysltqlke")
                for kkkwtxdnov in zyjsukrrpd:
                    if vwiozbtqgi(olxydpaqfn, kkkwtxdnov):
                        self.kqrtstlzkg(kkkwtxdnov)
                        self.pkbufziase.remove((kkkwtxdnov.x, kkkwtxdnov.y))
                        self.current_level.remove_sprite(kkkwtxdnov)
                        break
            self.dhrikuybfo()

    def step(self) -> None:
        self.yygfcvqoyx(self.action)
        if self.ymzfopzgbq():
            self.next_level()
        elif not self.kuncbnslnm.current_steps:
            self.lose()
        self.complete_action()
