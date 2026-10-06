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

import numpy as np
from arcengine import (
    ARCBaseGame,
    Camera,
    GameAction,
    InteractionMode,
    Level,
    RenderableUserDisplay,
    Sprite,
)
from utils.arc_game import AugmentedGame

sprites = {
    "cvcer": Sprite(
        pixels=[
            [9],
        ],
        name="cvcer",
        visible=True,
        collidable=True,
        tags=["sys_click", "nhiae"],
    ),
    "dfnuk-qeazm": Sprite(
        pixels=[
            [15],
            [15],
            [15],
        ],
        name="dfnuk-qeazm",
        visible=True,
        collidable=True,
    ),
    "dfnuk-raixb": Sprite(
        pixels=[
            [14],
            [14],
            [14],
        ],
        name="dfnuk-raixb",
        visible=True,
        collidable=True,
    ),
    "dfnuk-ujcze": Sprite(
        pixels=[
            [12],
            [12],
            [12],
        ],
        name="dfnuk-ujcze",
        visible=True,
        collidable=True,
    ),
    "hnutp-qeazm": Sprite(
        pixels=[
            [15],
        ],
        name="hnutp-qeazm",
        visible=True,
        collidable=True,
    ),
    "hnutp-raixb": Sprite(
        pixels=[
            [14],
        ],
        name="hnutp-raixb",
        visible=True,
        collidable=True,
    ),
    "hnutp-ujcze": Sprite(
        pixels=[
            [12],
        ],
        name="hnutp-ujcze",
        visible=True,
        collidable=True,
    ),
    "jggua-Level1": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, -1, -1, -1, 0, 0, 0, -1, 0],
            [0, -1, 0, -1, 0, -1, 0, 0, 0, -1, 0, -1, 0],
            [0, -1, 0, -1, 0, -1, 0, -1, -1, -1, -1, -1, 0],
            [0, -1, 0, -1, -1, -1, 0, -1, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, 0],
            [0, 0, 0, 0, 0, -1, 0, -1, -1, 0, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0],
            [0, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0],
            [0, 0, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="jggua-Level1",
        visible=True,
        collidable=True,
        tags=["jggua"],
    ),
    "jggua-Level2": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="jggua-Level2",
        visible=True,
        collidable=True,
        tags=["jggua"],
    ),
    "jggua-Level3": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0],
            [0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, -1, -1, -1, 0, 0, -1, 0, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0],
            [-1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, -1],
            [-1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1],
            [-1, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1],
            [-1, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1],
            [-1, 0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, 0, 0, -1],
            [-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0, 0, 0],
            [0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, -1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="jggua-Level3",
        visible=True,
        collidable=True,
        tags=["jggua"],
    ),
    "jggua-Level4": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0],
            [0, 0, -1, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0, -1, 0, 0],
            [0, 0, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0, -1, 0, 0],
            [0, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, 0, 0, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, -1, 0, 0, -1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, -1, -1, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, 0],
            [0, -1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="jggua-Level4",
        visible=True,
        collidable=True,
        tags=["jggua"],
    ),
    "jggua-Level5": Sprite(
        pixels=[
            [0, 0, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1],
            [-1, -1, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0, 0, 0, -1, -1],
            [-1, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, 0, -1, -1, -1],
            [-1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, -1, -1],
            [0, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, 0, 0, -1, -1],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0],
            [-1, -1, -1, -1, -1, -1, 0, -1, 0, 0, 0, -1, -1, -1, -1, 0],
            [-1, -1, -1, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, 0, 0],
        ],
        name="jggua-Level5",
        visible=True,
        collidable=True,
        tags=["jggua"],
    ),
    "jggua-Level6": Sprite(
        pixels=[
            [0, 0, -1, -1, -1, 0, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0],
            [0, -1, -1, 0, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 0],
            [0, -1, -1, 0, 0, 0, 0, 0, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, 0, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="jggua-Level6",
        visible=True,
        collidable=True,
        tags=["jggua"],
    ),
    "jggua-Level7": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="jggua-Level7",
        visible=True,
        collidable=True,
        tags=["jggua"],
    ),
    "jggua-Level8": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, 0, -1, -1, -1, -1, 0, -1, 0],
            [0, 0, 0, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, 0, 0, -1, -1, -1, 0, 0, 0],
            [-1, -1, -1, -1, -1, 0, 0, -1, -1, -1, -1, -1, -1, 0],
            [-1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1],
            [-1, -1, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, 0, 0],
            [-1, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0],
            [-1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="jggua-Level8",
        visible=True,
        collidable=True,
        tags=["jggua"],
    ),
    "jggua-Level9": Sprite(
        pixels=[
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, -1, -1, -1, 0, 0, 0, -1, -1, -1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ],
        name="jggua-Level9",
        visible=True,
        collidable=True,
        tags=["jggua"],
    ),
    "jggua-Level10": Sprite(
        pixels=[
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
            [0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0],
        ],
        name="jggua-Level10",
        visible=True,
        collidable=True,
        tags=["jggua"],
    ),
    "jggua-Level11": Sprite(
        pixels=[
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, -1, 0, 0, 0, 0, 0, 0, 0],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
            [-1, -1, -1, -1, 0, -1, -1, -1, -1],
        ],
        name="jggua-Level11",
        visible=True,
        collidable=True,
        tags=["jggua"],
    ),
    "qzfkx-kncqr-crkfz": Sprite(
        pixels=[
            [10],
        ],
        name="qzfkx-kncqr-crkfz",
        visible=True,
        collidable=True,
        tags=["sys_click"],
        layer=2,
    ),
    "qzfkx-kncqr-idtiq": Sprite(
        pixels=[
            [10],
        ],
        name="qzfkx-kncqr-idtiq",
        visible=True,
        collidable=True,
        tags=["sys_click"],
        layer=2,
    ),
    "qzfkx-ubwff-crkfz": Sprite(
        pixels=[
            [10],
        ],
        name="qzfkx-ubwff-crkfz",
        visible=True,
        collidable=True,
        tags=["sys_click", "pxwnx"],
        layer=2,
    ),
    "qzfkx-ubwff-idtiq": Sprite(
        pixels=[
            [10],
        ],
        name="qzfkx-ubwff-idtiq",
        visible=True,
        collidable=True,
        tags=["sys_click", "pxwnx"],
        layer=2,
    ),
    "wyiex": Sprite(
        pixels=[
            [8],
        ],
        name="wyiex",
        visible=True,
        collidable=True,
        tags=["wyiex"],
    ),
}
levels = [
    # Level 1
    Level(
        sprites=[
            sprites["jggua-Level6"].clone().set_rotation(180),
            sprites["qzfkx-ubwff-crkfz"].clone().set_position(7, 9),
            sprites["qzfkx-ubwff-idtiq"].clone().set_position(3, 9),
        ],
        grid_size=(11, 11),
        data={
            "npwxa": [11, 12],
        },
    ),
    # Level 2
    Level(
        sprites=[
            sprites["jggua-Level11"].clone().set_position(2, 0),
            sprites["qzfkx-ubwff-crkfz"].clone().set_position(8, 1),
            sprites["qzfkx-ubwff-idtiq"].clone().set_position(4, 1),
            sprites["wyiex"].clone().set_position(5, 5),
            sprites["wyiex"].clone().set_position(5, 6),
            sprites["wyiex"].clone().set_position(12, 8),
            sprites["wyiex"].clone().set_position(11, 8),
            sprites["wyiex"].clone().set_position(10, 8),
            sprites["wyiex"].clone().set_position(9, 8),
            sprites["wyiex"].clone().set_position(8, 8),
            sprites["wyiex"].clone().set_position(4, 8),
            sprites["wyiex"].clone().set_position(2, 8),
            sprites["wyiex"].clone().set_position(1, 8),
            sprites["wyiex"].clone().set_position(0, 8),
            sprites["wyiex"].clone().set_position(5, 8),
            sprites["wyiex"].clone().set_position(5, 7),
            sprites["wyiex"].clone().set_position(4, 12),
            sprites["wyiex"].clone().set_position(3, 12),
            sprites["wyiex"].clone().set_position(2, 12),
            sprites["wyiex"].clone().set_position(1, 12),
            sprites["wyiex"].clone().set_position(0, 12),
            sprites["wyiex"].clone().set_position(9, 12),
            sprites["wyiex"].clone().set_position(8, 12),
            sprites["wyiex"].clone().set_position(7, 12),
            sprites["wyiex"].clone().set_position(6, 12),
            sprites["wyiex"].clone().set_position(5, 12),
            sprites["wyiex"].clone().set_position(12, 12),
            sprites["wyiex"].clone().set_position(11, 12),
            sprites["wyiex"].clone().set_position(10, 12),
        ],
        grid_size=(13, 13),
        data={
            "npwxa": [6, 15],
        },
    ),
    # Level 3
    Level(
        sprites=[
            sprites["cvcer"].clone().set_position(1, 3),
            sprites["cvcer"].clone().set_position(6, 2),
            sprites["cvcer"].clone().set_position(8, 6),
            sprites["jggua-Level1"].clone(),
            sprites["qzfkx-ubwff-crkfz"].clone().set_position(8, 10),
            sprites["qzfkx-ubwff-idtiq"].clone().set_position(4, 10),
        ],
        grid_size=(13, 13),
        data={
            "npwxa": [15, 8],
        },
    ),
    # Level 4
    Level(
        sprites=[
            sprites["cvcer"].clone().set_position(5, 5),
            sprites["jggua-Level9"].clone(),
            sprites["qzfkx-ubwff-crkfz"].clone().set_position(8, 4),
            sprites["qzfkx-ubwff-idtiq"].clone().set_position(2, 6),
            sprites["wyiex"].clone().set_position(1, 1),
            sprites["wyiex"].clone().set_position(2, 1),
            sprites["wyiex"].clone().set_position(3, 1),
            sprites["wyiex"].clone().set_position(9, 1),
            sprites["wyiex"].clone().set_position(8, 1),
            sprites["wyiex"].clone().set_position(7, 1),
            sprites["wyiex"].clone().set_position(7, 9),
            sprites["wyiex"].clone().set_position(9, 9),
            sprites["wyiex"].clone().set_position(8, 9),
            sprites["wyiex"].clone().set_position(3, 9),
            sprites["wyiex"].clone().set_position(2, 9),
            sprites["wyiex"].clone().set_position(1, 9),
            sprites["wyiex"].clone().set_position(4, 6),
            sprites["wyiex"].clone().set_position(5, 6),
            sprites["wyiex"].clone().set_position(6, 6),
            sprites["wyiex"].clone().set_position(4, 4),
            sprites["wyiex"].clone().set_position(5, 4),
            sprites["wyiex"].clone().set_position(6, 4),
        ],
        grid_size=(11, 11),
        data={
            "npwxa": [11, 15],
        },
    ),
    # Level 5
    Level(
        sprites=[
            sprites["dfnuk-qeazm"].clone().set_position(10, 9).set_rotation(90),
            sprites["dfnuk-qeazm"].clone().set_position(10, 5).set_rotation(90),
            sprites["dfnuk-raixb"].clone().set_position(3, 5).set_rotation(90),
            sprites["dfnuk-ujcze"].clone().set_position(2, 9).set_rotation(90),
            sprites["hnutp-qeazm"].clone().set_position(3, 12),
            sprites["hnutp-qeazm"].clone().set_position(3, 1),
            sprites["hnutp-raixb"].clone().set_position(8, 6),
            sprites["hnutp-ujcze"].clone().set_position(14, 6),
            sprites["jggua-Level2"].clone(),
            sprites["qzfkx-ubwff-crkfz"].clone().set_position(1, 12),
            sprites["qzfkx-ubwff-idtiq"].clone().set_position(13, 12),
        ],
        grid_size=(15, 15),
        data={
            "npwxa": [6, 7],
        },
    ),
    # Level 6
    Level(
        sprites=[
            sprites["cvcer"].clone().set_position(6, 9),
            sprites["dfnuk-raixb"].clone().set_position(2, 6).set_rotation(90),
            sprites["dfnuk-ujcze"].clone().set_position(8, 6).set_rotation(90),
            sprites["hnutp-raixb"].clone().set_position(9, 2),
            sprites["hnutp-raixb"].clone().set_position(3, 9),
            sprites["hnutp-ujcze"].clone().set_position(3, 2),
            sprites["qzfkx-ubwff-crkfz"].clone().set_position(9, 4),
            sprites["qzfkx-ubwff-idtiq"].clone().set_position(3, 4),
            sprites["wyiex"].clone(),
            sprites["wyiex"].clone().set_position(1, 0),
            sprites["wyiex"].clone().set_position(2, 0),
            sprites["wyiex"].clone().set_position(3, 0),
            sprites["wyiex"].clone().set_position(4, 0),
            sprites["wyiex"].clone().set_position(5, 0),
            sprites["wyiex"].clone().set_position(6, 0),
            sprites["wyiex"].clone().set_position(7, 0),
            sprites["wyiex"].clone().set_position(8, 0),
            sprites["wyiex"].clone().set_position(9, 0),
            sprites["wyiex"].clone().set_position(10, 0),
            sprites["wyiex"].clone().set_position(11, 0),
            sprites["wyiex"].clone().set_position(12, 0),
            sprites["wyiex"].clone().set_position(0, 12),
            sprites["wyiex"].clone().set_position(1, 12),
            sprites["wyiex"].clone().set_position(2, 12),
            sprites["wyiex"].clone().set_position(3, 12),
            sprites["wyiex"].clone().set_position(4, 12),
            sprites["wyiex"].clone().set_position(5, 12),
            sprites["wyiex"].clone().set_position(6, 12),
            sprites["wyiex"].clone().set_position(7, 12),
            sprites["wyiex"].clone().set_position(8, 12),
            sprites["wyiex"].clone().set_position(9, 12),
            sprites["wyiex"].clone().set_position(10, 12),
            sprites["wyiex"].clone().set_position(11, 12),
            sprites["wyiex"].clone().set_position(12, 12),
            sprites["wyiex"].clone().set_position(0, 11),
            sprites["wyiex"].clone().set_position(0, 9),
            sprites["wyiex"].clone().set_position(0, 10),
            sprites["wyiex"].clone().set_position(0, 5),
            sprites["wyiex"].clone().set_position(0, 6),
            sprites["wyiex"].clone().set_position(0, 7),
            sprites["wyiex"].clone().set_position(0, 8),
            sprites["wyiex"].clone().set_position(0, 1),
            sprites["wyiex"].clone().set_position(0, 2),
            sprites["wyiex"].clone().set_position(0, 3),
            sprites["wyiex"].clone().set_position(0, 4),
            sprites["wyiex"].clone().set_position(12, 11),
            sprites["wyiex"].clone().set_position(12, 9),
            sprites["wyiex"].clone().set_position(12, 10),
            sprites["wyiex"].clone().set_position(12, 5),
            sprites["wyiex"].clone().set_position(12, 6),
            sprites["wyiex"].clone().set_position(12, 7),
            sprites["wyiex"].clone().set_position(12, 8),
            sprites["wyiex"].clone().set_position(12, 1),
            sprites["wyiex"].clone().set_position(12, 2),
            sprites["wyiex"].clone().set_position(12, 3),
            sprites["wyiex"].clone().set_position(12, 4),
            sprites["wyiex"].clone().set_position(6, 11),
            sprites["wyiex"].clone().set_position(6, 10),
            sprites["wyiex"].clone().set_position(6, 5),
            sprites["wyiex"].clone().set_position(6, 6),
            sprites["wyiex"].clone().set_position(6, 7),
            sprites["wyiex"].clone().set_position(6, 8),
            sprites["wyiex"].clone().set_position(6, 1),
            sprites["wyiex"].clone().set_position(6, 2),
            sprites["wyiex"].clone().set_position(6, 3),
            sprites["wyiex"].clone().set_position(6, 4),
            sprites["wyiex"].clone().set_position(5, 6),
            sprites["wyiex"].clone().set_position(1, 6),
            sprites["wyiex"].clone().set_position(7, 6),
            sprites["wyiex"].clone().set_position(11, 6),
        ],
        grid_size=(13, 13),
        data={
            "npwxa": [6, 7],
        },
    ),
]
BACKGROUND_COLOR = 5
PADDING_COLOR = 0
_M0R0_COLOR_POOL = [2, 3, 4, 6, 7, 8, 10, 13]


class lradlpwbhu(RenderableUserDisplay):
    """."""

    def __init__(self, fehzjtpngn: int):
        """."""
        self.fehzjtpngn = fehzjtpngn
        self.current_steps = fehzjtpngn

    def ekyafbirsw(self, pdbnnrynof: int) -> None:
        """."""
        self.current_steps = max(0, min(pdbnnrynof, self.fehzjtpngn))

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        """."""
        if self.fehzjtpngn == 0:
            return frame
        qjawlaefdi = self.current_steps / self.fehzjtpngn
        dilxnftulx = round(64 * qjawlaefdi)
        dilxnftulx = min(dilxnftulx, 64)
        for x in range(64):
            if x < dilxnftulx:
                frame[0, x] = 5
            else:
                frame[0, x] = 0
        for x in range(64):
            if 63 - x < dilxnftulx:
                frame[63, x] = 5
            else:
                frame[63, x] = 0
        return frame


class kyrgqgqaes(RenderableUserDisplay):
    """."""

    def __init__(self, rirqhjdvjw: "M0r0") -> None:
        self.rirqhjdvjw = rirqhjdvjw

    def render_interface(self, frame: np.ndarray) -> np.ndarray:
        """."""
        dbgvboxdha = self.rirqhjdvjw.current_level.get_data("npwxa")
        if dbgvboxdha and len(dbgvboxdha) >= 2:
            color1, color2 = (dbgvboxdha[0], dbgvboxdha[1])
            zhnunfpjjg = frame == 0
            if np.any(zhnunfpjjg):
                rmojqczxdn = 0
                for vgnqmdpdwr in [
                    "qzfkx-ubwff-idtiq",
                    "qzfkx-ubwff-crkfz",
                    "qzfkx-kncqr-idtiq",
                    "qzfkx-kncqr-crkfz",
                ]:
                    if vgnqmdpdwr not in self.rirqhjdvjw.fqunj:
                        sprites = self.rirqhjdvjw.current_level.get_sprites_by_name(vgnqmdpdwr)
                        if sprites and sprites[0].interaction != InteractionMode.REMOVED:
                            rmojqczxdn += 1
                tadnosdfoe = np.zeros((64, 64), dtype=np.int8)
                if rmojqczxdn == 4:
                    tadnosdfoe[:32, :32] = color1
                    tadnosdfoe[:32, 32:] = color2
                    tadnosdfoe[32:, :32] = color2
                    tadnosdfoe[32:, 32:] = color1
                else:
                    tadnosdfoe[:, :32] = color1
                    tadnosdfoe[:, 32:] = color2
                frame[zhnunfpjjg] = tadnosdfoe[zhnunfpjjg]
        grid_width, grid_height = self.rirqhjdvjw.current_level.grid_size or (64, 64)
        scale_x = 64 // grid_width
        scale_y = 64 // grid_height
        scale = min(scale_x, scale_y)
        scaled_width = grid_width * scale
        scaled_height = grid_height * scale
        x_offset = (64 - scaled_width) // 2
        y_offset = (64 - scaled_height) // 2
        ufhpknwkaw = self.rirqhjdvjw.current_level.get_sprites_by_name("cvcer")
        for camdxjtorn in ufhpknwkaw:
            if camdxjtorn.is_visible:
                aexzazbxyb = camdxjtorn.x * scale + x_offset
                cpgjocttbt = camdxjtorn.y * scale + y_offset
                for y in range(cpgjocttbt, min(cpgjocttbt + scale, 64)):
                    for x in range(aexzazbxyb, min(aexzazbxyb + scale, 64)):
                        if y == cpgjocttbt or y == cpgjocttbt + scale - 1 or x == aexzazbxyb or (x == aexzazbxyb + scale - 1):
                            frame[y, x] = 5
        for zpatktnghj in self.rirqhjdvjw.current_level.get_sprites_by_name("wyiex"):
            if self.rirqhjdvjw.current_level.get_sprite_at(zpatktnghj.x, zpatktnghj.y, "pxwnx"):
                continue
            aexzazbxyb, cpgjocttbt = (
                zpatktnghj.x * scale + x_offset,
                zpatktnghj.y * scale + y_offset,
            )
            for y in range(cpgjocttbt, min(cpgjocttbt + scale, 64)):
                for x in range(aexzazbxyb, min(aexzazbxyb + scale, 64)):
                    if (x + y) % 2 == 1:
                        frame[y, x] = 5
        return frame


class M0r0(AugmentedGame):
    def __init__(self, seed: int | None = None) -> None:
        self._init_augmentation(seed)   # FIRST: level_rng needs the seed
        self._rng = self.level_rng("layout")
        self.bwyvb: dict[str, tuple[int, int]] = {}
        self.fqunj: set[str] = set()
        self.erwkb = kyrgqgqaes(self)
        self.tmbjb = lradlpwbhu(fehzjtpngn=150)
        self.cfwgj: Sprite | None = None
        self.vmcbq = True
        self._qzfkx_color = 10
        self._wyiex_color = 8
        self._cvcer_color = 9
        ekbqh = Camera(background=BACKGROUND_COLOR, letter_box=PADDING_COLOR)
        _instance_levels = _copy_module.deepcopy(levels)
        super().__init__(
            game_id="m0r0",
            levels=_instance_levels,
            camera=ekbqh,
            available_actions=[1, 2, 3, 4, 5, 6],
        )
        self._camera.replace_interface([self.erwkb, self.tmbjb])

    def _randomize_colors(self) -> None:
        colors = self._rng.sample(_M0R0_COLOR_POOL, 5)
        self._qzfkx_color = colors[0]
        self._wyiex_color = colors[3]
        self._cvcer_color = colors[4]
        for name in [
            "qzfkx-ubwff-crkfz",
            "qzfkx-ubwff-idtiq",
            "qzfkx-kncqr-crkfz",
            "qzfkx-kncqr-idtiq",
        ]:
            for sprite in self.current_level.get_sprites_by_name(name):
                sprite.color_remap(None, self._qzfkx_color)
        for sprite in self.current_level.get_sprites_by_name("wyiex"):
            sprite.color_remap(None, self._wyiex_color)
        for sprite in self.current_level.get_sprites_by_name("cvcer"):
            sprite.color_remap(None, self._cvcer_color)
        npwxa = self.current_level.get_data("npwxa")
        if npwxa and len(npwxa) >= 2:
            npwxa[0] = colors[1]
            npwxa[1] = colors[2]

    def on_set_level(self, level: Level) -> None:
        # Re-key per level so the board is a pure function of (seed, level),
        # identical whether the level was jumped to or played into.
        self._rng = self.level_rng("layout")
        """."""
        self._randomize_colors()
        self.bwyvb = {}
        self.fqunj = set()
        self.tmbjb.ekyafbirsw(150)
        self.cfwgj = None
        self.vmcbq = True
        self.yvmbd: list[Sprite] = []
        self.zgdmc = -1
        self.myxvz = [(s, s.x, s.y) for s in self.current_level.get_sprites_by_tag("sys_click")]

    def step(self) -> None:
        """."""
        _action_id = self.screen_action_to_game(self.action.id)
        if self.zgdmc >= 0:
            mnwlrimfrw = self.zgdmc % 2 == 0 and self.zgdmc < 5
            for korvfzhhpg in self.yvmbd:
                korvfzhhpg.color_remap(None, 11 if mnwlrimfrw else self._qzfkx_color)
            self.zgdmc += 1
            if self.zgdmc > 6:
                for sprite, hifgk, wnawh in self.myxvz:
                    sprite.set_position(hifgk, wnawh)
                self.zgdmc = -1
                self.yvmbd = []
                self.complete_action()
            return
        yyzxcbjywp = 150 - self._action_count
        self.tmbjb.ekyafbirsw(yyzxcbjywp)
        if self._action_count > 150:
            self.lose()
            self.complete_action()
            return
        if _action_id == GameAction.ACTION6:
            hifgk = self.action.data["x"]
            wnawh = self.action.data["y"]
            hifgk, wnawh = self.screen_click_to_game(hifgk, wnawh)
            xdikvugkgm = self.camera.display_to_grid(hifgk, wnawh)
            if xdikvugkgm:
                imgcn, cmize = xdikvugkgm
                dxtbgxnfmv = self.current_level.get_sprite_at(imgcn, cmize, tag="sys_click")
                if dxtbgxnfmv and dxtbgxnfmv.name == "cvcer":
                    self.vmcbq = False
                    for vgnqmdpdwr in [
                        "qzfkx-ubwff-idtiq",
                        "qzfkx-ubwff-crkfz",
                        "qzfkx-kncqr-idtiq",
                        "qzfkx-kncqr-crkfz",
                    ]:
                        sprites = self.current_level.get_sprites_by_name(vgnqmdpdwr)
                        if sprites and vgnqmdpdwr not in self.fqunj:
                            sprites[0].color_remap(None, 1)
                    if self.cfwgj:
                        self.cfwgj.color_remap(None, self._cvcer_color)
                    self.cfwgj = dxtbgxnfmv
                    self.set_placeable_sprite(self.cfwgj)
                    self.cfwgj.color_remap(None, 11)
                else:
                    self.vmcbq = True
                    for vgnqmdpdwr in [
                        "qzfkx-ubwff-idtiq",
                        "qzfkx-ubwff-crkfz",
                        "qzfkx-kncqr-idtiq",
                        "qzfkx-kncqr-crkfz",
                    ]:
                        sprites = self.current_level.get_sprites_by_name(vgnqmdpdwr)
                        if sprites and vgnqmdpdwr not in self.fqunj:
                            sprites[0].color_remap(None, self._qzfkx_color)
                    if self.cfwgj:
                        self.cfwgj.color_remap(None, self._cvcer_color)
                        self.cfwgj = None
                        self.set_placeable_sprite(self.cfwgj)
            self.complete_action()
            return
        ujqjq = 0
        bpjkc = 0
        if _action_id == GameAction.ACTION1:
            bpjkc = -1
            self.set_placeable_sprite(None)
        elif _action_id == GameAction.ACTION2:
            bpjkc = 1
            self.set_placeable_sprite(None)
        elif _action_id == GameAction.ACTION3:
            ujqjq = -1
            self.set_placeable_sprite(None)
        elif _action_id == GameAction.ACTION4:
            ujqjq = 1
            self.set_placeable_sprite(None)
        if self.cfwgj and (not self.vmcbq):
            zqmht, bakee = self.current_level.grid_size or (64, 64)
            qdfjpcrtso = self.cfwgj.x + ujqjq
            zjjafxsusd = self.cfwgj.y + bpjkc
            if qdfjpcrtso < 0 or qdfjpcrtso >= zqmht or zjjafxsusd < 0 or (zjjafxsusd >= bakee):
                self.complete_action()
                return
            self.try_move_sprite(self.cfwgj, ujqjq, bpjkc)
            self.dssyxgdgjg()
            self.complete_action()
            return
        if self.vmcbq:
            dawfcldxmo = []
            for vgnqmdpdwr in [
                "qzfkx-ubwff-idtiq",
                "qzfkx-ubwff-crkfz",
                "qzfkx-kncqr-idtiq",
                "qzfkx-kncqr-crkfz",
            ]:
                sprites = self.current_level.get_sprites_by_name(vgnqmdpdwr)
                if sprites and vgnqmdpdwr not in self.fqunj:
                    dawfcldxmo.append((vgnqmdpdwr, sprites[0]))
            for name, sprite in dawfcldxmo:
                self.bwyvb[name] = (sprite.x, sprite.y)
            bmqdozyzak = []
            for name, sprite in dawfcldxmo:
                if "ubwff-idtiq" in name:
                    bmqdozyzak.append((sprite, ujqjq, bpjkc))
                elif "ubwff-crkfz" in name:
                    bmqdozyzak.append((sprite, -ujqjq, bpjkc))
                elif "kncqr-idtiq" in name:
                    bmqdozyzak.append((sprite, ujqjq, -bpjkc))
                elif "kncqr-crkfz" in name:
                    bmqdozyzak.append((sprite, -ujqjq, -bpjkc))
            for sprite, varawtappn, eajabdanyq in bmqdozyzak:
                if varawtappn != 0 or eajabdanyq != 0:
                    self.bpcdxdwyxx(sprite, varawtappn, eajabdanyq)
            for oruecfkkkl, eijxdryohv in dawfcldxmo:
                if self.current_level.get_sprite_at(eijxdryohv.x, eijxdryohv.y, "wyiex"):
                    self.yvmbd.append(eijxdryohv)
            if self.yvmbd:
                self.zgdmc = 0
                return
            yjzqwbpocl = [(name, sprite) for name, sprite in dawfcldxmo if name not in self.fqunj and sprite.interaction != InteractionMode.REMOVED]
            for i, (mqxkntcvbf, gyqirossdl) in enumerate(yjzqwbpocl):
                if mqxkntcvbf in self.fqunj:
                    continue
                for niyztfksgo, lnnwzdhlte in yjzqwbpocl[i + 1 :]:
                    if niyztfksgo in self.fqunj:
                        continue
                    if mqxkntcvbf in self.bwyvb and niyztfksgo in self.bwyvb:
                        wvvjk, znymb = self.bwyvb[mqxkntcvbf]
                        heyuv, phuxi = self.bwyvb[niyztfksgo]
                        if abs(wvvjk - heyuv) == 1 and znymb == phuxi:
                            if gyqirossdl.x == heyuv and gyqirossdl.y == phuxi or (lnnwzdhlte.x == wvvjk and lnnwzdhlte.y == znymb):
                                gwahbnacyy = (gyqirossdl.x + lnnwzdhlte.x) // 2
                                evplwyicmj = (gyqirossdl.y + lnnwzdhlte.y) // 2
                                gyqirossdl.set_position(gwahbnacyy, evplwyicmj)
                                lnnwzdhlte.set_position(gwahbnacyy, evplwyicmj)
            enrdvoxrsm: dict[tuple[int, int], list[tuple[str, Sprite]]] = {}
            for name, sprite in yjzqwbpocl:
                csunkjidhi = (sprite.x, sprite.y)
                if csunkjidhi not in enrdvoxrsm:
                    enrdvoxrsm[csunkjidhi] = []
                enrdvoxrsm[csunkjidhi].append((name, sprite))
            for csunkjidhi, kodimqfubx in enrdvoxrsm.items():
                if len(kodimqfubx) == 2:
                    for name, sprite in kodimqfubx:
                        self.fqunj.add(name)
                        sprite.set_interaction(InteractionMode.INTANGIBLE)
                elif len(kodimqfubx) > 2:
                    for name, sprite in kodimqfubx[:2]:
                        self.fqunj.add(name)
                        sprite.set_interaction(InteractionMode.INTANGIBLE)
                    for name, sprite in kodimqfubx[2:]:
                        if name in self.bwyvb:
                            prev_x, prev_y = self.bwyvb[name]
                            sprite.set_position(prev_x, prev_y)
            self.dssyxgdgjg()
            tttipukmnd = sum((1 for name, sprite in dawfcldxmo if name not in self.fqunj and sprite.interaction != InteractionMode.INTANGIBLE))
            if tttipukmnd == 0:
                self.next_level()
        self.complete_action()

    def bpcdxdwyxx(self, sprite: Sprite, dx: int, dy: int) -> None:
        """."""
        grid_width, grid_height = self.current_level.grid_size or (64, 64)
        qdfjpcrtso = sprite.x + dx
        zjjafxsusd = sprite.y + dy
        if qdfjpcrtso < 0 or qdfjpcrtso >= grid_width or zjjafxsusd < 0 or (zjjafxsusd >= grid_height):
            return
        orig_x, orig_y = (sprite.x, sprite.y)
        sprite.move(dx, dy)
        wrystrrgwc = self.current_level.get_sprites_by_tag("jggua")
        for rvyelwtmjg in wrystrrgwc:
            if sprite.collides_with(rvyelwtmjg):
                sprite.set_position(orig_x, orig_y)
                return
        tpolveggza = self.current_level.get_sprites_by_tag("nhiae")
        for camdxjtorn in tpolveggza:
            if sprite.collides_with(camdxjtorn):
                sprite.set_position(orig_x, orig_y)
                return
        for avrqsquxwq in ["raixb", "ujcze", "qeazm"]:
            pbgubcraos = self.current_level.get_sprites_by_name(f"dfnuk-{avrqsquxwq}")
            for enwvpaxztx in pbgubcraos:
                if enwvpaxztx.is_collidable and sprite.collides_with(enwvpaxztx):
                    sprite.set_position(orig_x, orig_y)
                    return

    def dssyxgdgjg(self) -> None:
        """."""
        yjzqwbpocl = []
        for vgnqmdpdwr in [
            "qzfkx-ubwff-idtiq",
            "qzfkx-ubwff-crkfz",
            "qzfkx-kncqr-idtiq",
            "qzfkx-kncqr-crkfz",
        ]:
            if vgnqmdpdwr not in self.fqunj:
                sprites = self.current_level.get_sprites_by_name(vgnqmdpdwr)
                if sprites:
                    yjzqwbpocl.append(sprites[0])
        for avrqsquxwq in ["raixb", "ujcze", "qeazm"]:
            adikitftsk = self.current_level.get_sprites_by_name(f"hnutp-{avrqsquxwq}")
            pbgubcraos = self.current_level.get_sprites_by_name(f"dfnuk-{avrqsquxwq}")
            if not adikitftsk or not pbgubcraos:
                continue
            hhxggdtilq = False
            for latugtiydf in adikitftsk:
                for eijxdryohv in yjzqwbpocl:
                    if eijxdryohv.x == latugtiydf.x and eijxdryohv.y == latugtiydf.y:
                        hhxggdtilq = True
                        break
                if hhxggdtilq:
                    break
            for enwvpaxztx in pbgubcraos:
                if hhxggdtilq:
                    enwvpaxztx.set_interaction(InteractionMode.REMOVED)
                else:
                    enwvpaxztx.set_interaction(InteractionMode.TANGIBLE)
