"""Evaluation deadlines and read-only enumeration of the existing test harness."""
from collections import Counter
import contextlib
import io
from pathlib import Path
import runpy
import sys
from unittest.mock import patch
import time


class EvaluationDeadline(Exception):
    """The allotted real evaluation time has elapsed."""


class BudgetedEnv:
    def __init__(self, env, deadline):
        self.env = env
        self.deadline = deadline
        self.steps = 0
        self.start_levels = None

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _check(self):
        if time.perf_counter() >= self.deadline:
            raise EvaluationDeadline()

    def reset(self):
        self._check()
        span = self.env.reset()
        self.start_levels = self.env.levels_completed
        self._check()
        return span

    @property
    def available_actions(self):
        self._check()
        return self.env.available_actions

    def step(self, action, xy=None):
        self._check()
        result = self.env.step(action, xy)
        self.steps += 1
        return result

    @property
    def cleared_levels(self):
        return 0 if self.start_levels is None else self.env.levels_completed-self.start_levels


def count_harness_cases(path, arguments):
    """Enumerate the exact requested cases without loading models or stepping games.

    The inner invocation has no time budget, which prevents recursion. Existing
    game filters and corpus-derived case rules are evaluated by the harness itself.
    """
    import solver as S
    forwarded = []
    skip_value = False
    for arg in arguments:
        if skip_value:
            skip_value = False
        elif arg == '--game-seconds':
            skip_value = True
        elif not arg.startswith('--game-seconds='):
            forwarded.append(arg)
    counts = Counter()
    def collect(game, seed, level):
        counts[game] += 1
        return object(), 'case enumeration'
    path = Path(path)
    with patch.object(S,'make_env',collect), \
         patch.object(S,'solve',return_value=(False,0,0)), \
         patch.object(S,'load_policy',return_value=(None,None)), \
         patch.object(S,'planner_from_args',return_value=None), \
         patch.object(sys,'argv',[str(path),*forwarded]), \
         contextlib.redirect_stdout(io.StringIO()):
        runpy.run_path(str(path),run_name='__main__')
    return dict(counts)
