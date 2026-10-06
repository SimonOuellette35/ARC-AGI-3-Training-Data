"""Restricted, resource-limited subprocess for generated frame classifiers.

No torch, transformers, engines, filesystem/network helpers, or imports are
exposed to generated functions. The parent enforces a wall-clock deadline.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import resource
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goal_verifier as gv  # noqa: E402

NUMPY_NAMES = set("abs absolute all allclose any arange argmax argmin argwhere array array_equal "
                  "asarray bincount bool_ clip concatenate count_nonzero diff empty empty_like "
                  "equal flatnonzero float32 float64 full full_like gcd indices int8 int16 int32 "
                  "int64 isin isfinite logical_and logical_not logical_or max maximum mean median "
                  "min minimum ndarray nonzero ones ones_like pad r_ rot90 sign sort sqrt stack "
                  "sum uint8 unique unravel_index where zeros zeros_like inf nan "
                  "repeat fliplr flipud ix_ meshgrid percentile rint not_equal shares_memory "
                  "floor uint64 vstack roll iinfo ptp union1d ogrid s_ round".split())
METHODS = set("T all any append argmax argmin argsort astype clear copy count discard dtype extend "
              "flatten get index intersection items keys max mean min ndim nonzero pop ravel "
              "remove reshape reverse shape size sort squeeze sum tolist transpose union update "
              "values item add join setdefault issubset hex tobytes flat reduce".split())
HELPERS = {name for name in vars(gv) if not name.startswith('_') and callable(getattr(gv, name))
           and getattr(getattr(gv, name), '__module__', None) == gv.__name__}
BUILTINS = {name: getattr(__import__('builtins'), name) for name in (
    "abs all any bool dict enumerate float int isinstance len list map max min range reversed "
    "round set slice sorted str sum tuple zip Exception ValueError TypeError IndexError KeyError"
).split()}


def compile_program(source):
    if len(source) > 100_000:
        raise ValueError("Generated program is too large")
    tree = ast.parse(source)
    if not tree.body or any(not isinstance(n, ast.FunctionDef) for n in tree.body):
        raise ValueError("Only function definitions are allowed at module scope")
    functions = [n for n in tree.body if n.name == 'is_win']
    if len(functions) != 1:
        raise ValueError("Expected exactly one is_win function")
    args = functions[0].args
    if ([a.arg for a in args.args] != ['frame0', 'current_state', 'action', 'next_state']
            or args.posonlyargs or args.kwonlyargs or args.vararg or args.kwarg or args.defaults):
        raise ValueError("is_win must take exactly frame0, current_state, action, next_state")
    forbidden = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
                 ast.AsyncFunctionDef, ast.Await, ast.With, ast.AsyncWith)
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f"Unsupported construct: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError("Dunder names are not available")
        if isinstance(node, ast.Attribute) and node.attr not in NUMPY_NAMES | METHODS | HELPERS:
            raise ValueError(f"Attribute is not available: {node.attr}")
        if isinstance(node, ast.FunctionDef):
            if node.decorator_list:
                raise ValueError("Decorators are not allowed")
            for default in node.args.defaults + node.args.kw_defaults:
                if default is not None:
                    ast.literal_eval(default)  # allow literal helper defaults only
            # Type annotations aren't needed at execution time.
            node.returns = None
            for arg in node.args.args + node.args.kwonlyargs:
                arg.annotation = None
    numeric = SimpleNamespace(**{name: getattr(np, name) for name in NUMPY_NAMES})
    helpers = SimpleNamespace(**{name: getattr(gv, name) for name in HELPERS})
    namespace = {'__builtins__': BUILTINS, 'np': numeric, 'gv': helpers}
    exec(compile(tree, '<generated-is-win>', 'exec'), namespace)
    return namespace['is_win']


def main():
    # Persistent worker: CPU limit is cumulative; wall deadlines are per request.
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    function = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if 'source' in request:
                function = compile_program(request['source'])
                response = {'ok': True}
            else:
                boards = [np.asarray(request[key], dtype=np.int64)
                          for key in ('frame0', 'current', 'next')]
                if any(b.shape != (64, 64) for b in boards):
                    raise ValueError("Expected 64x64 boards")
                result = function(boards[0], boards[1], request['action'], boards[2])
                if not isinstance(result, (bool, np.bool_)):
                    raise ValueError("is_win must return a boolean")
                response = {'ok': True, 'win': bool(result)}
        except Exception as exc:
            response = {'ok': False, 'error': f'{type(exc).__name__}: {exc}'}
        sys.stdout.write(json.dumps(response) + '\n')
        sys.stdout.flush()


if __name__ == '__main__':
    # Limit numerical-library thread pools in this small CPU worker.
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    main()
