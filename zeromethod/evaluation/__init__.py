"""Evaluation and solve-recording utilities for ZeroMethod.

Exports are lazy so running solver_benchmark as a module does not pre-import
the target module and trigger runpy warnings.
"""

__all__ = [
    "RecordedMove",
    "SolveReport",
    "SolveRecorder",
    "SolverBenchmark",
]


def __getattr__(name):
    if name in __all__:
        from . import solver_benchmark
        return getattr(solver_benchmark, name)
    raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
