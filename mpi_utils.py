# mpiwrap.py
from functools import wraps

from mpi4py import MPI

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()

_REGISTRY = {}
_STOP = "__stop__"
_in_parallel = False


def mpi(func):
    """Mark a function as collective. On rank 0, calling it signals the
    parked workers to enter the same function, then all ranks run the body
    together. Nested @mpi calls do not re-signal, since the workers are
    already inside the region."""
    name = f"{func.__module__}.{func.__qualname__}"
    _REGISTRY[name] = func

    @wraps(func)
    def wrapper(*args, **kwargs):
        global _in_parallel
        if _in_parallel:  # already collective, just run
            return func(*args, **kwargs)
        if RANK == 0:
            COMM.bcast((name, args, kwargs), root=0)
        _in_parallel = True
        try:
            return func(*args, **kwargs)
        finally:
            _in_parallel = False

    return wrapper


def _worker_loop():
    global _in_parallel
    while True:
        name, args, kwargs = COMM.bcast(None, root=0)
        if name == _STOP:
            break
        _in_parallel = True
        try:
            _REGISTRY[name](*args, **kwargs)  # return value discarded
        finally:
            _in_parallel = False


def mpi_run(main):
    """Entry point. Rank 0 runs main(). All other ranks park in the dispatch
    loop and only wake for @mpi functions. When main returns, the workers are
    released."""
    if RANK == 0:
        try:
            return main()
        finally:
            COMM.bcast((_STOP, (), {}), root=0)
    _worker_loop()
    return None
