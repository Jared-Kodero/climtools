"""Regression test: BaseProgress._start_progress() must not crash in
interactive mode (Jupyter, or any TTY-attached stdout).

Run directly (no MPI needed, but harmless under mpirun too)::

    python test/test_progress_stream.py

Background
----------
``BaseProgress._start_progress`` calls ``self.save_stream()`` when
``self._interactive`` is True (set whenever running under ipykernel, or
whenever the destination stream is a TTY -- both common in ordinary
interactive use, not just contrived cases). ``save_stream`` requires a
``stream`` argument and every other call site in this module passes one
and assigns the result back (``self._progress_stream =
self.save_stream(self._progress_stream)``, see ``_start_redirect``) --
but ``_start_progress`` called it with no argument at all and discarded
the return value, an immediate ``TypeError`` that made ``to_netcdf()``
(and any other consumer of ``BaseProgress``) crash outright the first
time it ran somewhere interactive, e.g. a Jupyter notebook.

A second, related bug lived in ``save_stream`` itself: given an
already-open stream (the ordinary case on any call after the first),
its ``if stream is not None: return`` branch returned ``None`` instead
of passing the existing stream through, silently discarding it (and
forcing a fresh, wasteful duplicate) on every subsequent call.

This test forces the interactive branch directly (bypassing real TTY
detection, so it's deterministic in any environment) and checks that
starting and stopping progress reporting neither raises nor loses an
already-open stream across repeated starts.
"""

from __future__ import annotations

import io
import sys

from climtools.core.progress import SerialProgressBar


def main() -> None:
    bar = SerialProgressBar(total=5, description="test", file=io.StringIO())
    bar._interactive = True  # force the branch that used to crash

    # First start: no stream yet, save_stream() must create one and the
    # call site must not raise TypeError for a missing argument.
    bar._start_progress(total=5, transient=True, refresh_per_second=10)
    first_stream = bar._progress_stream
    stream_ok = first_stream is not None

    # save_stream() must pass an *existing* stream through unchanged,
    # not silently return None and force a fresh duplicate.
    passthrough = bar.save_stream(first_stream)
    passthrough_ok = passthrough is first_stream

    bar._finish_progress(errored=False)  # restores the redirected fd

    ok = stream_ok and passthrough_ok
    print(f"[{'PASS' if ok else 'FAIL'}] progress stream setup: "
          f"stream_created={stream_ok} passthrough_preserved={passthrough_ok}")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
