# Copyright 2026 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Persistent XLA compilation cache setup and JIT compile-progress wrapper."""

import contextlib
import functools
import pathlib
import threading
from typing import Callable

import jax
from tqdm import tqdm

DEFAULT_CACHE_DIR = pathlib.Path.home() / ".cache" / "atari-jax" / "xla_cache"

_cache_configured = False


def setup_cache(cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR) -> None:
    """
    Configure a persistent XLA compilation cache.

    Idempotent — safe to call multiple times; only the first call takes effect.

    Parameters
    ----------
    cache_dir : Path | str | None
        Directory where compiled XLA kernels are stored.  Defaults to
        `~/.cache/atari-jax/xla_cache`. Pass `None` to disable caching.
    """
    global _cache_configured

    if cache_dir is None or _cache_configured:
        return

    path = pathlib.Path(cache_dir)
    path.mkdir(parents=True, exist_ok=True)

    jax.config.update("jax_compilation_cache_dir", str(path))
    _cache_configured = True


@contextlib.contextmanager
def _live_bar(bar: tqdm, interval: float = 0.1):
    """
    Refresh *bar* from a background thread while the body executes.

    tqdm only redraws when `update()` is called, so without this the elapsed
    timer appears frozen during a long blocking operation such as XLA
    compilation.

    Parameters
    ----------
    bar : tqdm
        The progress bar instance to refresh.
    interval : float (optional)
        Seconds between refreshes. Default is `0.1`.
    """
    stop = threading.Event()

    def _spin():
        while not stop.is_set():
            bar.refresh()
            stop.wait(interval)

    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    try:
        yield
    finally:
        stop.set()
        t.join()


def _wrap_with_tqdm(
    fns_and_labels: list[tuple[Callable, str]],
) -> list[Callable]:
    """
    Wrap a list of functions to share a single tqdm compile-progress bar.

    Each function advances the bar by one step on its first call.
    Subsequent calls bypass the progress display entirely.  The elapsed
    timer updates live while each compilation is in progress.

    Parameters
    ----------
    fns_and_labels : list[tuple[Callable, str]]
        Pairs of `(function, description)` to wrap.  The bar total equals
        `len(fns_and_labels)`.

    Returns
    -------
    wrapped : list[Callable]
        Wrapped functions in the same order, with identical signatures.
    """
    total = len(fns_and_labels)
    bar = tqdm(total=total, desc="Compiling...", leave=True)
    completed = [0]

    wrapped = []
    for fn, label in fns_and_labels:
        first_call = [True]

        @functools.wraps(fn)
        def wrapper(*args, _fn=fn, _label=label, _first_call=first_call, **kwargs):
            if not _first_call[0]:
                return _fn(*args, **kwargs)

            _first_call[0] = False
            bar.set_description(_label)
            with _live_bar(bar):
                result = _fn(*args, **kwargs)
            bar.update(1)
            completed[0] += 1

            if completed[0] == total:
                bar.close()

            return result

        wrapped.append(wrapper)

    return wrapped
