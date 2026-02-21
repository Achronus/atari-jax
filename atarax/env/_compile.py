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

"""Persistent XLA compilation cache setup and JIT progress spinner."""

import functools
import pathlib
import sys
import threading
from typing import Callable

import jax

DEFAULT_CACHE_DIR = pathlib.Path.home() / ".cache" / "atari-jax" / "xla_cache"

_cache_configured = False
_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
_DONE = "✓"


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


def _wrap_with_spinner(fn: Callable, message: str) -> Callable:
    """
    Wrap `fn` to display a progress spinner on the first (compilation) call.

    Subsequent calls bypass the spinner entirely.  Uses only stdlib
    (`threading`, `sys`) — no extra dependencies required.

    Parameters
    ----------
    fn : Callable
        Function to wrap (typically a `jax.jit`-compiled function).
    message : str
        Text displayed beside the spinner frame, e.g. `"Compiling reset..."`.

    Returns
    -------
    wrapper : Callable
        Wrapped function with identical signature.
    """
    first_call = [True]

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if not first_call[0]:
            return fn(*args, **kwargs)

        first_call[0] = False
        stop = threading.Event()

        def _spin():
            i = 0
            while not stop.is_set():
                sys.stdout.write(f"\r{_FRAMES[i % len(_FRAMES)]} {message}")
                sys.stdout.flush()
                i += 1
                stop.wait(0.1)

            sys.stdout.write(f"\r{_DONE} {message}\n")
            sys.stdout.flush()

        t = threading.Thread(target=_spin, daemon=True)
        t.start()

        try:
            result = fn(*args, **kwargs)
        finally:
            stop.set()
            t.join()

        return result

    return wrapper
