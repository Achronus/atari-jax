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

"""Factory functions for creating AtariEnv instances."""

import pathlib
import re
import sys
from typing import List, Type

import jax

from atarax.env._compile import DEFAULT_CACHE_DIR, _wrap_with_spinner, setup_cache
from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.env.spec import EnvSpec
from atarax.env.vec_env import VecEnv
from atarax.env.wrappers import AtariPreprocessing, Wrapper
from atarax.games.registry import GAME_IDS

_ENV_ID_RE = re.compile(r"^([^/]+)/(.+)-v(\d+)$")


def _resolve_spec(game_id: str | EnvSpec) -> str:
    """
    Return the ALE name from an `EnvSpec` or `"[engine]/[name]-v[N]"` string.

    Parameters
    ----------
    game_id : str | EnvSpec
        Environment identifier.

    Returns
    -------
    ale_name : str
        Internal ALE game name (e.g. `"breakout"`).

    Raises
    ------
    id_error : ValueError
        If `game_id` is a string that does not match the
        `"[engine]/[name]-v[N]"` format.
    """
    if isinstance(game_id, EnvSpec):
        return game_id.env_name

    m = _ENV_ID_RE.match(game_id)

    if not m:
        raise ValueError(
            f"Invalid environment ID {game_id!r}. "
            "Use an EnvSpec or the format 'atari/<game_name>-v0', "
            "e.g. EnvSpec('atari', 'breakout') or 'atari/breakout-v0'."
        )

    return m.group(2)


def make(
    game_id: str | EnvSpec,
    *,
    params: EnvParams | None = None,
    wrappers: List[Type] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
    show_compile_progress: bool = False,
) -> AtariEnv | Wrapper:
    """
    Create an `AtariEnv`, optionally with wrappers applied.

    Parameters
    ----------
    game_id : str | EnvSpec
        Environment identifier — either an `EnvSpec` (e.g.
        `EnvSpec("atari", "breakout")`) or the canonical string
        `"atari/breakout-v0"`.
    params : EnvParams (optional)
        Environment parameters; defaults to `EnvParams()`.
    wrappers : List[Type] (optional)
        Wrapper classes applied innermost-first around the base env.
        Mutually exclusive with `preset`.
    preset : bool (optional)
        Apply the DQN preprocessing stack (`AtariPreprocessing` wrapper) from
        [Mnih et al., 2015](https://www.nature.com/articles/nature14236).
        Mutually exclusive with `wrappers`.
    jit_compile : bool (optional)
        JIT-compile `reset`, `step`, and `sample` on the first call.
        Default is `True`.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache.  Defaults to
        `~/.cache/atari-jax/xla_cache`. Pass `None` to disable.
    show_compile_progress : bool (optional)
        Display a spinner on the first (compilation) call of each method.
        Default is `False`.

    Returns
    -------
    env : AtariEnv | Wrapper
        Configured environment.

    Raises
    ------
    wrapper_error : ValueError
        If both `wrappers` and `preset` are provided.
    id_error : ValueError
        If `game_id` is a string that does not match the
        `"[engine]/[name]-v[N]"` format.
    """
    ale_name = _resolve_spec(game_id)

    if wrappers is not None and preset:
        raise ValueError("Provide either `wrappers` or `preset`, not both.")

    setup_cache(cache_dir)

    env = AtariEnv(ale_name, params or EnvParams())

    if preset:
        env = AtariPreprocessing(env, h=84, w=84, n_stack=4)
    elif wrappers:
        for wrapper_cls in wrappers:
            env = wrapper_cls(env)

    if jit_compile:
        reset_fn = jax.jit(env.reset)
        step_fn = jax.jit(env.step)
        sample_fn = jax.jit(env.sample)

        if show_compile_progress:
            reset_fn = _wrap_with_spinner(reset_fn, "Compiling reset...")
            step_fn = _wrap_with_spinner(step_fn, "Compiling step...")
            sample_fn = _wrap_with_spinner(sample_fn, "Compiling sample...")

        env.reset = reset_fn
        env.step = step_fn
        env.sample = sample_fn

    return env


def make_vec(
    game_id: str | EnvSpec,
    n_envs: int,
    *,
    params: EnvParams | None = None,
    wrappers: List[Type] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
    show_compile_progress: bool = False,
) -> VecEnv:
    """
    Create a `VecEnv` with `n_envs` parallel environments.

    In JAX, parallelism is achieved via `jax.vmap` rather than
    multiprocessing.  `VecEnv.reset(key)` splits the key into `n_envs`
    sub-keys so each environment starts from a distinct random state.

    Parameters
    ----------
    game_id : str | EnvSpec
        Environment identifier — either an `EnvSpec` (e.g.
        `EnvSpec("atari", "breakout")`) or the canonical string
        `"atari/breakout-v0"`.
    n_envs : int
        Number of parallel environments.
    params : EnvParams (optional)
        Environment parameters; defaults to `EnvParams()`.
    wrappers : List[Type] (optional)
        Wrapper classes applied innermost-first around the base env.
        Mutually exclusive with `preset`.
    preset : bool (optional)
        Apply the DQN preprocessing stack (`AtariPreprocessing` wrapper) from
        [Mnih et al., 2015](https://www.nature.com/articles/nature14236).
        Mutually exclusive with `wrappers`.
    jit_compile : bool (optional)
        JIT-compile all vmapped functions on the first call.
        Default is `True`.
    cache_dir : Path or str or None (optional)
        Directory for the persistent XLA compilation cache.  Defaults to
        `~/.cache/atari-jax/xla_cache`.  Pass `None` to disable.
    show_compile_progress : bool (optional)
        Display a spinner on the first (compilation) call of each vmapped
        method.  Default is `False`.

    Returns
    -------
    vec_env : VecEnv
        Vectorised environment wrapping `n_envs` copies of the configured
        single-instance env.

    Raises
    ------
    wrapper_error : ValueError
        If both `wrappers` and `preset` are provided.
    id_error : ValueError
        If `game_id` is a string that does not match the
        `"[engine]/[name]-v[N]"` format.
    """
    env = make(
        game_id,
        params=params,
        wrappers=wrappers,
        preset=preset,
        jit_compile=False,  # Handled by VecEnv
        cache_dir=cache_dir,
    )

    return VecEnv(
        env,
        n_envs,
        jit_compile=jit_compile,
        show_compile_progress=show_compile_progress,
    )


def precompile_all(
    *,
    params: EnvParams | None = None,
    preset: bool = False,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> None:
    """
    Compile and cache all 57 game environments.

    Runs `reset()` and one `step()` for every game so that JAX traces and
    compiles the full emulation graph and writes the result to the XLA
    persistent cache.  Subsequent `make()` calls for any game will load from
    cache rather than recompiling.

    Useful before mass training runs or large test suites where many different
    games will be used.

    Parameters
    ----------
    params : EnvParams (optional)
        Shared environment parameters. Defaults to `EnvParams()`.
    preset : bool (optional)
        Apply the DQN preprocessing stack to every environment.
    cache_dir : Path | str | None (optional)
        Cache directory.  Defaults to `~/.cache/atari-jax/xla_cache`.
        Pass `None` to disable caching (compilation still occurs but is not
        stored to disk).
    """
    setup_cache(cache_dir)

    game_names = list(GAME_IDS.keys())
    total = len(game_names)
    key = jax.random.PRNGKey(0)

    for i, game_name in enumerate(game_names, start=1):
        spec = EnvSpec("atari", game_name)
        label = f"{spec.id} [{i}/{total}]"

        sys.stdout.write(f"\r\u29ff Compiling {label}...")
        sys.stdout.flush()

        env = make(
            spec,
            params=params,
            preset=preset,
            jit_compile=True,
            cache_dir=None,  # Configured above
        )

        _, state = env.reset(key)
        env.step(state, env.sample(key))

        sys.stdout.write(f"\r\u2713 {label}          \n")
        sys.stdout.flush()

    print(f"All {total} environments compiled and cached.")
