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
from typing import List, Type

import jax
import jax.numpy as jnp
from tqdm import tqdm

from atarax.env._compile import DEFAULT_CACHE_DIR, _live_bar, _wrap_with_tqdm, setup_cache
from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.env.spec import EnvSpec
from atarax.env.vec_env import VecEnv, make_rollout_fn
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
        Mutually exclusive with `preset`. Default is `None`
    preset : bool (optional)
        Apply the DQN preprocessing stack (`AtariPreprocessing` wrapper) from
        [Mnih et al., 2015](https://www.nature.com/articles/nature14236).
        Mutually exclusive with `wrappers`. Default is `False`
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
            reset_fn, step_fn, sample_fn = _wrap_with_tqdm(
                [
                    (reset_fn, "Compiling reset"),
                    (step_fn, "Compiling step"),
                    (sample_fn, "Compiling sample"),
                ]
            )

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
        Mutually exclusive with `preset`. Default is `None`
    preset : bool (optional)
        Apply the DQN preprocessing stack (`AtariPreprocessing` wrapper) from
        [Mnih et al., 2015](https://www.nature.com/articles/nature14236).
        Mutually exclusive with `wrappers`. Default is `False`
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
    n_envs: int = 1,
    n_steps: int | None = None,
    params: EnvParams | None = None,
    wrappers: List[Type] | None = None,
    preset: bool = False,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> None:
    """
    Compile and cache all 57 game environments.

    Runs `reset()`, `step()`, and optionally `rollout()` for every game so
    that JAX traces and compiles the full emulation graph and writes the result
    to the XLA persistent cache.  Subsequent `make()` / `make_vec()` calls for
    any game will load from cache rather than recompiling.

    The cache key depends on the exact input shapes and computation graph, so
    `n_envs`, `n_steps`, `wrappers`, and `preset` must all match how the
    environments will be used in your training loop for cache hits to occur.

    Parameters
    ----------
    n_envs : int (optional)
        Number of parallel environments to compile for.  `1` (default) uses
        `make()` and caches single-env `reset` / `step` shapes.  Any value
        greater than `1` uses `make_vec()` and caches batched shapes.
    n_steps : int (optional)
        If provided, also compiles and caches the rollout function for a
        fixed sequence length of `n_steps`.  This corresponds to `T` in
        the `actions` array passed to `rollout()`.  Leave as `None` to skip
        rollout precompilation. Default is `None`
    params : EnvParams (optional)
        Shared environment parameters. Defaults to `EnvParams()`.
    wrappers : List[Type] (optional)
        Wrapper classes applied innermost-first around the base env.
        Mutually exclusive with `preset`. Default is `None`
    preset : bool (optional)
        Apply the DQN preprocessing stack to every environment.
        Mutually exclusive with `wrappers`. Default is `False`
    cache_dir : Path | str | None (optional)
        Cache directory.  Defaults to `~/.cache/atari-jax/xla_cache`.
        Pass `None` to disable caching (compilation still occurs but is not
        stored to disk).
    """
    setup_cache(cache_dir)

    game_names = list(GAME_IDS.keys())
    key = jax.random.PRNGKey(0)

    print(
        "Hold tight! Compiling and caching all 57 Atari environments. This may take a while..."
    )
    with tqdm(game_names, unit="env") as bar:
        for game_name in bar:
            spec = EnvSpec("atari", game_name)
            bar.set_description(f"Compiling {spec.id}")

            with _live_bar(bar):
                if n_envs == 1:
                    env = make(
                        spec,
                        params=params,
                        wrappers=wrappers,
                        preset=preset,
                        jit_compile=True,
                        cache_dir=None,
                    )
                    _, state = env.reset(key)
                    env.step(state, env.sample(key))

                    if n_steps is not None:
                        rollout_fn = jax.jit(make_rollout_fn(env))
                        rollout_fn(state, jnp.zeros(n_steps, dtype=jnp.int32))
                else:
                    vec_env = make_vec(
                        spec,
                        n_envs=n_envs,
                        params=params,
                        wrappers=wrappers,
                        preset=preset,
                        jit_compile=True,
                        cache_dir=None,
                    )
                    _, states = vec_env.reset(key)
                    vec_env.step(states, jnp.zeros(n_envs, dtype=jnp.int32))

                    if n_steps is not None:
                        vec_env.rollout(
                            states, jnp.zeros((n_envs, n_steps), dtype=jnp.int32)
                        )
