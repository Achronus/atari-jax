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

"""Factory functions for creating `AtariEnv` instances."""

import pathlib
from typing import List, Type

import jax

from atarax.env._base import Env
from atarax.env._compile import DEFAULT_CACHE_DIR
from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.env.spec import EnvSpec
from atarax.env.vec_env import VecEnv
from atarax.env.wrappers import AtariPreprocessing, Wrapper
from atarax.env.wrappers.base import _WrapperFactory
from atarax.env.wrappers.jit_wrapper import JitWrapper
from atarax.games.registry import get_game


def make(
    game_id: str | EnvSpec,
    *,
    params: EnvParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> Env:
    """
    Create a single `AtariEnv`, optionally with wrappers applied.

    Parameters
    ----------
    game_id : str | EnvSpec
        Environment identifier — either an `EnvSpec` (e.g.
        `EnvSpec("atari", "breakout")`) or the canonical string
        `"atari/breakout-v0"`.
    params : EnvParams (optional)
        Environment parameters; defaults to `EnvParams()`.
    wrappers : List[Type[Wrapper] | _WrapperFactory] (optional)
        Wrapper classes or pre-configured factories applied innermost-first
        around the base env.  Mutually exclusive with `preset`.
        Default is `None`.
    preset : bool (optional)
        Apply the DQN preprocessing stack (`AtariPreprocessing`) from
        Mnih et al. (2015).  Mutually exclusive with `wrappers`.
        Default is `False`.
    jit_compile : bool (optional)
        Wrap the env in `JitWrapper` and run one warm-up `reset` + `step`
        to eagerly trigger XLA compilation.  Set to `False` to defer
        compilation to first use.  Default is `True`.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache. Defaults to
        `~/.cache/atarax/xla_cache`. Pass `None` to disable.

    Returns
    -------
    env : Env
        Configured environment, wrapped in `JitWrapper` when
        `jit_compile=True`.

    Raises
    ------
    ValueError
        If both `wrappers` and `preset` are provided, if `game_id` does not
        match the `"engine/game-vN"` format, or if the game is not registered.
    """
    if wrappers is not None and preset:
        raise ValueError("Provide either `wrappers` or `preset`, not both.")

    spec = EnvSpec.parse(game_id)
    game_cls = get_game(spec.env_name)
    env: Env = game_cls(params or EnvParams())

    if preset:
        env = AtariPreprocessing(env, h=84, w=84, n_stack=4)
    elif wrappers:
        for w in wrappers:
            env = w(env)

    if jit_compile:
        env = JitWrapper(env, cache_dir=cache_dir)
        _key = jax.random.PRNGKey(0)
        _, _state = env.reset(_key)
        env.step(_state, env.sample(_key))

    return env


def make_vec(
    game_id: str | EnvSpec,
    n_envs: int,
    *,
    params: EnvParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> VecEnv:
    """
    Create a `VecEnv` with `n_envs` parallel environments.

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
    wrappers : List[Type[Wrapper] | _WrapperFactory] (optional)
        Wrapper classes applied innermost-first.  Mutually exclusive with
        `preset`. Default is `None`.
    preset : bool (optional)
        Apply the DQN preprocessing stack.  Mutually exclusive with
        `wrappers`. Default is `False`.
    jit_compile : bool (optional)
        JIT-compile `reset`, `step`, and `rollout` on the `VecEnv` via
        `jit(vmap(f))`, then run one warm-up `reset` + `step`.
        Default is `True`.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache. Defaults to
        `~/.cache/atarax/xla_cache`. Pass `None` to disable.

    Returns
    -------
    vec_env : VecEnv
        Vectorised environment with JIT-compiled methods when
        `jit_compile=True`.

    Raises
    ------
    ValueError
        If both `wrappers` and `preset` are provided, if `game_id` does not
        match the `"engine/game-vN"` format, or if the game is not registered.
    """
    inner_env = make(
        game_id,
        params=params,
        wrappers=wrappers,
        preset=preset,
        jit_compile=False,
        cache_dir=None,
    )

    vec_env = VecEnv(
        inner_env,
        n_envs,
        jit_compile=jit_compile,
        cache_dir=cache_dir,
    )

    if jit_compile:
        _key = jax.random.PRNGKey(0)
        _, _states = vec_env.reset(_key)
        vec_env.step(_states, vec_env.sample(_key))

    return vec_env


def make_multi(
    game_ids: List[str | EnvSpec],
    *,
    params: EnvParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> List[Env]:
    """
    Create one `Env` per entry in `game_ids`.

    Parameters
    ----------
    game_ids : List[str | EnvSpec]
        Environment identifiers.  Each entry may be a canonical string
        (e.g. `"atari/breakout-v0"`) or an `EnvSpec`.  Use
        `GAME_SPECS` for all registered games or `GAME_GROUPS["atari5"]`
        for a predefined subset.
    params : EnvParams (optional)
        Shared `EnvParams` applied to every environment.
    wrappers : List[Type[Wrapper] | _WrapperFactory] (optional)
        Wrapper stack applied to every environment.  Mutually exclusive
        with `preset`.
    preset : bool (optional)
        Apply the DQN preprocessing stack to every environment.
        Mutually exclusive with `wrappers`.  Default is `False`.
    jit_compile : bool (optional)
        Wrap each env in `JitWrapper` and warm up.  Default is `True`.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache. Defaults to
        `~/.cache/atarax/xla_cache`. Pass `None` to disable.

    Returns
    -------
    envs : List[Env]
        One configured environment per entry in `game_ids`.
    """
    return [
        make(
            gid,
            params=params,
            wrappers=wrappers,
            preset=preset,
            jit_compile=jit_compile,
            cache_dir=cache_dir,
        )
        for gid in game_ids
    ]


def make_multi_vec(
    game_ids: List[str | EnvSpec],
    n_envs: int,
    *,
    params: EnvParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> List[VecEnv]:
    """
    Create one `VecEnv` per entry in `game_ids`.

    Intended for the meta-RL pattern where an outer training loop steps
    sequentially through multiple games, each running `n_envs` parallel
    copies.

    Parameters
    ----------
    game_ids : List[str | EnvSpec]
        Environment identifiers.  Each entry may be a canonical string
        (e.g. `"atari/breakout-v0"`) or an `EnvSpec`.  Use
        `GAME_SPECS` for all registered games or `GAME_GROUPS["atari5"]`
        for a predefined subset.
    n_envs : int
        Number of parallel environments per game.
    params : EnvParams (optional)
        Shared `EnvParams` applied to every environment.
    wrappers : List[Type[Wrapper] | _WrapperFactory] (optional)
        Wrapper stack applied to every environment.  Mutually exclusive
        with `preset`.
    preset : bool (optional)
        Apply the DQN preprocessing stack to every environment.
        Mutually exclusive with `wrappers`.  Default is `False`.
    jit_compile : bool (optional)
        JIT-compile `reset`, `step`, and `rollout` on each `VecEnv` via
        `jit(vmap(f))`, then run one warm-up `reset` + `step` per game.
        Default is `True`.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache. Defaults to
        `~/.cache/atarax/xla_cache`. Pass `None` to disable.

    Returns
    -------
    vec_envs : List[VecEnv]
        One vectorised environment per entry in `game_ids`.
    """
    return [
        make_vec(
            gid,
            n_envs,
            params=params,
            wrappers=wrappers,
            preset=preset,
            jit_compile=jit_compile,
            cache_dir=cache_dir,
        )
        for gid in game_ids
    ]
