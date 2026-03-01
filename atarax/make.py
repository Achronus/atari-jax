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

import pathlib
from typing import List, Tuple, Type

import jax

from atarax._compile import DEFAULT_CACHE_DIR, setup_cache
from atarax.game import AtaraxGame, AtaraxParams
from atarax.games.registry import get_game
from atarax.spec import EnvSpec
from atarax.wrappers import AtariPreprocessing, VmapEnv, Wrapper, _WrapperFactory
from atarax.wrappers.jit_wrapper import JitWrapper


def make(
    game_id: str | EnvSpec,
    *,
    params: AtaraxParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> Tuple[AtaraxGame, AtaraxParams]:
    """
    Create a single `AtaraxGame`, optionally with wrappers applied.

    Parameters
    ----------
    game_id : str | EnvSpec
        Environment identifier — either an `EnvSpec` (e.g.
        `EnvSpec("atari", "breakout")`) or the canonical string
        `"atari/breakout-v0"`.
    params : AtaraxParams (optional)
        Environment parameters; defaults to `AtaraxParams()`.
    wrappers : List[Type[Wrapper] | _WrapperFactory] (optional)
        Wrapper classes or pre-configured factories applied innermost-first
        around the base env.  Mutually exclusive with `preset`.
    preset : bool (optional)
        Apply the DQN preprocessing stack (`AtariPreprocessing`).
        Mutually exclusive with `wrappers`. Default is `False`.
    jit_compile : bool (optional)
        Wrap the env in `JitWrapper` and run one warm-up `reset` + `step`
        to eagerly trigger XLA compilation. Default is `True`.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache.
        Defaults to `~/.cache/atarax/xla_cache`. Pass `None` to disable.

    Returns
    -------
    env : AtaraxGame
        Configured environment, wrapped in `JitWrapper` when `jit_compile=True`.
    params : AtaraxParams
        Environment parameters used by the environment.

    Raises
    ------
    ValueError
        If both `wrappers` and `preset` are provided, if `game_id` does not
        match the `"engine/game-vN"` format, or if the game is not registered.
    """
    if wrappers is not None and preset:
        raise ValueError("Provide either `wrappers` or `preset`, not both.")

    game_cls = get_game(game_id)
    env = game_cls()
    params = params or AtaraxParams()

    if preset:
        env = AtariPreprocessing(env, h=84, w=84, n_stack=4)
    elif wrappers:
        for w in wrappers:
            env = w(env)

    if jit_compile:
        env = JitWrapper(env, cache_dir=cache_dir)
        _key = jax.random.PRNGKey(0)
        _, _state = env.reset(_key, params)
        env.step(_key, _state, env.action_space.sample(_key), params)

    return env, params


def make_vec(
    game_id: str | EnvSpec,
    n_envs: int,
    *,
    params: AtaraxParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> Tuple[VmapEnv, AtaraxParams]:
    """
    Create a `VmapEnv` with `n_envs` parallel environments.

    Parameters
    ----------
    game_id : str | EnvSpec
        Environment identifier.
    n_envs : int
        Number of parallel environments.
    params : AtaraxParams (optional)
        Environment parameters; defaults to `AtaraxParams()`.
    wrappers : List[Type[Wrapper] | _WrapperFactory] (optional)
        Wrapper classes applied innermost-first. Mutually exclusive with
        `preset`.
    preset : bool (optional)
        Apply the DQN preprocessing stack. Mutually exclusive with
        `wrappers`. Default is `False`.
    jit_compile : bool (optional)
        Run one warm-up `reset` + `step` to eagerly trigger XLA compilation.
        Default is `True`.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache.

    Returns
    -------
    vec_env : VmapEnv
        Vectorised environment.
    params : AtaraxParams
        Environment parameters used by the environment.
    """
    inner_env, params = make(
        game_id,
        params=params,
        wrappers=wrappers,
        preset=preset,
        jit_compile=False,
        cache_dir=None,
    )

    vec_env = VmapEnv(inner_env, n_envs)

    if jit_compile:
        setup_cache(cache_dir)
        _key = jax.random.PRNGKey(0)
        _, _states = vec_env.reset(_key, params)
        vec_env.step(
            _key, _states, jax.numpy.zeros(n_envs, dtype=jax.numpy.int32), params
        )

    return vec_env, params


def make_multi(
    game_ids: List[str | EnvSpec],
    *,
    params: AtaraxParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> List[Tuple[AtaraxGame, AtaraxParams]]:
    """Create one `(AtaraxGame, AtaraxParams)` tuple per entry in `game_ids`."""
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
    params: AtaraxParams | None = None,
    wrappers: List[Type[Wrapper] | _WrapperFactory] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
    cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
) -> List[Tuple[VmapEnv, AtaraxParams]]:
    """Create one `(VmapEnv, AtaraxParams)` tuple per entry in `game_ids`."""
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
