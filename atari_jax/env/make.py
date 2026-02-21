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

from typing import List, Type, Union

import jax

from atari_jax.env.atari_env import AtariEnv, EnvParams
from atari_jax.env.vec_env import VecEnv
from atari_jax.env.wrappers import (
    BaseWrapper,
    ClipRewardWrapper,
    EpisodicLifeWrapper,
    FrameStackWrapper,
    GrayscaleWrapper,
    ResizeWrapper,
)


def make(
    game_id: str,
    *,
    params: EnvParams | None = None,
    wrappers: List[Type] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
) -> Union[AtariEnv, BaseWrapper]:
    """
    Create an `AtariEnv`, optionally with wrappers applied.

    Parameters
    ----------
    game_id : str
        ALE game name (e.g. `"breakout"`).
    params : EnvParams (optional)
        Environment parameters; defaults to `EnvParams()`.
    wrappers : List[Type] (optional)
        Wrapper classes applied innermost-first around the base env.
        Mutually exclusive with `preset`.
    preset : bool (optional)
        Apply the DQN preprocessing stack from
        [Mnih et al., 2015](https://www.nature.com/articles/nature14236).
        Mutually exclusive with `wrappers`.

        Wrappers applied (in order):

        - `GrayscaleWrapper`
        - `ResizeWrapper` (84×84)
        - `FrameStackWrapper` (4 frames)
        - `ClipRewardWrapper`
        - `EpisodicLifeWrapper`

    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Recommended `True` to reduce training
        speed. Default is `True`

    Returns
    -------
    env : AtariEnv | BaseWrapper
        Configured environment.

    Raises
    ------
    wrapper_error : ValueError
        If both `wrappers` and `preset` are provided.
    """
    if wrappers is not None and preset:
        raise ValueError("Provide either `wrappers` or `preset`, not both.")

    env: Union[AtariEnv, BaseWrapper] = AtariEnv(game_id, params or EnvParams())

    if preset:
        env = GrayscaleWrapper(env)
        env = ResizeWrapper(env, h=84, w=84)
        env = FrameStackWrapper(env, n_stack=4)
        env = ClipRewardWrapper(env)
        env = EpisodicLifeWrapper(env)
    elif wrappers:
        for wrapper_cls in wrappers:
            env = wrapper_cls(env)

    if jit_compile:
        env.reset = jax.jit(env.reset)  # type: ignore[method-assign]
        env.step = jax.jit(env.step)  # type: ignore[method-assign]
        env.sample = jax.jit(env.sample)  # type: ignore[method-assign]

    return env


def make_vec(
    game_id: str,
    n_envs: int,
    *,
    params: EnvParams | None = None,
    wrappers: List[Type] | None = None,
    preset: bool = False,
    jit_compile: bool = True,
) -> VecEnv:
    """
    Create a `VecEnv` with `n_envs` parallel environments.

    In JAX, parallelism is achieved via `jax.vmap` rather than
    multiprocessing.  `VecEnv.reset(key)` splits the key into `n_envs`
    sub-keys so each environment starts from a distinct random state.

    Parameters
    ----------
    game_id : str
        ALE game name (e.g. `"breakout"`).
    n_envs : int
        Number of parallel environments.
    params : EnvParams (optional)
        Environment parameters; defaults to `EnvParams()`.
    wrappers : List[Type] (optional)
        Wrapper classes applied innermost-first around the base env.
        Mutually exclusive with `preset`.
    preset : bool (optional)
        Apply the DQN preprocessing stack from
        [Mnih et al., 2015](https://www.nature.com/articles/nature14236).
        Mutually exclusive with `wrappers`.
    jit_compile : bool (optional)
        Flag to enable/disable JIT compilation. Recommended `True` to reduce training
        speed. Default is `True`

    Returns
    -------
    vec_env : VecEnv
        Vectorised environment wrapping `n_envs` copies of the configured
        single-instance env.

    Raises
    ------
    wrapper_error : ValueError
        If both `wrappers` and `preset` are provided.
    """
    env = make(game_id, params=params, wrappers=wrappers, preset=preset)
    return VecEnv(env, n_envs, jit_compile=jit_compile)
