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

from typing import TYPE_CHECKING, Any, Dict, Tuple

import chex
import jax.numpy as jnp

if TYPE_CHECKING:
    from atarax.env.atari_env import AtariEnv

from atarax.core.state import AtariState
from atarax.env.spaces import Box
from atarax.env.wrappers.base import Wrapper


class NormalizeObservation(Wrapper):
    """
    Normalises pixel observations from `uint8 [0, 255]` to `float32 [0, 1]`.

    Divides observations by 255.0 and casts to float32. Equivalent to
    Gymnasium's `NormalizeObservation` for pixel environments.

    Parameters
    ----------
    env : AtariEnv | Wrapper
        Environment to wrap.

    Examples
    --------
    >>> env = NormalizeObservation(make("atari/breakout-v0"))
    >>> obs, state = env.reset(key)   # obs: float32[210, 160, 3] in [0, 1]
    """

    def __init__(self, env: "AtariEnv | Wrapper") -> None:
        super().__init__(env)

    def reset(self, key: chex.Array) -> Tuple[chex.Array, AtariState]:
        """
        Reset and return a normalised initial observation.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key.

        Returns
        -------
        obs : chex.Array
            Normalised observation, float32 in [0, 1].
        state : AtariState
            Inner environment state.
        """
        obs, state = self._env.reset(key)
        return obs.astype(jnp.float32) / jnp.float32(255.0), state

    def step(
        self, state: Any, action: chex.Array
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step and return a normalised observation.

        Parameters
        ----------
        state : Any
            Current environment state.
        action : chex.Array
            Action to take.

        Returns
        -------
        obs : chex.Array
            Normalised observation, float32 in [0, 1].
        new_state : Any
            Updated environment state.
        reward : chex.Array
            Step reward.
        done : chex.Array
            Terminal flag.
        info : Dict[str, Any]
            Environment metadata.
        """
        obs, new_state, reward, done, info = self._env.step(state, action)
        return (
            obs.astype(jnp.float32) / jnp.float32(255.0),
            new_state,
            reward,
            done,
            info,
        )

    @property
    def observation_space(self) -> Box:
        """
        Observation space with `float32` dtype and `[0, 1]` bounds.

        Returns
        -------
        space : Box
            Updated observation space.
        """
        inner = self._env.observation_space
        return Box(low=0.0, high=1.0, shape=inner.shape, dtype=jnp.float32)
