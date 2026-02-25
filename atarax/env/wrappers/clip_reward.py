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

from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp

from atarax.env._base import Env
from atarax.env.wrappers.base import Wrapper


class ClipReward(Wrapper):
    """
    Clip rewards to the sign of the reward: `{−1, 0, +1}`.

    Parameters
    ----------
    env : Env
        Inner environment to wrap.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)

    def reset(self, key: chex.Array) -> Tuple[chex.Array, Any]:
        return self._env.reset(key)

    def step(
        self,
        state,
        action: chex.Array,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment by one step and clip the reward to `{−1, 0, +1}`.

        Parameters
        ----------
        state : Any
            Current environment state.
        action : chex.Array
            int32 — ALE action index.

        Returns
        -------
        obs : chex.Array
            Observation from the inner step.
        new_state : Any
            Updated environment state.
        reward : chex.Array
            float32 — Reward clipped to sign: `{−1.0, 0.0, +1.0}`.
        done : chex.Array
            bool — Terminal flag from the inner step.
        info : Dict[str, Any]
            `Info dict from the inner step.
        """
        obs, new_state, reward, done, info = self._env.step(state, action)
        reward = jnp.sign(reward).astype(jnp.float32)

        return obs, new_state, reward, done, info
