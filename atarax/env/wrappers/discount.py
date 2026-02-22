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

from atarax.core.state import AtariState
from atarax.env.wrappers.base import Wrapper

if TYPE_CHECKING:
    from atarax.env.atari_env import AtariEnv


class EpisodeDiscount(Wrapper):
    """
    Convert the boolean `done` signal to a float32 episode discount.

    The 4th return value of `step()` changes from `bool` to `float32`:
    `1.0` while the episode is running, `0.0` on termination.  All other
    return values pass through unchanged.

    Useful when a downstream training loop expects a continuation mask,
    rather than a raw terminal flag.

    Parameters
    ----------
    env : AtariEnv | Wrapper
        Inner environment to wrap.
    """

    def __init__(self, env: "AtariEnv | Wrapper") -> None:
        super().__init__(env)

    def reset(self, key: chex.Array) -> Tuple[chex.Array, AtariState]:
        return self._env.reset(key)

    def step(
        self,
        state,
        action: chex.Array,
    ) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment by one step and return a float32 discount.

        Parameters
        ----------
        state : AtariState
            Current environment state.
        action : chex.Array
            int32 — ALE action index.

        Returns
        -------
        obs : chex.Array
            Observation from the inner step.
        new_state : AtariState
            Updated wrapper state.
        reward : chex.Array
            float32 — Reward from the inner step (unchanged).
        discount : chex.Array
            float32 — `1.0` if the episode continues, `0.0` if it terminated.
        info : Dict[str, Any]
            Info dict from the inner step.
        """
        obs, new_state, reward, done, info = self._env.step(state, action)
        discount = jnp.where(done, jnp.float32(0.0), jnp.float32(1.0))

        return obs, new_state, reward, discount, info
