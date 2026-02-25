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

"""ExpandDims wrapper — adds a trailing size-1 dimension to reward and done."""

from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp

from atarax.env._base import Env
from atarax.env.wrappers.base import Wrapper


class ExpandDims(Wrapper):
    """
    Add a trailing size-1 dimension to `reward` and `done`.

    Transforms scalar outputs from `step()` so that `reward` and `done`
    have shape `(..., 1)` instead of `(...)`.  This is useful when a
    training loop expects all trajectory tensors to share a consistent
    trailing feature dimension (e.g. `float32[T, 1]` rather than
    `float32[T]`).

    Because `rollout()` is inherited from `Env` and scans over `self.step`,
    the expansion is applied at every step automatically:

    - Single step: `float32 → float32[1]`, `bool → bool[1]`
    - After rollout: `float32[T] → float32[T, 1]`, `bool[T] → bool[T, 1]`

    Observations and the `info` dict are passed through unchanged.

    Note: actions are inputs to `step()`, not outputs, so they are not
    affected by this wrapper.  Unsqueeze action tensors in the training
    loop if needed (e.g. `actions[..., None]`).

    Parameters
    ----------
    env : Env
        Inner environment to wrap.
    """

    def __init__(self, env: Env) -> None:
        super().__init__(env)

    def reset(self, key: chex.Array) -> Tuple[chex.Array, Any]:
        """
        Reset the environment and return the initial observation and state.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key.

        Returns
        -------
        obs : chex.Array
            First observation (unchanged).
        state : Any
            Initial environment state.
        """
        return self._env.reset(key)

    def step(
        self,
        state: Any,
        action: chex.Array,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment and expand `reward` and `done`.

        Parameters
        ----------
        state : Any
            Current environment state.
        action : chex.Array
            int32 — ALE action index.

        Returns
        -------
        obs : chex.Array
            Observation after the step (unchanged).
        new_state : Any
            Updated environment state.
        reward : chex.Array
            float32[..., 1] — Reward with a trailing size-1 dimension.
        done : chex.Array
            bool[..., 1] — Terminal flag with a trailing size-1 dimension.
        info : Dict[str, Any]
            Auxiliary info dict (unchanged).
        """
        obs, new_state, reward, done, info = self._env.step(state, action)
        return (
            obs,
            new_state,
            jnp.expand_dims(reward, -1),
            jnp.expand_dims(done, -1),
            info,
        )
