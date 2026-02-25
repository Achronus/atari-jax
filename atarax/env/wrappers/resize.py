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

from atarax.env._base import Env
from atarax.env.spaces import Box
from atarax.env.wrappers.base import Wrapper
from atarax.env.wrappers.utils import resize


class ResizeObservation(Wrapper):
    """
    Resize 2-D grayscale observations to `(h, w)` using bilinear
    interpolation.

    Expects the inner environment to produce `uint8[H, W]` observations
    (i.e. wrap with `GrayscaleObservation` first).

    Parameters
    ----------
    env : Env | Wrapper
        Inner environment returning 2-D grayscale observations.
    h : int (optional)
        Output height in pixels. Default is `84`.
    w : int (optional)
        Output width in pixels. Default is `84`.
    """

    def __init__(self, env: Env, *, h: int = 84, w: int = 84) -> None:
        super().__init__(env)

        self._h = h
        self._w = w

    def reset(self, key: chex.Array) -> Tuple[chex.Array, Any]:
        """
        Reset the inner environment and resize the observation.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key

        Returns
        -------
        obs : chex.Array
            uint8[h, w] — Resized grayscale observation.
        state : Any
            Inner environment state.
        """
        obs, state = self._env.reset(key)
        return resize(obs, self._h, self._w), state

    def step(
        self,
        state,
        action: chex.Array,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step the inner environment and resize the observation.

        Parameters
        ----------
        state : Any
            Current environment state.
        action : chex.Array
            int32 — ALE action index.

        Returns
        -------
        obs : chex.Array
            uint8[h, w] — Resized grayscale observation.
        new_state : Any
            Updated environment state.
        reward : chex.Array
            float32 — Reward from the inner step.
        done : chex.Array
            bool — Terminal flag from the inner step.
        info : dict
            Info dict from the inner step.
        """
        obs, new_state, reward, done, info = self._env.step(state, action)
        return resize(obs, self._h, self._w), new_state, reward, done, info

    @property
    def observation_space(self) -> Box:
        inner = self._env.observation_space

        return Box(
            low=inner.low,
            high=inner.high,
            shape=(self._h, self._w),
            dtype=inner.dtype,
        )
