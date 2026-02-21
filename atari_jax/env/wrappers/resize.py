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

if TYPE_CHECKING:
    from atari_jax.env.atari_env import AtariEnv

from atari_jax.core.state import AtariState
from atari_jax.env.spaces import Box
from atari_jax.env.wrappers.base import BaseWrapper
from atari_jax.env.wrappers.utils import resize


class ResizeWrapper(BaseWrapper):
    """
    Resize 2-D grayscale observations to `(out_h, out_w)` using bilinear
    interpolation.

    Expects the inner environment to produce `uint8[H, W]` observations
    (i.e. wrap with `GrayscaleWrapper` first).

    Parameters
    ----------
    env : AtariEnv | BaseWrapper
        Inner environment returning 2-D grayscale observations.
    out_h : int
        Output height in pixels. Defaults to 84.
    out_w : int
        Output width in pixels. Defaults to 84.
    """

    def __init__(
        self,
        env: "AtariEnv | BaseWrapper",
        out_h: int = 84,
        out_w: int = 84,
    ) -> None:
        super().__init__(env)

        self._out_h = out_h
        self._out_w = out_w

    def reset(self, key: chex.Array) -> Tuple[chex.Array, AtariState]:
        """
        Reset the inner environment and resize the observation.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key

        Returns
        -------
        obs : chex.Array
            uint8[out_h, out_w] — Resized grayscale observation.
        state : AtariState
            Inner machine state.
        """
        obs, state = self._env.reset(key)
        return resize(obs, self._out_h, self._out_w), state

    def step(
        self,
        state,
        action: chex.Array,
    ) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step the inner environment and resize the observation.

        Parameters
        ----------
        state : AtariState
            Current machine state.
        action : chex.Array
            int32 — ALE action index.

        Returns
        -------
        obs : chex.Array
            uint8[out_h, out_w] — Resized grayscale observation.
        new_state : AtariState
            Updated machine state.
        reward : chex.Array
            float32 — Reward from the inner step.
        done : chex.Array
            bool — Terminal flag from the inner step.
        info : dict
            Info dict from the inner step.
        """
        obs, new_state, reward, done, info = self._env.step(state, action)
        return resize(obs, self._out_h, self._out_w), new_state, reward, done, info

    @property
    def observation_space(self) -> Box:
        inner = self._env.observation_space

        return Box(
            low=inner.low,
            high=inner.high,
            shape=(self._out_h, self._out_w),
            dtype=inner.dtype,
        )
