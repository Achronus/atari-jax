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
    from atarax.env.atari_env import AtariEnv

from atarax.core.state import AtariState
from atarax.env.spaces import Box
from atarax.env.wrappers.base import Wrapper
from atarax.env.wrappers.utils import to_gray


class GrayscaleObservation(Wrapper):
    """
    Convert RGB observations to grayscale using the NTSC luminance formula.

    Wraps any environment whose `reset` / `step` return `uint8[H, W, 3]`
    observations and converts them to `uint8[H, W]`.

    Parameters
    ----------
    env : AtariEnv | Wrapper
        Inner environment to wrap.
    """

    def __init__(self, env: "AtariEnv | Wrapper") -> None:
        super().__init__(env)

    def reset(self, key: chex.Array) -> Tuple[chex.Array, AtariState]:
        """
        Reset the inner environment and convert the observation to grayscale.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key.

        Returns
        -------
        obs : chex.Array
            uint8[H, W] — Grayscale observation.
        state : AtariState
            Inner machine state.
        """
        obs, state = self._env.reset(key)
        return to_gray(obs), state

    def step(
        self,
        state,
        action: chex.Array,
    ) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step the inner environment and convert the observation to grayscale.

        Parameters
        ----------
        state : AtariState
            Current machine state.
        action : chex.Array
            int32 — ALE action index.

        Returns
        -------
        obs : chex.Array
            uint8[H, W] — Grayscale observation.
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
        return to_gray(obs), new_state, reward, done, info

    @property
    def observation_space(self) -> Box:
        inner = self._env.observation_space
        h, w = inner.shape[0], inner.shape[1]

        return Box(low=inner.low, high=inner.high, shape=(h, w), dtype=inner.dtype)
