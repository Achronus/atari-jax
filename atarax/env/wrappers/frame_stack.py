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
import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from atarax.env.atari_env import AtariEnv

from atarax.core.state import AtariState
from atarax.env.spaces import Box
from atarax.env.wrappers.base import Wrapper


@chex.dataclass
class FrameStackState:
    """
    State for `FrameStackObservation`.

    Parameters
    ----------
    env_state : AtariState
        Underlying machine state (may itself be a wrapped state).
    obs_stack : jax.Array
        uint8[H, W, n_stack] — Ring buffer of the last `n_stack` processed
        observations, oldest frame at channel index 0.
    """

    env_state: AtariState
    obs_stack: jax.Array


class FrameStackObservation(Wrapper):
    """
    Maintain a sliding window of the last `n_stack` observations.

    Expects the inner environment to produce `uint8[H, W]` observations.
    The stacked observation has shape `uint8[H, W, n_stack]`.

    State is carried in `FrameStackState` (a `chex.dataclass` pytree), which
    is passed to / returned from `reset` and `step` in place of `AtariState`.

    Parameters
    ----------
    env : AtariEnv | Wrapper
        Inner environment returning 2-D observations.
    n_stack : int (optional)
        Number of frames to stack. Default is `4`.
    """

    def __init__(self, env: "AtariEnv | Wrapper", *, n_stack: int = 4) -> None:
        super().__init__(env)

        self._n_stack = n_stack

    def reset(self, key: chex.Array) -> Tuple[chex.Array, FrameStackState]:
        """
        Reset the inner environment and initialise the frame stack.

        The stack is filled by repeating the first observation `n_stack` times.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key

        Returns
        -------
        obs : chex.Array
            uint8[H, W, n_stack] — Initial stacked observation.
        state : FrameStackState
            Wrapper state containing the inner state and the stack.
        """
        obs, env_state = self._env.reset(key)
        stack = jnp.stack([obs] * self._n_stack, axis=-1)
        wrapped = FrameStackState(env_state=env_state, obs_stack=stack)
        return stack, wrapped

    def step(
        self,
        state: FrameStackState,
        action: chex.Array,
    ) -> Tuple[chex.Array, FrameStackState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step the inner environment and roll the frame stack.

        The oldest frame is discarded and the new observation is appended at
        the last channel index.

        Parameters
        ----------
        state : FrameStackState
            Current wrapper state.
        action : chex.Array
            int32 — ALE action index.

        Returns
        -------
        obs : chex.Array
            uint8[H, W, n_stack] — Updated stacked observation.
        new_state : FrameStackState
            Updated wrapper state.
        reward : chex.Array
            float32 — Reward from the inner step.
        done : chex.Array
            bool — Terminal flag from the inner step.
        info : Dict[str, Any]
            Info dict from the inner step.
        """
        obs, env_state, reward, done, info = self._env.step(state.env_state, action)

        new_stack = jnp.concatenate([state.obs_stack[..., 1:], obs[..., None]], axis=-1)  # type: ignore
        new_state = FrameStackState(env_state=env_state, obs_stack=new_stack)

        return new_stack, new_state, reward, done, info

    @property
    def observation_space(self) -> Box:
        inner = self._env.observation_space
        h, w = inner.shape[0], inner.shape[1]

        return Box(
            low=inner.low,
            high=inner.high,
            shape=(h, w, self._n_stack),
            dtype=inner.dtype,
        )
