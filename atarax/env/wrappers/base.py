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

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Tuple

import chex

if TYPE_CHECKING:
    from atarax.env.atari_env import AtariEnv

from atarax.core.state import AtariState
from atarax.env.spaces import Box, Discrete


class Wrapper(ABC):
    """
    Abstract base class for AtariEnv wrappers.

    Subclasses must implement `reset` and `step`. The `sample`,
    `observation_space`, and `action_space` members delegate to the
    inner environment by default and may be overridden when the wrapper
    changes the observation shape or action set.

    Parameters
    ----------
    env : AtariEnv | Wrapper
        Inner environment to wrap
    """

    def __init__(self, env: "AtariEnv | Wrapper") -> None:
        self._env = env

    @abstractmethod
    def reset(self, key: chex.Array) -> Tuple[chex.Array, AtariState]:
        """
        Reset the environment and return the initial observation and state.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key

        Returns
        -------
        obs : chex.Array
            uint8[210, 160, 3] — First RGB observation.
        state : AtariState
            Initial machine state after reset and no-ops.
        """
        raise NotImplementedError()

    @abstractmethod
    def step(
        self, state, action: chex.Array
    ) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Parameters
        ----------
        state : AtariState
            Current environment state.
        action : chex.Array
            int32 — ALE action index.

        Returns
        -------
        obs : chex.Array
            `uint8[210, 160, 3]` — Observation after the step
        new_state : AtariState
            Updated machine state
        reward : chex.Array
            float32 — Total reward accumulated over skipped frames
        done : chex.Array
            bool — `True` when the episode has ended
        info : Dict[str, Any]
            `{"lives": int32, "episode_frame": int32, "truncated": bool}`
        """
        raise NotImplementedError()

    def sample(self, key: chex.Array) -> chex.Array:
        """
        Sample a uniformly-random action from the action space.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key
        """
        return self._env.sample(key)

    @property
    def observation_space(self) -> Box:
        """Environment's observation space."""
        return self._env.observation_space

    @property
    def action_space(self) -> Discrete:
        """Environment's action space."""
        return self._env.action_space
