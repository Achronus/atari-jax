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
from typing import Any, Dict, Tuple

import chex
import jax

from atarax.env.spaces import Box, Discrete


class Env(ABC):
    """
    Abstract base class for all Atari environments.

    Both `AtariEnv` (the concrete emulator) and `Wrapper` (the decorator
    base) inherit from `Env`, so `isinstance(env, Env)` is `True` for any
    object returned by `make()` or produced by wrapping.

    Subclasses must implement `reset`, `step`, `observation_space`, and
    `action_space`.  `sample` and `rollout` have concrete defaults that
    subclasses may override for performance.
    """

    @abstractmethod
    def reset(self, key: chex.Array) -> Tuple[chex.Array, Any]:
        """
        Reset the environment and return the first observation.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key.

        Returns
        -------
        obs : chex.Array
            First observation.
        state : Any
            Initial environment state.
        """
        raise NotImplementedError

    @abstractmethod
    def step(
        self,
        state: Any,
        action: chex.Array,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Parameters
        ----------
        state : Any
            Current environment state.
        action : chex.Array
            int32 — ALE action index.

        Returns
        -------
        obs : chex.Array
            Observation after the step.
        new_state : Any
            Updated environment state.
        reward : chex.Array
            float32 — Reward for this step.
        done : chex.Array
            bool — `True` when the episode has ended.
        info : Dict[str, Any]
            Auxiliary diagnostic information.
        """
        raise NotImplementedError

    def sample(self, key: chex.Array) -> chex.Array:
        """
        Sample a uniformly-random action from the action space.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key.

        Returns
        -------
        action : chex.Array
            int32 — Random action index.
        """
        raise NotImplementedError

    def rollout(
        self,
        state: Any,
        actions: chex.Array,
    ) -> Tuple[Any, Tuple[chex.Array, chex.Array, chex.Array, Dict[str, Any]]]:
        """
        Run a multi-step rollout via `jax.lax.scan`.

        The default implementation scans `self.step` over `actions`, so
        all wrapper transforms are applied at every step.  `AtariEnv`
        overrides this with a JIT-compiled kernel for better performance.

        Parameters
        ----------
        state : Any
            Initial environment state.
        actions : chex.Array
            int32[T] — Action sequence of length T.

        Returns
        -------
        final_state : Any
            State after all T steps.
        transitions : Tuple[chex.Array, ...]
            `(obs, reward, done, info)` each with a leading T dimension.
        """

        def _step(carry, action):
            obs, new_state, reward, done, info = self.step(carry, action)
            return new_state, (obs, reward, done, info)

        return jax.lax.scan(_step, state, actions)

    @property
    def unwrapped(self) -> "Env":
        """
        Return the innermost (unwrapped) environment.

        For a bare `AtariEnv` this returns `self`.  For a `Wrapper` it
        recursively delegates to the inner environment until the base
        `AtariEnv` is reached.
        """
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    @property
    @abstractmethod
    def observation_space(self) -> Box:
        """Environment's observation space."""
        raise NotImplementedError

    @property
    @abstractmethod
    def action_space(self) -> Discrete:
        """Environment's action space."""
        raise NotImplementedError
