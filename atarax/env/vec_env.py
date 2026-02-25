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

"""Vectorised environment: runs `n_envs` copies in parallel via `jax.vmap`."""

import pathlib
from typing import Any, Dict, Tuple

import chex
import jax

from atarax.env._base import Env
from atarax.env._compile import DEFAULT_CACHE_DIR, setup_cache
from atarax.env.spaces import Box, Discrete


class VecEnv:
    """
    A vectorised environment that runs `n_envs` copies in parallel via
    `jax.vmap`.

    All mutable state lives in a batched pytree with a leading `n_envs`
    dimension.  Calling `reset` splits the PRNG key into `n_envs`
    sub-keys so that each environment starts from a distinct random state.

    Parameters
    ----------
    env : Env
        Single-instance environment to vectorise.
    n_envs : int
        Number of parallel environments.
    jit_compile : bool (optional)
        If `True`, replaces `reset`, `step`, and `rollout` with
        `jax.jit`-compiled versions on construction, giving the correct
        `jit(vmap(f))` ordering for maximum GPU utilisation.
        Default is `False`.
    cache_dir : Path | str | None (optional)
        Directory for the persistent XLA compilation cache. Defaults to
        `~/.cache/atarax/xla_cache`. Pass `None` to disable.
    """

    def __init__(
        self,
        env: Env,
        n_envs: int,
        jit_compile: bool = False,
        cache_dir: pathlib.Path | str | None = DEFAULT_CACHE_DIR,
    ) -> None:
        self._env = env
        self._n_envs = n_envs

        if jit_compile:
            setup_cache(cache_dir)
            self.reset = jax.jit(self.reset)
            self.step = jax.jit(self.step)
            self.rollout = jax.jit(self.rollout)

    def reset(self, key: chex.Array) -> Tuple[chex.Array, Any]:
        """
        Reset all `n_envs` environments with independent random starts.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key.

        Returns
        -------
        obs : chex.Array
            uint8[n_envs, ...] — Stacked first observations.
        states : Any
            Batched environment states (leading dim = `n_envs`).
        """
        keys = jax.random.split(key, self._n_envs)
        return jax.vmap(self._env.reset)(keys)

    def step(
        self,
        states: Any,
        actions: chex.Array,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, chex.Array]]:
        """
        Advance all environments by one step simultaneously.

        Parameters
        ----------
        states : Any
            Batched environment states (leading dim = `n_envs`).
        actions : chex.Array
            int32[n_envs] — One action per environment.

        Returns
        -------
        obs : chex.Array
            uint8[n_envs, ...] — Observations after the step.
        new_states : Any
            Updated batched states.
        reward : chex.Array
            float32[n_envs] — Per-environment rewards.
        done : chex.Array
            bool[n_envs] — Per-environment terminal flags.
        info : Dict[str, chex.Array]
            Batched info dict; each value has a leading `n_envs` dimension.
        """
        return jax.vmap(self._env.step)(states, actions)

    def sample(self, key: chex.Array) -> chex.Array:
        """
        Sample a random action for each environment.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key.

        Returns
        -------
        actions : chex.Array
            int32[n_envs] — One random action per environment.
        """
        keys = jax.random.split(key, self._n_envs)
        return jax.vmap(self._env.sample)(keys)

    def rollout(
        self,
        states: Any,
        actions: chex.Array,
    ) -> Tuple[Any, Tuple[chex.Array, chex.Array, chex.Array, Dict[str, Any]]]:
        """
        Run a compiled multi-step rollout across all environments.

        Uses `jax.lax.scan` over T timesteps for each environment, then
        `jax.vmap` across `n_envs`.

        Parameters
        ----------
        states : Any
            Batched initial states (leading dim = `n_envs`).
        actions : chex.Array
            int32[n_envs, T] — Action sequence for each environment.

        Returns
        -------
        final_states : Any
            Batched states after all steps.
        transitions : Tuple[chex.Array, ...]
            `(obs, reward, done, info)` each with shape `[n_envs, T, ...]`.
        """

        def _single_rollout(state: Any, acts: chex.Array):
            def _step(carry, action):
                obs, new_state, reward, done, info = self._env.step(carry, action)
                return new_state, (obs, reward, done, info)

            return jax.lax.scan(_step, state, acts)

        return jax.vmap(_single_rollout)(states, actions)

    @property
    def observation_space(self) -> Box:
        """Observation space of the inner environment."""
        return self._env.observation_space

    @property
    def action_space(self) -> Discrete:
        """Action space of the inner environment."""
        return self._env.action_space

    @property
    def n_envs(self) -> int:
        """Number of parallel environments."""
        return self._n_envs

    def __repr__(self) -> str:
        return f"VecEnv<{self._env!r}, n_envs={self._n_envs}>"
