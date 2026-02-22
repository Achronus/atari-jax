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
    from atarax.env.wrappers.base import Wrapper

from atarax.core.state import AtariState
from atarax.env._kernels import _jit_sample, jit_vec_reset, jit_vec_rollout, jit_vec_step
from atarax.env.spaces import Box, Discrete


class VecEnv:
    """
    A vectorised environment that runs `n_envs` copies in parallel via
    `jax.vmap`.

    All mutable state lives in a batched `AtariState` pytree.  Internally,
    all hot-path calls are forwarded to the module-level shared JIT kernels
    in `_kernels.py`, so every game with the same ROM size compiles only once
    regardless of how many `VecEnv` instances are created.

    Parameters
    ----------
    env : AtariEnv | Wrapper
        Single-instance environment to vectorize.
    n_envs : int
        Number of parallel environments.  Used to split the PRNG key in
        `reset` so each environment starts from a distinct random state.
    """

    def __init__(
        self,
        env: "AtariEnv | Wrapper",
        n_envs: int,
    ) -> None:
        self._env = env
        self._n_envs = n_envs

        # Traverse wrapper chain to reach the base AtariEnv.
        base = env
        while hasattr(base, "_env"):
            base = base._env
        self._rom = base._rom
        self._game_id_jax = base._game_id_jax
        self._warmup_frames = base._warmup_frames
        self._params = base._params

    def reset(self, key: chex.Array) -> Tuple[chex.Array, AtariState]:
        """
        Reset all `n_envs` environments with independent random starts.

        The key is split into `n_envs` sub-keys so each environment
        draws a different number of no-op steps.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key.

        Returns
        -------
        obs : chex.Array
            uint8[n_envs, ...] — Stacked first observations.
        states : AtariState
            Batched environment states (leading dim = `n_envs`).
        """
        keys = jax.random.split(key, self._n_envs)
        return jit_vec_reset(
            keys,
            self._rom,
            self._game_id_jax,
            self._warmup_frames,
            jnp.int32(self._params.noop_max),
        )

    def step(
        self,
        states: AtariState,
        actions: chex.Array,
    ) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, chex.Array]]:
        """
        Advance all environments by one step simultaneously.

        Parameters
        ----------
        states : AtariState
            Batched environment states (leading dim = `n_envs`).
        actions : chex.Array
            int32[n_envs] — One action per environment.

        Returns
        -------
        obs : chex.Array
            uint8[n_envs, ...] — Observations after the step.
        new_states : AtariState
            Updated batched states.
        reward : chex.Array
            float32[n_envs] — Per-environment rewards.
        done : chex.Array
            bool[n_envs] — Per-environment terminal flags.
        info : dict
            Batched info dict; each value has a leading `n_envs` dimension.
        """
        return jit_vec_step(
            states,
            self._rom,
            actions,
            self._game_id_jax,
            self._params.frame_skip,
            self._params.max_episode_steps,
        )

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
        return jax.vmap(_jit_sample)(keys)

    def rollout(
        self,
        states: AtariState,
        actions: chex.Array,
    ) -> Tuple[AtariState, Tuple[chex.Array, chex.Array, chex.Array, Dict[str, Any]]]:
        """
        Run a compiled multi-step rollout across all environments.

        Internally calls `jit_vec_rollout` which uses `lax.scan` over T
        timesteps, each advancing all `n_envs` environments simultaneously —
        no Python loop overhead.

        Parameters
        ----------
        states : AtariState
            Batched initial states (leading dim = `n_envs`).
        actions : chex.Array
            int32[n_envs, T] — Action sequence for each environment.

        Returns
        -------
        final_states : AtariState
            Batched states after all steps.
        transitions : tuple
            `(obs, reward, done, info)` each with shape `[T, n_envs, ...]`.
        """
        return jit_vec_rollout(
            states,
            self._rom,
            actions,
            self._game_id_jax,
            self._params.frame_skip,
            self._params.max_episode_steps,
        )

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
