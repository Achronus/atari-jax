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

from typing import TYPE_CHECKING, Any, Dict, Tuple, cast

import chex
import jax
import jax.numpy as jnp

from atarax.core.state import AtariState
from atarax.env._kernels import (
    _jit_sample,
    jit_vec_reset,
    jit_vec_reset_single,
    jit_vec_rollout,
    jit_vec_rollout_single,
    jit_vec_step,
    jit_vec_step_single,
)
from atarax.env.env import Env
from atarax.env.spaces import Box, Discrete

if TYPE_CHECKING:
    from atarax.env.atari_env import AtariEnv


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
    env : Env
        Single-instance environment to vectorize.
    n_envs : int
        Number of parallel environments.  Used to split the PRNG key in
        `reset` so each environment starts from a distinct random state.
    """

    def __init__(
        self,
        env: Env,
        n_envs: int,
    ) -> None:
        self._env = env
        self._n_envs = n_envs
        self._is_wrapped = env.unwrapped is not env

        base: "AtariEnv" = cast("AtariEnv", env.unwrapped)
        self._rom = base._rom
        self._game_id_jax = base._game_id_jax
        self._game_id_int = base._game_id_int
        self._compile_mode = base._compile_mode
        self._group_kernels = base._group_kernels
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

        if self._is_wrapped:
            return jax.vmap(self._env.reset)(keys)

        noop_max = jnp.int32(self._params.noop_max)

        if self._compile_mode == "single":
            return jit_vec_reset_single(
                keys, self._rom, self._game_id_int, self._warmup_frames, noop_max
            )

        if self._compile_mode == "group":
            return self._group_kernels.vec_reset(
                keys, self._rom, self._game_id_jax, self._warmup_frames, noop_max
            )

        return jit_vec_reset(
            keys, self._rom, self._game_id_jax, self._warmup_frames, noop_max
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
        info : Dict[str, chex.Array]
            Batched info dict; each value has a leading `n_envs` dimension.
        """
        if self._is_wrapped:
            return jax.vmap(self._env.step)(states, actions)

        fs = self._params.frame_skip
        me = self._params.max_episode_steps

        if self._compile_mode == "single":
            return jit_vec_step_single(
                states, self._rom, actions, self._game_id_int, fs, me
            )

        if self._compile_mode == "group":
            return self._group_kernels.vec_step(
                states, self._rom, actions, self._game_id_jax, fs, me
            )

        return jit_vec_step(states, self._rom, actions, self._game_id_jax, fs, me)

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
        transitions : Tuple[chex.Array, ...]
            `(obs, reward, done, info)` each with shape `[n_envs, T, ...]`.
        """
        if self._is_wrapped:
            def _single_rollout(state, acts):
                def _step(carry, action):
                    obs, new_state, reward, done, info = self._env.step(carry, action)
                    return new_state, (obs, reward, done, info)

                return jax.lax.scan(_step, state, acts)

            return jax.vmap(_single_rollout)(states, actions)

        fs = self._params.frame_skip
        me = self._params.max_episode_steps

        if self._compile_mode == "single":
            return jit_vec_rollout_single(
                states, self._rom, actions, self._game_id_int, fs, me
            )

        if self._compile_mode == "group":
            return self._group_kernels.vec_rollout(
                states, self._rom, actions, self._game_id_jax, fs, me
            )

        return jit_vec_rollout(states, self._rom, actions, self._game_id_jax, fs, me)

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
