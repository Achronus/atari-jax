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

from typing import TYPE_CHECKING, Any, Callable, Dict, Tuple

import chex
import jax

if TYPE_CHECKING:
    from atari_jax.env.atari_env import AtariEnv
    from atari_jax.env.wrappers.base import BaseWrapper

from atari_jax.core.state import AtariState
from atari_jax.env._compile import _wrap_with_spinner
from atari_jax.env.spaces import Box, Discrete


class VecEnv:
    """
    A vectorised environment that runs `n_envs` copies in parallel via
    `jax.vmap`.

    All mutable state lives in a batched `AtariState` pytree.

    Parameters
    ----------
    env : AtariEnv | BaseWrapper
        Single-instance environment to vectorize.
    n_envs : int
        Number of parallel environments.  Used to split the PRNG key in
        `reset` so each environment starts from a distinct random state.
    jit_compile : bool (optional)
        JIT-compile all vmapped functions on the first call.
        Default is `True`.
    show_compile_progress : bool (optional)
        Display a spinner on the first (compilation) call of each vmapped
        method.  Default is `False`.
    """

    def __init__(
        self,
        env: "AtariEnv | BaseWrapper",
        n_envs: int,
        jit_compile: bool = True,
        show_compile_progress: bool = False,
    ) -> None:
        self._env = env
        self._n_envs = n_envs

        reset_fn = jax.vmap(env.reset)
        step_fn = jax.vmap(env.step)
        rollout_fn = jax.vmap(make_rollout_fn(env))

        if jit_compile:
            reset_fn = jax.jit(reset_fn)
            step_fn = jax.jit(step_fn)
            rollout_fn = jax.jit(rollout_fn)

        if show_compile_progress:
            reset_fn = _wrap_with_spinner(reset_fn, "Compiling vectorized reset...")
            step_fn = _wrap_with_spinner(step_fn, "Compiling vectorized step...")
            rollout_fn = _wrap_with_spinner(
                rollout_fn, "Compiling vectorized rollout..."
            )

        self._reset_fn = reset_fn
        self._step_fn = step_fn
        self._rollout_fn = rollout_fn

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
        return self._reset_fn(keys)

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
        return self._step_fn(states, actions)

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
        states: AtariState,
        actions: chex.Array,
    ) -> Tuple[AtariState, Tuple[chex.Array, chex.Array, chex.Array, Dict[str, Any]]]:
        """
        Run a compiled multi-step rollout across all environments.

        Internally calls `jax.vmap(lax.scan(...))` so the full rollout
        compiles to a single XLA kernel — no Python loop overhead.

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
        transitions : Tuple[chex.Array, chex.Array, chex.Array, Dict[str, Any]]
            `(obs, reward, done, info)` each with shape `[n_envs, T, ...]`.
        """
        return self._rollout_fn(states, actions)

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


def make_rollout_fn(env: "AtariEnv | BaseWrapper") -> Callable:
    """
    Build a compiled rollout function for `env`.

    The returned function runs consecutive steps via `jax.lax.scan`,
    producing stacked transition arrays with no Python-level loop overhead.
    The number of steps is determined by `actions.shape[0]` at call time.
    Compose with `jax.vmap` for batched parallel rollouts:

    .. code-block:: python

        rollout = make_rollout_fn(env)
        batched = jax.vmap(rollout)
        final_states, (obs, reward, done, info) = batched(states, actions)

    Parameters
    ----------
    env : AtariEnv | BaseWrapper
        Environment that exposes `step(state, action)`.

    Returns
    -------
    rollout : Callable
        A function with signature
        `rollout(state, actions) -> (final_state, transitions)`
        where `transitions = (obs, reward, done, info)` and each array
        is stacked over the timestep dimension.
    """

    def rollout(
        state: AtariState,
        actions: chex.Array,
    ) -> Tuple:
        """
        Execute environment steps for each action and collect transitions.

        Parameters
        ----------
        state : AtariState
            Initial environment state.
        actions : chex.Array
            int32[T] — Action sequence; determines the number of steps.

        Returns
        -------
        final_state : AtariState
            Environment state after all steps.
        transitions : tuple
            `(obs, reward, done, info)` each with a leading timestep
            dimension.
        """

        def _step(state, action):
            obs, new_state, reward, done, info = env.step(state, action)
            return new_state, (obs, reward, done, info)

        final_state, transitions = jax.lax.scan(_step, state, actions)
        return final_state, transitions

    return rollout
