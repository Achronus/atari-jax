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

from typing import Callable, Tuple

import chex
import jax


def make_rollout_fn(env) -> Callable:
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
    env : AtariEnv or compatible wrapper
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
        state,
        actions: chex.Array,
    ) -> Tuple:
        """
        Execute environment steps for each action and collect transitions.

        Parameters
        ----------
        state : AtariState or compatible
            Initial environment state.
        actions : chex.Array
            int32[T] — Action sequence; determines the number of steps.

        Returns
        -------
        final_state : AtariState or compatible
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
