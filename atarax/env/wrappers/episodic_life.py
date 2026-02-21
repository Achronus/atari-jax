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
import jax.numpy as jnp

if TYPE_CHECKING:
    from atarax.env.atari_env import AtariEnv

from atarax.core.state import AtariState
from atarax.env.wrappers.base import BaseWrapper


@chex.dataclass
class EpisodicLifeState:
    """
    State for `EpisodicLifeWrapper`.

    Parameters
    ----------
    env_state : AtariState
        Underlying machine state (may itself be a wrapped state).
    prev_lives : chex.Array
        int32 — Lives count at the end of the previous step (or at `reset`);
        compared against `info["lives"]` each step to detect a life loss.
    real_done : chex.Array
        bool — `True` when the game itself (not just a life) is over.
    """

    env_state: AtariState
    prev_lives: chex.Array
    real_done: chex.Array


class EpisodicLifeWrapper(BaseWrapper):
    """
    Signal terminal on every life loss, not only on true game over.

    Standard training terminates episodes on life loss to give the agent
    a clear signal that a mistake was made. A true game reset only happens
    when `real_done` is True (accessible via `info["real_done"]`).

    Parameters
    ----------
    env : AtariEnv | BaseWrapper
        Inner environment. Its `step` must return `info["lives"]` (int32),
        and its `reset` state chain must terminate at an `AtariState`.
    """

    def __init__(self, env: "AtariEnv | BaseWrapper") -> None:
        super().__init__(env)

    def reset(self, key: chex.Array) -> Tuple[chex.Array, EpisodicLifeState]:
        """
        Reset the environment and initialise the lives counter.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key

        Returns
        -------
        obs : chex.Array
            First observation from the inner reset.
        state : EpisodicLifeState
            Wrapper state with `real_done=False` and `prev_lives` initialized
            from the base `AtariState` at the bottom of the state chain.
        """
        obs, env_state = self._env.reset(key)
        base = env_state

        while hasattr(base, "env_state"):
            base = base.env_state  # type: ignore

        return obs, EpisodicLifeState(
            env_state=env_state,
            prev_lives=base.lives,
            real_done=jnp.bool_(False),
        )

    def step(
        self,
        state: EpisodicLifeState,
        action: chex.Array,
    ) -> Tuple[chex.Array, EpisodicLifeState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment by one step and set terminal on any life loss.

        Parameters
        ----------
        state : EpisodicLifeState
            Current wrapper state.
        action : chex.Array
            int32 — ALE action index.

        Returns
        -------
        obs : chex.Array
            Observation from the inner step.
        new_state : EpisodicLifeState
            Updated wrapper state.
        reward : chex.Array
            float32 — Reward from the inner step.
        done : chex.Array
            bool — True on life loss *or* true game over.
        info : Dict[str, Any]
            Inner info dict extended with `"real_done"` (bool).
        """
        obs, env_state, reward, real_done, info = self._env.step(
            state.env_state, action
        )
        new_lives = info["lives"]

        done = (new_lives < state.prev_lives) | real_done
        new_state = EpisodicLifeState(
            env_state=env_state,
            prev_lives=new_lives,
            real_done=real_done,
        )

        return obs, new_state, reward, done, dict(info, real_done=real_done)
