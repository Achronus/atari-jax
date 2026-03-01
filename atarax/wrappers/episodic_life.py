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

from typing import Any, Dict, Tuple

import chex
import jax.numpy as jnp
from envrax.base import EnvParams, JaxEnv
from envrax.wrappers.base import Wrapper


@chex.dataclass
class EpisodicLifeState:
    """
    State for `EpisodicLife`.

    Parameters
    ----------
    env_state : Any
        Underlying environment state.
    prev_lives : chex.Array
        int32 — Lives count at the end of the previous step.
    real_done : chex.Array
        bool — `True` when the game itself is over.
    """

    env_state: Any
    prev_lives: chex.Array
    real_done: chex.Array


class EpisodicLife(Wrapper):
    """
    Signal terminal on every life loss, not only on true game over.

    Parameters
    ----------
    env : JaxEnv
        Inner environment. Its `step` must return `info["lives"]` (int32).
    """

    def __init__(self, env: JaxEnv) -> None:
        super().__init__(env)

    def reset(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EpisodicLifeState]:
        obs, env_state = self._env.reset(rng, params)
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
        rng: chex.PRNGKey,
        state: EpisodicLifeState,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, EpisodicLifeState, chex.Array, chex.Array, Dict[str, Any]]:
        obs, env_state, reward, real_done, info = self._env.step(
            rng, state.env_state, action, params
        )
        new_lives = info["lives"]

        done = (new_lives < state.prev_lives) | real_done
        new_state = EpisodicLifeState(
            env_state=env_state,
            prev_lives=new_lives,
            real_done=real_done,
        )

        return obs, new_state, reward, done, dict(info, real_done=real_done)
