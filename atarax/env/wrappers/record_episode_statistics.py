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

from atarax.env.wrappers.base import Wrapper


@chex.dataclass
class EpisodeStatisticsState:
    """
    State for `RecordEpisodeStatistics`.

    Parameters
    ----------
    env_state : Any
        Inner environment state.
    episode_return : chex.Array
        Cumulative reward for the current episode. float32 scalar.
    episode_length : chex.Array
        Number of steps taken in the current episode. int32 scalar.
    """

    env_state: Any
    episode_return: chex.Array
    episode_length: chex.Array


class RecordEpisodeStatistics(Wrapper):
    """
    Records episode return and length.

    Accumulates reward and step count in `EpisodeStatisticsState`.
    Episode statistics are written to `info["episode"]` on every `step()`
    call; values are non-zero only when `done=True`.

    Parameters
    ----------
    env : AtariEnv | Wrapper
        Environment to wrap.

    Examples
    --------
    >>> env = RecordEpisodeStatistics(make("atari/breakout-v0"))
    >>> obs, state = env.reset(key)
    >>> obs, state, reward, done, info = env.step(state, action)
    >>> info["episode"]   # {"r": 0.0, "l": 0} while running; non-zero at done
    """

    def __init__(self, env: "AtariEnv | Wrapper") -> None:
        super().__init__(env)

    def reset(self, key: chex.Array) -> Tuple[chex.Array, EpisodeStatisticsState]:
        """
        Reset the environment and episode accumulators.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key.

        Returns
        -------
        obs : chex.Array
            Initial observation.
        state : EpisodeStatisticsState
            Initial state with zeroed accumulators.
        """
        obs, env_state = self._env.reset(key)
        state = EpisodeStatisticsState(
            env_state=env_state,
            episode_return=jnp.float32(0.0),
            episode_length=jnp.int32(0),
        )
        return obs, state

    def step(
        self,
        state: EpisodeStatisticsState,
        action: chex.Array,
    ) -> Tuple[
        chex.Array, EpisodeStatisticsState, chex.Array, chex.Array, Dict[str, Any]
    ]:
        """
        Step the environment and update episode accumulators.

        Parameters
        ----------
        state : EpisodeStatisticsState
            Current state.
        action : chex.Array
            Action to take.

        Returns
        -------
        obs : chex.Array
            Next observation.
        new_state : EpisodeStatisticsState
            Updated state with accumulated return and length.
        reward : chex.Array
            Step reward.
        done : chex.Array
            Episode terminal flag.
        info : Dict[str, Any]
            Environment metadata extended with `"episode"`:
            `{"r": float32, "l": int32}` — non-zero only when `done=True`.
        """
        obs, env_state, reward, done, info = self._env.step(state.env_state, action)

        episode_return = state.episode_return + reward.astype(jnp.float32)
        episode_length = state.episode_length + jnp.int32(1)

        info["episode"] = {
            "r": jnp.where(done, episode_return, jnp.float32(0.0)),
            "l": jnp.where(done, episode_length, jnp.int32(0)),
        }

        new_state = EpisodeStatisticsState(
            env_state=env_state,
            episode_return=jnp.where(done, jnp.float32(0.0), episode_return),
            episode_length=jnp.where(done, jnp.int32(0), episode_length),
        )
        return obs, new_state, reward, done, info
