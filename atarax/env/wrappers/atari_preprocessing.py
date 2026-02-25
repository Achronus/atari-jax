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

if TYPE_CHECKING:
    from atarax.env.wrappers.record_episode_statistics import EpisodeStatisticsState

from atarax.env._base import Env
from atarax.env.wrappers.base import Wrapper
from atarax.env.wrappers.clip_reward import ClipReward
from atarax.env.wrappers.episodic_life import EpisodicLife
from atarax.env.wrappers.frame_stack import FrameStackObservation
from atarax.env.wrappers.grayscale import GrayscaleObservation
from atarax.env.wrappers.record_episode_statistics import RecordEpisodeStatistics
from atarax.env.wrappers.resize import ResizeObservation


class AtariPreprocessing(Wrapper):
    """
    Standard DQN preprocessing stack from Mnih et al. (2015).

    Applies, in order:

    - `GrayscaleObservation` — NTSC luminance, `uint8[H, W]`
    - `ResizeObservation(h, w)` — bilinear resize, `uint8[84, 84]`
    - `FrameStackObservation(n_stack)` — rolling buffer, `uint8[84, 84, 4]`
    - `ClipReward` — reward ∈ {-1, 0, +1}
    - `EpisodicLife` — terminal on every life loss
    - `RecordEpisodeStatistics` — episode return + length in `info["episode"]`

    Equivalent to passing `preset=True` to `make()`, but usable as a
    standalone wrapper class.

    Parameters
    ----------
    env : Env
        Base environment to wrap.
    h : int (optional)
        Output frame height after resize. Default is `84`.
    w : int (optional)
        Output frame width after resize. Default is `84`.
    n_stack : int (optional)
        Number of frames to stack. Default is `4`.

    Examples
    --------
    >>> env = AtariPreprocessing(make("breakout"))
    >>> obs, state = env.reset(key)   # obs: uint8[84, 84, 4]
    """

    def __init__(
        self,
        env: Env,
        *,
        h: int = 84,
        w: int = 84,
        n_stack: int = 4,
    ) -> None:
        env = GrayscaleObservation(env)
        env = ResizeObservation(env, h=h, w=w)
        env = FrameStackObservation(env, n_stack=n_stack)
        env = ClipReward(env)
        env = EpisodicLife(env)
        env = RecordEpisodeStatistics(env)
        super().__init__(env)

    def reset(self, key: chex.Array) -> Tuple[chex.Array, "EpisodeStatisticsState"]:
        """
        Reset the environment with full preprocessing applied.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key.

        Returns
        -------
        obs : chex.Array
            Preprocessed observation, `uint8[h, w, n_stack]`.
        state : EpisodeStatisticsState
            Nested wrapper state.
        """
        return self._env.reset(key)  # type: ignore

    def step(
        self, state: Any, action: chex.Array
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Step the environment with full preprocessing applied.

        Parameters
        ----------
        state : EpisodeStatisticsState
            Current nested wrapper state.
        action : chex.Array
            Action to take.

        Returns
        -------
        obs : chex.Array
            Preprocessed observation, `uint8[h, w, n_stack]`.
        new_state : EpisodeStatisticsState
            Updated nested wrapper state.
        reward : chex.Array
            Clipped reward ∈ {-1, 0, +1}.
        done : chex.Array
            True on life loss or game over.
        info : dict
            Includes `"real_done"`, `"lives"`, `"episode_step"`, and
            `"episode": {"r": float32, "l": int32}`.
        """
        return self._env.step(state, action)
