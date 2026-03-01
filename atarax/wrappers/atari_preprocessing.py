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
from envrax.base import EnvParams, JaxEnv
from envrax.wrappers.base import Wrapper
from envrax.wrappers.clip_reward import ClipReward
from envrax.wrappers.frame_stack import FrameStackObservation
from envrax.wrappers.grayscale import GrayscaleObservation
from envrax.wrappers.record_episode_statistics import RecordEpisodeStatistics
from envrax.wrappers.resize import ResizeObservation

from atarax.wrappers.episodic_life import EpisodicLife

if TYPE_CHECKING:
    from envrax.wrappers.record_episode_statistics import EpisodeStatisticsState


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

    Parameters
    ----------
    env : JaxEnv
        Base environment to wrap.
    h : int (optional)
        Output frame height after resize. Default is `84`.
    w : int (optional)
        Output frame width after resize. Default is `84`.
    n_stack : int (optional)
        Number of frames to stack. Default is `4`.
    """

    def __init__(
        self,
        env: JaxEnv,
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

    def reset(
        self, rng: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, "EpisodeStatisticsState"]:
        return self._env.reset(rng, params)  # type: ignore

    def step(
        self,
        rng: chex.PRNGKey,
        state: Any,
        action: chex.Array,
        params: EnvParams,
    ) -> Tuple[chex.Array, Any, chex.Array, chex.Array, Dict[str, Any]]:
        return self._env.step(rng, state, action, params)
