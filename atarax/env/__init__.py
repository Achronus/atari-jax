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

from atarax.env._compile import DEFAULT_CACHE_DIR, setup_cache
from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.env.make import make, make_vec, precompile_all
from atarax.env.spaces import Box, Discrete
from atarax.env.spec import EnvSpec
from atarax.env.vec_env import VecEnv, make_rollout_fn
from atarax.env.wrappers import (
    AtariPreprocessing,
    ClipReward,
    EpisodeDiscount,
    EpisodicLife,
    EpisodicLifeState,
    EpisodeStatisticsState,
    FrameStackObservation,
    FrameStackState,
    GrayscaleObservation,
    NormalizeObservation,
    RecordEpisodeStatistics,
    ResizeObservation,
    Wrapper,
)

__all__ = [
    "AtariEnv",
    "AtariPreprocessing",
    "Box",
    "ClipReward",
    "DEFAULT_CACHE_DIR",
    "Discrete",
    "EnvParams",
    "EnvSpec",
    "EpisodeDiscount",
    "EpisodeStatisticsState",
    "EpisodicLife",
    "EpisodicLifeState",
    "FrameStackObservation",
    "FrameStackState",
    "GrayscaleObservation",
    "make",
    "make_rollout_fn",
    "make_vec",
    "NormalizeObservation",
    "precompile_all",
    "RecordEpisodeStatistics",
    "ResizeObservation",
    "setup_cache",
    "VecEnv",
    "Wrapper",
]
