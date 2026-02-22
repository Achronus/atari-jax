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

"""Observation and reward wrappers for AtariEnv."""

from atarax.env.wrappers.atari_preprocessing import AtariPreprocessing
from atarax.env.wrappers.base import Wrapper
from atarax.env.wrappers.clip_reward import ClipReward
from atarax.env.wrappers.episodic_life import EpisodicLife, EpisodicLifeState
from atarax.env.wrappers.frame_stack import FrameStackObservation, FrameStackState
from atarax.env.wrappers.grayscale import GrayscaleObservation
from atarax.env.wrappers.normalize_obs import NormalizeObservation
from atarax.env.wrappers.record_episode_statistics import (
    EpisodeStatisticsState,
    RecordEpisodeStatistics,
)
from atarax.env.wrappers.resize import ResizeObservation
from atarax.env.wrappers.utils import to_gray

__all__ = [
    "AtariPreprocessing",
    "ClipReward",
    "EpisodicLife",
    "EpisodicLifeState",
    "EpisodeStatisticsState",
    "FrameStackObservation",
    "FrameStackState",
    "GrayscaleObservation",
    "NormalizeObservation",
    "RecordEpisodeStatistics",
    "ResizeObservation",
    "to_gray",
    "Wrapper",
]
