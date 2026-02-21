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

from atari_jax.env.wrappers.base import BaseWrapper
from atari_jax.env.wrappers.utils import to_gray
from atari_jax.env.wrappers.clip_reward import ClipRewardWrapper
from atari_jax.env.wrappers.episodic_life import EpisodicLifeState, EpisodicLifeWrapper
from atari_jax.env.wrappers.frame_stack import FrameStackState, FrameStackWrapper
from atari_jax.env.wrappers.grayscale import GrayscaleWrapper
from atari_jax.env.wrappers.resize import ResizeWrapper

__all__ = [
    "BaseWrapper",
    "ClipRewardWrapper",
    "to_gray",
    "EpisodicLifeState",
    "EpisodicLifeWrapper",
    "FrameStackState",
    "FrameStackWrapper",
    "GrayscaleWrapper",
    "ResizeWrapper",
]
