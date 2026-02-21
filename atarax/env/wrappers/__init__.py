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

from atarax.env.wrappers.base import BaseWrapper
from atarax.env.wrappers.utils import to_gray
from atarax.env.wrappers.clip_reward import ClipRewardWrapper
from atarax.env.wrappers.episodic_life import EpisodicLifeState, EpisodicLifeWrapper
from atarax.env.wrappers.frame_stack import FrameStackState, FrameStackWrapper
from atarax.env.wrappers.grayscale import GrayscaleWrapper
from atarax.env.wrappers.resize import ResizeWrapper

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
