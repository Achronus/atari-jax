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

from atari_jax.env._compile import DEFAULT_CACHE_DIR, setup_cache
from atari_jax.env.atari_env import AtariEnv, EnvParams
from atari_jax.env.make import make, make_vec, precompile_all
from atari_jax.env.spaces import Box, Discrete
from atari_jax.env.spec import EnvSpec
from atari_jax.env.vec_env import VecEnv, make_rollout_fn
from atari_jax.env.wrappers import (
    BaseWrapper,
    ClipRewardWrapper,
    EpisodicLifeState,
    EpisodicLifeWrapper,
    FrameStackState,
    FrameStackWrapper,
    GrayscaleWrapper,
    ResizeWrapper,
)

__all__ = [
    "AtariEnv",
    "BaseWrapper",
    "Box",
    "ClipRewardWrapper",
    "DEFAULT_CACHE_DIR",
    "Discrete",
    "EnvParams",
    "EnvSpec",
    "EpisodicLifeState",
    "EpisodicLifeWrapper",
    "FrameStackState",
    "FrameStackWrapper",
    "GrayscaleWrapper",
    "make",
    "make_rollout_fn",
    "make_vec",
    "precompile_all",
    "setup_cache",
    "VecEnv",
    "ResizeWrapper",
]
