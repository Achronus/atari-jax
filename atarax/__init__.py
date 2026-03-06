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

from envrax.spaces import Box, Discrete

from atarax.envs import (
    ATARI_57,
    ATARI_BASE,
    ATARI_EASY,
    ATARI_HARD,
    ATARI_MEDIUM,
    AtariEnvs,
)
from atarax.game import AtaraxGame, AtaraxParams
from atarax.env.registry import GAME_SPECS, GAMES, PARAMS, get_game
from atarax.make import make, make_multi, make_multi_vec, make_vec
from atarax.spec import EnvSpec
from atarax.state import AtariState, GameState
from atarax.wrappers import (
    AtariPreprocessing,
    EpisodicLife,
    EpisodicLifeState,
    JitWrapper,
    VmapEnv,
    Wrapper,
)

__all__ = [
    "ATARI_57",
    "ATARI_BASE",
    "ATARI_EASY",
    "ATARI_HARD",
    "ATARI_MEDIUM",
    "AtaraxGame",
    "AtaraxParams",
    "AtariEnvs",
    "AtariPreprocessing",
    "AtariState",
    "Box",
    "Discrete",
    "EnvSpec",
    "EpisodicLife",
    "EpisodicLifeState",
    "GAME_SPECS",
    "GAMES",
    "GameState",
    "PARAMS",
    "JitWrapper",
    "VmapEnv",
    "Wrapper",
    "get_game",
    "make",
    "make_multi",
    "make_multi_vec",
    "make_vec",
]
