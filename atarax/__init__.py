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

import envrax
from envrax.make import make, make_multi, make_multi_vec, make_vec
from envrax.spaces import Box, Discrete

from atarax.env.registry import GAME_SPECS, GAMES, PARAMS, get_game
from atarax.envs import (
    ATARI_57,
    ATARI_BASE,
    ATARI_EASY,
    ATARI_HARD,
    ATARI_MEDIUM,
    AtariEnvs,
)
from atarax.game import AtaraxGame, AtaraxParams
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

# Auto-register all implemented atarax games into the envrax registry.
# This runs once per Python process (module caching guarantees idempotency).
for _name, _cls in GAMES.items():
    _env_id = f"atari/{_name}-v0"
    if _env_id not in envrax.registered_names():
        envrax.register(_env_id, _cls, PARAMS[_name]())

del _name, _cls, _env_id

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
