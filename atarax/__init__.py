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

"""atarax — JAX-native Atari game environments (Brax-style)."""

from atarax.env._base import Env
from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.env.make import make, make_multi, make_multi_vec, make_vec
from atarax.env.spaces import Box, Discrete
from atarax.env.spec import EnvSpec
from atarax.env.vec_env import VecEnv
from atarax.env.wrappers.jit_wrapper import JitWrapper
from atarax.games._base import AtariState, GameState
from atarax.games.breakout import Breakout, BreakoutState
from atarax.games.registry import GAME_GROUPS, GAME_SPECS, GAMES, get_game

__all__ = [
    "AtariEnv",
    "AtariState",
    "Box",
    "Breakout",
    "BreakoutState",
    "Discrete",
    "Env",
    "EnvParams",
    "EnvSpec",
    "GAME_GROUPS",
    "GAME_SPECS",
    "GAMES",
    "GameState",
    "JitWrapper",
    "VecEnv",
    "get_game",
    "make",
    "make_multi",
    "make_multi_vec",
    "make_vec",
]
