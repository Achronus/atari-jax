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

"""Game registry — IDs and dispatch tables for reward and terminal functions."""

from typing import NamedTuple

from atari_jax.games.base import AtariGame
from atari_jax.games.roms.breakout import Breakout


class GameSpec(NamedTuple):
    """Metadata and game instance for one supported game."""

    game_id: int
    ale_name: str
    game: AtariGame


# Ordered list of supported games; index == game_id used by jax.lax.switch.
_GAMES: list[GameSpec] = [
    GameSpec(game_id=0, ale_name="breakout", game=Breakout()),
]

# Maps ALE game name → game_id integer.
GAME_IDS: dict[str, int] = {g.ale_name: g.game_id for g in _GAMES}

# Ordered lists of bound methods, indexed by game_id.
# Used by jax.lax.switch in the dispatch functions.
REWARD_FNS = [g.game.get_reward for g in _GAMES]
TERMINAL_FNS = [g.game.is_terminal for g in _GAMES]
