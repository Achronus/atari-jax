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

from typing import Dict, List, Type

from atarax.envs import ATARI_57
from atarax.game import AtaraxGame
from atarax.games.breakout import Breakout
from atarax.games.pong import Pong
from atarax.games.space_invaders import SpaceInvaders
from atarax.spec import EnvSpec

GAMES: Dict[str, Type[AtaraxGame]] = {
    "breakout": Breakout,
    "pong": Pong,
    "space_invaders": SpaceInvaders,
}

GAME_SPECS: List[EnvSpec] = [EnvSpec.parse(name) for name in ATARI_57.all_names()]


def get_game(env_id: str) -> Type[AtaraxGame]:
    """
    Return the registered game class for *env_id*.

    Parameters
    ----------
    env_id : str
        Environment ID in `"atari/{name}-v0"` format.

    Returns
    -------
    game_cls : Type[AtaraxGame]

    Raises
    ------
    env_error : ValueError
        If *env_id* is not yet implemented.
    """
    spec = EnvSpec.parse(env_id)
    if spec.env_name not in GAMES:
        available = sorted(GAMES)
        raise ValueError(
            f"Unknown game {spec.env_name!r}: not yet implemented. "
            f"Available: {available}"
        )
    return GAMES[spec.env_name]
