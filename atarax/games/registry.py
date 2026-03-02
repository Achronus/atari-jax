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
from atarax.games.assault import Assault
from atarax.games.atlantis import Atlantis
from atarax.games.boxing import Boxing
from atarax.games.breakout import Breakout
from atarax.games.demon_attack import DemonAttack
from atarax.games.fishing_derby import FishingDerby
from atarax.games.freeway import Freeway
from atarax.games.gopher import Gopher
from atarax.games.gravitar import Gravitar
from atarax.games.phoenix import Phoenix
from atarax.games.pitfall import Pitfall
from atarax.games.pong import Pong
from atarax.games.space_invaders import SpaceInvaders
from atarax.games.tennis import Tennis
from atarax.games.video_pinball import VideoPinball
from atarax.spec import EnvSpec

GAMES: Dict[str, Type[AtaraxGame]] = {
    "assault": Assault,
    "atlantis": Atlantis,
    "boxing": Boxing,
    "breakout": Breakout,
    "demon_attack": DemonAttack,
    "fishing_derby": FishingDerby,
    "freeway": Freeway,
    "gopher": Gopher,
    "gravitar": Gravitar,
    "phoenix": Phoenix,
    "pitfall": Pitfall,
    "pong": Pong,
    "space_invaders": SpaceInvaders,
    "tennis": Tennis,
    "video_pinball": VideoPinball,
}

GAME_SPECS: List[EnvSpec] = [EnvSpec.parse(name) for name in ATARI_57.all_names()]


def get_game(env_id: str | EnvSpec) -> Type[AtaraxGame]:
    """
    Return the registered game class for *env_id*.

    Parameters
    ----------
    env_id : str | EnvSpec
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
