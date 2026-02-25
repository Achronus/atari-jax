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

"""Game registry mapping lower-case game names to `AtariEnv` classes."""

from typing import Dict, List, Type

from atarax.env.atari_env import AtariEnv
from atarax.env.spec import EnvSpec
from atarax.games.breakout import Breakout

#: Mapping of lower-case game name → `AtariEnv` class.
#: Grows as games are implemented; order is alphabetical.
GAMES: Dict[str, Type[AtariEnv]] = {
    "breakout": Breakout,
}

#: Pre-built `EnvSpec` for every registered game (alphabetical order).
GAME_SPECS: List[EnvSpec] = [EnvSpec("atari", name) for name in sorted(GAMES)]

#: Predefined game subsets drawn from the standard Mnih et al. (2015) 57-game
#: suite.  Values are lists of `EnvSpec`; games not yet implemented will raise
#: `ValueError` at `make()` time, not at import time.
GAME_GROUPS: Dict[str, List[EnvSpec]] = {
    "atari5": [
        EnvSpec("atari", "breakout"),
        EnvSpec("atari", "ms_pacman"),
        EnvSpec("atari", "pong"),
        EnvSpec("atari", "qbert"),
        EnvSpec("atari", "space_invaders"),
    ],
    "atari10": [
        EnvSpec("atari", "alien"),
        EnvSpec("atari", "beam_rider"),
        EnvSpec("atari", "breakout"),
        EnvSpec("atari", "enduro"),
        EnvSpec("atari", "montezuma_revenge"),
        EnvSpec("atari", "ms_pacman"),
        EnvSpec("atari", "pitfall"),
        EnvSpec("atari", "pong"),
        EnvSpec("atari", "qbert"),
        EnvSpec("atari", "space_invaders"),
    ],
    "atari26": [
        EnvSpec("atari", "alien"),
        EnvSpec("atari", "amidar"),
        EnvSpec("atari", "assault"),
        EnvSpec("atari", "asterix"),
        EnvSpec("atari", "asteroids"),
        EnvSpec("atari", "atlantis"),
        EnvSpec("atari", "bank_heist"),
        EnvSpec("atari", "battle_zone"),
        EnvSpec("atari", "beam_rider"),
        EnvSpec("atari", "bowling"),
        EnvSpec("atari", "boxing"),
        EnvSpec("atari", "breakout"),
        EnvSpec("atari", "centipede"),
        EnvSpec("atari", "chopper_command"),
        EnvSpec("atari", "crazy_climber"),
        EnvSpec("atari", "demon_attack"),
        EnvSpec("atari", "enduro"),
        EnvSpec("atari", "fishing_derby"),
        EnvSpec("atari", "freeway"),
        EnvSpec("atari", "gopher"),
        EnvSpec("atari", "gravitar"),
        EnvSpec("atari", "ice_hockey"),
        EnvSpec("atari", "jamesbond"),
        EnvSpec("atari", "kangaroo"),
        EnvSpec("atari", "krull"),
        EnvSpec("atari", "kung_fu_master"),
    ],
    "atari57": [
        EnvSpec("atari", "alien"),
        EnvSpec("atari", "amidar"),
        EnvSpec("atari", "assault"),
        EnvSpec("atari", "asterix"),
        EnvSpec("atari", "asteroids"),
        EnvSpec("atari", "atlantis"),
        EnvSpec("atari", "bank_heist"),
        EnvSpec("atari", "battle_zone"),
        EnvSpec("atari", "beam_rider"),
        EnvSpec("atari", "berzerk"),
        EnvSpec("atari", "bowling"),
        EnvSpec("atari", "boxing"),
        EnvSpec("atari", "breakout"),
        EnvSpec("atari", "centipede"),
        EnvSpec("atari", "chopper_command"),
        EnvSpec("atari", "crazy_climber"),
        EnvSpec("atari", "defender"),
        EnvSpec("atari", "demon_attack"),
        EnvSpec("atari", "double_dunk"),
        EnvSpec("atari", "enduro"),
        EnvSpec("atari", "fishing_derby"),
        EnvSpec("atari", "freeway"),
        EnvSpec("atari", "frostbite"),
        EnvSpec("atari", "gopher"),
        EnvSpec("atari", "gravitar"),
        EnvSpec("atari", "hero"),
        EnvSpec("atari", "ice_hockey"),
        EnvSpec("atari", "jamesbond"),
        EnvSpec("atari", "kangaroo"),
        EnvSpec("atari", "krull"),
        EnvSpec("atari", "kung_fu_master"),
        EnvSpec("atari", "montezuma_revenge"),
        EnvSpec("atari", "ms_pacman"),
        EnvSpec("atari", "name_this_game"),
        EnvSpec("atari", "phoenix"),
        EnvSpec("atari", "pitfall"),
        EnvSpec("atari", "pong"),
        EnvSpec("atari", "pooyan"),
        EnvSpec("atari", "private_eye"),
        EnvSpec("atari", "qbert"),
        EnvSpec("atari", "riverraid"),
        EnvSpec("atari", "road_runner"),
        EnvSpec("atari", "robotank"),
        EnvSpec("atari", "seaquest"),
        EnvSpec("atari", "skiing"),
        EnvSpec("atari", "solaris"),
        EnvSpec("atari", "space_invaders"),
        EnvSpec("atari", "star_gunner"),
        EnvSpec("atari", "tennis"),
        EnvSpec("atari", "time_pilot"),
        EnvSpec("atari", "tutankham"),
        EnvSpec("atari", "up_n_down"),
        EnvSpec("atari", "venture"),
        EnvSpec("atari", "video_pinball"),
        EnvSpec("atari", "wizard_of_wor"),
        EnvSpec("atari", "yars_revenge"),
        EnvSpec("atari", "zaxxon"),
    ],
}


def get_game(name: str) -> Type[AtariEnv]:
    """
    Look up a game class by name.

    Parameters
    ----------
    name : str
        Game name — case-insensitive (e.g. `"Breakout"` or `"breakout"`).

    Returns
    -------
    game_cls : Type[AtariEnv]
        The `AtariEnv` subclass for the requested game.

    Raises
    ------
    ValueError
        If `name` is not in the registry.
    """
    key = name.lower()
    if key not in GAMES:
        available = sorted(GAMES)
        raise ValueError(
            f"Unknown game {name!r}. Available games: {available}"
        )
    return GAMES[key]
