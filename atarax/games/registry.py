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
from atarax.games.alien import Alien
from atarax.games.amidar import Amidar
from atarax.games.assault import Assault
from atarax.games.asterix import Asterix
from atarax.games.asteroids import Asteroids
from atarax.games.atlantis import Atlantis
from atarax.games.bank_heist import BankHeist
from atarax.games.battle_zone import BattleZone
from atarax.games.beam_rider import BeamRider
from atarax.games.berzerk import Berzerk
from atarax.games.bowling import Bowling
from atarax.games.boxing import Boxing
from atarax.games.breakout import Breakout
from atarax.games.centipede import Centipede
from atarax.games.chopper_command import ChopperCommand
from atarax.games.crazy_climber import CrazyClimber
from atarax.games.defender import Defender
from atarax.games.demon_attack import DemonAttack
from atarax.games.double_dunk import DoubleDunk
from atarax.games.enduro import Enduro
from atarax.games.fishing_derby import FishingDerby
from atarax.games.freeway import Freeway
from atarax.games.frostbite import Frostbite
from atarax.games.gopher import Gopher
from atarax.games.gravitar import Gravitar
from atarax.games.hero import Hero
from atarax.games.ice_hockey import IceHockey
from atarax.games.jamesbond import JamesBond
from atarax.games.kangaroo import Kangaroo
from atarax.games.krull import Krull
from atarax.games.kung_fu_master import KungFuMaster
from atarax.games.montezuma_revenge import MontezumaRevenge
from atarax.games.ms_pacman import MsPacman
from atarax.games.name_this_game import NameThisGame
from atarax.games.phoenix import Phoenix
from atarax.games.pitfall import Pitfall
from atarax.games.pong import Pong
from atarax.games.pooyan import Pooyan
from atarax.games.private_eye import PrivateEye
from atarax.games.qbert import Qbert
from atarax.games.riverraid import RiverRaid
from atarax.games.road_runner import RoadRunner
from atarax.games.robotank import Robotank
from atarax.games.seaquest import Seaquest
from atarax.games.skiing import Skiing
from atarax.games.solaris import Solaris
from atarax.games.space_invaders import SpaceInvaders
from atarax.games.star_gunner import StarGunner
from atarax.games.tennis import Tennis
from atarax.games.time_pilot import TimePilot
from atarax.games.tutankham import Tutankham
from atarax.games.up_n_down import UpNDown
from atarax.games.venture import Venture
from atarax.games.video_pinball import VideoPinball
from atarax.games.wizard_of_wor import WizardOfWor
from atarax.games.yars_revenge import YarsRevenge
from atarax.games.zaxxon import Zaxxon

#: Mapping of lower-case game name → `AtariEnv` class.
#: Grows as games are implemented; order is alphabetical.
GAMES: Dict[str, Type[AtariEnv]] = {
    "alien": Alien,
    "amidar": Amidar,
    "assault": Assault,
    "asterix": Asterix,
    "asteroids": Asteroids,
    "atlantis": Atlantis,
    "bank_heist": BankHeist,
    "battle_zone": BattleZone,
    "beam_rider": BeamRider,
    "berzerk": Berzerk,
    "bowling": Bowling,
    "boxing": Boxing,
    "breakout": Breakout,
    "centipede": Centipede,
    "chopper_command": ChopperCommand,
    "crazy_climber": CrazyClimber,
    "defender": Defender,
    "demon_attack": DemonAttack,
    "double_dunk": DoubleDunk,
    "enduro": Enduro,
    "fishing_derby": FishingDerby,
    "freeway": Freeway,
    "frostbite": Frostbite,
    "gopher": Gopher,
    "gravitar": Gravitar,
    "hero": Hero,
    "ice_hockey": IceHockey,
    "jamesbond": JamesBond,
    "kangaroo": Kangaroo,
    "krull": Krull,
    "kung_fu_master": KungFuMaster,
    "montezuma_revenge": MontezumaRevenge,
    "ms_pacman": MsPacman,
    "name_this_game": NameThisGame,
    "phoenix": Phoenix,
    "pitfall": Pitfall,
    "pong": Pong,
    "pooyan": Pooyan,
    "private_eye": PrivateEye,
    "qbert": Qbert,
    "riverraid": RiverRaid,
    "road_runner": RoadRunner,
    "robotank": Robotank,
    "seaquest": Seaquest,
    "skiing": Skiing,
    "solaris": Solaris,
    "space_invaders": SpaceInvaders,
    "star_gunner": StarGunner,
    "tennis": Tennis,
    "time_pilot": TimePilot,
    "tutankham": Tutankham,
    "up_n_down": UpNDown,
    "venture": Venture,
    "video_pinball": VideoPinball,
    "wizard_of_wor": WizardOfWor,
    "yars_revenge": YarsRevenge,
    "zaxxon": Zaxxon,
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
        raise ValueError(f"Unknown game {name!r}. Available games: {available}")
    return GAMES[key]
