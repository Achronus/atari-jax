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

from typing import Dict, List, NamedTuple

import jax.numpy as jnp

from atarax.games.base import AtariGame
from atarax.games.roms.alien import Alien
from atarax.games.roms.amidar import Amidar
from atarax.games.roms.assault import Assault
from atarax.games.roms.asterix import Asterix
from atarax.games.roms.asteroids import Asteroids
from atarax.games.roms.atlantis import Atlantis
from atarax.games.roms.bank_heist import BankHeist
from atarax.games.roms.battle_zone import BattleZone
from atarax.games.roms.beam_rider import BeamRider
from atarax.games.roms.berzerk import Berzerk
from atarax.games.roms.bowling import Bowling
from atarax.games.roms.boxing import Boxing
from atarax.games.roms.breakout import Breakout
from atarax.games.roms.centipede import Centipede
from atarax.games.roms.chopper_command import ChopperCommand
from atarax.games.roms.crazy_climber import CrazyClimber
from atarax.games.roms.defender import Defender
from atarax.games.roms.demon_attack import DemonAttack
from atarax.games.roms.double_dunk import DoubleDunk
from atarax.games.roms.enduro import Enduro
from atarax.games.roms.fishing_derby import FishingDerby
from atarax.games.roms.freeway import Freeway
from atarax.games.roms.frostbite import Frostbite
from atarax.games.roms.gopher import Gopher
from atarax.games.roms.gravitar import Gravitar
from atarax.games.roms.hero import Hero
from atarax.games.roms.ice_hockey import IceHockey
from atarax.games.roms.jamesbond import JamesBond
from atarax.games.roms.kangaroo import Kangaroo
from atarax.games.roms.krull import Krull
from atarax.games.roms.kung_fu_master import KungFuMaster
from atarax.games.roms.montezuma_revenge import MontezumaRevenge
from atarax.games.roms.ms_pacman import MsPacman
from atarax.games.roms.name_this_game import NameThisGame
from atarax.games.roms.phoenix import Phoenix
from atarax.games.roms.pitfall import Pitfall
from atarax.games.roms.pong import Pong
from atarax.games.roms.pooyan import Pooyan
from atarax.games.roms.private_eye import PrivateEye
from atarax.games.roms.qbert import Qbert
from atarax.games.roms.riverraid import RiverRaid
from atarax.games.roms.road_runner import RoadRunner
from atarax.games.roms.robotank import Robotank
from atarax.games.roms.seaquest import Seaquest
from atarax.games.roms.skiing import Skiing
from atarax.games.roms.solaris import Solaris
from atarax.games.roms.space_invaders import SpaceInvaders
from atarax.games.roms.star_gunner import StarGunner
from atarax.games.roms.tennis import Tennis
from atarax.games.roms.time_pilot import TimePilot
from atarax.games.roms.tutankham import Tutankham
from atarax.games.roms.up_n_down import UpNDown
from atarax.games.roms.venture import Venture
from atarax.games.roms.video_pinball import VideoPinball
from atarax.games.roms.wizard_of_wor import WizardOfWor
from atarax.games.roms.yars_revenge import YarsRevenge
from atarax.games.roms.zaxxon import Zaxxon


class GameSpec(NamedTuple):
    """Metadata and game instance for one supported game."""

    game_id: int
    ale_name: str
    game: AtariGame


# Ordered list of supported games; index == game_id used by jax.lax.switch.
_GAMES: list[GameSpec] = [
    GameSpec(game_id=0, ale_name="alien", game=Alien()),
    GameSpec(game_id=1, ale_name="amidar", game=Amidar()),
    GameSpec(game_id=2, ale_name="assault", game=Assault()),
    GameSpec(game_id=3, ale_name="asterix", game=Asterix()),
    GameSpec(game_id=4, ale_name="asteroids", game=Asteroids()),
    GameSpec(game_id=5, ale_name="atlantis", game=Atlantis()),
    GameSpec(game_id=6, ale_name="bank_heist", game=BankHeist()),
    GameSpec(game_id=7, ale_name="battle_zone", game=BattleZone()),
    GameSpec(game_id=8, ale_name="beam_rider", game=BeamRider()),
    GameSpec(game_id=9, ale_name="berzerk", game=Berzerk()),
    GameSpec(game_id=10, ale_name="bowling", game=Bowling()),
    GameSpec(game_id=11, ale_name="boxing", game=Boxing()),
    GameSpec(game_id=12, ale_name="breakout", game=Breakout()),
    GameSpec(game_id=13, ale_name="centipede", game=Centipede()),
    GameSpec(game_id=14, ale_name="chopper_command", game=ChopperCommand()),
    GameSpec(game_id=15, ale_name="crazy_climber", game=CrazyClimber()),
    GameSpec(game_id=16, ale_name="defender", game=Defender()),
    GameSpec(game_id=17, ale_name="demon_attack", game=DemonAttack()),
    GameSpec(game_id=18, ale_name="double_dunk", game=DoubleDunk()),
    GameSpec(game_id=19, ale_name="enduro", game=Enduro()),
    GameSpec(game_id=20, ale_name="fishing_derby", game=FishingDerby()),
    GameSpec(game_id=21, ale_name="freeway", game=Freeway()),
    GameSpec(game_id=22, ale_name="frostbite", game=Frostbite()),
    GameSpec(game_id=23, ale_name="gopher", game=Gopher()),
    GameSpec(game_id=24, ale_name="gravitar", game=Gravitar()),
    GameSpec(game_id=25, ale_name="hero", game=Hero()),
    GameSpec(game_id=26, ale_name="ice_hockey", game=IceHockey()),
    GameSpec(game_id=27, ale_name="jamesbond", game=JamesBond()),
    GameSpec(game_id=28, ale_name="kangaroo", game=Kangaroo()),
    GameSpec(game_id=29, ale_name="krull", game=Krull()),
    GameSpec(game_id=30, ale_name="kung_fu_master", game=KungFuMaster()),
    GameSpec(game_id=31, ale_name="montezuma_revenge", game=MontezumaRevenge()),
    GameSpec(game_id=32, ale_name="ms_pacman", game=MsPacman()),
    GameSpec(game_id=33, ale_name="name_this_game", game=NameThisGame()),
    GameSpec(game_id=34, ale_name="phoenix", game=Phoenix()),
    GameSpec(game_id=35, ale_name="pitfall", game=Pitfall()),
    GameSpec(game_id=36, ale_name="pong", game=Pong()),
    GameSpec(game_id=37, ale_name="pooyan", game=Pooyan()),
    GameSpec(game_id=38, ale_name="private_eye", game=PrivateEye()),
    GameSpec(game_id=39, ale_name="qbert", game=Qbert()),
    GameSpec(game_id=40, ale_name="riverraid", game=RiverRaid()),
    GameSpec(game_id=41, ale_name="road_runner", game=RoadRunner()),
    GameSpec(game_id=42, ale_name="robotank", game=Robotank()),
    GameSpec(game_id=43, ale_name="seaquest", game=Seaquest()),
    GameSpec(game_id=44, ale_name="skiing", game=Skiing()),
    GameSpec(game_id=45, ale_name="solaris", game=Solaris()),
    GameSpec(game_id=46, ale_name="space_invaders", game=SpaceInvaders()),
    GameSpec(game_id=47, ale_name="star_gunner", game=StarGunner()),
    GameSpec(game_id=48, ale_name="tennis", game=Tennis()),
    GameSpec(game_id=49, ale_name="time_pilot", game=TimePilot()),
    GameSpec(game_id=50, ale_name="tutankham", game=Tutankham()),
    GameSpec(game_id=51, ale_name="up_n_down", game=UpNDown()),
    GameSpec(game_id=52, ale_name="venture", game=Venture()),
    GameSpec(game_id=53, ale_name="video_pinball", game=VideoPinball()),
    GameSpec(game_id=54, ale_name="wizard_of_wor", game=WizardOfWor()),
    GameSpec(game_id=55, ale_name="yars_revenge", game=YarsRevenge()),
    GameSpec(game_id=56, ale_name="zaxxon", game=Zaxxon()),
]

# Maps ALE game name → game_id integer.
GAME_IDS: dict[str, int] = {g.ale_name: g.game_id for g in _GAMES}

# Maps canonical "atari/<name>-v0" identifier → game_id integer.
ENV_IDS: dict[str, int] = {f"atari/{name}-v0": idx for name, idx in GAME_IDS.items()}

# Ordered lists of bound methods, indexed by game_id.
# Used by jax.lax.switch in the dispatch functions.
SCORE_FNS = [g.game.get_score for g in _GAMES]
TERMINAL_FNS = [g.game.is_terminal for g in _GAMES]
LIVES_FNS = [g.game.get_lives for g in _GAMES]

# int32[57] — warmup frame count per game; used as a dynamic argument in jit_reset.
WARMUP_FRAMES_ARRAY = jnp.array(
    [g.game._WARMUP_FRAMES for g in _GAMES], dtype=jnp.int32
)


def _make_reward_score_branch(spec: GameSpec):
    g = spec.game
    if g._uses_score_tracking:

        def _branch(ram_prev, ram_curr, prev_score):
            new_score = g.get_score(ram_curr)
            return (new_score - prev_score).astype(jnp.float32), new_score
    else:

        def _branch(ram_prev, ram_curr, prev_score):
            return g.get_reward(ram_prev, ram_curr), prev_score

    return _branch


REWARD_SCORE_FNS = [_make_reward_score_branch(s) for s in _GAMES]

# Predefined game subsets for use with compile_mode="group".
# Each value is a list of ALE game names; order within the list is not
# significant — games are sorted by absolute game_id at env creation time.
GAME_GROUPS: Dict[str, List[str]] = {
    "atari5": [
        "breakout",
        "ms_pacman",
        "pong",
        "qbert",
        "space_invaders",
    ],
    "atari10": [
        "alien",
        "beam_rider",
        "breakout",
        "enduro",
        "montezuma_revenge",
        "ms_pacman",
        "pitfall",
        "pong",
        "qbert",
        "space_invaders",
    ],
    "atari26": [
        "alien",
        "amidar",
        "assault",
        "asterix",
        "asteroids",
        "atlantis",
        "bank_heist",
        "battle_zone",
        "beam_rider",
        "bowling",
        "boxing",
        "breakout",
        "centipede",
        "chopper_command",
        "crazy_climber",
        "demon_attack",
        "enduro",
        "fishing_derby",
        "freeway",
        "gopher",
        "gravitar",
        "ice_hockey",
        "jamesbond",
        "kangaroo",
        "krull",
        "kung_fu_master",
    ],
}
