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
from atari_jax.games.roms.alien import Alien
from atari_jax.games.roms.amidar import Amidar
from atari_jax.games.roms.assault import Assault
from atari_jax.games.roms.asterix import Asterix
from atari_jax.games.roms.asteroids import Asteroids
from atari_jax.games.roms.atlantis import Atlantis
from atari_jax.games.roms.bank_heist import BankHeist
from atari_jax.games.roms.battle_zone import BattleZone
from atari_jax.games.roms.beam_rider import BeamRider
from atari_jax.games.roms.berzerk import Berzerk
from atari_jax.games.roms.bowling import Bowling
from atari_jax.games.roms.boxing import Boxing
from atari_jax.games.roms.breakout import Breakout
from atari_jax.games.roms.centipede import Centipede
from atari_jax.games.roms.chopper_command import ChopperCommand
from atari_jax.games.roms.crazy_climber import CrazyClimber
from atari_jax.games.roms.defender import Defender
from atari_jax.games.roms.demon_attack import DemonAttack
from atari_jax.games.roms.double_dunk import DoubleDunk
from atari_jax.games.roms.enduro import Enduro
from atari_jax.games.roms.fishing_derby import FishingDerby
from atari_jax.games.roms.freeway import Freeway
from atari_jax.games.roms.frostbite import Frostbite
from atari_jax.games.roms.gopher import Gopher
from atari_jax.games.roms.gravitar import Gravitar
from atari_jax.games.roms.hero import Hero
from atari_jax.games.roms.ice_hockey import IceHockey
from atari_jax.games.roms.jamesbond import JamesBond
from atari_jax.games.roms.kangaroo import Kangaroo
from atari_jax.games.roms.krull import Krull
from atari_jax.games.roms.kung_fu_master import KungFuMaster
from atari_jax.games.roms.montezuma_revenge import MontezumaRevenge
from atari_jax.games.roms.ms_pacman import MsPacman
from atari_jax.games.roms.name_this_game import NameThisGame
from atari_jax.games.roms.phoenix import Phoenix
from atari_jax.games.roms.pitfall import Pitfall
from atari_jax.games.roms.pong import Pong
from atari_jax.games.roms.pooyan import Pooyan
from atari_jax.games.roms.private_eye import PrivateEye
from atari_jax.games.roms.qbert import Qbert
from atari_jax.games.roms.riverraid import RiverRaid
from atari_jax.games.roms.road_runner import RoadRunner
from atari_jax.games.roms.robotank import Robotank
from atari_jax.games.roms.seaquest import Seaquest
from atari_jax.games.roms.skiing import Skiing
from atari_jax.games.roms.solaris import Solaris
from atari_jax.games.roms.space_invaders import SpaceInvaders
from atari_jax.games.roms.star_gunner import StarGunner
from atari_jax.games.roms.tennis import Tennis
from atari_jax.games.roms.time_pilot import TimePilot
from atari_jax.games.roms.tutankham import Tutankham
from atari_jax.games.roms.up_n_down import UpNDown
from atari_jax.games.roms.venture import Venture
from atari_jax.games.roms.video_pinball import VideoPinball
from atari_jax.games.roms.wizard_of_wor import WizardOfWor
from atari_jax.games.roms.yars_revenge import YarsRevenge
from atari_jax.games.roms.zaxxon import Zaxxon


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

# Ordered lists of bound methods, indexed by game_id.
# Used by jax.lax.switch in the dispatch functions.
REWARD_FNS = [g.game.get_reward for g in _GAMES]
TERMINAL_FNS = [g.game.is_terminal for g in _GAMES]
