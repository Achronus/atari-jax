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

"""atarax.games — JAX-native game implementations."""

from atarax.games._base import AtariState, GameState
from atarax.games.alien import Alien, AlienState
from atarax.games.amidar import Amidar, AmidarState
from atarax.games.assault import Assault, AssaultState
from atarax.games.asterix import Asterix, AsterixState
from atarax.games.asteroids import Asteroids, AsteroidsState
from atarax.games.atlantis import Atlantis, AtlantisState
from atarax.games.bank_heist import BankHeist, BankHeistState
from atarax.games.battle_zone import BattleZone, BattleZoneState
from atarax.games.beam_rider import BeamRider, BeamRiderState
from atarax.games.berzerk import Berzerk, BerzerkState
from atarax.games.bowling import Bowling, BowlingState
from atarax.games.boxing import Boxing, BoxingState
from atarax.games.breakout import Breakout, BreakoutState
from atarax.games.centipede import Centipede, CentipedeState
from atarax.games.chopper_command import ChopperCommand, ChopperCommandState
from atarax.games.crazy_climber import CrazyClimber, CrazyClimberState
from atarax.games.defender import Defender, DefenderState
from atarax.games.demon_attack import DemonAttack, DemonAttackState
from atarax.games.double_dunk import DoubleDunk, DoubleDunkState
from atarax.games.enduro import Enduro, EnduroState
from atarax.games.fishing_derby import FishingDerby, FishingDerbyState
from atarax.games.freeway import Freeway, FreewayState
from atarax.games.frostbite import Frostbite, FrostbiteState
from atarax.games.gopher import Gopher, GopherState
from atarax.games.gravitar import Gravitar, GravitarState
from atarax.games.hero import Hero, HeroState
from atarax.games.ice_hockey import IceHockey, IceHockeyState
from atarax.games.jamesbond import JamesBond, JamesBondState
from atarax.games.kangaroo import Kangaroo, KangarooState
from atarax.games.krull import Krull, KrullState
from atarax.games.kung_fu_master import KungFuMaster, KungFuMasterState
from atarax.games.montezuma_revenge import MontezumaRevenge, MontezumaRevengeState
from atarax.games.ms_pacman import MsPacman, MsPacmanState
from atarax.games.name_this_game import NameThisGame, NameThisGameState
from atarax.games.phoenix import Phoenix, PhoenixState
from atarax.games.pitfall import Pitfall, PitfallState
from atarax.games.pong import Pong, PongState
from atarax.games.pooyan import Pooyan, PooyanState
from atarax.games.private_eye import PrivateEye, PrivateEyeState
from atarax.games.qbert import Qbert, QbertState
from atarax.games.registry import GAME_GROUPS, GAME_SPECS, GAMES, get_game
from atarax.games.riverraid import RiverRaid, RiverRaidState
from atarax.games.road_runner import RoadRunner, RoadRunnerState
from atarax.games.robotank import Robotank, RobotankState
from atarax.games.seaquest import Seaquest, SeaquestState
from atarax.games.skiing import Skiing, SkiingState
from atarax.games.solaris import Solaris, SolarisState
from atarax.games.space_invaders import SpaceInvaders, SpaceInvadersState
from atarax.games.star_gunner import StarGunner, StarGunnerState
from atarax.games.tennis import Tennis, TennisState
from atarax.games.time_pilot import TimePilot, TimePilotState
from atarax.games.tutankham import Tutankham, TutankhamState
from atarax.games.up_n_down import UpNDown, UpNDownState
from atarax.games.venture import Venture, VentureState
from atarax.games.video_pinball import VideoPinball, VideoPinballState
from atarax.games.wizard_of_wor import WizardOfWor, WizardOfWorState
from atarax.games.yars_revenge import YarsRevenge, YarsRevengeState
from atarax.games.zaxxon import Zaxxon, ZaxxonState

__all__ = [
    "AtariState",
    "GameState",
    "Alien",
    "AlienState",
    "Amidar",
    "AmidarState",
    "Assault",
    "AssaultState",
    "Asterix",
    "AsterixState",
    "Asteroids",
    "AsteroidsState",
    "Atlantis",
    "AtlantisState",
    "BankHeist",
    "BankHeistState",
    "BattleZone",
    "BattleZoneState",
    "BeamRider",
    "BeamRiderState",
    "Berzerk",
    "BerzerkState",
    "Bowling",
    "BowlingState",
    "Boxing",
    "BoxingState",
    "Breakout",
    "BreakoutState",
    "Centipede",
    "CentipedeState",
    "ChopperCommand",
    "ChopperCommandState",
    "CrazyClimber",
    "CrazyClimberState",
    "Defender",
    "DefenderState",
    "DemonAttack",
    "DemonAttackState",
    "DoubleDunk",
    "DoubleDunkState",
    "Enduro",
    "EnduroState",
    "FishingDerby",
    "FishingDerbyState",
    "Freeway",
    "FreewayState",
    "Frostbite",
    "FrostbiteState",
    "GAME_GROUPS",
    "GAME_SPECS",
    "GAMES",
    "Gopher",
    "GopherState",
    "Gravitar",
    "GravitarState",
    "Hero",
    "HeroState",
    "IceHockey",
    "IceHockeyState",
    "JamesBond",
    "JamesBondState",
    "Kangaroo",
    "KangarooState",
    "Krull",
    "KrullState",
    "KungFuMaster",
    "KungFuMasterState",
    "MontezumaRevenge",
    "MontezumaRevengeState",
    "MsPacman",
    "MsPacmanState",
    "NameThisGame",
    "NameThisGameState",
    "Phoenix",
    "PhoenixState",
    "Pitfall",
    "PitfallState",
    "Pong",
    "PongState",
    "Pooyan",
    "PooyanState",
    "PrivateEye",
    "PrivateEyeState",
    "Qbert",
    "QbertState",
    "RiverRaid",
    "RiverRaidState",
    "RoadRunner",
    "RoadRunnerState",
    "Robotank",
    "RobotankState",
    "Seaquest",
    "SeaquestState",
    "Skiing",
    "SkiingState",
    "Solaris",
    "SolarisState",
    "SpaceInvaders",
    "SpaceInvadersState",
    "StarGunner",
    "StarGunnerState",
    "Tennis",
    "TennisState",
    "TimePilot",
    "TimePilotState",
    "Tutankham",
    "TutankhamState",
    "UpNDown",
    "UpNDownState",
    "Venture",
    "VentureState",
    "VideoPinball",
    "VideoPinballState",
    "WizardOfWor",
    "WizardOfWorState",
    "YarsRevenge",
    "YarsRevengeState",
    "Zaxxon",
    "ZaxxonState",
    "get_game",
]
