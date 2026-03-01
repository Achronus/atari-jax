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

import re
from dataclasses import dataclass, field
from typing import List

from envrax.envs import EnvGroup


def _to_snake(name: str) -> str:
    """Convert CamelCase to snake_case (e.g. `"SpaceInvaders"` → `"space_invaders"`)."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


@dataclass
class AtariEnvs(EnvGroup):
    """
    [Atari Learning Environment](https://ale.farama.org/) (ALE).

    All 57 classic Atari 2600 games from the Mnih et al. (2015) benchmark.

    Environment IDs use the `"atari/{snake_case}-v0"` format, e.g.
    `"atari/space_invaders-v0"`.
    """

    prefix: str = "atari"
    category: str = "Atari"
    version: str = "v0"
    required_packages: List[str] = field(
        default_factory=lambda: ["gymnasium", "ale_py"]
    )
    envs: List[str] = field(
        default_factory=lambda: [
            "Alien",
            "Amidar",
            "Assault",
            "Asterix",
            "Asteroids",
            "Atlantis",
            "BankHeist",
            "BattleZone",
            "BeamRider",
            "Berzerk",
            "Bowling",
            "Boxing",
            "Breakout",
            "Centipede",
            "ChopperCommand",
            "CrazyClimber",
            "Defender",
            "DemonAttack",
            "DoubleDunk",
            "Enduro",
            "FishingDerby",
            "Freeway",
            "Frostbite",
            "Gopher",
            "Gravitar",
            "Hero",
            "IceHockey",
            "Jamesbond",
            "Kangaroo",
            "Krull",
            "KungFuMaster",
            "MontezumaRevenge",
            "MsPacman",
            "NameThisGame",
            "Phoenix",
            "Pitfall",
            "Pong",
            "PrivateEye",
            "Qbert",
            "Riverraid",
            "RoadRunner",
            "Robotank",
            "Seaquest",
            "Skiing",
            "Solaris",
            "SpaceInvaders",
            "StarGunner",
            "Surround",
            "Tennis",
            "TimePilot",
            "Tutankham",
            "UpNDown",
            "Venture",
            "VideoPinball",
            "WizardOfWor",
            "YarsRevenge",
            "Zaxxon",
        ]
    )

    def get_name(self, env: str, version: str | None = None) -> str:
        """Return `"atari/{snake_case}-{version}"`, e.g. `"atari/space_invaders-v0"`."""
        ver = version if version is not None else self.version
        return f"{self.prefix}/{_to_snake(env)}-{ver}"


# Pre-instantiated groups
ATARI_BASE = AtariEnvs(
    envs=[
        "Assault",
        "Atlantis",
        "Boxing",
        "Breakout",
        "CrazyClimber",
        "DemonAttack",
        "Gopher",
        "Kangaroo",
        "Krull",
        "NameThisGame",
        "RoadRunner",
        "Robotank",
        "StarGunner",
        "VideoPinball",
    ],
)

ATARI_EASY = AtariEnvs(
    envs=[
        "BeamRider",
        "Enduro",
        "FishingDerby",
        "Freeway",
        "Hero",
        "IceHockey",
        "Jamesbond",
        "KungFuMaster",
        "Phoenix",
        "Pong",
        "Qbert",
        "SpaceInvaders",
        "Tennis",
        "TimePilot",
        "Tutankham",
        "UpNDown",
    ],
)

ATARI_MEDIUM = AtariEnvs(
    envs=[
        "Alien",
        "Amidar",
        "Asterix",
        "BankHeist",
        "BattleZone",
        "Centipede",
        "ChopperCommand",
        "Defender",
        "Riverraid",
        "Seaquest",
        "Venture",
        "WizardOfWor",
        "Zaxxon",
    ],
)

ATARI_HARD = AtariEnvs(
    envs=[
        "Asteroids",
        "Berzerk",
        "Bowling",
        "DoubleDunk",
        "Frostbite",
        "Gravitar",
        "MontezumaRevenge",
        "MsPacman",
        "Pitfall",
        "PrivateEye",
        "Skiing",
        "Solaris",
        "Surround",
        "YarsRevenge",
    ],
)

ATARI_57 = AtariEnvs()
