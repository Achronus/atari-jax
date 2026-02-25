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

"""Environment specification following the Gymnasium naming convention."""

import re
from dataclasses import dataclass
from typing import Self

# Pattern: "engine/game_name-vN" — engine and game may contain letters,
# digits, and underscores; version N is a non-negative integer.
_ENV_ID_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_]*)/([A-Za-z][A-Za-z0-9_]*)-v(\d+)$")


@dataclass(frozen=True)
class EnvSpec:
    """
    Environment specification in Gymnasium-style format.

    Encodes the engine, environment name, and version that together form the
    standard environment ID string `"{engine}/{env_name}-v{version}"`.

    Use `EnvSpec.parse(env_id)` to construct from a string, or build
    directly with positional/keyword arguments.

    Parameters
    ----------
    engine : str
        Engine identifier, e.g. `"atari"`.
    env_name : str
        Lower-case environment name, e.g. `"breakout"`.
    version : int (optional)
        Environment version. Default is `0`.

    Examples
    --------
    >>> spec = EnvSpec("atari", "breakout")
    >>> spec.id
    'atari/breakout-v0'
    >>> EnvSpec.parse("atari/breakout-v0")
    EnvSpec(engine='atari', env_name='breakout', version=0)
    """

    engine: str
    env_name: str
    version: int = 0

    @classmethod
    def parse(cls, env_id: str | Self) -> Self:
        """
        Parse a Gymnasium-style environment ID string into an `EnvSpec`.

        Parameters
        ----------
        env_id : str | EnvSpec
            Environment ID in `"engine/env_name-vN"` format (case-insensitive),
            e.g. `"atari/breakout-v0"` or `"atari/Breakout-v0"`.

        Returns
        -------
        spec : EnvSpec
            Parsed specification with lower-cased `engine` and `game`.

        Raises
        ------
        ValueError
            If `env_id` does not match the expected format.
        """
        if isinstance(env_id, cls):
            return env_id

        m = _ENV_ID_RE.match(env_id)

        if not m:
            raise ValueError(
                f"Invalid environment ID {env_id!r}. "
                f"Expected format: 'engine/env_name-vN' (e.g. 'atari/breakout-v0')."
            )

        return cls(
            engine=m.group(1).lower(),
            env_name=m.group(2).lower(),
            version=int(m.group(3)),
        )

    @property
    def id(self) -> str:
        """Full environment ID string, e.g. `"atari/breakout-v0"`."""
        return f"{self.engine}/{self.env_name}-v{self.version}"

    def __str__(self) -> str:
        return self.id
