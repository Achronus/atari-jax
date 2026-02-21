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

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvSpec:
    """
    Environment specification.

    Encodes the engine, game name, and version as structured fields and
    exposes them as the canonical `"[engine]/[env_name]-v[N]"` identifier string.

    Parameters
    ----------
    engine : str
        Backend engine name (e.g. `"atari"`).
    env_name : str
        ALE game name in lowercase (e.g. `"breakout"`).
    version : int (optional)
        Spec version. Default is `0`

    Examples
    --------
    >>> spec = EnvSpec("atari", "breakout")
    >>> spec.id
    'atari/breakout-v0'
    >>> str(spec)
    'atari/breakout-v0'
    """

    engine: str
    env_name: str
    version: int = 0

    @property
    def id(self) -> str:
        """
        Full environment identifier string.

        Returns
        -------
        id : str
            Canonical identifier, e.g. `"atari/breakout-v0"`.
        """
        return f"{self.engine}/{self.env_name}-v{self.version}"

    def __str__(self) -> str:
        return self.id
