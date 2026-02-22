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

"""Road Runner — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/RoadRunner.cpp
# Score: four nibble-digits at 0xC9–0xCC (0xA sentinel = 0), × 100.
SCORE_D0 = 0xC9  # ones digit (× 100)
SCORE_D1 = 0xCA  # tens digit (× 100)
SCORE_D2 = 0xCB  # hundreds digit (× 100)
SCORE_D3 = 0xCC  # thousands digit (× 100)
LIVES_ADDR = 0xC4  # lower 3 bits hold 0-based lives count
Y_VEL = 0xB9  # non-zero during death animation
X_VEL_DEAD = 0xBD  # non-zero during death animation


class RoadRunner(AtariGame):
    """Road Runner game logic: reward and terminal extraction, reset, and step.

    Score is four nibble-encoded digits (×100), with the value 0xA acting as a
    zero sentinel.  Terminal when lives reach 0 and either velocity byte is
    non-zero (death animation is playing).
    """

    def _digit(self, byte: chex.Array) -> chex.Array:
        """Extract lower nibble and map sentinel 0xA to 0."""
        val = byte.astype(jnp.int32) & 0xF
        return jnp.where(val == jnp.int32(0xA), jnp.int32(0), val)

    def _score(self, ram: chex.Array) -> chex.Array:
        """Compute score from four nibble-encoded digits × 100."""
        raw = (
            self._digit(ram[SCORE_D0])
            + self._digit(ram[SCORE_D1]) * jnp.int32(10)
            + self._digit(ram[SCORE_D2]) * jnp.int32(100)
            + self._digit(ram[SCORE_D3]) * jnp.int32(1000)
        )
        return raw * jnp.int32(100)

    def get_lives(self, ram: chex.Array) -> chex.Array:
        """
        Read the lives counter from RAM.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM snapshot.

        Returns
        -------
        lives : chex.Array
            int32 — Lives remaining ((RAM[0xC4] & 0x7) + 1).
        """
        return (ram[LIVES_ADDR] & jnp.uint8(0x07)).astype(jnp.int32) + jnp.int32(1)

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when lives reach 0 and either y-velocity or x-death-velocity
        is non-zero (death animation active).

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (unused; velocity detection).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        lives_byte = ram[LIVES_ADDR] & jnp.uint8(0x07)
        animating = (ram[Y_VEL] != jnp.uint8(0)) | (ram[X_VEL_DEAD] != jnp.uint8(0))
        return (lives_byte == jnp.uint8(0)) & animating
