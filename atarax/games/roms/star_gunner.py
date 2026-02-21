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

"""Star Gunner — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/StarGunner.cpp
# Score: four nibble-digits at 0x83–0x86 (0xA sentinel = 0), × 100.
SCORE_D0 = 0x83  # ones digit (× 100)
SCORE_D1 = 0x84  # tens digit (× 100)
SCORE_D2 = 0x85  # hundreds digit (× 100)
SCORE_D3 = 0x86  # thousands digit (× 100)
LIVES_ADDR = 0x87  # lower nibble holds 0-based lives; 0 = terminal


class StarGunner(AtariGame):
    """Star Gunner game logic: reward and terminal extraction, reset, and step.

    Score is four nibble-encoded digits (×100), with value 0xA acting as a zero
    sentinel.  Terminal when RAM[0x87] equals 0.
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
            int32 — Lives remaining (RAM[0x87] & 0xF; 0-based).
        """
        return (ram[LIVES_ADDR] & jnp.uint8(0x0F)).astype(jnp.int32)

    def get_reward(self, ram_prev: chex.Array, ram_curr: chex.Array) -> chex.Array:
        """
        Compute the reward earned in the last step as a score delta.

        Parameters
        ----------
        ram_prev : chex.Array
            uint8[128] — RIOT RAM before the step.
        ram_curr : chex.Array
            uint8[128] — RIOT RAM after the step.

        Returns
        -------
        reward : chex.Array
            float32 — Score gained this step.
        """
        return (self._score(ram_curr) - self._score(ram_prev)).astype(jnp.float32)

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when RAM[0x87] equals 0.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (unused; direct zero detection).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return ram[LIVES_ADDR] == jnp.uint8(0)
