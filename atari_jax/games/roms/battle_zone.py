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

"""Battle Zone — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atari_jax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/BattleZone.cpp
# Score: custom nibble decode from two bytes; scores are multiples of 1,000.
# Nibble value 10 is treated as 0 (game-over/reset sentinel in BCD encoding).
SCORE_A = 0x9D  # high nibble = ones digit of (score / 1000)
SCORE_B = 0x9E  # high nibble = hundreds digit, low nibble = tens digit
LIVES_ADDR = 0xBA  # low nibble holds lives remaining


class BattleZone(AtariGame):
    """Battle Zone game logic: reward and terminal extraction, reset, and step."""

    def _digit10(self, val: chex.Array) -> chex.Array:
        """Replace nibble value 10 with 0 (BattleZone zero sentinel)."""
        return jnp.where(val == jnp.int32(10), jnp.int32(0), val)

    def _score(self, ram: chex.Array) -> chex.Array:
        """Decode the custom-packed score from two RAM bytes (× 1,000)."""
        a = ram[SCORE_A].astype(jnp.int32)
        b = ram[SCORE_B].astype(jnp.int32)
        ones = self._digit10(a >> 4)
        tens = self._digit10(b & 0xF)
        hundreds = self._digit10(b >> 4)
        return (ones + jnp.int32(10) * tens + jnp.int32(100) * hundreds) * jnp.int32(
            1000
        )

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
            int32 — Lives remaining (low nibble of 0xBA).
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
            float32 — Score gained this step (multiples of 1,000).
        """
        return (self._score(ram_curr) - self._score(ram_prev)).astype(jnp.float32)

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step.

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return (lives_prev > jnp.int32(0)) & (self.get_lives(ram) == jnp.int32(0))
