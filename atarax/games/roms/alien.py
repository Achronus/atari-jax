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

"""Alien — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Alien.cpp
# Score: 5 bytes, each encoding one decimal digit via a custom shift.
# Byte 0x80 is the zero sentinel; otherwise digit = byte >> 3.
SCORE_D0 = 0x8B  # ones
SCORE_D1 = 0x89  # tens
SCORE_D2 = 0x87  # hundreds
SCORE_D3 = 0x85  # thousands
SCORE_D4 = 0x83  # ten-thousands
LIVES_ADDR = 0xC0  # low nibble holds lives remaining


class Alien(AtariGame):
    """Alien game logic: reward and terminal extraction, reset, and step."""

    def _digit(self, byte: chex.Array) -> chex.Array:
        """Decode one Alien score digit from its RAM byte."""
        return jax.lax.select(
            byte == jnp.uint8(0x80),
            jnp.int32(0),
            byte.astype(jnp.int32) >> 3,
        )

    def _score(self, ram: chex.Array) -> chex.Array:
        """Decode the 5-digit custom-encoded score from RAM."""
        d0 = self._digit(ram[SCORE_D0])
        d1 = self._digit(ram[SCORE_D1])
        d2 = self._digit(ram[SCORE_D2])
        d3 = self._digit(ram[SCORE_D3])
        d4 = self._digit(ram[SCORE_D4])
        return (d0 + d1 * 10 + d2 * 100 + d3 * 1000 + d4 * 10000) * 10

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
            int32 — Lives remaining (low nibble of 0xC0).
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
