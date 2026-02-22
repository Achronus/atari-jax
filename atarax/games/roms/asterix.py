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

"""Asterix — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Asterix.cpp
# Score: getDecimalScore(0xE0, 0xDF, 0xDE) — lower/middle/higher packed BCD bytes.
SCORE_LO = 0xE0  # ones and tens
SCORE_MID = 0xDF  # hundreds and thousands
SCORE_HI = 0xDE  # ten-thousands and hundred-thousands
LIVES_ADDR = 0xD3  # low nibble holds lives remaining
DEATH_ADDR = 0xC7  # death counter; 0x01 on the terminal frame


class Asterix(AtariGame):
    """Asterix game logic: reward and terminal extraction, reset, and step.

    Terminal detection mirrors ALE: the episode ends when the death counter
    reaches 0x01 while the player has exactly 1 life left (the final death
    animation frame before the game resets itself).
    """

    def _score(self, ram: chex.Array) -> chex.Array:
        """Decode the 6-digit packed-BCD score from three RAM bytes."""
        lo = ram[SCORE_LO].astype(jnp.int32)
        mid = ram[SCORE_MID].astype(jnp.int32)
        hi = ram[SCORE_HI].astype(jnp.int32)
        return (
            jnp.int32(100000) * (hi >> 4)
            + jnp.int32(10000) * (hi & 0xF)
            + jnp.int32(1000) * (mid >> 4)
            + jnp.int32(100) * (mid & 0xF)
            + jnp.int32(10) * (lo >> 4)
            + jnp.int32(1) * (lo & 0xF)
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
            int32 — Lives remaining (low nibble of 0xD3).
        """
        return (ram[LIVES_ADDR] & jnp.uint8(0x0F)).astype(jnp.int32)

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when the death counter is 0x01 and lives equal 1 — this is
        the last-life death animation frame before the ROM resets itself.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (unused; counter-based detection).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        death = ram[DEATH_ADDR].astype(jnp.int32)
        return (death == jnp.int32(1)) & (self.get_lives(ram) == jnp.int32(1))
