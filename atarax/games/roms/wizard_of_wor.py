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

"""Wizard of Wor — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/WizardOfWor.cpp
# Score: getDecimalScore(0x86, 0x88) * 100 — non-sequential 2-byte BCD × 100.
# Values ≥ 8000 (before × 100) have 8000 subtracted (encoding artefact).
SCORE_LO = 0x86  # ones and tens (of hundreds)
SCORE_HI = 0x88  # hundreds and thousands (of hundreds)
LIVES_ADDR = 0x8D  # lower nibble holds lives; 0 when game over
TERM_ADDR = 0xF4  # 0xF8 on the game-over screen


class WizardOfWor(AtariGame):
    """Wizard of Wor game logic: reward and terminal extraction, reset, and step.

    Score is 2-byte packed BCD × 100 at non-sequential addresses (0x86, 0x88).
    BCD values ≥ 8000 have 8000 subtracted before scaling (encoding artefact,
    matching ALE behaviour).  Terminal when RAM[0x8D] lower nibble is 0 *and*
    RAM[0xF4] equals 0xF8.
    """

    def _score(self, ram: chex.Array) -> chex.Array:
        """Decode the 4-digit packed-BCD score (wrap-corrected) × 100."""
        lo = ram[SCORE_LO].astype(jnp.int32)
        hi = ram[SCORE_HI].astype(jnp.int32)
        bcd4 = (
            jnp.int32(1000) * (hi >> 4)
            + jnp.int32(100) * (hi & 0xF)
            + jnp.int32(10) * (lo >> 4)
            + jnp.int32(1) * (lo & 0xF)
        )
        bcd4 = jnp.where(bcd4 >= jnp.int32(8000), bcd4 - jnp.int32(8000), bcd4)
        return bcd4 * jnp.int32(100)

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
            int32 — Lives remaining (RAM[0x8D] & 0xF).
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

        Terminal when (RAM[0x8D] & 0xF) equals 0 and RAM[0xF4] equals 0xF8.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (unused; screen-based detection).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        lives_nibble = ram[LIVES_ADDR] & jnp.uint8(0x0F)
        return (lives_nibble == jnp.uint8(0)) & (ram[TERM_ADDR] == jnp.uint8(0xF8))
