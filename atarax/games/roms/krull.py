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

"""Krull — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Krull.cpp
# Score: getDecimalScore(0x9E, 0x9D, 0x9C) — lower/middle/higher packed BCD bytes.
SCORE_LO = 0x9E  # ones and tens
SCORE_MID = 0x9D  # hundreds and thousands
SCORE_HI = 0x9C  # ten-thousands and hundred-thousands
LIVES_ADDR = 0x9F  # lower 3 bits hold lives − 1; 0 means game over
TERM_BYTE1 = 0xA2  # must equal 0x03 at game over
TERM_BYTE2 = 0x80  # must equal 0x80 at game over


class Krull(AtariGame):
    """Krull game logic: reward and terminal extraction, reset, and step.

    Terminal when RAM[0x9F] equals 0, RAM[0xA2] equals 0x03, and RAM[0x80]
    equals 0x80 (all three conditions must hold simultaneously).
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
            int32 — Lives remaining ((RAM[0x9F] & 0x7) + 1).
        """
        return (ram[LIVES_ADDR] & jnp.uint8(0x07)).astype(jnp.int32) + jnp.int32(1)

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

        Terminal when RAM[0x9F]==0, RAM[0xA2]==0x03, and RAM[0x80]==0x80.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (unused; multi-byte detection).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return (
            (ram[LIVES_ADDR] == jnp.uint8(0))
            & (ram[TERM_BYTE1] == jnp.uint8(0x03))
            & (ram[TERM_BYTE2] == jnp.uint8(0x80))
        )
