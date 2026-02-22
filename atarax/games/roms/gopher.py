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

"""Gopher — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Gopher.cpp
# Score: getDecimalScore(0xB2, 0xB1, 0xB0) — lower/middle/higher packed BCD bytes.
SCORE_LO = 0xB2  # ones and tens
SCORE_MID = 0xB1  # hundreds and thousands
SCORE_HI = 0xB0  # ten-thousands and hundred-thousands
CARROT_ADDR = 0xB4  # 3-bit bitmask of surviving carrots (bits 0–2); popcount = lives


class Gopher(AtariGame):
    """Gopher game logic: reward and terminal extraction, reset, and step.

    Lives in Gopher are the count of surviving carrots encoded as a 3-bit
    bitmask at RAM[0xB4].  The episode ends when all three carrots are eaten
    (bitmask == 0).
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

        Lives equal the number of surviving carrots — the popcount of the
        3-bit bitmask at RAM[0xB4].

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM snapshot.

        Returns
        -------
        lives : chex.Array
            int32 — Carrots remaining (0–3).
        """
        bits = ram[CARROT_ADDR].astype(jnp.int32) & 0x7
        return (bits & 1) + ((bits >> 1) & 1) + ((bits >> 2) & 1)

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when all carrots have been eaten (carrot bitmask == 0).

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (unused; bitmask detection).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return (ram[CARROT_ADDR].astype(jnp.int32) & 0x7) == jnp.int32(0)
