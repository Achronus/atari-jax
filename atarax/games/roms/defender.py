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

"""Defender — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Defender.cpp
# Score: 6 bytes at 0x9C–0xA1; each byte's low nibble is one decimal digit.
# Nibble value 0xA is a blank/zero sentinel → treat as 0.
SCORE_BASE = 0x9C  # digit 0 (ones) through digit 5 (hundred-thousands) at +0..+5
LIVES_ADDR = 0xC2  # lives remaining (full byte)


class Defender(AtariGame):
    """Defender game logic: reward and terminal extraction, reset, and step."""

    def _score(self, ram: chex.Array) -> chex.Array:
        """Decode the 6-digit score from six RAM bytes (one nibble digit each)."""
        mults = jnp.array([1, 10, 100, 1000, 10000, 100000], dtype=jnp.int32)
        digits = (
            jnp.array([ram[SCORE_BASE + i] for i in range(6)], dtype=jnp.int32) & 0xF
        )
        digits = jnp.where(digits == jnp.int32(0xA), jnp.int32(0), digits)
        return jnp.sum(digits * mults)

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
            int32 — Lives remaining.
        """
        return ram[LIVES_ADDR].astype(jnp.int32)

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
