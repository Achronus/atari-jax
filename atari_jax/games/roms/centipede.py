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

"""Centipede — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atari_jax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Centipede.cpp
# Score: getDecimalScore(0x76, 0x75, 0x74) — lower/middle/higher packed BCD bytes.
SCORE_LO = 0x76  # ones and tens
SCORE_MID = 0x75  # hundreds and thousands
SCORE_HI = 0x74  # ten-thousands and hundred-thousands
LIVES_ADDR = 0xED  # bits [6:4] hold lives − 1
TERM_ADDR = 0xA6  # bit 6 set when the game is over


class Centipede(AtariGame):
    """Centipede game logic: reward and terminal extraction, reset, and step.

    Negative reward deltas are clamped to 0 to guard against a score-reset
    glitch that ALE also patches out.
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
            int32 — Lives remaining (bits [6:4] of 0xED, plus 1).
        """
        return ((ram[LIVES_ADDR].astype(jnp.int32) >> 4) & 0x7) + jnp.int32(1)

    def get_reward(self, ram_prev: chex.Array, ram_curr: chex.Array) -> chex.Array:
        """
        Compute the reward earned in the last step as a score delta.

        Negative deltas are clamped to 0 to guard against a score-reset glitch.

        Parameters
        ----------
        ram_prev : chex.Array
            uint8[128] — RIOT RAM before the step.
        ram_curr : chex.Array
            uint8[128] — RIOT RAM after the step.

        Returns
        -------
        reward : chex.Array
            float32 — Score gained this step (≥ 0).
        """
        delta = (self._score(ram_curr) - self._score(ram_prev)).astype(jnp.float32)
        return jnp.maximum(delta, jnp.float32(0.0))

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when bit 6 of RAM[0xA6] is set.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (unused; bit-flag detection).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return (ram[TERM_ADDR].astype(jnp.int32) & 0x40) != jnp.int32(0)
