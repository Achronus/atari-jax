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

"""Q*bert — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/QBert.cpp
# Score: getDecimalScore(0xDB, 0xDA, 0xD9) — lower/middle/higher packed BCD bytes.
SCORE_LO = 0xDB  # ones and tens
SCORE_MID = 0xDA  # hundreds and thousands
SCORE_HI = 0xD9  # ten-thousands and hundred-thousands
LIVES_ADDR = 0x88  # signed byte: starts at 2 (4 lives); 0xFE = game over


class Qbert(AtariGame):
    """Q*bert game logic: reward and terminal extraction, reset, and step.

    The lives byte (RAM[0x88]) is a signed counter that starts at 2 (= 4 lives)
    and decrements on each death; 0xFE (signed −2) signals game over.  Terminal
    when RAM[0x88] equals 0xFE.  On the terminal step the reward is forced to 0,
    matching ALE behaviour (score may reset before ALE reads it).
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
            int32 — Lives remaining (signed RAM[0x88] + 2, clamped to ≥0).
        """
        signed = ram[LIVES_ADDR].astype(jnp.int8).astype(jnp.int32)
        return jnp.maximum(jnp.int32(0), signed + jnp.int32(2))

    def get_reward(self, ram_prev: chex.Array, ram_curr: chex.Array) -> chex.Array:
        """
        Compute the reward earned in the last step as a score delta.

        Returns 0 on the terminal step (score may reset before being read).

        Parameters
        ----------
        ram_prev : chex.Array
            uint8[128] — RIOT RAM before the step.
        ram_curr : chex.Array
            uint8[128] — RIOT RAM after the step.

        Returns
        -------
        reward : chex.Array
            float32 — Score gained this step (0 when terminal).
        """
        terminal = ram_curr[LIVES_ADDR] == jnp.uint8(0xFE)
        delta = (self._score(ram_curr) - self._score(ram_prev)).astype(jnp.float32)
        return jnp.where(terminal, jnp.float32(0.0), delta)

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when RAM[0x88] equals 0xFE (game-over sentinel).

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (unused; sentinel-based detection).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return ram[LIVES_ADDR] == jnp.uint8(0xFE)
