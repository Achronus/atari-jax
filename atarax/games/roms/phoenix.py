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

"""Phoenix — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Phoenix.cpp
# Score: (getDecimalScore(0xC8, 0xC9) * 10 + (RAM[0xC7] >> 4)) * 10
SCORE_LO = 0xC8  # tens and ones (of hundreds)
SCORE_HI = 0xC9  # thousands and hundreds (of hundreds)
SCORE_DIGIT = 0xC7  # upper nibble holds the tens digit of the raw score
LIVES_ADDR = 0xCB  # lower 3 bits hold 0-based lives count
TERM_ADDR = 0xCC  # 0x80 on the game-over screen


class Phoenix(AtariGame):
    """Phoenix game logic: reward and terminal extraction, reset, and step.

    Score encoding: two BCD bytes at 0xC8/0xC9 (giving 4 digits), shifted up by
    one decimal place, then augmented with the upper nibble of 0xC7 as the ones
    digit, then multiplied by 10 again — matching ALE behaviour exactly.
    Terminal when RAM[0xCC] equals 0x80 (game-over screen byte).
    """

    def _score(self, ram: chex.Array) -> chex.Array:
        """Compute the score matching ALE: (bcd4 * 10 + nibble) * 10."""
        lo = ram[SCORE_LO].astype(jnp.int32)
        hi = ram[SCORE_HI].astype(jnp.int32)
        bcd4 = (
            (hi >> 4) * jnp.int32(1000)
            + (hi & 0xF) * jnp.int32(100)
            + (lo >> 4) * jnp.int32(10)
            + (lo & 0xF)
        )
        digit = ram[SCORE_DIGIT].astype(jnp.int32) >> 4
        return (bcd4 * jnp.int32(10) + digit) * jnp.int32(10)

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
            int32 — Lives remaining (RAM[0xCB] & 0x7; 0-based).
        """
        return (ram[LIVES_ADDR] & jnp.uint8(0x07)).astype(jnp.int32)

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when RAM[0xCC] equals 0x80 (game-over screen byte).

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
        return ram[TERM_ADDR] == jnp.uint8(0x80)
