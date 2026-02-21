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

"""Tutankham — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atari_jax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Tutankham.cpp
# Score: getDecimalScore(0x9C, 0x9A) — lower/higher BCD (non-sequential).
SCORE_LO = 0x9C  # ones and tens
SCORE_HI = 0x9A  # hundreds and thousands (address 0x9B skipped)
LIVES_ADDR = 0x9E  # lower 2 bits hold lives; 0 when game over
INIT_ADDR = 0x81  # 0x84 when game is loaded but not yet started; terminal guard


class Tutankham(AtariGame):
    """Tutankham game logic: reward and terminal extraction, reset, and step.

    Score is 2-byte packed BCD at non-sequential addresses 0x9C and 0x9A.
    Terminal when RAM[0x9E] equals 0 *and* RAM[0x81] is not 0x84 (game loaded
    but not yet initialised), matching ALE behaviour.
    """

    def _score(self, ram: chex.Array) -> chex.Array:
        """Decode the 4-digit packed-BCD score from two non-sequential RAM bytes."""
        lo = ram[SCORE_LO].astype(jnp.int32)
        hi = ram[SCORE_HI].astype(jnp.int32)
        return (
            jnp.int32(1000) * (hi >> 4)
            + jnp.int32(100) * (hi & 0xF)
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
            int32 — Lives remaining (RAM[0x9E] & 0x3).
        """
        return (ram[LIVES_ADDR] & jnp.uint8(0x03)).astype(jnp.int32)

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

        Terminal when RAM[0x9E] equals 0 and RAM[0x81] is not 0x84
        (the game-loaded-but-not-started guard value).

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
        return (ram[LIVES_ADDR] == jnp.uint8(0)) & (ram[INIT_ADDR] != jnp.uint8(0x84))
