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

"""Fishing Derby — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atari_jax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/FishingDerby.cpp
# Score: differential (player_score - opponent_score); each is 1-byte BCD.
# Terminal fires when the raw byte equals 0x99 (BCD 99 = winning score).
PLAYER_ADDR = 0xBD  # player score (1-byte BCD; 0x99 = win)
OPPONENT_ADDR = 0xBE  # opponent score (1-byte BCD; 0x99 = win)


class FishingDerby(AtariGame):
    """Fishing Derby game logic: reward and terminal extraction, reset, and step.

    Fishing Derby has no lives counter.  The reward is the score differential
    (player minus opponent) delta per step.  The episode ends when either
    player's raw score byte reaches 0x99 (BCD 99).
    """

    def _decode_bcd1(self, byte: chex.Array) -> chex.Array:
        """Decode a 1-byte packed BCD value; clamp to 0 if negative would result."""
        b = byte.astype(jnp.int32)
        return jnp.maximum((b >> 4) * jnp.int32(10) + (b & 0xF), jnp.int32(0))

    def _score(self, ram: chex.Array) -> chex.Array:
        """Compute the score differential (player − opponent)."""
        return self._decode_bcd1(ram[PLAYER_ADDR]) - self._decode_bcd1(
            ram[OPPONENT_ADDR]
        )

    def get_lives(self, ram: chex.Array) -> chex.Array:
        """
        Read the lives counter from RAM.

        Fishing Derby has no lives concept; always returns 0.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM snapshot.

        Returns
        -------
        lives : chex.Array
            int32 — Always 0.
        """
        return jnp.int32(0)

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
            float32 — Change in score differential this step.
        """
        return (self._score(ram_curr) - self._score(ram_prev)).astype(jnp.float32)

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when either player's raw score byte equals 0x99 (BCD 99).

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Unused (no lives concept in Fishing Derby).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return (ram[PLAYER_ADDR] == jnp.uint8(0x99)) | (
            ram[OPPONENT_ADDR] == jnp.uint8(0x99)
        )
