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

"""Boxing — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atari_jax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Boxing.cpp
# Score: differential (player_score - opponent_score); range -99 to +99.
# KO sentinel: byte value 0xC0 maps to 100 (knockout).
PLAYER_ADDR = 0x92  # player score (1-byte BCD; 0xC0 = KO = 100)
OPPONENT_ADDR = 0x93  # opponent score (1-byte BCD; 0xC0 = KO = 100)
TIMER_MIN = 0x90  # minutes remaining (high nibble)
TIMER_SEC = 0x91  # seconds remaining (packed BCD)


class Boxing(AtariGame):
    """Boxing game logic: reward and terminal extraction, reset, and step.

    Boxing has no lives counter.  The reward is the score differential
    (player minus opponent) delta per step.  The episode ends on a KO
    (either boxer reaches 100 punches) or when the fight clock reaches 0:00.
    """

    def _decode_score(self, byte: chex.Array) -> chex.Array:
        """Decode one boxer's score; maps KO sentinel 0xC0 to 100."""
        b = byte.astype(jnp.int32)
        normal = (b >> 4) * jnp.int32(10) + (b & 0xF)
        return jnp.where(byte == jnp.uint8(0xC0), jnp.int32(100), normal)

    def _score(self, ram: chex.Array) -> chex.Array:
        """Compute the score differential (player − opponent)."""
        return self._decode_score(ram[PLAYER_ADDR]) - self._decode_score(
            ram[OPPONENT_ADDR]
        )

    def get_lives(self, ram: chex.Array) -> chex.Array:
        """
        Read the lives counter from RAM.

        Boxing has no lives concept; always returns 0.

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

        Terminal on a KO (either boxer reaches 100) or when the fight
        clock reaches 0 minutes and 0 seconds.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Unused (no lives concept in Boxing).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        player = self._decode_score(ram[PLAYER_ADDR])
        opponent = self._decode_score(ram[OPPONENT_ADDR])
        ko = (player == jnp.int32(100)) | (opponent == jnp.int32(100))
        minutes = ram[TIMER_MIN].astype(jnp.int32) >> 4
        sec_byte = ram[TIMER_SEC].astype(jnp.int32)
        seconds = (sec_byte >> 4) * jnp.int32(10) + (sec_byte & 0xF)
        time_up = (minutes == jnp.int32(0)) & (seconds == jnp.int32(0))
        return ko | time_up
