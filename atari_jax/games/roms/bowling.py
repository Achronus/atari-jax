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

"""Bowling — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atari_jax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Bowling.cpp
# Score: getDecimalScore(0xA1, 0xA6) — lower/higher packed BCD bytes.
SCORE_LO = 0xA1  # ones and tens
SCORE_HI = 0xA6  # hundreds and thousands
ROUND_ADDR = 0xA4  # current round; > 0x10 signals game over (10-frame game)


class Bowling(AtariGame):
    """Bowling game logic: reward and terminal extraction, reset, and step.

    Bowling has no lives counter.  The episode ends after round 0x10 (16th
    round pointer, which represents the end of a 10-frame game).
    """

    def _score(self, ram: chex.Array) -> chex.Array:
        """Decode the 4-digit packed-BCD score from two RAM bytes."""
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

        Bowling has no lives concept; always returns 0.

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
            float32 — Score gained this step.
        """
        return (self._score(ram_curr) - self._score(ram_prev)).astype(jnp.float32)

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when the round counter exceeds 0x10 (10-frame game complete).

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Unused (no lives concept in Bowling).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return ram[ROUND_ADDR].astype(jnp.int32) > jnp.int32(0x10)
