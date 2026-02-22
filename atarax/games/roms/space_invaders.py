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

"""Space Invaders — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/SpaceInvaders.cpp
# Score: getDecimalScore(0xE8, 0xE6) — lower/higher packed BCD (non-sequential).
SCORE_LO = 0xE8  # ones and tens
SCORE_HI = 0xE6  # hundreds and thousands
LIVES_ADDR = 0xC9  # raw lives counter
TERM_ADDR = 0x98  # bit 0x80 set when game is over; also terminal when lives == 0


class SpaceInvaders(AtariGame):
    """Space Invaders game logic: reward and terminal extraction, reset, and step.

    Score is 2-byte packed BCD (non-sequential addresses 0xE8 and 0xE6); maximum
    is 10,000.  When the computed delta is negative the score has wrapped, and
    the reward is corrected to `(10000 − prev) + curr`, matching ALE behaviour.
    Terminal when RAM[0x98] bit 0x80 is set *or* lives reach 0.
    """

    _uses_score_tracking: bool = False

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
            int32 — Lives remaining (RAM[0xC9] directly).
        """
        return ram[LIVES_ADDR].astype(jnp.int32)

    def get_reward(self, ram_prev: chex.Array, ram_curr: chex.Array) -> chex.Array:
        """
        Compute the reward earned in the last step as a score delta.

        Handles score wrap-around at 10,000 (a negative delta implies overflow).

        Parameters
        ----------
        ram_prev : chex.Array
            uint8[128] — RIOT RAM before the step.
        ram_curr : chex.Array
            uint8[128] — RIOT RAM after the step.

        Returns
        -------
        reward : chex.Array
            float32 — Score gained this step (wrap-corrected).
        """
        prev = self._score(ram_prev)
        curr = self._score(ram_curr)
        delta = curr - prev
        wrapped = jnp.int32(10000) - prev + curr
        return jnp.where(delta < jnp.int32(0), wrapped, delta).astype(jnp.float32)

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when RAM[0x98] bit 0x80 is set, or lives reach 0.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (unused; flag+lives detection).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        game_over_flag = (ram[TERM_ADDR] & jnp.uint8(0x80)) != jnp.uint8(0)
        return game_over_flag | (ram[LIVES_ADDR] == jnp.uint8(0))
