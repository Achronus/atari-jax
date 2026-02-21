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

"""Ice Hockey — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/IceHockey.cpp
# Score: max(getDecimalScore(0x8A), 0) - max(getDecimalScore(0x8B), 0)
# (player 1 goals minus player 2 goals, each clamped to ≥0).
SCORE_P1 = 0x8A  # player-1 BCD goal count
SCORE_P2 = 0x8B  # player-2 BCD goal count
TIMER_MIN = 0x87  # period clock minutes remaining
TIMER_SEC = 0x86  # period clock seconds remaining


class IceHockey(AtariGame):
    """Ice Hockey game logic: reward and terminal extraction, reset, and step.

    Score is the differential (player goals minus opponent goals), each
    clamped to ≥0 before subtraction.  The per-step reward is additionally
    clamped to ≤1, matching ALE behaviour.  Terminal when the period clock
    reaches 0:00 (RAM[0x87] and RAM[0x86] both zero).  There is no lives
    counter.
    """

    def _score_single(self, byte: chex.Array) -> chex.Array:
        """Decode a single 1-byte BCD score and clamp to ≥0."""
        b = byte.astype(jnp.int32)
        return jnp.maximum(jnp.int32(0), (b >> 4) * jnp.int32(10) + (b & 0xF))

    def _score(self, ram: chex.Array) -> chex.Array:
        """Return player-1 score minus player-2 score."""
        return self._score_single(ram[SCORE_P1]) - self._score_single(ram[SCORE_P2])

    def get_lives(self, ram: chex.Array) -> chex.Array:
        """
        Read the lives counter from RAM.

        Ice Hockey has no lives; always returns 0.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM snapshot.

        Returns
        -------
        lives : chex.Array
            int32 — Always 0 (no lives counter in this game).
        """
        return jnp.int32(0)

    def get_reward(self, ram_prev: chex.Array, ram_curr: chex.Array) -> chex.Array:
        """
        Compute the reward earned in the last step.

        The reward is the change in score differential, clamped to ≤1.

        Parameters
        ----------
        ram_prev : chex.Array
            uint8[128] — RIOT RAM before the step.
        ram_curr : chex.Array
            uint8[128] — RIOT RAM after the step.

        Returns
        -------
        reward : chex.Array
            float32 — Score differential change this step, at most +1.
        """
        delta = (self._score(ram_curr) - self._score(ram_prev)).astype(jnp.float32)
        return jnp.minimum(delta, jnp.float32(1.0))

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when the period clock runs out (RAM[0x87] and RAM[0x86] both 0).

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (unused; timer-based detection).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return (ram[TIMER_MIN] == jnp.uint8(0)) & (ram[TIMER_SEC] == jnp.uint8(0))
