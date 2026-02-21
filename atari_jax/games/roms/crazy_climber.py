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

"""Crazy Climber — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atari_jax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/CrazyClimber.cpp
# Score: 4 bytes each holding one decimal digit (0–9), multiplied by 100.
SCORE_D0 = 0x82  # ones digit
SCORE_D1 = 0x83  # tens digit
SCORE_D2 = 0x84  # hundreds digit
SCORE_D3 = 0x85  # thousands digit
LIVES_ADDR = 0xAA  # lives remaining (full byte)


class CrazyClimber(AtariGame):
    """Crazy Climber game logic: reward and terminal extraction, reset, and step.

    Negative reward deltas are clamped to 0 to guard against a score-reset
    glitch that ALE also patches out.
    """

    def _score(self, ram: chex.Array) -> chex.Array:
        """Decode the 4-digit single-byte score (× 100) from four RAM bytes."""
        d0 = ram[SCORE_D0].astype(jnp.int32)
        d1 = ram[SCORE_D1].astype(jnp.int32)
        d2 = ram[SCORE_D2].astype(jnp.int32)
        d3 = ram[SCORE_D3].astype(jnp.int32)
        return (d0 + d1 * 10 + d2 * 100 + d3 * 1000) * 100

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
            float32 — Score gained this step (≥ 0, multiples of 100).
        """
        delta = (self._score(ram_curr) - self._score(ram_prev)).astype(jnp.float32)
        return jnp.maximum(delta, jnp.float32(0.0))

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
