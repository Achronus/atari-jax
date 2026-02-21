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

"""Robotank — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atari_jax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/RoboTank.cpp
# Score: RAM[0xB6] * 12 + RAM[0xB5] (dead squadrons × 12 + dead tanks).
DEAD_SQUADRONS = 0xB6  # number of enemy squadrons destroyed
DEAD_TANKS = 0xB5  # number of individual enemy tanks destroyed
LIVES_ADDR = 0xA8  # lower nibble holds lives − 1
TERM_ADDR = 0xB4  # 0xFF when game is over


class Robotank(AtariGame):
    """Robotank game logic: reward and terminal extraction, reset, and step.

    Score = RAM[0xB6] × 12 + RAM[0xB5] (destroyed squadrons and tanks).
    Terminal when RAM[0xA8] equals 0 *and* RAM[0xB4] equals 0xFF.
    """

    def _score(self, ram: chex.Array) -> chex.Array:
        """Compute score from destroyed squadrons and tanks."""
        return ram[DEAD_SQUADRONS].astype(jnp.int32) * jnp.int32(12) + ram[
            DEAD_TANKS
        ].astype(jnp.int32)

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
            int32 — Lives remaining ((RAM[0xA8] & 0xF) + 1).
        """
        return (ram[LIVES_ADDR] & jnp.uint8(0x0F)).astype(jnp.int32) + jnp.int32(1)

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

        Terminal when RAM[0xA8] equals 0 and RAM[0xB4] equals 0xFF.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (unused; flag-based detection).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return (ram[LIVES_ADDR] == jnp.uint8(0)) & (ram[TERM_ADDR] == jnp.uint8(0xFF))
