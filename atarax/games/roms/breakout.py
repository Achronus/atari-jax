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

"""Breakout — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Breakout.cpp
LIVES_ADDR = 57  # 0x39 — lives remaining; starts at 5, terminal when 0
SCORE_X = 77  # 0x4D — ones and tens digits (BCD packed)
SCORE_Y = 76  # 0x4C — hundreds digit (BCD packed)


class Breakout(AtariGame):
    """Breakout game logic: reward and terminal extraction, reset, and step."""

    def _score(self, ram: chex.Array) -> chex.Array:
        """Decode the packed BCD score from two RAM bytes."""
        x = ram[SCORE_X].astype(jnp.int32)
        y = ram[SCORE_Y].astype(jnp.int32)
        return (
            jnp.int32(1) * (x & 0xF)
            + jnp.int32(10) * ((x >> 4) & 0xF)
            + jnp.int32(100) * (y & 0xF)
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
            int32 — Lives remaining (0–5).
        """
        return ram[LIVES_ADDR].astype(jnp.int32)

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
            float32 — Score gained this step (non-negative under normal play).
        """
        return (self._score(ram_curr) - self._score(ram_prev)).astype(jnp.float32)

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        The episode is terminal when the game had started (lives_prev > 0) and
        the current lives count has reached zero.

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
