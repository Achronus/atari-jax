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

"""Freeway — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atari_jax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Freeway.cpp
# Score: getDecimalScore(0x67) — 1-byte BCD, max 30 crossings.
# Reward is clamped to [0, 1]: each crossing scores exactly +1.
SCORE_ADDR = 0x67  # number of chickens crossed (1-byte BCD)
TERM_ADDR = 0x16  # 1 signals end of timed session


class Freeway(AtariGame):
    """Freeway game logic: reward and terminal extraction, reset, and step.

    Freeway has no lives counter.  The reward is clamped to [0, 1] per step
    so each successful crossing contributes exactly +1.
    """

    def _score(self, ram: chex.Array) -> chex.Array:
        """Decode the 1-byte packed-BCD crossing count."""
        b = ram[SCORE_ADDR].astype(jnp.int32)
        return (b >> 4) * jnp.int32(10) + (b & 0xF)

    def get_lives(self, ram: chex.Array) -> chex.Array:
        """
        Read the lives counter from RAM.

        Freeway has no lives concept; always returns 0.

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

        Reward is clamped to [0.0, 1.0] per step.

        Parameters
        ----------
        ram_prev : chex.Array
            uint8[128] — RIOT RAM before the step.
        ram_curr : chex.Array
            uint8[128] — RIOT RAM after the step.

        Returns
        -------
        reward : chex.Array
            float32 — Crossings completed this step (0 or 1).
        """
        delta = (self._score(ram_curr) - self._score(ram_prev)).astype(jnp.float32)
        return jnp.clip(delta, jnp.float32(0.0), jnp.float32(1.0))

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when the timed session ends (RAM[0x16] == 1).

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Unused (no lives concept in Freeway).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return ram[TERM_ADDR] == jnp.uint8(0x01)
