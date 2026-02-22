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

"""Enduro — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Enduro.cpp
# Score: cumulative cars passed; day 1 quota = 200, subsequent days = 300.
# Remaining cars (counting down) stored as getDecimalScore(0xAB, 0xAC).
CARS_LO = 0xAB  # ones and tens of remaining cars
CARS_HI = 0xAC  # hundreds and thousands of remaining cars
DAY_ADDR = 0xAD  # current day (1-indexed; 0 = not started)
TERM_ADDR = 0xAF  # 0xFF signals game over


class Enduro(AtariGame):
    """Enduro game logic: reward and terminal extraction, reset, and step.

    Enduro has no lives counter.  The score is cumulative cars passed across
    all days; the reward is the per-step delta of this cumulative total.
    """

    def _score(self, ram: chex.Array) -> chex.Array:
        """Compute cumulative cars passed from RAM."""
        level = ram[DAY_ADDR].astype(jnp.int32)
        lo = ram[CARS_LO].astype(jnp.int32)
        hi = ram[CARS_HI].astype(jnp.int32)
        remaining = (
            jnp.int32(1000) * (hi >> 4)
            + jnp.int32(100) * (hi & 0xF)
            + jnp.int32(10) * (lo >> 4)
            + jnp.int32(1) * (lo & 0xF)
        )
        quota = jnp.where(level == jnp.int32(1), jnp.int32(200), jnp.int32(300))
        passed_today = quota - remaining
        prev_total = jnp.where(
            level >= jnp.int32(2),
            jnp.int32(200) + (level - jnp.int32(2)) * jnp.int32(300),
            jnp.int32(0),
        )
        return jnp.where(level == jnp.int32(0), jnp.int32(0), prev_total + passed_today)

    def get_lives(self, ram: chex.Array) -> chex.Array:
        """
        Read the lives counter from RAM.

        Enduro has no lives concept; always returns 0.

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

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when RAM[0xAF] equals 0xFF (ran out of time on day 1).

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Unused (no lives concept in Enduro).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return ram[TERM_ADDR] == jnp.uint8(0xFF)
