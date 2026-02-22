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

"""Skiing — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Skiing.cpp
# Score: -(minutes * 6000 + centiseconds); reward is negative (time penalty).
TIME_CS_LO = 0xEA  # centiseconds, BCD ones and tens
TIME_CS_HI = 0xE9  # centiseconds, BCD hundreds and thousands
TIME_MIN = 0xE8  # minutes remaining (raw integer)
TERM_ADDR = 0x91  # 0xFF when the run is complete


class Skiing(AtariGame):
    """Skiing game logic: reward and terminal extraction, reset, and step.

    Score represents elapsed time (higher = worse): minutes × 6000 + centiseconds
    (BCD-encoded).  Reward is the *decrease* in elapsed time per step, i.e.
    `prev_score − curr_score`, so it is ≤0 (time-penalty formulation matching
    ALE).  Terminal when RAM[0x91] equals 0xFF (run complete).  No lives counter.

    `get_score` returns the negated elapsed time so that the standard
    `state.score` delta naturally yields `(−curr) − (−prev) = prev − curr`.
    """

    def _score(self, ram: chex.Array) -> chex.Array:
        """Compute elapsed time in centiseconds (minutes × 6000 + cs BCD)."""
        lo = ram[TIME_CS_LO].astype(jnp.int32)
        hi = ram[TIME_CS_HI].astype(jnp.int32)
        centiseconds = (
            jnp.int32(1000) * (hi >> 4)
            + jnp.int32(100) * (hi & 0xF)
            + jnp.int32(10) * (lo >> 4)
            + jnp.int32(1) * (lo & 0xF)
        )
        minutes = ram[TIME_MIN].astype(jnp.int32)
        return minutes * jnp.int32(6000) + centiseconds

    def get_score(self, ram: chex.Array) -> chex.Array:
        """
        Return the negated elapsed time so the reward delta is ≤0.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM snapshot.

        Returns
        -------
        score : chex.Array
            int32 — Negated elapsed time (`−(minutes × 6000 + centiseconds)`).
        """
        return -self._score(ram)

    def get_lives(self, ram: chex.Array) -> chex.Array:
        """
        Read the lives counter from RAM.

        Skiing has no lives counter; always returns 0.

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

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when RAM[0x91] equals 0xFF (run complete).

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
        return ram[TERM_ADDR] == jnp.uint8(0xFF)
