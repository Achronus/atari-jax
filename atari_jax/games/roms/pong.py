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

"""Pong — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atari_jax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Pong.cpp
# Score: RAM[14] (player) - RAM[13] (CPU); raw integers 0–21, not BCD.
SCORE_CPU = 13  # CPU score (0–21)
SCORE_PLAYER = 14  # player score (0–21)


class Pong(AtariGame):
    """Pong game logic: reward and terminal extraction, reset, and step.

    Score is player score minus CPU score (differential, not BCD-encoded).
    Terminal when either player reaches 21 points.  No lives counter.
    """

    def _score(self, ram: chex.Array) -> chex.Array:
        """Return player score minus CPU score."""
        return ram[SCORE_PLAYER].astype(jnp.int32) - ram[SCORE_CPU].astype(jnp.int32)

    def get_lives(self, ram: chex.Array) -> chex.Array:
        """
        Read the lives counter from RAM.

        Pong has no lives counter; always returns 0.

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
        Compute the reward earned in the last step as a score differential delta.

        Parameters
        ----------
        ram_prev : chex.Array
            uint8[128] — RIOT RAM before the step.
        ram_curr : chex.Array
            uint8[128] — RIOT RAM after the step.

        Returns
        -------
        reward : chex.Array
            float32 — Change in (player score − CPU score) this step.
        """
        return (self._score(ram_curr) - self._score(ram_prev)).astype(jnp.float32)

    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Terminal when either player reaches 21 points.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (unused; score-based detection).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return (ram[SCORE_CPU] == jnp.uint8(21)) | (ram[SCORE_PLAYER] == jnp.uint8(21))
