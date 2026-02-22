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

"""Double Dunk — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/DoubleDunk.cpp
# Score: differential (player_score - opponent_score); each is 1-byte BCD.
PLAYER_ADDR = 0xF6  # player score (1-byte BCD)
OPPONENT_ADDR = 0xF7  # opponent score (1-byte BCD)
SYNC_ADDR = 0xFE  # game-phase sentinel; 0xE7 signals game-over state


class DoubleDunk(AtariGame):
    """Double Dunk game logic: reward and terminal extraction, reset, and step.

    Double Dunk has no lives counter.  The reward is the score differential
    (player minus opponent) delta per step.  The episode ends when either
    team reaches 24 points and the game-over sync byte is set.
    """

    def _decode_bcd1(self, byte: chex.Array) -> chex.Array:
        """Decode a 1-byte packed BCD value (0x00–0x99 → 0–99)."""
        b = byte.astype(jnp.int32)
        return (b >> 4) * jnp.int32(10) + (b & 0xF)

    def _score(self, ram: chex.Array) -> chex.Array:
        """Compute the score differential (player − opponent)."""
        return self._decode_bcd1(ram[PLAYER_ADDR]) - self._decode_bcd1(
            ram[OPPONENT_ADDR]
        )

    def get_lives(self, ram: chex.Array) -> chex.Array:
        """
        Read the lives counter from RAM.

        Double Dunk has no lives concept; always returns 0.

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

        Terminal when either team reaches 24 points and the game-over sync
        byte (0xE7) is set at RAM[0xFE].

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Unused (no lives concept in Double Dunk).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        player = self._decode_bcd1(ram[PLAYER_ADDR])
        opponent = self._decode_bcd1(ram[OPPONENT_ADDR])
        score_reached = (player >= jnp.int32(24)) | (opponent >= jnp.int32(24))
        sync = ram[SYNC_ADDR] == jnp.uint8(0xE7)
        return score_reached & sync
