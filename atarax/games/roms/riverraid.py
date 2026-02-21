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

"""River Raid — reward and terminal extraction, plus reset/step entry points."""

import chex
import jax.numpy as jnp

from atarax.games.base import AtariGame

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/RiverRaid.cpp
# Score: six single-digit values at odd addresses 77–87; each digit = RAM[addr] / 8.
SCORE_ADDRS = (77, 79, 81, 83, 85, 87)  # highest to lowest digit
LIVES_BYTE = 0xC0  # 0x58 = 4 lives (start/reset); 0x59 = 1 life; else = val/8 + 1


class RiverRaid(AtariGame):
    """River Raid game logic: reward and terminal extraction, reset, and step.

    Score is encoded across six RAM bytes at addresses 77, 79, 81, 83, 85, 87
    (highest to lowest digit) using a ×8 encoding: ``digit = RAM[addr] // 8``.
    Terminal on the 0x59 → 0x58 transition of RAM[0xC0] (last life exhausted),
    detected via ``lives_prev == 1 and RAM[0xC0] == 0x58``.
    """

    def _score(self, ram: chex.Array) -> chex.Array:
        """Decode the 6-digit score from the ×8-encoded RAM bytes."""
        multipliers = jnp.array([100000, 10000, 1000, 100, 10, 1], dtype=jnp.int32)
        digits = jnp.array(
            [ram[a].astype(jnp.int32) >> 3 for a in SCORE_ADDRS],
            dtype=jnp.int32,
        )
        return jnp.sum(digits * multipliers)

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
            int32 — Lives remaining: 4 for byte 0x58, 1 for 0x59, else byte/8 + 1.
        """
        byte = ram[LIVES_BYTE].astype(jnp.int32)
        normal = (byte >> 3) + jnp.int32(1)
        return jnp.where(
            ram[LIVES_BYTE] == jnp.uint8(0x58),
            jnp.int32(4),
            jnp.where(ram[LIVES_BYTE] == jnp.uint8(0x59), jnp.int32(1), normal),
        )

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

        Terminal on the 0x59 → 0x58 transition of RAM[0xC0], detected as
        ``RAM[0xC0] == 0x58 and lives_prev == 1``.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count from before the step (1 when RAM[0xC0] was 0x59).

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        return (ram[LIVES_BYTE] == jnp.uint8(0x58)) & (lives_prev == jnp.int32(1))
