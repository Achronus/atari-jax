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

"""Abstract base class for a supported Atari game ROM module."""

from abc import ABC, abstractmethod

import chex

from atari_jax.core.state import AtariState


class AtariGame(ABC):
    """Abstract base class for a supported Atari game.

    Each subclass implements reward extraction, terminal detection,
    and the reset/step loop for one ROM.  All methods operate on
    JAX arrays and are fully compatible with `jax.jit` and `jax.vmap` —
    ``self`` is a Python-level constant that JAX never traces.
    """

    @abstractmethod
    def get_lives(self, ram: chex.Array) -> chex.Array:
        """
        Read the current lives count from RAM.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM snapshot.

        Returns
        -------
        lives : chex.Array
            int32 — Lives remaining.
        """
        raise NotImplementedError()

    @abstractmethod
    def get_reward(self, ram_prev: chex.Array, ram_curr: chex.Array) -> chex.Array:
        """
        Compute the reward earned between two RAM snapshots.

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
        raise NotImplementedError()

    @abstractmethod
    def is_terminal(self, ram: chex.Array, lives_prev: chex.Array) -> chex.Array:
        """
        Determine whether the episode has ended.

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM after the step.
        lives_prev : chex.Array
            int32 — Lives count before the step.

        Returns
        -------
        terminal : chex.Array
            bool — True when the episode ended on this step.
        """
        raise NotImplementedError()

    @abstractmethod
    def reset(self, rom: chex.Array) -> AtariState:
        """
        Initialise the machine and run warm-up frames.

        Parameters
        ----------
        rom : chex.Array
            uint8[ROM_SIZE] — ROM bytes for this game.

        Returns
        -------
        state : AtariState
            Ready-to-play machine state.
        """
        raise NotImplementedError()

    @abstractmethod
    def step(
        self, state: AtariState, rom: chex.Array, action: chex.Array
    ) -> AtariState:
        """
        Apply one action and emulate one ALE frame.

        Parameters
        ----------
        state : AtariState
            Current machine state.
        rom : chex.Array
            uint8[ROM_SIZE] — ROM bytes for this game.
        action : chex.Array
            int32 — ALE action index.

        Returns
        -------
        state : AtariState
            Updated state after the frame, with episode fields populated.
        """
        raise NotImplementedError()
