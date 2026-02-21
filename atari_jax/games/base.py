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
import jax
import jax.numpy as jnp

from atari_jax.core.cpu import cpu_reset
from atari_jax.core.frame import emulate_frame
from atari_jax.core.state import AtariState, new_atari_state


class AtariGame(ABC):
    """Abstract base class for a supported Atari game.

    Subclasses implement reward extraction, terminal detection, and lives
    counting for one ROM.  The shared `reset` and `step` entry points are
    provided by this base class.  All methods operate on JAX arrays and are
    fully compatible with `jax.jit` and `jax.vmap` — ``self`` is a
    Python-level constant that JAX never traces.
    """

    _WARMUP_FRAMES: int = 60

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

    def reset(self, rom: chex.Array) -> AtariState:
        """
        Initialise the machine and run warm-up frames.

        Loads the CPU reset vector from the ROM, runs `_WARMUP_FRAMES` NOOP
        frames, then captures the initial lives count and zeroes the episode
        counters.

        Parameters
        ----------
        rom : chex.Array
            uint8[ROM_SIZE] — ROM bytes for this game.

        Returns
        -------
        state : AtariState
            Ready-to-play machine state with `lives`, `reward`, `terminal`,
            and `episode_frame` initialised.
        """
        state = new_atari_state()
        state = cpu_reset(state, rom)
        state = jax.lax.fori_loop(
            0,
            self._WARMUP_FRAMES,
            lambda _, s: emulate_frame(s, rom, jnp.int32(0)),
            state,
        )
        return state.__replace__(
            lives=self.get_lives(state.riot.ram),
            episode_frame=jnp.int32(0),
            terminal=jnp.bool_(False),
            reward=jnp.float32(0.0),
        )

    def step(
        self, state: AtariState, rom: chex.Array, action: chex.Array
    ) -> AtariState:
        """
        Apply one action and emulate one ALE frame.

        Captures the pre-step RAM, runs `emulate_frame`, then updates
        `reward`, `lives`, `terminal`, and `episode_frame` using this
        game's RAM-extraction logic.

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
        ram_prev = state.riot.ram
        lives_prev = state.lives
        state = emulate_frame(state, rom, action)
        ram_curr = state.riot.ram
        return state.__replace__(
            reward=self.get_reward(ram_prev, ram_curr),
            lives=self.get_lives(ram_curr),
            terminal=self.is_terminal(ram_curr, lives_prev),
            episode_frame=(state.episode_frame + jnp.int32(1)).astype(jnp.int32),
        )
