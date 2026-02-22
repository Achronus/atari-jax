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

from atarax.core.cpu import cpu_reset
from atarax.core.frame import emulate_frame
from atarax.core.state import AtariState, new_atari_state


class AtariGame(ABC):
    """Abstract base class for a supported Atari game.

    Subclasses implement reward extraction, terminal detection, and lives
    counting for one ROM.  The shared `reset` and `step` entry points are
    provided by this base class.  All methods operate on JAX arrays and are
    fully compatible with `jax.jit` and `jax.vmap` — `self` is a
    Python-level constant that JAX never traces.

    The default reward strategy mirrors ALE's `m_score` baseline: after
    warmup, `reset` captures the current raw score into `state.score`; each
    `step` computes `reward = get_score(ram_curr) - state.score` and writes
    the new score back.  Games that cannot use this strategy (e.g. Tennis,
    which needs two RAM snapshots for priority-based reward) should set
    `_uses_score_tracking = False` and override `get_reward` instead.
    """

    _WARMUP_FRAMES: int = 60
    _uses_score_tracking: bool = True

    @abstractmethod
    def _score(self, ram: chex.Array) -> chex.Array:
        """
        Decode the raw game score from RAM.

        Every standard game subclass implements this helper.  The base
        `get_score` default calls it; `step` uses `get_score` rather than
        `_score` directly so subclasses can transform the value (e.g.
        Skiing negates, Tennis stubs to 0).

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM snapshot.

        Returns
        -------
        score : chex.Array
            int32 — Raw decoded score.
        """
        raise NotImplementedError()

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

    def get_score(self, ram: chex.Array) -> chex.Array:
        """
        Read the raw game score from RAM.

        Default implementation calls `self._score(ram)`, which every
        standard game subclass defines.  Override when the score sign or
        encoding differs (e.g. Skiing negates, Tennis stubs to 0).

        Parameters
        ----------
        ram : chex.Array
            uint8[128] — RIOT RAM snapshot.

        Returns
        -------
        score : chex.Array
            int32 — Raw game score.
        """
        return self._score(ram)

    def get_reward(self, ram_prev: chex.Array, ram_curr: chex.Array) -> chex.Array:
        """
        Compute reward directly from two consecutive RAM snapshots.

        Override this when `_uses_score_tracking = False` (e.g. Tennis).
        The default raises `NotImplementedError`; it is never called when
        `_uses_score_tracking = True`.

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
        raise NotImplementedError(
            f"{type(self).__name__} sets _uses_score_tracking=False but "
            "does not override get_reward."
        )

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

        Loads the CPU reset vector from the ROM, runs 10 NOOP frames, fires
        one FIRE action (action 1) to start the game and trigger the ROM's
        score-initialisation path, then runs `_WARMUP_FRAMES - 11` more NOOP
        frames before capturing the initial lives count and zeroing the
        episode counters.

        Parameters
        ----------
        rom : chex.Array
            uint8[ROM_SIZE] — ROM bytes for this game.

        Returns
        -------
        state : AtariState
            Ready-to-play machine state with `lives`, `score`, `reward`,
            `terminal`, and `episode_frame` initialised.
        """
        state = new_atari_state()
        state = cpu_reset(state, rom)
        state = jax.lax.fori_loop(
            0,
            10,
            lambda _, s: emulate_frame(s, rom, jnp.int32(0)),
            state,
        )
        state = emulate_frame(state, rom, jnp.int32(1))
        state = jax.lax.fori_loop(
            0,
            self._WARMUP_FRAMES - 11,
            lambda _, s: emulate_frame(s, rom, jnp.int32(0)),
            state,
        )
        if self._uses_score_tracking:
            init_score = self.get_score(state.riot.ram)
        else:
            init_score = jnp.int32(0)
        return state.__replace__(
            score=init_score,
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
        `reward`, `score`, `lives`, `terminal`, and `episode_frame` using
        this game's RAM-extraction logic.

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
        if self._uses_score_tracking:
            curr_score = self.get_score(ram_curr)
            reward = (curr_score - state.score).astype(jnp.float32)
            new_score = curr_score
        else:
            reward = self.get_reward(ram_prev, ram_curr)
            new_score = state.score
        return state.__replace__(
            score=new_score,
            reward=reward,
            lives=self.get_lives(ram_curr),
            terminal=self.is_terminal(ram_curr, lives_prev),
            episode_frame=(state.episode_frame + jnp.int32(1)).astype(jnp.int32),
        )
