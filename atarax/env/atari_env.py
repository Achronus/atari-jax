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

"""AtariEnv — abstract base class for JAX-native Atari game implementations."""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Self, Tuple

import chex
import jax
import jax.numpy as jnp

from atarax.env._base import Env
from atarax.env.spaces import Box, Discrete


@chex.dataclass
class GameState:
    """
    Base dataclass for all JAX game state structs.

    Every game's concrete state class must inherit from `GameState` and declare
    these four fields (plus any game-specific fields).

    Parameters
    ----------
    reward : chex.Array
        float32 — Reward earned on the last `step` call.
    done : chex.Array
        bool — `True` when the episode has ended (game over or truncated).
    step : chex.Array
        int32 — Total emulated steps since power-on.
    episode_step : chex.Array
        int32 — Agent steps elapsed in the current episode.
    """

    reward: chex.Array
    done: chex.Array
    step: chex.Array
    episode_step: chex.Array


@chex.dataclass
class AtariState(GameState):
    """
    Atari-specific extension of `GameState`.

    Adds `lives` and `score` fields common to all Atari 2600 games.
    Game-specific state dataclasses (e.g. `BreakoutState`) inherit from
    `AtariState` and add their own fields.

    Parameters
    ----------
    lives : chex.Array
        int32 — Remaining lives (use `0` for games without lives).
    score : chex.Array
        int32 — Cumulative episode score.
    """

    lives: chex.Array
    score: chex.Array


@dataclass(frozen=True)
class EnvParams:
    """
    Hyper-parameters for `AtariEnv`.

    Parameters
    ----------
    noop_max : int
        Maximum number of NOOP actions taken uniformly at random at the
        start of each episode for stochastic initialisation.  Set to `0`
        to disable random starts.  Default is `30`.
    max_episode_steps : int
        Hard episode-length limit (in agent steps, i.e. after frame-skip).
        `done` is forced to `True` when `episode_step` exceeds this value.
        Default is `27000` (matching the ALE 108 000-frame limit at 4×
        frame-skip).
    """

    noop_max: int = 30
    max_episode_steps: int = 27000


class AtariEnv(Env):
    """
    Abstract base class for JAX-native Atari game implementations.

    Concrete subclasses (e.g. `Breakout`) implement the physics layer via
    three methods: `_reset`, `_step`, and `render`.  `AtariEnv` provides
    the standard `Env` interface (`reset`, `step`) with stochastic NOOP
    starts and episode truncation built in.

    Subclasses must define a class-level integer `num_actions` and
    implement `_reset`, `_step`, and `render`.

    Parameters
    ----------
    params : EnvParams (optional)
        Environment hyper-parameters.  Defaults to `EnvParams()`.
    """

    num_actions: int

    def __init__(self, params: EnvParams | None = None) -> None:
        self._params = params or EnvParams()

    @abstractmethod
    def _reset(self, key: chex.Array) -> AtariState:
        """
        Return the canonical initial game state.

        Called internally by `reset`.  Must be deterministic given the
        same `key` and compatible with `jax.jit`, `jax.vmap`, and
        `jax.lax.scan`.

        Parameters
        ----------
        key : chex.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : AtariState
            Initial game state pytree.
        """
        raise NotImplementedError

    @abstractmethod
    def _step(self, state: AtariState, action: chex.Array) -> AtariState:
        """
        Advance the game by one logical step (branch-free).

        Called internally by `step`.  All conditionals must use
        `jnp.where` — no Python branching on traced values.
        Implementations bake in the standard 4-frame skip via
        `jax.lax.fori_loop`.

        Parameters
        ----------
        state : AtariState
            Current game state pytree.
        action : chex.Array
            int32 — Action index in `[0, num_actions)`.

        Returns
        -------
        new_state : AtariState
            Updated game state pytree.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self, state: AtariState) -> chex.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : AtariState
            Current game state pytree.

        Returns
        -------
        frame : chex.Array
            uint8[210, 160, 3] — RGB image.
        """
        raise NotImplementedError

    def reset(self, key: chex.Array) -> Tuple[chex.Array, AtariState]:
        """
        Reset the environment and return the first observation.

        Applies up to `EnvParams.noop_max` random NOOP steps for
        stochastic starts.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key.

        Returns
        -------
        obs : chex.Array
            uint8[210, 160, 3] — First RGB observation.
        state : AtariState
            Initial game state after reset and any NOOP steps.  The
            concrete type is the game's own state dataclass (e.g.
            `BreakoutState`).
        """
        key, game_key, noop_key = jax.random.split(key, 3)
        state = self._reset(game_key)

        noop_max = self._params.noop_max
        if noop_max > 0:
            n_noops = jax.random.randint(
                noop_key, shape=(), minval=0, maxval=noop_max + 1, dtype=jnp.int32
            )
            state = jax.lax.fori_loop(
                0,
                n_noops,
                lambda _i, s: self._step(s, jnp.int32(0)),
                state,
            )

        obs = self.render(state).astype(jnp.uint8)
        return obs, state

    def step(
        self,
        state: AtariState,
        action: chex.Array,
    ) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Parameters
        ----------
        state : AtariState
            Current game state.
        action : chex.Array
            int32 — Action index.

        Returns
        -------
        obs : chex.Array
            uint8[210, 160, 3] — Observation after the step.
        new_state : AtariState
            Updated game state with `done` reflecting both game-over and
            episode truncation.
        reward : chex.Array
            float32 — Reward for this step.
        done : chex.Array
            bool — `True` when the episode has ended (game over or truncated).
        info : Dict[str, Any]
            `{"lives": int32, "score": int32, "episode_step": int32,
              "truncated": bool}`.
        """
        new_state = self._step(state, action)
        obs = self.render(new_state).astype(jnp.uint8)

        truncated = new_state.episode_step >= jnp.int32(self._params.max_episode_steps)
        done = new_state.done | truncated
        new_state = new_state.__replace__(done=done)

        info: Dict[str, Any] = {
            "lives": new_state.lives,
            "score": new_state.score,
            "episode_step": new_state.episode_step,
            "truncated": truncated,
        }
        return obs, new_state, new_state.reward, done, info

    def sample(self, key: chex.Array) -> chex.Array:
        """
        Sample a uniformly-random action.

        Parameters
        ----------
        key : chex.Array
            JAX PRNG key.

        Returns
        -------
        action : chex.Array
            int32 — Random action in `[0, num_actions)`.
        """
        return jax.random.randint(
            key,
            shape=(),
            minval=0,
            maxval=self.num_actions,
            dtype=jnp.int32,
        )

    @property
    def observation_space(self) -> Box:
        """Observation space: uint8[210, 160, 3]."""
        return Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    @property
    def action_space(self) -> Discrete:
        """Action space: discrete with `num_actions` actions."""
        return Discrete(n=self.num_actions)

    @property
    def unwrapped(self) -> Self:
        """Return `self` — `AtariEnv` is the innermost env."""
        return self

    def _key_map(self) -> Dict[int, int]:
        """
        Return the key-to-action mapping for interactive play.

        Override in subclasses to provide game-specific bindings.  The
        default returns an empty mapping (all key presses produce NOOP).

        Returns
        -------
        key_map : Dict[int, int]
            Mapping of pygame key constants to action indices.
        """
        return {}

    def play(self, *, scale: int = 3, fps: int = 15) -> None:
        """
        Play the game interactively in a pygame window.

        Keyboard controls are defined by `_key_map`.  Override `_key_map`
        in subclasses to provide game-specific bindings.  Pressing `Esc`
        or closing the window exits the loop.

        Requires `pygame` (`pip install pygame`).

        Parameters
        ----------
        scale : int (optional)
            Integer upscale factor applied to the native 160 × 210 screen.
            Default is `3`, giving a 480 × 630 window.
        fps : int (optional)
            Target agent steps per second.  Default is `15`, matching the
            ALE 60 Hz display rate at 4× frame skip.
        """
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "pygame is required for interactive play. "
                "Install it with: pip install pygame"
            ) from exc

        import numpy as np

        reset_fn = jax.jit(self.reset)
        step_fn = jax.jit(self.step)

        key = jax.random.PRNGKey(42)
        obs, state = reset_fn(key)

        # Warm up XLA compilation before the game loop
        _wu, *_ = step_fn(state, jnp.int32(0))
        _wu.block_until_ready()

        pygame.init()
        display = pygame.display.set_mode((160 * scale, 210 * scale))
        pygame.display.set_caption(
            f"atari-jax \u2014 {self.__class__.__name__.lower()}"
        )
        clock = pygame.time.Clock()
        key_map = self._key_map()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            keys = pygame.key.get_pressed()
            action = next((a for k, a in key_map.items() if keys[k]), 0)

            obs, state, _, done, _ = step_fn(state, jnp.int32(action))

            if bool(done):
                key, subkey = jax.random.split(key)
                obs, state = reset_fn(subkey)

            frame_np = np.asarray(obs)  # uint8[210, 160, 3]
            surf = pygame.surfarray.make_surface(frame_np.transpose(1, 0, 2))
            scaled = pygame.transform.scale(surf, (160 * scale, 210 * scale))
            display.blit(scaled, (0, 0))
            pygame.display.flip()
            clock.tick(fps)

        pygame.quit()

    def __repr__(self) -> str:
        return f"AtariEnv<{self.__class__.__name__.lower()}>"
