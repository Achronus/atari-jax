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

from abc import abstractmethod
from typing import Any, ClassVar, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from envrax.base import EnvParams, JaxEnv
from envrax.spaces import Box, Discrete

from atarax.state import AtariState


@chex.dataclass
class AtaraxParams(EnvParams):
    """
    Atari-specific environment parameters, extending `envrax.EnvParams`.

    Parameters
    ----------
    max_steps : int
        Maximum steps per episode (inherited from `EnvParams`). Default: 27000.
        Matches the ALE 108 000-frame limit at 4× frame-skip.
    noop_max : int
        Maximum number of NOOP actions at episode start for stochastic
        initialisation. Set to 0 to disable. Default: 30.
    """

    max_steps: int = 27000
    noop_max: int = 30


class AtaraxGame(JaxEnv):
    """
    Abstract base class for all JAX-native Atari game implementations.

    Implements `envrax.JaxEnv` with the new API:
      - `reset(rng, params)` — stochastic NOOP starts, returns (obs, state)
      - `step(rng, state, action, params)` — episode truncation, returns
        (obs, new_state, reward, done, info)

    Subclasses must:
      - Define `num_actions: int` as a class attribute
      - Define `game_id: ClassVar[str]` as a class attribute (snake_case name)
      - Implement `_reset(rng)` → game-specific state
      - Implement `_step(rng, state, action, params)` → new state (branch-free)
      - Implement `render(state)` → uint8[210, 160, 3] RGB frame

    The `step_env()` method from `JaxEnv` (auto-reset on done) is inherited
    and works correctly with this API.
    """

    game_id: ClassVar[str] = ""
    num_actions: int

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _reset(self, rng: chex.PRNGKey) -> AtariState:
        """
        Return the canonical initial game state.

        Called internally by `reset()`. Must be deterministic given the
        same `rng` and compatible with `jax.jit`, `jax.vmap`, and
        `jax.lax.scan`.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.

        Returns
        -------
        state : AtariState
            Initial game state pytree.
        """
        raise NotImplementedError

    @abstractmethod
    def _step(
        self,
        rng: chex.PRNGKey,
        state: AtariState,
        action: chex.Array,
        params: AtaraxParams,
    ) -> AtariState:
        """
        Advance the game by one logical step (branch-free).

        Called internally by `step()`. All conditionals must use
        `jnp.where` — no Python branching on traced values.
        Implementations include the 4-frame skip via `jax.lax.fori_loop`.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key for any in-step randomness.
        state : AtariState
            Current game state pytree.
        action : chex.Array
            int32 — Action index in `[0, num_actions)`.
        params : AtaraxParams
            Static environment parameters.

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
            Current game state.

        Returns
        -------
        frame : chex.Array
            uint8[210, 160, 3] — RGB image.
        """
        raise NotImplementedError

    def reset(
        self,
        rng: chex.PRNGKey,
        params: AtaraxParams,
    ) -> Tuple[chex.Array, AtariState]:
        """
        Reset the environment and return the first observation.

        Applies up to `params.noop_max` random NOOP steps for stochastic
        initialisation.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        params : AtaraxParams
            Environment parameters controlling NOOP starts.

        Returns
        -------
        obs : chex.Array
            uint8[210, 160, 3] — First RGB observation.
        state : AtariState
            Initial game state after reset and any NOOP steps.
        """
        rng, game_key, noop_key = jax.random.split(rng, 3)
        state = self._reset(game_key)

        n_noops = jax.random.randint(
            noop_key, shape=(), minval=0, maxval=params.noop_max + 1, dtype=jnp.int32
        )
        state = jax.lax.fori_loop(
            0,
            n_noops,
            lambda _i, s: self._step(rng, s, jnp.int32(0), params),
            state,
        )

        obs = self.render(state).astype(jnp.uint8)
        return obs, state

    def step(
        self,
        rng: chex.PRNGKey,
        state: AtariState,
        action: chex.Array,
        params: AtaraxParams,
    ) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Parameters
        ----------
        rng : chex.PRNGKey
            JAX PRNG key.
        state : AtariState
            Current game state.
        action : chex.Array
            int32 — Action index.
        params : AtaraxParams
            Environment parameters.

        Returns
        -------
        obs : chex.Array
            uint8[210, 160, 3] — Observation after the step.
        new_state : AtariState
            Updated game state with `done` reflecting game-over and truncation.
        reward : chex.Array
            float32 — Reward for this step.
        done : chex.Array
            bool — `True` when the episode has ended.
        info : Dict[str, Any]
            `{"lives", "score", "level", "episode_step", "truncated"}`.
        """
        new_state = self._step(rng, state, action, params)
        obs = self.render(new_state).astype(jnp.uint8)

        truncated = new_state.episode_step >= jnp.int32(params.max_steps)
        done = new_state.done | truncated
        new_state = new_state.__replace__(done=done)

        info: Dict[str, Any] = {
            "lives": new_state.lives,
            "score": new_state.score,
            "level": new_state.level,
            "episode_step": new_state.episode_step,
            "truncated": truncated,
        }
        return obs, new_state, new_state.reward, done, info

    @property
    def observation_space(self) -> Box:
        """Observation space: uint8[210, 160, 3]."""
        return Box(low=0, high=255, shape=(210, 160, 3), dtype=jnp.uint8)

    @property
    def action_space(self) -> Discrete:
        """Action space: discrete with `num_actions` actions."""
        return Discrete(n=self.num_actions)

    def _key_map(self) -> Dict[int, int]:
        """Key-to-action mapping for interactive play. Override in subclasses."""
        return {}

    def play(self, *, scale: int = 3, fps: int = 15) -> None:
        """Play the game interactively in a pygame window."""
        try:
            import pygame
        except ImportError as exc:
            raise ImportError(
                "pygame is required for interactive play. "
                "Install it with: pip install pygame"
            ) from exc

        import numpy as np

        params = AtaraxParams(noop_max=0)
        reset_fn = jax.jit(lambda key: self.reset(key, params))
        step_fn = jax.jit(
            lambda key, state, action: self.step(key, state, action, params)
        )

        key = jax.random.PRNGKey(42)
        obs, state = reset_fn(key)

        # Warm-up compile
        key, wk = jax.random.split(key)
        step_fn(wk, state, jnp.int32(0))[0].block_until_ready()

        pygame.init()
        display = pygame.display.set_mode((160 * scale, 210 * scale))
        pygame.display.set_caption(f"atarax \u2014 {self.__class__.__name__.lower()}")
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

            key, subkey = jax.random.split(key)
            obs, state, _, done, _ = step_fn(subkey, state, jnp.int32(action))

            if bool(done):
                key, subkey = jax.random.split(key)
                obs, state = reset_fn(subkey)

            frame_np = np.asarray(obs)
            surf = pygame.surfarray.make_surface(frame_np.transpose(1, 0, 2))
            scaled = pygame.transform.scale(surf, (160 * scale, 210 * scale))
            display.blit(scaled, (0, 0))

            font = pygame.font.SysFont("monospace", max(12, 5 * scale), bold=True)
            score_val = int(state.score)
            lives_val = int(state.lives)
            label = (
                f"SCORE {score_val}"
                if lives_val == 0
                else f"SCORE {score_val}  LIVES {lives_val}"
            )
            text_surf = font.render(label, True, (255, 255, 255), (0, 0, 0))
            display.blit(text_surf, (4, 4))

            pygame.display.flip()
            clock.tick(fps)

        pygame.quit()

    def __repr__(self) -> str:
        return f"AtaraxGame<{self.__class__.__name__.lower()}>"
