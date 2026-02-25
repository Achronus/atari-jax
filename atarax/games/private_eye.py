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

"""Private Eye — JAX-native game implementation.

A side-scrolling detective game.  Walk through city streets, collect clues
and arrest criminals.  Hook-shot to cross obstacles; handcuffs to arrest.

Action space (9 actions):
    0 — NOOP
    1 — FIRE (shoot hook)
    2 — UP   (jump / use hook)
    3 — RIGHT
    4 — DOWN (duck / use handcuffs)
    5 — LEFT
    6 — RIGHT+FIRE
    7 — LEFT+FIRE
    8 — UP+FIRE

Scoring:
    Clue collected — +1
    Criminal arrested — +25
    Episode ends when all lives are lost; lives: 5.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_GROUND_Y: int = 160
_PLAYER_SPEED: float = 2.0
_SCROLL_SPEED: float = 2.0
_N_CLUES: int = 8
_N_CRIMINALS: int = 3

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([80, 80, 120], dtype=jnp.uint8)
_COLOR_GROUND = jnp.array([100, 100, 140], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 220, 150], dtype=jnp.uint8)
_COLOR_CLUE = jnp.array([255, 200, 50], dtype=jnp.uint8)
_COLOR_CRIMINAL = jnp.array([200, 80, 80], dtype=jnp.uint8)
_COLOR_BUILDING = jnp.array([120, 110, 160], dtype=jnp.uint8)


@chex.dataclass
class PrivateEyeState(AtariState):
    """
    Complete Private Eye game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x on screen.
    player_y : jax.Array
        float32 — Player y.
    player_vy : jax.Array
        float32 — Player vertical velocity.
    scroll_x : jax.Array
        float32 — World scroll offset.
    clue_x : jax.Array
        float32[8] — Clue world x positions.
    clue_active : jax.Array
        bool[8] — Clue not yet collected.
    criminal_x : jax.Array
        float32[3] — Criminal world x.
    criminal_active : jax.Array
        bool[3] — Criminal not yet arrested.
    clues_collected : jax.Array
        int32 — Clues collected this episode.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    player_vy: jax.Array
    scroll_x: jax.Array
    clue_x: jax.Array
    clue_active: jax.Array
    criminal_x: jax.Array
    criminal_active: jax.Array
    clues_collected: jax.Array
    key: jax.Array


class PrivateEye(AtariEnv):
    """
    Private Eye implemented as a pure JAX function suite.

    Collect clues and arrest criminals.  Lives: 5.
    """

    num_actions: int = 9

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> PrivateEyeState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : PrivateEyeState
            Player at start of street, clues and criminals spread out.
        """
        return PrivateEyeState(
            player_x=jnp.float32(30.0),
            player_y=jnp.float32(float(_GROUND_Y - 12)),
            player_vy=jnp.float32(0.0),
            scroll_x=jnp.float32(0.0),
            clue_x=jnp.linspace(200.0, 1500.0, _N_CLUES, dtype=jnp.float32),
            clue_active=jnp.ones(_N_CLUES, dtype=jnp.bool_),
            criminal_x=jnp.array([500.0, 900.0, 1300.0], dtype=jnp.float32),
            criminal_active=jnp.ones(_N_CRIMINALS, dtype=jnp.bool_),
            clues_collected=jnp.int32(0),
            lives=jnp.int32(5),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(
        self, state: PrivateEyeState, action: jax.Array
    ) -> PrivateEyeState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : PrivateEyeState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : PrivateEyeState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        on_ground = state.player_y >= jnp.float32(_GROUND_Y - 13)
        move_r = (action == jnp.int32(3)) | (action == jnp.int32(6))
        move_l = (action == jnp.int32(5)) | (action == jnp.int32(7))
        jump = (action == jnp.int32(2)) | (action == jnp.int32(8))

        dx = jnp.where(move_r, _PLAYER_SPEED, jnp.where(move_l, -_PLAYER_SPEED, 0.0))
        new_px = jnp.clip(state.player_x + dx, jnp.float32(10.0), jnp.float32(130.0))

        # Scroll world when player moves right
        new_scroll_x = state.scroll_x + jnp.where(
            move_r & (new_px >= jnp.float32(100.0)), _SCROLL_SPEED, 0.0
        )

        # Jump
        new_vy = jnp.where(
            jump & on_ground, jnp.float32(-5.0), state.player_vy + jnp.float32(0.4)
        )
        new_py = jnp.minimum(
            state.player_y + new_vy, jnp.float32(float(_GROUND_Y - 12))
        )
        new_vy = jnp.where(
            new_py >= jnp.float32(float(_GROUND_Y - 12)), jnp.float32(0.0), new_vy
        )

        # Clue positions on screen
        screen_clue_x = state.clue_x - new_scroll_x
        clue_hit = (
            state.clue_active
            & (jnp.abs(screen_clue_x - new_px) < jnp.float32(10.0))
            & on_ground
        )
        step_reward = step_reward + jnp.sum(clue_hit).astype(jnp.float32)
        new_clue_active = state.clue_active & ~clue_hit
        new_clues = state.clues_collected + jnp.sum(clue_hit, dtype=jnp.int32)

        # Criminal arrest (need clues first)
        screen_crim_x = state.criminal_x - new_scroll_x
        arrest = action == jnp.int32(4)
        crim_arrest = (
            state.criminal_active
            & arrest
            & (jnp.abs(screen_crim_x - new_px) < jnp.float32(12.0))
            & on_ground
            & (new_clues > jnp.int32(0))
        )
        step_reward = step_reward + jnp.sum(crim_arrest).astype(
            jnp.float32
        ) * jnp.float32(25.0)
        new_criminal_active = state.criminal_active & ~crim_arrest

        # Criminal touches player → life lost
        crim_touches = (
            state.criminal_active
            & (jnp.abs(screen_crim_x - new_px) < jnp.float32(10.0))
            & on_ground
            & ~arrest
        )
        life_lost = jnp.any(crim_touches)
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return PrivateEyeState(
            player_x=new_px,
            player_y=new_py,
            player_vy=new_vy,
            scroll_x=new_scroll_x,
            clue_x=state.clue_x,
            clue_active=new_clue_active,
            criminal_x=state.criminal_x,
            criminal_active=new_criminal_active,
            clues_collected=new_clues,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: PrivateEyeState, action: jax.Array) -> PrivateEyeState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : PrivateEyeState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : PrivateEyeState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: PrivateEyeState) -> PrivateEyeState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: PrivateEyeState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : PrivateEyeState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Ground
        ground_mask = _ROW_IDX >= _GROUND_Y
        frame = jnp.where(ground_mask[:, :, None], _COLOR_GROUND, frame)

        # Clues on screen
        def draw_clue(frm, i):
            cx = (state.clue_x[i] - state.scroll_x).astype(jnp.int32)
            mask = (
                state.clue_active[i]
                & (_ROW_IDX >= _GROUND_Y - 8)
                & (_ROW_IDX < _GROUND_Y)
                & (_COL_IDX >= cx - 4)
                & (_COL_IDX < cx + 4)
            )
            return jnp.where(mask[:, :, None], _COLOR_CLUE, frm), None

        frame, _ = jax.lax.scan(draw_clue, frame, jnp.arange(_N_CLUES))

        # Criminals
        def draw_criminal(frm, i):
            crx = (state.criminal_x[i] - state.scroll_x).astype(jnp.int32)
            mask = (
                state.criminal_active[i]
                & (_ROW_IDX >= _GROUND_Y - 14)
                & (_ROW_IDX < _GROUND_Y)
                & (_COL_IDX >= crx - 4)
                & (_COL_IDX < crx + 4)
            )
            return jnp.where(mask[:, :, None], _COLOR_CRIMINAL, frm), None

        frame, _ = jax.lax.scan(draw_criminal, frame, jnp.arange(_N_CRIMINALS))

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        pm = (
            (_ROW_IDX >= py)
            & (_ROW_IDX < py + 12)
            & (_COL_IDX >= px)
            & (_COL_IDX < px + 8)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Private Eye action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
            pygame.K_DOWN: 4,
            pygame.K_s: 4,
            pygame.K_LEFT: 5,
            pygame.K_a: 5,
        }
