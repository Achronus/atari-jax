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

"""Pooyan — JAX-native game implementation.

A mama pig defends her piglets from wolves descending on balloons.  The
player's basket moves up and down; shoot arrows to pop balloons and hit
wolves before they reach the bottom.

Action space (5 actions):
    0 — NOOP
    1 — FIRE (shoot arrow)
    2 — UP
    3 — DOWN
    4 — FIRE+UP

Scoring:
    Balloon popped (wolf falls) — +110
    Wolf hits ground → life lost
    Episode ends when all lives are lost; lives: 5.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_WOLVES: int = 6
_BASKET_X: int = 10  # player basket x (fixed, left side)
_SCREEN_RIGHT: int = 150

_PLAYER_SPEED: float = 2.0
_ARROW_SPEED: float = 5.0
_WOLF_SPEED: float = 0.6

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([130, 200, 100], dtype=jnp.uint8)
_COLOR_BASKET = jnp.array([180, 120, 60], dtype=jnp.uint8)
_COLOR_WOLF = jnp.array([100, 80, 60], dtype=jnp.uint8)
_COLOR_BALLOON = jnp.array([255, 50, 50], dtype=jnp.uint8)
_COLOR_ARROW = jnp.array([200, 170, 80], dtype=jnp.uint8)
_COLOR_ROPE = jnp.array([100, 80, 40], dtype=jnp.uint8)


@chex.dataclass
class PooyanState(AtariState):
    """
    Complete Pooyan game state — a JAX pytree.

    Parameters
    ----------
    basket_y : jax.Array
        float32 — Basket y (top edge).
    arrow_x : jax.Array
        float32 — Arrow x.
    arrow_y : jax.Array
        float32 — Arrow y.
    arrow_active : jax.Array
        bool — Arrow in flight.
    wolf_x : jax.Array
        float32[6] — Wolf x (descending from right on balloons).
    wolf_y : jax.Array
        float32[6] — Wolf y.
    wolf_active : jax.Array
        bool[6] — Wolf alive (on balloon).
    spawn_timer : jax.Array
        int32 — Frames until next wolf spawns.
    wave : jax.Array
        int32 — Wave number.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    basket_y: jax.Array
    arrow_x: jax.Array
    arrow_y: jax.Array
    arrow_active: jax.Array
    wolf_x: jax.Array
    wolf_y: jax.Array
    wolf_active: jax.Array
    spawn_timer: jax.Array
    wave: jax.Array
    key: jax.Array


class Pooyan(AtariEnv):
    """
    Pooyan implemented as a pure JAX function suite.

    Pop balloons and protect piglets.  Lives: 5.
    """

    num_actions: int = 5

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> PooyanState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : PooyanState
            Basket at mid-screen, no wolves active.
        """
        return PooyanState(
            basket_y=jnp.float32(95.0),
            arrow_x=jnp.float32(float(_BASKET_X + 16)),
            arrow_y=jnp.float32(100.0),
            arrow_active=jnp.bool_(False),
            wolf_x=jnp.full(_N_WOLVES, float(_SCREEN_RIGHT), dtype=jnp.float32),
            wolf_y=jnp.linspace(30.0, 170.0, _N_WOLVES, dtype=jnp.float32),
            wolf_active=jnp.zeros(_N_WOLVES, dtype=jnp.bool_),
            spawn_timer=jnp.int32(40),
            wave=jnp.int32(1),
            lives=jnp.int32(5),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: PooyanState, action: jax.Array) -> PooyanState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : PooyanState
            Current game state.
        action : jax.Array
            int32 — Action index (0–4).

        Returns
        -------
        new_state : PooyanState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Basket movement
        move_u = (action == jnp.int32(2)) | (action == jnp.int32(4))
        move_d = action == jnp.int32(3)
        new_by = jnp.clip(
            state.basket_y
            + jnp.where(move_u, -_PLAYER_SPEED, jnp.where(move_d, _PLAYER_SPEED, 0.0)),
            jnp.float32(20.0),
            jnp.float32(175.0),
        )

        # Fire arrow
        fire = (
            (action == jnp.int32(1)) | (action == jnp.int32(4))
        ) & ~state.arrow_active
        new_ax = jnp.where(fire, jnp.float32(float(_BASKET_X + 16)), state.arrow_x)
        new_ay = jnp.where(fire, new_by + jnp.float32(6.0), state.arrow_y)
        new_aactive = state.arrow_active | fire
        new_ax = jnp.where(new_aactive, new_ax + _ARROW_SPEED, new_ax)
        new_aactive = new_aactive & (new_ax < jnp.float32(_SCREEN_RIGHT))

        # Wolves descend from right
        new_wx = state.wolf_x - jnp.where(state.wolf_active, _WOLF_SPEED, 0.0)

        # Arrow hits wolf/balloon
        arrow_hits = (
            new_aactive
            & state.wolf_active
            & (jnp.abs(new_ax - new_wx) < jnp.float32(10.0))
            & (jnp.abs(new_ay - state.wolf_y) < jnp.float32(10.0))
        )
        step_reward = step_reward + jnp.sum(arrow_hits).astype(
            jnp.float32
        ) * jnp.float32(110.0)
        new_wolf_active = state.wolf_active & ~arrow_hits
        new_aactive = new_aactive & ~jnp.any(arrow_hits)

        # Wolf reaches basket (left side) → life lost
        wolf_reaches = new_wolf_active & (new_wx <= jnp.float32(float(_BASKET_X + 20)))
        wolf_hits_basket = jnp.any(
            wolf_reaches & (jnp.abs(state.wolf_y - new_by) < jnp.float32(15.0))
        )
        # Wolves that passed left: deactivate them
        new_wolf_active2 = new_wolf_active & (new_wx > jnp.float32(float(_BASKET_X)))

        # Spawn new wolf
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        can_spawn = (new_spawn_timer <= jnp.int32(0)) & ~jnp.all(new_wolf_active2)
        free_w = jnp.argmin(new_wolf_active2.astype(jnp.int32))
        new_wolf_active3 = jnp.where(
            can_spawn, new_wolf_active2.at[free_w].set(True), new_wolf_active2
        )
        new_wx2 = jnp.where(
            can_spawn, new_wx.at[free_w].set(jnp.float32(_SCREEN_RIGHT)), new_wx
        )
        new_wy = jnp.where(
            can_spawn,
            state.wolf_y.at[free_w].set(
                jax.random.uniform(key, minval=30.0, maxval=175.0)
            ),
            state.wolf_y,
        )
        new_spawn_timer = jnp.where(can_spawn, jnp.int32(60), new_spawn_timer)

        life_lost = wolf_hits_basket
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return PooyanState(
            basket_y=new_by,
            arrow_x=new_ax,
            arrow_y=new_ay,
            arrow_active=new_aactive,
            wolf_x=new_wx2,
            wolf_y=new_wy,
            wolf_active=new_wolf_active3,
            spawn_timer=new_spawn_timer,
            wave=state.wave,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: PooyanState, action: jax.Array) -> PooyanState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : PooyanState
            Current game state.
        action : jax.Array
            int32 — Action index (0–4).

        Returns
        -------
        new_state : PooyanState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: PooyanState) -> PooyanState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: PooyanState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : PooyanState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Basket (rope at left)
        rope_mask = (
            (_COL_IDX >= _BASKET_X)
            & (_COL_IDX < _BASKET_X + 4)
            & (_ROW_IDX >= 20)
            & (_ROW_IDX < 190)
        )
        frame = jnp.where(rope_mask[:, :, None], _COLOR_ROPE, frame)

        by = state.basket_y.astype(jnp.int32)
        basket_mask = (
            (_ROW_IDX >= by)
            & (_ROW_IDX < by + 14)
            & (_COL_IDX >= _BASKET_X)
            & (_COL_IDX < _BASKET_X + 16)
        )
        frame = jnp.where(basket_mask[:, :, None], _COLOR_BASKET, frame)

        # Wolves on balloons
        def draw_wolf(frm, i):
            wx = state.wolf_x[i].astype(jnp.int32)
            wy = state.wolf_y[i].astype(jnp.int32)
            balloon_mask = (
                state.wolf_active[i]
                & (_ROW_IDX >= wy - 8)
                & (_ROW_IDX < wy + 2)
                & (_COL_IDX >= wx - 5)
                & (_COL_IDX < wx + 5)
            )
            wolf_mask = (
                state.wolf_active[i]
                & (_ROW_IDX >= wy)
                & (_ROW_IDX < wy + 10)
                & (_COL_IDX >= wx - 4)
                & (_COL_IDX < wx + 4)
            )
            frm = jnp.where(balloon_mask[:, :, None], _COLOR_BALLOON, frm)
            frm = jnp.where(wolf_mask[:, :, None], _COLOR_WOLF, frm)
            return frm, None

        frame, _ = jax.lax.scan(draw_wolf, frame, jnp.arange(_N_WOLVES))

        # Arrow
        ax = state.arrow_x.astype(jnp.int32)
        ay = state.arrow_y.astype(jnp.int32)
        am = (
            state.arrow_active
            & (_ROW_IDX >= ay - 1)
            & (_ROW_IDX < ay + 1)
            & (_COL_IDX >= ax)
            & (_COL_IDX < ax + 8)
        )
        frame = jnp.where(am[:, :, None], _COLOR_ARROW, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Pooyan action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_DOWN: 3,
            pygame.K_s: 3,
        }
