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

"""Gopher — JAX-native game implementation.

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Ground surface : y = 150
    Carrot y       : y = 160
    Carrot positions (centre x) : x = 40, 80, 120
    Player y       : y = 130
    Player x range : x ∈ [8, 144]

Action space (8 actions — ALE minimal set):
    0  NOOP
    1  FIRE
    2  UP
    3  RIGHT
    4  LEFT
    5  UPFIRE
    6  RIGHTFIRE
    7  LEFTFIRE
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxGame, AtaraxParams
from atarax.state import AtariState

# Geometry
_PLAYER_Y: int = 130
_PLAYER_LEFT: float = 8.0
_PLAYER_RIGHT: float = 144.0
_PLAYER_W: int = 10
_PLAYER_H: int = 10
_PLAYER_SPEED: float = 2.0

_BULLET_W: int = 1
_BULLET_H: int = 4
_BULLET_SPEED: float = 4.0

# Ground and carrots
_GROUND_Y: int = 150
_CARROT_Y: int = 160
_CARROT_W: int = 8
_CARROT_H: int = 12
_N_CARROTS: int = 3
_CARROT_X = jnp.array([40.0, 80.0, 120.0], dtype=jnp.float32)

# Gopher
_GOPHER_W: int = 12
_GOPHER_H: int = 10
_GOPHER_SPEED_X: float = 1.5
_GOPHER_SPEED_Y: float = 2.0
_GOPHER_POINTS: int = 200
_GOPHER_START_X: float = 5.0
_GOPHER_START_Y: float = 50.0  # above ground
_GOPHER_ABOVE_Y: float = 50.0  # y when emerging from tunnel above ground

_INIT_LIVES: int = 3
_FRAME_SKIP: int = 4

# Render
_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_SKY_COLOR = jnp.array([50, 120, 200], dtype=jnp.uint8)
_GROUND_COLOR = jnp.array([100, 70, 30], dtype=jnp.uint8)
_CARROT_COLOR = jnp.array([255, 100, 30], dtype=jnp.uint8)
_PLAYER_COLOR = jnp.array([100, 200, 100], dtype=jnp.uint8)
_BULLET_COLOR = jnp.array([255, 255, 255], dtype=jnp.uint8)
_GOPHER_COLOR = jnp.array([180, 120, 60], dtype=jnp.uint8)


@chex.dataclass
class GopherState(AtariState):
    """
    Complete Gopher game state.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    `lives` = sum(carrot_alive).

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player left-edge x ∈ [8, 144].
    bullet_x : jax.Array
        float32 — Bullet x.
    bullet_y : jax.Array
        float32 — Bullet y.
    bullet_active : jax.Array
        bool — True while bullet in flight.
    gopher_x : jax.Array
        float32 — Gopher x position.
    gopher_y : jax.Array
        float32 — Gopher y position.
    gopher_alive : jax.Array
        bool — True when gopher is visible and active.
    gopher_target : jax.Array
        int32 — Which carrot the gopher is heading for (0, 1, 2).
    gopher_digging : jax.Array
        bool — True when gopher is underground moving to carrot.
    carrot_alive : jax.Array
        bool[3] — Surviving carrots.
    """

    player_x: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_active: chex.Array
    gopher_x: chex.Array
    gopher_y: chex.Array
    gopher_alive: chex.Array
    gopher_target: chex.Array
    gopher_digging: chex.Array
    carrot_alive: chex.Array


class Gopher(AtaraxGame):
    """
    Gopher implemented as a pure-JAX function suite.

    A gopher emerges from a tunnel and heads toward one of three carrots.
    Shoot it before it steals the carrot.  Each stolen carrot costs one life.
    """

    num_actions: int = 8

    def _reset(self, key: chex.PRNGKey) -> GopherState:
        """Return the canonical initial game state."""
        return GopherState(
            player_x=jnp.float32(76.0),
            bullet_x=jnp.float32(0.0),
            bullet_y=jnp.float32(0.0),
            bullet_active=jnp.bool_(False),
            gopher_x=jnp.float32(_GOPHER_START_X),
            gopher_y=jnp.float32(_GOPHER_ABOVE_Y),
            gopher_alive=jnp.bool_(True),
            gopher_target=jnp.int32(1),  # start targeting middle carrot
            gopher_digging=jnp.bool_(False),
            carrot_alive=jnp.ones(_N_CARROTS, dtype=jnp.bool_),
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: GopherState, action: jax.Array) -> GopherState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : GopherState
            Current game state.
        action : jax.Array
            int32 — ALE action index (0–7).

        Returns
        -------
        new_state : GopherState
            State after one emulated frame.
        """
        key, subkey = jax.random.split(state.key)

        # --- Action decode ---
        move_right = (action == 3) | (action == 6)
        move_left = (action == 4) | (action == 7)
        has_fire = (action == 1) | (action == 5) | (action == 6) | (action == 7)

        # --- Player movement ---
        dx = jnp.where(
            move_right,
            jnp.float32(_PLAYER_SPEED),
            jnp.where(move_left, jnp.float32(-_PLAYER_SPEED), jnp.float32(0.0)),
        )
        player_x = jnp.clip(state.player_x + dx, _PLAYER_LEFT, _PLAYER_RIGHT)

        # --- Player fires ---
        fire = has_fire & ~state.bullet_active
        new_bx = jnp.where(fire, player_x + jnp.float32(_PLAYER_W / 2), state.bullet_x)
        new_by = jnp.where(fire, jnp.float32(_PLAYER_Y - _BULLET_H), state.bullet_y)
        bullet_active = state.bullet_active | fire

        # --- Move bullet ---
        new_by = jnp.where(bullet_active, new_by - jnp.float32(_BULLET_SPEED), new_by)
        pb_oob = bullet_active & (new_by < jnp.float32(0.0))
        bullet_active = bullet_active & ~pb_oob

        # --- Gopher AI ---
        target_x = _CARROT_X[state.gopher_target]
        target_alive = state.carrot_alive[state.gopher_target]

        # Above ground: move horizontally toward target carrot x
        gopher_dx = jnp.where(
            state.gopher_alive
            & ~state.gopher_digging
            & target_alive
            & (state.gopher_x < target_x - 1.0),
            jnp.float32(_GOPHER_SPEED_X),
            jnp.where(
                state.gopher_alive
                & ~state.gopher_digging
                & target_alive
                & (state.gopher_x > target_x + 1.0),
                jnp.float32(-_GOPHER_SPEED_X),
                jnp.float32(0.0),
            ),
        )
        new_gopher_x = state.gopher_x + gopher_dx

        # When gopher reaches target x column, start digging
        at_target_x = jnp.abs(state.gopher_x - target_x) < jnp.float32(
            _GOPHER_SPEED_X + 1.0
        )
        start_digging = (
            state.gopher_alive & ~state.gopher_digging & target_alive & at_target_x
        )
        new_gopher_digging = state.gopher_digging | start_digging

        # Digging: gopher moves down
        gopher_dy = jnp.where(
            state.gopher_alive & new_gopher_digging,
            jnp.float32(_GOPHER_SPEED_Y),
            jnp.float32(0.0),
        )
        new_gopher_y = state.gopher_y + gopher_dy

        # --- Gopher steals carrot ---
        gopher_at_carrot = (
            state.gopher_alive
            & new_gopher_digging
            & (new_gopher_y >= jnp.float32(_CARROT_Y))
            & target_alive
        )
        # Steal the target carrot
        new_carrot_alive = state.carrot_alive.at[state.gopher_target].set(
            jnp.where(
                gopher_at_carrot,
                jnp.bool_(False),
                state.carrot_alive[state.gopher_target],
            )
        )

        # After stealing, pick new target and reset gopher above ground
        new_target = jax.random.randint(subkey, (), 0, _N_CARROTS)
        new_gopher_x = jnp.where(
            gopher_at_carrot, jnp.float32(_GOPHER_START_X), new_gopher_x
        )
        new_gopher_y = jnp.where(
            gopher_at_carrot, jnp.float32(_GOPHER_ABOVE_Y), new_gopher_y
        )
        new_gopher_digging = jnp.where(
            gopher_at_carrot, jnp.bool_(False), new_gopher_digging
        )
        new_gopher_target = jnp.where(gopher_at_carrot, new_target, state.gopher_target)

        # Also pick new target if current target carrot is already dead
        no_target = ~new_carrot_alive[new_gopher_target]
        # Fall back to any alive carrot
        fallback_target = jnp.argmax(new_carrot_alive.astype(jnp.int32))
        new_gopher_target = jnp.where(no_target, fallback_target, new_gopher_target)

        # --- Bullet hits gopher (only when above ground and alive) ---
        gopher_above_ground = state.gopher_alive & ~state.gopher_digging
        bullet_hits_gopher = (
            bullet_active
            & gopher_above_ground
            & (new_bx + jnp.float32(_BULLET_W) > new_gopher_x)
            & (new_bx < new_gopher_x + jnp.float32(_GOPHER_W))
            & (new_by + jnp.float32(_BULLET_H) > new_gopher_y)
            & (new_by < new_gopher_y + jnp.float32(_GOPHER_H))
        )
        step_reward = jnp.where(
            bullet_hits_gopher, jnp.float32(_GOPHER_POINTS), jnp.float32(0.0)
        )
        bullet_active = bullet_active & ~bullet_hits_gopher

        # Respawn gopher at random side after being shot
        rand_x = jax.random.uniform(subkey, (), minval=8.0, maxval=30.0)
        new_gopher_alive = state.gopher_alive  # gopher immediately respawns
        new_gopher_x = jnp.where(bullet_hits_gopher, rand_x, new_gopher_x)
        new_gopher_y = jnp.where(
            bullet_hits_gopher, jnp.float32(_GOPHER_ABOVE_Y), new_gopher_y
        )
        new_gopher_digging = jnp.where(
            bullet_hits_gopher, jnp.bool_(False), new_gopher_digging
        )

        # --- Lives ---
        new_lives = jnp.sum(new_carrot_alive.astype(jnp.int32))
        done = new_lives <= jnp.int32(0)

        return state.__replace__(
            player_x=player_x,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=bullet_active,
            gopher_x=new_gopher_x,
            gopher_y=new_gopher_y,
            gopher_alive=new_gopher_alive,
            gopher_target=new_gopher_target,
            gopher_digging=new_gopher_digging,
            carrot_alive=new_carrot_alive,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            key=key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: GopherState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> GopherState:
        state = state.__replace__(reward=jnp.float32(0.0), key=rng)
        new_state = jax.lax.fori_loop(
            0,
            _FRAME_SKIP,
            lambda _i, s: self._step_physics(s, action),
            state,
        )
        return new_state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: GopherState) -> jax.Array:
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # --- Sky ---
        sky_mask = _ROW_IDX < _GROUND_Y
        frame = jnp.where(sky_mask[:, :, None], _SKY_COLOR[None, None, :], frame)

        # --- Ground ---
        ground_mask = _ROW_IDX >= _GROUND_Y
        frame = jnp.where(ground_mask[:, :, None], _GROUND_COLOR[None, None, :], frame)

        # --- Carrots ---
        for i in range(_N_CARROTS):
            cx = jnp.int32(_CARROT_X[i]) - _CARROT_W // 2
            carrot_mask = (
                state.carrot_alive[i]
                & (_ROW_IDX >= _CARROT_Y)
                & (_ROW_IDX < _CARROT_Y + _CARROT_H)
                & (_COL_IDX >= cx)
                & (_COL_IDX < cx + _CARROT_W)
            )
            frame = jnp.where(
                carrot_mask[:, :, None], _CARROT_COLOR[None, None, :], frame
            )

        # --- Player ---
        px = jnp.int32(state.player_x)
        player_mask = (
            (_ROW_IDX >= _PLAYER_Y)
            & (_ROW_IDX < _PLAYER_Y + _PLAYER_H)
            & (_COL_IDX >= px)
            & (_COL_IDX < px + _PLAYER_W)
        )
        frame = jnp.where(player_mask[:, :, None], _PLAYER_COLOR[None, None, :], frame)

        # --- Bullet ---
        bx = jnp.int32(state.bullet_x)
        by = jnp.int32(state.bullet_y)
        bullet_mask = (
            state.bullet_active
            & (_ROW_IDX >= by)
            & (_ROW_IDX < by + _BULLET_H)
            & (_COL_IDX >= bx)
            & (_COL_IDX < bx + _BULLET_W)
        )
        frame = jnp.where(bullet_mask[:, :, None], _BULLET_COLOR[None, None, :], frame)

        # --- Gopher ---
        gx = jnp.int32(state.gopher_x)
        gy = jnp.int32(state.gopher_y)
        gopher_visible = state.gopher_alive & ~state.gopher_digging
        gopher_mask = (
            gopher_visible
            & (_ROW_IDX >= gy)
            & (_ROW_IDX < gy + _GOPHER_H)
            & (_COL_IDX >= gx)
            & (_COL_IDX < gx + _GOPHER_W)
        )
        frame = jnp.where(gopher_mask[:, :, None], _GOPHER_COLOR[None, None, :], frame)

        return frame

    def _key_map(self):
        try:
            import pygame

            return {
                pygame.K_SPACE: 1,
                pygame.K_UP: 5,
                pygame.K_w: 5,
                pygame.K_RIGHT: 3,
                pygame.K_d: 3,
                pygame.K_LEFT: 4,
                pygame.K_a: 4,
            }
        except ImportError:
            return {}
