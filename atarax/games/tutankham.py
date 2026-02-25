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

"""Tutankham — JAX-native game implementation.

Navigate a multi-room Egyptian tomb, collect treasures, and shoot enemies.
Enemies are birds and snakes; there's a limited-use magic wand.  Collect
all treasures in a room to advance.

Action space (9 actions):
    0 — NOOP
    1 — FIRE (shoot laser — only fires left/right)
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT
    6 — MAGIC WAND (kills all enemies on screen)
    7 — FIRE+RIGHT
    8 — FIRE+LEFT

Scoring:
    Enemy shot — +100
    Treasure collected — +50 to +250
    Room clear bonus — +200
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

_N_ENEMIES: int = 4
_N_TREASURES: int = 5
_PLAYER_SPEED: float = 2.0
_BULLET_SPEED: float = 5.0
_ENEMY_SPEED: float = 0.8

_ROOM_W: int = 140
_ROOM_H: int = 150
_ROOM_X0: int = 10
_ROOM_Y0: int = 30

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_WALL = jnp.array([100, 80, 40], dtype=jnp.uint8)
_COLOR_FLOOR = jnp.array([60, 50, 30], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 220, 100], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([200, 80, 80], dtype=jnp.uint8)
_COLOR_TREASURE = jnp.array([255, 200, 0], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([100, 200, 255], dtype=jnp.uint8)


@chex.dataclass
class TutankhamState(AtariState):
    """
    Complete Tutankham game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    player_y : jax.Array
        float32 — Player y.
    player_dir : jax.Array
        int32 — Last horizontal facing direction (1=right, -1=left).
    bullet_x : jax.Array
        float32 — Bullet x.
    bullet_y : jax.Array
        float32 — Bullet y.
    bullet_active : jax.Array
        bool — Bullet in flight (horizontal).
    bullet_dir : jax.Array
        int32 — Bullet direction (1=right, -1=left).
    enemy_x : jax.Array
        float32[4] — Enemy x.
    enemy_y : jax.Array
        float32[4] — Enemy y.
    enemy_dx : jax.Array
        float32[4] — Enemy x velocity.
    enemy_active : jax.Array
        bool[4] — Enemy alive.
    treasure_x : jax.Array
        float32[5] — Treasure x.
    treasure_y : jax.Array
        float32[5] — Treasure y.
    treasure_active : jax.Array
        bool[5] — Treasure not yet collected.
    wand_uses : jax.Array
        int32 — Magic wand uses remaining.
    room : jax.Array
        int32 — Current room.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    player_dir: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    bullet_dir: jax.Array
    enemy_x: jax.Array
    enemy_y: jax.Array
    enemy_dx: jax.Array
    enemy_active: jax.Array
    treasure_x: jax.Array
    treasure_y: jax.Array
    treasure_active: jax.Array
    wand_uses: jax.Array
    room: jax.Array
    key: jax.Array


class Tutankham(AtariEnv):
    """
    Tutankham implemented as a pure JAX function suite.

    Collect treasures and defeat enemies.  Lives: 3.
    """

    num_actions: int = 9

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> TutankhamState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : TutankhamState
            Player at room entrance, enemies and treasures placed.
        """
        return TutankhamState(
            player_x=jnp.float32(20.0),
            player_y=jnp.float32(100.0),
            player_dir=jnp.int32(1),
            bullet_x=jnp.float32(20.0),
            bullet_y=jnp.float32(100.0),
            bullet_active=jnp.bool_(False),
            bullet_dir=jnp.int32(1),
            enemy_x=jnp.array([60.0, 100.0, 80.0, 120.0], dtype=jnp.float32),
            enemy_y=jnp.array([60.0, 80.0, 120.0, 140.0], dtype=jnp.float32),
            enemy_dx=jnp.array([0.8, -0.8, 0.8, -0.8], dtype=jnp.float32),
            enemy_active=jnp.ones(_N_ENEMIES, dtype=jnp.bool_),
            treasure_x=jnp.array([30.0, 60.0, 90.0, 120.0, 140.0], dtype=jnp.float32),
            treasure_y=jnp.array([50.0, 70.0, 90.0, 110.0, 130.0], dtype=jnp.float32),
            treasure_active=jnp.ones(_N_TREASURES, dtype=jnp.bool_),
            wand_uses=jnp.int32(3),
            room=jnp.int32(0),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: TutankhamState, action: jax.Array) -> TutankhamState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : TutankhamState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : TutankhamState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Player movement (maze-constrained; simplified to room bounds)
        dx = jnp.where(
            action == 3, _PLAYER_SPEED, jnp.where(action == 5, -_PLAYER_SPEED, 0.0)
        )
        dy = jnp.where(
            action == 2, -_PLAYER_SPEED, jnp.where(action == 4, _PLAYER_SPEED, 0.0)
        )
        new_px = jnp.clip(
            state.player_x + dx, float(_ROOM_X0), float(_ROOM_X0 + _ROOM_W - 8)
        )
        new_py = jnp.clip(
            state.player_y + dy, float(_ROOM_Y0), float(_ROOM_Y0 + _ROOM_H - 12)
        )

        # Update facing direction
        new_dir = jnp.where(
            dx > 0, jnp.int32(1), jnp.where(dx < 0, jnp.int32(-1), state.player_dir)
        )

        # Fire horizontal laser
        fire = (
            (action == jnp.int32(1))
            | (action == jnp.int32(7))
            | (action == jnp.int32(8))
        ) & ~state.bullet_active
        new_bx = jnp.where(
            fire,
            new_px + jnp.where(new_dir > 0, jnp.float32(8.0), jnp.float32(-4.0)),
            state.bullet_x,
        )
        new_by = jnp.where(fire, new_py + jnp.float32(6.0), state.bullet_y)
        new_bdir = jnp.where(fire, new_dir, state.bullet_dir)
        new_bactive = state.bullet_active | fire
        new_bx = jnp.where(
            new_bactive, new_bx + new_bdir.astype(jnp.float32) * _BULLET_SPEED, new_bx
        )
        new_bactive = (
            new_bactive
            & (new_bx > float(_ROOM_X0))
            & (new_bx < float(_ROOM_X0 + _ROOM_W))
        )

        # Enemy patrol (horizontal bounce)
        new_ex = state.enemy_x + state.enemy_dx * jnp.where(
            state.enemy_active, jnp.float32(1.0), jnp.float32(0.0)
        )
        at_edge = (new_ex < float(_ROOM_X0)) | (new_ex > float(_ROOM_X0 + _ROOM_W - 8))
        new_edx = jnp.where(
            at_edge & state.enemy_active, -state.enemy_dx, state.enemy_dx
        )
        new_ex = jnp.clip(new_ex, float(_ROOM_X0), float(_ROOM_X0 + _ROOM_W - 8))

        # Bullet hits enemy
        b_hits_e = (
            new_bactive
            & state.enemy_active
            & (jnp.abs(new_bx - new_ex) < jnp.float32(10.0))
            & (jnp.abs(new_by - state.enemy_y) < jnp.float32(10.0))
        )
        step_reward = step_reward + jnp.sum(b_hits_e).astype(jnp.float32) * jnp.float32(
            100.0
        )
        new_enemy_active = state.enemy_active & ~b_hits_e
        new_bactive = new_bactive & ~jnp.any(b_hits_e)

        # Magic wand: kills all enemies
        use_wand = (action == jnp.int32(6)) & (state.wand_uses > jnp.int32(0))
        new_enemy_active2 = jnp.where(
            use_wand, jnp.zeros(_N_ENEMIES, dtype=jnp.bool_), new_enemy_active
        )
        step_reward = step_reward + jnp.where(
            use_wand,
            jnp.sum(new_enemy_active).astype(jnp.float32) * jnp.float32(100.0),
            jnp.float32(0.0),
        )
        new_wand = state.wand_uses - jnp.where(use_wand, jnp.int32(1), jnp.int32(0))

        # Treasure pickup
        treas_hit = (
            state.treasure_active
            & (jnp.abs(state.treasure_x - new_px) < jnp.float32(10.0))
            & (jnp.abs(state.treasure_y - new_py) < jnp.float32(10.0))
        )
        step_reward = step_reward + jnp.sum(treas_hit).astype(
            jnp.float32
        ) * jnp.float32(50.0)
        new_treas_active = state.treasure_active & ~treas_hit

        # Enemy touches player
        enemy_hits_player = (
            new_enemy_active2
            & (jnp.abs(new_ex - new_px) < jnp.float32(8.0))
            & (jnp.abs(state.enemy_y - new_py) < jnp.float32(8.0))
        )
        hit_by_enemy = jnp.any(enemy_hits_player)

        # Room clear
        room_clear = ~jnp.any(new_treas_active)
        step_reward = step_reward + jnp.where(
            room_clear, jnp.float32(200.0), jnp.float32(0.0)
        )
        new_room = state.room + jnp.where(room_clear, jnp.int32(1), jnp.int32(0))

        life_lost = hit_by_enemy
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        done = new_lives <= jnp.int32(0)

        return TutankhamState(
            player_x=new_px,
            player_y=new_py,
            player_dir=new_dir,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            bullet_dir=new_bdir,
            enemy_x=new_ex,
            enemy_y=state.enemy_y,
            enemy_dx=new_edx,
            enemy_active=new_enemy_active2,
            treasure_x=state.treasure_x,
            treasure_y=state.treasure_y,
            treasure_active=new_treas_active,
            wand_uses=new_wand,
            room=new_room,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: TutankhamState, action: jax.Array) -> TutankhamState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : TutankhamState
            Current game state.
        action : jax.Array
            int32 — Action index (0–8).

        Returns
        -------
        new_state : TutankhamState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: TutankhamState) -> TutankhamState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: TutankhamState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : TutankhamState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Floor
        floor_mask = (
            (_ROW_IDX >= _ROOM_Y0)
            & (_ROW_IDX < _ROOM_Y0 + _ROOM_H)
            & (_COL_IDX >= _ROOM_X0)
            & (_COL_IDX < _ROOM_X0 + _ROOM_W)
        )
        frame = jnp.where(floor_mask[:, :, None], _COLOR_FLOOR, frame)

        # Treasures
        def draw_treasure(frm, i):
            tx = state.treasure_x[i].astype(jnp.int32)
            ty = state.treasure_y[i].astype(jnp.int32)
            mask = (
                state.treasure_active[i]
                & (_ROW_IDX >= ty - 4)
                & (_ROW_IDX < ty + 4)
                & (_COL_IDX >= tx - 4)
                & (_COL_IDX < tx + 4)
            )
            return jnp.where(mask[:, :, None], _COLOR_TREASURE, frm), None

        frame, _ = jax.lax.scan(draw_treasure, frame, jnp.arange(_N_TREASURES))

        # Enemies
        def draw_enemy(frm, i):
            ex = state.enemy_x[i].astype(jnp.int32)
            ey = state.enemy_y[i].astype(jnp.int32)
            mask = (
                state.enemy_active[i]
                & (_ROW_IDX >= ey - 5)
                & (_ROW_IDX < ey + 5)
                & (_COL_IDX >= ex - 5)
                & (_COL_IDX < ex + 5)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Bullet
        bm = (
            state.bullet_active
            & (_ROW_IDX >= state.bullet_y.astype(jnp.int32) - 2)
            & (_ROW_IDX < state.bullet_y.astype(jnp.int32) + 2)
            & (_COL_IDX >= state.bullet_x.astype(jnp.int32) - 2)
            & (_COL_IDX < state.bullet_x.astype(jnp.int32) + 2)
        )
        frame = jnp.where(bm[:, :, None], _COLOR_BULLET, frame)

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
            Mapping of pygame key constants to Tutankham action indices.
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
            pygame.K_LSHIFT: 6,
        }
