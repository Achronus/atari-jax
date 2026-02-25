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

"""Montezuma's Revenge — JAX-native game implementation.

Panama Joe must navigate a complex Aztec pyramid collecting keys,
avoiding enemies, and solving puzzles to reach the treasure.

Action space (18 actions, minimal set):
    0 — NOOP
    1 — FIRE  (jump)
    2 — UP    (climb ladder up)
    3 — RIGHT
    4 — DOWN  (climb ladder down)
    5 — LEFT
    6 — UP + RIGHT
    7 — UP + LEFT

Scoring:
    Key collected   — +100
    Item collected  — +200
    Room cleared    — +300
    Episode ends when all lives are lost; lives: 5.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_N_ITEMS: int = 6  # keys and items per room
_N_ENEMIES: int = 3

_PLAYER_SPEED: float = 2.0
_JUMP_VEL: float = -5.5
_GRAVITY: float = 0.35
_ENEMY_SPEED: float = 1.2

_GROUND_Y: int = 175
_LADDER_Y0: int = 80
_LADDER_Y1: int = 175
_LADDER_X = jnp.array([40, 80, 120], dtype=jnp.int32)

_INIT_LIVES: int = 5

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_PLATFORM = jnp.array([100, 60, 20], dtype=jnp.uint8)
_COLOR_LADDER = jnp.array([120, 80, 30], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([220, 100, 40], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([80, 80, 200], dtype=jnp.uint8)
_COLOR_KEY = jnp.array([255, 215, 0], dtype=jnp.uint8)
_COLOR_ITEM = jnp.array([200, 100, 200], dtype=jnp.uint8)


@chex.dataclass
class MontezumaRevengeState(AtariState):
    """
    Complete Montezuma's Revenge game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player x.
    player_y : jax.Array
        float32 — Player y.
    player_vy : jax.Array
        float32 — Vertical velocity.
    on_ladder : jax.Array
        bool — Player is on a ladder.
    room : jax.Array
        int32 — Current room (simplified to single room cycling).
    item_x : jax.Array
        float32[6] — Item x positions.
    item_y : jax.Array
        float32[6] — Item y positions.
    item_active : jax.Array
        bool[6] — Items not yet collected.
    item_is_key : jax.Array
        bool[6] — Item is a key (True) or other item (False).
    enemy_x : jax.Array
        float32[3] — Enemy x.
    enemy_dir : jax.Array
        int32[3] — Enemy direction.
    enemy_active : jax.Array
        bool[3] — Enemy alive.
    keys_held : jax.Array
        int32 — Keys currently carried.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_y: jax.Array
    player_vy: jax.Array
    on_ladder: jax.Array
    room: jax.Array
    item_x: jax.Array
    item_y: jax.Array
    item_active: jax.Array
    item_is_key: jax.Array
    enemy_x: jax.Array
    enemy_dir: jax.Array
    enemy_active: jax.Array
    keys_held: jax.Array
    key: jax.Array


_PLATFORM_Y = jnp.array([80, 130, 175], dtype=jnp.int32)


class MontezumaRevenge(AtariEnv):
    """
    Montezuma's Revenge implemented as a pure JAX function suite.

    Collect keys and items; avoid skull enemies.  Lives: 5.
    """

    num_actions: int = 8

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> MontezumaRevengeState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : MontezumaRevengeState
            Player at start, 6 items, 3 enemies, 5 lives.
        """
        return MontezumaRevengeState(
            player_x=jnp.float32(20.0),
            player_y=jnp.float32(float(_GROUND_Y) - 14.0),
            player_vy=jnp.float32(0.0),
            on_ladder=jnp.bool_(False),
            room=jnp.int32(0),
            item_x=jnp.array([40.0, 80.0, 120.0, 30.0, 90.0, 130.0], dtype=jnp.float32),
            item_y=jnp.array(
                [70.0, 70.0, 70.0, 125.0, 125.0, 125.0], dtype=jnp.float32
            ),
            item_active=jnp.ones(_N_ITEMS, dtype=jnp.bool_),
            item_is_key=jnp.array(
                [True, False, True, False, True, False], dtype=jnp.bool_
            ),
            enemy_x=jnp.array([60.0, 100.0, 130.0], dtype=jnp.float32),
            enemy_dir=jnp.array([1, -1, 1], dtype=jnp.int32),
            enemy_active=jnp.ones(_N_ENEMIES, dtype=jnp.bool_),
            keys_held=jnp.int32(0),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(
        self, state: MontezumaRevengeState, action: jax.Array
    ) -> MontezumaRevengeState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : MontezumaRevengeState
            Current game state.
        action : jax.Array
            int32 — Action index (0–7).

        Returns
        -------
        new_state : MontezumaRevengeState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        # Horizontal movement
        move_r = (action == jnp.int32(3)) | (action == jnp.int32(6))
        move_l = (action == jnp.int32(5)) | (action == jnp.int32(7))
        new_px = jnp.clip(
            state.player_x
            + jnp.where(move_r, _PLAYER_SPEED, 0.0)
            + jnp.where(move_l, -_PLAYER_SPEED, 0.0),
            5.0,
            155.0,
        )

        # Ladder check
        near_ladder = jnp.any(jnp.abs(jnp.float32(_LADDER_X) - new_px) < 8.0)

        # Climb ladder
        climb_up = (action == jnp.int32(2)) & near_ladder
        climb_dn = (action == jnp.int32(4)) & near_ladder
        new_on_ladder = near_ladder & (climb_up | climb_dn | state.on_ladder)

        # Gravity and jumping
        on_ground = state.player_y >= jnp.float32(_GROUND_Y) - 16.0
        on_platform = jnp.any(
            (jnp.abs(state.player_y - (jnp.float32(_PLATFORM_Y) - 14.0)) < 4.0)
        )
        on_surface = on_ground | on_platform

        do_jump = (
            (
                (action == jnp.int32(1))
                | (action == jnp.int32(6))
                | (action == jnp.int32(7))
            )
            & on_surface
            & ~new_on_ladder
        )
        new_vy = jnp.where(
            do_jump,
            jnp.float32(_JUMP_VEL),
            jnp.where(new_on_ladder, jnp.float32(0.0), state.player_vy + _GRAVITY),
        )

        # Ladder movement
        new_vy = jnp.where(climb_up, jnp.float32(-_PLAYER_SPEED), new_vy)
        new_vy = jnp.where(climb_dn, jnp.float32(_PLAYER_SPEED), new_vy)

        new_py = state.player_y + new_vy

        # Ground collision
        ground_y = jnp.float32(_GROUND_Y) - 14.0
        landed = new_py >= ground_y
        new_py = jnp.where(landed, ground_y, new_py)
        new_vy = jnp.where(landed, jnp.float32(0.0), new_vy)

        # Platform collision
        for ply in [80, 130]:
            plat_y = jnp.float32(ply) - 14.0
            on_plat = (new_py >= plat_y) & (new_py <= plat_y + 4.0) & (new_vy > 0.0)
            new_py = jnp.where(on_plat, plat_y, new_py)
            new_vy = jnp.where(on_plat, jnp.float32(0.0), new_vy)

        # Item collection
        collects = (
            state.item_active
            & (jnp.abs(state.item_x - new_px) < 10.0)
            & (jnp.abs(state.item_y - new_py) < 10.0)
        )
        key_collects = collects & state.item_is_key
        item_collects = collects & ~state.item_is_key
        n_keys = jnp.sum(key_collects).astype(jnp.int32)
        n_items = jnp.sum(item_collects).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_keys * 100 + n_items * 200)
        new_item_active = state.item_active & ~collects
        new_keys_held = state.keys_held + n_keys

        # Enemy movement
        new_ex = state.enemy_x + state.enemy_dir.astype(jnp.float32) * _ENEMY_SPEED
        at_edge = (new_ex < 10.0) | (new_ex > 150.0)
        new_edir = jnp.where(at_edge, -state.enemy_dir, state.enemy_dir)
        new_ex = jnp.clip(new_ex, 10.0, 150.0)

        # Enemy catches player
        enemy_y = jnp.float32(_GROUND_Y) - 14.0
        enemy_hits = (
            state.enemy_active
            & (jnp.abs(new_ex - new_px) < 12.0)
            & (jnp.abs(enemy_y - new_py) < 12.0)
        )
        caught = jnp.any(enemy_hits)
        new_lives = state.lives - jnp.where(caught, jnp.int32(1), jnp.int32(0))
        new_px = jnp.where(caught, jnp.float32(20.0), new_px)
        new_py = jnp.where(caught, jnp.float32(float(_GROUND_Y) - 14.0), new_py)
        new_vy = jnp.where(caught, jnp.float32(0.0), new_vy)

        # Room clear: all items collected
        room_clear = ~jnp.any(new_item_active)
        step_reward = step_reward + jnp.where(
            room_clear, jnp.float32(300.0), jnp.float32(0.0)
        )
        new_room = state.room + jnp.where(room_clear, jnp.int32(1), jnp.int32(0))
        new_item_active = jnp.where(
            room_clear, jnp.ones(_N_ITEMS, dtype=jnp.bool_), new_item_active
        )

        done = new_lives <= jnp.int32(0)

        return MontezumaRevengeState(
            player_x=new_px,
            player_y=new_py,
            player_vy=new_vy,
            on_ladder=new_on_ladder,
            room=new_room,
            item_x=state.item_x,
            item_y=state.item_y,
            item_active=new_item_active,
            item_is_key=state.item_is_key,
            enemy_x=new_ex,
            enemy_dir=new_edir,
            enemy_active=state.enemy_active,
            keys_held=new_keys_held,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(
        self, state: MontezumaRevengeState, action: jax.Array
    ) -> MontezumaRevengeState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : MontezumaRevengeState
            Current game state.
        action : jax.Array
            int32 — Action index (0–7).

        Returns
        -------
        new_state : MontezumaRevengeState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: MontezumaRevengeState) -> MontezumaRevengeState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: MontezumaRevengeState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : MontezumaRevengeState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Platforms
        def draw_platform(frm, i):
            py = _PLATFORM_Y[i]
            mask = (
                (_ROW_IDX >= py - 3)
                & (_ROW_IDX <= py)
                & (_COL_IDX >= 10)
                & (_COL_IDX <= 150)
            )
            return jnp.where(mask[:, :, None], _COLOR_PLATFORM, frm), None

        frame, _ = jax.lax.scan(draw_platform, frame, jnp.arange(3))

        # Ladders
        def draw_ladder(frm, i):
            lx = _LADDER_X[i]
            mask = (
                (_COL_IDX >= lx - 3)
                & (_COL_IDX <= lx + 3)
                & (_ROW_IDX >= _LADDER_Y0)
                & (_ROW_IDX <= _LADDER_Y1)
            )
            return jnp.where(mask[:, :, None], _COLOR_LADDER, frm), None

        frame, _ = jax.lax.scan(draw_ladder, frame, jnp.arange(3))

        # Items
        def draw_item(frm, i):
            ix = state.item_x[i].astype(jnp.int32)
            iy = state.item_y[i].astype(jnp.int32)
            color = jnp.where(state.item_is_key[i], _COLOR_KEY, _COLOR_ITEM)
            mask = (
                state.item_active[i]
                & (_ROW_IDX >= iy - 4)
                & (_ROW_IDX <= iy + 4)
                & (_COL_IDX >= ix - 4)
                & (_COL_IDX <= ix + 4)
            )
            return jnp.where(mask[:, :, None], color, frm), None

        frame, _ = jax.lax.scan(draw_item, frame, jnp.arange(_N_ITEMS))

        # Enemies
        def draw_enemy(frm, i):
            ex = state.enemy_x[i].astype(jnp.int32)
            ey = _GROUND_Y - 14
            mask = (
                state.enemy_active[i]
                & (_ROW_IDX >= ey - 6)
                & (_ROW_IDX <= ey + 6)
                & (_COL_IDX >= ex - 6)
                & (_COL_IDX <= ex + 6)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Player
        px = state.player_x.astype(jnp.int32)
        py = state.player_y.astype(jnp.int32)
        player_mask = (
            (_ROW_IDX >= py - 8)
            & (_ROW_IDX <= py + 8)
            & (_COL_IDX >= px - 4)
            & (_COL_IDX <= px + 4)
        )
        frame = jnp.where(player_mask[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Montezuma's Revenge action indices.
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
