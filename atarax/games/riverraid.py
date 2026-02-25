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

"""River Raid — JAX-native game implementation.

Fly a jet down an endless river; shoot tanks, helicopters, and ships
while managing fuel.  Fly over fuel depots to refuel.

Action space (6 actions):
    0 — NOOP
    1 — FIRE
    2 — RIGHT
    3 — LEFT
    4 — UP   (increase speed)
    5 — DOWN (decrease speed)

Scoring:
    Tanker  — +30
    Helicopter — +60
    Jet     — +100
    Ship    — +80
    Fuel depot — +0 (refuels)
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------
_SCREEN_W: int = 160
_SCREEN_H: int = 210

_RIVER_LEFT: int = 20  # river left bank x
_RIVER_RIGHT: int = 140  # river right bank x
_RIVER_W: int = _RIVER_RIGHT - _RIVER_LEFT

_PLAYER_Y: int = 170  # player jet y (fixed near bottom)
_PLAYER_W: int = 8
_PLAYER_H: int = 8
_PLAYER_SPEED_MIN: float = 1.0
_PLAYER_SPEED_MAX: float = 3.0

_BULLET_SPEED: float = 6.0
_BULLET_W: int = 2
_BULLET_H: int = 6

_N_ENEMIES: int = 8
_N_FUEL: int = 4

_ENEMY_H: int = 8
_ENEMY_W: int = 10
_FUEL_W: int = 14
_FUEL_H: int = 8

_SCROLL_BASE: float = 1.5

_ROW_IDX = jnp.arange(_SCREEN_H)[:, None]
_COL_IDX = jnp.arange(_SCREEN_W)[None, :]

_COLOR_BG = jnp.array([100, 150, 80], dtype=jnp.uint8)  # banks
_COLOR_WATER = jnp.array([40, 80, 200], dtype=jnp.uint8)
_COLOR_PLAYER = jnp.array([255, 200, 0], dtype=jnp.uint8)
_COLOR_ENEMY = jnp.array([200, 50, 50], dtype=jnp.uint8)
_COLOR_FUEL = jnp.array([80, 200, 80], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)


@chex.dataclass
class RiverRaidState(AtariState):
    """
    Complete River Raid game state — a JAX pytree.

    Parameters
    ----------
    player_x : jax.Array
        float32 — Player jet x (left edge).
    player_speed : jax.Array
        float32 — Scroll speed (also player forward speed).
    bullet_x : jax.Array
        float32 — Bullet x.
    bullet_y : jax.Array
        float32 — Bullet y.
    bullet_active : jax.Array
        bool — Bullet in flight.
    enemy_x : jax.Array
        float32[8] — Enemy x positions.
    enemy_y : jax.Array
        float32[8] — Enemy y positions (scroll with river).
    enemy_type : jax.Array
        int32[8] — Enemy type (0=tanker, 1=helicopter, 2=jet, 3=ship).
    enemy_active : jax.Array
        bool[8] — Enemy alive.
    fuel_x : jax.Array
        float32[4] — Fuel depot x positions.
    fuel_y : jax.Array
        float32[4] — Fuel depot y positions.
    fuel_active : jax.Array
        bool[4] — Fuel depot available.
    fuel_level : jax.Array
        int32 — Remaining fuel (0 → life lost).
    scroll_y : jax.Array
        float32 — Global scroll offset (for generating enemies).
    spawn_timer : jax.Array
        int32 — Frames until next enemy spawn.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    player_x: jax.Array
    player_speed: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    enemy_x: jax.Array
    enemy_y: jax.Array
    enemy_type: jax.Array
    enemy_active: jax.Array
    fuel_x: jax.Array
    fuel_y: jax.Array
    fuel_active: jax.Array
    fuel_level: jax.Array
    scroll_y: jax.Array
    spawn_timer: jax.Array
    key: jax.Array


_ENEMY_SCORES = jnp.array([30, 60, 100, 80], dtype=jnp.int32)


class RiverRaid(AtariEnv):
    """
    River Raid implemented as a pure JAX function suite.

    Fly over the river, shoot enemies, collect fuel.  Lives: 3.
    """

    num_actions: int = 6

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> RiverRaidState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : RiverRaidState
            Player centred, full fuel, no enemies active.
        """
        return RiverRaidState(
            player_x=jnp.float32(76.0),
            player_speed=jnp.float32(2.0),
            bullet_x=jnp.float32(80.0),
            bullet_y=jnp.float32(float(_PLAYER_Y)),
            bullet_active=jnp.bool_(False),
            enemy_x=jnp.zeros(_N_ENEMIES, dtype=jnp.float32),
            enemy_y=jnp.linspace(20, 140, _N_ENEMIES, dtype=jnp.float32),
            enemy_type=jnp.zeros(_N_ENEMIES, dtype=jnp.int32),
            enemy_active=jnp.zeros(_N_ENEMIES, dtype=jnp.bool_),
            fuel_x=jnp.array([30.0, 60.0, 90.0, 120.0], dtype=jnp.float32),
            fuel_y=jnp.array([50.0, 80.0, 110.0, 140.0], dtype=jnp.float32),
            fuel_active=jnp.ones(_N_FUEL, dtype=jnp.bool_),
            fuel_level=jnp.int32(2000),
            scroll_y=jnp.float32(0.0),
            spawn_timer=jnp.int32(60),
            lives=jnp.int32(3),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: RiverRaidState, action: jax.Array) -> RiverRaidState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : RiverRaidState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : RiverRaidState
            State after one emulated frame.
        """
        key, k_spawn, k_type, k_x = jax.random.split(state.key, 4)
        step_reward = jnp.float32(0.0)

        # Speed control
        speed_up = action == jnp.int32(4)
        speed_dn = action == jnp.int32(5)
        new_speed = jnp.clip(
            state.player_speed
            + jnp.where(speed_up, jnp.float32(0.2), jnp.float32(0.0))
            + jnp.where(speed_dn, jnp.float32(-0.2), jnp.float32(0.0)),
            _PLAYER_SPEED_MIN,
            _PLAYER_SPEED_MAX,
        )

        # Player horizontal movement
        move_r = action == jnp.int32(2)
        move_l = action == jnp.int32(3)
        new_px = jnp.clip(
            state.player_x
            + jnp.where(move_r, jnp.float32(2.0), jnp.float32(0.0))
            + jnp.where(move_l, jnp.float32(-2.0), jnp.float32(0.0)),
            jnp.float32(_RIVER_LEFT),
            jnp.float32(_RIVER_RIGHT - _PLAYER_W),
        )

        # Crash into banks
        bank_crash = (new_px <= jnp.float32(_RIVER_LEFT)) | (
            new_px >= jnp.float32(_RIVER_RIGHT - _PLAYER_W)
        )

        # Fire bullet
        fire = (action == jnp.int32(1)) & ~state.bullet_active
        new_bx = jnp.where(fire, new_px + jnp.float32(_PLAYER_W // 2), state.bullet_x)
        new_by = jnp.where(
            fire, jnp.float32(float(_PLAYER_Y - _BULLET_H)), state.bullet_y
        )
        new_bactive = state.bullet_active | fire
        new_by = jnp.where(new_bactive, new_by - _BULLET_SPEED, new_by)
        new_bactive = new_bactive & (new_by >= jnp.float32(0.0))

        # Scroll enemies and fuel down
        scroll = new_speed
        new_ey = state.enemy_y + scroll
        new_fy = state.fuel_y + scroll

        # Despawn enemies/fuel that scroll off bottom
        new_enemy_active = state.enemy_active & (
            new_ey < jnp.float32(_PLAYER_Y - _ENEMY_H)
        )
        new_fuel_active = state.fuel_active & (new_fy < jnp.float32(_PLAYER_Y))

        # Bullet hits enemies
        bullet_hit_e = (
            new_bactive
            & new_enemy_active
            & (jnp.abs(new_bx - state.enemy_x) < jnp.float32(_ENEMY_W))
            & (jnp.abs(new_by - new_ey) < jnp.float32(_ENEMY_H))
        )
        scores_added = jnp.where(
            bullet_hit_e,
            _ENEMY_SCORES[state.enemy_type],
            jnp.zeros(_N_ENEMIES, dtype=jnp.int32),
        )
        step_reward = step_reward + jnp.sum(scores_added).astype(jnp.float32)
        new_enemy_active = new_enemy_active & ~bullet_hit_e
        bullet_hit_any = jnp.any(bullet_hit_e)
        new_bactive = new_bactive & ~bullet_hit_any

        # Player collects fuel
        player_cx = new_px + jnp.float32(_PLAYER_W // 2)
        fuel_hit = (
            new_fuel_active
            & (jnp.abs(player_cx - state.fuel_x) < jnp.float32(_FUEL_W // 2))
            & (
                jnp.abs(jnp.float32(_PLAYER_Y) - new_fy)
                < jnp.float32(_FUEL_H + _PLAYER_H)
            )
        )
        new_fuel_active = new_fuel_active & ~fuel_hit
        refueled = jnp.any(fuel_hit)
        new_fuel = jnp.where(refueled, jnp.int32(2000), state.fuel_level - jnp.int32(1))

        # Player hits enemy
        enemy_hit_player = (
            new_enemy_active
            & (
                jnp.abs(player_cx - state.enemy_x)
                < jnp.float32(_ENEMY_W // 2 + _PLAYER_W // 2)
            )
            & (
                jnp.abs(jnp.float32(_PLAYER_Y) - new_ey)
                < jnp.float32(_ENEMY_H + _PLAYER_H)
            )
        )
        hit_by_enemy = jnp.any(enemy_hit_player)

        # Spawn new enemy
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        can_spawn = new_spawn_timer <= jnp.int32(0)
        free_slot = jnp.argmin(
            new_enemy_active.astype(jnp.int32)
            + jnp.arange(_N_ENEMIES, dtype=jnp.int32) * 1000
        )
        spawn_x = jax.random.uniform(
            k_x,
            minval=float(_RIVER_LEFT + 5),
            maxval=float(_RIVER_RIGHT - _ENEMY_W - 5),
        )
        spawn_type = jax.random.randint(k_type, shape=(), minval=0, maxval=4)
        new_ex = jnp.where(
            can_spawn,
            state.enemy_x.at[free_slot].set(spawn_x),
            state.enemy_x,
        )
        new_ey2 = jnp.where(
            can_spawn,
            new_ey.at[free_slot].set(jnp.float32(0.0)),
            new_ey,
        )
        new_etype = jnp.where(
            can_spawn,
            state.enemy_type.at[free_slot].set(spawn_type),
            state.enemy_type,
        )
        new_enemy_active2 = jnp.where(
            can_spawn,
            new_enemy_active.at[free_slot].set(True),
            new_enemy_active,
        )
        new_spawn_timer = jnp.where(can_spawn, jnp.int32(40), new_spawn_timer)

        # Also respawn fuel
        fuel_free = jnp.argmin(new_fuel_active.astype(jnp.int32))
        new_fx = jnp.where(
            can_spawn & ~jnp.all(new_fuel_active),
            state.fuel_x.at[fuel_free].set(
                jax.random.uniform(
                    k_x,
                    minval=float(_RIVER_LEFT + 5),
                    maxval=float(_RIVER_RIGHT - _FUEL_W - 5),
                )
            ),
            state.fuel_x,
        )
        new_fy2 = jnp.where(
            can_spawn & ~jnp.all(new_fuel_active),
            new_fy.at[fuel_free].set(jnp.float32(0.0)),
            new_fy,
        )
        new_fuel_active2 = jnp.where(
            can_spawn & ~jnp.all(new_fuel_active),
            new_fuel_active.at[fuel_free].set(True),
            new_fuel_active,
        )

        # Life loss
        fuel_empty = new_fuel <= jnp.int32(0)
        life_lost = hit_by_enemy | fuel_empty | bank_crash
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        new_fuel = jnp.where(fuel_empty, jnp.int32(2000), new_fuel)

        done = new_lives <= jnp.int32(0)

        return RiverRaidState(
            player_x=new_px,
            player_speed=new_speed,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            enemy_x=new_ex,
            enemy_y=new_ey2,
            enemy_type=new_etype,
            enemy_active=new_enemy_active2,
            fuel_x=new_fx,
            fuel_y=new_fy2,
            fuel_active=new_fuel_active2,
            fuel_level=new_fuel,
            scroll_y=state.scroll_y + scroll,
            spawn_timer=new_spawn_timer,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: RiverRaidState, action: jax.Array) -> RiverRaidState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : RiverRaidState
            Current game state.
        action : jax.Array
            int32 — Action index (0–5).

        Returns
        -------
        new_state : RiverRaidState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: RiverRaidState) -> RiverRaidState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: RiverRaidState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : RiverRaidState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # River channel
        river_mask = (_COL_IDX >= _RIVER_LEFT) & (_COL_IDX < _RIVER_RIGHT)
        frame = jnp.where(river_mask[:, :, None], _COLOR_WATER, frame)

        # Fuel depots
        def draw_fuel(frm, i):
            fx = state.fuel_x[i].astype(jnp.int32)
            fy = state.fuel_y[i].astype(jnp.int32)
            mask = (
                state.fuel_active[i]
                & (_ROW_IDX >= fy)
                & (_ROW_IDX < fy + _FUEL_H)
                & (_COL_IDX >= fx)
                & (_COL_IDX < fx + _FUEL_W)
            )
            return jnp.where(mask[:, :, None], _COLOR_FUEL, frm), None

        frame, _ = jax.lax.scan(draw_fuel, frame, jnp.arange(_N_FUEL))

        # Enemies
        def draw_enemy(frm, i):
            ex = state.enemy_x[i].astype(jnp.int32)
            ey = state.enemy_y[i].astype(jnp.int32)
            mask = (
                state.enemy_active[i]
                & (_ROW_IDX >= ey)
                & (_ROW_IDX < ey + _ENEMY_H)
                & (_COL_IDX >= ex)
                & (_COL_IDX < ex + _ENEMY_W)
            )
            return jnp.where(mask[:, :, None], _COLOR_ENEMY, frm), None

        frame, _ = jax.lax.scan(draw_enemy, frame, jnp.arange(_N_ENEMIES))

        # Bullet
        bm = (
            state.bullet_active
            & (_ROW_IDX >= state.bullet_y.astype(jnp.int32))
            & (_ROW_IDX < state.bullet_y.astype(jnp.int32) + _BULLET_H)
            & (_COL_IDX >= state.bullet_x.astype(jnp.int32))
            & (_COL_IDX < state.bullet_x.astype(jnp.int32) + _BULLET_W)
        )
        frame = jnp.where(bm[:, :, None], _COLOR_BULLET, frame)

        # Player
        px = state.player_x.astype(jnp.int32)
        pm = (
            (_ROW_IDX >= _PLAYER_Y)
            & (_ROW_IDX < _PLAYER_Y + _PLAYER_H)
            & (_COL_IDX >= px)
            & (_COL_IDX < px + _PLAYER_W)
        )
        frame = jnp.where(pm[:, :, None], _COLOR_PLAYER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to River Raid action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_RIGHT: 2,
            pygame.K_d: 2,
            pygame.K_LEFT: 3,
            pygame.K_a: 3,
            pygame.K_UP: 4,
            pygame.K_w: 4,
            pygame.K_DOWN: 5,
            pygame.K_s: 5,
        }
