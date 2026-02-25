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

"""Chopper Command — JAX-native game implementation.

Pilot a helicopter defending a convoy against enemy planes and tanks.
Enemies approach from both sides; shoot them before they reach the trucks.

Action space (7 actions):
    0 — NOOP
    1 — FIRE
    2 — UP
    3 — RIGHT
    4 — DOWN
    5 — LEFT
    6 — UP + FIRE

Scoring:
    Enemy jet shot   — +100
    Enemy tank shot  — +200
    Convoy truck destroyed — -100 (penalty)
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
_N_JETS: int = 6
_N_TANKS: int = 4
_N_BULLETS: int = 4
_N_TRUCKS: int = 5

_SCREEN_W: int = 160
_SCREEN_H: int = 210

_CHOPPER_SPEED: float = 2.0
_JET_SPEED: float = 1.5
_TANK_SPEED: float = 0.8
_BULLET_SPEED: float = 5.0

_CHOPPER_Y_MIN: float = 30.0
_CHOPPER_Y_MAX: float = 160.0
_CHOPPER_X_MIN: float = 8.0
_CHOPPER_X_MAX: float = 152.0

_GROUND_Y: int = 170
_SKY_Y: int = 50  # jets fly in sky
_TRUCK_Y: int = 165  # trucks at ground level

_INIT_LIVES: int = 3
_SPAWN_INTERVAL: int = 30
_CONVOY_SPACING: int = 25

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([80, 160, 80], dtype=jnp.uint8)  # sky / ground
_COLOR_GROUND = jnp.array([100, 60, 20], dtype=jnp.uint8)
_COLOR_CHOPPER = jnp.array([200, 200, 200], dtype=jnp.uint8)
_COLOR_JET = jnp.array([220, 60, 60], dtype=jnp.uint8)
_COLOR_TANK = jnp.array([80, 80, 40], dtype=jnp.uint8)
_COLOR_TRUCK = jnp.array([180, 140, 40], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 80], dtype=jnp.uint8)


@chex.dataclass
class ChopperCommandState(AtariState):
    """
    Complete Chopper Command game state — a JAX pytree.

    Parameters
    ----------
    chopper_x : jax.Array
        float32 — Helicopter x position.
    chopper_y : jax.Array
        float32 — Helicopter y position.
    bullet_x : jax.Array
        float32[4] — Bullet x positions.
    bullet_y : jax.Array
        float32[4] — Bullet y positions.
    bullet_dir : jax.Array
        int32[4] — Bullet direction (+1=right, -1=left).
    bullet_active : jax.Array
        bool[4] — Bullet in-flight.
    jet_x : jax.Array
        float32[6] — Jet x positions.
    jet_dir : jax.Array
        int32[6] — Jet direction (+1=right, -1=left).
    jet_active : jax.Array
        bool[6] — Jet alive.
    tank_x : jax.Array
        float32[4] — Tank x positions.
    tank_dir : jax.Array
        int32[4] — Tank direction.
    tank_active : jax.Array
        bool[4] — Tank alive.
    truck_x : jax.Array
        float32[5] — Convoy truck x positions.
    truck_alive : jax.Array
        bool[5] — Truck intact.
    spawn_timer : jax.Array
        int32 — Frames until next enemy spawn.
    wave : jax.Array
        int32 — Current wave.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    chopper_x: jax.Array
    chopper_y: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_dir: jax.Array
    bullet_active: jax.Array
    jet_x: jax.Array
    jet_dir: jax.Array
    jet_active: jax.Array
    tank_x: jax.Array
    tank_dir: jax.Array
    tank_active: jax.Array
    truck_x: jax.Array
    truck_alive: jax.Array
    spawn_timer: jax.Array
    wave: jax.Array
    key: jax.Array


class ChopperCommand(AtariEnv):
    """
    Chopper Command implemented as a pure JAX function suite.

    Defend the convoy by shooting enemies.  Lives: 3.
    """

    num_actions: int = 7

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=100_000)

    def _reset(self, key: jax.Array) -> ChopperCommandState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : ChopperCommandState
            Chopper at centre, convoy intact, 3 lives.
        """
        truck_x = jnp.array(
            [10 + i * _CONVOY_SPACING for i in range(_N_TRUCKS)], dtype=jnp.float32
        )
        return ChopperCommandState(
            chopper_x=jnp.float32(80.0),
            chopper_y=jnp.float32(100.0),
            bullet_x=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_y=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_dir=jnp.ones(_N_BULLETS, dtype=jnp.int32),
            bullet_active=jnp.zeros(_N_BULLETS, dtype=jnp.bool_),
            jet_x=jnp.full(_N_JETS, -20.0, dtype=jnp.float32),
            jet_dir=jnp.ones(_N_JETS, dtype=jnp.int32),
            jet_active=jnp.zeros(_N_JETS, dtype=jnp.bool_),
            tank_x=jnp.full(_N_TANKS, -20.0, dtype=jnp.float32),
            tank_dir=jnp.ones(_N_TANKS, dtype=jnp.int32),
            tank_active=jnp.zeros(_N_TANKS, dtype=jnp.bool_),
            truck_x=truck_x,
            truck_alive=jnp.ones(_N_TRUCKS, dtype=jnp.bool_),
            spawn_timer=jnp.int32(_SPAWN_INTERVAL),
            wave=jnp.int32(0),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(
        self, state: ChopperCommandState, action: jax.Array
    ) -> ChopperCommandState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : ChopperCommandState
            Current game state.
        action : jax.Array
            int32 — Action index (0–6).

        Returns
        -------
        new_state : ChopperCommandState
            State after one emulated frame.
        """
        key, sk1, sk2 = jax.random.split(state.key, 3)
        step_reward = jnp.float32(0.0)

        # Chopper movement
        dx = jnp.where(action == jnp.int32(3), _CHOPPER_SPEED, 0.0) + jnp.where(
            action == jnp.int32(5), -_CHOPPER_SPEED, 0.0
        )
        dy = jnp.where(
            (action == jnp.int32(2)) | (action == jnp.int32(6)), -_CHOPPER_SPEED, 0.0
        ) + jnp.where(action == jnp.int32(4), _CHOPPER_SPEED, 0.0)
        new_cx = jnp.clip(state.chopper_x + dx, _CHOPPER_X_MIN, _CHOPPER_X_MAX)
        new_cy = jnp.clip(state.chopper_y + dy, _CHOPPER_Y_MIN, _CHOPPER_Y_MAX)

        # Fire bullet (horizontal, toward whichever direction chopper is facing)
        do_fire = (action == jnp.int32(1)) | (action == jnp.int32(6))
        fire_dir = jnp.where(dx >= 0.0, jnp.int32(1), jnp.int32(-1))
        free_slot = jnp.argmin(state.bullet_active.astype(jnp.int32))
        has_free = ~jnp.all(state.bullet_active)
        new_bx = jnp.where(
            do_fire & has_free,
            state.bullet_x.at[free_slot].set(new_cx),
            state.bullet_x,
        )
        new_by = jnp.where(
            do_fire & has_free,
            state.bullet_y.at[free_slot].set(new_cy),
            state.bullet_y,
        )
        new_bdir = jnp.where(
            do_fire & has_free,
            state.bullet_dir.at[free_slot].set(fire_dir),
            state.bullet_dir,
        )
        new_bactive = jnp.where(
            do_fire & has_free,
            state.bullet_active.at[free_slot].set(True),
            state.bullet_active,
        )

        # Move bullets
        new_bx = new_bx + new_bdir.astype(jnp.float32) * _BULLET_SPEED
        new_bactive = new_bactive & (new_bx >= 0.0) & (new_bx < _SCREEN_W)

        # Move jets
        new_jet_x = state.jet_x + state.jet_dir.astype(jnp.float32) * _JET_SPEED
        jet_out = (new_jet_x < -10.0) | (new_jet_x > _SCREEN_W + 10.0)
        new_jet_active = state.jet_active & ~jet_out

        # Move tanks
        new_tank_x = state.tank_x + state.tank_dir.astype(jnp.float32) * _TANK_SPEED
        tank_out = (new_tank_x < -10.0) | (new_tank_x > _SCREEN_W + 10.0)
        new_tank_active = state.tank_active & ~tank_out

        # Bullet–jet collision
        jet_y = jnp.float32(_SKY_Y)
        bul_hits_jet = (
            new_bactive[:, None]
            & new_jet_active[None, :]
            & (jnp.abs(new_bx[:, None] - new_jet_x[None, :]) < 8.0)
            & (jnp.abs(new_by[:, None] - jet_y) < 8.0)
        )
        jet_killed = jnp.any(bul_hits_jet, axis=0)
        bul_used_jet = jnp.any(bul_hits_jet, axis=1)
        n_jets_killed = jnp.sum(jet_killed).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_jets_killed * 100)
        new_jet_active = new_jet_active & ~jet_killed
        new_bactive = new_bactive & ~bul_used_jet

        # Bullet–tank collision
        tank_y = jnp.float32(_GROUND_Y - 10)
        bul_hits_tank = (
            new_bactive[:, None]
            & new_tank_active[None, :]
            & (jnp.abs(new_bx[:, None] - new_tank_x[None, :]) < 8.0)
            & (jnp.abs(new_by[:, None] - tank_y) < 8.0)
        )
        tank_killed = jnp.any(bul_hits_tank, axis=0)
        bul_used_tank = jnp.any(bul_hits_tank, axis=1)
        n_tanks_killed = jnp.sum(tank_killed).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_tanks_killed * 200)
        new_tank_active = new_tank_active & ~tank_killed
        new_bactive = new_bactive & ~bul_used_tank

        # Spawn enemies
        new_spawn_timer = state.spawn_timer - jnp.int32(1)
        do_spawn = new_spawn_timer <= jnp.int32(0)
        new_spawn_timer = jnp.where(
            do_spawn, jnp.int32(_SPAWN_INTERVAL), new_spawn_timer
        )
        spawn_side = jax.random.randint(sk1, (), 0, 2)  # 0=left, 1=right
        spawn_x = jnp.where(
            spawn_side == 0, jnp.float32(-5.0), jnp.float32(_SCREEN_W + 5.0)
        )
        spawn_dir = jnp.where(spawn_side == 0, jnp.int32(1), jnp.int32(-1))
        spawn_type = jax.random.randint(sk2, (), 0, 2)  # 0=jet, 1=tank

        free_jet = jnp.argmin(new_jet_active.astype(jnp.int32))
        new_jet_x = jnp.where(
            do_spawn & (spawn_type == 0),
            new_jet_x.at[free_jet].set(spawn_x),
            new_jet_x,
        )
        new_jet_dir = jnp.where(
            do_spawn & (spawn_type == 0),
            state.jet_dir.at[free_jet].set(spawn_dir),
            state.jet_dir,
        )
        new_jet_active = jnp.where(
            do_spawn & (spawn_type == 0),
            new_jet_active.at[free_jet].set(True),
            new_jet_active,
        )
        free_tank = jnp.argmin(new_tank_active.astype(jnp.int32))
        new_tank_x = jnp.where(
            do_spawn & (spawn_type == 1),
            new_tank_x.at[free_tank].set(spawn_x),
            new_tank_x,
        )
        new_tank_dir = jnp.where(
            do_spawn & (spawn_type == 1),
            state.tank_dir.at[free_tank].set(spawn_dir),
            state.tank_dir,
        )
        new_tank_active = jnp.where(
            do_spawn & (spawn_type == 1),
            new_tank_active.at[free_tank].set(True),
            new_tank_active,
        )

        # Chopper hit by jet
        jet_hits_chopper = (
            new_jet_active
            & (jnp.abs(new_jet_x - new_cx) < 10.0)
            & (jnp.abs(jnp.float32(_SKY_Y) - new_cy) < 10.0)
        )
        hit = jnp.any(jet_hits_chopper)
        new_lives = state.lives - jnp.where(hit, jnp.int32(1), jnp.int32(0))

        done = new_lives <= jnp.int32(0)

        return ChopperCommandState(
            chopper_x=new_cx,
            chopper_y=new_cy,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_dir=new_bdir,
            bullet_active=new_bactive,
            jet_x=new_jet_x,
            jet_dir=new_jet_dir,
            jet_active=new_jet_active,
            tank_x=new_tank_x,
            tank_dir=new_tank_dir,
            tank_active=new_tank_active,
            truck_x=state.truck_x,
            truck_alive=state.truck_alive,
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

    def _step(
        self, state: ChopperCommandState, action: jax.Array
    ) -> ChopperCommandState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : ChopperCommandState
            Current game state.
        action : jax.Array
            int32 — Action index (0–6).

        Returns
        -------
        new_state : ChopperCommandState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: ChopperCommandState) -> ChopperCommandState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: ChopperCommandState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : ChopperCommandState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Ground strip
        ground_mask = _ROW_IDX >= _GROUND_Y
        frame = jnp.where(ground_mask[:, :, None], _COLOR_GROUND, frame)

        # Trucks
        def draw_truck(frm, i):
            tx = state.truck_x[i].astype(jnp.int32)
            mask = (
                state.truck_alive[i]
                & (_ROW_IDX >= _TRUCK_Y - 5)
                & (_ROW_IDX <= _TRUCK_Y + 5)
                & (_COL_IDX >= tx)
                & (_COL_IDX < tx + 14)
            )
            return jnp.where(mask[:, :, None], _COLOR_TRUCK, frm), None

        frame, _ = jax.lax.scan(draw_truck, frame, jnp.arange(_N_TRUCKS))

        # Jets
        def draw_jet(frm, i):
            jx = state.jet_x[i].astype(jnp.int32)
            mask = (
                state.jet_active[i]
                & (_ROW_IDX >= _SKY_Y - 4)
                & (_ROW_IDX <= _SKY_Y + 4)
                & (_COL_IDX >= jx - 6)
                & (_COL_IDX <= jx + 6)
            )
            return jnp.where(mask[:, :, None], _COLOR_JET, frm), None

        frame, _ = jax.lax.scan(draw_jet, frame, jnp.arange(_N_JETS))

        # Tanks
        def draw_tank(frm, i):
            tx = state.tank_x[i].astype(jnp.int32)
            mask = (
                state.tank_active[i]
                & (_ROW_IDX >= _GROUND_Y - 10)
                & (_ROW_IDX <= _GROUND_Y)
                & (_COL_IDX >= tx)
                & (_COL_IDX < tx + 12)
            )
            return jnp.where(mask[:, :, None], _COLOR_TANK, frm), None

        frame, _ = jax.lax.scan(draw_tank, frame, jnp.arange(_N_TANKS))

        # Bullets
        def draw_bullet(frm, i):
            bx = state.bullet_x[i].astype(jnp.int32)
            by = state.bullet_y[i].astype(jnp.int32)
            mask = (
                state.bullet_active[i]
                & (_ROW_IDX >= by - 2)
                & (_ROW_IDX <= by + 2)
                & (_COL_IDX >= bx - 2)
                & (_COL_IDX <= bx + 2)
            )
            return jnp.where(mask[:, :, None], _COLOR_BULLET, frm), None

        frame, _ = jax.lax.scan(draw_bullet, frame, jnp.arange(_N_BULLETS))

        # Chopper
        cx = state.chopper_x.astype(jnp.int32)
        cy = state.chopper_y.astype(jnp.int32)
        chopper_mask = (
            (_ROW_IDX >= cy - 5)
            & (_ROW_IDX <= cy + 5)
            & (_COL_IDX >= cx - 8)
            & (_COL_IDX <= cx + 8)
        )
        frame = jnp.where(chopper_mask[:, :, None], _COLOR_CHOPPER, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Chopper Command action indices.
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
