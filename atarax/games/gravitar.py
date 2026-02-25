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

"""Gravitar — JAX-native game implementation.

Navigate a gravity-filled star system, rescuing astronauts from planet
surfaces while fighting bunkers and fuel pods.

Action space (18 actions, minimal set):
    0 — NOOP
    1 — FIRE
    2 — THRUST
    3 — RIGHT (rotate clockwise)
    4 — LEFT  (rotate counter-clockwise)
    5 — SHIELD
    6 — THRUST + FIRE

Scoring:
    Bunker destroyed   — +250
    Fuel pod destroyed — +500 + fuel
    Astronaut rescued  — +1000
    Episode ends when all lives are lost; lives: 3.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.games._base import AtariState

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
_N_BUNKERS: int = 5
_N_ASTRONAUTS: int = 4
_N_BULLETS: int = 3

_THRUST: float = 0.12
_MAX_SPEED: float = 3.5
_GRAVITY: float = 0.04  # downward acceleration (y+)
_ROTATE_SPEED: float = 0.1
_BULLET_SPEED: float = 4.0
_FRICTION: float = 0.995
_FUEL_MAX: int = 700

_SCREEN_W: float = 160.0
_SCREEN_H: float = 175.0
_GROUND_Y: float = 180.0

_INIT_LIVES: int = 3

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_GROUND = jnp.array([60, 80, 40], dtype=jnp.uint8)
_COLOR_SHIP = jnp.array([200, 200, 255], dtype=jnp.uint8)
_COLOR_BUNKER = jnp.array([200, 60, 60], dtype=jnp.uint8)
_COLOR_ASTRONAUT = jnp.array([100, 220, 100], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 80], dtype=jnp.uint8)
_COLOR_FUEL = jnp.array([255, 160, 40], dtype=jnp.uint8)


@chex.dataclass
class GravitarState(AtariState):
    """
    Complete Gravitar game state — a JAX pytree.

    Parameters
    ----------
    ship_x : jax.Array
        float32 — Ship x.
    ship_y : jax.Array
        float32 — Ship y.
    ship_vx : jax.Array
        float32 — Ship x velocity.
    ship_vy : jax.Array
        float32 — Ship y velocity.
    ship_angle : jax.Array
        float32 — Heading (radians).
    fuel : jax.Array
        int32 — Remaining fuel.
    bullet_x : jax.Array
        float32[3] — Bullet x.
    bullet_y : jax.Array
        float32[3] — Bullet y.
    bullet_active : jax.Array
        bool[3] — Bullets in-flight.
    bullet_timer : jax.Array
        int32[3] — Expiry timers.
    bunker_x : jax.Array
        float32[5] — Bunker x positions.
    bunker_active : jax.Array
        bool[5] — Bunkers alive.
    astronaut_x : jax.Array
        float32[4] — Astronaut x.
    astronaut_rescued : jax.Array
        bool[4] — Astronauts rescued.
    shield_active : jax.Array
        bool — Shield on.
    wave : jax.Array
        int32 — Current wave.
    key : jax.Array
        uint32[2] — PRNG key.
    """

    ship_x: jax.Array
    ship_y: jax.Array
    ship_vx: jax.Array
    ship_vy: jax.Array
    ship_angle: jax.Array
    fuel: jax.Array
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    bullet_timer: jax.Array
    bunker_x: jax.Array
    bunker_active: jax.Array
    astronaut_x: jax.Array
    astronaut_rescued: jax.Array
    shield_active: jax.Array
    wave: jax.Array
    key: jax.Array


_BUNKER_X = jnp.array([25.0, 55.0, 80.0, 105.0, 135.0], dtype=jnp.float32)
_BUNKER_Y: float = 175.0
_ASTRONAUT_X = jnp.array([30.0, 65.0, 95.0, 130.0], dtype=jnp.float32)
_ASTRONAUT_Y: float = 170.0


class Gravitar(AtariEnv):
    """
    Gravitar implemented as a pure JAX function suite.

    Destroy bunkers, rescue astronauts, manage fuel.  Lives: 3.
    """

    num_actions: int = 7

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=500_000)

    def _reset(self, key: jax.Array) -> GravitarState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : GravitarState
            Ship at top-centre, full fuel, 3 lives.
        """
        return GravitarState(
            ship_x=jnp.float32(80.0),
            ship_y=jnp.float32(30.0),
            ship_vx=jnp.float32(0.0),
            ship_vy=jnp.float32(0.0),
            ship_angle=jnp.float32(0.0),
            fuel=jnp.int32(_FUEL_MAX),
            bullet_x=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_y=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_active=jnp.zeros(_N_BULLETS, dtype=jnp.bool_),
            bullet_timer=jnp.zeros(_N_BULLETS, dtype=jnp.int32),
            bunker_x=_BUNKER_X,
            bunker_active=jnp.ones(_N_BUNKERS, dtype=jnp.bool_),
            astronaut_x=_ASTRONAUT_X,
            astronaut_rescued=jnp.zeros(_N_ASTRONAUTS, dtype=jnp.bool_),
            shield_active=jnp.bool_(False),
            wave=jnp.int32(0),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: GravitarState, action: jax.Array) -> GravitarState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : GravitarState
            Current game state.
        action : jax.Array
            int32 — Action index (0–6).

        Returns
        -------
        new_state : GravitarState
            State after one emulated frame.
        """
        key = state.key
        step_reward = jnp.float32(0.0)

        rotate_cw = action == jnp.int32(3)
        rotate_ccw = action == jnp.int32(4)
        thrust = (action == jnp.int32(2)) | (action == jnp.int32(6))
        do_fire = (action == jnp.int32(1)) | (action == jnp.int32(6))
        shield = action == jnp.int32(5)

        new_angle = (
            state.ship_angle
            + jnp.where(rotate_cw, _ROTATE_SPEED, 0.0)
            - jnp.where(rotate_ccw, _ROTATE_SPEED, 0.0)
        )

        has_fuel = state.fuel > jnp.int32(0)
        ax = jnp.sin(new_angle) * jnp.where(thrust & has_fuel, _THRUST, 0.0)
        ay = -jnp.cos(new_angle) * jnp.where(thrust & has_fuel, _THRUST, 0.0) + _GRAVITY
        new_fuel = state.fuel - jnp.where(thrust & has_fuel, jnp.int32(1), jnp.int32(0))

        new_vx = jnp.clip((state.ship_vx + ax) * _FRICTION, -_MAX_SPEED, _MAX_SPEED)
        new_vy = jnp.clip((state.ship_vy + ay) * _FRICTION, -_MAX_SPEED, _MAX_SPEED)

        new_sx = jnp.clip(state.ship_x + new_vx, 0.0, _SCREEN_W)
        new_sy = jnp.clip(state.ship_y + new_vy, 0.0, _SCREEN_H)

        # Ground collision
        hit_ground = new_sy >= _GROUND_Y
        shielded = shield & (state.fuel > jnp.int32(0))

        # Fire bullet
        free_slot = jnp.argmin(state.bullet_active.astype(jnp.int32))
        has_free = ~jnp.all(state.bullet_active)
        bvx = jnp.sin(new_angle) * _BULLET_SPEED
        bvy = -jnp.cos(new_angle) * _BULLET_SPEED
        new_bx = jnp.where(
            do_fire & has_free,
            state.bullet_x.at[free_slot].set(new_sx),
            state.bullet_x,
        )
        new_by = jnp.where(
            do_fire & has_free,
            state.bullet_y.at[free_slot].set(new_sy),
            state.bullet_y,
        )
        new_bactive = jnp.where(
            do_fire & has_free,
            state.bullet_active.at[free_slot].set(True),
            state.bullet_active,
        )
        new_btimer = jnp.where(
            do_fire & has_free,
            state.bullet_timer.at[free_slot].set(jnp.int32(50)),
            state.bullet_timer,
        )

        new_bx = new_bx + jnp.where(new_bactive, bvx, 0.0)
        new_by = new_by + jnp.where(new_bactive, bvy, 0.0)
        new_btimer = new_btimer - jnp.where(new_bactive, jnp.int32(1), jnp.int32(0))
        new_bactive = new_bactive & (new_btimer > jnp.int32(0)) & (new_by < _GROUND_Y)

        # Bullet–bunker collision
        bul_hits_bunker = (
            new_bactive[:, None]
            & state.bunker_active[None, :]
            & (jnp.abs(new_bx[:, None] - _BUNKER_X[None, :]) < 10.0)
            & (jnp.abs(new_by[:, None] - _BUNKER_Y) < 8.0)
        )
        bunker_killed = jnp.any(bul_hits_bunker, axis=0)
        bul_used = jnp.any(bul_hits_bunker, axis=1)
        n_bunkers = jnp.sum(bunker_killed).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_bunkers * 250)
        new_bunker_active = state.bunker_active & ~bunker_killed
        new_bactive = new_bactive & ~bul_used

        # Rescue astronaut (ship near ground astronaut position, y near ground)
        near_ground = new_sy >= _ASTRONAUT_Y - 10.0
        rescues = (
            near_ground
            & (jnp.abs(_ASTRONAUT_X - new_sx) < 12.0)
            & ~state.astronaut_rescued
        )
        n_rescued = jnp.sum(rescues).astype(jnp.int32)
        step_reward = step_reward + jnp.float32(n_rescued * 1000)
        new_astronaut_rescued = state.astronaut_rescued | rescues

        # Wave complete
        all_done = jnp.all(new_bunker_active == jnp.bool_(False))
        new_wave = state.wave + jnp.where(all_done, jnp.int32(1), jnp.int32(0))
        new_bunker_active = jnp.where(
            all_done, jnp.ones(_N_BUNKERS, dtype=jnp.bool_), new_bunker_active
        )

        # Life loss conditions
        fuel_out = new_fuel <= jnp.int32(0)
        crashed = hit_ground & ~shielded
        life_lost = fuel_out | crashed
        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        new_fuel = jnp.where(life_lost, jnp.int32(_FUEL_MAX), new_fuel)
        new_sx = jnp.where(life_lost, jnp.float32(80.0), new_sx)
        new_sy = jnp.where(life_lost, jnp.float32(30.0), new_sy)
        new_vx = jnp.where(life_lost, jnp.float32(0.0), new_vx)
        new_vy = jnp.where(life_lost, jnp.float32(0.0), new_vy)

        done = new_lives <= jnp.int32(0)

        return GravitarState(
            ship_x=new_sx,
            ship_y=new_sy,
            ship_vx=new_vx,
            ship_vy=new_vy,
            ship_angle=new_angle,
            fuel=new_fuel,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            bullet_timer=new_btimer,
            bunker_x=_BUNKER_X,
            bunker_active=new_bunker_active,
            astronaut_x=_ASTRONAUT_X,
            astronaut_rescued=new_astronaut_rescued,
            shield_active=shield,
            wave=new_wave,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: GravitarState, action: jax.Array) -> GravitarState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : GravitarState
            Current game state.
        action : jax.Array
            int32 — Action index (0–6).

        Returns
        -------
        new_state : GravitarState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: GravitarState) -> GravitarState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: GravitarState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : GravitarState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Ground
        ground_mask = _ROW_IDX >= int(_GROUND_Y)
        frame = jnp.where(ground_mask[:, :, None], _COLOR_GROUND, frame)

        # Fuel bar
        fuel_frac = state.fuel.astype(jnp.float32) / jnp.float32(_FUEL_MAX)
        fuel_w = (fuel_frac * 100.0).astype(jnp.int32)
        fuel_mask = (_ROW_IDX < 8) & (_COL_IDX < fuel_w)
        frame = jnp.where(fuel_mask[:, :, None], _COLOR_FUEL, frame)

        # Bunkers
        def draw_bunker(frm, i):
            bx = state.bunker_x[i].astype(jnp.int32)
            by = int(_BUNKER_Y)
            mask = (
                state.bunker_active[i]
                & (_ROW_IDX >= by - 6)
                & (_ROW_IDX <= by + 6)
                & (_COL_IDX >= bx - 8)
                & (_COL_IDX <= bx + 8)
            )
            return jnp.where(mask[:, :, None], _COLOR_BUNKER, frm), None

        frame, _ = jax.lax.scan(draw_bunker, frame, jnp.arange(_N_BUNKERS))

        # Astronauts
        def draw_astronaut(frm, i):
            ax = state.astronaut_x[i].astype(jnp.int32)
            ay = int(_ASTRONAUT_Y)
            mask = (
                ~state.astronaut_rescued[i]
                & (_ROW_IDX >= ay - 5)
                & (_ROW_IDX <= ay + 5)
                & (_COL_IDX >= ax - 3)
                & (_COL_IDX <= ax + 3)
            )
            return jnp.where(mask[:, :, None], _COLOR_ASTRONAUT, frm), None

        frame, _ = jax.lax.scan(draw_astronaut, frame, jnp.arange(_N_ASTRONAUTS))

        # Bullets
        def draw_bullet(frm, i):
            bx = state.bullet_x[i].astype(jnp.int32)
            by = state.bullet_y[i].astype(jnp.int32)
            mask = state.bullet_active[i] & (_ROW_IDX == by) & (_COL_IDX == bx)
            return jnp.where(mask[:, :, None], _COLOR_BULLET, frm), None

        frame, _ = jax.lax.scan(draw_bullet, frame, jnp.arange(_N_BULLETS))

        # Ship
        sx = state.ship_x.astype(jnp.int32)
        sy = state.ship_y.astype(jnp.int32)
        ship_mask = (
            (_ROW_IDX >= sy - 5)
            & (_ROW_IDX <= sy + 5)
            & (_COL_IDX >= sx - 4)
            & (_COL_IDX <= sx + 4)
        )
        frame = jnp.where(ship_mask[:, :, None], _COLOR_SHIP, frame)

        return frame

    def _key_map(self) -> dict:
        """
        Return the key-to-action mapping for interactive play.

        Returns
        -------
        key_map : dict
            Mapping of pygame key constants to Gravitar action indices.
        """
        import pygame

        return {
            pygame.K_SPACE: 1,
            pygame.K_UP: 2,
            pygame.K_w: 2,
            pygame.K_RIGHT: 3,
            pygame.K_d: 3,
            pygame.K_LEFT: 4,
            pygame.K_a: 4,
            pygame.K_LSHIFT: 5,
        }
