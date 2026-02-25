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

"""Asteroids — JAX-native game implementation.

Pilot a rotating spaceship through an asteroid field, destroying rocks
with photon torpedoes.  Large asteroids split into smaller ones.

Action space (14 actions):
    0 — NOOP
    1 — FIRE
    2 — UP    (thrust)
    3 — RIGHT (rotate clockwise)
    4 — LEFT  (rotate counter-clockwise)
    5 — DOWN  (hyperspace warp)
    6 — UP + FIRE
    7 — RIGHT + FIRE
    8 — LEFT + FIRE
    9–13 — combinations (mapped to above)

Scoring:
    Large asteroid  — +20
    Medium asteroid — +50
    Small asteroid  — +100
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
_N_ROCKS: int = 12  # maximum concurrent rock pieces
_N_BULLETS: int = 4

_THRUST: float = 0.15
_MAX_SPEED: float = 4.0
_ROTATE_SPEED: float = 0.12  # radians per frame
_BULLET_SPEED: float = 5.0
_FRICTION: float = 0.99

_SCREEN_W: float = 160.0
_SCREEN_H: float = 210.0
_INIT_LIVES: int = 3

# Rock sizes: 0=large, 1=medium, 2=small
_ROCK_SPEED = jnp.array([0.6, 1.2, 2.0], dtype=jnp.float32)
_ROCK_RADIUS = jnp.array([12.0, 7.0, 4.0], dtype=jnp.float32)
_ROCK_SCORE = jnp.array([20, 50, 100], dtype=jnp.int32)

_ROW_IDX = jnp.arange(210)[:, None]
_COL_IDX = jnp.arange(160)[None, :]

_COLOR_BG = jnp.array([0, 0, 0], dtype=jnp.uint8)
_COLOR_SHIP = jnp.array([200, 200, 255], dtype=jnp.uint8)
_COLOR_ROCK = jnp.array([160, 160, 160], dtype=jnp.uint8)
_COLOR_BULLET = jnp.array([255, 255, 100], dtype=jnp.uint8)


@chex.dataclass
class AsteroidsState(AtariState):
    """
    Complete Asteroids game state — a JAX pytree.

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
        float32 — Ship heading (radians, 0=up).
    bullet_x : jax.Array
        float32[4] — Bullet x positions.
    bullet_y : jax.Array
        float32[4] — Bullet y positions.
    bullet_active : jax.Array
        bool[4] — Bullets in-flight.
    bullet_timer : jax.Array
        int32[4] — Remaining frames before expiry.
    rock_x : jax.Array
        float32[12] — Rock x positions.
    rock_y : jax.Array
        float32[12] — Rock y positions.
    rock_vx : jax.Array
        float32[12] — Rock x velocities.
    rock_vy : jax.Array
        float32[12] — Rock y velocities.
    rock_size : jax.Array
        int32[12] — Rock size (0=large, 1=medium, 2=small).
    rock_active : jax.Array
        bool[12] — Rock present.
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
    bullet_x: jax.Array
    bullet_y: jax.Array
    bullet_active: jax.Array
    bullet_timer: jax.Array
    rock_x: jax.Array
    rock_y: jax.Array
    rock_vx: jax.Array
    rock_vy: jax.Array
    rock_size: jax.Array
    rock_active: jax.Array
    wave: jax.Array
    key: jax.Array


def _spawn_rocks(key: jax.Array, wave: jax.Array):
    """Spawn 4 large asteroids at random positions away from centre."""
    k1, k2, k3, k4 = jax.random.split(key, 4)
    n = 4
    angles = jax.random.uniform(k1, (n,)) * 2.0 * jnp.pi
    radii = jax.random.uniform(k2, (n,)) * 30.0 + 40.0
    rx = 80.0 + jnp.cos(angles) * radii
    ry = 105.0 + jnp.sin(angles) * radii
    va = jax.random.uniform(k3, (n,)) * 2.0 * jnp.pi
    spd = _ROCK_SPEED[0] * (1.0 + wave.astype(jnp.float32) * 0.1)
    vx = jnp.cos(va) * spd
    vy = jnp.sin(va) * spd
    # Pad to _N_ROCKS
    rx_full = jnp.concatenate([rx, jnp.zeros(_N_ROCKS - n, dtype=jnp.float32)])
    ry_full = jnp.concatenate([ry, jnp.zeros(_N_ROCKS - n, dtype=jnp.float32)])
    vx_full = jnp.concatenate([vx, jnp.zeros(_N_ROCKS - n, dtype=jnp.float32)])
    vy_full = jnp.concatenate([vy, jnp.zeros(_N_ROCKS - n, dtype=jnp.float32)])
    sz_full = jnp.concatenate(
        [jnp.zeros(n, dtype=jnp.int32), jnp.zeros(_N_ROCKS - n, dtype=jnp.int32)]
    )
    ac_full = jnp.concatenate(
        [jnp.ones(n, dtype=jnp.bool_), jnp.zeros(_N_ROCKS - n, dtype=jnp.bool_)]
    )
    return rx_full, ry_full, vx_full, vy_full, sz_full, ac_full


class Asteroids(AtariEnv):
    """
    Asteroids implemented as a pure JAX function suite.

    Destroy all asteroids to advance.  Lives: 3.
    """

    num_actions: int = 14

    def default_params(self) -> EnvParams:
        return EnvParams(noop_max=0, max_episode_steps=500_000)

    def _reset(self, key: jax.Array) -> AsteroidsState:
        """
        Return the canonical initial game state.

        Parameters
        ----------
        key : jax.Array
            uint32[2] — JAX PRNG key.

        Returns
        -------
        state : AsteroidsState
            Ship at centre, 4 large asteroids, 3 lives.
        """
        key, sk = jax.random.split(key)
        rx, ry, rvx, rvy, rsz, rac = _spawn_rocks(sk, jnp.int32(0))
        return AsteroidsState(
            ship_x=jnp.float32(80.0),
            ship_y=jnp.float32(105.0),
            ship_vx=jnp.float32(0.0),
            ship_vy=jnp.float32(0.0),
            ship_angle=jnp.float32(0.0),
            bullet_x=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_y=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_active=jnp.zeros(_N_BULLETS, dtype=jnp.bool_),
            bullet_timer=jnp.zeros(_N_BULLETS, dtype=jnp.int32),
            rock_x=rx,
            rock_y=ry,
            rock_vx=rvx,
            rock_vy=rvy,
            rock_size=rsz,
            rock_active=rac,
            wave=jnp.int32(0),
            key=key,
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
        )

    def _step_physics(self, state: AsteroidsState, action: jax.Array) -> AsteroidsState:
        """
        Advance the game by one emulated frame.

        Parameters
        ----------
        state : AsteroidsState
            Current game state.
        action : jax.Array
            int32 — Action index (0–13).

        Returns
        -------
        new_state : AsteroidsState
            State after one emulated frame.
        """
        key, sk1, sk2 = jax.random.split(state.key, 3)
        step_reward = jnp.float32(0.0)

        # Map actions
        thrust = (action == jnp.int32(2)) | (action == jnp.int32(6))
        rotate_cw = (action == jnp.int32(3)) | (action == jnp.int32(7))
        rotate_ccw = (action == jnp.int32(4)) | (action == jnp.int32(8))
        do_fire = (
            (action == jnp.int32(1))
            | (action == jnp.int32(6))
            | (action == jnp.int32(7))
            | (action == jnp.int32(8))
        )
        hyperspace = action == jnp.int32(5)

        # Rotation
        new_angle = (
            state.ship_angle
            + jnp.where(rotate_cw, _ROTATE_SPEED, 0.0)
            - jnp.where(rotate_ccw, _ROTATE_SPEED, 0.0)
        )

        # Thrust
        ax = jnp.sin(new_angle) * jnp.where(thrust, _THRUST, 0.0)
        ay = -jnp.cos(new_angle) * jnp.where(thrust, _THRUST, 0.0)
        new_vx = jnp.clip((state.ship_vx + ax) * _FRICTION, -_MAX_SPEED, _MAX_SPEED)
        new_vy = jnp.clip((state.ship_vy + ay) * _FRICTION, -_MAX_SPEED, _MAX_SPEED)

        # Move ship (wrap around screen)
        new_sx = (state.ship_x + new_vx) % _SCREEN_W
        new_sy = (state.ship_y + new_vy) % _SCREEN_H

        # Hyperspace (random teleport)
        warp_x = jax.random.uniform(sk1) * _SCREEN_W
        warp_y = jax.random.uniform(sk2) * _SCREEN_H
        new_sx = jnp.where(hyperspace, warp_x, new_sx)
        new_sy = jnp.where(hyperspace, warp_y, new_sy)

        # Fire bullet
        free_slot = jnp.argmin(state.bullet_active.astype(jnp.int32))
        has_free = ~jnp.all(state.bullet_active)
        bvx = jnp.sin(new_angle) * _BULLET_SPEED
        bvy = -jnp.cos(new_angle) * _BULLET_SPEED
        new_bx = jnp.where(
            do_fire & has_free,
            state.bullet_x.at[free_slot].set(new_sx + bvx),
            state.bullet_x,
        )
        new_by = jnp.where(
            do_fire & has_free,
            state.bullet_y.at[free_slot].set(new_sy + bvy),
            state.bullet_y,
        )
        new_bactive = jnp.where(
            do_fire & has_free,
            state.bullet_active.at[free_slot].set(True),
            state.bullet_active,
        )
        new_btimer = jnp.where(
            do_fire & has_free,
            state.bullet_timer.at[free_slot].set(jnp.int32(40)),
            state.bullet_timer,
        )

        # Move bullets (wrap)
        new_bx = (new_bx + jnp.where(new_bactive, bvx, 0.0)) % _SCREEN_W
        new_by = (new_by + jnp.where(new_bactive, bvy, 0.0)) % _SCREEN_H
        new_btimer = new_btimer - jnp.where(new_bactive, jnp.int32(1), jnp.int32(0))
        new_bactive = new_bactive & (new_btimer > jnp.int32(0))

        # Move rocks (wrap)
        new_rx = (state.rock_x + state.rock_vx) % _SCREEN_W
        new_ry = (state.rock_y + state.rock_vy) % _SCREEN_H
        new_rock_active = state.rock_active

        # Bullet–rock collision
        rock_radii = _ROCK_RADIUS[state.rock_size]
        bul_hits_rock = (
            new_bactive[:, None]
            & new_rock_active[None, :]
            & (jnp.abs(new_bx[:, None] - new_rx[None, :]) < rock_radii[None, :])
            & (jnp.abs(new_by[:, None] - new_ry[None, :]) < rock_radii[None, :])
        )
        rock_hit = jnp.any(bul_hits_rock, axis=0)
        bul_used = jnp.any(bul_hits_rock, axis=1)
        n_hit = jnp.sum(rock_hit).astype(jnp.int32)
        scores = jnp.sum(
            jnp.where(rock_hit, _ROCK_SCORE[state.rock_size], jnp.int32(0))
        )
        step_reward = step_reward + scores.astype(jnp.float32)
        new_bactive = new_bactive & ~bul_used

        # Remove hit rocks
        new_rock_active = new_rock_active & ~rock_hit

        # Ship–rock collision
        ship_hits = (
            new_rock_active
            & (jnp.abs(new_rx - new_sx) < rock_radii + 5.0)
            & (jnp.abs(new_ry - new_sy) < rock_radii + 5.0)
        )
        ship_hit = jnp.any(ship_hits)
        new_lives = state.lives - jnp.where(ship_hit, jnp.int32(1), jnp.int32(0))
        # Respawn ship at centre
        new_sx = jnp.where(ship_hit, jnp.float32(80.0), new_sx)
        new_sy = jnp.where(ship_hit, jnp.float32(105.0), new_sy)
        new_vx = jnp.where(ship_hit, jnp.float32(0.0), new_vx)
        new_vy = jnp.where(ship_hit, jnp.float32(0.0), new_vy)

        # Wave complete: spawn new wave
        wave_clear = ~jnp.any(new_rock_active)
        new_wave = state.wave + jnp.where(wave_clear, jnp.int32(1), jnp.int32(0))
        rx_new, ry_new, rvx_new, rvy_new, rsz_new, rac_new = _spawn_rocks(sk2, new_wave)
        new_rx = jnp.where(wave_clear, rx_new, new_rx)
        new_ry = jnp.where(wave_clear, ry_new, new_ry)
        new_rvx = jnp.where(wave_clear, rvx_new, state.rock_vx)
        new_rvy = jnp.where(wave_clear, rvy_new, state.rock_vy)
        new_rsz = jnp.where(wave_clear, rsz_new, state.rock_size)
        new_rock_active = jnp.where(wave_clear, rac_new, new_rock_active)

        done = new_lives <= jnp.int32(0)

        return AsteroidsState(
            ship_x=new_sx,
            ship_y=new_sy,
            ship_vx=new_vx,
            ship_vy=new_vy,
            ship_angle=new_angle,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_active=new_bactive,
            bullet_timer=new_btimer,
            rock_x=new_rx,
            rock_y=new_ry,
            rock_vx=new_rvx,
            rock_vy=new_rvy,
            rock_size=new_rsz,
            rock_active=new_rock_active,
            wave=new_wave,
            key=key,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            episode_step=state.episode_step + jnp.int32(1),
        )

    def _step(self, state: AsteroidsState, action: jax.Array) -> AsteroidsState:
        """
        Advance the game by one agent step (4 emulated frames).

        Parameters
        ----------
        state : AsteroidsState
            Current game state.
        action : jax.Array
            int32 — Action index (0–13).

        Returns
        -------
        new_state : AsteroidsState
            State after 4 emulated frames.
        """
        state = state.__replace__(reward=jnp.float32(0.0))

        def body(i: jax.Array, s: AsteroidsState) -> AsteroidsState:
            return self._step_physics(s, action)

        return jax.lax.fori_loop(0, 4, body, state)

    def render(self, state: AsteroidsState) -> jax.Array:
        """
        Render the current game state as an RGB frame.

        Parameters
        ----------
        state : AsteroidsState
            Current game state.

        Returns
        -------
        frame : jax.Array
            uint8[210, 160, 3] — RGB image.
        """
        frame = jnp.full((210, 160, 3), _COLOR_BG, dtype=jnp.uint8)

        # Rocks
        def draw_rock(frm, i):
            rx = state.rock_x[i].astype(jnp.int32)
            ry = state.rock_y[i].astype(jnp.int32)
            r = _ROCK_RADIUS[state.rock_size[i]].astype(jnp.int32)
            mask = (
                state.rock_active[i]
                & (_ROW_IDX >= ry - r)
                & (_ROW_IDX <= ry + r)
                & (_COL_IDX >= rx - r)
                & (_COL_IDX <= rx + r)
            )
            return jnp.where(mask[:, :, None], _COLOR_ROCK, frm), None

        frame, _ = jax.lax.scan(draw_rock, frame, jnp.arange(_N_ROCKS))

        # Bullets
        def draw_bullet(frm, i):
            bx = state.bullet_x[i].astype(jnp.int32)
            by = state.bullet_y[i].astype(jnp.int32)
            mask = state.bullet_active[i] & (_ROW_IDX == by) & (_COL_IDX == bx)
            return jnp.where(mask[:, :, None], _COLOR_BULLET, frm), None

        frame, _ = jax.lax.scan(draw_bullet, frame, jnp.arange(_N_BULLETS))

        # Ship (small triangle approximated by a square)
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
            Mapping of pygame key constants to Asteroids action indices.
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
            pygame.K_DOWN: 5,
        }
