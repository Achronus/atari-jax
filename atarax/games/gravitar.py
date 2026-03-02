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

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Screen geometry (pixels, y=0 at top):
    Ship     : x ∈ [0, 160], y ∈ [0, 180]; starts at (80, 30)
    Ground   : y = 180 (fatal without shield)
    Bunkers  : y = 175; x = 25, 55, 80, 105, 135
    Astronauts: y = 170; x = 30, 65, 95, 130
    Fuel bar : top 8 rows, width proportional to remaining fuel

Action space (18 actions — ALE minimal set):
    0   NOOP
    1   FIRE
    2   UP        — thrust forward
    3   RIGHT     — rotate clockwise
    4   LEFT      — rotate counter-clockwise
    5   DOWN      — shield
    6   UPRIGHT   — thrust + rotate CW
    7   UPLEFT    — thrust + rotate CCW
    8   DOWNRIGHT — shield + rotate CW
    9   DOWNLEFT  — shield + rotate CCW
    10  UPFIRE    — thrust + fire
    11  RIGHTFIRE — rotate CW + fire
    12  LEFTFIRE  — rotate CCW + fire
    13  DOWNFIRE  — shield + fire
    14  UPRIGHTFIRE  — thrust + rotate CW + fire
    15  UPLEFTFIRE   — thrust + rotate CCW + fire
    16  DOWNRIGHTFIRE — shield + rotate CW + fire
    17  DOWNLEFTFIRE  — shield + rotate CCW + fire
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxGame, AtaraxParams
from atarax.state import AtariState

# --- Ship physics ---
_THRUST: float = 0.12  # acceleration per frame when thrusting
_MAX_SPEED: float = 3.5  # px/frame speed cap
_GRAVITY: float = 0.04  # downward acceleration per frame (y increases downward)
_ROTATE_RATE: float = 0.10  # radians per frame per rotate action
_FRICTION: float = 0.995  # velocity decay per frame

# --- Bullets ---
_N_BULLETS: int = 3
_BULLET_SPEED: float = 4.0  # px/frame
_BULLET_LIFETIME: int = 50  # frames before a bullet expires

# --- Fuel ---
_FUEL_MAX: int = 700
_FUEL_DRAIN: int = 1  # fuel units consumed per thrust frame

# --- Ground ---
_GROUND_Y: float = 180.0  # fatal crash y (without shield)
_SCREEN_W: float = 160.0
_PLAY_TOP: float = 10.0  # upper bound (ship can't fly above HUD)

# --- Bunkers ---
_N_BUNKERS: int = 5
_BUNKER_XS: tuple = (25.0, 55.0, 80.0, 105.0, 135.0)
_BUNKER_Y: float = 175.0
_BUNKER_HIT_DX: float = 10.0  # x half-width for bullet collision
_BUNKER_HIT_DY: float = 8.0  # y half-height for bullet collision
_BUNKER_W: int = 16
_BUNKER_H: int = 12
_BUNKER_REWARD: int = 250

# --- Astronauts ---
_N_ASTRONAUTS: int = 4
_ASTRONAUT_XS: tuple = (30.0, 65.0, 95.0, 130.0)
_ASTRONAUT_Y: float = 170.0
_ASTRONAUT_RESCUE_DX: float = 12.0  # rescue x range
_ASTRONAUT_RESCUE_DY: float = 12.0  # rescue y range
_ASTRONAUT_W: int = 8
_ASTRONAUT_H: int = 10
_ASTRONAUT_REWARD: int = 1000

# --- Ship ---
_SHIP_W: int = 8
_SHIP_H: int = 8
_SHIP_INIT_X: float = 80.0
_SHIP_INIT_Y: float = 30.0

# --- Lives ---
_INIT_LIVES: int = 6  # ALE mode 0 default

# --- Fire cooldown ---
_FIRE_COOLDOWN: int = 60  # minimum emulated frames between shots

# --- Enemy (bunker) fire ---
_ENEMY_FIRE_RATE: int = 45      # emulated frames between each bunker shot
_ENEMY_RANGE_X: float = 30.0   # horizontal distance within which bunker hits ship
_ENEMY_MIN_Y: float = 115.0    # ship must be at or below this y to be in fire range

# --- Frame skip ---
_FRAME_SKIP: int = 4

# --- Precomputed position arrays (module-level, for render and collision only) ---
_BUNKER_X_ARR = jnp.array(_BUNKER_XS, dtype=jnp.float32)
_ASTRONAUT_X_ARR = jnp.array(_ASTRONAUT_XS, dtype=jnp.float32)

_ROW_IDX_R = jnp.arange(210)[:, None]
_COL_IDX_R = jnp.arange(160)[None, :]

# --- Colours ---
_BG_COLOR = jnp.array([0, 0, 0], dtype=jnp.uint8)
_GROUND_COLOR = jnp.array([60, 80, 40], dtype=jnp.uint8)
_SHIP_COLOR = jnp.array([200, 200, 255], dtype=jnp.uint8)
_SHIELD_COLOR = jnp.array([80, 80, 220], dtype=jnp.uint8)
_BUNKER_COLOR = jnp.array([200, 60, 60], dtype=jnp.uint8)
_ASTRONAUT_COLOR = jnp.array([100, 220, 100], dtype=jnp.uint8)
_BULLET_COLOR = jnp.array([255, 255, 80], dtype=jnp.uint8)
_FUEL_COLOR = jnp.array([255, 160, 40], dtype=jnp.uint8)


@chex.dataclass
class GravitarState(AtariState):
    """
    Complete Gravitar game state.

    Inherits `reward`, `done`, `step`, `episode_step` from `GameState` and
    `lives`, `score`, `level`, `key` from `AtariState`.

    `level` stores the current wave index.

    Parameters
    ----------
    ship_x : chex.Array
        float32 — Ship x position.
    ship_y : chex.Array
        float32 — Ship y position.
    ship_vx : chex.Array
        float32 — Ship x velocity.
    ship_vy : chex.Array
        float32 — Ship y velocity.
    ship_angle : chex.Array
        float32 — Heading angle in radians (0 = pointing up).
    fuel : chex.Array
        int32 — Remaining fuel (0 = empty → life lost).
    bullet_x : chex.Array
        float32[3] — Bullet x positions.
    bullet_y : chex.Array
        float32[3] — Bullet y positions.
    bullet_vx : chex.Array
        float32[3] — Bullet x velocities (set at launch, constant after).
    bullet_vy : chex.Array
        float32[3] — Bullet y velocities (set at launch, constant after).
    bullet_active : chex.Array
        bool[3] — Bullets currently in flight.
    bullet_timer : chex.Array
        int32[3] — Frames remaining before bullet expires.
    bunker_active : chex.Array
        bool[5] — Bunkers alive; reset when wave clears.
    astronaut_rescued : chex.Array
        bool[4] — Astronauts already rescued (persistent across waves).
    shield_active : chex.Array
        bool — Shield on (blocks ground crash; set from action each frame).
    fire_cooldown : chex.Array
        int32 — Frames remaining before next player bullet can be fired.
    enemy_fire_timer : chex.Array
        int32[5] — Per-bunker countdown until next shot (fires when reaches 0).
    """

    ship_x: chex.Array
    ship_y: chex.Array
    ship_vx: chex.Array
    ship_vy: chex.Array
    ship_angle: chex.Array
    fuel: chex.Array
    bullet_x: chex.Array
    bullet_y: chex.Array
    bullet_vx: chex.Array
    bullet_vy: chex.Array
    bullet_active: chex.Array
    bullet_timer: chex.Array
    bunker_active: chex.Array
    astronaut_rescued: chex.Array
    shield_active: chex.Array
    fire_cooldown: chex.Array
    enemy_fire_timer: chex.Array


class Gravitar(AtaraxGame):
    """
    Gravitar implemented as a pure-JAX function suite.

    Thrust through a gravity field, destroy bunkers, rescue astronauts,
    and manage fuel. Starts with 6 lives (ALE mode 0).
    """

    num_actions: int = 18

    def _reset(self, key: chex.PRNGKey) -> GravitarState:
        """Return the canonical initial game state."""
        return GravitarState(
            ship_x=jnp.float32(_SHIP_INIT_X),
            ship_y=jnp.float32(_SHIP_INIT_Y),
            ship_vx=jnp.float32(0.0),
            ship_vy=jnp.float32(0.0),
            ship_angle=jnp.float32(0.0),
            fuel=jnp.int32(_FUEL_MAX),
            bullet_x=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_y=jnp.full(_N_BULLETS, -10.0, dtype=jnp.float32),
            bullet_vx=jnp.zeros(_N_BULLETS, dtype=jnp.float32),
            bullet_vy=jnp.zeros(_N_BULLETS, dtype=jnp.float32),
            bullet_active=jnp.zeros(_N_BULLETS, dtype=jnp.bool_),
            bullet_timer=jnp.zeros(_N_BULLETS, dtype=jnp.int32),
            bunker_active=jnp.ones(_N_BUNKERS, dtype=jnp.bool_),
            astronaut_rescued=jnp.zeros(_N_ASTRONAUTS, dtype=jnp.bool_),
            shield_active=jnp.bool_(False),
            fire_cooldown=jnp.int32(0),
            enemy_fire_timer=jnp.array([0, 18, 36, 54, 72], dtype=jnp.int32),
            lives=jnp.int32(_INIT_LIVES),
            score=jnp.int32(0),
            level=jnp.int32(0),  # wave index
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=key,
        )

    def _step_physics(self, state: GravitarState, action: jax.Array) -> GravitarState:
        """
        Advance the game by one emulated frame (branch-free).

        Parameters
        ----------
        state : GravitarState
            Current game state.
        action : jax.Array
            int32 — ALE action index (0–17).

        Returns
        -------
        new_state : GravitarState
            State after one emulated frame.
        """
        # --- Action decode ---
        has_fire = (
            (action == 1)
            | (action == 10)
            | (action == 11)
            | (action == 12)
            | (action == 13)
            | (action == 14)
            | (action == 15)
            | (action == 16)
            | (action == 17)
        )
        has_thrust = (
            (action == 2)
            | (action == 6)
            | (action == 7)
            | (action == 10)
            | (action == 14)
            | (action == 15)
        )
        has_rotate_cw = (
            (action == 3)
            | (action == 6)
            | (action == 8)
            | (action == 11)
            | (action == 14)
            | (action == 16)
        )
        has_rotate_ccw = (
            (action == 4)
            | (action == 7)
            | (action == 9)
            | (action == 12)
            | (action == 15)
            | (action == 17)
        )
        has_shield = (
            (action == 5)
            | (action == 8)
            | (action == 9)
            | (action == 13)
            | (action == 16)
            | (action == 17)
        )

        step_reward = jnp.float32(0.0)

        # --- Rotate ---
        new_angle = state.ship_angle + jnp.where(
            has_rotate_cw,
            jnp.float32(_ROTATE_RATE),
            jnp.where(has_rotate_ccw, jnp.float32(-_ROTATE_RATE), jnp.float32(0.0)),
        )

        # --- Thrust ---
        has_fuel = state.fuel > jnp.int32(0)
        do_thrust = has_thrust & has_fuel
        ax = jnp.sin(new_angle) * jnp.where(
            do_thrust, jnp.float32(_THRUST), jnp.float32(0.0)
        )
        ay = -jnp.cos(new_angle) * jnp.where(
            do_thrust, jnp.float32(_THRUST), jnp.float32(0.0)
        )
        ay = ay + jnp.float32(_GRAVITY)  # gravity always applied

        new_fuel = state.fuel - jnp.where(
            do_thrust, jnp.int32(_FUEL_DRAIN), jnp.int32(0)
        )
        new_fuel = jnp.maximum(new_fuel, jnp.int32(0))

        # --- Update velocity (with friction and speed cap) ---
        new_vx = jnp.clip(
            (state.ship_vx + ax) * jnp.float32(_FRICTION),
            jnp.float32(-_MAX_SPEED),
            jnp.float32(_MAX_SPEED),
        )
        new_vy = jnp.clip(
            (state.ship_vy + ay) * jnp.float32(_FRICTION),
            jnp.float32(-_MAX_SPEED),
            jnp.float32(_MAX_SPEED),
        )

        # --- Update position ---
        raw_sx = state.ship_x + new_vx
        new_sx = raw_sx % jnp.float32(_SCREEN_W)  # wrap horizontally
        new_sy = jnp.clip(
            state.ship_y + new_vy,
            jnp.float32(_PLAY_TOP),
            jnp.float32(_GROUND_Y),
        )

        # --- Ground interaction ---
        hit_ground = new_sy >= jnp.float32(_GROUND_Y)
        shielded = has_shield & has_fuel
        crashed = hit_ground & ~shielded

        # Shield bounce off ground
        new_vy = jnp.where(
            hit_ground & shielded,
            -jnp.abs(new_vy) * jnp.float32(0.5),
            new_vy,
        )
        new_sy = jnp.where(
            hit_ground & shielded,
            jnp.float32(_GROUND_Y - 2.0),
            new_sy,
        )

        # --- Fire bullet (with cooldown to limit fire rate) ---
        bvx_launch = jnp.sin(new_angle) * jnp.float32(_BULLET_SPEED)
        bvy_launch = -jnp.cos(new_angle) * jnp.float32(_BULLET_SPEED)

        new_fire_cooldown = jnp.maximum(
            state.fire_cooldown - jnp.int32(1), jnp.int32(0)
        )
        free_slot = jnp.argmin(state.bullet_active.astype(jnp.int32))
        has_free = ~jnp.all(state.bullet_active)
        do_spawn = has_fire & has_free & (new_fire_cooldown == jnp.int32(0))
        new_fire_cooldown = jnp.where(
            do_spawn, jnp.int32(_FIRE_COOLDOWN), new_fire_cooldown
        )

        new_bx = state.bullet_x.at[free_slot].set(
            jnp.where(do_spawn, new_sx, state.bullet_x[free_slot])
        )
        new_by = state.bullet_y.at[free_slot].set(
            jnp.where(do_spawn, new_sy, state.bullet_y[free_slot])
        )
        new_bvx = state.bullet_vx.at[free_slot].set(
            jnp.where(do_spawn, bvx_launch, state.bullet_vx[free_slot])
        )
        new_bvy = state.bullet_vy.at[free_slot].set(
            jnp.where(do_spawn, bvy_launch, state.bullet_vy[free_slot])
        )
        new_bactive = state.bullet_active.at[free_slot].set(
            state.bullet_active[free_slot] | do_spawn
        )
        new_btimer = state.bullet_timer.at[free_slot].set(
            jnp.where(
                do_spawn, jnp.int32(_BULLET_LIFETIME), state.bullet_timer[free_slot]
            )
        )

        # --- Move all active bullets by their stored velocities ---
        new_bx = new_bx + jnp.where(new_bactive, new_bvx, jnp.float32(0.0))
        new_by = new_by + jnp.where(new_bactive, new_bvy, jnp.float32(0.0))
        new_btimer = new_btimer - jnp.where(new_bactive, jnp.int32(1), jnp.int32(0))
        new_bactive = (
            new_bactive
            & (new_btimer > jnp.int32(0))
            & (new_by < jnp.float32(_GROUND_Y))
            & (new_by >= jnp.float32(0.0))
        )

        # --- Bullet vs bunker collision ---
        # Shape: (N_BULLETS, N_BUNKERS)
        # Only bullets moving downward (toward the ground) can hit bunkers.
        # This prevents horizontal bullets at bunker altitude from trivially
        # sweeping across all bunkers — the primary cause of inflated scores.
        bul_hits_bunker = (
            new_bactive[:, None]
            & state.bunker_active[None, :]
            & (new_bvy[:, None] > jnp.float32(0.0))
            & (
                jnp.abs(new_bx[:, None] - _BUNKER_X_ARR[None, :])
                < jnp.float32(_BUNKER_HIT_DX)
            )
            & (
                jnp.abs(new_by[:, None] - jnp.float32(_BUNKER_Y))
                < jnp.float32(_BUNKER_HIT_DY)
            )
        )
        bunker_killed = jnp.any(bul_hits_bunker, axis=0)  # bool[5]
        bul_used = jnp.any(bul_hits_bunker, axis=1)  # bool[3]

        n_bunkers_killed = jnp.sum(bunker_killed.astype(jnp.int32))
        step_reward = step_reward + jnp.float32(n_bunkers_killed * _BUNKER_REWARD)
        new_bunker_active = state.bunker_active & ~bunker_killed
        new_bactive = new_bactive & ~bul_used

        # --- Astronaut rescue (ship proximity) ---
        near_y = new_sy >= jnp.float32(_ASTRONAUT_Y - _ASTRONAUT_RESCUE_DY)
        rescues = (
            near_y
            & (jnp.abs(_ASTRONAUT_X_ARR - new_sx) < jnp.float32(_ASTRONAUT_RESCUE_DX))
            & ~state.astronaut_rescued
        )
        n_rescued = jnp.sum(rescues.astype(jnp.int32))
        step_reward = step_reward + jnp.float32(n_rescued * _ASTRONAUT_REWARD)
        new_astronaut_rescued = state.astronaut_rescued | rescues

        # --- Wave complete: respawn all bunkers ---
        all_bunkers_cleared = ~jnp.any(new_bunker_active)
        new_wave = state.level + jnp.where(
            all_bunkers_cleared, jnp.int32(1), jnp.int32(0)
        )
        new_bunker_active = jnp.where(
            all_bunkers_cleared,
            jnp.ones(_N_BUNKERS, dtype=jnp.bool_),
            new_bunker_active,
        )

        # --- Enemy (bunker) fire ---
        # Each active bunker counts down independently; when its timer reaches
        # zero it "fires" at the ship.  We skip full bullet physics and instead
        # check whether the ship is within _ENEMY_RANGE_X horizontally — if so,
        # the shot connects (unless the shield is active).
        new_enemy_timer = jnp.maximum(
            state.enemy_fire_timer - jnp.int32(1), jnp.int32(0)
        )
        bunker_fires = (new_enemy_timer <= jnp.int32(0)) & state.bunker_active
        new_enemy_timer = jnp.where(
            bunker_fires, jnp.int32(_ENEMY_FIRE_RATE), new_enemy_timer
        )
        ship_in_range = (
            jnp.abs(_BUNKER_X_ARR - new_sx) < jnp.float32(_ENEMY_RANGE_X)
        )
        enemy_hit = jnp.any(bunker_fires & ship_in_range) & ~shielded
        crashed = crashed | enemy_hit

        # --- Life loss ---
        fuel_out = new_fuel <= jnp.int32(0)
        life_lost = crashed | fuel_out

        new_lives = state.lives - jnp.where(life_lost, jnp.int32(1), jnp.int32(0))
        # Respawn ship at top-centre on life loss
        new_sx = jnp.where(life_lost, jnp.float32(_SHIP_INIT_X), new_sx)
        new_sy = jnp.where(life_lost, jnp.float32(_SHIP_INIT_Y), new_sy)
        new_vx = jnp.where(life_lost, jnp.float32(0.0), new_vx)
        new_vy = jnp.where(life_lost, jnp.float32(0.0), new_vy)
        new_fuel = jnp.where(life_lost, jnp.int32(_FUEL_MAX), new_fuel)

        done = new_lives <= jnp.int32(0)

        return state.__replace__(
            ship_x=new_sx,
            ship_y=new_sy,
            ship_vx=new_vx,
            ship_vy=new_vy,
            ship_angle=new_angle,
            fuel=new_fuel,
            bullet_x=new_bx,
            bullet_y=new_by,
            bullet_vx=new_bvx,
            bullet_vy=new_bvy,
            bullet_active=new_bactive,
            bullet_timer=new_btimer,
            bunker_active=new_bunker_active,
            astronaut_rescued=new_astronaut_rescued,
            shield_active=has_shield,
            fire_cooldown=new_fire_cooldown,
            enemy_fire_timer=new_enemy_timer,
            lives=new_lives,
            score=state.score + jnp.int32(step_reward),
            level=new_wave,
            reward=state.reward + step_reward,
            done=done,
            step=state.step + jnp.int32(1),
            key=state.key,
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: GravitarState,
        action: jax.Array,
        params: AtaraxParams,
    ) -> GravitarState:
        state = state.__replace__(reward=jnp.float32(0.0), key=rng)
        new_state = jax.lax.fori_loop(
            0,
            _FRAME_SKIP,
            lambda _i, s: self._step_physics(s, action),
            state,
        )
        return new_state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: GravitarState) -> jax.Array:
        frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)

        # --- Ground ---
        ground_mask = _ROW_IDX_R >= jnp.int32(_GROUND_Y)
        frame = jnp.where(ground_mask[:, :, None], _GROUND_COLOR[None, None, :], frame)

        # --- Fuel bar (top 8 rows) ---
        fuel_frac = state.fuel.astype(jnp.float32) / jnp.float32(_FUEL_MAX)
        fuel_w = jnp.int32(fuel_frac * jnp.float32(100.0))
        fuel_mask = (_ROW_IDX_R < 8) & (_COL_IDX_R < fuel_w)
        frame = jnp.where(fuel_mask[:, :, None], _FUEL_COLOR[None, None, :], frame)

        # --- Bunkers ---
        by_int = jnp.int32(_BUNKER_Y)
        for i in range(_N_BUNKERS):
            bx = jnp.int32(state.bunker_active[i]) * jnp.int32(_BUNKER_XS[i])
            bunker_mask = (
                state.bunker_active[i]
                & (_ROW_IDX_R >= by_int - _BUNKER_H // 2)
                & (_ROW_IDX_R < by_int + _BUNKER_H // 2)
                & (_COL_IDX_R >= jnp.int32(_BUNKER_XS[i]) - _BUNKER_W // 2)
                & (_COL_IDX_R < jnp.int32(_BUNKER_XS[i]) + _BUNKER_W // 2)
            )
            frame = jnp.where(
                bunker_mask[:, :, None], _BUNKER_COLOR[None, None, :], frame
            )

        # --- Astronauts ---
        ay_int = jnp.int32(_ASTRONAUT_Y)
        for i in range(_N_ASTRONAUTS):
            astronaut_mask = (
                ~state.astronaut_rescued[i]
                & (_ROW_IDX_R >= ay_int - _ASTRONAUT_H // 2)
                & (_ROW_IDX_R < ay_int + _ASTRONAUT_H // 2)
                & (_COL_IDX_R >= jnp.int32(_ASTRONAUT_XS[i]) - _ASTRONAUT_W // 2)
                & (_COL_IDX_R < jnp.int32(_ASTRONAUT_XS[i]) + _ASTRONAUT_W // 2)
            )
            frame = jnp.where(
                astronaut_mask[:, :, None], _ASTRONAUT_COLOR[None, None, :], frame
            )

        # --- Bullets ---
        for i in range(_N_BULLETS):
            bx = jnp.int32(state.bullet_x[i])
            by = jnp.int32(state.bullet_y[i])
            bullet_mask = (
                state.bullet_active[i]
                & (_ROW_IDX_R >= by)
                & (_ROW_IDX_R < by + 3)
                & (_COL_IDX_R >= bx)
                & (_COL_IDX_R < bx + 2)
            )
            frame = jnp.where(
                bullet_mask[:, :, None], _BULLET_COLOR[None, None, :], frame
            )

        # --- Ship ---
        sx = jnp.int32(state.ship_x)
        sy = jnp.int32(state.ship_y)
        ship_color = jnp.where(state.shield_active, _SHIELD_COLOR, _SHIP_COLOR)
        ship_mask = (
            (_ROW_IDX_R >= sy - _SHIP_H // 2)
            & (_ROW_IDX_R < sy + _SHIP_H // 2)
            & (_COL_IDX_R >= sx - _SHIP_W // 2)
            & (_COL_IDX_R < sx + _SHIP_W // 2)
        )
        frame = jnp.where(ship_mask[:, :, None], ship_color[None, None, :], frame)

        return frame

    def _key_map(self):
        try:
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
                pygame.K_RSHIFT: 5,
            }
        except ImportError:
            return {}
