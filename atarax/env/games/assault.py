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

"""Assault — JAX-native SDF game implementation.

A turret (duck) is fixed to the ground. One large mothership crosses the top
of the screen; it periodically releases attackers that float down toward the
ground. Only 3 attackers are on screen at once (10 per wave).

Firing builds a HEAT bar — overheat kills the player. The heat bar is the
green indicator visible at the bottom of the screen.

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Action space (7 actions, matching ALE):
    0 — NOOP
    1 — FIRE        (straight up)
    2 — UP          (redirected to FIRE — turret raises barrel)
    3 — RIGHT
    4 — LEFT
    5 — RIGHTFIRE   (move right + fire diagonally right)
    6 — LEFTFIRE    (move left + fire diagonally left)
"""

from typing import ClassVar

import chex
import jax
import jax.numpy as jnp

from atarax.env._base.fixed_shooter import FixedShooterGame, FixedShooterState
from atarax.env.hud import HUD_PIP_XS, HUD_PIP_Y, render_life_pips, render_score
from atarax.env.sdf import (
    finalise_rgb,
    make_canvas,
    paint_layer,
    paint_sdf,
    render_circle_pool,
    render_rect_pool,
    sdf_rect,
)
from atarax.game import AtaraxParams

# ── Geometry ────────────────────────────────────────────────────────────────
_WORLD_W: float = 160.0
_WORLD_H: float = 210.0

_PLAYER_HW: float = 5.0
_PLAYER_HH: float = 4.0
_PLAYER_SPEED: float = 2.0
_PLAYER_X_MIN: float = 10.0
_PLAYER_X_MAX: float = 150.0
_PLAYER_Y_INIT: float = 188.0  # fixed ground position (no vertical movement)
_FIRE_COOLDOWN: int = 8
_MAX_PLAYER_BULLETS: int = 1  # one bullet on screen at a time (matches ALE)
_BULLET_SPEED: float = 6.0
_BULLET_VX: float = 3.5  # horizontal component for LEFTFIRE / RIGHTFIRE

# Heat bar — fires build heat; overheat kills the player
_MAX_HEAT: int = 100
_HEAT_COST: int = 15  # heat added per shot fired
_HEAT_REGEN: int = 1  # heat removed per frame (passive cooldown)

# Mothership (slot 0 of enemies array) — starts left side, crosses screen
_MOTHER_HW: float = 18.0
_MOTHER_HH: float = 8.0
_MOTHER_HIT_HW: float = 10.0  # smaller hit-box (requires precise aim)
_MOTHER_HIT_HH: float = 5.0
_MOTHER_HP: int = 5  # hits required to destroy mothership
_MOTHER_SPEED: float = 1.0
_MOTHER_X_INIT: float = 40.0  # starts on left (matches ALE cols 26-53 → centre ≈ 40)
_MOTHER_Y: float = 25.0
_MOTHER_RESPAWN: int = 90  # frames after kill until respawn

# Attackers (slots 1-3 of enemies array)
# type 1.0 = fighter (small square),  type 2.0 = bomber (circle)
_MAX_ATTACKERS: int = 3  # max on screen simultaneously
_MAX_ENEMIES: int = 1 + _MAX_ATTACKERS  # 4 total slots
_WAVE_SIZE: int = 10  # attackers per wave before wave advances
_ATCK_HW: float = 5.0  # fighter half-width
_ATCK_HH: float = 4.0  # fighter half-height
_ATCK_RADIUS: float = 6.0  # bomber radius
_ATCK_VX_MAX: float = 0.6  # max horizontal drift speed (slow wiggle)
_ATCK_VY_MIN: float = 0.3  # min descent speed
_ATCK_VY_MAX: float = 0.6  # max descent speed
_ATCK_RELEASE: int = 80  # frames between attacker releases
_ATCK_FIRE_PROB: float = 0.025  # probability of firing per attacker per frame
_ATCK_WIGGLE_PROB: float = 0.05  # probability of reversing vx per frame (wiggle)

_MAX_ENEMY_BULLETS: int = 3   # ALE Assault has ≤3 enemy bullets on screen at once
_ENEMY_BULLET_SPEED: float = 1.8

# Scoring
_SCORE_ATTACKER: int = 65
_SCORE_MOTHER: int = 130

_GROUND_Y: float = 194.0

# ── Colours ─────────────────────────────────────────────────────────────────
_COL_BG = jnp.array([0.00, 0.00, 0.04], dtype=jnp.float32)  # near-black
_COL_GROUND = jnp.array(
    [0.26, 0.62, 0.51], dtype=jnp.float32
)  # cyan (ALE RGB 66,158,130)
_COL_PLAYER = jnp.array([0.30, 0.60, 1.00], dtype=jnp.float32)  # blue
_COL_PBULLET = jnp.array([1.00, 1.00, 0.20], dtype=jnp.float32)  # yellow
_COL_MOTHER = jnp.array([0.20, 0.90, 0.30], dtype=jnp.float32)  # green
_COL_FIGHTER = jnp.array([0.90, 0.25, 0.15], dtype=jnp.float32)  # red (fighter square)
_COL_BOMBER = jnp.array([0.90, 0.55, 0.10], dtype=jnp.float32)  # orange (bomber circle)
_COL_EBULLET = jnp.array([1.00, 0.50, 0.10], dtype=jnp.float32)  # orange
_COL_HUD = jnp.array([0.20, 0.50, 0.90], dtype=jnp.float32)  # blue (player colour)
_COL_HEAT = jnp.array([0.10, 0.85, 0.20], dtype=jnp.float32)  # green (heat bar)

# enemies array: (4, 6) float32 [x, y, vx, vy, type, active]
# type 0.0 = mothership, type 1.0 = fighter, type 2.0 = bomber
_ENEMIES_INIT: jnp.ndarray = jnp.concatenate(
    [
        jnp.array(
            [[_MOTHER_X_INIT, _MOTHER_Y, _MOTHER_SPEED, 0.0, 0.0, 1.0]],
            dtype=jnp.float32,
        ),
        jnp.zeros((_MAX_ATTACKERS, 6), dtype=jnp.float32),
    ],
    axis=0,
)  # (4, 6)


@chex.dataclass
class AssaultParams(AtaraxParams):
    """Static configuration for Assault."""

    max_steps: int = 10000
    num_lives: int = 4


@chex.dataclass
class AssaultState(FixedShooterState):
    """
    Assault game state.

    Extends `FixedShooterState`. Field mapping:
    - `player_x`       → turret x (horizontal only; y is fixed)
    - `fire_cooldown`  → player fire cooldown
    - `player_bullets` → (1, 3) float32 [x, y, active]
    - `enemy_bullets`  → (6, 3) float32 [x, y, active]
    - `enemy_grid`, `fleet_*` → stubbed (not used)

    Extra fields:
    - `player_y`      → turret y (always == _PLAYER_Y_INIT; stored for render)
    - `enemies`       → (4, 6) float32 [x, y, vx, vy, type, active]
                        slot 0 = mothership (type 0), slots 1-3 = attackers
    - `spawn_timer`   → countdown to next attacker release / mothership respawn
    - `wave`          → current wave number
    - `wave_spawned`  → attackers spawned this wave (0-10)
    - `mother_hp`     → hits remaining to destroy mothership
    - `bullet_vx`     → x velocity of the single player bullet
    - `heat`          → current heat level (0-100); overheat = lose a life
    """

    player_y: chex.Array  # () float32
    enemies: chex.Array  # (4, 6) float32
    spawn_timer: chex.Array  # () int32
    wave: chex.Array  # () int32
    wave_spawned: chex.Array  # () int32 — attackers spawned this wave (0-10)
    mother_hp: chex.Array  # () int32
    bullet_vx: chex.Array  # () float32
    heat: chex.Array  # () int32 — 0..._MAX_HEAT


class Assault(FixedShooterGame):
    """
    Assault implemented as a pure-JAX function suite.

    All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.
    """

    num_actions: ClassVar[int] = 7
    game_id: ClassVar[str] = "assault"

    def _reset(self, rng: chex.PRNGKey) -> AssaultState:
        """Return the canonical initial game state."""
        return AssaultState(
            # FixedShooterState fields
            player_x=jnp.float32(80.0),
            fire_cooldown=jnp.int32(0),
            enemy_grid=jnp.zeros((1, 1), dtype=jnp.bool_),
            fleet_x=jnp.float32(0.0),
            fleet_y=jnp.float32(0.0),
            fleet_dir=jnp.int32(0),
            fleet_speed=jnp.float32(0.0),
            player_bullets=jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32),
            enemy_bullets=jnp.zeros((_MAX_ENEMY_BULLETS, 3), dtype=jnp.float32),
            # AssaultState fields
            player_y=jnp.float32(_PLAYER_Y_INIT),
            enemies=_ENEMIES_INIT,
            spawn_timer=jnp.int32(_ATCK_RELEASE),
            wave=jnp.int32(1),
            wave_spawned=jnp.int32(0),
            mother_hp=jnp.int32(_MOTHER_HP),
            bullet_vx=jnp.float32(0.0),
            heat=jnp.int32(0),
            # AtariState fields
            lives=jnp.int32(4),
            score=jnp.int32(0),
            level=jnp.int32(0),
            reward=jnp.float32(0.0),
            done=jnp.bool_(False),
            step=jnp.int32(0),
            episode_step=jnp.int32(0),
            key=rng,
        )

    def _step_physics(
        self,
        state: AssaultState,
        action: chex.Array,
        params: AssaultParams,
        rng: chex.PRNGKey,
    ) -> AssaultState:
        """Advance the game by one emulated frame (branch-free)."""

        rng, spawn_rng, fire_rng, wiggle_rng = jax.random.split(rng, 4)

        # ── 1. Player movement (horizontal only) ──────────────────────────
        move_right = (action == jnp.int32(3)) | (action == jnp.int32(5))
        move_left = (action == jnp.int32(4)) | (action == jnp.int32(6))
        dx = jnp.where(
            move_right,
            jnp.float32(_PLAYER_SPEED),
            jnp.where(move_left, jnp.float32(-_PLAYER_SPEED), jnp.float32(0.0)),
        )
        player_x = jnp.clip(
            state.player_x + dx, jnp.float32(_PLAYER_X_MIN), jnp.float32(_PLAYER_X_MAX)
        )
        player_y = jnp.float32(_PLAYER_Y_INIT)

        # ── 2. Heat management ────────────────────────────────────────────
        # Overheat check is done BEFORE firing this frame — prevents instant re-trigger
        overheated = state.heat >= jnp.int32(_MAX_HEAT)

        # ── 3. Fire player bullet ─────────────────────────────────────────
        want_fire = (
            (action == jnp.int32(1))
            | (action == jnp.int32(2))
            | (action == jnp.int32(5))
            | (action == jnp.int32(6))
        )
        can_fire = want_fire & (state.fire_cooldown <= jnp.int32(0)) & ~overheated
        pb_slot = jnp.argmin(state.player_bullets[:, 2])
        pb_slot_free = state.player_bullets[pb_slot, 2] < jnp.float32(0.5)
        do_fire = can_fire & pb_slot_free
        new_pb = jnp.stack([player_x, player_y - jnp.float32(5.0), jnp.float32(1.0)])
        player_bullets = jnp.where(
            do_fire,
            state.player_bullets.at[pb_slot].set(new_pb),
            state.player_bullets,
        )
        fire_cooldown = jnp.where(
            do_fire,
            jnp.int32(_FIRE_COOLDOWN),
            jnp.maximum(state.fire_cooldown - jnp.int32(1), jnp.int32(0)),
        )
        # Bullet x-velocity: diagonal for LEFTFIRE/RIGHTFIRE, straight for FIRE/UP
        new_bvx = jnp.where(
            action == jnp.int32(5),
            jnp.float32(_BULLET_VX),
            jnp.where(
                action == jnp.int32(6), jnp.float32(-_BULLET_VX), jnp.float32(0.0)
            ),
        )
        bullet_vx = jnp.where(do_fire, new_bvx, state.bullet_vx)

        # Heat: build on fire, cool passively, clamp to [0, _MAX_HEAT]
        heat = state.heat + jnp.where(do_fire, jnp.int32(_HEAT_COST), jnp.int32(0))
        heat = heat - jnp.int32(_HEAT_REGEN)
        heat = jnp.clip(heat, jnp.int32(0), jnp.int32(_MAX_HEAT))

        # ── 4. Move player bullet (upward + optional horizontal drift) ────
        new_bx = player_bullets[:, 0] + bullet_vx * player_bullets[:, 2]
        player_bullets = player_bullets.at[:, 0].set(new_bx)
        player_bullets = self._move_bullets(player_bullets, jnp.float32(-_BULLET_SPEED))
        x_oob = (player_bullets[:, 0] < jnp.float32(0.0)) | (
            player_bullets[:, 0] > jnp.float32(_WORLD_W)
        )
        player_bullets = player_bullets.at[:, 2].set(
            player_bullets[:, 2] * (~x_oob).astype(jnp.float32)
        )

        # ── 5. Move mothership (slot 0) ───────────────────────────────────
        enemies = state.enemies
        m_x = enemies[0, 0]
        m_vx = enemies[0, 2]
        m_act = enemies[0, 5]
        m_hit_left = m_x - jnp.float32(_MOTHER_HW) <= jnp.float32(8.0)
        m_hit_right = m_x + jnp.float32(_MOTHER_HW) >= jnp.float32(152.0)
        m_reverse = (m_hit_left | m_hit_right) & (m_act > jnp.float32(0.5))
        new_m_vx = jnp.where(m_reverse, -m_vx, m_vx)
        new_m_x = m_x + new_m_vx * m_act
        # Mothership respawn when inactive and spawn_timer reaches 0
        m_respawn = (m_act < jnp.float32(0.5)) & (state.spawn_timer <= jnp.int32(1))
        new_m_x = jnp.where(m_respawn, jnp.float32(80.0), new_m_x)
        new_m_vx = jnp.where(m_respawn, jnp.float32(_MOTHER_SPEED), new_m_vx)
        new_m_act = jnp.where(m_respawn, jnp.float32(1.0), m_act)
        enemies = enemies.at[0].set(
            jnp.stack(
                [
                    new_m_x,
                    jnp.float32(_MOTHER_Y),
                    new_m_vx,
                    jnp.float32(0.0),
                    jnp.float32(0.0),
                    new_m_act,
                ]
            )
        )

        # ── 6. Release attacker from mothership ───────────────────────────
        # Only release if: timer up, mothership active, n_active_attackers < 3, wave not done
        n_active_attackers = jnp.sum(enemies[1:, 5]).astype(jnp.int32)
        wave_not_done = state.wave_spawned < jnp.int32(_WAVE_SIZE)
        spawn_timer = state.spawn_timer - jnp.int32(1)
        do_release = (
            (spawn_timer <= jnp.int32(0))
            & (enemies[0, 5] > jnp.float32(0.5))
            & (n_active_attackers < jnp.int32(_MAX_ATTACKERS))
            & wave_not_done
        )

        spawn_rng, vx_rng, vy_rng, type_rng = jax.random.split(spawn_rng, 4)
        atck_vx = jax.random.uniform(
            vx_rng, (), minval=-_ATCK_VX_MAX, maxval=_ATCK_VX_MAX
        )
        atck_vy = jax.random.uniform(
            vy_rng, (), minval=_ATCK_VY_MIN, maxval=_ATCK_VY_MAX
        )
        # type: 1.0 = fighter (square), 2.0 = bomber (circle)
        atck_type = jnp.where(
            jax.random.uniform(type_rng) < 0.5, jnp.float32(1.0), jnp.float32(2.0)
        )

        # Find first free attacker slot (argmin of active among slots 1-3)
        free_slot = jnp.argmin(enemies[1:, 5]) + jnp.int32(1)
        new_atck = jnp.stack(
            [
                enemies[0, 0],  # spawn at mothership x
                jnp.float32(_MOTHER_Y + 12.0),  # just below mothership
                atck_vx,
                atck_vy,
                atck_type,
                jnp.float32(1.0),  # active
            ]
        )
        enemies = jnp.where(
            do_release,
            enemies.at[free_slot].set(new_atck),
            enemies,
        )
        wave_spawned = state.wave_spawned + jnp.where(
            do_release, jnp.int32(1), jnp.int32(0)
        )
        spawn_timer = jnp.where(
            do_release | (spawn_timer <= jnp.int32(0)),
            jnp.int32(_ATCK_RELEASE),
            spawn_timer,
        )
        # Mothership respawn timer: count down only when mothership is dead
        spawn_timer = jnp.where(
            (enemies[0, 5] < jnp.float32(0.5)) & ~m_respawn,
            jnp.where(
                state.spawn_timer > jnp.int32(0),
                state.spawn_timer - jnp.int32(1),
                jnp.int32(0),
            ),
            spawn_timer,
        )

        # ── 7. Move attackers (slots 1-3) — slow float + wiggle ──────────
        _GROUND_LEVEL = jnp.float32(_PLAYER_Y_INIT - 4.0)
        a_xs = enemies[1:, 0]
        a_ys = enemies[1:, 1]
        a_vxs = enemies[1:, 2]
        a_vys = enemies[1:, 3]
        a_typ = enemies[1:, 4]
        a_act = enemies[1:, 5]

        # Wiggle: random chance per attacker to reverse vx
        wiggle_rolls = jax.random.uniform(wiggle_rng, (_MAX_ATTACKERS,))
        do_wiggle = (wiggle_rolls < jnp.float32(_ATCK_WIGGLE_PROB)) & (
            a_act > jnp.float32(0.5)
        )
        new_a_vxs = jnp.where(do_wiggle, -a_vxs, a_vxs)

        new_a_xs = a_xs + new_a_vxs * a_act
        new_a_ys = a_ys + a_vys * a_act

        # Landing: clamp at ground level and stop vertical movement
        landed = (new_a_ys >= _GROUND_LEVEL) & (a_act > jnp.float32(0.5))
        new_a_ys = jnp.where(landed, _GROUND_LEVEL, new_a_ys)
        new_a_vys = jnp.where(landed, jnp.float32(0.0), a_vys)

        # Bounce at screen edges (attackers never leave)
        bounce_x = (
            (new_a_xs >= jnp.float32(_WORLD_W - 8.0)) | (new_a_xs <= jnp.float32(8.0))
        ) & (a_act > jnp.float32(0.5))
        new_a_vxs = jnp.where(bounce_x, -new_a_vxs, new_a_vxs)
        new_a_xs = jnp.clip(new_a_xs, jnp.float32(8.0), jnp.float32(_WORLD_W - 8.0))

        attackers = jnp.stack(
            [new_a_xs, new_a_ys, new_a_vxs, new_a_vys, a_typ, a_act], axis=-1
        )
        enemies = enemies.at[1:].set(attackers)

        # ── 8. Enemy fire (attackers only) ────────────────────────────────
        fire_rngs = jax.random.split(fire_rng, _MAX_ATTACKERS)
        fire_rolls = jax.vmap(lambda k: jax.random.uniform(k))(fire_rngs)
        fires = (fire_rolls < jnp.float32(_ATCK_FIRE_PROB)) & (
            enemies[1:, 5] > jnp.float32(0.5)
        )
        any_fires = jnp.any(fires)
        firing_idx = jnp.argmax(fires) + jnp.int32(1)
        eb_slot = jnp.argmin(state.enemy_bullets[:, 2])
        eb_free = state.enemy_bullets[eb_slot, 2] < jnp.float32(0.5)
        new_eb = jnp.stack(
            [
                enemies[firing_idx, 0],
                enemies[firing_idx, 1] + jnp.float32(6.0),
                jnp.float32(1.0),
            ]
        )
        enemy_bullets = jnp.where(
            any_fires & eb_free,
            state.enemy_bullets.at[eb_slot].set(new_eb),
            state.enemy_bullets,
        )
        enemy_bullets = self._move_bullets(
            enemy_bullets, jnp.float32(_ENEMY_BULLET_SPEED)
        )

        # ── 9. Player bullets vs enemies ──────────────────────────────────
        en_xs = enemies[:, 0]
        en_ys = enemies[:, 1]
        en_act = enemies[:, 5]

        # Mothership: smaller hit-box
        hit_mat_m = self._bullet_rect_hits(
            player_bullets,
            en_xs,
            en_ys,
            jnp.float32(_MOTHER_HIT_HW),
            jnp.float32(_MOTHER_HIT_HH),
        )  # (1, 4)
        # Attackers: fighter hit-box
        hit_mat_a = self._bullet_rect_hits(
            player_bullets, en_xs, en_ys, jnp.float32(_ATCK_HW), jnp.float32(_ATCK_HH)
        )  # (1, 4)

        is_mother = (enemies[:, 4] < jnp.float32(0.5))[None, :]
        is_atck = (enemies[:, 4] > jnp.float32(0.5))[None, :]
        hit_mat = (hit_mat_m & is_mother) | (hit_mat_a & is_atck)
        hit_mat = (
            hit_mat
            & (player_bullets[:, 2][:, None] > jnp.float32(0.5))
            & (en_act[None, :] > jnp.float32(0.5))
        )

        enemy_hit = jnp.any(hit_mat, axis=0)  # (4,)
        bullet_hit = jnp.any(hit_mat, axis=1)  # (1,)

        # Mothership HP
        mother_hit_this_frame = enemy_hit[0] & (enemies[0, 5] > jnp.float32(0.5))
        new_mother_hp = state.mother_hp - jnp.where(
            mother_hit_this_frame, jnp.int32(1), jnp.int32(0)
        )
        mother_killed = mother_hit_this_frame & (new_mother_hp <= jnp.int32(0))

        # Attackers deactivate on hit
        atck_hit = enemy_hit[1:]  # (3,)
        new_atck_act = enemies[1:, 5] * (~atck_hit).astype(jnp.float32)
        enemies = enemies.at[1:, 5].set(new_atck_act)
        enemies = enemies.at[0, 5].set(
            jnp.where(mother_killed, jnp.float32(0.0), enemies[0, 5])
        )

        player_bullets = player_bullets.at[:, 2].set(
            player_bullets[:, 2] * (~bullet_hit).astype(jnp.float32)
        )

        # Mothership kill: start respawn timer, reset HP
        spawn_timer = jnp.where(mother_killed, jnp.int32(_MOTHER_RESPAWN), spawn_timer)
        new_mother_hp = jnp.where(mother_killed, jnp.int32(_MOTHER_HP), new_mother_hp)

        # Score
        score_delta = jnp.sum(atck_hit).astype(jnp.int32) * jnp.int32(
            _SCORE_ATTACKER
        ) + jnp.where(mother_killed, jnp.int32(_SCORE_MOTHER), jnp.int32(0))
        reward = score_delta.astype(jnp.float32)

        # ── 10. Enemy bullets vs player ───────────────────────────────────
        ph_mat = self._bullet_rect_hits(
            enemy_bullets,
            jnp.array([player_x]),
            jnp.array([player_y]),
            jnp.float32(_PLAYER_HW),
            jnp.float32(_PLAYER_HH),
        )  # (6, 1)
        bullet_hit_player = jnp.any(ph_mat)
        enemy_bullets = enemy_bullets.at[:, 2].set(
            enemy_bullets[:, 2] * (~jnp.any(ph_mat, axis=1)).astype(jnp.float32)
        )

        # Overheat also kills player
        player_hit = bullet_hit_player | overheated
        new_lives = state.lives - jnp.where(player_hit, jnp.int32(1), jnp.int32(0))
        # Reset player and heat on death
        player_x = jnp.where(player_hit, jnp.float32(80.0), player_x)
        player_y = jnp.where(player_hit, jnp.float32(_PLAYER_Y_INIT), player_y)
        heat = jnp.where(player_hit, jnp.int32(0), heat)
        # Clear all bullets on death
        player_bullets = jnp.where(
            player_hit, jnp.zeros_like(player_bullets), player_bullets
        )
        enemy_bullets = jnp.where(
            player_hit, jnp.zeros_like(enemy_bullets), enemy_bullets
        )

        # ── 11. Wave advance ──────────────────────────────────────────────
        # Wave is done when all 10 have been sent AND all active attackers are killed
        wave_complete = (wave_spawned >= jnp.int32(_WAVE_SIZE)) & (
            jnp.sum(enemies[1:, 5]) < jnp.float32(0.5)
        )
        new_wave = jnp.where(wave_complete, state.wave + jnp.int32(1), state.wave)
        wave_spawned = jnp.where(wave_complete, jnp.int32(0), wave_spawned)
        # On wave clear: respawn mothership, reset HP
        enemies = jnp.where(wave_complete, _ENEMIES_INIT, enemies)
        new_mother_hp = jnp.where(wave_complete, jnp.int32(_MOTHER_HP), new_mother_hp)
        spawn_timer = jnp.where(wave_complete, jnp.int32(_ATCK_RELEASE), spawn_timer)

        # ── 12. Done ──────────────────────────────────────────────────────
        done = (new_lives <= jnp.int32(0)) | (
            state.step + jnp.int32(1) >= jnp.int32(params.max_steps)
        )

        return state.__replace__(
            player_x=player_x,
            player_y=player_y,
            fire_cooldown=fire_cooldown,
            player_bullets=player_bullets,
            enemy_bullets=enemy_bullets,
            enemies=enemies,
            spawn_timer=spawn_timer,
            wave=new_wave,
            wave_spawned=wave_spawned,
            mother_hp=new_mother_hp,
            bullet_vx=bullet_vx,
            heat=heat,
            lives=new_lives,
            score=state.score + score_delta,
            reward=state.reward + reward,
            done=done,
            step=state.step + jnp.int32(1),
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: AssaultState,
        action: chex.Array,
        params: AssaultParams,
    ) -> AssaultState:
        """Advance the game by one agent step (4 emulated frames)."""
        state = state.__replace__(reward=jnp.float32(0.0))

        def physics_step(i: int, s: AssaultState) -> AssaultState:
            return self._step_physics(s, action, params, jax.random.fold_in(rng, i))

        state = jax.lax.fori_loop(0, 4, physics_step, state)
        return state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: AssaultState) -> chex.Array:
        """
        Render the current game state as an RGB frame.

        Returns
        -------
        frame : chex.Array
            uint8[210, 160, 3] — RGB image.
        """
        canvas = make_canvas(_COL_BG)

        # Layer 1 — Ground bar (cyan, full width)
        canvas = paint_sdf(
            canvas,
            sdf_rect(
                jnp.float32(80.0),
                jnp.float32(_GROUND_Y),
                jnp.float32(80.0),
                jnp.float32(2.0),
            ),
            _COL_GROUND,
        )

        # Layer 2a — Fighter attackers (red squares, type == 1.0)
        atck_xs = state.enemies[1:, 0]
        atck_ys = state.enemies[1:, 1]
        atck_types = state.enemies[1:, 4]
        atck_active = state.enemies[1:, 5]
        fighter_act = atck_active * (atck_types < jnp.float32(1.5)).astype(jnp.float32)
        bomber_act = atck_active * (atck_types > jnp.float32(1.5)).astype(jnp.float32)
        fighter_pool = jnp.stack([atck_xs, atck_ys, fighter_act], axis=-1)
        bomber_pool = jnp.stack([atck_xs, atck_ys, bomber_act], axis=-1)
        canvas = paint_layer(
            canvas, render_rect_pool(fighter_pool, _ATCK_HW, _ATCK_HH), _COL_FIGHTER
        )

        # Layer 2b — Bomber attackers (orange circles, type == 2.0)
        canvas = paint_layer(
            canvas, render_circle_pool(bomber_pool, _ATCK_RADIUS), _COL_BOMBER
        )

        # Layer 3 — Mothership (green rect + inner dome hole)
        m_x = state.enemies[0, 0]
        m_y = state.enemies[0, 1]
        m_act = state.enemies[0, 5]
        m_pool = jnp.stack([m_x[None], m_y[None], m_act[None]], axis=-1)
        canvas = paint_layer(
            canvas, render_rect_pool(m_pool, _MOTHER_HW, _MOTHER_HH), _COL_MOTHER
        )
        canvas = paint_layer(canvas, render_circle_pool(m_pool, 5.0), _COL_BG)

        # Layer 4 — Enemy bullets (orange circles)
        canvas = paint_layer(
            canvas, render_circle_pool(state.enemy_bullets, 2.0), _COL_EBULLET
        )

        # Layer 5 — Player bullets (yellow slim rect)
        canvas = paint_layer(
            canvas, render_rect_pool(state.player_bullets, 1.0, 3.0), _COL_PBULLET
        )

        # Layer 6 — Player turret (blue rect + barrel nose)
        canvas = paint_sdf(
            canvas,
            sdf_rect(
                state.player_x,
                state.player_y,
                jnp.float32(_PLAYER_HW),
                jnp.float32(_PLAYER_HH),
            ),
            _COL_PLAYER,
        )
        canvas = paint_sdf(
            canvas,
            sdf_rect(
                state.player_x,
                state.player_y - jnp.float32(5.0),
                jnp.float32(2.0),
                jnp.float32(2.0),
            ),
            _COL_PLAYER,
        )

        # Layer 7 — Heat bar (green, bottom-left, fills as heat increases)
        heat_frac = state.heat.astype(jnp.float32) / jnp.float32(_MAX_HEAT)
        heat_bar_hw = jnp.float32(20.0) * heat_frac  # max 20px half-width (40px bar)
        heat_bar_cx = jnp.float32(10.0) + heat_bar_hw  # left-anchored
        canvas = paint_sdf(
            canvas,
            sdf_rect(heat_bar_cx, jnp.float32(200.0), heat_bar_hw, jnp.float32(2.0)),
            _COL_HEAT,
        )

        # Layer 8 — Score HUD
        canvas = render_score(canvas, state.score)

        # Layer 9 — Life pips (small blue turret icons)
        canvas = render_life_pips(
            canvas,
            state.lives,
            pip_sdf_fn=lambda cx, cy: sdf_rect(
                jnp.float32(cx), jnp.float32(cy), jnp.float32(3.0), jnp.float32(2.5)
            ),
            pip_colour=_COL_HUD,
        )

        return finalise_rgb(canvas)
