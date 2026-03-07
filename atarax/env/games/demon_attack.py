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

"""Demon Attack — JAX-native SDF game implementation.

Up to 3 demons are visible simultaneously, one per height zone (top/mid/bottom).
They move horizontally across the screen, spawning from the left or right edge.
Enemy bullets split into two diagonal shots at mid-screen.

All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.

Action space (6 actions, matching ALE):
    0 — NOOP
    1 — FIRE
    2 — RIGHT
    3 — LEFT
    4 — RIGHTFIRE
    5 — LEFTFIRE
"""

from typing import ClassVar

import chex
import jax
import jax.numpy as jnp

from atarax.env._base.fixed_shooter import FixedShooterGame, FixedShooterState
from atarax.env.hud import render_life_pips, render_score
from atarax.env.sdf import (
    finalise_rgb,
    make_canvas,
    paint_layer,
    paint_sdf,
    render_rect_pool,
    sdf_rect,
    sdf_ship_triangle,
    sdf_triangle,
    sdf_union,
)
from atarax.game import AtaraxParams

# ── Geometry ─────────────────────────────────────────────────────────────────
_WORLD_W: float = 160.0
_WORLD_H: float = 210.0

_PLAYER_HW: float = 5.0
_PLAYER_HH: float = 4.0
_PLAYER_SPEED: float = 2.0
_PLAYER_X_MIN: float = 8.0
_PLAYER_X_MAX: float = 152.0
_PLAYER_Y: float = 178.0
_FIRE_COOLDOWN: int = 8
_BULLET_SPEED: float = 6.0   # upward

# ── Demon geometry ────────────────────────────────────────────────────────────
# 3 zone heights: top / middle / bottom
_NUM_ZONES: int = 3
_ZONE_YS_PY: tuple[float, ...] = (45.0, 78.0, 112.0)
_ZONE_YS = jnp.array(_ZONE_YS_PY, dtype=jnp.float32)

# Big demon (slots 0-2 of enemies array)
_DEMON_HW_BIG: float = 8.0
_DEMON_HH_BIG: float = 7.0
# Small demon (split; slots 3-5)
_DEMON_HW_SM: float = 5.0
_DEMON_HH_SM: float = 4.5

# Enemies array: (6, 5) float32  [x, y, vx, size, active]
#   size: 1.0 = big,  0.0 = small (split)
_MAX_ENEMIES: int = 6

# Demon horizontal speed (big)
_DEMON_SPEED: float = 1.5      # px/frame
# Small demons move faster (diverge from kill point)
_DEMON_SPEED_SM: float = 2.0

# Respawn delay after a zone demon is killed
_SPAWN_DELAY: int = 50

# Wave advance: kills required per wave
_WAVE_SIZE: int = 12

# Split: available from wave 5+; killed big demon → 2 small demons
_SPLIT_WAVE: int = 5

# ── Enemy bullets ─────────────────────────────────────────────────────────────
_MAX_ENEMY_BULLETS: int = 12          # larger pool for burst spreads
_MAX_BURST: int = 8                   # max shots per burst
_VX_SPREAD: float = 2.0              # half-spread of burst fan (px/frame)
_DEMON_BULLET_SPEED: float = 2.8
_DEMON_FIRE_PROB: float = 0.018      # per-frame burst probability (lowest demon only)

# ── Scoring ───────────────────────────────────────────────────────────────────
_SCORE_BASE: int = 10
_SCORE_PER_WAVE: int = 5
_SCORE_SM_MULT: float = 2.0          # small demons worth 2x

# ── Demon wave colours (index = wave // 2 % 4) ───────────────────────────────
_DEMON_WAVE_COLS = jnp.array(
    [
        [0.90, 0.20, 0.90],   # waves 0-1  purple
        [1.00, 0.55, 0.10],   # waves 2-3  orange
        [0.20, 0.90, 0.90],   # waves 4-5  cyan
        [1.00, 0.25, 0.25],   # waves 6+   red
    ],
    dtype=jnp.float32,
)

# ── Other colours ─────────────────────────────────────────────────────────────
_COL_BG = jnp.array([0.02, 0.00, 0.05], dtype=jnp.float32)
_COL_GROUND = jnp.array([0.14, 0.26, 0.52], dtype=jnp.float32)        # dark blue ground strip
_COL_GROUND_EDGE = jnp.array([0.42, 0.62, 0.90], dtype=jnp.float32)   # lighter highlight rim
_COL_PLAYER = jnp.array([0.20, 1.00, 0.35], dtype=jnp.float32)
_COL_PBULLET = jnp.array([1.00, 1.00, 0.20], dtype=jnp.float32)
_COL_EBULLET = jnp.array([1.00, 0.35, 0.10], dtype=jnp.float32)
_COL_HUD = jnp.array([0.20, 1.00, 0.35], dtype=jnp.float32)

# ── Initial state ─────────────────────────────────────────────────────────────
# 3 big zone demons alternating sides, 3 inactive split slots
_INIT_ENEMIES = jnp.array(
    [
        # zone 0 (top, y=45): starts from left, moves right
        [8.0,   _ZONE_YS_PY[0],  _DEMON_SPEED, 1.0, 1.0],
        # zone 1 (mid, y=78): starts from right, moves left
        [152.0, _ZONE_YS_PY[1], -_DEMON_SPEED, 1.0, 1.0],
        # zone 2 (bot, y=112): starts from left, moves right
        [8.0,   _ZONE_YS_PY[2],  _DEMON_SPEED, 1.0, 1.0],
        # split slots (inactive)
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=jnp.float32,
)


@chex.dataclass
class DemonAttackParams(AtaraxParams):
    """Static configuration for Demon Attack."""

    max_steps: int = 10000
    num_lives: int = 3


@chex.dataclass
class DemonAttackState(FixedShooterState):
    """
    Demon Attack game state.

    Extends `FixedShooterState`.  Field mapping:
    - `player_x`        → cannon x (horizontal only; y fixed at _PLAYER_Y)
    - `fire_cooldown`   → player fire cooldown
    - `enemy_grid`      → stubbed (1, 1)
    - `fleet_*`         → stubbed (not used)
    - `player_bullets`  → (1, 3) float32 [x, y, active]
    - `enemy_bullets`   → (6, 3) float32 [x, y, active]

    Extra fields:
    - `enemies`         → (6, 5) float32 [x, y, vx, size, active]
                          slots 0-2: zone demons (size=1 big)
                          slots 3-5: split demons (size=0 small)
    - `spawn_timer`     → (3,) int32 per-zone respawn countdown
    - `kills_in_wave`   → int32 kills toward wave advance
    - `enemy_bvx`       → (6,) float32 bullet x velocity
    - `bullet_split`    → (6,) bool bullet has split
    - `wave`            → int32 current wave (0-indexed)
    """

    enemies: chex.Array         # (6, 5) float32
    spawn_timer: chex.Array     # (3,) int32
    kills_in_wave: chex.Array   # () int32
    enemy_bvx: chex.Array       # (6,) float32
    bullet_split: chex.Array    # (6,) bool
    wave: chex.Array            # () int32


class DemonAttack(FixedShooterGame):
    """
    Demon Attack implemented as a pure-JAX function suite.

    All conditionals use `jnp.where`; the 4-frame skip uses `jax.lax.fori_loop`.
    """

    num_actions: ClassVar[int] = 6
    game_id: ClassVar[str] = "demon_attack"

    def _reset(self, rng: chex.PRNGKey) -> DemonAttackState:
        return DemonAttackState(
            # FixedShooterState fields (fleet/grid stubbed)
            player_x=jnp.float32(80.0),
            fire_cooldown=jnp.int32(0),
            enemy_grid=jnp.zeros((1, 1), dtype=jnp.bool_),
            fleet_x=jnp.float32(0.0),
            fleet_y=jnp.float32(0.0),
            fleet_dir=jnp.int32(0),
            fleet_speed=jnp.float32(0.0),
            player_bullets=jnp.zeros((1, 3), dtype=jnp.float32),
            enemy_bullets=jnp.zeros((_MAX_ENEMY_BULLETS, 3), dtype=jnp.float32),
            # DemonAttackState extra fields
            enemies=_INIT_ENEMIES,
            spawn_timer=jnp.zeros(_NUM_ZONES, dtype=jnp.int32),
            kills_in_wave=jnp.int32(0),
            enemy_bvx=jnp.zeros(_MAX_ENEMY_BULLETS, dtype=jnp.float32),
            bullet_split=jnp.zeros(_MAX_ENEMY_BULLETS, dtype=jnp.bool_),
            wave=jnp.int32(0),
            # AtariState fields
            lives=jnp.int32(3),
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
        state: DemonAttackState,
        action: chex.Array,
        params: DemonAttackParams,
        rng: chex.PRNGKey,
    ) -> DemonAttackState:
        """Advance the game by one emulated frame (branch-free)."""

        rng, fire_rng, spawn_rng = jax.random.split(rng, 3)

        # ── 1. Player movement ────────────────────────────────────────────
        move_right = (action == jnp.int32(2)) | (action == jnp.int32(4))
        move_left = (action == jnp.int32(3)) | (action == jnp.int32(5))
        dx = jnp.where(
            move_right, jnp.float32(_PLAYER_SPEED),
            jnp.where(move_left, jnp.float32(-_PLAYER_SPEED), jnp.float32(0.0)),
        )
        player_x = jnp.clip(
            state.player_x + dx,
            jnp.float32(_PLAYER_X_MIN),
            jnp.float32(_PLAYER_X_MAX),
        )

        # ── 2. Fire player bullet ─────────────────────────────────────────
        want_fire = (action == jnp.int32(1)) | (action == jnp.int32(4)) | (action == jnp.int32(5))
        can_fire = want_fire & (state.fire_cooldown <= jnp.int32(0))
        pb_free = state.player_bullets[0, 2] < jnp.float32(0.5)
        do_fire = can_fire & pb_free
        new_pb = jnp.stack([player_x, jnp.float32(_PLAYER_Y - 6.0), jnp.float32(1.0)])
        player_bullets = jnp.where(
            do_fire, state.player_bullets.at[0].set(new_pb), state.player_bullets
        )
        fire_cooldown = jnp.where(
            do_fire, jnp.int32(_FIRE_COOLDOWN),
            jnp.maximum(state.fire_cooldown - jnp.int32(1), jnp.int32(0)),
        )
        player_bullets = self._move_bullets(player_bullets, jnp.float32(-_BULLET_SPEED))

        # ── 3. Move demons ────────────────────────────────────────────────
        en_x = state.enemies[:, 0]
        en_y = state.enemies[:, 1]
        en_vx = state.enemies[:, 2]
        en_size = state.enemies[:, 3]
        en_act = state.enemies[:, 4]

        new_en_x = en_x + en_vx * en_act
        # Bounce at screen edges
        hit_left = (new_en_x < jnp.float32(5.0)) & (en_act > jnp.float32(0.5))
        hit_right = (new_en_x > jnp.float32(155.0)) & (en_act > jnp.float32(0.5))
        new_en_vx = jnp.where(hit_left | hit_right, -en_vx, en_vx)
        new_en_x = jnp.clip(new_en_x, jnp.float32(5.0), jnp.float32(155.0))

        enemies = jnp.stack([new_en_x, en_y, new_en_vx, en_size, en_act], axis=1)

        # ── 4. Player bullet vs demon collision ───────────────────────────
        en_xs = enemies[:, 0]
        en_ys = enemies[:, 1]
        en_is_big = enemies[:, 3] > jnp.float32(0.5)   # (6,) bool

        # Separate hitboxes for big and small
        pb_arr = player_bullets  # (1, 3)
        hit_big = self._bullet_rect_hits(
            pb_arr, en_xs, en_ys,
            jnp.float32(_DEMON_HW_BIG), jnp.float32(_DEMON_HH_BIG)
        )  # (1, 6)
        hit_sm = self._bullet_rect_hits(
            pb_arr, en_xs, en_ys,
            jnp.float32(_DEMON_HW_SM), jnp.float32(_DEMON_HH_SM)
        )  # (1, 6)
        # Use appropriate hitbox per demon type
        hit_mat = jnp.where(en_is_big[None, :], hit_big, hit_sm)  # (1, 6)

        # Active check
        active_mask = enemies[:, 4] > jnp.float32(0.5)
        killed = jnp.any(hit_mat, axis=0) & active_mask  # (6,) bool

        n_big_killed = jnp.sum((killed & en_is_big).astype(jnp.int32))
        n_sm_killed = jnp.sum((killed & ~en_is_big).astype(jnp.int32))
        base_pts = _SCORE_BASE + state.wave * _SCORE_PER_WAVE
        score_gain = (
            n_big_killed * base_pts
            + n_sm_killed * jnp.int32(jnp.round(jnp.float32(base_pts) * _SCORE_SM_MULT))
        )

        # Deactivate player bullet if hit
        pb_hit = jnp.any(hit_mat & active_mask[None, :], axis=1)[0]
        player_bullets = player_bullets.at[0, 2].set(
            player_bullets[0, 2] * (~pb_hit).astype(jnp.float32)
        )

        # ── 5. Handle kills ───────────────────────────────────────────────
        # Deactivate killed demons
        new_en_act = enemies[:, 4] * (~killed).astype(jnp.float32)
        enemies = enemies.at[:, 4].set(new_en_act)

        # For big demons killed in split-eligible wave (wave >= SPLIT_WAVE):
        # spawn 2 small demons in slots 3-5
        can_split = (state.wave >= jnp.int32(_SPLIT_WAVE))
        big_killed = killed & en_is_big  # (6,)

        for i in range(_NUM_ZONES):   # only zone slots 0-2 are big
            do_split_i = big_killed[i] & can_split
            # Find 2 free split slots (3-5)
            split_free = enemies[3:, 4] < jnp.float32(0.5)  # (3,) bool

            slot_a = jnp.argmax(split_free)  # first free among 3-5
            slot_b = jnp.argmax(split_free.at[slot_a].set(False))  # second free

            abs_a = slot_a + jnp.int32(3)
            abs_b = slot_b + jnp.int32(3)
            has_two = jnp.sum(split_free.astype(jnp.int32)) >= jnp.int32(2)

            kx = enemies[i, 0]
            ky = enemies[i, 1]

            # Spawn two small demons diverging left/right
            new_sm_a = jnp.stack([kx - jnp.float32(6.0), ky, -jnp.float32(_DEMON_SPEED_SM), jnp.float32(0.0), jnp.float32(1.0)])
            new_sm_b = jnp.stack([kx + jnp.float32(6.0), ky, +jnp.float32(_DEMON_SPEED_SM), jnp.float32(0.0), jnp.float32(1.0)])

            mask_a = (jnp.arange(_MAX_ENEMIES) == abs_a)
            mask_b = (jnp.arange(_MAX_ENEMIES) == abs_b)

            enemies = jnp.where(
                (do_split_i & has_two) & mask_a[:, None],
                jnp.tile(new_sm_a[None, :], (_MAX_ENEMIES, 1)),
                enemies,
            )
            enemies = jnp.where(
                (do_split_i & has_two) & mask_b[:, None],
                jnp.tile(new_sm_b[None, :], (_MAX_ENEMIES, 1)),
                enemies,
            )

        # ── 6. Zone respawn logic (slots 0-2) ────────────────────────────
        spawn_timer = jnp.maximum(state.spawn_timer - jnp.int32(1), jnp.int32(0))

        # When a zone demon is killed, start its respawn timer
        for z in range(_NUM_ZONES):
            killed_z = killed[z]
            spawn_timer = spawn_timer.at[z].set(
                jnp.where(killed_z, jnp.int32(_SPAWN_DELAY), spawn_timer[z])
            )

        # Spawn new zone demon when slot empty and timer expired
        spawn_keys = jax.random.split(spawn_rng, _NUM_ZONES)
        for z in range(_NUM_ZONES):
            slot_empty = enemies[z, 4] < jnp.float32(0.5)
            do_spawn = slot_empty & (spawn_timer[z] <= jnp.int32(0))
            from_left = jax.random.randint(spawn_keys[z], (), 0, 2) == jnp.int32(0)
            sx = jnp.where(from_left, jnp.float32(8.0), jnp.float32(152.0))
            svx = jnp.where(from_left, jnp.float32(_DEMON_SPEED), jnp.float32(-_DEMON_SPEED))
            new_demon = jnp.stack([sx, _ZONE_YS[z], svx, jnp.float32(1.0), jnp.float32(1.0)])
            mask = (jnp.arange(_MAX_ENEMIES) == jnp.int32(z))
            enemies = jnp.where(
                do_spawn & mask[:, None],
                jnp.tile(new_demon[None, :], (_MAX_ENEMIES, 1)),
                enemies,
            )

        # ── 7. Kills, wave advance, life gain ────────────────────────────────
        n_killed_total = jnp.sum(killed.astype(jnp.int32))
        kills_in_wave = state.kills_in_wave + n_killed_total
        wave_done = kills_in_wave >= jnp.int32(_WAVE_SIZE)
        new_wave = jnp.where(wave_done, state.wave + jnp.int32(1), state.wave)
        kills_in_wave = jnp.where(wave_done, jnp.int32(0), kills_in_wave)
        # +1 life on wave completion, capped at 6
        pre_hit_lives = jnp.where(
            wave_done,
            jnp.minimum(state.lives + jnp.int32(1), jnp.int32(6)),
            state.lives,
        )

        # ── 8. Move enemy bullets (Y via helper, X inline) ────────────────
        enemy_bullets = self._move_bullets(state.enemy_bullets, jnp.float32(_DEMON_BULLET_SPEED))
        new_eb_x = enemy_bullets[:, 0] + state.enemy_bvx * enemy_bullets[:, 2]
        x_oob = (new_eb_x < jnp.float32(0.0)) | (new_eb_x > jnp.float32(_WORLD_W))
        new_eb_active = enemy_bullets[:, 2] * (~x_oob).astype(jnp.float32)
        enemy_bullets = jnp.stack([new_eb_x, enemy_bullets[:, 1], new_eb_active], axis=1)
        enemy_bvx = state.enemy_bvx

        # ── 9. Lowest demon fires a burst-spread of bullets ────────────────
        # Only the demon closest to the ground fires (highest y among active)
        active_mask = enemies[:, 4] > jnp.float32(0.5)
        ys_for_lowest = jnp.where(active_mask, enemies[:, 1], jnp.float32(-1.0))
        lowest_idx = jnp.argmax(ys_for_lowest)
        shooter_alive = active_mask[lowest_idx]

        n_eb_active = jnp.sum(enemy_bullets[:, 2] > jnp.float32(0.5)).astype(jnp.int32)
        can_fire_enemy = n_eb_active <= jnp.int32(_MAX_ENEMY_BULLETS - _MAX_BURST)
        fire_roll = jax.random.uniform(fire_rng) < jnp.float32(_DEMON_FIRE_PROB)
        do_fire_enemy = can_fire_enemy & fire_roll & shooter_alive

        shooter_x = enemies[lowest_idx, 0]
        shooter_y = enemies[lowest_idx, 1] + jnp.float32(_DEMON_HH_BIG)

        # Wave-scaled burst size: 2 shots at wave 0, +1 per wave, max _MAX_BURST
        n_shots = jnp.minimum(jnp.int32(2) + new_wave, jnp.int32(_MAX_BURST))

        # Spawn burst — unrolled; each slot gets a unique vx in the fan
        bullet_split = enemy_bullets[:, 2] > jnp.float32(0.5)  # active = already "split"
        for b in range(_MAX_BURST):
            # vx: evenly spaced from -_VX_SPREAD to +_VX_SPREAD
            t = jnp.float32(b) / jnp.float32(_MAX_BURST - 1) * jnp.float32(2.0) - jnp.float32(1.0)
            vx_b = t * jnp.float32(_VX_SPREAD)
            in_burst = jnp.int32(b) < n_shots

            is_free_eb = enemy_bullets[:, 2] < jnp.float32(0.5)
            eb_slot = jnp.argmax(is_free_eb)
            has_free = jnp.any(is_free_eb)
            slot_mask = jnp.arange(_MAX_ENEMY_BULLETS) == eb_slot

            new_eb_bullet = jnp.stack([shooter_x, shooter_y, jnp.float32(1.0)])
            do_spawn_b = do_fire_enemy & in_burst & has_free

            enemy_bullets = jnp.where(
                do_spawn_b & slot_mask[:, None],
                jnp.tile(new_eb_bullet[None, :], (_MAX_ENEMY_BULLETS, 1)),
                enemy_bullets,
            )
            enemy_bvx = jnp.where(do_spawn_b & slot_mask, vx_b, enemy_bvx)
            bullet_split = jnp.where(do_spawn_b & slot_mask, True, bullet_split)

        # ── 10. Enemy bullet vs player ─────────────────────────────────────
        eb_xs = enemy_bullets[:, 0]
        eb_ys = enemy_bullets[:, 1]
        eb_act = enemy_bullets[:, 2]
        player_hit = jnp.any(
            (jnp.abs(eb_xs - player_x) < jnp.float32(_PLAYER_HW + 2.0))
            & (jnp.abs(eb_ys - jnp.float32(_PLAYER_Y)) < jnp.float32(_PLAYER_HH + 3.0))
            & (eb_act > jnp.float32(0.5))
        )
        new_lives = jnp.where(player_hit, pre_hit_lives - jnp.int32(1), pre_hit_lives)

        # Clear all bullets on player hit (prevents multi-frame re-triggering)
        enemy_bullets = jnp.where(player_hit, jnp.zeros_like(enemy_bullets), enemy_bullets)
        enemy_bvx = jnp.where(player_hit, jnp.zeros_like(enemy_bvx), enemy_bvx)
        bullet_split = jnp.where(player_hit, jnp.zeros_like(bullet_split), bullet_split)

        # ── 12. Done ──────────────────────────────────────────────────────
        done = (new_lives <= jnp.int32(0)) | (
            state.step + jnp.int32(1) >= jnp.int32(params.max_steps)
        )

        return state.__replace__(
            player_x=player_x,
            fire_cooldown=fire_cooldown,
            player_bullets=player_bullets,
            enemy_bullets=enemy_bullets,
            enemies=enemies,
            spawn_timer=spawn_timer,
            kills_in_wave=kills_in_wave,
            enemy_bvx=enemy_bvx,
            bullet_split=bullet_split,
            wave=new_wave,
            lives=new_lives,
            score=state.score + jnp.int32(score_gain),
            reward=state.reward + jnp.float32(score_gain),
            done=done,
            step=state.step + jnp.int32(1),
        )

    def _step(
        self,
        rng: chex.PRNGKey,
        state: DemonAttackState,
        action: chex.Array,
        params: DemonAttackParams,
    ) -> DemonAttackState:
        state = state.__replace__(reward=jnp.float32(0.0))

        def physics_step(i: int, s: DemonAttackState) -> DemonAttackState:
            return self._step_physics(s, action, params, jax.random.fold_in(rng, i))

        state = jax.lax.fori_loop(0, 4, physics_step, state)
        return state.__replace__(episode_step=state.episode_step + jnp.int32(1))

    def render(self, state: DemonAttackState) -> chex.Array:
        """
        Render the current game state as an RGB frame.

        Returns
        -------
        frame : chex.Array
            uint8[210, 160, 3] — RGB image.
        """
        canvas = make_canvas(_COL_BG)

        # ── Blue ground strip (ALE-style) + highlight rim ─────────────────
        canvas = paint_sdf(
            canvas,
            sdf_rect(jnp.float32(80.0), jnp.float32(202.5), jnp.float32(80.0), jnp.float32(8.5)),
            _COL_GROUND,
        )
        # Thin lighter edge at the top of the ground strip
        canvas = paint_sdf(
            canvas,
            sdf_rect(jnp.float32(80.0), jnp.float32(193.5), jnp.float32(80.0), jnp.float32(1.0)),
            _COL_GROUND_EDGE,
        )

        # ── Demon colour (cycles every 2 waves) ───────────────────────────
        wave_idx = jnp.clip(state.wave // jnp.int32(2), jnp.int32(0), jnp.int32(3))
        demon_col = _DEMON_WAVE_COLS[wave_idx]
        sm_col = _DEMON_WAVE_COLS[jnp.minimum(wave_idx + jnp.int32(1), jnp.int32(3))]

        # ── Demons — V-wing shape per demon ───────────────────────────────
        # Each demon: two wing triangles + center body, inactive → SDF = +inf
        for i in range(_MAX_ENEMIES):
            is_big = state.enemies[i, 3] > jnp.float32(0.5)
            active = state.enemies[i, 4] > jnp.float32(0.5)
            cx = state.enemies[i, 0]
            cy = state.enemies[i, 1]
            hw = jnp.where(is_big, jnp.float32(_DEMON_HW_BIG), jnp.float32(_DEMON_HW_SM))
            hh = jnp.where(is_big, jnp.float32(_DEMON_HH_BIG), jnp.float32(_DEMON_HH_SM))
            col = jnp.where(is_big, demon_col, sm_col)

            # Wings: two triangles spreading from top-centre outward
            lw = sdf_triangle(
                cx, cy - hh * jnp.float32(0.4),        # top centre
                cx - hw, cy + hh * jnp.float32(0.5),   # far left
                cx - hw * jnp.float32(0.35), cy + hh,  # inner left tip
            )
            rw = sdf_triangle(
                cx, cy - hh * jnp.float32(0.4),        # top centre
                cx + hw, cy + hh * jnp.float32(0.5),   # far right
                cx + hw * jnp.float32(0.35), cy + hh,  # inner right tip
            )
            body = sdf_rect(cx, cy + hh * jnp.float32(0.15), jnp.float32(2.2), hh * jnp.float32(0.65))
            shape = sdf_union(body, sdf_union(lw, rw))
            shape = jnp.where(active, shape, jnp.full_like(shape, 1.0e6))
            canvas = paint_sdf(canvas, shape, col)

        # ── Enemy bullets — thin small rectangles ─────────────────────────
        eb_mask = render_rect_pool(state.enemy_bullets, 1.0, 2.5)
        canvas = paint_layer(canvas, eb_mask, _COL_EBULLET)

        # ── Player bullet ──────────────────────────────────────────────────
        pb_mask = render_rect_pool(state.player_bullets, 1.5, 4.0)
        canvas = paint_layer(canvas, pb_mask, _COL_PBULLET)

        # ── Player ship: nose triangle + wide base ─────────────────────────
        nose = sdf_ship_triangle(
            state.player_x, jnp.float32(_PLAYER_Y - 3.0),
            jnp.float32(-jnp.pi / 2.0),
            size=5.0,
        )
        base = sdf_rect(
            state.player_x, jnp.float32(_PLAYER_Y + 3.0),
            jnp.float32(_PLAYER_HW + 1.5), jnp.float32(3.0),
        )
        canvas = paint_sdf(canvas, sdf_union(nose, base), _COL_PLAYER)

        # ── HUD ───────────────────────────────────────────────────────────
        canvas = render_score(canvas, state.score)
        canvas = render_life_pips(
            canvas,
            state.lives,
            pip_sdf_fn=lambda cx, cy: sdf_rect(
                jnp.float32(cx), jnp.float32(cy), jnp.float32(3.0), jnp.float32(2.5)
            ),
            pip_colour=_COL_HUD,
        )

        return finalise_rgb(canvas)
