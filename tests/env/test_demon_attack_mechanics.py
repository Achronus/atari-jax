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

"""Unit tests for Demon Attack game mechanics."""

import jax
import jax.numpy as jnp
import pytest

from atarax.env.games.demon_attack import (
    DemonAttack,
    DemonAttackParams,
    DemonAttackState,
    _BULLET_SPEED,
    _DEMON_HH_BIG,
    _DEMON_HW_BIG,
    _DEMON_SPEED,
    _FIRE_COOLDOWN,
    _INIT_ENEMIES,
    _MAX_ENEMY_BULLETS,
    _MAX_ENEMIES,
    _NUM_ZONES,
    _PLAYER_SPEED,
    _PLAYER_X_MAX,
    _PLAYER_X_MIN,
    _PLAYER_Y,
    _SCORE_BASE,
    _SPAWN_DELAY,
    _WAVE_SIZE,
    _ZONE_YS,
    _ZONE_YS_PY,
)

KEY = jax.random.PRNGKey(42)


def make_game():
    return DemonAttack(), DemonAttackParams()


def make_state(**overrides) -> DemonAttackState:
    """Return a DemonAttackState with defaults; all enemies cleared unless overridden."""
    e, p = make_game()
    _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
    # Clear all enemies and bullets by default
    s = s.__replace__(
        enemies=jnp.zeros((_MAX_ENEMIES, 5), dtype=jnp.float32),
        enemy_bullets=jnp.zeros((_MAX_ENEMY_BULLETS, 3), dtype=jnp.float32),
        enemy_bvx=jnp.zeros(_MAX_ENEMY_BULLETS, dtype=jnp.float32),
        bullet_split=jnp.zeros(_MAX_ENEMY_BULLETS, dtype=jnp.bool_),
        spawn_timer=jnp.full(_NUM_ZONES, _SPAWN_DELAY, dtype=jnp.int32),
    )
    for k, v in overrides.items():
        s = s.__replace__(**{k: v})
    return s


def step_physics(state: DemonAttackState, action: int = 0):
    e, p = make_game()
    return e._step_physics(state, jnp.int32(action), p, KEY)


def step_agent(state: DemonAttackState, action: int = 0):
    e, p = make_game()
    _, s2, r, done, _ = jax.jit(e.step)(KEY, state, jnp.int32(action), p)
    return s2, float(r), bool(done)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Reset
# ─────────────────────────────────────────────────────────────────────────────

class TestReset:
    def test_obs_shape_dtype(self):
        e, p = make_game()
        obs, _ = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert obs.shape == (210, 160, 3)
        assert obs.dtype == jnp.uint8

    def test_initial_player_position(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert float(s.player_x) == pytest.approx(80.0)

    def test_initial_lives(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert int(s.lives) == 3

    def test_three_zone_demons_active_at_reset(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        active_big = s.enemies[:3, 4]
        assert float(jnp.sum(active_big)) == pytest.approx(3.0)

    def test_split_slots_inactive_at_reset(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        active_split = s.enemies[3:, 4]
        assert float(jnp.sum(active_split)) == pytest.approx(0.0)

    def test_no_active_bullets_at_reset(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert float(jnp.sum(s.player_bullets[:, 2])) == pytest.approx(0.0)
        assert float(jnp.sum(s.enemy_bullets[:, 2])) == pytest.approx(0.0)

    def test_score_zero_at_reset(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert int(s.score) == 0
        assert not bool(s.done)

    def test_zone_y_positions(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        for z in range(_NUM_ZONES):
            assert float(s.enemies[z, 1]) == pytest.approx(_ZONE_YS_PY[z])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Player movement
# ─────────────────────────────────────────────────────────────────────────────

class TestPlayerMovement:
    def test_right_increases_x(self):
        s = make_state(player_x=jnp.float32(80.0))
        s2 = step_physics(s, action=2)  # RIGHT
        assert float(s2.player_x) == pytest.approx(80.0 + _PLAYER_SPEED)

    def test_left_decreases_x(self):
        s = make_state(player_x=jnp.float32(80.0))
        s2 = step_physics(s, action=3)  # LEFT
        assert float(s2.player_x) == pytest.approx(80.0 - _PLAYER_SPEED)

    def test_noop_no_movement(self):
        s = make_state(player_x=jnp.float32(80.0))
        s2 = step_physics(s, action=0)
        assert float(s2.player_x) == pytest.approx(80.0)

    def test_x_clamped_at_right_wall(self):
        s = make_state(player_x=jnp.float32(_PLAYER_X_MAX))
        s2 = step_physics(s, action=2)
        assert float(s2.player_x) <= _PLAYER_X_MAX

    def test_x_clamped_at_left_wall(self):
        s = make_state(player_x=jnp.float32(_PLAYER_X_MIN))
        s2 = step_physics(s, action=3)
        assert float(s2.player_x) >= _PLAYER_X_MIN

    def test_rightfire_moves_right(self):
        s = make_state(player_x=jnp.float32(80.0))
        s2 = step_physics(s, action=4)  # RIGHTFIRE
        assert float(s2.player_x) > 80.0

    def test_leftfire_moves_left(self):
        s = make_state(player_x=jnp.float32(80.0))
        s2 = step_physics(s, action=5)  # LEFTFIRE
        assert float(s2.player_x) < 80.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Player firing
# ─────────────────────────────────────────────────────────────────────────────

class TestPlayerFiring:
    def test_fire_spawns_bullet(self):
        s = make_state(
            player_x=jnp.float32(80.0),
            fire_cooldown=jnp.int32(0),
            player_bullets=jnp.zeros((1, 3), dtype=jnp.float32),
        )
        s2 = step_physics(s, action=1)  # FIRE
        assert float(s2.player_bullets[0, 2]) == pytest.approx(1.0)

    def test_fire_sets_cooldown(self):
        s = make_state(
            player_x=jnp.float32(80.0),
            fire_cooldown=jnp.int32(0),
            player_bullets=jnp.zeros((1, 3), dtype=jnp.float32),
        )
        s2 = step_physics(s, action=1)
        assert int(s2.fire_cooldown) == _FIRE_COOLDOWN

    def test_cannot_fire_during_cooldown(self):
        s = make_state(
            player_x=jnp.float32(80.0),
            fire_cooldown=jnp.int32(5),
            player_bullets=jnp.zeros((1, 3), dtype=jnp.float32),
        )
        s2 = step_physics(s, action=1)
        assert float(s2.player_bullets[0, 2]) == pytest.approx(0.0)

    def test_cooldown_decrements(self):
        s = make_state(fire_cooldown=jnp.int32(4))
        s2 = step_physics(s, action=0)
        assert int(s2.fire_cooldown) == 3

    def test_bullet_moves_upward(self):
        pb = jnp.array([[80.0, 150.0, 1.0]], dtype=jnp.float32)
        s = make_state(player_x=jnp.float32(80.0), fire_cooldown=jnp.int32(1), player_bullets=pb)
        s2 = step_physics(s, action=0)
        assert float(s2.player_bullets[0, 1]) < 150.0

    def test_bullet_deactivated_at_top(self):
        pb = jnp.array([[80.0, 1.0, 1.0]], dtype=jnp.float32)
        s = make_state(player_x=jnp.float32(80.0), fire_cooldown=jnp.int32(1), player_bullets=pb)
        s2 = step_physics(s, action=0)
        assert float(s2.player_bullets[0, 2]) == pytest.approx(0.0)

    def test_single_bullet_at_a_time(self):
        # Active bullet in slot prevents firing a new one
        pb = jnp.array([[60.0, 100.0, 1.0]], dtype=jnp.float32)
        s = make_state(
            player_x=jnp.float32(80.0),
            fire_cooldown=jnp.int32(0),
            player_bullets=pb,
        )
        s2 = step_physics(s, action=1)
        # Slot still occupied by existing bullet (x shouldn't jump to 80)
        assert float(s2.player_bullets[0, 0]) == pytest.approx(60.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Demon movement
# ─────────────────────────────────────────────────────────────────────────────

class TestDemonMovement:
    def test_demon_moves_right(self):
        enemies = jnp.zeros((_MAX_ENEMIES, 5), dtype=jnp.float32)
        enemies = enemies.at[0].set(jnp.array([80.0, _ZONE_YS_PY[0], _DEMON_SPEED, 1.0, 1.0]))
        s = make_state(enemies=enemies)
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 0]) > 80.0

    def test_demon_moves_left(self):
        enemies = jnp.zeros((_MAX_ENEMIES, 5), dtype=jnp.float32)
        enemies = enemies.at[0].set(jnp.array([80.0, _ZONE_YS_PY[0], -_DEMON_SPEED, 1.0, 1.0]))
        s = make_state(enemies=enemies)
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 0]) < 80.0

    def test_demon_bounces_at_right_wall(self):
        enemies = jnp.zeros((_MAX_ENEMIES, 5), dtype=jnp.float32)
        enemies = enemies.at[0].set(jnp.array([155.0, _ZONE_YS_PY[0], _DEMON_SPEED, 1.0, 1.0]))
        s = make_state(enemies=enemies)
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 2]) < 0.0

    def test_demon_bounces_at_left_wall(self):
        enemies = jnp.zeros((_MAX_ENEMIES, 5), dtype=jnp.float32)
        enemies = enemies.at[0].set(jnp.array([5.0, _ZONE_YS_PY[0], -_DEMON_SPEED, 1.0, 1.0]))
        s = make_state(enemies=enemies)
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 2]) > 0.0

    def test_inactive_demon_does_not_move(self):
        enemies = jnp.zeros((_MAX_ENEMIES, 5), dtype=jnp.float32)
        enemies = enemies.at[0].set(jnp.array([80.0, _ZONE_YS_PY[0], _DEMON_SPEED, 1.0, 0.0]))
        s = make_state(enemies=enemies)
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 0]) == pytest.approx(80.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Player bullet vs demon collision
# ─────────────────────────────────────────────────────────────────────────────

class TestPlayerBulletDemonCollision:
    def _state_with_bullet_on_demon(self, zone: int = 0) -> DemonAttackState:
        cy = _ZONE_YS_PY[zone]
        enemies = jnp.zeros((_MAX_ENEMIES, 5), dtype=jnp.float32)
        enemies = enemies.at[zone].set(jnp.array([80.0, cy, 0.0, 1.0, 1.0]))
        # Bullet 3px below demon centre — after moving up by _BULLET_SPEED=6 lands 3px above
        pb = jnp.array([[80.0, cy + 3.0, 1.0]], dtype=jnp.float32)
        return make_state(
            enemies=enemies,
            player_bullets=pb,
            fire_cooldown=jnp.int32(10),
        )

    def test_demon_deactivated_on_hit(self):
        s = self._state_with_bullet_on_demon(zone=0)
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 4]) == pytest.approx(0.0)

    def test_kill_awards_score(self):
        s = self._state_with_bullet_on_demon(zone=0)
        s2 = step_physics(s, action=0)
        assert int(s2.score) == _SCORE_BASE  # wave=0 → base pts

    def test_player_bullet_deactivated_on_hit(self):
        s = self._state_with_bullet_on_demon(zone=0)
        s2 = step_physics(s, action=0)
        assert float(s2.player_bullets[0, 2]) == pytest.approx(0.0)

    def test_bottom_zone_demon_hit(self):
        s = self._state_with_bullet_on_demon(zone=2)
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[2, 4]) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Enemy bullet vs player collision
# ─────────────────────────────────────────────────────────────────────────────

class TestEnemyBulletPlayerCollision:
    def _state_with_eb_on_player(self) -> DemonAttackState:
        eb = jnp.zeros((_MAX_ENEMY_BULLETS, 3), dtype=jnp.float32)
        eb = eb.at[0].set(jnp.array([80.0, _PLAYER_Y - 2.0, 1.0]))
        return make_state(player_x=jnp.float32(80.0), enemy_bullets=eb)

    def test_player_hit_loses_life(self):
        s = self._state_with_eb_on_player()
        s2 = step_physics(s, action=0)
        assert int(s2.lives) == int(s.lives) - 1

    def test_enemy_bullets_cleared_on_hit(self):
        s = self._state_with_eb_on_player()
        s2 = step_physics(s, action=0)
        assert float(jnp.sum(s2.enemy_bullets[:, 2])) == pytest.approx(0.0)

    def test_no_score_on_player_hit(self):
        s = self._state_with_eb_on_player()
        s2 = step_physics(s, action=0)
        assert int(s2.score) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Wave advance and life gain
# ─────────────────────────────────────────────────────────────────────────────

class TestWaveAdvance:
    def _state_near_wave_end(self) -> DemonAttackState:
        """One active demon with a bullet about to hit it — completes the wave."""
        cy = _ZONE_YS_PY[0]
        enemies = jnp.zeros((_MAX_ENEMIES, 5), dtype=jnp.float32)
        enemies = enemies.at[0].set(jnp.array([80.0, cy, 0.0, 1.0, 1.0]))
        pb = jnp.array([[80.0, cy + 3.0, 1.0]], dtype=jnp.float32)
        return make_state(
            enemies=enemies,
            player_bullets=pb,
            kills_in_wave=jnp.int32(_WAVE_SIZE - 1),
            fire_cooldown=jnp.int32(10),
        )

    def test_wave_increments_on_completion(self):
        s = self._state_near_wave_end()
        s2 = step_physics(s, action=0)
        assert int(s2.wave) == 1

    def test_kills_in_wave_resets_on_completion(self):
        s = self._state_near_wave_end()
        s2 = step_physics(s, action=0)
        assert int(s2.kills_in_wave) == 0

    def test_life_gained_on_wave_completion(self):
        s = self._state_near_wave_end()
        s2 = step_physics(s, action=0)
        assert int(s2.lives) == int(s.lives) + 1

    def test_lives_capped_at_six(self):
        s = self._state_near_wave_end().__replace__(lives=jnp.int32(6))
        s2 = step_physics(s, action=0)
        assert int(s2.lives) <= 6

    def test_kills_accumulate_within_wave(self):
        cy = _ZONE_YS_PY[0]
        enemies = jnp.zeros((_MAX_ENEMIES, 5), dtype=jnp.float32)
        enemies = enemies.at[0].set(jnp.array([80.0, cy, 0.0, 1.0, 1.0]))
        pb = jnp.array([[80.0, cy + 3.0, 1.0]], dtype=jnp.float32)
        s = make_state(
            enemies=enemies,
            player_bullets=pb,
            kills_in_wave=jnp.int32(0),
            fire_cooldown=jnp.int32(10),
        )
        s2 = step_physics(s, action=0)
        assert int(s2.kills_in_wave) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. Zone respawn
# ─────────────────────────────────────────────────────────────────────────────

class TestZoneRespawn:
    def test_spawn_timer_set_when_demon_killed(self):
        cy = _ZONE_YS_PY[0]
        enemies = jnp.zeros((_MAX_ENEMIES, 5), dtype=jnp.float32)
        enemies = enemies.at[0].set(jnp.array([80.0, cy, 0.0, 1.0, 1.0]))
        pb = jnp.array([[80.0, cy + 3.0, 1.0]], dtype=jnp.float32)
        s = make_state(
            enemies=enemies,
            player_bullets=pb,
            fire_cooldown=jnp.int32(10),
            spawn_timer=jnp.zeros(_NUM_ZONES, dtype=jnp.int32),
        )
        s2 = step_physics(s, action=0)
        assert int(s2.spawn_timer[0]) == _SPAWN_DELAY

    def test_demon_respawns_when_timer_expires(self):
        s = make_state(
            enemies=jnp.zeros((_MAX_ENEMIES, 5), dtype=jnp.float32),
            spawn_timer=jnp.array([1, _SPAWN_DELAY, _SPAWN_DELAY], dtype=jnp.int32),
        )
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 4]) == pytest.approx(1.0)

    def test_demon_not_respawned_before_timer_expires(self):
        s = make_state(
            enemies=jnp.zeros((_MAX_ENEMIES, 5), dtype=jnp.float32),
            spawn_timer=jnp.array([5, _SPAWN_DELAY, _SPAWN_DELAY], dtype=jnp.int32),
        )
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 4]) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Game termination
# ─────────────────────────────────────────────────────────────────────────────

class TestGameOver:
    def test_done_when_lives_zero(self):
        eb = jnp.zeros((_MAX_ENEMY_BULLETS, 3), dtype=jnp.float32)
        eb = eb.at[0].set(jnp.array([80.0, _PLAYER_Y - 2.0, 1.0]))
        s = make_state(player_x=jnp.float32(80.0), enemy_bullets=eb, lives=jnp.int32(1))
        s2 = step_physics(s, action=0)
        assert bool(s2.done)

    def test_not_done_with_lives_remaining(self):
        s = make_state(lives=jnp.int32(2))
        s2 = step_physics(s, action=0)
        assert not bool(s2.done)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Episode step counter
# ─────────────────────────────────────────────────────────────────────────────

class TestStepCounter:
    def test_episode_step_increments_once_per_agent_step(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        _, s2, _, _, _ = jax.jit(e.step)(KEY, s, jnp.int32(0), p)
        assert int(s2.episode_step) == 1

    def test_step_counter_increments_each_physics_frame(self):
        s = make_state()
        s2 = step_physics(s, action=0)
        assert int(s2.step) == int(s.step) + 1


# ─────────────────────────────────────────────────────────────────────────────
# 11. JIT + vmap smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestJITAndVmap:
    def test_jit_reset(self):
        e, p = make_game()
        obs, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert obs.shape == (210, 160, 3)

    def test_jit_step(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        obs2, s2, r, done, _ = jax.jit(e.step)(KEY, s, jnp.int32(0), p)
        assert obs2.shape == (210, 160, 3)

    def test_vmap_reset(self):
        e, p = make_game()
        obs_b, _ = jax.vmap(lambda k: e.reset(k, p))(jax.random.split(KEY, 8))
        assert obs_b.shape == (8, 210, 160, 3)
        assert obs_b.dtype == jnp.uint8

    def test_vmap_step(self):
        e, p = make_game()
        keys = jax.random.split(KEY, 8)
        obs_b, s_b = jax.vmap(lambda k: e.reset(k, p))(keys)
        actions = jnp.zeros(8, dtype=jnp.int32)
        obs2_b, s2_b, r_b, done_b, _ = jax.vmap(
            e.step, in_axes=(0, 0, 0, None)
        )(keys, s_b, actions, p)
        assert obs2_b.shape == (8, 210, 160, 3)
