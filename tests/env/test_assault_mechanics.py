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

"""Unit tests for Assault game mechanics."""

import jax
import jax.numpy as jnp
import pytest

from atarax.env.games.assault import (
    Assault,
    AssaultParams,
    AssaultState,
    _ATCK_RELEASE,
    _ATCK_HW,
    _ATCK_HH,
    _BULLET_SPEED,
    _BULLET_VX,
    _FIRE_COOLDOWN,
    _HEAT_COST,
    _HEAT_REGEN,
    _MAX_ATTACKERS,
    _MAX_ENEMIES,
    _MAX_ENEMY_BULLETS,
    _MAX_HEAT,
    _MAX_PLAYER_BULLETS,
    _MOTHER_HH,
    _MOTHER_HIT_HH,
    _MOTHER_HIT_HW,
    _MOTHER_HP,
    _MOTHER_HW,
    _MOTHER_RESPAWN,
    _MOTHER_SPEED,
    _MOTHER_Y,
    _PLAYER_SPEED,
    _PLAYER_X_MAX,
    _PLAYER_X_MIN,
    _PLAYER_Y_INIT,
    _SCORE_ATTACKER,
    _SCORE_MOTHER,
    _WAVE_SIZE,
)

KEY = jax.random.PRNGKey(42)


def make_game():
    return Assault(), AssaultParams()


def make_state(**overrides) -> AssaultState:
    """Return an AssaultState with sensible defaults, overridable by kwargs.

    All enemy slots are cleared by default (mothership inactive, no attackers)
    to avoid spurious collisions in tests not specifically testing enemies.
    """
    e, p = make_game()
    _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
    # Clear all enemies by default to avoid spurious collisions
    s = s.__replace__(
        enemies=jnp.zeros((_MAX_ENEMIES, 6), dtype=jnp.float32),
        wave_spawned=jnp.int32(0),
        heat=jnp.int32(0),
    )
    for k, v in overrides.items():
        s = s.__replace__(**{k: v})
    return s


def step_physics(state: AssaultState, action: int = 0):
    e, p = make_game()
    return e._step_physics(state, jnp.int32(action), p, KEY)


def step_agent(state: AssaultState, action: int = 0):
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
        assert float(s.player_y) == pytest.approx(_PLAYER_Y_INIT)

    def test_initial_lives(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert int(s.lives) == 4

    def test_mothership_active_at_reset(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert float(s.enemies[0, 5]) == pytest.approx(1.0)  # active

    def test_mothership_at_correct_y(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert float(s.enemies[0, 1]) == pytest.approx(_MOTHER_Y)

    def test_no_active_attackers_at_reset(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        attacker_active = s.enemies[1:, 5]
        assert float(jnp.sum(attacker_active)) == pytest.approx(0.0)

    def test_no_active_bullets_at_reset(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert float(jnp.sum(s.player_bullets[:, 2])) == pytest.approx(0.0)
        assert float(jnp.sum(s.enemy_bullets[:, 2])) == pytest.approx(0.0)

    def test_score_and_done_zero(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert int(s.score) == 0
        assert not bool(s.done)

    def test_mother_hp_at_reset(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert int(s.mother_hp) == _MOTHER_HP


# ─────────────────────────────────────────────────────────────────────────────
# 2. Player movement
# ─────────────────────────────────────────────────────────────────────────────

class TestPlayerMovement:
    def test_right_increases_x(self):
        s = make_state(player_x=jnp.float32(80.0), player_y=jnp.float32(180.0))
        s2 = step_physics(s, action=3)  # RIGHT
        assert float(s2.player_x) > 80.0

    def test_left_decreases_x(self):
        s = make_state(player_x=jnp.float32(80.0), player_y=jnp.float32(180.0))
        s2 = step_physics(s, action=4)  # LEFT
        assert float(s2.player_x) < 80.0

    def test_up_acts_as_fire(self):
        # UP redirected to FIRE — y stays fixed, bullet spawns
        s = make_state(
            player_x=jnp.float32(80.0),
            player_bullets=jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32),
            fire_cooldown=jnp.int32(0),
        )
        s2 = step_physics(s, action=2)  # UP
        assert float(s2.player_y) == pytest.approx(_PLAYER_Y_INIT)
        assert float(jnp.sum(s2.player_bullets[:, 2])) >= 1.0

    def test_player_y_always_fixed(self):
        # Player y must remain at ground position regardless of action
        for action in range(7):
            s = make_state(player_x=jnp.float32(80.0))
            s2 = step_physics(s, action=action)
            assert float(s2.player_y) == pytest.approx(_PLAYER_Y_INIT), f"action {action}"

    def test_noop_no_x_movement(self):
        s = make_state(player_x=jnp.float32(80.0))
        s2 = step_physics(s, action=0)  # NOOP
        assert float(s2.player_x) == pytest.approx(80.0)

    def test_right_speed(self):
        s = make_state(player_x=jnp.float32(80.0))
        s2 = step_physics(s, action=3)
        assert float(s2.player_x) == pytest.approx(80.0 + _PLAYER_SPEED)

    def test_left_speed(self):
        s = make_state(player_x=jnp.float32(80.0))
        s2 = step_physics(s, action=4)
        assert float(s2.player_x) == pytest.approx(80.0 - _PLAYER_SPEED)

    def test_x_clamped_at_right_wall(self):
        s = make_state(player_x=jnp.float32(_PLAYER_X_MAX))
        s2 = step_physics(s, action=3)
        assert float(s2.player_x) <= _PLAYER_X_MAX

    def test_x_clamped_at_left_wall(self):
        s = make_state(player_x=jnp.float32(_PLAYER_X_MIN))
        s2 = step_physics(s, action=4)
        assert float(s2.player_x) >= _PLAYER_X_MIN

    def test_rightfire_also_moves_right(self):
        s = make_state(player_x=jnp.float32(80.0), player_y=jnp.float32(180.0))
        s2 = step_physics(s, action=5)  # RIGHTFIRE
        assert float(s2.player_x) > 80.0

    def test_leftfire_also_moves_left(self):
        s = make_state(player_x=jnp.float32(80.0), player_y=jnp.float32(180.0))
        s2 = step_physics(s, action=6)  # LEFTFIRE
        assert float(s2.player_x) < 80.0


# ─────────────────────────────────────────────────────────────────────────────
# 3. Player firing
# ─────────────────────────────────────────────────────────────────────────────

class TestPlayerFiring:
    def test_fire_spawns_bullet(self):
        s = make_state(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(_PLAYER_Y_INIT),
            fire_cooldown=jnp.int32(0),
            player_bullets=jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32),
        )
        s2 = step_physics(s, action=1)  # FIRE
        assert float(jnp.sum(s2.player_bullets[:, 2])) >= 1.0

    def test_fire_sets_cooldown(self):
        s = make_state(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(_PLAYER_Y_INIT),
            fire_cooldown=jnp.int32(0),
            player_bullets=jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32),
        )
        s2 = step_physics(s, action=1)
        assert int(s2.fire_cooldown) == _FIRE_COOLDOWN

    def test_cannot_fire_during_cooldown(self):
        s = make_state(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(_PLAYER_Y_INIT),
            fire_cooldown=jnp.int32(5),
            player_bullets=jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32),
        )
        s2 = step_physics(s, action=1)
        assert float(jnp.sum(s2.player_bullets[:, 2])) == pytest.approx(0.0)

    def test_cooldown_decrements(self):
        s = make_state(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(_PLAYER_Y_INIT),
            fire_cooldown=jnp.int32(4),
            player_bullets=jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32),
        )
        s2 = step_physics(s, action=0)  # NOOP
        assert int(s2.fire_cooldown) == 3

    def test_bullet_moves_up(self):
        # Bullet starts active at y=180
        pb = jnp.array([[80.0, 180.0, 1.0]], dtype=jnp.float32)
        s = make_state(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(_PLAYER_Y_INIT),
            fire_cooldown=jnp.int32(1),
            player_bullets=pb,
        )
        s2 = step_physics(s, action=0)
        assert float(s2.player_bullets[0, 1]) < 180.0

    def test_rightfire_sets_positive_bullet_vx(self):
        s = make_state(
            player_x=jnp.float32(80.0),
            fire_cooldown=jnp.int32(0),
            player_bullets=jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32),
        )
        s2 = step_physics(s, action=5)  # RIGHTFIRE
        assert float(s2.bullet_vx) == pytest.approx(_BULLET_VX)

    def test_leftfire_sets_negative_bullet_vx(self):
        s = make_state(
            player_x=jnp.float32(80.0),
            fire_cooldown=jnp.int32(0),
            player_bullets=jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32),
        )
        s2 = step_physics(s, action=6)  # LEFTFIRE
        assert float(s2.bullet_vx) == pytest.approx(-_BULLET_VX)

    def test_straight_fire_zero_vx(self):
        s = make_state(
            player_x=jnp.float32(80.0),
            fire_cooldown=jnp.int32(0),
            player_bullets=jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32),
        )
        s2 = step_physics(s, action=1)  # FIRE
        assert float(s2.bullet_vx) == pytest.approx(0.0)

    def test_bullet_deactivated_at_top(self):
        # Bullet near top edge — should deactivate
        pb = jnp.array([[80.0, 1.0, 1.0]], dtype=jnp.float32)
        s = make_state(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(_PLAYER_Y_INIT),
            fire_cooldown=jnp.int32(1),
            player_bullets=pb,
        )
        s2 = step_physics(s, action=0)
        assert float(s2.player_bullets[0, 2]) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Mothership movement
# ─────────────────────────────────────────────────────────────────────────────

class TestMothershipMovement:
    def test_mothership_moves_right(self):
        enemies = jnp.zeros((_MAX_ENEMIES, 6), dtype=jnp.float32)
        # Slot 0: mothership at x=80, vx=+1, active
        enemies = enemies.at[0].set(jnp.array([80.0, _MOTHER_Y, _MOTHER_SPEED, 0.0, 0.0, 1.0]))
        s = make_state(enemies=enemies, spawn_timer=jnp.int32(_ATCK_RELEASE))
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 0]) > 80.0

    def test_mothership_reverses_at_right_wall(self):
        enemies = jnp.zeros((_MAX_ENEMIES, 6), dtype=jnp.float32)
        # Place mothership near right wall, moving right
        enemies = enemies.at[0].set(jnp.array([152.0, _MOTHER_Y, _MOTHER_SPEED, 0.0, 0.0, 1.0]))
        s = make_state(enemies=enemies, spawn_timer=jnp.int32(_ATCK_RELEASE))
        s2 = step_physics(s, action=0)
        # vx should have reversed (negative)
        assert float(s2.enemies[0, 2]) < 0.0

    def test_mothership_reverses_at_left_wall(self):
        enemies = jnp.zeros((_MAX_ENEMIES, 6), dtype=jnp.float32)
        enemies = enemies.at[0].set(jnp.array([8.0, _MOTHER_Y, -_MOTHER_SPEED, 0.0, 0.0, 1.0]))
        s = make_state(enemies=enemies, spawn_timer=jnp.int32(_ATCK_RELEASE))
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 2]) > 0.0

    def test_inactive_mothership_does_not_move(self):
        enemies = jnp.zeros((_MAX_ENEMIES, 6), dtype=jnp.float32)
        enemies = enemies.at[0].set(jnp.array([80.0, _MOTHER_Y, _MOTHER_SPEED, 0.0, 0.0, 0.0]))
        s = make_state(enemies=enemies, spawn_timer=jnp.int32(10))
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 0]) == pytest.approx(80.0)

    def test_mothership_respawns_after_timer(self):
        enemies = jnp.zeros((_MAX_ENEMIES, 6), dtype=jnp.float32)
        enemies = enemies.at[0].set(jnp.array([80.0, _MOTHER_Y, _MOTHER_SPEED, 0.0, 0.0, 0.0]))
        s = make_state(enemies=enemies, spawn_timer=jnp.int32(1))
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 5]) == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Bullet vs mothership collision (5-HP system)
# ─────────────────────────────────────────────────────────────────────────────

class TestMothershipCollision:
    def _state_with_bullet_on_mother(self, hp: int = _MOTHER_HP) -> AssaultState:
        enemies = jnp.zeros((_MAX_ENEMIES, 6), dtype=jnp.float32)
        enemies = enemies.at[0].set(jnp.array([80.0, _MOTHER_Y, 0.0, 0.0, 0.0, 1.0]))
        # Keep one attacker alive so that killing the mothership does NOT trigger
        # an immediate wave clear (which would respawn everything).
        enemies = enemies.at[1].set(jnp.array([40.0, 60.0, 0.0, 0.5, 1.0, 1.0]))
        # Bullet placed 3px below mothership centre so that after moving up by
        # _BULLET_SPEED=6, it lands 3px above centre — within _MOTHER_HIT_HH=5.
        pb = jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32)
        pb = pb.at[0].set(jnp.array([80.0, _MOTHER_Y + 3.0, 1.0]))
        return make_state(
            enemies=enemies,
            player_bullets=pb,
            mother_hp=jnp.int32(hp),
            spawn_timer=jnp.int32(_ATCK_RELEASE),
            fire_cooldown=jnp.int32(10),
        )

    def test_hit_reduces_hp(self):
        s = self._state_with_bullet_on_mother(hp=5)
        s2 = step_physics(s, action=0)
        assert int(s2.mother_hp) == 4

    def test_mother_survives_below_max_hp(self):
        s = self._state_with_bullet_on_mother(hp=2)
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 5]) == pytest.approx(1.0)

    def test_mother_killed_when_hp_reaches_zero(self):
        s = self._state_with_bullet_on_mother(hp=1)
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 5]) == pytest.approx(0.0)

    def test_mother_kill_awards_score(self):
        s = self._state_with_bullet_on_mother(hp=1)
        s2 = step_physics(s, action=0)
        assert int(s2.score) == _SCORE_MOTHER

    def test_mother_kill_starts_respawn_timer(self):
        s = self._state_with_bullet_on_mother(hp=1)
        s2 = step_physics(s, action=0)
        assert int(s2.spawn_timer) == _MOTHER_RESPAWN

    def test_mother_hp_resets_after_kill(self):
        s = self._state_with_bullet_on_mother(hp=1)
        s2 = step_physics(s, action=0)
        assert int(s2.mother_hp) == _MOTHER_HP

    def test_bullet_deactivated_on_hit(self):
        s = self._state_with_bullet_on_mother(hp=5)
        s2 = step_physics(s, action=0)
        assert float(s2.player_bullets[0, 2]) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Bullet vs attacker collision
# ─────────────────────────────────────────────────────────────────────────────

class TestAttackerCollision:
    def _state_with_bullet_on_attacker(self) -> AssaultState:
        enemies = jnp.zeros((_MAX_ENEMIES, 6), dtype=jnp.float32)
        # Attacker at slot 1, centred at (80, 100)
        enemies = enemies.at[1].set(jnp.array([80.0, 100.0, 0.0, 0.0, 1.0, 1.0]))
        # Bullet 3px below centre so after moving -6 it lands 3px above: dy=3 < _ATCK_HH=5
        pb = jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32)
        pb = pb.at[0].set(jnp.array([80.0, 103.0, 1.0]))
        return make_state(
            enemies=enemies,
            player_bullets=pb,
            fire_cooldown=jnp.int32(10),
            spawn_timer=jnp.int32(_ATCK_RELEASE),
        )

    def test_attacker_deactivated_on_hit(self):
        s = self._state_with_bullet_on_attacker()
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[1, 5]) == pytest.approx(0.0)

    def test_attacker_kill_awards_score(self):
        s = self._state_with_bullet_on_attacker()
        s2 = step_physics(s, action=0)
        assert int(s2.score) == _SCORE_ATTACKER

    def test_bullet_deactivated_on_attacker_hit(self):
        s = self._state_with_bullet_on_attacker()
        s2 = step_physics(s, action=0)
        assert float(s2.player_bullets[0, 2]) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Enemy bullet vs player collision
# ─────────────────────────────────────────────────────────────────────────────

class TestEnemyBulletPlayerCollision:
    def _state_with_eb_on_player(self) -> AssaultState:
        # Place bullet at player_y - 2 so it's already inside the AABB (player_hh=4)
        eb = jnp.zeros((_MAX_ENEMY_BULLETS, 3), dtype=jnp.float32)
        eb = eb.at[0].set(jnp.array([80.0, _PLAYER_Y_INIT - 2.0, 1.0]))
        return make_state(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(_PLAYER_Y_INIT),
            enemy_bullets=eb,
        )

    def test_player_hit_loses_life(self):
        s = self._state_with_eb_on_player()
        s2 = step_physics(s, action=0)
        assert int(s2.lives) == int(s.lives) - 1

    def test_player_resets_to_initial_position_on_hit(self):
        s = self._state_with_eb_on_player()
        s2 = step_physics(s, action=0)
        assert float(s2.player_x) == pytest.approx(80.0)
        assert float(s2.player_y) == pytest.approx(_PLAYER_Y_INIT)

    def test_enemy_bullet_deactivated_on_player_hit(self):
        s = self._state_with_eb_on_player()
        s2 = step_physics(s, action=0)
        assert float(s2.enemy_bullets[0, 2]) == pytest.approx(0.0)

    def test_no_score_on_player_hit(self):
        s = self._state_with_eb_on_player()
        s2 = step_physics(s, action=0)
        assert int(s2.score) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Wave advance
# ─────────────────────────────────────────────────────────────────────────────

class TestWaveAdvance:
    def test_wave_increments_when_all_enemies_cleared(self):
        # Wave completes when all 10 attackers have been sent AND none are active
        s = make_state(wave=jnp.int32(1), wave_spawned=jnp.int32(_WAVE_SIZE))
        s2 = step_physics(s, action=0)
        assert int(s2.wave) == 2

    def test_wave_does_not_increment_with_active_enemies(self):
        # Active attacker in slot 1 prevents wave completion even if wave_spawned == _WAVE_SIZE
        enemies = jnp.zeros((_MAX_ENEMIES, 6), dtype=jnp.float32)
        enemies = enemies.at[1].set(jnp.array([80.0, 60.0, 0.0, 0.3, 1.0, 1.0]))
        s = make_state(
            enemies=enemies, wave=jnp.int32(1),
            wave_spawned=jnp.int32(_WAVE_SIZE),
            spawn_timer=jnp.int32(_ATCK_RELEASE),
        )
        s2 = step_physics(s, action=0)
        assert int(s2.wave) == 1

    def test_wave_does_not_increment_before_wave_size(self):
        # All enemies clear but wave_spawned < _WAVE_SIZE — no wave advance yet
        s = make_state(wave=jnp.int32(1), wave_spawned=jnp.int32(5))
        s2 = step_physics(s, action=0)
        assert int(s2.wave) == 1

    def test_mothership_respawns_on_wave_clear(self):
        s = make_state(wave=jnp.int32(1), wave_spawned=jnp.int32(_WAVE_SIZE))
        s2 = step_physics(s, action=0)
        assert float(s2.enemies[0, 5]) == pytest.approx(1.0)

    def test_mother_hp_resets_on_wave_clear(self):
        s = make_state(wave=jnp.int32(1), wave_spawned=jnp.int32(_WAVE_SIZE), mother_hp=jnp.int32(2))
        s2 = step_physics(s, action=0)
        assert int(s2.mother_hp) == _MOTHER_HP

    def test_wave_spawned_resets_on_wave_clear(self):
        s = make_state(wave=jnp.int32(1), wave_spawned=jnp.int32(_WAVE_SIZE))
        s2 = step_physics(s, action=0)
        assert int(s2.wave_spawned) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Game termination (section header only — tests below unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class TestGameOver:
    def test_done_when_lives_zero(self):
        eb = jnp.zeros((_MAX_ENEMY_BULLETS, 3), dtype=jnp.float32)
        eb = eb.at[0].set(jnp.array([80.0, _PLAYER_Y_INIT - 2.0, 1.0]))
        s = make_state(
            player_x=jnp.float32(80.0),
            player_y=jnp.float32(_PLAYER_Y_INIT),
            enemy_bullets=eb,
            lives=jnp.int32(1),
        )
        s2 = step_physics(s, action=0)
        assert bool(s2.done)

    def test_not_done_with_lives_remaining(self):
        s = make_state(lives=jnp.int32(3))
        s2 = step_physics(s, action=0)
        assert not bool(s2.done)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Heat mechanic
# ─────────────────────────────────────────────────────────────────────────────

class TestHeatMechanic:
    def test_heat_increases_on_fire(self):
        s = make_state(
            player_x=jnp.float32(80.0),
            fire_cooldown=jnp.int32(0),
            player_bullets=jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32),
            heat=jnp.int32(0),
        )
        s2 = step_physics(s, action=1)  # FIRE
        # heat goes up by _HEAT_COST then down by _HEAT_REGEN
        assert int(s2.heat) == _HEAT_COST - _HEAT_REGEN

    def test_heat_cools_passively(self):
        s = make_state(heat=jnp.int32(50), fire_cooldown=jnp.int32(5))
        s2 = step_physics(s, action=0)  # NOOP — no fire
        assert int(s2.heat) == 50 - _HEAT_REGEN

    def test_heat_clamps_at_zero(self):
        s = make_state(heat=jnp.int32(0), fire_cooldown=jnp.int32(5))
        s2 = step_physics(s, action=0)
        assert int(s2.heat) >= 0

    def test_heat_clamps_at_max(self):
        s = make_state(
            heat=jnp.int32(_MAX_HEAT - 1),
            fire_cooldown=jnp.int32(0),
            player_bullets=jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32),
        )
        # Even if we try to fire, heat can't exceed _MAX_HEAT
        # (but overheat check happens before firing so can_fire is blocked)
        s2 = step_physics(s, action=1)
        assert int(s2.heat) <= _MAX_HEAT

    def test_overheat_kills_player(self):
        # heat at max → player_hit triggered on next frame even with NOOP
        s = make_state(
            heat=jnp.int32(_MAX_HEAT),
            lives=jnp.int32(4),
        )
        s2 = step_physics(s, action=0)
        assert int(s2.lives) == 3

    def test_overheat_resets_heat(self):
        s = make_state(heat=jnp.int32(_MAX_HEAT), lives=jnp.int32(4))
        s2 = step_physics(s, action=0)
        assert int(s2.heat) == 0

    def test_cannot_fire_when_overheated(self):
        s = make_state(
            heat=jnp.int32(_MAX_HEAT),
            fire_cooldown=jnp.int32(0),
            player_bullets=jnp.zeros((_MAX_PLAYER_BULLETS, 3), dtype=jnp.float32),
        )
        s2 = step_physics(s, action=1)  # FIRE while overheated
        assert float(jnp.sum(s2.player_bullets[:, 2])) == pytest.approx(0.0)

    def test_heat_zero_at_reset(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert int(s.heat) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 11. Episode step counter
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
        obs_b, _ = jax.vmap(lambda k: e.reset(k, p))(
            jax.random.split(KEY, 8)
        )
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
