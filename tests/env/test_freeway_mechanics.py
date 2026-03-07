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

"""Unit tests for Freeway game mechanics."""

import jax
import jax.numpy as jnp
import pytest

from atarax.env.games.freeway import (
    Freeway,
    FreewayParams,
    FreewayState,
    _CARS_PER_LANE,
    _CHICKEN_HH,
    _CHICKEN_SPEED,
    _SAFE_BOT_Y,
    _SAFE_TOP_Y,
    _TIME_LIMIT,
)

KEY = jax.random.PRNGKey(42)


def make_game():
    return Freeway(), FreewayParams()


def make_state(**overrides) -> FreewayState:
    """Return a FreewayState with sensible defaults, overridable by kwargs.

    By default uses cleared obstacles (all inactive) so tests that don't
    explicitly test collision are not affected by initial car positions.
    """
    e, p = make_game()
    _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
    # Default: no active cars (avoid spurious collisions from initial positions)
    s = s.__replace__(obstacles=jnp.zeros((10, _CARS_PER_LANE, 2), dtype=jnp.float32))
    for k, v in overrides.items():
        s = s.__replace__(**{k: v})
    return s


def step_physics(state: FreewayState, action: int = 0):
    e, p = make_game()
    return e._step_physics(state, jnp.int32(action), p, KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Reset
# ─────────────────────────────────────────────────────────────────────────────

class TestReset:
    def test_obs_shape(self):
        e, p = make_game()
        obs, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert obs.shape == (210, 160, 3)
        assert obs.dtype == jnp.uint8

    def test_score_zero(self):
        s = make_state()
        assert int(s.score) == 0

    def test_crossings_zero(self):
        s = make_state()
        assert int(s.crossings) == 0

    def test_timer_full(self):
        s = make_state()
        assert int(s.timer) == _TIME_LIMIT

    def test_chicken_starts_at_bottom(self):
        s = make_state()
        assert abs(float(s.player_y) - _SAFE_BOT_Y) < 1.0

    def test_not_done_initially(self):
        s = make_state()
        assert not bool(s.done)

    def test_pushback_zero(self):
        s = make_state()
        assert float(s.jump_vy) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Chicken movement
# ─────────────────────────────────────────────────────────────────────────────

class TestChickenMovement:
    def test_up_decreases_y(self):
        s = make_state(player_y=jnp.float32(100.0), jump_vy=jnp.float32(0.0))
        s2 = step_physics(s, action=1)  # UP
        assert float(s2.player_y) < 100.0

    def test_down_increases_y(self):
        # Only makes a difference if not already at bottom
        s = make_state(player_y=jnp.float32(100.0), jump_vy=jnp.float32(0.0))
        s2 = step_physics(s, action=2)  # DOWN
        assert float(s2.player_y) > 100.0

    def test_noop_no_movement_without_pushback(self):
        s = make_state(player_y=jnp.float32(100.0), jump_vy=jnp.float32(0.0))
        s2 = step_physics(s, action=0)  # NOOP
        assert abs(float(s2.player_y) - 100.0) < 0.01

    def test_up_speed(self):
        s = make_state(player_y=jnp.float32(100.0), jump_vy=jnp.float32(0.0))
        s2 = step_physics(s, action=1)
        assert abs(float(s2.player_y) - (100.0 - _CHICKEN_SPEED)) < 0.01

    def test_chicken_clamped_at_bottom(self):
        # Try to move DOWN when already at the bottom boundary
        s = make_state(player_y=jnp.float32(_SAFE_BOT_Y), jump_vy=jnp.float32(0.0))
        s2 = step_physics(s, action=2)
        assert float(s2.player_y) <= _SAFE_BOT_Y + 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Pushback
# ─────────────────────────────────────────────────────────────────────────────

class TestPushback:
    def test_pushback_decays(self):
        s = make_state(player_y=jnp.float32(100.0), jump_vy=jnp.float32(8.0))
        s2 = step_physics(s, action=0)
        assert float(s2.jump_vy) < 8.0

    def test_pushback_decay_factor(self):
        # No active cars, so no new collision impulse — pure decay
        s = make_state(player_y=jnp.float32(100.0), jump_vy=jnp.float32(10.0))
        s2 = step_physics(s, action=0)
        # Decay is 0.8: new pushback = 10.0 * 0.8 = 8.0
        assert abs(float(s2.jump_vy) - 8.0) < 0.01

    def test_pushback_moves_chicken_down(self):
        s = make_state(player_y=jnp.float32(100.0), jump_vy=jnp.float32(8.0))
        s2 = step_physics(s, action=0)
        # pushback_vy is positive (downward), so chicken_y increases
        assert float(s2.player_y) > 100.0


# ─────────────────────────────────────────────────────────────────────────────
# Crossing
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossing:
    def test_crossing_increments_score(self):
        # Place chicken just at the top threshold
        s = make_state(
            player_y=jnp.float32(_SAFE_TOP_Y + _CHICKEN_HH),
            jump_vy=jnp.float32(0.0),
        )
        s2 = step_physics(s, action=1)  # Move UP to cross
        assert int(s2.score) == 1

    def test_crossing_increments_crossings(self):
        s = make_state(
            player_y=jnp.float32(_SAFE_TOP_Y + _CHICKEN_HH),
            jump_vy=jnp.float32(0.0),
        )
        s2 = step_physics(s, action=1)
        assert int(s2.crossings) == 1

    def test_crossing_resets_chicken_to_bottom(self):
        s = make_state(
            player_y=jnp.float32(_SAFE_TOP_Y + _CHICKEN_HH),
            jump_vy=jnp.float32(0.0),
        )
        s2 = step_physics(s, action=1)
        assert abs(float(s2.player_y) - _SAFE_BOT_Y) < 1.0

    def test_crossing_resets_pushback(self):
        # Position chicken so that UP action crosses even without pushback
        s = make_state(
            player_y=jnp.float32(_SAFE_TOP_Y + _CHICKEN_HH),
            jump_vy=jnp.float32(0.0),
        )
        s2 = step_physics(s, action=1)
        assert float(s2.jump_vy) == pytest.approx(0.0)

    def test_crossing_reward(self):
        s = make_state(
            player_y=jnp.float32(_SAFE_TOP_Y + _CHICKEN_HH),
            jump_vy=jnp.float32(0.0),
        )
        s2 = step_physics(s, action=1)
        assert float(s2.reward) > 0.0

    def test_no_crossing_no_score(self):
        s = make_state(player_y=jnp.float32(100.0), jump_vy=jnp.float32(0.0))
        s2 = step_physics(s, action=1)
        assert int(s2.score) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Timer and done
# ─────────────────────────────────────────────────────────────────────────────

class TestTimerAndDone:
    def test_timer_decrements(self):
        s = make_state()
        s2 = step_physics(s)
        assert int(s2.timer) == _TIME_LIMIT - 1

    def test_done_when_timer_zero(self):
        s = make_state(timer=jnp.int32(1))
        s2 = step_physics(s)
        assert bool(s2.done)

    def test_not_done_when_timer_positive(self):
        s = make_state(timer=jnp.int32(100))
        s2 = step_physics(s)
        assert not bool(s2.done)


# ─────────────────────────────────────────────────────────────────────────────
# Frame skip
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameSkip:
    def test_episode_step_increments_once(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert int(s.episode_step) == 0
        _, s2, _, _, _ = jax.jit(lambda k, st, a: e.step(k, st, a, p))(
            KEY, s, jnp.int32(0)
        )
        assert int(s2.episode_step) == 1

    def test_step_increments_4x(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert int(s.step) == 0
        _, s2, _, _, _ = jax.jit(lambda k, st, a: e.step(k, st, a, p))(
            KEY, s, jnp.int32(0)
        )
        assert int(s2.step) == 4

    def test_timer_decrements_4x_per_agent_step(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        _, s2, _, _, _ = jax.jit(lambda k, st, a: e.step(k, st, a, p))(
            KEY, s, jnp.int32(0)
        )
        assert int(s2.timer) == _TIME_LIMIT - 4
