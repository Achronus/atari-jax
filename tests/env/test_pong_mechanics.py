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

"""Unit tests for Pong game mechanics."""

import jax
import jax.numpy as jnp
import pytest

from atarax.env.games.pong import (
    Pong,
    PongParams,
    PongState,
    _AI_X,
    _BALL_R,
    _BALL_SPEED_INIT,
    _BOT_WALL,
    _CENTER_X,
    _CENTER_Y,
    _PADDLE_HH,
    _PADDLE_HW,
    _PADDLE_SPEED,
    _PADDLE_Y_MAX,
    _PADDLE_Y_MIN,
    _PLAYER_X,
    _TOP_WALL,
)

KEY = jax.random.PRNGKey(42)


def make_game():
    return Pong(), PongParams()


def make_state(**overrides) -> PongState:
    """Return a PongState with sensible defaults, overridable by kwargs."""
    e, _ = make_game()
    _, s = jax.jit(lambda k: e.reset(k, PongParams()))(KEY)
    for k, v in overrides.items():
        s = s.__replace__(**{k: v})
    return s


def step_physics(state: PongState, action: int = 0):
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

    def test_initial_scores_zero(self):
        s = make_state()
        assert int(s.player_score) == 0
        assert int(s.ai_score) == 0

    def test_ball_starts_in_play(self):
        s = make_state()
        assert bool(s.ball_in_play)

    def test_ball_at_centre(self):
        s = make_state()
        assert abs(float(s.ball_x) - _CENTER_X) < 1.0
        assert abs(float(s.ball_y) - _CENTER_Y) < 5.0

    def test_paddles_start_centred(self):
        s = make_state()
        assert abs(float(s.player_y) - _CENTER_Y) < 1.0
        assert abs(float(s.ai_y) - _CENTER_Y) < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Paddle movement
# ─────────────────────────────────────────────────────────────────────────────

class TestPaddleMovement:
    def test_up_decreases_y(self):
        s = make_state(player_y=jnp.float32(100.0))
        s2 = step_physics(s, action=2)  # RIGHT → UP
        assert float(s2.player_y) < 100.0

    def test_down_increases_y(self):
        s = make_state(player_y=jnp.float32(100.0))
        s2 = step_physics(s, action=3)  # LEFT → DOWN
        assert float(s2.player_y) > 100.0

    def test_noop_no_paddle_movement(self):
        s = make_state(player_y=jnp.float32(100.0))
        s2 = step_physics(s, action=0)
        assert float(s2.player_y) == pytest.approx(100.0)

    def test_rightfire_acts_as_up(self):
        s = make_state(player_y=jnp.float32(100.0))
        s2 = step_physics(s, action=4)
        assert float(s2.player_y) < 100.0

    def test_leftfire_acts_as_down(self):
        s = make_state(player_y=jnp.float32(100.0))
        s2 = step_physics(s, action=5)
        assert float(s2.player_y) > 100.0

    def test_paddle_clamps_at_top(self):
        s = make_state(player_y=jnp.float32(_PADDLE_Y_MIN))
        s2 = step_physics(s, action=2)
        assert float(s2.player_y) >= _PADDLE_Y_MIN

    def test_paddle_clamps_at_bottom(self):
        s = make_state(player_y=jnp.float32(_PADDLE_Y_MAX))
        s2 = step_physics(s, action=3)
        assert float(s2.player_y) <= _PADDLE_Y_MAX

    def test_paddle_speed(self):
        s = make_state(player_y=jnp.float32(100.0))
        s2 = step_physics(s, action=2)
        assert abs(float(s2.player_y) - (100.0 - _PADDLE_SPEED)) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# AI tracking
# ─────────────────────────────────────────────────────────────────────────────

class TestAITracking:
    def test_ai_moves_toward_ball(self):
        s = make_state(
            ai_y=jnp.float32(80.0),
            ball_y=jnp.float32(120.0),
        )
        s2 = step_physics(s)
        assert float(s2.ai_y) > 80.0  # AI moves down toward ball

    def test_ai_moves_up_when_ball_above(self):
        s = make_state(
            ai_y=jnp.float32(140.0),
            ball_y=jnp.float32(80.0),
        )
        s2 = step_physics(s)
        assert float(s2.ai_y) < 140.0

    def test_ai_speed_capped(self):
        s = make_state(
            ai_y=jnp.float32(100.0),
            ball_y=jnp.float32(200.0),
        )
        s2 = step_physics(s)
        assert abs(float(s2.ai_y) - 100.0) <= _PADDLE_SPEED + 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Wall bounces
# ─────────────────────────────────────────────────────────────────────────────

class TestWallBounce:
    def test_top_wall_bounce(self):
        s = make_state(
            ball_x=jnp.float32(80.0),
            ball_y=jnp.float32(_TOP_WALL + _BALL_R - 0.5),
            ball_vx=jnp.float32(0.0),
            ball_vy=jnp.float32(-3.0),  # moving up
        )
        s2 = step_physics(s)
        assert float(s2.ball_vy) > 0.0, "Ball should bounce downward off top wall"

    def test_bottom_wall_bounce(self):
        s = make_state(
            ball_x=jnp.float32(80.0),
            ball_y=jnp.float32(_BOT_WALL - _BALL_R + 0.5),
            ball_vx=jnp.float32(0.0),
            ball_vy=jnp.float32(3.0),  # moving down
        )
        s2 = step_physics(s)
        assert float(s2.ball_vy) < 0.0, "Ball should bounce upward off bottom wall"


# ─────────────────────────────────────────────────────────────────────────────
# Paddle bounces
# ─────────────────────────────────────────────────────────────────────────────

class TestPaddleBounce:
    def test_player_paddle_deflects_ball(self):
        # Position ball just touching player paddle face, moving right
        s = make_state(
            ball_x=jnp.float32(_PLAYER_X - _PADDLE_HW - _BALL_R - 0.5),
            ball_y=jnp.float32(_CENTER_Y),
            ball_vx=jnp.float32(3.5),
            ball_vy=jnp.float32(0.0),
            player_y=jnp.float32(_CENTER_Y),
        )
        s2 = step_physics(s)
        assert float(s2.ball_vx) < 0.0, "Ball should bounce left off player paddle"

    def test_ai_paddle_deflects_ball(self):
        # Position ball just touching AI paddle face, moving left
        s = make_state(
            ball_x=jnp.float32(_AI_X + _PADDLE_HW + _BALL_R + 0.5),
            ball_y=jnp.float32(_CENTER_Y),
            ball_vx=jnp.float32(-3.5),
            ball_vy=jnp.float32(0.0),
            ai_y=jnp.float32(_CENTER_Y),
        )
        s2 = step_physics(s)
        assert float(s2.ball_vx) > 0.0, "Ball should bounce right off AI paddle"

    def test_player_paddle_miss_does_not_deflect(self):
        # Ball misses player paddle (far above)
        s = make_state(
            ball_x=jnp.float32(_PLAYER_X - _PADDLE_HW - _BALL_R - 0.5),
            ball_y=jnp.float32(_CENTER_Y - _PADDLE_HH * 3),
            ball_vx=jnp.float32(3.5),
            ball_vy=jnp.float32(0.0),
            player_y=jnp.float32(_CENTER_Y),
        )
        s2 = step_physics(s)
        assert float(s2.ball_vx) > 0.0, "Ball should continue right when it misses the paddle"

    def test_volley_count_increments_on_player_hit(self):
        s = make_state(
            ball_x=jnp.float32(_PLAYER_X - _PADDLE_HW - _BALL_R - 0.5),
            ball_y=jnp.float32(_CENTER_Y),
            ball_vx=jnp.float32(3.5),
            ball_vy=jnp.float32(0.0),
            player_y=jnp.float32(_CENTER_Y),
            volley_count=jnp.int32(0),
        )
        s2 = step_physics(s)
        assert int(s2.volley_count) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

class TestScoring:
    def test_ai_scores_when_ball_exits_right(self):
        # Ball moves past right edge — player missed
        s = make_state(
            ball_x=jnp.float32(160.0 - _BALL_R + 1.0),
            ball_y=jnp.float32(_CENTER_Y),
            ball_vx=jnp.float32(4.0),
            ball_vy=jnp.float32(0.0),
            player_score=jnp.int32(0),
            ai_score=jnp.int32(0),
        )
        s2 = step_physics(s)
        assert int(s2.ai_score) == 1

    def test_player_scores_when_ball_exits_left(self):
        # Ball moves past left edge — AI missed
        s = make_state(
            ball_x=jnp.float32(_BALL_R - 1.0),
            ball_y=jnp.float32(_CENTER_Y),
            ball_vx=jnp.float32(-4.0),
            ball_vy=jnp.float32(0.0),
            player_score=jnp.int32(0),
            ai_score=jnp.int32(0),
        )
        s2 = step_physics(s)
        assert int(s2.player_score) == 1

    def test_reward_positive_on_player_score(self):
        s = make_state(
            ball_x=jnp.float32(_BALL_R - 1.0),
            ball_y=jnp.float32(_CENTER_Y),
            ball_vx=jnp.float32(-4.0),
            ball_vy=jnp.float32(0.0),
        )
        s2 = step_physics(s)
        assert float(s2.reward) > 0.0

    def test_reward_negative_on_ai_score(self):
        s = make_state(
            ball_x=jnp.float32(160.0 - _BALL_R + 1.0),
            ball_y=jnp.float32(_CENTER_Y),
            ball_vx=jnp.float32(4.0),
            ball_vy=jnp.float32(0.0),
        )
        s2 = step_physics(s)
        assert float(s2.reward) < 0.0

    def test_ball_resets_after_score(self):
        s = make_state(
            ball_x=jnp.float32(160.0 - _BALL_R + 1.0),
            ball_y=jnp.float32(_CENTER_Y),
            ball_vx=jnp.float32(4.0),
            ball_vy=jnp.float32(0.0),
        )
        s2 = step_physics(s)
        assert abs(float(s2.ball_x) - _CENTER_X) < 2.0, "Ball should reset to centre"

    def test_volley_count_resets_on_score(self):
        s = make_state(
            ball_x=jnp.float32(160.0 - _BALL_R + 1.0),
            ball_y=jnp.float32(_CENTER_Y),
            ball_vx=jnp.float32(4.0),
            ball_vy=jnp.float32(0.0),
            volley_count=jnp.int32(5),
        )
        s2 = step_physics(s)
        assert int(s2.volley_count) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Done / episode termination
# ─────────────────────────────────────────────────────────────────────────────

class TestDone:
    def test_not_done_initially(self):
        s = make_state()
        assert not bool(s.done)

    def test_done_when_player_reaches_21(self):
        s = make_state(
            ball_x=jnp.float32(_BALL_R - 1.0),
            ball_y=jnp.float32(_CENTER_Y),
            ball_vx=jnp.float32(-4.0),
            ball_vy=jnp.float32(0.0),
            player_score=jnp.int32(20),
            ai_score=jnp.int32(0),
        )
        s2 = step_physics(s)
        assert bool(s2.done), "Done when player reaches 21"

    def test_done_when_ai_reaches_21(self):
        s = make_state(
            ball_x=jnp.float32(160.0 - _BALL_R + 1.0),
            ball_y=jnp.float32(_CENTER_Y),
            ball_vx=jnp.float32(4.0),
            ball_vy=jnp.float32(0.0),
            player_score=jnp.int32(0),
            ai_score=jnp.int32(20),
        )
        s2 = step_physics(s)
        assert bool(s2.done), "Done when AI reaches 21"


# ─────────────────────────────────────────────────────────────────────────────
# Frame skip
# ─────────────────────────────────────────────────────────────────────────────

class TestFrameSkip:
    def test_episode_step_increments_once_per_agent_step(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert int(s.episode_step) == 0
        _, s2, _, _, _ = jax.jit(lambda k, st, a: e.step(k, st, a, p))(
            KEY, s, jnp.int32(0)
        )
        assert int(s2.episode_step) == 1

    def test_step_increments_4x_per_agent_step(self):
        e, p = make_game()
        _, s = jax.jit(lambda k: e.reset(k, p))(KEY)
        assert int(s.step) == 0
        _, s2, _, _, _ = jax.jit(lambda k, st, a: e.step(k, st, a, p))(
            KEY, s, jnp.int32(0)
        )
        assert int(s2.step) == 4
