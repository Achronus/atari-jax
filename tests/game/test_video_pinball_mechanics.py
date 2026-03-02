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

"""Physics-level mechanics tests for Video Pinball.

Each test calls ``game._step_physics(state, action)`` directly (one emulated
frame) to verify specific physics events in isolation.

Run::

    pytest tests/game/test_video_pinball_mechanics.py -v
"""

import jax
import jax.numpy as jnp

from atarax.games.video_pinball import (
    VideoPinball,
    _BALL_GRAVITY,
    _BALL_LAUNCH_X,
    _BALL_LAUNCH_Y,
    _BALL_R,
    _BUMPER_CENTERS,
    _BUMPER_POINTS,
    _BUMPER_R,
    _FLIPPER_KICK,
    _FLIPPER_Y,
    _LEFT_FLIPPER_X0,
    _LEFT_FLIPPER_X1,
    _N_BUMPERS,
    _N_TARGETS,
    _RIGHT_FLIPPER_X0,
    _RIGHT_FLIPPER_X1,
    _TABLE_BOTTOM,
    _TABLE_LEFT,
    _TABLE_RIGHT,
    _TARGET_ORIGINS,
    _TARGET_POINTS,
)

_KEY = jax.random.PRNGKey(0)
_GAME = VideoPinball()
_INIT = _GAME._reset(_KEY)


def _state(**kw):
    """Return a fresh initial state with optional field overrides."""
    return _INIT.__replace__(**kw)


# ---------------------------------------------------------------------------
# Flipper state
# ---------------------------------------------------------------------------
def test_left_flipper_action_sets_flag():
    """Action 4 (LEFT) → left_flipper_up=True."""
    state = _state(reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(4))
    assert bool(new_state.left_flipper_up), "Left flipper should be up for action 4"
    assert not bool(new_state.right_flipper_up), "Right flipper should be down"


def test_right_flipper_action_sets_flag():
    """Action 3 (RIGHT) → right_flipper_up=True."""
    state = _state(reward=jnp.float32(0.0))
    new_state = _GAME._step_physics(state, jnp.int32(3))
    assert bool(new_state.right_flipper_up), "Right flipper should be up for action 3"
    assert not bool(new_state.left_flipper_up), "Left flipper should be down"


# ---------------------------------------------------------------------------
# Ball wall bounces
# ---------------------------------------------------------------------------
def test_ball_bounces_off_left_wall():
    """Ball moving left into left wall → ball_dx reverses sign."""
    state = _state(
        ball_x=jnp.float32(_TABLE_LEFT + _BALL_R + 0.1),
        ball_y=jnp.float32(100.0),
        ball_dx=jnp.float32(-2.0),
        ball_dy=jnp.float32(0.0),
        ball_active=jnp.bool_(True),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert float(new_state.ball_dx) > 0.0, (
        f"ball_dx should be positive after left wall bounce, got {float(new_state.ball_dx)}"
    )


def test_ball_bounces_off_right_wall():
    """Ball moving right into right wall → ball_dx reverses sign."""
    state = _state(
        ball_x=jnp.float32(_TABLE_RIGHT - _BALL_R - 0.1),
        ball_y=jnp.float32(100.0),
        ball_dx=jnp.float32(2.0),
        ball_dy=jnp.float32(0.0),
        ball_active=jnp.bool_(True),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert float(new_state.ball_dx) < 0.0, (
        f"ball_dx should be negative after right wall bounce, got {float(new_state.ball_dx)}"
    )


# ---------------------------------------------------------------------------
# Bumper scoring
# ---------------------------------------------------------------------------
def test_bumper_hit_scores_100():
    """Ball at bumper[0] centre → score += 100."""
    bx, by = float(_BUMPER_CENTERS[0][0]), float(_BUMPER_CENTERS[0][1])
    # Place ball just outside bumper radius, moving inward
    state = _state(
        ball_x=jnp.float32(bx + _BUMPER_R - _BALL_R - 0.1),
        ball_y=jnp.float32(by),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(0.0),
        ball_active=jnp.bool_(True),
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.score) == _BUMPER_POINTS, (
        f"Expected score {_BUMPER_POINTS} from bumper hit, got {int(new_state.score)}"
    )


# ---------------------------------------------------------------------------
# Target scoring
# ---------------------------------------------------------------------------
def test_target_hit_scores_500():
    """Ball at target[0] → score += 500, target marked hit."""
    tx, ty = float(_TARGET_ORIGINS[0][0]), float(_TARGET_ORIGINS[0][1])
    state = _state(
        ball_x=jnp.float32(tx + 5.0),
        ball_y=jnp.float32(ty + 3.0),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(0.0),
        ball_active=jnp.bool_(True),
        score=jnp.int32(0),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.score) == _TARGET_POINTS, (
        f"Expected score {_TARGET_POINTS} from target hit, got {int(new_state.score)}"
    )
    assert bool(new_state.target_hit[0]), "Target 0 should be marked hit"


# ---------------------------------------------------------------------------
# Ball drain
# ---------------------------------------------------------------------------
def test_ball_drain_loses_life():
    """Ball passing table bottom → lives -= 1, ball reset to launch position."""
    state = _state(
        ball_x=jnp.float32(130.0),
        ball_y=jnp.float32(_TABLE_BOTTOM - _BALL_R + 0.5),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(2.0),
        ball_active=jnp.bool_(True),
        lives=jnp.int32(3),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert int(new_state.lives) == 2, (
        f"Expected lives=2 after drain, got {int(new_state.lives)}"
    )
    assert not bool(new_state.ball_active), "Ball should be inactive after drain"


def test_episode_ends_when_balls_zero():
    """When all balls are drained (lives=0) → done."""
    state = _state(
        ball_x=jnp.float32(130.0),
        ball_y=jnp.float32(_TABLE_BOTTOM - _BALL_R + 0.5),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(2.0),
        ball_active=jnp.bool_(True),
        lives=jnp.int32(1),
        reward=jnp.float32(0.0),
    )
    new_state = _GAME._step_physics(state, jnp.int32(0))
    assert bool(new_state.done)
