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

"""Pitfall! mechanics tests — direct _step_physics calls (one emulated frame).

Each test exercises a single physics rule in isolation.  All assertions use the
_INIT state as a baseline and call _step_physics once so results are exact.
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.games.pitfall import (
    Pitfall,
    _CROC_OPEN_AT,
    _GROUND_Y,
    _HARRY_INIT_X,
    _HARRY_SPEED,
    _INIT_LIVES,
    _LOG_COOLDOWN_FRAMES,
    _LOG_INIT_X,
    _LOG_PENALTY,
    _LOG_VX,
    _SCREEN_LEFT,
    _TREASURE_REWARD,
    _UNDERGROUND_Y,
)

_KEY = jax.random.PRNGKey(0)
_GAME = Pitfall()
_INIT = _GAME._reset(_KEY)


def _state(**kw):
    return _INIT.__replace__(**kw)


# ---------------------------------------------------------------------------
# Harry movement
# ---------------------------------------------------------------------------


def test_noop_harry_x_unchanged():
    """NOOP: Harry's x position must not change."""
    state = _state()
    new = _GAME._step_physics(state, jnp.int32(0))
    assert float(new.harry_x) == pytest.approx(_HARRY_INIT_X)


@pytest.mark.parametrize(
    "action,expected_dx",
    [
        (3, _HARRY_SPEED),   # RIGHT
        (4, -_HARRY_SPEED),  # LEFT
        (11, _HARRY_SPEED),  # RIGHTFIRE
        (12, -_HARRY_SPEED), # LEFTFIRE
    ],
)
def test_action_moves_harry(action, expected_dx):
    """Horizontal actions shift Harry's x by exactly ±_HARRY_SPEED."""
    state = _state()
    new = _GAME._step_physics(state, jnp.int32(action))
    assert float(new.harry_x) == pytest.approx(_HARRY_INIT_X + expected_dx)


# ---------------------------------------------------------------------------
# Jump physics
# ---------------------------------------------------------------------------


def test_jump_initiation():
    """UP from ground: Harry becomes airborne and moves upward."""
    state = _state()
    new = _GAME._step_physics(state, jnp.int32(2))  # UP
    assert bool(new.is_jumping)
    assert float(new.harry_y) < _GROUND_Y


def test_no_double_jump():
    """UP while already airborne does not re-initiate a jump."""
    state = _state(is_jumping=jnp.bool_(True), harry_vy=jnp.float32(-3.0))
    new = _GAME._step_physics(state, jnp.int32(2))  # UP while mid-air
    # Velocity must not snap back to _JUMP_VY
    assert float(new.harry_vy) > -4.0


# ---------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------


def test_logs_move_each_frame():
    """Logs must advance by their individual velocities each physics frame."""
    state = _state()
    new = _GAME._step_physics(state, jnp.int32(0))
    for i in range(3):
        expected = _LOG_INIT_X[i] + _LOG_VX[i]
        assert float(new.log_x[i]) == pytest.approx(expected)


def test_log_collision_deducts_score():
    """Harry at log 0's position receives a _LOG_PENALTY and cooldown is set."""
    # Log 0 starts at x=20.0, moves to 21.0 (right=37.0); Harry at 20 overlaps
    state = _state(harry_x=jnp.float32(20.0))
    new = _GAME._step_physics(state, jnp.int32(0))
    assert int(new.score) == _LOG_PENALTY
    assert int(new.log_cooldown) == _LOG_COOLDOWN_FRAMES


def test_log_cooldown_prevents_repeat_penalty():
    """While cooldown > 0, a log overlap does NOT deduct another penalty."""
    state = _state(
        harry_x=jnp.float32(20.0),
        log_cooldown=jnp.int32(_LOG_COOLDOWN_FRAMES),
    )
    new = _GAME._step_physics(state, jnp.int32(0))
    assert int(new.score) == 0


# ---------------------------------------------------------------------------
# Crocodiles
# ---------------------------------------------------------------------------


def test_croc_open_mouth_loses_life():
    """Harry inside an open-mouthed croc zone loses a life."""
    # Croc 0 spans x=[60,80]; Harry at x=70 overlaps.  timer=61 → open.
    state = _state(
        harry_x=jnp.float32(70.0),
        croc_timer=jnp.int32(_CROC_OPEN_AT + 1),
    )
    new = _GAME._step_physics(state, jnp.int32(0))
    assert int(new.lives) == _INIT_LIVES - 1


def test_croc_closed_mouth_safe():
    """Harry inside a closed-mouthed croc zone does NOT lose a life."""
    state = _state(
        harry_x=jnp.float32(70.0),
        croc_timer=jnp.int32(_CROC_OPEN_AT - 30),  # timer 30 → after frame 31 ≤ 60
    )
    new = _GAME._step_physics(state, jnp.int32(0))
    assert int(new.lives) == _INIT_LIVES


# ---------------------------------------------------------------------------
# Treasure
# ---------------------------------------------------------------------------


def test_treasure_collected():
    """Harry at the treasure position collects it: +2000 score, deactivated."""
    # Treasure left edge x=75, width=10; Harry left edge at 75 overlaps.
    state = _state(
        harry_x=jnp.float32(75.0),
        treasure_active=jnp.bool_(True),
    )
    new = _GAME._step_physics(state, jnp.int32(0))
    assert int(new.score) == _TREASURE_REWARD
    assert not bool(new.treasure_active)


def test_no_treasure_no_reward():
    """When treasure is inactive, Harry passing through it scores nothing."""
    state = _state(
        harry_x=jnp.float32(75.0),
        treasure_active=jnp.bool_(False),
    )
    new = _GAME._step_physics(state, jnp.int32(0))
    assert int(new.score) == 0


# ---------------------------------------------------------------------------
# Screen transition
# ---------------------------------------------------------------------------


def test_screen_transition_right():
    """Harry past the right boundary advances the screen and wraps to left."""
    state = _state(harry_x=jnp.float32(151.0))  # > _SCREEN_RIGHT=150
    new = _GAME._step_physics(state, jnp.int32(0))
    assert int(new.level) == 1
    assert float(new.harry_x) == pytest.approx(_SCREEN_LEFT + 2.0)


def test_screen_transition_left():
    """Harry past the left boundary retreats the screen and wraps to right."""
    state = _state(harry_x=jnp.float32(9.0))  # < _SCREEN_LEFT=10
    new = _GAME._step_physics(state, jnp.int32(0))
    # level wraps mod 255: 0 - 1 = -1 ≡ 254
    assert int(new.level) == 254


# ---------------------------------------------------------------------------
# Underground
# ---------------------------------------------------------------------------


def test_underground_entry():
    """DOWN near a hole entrance transitions Harry underground."""
    # Hole centre at x=80; Harry at 80 is within ±12 activation zone.
    state = _state(harry_x=jnp.float32(80.0))
    new = _GAME._step_physics(state, jnp.int32(5))  # DOWN
    assert bool(new.is_underground)
    assert float(new.harry_y) == pytest.approx(_UNDERGROUND_Y)


def test_down_away_from_hole_no_entry():
    """DOWN when not near a hole does NOT enter underground.

    x=130 is 10 px from the hole centre at x=120, which is outside the ±8 px
    activation zone [112, 128], so Harry must stay above-ground.
    """
    state = _state(harry_x=jnp.float32(130.0))
    new = _GAME._step_physics(state, jnp.int32(5))  # DOWN
    assert not bool(new.is_underground)


# ---------------------------------------------------------------------------
# Terminal condition
# ---------------------------------------------------------------------------


def test_terminal_when_lives_zero():
    """done=True when lives reach zero."""
    state = _state(lives=jnp.int32(0))
    new = _GAME._step_physics(state, jnp.int32(0))
    assert bool(new.done)


def test_not_terminal_with_lives():
    """done=False while lives > 0 and no fatal collision."""
    state = _state(lives=jnp.int32(2))
    new = _GAME._step_physics(state, jnp.int32(0))
    assert not bool(new.done)
