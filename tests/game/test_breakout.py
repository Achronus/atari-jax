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

"""Breakout-specific unit tests (physics and state contract).

Call ``game._reset(key)`` directly throughout to bypass the NOOP warmup in
``AtariEnv.reset()``.  This gives the canonical deterministic initial state.
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from atarax.games.breakout import Breakout, BreakoutState

_KEY = jax.random.PRNGKey(0)
_NOOP = jnp.int32(0)
_FIRE = jnp.int32(1)

_BRICK_ROWS = 6
_BRICK_COLS = 18


@pytest.fixture(scope="module")
def game():
    return Breakout()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_bricks() -> jax.Array:
    """All-False brick grid."""
    return jnp.zeros((_BRICK_ROWS, _BRICK_COLS), dtype=jnp.bool_)


def _only_brick(row: int, col: int) -> jax.Array:
    """Brick grid with a single brick at (row, col)."""
    return _empty_bricks().at[row, col].set(True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_initial_state(game):
    state = game._reset(_KEY)
    assert isinstance(state, BreakoutState)
    assert int(state.level) == 0
    assert int(state.speed_tier) == 0
    assert int(state.lives) == 5
    assert bool(jnp.all(state.bricks)), "All bricks should be active on reset"
    assert not bool(state.ball_active), "Ball should be inactive until FIRE"
    assert state.level.dtype == jnp.int32
    assert state.speed_tier.dtype == jnp.int32


def test_speed_tier_upgrades_on_upper_row_hit(game):
    # Place a single brick in row 0 and position the ball directly on it.
    # Row 0 spans y ∈ [57, 63), col 0 spans x ∈ [8, 16).
    base = game._reset(_KEY)
    state = base.__replace__(
        bricks=_only_brick(0, 0),
        ball_x=jnp.float32(8.0),
        ball_y=jnp.float32(58.0),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(-2.0),
        ball_active=jnp.bool_(True),
        speed_tier=jnp.int32(0),
    )
    new_state = game._step_physics(state, _NOOP)
    assert int(new_state.speed_tier) >= 1, (
        "Speed tier should rise to at least 1 after hitting a row-0 brick"
    )


def test_level_and_speed_tier_on_board_clear(game):
    # One yellow brick (row 4, col 9) remains — hitting it clears the board.
    # Row 4 spans y ∈ [81, 87), col 9 spans x ∈ [8 + 9*8, 8 + 10*8) = [80, 88).
    base = game._reset(_KEY)
    state = base.__replace__(
        bricks=_only_brick(4, 9),
        ball_x=jnp.float32(80.0),
        ball_y=jnp.float32(82.0),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(-2.0),
        ball_active=jnp.bool_(True),
        speed_tier=jnp.int32(0),
        level=jnp.int32(0),
    )
    new_state = game._step_physics(state, _NOOP)
    assert int(new_state.level) == 1, "Level should increment on board clear"
    assert int(new_state.speed_tier) > int(state.speed_tier), (
        "Speed tier should increment on board clear"
    )
    assert bool(jnp.all(new_state.bricks)), "Bricks should reset after board clear"


def test_ball_velocity_scales_with_tier(game):
    # Start tier 0, ball moving straight up at (0, -2).
    # A row-0 brick collision raises tier to 1, scaling |dy| by 3.0/2.0.
    # Keep a second brick in row 5 so the board-clear path does NOT fire
    # (board-clear would raise the tier a second time in the same step).
    base = game._reset(_KEY)
    bricks = _empty_bricks().at[0, 0].set(True).at[5, 0].set(True)
    state = base.__replace__(
        bricks=bricks,
        ball_x=jnp.float32(8.0),
        ball_y=jnp.float32(58.0),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(-2.0),
        ball_active=jnp.bool_(True),
        speed_tier=jnp.int32(0),
    )
    new_state = game._step_physics(state, _NOOP)

    expected_scale = 3.0 / 2.0  # tier-1 speed / tier-0 speed
    original_speed = float(jnp.abs(state.ball_dy))
    new_speed = float(jnp.abs(new_state.ball_dy))
    assert abs(new_speed / original_speed - expected_scale) < 0.01, (
        f"Velocity magnitude should scale by {expected_scale}; "
        f"got {new_speed / original_speed:.3f}"
    )


def test_done_on_last_life_lost(game):
    # One life left; ball is moving down and will go out of bounds this frame.
    base = game._reset(_KEY)
    state = base.__replace__(
        lives=jnp.int32(1),
        ball_x=jnp.float32(80.0),
        ball_y=jnp.float32(200.0),
        ball_dx=jnp.float32(0.0),
        ball_dy=jnp.float32(10.0),
        ball_active=jnp.bool_(True),
    )
    new_state = game._step_physics(state, _NOOP)
    assert int(new_state.lives) == 0
    assert bool(new_state.done), "Episode should end when last life is lost"
