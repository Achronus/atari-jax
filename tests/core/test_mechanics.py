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

"""Unit tests for atarax.core.mechanics — physics primitives.

Run with:
    pytest tests/core/test_mechanics.py -v
"""

import chex
import jax.numpy as jnp

from atarax.core.mechanics import aabb_overlap, clamp, grid_hit_test, reflect_axis, wrap_position


# ---------------------------------------------------------------------------
# aabb_overlap
# ---------------------------------------------------------------------------


def test_aabb_overlap_hit():
    pos_a = jnp.array([5.0, 5.0])
    size_a = jnp.array([4.0, 4.0])
    pos_b = jnp.array([7.0, 7.0])
    size_b = jnp.array([4.0, 4.0])
    result = aabb_overlap(pos_a, size_a, pos_b, size_b)
    chex.assert_type(result, jnp.bool_)
    assert bool(result)


def test_aabb_overlap_no_hit_x():
    pos_a = jnp.array([0.0, 5.0])
    size_a = jnp.array([4.0, 4.0])
    pos_b = jnp.array([10.0, 5.0])
    size_b = jnp.array([4.0, 4.0])
    assert not bool(aabb_overlap(pos_a, size_a, pos_b, size_b))


def test_aabb_overlap_no_hit_y():
    pos_a = jnp.array([5.0, 0.0])
    size_a = jnp.array([4.0, 4.0])
    pos_b = jnp.array([5.0, 10.0])
    size_b = jnp.array([4.0, 4.0])
    assert not bool(aabb_overlap(pos_a, size_a, pos_b, size_b))


def test_aabb_overlap_touching_edge():
    pos_a = jnp.array([0.0, 0.0])
    size_a = jnp.array([4.0, 4.0])
    pos_b = jnp.array([4.0, 0.0])
    size_b = jnp.array([4.0, 4.0])
    assert not bool(aabb_overlap(pos_a, size_a, pos_b, size_b))


# ---------------------------------------------------------------------------
# reflect_axis
# ---------------------------------------------------------------------------


def test_reflect_axis_x():
    vel = jnp.array([2.0, -3.0])
    result = reflect_axis(vel, jnp.bool_(True), jnp.bool_(False))
    chex.assert_shape(result, (2,))
    assert float(result[0]) == pytest.approx(-2.0)
    assert float(result[1]) == pytest.approx(-3.0)


def test_reflect_axis_y():
    vel = jnp.array([2.0, -3.0])
    result = reflect_axis(vel, jnp.bool_(False), jnp.bool_(True))
    assert float(result[0]) == pytest.approx(2.0)
    assert float(result[1]) == pytest.approx(3.0)


def test_reflect_axis_none():
    vel = jnp.array([2.0, -3.0])
    result = reflect_axis(vel, jnp.bool_(False), jnp.bool_(False))
    assert float(result[0]) == pytest.approx(2.0)
    assert float(result[1]) == pytest.approx(-3.0)


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------


def test_clamp_within():
    result = clamp(jnp.float32(5.0), jnp.float32(0.0), jnp.float32(10.0))
    assert float(result) == pytest.approx(5.0)


def test_clamp_below():
    result = clamp(jnp.float32(-1.0), jnp.float32(0.0), jnp.float32(10.0))
    assert float(result) == pytest.approx(0.0)


def test_clamp_above():
    result = clamp(jnp.float32(15.0), jnp.float32(0.0), jnp.float32(10.0))
    assert float(result) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# wrap_position
# ---------------------------------------------------------------------------


def test_wrap_position_within():
    result = wrap_position(jnp.float32(5.0), jnp.float32(0.0), jnp.float32(10.0))
    assert float(result) == pytest.approx(5.0)


def test_wrap_position_above():
    result = wrap_position(jnp.float32(12.0), jnp.float32(0.0), jnp.float32(10.0))
    assert float(result) == pytest.approx(2.0)


def test_wrap_position_below():
    result = wrap_position(jnp.float32(-1.0), jnp.float32(0.0), jnp.float32(10.0))
    assert float(result) == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# grid_hit_test
# ---------------------------------------------------------------------------


def test_grid_hit_test_hit():
    grid = jnp.ones((2, 3), dtype=jnp.bool_)
    hit_mask, any_hit = grid_hit_test(
        ball_x=jnp.float32(10.0),
        ball_y=jnp.float32(10.0),
        ball_w=jnp.float32(2.0),
        ball_h=jnp.float32(2.0),
        grid=grid,
        cell_y0=jnp.float32(9.0),
        cell_x0=jnp.float32(9.0),
        cell_h=jnp.float32(6.0),
        cell_w=jnp.float32(8.0),
    )
    chex.assert_shape(hit_mask, (2, 3))
    assert bool(any_hit)


def test_grid_hit_test_no_hit():
    grid = jnp.ones((2, 3), dtype=jnp.bool_)
    hit_mask, any_hit = grid_hit_test(
        ball_x=jnp.float32(200.0),
        ball_y=jnp.float32(200.0),
        ball_w=jnp.float32(2.0),
        ball_h=jnp.float32(2.0),
        grid=grid,
        cell_y0=jnp.float32(9.0),
        cell_x0=jnp.float32(9.0),
        cell_h=jnp.float32(6.0),
        cell_w=jnp.float32(8.0),
    )
    assert not bool(any_hit)


def test_grid_hit_test_inactive_cell():
    grid = jnp.zeros((2, 3), dtype=jnp.bool_)
    _, any_hit = grid_hit_test(
        ball_x=jnp.float32(10.0),
        ball_y=jnp.float32(10.0),
        ball_w=jnp.float32(2.0),
        ball_h=jnp.float32(2.0),
        grid=grid,
        cell_y0=jnp.float32(9.0),
        cell_x0=jnp.float32(9.0),
        cell_h=jnp.float32(6.0),
        cell_w=jnp.float32(8.0),
    )
    assert not bool(any_hit)


# need pytest for approx
import pytest  # noqa: E402
