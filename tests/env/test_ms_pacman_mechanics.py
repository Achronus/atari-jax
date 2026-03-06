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

"""Unit tests for Ms. Pac-Man game mechanics and rendering.

Run with::

    pytest tests/env/test_ms_pacman_mechanics.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from atarax.env.games.ms_pacman import (
    MsPacMan,
    MsPacManParams,
    _COLS,
    _GHOST_COLS,
    _GHOST_ROWS,
    _PAC_COL,
    _PAC_ROW,
    _ROWS,
)

_KEY = jax.random.PRNGKey(42)


@pytest.fixture(scope="module")
def env():
    return MsPacMan()


@pytest.fixture(scope="module")
def default_params():
    return MsPacManParams(noop_max=0)


@pytest.fixture(scope="module")
def initial_state(env, default_params):
    _, state = env.reset(_KEY, default_params)
    return state


# ---------------------------------------------------------------------------
# Tile map
# ---------------------------------------------------------------------------


def test_tile_map_shape(initial_state):
    assert initial_state.tile_map.shape == (_ROWS, _COLS)


def test_tile_map_border_is_wall(initial_state):
    """Entire top and bottom rows must be walls."""
    assert jnp.all(initial_state.tile_map[0, :] == 1)
    assert jnp.all(initial_state.tile_map[_ROWS - 1, :] == 1)


def test_collectibles_shape(initial_state):
    assert initial_state.collectibles.shape == (_ROWS, _COLS)


def test_collectibles_count(initial_state):
    """154 dots/power-pellets expected in the initial map."""
    assert int(jnp.sum(initial_state.collectibles)) == 154


def test_collectibles_only_on_passable_tiles(initial_state):
    """No collectible should sit on a wall tile."""
    wall = initial_state.tile_map == 1
    overlap = wall & initial_state.collectibles
    assert not jnp.any(overlap)


# ---------------------------------------------------------------------------
# Rendering — output shape / dtype
# ---------------------------------------------------------------------------


def test_render_shape_dtype(env, initial_state):
    frame = env.render(initial_state)
    assert frame.shape == (210, 160, 3)
    assert frame.dtype == np.uint8


# ---------------------------------------------------------------------------
# hide_borders — pixel-level coverage
# ---------------------------------------------------------------------------


def _unique_bg_pixels(frame: np.ndarray) -> int:
    """Count pixels that are pure black (background colour)."""
    bg = np.array([0, 0, 0], dtype=np.uint8)
    return int(np.all(frame == bg, axis=-1).sum())


def test_hide_borders_reduces_black_pixels(env, initial_state):
    """With hide_borders=True the maze should have fewer background pixels
    than with hide_borders=False because tile gaps are filled in."""
    state_hide = initial_state  # default: hide_borders=True
    state_show = initial_state.__replace__(hide_borders=jnp.bool_(False))

    frame_hide = np.asarray(env.render(state_hide))
    frame_show = np.asarray(env.render(state_show))

    bg_hide = _unique_bg_pixels(frame_hide)
    bg_show = _unique_bg_pixels(frame_show)

    assert bg_hide < bg_show, (
        f"hide_borders=True should reduce background pixels "
        f"(got {bg_hide} vs {bg_show} with borders)"
    )


def test_hide_borders_no_gap_lines(env, initial_state):
    """With hide_borders=True, no full-width or full-height black lines should
    exist inside the maze area (rows 30–209, all 160 columns)."""
    frame = np.asarray(env.render(initial_state))  # hide_borders=True by default

    maze = frame[30:, :, :]  # strip HUD
    black = np.all(maze == 0, axis=-1)  # (180, 160) bool

    # No row inside the maze area should be entirely black
    all_black_rows = np.all(black, axis=1)
    assert not np.any(all_black_rows), (
        f"Row(s) {np.where(all_black_rows)[0] + 30} are entirely black "
        "with hide_borders=True"
    )


def test_show_borders_has_gap_lines(env, initial_state):
    """With hide_borders=False, vertical black grid lines should be present
    in the maze — verify by checking column-wise black pixel counts."""
    state_show = initial_state.__replace__(hide_borders=jnp.bool_(False))
    frame = np.asarray(env.render(state_show))

    maze = frame[30:, :, :]
    black = np.all(maze == 0, axis=-1)  # (180, 160)

    # At least one column in the maze area should have a notable run of black
    # pixels (the tile-gap lines every 4px)
    col_black_counts = black.sum(axis=0)  # (160,)
    assert np.any(col_black_counts > 10), (
        "Expected visible gap lines in maze with hide_borders=False"
    )


def test_hide_borders_default_is_true(initial_state):
    """Initial state should have hide_borders=True."""
    assert bool(initial_state.hide_borders) is True


def test_render_consistency(env, initial_state):
    """render(state) called twice on the same state returns identical frames."""
    frame_a = np.asarray(env.render(initial_state))
    frame_b = np.asarray(env.render(initial_state))
    np.testing.assert_array_equal(frame_a, frame_b)


# ---------------------------------------------------------------------------
# HUD — lives pips and score digits
# ---------------------------------------------------------------------------


def test_hud_lives_pips_visible(env, initial_state):
    """HUD area (y=0..29) should contain yellow pixels for the 3 life pips."""
    frame = np.asarray(env.render(initial_state))
    hud = frame[:30, :, :]
    # Pac-Man yellow: R≈255, G≈230, B≈0 → check for high-R, high-G, low-B pixels
    yellow = (hud[:, :, 0] > 200) & (hud[:, :, 1] > 200) & (hud[:, :, 2] < 50)
    assert yellow.sum() > 0, "Expected yellow life-pip pixels in HUD"


def test_hud_score_digits_visible(env, initial_state):
    """HUD area should contain lit score-digit pixels (warm/cream colour)."""
    frame = np.asarray(env.render(initial_state))
    hud = frame[:30, :, :]
    # Score colour is _COL_DOT ≈ (255, 230, 185) — high R, high G, moderate B
    warm = (hud[:, :, 0] > 200) & (hud[:, :, 1] > 180) & (hud[:, :, 2] > 100)
    assert warm.sum() > 0, "Expected score-digit pixels in HUD"


def test_hud_score_increases_after_eating_dot(env, default_params, initial_state):
    """Score displayed should increase after Pac-Man eats a dot."""
    score_before = int(initial_state.score)
    # Move right a few steps so Pac-Man eats a dot
    state = initial_state
    for _ in range(10):
        _, state, _, _, _ = env.step(_KEY, state, jnp.int32(2), default_params)
    assert int(state.score) > score_before, "Score should increase after eating dots"


# ---------------------------------------------------------------------------
# Collision — death and respawn
# ---------------------------------------------------------------------------


def _place_ghost_on_pac(state, ghost_idx: int = 0):
    """Return state with ghost `ghost_idx` teleported onto Pac-Man's tile."""
    new_ghost_row = state.ghost_row.at[ghost_idx].set(state.player_row)
    new_ghost_col = state.ghost_col.at[ghost_idx].set(state.player_col)
    return state.__replace__(
        ghost_row=new_ghost_row,
        ghost_col=new_ghost_col,
        power_timer=jnp.int32(0),    # ensure NOT frightened
        respawn_timer=jnp.int32(0),  # ensure NOT invincible
    )


def test_death_costs_one_life(env, default_params, initial_state):
    """Placing a non-frightened ghost on Pac-Man's tile should cost one life."""
    state = _place_ghost_on_pac(initial_state)
    lives_before = int(state.lives)
    _, state_after, _, _, _ = env.step(_KEY, state, jnp.int32(0), default_params)
    assert int(state_after.lives) == lives_before - 1, (
        f"Expected lives to drop from {lives_before} to {lives_before - 1}, "
        f"got {int(state_after.lives)}"
    )


def test_death_respawns_pac_at_start(env, default_params, initial_state):
    """After dying Pac-Man should respawn at her start position."""
    state = _place_ghost_on_pac(initial_state)
    _, state_after, _, _, _ = env.step(_KEY, state, jnp.int32(0), default_params)
    assert int(state_after.player_row) == _PAC_ROW
    assert int(state_after.player_col) == _PAC_COL


def test_death_respawns_ghosts_at_start(env, default_params, initial_state):
    """After dying all ghosts should reset to their start positions."""
    state = _place_ghost_on_pac(initial_state)
    _, state_after, _, _, _ = env.step(_KEY, state, jnp.int32(0), default_params)
    for g in range(4):
        assert int(state_after.ghost_row[g]) == _GHOST_ROWS[g]
        assert int(state_after.ghost_col[g]) == _GHOST_COLS[g]


def test_no_death_when_frightened(env, default_params, initial_state):
    """Pac-Man should NOT lose a life when colliding with a frightened ghost."""
    state = _place_ghost_on_pac(initial_state)
    state = state.__replace__(power_timer=jnp.int32(30))  # ghosts frightened
    lives_before = int(state.lives)
    _, state_after, _, _, _ = env.step(_KEY, state, jnp.int32(0), default_params)
    assert int(state_after.lives) == lives_before, (
        "Should not lose a life when ghost is frightened"
    )


def test_death_sets_respawn_timer(env, default_params, initial_state):
    """After dying, respawn_timer should be 15 (invincibility window)."""
    state = _place_ghost_on_pac(initial_state)
    _, state_after, _, _, _ = env.step(_KEY, state, jnp.int32(0), default_params)
    assert int(state_after.respawn_timer) == 15, (
        f"Expected respawn_timer=15 after death, got {int(state_after.respawn_timer)}"
    )


def test_no_death_during_respawn_invincibility(env, default_params, initial_state):
    """Pac-Man should NOT lose a second life while respawn_timer > 0."""
    # Construct state with a ghost on Pac-Man's tile, invincibility active
    state = initial_state.__replace__(respawn_timer=jnp.int32(15))
    state = state.__replace__(
        ghost_row=state.ghost_row.at[0].set(state.player_row),
        ghost_col=state.ghost_col.at[0].set(state.player_col),
        power_timer=jnp.int32(0),  # NOT frightened
    )
    lives_before = int(state.lives)
    _, state_after, _, _, _ = env.step(_KEY, state, jnp.int32(0), default_params)
    assert int(state_after.lives) == lives_before, (
        "Should not lose a life during respawn invincibility window"
    )


def test_respawn_timer_decrements(env, default_params, initial_state):
    """respawn_timer should tick down by 1 each step when no death occurs."""
    state = initial_state.__replace__(respawn_timer=jnp.int32(10))
    _, state_after, _, _, _ = env.step(_KEY, state, jnp.int32(0), default_params)
    assert int(state_after.respawn_timer) == 9, (
        f"Expected respawn_timer to decrement to 9, got {int(state_after.respawn_timer)}"
    )


def test_death_after_invincibility_expires(env, default_params, initial_state):
    """Once respawn_timer reaches 0, a ghost collision should cost a life."""
    state = _place_ghost_on_pac(initial_state)
    state = state.__replace__(respawn_timer=jnp.int32(0))  # invincibility expired
    lives_before = int(state.lives)
    _, state_after, _, _, _ = env.step(_KEY, state, jnp.int32(0), default_params)
    assert int(state_after.lives) == lives_before - 1, (
        "Should lose a life when respawn_timer has expired"
    )


# ---------------------------------------------------------------------------
# Power pellet — ghost eating
# ---------------------------------------------------------------------------


def test_frightened_ghost_eaten_scores_200(env, default_params, initial_state):
    """Eating a frightened ghost should award 200 points (first in chain)."""
    state = _place_ghost_on_pac(initial_state)
    state = state.__replace__(
        power_timer=jnp.int32(30),
        combo_count=jnp.int32(1),
        score=jnp.int32(0),
    )
    _, state_after, reward, _, _ = env.step(_KEY, state, jnp.int32(0), default_params)
    assert float(reward) == 200.0, f"Expected 200 ghost-eat reward, got {float(reward)}"


def test_frightened_ghost_respawns_at_start(env, default_params, initial_state):
    """Eaten ghost should respawn at its start position, not the collision tile."""
    ghost_idx = 0
    state = _place_ghost_on_pac(initial_state, ghost_idx)
    state = state.__replace__(power_timer=jnp.int32(30), combo_count=jnp.int32(1))
    _, state_after, _, _, _ = env.step(_KEY, state, jnp.int32(0), default_params)
    assert int(state_after.ghost_row[ghost_idx]) == _GHOST_ROWS[ghost_idx]
    assert int(state_after.ghost_col[ghost_idx]) == _GHOST_COLS[ghost_idx]


def test_combo_chain_scores_double(env, default_params, initial_state):
    """Second ghost eaten in the same power-pellet should score 400.

    Ghost 0's scatter target is corner (5, 0) — same direction Pac-Man moves
    (LEFT), so both end up on the same tile and a collision is guaranteed.
    """
    state = _place_ghost_on_pac(initial_state, ghost_idx=0)
    state = state.__replace__(
        power_timer=jnp.int32(30),
        combo_count=jnp.int32(2),  # already ate one ghost this pellet → next = 400
        score=jnp.int32(0),
    )
    _, state_after, reward, _, _ = env.step(_KEY, state, jnp.int32(0), default_params)
    assert float(reward) == 400.0, f"Expected 400 combo reward, got {float(reward)}"
