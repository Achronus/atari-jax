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

"""Tests for the AtaraxGame ABC contract — applicable to all game implementations.

Run with:
    pytest tests/game/test_jax_game.py -v
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from atarax.game import AtaraxGame, AtaraxParams
from atarax.state import AtariState, GameState
from atarax.games.breakout import Breakout

_key = jax.random.PRNGKey(0)
_params = AtaraxParams()


@pytest.fixture
def game():
    return Breakout()


def test_game_is_atarax_game(game):
    assert isinstance(game, AtaraxGame)


def test_num_actions_positive(game):
    assert isinstance(game.num_actions, int)
    assert game.num_actions > 0


def test_reset_returns_obs_and_state(game):
    result = game.reset(_key, _params)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_reset_obs_shape(game):
    obs, _ = game.reset(_key, _params)
    chex.assert_shape(obs, (210, 160, 3))
    assert obs.dtype == jnp.uint8


def test_reset_returns_atari_state(game):
    _, state = game.reset(_key, _params)
    assert isinstance(state, AtariState)
    assert isinstance(state, GameState)


def test_reset_returns_pytree(game):
    _, state = game.reset(_key, _params)
    flat, _ = jax.tree_util.tree_flatten(state)
    assert len(flat) > 0


def test_state_has_required_fields(game):
    _, state = game.reset(_key, _params)
    assert hasattr(state, "reward")
    assert hasattr(state, "done")
    assert hasattr(state, "lives")
    assert hasattr(state, "score")
    assert hasattr(state, "step")
    assert hasattr(state, "episode_step")


def test_state_field_dtypes(game):
    _, state = game.reset(_key, _params)
    assert state.reward.dtype == jnp.float32
    assert state.done.dtype == jnp.bool_
    assert state.lives.dtype == jnp.int32
    assert state.score.dtype == jnp.int32
    assert state.step.dtype == jnp.int32
    assert state.episode_step.dtype == jnp.int32


def test_step_returns_five_tuple(game):
    _, state = game.reset(_key, _params)
    result = game.step(_key, state, jnp.int32(0), _params)
    assert isinstance(result, tuple)
    assert len(result) == 5


def test_step_returns_same_state_structure(game):
    _, state = game.reset(_key, _params)
    _, new_state, _, _, _ = game.step(_key, state, jnp.int32(0), _params)
    flat_old, treedef_old = jax.tree_util.tree_flatten(state)
    flat_new, treedef_new = jax.tree_util.tree_flatten(new_state)
    assert treedef_old == treedef_new
    assert len(flat_old) == len(flat_new)


def test_render_returns_rgb_frame(game):
    _, state = game.reset(_key, _params)
    frame = game.render(state)
    chex.assert_shape(frame, (210, 160, 3))
    assert frame.dtype == jnp.uint8


def test_vmap_reset(game):
    keys = jax.random.split(_key, 4)
    obs_batch, states = jax.vmap(game.reset, in_axes=(0, None))(keys, _params)
    chex.assert_shape(obs_batch, (4, 210, 160, 3))
    chex.assert_shape(states.lives, (4,))


def test_jit_step(game):
    _, state = game.reset(_key, _params)
    step_fn = jax.jit(game.step)
    _, new_state, _, _, _ = step_fn(_key, state, jnp.int32(0), _params)
    assert hasattr(new_state, "done")
