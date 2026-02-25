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

"""Tests for the AtariEnv ABC contract — applicable to all game implementations.

Run with:
    pytest tests/game/test_jax_game.py -v
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from atarax.env._base import Env
from atarax.env.atari_env import AtariEnv
from atarax.games._base import AtariState, GameState
from atarax.games.breakout import Breakout

_key = jax.random.PRNGKey(0)


@pytest.fixture
def game():
    return Breakout()


def test_game_is_atari_env(game):
    assert isinstance(game, AtariEnv)


def test_game_is_env(game):
    assert isinstance(game, Env)


def test_num_actions_positive(game):
    assert isinstance(game.num_actions, int)
    assert game.num_actions > 0


def test_reset_returns_obs_and_state(game):
    result = game.reset(_key)
    assert isinstance(result, tuple)
    assert len(result) == 2


def test_reset_obs_shape(game):
    obs, _ = game.reset(_key)
    chex.assert_shape(obs, (210, 160, 3))
    chex.assert_type(obs, jnp.uint8)


def test_reset_returns_atari_state(game):
    _, state = game.reset(_key)
    assert isinstance(state, AtariState)
    assert isinstance(state, GameState)


def test_reset_returns_pytree(game):
    _, state = game.reset(_key)
    flat, _ = jax.tree_util.tree_flatten(state)
    assert len(flat) > 0


def test_state_has_required_fields(game):
    _, state = game.reset(_key)
    assert hasattr(state, "reward")
    assert hasattr(state, "done")
    assert hasattr(state, "lives")
    assert hasattr(state, "score")
    assert hasattr(state, "step")
    assert hasattr(state, "episode_step")


def test_state_field_dtypes(game):
    _, state = game.reset(_key)
    chex.assert_type(state.reward, jnp.float32)
    chex.assert_type(state.done, jnp.bool_)
    chex.assert_type(state.lives, jnp.int32)
    chex.assert_type(state.score, jnp.int32)
    chex.assert_type(state.step, jnp.int32)
    chex.assert_type(state.episode_step, jnp.int32)


def test_step_returns_five_tuple(game):
    _, state = game.reset(_key)
    result = game.step(state, jnp.int32(0))
    assert isinstance(result, tuple)
    assert len(result) == 5


def test_step_returns_same_state_structure(game):
    _, state = game.reset(_key)
    _, new_state, _, _, _ = game.step(state, jnp.int32(0))
    flat_old, treedef_old = jax.tree_util.tree_flatten(state)
    flat_new, treedef_new = jax.tree_util.tree_flatten(new_state)
    assert treedef_old == treedef_new
    assert len(flat_old) == len(flat_new)


def test_render_returns_rgb_frame(game):
    _, state = game.reset(_key)
    frame = game.render(state)
    chex.assert_shape(frame, (210, 160, 3))
    chex.assert_type(frame, jnp.uint8)


def test_vmap_reset(game):
    keys = jax.random.split(_key, 4)
    obs_batch, states = jax.vmap(game.reset)(keys)
    chex.assert_shape(obs_batch, (4, 210, 160, 3))
    chex.assert_shape(states.lives, (4,))


def test_jit_step(game):
    _, state = game.reset(_key)
    step_fn = jax.jit(game.step)
    _, new_state, _, _, _ = step_fn(state, jnp.int32(0))
    assert hasattr(new_state, "done")


def test_scan_rollout(game):
    _, state = game.reset(_key)
    actions = jnp.zeros(10, dtype=jnp.int32)
    _, (obs_seq, rew_seq, done_seq, _) = game.rollout(state, actions)
    chex.assert_shape(obs_seq, (10, 210, 160, 3))
    chex.assert_shape(rew_seq, (10,))
    chex.assert_shape(done_seq, (10,))
