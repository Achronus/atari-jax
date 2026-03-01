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

"""Tests for AtaraxGame — reset/step shapes, vmap, spaces.

Run with:
    pytest tests/env/test_jax_game_env.py -v
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxGame, AtaraxParams
from envrax.spaces import Box, Discrete
from atarax.state import AtariState, GameState
from atarax.games.breakout import Breakout, BreakoutState

_key = jax.random.PRNGKey(0)
_params = AtaraxParams()


def _make_env():
    return Breakout()


def test_is_atarax_game():
    env = _make_env()
    assert isinstance(env, AtaraxGame)


def test_observation_space():
    env = _make_env()
    space = env.observation_space
    assert isinstance(space, Box)
    assert space.shape == (210, 160, 3)


def test_action_space():
    env = _make_env()
    space = env.action_space
    assert isinstance(space, Discrete)
    assert space.n == Breakout.num_actions


def test_reset_state_type():
    env = _make_env()
    _, state = env.reset(_key, _params)
    assert isinstance(state, GameState)
    assert isinstance(state, AtariState)
    assert isinstance(state, BreakoutState)


def test_reset_obs_shape():
    env = _make_env()
    obs, _ = env.reset(_key, _params)
    chex.assert_shape(obs, (210, 160, 3))
    assert obs.dtype == jnp.uint8


def test_step_obs_shape():
    env = _make_env()
    _, state = env.reset(_key, _params)
    obs, _, _, _, _ = env.step(_key, state, jnp.int32(0), _params)
    chex.assert_shape(obs, (210, 160, 3))


def test_step_reward_scalar():
    env = _make_env()
    _, state = env.reset(_key, _params)
    _, _, reward, _, _ = env.step(_key, state, jnp.int32(0), _params)
    chex.assert_rank(reward, 0)
    assert reward.dtype == jnp.float32


def test_step_done_scalar():
    env = _make_env()
    _, state = env.reset(_key, _params)
    _, _, _, done, _ = env.step(_key, state, jnp.int32(0), _params)
    chex.assert_rank(done, 0)
    assert done.dtype == jnp.bool_


def test_step_info_keys():
    env = _make_env()
    _, state = env.reset(_key, _params)
    _, _, _, _, info = env.step(_key, state, jnp.int32(0), _params)
    assert "lives" in info
    assert "score" in info
    assert "episode_step" in info


def test_vmap_reset():
    env = _make_env()
    keys = jax.random.split(_key, 8)
    obs, _ = jax.vmap(env.reset, in_axes=(0, None))(keys, _params)
    chex.assert_shape(obs, (8, 210, 160, 3))


def test_repr():
    env = _make_env()
    r = repr(env)
    assert "AtaraxGame" in r
    assert "breakout" in r.lower()


def test_env_params_noop_max():
    params = AtaraxParams(noop_max=0)
    env = Breakout()
    obs, _ = env.reset(_key, params)
    chex.assert_shape(obs, (210, 160, 3))
