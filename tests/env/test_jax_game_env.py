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

"""Tests for AtariEnv — reset/step shapes, vmap, lax.scan rollout.

Run with:
    pytest tests/env/test_jax_game_env.py -v
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env._base import Env
from atarax.env.atari_env import AtariEnv, EnvParams
from atarax.env.spaces import Box, Discrete
from atarax.games._base import AtariState, GameState
from atarax.games.breakout import Breakout, BreakoutState

_key = jax.random.PRNGKey(0)


def _make_env(params=None):
    return Breakout(params)


def test_is_env_subclass():
    env = _make_env()
    assert isinstance(env, Env)


def test_is_atari_env():
    env = _make_env()
    assert isinstance(env, AtariEnv)


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
    _, state = env.reset(_key)
    assert isinstance(state, GameState)
    assert isinstance(state, AtariState)
    assert isinstance(state, BreakoutState)


def test_reset_obs_shape():
    env = _make_env()
    obs, _ = env.reset(_key)
    chex.assert_shape(obs, (210, 160, 3))
    chex.assert_type(obs, jnp.uint8)


def test_step_obs_shape():
    env = _make_env()
    _, state = env.reset(_key)
    obs, _, _, _, _ = env.step(state, jnp.int32(0))
    chex.assert_shape(obs, (210, 160, 3))


def test_step_reward_scalar():
    env = _make_env()
    _, state = env.reset(_key)
    _, _, reward, _, _ = env.step(state, jnp.int32(0))
    chex.assert_rank(reward, 0)
    chex.assert_type(reward, jnp.float32)


def test_step_done_scalar():
    env = _make_env()
    _, state = env.reset(_key)
    _, _, _, done, _ = env.step(state, jnp.int32(0))
    chex.assert_rank(done, 0)
    chex.assert_type(done, jnp.bool_)


def test_step_info_keys():
    env = _make_env()
    _, state = env.reset(_key)
    _, _, _, _, info = env.step(state, jnp.int32(0))
    assert "lives" in info
    assert "score" in info
    assert "episode_step" in info


def test_sample_shape():
    env = _make_env()
    action = env.sample(_key)
    chex.assert_rank(action, 0)
    chex.assert_type(action, jnp.int32)


def test_vmap_reset():
    env = _make_env()
    keys = jax.random.split(_key, 8)
    obs, _ = jax.vmap(env.reset)(keys)
    chex.assert_shape(obs, (8, 210, 160, 3))


def test_lax_scan_rollout():
    env = _make_env()
    _, state = env.reset(_key)
    actions = jnp.zeros(10, dtype=jnp.int32)
    _, (obs_seq, rew_seq, done_seq, _) = env.rollout(state, actions)
    chex.assert_shape(obs_seq, (10, 210, 160, 3))
    chex.assert_shape(rew_seq, (10,))
    chex.assert_shape(done_seq, (10,))


def test_unwrapped_returns_self():
    env = _make_env()
    assert env.unwrapped is env


def test_repr():
    env = _make_env()
    r = repr(env)
    assert "AtariEnv" in r
    assert "breakout" in r.lower()


def test_env_params_noop_max():
    params = EnvParams(noop_max=0)
    env = Breakout(params)
    obs, _ = env.reset(_key)
    chex.assert_shape(obs, (210, 160, 3))
