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

"""Tests for AtariPreprocessing."""

import chex
import jax
import jax.numpy as jnp

from atarax.env.wrappers import (
    AtariPreprocessing,
    EpisodeStatisticsState,
    EpisodicLifeState,
    FrameStackState,
)

_key = jax.random.PRNGKey(0)
_action = jnp.int32(0)
_H, _W = 20, 20


def test_reset_obs_shape(fake_env):
    env = AtariPreprocessing(fake_env, h=_H, w=_W)
    obs, _ = env.reset(_key)
    chex.assert_shape(obs, (_H, _W, 4))
    chex.assert_type(obs, jnp.uint8)


def test_reset_state_type(fake_env):
    env = AtariPreprocessing(fake_env, h=_H, w=_W)
    _, state = env.reset(_key)
    assert isinstance(state, EpisodeStatisticsState)


def test_step_obs_shape(fake_env):
    env = AtariPreprocessing(fake_env, h=_H, w=_W)
    _, state = env.reset(_key)
    obs, _, _, _, _ = env.step(state, _action)
    chex.assert_shape(obs, (_H, _W, 4))


def test_step_reward_clipped(fake_env):
    env = AtariPreprocessing(fake_env, h=_H, w=_W)
    _, state = env.reset(_key)
    _, _, reward, _, _ = env.step(state, _action)
    assert float(reward) in {-1.0, 0.0, 1.0}


def test_step_info_has_real_done(fake_env):
    env = AtariPreprocessing(fake_env, h=_H, w=_W)
    _, state = env.reset(_key)
    _, _, _, _, info = env.step(state, _action)
    assert "real_done" in info


def test_step_info_has_episode(fake_env):
    env = AtariPreprocessing(fake_env, h=_H, w=_W)
    _, state = env.reset(_key)
    _, _, _, _, info = env.step(state, _action)
    assert "episode" in info
    assert "r" in info["episode"]
    assert "l" in info["episode"]


def test_step_state_types(fake_env):
    env = AtariPreprocessing(fake_env, h=_H, w=_W)
    _, state = env.reset(_key)
    _, new_state, _, _, _ = env.step(state, _action)
    assert isinstance(new_state, EpisodeStatisticsState)
    assert isinstance(new_state.env_state, EpisodicLifeState)
    assert isinstance(new_state.env_state.env_state, FrameStackState)


def test_custom_size(fake_env):
    env = AtariPreprocessing(fake_env, h=42, w=42)
    obs, _ = env.reset(_key)
    chex.assert_shape(obs, (42, 42, 4))


def test_custom_n_stack(fake_env):
    env = AtariPreprocessing(fake_env, h=_H, w=_W, n_stack=2)
    obs, _ = env.reset(_key)
    chex.assert_shape(obs, (_H, _W, 2))


def test_observation_space(fake_env):
    env = AtariPreprocessing(fake_env)
    assert env.observation_space.shape == (84, 84, 4)
    assert env.observation_space.dtype == jnp.uint8


def test_action_space_delegated(fake_env):
    env = AtariPreprocessing(fake_env, h=_H, w=_W)
    assert env.action_space.n == 18


def test_vmap_compatible(fake_env):
    env = AtariPreprocessing(fake_env, h=_H, w=_W)
    _, state = env.reset(_key)
    states = jax.tree_util.tree_map(lambda x: jnp.stack([x, x]), state)
    actions = jnp.zeros(2, dtype=jnp.int32)
    obs, _, reward, done, _ = jax.vmap(env.step)(states, actions)
    chex.assert_shape(obs, (2, _H, _W, 4))
    chex.assert_shape(reward, (2,))
    chex.assert_shape(done, (2,))
