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

"""Tests for composed wrapper stacks.

Covers stateless chains, the full DQN preprocessing stack, JIT compilation,
and rollout compatibility.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env.vec_env import make_rollout_fn
from atarax.env.wrappers import (
    AtariPreprocessing,
    ClipReward,
    EpisodeStatisticsState,
    EpisodicLifeState,
    FrameStackState,
    GrayscaleObservation,
)

_key = jax.random.PRNGKey(0)
_action = jnp.int32(0)


def _dqn_stack(env):
    """Standard DQN preprocessing via AtariPreprocessing composite wrapper."""
    return AtariPreprocessing(env)


def test_stateless_chain_reset_shape(fake_env):
    env = ClipReward(GrayscaleObservation(fake_env))
    obs, _ = env.reset(_key)
    chex.assert_shape(obs, (210, 160))
    chex.assert_type(obs, jnp.uint8)


def test_stateless_chain_step_shapes(fake_env):
    env = ClipReward(GrayscaleObservation(fake_env))
    _, state = env.reset(_key)
    obs, _, reward, done, _ = env.step(state, _action)
    chex.assert_shape(obs, (210, 160))
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)


def test_stateless_chain_reward_clipped(fake_env):
    env = ClipReward(GrayscaleObservation(fake_env))
    _, state = env.reset(_key)
    _, _, reward, _, _ = env.step(state, _action)
    assert float(reward) == 1.0


def test_dqn_reset_obs_shape(fake_env):
    env = _dqn_stack(fake_env)
    obs, state = env.reset(_key)
    chex.assert_shape(obs, (84, 84, 4))
    chex.assert_type(obs, jnp.uint8)


def test_dqn_reset_state_types(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key)
    assert isinstance(state, EpisodeStatisticsState)
    assert isinstance(state.env_state, EpisodicLifeState)
    assert isinstance(state.env_state.env_state, FrameStackState)


def test_dqn_step_obs_shape(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key)
    obs, _, _, _, _ = env.step(state, _action)
    chex.assert_shape(obs, (84, 84, 4))


def test_dqn_step_reward_clipped(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key)
    _, _, reward, _, _ = env.step(state, _action)
    assert float(reward) in {-1.0, 0.0, 1.0}


def test_dqn_step_info_has_real_done(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key)
    _, _, _, _, info = env.step(state, _action)
    assert "real_done" in info


def test_dqn_step_state_types_preserved(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key)
    _, new_state, _, _, _ = env.step(state, _action)
    assert isinstance(new_state, EpisodeStatisticsState)
    assert isinstance(new_state.env_state, EpisodicLifeState)
    assert isinstance(new_state.env_state.env_state, FrameStackState)


def test_dqn_step_info_has_episode(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key)
    _, _, _, _, info = env.step(state, _action)
    assert "episode" in info
    assert "r" in info["episode"]
    assert "l" in info["episode"]


def test_dqn_jit_compiles(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key)
    obs, new_state, reward, done, info = jax.jit(env.step)(state, _action)
    chex.assert_shape(obs, (84, 84, 4))
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)


def test_dqn_rollout_obs_shape(fake_env):
    env = _dqn_stack(fake_env)
    rollout = make_rollout_fn(env)
    _, state = env.reset(_key)
    actions = jnp.zeros(8, dtype=jnp.int32)
    _, (obs, reward, done, info) = rollout(state, actions)
    chex.assert_shape(obs, (8, 84, 84, 4))
    chex.assert_shape(reward, (8,))
    chex.assert_shape(done, (8,))


def test_dqn_rollout_jit_compiles(fake_env):
    env = _dqn_stack(fake_env)
    rollout = jax.jit(make_rollout_fn(env))
    _, state = env.reset(_key)
    actions = jnp.zeros(8, dtype=jnp.int32)
    final_state, (obs, reward, done, info) = rollout(state, actions)
    chex.assert_shape(obs, (8, 84, 84, 4))
    assert isinstance(final_state, EpisodeStatisticsState)


def test_dqn_rollout_vmap(fake_env):
    n_envs = 2
    env = _dqn_stack(fake_env)
    rollout = jax.vmap(make_rollout_fn(env))
    _, state = env.reset(_key)
    states = jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_envs), state)
    actions = jnp.zeros((n_envs, 8), dtype=jnp.int32)
    _, (obs, reward, done, info) = rollout(states, actions)
    chex.assert_shape(obs, (n_envs, 8, 84, 84, 4))
    chex.assert_shape(reward, (n_envs, 8))
