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

"""Tests for RecordEpisodeStatistics."""

import chex
import jax
import jax.numpy as jnp

from atarax.env.wrappers import EpisodeStatisticsState, RecordEpisodeStatistics

_key = jax.random.PRNGKey(0)
_action = jnp.int32(0)


def test_reset_state_type(fake_env):
    env = RecordEpisodeStatistics(fake_env)
    _, state = env.reset(_key)
    assert isinstance(state, EpisodeStatisticsState)


def test_reset_accumulators_zeroed(fake_env):
    env = RecordEpisodeStatistics(fake_env)
    _, state = env.reset(_key)
    assert float(state.episode_return) == 0.0
    assert int(state.episode_length) == 0


def test_step_increments_length(fake_env):
    env = RecordEpisodeStatistics(fake_env)
    _, state = env.reset(_key)
    _, new_state, _, _, _ = env.step(state, _action)
    assert int(new_state.episode_length) == 1


def test_step_accumulates_reward(fake_env):
    env = RecordEpisodeStatistics(fake_env)
    _, state = env.reset(_key)
    _, new_state, reward, _, _ = env.step(state, _action)
    assert float(new_state.episode_return) == float(reward)


def test_multi_step_accumulation(fake_env):
    env = RecordEpisodeStatistics(fake_env)
    _, state = env.reset(_key)
    total_reward = 0.0
    for _ in range(4):
        _, state, reward, _, _ = env.step(state, _action)
        total_reward += float(reward)
    assert float(state.episode_return) == total_reward
    assert int(state.episode_length) == 4


def test_info_has_episode_key(fake_env):
    env = RecordEpisodeStatistics(fake_env)
    _, state = env.reset(_key)
    _, _, _, _, info = env.step(state, _action)
    assert "episode" in info
    assert "r" in info["episode"]
    assert "l" in info["episode"]


def test_info_episode_zero_while_running(fake_env):
    env = RecordEpisodeStatistics(fake_env)
    _, state = env.reset(_key)
    _, _, _, done, info = env.step(state, _action)
    assert not bool(done)
    assert float(info["episode"]["r"]) == 0.0
    assert int(info["episode"]["l"]) == 0


def test_info_episode_nonzero_at_done(fake_env_class):
    class _DoneEnv(fake_env_class):
        def step(self, state, action):
            obs, ns, reward, _, info = super().step(state, action)
            return obs, ns, jnp.float32(5.0), jnp.bool_(True), info

    env = RecordEpisodeStatistics(_DoneEnv())
    _, state = env.reset(_key)
    _, _, _, done, info = env.step(state, _action)
    assert bool(done)
    assert float(info["episode"]["r"]) == 5.0
    assert int(info["episode"]["l"]) == 1


def test_accumulators_reset_after_done(fake_env_class):
    class _DoneEnv(fake_env_class):
        def step(self, state, action):
            obs, ns, reward, _, info = super().step(state, action)
            return obs, ns, jnp.float32(3.0), jnp.bool_(True), info

    env = RecordEpisodeStatistics(_DoneEnv())
    _, state = env.reset(_key)
    _, new_state, _, done, _ = env.step(state, _action)
    assert bool(done)
    assert float(new_state.episode_return) == 0.0
    assert int(new_state.episode_length) == 0


def test_episode_return_dtype(fake_env):
    env = RecordEpisodeStatistics(fake_env)
    _, state = env.reset(_key)
    chex.assert_type(state.episode_return, jnp.float32)


def test_episode_length_dtype(fake_env):
    env = RecordEpisodeStatistics(fake_env)
    _, state = env.reset(_key)
    chex.assert_type(state.episode_length, jnp.int32)


def test_jit_compiles(fake_env):
    env = RecordEpisodeStatistics(fake_env)
    _, state = env.reset(_key)
    obs, new_state, reward, done, info = jax.jit(env.step)(state, _action)
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)
    assert isinstance(new_state, EpisodeStatisticsState)
    assert "episode" in info
