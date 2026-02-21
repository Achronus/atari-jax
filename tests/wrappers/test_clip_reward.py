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

"""Tests for ClipRewardWrapper."""

import chex
import jax
import jax.numpy as jnp

from atarax.env.wrappers import ClipRewardWrapper

_key = jax.random.PRNGKey(0)
_action = jnp.int32(0)


def test_positive_reward_clipped(fake_env):
    env = ClipRewardWrapper(fake_env)
    _, state = env.reset(_key)
    _, _, reward, _, _ = env.step(state, _action)
    assert float(reward) == 1.0


def test_large_positive_reward_clipped(fake_env_class):
    class _BigRewardEnv(fake_env_class):
        def step(self, state, action):
            obs, ns, _, done, info = super().step(state, action)
            return obs, ns, jnp.float32(100.0), done, info

    env = ClipRewardWrapper(_BigRewardEnv())
    _, state = env.reset(_key)
    _, _, reward, _, _ = env.step(state, _action)
    assert float(reward) == 1.0


def test_zero_reward_unchanged(fake_env_class):
    class _ZeroRewardEnv(fake_env_class):
        def step(self, state, action):
            obs, ns, _, done, info = super().step(state, action)
            return obs, ns, jnp.float32(0.0), done, info

    env = ClipRewardWrapper(_ZeroRewardEnv())
    _, state = env.reset(_key)
    _, _, reward, _, _ = env.step(state, _action)
    assert float(reward) == 0.0


def test_negative_reward_clipped(fake_env_class):
    class _NegRewardEnv(fake_env_class):
        def step(self, state, action):
            obs, ns, _, done, info = super().step(state, action)
            return obs, ns, jnp.float32(-5.0), done, info

    env = ClipRewardWrapper(_NegRewardEnv())
    _, state = env.reset(_key)
    _, _, reward, _, _ = env.step(state, _action)
    assert float(reward) == -1.0


def test_reward_dtype_preserved(fake_env):
    env = ClipRewardWrapper(fake_env)
    _, state = env.reset(_key)
    _, _, reward, _, _ = env.step(state, _action)
    chex.assert_type(reward, jnp.float32)
    chex.assert_rank(reward, 0)


def test_jit_compiles(fake_env):
    env = ClipRewardWrapper(fake_env)
    _, state = env.reset(_key)
    _, _, reward, done, _ = jax.jit(env.step)(state, _action)
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)
