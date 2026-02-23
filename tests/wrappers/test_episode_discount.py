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

"""Tests for EpisodeDiscount."""

import chex
import jax
import jax.numpy as jnp

from atarax.env.wrappers import EpisodeDiscount

_key = jax.random.PRNGKey(0)
_action = jnp.int32(0)


def test_discount_when_not_done(fake_env):
    env = EpisodeDiscount(fake_env)
    _, state = env.reset(_key)
    _, _, _, discount, _ = env.step(state, _action)
    assert float(discount) == 1.0


def test_discount_when_done(fake_env_class):
    class _DoneEnv(fake_env_class):
        def step(self, state, action):
            obs, ns, reward, _, info = super().step(state, action)
            return obs, ns, reward, jnp.bool_(True), info

    env = EpisodeDiscount(_DoneEnv())
    _, state = env.reset(_key)
    _, _, _, discount, _ = env.step(state, _action)
    assert float(discount) == 0.0


def test_discount_dtype(fake_env):
    env = EpisodeDiscount(fake_env)
    _, state = env.reset(_key)
    _, _, _, discount, _ = env.step(state, _action)
    chex.assert_type(discount, jnp.float32)


def test_discount_rank(fake_env):
    env = EpisodeDiscount(fake_env)
    _, state = env.reset(_key)
    _, _, _, discount, _ = env.step(state, _action)
    chex.assert_rank(discount, 0)


def test_obs_passthrough(fake_env):
    env = EpisodeDiscount(fake_env)
    obs_reset, state = env.reset(_key)
    obs_step, _, _, _, _ = env.step(state, _action)
    assert obs_reset.shape == obs_step.shape


def test_reward_passthrough(fake_env_class):
    class _KnownRewardEnv(fake_env_class):
        def step(self, state, action):
            obs, ns, _, done, info = super().step(state, action)
            return obs, ns, jnp.float32(7.0), done, info

    env = EpisodeDiscount(_KnownRewardEnv())
    _, state = env.reset(_key)
    _, _, reward, _, _ = env.step(state, _action)
    assert float(reward) == 7.0
