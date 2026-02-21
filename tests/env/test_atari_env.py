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

"""Unit tests for AtariEnv and EnvParams.

ROM-free: uses new_atari_state() + FakeEnv from conftest to avoid the
ale-py dependency.

Run with:
    pytest tests/env/test_atari_env.py -v
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from atarax.env.atari_env import EnvParams
from atarax.env.spaces import Box, Discrete


def test_env_params_defaults():
    p = EnvParams()
    assert p.noop_max == 30
    assert p.frame_skip == 4
    assert p.max_episode_steps == 27000


def test_env_params_frozen():
    p = EnvParams()
    with pytest.raises((TypeError, AttributeError)):
        p.noop_max = 0  # type: ignore[misc]


def test_env_params_custom():
    p = EnvParams(noop_max=0, frame_skip=1, max_episode_steps=1000)
    assert p.noop_max == 0
    assert p.frame_skip == 1
    assert p.max_episode_steps == 1000


def test_discrete_sample_dtype():
    sp = Discrete(n=18)
    key = jax.random.PRNGKey(0)
    action = sp.sample(key)
    chex.assert_rank(action, 0)
    chex.assert_type(action, jnp.int32)
    assert 0 <= int(action) < 18


def test_box_sample_shape():
    sp = Box(low=0.0, high=255.0, shape=(210, 160, 3), dtype=jnp.uint8)
    key = jax.random.PRNGKey(0)
    obs = sp.sample(key)
    chex.assert_shape(obs, (210, 160, 3))
    chex.assert_type(obs, jnp.uint8)


def test_box_frozen():
    sp = Box(low=0.0, high=255.0, shape=(84, 84), dtype=jnp.uint8)
    with pytest.raises((TypeError, AttributeError)):
        sp.low = 1.0  # type: ignore[misc]


def test_fake_reset_obs_shape(fake_env):
    key = jax.random.PRNGKey(0)
    obs, state = fake_env.reset(key)
    chex.assert_shape(obs, (210, 160, 3))
    chex.assert_type(obs, jnp.uint8)


def test_fake_step_episode_frame_increments(fake_env):
    key = jax.random.PRNGKey(0)
    _, state = fake_env.reset(key)
    _, new_state, _, _, _ = fake_env.step(state, jnp.int32(0))
    assert int(new_state.episode_frame) == 1


def test_fake_step_jit_compiles(fake_env):
    key = jax.random.PRNGKey(0)
    _, state = fake_env.reset(key)
    jit_step = jax.jit(fake_env.step)
    obs, new_state, reward, done, info = jit_step(state, jnp.int32(0))
    chex.assert_shape(obs, (210, 160, 3))
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)


def test_sample_dtype(fake_env):
    key = jax.random.PRNGKey(42)
    action = fake_env.sample(key)
    chex.assert_rank(action, 0)
    chex.assert_type(action, jnp.int32)
    assert 0 <= int(action) < 18
