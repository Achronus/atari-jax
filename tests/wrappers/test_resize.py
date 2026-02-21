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

"""Tests for ResizeWrapper and the preprocess() utility."""

import chex
import jax
import jax.numpy as jnp

from atari_jax.env.wrappers import GrayscaleWrapper, ResizeWrapper
from atari_jax.utils.preprocess import preprocess

_key = jax.random.PRNGKey(0)
_action = jnp.int32(0)


def test_preprocess_output_shape():
    frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)
    obs = preprocess(frame)
    chex.assert_shape(obs, (84, 84))
    chex.assert_type(obs, jnp.uint8)


def test_preprocess_jit_compiles():
    frame = jnp.zeros((210, 160, 3), dtype=jnp.uint8)
    obs = jax.jit(preprocess)(frame)
    chex.assert_shape(obs, (84, 84))


def test_reset_obs_shape(fake_env):
    env = ResizeWrapper(GrayscaleWrapper(fake_env))
    obs, _ = env.reset(_key)
    chex.assert_shape(obs, (84, 84))
    chex.assert_type(obs, jnp.uint8)


def test_step_obs_shape(fake_env):
    env = ResizeWrapper(GrayscaleWrapper(fake_env))
    _, state = env.reset(_key)
    obs, _, _, _, _ = env.step(state, _action)
    chex.assert_shape(obs, (84, 84))


def test_observation_space(fake_env):
    env = ResizeWrapper(GrayscaleWrapper(fake_env), h=84, w=84)
    assert env.observation_space.shape == (84, 84)


def test_custom_output_size(fake_env):
    env = ResizeWrapper(GrayscaleWrapper(fake_env), h=42, w=42)
    _, state = env.reset(_key)
    obs, _, _, _, _ = env.step(state, _action)
    chex.assert_shape(obs, (42, 42))
    assert env.observation_space.shape == (42, 42)


def test_jit_compiles(fake_env):
    env = ResizeWrapper(GrayscaleWrapper(fake_env))
    _, state = env.reset(_key)
    obs, _, reward, done, _ = jax.jit(env.step)(state, _action)
    chex.assert_shape(obs, (84, 84))
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)
