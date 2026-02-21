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

"""Tests for FrameStackWrapper."""

import chex
import jax
import jax.numpy as jnp

from atarax.env.wrappers import (
    FrameStackState,
    FrameStackWrapper,
    GrayscaleWrapper,
    ResizeWrapper,
)

_key = jax.random.PRNGKey(0)
_action = jnp.int32(0)


def _make_env(fake_env, n_stack=4):
    return FrameStackWrapper(ResizeWrapper(GrayscaleWrapper(fake_env)), n_stack=n_stack)


def test_reset_obs_shape(fake_env):
    env = _make_env(fake_env)
    obs, state = env.reset(_key)
    chex.assert_shape(obs, (84, 84, 4))
    assert isinstance(state, FrameStackState)


def test_step_obs_shape(fake_env):
    env = _make_env(fake_env)
    _, state = env.reset(_key)
    obs, _, _, _, _ = env.step(state, _action)
    chex.assert_shape(obs, (84, 84, 4))


def test_observation_space(fake_env):
    env = _make_env(fake_env)
    assert env.observation_space.shape == (84, 84, 4)


def test_custom_n_stack(fake_env):
    env = _make_env(fake_env, n_stack=8)
    obs, state = env.reset(_key)
    chex.assert_shape(obs, (84, 84, 8))
    assert env.observation_space.shape == (84, 84, 8)


def test_rolls_oldest_frame(fake_env):
    env = _make_env(fake_env)
    _, state = env.reset(_key)
    _, new_state, _, _, _ = env.step(state, _action)
    # channels 0..2 of the new stack == channels 1..3 of the old stack
    assert jnp.all(new_state.obs_stack[..., :3] == state.obs_stack[..., 1:])


def test_jit_compiles(fake_env):
    env = _make_env(fake_env)
    _, state = env.reset(_key)
    obs, new_state, reward, done, _ = jax.jit(env.step)(state, _action)
    chex.assert_shape(obs, (84, 84, 4))
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)
    assert isinstance(new_state, FrameStackState)
    chex.assert_shape(obs, (84, 84, 4))
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)
    assert isinstance(new_state, FrameStackState)
