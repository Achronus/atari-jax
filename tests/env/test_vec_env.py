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

"""Unit tests for make_rollout_fn.

ROM-free: uses FakeEnv from conftest.

Run with:
    pytest tests/env/test_vec_env.py -v
"""

import chex
import jax
import jax.numpy as jnp

from atari_jax.core.state import new_atari_state
from atari_jax.env.vec_env import make_rollout_fn

_N_STEPS = 8


def test_rollout_fn_obs_shape(fake_env):
    rollout = make_rollout_fn(fake_env)
    state = new_atari_state()
    actions = jnp.zeros(_N_STEPS, dtype=jnp.int32)
    _, (obs, reward, done, info) = rollout(state, actions)
    chex.assert_shape(obs, (_N_STEPS, 210, 160, 3))


def test_rollout_fn_reward_shape(fake_env):
    rollout = make_rollout_fn(fake_env)
    state = new_atari_state()
    actions = jnp.zeros(_N_STEPS, dtype=jnp.int32)
    _, (obs, reward, done, info) = rollout(state, actions)
    chex.assert_shape(reward, (_N_STEPS,))
    chex.assert_type(reward, jnp.float32)


def test_rollout_fn_done_shape(fake_env):
    rollout = make_rollout_fn(fake_env)
    state = new_atari_state()
    actions = jnp.zeros(_N_STEPS, dtype=jnp.int32)
    _, (obs, reward, done, info) = rollout(state, actions)
    chex.assert_shape(done, (_N_STEPS,))


def test_rollout_fn_jit_compiles(fake_env):
    rollout = jax.jit(make_rollout_fn(fake_env))
    state = new_atari_state()
    actions = jnp.zeros(_N_STEPS, dtype=jnp.int32)
    final_state, transitions = rollout(state, actions)
    chex.assert_rank(final_state.episode_frame, 0)
    assert int(final_state.episode_frame) == _N_STEPS


def test_rollout_fn_vmap_two_envs(fake_env):
    n_envs = 2
    rollout = make_rollout_fn(fake_env)
    batched = jax.vmap(rollout)
    state = new_atari_state()
    states = jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_envs), state)
    actions = jnp.zeros((n_envs, _N_STEPS), dtype=jnp.int32)
    _, (obs, reward, done, info) = batched(states, actions)
    chex.assert_shape(obs, (n_envs, _N_STEPS, 210, 160, 3))
    chex.assert_shape(reward, (n_envs, _N_STEPS))
