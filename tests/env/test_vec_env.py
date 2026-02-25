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

"""Unit tests for VecEnv.

Run with:
    pytest tests/env/test_vec_env.py -v
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env import VecEnv, make_vec

_key = jax.random.PRNGKey(0)
_BREAKOUT = "atari/breakout-v0"
_N_ENVS = 2
_N_STEPS = 4


def test_vec_env_rollout_obs_shape():
    vec_env = make_vec(_BREAKOUT, n_envs=_N_ENVS)
    _, states = vec_env.reset(_key)
    actions = jnp.zeros((_N_ENVS, _N_STEPS), dtype=jnp.int32)
    _, (obs, reward, done, _info) = vec_env.rollout(states, actions)
    chex.assert_shape(obs, (_N_ENVS, _N_STEPS, 210, 160, 3))


def test_vec_env_rollout_reward_shape():
    vec_env = make_vec(_BREAKOUT, n_envs=_N_ENVS)
    _, states = vec_env.reset(_key)
    actions = jnp.zeros((_N_ENVS, _N_STEPS), dtype=jnp.int32)
    _, (obs, reward, done, _info) = vec_env.rollout(states, actions)
    chex.assert_shape(reward, (_N_ENVS, _N_STEPS))
    chex.assert_type(reward, jnp.float32)


def test_vec_env_rollout_done_shape():
    vec_env = make_vec(_BREAKOUT, n_envs=_N_ENVS)
    _, states = vec_env.reset(_key)
    actions = jnp.zeros((_N_ENVS, _N_STEPS), dtype=jnp.int32)
    _, (obs, reward, done, _info) = vec_env.rollout(states, actions)
    chex.assert_shape(done, (_N_ENVS, _N_STEPS))


def test_vec_env_sample_shape():
    vec_env = make_vec(_BREAKOUT, n_envs=_N_ENVS)
    actions = vec_env.sample(_key)
    chex.assert_shape(actions, (_N_ENVS,))
    chex.assert_type(actions, jnp.int32)


def test_vec_env_is_instance():
    vec_env = make_vec(_BREAKOUT, n_envs=_N_ENVS)
    assert isinstance(vec_env, VecEnv)
    assert vec_env.n_envs == _N_ENVS
