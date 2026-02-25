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

"""Tests for the make_vec() factory function.

Run with:
    pytest tests/make/test_make_vec.py -v
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env import VecEnv, make_vec
from atarax.env.wrappers import EpisodeStatisticsState

_key = jax.random.PRNGKey(0)
_BREAKOUT = "atari/breakout-v0"


def test_make_vec_returns_vec_env():
    vec_env = make_vec(_BREAKOUT, n_envs=2, jit_compile=False)
    assert isinstance(vec_env, VecEnv)
    assert vec_env.n_envs == 2


def test_make_vec_reset_shape():
    vec_env = make_vec(_BREAKOUT, n_envs=2, jit_compile=False)
    obs, _ = vec_env.reset(_key)
    chex.assert_shape(obs, (2, 210, 160, 3))


def test_make_vec_step_shape():
    vec_env = make_vec(_BREAKOUT, n_envs=2, jit_compile=False)
    _, states = vec_env.reset(_key)
    obs, _, reward, done, _ = vec_env.step(states, jnp.zeros(2, dtype=jnp.int32))
    chex.assert_shape(obs, (2, 210, 160, 3))
    chex.assert_shape(reward, (2,))


def test_make_vec_preset_reset_shape():
    n_envs = 2
    vec_env = make_vec(_BREAKOUT, n_envs=n_envs, preset=True, jit_compile=False)
    obs, _ = vec_env.reset(_key)
    assert obs.shape == (n_envs, 84, 84, 4)


def test_make_vec_preset_step_shape():
    n_envs = 2
    vec_env = make_vec(_BREAKOUT, n_envs=n_envs, preset=True, jit_compile=False)
    _, states = vec_env.reset(_key)
    actions = jnp.zeros(n_envs, dtype=jnp.int32)
    obs, _, reward, done, _ = vec_env.step(states, actions)
    assert obs.shape == (n_envs, 84, 84, 4)
    assert reward.shape == (n_envs,)
    assert done.shape == (n_envs,)


def test_make_vec_rollout_shape():
    n_envs = 2
    n_steps = 4
    vec_env = make_vec(_BREAKOUT, n_envs=n_envs, preset=True, jit_compile=False)
    _, states = vec_env.reset(_key)
    actions = jnp.zeros((n_envs, n_steps), dtype=jnp.int32)
    _, (obs, reward, done, _info) = vec_env.rollout(states, actions)
    assert obs.shape == (n_envs, n_steps, 84, 84, 4)
    assert reward.shape == (n_envs, n_steps)
    assert done.shape == (n_envs, n_steps)
