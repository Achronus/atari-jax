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

from atarax import VmapEnv, make_vec

_key = jax.random.PRNGKey(0)
_BREAKOUT = "atari/breakout-v0"


def test_make_vec_returns_vmap_env():
    vec_env, _ = make_vec(_BREAKOUT, n_envs=2, jit_compile=False)
    assert isinstance(vec_env, VmapEnv)
    assert vec_env.num_envs == 2


def test_make_vec_reset_shape():
    vec_env, params = make_vec(_BREAKOUT, n_envs=2, jit_compile=False)
    obs, _ = vec_env.reset(_key, params)
    chex.assert_shape(obs, (2, 210, 160, 3))


def test_make_vec_step_shape():
    vec_env, params = make_vec(_BREAKOUT, n_envs=2, jit_compile=False)
    _, states = vec_env.reset(_key, params)
    obs, _, reward, done, _ = vec_env.step(_key, states, jnp.zeros(2, dtype=jnp.int32), params)
    chex.assert_shape(obs, (2, 210, 160, 3))
    chex.assert_shape(reward, (2,))


def test_make_vec_preset_reset_shape():
    n_envs = 2
    vec_env, params = make_vec(_BREAKOUT, n_envs=n_envs, preset=True, jit_compile=False)
    obs, _ = vec_env.reset(_key, params)
    assert obs.shape == (n_envs, 84, 84, 4)


def test_make_vec_preset_step_shape():
    n_envs = 2
    vec_env, params = make_vec(_BREAKOUT, n_envs=n_envs, preset=True, jit_compile=False)
    _, states = vec_env.reset(_key, params)
    actions = jnp.zeros(n_envs, dtype=jnp.int32)
    obs, _, reward, done, _ = vec_env.step(_key, states, actions, params)
    assert obs.shape == (n_envs, 84, 84, 4)
    assert reward.shape == (n_envs,)
    assert done.shape == (n_envs,)
