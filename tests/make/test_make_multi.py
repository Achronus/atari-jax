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

"""Tests for make_multi() and make_multi_vec().

Run with:
    pytest tests/make/test_make_multi.py -v
"""

import chex
import jax
import jax.numpy as jnp

from atarax.env import Env, VecEnv, make_multi, make_multi_vec
from atarax.env.spec import EnvSpec

_key = jax.random.PRNGKey(0)
_BREAKOUT = "atari/breakout-v0"
_BREAKOUT_SPEC = EnvSpec("atari", "breakout")
_IDS = [_BREAKOUT, _BREAKOUT_SPEC]   # one str, one EnvSpec
_N_ENVS = 2


def test_make_multi_returns_list():
    envs = make_multi([_BREAKOUT], jit_compile=False)
    assert isinstance(envs, list)


def test_make_multi_length():
    envs = make_multi(_IDS, jit_compile=False)
    assert len(envs) == len(_IDS)


def test_make_multi_all_are_env():
    envs = make_multi(_IDS, jit_compile=False)
    assert all(isinstance(e, Env) for e in envs)


def test_make_multi_reset_shape():
    envs = make_multi([_BREAKOUT], jit_compile=False)
    obs, _ = envs[0].reset(_key)
    chex.assert_shape(obs, (210, 160, 3))


def test_make_multi_accepts_env_spec():
    envs = make_multi([_BREAKOUT_SPEC], jit_compile=False)
    assert len(envs) == 1
    assert isinstance(envs[0], Env)


def test_make_multi_preset():
    envs = make_multi([_BREAKOUT], preset=True, jit_compile=False)
    obs, _ = envs[0].reset(_key)
    assert obs.shape == (84, 84, 4)


def test_make_multi_vec_returns_list():
    vec_envs = make_multi_vec([_BREAKOUT], _N_ENVS, jit_compile=False)
    assert isinstance(vec_envs, list)


def test_make_multi_vec_length():
    vec_envs = make_multi_vec(_IDS, _N_ENVS, jit_compile=False)
    assert len(vec_envs) == len(_IDS)


def test_make_multi_vec_all_are_vec_env():
    vec_envs = make_multi_vec(_IDS, _N_ENVS, jit_compile=False)
    assert all(isinstance(v, VecEnv) for v in vec_envs)


def test_make_multi_vec_n_envs():
    vec_envs = make_multi_vec([_BREAKOUT], _N_ENVS, jit_compile=False)
    assert vec_envs[0].n_envs == _N_ENVS


def test_make_multi_vec_reset_shape():
    vec_envs = make_multi_vec([_BREAKOUT], _N_ENVS, jit_compile=False)
    obs, _ = vec_envs[0].reset(_key)
    chex.assert_shape(obs, (_N_ENVS, 210, 160, 3))


def test_make_multi_vec_step_shape():
    vec_envs = make_multi_vec([_BREAKOUT], _N_ENVS, jit_compile=False)
    _, states = vec_envs[0].reset(_key)
    actions = jnp.zeros(_N_ENVS, dtype=jnp.int32)
    obs, _, reward, done, _ = vec_envs[0].step(states, actions)
    chex.assert_shape(obs, (_N_ENVS, 210, 160, 3))
    chex.assert_shape(reward, (_N_ENVS,))
    chex.assert_shape(done, (_N_ENVS,))


def test_make_multi_vec_accepts_env_spec():
    vec_envs = make_multi_vec([_BREAKOUT_SPEC], _N_ENVS, jit_compile=False)
    assert len(vec_envs) == 1
    assert isinstance(vec_envs[0], VecEnv)
