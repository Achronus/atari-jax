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

"""Unit tests for the make() and make_vec() factory functions.

ROM-backed: requires ale-py to load game ROMs.

Run with:
    pytest tests/env/test_make.py -v
"""

import jax
import jax.numpy as jnp
import pytest

from atarax.env import (
    AtariEnv,
    EpisodicLifeState,
    GrayscaleWrapper,
    VecEnv,
    make,
    make_vec,
)

_key = jax.random.PRNGKey(0)


def test_make_raw_returns_atari_env():
    env = make("breakout")
    assert isinstance(env, AtariEnv)


def test_make_wrappers_list():
    env = make("breakout", wrappers=[GrayscaleWrapper])
    assert isinstance(env, GrayscaleWrapper)


def test_make_preset_dqn_obs_shape():
    env = make("breakout", preset=True)
    obs, state = env.reset(_key)
    assert obs.shape == (84, 84, 4)
    assert isinstance(state, EpisodicLifeState)


def test_make_preset_and_wrappers_raises():
    with pytest.raises(ValueError, match="not both"):
        make("breakout", wrappers=[GrayscaleWrapper], preset=True)


def test_make_vec_returns_vec_env():
    vec_env = make_vec("breakout", n_envs=2, preset=True)
    assert isinstance(vec_env, VecEnv)
    assert vec_env.n_envs == 2


def test_make_vec_reset_shape():
    n_envs = 2
    vec_env = make_vec("breakout", n_envs=n_envs, preset=True)
    obs, states = vec_env.reset(_key)
    assert obs.shape == (n_envs, 84, 84, 4)
    assert isinstance(states, EpisodicLifeState)


def test_make_vec_step_shape():
    n_envs = 2
    vec_env = make_vec("breakout", n_envs=n_envs, preset=True)
    _, states = vec_env.reset(_key)
    actions = jnp.zeros(n_envs, dtype=jnp.int32)
    obs, new_states, reward, done, _ = vec_env.step(states, actions)
    assert obs.shape == (n_envs, 84, 84, 4)
    assert reward.shape == (n_envs,)
    assert done.shape == (n_envs,)


def test_make_vec_rollout_shape():
    n_envs = 2
    n_steps = 4
    vec_env = make_vec("breakout", n_envs=n_envs, preset=True)
    _, states = vec_env.reset(_key)
    actions = jnp.zeros((n_envs, n_steps), dtype=jnp.int32)
    _, (obs, reward, done, _info) = vec_env.rollout(states, actions)
    assert obs.shape == (n_envs, n_steps, 84, 84, 4)
    assert reward.shape == (n_envs, n_steps)
    assert done.shape == (n_envs, n_steps)


def test_make_jit_compile_true_reset():
    env = make("breakout", preset=True, jit_compile=True)
    obs, state = env.reset(_key)
    assert obs.shape == (84, 84, 4)
    assert isinstance(state, EpisodicLifeState)


def test_make_jit_compile_true_step():
    env = make("breakout", preset=True, jit_compile=True)
    _, state = env.reset(_key)
    obs, new_state, reward, done, _ = env.step(state, env.sample(_key))
    assert obs.shape == (84, 84, 4)
    assert reward.shape == ()
    assert done.shape == ()


def test_make_jit_compile_false_reset():
    env = make("breakout", preset=True, jit_compile=False)
    obs, state = env.reset(_key)
    assert obs.shape == (84, 84, 4)
    assert isinstance(state, EpisodicLifeState)


def test_make_vec_jit_compile_true_reset():
    n_envs = 2
    vec_env = make_vec("breakout", n_envs=n_envs, preset=True, jit_compile=True)
    obs, states = vec_env.reset(_key)
    assert obs.shape == (n_envs, 84, 84, 4)
    assert isinstance(states, EpisodicLifeState)


def test_make_vec_jit_compile_true_step():
    n_envs = 2
    vec_env = make_vec("breakout", n_envs=n_envs, preset=True, jit_compile=True)
    _, states = vec_env.reset(_key)
    actions = jnp.zeros(n_envs, dtype=jnp.int32)
    obs, _, reward, done, _ = vec_env.step(states, actions)
    assert obs.shape == (n_envs, 84, 84, 4)
    assert reward.shape == (n_envs,)
    assert done.shape == (n_envs,)


def test_make_vec_jit_compile_true_rollout():
    n_envs = 2
    n_steps = 4
    vec_env = make_vec("breakout", n_envs=n_envs, preset=True, jit_compile=True)
    _, states = vec_env.reset(_key)
    actions = jnp.zeros((n_envs, n_steps), dtype=jnp.int32)
    _, (obs, reward, done, _) = vec_env.rollout(states, actions)
    assert obs.shape == (n_envs, n_steps, 84, 84, 4)
    assert reward.shape == (n_envs, n_steps)
    assert done.shape == (n_envs, n_steps)


def test_make_vec_jit_compile_false_reset():
    n_envs = 2
    vec_env = make_vec("breakout", n_envs=n_envs, preset=True, jit_compile=False)
    obs, states = vec_env.reset(_key)
    assert obs.shape == (n_envs, 84, 84, 4)
    assert isinstance(states, EpisodicLifeState)
