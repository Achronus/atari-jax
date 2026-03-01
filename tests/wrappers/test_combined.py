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

"""Tests for composed wrapper stacks.

Covers stateless chains, the full DQN preprocessing stack, and vmap
compatibility.
"""

import chex
import jax
import jax.numpy as jnp

from atarax.game import AtaraxParams
from atarax.wrappers import (
    AtariPreprocessing,
    ClipReward,
    EpisodeStatisticsState,
    EpisodicLifeState,
    FrameStackState,
    GrayscaleObservation,
)

_key = jax.random.PRNGKey(0)
_params = AtaraxParams()
_action = jnp.int32(0)
_H, _W = 20, 20


def _dqn_stack(env):
    """Standard DQN preprocessing via AtariPreprocessing composite wrapper."""
    return AtariPreprocessing(env, h=_H, w=_W)


def test_stateless_chain_reset_shape(fake_env):
    env = ClipReward(GrayscaleObservation(fake_env))
    obs, _ = env.reset(_key, _params)
    chex.assert_shape(obs, (210, 160))
    assert obs.dtype == jnp.uint8


def test_stateless_chain_step_shapes(fake_env):
    env = ClipReward(GrayscaleObservation(fake_env))
    _, state = env.reset(_key, _params)
    obs, _, reward, done, _ = env.step(_key, state, _action, _params)
    chex.assert_shape(obs, (210, 160))
    chex.assert_rank(reward, 0)
    chex.assert_rank(done, 0)


def test_stateless_chain_reward_clipped(fake_env):
    env = ClipReward(GrayscaleObservation(fake_env))
    _, state = env.reset(_key, _params)
    _, _, reward, _, _ = env.step(_key, state, _action, _params)
    assert float(reward) == 1.0


def test_dqn_reset_obs_shape(fake_env):
    env = _dqn_stack(fake_env)
    obs, _ = env.reset(_key, _params)
    chex.assert_shape(obs, (_H, _W, 4))
    assert obs.dtype == jnp.uint8


def test_dqn_reset_state_types(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key, _params)
    assert isinstance(state, EpisodeStatisticsState)
    assert isinstance(state.env_state, EpisodicLifeState)
    assert isinstance(state.env_state.env_state, FrameStackState)


def test_dqn_step_obs_shape(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key, _params)
    obs, _, _, _, _ = env.step(_key, state, _action, _params)
    chex.assert_shape(obs, (_H, _W, 4))


def test_dqn_step_reward_clipped(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key, _params)
    _, _, reward, _, _ = env.step(_key, state, _action, _params)
    assert float(reward) in {-1.0, 0.0, 1.0}


def test_dqn_step_info_has_real_done(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key, _params)
    _, _, _, _, info = env.step(_key, state, _action, _params)
    assert "real_done" in info


def test_dqn_step_state_types_preserved(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key, _params)
    _, new_state, _, _, _ = env.step(_key, state, _action, _params)
    assert isinstance(new_state, EpisodeStatisticsState)
    assert isinstance(new_state.env_state, EpisodicLifeState)
    assert isinstance(new_state.env_state.env_state, FrameStackState)


def test_dqn_step_info_has_episode(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key, _params)
    _, _, _, _, info = env.step(_key, state, _action, _params)
    assert "episode" in info
    assert "r" in info["episode"]
    assert "l" in info["episode"]


def test_dqn_scan_rollout(fake_env):
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key, _params)
    actions = jnp.zeros(8, dtype=jnp.int32)
    keys = jax.random.split(_key, 8)

    def _step(carry, xs):
        rng, action = xs
        obs, new_state, reward, done, info = env.step(rng, carry, action, _params)
        return new_state, (obs, reward, done, info)

    _, (obs, reward, done, _) = jax.lax.scan(_step, state, (keys, actions))
    chex.assert_shape(obs, (8, _H, _W, 4))
    chex.assert_shape(reward, (8,))
    chex.assert_shape(done, (8,))


def test_dqn_vmap(fake_env):
    n_envs = 2
    env = _dqn_stack(fake_env)
    _, state = env.reset(_key, _params)
    states = jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_envs), state)
    actions = jnp.zeros(n_envs, dtype=jnp.int32)
    keys = jax.random.split(_key, n_envs)
    obs, _, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
        keys, states, actions, _params
    )
    chex.assert_shape(obs, (n_envs, _H, _W, 4))
    chex.assert_shape(reward, (n_envs,))
