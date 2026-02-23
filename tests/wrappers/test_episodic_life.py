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

"""Tests for EpisodicLife."""

import jax
import jax.numpy as jnp

from atarax.env.wrappers import EpisodicLifeState, EpisodicLife

_key = jax.random.PRNGKey(0)
_action = jnp.int32(0)


def test_reset_state_type(fake_env):
    env = EpisodicLife(fake_env)
    _, state = env.reset(_key)
    assert isinstance(state, EpisodicLifeState)
    assert not bool(state.real_done)


def test_reset_prev_lives_initialised(fake_env):
    env = EpisodicLife(fake_env)
    _, state = env.reset(_key)
    # prev_lives is set from the inner state's lives at reset
    assert int(state.prev_lives) == int(state.env_state.lives)


def test_no_life_loss_not_terminal(fake_env):
    env = EpisodicLife(fake_env)
    _, state = env.reset(_key)
    _, _, _, done, info = env.step(state, _action)
    assert not bool(done)
    assert "real_done" in info
    assert not bool(info["real_done"])


def test_life_loss_signals_terminal(fake_env_class):
    class _LifeLossEnv(fake_env_class):
        def step(self, state, action):
            obs, ns, reward, _, info = super().step(state, action)
            ns = ns.__replace__(lives=ns.lives - jnp.int32(1))
            info = {"lives": ns.lives, "episode_frame": ns.episode_frame}
            return obs, ns, reward, jnp.bool_(False), info

    env = EpisodicLife(_LifeLossEnv())
    _, state = env.reset(_key)
    state = EpisodicLifeState(
        env_state=state.env_state,
        prev_lives=jnp.int32(3),
        real_done=jnp.bool_(False),
    )
    _, _, _, done, info = env.step(state, _action)
    assert bool(done), "should signal terminal on life loss"
    assert not bool(info["real_done"]), (
        "real_done should remain False on mere life loss"
    )


def test_game_over_signals_both(fake_env_class):
    class _GameOverEnv(fake_env_class):
        def step(self, state, action):
            obs, ns, reward, _, info = super().step(state, action)
            ns = ns.__replace__(lives=jnp.int32(0))
            info = {"lives": ns.lives, "episode_frame": ns.episode_frame}
            return obs, ns, reward, jnp.bool_(True), info

    env = EpisodicLife(_GameOverEnv())
    _, state = env.reset(_key)
    state = EpisodicLifeState(
        env_state=state.env_state,
        prev_lives=jnp.int32(3),
        real_done=jnp.bool_(False),
    )
    _, _, _, done, info = env.step(state, _action)
    assert bool(done)
    assert bool(info["real_done"])
