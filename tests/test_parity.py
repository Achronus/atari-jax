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

"""Field-level sanity checks for the shared AtariEnv step/reset kernels.

Tests exercise `AtariEnv` directly (no wrappers) via the shared module-level
JIT kernels, so all compilations here are cache hits after the first test run.
ROM-based tests require ale-py to supply the Breakout ROM bytes.

Run with:
    pytest tests/test_parity.py -v
"""

import chex
import jax
import jax.numpy as jnp
import pytest

from atarax.env.atari_env import AtariEnv, EnvParams


@pytest.fixture(scope="module")
def env() -> AtariEnv:
    """Breakout environment with no-ops disabled for deterministic resets."""
    return AtariEnv("breakout", EnvParams(noop_max=0))


@pytest.fixture(scope="module")
def reset_state(env):
    """Reset state shared across all tests in this module."""
    _, state = env.reset(jax.random.PRNGKey(0))
    return state


def test_reset_lives_is_int32(reset_state):
    """lives field after reset should be an int32 scalar.

    The exact value depends on how far the ROM initialises before the first
    FIRE press; correctness of the 0–5 range is deferred to parity testing
    once cycle-accurate TIA timing is in place.
    """
    chex.assert_rank(reset_state.lives, 0)
    chex.assert_type(reset_state.lives, jnp.int32)


def test_step_episode_frame_increments(env, reset_state):
    """episode_frame should increment by frame_skip per RL step."""
    _, state2, _, _, _ = env.step(reset_state, jnp.int32(0))
    chex.assert_rank(state2.episode_frame, 0)
    assert int(state2.episode_frame) == int(reset_state.episode_frame) + env.default_params.frame_skip


def test_step_reward_is_float(env, reset_state):
    """Reward after a NOOP step should be a finite float32."""
    _, _, reward, _, _ = env.step(reset_state, jnp.int32(0))
    chex.assert_rank(reward, 0)
    chex.assert_type(reward, jnp.float32)
    assert jnp.isfinite(reward)


def test_step_terminal_is_bool(env, reset_state):
    """terminal field after a step should be a bool scalar."""
    _, _, _, done, _ = env.step(reset_state, jnp.int32(0))
    chex.assert_rank(done, 0)
    chex.assert_type(done, bool)


def test_reset_score_is_zero(reset_state):
    """state.score is always 0 after reset, matching ALE's m_score=0 baseline.

    Regression test: previously, score was read from RAM after warmup, which
    caused the first step reward to be negative when the ROM had written a
    non-zero score to RAM during initialisation.
    """
    chex.assert_rank(reset_state.score, 0)
    chex.assert_type(reset_state.score, jnp.int32)
    assert int(reset_state.score) == 0
