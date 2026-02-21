"""Parity and compilation tests for the Breakout step/reset loop.

ROM-based tests require ale-py (in the test dependency group) to supply the
Breakout ROM bytes.  Step-exact reward parity with ALE is deferred until the
emulator achieves cycle-accurate TIA timing.

Run with:
    pytest tests/test_parity.py -v
"""

import chex
import jax
import jax.numpy as jnp
import pytest
from ale_py.roms import get_rom_path

from atari_jax.core.state import new_atari_state
from atari_jax.games.roms.breakout import reset, step


@pytest.fixture(scope="module")
def breakout_rom() -> chex.Array:
    """Load the Breakout ROM bytes from ale-py as a JAX uint8 array."""
    rom_path = get_rom_path("breakout")
    data = bytearray(rom_path.read_bytes())
    return jnp.array(data, dtype=jnp.uint8)


@pytest.fixture(scope="module")
def reset_state(breakout_rom) -> tuple:
    """JIT-compiled reset — shared across tests in this module."""
    return jax.jit(reset)(breakout_rom), breakout_rom


def test_step_jit_compiles(breakout_rom):
    """jax.jit(step) should trace and compile without error."""
    state = new_atari_state()
    jit_step = jax.jit(step)
    out = jit_step(state, breakout_rom, jnp.int32(0))
    chex.assert_rank(out.episode_frame, 0)
    assert out.episode_frame == jnp.int32(1)


def test_reset_jit_compiles(breakout_rom):
    """jax.jit(reset) should trace and compile without error."""
    state = jax.jit(reset)(breakout_rom)
    chex.assert_rank(state.episode_frame, 0)
    assert state.episode_frame == jnp.int32(0)
    assert not bool(state.terminal)
    assert float(state.reward) == 0.0


def test_step_vmap(breakout_rom):
    """jax.vmap over 4 independent environments should compile and run."""
    n_envs = 4
    state = new_atari_state()
    batched = jax.tree_util.tree_map(lambda x: jnp.stack([x] * n_envs), state)
    actions = jnp.zeros(n_envs, dtype=jnp.int32)
    vmapped = jax.vmap(lambda s, a: step(s, breakout_rom, a))
    out = vmapped(batched, actions)
    chex.assert_shape(out.episode_frame, (n_envs,))
    assert jnp.all(out.episode_frame == jnp.int32(1))


def test_reset_lives_is_int32(reset_state):
    """lives field after reset should be an int32 scalar.

    The exact value depends on how far the ROM initialises before the first
    FIRE press; correctness of the 0–5 range is deferred to parity testing
    once cycle-accurate TIA timing is in place.
    """
    state, _ = reset_state
    chex.assert_rank(state.lives, 0)
    chex.assert_type(state.lives, jnp.int32)


def test_step_episode_frame_increments(reset_state):
    """episode_frame should increment by 1 per step."""
    state, rom = reset_state
    state2 = jax.jit(step)(state, rom, jnp.int32(0))
    chex.assert_rank(state2.episode_frame, 0)
    assert int(state2.episode_frame) == int(state.episode_frame) + 1


def test_step_reward_is_float(reset_state):
    """Reward after a NOOP step should be a finite float32."""
    state, rom = reset_state
    state2 = jax.jit(step)(state, rom, jnp.int32(0))
    chex.assert_rank(state2.reward, 0)
    chex.assert_type(state2.reward, jnp.float32)
    assert jnp.isfinite(state2.reward)


def test_step_terminal_is_bool(reset_state):
    """terminal field after a step should be a bool scalar."""
    state, rom = reset_state
    state2 = jax.jit(step)(state, rom, jnp.int32(0))
    chex.assert_rank(state2.terminal, 0)
    chex.assert_type(state2.terminal, bool)
