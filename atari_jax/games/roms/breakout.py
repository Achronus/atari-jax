"""Breakout — reward and terminal extraction from RIOT RAM."""

import jax.numpy as jnp
import jax

# RAM addresses (decimal indices into the 128-byte RIOT RAM array).
# Source: Arcade-Learning-Environment src/ale/games/supported/Breakout.cpp
LIVES_ADDR = 57    # 0x39 — lives remaining; starts at 5, terminal when 0
SCORE_X    = 77    # 0x4D — ones and tens digits (BCD packed)
SCORE_Y    = 76    # 0x4C — hundreds digit (BCD packed)


def _score(ram: jax.Array) -> jax.Array:
    """
    Decode the packed BCD score from two RAM bytes.

    Parameters
    ----------
    ram : jax.Array
        uint8[128] — RIOT RAM snapshot.

    Returns
    -------
    score : jax.Array
        int32 — Decoded score value.
    """
    x = ram[SCORE_X].astype(jnp.int32)
    y = ram[SCORE_Y].astype(jnp.int32)
    return jnp.int32(1) * (x & 0xF) + jnp.int32(10) * ((x >> 4) & 0xF) + jnp.int32(100) * (y & 0xF)


def get_lives(ram: jax.Array) -> jax.Array:
    """
    Read the lives counter from RAM.

    Parameters
    ----------
    ram : jax.Array
        uint8[128] — RIOT RAM snapshot.

    Returns
    -------
    lives : jax.Array
        int32 — Lives remaining (0–5).
    """
    return ram[LIVES_ADDR].astype(jnp.int32)


def get_reward(ram_prev: jax.Array, ram_curr: jax.Array) -> jax.Array:
    """
    Compute the reward earned in the last step as a score delta.

    Parameters
    ----------
    ram_prev : jax.Array
        uint8[128] — RIOT RAM before the step.
    ram_curr : jax.Array
        uint8[128] — RIOT RAM after the step.

    Returns
    -------
    reward : jax.Array
        float32 — Score gained this step (non-negative under normal play).
    """
    return (_score(ram_curr) - _score(ram_prev)).astype(jnp.float32)


def is_terminal(ram: jax.Array, lives_prev: jax.Array) -> jax.Array:
    """
    Determine whether the episode has ended.

    The episode is terminal when the game had started (lives_prev > 0) and
    the current lives count has reached zero.

    Parameters
    ----------
    ram : jax.Array
        uint8[128] — RIOT RAM after the step.
    lives_prev : jax.Array
        int32 — Lives count from before the step.

    Returns
    -------
    terminal : jax.Array
        bool — True when the episode ended on this step.
    """
    return (lives_prev > jnp.int32(0)) & (get_lives(ram) == jnp.int32(0))
