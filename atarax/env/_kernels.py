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

"""Shared module-level JIT kernels for all Atari games.

Two compile modes are supported, selectable via `AtariEnv(compile_mode=...)`:

``"all"`` (default)
    `game_id` is a dynamic JAX argument at the JIT boundary.  Every game with
    the same ROM size and the same `frame_skip` / `max_episode_steps` values
    shares a **single XLA compilation** (3 programs total).  All 57 game
    branches are compiled into each program via `jax.lax.switch`.

``"single"``
    `game_id` is a **static** Python-level constant at the JIT boundary.
    JAX constant-folds the `jax.lax.switch` dispatch calls, tracing only the
    one selected game branch.  Each game produces its own XLA program (3
    programs per game), which is smaller and faster to compile.

Public API — ``"all"`` mode
---------------------------
jit_step        -- Single-env frame-skip step.
jit_reset       -- Single-env reset with warmup + stochastic start.
jit_rollout     -- Single-env multi-step rollout via `lax.scan`.
jit_vec_step    -- Vectorised frame-skip step (n_envs in parallel).
jit_vec_reset   -- Vectorised reset.
jit_vec_rollout -- Vectorised multi-step rollout via `lax.scan`.
_jit_sample     -- Shared action sampler (identical for all games).

Public API — ``"single"`` mode
-------------------------------
jit_step_single        -- Single-env step (game_id static).
jit_reset_single       -- Single-env reset (game_id static).
jit_rollout_single     -- Single-env rollout (game_id static).
jit_vec_step_single    -- Vectorised step (game_id static).
jit_vec_reset_single   -- Vectorised reset (game_id static).
jit_vec_rollout_single -- Vectorised rollout (game_id static).
"""

import functools
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from atarax.core.cpu import cpu_reset
from atarax.core.frame import emulate_frame
from atarax.core.state import AtariState, new_atari_state
from atarax.games import compute_reward_and_score, get_lives, is_terminal


def _base_step(
    state: AtariState,
    rom: chex.Array,
    action: chex.Array,
    game_id: chex.Array,
) -> AtariState:
    """
    Emulate one ALE frame and update all episode fields.

    This is the primitive used inside frame-skip loops.  It is not JIT-compiled
    itself so that it can be efficiently inlined by the enclosing `jit_step` /
    `jit_vec_step` JIT boundaries.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : chex.Array
        uint8[ROM_SIZE] — ROM bytes (dynamic argument; not a JIT constant).
    action : chex.Array
        int32 — ALE action index.
    game_id : chex.Array
        int32 — Index into the game registry.

    Returns
    -------
    state : AtariState
        Updated state after the frame with `score`, `reward`, `lives`,
        `terminal`, and `episode_frame` populated.
    """
    ram_prev = state.riot.ram
    lives_prev = state.lives
    state = emulate_frame(state, rom, action)
    ram_curr = state.riot.ram
    reward, new_score = compute_reward_and_score(
        game_id, ram_prev, ram_curr, state.score
    )
    return state.__replace__(
        score=new_score,
        reward=reward,
        lives=get_lives(game_id, ram_curr),
        terminal=is_terminal(game_id, ram_curr, lives_prev),
        episode_frame=(state.episode_frame + jnp.int32(1)).astype(jnp.int32),
    )


_vmapped_base_step = jax.vmap(_base_step, in_axes=(0, None, 0, None))


# ---------------------------------------------------------------------------
# Private body functions — shared by "all" and "single" JIT variants.
# ---------------------------------------------------------------------------


def _reset_fn(
    key: chex.Array,
    rom: chex.Array,
    game_id,
    warmup_frames: chex.Array,
    noop_max: chex.Array,
) -> Tuple[chex.Array, AtariState]:
    state = new_atari_state()
    state = cpu_reset(state, rom)
    state = jax.lax.fori_loop(
        0, 10, lambda _, s: emulate_frame(s, rom, jnp.int32(0)), state
    )
    state = emulate_frame(state, rom, jnp.int32(1))
    remaining = jnp.maximum(warmup_frames - jnp.int32(11), jnp.int32(0))
    state = jax.lax.fori_loop(
        0, remaining, lambda _, s: emulate_frame(s, rom, jnp.int32(0)), state
    )
    state = state.__replace__(
        score=jnp.int32(0),
        lives=get_lives(game_id, state.riot.ram),
        episode_frame=jnp.int32(0),
        terminal=jnp.bool_(False),
        reward=jnp.float32(0.0),
    )
    noop_steps = jax.random.randint(key, shape=(), minval=0, maxval=noop_max + 1)
    state = jax.lax.fori_loop(
        0,
        noop_steps,
        lambda _, s: _base_step(s, rom, jnp.int32(0), game_id),
        state,
    )
    return state.screen, state


def _step_fn(
    state: AtariState,
    rom: chex.Array,
    action: chex.Array,
    game_id,
    frame_skip: int,
    max_episode_steps: int,
) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
    lives_before = state.lives

    def _skip(carry, _):
        s, acc = carry
        new_s = _base_step(s, rom, action, game_id)
        return (new_s, acc + new_s.reward), None

    (new_state, total_reward), _ = jax.lax.scan(
        _skip, (state, jnp.float32(0.0)), None, length=frame_skip
    )

    terminal = is_terminal(game_id, new_state.riot.ram, lives_before)
    truncated = new_state.episode_frame >= jnp.int32(max_episode_steps)
    done = terminal | truncated
    new_state = new_state.__replace__(reward=total_reward, terminal=done)
    info = {
        "lives": new_state.lives,
        "episode_frame": new_state.episode_frame,
        "truncated": truncated,
    }
    return new_state.screen, new_state, total_reward, done, info


def _rollout_fn(
    state: AtariState,
    rom: chex.Array,
    actions: chex.Array,
    game_id,
    frame_skip: int,
    max_episode_steps: int,
) -> Tuple[AtariState, Tuple]:
    def _step(carry_state, action):
        obs, new_state, reward, done, info = _step_fn(
            carry_state, rom, action, game_id, frame_skip, max_episode_steps
        )
        return new_state, (obs, reward, done, info)

    return jax.lax.scan(_step, state, actions)


def _vec_reset_fn(
    keys: chex.Array,
    rom: chex.Array,
    game_id,
    warmup_frames: chex.Array,
    noop_max: chex.Array,
) -> Tuple[chex.Array, AtariState]:
    return jax.vmap(lambda k: _reset_fn(k, rom, game_id, warmup_frames, noop_max))(keys)


def _vec_step_fn(
    states: AtariState,
    rom: chex.Array,
    actions: chex.Array,
    game_id,
    frame_skip: int,
    max_episode_steps: int,
) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
    lives_before = states.lives

    def _skip(carry, _):
        ss, acc = carry
        new_ss = _vmapped_base_step(ss, rom, actions, game_id)
        return (new_ss, acc + new_ss.reward), None

    (new_states, total_rewards), _ = jax.lax.scan(
        _skip, (states, jnp.zeros_like(states.reward)), None, length=frame_skip
    )

    terminals = jax.vmap(is_terminal, in_axes=(None, 0, 0))(
        game_id, new_states.riot.ram, lives_before
    )
    truncated = new_states.episode_frame >= jnp.int32(max_episode_steps)
    done = terminals | truncated
    new_states = new_states.__replace__(reward=total_rewards, terminal=done)
    info = {
        "lives": new_states.lives,
        "episode_frame": new_states.episode_frame,
        "truncated": truncated,
    }
    return new_states.screen, new_states, total_rewards, done, info


def _vec_rollout_fn(
    states: AtariState,
    rom: chex.Array,
    actions: chex.Array,
    game_id,
    frame_skip: int,
    max_episode_steps: int,
) -> Tuple[AtariState, Tuple]:
    def _step(carry_states, t_actions):
        screens, new_states, rewards, done, info = _vec_step_fn(
            carry_states, rom, t_actions, game_id, frame_skip, max_episode_steps
        )
        return new_states, (screens, rewards, done, info)

    return jax.lax.scan(_step, states, jnp.moveaxis(actions, 1, 0))


# ---------------------------------------------------------------------------
# "all" mode — game_id is a dynamic JAX array; all 57 branches compile once.
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(4, 5))
def jit_step(
    state: AtariState,
    rom: chex.Array,
    action: chex.Array,
    game_id: chex.Array,
    frame_skip: int,
    max_episode_steps: int,
) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
    """
    Advance the environment by one RL step (frame_skip emulator frames).

    `rom` and `game_id` are dynamic JAX arguments so all games with the same
    ROM size share a single XLA compilation.  `frame_skip` and
    `max_episode_steps` are Python-level static args required by `lax.scan`.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : chex.Array
        uint8[ROM_SIZE] — ROM bytes.
    action : chex.Array
        int32 — ALE action index (repeated for every skipped frame).
    game_id : chex.Array
        int32 — Index into the game registry.
    frame_skip : int
        Number of emulator frames to repeat each action (static).
    max_episode_steps : int
        Hard episode length limit (static).

    Returns
    -------
    obs : chex.Array
        uint8[210, 160, 3] — Observation after the step.
    new_state : AtariState
        Updated machine state.
    reward : chex.Array
        float32 — Total reward accumulated over skipped frames.
    done : chex.Array
        bool — True when the episode has ended.
    info : dict
        ``{"lives": int32, "episode_frame": int32, "truncated": bool}``
    """
    return _step_fn(state, rom, action, game_id, frame_skip, max_episode_steps)


@jax.jit
def jit_reset(
    key: chex.Array,
    rom: chex.Array,
    game_id: chex.Array,
    warmup_frames: chex.Array,
    noop_max: chex.Array,
) -> Tuple[chex.Array, AtariState]:
    """
    Initialise the machine, run warm-up frames, and apply random no-ops.

    `rom`, `game_id`, `warmup_frames`, and `noop_max` are all dynamic JAX
    arguments so all games with the same ROM size share a single XLA
    compilation.

    Parameters
    ----------
    key : chex.Array
        JAX PRNG key used to sample the number of no-op steps.
    rom : chex.Array
        uint8[ROM_SIZE] — ROM bytes.
    game_id : chex.Array
        int32 — Index into the game registry.
    warmup_frames : chex.Array
        int32 — Total warm-up frames for this game (from `WARMUP_FRAMES_ARRAY`).
    noop_max : chex.Array
        int32 — Maximum number of NOOP actions at episode start.

    Returns
    -------
    obs : chex.Array
        uint8[210, 160, 3] — First RGB observation.
    state : AtariState
        Initial machine state after reset and no-ops.
    """
    return _reset_fn(key, rom, game_id, warmup_frames, noop_max)


@functools.partial(jax.jit, static_argnums=(4, 5))
def jit_rollout(
    state: AtariState,
    rom: chex.Array,
    actions: chex.Array,
    game_id: chex.Array,
    frame_skip: int,
    max_episode_steps: int,
) -> Tuple[AtariState, Tuple]:
    """
    Run a compiled single-env multi-step rollout.

    Parameters
    ----------
    state : AtariState
        Initial machine state.
    rom : chex.Array
        uint8[ROM_SIZE] — ROM bytes.
    actions : chex.Array
        int32[T] — Action sequence; T determines the number of RL steps.
    game_id : chex.Array
        int32 — Index into the game registry.
    frame_skip : int
        Number of emulator frames per RL step (static).
    max_episode_steps : int
        Hard episode length limit (static).

    Returns
    -------
    final_state : AtariState
        State after all T steps.
    transitions : tuple
        ``(obs, reward, done, info)`` each with a leading T dimension.
    """
    return _rollout_fn(state, rom, actions, game_id, frame_skip, max_episode_steps)


@jax.jit
def jit_vec_reset(
    keys: chex.Array,
    rom: chex.Array,
    game_id: chex.Array,
    warmup_frames: chex.Array,
    noop_max: chex.Array,
) -> Tuple[chex.Array, AtariState]:
    """
    Reset `n_envs` environments with independent random starts.

    `n_envs` is inferred from the leading dimension of `keys`.

    Parameters
    ----------
    keys : chex.Array
        JAX PRNG keys, one per environment (shape [n_envs, 2]).
    rom : chex.Array
        uint8[ROM_SIZE] — Shared ROM bytes (not batched).
    game_id : chex.Array
        int32 — Shared game index (not batched).
    warmup_frames : chex.Array
        int32 — Warm-up frame count for this game.
    noop_max : chex.Array
        int32 — Maximum NOOP actions at episode start.

    Returns
    -------
    obs : chex.Array
        uint8[n_envs, 210, 160, 3] — First observations.
    states : AtariState
        Batched initial machine states.
    """
    return _vec_reset_fn(keys, rom, game_id, warmup_frames, noop_max)


@functools.partial(jax.jit, static_argnums=(4, 5))
def jit_vec_step(
    states: AtariState,
    rom: chex.Array,
    actions: chex.Array,
    game_id: chex.Array,
    frame_skip: int,
    max_episode_steps: int,
) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
    """
    Advance `n_envs` environments by one RL step simultaneously.

    `n_envs` is inferred from the leading dimension of `states` / `actions` —
    it does not need to be a static argument.

    Parameters
    ----------
    states : AtariState
        Batched machine states (leading dim = n_envs).
    rom : chex.Array
        uint8[ROM_SIZE] — Shared ROM bytes (not batched).
    actions : chex.Array
        int32[n_envs] — One action per environment.
    game_id : chex.Array
        int32 — Shared game index (not batched).
    frame_skip : int
        Number of emulator frames per RL step (static).
    max_episode_steps : int
        Hard episode length limit (static).

    Returns
    -------
    obs : chex.Array
        uint8[n_envs, 210, 160, 3] — Observations after the step.
    new_states : AtariState
        Updated batched states.
    reward : chex.Array
        float32[n_envs] — Per-environment rewards.
    done : chex.Array
        bool[n_envs] — Per-environment terminal flags.
    info : dict
        Batched info dict; each value has a leading n_envs dimension.
    """
    return _vec_step_fn(states, rom, actions, game_id, frame_skip, max_episode_steps)


@functools.partial(jax.jit, static_argnums=(4, 5))
def jit_vec_rollout(
    states: AtariState,
    rom: chex.Array,
    actions: chex.Array,
    game_id: chex.Array,
    frame_skip: int,
    max_episode_steps: int,
) -> Tuple[AtariState, Tuple]:
    """
    Run a compiled multi-step rollout across all environments.

    Each scan iteration advances all `n_envs` environments by one RL step
    (which internally applies `frame_skip` emulator frames).  No Python
    loop overhead.

    Parameters
    ----------
    states : AtariState
        Batched initial states (leading dim = n_envs).
    rom : chex.Array
        uint8[ROM_SIZE] — Shared ROM bytes (not batched).
    actions : chex.Array
        int32[n_envs, T] — Action sequence per environment.
    game_id : chex.Array
        int32 — Shared game index (not batched).
    frame_skip : int
        Number of emulator frames per RL step (static).
    max_episode_steps : int
        Hard episode length limit (static).

    Returns
    -------
    final_states : AtariState
        Batched states after all T steps.
    transitions : tuple
        ``(obs, reward, done, info)`` each with shape ``[T, n_envs, ...]``.
    """
    return _vec_rollout_fn(states, rom, actions, game_id, frame_skip, max_episode_steps)


_jit_sample = jax.jit(
    lambda key: jax.random.randint(key, shape=(), minval=0, maxval=18, dtype=jnp.int32)
)


# ---------------------------------------------------------------------------
# "single" mode — game_id is a static Python int; only one branch compiles.
# ---------------------------------------------------------------------------


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def jit_step_single(
    state: AtariState,
    rom: chex.Array,
    action: chex.Array,
    game_id: int,
    frame_skip: int,
    max_episode_steps: int,
) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
    """
    Single-mode variant of `jit_step`.

    `game_id` is a **static** Python int so JAX constant-folds the
    `jax.lax.switch` dispatch, tracing only the selected game branch.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    rom : chex.Array
        uint8[ROM_SIZE] — ROM bytes.
    action : chex.Array
        int32 — ALE action index.
    game_id : int
        Python int — Index into the game registry (static).
    frame_skip : int
        Number of emulator frames per RL step (static).
    max_episode_steps : int
        Hard episode length limit (static).

    Returns
    -------
    obs : chex.Array
        uint8[210, 160, 3] — Observation after the step.
    new_state : AtariState
        Updated machine state.
    reward : chex.Array
        float32 — Total reward accumulated over skipped frames.
    done : chex.Array
        bool — True when the episode has ended.
    info : dict
        ``{"lives": int32, "episode_frame": int32, "truncated": bool}``
    """
    return _step_fn(state, rom, action, game_id, frame_skip, max_episode_steps)


@functools.partial(jax.jit, static_argnums=(2,))
def jit_reset_single(
    key: chex.Array,
    rom: chex.Array,
    game_id: int,
    warmup_frames: chex.Array,
    noop_max: chex.Array,
) -> Tuple[chex.Array, AtariState]:
    """
    Single-mode variant of `jit_reset`.

    `game_id` is a **static** Python int so JAX constant-folds the
    `jax.lax.switch` dispatch, tracing only the selected game branch.

    Parameters
    ----------
    key : chex.Array
        JAX PRNG key used to sample the number of no-op steps.
    rom : chex.Array
        uint8[ROM_SIZE] — ROM bytes.
    game_id : int
        Python int — Index into the game registry (static).
    warmup_frames : chex.Array
        int32 — Total warm-up frames for this game.
    noop_max : chex.Array
        int32 — Maximum number of NOOP actions at episode start.

    Returns
    -------
    obs : chex.Array
        uint8[210, 160, 3] — First RGB observation.
    state : AtariState
        Initial machine state after reset and no-ops.
    """
    return _reset_fn(key, rom, game_id, warmup_frames, noop_max)


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def jit_rollout_single(
    state: AtariState,
    rom: chex.Array,
    actions: chex.Array,
    game_id: int,
    frame_skip: int,
    max_episode_steps: int,
) -> Tuple[AtariState, Tuple]:
    """
    Single-mode variant of `jit_rollout`.

    `game_id` is a **static** Python int so JAX constant-folds the
    `jax.lax.switch` dispatch, tracing only the selected game branch.

    Parameters
    ----------
    state : AtariState
        Initial machine state.
    rom : chex.Array
        uint8[ROM_SIZE] — ROM bytes.
    actions : chex.Array
        int32[T] — Action sequence; T determines the number of RL steps.
    game_id : int
        Python int — Index into the game registry (static).
    frame_skip : int
        Number of emulator frames per RL step (static).
    max_episode_steps : int
        Hard episode length limit (static).

    Returns
    -------
    final_state : AtariState
        State after all T steps.
    transitions : tuple
        ``(obs, reward, done, info)`` each with a leading T dimension.
    """
    return _rollout_fn(state, rom, actions, game_id, frame_skip, max_episode_steps)


@functools.partial(jax.jit, static_argnums=(2,))
def jit_vec_reset_single(
    keys: chex.Array,
    rom: chex.Array,
    game_id: int,
    warmup_frames: chex.Array,
    noop_max: chex.Array,
) -> Tuple[chex.Array, AtariState]:
    """
    Single-mode variant of `jit_vec_reset`.

    `game_id` is a **static** Python int so JAX constant-folds the
    `jax.lax.switch` dispatch, tracing only the selected game branch.

    Parameters
    ----------
    keys : chex.Array
        JAX PRNG keys, one per environment (shape [n_envs, 2]).
    rom : chex.Array
        uint8[ROM_SIZE] — Shared ROM bytes (not batched).
    game_id : int
        Python int — Shared game index (static).
    warmup_frames : chex.Array
        int32 — Warm-up frame count for this game.
    noop_max : chex.Array
        int32 — Maximum NOOP actions at episode start.

    Returns
    -------
    obs : chex.Array
        uint8[n_envs, 210, 160, 3] — First observations.
    states : AtariState
        Batched initial machine states.
    """
    return _vec_reset_fn(keys, rom, game_id, warmup_frames, noop_max)


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def jit_vec_step_single(
    states: AtariState,
    rom: chex.Array,
    actions: chex.Array,
    game_id: int,
    frame_skip: int,
    max_episode_steps: int,
) -> Tuple[chex.Array, AtariState, chex.Array, chex.Array, Dict[str, Any]]:
    """
    Single-mode variant of `jit_vec_step`.

    `game_id` is a **static** Python int so JAX constant-folds the
    `jax.lax.switch` dispatch, tracing only the selected game branch.

    Parameters
    ----------
    states : AtariState
        Batched machine states (leading dim = n_envs).
    rom : chex.Array
        uint8[ROM_SIZE] — Shared ROM bytes (not batched).
    actions : chex.Array
        int32[n_envs] — One action per environment.
    game_id : int
        Python int — Shared game index (static).
    frame_skip : int
        Number of emulator frames per RL step (static).
    max_episode_steps : int
        Hard episode length limit (static).

    Returns
    -------
    obs : chex.Array
        uint8[n_envs, 210, 160, 3] — Observations after the step.
    new_states : AtariState
        Updated batched states.
    reward : chex.Array
        float32[n_envs] — Per-environment rewards.
    done : chex.Array
        bool[n_envs] — Per-environment terminal flags.
    info : dict
        Batched info dict; each value has a leading n_envs dimension.
    """
    return _vec_step_fn(states, rom, actions, game_id, frame_skip, max_episode_steps)


@functools.partial(jax.jit, static_argnums=(3, 4, 5))
def jit_vec_rollout_single(
    states: AtariState,
    rom: chex.Array,
    actions: chex.Array,
    game_id: int,
    frame_skip: int,
    max_episode_steps: int,
) -> Tuple[AtariState, Tuple]:
    """
    Single-mode variant of `jit_vec_rollout`.

    `game_id` is a **static** Python int so JAX constant-folds the
    `jax.lax.switch` dispatch, tracing only the selected game branch.

    Parameters
    ----------
    states : AtariState
        Batched initial states (leading dim = n_envs).
    rom : chex.Array
        uint8[ROM_SIZE] — Shared ROM bytes (not batched).
    actions : chex.Array
        int32[n_envs, T] — Action sequence per environment.
    game_id : int
        Python int — Shared game index (static).
    frame_skip : int
        Number of emulator frames per RL step (static).
    max_episode_steps : int
        Hard episode length limit (static).

    Returns
    -------
    final_states : AtariState
        Batched states after all T steps.
    transitions : tuple
        ``(obs, reward, done, info)`` each with shape ``[T, n_envs, ...]``.
    """
    return _vec_rollout_fn(states, rom, actions, game_id, frame_skip, max_episode_steps)
