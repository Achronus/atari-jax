"""AtariState pytree — single JAX-traceable state container for all hardware."""

import chex
import jax
import jax.numpy as jnp


@chex.dataclass
class CPUState:
    """
    MOS 6507 CPU registers.

    Parameters
    ----------
    a : jax.Array
        uint8 — Accumulator.
    x : jax.Array
        uint8 — X index register.
    y : jax.Array
        uint8 — Y index register.
    sp : jax.Array
        uint8 — Stack pointer (page-1 offset, grows downward).
    pc : jax.Array
        uint16 — Program counter.
    flags : jax.Array
        uint8 — Processor status byte: N V 1 B D I Z C (bits 7..0).
        Bit positions: 7=N  6=V  5=1(always)  4=B  3=D  2=I  1=Z  0=C.
    """

    a: jax.Array
    x: jax.Array
    y: jax.Array
    sp: jax.Array
    pc: jax.Array
    flags: jax.Array


@chex.dataclass
class RIOTState:
    """
    M6532 RIOT — 128-byte RAM, interval timer, and I/O ports.

    Parameters
    ----------
    ram : jax.Array
        uint8[128] — General-purpose working RAM.
    timer : jax.Array
        uint8 — Value written at the last TIM*T write.
    interval_shift : jax.Array
        uint8 — Divide-by exponent: 0/3/6/10 maps to ÷1/÷8/÷64/÷1024.
    cycles_when_set : jax.Array
        int32 — `state.cycles` captured at the moment of the last timer
        write.  Used to compute the live timer value on demand rather than
        storing a decrementing counter.
    port_a : jax.Array
        uint8 — Joystick byte, active-low (0xFF = no buttons pressed).
    port_b : jax.Array
        uint8 — Console switches, active-low (0xFF = default positions).
    """

    ram: jax.Array
    timer: jax.Array
    interval_shift: jax.Array
    cycles_when_set: jax.Array
    port_a: jax.Array
    port_b: jax.Array


@chex.dataclass
class TIAState:
    """
    TIA — Television Interface Adaptor.

    Parameters
    ----------
    regs : jax.Array
        uint8[64] — Shadow copy of the 64 TIA write-only registers.
        `regs[i]` holds the last value written to TIA address `i & 0x3F`.
    collisions : jax.Array
        uint16 — 15 collision-latch bits (Phase 2).
    scanline : jax.Array
        uint8[160] — Colour-index pixel buffer for the current scanline.
    """

    regs: jax.Array
    collisions: jax.Array
    scanline: jax.Array


@chex.dataclass
class AtariState:
    """
    Complete Atari 2600 machine state.

    Parameters
    ----------
    cpu : CPUState
        MOS 6507 CPU registers.
    riot : RIOTState
        M6532 RIOT chip (RAM, timer, I/O).
    tia : TIAState
        TIA chip (register shadow + scanline buffer).
    bank : jax.Array
        uint8 — Active ROM bank index (used by the cartridge read path).
    screen : jax.Array
        uint8[210, 160, 3] — Rendered RGB frame.
    frame : jax.Array
        int32 — Total emulated frames since power-on.
    episode_frame : jax.Array
        int32 — Frames elapsed in the current episode.
    lives : jax.Array
        int32 — Lives remaining (address is game-specific).
    terminal : jax.Array
        bool — True when the episode ended on this step.
    reward : jax.Array
        float32 — Reward earned during the last step.
    cycles : jax.Array
        int32 — Cumulative CPU cycles since power-on; used by the RIOT timer.
    """

    cpu: CPUState
    riot: RIOTState
    tia: TIAState
    bank: jax.Array
    screen: jax.Array
    frame: jax.Array
    episode_frame: jax.Array
    lives: jax.Array
    terminal: jax.Array
    reward: jax.Array
    cycles: jax.Array


def new_cpu_state() -> CPUState:
    """
    Return a zeroed CPUState.

    Returns
    -------
    state : CPUState
        All registers zero except `sp=0xFF` and `flags=0x24`
        (I=1, bit-5=1 always set) to match the ALE reset state.
        `pc` must be overwritten from the ROM reset vector before execution.
    """
    return CPUState(
        a=jnp.uint8(0),
        x=jnp.uint8(0),
        y=jnp.uint8(0),
        sp=jnp.uint8(0xFF),
        pc=jnp.uint16(0),
        flags=jnp.uint8(0x24),
    )


def new_riot_state() -> RIOTState:
    """
    Return a zeroed RIOTState.

    Returns
    -------
    state : RIOTState
        All RAM zeroed; `interval_shift=6` (divide-by-64, ALE default);
        both ports at 0xFF (no input).
    """
    return RIOTState(
        ram=jnp.zeros(128, dtype=jnp.uint8),
        timer=jnp.uint8(0),
        interval_shift=jnp.uint8(6),
        cycles_when_set=jnp.int32(0),
        port_a=jnp.uint8(0xFF),
        port_b=jnp.uint8(0xFF),
    )


def new_tia_state() -> TIAState:
    """
    Return a zeroed TIAState.

    Returns
    -------
    state : TIAState
        All registers, collision latches, and scanline buffer zeroed.
    """
    return TIAState(
        regs=jnp.zeros(64, dtype=jnp.uint8),
        collisions=jnp.uint16(0),
        scanline=jnp.zeros(160, dtype=jnp.uint8),
    )


def new_atari_state() -> AtariState:
    """
    Return a blank AtariState — structural template for `jax.vmap`.

    Returns
    -------
    state : AtariState
        Every field has a concrete dtype and shape so JAX can trace
        emulation functions without a real ROM.  Actual gameplay state is
        produced by loading ROM bytes and calling `cpu_reset()` followed
        by `emulate_frame()`.
    """
    return AtariState(
        cpu=new_cpu_state(),
        riot=new_riot_state(),
        tia=new_tia_state(),
        bank=jnp.uint8(0),
        screen=jnp.zeros((210, 160, 3), dtype=jnp.uint8),
        frame=jnp.int32(0),
        episode_frame=jnp.int32(0),
        lives=jnp.int32(0),
        terminal=jnp.bool_(False),
        reward=jnp.float32(0.0),
        cycles=jnp.int32(0),
    )
