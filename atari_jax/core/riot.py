"""M6532 RIOT — 128-byte RAM, programmable interval timer, and I/O ports.

Address decoding (from M6532.cxx install loop):
  RAM:    (addr & 0x1080) == 0x0080  AND  (addr & 0x0200) == 0
  I/O:    (addr & 0x1080) == 0x0080  AND  (addr & 0x0200) != 0

Timer modes — dispatch on (addr & 0x17):
  0x14 → interval_shift=0  (divide by 1)
  0x15 → interval_shift=3  (divide by 8)
  0x16 → interval_shift=6  (divide by 64)   ← ALE default at reset
  0x17 → interval_shift=10 (divide by 1024)

Timer current value (from M6532.cxx peek logic):
  delta      = cycles - cycles_when_set
  timer_val  = timer - (delta >> interval_shift) - 1
  if timer_val < 0: counts down at rate 1 per cycle after underflow
"""

import jax
import jax.numpy as jnp

from atari_jax.core.state import AtariState


def riot_ram_read(state: AtariState, addr13: jax.Array) -> jax.Array:
    """
    Read one byte from RIOT RAM.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    addr13 : jax.Array
        int32 — 13-bit bus address; RAM index = `addr13 & 0x7F`.

    Returns
    -------
    byte : jax.Array
        uint8 — Byte at the 7-bit RAM index.
    """
    ram_idx = addr13 & jnp.int32(0x7F)
    return state.riot.ram[ram_idx]


def riot_ram_write(
    state: AtariState, addr13: jax.Array, value: jax.Array
) -> AtariState:
    """
    Write one byte to RIOT RAM.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    addr13 : jax.Array
        int32 — 13-bit bus address; RAM index = `addr13 & 0x7F`.
    value : jax.Array
        uint8 — Byte to write.

    Returns
    -------
    state : AtariState
        Updated state with the new RAM byte.
    """
    ram_idx = addr13 & jnp.int32(0x7F)
    new_ram = state.riot.ram.at[ram_idx].set(value.astype(jnp.uint8))
    return state.__replace__(riot=state.riot.__replace__(ram=new_ram))


def _timer_current(state: AtariState) -> jax.Array:
    """
    Compute the live timer value from elapsed cycles.

    Mirrors ALE M6532::peek case 0x04 logic.

    Parameters
    ----------
    state : AtariState
        Current machine state.

    Returns
    -------
    value : jax.Array
        uint8 — Current timer countdown value.
    """
    delta = (state.cycles - state.riot.cycles_when_set).astype(jnp.int32)
    shift = state.riot.interval_shift.astype(jnp.int32)
    timer_base = state.riot.timer.astype(jnp.int32)

    # Normal countdown: decrements once per (1 << shift) cycles.
    normal = timer_base - (delta >> shift) - jnp.int32(1)

    # After underflow: counts down 1 per cycle from the overflow point.
    overflow_point = timer_base << shift
    after_overflow = overflow_point - delta - jnp.int32(1)

    result = jax.lax.cond(
        normal >= jnp.int32(0),
        lambda: normal,
        lambda: after_overflow,
    )
    return jnp.astype(result, jnp.uint8)


def _timer_irq_flag(state: AtariState) -> jax.Array:
    """
    Return the timer interrupt flag byte.

    Parameters
    ----------
    state : AtariState
        Current machine state.

    Returns
    -------
    flag : jax.Array
        uint8 — 0x80 if the timer has underflowed (interrupt pending),
        0x00 otherwise.
    """
    delta = (state.cycles - state.riot.cycles_when_set).astype(jnp.int32)
    shift = state.riot.interval_shift.astype(jnp.int32)
    normal = state.riot.timer.astype(jnp.int32) - (delta >> shift) - jnp.int32(1)
    return jnp.where(normal < jnp.int32(0), jnp.uint8(0x80), jnp.uint8(0x00))


def riot_io_read(state: AtariState, addr13: jax.Array) -> jax.Array:
    """
    Read a RIOT I/O / timer register.

    Register map (addr & 0x07):
      0x00  SWCHA  — Port A (joystick)
      0x01  SWACNT — DDRA stub (0x00)
      0x02  SWCHB  — Port B (console switches)
      0x03  SWBCNT — DDRB stub (0x00)
      0x04  INTIM  — Timer output
      0x05  INSTAT — Interrupt flag
      0x06  —      — Timer output (mirror)
      0x07  —      — Interrupt flag (mirror)

    Parameters
    ----------
    state : AtariState
        Current machine state.
    addr13 : jax.Array
        int32 — 13-bit bus address; register = `addr13 & 0x07`.

    Returns
    -------
    byte : jax.Array
        uint8 — Register value.
    """
    reg = (addr13 & jnp.int32(0x07)).astype(jnp.int32)
    timer_val = _timer_current(state)
    irq_flag = _timer_irq_flag(state)

    return jax.lax.switch(
        reg,
        [
            lambda _: state.riot.port_a,  # 0x00
            lambda _: jnp.uint8(0x00),  # 0x01  DDRA stub
            lambda _: state.riot.port_b,  # 0x02
            lambda _: jnp.uint8(0x00),  # 0x03  DDRB stub
            lambda _: timer_val,  # 0x04
            lambda _: irq_flag,  # 0x05
            lambda _: timer_val,  # 0x06
            lambda _: irq_flag,  # 0x07
        ],
        jnp.int32(0),
    )


# Interval shifts for the 4 timer write modes (addr & 0x03 → mode index).
_INTERVAL_SHIFTS = jnp.array([0, 3, 6, 10], dtype=jnp.uint8)


def riot_io_write(state: AtariState, addr13: jax.Array, value: jax.Array) -> AtariState:
    """
    Write to a RIOT I/O / timer register.

    Timer write condition: (addr & 0x14) == 0x14 (bits 4 and 2 both set).
    Mode (÷1/÷8/÷64/÷1024) is selected by addr & 0x03.

    Parameters
    ----------
    state : AtariState
        Current machine state.
    addr13 : jax.Array
        int32 — 13-bit bus address of the write.
    value : jax.Array
        uint8 — Timer initial value (used only on timer writes).

    Returns
    -------
    state : AtariState
        Updated state with new timer fields (or unchanged for DDR writes).
    """
    is_timer_write = (addr13 & jnp.int32(0x14)) == jnp.int32(0x14)
    mode = (addr13 & jnp.int32(0x03)).astype(jnp.int32)
    new_shift = _INTERVAL_SHIFTS[mode]

    new_riot = jax.lax.cond(
        is_timer_write,
        lambda riot: riot.__replace__(
            timer=value.astype(jnp.uint8),
            interval_shift=new_shift,
            cycles_when_set=state.cycles,
        ),
        lambda riot: riot,  # Port A/B DDR writes — stub (games rarely use them)
        state.riot,
    )
    return state.__replace__(riot=new_riot)
