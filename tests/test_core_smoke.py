"""Smoke tests for atari_jax.core — import, pytree shape, and one cpu_step.

Run inside the container:
    pytest tests/test_core_smoke.py -v
"""

import jax.numpy as jnp
import pytest

from atari_jax.core.state import new_atari_state, CPUState, RIOTState, TIAState, AtariState
from atari_jax.core.cpu import cpu_step, cpu_reset, OPCODE_TABLE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rom(size=4096, fill=0xEA):
    """Return a uint8 ROM array pre-filled with NOP (0xEA) by default."""
    rom = jnp.full((size,), fill, dtype=jnp.uint8)
    return rom


def _rom_with_reset(prog: bytes, rom_size=4096) -> jnp.ndarray:
    """Place prog at 0xF000 in a 4 KB ROM and set reset vector to 0xF000."""
    data = bytearray(rom_size)
    base = 0xF000 & 0x0FFF   # page offset within 4 KB bank = 0x000
    for i, b in enumerate(prog):
        data[base + i] = b
    # Reset vector at 0xFFC/0xFFD (page offset in 4 KB)
    data[0xFFC] = 0x00   # lo of 0xF000
    data[0xFFD] = 0xF0   # hi
    return jnp.array(data, dtype=jnp.uint8)


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def test_new_atari_state_shapes():
    state = new_atari_state()
    assert state.cpu.a.shape == ()
    assert state.cpu.pc.shape == ()
    assert state.riot.ram.shape == (128,)
    assert state.tia.regs.shape == (64,)


def test_opcode_table_length():
    assert len(OPCODE_TABLE) == 256


# ---------------------------------------------------------------------------
# cpu_reset loads the vector
# ---------------------------------------------------------------------------

def test_cpu_reset():
    state = new_atari_state()
    # ROM: reset vector at offset 0xFFC/0xFFD -> PC = 0x1234
    data = bytearray(4096)
    data[0xFFC] = 0x34
    data[0xFFD] = 0x12
    rom = jnp.array(data, dtype=jnp.uint8)
    state = cpu_reset(state, rom)
    assert int(state.cpu.pc) == 0x1234


# ---------------------------------------------------------------------------
# NOP (0xEA) — single step
# ---------------------------------------------------------------------------

def test_nop_advances_pc():
    state = new_atari_state()
    # Patch PC to ROM region (0x1000) so bus_read hits the cartridge
    state = state.replace(cpu=state.cpu.replace(pc=jnp.uint16(0x1000)))
    # ROM full of NOPs
    rom = _rom(4096, fill=0xEA)
    new_state, cycles = cpu_step(state, rom)
    assert int(new_state.cpu.pc) == 0x1001
    assert int(cycles) == 2


# ---------------------------------------------------------------------------
# LDA immediate (0xA9)
# ---------------------------------------------------------------------------

def test_lda_imm():
    state = new_atari_state()
    state = state.replace(cpu=state.cpu.replace(pc=jnp.uint16(0x1000)))
    data = bytearray(4096)
    data[0x000] = 0xA9   # LDA #$42
    data[0x001] = 0x42
    rom = jnp.array(data, dtype=jnp.uint8)
    new_state, cycles = cpu_step(state, rom)
    assert int(new_state.cpu.a) == 0x42
    assert int(new_state.cpu.pc) == 0x1002
    assert int(cycles) == 2
    # Z flag clear, N flag clear
    assert (int(new_state.cpu.flags) & 0x02) == 0
    assert (int(new_state.cpu.flags) & 0x80) == 0


def test_lda_imm_sets_zero_flag():
    state = new_atari_state()
    state = state.replace(cpu=state.cpu.replace(pc=jnp.uint16(0x1000)))
    data = bytearray(4096)
    data[0x000] = 0xA9   # LDA #$00
    data[0x001] = 0x00
    rom = jnp.array(data, dtype=jnp.uint8)
    new_state, _ = cpu_step(state, rom)
    assert (int(new_state.cpu.flags) & 0x02) != 0   # Z set


def test_lda_imm_sets_negative_flag():
    state = new_atari_state()
    state = state.replace(cpu=state.cpu.replace(pc=jnp.uint16(0x1000)))
    data = bytearray(4096)
    data[0x000] = 0xA9   # LDA #$80
    data[0x001] = 0x80
    rom = jnp.array(data, dtype=jnp.uint8)
    new_state, _ = cpu_step(state, rom)
    assert (int(new_state.cpu.flags) & 0x80) != 0   # N set


# ---------------------------------------------------------------------------
# BNE branch not-taken / taken
# ---------------------------------------------------------------------------

def test_bne_not_taken():
    state = new_atari_state()
    state = state.replace(cpu=state.cpu.replace(
        pc=jnp.uint16(0x1000),
        flags=jnp.uint8(0x02),   # Z set -> BNE not taken
    ))
    data = bytearray(4096)
    data[0x000] = 0xD0   # BNE +10
    data[0x001] = 0x0A
    rom = jnp.array(data, dtype=jnp.uint8)
    new_state, cycles = cpu_step(state, rom)
    assert int(new_state.cpu.pc) == 0x1002   # not taken
    assert int(cycles) == 2


def test_bne_taken():
    state = new_atari_state()
    state = state.replace(cpu=state.cpu.replace(
        pc=jnp.uint16(0x1000),
        flags=jnp.uint8(0x00),   # Z clear -> BNE taken
    ))
    data = bytearray(4096)
    data[0x000] = 0xD0   # BNE +10
    data[0x001] = 0x0A
    rom = jnp.array(data, dtype=jnp.uint8)
    new_state, _ = cpu_step(state, rom)
    assert int(new_state.cpu.pc) == 0x100C   # 0x1002 + 0x0A


# ---------------------------------------------------------------------------
# JSR / RTS round-trip
# ---------------------------------------------------------------------------

def test_jsr_rts():
    state = new_atari_state()
    state = state.replace(cpu=state.cpu.replace(pc=jnp.uint16(0x1000)))
    data = bytearray(4096)
    # At 0x1000: JSR 0x1010
    data[0x000] = 0x20; data[0x001] = 0x10; data[0x002] = 0x10
    # At 0x1010: RTS
    data[0x010] = 0x60
    rom = jnp.array(data, dtype=jnp.uint8)

    state_after_jsr, _ = cpu_step(state, rom)
    assert int(state_after_jsr.cpu.pc) == 0x1010

    state_after_rts, _ = cpu_step(state_after_jsr, rom)
    assert int(state_after_rts.cpu.pc) == 0x1003   # return address +1


# ---------------------------------------------------------------------------
# STA zero-page writes to RIOT RAM
# ---------------------------------------------------------------------------

def test_sta_zp_writes_riot_ram():
    state = new_atari_state()
    state = state.replace(cpu=state.cpu.replace(
        pc=jnp.uint16(0x1000),
        a=jnp.uint8(0xBE),
    ))
    data = bytearray(4096)
    data[0x000] = 0x85   # STA $90  (zero-page -> RIOT RAM at index 0x10)
    data[0x001] = 0x90
    rom = jnp.array(data, dtype=jnp.uint8)
    new_state, cycles = cpu_step(state, rom)
    assert int(new_state.riot.ram[0x10]) == 0xBE
    assert int(cycles) == 3
