# Fidelity Testing

Atarax games are validated against the reference [ALE](https://github.com/Farama-Foundation/Arcade-Learning-Environment) C++ engine using a
statistical methodology based on random-policy episode returns. This document
describes the approach, the calibration setup, and known structural deviations
from ALE.

## What we test

Each game must pass three levels of validation before it is considered complete:

**Smoke tests** — every registered game must `reset`, `step`, and `render`
correctly, compile with `jax.jit`, batch with `jax.vmap`, and produce a valid
JAX pytree state.

**ALE fidelity calibration** — the game's random-policy mean return must fall
inside a pre-measured calibration band. See [Fidelity calibration](#fidelity-calibration)
below.

**Per-game mechanics tests** — game-specific unit tests covering collision
detection, scoring, terminal conditions, lives tracking, and boundary physics,
written against the ALE C++ source as ground truth.

## Observations and the HUD

Atarax observations are `uint8[210, 160, 3]` RGB frames — the same dimensions as
the ALE standard. The **HUD is part of the observation**, consistent with ALE.

In ALE the full `210×160` pixel buffer is returned as the observation regardless of
what is rendered there — score digits, life counters, and game-specific overlays
are always visible to the agent. Atarax matches this by rendering the HUD (score
and lives) into the top ~30 pixels of the same canvas.

Excluding the HUD from the observation would diverge from the ALE standard and
require agents to use a different observation crop. The HUD is also a useful
signal — a CNN policy can learn to read lives and score from pixels, exactly as
in the ALE setting.

## Fidelity calibration

### Methodology

The random-policy episode return is a reliable calibration target for aligning
JAX-native implementations against ALE: it is reproducible, requires no trained
policy, and is sensitive to broken scoring, missing terminals, and wrong episode
dynamics.

**Protocol:**

- N = 1,000 parallel environments via `jax.vmap`
- SEED = 42
- 1,000 agent steps per episode (4,000 emulated frames at 4× frame-skip)
- Single `jax.lax.scan` pass — no Python loop

**Band formula:** `mean ± 3·SE` where `SE = std / √1000`, giving ≈ 99.7%
coverage under the Central Limit Theorem.

### What the bands catch

- Wrong scoring multipliers
- Missing or broken terminal conditions
- Physics that ends or extends episodes at the wrong rate

### What the bands do not catch

- Cancelling errors (e.g. double score + double kill rate yielding the same
  expected return)
- Differences that only manifest under a non-random policy

Per-game mechanics tests cover the most critical structural invariants beyond
what the fidelity bands detect.

### Target

We target a ≤5% deviation from the ALE random-policy baseline (JAX mean / ALE
mean within [0.95×, 1.05×]), with 1× being optimal. Exact replication is not the
goal — Atarax uses JAX's XLA-based PRNG and branch-free collision detection, so
some deviation is inherent by design. Ratios outside the target window are
tracked as known deviations (see below) or as open bugs to resolve.

## On visual differences from ALE

Atarax renders look deliberately different from ALE ROM output: **procedural
solid-colour blocks on flat backgrounds** rather than sprite art with CRT scan
lines. This is the design intent — see the
[Design Philosophy](../README.md#design-philosophy) for the full rationale.

The fidelity bands are the primary statistical evidence that the RL challenge is
preserved despite the visual difference. Matching return distributions under a
random policy is a strong signal that reward magnitudes, terminal conditions, and
episode dynamics are consistent with ALE.

**Visual pixel fidelity to the ROM is explicitly not a goal.**

## Known structural deviations

These are intentional or inherent differences from ALE that are accepted and
documented rather than treated as bugs.

| Game | Deviation | Accepted? |
| --- | --- | --- |
| **Breakout** | Fixed π/4 serve angle instead of ROM-randomised angle — consistent brick coverage vs inconsistent ALE | Yes — fidelity band acts as regression guard |

## Fidelity table

Calibration setup: N=1,000 environments (JAX vmap), SEED=42, 1,000 agent steps
(4,000 emulated frames), single vmap pass. Band = `mean ± 3·SE`.

Target ratio: **0.95×–1.05×** (JAX mean / ALE baseline). Deviations outside this
window are noted in the table.

| Game | ALE Baseline | JAX Mean | JAX Std | Fidelity Band | Ratio |
| --- | --- | --- | --- | --- | --- |
| Asteroids | 754.5 | 2788.1 | 1523.3 | [2643.6, 2932.6] | 3.70× ⚠ |
| Breakout | 1.1 | 9.4 | 8.0 | [8.6, 10.1] | 8.84× † |
| Ms. Pac-Man | 257.0 | 25.5 | 11.7 | [24.3, 26.6] | 0.10× ⚠ |
| Space Invaders | 154.3 | 358.0 | 90.3 | [349.4, 366.6] | 2.32× ⚠ |

† **Breakout (8.84×):** Fixed π/4 serve angle instead of ROM-randomised angle,
so the random policy achieves consistent brick coverage that the ALE random
policy does not. The fidelity band `[8.6, 10.1]` acts as a regression guard for
broken physics, not an ALE mirror. This is an accepted structural deviation.

⚠ **Asteroids, Ms. Pac-Man, Space Invaders** have ratios outside the 0.95×–1.05×
target. These are open fidelity bugs — physics or scoring needs investigation to
close the gap.
