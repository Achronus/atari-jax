# Beam Rider

> ALE name: `beam_rider` · Game ID: `8`

Ride a segmented beam toward a distant point in space, shooting enemy ships that approach along the beam while dodging obstacles.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned by shooting enemy ships and completing beam segments. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when a game-over sentinel value (`0xFF`) is written to a RAM address, indicating all lives have been exhausted.

## Lives

Lives are stored as a zero-based count; the displayed life count is the RAM value plus one. A life is lost by colliding with an enemy or being hit by a projectile.
