# Berzerk

> ALE name: `berzerk` · Game ID: `9`

Navigate a maze of rooms and destroy robots using your laser pistol while avoiding their fire and the indestructible, bouncing Evil Otto.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for each robot destroyed. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the game-over sentinel value (`0xFF`) is written to the lives RAM byte. This underflow sentinel occurs when the zero-based life count decrements below zero.

## Lives

Lives are stored as a zero-based count (actual lives = RAM value + 1). The sentinel value `0xFF` represents an underflow past zero, signalling game over. A life is lost by being shot by a robot, touching a wall, or being caught by Evil Otto.
