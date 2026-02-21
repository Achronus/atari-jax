# Demon Attack

> ALE name: `demon_attack` · Game ID: `17`

Destroy waves of increasingly complex demons that descend from the sky, using your cannon on an ice planet surface.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for each demon destroyed; later waves award more points. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the lives display reaches zero and the game-over screen is shown (detected by a specific sentinel value in the display RAM byte).

## Lives

Lives are stored as a zero-based count (displayed lives = RAM value + 1). A life is lost each time a demon's projectile hits the player's cannon.
