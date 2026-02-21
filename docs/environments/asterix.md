# Asterix

> ALE name: `asterix` · Game ID: `3`

Guide Asterix horizontally across the screen collecting magic potions while avoiding enemies and their projectiles in this endless side-scroller.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned by collecting magic potion objects that scroll across the screen. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends on the final death animation frame, detected when the death counter reaches 1 while the player has exactly 1 life remaining.

## Lives

The player starts with several lives stored in the low nibble of a RAM byte. A life is lost by touching an enemy or their projectile.
