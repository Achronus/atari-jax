# Battle Zone

> ALE name: `battle_zone` · Game ID: `7`

Destroy enemy tanks and flying saucers in a first-person 3D tank combat arena with an overhead radar to track approaching enemies.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned in multiples of 1,000 for destroying enemy tanks and saucers; different enemy types award different amounts. Score uses a custom nibble encoding where a nibble value of 10 is treated as zero.

## Episode End

The episode ends when the player's tank is destroyed and all lives are lost. Lives are stored in the low nibble of a RAM byte, and the episode terminates when this drops from a positive value to zero.

## Lives

The player starts with several lives stored in the low nibble of a RAM byte. A life is lost each time the player's tank is hit and destroyed.
