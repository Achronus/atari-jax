# Kung-Fu Master

> ALE name: `kung_fu_master` · Game ID: `30`

Battle through five floors of enemies using kicks, punches, and jumps to rescue a hostage held by the evil Mr. X.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for defeating each type of enemy. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the game-over sentinel value (`0xFF`) is written to the lives RAM byte, representing an underflow past zero lives.

## Lives

Lives are stored in the lower 3 bits of a RAM byte (displayed lives = bits + 1). The sentinel `0xFF` signals game over. A life is lost each time the player's health is depleted by enemy attacks.
