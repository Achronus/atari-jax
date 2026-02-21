# Kangaroo

> ALE name: `kangaroo` · Game ID: `28`

Guide a mother kangaroo up a series of platforms to rescue her joey, punching monkeys and avoiding thrown apples and bells.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for punching monkeys, collecting fruit, and rescuing the joey. Score values are multiples of 100. Score is stored as packed BCD across two RAM bytes.

## Episode End

The episode ends when the game-over sentinel value (`0xFF`) is written to the lives RAM byte, representing an underflow past zero lives.

## Lives

Lives are stored in the lower 3 bits of a RAM byte (displayed lives = bits + 1). The sentinel `0xFF` signals game over. A life is lost each time the kangaroo is hit by a thrown object or catches a claw.
