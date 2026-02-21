# Chopper Command

> ALE name: `chopper_command` · Game ID: `14`

Pilot a military helicopter gunship to protect a convoy of ground trucks from waves of enemy aircraft and helicopters.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for shooting down enemy aircraft; scores are multiples of 100. Score is stored as packed BCD across two RAM bytes.

## Episode End

The episode ends when the game has started and all lives have been depleted, detected when the started-flag is set and the lives count reaches zero.

## Lives

Lives are stored in the low nibble of a RAM byte. A life is lost each time the player's helicopter is shot down.
