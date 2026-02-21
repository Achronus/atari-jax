# Name This Game

> ALE name: `name_this_game` · Game ID: `33`

Protect an underwater treasure chest from a series of sea creatures as a scuba diver, shooting at enemies before they can steal it.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for shooting sea creatures before they reach the treasure chest. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when all lives are exhausted, detected when the 3-bit lives field in RAM reaches zero.

## Lives

Lives are stored as a 0-based count in the lower 3 bits of a RAM byte. A life is lost when a sea creature successfully steals from the chest or the player is caught.
