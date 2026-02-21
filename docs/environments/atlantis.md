# Atlantis

> ALE name: `atlantis` · Game ID: `5`

Defend the underwater city of Atlantis from waves of descending alien ships using three gun emplacements positioned around the city.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for each alien ship destroyed. Score values are multiples of 100. On the terminal frame the reward is forced to zero because the ROM writes invalid data to the score RAM at game over.

## Episode End

The episode ends when the game-over sentinel value (`0xFF`) is written to the lives RAM byte, indicating all city sections have been destroyed.

## Lives

Atlantis uses city sections rather than traditional lives. The episode ends when all sections are destroyed, signalled by a sentinel byte value rather than a zero count.
