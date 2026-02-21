# Private Eye

> ALE name: `private_eye` · Game ID: `38`

Drive through a city as private detective Henri Le Fiend, gathering clues and chasing criminals across multiple screens to solve a case.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned by collecting clues and apprehending criminals. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when a game-state RAM byte transitions away from the two valid in-play values (`0x00` for running and `0x01` for start-of-game), indicating the game has ended.

## Lives

No lives system. Private Eye ends based on a game-state flag rather than life depletion.
