# Amidar

> ALE name: `amidar` · Game ID: `1`

Paint every cell of a grid by walking along its paths while enemies patrol and chase you; bonus points are awarded for completing boxes.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for each grid segment painted and for completing enclosed rectangular boxes. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the game-over sentinel value is written to the lives RAM byte (a special value of `0x80` rather than a simple zero count). All lives must be exhausted to trigger this condition.

## Lives

The player starts with several lives, stored in the low nibble of a RAM byte. A life is lost when an enemy catches the player. The episode ends when all lives are gone.
