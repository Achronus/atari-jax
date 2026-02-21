# Wizard of Wor

> ALE name: `wizard_of_wor` · Game ID: `54`

Fight through dungeon mazes shooting Worlings, Garwors, Thorwors, and the deadly Wizard of Wor himself in an arcade maze shooter.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for shooting each type of monster, with rarer creatures awarding more points. Score values are multiples of 100 using a BCD encoding with a special subtraction correction for values above 8000.

## Episode End

The episode ends when all lives are exhausted and the game-over screen is shown, detected when the lives nibble reaches zero and a terminal screen byte equals `0xF8`.

## Lives

Lives are stored in the lower nibble of a RAM byte. A life is lost each time the player's warrior is shot by a monster.
