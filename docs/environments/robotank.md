# Robotank

> ALE name: `robotank` · Game ID: `42`

Command a lone tank to seek out and destroy enemy robot tank squadrons using radar and your cannon in a first-person arena.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for destroying individual enemy tanks (1 point each) and completing full squadrons (12 points per squadron). Score equals destroyed squadrons times 12 plus individual tanks destroyed.

## Episode End

The episode ends when all lives are exhausted and the terminal flag is set, detected when the lives byte reaches zero and a terminal RAM byte equals `0xFF`.

## Lives

Lives are stored in the lower nibble of a RAM byte (displayed lives = nibble + 1). A life is lost each time the player's tank is destroyed by enemy fire.
