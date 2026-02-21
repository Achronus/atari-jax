# Road Runner

> ALE name: `road_runner` · Game ID: `41`

Race ahead of Wile E. Coyote along a winding desert road, collecting birdseed and dodging boulders, trucks, and other obstacles.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for collecting birdseed piles and for putting distance between Road Runner and the Coyote. Score values are multiples of 100 using a nibble-encoded format.

## Episode End

The episode ends when all lives are exhausted and the death animation is playing, detected when lives reach zero and a velocity byte is non-zero.

## Lives

Lives are stored in the lower 3 bits of a RAM byte (displayed lives = bits + 1). A life is lost each time Wile E. Coyote catches Road Runner or a hazard kills him.
