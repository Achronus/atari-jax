# River Raid

> ALE name: `riverraid` · Game ID: `40`

Pilot a jet plane over a scrolling river, shooting ships, helicopters, and fuel depots while landing on fuel pads to refuel.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for destroying ships, helicopters, bridges, and balloons, with different amounts for each target type. Score is encoded using a multiply-by-8 scheme across six RAM bytes.

## Episode End

The episode ends on the transition from 1 life to 0 lives, detected when the lives byte resets to its 4-life starting value after having been at the 1-life value.

## Lives

The player starts with 4 lives, tracked via a special encoding in a single RAM byte. A life is lost by colliding with terrain, an enemy, or running out of fuel.
