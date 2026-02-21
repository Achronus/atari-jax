# Up 'n Down

> ALE name: `up_n_down` · Game ID: `51`

Drive a car along a hilly, scrolling road, jumping over or crashing into other vehicles to collect flags and score points.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for destroying other vehicles by jumping on them and for collecting flags along the road. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the death timer exceeds `0x40` and the lives nibble has reached zero, indicating the final death sequence is active.

## Lives

Lives are stored in the lower nibble of a RAM byte (displayed lives = nibble + 1). A life is lost each time the player's car collides with another vehicle without jumping on it.
