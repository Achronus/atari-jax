# Star Gunner

> ALE name: `star_gunner` · Game ID: `47`

Pilot a star fighter through waves of alien ships, destroying enemies for points while dodging their projectiles in a side-scrolling space shooter.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for destroying enemy ships; scores are multiples of 100 using a nibble-encoded format with a sentinel value of `0xA` representing zero.

## Episode End

The episode ends when the lives byte reaches zero.

## Lives

Lives are stored as a 0-based count in the lower nibble of a RAM byte. A life is lost each time the player's ship is hit by enemy fire.
