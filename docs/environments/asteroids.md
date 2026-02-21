# Asteroids

> ALE name: `asteroids` · Game ID: `4`

Pilot a spaceship in open space, destroying asteroids that split into smaller pieces and shooting flying saucers while managing your inertia.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for destroying asteroids (larger ones score less than smaller ones) and for shooting flying saucers. The score wraps at 100,000; negative deltas are corrected by adding 100,000 to recover the true reward.

## Episode End

The episode ends when the player loses their last life. A life is lost by colliding with an asteroid or being hit by a flying saucer's projectile.

## Lives

Lives are stored in the upper nibble of a RAM byte. The episode terminates when the lives count drops from a positive value to zero.
