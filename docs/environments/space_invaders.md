# Space Invaders

> ALE name: `space_invaders` · Game ID: `46`

Defend Earth from descending waves of alien invaders, shooting them before they reach the ground while using bunkers for cover.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for each alien destroyed; aliens in higher rows are worth more points. The score wraps at 10,000; negative deltas are corrected to handle the wrap-around.

## Episode End

The episode ends when either the game-over bit flag is set or the lives count reaches zero.

## Lives

The player starts with several lives stored directly in a RAM byte. A life is lost each time the player's cannon is hit by an alien projectile.
