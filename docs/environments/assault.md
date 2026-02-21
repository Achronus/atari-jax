# Assault

> ALE name: `assault` · Game ID: `2`

Pilot a ground-based turret and shoot down waves of alien spaceships that attack from above in increasingly difficult formations.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned by destroying enemy ships. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the player loses their last life. Lives decrease when the player's turret is destroyed by enemy fire or collision.

## Lives

The player starts with a number of lives stored directly in a RAM byte. The episode terminates when the lives count drops from a positive value to zero.
