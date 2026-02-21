# Freeway

> ALE name: `freeway` · Game ID: `21`

Guide a chicken across a busy multi-lane highway, dodging cars in both directions within a timed session to score as many crossings as possible.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is clamped to exactly 0 or 1 per step: the player earns +1 for each successful crossing. The maximum possible score is 30 crossings per session.

## Episode End

The episode ends when the timed session expires, detected when a terminal flag byte is set to 1.

## Lives

No lives system. Being hit by a car pushes the chicken back rather than costing a life.
