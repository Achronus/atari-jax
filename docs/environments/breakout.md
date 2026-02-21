# Breakout

> ALE name: `breakout` · Game ID: `12`

Use a paddle to bounce a ball upward and destroy all the coloured bricks arranged in rows at the top of the screen.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for each brick destroyed; bricks in different rows award different point values. Score is stored as packed BCD across two RAM bytes.

## Episode End

The episode ends when the player loses all five lives, detected when the lives count transitions from a positive value to zero.

## Lives

The player starts with 5 lives. A life is lost each time the ball passes the paddle and falls off the bottom of the screen.
