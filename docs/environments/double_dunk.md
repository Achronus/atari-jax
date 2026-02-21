# Double Dunk

> ALE name: `double_dunk` · Game ID: `18`

Play a two-on-two street basketball game, selecting plays and shooting baskets to outscore your opponents to 24 points.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the change in score differential (player team score minus opponent score) per step. Each basket scores the appropriate number of points (2 or 3); the reward reflects the net change in the score gap.

## Episode End

The episode ends when either team reaches 24 points and a game-over sync byte is set to a specific sentinel value.

## Lives

No lives system. Double Dunk is a scored basketball match with no lives concept.
