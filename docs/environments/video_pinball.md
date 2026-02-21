# Video Pinball

> ALE name: `video_pinball` · Game ID: `53`

Play a digital pinball table featuring bumpers, flippers, and Atari-themed targets, keeping the ball in play to accumulate a high score.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned by striking bumpers, targets, and completing multi-ball features. Score is stored as packed BCD across three non-sequential RAM bytes.

## Episode End

The episode ends when a game-over bit flag is set in a terminal RAM byte.

## Lives

The player starts with 3 balls (lives), calculated as `4 + extra_ball_flag - ball_number`. An extra ball may be earned through gameplay. A ball is lost each time it drains past the flippers.
