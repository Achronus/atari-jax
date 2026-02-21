# Ice Hockey

> ALE name: `ice_hockey` · Game ID: `26`

Play a one-on-one ice hockey match against a CPU opponent, scoring goals by shooting the puck past the opposing goalie.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the change in score differential (player goals minus opponent goals) per step, clamped to a maximum of +1 per step. Scoring a goal yields +1; conceding a goal yields -1.

## Episode End

The episode ends when the period clock runs out, detected when both the minutes and seconds bytes of the timer reach zero.

## Lives

No lives system. Ice Hockey is a timed period match with no lives concept.
