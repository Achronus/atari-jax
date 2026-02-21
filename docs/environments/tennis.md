# Tennis

> ALE name: `tennis` · Game ID: `48`

Play a singles tennis match against a CPU opponent, rallying to win games and sets using standard tennis scoring rules.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward reflects changes in point differential first, then game differential if no points changed. Winning a point yields a positive reward; losing a point yields a negative reward. This matches the ALE behaviour of prioritising point-level signals.

## Episode End

The episode ends when either player wins the set: a player must have at least 6 games with a 2-game lead, or reach 7 games in a tiebreak.

## Lives

No lives system. Tennis is a set-based match with no lives concept.
