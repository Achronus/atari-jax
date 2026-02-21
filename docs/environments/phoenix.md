# Phoenix

> ALE name: `phoenix` · Game ID: `34`

Shoot waves of alien birds that swoop down in increasingly aggressive patterns, then destroy the giant alien mothership to complete the cycle.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for each bird destroyed, with the mothership awarding the highest bonus. Score uses a multi-byte BCD encoding that is shifted and recombined from three RAM bytes.

## Episode End

The episode ends when the game-over screen is shown, detected when a terminal RAM byte equals `0x80`.

## Lives

Lives are stored as a 0-based count in the lower 3 bits of a RAM byte. A life is lost each time the player's cannon is destroyed by an enemy ship or projectile.
