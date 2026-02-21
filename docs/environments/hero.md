# H.E.R.O.

> ALE name: `hero` · Game ID: `25`

Fly through underground mine shafts using a jetpack and shoulder-mounted laser to rescue miners trapped in deep caverns.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for rescuing miners, destroying obstacles with dynamite, and shooting creatures. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the lives counter reaches zero, detected directly when the lives RAM byte equals zero.

## Lives

The player starts with several lives stored directly in a RAM byte. A life is lost by touching an enemy creature, falling into lava, or running out of power for the jetpack.
