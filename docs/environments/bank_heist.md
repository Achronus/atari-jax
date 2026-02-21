# Bank Heist

> ALE name: `bank_heist` · Game ID: `6`

Drive a getaway car through a city, robbing banks by parking outside them while dropping dynamite to destroy pursuing police cars.

## Spaces

| | Value |
| --- | --- |
| **Observation** | `Box(uint8, shape=(210, 160, 3))` |
| **Actions** | `Discrete(18)` |

## Reward

The reward is the increase in score on each step. Points are earned for each bank robbed and for destroying police cars with dynamite. Score is stored as packed BCD across three RAM bytes.

## Episode End

The episode ends when the death timer reaches 1 and the lives count has reached zero. A life is lost when a police car catches the player or when the player runs out of fuel.

## Lives

The player starts with several lives stored directly in a RAM byte. A life is lost through police capture or fuel exhaustion. The episode terminates when all lives are gone.
