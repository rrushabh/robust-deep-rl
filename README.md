# robust-deep-rl

## Potential issues

- The Tanh() activation in the actor.
- TruncatedNormal limits.
- The optimizers need to be reinit? or init??
- Remove all stochasticity from the environment? Atleast remove it from the actor while BC training.
- why is mu all 1s?
- weight_init can be removed.