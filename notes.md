environment criteria:

- simple
- clear states where expert gives wrong advice

plan:
use a separate head to predict whether or not we should take expert advice. should start high i.e. we always trust expert at the start.
head predicts confidence in expert decision - 0.0 to 1.0
look at what supervised learning papers do
aim: after training, can our agent summarize the cases each expert is adversarial on?

try adding contrastive learning to bc to improve detection of adversarial advice - give observation, bad_action, good_action in fixed dataset