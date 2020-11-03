Abstract:
Imitation learning allows agents to learn complex behaviors from demonstrations.
However learning a complex vision-based task may require an impractical number
of demonstrations. Meta-imitation learning is a promising approach towards
enabling agents to learn a new task from one or a few demonstrations by
leveraging previous data. In the presence of task ambiguity or unobserved
dynamics, demonstrations alone may not provide enough information;
an agent must also try the task to successfully infer a policy. In this work,
we propose a method that can learn to learn from both demonstrations and
trial-and-error experience with sparse reward feedback. In comparison to
meta-imitation, this approach enables the agent to effectively and efficiently
improve itself autonomously beyond demonstration data. In comparison to
meta-reinforcement learning, we can scale to substantially broader distributions
of tasks, as the demonstration reduces the burden of exploration. Our
experiments show that our method significantly outperforms prior approaches on
a set of challenging, vision-based control tasks.

Run an experiment using your chosen gin config:

```
# Train WTL trial policy.
python -m train.py --gin_config configs/train_ctx_trial.gin
# Collect dataset from trial policy.
python -m run_run_meta_env.py --gin_config configs/collect_trained_trial.gin
# Train WTL retrial policy.
python -m run_run_meta_env.py --gin_config configs/train_ctx_retrial.gin
```
