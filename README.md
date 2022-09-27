# Implementing EvoJax for Physics-Informed Neural Networks

EvoJax is a neuroevolution toolkit build on top of the JAX library. This enables the neuroevolution algoritms to run faster as it runs the neural networks in parallel across multiple TPU/GPUs. On some popular tasks, EvoJAX demonstrates 10-20x training speedup.

In this repo, we explore the implementation of EvoJax for Physics-Informed Neural Networks (PINNs). To begin, EvoJAX framework has the following three major components:
1. **Neuroevolution Algorithms** All neuroevolution algorithms should implement the `evojax.algo.base.NEAlgorithm` interface and reside in `evojax/algo/`.
See [here](https://github.com/google/evojax/blob/main/evojax/algo/README.md) for the available algorithms in EvoJAX.
2. **Policy Networks** All neural networks should implement the `evojax.policy.base.PolicyNetwork` interface and be saved in `evojax/policy/`.
Some example implementations of the MLP, ConvNet, Seq2Seq and [PermutationInvariant](https://attentionneuron.github.io/) models.
3. **Tasks** All tasks should implement `evojax.task.base.VectorizedTask` and be in `evojax/task/`.


