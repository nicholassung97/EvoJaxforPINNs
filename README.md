# Implementing EvoJax for Physics-Informed Neural Networks

EvoJax is a neuroevolution toolkit build on top of the JAX library. This enables the neuroevolution algoritms to run faster as it runs the neural networks in parallel across multiple TPU/GPUs. On some popular tasks, EvoJAX demonstrates 10-20x training speedup.

In this repo, we explore the implementation of EvoJax for Physics-Informed Neural Networks (PINNs). To begin, EvoJAX framework has the following three major components:
1. **Neuroevolution Algorithms** All neuroevolution algorithms should implement the `evojax.algo.base.NEAlgorithm` interface and reside in `evojax/algo/`.
The evolutionary algorithms can be found [here](https://github.com/google/evojax/blob/main/evojax/algo/README.md). These are algorithms that have already been developed by the EvoJax team or external parties

2. **Policy Networks** All neural networks should implement the `evojax.policy.base.PolicyNetwork` interface and be saved in `evojax/policy/`.
Some example implementations of the policy are :MLP, ConvNet, Seq2Seq and [PermutationInvariant](https://attentionneuron.github.io/) models. Essentially, the policy refers to the configurations of the neural network.

3. **Tasks** All tasks should implement `evojax.task.base.VectorizedTask` and be in `evojax/task/`. The task refers to everything outside of the neural network (eg. inputs, outputs, loss formulation, etc)

In this implementaiton, we benchmarked the performance of different solvers (eg. CMA-ES JAX based, CMA-ES Wrapper, PGPE, ARS, Open ES, Simple GA) in the context of PINNs.

Attached is the implementation, enjoy!!
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://github.com/nicholassung97/EvoJaxforPINNs/blob/main/Adapting_to_PINNs_final.ipynb)

The main struggle with the implementation is that everything is in a blackbox/wrapper. Hence, lots of unwrapping was done in the developement of the code. Moving forward, we will explore the implementation of PINNs on different EA libraries to compare their efficiecy. Also, the possibility of developing the t-NES solver will be explored.

