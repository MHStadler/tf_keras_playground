# tf_keras_playground

A small repository to consilidate, and test my implementation of various deep learning utilities for tensorflow/keras

## Current Tools
 - Stochastic Depth, a technique to train very deep ResNets [1]. Early, minimal viable implementation available as keras wrapper (layers.StochDep) or keras layer (layers.ResDrop)
 - WarmUp LearningRate Schedule [2] - a learning rate schedule that linearly scales towards a goal learning rate for a certain number of warmup steps, before delegating the rest of the schedule to a follow_up tf.keras.optimizers.LearningRateSchedule

## References
[1] Huang, Gao, et al. "Deep networks with stochastic depth." European conference on computer vision. Springer, Cham, 2016.

[2] Goyal, Priya, et al. "Accurate, large minibatch sgd: Training imagenet in 1 hour." arXiv preprint arXiv:1706.02677 (2017).
