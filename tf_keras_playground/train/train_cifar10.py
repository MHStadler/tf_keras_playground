import os
import numpy as np
import pandas as pd
import sys
import tensorflow as tf

from tf_keras_playground.dataset import Cifar10Dataset
from tf_keras_playground.models import ResNet

def train_cifar10_resnet(model_name, stoch_depth = '0', blocks = 5):
    epochs = 180
    batch_size = 128

    meta_data, train_dataset, val_dataset = Cifar10Dataset()(batch_size = batch_size)
    
    n_steps = np.ceil(meta_data['N_train'] / batch_size)
    n_val_steps = np.ceil(meta_data['N_test'] / batch_size)

    model = ResNet()((32, 32, 3), model_name, blocks = [blocks, blocks, blocks], stoch_depth = stoch_depth)
    
    optimizer = tf.keras.optimizers.SGD(learning_rate = 1e-1, momentum = 0.9, nesterov = True)
    model.compile(loss = 'sparse_categorical_crossentropy', metrics = ['sparse_categorical_accuracy'], optimizer = optimizer)
    
    model.summary()
    
    def scheduler(epoch):
        if epoch < 90:
            return 0.1
        elif epoch < 135:
            return 0.01
        else:
            return 0.001

    callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = model.fit(
        train_dataset, epochs = epochs, steps_per_epoch = n_steps, verbose = 2,
            validation_data = val_dataset, validation_steps = n_val_steps, callbacks = [callback])
    
    return model, history