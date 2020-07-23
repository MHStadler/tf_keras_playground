import os
import pandas as pd
import sys

from tf_keras_playground.train import train_cifar10_resnet

if __name__ == '__main__':
    model_name = sys.argv[1]
    stoch_depth = sys.argv[2]
    
    model, history = train_cifar10_resnet(model_name, stoch_depth = stoch_depth)

    model.save(f'../trained_models/{model_name}.h5')
    pd.DataFrame(history.history).to_csv(f'../trained_models/{model_name}_hist.csv')