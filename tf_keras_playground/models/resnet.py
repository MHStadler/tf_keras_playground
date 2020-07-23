import numpy as np

from tensorflow import keras

from tf_keras_playground.layers import ResDrop, StochDepth

class ResNet():
    def __call__(self, input_shape, model_name, blocks = [3, 3, 3], no_classes = 10, no_filters = 16, weight_decay = 1e-4, stoch_depth = '0'):
        if weight_decay is not None:
            kernel_regularizer = keras.regularizers.l2(weight_decay)
        else:
            kernel_regularizer = None
        
        input = keras.layers.Input(shape = input_shape)

        conv = keras.layers.Conv2D(no_filters, 3, padding='same', kernel_initializer = 'he_uniform', kernel_regularizer = kernel_regularizer, name='head_conv')(input)
        conv = keras.layers.BatchNormalization(name = 'head_norm')(conv)
        conv = keras.layers.ReLU(name = 'head_relu')(conv)
        
        l = 1
        L = sum(blocks)

        for idx, block in enumerate(blocks):
            conv = _res_stack(conv, block, no_filters * np.power(2, idx), f'res_stack_{idx}', first_stride = 1, l = l, L = L, stoch_depth = stoch_depth)

            l = l + block
        
        conv = keras.layers.GlobalAveragePooling2D()(conv)
        
        output = keras.layers.Dense(no_classes, activation = 'softmax')(conv)

        model = keras.models.Model(
            inputs = input,
            outputs = output,
            name = model_name)
        
        return model

def _res_stack(input, no_blocks, no_filters, block_prefix, kernel_regularizer = None, first_stride = 2, l = 1, L = 1, stoch_depth = '0'):
    conv = _res_block(input, no_filters, f'{block_prefix}_res_block_0', kernel_regularizer = kernel_regularizer, use_conv = True, stride = first_stride, l = l, L = L, stoch_depth = stoch_depth)
    for n in range(1, no_blocks):
        conv = _res_block(conv, no_filters, f'{block_prefix}_res_block_{n}', kernel_regularizer = kernel_regularizer, l = l, L = L, stoch_depth = stoch_depth)
        
    return conv
    
def _res_block(input, no_filters, block_prefix, kernel_regularizer = None, use_conv = False, stride = 1, l = 1, L = 1, stoch_depth = '0'):
    # Create the shortcut connection
    # First shortcut uses a conv to update the number of channels to the expected output of the res block   
    if use_conv:
        shortcut = keras.layers.Conv2D(no_filters, 1, strides = stride, padding='same', kernel_initializer = 'he_uniform', kernel_regularizer = kernel_regularizer, name = f'{block_prefix}_skip_conv')(input)
        shortcut = keras.layers.BatchNormalization(name = f'{block_prefix}_skip_norm')(shortcut)
    else:
        shortcut = input
        
    conv = keras.layers.Conv2D(no_filters, 3, strides = stride, padding='same', kernel_initializer = 'he_uniform', kernel_regularizer = kernel_regularizer, name = f'{block_prefix}_conv_1')(input)
    conv = keras.layers.BatchNormalization(name = f'{block_prefix}_norm_1')(conv)
    conv = keras.layers.ReLU(name = f'{block_prefix}_relu_1')(conv)
    conv = keras.layers.Conv2D(no_filters, 3, padding = 'same', kernel_initializer = 'he_uniform', kernel_regularizer = kernel_regularizer, name = f'{block_prefix}_conv_2')(conv)
    conv = keras.layers.BatchNormalization(name = f'{block_prefix}_norm_2')(conv)

    # Default p_L = 0.5
    p_l = 1 - (l / L) * (1 - 0.5)
    
    if stoch_depth == '0':
        conv = keras.layers.Add()([shortcut, conv])
    elif stoch_depth == '1':                                  # Use Wrapper
        conv = StochDepth(keras.layers.Add(name = f'{block_prefix}_add'), p_l, name = f'{block_prefix}_stoch_depth')([shortcut, conv])
    else:                                                   # Use layer
        conv = ResDrop(p_l, name = f'{block_prefix}_stoch_depth')([shortcut, conv])
    
    conv = keras.layers.ReLU(name = f'{block_prefix}_relu_2')(conv)
    
    return conv