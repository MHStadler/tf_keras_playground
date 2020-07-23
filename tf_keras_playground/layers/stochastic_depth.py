import tensorflow as tf

class ResDrop(tf.keras.layers.Layer):
    def __init__(self, p_l, **kwargs):
        super(ResDrop, self).__init__(**kwargs)
        self.p_l = p_l

    def build(self, input_shape):
        super(ResDrop, self).build(input_shape)
        
    def call(self, x):
        # Random bernoulli variable with probability p_l, indiciathing wheter this branch should be kept or not or not
        b_l = tf.keras.backend.random_binomial([], p = self.p_l)
        
        def x_train():
            return x[0] + b_l * x[1]
        
        def x_test():
            return x[0] + self.p_l * x[1]
        
        return tf.keras.backend.in_train_phase(x_train, x_test)
    
    def get_config(self):
        config = super(ResDrop, self).get_config()
        
        config['p_l'] = self.p_l
        
        return config