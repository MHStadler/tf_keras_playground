import tensorflow as tf

class StochDepth(tf.keras.layers.Wrapper):
    def __init__(self, layer, p_l, **kwargs):
        super().__init__(layer, **kwargs)
        
        self.layer = layer
        self.p_l = p_l
        
    def build(self, input_shape):
        if not self.layer.built:
            self.layer.build(input_shape)
            
        self.built = True
        
    def call(self, inputs):
        def train():
            def full_layer():
                return self.layer(inputs)
            def skip_layer():
                return inputs[0]

            return tf.cond(tf.random.uniform([]) < self.p_l, lambda: full_layer(), lambda: skip_layer())
        
        def test():
            return self.layer([inputs[0], inputs[1] * self.p_l])
    
        return tf.keras.backend.in_train_phase(train, test)
        
    def compute_output_shape(self, input_shape):
        return tf.TensorShape(self.layer.compute_output_shape(input_shape).as_list())
    
    def get_config(self):
        base_config = super().get_config()
        
        config = {"p_l": self.p_l}
        
        return {**base_config, **config}