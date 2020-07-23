import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops

class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, goal_learning_rate, warmup_steps, follow_up_schedule, name = None):
        super(WarmUpSchedule, self).__init__()

        self.goal_learning_rate = goal_learning_rate
        self.warmup_steps = warmup_steps
        self.follow_up_schedule = follow_up_schedule
        self.name = name
    
    def __call__(self, step):
        def warmup(step):
            with ops.name_scope_v2(self.name or 'WarmUpSchedule'):
                goal_learning_rate = ops.convert_to_tensor_v2(self.goal_learning_rate, name = 'goal_learning_rate')
                dtype = goal_learning_rate.dtype
      
                warmup_steps = math_ops.cast(self.warmup_steps, dtype)
                global_step = math_ops.cast(step, dtype)
        
                # Linearly scale up learning rate towards goal_learning_rate
                warmup_frac = global_step / warmup_steps
                
                return math_ops.multiply(goal_learning_rate, warmup_frac)
            
        def follow_up(step):
            step = step - self.warmup_steps
            
            lr_t = self.follow_up_schedule
        
            if callable(lr_t):
                lr_t = lr_t()
        
            if isinstance(lr_t, tf.keras.optimizers.schedules.LearningRateSchedule):
                lr_t = lr_t(step)
                
            return lr_t
        
        # Return warmup schedule, or delegate on to the follow up schedule once warmup steps are exceeded    
        return tf.cond(step < self.warmup_steps, lambda: warmup(step), lambda: follow_up(step))
    
    def _follow_up_lr(self, step):
        lr_t = self.follow_up_schedule
        
        if callable(lr_t):
            lr_t = lr_t()
        
        if isinstance(lr_t, tf.keras.optimizers.schedules.LearningRateSchedule):
            lr_t = lr_t(step)
        
        return lr_t
    
    def get_config(self):
        config = {
            'goal_learning_rate': self.goal_learning_rate,
            'warmup_steps': self.warmup_steps,
            'follow_up_schedule': self._serialize_follow_up_schedule(),
            'name': self.name
        }
            
        return config        
    
    @classmethod
    def from_config(cls, config, custom_objects=None):
        if isinstance(config["follow_up_schedule"], dict):
            config["follow_up_schedule"] = tf.keras.optimizers.schedules.deserialize(config["follow_up_schedule"], custom_objects=custom_objects)
        
        return cls(**config)
    
    def _serialize_follow_up_schedule(self):
        follow_up_schedule = self.follow_up_schedule
        
        if callable(follow_up_schedule):
            follow_up_schedule = follow_up_schedule()
        
        if isinstance(follow_up_schedule, tf.keras.optimizers.schedules.LearningRateSchedule):
            return learning_rate_schedule.serialize(follow_up_schedule)
        elif tensor_util.is_tensor(follow_up_schedule):
            return tf.keras.backend.get_value(follow_up_schedule)
        else:
            return follow_up_schedule