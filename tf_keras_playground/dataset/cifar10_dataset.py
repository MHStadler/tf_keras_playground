import numpy as np
import tensorflow as tf

class Cifar10Dataset():
    def __call__(self, batch_size = 128):
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
        x_train = x_train / 255
        
        shuffle_idx = np.random.permutation(x_train.shape[0])
        shuffled_x_train = x_train[shuffle_idx, :] 
        shuffled_y_train = y_train[shuffle_idx]
        
        X_train = shuffled_x_train[0:45000, :]
        y_train = shuffled_y_train[0:45000]
        X_val = shuffled_x_train[45000:, :]
        y_val = shuffled_y_train[45000:]

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.cache()
        train_dataset = train_dataset.shuffle(buffer_size = 10 * batch_size)
        train_dataset = train_dataset.repeat()
        train_dataset = _prepare_training_dataset(train_dataset)
        train_dataset = train_dataset.batch(batch_size)
        train_dataset = train_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)

        test_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        test_dataset = test_dataset.batch(batch_size)
        test_dataset = test_dataset.prefetch(buffer_size = tf.data.experimental.AUTOTUNE)
    
        meta_data = {
            'N_train': X_train.shape[0],
            'N_test': X_val.shape[0]
        }

        return meta_data, train_dataset, test_dataset

def _prepare_training_dataset(dataset):
    def _prepare_training_dataset(img, y):
        paddings = tf.constant([[4, 4],   # top and bottom of image
                        [4, 4],   # left and right
                        [0, 0]])  # the channels dimension

        img = tf.pad(img, paddings)
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_crop(img, (32, 32, 3))
        
        return img, y

    return dataset.map(_prepare_training_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)