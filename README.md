# code-guideline
Master code design guideline for Elint standards of quality.

![Alt text](./img/meme.jpg)

# Guideline

## ```Files```

Always include the shebang line ```#!/usr/bin/env python``` to new files so $PATH configurations are respected.

## ```Folders```

Always include ```__init__.py``` to folders you intend importing modules from.

## ```Functions```

Suppose you want to write a function that multiplies and integer by two. A good example of formatting is:

```
def double(value):
    """
    Utility function to double a value by two

    Args:
        value (int or float): Input value to be doubled.

    Returns:
        float: Input value doubled
    """

    return value * 2.
```

Aways start with ```Utility function to...```; it helps to understand what the functions are supposed to do. Also, follow the formatation of arguments and returns, it helps to debug the code and also to understand the code when someone faces it by first time.

Follow the example to know how to format a function that returns more than one argument:

```
def get_gradients_and_filters(
    model, images, layer_name, class_index, use_guided_grads
):
    """
    Generate guided gradients and convolutional outputs with an inference.

    Args:
        model (tf.keras.Model): tf.keras model to inspect
        images (numpy.ndarray): 4D-Tensor with shape (batch_size, H, W, 3)
        layer_name (str): Targeted layer for GradCAM
        class_index (int): Index of targeted class
        use_guided_grads (boolean): Whether to use guided grads or raw gradients

    Returns:
        Tuple[tf.Tensor, tf.Tensor]: (Target layer outputs, Guided gradients)
    """
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        inputs = tf.cast(images, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, class_index]

    grads = tape.gradient(loss, conv_outputs)

    if use_guided_grads:
        grads = (
            tf.cast(conv_outputs > 0, "float32")
            * tf.cast(grads > 0, "float32")
            * grads
        )

    return conv_outputs, grads
```

**Note**: do not focus on what the function does. Focus on formatting.

## ```Classes```

Classes follow the formatation of function, e.g.:

```
class Father:
    """
    Utility class to define a Father

    Args:
        age (int): Age of the father, e.g. 47, 55, etc...
        name (str): Name of the father, e.g. Kaue Cano, Matheus Tosta, etc...
    """

    def __init__(self, age, name):

        self.age = age
        self.name = name
```

Note that the constructor doesn't require any explanation since it is a constructor. Beware to follow the function guideline for each function of the class. See a more complex example:

```
import tensorflow as tf

class ConvDiscriminator:
    """Utility class to build the discriminator.
    By the [paper](https://arxiv.org/abs/1703.10593v6) in section 4, the
    generative network architecture is adopt from [Johnson et al.](
                                                            https://arxiv.org/abs/1603.08155)

    Args:
        input_shape (tuple): Shape of input tensor
        dim (int): Number of filters of first convolutional layer
        num_downsamplings (int): Number of downsamplings
        norm (str): String indicating which normalization to use. Available: None, 'batch_norm',
                    'instance_norm' and 'layer_norm'
        lr_scheduler (tf.keras.optimizers.schedules.LearningRateSchedule): Learning rate scheduler
                    class
    """

    def __init__(self,
        input_shape=(256, 256, 3),
        dim=64,
        num_downsamplings=3,
        norm='instance_norm',
        lr_scheduler=LinearDecay(0.0001, 200, 100, 0.5)):

        self.norm = get_norm_layer(norm)
        self.input_shape = input_shape
        self.dim = dim
        self.dim_ = dim
        self.num_downsamplings = num_downsamplings
        self.lr_scheduler = lr_scheduler

        # build model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_scheduler, beta_1=self.lr_scheduler.beta_1)
        self.model = self.build()

    def build(self):
        """
        Utility function to build layers

        Returns:
            tf.keras.Model instance
        """
        x = inputs = tf.keras.Input(shape=self.input_shape)

        # 1
        x = tf.keras.layers.Conv2D(self.dim, 4, strides=2, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)        

        for _ in range(self.num_downsamplings - 1):
            self.dim = min(self.dim * 2, self.dim_ * 8)
            x = tf.keras.layers.Conv2D(self.dim, 4, strides=2, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER, use_bias=False)(x)
            x = self.norm()(x)
            x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # 2
        self.dim = min(self.dim * 2, self.dim_ * 8)
        x = tf.keras.layers.Conv2D(self.dim, 4, strides=1, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER, use_bias=False)(x)
        x = self.norm()(x)
        x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)

        # 3
        x = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same', kernel_initializer=CONV_KERNEL_INITIALIZER)(x)

        model = tf.keras.Model(inputs=inputs, outputs=x)

        model.compile(
            optimizer=self.optimizer,
            loss='mse',
            metrics=['accuracy']
        )

        return model
```

The above example shows a class that builds the convolutional discriminator of a generative adversarial network. Again, focus just on the formatation.

## ```Creating a module```

Suppose I want to write a module on my code called ```image``` and inside I'll write two functions: ```resize_activations``` and ```normalize_activations```. Follow the example to make such a thing:

```
"""Core module for image related operations"""
from skimage import transform
import tensorflow as tf
import numpy as np

def resize_activations(tensor, input_shape):
    """
    Utility function to resize a given tensor

    Args:
        tensor (tf.Tensor): 4D-Tensor with shape (batch_size, H, W, K)
        input_shape (Tuple[int, int]): shape of the input, e.g. (224, 224)

    Returns
        tensor (tf.Tensor): 4D-Tensor with shape (batch_size, K, H, W)
    """

    resized_activations = list()

    for j in range(tensor.shape[0]):

        activations = list()

        for i in range(tensor.shape[-1]):
            activations.append(
                transform.resize(tensor[j, ..., i], input_shape, preserve_range=True)
            )
        
        resized_activations.append(np.array(activations))

    return tf.convert_to_tensor(np.array(resized_activations), dtype=tf.float32)

def normalize_activations(tensor):
    """
    Utility function to normalize a given tensor

    Args:
        tensor (tf.Tensor): 4D-Tensor with shape (batch_size, K, H, W)
    
    Returns
        tf.Tensor: 4D-Tensor with shape (batch_size, K, H, W)
    """

    tensors = list()

    # goes through each image
    for i in range(tensor.shape[0]):
        flattened = tf.reshape(tensor[i], (tensor[i].shape[0], -1))

        max_a = tf.math.reduce_max(flattened, axis=1)
        min_a = tf.math.reduce_min(flattened, axis=1)

        diffs = tf.where(max_a > min_a, max_a - min_a, 1)

        normalized_tensor = (tensor[i] - tf.reshape(min_a, (-1, 1, 1))) / tf.reshape(diffs, (-1, 1, 1))

        tensors.append(normalized_tensor)
    
    return tf.stack(tensors, axis=0)
```

**Note**: focus just on formatting by now.

Note that the module starts with ```"""Core module for image related operations"""```. It helps anyone who reads the code to understand what happens in this specific module. **Aways** start with it.

## ```Importing modules```

It is very easy to make your code a mess, so, when importing modules beware to make it organized. Split the modules imported from your own code and installed. Example:

Suppode I have the same module described above. When importing, do:

```
# installed packages. Note that longer imports go firt
from PIL import Image
import pandas as pd
import numpy as np
import json
import io
import os

# modules from my code
from image import resize_activations, normalize_activations
```

Another option is to write ```import image``` and use ```image.resize_activations``` or ```image.normalize_activations```. This way who reads the code can know exactly from those functions are called.

## ```Defining requirements.txt```

When writing some code, you shall need to define which libraries you're using. **Aways** define the versions, e.g.:

```
tensorflow==2.1.0
keras==2.3.1
scikit-image==0.16.1
opencv-python==4.1.0.25
```

It is needed to prevent errors when running the code in another machine beyond it was tested.
