# code-guideline
Master Python code design guideline for Elint standards of quality.

_Programs must be written for people to read, and only incidentally for machines to execute._ 
_-Hal Abelson_

![Alt text](./img/meme.jpg)

# Guideline

## ```Files```

Always include the shebang line ```#!/usr/bin/env python``` to new files so $PATH configurations are respected.

## ```Folders```

Always include ```__init__.py``` to folders you intend importing modules from.

## ```Functions```

Suppose you want to write a function that multiplies an integer by two. A good example of formatting is:

```python
def double(value: int|float) -> int|float:
    """
    Utility function to double a value by two

    Args:
        value (int or float): Input value to be doubled.

    Returns:
        float: Input value doubled
    """

    return value * 2.
```

Try always to start with ```Utility function to...```; it helps to understand what the functions are supposed to do. Also, follow the formatting of arguments and returns as well as the type hinting, it helps to debug the code and to understand it better when someone faces it by the first time.

Remember that giving intuitive names to your functions, classes and variables as well as keeping your functions small and atomical are good practices and help on the code readability. If you have a code snippet that repeats itself, it is also worth of creating a specific function so it becomes reusable.

Follow the example to know how to format a function that returns more than one argument:

```python
from typing import Tuple

def get_gradients_and_filters(
    model: tf.keras.Model,
    images: numpy.ndarray,
    layer_name: str,
    class_index: int,
    use_guided_grads: boolean
) -> Tuple[tf.Tensor, tf.Tensor]:
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

**Note**: do not focus on what the function does. Focus on the formatting.

## ```Classes```

Classes follow the formatting of functions, e.g.:

```python
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

```python
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
        input_shape: tuple = (256, 256, 3),
        dim: int = 64,
        num_downsamplings: int = 3,
        norm: str = 'instance_norm',
        lr_scheduler: tf.keras.optimizers.schedules.LearningRateSchedule = LinearDecay(0.0001, 200, 100, 0.5)
    ):

        self.norm = get_norm_layer(norm)
        self.input_shape = input_shape
        self.dim = dim
        self.dim_ = dim
        self.num_downsamplings = num_downsamplings
        self.lr_scheduler = lr_scheduler

        # build model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr_scheduler, beta_1=self.lr_scheduler.beta_1)
        self.model = self.build()

    def build(self) -> tf.keras.Model:
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

## ```Creating a module```

Suppose you want to write a module on the code called ```image``` and inside it you need to write two functions: ```resize_activations``` and ```normalize_activations```. Follow the example to make such a thing:

```python
"""Core module for image related operations"""
from skimage import transform
import tensorflow as tf
import numpy as np

def resize_activations(tensor: tf.Tensor, input_shape: Tuple[int, int]) -> tf.Tensor:
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

def normalize_activations(tensor: tf.Tensor) -> tf.Tensor:
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

Note that the module starts with ```"""Core module for image related operations"""```. It helps anyone who reads the code to understand what happens in this specific module. Try **always** to start with it.

## ```Importing modules```

It is very easy to make your code a mess, so, when importing modules beware to make it organized. Split the modules between the ones imported from your own code and the installed ones. Example:

Suppose you have the same module described above. When importing, seek to do it as the following:

```python
# installed packages. Note that longer imports go first
from PIL import Image
import pandas as pd
import numpy as np
import json
import io
import os

# modules from your code
from image import resize_activations, normalize_activations
```

Another option is to write ```import image``` and use ```image.resize_activations``` or ```image.normalize_activations```. This way who reads the code can know exactly where those functions are called from.

## ```Dealing with Exceptions```

When you have to treat errors in the code, avoid using bare `except` clauses. The problem with these is that they catch `SystemExit` and `KeyboardInterrupt` exceptions, which makes it harder to interrupt a program using CTRL-C, and can also disguise other problems. It is recommended catching `Exception`, which will catch all errors. See an example below:

```python
# Instead of this:
try:
    some_important_code()
except:
    exception_handling_code()

# Seek to do the following:
try:
    some_important_code()
except Exception:
    exception_handling_code()
```

If possible, it is even better to catch specific errors so you know exactly how your code should behave in these specific cases. E.g.:
```python
async def token_in_blacklist(token: str):
    """
    Checks if token is in blacklist table.
    Returns True if token is in blacklist and False if not.
    """
    try:
        async with async_session.begin():
            query = select(TokenBlacklist).where(TokenBlacklist.token == token)
            results = await async_session.execute(query)
            data = results.scalars().one()
        return True
    except NoResultFound:
        return False
```

Also, raise specific errors so that callers of the code can handle the error appropriately. Besides that, it makes it easier to understand and debug. It can be done with Python built-in exceptions or custom defined ones.
```python
# built-in error
if incorrect_value():
    raise ValueError

# custom error
if incorrect_value():
    raise SpecificIncorrectValueError
```

When raising custom errors, define them following the base model:
```python
class ProductMismatchException(Exception):
    """Exception raised when user is trying to transfer from one compartment to another with different products ids.
    Attributes:
        message: explanation of the error
    """

    def __init__(self, message="Destination product does not match with origin product."):
        self.message = message
        super().__init__(self.message)
```

## ```Defining requirements.txt```

When writing some code, you shall need to define which libraries you're using. **Always** define the versions, e.g.:

```
tensorflow==2.1.0
keras==2.3.1
scikit-image==0.16.1
opencv-python==4.1.0.25
```

It is needed to prevent errors when running the code in another machine beyond it was tested.
