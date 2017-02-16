from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.models import Model
from keras.layers import Flatten, Dense, Input, BatchNormalization, merge
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.applications.imagenet_utils import decode_predictions, _obtain_input_shape
from keras import backend as K

def conv2d_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              name=None):
    """Utility function to apply conv + BN.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if K.image_dim_ordering() == 'th':
        bn_axis = 1
    else:
        bn_axis = 3
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation='relu',
                      border_mode=border_mode,
                      name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
    return x

def LearnKit19(include_top=True,input_tensor=None, input_shape=None,classes=1000):

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=139,
                                      dim_ordering=K.image_dim_ordering(),
                                      include_top=include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor


    #1 224x224x32
    x = conv2d_bn(img_input, 32, 3, 3, subsample=(1, 1))

    # 112x112x32
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #2 112x112x64
    x = conv2d_bn(x, 64, 3, 3)

    # 56x56x64
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #3 56x56x128
    x = conv2d_bn(x, 128, 3, 3)
    #4 56x56x64
    x = conv2d_bn(x, 64, 1, 1)
    #5 56x56x128
    x = conv2d_bn(x, 128, 3, 3)

    # 28x28x128
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #6 28x28x256
    x = conv2d_bn(x, 256, 3, 3)
    #7 28x28x128
    x = conv2d_bn(x, 128, 1, 1)
    #8 28x28x256
    x = conv2d_bn(x, 256, 3, 3)

    # 14x14x256
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #9 14x14x512
    x = conv2d_bn(x, 512, 3, 3)
    #10 14x14x256
    x = conv2d_bn(x, 256, 1, 1)
    #11 14x14x512
    x = conv2d_bn(x, 512, 3, 3)
    #12 14x14x256
    x = conv2d_bn(x, 256, 1, 1)
    #13 14x14x512
    x = conv2d_bn(x, 512, 3, 3)

    # 7x7x512
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)
    #14 7x7x1024
    x = conv2d_bn(x, 1024, 3, 3)
    #15 7x7x512
    x = conv2d_bn(x, 512, 1, 1)
    #16 7x7x1024
    x = conv2d_bn(x, 1024, 3, 3)
    #17 7x7x512
    x = conv2d_bn(x, 512, 1, 1)
    #18 7x7x1024
    x = conv2d_bn(x, 1024, 3, 3)

    #17 7x7x1024
    x = conv2d_bn(x, 1024, 1, 1)

    if include_top:
        # Classification block
        x = AveragePooling2D((7, 7), strides=None, name='avg_pool')(x)
        x = Flatten(name='flatten')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)

    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='learnkit_19')

    return model

if __name__=="__main__":
    base_model = LearnKit19(include_top=False)
    print(base_model.output)
