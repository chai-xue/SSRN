import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout
)
from keras.layers.convolutional import (
    Conv3D,
    MaxPooling3D,
    AveragePooling3D
)
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, Input
from keras.regularizers import l2
from keras import backend as K
from keras.layers import add
from keras import regularizers


def _bn_relu(input):
    """Helper to build a BN -> relu block"""
    norm = BatchNormalization(axis=-1)(input)  # Adjusted for Keras backend axis
    return Activation("relu")(norm)


def _bn_relu_spc(input):
    """Helper to build a BN -> relu block"""
    norm = BatchNormalization(axis=-1)(input)  # Adjusted for Keras backend axis
    return Activation("relu")(norm)


def _conv_bn_relu_spc(**conv_params):
    """Helper to build a conv -> BN -> relu block"""
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))  # strides
    init = conv_params.setdefault("init", "he_normal")
    W_regularizer = conv_params.setdefault("W_regularizer", regularizers.l2(1.e-4))

    def f(input):
        print(f'Input shape before BN and ReLU: {input.shape}')

        # Apply BN and ReLU activation
        activation = _bn_relu_spc(input)
        print(f'Input shape after BN and ReLU: {activation.shape}')

        # Apply Conv3D with 'same' padding and reduced stride to avoid excessive dimension reduction
        conv_output = Conv3D(filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3),
                             strides=(1, 1, 1), padding='same',  # Reduce stride and use 'same' padding
                             kernel_initializer=init, kernel_regularizer=W_regularizer)(activation)

        print(f'Output shape after Conv3D: {conv_output.shape}')

        # Optionally add pooling layer to avoid further downsampling
        conv_output = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), padding='same')(conv_output)
        print(f'Output shape after MaxPooling: {conv_output.shape}')

        return conv_output

    return f






def _bn_relu_spc(input):
    """Helper to add batch normalization and ReLU"""
    norm = BatchNormalization()(input)
    return Activation('relu')(norm)


def _bn_relu_conv_spc(**conv_params):
    """Helper to build a BN -> relu -> conv block"""
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.setdefault("subsample", (1, 1, 1))
    init = conv_params.setdefault("init", "he_normal")
    W_regularizer = conv_params.setdefault("W_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu_spc(input)
        print(f'Input shape: {input.shape}')

        return Conv3D(filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3),
                      strides=subsample, kernel_initializer=init, kernel_regularizer=W_regularizer)(activation)


    return f


def _shortcut_spc(input, residual):
    """Adds a shortcut between input and residual block and merges them with "sum"."""
    stride_dim3 = (input.shape[3] + 1) // residual.shape[3]
    equal_channels = residual.shape[-1] == input.shape[-1]

    shortcut = input
    if stride_dim3 > 1 or not equal_channels:
        shortcut = Conv3D(filters=residual.shape[-1], kernel_size=(1, 1, 1),
                          strides=(1, 1, stride_dim3), padding="valid", kernel_initializer="he_normal",
                          kernel_regularizer=l2(0.0001))(input)

    return add([shortcut, residual])


def _residual_block_spc(block_function, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating bottleneck blocks."""
    def f(input):
        for i in range(repetitions):
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (1, 1, 2)
            input = block_function(nb_filter=nb_filter, init_subsample=init_subsample,
                                   is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


def basic_block_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Basic 3 x 3 convolution blocks for use on ResNets with layers <= 34."""
    def f(input):
        if is_first_block_of_first_layer:
            conv1 = Conv3D(filters=nb_filter, kernel_size=(1, 1, 7), strides=init_subsample,
                           padding="same", kernel_initializer="he_normal", kernel_regularizer=l2(0.0001))(input)
        else:
            conv1 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7,
                                      subsample=init_subsample)(input)

        residual = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=7)(conv1)
        return _shortcut_spc(input, residual)

    return f


def bottleneck_spc(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for > 34 layer ResNet."""
    def f(input):
        if is_first_block_of_first_layer:
            conv_1_1 = Conv3D(filters=nb_filter, kernel_size=(1, 1, 1), strides=init_subsample, padding="same",
                              kernel_initializer="he_normal", kernel_regularizer=l2(0.0001))(input)
        else:
            conv_1_1 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1,
                                         subsample=init_subsample)(input)

        conv_3_3 = _bn_relu_conv_spc(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv_1_1)
        residual = _bn_relu_conv_spc(nb_filter=nb_filter * 4, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1)(conv_3_3)
        return _shortcut_spc(input, residual)

    return f


def _handle_dim_ordering():
    global CONV_DIM1, CONV_DIM2, CONV_DIM3, CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        CONV_DIM1, CONV_DIM2, CONV_DIM3 = 1, 2, 3
        CHANNEL_AXIS = -1
    else:
        CHANNEL_AXIS = 1
        CONV_DIM1, CONV_DIM2, CONV_DIM3 = 2, 3, 4


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn_spc, block_fn, repetitions1, repetitions2):
        """Builds a custom ResNet like architecture."""
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols, nb_depth)")

        inputs = Input(shape=input_shape)

        # Define the ResNet architecture
        x = _conv_bn_relu_spc(nb_filter=64, kernel_dim1=7, kernel_dim2=7, kernel_dim3=7, subsample=(2, 2, 2))(inputs)

        x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(x)

        # Apply residual blocks
        x = _residual_block_spc(block_fn_spc, 64, repetitions1)(x)
        x = _residual_block_spc(block_fn_spc, 128, repetitions2)(x)

        # Final layers
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_outputs, activation="softmax")(x)

        model = Model(inputs, outputs)
        return model

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        """Builds a custom ResNet architecture with 8 layers."""
        _handle_dim_ordering()
        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols, nb_depth)")

        inputs = Input(shape=input_shape)

        # Define the ResNet architecture with 8 layers
        x = _conv_bn_relu_spc(nb_filter=64, kernel_dim1=7, kernel_dim2=7, kernel_dim3=7, subsample=(2, 2, 2))(inputs)
        x = MaxPooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same")(x)

        # Apply residual blocks (adjust the number of repetitions)
        x = _residual_block_spc(bottleneck_spc, 64, 2)(x)  # First 2 blocks
        x = _residual_block_spc(bottleneck_spc, 128, 2)(x)  # Next 2 blocks
        x = _residual_block_spc(bottleneck_spc, 256, 2)(x)  # Next 2 blocks
        x = _residual_block_spc(bottleneck_spc, 512, 2)(x)  # Last 2 blocks

        # Final layers
        x = Flatten()(x)
        x = Dense(512, activation="relu")(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_outputs, activation="softmax")(x)

        model = Model(inputs, outputs)
        return model

