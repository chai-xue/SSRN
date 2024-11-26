import six
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
    Conv3D,
    Add
)
from keras import regularizers
from keras import backend as K

# Handle channel ordering based on Keras configuration
def _handle_dim_ordering():
    global CONV_DIM1, CONV_DIM2, CONV_DIM3, CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':  # 'tf' dimension ordering
        CONV_DIM1, CONV_DIM2, CONV_DIM3 = 1, 2, 3
        CHANNEL_AXIS = 4
    else:  # 'th' dimension ordering
        CHANNEL_AXIS = 1
        CONV_DIM1, CONV_DIM2, CONV_DIM3 = 2, 3, 4

# BN -> ReLU block
def _bn_relu(input):
    """Helper to build a BN -> ReLU block"""
    norm = BatchNormalization(axis=CHANNEL_AXIS)(input)
    return Activation("relu")(norm)

# Conv -> BN -> ReLU block
def _conv_bn_relu(**conv_params):
    """Helper to build a Conv -> BN -> ReLU block"""
    nb_filter = conv_params["nb_filter"]
    kernel_dim1 = conv_params["kernel_dim1"]
    kernel_dim2 = conv_params["kernel_dim2"]
    kernel_dim3 = conv_params["kernel_dim3"]
    subsample = conv_params.get("subsample", (1, 1, 1))
    init = conv_params.get("init", "he_normal")
    padding = conv_params.get("padding", "same")
    W_regularizer = conv_params.get("W_regularizer", regularizers.l2(1.e-4))

    def f(input):
        conv = Conv3D(filters=nb_filter, kernel_size=(kernel_dim1, kernel_dim2, kernel_dim3),
                      strides=subsample, kernel_initializer=init, kernel_regularizer=W_regularizer, padding=padding)(input)
        return _bn_relu(conv)

    return f

# Shortcut connection with residual block addition
def _shortcut(input, residual):
    """Adds a shortcut between input and residual block"""
    stride_dim1 = (K.int_shape(input)[CONV_DIM1] + 1) // K.int_shape(residual)[CONV_DIM1]
    stride_dim2 = (K.int_shape(input)[CONV_DIM2] + 1) // K.int_shape(residual)[CONV_DIM2]
    stride_dim3 = (K.int_shape(input)[CONV_DIM3] + 1) // K.int_shape(residual)[CONV_DIM3]
    equal_channels = K.int_shape(residual)[CHANNEL_AXIS] == K.int_shape(input)[CHANNEL_AXIS]

    shortcut = input
    if not equal_channels or stride_dim1 > 1 or stride_dim2 > 1 or stride_dim3 > 1:
        shortcut = Conv3D(filters=K.int_shape(residual)[CHANNEL_AXIS], kernel_size=(1, 1, 1),
                          strides=(stride_dim1, stride_dim2, stride_dim3), padding='valid',
                          kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.0001))(input)

    return Add()([shortcut, residual])

# Residual block definition
# Residual block definition
def residual_block(block_fn, nb_filter, repetitions, is_first_layer=False):
    """Builds a residual block with repeating blocks"""
    def f(input):
        for i, rep in enumerate(repetitions):  # 修改这里，遍历 repetitions 列表
            init_subsample = (1, 1, 1)
            if i == 0 and not is_first_layer:
                init_subsample = (2, 2, 1)
            input = block_fn(nb_filter=nb_filter, init_subsample=init_subsample, is_first_block_of_first_layer=(is_first_layer and i == 0))(input)
        return input

    return f


# Basic block for resnet
def basic_block(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Basic 3x3 conv blocks for resnets with layers <= 34"""
    def f(input):
        if is_first_block_of_first_layer:
            conv1 = Conv3D(filters=nb_filter, kernel_size=(3, 3, 1), strides=init_subsample, padding='same',
                           kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.0001))(input)
        else:
            conv1 = _conv_bn_relu(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1,
                                  subsample=init_subsample)(input)

        residual = _conv_bn_relu(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv1)
        return _shortcut(input, residual)

    return f

# Bottleneck block for resnet (>34 layers)
def bottleneck(nb_filter, init_subsample=(1, 1, 1), is_first_block_of_first_layer=False):
    """Bottleneck architecture for resnets with >34 layers"""
    def f(input):
        if is_first_block_of_first_layer:
            conv_1_1 = Conv3D(filters=nb_filter, kernel_size=(1, 1, 1), strides=init_subsample, padding='same',
                              kernel_initializer="he_normal", kernel_regularizer=regularizers.l2(0.0001))(input)
        else:
            conv_1_1 = _conv_bn_relu(nb_filter=nb_filter, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1, subsample=init_subsample)(input)

        conv_3_3 = _conv_bn_relu(nb_filter=nb_filter, kernel_dim1=3, kernel_dim2=3, kernel_dim3=1)(conv_1_1)
        residual = _conv_bn_relu(nb_filter=nb_filter * 4, kernel_dim1=1, kernel_dim2=1, kernel_dim3=1)(conv_3_3)
        return _shortcut(input, residual)

    return f

# Build function for ResNet-like architecture
class ResnetBuilder(object):
    @staticmethod
    def build(input_shape, num_outputs, block_fn, repetitions):
        """Builds a ResNet-like model"""
        _handle_dim_ordering()

        if len(input_shape) != 4:
            raise Exception("Input shape should be a tuple (nb_channels, nb_rows, nb_cols, nb_depth)")

        # Permute dimension order if necessary
        if K.image_data_format() == 'channels_last':
            input_shape = (input_shape[1], input_shape[2], input_shape[3], input_shape[0])

        # Input layer
        inputs = Input(shape=input_shape)

        # Initial Conv Block
        x = _conv_bn_relu(nb_filter=64, kernel_dim1=7, kernel_dim2=7, kernel_dim3=3, subsample=(2, 2, 2))(inputs)

        # Residual Blocks
        x = residual_block(block_fn, nb_filter=64, repetitions=repetitions)(x)

        # Final layers
        x = Flatten()(x)
        x = Dense(num_outputs, activation='softmax')(x)

        return Model(inputs=inputs, outputs=x)

    @staticmethod
    def build_resnet_8(input_shape, num_outputs):
        """Builds a ResNet-8 model"""
        block_fn = bottleneck  # 或者选择 basic_block
        repetitions = [2, 2, 2, 2]  # ResNet-8 中每个阶段的重复次数
        return ResnetBuilder.build(input_shape, num_outputs, block_fn, repetitions)


# Example usage (Modify according to your own settings)
img_rows, img_cols, img_channels = 145, 145, 200  # Modify this based on your data
nb_classes = 10  # Modify based on your class count

# Define block function and repetitions
block_fn = bottleneck  # Or basic_block
repetitions = [3, 4, 6, 3]  # Example: 3 layers in stage 1, 4 in stage 2, etc.

# Build the model
model_res4 = ResnetBuilder.build(
    input_shape=(1, img_rows, img_cols, img_channels),  # (depth, height, width, channels)
    num_outputs=nb_classes,
    block_fn=block_fn,
    repetitions=repetitions
)

# Summary of the model
model_res4.summary()
