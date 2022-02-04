from keras.layers import Conv2D, BatchNormalization, Input, Concatenate, Conv2DTranspose, MaxPool2D
from keras.layers import Activation
from keras.models import Model

def conv_block(input, num_filters):
    x = Conv2D(num_filters, 3, padding="same")(input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(input, num_filters):
    x = conv_block(input, num_filters)
    p = MaxPool2D((2, 2))(x)

    return x, p

def decoder_block(input, num_filters, skip_connection):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(input)
    x = Concatenate()([x, skip_connection])
    x = conv_block(x, num_filters)

    return x

def build_unet(input_shape):
    input = Input(input_shape)
    s1, p1 = encoder_block(input, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    b = conv_block(p4, 1024)

    d1 = decoder_block(b, 512, s4)
    d2 = decoder_block(d1, 256, s3)
    d3 = decoder_block(d2, 128, s2)
    d4 = decoder_block(d3, 64, s1)

    output = Conv2D(3, 3, padding='same', activation='softmax')(d4)

    model = Model(input, output)

    return model