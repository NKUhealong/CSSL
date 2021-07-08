"""
Attention UNet architecture
"""
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

# NOTE: IMAGE CONVENTION IS (W, H, C)

def conv_block(inputs, ch_out, trainit=True, dropout_rate=0.0, block_id=-1):
    prefix = 'conv_' + str(block_id) + '_'
    x = Conv2D(ch_out, kernel_size=3, strides=(1, 1), padding='same', name=prefix+'c1', trainable=trainit)(inputs)
    x = Dropout(dropout_rate, name=prefix+'d1', trainable=trainit)(x)
    x = BatchNormalization(name=prefix+'bn1', trainable=trainit)(x)
    x = ReLU(name=prefix+'relu1', trainable=trainit)(x)
    x = Conv2D(ch_out, kernel_size=3, strides=(1, 1), padding='same', name=prefix+'x2', trainable=trainit)(x)
    x = Dropout(dropout_rate, name=prefix+'d2', trainable=trainit)(x)
    x = BatchNormalization(name=prefix+'bn2', trainable=trainit)(x)
    x = ReLU(name=prefix+'relu2', trainable=trainit)(x)
    return x

def deconv_block(inputs, ch_out, trainit=True, dropout_rate=0.0, block_id=-1):
    prefix = 'deconv_' + str(block_id) + '_'
    x = Conv2DTranspose(ch_out, kernel_size=2, strides=(2, 2), padding='valid', name=prefix+'dc', trainable=trainit)(inputs)
    x = Dropout(dropout_rate, name=prefix+'d', trainable=trainit)(x)
    return x

def attention_block(inputs, inputs_skip, ch_mid, trainit=True, dropout_rate=0.0, block_id=-1):
    prefix = 'att_' + str(block_id) + '_'
    x = Conv2D(ch_mid, kernel_size=1, padding='same', name=prefix+'cx', trainable=trainit)(inputs)
    x = Dropout(dropout_rate, name=prefix+'dx', trainable=trainit)(x)
    x = BatchNormalization(name=prefix+'bnx', trainable=trainit)(x)

    g = Conv2D(ch_mid, kernel_size=1, padding='same', name=prefix+'cs', trainable=trainit)(inputs_skip)
    g = Dropout(dropout_rate, name=prefix+'ds', trainable=trainit)(g)
    g = BatchNormalization(name=prefix+'bns', trainable=trainit)(g)

    psi = Add(name=prefix+'add', trainable=trainit)([g, x])
    psi = ReLU(name=prefix+'relu', trainable=trainit)(psi)
    psi = Conv2D(1, kernel_size=1, padding='same', name=prefix+'c', trainable=trainit)(psi)
    psi = BatchNormalization(name=prefix+'bn', trainable=trainit)(psi)
    psi = Activation('sigmoid', name=prefix+'sig', trainable=trainit)(psi)

    ret = Multiply(name=prefix+'mul', trainable=trainit)([inputs, psi])
    return ret

def attention_unet_refined(input_shape, out_channels, multiplier, freeze_encoder, freeze_decoder, dropout_rate):
    """
    input_shape = (W, H) -- RGB Image
    out_channels = number of output segmentation masks
    multiplier = the scale by which the channels of the network can be increased
    dim1 is the number of values per IR box and dim2...4 are for each tissue parts
    returns a TF Keras model for attention UNet
    """
    drate = dropout_rate
    scale = int(multiplier)
    train_enc = not freeze_encoder
    train_dec = not freeze_decoder

    image_input = Input(shape=(input_shape[0], input_shape[1], 3), name='input')

    # ENCODER Network
    x1 = conv_block(inputs=image_input, ch_out=8*scale, trainit=train_enc, dropout_rate=drate, block_id=1)
    
    x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x1)
    x2 = conv_block(x2, 16*scale, train_enc, drate, block_id=2)

    x3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x2)
    x3 = conv_block(x3, 32*scale, train_enc, drate, block_id=3)

    x4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x3)
    x4 = conv_block(x4, 64*scale, train_enc, drate, block_id=4)

    x5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x4)
    x5 = conv_block(x5, 128*scale, train_enc, drate, block_id=5)

    # DECODER Network
    d5 = deconv_block(inputs=x5, ch_out=64*scale, trainit=train_dec, dropout_rate=drate, block_id=5)
    s4 = attention_block(inputs=d5, inputs_skip=x4, ch_mid=32*scale, trainit=train_dec, dropout_rate=drate, block_id=5)
    d5 = Concatenate(axis=-1)([d5, s4])
    d5 = conv_block(d5, 64*scale, train_dec, drate, block_id=6)

    d4 = deconv_block(inputs=d5, ch_out=32*scale, trainit=train_dec, dropout_rate=drate, block_id=4)
    s3 = attention_block(inputs=d4, inputs_skip=x3, ch_mid=16*scale, trainit=train_dec, dropout_rate=drate, block_id=4)
    d4 = Concatenate(axis=-1)([d4, s3])
    d4 = conv_block(d4, 32*scale, train_dec, drate, block_id=7)

    d3 = deconv_block(inputs=d4, ch_out=16*scale, trainit=train_dec, dropout_rate=drate, block_id=3)
    s2 = attention_block(inputs=d3, inputs_skip=x2, ch_mid=8*scale, trainit=train_dec, dropout_rate=drate, block_id=3)
    d3 = Concatenate(axis=-1)([d3, s2])
    d3 = conv_block(d3, 16*scale, train_dec, drate, block_id=8)

    d2 = deconv_block(inputs=d3, ch_out=8*scale, trainit=train_dec, dropout_rate=drate, block_id=2)
    s1 = attention_block(inputs=d2, inputs_skip=x1, ch_mid=4*scale, trainit=train_dec, dropout_rate=drate, block_id=2)
    d2 = Concatenate(axis=-1)([d2, s1])
    d2 = conv_block(d2, 8*scale, train_dec, drate, block_id=9)

    d1 = Conv2D(filters=out_channels, kernel_size=1, strides=(1, 1), padding='valid', name='conv1x1', trainable=train_dec)(d2)
    d1 = Activation('sigmoid', name='out')(d1)

    # Model the Attention-Unet network
    att_unet = Model(inputs=image_input, outputs=d1 , name='attention_unet')

    return att_unet



if __name__ == "__main__":

    model = attention_unet((320, 256), 3, 7, False, 0.1)
    model.summary()
    # import tensorflow as tf
    # tf.keras.utils.plot_model(model, show_shapes=True)