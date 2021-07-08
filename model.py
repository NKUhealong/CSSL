"""
Attention UNet architecture for multitask learning - Segmentation
"""
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.backend import random_normal
import tensorflow.keras.backend as K

# NOTE: IMAGE CONVENTION IS (W, H, C)

class FeaturePerturb(Layer):
    def __init__(self, intensity_var):
        # Intensity will be updated when the silhouette coeffecient of clusters improvs during training
        # self.intensity = 0.1
        super(FeaturePerturb, self).__init__()
        self.intensity_var = intensity_var
    
    def build(self, input_shape):
        self.dims = (1, int(input_shape[-3]), int(input_shape[-2]), int(input_shape[-1]))
        pass

    def call(self, input):
        # print(input.shape)
        rand_num = random_normal(self.dims, mean=0.0, stddev=K.get_value(self.intensity_var), dtype=tf.float32)
        return input + rand_num

        

def conv_block(inputs, ch_out, trainit=True, dropout_rate=0.0, block_id=-1):
    prefix = 'conv_' + block_id + '_'
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
    prefix = 'deconv_' + block_id + '_'
    x = Conv2DTranspose(ch_out, kernel_size=2, strides=(2, 2), padding='valid', name=prefix+'dc', trainable=trainit)(inputs)
    x = Dropout(dropout_rate, name=prefix+'d', trainable=trainit)(x)
    return x

def attention_block(inputs, inputs_skip, ch_mid, trainit=True, dropout_rate=0.0, block_id=-1):
    prefix = 'att_' + block_id + '_'
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

def encoder(multiplier, freeze_encoder, dropout_rate, prefix):
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

    image_input = Input(shape=(None,None,3), name =prefix+"_input")

    # ENCODER Network
    x1 = conv_block(inputs=image_input, ch_out=8*scale, trainit=train_enc, dropout_rate=drate, block_id=prefix+"_1")
    
    x2 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x1)
    x2 = conv_block(x2, 16*scale, train_enc, drate, block_id=prefix+"_2")

    x3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x2)
    x3 = conv_block(x3, 32*scale, train_enc, drate, block_id=prefix+"_3")

    x4 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x3)
    x4 = conv_block(x4, 64*scale, train_enc, drate, block_id=prefix+"_4")

    x5 = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x4)
    x5 = conv_block(x5, 128*scale, train_enc, drate, block_id=prefix+"_5")

    encoder = Model(inputs=image_input, outputs=[x1,x2,x3,x4,x5], name="encoder")

    return encoder

def decoder(out_channels, multiplier, freeze_decoder, dropout_rate, prefix):
    
    drate = dropout_rate
    scale = int(multiplier)
    train_dec = not freeze_decoder

    x1 = Input((320,256,8*scale), name =prefix+"_skip4")
    x2 = Input((160,128,16*scale), name =prefix+"_skip3")
    x3 = Input((80,64,32*scale), name =prefix+"_skip2")
    x4 = Input((40,32,64*scale), name =prefix+"_skip1")
    x5 = Input((20,16,128*scale), name=prefix+"_latent_features")

    d5 = deconv_block(inputs=x5, ch_out=64*scale, trainit=train_dec, dropout_rate=drate, block_id=prefix+"_5")
    s4 = attention_block(inputs=d5, inputs_skip=x4, ch_mid=32*scale, trainit=train_dec, dropout_rate=drate, block_id=prefix+"_5")
    d5 = Concatenate(axis=-1)([d5, s4])
    d5 = conv_block(d5, 64*scale, train_dec, drate, block_id=prefix+"_6")

    d4 = deconv_block(inputs=d5, ch_out=32*scale, trainit=train_dec, dropout_rate=drate, block_id=prefix+"_4")
    s3 = attention_block(inputs=d4, inputs_skip=x3, ch_mid=16*scale, trainit=train_dec, dropout_rate=drate, block_id=prefix+"_4")
    d4 = Concatenate(axis=-1)([d4, s3])
    d4 = conv_block(d4, 32*scale, train_dec, drate, block_id=prefix+"_7")

    d3 = deconv_block(inputs=d4, ch_out=16*scale, trainit=train_dec, dropout_rate=drate, block_id=prefix+"_3")
    s2 = attention_block(inputs=d3, inputs_skip=x2, ch_mid=8*scale, trainit=train_dec, dropout_rate=drate, block_id=prefix+"_3")
    d3 = Concatenate(axis=-1)([d3, s2])
    d3 = conv_block(d3, 16*scale, train_dec, drate, block_id=prefix+"_8")

    d2 = deconv_block(inputs=d3, ch_out=8*scale, trainit=train_dec, dropout_rate=drate, block_id=prefix+"_2")
    s1 = attention_block(inputs=d2, inputs_skip=x1, ch_mid=4*scale, trainit=train_dec, dropout_rate=drate, block_id=prefix+"_2")
    d2 = Concatenate(axis=-1)([d2, s1])
    d2 = conv_block(d2, 8*scale, train_dec, drate, block_id=prefix+"_9")

    d1 = Conv2D(filters=out_channels, kernel_size=1, strides=(1, 1), padding='valid', name=prefix+'_conv1x1', trainable=train_dec)(d2)
    d1 = Activation('sigmoid', name=prefix+'_out')(d1)

    decoder = Model(inputs=[x1,x2,x3,x4,x5], outputs=d1, name=prefix+"_decoder")

    return decoder


def CLCR_model_cl(img_shape, encoder, decoder, intensity_var, num_consistents=2):
    # encode the patches with a backbone encoder
    inp = Input((64, 64, 3))
    _, _, _, _, feats = encoder(inp)
    emb = Flatten()(feats)
    emb = Dense(units=256, name='emb')(emb)

    # Consistency regularization of image level inputs
    img_inp = Input((img_shape[0], img_shape[1], 3))
    img_feats = encoder(img_inp)
    img_feats_rand = [FeaturePerturb(intensity_var)(img_feat) for img_feat in img_feats]
    # mask_out = decoder(img_feats)
    mask_out_rand = decoder(img_feats_rand)

    model = Model(inputs=[inp, img_inp], outputs=[emb, mask_out_rand], name='CLCR_encoder_cl')
    return model
