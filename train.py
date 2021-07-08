import os
from tqdm import tqdm
from PIL import ImageEnhance, Image
from data_loader import DataGenerator
from model import encoder, CLCR_model_cl, decoder
import time
from attn_unet import *
from metrics import *
from utils import *
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger, TensorBoard
from tensorflow.keras.utils import plot_model
# tf.random.set_seed(100)
# np.random.seed(100)

## Config
model_path = "weights/temp.h5" # Saving location
log_path = "logs/temp"
weights_paths = ["Path here"]   # Weight path for trainOn50.h5 (or loading location for resuming training)
train_path = "Path here"
valid_path = "Path here"
image_folder = "Path here"
batch_size = 8
S = 816
image_size = (320, 256)
target_classes = ["Good Crypts", "Good Villi", "Epithelium", "Brunner's Gland"]
lr = 1e-3
epochs = 500
aux_class = False

# ADDS AN AUXILLIARY CLASS IN CASE OF SINGLE CLASS DATASETS
if len(target_classes) == 1:
    aux_class = True

enc = encoder(multiplier=8, freeze_encoder=False, dropout_rate=0.0, prefix='enc')
dec = decoder(len(target_classes), 8, False, 0.0, 'dec')
# A VARIABLE THAT HOLDS THE DISTORTION INTENSITY VALUE FOR CR
intensity_var = tf.Variable(0.1, trainable=False, name='intensity', dtype=tf.float32)
model = CLCR_model_cl(image_size, enc, dec, intensity_var)
model.summary()

cl_loss = CLCR_CL()
metrics = {'emb': silhouette}
optimizer = Adam(learning_rate=lr)
model.compile(loss=[cl_loss.cl_loss_func, focal_loss], optimizer=optimizer, metrics=metrics)

att_unet = attention_unet_refined(input_shape=image_size, 
                                   out_channels=4, 
                                   multiplier=10, 
                                   freeze_encoder=False,
                                   freeze_decoder=False, 
                                   dropout_rate=0.0)

supModel = att_unet
supInit = False
for wp in weights_paths:
    supModel.load_weights(wp, by_name=True)
    supInit = True
    print("Loaded supervised weights!")
if not supInit:
    print("WARNING: Supervised model not initialised")
    exit

data_gen = DataGenerator(image_folder, train_path, image_size, target_classes, batch_size, model=supModel, augment=True, S=S, aux_class=aux_class)
valid_generator = DataGenerator(valid_path, None, image_size, target_classes, batch_size, model=supModel, augment=False, S=S, aux_class=aux_class)
callbacks = [ModelCheckpoint(model_path, save_weights_only=True),
            ReduceLROnPlateau(monitor='loss', factor=0.1, patience=15, verbose=1, min_lr=1e-6),
            EarlyStopping(monitor='loss', patience=30, restore_best_weights=False), 
            TensorBoard(log_dir=log_path, update_freq='epoch', write_graph=False, profile_batch=0),
            UpdateCRCallback(intensity_var)
]


model.fit_generator(generator = data_gen
                , steps_per_epoch = data_gen.__len__()
                , epochs = epochs
                , verbose = 1
                , callbacks = callbacks
                , validation_data= valid_generator
                , validation_steps= valid_generator.__len__()
                , workers = 8
                , max_queue_size = 20
                , initial_epoch = 0)
