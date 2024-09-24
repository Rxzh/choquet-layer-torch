import argparse
import io
import os
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from skimage.io import imread
from shutil import copyfile
print(tf.__version__)
import matplotlib.pyplot as plt
plt.gray()
import random
import json
import datetime

parser = argparse.ArgumentParser(description='Results in Neyman-Generated',epilog='Enjoy! S. Velasco-Forero')
parser.add_argument('--pathoutput',type=str, default='/gpfsstore/rech/qpj/uyn98cq/choquet/', help='path output')
parser.add_argument('--batch', type=float, default=32., help='batch_size')
parser.add_argument('--epochs',  type=float, default=512.,   help='epochs')
parser.add_argument('--nlayers', type=float, default=1, help='nlayers')
parser.add_argument('--nfilters', type=float, default=48, help='nfilters')
parser.add_argument('--ksize', type=float, default=13, help='ksize')
parser.add_argument('--featurespace', type=float, default=12, help='featurespace')
parser.add_argument('--shrink', type=float, default=4, help='shrink')
parser.add_argument('--generatedata', type=float, default=1, help='One if to generate the Neyman-Scott point process')

args = parser.parse_args()


PATHOUT=args.pathoutput

BATCH_SIZE = int(args.batch)
EPOCHS = int(args.epochs)

batch_size = BATCH_SIZE
epochs = EPOCHS
learning_rate=0.001
CHANNELS=1
NLAYERS=int(args.nlayers)
NFILTERS=int(args.nfilters)
KSIZE=int(args.ksize)
SHRINK=int(args.shrink)
SUBSPACE=int(args.featurespace)
PATIENCE_ES=40
PATIENCE_RP=5
pathoutput=str(args.pathoutput)
output_dir_root =pathoutput

test_name=random.choice(number)+random.choice(adjective)+random.choice(colors)+random.choice(words)
dir_name = output_dir_root + "_" + test_name
print('dir_name',dir_name)

LOG=True
if LOG:
   os.makedirs(dir_name)
   dir_autosave_model_weights = os.path.join(dir_name, "autosave_model_weights")
   dir_autosave_model_stat = os.path.join(dir_name, "accuracy")
   os.makedirs(dir_autosave_model_weights)
   this_file_name = os.path.basename(__file__)
   copyfile(__file__, os.path.join(dir_name, this_file_name))
   print('copyfile','ok')



NSAMPLES_TRAINING=2024*2
IMG_SIZE=128
poisson_mean=100
daughter_max=50
pareto_scale=.02
pareto_alpha=1. #GENERATION ON IT


@tf.function
def dilation2d(x, st_element, strides, padding,rates=(1, 1)):
    """

    From MORPHOLAYERS (https://github.com/Jacobiano/morpholayers)
    Basic Dilation Operator
    :param st_element: Nonflat structuring element
    :strides: strides as classical convolutional layers
    :padding: padding as classical convolutional layers
    :rates: rates as classical convolutional layers
    """
    x = tf.nn.dilation2d(x, st_element, (1, ) + strides + (1, ),padding.upper(),"NHWC",(1,)+rates+(1,))
    return x

class DepthwiseDilation2D(Layer):
    '''
    Depthwise Dilation 2D Layer: Depthwise Dilation for now assuming channel last
    '''
    def __init__(self, kernel_size,depth_multiplier=1, strides=(1, 1),padding='same', dilation_rate=(1,1), kernel_initializer=tf.keras.initializers.RandomUniform(minval=-1., maxval=0.),
    kernel_constraint=None,kernel_regularization=None,**kwargs):
        super(DepthwiseDilation2D, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.depth_multiplier= depth_multiplier
        self.strides = strides
        self.padding = padding
        self.rates=dilation_rate
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.kernel_constraint = tf.keras.constraints.get(kernel_constraint)
        self.kernel_regularization = tf.keras.regularizers.get(kernel_regularization)
        # for we are assuming channel last
        self.channel_axis = -1

        # self.output_dim = output_dim

    def build(self, input_shape):
        if input_shape[self.channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')

        input_dim = input_shape[self.channel_axis]
        kernel_shape = self.kernel_size + (input_dim,self.depth_multiplier)
        self.kernel2D = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='kernel2D',constraint =self.kernel_constraint,regularizer=self.kernel_regularization)
        super(DepthwiseDilation2D, self).build(input_shape)

    def call(self, x):
        res=[]
        for di in range(self.depth_multiplier):
            H=tf.nn.dilation2d(x,self.kernel2D[:,:,:,di],strides=(1, ) + self.strides + (1, ),padding=self.padding.upper(),data_format="NHWC",dilations=(1,)+self.rates+(1,))
            res.append(H)
        return tf.concat(res,axis=-1)

    def compute_output_shape(self, input_shape):

        space = input_shape[1:-1]
        new_space = []
        for i in range(len(space)):
            new_dim = conv_utils.conv_output_length(
                space[i],
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.rates[i])
            new_space.append(new_dim)

        return (input_shape[0],) + tuple(new_space) + (self.depth_multiplier,)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_filters': self.num_filters,
            'kernel_size': self.kernel_size,
            'depth_multiplier': self.depth_multiplier,
            'strides': self.strides,
            'padding': self.padding,
            'dilation_rate': self.rates,
        })
        return config


##Create the Proposed Model
xinput = layers.Input(shape=(IMG_SIZE, IMG_SIZE, CHANNELS))
xconv=xinput
for i in range(NLAYERS):
    xconv = layers.Conv2D(NFILTERS,(3,3),padding='same',activation='relu')(xconv)
if NLAYERS>0:
    xconv = layers.Conv2D(NFILTERS//SHRINK,(1,1),padding='same',activation='relu')(xconv)
    xconv = DepthwiseDilation2D((KSIZE,KSIZE),depth_multiplier=SHRINK,padding='same')(xconv)
else:
    xconv = DepthwiseDilation2D((KSIZE,KSIZE),depth_multiplier=NFILTERS,padding='same')(xconv)
xfeatures=layers.GlobalAveragePooling2D()(xconv)
xfeatures=layers.BatchNormalization()(xfeatures)
xfeatures=layers.Dense(SUBSPACE,activation='relu')(xfeatures)
xfeatures=layers.Dense(SUBSPACE)(xfeatures)
xend=layers.Dense(1,activation='sigmoid')(xfeatures)
modelDil=tf.keras.Model(xinput,xend)
modelDil.summary()
print(modelDil.count_params())

CB2=[tf.keras.callbacks.EarlyStopping(patience=PATIENCE_ES,restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(patience=PATIENCE_RP,min_lr=1e-6),
    tf.keras.callbacks.CSVLogger(dir_autosave_model_stat+'Dil', separator=',', append=False)
   ]
modelDil.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=["mse","mae"])
histDil=modelDil.fit(listIm, listY, batch_size=batch_size, epochs=epochs,callbacks=CB2,validation_data=(listImVal, listYVal))

if LOG:
   print('dir_name plus parameter  json')
   print(os.path.join(dir_name, "parameter.json"))
   outfile=open(os.path.join(dir_name, "parameter.json"), 'w' )
   jsondic=vars(args)
   jsondic["dir_name"] = dir_name
   jsondic["num_parameters_Morpho"]=modelDil.count_params()
   jsondic["train_capacity"]=2
   json.dump(jsondic,outfile)
   outfile.close()