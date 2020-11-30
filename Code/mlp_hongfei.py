import os
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torchvision
from torch.utils.data import TensorDataset
from torch.autograd import Variable
from torch.utils.data import DataLoader

from keras.models import Sequential, Input, Model
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Activation, LeakyReLU, Add, Multiply
from keras.regularizers import l2
from keras.layers.core import Lambda
from keras.layers.core import Lambda
import cv2
import os
import random
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dense,  BatchNormalization, Activation
from keras.models import Sequential,load_model


# The targets will be one-hot-encoded following the order:
## ["red blood cell", "difficult", "gametocyte", "trophozoite","ring", "schizont", "leukocyte"].
# For example, if an image contains red blood cells and trophozoite cells, the target will be [1, 0, 0, 1, 0, 0, 0].


DATA_DIR = os.getcwd()+"/npy/Train100"
DATA_DIR_Test = os.getcwd()+"/npy/Test30"
def stack(path,label_names):
    x,y=[],[]
    for f1 in label_names:
        path_loop = os.path.join(path, f1)
        for f2 in [f2 for f2 in os.listdir(path_loop) if f2[-4:] == ".npy"]:
            a = np.load(path_loop + "/" + f2)
            a = a.reshape(len(a), -1)
            x.append(a)
            temp1 = np.ones(len(a))
            temp2 = np.zeros(len(a))
            if f1=="Violence":
                y_temp = np.dstack((temp1, temp2)).reshape(-1, 2)
            elif f1=="NoViolence":
                y_temp = np.dstack((temp2, temp1)).reshape(-1, 2)
            y.append(y_temp)
    y = np.vstack(y)
    x = np.vstack(x)
    return x, y

label_names = ["Violence", "NoViolence"]
x_train, x_test, y_train, y_test=[],[],[],[]
x_train, y_train = stack(DATA_DIR, label_names)
x_test, y_test = stack(DATA_DIR_Test, label_names)


shuffle_train, shuffle_test = np.random.permutation(np.arange(len(x_train))),np.random.permutation(np.arange(len(x_test)))
x_train, y_train = x_train[shuffle_train], y_train[shuffle_train]
x_test, y_test = x_test[shuffle_test], y_test[shuffle_test]


LR = 1e-4
N_NEURONS = (500, 4100, 2050, 1000, 700)
N_EPOCHS = 250
BATCH_SIZE = 20
DROPOUT = 0.2


model = Sequential([
        Dense(N_NEURONS[0], input_dim=12500),
        Activation("relu"),
        BatchNormalization()
    ])

for n_neurons in N_NEURONS[1:-1]:
    model.add(Dense(n_neurons, activation="relu"))
    BatchNormalization()

model.add(Dense(N_NEURONS[-1], activation="relu"))
model.add(Dense(2, activation="softmax"))
model.compile(optimizer=Adam(lr=LR), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS,
      validation_data=(x_test, y_test),
          callbacks=[
                  ModelCheckpoint("mlp_hongfei_niu.hdf5", monitor="val_loss", save_best_only=True)])

print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test), axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))

# %% ------------------------------------------ Final test -------------------------------------------------------------
model = load_model('mlp_hongfei_niu.hdf5')
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test), axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# SEED = 42
# os.environ['PYTHONHASHSEED'] = str(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# def get_rgb(input_x):
#     rgb = input_x[...,:3]
#     return rgb
#
# # extract the optical flows
# def get_opt(input_x):
#     opt= input_x[...,3:5]
#     return opt
#
# inputs = Input(shape=(144,50,50,5))
#
# rgb = Lambda(get_rgb,output_shape=None)(inputs)
# opt = Lambda(get_opt,output_shape=None)(inputs)
#
# ##################################################### RGB channel
# rgb = Conv3D(
#     4, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
# rgb = Conv3D(
#     4, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(rgb)
# rgb = MaxPooling3D(pool_size=(1,2,2))(rgb)
#
# print(rgb.shape)
# ##################################################### Optical Flow channel
# opt = Conv3D(
#     4, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
# opt = Conv3D(
#     4, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(opt)
# opt = MaxPooling3D(pool_size=(1,2,2))(opt)
#
# print(opt.shape)
# ##################################################### Fusion and Pooling
# x = Multiply()([rgb,opt])
# x = MaxPooling3D(pool_size=(8,1,1))(x)
#
# print(x.shape)
# ##################################################### Merging Block
# x = Conv3D(
#     16, kernel_size=(1,3,3), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
# x = Conv3D(
#     16, kernel_size=(3,1,1), strides=(1,1,1), kernel_initializer='he_normal', activation='relu', padding='same')(x)
# x = MaxPooling3D(pool_size=(1,2,2))(x)
#
# print(x.shape)
# ##################################################### FC Layers
# x = Flatten()(x)
# x = Dense(128,activation='relu')(x)
# x = Dropout(0.2)(x)
# x = Dense(32, activation='relu')(x)
#
# # Build the model
# pred = Dense(2, activation='softmax')(x)
# model = Model(inputs=inputs, outputs=pred)
# model.summary()
#
# from keras.optimizers import Adam, SGD
#
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
# model.compile(optimizer=sgd, loss='BinaryCrossentropy', metrics=['accuracy'])
#
# model.fit(x_train, y_train, batch_size=8, epochs=16, validation_data=(x_test, y_test))







