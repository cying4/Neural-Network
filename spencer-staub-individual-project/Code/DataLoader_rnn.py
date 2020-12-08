# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")
# import the necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D, AveragePooling3D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle
import cv2
import os
import random
import glob
from keras.utils import to_categorical
from keras.layers import LSTM
from keras.layers.core import Lambda

source_path = os.getcwd() + "/RWF-2000_preprocessed_64"

train_violence = source_path + "/train/Fight"
train_non = source_path + "/train/NonFight"
valid_violence = source_path + "/val/Fight"
valid_non = source_path + "/val/NonFight"

def sample(data_source, batch_size):
    files = sorted(glob.glob(data_source + '/*.npy'))
    arrays = []
    for i in range(batch_size):
        f = random.choice(files)
        a = np.load(f)
        arrays.append(a)
    return (arrays)

# create train database
def preprocess(violence, non, batch_size):

    x_violence_data = sample(violence, batch_size)
    x_non_data = sample(non, batch_size)

    y_violence_data, y_non_data = np.ones(len(x_violence_data)), np.zeros(len(x_non_data))

    #x_violence = np.vstack(x_violence_data)
    #x_non = np.vstack(x_non_data)

    y_violence = np.hstack(y_violence_data)
    y_non = np.hstack(y_non_data)

    y_violence = np.array(y_violence)
    y_non = np.array(y_non)

    x_violence_data = np.array(x_violence_data)
    x_non_data = np.array(x_non_data)

    return x_violence_data, x_non_data, y_violence, y_non



x_train_violence, x_train_non, y_train_violence, y_train_non = preprocess(train_violence, train_non, 128)

x_test_violence, x_test_non, y_test_violence, y_test_non = preprocess(valid_violence, valid_non, 32)

y_train, y_test = np.hstack((y_train_violence,y_train_non)), np.hstack((y_test_violence,y_test_non))
y_train, y_test = to_categorical(y_train, num_classes=2), to_categorical(y_test, num_classes=2)

print(y_train.shape, y_test.shape)

x_train, x_test = np.vstack((x_train_violence, x_train_non)), np.vstack((x_test_violence, x_test_non))


def get_rgb(input_x):
    rgb = input_x[...,:3]
    return rgb


def get_opt(input_x):
    opt= input_x[...,3:5]
    return opt


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std


x_train = get_rgb(x_train)
x_test = get_rgb(x_test)

#x_train = normalize(x_train)
#x_test = normalize(x_test)

print(x_train.shape,x_test.shape)

x_train = np.resize(x_train, (256,149,64*64,3))
x_test = np.resize(x_test, (64,149,64*64,3))

print(x_train.shape,x_test.shape)

n_epochs = 16

baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(149, 64*64, 3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(2, 2))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="sigmoid")(headModel)
model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
	layer.trainable = False

print("[INFO] compiling model...")
opt = SGD(lr=1e-4, momentum=0.9, decay=1e-4 / n_epochs)
model.compile(loss="BinaryCrossentropy", optimizer=opt,
	metrics=["accuracy"])

#print((trainAug.flow(x_train, y_train, batch_size=4)).shape)
#print((valAug.flow(x_train, y_train, batch_size=4)).shape)

from keras.callbacks import ModelCheckpoint, EarlyStopping
#es = EarlyStopping(monitor='val_loss',mode='min',patience=10,restore_best_weights=True)
mc = ModelCheckpoint('mlp_sstaub.h5', monitor='val_loss', mode='max',save_best_only=True)

print("[INFO] training head...")
H = model.fit(
	x_train, y_train,
	steps_per_epoch=len(x_train) // 8,
	validation_data=(x_test, y_test),
	validation_steps=len(x_test) // 8,
	epochs=n_epochs,
    callbacks=[mc])

from sklearn.metrics import cohen_kappa_score, f1_score
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))
#%%
model.save('netork_cnn_resnet50.hdf5')

predict = np.argmax(model.predict(x_test), axis=1)
# %%
raw = np.argmax(y_test, axis=1)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_pred=predict, y_true=raw)
print('confusion matrix')
print(cm)

N = 16
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.show()
plt.savefig(os.getcwd() + "conv2d_plot.png")
print("hello")