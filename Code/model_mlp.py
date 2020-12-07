import glob
import os
import random
import numpy as np
from keras.utils import to_categorical
from keras.models import Sequential, Input, Model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Conv3D, MaxPooling3D, Dropout, BatchNormalization, Activation, LeakyReLU, Add, Multiply
from keras.regularizers import l2
from keras.layers.core import Lambda
from keras.optimizers import Adam,Nadam,SGD,RMSprop
#%%
source_path = os.getcwd() + "/video"
train_violence = source_path + "/train/Fight"
train_non = source_path + "/train/NonFight"
#%%
def sample(data_source, batch_size):
    files = sorted(glob.glob(data_source + '/*.npy'))
    arrays = []
    for i in range(batch_size):
        f = random.choice(files)
        a = np.load(f)
        arrays.append(a)
    data = np.concatenate(arrays)
    return (data)
#%%
def preprocess(violence, non, batch_size):
    x_violence_data = sample(violence, batch_size)
    x_non_data = sample(non, batch_size)
    y_violence_data, y_non_data = np.ones(len(x_violence_data)), np.zeros(len(x_non_data))
    y_violence = np.hstack(y_violence_data)
    y_non = np.hstack(y_non_data)
    return x_violence_data, x_non_data, y_violence, y_non
#%%
x_train_violence, x_train_non, y_train_violence, y_train_non = preprocess(train_violence, train_non, 64)
x_test_violence, x_test_non, y_test_violence, y_test_non = preprocess(train_violence, train_non, 16)
x_train, x_test = np.vstack((x_train_violence,x_train_non)), np.vstack((x_test_violence,x_test_non))
y_train, y_test =np.hstack((y_train_violence,y_train_non)), np.hstack((y_test_violence,y_test_non))
y_train, y_test = to_categorical(y_train, num_classes=2), to_categorical(y_test, num_classes=2)
#%%
def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std
x_train = normalize(x_train)
x_test = normalize(x_test)
print(x_train.shape)
print(x_test.shape)
# %%
x_train = x_train.reshape(9536, 50*50*5)
x_test = x_test.reshape(2384, 50*50*5)

#%%
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam,Nadam,SGD,RMSprop
from keras.callbacks import EarlyStopping
#%%
model = Sequential()
model.add(Dense(100, input_dim=12500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation = 'sigmoid'))
model.compile(loss='BinaryCrossentropy', optimizer=Adam(lr=0.001), metrics=['Accuracy'])
#from keras.callbacks import ModelCheckpoint
#es = EarlyStopping(monitor='val_loss',mode='min',patience=10,restore_best_weights=True)
#mc = ModelCheckpoint('mlp_cying4.h5', monitor='val_loss', mode='max',save_best_only=True)
model.fit(x_train, y_train, batch_size=512, epochs=200, validation_data=(x_test, y_test))
#%%
from sklearn.metrics import cohen_kappa_score, f1_score
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))
#%%
predict=np.argmax(model.predict(x_test),axis=1)
raw=np.argmax(y_test,axis=1)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred = predict,y_true = raw)
print('confusion matrix')
print(cm)



