import os
import numpy as np
from keras.utils import to_categorical
#%%
DATA_DIR = "/home/ubuntu/mlproj/video/train/Fight/"
x_fight =[]
y_fight=[]
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".npy"]:
    a = np.load(DATA_DIR + path)
    a=a.reshape(len(a),-1)
    b=np.ones(len(a))
    y_fight.append(b)
    x_fight.append(a)
x_fight=np.vstack(x_fight)
y_fight=np.hstack(y_fight)
y_fight=np.asarray(y_fight,dtype='uint8')
#%%
DATA_DIR = "/home/ubuntu/mlproj/video/train/NonFight/"
x_nonfight =[]
y_nonfight=[]
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".npy"]:
    a = np.load(DATA_DIR + path)
    a=a.reshape(len(a),-1)
    b=np.zeros(len(a))
    y_nonfight.append(b)
    x_nonfight.append(a)
x_nonfight=np.vstack(x_nonfight)
y_nonfight=np.hstack(y_nonfight)
y_nonfight=np.asarray(y_nonfight,dtype='uint8')
#%%
y=np.hstack((y_fight,y_nonfight))
y = to_categorical(y, num_classes=2)
print(y.shape)
#%%
from sklearn.preprocessing import scale
x=np.vstack((x_fight,x_nonfight))
x=scale(x)
x=x.reshape(29800,12500,1)
print(x.shape)
#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
#%%
#x_train = ((x_train - np.mean(x_train, axis=1).reshape(-1,1)) / np.std(x_train, axis=1).reshape(-1,1))[:,:,np.newaxis]
#x_test = ((x_test - np.mean(x_test, axis=1).reshape(-1,1)) / np.std(x_test, axis=1).reshape(-1,1))[:,:,np.newaxis]
#print(x_train.shape,x_test.shape)
#%%
from keras.models import Sequential, Model
from keras.layers import Conv1D,Dense, Dropout, Flatten,AvgPool1D, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import cohen_kappa_score, f1_score
#%%
from keras.layers import MaxPool1D
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
model = Sequential()
model.add(Conv1D(filters=3, kernel_size=5,activation='relu',input_shape=(12500,1)))
#model.add(BatchNormalization())
#model.add(Conv1D(filters=16, kernel_size=3, activation='relu'))
#model.add(BatchNormalization())
#model.add(Conv1D(filters=32, kernel_size=3, activation='relu'))
#model.add(BatchNormalization())
#model.add(Dropout(0.2))
model.add(MaxPool1D(strides=3))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer=Adam(lr=0.001), loss ='BinaryCrossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=100, epochs=10,validation_data=(x_test, y_test))
#%%
from sklearn.metrics import cohen_kappa_score, f1_score
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))
#%%
predict=np.argmax(model.predict(x_test),axis=1)
#%%
raw=np.argmax(y_test,axis=1)
#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred = predict,y_true = raw)
print('confusion matrix')
print(cm)
#%%
model.save('c1d_cying4.hdf5')




