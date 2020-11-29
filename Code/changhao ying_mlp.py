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
print(x.shape)
#%%
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, test_size=0.2)
#%%
'''
DATA_DIR = "/home/ubuntu/mlproj/video/val/Fight/"
x_test_fight =[]
y_test_fight=[]
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".npy"]:
    a = np.load(DATA_DIR + path)
    a=a.reshape(len(a),-1)
    b=np.ones(len(a))
    y_test_fight.append(b)
    x_test_fight.append(a)
x_test_fight=np.vstack(x_test_fight)
y_test_fight=np.hstack(y_test_fight)
y_test_fight=np.asarray(y_test_fight,dtype='uint8')
#%%
DATA_DIR = "/home/ubuntu/mlproj/video/val/NonFight/"
x_test_nonfight =[]
y_test_nonfight=[]
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".npy"]:
    a = np.load(DATA_DIR + path)
    a=a.reshape(len(a),-1)
    b=np.zeros(len(a))
    y_test_nonfight.append(b)
    x_test_nonfight.append(a)
x_test_nonfight=np.vstack(x_test_nonfight)
y_test_nonfight=np.hstack(y_test_nonfight)
y_test_nonfight=np.asarray(y_test_nonfight,dtype='uint8')
#%%
y_train=np.hstack((y_train_fight,y_train_nonfight))
y_test=np.hstack((y_test_fight,y_test_nonfight))
y_train = to_categorical(y_train, num_classes=2)
y_test= to_categorical(y_test, num_classes=2)
#%%
print(y_train.shape,y_test.shape)
#%%
x_train=np.vstack((x_train_fight,x_train_nonfight))
x_test=np.vstack((x_test_fight,x_test_nonfight))
x_train,x_test=x_train/255,x_test/255
print(x_train.shape,x_test.shape)
'''
#%%
from keras.initializers import glorot_uniform
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
model.save('mlp_cying4.hdf5')
#%%
predict=np.argmax(model.predict(x_test),axis=1)
raw=np.argmax(y_test,axis=1)
#%%
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred = predict,y_true = raw)
print('confusion matrix')
print(cm)


