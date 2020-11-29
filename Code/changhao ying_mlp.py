import os
import numpy as np
from keras.utils import to_categorical
#%%
DATA_DIR = "/home/ubuntu/mlproj/target/train/Fight/"
x_train_fight =[]
y_train_fight=[]
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".npy"]:
    a = np.load(DATA_DIR + path)
    a=a.reshape(len(a),-1)
    b=np.ones(len(a))
    y_train_fight.append(b)
    x_train_fight.append(a)
x_train_fight=np.vstack(x_train_fight)
y_train_fight=np.hstack(y_train_fight)
y_train_fight=np.asarray(y_train_fight,dtype='uint8')
#%%
DATA_DIR = "/home/ubuntu/mlproj/target/train/NonFight/"
x_train_nonfight =[]
y_train_nonfight=[]
for path in [f for f in os.listdir(DATA_DIR) if f[-4:] == ".npy"]:
    a = np.load(DATA_DIR + path)
    a=a.reshape(len(a),-1)
    b=np.zeros(len(a))
    y_train_nonfight.append(b)
    x_train_nonfight.append(a)
x_train_nonfight=np.vstack(x_train_nonfight)
y_train_nonfight=np.hstack(y_train_nonfight)
y_train_nonfight=np.asarray(y_train_nonfight,dtype='uint8')
#%%
DATA_DIR = "/home/ubuntu/mlproj/target/val/Fight/"
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
DATA_DIR = "/home/ubuntu/mlproj/target/val/NonFight/"
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
#%%
from keras.initializers import glorot_uniform
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.optimizers import Adam,Nadam,SGD,RMSprop
from keras.callbacks import EarlyStopping
#%%
model = Sequential()
model.add(Dense(200, input_dim=18000, activation='relu'))
model.add(Dropout(0.1))
#model.add(BatchNormalization())
model.add(Dense(300, activation='relu'))
model.add(Dropout(0.1))
#model.add(BatchNormalization())
model.add(Dense(400, activation='relu'))
model.add(Dropout(0.1))
#model.add(BatchNormalization())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(2, activation = 'sigmoid'))
model.compile(loss='BinaryCrossentropy', optimizer=SGD(lr=0.005), metrics=['Accuracy'])
from keras.callbacks import ModelCheckpoint
es = EarlyStopping(monitor='val_loss',mode='min',patience=10,restore_best_weights=True)
mc = ModelCheckpoint('mlp_cying4.h5', monitor='val_loss', mode='max',save_best_only=True)
model.fit(x_train, y_train, batch_size=50, epochs=2000, validation_data=(x_test, y_test), callbacks=[es, mc])
#%%
from sklearn.metrics import cohen_kappa_score, f1_score
print("Final accuracy on validations set:", 100*model.evaluate(x_test, y_test)[1], "%")
print("Cohen Kappa", cohen_kappa_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1)))
print("F1 score", f1_score(np.argmax(model.predict(x_test),axis=1),np.argmax(y_test,axis=1), average = 'macro'))
#%%
model.save('mlp_cying4.hdf5')