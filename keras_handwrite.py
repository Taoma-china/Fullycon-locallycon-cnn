import keras
from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
# deal with data
train_data = np.loadtxt('zip_train.txt',dtype=np.float32)
test_data = np.loadtxt('zip_test.txt',dtype=np.float32)
train_label = train_data[:,0]
test_label = test_data[:,0]
train_data=np.delete(train_data,0,axis=1)
test_data = np.delete(test_data,0,axis=1)


train_data=train_data.reshape(-1,1,16,16)
test_data=test_data.reshape(-1,1,16,16)




test_label = keras.utils.to_categorical(test_label,10)
train_label = keras.utils.to_categorical(train_label,10)

#add layer....
model =Sequential()
#conv 1
model.add(Convolution2D(
    nb_filter=32,
    nb_row=5,
    nb_col=5,
    border_mode='same',
    input_shape=(1,16,16),
    ))
model.add(Activation('relu'))

#pooling
model.add(MaxPooling2D(
    pool_size=(2,2),
    strides=(2,2),
    border_mode='same',#padding method
    ))

#conv 2 
model.add(Convolution2D(64,5,5,border_mode='same'))
model.add(Activation('relu'))

#pooling
model.add(MaxPooling2D(pool_size=(2,2),border_mode='same'))

#fully
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

#fully
model.add(Dense(10))
model.add(Activation('softmax'))

#optimizer
adam = Adam(lr=1e-4)

#compile
model.compile(optimizer=adam,
        loss='categorical_crossentropy',
        metrics=['accuracy'])

print ('Training ------------')

model.fit(train_data,train_label,nb_epoch=1,batch_size=16,)

print('\nTesting-------------')
loss,accuracy = model.evaluate(test_data,test_label)

print('\ntest loss: ',loss)
print('\ntest accuracy: ',accuracy)
