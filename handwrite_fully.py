import keras
from keras.models import Sequential
import numpy as np
from keras.layers import Dense, Activation, Dropout, MaxPooling2D, Flatten
from keras.optimizers import RMSprop
# deal with data
train_data = np.loadtxt('zip_train.txt',dtype=np.float32)
test_data = np.loadtxt('zip_test.txt',dtype=np.float32)
train_label = train_data[:,0]
test_label = test_data[:,0]
train_data=np.delete(train_data,0,axis=1)
test_data = np.delete(test_data,0,axis=1)


train_data=train_data.reshape(7291,256)
test_data=test_data.reshape(2007,256)

train_data /=255
test_data /=255




test_label = keras.utils.to_categorical(test_label,10)
train_label = keras.utils.to_categorical(train_label,10)


model = Sequential()

model.add(Dense(512,activation='relu', input_shape=(256,)))
model.add(Dropout(0.2))
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
model.summary()


model.compile(loss='categorical_crossentropy',
        optimizer=RMSprop(),
        metrics=['accuracy'])

history = model.fit(train_data, train_label,
        batch_size = 128,
        epochs=10,
        verbose=1,
        validation_data=(test_data, test_label))
score = model.evaluate(test_data, test_label, verbose=0)
print ('Test loss:', score[0])
print ('Test accuracy', score[1])
