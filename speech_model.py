
#<<<<<<<<Art By Ankit>>>>>>>>>>#

import os
import librosa
import numpy as np
import glob
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten, BatchNormalization, Conv2D, MaxPooling2D, Activation
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

#load the path of data's directory
print("loading the path of data's directory")
dirs = []
for labels in os.listdir("data"):
    dirs.append("data/"+labels)

print(dirs)

classes = []
speeches = []
#extracting mfcc feature and preparing train data
#this will take some time
print("extracting mfcc feature and preparing train data")
print("this will take some time")
for i, file in enumerate(dirs):
    for filename in glob.glob(os.path.join(file, '*.wav')):
        y, sr = librosa.load(filename)
        ps = librosa.feature.melspectrogram(y=y, sr=sr)
        #passing only (128, 44) size's array in traning data
        #because CNN need same shape data
        if ps.shape==(128,44):
            ps = np.array(ps.reshape((128, 44, 1)))
            speeches.append(ps)
            classes.append(i)

#number of wave files for training
print("number of clasese for training", len(classes))
print("number of wave files for training", len(speeches))



# Modeling
features_shape = (128, 44, 1)
class_num = len(dirs)
inputs = Input(shape=features_shape)

# Block 1
o = Conv2D(24, (5, 5), strides=(1, 1), input_shape=features_shape)(inputs)
o = MaxPooling2D(pool_size=(4, 2), strides=(4, 2))(o)
o = BatchNormalization()(o)

# Block 2
o = Conv2D(48, (5, 5), padding="valid")(o)
o = MaxPooling2D((4, 2), strides=(4, 2))(o)
o = BatchNormalization()(o)

# Block 3
o = Conv2D(48, (5, 5), padding="valid")(o)
o = Activation("relu")(o)

# Flatten
o = Flatten()(o)

# Dense layer
o = Dense(64, activation="relu")(o)
o = BatchNormalization()(o)
o = Dropout(rate=0.5)(o)

# Predictions
outputs = Dense(class_num, activation="softmax")(o)

model = Model(inputs, outputs)
model.summary()
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])
plot_model(model, to_file='model.png')




#splitting data into train and test data
x_train, x_test, y_train, y_test = train_test_split(speeches, classes, test_size=0.2)

#categoricaling data crrosponding classes
y_train = np.array(to_categorical(y_train, class_num))
y_test = np.array(to_categorical(y_test, class_num))

print("length of y_train data", len(y_train))
print("length of y_test data", len(y_test))



EPOCHS = 5000
BATCH_SIZE = 64

#training model
print("training....")
history = model.fit(np.array(x_train), np.array(y_train), epochs=EPOCHS, batch_size=BATCH_SIZE)
#saving trained model
model.save("model.h5")
print("trained model saved")

#achieved 96.05% training accuracy 
score = model.evaluate(np.array(x_test), np.array(y_test), batch_size=BATCH_SIZE, verbose=1)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
#achieved 95.45% test accuracy 
