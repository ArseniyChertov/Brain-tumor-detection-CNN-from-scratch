from keras.datasets import mnist
from tensorflow.python import metrics


from keras.utils import to_categorical
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
from keras import Sequential
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
import cv2
import os


def load_data():
    images = []
    labels = []
    data_path = 'brain_data'
    for image in os.listdir(data_path):
        image_path = os.path.join(data_path, image)
        image = cv2.imread(image_path)
        # print(image_path)

        images.append(image)
        # plt.imshow(image, cmap='gray')
        # plt.show()

        if 'Y' in image_path:
            labels.append(1)
        else:
            labels.append(0)

    im = np.array(images)
    labels = np.array(labels)

    return im, labels


def prep_data(img, labels):
    train_prep = []
    for x in range(len(img)):
        resized_image = cv2.resize(img[x], (512, 512))
        train_prep.append(resized_image)
    train_prep = asarray(train_prep)
    train_prep = train_prep.reshape(len(train_prep), 512, 512, 3)
    print(train_prep.shape[0])
    indices = np.arange(train_prep.shape[0])
    np.random.shuffle(indices)  # shuffle indices to make sure data does not maintain order

    train_prep = train_prep[indices]
    labels = labels[indices]

    val_x = train_prep[0:20]
    test_x = train_prep[20:40]
    train_x = train_prep[40:]

    val_y = labels[0:20]
    test_y = labels[20:40]
    train_y = labels[40:]

    print(val_y[0])

    # val_y = to_categorical(val_y)
    # test_y = to_categorical(test_y)
    # train_y = to_categorical(train_y)

    # print(val_y)

    # print(trainx.shape)
    # print(testx.shape)
    # print(valx.shape)

    return train_x, train_y, test_x, test_y, val_x, val_y


def normalize_data(train, test, val):
    train_normalize = (train.astype('float32'))/255
    test_normalize = (test.astype('float32'))/255
    val_normalize = (val.astype('float32'))/255

    return train_normalize, test_normalize, val_normalize


def define_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(512, 512, 3)))
    model.add(MaxPooling2D((2, 2)))
    #model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    #model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    #model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(50, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    #model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['binary_accuracy'])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['binary_accuracy'])
    return model


images, labels = load_data()

trainx, trainy, testx, testy, valx, valy = prep_data(images, labels)


train_norm, test_norm, val_norm = normalize_data(trainx, testx, valx)

#testy = np.char.split(testy)
'''
print(testy)
split_list = []
for x in testy:
    split_list.append(x)
print(split_list)
'''

model = define_model()
history = model.fit(train_norm, trainy, batch_size=16, epochs=20, verbose=1, validation_data=(val_norm, valy), shuffle=True)

train_loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(train_loss)
plt.plot(val_loss)

plt.title('loss curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


evaluate = model.evaluate(testx, testy, verbose=0)
#print('binary accuracy: ' + str(evaluate[1]))

for x in range(len(test_norm)):
    print('ground truth label: ' + str(testy[x]))
    prediction = model.predict(test_norm[x].reshape(1, 512, 512, 3))
    #prediction_label = prediction.argmax(axis=1)[0]
    print('pred label: ' + str(prediction))

    plt.imshow(test_norm[x], cmap='gray')
    plt.show()


