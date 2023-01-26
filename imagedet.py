import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
#x_train-contains all the images of 90 percent
#x_test contains remaining 10 percent images
#y_train-contains results of x_train(output)
#y_test-contains output of x_test
len(X_train)
len(X_test)
X_train[0].shape
X_train[0]
plt.matshow(X_train[0])
y_train[0]
#normalization
X_train = X_train / 255
X_test = X_test / 255
X_train[0]
#flattening from 2d to 1d of training sets
X_train_flattened = X_train.reshape(len(X_train), 28*28)
X_test_flattened = X_test.reshape(len(X_test), 28*28)
X_train_flattened.shape
X_train_flattened[0]
#with no hidden layers
model = keras.Sequential([
    keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#fitting data with 50000 images and those results
model.fit(X_train_flattened, y_train, epochs=5)
model.evaluate(X_test_flattened, y_test)
#getting output for the x_test(10000) objects
y_predicted = model.predict(X_test_flattened)
y_predicted[0]
plt.matshow(X_test[0])
np.argmax(y_predicted[0])
y_predicted_labels = [np.argmax(i) for i in y_predicted]
y_predicted_labels[:5]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)
cm
import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
#using hidden layers by keeping 100 for more accuracy
model = keras.Sequential([
    keras.layers.Dense(100, input_shape=(784,), activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',#sparse-ytrain-int
              metrics=['accuracy'])

model.fit(X_train_flattened, y_train, epochs=5)
model.evaluate(X_test_flattened,y_test)
y_predicted = model.predict(X_test_flattened)
y_predicted_labels = [np.argmax(i) for i in y_predicted]
cm = tf.math.confusion_matrix(labels=y_test,predictions=y_predicted_labels)

plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
