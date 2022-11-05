import tensorflow as tf 
from tensorflow import keras 
import matplotlib.pyplot as plt 
import numpy as np
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train=x_train/255
x_test=x_test/255
#x_train_flattened=x_train.reshape(len(x_train),28*28)
#x_test_flattened=x_test.reshape(len(x_test),28*28)
model=keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),
                        keras.layers.Dense(100,activation='relu'),
                        keras.layers.Dense(10,activation='sigmoid')])
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
#model.fit(x_train_flattened,y_train,epochs=10)
#print(model.evaluate(x_test_flattened,y_test))
model.fit(x_train,y_train,epochs=10)
model.evaluate(x_test,y_test)